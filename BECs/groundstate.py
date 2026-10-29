import numpy as np
import scipy.sparse as sps
import xarray as xr
from bloch_schrodinger.fdsolver import FDSolver, check_name
from bloch_schrodinger.potential import Potential
from bloch_schrodinger.progress import bar, parallel_map
from BECs.backend import (
    array_module,
    device_warnings,
    fft_backend,
    fftn,
    get_xp,
    guarded_run,
    ifftn,
    resolve_device,
    to_host,
)
from BECs.io import RunCheckpoint
from scipy.ndimage import gaussian_filter
from scipy.sparse.linalg import eigsh


from BECs.spectral import (
    KineticStep,
    SpectralSolver,
    categories,
    density,
    register_coord,
)

# --- RKF45 coefficients for the adaptative time step (Fehlberg) ---
a2 = 1 / 4
a3 = 3 / 8
a4 = 12 / 13
a5 = 1
a6 = 1 / 2

b21 = 1 / 4

b31 = 3 / 32
b32 = 9 / 32

b41 = 1932 / 2197
b42 = -7200 / 2197
b43 = 7296 / 2197

b51 = 439 / 216
b52 = -8
b53 = 3680 / 513
b54 = -845 / 4104

b61 = -8 / 27
b62 = 2
b63 = -3544 / 2565
b64 = 1859 / 4104
b65 = -11 / 40

# 4th order
c1 = 25 / 216
c2 = 0
c3 = 1408 / 2565
c4 = 2197 / 4104
c5 = -1 / 5
c6 = 0

# 5th order (for error estimation)
d1 = 16 / 135
d2 = 0
d3 = 6656 / 12825
d4 = 28561 / 56430
d5 = -9 / 50
d6 = 2 / 55


def distance(psi1: np.ndarray, psi2: np.ndarray) -> float:
    """Compture the Fubiny=i-study distance between two states.

    Args:
        psi1 (np.ndarray): First state
        psi2 (np.ndarray): Second state

    Returns:
        float: The absolute value fubini-study metric, it should always be positive but numerical errors happens.
    """
    # Normalizing each state up front builds two full copies of the grid, and dividing the overlap by
    # the two norms afterwards would avoid them -- but not to the last bit, and this value drives the
    # adaptive controller, where a one-ulp change can flip a step from accepted to rejected and take
    # the whole run down a different path. Left as it is on purpose.
    xp = array_module(psi1)
    psi1_norm = psi1 / max(float(xp.linalg.norm(psi1)), 1e-15)
    psi2_norm = psi2 / max(float(xp.linalg.norm(psi2)), 1e-15)
    return float(
        abs(1 - abs(complex(xp.sum(xp.conjugate(psi1_norm) * psi2_norm))) ** 2)
    )


def fs_residual(psi1: np.ndarray, psi2: np.ndarray) -> float:
    """The Fubini-Study distance again, computed so that it does not saturate.

    'distance' above forms 1 - |<a,b>|^2. When the two states agree to within an ulp, which is
    exactly the regime an adaptive controller lives in, that subtraction cancels away most of the
    mantissa before returning: it bottoms out at 2*eps and gives back a flat 0.0 once the true
    relative error drops below about 1e-8. A fourth-order stepper reaches that floor at an ordinary
    step size, and the controller is then steering on no information at all.

    This is the same geometric quantity -- the sine of the Fubini-Study angle, equal to
    sqrt(1 - |<a,b>|^2) in exact arithmetic -- written as the norm of the part of b lying orthogonal
    to a. That is a vector norm rather than a difference of two nearly equal scalars, so nothing
    cancels and it keeps resolving down to machine precision, six decades further than 'distance'.

    It stays invariant under a global phase for the same reason 'distance' does: sending b to
    exp(i.theta) b multiplies both b and a<a,b> by that factor, leaving the residual's norm alone.

    Note that it is the square root of 'distance', so a tolerance expressed against this function is
    the square root of the equivalent tolerance against that one. It is also linear rather than
    quadratic in the error, which is what makes the usual 1/(p+1) controller exponent correct.

    Args:
        psi1 (np.ndarray): First state.
        psi2 (np.ndarray): Second state.

    Returns:
        float: The sine of the angle between the two rays, in [0, 1].
    """
    xp = array_module(psi1)
    a = psi1 / max(float(xp.linalg.norm(psi1)), 1e-300)
    b = psi2 / max(float(xp.linalg.norm(psi2)), 1e-300)
    return float(xp.linalg.norm(b - a * xp.vdot(a, b)))


def normalize(psi: np.ndarray, norm: float) -> np.ndarray:
    """Renormalize a vector to a specified value, intended to be NaN resistant.

    Args:
        psi (np.ndarray): The vector to renormalize
        norm (float): Its desired norm

    Returns:
        np.ndarray: The normalized array
    """
    n = float(array_module(psi).linalg.norm(psi))
    if not np.isfinite(n) or n < 1e-15:
        n = 1e-15
    return psi * norm / n


def f(H0: sps.dia_array, g: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """The function describing "y' = f(y)" for the RK4 method. It is the Gross-Pitaevskii equation with imaginary time.

    Args:
        H0 (sps.dia_array): The linear Hamiltonian
        g (np.ndarray): The interaction vector
        psi (np.ndarray): the vector to propagate

    Returns:
        np.ndarray: y'
    """
    return -(H0 @ psi + (g * np.abs(psi) ** 2) * psi)


def oneStep(
    H0: sps.dia_array,
    g: np.ndarray,
    dt: float,
    psi: np.ndarray,
    norm: float,
    tol: float,
) -> tuple[float, float, np.ndarray]:
    """A recursive RKF45 adaptative method to propagate the vector one step. Calls itself with a smaller time-step if the tolerance set is not matched.

    Args:
        H0 (sps.dia_array): The linear Hamiltonian
        g (np.ndarray): The interaction vector
        dt (float): The time-step
        norm (float): The norm of the vector, important to specify it as not to propagate numerical errors.
        psi (np.ndarray): the vector to propagate.
        tol (float): The tolerance (fubini-study distance) specifying wheter to restarts the step

    Returns:
        np.ndarray: The propagated vector
    """

    # Computing the 6 terms of the RKF45 method, making sure each propagated vector is normalized.
    k1 = f(H0, g, psi)
    k2 = f(H0, g, normalize(psi + dt * (b21 * k1), norm))
    k3 = f(H0, g, normalize(psi + dt * (b31 * k1 + b32 * k2), norm))
    k4 = f(H0, g, normalize(psi + dt * (b41 * k1 + b42 * k2 + b43 * k3), norm))
    k5 = f(
        H0, g, normalize(psi + dt * (b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4), norm)
    )
    k6 = f(
        H0,
        g,
        normalize(
            psi + dt * (b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5), norm
        ),
    )

    # 4th and 5th order estimates
    y4 = psi + dt * (c1 * k1 + c2 * k2 + c3 * k3 + c4 * k4 + c5 * k5 + c6 * k6)
    y4 = normalize(y4, norm)
    y5 = psi + dt * (d1 * k1 + d2 * k2 + d3 * k3 + d4 * k4 + d5 * k5 + d6 * k6)
    y5 = normalize(y5, norm)

    # Error estimate
    err = distance(y4, y5)
    err = err if not np.isnan(err) else 0
    dt = dt if not np.isnan(dt) else 1e-5

    if err > tol:
        return oneStep(H0, g, dt / 2, psi, norm, tol)
    else:
        y_next = y5  # usually take 5th-order solution

        # Step size control
        if err == 0:
            s = 2  # increase aggressively if perfect
        else:
            s = 0.7 * (tol / err) ** 0.25

        dt = min(
            max(s * dt, 1e-8), 1
        )  # Constrain the time step so it does not ballon too much.
        E = np.real(
            np.dot(np.conjugate(y_next), -f(H0, g, y_next))
        )  # Compute the total energy of the ensemble of atoms.

        return dt, E, y_next


def findGroundState(
    H0: sps.dia_matrix,
    gs: list,
    psi0: np.ndarray,
    dt: float,
    npoints: int,
    tol: float = 1e-8,
    maxiter=100,
) -> tuple[float, xr.DataArray]:
    """Finds the ground state of the Hamiltonian by imaginary time propagation

    Args:
        H0 (sps.dia_matrix): The linear Hamiltonian
        gs (list): The value of the interactions for each field.
        psi0 (np.ndarray): An initial guess for the ground state.
        dt (float): The time-step length.
        npoints (int): The number of grid points, needed to generate the g-vector
        tol (float, optional): Tolerance for convergence. Defaults to 1e-8.
        maxiter (int, optional): Maximum number of iterations. Defaults to 100.

    Returns:
        tuple[float,xr.DataArray]: The energy and mode profile of the ground state
    """

    g = np.array([[g] * npoints for g in gs]).reshape(
        -1
    )  # Construct the interaction vector.

    norm = np.linalg.norm(
        psi0
    )  # Get the norm of psi0, to be conserved during propagation

    dt, E, psi1 = oneStep(H0, g, dt, psi0, norm, tol)  # Take a first step to initialize

    # Iterate until converged or maximum number of iteration attainded
    count = 1
    while distance(psi1, psi0) > tol and count < maxiter:
        psi0 = psi1 * 1
        dt, E, psi1 = oneStep(H0, g, dt, psi0, norm, tol)
        count += 1
    if count >= maxiter - 1:
        print(
            "maximum number of iterations reached, the returned state might not be converged"
        )

    return E, psi1


def subselect(indexes: tuple[int], ptype: str, allcoords: dict) -> dict[float]:
    """Create a subselction from allcoords based on a keyword 'ptype'

    A dimension may answer to several keywords at once -- the evolution time of a time-of-flight
    sequence parameterizes the potential and also indexes the sampling times -- so the test is a
    membership rather than an equality, and such a dimension turns up in each of its categories'
    selections.

    Args:
        indexes (tuple[int]): The indexes of the values to select in each coordinate array.
        ptype (str): The keyword to select the subselection, can be 'potential', 'alpha', 'g', 'reciprocal', 'population', 'coupling', 'psi0' or 't'.
        allcoords (dict): The dictionnary containing the coordinates and their type

    Returns:
        dict: The subselection
    """
    subselect = {
        dim: allcoords[dim][1][i]
        for i, dim in zip(indexes, allcoords.keys())
        if ptype in categories(allcoords[dim])
    }
    return subselect


def selected_value(value, sel: dict):
    """One parameter point's value, as a plain Python number when it is a scalar.

    Selecting a single point of a swept parameter gives a 0-d DataArray, whose '.data' is a 0-d
    *numpy* array rather than a number. On the CPU that is invisible -- it multiplies a grid just
    like a float would. On a device it is not: cupy refuses a host array as an operand rather than
    transferring it, so 'g * density(psi)' raises

        TypeError: Unsupported type <class 'numpy.ndarray'>

    from inside the kernel machinery, with nothing in the message pointing at the interaction
    strength. Unwrapping to a Python number keeps the value a scalar for both backends.

    An entry that is genuinely an array -- a spatially varying interaction, say -- is handed back
    untouched, for the caller to move to wherever the field lives.

    Args:
        value: The parameter, an xr.DataArray when it is swept and a plain value when it is not.
        sel (dict): The selection for this run, from 'subselect'.

    Returns:
        The value at that point: a Python number for a scalar, otherwise the array itself.
    """
    v = value.sel(sel).data if isinstance(value, xr.DataArray) else value
    if isinstance(v, np.ndarray) and v.ndim == 0:
        return v.item()
    return v


class GroundState(FDSolver):
    """The ground state class uses an imaginary time propagation method to find the ground state of the Gross-Pitaevskii equation.
    It uses a RKF45 propagation method and is built on the Solver class from the 'bloch_schrodinger' package."""

    def __init__(
        self,
        potentials: Potential | list[Potential],
        gs: float | xr.DataArray | list[float | xr.DataArray],
    ):
        """Instantiate the solver.

        Args:
            potentials (Union[Potential,list[Potential]]): The potential felt by each field. They must all be defined on the same grid.
            A single potential can also be passed for a scalar equation.
            gs (Union[Union[float, xr.DataArray], list[Union[float, xr.DataArray]]]): The interaction energy for each field.
            A single coefficient can be passed for a scalar equation.
            The kinetic energy coefficient hbar²/2m is fixed to 1/2 for every field.

        Raises:
            ValueError: Not the same number of potentials and interaction terms given.
        """

        # one kinetic coefficient per field, as FDSolver expects a list as soon as it gets one
        n_fields = 1 if isinstance(potentials, Potential) else len(potentials)
        super().__init__(potentials, 1 / 2 if n_fields == 1 else [1 / 2] * n_fields)

        if self.nb == 1 and isinstance(gs, (int, float, xr.DataArray)):
            self.gs = [gs]
        else:
            self.gs = gs

        # Adding the interaction terms to the list of coordinates
        for g in self.gs:
            if isinstance(g, xr.DataArray):
                for dim in g.dims:
                    check_name(dim, self.n_dims)
                    coords_g = {dim: ["g", g.coords[dim]] for dim in g.dims}
                    self.allcoords.update(coords_g)

    def make_g_list(self, sel: dict) -> list[float]:
        """Select the proper g value for each field given a dimension selection

        Args:
            sel (dict): The selection of coordinate index for gs.

        Raises:
            TypeError: Raises an error if one of the g given to the constructor is not of type int, float or DataArray.

        Returns:
            list[float]: A list of single valued alphas.
        """
        gs = []
        for u in range(len(self.gs)):
            if isinstance(self.gs[u], float) or isinstance(self.gs[u], int):
                gs += [self.gs[u]]
            elif isinstance(self.gs[u], xr.DataArray):
                sub_sel = {
                    dim: value for dim, value in sel.items() if dim in self.gs[u].dims
                }
                gs += [float(self.gs[u].sel(sub_sel).data)]
            else:
                raise TypeError(
                    f"{u}-th g term not of a recognized type (int, float or xr.DataArray)"
                )
        return gs

    def solve(
        self,
        population: float | xr.DataArray = 1,
        tol: float = 1e-8,
        maxiter=1000,
        phase0: tuple[float] | None = None,
        parallel: bool = False,
        skip_guess: bool = False,
        n_cores: int = 4,
        verbose: bool = True,
    ) -> tuple[xr.DataArray]:
        """Find the ground states of the Hamiltonians, using an imaginary time propagation initialized with the ground state of the linear Hamiltonian H0.

        Args:
            population (Union[float,xr.DataArray], optional): The number of atoms in the condensate. Defaults to 1.
            tol (float, optional): The tolerance for the adaptative time-step evolution. Defaults to 1e-8.
            maxiter (int, optional): Maximal number of iterations for convergence. Defaults to 1000.
            phase0 (tuple[float], optional): Where to fix the phase of the ground state to 0, in the (a1, ..., an, field) basis. Defaults to (0.01, ..., 0.01, 0).
            parallel (bool, optional): Whether to use the parallel solver, this involve some overhead, so do not use it for too small parameter spaces. default to False.
            skip_guess(bool, optional): Wheter to skip the initial linear Hamitlonian diagonalization. Can speed up the overall solver if the matrices are big enough.
            n_cores (int, optional): The number of cores to use for the parallelized solver.
            verbose (bool, optional): Whether to plot progress bars for the two phases of the solve.
            Defaults to True.

        Returns:
            tuple[xr.DataArray]: The energy and mode profiel of the ground state for all parameters.
        """

        if isinstance(population, xr.DataArray):
            for dim in population.dims:
                check_name(dim, self.n_dims)
            coords_pops = {
                dim: ["population", population.coords[dim]] for dim in population.dims
            }
            self.allcoords.update(coords_pops)

        # Create empty DataArrays to store the eigenvalues and vectors
        eigva = self.initialize_eigva(1).squeeze(dim="band")
        eigve = self.initialize_eigve(1).squeeze(dim="band")

        # We are going to create all the linear Hamiltonian and find they lowest eigenvector first, before finding the ground states
        ## First we create a list of tuples that select a single value for each of the parameter dimensions

        indexes = [np.arange(len(coord[1])) for coord in self.allcoords.values()]
        indexGrid = np.meshgrid(*indexes, indexing="ij")
        indexGrid = [grid.reshape(-1) for grid in indexGrid]
        selections = [tup for tup in zip(*indexGrid)]

        if len(selections) == 0:
            selections = [()]

        H0_list = []
        g_list = []
        psi0_list = []
        dt_list = []

        # Initializing the vector guess. The solver works better with a good guess for the lowest eigenvector
        X = np.random.rand(self.n, 1)

        # Looping over every point of the parameter space, one Hamiltonian per point
        for indexes in bar(
            selections, desc="Computing initial guesses", unit="guess", verbose=verbose
        ):
            # --- Constructing the Hamiltonian from the parameter selection ---
            potential_sel = subselect(indexes, "potential", self.allcoords)
            alpha_sel = subselect(indexes, "alpha", self.allcoords)
            reciprocal_sel = subselect(indexes, "reciprocal", self.allcoords)
            coupling_sel = subselect(indexes, "coupling", self.allcoords)

            # The Hamiltonian, couplings included, is built by the parent solver
            ham = self.create_hamiltonian(
                potential_sel, alpha_sel, reciprocal_sel, coupling_sel
            )

            # Select the interaction strength
            g_sel = subselect(indexes, "g", self.allcoords)
            gs = self.make_g_list(g_sel)

            # Select the population
            pop_sel = subselect(indexes, "population", self.allcoords)

            if not skip_guess:
                # Find the ground state of the linear Hamiltonian H0
                eigvals, eigvec = eigsh(ham, k=1, v0=X, which="SM")
                X = eigvec
            else:
                # A flat, unit-norm seed. FDSolver.normalize divides a raw array by the sum of
                # |psi|^2 rather than by its square root, so anything else than a unit-norm seed
                # comes out with the wrong population.
                eigvec = np.ones((self.n, 1)) / self.n**0.5
                eigvals = np.mean(np.abs(self.potential_data.sel(potential_sel).data))

            if isinstance(population, xr.DataArray):
                pop = float(population.sel(pop_sel).data)
            else:
                pop = population

            # Computing an approximately good time step
            dt = max(2 * np.pi / (eigvals + max(np.max(gs) * pop, 1000)), 1e-5)
            # Store the arguments for the ground state solver as lists
            H0_list += [ham]
            g_list += [[gs]]
            psi0_list += [self.normalize(eigvec, pop)[:, 0]]
            dt_list += [dt]

        def x(H0, gs, psi0, dt):
            return findGroundState(H0, gs, psi0, dt, self.np, tol, maxiter)

        if not parallel:
            # Now we can compute the ground state and store them in eigve
            for i in bar(
                range(len(psi0_list)),
                desc="Computing ground states",
                unit="state",
                verbose=verbose,
            ):
                indexes = selections[i]
                energ, eigvec = x(H0_list[i], g_list[i], psi0_list[i], dt_list[i])

                eigva[*indexes] = energ
                eigve[*indexes] = eigvec

        else:
            ev_list = parallel_map(
                x,
                list(zip(H0_list, g_list, psi0_list, dt_list)),
                n_jobs=n_cores,
                desc="Computing ground states",
                unit="state",
                verbose=verbose,
            )

            for i in range(len(psi0_list)):
                indexes = selections[i]
                eigva[*indexes] = ev_list[i][0]
                eigve[*indexes] = ev_list[i][1]

        eigve = eigve.unstack(dim="component").rename("ground state")

        pos0 = phase0 if phase0 is not None else (0.01,) * self.n_dims + (0,)
        sel0 = {self.spatial_dims[d]: pos0[d] for d in range(self.n_dims)}
        sel0["field"] = pos0[self.n_dims]

        eigve = eigve * xr.ufuncs.exp(
            -1j * xr.ufuncs.angle(eigve.sel(sel0, method="nearest"))
        )
        eigve = eigve.assign_coords(
            {
                self.potentials[0].coord_names[d]: self.potentials[0].coords[d]
                for d in range(self.n_dims)
            }
        )

        return eigva.squeeze() * self.potentials[0].get_dS(), eigve.squeeze()


# ====================================
# --- Spectral methods ---
# ====================================


def get_energy(
    psi: np.ndarray,
    k2: np.ndarray,
    V: np.ndarray | xr.DataArray,
    g: float,
) -> float:
    """Compute the energy of the state psi.

    Args:
        psi (np.ndarray): State considered
        k2 (np.ndarray): The squared norm of the wavevector at each point of reciprocal space.
        V (Union[np.ndarray,xr.DataArray]): Potential landscape.
        g (float): Non-linear coefficient.

    Returns:
        float: The chemical potential of the mode times its population, computed as <psi|H|psi>.
        Note that this is mu.N and *not* the Gross-Pitaevskii energy functional: H carries the
        full interaction g|psi|^2, so the value returned holds g.int|psi|^4 where the energy
        functional holds g/2.int|psi|^4. The two differ by exactly the interaction energy. Use
        'energy_components' when that distinction matters. The dS factor is not applied here,
        the caller applies it once on return.
    """

    # The orthogonal normalization makes the Fourier and the real space sums directly comparable
    psi_fft = fftn(psi, norm="ortho")
    Hpsi_fft = 1 / 2 * k2 * psi_fft
    Hpsi = (g * density(psi) + V) * psi

    xp = array_module(psi)
    E_fourier = float(xp.real(xp.sum(xp.conjugate(psi_fft) * Hpsi_fft)))
    E_real = float(xp.real(xp.sum(xp.conjugate(psi) * Hpsi)))
    return E_fourier + E_real


def energy_components(
    psi: np.ndarray,
    k2: np.ndarray,
    V: np.ndarray | xr.DataArray,
    g: float,
    dS: float,
) -> dict[str, float]:
    """Split the Gross-Pitaevskii functional into its kinetic, potential and interaction parts.

    'get_energy' returns the single number <psi|H|psi>, which is enough to drive the solver but
    hides the interaction term inside it. This function reports the pieces separately, so that
    the interaction energy can be compared against its theoretical scalings.

    Args:
        psi (np.ndarray): State considered, normalized so that dS.sum|psi|^2 is the population.
        k2 (np.ndarray): The squared norm of the wavevector at each point of reciprocal space.
        V (Union[np.ndarray,xr.DataArray]): Potential landscape.
        g (float): Non-linear coefficient.
        dS (float): The length/surface/volume element, from 'Potential.get_dS'. Unlike
        'get_energy', this function applies it itself, so every value it returns is physical.

    Returns:
        dict[str,float]: 'population' (dS.sum|psi|^2), 'kinetic', 'potential', 'interaction'
        (g/2.int|psi|^4) and 'total', the energy functional summing the three. 'mu' is the
        chemical potential *per particle*, which carries the full g and so exceeds
        'total'/'population' by the interaction energy per particle.
    """

    xp = array_module(psi)
    rho = density(psi)

    # Parseval, with the same orthogonal normalization as 'get_energy' so that the Fourier and
    # the real space sums stay directly comparable
    rho_fft = density(fftn(psi, norm="ortho"))

    # '.data' rather than 'np.asarray', as in 'PotentialPhase._bare': V reaches this function
    # already on the same device as psi, and np.asarray would refuse a cupy array
    V_bare = V.data if isinstance(V, xr.DataArray) else V

    population = float(xp.sum(rho)) * dS
    kinetic = float(xp.sum(k2 * rho_fft)) / 2 * dS
    potential = float(xp.real(xp.sum(V_bare * rho))) * dS
    interaction = g / 2 * float(xp.sum(rho * rho)) * dS

    total = kinetic + potential + interaction
    return {
        "population": population,
        "kinetic": kinetic,
        "potential": potential,
        "interaction": interaction,
        "total": total,
        "mu": (total + interaction) / population,
    }


def gp_residual(
    psi: np.ndarray,
    kin: KineticStep,
    V: np.ndarray | xr.DataArray,
    g: float,
) -> tuple[float, float]:
    """How far psi is from being a stationary state of the Gross-Pitaevskii equation.

    This is the relative residual ||P(H - mu)psi|| / ||P.H.psi||, with mu = <psi|H|psi>/<psi|psi>
    and P the anti-aliasing projector, which is the Hamiltonian the real-time solvers actually
    propagate with. It is zero exactly when psi is an eigenstate of that Hamiltonian, so unlike the
    distance between two successive imaginary-time iterates it does not depend on the time step:
    the split-step iteration converges to a fixed point displaced from the ground state by an
    amount growing with dt, where successive iterates agree to the last bit while psi is still
    measurably off. Handed to a real-time solver, that displacement is released as sound waves.

    Args:
        psi (np.ndarray): State considered.
        kin (KineticStep): The grid's kinetic propagator, for its reciprocal space and mask.
        V (Union[np.ndarray,xr.DataArray]): Potential landscape.
        g (float): Non-linear coefficient.

    Returns:
        tuple[float, float]: The relative residual, and ||P.H.psi||/||psi||, the energy scale it
        is relative to.
    """
    xp = array_module(psi)
    V_bare = V.data if isinstance(V, xr.DataArray) else V

    # The kinetic term is added in reciprocal space, where the projection is applied anyway
    Hpsi_fft = kin.k2 / 2 * fftn(psi, kin.workers)
    Hpsi_fft += fftn((V_bare + g * density(psi)) * psi, kin.workers)
    Hpsi = ifftn(kin.aliasing * Hpsi_fft, kin.workers)

    psi_norm = float(xp.linalg.norm(psi))
    H_norm = max(float(xp.linalg.norm(Hpsi)), 1e-300)
    mu = float(xp.real(xp.vdot(psi, Hpsi))) / psi_norm**2
    return float(xp.linalg.norm(Hpsi - mu * psi)) / H_norm, H_norm / psi_norm


def ilinear_step(
    psi: np.ndarray,
    dt: float,
    kin: KineticStep,
) -> np.ndarray:
    """Linear propagation of the vector psi for a step 1j*dt by multiplication in Fourier space.

    Args:
        psi (np.ndarray): The vector to propagate.
        dt (float): The time step.
        kin (KineticStep): The grid's kinetic propagator. Imaginary time is a complex time step, so
        the real-time propagator gives the diffusive kernel exp(-dt.k^2/2) when handed 1j*dt.

    Returns:
        np.ndarray: Propagated vector.
    """
    return kin(psi, 1j * dt)


def inl_step(
    psi: np.ndarray,
    dt: float,
    V: np.ndarray | xr.DataArray,
    g: float,
):
    """Non-linear propagation of the vector psi for a step 1j*dt by multiplication in real space.

    Args:
        psi (np.ndarray): The vector to propagate.
        dt (float): The time step.
        V (Union[np.ndarray,xr.DataArray]): Potential landscape.
        g (float): Non-linear coefficient.

    Returns:
        np.ndarray: Propagated vector.
    """
    return array_module(psi).exp(-dt * (g * density(psi) + V)) * psi


def istep(
    psi: np.ndarray,
    kin: KineticStep,
    V: np.ndarray | xr.DataArray,
    dt: float,
    g: float,
) -> np.ndarray:
    """Propagate psi for a full step 1j*dt

    Args:
        psi (np.ndarray): The vector to propagate.
        kin (KineticStep): The grid's kinetic propagator.
        V (Union[np.ndarray,xr.DataArray]): Potential landscape.
        dt (float): time step.
        g (float): Non-linear coefficient.

    Returns:
        np.ndarray: Propagated vector.
    """
    norm = np.linalg.norm(psi)
    psi_1 = ilinear_step(psi, dt / 2, kin)
    psi_1 = normalize(psi_1, norm)

    # The population has to be restored between the sub-steps and not only at the end: the
    # interaction term reads the density, so it sees a different state if the norm has drifted.
    #
    # The density is read half way through the sub-step rather than at its start. In real time the
    # non-linear sub-step leaves |psi| alone, so freezing the density is exact; in imaginary time
    # it reshapes |psi|, and the frozen density is only first-order accurate. That one sub-step
    # used to cost the whole splitting an order: its fixed point sat O(dt) away from the ground
    # state instead of O(dt^2) -- for g=10 in a harmonic trap at dt=1e-2, a residual of 2.5e-4
    # where the midpoint gives 7.7e-6 -- and with g=0 the two coincide.
    psi_mid = normalize(inl_step(psi_1, dt / 2, V, g), norm)
    psi_2 = array_module(psi_1).exp(-dt * (g * density(psi_mid) + V)) * psi_1
    psi_2 = normalize(psi_2, norm)

    psi_3 = ilinear_step(psi_2, dt / 2, kin)
    psi_3 = normalize(psi_3, norm)

    return psi_3


def iadaptative_step(
    psi: np.ndarray,
    kin: KineticStep,
    V: np.ndarray | xr.DataArray,
    dt: float,
    g: float,
    tol: float,
    psi_full: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Propagate psi for a full step 1j*dt, using a recursive adaptative step-doubling method.
    This function propagate psi for dt and for 2*dt/2, then compares the results. If its above a certain tolerance,
    the function calls itself again with a halved time step.

    Args:
        psi (np.ndarray): The vector to propagate.
        kin (KineticStep): The grid's kinetic propagator.
        V (Union[np.ndarray,xr.DataArray]): Potential landscape.
        dt (float): time step.
        g (float): Non-linear coefficient.
        tol (float): The tolerance for step doubling
        psi_full (np.ndarray, optional): The single step of dt, when the caller has already computed
        it. A rejected step of 2.dt hands down the half step it took, which is exactly this call's
        full step. Defaults to None.

    Returns:
        tuple[float, np.ndarray]: The optimal next time step length and the propagated vector.
    """
    if psi_full is None:
        psi_full = istep(psi, kin, V, dt, g)

    psi_half = istep(psi, kin, V, dt / 2, g)
    psi_double = istep(psi_half, kin, V, dt / 2, g)

    # Computing the error, using a standard 2-norm.
    xp = array_module(psi_full)
    err = float(xp.sum(density(psi_full - psi_double))) / float(xp.sum(density(psi_full)))

    if err > tol:  # If the error is superior, try again with a time step dt/2
        return iadaptative_step(psi, kin, V, dt / 2, g, tol, psi_full=psi_half)
    else:  # else return the results and compute a new time-step
        if err == 0:
            s = 2
        else:
            s = max(min(0.6 * (tol / err) ** 0.25, 2), 0.1)
        return s * dt, psi_double


def findGroundStateSSFM(
    kin: KineticStep,
    psi0: np.ndarray,
    V: xr.DataArray,
    g: float,
    tol_adapt: float,
    tol_stop: float,
    maxiter: int,
    dt_max: float | None = None,
) -> tuple[float, np.ndarray]:
    """The main simulation function of the GroundStateSSFM class. Solves the Gross-Pitaevskii equation with an imaginary time step using an adaptative split-step Fourier method.

    The split-step iteration does not converge to the ground state itself but to a fixed point
    displaced from it by O(dt^2), so stopping once the iterates stop moving leaves that
    displacement in. The loop therefore stops on the true residual of the equation, 'gp_residual',
    and whenever the iterates have settled on a fixed point that is still too far, it lowers a
    ceiling on the time step and carries on from there, the fixed point following the step down.
    Each lowering has to relax again through the slowest modes of the trap, at the smaller step,
    so a tight 'tol_stop' is paid for in iterations.

    Args:
        kin (KineticStep): The grid's kinetic propagator, carrying reciprocal space and the
        anti-aliasing mask.
        psi0 (np.ndarray): Initial vector.
        V (xr.DataArray): Potential landscape.
        g (float): Interaction strength term.
        tol_adapt (float): Tolerance for the adaptative time step method.
        tol_stop (float): The relative residual of the Gross-Pitaevskii equation at which the
        state is accepted as the ground state, see 'gp_residual'.
        maxiter (int): maximal number of iteration for the solver.
        dt_max (float, optional): A ceiling on the imaginary time step from the start. The solver
        lowers it on its own when it has to, so this is only needed to keep the first iterations
        tame. Defaults to None, no ceiling.

    Returns:
        tuple[float, np.ndarray]: <psi|H|psi>, see 'get_energy', and the ground state.
    """
    psi = psi0
    dt = 2 * np.pi / 100 / get_energy(psi, kin.k2, V, g)
    dt_cap = np.inf if dt_max is None else dt_max

    residual, scale = gp_residual(psi, kin, V, g)
    best, stalled = residual, 0

    for count in range(1, maxiter + 1):
        dt_used = min(dt, dt_cap)
        dt, psi_next = iadaptative_step(psi, kin, V, dt_used, g, tol_adapt)

        # How fast the iterates still move, per unit of imaginary time and relative to the energy
        # scale: the residual of the effective problem whose fixed point the splitting converges
        # to. It comes for free, where 'gp_residual' costs three transforms, so the latter is only
        # evaluated once the iterates have settled or every hundred steps.
        drift = fs_residual(psi_next, psi) / (dt_used * scale)
        psi = psi_next

        settled = drift < tol_stop / 4
        if count < 50 or not (settled or count % 100 == 0):
            continue

        residual, scale = gp_residual(psi, kin, V, g)
        if residual < tol_stop:
            break

        if settled:
            # What is left is the displacement of the fixed point, which shrinks as the square of
            # the step: aim the ceiling at the tolerance, by no more than 20x at once.
            dt_cap = dt_used * min(max(0.7 * (tol_stop / residual) ** 0.5, 0.05), 0.5)
            best, stalled = residual, 0
        elif residual < 0.9 * best:
            best, stalled = residual, 0
        else:
            # The iterates neither settle nor improve: a step the controller grew past what the
            # iteration can contract with, which then cycles rather than converging. The patience
            # is long on purpose, since a residual also rises for a while on an ordinary transient
            # -- an attractive gas collapsing onto its peak, say -- and halving the step there
            # every time only freezes it in place.
            stalled += 1
            if stalled >= 30:
                dt_cap = dt_used / 2
                best, stalled = residual, 0
    else:
        print(
            "maximal number of iteration reached, the result might not be converged "
            f"(residual {residual:.1e} for tol_stop = {tol_stop:.0e})"
        )

    return get_energy(psi, kin.k2, V, g), psi


# ====================================
# --- Preconditioned gradient descent ---
# ====================================
#
# The ground state minimizes the Gross-Pitaevskii energy functional at fixed population, which
# makes it a minimization on a sphere in L^2 rather than an initial value problem. Imaginary time
# is one way to descend that sphere, but it is projected *gradient* descent with a step size, and
# it pays for both halves of that: it converges at the rate gradient descent manages on a stiff
# Hessian, and the operator splitting puts its fixed point O(dt^2) away from the minimizer.
#
# The second half is the expensive one in practice. Measured on a 256^2 harmonic trap, a fixed step
# converges to a residual of 1.85e-6 at dt = 2.08e-3 and 6.85e-6 at dt = 4.0e-3 -- a ratio of 3.70
# against the 3.70 that dt^2 predicts -- so no single step reaches an arbitrary tolerance and
# 'findGroundStateSSFM' has to walk dt down, re-relaxing through the trap's slowest modes at every
# rung. For g = 1 that re-relaxation was 605 of the run's 700 iterations.
#
# The functions below descend the same sphere with no time step at all, so there is no splitting
# error and no ladder, and they precondition the gradient, which is what fixes the slow modes the
# ladder kept re-crossing. Measured against 'findGroundStateSSFM' on a 256^2 harmonic trap, both
# run to a residual of 1e-6:
#
#     g            0       1      10      50
#     speedup   10.1x   11.8x   22.9x   24.7x
#
# with the two energies agreeing to twelve significant figures and |<ssfm|pgd>| = 1 to all twelve.
# The gap widens with the tolerance rather than closing: imaginary time pays about three times the
# iterations per decade of residual, where this pays roughly three times in total between 1e-6 and
# 1e-12.


def _linear_part(
    psi: np.ndarray,
    kin: KineticStep,
    V: np.ndarray,
) -> np.ndarray:
    """The part of H.psi that does not depend on the density, (-grad^2/2 + V).psi.

    Split out from the interaction because it is *linear*, and the line search below needs it
    applied to the search direction as well as to the state. Keeping the two apart also keeps the
    interaction term free of transforms: it is elementwise, so 'gradient' adds it without paying
    for a second pair.

    The anti-aliasing mask is applied to the kinetic term alone, which is where 'KineticStep' puts
    it too -- the real-space factor of a split step is never masked. That makes this the same
    operator the imaginary-time solver propagates, so the two solvers converge to the same state
    rather than to two slightly different ones.

    Args:
        psi (np.ndarray): The state, or any array on the grid.
        kin (KineticStep): The grid's kinetic propagator, for its reciprocal space and mask.
        V (np.ndarray): Potential landscape.

    Returns:
        np.ndarray: (-grad^2/2 + V).psi.
    """
    f = fftn(psi, kin.workers)
    f *= kin.aliasing * kin.k2 / 2
    return ifftn(f, kin.workers, overwrite_x=True) + V * psi


def _precondition(
    r: np.ndarray,
    kin: KineticStep,
    V: np.ndarray,
    g: float,
    rho: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Apply the symmetric preconditioner of Antoine, Levitt and Tang to a residual.

    The stiffness of the problem has two separable sources: the kinetic operator, diagonal in
    Fourier space and large at high k, and the potential plus interaction, diagonal in real space
    and large where the trap is steep. Neither is diagonal in the same basis as the other, but each
    is cheap to invert in its own, so

        P = P_V^(1/2) . P_k . P_V^(1/2),  P_k = (alpha + k^2/2)^-1,  P_V = (alpha + V + g.rho)^-1

    inverts both approximately for two transforms. The shift alpha is the chemical potential, which
    is the scale that makes each factor invertible rather than singular near the condensate's edge.

    This is the part that earns the speedup. The modes the imaginary-time ladder kept re-relaxing
    through are the small-eigenvalue end of the Hessian, and this amplifies them by exactly the
    factor that brings them back level with the rest.

    Args:
        r (np.ndarray): The tangent residual to precondition.
        kin (KineticStep): The grid's kinetic propagator.
        V (np.ndarray): Potential landscape.
        g (float): Non-linear coefficient.
        rho (np.ndarray): The current density, |psi|^2, which the real-space factor reads.
        alpha (float): The shift, the chemical potential of the current state. Assumed positive;
        it is what keeps the real-space factor invertible at the edge of the condensate.

    Returns:
        np.ndarray: P.r, on whichever of host or device 'r' lives.
    """
    xp = array_module(r)

    # The interaction's contribution is clamped at zero, which changes nothing at all for a
    # repulsive gas and is what makes an attractive one work.
    #
    # A preconditioner has to approximate the inverse of something positive definite, or it stops
    # being a descent direction and the whole argument for it collapses. With g < 0 the real-space
    # factor (alpha + V + g.rho) is not: it sinks towards zero at the centre of the trap, where
    # the density is highest, and its reciprocal blows up exactly there. Measured on a 96^2 trap
    # at g = -0.5, keeping the negative term left the solver stuck at a residual of 0.55 after
    # 20000 iterations; clamping it reaches 1e-6 in 0.2 s and lands on the imaginary-time solver's
    # state to ten digits of overlap.
    #
    # Dropping the term rather than taking its magnitude keeps this honestly a preconditioner for
    # the positive part of the operator: where attraction dominates there is simply nothing useful
    # to invert, and alpha + V is the best cheap approximation available.
    scale = max(abs(alpha), 1e-300)
    sv = 1.0 / xp.sqrt(
        xp.maximum(alpha + V + xp.maximum(g * rho, 0.0), 1e-8 * scale)
    )

    z = sv * r
    f = fftn(z, kin.workers)
    f *= kin.aliasing / (alpha + kin.k2 / 2)
    z = sv * ifftn(f, kin.workers, overwrite_x=True)

    # The trailing multiply by sv is a real-space function, so it puts content back outside the
    # band that the mask in the middle had just removed, and the direction would walk the state
    # out of the band a little on every iteration. The residual then cannot fall below the size
    # of that leak -- it floored at 7e-6 on a 128^2 trap, whatever tolerance was asked for.
    # Projecting here also keeps the operator symmetric on the band, since the residual arriving
    # is already band-limited and so the matching projection on the right is a no-op.
    f = fftn(z, kin.workers)
    f *= kin.aliasing
    return ifftn(f, kin.workers, overwrite_x=True)


def _line_search(
    psi: np.ndarray,
    d: np.ndarray,
    Tpsi: np.ndarray,
    Td: np.ndarray,
    r: np.ndarray,
    g: float,
) -> tuple[float, float]:
    """Minimize the energy functional exactly along the great circle through psi and d.

    'psi' and 'd' are orthogonal and of equal norm, so psi(theta) = cos(theta).psi + sin(theta).d
    stays on the sphere for every theta and the population needs no renormalization afterwards.

    Along that circle the energy is a trigonometric polynomial, which is why this can be exact
    instead of a backtracking search. The quadratic part of the functional contributes harmonics in
    2.theta, and the quartic interaction term contributes up to 4.theta:

        E(theta) = const + p1.cos2t + q1.sin2t + p2.cos4t + q2.sin4t

    and the constant is dropped, so four numbers -- each a reduction over arrays already in hand --
    describe the whole line. It matters that this is exact. A backtracking search on the same
    descent direction stalls at a residual of about 1e-6: it starts failing to find a theta that
    lowers the energy while the state is still measurably off, which is exactly the regime a
    ground state solver exists to get past.

    Note that the energy minimized here is the *functional*, carrying g/2.int|psi|^4, and not
    'get_energy's <psi|H|psi>, which carries the full g. Descending the latter would converge to
    the wrong state.

    Args:
        psi (np.ndarray): The current state.
        d (np.ndarray): The search direction, orthogonal to psi and of the same norm.
        Tpsi (np.ndarray): The linear part of the Hamiltonian applied to psi, from '_linear_part'.
        Td (np.ndarray): The same applied to d.
        r (np.ndarray): The tangent residual at psi, which fixes the slope at theta = 0. See the
        note on cancellation below for why it is taken as an argument rather than reconstructed.
        g (float): Non-linear coefficient.

    Returns:
        tuple[float, float]: The minimizing theta, and the energy change it brings about.
    """
    xp = array_module(psi)

    # --- the quadratic part, from the linear operator ---
    Qpp = float(xp.real(xp.vdot(psi, Tpsi)))
    Qdd = float(xp.real(xp.vdot(d, Td)))

    # --- the quartic part, from the interaction ---
    # rho(theta) = c^2.A + s^2.B + 2cs.C, so sum rho^2 expands over these six reductions
    A = density(psi)
    B = density(d)
    C = xp.real(xp.conjugate(psi) * d)
    a = float(xp.sum(A * A))
    b = float(xp.sum(B * B))
    e = 4 * float(xp.sum(C * C)) + 2 * float(xp.sum(A * B))
    f = 4 * float(xp.sum(A * C))
    h = 4 * float(xp.sum(B * C))

    # c^4, s^4, c^2s^2, c^3s and cs^3 resolved into harmonics, then collected
    p1 = (Qpp - Qdd) / 2 + g / 2 * (a - b) / 2
    p2 = g / 2 * ((a + b) / 8 - e / 8)
    q2 = g / 2 * (f - h) / 8

    # q1 carries the slope at theta = 0, and it is the one coefficient that must stay accurate
    # when the state is nearly converged -- it is what tells the search how far to go, and
    # everything else only shapes the curve around it.
    #
    # Writing it the obvious way, as Re<psi|T|d>, destroys it. That inner product sums terms of
    # the size of the kinetic energy to a total of the size of the residual, so near convergence
    # it cancels away its whole mantissa: measured on a 128^2 trap, the search started returning
    # theta ~ 1e-18 from iteration 100 onwards and the residual froze at 6.7e-6, while the state
    # was still visibly short of the ground state.
    #
    # The same number is available without cancellation. dE/dtheta at 0 is 2.Re<d|grad E> =
    # 2.Re<d|r>, since d is orthogonal to psi, and reading it off the harmonics above gives
    # 2.q1 + 4.q2. Here r is small everywhere and d is O(1), so nothing cancels.
    q1 = float(xp.real(xp.vdot(d, r))) - 2 * q2

    def E(t):
        """The energy *change* from theta = 0, which is the only part that matters here.

        Written through cos(2t) - 1 = -2.sin^2(t) rather than as a difference of two evaluations.
        The coefficients are the size of the kinetic energy while the change being resolved near
        convergence is many orders smaller, so forming E(t) - E(0) numerically cancels the answer
        away and the search cannot tell a descent from a flat line -- which stalled it at a
        residual of 2e-6 regardless of the tolerance asked for. These identities are exact and
        leave nothing to cancel.
        """
        s2 = np.sin(t) ** 2
        s4 = np.sin(2 * t) ** 2
        return (
            -2 * p1 * s2 + q1 * np.sin(2 * t) - 2 * p2 * s4 + q2 * np.sin(4 * t)
        )

    def dE(t):
        return (
            -2 * p1 * np.sin(2 * t) + 2 * q1 * np.cos(2 * t)
            - 4 * p2 * np.sin(4 * t) + 4 * q2 * np.cos(4 * t)
        )

    def d2E(t):
        return (
            -4 * p1 * np.cos(2 * t) - 4 * q1 * np.sin(2 * t)
            - 16 * p2 * np.cos(4 * t) - 16 * q2 * np.sin(4 * t)
        )

    # E has period pi, so a coarse scan over one period brackets the global minimum and cannot be
    # fooled by the second harmonic the way a local search from theta=0 could be. The polynomial is
    # five scalars, so the scan costs nothing next to a single grid operation.
    ts = np.linspace(0, np.pi, 180, endpoint=False)
    t = float(ts[np.argmin(E(ts))])

    # Newton to machine precision from inside the bracket
    for _ in range(50):
        curv = d2E(t)
        if curv <= 0:
            break
        step = dE(t) / curv
        t -= step
        if abs(step) < 1e-15:
            break

    return t, float(E(t))


def findGroundStatePGD(
    kin: KineticStep,
    psi0: np.ndarray,
    V: np.ndarray | xr.DataArray,
    g: float,
    tol_stop: float,
    maxiter: int,
) -> tuple[float, np.ndarray]:
    """Find the ground state by preconditioned gradient descent on the constrained energy.

    The alternative to 'findGroundStateSSFM', and the default of 'GroundStateSSFM.solve'. There is
    no time step anywhere in it: each iteration forms the gradient of the energy functional,
    projects it onto the sphere of fixed population, preconditions it, and minimizes exactly along
    the resulting great circle. So there is no splitting error to chase and no ceiling to walk
    down, and the tolerance is reached by converging rather than by shrinking a step.

    One iteration costs eight transforms, against the twelve an accepted adaptive imaginary-time
    step costs, and it takes one to two orders of magnitude fewer of them. See the comment above
    '_linear_part' for the measured comparison.

    An attractive gas was expected to be the case where imaginary time kept the advantage, on the
    reasoning that the functional is unbounded below past the collapse threshold and a descent
    method would walk off it where a capped time step merely crawls. Measured on a 128^2 trap, the
    opposite happens: at g = -1 and g = -2 it is 'findGroundStateSSFM' that runs out of iterations,
    at residuals of 8.9e-4 and 1.3e-4, while this reaches 1e-6 in both. The reasoning was not
    wrong about the collapse, only about which solver meets it first -- the exact line search never
    takes a step that raises the energy, which is a firmer guard than a step-size ceiling.

    What that case does need is the clamp in '_precondition': without it this solver stalls
    completely for g < 0. See the note there.

    Args:
        kin (KineticStep): The grid's kinetic propagator, carrying reciprocal space and the
        anti-aliasing mask.
        psi0 (np.ndarray): Initial vector, normalized to the population wanted. The population is
        conserved exactly from here on, since every step moves along the sphere.
        V (Union[np.ndarray,xr.DataArray]): Potential landscape.
        g (float): Interaction strength term.
        tol_stop (float): The relative residual of the Gross-Pitaevskii equation at which the state
        is accepted, see 'gp_residual'. The same meaning and the same scale as the argument of
        'findGroundStateSSFM'.
        maxiter (int): Maximal number of iterations.

    Returns:
        tuple[float, np.ndarray]: <psi|H|psi>, see 'get_energy', and the ground state.
    """
    xp = array_module(psi0)
    V_bare = V.data if isinstance(V, xr.DataArray) else V

    def project(a):
        """Drop whatever lies outside the anti-aliasing band."""
        f = fftn(a, kin.workers)
        f *= kin.aliasing
        return ifftn(f, kin.workers, overwrite_x=True)

    # The seed is projected once, and every later direction is band-limited because the
    # preconditioner masks, so the state stays inside the band for the whole run. Without this the
    # out-of-band part of the seed is untouchable -- no search direction can reach it -- and it
    # sits in the state forever, holding the residual up at a floor the loop never clears.
    nrm = float(xp.linalg.norm(psi0))
    psi = project(psi0)
    psi *= nrm / float(xp.linalg.norm(psi))
    Tpsi = _linear_part(psi, kin, V_bare)

    residual = np.inf
    for _count in range(1, maxiter + 1):
        rho = density(psi)

        # V.psi and g.rho.psi both spill outside the band even when psi does not, and that spill
        # is not part of the problem being solved: the projected Hamiltonian is what the split-step
        # solver propagates and what 'gp_residual' measures. Projecting here is what lets the
        # residual reach zero instead of flooring on content no search direction can correct.
        Hpsi = project(Tpsi + g * rho * psi)

        n2 = float(xp.real(xp.vdot(psi, psi)))
        mu = float(xp.real(xp.vdot(psi, Hpsi))) / n2

        # The tangent residual: the part of H.psi that is not along psi. This is exactly the
        # quantity 'gp_residual' forms to decide whether to stop -- here it also points the way.
        r = Hpsi - mu * psi
        H_norm = max(float(xp.linalg.norm(Hpsi)), 1e-300)
        residual = float(xp.linalg.norm(r)) / H_norm
        if residual < tol_stop:
            break

        z = _precondition(r, kin, V_bare, g, rho, mu)
        z -= psi * (xp.vdot(psi, z) / n2)  # back onto the tangent space

        # The preconditioner is symmetric positive definite, so -z descends; if rounding near
        # convergence says otherwise, the unpreconditioned residual still does.
        if float(xp.real(xp.vdot(r, z))) <= 0:
            z = r

        dn = float(xp.linalg.norm(z))
        if dn < 1e-300:
            break
        d = z * (-nrm / dn)

        Td = _linear_part(d, kin, V_bare)
        theta, dE = _line_search(psi, d, Tpsi, Td, r, g)

        if dE >= 0.0:
            break  # the steepest descent line has no lower point on it: converged to rounding

        c, s = np.cos(theta), np.sin(theta)
        psi = c * psi + s * d

        # T is linear, so the next iteration's linear part comes for free from the two already in
        # hand, saving the two transforms rebuilding it would cost. Rebuilt from scratch every so
        # often all the same, so that the reuse cannot accumulate drift over a long run -- measured
        # at 1e-14 relative after fifty iterations, so the period is set well inside what matters.
        Tpsi = _linear_part(psi, kin, V_bare) if _count % 50 == 0 else c * Tpsi + s * Td

        # The circle preserves the norm analytically; this only stops rounding from drifting it
        # over thousands of iterations. The same scalar goes onto Tpsi, which is linear in psi.
        kappa = nrm / float(xp.linalg.norm(psi))
        psi *= kappa
        Tpsi *= kappa
    else:
        print(
            "maximal number of iteration reached, the result might not be converged "
            f"(residual {residual:.1e} for tol_stop = {tol_stop:.0e})"
        )

    return get_energy(psi, kin.k2, V_bare, g), psi


class GroundStateSSFM(SpectralSolver):
    """A ground state solver for rectangular grids and scalar equations, much faster than the solver form the GroundState class, but less general."""

    def __init__(
        self,
        potential: Potential,
        g: float | xr.DataArray,
    ):
        """Initialize a ground state finder instance for the Gross-Pitaevskii equation. This solver handles only scalar equations on rectangular grids.

        Args:
            potential (Potential): The potential landscape, must be describing an axis-aligned rectangular grid.
            The solver will iterate over each additional dimensions (not a1, a2, ...).
            g (Union[float, xr.DataArray]): Interaction strength term. Can be passed as an array over which to iterate.

        Raises:
            ValueError: If the potential grid is not rectangular and axis-aligned.
        """

        super().__init__(potential, g)

    def initialize_eigva(self):
        return super().initialize_eigva(1).squeeze("band")

    def energy_components(self, psi: xr.DataArray) -> xr.Dataset:
        """Split the energy of a ground state into its kinetic, potential and interaction parts.

        The xarray front-end to the 'energy_components' function, applied over whatever parameter
        dimensions the state carries: hand it the array 'solve' returned and the sweep comes back
        whole, with the potential and the interaction strength selected point by point alongside it.

        Args:
            psi (xr.DataArray): A ground state over the spatial dims, with any number of parameter
            dims on top of them.

        Returns:
            xr.Dataset: 'population', 'kinetic', 'potential', 'interaction', 'total' and 'mu', each
            over the parameter dims of psi. See the 'energy_components' function for their meaning,
            and note that 'total' is the energy functional while 'mu' carries the full g.
        """
        dS = self.potential.get_dS()
        keys = ("population", "kinetic", "potential", "interaction", "total", "mu")

        def _one(w, V, g):
            comps = energy_components(w, self.k2, V, float(g), dS)
            return tuple(comps[k] for k in keys)

        outputs = xr.apply_ufunc(
            _one,
            psi,
            self.potential.V,
            xr.DataArray(self.g) if not isinstance(self.g, xr.DataArray) else self.g,
            input_core_dims=[self.spatial_dims, self.spatial_dims, []],
            output_core_dims=[[]] * len(keys),
            vectorize=True,
        )
        return xr.Dataset(dict(zip(keys, outputs)))

    def initialize_psi(self):
        return (
            super()
            .initialize_eigve(1, False)
            .squeeze("band")
            .transpose(..., *self.spatial_dims)
            .rename("ground state")
        )

    def solve(
        self,
        population: float | xr.DataArray,
        tol_adapt: float = 1e-8,
        tol_stop: float = 1e-6,
        maxiter: int = 10000,
        dt_max: float | None = None,
        method: str = "pgd",
        phase0: tuple[float] | None = None,
        parallel: bool = False,
        n_cores: int = 4,
        workers: int = 1,
        verbose: bool = True,
        device: str | bool = "auto",
        checkpoint: str | None = None,
    ) -> xr.DataArray:
        """Solves the gross-Pitaevskii equation for each point in parameter space and return the ground state. see doc of 'findGroundStateSSFM' for more infos.

        Args:
            method (str, optional): Which solver to use. "pgd" is preconditioned gradient descent
            on the constrained energy, see 'findGroundStatePGD': it has no time step, so no
            splitting error and no step-size ladder, and it is the faster of the two by better
            than a decade on every trap measured -- including the attractive ones, where it is
            also the more robust of the two. "ssfm" is the adaptative imaginary-time split-step of
            'findGroundStateSSFM', kept for comparison and for anything tuned against its exact
            output. 'tol_adapt' and 'dt_max' belong to "ssfm" alone and are ignored under "pgd".
            Defaults to "pgd".
            tol_adapt (float): Tolerance for the adaptative time step method. "ssfm" only.
            tol_stop (float): The relative residual ||P(H - mu)psi|| / ||P.H.psi|| of the
            Gross-Pitaevskii equation at which a state is accepted, see 'gp_residual'. This is the
            accuracy knob: the solver shrinks its time step as far as needed to reach it, and each
            decade costs about three times more iterations. The real-time solvers see a state
            accepted at a residual r as stationary up to density fluctuations of a few r. Defaults
            to 1e-6.
            NOTE that the meaning of this number changed: it used to bound the Fubini-Study
            distance between two successive iterates, which is met at a fixed point displaced from
            the ground state by the splitting error, however tight it is. The values of 1e-9 to
            1e-13 that older scripts pass now ask for a far tighter state than they used to, and
            are likely to run to 'maxiter' instead.
            maxiter (int): maximal number of iteration for the solver.
            dt_max (float, optional): A ceiling on the imaginary time step, see
            'findGroundStateSSFM'. "ssfm" only. Defaults to None, no ceiling.
            phase0 (tuple[float], optional): Where to fix the phase of the ground state to 0, in the (a1, ..., an, field) basis. Defaults to (0.01, ..., 0.01, 0).
            parallel (bool, optional): Whether to use the parallel solver, this involve some overhead, so do not use it for too small parameter spaces. default to False.
            n_cores (int, optional): The number of cores to use for the parallelized solver.
            workers (int, optional): Threads given to each Fourier transform, -1 for every core. Worth
            raising for grids of 512^2 and upwards. Leave it at 1 when 'parallel' is set. Defaults to 1.
            verbose (bool, optional): Whether to plot a progress bar over the ground states.
            Defaults to True.
            device (str or bool, optional): "cpu", "gpu", or "auto" to let 'BECs.backend' decide.
            Imaginary-time propagation is the same split-step inner loop as the real-time solvers,
            so it gains the same way on a device, and "auto" applies the same measured crossover:
            the GPU is picked only when there is a working one and the grid is above about 128^2,
            below which kernel launches cost more than they save. With one card the sweep runs as a
            serial loop, since a single grid already saturates it. Defaults to "auto".
            checkpoint (str or Path, optional): A directory to keep each finished state in, so
            that a sweep killed partway through can be resumed by calling this again with the
            same argument -- the states already there are loaded rather than recomputed. Worth
            passing for anything long on a GPU that also drives a display, where the driver's
            watchdog destroys the CUDA context without warning and nothing in this process can
            recover it. The directory is deleted again the moment the sweep completes. Defaults
            to None, which keeps nothing.

        Returns:
            xr.DataArray: The value of the ground state for each time sampling point at each point of the parameter space.
        """
        if method not in ("pgd", "ssfm"):
            raise ValueError(
                f"method must be 'pgd' or 'ssfm', got {method!r}"
            )
        if method == "pgd" and dt_max is not None and verbose:
            print(
                "  ignoring dt_max: 'pgd' has no time step to put a ceiling on. Pass "
                "method='ssfm' for the imaginary-time solver that does."
            )

        n_points = int(np.prod([self.potential.V.sizes[d] for d in self.spatial_dims]))
        device, why = resolve_device(
            device,
            n_points=n_points,
            n_bytes=n_points * np.dtype(complex).itemsize * 10,
        )
        xp = get_xp(device)
        if verbose:
            print(f"GroundStateSSFM running on {device.upper()}: {why}")
            for line in device_warnings(device):
                print(line)
        if device == "gpu" and parallel and n_cores > 1:
            if verbose:
                print(
                    f"  ignoring n_cores={n_cores}: one device is already saturated by a single "
                    "grid, so the sweep is run as a serial loop on it."
                )
            parallel = False

        # add a population variable is needed. Registering rather than updating: a dimension the
        # potential already depends on would otherwise be relabelled 'population' outright, and
        # the potential selection would quietly stop seeing it.
        if isinstance(population, xr.DataArray):
            for dim in population.dims:
                check_name(dim, self.n_dims)
                register_coord(
                    self.allcoords, dim, "population", population.coords[dim]
                )

        # Create empty DataArrays to store the eigenvalues and vectors
        grounds = self.initialize_psi()
        energies = self.initialize_eigva()

        # We create a list of tuples that select a single value for each of the parameter dimensions
        selections = self.selections()

        # We will store the value of each parameter of the 'propagate' function for each iteration in lists.
        V_list = []
        g_list = []
        psi0_list = []

        for indexes in selections:
            # --- Constructing the inputs for 'propagate' ---
            ## select the potential, with its axes in the same order as psi
            potential_sel = subselect(indexes, "potential", self.allcoords)
            potential_selected = (
                self.potential.V.sel(potential_sel).transpose(*self.spatial_dims).data
            )

            ## Select the interaction strength
            g_sel = subselect(indexes, "g", self.allcoords)
            g_selected = selected_value(self.g, g_sel)

            if isinstance(population, xr.DataArray):
                pop_sel = subselect(indexes, "population", self.allcoords)
                pop = float(population.sel(pop_sel).data)
            else:
                pop = population

            psi0 = gaussian_filter(potential_selected.real.max()-potential_selected.real, sigma = 3)
            psi0 = self.normalize(psi0, pop)
                        
            # Store the arguments for the ground state solver as lists
            V_list += [potential_selected]
            g_list += [g_selected]
            psi0_list += [psi0]

        def x(psi0, pot, g):
            # Moved onto the device here rather than in the loop above, so that under
            # 'parallel=True' -- which is CPU-only -- what crosses the process boundary stays a
            # host array. 'xp.asarray' is a no-op on the CPU.
            kin = self.kinetic_step(workers, xp=xp)
            if method == "pgd":
                energ, eigvec = findGroundStatePGD(
                    kin, xp.asarray(psi0), xp.asarray(pot), g, tol_stop, maxiter
                )
            else:
                energ, eigvec = findGroundStateSSFM(
                    kin,
                    xp.asarray(psi0),
                    xp.asarray(pot),
                    g,
                    tol_adapt,
                    tol_stop,
                    maxiter,
                    dt_max,
                )
            return float(energ), to_host(eigvec)

        def store(indexes, energ, eigvec):
            # The dS factor is applied once, on return, for every branch
            energies[*indexes] = energ
            grounds[*indexes] = eigvec

        # The states already on disk from an earlier attempt, if this sweep is checkpointing. The
        # signature is what stops a resume from mixing two calculations: point the same directory
        # at a sweep of a different shape or tolerance and it refuses rather than filling the gaps
        # with states that answer a different question.
        ckpt = (
            None
            if checkpoint is None
            else RunCheckpoint(
                checkpoint,
                len(psi0_list),
                {
                    "solver": type(self).__name__,
                    "dims": list(grounds.dims),
                    "shape": list(grounds.shape),
                    "method": method,
                    "tol_adapt": float(tol_adapt),
                    "tol_stop": float(tol_stop),
                    "maxiter": int(maxiter),
                    "dt_max": None if dt_max is None else float(dt_max),
                },
            )
        )
        done = set() if ckpt is None else ckpt.done()
        if done and verbose:
            print(
                f"  resuming from {ckpt.path}: {len(done)} of {len(psi0_list)} states already done"
            )

        backend_ctx = fft_backend(device)

        if not parallel:
            # Now we can compute the ground state and store them in eigve
            with backend_ctx:
                for i in bar(
                    range(len(psi0_list)),
                    desc="Computing ground states",
                    unit="state",
                    verbose=verbose,
                ):
                    indexes = selections[i]
                    if i in done:
                        store(indexes, *ckpt.load(i))
                        continue
                    energ, eigvec = guarded_run(
                        x, (psi0_list[i], V_list[i], g_list[i]), i, len(psi0_list), ckpt
                    )
                    if ckpt is not None:
                        ckpt.save(i, energ, eigvec)
                    store(indexes, energ, eigvec)

        else:
            # Only the states still missing are submitted, so 'parallel_map' works over that
            # shorter list and 'pending' maps its answers back onto the sweep.
            pending = [i for i in range(len(psi0_list)) if i not in done]
            for i in done:
                store(selections[i], *ckpt.load(i))

            ev_list = parallel_map(
                x,
                [(psi0_list[i], V_list[i], g_list[i]) for i in pending],
                n_jobs=n_cores,
                desc="Computing ground states",
                unit="state",
                verbose=verbose,
            )

            for k, i in enumerate(pending):
                if ckpt is not None:
                    ckpt.save(i, ev_list[k][0], ev_list[k][1])
                store(selections[i], ev_list[k][0], ev_list[k][1])

        sel0 = self.phase_reference(phase0)

        grounds = grounds * xr.ufuncs.exp(
            -1j * xr.ufuncs.angle(grounds.sel(sel0, method="nearest"))
        )
        grounds = self.assign_cartesian(grounds)

        # The sweep is complete and every run of it is in the array below, so the copies on disk
        # have nothing left to protect against.
        if ckpt is not None:
            ckpt.discard()
            if verbose:
                print(f"  sweep complete: removed the checkpoint in {ckpt.path}")

        return energies.squeeze() * self.potential.get_dS(), grounds.squeeze()



if __name__ == '__main__':
    from bloch_schrodinger.potential import Potential, create_parameter

    lx = 5
    ly = 5
    nx = 100
    ny = 100
    
    omx = 5
    omy = 5
    
    pot = Potential(
        [[lx, 0], [0, ly]],
        resolution = (nx, ny),
        v0 = 0
    )
    
    gp = create_parameter('g', np.linspace(0,10,14))
    
    pot.set(
        pot.x**2 * omx**2 / 2 + pot.y**2 * omy**2/2
    )
    
    
    foo = GroundStateSSFM(
        pot, gp
    )
    
    eigva, eigve = foo.solve(
        4, 
        # tol = 1e-8, 
        maxiter = 5000, 
        parallel=False
    )
    
    # print(eigve)
    
    #%%
    import matplotlib.pyplot as plt
    from bloch_schrodinger.plotting import plot_cuts, plot_eigenvector
    plot_eigenvector(
        [[abs(eigve)**2, eigve.real]], [[pot, pot]], [['amplitude', 'real']]
    )
    plt.show()
    #%%
    plot_cuts(eigva, 'g', groupby=[])

# %%
