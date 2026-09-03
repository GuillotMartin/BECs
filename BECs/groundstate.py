import numpy as np
import scipy.sparse as sps
import xarray as xr
from bloch_schrodinger.fdsolver import FDSolver, check_name
from bloch_schrodinger.potential import Potential
from bloch_schrodinger.progress import bar, parallel_map
from scipy.fft import fftn
from scipy.ndimage import gaussian_filter
from scipy.sparse.linalg import eigsh

from BECs.spectral import KineticStep, SpectralSolver, density

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
    psi1_norm = psi1 / max(np.linalg.norm(psi1), 1e-15)
    psi2_norm = psi2 / max(np.linalg.norm(psi2), 1e-15)
    return np.abs((1 - np.abs(np.sum(np.conjugate(psi1_norm) * psi2_norm)) ** 2))


def normalize(psi: np.ndarray, norm: float) -> np.ndarray:
    """Renormalize a vector to a specified value, intended to be NaN resistant.

    Args:
        psi (np.ndarray): The vector to renormalize
        norm (float): Its desired norm

    Returns:
        np.ndarray: The normalized array
    """
    n = np.linalg.norm(psi)
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

    Args:
        indexes (tuple[int]): The indexes of the values to select in each coordinate array.
        ptype (str): The keyword to select the subselection, can be 'potential', 'alpha', 'g', 'reciprocal', 'population' or 'coupling'.
        allcoords (dict): The dictionnary containing the coordinates and their type

    Returns:
        dict: The subselection
    """
    subselect = {
        dim: allcoords[dim][1][i]
        for i, dim in zip(indexes, allcoords.keys())
        if allcoords[dim][0] == ptype
    }
    return subselect


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
        n_cores: int = 8,
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
        float: The energy of the mode, computed as <psi|H|psi>
    """

    # The orthogonal normalization makes the Fourier and the real space sums directly comparable
    psi_fft = fftn(psi, norm="ortho")
    Hpsi_fft = 1 / 2 * k2 * psi_fft
    Hpsi = (g * density(psi) + V) * psi

    E_fourier = np.real(np.sum(np.conjugate(psi_fft) * Hpsi_fft))
    E_real = np.real(np.sum(np.conjugate(psi) * Hpsi))
    return E_fourier + E_real


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
    return np.exp(-dt * (g * density(psi) + V)) * psi


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
    # interaction term reads the density, so it sees a different state if the norm has drifted
    psi_2 = inl_step(psi_1, dt, V, g)
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
    err = np.sum(density(psi_full - psi_double)) / np.sum(density(psi_full))

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
) -> tuple[float, np.ndarray]:
    """The main simulation function of the GroundStateSSFM class. Solves the Gross-Pitaevskii equation with an imaginary time step using an adaptative split-step Fourier method.

    Args:
        kin (KineticStep): The grid's kinetic propagator, carrying reciprocal space and the
        anti-aliasing mask.
        psi0 (np.ndarray): Initial vector.
        V (xr.DataArray): Potential landscape.
        g (float): Interaction strength term.
        tol_adapt (float): Tolerance for the adaptative time step method.
        tol_stop (float): Tolerance for determining wheter the ground state was found.
        maxiter (int): maximal number of iteration for the solver.

    Returns:
        np.ndarray: The ground state of the system
    """
    k2 = kin.k2
    E_list = [get_energy(psi0, k2, V, g)]

    dt, psi_next = iadaptative_step(
        psi0, kin, V, 2 * np.pi / 100 / E_list[0], g, tol_adapt
    )
    err = 1
    count = 0
    # propagating psi and storing at each time-step reaching the next t_sampling point
    while (err > tol_stop and count < maxiter) or count < 50:
        psi0 = psi_next * 1
        E_list += [get_energy(psi0, k2, V, g)]
        dt, psi_next = iadaptative_step(psi0, kin, V, dt, g, tol_adapt)
        err = distance(psi_next, psi0)
        count += 1

    E_list += [get_energy(psi_next, k2, V, g)]

    if count > maxiter - 1:
        print("maximal number of iteration reached, the result might not be converged")

    return E_list[-1], psi_next


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
        tol_stop: float = 1e-9,
        maxiter: int = 10000,
        phase0: tuple[float] | None = None,
        parallel: bool = False,
        n_cores: int = 8,
        workers: int = 1,
        verbose: bool = True,
    ) -> xr.DataArray:
        """Solves the gross-Pitaevskii equation for each point in parameter space and return the ground state. see doc of 'findGroundStateSSFM' for more infos.

        Args:
            tol_adapt (float): Tolerance for the adaptative time step method.
            tol_stop (float): Tolerance for determining wheter the ground state was found.
            maxiter (int): maximal number of iteration for the solver.
            phase0 (tuple[float], optional): Where to fix the phase of the ground state to 0, in the (a1, ..., an, field) basis. Defaults to (0.01, ..., 0.01, 0).
            parallel (bool, optional): Whether to use the parallel solver, this involve some overhead, so do not use it for too small parameter spaces. default to False.
            n_cores (int, optional): The number of cores to use for the parallelized solver.
            workers (int, optional): Threads given to each Fourier transform, -1 for every core. Worth
            raising for grids of 512^2 and upwards. Leave it at 1 when 'parallel' is set. Defaults to 1.
            verbose (bool, optional): Whether to plot a progress bar over the ground states.
            Defaults to True.

        Returns:
            xr.DataArray: The value of the ground state for each time sampling point at each point of the parameter space.
        """

        # add a population variable is needed
        if isinstance(population, xr.DataArray):
            for dim in population.dims:
                check_name(dim, self.n_dims)
            coords_pops = {
                dim: ["population", population.coords[dim]] for dim in population.dims
            }
            self.allcoords.update(coords_pops)

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
            g_selected = (
                self.g.sel(g_sel).data if isinstance(self.g, xr.DataArray) else self.g
            )

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
            return findGroundStateSSFM(
                self.kinetic_step(workers),
                psi0,
                pot,
                g,
                tol_adapt,
                tol_stop,
                maxiter,
            )

        if not parallel:
            # Now we can compute the ground state and store them in eigve
            for i in bar(
                range(len(psi0_list)),
                desc="Computing ground states",
                unit="state",
                verbose=verbose,
            ):
                indexes = selections[i]
                energ, eigvec = x(psi0_list[i], V_list[i], g_list[i])

                energies[*indexes] = energ
                grounds[*indexes] = eigvec

        else:
            ev_list = parallel_map(
                x,
                list(zip(psi0_list, V_list, g_list)),
                n_jobs=n_cores,
                desc="Computing ground states",
                unit="state",
                verbose=verbose,
            )

            for i in range(len(psi0_list)):
                indexes = selections[i]
                # The dS factor is applied once, on return, for both branches
                energies[*indexes] = ev_list[i][0]
                grounds[*indexes] = ev_list[i][1]

        sel0 = self.phase_reference(phase0)

        grounds = grounds * xr.ufuncs.exp(
            -1j * xr.ufuncs.angle(grounds.sel(sel0, method="nearest"))
        )
        grounds = self.assign_cartesian(grounds)

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
