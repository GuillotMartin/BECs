# %%
from collections.abc import Callable

import numpy as np
import xarray as xr
from joblib import Parallel, delayed
from scipy.fft import fftn, ifftn
from tqdm import tqdm

from BECs.potentialT import AnalyticPotential
from BECs.spectral import density
from BECs.ssfm import SSFM, distance


def der(psi: np.ndarray, da: list[float]) -> list[np.ndarray]:
    """A simple finite difference first derivative formula, along every axis.

    Args:
        psi (np.ndarray): The function to derivate
        da (list[float]): step along each axis of the grid

    Returns:
        list[np.ndarray]: [dpsi/drho_1, dpsi/drho_2, ...]
    """
    return [
        (np.roll(psi, -1, axis=i) - np.roll(psi, 1, axis=i)) / 2 / da[i]
        for i in range(len(da))
    ]


def J(phi: np.ndarray, ks: list[np.ndarray]) -> list[np.ndarray]:
    """Compute the current density in invariant coordinates.

    The solvers of this package propagate with exp(+i dt H) rather than exp(-i dt H), that is they
    carry the complex conjugate of the usual convention. The physical current is therefore
    -Im(conj(phi) grad phi) and not +Im(conj(phi) grad phi): with the other sign the rescaled frame
    expands while the cloud contracts, and vice versa.

    Args:
        phi (np.ndarray): wavefunction
        ks (list[np.ndarray]): the wavevector components, one array per axis.

    Returns:
        list[np.ndarray]: [J_1, J_2, ...]
    """
    phi_f = fftn(phi)
    dphi = [ifftn(phi_f * 1j * k) for k in ks]  # the full complex gradient
    return [-np.imag(np.conjugate(phi) * dphi_i) for dphi_i in dphi]


def a(psi: np.ndarray, coo: list[np.ndarray], dS: float) -> np.ndarray:
    """Compute the characteristic system size a_i in the coordinate system given by coo.

    Args:
        psi (np.ndarray): Initial wavefunction
        coo (list[np.ndarray]): cartesian coordinates, one array per axis
        dS (float): the length/surface/volume element of the grid
    """
    rho_sq = density(psi)
    return np.array([(np.sum(c**2 * rho_sq) * dS) ** 0.5 for c in coo])


def kinetic_factor(
    dt: complex,
    lambdas: np.ndarray,
    ks: list[np.ndarray],
    aliasing: np.ndarray,
) -> np.ndarray:
    """The reciprocal-space propagator of the rescaled frame, with the anti-aliasing mask folded in.

    Unlike the fixed-grid solver, this depends on the frame as well as on the step, so it cannot be
    cached across steps: lambda moves every one of them. It is built once per Strang step instead and
    used for both of that step's half kicks, which are always at the same dt and the same lambda.

    Args:
        dt (float): The time step.
        lambdas (np.ndarray): The values of the rescaling coefficients at time t.
        ks (list[np.ndarray]): Values of the wavevector components in the rescaled coordinates.
        Each axis is scaled by its own lambda, so this needs the components and not their norm.
        aliasing (np.ndarray): A high-k cut off mask for anti-aliasing.

    Returns:
        np.ndarray: The factor to multiply the transform of phi by.
    """
    return aliasing * np.exp(
        1j * dt * sum((k / lam) ** 2 for k, lam in zip(ks, lambdas)) / 2
    )


def linear_step(
    phi: np.ndarray,
    factor: np.ndarray,
) -> np.ndarray:
    """Linear propagation of the vector phi(t) by multiplication in Fourier space.

    Args:
        phi (np.ndarray): The vector to propagate.
        factor (np.ndarray): The propagator from 'kinetic_factor', which already carries the step,
        the frame and the anti-aliasing mask.

    Returns:
        np.ndarray: Propagated vector.
    """
    return ifftn(fftn(phi) * factor)

def potential_step(
    psi: np.ndarray,
    dt: complex,
    V: np.ndarray,
):
    """Phase rotation of the vector psi due to potential for a step dt by multiplication in real space.

    Args:
        psi (np.ndarray): The vector to propagate.
        dt (float): The time step.
        V (Union[np.ndarray,xr.DataArray]): Potential landscape in the rescaled coordinates.

    Returns:
        np.ndarray: Propagated vector.
    """
    return np.exp(1j * dt * V) * psi

def nl_step(
    psi: np.ndarray,
    dt: complex,
    g: float,
    lambdas: np.ndarray,
):
    """Non-linear propagation of the vector psi for a step dt by multiplication in real space.

    Args:
        psi (np.ndarray): The vector to propagate.
        dt (float): The time step.
        g (float): Non-linear coefficient.
        lambdas (np.ndarray): The values of the rescaling coefficients at time t.

    Returns:
        np.ndarray: Propagated vector.
    """
    return np.exp(1j * dt * g * density(psi) / np.prod(lambdas)) * psi


def dsigma(
    phi: np.ndarray,
    lambdas: np.ndarray,
    V: np.ndarray,
    consts: dict,
) -> np.ndarray:
    """The variational equation driving the rescaling coefficients, one per axis:

        a_i^2 lambda_ddot_i = 2 K_i / lambda_i^3 - <rho_i d_i V> + U / (lambda_i prod(lambda))

    obtained from the Euler-Lagrange equation of E(lambda) = sum_i K_i/lambda_i^2 + <V(lambda rho)>
    + U/prod(lambda). Note that 'der' differentiates on the rho grid, so the potential term carries a
    lambda_i that cancels the one in 'prefac' -- which is why only the kinetic and interaction terms
    end up scaled by 1/lambda_i.

    Args:
        phi (np.ndarray): The wavefunction in the rescaled frame.
        lambdas (np.ndarray): The rescaling coefficients at time t.
        V (np.ndarray): The potential, already evaluated on the rescaled grid lambda*rho.
        consts (dict): The constant parameters of the run.

    Returns:
        np.ndarray: The second derivative of each rescaling coefficient.
    """
    n_dims = len(lambdas)
    prefac = 1 / lambdas / consts["ai"] ** 2 * consts["dS"]

    rho_sq = density(phi)  # every term below wants it, so it is formed once

    # Non linear contribution
    nl_term = consts["g"] / 2 * np.sum(rho_sq**2) / (np.prod(lambdas))

    # Potential contribution
    dV = der(V, consts["da"])
    pot_term = [np.sum(rho_sq * consts["rho"][i] * dV[i]) for i in range(n_dims)]

    # Kinetic contribution, through Parseval rather than through a finite difference. The field lives
    # in a plane-wave basis, so sum_rho |d_i phi|^2 is exactly sum_k k_i^2 |phi_k|^2 / npoints -- no
    # discretization error, and one transform covers every axis instead of two array copies per axis.
    weights = density(fftn(phi)) / phi.size
    kin_term = [
        np.sum(consts["ks"][i] ** 2 * weights) / lambdas[i] ** 2 for i in range(n_dims)
    ]
    # The real part matters: an absorbing layer makes V complex, and the frame scaling is a real
    # geometric quantity. Without this the lambdas turn complex and their imaginary part is then
    # silently dropped when the moving coordinates are stored.
    return np.real(
        np.array(
            [prefac[i] * (nl_term - pot_term[i] + kin_term[i]) for i in range(n_dims)]
        )
    )


def compute_energy(
    phi: np.ndarray,
    sigmas: np.ndarray,
    lambdas: np.ndarray,
    V: np.ndarray,
    consts: dict,
) -> float:
    """Compute the total energy of the wavefunction phi using a modified GP Hamiltonian
    """
    n_dims = len(lambdas)

    rho_sq = density(phi)

    # Non linear contribution
    nl_term = consts["g"] / 2 * np.sum(rho_sq**2) / (np.prod(lambdas))

    # Potential contribution
    pot_term = np.sum(rho_sq * V)

    # Rescaling contribution: the kinetic energy of the frame motion, v_i = sigma_i rho_i
    V_rescaling = 0
    for i in range(n_dims):
        V_rescaling += sigmas[i] * sigmas[i] * consts["rho"][i] ** 2 / 2
    res_term = np.sum(rho_sq * V_rescaling)

    # Kinetic contribution, through Parseval as in 'dsigma' so that the energy and the equation
    # driving the frame are built from the same quantity
    weights = density(fftn(phi)) / phi.size
    kin_term = (
        sum(
            np.sum(consts["ks"][i] ** 2 * weights) / lambdas[i] ** 2
            for i in range(n_dims)
        )
        / 2
    )

    return (nl_term + pot_term + res_term + kin_term) * consts["dS"]


def potential_at(t: float, lambdas: np.ndarray, consts: dict) -> np.ndarray:
    """The potential evaluated on the moving grid lambda*rho at time t."""
    return consts["V"](
        t, *[lambdas[i] * consts["rho"][i] for i in range(len(lambdas))]
    )


def rescaling_potential(
    dsig: np.ndarray, lambdas: np.ndarray, consts: dict
) -> np.ndarray:
    """The frame's acceleration seen by the field, sum_i lambda_ddot_i lambda_i rho_i^2 / 2."""
    return sum(
        dsig[i] * lambdas[i] * consts["rho"][i] ** 2 / 2 for i in range(len(lambdas))
    )


def field_step(
    phi: np.ndarray,
    lambdas: np.ndarray,
    V: np.ndarray,
    dt: float,
    consts: dict,
) -> np.ndarray:
    """One symmetric Strang step of the field alone, on a frame held fixed at 'lambdas'.

    Args:
        phi (np.ndarray): The vector to propagate.
        lambdas (np.ndarray): The frame to step on. The caller passes the midpoint frame, which is
        what makes the coupling second order.
        V (np.ndarray): The full potential, external plus rescaling.
        dt (float): The time step.
        consts (dict): The constant parameters of the run.

    Returns:
        np.ndarray: Propagated vector.
    """
    # Both half kicks are at the same dt and the same frame, and both phase rotations are the same
    # exponential of the same potential, so each is built once rather than twice
    factor = kinetic_factor(dt / 2, lambdas, consts["ks"], consts["aliasing"])
    phase = np.exp(1j * (dt / 2) * V)

    p = linear_step(phi, factor)
    p = phase * p
    p = nl_step(p, dt, consts["g"], lambdas)
    p = phase * p
    return linear_step(p, factor)


def strang_step(
    phi: np.ndarray,
    sigmas: np.ndarray,
    lambdas: np.ndarray,
    t: float,
    dt: float,
    consts: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Propagate the field and the frame together for a full step dt, both to second order.

    The field takes a symmetric Strang step and the frame a velocity Verlet one, on the frame's
    equation lambda_ddot = F(phi, lambda) with F given by 'dsigma'. Three things have to be centred on
    the middle of the interval for the pair to be second order rather than first, and all three are:

      - the frame itself. 'lambda_mid' is what the field is stepped on, so the potential, the kinetic
        factor (k/lambda)^2 and the interaction's 1/prod(lambda) all see the midpoint frame.
      - the Verlet update of sigma, which uses the average of F at the two ends of the step. That is
        the trapezoid, equal to the midpoint value to second order.
      - the rescaling potential. Building it from F at the start of the step is what kept the method
        first order even with Verlet in place (measured convergence ratios 2.1 rather than 4.0), so a
        predictor pass gives F at the far end, and the field step is redone with the average.

    Args:
        phi (np.ndarray): The vector to propagate.
        sigmas (np.ndarray): first order time derivative of the rescaling coefficients at time t.
        lambdas (np.ndarray): Rescaling coefficients at t.
        t (float): time t for potential selection.
        dt (float): time step.
        consts (dict): The constant parameters of the run.

    Returns:
        np.ndarray: Propagated vector.
        np.ndarray : sigmas
        np.ndarray : lambdas
    """
    # --- the frame, drifted with the acceleration at the start of the step ---
    F0 = dsigma(phi, lambdas, potential_at(t, lambdas, consts), consts)
    lambdas_new = lambdas + sigmas * dt + F0 * dt**2 / 2
    lambdas_mid = lambdas + sigmas * dt / 2 + F0 * dt**2 / 8

    # --- the field, on the midpoint frame ---
    V_t = potential_at(t + dt / 2, lambdas_mid, consts)

    phi_pred = field_step(
        phi, lambdas_mid, V_t + rescaling_potential(F0, lambdas_mid, consts), dt, consts
    )
    F_end = dsigma(
        phi_pred, lambdas_new, potential_at(t + dt, lambdas_new, consts), consts
    )
    F_mid = (F0 + F_end) / 2

    phi_new = field_step(
        phi,
        lambdas_mid,
        V_t + rescaling_potential(F_mid, lambdas_mid, consts),
        dt,
        consts,
    )

    # --- the frame's velocity, closed on the corrected field ---
    F1 = dsigma(
        phi_new, lambdas_new, potential_at(t + dt, lambdas_new, consts), consts
    )
    sigmas_new = sigmas + (F0 + F1) * dt / 2

    return phi_new, sigmas_new, lambdas_new

def adaptative_step(
    phi: np.ndarray,
    sigmas: np.ndarray,
    lambdas: np.ndarray,
    t: float,
    dt: float,
    consts: dict,
    full: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Propagate phi for a full step dt, using a recursive adaptative step-doubling method.
    This function propagate psi for dt and for 2*dt/2, then compares the results. If its above a certain tolerance,
    the function calls itself again with a halved time step.

    Args:
        phi (np.ndarray): The vector to propagate.
        sigmas (np.ndarray): first order time derivative of the rescaling coefficients at time t.
        lambdas (np.ndarray): Rescaling coefficients at t.
        t (float): time t.
        dt (float): time step.
        consts (dict): all constant parameters
        full (tuple, optional): The single step of dt, when the caller has already computed it. A
        rejected step of 2.dt hands down the half step it took, which is exactly this call's full
        step, saving a third of the work of every rejection. Defaults to None.

    Returns:
        tuple[float, float, np.ndarray, np.ndarray, np.ndarray]: The time step length used, the optimal next time step length and the propagated vector, sigmas and lambdas
    """

    if full is None:
        full = strang_step(phi, sigmas, lambdas, t, dt, consts)
    phi_full, _, lambdas_full = full
    half = strang_step(phi, sigmas, lambdas, t, dt/2, consts)
    phi_half, sigmas_half, lambdas_half = half
    phi_double, sigmas_double, lambdas_double = strang_step(phi_half, sigmas_half, lambdas_half, t+dt/2, dt / 2, consts)


    # Computing the error, using a standard 2-norm.
    # err = np.sum(np.abs(psi_full - psi_double) ** 2) / np.sum(np.abs(psi_full) ** 2)
    # The frame counts too. Measuring the field alone left 'tol' with no influence at all over how
    # well lambda was tracked, which is the quantity the whole rescaling ansatz rests on.
    err = max(
        distance(phi_double, phi_full),
        float(np.max(np.abs(lambdas_double - lambdas_full)) / np.max(np.abs(lambdas_full))),
    )
    if err > consts["tol"]:  # If the error is superior, try again with a time step dt/2
        return adaptative_step(phi, sigmas, lambdas, t, dt / 2, consts, full=half)
    else:  # else return the results and compute a new time-step
        if err == 0:
            s = 10
        else:
            s = max(min(0.6 * (consts["tol"] / err) ** 0.25, 10), 0.1)
        return dt, s * dt, phi_double, sigmas_double, lambdas_double


def propagate(
    t_init: float,
    t_final: float,
    aliasing: np.ndarray,
    rho: list[np.ndarray],
    ks: list[np.ndarray],
    da: list[float],
    t_samples: xr.DataArray,
    psi: np.ndarray,
    V: Callable,
    dt: float,
    g: float,
    tol: float,
    verbose: bool = False,
    **kwargs,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """The main simualtion function of the submodule. Solves the Gross-Pitaevskii equation for the initial vector psi
    between t_init and t_final using an adaptative split-step Fourier method.

    Args:
        t_init (float): Initial time of simulation.
        t_final (float): Time when to stop the simulation.
        aliasing (np.ndarray): A high-k cut off mask for anti-aliasing.
        rho (list[np.ndarray]): System coordinates at t = 0, where x_i = rho_i, one array per axis.
        ks (list[np.ndarray]): The wavevector components, one array per axis.
        da (list[float]): The grid step along each axis.
        t_samples (xr.DataArray): List of sampling time at which to keep psi.
        psi (np.ndarray): Initial vector.
        V (Callable): Potential landscape, a function of the time and of the coordinates.
        dt (float): Initial time step.
        g (float): Interaction strength term.
        tol (float): Tolerance for the adaptative time step method.
        verbose (bool, optional): Wheter to plot a progress bar, useful for knowing where blow-up might happen. Defaults to False.

    Raises:
        ValueError: Raises an error if the initial sampling point is before t_init.

    Returns:
        tuple[list[np.ndarray], list[np.ndarray]]: The rescaling coefficients lambdas and the wavefunction psi sampled at the times specified by t_samples.
    """
    t = t_init
    n_dims = len(rho)
    dS = float(np.prod(da))

    # Compute initial conditions
    lambdas = np.ones(n_dims)

    J0 = J(psi, ks)
    a0 = a(psi, rho, dS) # at t = 0, the laboratory coordinates are equal to the rescaled coordinates rho_i
    sigmas = (
        np.array([np.sum(J0[i] * rho[i]) / a0[i] ** 2 for i in range(n_dims)]) * dS
    )

    def chirp(lambdas: np.ndarray) -> np.ndarray:
        """The phase the scaling ansatz carries, exp(-i/2 sum_i sigma_i lambda_i rho_i^2)."""
        return np.exp(
            -1j / 2 * sum(rho[i] ** 2 * sigmas[i] * lambdas[i] for i in range(n_dims))
        )

    # phi is psi with that phase removed, so this is the inverse of the reconstruction below
    phi = psi / chirp(lambdas)

    count_t = 0  # tracking what is the next sampling time.

    psi_list = [] #Regular wavefunction
    lambda_list = []
    dt_max = t_samples.data[1] - t_samples.data[0]

    consts = {
        "rho":rho, # The invariant coordinates rho.
        "ks":ks, # The invariant k-space coordinates
        "V":V, # The potential function
        "ai":a0, # The initial characteristic sizes
        "da":da, # The grid step along each axis
        "dS":dS,
        "aliasing": aliasing, # Aliasing mask
        "g":g, # non-linear factor
        "tol":tol # tolerance for the adaptative step
    }

    # verify that the first sampling point is not before t_init
    if t == t_samples[0]:
        psi_list += [psi]
        lambda_list += [lambdas]
        count_t += 1
    elif t > t_samples[0]:
        raise ValueError("First sampling point before initial simulation time")

    # The bounds on the step live here rather than inside the loop so that a progress bar cannot
    # change the numerics: the two branches this loop used to be written as disagreed on them, and a
    # run with verbose=True took a different sequence of steps from the same run without it.
    dtmax = kwargs.get("dtmax", 0.1)
    dtmin = kwargs.get("dtmin", 1e-6)

    # create a progress bar if asked
    pbar = (
        tqdm(
            total=t_final - t_init,
            bar_format="{l_bar}{bar}| {n:.3f}/{total_fmt}, {rate_fmt}, [{elapsed} < {remaining}]",
        )
        if verbose
        else None
    )

    # propagating psi and storing at each time-step reaching the next t_sampling point
    while t < t_final and count_t < len(t_samples):
        dt_used, dt, phi, sigmas, lambdas = adaptative_step(
            phi, sigmas, lambdas, t, dt, consts
        )
        t += dt_used
        dt = min(dt, dt_max)  # making sure not to skip sampling times
        dt = max(min(dt, dtmax), dtmin)  # bounding the step time to reasonable values

        if t >= t_samples[count_t]:
            psi_list += [phi / np.prod(lambdas) ** 0.5 * chirp(lambdas)]
            lambda_list += [lambdas]
            count_t += 1
        if pbar is not None and t + dt_used < t_final:
            pbar.update(dt_used)

    if pbar is not None:
        pbar.close()

    n_samples = len(psi_list)
    if n_samples != len(t_samples):
        print(
            f"Less time steps than required samples, padding the array with last psi, last proper sample is {n_samples}"
        )
        psi_list += [psi_list[-1]] * (len(t_samples) - n_samples)
        lambda_list += [lambdas] * (len(t_samples) - n_samples)

    return lambda_list, psi_list



class rSSFM(SSFM):
    """A Gross-Pitaevskii solver working in a rescaled frame.

    With x_i = lambda_i(t) rho_i, the field is written

        psi(x, t) = prod(lambda)^(-1/2) phi(rho, t) exp(-i/2 sum_i sigma_i lambda_i rho_i^2)

    so that phi lives on a grid of fixed resolution while the frame follows the cloud. The equation
    for phi is a modified GPE: its kinetic term carries 1/lambda_i^2, its interaction term
    1/prod(lambda), and an extra potential sum_i lambda_ddot_i lambda_i rho_i^2 / 2 accounts for the
    frame's acceleration, with lambda driven by the variational equation in 'dsigma'. This is what
    lets an expanding cloud be followed without an ever-growing simulation box.
    """

    def __init__(
        self,
        potential: AnalyticPotential,
        psi0: xr.DataArray,
        g: float | xr.DataArray,
    ):
        """Initialize a solver instance for the Gross-Pitaevskii equation. This solver handles only scalar equations on rectangular grids.

        Args:
            potential (AnalyticPotential): The potential landscape, which must be analytic: it is
            evaluated on a grid that moves at every time step. It must describe an axis-aligned
            rectangular grid, giving the rescaled coordinates rho_i at t = 0. The solver will iterate
            over each additional dimensions (not a1, a2, ...).
            psi0 (xr.DataArray): Initial vector, must have shape and spatial dimensions (a1, a2, ...) consistent with the potential.
            g (Union[float, xr.DataArray]): Interaction strength term. Can be passed as an array over which to iterate.

        Raises:
            ValueError: If the potential is not an AnalyticPotential.
            ValueError: If the potential and initial vector given do not have the proper dimensions.
            ValueError: If the potential grid is not rectangular and axis-aligned.
        """

        if not isinstance(potential, AnalyticPotential):
            raise ValueError(
                "The rescaling solver evaluates the potential on a moving grid, so it needs an "
                f"AnalyticPotential, got a {type(potential).__name__}"
            )

        super().__init__(potential, psi0, g)

    def initialize_lambda(self):
        lambdas = super().initialize_eigva().squeeze(drop=True)
        lambdas = lambdas.expand_dims({"axis": np.arange(self.n_dims)})
        lambdas.name = "lambda"
        return lambdas

    def initialize_psi(self):
        psi = super().initialize_psi()
        Lambda = self.initialize_lambda()

        names = self.potential.coord_names[: self.n_dims]

        # The invariant coordinates keep a name of their own, the cartesian ones follow the frame
        psi = psi.assign_coords({f"rho_{name}": psi.coords[name] for name in names})
        psi = psi.assign_coords(
            {
                name: self.potential.V.coords[name] * Lambda[{"axis": i}]
                for i, name in enumerate(names)
            }
        )
        return psi

    def solve(
        self,
        t_init: float,
        t_final: float,
        t_samples: xr.DataArray,
        dt0: float | xr.DataArray = 1e-3,
        tol: float | xr.DataArray = 1e-6,
        parallel: bool = False,
        verbose: bool = False,
        n_cores: int = 8,
        **kwargs,
    ) -> xr.DataArray:
        """Solves the gross-Pitaevskii equation for each point in parameter space. see doc of 'propagate' for more infos.

        Args:
            t_init (float): Initial time of simulation
            t_final (float): End time of simulation
            t_samples (xr.DataArray): Sampling times for psi. t_sample can have multiple dimensions, but one of them must be 't'.
            dt0 (Union[float, xr.DataArray], optional): Initial time step. Can have multiple dimensions, but they must be a subset of the parameter space. Defaults to 1e-3.
            tol (Union[float, xr.DataArray], optional): Tolerance for adaptative method. Can have multiple dimensions, but they must be a subset of the parameter space
            As a rule, the tolerance should decrease for higher values of g. Defaults to 1e-10.
            parallel (bool, optional): Whether to use the parallel solver, this involve some overhead, so do not use it for too small parameter spaces. default to False.
            verbose (bool, optional): Wheter to plot a progress bar, useful for knowing where blow-up might happen. Defaults to False.
            n_cores (int, optional): The number of cores to use for the parallelized solver.

        Returns:
            xr.DataArray: The value of Psi for each time sampling point at each point of the parameter space.
            Its cartesian coordinates follow the rescaled frame, so they carry a time dimension of their own.
        """

        names = self.potential.coord_names[: self.n_dims]

        psi, selections, list_args = self.prepare_runs(t_samples, dt0, tol)

        # Unlike the fixed-grid solver, the cartesian coordinates move with the frame, so they get
        # a time dimension too
        if "t" not in psi.coords[names[0]].dims:
            for name in names:
                psi.coords[name] = psi.coords[name].expand_dims(
                    dim={"t": t_samples.coords["t"]}
                )
        for name in names:
            psi.coords[name] = psi.coords[name].transpose("t", ...).copy()

        n_samples = len(t_samples.coords["t"].data)

        def x(y):
            return propagate(
                t_init,
                t_final,
                self.aliasing,
                [coord.data for coord in self.potential.coords],
                self.ks,
                self.da,
                *y,
                verbose=verbose,
                **kwargs,
            )

        def store(indexes, lambda_list, psi_list):
            """Write one run into psi, together with the frame it was computed in."""
            for j in range(n_samples):
                slic = [j, *indexes]
                for i, name in enumerate(names):
                    psi.coords[name][*slic] = (
                        psi.coords[f"rho_{name}"] * lambda_list[j][i]
                    )
                psi[*slic] = psi_list[j]

        if not parallel:
            print(
                f"Propagating the initial states. {len(selections)} iterations to perform"
            )
            for i, indexes in enumerate(selections):
                lambda_list, psi_list = x(list_args[i])
                store(indexes, lambda_list, psi_list)
        else:
            pool = Parallel(
                n_jobs=n_cores, return_as="list", verbose=51 if verbose else 5
            )
            results = pool(delayed(x)(y) for y in list_args)

            for i, indexes in enumerate(selections):
                store(indexes, results[i][0], results[i][1])

        return psi.squeeze()

