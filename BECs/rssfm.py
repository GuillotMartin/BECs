# %%
import json
from collections.abc import Callable

import numpy as np
import xarray as xr
from bloch_schrodinger.progress import bar, parallel_imap
from scipy.fft import fftfreq

from BECs.backend import (
    array_module,
    device_warnings,
    fft_backend,
    get_xp,
    guarded_run,
    resolve_device,
    to_host,
)
from BECs.backend import fftn as _fftn
from BECs.backend import ifftn as _ifftn
from BECs.io import RunCheckpoint
from BECs.potentialT import AnalyticPotential
from BECs.spectral import density, phase
from BECs.groundstate import fs_residual  # noqa: E402  (kept next to the other BECs imports)
from BECs.ssfm import SSFM


_EINSUM_LETTERS = "abcdefgh"


def _weighted_axis_sum(A: np.ndarray, B: np.ndarray, w: np.ndarray, axis: int) -> complex:
    """sum over the whole grid of A * B * w, where w varies only along 'axis'.

    Written as a contraction rather than as three broadcast multiplies and a sum. The dense form
    builds two full-grid temporaries; worse, once 'w' is the rank-1 column vector it ought to be,
    multiplying a C-ordered grid by it strides badly enough that the "optimized" version is
    measurably slower than the dense one. Reducing A*B onto 'axis' first and taking a dot product
    against 'w' there avoids both -- one pass, no temporary, and the vector it finally touches is
    512 long rather than 512^2. Measured at 512^2: 0.383 ms dense, 0.099 ms this way.

    Args:
        A (np.ndarray): A full grid.
        B (np.ndarray): A full grid.
        w (np.ndarray): The per-axis weight, of any shape that broadcasts along 'axis' alone.
        axis (int): The axis 'w' varies along.

    Returns:
        complex: The full-grid sum.
    """
    xp = array_module(A)
    idx = _EINSUM_LETTERS[: A.ndim]
    reduced = xp.einsum(f"{idx},{idx}->{idx[axis]}", A, B)
    return complex(xp.dot(reduced, xp.asarray(w).ravel()))


def _axis_profiles(field: np.ndarray) -> list[np.ndarray]:
    """'field' summed down to one vector per axis, so a per-axis weight can dot against it.

    The kinetic sums want sum(k_i^2 * weights) for every i off one array. Each of those is the
    same grid reduced along every axis but i, so the reductions are taken once here and shared
    rather than streaming the grid once per axis. Measured at 512^2: 0.230 ms dense, 0.069 ms here.

    Args:
        field (np.ndarray): A full grid.

    Returns:
        list[np.ndarray]: One 1-D array per axis.
    """
    return [
        field.sum(axis=tuple(j for j in range(field.ndim) if j != i))
        for i in range(field.ndim)
    ]


def der(psi: np.ndarray, da: list[float]) -> list[np.ndarray]:
    """A simple finite difference first derivative formula, along every axis.

    Args:
        psi (np.ndarray): The function to derivate
        da (list[float]): step along each axis of the grid

    Returns:
        list[np.ndarray]: [dpsi/drho_1, dpsi/drho_2, ...]
    """
    xp = array_module(psi)
    return [
        (xp.roll(psi, -1, axis=i) - xp.roll(psi, 1, axis=i)) / 2 / da[i]
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
    xp = array_module(phi)
    phi_f = _fftn(phi)
    dphi = [_ifftn(phi_f * 1j * k) for k in ks]  # the full complex gradient
    return [-xp.imag(xp.conjugate(phi) * dphi_i) for dphi_i in dphi]


def a(psi: np.ndarray, coo: list[np.ndarray], dS: float) -> np.ndarray:
    """Compute the characteristic system size a_i in the coordinate system given by coo.

    Args:
        psi (np.ndarray): Initial wavefunction
        coo (list[np.ndarray]): cartesian coordinates, one array per axis
        dS (float): the length/surface/volume element of the grid

    Returns:
        np.ndarray: The characteristic sizes, always on the host -- they belong to the frame, which
        is a handful of numbers integrated on the CPU whatever the field is running on.
    """
    xp = array_module(psi)
    rho_sq = density(psi)
    return np.array([(float(xp.sum(c**2 * rho_sq)) * dS) ** 0.5 for c in coo])


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
    return aliasing * phase(
        dt, sum((k / lam) ** 2 for k, lam in zip(ks, lambdas)) / 2
    )


def linear_step(
    phi: np.ndarray,
    factor: np.ndarray,
    workers: int = 1,
) -> np.ndarray:
    """Linear propagation of the vector phi(t) by multiplication in Fourier space.

    Args:
        phi (np.ndarray): The vector to propagate.
        factor (np.ndarray): The propagator from 'kinetic_factor', which already carries the step,
        the frame and the anti-aliasing mask.
        workers (int, optional): Threads handed to scipy's transforms. Defaults to 1.

    Returns:
        np.ndarray: Propagated vector.
    """
    # The k-space product is a temporary that nothing else can reach, so the inverse transform is
    # allowed to consume it rather than allocate a second grid for its output. Same for the
    # multiplication by the propagator, which writes back into the forward transform's own buffer.
    #
    # 'overwrite_x' is not free of consequence: an in-place transform is a different algorithm from
    # an out-of-place one, so this changes the result in the last bits -- measured at 3e-16 relative
    # over a full run, against tolerances of 1e-5 and looser. The hoisting in 'strang_step' is
    # exactly bit-identical on its own; this line is what makes the pair merely equal to roundoff.
    f = _fftn(phi, workers)
    f *= factor
    return _ifftn(f, workers, overwrite_x=True)


def linear_step_kept(
    phi: np.ndarray,
    factor: np.ndarray,
    workers: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """'linear_step', also handing back the k-space array it built on the way.

    Args:
        phi (np.ndarray): The vector to propagate.
        factor (np.ndarray): The propagator from 'kinetic_factor'.
        workers (int, optional): Threads handed to scipy's transforms. Defaults to 1.

    Returns:
        np.ndarray: Propagated vector.
        np.ndarray: Its Fourier transform.
    """
    f = _fftn(phi, workers)
    f *= factor
    return _ifftn(f, workers), f


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
    return phase(dt, V) * psi

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
    return phase(dt, g * density(psi) / np.prod(lambdas)) * psi


def dsigma(
    phi: np.ndarray,
    lambdas: np.ndarray,
    V: np.ndarray,
    consts: dict,
    dV: list[np.ndarray] | None = None,
    phi_f: np.ndarray | None = None,
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
        dV (list[np.ndarray], optional): The gradient of V, when the caller already holds it. The
        two calls at the far end of a Strang step differ only in 'phi' and share this, and it costs
        four full-grid rolls to take. Defaults to None, which computes it here.
        phi_f (np.ndarray, optional): fftn of 'phi', when the caller already holds it. Every call
        of this that follows a 'field_step' does, because the field step ends on an inverse
        transform and 'linear_step_kept' hands back what went into it. Defaults to None, which
        transforms here and costs a full grid transform.

    Returns:
        np.ndarray: The second derivative of each rescaling coefficient.
    """
    n_dims = len(lambdas)
    xp = array_module(phi)
    prefac = 1 / lambdas / consts["ai"] ** 2 * consts["dS"]

    rho_sq = density(phi)  # every term below wants it, so it is formed once

    # A dot product against itself rather than sum(rho_sq**2): same value, but it goes through
    # BLAS and never forms the squared grid.
    nl_term = (
        consts["g"] / 2 * float(xp.dot(rho_sq.ravel(), rho_sq.ravel()))
        / (np.prod(lambdas))
    )

    # Potential contribution
    if dV is None:
        dV = der(V, consts["da"])
    pot_term = [
        _weighted_axis_sum(rho_sq, dV[i], consts["rho"][i], i) for i in range(n_dims)
    ]

    # Kinetic contribution, through Parseval rather than through a finite difference.
    if phi_f is None:
        phi_f = _fftn(phi, consts["workers"])
    weights = density(phi_f) / phi.size
    w_axis = _axis_profiles(weights)
    kin_term = [
        float(xp.dot(w_axis[i], xp.asarray(consts["ks2"][i]).ravel()))
        / lambdas[i] ** 2
        for i in range(n_dims)
    ]
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
    xp = array_module(phi)

    rho_sq = density(phi)

    # Non linear contribution
    nl_term = (
        consts["g"] / 2 * float(xp.dot(rho_sq.ravel(), rho_sq.ravel()))
        / (np.prod(lambdas))
    )

    # Potential contribution
    pot_term = complex(xp.sum(rho_sq * V))

    # Rescaling contribution: the kinetic energy of the frame motion, v_i = sigma_i rho_i
    V_rescaling = 0
    for i in range(n_dims):
        V_rescaling += sigmas[i] * sigmas[i] * consts["rho2"][i] / 2
    res_term = float(xp.sum(rho_sq * V_rescaling))

    # Kinetic contribution, through Parseval as in 'dsigma' so that the energy and the equation
    # driving the frame are built from the same quantity
    weights = density(_fftn(phi, consts["workers"])) / phi.size
    w_axis = _axis_profiles(weights)
    kin_term = (
        sum(
            float(xp.dot(w_axis[i], xp.asarray(consts["ks2"][i]).ravel()))
            / lambdas[i] ** 2
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
        dsig[i] * lambdas[i] * consts["rho2"][i] / 2 for i in range(len(lambdas))
    )


def field_step(
    p0: np.ndarray,
    lambdas: np.ndarray,
    V: np.ndarray,
    dt: float,
    consts: dict,
    factor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """One symmetric Strang step of the field alone, on a frame held fixed at 'lambdas'.

    Args:
        p0 (np.ndarray): The field with its opening half kick already taken, that is
        'linear_step(phi, factor)'. It is passed in rather than taken here because 'strang_step'
        calls this twice per step -- predictor and corrector -- on the same field and the same
        propagator, and only the potential differs between them; that opening kick is therefore
        the same two transforms both times.
        lambdas (np.ndarray): The frame to step on. The caller passes the midpoint frame, which is
        what makes the coupling second order.
        V (np.ndarray): The full potential, external plus rescaling.
        dt (float): The time step.
        consts (dict): The constant parameters of the run.
        factor (np.ndarray): The reciprocal-space propagator for dt/2 on this frame, from
        'kinetic_factor'. Passed in rather than built here for the same reason as 'p0': the two
        calls would otherwise build the identical array twice, at about one and a half transforms
        each.

    Returns:
        np.ndarray: Propagated vector.
        np.ndarray: Its Fourier transform, for the 'dsigma' that follows.
    """
    # Both phase rotations are the same exponential of the same potential, so it is built once
    half_phase = phase(dt / 2, V)

    p = half_phase * p0
    p = nl_step(p, dt, consts["g"], lambdas)
    p *= half_phase
    return linear_step_kept(p, factor, consts["workers"])


def strang_step(
    phi: np.ndarray,
    sigmas: np.ndarray,
    lambdas: np.ndarray,
    t: float,
    dt: float,
    consts: dict,
    F0: np.ndarray | None = None,
    phi_f: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        F0 (np.ndarray, optional): The frame's acceleration at the start of the step, when the
        caller already holds it. It depends on phi, lambda and t but not on dt, so the step-doubling
        controller can compute it once and hand the same value to its full step, its first half step
        and every retry. Defaults to None, which computes it
        here, and costs a transform, a potential evaluation and a gradient.
        phi_f (np.ndarray, optional): fftn of 'phi', when the caller already holds it. Like 'F0'
        this depends on the starting state and not on dt, so the controller computes it once for
        its full step, its first half step and every retry. Defaults to None.

    Returns:
        np.ndarray: Propagated vector.
        np.ndarray : sigmas
        np.ndarray : lambdas
        np.ndarray : fftn of the propagated vector, so that the next step down the chain starts
        with its own 'phi_f' already in hand.
    """
    # --- the frame, drifted with the acceleration at the start of the step ---
    if F0 is None:
        F0 = dsigma(phi, lambdas, potential_at(t, lambdas, consts), consts, phi_f=phi_f)
    lambdas_new = lambdas + sigmas * dt + F0 * dt**2 / 2
    lambdas_mid = lambdas + sigmas * dt / 2 + F0 * dt**2 / 8

    # --- the field, on the midpoint frame ---
    V_t = potential_at(t + dt / 2, lambdas_mid, consts)

    # Predictor and corrector step the same field over the same dt on the same midpoint frame.
    factor = kinetic_factor(dt / 2, lambdas_mid, consts["ks"], consts["aliasing"])
    p0 = linear_step(phi, factor, consts["workers"])

    phi_pred, phi_pred_f = field_step(
        p0,
        lambdas_mid,
        V_t + rescaling_potential(F0, lambdas_mid, consts),
        dt,
        consts,
        factor,
    )

    V_end = potential_at(t + dt, lambdas_new, consts)
    dV_end = der(V_end, consts["da"])

    F_end = dsigma(phi_pred, lambdas_new, V_end, consts, dV=dV_end, phi_f=phi_pred_f)
    F_mid = (F0 + F_end) / 2

    phi_new, phi_new_f = field_step(
        p0,
        lambdas_mid,
        V_t + rescaling_potential(F_mid, lambdas_mid, consts),
        dt,
        consts,
        factor,
    )

    # --- the frame's velocity, closed on the corrected field ---
    F1 = dsigma(phi_new, lambdas_new, V_end, consts, dV=dV_end, phi_f=phi_new_f)
    sigmas_new = sigmas + (F0 + F1) * dt / 2

    return phi_new, sigmas_new, lambdas_new, phi_new_f

def adaptative_step(
    phi: np.ndarray,
    sigmas: np.ndarray,
    lambdas: np.ndarray,
    t: float,
    dt: float,
    consts: dict,
    full: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
    F0: np.ndarray | None = None,
    phi_f: np.ndarray | None = None,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        F0 (np.ndarray, optional): The frame's acceleration at (phi, lambdas, t), shared by the two
        steps that start there and by every retry. Defaults to None, which computes it here.
        phi_f (np.ndarray, optional): fftn of 'phi', shared exactly as 'F0' is. Defaults to None.

    Returns:
        tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The time step length used, the optimal next time step length, the propagated vector, sigmas, lambdas, and fftn of the propagated vector
    """

    # The full step and the first half step leave from the same state, so the acceleration there is
    # one and the same array; so is the one every rejected retry would recompute, and so is that
    # state's transform. Only the second half step starts somewhere else -- and it gets its own
    # transform for free, because the half step it follows ends on one.
    if F0 is None:
        F0 = dsigma(phi, lambdas, potential_at(t, lambdas, consts), consts, phi_f=phi_f)

    if full is None:
        full = strang_step(phi, sigmas, lambdas, t, dt, consts, F0=F0, phi_f=phi_f)
    # The full step's own transform is never read: this branch of the step doubling exists only to
    # be differenced against the doubled one. It rides along so that 'half' can be handed straight
    # back in as the next call's 'full'.
    phi_full, _, lambdas_full, _ = full
    half = strang_step(phi, sigmas, lambdas, t, dt/2, consts, F0=F0, phi_f=phi_f)
    phi_half, sigmas_half, lambdas_half, phi_half_f = half
    phi_double, sigmas_double, lambdas_double, phi_double_f = strang_step(
        phi_half, sigmas_half, lambdas_half, t+dt/2, dt / 2, consts, phi_f=phi_half_f
    )


    # Computing the error, using a standard 2-norm.
    # err = np.sum(np.abs(psi_full - psi_double) ** 2) / np.sum(np.abs(psi_full) ** 2)
    err = max(
        fs_residual(phi_double, phi_full),
        float(np.max(np.abs(lambdas_double - lambdas_full)) / np.max(np.abs(lambdas_full))),
    )
    if err > consts["tol"]:  # If the error is superior, try again with a time step dt/2
        return adaptative_step(
            phi, sigmas, lambdas, t, dt / 2, consts, full=half, F0=F0, phi_f=phi_f
        )
    else:  # else return the results and compute a new time-step
        if err == 0:
            s = 5
        else:
            s = max(min(0.6 * (consts["tol"] / err) ** (1 / 3), 5), 0.1)
        return dt, s * dt, phi_double, sigmas_double, lambdas_double, phi_double_f


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
    workers: int = 1,
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
    xp = array_module(psi)

    # Compute initial conditions
    lambdas = np.ones(n_dims)

    J0 = J(psi, ks)
    a0 = a(psi, rho, dS) # at t = 0, the laboratory coordinates are equal to the rescaled coordinates rho_i
    sigmas = (
        np.array(
            [float(xp.sum(J0[i] * rho[i])) / a0[i] ** 2 for i in range(n_dims)]
        )
        * dS
    )

    def chirp(lambdas: np.ndarray) -> np.ndarray:
        """The phase the scaling ansatz carries, exp(-i/2 sum_i sigma_i lambda_i rho_i^2)."""
        return xp.exp(
            -1j / 2 * sum(rho[i] ** 2 * sigmas[i] * lambdas[i] for i in range(n_dims))
        )

    # phi is psi with that phase removed, so this is the inverse of the reconstruction below
    phi = psi / chirp(lambdas)
    # Carried across the loop: every step ends on an inverse transform, so the next step's forward
    # transform is already in hand and only the very first one has to be taken.
    phi_f = None

    count_t = 0  # tracking what is the next sampling time.

    psi_list = [] #Regular wavefunction
    lambda_list = []
    dt_max = t_samples.data[1] - t_samples.data[0]

    consts = {
        "rho":rho, # The invariant coordinates rho.
        "ks":ks, # The invariant k-space coordinates
        # Both squares are constant for the whole run, and both used to be taken afresh on every
        # call of 'dsigma', 'compute_energy' and 'rescaling_potential' -- about six full-grid
        # squarings per Strang step. Formed once here instead.
        "rho2":[r**2 for r in rho],
        "ks2":[k**2 for k in ks],
        "V":V, # The potential function
        "ai":a0, # The initial characteristic sizes
        "da":da, # The grid step along each axis
        "dS":dS,
        "aliasing": aliasing, # Aliasing mask
        "g":g, # non-linear factor
        "tol":tol, # tolerance for the adaptative step
        "workers":workers # threads handed to each transform
    }

    # verify that the first sampling point is not before t_init
    if t == t_samples[0]:
        psi_list += [to_host(psi)]
        lambda_list += [lambdas]
        count_t += 1
    elif t > t_samples[0]:
        raise ValueError("First sampling point before initial simulation time")

    # The bounds on the step live here rather than inside the loop so that a progress bar cannot
    # change the numerics: the two branches this loop used to be written as disagreed on them, and a
    # run with verbose=True took a different sequence of steps from the same run without it.
    dtmax = kwargs.get("dtmax", 0.1)
    dtmin = kwargs.get("dtmin", 1e-6)

    # The bar counts simulated time rather than steps, so its rate reads as "time units per
    # second" -- the number that actually predicts how long a run has left. It is transient and
    # sits on the second line, since the caller keeps a bar over runs on the first one.
    #
    # The loop below stops at whichever comes first, t_final or the last sampling time, and the
    # samples usually end well before t_final. Sizing the bar by t_final alone would leave it short
    # on a perfectly ordinary run, and tqdm.notebook reads a manual-mode bar that closes below its
    # total as a crash: it paints it red and then declines to hide it, so 'leave=False' stops
    # applying and every run leaves a red bar behind. Ascending t_samples is assumed here, as it
    # already is by 'dt_max' above and by 'count_t' only ever walking forward.
    total_t = max(min(float(t_samples[-1]), t_final) - t_init, 0.0)
    pbar = bar(
        total=total_t,
        desc="simulated time",
        unit="t",
        verbose=verbose,
        leave=False,
        position=1,
    )
    pbar.bar_format = (
        "{l_bar}{bar}| {n:.3g}/{total:.3g} {unit} [{elapsed}<{remaining}, {rate_fmt}]"
    )

    # propagating psi and storing at each time-step reaching the next t_sampling point
    while t < t_final and count_t < len(t_samples):
        remaining = float(t_samples[count_t]) - t
        if remaining > 0:
            dt = min(dt, remaining)

        dt_used, dt, phi, sigmas, lambdas, phi_f = adaptative_step(
            phi, sigmas, lambdas, t, dt, consts, phi_f=phi_f
        )
        t += dt_used
        dt = min(dt, dt_max)  # making sure not to skip sampling times
        dt = max(min(dt, dtmax), dtmin)  # bounding the step time to reasonable values

        # A rejected step lands short of the sample, so this only fires once the step actually
        # reached it; the tolerance absorbs the rounding of t += dt_used.
        if t >= float(t_samples[count_t]) - 1e-12 * max(1.0, abs(t)):
            psi_list += [to_host(phi / np.prod(lambdas) ** 0.5 * chirp(lambdas))]
            lambda_list += [lambdas]
            count_t += 1
        pbar.update(min(dt_used, total_t - pbar.n))

    # The sample-reached test fires within a tolerance of the sampling time, so the loop can stop a
    # whisker short of the horizon. That sliver still reads as 'n < total' to tqdm, which is enough
    # to colour a completed run red, so it is paid in before closing.
    pbar.update(max(total_t - pbar.n, 0.0))
    pbar.close()

    n_samples = len(psi_list)
    if n_samples != len(t_samples):
        print(
            f"Less time steps than required samples, padding the array with last psi, last proper sample is {n_samples}"
        )
        psi_list += [psi_list[-1]] * (len(t_samples) - n_samples)
        lambda_list += [lambdas] * (len(t_samples) - n_samples)

    return lambda_list, psi_list



def attach_frame(
    psi: xr.DataArray, names: list[str] | None = None
) -> xr.DataArray:
    """Rebuild the moving cartesian coordinates from the stored 'lambda' coordinate.

    The result deliberately carries both spatial dimensions on each coordinate even where one of
    them is redundant: 'create_cmesh' transposes the coordinate onto the full ('a1', 'a2') mesh
    before drawing, and a coordinate carrying only its own axis would raise there.

    Args:
        psi (xr.DataArray): A result from 'rSSFM.solve', carrying the 'lambda_*' and invariant
        'rho_*' coordinates.
        names (list[str], optional): The cartesian names to rebuild. Defaults to None, which takes
        every axis that has both a 'rho_*' and a 'lambda_*' coordinate.

    Returns:
        xr.DataArray: The same array with its cartesian coordinates materialized.

    Raises:
        ValueError: If the array carries no 'lambda_*' coordinates, i.e. it did not come from this
        solver or they were stripped on the way.
    """
    if names is None:
        names = [
            str(c)[len("rho_"):]
            for c in psi.coords
            if str(c).startswith("rho_") and f"lambda_{str(c)[len('rho_'):]}" in psi.coords
        ]
    if not names:
        raise ValueError(
            "No 'lambda_*' coordinates: the moving frame cannot be rebuilt. This array did not "
            "come from rSSFM.solve, or the coordinates were dropped."
        )

    # A zero-weighted sum over every invariant grid, which is what forces each rebuilt coordinate
    # to span the full spatial mesh rather than only its own axis.
    spatial = sum(0 * psi.coords[f"rho_{name}"] for name in names)
    # The recipe and the materialised coordinate must never both be present, or a reader would have
    # two answers for the same thing.
    psi = psi.copy()
    psi.attrs.pop("cart_factors", None)
    return psi.assign_coords(
        {
            name: (psi.coords[f"rho_{name}"] + spatial) * psi.coords[f"lambda_{name}"]
            for name in names
        }
    )


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
        verbose: bool = True,
        n_cores: int = 4,
        workers: int = 1,
        device: str | bool = "auto",
        checkpoint: str | None = None,
        **kwargs,
    ) -> xr.DataArray:
        """Solves the gross-Pitaevskii equation for each point in parameter space. see doc of 'propagate' for more infos.

        Args:
            t_init (float): Initial time of simulation
            t_final (float): End time of simulation
            t_samples (xr.DataArray): Sampling times for psi. t_sample can have multiple dimensions, but one of them must be 't'.
            dt0 (Union[float, xr.DataArray], optional): Initial time step. Can have multiple dimensions, but they must be a subset of the parameter space. Defaults to 1e-3.
            tol (Union[float, xr.DataArray], optional): Tolerance for adaptative method. Can have multiple dimensions, but they must be a subset of the parameter space
            As a rule, the tolerance should decrease for higher values of g. Defaults to 1e-6.
            checkpoint (str or Path, optional): A directory to keep each finished run in, so that
            a sweep killed partway through can be resumed by calling this again with the same
            argument -- the runs already there are loaded rather than recomputed. Worth passing
            for anything long on a GPU that also drives a display, where the driver's watchdog
            destroys the CUDA context without warning and nothing in this process can recover it.
            Costs one full sample stack per run on disk while the sweep is running, and is
            deleted again the moment the sweep completes -- a finished sweep holds every run in
            what it returns. Defaults to None, which keeps nothing.

        Returns:
            xr.DataArray: The value of Psi for each time sampling point at each point of the parameter space.
            It carries the invariant grid as 'rho_x', 'rho_y', ... and the frame as 'lambda_x',
            'lambda_y', ...; the cartesian coordinates are present only when 'store_frame' kept
            them, and 'attach_frame' puts them back on demand.
        """

        names = self.potential.coord_names[: self.n_dims]

        # --- where to run, decided before the runs are laid out: the analytic potentials built by
        # 'prepare_runs' have to be built against whichever array module the field will live on ---
        n_points = int(np.prod(self.potential.resolution))
        # A run holds of order ten live grids at once, plus cuFFT's own workspace.
        run_bytes = n_points * np.dtype(complex).itemsize * 12
        device, why = resolve_device(
            "auto" if device is None else device,
            n_points=n_points,
            n_bytes=run_bytes,
        )
        xp = get_xp(device)
        if verbose:
            print(f"rSSFM running on {device.upper()}: {why}")
            for line in device_warnings(device):
                print(line)

        psi, selections, list_args = self.prepare_runs(t_samples, dt0, tol, xp=xp)

        n_samples = len(t_samples.coords["t"].data)

        # The runs already on disk from an earlier attempt, if this sweep is checkpointing. The
        # signature is what stops a resume from mixing two calculations: point the same directory
        # at a sweep of a different shape or time span and it refuses rather than filling the
        # gaps with runs that answer a different question.
        ckpt = (
            None
            if checkpoint is None
            else RunCheckpoint(
                checkpoint,
                len(selections),
                {
                    "solver": type(self).__name__,
                    "dims": list(psi.dims),
                    "shape": list(psi.shape),
                    "t_init": float(t_init),
                    "t_final": float(t_final),
                    "t_samples": [float(v) for v in np.ravel(t_samples.data)],
                },
            )
        )
        done = set() if ckpt is None else ckpt.done()
        if done and verbose:
            print(
                f"  resuming from {ckpt.path}: {len(done)} of {len(selections)} runs "
                "already done"
            )

        if device == "gpu" and parallel and n_cores > 1:
            # One card gains nothing from several processes queueing on it, and a single grid
            # already saturates it, so the sweep is simply looped.
            if verbose:
                print(
                    f"  ignoring n_cores={n_cores}: the sweep is run as a serial loop on the "
                    "device, which already reaches full throughput on one run at a time."
                )
            parallel = False


        frame_bytes = psi.size * np.dtype(float).itemsize * self.n_dims
        store_frame = kwargs.pop("store_frame", "auto")
        if store_frame == "auto":
            budget = kwargs.pop("frame_budget", 256 * 1024**2)
            store_frame = frame_bytes <= budget
        else:
            kwargs.pop("frame_budget", None)

        if store_frame:
            if "t" not in psi.coords[names[0]].dims:
                for name in names:
                    psi.coords[name] = psi.coords[name].expand_dims(
                        dim={"t": t_samples.coords["t"]}
                    )
            for name in names:
                psi.coords[name] = psi.coords[name].transpose("t", ...).copy()
        else:
            # The cartesian coords cannot simply be left as they are: 'initialize_psi' built them
            # from the initial frame, so keeping them would leave the array labelled with
            # coordinates that are silently wrong everywhere except t = 0.
            psi = psi.drop_vars([name for name in names if name in psi.coords])
            # Left instead is the recipe for rebuilding them, which the plotting helpers read so
            # that a run saved this way draws without the coords ever being materialised. Written
            # as JSON because a netCDF attribute cannot hold a mapping.
            psi.attrs["cart_factors"] = json.dumps(
                {name: [f"rho_{name}", f"lambda_{name}"] for name in names}
            )

        # One row per sample per run, holding that run's frame. Tiny, and always present, so that
        # a result can be re-expanded later whether or not the coords were materialized.
        lam_dims = ("t", *[dim for dim in psi.dims if dim not in ("t", *self.spatial_dims)])
        lam_shape = tuple(len(psi.coords[d]) if d in psi.coords else psi.sizes[d] for d in lam_dims)
        lam_array = np.ones((*lam_shape, self.n_dims))

        # The invariant coordinates as 1-D broadcastable arrays where the box allows it, which for
        # this solver is always: it refuses a non-diagonal unit cell up front. They are built dense
        # by 'Potential', so each of x and y is a 512^2 array holding 512 distinct values, and the
        # frame scales them, squares them and reduces against them on every step. The rank-1 form
        # broadcasts identically wherever it is multiplied, and the reductions that would stride
        # badly against it are written per-axis in 'dsigma' and 'compute_energy'.
        #
        # 'aliasing' stays dense: it is a radial cut-off in k, which is the one thing here that is
        # genuinely not separable.
        rho_1d = self.potential.coords_1d()
        if rho_1d is None:  # skewed box, which this solver rejects, but do not assume it
            rho_1d = [coord.data for coord in self.potential.coords]
            ks_1d = list(self.ks)
        else:
            ks_1d = []
            for i in range(self.n_dims):
                line = fftfreq(self.n_a[i], self.da[i]) * 2 * np.pi
                shape = [1] * self.n_dims
                shape[i] = line.size
                ks_1d.append(line.reshape(shape))

        # Moved across once rather than per step. On the CPU 'xp.asarray' is a no-op on arrays that
        # are already numpy, so this costs nothing there.
        grids = {
            "aliasing": xp.asarray(self.aliasing),
            "rho": [xp.asarray(r) for r in rho_1d],
            "ks": [xp.asarray(k) for k in ks_1d],
        }

        def x(t_samples_sel, psi0_sel, V_sel, dt0_sel, g_sel, tol_sel):
            return propagate(
                t_init,
                t_final,
                grids["aliasing"],
                grids["rho"],
                grids["ks"],
                self.da,
                t_samples_sel,
                xp.asarray(psi0_sel),
                V_sel,
                dt0_sel,
                g_sel,
                tol_sel,
                workers=workers,
                # A worker process must never open a bar: several of them writing carriage
                # returns to the one stderr produces nothing readable.
                verbose=verbose and not parallel,
                **kwargs,
            )

        def store(indexes, lambda_list, psi_list):
            """Write one run into psi, together with the frame it was computed in."""
            for j in range(n_samples):
                slic = [j, *indexes]
                lam_array[*slic] = lambda_list[j]
                if store_frame:
                    for i, name in enumerate(names):
                        psi.coords[name][*slic] = (
                            psi.coords[f"rho_{name}"] * lambda_list[j][i]
                        )
                psi[*slic] = psi_list[j]

        # scipy dispatches its transforms through uarray, so registering cuFFT as the backend for
        # the duration of the run sends every fftn/ifftn in this module to the device without any
        # of them naming a module. On the CPU this is a null context and the MKL global backend
        # registered by 'BECs.spectral' stays in force.
        backend_ctx = fft_backend(device)

        if not parallel:
            with backend_ctx:
                for i, indexes in enumerate(
                    bar(selections, desc="Propagating runs", unit="run", verbose=verbose)
                ):
                    if i in done:
                        store(indexes, *ckpt.load(i))
                        continue
                    lambda_list, psi_list = guarded_run(
                        x, list_args[i], i, len(selections), ckpt
                    )
                    if ckpt is not None:
                        ckpt.save(i, lambda_list, psi_list)
                    store(indexes, lambda_list, psi_list)
        else:
            # Only the runs still missing are submitted, so 'parallel_imap' indexes that shorter
            # list and 'pending' maps its answers back onto the sweep.
            pending = [i for i in range(len(selections)) if i not in done]
            for i in done:
                store(selections[i], *ckpt.load(i))

            # Stored as each run lands rather than collected first. 
            for i, (lambda_list, psi_list) in parallel_imap(
                x,
                [list_args[i] for i in pending],
                n_jobs=n_cores,
                desc="Propagating runs",
                unit="run",
                verbose=verbose,
                longest_first=True,
            ):
                if ckpt is not None:
                    ckpt.save(pending[i], lambda_list, psi_list)
                store(selections[pending[i]], lambda_list, psi_list)

        # One coordinate per axis rather than a single one carrying an 'axis' dimension: xarray
        # only accepts coordinates whose dimensions the array itself has, and psi has no such axis.
        # The names mirror the invariant 'rho_*' ones they pair with.
        psi = psi.assign_coords(
            {
                f"lambda_{name}": (lam_dims, lam_array[..., i])
                for i, name in enumerate(names)
            }
        )
        # The sweep is complete and every run of it is in the array below, so the copies on disk
        # have nothing left to protect against.
        if ckpt is not None:
            ckpt.discard()
            if verbose:
                print(f"  sweep complete: removed the checkpoint in {ckpt.path}")

        return psi.squeeze()

