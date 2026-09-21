# %%
from collections.abc import Callable
from copy import deepcopy

import numpy as np
import xarray as xr
from bloch_schrodinger.fdsolver import check_name as _check_name
from bloch_schrodinger.potential import Potential
from bloch_schrodinger.progress import bar, parallel_imap

from BECs.backend import fft_backend, get_xp, resolve_device, to_host
from BECs.groundstate import distance, fs_residual, selected_value, subselect
from BECs.potentialT import AnalyticPotential, PotentialT
from BECs.spectral import (
    KineticStep,
    PotentialPhase,
    SpectralSolver,
    density,
    phase,
    register_coord,
)

# Yoshida splitting coefficient
cbrt2 = 2**(1/3)
w1    = 1/(2 - cbrt2)
w0    = -cbrt2/(2 - cbrt2)


def check_name(name: str, n_dims: int = 2):
    """Check whether the name is a valid one, and raises an error if not.

    Args:
        name (str): The name to check
        n_dims (int, optional): The dimensionality of the solver, which sets the list of forbidden
        names. Defaults to 2, the dimensionality this module was written for.

    Raises:
        ValueError: If the name is forbidden
    """
    _check_name(name, n_dims)


def losses(
    coords: list[xr.DataArray], width: float, gamma: float
) -> xr.DataArray:
    """Creat a lose term for an absorbing boundaries. This absorbing layer has a sinuosidal shape for smooth absorption.

    Args:
        coords (list[xr.DataArray]): The cartesian coordinates of the grid, one array per dimension.
        width (float): The width of the lossy layer
        gamma (float): The amplitude of the loss layer

    Returns:
        xr.DataArray: The modified potential
    """
    width = 2 * width

    # Distance to the centre of the box along each axis, normalized to the layer width
    dists = []
    for coord in coords:
        rg = (coord.max() - coord.min()) / 2
        mn = (coord.max() + coord.min()) / 2
        dists += [abs((coord - mn) / rg) / width]

    # The layer follows the box, so a point is as deep in it as its deepest axis
    losses = dists[0]
    for dist in dists[1:]:
        losses = xr.where(losses < dist, dist, losses)

    losses = xr.where(losses < (1 - width) / width, 0, losses - (1 - width) / width)
    losses = -(xr.ufuncs.cos(np.pi * losses) - 1) / 2

    return 1j * losses * gamma


def _on_device(Vt: Callable, xp) -> Callable:
    """Wrap a potential-of-time so its grid comes back on 'xp', preserving its 'static' flag.

    A tabulated potential is a stored array and is built on the host, so on the GPU every step
    would otherwise try to multiply a device field by a host grid -- which cupy refuses rather than
    transferring behind the caller's back. A static one is converted once and kept, since it is the
    same array at every t.

    Args:
        Vt (Callable): The potential as a function of time, from 'PotentialT.make_Vt'.
        xp (module): The array module to deliver on.

    Returns:
        Callable: The same function, returning arrays on 'xp'.
    """
    if xp is np:
        return Vt

    static = getattr(Vt, "static", False)
    cache = {}

    def moved(t: float):
        if static:
            if "V" not in cache:
                V = Vt(t)
                cache["V"] = xp.asarray(V.data if hasattr(V, "dims") else V)
            return cache["V"]
        V = Vt(t)
        return xp.asarray(V.data if hasattr(V, "dims") else V)

    moved.static = static
    return moved


def linear_step(
    psi: np.ndarray,
    dt: complex,
    kin: KineticStep,
) -> np.ndarray:
    """Linear propagation of the vector psi for a step dt by multiplication in Fourier space.

    Args:
        psi (np.ndarray): The vector to propagate.
        dt (float): The time step.
        kin (KineticStep): The grid's kinetic propagator, which owns the reciprocal space, the
        anti-aliasing mask and the cache of propagators already built.

    Returns:
        np.ndarray: Propagated vector.
    """
    return kin(psi, dt)


def potential_step(
    psi: np.ndarray,
    dt: complex,
    V: np.ndarray | xr.DataArray,
):
    """Phase rotation of the vector psi due to potential for a step dt by multiplication in real space.

    Args:
        psi (np.ndarray): The vector to propagate.
        dt (float): The time step.
        V (Union[np.ndarray,xr.DataArray]): Potential landscape.

    Returns:
        np.ndarray: Propagated vector.
    """
    return phase(dt, V) * psi


def nl_step(
    psi: np.ndarray,
    dt: complex,
    g: float,
):
    """Non-linear propagation of the vector psi for a step dt by multiplication in real space.

    Args:
        psi (np.ndarray): The vector to propagate.
        dt (float): The time step.
        g (float): Non-linear coefficient.

    Returns:
        np.ndarray: Propagated vector.
    """
    return phase(dt, g * density(psi)) * psi


def strang_step(
    psi: np.ndarray,
    kin: KineticStep,
    V: PotentialPhase,
    t: float,
    dt: complex,
    g: float,
) -> np.ndarray:
    """Propagate psi for a full step dt using a symmetric strang splitting

    Args:
        psi (np.ndarray): The vector to propagate.
        kin (KineticStep): The grid's kinetic propagator.
        V (PotentialPhase): The grid's potential phase rotation, which owns the potential and the
        cache of phases already built for a static one.
        t (float): time t for potential selection.
        dt (float): time step.
        g (float): Non-linear coefficient.

    Returns:
        np.ndarray: Propagated vector.
    """

    # The same half-step phase rotation is applied before and after the non-linear step, so the
    # exponential -- which costs about as much as a transform -- is built once and reused. For a
    # potential that does not move it is not even built once: 'PotentialPhase' keeps it across steps.
    phase = V(t + dt.real / 2, dt)

    psi_1 = linear_step(psi, dt / 2, kin)
    psi_2 = phase * psi_1
    psi_3 = nl_step(psi_2, dt, g)
    psi_4 = phase * psi_3
    psi_5 = linear_step(psi_4, dt / 2, kin)
    return psi_5

def yoshida_step(
    psi: np.ndarray,
    kin: KineticStep,
    V: PotentialPhase,
    t: float,
    dt: float,
    g: float,
) -> np.ndarray:
    """Propagate psi for a full step dt using a fourth order Yoshida step
    Args:
        psi (np.ndarray): The vector to propagate.
        kin (KineticStep): The grid's kinetic propagator.
        V (Union[np.ndarray,xr.DataArray]): Potential landscape.
        t (float): time t for potential selection.
        dt (float): time step.
        g (float): Non-linear coefficient.

    Returns:
        np.ndarray: Propagated vector.
    """

    psi1 = strang_step(psi, kin, V, t, dt*w1, g)
    psi2 = strang_step(psi1, kin, V, t + dt*w1, dt*w0, g)
    psi3 = strang_step(psi2, kin, V, t + dt*w1 + dt*w0, dt*w1, g)
    return psi3


def adaptative_step(
    psi: np.ndarray,
    kin: KineticStep,
    V: PotentialPhase,
    t: float,
    dt: float,
    g: float,
    tol: float,
    imagt: Callable,
    psi_full: np.ndarray | None = None,
    order: int = 2,
) -> tuple[float, float, np.ndarray]:
    """Propagate psi for a full step dt, using a recursive adaptative step-doubling method.
    This function propagate psi for dt and for 2*dt/2, then compares the results. If its above a certain tolerance,
    the function calls itself again with a halved time step.

    Args:
        psi (np.ndarray): The vector to propagate.
        kin (KineticStep): The grid's kinetic propagator.
        V (PotentialPhase): The grid's potential phase rotation.
        dt (float): time step.
        g (float): Non-linear coefficient.
        tol (float): The tolerance for step doubling
        imagt (Callable): A function of time t such that dt(t) = dt * (1 + 1j * imagt(t)).
        psi_full (np.ndarray, optional): The single step of dt, when the caller has already computed
        it. A rejected step of 2.dt hands down the half step it took, which is exactly this call's
        full step, saving a third of the work of every rejection. Defaults to None.
        order (int, optional): 2 for the Strang splitting, 4 for the Yoshida composition. The
        controller's exponent follows from it, so the two cannot drift apart. Defaults to 2.

    Returns:
        tuple[float, float, np.ndarray]: The time step length used, the optimal next time step length and the propagated vector.
    """
    step = strang_step if order == 2 else yoshida_step

    # The step-doubling difference scales as dt^(order+1), and 'fs_residual' is linear in that
    # difference -- unlike 'distance', which is its square and would call for half this exponent.
    # Tying it to 'order' here is what keeps a fourth-order stepper from being driven at a
    # second-order method's exponent, which overshoots every step and then has to reject.
    exponent = 1 / (order + 1)

    dt_i = dt * (1 + 1j * imagt(t))

    if psi_full is None:
        psi_full = step(psi, kin, V, t, dt_i, g)
    psi_half = step(psi, kin, V, t, dt_i / 2, g)
    # The second half step starts at t + dt/2, not at t. Evaluating the potential at t here costs the
    # splitting an order of accuracy whenever V depends on time, and understates the error estimate
    # below with it, so the controller then takes steps far larger than the tolerance asked for.
    psi_double = step(psi_half, kin, V, t + dt_i.real / 2, dt_i / 2, g)

    # 'fs_residual' rather than 'distance': same Fubini-Study geometry and the same invariance under
    # a global phase, but formed as an orthogonal residual instead of 1 - |<a,b>|^2, which cancels
    # to nothing once the two states agree to an ulp. The old form returned a flat 0.0 below a
    # relative error of about 1e-8 -- a floor a fourth-order stepper reaches at an ordinary step,
    # leaving the controller to grow the step on no information.
    err = fs_residual(psi_double, psi_full)
    if err > tol:  # If the error is superior, try again with a time step dt/2
        return adaptative_step(
            psi, kin, V, t, dt / 2, g, tol, imagt, psi_full=psi_half, order=order
        )
    else:  # else return the results and compute a new time-step
        if err == 0:
            s = 5
        else:
            # The cap is 5 rather than 10: with the exponent matched to the method the prediction no
            # longer overshoots, so it does not need the head-room it used to.
            s = max(min(0.6 * (tol / err) ** exponent, 5), 0.1)
        return dt, s * dt, psi_double


def propagate(
    t_init: float,
    t_final: float,
    kin: KineticStep,
    t_samples: xr.DataArray,
    psi: np.ndarray,
    V: Callable | PotentialPhase,
    dt: float,
    g: float,
    tol: float,
    imagt: Callable,
    order: int = 2,
    verbose: bool = False,
    **kwargs,
) -> tuple[list[float], list[np.ndarray]]:
    """The main simualtion function of the submodule. Solves the Gross-Pitaevskii equation for the initial vector psi
    between t_init and t_final using an adaptative split-step Fourier method.

    Args:
        t_init (float): Initial time of simulation.
        t_final (float): Time when to stop the simulation.
        kin (KineticStep): The grid's kinetic propagator, carrying reciprocal space and the
        anti-aliasing mask.
        t_samples (xr.DataArray): List of sampling time at which to keep psi.
        psi (np.ndarray): Initial vector.
        V (xr.DataArray): Potential landscape.
        dt (float): Initial time step.
        g (float): Interaction strength term.
        tol (float): Tolerance for the adaptative time step method.
        imagt (Callable): A function of time t such that dt(t) = dt * (1 + 1j * imagt(t)).
        order (int, optional): The splitting order, 2 (Strang) or 4 (Yoshida). Defaults to 2.
        verbose (bool, optional): Wheter to plot a progress bar, useful for knowing where blow-up might happen. Defaults to False.

    Raises:
        ValueError: Raises an error if the initial sampling point is before t_init.

    Returns:
        tuple[list[np.ndarray]]: The vector psi sampled at the times specified by t_samples.
    """
    # One phase cache per run, for the same reason the kinetic one is per run: it is keyed on the
    # time step alone and so belongs to this grid and this potential. Accepting a bare callable too
    # keeps every existing caller working.
    if not isinstance(V, PotentialPhase):
        V = PotentialPhase(V)

    t = t_init
    count_t = 0  # tracking what is the next sampling time.

    psi_list = []
    dt_max = t_samples.data[1] - t_samples.data[0]

    # verify that the first sampling point is not before t_init
    if t == t_samples[0]:
        psi_list += [to_host(psi)]
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
        # Land exactly on the next sampling time rather than storing whatever state the stepper
        # happens to reach once it has passed it. Overshooting puts an error of order dt into the
        # time at which each frame is recorded, and an error in t is a first-order error in psi --
        # it caps the accuracy of the whole run at first order no matter how good the splitting is,
        # and it moves around with the step sequence, so tightening 'tol' does not reliably improve
        # the result. Clamping here costs nothing: the step was already bounded by the sample
        # spacing, so this only aligns it.
        remaining = float(t_samples[count_t]) - t
        if remaining > 0:
            dt = min(dt, remaining)

        dt_used, dt, psi = adaptative_step(psi, kin, V, t, dt, g, tol, imagt, order=order)
        t += dt_used
        dt = min(dt, dt_max)  # making sure not to skip sampling times
        dt = max(min(dt, dtmax), dtmin)  # bounding the step time to reasonable values

        # A rejected step lands short of the sample, so this only fires once the step actually
        # reached it; the tolerance absorbs the rounding of t += dt_used.
        if t >= float(t_samples[count_t]) - 1e-12 * max(1.0, abs(t)):
            # Brought back as each sample is taken: the caller stores them in a host array
            # either way, and keeping the whole time series on the device fills it for nothing.
            psi_list += [to_host(psi)]
            count_t += 1
        # Clamped to what is left rather than skipped near the end, so the bar reaches 100%
        # without overshooting. This only reads dt_used, it never feeds back into dt.
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
        psi_list += [to_host(psi)] * (len(t_samples) - n_samples)

    return psi_list


class SSFM(SpectralSolver):
    def __init__(
        self,
        potential: Potential | PotentialT | AnalyticPotential,
        psi0: xr.DataArray,
        g: float | xr.DataArray,
    ):
        """Initialize a solver instance for the Gross-Pitaevskii equation. This solver handles only scalar equations on rectangular grids.

        Args:
            potential (Union[Potential,PotentialT,AnalyticPotential]): The potential landscape, must be describing an
            axis-aligned rectangular grid. The solver will iterate over each additional dimensions (not a1, a2, ...).
            psi0 (xr.DataArray): Initial vector, must have shape and spatial dimensions (a1, a2, ...) consistent with the potential.
            g (Union[float, xr.DataArray]): Interaction strength term. Can be passed as an array over which to iterate.

        Raises:
            ValueError: If the potential and initial vector given do not have the proper dimensions.
            ValueError: If the potential grid is not rectangular and axis-aligned.
        """
        self.analytic = False # Whether the potential is in an analytic form
        if isinstance(potential, AnalyticPotential):
            self.analytic = True
            self.potential = potential
        elif isinstance(potential, PotentialT):
            self.potential = deepcopy(potential)  # copied to add losses without modifying the original object
        else:
            self.potential = PotentialT.fromPotential(potential)

        if not self.analytic:
            # Fold whatever sits in V into the parameter dictionnary before it is read below
            self.potential.update_V0()

        if "band" in psi0.dims: # A check to avoid conflicts with the initialize_eigve from the fdsolver class
            self.psi0 = psi0.rename({"band":"band1"})
            self.is_band_dim = True
        else:
            self.psi0 = psi0
            self.is_band_dim = False

        # The grid, the reciprocal space and the parameter dimensions carried by V
        super().__init__(self.potential, g)

        # A time-dependent potential also carries the parameters of its time functions
        for dim, coord in self.potential.param_coords.items():
            register_coord(self.allcoords, dim, "potential", coord)

        missing = [dim for dim in self.spatial_dims if dim not in psi0.dims]
        if missing:
            raise ValueError(f"psi0 is missing the dimension(s) {missing} of the potential")

        # Adding all additional dimensions of psi0 to the coordinates dictionnary.
        spatial_names = self.spatial_dims + self.potential.coord_names[: self.n_dims]
        # A dimension psi0 shares with the potential or with g is deliberately left out of this
        # registration: psi0 is allowed to live on a coarser parameter grid than the run, which is
        # what the 'nearest' lookup in 'prepare_runs' is for, and 'register_coord' would reject the
        # coordinates as mismatched. Such a dimension reaches 'psi0_sel' through 'total_sel' there
        # instead.
        coords_psi0 = {
            dim: ["psi0", self.psi0.coords[dim]]
            for dim in self.psi0.dims
            if dim not in spatial_names and dim not in self.allcoords
        }
        self.allcoords.update(coords_psi0)

        self.imagt = lambda t: 0 # A function to add a imaginary part to the time steps dt. makes it so dt(t) = dt * (1 + 1j * imagt(t))

    def initialize_eigva(self):
        eigva = super().initialize_eigva(1)
        if self.is_band_dim:
            eigva = eigva.rename({"band1":"band"})
        return eigva

    def initialize_psi(self):
        psi = (
            super()
            .initialize_eigve(1, False)
            .transpose(..., *self.spatial_dims)
            .rename("psi")
        )
        psi = psi.squeeze(["band", "field"], drop=True)
        if self.is_band_dim:
            psi = psi.rename({"band1":"band"})

        return psi
    
    def imaginary_time(self, func:Callable):
        """Set the imaginary time function 'f', such that the time step dt(t) = dt * (1 + 1j * f(t))

        Args:
            func (Callable): _description_
        """
        self.imagt = func
        

    def add_losses(self, width: float, amp: float):
        """Add sinusoidal losses to the potential. see 'losses' for more doc.

        Args:
            width (float): width of the absorbing layer.
            amp (float): height of the absorbing layer.
        """
        if self.analytic:
            # An AnalyticPotential never reads V, so the layer is registered as a term instead.
            # It is built on the coordinates it is called with, so it follows the grid it is
            # evaluated on rather than being pinned to the initial one.
            def loss_func(t, *coords):
                return losses(list(coords), width, amp)

            self.potential.add_function("loss", loss_func)
            self.potential.add_term("loss")
        else:
            loss = losses(self.potential.coords, width, amp)
            self.potential.V = self.potential.V + loss
            self.potential.update_V0()

    def prepare_runs(
        self,
        t_samples: xr.DataArray,
        dt0: float | xr.DataArray,
        tol: float | xr.DataArray,
        xp=np,
    ) -> tuple[xr.DataArray, list[tuple[int]], list[tuple]]:
        """Lay out the storage array and the argument list of every run to perform, one per point of
        the parameter space. Shared by this solver and the rescaling one, which only differ in how
        they propagate and store the results.

        Args:
            t_samples (xr.DataArray): Sampling times for psi, with at least a dimension 't'.
            dt0 (Union[float, xr.DataArray]): Initial time step, possibly parameter dependant.
            tol (Union[float, xr.DataArray]): Adaptative method tolerance, possibly parameter dependant.
            xp (module, optional): The array module the analytic potentials should be built
            against. A potential evaluated on device coordinates has to be built with cupy, or its
            shape-fixing zero grid stays on the host and the first addition raises rather than
            transferring. Ignored for the tabulated 'PotentialT' flavour, which is a stored grid
            and is moved across by the caller. Defaults to numpy.

        Returns:
            tuple[xr.DataArray, list[tuple[int]], list[tuple]]: The empty psi array to fill, the index
            tuples covering the parameter space, and the arguments of 'propagate' for each of them.
        """

        # Adding eventual time sampling dimensions to the context. A dimension already known --
        # the evolution time of a time-of-flight sequence, which parameterizes the potential and
        # sets the sampling times along with it -- gains the 't' category rather than being left
        # alone, so that the selection below actually reduces t_samples to the 't' axis.
        for dim in t_samples.dims:
            check_name(dim, self.n_dims)
            if dim != "t":
                register_coord(self.allcoords, dim, "t", t_samples.coords[dim])

        # Create the empty DataArray that will store the propagated states
        psi = self.initialize_psi()
        if "t" not in psi.dims:
            psi = psi.expand_dims(dim={"t": t_samples.coords["t"]})
        psi = psi.transpose("t", ...).copy()

        # We create a list of tuples that select a single value for each of the parameter dimensions
        selections = self.selections()

        list_args = []
        for indexes in selections:
            # --- Constructing the inputs for 'propagate' ---
            ## select the potential. Both flavours return a function of the time alone, an
            ## AnalyticPotential defaulting to its own grid when called without coordinates.
            potential_sel = subselect(indexes, "potential", self.allcoords)

            if not self.analytic:
                # A tabulated potential is a stored grid, so it comes back on the host whatever
                # the field is running on and has to be carried across. The wrapper keeps the
                # 'static' flag, which is what lets 'PotentialPhase' cache the exponential.
                Vt = _on_device(self.potential.make_Vt(potential_sel), xp)
            else:
                Vt = self.potential.make_V(potential_sel, xp=xp)

            ## select t_samples
            samples_sel = subselect(indexes, "t", self.allcoords)
            t_samples_selected = t_samples.sel(samples_sel)

            ## Select the interaction strength
            g_sel = subselect(indexes, "g", self.allcoords)
            g_selected = selected_value(self.g, g_sel)

            ## Aggregate the selection, to select tol and dt0 if needed
            total_sel = {**potential_sel, **g_sel}

            ## select psi0
            psi0_sel = subselect(indexes, "psi0", self.allcoords)
            psi0_sel.update(
                {dim: total_sel[dim] for dim in total_sel if dim in self.psi0.dims}
            )
            psi0_selected = self.psi0.sel(psi0_sel, method='nearest')

            if isinstance(tol, xr.DataArray):
                tol_sel = {dim: total_sel[dim] for dim in total_sel if dim in tol.dims}
                tol_selected = tol.sel(tol_sel, method="nearest")
            else:
                tol_selected = tol

            if isinstance(dt0, xr.DataArray):
                dt0_sel = {dim: total_sel[dim] for dim in total_sel if dim in dt0.dims}
                dt0_selected = dt0.sel(dt0_sel, method="nearest")
            else:
                dt0_selected = dt0

            # psi0 is handed over with its axes in the grid order, so that the transforms act on
            # the right ones.
            list_args += [
                (
                    t_samples_selected,
                    psi0_selected.transpose(*self.spatial_dims).data,
                    Vt,
                    dt0_selected,
                    g_selected,
                    tol_selected,
                )
            ]

        return psi, selections, list_args

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
        order: int = 2,
        device: str | bool = "auto",
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
            NOTE that the meaning of this number changed: the controller now measures the error with
            'fs_residual', which is linear in the error, where it used to use 'distance', which is
            its square. A tolerance that used to read 1e-6 therefore corresponds to about 1e-3 here,
            and a value carried over unchanged from older scripts will ask for a far tighter run
            than it did before. The upside is that 'tol' is now directly a relative error per step,
            so 1e-6 means what it looks like.
            parallel (bool, optional): Whether to use the parallel solver, this involve some overhead, so do not use it for too small parameter spaces. default to False.
            verbose (bool, optional): Whether to plot progress bars: one over the runs, and, when
            'parallel' is not set, one over the simulated time of the current run, useful for knowing
            where blow-up might happen. Defaults to True.
            n_cores (int, optional): The number of cores to use for the parallelized solver.
            workers (int, optional): Threads given to each Fourier transform, -1 for every core. Worth
            raising for grids of 512^2 and upwards, where it buys about a third of the transform time;
            below that the transforms are too small to gain anything. Leave it at 1 when 'parallel' is
            set, since the parameter space is then already spread over the cores. Defaults to 1.
            order (int, optional): The order of the splitting: 2 for Strang, 4 for the Yoshida
            composition. Yoshida costs three Strang sub-steps per step but converges two orders
            faster, so it is cheaper for anything but coarse work -- measured on a harmonic trap it
            reaches a relative error of 1e-4 for 2.1x less work than Strang, 1e-6 for 6.7x less and
            1e-8 for 21x less, with the two breaking even around 2e-3. Both are exactly unitary.
            Defaults to 2.
            device (str or bool, optional): "cpu", "gpu", or "auto" to let 'BECs.backend' decide.
            "auto" picks the GPU only when there is a working one and the grid is above the measured
            crossover of about 128^2, below which kernel launches cost more than they save. "gpu"
            raises rather than falling back. With one card the sweep runs as a serial loop, since a
            single grid already saturates it. Defaults to "auto".

        Returns:
            xr.DataArray: The value of Psi for each time sampling point at each point of the parameter space.
        """
        n_points = int(np.prod([self.potential.V.sizes[d] for d in self.spatial_dims]))
        device, why = resolve_device(
            device,
            n_points=n_points,
            n_bytes=n_points * np.dtype(complex).itemsize * 10,
        )
        xp = get_xp(device)
        if verbose:
            print(f"SSFM running on {device.upper()}: {why}")
        if device == "gpu" and parallel and n_cores > 1:
            if verbose:
                print(
                    f"  ignoring n_cores={n_cores}: one device is already saturated by a single "
                    "grid, so the sweep is run as a serial loop on it."
                )
            parallel = False

        psi, selections, list_args = self.prepare_runs(t_samples, dt0, tol, xp=xp)
        n_samples = len(t_samples.coords["t"].data)

        def x(t_samples_sel, psi0_sel, V_sel, dt0_sel, g_sel, tol_sel):
            # One propagator cache per run: it is keyed on the time step alone, so it must not
            # outlive the grid it was built for
            return propagate(
                t_init,
                t_final,
                self.kinetic_step(workers, xp=xp),
                t_samples_sel,
                xp.asarray(psi0_sel),
                V_sel,
                dt0_sel,
                g_sel,
                tol_sel,
                imagt = self.imagt,
                order = order,
                # A worker process must never open a bar: several of them writing carriage
                # returns to the one stderr produces nothing readable.
                verbose=verbose and not parallel,
                **kwargs,
            )

        backend_ctx = fft_backend(device)

        if not parallel:
            with backend_ctx:
                for i, indexes in enumerate(
                    bar(selections, desc="Propagating runs", unit="run", verbose=verbose)
                ):
                    psi_list = x(*list_args[i])
                    for j in range(n_samples):
                        slic = [j, *indexes]
                        psi[*slic] = psi_list[j]
        else:
            # Stored as each run lands rather than collected first: a run's psi_list is one full
            # grid per sampling time, so holding the whole sweep costs that times the parameter
            # space on top of 'psi'. See 'parallel_imap' for the 'longest_first' argument.
            for i, psi_list in parallel_imap(
                x,
                list_args,
                n_jobs=n_cores,
                desc="Propagating runs",
                unit="run",
                verbose=verbose,
                longest_first=True,
            ):
                indexes = selections[i]
                for j in range(n_samples):
                    slic = [j, *indexes]
                    psi[*slic] = psi_list[j]

        return psi.squeeze()
