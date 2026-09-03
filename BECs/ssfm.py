# %%
from collections.abc import Callable
from copy import deepcopy

import numpy as np
import xarray as xr
from bloch_schrodinger.fdsolver import check_name as _check_name
from bloch_schrodinger.potential import Potential

from BECs.groundstate import distance, subselect
from BECs.potentialT import AnalyticPotential, PotentialT
from BECs.progress import bar, parallel_map
from BECs.spectral import KineticStep, SpectralSolver, density

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
    return np.exp(1j * dt * (V)) * psi


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
    return np.exp(1j * dt * (g * density(psi))) * psi


def strang_step(
    psi: np.ndarray,
    kin: KineticStep,
    V: Callable,
    t: float,
    dt: complex,
    g: float,
) -> np.ndarray:
    """Propagate psi for a full step dt using a symmetric strang splitting

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

    # np.asarray because a PotentialT hands back a DataArray, and every multiply below would
    # otherwise go through xarray's dispatch machinery for no benefit
    V_t = np.asarray(V(t + dt.real / 2))

    # The same half-step phase rotation is applied before and after the non-linear step, so the
    # exponential -- which costs about as much as a transform -- is built once and reused
    phase = np.exp(1j * (dt / 2) * V_t)

    psi_1 = linear_step(psi, dt / 2, kin)
    psi_2 = phase * psi_1
    psi_3 = nl_step(psi_2, dt, g)
    psi_4 = phase * psi_3
    psi_5 = linear_step(psi_4, dt / 2, kin)
    return psi_5

def yoshida_step(
    psi: np.ndarray,
    kin: KineticStep,
    V: Callable,
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
    V: Callable,
    t: float,
    dt: float,
    g: float,
    tol: float,
    imagt: Callable,
    psi_full: np.ndarray | None = None,
) -> tuple[float, float, np.ndarray]:
    """Propagate psi for a full step dt, using a recursive adaptative step-doubling method.
    This function propagate psi for dt and for 2*dt/2, then compares the results. If its above a certain tolerance,
    the function calls itself again with a halved time step.

    Args:
        psi (np.ndarray): The vector to propagate.
        kin (KineticStep): The grid's kinetic propagator.
        V (Union[np.ndarray,xr.DataArray]): The potential landscape, must have a dimension 't'.
        dt (float): time step.
        g (float): Non-linear coefficient.
        tol (float): The tolerance for step doubling
        imagt (Callable): A function of time t such that dt(t) = dt * (1 + 1j * imagt(t)).
        psi_full (np.ndarray, optional): The single step of dt, when the caller has already computed
        it. A rejected step of 2.dt hands down the half step it took, which is exactly this call's
        full step, saving a third of the work of every rejection. Defaults to None.

    Returns:
        tuple[float, float, np.ndarray]: The time step length used, the optimal next time step length and the propagated vector.
    """

    dt_i = dt * (1 + 1j * imagt(t))

    if psi_full is None:
        psi_full = strang_step(psi, kin, V, t, dt_i, g)
    psi_half = strang_step(psi, kin, V, t, dt_i / 2, g)
    # The second half step starts at t + dt/2, not at t. Evaluating the potential at t here costs the
    # splitting an order of accuracy whenever V depends on time, and understates the error estimate
    # below with it, so the controller then takes steps far larger than the tolerance asked for.
    psi_double = strang_step(psi_half, kin, V, t + dt_i.real / 2, dt_i / 2, g)

    # Computing the error, using a standard 2-norm.
    # err = np.sum(np.abs(psi_full - psi_double) ** 2) / np.sum(np.abs(psi_full) ** 2)
    err = distance(psi_double, psi_full)
    if err > tol:  # If the error is superior, try again with a time step dt/2
        return adaptative_step(psi, kin, V, t, dt / 2, g, tol, imagt, psi_full=psi_half)
    else:  # else return the results and compute a new time-step
        if err == 0:
            s = 10
        else:
            s = max(min(0.6 * (tol / err) ** 0.25, 10), 0.1)
        return dt, s * dt, psi_double


def propagate(
    t_init: float,
    t_final: float,
    kin: KineticStep,
    t_samples: xr.DataArray,
    psi: np.ndarray,
    V: Callable,
    dt: float,
    g: float,
    tol: float,
    imagt: Callable,
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
        verbose (bool, optional): Wheter to plot a progress bar, useful for knowing where blow-up might happen. Defaults to False.

    Raises:
        ValueError: Raises an error if the initial sampling point is before t_init.

    Returns:
        tuple[list[np.ndarray]]: The vector psi sampled at the times specified by t_samples.
    """
    t = t_init
    count_t = 0  # tracking what is the next sampling time.

    psi_list = []
    dt_max = t_samples.data[1] - t_samples.data[0]

    # verify that the first sampling point is not before t_init
    if t == t_samples[0]:
        psi_list += [psi]
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
    total_t = t_final - t_init
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
        dt_used, dt, psi = adaptative_step(psi, kin, V, t, dt, g, tol, imagt)
        t += dt_used
        dt = min(dt, dt_max)  # making sure not to skip sampling times
        dt = max(min(dt, dtmax), dtmin)  # bounding the step time to reasonable values

        if t >= t_samples[count_t]:
            psi_list += [psi]
            count_t += 1
        # Clamped to what is left rather than skipped near the end, so the bar reaches 100%
        # without overshooting. This only reads dt_used, it never feeds back into dt.
        pbar.update(min(dt_used, total_t - pbar.n))

    pbar.close()

    n_samples = len(psi_list)
    if n_samples != len(t_samples):
        print(
            f"Less time steps than required samples, padding the array with last psi, last proper sample is {n_samples}"
        )
        psi_list += [psi] * (len(t_samples) - n_samples)

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
        self.allcoords.update(
            {
                dim: ["potential", coord]
                for dim, coord in self.potential.param_coords.items()
            }
        )

        missing = [dim for dim in self.spatial_dims if dim not in psi0.dims]
        if missing:
            raise ValueError(f"psi0 is missing the dimension(s) {missing} of the potential")

        # Adding all additional dimensions of psi0 to the coordinates dictionnary.
        spatial_names = self.spatial_dims + self.potential.coord_names[: self.n_dims]
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
    ) -> tuple[xr.DataArray, list[tuple[int]], list[tuple]]:
        """Lay out the storage array and the argument list of every run to perform, one per point of
        the parameter space. Shared by this solver and the rescaling one, which only differ in how
        they propagate and store the results.

        Args:
            t_samples (xr.DataArray): Sampling times for psi, with at least a dimension 't'.
            dt0 (Union[float, xr.DataArray]): Initial time step, possibly parameter dependant.
            tol (Union[float, xr.DataArray]): Adaptative method tolerance, possibly parameter dependant.

        Returns:
            tuple[xr.DataArray, list[tuple[int]], list[tuple]]: The empty psi array to fill, the index
            tuples covering the parameter space, and the arguments of 'propagate' for each of them.
        """

        # Adding eventual time sampling dimensions to the context.
        for dim in t_samples.dims:
            check_name(dim, self.n_dims)
            if dim != "t" and dim not in self.allcoords:
                self.allcoords.update({dim: ["t", t_samples.coords[dim]]})

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
                Vt = self.potential.make_Vt(potential_sel)
            else:
                Vt = self.potential.make_V(potential_sel)

            ## select t_samples
            samples_sel = subselect(indexes, "t", self.allcoords)
            t_samples_selected = t_samples.sel(samples_sel)

            ## Select the interaction strength
            g_sel = subselect(indexes, "g", self.allcoords)
            g_selected = (
                self.g.sel(g_sel).data if isinstance(self.g, xr.DataArray) else self.g
            )

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
        n_cores: int = 8,
        workers: int = 1,
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
            verbose (bool, optional): Whether to plot progress bars: one over the runs, and, when
            'parallel' is not set, one over the simulated time of the current run, useful for knowing
            where blow-up might happen. Defaults to True.
            n_cores (int, optional): The number of cores to use for the parallelized solver.
            workers (int, optional): Threads given to each Fourier transform, -1 for every core. Worth
            raising for grids of 512^2 and upwards, where it buys about a third of the transform time;
            below that the transforms are too small to gain anything. Leave it at 1 when 'parallel' is
            set, since the parameter space is then already spread over the cores. Defaults to 1.

        Returns:
            xr.DataArray: The value of Psi for each time sampling point at each point of the parameter space.
        """

        psi, selections, list_args = self.prepare_runs(t_samples, dt0, tol)
        n_samples = len(t_samples.coords["t"].data)

        def x(*y):
            # One propagator cache per run: it is keyed on the time step alone, so it must not
            # outlive the grid it was built for
            return propagate(
                t_init,
                t_final,
                self.kinetic_step(workers),
                *y,
                imagt = self.imagt,
                # A worker process must never open a bar: several of them writing carriage
                # returns to the one stderr produces nothing readable.
                verbose=verbose and not parallel,
                **kwargs,
            )

        if not parallel:
            for i, indexes in enumerate(
                bar(selections, desc="Propagating runs", unit="run", verbose=verbose)
            ):
                psi_list = x(*list_args[i])
                for j in range(n_samples):
                    slic = [j, *indexes]
                    psi[*slic] = psi_list[j]
        else:
            psi_list_list = parallel_map(
                x,
                list_args,
                n_jobs=n_cores,
                desc="Propagating runs",
                unit="run",
                verbose=verbose,
            )

            for i, indexes in enumerate(selections):
                for j in range(n_samples):
                    slic = [j, *indexes]
                    psi[*slic] = psi_list_list[i][j]

        return psi.squeeze()
