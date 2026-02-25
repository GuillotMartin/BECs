# %%
from typing import Union, Callable
import numpy as np
import xarray as xr
from scipy.fft import fftn, ifftn, fftfreq
from tqdm import tqdm
from bloch_schrodinger.potential import Potential
from BECs.potentialT import PotentialT
from bloch_schrodinger.fdsolver import FDSolver
from BECs.groundstate import subselect
from joblib import Parallel, delayed
from copy import deepcopy

# Yoshida splitting coefficient
cbrt2 = 2**(1/3)          
w1    = 1/(2 - cbrt2)     
w0    = -cbrt2/(2 - cbrt2)

def distance(psi1: np.ndarray, psi2: np.ndarray) -> float:
    """Compture the Fubiny=i-study distance between two states.

    Args:
        psi1 (np.ndarray): First state
        psi2 (np.ndarray): Second state

    Returns:
        float: The absolute value fubini-study metric, it should always be positive but numerical errors happens.
    """
    psi1_norm = psi1 / max(np.linalg.norm(psi1), 1e-15)
    psi2_norm = psi2 / max(np.linalg.norm(psi2), 1e-15)
    return np.abs((1 - np.abs(np.sum(np.conjugate(psi1_norm) * psi2_norm)) ** 2))


def check_name(name: str):
    """Check wether the name is a valid one. and raises an error if not.

    Args:
        name (str): The name to check

    Raises:
        ValueError: If the name is forbidden
    """

    forbidden_names = ["field", "band", "a1", "a2", "x", "y"]
    if name in forbidden_names:
        raise ValueError(
            f"{name} is not a valid name for the object, as it is already used. The forbidden names are: {forbidden_names}"
        )


def losses(
    x: xr.DataArray, y: xr.DataArray, width: float, gamma: float
) -> xr.DataArray:
    """Creat a lose term for an absorbing boundaries. This absorbing layer has a sinuosidal shape for smooth absorption.

    Args:
        V (xr.DataArray): The potential on width to add the losses
        width (float): The width of the lossy layer
        gamma (float): The amplitude of the loss layer

    Returns:
        xr.DataArray: The modified potential
    """
    width = 2 * width
    rgx = (x.max() - x.min()) / 2
    mnx = (x.max() + x.min()) / 2
    distx = abs((x - mnx) / (rgx)) / (width)

    rgy = (y.max() - y.min()) / 2
    mny = (y.max() + y.min()) / 2
    disty = abs((y - mny) / (rgy)) / width

    losses = xr.where(disty < distx, distx, disty)
    losses = xr.where(losses < (1 - width) / width, 0, losses - (1 - width) / width)
    losses = -(xr.ufuncs.cos(np.pi * losses) - 1) / 2

    return 1j * losses * gamma


def linear_step(
    psi: np.ndarray,
    dt: complex,
    ks: tuple[np.ndarray, np.ndarray],
    aliasing: np.ndarray,
) -> np.ndarray:
    """Linear propagation of the vector psi for a step dt by multiplication in Fourier space.

    Args:
        psi (np.ndarray): The vector to propagate.
        dt (float): The time step.
        ks (tuple[np.ndarray, np.ndarray]): Values of kx and ky.
        aliasing (np.ndarray): A high-k cut off mask for anti-aliasing.

    Returns:
        np.ndarray: Propagated vector.
    """
    psi_f = fftn(psi, axes=[0, 1]) * aliasing
    psi_f *= np.exp(1j * dt * (ks[0] ** 2 + ks[1] ** 2)/2)
    return ifftn(psi_f, axes=[0, 1])


def potential_step(
    psi: np.ndarray,
    dt: complex,
    V: Union[np.ndarray, xr.DataArray],
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
    psi_sq = np.abs(psi) ** 2
    return np.exp(1j * dt * (g * psi_sq)) * psi


def strang_step(
    psi: np.ndarray,
    ks: tuple[np.ndarray, np.ndarray],
    aliasing: np.ndarray,
    V: Callable,
    t: float,
    dt: complex,
    g: float,
) -> np.ndarray:
    """Propagate psi for a full step dt using a symmetric strang splitting

    Args:
        psi (np.ndarray): The vector to propagate.
        ks (tuple[np.ndarray,np.ndarray]): Values of kx and ky.
        aliasing (np.ndarray): A high-k cut off mask for anti-aliasing.
        V (Union[np.ndarray,xr.DataArray]): Potential landscape.
        t (float): time t for potential selection.
        dt (float): time step.
        g (float): Non-linear coefficient.

    Returns:
        np.ndarray: Propagated vector.
    """

    V_t = V(t + dt.real / 2)

    psi_1 = linear_step(psi, dt / 2, ks, aliasing)
    psi_2 = potential_step(psi_1, dt / 2, V_t)
    psi_3 = nl_step(psi_2, dt, g)
    psi_4 = potential_step(psi_3, dt / 2, V_t)
    psi_5 = linear_step(psi_4, dt / 2, ks, aliasing)
    return psi_5

def yoshida_step(
    psi: np.ndarray,
    ks: tuple[np.ndarray, np.ndarray],
    aliasing: np.ndarray,
    V: Callable,
    t: float,
    dt: float,
    g: float,
) -> np.ndarray:
    """Propagate psi for a full step dt using a fourth order Yoshida step
    Args:
        psi (np.ndarray): The vector to propagate.
        ks (tuple[np.ndarray,np.ndarray]): Values of kx and ky.
        aliasing (np.ndarray): A high-k cut off mask for anti-aliasing.
        V (Union[np.ndarray,xr.DataArray]): Potential landscape.
        t (float): time t for potential selection.
        dt (float): time step.
        g (float): Non-linear coefficient.

    Returns:
        np.ndarray: Propagated vector.
    """
    
    psi1 = strang_step(psi, ks, aliasing, V, t, dt*w1, g)
    psi2 = strang_step(psi1, ks, aliasing, V, t + dt*w1, dt*w0, g)
    psi3 = strang_step(psi2, ks, aliasing, V, t + dt*w1 + dt*w0, dt*w1, g)
    return psi3


def adaptative_step(
    psi: np.ndarray,
    ks: tuple[np.ndarray],
    aliasing: np.ndarray,
    V: Callable,
    t: float,
    dt: float,
    g: float,
    tol: float,
    imagt: Callable
) -> tuple[float, float, np.ndarray]:
    """Propagate psi for a full step dt, using a recursive adaptative step-doubling method.
    This function propagate psi for dt and for 2*dt/2, then compares the results. If its above a certain tolerance,
    the function calls itself again with a halved time step.

    Args:
        psi (np.ndarray): The vector to propagate.
        ks (tuple[np.ndarray,np.ndarray]): Values of kx and ky.
        aliasing (np.ndarray): A high-k cut off mask for anti-aliasing.
        V (Union[np.ndarray,xr.DataArray]): The potential landscape, must have a dimension 't'.
        dt (float): time step.
        g (float): Non-linear coefficient.
        tol (float): The tolerance for step doubling
        imagt (Callable): A function of time t such that dt(t) = dt * (1 + 1j * imagt(t)).

    Returns:
        tuple[float, float, np.ndarray]: The time step length used, the optimal next time step length and the propagated vector.
    """
    
    dt_i = dt * (1 + 1j * imagt(t))
    
    psi_full = strang_step(psi, ks, aliasing, V, t, dt_i, g)
    psi_half = strang_step(psi, ks, aliasing, V, t, dt_i / 2, g)
    psi_double = strang_step(psi_half, ks, aliasing, V, t, dt_i / 2, g)

    # Computing the error, using a standard 2-norm.
    # err = np.sum(np.abs(psi_full - psi_double) ** 2) / np.sum(np.abs(psi_full) ** 2)
    err = distance(psi_double, psi_full)
    if err > tol:  # If the error is superior, try again with a time step dt/2
        return adaptative_step(psi, ks, aliasing, V, t, dt / 2, g, tol, imagt)
    else:  # else return the results and compute a new time-step
        if err == 0:
            s = 10
        else:
            s = max(min(0.6 * (tol / err) ** 0.25, 10), 0.1)
        return dt, s * dt, psi_double


def propagate(
    t_init: float,
    t_final: float,
    aliasing: np.ndarray,
    ks: tuple[np.ndarray],
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
        aliasing (np.ndarray): A high-k cut off mask for anti-aliasing.
        ks (tuple[np.ndarray]): Values of kx and ky.
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

    # create a progress bar if asked
    if verbose:
        with tqdm(
            total=t_final - t_init,
            bar_format="{l_bar}{bar}| {n:.3f}/{total_fmt}, {rate_fmt}, [{elapsed} < {remaining}]",
        ) as pbar:
            # propagating psi and storing at each time-step reaching the next t_sampling point
            while t < t_final and count_t < len(t_samples):
                dt_used, dt, psi = adaptative_step(
                    psi, ks, aliasing, V, t, dt, g, tol, imagt
                )                
                t += dt_used
                dt = min(dt, dt_max)  # making sure not to skip sampling times
                dt = max(
                    min(dt, kwargs.get("dtmax", 1)), kwargs.get("dtmin", 1e-5)
                )  # bounding the step time to reasonable values

                if t >= t_samples[count_t]:
                    psi_list += [psi]
                    count_t += 1
                if t + dt_used < t_final and verbose:
                    pbar.update(dt_used)

    else:
        while t < t_final and count_t < len(t_samples):
            dt_used, dt, psi = adaptative_step(
                psi, ks, aliasing, V, t, dt, g, tol, imagt
            )
            t += dt_used
            dt = min(dt, dt_max)  # making sure not to skip sampling times
            dt = max(
                min(dt, kwargs.get("dtmax", 0.1)), kwargs.get("dtmin", 1e-6)
            )  # bounding the step time to reasonable values
            if t >= t_samples[count_t]:
                psi_list += [psi]
                count_t += 1

    n_samples = len(psi_list)
    if n_samples != len(t_samples):
        print(
            f"Less time steps than required samples, padding the array with last psi, last proper sample is {n_samples}"
        )
        psi_list += [psi] * (len(t_samples) - n_samples)

    return psi_list


class SSFM(FDSolver):
    def __init__(
        self,
        potential: Union[Potential, PotentialT],
        psi0: xr.DataArray,
        g: Union[float, xr.DataArray],
    ):
        """Initialize a solver instance for the Gross-Pitaevskii equation. This solver handles only scalar equations on rectangular grids.

        Args:
            potential (Union[Potential,PotentialT]): The potential landscape, must be describing a rectangular grid. The solver will iterate over each additional dimensions (not a1 and a2).
            psi0 (xr.DataArray): Initial vector, must have shape and dimensions (a1,a2) constitant with the potential.
            g (Union[float, xr.DataArray]): Interaction strength term. Can be passed as an array over which to iterate.

        Raises:
            ValueError: If the potential and initial vector given do not have the proper dimensions.
            ValueError: If the potential grid is not rectangular and aligned with x and y.
        """
        if not isinstance(potential, PotentialT):
            self.potential = PotentialT.fromPotential(
                potential
            )  # deepcopy to add losses without modifying the original object
        else:
            self.potential = deepcopy(potential)
        
        self.g = g

        # storing all parameter coordinates from potential, alpha and g. The final solver will run on all these dimensions.
        self.allcoords = {}
        coords_pot = {
            dim: ["potential", self.potential.coords[dim]] for dim in self.potential.coords
        }
        self.allcoords.update(coords_pot)

        if isinstance(g, xr.DataArray):
            for dim in g.dims:
                check_name(dim)
                coords_alpha = {dim: ["g", g.coords[dim]] for dim in g.dims}
                self.allcoords.update(coords_alpha)

        if "a1" not in psi0.dims or "a2" not in psi0.dims:
            raise ValueError("psi0 dimensions not consistant with V")
        self.psi0 = psi0

        # Adding all additional dimensions of psi0 to the coordinates dictionnary.
        coords_psi0 = {
            dim: ["psi0", self.psi0.coords[dim]]
            for dim in self.psi0.dims
            if dim not in ["a1", "a2", "x", "y"] and dim not in self.allcoords
        }
        self.allcoords.update(coords_psi0)

        self.a1 = potential.a1  # The first lattice vector
        self.a2 = potential.a2  # The second lattice vector

        if self.a1 @ self.a2 != 0 or self.a1[1] != 0 or self.a2[0] != 0:
            raise ValueError("This solver only works for x-y aligned rectangular grids")

        self.nb = 1  # important for 'initialize_eigve'
        self.na1 = len(self.potential.V.a1.data)  # discretization along x
        self.na2 = len(self.potential.V.a2.data)  # discretization along y
        self.n = self.na1 * self.na2
        self.a1_coord = self.potential.V.coords["a1"]
        self.a2_coord = self.potential.V.coords["a2"]
        
        # length steps along a1 and a2
        self.dx = (
            float(abs(self.potential.V.a1[1] - self.potential.V.a1[0]))
            * (self.a1 @ self.a1) ** 0.5
        )  # smallest increment of length along x
        self.dy = (
            float(abs(self.potential.V.a2[1] - self.potential.V.a2[0]))
            * (self.a2 @ self.a2) ** 0.5
        )  # smallest increment of length along y

        # kxmax = np.pi

        kx = fftfreq(self.na1, self.dx) * 2 * np.pi
        ky = fftfreq(self.na2, self.dy) * 2 * np.pi
        self.kx, self.ky = np.meshgrid(kx, ky, indexing="ij")

        kxmax = np.max(kx)
        kymax = np.max(ky)

        self.aliasing = np.where(
            (self.kx**2 + self.ky**2) ** 0.5 > max(kxmax, kymax) / 3 * 2, 0, 1
        )
        
        self.imagt = lambda t: 0 # A function to add a imaginary part to the time steps dt. makes it so dt(t) = dt * (1 + 1j * imagt(t))

    def initialize_eigva(self):
        return super().initialize_eigva(1)

    def initialize_psi(self):
        return (
            super().initialize_eigve(1, False).transpose(..., "a1", "a2").rename("psi")
        )
        
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
        loss = losses(self.potential.x, self.potential.y, width, amp)
        self.potential.V = self.potential.V + loss
        self.potential.update_V0()

    def solve(
        self,
        t_init: float,
        t_final: float,
        t_samples: xr.DataArray,
        dt0: Union[float, xr.DataArray] = 1e-3,
        tol: Union[float, xr.DataArray] = 1e-6,
        parallelize: bool = False,
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
            verbose (bool, optional): Wheter to plot a progress bar, useful for knowing where blow-up might happen. Defaults to False.
            n_cores (int, optional): The number of cores to use for the parallelized solver.

        Returns:
            xr.DataArray: The value of Psi for each time sampling point at each point of the parameter space.
        """

        # Adding eventual time sampling dimensions to the context.
        for dim in t_samples.dims:
            check_name(dim)
            if dim != "t" and dim not in self.allcoords:
                self.allcoords.update({dim: ["t", t_samples.coords[dim]]})

        # Create empty DataArrays to store the eigenvalues and vectors
        psi = self.initialize_psi()
        if "t" not in psi.dims:
            psi = psi.expand_dims(dim={"t": t_samples.coords["t"]})
        psi = psi.transpose("t", ...).copy()

        # We create a list of tuples that select a single value for each of the parameter dimensions
        indexes = [np.arange(len(coord[1])) for coord in self.allcoords.values()]
        indexGrid = np.meshgrid(*indexes, indexing="ij")
        indexGrid = [grid.reshape(-1) for grid in indexGrid]
        selections = [tup for tup in zip(*indexGrid)]

        if len(selections) == 0:
            selections = [()]

        # We will store the value of each parameter of the 'propagate' function for each iteration in lists.
        V_list = []
        g_list = []
        psi0_list = []
        t_samples_list = []
        dt0_list = []
        tol_list = []

        for indexes in selections:
            # --- Constructing the inputs for 'propagate' ---
            ## select the potential
            potential_sel = subselect(indexes, "potential", self.allcoords)

            Vt = self.potential.make_Vt(potential_sel)
                        
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

            # Store the arguments for the ground state solver as lists
            V_list += [Vt]
            g_list += [g_selected]
            psi0_list += [psi0_selected]
            t_samples_list += [t_samples_selected]
            dt0_list += [dt0_selected]
            tol_list += [tol_selected]

        
        list_args = list(
            zip(
                t_samples_list,
                psi0_list,
                V_list,
                dt0_list,
                g_list,
                tol_list,
            )
        )

        n_samples = len(t_samples.coords["t"].data)

        def x(y):
            return propagate(
                t_init,
                t_final,
                self.aliasing,
                (self.kx, self.ky),
                *y,
                imagt = self.imagt,
                verbose=verbose,
                **kwargs,
            )

        if not parallelize:
            print(
                f"Propagating the initial states. {len(selections)} iterations to perform"
            )
            for i, indexes in enumerate(selections):
                psi_list = x(list_args[i])

                for j in range(n_samples):
                    slic = [j, *indexes, 0, 0]
                    psi[*slic] = psi_list[j]
        else:
            parallel = Parallel(
                n_jobs=n_cores, return_as="list", verbose=51 if verbose else 5
            )
            psi_list_list = parallel(delayed(x)(y) for y in list_args)

            for i, indexes in enumerate(selections):
                for j in range(n_samples):
                    slic = [j, *indexes, 0, 0]
                    psi[*slic] = psi_list_list[i][j]

        return psi.squeeze()
