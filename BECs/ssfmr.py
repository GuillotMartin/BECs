# %%
from typing import Union, Callable
import numpy as np
import xarray as xr
from scipy.fft import fft2, ifft2
from tqdm import tqdm
from BECs.potentialT import AnalyticPotential
from BECs.groundstate import subselect
from joblib import Parallel, delayed
from BECs.ssfm import SSFM, check_name, distance


def J(phi:np.ndarray, ks:tuple[np.ndarray, np.ndarray])->np.ndarray[np.ndarray, np.ndarray]:
    """Compute the current density in the invariant coordinates.

    Args:
        phi (np.ndarray): wavefunction
        ks (tuple[np.ndarray, np.ndarray]): kx and ky coordinates.

    Returns:
        tuple[np.ndarray, np.ndarray]: (J_x, J_y)
    """
    phi_f = fft2(phi)
    dphi = [np.imag(ifft2(phi_f*k)) for k in ks]
    return [np.real(1j * (phi * dphi_i.conjugate() - phi.conjugate() * dphi_i)/2) for dphi_i in dphi]

def a(psi:np.ndarray, coo:tuple[np.ndarray, np.ndarray])->np.ndarray[float,float]:
    """Compute the characteristic psystem size a in the coordinate system given by coo.

    Args:
        psi (np.ndarray): Initial wavefunction
        coo (tuple[np.ndarray, np.ndarray]): cartesian coordinates
    """
    dS = np.abs(
        (coo[0][1,1]-coo[0][0,0]) *
        (coo[1][1,1]-coo[1][0,0])
    )
    
    a = [(np.sum(c**2 * np.abs(psi)**2) * dS)**0.5 for c in coo]
    return np.array(a)

def linear_step(
    phi: np.ndarray,
    dt: complex,
    lambdas: list[float,float],
    ks: tuple[np.ndarray, np.ndarray],
    aliasing: np.ndarray,
) -> np.ndarray:
    """Linear propagation of the vector phi(t) for a step dt by multiplication in Fourier space.

    Args:
        psi (np.ndarray): The vector to propagate.
        dt (float): The time step.
        lambdas (list[float,float]): The values of the rescaling coefficients at time t.
        ks (tuple[np.ndarray, np.ndarray]): Values of kx and ky in the rescaled coordinates.
        aliasing (np.ndarray): A high-k cut off mask for anti-aliasing.

    Returns:
        np.ndarray: Propagated vector.
    """
    psi_f = fft2(phi) * aliasing
    psi_f *= np.exp(1j * dt * ((ks[0]/lambdas[0])**2 + (ks[1]/lambdas[1])**2)/2)
    return ifft2(psi_f)

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
    lambdas: list[float,float],
):
    """Non-linear propagation of the vector psi for a step dt by multiplication in real space.

    Args:
        psi (np.ndarray): The vector to propagate.
        dt (float): The time step.
        g (float): Non-linear coefficient.
        lambdas (list[float,float]): The values of the rescaling coefficients at time t.

    Returns:
        np.ndarray: Propagated vector.
    """
    psi_sq = np.abs(psi) ** 2
    return np.exp(1j * dt * g * psi_sq / lambdas[0] / lambdas[1]) * psi

def lambda_step(lambdas:np.ndarray[float,float], sigmas:np.ndarray[float,float], dt)->np.ndarray[float, float]:
    """Propagates lambdas over a time step dt using the Euler method
    """
    return lambdas+dt*sigmas
    # return lambdas

def dsigma(
    phi: np.ndarray,
    lambdas:np.ndarray[float,float],
    V: np.ndarray,
    consts:dict
    ) -> list[float, float]:

    prefac = 1/lambdas/consts["ai"]**2 * consts["dS"]
    
    # Non linear contribution
    nl_term = consts["g"] / 2 * np.sum(np.abs(phi)**4) / (np.prod(lambdas))
    
    # Potential contribution
    dV = [np.imag(ifft2(fft2(V)*k*consts["aliasing"])) for k in consts["ks"]]
    pot_term = [np.sum(np.abs(phi)**2 * consts["rho"][i] * dV[i]) for i in range(2)]
    
    # kinetic contribution
    dphi = [np.imag(ifft2(fft2(phi)*k*consts["aliasing"])) for k in consts["ks"]]
    kin_term = [np.sum(np.abs(dphi[i])**2 / lambdas[i]**2) for i in range(2)]

    return np.array([prefac[i] * (nl_term - pot_term[i] + kin_term[i]) for i in range(2)])

def sigmas_step(
    phi: np.ndarray,
    lambdas:np.ndarray[float,float],
    sigmas:np.ndarray[float,float],
    V: np.ndarray,
    dt:float,
    consts:dict
    ) -> list[float, float]: 
    return sigmas + dsigma(phi, lambdas, V, consts) * dt

import matplotlib.pyplot as plt

def strang_step(
    phi: np.ndarray,
    sigmas:np.ndarray[float,float],
    lambdas:np.ndarray[float,float],
    t: float,
    dt: float,
    consts:dict
    )-> tuple[np.ndarray,list[float,float], list[float,float]]:
    """Propagate psi for a full step dt using a symmetric strang splitting

    Args:
        psi (np.ndarray): The vector to propagate.
        sigmas (list[float,float]): Rescaling coefficients at t.
        lambdas (list[float,float]): first order time derivative of lambdas at time t.
        t (float): time t for potential selection.
        dt (float): time step.

    Returns:
        np.ndarray: Propagated vector.
        list[float,float] : sigmas
        list[float,float] : lambdas
    """
    V_t = consts["V"](t + dt / 2, lambdas[0]*consts["rho"][0], lambdas[1]*consts["rho"][1])
    
    dsig = dsigma(phi, lambdas, V_t, consts)
    V_rescaling = 0
    # print(dsig)
    for i in range(2):
        V_rescaling += dsig[i]*lambdas[i]*consts["rho"][i]**2 / 2

    # plt.imshow(V_rescaling)
    # plt.colorbar()
    # plt.show()
    # return None

    phi_1 = linear_step(phi, dt / 2, lambdas, consts["ks"], consts["aliasing"])
    phi_2 = potential_step(phi_1, dt / 2, V_t+V_rescaling)
    phi_3 = nl_step(phi_2, dt, consts["g"], lambdas)
    phi_4 = potential_step(phi_3, dt / 2, V_t+V_rescaling)
    phi_5 = linear_step(phi_4, dt / 2, lambdas, consts["ks"], consts["aliasing"])
    
    lambdas, sigmas = lambda_step(lambdas, sigmas, dt), sigmas_step(phi, lambdas, sigmas, V_t, dt, consts)
    
    return phi_5, sigmas, lambdas

def adaptative_step(
    phi: np.ndarray,
    sigmas:np.ndarray[float,float],
    lambdas:np.ndarray[float,float],
    t: float,
    dt: float,
    consts:dict
    ) -> tuple[float, float, np.ndarray]:
    """Propagate phi for a full step dt, using a recursive adaptative step-doubling method.
    This function propagate psi for dt and for 2*dt/2, then compares the results. If its above a certain tolerance,
    the function calls itself again with a halved time step.

    Args:
        phi (np.ndarray): The vector to propagate.
        sigmas (list[float,float]): Rescaling coefficients at t.
        lambdas (list[float,float]): first order time derivative of lambdas at time t.
        t (float): time t.
        dt (float): time step.
        consts (dict): all constant parameters

    Returns:
        tuple[float, float, np.ndarray, list[float,float], list[float,float]]: The time step length used, the optimal next time step length and the propagated vector, sigmas and lambdas
    """
    
   
    phi_full, sigmas_full, lambdas_full = strang_step(phi, sigmas, lambdas, t, dt, consts)
    phi_half, sigmas_half, lambdas_half = strang_step(phi, sigmas, lambdas, t, dt/2, consts)
    phi_double, sigmas_double, lambdas_double = strang_step(phi_half, sigmas_half, lambdas_half, t+dt/2, dt / 2, consts)

    # Computing the error, using a standard 2-norm.
    # err = np.sum(np.abs(psi_full - psi_double) ** 2) / np.sum(np.abs(psi_full) ** 2)
    err = distance(phi_double, phi_full)
    if err > consts["tol"]:  # If the error is superior, try again with a time step dt/2
        return adaptative_step(phi, sigmas, lambdas, t, dt / 2, consts)
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
    rho:tuple[np.ndarray,np.ndarray],
    ks: tuple[np.ndarray,np.ndarray],
    t_samples: xr.DataArray,
    psi: np.ndarray,
    V: Callable,
    dt: float,
    g: float,
    tol: float,
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
        tuple[list[list[float,float]], list[np.ndarray], list[np.ndarray]]: The rescaling coefficients lambdas and the wavefunctions phi and psi sampled at the times specified by t_samples.
    """
    t = t_init
    dS = np.abs( # infinitesimal surface element
        (rho[0][1,1]-rho[0][0,0]) *
        (rho[1][1,1]-rho[1][0,0])
    )
    
    # Compute initial conditions
    lambdas = np.ones(2)
    
    J0 = J(psi, ks)
    a0 = a(psi, rho) # at t = 0, the laboratory coordinates x-y are equal to the rescaled coordinates rho_i
    sigmas = np.array([np.sum(J0[i] * rho[i]) / a0[i]**2 for i in range(2)]) * dS

    phi = psi * np.exp(-1j/2 * (
        rho[0]**2 * sigmas[0] +
        rho[1]**2 * sigmas[1]
    ))
    
    count_t = 0  # tracking what is the next sampling time.

    phi_list = [] #Rescaled waefunction
    psi_list = [] #Regular wavefunction
    lambda_list = []
    dt_max = t_samples.data[1] - t_samples.data[0]

    consts = {
        "rho":rho, # The invariant coordinates rho.
        "ks":ks, # The invariant k-space coordinates
        "V":V, # The potential function
        "ai":a0, # The initial characteristic sizes
        "dS":dS,
        "aliasing": aliasing, # Aliasing mask
        "g":g, # non-linear factor
        "tol":tol # tolerance for the adaptative step
    }

    # verify that the first sampling point is not before t_init
    if t == t_samples[0]:
        phi_list += [phi]
        psi_list += [psi]
        lambda_list += [lambdas]
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
                dt_used, dt, phi, sigmas, lambdas = adaptative_step(
                    phi, sigmas, lambdas, t, dt, consts
                )                
                t += dt_used
                dt = min(dt, dt_max)  # making sure not to skip sampling times
                dt = max(
                    min(dt, kwargs.get("dtmax", 1)), kwargs.get("dtmin", 1e-5)
                )  # bounding the step time to reasonable values

                if t >= t_samples[count_t]:
                    phi_list += [phi]
                    psi_list += [
                        phi / np.prod(lambdas)**0.5 * 
                        np.exp(-1j/2*(
                            rho[0]**2 * sigmas[0]*lambdas[0] + 
                            rho[1]**2 * sigmas[1]*lambdas[1]
                        ))
                    ]
                    lambda_list += [lambdas]
                    count_t += 1
                if t + dt_used < t_final and verbose:
                    pbar.update(dt_used)

    else:
        while t < t_final and count_t < len(t_samples):
            dt_used, dt, phi, sigmas, lambdas = adaptative_step(
                    phi, sigmas, lambdas, t, dt, consts
                )             
            t += dt_used
            dt = min(dt, dt_max)  # making sure not to skip sampling times
            dt = max(
                min(dt, kwargs.get("dtmax", 0.1)), kwargs.get("dtmin", 1e-6)
            )  # bounding the step time to reasonable values
            if t >= t_samples[count_t]:
                psi_list += [phi]
                psi_list += [
                    phi / np.prod(lambdas)**0.5 * 
                    np.exp(1j*(
                        rho[0]**2 * sigmas[0]*lambdas[0] + 
                        rho[1]**2 * sigmas[1]*lambdas[1]
                    ))
                ]
                lambda_list += [lambdas]
                count_t += 1

    n_samples = len(phi_list)
    if n_samples != len(t_samples):
        print(
            f"Less time steps than required samples, padding the array with last psi, last proper sample is {n_samples}"
        )
        phi_list += [phi] * (len(t_samples) - n_samples)
        lambda_list += [lambdas] * (len(t_samples) - n_samples)

    return lambda_list, psi_list



class SSFMr(SSFM):
    def __init__(
        self,
        potential: AnalyticPotential,
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
        
        super().__init__(potential, psi0, g)

    def initialize_lambda(self):
        lambdas = super().initialize_eigva().squeeze(drop=True)
        lambdas = lambdas.expand_dims({"xy":np.arange(2)})
        lambdas.name = "lambda"
        return lambdas
    
    def initialize_psi(self):
        psi = super().initialize_psi()
        Lambda = self.initialize_lambda()
        
        psi = psi.assign_coords({"rho_x": psi.x, "rho_y": psi.y})
        psi = psi.assign_coords({"x": self.potential.V.x*Lambda[{"xy":0}], "y": self.potential.V.y*Lambda[{"xy":1}]})
        return psi

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
            psi.coords["x"] = psi.coords["x"].expand_dims(dim={"t": t_samples.coords["t"]})
            psi.coords["y"] = psi.coords["y"].expand_dims(dim={"t": t_samples.coords["t"]})
            
        psi = psi.transpose("t", ...).copy()
        psi.coords["x"] = psi.coords["x"].transpose("t", ...).copy()
        psi.coords["y"] = psi.coords["y"].transpose("t", ...).copy()
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

            Vtxy = self.potential.make_Vtxy(potential_sel)
                        
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
            V_list += [Vtxy]
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
                (self.potential.x.data, self.potential.y.data),
                (self.kx, self.ky),
                *y,
                verbose=verbose,
                **kwargs,
            )

        if not parallelize:
            print(
                f"Propagating the initial states. {len(selections)} iterations to perform"
            )
            for i, indexes in enumerate(selections):
                lambda_list, psi_list = x(list_args[i])              
                for j in range(n_samples):
                    slic = [j, *indexes]
                    psi.coords["x"][*slic] =  psi.coords["rho_x"] * lambda_list[j][0]
                    psi.coords["y"][*slic] =  psi.coords["rho_y"] * lambda_list[j][1]
                    psi[*slic] = psi_list[j]
        else:
            parallel = Parallel(
                n_jobs=n_cores, return_as="list", verbose=51 if verbose else 5
            )
            lambda_list_list, psi_list_list = parallel(delayed(x)(y) for y in list_args)

            for i, indexes in enumerate(selections):
                for j in range(n_samples):
                    slic = [j, *indexes]
                    psi.coords["x"][*slic] =  psi.coords["rho_x"] * lambda_list_list[i][j][0]
                    psi.coords["y"][*slic] =  psi.coords["rho_y"] * lambda_list_list[i][j][1]

                    psi[*slic] = psi_list_list[i][j]

        return psi.squeeze()

