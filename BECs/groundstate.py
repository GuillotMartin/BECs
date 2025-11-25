import numpy as np
import xarray as xr
from typing import Union
from tqdm import tqdm
import scipy.sparse as sps
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

from bloch_schrodinger.potential import Potential
from bloch_schrodinger.solver import Solver, check_name

# --- RKF45 coefficients for the adaptative time step (Fehlberg) ---
a2 = 1/4
a3 = 3/8
a4 = 12/13
a5 = 1
a6 = 1/2

b21 = 1/4

b31 = 3/32
b32 = 9/32

b41 = 1932/2197
b42 = -7200/2197
b43 = 7296/2197

b51 = 439/216
b52 = -8
b53 = 3680/513
b54 = -845/4104

b61 = -8/27
b62 = 2
b63 = -3544/2565
b64 = 1859/4104
b65 = -11/40

# 4th order
c1 = 25/216
c2 = 0
c3 = 1408/2565
c4 = 2197/4104
c5 = -1/5
c6 = 0

# 5th order (for error estimation)
d1 = 16/135
d2 = 0
d3 = 6656/12825
d4 = 28561/56430
d5 = -9/50
d6 = 2/55



def distance(psi1, psi2):
    psi1_norm = psi1/max(np.linalg.norm(psi1), 1e-15)
    psi2_norm = psi2/max(np.linalg.norm(psi2), 1e-15)
    return np.abs((1-np.abs(np.sum(np.conjugate(psi1_norm)*psi2_norm))**2))

# def normalize(psi, norm):
#     return norm * psi/max(np.linalg.norm(psi), 1e-15)

def normalize(psi, norm):
    n = np.linalg.norm(psi)
    if not np.isfinite(n) or n < 1e-15:
        n= 1e-15
    return psi*norm/n

def f(H0:sps.dia_array, g:np.ndarray, psi:np.ndarray)->np.ndarray:
    """The function describing "y' = f(y)" for the RK4 method. It is the Gross-Pitaevskii equation with imaginary time. 

    Args:
        H0 (sps.dia_array): The linear Hamiltonian
        g (np.ndarray): The interaction vector
        psi (np.ndarray): the vector to propagate

    Returns:
        np.ndarray: y'
    """
    return -(H0@psi + (g * np.abs(psi)**2) * psi)
    

def oneStep(H0:sps.dia_array, g:np.ndarray, dt:float, psi:np.ndarray, norm:float, tol:float)->tuple[float, float, np.ndarray]:
    """A simple RK4 method to propagate the vector.

    Args:
        H0 (sps.dia_array): The linear Hamiltonian
        g (np.ndarray): The interaction vector
        dt (float): The time-step
        psi (np.ndarray): the vector to propagate

    Returns:
        np.ndarray: The propagated vector
    """
    
    k1 = f(H0, g, psi)
    k2 = f(H0, g, normalize(psi + dt*(b21*k1), norm))
    k3 = f(H0, g, normalize(psi + dt*(b31*k1 + b32*k2), norm))
    k4 = f(H0, g, normalize(psi + dt*(b41*k1 + b42*k2 + b43*k3), norm))
    k5 = f(H0, g, normalize(psi + dt*(b51*k1 + b52*k2 + b53*k3 + b54*k4), norm))
    k6 = f(H0, g, normalize(psi + dt*(b61*k1 + b62*k2 + b63*k3 + b64*k4 + b65*k5), norm))

    # 4th and 5th order estimates
    y4 = psi + dt*(c1*k1 + c2*k2 + c3*k3 + c4*k4 + c5*k5 + c6*k6)
    y4 = normalize(y4, norm)
    y5 = psi + dt*(d1*k1 + d2*k2 + d3*k3 + d4*k4 + d5*k5 + d6*k6)
    y5 = normalize(y5, norm)

    # Error estimate
    err = distance(y4,y5)
    err = err if not np.isnan(err) else 0
    dt = dt if not np.isnan(dt) else 1e-5
    
    if err > tol:
        return oneStep(H0, g, dt/2, psi, norm, tol)
    else:
        y_next = y5  # usually take 5th-order solution

        # Step size control
        if err == 0:
            s = 2  # increase aggressively if perfect
        else:
            s = (tol / err)**0.25
        
        dt = min(max(s * dt, 1e-8), 1)
        E = np.real(np.dot(np.conjugate(y_next), -f(H0, g, y_next)))

        return dt, E, y_next
    
def findGroundState(
        H0:sps.dia_matrix, 
        gs:list, 
        psi0:np.ndarray,
        dt:float,
        npoints:int, 
        tol:float = 1e-8,
        maxiter = 100
    )->tuple[float,xr.DataArray]:
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
        np.ndarray: The ground state vector.
    """
    El = []
    g = np.array([[g]*npoints for g in gs]).reshape(-1)
    
    norm = np.linalg.norm(psi0)
    dt, E, psi1 = oneStep(H0, g, dt, psi0, norm, tol)
    
    El += [E]
    count = 1
    while distance(psi1, psi0) > tol and count<maxiter:
        psi0 = psi1*1
        dt, E, psi1 = oneStep(H0, g, dt, psi0, norm, tol)
        El += [E]
        count +=1
    if count>=maxiter - 1:
        print("maximum number of iterations reached, the returned state might not be converged")

    # plt.plot(El)
    # plt.show()

    return El[-1], psi1


def subselect(indexes:tuple[int], ptype:str, allcoords:dict)->dict[float]:
    """Create a subselction from allcoords based on a keyword 'ptype'

    Args:
        indexes (tuple[int]): The indexes of the values to select in each coordinate array.
        ptype (str): The keyword to select the subselection, can be 'potential', 'alpha', 'g', 'reciprocal', 'population' or 'coupling'.
        allcoords (dict): The dictionnary containing the coordinates and their type

    Returns:
        dict: The subselection
    """
    subselect = {
        dim:allcoords[dim][1][i]
        for i, dim in zip(indexes, allcoords.keys())
        if allcoords[dim][0] == ptype
    }
    return subselect

class GroundState(Solver):
    """The ground state class uses an imaginary time propagation method to find the ground state of the Gross-Pitaevskii equation.
    It uses a RK4 propagation method and is built on the Solver class from the 'bloch_schrodinger' package."""
    
    def __init__(
            self, 
            potentials:Union[Potential,list[Potential]],
            alphas:Union[Union[float,xr.DataArray],list[Union[float,xr.DataArray]]],
            gs:Union[Union[float,xr.DataArray],list[Union[float,xr.DataArray]]]
        ):
        """Instantiate the solver.

        Args:
            potentials (Union[Potential,list[Potential]]): The potential felt by each field. They must all be defined on the same grid. 
            A single potential can also be passed for a scalar equation.
            alphas (Union[Potential,list[Potential]]): The kinetic energy coefficient hbar²/2m for each field. 
            A single coefficient can be passed for a scalar equation.
            alphas (Union[Potential,list[Potential]]): The interaction energy for each field. 
            A single coefficient can be passed for a scalar equation.

        Raises:
            ValueError: Not the same number of potentials and kinetic terms given.
        """
        
        super().__init__(potentials, alphas)
        
        if self.nb == 1 and (isinstance(gs, float) or isinstance(gs, xr.DataArray)):
            self.gs = [gs]
        else:
            self.gs = gs
        
        # Adding the interaction terms to the list of coordinates
        for g in self.gs:
            if isinstance(g, xr.DataArray):
                for dim in g.dims:
                    check_name(dim)
                coords_g = {dim:['g',g.coords[dim]] for dim in g.dims}
                self.allcoords.update(coords_g)

    def make_g_list(self, sel:dict)->list[float]:
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
            if isinstance(self.gs[u],float) or isinstance(self.gs[u],int):
                gs += [self.gs[u]]
            elif isinstance(self.gs[u],xr.DataArray):
                sub_sel = {dim:value for dim,value in sel.items() if dim in self.gs[u].dims}
                gs += [float(self.gs[u].sel(sub_sel).data)]
            else:
                raise TypeError(f"{u}-th g term not of a recognized type (int, float or xr.DataArray)")
        return gs

    def solve(
        self, 
        population:Union[float,xr.DataArray] = 1, 
        tol:float = 1e-8,
        maxiter = 1000,
        phase0:tuple[float,float,int] = (0.51,0.51,0)
        )->tuple[xr.DataArray]:
        """Find the ground states of the Hamiltonians, using an imaginary time propagation initialized with the ground state of the linear Hamiltonian H0
        """

        if isinstance(population, xr.DataArray):
            for dim in population.dims:
                check_name(dim)
            coords_pops = {dim:['population',population.coords[dim]] for dim in population.dims}
            self.allcoords.update(coords_pops)
        
        # Create empty DataArrays to store the eigenvalues and vectors
        eigva = self.initialize_eigva(1).squeeze(dim = 'band')
        eigve = self.initialize_eigve(1).squeeze(dim = 'band')

        # We are going to create all the linear Hamiltonian and find they lowest eigenvector first, before finding the ground states
        ## First we create a list of tuples that select a single value for each of the parameter dimensions
        
        indexes = [np.arange(len(coord[1])) for coord in self.allcoords.values()]
        indexGrid = np.meshgrid(*indexes , indexing = 'ij')
        indexGrid = [grid.reshape(-1) for grid in indexGrid]
        selections = [tup for tup in zip(*indexGrid)]

        if len(selections) == 0:
            selections = [()]

        H0_list = []
        g_list = []
        psi0_list = []
        dt_list = []
        
        # Initializing the vector guess. The solver works better with a good guess for the lowest eigenvector
        X = np.random.rand(self.n)
        
        # Initializing the progress bar
        print("Computing the initial guesses")
        pbar = tqdm(total=len(selections))
        # Looping over first the potential dimensions, then the alpha dimensions, then the reciprocal dimensions and finally the coupling dimensions.
        for indexes in selections:
            # print(indexes)
            # --- Constructing the Hamiltonian from the parameter selection ---
            # The potential is a diagonal matrix, which we stored as a data array.
            potential_sel = subselect(indexes, 'potential', self.allcoords)
            potdiag = self.potential_data.sel(potential_sel).data
            potential_matrix = sps.diags(potdiag, offsets=0)
            
            alpha_sel = subselect(indexes, 'alpha', self.allcoords)            
            alphas = self.make_alpha_list(alpha_sel)
            self.compute_full_operators(alphas)
            
            reciprocal_sel = subselect(indexes, 'reciprocal', self.allcoords)
            if len(reciprocal_sel) == 0:
                recs = [0,0]
            else:
                recs = [reciprocal_sel['kx'], reciprocal_sel['ky']]   
            kinetic_matrix = self.compute_kinetic(recs)
            
            coupling_sel = subselect(indexes, 'reciprocal', self.allcoords)

            g_sel = subselect(indexes, 'g', self.allcoords)            
            gs = self.make_g_list(g_sel)
            
            pop_sel = subselect(indexes, 'population', self.allcoords)
            
            total_sel = {**potential_sel,**alpha_sel,**reciprocal_sel, **coupling_sel, **g_sel, **pop_sel} # aggregating all the values of each selections
            ham = kinetic_matrix + potential_matrix # The initial Hamiltonian contains only the kinetic operator and the potential operator 
            
            # --- Add the coupling terms to the Hamiltonian ---
            self.coupling_context.update(total_sel)
            for coupling in self.couplings:
                ham += eval(coupling, {"__builtins__": {}}, self.coupling_context)

            
            eigvals, eigvec = eigsh(ham, k = 1, v0 = X, which = 'SM')                                    
            X = eigvec
            eigvec_norm = eigvec/(self.da1*self.da2)**0.5       
            
            if isinstance(population, xr.DataArray):
                pop = float(population.sel(pop_sel).data)
            else:
                pop = population
                
            # Computing an approximately good time step
            dt = 2*np.pi / 1 / (eigvals + max(np.max(gs)*pop, 1000))     
            
            H0_list += [kinetic_matrix + potential_matrix]
            g_list += [[gs]]
            psi0_list +=  [eigvec_norm[:,0] * pop**0.5]
            dt_list += [dt[0]]
            pbar.update(1)
        pbar.close()
        
        print("Computing the ground states")
        pbar = tqdm(total=len(selections))
        for i in range(len(psi0_list)):
            indexes = selections[i]
            psi0 = psi0_list[i]
            H0 = H0_list[i]
            gs = g_list[i]
            dt = dt_list[i]
            
            energ, eigvec = findGroundState(
                H0, gs, psi0, dt, self.np, 
                tol = tol, maxiter = maxiter
            )
            
            eigva[*indexes] = energ*(self.da1*self.da2)
            eigve[*indexes] = eigvec
            
            pbar.update(1)
        pbar.close()
        
        eigve = eigve.unstack(dim='component').rename('ground state')
        sel0 = dict(a1 = phase0[0], a2 = phase0[1], field = phase0[2])
        
        eigve = eigve * xr.ufuncs.exp(1j*xr.ufuncs.angle(eigve.sel(sel0, method='nearest')))
        x = self.a1[0]*eigve.a1 + self.a2[0]*eigve.a2
        y = self.a1[1]*eigve.a1 + self.a2[1]*eigve.a2
        eigve = eigve.assign_coords({
            "x":x,
            "y":y,
        })
            
        return eigva.squeeze(), eigve.squeeze()
