import numpy as np
import xarray as xr
from typing import Union
from tqdm import tqdm
import scipy.sparse as sps
from scipy.sparse.linalg import eigsh
from joblib import Parallel, delayed
from scipy.fft import fftn, ifftn, fftfreq
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

from bloch_schrodinger.potential import Potential, create_parameter
from bloch_schrodinger.fdsolver import check_name, FDSolver

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

paramType = Union[float, xr.DataArray]


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



def normalize(psi: np.ndarray, norm: float) -> np.ndarray:
    """Renormalize a vector to a specified value, intended to me NaN resistant.

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

def project(psi:np.ndarray, proj:list[np.ndarray])->np.ndarray:
    """Projects psi on '1 - |vec><vec|' with vec running on all the elements of 'proj'

    Args:
        psi (np.ndarray): The vector to project
        proj (list[np.ndarray]): A list of vectors to project out

    Returns:
        np.ndarray: the projected out vector
    """
    psi_out = psi
    for vec in proj:
        psi_out -= np.sum(np.conjugate(vec)*psi) * vec
    
    return psi_out
    


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
    proj:list[np.ndarray]
) -> tuple[float, float, np.ndarray]:
    """A recursive RKF45 adaptative method to propagate the vector one step. Calls itself with a smaller time-step if the tolerance set is not matched.

    Args:
        H0 (sps.dia_array): The linear Hamiltonian
        g (np.ndarray): The interaction vector
        dt (float): The time-step
        norm (float): The norm of the vector, important to specify it as not to propagate numerical errors.
        psi (np.ndarray): the vector to propagate.
        tol (float): The tolerance (fubini-study distance) specifying wheter to restarts the step
        proj (list[np.ndarray]): The list of previous eigenstates, to project out and find the next eigenstate.
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
    y4 = project(y4, proj)
    y4 = normalize(y4, norm)
    y5 = psi + dt * (d1 * k1 + d2 * k2 + d3 * k3 + d4 * k4 + d5 * k5 + d6 * k6)
    y5 = project(y5, proj)
    y5 = normalize(y5, norm)

    # Error estimate
    err = distance(y4, y5)
    err = err if not np.isnan(err) else 0
    dt = dt if not np.isnan(dt) else 1e-5

    if err > tol:
        return oneStep(H0, g, dt / 2, psi, norm, tol, proj)
    else:
        y_next = y5  # usually take 5th-order solution

        # plt.imshow(np.abs(y5).reshape((50,50))**2)
        # plt.show()
        
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


def findStates(
    H0: sps.dia_matrix,
    g: np.ndarray,
    psi0: np.ndarray,
    n_eig: int,
    dt: float,
    tol: float,
    maxiter=100,
) -> tuple[float, xr.DataArray]:
    """Finds the first non-linear eigenstates of the Hamiltonian by imaginary time propagation

    Args:
        H0 (sps.dia_matrix): The linear Hamiltonian
        g (np.ndarray): The interaction array
        psi0 (np.ndarray): An initial guess for the states.
        n_eig (int): The number of eigenstates to find.
        dt (float): The time-step length.
        tol (float, optional): Tolerance for convergence.
        maxiter (int, optional): Maximum number of iterations. Defaults to 100.

    Returns:
        tuple[float,xr.DataArray]: The energy and mode profile of the ground state
    """

    norms = [np.linalg.norm(
        psi0[:,i]
    ) for i in range(n_eig)]  # Get the norm of psi0, to be conserved during propagation

    N = psi0.shape[0]
    E_list = [[]]*n_eig
    psi_final = np.zeros_like(psi0)
    
    proj = [] # Initialize the projector matrix in a dense structure
    
    for u in range(n_eig):
        
        psi_i = psi0[:,u]
        dt_i, E, psi_f = oneStep(H0, g, dt, psi_i, norms[u], tol, proj)  # Take a first step to initialize
        
        # Iterate until converged or maximum number of iteration attainded
        count = 1            
        while distance(psi_f, psi_i) > tol and count < maxiter:

            psi_i = psi_f
            dt_i, E, psi_f = oneStep(H0, g, dt_i, psi_i, norms[u], tol, proj)
            count += 1
            # E_list[u] += [E]
            
        if count >= maxiter - 1:
            print(
                "maximum number of iterations reached, the returned state might not be converged"
            )

        E_list[u] = E
        # psi_f *= np.exp(-1j*np.angle(psi_f[N//2]))
        psi_final[:,u] = psi_f
        proj += [normalize(psi_f, 1)]

    return E_list, psi_final


def subselect(indexes: tuple[int], ptype: str, allcoords: dict) -> dict[float]:
    """Create a subselction from allcoords based on a keyword 'ptype'

    Args:
        indexes (tuple[int]): The indexes of the values to select in each coordinate array.
        ptype (str): The keyword to select the subselection, can be 'potential', 'alpha', 'interaction', 'reciprocal', 'population' or 'coupling'.
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


class NLEigve(FDSolver):
    """The NLEigve class uses an imaginary time propagation method to find non-linear eigenvectors of the Gross-Pitaevskii equation. 
    It uses a RKF45 propagation method and is built on the Solver class from the 'bloch_schrodinger' package. See the dedicated tutorial 
    for more informations."""

    def __init__(
        self,
        potentials: Union[Potential, list[Potential]],
        gs: Union[Union[paramType, list[paramType]], Union[Potential, list[Potential]]],
    ):
        """Instantiate the solver.

        Args:
            potentials (Union[Potential,list[Potential]]): The potential felt by each field. They must all be defined on the same grid.
            A single potential can also be passed for a scalar equation.
            alphas (Union[Union[float, xr.DataArray], list[Union[float, xr.DataArray]]]): The kinetic energy coefficient hbar²/2m for each field.
            A single coefficient can be passed for a scalar equation.
            gs (Union[Potential, list[Potential]]]): The interaction energy for each field, given as a potential object.
            A single object can be passed for a scalar equation.

        Raises:
            ValueError: Not the same number of potentials and kinetic terms given.
        """

        super().__init__(potentials, 1/2)

        if self.nb == 1 and isinstance(gs, Potential):
            self.gs = [gs]
        else:
            self.gs = gs

        # Adding the interaction terms to the list of coordinates
        for g in self.gs:
            coords_g = {
                dim: ["interaction", g.V.coords[dim]]
                for dim in g.V.dims
                if dim not in ["a1", "a2"]
            }
            self.allcoords.update(coords_g)
            
        self.interaction_vector()

    def interaction_vector(self):
        """prepare the interaction term in a shape that can be sliced efficiently"""

        # flattening the potentials
        gflats = [g.V.stack(idiag=["a1", "a2"]) for g in self.gs]

        # Expanding each potential dimensions to the full parameter space, composed of all dimensions of each potential
        for i, g in enumerate(gflats):
            g = g.drop_vars(["idiag", "a1", "a2"]).assign_coords(
                idiag=("idiag", np.arange(i * self.np, (i + 1) * self.np))
            )
            for d, coords in self.allcoords.items():
                if coords[0] == "interaction":
                    if d not in g.dims:
                        V = g.expand_dims({d: coords[1]})
            gflats[i] = g

        # data in the shape (...,self.n) with the first dimensions being all the parameter dimensions of potentials
        self.interaction_data = xr.concat(gflats, dim="idiag")

    
    def solve(
        self,
        n_eig: int,
        population: Union[float, xr.DataArray] = 1,
        tol: float = 1e-12,
        maxiter=1000,
        phase0: tuple[float, float, int] = (0.01, 0.01, 0),
        parallel: bool = False,
        skip_guess: bool = False,
        n_cores:int = -1
    ) -> tuple[xr.DataArray]:
        """Find the 'n_eig' first non-linear eigenstates of the Hamiltonians, using an imaginary time propagation initialized with the eigenstates of the linear Hamiltonian H0.

        Args:
            n_eigva (int): The number of eigenvalues and vectors to find.
            population (Union[float,xr.DataArray], optional): The number of atoms in the condensate. Defaults to 1.
            tol (float, optional): The tolerance for the adaptative time-step evolution. Defaults to 1e-8.
            maxiter (int, optional): Maximal number of iterations for convergence. Defaults to 1000.
            phase0 (tuple[float,float,int], optional): Where to fix the phase of the ground state to 0, in the (a1,a2,field) basis. Defaults to (0.01,0.01,0).
            parallel(bool, optional): Wheter to use the parallel solver, this involve some overhead, so do not use it for too small parameter spaces. default to False.
            skip_guess(bool, optional): Wheter to skip the initial linear Hamiltonian diagonalization. Can speed up the overall solver if the matrices are big enough. default to False
            n_cores (int, optional): The number of cores to use for the parallelized solver.

        Returns:
            tuple[xr.DataArray]: The energy and mode profiel of the ground state for all parameters.
        """

        if isinstance(population, xr.DataArray):
            for dim in population.dims:
                check_name(dim)
            coords_pops = {
                dim: ["population", population.coords[dim]] for dim in population.dims
            }
            self.allcoords.update(coords_pops)

        # Create empty DataArrays to store the eigenvalues and vectors
        eigva = self.initialize_eigva(n_eig)
        eigve = self.initialize_eigve(n_eig)

        # We are going to create all the linear Hamiltonian and find they lowest eigenvector first, before finding the ground states
        ## First we create a list of tuples that select a single value for each of the parameter dimensions

        indexes = [np.arange(len(coord[1])) for coord in self.allcoords.values()]
        indexGrid = np.meshgrid(*indexes, indexing="ij")
        indexGrid = [grid.reshape(-1) for grid in indexGrid]
        selections = [tup for tup in zip(*indexGrid)]

        if len(selections) == 0:
            selections = [()]
       
        potential_sels = []
        reciprocal_sels = []
        coupling_sels = [] 
        alpha_sels = [] 
        interaction_sels = [] 
        pop = []
        sels = []

        # Initializing the vector guess. The solver works better with a good guess for the lowest eigenvector
        X = np.random.rand(self.n, n_eig)

        # Initializing the progress bar
            # Looping over first the potential dimensions, then the alpha dimensions, then the reciprocal dimensions and finally the coupling dimensions.
        for indexes in selections:
            potential_sel = subselect(indexes, "potential", self.allcoords)
            alpha_sel = subselect(indexes, "alpha", self.allcoords)
            reciprocal_sel = subselect(indexes, "reciprocal", self.allcoords)
            coupling_sel = subselect(indexes, "coupling", self.allcoords)
            interaction_sel = subselect(indexes, "interaction", self.allcoords)

            potential_sels += [potential_sel]
            alpha_sels += [alpha_sel]
            reciprocal_sels += [reciprocal_sel]
            coupling_sels += [coupling_sel]
            interaction_sels += [interaction_sel]
            
            population_sel = subselect(indexes, "population", self.allcoords)
            # Aggregate the selection, to add to the coupling evaluation context
            sels += [{
                **potential_sel,
                **alpha_sel,
                **reciprocal_sel,
                **coupling_sel,
                **interaction_sel,
                **population_sel,
            }]  # aggregating all the values of each selections
            
            if isinstance(population, xr.DataArray):
                pop += [float(population.sel(population_sel).item())]
            else:
                pop += [population]
                                
        # Diagonalize a first linear hamiltonian to get a first guess
        e, X = eigsh(
            self.create_hamiltonian(
                potential_sels[0], alpha_sels[0], reciprocal_sels[0], coupling_sels[0]), 
            k=n_eig, 
            which="SA"
        )
        
        # defining the function to parallelize
        def f(pot_sel, alpha_sel, rec_sel, cou_sel, int_sel, pop):
            H0 = self.create_hamiltonian(pot_sel, alpha_sel, rec_sel, cou_sel)
            if not skip_guess:
                eigvals, eigvec = eigsh(H0, k=n_eig, v0=X[:, 0], which="SA")
            else:
                eigvec = np.ones((self.n, 1))
                eigvals = np.mean(np.abs(H0.diagonal()))
            
            # Computing an approximately good time step
            int_diag = self.interaction_data.sel(int_sel).data
            dt = max(2 * np.pi / 1000 / (eigvals[0] + max(np.mean(int_diag) * pop, 1000)), 1e-5)

            energ, eigvec = findStates(
                H0, int_diag, self.normalize(eigvec, pop) , n_eig , dt, tol=tol, maxiter=maxiter
            )
            
            return energ, eigvec
            
        
        zipped = zip(
            potential_sels, 
            alpha_sels, 
            reciprocal_sels, 
            coupling_sels, 
            interaction_sels, 
            pop
        )
        n_tot = len(potential_sels)
        print(f"Performing {n_tot} diagonalizations...")
        
        if parallel:
            parallel = Parallel(n_jobs=n_cores, return_as="list", verbose=5)
            results = parallel(delayed(f)(p,a,r,c, i, po) for p, a, r, c, i, po in zipped)
        else:
            results = []
            with tqdm(total=n_tot) as pbar:
                for p, a, r, c, i, po in zipped:
                    results += [f(p,a,r,c, i, po)]
                    pbar.update(1)

        print("storing the results")
        with tqdm(total=n_tot) as pbar:
            for i in range(n_tot):
                eigvals, eigvecs = np.array(results[i][0]), np.array(results[i][1])
                
                idx = eigvals.argsort()
                eigvals = eigvals[idx]
                eigvecs = eigvecs[:, idx]
                                       
                eigva.loc[sels[i]] = eigvals
                eigve.loc[sels[i]] = eigvecs
                pbar.update(1)
        
        eigva *= self.da1 * self.da2

        eigve = eigve.unstack(dim="component").rename("ground state")
        sel0 = dict(a1=phase0[0], a2=phase0[1], field=phase0[2])

        eigve = eigve * xr.ufuncs.exp(
            1j * xr.ufuncs.angle(eigve.sel(sel0, method="nearest"))
        )
        x = self.a1[0] * eigve.a1 + self.a2[0] * eigve.a2
        y = self.a1[1] * eigve.a1 + self.a2[1] * eigve.a2
        eigve = eigve.assign_coords(
            {
                "x": x,
                "y": y,
            }
        )

        return eigva.squeeze(), eigve.squeeze()


if __name__ == '__main__':
    
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
    
    gp = create_parameter('g', np.linspace(0,1000,14))
    g = Potential(
        [[lx, 0], [0, ly]],
        resolution = (nx, ny),
        v0 = gp
    )
    
    pot.set(
        pot.x**2 * omx**2 / 2 + pot.y**2 * omy**2/2
    )
    
    
    foo = NLEigve(
        pot, g
    )
    
    eigva, eigve = foo.solve(
        10, 4, tol = 1e-8, maxiter = 5000, parallel=True
    )
    
    # print(eigve)
    
    #%%
    import matplotlib.pyplot as plt
    from bloch_schrodinger.plotting import plot_eigenvector, plot_cuts
    plot_eigenvector(
        [[abs(eigve)**2, eigve.real]], [[pot, pot]], [['amplitude', 'real']]
    )
    plt.show()
    #%%
    plot_cuts(eigva, 'g')
    
    
# %%
