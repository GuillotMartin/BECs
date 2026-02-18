import numpy as np
import xarray as xr
from typing import Union
from bloch_schrodinger.potential import Potential
from bloch_schrodinger.pwsolver import PWSolver
from tqdm import tqdm, trange
from scipy.sparse.linalg import eigsh
from joblib import Parallel, delayed
from numpy.linalg import inv
from scipy.fft import fftn, fftshift


def real(arr: xr.DataArray) -> xr.DataArray:
    return xr.apply_ufunc(np.real, arr)


def imag(arr: xr.DataArray) -> xr.DataArray:
    return xr.apply_ufunc(np.imag, arr)

class NLPWSolver(PWSolver):

    def __init__(
        self,
        potential: Potential,
        alpha: float,
        g: Potential,
        E_lim: float = None
    ):

        super().__init__(potential, alpha, E_lim)
        self.compute_fg()
                

    def __repr__(self) -> str:
        return f"Non-linear plane wave Solver object \n E_lim = {self.E_lim:.3f} \n Expanded on {self.nGs} plane waves"

    def compute_fg(self):
        """Compute the fourier transform of the potential and interaction matrices"""

        ka1 = np.arange(self.na1) - (self.na1) // 2
        ka2 = np.arange(self.na2) - (self.na2) // 2

        self.pwka1 = xr.DataArray(ka1, coords={"pwka1": ka1})
        self.pwka2 = xr.DataArray(ka2, coords={"pwka2": ka2})

        self.pwkx = self.b1[0] * self.pwka1 + self.b2[0] * self.pwka2
        self.pwky = self.b1[1] * self.pwka1 + self.b2[1] * self.pwka2
        self.fg = xr.apply_ufunc(
            lambda arr: fftshift(
                fftn(arr, axes=[-2, -1], norm="forward"), axes=[-2, -1]
            ),
            self.g.V,
            input_core_dims=[["a1", "a2"]],
            output_core_dims=[["pwka1", "pwka2"]],
        )
        self.fg.assign_coords({"pwka1": self.pwka1, "pwka2": self.pwka2})
        
        
    def compute_mat(
        self,
        potential_sel: dict,
        reciprocal_sel: dict,
        interaction_sel: dict
    ) -> np.ndarray:
        """Construct the central equation matrix for the given parameter selection

        Args:
            potential_sel (dict): The parameters selection for the potential.
            reciprocal_sel (dict): The position in k-space.
            interaction_sel (dict): The parameters selection for the interaction term.

        Returns:
            np.ndarray
        """

        # The potential is a diagonal matrix, which we stored as a data array.
        potential_matrix = self.matV.sel(potential_sel).data

        if isinstance(self.kx, xr.DataArray):
            kxsel = {
                key: value
                for key, value in reciprocal_sel.items()
                if key in self.kx.dims
            }
            kx = float(self.kx.sel(kxsel).data)
        else:
            kx = self.kx

        if isinstance(self.ky, xr.DataArray):
            kysel = {
                key: value
                for key, value in reciprocal_sel.items()
                if key in self.ky.dims
            }
            ky = float(self.ky.sel(kysel).data)
        else:
            ky = self.ky
        kinetic_matrix = self.compute_kinetic([kx, ky])

        mat = potential_matrix + kinetic_matrix
        return mat

    def solve(
        self,
        n_eigva: int,
        parallel:bool = False,
        n_cores: int = -1,
        verbose:bool = True
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Solve the central equation.

        Args:
            n_eigva (int): The number of eigenvalues to compute.
            parallel (bool, optional): Wheter to parallelise the solver with joblib. Default to False.
            n_cores (int, optional): The number of cores to use for the solver, set to -1 to use all cores available. default to -1.
            verbose (bool, optional): Wheter to inform the user of the solver progress. Default to True.
        Returns:
            tuple[xr.DataArray]: the eigenvalues and the eigenvectors.
        """

        # Create empty DataArrays to store the eigenvalues and vectors
        eigva = self.initialize_eigva(n_eigva)
        eigve = self.initialize_eigve(n_eigva)

        # Create lists of dimensions. The dimensions are separated by type, for efficient Hamiltonian matrix construction.
        potential_dims = [
            dim for dim in self.allcoords if self.allcoords[dim][0] == "potential"
        ]
        reciprocal_dims = [
            dim for dim in self.allcoords if self.allcoords[dim][0] == "reciprocal"
        ]

        # Intializing indexes to loop over.
        potential_index, reciprocal_index = (
            [()],
            [(0, 0)],
        )

        # Stacking the dimensions of each type and replacing the associated index. If there is no dimensions of a given type,
        # the corresponding index is kept as the default defined above
        if len(potential_dims) > 0:
            eigva = eigva.stack(potdims=potential_dims)
            potential_index = eigva.potdims.to_index()
            eigve = eigve.stack(potdims=potential_dims)
        if len(reciprocal_dims) > 0:
            eigva = eigva.stack(recdims=reciprocal_dims)
            reciprocal_index = eigva.recdims.to_index()
            eigve = eigve.stack(recdims=reciprocal_dims)

        # The total number of matrix to diagonalize, used for the progress bar
        n_tot = len(eigva.sel(band=0).data.reshape(-1))

        # Initializing the progress bar
        sels = []
        pot_sels = []
        rec_sels = []
        # Looping over everything
        for pots in potential_index:
            for recs in reciprocal_index:
                # Selecting only one value for each potential dimensions, the selection will be empty if there is no potential dimensions
                potential_sel = (
                    dict(zip(potential_index.names, pots))
                    if hasattr(potential_index, "names")
                    else {}
                )
                reciprocal_sel = (
                    dict(zip(reciprocal_index.names, recs))
                    if hasattr(reciprocal_index, "names")
                    else {}
                )

                pot_sels += [potential_sel]
                rec_sels += [reciprocal_sel]
                
                sels += [
                    {
                        **potential_sel,
                        **reciprocal_sel,
                    }
                ]

        e, X = eigsh(self.compute_mat(potential_sel, reciprocal_sel), k=n_eigva, which="SA")

        def x(p_sel, r_sel):
            mat = self.compute_mat(p_sel, r_sel)
            return eigsh(mat, k=n_eigva, v0=X[:, 0], which="SA")

        if verbose: 
            print(f"Performing {n_tot} diagonalizations...")
            
        if parallel:
            parallel = Parallel(n_jobs=min(n_cores, n_tot), return_as="list", verbose = 5 if verbose else 0)
            results = parallel(delayed(x)(y, z) for y, z in zip(pot_sels, rec_sels))
        else:
            results = []
            if verbose:
                with tqdm(total=n_tot) as pbar:
                    for p_sel, r_sel in zip(pot_sels, rec_sels):
                        results += [x(p_sel, r_sel)]
                        pbar.update(1)
            else:
                for p_sel, r_sel in zip(pot_sels, rec_sels):
                        results += [x(p_sel, r_sel)]
        
        

        if verbose:
            print("storing the results")
            with tqdm(total=n_tot) as pbar:
                for i in range(n_tot):
                    eigvals, eigvecs = results[i][0], results[i][1]

                    idx = eigvals.argsort()
                    eigvals = eigvals[idx]
                    eigvecs = eigvecs[:, idx]

                    eigva.loc[sels[i]] = eigvals
                    eigve.loc[sels[i]] = eigvecs
                    pbar.update(1)
        else:
            for i in range(n_tot):
                    eigvals, eigvecs = results[i][0], results[i][1]

                    idx = eigvals.argsort()
                    eigvals = eigvals[idx]
                    eigvecs = eigvecs[:, idx]

                    eigva.loc[sels[i]] = eigvals
                    eigve.loc[sels[i]] = eigvecs

        if len(potential_dims) > 0:
            eigva = eigva.unstack(dim="potdims")
            eigve = eigve.unstack(dim="potdims")
        if len(reciprocal_dims) > 0:
            eigva = eigva.unstack(dim="recdims")
            eigve = eigve.unstack(dim="recdims")

        return eigva.squeeze(), eigve.squeeze()

    def compute_u(
        self, 
        eigve: xr.DataArray, 
        x: xr.DataArray = None, 
        y: xr.DataArray = None,
        vectorized:bool = False,
        phase0 = (0.01,0.01)
    ) -> xr.DataArray:
        """Compute the spatial shape of the eigenvectors from their plane-wave expression

        Args:
            eigve (xr.DataArray): The eigenvector in the plane wave basis
            x (xr.DataArray, optional): The x-grid over which to sample the eigenvector, if None, then the grid from the potential is chosen. Defaults to None.
            y (xr.DataArray, optional): The y-grid over which to sample the eigenvector, if None, then the grid from the potential is chosen. Defaults to None.
            vectorized (bool, optional): Wheter to sum over reciprocal vectors all at once or sequencially. The fully vectorized sum can be slow if the 
            resulting matrix is too large. Defaults to False.

        Returns:
            xr.DataArray
        """
        if x is None:
            x = self.potential.x - (self.potential.a1[0] + self.potential.a2[0])/2
        if y is None:
            y = self.potential.y - (self.potential.a1[1] + self.potential.a2[1])/2
        
        if vectorized:
            u = (
                eigve * np.exp(1j * (eigve.pwkx * x + eigve.pwky * y))
            ).sum('g')
            print(end = '\r')
        else:
            print('summing...')
            u = 0
            for ig in trange(eigve.sizes['g']):
                u += eigve[{'g':ig}] * np.exp(1j * (eigve.pwkx[{'g':ig}] * x + eigve.pwky[{'g':ig}] * y))
        
        # sel0 = dict(a1=phase0[0], a2=phase0[1])
        # u *= xr.ufuncs.exp(
        #     -1j * xr.ufuncs.angle(u.sel(sel0, method="nearest"))
        # ).conj()
        
        return u.conj() / (abs(u)**2).sum(['a1', 'a2'])**0.5
    
    


if __name__ == "__main__":
    import matplotlib.pyplot as plt  # noqa: E402
    from bloch_schrodinger.plotting import plot_eigenvector, get_template  # noqa: E402

    
    lam = 2
    kl = 2*np.pi / lam

    alpha = 0
    epsilon = 1
    eta = 1

    theta = np.pi/2 + np.pi/21
    # theta = np.pi/2

    a1 = np.array([1, 0])*2**0.5/2 * lam
    a2 = np.array([0, 1])*2**0.5/2 * lam

    b1 = a2 * np.pi * 2 / lam**2
    b2 = a1 * np.pi * 2 / lam**2

    klim = kl
    na1 = 64
    na2 = 64

    V0 = 20
    
    checker = Potential(
        unitvecs = [a1, a2],
        resolution = (na1, na2),
        v0 = 0,
    )

    xmy = (checker.x - checker.y)/2**0.5 - 1/4 * lam
    xpy = (checker.x + checker.y)/2**0.5

    checker.set(
        -V0/4 * abs(
            eta * ( np.exp(1j * kl * (xmy)) + epsilon * np.exp(-1j * kl * (xmy))) +
            np.exp(1j * theta) * (np.exp(1j * kl * xpy) + epsilon * np.exp(-1j * kl * xpy))
        )**2
    )
    
    g = checker.copy()
    g.clear()
    g.add(1)
    
    pw = NLPWSolver(checker, 1/2, g, 500)
    
    eigva, eigve = pw.solve(2)

    u = pw.compute_u(eigve)
    
    plot_eigenvector(
        [[abs(u)**2]], [[checker]], [['amplitude']]
    )
    plt.show()
