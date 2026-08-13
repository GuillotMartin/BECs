import numpy as np
import xarray as xr
from bloch_schrodinger.fdsolver import FDSolver, check_name
from bloch_schrodinger.potential import Potential
from scipy.fft import fftfreq, fftn, ifftn


def density(psi: np.ndarray) -> np.ndarray:
    """|psi|^2, without the square root np.abs takes and then squares away again.

    Args:
        psi (np.ndarray): A wavefunction.

    Returns:
        np.ndarray: Its density, real-valued.
    """
    return psi.real**2 + psi.imag**2


class KineticStep:
    """The kinetic half of a split-step, with its propagator cached across time steps.

    Building exp(1j dt k^2 / 2) over the grid costs about three times a transform of the same array,
    so the propagator, and not the transform, is what a split-step solver spends most of its time on.
    It only depends on dt, and the adaptive controller settles on a handful of step sizes almost
    immediately -- a representative run used 4 distinct values over 246 steps -- so caching it removes
    nearly all of that cost. The anti-aliasing mask is folded into the cached factor, which also saves
    a multiply over the whole grid on every call.

    The cache belongs to the instance rather than to the module because two solvers on different grids
    may well be alive at once in a notebook, and they must not share propagators.

    Imaginary time needs no separate implementation: exp(-dt k^2 / 2) is this propagator at 1j*dt, so
    the ground state solver passes a complex step and gets the diffusive kernel back.
    """

    def __init__(
        self,
        k2: np.ndarray,
        aliasing: np.ndarray,
        workers: int = 1,
        maxsize: int = 8,
    ):
        """
        Args:
            k2 (np.ndarray): The squared norm of the wavevector at each point of reciprocal space.
            aliasing (np.ndarray): A high-k cut off mask for anti-aliasing.
            workers (int, optional): Threads handed to scipy's transforms. Worth raising for grids of
            512^2 and above; below that the transforms are too small to gain from it. Defaults to 1.
            maxsize (int, optional): How many propagators to keep. Small: the controller reuses a few
            step sizes, and each entry is a full complex grid. Defaults to 8.
        """
        self.k2 = k2
        self.aliasing = aliasing
        self.workers = workers
        self.maxsize = maxsize
        self._cache: dict[complex, np.ndarray] = {}

    def factor(self, dt: complex) -> np.ndarray:
        """The propagator for a step dt, built on a miss and kept for the next call."""
        key = complex(dt)
        cached = self._cache.get(key)
        if cached is None:
            if len(self._cache) >= self.maxsize:
                self._cache.clear()
            cached = self._cache[key] = self.aliasing * np.exp(1j * dt * self.k2 / 2)
        return cached

    def __call__(self, psi: np.ndarray, dt: complex) -> np.ndarray:
        """Propagate psi for a step dt through reciprocal space."""
        return ifftn(
            fftn(psi, workers=self.workers) * self.factor(dt), workers=self.workers
        )


class SpectralSolver(FDSolver):
    """The common setup of the split-step Fourier solvers of this package.

    These solvers propagate in a plane-wave basis, so none of FDSolver's finite-difference machinery
    (the stencil, the hopping matrices, the gradient and the laplacian) is of any use to them, and
    building it is by far the most expensive part of FDSolver.__init__. This class therefore
    deliberately does not call it. It sets instead, by hand, the handful of attributes that the
    inherited helpers ('initialize_eigva', 'initialize_eigve' and 'normalize') actually read, plus
    the reciprocal space grid that the propagation steps need. Any new attribute those helpers start
    reading has to be added here too.
    """

    def __init__(
        self,
        potential: Potential,
        g: float | xr.DataArray = 0,
        cutoff: float = 1 / 3,
    ):
        """Set up the grid, the reciprocal space and the parameter space of a spectral solver.

        Args:
            potential (Potential): The potential landscape, which must describe an axis-aligned
            rectangular grid. The solver will iterate over each of its additional dimensions.
            g (Union[float, xr.DataArray], optional): Interaction strength term. Can be passed as an
            array over which to iterate. Defaults to 0.
            cutoff (float, optional): The anti-aliasing cut-off, as a fraction of the full extent of
            reciprocal space. Defaults to 1/3, the usual two-thirds rule.

        Raises:
            ValueError: If the potential grid is not rectangular and axis-aligned.
        """

        self.potential = potential
        self.potentials = [potential]  # for compatibility with the FDSolver helpers
        self.g = g

        self.nb = 1  # these solvers handle scalar equations only
        self.n_dims = potential.n_dims
        self.spatial_dims = [f"a{i + 1}" for i in range(self.n_dims)]

        self.a = potential.a  # The lattice vectors
        if not np.allclose(self.a, np.diag(np.diag(self.a))):
            raise ValueError(
                "This solver only works for axis-aligned rectangular grids, "
                f"got unit vectors {self.a.tolist()}"
            )

        self.a_coords = [potential.V.coords[dim] for dim in self.spatial_dims]
        self.n_a = [potential.V.sizes[dim] for dim in self.spatial_dims]  # discretization per axis
        self.np = int(np.prod(self.n_a))  # Number of mesh sampling points
        self.n = self.np * self.nb
        self.da = potential.da  # length increments along each axis

        # storing all parameter coordinates from the potential and g. The solver will run on all of them.
        self.allcoords = {}
        self.allcoords.update(
            {
                dim: ["potential", potential.V.coords[dim]]
                for dim in potential.V.dims
                if dim not in self.spatial_dims
            }
        )

        if isinstance(g, xr.DataArray):
            for dim in g.dims:
                check_name(dim, self.n_dims)
            self.allcoords.update({dim: ["g", g.coords[dim]] for dim in g.dims})

        # --- Reciprocal space ---
        self.ks = np.meshgrid(
            *[
                fftfreq(self.n_a[i], self.da[i]) * 2 * np.pi
                for i in range(self.n_dims)
            ],
            indexing="ij",
        )
        self.k2 = sum(k**2 for k in self.ks)  # every propagation step only ever wants k squared

        # Cut off the high-k corner of the grid, as a fraction of the full reciprocal extent
        k_full = max(2 * np.pi / da for da in self.da)
        self.aliasing = np.where(self.k2**0.5 > cutoff * k_full, 0, 1)

        # Definitions for retrocompatibility
        if self.n_dims == 2:
            self.dx, self.dy = self.da
            self.kx, self.ky = self.ks
            self.na1, self.na2 = self.n_a
            self.a1, self.a2 = self.a[0], self.a[1]
            self.a1_coord, self.a2_coord = self.a_coords

    def kinetic_step(self, workers: int = 1) -> KineticStep:
        """Build the cached kinetic propagator of this grid.

        One per run: the cache is keyed on the time step alone, so it is only valid for the grid it
        was built from.

        Args:
            workers (int, optional): Threads handed to scipy's transforms. Defaults to 1.

        Returns:
            KineticStep: The propagator, callable as step(psi, dt).
        """
        return KineticStep(self.k2, self.aliasing, workers)

    def __repr__(self) -> str:
        shape = {dim: len(self.allcoords[dim][1].data) for dim in self.allcoords}
        return (
            f"{type(self).__name__} ({self.n_dims}D): resolution {tuple(self.n_a)} \n"
            f" dimensions: {shape}"
        )

    def normalize(
        self, psi: xr.DataArray | np.ndarray, norm: float = 1
    ) -> xr.DataArray | np.ndarray:
        """Normalize a wavefunction to a specified population in real-space units.

        Args:
            psi (Union[xr.DataArray, np.ndarray]): The wavefunction to normalize.
            norm (float, optional): The population it should hold. Defaults to 1.

        Returns:
            Union[xr.DataArray, np.ndarray]: The normalized wavefunction.
        """
        normed = psi / np.sum(np.abs(psi) ** 2) ** 0.5
        return normed * (norm / self.potential.get_dS()) ** 0.5

    def selections(self) -> list[tuple[int]]:
        """Build the list of index tuples covering the whole parameter space, one entry per point.

        Returns:
            list[tuple[int]]: The index of every point of the parameter space, in the order of
            'allcoords'. A single empty tuple if there is no parameter dimension at all.
        """
        indexes = [np.arange(len(coord[1])) for coord in self.allcoords.values()]
        indexGrid = np.meshgrid(*indexes, indexing="ij")
        indexGrid = [grid.reshape(-1) for grid in indexGrid]
        selections = [tup for tup in zip(*indexGrid)]

        return selections if len(selections) > 0 else [()]

    def phase_reference(self, phase0: tuple[float] | None = None) -> dict:
        """Build the selection at which the phase of the wavefunctions is fixed to zero.

        Args:
            phase0 (tuple[float], optional): The position at which to fix the phase, in the
            (a1, ..., an, field) basis. Defaults to (0.01, ..., 0.01, 0).

        Returns:
            dict: The selection, ready to be passed to xr.DataArray.sel.
        """
        pos0 = phase0 if phase0 is not None else (0.01,) * self.n_dims + (0,)
        sel0 = {self.spatial_dims[d]: pos0[d] for d in range(self.n_dims)}
        sel0["field"] = pos0[self.n_dims]
        return sel0

    def assign_cartesian(self, psi: xr.DataArray) -> xr.DataArray:
        """Attach the cartesian coordinates ('x', 'y', 'z') of the potential to a wavefunction array.

        Args:
            psi (xr.DataArray): The array to label.

        Returns:
            xr.DataArray: The same array, with its cartesian coordinates.
        """
        return psi.assign_coords(
            {
                self.potential.coord_names[d]: self.potential.coords[d]
                for d in range(self.n_dims)
            }
        )
