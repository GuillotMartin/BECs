import os
from collections.abc import Callable

import numpy as np
import scipy.fft
import xarray as xr
from bloch_schrodinger.fdsolver import FDSolver, check_name
from bloch_schrodinger.potential import Potential
from scipy.fft import fftfreq

from BECs.backend import array_module, fftn, ifftn


def use_mkl_fft(enable: bool = True) -> bool:
    """Route scipy's transforms through MKL's DFTI, when it is available.

    The split-step solvers are memory-bandwidth-bound, and the transforms are the largest single
    item in the step. MKL's DFTI is roughly twice pocketfft's speed on the sizes this package
    works at -- measured on a 512^2 complex128 grid, one ifftn(fftn(x)*f) goes from 2.81 ms to
    1.03 ms -- and agrees with it to 5.7e-16, well below any tolerance the adaptive controllers
    are ever run at.

    This is registered as a *global* backend rather than threaded through the call sites, so that
    one call covers every fftn/ifftn in the package: the two solvers here, 'KineticStep', and the
    rescaling solver's transforms. 'only=False' leaves an unsupported call falling back to
    pocketfft rather than raising.

    Set BECS_NO_MKL_FFT in the environment to skip this, which is also what the benchmarks do when
    comparing the two backends.

    MKL's FFT domain is pinned to one thread, which sets only the default: every transform in this
    package passes an explicit 'workers', and that argument still wins. What the pin protects
    against is a call that does not pass one going away and opening a thread per core inside a
    joblib worker that is already one of many processes. Measured on a 512^2 grid, one
    ifftn(fftn(x)*f), best of three interleaved sweeps:

        workers          1        4       10
        pocketfft    2.961    0.924    0.799 ms
        mkl free     1.825    0.745    0.595 ms
        mkl pinned   1.825    0.743    0.597 ms

    So the pin costs nothing where 'workers' is given, and MKL is ahead of pocketfft at every
    thread count. Only the FFT domain is touched; BLAS keeps whatever threading it had.

    ONE BEHAVIOURAL CAVEAT, measured rather than assumed. MKL selects its transform kernels at
    initialization, and a freshly forked worker process does not always resolve that selection the
    way the long-lived parent did. The result is that a sweep run over several worker processes can
    differ from the identical sweep run serially, by about 2e-15 relative and growing slowly with
    the number of steps. It is not a threading effect -- the transform is bit-identical across
    thread counts and across argument alignment, and 'parallel=True' with a single job, which
    joblib runs in-process, reproduces the serial result exactly. Pocketfft has no such split.

    That is far below any tolerance these solvers are run at, but it does mean serial and parallel
    results are no longer bit-comparable. Set BECS_NO_MKL_FFT to restore pocketfft, which is what
    the benchmarks' bit-exactness checks want and what to reach for when a change is supposed to be
    pure redundancy elimination and has to be proved so.

    Args:
        enable (bool, optional): False restores scipy's own backend. Defaults to True.

    Returns:
        bool: Whether MKL is now in use.
    """
    if not enable:
        # By name: passing the scipy.fft module itself is rejected, since the module is not one of
        # its own uarray backend objects.
        scipy.fft.set_global_backend("scipy", only=False)
        return False
    if os.environ.get("BECS_NO_MKL_FFT"):
        return False
    try:
        import mkl_fft.interfaces.scipy_fft as _mkl_fft
    except ImportError:
        return False
    try:
        import mkl

        mkl.domain_set_num_threads(1, domain="fft")
    except ImportError:
        pass  # without mkl-service the default stays MKL's own; 'workers' still decides per call
    scipy.fft.set_global_backend(_mkl_fft, only=False)
    return True


USING_MKL_FFT = use_mkl_fft()


def categories(entry: list) -> tuple[str, ...]:
    """The categories an 'allcoords' entry belongs to.

    An entry is [category, coord]. The category slot holds either a single keyword or a tuple of
    them, because one dimension can drive several parts of a run at once: the evolution time of a
    time-of-flight sequence parameterizes the potential *and* indexes the sampling times, and has
    to answer to both 'potential' and 't'. Reading the slot through this function lets the
    single-keyword form -- which is what every registration wrote before dimensions could be
    shared, and what 'allcoords' dictionnaries built elsewhere still write -- keep working as it is.

    Args:
        entry (list): One value of an 'allcoords' dictionnary, as [category, coord].

    Returns:
        tuple[str,...]: Every category the dimension answers to.
    """
    ptypes = entry[0]
    return (ptypes,) if isinstance(ptypes, str) else tuple(ptypes)


def register_coord(
    allcoords: dict, dim: str, ptype: str, coord: xr.DataArray
) -> None:
    """Register 'dim' in 'allcoords' under the category 'ptype'.

    When the dimension is already known, the category is *added* to its entry rather than
    replacing what is there. A plain 'allcoords.update' would drop the earlier keyword, and the
    selection asking for it would come back empty -- leaving its consumer the whole unselected
    array, which is how a shared dimension used to fail.

    The entry is modified in place, so a dimension that is already present keeps its position in
    the dictionnary. That order is the contract 'selections' and every positional write-back
    ('psi[*slic]', 'energies[*indexes]') rely on: re-inserting the key would scatter the results
    of a run across the parameter space.

    Args:
        allcoords (dict): The dictionnary to register into, modified in place.
        dim (str): The dimension name.
        ptype (str): The category to register it under.
        coord (xr.DataArray): The coordinate of the swept values.

    Raises:
        ValueError: If 'dim' is already registered with different coordinate values. These
        selections go by exact label, so two disagreeing coordinates cannot both be honoured, and
        the mismatch is worth reporting here rather than deep inside a later '.sel'.
    """
    entry = allcoords.get(dim)
    if entry is None:
        allcoords[dim] = [(ptype,), coord]
        return

    known, incoming = np.asarray(entry[1].data), np.asarray(coord.data)
    if known.shape != incoming.shape or not np.array_equal(known, incoming):
        raise ValueError(
            f"the dimension '{dim}' is already registered as "
            f"{'/'.join(categories(entry))} over {known}, which does not match the {incoming} "
            f"it is now being registered over as '{ptype}'. A dimension shared between several "
            "parts of a run must carry the same coordinate everywhere, as the selections go by "
            "exact label."
        )

    ptypes = categories(entry)
    if ptype not in ptypes:
        entry[0] = ptypes + (ptype,)


def density(psi: np.ndarray) -> np.ndarray:
    """|psi|^2, without the square root np.abs takes and then squares away again.

    Args:
        psi (np.ndarray): A wavefunction.

    Returns:
        np.ndarray: Its density, real-valued.
    """
    return psi.real**2 + psi.imag**2


def expi(a: np.ndarray) -> np.ndarray:
    """exp(1j*a) for a real array, without paying for a complex exponential.

    np.exp on a complex argument runs the full complex kernel -- an exponential of the real part
    and a sine and a cosine of the imaginary one -- even when the real part is known to be zero.
    Writing the cosine and sine straight into the two halves of the result skips the exponential
    and the multiply that goes with it, and on a 512^2 grid takes 4.8 ms where np.exp takes 9.5.

    The same two lines serve a device array: cupy takes 'out=' on a '.real' view exactly as numpy
    does, agrees with exp(1j*a) to the bit, and is a shade faster there too.

    Args:
        a (np.ndarray): A real array, on the host or on a device.

    Returns:
        np.ndarray: exp(1j*a), elementwise, on whichever of the two 'a' was on.
    """
    xp = array_module(a)
    out = xp.empty(a.shape, xp.complex128)
    xp.cos(a, out=out.real)
    xp.sin(a, out=out.imag)
    return out


def phase(dt: complex, a: np.ndarray) -> np.ndarray:
    """exp(1j*dt*a), on the fast path whenever the exponent is purely imaginary.

    Two things in this package can put a real part into the exponent, and both must fall back to
    np.exp or the result is silently wrong rather than merely slower:

      * an absorbing boundary layer. 'add_losses' adds 1j*gamma*mask to the potential, so V is
        complex and exp(1j*dt*V) has a decaying magnitude, which is the entire point of it.
      * imaginary time. 'SSFM.imaginary_time' makes dt complex, which is how the ground-state
        solvers turn this same propagator into a diffusive one.

    Where neither applies -- the common case -- the exponent is i times a real array and 'expi'
    halves the cost. The argument is formed as dt.real*a, which is bit-for-bit what (1j*dt*a) puts
    in its imaginary part, so the fast path returns exactly what the slow one would.

    Args:
        dt (complex): The time step, possibly complex.
        a (np.ndarray): The real-or-complex array multiplying it.

    Returns:
        np.ndarray: exp(1j*dt*a), elementwise.
    """
    if np.iscomplexobj(a) or complex(dt).imag != 0:
        return array_module(a).exp(1j * dt * a)
    return expi(complex(dt).real * a)


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
            cached = self._cache[key] = self.aliasing * phase(dt, self.k2 / 2)
        return cached

    def __call__(self, psi: np.ndarray, dt: complex) -> np.ndarray:
        """Propagate psi for a step dt through reciprocal space.

        The cached propagator follows the field onto whichever device it is on: the first call with
        a device array moves it once and keeps the moved copy, so the transfer is paid once per step
        size rather than once per step. The k-space product is dead on return, so the inverse
        transform consumes it.
        """
        xp = array_module(psi)
        factor = self.factor(dt)
        if array_module(factor) is not xp:
            factor = self._cache[complex(dt)] = xp.asarray(factor)
        f = fftn(psi, self.workers)
        f *= factor
        return ifftn(f, self.workers, overwrite_x=True)


class PotentialPhase:
    """The real-space half of a split-step, exp(1j dt V(t) / 2), cached when V does not move.

    The same reasoning as 'KineticStep', applied to the other factor. Building this exponential over
    a 512^2 grid costs about as much as a transform, and a Strang step contains four transforms and
    two exponentials -- so the exponentials are roughly a third of the step, not a rounding error.

    When the potential is static the array only depends on dt, and the adaptive controller settles
    on a handful of step sizes, so caching removes the cost outright. When it is not, there is
    nothing to cache: V is different at every t, and the exponential is built on each call.

    Like 'KineticStep' the cache belongs to the instance: it is keyed on dt alone and so is only
    valid for the one potential and grid it was built for.
    """

    def __init__(self, V: Callable, static: bool | None = None, maxsize: int = 8):
        """
        Args:
            V (Callable): The potential as a function of time, from 'PotentialT.make_Vt'.
            static (bool, optional): Whether V ignores its argument. Defaults to None, which reads
            the 'static' attribute that 'make_Vt' sets on the function it returns.
            maxsize (int, optional): How many phases to keep. Small, for the same reason as in
            'KineticStep': the controller reuses few step sizes and each entry is a full complex
            grid. Defaults to 8.
        """
        self.V = V
        self.static = getattr(V, "static", False) if static is None else static
        self.maxsize = maxsize
        self._cache: dict[complex, np.ndarray] = {}

    @staticmethod
    def _bare(V_t):
        """The potential as a plain array, host or device.

        A PotentialT hands back a DataArray, and every multiply against it would otherwise go
        through xarray's dispatch machinery for no benefit. Unwrapping by '.data' rather than by
        'np.asarray' is what makes this work on the GPU: cupy refuses to be silently converted to a
        host array, so 'np.asarray' would raise on a potential built against cupy.
        """
        return V_t.data if hasattr(V_t, "dims") else V_t

    def __call__(self, t: float, dt: complex) -> np.ndarray:
        """The phase rotation exp(1j dt V(t) / 2), from the cache when the potential allows it."""
        if not self.static:
            return phase(dt / 2, self._bare(self.V(t)))

        key = complex(dt)
        cached = self._cache.get(key)
        if cached is None:
            if len(self._cache) >= self.maxsize:
                self._cache.clear()
            cached = self._cache[key] = phase(dt / 2, self._bare(self.V(t)))
        return cached


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

        # storing all parameter coordinates from the potential and g. The solver will run on all of
        # them. Registering rather than updating, so that a dimension appearing in both -- an
        # interaction strength that the potential also depends on, say -- answers to both
        # categories instead of the second registration dropping the first.
        self.allcoords = {}
        for dim in potential.V.dims:
            if dim not in self.spatial_dims:
                register_coord(
                    self.allcoords, dim, "potential", potential.V.coords[dim]
                )

        if isinstance(g, xr.DataArray):
            for dim in g.dims:
                check_name(dim, self.n_dims)
                register_coord(self.allcoords, dim, "g", g.coords[dim])

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

    def kinetic_step(self, workers: int = 1, xp=np) -> KineticStep:
        """Build the cached kinetic propagator of this grid.

        One per run: the cache is keyed on the time step alone, so it is only valid for the grid it
        was built from.

        Args:
            workers (int, optional): Threads handed to scipy's transforms. Defaults to 1.
            xp (module, optional): The array module to build the reciprocal-space grids with, numpy
            or cupy. Moving them once here keeps every step's propagator on the same side as the
            field. Defaults to numpy.

        Returns:
            KineticStep: The propagator, callable as step(psi, dt).
        """
        return KineticStep(xp.asarray(self.k2), xp.asarray(self.aliasing), workers)

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
