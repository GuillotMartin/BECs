"""Choosing between the CPU and the GPU, and the array module that goes with the choice.

The solvers are written against a single array module, held as 'consts["xp"]' and reaching
everything through the same numpy-shaped API. This module decides which one to hand them.

'Is there a GPU?' is not the whole question, and answering only that gets it wrong in both
directions. A GPU can be present and unusable -- a driver that never got loaded, a cupy installed
without the CUDA headers its kernels are compiled with -- and it can be present, usable, and still
the slower choice, because a small grid never covers the cost of launching the kernels. So
'resolve_device' asks four separate questions and reports which one decided.

The size threshold is measured rather than guessed. One full Strang step at complex128, this
package's inner loop, on an RTX 5070 against an i9-14900K with MKL:

    grid      32^2   64^2  128^2  256^2  512^2  1024^2  2048^2
    speedup   0.83x  0.82x  2.02x  6.38x  16.0x   18.0x   18.2x

Below about 128^2 the GPU is genuinely slower: the work per kernel is too small to hide the launch
and synchronization costs, and the frame equation forces a device-to-host copy of a couple of
scalars several times per step no matter how big the grid is. Above it the advantage climbs fast
and then plateaus around 18x, where the card is bandwidth-saturated. DEFAULT_MIN_POINTS sits at
the crossover; the numbers will differ on other hardware, but the shape of the curve -- a fixed
overhead per step against work that grows with the grid -- does not, so the threshold transfers
better than a raw timing would.
"""

from __future__ import annotations

import contextlib
import os

import numpy as np
import scipy.fft

# The grid size, in total points, at which the GPU starts to be worth its overhead. 128^2.
DEFAULT_MIN_POINTS = 128 * 128

# Fraction of free device memory a run is allowed to ask for. The solver holds around ten live
# grids at once and cuFFT takes a workspace of its own, so the estimate the caller passes is a
# floor rather than a bound; leaving most of the card free keeps that from mattering.
DEFAULT_MEMORY_FRACTION = 0.5

_probe: tuple[bool, str] | None = None


def gpu_available(recheck: bool = False) -> tuple[bool, str]:
    """Whether this machine has a GPU that actually works, and why not when it does not.

    Five things have to hold, and each of them has been seen to fail on its own:

      * BECS_NO_GPU is not set. The escape hatch, for pinning a machine to the CPU without
        editing call sites -- comparing backends, or working on a shared card.
      * cupy is installed at all.
      * it can talk to the driver. A machine can have the card, the kernel module built and the
        package installed, and still report zero devices because the module was never loaded.
      * a device is present.
      * a kernel compiles and runs. cupy imports happily without the CUDA headers it needs to
        build its elementwise kernels, so the failure surfaces on the first arithmetic rather than
        on the import, which is far too late to be choosing a backend.

    The answer is cached, since it costs a kernel compile and cannot change inside a process.

    Args:
        recheck (bool, optional): Probe again instead of returning the cached answer. Defaults to
        False.

    Returns:
        tuple[bool, str]: Whether the GPU is usable, and a one-line reason either way.
    """
    global _probe
    if _probe is not None and not recheck:
        return _probe

    if os.environ.get("BECS_NO_GPU"):
        _probe = (False, "BECS_NO_GPU is set")
        return _probe

    try:
        import cupy as cp
    except ImportError as exc:
        _probe = (False, f"cupy is not installed ({exc})")
        return _probe

    try:
        n = cp.cuda.runtime.getDeviceCount()
    except Exception as exc:  # driver absent, module not loaded, version mismatch
        _probe = (False, f"no usable CUDA driver ({type(exc).__name__}: {exc})")
        return _probe

    if n < 1:
        _probe = (False, "cupy is installed but reports no CUDA device")
        return _probe

    try:
        # Deliberately an elementwise kernel and not just an allocation: allocation goes through
        # the driver, while this is what needs the toolkit headers and a compiler.
        if float((cp.arange(4, dtype=cp.float64) * 2).sum()) != 12.0:
            raise RuntimeError("arithmetic returned the wrong answer")
    except Exception as exc:
        _probe = (False, f"cupy cannot run a kernel ({type(exc).__name__}: {exc})")
        return _probe

    name = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"].decode()
    _probe = (True, f"{name}")
    return _probe


def array_module(a):
    """numpy or cupy, whichever owns the array.

    Args:
        a: Any array.

    Returns:
        module: The array module 'a' belongs to.
    """
    try:
        import cupy as cp
    except ImportError:
        return np
    return cp.get_array_module(a)


def get_xp(device: str):
    """The array module for a resolved device name.

    Args:
        device (str): "cpu" or "gpu", as returned by 'resolve_device'.

    Returns:
        module: numpy for "cpu", cupy for "gpu".
    """
    if device == "gpu":
        import cupy as cp

        return cp
    return np


def free_device_memory() -> int:
    """Free bytes on the current CUDA device, or 0 when there is no usable one."""
    ok, _ = gpu_available()
    if not ok:
        return 0
    import cupy as cp

    free, _total = cp.cuda.runtime.memGetInfo()
    return int(free)


def resolve_device(
    device: str | bool = "auto",
    n_points: int = 0,
    n_bytes: int = 0,
    min_points: int = DEFAULT_MIN_POINTS,
    memory_fraction: float = DEFAULT_MEMORY_FRACTION,
) -> tuple[str, str]:
    """Decide whether to run on the CPU or the GPU, and say why.

    Args:
        device (str or bool, optional): "auto" to decide here, "gpu"/True to demand the GPU,
        "cpu"/False to demand the CPU. Demanding the GPU where it is unusable raises rather than
        falling back, because a run that silently lands on the CPU after being asked for the GPU
        is a performance bug that hides for a long time. Defaults to "auto".
        n_points (int, optional): Points in one spatial grid, the quantity the crossover is in
        terms of. Defaults to 0, which skips the size test.
        n_bytes (int, optional): A run's estimated device footprint. Defaults to 0, which skips
        the memory test.
        min_points (int, optional): Below this the CPU wins. Defaults to DEFAULT_MIN_POINTS.
        memory_fraction (float, optional): How much of free device memory a run may ask for.
        Defaults to DEFAULT_MEMORY_FRACTION.

    Returns:
        tuple[str, str]: "cpu" or "gpu", and a one-line reason.

    Raises:
        RuntimeError: If the GPU was demanded explicitly and is not usable.
        ValueError: If 'device' is not one of the accepted values.
    """
    if device is True:
        device = "gpu"
    elif device is False:
        device = "cpu"
    if device not in ("auto", "cpu", "gpu"):
        raise ValueError(f"device must be 'auto', 'cpu' or 'gpu', got {device!r}")

    if device == "cpu":
        return "cpu", "asked for"

    ok, why = gpu_available()

    if device == "gpu":
        if not ok:
            raise RuntimeError(
                f"device='gpu' was asked for but the GPU is not usable: {why}. Pass device='auto' "
                "to fall back to the CPU, or device='cpu' to say so explicitly."
            )
        return "gpu", f"asked for ({why})"

    # --- device == "auto" ---
    if not ok:
        return "cpu", why

    if n_points and n_points < min_points:
        return "cpu", (
            f"grid of {n_points} points is below the {min_points}-point crossover, where kernel "
            "launch and synchronization cost more than the GPU saves"
        )

    if n_bytes:
        free = free_device_memory()
        if n_bytes > memory_fraction * free:
            return "cpu", (
                f"run needs about {n_bytes / 1024**3:.2f} GB but only {free / 1024**3:.2f} GB is "
                f"free on the device"
            )

    return "gpu", why


def fftn(a, workers: int = 1, **kwargs):
    """scipy's fftn, without offering 'workers' to a backend that has no CPU threads to spend.

    cuFFT's scipy backend takes 'overwrite_x' but not 'workers' -- the parallelism there is the
    device's, not the caller's -- and raises rather than ignoring it. Every transform in the
    package goes through here so that one call site serves both.

    Args:
        a: The array to transform.
        workers (int, optional): CPU threads, ignored on a device array. Defaults to 1.
        **kwargs: Passed through to scipy, e.g. 'overwrite_x' or 'norm'.

    Returns:
        The transform of 'a', on whichever of host or device it came from.
    """
    if array_module(a) is np:
        return scipy.fft.fftn(a, workers=workers, **kwargs)
    return scipy.fft.fftn(a, **kwargs)


def ifftn(a, workers: int = 1, **kwargs):
    """scipy's ifftn, with 'workers' dropped on the device. See 'fftn'."""
    if array_module(a) is np:
        return scipy.fft.ifftn(a, workers=workers, **kwargs)
    return scipy.fft.ifftn(a, **kwargs)


def fft_backend(device: str):
    """A context registering cuFFT for scipy's transforms on the GPU, and nothing on the CPU.

    scipy.fft dispatches through uarray and cupyx registers as a backend for it, so entering this
    is all that is needed to send every fftn/ifftn inside it to the device -- none of the call
    sites has to know which array module it is working on. On the CPU it is a null context, which
    leaves the MKL backend that 'BECs.spectral' registers globally in force.

    Args:
        device (str): "cpu" or "gpu".

    Returns:
        AbstractContextManager: The backend context to run inside.
    """
    if device != "gpu":
        return contextlib.nullcontext()
    import cupyx.scipy.fft as cufft

    return scipy.fft.set_backend(cufft)


def to_host(a):
    """Bring an array back to the host, whichever module it came from.

    Args:
        a: A numpy or cupy array.

    Returns:
        np.ndarray: The same data as a numpy array.
    """
    return a.get() if hasattr(a, "get") else np.asarray(a)
