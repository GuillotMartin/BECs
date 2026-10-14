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
and synchronization costs. Above it the advantage climbs fast and then plateaus around 18x, where
the card is bandwidth-saturated. DEFAULT_MIN_POINTS sits at the crossover; the numbers will differ
on other hardware, but the shape of the curve -- a fixed overhead per step against work that grows
with the grid -- does not, so the threshold transfers better than a raw timing would.

Picking the device is only half of it, because a device that was usable when the sweep started can
stop being one halfway through. The second half of this module is about that:
'classify_device_error' sorts a failure into the kinds worth treating differently, 'DeviceLost' is
what a solver raises when the context is gone for good, 'guarded_run' is the one place that decides
between retrying and giving up, and 'device_warnings' says up front which failures a given card is
likely to hand out.
"""

from __future__ import annotations

import contextlib
import os
import subprocess

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


# --- surviving a device that fails in the middle of a sweep -------------------------------------

# The CUDA errors that destroy the context rather than merely failing the call that hit them.
# After one of these, every subsequent CUDA call in the process fails the same way, so there is
# nothing to retry: the context is gone and only a fresh interpreter brings it back.
#
# 'cudaErrorLaunchTimeout' is the one that turns up here in practice. The driver applies a
# watchdog to a GPU that is also driving a display and kills any kernel that holds the channel
# for more than a few seconds -- it surfaces in the kernel log as 'Xid 8' next to a line from
# 'krcWatchdog_IMPL'. It is why this section exists: a sweep that hits it at run 41 of 82 loses
# all 41 finished runs unless something has been writing them down.
_STICKY_ERRORS = (
    "cudaErrorLaunchTimeout",
    "cudaErrorLaunchFailure",
    "cudaErrorIllegalAddress",
    "cudaErrorIllegalInstruction",
    "cudaErrorMisalignedAddress",
    "cudaErrorECCUncorrectable",
    "cudaErrorHardwareStackError",
    "cudaErrorContextIsDestroyed",
    "cudaErrorDeviceUninitialized",
    "cudaErrorUnknown",
)

# The packages a device error can come out of. cupy spreads them over three.
_CUPY_ROOTS = frozenset({"cupy", "cupyx", "cupy_backends"})

_watchdog: bool | None = None


def classify_device_error(exc: BaseException) -> str | None:
    """What kind of device failure an exception is, or None when it is not one at all.

    Matched on the exception's module and message rather than against imported exception
    classes. cupy spreads those over three packages -- 'CUDARuntimeError' lives in
    'cupy_backends.cuda.api.runtime' and 'OutOfMemoryError' in 'cupy.cuda.memory' -- and some of
    them cannot be imported when the CUDA library behind them is missing, which is precisely the
    situation where a caller most wants to classify an error. Strings cost nothing and survive
    that, and they survive cupy renumbering anything too.

    The None case is the one that has to be right. An ordinary bug -- a ValueError from a bad
    parameter, a shape mismatch -- must come back as None, so that the guard around a run
    re-raises it untouched instead of dressing it up as a hardware problem and retrying it.

    Args:
        exc (BaseException): The exception to classify.

    Returns:
        str or None: "sticky" for an error that destroyed the CUDA context, "memory" for one a
        freed pool might get past, "transient" for any other device error, and None when the
        exception did not come from the device.
    """
    if (type(exc).__module__ or "").split(".")[0] not in _CUPY_ROOTS:
        return None

    name = type(exc).__name__
    text = f"{name}: {exc}"

    if any(err in text for err in _STICKY_ERRORS):
        return "sticky"
    if (
        name == "OutOfMemoryError"
        or "cudaErrorMemoryAllocation" in text
        or "CUFFT_ALLOC_FAILED" in text
    ):
        return "memory"
    return "transient"


class DeviceLost(RuntimeError):
    """The CUDA context died in the middle of a sweep and cannot be brought back in this process.

    Raised in place of the cupy error so that what reaches a notebook is what its reader needs:
    how far the sweep got, where the finished runs are, and the fact that the interpreter has to
    be restarted before anything will run on the GPU again. The original error is kept as
    '__cause__' for anyone who wants it.

    The message is built without touching the device. Every CUDA call fails once the context is
    gone, so asking the driver anything here -- whether the watchdog is on, how much memory is
    free -- would raise from inside the handler and bury the error it was trying to explain.
    """

    def __init__(
        self,
        index: int,
        n_runs: int,
        checkpoint=None,
        cause: BaseException | None = None,
    ):
        """
        Args:
            index (int): The index of the run that failed, 0-based.
            n_runs (int): How many runs the sweep holds in total.
            checkpoint (RunCheckpoint, optional): The store the finished runs went into.
            Defaults to None, for a sweep that was not checkpointing.
            cause (BaseException, optional): The device error underneath. Defaults to None.
        """
        self.index = index
        self.n_runs = n_runs
        self.checkpoint = checkpoint
        self.cause = cause
        super().__init__(self._message())

    def _message(self) -> str:
        reason = f" ({self.cause})" if self.cause is not None else ""
        lines = [
            f"The CUDA context was destroyed at run {self.index + 1}/{self.n_runs}{reason}.",
            "",
            "The error is sticky: nothing else can run on the GPU in this process.",
        ]

        if "cudaErrorLaunchTimeout" in str(self.cause or ""):
            lines[2:2] = [
                "This GPU also drives the display, so the driver kills any kernel that holds it",
                "for more than a few seconds (it appears in the kernel log as Xid 8).",
                "",
            ]

        lines.append("")
        if self.checkpoint is not None:
            lines += [
                f"{self.index} finished run(s) are saved in {self.checkpoint.path}/.",
                "Restart the interpreter, then re-run the same solve() call with the same",
                "checkpoint= to pick up where this one stopped.",
            ]
        else:
            lines += [
                "Nothing was saved: this sweep was not checkpointing. Restart the interpreter",
                "and pass checkpoint='some/directory' to solve() so that the next attempt keeps",
                "each run as it lands and can be resumed.",
            ]
        return "\n".join(lines)


def watchdog_enabled() -> bool:
    """Whether the device kills kernels that run too long, which it does when it drives a display.

    'kernelExecTimeoutEnabled' is the driver saying that an X server is on this GPU and that a
    compute kernel therefore shares it with the desktop, under a watchdog of a few seconds. A
    run can be perfectly sized for the card and still die on it, so it is worth saying up front.

    The answer is cached: it cannot change inside a process, and once a sticky error has landed
    the query itself would raise.

    Returns:
        bool: True when long kernels will be killed, False when the device is free of a display
        or when there is no usable device to ask.
    """
    global _watchdog
    if _watchdog is not None:
        return _watchdog

    ok, _ = gpu_available()
    if not ok:
        _watchdog = False
        return _watchdog

    try:
        import cupy as cp

        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        _watchdog = bool(props.get("kernelExecTimeoutEnabled", 0))
    except Exception:
        _watchdog = False
    return _watchdog


def device_warnings(device: str) -> list[str]:
    """The things worth knowing about the device before a long sweep starts, if any.

    Both of these are cheap to check and expensive to discover at the far end of an overnight
    run: a watchdog that will kill the kernels, and a card most of whose memory is already held
    by something else -- very often a Jupyter kernel left open from an earlier session, which
    keeps its allocation until the process exits.

    Args:
        device (str): "cpu" or "gpu", as returned by 'resolve_device'.

    Returns:
        list[str]: One line per warning, ready to print. Empty when there is nothing to say.
    """
    if device != "gpu":
        return []

    lines = []
    if watchdog_enabled():
        lines.append(
            "  warning: this GPU also drives the display, so kernels that hold it for more "
            "than a\n           few seconds are killed (cudaErrorLaunchTimeout / Xid 8). The "
            "error destroys\n           the CUDA context, so pass checkpoint=... to make a long "
            "sweep resumable."
        )

    try:
        import cupy as cp

        free, total = cp.cuda.runtime.memGetInfo()
    except Exception:
        return lines

    if total and free < total / 3:
        lines.append(
            f"  warning: only {free / 1024**3:.2f} of {total / 1024**3:.2f} GB is free on the "
            "device.\n           'BECs.backend.gpu_report()' shows which processes hold the rest."
        )
    return lines


def gpu_report() -> str:
    """How much device memory is free, and which processes are holding the rest.

    A crashed or simply forgotten interpreter keeps every byte it allocated until it exits, and
    on a single-card machine that is usually why the next run does not fit. The per-process
    table comes from 'nvidia-smi', since neither cupy nor the CUDA runtime can see another
    process's allocations.

    Returns:
        str: A few lines, ready to print.
    """
    ok, why = gpu_available()
    if not ok:
        return f"no usable GPU: {why}"

    import cupy as cp

    free, total = cp.cuda.runtime.memGetInfo()
    lines = [
        f"{why}: {free / 1024**3:.2f} GB free of {total / 1024**3:.2f} GB"
        + (", kernels are watchdog-limited" if watchdog_enabled() else "")
    ]

    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory,process_name",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except Exception as exc:
        lines.append(f"  (per-process usage unavailable: {type(exc).__name__})")
        return "\n".join(lines)

    for row in out.splitlines():
        pid, used, name = (field.strip() for field in row.split(",", 2))
        lines.append(f"  pid {pid:>8}  {used:>10}  {name}")
    return "\n".join(lines)


def free_pool() -> None:
    """Hand cupy's cached blocks back to the driver, and do nothing at all on the CPU.

    cupy keeps freed device memory in a pool rather than returning it, which is what makes the
    inner loop cheap but also means an allocation can fail while plenty of memory is held
    unused -- fragmented, or reserved in the wrong block sizes. Emptying the pool is the one
    thing worth doing before retrying an allocation that failed.
    """
    try:
        import cupy as cp
    except ImportError:
        return

    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        # Only ever called while recovering from another error; failing here would replace it.
        pass


def guarded_run(fn, args, index: int, n_runs: int, checkpoint=None):
    """Call 'fn(*args)' for one run of a sweep, translating a device failure into something useful.

    A sweep is a long serial loop over independent runs, and the three ways it can fail want three
    different answers. An ordinary bug is re-raised untouched, because hiding it behind a retry is
    how a wrong result gets saved. A failed allocation is retried once on an emptied pool, since
    cupy holds freed blocks back and fragmentation is a real and recoverable cause. And a destroyed
    context is turned into 'DeviceLost', which is fatal but says so in terms the caller can act on:
    it cannot be retried here at any cost, because every later CUDA call in this process fails too.

    Args:
        fn (Callable): The function that performs one run.
        args (tuple): Its arguments.
        index (int): The run's index in the sweep, used only in the error message.
        n_runs (int): How many runs the sweep holds, likewise.
        checkpoint (RunCheckpoint, optional): The store holding the runs finished so far, named in
        the error so that the caller is told where to resume from. Defaults to None.

    Returns:
        Whatever 'fn' returns.

    Raises:
        DeviceLost: If the CUDA context was destroyed.
        MemoryError: If the device ran out of memory twice, the second time on an emptied pool.
    """
    try:
        return fn(*args)
    except Exception as exc:
        kind = classify_device_error(exc)
        if kind is None:
            raise
        if kind != "memory":
            raise DeviceLost(index, n_runs, checkpoint, exc) from exc

    # --- one retry, on an emptied pool ---
    free_pool()
    try:
        return fn(*args)
    except Exception as exc:
        if classify_device_error(exc) == "sticky":
            raise DeviceLost(index, n_runs, checkpoint, exc) from exc
        if classify_device_error(exc) != "memory":
            raise
        free = free_device_memory()
        raise MemoryError(
            f"the device ran out of memory at run {index + 1}/{n_runs}, again after emptying "
            f"cupy's pool; {free / 1024**3:.2f} GB was free at that point. Run with device='cpu', "
            "or free the device -- 'BECs.backend.gpu_report()' shows which processes are holding "
            "it, and an idle interpreter keeps everything it ever allocated."
        ) from exc
