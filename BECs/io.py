"""Opening saved runs without materialising them.

A sweep saved by these solvers is large -- a 512^2 grid sampled 100 times over ten parameter
values is 4.2 GB of field -- and almost every use of it displays or reduces one frame at a time.
The functions here open such a file lazily, so that the arithmetic a caller writes over the whole
array is recorded rather than performed, and only the frames actually looked at are ever read.

The saving convention is two files, one for the real part and one for the imaginary, because
netCDF has no complex type. Recombining them is where the cost hides: the obvious

    psi = xr.open_dataarray(real) + 1j * xr.open_dataarray(imag)

is an xarray binary operation, and that merges the two operands' coordinates. When the moving
cartesian coordinates are stored materialised -- 2.1 GB each for the sweep above -- merging means
computing and comparing both copies of them, which takes ten seconds and leaves the result eager.
Building the array from its parts instead skips the merge entirely and takes under a millisecond.

'RunCheckpoint' is the other direction: a place for a sweep to put each run as it finishes, rather
than holding the whole thing in memory and writing once at the end. A solver keeps nothing on disk
until its last run lands, so anything that ends the process early -- a CUDA context killed by the
driver's watchdog, an out-of-memory kill, a closed laptop -- costs every run that had finished.
With a checkpoint the cost is one run.
"""

import json
import os
import re
from pathlib import Path

import numpy as np
import xarray as xr
from bloch_schrodinger.frames import materialize

from BECs.backend import to_host

_LATTICE_DIM = re.compile(r"^a\d+$")


def frame_chunks(da: xr.DataArray) -> dict[str, int]:
    """One spatial frame per chunk: every lattice axis whole, everything else sliced to one.

    This is the shape almost every consumer wants. Plotting shows one frame per slider position;
    a transform along the spatial axes -- an FFT of each frame, say -- needs those axes contiguous
    and nothing else. Chunking finer than a frame would break both.

    Args:
        da (xr.DataArray): The array whose dimensions are to be chunked.

    Returns:
        dict[str, int]: A chunk specification for 'DataArray.chunk' or 'open_dataarray'.
    """
    return {
        dim: (da.sizes[dim] if _LATTICE_DIM.match(str(dim)) else 1) for dim in da.dims
    }


def open_result(
    real: str,
    imag: str | None = None,
    chunks: str | dict | None = "frame",
    frame: bool = False,
) -> xr.DataArray:
    """Open a saved run lazily, recombining its real and imaginary halves.

    Args:
        real (str): The file holding the real part, or the whole array if it is real-valued.
        imag (str, optional): The file holding the imaginary part. Defaults to None, which returns
        whatever 'real' holds unchanged.
        chunks (str or dict, optional): "frame" for one spatial frame per chunk, or any mapping
        'open_dataarray' accepts, or None to read eagerly. Defaults to "frame".
        frame (bool, optional): Whether to write out cartesian coordinates the run stores as a
        product of factors (see bloch_schrodinger.frames.materialize), for the consumers that
        want them as plain coordinates. The rebuilt coordinates stay lazy, so this does not undo
        the point of opening lazily. Defaults to False.

    Returns:
        xr.DataArray: The run, dask-backed unless 'chunks' is None.

    Raises:
        ValueError: If the two halves do not have the same dimensions.
    """
    opened = xr.open_dataarray(real, chunks=None if chunks is None else {})
    spec = frame_chunks(opened) if chunks == "frame" else chunks
    if spec is not None:
        opened = opened.chunk(spec)

    if imag is None:
        out = opened
    else:
        other = xr.open_dataarray(imag, chunks=None if chunks is None else {})
        if spec is not None:
            other = other.chunk(spec)
        if opened.dims != other.dims:
            raise ValueError(
                f"the two halves disagree on their dimensions: {opened.dims} and {other.dims}"
            )
        # Built from the parts rather than as 'opened + 1j * other'. The addition is an xarray
        # binary operation, which aligns and merges the operands' coordinates; where those carry
        # large materialised coordinates that means computing and comparing gigabytes, and the result
        # comes back eager. Here the coordinates are taken from one side and never compared.
        out = xr.DataArray(
            opened.data + 1j * other.data,
            dims=opened.dims,
            coords=opened.coords,
            attrs=opened.attrs,
            name=opened.name,
        )

    if frame:
        out = materialize(out)
    return out


def as_lazy(da: xr.DataArray, chunks: str | dict = "frame") -> xr.DataArray:
    """Wrap an in-memory result so that whole-array arithmetic over it is deferred.

    For an array the solver has just returned, 'chunk' is a zero-copy wrap: it does not move or
    copy the data, it only puts a dask graph in front of it. What that buys is that the expression
    a caller writes -- 'abs(psi)**2' over a 4.2 GB sweep, say -- stops being a full traversal and a
    2.1 GB allocation, and becomes a graph that is evaluated one frame at a time by whoever
    actually looks at a frame.

    Args:
        da (xr.DataArray): An in-memory result.
        chunks (str or dict, optional): "frame", or a mapping. Defaults to "frame".

    Returns:
        xr.DataArray: The same data, dask-backed.
    """
    return da.chunk(frame_chunks(da) if chunks == "frame" else chunks)


class RunCheckpoint:
    """A directory holding the finished runs of a sweep, so that an interrupted one can resume.

    One '.npz' per run, named by the run's index in the sweep, plus a 'meta.json' describing the
    sweep the directory belongs to. A solver asks 'done' which indices it can skip, loads those
    from here, and calls 'save' on each run it computes.

    The store is deliberately flat and dumb. It is written while something is going wrong -- the
    whole point is to survive a process that is about to die -- so it does nothing that could
    itself fail halfway: no appending to a single growing file, no index to keep consistent, and
    every write lands through a temporary name and an atomic rename, so a crash in the middle of
    one leaves the previous state rather than a truncated file that the next resume would trust.

    Arrays go in uncompressed. These are full complex grids and compression would cost more time
    than the write does, but the size is worth knowing before switching it on: the 3D Talbot
    sweep, at 82 runs of two 512x512x32 complex128 samples, is about 22 GB. The directory is
    disposable once the result has been written out properly, and can just be deleted.

    Attributes:
        path (Path): The directory the runs are kept in.
        n_runs (int): How many runs the sweep holds in total.
    """

    def __init__(self, path, n_runs: int, signature: dict):
        """
        Args:
            path (str or Path): The directory to keep the runs in. Created if it does not exist.
            n_runs (int): How many runs the sweep holds.
            signature (dict): Whatever identifies this sweep -- its shape, its dimensions, its
            time span. Stored on the first call and compared on every later one, so that
            pointing a changed sweep at an old directory raises instead of quietly mixing runs
            from two different calculations into one array.

        Raises:
            ValueError: If the directory already holds a different sweep.
        """
        self.path = Path(path)
        self.n_runs = int(n_runs)
        self.path.mkdir(parents=True, exist_ok=True)

        # Round-tripped through json before comparing, so that a tuple written last time and the
        # list it comes back as do not read as a difference.
        meta = json.loads(
            json.dumps({"n_runs": self.n_runs, "signature": signature}, sort_keys=True, default=str)
        )
        meta_file = self.path / "meta.json"

        if meta_file.exists():
            previous = json.loads(meta_file.read_text())
            if previous != meta:
                differing = sorted(
                    key
                    for key in set(previous.get("signature", {})) | set(meta["signature"])
                    if previous.get("signature", {}).get(key) != meta["signature"].get(key)
                )
                raise ValueError(
                    f"{self.path} holds a different sweep"
                    + (f" (differs in: {', '.join(differing)})" if differing else "")
                    + ". Point 'checkpoint' at a new directory, or delete this one to start over."
                )
        else:
            meta_file.write_text(json.dumps(meta, indent=2, sort_keys=True))

    def _run_file(self, index: int) -> Path:
        return self.path / f"run_{index:05d}.npz"

    def done(self) -> set[int]:
        """The indices of the runs already on disk.

        Returns:
            set[int]: Every index this store can hand back, which the caller can skip computing.
        """
        return {int(f.stem.split("_")[1]) for f in self.path.glob("run_[0-9]*.npz")}

    def save(self, index: int, *results) -> None:
        """Write one finished run.

        Args:
            index (int): The run's index in the sweep.
            *results: Whatever the solver's storing step wants back, in that order. Each one may
            be an array, a scalar, or a list of arrays -- the three shapes the solvers hand
            around -- and the structure is recorded alongside so that 'load' returns the same
            thing rather than a flattened pile.
        """
        arrays, struct = {}, []
        for k, result in enumerate(results):
            if isinstance(result, (list, tuple)):
                struct.append({"kind": "list", "n": len(result)})
                for j, item in enumerate(result):
                    arrays[f"a{k}_{j}"] = to_host(item)
            elif np.ndim(result) == 0:
                struct.append({"kind": "scalar"})
                arrays[f"a{k}"] = np.asarray(result)
            else:
                struct.append({"kind": "array"})
                arrays[f"a{k}"] = to_host(result)

        # Written beside the final name and renamed onto it: os.replace is atomic within a
        # filesystem, so a resume never sees a half-written run.
        target = self._run_file(index)
        tmp = target.with_suffix(".npz.tmp")
        with open(tmp, "wb") as handle:
            np.savez(handle, _struct=np.array(json.dumps(struct)), **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)

    def load(self, index: int) -> tuple:
        """Read one finished run back, in the shape 'save' was given it.

        Args:
            index (int): The run's index in the sweep.

        Returns:
            tuple: The results of that run.
        """
        with np.load(self._run_file(index), allow_pickle=False) as data:
            struct = json.loads(str(data["_struct"]))
            results = []
            for k, entry in enumerate(struct):
                if entry["kind"] == "list":
                    results.append([data[f"a{k}_{j}"] for j in range(entry["n"])])
                elif entry["kind"] == "scalar":
                    results.append(data[f"a{k}"].item())
                else:
                    results.append(data[f"a{k}"])
        return tuple(results)

    def discard(self) -> None:
        """Delete the saved runs and the directory, once the sweep that wrote them has finished.

        A finished sweep holds every run in the array it is about to return, so the copies here
        have become dead weight -- and at tens of gigabytes for a 3D sweep that is not weight
        worth leaving lying around. The solvers therefore call this on their way out, and only
        on their way out: a sweep that raised keeps its checkpoint, which is the whole point of
        having one.

        Only the files this class wrote are removed, and the directory itself only if nothing
        else is left in it. A checkpoint path can perfectly well be a directory with other
        things in it, and a sweep finishing is not a reason to delete someone else's data.
        """
        for f in self.path.glob("run_[0-9]*.npz*"):
            f.unlink(missing_ok=True)
        (self.path / "meta.json").unlink(missing_ok=True)
        try:
            self.path.rmdir()
        except OSError:
            pass  # not empty, so something that is not ours lives here
