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
"""

import re

import numpy as np
import xarray as xr

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
        frame (bool, optional): Whether to rebuild the moving cartesian coordinates from the
        compact 'lambda_*' form, for a run saved with 'store_frame=False'. The rebuilt coordinates
        stay lazy, so this does not undo the point of opening lazily. Defaults to False.

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
        # binary operation, which aligns and merges the operands' coordinates; where those carry a
        # materialised moving frame that means computing and comparing gigabytes, and the result
        # comes back eager. Here the coordinates are taken from one side and never compared.
        out = xr.DataArray(
            opened.data + 1j * other.data,
            dims=opened.dims,
            coords=opened.coords,
            attrs=opened.attrs,
            name=opened.name,
        )

    if frame:
        from BECs.rssfm import attach_frame

        out = attach_frame(out)
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
