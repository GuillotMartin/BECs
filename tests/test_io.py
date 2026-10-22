"""Opening a saved run, here one whose moving frame is stored as factors."""

import numpy as np
import xarray as xr
from bloch_schrodinger.frames import cart_factors, materialize, set_cart_factors

from BECs.io import open_result


def test_open_result_writes_out_a_factored_frame(tmp_path):
    """frame=True gives the same coordinates as materialising in memory, and keeps them lazy."""
    a1, a2 = np.linspace(-1, 1, 6), np.linspace(-2, 2, 5)
    tv = np.linspace(0, 1, 3)
    rho_x, rho_y = np.meshgrid(a1, a2, indexing="ij")
    field = xr.DataArray(
        np.random.default_rng(0).normal(size=(3, 6, 5)) * (1 + 1j),
        dims=("t", "a1", "a2"),
        coords={"t": tv, "a1": a1, "a2": a2},
        name="psi",
    ).assign_coords(
        rho_x=(("a1", "a2"), rho_x),
        rho_y=(("a1", "a2"), rho_y),
        lambda_x=("t", 1 + tv),
        lambda_y=("t", 1 + 2 * tv),
    )
    field = set_cart_factors(field, {"x": ("rho_x", "lambda_x"), "y": ("rho_y", "lambda_y")})

    field.real.to_netcdf(tmp_path / "re.nc")
    field.imag.to_netcdf(tmp_path / "im.nc")

    kept = open_result(tmp_path / "re.nc", tmp_path / "im.nc")
    assert cart_factors(kept) == cart_factors(field), "opened as stored, recipe included"

    opened = open_result(tmp_path / "re.nc", tmp_path / "im.nc", frame=True)
    expected = materialize(field)
    assert "cart_factors" not in opened.attrs
    assert type(opened.x.data).__module__.startswith("dask"), "the rebuilt frame stays lazy"
    for name in ("x", "y"):
        assert np.allclose(
            opened[name].transpose(*expected[name].dims).values, expected[name].values
        )
    assert np.allclose(opened.values, field.values)
