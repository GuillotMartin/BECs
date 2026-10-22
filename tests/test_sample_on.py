"""A potential asked for its landscape as seen by a field, as the plotting functions ask it.

What has to hold: the result lives exactly where the field does, frame for frame, whether the
field's grid is fixed or moving and whether its coordinates are stored dense or factored; it is the
landscape the solvers' own 'make_Vt'/'make_V' give; and it is lazy, so that drawing one frame of a
sweep costs one frame of potential.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from bloch_schrodinger.frames import materialize, set_cart_factors
from bloch_schrodinger.plotting import plot_eigenvector
from bloch_schrodinger.potential import create_parameter

from BECs.potentialT import AnalyticPotential, PotentialT
from BECs.ssfm import SSFM

BOX = [[8, 0], [0, 6]]
RES = (16, 12)


def modulated() -> PotentialT:
    """A trap whose depth oscillates in time, at two swept frequencies."""
    pot = PotentialT(BOX, RES, v0=0)
    pot.V = pot.x**2 + pot.y**2
    pot.sine("mod", omega=create_parameter("omega", np.array([1.0, 2.0])), amplitude=0.5, mean=1)
    pot.modulate("mod")
    return pot


@pytest.fixture(scope="module")
def run():
    """A short SSFM run in the modulated trap, also swept over an interaction strength."""
    pot = modulated()
    psi0 = np.exp(-(pot.x**2 + pot.y**2))
    psi0 = psi0 / float((abs(psi0) ** 2).sum() * pot.get_dS()) ** 0.5
    g = create_parameter("g", np.array([0.0, 1.0]))
    t = create_parameter("t", np.linspace(0, 0.2, 3))
    psi = SSFM(pot, psi0, g).solve(0, 0.2, t, dt0=1e-3, tol=1e-6, verbose=False)
    return pot, psi


@pytest.fixture
def expanding():
    """A field on a grid stretching with time, stored factored, and an analytic trap swept over 'k'."""
    pot = AnalyticPotential(BOX, RES)
    pot.add_function(
        "harm",
        lambda t, x, y, k: k * (x**2 + y**2),
        parameters={"k": create_parameter("k", np.array([1.0, 3.0]))},
    )
    pot.add_term("harm")

    tv = np.linspace(0, 1, 4)
    lam = xr.DataArray(1 + 0.5 * tv, dims="t", coords={"t": tv})
    rho_x, rho_y = xr.broadcast(pot.x, pot.y)
    field = (np.exp(-(rho_x**2 + rho_y**2)) * (1 + 0 * lam)).transpose("t", "a1", "a2")
    field = field.drop_vars(["x", "y"]).assign_coords(
        rho_x=rho_x.drop_vars(["x", "y"], errors="ignore"),
        rho_y=rho_y.drop_vars(["x", "y"], errors="ignore"),
        lambda_x=lam,
        lambda_y=lam * 1.2,
    )
    field = set_cart_factors(field, {"x": ("rho_x", "lambda_x"), "y": ("rho_y", "lambda_y")})
    return pot, field


# --- a fixed grid -------------------------------------------------------------------------------

def test_a_tabulated_potential_is_sampled_frame_by_frame(run):
    pot, psi = run
    V = pot.sample_on(psi)

    assert type(V.data).__module__.startswith("dask"), "lazy, one chunk per frame"
    # 'g' does not change the landscape, so it is left to broadcasting
    assert set(V.dims) == {"t", "omega", "a1", "a2"}
    assert np.allclose(V.x.values, psi.x.values) and np.allclose(V.y.values, psi.y.values)

    truth = pot.to_potential(t_coord=psi.t).V.transpose(*V.dims)
    assert np.allclose(V.values, truth.values)


def test_frames_computed_concurrently_stay_distinct():
    """The frames of a run share one evaluation context, so evaluating two at once mixes them.

    dask's threaded scheduler computes several chunks at the same time, which is exactly that
    case: with two functions to fill the context, one frame reliably picks up another's half.
    """
    pot = AnalyticPotential(BOX, (64, 64))
    pot.add_function("f1", lambda t, x, y: np.cos(x * t) * np.exp(-(y**2)))
    pot.add_function("f2", lambda t, x, y: np.sin(y * t) * np.exp(-(x**2)))
    pot.add_term("f1 + 2 * f2")
    tv = np.linspace(0, 3, 16)
    field = (pot.x * 0 + xr.DataArray(tv, dims="t", coords={"t": tv})).transpose("t", "a1", "a2")

    V = pot.sample_on(field).compute(scheduler="threads", num_workers=8)
    truth = pot.to_potential(t_coord=field.t).V.transpose(*V.dims)
    assert np.allclose(V.values, truth.values)


def test_a_field_at_one_point_is_sampled_at_that_point(run):
    pot, psi = run
    V = pot.sample_on(psi.sel(omega=2.0, t=0.1))
    assert set(V.dims) == {"a1", "a2"}
    truth = pot.to_potential(t=0.1).V.sel(omega=2.0)
    assert np.allclose(V.values, truth.transpose(*V.dims).values)


def test_a_field_without_time_is_sampled_at_t0():
    """An initial state drawn over a time-dependent trap, as the tutorials do with psi0."""
    pot = modulated()
    psi0 = np.exp(-(pot.x**2 + pot.y**2))
    V = pot.sample_on(psi0)
    assert set(V.dims) == {"omega", "a1", "a2"}, "the sweep the potential carries is kept"
    truth = pot.to_potential(t=0.0).V
    assert np.allclose(V.values, truth.transpose(*V.dims).values)


def test_a_static_potential_without_a_field_is_its_landscape():
    pot = PotentialT(BOX, RES, v0=0)
    pot.V = pot.x**2
    assert np.allclose(pot.sample_on().values, (pot.x**2).values)


def test_a_time_dependent_potential_needs_a_field():
    with pytest.raises(ValueError, match="no time to sample"):
        modulated().sample_on()


def test_a_field_on_another_lattice_is_refused(run):
    pot, _ = run
    other = PotentialT(BOX, (8, 8), v0=0)
    with pytest.raises(ValueError, match="lattice"):
        pot.sample_on(other.V)


def test_a_tabulated_potential_refuses_a_foreign_grid():
    pot = modulated()
    elsewhere = PotentialT([[4, 0], [0, 3]], RES, v0=0)  # same lattice, a different box
    with pytest.raises(ValueError, match="not the potential's own"):
        pot.sample_on(elsewhere.V)


# --- a moving grid ------------------------------------------------------------------------------

def test_an_analytic_potential_follows_a_factored_moving_grid(expanding):
    pot, field = expanding
    V = pot.sample_on(field)

    assert set(V.dims) == {"t", "k", "a1", "a2"}, "'k' is swept by the potential only"
    assert "cart_factors" in V.attrs, "laid out on the very same frame as the field"
    for t in field.t.values:
        for k in (1.0, 3.0):
            x = (field.rho_x * field.lambda_x.sel(t=t)).transpose("a1", "a2").values
            y = (field.rho_y * field.lambda_y.sel(t=t)).transpose("a1", "a2").values
            frame = V.sel(t=t, k=k).transpose("a1", "a2").values
            assert np.allclose(frame, k * (x**2 + y**2))


def test_both_storage_forms_of_a_moving_grid_give_the_same_landscape(expanding):
    pot, field = expanding
    lean = pot.sample_on(field)
    full = pot.sample_on(materialize(field))
    assert np.allclose(lean.values, full.transpose(*lean.dims).values)


def test_a_tabulated_potential_refuses_a_moving_grid(expanding):
    _, field = expanding
    with pytest.raises(ValueError, match="moves with"):
        modulated().sample_on(field)


def test_plot_eigenvector_overlays_it_on_the_moving_grid(expanding):
    """The overlay that a moving grid could not have before: contours that follow the frame."""
    pot, field = expanding
    for cst_bds in (True, False):
        fig, axes = plot_eigenvector([[field]], [[pot]], [["amplitude"]], cst_bds=cst_bds)
        try:
            assert len(axes[0][0].collections) == 2, "the field's mesh and the potential's contours"
        finally:
            plt.close(fig)
