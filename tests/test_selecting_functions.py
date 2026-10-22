"""Functions that carry parameter dimensions of their own, and reduce themselves to a point of them.

A parameter handed to 'add_function' as a DataArray reaches the function as its *value* at the
point a run sits at. That is all an ordinary function needs, but not one whose dependence on a
parameter goes through more than a number -- an object built from swept inputs. Such a function
declares its dims in 'param_coords' and reduces itself with 'sel', and the potential does the rest.
"""

import numpy as np
import pytest
import xarray as xr
from bloch_schrodinger.potential import create_parameter

from BECs.potentialT import AnalyticPotential, PotentialT, selects_itself
from BECs.ssfm import SSFM

BOX = [[8, 0], [0, 6]]
RES = (16, 12)


class Table:
    """A time function whose dependence on 'k' goes through its own state.

    Reduced, it is a plain f(t, *coords) holding one slope; unreduced, it keeps 'k' as a dim of
    its result, as 'to_potential' expects of every registered function.
    """

    def __init__(self, k: xr.DataArray):
        self.param_coords = {"k": k}
        self.reduced = []  # the selection of every reduction, to count them

    def sel(self, selection: dict):
        self.reduced.append(dict(selection))
        slope = float(selection["k"]) ** 2  # stands in for anything costlier than a number

        def at(t, *coords, **params):
            return slope * t + 0 * (coords[0] if coords else 0)

        return at

    def __call__(self, t, *coords, **params):
        return self.param_coords["k"] ** 2 * t + 0 * (coords[0] if coords else 0)


@pytest.fixture
def k():
    return create_parameter("k", np.array([1.0, 2.0, 3.0]))


def test_the_protocol_is_recognised_by_its_two_members(k):
    assert selects_itself(Table(k))
    assert not selects_itself(lambda t: t)
    assert not selects_itself(object())


def test_its_dims_become_parameter_dims_of_the_potential(k):
    pot = AnalyticPotential(BOX, RES)
    pot.add_function("tab", Table(k))
    assert "k" in pot.param_coords
    assert np.array_equal(pot.param_coords["k"].values, k.values)


def test_make_V_reduces_it_once_per_run(k):
    pot = AnalyticPotential(BOX, RES)
    table = Table(k)
    pot.add_function("tab", table)
    pot.add_term("tab")

    V = pot.make_V({"k": 2.0})
    for t in (0.0, 0.5, 1.0):
        assert np.allclose(np.asarray(V(t)), 4.0 * t)
    assert table.reduced == [{"k": 2.0}], "reduced once, when the run is set up"


def test_make_Vt_reduces_it_too(k):
    pot = PotentialT(BOX, RES, v0=0)
    pot.V = pot.x * 0 + 1.0
    pot.add_function("tab", Table(k))
    pot.add_term("tab")
    assert np.allclose(np.asarray(pot.make_Vt({"k": 3.0})(2.0)), 1.0 + 9.0 * 2.0)


def test_only_the_dims_it_declared_are_handed_over(k):
    pot = AnalyticPotential(BOX, RES)
    table = Table(k)
    pot.add_function("tab", table)
    pot.sine("wave", omega=create_parameter("omega", np.array([1.0, 2.0])))
    pot.add_term("tab * wave")

    pot.make_V({"k": 1.0, "omega": 2.0})
    assert table.reduced == [{"k": 1.0}]


def test_to_potential_keeps_its_dims(k):
    pot = AnalyticPotential(BOX, RES)
    pot.add_function("tab", Table(k))
    pot.add_term("tab")
    V = pot.to_potential(t=1.0).V
    assert "k" in V.dims
    assert np.allclose(V.max(["a1", "a2"]).values, k.values**2)


def test_the_solver_sweeps_its_dims(k):
    """The whole point: a run per point of the function's own dims, each with its own slope."""
    pot = AnalyticPotential(BOX, RES)
    pot.add_function("tab", Table(k))
    pot.add_term("tab")
    psi0 = xr.ones_like(pot.x, dtype=complex)
    psi0 = psi0 / float((abs(psi0) ** 2).sum() * pot.get_dS()) ** 0.5

    t = create_parameter("t", np.array([0.0, 0.1]))
    psi = SSFM(pot, psi0, 0.0).solve(0.0, 0.1, t, dt0=1e-3, tol=1e-8, verbose=False)
    assert "k" in psi.dims

    # A uniform V(t) = k^2 t only winds the phase, by k^2 t^2 / 2 at time t: each run must have
    # wound it at its own rate (the sign is the solver's convention, not what is checked here)
    phase = np.angle(psi.sel(t=0.1).isel(a1=0, a2=0).values / psi.sel(t=0.0).isel(a1=0, a2=0).values)
    assert np.allclose(abs(phase), k.values**2 * 0.1**2 / 2, atol=1e-4)
