"""Shared fixtures. Keeps the suite headless and quiet."""

import matplotlib
import pytest

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _quiet_display(monkeypatch):
    """The plotting functions display ipywidgets on call, which is noise outside a kernel."""
    import bloch_schrodinger.plotting as plotting

    monkeypatch.setattr(plotting, "display", lambda *a, **k: None)
