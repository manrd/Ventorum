# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""The vortex lattice warns when the control-point factor CLAF is limited."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from ventorum.core.datatypes import (
    Aircraft,
    FlightCondition,
    LiftingSurface,
    LinearAirfoil,
    SolverSettings,
    WingSection,
)
from ventorum.solvers.factory import make_solver


def _solve(a0: float):
    af = LinearAirfoil(a0=a0)
    ac = Aircraft(surfaces=[LiftingSurface(
        name="wing", semi_span=4.0,
        sections=[WingSection(y_frac=e, chord=1.0, airfoil=af) for e in (0.0, 1.0)],
    )])
    return make_solver("vlm").solve(ac, FlightCondition(V_inf=30.0, alpha=np.radians(4.0)),
                                    SolverSettings(n_panels=8, n_chord=2))


@pytest.mark.parametrize("factor", [0.1, 2.5])
def test_limited_claf_warns(factor):
    """A section slope outside 0.2 to 2 times 2 pi gives a RuntimeWarning."""
    with pytest.warns(RuntimeWarning, match="control point is limited"):
        _solve(factor * 2.0 * np.pi)


@pytest.mark.parametrize("factor", [0.2, 0.9, 1.0, 2.0])
def test_claf_in_range_does_not_warn(factor):
    """A section slope inside the limits gives no CLAF warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _solve(factor * 2.0 * np.pi)
    assert not any("control point is limited" in str(w.message) for w in caught)
