"""The sweep of the trust error model comes from the main surface only.

Regression test: the sweep that the error model and the lifting-line sweep
warning use was a mean over all surfaces. A swept fin on an unswept wing
then gave a false sweep, a larger predicted error and false sweep warnings.
"""

import warnings

import numpy as np
import pytest

from ventorum.core.datatypes import Aircraft, FlightCondition, LiftingSurface, SolverSettings, WingSection
from ventorum.solvers.factory import make_solver


def _wing() -> LiftingSurface:
    """Return an unswept rectangular wing (semi-span 5 m, chord 1 m)."""
    return LiftingSurface(name="W", semi_span=5.0, sections=[WingSection(0, 1.0), WingSection(1, 1.0)])


def _fin(sweep_deg: float) -> LiftingSurface:
    """Return a vertical fin with the leading-edge sweep *sweep_deg* [deg]."""
    return LiftingSurface(name="F", semi_span=1.5, dihedral=np.pi / 2, is_symmetric=False,
                          sweep_le=np.radians(sweep_deg), position=np.array([5.0, 0.0, 0.0]),
                          sections=[WingSection(0, 1.2), WingSection(1, 1.2)])


def _solve(solver: str, surfaces: list[LiftingSurface]):
    """Solve *surfaces* at alpha 4 deg with 16 spanwise panels."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return make_solver(solver).solve(Aircraft(surfaces=surfaces), FlightCondition(alpha=np.radians(4.0)),
                                         SolverSettings(solver_type=solver, n_panels=16))


@pytest.mark.parametrize("solver", ["vlm", "linear"])
def test_swept_fin_does_not_change_the_main_wing_sweep(solver):
    """A 45 deg fin gives the sweep and the predicted error of the wing alone."""
    alone = _solve(solver, [_wing()])
    with_fin = _solve(solver, [_wing(), _fin(45.0)])
    assert with_fin.details["sweep_deg"] == pytest.approx(0.0, abs=1e-12)
    assert with_fin.details["sweep_deg"] == pytest.approx(alone.details["sweep_deg"], abs=1e-12)
    t_alone, t_fin = alone.totals.trust, with_fin.totals.trust
    assert t_fin.uncertainty_CL == pytest.approx(t_alone.uncertainty_CL, rel=1e-12)
    assert t_fin.factors.get("error_model") == t_alone.factors.get("error_model")
    for msg in t_fin.warnings:
        assert "sweep angle" not in msg
        assert "not grid convergent with sweep" not in msg


def test_swept_fin_gives_the_unswept_lifting_line_error():
    """The lifting-line predicted error with a 45 deg fin is the unswept value (about 9.3 %)."""
    res = _solve("linear", [_wing(), _fin(45.0)])
    ref = _solve("linear", [_wing(), _fin(0.0)])
    assert res.totals.trust.score == pytest.approx(ref.totals.trust.score, rel=1e-12)
    assert any("Predicted error about 9.3 %" in m for m in res.totals.trust.warnings)


def test_swept_main_wing_keeps_its_sweep():
    """The sweep of a swept main wing does not change when an unswept tail is added."""
    swept = LiftingSurface(name="W", semi_span=5.0, sweep_le=np.radians(30.0),
                           sections=[WingSection(0, 1.0), WingSection(1, 1.0)])
    tail = LiftingSurface(name="T", semi_span=1.5, position=np.array([8.0, 0.0, 0.0]),
                          sections=[WingSection(0, 0.8), WingSection(1, 0.8)])
    alone = _solve("vlm", [swept])
    both = _solve("vlm", [swept, tail])
    assert alone.details["sweep_deg"] > 25.0
    assert both.details["sweep_deg"] == pytest.approx(alone.details["sweep_deg"], rel=1e-12)
