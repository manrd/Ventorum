# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests of the Ventorum-2D robustness rules: long wake warning, budgets of the continuation path, polar gaps.

* A wake longer than ``WAKE_LENGTH_LIMIT`` chords gives a warning and a note in the reason.
* The continuation path from the zero-lift angle (cold start and continuation steps) has its own
  work budgets; ``WORK_BUDGET`` counts the work at the angle itself. A failed stage of the path is
  named in the reason.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import pytest

from ventorum.airfoil import airfoil
from ventorum.panel2d import viscous as vs


# ---------------------------------------------------------------------------------------------
# Budgets of the continuation path
# ---------------------------------------------------------------------------------------------

@pytest.mark.slow
def test_naca4412_320_panels_converges_with_the_path_budgets():
    """NACA 4412, Re 1e6, 4 deg, 320 panels (321 generated points, as in the polar source): the single
    angle converges (the budget of 250 for the whole solve stopped it before)."""
    r = vs.solve_viscous(airfoil("4412", n_points=321), 4.0, 1e6)
    assert r.converged, r.reason
    assert r.residual < 1e-6
    assert r.newton_iterations > vs.WORK_BUDGET
    assert 0.0 < r.Cd < 0.02 and 0.8 < r.Cl < 1.0


def test_stage_budget_is_separate_from_the_angle_budget():
    """A solve at another angle inside a stage uses the stage budget; a solve at the angle uses the angle budget."""
    ctl = vs._AngleWork(4.0, vs.StopRules(work_budget=10, cold_start_budget=7, step_budget=3))
    ctl.iterations = 9
    assert ctl.cap(50, 4.0) == 1
    ctl.stage = vs._Stage("cold start at the zero-lift angle -4 deg", "cold-start", 7)
    assert ctl.cap(50, -4.0) == 7
    assert ctl.cap(50, 4.0) == 1
    ctl.stage.iterations = 7
    with pytest.raises(vs._StageBudgetUsedUp):
        ctl.check(50, -4.0)
    ctl.iterations = 10
    with pytest.raises(vs._BudgetUsedUp):
        ctl.check(50, 4.0)
    # work_budget = 0 switches every budget off.
    off = vs._AngleWork(4.0, vs.StopRules(work_budget=0, cold_start_budget=1, step_budget=1))
    off.stage = vs._Stage("x", "cold-start", 1)
    off.stage.iterations = 5
    assert off.cap(50, -4.0) == 50


def test_counters_hold_the_path_work():
    """The work counter of the result is the sum of the angle work and the path work."""
    ctl = vs._AngleWork(4.0, vs.StopRules())
    ctl.stage = vs._Stage("continuation step 1 of 3", "continuation-step", vs.STEP_BUDGET)
    res = SimpleNamespace(status="converged", iterations=12, linear_solves=12, _first_stop=None, residual=1e-8,
                          reason="converged")
    ctl.add(SimpleNamespace(sys=SimpleNamespace(alpha=-1.0)), res, 50, 50)
    ctl.add(SimpleNamespace(sys=SimpleNamespace(alpha=4.0)), res, 50, 50)
    assert ctl.iterations == 12 and ctl.path_iterations == 12
    assert ctl.counters()["newton_iterations"] == 24


def test_stop_rules_check_the_path_budgets():
    """The path budgets are integers >= 0."""
    for bad in ({"cold_start_budget": -1}, {"step_budget": 2.5}):
        with pytest.raises(ValueError):
            vs.StopRules(**bad)
    assert vs.StopRules().cold_start_budget == vs.COLD_START_BUDGET
    assert vs.StopRules().step_budget == vs.STEP_BUDGET


@pytest.mark.slow
def test_failed_cold_start_stage_is_named():
    """A cold start at the zero-lift angle cut by its budget is named in the reason of the failed angle."""
    coords = airfoil("0012").repanel(n_panels=120).coords
    kw = dict(ncrit=9.0, xtr=(1.0, 1.0), height=None, height_reference="pivot", wake_length=1.0, n_wake=30,
              chord_ref=None, moment_ref=None, pivot=None)
    body, chord, moment, pivot = vs.resolve_body(coords)
    kw.update(chord_ref=chord, moment_ref=moment, pivot=pivot)
    rules = vs.StopRules(work_budget=2, cold_start_budget=2, step_budget=2)
    row, _ = vs.finish_angle(body, 6.0, 1e6, None, 1e-6, 50, kw, rules=rules)
    assert not row["converged"]
    assert "continuation path: cold start at the zero-lift angle" in row["reason"]
    assert "cold-start budget of 2 Newton iterations" in row["reason"]
    assert row["newton_iterations"] <= 2 + 2


@pytest.mark.slow
def test_failed_continuation_step_is_named():
    """A continuation step cut by its budget is named in the reason of the failed angle."""
    coords = airfoil("0012").repanel(n_panels=120).coords
    body, chord, moment, pivot = vs.resolve_body(coords)
    kw = dict(ncrit=9.0, xtr=(1.0, 1.0), height=None, height_reference="pivot", wake_length=1.0, n_wake=30,
              chord_ref=chord, moment_ref=moment, pivot=pivot)
    rules = vs.StopRules(work_budget=2, step_budget=1)
    row, _ = vs.finish_angle(body, 6.0, 1e6, None, 1e-6, 50, kw, rules=rules)
    assert not row["converged"]
    assert "continuation path: continuation step 1 of" in row["reason"]
    assert "continuation-step budget of 1 Newton iterations used up" in row["reason"]


# ---------------------------------------------------------------------------------------------
# Long wake
# ---------------------------------------------------------------------------------------------

def test_wake_length_limit_note():
    """The limit is 1.5 chords: no note at the limit, a note above it."""
    assert vs.WAKE_LENGTH_LIMIT == 1.5
    assert vs.wake_length_note(1.5) is None
    assert vs.wake_length_note(1.0) is None
    assert "above 1.5 chords" in vs.wake_length_note(2.0)


@pytest.mark.slow
def test_wake_of_two_chords_warns():
    """A wake of 2 c gives a UserWarning and a note in the reason (no refusal)."""
    coords = airfoil("0012").repanel(n_panels=80).coords
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = vs.solve_viscous(coords, 0.0, 1e6, wake_length=2.0, n_wake=60, work_budget=1)
    texts = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any("wake length 2 chords is above 1.5 chords" in t for t in texts), texts
    assert "wake length 2 chords is above 1.5 chords" in r.reason
