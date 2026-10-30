# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests of the Ventorum-2D robustness rules: long wake warning, budgets of the continuation path, polar gaps.

* A wake longer than ``WAKE_LENGTH_LIMIT`` chords gives a warning and a note in the reason.
* The continuation path from the zero-lift angle (cold start and continuation steps) has its own
  work budgets; ``WORK_BUDGET`` counts the work at the angle itself. A failed stage of the path is
  named in the reason.
* A polar with dropped angles keeps a gap mask; a lookup inside a bridged interval warns once; the
  V1 coupling gives a trust note with the number of strips that read a bridged interval.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from ventorum.aero.polar_cache import PolarCache
from ventorum.aero.polar_source_v2d import Ventorum2DPolarSource, gap_mask
from ventorum.aero.polar_table import PolarTable
from ventorum.airfoil import airfoil
from ventorum.core.datatypes import (
    FlightCondition,
    LiftingSurface,
    SolverSettings,
    TabulatedAirfoil,
    WingSection,
)
from ventorum.panel2d import viscous as vs
from ventorum.viscous.alpha_correction import SOURCE_ALPHAS_DEG
from ventorum.viscous.alpha_correction import solve_viscous as solve_coupled


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


# ---------------------------------------------------------------------------------------------
# Polar gaps
# ---------------------------------------------------------------------------------------------

def test_gap_mask_marks_the_bridged_intervals():
    """Only intervals between kept angles that hold a dropped angle are marked; end drops are not."""
    req = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
    kept = [-1.0, 0.0, 2.0, 3.0]
    np.testing.assert_array_equal(gap_mask(req, kept), [False, True, False])
    assert gap_mask(req, req).size == len(req) - 1 and not gap_mask(req, req).any()
    assert gap_mask(req, [1.0]).size == 0


def _fake_result(failed_deg):
    """Return a fake 2D solver whose angles in ``failed_deg`` do not converge."""

    def fake_solve(bodies, alpha_deg, Re, **kwargs):
        alphas = np.atleast_1d(np.asarray(alpha_deg, dtype=float))
        ok = ~np.isin(np.round(alphas, 6), np.round(np.asarray(failed_deg, dtype=float), 6))
        cl = 2.0 * np.pi * np.radians(alphas)
        return SimpleNamespace(alpha=alphas, Cl=np.where(ok, cl, np.nan), Cl_circulation=np.where(ok, cl, np.nan),
                               Cd=np.where(ok, 0.008, np.nan), Cm=np.where(ok, 0.0, np.nan), converged=ok)

    return fake_solve


def test_polar_with_a_failed_angle_has_a_gap_and_warns_once(monkeypatch, tmp_path):
    """One forced failed angle: the polar carries the gap mask, a lookup in the gap warns once."""
    import ventorum.aero.polar_source_v2d as module

    monkeypatch.setattr(module, "solve_viscous", _fake_result([2.0]))
    source = Ventorum2DPolarSource(cache=PolarCache(tmp_path))
    alphas = np.arange(-2.0, 6.1, 1.0)
    polar = source.polar("naca 0012", 1e6, alphas)
    assert source.last_dropped == 1
    np.testing.assert_array_equal(polar.gap_mask, [False, False, False, True, False, False, False])
    assert polar.has_gaps
    assert not polar.in_gap(np.radians(1.0)) and polar.in_gap(np.radians(2.0))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        polar.Cl(np.radians(0.5))
        assert not caught
        polar.Cl(np.radians(2.0))
        polar.Cd(np.radians(2.5))
        polar.Cl_slope(np.radians(1.5))
    gap_warnings = [w for w in caught if "bridges across a failed angle" in str(w.message)]
    assert len(gap_warnings) == 1
    # The cache hit gives the gap mask again from the requested angles.
    monkeypatch.setattr(module, "solve_viscous", None)
    hit = source.polar("naca 0012", 1e6, alphas)
    np.testing.assert_array_equal(hit.gap_mask, polar.gap_mask)
    assert source.last_dropped == 1
    # A clone keeps the mask.
    np.testing.assert_array_equal(polar.clone().gap_mask, polar.gap_mask)


def test_polar_without_dropped_angles_has_no_gap(monkeypatch):
    """A polar without dropped angles has no mask and never warns."""
    import ventorum.aero.polar_source_v2d as module

    monkeypatch.setattr(module, "solve_viscous", _fake_result([]))
    polar = Ventorum2DPolarSource(cache=False).polar("naca 0012", 1e6, np.arange(-2.0, 4.1, 1.0))
    assert polar.gap_mask is None and not polar.has_gaps
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        polar.Cl(np.radians(1.5))


def test_gap_mask_size_is_checked():
    """A gap mask with the wrong size is refused at the lookup."""
    a = np.radians([0.0, 1.0, 2.0])
    polar = TabulatedAirfoil(alpha=a, Cl_data=a, Cd_data=0.01 + 0 * a, gap_mask=np.array([True]))
    with pytest.raises(ValueError, match="gap_mask"):
        polar.in_gap(0.01)


def test_old_polar_files_without_a_mask_load(tmp_path):
    """An npz polar table and a cached polar written without a gap mask load with an empty mask."""
    alpha = np.arange(-4.0, 8.1, 2.0)
    cl = 2.0 * np.pi * np.radians(alpha)
    table = PolarTable(Re=np.array([1e6]), flap_deg=np.array([0.0]), alpha_deg=alpha,
                       Cl=cl[None, None, :], Cd=np.full((1, 1, alpha.size), 0.01),
                       Cm=np.zeros((1, 1, alpha.size)), source_info={"source": "test"})
    path = table.to_npz(tmp_path / "old_table.npz")
    with np.load(path) as stored:
        assert not any("gap" in k for k in stored.files)
    polar = PolarTable.from_npz(path).at(Re=1e6, flap_deg=0.0)
    assert polar.gap_mask is None and not polar.has_gaps
    assert not polar.in_gap(np.radians(1.0))
    cache = PolarCache(tmp_path / "cache")
    payload = {"source": "test", "section": "naca 0012"}
    cache.put(payload, TabulatedAirfoil(name="old", alpha=np.radians(alpha), Cl_data=cl, Cd_data=0.01 + 0 * cl))
    hit = cache.get(payload)
    assert hit is not None and hit.gap_mask is None


def test_coupling_trust_note_counts_strips_in_a_gap(monkeypatch):
    """The V1 coupling gives a trust note with the strips that read a bridged interval and last_dropped."""
    import ventorum.aero.polar_source_v2d as module

    failed = [a for a in SOURCE_ALPHAS_DEG if 1.0 <= a <= 6.0]
    monkeypatch.setattr(module, "solve_viscous", _fake_result(failed))
    wing = LiftingSurface(name="Wing", semi_span=4.0,
                          sections=[WingSection(y_frac=0.0, chord=1.0, shape="0012"),
                                    WingSection(y_frac=1.0, chord=1.0, shape="0012")])
    cond = FlightCondition(V_inf=40.0, alpha=np.radians(5.0))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = solve_coupled(wing, cond, SolverSettings(n_panels=12, n_chord=4), Ventorum2DPolarSource(cache=False),
                            model="vlm")
    notes = [n for n in res.details["notes"] if "strips read their section polar inside" in n]
    assert len(notes) == 1, res.details["notes"]
    assert f"last_dropped: {len(failed)} angles dropped in total" in notes[0]
    n_strips = int(notes[0].split(" of ")[0])
    assert n_strips > 0
    assert notes[0] in res.totals.trust.warnings
    assert any("bridges across a failed angle" in str(w.message) for w in caught)


def test_coupling_without_gaps_has_no_gap_note(monkeypatch):
    """A coupling whose polars have no gaps has no gap note."""
    import ventorum.aero.polar_source_v2d as module

    monkeypatch.setattr(module, "solve_viscous", _fake_result([]))
    wing = LiftingSurface(name="Wing", semi_span=4.0,
                          sections=[WingSection(y_frac=0.0, chord=1.0, shape="0012"),
                                    WingSection(y_frac=1.0, chord=1.0, shape="0012")])
    cond = FlightCondition(V_inf=40.0, alpha=np.radians(5.0))
    res = solve_coupled(wing, cond, SolverSettings(n_panels=12, n_chord=4), Ventorum2DPolarSource(cache=False),
                        model="vlm")
    assert not [n for n in res.details["notes"] if "strips read their section polar inside" in n]


# ---------------------------------------------------------------------------------------------
# Gap mask of derived polars: table lookup, blend, flap deflection
# ---------------------------------------------------------------------------------------------

def _table(cl_rows):
    """Return a free-air PolarTable at Re 1e6 and 2e6 (flap 0 deg) with the given lift rows (NaN = dropped)."""
    alpha = np.arange(-2.0, 4.1, 1.0)
    cl = np.asarray(cl_rows, dtype=float).reshape(2, 1, alpha.size)
    cd = np.where(np.isfinite(cl), 0.01, np.nan)
    return PolarTable(Re=np.array([1e6, 2e6]), flap_deg=np.array([0.0]), alpha_deg=alpha, Cl=cl, Cd=cd,
                      Cm=np.where(np.isfinite(cl), 0.0, np.nan), source_info={"source": "test"})


def test_table_lookup_marks_the_intervals_it_bridges():
    """PolarTable.at: a NaN point between kept points marks its interval; end drops mark none."""
    full = 0.1 * np.arange(7.0)
    row = full.copy()
    row[3] = np.nan  # 1 deg dropped at Re 1e6
    table = _table([row, full])
    polar = table.at(Re=1e6, flap_deg=0.0)
    np.testing.assert_allclose(np.degrees(polar.alpha), [-2.0, -1.0, 0.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(polar.gap_mask, [False, False, True, False, False])
    assert polar.in_gap(np.radians(1.0)) and not polar.in_gap(np.radians(2.5))
    with pytest.warns(RuntimeWarning, match="bridges across a failed angle"):
        polar.Cl(np.radians(1.0))
    # Between the Reynolds rows both rows are used: the NaN of the lower row drops the point.
    mid = table.at(Re=1.5e6, flap_deg=0.0)
    np.testing.assert_array_equal(mid.gap_mask, [False, False, True, False, False])
    # At the upper row only the full row is used: no point is dropped and no interval is marked.
    upper = table.at(Re=2e6, flap_deg=0.0)
    assert upper.alpha.size == 7 and upper.gap_mask is None and not upper.has_gaps
    # Dropped points at the ends of the table bridge no interval.
    ends = full.copy()
    ends[[0, 6]] = np.nan
    edge = _table([ends, full]).at(Re=1e6, flap_deg=0.0)
    assert edge.alpha.size == 5 and edge.gap_mask is None
    # Two neighbouring dropped points make one bridged interval.
    two = full.copy()
    two[[2, 3]] = np.nan
    np.testing.assert_array_equal(_table([two, full]).at(Re=1e6, flap_deg=0.0).gap_mask, [False, True, False, False])


def _gap_polar(name, alpha_deg, gap_after_deg):
    """Return a polar whose interval that starts at ``gap_after_deg`` is bridged."""
    a = np.radians(np.asarray(alpha_deg, dtype=float))
    mask = np.isclose(np.asarray(alpha_deg[:-1], dtype=float), gap_after_deg)
    return TabulatedAirfoil(name=name, alpha=a, Cl_data=2.0 * np.pi * a, Cd_data=0.01 + 0.0 * a,
                            Cm_data=0.0 * a, Re=1e6, gap_mask=mask)


def test_blend_marks_an_interval_when_a_used_component_interval_is_bridged():
    """blend_tabulated: a blend interval is bridged when a polar with a weight above 0 bridges it."""
    from ventorum.aero.polars import blend_tabulated

    one = _gap_polar("one", np.array([-2.0, 0.0, 2.0, 4.0]), 0.0)  # bridged [0, 2] deg
    two = _gap_polar("two", np.array([-2.0, -1.0, 1.0, 3.0, 4.0]), 3.0)  # bridged [3, 4] deg
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        half = blend_tabulated(one, two, 0.5)
        only_one = blend_tabulated(one, two, 0.0)
        only_two = blend_tabulated(one, two, 1.0)
    np.testing.assert_allclose(np.degrees(half.alpha), [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(half.gap_mask, [False, False, True, True, False, True])
    np.testing.assert_array_equal(only_one.gap_mask, [False, False, True, True, False, False])
    np.testing.assert_array_equal(only_two.gap_mask, [False, False, False, False, False, True])
    clean = TabulatedAirfoil(name="clean", alpha=one.alpha, Cl_data=one.Cl_data, Cd_data=one.Cd_data,
                             Cm_data=one.Cm_data, Re=1e6)
    assert blend_tabulated(clean, clean, 0.5).gap_mask is None
    assert blend_tabulated(one, clean, 1.0).gap_mask is None


def test_deflected_polar_keeps_the_gap_mask():
    """deflected_airfoil: the shifted table keeps the mask, so the bridged interval moves with the angles."""
    from ventorum.geometry.controls import deflected_airfoil, flap_effectiveness

    base = _gap_polar("base", np.array([-2.0, 0.0, 2.0, 4.0]), 0.0)
    delta = np.radians(5.0)
    flapped = deflected_airfoil(base, delta, 0.75)
    np.testing.assert_array_equal(flapped.gap_mask, base.gap_mask)
    assert flapped.gap_mask is not base.gap_mask
    shift = flap_effectiveness(0.75) * delta
    assert flapped.in_gap(np.radians(1.0) - shift) and not flapped.in_gap(np.radians(3.0) - shift)
    clean = TabulatedAirfoil(name="clean", alpha=base.alpha, Cl_data=base.Cl_data, Cd_data=base.Cd_data)
    assert deflected_airfoil(clean, delta, 0.75).gap_mask is None
