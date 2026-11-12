# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Automatic spanwise fit of the agent tools.

When the caller sets no spanwise count and a call is above the mesh
limit or the work budget, the tools reduce the spanwise strips first and
keep the chordwise count. A count that the caller set is never changed.
The tests lower the two limits in the tools module, so that small meshes
show the rule and the solves stay fast.
"""

from __future__ import annotations

import pytest

import ventorum.agent.tools as tools
from ventorum.agent import (
    batch_evaluate,
    derivative_matrix,
    error_bars,
    ground_effect,
    polar_sweep,
    stability_derivatives,
    trim,
    wing_analysis,
)
from ventorum.core.datatypes import SolverSettings
from ventorum.solvers.lattice_base import CONTROL_MIN_AUTO_N_CHORD

# A wing with a control: the automatic n_chord is at least 16.
FLAP_WING = {"span_m": 10.0, "chord_m": 1.0,
             "controls": [{"name": "flap", "eta_start": 0.0, "eta_end": 0.6, "hinge_x_c": 0.75}]}
FC = {"V_inf_m_s": 40.0, "alpha_deg": 4.0}
TAIL_PLANE = {
    "name": "TP",
    "ref_point_m": [0.3, 0.0, 0.0],
    "surfaces": [
        {"name": "wing", "span_m": 10.0, "chord_m": 1.0},
        {"name": "htail", "span_m": 3.0, "chord_m": 0.6, "position_m": [4.0, 0.0, 0.0],
         "controls": [{"name": "elevator", "hinge_x_c": 0.7}]},
    ],
}


@pytest.fixture
def small_limits(monkeypatch):
    """Lower the mesh limit of the tools to 1000 panels (the budget stays)."""
    monkeypatch.setattr(tools, "MAX_TOTAL_PANELS", 1000)


def _record_analyze(monkeypatch):
    """Record the spanwise and chordwise counts of every vt.analyze call."""
    seen: list[tuple[int, int | None]] = []
    real = tools.vt.analyze

    def wrapped(ac, fc, s, *args, **kwargs):
        seen.append((int(s.n_panels), s.n_chord))
        return real(ac, fc, s, *args, **kwargs)

    monkeypatch.setattr(tools.vt, "analyze", wrapped)
    return seen


def test_fit_span_finds_the_largest_count():
    """The fit gives the largest spanwise count that fits both limits."""
    ac = tools.build_aircraft_from_spec(wing=dict(FLAP_WING))
    sett = SolverSettings()

    def panels(s):
        return 2 * s.n_panels * 16

    def work(s):
        return 5 * panels(s) ** 2

    out, fit = tools._fit_span(ac, sett, None, panels, lambda s: 0)
    assert fit is None and out is sett  # 2560 panels: below the real limit
    out, fit = tools._fit_span(ac, sett, None, panels, work)
    # 5 * (32 n)^2 <= 2e8 gives n <= 197: no change for n = 80.
    assert fit is None and out is sett
    big = SolverSettings(n_panels=400)
    out, fit = tools._fit_span(ac, big, None, panels, work)
    assert fit is not None and out is not big
    assert panels(out) <= tools.MAX_TOTAL_PANELS and work(out) <= tools.MAX_CALL_WORK
    nxt = out.clone()
    nxt.n_panels += 1
    assert panels(nxt) > tools.MAX_TOTAL_PANELS or work(nxt) > tools.MAX_CALL_WORK
    assert fit.n_panels_from == 400 and fit.n_panels_to == out.n_panels == 125
    assert big.n_panels == 400  # the input settings do not change


def test_fit_span_keeps_caller_counts():
    """A spanwise count in the settings or on a surface is not changed."""
    def panels(s):
        return 10 ** 6

    ac = tools.build_aircraft_from_spec(wing=dict(FLAP_WING))
    sett = SolverSettings()
    out, fit = tools._fit_span(ac, sett, {"n_panels": 80}, panels, lambda s: 0)
    assert fit is None and out is sett
    ac2 = tools.build_aircraft_from_spec(wing={**FLAP_WING, "n_panels": 80})
    out, fit = tools._fit_span(ac2, sett, None, panels, lambda s: 0)
    assert fit is None and out is sett
    assert tools._span_is_automatic(ac, None) and tools._span_is_automatic(ac, {"n_chord": 8})
    assert not tools._span_is_automatic(ac2, None)


def test_wing_analysis_reduces_strips_first(small_limits):
    """Above the mesh limit, the strips are reduced and n_chord = 16 is kept."""
    p = wing_analysis(wing=dict(FLAP_WING), flight_condition=dict(FC), detail_level="summary")
    assert p["status"] == "success", p
    su = p["settings_used"]
    # 2 halves x n x 16 <= 1000 gives n = 31.
    assert su["n_panels"] == 31, su
    assert su["n_chord"] == "auto" and su["n_chord_used"] == CONTROL_MIN_AUTO_N_CHORD, su
    note = su["mesh_note"]
    assert "n_panels=80 to n_panels=31" in note, note
    assert "160 to 62 strips" in note, note
    assert "n_chord=16" in note and "1000 panels" in note, note
    assert "kept" in note, note


def test_wing_analysis_caller_span_cuts_chord(small_limits):
    """With n_panels set by the caller, the automatic n_chord is reduced as before."""
    p = wing_analysis(wing=dict(FLAP_WING), flight_condition=dict(FC), settings={"n_panels": 80},
                      detail_level="summary")
    assert p["status"] == "success", p
    su = p["settings_used"]
    assert su["n_panels"] == 80 and su["n_chord_used"] == 1000 // 160, su
    assert "spanwise" not in su["mesh_note"], su["mesh_note"]
    assert "n_chord was reduced to n_chord=6" in su["mesh_note"], su["mesh_note"]
    # A surface count of the caller is kept too.
    p = wing_analysis(wing={**FLAP_WING, "n_panels": 80}, flight_condition=dict(FC), detail_level="summary")
    assert p["status"] == "success", p
    assert p["settings_used"]["n_chord_used"] == 6, p["settings_used"]


def test_caller_span_above_budget_is_refused(monkeypatch):
    """A call above the budget with a spanwise count of the caller is refused, not reduced."""
    monkeypatch.setattr(tools, "MAX_CALL_WORK", 2.0e6)

    def _fail(*args, **kwargs):
        raise AssertionError("no solve must run after a budget refusal")

    monkeypatch.setattr(tools.vt, "analyze", _fail)
    p = stability_derivatives(wing=dict(FLAP_WING), flight_condition=dict(FC), settings={"n_panels": 40})
    assert p["status"] == "error", p
    assert p["error"]["type"] == "invalid_input", p
    assert "budget" in p["error"]["message"].lower(), p["error"]["message"]


def test_work_budget_reduces_strips_with_one_mesh(monkeypatch):
    """Above the budget, the strips are reduced; the 5 cases use one mesh."""
    # 5 solves of at most 800 panels.
    monkeypatch.setattr(tools, "MAX_CALL_WORK", 5 * 800 ** 2)
    seen = _record_analyze(monkeypatch)
    p = stability_derivatives(wing=dict(FLAP_WING), flight_condition=dict(FC))
    assert p["status"] == "success", p
    su = p["settings_used"]
    # 5 (2 n 16)^2 <= 5 * 800^2 gives n = 25.
    assert su["n_panels"] == 25, su
    assert su["n_chord_used"] == CONTROL_MIN_AUTO_N_CHORD, su
    assert "estimated work" in su["mesh_note"] and "budget" in su["mesh_note"], su["mesh_note"]
    assert len(seen) == 5 and len(set(seen)) == 1, seen


def test_floor_then_chord_is_reduced(monkeypatch):
    """At the spanwise floor of 4 strips per semi-span, the automatic n_chord is reduced."""
    monkeypatch.setattr(tools, "MAX_TOTAL_PANELS", 100)
    p = wing_analysis(wing=dict(FLAP_WING), flight_condition=dict(FC), detail_level="summary")
    assert p["status"] == "success", p
    su = p["settings_used"]
    assert su["n_panels"] == tools.MIN_AUTO_SPAN_PANELS == 4, su
    # 8 strips: 100 // 8 = 12 chordwise panels.
    assert su["n_chord_used"] == 12, su
    note = su["mesh_note"]
    assert "floor" in note and "n_chord was reduced to n_chord=12" in note, note


def test_polar_sweep_in_ground_uses_one_fitted_mesh(small_limits, monkeypatch):
    """A polar sweep in ground effect fits the strips and uses one mesh for all angles."""
    seen = _record_analyze(monkeypatch)
    p = polar_sweep(wing=dict(FLAP_WING), alpha_start_deg=0.0, alpha_end_deg=4.0, alpha_step_deg=2.0,
                    flight_condition={"V_inf_m_s": 40.0, "h_m": 1.0})
    assert p["status"] == "success", p
    su = p["settings_used"]
    assert su["n_panels"] < 80 and "spanwise" in su["mesh_note"], su
    assert len({s[0] for s in seen}) == 1, seen
    assert len({row["n_chord"] for row in p["polar_table"]}) == 1, p["polar_table"]
    assert all(row["n_chord"] >= CONTROL_MIN_AUTO_N_CHORD for row in p["polar_table"])



def test_polar_sweep_work_counts_the_control_chord(monkeypatch):
    """Out of ground effect, the work estimate of a polar sweep uses the chordwise count of a control.

    The estimate used the default of 4 chordwise panels, so for a wing with
    a control (16 panels) it was 16 times too low and a call above the
    budget was not reduced.
    """
    # 3 angles of at most 800 panels.
    monkeypatch.setattr(tools, "MAX_CALL_WORK", 3 * 800 ** 2)
    p = polar_sweep(wing=dict(FLAP_WING), alpha_start_deg=0.0, alpha_end_deg=4.0, alpha_step_deg=2.0,
                    flight_condition={"V_inf_m_s": 40.0})
    assert p["status"] == "success", p
    su = p["settings_used"]
    # 3 (2 n 16)^2 <= 3 * 800^2 gives n = 25.
    assert su["n_panels"] == 25, su
    assert su["n_chord_used"] == CONTROL_MIN_AUTO_N_CHORD, su
    assert "budget" in su["mesh_note"], su["mesh_note"]

@pytest.mark.slow
def test_ground_effect_fits_the_strips(small_limits):
    """The heights and free air use the fitted spanwise count and one chordwise count."""
    p = ground_effect(wing=dict(FLAP_WING), heights_m=[1.0, 2.0], alpha_deg=4.0, V_inf_m_s=40.0)
    assert p["status"] == "success", p
    su = p["settings_used"]
    assert su["n_panels"] < 80 and "spanwise" in su["mesh_note"], su
    assert 2 * su["n_panels"] * su["n_chord"] <= 1000, su
    assert len({r["n_chord"] for r in p["rows"] if r["status"] == "ok"}) == 1, p["rows"]


@pytest.mark.slow
def test_trim_fits_the_strips(small_limits):
    """A trim above the mesh limit keeps the 16 chordwise panels."""
    p = trim(aircraft=TAIL_PLANE, CL_target=0.4, pitch_control="elevator",
             flight_condition={"V_inf_m_s": 40.0, "alpha_deg": 3.0}, detail_level="summary")
    assert p["status"] == "success", p
    su = p["settings_used"]
    # Wing and tail: 4 halves x n x 16 <= 1000 gives n = 15.
    assert su["n_panels"] == 15 and su["n_chord_used"] == CONTROL_MIN_AUTO_N_CHORD, su
    assert "spanwise" in su["mesh_note"], su


def test_derivative_matrix_one_fitted_mesh(small_limits, monkeypatch):
    """The derivative matrix gets one settings object with the fitted strips."""
    got: list[SolverSettings] = []
    real = tools.vt.derivative_matrix

    def wrapped(ac, fc, s, **kwargs):
        got.append(s)
        return real(ac, fc, s, **kwargs)

    monkeypatch.setattr(tools.vt, "derivative_matrix", wrapped)
    p = derivative_matrix(aircraft=TAIL_PLANE, flight_condition={"V_inf_m_s": 40.0, "alpha_deg": 2.0},
                          groups=["static", "control"])
    assert p["status"] == "success", p
    assert len(got) == 1 and got[0].n_panels == 15, [s.n_panels for s in got]
    assert p["settings_used"]["n_panels"] == 15 and "spanwise" in p["settings_used"]["mesh_note"]


def test_batch_reduces_only_automatic_candidates(small_limits):
    """A candidate with a surface count of the caller keeps it; an automatic one is fitted."""
    p = batch_evaluate([{"wing": dict(FLAP_WING)}, {"wing": {**FLAP_WING, "n_panels": 20}}],
                       dict(FC), "max_CL")
    assert p["status"] == "success", p
    rows = {r["index"]: r for r in p["rankings"]}
    assert rows[0]["n_chord"] == CONTROL_MIN_AUTO_N_CHORD
    assert "n_panels=80 to n_panels=31" in rows[0]["mesh_note"], rows[0]
    assert "mesh_note" not in rows[1] and rows[1]["n_chord"] == CONTROL_MIN_AUTO_N_CHORD, rows[1]


def test_error_bars_floor_is_the_smallest_fine_level(monkeypatch):
    """The fine level of the error bars does not go below 16 strips per semi-span."""
    monkeypatch.setattr(tools, "MAX_TOTAL_PANELS", 200)
    p = error_bars(wing=dict(FLAP_WING), flight_condition=dict(FC))
    assert p["status"] == "success", p
    assert p["levels"][0] == 16, p["levels"]
    note = p["settings_used"]["mesh_note"]
    assert "floor" in note and "n_chord was reduced" in note, note
