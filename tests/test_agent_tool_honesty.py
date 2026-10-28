# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
The agent tools give the trust and the stability verdicts that the solver results support.

* The Irodov margin of the ground-effect tool does not depend on the
  spacing of the height grid.
* The grid derivatives of ``GroundEffectSweepResult`` warn when the
  height step is large.
* ``ventorum_polar_sweep`` gives the trust rating and the warnings per row
  and in the summary.
* In ground effect, the pitch verdict of ``ventorum_stability_derivatives``
  is the Irodov height-pitch margin, not the static margin alone.
* Every tool that solves gives the trust rating and the warnings.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import ventorum as vt
from ventorum.agent import (
    call_tool,
    error_bars,
    ground_effect,
    mesh_convergence,
    polar_sweep,
    stability_derivatives,
)
from ventorum.agent.response import merge_warnings, row_warnings, warning_label
from ventorum.core.datatypes import FlightCondition, SolverSettings

# A rectangular wing of chord 1 m with the pivot at the quarter chord.
# At alpha = 4 deg and h = 0.3 m its static margin is positive and its
# Irodov margin is negative.
WING = {"semi_span_m": 3.0, "chord_m": 1.0}
PIVOT = [0.25, 0.0, 0.0]
MESH = {"n_panels": 8, "n_chord": 6}
H0, A0 = 0.3, 4.0


def strict_json(payload):
    """The payload must be strict JSON (no NaN, no infinity)."""
    return json.loads(json.dumps(payload, allow_nan=False))


def _aircraft():
    from ventorum.agent.schemas import build_aircraft_from_spec

    ac = build_aircraft_from_spec(WING)
    ac.ref_point = np.array(PIVOT)
    return ac


def _reference_margin() -> float:
    ac = _aircraft()
    fc = FlightCondition(alpha=np.radians(A0), h=H0)
    return vt.height_derivatives(ac, fc, SolverSettings(**MESH)).irodov["margin"]


# ── Irodov margin of the ground-effect tool ─────────────────────────────────

@pytest.mark.parametrize("grid", [[0.29, 0.30, 0.31], [0.1, 0.3, 0.9], [0.3]])
def test_ground_effect_irodov_margin_does_not_depend_on_the_height_grid(grid):
    """The old tool took np.gradient on the user grid: [0.1, 0.3, 0.9] gave a margin 33 % too
    negative, and one height gave no margin."""
    p = ground_effect(WING, grid, alpha_deg=A0, ref_point_m=PIVOT, settings=MESH)
    strict_json(p)
    assert p["status"] == "success", p
    row = next(r for r in p["irodov"]["rows"] if abs(r["h_m"] - H0) < 1e-12)
    assert row["irodov_margin"] == pytest.approx(_reference_margin(), abs=2e-5)
    assert row["verdict"].startswith("unstable")
    assert row["h_step_m"] > 0
    assert p["irodov"]["computed_at"] == "every valid height"
    assert "Irodov margin at h=" in p["executive_summary"]


def test_ground_effect_irodov_only_at_lowest_height_when_the_budget_is_short(monkeypatch):
    """When the Irodov cases at every height exceed the work budget, only the lowest valid
    height gets a margin, and the payload says so."""
    import ventorum.agent.tools as tools

    heights = [0.3, 0.6, 1.0]
    n = tools._planned_panels(tools.build_aircraft_from_spec(WING), tools.parse_settings(MESH))
    # Room for the rows, free air and one height of Irodov cases, not for three.
    budget = (len(heights) + 1 + tools.HEIGHT_DERIVATIVE_SOLVES + 1) * n * n
    monkeypatch.setattr(tools, "MAX_CALL_WORK", budget)
    p = ground_effect(WING, heights, alpha_deg=A0, ref_point_m=PIVOT, settings=MESH)
    strict_json(p)
    assert p["status"] == "success", p
    rows = p["irodov"]["rows"]
    assert rows[0]["irodov_margin"] is not None
    assert all(r["irodov_margin"] is None and r["verdict"].startswith("not computed") for r in rows[1:])
    assert p["irodov"]["computed_at"] == "lowest valid height"
    assert any("work budget" in n for n in p["irodov"]["notes"])


# ── Step warning of the grid derivatives ────────────────────────────────────

def test_sweep_grid_derivatives_warn_for_a_coarse_height_step():
    from ventorum.ground_effect import GroundEffectSweep

    sweep = GroundEffectSweep(_aircraft(), settings=SolverSettings(**MESH), n_jobs=1)
    coarse = sweep.run_sweep([0.1, 0.3, 0.9], [A0 - 0.5, A0, A0 + 0.5], [0.0],
                             compute_strike_limit=False).compute_stability_derivatives()
    fine = sweep.run_sweep([0.29, 0.30, 0.31], [A0 - 0.5, A0, A0 + 0.5], [0.0],
                           compute_strike_limit=False).compute_stability_derivatives()
    assert coarse["h_step_over_clearance"].shape == (3, 3)
    assert any("Coarse height grid" in w for w in coarse["warnings"])
    assert fine["warnings"] == []
    assert np.nanmax(fine["h_step_over_clearance"]) < 0.1


# ── Trust of the polar sweep ────────────────────────────────────────────────

def test_polar_sweep_rows_and_summary_carry_rating_and_warnings():
    """At 20 deg the linear sections give UNRELIABLE with a stall warning. The old rows had only
    trust_score, and the summary had no trust."""
    args = {"wing": WING, "alpha_start_deg": 0, "alpha_end_deg": 20, "alpha_step_deg": 5,
            "settings": {"n_panels": 12}}
    p = call_tool("ventorum_polar_sweep", args)
    strict_json(p)
    assert p["status"] == "success"
    last = p["polar_table"][-1]
    assert last["alpha_deg"] == 20.0
    assert last["trust_rating"] == "UNRELIABLE"
    assert "Severe stall risk" in last["warnings"]
    assert all("trust_rating" in r and isinstance(r["warnings"], list) for r in p["polar_table"])
    s = p["polar_summary"]
    assert s["trust_rating_lowest"] == "UNRELIABLE"
    assert s["trust_alpha_deg_at_lowest"] == 20.0
    assert s["trust_score_min"] == last["trust_score"]
    stall = [w for w in s["warnings"] if w["warning"].startswith("Severe stall risk")]
    assert len(stall) == 1 and 20.0 in stall[0]["alpha_deg"]
    assert s["warnings_not_shown"] == 0
    assert "UNRELIABLE" in p["executive_summary"] and "Severe stall risk" in p["executive_summary"]

    short = call_tool("ventorum_polar_sweep", {**args, "detail_level": "summary"})
    assert "polar_table" not in short
    assert short["polar_summary"]["trust_rating_lowest"] == "UNRELIABLE"
    assert short["polar_summary"]["warnings"] == s["warnings"]


def test_warning_labels_deduplicate_and_cap():
    assert warning_label("Severe stall risk: peak section Cl=1.72 (alpha_eff=15.7 deg).") == "Severe stall risk"
    assert warning_label("Very low aspect ratio (AR=2.00 < 3) for a lifting-line method.") == \
        "Very low aspect ratio"
    assert warning_label("Solver did not converge to the specified tolerance.") == \
        "Solver did not converge to the specified tolerance"
    assert warning_label("Lifting line with sweep angle 30.0 deg: x") == "Lifting line with sweep angle 30.0 deg"
    labels, more = row_warnings(["A: 1", "A: 2", "B: 1", "C (x)", "D. y", "E"], cap=3)
    assert labels == ["A", "B", "C"] and more == 2
    union, more = merge_warnings([(0.0, 0.9, ["A: low"]), (5.0, 0.4, ["A: high", "B: x"]),
                                  (10.0, None, ["C: y"])], "alpha_deg", cap=2)
    assert union == [{"warning": "A: high", "alpha_deg": [0.0, 5.0]}, {"warning": "B: x", "alpha_deg": [5.0]}]
    assert more == 1


# ── Stability verdict in ground effect ──────────────────────────────────────

def test_stability_in_ground_effect_gives_the_irodov_verdict():
    """The static margin is positive but the Irodov margin is negative: the old headline said
    'Pitch: statically stable'."""
    p = stability_derivatives(aircraft={"surfaces": [WING], "ref_point_m": PIVOT},
                              flight_condition={"alpha_deg": A0, "h_m": H0}, settings=MESH)
    strict_json(p)
    assert p["status"] == "success", p
    assert p["stability_derivatives"]["static_margin_fraction"] > 0
    assert p["irodov"]["irodov_margin"] == pytest.approx(_reference_margin(), abs=2e-5)
    assert p["irodov"]["verdict"].startswith("unstable")
    pitch = p["stability_assessment"]["pitch"]
    assert pitch.startswith("statically UNSTABLE in height and pitch")
    assert "not the static stability criterion" in pitch
    assert p["stability_assessment"]["height_pitch_irodov"].startswith("unstable")
    summary = p["executive_summary"]
    assert "Pitch: statically stable" not in summary
    assert "Pitch in ground effect: statically UNSTABLE" in summary
    assert "Irodov margin" in summary and "not the static stability criterion" in summary


def test_stability_in_free_air_has_no_irodov():
    p = stability_derivatives(WING, {"alpha_deg": A0}, x_cg_m=0.1, settings=MESH)
    assert p["status"] == "success"
    assert "irodov" not in p and "height_pitch_irodov" not in p["stability_assessment"]
    assert "Pitch: statically stable." in p["executive_summary"]


# ── Every solving tool carries the trust ────────────────────────────────────

def test_stability_summary_level_keeps_the_trust():
    """The summary level used to drop the trust of the stability tool."""
    p = stability_derivatives(WING, {"alpha_deg": 20.0}, settings={"n_panels": 12}, detail_level="summary")
    assert p["status"] == "success"
    t = p["trust"]
    assert t["rating"] == "UNRELIABLE"
    assert any("stall" in w for w in t["warnings"])
    assert "heuristic_bands_not_calibrated" not in t
    assert t["warnings_not_shown"] >= 0


def test_mesh_convergence_and_error_bars_carry_the_trust():
    fc = {"alpha_deg": 20.0}
    m = mesh_convergence(WING, fc, panel_counts=[8, 12], spacing_schemes=["half-cosine"], ref_n_panels=24)
    assert m["status"] == "success", m
    assert m["trust"]["rating"] in ("UNRELIABLE", "LOW")
    assert any("stall" in w for w in m["trust"]["warnings"])
    assert m["trust_of"] in ("recommended_mesh", "reference mesh")
    e = error_bars(WING, fc, settings={"n_panels": 16})
    assert e["status"] == "success", e
    assert e["trust"]["rating"] in ("UNRELIABLE", "LOW")
    assert any("stall" in w for w in e["trust"]["warnings"])


def test_polar_sweep_in_ground_effect_rows_have_rating():
    p = polar_sweep(WING, 2.0, 4.0, 2.0, {"h_m": 0.3}, MESH)
    assert p["status"] == "success"
    assert all(r["trust_rating"] is not None for r in p["polar_table"])
    assert any(w["warning"].startswith("Strong ground effect") for w in p["polar_summary"]["warnings"])
