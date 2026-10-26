# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests of the 3D panel method agent tool and MCP interface.

Tests verify results against direct panel3d solves, work budget and mesh limits,
unexposed feature refusals, ground strikes, MCP server validation, and Cp handling.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from ventorum.agent import call_tool, get_tool_schemas, handle_message
from ventorum.agent.schemas import MAX_PANEL_CALL_WORK, MAX_PANEL_PANELS
from ventorum.agent.tools import panel_analysis
from ventorum.solvers.factory import make_solver

SMALL_MESH = {"n_chord": 6, "n_span": 8}
RECT = {"span_m": 8.0, "chord_m": 1.0}


def strict_json(payload):
    """Verify that the payload is valid strict JSON."""
    return json.loads(json.dumps(payload, allow_nan=False))


def test_constants():
    """Module constants have expected values."""
    assert MAX_PANEL_PANELS == 15000
    assert MAX_PANEL_CALL_WORK == 2.0e12


def test_panel_analysis_equals_direct_solve():
    """panel_analysis produces the same results as direct make_solver('panel3d')."""
    cond = {"V_inf_m_s": 40.0, "alpha_deg": 4.0, "beta_deg": 0.0}
    p = panel_analysis(wing=RECT, flight_condition=cond, **SMALL_MESH)
    strict_json(p)
    assert p["status"] == "success", p

    from ventorum.agent.schemas import build_aircraft_from_spec, build_flight_condition_from_spec
    from ventorum.core.datatypes import SolverSettings

    ac = build_aircraft_from_spec(wing=RECT)
    fc = build_flight_condition_from_spec(cond)
    sett = SolverSettings(
        solver_type="panel3d",
        n_chord=SMALL_MESH["n_chord"],
        n_panels=SMALL_MESH["n_span"],
        panel_kutta="morino",
        panel_backend="auto",
        panel_ground_solver="auto",
        panel_cache=True,
    )
    solver = make_solver("panel3d")
    direct = solver.solve(ac, fc, sett)

    m = p["metrics"]
    assert m["CL"] == pytest.approx(direct.totals.CL, abs=1e-6)
    assert m["CDi"] == pytest.approx(direct.totals.CDi, abs=1e-6)
    assert m["CD"] == pytest.approx(direct.totals.CDi_nearfield, abs=1e-6)
    assert m["CD_pressure"] == pytest.approx(direct.totals.CDi_nearfield, abs=1e-6)
    assert m["Cm"] == pytest.approx(direct.totals.Cm, abs=1e-6)
    assert m["CY"] == pytest.approx(direct.totals.CY, abs=1e-6)
    assert m["Cl"] == pytest.approx(direct.totals.Cl, abs=1e-6)
    assert m["Cn"] == pytest.approx(direct.totals.Cn, abs=1e-6)

    assert p["trust"]["score"] == pytest.approx(direct.totals.trust.score, abs=1e-6)
    assert p["dispatch"]["backend"] == direct.details["dispatch"]["backend"]
    assert p["dispatch"]["device"] == direct.details["dispatch"]["device"]
    assert p["dispatch"]["ground_solver"] == direct.details["dispatch"]["ground_solver"]


def test_panel_analysis_aircraft_with_sections():
    """panel_analysis supports aircraft specifications with explicit section shapes."""
    ac_spec = {
        "name": "Biplane",
        "surfaces": [
            {
                "name": "MainWing",
                "span_m": 8.0,
                "sections": [
                    {"y_frac": 0.0, "chord_m": 1.2, "shape": "2412"},
                    {"y_frac": 1.0, "chord_m": 0.8, "shape": "0012"},
                ],
            }
        ],
        "ref_point_m": [0.25, 0.0, 0.0],
    }
    p = panel_analysis(aircraft=ac_spec, flight_condition={"alpha_deg": 3.0}, **SMALL_MESH)
    strict_json(p)
    assert p["status"] == "success", p
    assert p["metrics"]["CL"] > 0.0
    assert p["metrics"]["solver"] == "panel3d"


def test_panel_analysis_mesh_refusal():
    """A mesh larger than MAX_PANEL_PANELS is refused before solving."""
    with patch("ventorum.agent.tools.MAX_PANEL_PANELS", 50):
        p = panel_analysis(wing=RECT, flight_condition={"alpha_deg": 2.0}, **SMALL_MESH)
        assert p["status"] == "error", p
        assert p["error"]["type"] == "invalid_input"
        assert "mesh is too large" in p["error"]["message"].lower()


def test_panel_analysis_budget_refusal():
    """A call with estimated operations above MAX_PANEL_CALL_WORK is refused."""
    with patch("ventorum.agent.tools.MAX_PANEL_CALL_WORK", 100):
        p = panel_analysis(wing=RECT, flight_condition={"alpha_deg": 2.0}, **SMALL_MESH)
        assert p["status"] == "error", p
        assert p["error"]["type"] == "invalid_input"
        msg = p["error"]["message"].lower()
        assert "budget" in msg
        assert "estimate" in msg
        assert "fewer" in msg


def test_panel_analysis_refuses_controls():
    """Control surfaces are not exposed yet and are refused with a clear message."""
    wing_ctrl = {
        **RECT,
        "controls": [
            {"name": "aileron", "hinge_x_c": 0.75, "deflection_deg": 5.0},
        ],
    }
    p = panel_analysis(wing=wing_ctrl, **SMALL_MESH)
    assert p["status"] == "error", p
    assert p["error"]["type"] == "invalid_input"
    assert "controls" in p["error"]["message"].lower()
    assert "not exposed yet" in p["error"]["message"].lower()


def test_panel_analysis_refuses_pressure_kutta():
    """The pressure Kutta condition is not exposed yet and is refused."""
    p = panel_analysis(wing=RECT, settings={"panel_kutta": "pressure"}, **SMALL_MESH)
    assert p["status"] == "error", p
    assert p["error"]["type"] == "invalid_input"
    assert "pressure kutta" in p["error"]["message"].lower()
    assert "not exposed yet" in p["error"]["message"].lower()


def test_panel_analysis_ground_strike_returns_error():
    """A ground strike returns an error result of type ground_strike without crashing."""
    p = panel_analysis(wing=RECT, flight_condition={"alpha_deg": 20.0, "h_m": 0.001}, **SMALL_MESH)
    assert p["status"] == "error", p
    assert p["error"]["type"] == "ground_strike"


def test_panel_analysis_cp_option():
    """Cp and centroids are present in the output only when requested."""
    no_cp = panel_analysis(wing=RECT, cp=False, **SMALL_MESH)
    assert no_cp["status"] == "success"
    assert "cp" not in no_cp
    assert "cp" not in no_cp["metrics"]
    assert "panel_centroids_m" not in no_cp
    assert "centroids_m" not in no_cp

    with_cp = panel_analysis(wing=RECT, cp=True, **SMALL_MESH)
    assert with_cp["status"] == "success"
    assert "cp" in with_cp
    assert isinstance(with_cp["cp"], list)
    assert len(with_cp["cp"]) > 0
    assert "panel_centroids_m" in with_cp
    assert "centroids_m" in with_cp
    assert isinstance(with_cp["panel_centroids_m"], list)
    assert len(with_cp["panel_centroids_m"]) == len(with_cp["cp"])
    assert len(with_cp["panel_centroids_m"][0]) == 3


def test_panel_analysis_mcp_validation():
    """Schema validates through the MCP server tools/list and tools/call paths."""
    mcp_schemas = get_tool_schemas("mcp")
    panel_schema = next((s for s in mcp_schemas if s["name"] == "ventorum_panel_analysis"), None)
    assert panel_schema is not None
    assert panel_schema["inputSchema"]["additionalProperties"] is False

    req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "ventorum_panel_analysis",
            "arguments": {
                "wing": RECT,
                "n_chord": SMALL_MESH["n_chord"],
                "n_span": SMALL_MESH["n_span"],
            },
        },
    }
    res_str = handle_message(json.dumps(req))
    assert res_str is not None
    reply = json.loads(res_str)
    assert reply.get("error") is None
    assert reply["result"]["isError"] is False
    content = json.loads(reply["result"]["content"][0]["text"])
    assert content["status"] == "success"
    assert content["metrics"]["CL"] > 0.0


def test_panel_analysis_dispatcher_resolution():
    """Tool resolves with and without ventorum_ prefix."""
    p1 = call_tool("ventorum_panel_analysis", {"wing": RECT, **SMALL_MESH})
    assert p1["status"] == "success"
    p2 = call_tool("panel_analysis", {"wing": RECT, **SMALL_MESH})
    assert p2["status"] == "success"

def test_panel_analysis_budget_refusal_before_solve():
    """A case over budget is refused before any solve runs (solver is never called)."""
    with patch("ventorum.solvers.factory.make_solver") as mock_solver:
        mock_solver.side_effect = RuntimeError("Solve was invoked despite being over budget!")
        with patch("ventorum.agent.tools.MAX_PANEL_CALL_WORK", 100):
            p = panel_analysis(wing=RECT, flight_condition={"alpha_deg": 2.0}, **SMALL_MESH)
            assert p["status"] == "error", p
            assert p["error"]["type"] == "invalid_input"
            assert "budget" in p["error"]["message"].lower()
            mock_solver.assert_not_called()


def test_panel_analysis_budget_asymmetric_counts_full_mesh():
    """Work budget counts the full mesh when beta != 0 or symmetric is False."""
    # Half mesh: (2*8 + 1) * 6 = 102 panels. Est ops = 102^3 / 3 = 353,736 ops.
    # Full mesh: 2 * 102 = 204 panels. Est ops = 204^3 / 3 = 2,829,888 ops.
    with patch("ventorum.agent.tools.MAX_PANEL_CALL_WORK", 400_000):
        sym_ok = panel_analysis(wing=RECT, flight_condition={"alpha_deg": 2.0, "beta_deg": 0.0}, **SMALL_MESH)
        assert sym_ok["status"] == "success", sym_ok

        beta_over = panel_analysis(wing=RECT, flight_condition={"alpha_deg": 2.0, "beta_deg": 5.0}, **SMALL_MESH)
        assert beta_over["status"] == "error", beta_over
        assert beta_over["error"]["type"] == "invalid_input"
        assert "budget" in beta_over["error"]["message"].lower()

        asym_surf = {"semi_span_m": 4.0, "chord_m": 1.0, "symmetric": False}
        asym_over = panel_analysis(wing=asym_surf, flight_condition={"alpha_deg": 2.0}, **SMALL_MESH)
        assert asym_over["status"] == "error", asym_over
        assert asym_over["error"]["type"] == "invalid_input"
        assert "budget" in asym_over["error"]["message"].lower()


def test_panel_analysis_strict_validation():
    """Strict input validation: unknown settings, non-booleans, and conflicting inputs are refused."""
    # Unknown key in settings
    p = panel_analysis(wing=RECT, settings={"unknown_setting": 1}, **SMALL_MESH)
    assert p["status"] == "error"
    assert p["error"]["type"] == "invalid_input"
    assert "unknown key" in p["error"]["message"].lower()

    # Old alias keys in settings are refused
    for old_key in ("n_span", "kutta", "backend", "ground_solver", "cp", "return_cp"):
        p = panel_analysis(wing=RECT, settings={old_key: "auto"}, **SMALL_MESH)
        assert p["status"] == "error"
        assert p["error"]["type"] == "invalid_input"
        assert "unknown key" in p["error"]["message"].lower()

    # Boolean option that is not a real boolean
    p_cache = panel_analysis(wing=RECT, settings={"panel_cache": "no"}, **SMALL_MESH)
    assert p_cache["status"] == "error"
    assert p_cache["error"]["type"] == "invalid_input"
    assert "boolean" in p_cache["error"]["message"].lower()

    p_cp = panel_analysis(wing=RECT, cp="yes", **SMALL_MESH)
    assert p_cp["status"] == "error"
    assert p_cp["error"]["type"] == "invalid_input"
    assert "boolean" in p_cp["error"]["message"].lower()

    # Top-level n_chord differing from settings.n_chord
    p_nc = panel_analysis(wing=RECT, n_chord=10, settings={"n_chord": 20}, n_span=SMALL_MESH["n_span"])
    assert p_nc["status"] == "error"
    assert p_nc["error"]["type"] == "invalid_input"
    assert "differs from" in p_nc["error"]["message"].lower()

    # Top-level n_span differing from settings.n_panels
    p_ns = panel_analysis(wing=RECT, n_span=10, settings={"n_panels": 20}, n_chord=SMALL_MESH["n_chord"])
    assert p_ns["status"] == "error"
    assert p_ns["error"]["type"] == "invalid_input"
    assert "differs from" in p_ns["error"]["message"].lower()

    # Section shape that is not a string
    bad_sec_ac = {
        "surfaces": [
            {
                "name": "Wing",
                "span_m": 8.0,
                "sections": [
                    {"y_frac": 0.0, "chord_m": 1.0, "shape": 123},
                    {"y_frac": 1.0, "chord_m": 1.0},
                ],
            }
        ]
    }
    p_sec = panel_analysis(aircraft=bad_sec_ac, **SMALL_MESH)
    assert p_sec["status"] == "error"
    assert p_sec["error"]["type"] == "invalid_input"
    assert "string" in p_sec["error"]["message"].lower()


def test_wing_analysis_with_and_without_shape():
    """wing_analysis accepts shape; cambered shape differs from default, symmetric shape matches."""
    from ventorum.agent.tools import wing_analysis

    res_none = wing_analysis(wing=RECT, flight_condition={"alpha_deg": 4.0}, detail_level="summary")
    assert res_none["status"] == "success", res_none

    res_sym = wing_analysis(wing={**RECT, "shape": "0012"}, flight_condition={"alpha_deg": 4.0}, detail_level="summary")
    assert res_sym["status"] == "success", res_sym

    res_cam = wing_analysis(wing={**RECT, "shape": "2412"}, flight_condition={"alpha_deg": 4.0}, detail_level="summary")
    assert res_cam["status"] == "success", res_cam

    assert res_sym["metrics"]["CL"] == pytest.approx(res_none["metrics"]["CL"], abs=1e-5)
    assert abs(res_cam["metrics"]["CL"] - res_none["metrics"]["CL"]) > 0.05


def test_panel_analysis_axes_all():
    """panel_analysis with axes='all' matches direct solve."""
    from ventorum.agent.schemas import build_aircraft_from_spec, build_flight_condition_from_spec
    from ventorum.core.datatypes import SolverSettings

    cond = {"V_inf_m_s": 40.0, "alpha_deg": 3.0, "beta_deg": 2.0}
    p = panel_analysis(wing=RECT, flight_condition=cond, axes="all", **SMALL_MESH)
    strict_json(p)
    assert p["status"] == "success", p

    ac = build_aircraft_from_spec(wing=RECT)
    fc = build_flight_condition_from_spec(cond)
    sett = SolverSettings(
        solver_type="panel3d",
        n_chord=SMALL_MESH["n_chord"],
        n_panels=SMALL_MESH["n_span"],
        panel_kutta="morino",
        panel_backend="auto",
        panel_ground_solver="auto",
        panel_cache=True,
    )
    direct = make_solver("panel3d").solve(ac, fc, sett)

    m = p["metrics"]
    assert "moments" in m
    assert set(m["moments"].keys()) == {"body", "stability", "wind"}
    assert m["moments"]["body"]["Cl"] == pytest.approx(direct.totals.Cl, abs=1e-6)
    assert m["moments"]["body"]["Cm"] == pytest.approx(direct.totals.Cm, abs=1e-6)
    assert m["moments"]["body"]["Cn"] == pytest.approx(direct.totals.Cn, abs=1e-6)
    assert m["CL"] == pytest.approx(direct.totals.CL, abs=1e-6)
    assert m["CD"] == pytest.approx(direct.totals.CDi_nearfield, abs=1e-6)
    assert m["CDi"] == pytest.approx(direct.totals.CDi, abs=1e-6)


def test_panel_analysis_detail_levels():
    """panel_analysis with detail_level='summary' and 'full' matches direct solve."""
    from ventorum.agent.schemas import build_aircraft_from_spec, build_flight_condition_from_spec
    from ventorum.core.datatypes import SolverSettings

    cond = {"V_inf_m_s": 40.0, "alpha_deg": 3.0, "beta_deg": 0.0}
    ac = build_aircraft_from_spec(wing=RECT)
    fc = build_flight_condition_from_spec(cond)
    sett = SolverSettings(
        solver_type="panel3d",
        n_chord=SMALL_MESH["n_chord"],
        n_panels=SMALL_MESH["n_span"],
        panel_kutta="morino",
        panel_backend="auto",
        panel_ground_solver="auto",
        panel_cache=True,
    )
    direct = make_solver("panel3d").solve(ac, fc, sett)

    p_sum = panel_analysis(wing=RECT, flight_condition=cond, detail_level="summary", **SMALL_MESH)
    strict_json(p_sum)
    assert p_sum["status"] == "success"
    assert "geometry" not in p_sum
    assert "sectional_diagnostics" not in p_sum
    assert "spanwise_distributions" not in p_sum
    assert p_sum["metrics"]["CL"] == pytest.approx(direct.totals.CL, abs=1e-6)

    p_full = panel_analysis(wing=RECT, flight_condition=cond, detail_level="full", **SMALL_MESH)
    strict_json(p_full)
    assert p_full["status"] == "success"
    assert "geometry" in p_full
    assert "sectional_diagnostics" in p_full
    assert "spanwise_distributions" in p_full
    assert p_full["metrics"]["CL"] == pytest.approx(direct.totals.CL, abs=1e-6)
    assert p_full["dispatch"]["backend"] == direct.details["dispatch"]["backend"]


def test_panel_analysis_sideslip_beta_matches_direct():
    """panel_analysis with sideslip beta != 0 matches direct solve."""
    from ventorum.agent.schemas import build_aircraft_from_spec, build_flight_condition_from_spec
    from ventorum.core.datatypes import SolverSettings

    cond = {"V_inf_m_s": 35.0, "alpha_deg": 3.0, "beta_deg": 5.0}
    p = panel_analysis(wing=RECT, flight_condition=cond, **SMALL_MESH)
    strict_json(p)
    assert p["status"] == "success", p

    ac = build_aircraft_from_spec(wing=RECT)
    fc = build_flight_condition_from_spec(cond)
    sett = SolverSettings(
        solver_type="panel3d",
        n_chord=SMALL_MESH["n_chord"],
        n_panels=SMALL_MESH["n_span"],
        panel_kutta="morino",
        panel_backend="auto",
        panel_ground_solver="auto",
        panel_cache=True,
    )
    direct = make_solver("panel3d").solve(ac, fc, sett)

    m = p["metrics"]
    assert m["CL"] == pytest.approx(direct.totals.CL, abs=1e-6)
    assert m["CY"] == pytest.approx(direct.totals.CY, abs=1e-6)
    assert m["Cl"] == pytest.approx(direct.totals.Cl, abs=1e-6)
    assert m["Cm"] == pytest.approx(direct.totals.Cm, abs=1e-6)
    assert m["Cn"] == pytest.approx(direct.totals.Cn, abs=1e-6)
    assert m["CD"] == pytest.approx(direct.totals.CDi_nearfield, abs=1e-6)
    assert m["CDi"] == pytest.approx(direct.totals.CDi, abs=1e-6)


def test_panel_analysis_ground_effect_matches_direct():
    """panel_analysis in ground effect matches direct solve."""
    from ventorum.agent.schemas import build_aircraft_from_spec, build_flight_condition_from_spec
    from ventorum.core.datatypes import SolverSettings

    cond = {"V_inf_m_s": 35.0, "alpha_deg": 3.0, "h_m": 1.5}
    p = panel_analysis(wing=RECT, flight_condition=cond, **SMALL_MESH)
    strict_json(p)
    assert p["status"] == "success", p

    ac = build_aircraft_from_spec(wing=RECT)
    fc = build_flight_condition_from_spec(cond)
    sett = SolverSettings(
        solver_type="panel3d",
        n_chord=SMALL_MESH["n_chord"],
        n_panels=SMALL_MESH["n_span"],
        panel_kutta="morino",
        panel_backend="auto",
        panel_ground_solver="auto",
        panel_cache=True,
    )
    direct = make_solver("panel3d").solve(ac, fc, sett)

    m = p["metrics"]
    assert m["CL"] == pytest.approx(direct.totals.CL, abs=1e-6)
    assert m["CD"] == pytest.approx(direct.totals.CDi_nearfield, abs=1e-6)
    assert m["CDi"] == pytest.approx(direct.totals.CDi, abs=1e-6)
    assert m["Cm"] == pytest.approx(direct.totals.Cm, abs=1e-6)
    assert p["dispatch"]["ground_solver"] == direct.details["dispatch"]["ground_solver"]

