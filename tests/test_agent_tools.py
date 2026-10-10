# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Tests of the strict AI agent tools, the dispatcher and the MCP request handler.
"""

from __future__ import annotations

import json

import pytest

from ventorum.agent import (
    AGENT_TOOL_DEFINITIONS,
    batch_evaluate,
    call_tool,
    get_tool_schemas,
    ground_effect,
    handle_message,
    polar_sweep,
    stability_derivatives,
    wing_analysis,
)
from ventorum.agent.schemas import CONDITION_PROPS, SETTINGS_PROPS, SURFACE_PROPS

FAST = {"n_panels": 8}
RECT = {"span_m": 10.0, "chord_m": 1.0}


def strict_json(payload):
    """The payload must be strict JSON (no NaN, no infinity)."""
    return json.loads(json.dumps(payload, allow_nan=False))


def assert_error(payload, kind, text=None):
    assert payload["status"] == "error", payload
    assert payload["error"]["type"] == kind, payload
    if text is not None:
        assert text in payload["error"]["message"], payload["error"]["message"]
    strict_json(payload)


# ── Schemas ──────────────────────────────────────────────────────────────────

def test_schema_exporters():
    n = len(AGENT_TOOL_DEFINITIONS)
    assert n == 11  # six analysis tools, trim, error bars, undeformed nodes, and the two machine tools
    openai = get_tool_schemas("openai")
    assert all(s["type"] == "function" and s["function"]["name"].startswith("ventorum_") for s in openai)
    anthropic = get_tool_schemas("anthropic")
    assert all("input_schema" in s for s in anthropic)
    mcp = get_tool_schemas("mcp")
    assert all("inputSchema" in s for s in mcp)
    assert all(s["inputSchema"]["additionalProperties"] is False for s in mcp)
    gemini = get_tool_schemas("gemini")
    assert all("parameters" in s for s in gemini)
    assert "additionalProperties" not in json.dumps(gemini)
    for fmt in (openai, anthropic, mcp, gemini):
        assert len(fmt) == n
        strict_json(fmt)


def test_schema_keys_match_parser_tables():
    wa = next(t for t in AGENT_TOOL_DEFINITIONS if t["name"] == "ventorum_wing_analysis")["parameters"]
    surface = wa["properties"]["wing"]["anyOf"][0]
    assert set(surface["properties"]) == set(SURFACE_PROPS)
    assert surface["additionalProperties"] is False
    assert set(wa["properties"]["flight_condition"]["properties"]) == set(CONDITION_PROPS)
    assert set(wa["properties"]["settings"]["properties"]) == set(SETTINGS_PROPS)
    assert "tip_twist_deg" in surface["properties"]
    assert "NEGATIVE" in surface["properties"]["tip_twist_deg"]["description"]


# ── Wing analysis ────────────────────────────────────────────────────────────

def test_wing_analysis_success():
    p = wing_analysis({"span_m": 12.0, "root_chord_m": 1.6, "tip_chord_m": 0.8},
                      {"V_inf_m_s": 45.0, "alpha_deg": 5.0}, FAST, "summary")
    strict_json(p)
    assert p["status"] == "success"
    m = p["metrics"]
    for key in ("CL", "CDi", "CDp", "CD", "L_over_D", "e", "Cm", "Cl", "Cn", "CY", "solver", "converged"):
        assert key in m
    assert m["CL"] > 0.3 and m["solver"] == "vlm" and m["converged"] is True
    assert m["CDp"] is None and m["drag_basis"].startswith("CDi only")
    assert p["executive_summary"].startswith("[Ventorum RESULT]")
    assert "NOT calibrated" in p["trust"]["note"]
    assert "heuristic_bands_not_calibrated" in p["trust"]
    assert p["condition_used"]["alpha_deg"] == 5.0


def test_wing_analysis_detail_levels():
    s = wing_analysis(RECT, None, FAST, "summary")
    assert "geometry" not in s and "spanwise_distributions" not in s
    st = wing_analysis(RECT, None, FAST, "standard")
    assert "geometry" in st and "sectional_diagnostics" in st and "spanwise_distributions" not in st
    f = wing_analysis(RECT, None, FAST, "full")
    assert f["spanwise_distributions"][0]["gamma_m2_s"]
    strict_json(f)
    assert_error(wing_analysis(RECT, None, FAST, "verbose"), "invalid_input", "detail_level")


def test_constant_chord_area():
    p = wing_analysis({"span_m": 10.0, "chord_m": 1.0}, None, FAST, "standard")
    g = p["geometry"]
    assert g["S_ref_m2"] == pytest.approx(10.0)
    assert g["b_ref_m"] == pytest.approx(10.0)
    assert g["c_ref_m"] == pytest.approx(1.0)


def test_tip_twist_sign():
    base = wing_analysis(RECT, {"alpha_deg": 4.0}, FAST)["metrics"]["CL"]
    washout = wing_analysis({**RECT, "tip_twist_deg": -3.0}, {"alpha_deg": 4.0}, FAST)["metrics"]["CL"]
    washin = wing_analysis({**RECT, "tip_twist_deg": 3.0}, {"alpha_deg": 4.0}, FAST)["metrics"]["CL"]
    assert washout < base < washin


def test_altitude_gives_isa_density():
    p = wing_analysis(RECT, {"altitude_m": 3000.0}, FAST, "summary")
    assert p["condition_used"]["rho_kg_m3"] == pytest.approx(0.9093, abs=2e-3)
    assert "ISA" in p["condition_used"]["density_source"]


# ── Strict input refusals ────────────────────────────────────────────────────

def test_refuses_unknown_and_ambiguous_keys():
    p = wing_analysis(RECT, {"alpha": 8.0}, FAST)
    assert_error(p, "invalid_input", "'alpha'")
    assert "alpha_deg" in p["error"]["message"]
    assert_error(wing_analysis({**RECT, "washout": 3.0}, None, FAST), "invalid_input", "tip_twist_deg")
    assert_error(wing_analysis({"span_m": 10.0, "aspect_ratio": 8.0, "chord_m": 1.0}, None, FAST),
                 "invalid_input", "aspect_ratio")
    assert_error(wing_analysis({"span_m": 10.0, "mean_chord": 1.0}, None, FAST), "invalid_input", "mean_chord")
    assert_error(wing_analysis(RECT, None, {"panels": 10}), "invalid_input", "n_panels")
    assert_error(wing_analysis({**RECT, "span": 12.0}, None, FAST), "invalid_input", "span_m")


@pytest.mark.parametrize("wing, cond, key", [
    ({"span": 10.0, "chord_m": 1.0}, None, "span"),
    ({"span_m": 10.0, "chord": 1.0}, None, "chord"),
    ({"semi_span": 5.0, "chord_m": 1.0}, None, "semi_span"),
    ({"span_m": 10.0, "root_chord": 1.0, "tip_chord_m": 0.5}, None, "root_chord"),
    ({"span_m": 10.0, "root_chord_m": 1.0, "tip_chord": 0.5}, None, "tip_chord"),
    ({**RECT, "sweep_deg": 10.0}, None, "sweep_deg"),
    (RECT, {"V_inf": 30.0}, "V_inf"),
    (RECT, {"rho": 1.0}, "rho"),
])
def test_refuses_former_aliases(wing, cond, key):
    """Defect 4: the parsers accept only the schema keys (additionalProperties false)."""
    p = wing_analysis(wing, cond, FAST, "summary")
    assert_error(p, "invalid_input", f"unknown key '{key}'")
    assert "Hint" in p["error"]["message"]


def test_refuses_wrong_types():
    assert_error(wing_analysis(RECT, {"V_inf_m_s": "60 m/s"}, FAST), "invalid_input", "V_inf_m_s")
    assert_error(wing_analysis({"span_m": "12 m", "chord_m": 1.0}, None, FAST), "invalid_input", "span_m")
    assert_error(wing_analysis({**RECT, "symmetric": "false"}, None, FAST), "invalid_input", "symmetric")
    assert_error(wing_analysis(RECT, {"alpha_deg": True}, FAST), "invalid_input", "alpha_deg")


def test_refuses_rho_and_altitude_together():
    assert_error(wing_analysis(RECT, {"rho_kg_m3": 1.0, "altitude_m": 500.0}, FAST), "invalid_input", "altitude_m")


def test_refuses_bad_geometry():
    assert_error(wing_analysis({"span_m": 10.0, "root_chord_m": 1.0}, None, FAST), "invalid_input", "tip_chord_m")
    assert_error(wing_analysis({"span_m": 10.0, "chord_m": -1.0}, None, FAST), "invalid_input", "chord_m")
    assert_error(wing_analysis({"span_m": 10.0, "chord_m": 1.0, "root_chord_m": 1.0, "tip_chord_m": 0.5}, None, FAST),
                 "invalid_input", "exactly one chord")
    assert_error(wing_analysis(RECT, None, {"n_panels": 2}), "invalid_input", "n_panels")
    assert_error(wing_analysis({**RECT, "mirror": True}, None, FAST), "invalid_input", "mirror")


def test_sections_and_multi_surface_spec():
    wing = {"semi_span_m": 5.0, "sections": [
        {"y_frac": 0.0, "chord_m": 1.2},
        {"y_frac": 0.5, "chord_m": 1.0, "twist_deg": -1.0},
        {"y_frac": 1.0, "chord_m": 0.6, "twist_deg": -2.0, "airfoil": {"cd0": 0.01}},
    ]}
    p = wing_analysis(wing, None, FAST, "standard")
    assert p["status"] == "success"
    assert p["geometry"]["surfaces"][0]["n_sections"] == 3
    ac = {"surfaces": [RECT, {"name": "tail", "span_m": 3.0, "chord_m": 0.5, "position_m": [4.0, 0.0, 0.0]}],
          "ref_point_m": [0.3, 0.0, 0.0]}
    p = wing_analysis(ac, None, FAST, "standard")
    assert p["status"] == "success"
    assert p["geometry"]["S_ref_m2"] == pytest.approx(10.0)
    assert p["geometry"]["ref_point_m"] == [0.3, 0.0, 0.0]
    assert len(p["geometry"]["surfaces"]) == 2


# ── Polar sweep ──────────────────────────────────────────────────────────────

def test_polar_sweep():
    p = polar_sweep({**RECT, "airfoil": {"cd0": 0.01}}, 0.0, 8.0, 2.0, {"V_inf_m_s": 40.0}, FAST, "standard")
    strict_json(p)
    assert p["status"] == "success"
    assert len(p["polar_table"]) == 5
    s = p["polar_summary"]
    assert 3.0 < s["CL_alpha_per_rad"] < 6.3
    assert s["drag_basis"].startswith("CD_total")
    assert s["max_L_over_D"] > 0
    assert s["alpha_at_max_L_over_D_deg"] in (0.0, 2.0, 4.0, 6.0, 8.0)
    assert_error(polar_sweep(RECT, 0.0, 4.0, 2.0, {"alpha_deg": 3.0}, FAST), "invalid_input", "alpha_deg")


# ── Ground effect ────────────────────────────────────────────────────────────

def test_ground_effect_strike_rows_and_strict_json():
    p = ground_effect(RECT, [0.05, 0.3, 0.6, 1.5], alpha_deg=4.0, settings=FAST)
    strict_json(p)
    assert p["status"] == "success"
    rows = {r["h_m"]: r for r in p["rows"]}
    assert rows[0.05]["status"] == "ground_strike"
    assert "strike" in rows[0.05]["message"].lower()
    ok = [r for r in p["rows"] if r["status"] == "ok"]
    assert len(ok) == 3
    assert len({r["n_chord"] for r in ok}) == 1
    assert p["settings_used"]["n_chord"] == ok[0]["n_chord"]
    assert ok[0]["CL_ratio"] > ok[-1]["CL_ratio"] > 1.0
    assert ok[0]["induced_drag_factor_ratio"] < 1.0
    assert ok[0]["phi_strike_limit_deg"] > 0
    assert p["summary"]["n_ground_strike"] == 1
    assert len(p["irodov"]["rows"]) == 3
    assert "irodov_margin" in p["irodov"]["rows"][0]
    assert "0.05 m (ground_strike)" in p["executive_summary"]


def test_ground_effect_all_heights_strike_is_error():
    p = ground_effect(RECT, [0.01, 0.02], alpha_deg=4.0, settings=FAST)
    assert_error(p, "ground_strike")
    assert len(p["rows"]) == 2


def test_ground_effect_refuses_lifting_line_near_ground():
    p = ground_effect(RECT, [0.3], alpha_deg=4.0, settings={"solver": "linear", "n_panels": 8})
    assert_error(p, "invalid_method")


# ── Stability derivatives ────────────────────────────────────────────────────

def test_static_margin_sign():
    fwd = stability_derivatives(RECT, {"alpha_deg": 4.0}, x_cg_m=0.10, settings=FAST)
    aft = stability_derivatives(RECT, {"alpha_deg": 4.0}, x_cg_m=0.45, settings=FAST)
    for p in (fwd, aft):
        strict_json(p)
        assert p["status"] == "success"
    sf, sa = fwd["stability_derivatives"], aft["stability_derivatives"]
    # Neutral point of a rectangular wing is near the quarter chord.
    assert 0.2 < sf["neutral_point_x_m"] < 0.3
    assert sf["static_margin_fraction"] > 0
    assert fwd["stability_assessment"]["pitch"].startswith("statically stable")
    assert sa["static_margin_fraction"] < 0
    assert aft["stability_assessment"]["pitch"].startswith("statically UNSTABLE")
    assert sa["Cm_alpha_per_rad"] > 0 > sf["Cm_alpha_per_rad"]


def test_stability_in_ground_effect_uses_height():
    free = stability_derivatives(RECT, {"alpha_deg": 4.0}, x_cg_m=0.1, settings=FAST)
    ige = stability_derivatives(RECT, {"alpha_deg": 4.0, "h_m": 0.5}, x_cg_m=0.1, settings=FAST)
    assert ige["status"] == "success"
    assert ige["stability_derivatives"]["CL_alpha_per_rad"] > free["stability_derivatives"]["CL_alpha_per_rad"]
    assert any("Irodov" in n for n in ige["notes"])
    assert_error(stability_derivatives(RECT, {"alpha_deg": 4.0, "h_m": 0.01}, settings=FAST), "ground_strike")


# ── Batch ────────────────────────────────────────────────────────────────────

def test_batch_reports_failed_candidates():
    cands = [
        {"name": "AR6", "span_m": 6.0, "chord_m": 1.0},
        {"name": "AR10", "span_m": 10.0, "chord_m": 1.0},
        {"name": "bad", "span_m": 8.0, "chord": "1 m"},
    ]
    p = batch_evaluate(cands, {"alpha_deg": 4.0}, "min_CDi", FAST)
    strict_json(p)
    assert p["status"] == "success"
    assert p["n_candidates"] == 3 and p["n_succeeded"] == 2 and p["n_failed"] == 1
    assert p["failed"][0]["name"] == "bad"
    assert p["failed"][0]["error"]["type"] == "invalid_input"
    assert "candidates[2]" in p["failed"][0]["error"]["message"]
    assert [r["name"] for r in p["rankings"]] == ["AR10", "AR6"]
    assert "bad" in p["executive_summary"]


def test_batch_all_failed_is_error():
    p = batch_evaluate([{"span_m": 8.0}], None, "max_CL", FAST)
    assert_error(p, "invalid_input")
    assert len(p["failed"]) == 1


# ── Dispatcher ───────────────────────────────────────────────────────────────

def test_dispatcher_json_string_and_prefix():
    args = json.dumps({"wing": RECT, "flight_condition": {"alpha_deg": 4.0}, "settings": FAST,
                       "detail_level": "summary"})
    p = call_tool("ventorum_wing_analysis", args)
    assert p["status"] == "success" and p["tool"] == "ventorum_wing_analysis"
    assert call_tool("wing_analysis", args)["status"] == "success"


def test_dispatcher_error_types():
    assert_error(call_tool("ventorum_wing_analysis", "NOT JSON"), "invalid_input", "JSON")
    assert_error(call_tool("ventorum_wing_analysis", "[1, 2]"), "invalid_input", "object")
    p = call_tool("ventorum_nonexistent", {})
    assert_error(p, "invalid_input", "ventorum_ground_effect")
    assert_error(call_tool("ventorum_wing_analysis", {"wing": RECT, "alpha_deg": 8}), "invalid_input", "alpha_deg")
    assert_error(call_tool("ventorum_wing_analysis", {}), "invalid_input", "wing")
    assert_error(call_tool("ventorum_ground_effect", {"wing": RECT, "heights_m": [0.01], "settings": FAST}),
                 "ground_strike")
    assert_error(call_tool("ventorum_ground_effect",
                           {"wing": RECT, "heights_m": [0.3], "settings": {"solver": "linear"}}), "invalid_method")
    assert_error(call_tool(None, {}), "invalid_input")


# ── MCP request handling ─────────────────────────────────────────────────────

def _rpc(line):
    out = handle_message(line)
    return None if out is None else json.loads(out)


def test_mcp_parse_error_and_invalid_requests():
    r = _rpc("{not json")
    assert r["id"] is None and r["error"]["code"] == -32700
    r = _rpc("42")
    assert r["error"]["code"] == -32600
    r = _rpc(json.dumps({"jsonrpc": "2.0", "id": 1}))
    assert r["id"] == 1 and r["error"]["code"] == -32600
    r = _rpc(json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": None}))
    assert r["id"] == 2 and r["error"]["code"] == -32602
    r = _rpc(json.dumps({"jsonrpc": "2.0", "id": 3, "method": "no/such"}))
    assert r["error"]["code"] == -32601
    r = _rpc(json.dumps({"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"arguments": {}}}))
    assert r["error"]["code"] == -32602
    assert _rpc("[]")["error"]["code"] == -32600


def test_mcp_notification_gets_no_reply():
    assert handle_message(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"})) is None
    assert handle_message(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized",
                                      "params": None})) is None


def test_mcp_initialize_list_and_call():
    r = _rpc(json.dumps({"jsonrpc": "2.0", "id": "a", "method": "initialize",
                         "params": {"protocolVersion": "2024-11-05"}}))
    assert r["result"]["protocolVersion"] == "2024-11-05"
    r = _rpc(json.dumps({"jsonrpc": "2.0", "id": "b", "method": "tools/list"}))
    assert len(r["result"]["tools"]) == 11  # with trim, error bars, undeformed nodes, and the two machine tools
    call = {"jsonrpc": "2.0", "id": 7, "method": "tools/call",
            "params": {"name": "ventorum_wing_analysis",
                       "arguments": {"wing": RECT, "settings": FAST, "detail_level": "summary"}}}
    r = _rpc(json.dumps(call))
    assert r["id"] == 7 and r["result"]["isError"] is False
    assert json.loads(r["result"]["content"][0]["text"])["status"] == "success"
    call["params"]["arguments"] = {"wing": RECT, "flight_condition": {"aoa": 3}}
    r = _rpc(json.dumps(call))
    assert r["result"]["isError"] is True


def test_mcp_batch_array_is_invalid_request():
    """Defect 5: MCP 2025-06-18 has no JSON-RPC batches."""
    batch = [{"jsonrpc": "2.0", "id": 1, "method": "ping"}]
    r = _rpc(json.dumps(batch))
    assert r == {"jsonrpc": "2.0", "id": None,
                 "error": {"code": -32600, "message": r["error"]["message"]}}
    assert _rpc(json.dumps([{"jsonrpc": "2.0", "method": "notifications/initialized"}]))["error"]["code"] == -32600


def test_mcp_unknown_protocol_version_gives_latest():
    r = _rpc(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                         "params": {"protocolVersion": "2099-01-01"}}))
    assert r["result"]["protocolVersion"] == "2025-06-18"
    r = _rpc(json.dumps({"jsonrpc": "2.0", "id": 2, "method": "initialize", "params": {}}))
    assert r["result"]["protocolVersion"] == "2025-06-18"


def test_mcp_server_version_is_the_package_version():
    import ventorum

    r = _rpc(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}))
    assert r["result"]["serverInfo"]["version"] == ventorum.__version__


def test_mcp_refuses_nan_and_infinity():
    for const in ("NaN", "Infinity", "-Infinity"):
        line = ('{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": '
                '"ventorum_wing_analysis", "arguments": {"wing": {"span_m": CONST, "chord_m": 1}}}}'
                ).replace("CONST", const)
        r = _rpc(line)
        assert r["id"] is None and r["error"]["code"] == -32700, r
    assert _rpc("NaN")["error"]["code"] == -32700
    assert_error(call_tool("ventorum_wing_analysis", '{"wing": {"span_m": NaN, "chord_m": 1}}'),
                 "invalid_input", "JSON")


def test_mcp_null_id_is_invalid_request():
    r = _rpc(json.dumps({"jsonrpc": "2.0", "id": None, "method": "ping"}))
    assert r["id"] is None and r["error"]["code"] == -32600


def test_mcp_unknown_tool_is_invalid_params():
    r = _rpc(json.dumps({"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                         "params": {"name": "no_such_tool", "arguments": {}}}))
    assert r["id"] == 5 and r["error"]["code"] == -32602 and "result" not in r
    # A tool execution error is still a result with isError true.
    r = _rpc(json.dumps({"jsonrpc": "2.0", "id": 6, "method": "tools/call",
                         "params": {"name": "ventorum_wing_analysis", "arguments": {"wing": {"span_m": 1}}}}))
    assert r["result"]["isError"] is True
    assert r["result"]["content"][0]["type"] == "text"


# ── Ground-effect rows carry trust ────────────────────────────────────────────

def test_ground_effect_rows_carry_trust():
    """Each ground-effect row reports trust_score, trust_rating, warnings and converged."""
    import ventorum as vt
    from ventorum.agent.schemas import build_aircraft_from_spec
    from ventorum.ground_effect import analyze_ground_effect

    p = ground_effect(RECT, [0.15, 0.5, 1.0], alpha_deg=4.0, settings={"n_panels": 12})
    strict_json(p)
    assert p["status"] == "success", p
    ok = [r for r in p["rows"] if r["status"] == "ok"]
    assert len(ok) == 3
    for r in ok:
        assert "trust_score" in r and "trust_rating" in r
        assert "warnings" in r and "converged" in r
        assert isinstance(r["trust_score"], float) and isinstance(r["trust_rating"], str)
        assert isinstance(r["warnings"], list) and r["converged"] is True
    low = next(r for r in ok if r["h_m"] == 0.15)
    assert low["trust_rating"] == "LOW"
    assert any("Extreme ground proximity" in w for w in low["warnings"])
    # The tool trust matches the library trust of the same case.
    lib = analyze_ground_effect(
        build_aircraft_from_spec(RECT), 0.15, alpha_deg=4.0,
        settings=vt.SolverSettings(n_panels=12, n_chord=int(p["settings_used"]["n_chord"])))
    assert low["trust_score"] == pytest.approx(lib.solver_result.totals.trust.score, abs=1e-3)
    # The summary gives the lowest (worst) rating of the rows.
    assert p["summary"]["trust_rating_lowest"] == "LOW"
    assert p["summary"]["trust_score_min"] == pytest.approx(low["trust_score"])


def test_ground_effect_summary_detail_keeps_trust():
    """The summary detail level keeps the trust fields of each row."""
    p = ground_effect(RECT, [0.5, 1.0], alpha_deg=4.0, settings=SMALL, detail_level="summary")
    assert p["status"] == "success", p
    for r in p["rows"]:
        assert {"h_m", "status", "trust_score", "trust_rating", "warnings", "converged"} <= set(r)


# ── The mesh study solves the requested angles ───────────────────────────────

def test_mesh_convergence_tool_solves_requested_angle():
    """alpha_sweep_deg=[3.0] solves 3 deg (plus the condition alpha), not the default sweep."""
    from ventorum.agent import mesh_convergence as mc_tool

    p = mc_tool({"span_m": 10.0, "chord_m": 1.25},
                {"V_inf_m_s": 40.0, "alpha_deg": 4.0},
                tolerance_pct=5.0, panel_counts=[10, 12], spacing_schemes=["half-cosine"],
                alpha_sweep_deg=[3.0], ref_n_panels=20)
    strict_json(p)
    assert p["status"] == "success", p
    assert sorted(p["alpha_tested_deg"]) == pytest.approx([3.0, 4.0])
    assert len(p["alpha_tested_deg"]) == 2


# ── A height step below round-off is refused ─────────────────────────────────

def test_ground_effect_refuses_tiny_height_step():
    """Two heights closer than 1e-6 * c_ref are refused as invalid input."""
    p = ground_effect(RECT, [1.0, 1.0000000000001], alpha_deg=4.0, settings=FAST)
    assert_error(p, "invalid_input")
    assert "1e-6" in p["error"]["message"] or "1e-06" in p["error"]["message"]


# ── Budget docs and the reason of an all-failed polar ────────────────────────

def test_polar_all_failed_copies_first_row_error():
    """When every angle fails, the top-level error copies the first row's error."""
    p = polar_sweep(dict(RECT), 1.0, 3.0, 1.0, {"h_m": 0.01}, dict(FAST), "standard")
    assert p["status"] == "error", p
    assert p["error"]["type"] == "ground_strike"
    table = p["polar_table"]
    assert len(table) == 3 and all(r["status"] != "ok" for r in table)
    assert table[0]["message"] in p["error"]["message"]
    strict_json(p)


def test_agent_guide_documents_work_budget():
    """The agent guide documents the work budget (MAX_CALL_WORK, N^2 units)."""
    from pathlib import Path

    from ventorum.agent.schemas import MAX_CALL_WORK

    text = (Path(__file__).resolve().parent.parent / "docs" / "agent" / "index.md").read_text()
    assert str(int(MAX_CALL_WORK)) in text
    assert "N^2" in text or "N**2" in text
    assert "budget" in text.lower()


# ── Audit log ────────────────────────────────────────────────────────────────

def test_audit_log_only_with_env_var(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VENTORUM_AGENT_AUDIT_LOG", raising=False)
    call_tool("ventorum_wing_analysis", {"wing": RECT, "settings": FAST, "detail_level": "summary"})
    assert list(tmp_path.iterdir()) == []

    log = tmp_path / "logs" / "audit.jsonl"
    log.parent.mkdir()
    monkeypatch.setenv("VENTORUM_AGENT_AUDIT_LOG", str(log))
    call_tool("ventorum_wing_analysis", {"wing": RECT, "settings": FAST, "detail_level": "summary"})
    call_tool("ventorum_nonexistent", {})
    lines = [json.loads(x) for x in log.read_text().splitlines()]
    assert len(lines) == 2
    assert lines[0]["status"] == "success" and lines[0]["tool"] == "ventorum_wing_analysis"
    assert lines[1]["status"] == "error" and lines[1]["error_type"] == "invalid_input"


# ── Regression tests of the review defects ───────────────────────────────────

SMALL = {"n_panels": 12, "n_chord": 2}
FIN = {"semi_span_m": 3.0, "chord_m": 1.0, "symmetric": False, "dihedral_deg": 90.0}


def test_zero_lift_gives_null_ld_and_e():
    """Defect 1 and 8: CL = 0 must not crash; L/D and e are null when they are not defined."""
    for wing, cond in ((FIN, None), (RECT, {"alpha_deg": 0.0}), (FIN, {"h_m": 1.0})):
        p = wing_analysis(wing, cond, SMALL, "summary")
        strict_json(p)
        assert p["status"] == "success", p
        assert p["metrics"]["L_over_D"] is None and p["metrics"]["e"] is None
        assert "L/D=n/a" in p["executive_summary"] and "e=n/a" in p["executive_summary"]
    p = polar_sweep(FIN, 0.0, 2.0, 2.0, None, SMALL, "standard")
    assert p["status"] == "success" and p["polar_table"][0]["L_over_D"] is None


def test_ground_effect_ld_is_null_at_zero_drag():
    """Zero drag gives L_over_D = null (NaN) in both library and agent."""
    p = ground_effect(RECT, [1.0], alpha_deg=0.0, settings=SMALL)
    strict_json(p)
    assert p["status"] == "success"
    assert p["rows"][0]["L_over_D"] is None and p["free_air"]["L_over_D"] is None
    assert "None" not in p["executive_summary"]


def test_mesh_limit_refuses_large_mesh():
    """Defect 2: a mesh larger than MAX_TOTAL_PANELS is refused before any solve."""
    from ventorum.agent.schemas import MAX_TOTAL_PANELS

    assert MAX_TOTAL_PANELS == 4000
    assert str(MAX_TOTAL_PANELS) in SETTINGS_PROPS["n_panels"]["description"]
    assert str(MAX_TOTAL_PANELS) in SETTINGS_PROPS["n_chord"]["description"]
    assert_error(wing_analysis(RECT, None, {"n_panels": 400, "n_chord": 64}), "invalid_input", "4000")
    assert_error(wing_analysis({**RECT, "n_panels": 400}, None, {"n_chord": 8}), "invalid_input", "limit")
    # Automatic n_chord: even the smallest automatic value (4) is too large.
    big = {"surfaces": [{**RECT, "n_panels": 400}, {"name": "t", "span_m": 4.0, "chord_m": 0.5,
                                                     "position_m": [4.0, 0.0, 0.0], "n_panels": 400}]}
    assert_error(wing_analysis(big, None, None), "invalid_input", "4000")
    assert_error(ground_effect(RECT, [1.0], settings={"n_panels": 400, "n_chord": 64}), "invalid_input", "4000")
    assert_error(polar_sweep(RECT, 0.0, 2.0, 1.0, None, {"n_panels": 400, "n_chord": 64}), "invalid_input",
                 "4000")
    assert_error(stability_derivatives(RECT, None, settings={"n_panels": 400, "n_chord": 64}),
                 "invalid_input", "4000")
    p = batch_evaluate([RECT, {**RECT, "n_panels": 400}], None, "max_CL", {"n_chord": 8, "n_panels": 8})
    assert p["status"] == "success" and p["n_failed"] == 1
    assert p["failed"][0]["error"]["type"] == "invalid_input"
    p = call_tool("ventorum_mesh_convergence", {"wing": RECT, "flight_condition": {"h_m": 0.1},
                                                 "panel_counts": [8, 12], "ref_n_panels": 400})
    assert_error(p, "invalid_input", "ref_n_panels")


def test_mesh_limit_reduces_automatic_n_chord(monkeypatch):
    """Defect 2: an automatic n_chord above the limit is reduced (here with a small test limit)."""
    from ventorum.agent import tools

    monkeypatch.setattr(tools, "MAX_TOTAL_PANELS", 100)  # 24 strips -> at most 4 chordwise panels
    p = wing_analysis(RECT, {"alpha_deg": 2.0, "h_m": 0.12}, {"n_panels": 12}, "summary")
    assert p["status"] == "success", p
    assert p["settings_used"]["n_chord_used"] == 4
    assert "reduced" in p["settings_used"]["mesh_note"]
    g = ground_effect(RECT, [0.12, 0.5], alpha_deg=2.0, settings={"n_panels": 12})
    assert g["status"] == "success", g
    assert g["settings_used"]["n_chord"] == 4 and "mesh_note" in g["settings_used"]
    # Free air needs only 4 chordwise panels: no reduction.
    p = wing_analysis(RECT, None, {"n_panels": 12}, "summary")
    assert p["settings_used"]["n_chord_used"] == 4 and "mesh_note" not in p["settings_used"]


def test_fourier_in_ground_effect_is_invalid_method_in_every_tool():
    """Defect 3: Fourier solver in ground effect gives invalid_method in every tool."""
    F = {"solver": "fourier", "n_panels": 8}
    gc = {"alpha_deg": 4.0, "h_m": 1.0}
    assert_error(wing_analysis(RECT, gc, F), "invalid_method", "Fourier")
    assert_error(polar_sweep(RECT, 0.0, 2.0, 1.0, {"h_m": 1.0}, F), "invalid_method", "Fourier")
    assert_error(ground_effect(RECT, [0.5, 1.0], settings=F), "invalid_method", "Fourier")
    assert_error(stability_derivatives(RECT, gc, settings=F), "invalid_method", "Fourier")
    assert_error(batch_evaluate([RECT], gc, "max_CL", F), "invalid_method")
    assert_error(call_tool("ventorum_mesh_convergence", {"wing": RECT, "flight_condition": gc,
                                                          "panel_counts": [8, 12], "ref_n_panels": 20,
                                                          "solver": "fourier"}), "invalid_method", "Fourier")


def test_body_wake_in_ground_effect_is_invalid_input():
    """Defect 3: settings.wake_alignment 'body' is refused in ground effect, as the schema says."""
    B = {"wake_alignment": "body", "n_panels": 8}
    gc = {"alpha_deg": 4.0, "h_m": 1.0}
    assert_error(wing_analysis(RECT, gc, B), "invalid_input", "body")
    assert_error(polar_sweep(RECT, 0.0, 2.0, 1.0, {"h_m": 1.0}, B), "invalid_input", "body")
    assert_error(ground_effect(RECT, [1.0], settings=B), "invalid_input", "body")
    assert_error(stability_derivatives(RECT, gc, settings=B), "invalid_input", "body")
    assert_error(batch_evaluate([RECT], gc, "max_CL", B), "invalid_input", "body")
    # In free air the body-axis wake is allowed.
    assert wing_analysis(RECT, None, B, "summary")["status"] == "success"


def test_linalg_error_is_invalid_input():
    """A singular system is invalid input, not an internal failure.

    A LinAlgError used to be reported as "internal". The owner decided to
    map it to "invalid_input", with a message that names the usual cause.
    """
    import numpy as np

    from ventorum.agent.response import error_info, error_type_of
    from ventorum.agent.schemas import InputError

    assert error_type_of(np.linalg.LinAlgError("Singular matrix")) == "invalid_input"
    assert "singular" in error_info(np.linalg.LinAlgError("Singular matrix"))["message"]
    assert error_type_of(InputError("x")) == "invalid_input"
    assert error_type_of(ValueError("x")) == "invalid_input"


def test_alpha_step_has_a_minimum():
    """Defect 3: a tiny alpha step (5e-324) gave OverflowError -> internal."""
    prop = next(t for t in AGENT_TOOL_DEFINITIONS
                if t["name"] == "ventorum_polar_sweep")["parameters"]["properties"]["alpha_step_deg"]
    assert prop["minimum"] == 0.01 and "exclusiveMinimum" not in prop
    for step in (5e-324, 1e-6, 0.005):
        assert_error(polar_sweep(RECT, 0.0, 5.0, step, None, FAST), "invalid_input", "alpha_step_deg")
    p = call_tool("ventorum_polar_sweep", {"wing": RECT, "alpha_start_deg": 0.0, "alpha_end_deg": 0.02,
                                            "alpha_step_deg": 0.01, "settings": FAST,
                                            "detail_level": "summary"})
    assert p["status"] == "success" and p["polar_summary"]["n_points"] == 3


def test_ground_effect_irodov_headline_uses_central_difference():
    """Defect 6: the headline margin is at the lowest height with a central difference, and the
    reused centre-alpha cases give the same margins as a full 3-alpha sweep."""
    import numpy as np

    from ventorum.agent.schemas import build_aircraft_from_spec
    from ventorum.core.datatypes import SolverSettings
    from ventorum.ground_effect import GroundEffectSweep

    hs = [0.3, 0.6, 1.0, 1.5]
    p = ground_effect(RECT, hs, alpha_deg=4.0, settings=SMALL, detail_level="summary")
    strict_json(p)
    assert p["status"] == "success"
    ir = p["irodov"]
    assert [r["height_difference"] for r in ir["rows"]] == ["one-sided", "central", "central", "one-sided"]
    assert ir["headline_h_m"] == 0.6
    assert "Irodov margin at h=0.6 m (central height difference)" in p["executive_summary"]

    ref = GroundEffectSweep(build_aircraft_from_spec(RECT), settings=SolverSettings(n_panels=12, n_chord=2))
    der = ref.run_sweep(hs, [3.0, 4.0, 5.0], [0.0], compute_strike_limit=False).compute_stability_derivatives()
    got = [r["irodov_margin"] for r in ir["rows"]]
    assert got == pytest.approx(np.round(der["irodov_margin"][:, 1], 5).tolist(), abs=2e-5)

    two = ground_effect(RECT, [0.5, 1.0], alpha_deg=4.0, settings=SMALL)
    assert two["status"] == "success" and len(two["irodov"]["rows"]) == 2
    assert two["irodov"]["headline_h_m"] == 0.5
    assert any("one-sided" in n for n in two["irodov"]["notes"])
    assert "one-sided height difference" in two["executive_summary"]


def test_ground_effect_bank_strike_limit_not_found_is_null():
    """Defect 6: no contact up to the 60 deg search limit gives null, not 60."""
    p = ground_effect(RECT, [1.0, 6.0], alpha_deg=4.0, settings=SMALL)
    strict_json(p)
    rows = {r["h_m"]: r for r in p["rows"]}
    assert rows[1.0]["bank_strike_limit_found"] is True and 0 < rows[1.0]["phi_strike_limit_deg"] < 60
    assert rows[6.0]["bank_strike_limit_found"] is False and rows[6.0]["phi_strike_limit_deg"] is None
    assert p["summary"]["bank_strike_limit_found_at_lowest"] is True
    assert "60" in p["summary"]["bank_strike_note"]


def test_gemini_schema_uses_the_openapi_subset():
    """Defect 7: Gemini declarations use only the documented OpenAPI 3.0 subset."""
    from ventorum.agent.schemas import GEMINI_SCHEMA_KEYS

    def walk(node):
        assert set(node) <= set(GEMINI_SCHEMA_KEYS), set(node) - set(GEMINI_SCHEMA_KEYS)
        for sub in node.get("properties", {}).values():
            walk(sub)
        if "items" in node:
            walk(node["items"])

    for decl in get_tool_schemas("gemini"):
        walk(decl["parameters"])
    wa = get_tool_schemas("gemini")[0]["parameters"]["properties"]["wing"]
    assert wa["type"] == "object" and {"span_m", "surfaces", "S_ref_m2"} <= set(wa["properties"])
    span = wa["properties"]["span_m"]
    assert span["minimum"] == 0 and "larger than 0" in span["description"]
    nw = get_tool_schemas("gemini")[4]["parameters"]["properties"]["n_jobs"]
    assert nw["type"] == "integer" and "'auto'" in nw["description"]
    # The other formats keep the full JSON schema.
    assert "anyOf" in json.dumps(get_tool_schemas("mcp"))


def test_stability_derivatives_ge_one_mesh():
    """In ground effect all five cases use one chordwise mesh (review 5, A1).

    At h = 0.255 m the automatic n_chord is 6 at alpha 4.5 deg and 5 at
    3.5 deg. The derivatives must equal central differences on one mesh
    (the larger count), and n_chord_used must name that mesh.
    """
    import numpy as np

    import ventorum as vt

    out = stability_derivatives({"span_m": 6.0, "chord_m": 1.0},
                                {"alpha_deg": 4.0, "V_inf_m_s": 30.0, "h_m": 0.255},
                                x_cg_m=0.25, settings={"n_panels": 20, "solver": "vlm"})
    assert out["status"] == "success"
    n_used = out["settings_used"]["n_chord_used"]

    wing = vt.LiftingSurface(semi_span=3.0, sections=[vt.WingSection(y_frac=0.0, chord=1.0),
                                                      vt.WingSection(y_frac=1.0, chord=1.0)])
    ac = vt.Aircraft(surfaces=[wing], ref_point=np.array([0.25, 0.0, 0.0]))

    def run(a, nc):
        return vt.analyze(ac.clone(), vt.FlightCondition(V_inf=30.0, alpha=np.radians(a), h=0.255),
                          vt.SolverSettings(solver_type="vlm", n_panels=20, n_chord=nc))

    counts = {run(a, None).details["lattice"].n_chord for a in (3.5, 4.5)}
    assert len(counts) == 2  # the case crosses a change of the automatic count
    assert n_used == max(counts)
    rp, rm = run(4.5, n_used), run(3.5, n_used)
    cla = (rp.totals.CL - rm.totals.CL) / np.radians(1.0)
    cma = (rp.totals.Cm - rm.totals.Cm) / np.radians(1.0)
    d = out["stability_derivatives"]
    assert d["CL_alpha_per_rad"] == pytest.approx(cla, abs=2e-5)
    assert d["Cm_alpha_per_rad"] == pytest.approx(cma, abs=2e-5)


# ── Device and precision ─────────────────────────────────────────────────────

def test_payload_states_device_precision():
    """Every result states its device (cpu or gpu) and precision (float32 or float64)."""
    from ventorum.agent import mesh_convergence

    p = wing_analysis(RECT, {"alpha_deg": 4.0}, FAST, "summary")
    strict_json(p)
    assert p["status"] == "success"
    assert p["settings_used"]["device"] in ("cpu", "gpu")
    assert p["settings_used"]["precision"] in ("float32", "float64")
    assert p["settings_used"]["device"] in p["executive_summary"]
    assert p["settings_used"]["precision"] in p["executive_summary"]

    pol = polar_sweep(RECT, 0.0, 4.0, 2.0, None, FAST, "standard")
    strict_json(pol)
    assert pol["status"] == "success"
    assert pol["settings_used"]["device"] in ("cpu", "gpu")
    assert pol["settings_used"]["precision"] in ("float32", "float64")
    assert pol["settings_used"]["device"] in pol["executive_summary"]
    assert pol["settings_used"]["precision"] in pol["executive_summary"]
    ok_rows = [r for r in pol["polar_table"] if r["status"] == "ok"]
    assert ok_rows
    for row in ok_rows:
        assert row["device"] in ("cpu", "gpu"), row
        assert row["precision"] in ("float32", "float64"), row

    g = ground_effect(RECT, [0.6, 1.0, 1.5], alpha_deg=4.0, settings=FAST)
    strict_json(g)
    assert g["status"] == "success"
    assert g["settings_used"]["device"] in ("cpu", "gpu")
    assert g["settings_used"]["precision"] in ("float32", "float64")
    assert g["settings_used"]["device"] in g["executive_summary"]
    assert g["settings_used"]["precision"] in g["executive_summary"]
    ok_g = [r for r in g["rows"] if r["status"] == "ok"]
    assert ok_g
    for row in ok_g:
        assert row["device"] in ("cpu", "gpu"), row
        assert row["precision"] in ("float32", "float64"), row

    s = stability_derivatives(RECT, {"alpha_deg": 4.0}, x_cg_m=0.1, settings=FAST)
    strict_json(s)
    assert s["status"] == "success"
    assert s["settings_used"]["device"] in ("cpu", "gpu")
    assert s["settings_used"]["precision"] in ("float32", "float64")
    assert s["settings_used"]["device"] in s["executive_summary"]
    assert s["settings_used"]["precision"] in s["executive_summary"]

    b = batch_evaluate([RECT, {"span_m": 6.0, "chord_m": 1.0}], None, "max_CL", FAST)
    strict_json(b)
    assert b["status"] == "success"
    assert b["settings_used"]["device"] in ("cpu", "gpu")
    assert b["settings_used"]["precision"] in ("float32", "float64")
    assert b["settings_used"]["device"] in b["executive_summary"]
    assert b["settings_used"]["precision"] in b["executive_summary"]
    assert b["rankings"]
    for row in b["rankings"]:
        assert row["device"] in ("cpu", "gpu"), row
        assert row["precision"] in ("float32", "float64"), row

    m = mesh_convergence(
        {"span_m": 10.0, "root_chord_m": 1.5, "tip_chord_m": 1.0},
        {"V_inf_m_s": 45.0, "alpha_deg": 4.0},
        panel_counts=[12, 20],
        spacing_schemes=["half-cosine"],
        ref_n_panels=30,
    )
    strict_json(m)
    assert m["status"] == "success"
    assert m["device"] in ("cpu", "gpu", "mixed")
    assert m["precision"] in ("float32", "float64", "mixed")
    assert m["device"] in m["executive_summary"]
    assert m["precision"] in m["executive_summary"]


def test_machine_capabilities_has_no_fingerprint():
    """The public capabilities send no machine identifier and no fingerprint hash."""
    from ventorum.agent import call_tool as _call

    p = _call("ventorum_machine_capabilities", {})
    strict_json(p)
    assert p["status"] == "success"
    blob = json.dumps(p)
    assert "machine_guid" not in blob
    assert "node_name" not in blob
    assert "fingerprint" not in blob
    assert "fingerprint" not in p["hardware"]


def test_machine_capabilities_reports_gpu_pipeline():
    """The capabilities report the GPU pipeline device and precision."""
    from ventorum import gpu
    from ventorum.agent import call_tool as _call

    info = gpu.info()
    p = _call("ventorum_machine_capabilities", {})
    strict_json(p)
    assert p["status"] == "success"
    assert "gpu_pipeline" in p
    assert p["gpu_pipeline"]["device"] == info["device"]
    assert p["gpu_pipeline"]["precision"] == info["precision"]
    assert p["gpu_pipeline"]["available"] == info["available"]


# ── Explicit null is refused for every key ──────────────────────────────────

def test_null_values_are_refused_with_key_name():
    """{"alpha_deg": null} (also V, twist, a0, cd0, S_ref, settings keys) is invalid_input."""
    cases = [
        (RECT, {"alpha_deg": None}, FAST, "flight_condition", "alpha_deg"),
        (RECT, {"V_inf_m_s": None}, FAST, "flight_condition", "V_inf_m_s"),
        ({**RECT, "tip_twist_deg": None}, None, FAST, "wing", "tip_twist_deg"),
        ({"span_m": 10.0, "chord_m": 1.0, "airfoil": {"a0_per_rad": None}}, None, FAST, "wing.airfoil",
         "a0_per_rad"),
        ({"span_m": 10.0, "chord_m": 1.0, "airfoil": {"cd0": None}}, None, FAST, "wing.airfoil", "cd0"),
        ({"surfaces": [RECT], "S_ref_m2": None}, None, FAST, "wing", "S_ref_m2"),
        (RECT, None, {"n_panels": None}, "settings", "n_panels"),
        (RECT, None, {"n_panels": 8, "n_chord": None}, "settings", "n_chord"),
    ]
    for wing, cond, sett, where, key in cases:
        p = wing_analysis(wing, cond, sett, "summary")
        assert_error(p, "invalid_input", f"{where}: '{key}' is null")
        assert "got None" not in p["error"]["message"]
    # Null through the dispatcher (JSON null) names the key too.
    assert_error(call_tool("ventorum_wing_analysis",
                           {"wing": RECT, "flight_condition": {"alpha_deg": None}, "settings": FAST}),
                 "invalid_input", "'alpha_deg' is null")
    assert_error(call_tool("ventorum_wing_analysis",
                           json.dumps({"wing": {**RECT, "dihedral_deg": None}, "settings": FAST})),
                 "invalid_input", "'dihedral_deg' is null")


_REQUIRED_ARGS = {
    "wing": RECT,
    "alpha_start_deg": 0.0,
    "alpha_end_deg": 4.0,
    "alpha_step_deg": 2.0,
    "heights_m": [1.0],
    "candidates": [RECT],
}


def test_null_top_level_arguments_are_refused(monkeypatch):
    """JSON null for an optional tool argument is invalid_input; the tool is not called.

    Before the fix, a null top-level argument (for example
    ``"settings": null``) meant "omitted" and the tool ran.
    """
    from ventorum.agent import dispatcher

    calls = []

    def stub(**kwargs):
        calls.append(kwargs)
        return {"status": "success"}

    n_checked = 0
    for t in AGENT_TOOL_DEFINITIONS:
        name = t["name"]
        schema = t["parameters"]
        required = schema.get("required", [])
        monkeypatch.setitem(dispatcher.TOOL_FUNCTIONS, name, stub)
        base = {k: _REQUIRED_ARGS[k] for k in required}
        for key in sorted(set(schema["properties"]) - set(required)):
            p = call_tool(name, {**base, key: None})
            assert_error(p, "invalid_input", f"'{key}' is null")
            assert "Omit the key" in p["error"]["message"]
            # The same through a JSON string (a real JSON null).
            p = call_tool(name, json.dumps({**base, key: None}))
            assert_error(p, "invalid_input", f"'{key}' is null")
            n_checked += 1
    assert n_checked >= 20
    assert calls == []


def test_python_none_still_means_omitted():
    """In a Python call, None for an optional argument keeps its meaning "use the default"."""
    p = wing_analysis(RECT, flight_condition=None, settings=FAST, axes=None)
    assert p["status"] == "success", p
    assert p["condition_used"]["alpha_deg"] == 5.0
    g = ground_effect(RECT, [1.0], ref_point_m=None, settings=None)
    assert g["status"] == "success", g


# ── The default angle of attack is 5 deg ────────────────────────────────────

def test_default_alpha_is_five_deg_in_every_interface():
    """An omitted alpha is 5 deg in the agent schemas, vt.analyze, FlightCondition and Ventorum."""
    import inspect

    import numpy as np

    import ventorum as vt
    from ventorum.agent.schemas import CONDITION_PROPS

    assert "Default 5." in CONDITION_PROPS["alpha_deg"]["description"]
    ge_def = next(t for t in AGENT_TOOL_DEFINITIONS if t["name"] == "ventorum_ground_effect")
    assert "Default 5." in ge_def["parameters"]["properties"]["alpha_deg"]["description"]
    assert inspect.signature(ground_effect).parameters["alpha_deg"].default == 5.0
    assert "default 5.0" in ground_effect.__doc__
    p = wing_analysis(RECT, None, FAST, "summary")
    assert p["status"] == "success", p
    assert p["condition_used"]["alpha_deg"] == 5.0
    g = ground_effect(RECT, [1.0], settings=FAST)
    assert g["status"] == "success", g
    assert g["condition_used"]["alpha_deg"] == 5.0
    assert vt.FlightCondition().alpha == pytest.approx(np.radians(5.0))
    wing = vt.LiftingSurface(semi_span=5.0, sections=[vt.WingSection(y_frac=0.0, chord=1.5),
                                                     vt.WingSection(y_frac=1.0, chord=1.0)])
    assert vt.Ventorum(name="t5", geometry=wing).condition.alpha == pytest.approx(np.radians(5.0))
    # vt.analyze without a condition and without alpha_deg solves at 5 deg.
    res = vt.analyze(wing, n_panels=8)
    assert res.condition.alpha == pytest.approx(np.radians(5.0))
    ref = vt.analyze(wing, n_panels=8, alpha_deg=5.0)
    assert res.totals.CL == pytest.approx(ref.totals.CL, rel=1e-12)


# ── The main surface has the largest projected area ─────────────────────────

def test_main_surface_is_largest_projected_area():
    """The reference values come from the surface with the largest projected planform area."""
    from ventorum.agent.schemas import AIRCRAFT_PROPS

    assert "largest projected planform area" in AIRCRAFT_PROPS["surfaces"]["description"]
    ac = {"surfaces": [{"name": "tail", "span_m": 3.0, "chord_m": 0.5},
                       {"name": "wing", "span_m": 10.0, "chord_m": 1.0}]}
    p = wing_analysis(ac, None, FAST, "standard")
    assert p["status"] == "success", p
    assert p["geometry"]["S_ref_m2"] == pytest.approx(10.0)
    assert p["geometry"]["b_ref_m"] == pytest.approx(10.0)


# ── Tiny geometry is invalid input, not internal ───────────────────────────

def test_tiny_geometry_is_invalid_input_not_internal():
    """Span 1e-300 is invalid_input (it gave internal ZeroDivisionError)."""
    assert_error(wing_analysis({"span_m": 1e-300, "chord_m": 1.0}, None, FAST, "summary"),
                 "invalid_input", "semi_span")
    assert_error(wing_analysis({"span_m": 10.0, "chord_m": 1e-300}, None, FAST, "summary"),
                 "invalid_input", "chord")
    assert_error(wing_analysis({"surfaces": [RECT], "S_ref_m2": 0.0}, None, FAST, "summary"),
                 "invalid_input", "S_ref_m2")


# ── 1-D numpy arrays are accepted where lists are ──────────────────────────

def test_numpy_arrays_are_accepted_as_lists():
    """A 1-D numpy array is accepted for vector and list inputs."""
    import numpy as np

    p = wing_analysis({**RECT, "position_m": np.array([0.0, 0.0, 0.0])}, None, FAST, "summary")
    assert p["status"] == "success", p
    ref = wing_analysis(dict(RECT), None, FAST, "summary")
    assert p["metrics"]["CL"] == pytest.approx(ref["metrics"]["CL"])
    g = ground_effect(RECT, np.array([0.6, 1.0]), settings=FAST)
    assert g["status"] == "success", g
    assert [r["h_m"] for r in g["rows"]] == pytest.approx([0.6, 1.0])


def test_tiny_reference_values_are_invalid_input():
    """Reference values of 1e-300 are invalid_input that names the key.

    Before the fix, b_ref_m = 1e-300 gave an internal ZeroDivisionError
    and S_ref_m2 = 1e-300 gave an internal OverflowError in the loads.
    """
    from ventorum.agent.schemas import build_aircraft_from_spec

    for key in ("S_ref_m2", "b_ref_m", "c_ref_m"):
        p = call_tool("ventorum_wing_analysis",
                      {"wing": {"surfaces": [RECT], key: 1e-300}, "settings": FAST})
        assert_error(p, "invalid_input", key)
    # A small but sane value passes.
    p = call_tool("ventorum_wing_analysis",
                  {"wing": {"surfaces": [RECT], "S_ref_m2": 1e-3, "b_ref_m": 1e-3, "c_ref_m": 1e-3},
                   "settings": FAST, "detail_level": "summary"})
    assert p["status"] == "success", p
    # The Python call path with an Aircraft object is checked too.
    for attr in ("S_ref", "b_ref", "c_ref"):
        ac = build_aircraft_from_spec({"surfaces": [RECT]})
        setattr(ac, attr, 1e-300)
        assert_error(wing_analysis(ac, None, FAST, "summary"), "invalid_input", attr)


def test_speed_below_minimum_is_invalid_input():
    """V_inf below 0.1 m/s is invalid_input in every tool; 0.1 m/s is accepted.

    Before the fix, V_inf_m_s = 1e-300 gave an internal ZeroDivisionError
    and 0.05 m/s was accepted.
    """
    for v in (1e-300, 0.05):
        assert_error(call_tool("ventorum_wing_analysis",
                               {"wing": RECT, "flight_condition": {"V_inf_m_s": v}, "settings": FAST}),
                     "invalid_input", "V_inf_m_s")
        assert_error(call_tool("ventorum_ground_effect",
                               {"wing": RECT, "heights_m": [1.0], "V_inf_m_s": v, "settings": FAST}),
                     "invalid_input", "V_inf_m_s")
    p = call_tool("ventorum_wing_analysis",
                  {"wing": RECT, "flight_condition": {"V_inf_m_s": 0.1}, "settings": FAST,
                   "detail_level": "summary"})
    assert p["status"] == "success", p
    g = call_tool("ventorum_ground_effect",
                  {"wing": RECT, "heights_m": [1.0], "V_inf_m_s": 0.1, "settings": FAST,
                   "detail_level": "summary"})
    assert g["status"] == "success", g
    assert CONDITION_PROPS["V_inf_m_s"]["minimum"] == 0.1
