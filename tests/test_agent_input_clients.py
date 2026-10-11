# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Tests for agent tool input schemas across LLM clients (T-0052).
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

from ventorum.agent import (
    AGENT_TOOL_DEFINITIONS,
    call_tool,
    get_tool_schemas,
    stability_derivatives,
)
from ventorum.agent.schemas import build_aircraft_from_spec

FAST = {"n_panels": 8}
RECT = {"span_m": 10.0, "chord_m": 1.0}


def _check_no_object_anyof(node: Any, path: str = "") -> None:
    if isinstance(node, dict):
        for key in ("anyOf", "oneOf"):
            if key in node:
                for idx, opt in enumerate(node[key]):
                    if isinstance(opt, dict):
                        assert opt.get("type") != "object", (
                            f"Found {key} with object option at {path}.{key}[{idx}]: {opt}"
                        )
                        assert "properties" not in opt, (
                            f"Found {key} with properties at {path}.{key}[{idx}]: {opt}"
                        )
        for k, v in node.items():
            _check_no_object_anyof(v, f"{path}.{k}" if path else k)
    elif isinstance(node, list):
        for idx, item in enumerate(node):
            _check_no_object_anyof(item, f"{path}[{idx}]")


def test_no_object_anyof_in_exported_schemas():
    for fmt in ("openai", "anthropic", "gemini", "mcp"):
        schemas = get_tool_schemas(fmt)
        for s in schemas:
            _check_no_object_anyof(s, f"{fmt}:{s.get('name', 'tool')}")


def test_aircraft_argument_equals_python():
    tail = {
        "name": "tail",
        "span_m": 3.0,
        "chord_m": 0.5,
        "position_m": [5.0, 0.0, 0.3],
        "incidence_deg": -2.0,
    }
    spec = {
        "surfaces": [RECT, tail],
        "ref_point_m": [0.45, 0.0, 0.0],
    }
    ac_obj = build_aircraft_from_spec(aircraft=spec)

    res_tool = stability_derivatives(aircraft=spec, settings=FAST)
    assert res_tool["status"] == "success", res_tool

    res_py = stability_derivatives(aircraft=ac_obj, settings=FAST)
    assert res_py["status"] == "success", res_py

    m_tool = res_tool["stability_derivatives"]
    m_py = res_py["stability_derivatives"]
    assert m_tool["CL_alpha_per_rad"] == m_py["CL_alpha_per_rad"]
    assert m_tool["Cm_alpha_per_rad"] == m_py["Cm_alpha_per_rad"]
    assert m_tool["static_margin_fraction"] == m_py["static_margin_fraction"]


def test_exactly_one_geometry_argument():
    documented_msg = "Give exactly one of 'wing' (one surface) or 'aircraft' (several surfaces, with 'surfaces')."
    tail = {"name": "tail", "span_m": 3.0, "chord_m": 0.5, "position_m": [4.0, 0.0, 0.0]}
    ac_valid = {"surfaces": [RECT, tail]}

    for tool_def in AGENT_TOOL_DEFINITIONS:
        name = tool_def["name"]
        props = tool_def["parameters"]["properties"]
        if "wing" not in props or "aircraft" not in props:
            continue

        base_args: dict[str, Any] = {}
        if "settings" in props:
            base_args["settings"] = FAST
        if name == "ventorum_polar_sweep":
            base_args.update({"alpha_start_deg": 0.0, "alpha_end_deg": 2.0, "alpha_step_deg": 1.0})
        elif name == "ventorum_ground_effect":
            base_args.update({"heights_m": [1.0]})
        elif name == "ventorum_trim":
            base_args.update({
                "CL_target": 0.5,
                "pitch_control": "elev",
            })

        # 1. Neither wing nor aircraft
        args_neither = dict(base_args)
        args_neither.pop("wing", None)
        args_neither.pop("aircraft", None)
        res_neither = call_tool(name, args_neither)
        assert res_neither["status"] == "error", f"{name} succeeded with neither"
        assert res_neither["error"]["type"] == "invalid_input"
        assert documented_msg in res_neither["error"]["message"]

        # 2. Both wing and aircraft
        args_both = dict(base_args)
        if name == "ventorum_trim":
            args_both["wing"] = {
                "span_m": 10.0,
                "chord_m": 1.0,
                "controls": [{"name": "elev", "hinge_x_c": 0.75}],
            }
            args_both["aircraft"] = {
                "surfaces": [{
                    "name": "w",
                    "span_m": 10.0,
                    "chord_m": 1.0,
                    "controls": [{"name": "elev", "hinge_x_c": 0.75}],
                }],
            }
        else:
            args_both["wing"] = RECT
            args_both["aircraft"] = ac_valid
        res_both = call_tool(name, args_both)
        assert res_both["status"] == "error", f"{name} succeeded with both"
        assert res_both["error"]["type"] == "invalid_input"
        assert documented_msg in res_both["error"]["message"]

        # 3. wing with surfaces
        args_wing_surfaces = dict(base_args)
        args_wing_surfaces.pop("aircraft", None)
        args_wing_surfaces["wing"] = {"surfaces": [RECT]}
        res_ws = call_tool(name, args_wing_surfaces)
        assert res_ws["status"] == "error", f"{name} succeeded with wing.surfaces"
        assert res_ws["error"]["type"] == "invalid_input"
        assert "unknown key 'surfaces'" in res_ws["error"]["message"]
        assert "belongs in 'aircraft'" in res_ws["error"]["message"]


def test_dispatcher_keeps_aircraft_keys():
    captured: list[dict] = []
    orig = build_aircraft_from_spec

    def spy_build(*args, **kwargs):
        if "aircraft" in kwargs and isinstance(kwargs["aircraft"], dict):
            captured.append(dict(kwargs["aircraft"]))
        elif len(args) > 1 and isinstance(args[1], dict):
            captured.append(dict(args[1]))
        return orig(*args, **kwargs)

    ac_input = {
        "name": "TestPlane",
        "surfaces": [RECT],
        "ref_point_m": [0.45, 0.0, 0.0],
        "S_ref_m2": 10.0,
        "b_ref_m": 10.0,
        "c_ref_m": 1.0,
    }
    call_args = json.dumps({"aircraft": ac_input, "settings": FAST})
    with patch("ventorum.agent.tools.build_aircraft_from_spec", side_effect=spy_build):
        res = call_tool("ventorum_wing_analysis", call_args)

    assert res["status"] == "success", res
    assert len(captured) == 1
    passed = captured[0]
    for key in ("name", "surfaces", "ref_point_m", "S_ref_m2", "b_ref_m", "c_ref_m"):
        assert key in passed, f"Key {key} dropped in dispatcher MCP path"


def test_placement_hints():
    # 1. solver at top level of ventorum_wing_analysis names settings
    r1 = call_tool("ventorum_wing_analysis", {"wing": RECT, "solver": "vlm"})
    assert r1["status"] == "error"
    assert "unknown key 'solver'" in r1["error"]["message"]
    assert "'solver' belongs in 'settings'" in r1["error"]["message"]

    # 2. ref_point_m in wing names aircraft
    r2 = call_tool("ventorum_wing_analysis", {"wing": {**RECT, "ref_point_m": [0.45, 0.0, 0.0]}})
    assert r2["status"] == "error"
    assert "unknown key 'ref_point_m'" in r2["error"]["message"]
    assert "'ref_point_m' belongs in 'aircraft'" in r2["error"]["message"]

    # 3. alpha_deg in settings names flight_condition
    r3 = call_tool("ventorum_wing_analysis", {"wing": RECT, "settings": {"alpha_deg": 5.0}})
    assert r3["status"] == "error"
    assert "unknown key 'alpha_deg'" in r3["error"]["message"]
    assert "'alpha_deg' belongs in 'flight_condition'" in r3["error"]["message"]

    # 4. Context of tool call: alpha_deg in settings for ground_effect names top level
    r4 = call_tool("ventorum_ground_effect", {"wing": RECT, "heights_m": [1.0], "settings": {"alpha_deg": 5.0}})
    assert r4["status"] == "error"
    assert "unknown key 'alpha_deg'" in r4["error"]["message"]
    assert "top level" in r4["error"]["message"]


def test_invented_key_hints():
    expected_hints = {
        "x_start": "use position_m ([x, y, z] of the root leading edge, metres)",
        "y_start": "use position_m ([x, y, z] of the root leading edge, metres)",
        "z_start": "use position_m ([x, y, z] of the root leading edge, metres)",
        "position": "use position_m ([x, y, z] of the root leading edge, metres)",
        "x_le": "use position_m ([x, y, z] of the root leading edge, metres)",
        "offset": "use position_m ([x, y, z] of the root leading edge, metres)",
        "location": "use position_m ([x, y, z] of the root leading edge, metres)",
        "root_le": "use position_m",
        "solver_type": "use solver in settings",
        "method": "use solver in settings",
        "cg": "use x_cg_m (stability tool) or aircraft.ref_point_m",
        "cg_m": "use x_cg_m (stability tool) or aircraft.ref_point_m",
        "mach": "Mach number is not an input (incompressible flow); give V_inf_m_s",
        "wings": "use the 'aircraft' argument with 'surfaces'",
        "components": "use the 'aircraft' argument with 'surfaces'",
        "parts": "use the 'aircraft' argument with 'surfaces'",
        "tail": "add it as a surface in aircraft.surfaces",
        "stabilizer": "add it as a surface in aircraft.surfaces",
        "htail": "add it as a surface in aircraft.surfaces",
        "vtail": "add it as a surface in aircraft.surfaces",
        "fin": "add it as a surface in aircraft.surfaces",
    }
    for key, expected in expected_hints.items():
        res = call_tool("ventorum_wing_analysis", {"wing": {**RECT, key: 1.0}})
        assert res["status"] == "error", f"Key {key} unexpectedly succeeded"
        msg = res["error"]["message"]
        assert f"unknown key '{key}'" in msg, f"Key {key} not reported as unknown"
        assert expected in msg, f"Key {key} hint mismatch: expected {expected!r} in {msg!r}"


def test_batch_candidates_forms():
    tail = {"name": "tail", "span_m": 3.0, "chord_m": 0.5, "position_m": [4.0, 0.0, 0.0]}
    ac_valid = {"name": "plane", "surfaces": [RECT, tail]}

    # Candidates with wing and aircraft both work
    res_ok = call_tool("ventorum_batch_evaluate", {
        "candidates": [
            {"wing": {**RECT, "name": "cand_wing"}},
            {"aircraft": ac_valid},
        ],
        "settings": FAST,
    })
    assert res_ok["status"] == "success", res_ok
    assert res_ok["n_candidates"] == 2
    assert res_ok["n_succeeded"] == 2
    assert res_ok["n_failed"] == 0

    # Candidate with both or neither refused with index in message
    documented_msg = "Give exactly one of 'wing' (one surface) or 'aircraft' (several surfaces, with 'surfaces')."
    res_bad = call_tool("ventorum_batch_evaluate", {
        "candidates": [
            {"wing": RECT},
            {"wing": RECT, "aircraft": ac_valid},
            {},
        ],
        "settings": FAST,
    })
    assert res_bad["status"] == "success", res_bad
    assert res_bad["n_candidates"] == 3
    assert res_bad["n_succeeded"] == 1
    assert res_bad["n_failed"] == 2

    failed = res_bad["failed"]
    assert failed[0]["index"] == 1
    assert "candidates[1]" in failed[0]["error"]["message"]
    assert documented_msg in failed[0]["error"]["message"]

    assert failed[1]["index"] == 2
    assert "candidates[2]" in failed[1]["error"]["message"]
    assert documented_msg in failed[1]["error"]["message"]


def test_batch_candidate_without_geometry_key_is_refused():
    # Review T-0052: a bare surface object as a candidate was silently taken as 'wing'
    # (decision 3: exactly one of the keys 'wing' or 'aircraft').
    res = call_tool("ventorum_batch_evaluate", {
        "candidates": [{"wing": RECT}, {"name": "bare", **RECT}],
        "settings": FAST,
    })
    assert res["status"] == "success", res
    assert res["n_succeeded"] == 1 and res["n_failed"] == 1
    msg = res["failed"][0]["error"]["message"]
    assert res["failed"][0]["index"] == 1
    assert "candidates[1]" in msg and "unknown key" in msg


def test_batch_candidate_takes_only_wing_or_aircraft():
    # Review T-0052: per-candidate flight_condition and settings are not part of the card.
    res = call_tool("ventorum_batch_evaluate", {
        "candidates": [{"wing": RECT}, {"wing": RECT, "flight_condition": {"alpha_deg": 2.0}}],
        "settings": FAST,
    })
    assert res["status"] == "success", res
    assert res["n_failed"] == 1
    assert "unknown key 'flight_condition'" in res["failed"][0]["error"]["message"]
    props = next(t for t in AGENT_TOOL_DEFINITIONS if t["name"] == "ventorum_batch_evaluate")[
        "parameters"]["properties"]["candidates"]["items"]["properties"]
    assert set(props) == {"wing", "aircraft"}


def test_batch_candidate_error_path_names_the_argument():
    # Review T-0052: the card's example message is "candidates[0].wing: unknown key 'ref_point_m'".
    res = call_tool("ventorum_batch_evaluate", {
        "candidates": [{"wing": {**RECT, "ref_point_m": [0.0, 0.0, 0.0]}}],
        "settings": FAST,
    })
    msg = res["failed"][0]["error"]["message"]
    assert "candidates[0].wing: unknown key 'ref_point_m'" in msg
    assert "'ref_point_m' belongs in 'aircraft'" in msg


def test_aircraft_only_key_in_wing_gives_decision_2_hint():
    for key, val in (("surfaces", [RECT]), ("S_ref_m2", 10.0), ("b_ref_m", 10.0), ("c_ref_m", 1.0),
                     ("ref_point_m", [0.0, 0.0, 0.0])):
        res = call_tool("ventorum_wing_analysis", {"wing": {**RECT, key: val}, "settings": FAST})
        assert res["error"]["type"] == "invalid_input"
        assert "use the 'aircraft' argument for several surfaces or for reference values" in \
            res["error"]["message"], key


def test_python_polar_sweep_requires_the_alpha_range():
    # Review T-0052: the agent gave silent defaults (0, 10, 2 deg) to required arguments.
    from ventorum.agent import polar_sweep

    res = polar_sweep(wing=RECT, settings=FAST)
    assert res["status"] == "error" and res["error"]["type"] == "invalid_input"
    assert "alpha_start_deg" in res["error"]["message"]
