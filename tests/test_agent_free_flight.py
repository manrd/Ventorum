# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for the free_flight agent tool and its MCP integration."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

import ventorum as vt
from ventorum.agent import call_tool
from ventorum.agent.mcp_server import handle_message

RECT_FLAP = {
    "span_m": 6.0,
    "chord_m": 1.0,
    "controls": [
        {"name": "flap", "eta_start": 0.0, "eta_end": 1.0, "hinge_x_c": 0.75, "deflection_deg": 0.0},
    ],
}
FAST = {"n_panels": 4, "n_chord": 4}
INERTIA = {"xx": 50.0, "yy": 75.0, "zz": 100.0}


def _strict_json(payload):
    """Verify *payload* serialises as strict JSON (no NaN, no Infinity)."""
    return json.loads(json.dumps(payload, allow_nan=False))


def _rpc(msg: dict) -> dict:
    """Send one JSON-RPC request to the MCP handler and decode the reply."""
    reply = handle_message(json.dumps(msg))
    assert reply is not None
    return json.loads(reply)


def test_agent_free_flight_matches_library():
    """A free z and pitch run with thrust, a flap step and an initial pitch rate matches ventorum.free_flight."""
    x_cg = 0.2
    payload = call_tool(
        "ventorum_free_flight",
        {
            "wing": RECT_FLAP,
            "flight_condition": {"V_inf_m_s": 20.0, "alpha_deg": 2.0},
            "x_cg_m": x_cg,
            "mass_kg": 250.0,
            "inertia_kg_m2": {**INERTIA, "xz": -2.0},
            "dofs": ["z", "pitch"],
            "n_steps": 6,
            "initial_state": {"pitch_dot_deg_s": 3.0},
            "controls": {"flap": {"kind": "step", "amplitude": 2.0, "t_start_s": 0.02}},
            "thrust": {"force_N": [-30.0, 0.0, 0.0], "moment_Nm": [0.0, 4.0, 0.0], "point_m": [0.5, 0.0, 0.1]},
            "settings": FAST,
        },
    )
    _strict_json(payload)
    assert payload["status"] == "success", payload
    assert payload["executive_summary"].startswith("[Ventorum RESULT] Free flight")
    assert payload["dofs"] == ["z", "pitch"]
    assert payload["n_steps"] == 6

    wing = vt.LiftingSurface(
        name="Wing",
        semi_span=3.0,
        sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=1.0)],
        controls=[vt.ControlSurface(name="flap", eta_start=0.0, eta_end=1.0, hinge_x_c=0.75, deflection=0.0)],
    )
    ac = vt.Aircraft(name="Wing", surfaces=[wing], ref_point=np.array([x_cg, 0.0, 0.0]))
    ac.compute_reference_values()
    ref = vt.free_flight(
        ac,
        vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=math.radians(2.0)),
        vt.SolverSettings(solver_type="vlm", n_panels=4, n_chord=4),
        mass=250.0,
        inertia=np.array([[50.0, 0.0, -2.0], [0.0, 75.0, 0.0], [-2.0, 0.0, 100.0]]),
        dofs=("z", "pitch"),
        n_steps=6,
        initial_state={"pitch_dot": math.radians(3.0)},
        controls={"flap": lambda t: math.radians(2.0) if t >= 0.02 else 0.0},
        thrust=lambda t: (np.array([-30.0, 0.0, 0.0]), np.array([0.0, 4.0, 0.0])),
        thrust_point=np.array([0.5, 0.0, 0.1]),
    ).to_dict()
    for key in ("CL", "Cm", "time"):
        assert payload[key] == pytest.approx(ref[key], abs=1e-12)
    for key in ("z_m", "pitch_rad"):
        assert payload["motion"][key] == pytest.approx(ref["motion"][key], abs=1e-12)
    assert payload["initial_imbalance"]["force_N"] == pytest.approx(ref["initial_imbalance"]["force_N"], abs=1e-9)


def test_agent_free_flight_through_mcp_and_summary_detail():
    """The tool is listed and runs through MCP tools/call; summary detail drops strip_cl and geometry."""
    listed = _rpc({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert "ventorum_free_flight" in [t["name"] for t in listed["result"]["tools"]]
    called = _rpc({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "ventorum_free_flight",
            "arguments": {
                "wing": RECT_FLAP,
                "flight_condition": {"V_inf_m_s": 20.0, "alpha_deg": 0.0, "h_m": 0.6},
                "mass_kg": 250.0,
                "inertia_kg_m2": INERTIA,
                "dofs": ["z", "pitch"],
                "n_steps": 3,
                "detail_level": "summary",
                "settings": FAST,
            },
        },
    })
    assert called["result"]["isError"] is False
    body = json.loads(called["result"]["content"][0]["text"])
    assert body["status"] == "success", body
    assert body["in_ground_effect"] is True
    assert "strip_cl" not in body and "geometry" not in body


def test_agent_free_flight_input_checks_and_work_budget_refusal():
    """Invalid inputs and calls over the work budget are refused with invalid_input."""
    ok = {"wing": RECT_FLAP, "mass_kg": 250.0, "inertia_kg_m2": INERTIA, "n_steps": 3, "settings": FAST}
    bad = [
        {k: v for k, v in ok.items() if k != "mass_kg"},
        {k: v for k, v in ok.items() if k != "inertia_kg_m2"},
        {**ok, "mass_kg": -1.0},
        {**ok, "inertia_kg_m2": {"xx": 1.0, "yy": 1.0}},
        {**ok, "inertia_kg_m2": {"xx": 1.0, "yy": 1.0, "zz": 1.0, "xy": 5.0}},  # not positive definite
        {**ok, "inertia_kg_m2": {**INERTIA, "ixx": 1.0}},
        {**ok, "dofs": ["heave"]},
        {**ok, "dofs": ["z", "z"]},
        {**ok, "dofs": ["z"], "initial_state": {"pitch_deg": 1.0}},
        {**ok, "dofs": ["z"], "motion": {"z_m": {"kind": "step", "amplitude": 0.1}}},
        {**ok, "thrust": {"force": [1.0, 0.0, 0.0]}},
        {**ok, "controls": {"aileron": {"kind": "step", "amplitude": 1.0}}},
        {**ok, "n_steps": None},
        {**ok, "t_end_s": 0.1},
        {**ok, "settings": {"solver": "linear", "n_panels": 4}},
    ]
    for args in bad:
        args = {k: v for k, v in args.items() if v is not None}
        out = call_tool("ventorum_free_flight", args)
        _strict_json(out)
        assert out["status"] == "error", args
        assert out["error"]["type"] == "invalid_input", out

    # 20 sub-iterations * n_steps * n_bound * n_wake: 20 * 60 * 320 * (60 * 40) = 9.2e8 > 2e8.
    over = call_tool(
        "ventorum_free_flight",
        {**ok, "n_steps": 60, "settings": {"n_panels": 20, "n_chord": 8}},
    )
    _strict_json(over)
    assert over["status"] == "error"
    assert over["error"]["type"] == "invalid_input"
    assert "budget" in over["error"]["message"].lower()


def test_free_flight_schema_exports_cleanly_for_all_formats():
    """The tool schema exports as strict JSON for openai, anthropic, mcp and gemini."""
    from ventorum.agent import get_tool_schemas

    for fmt in ("openai", "anthropic", "mcp", "gemini"):
        schemas = get_tool_schemas(fmt)
        _strict_json(schemas)
        found = {s["function"]["name"] for s in schemas} if fmt == "openai" else {s["name"] for s in schemas}
        assert "ventorum_free_flight" in found
        if fmt == "gemini":
            for s in schemas:
                if s["name"] == "ventorum_free_flight":
                    assert "additionalProperties" not in json.dumps(s)
                    assert "anyOf" not in json.dumps(s)
