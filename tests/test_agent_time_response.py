# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for the time_response and indicial_response agent tools and MCP integration."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

import ventorum as vt
from ventorum.agent import call_tool
from ventorum.agent.mcp_server import handle_message

RECT = {"span_m": 6.0, "chord_m": 1.0}
RECT_FLAP = {
    "span_m": 6.0,
    "chord_m": 1.0,
    "controls": [
        {"name": "flap", "eta_start": 0.0, "eta_end": 1.0, "hinge_x_c": 0.75, "deflection_deg": 0.0},
    ],
}
FAST = {"n_panels": 4, "n_chord": 4}


def _strict_json(payload):
    """Verify *payload* serialises as strict JSON (no NaN, no Infinity)."""
    return json.loads(json.dumps(payload, allow_nan=False))


def _rpc(msg: dict) -> dict:
    """Send one JSON-RPC request to the MCP handler and decode the reply."""
    reply = handle_message(json.dumps(msg))
    assert reply is not None
    return json.loads(reply)


def test_agent_time_response_harmonic_pitch_and_flap_matches_library():
    """Harmonic pitch with a moving flap through call_tool matches ventorum.time_response."""
    freq_hz = 1.5
    phase_deg = 30.0
    pitch_amp_deg = 2.0
    flap_amp_deg = 5.0
    x_cg = 0.25

    payload = call_tool(
        "ventorum_time_response",
        {
            "wing": RECT_FLAP,
            "flight_condition": {"V_inf_m_s": 25.0, "alpha_deg": 3.0},
            "x_cg_m": x_cg,
            "n_steps": 6,
            "motion": {
                "pitch_deg": {
                    "kind": "harmonic",
                    "amplitude": pitch_amp_deg,
                    "frequency_hz": freq_hz,
                    "phase_deg": phase_deg,
                },
            },
            "controls": {
                "flap": {
                    "kind": "harmonic",
                    "amplitude": flap_amp_deg,
                    "frequency_hz": freq_hz,
                    "phase_deg": 0.0,
                },
            },
            "settings": FAST,
            "detail_level": "standard",
        },
    )
    _strict_json(payload)
    assert payload["status"] == "success", payload
    assert payload["executive_summary"].startswith("[Ventorum RESULT]")
    assert payload["n_steps"] == 6
    assert payload["device"] == "cpu"

    wing = vt.LiftingSurface(
        name="Wing",
        semi_span=3.0,
        sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=1.0)],
        controls=[vt.ControlSurface(name="flap", eta_start=0.0, eta_end=1.0, hinge_x_c=0.75, deflection=0.0)],
    )
    ac = vt.Aircraft(name="Wing", surfaces=[wing], ref_point=np.array([x_cg, 0.0, 0.0]))
    ac.compute_reference_values()
    fc = vt.FlightCondition(V_inf=25.0, rho=1.225, alpha=math.radians(3.0))
    st = vt.SolverSettings(solver_type="vlm", n_panels=4, n_chord=4)
    omega = 2.0 * math.pi * freq_hz
    ref = vt.time_response(
        ac,
        fc,
        st,
        n_steps=6,
        motion={"pitch": lambda t: math.radians(pitch_amp_deg) * math.sin(omega * t + math.radians(phase_deg))},
        controls={"flap": lambda t: math.radians(flap_amp_deg) * math.sin(omega * t)},
    ).to_dict()

    for key in ("CL", "CD", "Cm", "time", "s"):
        assert payload[key] == pytest.approx(ref[key], abs=1e-12)


def test_agent_time_response_one_minus_cosine_gust_matches_library():
    """1-cosine gust through call_tool matches ventorum.time_response."""
    payload = call_tool(
        "ventorum_time_response",
        {
            "wing": RECT,
            "flight_condition": {"V_inf_m_s": 25.0, "alpha_deg": 2.0},
            "x_cg_m": 0.25,
            "n_steps": 8,
            "gust": {
                "kind": "one_minus_cosine",
                "amplitude_m_s": 2.0,
                "length_m": 4.0,
                "direction": "vertical",
                "x_front_m": 0.0,
            },
            "settings": FAST,
        },
    )
    _strict_json(payload)
    assert payload["status"] == "success", payload

    wing = vt.LiftingSurface(
        name="Wing",
        semi_span=3.0,
        sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=1.0)],
    )
    ac = vt.Aircraft(name="Wing", surfaces=[wing], ref_point=np.array([0.25, 0.0, 0.0]))
    ac.compute_reference_values()
    fc = vt.FlightCondition(V_inf=25.0, rho=1.225, alpha=math.radians(2.0))
    st = vt.SolverSettings(solver_type="vlm", n_panels=4, n_chord=4)
    ref = vt.time_response(
        ac,
        fc,
        st,
        n_steps=8,
        gust=vt.one_minus_cosine(2.0, 4.0, "vertical", x_front=0.0),
    ).to_dict()

    for key in ("CL", "CD", "Cm"):
        assert payload[key] == pytest.approx(ref[key], abs=1e-12)


def test_agent_time_response_wind_shear_ground_effect_and_waveforms():
    """Linear and logarithmic wind shear in ground effect, plus step/ramp/array waveforms."""
    payload = call_tool(
        "ventorum_time_response",
        {
            "wing": RECT,
            "flight_condition": {"V_inf_m_s": 25.0, "alpha_deg": 4.0, "h_m": 0.6},
            "x_cg_m": 0.25,
            "n_steps": 6,
            "motion": {
                "z_m": {"kind": "ramp", "amplitude": 0.05, "t_start_s": 0.0, "duration_s": 0.05},
                "pitch_deg": {"kind": "step", "amplitude": 1.0, "t_start_s": 0.01},
                "x_m": {"kind": "samples", "values": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
            },
            "gust": {
                "kind": "wind_shear",
                "profile": "logarithmic",
                "u_ref_m_s": 6.0,
                "z_ref_m": 10.0,
                "z0_m": 0.05,
                "direction": "head",
            },
            "settings": FAST,
            "store": "strengths",
            "detail_level": "full",
        },
    )
    _strict_json(payload)
    assert payload["status"] == "success", payload
    assert payload["in_ground_effect"] is True
    assert "gamma_bound" in payload and "gamma_wake" in payload and "wake_corners" in payload
    notes_txt = " ".join(payload["notes"]).lower()
    assert "rotational" in notes_txt
    assert "parallel to the ground" in notes_txt

    # Linear shear profile also runs cleanly.
    lin = call_tool(
        "ventorum_time_response",
        {
            "wing": RECT,
            "flight_condition": {"V_inf_m_s": 25.0, "alpha_deg": 3.0, "h_m": 0.8},
            "t_end_s": 0.05,
            "gust": {
                "kind": "wind_shear",
                "profile": "linear",
                "u0_m_s": 1.0,
                "du_dz_per_s": 2.0,
                "z_ref_m": 0.0,
                "direction": "head",
            },
            "settings": FAST,
            "detail_level": "summary",
        },
    )
    _strict_json(lin)
    assert lin["status"] == "success", lin
    assert "geometry" not in lin
    assert "strip_cl" not in lin


def test_agent_indicial_response_and_mcp_server():
    """ventorum_indicial_response runs via call_tool and MCP tools/call and matches library."""
    listed = _rpc({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    names = [t["name"] for t in listed["result"]["tools"]]
    assert "ventorum_time_response" in names
    assert "ventorum_indicial_response" in names

    called = _rpc({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "ventorum_indicial_response",
            "arguments": {
                "wing": RECT_FLAP,
                "flight_condition": {"V_inf_m_s": 20.0, "alpha_deg": 0.0},
                "x_cg_m": 0.25,
                "kind": "control",
                "control": "flap",
                "s_end": 4.0,
                "settings": FAST,
            },
        },
    })
    assert called["result"]["isError"] is False
    body = json.loads(called["result"]["content"][0]["text"])
    assert body["status"] == "success", body
    assert body["kind"] == "control"
    assert body["steady"] is not None
    assert body["executive_summary"].startswith("[Ventorum RESULT]")

    # Also test kind="alpha" and kind="gust" via call_tool.
    for kind in ("alpha", "gust"):
        res = call_tool(
            "ventorum_indicial_response",
            {
                "wing": RECT,
                "flight_condition": {"V_inf_m_s": 20.0, "alpha_deg": 0.0},
                "kind": kind,
                "s_end": 4.0,
                "settings": FAST,
            },
        )
        _strict_json(res)
        assert res["status"] == "success", res
        assert res["kind"] == kind


def test_agent_time_response_input_checks_and_work_budget_refusal():
    """Invalid inputs and calls exceeding the work budget are refused with invalid_input."""
    bad_tr = [
        {"wing": RECT, "settings": FAST},  # missing t_end_s and n_steps
        {"wing": RECT, "t_end_s": 0.1, "n_steps": 4, "settings": FAST},  # both given
        {"wing": RECT, "n_steps": 0, "settings": FAST},
        {"wing": RECT, "n_steps": 4, "store": "cp", "settings": FAST},
        {"wing": RECT, "n_steps": 4, "settings": {"solver": "linear", "n_panels": 4}},
        {"wing": RECT, "n_steps": 4, "flight_condition": {"V_inf_m_s": 120.0}, "settings": FAST},
        {"wing": RECT, "n_steps": 4, "gust": {"kind": "unknown"}, "settings": FAST},
        {"wing": RECT, "n_steps": 4, "gust": {"kind": "sharp_edged", "amplitude_m_s": 1.0, "direction": "up"}, "settings": FAST},
        {"wing": RECT, "n_steps": 4, "motion": {"heave_m": [0, 0, 0, 0, 0]}, "settings": FAST},
        {"wing": RECT, "n_steps": 4, "motion": {"pitch_deg": {"kind": "triangle", "amplitude": 1.0}}, "settings": FAST},
        {"wing": RECT, "n_steps": 4, "unknown_key": 1, "settings": FAST},
    ]
    for args in bad_tr:
        out = call_tool("ventorum_time_response", args)
        _strict_json(out)
        assert out["status"] == "error", args
        assert out["error"]["type"] == "invalid_input", out

    bad_ind = [
        {"wing": RECT, "settings": FAST},  # missing kind
        {"wing": RECT, "kind": "pitch", "settings": FAST},
        {"wing": RECT, "kind": "control", "settings": FAST},  # missing control
        {"wing": RECT, "kind": "alpha", "control": "flap", "settings": FAST},
        {"wing": RECT, "kind": "alpha", "s_end": -1.0, "settings": FAST},
        {"wing": RECT, "kind": "alpha", "settings": {"solver": "fourier", "n_panels": 4}},
    ]
    for args in bad_ind:
        out = call_tool("ventorum_indicial_response", args)
        _strict_json(out)
        assert out["status"] == "error", args
        assert out["error"]["type"] == "invalid_input", out

    # Work budget refusal: n_steps * n_bound * (n_steps * n_strips) > MAX_CALL_WORK (2e8).
    # n_panels = 20 -> n_strips = 40, n_chord = 8 -> n_bound = 320.
    # n_steps = 200 -> n_wake = 8000 -> work = 200 * 320 * 8000 = 5.12e8 > 2e8.
    over_budget = call_tool(
        "ventorum_time_response",
        {
            "wing": RECT,
            "n_steps": 200,
            "settings": {"n_panels": 20, "n_chord": 8},
        },
    )
    _strict_json(over_budget)
    assert over_budget["status"] == "error"
    assert over_budget["error"]["type"] == "invalid_input"
    assert "budget" in over_budget["error"]["message"].lower()

    # Work budget refusal on indicial_response as well.
    over_ind = call_tool(
        "ventorum_indicial_response",
        {
            "wing": RECT,
            "kind": "alpha",
            "s_end": 100.0,  # 400 steps at n_chord=8 -> 400 * 320 * 16000 = 2.048e9 > 2e8
            "settings": {"n_panels": 20, "n_chord": 8},
        },
    )
    _strict_json(over_ind)
    assert over_ind["status"] == "error"
    assert over_ind["error"]["type"] == "invalid_input"
    assert "budget" in over_ind["error"]["message"].lower()


def test_time_response_schemas_export_cleanly_for_all_formats():
    """The two new tool schemas export as strict JSON for openai, anthropic, mcp and gemini."""
    from ventorum.agent import get_tool_schemas

    new_names = {"ventorum_time_response", "ventorum_indicial_response"}
    for fmt in ("openai", "anthropic", "mcp", "gemini"):
        schemas = get_tool_schemas(fmt)
        _strict_json(schemas)
        if fmt == "openai":
            found = {s["function"]["name"] for s in schemas}
        else:
            found = {s["name"] for s in schemas}
        assert new_names <= found
        if fmt == "gemini":
            for s in schemas:
                if s["name"] in new_names:
                    assert "additionalProperties" not in json.dumps(s)
                    assert "anyOf" not in json.dumps(s)

