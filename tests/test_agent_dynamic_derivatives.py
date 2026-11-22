# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for the dynamic_derivatives agent tool and its MCP server integration."""

from __future__ import annotations

import json

import numpy as np
import pytest

import ventorum as vt
from ventorum.agent.mcp_server import handle_message
from ventorum.agent.tools import dynamic_derivatives as tool_dynamic_derivatives

RECT = {"span_m": 6.0, "chord_m": 1.0}
FAST = {"n_panels": 6, "n_chord": 8}


def _strict_json(payload):
    """Verify *payload* serialises as strict JSON (no NaN, no Infinity)."""
    return json.loads(json.dumps(payload, allow_nan=False))


def _rpc(msg: dict) -> dict:
    """Send one JSON-RPC request to the MCP handler and decode the reply."""
    reply = handle_message(json.dumps(msg))
    assert reply is not None
    return json.loads(reply)


@pytest.mark.parametrize("h_m", [None, 0.5])
@pytest.mark.parametrize("axes", ["body", "stability"])
def test_agent_tool_matches_library_function(h_m, axes):
    """The dynamic_derivatives agent tool returns the exact numbers of ventorum.dynamic_derivatives."""
    fc_spec: dict = {"V_inf_m_s": 30.0, "alpha_deg": 4.0}
    if h_m is not None:
        fc_spec["h_m"] = h_m
    k_vals = [0.0, 0.05, 0.2]
    x_cg = 0.25

    payload = tool_dynamic_derivatives(
        wing=RECT,
        flight_condition=fc_spec,
        x_cg_m=x_cg,
        k=k_vals,
        wake_chords=5.0,
        settings=FAST,
        detail_level="standard",
        axes=axes,
    )
    _strict_json(payload)
    assert payload["status"] == "success", payload
    assert payload["axes"] == axes
    assert payload["executive_summary"].startswith("[Ventorum RESULT]")
    assert payload["k"] == k_vals

    # Library reference with identical geometry, pivot, condition and settings.
    wing = vt.LiftingSurface(
        name="Wing",
        semi_span=3.0,
        sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=1.0)],
    )
    ac = vt.Aircraft(name="Wing", surfaces=[wing], ref_point=np.array([x_cg, 0.0, 0.0]))
    ac.compute_reference_values()
    fc = vt.FlightCondition(V_inf=30.0, rho=1.225, alpha=np.radians(4.0), h=h_m)
    st = vt.SolverSettings(solver_type="vlm", n_panels=6, n_chord=8)
    ref = vt.dynamic_derivatives(ac, fc, st, k=k_vals, wake_chords=5.0, axes=axes).to_dict()

    assert payload["CL0"] == pytest.approx(ref["CL0"], abs=1e-12)
    assert payload["Cm0"] == pytest.approx(ref["Cm0"], abs=1e-12)
    for key, ref_val in ref["k0"].items():
        assert payload["k0"][key] == pytest.approx(ref_val, abs=1e-12)
    for key, ref_arr in ref["derivatives"].items():
        assert payload["derivatives"][key] == pytest.approx(ref_arr, abs=1e-12)
    assert payload["mesh"]["n_chord"] == ref["mesh"]["n_chord"]
    assert payload["mesh"]["n_panels"] == ref["mesh"]["n_panels"]


def test_agent_tool_detail_levels_and_trust_notes():
    """detail_level controls verbosity and trust_notes reports coarse chordwise count."""
    coarse = {"n_panels": 6, "n_chord": 4}
    summary = tool_dynamic_derivatives(wing=RECT, settings=coarse, detail_level="summary")
    standard = tool_dynamic_derivatives(wing=RECT, settings=coarse, detail_level="standard")
    full = tool_dynamic_derivatives(wing=RECT, settings=coarse, detail_level="full")

    for p in (summary, standard, full):
        _strict_json(p)
        assert p["status"] == "success", p
        assert any("below 8" in msg for msg in p["trust_notes"])

    assert "geometry" not in summary
    assert "responses" not in summary
    assert "geometry" in standard and "responses" in standard
    assert "geometry" in full and "responses" in full


def test_agent_tool_input_refusals():
    """Invalid inputs return structured error payloads with strict JSON."""
    bad_calls = [
        {"wing": RECT, "k": [-0.1], "settings": FAST},
        {"wing": RECT, "k": [], "settings": FAST},
        {"wing": RECT, "wake_chords": 0.0, "settings": FAST},
        {"wing": RECT, "axes": "wind", "settings": FAST},
        {"wing": RECT, "settings": {"solver": "linear", "n_panels": 6}},
        {"wing": RECT, "flight_condition": {"V_inf_m_s": 120.0}, "settings": FAST},
        {"wing": RECT, "detail_level": "verbose", "settings": FAST},
        {"settings": FAST},  # missing wing / aircraft
    ]
    for kwargs in bad_calls:
        out = tool_dynamic_derivatives(**kwargs)
        _strict_json(out)
        assert out["status"] == "error", kwargs
        assert out["error"]["type"] == "invalid_input", out


def test_mcp_server_lists_and_calls_dynamic_derivatives():
    """MCP server exposes ventorum_dynamic_derivatives in tools/list and executes it in tools/call."""
    listed = _rpc({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    tools = listed["result"]["tools"]
    names = [t["name"] for t in tools]
    assert "ventorum_dynamic_derivatives" in names
    schema = next(t for t in tools if t["name"] == "ventorum_dynamic_derivatives")
    assert schema["inputSchema"]["additionalProperties"] is False
    assert "k" in schema["inputSchema"]["properties"]
    assert "wake_chords" in schema["inputSchema"]["properties"]

    # Valid call via MCP tools/call.
    called = _rpc({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "ventorum_dynamic_derivatives",
            "arguments": {
                "wing": RECT,
                "flight_condition": {"V_inf_m_s": 30.0, "alpha_deg": 4.0},
                "k": [0.0, 0.1],
                "settings": FAST,
            },
        },
    })
    assert called["result"]["isError"] is False
    body = json.loads(called["result"]["content"][0]["text"])
    assert body["status"] == "success"
    assert body["k"] == [0.0, 0.1]

    direct = tool_dynamic_derivatives(
        wing=RECT,
        flight_condition={"V_inf_m_s": 30.0, "alpha_deg": 4.0},
        k=[0.0, 0.1],
        settings=FAST,
    )
    assert body["k0"]["CL_alpha"] == pytest.approx(direct["k0"]["CL_alpha"], abs=1e-12)
    assert body["derivatives"]["Cm_q"] == pytest.approx(direct["derivatives"]["Cm_q"], abs=1e-12)

    # Unknown argument key returns isError=True.
    bad = _rpc({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "ventorum_dynamic_derivatives",
            "arguments": {"wing": RECT, "unknown_key": 1},
        },
    })
    assert bad["result"]["isError"] is True
    bad_body = json.loads(bad["result"]["content"][0]["text"])
    assert bad_body["status"] == "error"
    assert bad_body["error"]["type"] == "invalid_input"
