# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests of the 3D panel method in the error-bar agent tool.

The tool gives the mesh error estimate of the user mesh from a chordwise
and a spanwise refinement. The tests check finite bars on a small wing
(through the MCP server), the refusals before any solve (work budget,
mesh limit, mixed arguments) and the schema of the new arguments.
"""

from __future__ import annotations

import json
import math
from unittest.mock import patch

import pytest

from ventorum.agent import call_tool, get_tool_schemas, handle_message
from ventorum.agent.tools import error_bars, panel_analysis

RECT = {"span_m": 8.0, "chord_m": 1.0, "shape": "NACA 0012"}
# Base mesh with an estimate for CL, CD and Cm in both directions (see the
# rectangular wing study of tests/test_panel3d_convergence.py).
PANEL_MESH = {"n_chord": 16, "n_panels": 12}
NAMES = ("CL", "CD", "CY", "Cl", "Cm", "Cn")


def _mcp_call(arguments: dict) -> dict:
    """Call the error-bar tool through the MCP server and return the payload."""
    req = {
        "jsonrpc": "2.0",
        "id": 7,
        "method": "tools/call",
        "params": {"name": "ventorum_error_bars", "arguments": arguments},
    }
    reply = json.loads(handle_message(json.dumps(req)))
    assert reply.get("error") is None, reply
    return json.loads(reply["result"]["content"][0]["text"])


@pytest.fixture(scope="module")
def panel_bars() -> dict:
    """Panel error bars of the small rectangular wing, through the MCP server."""
    return _mcp_call({
        "wing": RECT,
        "flight_condition": {"alpha_deg": 4.0, "V_inf_m_s": 30.0},
        "method": "panel3d",
        "panel_settings": PANEL_MESH,
    })


@pytest.mark.slow
def test_panel_error_bars_are_finite_for_a_small_wing(panel_bars):
    """The six bars are present; CL, CD and Cm have finite intervals that hold the value."""
    out = panel_bars
    assert out["status"] == "success", out
    json.dumps(out, allow_nan=False)
    bars = out["error_bars"]["bars"]
    assert set(bars) == set(NAMES)
    for name in ("CL", "CD", "Cm"):
        bar = bars[name]
        assert bar["status"] == "numerical_only"
        assert bar["coverage"] is None
        low, high, value = bar["interval_low"], bar["interval_high"], bar["value"]
        assert all(math.isfinite(v) for v in (low, high, value)), name
        assert low < value < high, name
        half = bar["layers"]["numerical"]["half_width"]
        assert half == pytest.approx(high - value, rel=1e-9)
        assert bar["layers"]["numerical"]["state"] in ("monotonic", "estimate")
    assert out["error_bars"]["solver"] == "panel3d"
    assert out["error_bars"]["drag_basis"].startswith("CDi (Trefftz plane)")
    assert out["status_of_bars"] == "numerical_only"
    assert out["levels"] == [12, 18, 27]
    assert out["chord_levels"] == [16, 24, 36]
    assert len(out["meshes"]) == 5
    # Only the mesh part: the output says that the model-form error is not given.
    assert any("model-form error is not implemented" in n for n in out["notes"])
    assert "model-form error" in out["executive_summary"]
    for key in ("trust", "device", "precision", "condition_used", "settings_used", "axes"):
        assert key in out
    assert out["settings_used"]["solver"] == "panel3d"


@pytest.mark.slow
def test_panel_error_bar_value_is_the_user_mesh_value(panel_bars):
    """The value of each bar is the result of the user mesh (the coarsest level)."""
    direct = panel_analysis(wing=RECT, flight_condition={"alpha_deg": 4.0, "V_inf_m_s": 30.0},
                            settings=dict(PANEL_MESH))
    assert direct["status"] == "success", direct
    bars = panel_bars["error_bars"]["bars"]
    assert bars["CL"]["value"] == pytest.approx(direct["metrics"]["CL"], abs=1e-6)
    assert bars["CD"]["value"] == pytest.approx(direct["metrics"]["CD"], abs=1e-6)
    assert bars["Cm"]["value"] == pytest.approx(direct["metrics"]["Cm"], abs=1e-6)


def test_panel_error_bars_refuse_over_budget_before_any_solve():
    """A study above the panel work budget is refused before any solve."""
    with patch("ventorum.panel3d.convergence.panel_mesh_convergence") as study, \
            patch("ventorum.agent.tools.MAX_PANEL_CALL_WORK", 100):
        study.side_effect = RuntimeError("the study ran despite the budget")
        out = error_bars(wing=RECT, method="panel3d", panel_settings=PANEL_MESH)
        study.assert_not_called()
    assert out["status"] == "error", out
    assert out["error"]["type"] == "invalid_input"
    msg = out["error"]["message"]
    assert "budget" in msg and "mesh study of 5 meshes" in msg


def test_panel_error_bars_default_mesh_is_over_budget():
    """The default panel mesh (80 x 40) with its 5-mesh study is above the budget."""
    out = error_bars(wing=RECT, method="panel3d")
    assert out["status"] == "error", out
    assert "budget" in out["error"]["message"]


def test_panel_error_bars_refuse_a_finest_mesh_above_the_limit():
    """The mesh limit applies to each mesh of the study, also the finest ones."""
    # Base mesh 117 half-mesh panels; the finest chordwise mesh has 2.25 times more.
    with patch("ventorum.agent.tools.MAX_PANEL_PANELS", 200):
        out = error_bars(wing=RECT, method="panel3d", panel_settings={"n_chord": 9, "n_panels": 6})
    assert out["status"] == "error", out
    assert "mesh is too large" in out["error"]["message"].lower()


@pytest.mark.parametrize(
    "kwargs, text",
    [
        ({"method": "panel3d", "settings": {"n_panels": 16}}, "panel_settings"),
        ({"method": "panel3d", "panel_settings": PANEL_MESH, "input_uncertainties": {"alpha_deg": 0.1}},
         "input_uncertainties"),
        ({"panel_settings": PANEL_MESH}, "panel3d"),
        ({"method": "vortex"}, "method"),
        ({"method": "panel3d", "panel_settings": {"n_chord": 16, "n_panels": 12, "panel_kutta": "pressure"}},
         "Kutta"),
        ({"method": "panel3d", "panel_settings": {"n_span": 12}}, "n_span"),
    ],
)
def test_panel_error_bars_refuse_mixed_or_unknown_arguments(kwargs, text):
    """Lattice and panel arguments do not mix; unknown values are refused."""
    out = error_bars(wing=RECT, **kwargs)
    assert out["status"] == "error", out
    assert out["error"]["type"] == "invalid_input", out
    assert text in out["error"]["message"], out["error"]["message"]


def test_panel_error_bars_refuse_controls():
    """Control surfaces are not exposed for the panel solver."""
    wing = dict(RECT, controls=[{"name": "aileron", "hinge_x_c": 0.75, "deflection_deg": 5.0}])
    out = error_bars(wing=wing, method="panel3d", panel_settings=PANEL_MESH)
    assert out["status"] == "error", out
    assert "controls" in out["error"]["message"]


def test_error_bars_schema_has_the_panel_arguments():
    """All formats export 'method' and 'panel_settings'; the MCP schema stays strict."""
    for fmt in ("openai", "anthropic", "gemini", "mcp"):
        schemas = get_tool_schemas(fmt)
        json.dumps(schemas, allow_nan=False)
    mcp = {t["name"]: t for t in get_tool_schemas("mcp")}
    props = mcp["ventorum_error_bars"]["inputSchema"]["properties"]
    assert props["method"]["enum"] == ["lattice", "panel3d"]
    assert props["method"]["default"] == "lattice"
    panel = props["panel_settings"]
    assert panel["additionalProperties"] is False
    assert set(panel["properties"]) == {"n_chord", "n_panels", "panel_kutta", "panel_backend",
                                        "panel_ground_solver", "panel_cache"}
    assert mcp["ventorum_error_bars"]["inputSchema"]["additionalProperties"] is False


def test_error_bars_dispatcher_refuses_unknown_panel_key():
    """The dispatcher refuses a key that the panel settings schema does not have."""
    out = call_tool("ventorum_error_bars", {"wing": RECT, "method": "panel3d",
                                            "panel_settings": {"n_chord": 16, "wake": 1}})
    assert out["status"] == "error", out
    assert out["error"]["type"] == "invalid_input"
