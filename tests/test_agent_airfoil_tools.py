# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests of the 2D airfoil geometry and polar agent tools."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from ventorum.aero.xfoil_runner import resolve_xfoil_executable, xfoil_version
from ventorum.agent import call_tool
from ventorum.agent.mcp_server import handle_message
from ventorum.agent.schemas import get_tool_schemas
from ventorum.airfoil import airfoil, write_lednicer


def strict_json(payload: dict[str, Any]) -> dict[str, Any]:
    """Ensure the payload is strict JSON (no NaN, no infinity)."""
    return json.loads(json.dumps(payload, allow_nan=False))


def _sample_lednicer_text(tmp_path) -> str:
    """Build a Lednicer-format coordinate text for a symmetric airfoil."""
    shape = airfoil("0012", n_points=41)
    p = tmp_path / "sample.led"
    write_lednicer(shape, p, precision=6)
    return p.read_text()


# ── Airfoil geometry tool ───────────────────────────────────────────────────


def test_airfoil_geometry_naca2412():
    """NACA 2412 geometry: check coordinates, thickness, camber and TE gap."""
    res = call_tool("ventorum_airfoil_geometry", {"section": "NACA 2412"})
    strict_json(res)
    assert res["status"] == "success", res
    assert res["name"] == "NACA 2412"
    assert "coordinates" in res
    assert "normalized_coordinates" in res
    coords = np.array(res["coordinates"], dtype=float)
    assert coords.ndim == 2 and coords.shape[1] == 2
    assert coords.shape[0] >= 10

    # NACA 2412: 12 % thickness at 30 % chord, 2 % camber at 40 % chord
    assert res["max_thickness"] == pytest.approx(0.12, abs=0.005)
    assert res["max_thickness_x"] == pytest.approx(0.30, abs=0.05)
    assert res["max_camber"] == pytest.approx(0.02, abs=0.003)
    assert res["max_camber_x"] == pytest.approx(0.40, abs=0.05)

    assert "trailing_edge_gap" in res
    assert res["trailing_edge_gap"] >= 0.0
    assert res["problems"] == []


def test_airfoil_geometry_lednicer(tmp_path):
    """Airfoil geometry tool correctly parses Lednicer format text."""
    text = _sample_lednicer_text(tmp_path)
    res = call_tool("ventorum_airfoil_geometry", {"section": text})
    strict_json(res)
    assert res["status"] == "success", res
    assert "coordinates" in res
    coords = np.array(res["coordinates"], dtype=float)
    assert coords.ndim == 2 and coords.shape[1] == 2
    assert res["max_thickness"] == pytest.approx(0.12, abs=0.005)
    assert res["problems"] == []


def test_airfoil_geometry_flap_and_repanel():
    """Airfoil geometry tool repanels nodes and deflects plain flap."""
    res = call_tool(
        "ventorum_airfoil_geometry",
        {
            "section": "NACA 0012",
            "n_panels": 80,
            "flap": {"x_hinge": 0.75, "deflection_deg": 10.0},
        },
    )
    strict_json(res)
    assert res["status"] == "success", res
    assert len(res["coordinates"]) >= 81
    assert all("monotonic" in p for p in res["problems"])


def test_airfoil_geometry_invalid_input():
    """Invalid section specification returns invalid_input error."""
    res = call_tool("ventorum_airfoil_geometry", {"section": "NOT_AN_AIRFOIL_12345"})
    strict_json(res)
    assert res["status"] == "error", res
    assert res["error"]["type"] == "invalid_input", res


def test_airfoil_geometry_panel_limit():
    """Repanel count above the limit is refused."""
    res = call_tool(
        "ventorum_airfoil_geometry",
        {"section": "NACA 0012", "n_panels": 5000},
    )
    strict_json(res)
    assert res["status"] == "error", res
    assert res["error"]["type"] == "invalid_input", res
    assert "limit" in res["error"]["message"].lower()


# ── Airfoil polar tool ──────────────────────────────────────────────────────


def test_airfoil_polar_ventorum2d_naca0012():
    """Ventorum-2D on NACA 0012 (Re 1e6, alpha 0 to 4) returns converged points."""
    res = call_tool(
        "ventorum_airfoil_polar",
        {
            "section": "NACA 0012",
            "Re": 1.0e6,
            "alphas_deg": [0.0, 1.0, 2.0, 3.0, 4.0],
            "source": "ventorum2d",
        },
    )
    strict_json(res)
    assert res["status"] == "success", res
    assert res["source"] == "ventorum2d"
    assert "options" in res
    assert "warnings" in res
    assert len(res["alpha_deg"]) == 5
    assert len(res["Cl"]) == 5
    assert len(res["Cd"]) == 5
    assert len(res["Cm"]) == 5
    assert len(res["converged"]) == 5
    assert all(res["converged"]), res["converged"]
    assert res["Cl"][0] == pytest.approx(0.0, abs=0.02)
    assert res["Cl"][4] > res["Cl"][0]
    assert res["transition_positions"] is not None
    assert len(res["transition_positions"]) == 5


def test_airfoil_polar_range_input():
    """Polar tool accepts alpha range parameters."""
    res = call_tool(
        "ventorum_airfoil_polar",
        {
            "section": "NACA 0012",
            "Re": 1.0e6,
            "alpha_start_deg": 0.0,
            "alpha_end_deg": 2.0,
            "alpha_step_deg": 1.0,
            "source": "ventorum2d",
        },
    )
    strict_json(res)
    assert res["status"] == "success", res
    assert res["alpha_deg"] == [0.0, 1.0, 2.0]
    assert all(res["converged"])


def test_airfoil_polar_with_flap_and_ground():
    """Polar tool with plain flap and ground effect runs correctly."""
    res = call_tool(
        "ventorum_airfoil_polar",
        {
            "section": "NACA 0012",
            "Re": 1.0e6,
            "alphas_deg": [0.0, 2.0],
            "flap": {"x_hinge": 0.75, "deflection_deg": 5.0},
            "h_c": 1.0,
            "source": "ventorum2d",
        },
    )
    strict_json(res)
    assert res["status"] == "success", res
    assert len(res["Cl"]) == 2


def test_airfoil_polar_budget_refusal():
    """Polar call above work budget is refused before solving."""
    # Too many angles
    many_alphas = [round(float(a), 4) for a in np.linspace(-15.0, 15.0, 70)]
    res = call_tool(
        "ventorum_airfoil_polar",
        {
            "section": "NACA 0012",
            "Re": 1.0e6,
            "alphas_deg": many_alphas,
            "source": "ventorum2d",
        },
    )
    strict_json(res)
    assert res["status"] == "error", res
    assert res["error"]["type"] == "invalid_input", res
    assert "budget" in res["error"]["message"].lower()
    assert "split" in res["error"]["message"].lower()

    # Panel count too large
    res_panels = call_tool(
        "ventorum_airfoil_polar",
        {
            "section": "NACA 0012",
            "Re": 1.0e6,
            "alphas_deg": [0.0, 2.0],
            "n_panels": 5000,
            "source": "ventorum2d",
        },
    )
    strict_json(res_panels)
    assert res_panels["status"] == "error", res_panels
    assert res_panels["error"]["type"] == "invalid_input", res_panels
    assert "limit" in res_panels["error"]["message"].lower()


def test_airfoil_polar_xfoil_skips_when_absent():
    """XFOIL source runs when installed, or skips cleanly when absent."""
    exe = resolve_xfoil_executable()
    ver = xfoil_version(exe)
    if ver == "unknown":
        pytest.skip("XFOIL is absent")

    res = call_tool(
        "ventorum_airfoil_polar",
        {
            "section": "NACA 0012",
            "Re": 1.0e6,
            "alphas_deg": [0.0, 2.0],
            "source": "xfoil",
        },
    )
    strict_json(res)
    assert res["status"] == "success", res
    assert res["source"] == "xfoil"
    assert len(res["Cl"]) == 2


def test_airfoil_polar_xfoil_missing_gives_clear_error(monkeypatch):
    """When XFOIL executable is missing, a clear tool error is returned."""
    import ventorum.agent.tools as agent_tools

    monkeypatch.setattr(
        agent_tools,
        "xfoil_version",
        lambda *args, **kwargs: "unknown",
    )
    res = call_tool(
        "ventorum_airfoil_polar",
        {
            "section": "NACA 0012",
            "Re": 1.0e6,
            "alphas_deg": [0.0, 2.0],
            "source": "xfoil",
        },
    )
    strict_json(res)
    assert res["status"] == "error", res
    assert res["error"]["type"] in ("invalid_input", "invalid_method")
    assert "xfoil" in res["error"]["message"].lower()


def test_mcp_airfoil_tools_registered():
    """MCP tools/list includes the 2D airfoil tools and tools/call works."""
    schemas = get_tool_schemas("mcp")
    names = [s["name"] for s in schemas]
    assert "ventorum_airfoil_geometry" in names
    assert "ventorum_airfoil_polar" in names

    msg = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "ventorum_airfoil_geometry",
            "arguments": {"section": "NACA 0012"},
        },
    }
    raw = handle_message(json.dumps(msg))
    assert raw is not None
    resp = json.loads(raw)
    assert resp["id"] == 1
    assert "result" in resp
    content = json.loads(resp["result"]["content"][0]["text"])
    assert content["status"] == "success"
    assert content["name"] == "NACA 0012"
