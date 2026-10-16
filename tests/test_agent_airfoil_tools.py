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
    # One key per output: the coordinates are the normalized ones.
    assert "normalized_coordinates" not in res
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
    for repeated in ("te_gap", "thickness", "camber", "normalized_coordinates"):
        assert repeated not in res
    assert coords[0, 0] == pytest.approx(1.0, abs=1e-3)  # trailing edge at unit chord


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
    assert "alpha" not in res  # one key per output
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


# ── Security: the caller cannot choose the program or free options ─────────


@pytest.fixture
def no_subprocess(monkeypatch):
    """Fail the test if any process starts."""
    import subprocess

    import ventorum.aero.xfoil_runner as runner

    started: list = []

    def refuse(*args, **kwargs):
        started.append(args)
        raise AssertionError(f"a process was started: {args!r}")

    monkeypatch.setattr(subprocess, "run", refuse)
    monkeypatch.setattr(subprocess, "Popen", refuse)
    monkeypatch.setattr(runner.subprocess, "run", refuse)
    return started


def _mcp_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    msg = {"jsonrpc": "2.0", "id": 9, "method": "tools/call",
           "params": {"name": name, "arguments": arguments}}
    resp = json.loads(handle_message(json.dumps(msg)))
    assert resp["result"]["isError"] is True, resp
    return json.loads(resp["result"]["content"][0]["text"])


def test_mcp_executable_in_options_is_refused(no_subprocess):
    """Security regression: options.executable reached the XFOIL launcher."""
    payload = _mcp_call("ventorum_airfoil_polar", {
        "section": "NACA 0012", "Re": 1e6, "alphas_deg": [0.0], "source": "xfoil",
        "options": {"executable": "calc.exe"},
    })
    assert payload["error"]["type"] == "invalid_input"
    assert "options" in payload["error"]["message"]
    payload = _mcp_call("ventorum_airfoil_polar", {
        "section": "NACA 0012", "Re": 1e6, "alphas_deg": [0.0], "source": "xfoil",
        "executable": "calc.exe",
    })
    assert payload["error"]["type"] == "invalid_input"
    assert "executable" in payload["error"]["message"]
    assert no_subprocess == []


def test_free_form_and_unbounded_options_are_refused(no_subprocess):
    base = {"section": "NACA 0012", "Re": 1e6, "alphas_deg": [0.0, 2.0]}
    for extra, fragment in (
        ({"tol": 1e-30}, "tol"),
        ({"timeout": None}, "timeout"),
        ({"max_iter": 10_000_000}, "max_iter"),
        ({"max_iter": 0}, "max_iter"),
        ({"ncrit": 50.0}, "ncrit"),
        ({"xtr_top": 1.5}, "xtr_top"),
        ({"n_panels": 2}, "n_panels"),
    ):
        res = call_tool("ventorum_airfoil_polar", {**base, **extra})
        assert res["status"] == "error", extra
        assert res["error"]["type"] == "invalid_input", extra
        assert fragment in res["error"]["message"], (extra, res["error"]["message"])
    assert no_subprocess == []


@pytest.mark.parametrize("alias", ["re", "height", "h_over_c", "alpha_deg", "options"])
def test_polar_aliases_are_refused(alias):
    args = {"section": "NACA 0012", "Re": 1e6, "alphas_deg": [0.0], alias: 1.0}
    res = call_tool("ventorum_airfoil_polar", args)
    assert res["status"] == "error"
    assert res["error"]["type"] == "invalid_input"
    assert f"unknown key '{alias}'" in res["error"]["message"]


def test_geometry_alias_is_refused():
    res = call_tool("ventorum_airfoil_geometry", {"section": "NACA 0012", "repanel_count": 80})
    assert res["status"] == "error"
    assert "unknown key 'repanel_count'" in res["error"]["message"]


@pytest.mark.parametrize("flap, fragment", [
    ({"hinge_x_c": 0.75, "deflection_deg": 5.0}, "unknown key 'hinge_x_c'"),
    ({"x_hinge_c": 0.75, "deflection_deg": 5.0}, "unknown key 'x_hinge_c'"),
    ({"x_hinge": 0.75, "deflection": 5.0}, "unknown key 'deflection'"),
    ({"x_hinge": 0.75, "deflection_deg": 5.0, "y_hinge": 1.5}, "y_hinge"),
    ({"x_hinge": 0.75, "deflection_deg": 90.0}, "deflection_deg"),
    ({"x_hinge": 0.75, "deflection_deg": None}, "'deflection_deg' is null"),
    ({"x_hinge": 0.75}, "deflection_deg"),
    ([0.75, 5.0], "JSON object"),
])
def test_nested_flap_keys_follow_the_schema(flap, fragment):
    for tool, extra in (("ventorum_airfoil_geometry", {}),
                        ("ventorum_airfoil_polar", {"Re": 1e6, "alphas_deg": [0.0]})):
        res = call_tool(tool, {"section": "NACA 0012", "flap": flap, **extra})
        assert res["status"] == "error", (tool, flap)
        assert res["error"]["type"] == "invalid_input"
        assert fragment in res["error"]["message"], (tool, res["error"]["message"])


def test_flap_y_hinge_is_a_thickness_fraction():
    hi = call_tool("ventorum_airfoil_geometry", {
        "section": "NACA 0012", "flap": {"x_hinge": 0.75, "y_hinge": 1.0, "deflection_deg": 20.0}})
    lo = call_tool("ventorum_airfoil_geometry", {
        "section": "NACA 0012", "flap": {"x_hinge": 0.75, "y_hinge": 0.0, "deflection_deg": 20.0}})
    assert hi["status"] == lo["status"] == "success"
    assert hi["flap"] == {"x_hinge": 0.75, "y_hinge": 1.0, "deflection_deg": 20.0}
    # A different hinge height changes the deflected shape.
    assert hi["coordinates"] != lo["coordinates"]


def test_xfoil_gets_parsed_coordinates_not_text(tmp_path, monkeypatch, no_subprocess):
    """Security regression: coordinate text went to XFOIL as a file name."""
    import ventorum.agent.tools as agent_tools
    from ventorum.aero import polar_source
    from ventorum.core.datatypes import TabulatedAirfoil

    seen: list = []

    def fake_polar(self, section, Re, alphas_deg, *, mach=0.0, flap=None, **options):
        seen.append({"section": section, "flap": flap, "settings": self.effective_options()})
        alphas = np.radians(np.asarray(alphas_deg, dtype=float))
        return TabulatedAirfoil(name="fake", alpha=alphas, Cl_data=2 * np.pi * alphas,
                                Cd_data=np.full(alphas.shape, 0.01),
                                Cm_data=np.zeros(alphas.shape), Re=Re)

    monkeypatch.setattr(agent_tools, "xfoil_version", lambda *a, **k: "XFOIL Version 6.99")
    monkeypatch.setattr(polar_source, "xfoil_version", lambda *a, **k: "XFOIL Version 6.99")
    monkeypatch.setattr(polar_source.XfoilPolarSource, "polar", fake_polar)
    monkeypatch.setenv("VENTORUM_CACHE", str(tmp_path / "cache"))
    text = _sample_lednicer_text(tmp_path)
    res = call_tool("ventorum_airfoil_polar", {
        "section": text, "Re": 1e6, "alphas_deg": [0.0, 2.0], "source": "xfoil",
        "flap": {"x_hinge": 0.7, "deflection_deg": 5.0},
    })
    assert res["status"] == "success", res
    assert len(seen) == 1
    section = seen[0]["section"]
    assert isinstance(section, np.ndarray) and section.ndim == 2 and section.shape[1] == 2
    assert seen[0]["flap"] == (0.7, 0.5, 5.0)
    settings = seen[0]["settings"]
    # Server-side limits: fixed timeout, bounded iterations, no executable from the caller.
    assert settings["timeout"] == agent_tools.AIRFOIL_XFOIL_TIMEOUT_S
    assert settings["max_iter"] == agent_tools.AIRFOIL_DEFAULT_MAX_ITER["xfoil"]
    assert settings["executable"] is None
    assert res["options"]["version"] == "XFOIL Version 6.99"
    assert no_subprocess == []


def test_xfoil_refuses_ground_height(monkeypatch):
    import ventorum.agent.tools as agent_tools

    monkeypatch.setattr(agent_tools, "xfoil_version", lambda *a, **k: "XFOIL Version 6.99")
    res = call_tool("ventorum_airfoil_polar", {
        "section": "NACA 0012", "Re": 1e6, "alphas_deg": [0.0], "source": "xfoil", "h_c": 1.0})
    assert res["status"] == "error"
    assert res["error"]["type"] == "invalid_method"


def test_work_budget_counts_iterations_and_panels():
    res = call_tool("ventorum_airfoil_polar", {
        "section": "NACA 0012", "Re": 1e6, "alpha_start_deg": -15.0, "alpha_end_deg": 15.0,
        "alpha_step_deg": 0.5, "n_panels": 400, "max_iter": 200})
    assert res["status"] == "error"
    msg = res["error"]["message"]
    assert "budget" in msg and "split" in msg.lower()
    # 2e8 / (200 * 400^2) = 6 angles per call.
    assert "at most 6 angles" in msg
    too_low = call_tool("ventorum_airfoil_polar", {
        "section": "NACA 0012", "Re": 1e6, "alphas_deg": [0.0], "h_c": 0.01})
    assert too_low["status"] == "error" and "h_c" in too_low["error"]["message"]


def test_section_size_limits():
    many = "\n".join(f"{x:.6f} 0.0" for x in np.linspace(1.0, 0.0, 2500))
    res = call_tool("ventorum_airfoil_geometry", {"section": many})
    assert res["status"] == "error" and "limit" in res["error"]["message"]


def test_agent_guide_agrees_with_the_code():
    """The guide states the same budget, height limit and key names as the schema."""
    from pathlib import Path

    from ventorum.agent import schemas

    guide = (Path(__file__).resolve().parents[1] / "docs" / "agent" / "index.md").read_text(encoding="utf-8")
    start = guide.index("## 2D airfoil tools")
    section = guide[start:guide.index("\n## ", start + 5)]
    assert f"{schemas.MAX_AIRFOIL_SOLVES} angles" in section
    assert f"{schemas.MAX_AIRFOIL_PANELS} panels" in section
    assert f"{schemas.MAX_AIRFOIL_ITER}" in section
    assert f"{schemas.MIN_AIRFOIL_H_C:g}" in section
    assert f"{int(schemas.MAX_CALL_WORK)}" in section
    assert f"{schemas.AIRFOIL_XFOIL_TIMEOUT_S:g} s" in section
    for key in list(schemas.AIRFOIL_POLAR_PROPS) + list(schemas.AIRFOIL_GEOMETRY_PROPS):
        assert f"`{key}`" in section, key
    for key in schemas.AIRFOIL_FLAP_PROPS:
        assert f"`{key}`" in section, key
    for gone in ("hinge_x_c", "xtr_upper", "repanel_count", "executive_summary"):
        assert gone not in section, gone
