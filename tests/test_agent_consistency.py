# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Consistency tests of task T-0009: selectable moment axes, error-type
mapping, batch n_chord report, single-mesh ground-effect polar sweep and
MCP survival on invalid input.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from ventorum.agent import ground_effect, wing_analysis
from ventorum.agent.response import error_info, error_type_of
from ventorum.agent.schemas import InputError
from ventorum.core.axes import transform_moments
from ventorum.core.errors import GroundStrikeError, ValidityError

FAST = {"n_panels": 8}
SMALL = {"n_panels": 12, "n_chord": 2}
RECT = {"span_m": 10.0, "chord_m": 1.0}


def test_cn_sign_is_the_same_in_all_tools():
    """At beta = 5 deg the two tools give Cl and Cn with the same sign."""
    cond = {"V_inf_m_s": 50.0, "alpha_deg": 4.0, "beta_deg": 5.0}
    free = wing_analysis(RECT, cond, FAST, "summary")
    assert free["status"] == "success", free
    assert free["axes"] == "body"
    ge = ground_effect(RECT, [20.0], alpha_deg=4.0, beta_deg=5.0, settings=FAST)
    assert ge["status"] == "success", ge
    assert ge["axes"] == "body"
    row = next(r for r in ge["rows"] if r["status"] == "ok")
    m = free["metrics"]
    for key in ("Cl", "Cn"):
        assert m[key] * row[key] > 0.0, (key, m[key], row[key])
        assert abs(m[key] - row[key]) / max(abs(m[key]), 1e-12) < 0.02


def test_axes_round_trip():
    """Body to stability to wind and back gives the input to 1e-12."""
    alpha, beta = np.radians(7.5), np.radians(5.0)
    start = (0.0123, -0.0456, 0.0078)
    stab = transform_moments(*start, alpha, beta, "stability")
    wind = transform_moments(stab["Cl"], stab["Cm"], stab["Cn"], alpha, beta, "wind",
                             from_axes="stability")
    back = transform_moments(wind["Cl"], wind["Cm"], wind["Cn"], alpha, beta, "body",
                             from_axes="wind")
    assert [back["Cl"], back["Cm"], back["Cn"]] == pytest.approx(list(start), abs=1e-12)


def test_axes_hand_cases():
    """Hand-computed rotations at alpha = 90 deg and at beta = 90 deg."""
    cl, cm, cn = 1.0, 2.0, 3.0
    got = transform_moments(cl, cm, cn, np.radians(90.0), 0.0, "stability")
    assert [got["Cl"], got["Cm"], got["Cn"]] == pytest.approx([3.0, 2.0, -1.0], abs=1e-12)
    got = transform_moments(cl, cm, cn, np.radians(90.0), 0.0, "wind")
    assert [got["Cl"], got["Cm"], got["Cn"]] == pytest.approx([3.0, 2.0, -1.0], abs=1e-12)
    got = transform_moments(cl, cm, cn, 0.0, np.radians(90.0), "stability")
    assert [got["Cl"], got["Cm"], got["Cn"]] == pytest.approx([1.0, 2.0, 3.0], abs=1e-12)
    got = transform_moments(cl, cm, cn, 0.0, np.radians(90.0), "wind")
    assert [got["Cl"], got["Cm"], got["Cn"]] == pytest.approx([2.0, -1.0, 3.0], abs=1e-12)


def test_axes_all_returns_three_sets():
    """axes='all' returns the body, stability and wind sets."""
    one = wing_analysis(RECT, {"alpha_deg": 4.0, "beta_deg": 5.0}, FAST, "summary", axes="body")
    all_axes = wing_analysis(RECT, {"alpha_deg": 4.0, "beta_deg": 5.0}, FAST, "summary",
                             axes="all")
    assert all_axes["status"] == "success", all_axes
    assert all_axes["axes"] == "all"
    sets = all_axes["metrics"]["moments"]
    assert set(sets) == {"body", "stability", "wind"}
    for _name, triple in sets.items():
        assert set(triple) == {"Cl", "Cm", "Cn"}
    assert sets["body"]["Cl"] == pytest.approx(one["metrics"]["Cl"])
    assert sets["body"]["Cm"] == pytest.approx(one["metrics"]["Cm"])
    assert sets["body"]["Cn"] == pytest.approx(one["metrics"]["Cn"])
    single = wing_analysis(RECT, {"alpha_deg": 4.0, "beta_deg": 5.0}, FAST, "summary",
                           axes="wind")
    assert single["metrics"]["Cl"] == pytest.approx(sets["wind"]["Cl"])
    assert single["metrics"]["Cm"] == pytest.approx(sets["wind"]["Cm"])
    assert single["metrics"]["Cn"] == pytest.approx(sets["wind"]["Cn"])


def test_agent_axes_input():
    """Each axes value is accepted; an unknown value is refused."""
    from ventorum.agent import batch_evaluate, polar_sweep, stability_derivatives

    for axes in ("body", "stability", "wind", "all"):
        p = wing_analysis(RECT, None, FAST, "summary", axes=axes)
        assert p["status"] == "success", (axes, p)
        assert p["axes"] == axes
        p = polar_sweep(RECT, 0.0, 2.0, 2.0, None, FAST, "summary", axes=axes)
        assert p["status"] == "success", (axes, p)
        assert p["axes"] == axes
        p = ground_effect(RECT, [1.0], settings=FAST, detail_level="summary", axes=axes)
        assert p["status"] == "success", (axes, p)
        assert p["axes"] == axes
        p = stability_derivatives(RECT, None, settings=FAST, detail_level="summary", axes=axes)
        assert p["status"] == "success", (axes, p)
        assert p["axes"] == axes
        p = batch_evaluate([RECT], None, "max_CL", FAST, axes=axes)
        assert p["status"] == "success", (axes, p)
        assert p["axes"] == axes
    bad = wing_analysis(RECT, None, FAST, "summary", axes="horizon")
    assert bad["status"] == "error" and bad["error"]["type"] == "invalid_input"


def test_error_type_table():
    """One case per row of the error-type table of decision 2."""
    assert error_type_of(InputError("bad key")) == "invalid_input"
    assert error_type_of(ValueError("bad value")) == "invalid_input"
    assert error_type_of(ValidityError("no ground effect")) == "invalid_method"
    assert error_type_of(GroundStrikeError("strike")) == "ground_strike"
    assert error_type_of(RuntimeError("boom")) == "internal"
    info = error_info(np.linalg.LinAlgError("Singular matrix"))
    assert info["type"] == "invalid_input"
    assert "singular" in info["message"].lower()
    assert "degenerate" in info["message"].lower() or "overlapping" in info["message"].lower()
    # A ValueError from the input checks (absurd position) is invalid_input.
    huge = wing_analysis({**RECT, "position_m": [1e300, 0.0, 0.0]}, None, FAST, "summary")
    assert huge["status"] == "error" and huge["error"]["type"] == "invalid_input"
    fourier = ground_effect(RECT, [1.0], settings={"solver": "fourier", "n_panels": 8})
    assert fourier["status"] == "error" and fourier["error"]["type"] == "invalid_method"
    strike = ground_effect(RECT, [0.01], settings=FAST)
    assert strike["status"] == "error" and strike["error"]["type"] == "ground_strike"


def test_batch_reports_reduced_n_chord(monkeypatch):
    """batch_evaluate reports the n_chord used per case, with a note when reduced."""
    from ventorum.agent import batch_evaluate, tools

    monkeypatch.setattr(tools, "MAX_TOTAL_PANELS", 100)
    p = batch_evaluate([RECT, {**RECT, "span_m": 8.0}], {"alpha_deg": 2.0, "h_m": 0.12},
                       "max_CL", {"n_panels": 12})
    assert p["status"] == "success", p
    assert p["axes"] == "body"
    assert len(p["rankings"]) == 2
    for row in p["rankings"]:
        assert row["n_chord"] == 4
        assert "reduced" in row["mesh_note"]
    # Without a reduction the n_chord is still reported and there is no note.
    p = batch_evaluate([RECT], {"alpha_deg": 4.0}, "max_CL", FAST)
    assert p["status"] == "success", p
    assert p["rankings"][0]["n_chord"] == 2 or isinstance(p["rankings"][0]["n_chord"], int)
    assert "mesh_note" not in p["rankings"][0]


def test_ground_polar_sweep_uses_one_mesh():
    """A ground-effect polar sweep uses the mesh of the first angle for all angles."""
    from ventorum.agent import polar_sweep

    p = polar_sweep(RECT, 2.0, 8.0, 2.0, {"h_m": 0.5}, {"n_panels": 12}, "standard")
    assert p["status"] == "success", p
    rows = [r for r in p["polar_table"] if r["status"] == "ok"]
    assert len(rows) == 4
    assert {r["n_chord"] for r in rows} == {rows[0]["n_chord"]}
    assert p["settings_used"]["n_chord_used"] == rows[0]["n_chord"]


def test_mcp_survives_invalid_utf8():
    """The MCP server answers invalid bytes with a parse error and keeps running."""
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = os.getcwd() + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else "")
    env["MPLBACKEND"] = "Agg"
    proc = subprocess.Popen(
        [sys.executable, "-m", "ventorum.agent", "--mcp"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        env=env, cwd=os.getcwd(),
    )
    try:
        valid = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}).encode() + b"\n"
        out, _ = proc.communicate(b"\xff\xfe not utf-8\n" + valid, timeout=120)
    finally:
        if proc.poll() is None:
            proc.kill()
    lines = [ln for ln in out.decode().splitlines() if ln.strip()]
    assert len(lines) == 2, lines
    first, second = (json.loads(ln) for ln in lines)
    assert first["id"] is None and first["error"]["code"] == -32700
    assert second["id"] == 1 and second["result"] == {}


def test_wind_axes_follow_the_flight_velocity_of_the_solver():
    """Review of T-0009: the wind x axis is the flight velocity of the solver's own sideslip sign.

    The free stream of the solver is in the geometry axes (x aft, y right,
    z up); the moment triple is in the standard body axes (x forward,
    y right, z down).
    """
    import numpy as np

    from ventorum.aero.system import freestream_direction
    from ventorum.core.axes import rotation_body_to_stability, rotation_body_to_wind

    for a_deg, b_deg in ((7.0, 11.0), (-4.0, -6.0), (15.0, 0.0)):
        a, b = np.radians(a_deg), np.radians(b_deg)
        v_geom = -freestream_direction(a, b)                    # flight velocity, geometry axes
        v_body = np.array([-v_geom[0], v_geom[1], -v_geom[2]])  # standard body axes
        np.testing.assert_allclose(rotation_body_to_wind(a, b) @ v_body, [1.0, 0.0, 0.0], atol=1e-14)
        v_stab = rotation_body_to_stability(a) @ v_body
        assert abs(v_stab[2]) < 1e-14                           # in the stability x-y plane
