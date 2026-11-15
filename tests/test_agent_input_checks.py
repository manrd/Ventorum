# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Regression tests of agent input checks, error messages and the mesh report.

- ``settings_used.surface_mesh`` gives the spanwise strips of each surface of
  the solved lattice (``settings_used.n_panels`` is only the settings value).
- A symmetric surface with a dihedral near 90 deg (both halves on y = 0) is
  refused.
- A size-limit error of a section names the argument, not ``None``.
- Two surfaces with the same name are refused.
- The MCP server accepts a JSON number id with no fraction part.
"""

from __future__ import annotations

import json

import pytest

from ventorum.agent import call_tool
from ventorum.agent.mcp_server import handle_message
from ventorum.agent.schemas import (
    MIN_SYMMETRIC_DIHEDRAL_MARGIN_DEG,
    InputError,
    build_aircraft_from_spec,
)

WING = {"span_m": 10.0, "chord_m": 1.0}


def _ok(result):
    assert result["status"] == "success", result.get("error")
    return result


# ── Mesh of each surface ──────────────────────────────────────────────────────

def test_surface_mesh_reports_the_count_of_each_surface():
    """A surface n_panels shows in settings_used.surface_mesh; settings n_panels stays the settings value."""
    ac = {"surfaces": [
        {**WING, "name": "wing", "n_panels": 20},
        {"name": "fin", "semi_span_m": 1.5, "chord_m": 0.8, "symmetric": False, "dihedral_deg": 90.0,
         "position_m": [4.0, 0.0, 0.0], "n_panels": 7},
    ]}
    r = _ok(call_tool("ventorum_wing_analysis", {"aircraft": ac, "settings": {"n_panels": 30}}))
    used = r["settings_used"]
    assert used["n_panels"] == 30
    assert used["surface_mesh"] == [
        {"name": "wing", "n_panels": 20, "n_strips": 40},
        {"name": "fin", "n_panels": 7, "n_strips": 7},
    ]


def test_surface_mesh_uses_the_settings_count_without_a_surface_count():
    r = _ok(call_tool("ventorum_wing_analysis", {"wing": WING, "settings": {"n_panels": 12}}))
    assert r["settings_used"]["surface_mesh"] == [{"name": "Wing", "n_panels": 12, "n_strips": 24}]


def test_surface_mesh_of_a_mirror_copy():
    wing = {"semi_span_m": 3.0, "chord_m": 1.0, "symmetric": False, "mirror": True,
            "position_m": [0.0, 1.0, 0.0], "n_panels": 6}
    r = _ok(call_tool("ventorum_wing_analysis", {"wing": wing}))
    assert r["settings_used"]["surface_mesh"] == [
        {"name": "Wing", "n_panels": 6, "n_strips": 6},
        {"name": "Wing (mirror)", "n_panels": 6, "n_strips": 6},
    ]


@pytest.mark.gpu
def test_surface_mesh_of_a_gpu_solve():
    """A GPU solve reports the same surface_mesh as the CPU."""
    from ventorum import gpu

    if not gpu.available():
        pytest.skip(f"The GPU pipelines cannot run: {gpu.unavailable_reason()}")
    old = gpu.get_device()
    try:
        gpu.set_device("gpu")
        r = _ok(call_tool("ventorum_wing_analysis", {"wing": {**WING, "n_panels": 40}}))
    finally:
        gpu.set_device(old)
    assert r["settings_used"]["device"] == "gpu"
    assert r["settings_used"]["surface_mesh"] == [{"name": "Wing", "n_panels": 40, "n_strips": 80}]


def test_surface_mesh_is_absent_without_a_lattice():
    """The Fourier solver has no vortex lattice, so the payload has no surface_mesh."""
    r = _ok(call_tool("ventorum_wing_analysis", {"wing": WING, "settings": {"solver": "fourier"}}))
    assert "surface_mesh" not in r["settings_used"]


# ── Dihedral of a symmetric surface ───────────────────────────────────────────

@pytest.mark.parametrize("dihedral", [90.0, -90.0, 89.9, -89.5])
def test_symmetric_surface_near_90_deg_dihedral_is_refused(dihedral):
    with pytest.raises(InputError, match="symmetric.*false"):
        build_aircraft_from_spec(wing={**WING, "dihedral_deg": dihedral})
    r = call_tool("ventorum_wing_analysis", {"wing": {**WING, "dihedral_deg": dihedral}})
    assert r["status"] == "error" and r["error"]["type"] == "invalid_input"
    assert "y = 0" in r["error"]["message"]


def test_dihedral_limit_of_a_symmetric_surface():
    """The limit is 90 deg minus the margin; the limit itself and a fin at 90 deg are accepted."""
    limit = 90.0 - MIN_SYMMETRIC_DIHEDRAL_MARGIN_DEG
    assert MIN_SYMMETRIC_DIHEDRAL_MARGIN_DEG == 1.0
    build_aircraft_from_spec(wing={**WING, "dihedral_deg": limit})
    build_aircraft_from_spec(wing={**WING, "dihedral_deg": -limit})
    with pytest.raises(InputError):
        build_aircraft_from_spec(wing={**WING, "dihedral_deg": limit + 1e-6})
    build_aircraft_from_spec(wing={"semi_span_m": 2.0, "chord_m": 1.0, "symmetric": False, "dihedral_deg": 90.0})


# ── Error message names the argument ──────────────────────────────────────────

@pytest.mark.parametrize("key, where", [("wing", "wing"), ("aircraft", "aircraft")])
def test_size_limit_message_names_the_argument(key, where):
    surf = {"span_m": 10.0, "sections": [{"y_frac": 0.0, "chord_m": 1.0},
                                         {"y_frac": 1.0, "chord_m": 1.0, "x_le_m": 1e6}]}
    spec = surf if key == "wing" else {"surfaces": [surf]}
    with pytest.raises(InputError) as info:
        build_aircraft_from_spec(**{key: spec})
    msg = str(info.value)
    assert msg.startswith(f"{where}: "), msg
    assert "None" not in msg


# ── Unique surface names ──────────────────────────────────────────────────────

def test_duplicate_surface_names_are_refused():
    ac = {"surfaces": [{**WING, "name": "a"}, {**WING, "name": "a", "position_m": [5.0, 0.0, 0.0]}]}
    with pytest.raises(InputError, match="unique"):
        build_aircraft_from_spec(aircraft=ac)
    r = call_tool("ventorum_wing_analysis", {"aircraft": ac})
    assert r["status"] == "error" and r["error"]["type"] == "invalid_input"
    assert "'a'" in r["error"]["message"]


def test_default_and_mirror_names_are_unique():
    ac = {"surfaces": [WING, {"semi_span_m": 1.0, "chord_m": 0.5, "symmetric": False, "mirror": True,
                              "position_m": [4.0, 1.0, 0.0]}]}
    built = build_aircraft_from_spec(aircraft=ac)
    assert [s.name for s in built.surfaces] == ["Wing", "Surface_2", "Surface_2 (mirror)"]


# ── MCP request id ────────────────────────────────────────────────────────────

def _rpc_text(line):
    return handle_message(line)


def test_mcp_accepts_an_integral_float_id_and_echoes_it():
    out = _rpc_text('{"jsonrpc": "2.0", "id": 5.0, "method": "ping"}')
    reply = json.loads(out)
    assert "error" not in reply and reply["result"] == {}
    assert isinstance(reply["id"], float) and reply["id"] == 5.0
    assert '"id": 5.0' in out


@pytest.mark.parametrize("bad_id", ["5.5", "true", "false", "null", "1e400"])
def test_mcp_refuses_other_ids(bad_id):
    reply = json.loads(_rpc_text('{"jsonrpc": "2.0", "id": ' + bad_id + ', "method": "ping"}'))
    assert reply["id"] is None and reply["error"]["code"] == -32600


def test_mcp_keeps_integer_and_string_ids():
    for raw, value in (("7", 7), ('"abc"', "abc")):
        reply = json.loads(_rpc_text('{"jsonrpc": "2.0", "id": ' + raw + ', "method": "ping"}'))
        assert reply["id"] == value and "error" not in reply
