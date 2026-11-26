# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""The unsteady agent tools keep the lateral moment reference coordinate.

With ``aircraft.ref_point_m = [x, y, z]`` and ``y != 0`` the tools must use
``[x_cg_m, y, z]``, as the library does. They used ``[x_cg_m, 0, z]``, so the
rolling moment of the offset reference and the free-flight roll were lost.
"""

from __future__ import annotations

import math

import pytest

import ventorum as vt
from ventorum.agent import call_tool

_SETT = {"n_panels": 4, "n_chord": 4}
_FC = {"V_inf_m_s": 25.0, "alpha_deg": 4.0}
_CASES = {
    "ventorum_time_response": {"n_steps": 2},
    "ventorum_dynamic_derivatives": {"k": [0.0]},
    "ventorum_indicial_response": {"kind": "alpha", "s_end": 2.0},
    "ventorum_free_flight": {
        "n_steps": 3, "mass_kg": 100.0, "inertia_kg_m2": {"xx": 20.0, "yy": 30.0, "zz": 40.0}, "dofs": ["roll"],
    },
}


def _run(tool: str, y_ref: float) -> dict:
    ac = {"surfaces": [{"name": "wing", "span_m": 6.0, "chord_m": 1.0}], "ref_point_m": [0.25, y_ref, 0.0]}
    out = call_tool(tool, {"aircraft": ac, "flight_condition": _FC, "settings": _SETT, **_CASES[tool]})
    assert out["status"] == "success", out.get("error")
    return out


def _roll_value(tool: str, out: dict) -> float:
    if tool == "ventorum_dynamic_derivatives":
        return float(out["mean_state"]["Cl"])
    return float(out["Cl"][-1])


@pytest.mark.parametrize("tool", list(_CASES))
def test_unsteady_tools_keep_y_ref(tool):
    """An offset y_ref gives the rolling moment of the library."""
    centred = _run(tool, 0.0)
    offset = _run(tool, 0.5)
    assert abs(_roll_value(tool, centred)) < 1e-9
    assert abs(_roll_value(tool, offset)) > 1e-3
    if "ref_point_m" in offset:
        assert offset["ref_point_m"][1] == pytest.approx(0.5)


def test_free_flight_tool_rolls_with_an_offset_cg():
    """The free-flight tool rolls like the library with cg y = 0.5 m."""
    out = _run("ventorum_free_flight", 0.5)
    wing = vt.LiftingSurface(
        name="wing", semi_span=3.0, n_panels=4,
        sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=1.0)],
    )
    ac = vt.Aircraft(surfaces=[wing], ref_point=[0.25, 0.5, 0.0])
    lib = vt.free_flight(
        ac, vt.FlightCondition(V_inf=25.0, alpha=math.radians(4.0)), vt.SolverSettings(n_panels=4, n_chord=4),
        mass=100.0, inertia=[[20.0, 0.0, 0.0], [0.0, 30.0, 0.0], [0.0, 0.0, 40.0]], dofs=("roll",), n_steps=3,
    )
    assert abs(float(lib.roll[-1])) > 1e-4
    assert float(out["motion"]["roll_rad"][-1]) == pytest.approx(float(lib.roll[-1]), rel=1e-9)
