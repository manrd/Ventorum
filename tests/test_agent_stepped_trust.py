# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests of ``stepped_trust_min`` in the agent derivative tools.

Both ``ventorum_stability_derivatives`` and ``ventorum_derivative_matrix``
include ``stepped_trust_min`` in their JSON payload (None / null when no
stepped case has a lower trust than the base case) and include the stepped
trust note in ``notes`` when a stepped case has a lower trust.
"""

from __future__ import annotations

import json

import pytest

from ventorum.agent import call_tool, derivative_matrix as agent_derivative_matrix, stability_derivatives


def _strict_json(payload: dict) -> dict:
    """Verify strict JSON serialisation and return the decoded object."""
    return json.loads(json.dumps(payload, allow_nan=False))


# Base clearance 0.102 c (above the proximity limit 0.1); stepped height or bank cases go below it.
_H = 0.102

_WING_PIVOT_TE = {
    "surfaces": [{"name": "w", "span_m": 6.0, "chord_m": 1.0, "n_panels": 12}],
    "ref_point_m": [1.0, 0.0, 0.0],
}

_WING_WITH_FLAP = {
    "surfaces": [{
        "name": "w",
        "span_m": 6.0,
        "chord_m": 1.0,
        "n_panels": 12,
        "controls": [{"name": "flap", "eta_start": 0.0, "eta_end": 1.0, "hinge_x_c": 0.7, "deflection_deg": 0.0}],
    }],
    "ref_point_m": [1.0, 0.0, 0.0],
}

_SETTINGS = {"solver": "vlm", "n_panels": 12, "n_chord": 8}


@pytest.mark.parametrize("axes", ["body", "all"])
@pytest.mark.parametrize("detail_level", ["summary", "standard", "full"])
def test_derivative_matrix_reports_stepped_trust_min_and_note(axes, detail_level):
    """ventorum_derivative_matrix reports stepped_trust_min and its note when a stepped case has lower trust."""
    raw = call_tool("ventorum_derivative_matrix", {
        "aircraft": _WING_PIVOT_TE,
        "flight_condition": {"V_inf_m_s": 50.0, "alpha_deg": 4.0, "h_m": _H},
        "settings": _SETTINGS,
        "groups": ["static", "height"],
        "axes": axes,
        "detail_level": detail_level,
    })
    res = _strict_json(raw)
    assert res["status"] == "success", res
    info = res["stepped_trust_min"]
    assert isinstance(info, dict)
    assert set(info.keys()) == {"case", "rating", "score", "base_rating", "base_score", "group"}
    assert info["group"] == "height"
    assert info["score"] < info["base_score"]
    assert any("lower trust" in n and info["case"] in n for n in res["notes"])


def test_derivative_matrix_control_group_reports_stepped_trust_min():
    """A trailing-edge flap step below h_min/c = 0.1 appears in stepped_trust_min."""
    raw = agent_derivative_matrix(
        aircraft=_WING_WITH_FLAP,
        flight_condition={"V_inf_m_s": 50.0, "alpha_deg": 4.0, "h_m": _H},
        settings={"solver": "vlm", "n_panels": 12, "n_chord": 10},
        groups=["control"],
    )
    res = _strict_json(raw)
    assert res["status"] == "success", res
    info = res["stepped_trust_min"]
    assert isinstance(info, dict)
    assert info["group"] == "control"
    assert info["case"] == "flap +"
    assert info["score"] < info["base_score"]
    assert any("lower trust" in n and "'flap +'" in n for n in res["notes"])


def test_derivative_matrix_stepped_trust_min_is_none_when_no_case_is_lower():
    """ventorum_derivative_matrix reports stepped_trust_min as null when no stepped case is lower."""
    free = _strict_json(call_tool("ventorum_derivative_matrix", {
        "aircraft": _WING_PIVOT_TE,
        "flight_condition": {"V_inf_m_s": 50.0, "alpha_deg": 4.0},
        "settings": _SETTINGS,
        "groups": ["static", "rate"],
    }))
    assert free["status"] == "success", free
    assert "stepped_trust_min" in free
    assert free["stepped_trust_min"] is None
    assert not any("lower trust" in n for n in free["notes"])

    static_ge = _strict_json(agent_derivative_matrix(
        aircraft=_WING_PIVOT_TE,
        flight_condition={"V_inf_m_s": 50.0, "alpha_deg": 4.0, "h_m": _H},
        settings=_SETTINGS,
        groups=["static"],
    ))
    assert static_ge["status"] == "success", static_ge
    assert static_ge["stepped_trust_min"] is None
    assert not any("lower trust" in n for n in static_ge["notes"])


@pytest.mark.parametrize("detail_level", ["summary", "standard", "full"])
def test_stability_derivatives_reports_stepped_trust_min_and_note(detail_level):
    """ventorum_stability_derivatives reports stepped_trust_min and its note when a stepped case has lower trust."""
    # Pivot at TE in ground effect: the Irodov height_derivatives stepped cases cross h_min/c = 0.1.
    raw = call_tool("ventorum_stability_derivatives", {
        "aircraft": _WING_PIVOT_TE,
        "flight_condition": {"V_inf_m_s": 50.0, "alpha_deg": 4.0, "h_m": _H},
        "x_cg_m": 1.0,
        "settings": _SETTINGS,
        "detail_level": detail_level,
    })
    res = _strict_json(raw)
    assert res["status"] == "success", res
    info = res["stepped_trust_min"]
    assert isinstance(info, dict)
    assert set(info.keys()) == {"case", "rating", "score", "base_rating", "base_score"}
    assert info["score"] < info["base_score"]
    assert any("lower trust" in n and info["case"] in n for n in res["notes"])


def test_stability_derivatives_alpha_step_reports_stepped_trust_min():
    """An alpha + step that crosses h_min/c = 0.1 (pivot at LE) appears in stepped_trust_min."""
    # At alpha = 4 deg, TE is sin(4 deg) = 0.0698 m below LE; at h = 0.172 m base h_min/c = 0.1022 > 0.1,
    # and alpha + 0.5 deg lowers TE to h_min/c = 0.0935 < 0.1.
    raw = stability_derivatives(
        wing={"span_m": 6.0, "chord_m": 1.0},
        flight_condition={"V_inf_m_s": 50.0, "alpha_deg": 4.0, "h_m": 0.172},
        x_cg_m=0.0,
        settings=_SETTINGS,
    )
    res = _strict_json(raw)
    assert res["status"] == "success", res
    info = res["stepped_trust_min"]
    assert isinstance(info, dict)
    assert set(info.keys()) == {"case", "rating", "score", "base_rating", "base_score"}
    assert info["score"] < info["base_score"]
    assert any("lower trust" in n and info["case"] in n for n in res["notes"])


def test_stability_derivatives_stepped_trust_min_is_none_in_free_air():
    """ventorum_stability_derivatives reports stepped_trust_min as null when no stepped case is lower."""
    raw = call_tool("ventorum_stability_derivatives", {
        "aircraft": _WING_PIVOT_TE,
        "flight_condition": {"V_inf_m_s": 50.0, "alpha_deg": 4.0},
        "x_cg_m": 1.0,
        "settings": _SETTINGS,
    })
    res = _strict_json(raw)
    assert res["status"] == "success", res
    assert "stepped_trust_min" in res
    assert res["stepped_trust_min"] is None
    assert not any("lower trust" in n for n in res["notes"])
