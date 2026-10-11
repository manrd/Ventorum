# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests of the ventorum_derivative_matrix and updated ventorum_error_bars agent tools."""

from __future__ import annotations

import json

from ventorum.agent import call_tool, derivative_matrix as agent_derivative_matrix, error_bars


def strict_json(payload):
    """Ensure the payload is JSON-compliant with no non-finite numbers."""
    return json.loads(json.dumps(payload, allow_nan=False))


def assert_error(payload, kind, text=None):
    assert payload["status"] == "error", payload
    assert payload["error"]["type"] == kind, payload
    if text is not None:
        assert text in payload["error"]["message"], payload


RECT_WING = {
    "span_m": 8.0,
    "chord_m": 1.0,
    "controls": [
        {"name": "flap", "eta_start": 0.0, "eta_end": 0.5, "hinge_x_c": 0.75, "deflection_deg": 2.0},
        {"name": "ail", "eta_start": 0.6, "eta_end": 1.0, "hinge_x_c": 0.75, "deflection_deg": 0.0, "symmetric": False},
    ],
}

AIRCRAFT_SPEC = {
    "surfaces": [
        {"name": "wing", "span_m": 10.0, "root_chord_m": 1.5, "tip_chord_m": 1.0,
         "controls": [{"name": "elevon", "eta_start": 0.2, "eta_end": 0.8, "hinge_x_c": 0.75, "deflection_deg": 1.5}]},
    ],
    "ref_point_m": [0.3, 0.0, 0.0],
}


def test_derivative_matrix_with_wing_and_degrees_input():
    """Tool accepts wing spec with deflections in degrees and returns derivatives per rad."""
    res = agent_derivative_matrix(
        wing=dict(RECT_WING),
        flight_condition={"V_inf_m_s": 25.0, "alpha_deg": 3.0, "beta_deg": 1.0},
        settings={"solver": "linear", "n_panels": 10},
        axes="body",
        detail_level="standard",
    )
    strict_json(res)
    assert res["status"] == "success", res
    assert "executive_summary" in res
    assert res["axes"] == "body"

    matrix = res["matrix"]
    assert isinstance(matrix, dict)
    assert set(matrix.keys()) == {"CL", "CD", "CY", "Cl", "Cm", "Cn"}

    # Columns include static (alpha, beta), controls (delta:ail, delta:flap), and rates (p_hat, q_hat, r_hat)
    cols = res["columns"]
    assert "alpha" in cols and "beta" in cols
    assert "delta:flap" in cols and "delta:ail" in cols
    assert "p_hat" in cols and "q_hat" in cols and "r_hat" in cols

    # Rows contain the per_rad keyed columns
    cl_row = matrix["CL"]
    assert "alpha_per_rad" in cl_row
    assert "delta:flap_per_rad" in cl_row
    assert "p_hat" in cl_row

    # Units check
    assert res["units"]["alpha"] == "1/rad"
    assert res["units"]["delta:flap"] == "1/rad"
    assert res["units"]["p_hat"] == "-"

    # Diagnostics contain static margin and neutral point
    diag = res["diagnostics"]
    assert "static_margin" in diag
    assert "neutral_point_x" in diag


def test_derivative_matrix_with_aircraft_and_ground_effect():
    """Tool accepts aircraft spec and condition with h_m for ground effect."""
    res = agent_derivative_matrix(
        aircraft=dict(AIRCRAFT_SPEC),
        flight_condition={"V_inf_m_s": 30.0, "alpha_deg": 2.0, "h_m": 2.5},
        settings={"solver": "linear", "n_panels": 10},
        detail_level="standard",
    )
    strict_json(res)
    assert res["status"] == "success", res
    cols = res["columns"]
    assert "phi" in cols
    assert "h_over_c" in cols
    assert "delta:elevon" in cols
    assert "irodov_margin" in res["diagnostics"]


def test_derivative_matrix_detail_levels():
    """detail_level controls the payload contents: summary vs standard vs full."""
    # Summary level
    res_sum = agent_derivative_matrix(
        wing=dict(RECT_WING),
        flight_condition={"alpha_deg": 2.0},
        settings={"n_panels": 8},
        detail_level="summary",
    )
    assert res_sum["status"] == "success"
    assert "matrix" in res_sum
    assert "base_result" not in res_sum

    # Full level includes base_result
    res_full = agent_derivative_matrix(
        wing=dict(RECT_WING),
        flight_condition={"alpha_deg": 2.0},
        settings={"n_panels": 8},
        detail_level="full",
    )
    assert res_full["status"] == "success"
    assert "base_result" in res_full
    assert "metrics" in res_full["base_result"]


def test_derivative_matrix_work_budget(monkeypatch):
    """Refuse calculations that exceed the work budget without running a solve."""
    import ventorum.agent.tools as tools

    monkeypatch.setattr(tools, "MAX_CALL_WORK", 1000)
    res = agent_derivative_matrix(
        wing={"span_m": 10.0, "chord_m": 1.0},
        settings={"n_panels": 16},
    )
    assert_error(res, "invalid_input", "budget")


def test_error_bars_with_and_without_input_uncertainties():
    """Without input_uncertainties, output is unchanged. With them, combined bars are computed."""
    base_wing = {"span_m": 10.0, "chord_m": 1.0}
    fc = {"alpha_deg": 3.0, "V_inf_m_s": 30.0}
    sett = {"n_panels": 16}

    # 1. Without input uncertainties: numerical_only
    res_num = error_bars(wing=dict(base_wing), flight_condition=fc, settings=sett)
    strict_json(res_num)
    assert res_num["status"] == "success"
    assert res_num["status_of_bars"] == "numerical_only"
    bars_num = res_num["error_bars"]["bars"]
    assert "CL" in bars_num

    # 2. With input uncertainties: numerical_and_input (angles in degrees in the agent input)
    unc = {"alpha": 0.2, "V_inf": 0.5}
    res_comb = error_bars(wing=dict(base_wing), flight_condition=fc, settings=sett, input_uncertainties=unc)
    strict_json(res_comb)
    assert res_comb["status"] == "success"
    assert res_comb["status_of_bars"] == "numerical_and_input"
    bars_comb = res_comb["error_bars"]["bars"]

    # Combined bar should have input layer present
    for coeff in ("CL", "CD", "CY", "Cl", "Cm", "Cn"):
        assert "input" in bars_comb[coeff]["layers"]
        # The interval width of the combined bar should be at least as large as the numerical bar
        if bars_num[coeff]["interval_high"] is not None and bars_comb[coeff]["interval_high"] is not None:
            width_num = bars_num[coeff]["interval_high"] - bars_num[coeff]["interval_low"]
            width_comb = bars_comb[coeff]["interval_high"] - bars_comb[coeff]["interval_low"]
            assert width_comb >= width_num - 1e-12


def test_call_tool_dispatcher():
    """Test calling the new tool through the call_tool entry point."""
    payload = {
        "wing": dict(RECT_WING),
        "flight_condition": {"alpha_deg": 2.0},
        "settings": {"n_panels": 8},
        "detail_level": "standard",
    }
    res = call_tool("ventorum_derivative_matrix", payload)
    strict_json(res)
    assert res["status"] == "success", res
    assert "executive_summary" in res
