# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests of the ventorum_derivative_matrix and updated ventorum_error_bars agent tools."""

from __future__ import annotations

import json
import math
import warnings

import numpy as np
import pytest

import ventorum as vt
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

    # Diagnostics contain static margin and neutral point, with the unit in the key name
    diag = res["diagnostics"]
    assert "static_margin_fraction" in diag
    assert "neutral_point_x_m" in diag
    assert "static_margin" not in diag and "neutral_point_x" not in diag
    # Control diagnostics appear once, under 'controls', with the unit in the key name
    assert set(diag["controls"]) == {"flap", "ail"}
    for name in ("flap", "ail"):
        assert name not in diag
        assert set(diag["controls"][name]) == {"CY_farfield_delta_per_rad", "Cn_trefftz_delta_per_rad", "warnings"}


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
    diag = res["diagnostics"]
    assert "irodov_margin" in diag
    assert set(diag["irodov"]) == {"x_alpha_m", "x_h_m", "irodov_margin", "verdict"}


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
    unc = {"alpha_deg": 0.2, "V_inf_m_s": 0.5}
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


def _identity(value, *args, **kwargs):
    return None if value is None else float(value)


def _no_rounding(monkeypatch):
    """Replace the rounding of the agent output by the identity."""
    import ventorum.agent.response as response
    import ventorum.agent.tools as tools

    monkeypatch.setattr(tools, "rnd", _identity)
    monkeypatch.setattr(response, "rnd", _identity)


def test_deflection_in_degrees_reaches_the_solver(monkeypatch):
    """A control deflection given in degrees reaches the solver as the same deflection in rad."""
    from ventorum.agent.schemas import build_aircraft_from_spec, make_flight_condition, parse_condition

    _no_rounding(monkeypatch)
    spec = {"span_m": 8.0, "chord_m": 1.0,
            "controls": [{"name": "flap", "eta_start": 0.0, "eta_end": 0.5, "hinge_x_c": 0.75,
                          "deflection_deg": 8.0}]}
    fc_spec = {"V_inf_m_s": 25.0, "alpha_deg": 3.0}
    res = agent_derivative_matrix(wing=spec, flight_condition=fc_spec, settings={"solver": "vlm", "n_panels": 10,
                                  "n_chord": 4}, groups=["control"], detail_level="full")
    assert res["status"] == "success", res

    # The same case in the Python API, with the deflection set in rad.
    spec0 = json.loads(json.dumps(spec))
    spec0["controls"][0]["deflection_deg"] = 0.0
    ac = build_aircraft_from_spec(wing=spec0)
    base = ac.moment_reference()
    ac.ref_point = np.array([float(base[0]), 0.0, float(base[2])])
    ac.compute_reference_values()
    cond = make_flight_condition(parse_condition(fc_spec))
    st = vt.SolverSettings(solver_type="vlm", n_panels=10, n_chord=4)

    def run(defl_rad):
        ac.surfaces[0].controls[0].deflection = defl_rad
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return vt.derivative_matrix(ac, cond, st, groups=("control",))

    ref = run(math.radians(8.0))
    zero = run(0.0)
    j = ref.columns.index("delta:flap")
    for i, row in enumerate(ref.rows):
        assert res["matrix"][row]["delta:flap_per_rad"] == pytest.approx(ref.matrix[i, j], rel=1e-12)
    assert res["base_result"]["metrics"]["CL"] == pytest.approx(ref.base.totals.CL, rel=1e-12)
    # The deflection changes the base lift: the comparison is sensitive to the deflection.
    assert abs(ref.base.totals.CL - zero.base.totals.CL) > 1e-3


def _check_mesh_cap(wing, fc, sett):
    """Run the tool with a capped mesh and compare with ventorum_stability_derivatives."""
    import ventorum.agent.tools as tools

    res = agent_derivative_matrix(wing=wing, flight_condition=fc, settings=sett, groups=["static"])
    assert res["status"] == "success", res
    assert "mesh_note" in res["settings_used"]
    strips = tools._check_mesh(tools.build_aircraft_from_spec(wing=wing), tools.parse_settings(sett))
    cap = tools.MAX_TOTAL_PANELS // strips
    assert f"n_chord={cap}" in res["settings_used"]["mesh_note"]

    stab = tools.stability_derivatives(wing=wing, flight_condition=fc, settings=sett)
    assert stab["status"] == "success", stab
    assert f"n_chord={cap}" in stab["settings_used"]["mesh_note"]
    assert res["matrix"]["CL"]["alpha_per_rad"] == pytest.approx(
        stab["stability_derivatives"]["CL_alpha_per_rad"], rel=1e-12)


@pytest.mark.slow
def test_mesh_cap_applies_to_all_cases(monkeypatch):
    """Regression test: the tool reduces the automatic n_chord to the panel limit for all cases.

    The case (span 8 m, 300 spanwise panels, h 0.12 m) needs more than
    the panel limit with the automatic chordwise count. The tool must use
    one reduced count, report it in the mesh note, and give the CL_alpha of
    ventorum_stability_derivatives.
    """
    _no_rounding(monkeypatch)
    _check_mesh_cap({"span_m": 8.0, "chord_m": 1.0}, {"V_inf_m_s": 30.0, "alpha_deg": 3.0, "h_m": 0.12},
                    {"n_panels": 300})


def test_mesh_cap_applies_to_all_cases_small_limit(monkeypatch):
    """The regression test of the mesh cap with a smaller panel limit, so that it runs fast."""
    import ventorum.agent.tools as tools

    _no_rounding(monkeypatch)
    monkeypatch.setattr(tools, "MAX_TOTAL_PANELS", 300)
    _check_mesh_cap({"span_m": 8.0, "chord_m": 1.0}, {"V_inf_m_s": 30.0, "alpha_deg": 3.0, "h_m": 0.12},
                    {"n_panels": 30})


def test_work_estimate_counts_the_real_solves(monkeypatch):
    """The work estimate is the number of solves of the source functions times N^2."""
    import ventorum.agent.tools as tools

    seen = []
    monkeypatch.setattr(tools, "_check_call_work", lambda estimate, hint: seen.append(int(estimate)))
    monkeypatch.setattr(tools.vt, "derivative_matrix", _stop)
    wing = dict(RECT_WING)
    sett = {"solver": "vlm", "n_panels": 10, "n_chord": 4}
    ac = tools.build_aircraft_from_spec(wing=wing)
    n_panels = tools._planned_panels(ac, tools.parse_settings(sett))
    assert n_panels % 4 == 0
    cases = [
        ({"alpha_deg": 2.0}, None, 5 + (1 + 2 * 2) + 7),
        ({"alpha_deg": 2.0, "h_m": 1.0}, None, 17 + 2 * 2),
        ({"alpha_deg": 2.0}, ["control"], 1 + 2 * 2),
        ({"alpha_deg": 2.0, "h_m": 1.0}, ["height"], 9),
        ({"alpha_deg": 2.0}, ["rate"], 7),
    ]
    for fc, groups, n_solves in cases:
        seen.clear()
        res = agent_derivative_matrix(wing=wing, flight_condition=fc, settings=sett, groups=groups)
        assert res["status"] == "error" and "stop" in res["error"]["message"], res
        assert seen == [n_solves * n_panels ** 2], (fc, groups)


def _stop(*args, **kwargs):
    raise RuntimeError("stop after the work check")


def test_input_uncertainty_errors_in_user_units():
    """Input uncertainties use unit key names; an error gives the value in the unit of the input."""
    base = {"wing": {"span_m": 10.0, "chord_m": 1.0}, "flight_condition": {"alpha_deg": 3.0},
            "settings": {"n_panels": 16}}
    res = error_bars(**base, input_uncertainties={"alpha_deg": -0.5})
    assert_error(res, "invalid_input", "input_uncertainties.alpha_deg")
    assert "-0.5" in res["error"]["message"]
    res = error_bars(**base, input_uncertainties={"alpha": 0.5})
    assert_error(res, "invalid_input", "alpha_deg")
    res = error_bars(**base, input_uncertainties={"h_m": 0.01})
    assert_error(res, "invalid_input", "ground effect")
    res = error_bars(**base, input_uncertainties={"sections": {"nope": {"Cd0": 0.001}}})
    assert_error(res, "invalid_input", "nope")
    res = error_bars(**base, input_uncertainties={"sections": {"Wing": {"cd0": 0.001}}})
    assert_error(res, "invalid_input", "Cd0")
    res = error_bars(**base, input_uncertainties={})
    assert_error(res, "invalid_input", "no uncertain input")


def test_input_uncertainties_converted_to_rad(monkeypatch):
    """The agent angles in degrees reach input_error_bars in rad; the other values keep their units."""
    import ventorum.agent.tools as tools

    seen = {}

    def fake(*args, uncertainties, **kwargs):
        seen.update(uncertainties)
        raise RuntimeError("stop")

    monkeypatch.setattr(tools.vt, "input_error_bars", fake)
    res = error_bars(wing={"span_m": 10.0, "chord_m": 1.0}, flight_condition={"alpha_deg": 3.0, "h_m": 3.0},
               settings={"n_panels": 16},
               input_uncertainties={"alpha_deg": 0.5, "beta_deg": 1.0, "phi_deg": 2.0, "V_inf_m_s": 0.3,
                                    "rho_kg_m3": 0.01, "h_m": 0.02,
                                    "sections": {"Wing": {"a0_per_rad": 0.1, "alpha_L0_deg": 0.5,
                                                          "Cd0": 0.001, "Cm0": 0.002}}})
    assert res["status"] == "error" and "stop" in res["error"]["message"], res
    assert seen["alpha"] == pytest.approx(math.radians(0.5), rel=1e-15)
    assert seen["beta"] == pytest.approx(math.radians(1.0), rel=1e-15)
    assert seen["phi"] == pytest.approx(math.radians(2.0), rel=1e-15)
    assert seen["V_inf"] == 0.3 and seen["rho"] == 0.01 and seen["h"] == 0.02
    sec = seen["sections"]["Wing"]
    assert sec["a0"] == 0.1 and sec["Cd0"] == 0.001 and sec["Cm0"] == 0.002
    assert sec["alpha_L0"] == pytest.approx(math.radians(0.5), rel=1e-15)


def test_error_bars_budget_counts_input_solves(monkeypatch):
    """The work estimate of ventorum_error_bars adds the solves of the input layer."""
    import ventorum.agent.tools as tools

    seen = []
    monkeypatch.setattr(tools, "_check_call_work", lambda estimate, hint: seen.append(int(estimate)))
    monkeypatch.setattr(tools.vt, "numerical_error_bars", _stop)
    base = {"wing": {"span_m": 10.0, "chord_m": 1.0}, "flight_condition": {"alpha_deg": 3.0},
            "settings": {"solver": "vlm", "n_panels": 16, "n_chord": 2}}
    error_bars(**base)
    error_bars(**base, input_uncertainties={"alpha_deg": 0.1, "V_inf_m_s": 0.2})
    n_fine = tools._planned_panels(tools.build_aircraft_from_spec(wing=base["wing"]),
                                   tools.parse_settings(base["settings"]))
    assert seen[1] - seen[0] == (1 + 2 * 2) * n_fine ** 2


def _error_bars_before_change(wing, flight_condition, settings, axes="body"):
    """The ventorum_error_bars logic before input_uncertainties was added (reference)."""
    import ventorum.agent.tools as t
    from ventorum.core.error_bars import mesh_levels

    ac = t.build_aircraft_from_spec(wing=wing)
    c = t.parse_condition(flight_condition)
    sett = t.parse_settings(settings)
    ax = axes
    strips = t._check_mesh(ac, sett)
    levels = mesh_levels(int(sett.n_panels))
    fc = t.make_flight_condition(c)
    case_sett = t._case_settings(ac, sett, strips, fc)
    res = vt.numerical_error_bars(ac, condition=fc, settings=case_sett, axes=ax)
    device, precision = t.result_device_precision(res.results[0] if res.results else None)
    lines = []
    for name in ("CL", "CD", "CY", "Cl", "Cm", "Cn"):
        bar = res.bars.get(name)
        if bar is None:
            lines.append(f"{name}: no bar (the drag basis is not the same at the three levels)")
            continue
        state = bar.layers["numerical"].state if "numerical" in bar.layers else None
        lines.append(f"{name}={t.fmt(bar.value, '.6f')} [{t.fmt(bar.interval_low, '.6f')}, "
                     f"{t.fmt(bar.interval_high, '.6f')}] ({state})")
    summary = (
        f"[Ventorum RESULT] Numerical error bars of {ac.name} "
        f"(spanwise levels {levels[0]}/{levels[1]}/{levels[2]}):\n" + "\n".join(lines) +
        "\nOnly the numerical (spanwise discretisation) layer is included; "
        "the other layers are not implemented. "
        f"{t.device_summary_text(device, precision)}"
    )
    return t.json_safe({
        "status": "success",
        "executive_summary": summary,
        "error_bars": res.to_dict(),
        "levels": [int(n) for n in levels],
        "notes": list(res.notes),
        "status_of_bars": "numerical_only",
        "device": device,
        "precision": precision,
        "condition_used": t.condition_payload(c),
        "settings_used": {**t.settings_payload(sett), **t._mesh_note(case_sett is not sett, case_sett)},
        "axes": ax,
    })


@pytest.mark.parametrize("axes", ["body", "wind"])
def test_error_bars_without_input_uncertainties_unchanged(axes):
    """Without input_uncertainties the output equals the output of the logic before the change."""
    args = {"wing": {"span_m": 8.0, "root_chord_m": 1.2, "tip_chord_m": 0.8},
            "flight_condition": {"alpha_deg": 2.0, "beta_deg": 3.0, "V_inf_m_s": 30.0},
            "settings": {"n_panels": 16}}
    got = call_tool("ventorum_error_bars", {**args, "axes": axes})
    # The dispatcher adds the tool name and the run time to every payload.
    assert got.pop("tool") == "ventorum_error_bars"
    got.pop("execution_time_ms", None)
    expected = _error_bars_before_change(args["wing"], args["flight_condition"], args["settings"], axes)
    assert got == expected
