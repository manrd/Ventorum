# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests of the full aerodynamic derivative matrix."""

from __future__ import annotations

import json
import math
import warnings

import numpy as np
import pytest

import ventorum as vt
from ventorum.agent.tools import build_aircraft_from_spec, stability_derivatives as agent_stability_derivatives
from ventorum.core.axes import transform_moments
from ventorum.solvers.control_derivatives import control_derivatives
from ventorum.solvers.derivative_matrix import derivative_matrix
from ventorum.solvers.height_derivatives import height_derivatives
from ventorum.solvers.rate_derivatives import rate_derivatives
from ventorum.solvers.static_derivatives import static_derivatives


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


def _wing_two_controls(semi_span: float = 4.0) -> vt.LiftingSurface:
    sec = [vt.WingSection(0.0, 1.0), vt.WingSection(1.0, 1.0)]
    ctrls = [
        vt.ControlSurface("flap", 0.0, 0.5, 0.75, np.radians(1.0)),
        vt.ControlSurface("ail", 0.6, 1.0, 0.75, np.radians(0.0), symmetric=False),
    ]
    return vt.LiftingSurface(name="main_wing", semi_span=semi_span, sections=sec, controls=ctrls)


def _aircraft_two_controls(semi_span: float = 4.0) -> vt.Aircraft:
    w = _wing_two_controls(semi_span=semi_span)
    b = 2.0 * semi_span
    return vt.Aircraft(
        surfaces=[w],
        S_ref=b * 1.0,
        b_ref=b,
        c_ref=1.0,
        ref_point=np.array([0.25, 0.0, 0.0]),
    )


def test_columns_equal_existing_functions():
    """Every column of the matrix equals the values of the existing solver functions."""
    ac = _aircraft_two_controls(semi_span=3.0)
    cond_free = vt.FlightCondition(V_inf=30.0, alpha=np.radians(3.0), beta=np.radians(2.0))
    st = vt.SolverSettings(solver_type="linear", n_panels=12)

    # 1. Free air matrix
    mat_res = _quiet(derivative_matrix, ac, cond_free, st, groups=("static", "control", "rate"))
    col_idx = {name: i for i, name in enumerate(mat_res.columns)}

    # Static columns vs static_derivatives
    stat = _quiet(static_derivatives, ac, cond_free, st)
    for row_i, coeff in enumerate(mat_res.rows):
        assert mat_res.matrix[row_i, col_idx["alpha"]] == pytest.approx(stat.derivatives[f"{coeff}_alpha"], rel=1e-12)
        assert mat_res.matrix[row_i, col_idx["beta"]] == pytest.approx(stat.derivatives[f"{coeff}_beta"], rel=1e-12)

    # Control columns vs control_derivatives
    ctrl = _quiet(control_derivatives, ac, cond_free, st, hinge_alpha=False)
    for c_name in ("flap", "ail"):
        col_name = f"delta:{c_name}"
        c_res = ctrl.derivatives[c_name]
        for row_i, coeff in enumerate(mat_res.rows):
            assert mat_res.matrix[row_i, col_idx[col_name]] == pytest.approx(getattr(c_res, coeff), rel=1e-12)

    # Rate columns vs rate_derivatives
    rate = _quiet(rate_derivatives, ac, cond_free, st)
    for col_j, r_name in enumerate(("p_hat", "q_hat", "r_hat")):
        for row_i in range(len(mat_res.rows)):
            assert mat_res.matrix[row_i, col_idx[r_name]] == pytest.approx(rate.matrix[row_i, col_j], rel=1e-12)

    # 2. Ground effect matrix vs height_derivatives
    cond_ground = vt.FlightCondition(V_inf=30.0, alpha=np.radians(3.0), beta=np.radians(2.0), h=2.5)
    mat_ground = _quiet(derivative_matrix, ac, cond_ground, st, groups=("static", "height", "control"))
    col_g_idx = {name: i for i, name in enumerate(mat_ground.columns)}
    h_res = _quiet(height_derivatives, ac, cond_ground, st)

    for col_j, v_name in enumerate(("alpha", "beta", "phi", "h_over_c")):
        for row_i in range(len(mat_ground.rows)):
            assert mat_ground.matrix[row_i, col_g_idx[v_name]] == pytest.approx(h_res.matrix[row_i, col_j], rel=1e-12)


def test_static_equals_agent_tool():
    """static_derivatives values equal the numbers of ventorum_stability_derivatives."""
    spec = {"span_m": 6.0, "root_chord_m": 1.0, "tip_chord_m": 1.0}
    ac = build_aircraft_from_spec(wing=spec)
    cond = vt.FlightCondition(V_inf=25.0, alpha=math.radians(2.0), beta=math.radians(1.0))
    st = vt.SolverSettings(solver_type="linear", n_panels=10)

    res_py = _quiet(static_derivatives, ac, cond, st)
    res_agent = _quiet(
        agent_stability_derivatives,
        wing=spec,
        flight_condition={"V_inf_m_s": 25.0, "alpha_deg": 2.0, "beta_deg": 1.0},
        settings={"solver": "linear", "n_panels": 10},
        detail_level="full",
    )
    m = res_agent["stability_derivatives"]
    assert round(res_py.derivatives["CL_alpha"], 5) == m["CL_alpha_per_rad"]
    assert round(res_py.derivatives["Cm_alpha"], 5) == m["Cm_alpha_per_rad"]
    assert round(res_py.derivatives["CY_beta"], 6) == m["CY_beta_per_rad"]
    assert round(res_py.derivatives["Cl_beta"], 6) == m["Cl_beta_per_rad"]
    assert round(res_py.derivatives["Cn_beta"], 6) == m["Cn_beta_per_rad"]
    assert round(res_py.static_margin, 5) == m["static_margin_fraction"]
    assert round(res_py.neutral_point_x, 5) == m["neutral_point_x_m"]


def test_layout_and_units():
    """Rows, column order, units, and methods follow the specifications in free air and ground effect."""
    ac = _aircraft_two_controls(semi_span=4.0)
    st = vt.SolverSettings(solver_type="linear", n_panels=8)

    # Free air
    cond_free = vt.FlightCondition(V_inf=30.0, alpha=np.radians(2.0))
    res_free = _quiet(derivative_matrix, ac, cond_free, st)

    assert res_free.rows == ("CL", "CD", "CY", "Cl", "Cm", "Cn")
    # Control names are sorted: ail before flap
    expected_cols_free = ("alpha", "beta", "delta:ail", "delta:flap", "p_hat", "q_hat", "r_hat")
    assert res_free.columns == expected_cols_free
    col_idx_free = {name: i for i, name in enumerate(res_free.columns)}
    assert res_free.units[col_idx_free["alpha"]] == "1/rad"
    assert res_free.units[col_idx_free["beta"]] == "1/rad"
    assert res_free.units[col_idx_free["delta:ail"]] == "1/rad"
    assert res_free.units[col_idx_free["delta:flap"]] == "1/rad"
    assert res_free.units[col_idx_free["p_hat"]] == "-"
    assert res_free.units[col_idx_free["q_hat"]] == "-"
    assert res_free.units[col_idx_free["r_hat"]] == "-"
    assert "central" in res_free.methods[col_idx_free["alpha"]]
    assert "central" in res_free.methods[col_idx_free["beta"]]

    # Ground effect: automatically adds height group (use h=2.0 so LLT is valid)
    cond_ground = vt.FlightCondition(V_inf=30.0, alpha=np.radians(2.0), h=2.0)
    res_ground = _quiet(derivative_matrix, ac, cond_ground, st)
    expected_cols_ground = ("alpha", "beta", "phi", "h_over_c", "delta:ail", "delta:flap", "p_hat", "q_hat", "r_hat")
    assert res_ground.columns == expected_cols_ground
    col_idx_g = {name: i for i, name in enumerate(res_ground.columns)}
    assert res_ground.units[col_idx_g["phi"]] == "1/rad"
    assert res_ground.units[col_idx_g["h_over_c"]] == "-"


def test_axes_all():
    """axes='all' returns three matrices where force rows match and moment rows follow transform_moments."""
    ac = _aircraft_two_controls(semi_span=3.0)
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(5.0), beta=np.radians(3.0))
    st = vt.SolverSettings(solver_type="linear", n_panels=10)

    res_all = _quiet(derivative_matrix, ac, cond, st, axes="all")
    assert isinstance(res_all, dict)
    assert set(res_all.keys()) == {"body", "stability", "wind"}

    m_body = res_all["body"]
    m_stab = res_all["stability"]
    m_wind = res_all["wind"]

    # Force rows (0, 1, 2) are identical across all axis systems
    for r in range(3):
        assert np.allclose(m_body[r, :], m_stab[r, :])
        assert np.allclose(m_body[r, :], m_wind[r, :])

    # Moment rows (3, 4, 5) match transform_moments
    n_cols = m_body.shape[1]
    for c in range(n_cols):
        body_mom = m_body[3:6, c]
        stab_dict = transform_moments(
            body_mom[0], body_mom[1], body_mom[2],
            float(cond.alpha), float(cond.beta),
            to="stability", from_axes="body",
            b_ref=ac.b_ref, c_ref=ac.c_ref,
        )
        wind_dict = transform_moments(
            body_mom[0], body_mom[1], body_mom[2],
            float(cond.alpha), float(cond.beta),
            to="wind", from_axes="body",
            b_ref=ac.b_ref, c_ref=ac.c_ref,
        )
        stab_mom = np.array([stab_dict["Cl"], stab_dict["Cm"], stab_dict["Cn"]])
        wind_mom = np.array([wind_dict["Cl"], wind_dict["Cm"], wind_dict["Cn"]])
        assert np.allclose(m_stab[3:6, c], stab_mom, rtol=1e-12, atol=1e-14)
        assert np.allclose(m_wind[3:6, c], wind_mom, rtol=1e-12, atol=1e-14)


def test_far_field_diagnostics_present():
    """Per-control far-field diagnostics are present and warnings are raised when applicable."""
    w = vt.LiftingSurface(
        name="wing",
        semi_span=5.0,
        sections=[vt.WingSection(0.0, 1.0), vt.WingSection(1.0, 1.0)],
        controls=[vt.ControlSurface("ail", 0.6, 1.0, 0.75, 0.0, symmetric=False)],
    )
    cond = vt.FlightCondition(V_inf=20.0, alpha=math.radians(3.0), beta=math.radians(3.0))
    st = vt.SolverSettings(solver_type="vlm", n_panels=16, n_chord=4)

    res = _quiet(derivative_matrix, w, cond, st, groups=("control",))
    assert "ail" in res.diagnostics
    diag_ail = res.diagnostics["ail"]
    assert "CY_farfield" in diag_ail
    assert "Cn_trefftz" in diag_ail
    assert math.isfinite(diag_ail["CY_farfield"])
    assert math.isfinite(diag_ail["Cn_trefftz"])


def test_height_refused_in_free_air():
    """Requesting height group in free air or invalid options raises ValueError."""
    ac = _aircraft_two_controls(semi_span=3.0)
    cond_free = vt.FlightCondition(V_inf=30.0, alpha=np.radians(2.0))
    cond_ground = vt.FlightCondition(V_inf=30.0, alpha=np.radians(2.0), h=1.0)
    st = vt.SolverSettings(solver_type="linear", n_panels=8)

    # Height group in free air raises ValueError
    with pytest.raises(ValueError, match="free air"):
        derivative_matrix(ac, cond_free, st, groups=("static", "height"))

    # Unknown group
    with pytest.raises(ValueError, match="Unknown group"):
        derivative_matrix(ac, cond_free, st, groups=("static", "unknown_group"))

    # Unknown control name
    with pytest.raises(ValueError, match="Unknown control"):
        derivative_matrix(ac, cond_free, st, groups=("control",), controls=["nonexistent_control"])

    # Unknown axes
    with pytest.raises(ValueError, match="'axes' must be"):
        derivative_matrix(ac, cond_ground, st, axes="invalid_axes")


def test_to_dict_json():
    """to_dict produces JSON-safe outputs without NaN values."""
    ac = _aircraft_two_controls(semi_span=3.0)
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(3.0), h=1.2)
    st = vt.SolverSettings(solver_type="linear", n_panels=10)

    res = _quiet(derivative_matrix, ac, cond, st)
    d = res.to_dict()
    assert isinstance(d["matrix"], list)
    assert len(d["matrix"]) == 6
    assert isinstance(d["columns"], list)
    assert isinstance(d["rows"], list)
    assert isinstance(d["diagnostics"], dict)

    raw_json = json.dumps(d, allow_nan=False)
    loaded = json.loads(raw_json)
    assert loaded["axes"] == "body"


def test_as_table():
    """as_table generates a Markdown table containing the axis system in the caption."""
    ac = _aircraft_two_controls(semi_span=3.0)
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(3.0))
    st = vt.SolverSettings(solver_type="linear", n_panels=8)

    res = _quiet(derivative_matrix, ac, cond, st, axes="stability")
    table = res.as_table()
    assert "stability" in table
    assert "| Coefficient |" in table
    assert "CL" in table
    assert "Cm" in table
