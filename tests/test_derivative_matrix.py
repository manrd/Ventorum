# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests of the full aerodynamic derivative matrix."""

from __future__ import annotations

import json
import math
import warnings

import numpy as np
import pytest

import ventorum as vt
from ventorum.agent.schemas import make_flight_condition, parse_condition, parse_settings
from ventorum.agent.tools import build_aircraft_from_spec, stability_derivatives as agent_stability_derivatives
from ventorum.core.axes import transform_moments
from ventorum.core.trust import near_far_lateral_warnings
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


_AXES = ("body", "stability", "wind")


def _assert_column(matrix, col, expected):
    """Assert that one matrix column equals the reference vector (1e-12 relative)."""
    for row_i in range(6):
        assert matrix[row_i, col] == pytest.approx(float(expected[row_i]), rel=1e-12)


def _control_vector(c_res, name):
    d = c_res.derivatives[name]
    return [d.CL, d.CD, d.CY, d.Cl, d.Cm, d.Cn]


@pytest.mark.parametrize("solver", ["linear", "vlm"])
def test_columns_equal_existing_functions(solver):
    """Every column equals the existing function in body, stability and wind axes.

    Regression test: the rate columns in the stability and wind axes once
    turned only the moment rows. rate_derivatives turns the rates and the
    moments together, and the matrix must give the same values.
    """
    ac = _aircraft_two_controls(semi_span=3.0)
    n_chord = 4 if solver == "vlm" else None
    st = vt.SolverSettings(solver_type=solver, n_panels=12, n_chord=n_chord)

    # 1. Free air: static, control and rate columns.
    cond_free = vt.FlightCondition(V_inf=30.0, alpha=np.radians(5.0), beta=np.radians(3.0))
    res_all = _quiet(derivative_matrix, ac, cond_free, st, groups=("static", "control", "rate"), axes="all")
    col = {name: i for i, name in enumerate(res_all.columns)}
    for ax in _AXES:
        m = res_all[ax]
        stat = _quiet(static_derivatives, ac, cond_free, st, axes=ax)
        _assert_column(m, col["alpha"], stat.matrix[:, 0])
        _assert_column(m, col["beta"], stat.matrix[:, 1])
        for coeff_i, coeff in enumerate(res_all.rows):
            assert m[coeff_i, col["alpha"]] == pytest.approx(stat.derivatives[f"{coeff}_alpha"], rel=1e-12)
        ctrl = _quiet(control_derivatives, ac, cond_free, st, axes=ax, hinge_alpha=False)
        for c_name in ("flap", "ail"):
            _assert_column(m, col[f"delta:{c_name}"], _control_vector(ctrl, c_name))
        rate = _quiet(rate_derivatives, ac, cond_free, st, axes=ax)
        for j, r_name in enumerate(("p_hat", "q_hat", "r_hat")):
            _assert_column(m, col[r_name], rate.matrix[:, j])

    # 2. Ground effect: static, height, control and rate columns.
    cond_ground = vt.FlightCondition(V_inf=30.0, alpha=np.radians(3.0), beta=np.radians(2.0), h=2.5)
    res_g = _quiet(derivative_matrix, ac, cond_ground, st, axes="all")
    col_g = {name: i for i, name in enumerate(res_g.columns)}
    assert set(col_g) == {"alpha", "beta", "phi", "h_over_c", "delta:ail", "delta:flap", "p_hat", "q_hat", "r_hat"}
    for ax in _AXES:
        m = res_g[ax]
        h_res = _quiet(height_derivatives, ac, cond_ground, st, axes=ax)
        for j, v_name in enumerate(("alpha", "beta", "phi", "h_over_c")):
            _assert_column(m, col_g[v_name], h_res.matrix[:, j])
        ctrl = _quiet(control_derivatives, ac, cond_ground, st, axes=ax, hinge_alpha=False)
        for c_name in ("flap", "ail"):
            _assert_column(m, col_g[f"delta:{c_name}"], _control_vector(ctrl, c_name))
        rate = _quiet(rate_derivatives, ac, cond_ground, st, axes=ax)
        for j, r_name in enumerate(("p_hat", "q_hat", "r_hat")):
            _assert_column(m, col_g[r_name], rate.matrix[:, j])


@pytest.mark.parametrize("solver", ["vlm", "linear"])
@pytest.mark.parametrize("axes", _AXES)
def test_static_equals_agent_tool(solver, axes, monkeypatch):
    """static_derivatives equals ventorum_stability_derivatives to 1e-12, without rounding.

    The tool rounds its output, so the test replaces its rounding function
    by the identity and compares the unrounded numbers. The tests run on
    the CPU device (conftest), where the two solve paths agree to round-off.
    """
    import ventorum.agent.response as response
    import ventorum.agent.tools as tools

    def _identity(value, *args, **kwargs):
        return None if value is None else float(value)

    monkeypatch.setattr(tools, "rnd", _identity)
    monkeypatch.setattr(response, "rnd", _identity)

    spec = {"span_m": 6.0, "root_chord_m": 1.2, "tip_chord_m": 0.8, "dihedral_deg": 4.0}
    fc_spec = {"V_inf_m_s": 25.0, "alpha_deg": 2.0, "beta_deg": 1.0}
    sett_spec = {"solver": solver, "n_panels": 10}
    res_agent = _quiet(agent_stability_derivatives, wing=spec, flight_condition=fc_spec, settings=sett_spec,
                       x_cg_m=0.4, axes=axes)
    assert res_agent["status"] == "success", res_agent
    m = res_agent["stability_derivatives"]

    # The same case as the tool builds it.
    ac = build_aircraft_from_spec(wing=spec)
    base = ac.moment_reference()
    ac.ref_point = np.array([0.4, 0.0, float(base[2])])
    ac.compute_reference_values()
    cond = make_flight_condition(parse_condition(fc_spec))
    st = parse_settings(sett_spec)
    res_py = _quiet(static_derivatives, ac, cond, st, axes=axes)

    pairs = (("CL_alpha", "CL_alpha_per_rad"), ("Cm_alpha", "Cm_alpha_per_rad"), ("CY_beta", "CY_beta_per_rad"),
             ("Cl_beta", "Cl_beta_per_rad"), ("Cn_beta", "Cn_beta_per_rad"))
    for py_key, tool_key in pairs:
        assert res_py.derivatives[py_key] == pytest.approx(m[tool_key], rel=1e-12), py_key
    assert res_py.static_margin == pytest.approx(m["static_margin_fraction"], rel=1e-12)
    assert res_py.neutral_point_x == pytest.approx(m["neutral_point_x_m"], rel=1e-12)


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
    """axes='all' gives three matrices with the transformation of each column type.

    Force rows of the static and control columns are equal in all sets and
    their moment rows equal transform_moments of the body set. The rate
    columns turn with the rates, so they equal rate_derivatives in each set.
    """
    ac = _aircraft_two_controls(semi_span=3.0)
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(5.0), beta=np.radians(3.0))
    st = vt.SolverSettings(solver_type="linear", n_panels=10)

    res_all = _quiet(derivative_matrix, ac, cond, st, axes="all")
    assert isinstance(res_all, dict)
    assert set(res_all.keys()) == {"body", "stability", "wind"}
    m_body = res_all["body"]
    rate_names = ("p_hat", "q_hat", "r_hat")
    rate_cols = [res_all.columns.index(n) for n in rate_names]
    other_cols = [j for j in range(m_body.shape[1]) if j not in rate_cols]
    assert other_cols and len(rate_cols) == 3

    for ax in ("stability", "wind"):
        m_ax = res_all[ax]
        for c in other_cols:
            assert np.array_equal(m_body[:3, c], m_ax[:3, c])
            t = transform_moments(
                m_body[3, c], m_body[4, c], m_body[5, c],
                float(cond.alpha), float(cond.beta),
                to=ax, from_axes="body",
                b_ref=ac.b_ref, c_ref=ac.c_ref,
            )
            assert np.allclose(m_ax[3:6, c], [t["Cl"], t["Cm"], t["Cn"]], rtol=1e-12, atol=1e-14)
        rate = _quiet(rate_derivatives, ac, cond, st, axes=ax)
        for j, c in enumerate(rate_cols):
            assert np.allclose(m_ax[:, c], rate.matrix[:, j], rtol=1e-12, atol=1e-14)
        # The rate columns differ from the body set (the rates turn too).
        assert not np.allclose(m_ax[:, rate_cols], m_body[:, rate_cols], rtol=1e-6, atol=1e-9)


def test_far_field_diagnostics_present():
    """Per control, the far-field derivatives and the near/far warning appear once, under 'controls'.

    The aileron case is the antisymmetric aileron on the flat AR 8 wing of
    the far-field tests, here with 3 deg of sideslip. The expected warnings
    come from the near/far rule applied to the control derivatives of
    control_derivatives.
    """
    af = vt.LinearAirfoil()
    w = vt.LiftingSurface(
        name="w", semi_span=4.0, is_symmetric=True,
        sections=[vt.WingSection(0.0, 1.0, airfoil=af), vt.WingSection(1.0, 1.0, airfoil=af)],
        controls=[vt.ControlSurface("ail", 0.6, 1.0, 0.75, math.radians(5.0), symmetric=False)],
    )
    cond = vt.FlightCondition(V_inf=20.0, alpha=math.radians(3.0), beta=math.radians(3.0))
    st = vt.SolverSettings(solver_type="vlm", n_panels=24, spacing="cosine", n_chord=4)

    res = _quiet(derivative_matrix, w, cond, st, groups=("control",))
    # One entry per control, only under 'controls' (no top-level copy that can collide).
    assert set(res.diagnostics) == {"controls"}
    diag = res.diagnostics["controls"]["ail"]
    assert set(diag) == {"CY_farfield", "Cn_trefftz", "warnings"}

    ref = _quiet(control_derivatives, w, cond, st, hinge_alpha=False)
    d = ref.derivatives["ail"]
    assert diag["CY_farfield"] == pytest.approx(d.CY_farfield, rel=1e-12)
    assert diag["Cn_trefftz"] == pytest.approx(d.Cn_trefftz, rel=1e-12)
    cn_stab = transform_moments(d.Cl, d.Cm, d.Cn, float(cond.alpha), float(cond.beta), to="stability",
                                from_axes="body", b_ref=res.b_ref, c_ref=res.c_ref)["Cn"]
    expected = near_far_lateral_warnings(CY=d.CY, CY_farfield=d.CY_farfield, Cn=cn_stab, Cn_trefftz=d.Cn_trefftz)
    assert diag["warnings"] == expected
    # The rule applies to this case: in sideslip the deflected aileron gives a near-field side force,
    # and the flat Trefftz trace of a deflected control gives none.
    assert expected, "the near/far rule must apply to the aileron case"
    assert any(txt.startswith("Near field and far field do not agree") for txt in diag["warnings"])


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
