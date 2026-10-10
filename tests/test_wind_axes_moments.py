# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Tests of moment coefficients and moment derivatives in wind axes.

Reference lengths b_ref and c_ref scale the moments before and after
rotation. This file tests that the transform scales dimensional moments
correctly for any combination of reference span and reference chord.
"""

from __future__ import annotations

import numpy as np
import pytest

import ventorum as vt
from ventorum.agent import stability_derivatives, wing_analysis
from ventorum.core.axes import (
    AXES_VALUES,
    moment_axes_matrix,
    rotation_body_to_stability,
    rotation_body_to_wind,
    transform_moments,
)
from ventorum.core.datatypes import IntegratedResult, SolverResult
from ventorum.solvers.control_derivatives import control_derivatives


def test_hand_case_span_not_equal_chord():
    """Hand-computed wind-axes moments at alpha 0 and beta 90 deg."""
    cl, cm, cn = 1.0, 2.0, 3.0
    alpha = 0.0
    beta = np.radians(90.0)

    # With b_ref = c_ref = 1 the result is (2, -1, 3).
    got1 = transform_moments(cl, cm, cn, alpha, beta, "wind", b_ref=1.0, c_ref=1.0)
    assert [got1["Cl"], got1["Cm"], got1["Cn"]] == pytest.approx([2.0, -1.0, 3.0], abs=1e-12)

    # With b_ref = 2, c_ref = 1 the result is (1, -2, 3).
    got2 = transform_moments(cl, cm, cn, alpha, beta, "wind", b_ref=2.0, c_ref=1.0)
    assert [got2["Cl"], got2["Cm"], got2["Cn"]] == pytest.approx([1.0, -2.0, 3.0], abs=1e-12)


def test_equals_rotation_of_dimensional_moments():
    """Transform equals the dimensional moment rotation divided by reference lengths."""
    alpha = np.radians(7.0)
    beta = np.radians(5.0)
    b_ref = 3.0
    c_ref = 0.7
    c_vec = np.array([0.15, -0.08, 0.04], dtype=float)
    dim_vec = np.array([b_ref * c_vec[0], c_ref * c_vec[1], b_ref * c_vec[2]], dtype=float)

    # Wind axes check
    r_bw = rotation_body_to_wind(alpha, beta)
    rot_dim_w = r_bw @ dim_vec
    exp_w = [rot_dim_w[0] / b_ref, rot_dim_w[1] / c_ref, rot_dim_w[2] / b_ref]
    got_w = transform_moments(*c_vec, alpha, beta, "wind", b_ref=b_ref, c_ref=c_ref)
    assert [got_w["Cl"], got_w["Cm"], got_w["Cn"]] == pytest.approx(exp_w, abs=1e-12)

    # Stability axes check
    r_bs = rotation_body_to_stability(alpha)
    rot_dim_s = r_bs @ dim_vec
    exp_s = [rot_dim_s[0] / b_ref, rot_dim_s[1] / c_ref, rot_dim_s[2] / b_ref]
    got_s = transform_moments(*c_vec, alpha, beta, "stability", b_ref=b_ref, c_ref=c_ref)
    assert [got_s["Cl"], got_s["Cm"], got_s["Cn"]] == pytest.approx(exp_s, abs=1e-12)


def test_round_trip_all_pairs():
    """Every pair of axes round-trips to the input within 1e-12."""
    alpha = np.radians(7.0)
    beta = np.radians(5.0)
    b_ref = 3.0
    c_ref = 0.7
    start = [0.12, -0.05, 0.03]

    for sys_from in AXES_VALUES:
        for sys_to in AXES_VALUES:
            forward = transform_moments(*start, alpha, beta, sys_to, from_axes=sys_from,
                                        b_ref=b_ref, c_ref=c_ref)
            back = transform_moments(forward["Cl"], forward["Cm"], forward["Cn"],
                                     alpha, beta, sys_from, from_axes=sys_to,
                                     b_ref=b_ref, c_ref=c_ref)
            assert [back["Cl"], back["Cm"], back["Cn"]] == pytest.approx(start, abs=1e-12)


def test_stability_axes_unchanged():
    """Stability axes moments equal the plain rotation matrix for any reference lengths."""
    alpha = np.radians(7.0)
    beta = np.radians(5.0)
    b_ref = 3.0
    c_ref = 0.7
    c_vec = np.array([0.15, -0.08, 0.04], dtype=float)

    plain = rotation_body_to_stability(alpha) @ c_vec
    got = transform_moments(*c_vec, alpha, beta, "stability", b_ref=b_ref, c_ref=c_ref)
    assert [got["Cl"], got["Cm"], got["Cn"]] == pytest.approx(list(plain), abs=1e-12)


def test_invalid_reference_lengths():
    """Invalid reference lengths raise ValueError or TypeError."""
    alpha = np.radians(4.0)
    beta = np.radians(2.0)

    with pytest.raises(ValueError, match="b_ref"):
        transform_moments(1.0, 2.0, 3.0, alpha, beta, "wind", b_ref=0.0, c_ref=1.0)

    with pytest.raises(ValueError, match="c_ref"):
        transform_moments(1.0, 2.0, 3.0, alpha, beta, "wind", b_ref=1.0, c_ref=-1.0)

    with pytest.raises(ValueError, match="b_ref"):
        transform_moments(1.0, 2.0, 3.0, alpha, beta, "wind", b_ref=float("nan"), c_ref=1.0)

    with pytest.raises(TypeError):
        # Missing keyword arguments
        transform_moments(1.0, 2.0, 3.0, alpha, beta, "wind")  # type: ignore[call-arg]


def _rectangular_wing(span: float = 6.0, chord: float = 1.0) -> vt.LiftingSurface:
    """Build a rectangular wing surface with known reference span and chord."""
    return vt.LiftingSurface(
        name="wing",
        semi_span=span / 2.0,
        is_symmetric=True,
        sections=[
            vt.WingSection(0.0, chord, airfoil=vt.LinearAirfoil()),
            vt.WingSection(1.0, chord, airfoil=vt.LinearAirfoil()),
        ],
    )


def test_result_has_reference_lengths():
    """Every solver populates result.b_ref and result.c_ref."""
    wing = _rectangular_wing(span=6.0, chord=1.0)
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=np.radians(5.0))
    cond_fourier = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=0.0)
    sett_fast = vt.SolverSettings(n_panels=8, n_chord=1)

    for solver_name in ("vlm", "linear", "nonlinear"):
        sett = vt.SolverSettings(solver_type=solver_name, n_panels=8, n_chord=1)
        res = vt.analyze(wing, cond, sett)
        assert res.b_ref == pytest.approx(6.0, abs=1e-12)
        assert res.c_ref == pytest.approx(1.0, abs=1e-12)

    res_fourier = vt.analyze(wing, cond_fourier, vt.SolverSettings(solver_type="fourier", n_panels=8))
    assert res_fourier.b_ref == pytest.approx(6.0, abs=1e-12)
    assert res_fourier.c_ref == pytest.approx(1.0, abs=1e-12)

    # CPU batch solve
    sweep_results = vt.analyze_sweep(
        wing,
        np.array([2.0, 4.0]),
        V_inf=30.0,
        settings=sett_fast,
    )
    for r in sweep_results:
        assert r.b_ref == pytest.approx(6.0, abs=1e-12)
        assert r.c_ref == pytest.approx(1.0, abs=1e-12)


def test_result_moments_wind():
    """Result moments in wind axes match dimensional rotation."""
    wing = _rectangular_wing(span=6.0, chord=1.0)
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=np.radians(5.0))
    sett = vt.SolverSettings(solver_type="vlm", n_panels=8, n_chord=2)
    res = vt.analyze(wing, cond, sett)

    body_mom = res.moments("body")
    assert body_mom["Cl"] == pytest.approx(res.totals.Cl, abs=1e-15)
    assert body_mom["Cm"] == pytest.approx(res.totals.Cm, abs=1e-15)
    assert body_mom["Cn"] == pytest.approx(res.totals.Cn, abs=1e-15)

    dim_vec = np.array([res.b_ref * res.totals.Cl, res.c_ref * res.totals.Cm, res.b_ref * res.totals.Cn], dtype=float)
    rot_dim = rotation_body_to_wind(float(cond.alpha), float(cond.beta)) @ dim_vec
    exp_w = [rot_dim[0] / res.b_ref, rot_dim[1] / res.c_ref, rot_dim[2] / res.b_ref]

    wind_mom = res.moments("wind")
    assert [wind_mom["Cl"], wind_mom["Cm"], wind_mom["Cn"]] == pytest.approx(exp_w, abs=1e-12)


def test_result_without_reference_raises():
    """SolverResult without reference lengths raises ValueError for non-body axes."""
    res = SolverResult(totals=IntegratedResult(Cl=0.01, Cm=-0.02, Cn=0.03))
    assert res.b_ref is None
    assert res.c_ref is None

    # Body axes works without reference lengths
    body = res.moments("body")
    assert body == {"Cl": 0.01, "Cm": -0.02, "Cn": 0.03}

    with pytest.raises(ValueError, match="b_ref"):
        res.moments("wind")

    with pytest.raises(ValueError, match="b_ref"):
        res.moments("stability")


def test_control_derivatives_wind():
    """Wind-axis control derivatives match moment_axes_matrix applied to body derivatives."""
    aileron = vt.ControlSurface("aileron", 0.5, 1.0, 0.75, 0.0, symmetric=False)
    surf = vt.LiftingSurface(
        name="wing",
        semi_span=3.0,
        is_symmetric=True,
        sections=[
            vt.WingSection(0.0, 1.0, airfoil=vt.LinearAirfoil()),
            vt.WingSection(1.0, 1.0, airfoil=vt.LinearAirfoil()),
        ],
        controls=[aileron],
    )
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=np.radians(3.0))
    sett = vt.SolverSettings(solver_type="vlm", n_panels=12, n_chord=2)

    res_body = control_derivatives(surf, cond, sett, axes="body", hinge_alpha=False)
    res_wind = control_derivatives(surf, cond, sett, axes="wind", hinge_alpha=False)

    d_body = res_body.derivatives["aileron"]
    d_wind = res_wind.derivatives["aileron"]

    m_body = np.array([d_body.Cl, d_body.Cm, d_body.Cn], dtype=float)
    m_mat = moment_axes_matrix(float(cond.alpha), float(cond.beta), to="wind",
                               b_ref=res_body.base.b_ref, c_ref=res_body.base.c_ref)
    exp_wind = m_mat @ m_body

    assert [d_wind.Cl, d_wind.Cm, d_wind.Cn] == pytest.approx(list(exp_wind), abs=1e-10)


def test_stability_derivatives_tool_wind(monkeypatch):
    """The stability_derivatives tool expresses wind moments via moment_axes_matrix."""
    import math as _math

    from ventorum.agent import tools

    def _noround(x, n=6):
        if x is None:
            return None
        v = float(x)
        return v if _math.isfinite(v) else None

    monkeypatch.setattr(tools, "rnd", _noround)
    wing_spec = {"span_m": 8.0, "chord_m": 1.0}
    cond_spec = {"alpha_deg": 5.0, "beta_deg": 4.0}
    sett_spec = {"n_panels": 8, "n_chord": 1}

    out_wind = stability_derivatives(wing_spec, cond_spec, settings=sett_spec, axes="wind")
    assert out_wind["status"] == "success", out_wind
    dw = out_wind["stability_derivatives"]

    # Rebuild body-axis finite difference moment derivative vectors
    cases = {"a+": (5.5, 4.0), "a-": (4.5, 4.0), "b+": (5.0, 5.0), "b-": (5.0, 3.0)}
    mom = {}
    for key, (aa, bb) in cases.items():
        p = wing_analysis(wing_spec, {"alpha_deg": aa, "beta_deg": bb}, sett_spec, "summary", axes="body")
        assert p["status"] == "success", p
        mom[key] = p["metrics"]

    da = np.radians(1.0)
    db = np.radians(2.0)
    d_m_da = np.array([
        (mom["a+"]["Cl"] - mom["a-"]["Cl"]) / da,
        (mom["a+"]["Cm"] - mom["a-"]["Cm"]) / da,
        (mom["a+"]["Cn"] - mom["a-"]["Cn"]) / da,
    ])
    d_m_db = np.array([
        (mom["b+"]["Cl"] - mom["b-"]["Cl"]) / db,
        (mom["b+"]["Cm"] - mom["b-"]["Cm"]) / db,
        (mom["b+"]["Cn"] - mom["b-"]["Cn"]) / db,
    ])

    a0 = np.radians(5.0)
    b0 = np.radians(4.0)
    m_mat = moment_axes_matrix(a0, b0, to="wind", b_ref=8.0, c_ref=1.0)
    exp_wind_a = m_mat @ d_m_da
    exp_wind_b = m_mat @ d_m_db

    assert dw["Cm_alpha_per_rad"] == pytest.approx(float(exp_wind_a[1]), abs=1e-10)
    assert dw["Cl_beta_per_rad"] == pytest.approx(float(exp_wind_b[0]), abs=1e-10)
    assert dw["Cn_beta_per_rad"] == pytest.approx(float(exp_wind_b[2]), abs=1e-10)


def test_agent_wind_set_equals_python():
    """wing_analysis with axes='all' gives wind moments equal to vt.analyze().moments('wind')."""
    wing_spec = {"span_m": 6.0, "chord_m": 1.2}
    cond_spec = {"alpha_deg": 3.0, "beta_deg": 5.0}
    sett_spec = {"n_panels": 8, "n_chord": 1}

    out = wing_analysis(wing_spec, cond_spec, sett_spec, "summary", axes="all")
    assert out["status"] == "success", out
    wind_agent = out["metrics"]["moments"]["wind"]

    wing = _rectangular_wing(span=6.0, chord=1.2)
    cond = vt.FlightCondition(V_inf=50.0, alpha=np.radians(3.0), beta=np.radians(5.0))
    sett = vt.SolverSettings(n_panels=8, n_chord=1)
    res = vt.analyze(wing, cond, sett)
    wind_py = res.moments("wind")

    assert wind_agent["Cl"] == pytest.approx(round(wind_py["Cl"], 6), abs=1e-6)
    assert wind_agent["Cm"] == pytest.approx(round(wind_py["Cm"], 6), abs=1e-6)
    assert wind_agent["Cn"] == pytest.approx(round(wind_py["Cn"], 6), abs=1e-6)


@pytest.mark.gpu
def test_gpu_result_reference_lengths():
    """GPU solver results carry b_ref and c_ref and match CPU moments within 1e-4."""
    from ventorum import gpu

    if not gpu.available():
        pytest.skip(f"GPU pipeline unavailable: {gpu.unavailable_reason()}")

    old_dev = gpu.get_device()
    wing = _rectangular_wing(span=6.0, chord=1.0)
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=np.radians(5.0))
    sett = vt.SolverSettings(solver_type="vlm", n_panels=16, n_chord=2)

    try:
        gpu.set_device("cpu")
        res_cpu = vt.analyze(wing, cond, sett)

        gpu.set_device("gpu")
        res_gpu = vt.analyze(wing, cond, sett)

        assert res_gpu.details.get("device") == "gpu"
        assert res_gpu.b_ref == pytest.approx(6.0, abs=1e-12)
        assert res_gpu.c_ref == pytest.approx(1.0, abs=1e-12)

        mom_cpu = res_cpu.moments("wind")
        mom_gpu = res_gpu.moments("wind")

        assert mom_gpu["Cl"] == pytest.approx(mom_cpu["Cl"], abs=1e-4)
        assert mom_gpu["Cm"] == pytest.approx(mom_cpu["Cm"], abs=1e-4)
        assert mom_gpu["Cn"] == pytest.approx(mom_cpu["Cn"], abs=1e-4)
    finally:
        gpu.set_device(old_dev)
