# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests of the control derivatives and the hinge moments (plan item P1-03)."""

from __future__ import annotations

import json
import math
import warnings

import numpy as np
import pytest
from scipy.integrate import quad

import ventorum as vt
from ventorum import gpu
from ventorum.aero.hinge import hinge_moments
from ventorum.geometry.controls import flap_effectiveness, hinge_moment_derivatives
from ventorum.solvers.control_derivatives import control_derivatives

COND = vt.FlightCondition(V_inf=20.0, alpha=np.radians(3.0))


def _wing(controls, semi_span=4.0, airfoil=None, is_symmetric=True, name="w"):
    af = airfoil if airfoil is not None else vt.LinearAirfoil()
    return vt.LiftingSurface(
        name=name, semi_span=semi_span, is_symmetric=is_symmetric,
        sections=[vt.WingSection(0.0, 1.0, airfoil=af), vt.WingSection(1.0, 1.0, airfoil=af)],
        controls=controls,
    )


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Thin-airfoil hinge-moment derivatives
# ═══════════════════════════════════════════════════════════════════════════════

def _quadrature_hinge(h: float) -> tuple[float, float]:
    """Integrate the thin-airfoil load times the arm aft of the hinge (independent of the closed forms)."""
    th_h = math.acos(1.0 - 2.0 * h)
    e = 1.0 - h

    def arm(th):
        return (math.cos(th_h) - math.cos(th)) / 2.0 * math.sin(th) / 2.0

    def plate(th):   # gamma / V of a flat plate at alpha = 1 rad
        return 2.0 * (1.0 + math.cos(th)) / math.sin(th)

    def flap(th):    # gamma / V of a flap deflection of 1 rad (Glauert series summed)
        a0 = (math.pi - th_h) / math.pi
        log = math.log(abs(math.sin((th + th_h) / 2.0) / math.sin((th - th_h) / 2.0))) / math.pi
        return 2.0 * (a0 * (1.0 + math.cos(th)) / math.sin(th) + log)

    def ch(load):
        val, _ = quad(lambda t: load(t) * arm(t), th_h, math.pi, epsabs=1e-14, epsrel=1e-13, limit=200)
        return -2.0 / e ** 2 * val

    return ch(plate), ch(flap)


@pytest.mark.parametrize("h", [0.2, 0.5, 0.7, 0.75, 0.9])
def test_thin_airfoil_hinge_derivatives_equal_quadrature(h):
    """The closed forms equal a quadrature of the thin-airfoil load."""
    ch_a, ch_d = hinge_moment_derivatives(h)
    q_a, q_d = _quadrature_hinge(h)
    assert ch_a == pytest.approx(q_a, rel=1e-9, abs=1e-12)
    assert ch_d == pytest.approx(q_d, rel=1e-9, abs=1e-12)
    assert ch_a < 0.0 and ch_d < 0.0


def test_thin_airfoil_hinge_derivatives_limits():
    """Hinge at the leading edge: moment of a flat plate about it; hinge at the trailing edge: finite Ch_delta."""
    ch_a, ch_d = hinge_moment_derivatives(1e-8)
    assert ch_a == pytest.approx(-math.pi / 2.0, rel=1e-5)
    assert ch_d == pytest.approx(-math.pi / 2.0, rel=1e-5)
    ch_a, ch_d = hinge_moment_derivatives(1.0 - 1e-6)
    assert abs(ch_a) < 1e-2
    assert ch_d == pytest.approx(-8.0 / (3.0 * math.pi), rel=1e-4)
    # A published value: hinge 0.75 (flap chord ratio 0.25).
    ch_a, ch_d = hinge_moment_derivatives(0.75)
    assert ch_a == pytest.approx(-0.5653472565, rel=1e-9)
    assert ch_d == pytest.approx(-0.9436078642, rel=1e-9)


@pytest.mark.parametrize("h", [0.0, 1.0, -0.1, 1.5, float("nan")])
def test_thin_airfoil_hinge_derivatives_refuse_bad_hinge(h):
    with pytest.raises(ValueError, match="hinge_x_c"):
        hinge_moment_derivatives(h)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Hinge moments of a solved case
# ═══════════════════════════════════════════════════════════════════════════════

def test_vlm_hinge_moment_converges_to_thin_airfoil():
    """Chordwise mode on a wing of aspect ratio 40: Ch approaches thin-airfoil theory at the strip lift."""
    h, d = 0.75, np.radians(5.0)
    ch_a, ch_d = hinge_moment_derivatives(h)
    tau = flap_effectiveness(h)
    errs = []
    for nc in (8, 16, 32):
        w = _wing([vt.ControlSurface("f", 0.0, 1.0, h, d)], semi_span=20.0)
        st = vt.SolverSettings(solver_type="vlm", n_panels=20, n_chord=nc, chord_spacing="uniform")
        r = _quiet(vt.analyze, w, vt.FlightCondition(V_inf=10.0, alpha=np.radians(3.0)), st)
        m = hinge_moments(r)
        assert [x.side for x in m] == ["right", "left"]
        assert m[0].Ch == pytest.approx(m[1].Ch, rel=1e-9)
        assert m[0].mode == "chordwise" and m[0].hinge_x_c == h
        lat = r.details["lattice"]
        idx = np.flatnonzero(lat.strip_is_right)
        cl = 2.0 * r.details["gamma"].reshape(-1, nc).sum(axis=1)[idx] / (10.0 * lat.chord[idx])
        wt = lat.width[idx] * ((1.0 - h) * lat.chord[idx]) ** 2
        pred = float(np.sum(wt * (ch_a * (cl / (2.0 * np.pi) - tau * d) + ch_d * d)) / np.sum(wt))
        errs.append(m[0].Ch / pred - 1.0)
    # Pre-check: +6.3 %, +3.2 %, +1.3 %.
    assert errs[0] > errs[1] > errs[2] > 0.0
    assert errs[2] < 0.015


def test_section_mode_hinge_moment_equals_thin_airfoil_formula():
    """Lifting line: the hinge moment is the weighted thin-airfoil value at the strip lift."""
    h, d = 0.7, np.radians(4.0)
    af = vt.LinearAirfoil(a0=5.8, alpha_L0=np.radians(-2.0), Cm0=-0.05)
    w = _wing([vt.ControlSurface("f", 0.2, 0.8, h, d)], airfoil=af)
    r = _quiet(vt.analyze, w, COND, vt.SolverSettings(solver_type="linear", n_panels=24))
    m = hinge_moments(r)
    ch_a, ch_d = hinge_moment_derivatives(h)
    tau = flap_effectiveness(h)
    sw = r.spanwise[0]
    lat = r.details["lattice"]
    q = 0.5 * COND.rho * COND.V_inf ** 2
    for item in m:
        right = item.side == "right"
        idx = np.flatnonzero((lat.strip_is_right == right) & (np.abs(lat.eta) > 0.2) & (np.abs(lat.eta) < 0.8))
        assert idx.size == item.n_strips
        a_fp = sw.Cl[idx] / af.a0 - tau * d
        cf = (1.0 - h) * lat.chord[idx]
        H = float(np.sum(q * cf ** 2 * lat.width[idx] * (ch_a * a_fp + ch_d * d)))
        assert item.mode == "section"
        assert item.H == pytest.approx(H, rel=1e-12)
        assert item.Ch == pytest.approx(H / (q * np.sum(cf ** 2 * lat.width[idx])), rel=1e-12)
        assert item.S_f == pytest.approx(float(np.sum(cf * lat.width[idx])), rel=1e-12)


def test_hinge_moment_signs():
    """A positive angle of attack and a positive deflection both give a negative (trailing edge up) moment."""
    for solver, nc in (("vlm", 6), ("linear", 1)):
        st = vt.SolverSettings(solver_type=solver, n_panels=12, n_chord=nc)
        w = _wing([vt.ControlSurface("f", 0.0, 1.0, 0.75, 0.0)])
        assert hinge_moments(_quiet(vt.analyze, w, COND, st))[0].Ch < 0.0
        w = _wing([vt.ControlSurface("f", 0.0, 1.0, 0.75, np.radians(5.0))])
        r = _quiet(vt.analyze, w, vt.FlightCondition(V_inf=20.0, alpha=0.0), st)
        assert hinge_moments(r)[0].Ch < 0.0


def test_hinge_moments_symmetric_wing_equal_half_plus_mirror():
    """A flap on a symmetric wing equals the right half plus its mirror copy."""
    st = vt.SolverSettings(solver_type="vlm", n_panels=10, n_chord=6)
    c = vt.ControlSurface("f", 0.3, 0.8, 0.7, np.radians(6.0))
    sym = _wing([c.clone()])
    right = _wing([c.clone()], is_symmetric=False)
    pair = vt.Aircraft(surfaces=[right, right.mirrored(name="w_left")])
    m_sym = hinge_moments(_quiet(vt.analyze, vt.Aircraft(surfaces=[sym]), COND, st))
    m_pair = hinge_moments(_quiet(vt.analyze, pair, COND, st))
    assert [x.side for x in m_pair] == ["single", "single"]
    assert m_pair[0].Ch == pytest.approx(m_sym[0].Ch, rel=1e-5)
    assert m_pair[1].Ch == pytest.approx(m_sym[1].Ch, rel=1e-5)
    assert m_pair[1].deflection == pytest.approx(np.radians(6.0))


def test_hinge_moments_refuse_fourier_and_no_controls():
    w = _wing([])
    r = _quiet(vt.analyze, w, COND, vt.SolverSettings(solver_type="vlm", n_panels=8))
    with pytest.raises(ValueError, match="no control"):
        hinge_moments(r)
    r = _quiet(vt.analyze, w, COND, vt.SolverSettings(solver_type="fourier", n_panels=8))
    with pytest.raises(ValueError, match="Fourier"):
        hinge_moments(r)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Control derivatives
# ═══════════════════════════════════════════════════════════════════════════════

def test_lifting_line_flap_derivative_is_tau_times_lift_slope():
    """Full-span flap on a linear lifting line: dCL/ddelta is close to tau CL_alpha."""
    h = 0.7
    w = _wing([vt.ControlSurface("flap", 0.0, 1.0, h, 0.0)])
    st = vt.SolverSettings(solver_type="linear", n_panels=20)
    res = _quiet(control_derivatives, w, COND, st)
    da = np.radians(0.5)
    cp = _quiet(vt.analyze, w, vt.FlightCondition(V_inf=20.0, alpha=COND.alpha + da), st).totals.CL
    cm = _quiet(vt.analyze, w, vt.FlightCondition(V_inf=20.0, alpha=COND.alpha - da), st).totals.CL
    cl_alpha = (cp - cm) / (2.0 * da)
    d = res.derivatives["flap"]
    assert d.scheme == "central" and d.step == pytest.approx(np.radians(1.0))
    # Pre-check: 0.15 % (the deflection shifts the section only; alpha also turns the wake).
    assert d.CL == pytest.approx(flap_effectiveness(h) * cl_alpha, rel=5e-3)
    assert d.Cm < 0.0
    assert res.n_solves == 5
    assert res.device == "cpu" and res.precision == "float64"


def test_vlm_full_span_flap_effectiveness_ratio():
    """VLM chordwise mode: CL_delta / CL_alpha approaches tau (pre-check of T-0060: -3 % at n_chord 8)."""
    h = 0.75
    w = _wing([vt.ControlSurface("flap", 0.0, 1.0, h, 0.0)])
    st = vt.SolverSettings(solver_type="vlm", n_panels=20, n_chord=16, chord_spacing="uniform")
    res = _quiet(control_derivatives, w, COND, st, hinge_alpha=False)
    da = np.radians(0.5)
    cp = _quiet(vt.analyze, w, vt.FlightCondition(V_inf=20.0, alpha=COND.alpha + da), st).totals.CL
    cm = _quiet(vt.analyze, w, vt.FlightCondition(V_inf=20.0, alpha=COND.alpha - da), st).totals.CL
    ratio = res.derivatives["flap"].CL / ((cp - cm) / (2.0 * da))
    assert ratio == pytest.approx(flap_effectiveness(h), rel=0.02)
    assert all(math.isnan(x.Ch_alpha) for x in res.hinge)


@pytest.mark.parametrize("solver,nc", [("vlm", 6), ("linear", 1)])
def test_aileron_derivatives(solver, nc):
    """Antisymmetric aileron: no lift or pitch change, roll to the left wing down, mirror hinge moments."""
    w = _wing([vt.ControlSurface("ail", 0.6, 1.0, 0.75, 0.0, symmetric=False)])
    st = vt.SolverSettings(solver_type=solver, n_panels=16, n_chord=nc)
    res = _quiet(control_derivatives, w, COND, st)
    d = res.derivatives["ail"]
    assert abs(d.CL) < 1e-9 and abs(d.Cm) < 1e-9
    # Right trailing edge down: more lift on the right, the right wing goes up (Cl < 0).
    assert d.Cl < -0.1
    right, left = res.hinge
    assert (right.moment.side, left.moment.side) == ("right", "left")
    assert right.moment.Ch == pytest.approx(left.moment.Ch, rel=1e-9)
    assert right.Ch_delta["ail"] < 0.0
    assert left.Ch_delta["ail"] == pytest.approx(-right.Ch_delta["ail"], rel=1e-6)
    assert right.Ch_alpha == pytest.approx(left.Ch_alpha, rel=1e-9)


def test_one_sided_difference_near_the_limit():
    """At +/-29.5 deg a central step of 1 deg would pass 30 deg: a second-order one-sided difference replaces it."""
    st = vt.SolverSettings(solver_type="linear", n_panels=12)
    d0 = np.radians(29.5)
    hi = _quiet(control_derivatives, _wing([vt.ControlSurface("f", 0.0, 1.0, 0.75, d0)]), COND, st,
                hinge_alpha=False)
    lo = _quiet(control_derivatives, _wing([vt.ControlSurface("f", 0.0, 1.0, 0.75, -d0)]), COND, st,
                hinge_alpha=False)
    assert hi.derivatives["f"].scheme == "backward"
    assert lo.derivatives["f"].scheme == "forward"
    # The formula, from three independent solves.
    step = np.radians(1.0)
    cl = [_quiet(vt.analyze, _wing([vt.ControlSurface("f", 0.0, 1.0, 0.75, d0 - k * step)]), COND, st).totals.CL
          for k in (0, 1, 2)]
    assert hi.derivatives["f"].CL == pytest.approx((3.0 * cl[0] - 4.0 * cl[1] + cl[2]) / (2.0 * step), rel=1e-12)
    # The same slope as a central difference with a half step.
    ref = _quiet(control_derivatives, _wing([vt.ControlSurface("f", 0.0, 1.0, 0.75, d0)]), COND, st,
                 step_deg=0.5, hinge_alpha=False)
    assert ref.derivatives["f"].scheme == "central"
    for name in ("CL", "Cm"):
        assert getattr(hi.derivatives["f"], name) == pytest.approx(getattr(ref.derivatives["f"], name), rel=1e-3)
    assert hi.hinge[0].Ch_delta["f"] == pytest.approx(ref.hinge[0].Ch_delta["f"], rel=1e-3)
    # Two controls with one name at +29.5 and -29.5 deg leave no room on either side.
    both = _wing([vt.ControlSurface("f", 0.0, 0.4, 0.75, d0), vt.ControlSurface("f", 0.5, 1.0, 0.75, -d0)])
    with pytest.raises(ValueError, match="no room"):
        control_derivatives(both, COND, st)


def test_moment_derivatives_in_stability_axes():
    """The moment derivatives in stability axes are the body-axis vector turned by alpha."""
    from ventorum.core.axes import rotation_body_to_stability

    w = _wing([vt.ControlSurface("ail", 0.6, 1.0, 0.75, 0.0, symmetric=False)])
    cond = vt.FlightCondition(V_inf=20.0, alpha=np.radians(8.0))
    st = vt.SolverSettings(solver_type="linear", n_panels=12)
    body = _quiet(control_derivatives, w, cond, st, hinge_alpha=False).derivatives["ail"]
    stab = _quiet(control_derivatives, w, cond, st, axes="stability", hinge_alpha=False).derivatives["ail"]
    expect = rotation_body_to_stability(cond.alpha) @ np.array([body.Cl, body.Cm, body.Cn])
    assert np.allclose([stab.Cl, stab.Cm, stab.Cn], expect, rtol=1e-10, atol=1e-12)
    assert stab.CL == body.CL


def test_two_controls_and_interference():
    """Two control names: one derivative set each, and the cross terms of the hinge moments."""
    w = _wing([vt.ControlSurface("flap", 0.0, 0.5, 0.75, np.radians(2.0)),
               vt.ControlSurface("ail", 0.6, 1.0, 0.75, 0.0, symmetric=False)])
    st = vt.SolverSettings(solver_type="vlm", n_panels=16, n_chord=4)
    res = _quiet(control_derivatives, w, COND, st, controls=["flap"])
    assert list(res.derivatives) == ["flap"]
    assert [(h.moment.control, h.moment.side) for h in res.hinge] == [
        ("flap", "right"), ("flap", "left"), ("ail", "right"), ("ail", "left")]
    assert all(set(h.Ch_delta) == {"flap"} for h in res.hinge)
    # The flap changes the load near the aileron: a small cross term.
    assert 0.0 < abs(res.hinge[2].Ch_delta["flap"]) < 0.2 * abs(res.hinge[0].Ch_delta["flap"])
    json.dumps(res.to_dict(), allow_nan=False)


def test_two_controls_with_one_name_on_one_surface():
    """Regression: two controls with one name on one surface keep separate hinge-moment derivatives."""
    w = _wing([vt.ControlSurface("f", 0.0, 0.3, 0.75, 0.0), vt.ControlSurface("f", 0.6, 1.0, 0.8, 0.0)])
    res = _quiet(control_derivatives, w, COND, vt.SolverSettings(solver_type="linear", n_panels=20))
    assert [h.moment.hinge_x_c for h in res.hinge] == [0.75, 0.75, 0.8, 0.8]
    # Section mode at zero deflection: dCh/ddelta of each copy is near its own thin-airfoil value.
    assert res.hinge[0].Ch_delta["f"] != pytest.approx(res.hinge[2].Ch_delta["f"], rel=1e-3)


def test_control_derivatives_in_ground_effect():
    """In ground effect the derivatives are finite and differ from free air."""
    w = _wing([vt.ControlSurface("flap", 0.0, 1.0, 0.75, 0.0)])
    st = vt.SolverSettings(solver_type="vlm", n_panels=12, n_chord=4)
    free = _quiet(control_derivatives, w, COND, st, hinge_alpha=False)
    ge = _quiet(control_derivatives, w, vt.FlightCondition(V_inf=20.0, alpha=np.radians(3.0), h=0.4), st,
                hinge_alpha=False)
    assert np.isfinite(ge.derivatives["flap"].CL)
    assert ge.derivatives["flap"].CL != pytest.approx(free.derivatives["flap"].CL, rel=1e-3)
    assert ge.hinge[0].moment.Ch != pytest.approx(free.hinge[0].moment.Ch, rel=1e-3)


def test_control_derivatives_input_errors():
    w = _wing([vt.ControlSurface("f", 0.0, 1.0, 0.75, 0.0)])
    st = vt.SolverSettings(solver_type="linear", n_panels=8)
    with pytest.raises(ValueError, match="Fourier"):
        control_derivatives(w, COND, vt.SolverSettings(solver_type="fourier", n_panels=8))
    with pytest.raises(ValueError, match="known control names"):
        control_derivatives(w, COND, st, controls=["rudder"])
    with pytest.raises(ValueError, match="unique"):
        control_derivatives(w, COND, st, controls=["f", "f"])
    for step in (0.0, -1.0, 11.0, float("nan")):
        with pytest.raises(ValueError, match="step_deg"):
            control_derivatives(w, COND, st, step_deg=step)
    with pytest.raises(ValueError, match="axes"):
        control_derivatives(w, COND, st, axes="all")
    with pytest.raises(ValueError, match="no control"):
        control_derivatives(_wing([]), COND, st)


def test_control_derivatives_do_not_change_the_input():
    c = vt.ControlSurface("f", 0.0, 1.0, 0.75, np.radians(3.0))
    ac = vt.Aircraft(surfaces=[_wing([c])])
    _quiet(control_derivatives, ac, COND, vt.SolverSettings(solver_type="linear", n_panels=8))
    assert ac.surfaces[0].controls[0].deflection == np.radians(3.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. GPU
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.gpu
@pytest.mark.parametrize("solver_type", ["vlm", "linear"])
def test_gpu_hinge_moments_equal_cpu(solver_type):
    """The hinge moments of a GPU result (float64) equal those of the CPU result."""
    if not gpu.available():
        pytest.skip(f"GPU pipeline unavailable: {gpu.unavailable_reason()}")
    old_dev, old_prec = gpu.get_device(), gpu._precision
    w = _wing([vt.ControlSurface("flap", 0.0, 0.5, 0.75, np.radians(4.0)),
               vt.ControlSurface("ail", 0.6, 1.0, 0.75, np.radians(3.0), symmetric=False)])
    st = vt.SolverSettings(solver_type=solver_type, n_panels=12, n_chord=4 if solver_type == "vlm" else 1)
    try:
        gpu.set_precision("float64")
        gpu.set_device("cpu")
        m_cpu = hinge_moments(_quiet(vt.analyze, w, COND, st))
        gpu.set_device("gpu")
        r = _quiet(vt.analyze, w, COND, st)
        assert r.details["device"] == "gpu"
        m_gpu = hinge_moments(r)
    finally:
        gpu.set_device(old_dev)
        gpu.set_precision(old_prec)
    for a, b in zip(m_cpu, m_gpu):
        assert (a.control, a.side) == (b.control, b.side)
        assert b.Ch == pytest.approx(a.Ch, rel=1e-9, abs=1e-12)
