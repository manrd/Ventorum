# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Consistency of the Prandtl-Glauert path: section moment, section slope, Mach and speed, Mach limit.

The Prandtl-Glauert rule (H. Glauert, Proc. R. Soc. Lond. A 118, 1928)
scales every section pressure coefficient by ``1 / beta`` with
``beta = sqrt(1 - M^2)``. The tests check that the constant section moment
``Cm0`` of a linear airfoil and the section slope of the effective angle
follow it, that a stated Mach number is compared with the speed, that a
Mach number of 0.7 or more is refused, and that mach 0 keeps the
incompressible results to the last bit.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import ventorum as vt
from ventorum import gpu
from ventorum.core.constants import A_SL, MACH_LIMIT_INCOMPRESSIBLE
from ventorum.core.datatypes import FlightCondition, LinearAirfoil, LiftingSurface, SolverSettings, WingSection
from ventorum.core.errors import ValidityError
from ventorum.solvers.factory import make_solver


def _high_ar_wing(airfoil: LinearAirfoil) -> LiftingSurface:
    """Return a rectangular wing of aspect ratio 1000 (chord 1 m, span 1000 m)."""
    return LiftingSurface(
        name="HighAR",
        semi_span=500.0,
        sections=[WingSection(y_frac=0.0, chord=1.0, airfoil=airfoil),
                  WingSection(y_frac=1.0, chord=1.0, airfoil=airfoil)],
    )


def _analyze(surface, condition, settings):
    """Run analyze and ignore the envelope warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return vt.analyze(surface, condition=condition, settings=settings)


def _spanwise(res):
    return res.spanwise[0] if isinstance(res.spanwise, list) else res.spanwise


def test_linear_cm0_scales_with_one_over_beta():
    """The Cm0 of a linear airfoil at Mach 0.5 is the Mach 0 value divided by beta."""
    wing = _high_ar_wing(LinearAirfoil(name="cm", a0=2.0 * np.pi, alpha_L0=0.0, Cm0=-0.1))
    settings = SolverSettings(solver_type="vlm", n_panels=8, n_chord=2)
    mach = 0.5
    beta = math.sqrt(1.0 - mach * mach)
    res0 = _analyze(wing, FlightCondition(V_inf=50.0, alpha=0.0, mach=0.0), settings)
    res5 = _analyze(wing, FlightCondition(V_inf=mach * A_SL, alpha=0.0, mach=mach), settings)
    cm0 = _spanwise(res0).Cm_section
    cm5 = _spanwise(res5).Cm_section
    np.testing.assert_allclose(cm0, -0.1, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(cm5, cm0 / beta, rtol=1e-14)
    # No lift (alpha = alpha_L0 = 0): the total Cm is the section moment.
    assert abs(res0.totals.CL) < 1e-12 and abs(res5.totals.CL) < 1e-12
    assert res0.totals.Cm == pytest.approx(-0.1, rel=1e-10)
    assert res5.totals.Cm == pytest.approx(-0.1 / beta, rel=1e-10)


def test_section_angle_uses_compressible_slope():
    """alpha_eff at Mach 0.5 is alpha_L0 + Cl beta / a0; alpha_i of a wing of AR 1000 is small and positive."""
    a0 = 2.0 * np.pi
    aL0 = np.radians(-1.0)
    wing = _high_ar_wing(LinearAirfoil(name="cam", a0=a0, alpha_L0=aL0))
    settings = SolverSettings(solver_type="vlm", n_panels=8, n_chord=2)
    mach = 0.5
    beta = math.sqrt(1.0 - mach * mach)
    alpha = np.radians(4.0)
    res = _analyze(wing, FlightCondition(V_inf=mach * A_SL, alpha=alpha, mach=mach), settings)
    sp = _spanwise(res)
    # Hand value of the effective angle with the compressible slope a0 / beta.
    np.testing.assert_allclose(sp.alpha_eff, aL0 + sp.Cl * beta / a0, rtol=0.0, atol=1e-15)
    # Centre strip: the wing of AR 1000 is nearly two-dimensional, so
    # alpha_i is about CL / (pi AR) = 0.01 deg here. The incompressible
    # slope a0 would give about (alpha - alpha_L0)(1 - 1 / beta) = -0.77 deg.
    i_mid = int(np.argmin(np.abs(sp.y)))
    ai_deg = float(np.degrees(sp.alpha_i[i_mid]))
    hand_deg = float(np.degrees(res.totals.CL / (np.pi * 1000.0)))
    assert 0.0 < ai_deg < 3.0 * hand_deg, (ai_deg, hand_deg)


def test_speed_and_mach_mismatch_warns_and_notes():
    """V 250 m/s with mach 0.2 gives a warning and a trust note; a matching speed gives neither."""
    wing = _high_ar_wing(LinearAirfoil())
    settings = SolverSettings(solver_type="vlm", n_panels=8, n_chord=1)
    cond = FlightCondition(V_inf=250.0, alpha=np.radians(2.0), mach=0.2)
    with pytest.warns(RuntimeWarning, match="does not agree with V_inf/a"):
        res = vt.analyze(wing, condition=cond, settings=settings)
    notes = [w for w in res.totals.trust.warnings if "does not agree with V_inf/a" in w]
    assert len(notes) == 1
    assert "340.29 m/s" in notes[0] and "ISA sea level" in notes[0]

    cond_ok = FlightCondition(V_inf=0.2 * A_SL, alpha=np.radians(2.0), mach=0.2)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        res_ok = vt.analyze(wing, condition=cond_ok, settings=settings)
    assert not any("does not agree" in w for w in res_ok.totals.trust.warnings)


def test_mach_zero_fast_gets_the_stated_mach_penalty():
    """mach 0 with V_inf/a above 0.3 warns and gets the penalty of the same stated Mach number."""
    wing = _high_ar_wing(LinearAirfoil())
    settings = SolverSettings(solver_type="vlm", n_panels=8, n_chord=1)
    m = 0.5
    with pytest.warns(RuntimeWarning, match="incompressible limit"):
        res0 = vt.analyze(wing, condition=FlightCondition(V_inf=m * A_SL, alpha=np.radians(2.0), mach=0.0),
                          settings=settings)
    res_m = _analyze(wing, FlightCondition(V_inf=m * A_SL, alpha=np.radians(2.0), mach=m), settings)
    expected = 0.60 + 0.30 * min(1.0, (m - MACH_LIMIT_INCOMPRESSIBLE) / 0.35)
    assert res0.totals.trust.factors["compressibility"] == pytest.approx(expected, rel=1e-12)
    assert res_m.totals.trust.factors["compressibility"] == pytest.approx(expected, rel=1e-12)
    assert res0.totals.trust.rating == "UNRELIABLE"


@pytest.mark.parametrize("mach", [0.7, 0.85, 0.9999])
def test_mach_at_or_above_limit_is_refused(mach):
    """Regression: mach 0.9999 ran and gave a result; mach >= 0.7 now raises ValidityError."""
    wing = _high_ar_wing(LinearAirfoil())
    settings = SolverSettings(solver_type="vlm", n_panels=8, n_chord=1)
    cond = FlightCondition(V_inf=mach * A_SL, alpha=np.radians(2.0), mach=mach)
    with pytest.raises(ValidityError, match="incompressible flow"):
        vt.analyze(wing, condition=cond, settings=settings)
    # The solver refuses it too when it is called without analyze.
    with pytest.raises(ValidityError, match="incompressible flow"):
        make_solver("vlm").solve(wing, cond, settings)


def test_mach_below_limit_still_solves():
    """mach 0.69 is UNRELIABLE but still solves."""
    wing = _high_ar_wing(LinearAirfoil())
    settings = SolverSettings(solver_type="vlm", n_panels=8, n_chord=1)
    res = _analyze(wing, FlightCondition(V_inf=0.69 * A_SL, alpha=np.radians(2.0), mach=0.69), settings)
    assert np.isfinite(res.totals.CL)
    assert res.totals.trust.rating == "UNRELIABLE"


def _stored_case():
    af = LinearAirfoil(name="c", a0=6.0, alpha_L0=np.radians(-2.0), Cd0=0.008, Cm0=-0.08)
    wing = LiftingSurface(
        name="W", semi_span=4.0, sweep_le=np.radians(10.0),
        sections=[WingSection(y_frac=0.0, chord=1.2, airfoil=af),
                  WingSection(y_frac=1.0, chord=0.8, airfoil=af)],
    )
    return wing, SolverSettings(solver_type="vlm", n_panels=12, n_chord=3)


@pytest.mark.parametrize("mach", [None, 0.0])
def test_mach_zero_results_bit_identical(mach):
    """mach None and 0 give the stored values of the code before this change, to the last bit."""
    wing, settings = _stored_case()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        res = vt.analyze(wing, condition=FlightCondition(V_inf=50.0, alpha=np.radians(4.0), mach=mach),
                         settings=settings)
    assert res.totals.CL == 0.47165222500722925
    assert res.totals.Cm == -0.34544423447247635
    assert res.totals.CD_total == 0.01700951581976442
    assert float(_spanwise(res).alpha_i[0]) == 0.0939928027825337
    assert res.totals.trust.score == 1.0
    assert not any("does not agree" in w for w in res.totals.trust.warnings)


@pytest.mark.gpu
@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_gpu_section_terms_match_cpu_with_mach(precision):
    """The GPU loads give the Cm0 / beta and the a0 / beta section angle of the CPU loads."""
    if not gpu.available():
        pytest.skip(f"The GPU pipelines cannot run: {gpu.unavailable_reason()}")
    wing, settings = _stored_case()
    solver = vt.VortexLatticeSolver()
    cond = FlightCondition(V_inf=0.5 * A_SL, alpha=np.radians(4.0), mach=0.5)
    old_dev, old_prec = gpu.get_device(), gpu._precision
    try:
        gpu.set_device("cpu")
        ref = solver.solve(wing, cond, settings)
        gpu.set_device("gpu")
        gpu.set_precision(precision)
        new = solver.solve(wing, cond, settings)
    finally:
        gpu.set_device(old_dev)
        gpu.set_precision(old_prec)
    tol = {"float64": 1e-11, "float32": 2e-5}[precision]
    sp_r, sp_n = _spanwise(ref), _spanwise(new)
    np.testing.assert_allclose(sp_n.Cm_section, sp_r.Cm_section, rtol=0.0, atol=tol)
    np.testing.assert_allclose(sp_n.alpha_eff, sp_r.alpha_eff, rtol=0.0, atol=tol)
    assert abs(new.totals.Cm - ref.totals.Cm) <= tol * max(abs(ref.totals.CL), 1e-3)
