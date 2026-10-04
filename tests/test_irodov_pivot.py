"""Irodov height-pitch margin: one value for one flight state (T-0014).

The pitch and height aerodynamic centres are taken about the moment
reference point, with the height of that point held fixed (Rozhdestvensky,
Aerodynamics of a Lifting System in Extreme Ground Effect). ``height_ref``
only sets where the given height is measured.
"""

import numpy as np
import pytest

import ventorum as vt
from ventorum.ground_effect import GroundEffectSweep, analyze_ground_effect

C = 1.0
REF = np.array([0.3, 0.0, 0.0])
H0 = 0.25            # height of the reference point [m]
A0 = 4.0             # angle of attack [deg]
SETT = vt.SolverSettings(n_panels=12, n_chord=4)


def _wing():
    return vt.LiftingSurface(
        name="wing", semi_span=1.5, sections=[vt.WingSection(0.0, C), vt.WingSection(1.0, C)],
    )


def _ac():
    return vt.Aircraft(surfaces=[_wing()], S_ref=3.0, b_ref=3.0, c_ref=C)


def _height_in(href, alpha_deg=A0, h_ref_point=H0):
    """Height in the convention *href* of the state with the reference point at *h_ref_point*."""
    r = analyze_ground_effect(_ac(), h=h_ref_point, alpha_deg=alpha_deg, ref_point=REF, height_ref="ref",
                              settings=SETT, compute_strike_limit=False)
    if href == "ref":
        return h_ref_point
    if href == "min":
        return r.h_min
    # Root quarter-chord or trailing-edge point: find h by its offset to the reference point.
    probe = analyze_ground_effect(_ac(), h=1.0, alpha_deg=alpha_deg, ref_point=REF, height_ref=href,
                                  settings=SETT, compute_strike_limit=False)
    return h_ref_point + (1.0 - probe.h_ref)


def _margin(href, dh=0.02, da=0.01):
    h = _height_in(href)
    sw = GroundEffectSweep(_ac(), settings=SETT, backend="serial").run_sweep(
        [h - dh, h, h + dh], [A0 - da, A0, A0 + da], [0.0], ref_point=REF, height_ref=href,
        compute_strike_limit=False)
    assert sw.results[4] is not None
    assert sw.results[4].h_ref == pytest.approx(H0, abs=1e-12)    # the same flight state
    d = sw.compute_stability_derivatives()
    return d, sw


@pytest.mark.parametrize("href", ["min", "qc", "te"])
def test_irodov_margin_independent_of_height_ref(href):
    """Small steps: the remaining difference is the truncation error of the differences.

    The correction term of the chain rule multiplies the height derivative,
    so its truncation error (step squared) enters only the non-'ref' grids.
    """
    ref, _ = _margin("ref", dh=0.002)
    other, _ = _margin(href, dh=0.002)
    assert other["irodov_margin"][1, 1] == pytest.approx(ref["irodov_margin"][1, 1], abs=1e-6)
    assert other["x_alpha"][1, 1] == pytest.approx(ref["x_alpha"][1, 1], abs=1e-6)
    assert other["x_h"][1, 1] == pytest.approx(ref["x_h"][1, 1], abs=1e-6)
    assert np.allclose(other["pivot_point_m"], REF)


def test_irodov_margin_with_coarse_steps_is_close_for_each_height_ref():
    """The 4th review measured -0.131 and -0.118 for one state; the difference is now truncation only."""
    m = [_margin(href, dh=0.05, da=1.0)[0]["irodov_margin"][1, 1] for href in ("ref", "te")]
    assert abs(m[0] - m[1]) < 2e-3


def test_irodov_against_finite_differences():
    """x_alpha and x_h match central differences of the solver about the reference point."""
    d, _ = _margin("te")

    def coeffs(h, a):
        r = analyze_ground_effect(_ac(), h=h, alpha_deg=a, ref_point=REF, height_ref="ref",
                                  settings=SETT, compute_strike_limit=False)
        return r.CL, r.Cm

    eh, ea = 1e-4, 1e-3
    (lp, mp), (lm, mm) = coeffs(H0 + eh, A0), coeffs(H0 - eh, A0)
    x_h = -C * (mp - mm) / (lp - lm)
    (lp, mp), (lm, mm) = coeffs(H0, A0 + ea), coeffs(H0, A0 - ea)
    x_alpha = -C * (mp - mm) / (lp - lm)
    assert d["x_h"][1, 1] == pytest.approx(x_h, abs=2e-4)
    assert d["x_alpha"][1, 1] == pytest.approx(x_alpha, abs=2e-4)
    assert d["irodov_margin"][1, 1] == pytest.approx((x_alpha - x_h) / C, abs=4e-4)


def test_sweep_without_reference_heights_refuses_the_pivot_change():
    """A grid in another height convention without the reference-point heights gives no margin."""
    _, sw = _margin("te")
    sw.h_ref_grid = None
    d = sw.compute_stability_derivatives()
    assert np.all(np.isnan(d["irodov_margin"]))
    assert "reference point" in d["note"]


def test_agent_tool_margin_independent_of_height_ref():
    """The ground_effect tool gives one margin for one flight state (4th review, item 8)."""
    from ventorum.agent import ground_effect

    wing = {"span_m": 3.0, "chord_m": C}
    st = {"n_panels": 12, "n_chord": 4}
    hs = [H0 - 0.05, H0, H0 + 0.05]
    off = _height_in("te") - H0
    p_ref = ground_effect(wing, hs, alpha_deg=A0, height_ref="ref", ref_point_m=REF.tolist(), settings=st)
    p_te = ground_effect(wing, [h + off for h in hs], alpha_deg=A0, height_ref="te",
                         ref_point_m=REF.tolist(), settings=st)
    m_ref, m_te = (p["irodov"]["rows"][1]["irodov_margin"] for p in (p_ref, p_te))
    assert m_ref is not None and m_te is not None
    assert abs(m_te - m_ref) < 2e-3          # truncation of the 1 deg and 0.05 m steps only
    assert p_te["irodov"]["pivot_point_m"] == pytest.approx(REF.tolist())
