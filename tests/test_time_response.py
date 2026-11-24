# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for the public time-domain response, gust fields and indicial helper.

Covers round-off parity between ``ventorum.time_response`` and
``_solve_time_rings``, zero-gust parity, uniform time-varying gust equivalence
with prescribed translation and heave, Wagner and Kussner indicial responses
on an AR 40 wing, 1-cosine gust superposition against the solver's own
sharp-edged response, steady linear horizontal shear in the long-time limit,
ground-effect mirror-wing parity and wake ground clearance, control-step
convergence to ``ventorum.control_derivatives``, strict JSON ``to_dict()``,
``store`` options, trust and diagnostic notes, and all input checks.

References
----------
* J. Katz and A. Plotkin, "Low-Speed Aerodynamics", 2nd ed., Cambridge
  University Press, 2001, chapter 13.
* R. T. Jones, "The Unsteady Lift of a Wing of Finite Aspect Ratio", NACA
  Report 681, 1940 (two-exponential approximation to the Wagner function).
* W. R. Sears and B. O. Sparks, "On the Reaction of an Elastic Wing to
  Vertical Gusts", Journal of the Aeronautical Sciences 9(2), 1941, pp. 64-67
  (two-exponential approximation to the Kussner function; also in
  R. L. Bisplinghoff, H. Ashley and R. L. Halfman, "Aeroelasticity",
  Addison-Wesley, 1955, chapter 5).
* R. B. Stull, "An Introduction to Boundary Layer Meteorology", Kluwer
  Academic Publishers, 1988, chapter 9 (logarithmic surface-layer wind
  profile).
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from scipy import integrate, special

import ventorum as vt
from ventorum.aero import vortex_ring as VR
from ventorum.aero.loads import leg_force_rules, leg_segments, leg_strength
from ventorum.aero.system import (
    freestream_direction,
    lift_direction,
    make_ground_plane,
    side_direction,
)
from ventorum.core.errors import InputError
from ventorum.geometry.lattice import build_lattice
from ventorum.unsteady import (
    IndicialResult,
    TimeResponseResult,
    indicial_response,
    linear_shear,
    logarithmic_shear,
    one_minus_cosine,
    sharp_edged,
    time_response,
    wind_shear,
)
from ventorum.unsteady.harmonic import _rings_to_horseshoe
from ventorum.unsteady.rings import build_ring_lattice
from ventorum.unsteady.timedomain import _solve_time_rings
from ventorum.utils.vec import cross3


def _rect_wing(
    semi_span: float = 3.0,
    chord: float = 1.0,
    n_panels: int = 6,
    *,
    flap_deflection_deg: float | None = None,
    flap_name: str = "flap",
    hinge_x_c: float = 0.75,
) -> vt.Aircraft:
    """Build a flat rectangular wing with quarter-chord moment reference."""
    controls = []
    if flap_deflection_deg is not None:
        controls.append(
            vt.ControlSurface(
                name=flap_name,
                eta_start=0.0,
                eta_end=1.0,
                hinge_x_c=hinge_x_c,
                deflection=math.radians(flap_deflection_deg),
            )
        )
    wing = vt.LiftingSurface(
        name="wing",
        semi_span=semi_span,
        n_panels=n_panels,
        sections=[
            vt.WingSection(y_frac=0.0, chord=chord),
            vt.WingSection(y_frac=1.0, chord=chord),
        ],
        controls=controls,
    )
    ac = vt.Aircraft(surfaces=[wing], ref_point=np.array([0.25 * chord, 0.0, 0.0]))
    ac.compute_reference_values()
    return ac


def _wagner_jones(s: float) -> float:
    """Return the R. T. Jones (1940, NACA Report 681) Wagner approximation [-]."""
    return 1.0 - 0.165 * math.exp(-0.0455 * s) - 0.335 * math.exp(-0.300 * s)


def _kussner_sears_sparks(s: float) -> float:
    """Return the Sears-Sparks (1941) / Bisplinghoff et al. (1955) Kussner approximation [-]."""
    return 1.0 - 0.500 * math.exp(-0.130 * s) - 0.500 * math.exp(-1.000 * s)


def _kussner_exact_2d(s: float) -> float:
    """Return the exact 2D Kussner function psi(s) [-] via the Fourier integral of Sears' function."""
    def _integrand(k: float) -> float:
        h1 = special.hankel2(1, k)
        h0 = special.hankel2(0, k)
        c_k = h1 / (h1 + 1.0j * h0)
        j0 = special.j0(k)
        j1 = special.j1(k)
        s_le = ((j0 - 1.0j * j1) * c_k + 1.0j * j1) * np.exp(-1.0j * k)
        return float(s_le.imag / k)

    val, _ = integrate.quad(_integrand, 1.0e-8, 200.0, weight="cos", wvar=s, epsabs=1.0e-6, epsrel=1.0e-6, limit=500)
    return 1.0 + (2.0 / math.pi) * val


# ---------------------------------------------------------------------------
# 1. Exports and parity with _solve_time_rings
# ---------------------------------------------------------------------------


def test_public_exports():
    """Public time-response, indicial and gust symbols are exported from ventorum and ventorum.unsteady."""
    for name, obj in (
        ("time_response", time_response),
        ("TimeResponseResult", TimeResponseResult),
        ("indicial_response", indicial_response),
        ("IndicialResult", IndicialResult),
        ("sharp_edged", sharp_edged),
        ("one_minus_cosine", one_minus_cosine),
        ("wind_shear", wind_shear),
        ("linear_shear", linear_shear),
        ("logarithmic_shear", logarithmic_shear),
    ):
        assert getattr(vt, name) is obj
        assert name in vt.unsteady.__all__


@pytest.mark.parametrize("h_ground", [None, 0.5])
def test_public_time_response_equals_solve_time_rings_roundoff(h_ground):
    """time_response equals _solve_time_rings on the same lattice to round-off with motion and flap."""
    ac = _rect_wing(semi_span=3.0, chord=1.0, n_panels=4, flap_deflection_deg=0.0)
    st = vt.SolverSettings(solver_type="vlm", n_panels=4, n_chord=4, spacing="half-cosine", chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=np.radians(4.0), h=h_ground)

    n_steps = 6
    omega = 2.0 * 0.25 * cond.V_inf / ac.c_ref
    motion = {
        "x": lambda t: 0.02 * math.sin(omega * t),
        "y": lambda t: 0.015 * math.sin(0.8 * omega * t),
        "z": lambda t: 0.03 * math.sin(omega * t),
        "roll": lambda t: math.radians(1.5) * math.sin(0.9 * omega * t),
        "pitch": lambda t: math.radians(2.0) * math.sin(omega * t + 0.2),
        "yaw": lambda t: math.radians(1.0) * math.sin(0.7 * omega * t),
    }
    controls = {"flap": lambda t: math.radians(6.0) * math.sin(omega * t + 0.3)}

    res_pub = vt.time_response(
        ac,
        cond,
        st,
        n_steps=n_steps,
        motion=motion,
        controls=controls,
        store="strengths",
    )

    lat = build_lattice(ac, st, collocation="vlm", n_chord=4, chord_spacing="uniform")
    ground = (
        make_ground_plane(lat, h_ground, cond.alpha, cond.beta, ref_point=ac.ref_point, height_ref="ref")
        if h_ground is not None
        else None
    )
    res_priv = _solve_time_rings(
        lat,
        cond,
        S_ref=ac.S_ref,
        b_ref=ac.b_ref,
        c_ref=ac.c_ref,
        ref_point=ac.ref_point,
        motion=motion,
        controls=controls,
        n_steps=n_steps,
        ground=ground,
    )

    assert isinstance(res_pub, TimeResponseResult)
    assert np.allclose(res_pub.time, res_priv.time, atol=1e-15, rtol=0.0)
    assert np.allclose(res_pub.s, 2.0 * cond.V_inf * res_priv.time / ac.c_ref, atol=1e-15, rtol=0.0)
    for name in (
        "CL", "CD", "CY", "Cl", "Cm", "Cn", "strip_cl",
        "x", "y", "z", "roll", "pitch", "yaw",
        "x_dot", "y_dot", "z_dot", "roll_dot", "pitch_dot", "yaw_dot",
        "gamma_bound", "gamma_wake", "wake_corners",
    ):
        a = getattr(res_pub, name)
        b = getattr(res_priv, name)
        scale = max(float(np.max(np.abs(b))), 1.0)
        assert np.max(np.abs(a - b)) <= 1e-12 * scale, f"{name} mismatch at h={h_ground}"
    assert np.max(np.abs(res_pub.controls["flap"] - res_priv.controls["flap"])) <= 1e-15
    assert np.max(np.abs(res_pub.controls_dot["flap"] - res_priv.controls_dot["flap"])) <= 1e-15
    assert np.all(res_pub.gust_velocity == 0.0)


# ---------------------------------------------------------------------------
# 2. Zero-gust parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("h_ground", [None, 0.5])
def test_zero_gust_matches_no_gust_roundoff(h_ground):
    """gust=None and a gust returning zero everywhere give identical results to round-off."""
    ac = _rect_wing(semi_span=3.0, chord=1.0, n_panels=4)
    st = vt.SolverSettings(solver_type="vlm", n_panels=4, n_chord=4, chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=np.radians(4.0), h=h_ground)
    motion = {"pitch": lambda t: math.radians(2.0) * math.sin(6.0 * t)}

    r_none = vt.time_response(ac, cond, st, n_steps=8, motion=motion, gust=None, store="strengths")
    r_zero_fn = vt.time_response(
        ac,
        cond,
        st,
        n_steps=8,
        motion=motion,
        gust=lambda pts, t: np.zeros_like(pts, dtype=np.float64),
        store="strengths",
    )
    r_zero_se = vt.time_response(
        ac,
        cond,
        st,
        n_steps=8,
        motion=motion,
        gust=sharp_edged(0.0, "vertical"),
        store="strengths",
    )

    for r_other in (r_zero_fn, r_zero_se):
        for name in ("CL", "CD", "CY", "Cl", "Cm", "Cn", "strip_cl", "gamma_bound", "gamma_wake", "wake_corners"):
            a = getattr(r_none, name)
            b = getattr(r_other, name)
            scale = max(float(np.max(np.abs(a))), 1.0)
            assert np.max(np.abs(a - b)) <= 1e-12 * scale, f"{name} mismatch with zero gust"


# ---------------------------------------------------------------------------
# 3. Uniform time-varying gust equivalence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("h_ground", [None, 0.5])
def test_uniform_headwind_gust_equals_prescribed_x_translation_roundoff(h_ground):
    """A uniform, time-varying head-wind gust g(t) equals translation rate x_dot = -g to round-off.

    This equivalence holds in free air and over flat ground (h/c = 0.5) because
    translating the wing along the mean flight path e_x (which is parallel to
    the ground plane) by x(t) = -integral_0^t g(tau) d(tau) produces the exact
    same local onset velocity V_0 + g(t) e_x = V_0 - x_dot(t) e_x and the exact
    same wake geometry relative to the wing (and to the ground mirror wing) at
    every time step.
    """
    ac = _rect_wing(semi_span=3.0, chord=1.0, n_panels=4)
    st = vt.SolverSettings(solver_type="vlm", n_panels=4, n_chord=4, chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=np.radians(4.0), h=h_ground)
    d_0 = freestream_direction(cond.alpha, cond.beta)

    # Time-varying linear head-wind gust g(t) = g0 + g1 * t [m/s].
    # Exact integral x(t) = -(g0 * t + 0.5 * g1 * t^2) has x_dot(t) = -g(t).
    g0, g1 = 1.5, 4.0
    n_steps = 8

    def head_gust(pts: np.ndarray, t: float) -> np.ndarray:
        speed = g0 + g1 * t
        return np.broadcast_to(speed * d_0[None, :], pts.shape).copy()

    res_gust = vt.time_response(ac, cond, st, n_steps=n_steps, gust=head_gust, store="strengths")
    t_grid = res_gust.time
    x_arr = -(g0 * t_grid + 0.5 * g1 * t_grid ** 2)
    res_trans = vt.time_response(ac, cond, st, n_steps=n_steps, motion={"x": x_arr}, store="strengths")

    for name in ("CL", "CD", "CY", "Cl", "Cm", "Cn", "strip_cl", "gamma_bound", "gamma_wake"):
        a = getattr(res_gust, name)
        b = getattr(res_trans, name)
        scale = max(float(np.max(np.abs(b))), 1.0)
        assert np.max(np.abs(a - b)) <= 1e-12 * scale, f"Head-wind equivalence mismatch in {name} at h={h_ground}"


def test_uniform_vertical_gust_equals_prescribed_heave_roundoff_free_air():
    """A uniform, time-varying vertical gust w(t) equals heave rate z_dot = -w to round-off in free air.

    This equivalence holds in free air because translating the wing along the
    lift direction e_z by z(t) = -integral_0^t w(tau) d(tau) produces the exact
    same local onset velocity V_0 + w(t) e_z = V_0 - z_dot(t) e_z and the exact
    same wake geometry relative to the wing at every time step.
    """
    ac = _rect_wing(semi_span=3.0, chord=1.0, n_panels=4)
    st = vt.SolverSettings(solver_type="vlm", n_panels=4, n_chord=4, chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=np.radians(4.0))
    L_0 = lift_direction(cond.alpha)

    w0, w1 = 0.8, 3.0
    n_steps = 8

    def vert_gust(pts: np.ndarray, t: float) -> np.ndarray:
        speed = w0 + w1 * t
        return np.broadcast_to(speed * L_0[None, :], pts.shape).copy()

    res_gust = vt.time_response(ac, cond, st, n_steps=n_steps, gust=vert_gust, store="strengths")
    t_grid = res_gust.time
    z_arr = -(w0 * t_grid + 0.5 * w1 * t_grid ** 2)
    res_heave = vt.time_response(ac, cond, st, n_steps=n_steps, motion={"z": z_arr}, store="strengths")

    for name in ("CL", "CD", "CY", "Cl", "Cm", "Cn", "strip_cl", "gamma_bound", "gamma_wake"):
        a = getattr(res_gust, name)
        b = getattr(res_heave, name)
        scale = max(float(np.max(np.abs(b))), 1.0)
        assert np.max(np.abs(a - b)) <= 1e-12 * scale, f"Vertical gust equivalence mismatch in {name}"


# ---------------------------------------------------------------------------
# 4. Wagner (kind="alpha") and 5. Kussner (kind="gust") on AR 40 wing
# ---------------------------------------------------------------------------


def test_indicial_wagner_ar40_matches_jones_approximation():
    """indicial_response(kind='alpha') on AR 40 wing matches Jones (1940) Wagner function.

    Uses the exact tolerance of the existing Wagner test in
    tests/test_unsteady_timedomain.py: 0.035 on cl_mid(s) / cl_mid_steady
    (accounting for ~0.027 AR 40 tip-relief lift shift plus ~0.008 discrete
    time-step and two-exponential error) and 0.015 on cl_mid(s) / (2 pi).
    """
    ac = _rect_wing(semi_span=20.0, chord=1.0, n_panels=20)
    st = vt.SolverSettings(solver_type="vlm", n_panels=20, n_chord=8, spacing="half-cosine", chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=0.0)

    res = vt.indicial_response(ac, cond, st, kind="alpha", s_end=10.0, amplitude=1.0e-4)
    assert isinstance(res, IndicialResult)
    assert res.steady is not None

    # Mid-span steady strip lift slope [1/rad] on the same lattice.
    lat = build_lattice(ac, st, collocation="vlm", n_chord=8, chord_spacing="uniform")
    mid_strip = int(np.argmin(np.abs(lat.qc_mid[:, 1])))
    steady_1 = vt.analyze(ac, vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=1.0e-4), st)
    cl_mid_steady_per_rad = float(steady_1.spanwise[0].Cl[mid_strip]) / 1.0e-4

    for s_target, step_idx in ((1.0, 4), (2.0, 8), (5.0, 20), (10.0, 40)):
        assert abs(res.s[step_idx] - s_target) <= 1e-12
        cl_mid = float(res.strip_cl[step_idx, mid_strip])
        ratio_steady = cl_mid / cl_mid_steady_per_rad
        ratio_2d = cl_mid / (2.0 * math.pi)
        phi_jones = _wagner_jones(s_target)
        assert abs(ratio_steady - phi_jones) <= 0.035, (
            f"s={s_target}: ratio_steady={ratio_steady:.5f}, Jones={phi_jones:.5f}"
        )
        assert abs(ratio_2d - phi_jones) <= 0.015, (
            f"s={s_target}: ratio_2d={ratio_2d:.5f}, Jones={phi_jones:.5f}"
        )


def test_indicial_kussner_ar40_matches_sears_sparks_approximation():
    """indicial_response(kind='gust') on AR 40 wing matches published Kussner theory.

    Published two-exponential approximation:
    psi_SS(s) = 1 - 0.500 exp(-0.130 s) - 0.500 exp(-1.000 s)
    (W. R. Sears and B. O. Sparks, J. Aero. Sci. 9(2), 1941; R. L. Bisplinghoff,
    H. Ashley and R. L. Halfman, "Aeroelasticity", 1955, chapter 5).
    Also checked directly against the exact 2D Kussner function psi_exact(s)
    from the Fourier integral of Sears' function.

    Tolerance derivation at s = 1, 2, 5, 10 for AR 40 and n_chord = 8
    (ds = 0.25 semichords):
    - Finite AR 40 lowers steady mid-span lift by ~2.7 % relative to 2 pi,
      raising cl_mid(s) / cl_mid_steady by ~0.012 to ~0.024 (3D tip relief);
    - Discretising a sharp-edged gust front over 8 chordwise panels (dx = c/8,
      collocation point at 0.75 dx) gives an O(ds) front-sampling lead of up to
      ~0.027 while the front crosses the chord (s = 1, 2) and < 0.005 afterward;
    - Against exact 2D Kussner psi_exact(s): tolerance 0.045 on cl_mid / cl_mid_steady
      and 0.030 on cl_mid / (2 pi) at all four stations;
    - Against the Sears-Sparks two-exponential formula, which itself underpredicts
      exact 2D Kussner by 0.040 at s = 1 (0.3770 vs 0.4168) and <= 0.008 at
      s = 2, 5, 10: tolerance is 0.075 at s = 1, 0.055 at s = 2, and 0.035 at
      s = 5, 10 on cl_mid / cl_mid_steady.
    At s = 0, cl_mid(0) = 0 identically.
    """
    ac = _rect_wing(semi_span=20.0, chord=1.0, n_panels=20)
    st = vt.SolverSettings(solver_type="vlm", n_panels=20, n_chord=8, spacing="half-cosine", chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=0.0)

    res = vt.indicial_response(ac, cond, st, kind="gust", s_end=10.0, amplitude=1.0e-4)
    assert isinstance(res, IndicialResult)
    assert abs(res.CL[0]) <= 1e-14

    lat = build_lattice(ac, st, collocation="vlm", n_chord=8, chord_spacing="uniform")
    mid_strip = int(np.argmin(np.abs(lat.qc_mid[:, 1])))
    steady_1 = vt.analyze(ac, vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=1.0e-4), st)
    cl_mid_steady_per_rad = float(steady_1.spanwise[0].Cl[mid_strip]) / 1.0e-4

    ss_tols = {1.0: 0.075, 2.0: 0.055, 5.0: 0.035, 10.0: 0.035}
    for s_target, step_idx in ((1.0, 4), (2.0, 8), (5.0, 20), (10.0, 40)):
        assert abs(res.s[step_idx] - s_target) <= 1e-12
        cl_mid = float(res.strip_cl[step_idx, mid_strip])
        ratio_steady = cl_mid / cl_mid_steady_per_rad
        ratio_2d = cl_mid / (2.0 * math.pi)
        psi_ss = _kussner_sears_sparks(s_target)
        psi_ex = _kussner_exact_2d(s_target)
        assert abs(ratio_steady - psi_ex) <= 0.045, (
            f"s={s_target}: ratio_steady={ratio_steady:.5f}, exact_Kussner={psi_ex:.5f}"
        )
        assert abs(ratio_2d - psi_ex) <= 0.030, (
            f"s={s_target}: ratio_2d={ratio_2d:.5f}, exact_Kussner={psi_ex:.5f}"
        )
        assert abs(ratio_steady - psi_ss) <= ss_tols[s_target], (
            f"s={s_target}: ratio_steady={ratio_steady:.5f}, Sears-Sparks={psi_ss:.5f}"
        )


# ---------------------------------------------------------------------------
# 6. 1-cos gust against Duhamel superposition of solver's sharp-edged response
# ---------------------------------------------------------------------------


def test_one_minus_cosine_matches_duhamel_superposition_of_sharp_edged():
    """At small amplitude, 1-cos gust agrees with Duhamel superposition of the sharp-edged response.

    Uses ref_point at the leading edge (0, 0, 0) so both CL and Cm have full
    aerodynamic scale (about the quarter chord, steady Cm vanishes and only a
    tiny O(c / L) gust-gradient residual remains).
    Tolerance derivation:
    - Gust amplitude is w_0 / V_inf = 1e-4 (w_0 = 2e-3 m/s at V_inf = 20 m/s),
      so wake convection with the local vertical velocity tilts the wake by
      O(w_0 / V_inf) = 1e-4, introducing O(1e-4) nonlinearity;
    - Discrete Duhamel superposition sums step increments Delta w_m over a
      gust length L = 8 c resolved with N_L = 32 steps (n_chord = 4, dx_w =
      0.25 c). The second-order quadrature error of sampling 1 - cos(2 pi xi / L)
      on step spacing dx_w is about (1/12) (2 pi / N_L)^2 ~ 3.2e-3 relative to
      the peak response.
    A relative tolerance of 1e-2 (1 %) bounds both the quadrature error and the
    O(1e-4) nonlinear wake convection term cleanly on both CL and Cm.
    """
    wing = vt.LiftingSurface(
        name="wing",
        semi_span=3.0,
        n_panels=4,
        sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=1.0)],
    )
    ac = vt.Aircraft(surfaces=[wing], ref_point=np.array([0.0, 0.0, 0.0]))
    ac.compute_reference_values()
    st = vt.SolverSettings(solver_type="vlm", n_panels=4, n_chord=4, chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=0.0)

    amp_ratio = 1.0e-4
    w0 = cond.V_inf * amp_ratio
    gust_len = 8.0 * ac.c_ref
    n_steps = 48

    res_cos = vt.time_response(
        ac,
        cond,
        st,
        n_steps=n_steps,
        gust=one_minus_cosine(w0, gust_len, "vertical", x_front=0.0),
    )
    res_step = vt.time_response(
        ac,
        cond,
        st,
        n_steps=n_steps,
        gust=sharp_edged(w0, "vertical", x_front=0.0),
    )

    # Unit-amplitude sharp-edged response per m/s of gust velocity.
    psi_cl = res_step.CL / w0
    psi_cm = res_step.Cm / w0

    t = res_cos.time
    xi_lo = cond.V_inf * t[:-1]
    xi_hi = cond.V_inf * t[1:]

    def _profile(xi: np.ndarray) -> np.ndarray:
        in_gust = (xi >= 0.0) & (xi <= gust_len)
        out = np.zeros_like(xi)
        out[in_gust] = 0.5 * w0 * (1.0 - np.cos(2.0 * np.pi * xi[in_gust] / gust_len))
        return out

    dw = _profile(xi_hi) - _profile(xi_lo)
    cl_duh = np.zeros_like(res_cos.CL)
    cm_duh = np.zeros_like(res_cos.Cm)
    for n in range(1, n_steps + 1):
        lags = np.arange(n, 0, -1)
        cl_duh[n] = float(np.dot(dw[:n], 0.5 * (psi_cl[lags] + psi_cl[lags - 1])))
        cm_duh[n] = float(np.dot(dw[:n], 0.5 * (psi_cm[lags] + psi_cm[lags - 1])))

    peak_cl = float(np.max(np.abs(res_cos.CL)))
    peak_cm = float(np.max(np.abs(res_cos.Cm)))
    assert peak_cl > 1e-4 and peak_cm > 1e-5
    err_cl = float(np.max(np.abs(res_cos.CL - cl_duh))) / peak_cl
    err_cm = float(np.max(np.abs(res_cos.Cm - cm_duh))) / peak_cm
    assert err_cl <= 1.0e-2, f"1-cos CL Duhamel rel error = {err_cl:.4e}"
    assert err_cm <= 1.0e-2, f"1-cos Cm Duhamel rel error = {err_cm:.4e}"


# ---------------------------------------------------------------------------
# 7. Steady linear horizontal shear (long-time limit vs direct steady ring solve)
# ---------------------------------------------------------------------------


def test_steady_linear_shear_long_time_limit_matches_direct_steady_ring_solve():
    """Long-time limit of a steady linear horizontal shear matches a direct steady ring solve.

    Uses a dihedral wing so spanwise panels sit at distinct heights z and
    experience a genuine vertical velocity gradient u(z) = u0 + du_dz * z.
    Direct steady reference is built from ring_normal_influence and
    ring_velocity with the same lattice core radii and a long straight wake
    (1000 chords), using the identical local onset velocity at every
    collocation and load point. Tolerance matches the 5e-4 finite-wake steady
    limit of tests/test_unsteady_timedomain.py at L_w = 50 chords (100 steps
    at n_chord = 2).
    """
    wing = vt.LiftingSurface(
        name="dihedral_wing",
        semi_span=3.0,
        n_panels=4,
        dihedral=np.radians(15.0),
        sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=1.0)],
    )
    ac = vt.Aircraft(surfaces=[wing], ref_point=np.array([0.25, 0.0, 0.0]))
    ac.compute_reference_values()
    st = vt.SolverSettings(solver_type="vlm", n_panels=4, n_chord=2, spacing="half-cosine", chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=25.0, rho=1.225, alpha=np.radians(5.0))

    shear_gust = wind_shear(linear_shear(u0=1.0, du_dz=3.0, z_ref=0.0), direction="head")

    n_steps = 100  # 50 chords at n_chord = 2
    lat = build_lattice(ac, st, collocation="vlm", n_chord=2, chord_spacing="uniform")
    res_td = _solve_time_rings(
        lat,
        cond,
        S_ref=ac.S_ref,
        b_ref=ac.b_ref,
        c_ref=ac.c_ref,
        ref_point=ac.ref_point,
        gust=shear_gust,
        n_steps=n_steps,
        te_extension_fraction=0.0,
    )

    # Independent direct steady ring solve with local shear onset and long straight wake.
    d_0 = freestream_direction(cond.alpha, cond.beta)
    L_0 = lift_direction(cond.alpha)
    S_0 = side_direction(cond.alpha, cond.beta)
    V_0 = cond.V_inf * d_0
    rings = build_ring_lattice(lat, wake_dir=d_0, te_extension_fraction=0.0)
    n_p = rings.n_panels
    n_s = rings.n_strips
    te_idx = rings.te_indices

    # Long straight wake ring extending 1000 chords along d_0 behind each TE strip.
    wake_far = 1000.0 * ac.c_ref * d_0[None, :]
    w_corners = np.empty((n_s, 4, 3), dtype=np.float64)
    w_corners[:, 0, :] = rings.te_left
    w_corners[:, 1, :] = rings.te_right
    w_corners[:, 2, :] = rings.te_right + wake_far
    w_corners[:, 3, :] = rings.te_left + wake_far

    A_bb = VR.ring_normal_influence(rings.cp, rings.normal_bc, rings.corners, rings.rc)
    A_bw = VR.ring_normal_influence(rings.cp, rings.normal_bc, w_corners, rings.rc[te_idx])
    A_tot = A_bb.copy()
    A_tot[:, te_idx] += A_bw

    bound_gust = shear_gust._bind_flow(cond, None)
    v_onset_cp = V_0[None, :] + bound_gust(rings.cp, 0.0)
    rhs = -np.sum(v_onset_cp * rings.normal_bc, axis=1)
    gamma_st = np.linalg.solve(A_tot, rhs)

    # Kutta-Joukowski steady loads with identical local onset at load points.
    mid = np.asarray(lat.force_points, dtype=np.float64)
    l_vec = np.asarray(lat.b - lat.a, dtype=np.float64)
    leg_start, leg_end = leg_segments(lat)
    leg_mid = 0.5 * (leg_start + leg_end)
    leg_vec = leg_end - leg_start
    leg_normal, leg_scale, _ = leg_force_rules(lat)
    n_leg_sets = leg_mid.shape[0] // n_p

    P_load = np.vstack([mid, leg_mid])
    seg_vec = np.vstack([l_vec, leg_vec])
    seg_normal = np.vstack([np.zeros((n_p, 3)), leg_normal])
    seg_scale = np.concatenate([np.ones(n_p), leg_scale])
    arm = P_load - ac.ref_point[None, :]

    v_ind = (
        VR.ring_velocity(P_load, rings.corners, rings.rc).swapaxes(1, 2) @ gamma_st
        + VR.ring_velocity(P_load, w_corners, rings.rc[te_idx]).swapaxes(1, 2) @ gamma_st[te_idx]
    )
    V_tot = V_0[None, :] + bound_gust(P_load, 0.0) + v_ind
    V_eff = V_tot - np.sum(V_tot * seg_normal, axis=1, keepdims=True) * seg_normal
    g_hs = _rings_to_horseshoe(gamma_st, rings.n_chord)
    g_leg = leg_strength(lat, g_hs)
    g_all = np.concatenate([g_hs] + [g_leg] * n_leg_sets)
    F_seg = (cond.rho * g_all * seg_scale)[:, None] * cross3(V_eff, seg_vec)
    F_tot = np.sum(F_seg, axis=0)
    M_tot = np.sum(cross3(arm, F_seg), axis=0)

    qS = 0.5 * cond.rho * cond.V_inf ** 2 * ac.S_ref
    CL_ref = float((F_tot @ L_0) / qS)
    CD_ref = float((F_tot @ d_0) / qS)
    CY_ref = float((F_tot @ S_0) / qS)
    Cl_ref = float(-M_tot[0] / (qS * ac.b_ref))
    Cm_ref = float(M_tot[1] / (qS * ac.c_ref))
    Cn_ref = float(-M_tot[2] / (qS * ac.b_ref))

    tol = 5.0e-4
    assert abs(res_td.CL[-1] - CL_ref) <= tol * max(abs(CL_ref), 1.0)
    assert abs(res_td.Cm[-1] - Cm_ref) <= tol * max(abs(Cm_ref), 1.0)
    assert abs(res_td.CD[-1] - CD_ref) <= tol * max(abs(CD_ref), 1.0)
    assert abs(res_td.CY[-1] - CY_ref) <= 1e-12
    assert abs(res_td.Cl[-1] - Cl_ref) <= 1e-12
    assert abs(res_td.Cn[-1] - Cn_ref) <= 1e-12
    assert any("rotational" in n.lower() for n in res_td.notes)


def test_logarithmic_shear_profile_values_and_clamping():
    """logarithmic_shear matches u_ref * ln(z/z0) / ln(z_ref/z0) for z > z0 and 0 for z <= z0."""
    prof = logarithmic_shear(u_ref=10.0, z_ref=10.0, z0=0.1)
    h = np.array([-0.5, 0.0, 0.1, 1.0, 10.0, 20.0])
    out = prof(h)
    assert out[0] == 0.0 and out[1] == 0.0 and out[2] == 0.0
    assert out[3] == pytest.approx(10.0 * math.log(10.0) / math.log(100.0), rel=1e-14)
    assert out[4] == pytest.approx(10.0, rel=1e-14)
    assert out[5] == pytest.approx(10.0 * math.log(200.0) / math.log(100.0), rel=1e-14)


# ---------------------------------------------------------------------------
# 8. Ground effect: horizontal gust mirror parity, wake clearance, notes
# ---------------------------------------------------------------------------


def test_ground_horizontal_gust_equals_two_wing_mirrored_model_roundoff():
    """Horizontal gust at h/c = 0.5 equals explicit two-wing free-air model with mirrored onset to round-off."""
    ac = _rect_wing(semi_span=3.0, chord=1.0, n_panels=4)
    st = vt.SolverSettings(solver_type="vlm", n_panels=4, n_chord=3, chord_spacing="uniform")
    h_ground = 0.5
    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=np.radians(4.0), h=h_ground)
    lat = build_lattice(ac, st, collocation="vlm", n_chord=3, chord_spacing="uniform")
    ground = make_ground_plane(lat, h_ground, cond.alpha, cond.beta, ref_point=ac.ref_point, height_ref="ref")

    # Non-uniform horizontal gust: 1-cos head-wind gust along d_0.
    gust = one_minus_cosine(amplitude=2.5, length=3.0, direction="head", x_front=0.0)
    n_steps = 6
    res_ge = vt.time_response(ac, cond, st, n_steps=n_steps, gust=gust, store="strengths")

    # Explicit 2N x 2N free-air two-wing time-stepping model with mirrored gust onset.
    d_0 = freestream_direction(cond.alpha, cond.beta)
    L_0 = lift_direction(cond.alpha)
    S_0 = side_direction(cond.alpha, cond.beta)
    V_0 = cond.V_inf * d_0
    rings = build_ring_lattice(lat, wake_dir=d_0, te_extension_fraction=0.25)
    n_p = rings.n_panels
    n_s = rings.n_strips
    te_idx = rings.te_indices
    rc_2 = np.concatenate([rings.rc, rings.rc])
    rc_w_2 = np.concatenate([rings.rc[te_idx], rings.rc[te_idx]])
    bound_gust = gust._bind_flow(cond, ground)

    cp_r = rings.cp
    n_r = rings.normal_bc
    c_r = rings.corners
    cp_i = ground.reflect_points(cp_r)
    n_i = ground.reflect_directions(n_r)
    c_i = ground.reflect_points(c_r)
    cp_2 = np.concatenate([cp_r, cp_i], axis=0)
    n_2 = np.concatenate([n_r, n_i], axis=0)
    c_2 = np.concatenate([c_r, c_i], axis=0)

    dt = res_ge.dt
    dx_te = np.asarray(lat.panel_length[te_idx], dtype=np.float64)
    f_te = dx_te / (2.0 * (dx_te + cond.V_inf * dt))

    mid_b = np.asarray(lat.force_points, dtype=np.float64)
    l_vec_b = np.asarray(lat.b - lat.a, dtype=np.float64)
    leg_start, leg_end = leg_segments(lat)
    leg_mid_b = 0.5 * (leg_start + leg_end)
    leg_vec_b = leg_end - leg_start
    leg_normal_b, leg_scale, _ = leg_force_rules(lat)
    n_leg_sets = leg_mid_b.shape[0] // n_p
    P_load = np.vstack([mid_b, leg_mid_b])
    seg_vec = np.vstack([l_vec_b, leg_vec_b])
    seg_normal = np.vstack([np.zeros((n_p, 3)), leg_normal_b])
    seg_scale = np.concatenate([np.ones(n_p), leg_scale])
    arm = P_load - ac.ref_point[None, :]
    l_te = np.asarray(lat.te_right - lat.te_left, dtype=np.float64)
    p_te = np.asarray(lat.te_left, dtype=np.float64) + np.asarray(lat.cp_frac, dtype=np.float64)[:, None] * l_te
    arm_te = p_te - ac.ref_point[None, :]
    from ventorum.unsteady.harmonic import _physical_ring_areas
    area_vec, ring_c, _ = _physical_ring_areas(lat, rings, ref_point=ac.ref_point)
    arm_ring = ring_c - ac.ref_point[None, :]

    # Real wake nodes (exact RK4 convection matching solver; mirror nodes are reflections).
    wake_l_r = np.empty((n_steps + 1, n_s, 3), dtype=np.float64)
    wake_r_r = np.empty((n_steps + 1, n_s, 3), dtype=np.float64)
    wake_c_2 = np.empty((n_steps, 2 * n_s, 4, 3), dtype=np.float64)
    wake_g_2 = np.empty((n_steps, 2 * n_s), dtype=np.float64)
    G_hist = np.empty((n_steps + 1, 2 * n_p), dtype=np.float64)

    def _advect_rk4(pts: np.ndarray, t0: float) -> np.ndarray:
        def _v(p: np.ndarray, t_s: float) -> np.ndarray:
            vg = bound_gust(p, t_s)
            vg = vg - (vg @ ground.normal)[:, None] * ground.normal[None, :]
            return V_0[None, :] + vg

        k1 = _v(pts, t0)
        k2 = _v(pts + 0.5 * dt * k1, t0 + 0.5 * dt)
        k3 = _v(pts + 0.5 * dt * k2, t0 + 0.5 * dt)
        k4 = _v(pts + dt * k3, t0 + dt)
        return pts + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    A_2 = VR.ring_normal_influence(cp_2, n_2, c_2, rc_2)
    qS = 0.5 * cond.rho * cond.V_inf ** 2 * ac.S_ref
    CL_2, CD_2, Cm_2 = (np.empty(n_steps + 1) for _ in range(3))

    for step in range(n_steps + 1):
        t_now = step * dt
        if step == 0:
            wake_l_r[0] = rings.te_left
            wake_r_r[0] = rings.te_right
        else:
            t_prev = (step - 1) * dt
            flat = np.vstack([wake_l_r[:step].reshape(-1, 3), wake_r_r[:step].reshape(-1, 3)])
            adv = _advect_rk4(flat, t_prev)
            half = step * n_s
            wake_l_r[:step] = adv[:half].reshape(step, n_s, 3)
            wake_r_r[:step] = adv[half:].reshape(step, n_s, 3)
            wake_l_r[step] = rings.te_left
            wake_r_r[step] = rings.te_right

            wc_r = np.empty((step, n_s, 4, 3), dtype=np.float64)
            wc_r[:, :, 0, :] = wake_l_r[1 : step + 1]
            wc_r[:, :, 1, :] = wake_r_r[1 : step + 1]
            wc_r[:, :, 2, :] = wake_r_r[:step]
            wc_r[:, :, 3, :] = wake_l_r[:step]
            wc_i = ground.reflect_points(wc_r)
            wake_c_2[:step] = np.concatenate([wc_r, wc_i], axis=1)
            te_idx_2 = np.concatenate([te_idx, te_idx + n_p])
            wake_g_2[step - 1] = G_hist[step - 1, te_idx_2]

        vg_r = bound_gust(cp_r, t_now)
        vg_i = ground.reflect_directions(vg_r)
        v_onset_2 = V_0[None, :] + np.concatenate([vg_r, vg_i], axis=0)
        rhs_2 = -np.sum(v_onset_2 * n_2, axis=1)
        if step > 0:
            wc_flat = wake_c_2[:step].reshape(-1, 4, 3)
            wg_flat = wake_g_2[:step].ravel()
            rc_w_flat = np.tile(rc_w_2, step)
            rhs_2 -= VR.ring_normal_influence(cp_2, n_2, wc_flat, rc_w_flat) @ wg_flat

        G_2 = np.linalg.solve(A_2, rhs_2)
        G_hist[step] = G_2
        assert np.max(np.abs(G_2[n_p:] + G_2[:n_p])) <= 1e-12 * max(float(np.max(np.abs(G_2[:n_p]))), 1.0)

        v_ind_load = VR.ring_velocity(P_load, c_2, rc_2).swapaxes(1, 2) @ G_2
        if step > 0:
            v_ind_load += VR.ring_velocity(P_load, wc_flat, rc_w_flat).swapaxes(1, 2) @ wg_flat
        V_tot = V_0[None, :] + bound_gust(P_load, t_now) + v_ind_load
        V_eff = V_tot - np.sum(V_tot * seg_normal, axis=1, keepdims=True) * seg_normal
        g_hs = _rings_to_horseshoe(G_2[:n_p], rings.n_chord)
        g_leg = leg_strength(lat, g_hs)
        g_all = np.concatenate([g_hs] + [g_leg] * n_leg_sets)
        F_seg = (cond.rho * g_all * seg_scale)[:, None] * cross3(V_eff, seg_vec)
        F_tot = np.sum(F_seg, axis=0)
        M_tot = np.sum(cross3(arm, F_seg), axis=0)

        if step == 1:
            dG_dt = (G_hist[1, :n_p] - G_hist[0, :n_p]) / dt
        elif step >= 2:
            dG_dt = (3.0 * G_hist[step, :n_p] - 4.0 * G_hist[step - 1, :n_p] + G_hist[step - 2, :n_p]) / (2.0 * dt)
        else:
            dG_dt = np.zeros(n_p)
        F_am = (cond.rho * dG_dt)[:, None] * area_vec
        F_tot += np.sum(F_am, axis=0)
        M_tot += np.sum(cross3(arm_ring, F_am), axis=0)

        if step >= 1:
            v_ind_te = VR.ring_velocity(p_te, c_2, rc_2).swapaxes(1, 2) @ G_2
            v_ind_te += VR.ring_velocity(p_te, wc_flat, rc_w_flat).swapaxes(1, 2) @ wg_flat
            V_te = V_0[None, :] + bound_gust(p_te, t_now) + v_ind_te
            dg_te = f_te * (wake_g_2[step - 1, :n_s] - G_2[te_idx])
            F_te = (cond.rho * dg_te)[:, None] * cross3(V_te, l_te)
            F_tot += np.sum(F_te, axis=0)
            M_tot += np.sum(cross3(arm_te, F_te), axis=0)

        CL_2[step] = float((F_tot @ L_0) / qS)
        CD_2[step] = float((F_tot @ d_0) / qS)
        Cm_2[step] = float(M_tot[1] / (qS * ac.c_ref))

    for name, ref_arr in (("CL", CL_2), ("CD", CD_2), ("Cm", Cm_2)):
        got = getattr(res_ge, name)
        scale = max(float(np.max(np.abs(ref_arr))), 1.0)
        assert np.max(np.abs(got - ref_arr)) <= 1e-12 * scale, f"Ground horizontal gust {name} mismatch"
    _ = S_0


def test_ground_vertical_gust_wake_never_crosses_ground_and_notes_present():
    """In ground effect with a strong downward vertical gust, wake nodes stay above ground and notes appear."""
    ac = _rect_wing(semi_span=3.0, chord=1.0, n_panels=4)
    st = vt.SolverSettings(solver_type="vlm", n_panels=4, n_chord=4, chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=np.radians(2.0), h=0.3)

    # Strong downward gust (-15 m/s) that would drive wake nodes below ground if not projected parallel.
    gust = sharp_edged(amplitude=-15.0, direction="vertical", x_front=-1.0)
    res = vt.time_response(ac, cond, st, n_steps=16, gust=gust, store="strengths")

    lat = build_lattice(ac, st, collocation="vlm", n_chord=4, chord_spacing="uniform")
    ground = make_ground_plane(lat, 0.3, cond.alpha, cond.beta, ref_point=ac.ref_point, height_ref="ref")
    wake_heights = ground.height(res.wake_corners.reshape(-1, 3))
    assert float(np.min(wake_heights)) > 0.0

    notes_joined = " ".join(res.notes).lower()
    assert "parallel to the ground" in notes_joined
    assert "not reflected" in notes_joined or "without mirror reflection" in notes_joined

    # Shear in ground effect adds the rotational-flow note as well.
    res_sh = vt.time_response(
        ac,
        cond,
        st,
        n_steps=6,
        gust=wind_shear(logarithmic_shear(u_ref=5.0, z_ref=10.0, z0=0.05), direction="head"),
    )
    notes_sh = " ".join(res_sh.notes).lower()
    assert "rotational" in notes_sh
    assert "parallel to the ground" in notes_sh


# ---------------------------------------------------------------------------
# 9. Control step (kind="control") vs ventorum.control_derivatives
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("h_ground", [None, 0.5])
def test_indicial_control_step_matches_control_derivatives(h_ground):
    """indicial_response(kind='control') long-time limit equals ventorum.control_derivatives.

    Run length: s_end = 100 semichords (50 chords) so finite-wake truncation
    error matches the 5e-4 steady-limit tolerance of tests/test_unsteady_timedomain.py
    plus the small default-ring-offset difference (5e-3 at default core/TE
    closure, as in tests/test_dynamic_derivatives.py).
    """
    ac = _rect_wing(semi_span=3.0, chord=1.0, n_panels=4, flap_deflection_deg=0.0, flap_name="flap")
    st = vt.SolverSettings(solver_type="vlm", n_panels=4, n_chord=4, spacing="half-cosine", chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=0.0, h=h_ground)

    res = vt.indicial_response(ac, cond, st, kind="control", control="flap", s_end=100.0, amplitude=1.0e-4)
    cd_ref = vt.control_derivatives(ac, cond, st, controls=["flap"], axes="body").derivatives["flap"]

    assert res.steady is not None
    assert res.steady["CL"] == pytest.approx(cd_ref.CL, rel=1e-12)
    assert res.steady["Cm"] == pytest.approx(cd_ref.Cm, rel=1e-12)

    tol = 5.0e-3
    assert abs(res.CL[-1] - cd_ref.CL) <= tol * max(abs(cd_ref.CL), 1.0)
    assert abs(res.Cm[-1] - cd_ref.Cm) <= tol * max(abs(cd_ref.Cm), 1.0)
    assert abs(res.CY[-1] - cd_ref.CY) <= 1e-12
    assert abs(res.Cl[-1] - cd_ref.Cl) <= 1e-12
    assert abs(res.Cn[-1] - cd_ref.Cn) <= 1e-12


# ---------------------------------------------------------------------------
# 10. to_dict(), store options, trust and notes
# ---------------------------------------------------------------------------


def test_to_dict_store_trust_and_notes():
    """to_dict() is strict JSON; store controls strength arrays; trust and notes record all conditions."""
    from ventorum import gpu

    ac = _rect_wing(semi_span=3.0, chord=1.0, n_panels=4)
    st_coarse = vt.SolverSettings(solver_type="vlm", n_panels=4, n_chord=4, chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=np.radians(4.0))

    # store="loads" (default), dt differing > 25%, n_chord < 8, GPU device
    # selected when the GPU pipelines can run (set_device refuses "gpu"
    # without them; the GPU part then runs at the GPU gate).
    gpu_selected = gpu.available()
    old_dev = gpu.get_device()
    try:
        if gpu_selected:
            gpu.set_device("gpu")
        res_loads = vt.time_response(
            ac, cond, st_coarse, t_end=0.05, dt=0.025, gust=sharp_edged(1.0, "head"), store="loads",
        )
    finally:
        gpu.set_device(old_dev)

    assert res_loads.store == "loads"
    assert res_loads.gamma_bound is None
    assert res_loads.gamma_wake is None
    assert res_loads.wake_corners is None
    if gpu_selected:
        assert res_loads.device == "gpu"
        assert res_loads.precision == gpu.get_precision()
    else:
        assert res_loads.device == "cpu"
        assert res_loads.precision == "float64"
    assert res_loads.trust is not None
    assert res_loads.trust.rating == res_loads.base.totals.trust.rating
    assert any("below 8" in w for w in res_loads.trust_notes)
    assert any("below 8" in w for w in res_loads.trust.warnings)
    notes_txt = " ".join(res_loads.notes).lower()
    assert "25 %" in notes_txt or "differs from the default" in notes_txt
    if gpu_selected:
        assert "gpu" in notes_txt
    else:
        assert "cpu" in notes_txt

    d_loads = res_loads.to_dict()
    encoded = json.dumps(d_loads, allow_nan=False)
    decoded = json.loads(encoded)
    assert "gamma_bound" not in decoded
    assert len(decoded["time"]) == res_loads.n_steps + 1
    assert len(decoded["gust_velocity"]) == res_loads.n_steps + 1

    # store="strengths" includes gamma_bound, gamma_wake and wake_corners.
    res_str = vt.time_response(ac, cond, st_coarse, n_steps=3, store="strengths")
    assert res_str.gamma_bound is not None
    assert res_str.gamma_wake is not None
    assert res_str.wake_corners is not None
    d_str = res_str.to_dict()
    json.dumps(d_str, allow_nan=False)
    assert len(d_str["gamma_bound"]) == res_str.n_panels
    assert len(d_str["gamma_wake"]) == 3
    assert len(d_str["wake_corners"]) == 3

    # IndicialResult.to_dict() is also strict JSON.
    ind = vt.indicial_response(ac, cond, st_coarse, kind="alpha", s_end=2.0)
    ind_d = ind.to_dict()
    json.dumps(ind_d, allow_nan=False)
    assert ind_d["kind"] == "alpha"
    assert ind_d["steady"] is not None


# ---------------------------------------------------------------------------
# 11. Every input check
# ---------------------------------------------------------------------------


def test_all_input_checks():
    """All invalid inputs to time_response, indicial_response and gusts raise InputError."""
    ac = _rect_wing(semi_span=3.0, chord=1.0, n_panels=4, flap_deflection_deg=0.0)
    st = vt.SolverSettings(solver_type="vlm", n_panels=4, n_chord=4)
    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=0.0)

    # Both or none of t_end and n_steps.
    with pytest.raises(InputError, match="t_end|n_steps"):
        vt.time_response(ac, cond, st)
    with pytest.raises(InputError, match="t_end|n_steps"):
        vt.time_response(ac, cond, st, t_end=0.1, n_steps=4)
    with pytest.raises(InputError, match="t_end"):
        vt.time_response(ac, cond, st, t_end=-0.1)
    with pytest.raises(InputError, match="t_end"):
        vt.time_response(ac, cond, st, t_end=float("nan"))
    with pytest.raises(InputError, match="n_steps"):
        vt.time_response(ac, cond, st, n_steps=0)
    with pytest.raises(InputError, match="dt"):
        vt.time_response(ac, cond, st, n_steps=2, dt=0.0)
    with pytest.raises(InputError, match="store"):
        vt.time_response(ac, cond, st, n_steps=2, store="cp")

    # Unknown kind, missing or stray control, invalid s_end / amplitude.
    with pytest.raises(InputError, match="kind"):
        vt.indicial_response(ac, cond, st, kind="pitch")
    with pytest.raises(InputError, match="control"):
        vt.indicial_response(ac, cond, st, kind="control")
    with pytest.raises(InputError, match="control"):
        vt.indicial_response(ac, cond, st, kind="alpha", control="flap")
    with pytest.raises(InputError, match="s_end"):
        vt.indicial_response(ac, cond, st, kind="alpha", s_end=0.0)
    with pytest.raises(InputError, match="amplitude"):
        vt.indicial_response(ac, cond, st, kind="alpha", amplitude=-1e-4)

    # Unknown gust direction or invalid direction vector or gust parameters.
    with pytest.raises(InputError, match="direction"):
        sharp_edged(1.0, "diagonal")
    with pytest.raises(InputError, match="direction"):
        sharp_edged(1.0, [0.0, 0.0, 0.0])
    with pytest.raises(InputError, match="direction"):
        sharp_edged(1.0, [2.0, 0.0, 0.0])
    with pytest.raises(InputError, match="direction"):
        wind_shear(linear_shear(0.0, 1.0), direction="vertical")
    with pytest.raises(InputError, match="length"):
        one_minus_cosine(1.0, length=0.0, direction="vertical")
    with pytest.raises(InputError, match="z0"):
        logarithmic_shear(5.0, z_ref=10.0, z0=0.0)
    with pytest.raises(InputError, match="z_ref"):
        logarithmic_shear(5.0, z_ref=0.1, z0=0.2)

    # Gust callable with wrong shape or non-finite values, or non-callable gust.
    with pytest.raises(InputError, match="gust"):
        vt.time_response(ac, cond, st, n_steps=2, gust="not_a_callable")  # type: ignore[arg-type]
    with pytest.raises(InputError, match="shape"):
        vt.time_response(ac, cond, st, n_steps=2, gust=lambda pts, t: np.zeros((pts.shape[0], 2)))
    with pytest.raises(InputError, match="finite"):
        vt.time_response(ac, cond, st, n_steps=2, gust=lambda pts, t: np.full_like(pts, np.nan))
    with pytest.raises(InputError, match="finite"):
        vt.time_response(ac, cond, st, n_steps=2, gust=lambda pts, t: np.full_like(pts, np.inf))

    # Mach > 0.3 (explicit mach and high V_inf).
    cond_mach = vt.FlightCondition(V_inf=50.0, rho=1.225, alpha=0.0, mach=0.4)
    with pytest.raises(InputError, match="Mach|mach"):
        vt.time_response(ac, cond_mach, st, n_steps=2)
    with pytest.raises(InputError, match="Mach|mach"):
        vt.indicial_response(ac, cond_mach, st, kind="alpha", s_end=2.0)

    cond_fast = vt.FlightCondition(V_inf=120.0, rho=1.225, alpha=0.0)
    with pytest.raises(InputError, match="Mach|mach"):
        vt.time_response(ac, cond_fast, st, n_steps=2)

    # Non-VLM solver refused.
    for bad_solver in ("linear", "nonlinear", "fourier"):
        st_bad = vt.SolverSettings(solver_type=bad_solver, n_panels=4)
        with pytest.raises(InputError, match="vlm"):
            vt.time_response(ac, cond, st_bad, n_steps=2)
        with pytest.raises(InputError, match="vlm"):
            vt.indicial_response(ac, cond, st_bad, kind="alpha", s_end=2.0)
