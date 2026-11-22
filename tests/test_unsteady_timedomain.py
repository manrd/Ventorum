# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for the prescribed-wake time-domain vortex-ring solver.

Covers the steady limit against ``ventorum.analyze``, the Wagner indicial lift
build-up on an AR 40 wing against the R. T. Jones (1940) approximation, the
small-amplitude periodic limit against ``_solve_harmonic_rings`` in free air
and ground effect, second-order time-step convergence at ``k = 0.5``, ground
image parity against an explicit antiphase two-wing free-air model, numpy
against Numba ring-kernel backend parity (including multi-surface cross-surface
cores), bent-polyline and junction lattices, time-step diagnostic notes, and
all input checks.

References
----------
* J. Katz and A. Plotkin, "Low-Speed Aerodynamics", 2nd ed., Cambridge
  University Press, 2001, chapter 13.
* R. T. Jones, "The Unsteady Lift of a Wing of Finite Aspect Ratio", NACA
  Report 681, 1940 (two-exponential approximation to the Wagner function).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import ventorum as vt
from ventorum.aero import vortex_ring as VR
from ventorum.aero import vortex_ring_numba as VR_NB
from ventorum.aero.loads import leg_force_rules, leg_segments, leg_strength
from ventorum.aero.system import freestream_direction, lift_direction, make_ground_plane, side_direction
from ventorum.core.errors import InputError
from ventorum.geometry import lattice as lattice_mod
from ventorum.geometry.lattice import build_lattice
from ventorum.unsteady.harmonic import (
    _ey_cross,
    _physical_ring_areas,
    _rings_to_horseshoe,
    _rot_y,
    _solve_harmonic_rings,
)
from ventorum.unsteady.rings import build_ring_lattice
from ventorum.unsteady.timedomain import _TimeDomainResponse, _solve_time_rings
from ventorum.utils.vec import cross3


def _rect_wing(
    semi_span: float = 3.0,
    chord: float = 1.0,
    n_panels: int = 8,
) -> vt.Aircraft:
    """Build a flat rectangular wing with quarter-chord moment reference."""
    wing = vt.LiftingSurface(
        name="wing",
        semi_span=semi_span,
        n_panels=n_panels,
        sections=[
            vt.WingSection(y_frac=0.0, chord=chord),
            vt.WingSection(y_frac=1.0, chord=chord),
        ],
    )
    ac = vt.Aircraft(surfaces=[wing], ref_point=np.array([0.25 * chord, 0.0, 0.0]))
    ac.compute_reference_values()
    return ac


def _swept_tapered_wing(n_panels: int = 4) -> vt.Aircraft:
    """Build a swept, tapered wing with dihedral and twist."""
    wing = vt.LiftingSurface(
        name="swept",
        semi_span=3.0,
        n_panels=n_panels,
        sweep_le=np.radians(20.0),
        dihedral=np.radians(6.0),
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.2, twist=np.radians(1.0)),
            vt.WingSection(y_frac=1.0, chord=0.6, twist=np.radians(-2.0)),
        ],
    )
    ac = vt.Aircraft(surfaces=[wing], ref_point=np.array([0.35, 0.0, 0.0]))
    ac.compute_reference_values()
    return ac


def _wagner_jones(s: float) -> float:
    """Return the R. T. Jones (1940, NACA Report 681) approximation to the Wagner function [-].

    Parameters
    ----------
    s : float
        Reduced time ``2 * V_inf * t / c`` in semichords travelled [-].

    Returns
    -------
    float
        Approximate indicial lift ratio ``Phi(s)`` [-].
    """
    return 1.0 - 0.165 * math.exp(-0.0455 * s) - 0.335 * math.exp(-0.300 * s)


def _fit_first_harmonic(t: np.ndarray, y: np.ndarray, omega: float, amp_nd: float) -> complex:
    """Fit ``c0 + A * Re(H * (-i) * exp(i * omega * t))`` to *y* by least squares.

    For motion ``q(t) = A * sin(omega * t) = Re((-i * A) * exp(i * omega * t))``,
    the steady periodic response is ``c0 + A * (H_r * sin(omega * t) + H_i * cos(omega * t))``.
    """
    basis = np.column_stack([np.ones_like(t), np.sin(omega * t), np.cos(omega * t)])
    coef, _, _, _ = np.linalg.lstsq(basis, y, rcond=None)
    return complex(coef[1] / amp_nd, coef[2] / amp_nd)


def _periodic_run_and_compare(
    lat: vt.VortexLattice,
    ac: vt.Aircraft,
    cond: vt.FlightCondition,
    ground,
    k: float,
    motion: str,
    *,
    n_periods_transient: float,
    n_periods_fit: int = 1,
    dt: float | None = None,
) -> tuple[complex, complex, complex, complex]:
    """Run time-domain harmonic motion and return ``(CL_td, CL_h, Cm_td, Cm_h)``.

    When ``cond.alpha != 0``, the zero-motion impulsive-start baseline over the
    same time steps is subtracted before fitting so the ``O(1 / L_w^2)`` steady
    starting-vortex drift does not contaminate a ``1e-4`` amplitude oscillation.
    """
    omega = 2.0 * k * cond.V_inf / ac.c_ref
    period = 2.0 * math.pi / omega
    te_idx = np.flatnonzero(np.tile(np.arange(lat.n_chord), lat.n_strips) == (lat.n_chord - 1))
    dx_te_mean = float(np.sum(lat.width * lat.panel_length[te_idx]) / np.sum(lat.width))
    dt_use = dx_te_mean / cond.V_inf if dt is None else float(dt)

    total_time = (n_periods_transient + n_periods_fit) * period
    n_steps = int(math.ceil(total_time / dt_use))

    if motion == "heave":
        amp_nd = 1.0e-4
        amp_dim = amp_nd * ac.c_ref
        z_fn = lambda t: amp_dim * math.sin(omega * t)  # noqa: E731
        th_fn = lambda t: 0.0  # noqa: E731
    elif motion == "pitch":
        amp_nd = 1.0e-4
        z_fn = lambda t: 0.0  # noqa: E731
        th_fn = lambda t: amp_nd * math.sin(omega * t)  # noqa: E731
    else:
        raise ValueError(f"Unknown motion {motion!r}.")

    res_td = _solve_time_rings(
        lat,
        cond,
        S_ref=ac.S_ref,
        b_ref=ac.b_ref,
        c_ref=ac.c_ref,
        ref_point=ac.ref_point,
        z=z_fn,
        theta=th_fn,
        n_steps=n_steps,
        dt=dt,
        ground=ground,
        te_correction=True,
    )
    res_h = _solve_harmonic_rings(
        lat,
        cond,
        [k],
        S_ref=ac.S_ref,
        b_ref=ac.b_ref,
        c_ref=ac.c_ref,
        ground=ground,
        ref_point=ac.ref_point,
        wake_chords=5.0,
        use_tail=True,
        te_correction=True,
    )

    if abs(cond.alpha) > 0.0:
        zeros = np.zeros(n_steps + 1, dtype=np.float64)
        res_base = _solve_time_rings(
            lat,
            cond,
            S_ref=ac.S_ref,
            b_ref=ac.b_ref,
            c_ref=ac.c_ref,
            ref_point=ac.ref_point,
            z=zeros,
            theta=zeros,
            n_steps=n_steps,
            dt=dt,
            ground=ground,
            te_correction=True,
        )
        cl_sig = res_td.CL - res_base.CL
        cm_sig = res_td.Cm - res_base.Cm
    else:
        cl_sig = res_td.CL
        cm_sig = res_td.Cm

    t_fit_start = res_td.time[-1] - n_periods_fit * period
    mask = res_td.time >= t_fit_start - 1e-12 * dt_use
    t_win = res_td.time[mask]
    cl_td = _fit_first_harmonic(t_win, cl_sig[mask], omega, amp_nd)
    cm_td = _fit_first_harmonic(t_win, cm_sig[mask], omega, amp_nd)

    h_motion = res_h.heave if motion == "heave" else res_h.pitch
    cl_h = complex(h_motion.CL[0])
    cm_h = complex(h_motion.Cm[0])
    return cl_td, cl_h, cm_td, cm_h


# ---------------------------------------------------------------------------
# 1. Steady limit against ventorum.analyze (exact-equivalence settings)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("h_ground", [None, 0.5])
def test_steady_limit_matches_analyze_exact_equivalence(monkeypatch, h_ground):
    """Impulsive start at constant alpha converges to ventorum.analyze as the wake grows.

    Run length: 50 chords (200 steps at n_chord = 4, dt = c / (4 V)).
    Tolerance derivation from the finite wake length L_w = 50 c on an AR 6 wing:
    the starting vortex (-Gamma_0) at L_w induces downwash (Gamma_0 b) / (4 pi L_w^2)
    while the omitted semi-infinite trailing legs beyond L_w remove half as much,
    leaving a net downwash deficit of about (Gamma_0 b) / (8 pi L_w^2). With
    Gamma_0 ~ (1/2) V c CL and 3D lift slope a_3d ~ 4.5, the relative lift error
    in free air is about (a_3d AR) / (16 pi (L_w / c)^2) ~ 2.2e-4 (and smaller
    in ground effect because the image starting vortex forms a dipole decaying
    as O(h^2 / L_w^4)). A relative tolerance of 5e-4 bounds this finite-wake
    truncation error with a clean margin of about 2x.
    """
    ac = _rect_wing(semi_span=3.0, chord=1.0, n_panels=6)
    st = vt.SolverSettings(
        solver_type="vlm",
        n_panels=6,
        n_chord=4,
        spacing="half-cosine",
        chord_spacing="uniform",
    )
    cond = vt.FlightCondition(V_inf=25.0, rho=1.225, alpha=np.radians(5.0), h=h_ground)

    monkeypatch.setattr(lattice_mod, "CORE_RADIUS_FRACTION", 0.0)
    lat = build_lattice(ac, st, collocation="vlm", n_chord=4, chord_spacing="uniform")
    ground = (
        make_ground_plane(lat, h_ground, cond.alpha, cond.beta, ref_point=ac.ref_point, height_ref="ref")
        if h_ground is not None
        else None
    )

    steady_ref = vt.analyze(ac, cond, st)
    n_steps = 200  # 50 chords at n_chord = 4 (dx_w = 0.25 c)
    zeros = np.zeros(n_steps + 1, dtype=np.float64)
    res = _solve_time_rings(
        lat,
        cond,
        S_ref=ac.S_ref,
        b_ref=ac.b_ref,
        c_ref=ac.c_ref,
        ref_point=ac.ref_point,
        z=zeros,
        theta=zeros,
        n_steps=n_steps,
        ground=ground,
        te_extension_fraction=0.0,
        rc=0.0,
    )

    assert isinstance(res, _TimeDomainResponse)
    assert res.notes == []
    assert abs(res.n_steps * res.dx_w - 50.0 * ac.c_ref) <= 1e-12

    tol = 5.0e-4
    assert abs(res.CL[-1] - steady_ref.totals.CL) <= tol * max(abs(steady_ref.totals.CL), 1.0)
    assert abs(res.Cm[-1] - steady_ref.totals.Cm) <= tol * max(abs(steady_ref.totals.Cm), 1.0)
    assert abs(res.CD[-1] - steady_ref.totals.CDi_nearfield) <= tol * max(abs(steady_ref.totals.CDi_nearfield), 1.0)
    assert abs(res.CY[-1] - steady_ref.totals.CY) <= 1e-12
    assert abs(res.Cl[-1] - steady_ref.totals.Cl) <= 1e-12
    assert abs(res.Cn[-1] - steady_ref.totals.Cn) <= 1e-12


# ---------------------------------------------------------------------------
# 2. Wagner indicial lift on an AR 40 rectangular wing (R. T. Jones 1940)
# ---------------------------------------------------------------------------


def test_wagner_ar40_impulsive_start_matches_jones_approximation():
    """Mid-span lift ratio of an AR 40 wing matches R. T. Jones (1940) Wagner function.

    Checked at s = 2 V t / c = 1, 2, 5, 10 (steps 4, 8, 20, 40 for n_chord = 8,
    ds = 0.25 semichords per step). Tolerance derivation: finite aspect ratio
    AR 40 lowers the steady mid-span lift by 2.7 % relative to 2 pi sin(alpha),
    which raises the finite-AR indicial ratio cl_mid(s) / cl_mid_steady by
    about +0.027 at early s (3D tip relief accelerates convergence to the
    steady state, R. T. Jones 1940, NACA Report 681). Adding ~0.008 for the
    discrete time step ds = 0.25 and the two-exponential Jones approximation
    error gives a tolerance of 0.035 on cl_mid(s) / cl_mid_steady (and 0.015 on
    cl_mid(s) / (2 pi sin(alpha))).
    """
    ac = _rect_wing(semi_span=20.0, chord=1.0, n_panels=20)
    lat = build_lattice(ac, n_chord=8, chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=np.radians(2.0))

    st = vt.SolverSettings(solver_type="vlm", n_panels=20, n_chord=8, spacing="half-cosine", chord_spacing="uniform")
    steady_res = vt.analyze(ac, cond, st)
    mid_strip = int(np.argmin(np.abs(lat.qc_mid[:, 1])))
    cl_mid_steady = float(steady_res.spanwise[0].Cl[mid_strip])
    cl_2d_steady = 2.0 * math.pi * math.sin(cond.alpha)

    n_steps = 40  # s = 40 * 0.25 = 10.0 semichords
    zeros = np.zeros(n_steps + 1, dtype=np.float64)
    res = _solve_time_rings(
        lat,
        cond,
        S_ref=ac.S_ref,
        b_ref=ac.b_ref,
        c_ref=ac.c_ref,
        ref_point=ac.ref_point,
        z=zeros,
        theta=zeros,
        n_steps=n_steps,
    )

    s_vals = 2.0 * cond.V_inf * res.time / ac.c_ref
    for s_target, step_idx in ((1.0, 4), (2.0, 8), (5.0, 20), (10.0, 40)):
        assert abs(s_vals[step_idx] - s_target) <= 1e-12
        cl_mid = float(res.strip_cl[step_idx, mid_strip])
        ratio_steady = cl_mid / cl_mid_steady
        ratio_2d = cl_mid / cl_2d_steady
        phi_jones = _wagner_jones(s_target)
        assert abs(ratio_steady - phi_jones) <= 0.035, (
            f"s={s_target}: mid-span ratio={ratio_steady:.5f}, Jones Phi={phi_jones:.5f}, "
            f"diff={abs(ratio_steady - phi_jones):.5f}"
        )
        assert abs(ratio_2d - phi_jones) <= 0.015, (
            f"s={s_target}: 2D-normalised ratio={ratio_2d:.5f}, Jones Phi={phi_jones:.5f}, "
            f"diff={abs(ratio_2d - phi_jones):.5f}"
        )


# ---------------------------------------------------------------------------
# 3. Periodic limit against _solve_harmonic_rings (free air & h/c = 0.5)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("motion", ["heave", "pitch"])
def test_periodic_limit_k05_free_air_alpha4_matches_harmonic(motion):
    """Fast default check: k = 0.5 free-air lifting mean state (alpha0 = 4 deg) matches within 1 %."""
    ac = _rect_wing(semi_span=3.0, chord=1.0, n_panels=4)
    lat = build_lattice(ac, n_chord=12, chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=np.radians(4.0))

    cl_td, cl_h, cm_td, cm_h = _periodic_run_and_compare(
        lat, ac, cond, None, 0.5, motion, n_periods_transient=4.0, n_periods_fit=1,
    )

    err_cl = abs(cl_td - cl_h) / abs(cl_h)
    err_cm = abs(cm_td - cm_h) / abs(cm_h)
    assert err_cl <= 1.0e-2, f"motion={motion}, h=None, k=0.5: CL err={err_cl:.4%}"
    assert err_cm <= 1.0e-2, f"motion={motion}, h=None, k=0.5: Cm err={err_cm:.4%}"


@pytest.mark.slow
@pytest.mark.parametrize("motion", ["heave", "pitch"])
@pytest.mark.parametrize("h_ground", [None, 0.5])
@pytest.mark.parametrize("k", [0.05, 0.2, 0.5])
def test_periodic_limit_alpha4_matches_harmonic(motion, h_ground, k):
    """Full periodic-limit sweep at alpha0 = 4 deg (lifting mean state), free air and h/c = 0.5.

    Targets: relative difference <= 0.1 % at k <= 0.2 and <= 1 % at k = 0.5 on
    both CL and Cm against _solve_harmonic_rings on the same lattice.
    """
    ac = _rect_wing(semi_span=3.0, chord=1.0, n_panels=4)
    if k == 0.05:
        n_chord, n_trans = 4, 1.0
    elif k == 0.2:
        n_chord, n_trans = 16, 2.0
    else:
        n_chord, n_trans = 12, 4.0

    lat = build_lattice(ac, n_chord=n_chord, chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=np.radians(4.0), h=h_ground)
    ground = (
        make_ground_plane(lat, h_ground, cond.alpha, cond.beta, ref_point=ac.ref_point, height_ref="ref")
        if h_ground is not None
        else None
    )

    cl_td, cl_h, cm_td, cm_h = _periodic_run_and_compare(
        lat, ac, cond, ground, k, motion, n_periods_transient=n_trans, n_periods_fit=1,
    )

    target = 1.0e-3 if k <= 0.2 else 1.0e-2
    err_cl = abs(cl_td - cl_h) / abs(cl_h)
    err_cm = abs(cm_td - cm_h) / abs(cm_h)
    assert err_cl <= target, (
        f"alpha0=4 deg, motion={motion}, h={h_ground}, k={k}: CL err={err_cl:.4%} (td={cl_td}, h={cl_h})"
    )
    assert err_cm <= target, (
        f"alpha0=4 deg, motion={motion}, h={h_ground}, k={k}: Cm err={err_cm:.4%} (td={cm_td}, h={cm_h})"
    )


@pytest.mark.slow
@pytest.mark.parametrize("motion", ["heave", "pitch"])
@pytest.mark.parametrize("h_ground", [None, 0.5])
@pytest.mark.parametrize("k", [0.05, 0.2, 0.5])
def test_periodic_limit_alpha0_matches_harmonic(motion, h_ground, k):
    """Unloaded mean state (alpha0 = 0 deg): time-domain solver matches _solve_harmonic_rings to target.

    At alpha0 = 0 deg (Gamma_0 = 0), the moving-trailing-edge wake undulation
    vanishes to first order in amplitude and both solvers share identical wake
    geometry, verifying the 0.1 % (k <= 0.2) and 1 % (k = 0.5) targets cleanly.
    """
    ac = _rect_wing(semi_span=3.0, chord=1.0, n_panels=4)
    if k == 0.05:
        n_chord, n_trans = 4, 1.0
    elif k == 0.2:
        n_chord, n_trans = 16, 2.0
    else:
        n_chord, n_trans = 12, 4.0

    lat = build_lattice(ac, n_chord=n_chord, chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=0.0, h=h_ground)
    ground = (
        make_ground_plane(lat, h_ground, cond.alpha, cond.beta, ref_point=ac.ref_point, height_ref="ref")
        if h_ground is not None
        else None
    )

    cl_td, cl_h, cm_td, cm_h = _periodic_run_and_compare(
        lat, ac, cond, ground, k, motion, n_periods_transient=n_trans, n_periods_fit=1,
    )

    target = 1.0e-3 if k <= 0.2 else 1.0e-2
    err_cl = abs(cl_td - cl_h) / abs(cl_h)
    err_cm = abs(cm_td - cm_h) / abs(cm_h)
    assert err_cl <= target, (
        f"alpha0=0 deg, motion={motion}, h={h_ground}, k={k}: CL err={err_cl:.4%} (td={cl_td}, h={cl_h})"
    )
    assert err_cm <= target, (
        f"alpha0=0 deg, motion={motion}, h={h_ground}, k={k}: Cm err={err_cm:.4%} (td={cm_td}, h={cm_h})"
    )


# ---------------------------------------------------------------------------
# 4. Time-step convergence at k = 0.5 (second-order backward difference)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_timestep_convergence_k05_second_order():
    """Periodic-limit difference from _solve_harmonic_rings at k = 0.5 falls by ~4 when dt halves.

    Doubling n_chord from 4 to 8 to 16 halves dt = dx_TE / V_inf while keeping
    the discrete wake row length dx_w = dx_TE identical between the time-domain
    and harmonic solvers at each level, isolating the O(dt^2) error of the
    second-order backward difference for d(Gamma)/dt.
    """
    ac = _rect_wing(semi_span=3.0, chord=1.0, n_panels=4)
    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=np.radians(4.0))
    k = 0.5

    errs_cl = []
    errs_cm = []
    for n_c in (4, 8, 16):
        lat = build_lattice(ac, n_chord=n_c, chord_spacing="uniform")
        cl_td, cl_h, cm_td, cm_h = _periodic_run_and_compare(
            lat, ac, cond, None, k, "heave", n_periods_transient=4.0, n_periods_fit=1,
        )
        errs_cl.append(abs(cl_td - cl_h) / abs(cl_h))
        errs_cm.append(abs(cm_td - cm_h) / abs(cm_h))

    order_cl_1 = math.log2(errs_cl[0] / errs_cl[1])
    order_cl_2 = math.log2(errs_cl[1] / errs_cl[2])
    order_cm_1 = math.log2(errs_cm[0] / errs_cm[1])
    order_cm_2 = math.log2(errs_cm[1] / errs_cm[2])

    assert 1.7 <= order_cl_1 <= 2.3, f"CL order (4->8)={order_cl_1:.3f}, errs={errs_cl}"
    assert 1.7 <= order_cl_2 <= 2.3, f"CL order (8->16)={order_cl_2:.3f}, errs={errs_cl}"
    assert 1.7 <= order_cm_1 <= 2.3, f"Cm order (4->8)={order_cm_1:.3f}, errs={errs_cm}"
    assert 1.7 <= order_cm_2 <= 2.3, f"Cm order (8->16)={order_cm_2:.3f}, errs={errs_cm}"


# ---------------------------------------------------------------------------
# 5. Ground parity: images against explicit antiphase two-wing free-air model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("wing_kind", ["rect", "swept"])
def test_ground_parity_antiphase_two_wing_model(wing_kind):
    """Ground-image time-domain solver equals an explicit antiphase two-wing free-air model to round-off."""
    ac = _rect_wing(n_panels=4) if wing_kind == "rect" else _swept_tapered_wing(n_panels=4)
    lat = build_lattice(ac, n_chord=3, chord_spacing="uniform")
    h_ground = 0.5
    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=np.radians(4.0), h=h_ground)
    ground = make_ground_plane(lat, h_ground, cond.alpha, cond.beta, ref_point=ac.ref_point, height_ref="ref")

    n_steps = 8
    omega = 2.0 * 0.3 * cond.V_inf / ac.c_ref
    z_fn = lambda t: 0.05 * math.sin(omega * t)  # noqa: E731
    th_fn = lambda t: np.radians(2.0) * math.sin(omega * t + 0.3)  # noqa: E731

    res_ge = _solve_time_rings(
        lat,
        cond,
        S_ref=ac.S_ref,
        b_ref=ac.b_ref,
        c_ref=ac.c_ref,
        ref_point=ac.ref_point,
        z=z_fn,
        theta=th_fn,
        n_steps=n_steps,
        ground=ground,
        te_correction=True,
    )

    # Independent explicit 2N x 2N free-air two-wing time-stepping model in antiphase.
    d_0 = freestream_direction(cond.alpha, cond.beta)
    L_0 = lift_direction(cond.alpha)
    S_0 = side_direction(cond.alpha, cond.beta)
    V_0 = cond.V_inf * d_0
    e_h = np.asarray(ground.normal, dtype=np.float64)
    rings = build_ring_lattice(lat, wake_dir=d_0, te_extension_fraction=0.25)
    n_p = rings.n_panels
    n_s = rings.n_strips
    te_idx = rings.te_indices
    rc_2 = np.concatenate([rings.rc, rings.rc])
    rc_w_2 = np.concatenate([rings.rc[te_idx], rings.rc[te_idx]])

    dt = res_ge.dt
    wake_conv_r = (cond.V_inf * dt) * rings.wake_dir
    wake_conv_i = ground.reflect_directions(wake_conv_r)
    wake_conv_2 = np.concatenate([wake_conv_r, wake_conv_i], axis=0)

    # Body-frame load geometry of the real wing.
    mid_b = np.asarray(lat.force_points, dtype=np.float64)
    l_vec_b = np.asarray(lat.b - lat.a, dtype=np.float64)
    leg_start, leg_end = leg_segments(lat)
    leg_mid_b = 0.5 * (leg_start + leg_end)
    leg_vec_b = leg_end - leg_start
    leg_normal_b, leg_scale, _ = leg_force_rules(lat)
    n_leg_sets = leg_mid_b.shape[0] // n_p
    P_load_b = np.vstack([mid_b, leg_mid_b])
    seg_vec_b = np.vstack([l_vec_b, leg_vec_b])
    seg_normal_b = np.vstack([np.zeros((n_p, 3)), leg_normal_b])
    seg_scale = np.concatenate([np.ones(n_p), leg_scale])
    arm_b = P_load_b - ac.ref_point[None, :]
    l_te_b = np.asarray(lat.te_right - lat.te_left, dtype=np.float64)
    p_te_b = np.asarray(lat.te_left, dtype=np.float64) + np.asarray(lat.cp_frac, dtype=np.float64)[:, None] * l_te_b
    arm_te_b = p_te_b - ac.ref_point[None, :]
    area_vec_b, ring_c_b, _ = _physical_ring_areas(lat, rings, ref_point=ac.ref_point)
    arm_ring_b = ring_c_b - ac.ref_point[None, :]
    dx_te = np.asarray(lat.panel_length[te_idx], dtype=np.float64)
    f_te = dx_te / (2.0 * (dx_te + cond.V_inf * dt))

    wake_c_2 = np.empty((n_steps, 2 * n_s, 4, 3), dtype=np.float64)
    wake_g_2 = np.empty((n_steps, 2 * n_s), dtype=np.float64)
    G_hist = np.empty((n_steps + 1, 2 * n_p), dtype=np.float64)
    prev_te_l_2 = None
    prev_te_r_2 = None

    qS = 0.5 * cond.rho * cond.V_inf ** 2 * ac.S_ref
    CL_2 = np.empty(n_steps + 1)
    CD_2 = np.empty(n_steps + 1)
    CY_2 = np.empty(n_steps + 1)
    Cl_2 = np.empty(n_steps + 1)
    Cm_2 = np.empty(n_steps + 1)
    Cn_2 = np.empty(n_steps + 1)

    for step in range(n_steps + 1):
        zn = res_ge.z[step]
        th = res_ge.theta[step]
        zd = res_ge.z_dot[step]
        thd = res_ge.theta_dot[step]
        x_ref = ac.ref_point + zn * e_h

        cp_r = x_ref[None, :] + _rot_y(rings.cp - ac.ref_point[None, :], th)
        n_r = _rot_y(rings.normal_bc, th)
        c_r = x_ref[None, None, :] + _rot_y(rings.corners - ac.ref_point[None, None, :], th)
        te_l_r = x_ref[None, :] + _rot_y(rings.te_left - ac.ref_point[None, :], th)
        te_r_r = x_ref[None, :] + _rot_y(rings.te_right - ac.ref_point[None, :], th)

        cp_i = ground.reflect_points(cp_r)
        n_i = ground.reflect_directions(n_r)
        c_i = ground.reflect_points(c_r)
        te_l_i = ground.reflect_points(te_l_r)
        te_r_i = ground.reflect_points(te_r_r)

        cp_2 = np.concatenate([cp_r, cp_i], axis=0)
        n_2 = np.concatenate([n_r, n_i], axis=0)
        c_2 = np.concatenate([c_r, c_i], axis=0)
        te_l_2 = np.concatenate([te_l_r, te_l_i], axis=0)
        te_r_2 = np.concatenate([te_r_r, te_r_i], axis=0)

        if step > 0:
            wake_c_2[: step - 1] += wake_conv_2[None, :, None, :]
            aft_l = prev_te_l_2 + wake_conv_2
            aft_r = prev_te_r_2 + wake_conv_2
            wake_c_2[step - 1, :, 0, :] = te_l_2
            wake_c_2[step - 1, :, 1, :] = te_r_2
            wake_c_2[step - 1, :, 2, :] = aft_r
            wake_c_2[step - 1, :, 3, :] = aft_l
            te_idx_2 = np.concatenate([te_idx, te_idx + n_p])
            wake_g_2[step - 1] = G_hist[step - 1, te_idx_2]

        A_2 = VR.ring_normal_influence(cp_2, n_2, c_2, rc_2)
        v_surf_r = zd * e_h[None, :] + thd * _ey_cross(cp_r - x_ref[None, :])
        v_surf_i = ground.reflect_directions(v_surf_r)
        v_surf_2 = np.concatenate([v_surf_r, v_surf_i], axis=0)
        rhs_2 = np.sum((v_surf_2 - V_0[None, :]) * n_2, axis=1)

        if step > 0:
            wc_flat = wake_c_2[:step].reshape(-1, 4, 3)
            wg_flat = wake_g_2[:step].ravel()
            rc_w_flat = np.tile(rc_w_2, step)
            rhs_2 -= VR.ring_normal_influence(cp_2, n_2, wc_flat, rc_w_flat) @ wg_flat

        G_2 = np.linalg.solve(A_2, rhs_2)
        G_hist[step] = G_2

        # Antiphase symmetry: mirror wing circulation is exact negative of real wing.
        scale_g = max(float(np.max(np.abs(G_2[:n_p]))), 1.0)
        assert np.max(np.abs(G_2[n_p:] + G_2[:n_p])) <= 1e-12 * scale_g

        # Compute real-wing loads using the two-wing free-air induced velocity.
        arm_all = _rot_y(arm_b, th)
        P_load = x_ref[None, :] + arm_all
        seg_vec = _rot_y(seg_vec_b, th)
        seg_normal = _rot_y(seg_normal_b, th)
        v_surf_load = zd * e_h[None, :] + thd * _ey_cross(arm_all)
        v_ind_load = VR.ring_velocity(P_load, c_2, rc_2).swapaxes(1, 2) @ G_2
        if step > 0:
            v_ind_load += VR.ring_velocity(P_load, wc_flat, rc_w_flat).swapaxes(1, 2) @ wg_flat
        V_tot = V_0[None, :] - v_surf_load + v_ind_load
        V_eff = V_tot - np.sum(V_tot * seg_normal, axis=1, keepdims=True) * seg_normal

        g_hs = _rings_to_horseshoe(G_2[:n_p], rings.n_chord)
        g_leg = leg_strength(lat, g_hs)
        g_all = np.concatenate([g_hs] + [g_leg] * n_leg_sets)
        F_seg = (cond.rho * g_all * seg_scale)[:, None] * cross3(V_eff, seg_vec)
        F_tot = np.sum(F_seg, axis=0)
        M_tot = np.sum(cross3(arm_all, F_seg), axis=0)

        if step == 1:
            dG_dt = (G_hist[1, :n_p] - G_hist[0, :n_p]) / dt
        elif step >= 2:
            dG_dt = (3.0 * G_hist[step, :n_p] - 4.0 * G_hist[step - 1, :n_p] + G_hist[step - 2, :n_p]) / (2.0 * dt)
        else:
            dG_dt = np.zeros(n_p)
        F_am = (cond.rho * dG_dt)[:, None] * _rot_y(area_vec_b, th)
        F_tot += np.sum(F_am, axis=0)
        M_tot += np.sum(cross3(_rot_y(arm_ring_b, th), F_am), axis=0)

        if step >= 1:
            arm_te = _rot_y(arm_te_b, th)
            p_te = x_ref[None, :] + arm_te
            l_te = _rot_y(l_te_b, th)
            v_surf_te = zd * e_h[None, :] + thd * _ey_cross(arm_te)
            v_ind_te = VR.ring_velocity(p_te, c_2, rc_2).swapaxes(1, 2) @ G_2
            v_ind_te += VR.ring_velocity(p_te, wc_flat, rc_w_flat).swapaxes(1, 2) @ wg_flat
            V_te = V_0[None, :] - v_surf_te + v_ind_te
            dg_te = f_te * (wake_g_2[step - 1, :n_s] - G_2[te_idx])
            F_te = (cond.rho * dg_te)[:, None] * cross3(V_te, l_te)
            F_tot += np.sum(F_te, axis=0)
            M_tot += np.sum(cross3(arm_te, F_te), axis=0)

        CL_2[step] = float((F_tot @ L_0) / qS)
        CD_2[step] = float((F_tot @ d_0) / qS)
        CY_2[step] = float((F_tot @ S_0) / qS)
        Cl_2[step] = float(-M_tot[0] / (qS * ac.b_ref))
        Cm_2[step] = float(M_tot[1] / (qS * ac.c_ref))
        Cn_2[step] = float(-M_tot[2] / (qS * ac.b_ref))

        prev_te_l_2 = te_l_2
        prev_te_r_2 = te_r_2

    # Final bound circulation and all per-step loads match to round-off.
    scale_g = max(float(np.max(np.abs(res_ge.gamma_bound))), 1.0)
    assert np.max(np.abs(G_hist[-1, :n_p] - res_ge.gamma_bound)) <= 1e-12 * scale_g
    for name, ref_arr in (("CL", CL_2), ("CD", CD_2), ("CY", CY_2), ("Cl", Cl_2), ("Cm", Cm_2), ("Cn", Cn_2)):
        got = getattr(res_ge, name)
        scale = max(float(np.max(np.abs(ref_arr))), 1.0)
        assert np.max(np.abs(got - ref_arr)) <= 1e-12 * scale, f"{name} mismatch"

    # Normal velocity induced on the ground plane at the last step vanishes to round-off.
    P_gnd = cp_r - ground.height(cp_r)[:, None] * ground.normal[None, :]
    n_gnd = np.broadcast_to(ground.normal, P_gnd.shape)
    v_n_gnd = (
        VR.ring_normal_influence(P_gnd, n_gnd, c_2, rc_2) @ G_hist[-1]
        + VR.ring_normal_influence(P_gnd, n_gnd, wc_flat, rc_w_flat) @ wg_flat
    )
    v_n_real = (
        VR.ring_normal_influence(P_gnd, n_gnd, c_r, rings.rc) @ G_hist[-1, :n_p]
        + VR.ring_normal_influence(
            P_gnd, n_gnd, wake_c_2[:n_steps, :n_s].reshape(-1, 4, 3), np.tile(rings.rc[te_idx], n_steps),
        ) @ wake_g_2[:n_steps, :n_s].ravel()
    )
    assert np.max(np.abs(v_n_gnd)) <= 1e-13 * max(float(np.max(np.abs(v_n_real))), 1.0)


# ---------------------------------------------------------------------------
# 6. Backend parity: numpy and Numba ring kernels give the same loads
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("h_ground", [None, 0.5])
@pytest.mark.parametrize("case_kind", ["swept", "cambered", "wing_tail"])
def test_backends_numpy_equals_numba(h_ground, case_kind):
    """Numpy and Numba ring kernels produce identical loads and circulations to round-off."""
    if not VR_NB.HAVE_NUMBA:
        pytest.skip("Numba is not available")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if case_kind == "cambered":
            wing = vt.LiftingSurface(
                name="cambered",
                semi_span=3.0,
                n_panels=4,
                sections=[
                    vt.WingSection(y_frac=0.0, chord=1.0, shape="4412"),
                    vt.WingSection(y_frac=1.0, chord=1.0, shape="4412"),
                ],
            )
            ac = vt.Aircraft(surfaces=[wing], ref_point=np.array([0.25, 0.0, 0.0]))
            ac.compute_reference_values()
        elif case_kind == "wing_tail":
            wing = vt.LiftingSurface(
                name="wing",
                semi_span=3.0,
                n_panels=3,
                sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=0.8)],
            )
            tail = vt.LiftingSurface(
                name="tail",
                semi_span=1.0,
                n_panels=2,
                position=np.array([3.0, 0.0, 0.25]),
                sections=[vt.WingSection(y_frac=0.0, chord=0.5), vt.WingSection(y_frac=1.0, chord=0.4)],
            )
            ac = vt.Aircraft(surfaces=[wing, tail], ref_point=np.array([0.25, 0.0, 0.0]))
            ac.compute_reference_values()
        else:
            ac = _swept_tapered_wing(n_panels=4)
        lat = build_lattice(ac, n_chord=3, chord_spacing="uniform")

    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=np.radians(4.0), h=h_ground)
    ground = (
        make_ground_plane(lat, h_ground, cond.alpha, cond.beta, ref_point=ac.ref_point, height_ref="ref")
        if h_ground is not None
        else None
    )

    n_steps = 6
    z_arr = 0.03 * np.sin(np.linspace(0.0, 1.2, n_steps + 1))
    th_arr = np.radians(1.5) * np.sin(np.linspace(0.0, 1.5, n_steps + 1))

    r_np = _solve_time_rings(
        lat, cond, S_ref=ac.S_ref, b_ref=ac.b_ref, c_ref=ac.c_ref, ref_point=ac.ref_point,
        z=z_arr, theta=th_arr, n_steps=n_steps, ground=ground, backend="numpy",
    )
    r_nb = _solve_time_rings(
        lat, cond, S_ref=ac.S_ref, b_ref=ac.b_ref, c_ref=ac.c_ref, ref_point=ac.ref_point,
        z=z_arr, theta=th_arr, n_steps=n_steps, ground=ground, backend="numba",
    )

    for name in ("CL", "CD", "CY", "Cl", "Cm", "Cn", "strip_cl", "gamma_bound", "gamma_wake"):
        a = getattr(r_np, name)
        b = getattr(r_nb, name)
        scale = max(float(np.max(np.abs(a))), 1.0)
        assert np.max(np.abs(a - b)) <= 1e-12 * scale, f"{name} backend mismatch"


# ---------------------------------------------------------------------------
# 7. Diagnostics, junction topology, and input checks
# ---------------------------------------------------------------------------


def test_dt_deviation_note_and_junction_topology():
    """Time step differing by > 25 % from dx_TE / V adds a note; joined winglet keeps continuous wake."""
    wing = vt.LiftingSurface(
        name="wing",
        semi_span=3.0,
        n_panels=4,
        sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=1.0)],
    )
    winglet = vt.LiftingSurface(
        name="winglet",
        semi_span=0.5,
        n_panels=2,
        is_symmetric=False,
        dihedral=np.radians(90.0),
        position=np.array([0.0, 3.0, 0.0]),
        sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=0.6)],
    )
    ac = vt.Aircraft(surfaces=[wing, winglet], ref_point=np.array([0.25, 0.0, 0.0]))
    ac.compute_reference_values()
    lat = build_lattice(ac, n_chord=3, chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=np.radians(3.0))

    n_steps = 4
    zeros = np.zeros(n_steps + 1)
    r_def = _solve_time_rings(
        lat, cond, S_ref=ac.S_ref, b_ref=ac.b_ref, c_ref=ac.c_ref, ref_point=ac.ref_point,
        z=zeros, theta=zeros, n_steps=n_steps,
    )
    assert r_def.notes == []
    assert np.all(np.isfinite(r_def.CL))
    # Joined edge between right wing tip and winglet root stays continuous in every wake row.
    s_w_r = lat.surfaces[0].strips.stop - 1
    s_wl_r = lat.surfaces[1].strips.start
    for row in range(n_steps):
        assert np.allclose(r_def.wake_corners[row, s_w_r, 1], r_def.wake_corners[row, s_wl_r, 0], atol=1e-13)
        assert np.allclose(r_def.wake_corners[row, s_w_r, 2], r_def.wake_corners[row, s_wl_r, 3], atol=1e-13)

    # dt within 25 % gives no note; dt outside 25 % adds a note.
    r_near = _solve_time_rings(
        lat, cond, S_ref=ac.S_ref, b_ref=ac.b_ref, c_ref=ac.c_ref, ref_point=ac.ref_point,
        z=zeros, theta=zeros, n_steps=n_steps, dt=1.2 * r_def.dt_default,
    )
    assert r_near.notes == []

    r_far = _solve_time_rings(
        lat, cond, S_ref=ac.S_ref, b_ref=ac.b_ref, c_ref=ac.c_ref, ref_point=ac.ref_point,
        z=zeros, theta=zeros, n_steps=n_steps, dt=0.5 * r_def.dt_default,
    )
    assert len(r_far.notes) == 1
    assert "25 %" in r_far.notes[0]


def test_input_checks():
    """Every invalid input raises InputError with a clear message."""
    ac = _rect_wing(n_panels=4)
    lat = build_lattice(ac, n_chord=2, chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=np.radians(2.0))
    common = {"S_ref": ac.S_ref, "b_ref": ac.b_ref, "c_ref": ac.c_ref, "ref_point": ac.ref_point}

    # n_steps below 1 or non-integer.
    for bad_n in (0, -3, True, 2.5):
        with pytest.raises(InputError, match="n_steps"):
            _solve_time_rings(lat, cond, z=np.zeros(2), theta=np.zeros(2), n_steps=bad_n, **common)

    # dt not positive or not finite.
    for bad_dt in (0.0, -0.01, np.nan, np.inf, -np.inf):
        with pytest.raises(InputError, match="dt"):
            _solve_time_rings(lat, cond, z=np.zeros(3), theta=np.zeros(3), n_steps=2, dt=bad_dt, **common)

    # Motion arrays of wrong length or non-finite.
    with pytest.raises(InputError, match="z"):
        _solve_time_rings(lat, cond, z=np.zeros(2), theta=np.zeros(3), n_steps=2, **common)
    with pytest.raises(InputError, match="theta"):
        _solve_time_rings(lat, cond, z=np.zeros(3), theta=np.zeros(4), n_steps=2, **common)
    with pytest.raises(InputError, match="z"):
        _solve_time_rings(lat, cond, z=np.array([0.0, np.nan, 0.0]), theta=np.zeros(3), n_steps=2, **common)
    with pytest.raises(InputError, match="theta"):
        _solve_time_rings(lat, cond, z=np.zeros(3), theta=np.array([0.0, np.inf, 0.0]), n_steps=2, **common)
    with pytest.raises(InputError, match="z"):
        _solve_time_rings(lat, cond, z=lambda t: np.nan, theta=np.zeros(3), n_steps=2, **common)

    # Lattice refused by build_ring_lattice (lifting-line collocation).
    lat_llt = build_lattice(ac, collocation="llt", n_chord=1)
    with pytest.raises(InputError, match="collocation"):
        _solve_time_rings(lat_llt, cond, z=np.zeros(3), theta=np.zeros(3), n_steps=2, **common)

    # Wing below or touching the ground raises InputError naming step and time.
    ground = make_ground_plane(lat, 0.5, cond.alpha, cond.beta, ref_point=ac.ref_point, height_ref="ref")
    # Strikes the ground at step 3 (z drops by -0.6 m from h = 0.5 m).
    z_strike = np.array([0.0, -0.1, -0.2, -0.6, -0.7])
    with pytest.raises(InputError, match=r"step 3") as exc_info:
        _solve_time_rings(
            lat, cond, z=z_strike, theta=np.zeros(5), n_steps=4, ground=ground, **common,
        )
    assert "t =" in str(exc_info.value)

    # Also refuses ground strike at step 0.
    z_strike0 = np.array([-0.55, 0.0, 0.0])
    with pytest.raises(InputError, match=r"step 0") as exc_info0:
        _solve_time_rings(
            lat, cond, z=z_strike0, theta=np.zeros(3), n_steps=2, ground=ground, **common,
        )
    assert "t =" in str(exc_info0.value)


# ---------------------------------------------------------------------------
# 8. Six-axis prescribed rigid-body motion
# ---------------------------------------------------------------------------


def _flapped_rect_wing(
    deflection_deg: float = 0.0,
    *,
    semi_span: float = 3.0,
    chord: float = 1.0,
    n_panels: int = 4,
    eta_start: float = 0.0,
    eta_end: float = 1.0,
    hinge_x_c: float = 0.75,
    symmetric: bool = True,
    name: str = "flap",
    shape: str | None = None,
) -> vt.Aircraft:
    """Build a rectangular wing with one chordwise control surface."""
    ctrl = vt.ControlSurface(
        name=name,
        eta_start=eta_start,
        eta_end=eta_end,
        hinge_x_c=hinge_x_c,
        deflection=np.radians(deflection_deg),
        symmetric=symmetric,
    )
    wing = vt.LiftingSurface(
        name="wing",
        semi_span=semi_span,
        n_panels=n_panels,
        controls=[ctrl],
        sections=[
            vt.WingSection(y_frac=0.0, chord=chord, shape=shape),
            vt.WingSection(y_frac=1.0, chord=chord, shape=shape),
        ],
    )
    ac = vt.Aircraft(surfaces=[wing], ref_point=np.array([0.25 * chord, 0.0, 0.0]))
    ac.compute_reference_values()
    return ac


def test_rotation_321_and_angular_velocity_kinematics():
    """3-2-1 rotation matrix is proper orthonormal and matches omega x r under differentiation."""
    from ventorum.unsteady.timedomain import _angular_velocity_321, _rotation_matrix_321

    # Single-axis sign conventions (AVL): roll > 0 right wing down, pitch > 0 nose up, yaw > 0 nose right.
    r_roll = _rotation_matrix_321(np.radians(15.0), 0.0, 0.0)
    pt_right = r_roll @ np.array([0.0, 1.0, 0.0])
    assert pt_right[2] < 0.0  # right wing goes down (-z)

    r_pitch = _rotation_matrix_321(0.0, np.radians(15.0), 0.0)
    pt_aft = r_pitch @ np.array([1.0, 0.0, 0.0])
    assert pt_aft[2] < 0.0  # trailing edge goes down (nose up)

    r_yaw = _rotation_matrix_321(0.0, 0.0, np.radians(15.0))
    pt_nose = r_yaw @ np.array([-1.0, 0.0, 0.0])
    assert pt_nose[1] > 0.0  # nose moves right (+y)

    # General coupled 3-2-1 motion: finite-difference dR/dt @ R^T equals [omega]x.
    roll_f = lambda t: 0.25 * math.sin(1.3 * t + 0.2)  # noqa: E731
    pitch_f = lambda t: -0.18 * math.cos(0.9 * t - 0.1)  # noqa: E731
    yaw_f = lambda t: 0.31 * math.sin(0.7 * t + 0.5)  # noqa: E731
    roll_df = lambda t: 0.25 * 1.3 * math.cos(1.3 * t + 0.2)  # noqa: E731
    pitch_df = lambda t: 0.18 * 0.9 * math.sin(0.9 * t - 0.1)  # noqa: E731
    yaw_df = lambda t: 0.31 * 0.7 * math.cos(0.7 * t + 0.5)  # noqa: E731

    t0 = 0.73
    h = 1.0e-7
    R0 = _rotation_matrix_321(roll_f(t0), pitch_f(t0), yaw_f(t0))
    assert np.allclose(R0 @ R0.T, np.eye(3), atol=1e-15)
    assert abs(float(np.linalg.det(R0)) - 1.0) <= 1e-15

    Rp = _rotation_matrix_321(roll_f(t0 + h), pitch_f(t0 + h), yaw_f(t0 + h))
    Rm = _rotation_matrix_321(roll_f(t0 - h), pitch_f(t0 - h), yaw_f(t0 - h))
    R_dot = (Rp - Rm) / (2.0 * h)
    omega = _angular_velocity_321(pitch_f(t0), yaw_f(t0), roll_df(t0), pitch_df(t0), yaw_df(t0))

    vec = np.array([0.4, -1.7, 0.9])
    v_fd = R_dot @ (R0.T @ vec)
    v_kin = np.cross(omega, vec)
    assert np.allclose(v_fd, v_kin, atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize("h_ground", [None, 0.5])
@pytest.mark.parametrize("use_callable", [False, True])
def test_motion_heave_pitch_matches_legacy_z_theta(h_ground, use_callable):
    """Heave and pitch through motion equal the legacy z and theta path to round-off."""
    ac = _swept_tapered_wing(n_panels=4)
    lat = build_lattice(ac, n_chord=3, chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=np.radians(4.0), h=h_ground)
    ground = (
        make_ground_plane(lat, h_ground, cond.alpha, cond.beta, ref_point=ac.ref_point, height_ref="ref")
        if h_ground is not None
        else None
    )
    n_steps = 6
    omega = 2.0 * 0.25 * cond.V_inf / ac.c_ref

    if use_callable:
        z_in = lambda t: 0.04 * math.sin(omega * t)  # noqa: E731
        th_in = lambda t: np.radians(2.0) * math.sin(omega * t + 0.4)  # noqa: E731
    else:
        t_grid = np.linspace(0.0, 1.0, n_steps + 1)
        z_in = 0.04 * np.sin(2.0 * t_grid)
        th_in = np.radians(2.0) * np.sin(2.5 * t_grid + 0.2)

    r_old = _solve_time_rings(
        lat, cond, S_ref=ac.S_ref, b_ref=ac.b_ref, c_ref=ac.c_ref, ref_point=ac.ref_point,
        z=z_in, theta=th_in, n_steps=n_steps, ground=ground,
    )
    r_new = _solve_time_rings(
        lat, cond, S_ref=ac.S_ref, b_ref=ac.b_ref, c_ref=ac.c_ref, ref_point=ac.ref_point,
        motion={"z": z_in, "pitch": th_in}, n_steps=n_steps, ground=ground,
    )

    for name in ("CL", "CD", "CY", "Cl", "Cm", "Cn", "strip_cl", "gamma_bound", "gamma_wake"):
        a = getattr(r_old, name)
        b = getattr(r_new, name)
        scale = max(float(np.max(np.abs(a))), 1.0)
        assert np.max(np.abs(a - b)) <= 1e-13 * scale, f"{name} mismatch"

    assert np.array_equal(r_new.pitch, r_new.theta)
    assert np.array_equal(r_new.pitch_dot, r_new.theta_dot)
    assert np.all(r_new.x == 0.0) and np.all(r_new.y == 0.0)
    assert np.all(r_new.roll == 0.0) and np.all(r_new.yaw == 0.0)
    assert r_new.controls == {}


@pytest.mark.parametrize("offset_kind", ["roll", "yaw", "y", "six_axis"])
@pytest.mark.parametrize("h_ground", [None, 1.0])
def test_constant_rigid_offsets_match_transformed_lattice(offset_kind, h_ground):
    """Constant rigid offsets match a lattice built already rotated and shifted to round-off."""
    from ventorum.geometry.transform import transform_lattice
    from ventorum.unsteady.timedomain import _rotation_matrix_321

    ac = _swept_tapered_wing(n_panels=4)
    lat = build_lattice(ac, n_chord=3, chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=np.radians(4.0), h=h_ground)
    ground = (
        make_ground_plane(lat, h_ground, cond.alpha, cond.beta, ref_point=ac.ref_point, height_ref="ref")
        if h_ground is not None
        else None
    )

    n_steps = 6
    vals = {"x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}
    if offset_kind == "roll":
        vals["roll"] = float(np.radians(8.0))
    elif offset_kind == "yaw":
        vals["yaw"] = float(np.radians(7.0))
    elif offset_kind == "y":
        vals["y"] = 0.5
    else:
        vals.update({
            "x": 0.25,
            "y": -0.4,
            "z": 0.15,
            "roll": float(np.radians(6.0)),
            "pitch": float(np.radians(3.0)),
            "yaw": float(np.radians(-5.0)),
        })

    motion = {k: np.full(n_steps + 1, v, dtype=np.float64) for k, v in vals.items() if v != 0.0}

    d_0 = freestream_direction(cond.alpha, cond.beta)
    L_0 = lift_direction(cond.alpha)
    S_0 = side_direction(cond.alpha, cond.beta)
    e_z = L_0 if ground is None else np.asarray(ground.normal, dtype=np.float64)
    e_y = S_0 if ground is None else np.cross(e_z, d_0)
    e_y = e_y / np.linalg.norm(e_y)
    shift = vals["x"] * d_0 + vals["y"] * e_y + vals["z"] * e_z
    R = _rotation_matrix_321(vals["roll"], vals["pitch"], vals["yaw"])
    lat_moved = transform_lattice(lat, rotation=R, about=ac.ref_point, translation=shift)

    r_motion = _solve_time_rings(
        lat, cond, S_ref=ac.S_ref, b_ref=ac.b_ref, c_ref=ac.c_ref, ref_point=ac.ref_point,
        motion=motion, n_steps=n_steps, ground=ground,
    )
    r_moved = _solve_time_rings(
        lat_moved, cond, S_ref=ac.S_ref, b_ref=ac.b_ref, c_ref=ac.c_ref, ref_point=ac.ref_point + shift,
        n_steps=n_steps, ground=ground,
    )

    for name in ("CL", "CD", "CY", "Cl", "Cm", "Cn", "strip_cl", "gamma_bound", "gamma_wake"):
        a = getattr(r_motion, name)
        b = getattr(r_moved, name)
        scale = max(float(np.max(np.abs(b))), 1.0)
        assert np.max(np.abs(a - b)) <= 1e-12 * scale, f"{offset_kind} h={h_ground} {name} mismatch"


@pytest.mark.parametrize("offset_kind", ["roll", "yaw", "y"])
def test_constant_rigid_offsets_steady_limit_matches_vlm(monkeypatch, offset_kind):
    """Constant roll, yaw or y offset converges to steady VLM on the moved lattice at 50 chords.

    Tolerance derivation matches test_steady_limit_matches_analyze_exact_equivalence:
    at wake length L_w = 50 c on an AR 6 wing, the finite-wake starting-vortex
    deficit is about (a_3d AR) / (16 pi (L_w / c)^2) ~ 2.2e-4, bounded by 5e-4.
    CORE_RADIUS_FRACTION = 1e-4 avoids 3-D rotated leg-midpoint round-off
    singularities while keeping exact VLM-ring parity.
    """
    from ventorum.aero.loads import compute_loads
    from ventorum.geometry.transform import transform_lattice
    from ventorum.solvers.core import solve_vlm
    from ventorum.unsteady.timedomain import _rotation_matrix_321

    ac = _rect_wing(semi_span=3.0, chord=1.0, n_panels=6)
    st = vt.SolverSettings(
        solver_type="vlm", n_panels=6, n_chord=4, spacing="half-cosine", chord_spacing="uniform",
    )
    cond = vt.FlightCondition(V_inf=25.0, rho=1.225, alpha=np.radians(4.0))
    monkeypatch.setattr(lattice_mod, "CORE_RADIUS_FRACTION", 1.0e-4)
    lat = build_lattice(ac, st, collocation="vlm", n_chord=4, chord_spacing="uniform")

    n_steps = 200  # 50 chords at n_chord = 4
    vals = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0, "y": 0.0}
    if offset_kind == "roll":
        vals["roll"] = float(np.radians(10.0))
    elif offset_kind == "yaw":
        vals["yaw"] = float(np.radians(8.0))
    else:
        vals["y"] = 0.5

    motion = {k: np.full(n_steps + 1, v, dtype=np.float64) for k, v in vals.items() if v != 0.0}
    S_0 = side_direction(cond.alpha, cond.beta)
    shift = vals["y"] * S_0
    R = _rotation_matrix_321(vals["roll"], vals["pitch"], vals["yaw"])
    lat_moved = transform_lattice(lat, rotation=R, about=ac.ref_point, translation=shift)
    wd = freestream_direction(cond.alpha, cond.beta)

    gamma_vlm, _ = solve_vlm(lat_moved, cond, ground=None, use_symmetry=False, wake_dir=wd, ref_point=ac.ref_point + shift)
    loads_ref = compute_loads(
        lat_moved, gamma_vlm, cond, ac.S_ref, ac.b_ref, ac.c_ref,
        ground=None, ref_point=ac.ref_point + shift, wake_dir=wd,
    )

    res = _solve_time_rings(
        lat, cond, S_ref=ac.S_ref, b_ref=ac.b_ref, c_ref=ac.c_ref, ref_point=ac.ref_point,
        motion=motion, n_steps=n_steps, te_extension_fraction=0.0,
    )

    tol = 5.0e-4
    t_ref = loads_ref.totals
    assert abs(res.CL[-1] - t_ref.CL) <= tol * max(abs(t_ref.CL), 1.0)
    assert abs(res.CD[-1] - t_ref.CDi_nearfield) <= tol * max(abs(t_ref.CDi_nearfield), 1.0)
    assert abs(res.CY[-1] - t_ref.CY) <= tol * max(abs(t_ref.CY), 1.0)
    assert abs(res.Cl[-1] - t_ref.Cl) <= tol * max(abs(t_ref.Cl), 1.0)
    assert abs(res.Cm[-1] - t_ref.Cm) <= tol * max(abs(t_ref.Cm), 1.0)
    assert abs(res.Cn[-1] - t_ref.Cn) <= tol * max(abs(t_ref.Cn), 1.0)


def test_symmetry_pure_roll_and_pure_yaw():
    """Pure roll and pure yaw of a symmetric wing obey exact symmetry in free air.

    For pure roll of a flat symmetric wing at alpha = beta = 0, circulation is
    antisymmetric across y = 0, so CL, Cm, CY and Cn vanish to round-off while
    Cl is nonzero (roll damping) and near-field CD is even in roll (O(p^2)).
    For pure yaw at alpha = beta = 0, reflection across z = 0 leaves a flat
    symmetric wing (and a symmetric vertical fin spanning z in [-h, +h])
    invariant, so CL and Cm vanish to round-off while CY and Cn are nonzero on
    the fin.
    """
    ac_flat = _rect_wing(semi_span=3.0, chord=1.0, n_panels=6)
    lat_flat = build_lattice(ac_flat, n_chord=3, chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=0.0, beta=0.0)

    n_steps = 8
    omega = 2.0 * 0.2 * cond.V_inf / ac_flat.c_ref
    roll_pos = lambda t: np.radians(5.0) * math.sin(omega * t)  # noqa: E731
    roll_neg = lambda t: -np.radians(5.0) * math.sin(omega * t)  # noqa: E731

    r_roll = _solve_time_rings(
        lat_flat, cond, S_ref=ac_flat.S_ref, b_ref=ac_flat.b_ref, c_ref=ac_flat.c_ref,
        ref_point=ac_flat.ref_point, motion={"roll": roll_pos}, n_steps=n_steps,
    )
    r_roll_neg = _solve_time_rings(
        lat_flat, cond, S_ref=ac_flat.S_ref, b_ref=ac_flat.b_ref, c_ref=ac_flat.c_ref,
        ref_point=ac_flat.ref_point, motion={"roll": roll_neg}, n_steps=n_steps,
    )

    assert np.max(np.abs(r_roll.CL)) <= 1e-13
    assert np.max(np.abs(r_roll.Cm)) <= 1e-13
    assert np.max(np.abs(r_roll.CY)) <= 1e-13
    assert np.max(np.abs(r_roll.Cn)) <= 1e-13
    assert np.max(np.abs(r_roll.Cl)) > 1e-3
    # Near-field drag is even in roll rate (O(p^2)): CD(+roll) == CD(-roll) to round-off.
    assert np.max(np.abs(r_roll.CD - r_roll_neg.CD)) <= 1e-13 * max(float(np.max(np.abs(r_roll.CD))), 1.0)
    # Antisymmetric bound ring strengths across y = 0.
    g = r_roll.gamma_bound
    g_scale = max(float(np.max(np.abs(g))), 1.0)
    assert np.max(np.abs(g + g[lat_flat.panel_mirror])) <= 1e-12 * g_scale

    # Pure yaw on a flat symmetric wing gives CL = Cm = 0 to round-off.
    yaw_fn = lambda t: np.radians(5.0) * math.sin(omega * t)  # noqa: E731
    r_yaw_flat = _solve_time_rings(
        lat_flat, cond, S_ref=ac_flat.S_ref, b_ref=ac_flat.b_ref, c_ref=ac_flat.c_ref,
        ref_point=ac_flat.ref_point, motion={"yaw": yaw_fn}, n_steps=n_steps,
    )
    assert np.max(np.abs(r_yaw_flat.CL)) <= 1e-13
    assert np.max(np.abs(r_yaw_flat.Cm)) <= 1e-13

    # Pure yaw on a z-symmetric wing + vertical fin gives nonzero CY and Cn with CL = Cm = 0.
    wing = vt.LiftingSurface(
        name="wing",
        semi_span=3.0,
        n_panels=4,
        sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=1.0)],
    )
    fin = vt.LiftingSurface(
        name="fin",
        semi_span=1.0,
        n_panels=4,
        spacing="uniform",
        is_symmetric=False,
        dihedral=np.radians(90.0),
        position=np.array([2.0, 0.0, -0.5]),
        sections=[vt.WingSection(y_frac=0.0, chord=0.6), vt.WingSection(y_frac=1.0, chord=0.6)],
    )
    ac_fin = vt.Aircraft(surfaces=[wing, fin], ref_point=np.array([0.25, 0.0, 0.0]))
    ac_fin.compute_reference_values()
    lat_fin = build_lattice(ac_fin, n_chord=3, chord_spacing="uniform")

    r_yaw_fin = _solve_time_rings(
        lat_fin, cond, S_ref=ac_fin.S_ref, b_ref=ac_fin.b_ref, c_ref=ac_fin.c_ref,
        ref_point=ac_fin.ref_point, motion={"yaw": yaw_fn}, n_steps=n_steps,
    )
    assert np.max(np.abs(r_yaw_fin.CL)) <= 1e-13
    assert np.max(np.abs(r_yaw_fin.Cm)) <= 1e-13
    assert np.max(np.abs(r_yaw_fin.Cl)) <= 1e-13
    assert np.max(np.abs(r_yaw_fin.CY)) > 1e-3
    assert np.max(np.abs(r_yaw_fin.Cn)) > 1e-3


def test_steady_roll_rate_matches_rate_derivatives(monkeypatch):
    """Slow steady roll rate gives Cl_p matching ventorum.rate_derivatives after start transient.

    Tolerance derivation: at p_hat = p b / (2 V) = 1e-3 and wake length L_w = 30 c
    (120 steps at n_chord = 4), the total roll angle is phi_max = p_hat * (2 L_w / b) = 0.01 rad,
    so the helical-wake departure from the straight quasi-steady wake of rate_derivatives
    is O(phi_max^2) ~ 1e-4. The antisymmetric starting vortex forms a quadrupole whose
    downwash deficit at L_w = 30 c (5 spans) is O((b / (2 L_w))^3 / (2 pi)) ~ 1.5e-4.
    A relative tolerance of 1e-3 bounds both effects with ~4x margin.
    """
    ac = _rect_wing(semi_span=3.0, chord=1.0, n_panels=6)
    st = vt.SolverSettings(
        solver_type="vlm", n_panels=6, n_chord=4, spacing="half-cosine", chord_spacing="uniform",
    )
    cond = vt.FlightCondition(V_inf=25.0, rho=1.225, alpha=0.0)

    monkeypatch.setattr(lattice_mod, "CORE_RADIUS_FRACTION", 1.0e-4)
    lat = build_lattice(ac, st, collocation="vlm", n_chord=4, chord_spacing="uniform")
    rd = vt.rate_derivatives(ac, cond, st, step=1.0e-3, axes="body")
    cl_p_ref = float(rd.derivatives["Cl_p"])
    assert cl_p_ref < -0.3  # roll damping is negative

    p_hat = 1.0e-3
    p_rate = 2.0 * cond.V_inf * p_hat / ac.b_ref
    n_steps = 120  # 30 chords at n_chord = 4
    res = _solve_time_rings(
        lat, cond, S_ref=ac.S_ref, b_ref=ac.b_ref, c_ref=ac.c_ref, ref_point=ac.ref_point,
        motion={"roll": lambda t: p_rate * t}, n_steps=n_steps,
        te_extension_fraction=0.0,
    )
    cl_p_td = float(res.Cl[-1]) / p_hat
    rel_err = abs(cl_p_td - cl_p_ref) / abs(cl_p_ref)
    assert rel_err <= 1.0e-3, f"Cl_p_td={cl_p_td:.6f}, Cl_p_ref={cl_p_ref:.6f}, rel_err={rel_err:.4e}"


# ---------------------------------------------------------------------------
# 9. Moving control surfaces: fixed flap, convergence, ground parity, backends
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case_kind",
    ["full_flap", "aileron_antisym", "cambered_flap"],
)
def test_fixed_flap_controls_match_built_deflected_lattice(case_kind):
    """Constant deflection through controls equals a lattice built with that deflection to round-off."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if case_kind == "full_flap":
            ac0 = _flapped_rect_wing(0.0, n_panels=4, hinge_x_c=0.75, symmetric=True, name="flap")
            ac_d = _flapped_rect_wing(15.0, n_panels=4, hinge_x_c=0.75, symmetric=True, name="flap")
            ctrl_name, deg = "flap", 15.0
        elif case_kind == "aileron_antisym":
            ac0 = _flapped_rect_wing(
                0.0, n_panels=6, eta_start=0.5, eta_end=1.0, hinge_x_c=0.75, symmetric=False, name="ail",
            )
            ac_d = _flapped_rect_wing(
                10.0, n_panels=6, eta_start=0.5, eta_end=1.0, hinge_x_c=0.75, symmetric=False, name="ail",
            )
            ctrl_name, deg = "ail", 10.0
        else:
            ac0 = _flapped_rect_wing(0.0, n_panels=4, hinge_x_c=0.75, symmetric=True, name="flap", shape="4412")
            ac_d = _flapped_rect_wing(12.0, n_panels=4, hinge_x_c=0.75, symmetric=True, name="flap", shape="4412")
            ctrl_name, deg = "flap", 12.0
        lat0 = build_lattice(ac0, n_chord=4, chord_spacing="uniform")
        lat_d = build_lattice(ac_d, n_chord=4, chord_spacing="uniform")

    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=np.radians(3.0))
    n_steps = 6
    delta_rad = float(np.radians(deg))

    r_ctrl = _solve_time_rings(
        lat0, cond, S_ref=ac0.S_ref, b_ref=ac0.b_ref, c_ref=ac0.c_ref, ref_point=ac0.ref_point,
        controls={ctrl_name: np.full(n_steps + 1, delta_rad)}, n_steps=n_steps,
    )
    r_built = _solve_time_rings(
        lat_d, cond, S_ref=ac_d.S_ref, b_ref=ac_d.b_ref, c_ref=ac_d.c_ref, ref_point=ac_d.ref_point,
        n_steps=n_steps,
    )

    # Geometry parity via shed wake corners (which trace the moved TE closure at every step).
    assert np.max(np.abs(r_ctrl.wake_corners - r_built.wake_corners)) <= 1e-13
    for name in ("CL", "CD", "CY", "Cl", "Cm", "Cn", "strip_cl", "gamma_bound", "gamma_wake"):
        a = getattr(r_ctrl, name)
        b = getattr(r_built, name)
        scale = max(float(np.max(np.abs(b))), 1.0)
        assert np.max(np.abs(a - b)) <= 1e-12 * scale, f"{case_kind} {name} mismatch"

    # Non-zero built lattice moved back to 0 deg matches undeflected lattice to round-off.
    r_back = _solve_time_rings(
        lat_d, cond, S_ref=ac_d.S_ref, b_ref=ac_d.b_ref, c_ref=ac_d.c_ref, ref_point=ac_d.ref_point,
        controls={ctrl_name: np.zeros(n_steps + 1)}, n_steps=n_steps,
    )
    r_clean = _solve_time_rings(
        lat0, cond, S_ref=ac0.S_ref, b_ref=ac0.b_ref, c_ref=ac0.c_ref, ref_point=ac0.ref_point,
        n_steps=n_steps,
    )
    assert np.max(np.abs(r_back.wake_corners - r_clean.wake_corners)) <= 1e-13
    for name in ("CL", "CD", "CY", "Cl", "Cm", "Cn", "gamma_bound"):
        a = getattr(r_back, name)
        b = getattr(r_clean, name)
        scale = max(float(np.max(np.abs(b))), 1.0)
        assert np.max(np.abs(a - b)) <= 1e-12 * scale

    # Zero history on undeflected lattice equals no controls to round-off.
    r_zero = _solve_time_rings(
        lat0, cond, S_ref=ac0.S_ref, b_ref=ac0.b_ref, c_ref=ac0.c_ref, ref_point=ac0.ref_point,
        controls={ctrl_name: np.zeros(n_steps + 1)}, n_steps=n_steps,
    )
    for name in ("CL", "CD", "CY", "Cl", "Cm", "Cn", "gamma_bound"):
        assert np.max(np.abs(getattr(r_zero, name) - getattr(r_clean, name))) <= 1e-14


def test_fixed_flap_steady_limit_matches_analyze(monkeypatch):
    """Constant 15 deg flap through controls converges to ventorum.analyze on the deflected wing.

    Tolerance derivation matches test_steady_limit_matches_analyze_exact_equivalence:
    at 50 chords (200 steps at n_chord = 4), the finite-wake starting-vortex
    truncation error on an AR 6 wing is about 2.2e-4, bounded by 5e-4.
    CORE_RADIUS_FRACTION = 1e-4 avoids 3-D deflected-flap leg-midpoint round-off
    singularities while keeping exact VLM-ring parity.
    """
    ac0 = _flapped_rect_wing(0.0, semi_span=3.0, chord=1.0, n_panels=6, hinge_x_c=0.75)
    ac15 = _flapped_rect_wing(15.0, semi_span=3.0, chord=1.0, n_panels=6, hinge_x_c=0.75)
    st = vt.SolverSettings(
        solver_type="vlm", n_panels=6, n_chord=4, spacing="half-cosine", chord_spacing="uniform",
    )
    cond = vt.FlightCondition(V_inf=25.0, rho=1.225, alpha=np.radians(2.0))

    monkeypatch.setattr(lattice_mod, "CORE_RADIUS_FRACTION", 1.0e-4)
    lat0 = build_lattice(ac0, st, collocation="vlm", n_chord=4, chord_spacing="uniform")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        steady_ref = vt.analyze(ac15, cond, st)

    n_steps = 200  # 50 chords
    res = _solve_time_rings(
        lat0, cond, S_ref=ac0.S_ref, b_ref=ac0.b_ref, c_ref=ac0.c_ref, ref_point=ac0.ref_point,
        controls={"flap": np.full(n_steps + 1, np.radians(15.0))},
        n_steps=n_steps, te_extension_fraction=0.0,
    )

    tol = 5.0e-4
    assert abs(res.CL[-1] - steady_ref.totals.CL) <= tol * max(abs(steady_ref.totals.CL), 1.0)
    assert abs(res.Cm[-1] - steady_ref.totals.Cm) <= tol * max(abs(steady_ref.totals.Cm), 1.0)
    assert abs(res.CD[-1] - steady_ref.totals.CDi_nearfield) <= tol * max(abs(steady_ref.totals.CDi_nearfield), 1.0)


@pytest.mark.slow
@pytest.mark.parametrize("k,n_trans", [(0.05, 1.0), (0.2, 2.0)])
def test_moving_flap_small_amplitude_dt_convergence(k, n_trans):
    """Harmonic 1e-4 rad flap motion at k = 0.05 and 0.2 converges at second order under dt halving.

    Doubling n_chord from 4 to 8 to 16 halves dt = dx_TE / V_inf while keeping
    hinge_x_c = 0.75 on an exact chordwise node and dx_w = dx_TE at every level.
    """
    ac = _flapped_rect_wing(0.0, semi_span=3.0, chord=1.0, n_panels=4, hinge_x_c=0.75)
    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=0.0)
    omega = 2.0 * k * cond.V_inf / ac.c_ref
    period = 2.0 * math.pi / omega
    amp = 1.0e-4

    cl_h = []
    cm_h = []
    for n_c in (4, 8, 16):
        lat = build_lattice(ac, n_chord=n_c, chord_spacing="uniform")
        dt_use = 1.0 / (n_c * cond.V_inf)
        n_steps = int(math.ceil((n_trans + 1.0) * period / dt_use))
        res = _solve_time_rings(
            lat, cond, S_ref=ac.S_ref, b_ref=ac.b_ref, c_ref=ac.c_ref, ref_point=ac.ref_point,
            controls={"flap": lambda t: amp * math.sin(omega * t)},
            n_steps=n_steps, te_correction=True,
        )
        t_start = res.time[-1] - period
        mask = res.time >= t_start - 1e-12 * dt_use
        cl_h.append(_fit_first_harmonic(res.time[mask], res.CL[mask], omega, amp))
        cm_h.append(_fit_first_harmonic(res.time[mask], res.Cm[mask], omega, amp))

    ord_cl = math.log2(abs(cl_h[0] - cl_h[1]) / abs(cl_h[1] - cl_h[2]))
    ord_cm = math.log2(abs(cm_h[0] - cm_h[1]) / abs(cm_h[1] - cm_h[2]))
    assert 1.7 <= ord_cl <= 2.3, (
        f"k={k}: CL order={ord_cl:.3f}, Cm order={ord_cm:.3f}, CL={cl_h}, Cm={cm_h}"
    )
    assert 1.7 <= ord_cm <= 2.3, (
        f"k={k}: CL order={ord_cl:.3f}, Cm order={ord_cm:.3f}, CL={cl_h}, Cm={cm_h}"
    )


def test_ground_parity_moving_flap_antiphase_two_wing_model():
    """Moving flap at h/c = 0.5 equals an explicit antiphase two-wing free-air model to round-off."""
    from ventorum.unsteady.timedomain import _angular_velocity_321, _rotation_matrix_321

    ac0 = _flapped_rect_wing(0.0, semi_span=3.0, chord=1.0, n_panels=4, hinge_x_c=0.75)
    lat0 = build_lattice(ac0, n_chord=4, chord_spacing="uniform")
    h_ground = 0.5
    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=np.radians(4.0), h=h_ground)
    ground = make_ground_plane(lat0, h_ground, cond.alpha, cond.beta, ref_point=ac0.ref_point, height_ref="ref")

    n_steps = 6
    omega = 2.0 * 0.25 * cond.V_inf / ac0.c_ref
    z_fn = lambda t: 0.03 * math.sin(omega * t)  # noqa: E731
    th_fn = lambda t: np.radians(1.5) * math.sin(omega * t + 0.2)  # noqa: E731
    fl_fn = lambda t: np.radians(10.0) * math.sin(omega * t + 0.4)  # noqa: E731

    res_ge = _solve_time_rings(
        lat0, cond, S_ref=ac0.S_ref, b_ref=ac0.b_ref, c_ref=ac0.c_ref, ref_point=ac0.ref_point,
        motion={"z": z_fn, "pitch": th_fn},
        controls={"flap": fl_fn},
        n_steps=n_steps, ground=ground, te_correction=True,
    )

    # Independent explicit 2N x 2N free-air two-wing model rebuilding each step's deflected lattice.
    d_0 = freestream_direction(cond.alpha, cond.beta)
    L_0 = lift_direction(cond.alpha)
    S_0 = side_direction(cond.alpha, cond.beta)
    V_0 = cond.V_inf * d_0
    e_h = np.asarray(ground.normal, dtype=np.float64)
    rp = ac0.ref_point
    dt = res_ge.dt
    n_p = lat0.n_panels
    n_s = lat0.n_strips
    n_c = lat0.n_chord
    te_idx = np.flatnonzero(np.tile(np.arange(n_c), n_s) == (n_c - 1))
    dx_te = np.asarray(lat0.panel_length[te_idx], dtype=np.float64)
    f_te = dx_te / (2.0 * (dx_te + cond.V_inf * dt))
    rc_2 = np.concatenate([lat0.rc, lat0.rc])
    rc_w_2 = np.concatenate([lat0.rc[te_idx], lat0.rc[te_idx]])

    wake_conv_r = (cond.V_inf * dt) * np.tile(d_0, (n_s, 1))
    wake_conv_i = ground.reflect_directions(wake_conv_r)
    wake_conv_2 = np.concatenate([wake_conv_r, wake_conv_i], axis=0)

    wake_c_2 = np.empty((n_steps, 2 * n_s, 4, 3), dtype=np.float64)
    wake_g_2 = np.empty((n_steps, 2 * n_s), dtype=np.float64)
    G_hist = np.empty((n_steps + 1, 2 * n_p), dtype=np.float64)
    prev_te_l_2 = None
    prev_te_r_2 = None
    qS = 0.5 * cond.rho * cond.V_inf ** 2 * ac0.S_ref
    CL_2, CD_2, CY_2, Cl_2, Cm_2, Cn_2 = (np.empty(n_steps + 1) for _ in range(6))

    # Constant hinge axis on this rectangular wing: (0, 1, 0) through x = 0.75, z = 0.
    hinge_ax = np.array([0.0, 1.0, 0.0])
    hinge_pt = np.array([0.75, 0.0, 0.0])

    for step in range(n_steps + 1):
        zn, th = float(res_ge.z[step]), float(res_ge.pitch[step])
        zd, thd = float(res_ge.z_dot[step]), float(res_ge.pitch_dot[step])
        fl_ang = float(res_ge.controls["flap"][step])
        fl_dot = float(res_ge.controls_dot["flap"][step])

        # Build the deflected wing from scratch at fl_ang.
        ac_s = _flapped_rect_wing(math.degrees(fl_ang), semi_span=3.0, chord=1.0, n_panels=4, hinge_x_c=0.75)
        lat_s = build_lattice(ac_s, n_chord=4, chord_spacing="uniform")
        rings_s = build_ring_lattice(lat_s, wake_dir=d_0, te_extension_fraction=0.25)

        x_ref = rp + zn * e_h
        R = _rotation_matrix_321(0.0, th, 0.0)
        omega_g = _angular_velocity_321(th, 0.0, 0.0, thd, 0.0)

        cp_r = x_ref[None, :] + (rings_s.cp - rp[None, :]) @ R.T
        n_r = rings_s.normal_bc @ R.T
        c_r = x_ref[None, None, :] + (rings_s.corners - rp[None, None, :]) @ R.T
        sn_r = x_ref[None, None, None, :] + (rings_s.side_nodes - rp[None, None, None, :]) @ R.T
        te_l_r = x_ref[None, :] + (rings_s.te_left - rp[None, :]) @ R.T
        te_r_r = x_ref[None, :] + (rings_s.te_right - rp[None, :]) @ R.T

        cp_i = ground.reflect_points(cp_r)
        n_i = ground.reflect_directions(n_r)
        c_i = ground.reflect_points(c_r)
        sn_i = ground.reflect_points(sn_r)
        te_l_i = ground.reflect_points(te_l_r)
        te_r_i = ground.reflect_points(te_r_r)

        cp_2 = np.concatenate([cp_r, cp_i], axis=0)
        n_2 = np.concatenate([n_r, n_i], axis=0)
        c_2 = np.concatenate([c_r, c_i], axis=0)
        sn_2 = np.concatenate([sn_r, sn_i], axis=0)
        sc_2 = np.concatenate([rings_s.side_counts, rings_s.side_counts], axis=0)
        te_l_2 = np.concatenate([te_l_r, te_l_i], axis=0)
        te_r_2 = np.concatenate([te_r_r, te_r_i], axis=0)

        if step > 0:
            wake_c_2[: step - 1] += wake_conv_2[None, :, None, :]
            wake_c_2[step - 1, :, 0, :] = te_l_2
            wake_c_2[step - 1, :, 1, :] = te_r_2
            wake_c_2[step - 1, :, 2, :] = prev_te_r_2 + wake_conv_2
            wake_c_2[step - 1, :, 3, :] = prev_te_l_2 + wake_conv_2
            te_idx_2 = np.concatenate([te_idx, te_idx + n_p])
            wake_g_2[step - 1] = G_hist[step - 1, te_idx_2]

        A_2 = VR.ring_normal_influence(cp_2, n_2, c_2, rc_2, side_nodes=sn_2, side_counts=sc_2)

        # Flap velocity in body frame for points aft of hinge (k == 3 on 4-chord uniform mesh).
        on_flap_p = (rings_s.panel_chord_index == 3)
        v_fl_cp_b = np.zeros((n_p, 3), dtype=np.float64)
        v_fl_cp_b[on_flap_p] = np.cross(fl_dot * hinge_ax, rings_s.cp[on_flap_p] - hinge_pt)
        v_surf_r = zd * e_h[None, :] + cross3(omega_g[None, :], cp_r - x_ref[None, :]) + v_fl_cp_b @ R.T
        v_surf_2 = np.concatenate([v_surf_r, ground.reflect_directions(v_surf_r)], axis=0)
        rhs_2 = np.sum((v_surf_2 - V_0[None, :]) * n_2, axis=1)

        if step > 0:
            wc_flat = wake_c_2[:step].reshape(-1, 4, 3)
            wg_flat = wake_g_2[:step].ravel()
            rc_w_flat = np.tile(rc_w_2, step)
            rhs_2 -= VR.ring_normal_influence(cp_2, n_2, wc_flat, rc_w_flat) @ wg_flat

        G_2 = np.linalg.solve(A_2, rhs_2)
        G_hist[step] = G_2
        assert np.max(np.abs(G_2[n_p:] + G_2[:n_p])) <= 1e-12 * max(float(np.max(np.abs(G_2[:n_p]))), 1.0)

        # Loads on the real wing.
        mid_b = np.asarray(lat_s.force_points, dtype=np.float64)
        l_vec_b = np.asarray(lat_s.b - lat_s.a, dtype=np.float64)
        leg_start, leg_end = leg_segments(lat_s)
        leg_mid_b = 0.5 * (leg_start + leg_end)
        leg_vec_b = leg_end - leg_start
        leg_normal_b, leg_scale, _ = leg_force_rules(lat_s)
        n_leg_sets = leg_mid_b.shape[0] // n_p

        P_load_b = np.vstack([mid_b, leg_mid_b])
        seg_vec_b = np.vstack([l_vec_b, leg_vec_b])
        seg_normal_b = np.vstack([np.zeros((n_p, 3)), leg_normal_b])
        seg_scale = np.concatenate([np.ones(n_p), leg_scale])

        # Exact body-frame velocity of load points (zero ahead of hinge, omega_hinge x r aft).
        v_a_b = np.zeros((n_p, 3))
        v_b_b = np.zeros((n_p, 3))
        v_a_b[on_flap_p] = np.cross(fl_dot * hinge_ax, lat_s.a[on_flap_p] - hinge_pt)
        v_b_b[on_flap_p] = np.cross(fl_dot * hinge_ax, lat_s.b[on_flap_p] - hinge_pt)
        v_te_l = np.cross(fl_dot * hinge_ax, lat_s.te_left - hinge_pt)
        v_te_r = np.cross(fl_dot * hinge_ax, lat_s.te_right - hinge_pt)
        v_ak = np.zeros((n_p, 3))
        v_bk = np.zeros((n_p, 3))
        v_ak[on_flap_p] = v_te_l[lat_s.panel_strip[on_flap_p]]
        v_bk[on_flap_p] = v_te_r[lat_s.panel_strip[on_flap_p]]
        f_p = lat_s.cp_frac[lat_s.panel_strip][:, None]
        v_mid_b = (1.0 - f_p) * v_a_b + f_p * v_b_b
        v_ls = np.vstack([v_ak, v_b_b, v_te_l[lat_s.panel_strip], v_bk])
        v_le = np.vstack([v_a_b, v_bk, v_ak, v_te_r[lat_s.panel_strip]])
        v_fl_load_b = np.vstack([v_mid_b, 0.5 * (v_ls + v_le)])

        arm_all = (P_load_b - rp[None, :]) @ R.T
        P_load = x_ref[None, :] + arm_all
        seg_vec = seg_vec_b @ R.T
        seg_normal = seg_normal_b @ R.T
        v_surf_load = zd * e_h[None, :] + cross3(omega_g[None, :], arm_all) + v_fl_load_b @ R.T
        v_ind_load = VR.ring_velocity(P_load, c_2, rc_2, side_nodes=sn_2, side_counts=sc_2).swapaxes(1, 2) @ G_2
        if step > 0:
            v_ind_load += VR.ring_velocity(P_load, wc_flat, rc_w_flat).swapaxes(1, 2) @ wg_flat
        V_tot = V_0[None, :] - v_surf_load + v_ind_load
        V_eff = V_tot - np.sum(V_tot * seg_normal, axis=1, keepdims=True) * seg_normal

        g_hs = _rings_to_horseshoe(G_2[:n_p], n_c)
        g_leg = leg_strength(lat_s, g_hs)
        g_all = np.concatenate([g_hs] + [g_leg] * n_leg_sets)
        F_seg = (cond.rho * g_all * seg_scale)[:, None] * cross3(V_eff, seg_vec)
        F_tot = np.sum(F_seg, axis=0)
        M_tot = np.sum(cross3(arm_all, F_seg), axis=0)

        area_vec_b, ring_c_b, area_mom_b = _physical_ring_areas(lat_s, rings_s, ref_point=rp)
        if step == 1:
            dG_dt = (G_hist[1, :n_p] - G_hist[0, :n_p]) / dt
        elif step >= 2:
            dG_dt = (3.0 * G_hist[step, :n_p] - 4.0 * G_hist[step - 1, :n_p] + G_hist[step - 2, :n_p]) / (2.0 * dt)
        else:
            dG_dt = np.zeros(n_p)
        F_am = (cond.rho * dG_dt)[:, None] * (area_vec_b @ R.T)
        F_tot += np.sum(F_am, axis=0)
        M_tot += np.sum((cond.rho * dG_dt)[:, None] * (area_mom_b @ R.T), axis=0)

        if step >= 1:
            l_te_b = lat_s.te_right - lat_s.te_left
            p_te_b = lat_s.te_left + lat_s.cp_frac[:, None] * l_te_b
            v_fl_te_b = (1.0 - lat_s.cp_frac[:, None]) * v_te_l + lat_s.cp_frac[:, None] * v_te_r
            arm_te = (p_te_b - rp[None, :]) @ R.T
            p_te = x_ref[None, :] + arm_te
            l_te = l_te_b @ R.T
            v_surf_te = zd * e_h[None, :] + cross3(omega_g[None, :], arm_te) + v_fl_te_b @ R.T
            v_ind_te = VR.ring_velocity(p_te, c_2, rc_2, side_nodes=sn_2, side_counts=sc_2).swapaxes(1, 2) @ G_2
            v_ind_te += VR.ring_velocity(p_te, wc_flat, rc_w_flat).swapaxes(1, 2) @ wg_flat
            V_te = V_0[None, :] - v_surf_te + v_ind_te
            dg_te = f_te * (wake_g_2[step - 1, :n_s] - G_2[te_idx])
            F_te = (cond.rho * dg_te)[:, None] * cross3(V_te, l_te)
            F_tot += np.sum(F_te, axis=0)
            M_tot += np.sum(cross3(arm_te, F_te), axis=0)

        CL_2[step] = float((F_tot @ L_0) / qS)
        CD_2[step] = float((F_tot @ d_0) / qS)
        CY_2[step] = float((F_tot @ S_0) / qS)
        Cl_2[step] = float(-M_tot[0] / (qS * ac0.b_ref))
        Cm_2[step] = float(M_tot[1] / (qS * ac0.c_ref))
        Cn_2[step] = float(-M_tot[2] / (qS * ac0.b_ref))
        prev_te_l_2 = te_l_2
        prev_te_r_2 = te_r_2

    scale_g = max(float(np.max(np.abs(res_ge.gamma_bound))), 1.0)
    assert np.max(np.abs(G_hist[-1, :n_p] - res_ge.gamma_bound)) <= 1e-12 * scale_g
    for name, ref_arr in (("CL", CL_2), ("CD", CD_2), ("CY", CY_2), ("Cl", Cl_2), ("Cm", Cm_2), ("Cn", Cn_2)):
        got = getattr(res_ge, name)
        scale = max(float(np.max(np.abs(ref_arr))), 1.0)
        assert np.max(np.abs(got - ref_arr)) <= 1e-12 * scale, f"{name} mismatch"

    # Normal velocity on the ground plane at the last step vanishes to round-off.
    P_gnd = cp_r - ground.height(cp_r)[:, None] * ground.normal[None, :]
    n_gnd = np.broadcast_to(ground.normal, P_gnd.shape)
    v_n_gnd = (
        VR.ring_normal_influence(P_gnd, n_gnd, c_2, rc_2, side_nodes=sn_2, side_counts=sc_2) @ G_hist[-1]
        + VR.ring_normal_influence(P_gnd, n_gnd, wc_flat, rc_w_flat) @ wg_flat
    )
    v_n_real = (
        VR.ring_normal_influence(P_gnd, n_gnd, c_r, lat0.rc, side_nodes=sn_r, side_counts=rings_s.side_counts)
        @ G_hist[-1, :n_p]
        + VR.ring_normal_influence(
            P_gnd, n_gnd, wake_c_2[:n_steps, :n_s].reshape(-1, 4, 3), np.tile(lat0.rc[te_idx], n_steps),
        ) @ wake_g_2[:n_steps, :n_s].ravel()
    )
    assert np.max(np.abs(v_n_gnd)) <= 1e-13 * max(float(np.max(np.abs(v_n_real))), 1.0)


@pytest.mark.parametrize("h_ground", [None, 0.5])
@pytest.mark.parametrize("case_kind", ["flat_flap", "cambered_flap", "aileron"])
def test_backends_numpy_equals_numba_moving_flap(h_ground, case_kind):
    """Numpy and Numba ring kernels give identical loads with moving flaps and 6-axis motion."""
    if not VR_NB.HAVE_NUMBA:
        pytest.skip("Numba is not available")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if case_kind == "cambered_flap":
            ac = _flapped_rect_wing(0.0, n_panels=4, hinge_x_c=0.75, name="flap", shape="4412")
            ctrl_name = "flap"
        elif case_kind == "aileron":
            ac = _flapped_rect_wing(
                0.0, n_panels=6, eta_start=0.5, eta_end=1.0, hinge_x_c=0.75, symmetric=False, name="ail",
            )
            ctrl_name = "ail"
        else:
            ac = _flapped_rect_wing(0.0, n_panels=4, hinge_x_c=0.75, name="flap")
            ctrl_name = "flap"
        lat = build_lattice(ac, n_chord=4, chord_spacing="uniform")

    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=np.radians(3.0), h=h_ground)
    ground = (
        make_ground_plane(lat, h_ground, cond.alpha, cond.beta, ref_point=ac.ref_point, height_ref="ref")
        if h_ground is not None
        else None
    )

    n_steps = 5
    grid = np.linspace(0.0, 1.0, n_steps + 1)
    motion = {
        "x": 0.02 * np.sin(1.2 * grid),
        "y": 0.03 * np.sin(0.9 * grid),
        "z": 0.02 * np.sin(1.5 * grid),
        "roll": np.radians(1.5) * np.sin(1.1 * grid),
        "pitch": np.radians(1.2) * np.sin(1.4 * grid),
        "yaw": np.radians(1.0) * np.sin(0.8 * grid),
    }
    controls = {ctrl_name: np.radians(8.0) * np.sin(1.6 * grid + 0.2)}

    r_np = _solve_time_rings(
        lat, cond, S_ref=ac.S_ref, b_ref=ac.b_ref, c_ref=ac.c_ref, ref_point=ac.ref_point,
        motion=motion, controls=controls, n_steps=n_steps, ground=ground, backend="numpy",
    )
    r_nb = _solve_time_rings(
        lat, cond, S_ref=ac.S_ref, b_ref=ac.b_ref, c_ref=ac.c_ref, ref_point=ac.ref_point,
        motion=motion, controls=controls, n_steps=n_steps, ground=ground, backend="numba",
    )

    for name in ("CL", "CD", "CY", "Cl", "Cm", "Cn", "strip_cl", "gamma_bound", "gamma_wake"):
        a = getattr(r_np, name)
        b = getattr(r_nb, name)
        scale = max(float(np.max(np.abs(a))), 1.0)
        assert np.max(np.abs(a - b)) <= 1e-12 * scale, f"{case_kind} h={h_ground} {name} mismatch"


def test_input_checks_six_axis_and_controls():
    """All new motion and control input checks raise InputError with clear messages."""
    ac = _flapped_rect_wing(0.0, semi_span=3.0, chord=1.0, n_panels=4, hinge_x_c=0.75, name="flap")
    lat = build_lattice(ac, n_chord=4, chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=0.0)
    common = {"S_ref": ac.S_ref, "b_ref": ac.b_ref, "c_ref": ac.c_ref, "ref_point": ac.ref_point}

    # Axis set twice: z and motion['z']; theta and motion['pitch'].
    with pytest.raises(InputError, match="z"):
        _solve_time_rings(lat, cond, z=np.zeros(3), motion={"z": np.zeros(3)}, n_steps=2, **common)
    with pytest.raises(InputError, match="pitch"):
        _solve_time_rings(lat, cond, theta=np.zeros(3), motion={"pitch": np.zeros(3)}, n_steps=2, **common)

    # Unknown motion axis or non-mapping motion.
    with pytest.raises(InputError, match="heave"):
        _solve_time_rings(lat, cond, motion={"heave": np.zeros(3)}, n_steps=2, **common)
    with pytest.raises(InputError, match="motion"):
        _solve_time_rings(lat, cond, motion=[0.0, 0.0, 0.0], n_steps=2, **common)  # type: ignore[arg-type]

    # Wrong array length and non-finite values in motion.
    for axis in ("x", "y", "z", "roll", "pitch", "yaw"):
        with pytest.raises(InputError, match=axis):
            _solve_time_rings(lat, cond, motion={axis: np.zeros(2)}, n_steps=2, **common)
        with pytest.raises(InputError, match=axis):
            _solve_time_rings(lat, cond, motion={axis: np.array([0.0, np.nan, 0.0])}, n_steps=2, **common)
        with pytest.raises(InputError, match=axis):
            _solve_time_rings(lat, cond, motion={axis: lambda t: np.inf}, n_steps=2, **common)

    # Unknown control name or non-mapping controls.
    with pytest.raises(InputError, match="rudder"):
        _solve_time_rings(lat, cond, controls={"rudder": np.zeros(3)}, n_steps=2, **common)
    with pytest.raises(InputError, match="controls"):
        _solve_time_rings(lat, cond, controls=[0.0, 0.0, 0.0], n_steps=2, **common)  # type: ignore[arg-type]

    # Wrong array length and non-finite values in controls.
    with pytest.raises(InputError, match="flap"):
        _solve_time_rings(lat, cond, controls={"flap": np.zeros(4)}, n_steps=2, **common)
    with pytest.raises(InputError, match="flap"):
        _solve_time_rings(lat, cond, controls={"flap": np.array([0.0, np.nan, 0.0])}, n_steps=2, **common)
    with pytest.raises(InputError, match="flap"):
        _solve_time_rings(lat, cond, controls={"flap": lambda t: np.nan}, n_steps=2, **common)

    # Moving control on a section-mode (n_chord = 1) lattice is refused.
    lat_1 = build_lattice(ac, n_chord=1)
    with pytest.raises(InputError, match="chordwise"):
        _solve_time_rings(lat_1, cond, controls={"flap": np.zeros(3)}, n_steps=2, **common)

    # Deflected flap striking the ground raises InputError naming step and time.
    # Wing clears ground at h = 0.08 m when undeflected; 0.25 m flap at 25 deg drops 0.106 m.
    ground_low = make_ground_plane(lat, 0.08, 0.0, 0.0, ref_point=ac.ref_point, height_ref="ref")
    flap_strike = np.radians(np.array([0.0, 5.0, 25.0, 10.0]))
    with pytest.raises(InputError, match=r"step 2") as exc_fl:
        _solve_time_rings(
            lat, cond, controls={"flap": flap_strike}, n_steps=3, ground=ground_low, **common,
        )
    assert "t =" in str(exc_fl.value)

