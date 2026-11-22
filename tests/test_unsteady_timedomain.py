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
