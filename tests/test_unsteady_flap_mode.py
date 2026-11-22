# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for the harmonic control-surface mode of the vortex-ring solver.

Covers unchanged baseline results when no control is requested, exact k -> 0
parity against ``ventorum.control_derivatives`` in free air and ground effect,
Woodbury against direct complex LU parity, finite-difference step independence,
explicit antiphase two-wing mirror parity in ground effect, antisymmetric
aileron symmetry, periodic-limit agreement against ``_solve_time_rings``,
second-order time-step convergence at ``k = 0.5``, comparison with the
documented time-domain moving-flap first harmonics, and input refusals.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import ventorum as vt
from ventorum.aero import vortex_ring as VR
from ventorum.aero.system import make_ground_plane
from ventorum.core.errors import InputError
from ventorum.geometry import lattice as lattice_mod
from ventorum.geometry.lattice import build_lattice
from ventorum.unsteady.harmonic import (
    _HarmonicRingSystem,
    _solve_harmonic_rings,
)
from ventorum.unsteady.rings import ground_image_rings
from ventorum.unsteady.tail import harmonic_tail_normal_influence
from ventorum.unsteady.timedomain import _solve_time_rings


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


def _fit_first_harmonic(t: np.ndarray, y: np.ndarray, omega: float, amp_rad: float) -> complex:
    """Fit ``c0 + A * Re(H * (-i) * exp(i * omega * t))`` to *y* by least squares."""
    basis = np.column_stack([np.ones_like(t), np.sin(omega * t), np.cos(omega * t)])
    coef, _, _, _ = np.linalg.lstsq(basis, y, rcond=None)
    return complex(coef[1] / amp_rad, coef[2] / amp_rad)


def _periodic_flap_run_and_compare(
    lat: vt.VortexLattice,
    ac: vt.Aircraft,
    cond: vt.FlightCondition,
    ground,
    k: float,
    control_name: str = "flap",
    *,
    n_periods_transient: float,
    n_periods_fit: int = 1,
    amp_rad: float = 1.0e-4,
    dt: float | None = None,
) -> tuple[complex, complex, complex, complex]:
    """Run harmonic flap deflection in time and frequency domains and return ``(CL_td, CL_h, Cm_td, Cm_h)``.

    When ``cond.alpha != 0``, the zero-motion impulsive-start baseline over the
    same time steps is subtracted before fitting so the ``O(1 / L_w^2)`` steady
    starting-vortex drift does not contaminate a ``1e-4`` rad oscillation.
    """
    omega = 2.0 * k * cond.V_inf / ac.c_ref
    period = 2.0 * math.pi / omega
    te_idx = np.flatnonzero(np.tile(np.arange(lat.n_chord), lat.n_strips) == (lat.n_chord - 1))
    dx_te_mean = float(np.sum(lat.width * lat.panel_length[te_idx]) / np.sum(lat.width))
    dt_use = dx_te_mean / cond.V_inf if dt is None else float(dt)

    total_time = (n_periods_transient + n_periods_fit) * period
    n_steps = int(math.ceil(total_time / dt_use))

    res_td = _solve_time_rings(
        lat,
        cond,
        S_ref=ac.S_ref,
        b_ref=ac.b_ref,
        c_ref=ac.c_ref,
        ref_point=ac.ref_point,
        controls={control_name: lambda t: amp_rad * math.sin(omega * t)},
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
        controls=(control_name,),
    )

    if abs(cond.alpha) > 0.0:
        res_base = _solve_time_rings(
            lat,
            cond,
            S_ref=ac.S_ref,
            b_ref=ac.b_ref,
            c_ref=ac.c_ref,
            ref_point=ac.ref_point,
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
    cl_td = _fit_first_harmonic(t_win, cl_sig[mask], omega, amp_rad)
    cm_td = _fit_first_harmonic(t_win, cm_sig[mask], omega, amp_rad)

    h_ctrl = res_h.controls[control_name]
    cl_h = complex(h_ctrl.CL[0])
    cm_h = complex(h_ctrl.Cm[0])
    return cl_td, cl_h, cm_td, cm_h


# ---------------------------------------------------------------------------
# 1. Baseline unchanged when no control is requested
# ---------------------------------------------------------------------------


def test_no_control_requested_keeps_existing_results_unchanged():
    """With controls=(), every existing response field stays identical to round-off."""
    ac = _flapped_rect_wing(5.0, n_panels=4, hinge_x_c=0.75)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lat = build_lattice(ac, n_chord=4, chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=25.0, rho=1.225, alpha=np.radians(4.0), h=0.5)
    ground = make_ground_plane(lat, 0.5, cond.alpha, cond.beta, ref_point=ac.ref_point, height_ref="ref")

    r_default = _solve_harmonic_rings(
        lat, cond, [0.0, 0.1, 0.3],
        S_ref=ac.S_ref, b_ref=ac.b_ref, c_ref=ac.c_ref, ground=ground, ref_point=ac.ref_point,
    )
    r_empty = _solve_harmonic_rings(
        lat, cond, [0.0, 0.1, 0.3],
        S_ref=ac.S_ref, b_ref=ac.b_ref, c_ref=ac.c_ref, ground=ground, ref_point=ac.ref_point,
        controls=(),
    )
    r_with = _solve_harmonic_rings(
        lat, cond, [0.0, 0.1, 0.3],
        S_ref=ac.S_ref, b_ref=ac.b_ref, c_ref=ac.c_ref, ground=ground, ref_point=ac.ref_point,
        controls=("flap",),
    )

    assert r_default.controls == {}
    assert r_default.k0_controls == {}
    assert r_empty.controls == {}
    assert r_empty.k0_controls == {}
    assert "flap" in r_with.controls
    assert "flap" in r_with.k0_controls

    for other in (r_empty, r_with):
        for motion in ("heave", "pitch"):
            ma = getattr(r_default, motion)
            mb = getattr(other, motion)
            for name in ("CL", "CD", "CY", "Cl", "Cm", "Cn"):
                a = getattr(ma, name)
                b = getattr(mb, name)
                assert np.max(np.abs(a - b)) <= 1e-13 * max(float(np.max(np.abs(a))), 1.0)
        for attr in ("k0_heave", "k0_pitch", "k0_pitch_rate", "mean_state"):
            sa = getattr(r_default, attr)
            sb = getattr(other, attr)
            for name in ("CL", "CD", "CY", "Cl", "Cm", "Cn"):
                assert abs(getattr(sa, name) - getattr(sb, name)) <= 1e-13 * max(abs(getattr(sa, name)), 1.0)


# ---------------------------------------------------------------------------
# 2. k -> 0 limit against ventorum.control_derivatives
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("h_ground", [None, 0.5])
@pytest.mark.parametrize("alpha_deg", [0.0, 4.0])
def test_k0_flap_limit_matches_control_derivatives(monkeypatch, h_ground, alpha_deg):
    """Exact k -> 0 flap derivatives match ventorum.control_derivatives on the same lattice.

    Tolerance derivation:
    * ``control_derivatives`` uses central differences with ``step_deg = 0.01``
      (``h = 1.745e-4`` rad), whose truncation error is ``O(h^2) ~ 3e-8`` on
      order-one derivatives.
    * The harmonic flap mode uses ``fd_step_delta = 1e-5`` rad for the loaded
      geometric influence terms, whose truncation error is ``O(1e-10)`` and
      round-off is ``~1e-11``.
    * Thus ``CL, CY, Cl, Cm, Cn`` (and near-field ``CD`` from ``analyze``) agree
      within ``1e-7 * max(|ref|, 1.0)``.
    * ``control_derivatives`` reports Trefftz-plane ``CDi`` for ``CD``, while
      the ring solver reports near-field Kutta-Joukowski ``CD``; at ``alpha = 4``
      deg on this 6 x 4 lattice the two induced-drag formulations differ by
      ``~1.3e-5``, bounded by ``5e-5``.
    """
    ac = _flapped_rect_wing(0.0, semi_span=3.0, chord=1.0, n_panels=6, hinge_x_c=0.75)
    st = vt.SolverSettings(
        solver_type="vlm",
        n_panels=6,
        n_chord=4,
        spacing="half-cosine",
        chord_spacing="uniform",
    )
    cond = vt.FlightCondition(V_inf=25.0, rho=1.225, alpha=np.radians(alpha_deg), h=h_ground)

    monkeypatch.setattr(lattice_mod, "CORE_RADIUS_FRACTION", 1.0e-4)
    lat = build_lattice(ac, st, collocation="vlm", n_chord=4, chord_spacing="uniform")
    ground = (
        make_ground_plane(lat, h_ground, cond.alpha, cond.beta, ref_point=ac.ref_point, height_ref="ref")
        if h_ground is not None
        else None
    )

    step_deg = 0.01
    step_rad = math.radians(step_deg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cd_res = vt.control_derivatives(
            ac, cond, st, controls=["flap"], step_deg=step_deg, axes="body", hinge_alpha=False,
        )
        ac_p = ac.clone()
        ac_m = ac.clone()
        ac_p.surfaces[0].controls[0].deflection = +step_rad
        ac_m.surfaces[0].controls[0].deflection = -step_rad
        rp = vt.analyze(ac_p, cond, st).totals
        rm = vt.analyze(ac_m, cond, st).totals
    cd_ref = cd_res.derivatives["flap"]
    cd_nf_ref = (rp.CDi_nearfield - rm.CDi_nearfield) / (2.0 * step_rad)

    resp = _solve_harmonic_rings(
        lat,
        cond,
        [0.0, 0.05],
        S_ref=ac.S_ref,
        b_ref=ac.b_ref,
        c_ref=ac.c_ref,
        ground=ground,
        ref_point=ac.ref_point,
        te_extension_fraction=0.0,
        controls=("flap",),
    )

    k0 = resp.k0_controls["flap"]
    h_ctrl = resp.controls["flap"]
    for name in ("CL", "CY", "Cl", "Cm", "Cn"):
        ref_val = float(getattr(cd_ref, name))
        got_val = float(getattr(k0, name))
        scale = max(abs(ref_val), 1.0)
        assert abs(got_val - ref_val) <= 1.0e-7 * scale, f"{name}: got={got_val}, ref={ref_val}"
        assert abs(getattr(h_ctrl, name)[0] - got_val) <= 1.0e-12 * scale

    assert abs(k0.CD - cd_nf_ref) <= 1.0e-7 * max(abs(cd_nf_ref), 1.0)
    assert abs(k0.CD - cd_ref.CD) <= 5.0e-5 * max(abs(cd_ref.CD), 1.0)
    assert abs(h_ctrl.CD[0] - k0.CD) <= 1.0e-12 * max(abs(k0.CD), 1.0)


# ---------------------------------------------------------------------------
# 3. Woodbury vs direct, loaded FD step independence, cambered & nonzero delta0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [None, "4412"])
@pytest.mark.parametrize("delta0_deg", [0.0, 10.0])
def test_flap_woodbury_equals_direct_and_fd_step_independence(shape, delta0_deg):
    """Flap mode Woodbury equals direct LU to round-off and is independent of fd_step_delta."""
    ac = _flapped_rect_wing(delta0_deg, semi_span=3.0, chord=1.0, n_panels=4, hinge_x_c=0.75, shape=shape)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lat = build_lattice(ac, n_chord=4, chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=25.0, rho=1.225, alpha=np.radians(4.0), h=0.5)
    ground = make_ground_plane(lat, 0.5, cond.alpha, cond.beta, ref_point=ac.ref_point, height_ref="ref")

    k_vals = [0.0, 0.05, 0.2]
    r_wb = _solve_harmonic_rings(
        lat, cond, k_vals, S_ref=ac.S_ref, b_ref=ac.b_ref, c_ref=ac.c_ref,
        ground=ground, ref_point=ac.ref_point, solve_method="woodbury", controls=("flap",),
    )
    r_dir = _solve_harmonic_rings(
        lat, cond, k_vals, S_ref=ac.S_ref, b_ref=ac.b_ref, c_ref=ac.c_ref,
        ground=ground, ref_point=ac.ref_point, solve_method="direct", controls=("flap",),
    )
    for name in ("CL", "CD", "CY", "Cl", "Cm", "Cn"):
        a = getattr(r_wb.controls["flap"], name)
        b = getattr(r_dir.controls["flap"], name)
        assert np.all(np.isfinite(a))
        assert np.max(np.abs(a - b)) <= 1e-12 * max(float(np.max(np.abs(b))), 1.0)

    if shape is None and delta0_deg == 0.0:
        res_steps = [
            _solve_harmonic_rings(
                lat, cond, [0.2], S_ref=ac.S_ref, b_ref=ac.b_ref, c_ref=ac.c_ref,
                ground=ground, ref_point=ac.ref_point, fd_step_delta=step, controls=("flap",),
            )
            for step in (1e-6, 1e-5, 1e-4)
        ]
        ref = res_steps[1].controls["flap"]
        for alt_r in (res_steps[0].controls["flap"], res_steps[2].controls["flap"]):
            for name in ("CL", "CD", "Cm"):
                scale = max(abs(getattr(ref, name)[0]), 1.0)
                assert abs(getattr(alt_r, name)[0] - getattr(ref, name)[0]) <= 1e-7 * scale


# ---------------------------------------------------------------------------
# 4. Ground mirror parity against explicit antiphase two-wing free-air model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_tail", [False, True])
def test_flap_ground_mirror_parity(use_tail):
    """Ground-image flap mode matches an explicit antiphase two-wing free-air model to round-off."""
    ac = _flapped_rect_wing(5.0, semi_span=3.0, chord=1.0, n_panels=4, hinge_x_c=0.75)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lat = build_lattice(ac, n_chord=4, chord_spacing="uniform")
    h_ground = 0.5
    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=np.radians(4.0), h=h_ground)
    ground = make_ground_plane(lat, h_ground, cond.alpha, cond.beta, ref_point=ac.ref_point, height_ref="ref")
    k = 0.2

    sys_ge = _HarmonicRingSystem(
        lat,
        cond,
        S_ref=ac.S_ref,
        b_ref=ac.b_ref,
        c_ref=ac.c_ref,
        ground=ground,
        ref_point=ac.ref_point,
        wake_chords=5.0,
        use_tail=use_tail,
        controls=("flap",),
    )
    rhs_ge, G_ge = sys_ge.solve_circulations(k, solve_method="woodbury")
    assert rhs_ge.shape == (sys_ge.rings.n_panels, 3)

    rings = sys_ge.rings
    img = ground_image_rings(rings, ground, wake_dir=rings.wake_dir)
    n = rings.n_panels
    n_s = rings.n_strips
    te_idx = rings.te_indices
    dx_w = sys_ge.dx_w
    dt = sys_ge.dt
    n_rows = sys_ge.n_wake_rows
    omega = 2.0 * k * cond.V_inf / ac.c_ref
    kappa = omega / cond.V_inf

    cp_2 = np.concatenate([rings.cp, ground.reflect_points(rings.cp)], axis=0)
    n_2 = np.concatenate([rings.normal_bc, ground.reflect_directions(rings.normal_bc)], axis=0)
    c_2 = np.concatenate([rings.corners, img.corners], axis=0)
    sn_2 = None if rings.side_nodes is None else np.concatenate([rings.side_nodes, img.side_nodes], axis=0)
    sc_2 = None if rings.side_counts is None else np.concatenate([rings.side_counts, img.side_counts], axis=0)
    rc_2 = np.concatenate([rings.rc, rings.rc], axis=0)
    rc_te_2 = np.concatenate([rings.rc[te_idx], rings.rc[te_idx]], axis=0)

    A_2 = VR.ring_normal_influence(cp_2, n_2, c_2, rc_2, side_nodes=sn_2, side_counts=sc_2).astype(np.complex128)

    te_l_2 = np.concatenate([rings.te_left, img.te_left], axis=0)
    te_r_2 = np.concatenate([rings.te_right, img.te_right], axis=0)
    wd_2 = np.concatenate([rings.wake_dir, img.wake_dir], axis=0)
    te_idx_2 = np.concatenate([te_idx, te_idx + n])

    lags = np.exp(-1j * omega * np.arange(1, n_rows + 1, dtype=float) * dt)
    W_2 = np.zeros((2 * n, 2 * n_s), dtype=np.complex128)
    for m in range(n_rows):
        p0_l = te_l_2 + (m * dx_w) * wd_2
        p0_r = te_r_2 + (m * dx_w) * wd_2
        p1_l = p0_l + dx_w * wd_2
        p1_r = p0_r + dx_w * wd_2
        w_corners = np.stack([p0_l, p0_r, p1_r, p1_l], axis=1)
        W_2 += lags[m] * VR.ring_normal_influence(cp_2, n_2, w_corners, rc_te_2)

    if use_tail:
        phase0 = np.exp(-1j * omega * (n_rows + 0.5) * dt)
        tail_l = te_l_2 + (n_rows * dx_w) * wd_2
        tail_r = te_r_2 + (n_rows * dx_w) * wd_2
        W_2 += phase0 * harmonic_tail_normal_influence(cp_2, n_2, tail_l, tail_r, wd_2, kappa, rc_te_2)

    A_2[:, te_idx_2] += W_2

    # Independently check the two-wing perturbed steady geometry derivative of the flap mode at k = 0.
    from ventorum.unsteady.timedomain import _deform_body_geometry

    fm = sys_ge._flap_modes["flap"]
    eps = 1.0e-5
    G0_2 = np.concatenate([sys_ge.Gamma_0, -sys_ge.Gamma_0])
    g0_te_2 = G0_2[te_idx_2]

    def _two_wing_steady_mv(sgn: float) -> np.ndarray:
        defm = _deform_body_geometry(fm.rig, lat, rings, sgn * eps * fm.unit_edges)
        cp_r, n_r, c_r, tl_r, tr_r, sn_r = defm[2], defm[3], defm[4], defm[5], defm[6], defm[9]
        cp_d2 = np.concatenate([cp_r, ground.reflect_points(cp_r)], axis=0)
        n_d2 = np.concatenate([n_r, ground.reflect_directions(n_r)], axis=0)
        c_d2 = np.concatenate([c_r, ground.reflect_points(c_r)], axis=0)
        sn_d2 = None if sn_r is None else np.concatenate([sn_r, ground.reflect_points(sn_r)], axis=0)
        tl_d2 = np.concatenate([tl_r, ground.reflect_points(tl_r)], axis=0)
        tr_d2 = np.concatenate([tr_r, ground.reflect_points(tr_r)], axis=0)
        return (
            VR.ring_normal_influence(cp_d2, n_d2, c_d2, rc_2, side_nodes=sn_d2, side_counts=sc_2) @ G0_2
            + VR.steady_wake_normal_influence(cp_d2, n_d2, tl_d2, tr_d2, wd_2, rc_te_2) @ g0_te_2
        )

    mv_p = _two_wing_steady_mv(+1.0)
    mv_m = _two_wing_steady_mv(-1.0)
    vp_ge = sys_ge._flap_steady_state_eval(fm, +eps * fm.unit_edges)[0]
    vm_ge = sys_ge._flap_steady_state_eval(fm, -eps * fm.unit_edges)[0]
    scale_mv = max(float(np.max(np.abs(vp_ge))), 1.0)
    assert np.max(np.abs(mv_p[:n] - vp_ge)) <= 1e-13 * scale_mv
    assert np.max(np.abs(mv_p[n:] - vp_ge)) <= 1e-13 * scale_mv
    assert np.max(np.abs(mv_m[:n] - vm_ge)) <= 1e-13 * scale_mv
    assert np.max(np.abs(mv_m[n:] - vm_ge)) <= 1e-13 * scale_mv

    rhs_geo_2 = -(mv_p - mv_m) / (2.0 * eps)
    scale_geo = max(float(np.max(np.abs(fm.rhs_geo))), 1.0)
    assert np.max(np.abs(rhs_geo_2[:n] - fm.rhs_geo)) <= 1e-9 * scale_geo
    assert np.max(np.abs(rhs_geo_2[n:] - fm.rhs_geo)) <= 1e-9 * scale_geo

    # Antiphase two-wing harmonic solve matches the ground-image flap mode to round-off.
    rhs_2 = np.concatenate([rhs_ge, rhs_ge], axis=0)
    G_2 = np.linalg.solve(A_2, rhs_2)
    scale_fl = max(float(np.max(np.abs(G_ge[:, 2]))), 1.0)
    assert np.max(np.abs(G_2[:n, 2] - G_ge[:, 2])) <= 1e-12 * scale_fl
    assert np.max(np.abs(G_2[n:, 2] + G_ge[:, 2])) <= 1e-12 * scale_fl


# ---------------------------------------------------------------------------
# 5. Antisymmetric aileron mode symmetry on a symmetric wing in free air
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alpha_deg", [0.0, 4.0])
def test_antisymmetric_aileron_mode_symmetry(alpha_deg):
    """Antisymmetric aileron on a symmetric wing gives CL = Cm = CD = 0 to round-off and Cl != 0."""
    ac = _flapped_rect_wing(
        0.0, semi_span=3.0, chord=1.0, n_panels=6,
        eta_start=0.5, eta_end=1.0, hinge_x_c=0.75, symmetric=False, name="aileron",
    )
    lat = build_lattice(ac, n_chord=4, chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=25.0, rho=1.225, alpha=np.radians(alpha_deg))

    resp = _solve_harmonic_rings(
        lat,
        cond,
        [0.0, 0.05, 0.2, 0.5],
        S_ref=ac.S_ref,
        b_ref=ac.b_ref,
        c_ref=ac.c_ref,
        ref_point=ac.ref_point,
        controls=("aileron",),
    )

    m_ail = resp.controls["aileron"]
    k0_ail = resp.k0_controls["aileron"]
    for name in ("CL", "Cm", "CD"):
        assert np.max(np.abs(getattr(m_ail, name))) <= 1e-12, f"{name} not zero: {getattr(m_ail, name)}"
        assert abs(getattr(k0_ail, name)) <= 1e-12, f"k0 {name} not zero: {getattr(k0_ail, name)}"
    assert np.min(np.abs(m_ail.Cl)) > 0.05
    assert abs(k0_ail.Cl) > 0.05


# ---------------------------------------------------------------------------
# 6. Periodic limit against _solve_time_rings (fast default + full slow sweep)
# ---------------------------------------------------------------------------


def test_flap_periodic_limit_k02_free_air_fast():
    """Fast default check: k = 0.2 free-air harmonic flap matches time-domain within 0.5 %."""
    ac = _flapped_rect_wing(0.0, semi_span=3.0, chord=1.0, n_panels=4, hinge_x_c=0.75)
    lat = build_lattice(ac, n_chord=4, chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=0.0)

    cl_td, cl_h, cm_td, cm_h = _periodic_flap_run_and_compare(
        lat, ac, cond, None, 0.2, "flap", n_periods_transient=2.0, n_periods_fit=1,
    )

    err_cl = abs(cl_td - cl_h) / abs(cl_h)
    err_cm = abs(cm_td - cm_h) / abs(cm_h)
    assert err_cl <= 5.0e-3, f"CL err={err_cl:.4%} (td={cl_td}, h={cl_h})"
    assert err_cm <= 5.0e-3, f"Cm err={err_cm:.4%} (td={cm_td}, h={cm_h})"


@pytest.mark.slow
@pytest.mark.parametrize("h_ground", [None, 0.5])
@pytest.mark.parametrize("alpha_deg", [0.0, 4.0])
@pytest.mark.parametrize("k", [0.05, 0.2, 0.5])
def test_flap_periodic_limit_matches_timedomain(h_ground, alpha_deg, k):
    """Harmonic 1e-4 rad flap matches _solve_time_rings on the same lattice (free air and h/c = 0.5).

    Targets match the existing heave and pitch periodic-limit tests in
    tests/test_unsteady_timedomain.py: <= 0.1 % at k <= 0.2 and <= 1 % at k = 0.5.
    """
    ac = _flapped_rect_wing(0.0, semi_span=3.0, chord=1.0, n_panels=4, hinge_x_c=0.75)
    if k == 0.05:
        n_chord, n_trans = 4, 1.0
    elif k == 0.2:
        n_chord, n_trans = 16, 2.0
    else:
        n_chord, n_trans = 12, 4.0

    lat = build_lattice(ac, n_chord=n_chord, chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=np.radians(alpha_deg), h=h_ground)
    ground = (
        make_ground_plane(lat, h_ground, cond.alpha, cond.beta, ref_point=ac.ref_point, height_ref="ref")
        if h_ground is not None
        else None
    )

    cl_td, cl_h, cm_td, cm_h = _periodic_flap_run_and_compare(
        lat, ac, cond, ground, k, "flap", n_periods_transient=n_trans, n_periods_fit=1,
    )

    target = 1.0e-3 if k <= 0.2 else 1.0e-2
    err_cl = abs(cl_td - cl_h) / abs(cl_h)
    err_cm = abs(cm_td - cm_h) / abs(cm_h)
    assert err_cl <= target, (
        f"alpha0={alpha_deg} deg, h={h_ground}, k={k}: CL err={err_cl:.4%} (td={cl_td}, h={cl_h})"
    )
    assert err_cm <= target, (
        f"alpha0={alpha_deg} deg, h={h_ground}, k={k}: Cm err={err_cm:.4%} (td={cm_td}, h={cm_h})"
    )


# ---------------------------------------------------------------------------
# 7. Second-order time-step convergence at k = 0.5 and documented first harmonics
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_flap_timestep_convergence_k05_second_order():
    """Time-domain minus harmonic flap first harmonic at k = 0.5 falls at order 1.7 to 2.3.

    Doubling n_chord from 4 to 8 to 16 halves dt = dx_TE / V_inf while keeping
    hinge_x_c = 0.75 on an exact chordwise node and dx_w = dx_TE identical
    between both solvers at each level, isolating the O(dt^2) error of the
    second-order backward difference for d(Gamma)/dt.
    """
    ac = _flapped_rect_wing(0.0, semi_span=3.0, chord=1.0, n_panels=4, hinge_x_c=0.75)
    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=np.radians(4.0))
    k = 0.5

    errs_cl = []
    errs_cm = []
    for n_c in (4, 8, 16):
        lat = build_lattice(ac, n_chord=n_c, chord_spacing="uniform")
        cl_td, cl_h, cm_td, cm_h = _periodic_flap_run_and_compare(
            lat, ac, cond, None, k, "flap", n_periods_transient=4.0, n_periods_fit=1,
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


@pytest.mark.slow
def test_moving_flap_documented_first_harmonics_match_harmonic_mode():
    """Reproduce the documented time-domain moving-flap first harmonics and compare with harmonic mode.

    Reference numbers from test_moving_flap_small_amplitude_dt_convergence in
    tests/test_unsteady_timedomain.py on n_chord = 4, 8, 16 at alpha0 = 0 deg.
    """
    documented = {
        0.05: {
            "n_trans": 1.0,
            "CL": [2.349665 - 0.076997j, 2.470463 - 0.079419j, 2.536497 - 0.080922j],
            "Cm": [-0.594247 - 0.021306j, -0.594784 - 0.023744j, -0.594600 - 0.024859j],
            "tol_h": [1.0e-3, 1.0e-3, 1.0e-3],
        },
        0.2: {
            "n_trans": 2.0,
            "CL": [2.140701 - 0.097039j, 2.252147 - 0.096328j, 2.312440 - 0.096306j],
            "Cm": [-0.593230 - 0.083426j, -0.594590 - 0.092851j, -0.594642 - 0.097195j],
            "tol_h": [5.0e-3, 2.0e-3, 1.0e-3],
        },
    }

    ac = _flapped_rect_wing(0.0, semi_span=3.0, chord=1.0, n_panels=4, hinge_x_c=0.75)
    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=0.0)

    for k, spec in documented.items():
        for idx, n_c in enumerate((4, 8, 16)):
            lat = build_lattice(ac, n_chord=n_c, chord_spacing="uniform")
            cl_td, cl_h, cm_td, cm_h = _periodic_flap_run_and_compare(
                lat, ac, cond, None, k, "flap", n_periods_transient=spec["n_trans"], n_periods_fit=1,
            )
            assert abs(cl_td - spec["CL"][idx]) <= 1.0e-5, (k, n_c, cl_td, spec["CL"][idx])
            assert abs(cm_td - spec["Cm"][idx]) <= 1.0e-5, (k, n_c, cm_td, spec["Cm"][idx])
            assert abs(cl_td - cl_h) / abs(cl_h) <= spec["tol_h"][idx]
            assert abs(cm_td - cm_h) / abs(cm_h) <= spec["tol_h"][idx]


# ---------------------------------------------------------------------------
# 8. Input checks for controls and fd_step_delta
# ---------------------------------------------------------------------------


def test_flap_mode_input_refusals():
    """Unknown controls, duplicate names, strings, section mode and bad fd_step_delta raise InputError."""
    ac = _flapped_rect_wing(0.0, semi_span=3.0, chord=1.0, n_panels=4, hinge_x_c=0.75, name="flap")
    lat = build_lattice(ac, n_chord=4, chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=0.0)
    common = {"S_ref": ac.S_ref, "b_ref": ac.b_ref, "c_ref": ac.c_ref, "ref_point": ac.ref_point}

    with pytest.raises(InputError, match="rudder"):
        _solve_harmonic_rings(lat, cond, [0.1], controls=("rudder",), **common)

    with pytest.raises(InputError, match="controls"):
        _solve_harmonic_rings(lat, cond, [0.1], controls="flap", **common)  # type: ignore[arg-type]

    with pytest.raises(InputError, match="unique|duplicate"):
        _solve_harmonic_rings(lat, cond, [0.1], controls=("flap", "flap"), **common)

    lat_1 = build_lattice(ac, n_chord=1)
    with pytest.raises(InputError, match="chordwise"):
        _solve_harmonic_rings(lat_1, cond, [0.1], controls=("flap",), **common)

    for bad_step in (0.0, -1e-5, np.nan, np.inf):
        with pytest.raises(InputError, match="fd_step_delta"):
            _solve_harmonic_rings(lat, cond, [0.1], controls=("flap",), fd_step_delta=bad_step, **common)
