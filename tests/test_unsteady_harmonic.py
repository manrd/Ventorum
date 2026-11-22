# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for the harmonic vortex-ring lattice solver and far-wake tail.

Covers Woodbury against direct complex solve parity, mirror-wing parity in
ground effect, k -> 0 limits against steady VLM solvers (ventorum.analyze,
height_derivatives and rate_derivatives), loaded-mean-state step independence,
trailing-edge correction (zero steady effect and second-order chordwise
convergence of Re(Cm_alpha_dot)), 3D far-wake tail accuracy, high-aspect-ratio
Theodorsen limit, and input refusals.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.special import hankel2

import ventorum as vt
from ventorum.aero import vortex_ring as VR
from ventorum.aero.system import freestream_direction, make_ground_plane
from ventorum.core.errors import GroundStrikeError, InputError
from ventorum.geometry import lattice as lattice_mod
from ventorum.geometry.lattice import build_lattice
from ventorum.unsteady.harmonic import (
    _HarmonicResponse,
    _HarmonicRingSystem,
    _solve_harmonic_rings,
)
from ventorum.unsteady.rings import build_ring_lattice, ground_image_rings
from ventorum.unsteady.tail import harmonic_tail_normal_influence, harmonic_tail_velocity


def _rect_wing(
    semi_span: float = 3.0,
    chord: float = 1.0,
    n_panels: int = 12,
    spacing: str = "half-cosine",
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


def _swept_tapered_wing(n_panels: int = 6) -> vt.Aircraft:
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


def _theodorsen_c(k: float) -> complex:
    """Return Theodorsen's circulation function C(k) for reduced frequency *k* [-].

    Uses Hankel functions of the second kind (Theodorsen 1935, NACA Report 496).
    """
    h1 = hankel2(1, k)
    h0 = hankel2(0, k)
    return complex(h1 / (h1 + 1j * h0))


def _steady_vlm_alpha_slopes(
    ac: vt.Aircraft,
    cond: vt.FlightCondition,
    st: vt.SolverSettings,
    d_alpha: float = 1e-4,
) -> tuple[float, float]:
    """Return central-difference (CL_alpha, Cm_alpha) [1/rad] from ventorum.analyze."""
    c_p = vt.FlightCondition(
        V_inf=cond.V_inf,
        rho=cond.rho,
        alpha=cond.alpha + d_alpha,
        beta=cond.beta,
        phi=cond.phi,
        h=cond.h,
    )
    c_m = vt.FlightCondition(
        V_inf=cond.V_inf,
        rho=cond.rho,
        alpha=cond.alpha - d_alpha,
        beta=cond.beta,
        phi=cond.phi,
        h=cond.h,
    )
    rp = vt.analyze(ac, c_p, st)
    rm = vt.analyze(ac, c_m, st)
    cl_a = (rp.totals.CL - rm.totals.CL) / (2.0 * d_alpha)
    cm_a = (rp.totals.Cm - rm.totals.Cm) / (2.0 * d_alpha)
    return float(cl_a), float(cm_a)


# ---------------------------------------------------------------------------
# 1. Woodbury solve equals direct complex solve to round-off
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("wing_kind", ["rect", "swept"])
@pytest.mark.parametrize("h_ground", [None, 0.5])
def test_woodbury_equals_direct_complex_solve(wing_kind, h_ground):
    """Woodbury rank-n_strips solve matches the direct complex LU solve to round-off."""
    ac = _rect_wing(n_panels=6) if wing_kind == "rect" else _swept_tapered_wing(n_panels=6)
    lat = build_lattice(ac, n_chord=4, chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=25.0, rho=1.225, alpha=np.radians(4.0), h=h_ground)
    ground = (
        make_ground_plane(lat, h_ground, cond.alpha, cond.beta, ref_point=ac.ref_point, height_ref="ref")
        if h_ground is not None
        else None
    )
    k_vals = [0.0, 0.05, 0.2, 0.5]

    r_wb = _solve_harmonic_rings(
        lat,
        cond,
        k_vals,
        S_ref=ac.S_ref,
        b_ref=ac.b_ref,
        c_ref=ac.c_ref,
        ground=ground,
        ref_point=ac.ref_point,
        solve_method="woodbury",
    )
    r_dir = _solve_harmonic_rings(
        lat,
        cond,
        k_vals,
        S_ref=ac.S_ref,
        b_ref=ac.b_ref,
        c_ref=ac.c_ref,
        ground=ground,
        ref_point=ac.ref_point,
        solve_method="direct",
    )

    assert isinstance(r_wb, _HarmonicResponse)
    assert np.array_equal(r_wb.k, np.asarray(k_vals, dtype=float))
    for motion in ("heave", "pitch"):
        m_wb = getattr(r_wb, motion)
        m_dir = getattr(r_dir, motion)
        for name in ("CL", "CD", "CY", "Cl", "Cm", "Cn"):
            a = getattr(m_wb, name)
            b = getattr(m_dir, name)
            scale = max(float(np.max(np.abs(b))), 1.0)
            assert np.max(np.abs(a - b)) <= 1e-12 * scale


# ---------------------------------------------------------------------------
# 2. Mirror-wing parity in ground effect (with and without 3D tail)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_tail", [False, True])
@pytest.mark.parametrize("wing_kind", ["rect", "swept"])
def test_mirror_wing_parity(use_tail, wing_kind):
    """Ground-image harmonic system equals an explicit antiphase two-wing free-air model to round-off."""
    ac = _rect_wing(n_panels=4) if wing_kind == "rect" else _swept_tapered_wing(n_panels=4)
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
    )
    rhs_ge, G_ge = sys_ge.solve_circulations(k, solve_method="woodbury")

    # Build explicit 2N x 2N free-air two-wing system (real wing + mirror wing, no ground kernel).
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
    rc_2 = np.concatenate([rings.rc, rings.rc], axis=0)
    rc_te_2 = np.concatenate([rings.rc[te_idx], rings.rc[te_idx]], axis=0)

    A_2 = VR.ring_normal_influence(cp_2, n_2, c_2, rc_2).astype(np.complex128)

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

    # Antiphase motion in the reflected frame gives rhs_mirror = +rhs_real and Gamma_mirror = -Gamma_real.
    rhs_2 = np.concatenate([rhs_ge, rhs_ge], axis=0)
    G_2 = np.linalg.solve(A_2, rhs_2)

    for col in range(G_ge.shape[1]):
        scale = max(float(np.max(np.abs(G_ge[:, col]))), 1.0)
        assert np.max(np.abs(G_2[:n, col] - G_ge[:, col])) <= 1e-12 * scale
        assert np.max(np.abs(G_2[n:, col] + G_ge[:, col])) <= 1e-12 * scale


# ---------------------------------------------------------------------------
# 3. k -> 0 limits against existing steady solvers (exact-equivalence & default)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("h_ground", [None, 0.5])
@pytest.mark.parametrize("alpha_deg", [0.0, 5.0])
def test_k0_pitch_limit_matches_analyze_exact_equivalence(monkeypatch, h_ground, alpha_deg):
    """Pitch k -> 0 limit matches ventorum.analyze CL_alpha and Cm_alpha to <= 1e-8 (exact-equivalence settings)."""
    ac = _rect_wing(semi_span=3.0, chord=1.0, n_panels=6)
    st = vt.SolverSettings(
        solver_type="vlm",
        n_panels=6,
        n_chord=4,
        spacing="half-cosine",
        chord_spacing="uniform",
    )
    cond = vt.FlightCondition(V_inf=30.0, rho=1.225, alpha=np.radians(alpha_deg), h=h_ground)

    monkeypatch.setattr(lattice_mod, "CORE_RADIUS_FRACTION", 0.0)
    lat = build_lattice(ac, st, collocation="vlm", n_chord=4, chord_spacing="uniform")
    ground = (
        make_ground_plane(lat, h_ground, cond.alpha, cond.beta, ref_point=ac.ref_point, height_ref="ref")
        if h_ground is not None
        else None
    )

    cl_a_ref, cm_a_ref = _steady_vlm_alpha_slopes(ac, cond, st, d_alpha=1e-5)
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
        rc=0.0,
    )

    assert abs(resp.k0_pitch.CL - cl_a_ref) <= 1e-8 * max(abs(cl_a_ref), 1.0)
    assert abs(resp.k0_pitch.Cm - cm_a_ref) <= 1e-8 * max(abs(cm_a_ref), 1.0)
    assert abs(resp.pitch.CL[0] - resp.k0_pitch.CL) <= 1e-12 * max(abs(cl_a_ref), 1.0)
    assert abs(resp.pitch.Cm[0] - resp.k0_pitch.Cm) <= 1e-12 * max(abs(cm_a_ref), 1.0)


def test_k0_heave_limit_matches_height_derivatives_exact_equivalence(monkeypatch):
    """Heave k -> 0 limit matches ventorum.height_derivatives CL_h and Cm_h to <= 1e-8 in loaded ground effect."""
    ac = _rect_wing(semi_span=3.0, chord=1.0, n_panels=6)
    st = vt.SolverSettings(
        solver_type="vlm",
        n_panels=6,
        n_chord=4,
        spacing="half-cosine",
        chord_spacing="uniform",
    )
    h_ground = 0.5
    cond = vt.FlightCondition(V_inf=30.0, rho=1.225, alpha=np.radians(5.0), h=h_ground)

    monkeypatch.setattr(lattice_mod, "CORE_RADIUS_FRACTION", 0.0)
    lat = build_lattice(ac, st, collocation="vlm", n_chord=4, chord_spacing="uniform")
    ground = make_ground_plane(lat, h_ground, cond.alpha, cond.beta, ref_point=ac.ref_point, height_ref="ref")

    # Small step in height_derivatives so FD truncation error is below 1e-9.
    hd = vt.height_derivatives(ac, cond, st, h_step=1e-4, alpha_step_deg=1e-3, axes="body")
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
        rc=0.0,
    )

    assert abs(resp.k0_heave.CL - hd.derivatives["CL_h"]) <= 1e-8 * max(abs(hd.derivatives["CL_h"]), 1.0)
    assert abs(resp.k0_heave.Cm - hd.derivatives["Cm_h"]) <= 1e-8 * max(abs(hd.derivatives["Cm_h"]), 1.0)
    assert abs(resp.k0_pitch.CL - hd.derivatives["CL_alpha"]) <= 1e-8 * max(abs(hd.derivatives["CL_alpha"]), 1.0)
    assert abs(resp.k0_pitch.Cm - hd.derivatives["Cm_alpha"]) <= 1e-8 * max(abs(hd.derivatives["Cm_alpha"]), 1.0)
    assert abs(resp.mean_state.CL - hd.base.totals.CL) <= 1e-10 * max(abs(hd.base.totals.CL), 1.0)
    assert abs(resp.mean_state.Cm - hd.base.totals.Cm) <= 1e-10 * max(abs(hd.base.totals.Cm), 1.0)


@pytest.mark.parametrize("h_ground", [None, 0.5])
@pytest.mark.parametrize("alpha_deg", [0.0, 5.0])
def test_k0_pitch_rate_limit_matches_rate_derivatives_exact_equivalence(monkeypatch, h_ground, alpha_deg):
    """Pitch-rate k -> 0 limit matches ventorum.rate_derivatives CL_q and Cm_q to <= 1e-8."""
    ac = _rect_wing(semi_span=3.0, chord=1.0, n_panels=6)
    st = vt.SolverSettings(
        solver_type="vlm",
        n_panels=6,
        n_chord=4,
        spacing="half-cosine",
        chord_spacing="uniform",
    )
    cond = vt.FlightCondition(V_inf=30.0, rho=1.225, alpha=np.radians(alpha_deg), h=h_ground)

    monkeypatch.setattr(lattice_mod, "CORE_RADIUS_FRACTION", 0.0)
    lat = build_lattice(ac, st, collocation="vlm", n_chord=4, chord_spacing="uniform")
    ground = (
        make_ground_plane(lat, h_ground, cond.alpha, cond.beta, ref_point=ac.ref_point, height_ref="ref")
        if h_ground is not None
        else None
    )

    rd = vt.rate_derivatives(ac, cond, st, step=0.01, axes="body")
    resp = _solve_harmonic_rings(
        lat,
        cond,
        [1e-4],
        S_ref=ac.S_ref,
        b_ref=ac.b_ref,
        c_ref=ac.c_ref,
        ground=ground,
        ref_point=ac.ref_point,
        te_extension_fraction=0.0,
        rc=0.0,
    )

    assert abs(resp.k0_pitch_rate.CL - rd.derivatives["CL_q"]) <= 1e-8 * max(abs(rd.derivatives["CL_q"]), 1.0)
    assert abs(resp.k0_pitch_rate.Cm - rd.derivatives["Cm_q"]) <= 1e-8 * max(abs(rd.derivatives["Cm_q"]), 1.0)

    # At zero mean load, check that the small-k harmonic response approaches CL_q and Cm_q smoothly.
    if alpha_deg == 0.0:
        k = resp.k[0]
        cl_q_from_k = ((resp.pitch.CL[0] - resp.heave.CL[0] / (-2.0j * k)) / (1j * k)).real
        cm_q_from_k = ((resp.pitch.Cm[0] - resp.heave.Cm[0] / (-2.0j * k)) / (1j * k)).real
        assert abs(cl_q_from_k - rd.derivatives["CL_q"]) <= 1e-3 * max(abs(rd.derivatives["CL_q"]), 1.0)
        assert abs(cm_q_from_k - rd.derivatives["Cm_q"]) <= 1e-3 * max(abs(rd.derivatives["Cm_q"]), 1.0)


def test_k0_default_settings_small_difference_from_steady_solvers():
    """At default ring settings (0.25 TE offset and Ventorum core), k -> 0 stays close to steady VLM."""
    ac = _rect_wing(semi_span=3.0, chord=1.0, n_panels=6)
    st = vt.SolverSettings(
        solver_type="vlm",
        n_panels=6,
        n_chord=4,
        spacing="half-cosine",
        chord_spacing="uniform",
    )
    h_ground = 0.5
    cond = vt.FlightCondition(V_inf=30.0, rho=1.225, alpha=np.radians(5.0), h=h_ground)
    lat = build_lattice(ac, st, collocation="vlm", n_chord=4, chord_spacing="uniform")
    ground = make_ground_plane(lat, h_ground, cond.alpha, cond.beta, ref_point=ac.ref_point, height_ref="ref")

    hd = vt.height_derivatives(ac, cond, st, h_step=1e-4, alpha_step_deg=1e-3, axes="body")
    rd = vt.rate_derivatives(ac, cond, st, step=0.01, axes="body")
    resp = _solve_harmonic_rings(
        lat,
        cond,
        [0.0],
        S_ref=ac.S_ref,
        b_ref=ac.b_ref,
        c_ref=ac.c_ref,
        ground=ground,
        ref_point=ac.ref_point,
    )

    assert abs(resp.k0_pitch.CL - hd.derivatives["CL_alpha"]) <= 5e-3 * abs(hd.derivatives["CL_alpha"])
    assert abs(resp.k0_heave.CL - hd.derivatives["CL_h"]) <= 5e-3 * abs(hd.derivatives["CL_h"])
    assert abs(resp.k0_pitch_rate.CL - rd.derivatives["CL_q"]) <= 5e-3 * abs(rd.derivatives["CL_q"])
    assert abs(resp.k0_pitch_rate.Cm - rd.derivatives["Cm_q"]) <= 5e-3 * abs(rd.derivatives["Cm_q"])


def test_loaded_mean_state_fd_step_independence():
    """Loaded-mean-state ground-effect response is independent of the geometry finite-difference step."""
    ac = _rect_wing(semi_span=3.0, chord=1.0, n_panels=6)
    lat = build_lattice(ac, n_chord=4, chord_spacing="uniform")
    h_ground = 0.5
    cond = vt.FlightCondition(V_inf=30.0, rho=1.225, alpha=np.radians(5.0), h=h_ground)
    ground = make_ground_plane(lat, h_ground, cond.alpha, cond.beta, ref_point=ac.ref_point, height_ref="ref")

    res_list = []
    for step in (1e-6, 1e-5, 1e-4):
        res_list.append(
            _solve_harmonic_rings(
                lat,
                cond,
                [0.2],
                S_ref=ac.S_ref,
                b_ref=ac.b_ref,
                c_ref=ac.c_ref,
                ground=ground,
                ref_point=ac.ref_point,
                fd_step_h=step,
                fd_step_theta=step,
            )
        )
    ref = res_list[1]
    for alt in (res_list[0], res_list[2]):
        for motion in ("heave", "pitch"):
            m_ref = getattr(ref, motion)
            m_alt = getattr(alt, motion)
            for name in ("CL", "Cm"):
                scale = max(float(np.abs(getattr(m_ref, name)[0])), 1.0)
                assert abs(getattr(m_alt, name)[0] - getattr(m_ref, name)[0]) <= 1e-7 * scale


# ---------------------------------------------------------------------------
# 4. Trailing-edge load correction: zero at k = 0 and 2nd-order Cm_alpha_dot
# ---------------------------------------------------------------------------


def test_trailing_edge_correction_zero_at_k0():
    """Trailing-edge correction vanishes to round-off at k = 0."""
    ac = _swept_tapered_wing(n_panels=6)
    lat = build_lattice(ac, n_chord=4, chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=30.0, rho=1.225, alpha=np.radians(5.0), h=0.8)
    ground = make_ground_plane(lat, 0.8, cond.alpha, cond.beta, ref_point=ac.ref_point, height_ref="ref")

    r_on = _solve_harmonic_rings(
        lat, cond, [0.0], S_ref=ac.S_ref, b_ref=ac.b_ref, c_ref=ac.c_ref,
        ground=ground, ref_point=ac.ref_point, te_correction=True,
    )
    r_off = _solve_harmonic_rings(
        lat, cond, [0.0], S_ref=ac.S_ref, b_ref=ac.b_ref, c_ref=ac.c_ref,
        ground=ground, ref_point=ac.ref_point, te_correction=False,
    )

    for motion in ("heave", "pitch"):
        m_on = getattr(r_on, motion)
        m_off = getattr(r_off, motion)
        for name in ("CL", "CD", "CY", "Cl", "Cm", "Cn"):
            assert abs(getattr(m_on, name)[0] - getattr(m_off, name)[0]) <= 1e-15


@pytest.mark.slow
@pytest.mark.parametrize("h_ground", [None, 0.5])
def test_trailing_edge_correction_second_order_cm_alpha_dot(h_ground):
    """Re(Cm_alpha_dot) at k = 0.2 on rectangular AR 6 wing converges at second order (1.7 <= p <= 2.3)."""
    ac = _rect_wing(semi_span=3.0, chord=1.0, n_panels=12)
    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=0.0, h=h_ground)
    k = 0.2

    vals_on = []
    vals_off = []
    for n_c in (4, 8, 16):
        lat = build_lattice(ac, n_chord=n_c, chord_spacing="uniform")
        ground = (
            make_ground_plane(lat, h_ground, cond.alpha, cond.beta, ref_point=ac.ref_point, height_ref="ref")
            if h_ground is not None
            else None
        )
        for te_corr, store in ((True, vals_on), (False, vals_off)):
            resp = _solve_harmonic_rings(
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
                te_correction=te_corr,
            )
            cm_adot = ((resp.heave.Cm[0] - resp.k0_heave.Cm) / (-2.0j * k) - resp.k0_pitch.Cm) / (1j * k)
            store.append(float(cm_adot.real))

    order_on = float(np.log2(abs(vals_on[0] - vals_on[1]) / abs(vals_on[1] - vals_on[2])))
    order_off = float(np.log2(abs(vals_off[0] - vals_off[1]) / abs(vals_off[1] - vals_off[2])))

    assert 1.7 <= order_on <= 2.3, f"Expected order in [1.7, 2.3] with TE correction, got {order_on:.3f} ({vals_on})"
    assert 0.8 <= order_off <= 1.2, f"Expected first-order without TE correction, got {order_off:.3f} ({vals_off})"


# ---------------------------------------------------------------------------
# 5. Wake tail accuracy: 5 chords + tail agrees with 40 chords + tail
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("h_ground", [None, 0.5])
def test_wake_tail_5c_matches_40c(h_ground):
    """5 chords plus the 3D harmonic tail agrees with 40 chords plus the tail within 1e-4 relative."""
    ac = _rect_wing(semi_span=3.0, chord=1.0, n_panels=8)
    lat = build_lattice(ac, n_chord=4, chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=np.radians(4.0), h=h_ground)
    ground = (
        make_ground_plane(lat, h_ground, cond.alpha, cond.beta, ref_point=ac.ref_point, height_ref="ref")
        if h_ground is not None
        else None
    )
    k_vals = [0.05, 0.2, 0.5]

    r_5c = _solve_harmonic_rings(
        lat, cond, k_vals, S_ref=ac.S_ref, b_ref=ac.b_ref, c_ref=ac.c_ref,
        ground=ground, ref_point=ac.ref_point, wake_chords=5.0, use_tail=True,
    )
    r_40c = _solve_harmonic_rings(
        lat, cond, k_vals, S_ref=ac.S_ref, b_ref=ac.b_ref, c_ref=ac.c_ref,
        ground=ground, ref_point=ac.ref_point, wake_chords=40.0, use_tail=True,
    )
    r_5c_notail = _solve_harmonic_rings(
        lat, cond, k_vals, S_ref=ac.S_ref, b_ref=ac.b_ref, c_ref=ac.c_ref,
        ground=ground, ref_point=ac.ref_point, wake_chords=5.0, use_tail=False,
    )

    for motion in ("heave", "pitch"):
        m_5 = getattr(r_5c, motion)
        m_40 = getattr(r_40c, motion)
        m_nt = getattr(r_5c_notail, motion)
        for name in ("CL", "Cm"):
            err_tail = np.abs(getattr(m_5, name) - getattr(m_40, name)) / np.maximum(np.abs(getattr(m_40, name)), 1.0)
            err_notail = np.abs(getattr(m_nt, name) - getattr(m_40, name)) / np.maximum(np.abs(getattr(m_40, name)), 1.0)
            # Measured max relative difference: 1.25e-5 (free air) and 1.03e-6 (h/c = 0.5).
            assert np.max(err_tail) <= 5e-5
            # At k = 0.05, the tail cuts truncation error by more than an order of magnitude.
            if name == "CL":
                assert err_tail[0] < 0.05 * err_notail[0]


def test_tail_velocity_k0_matches_steady_wake():
    """At kappa = 0, harmonic_tail_velocity matches steady_wake_velocity to round-off."""
    ac = _swept_tapered_wing(n_panels=4)
    lat = build_lattice(ac, n_chord=3, chord_spacing="uniform")
    rings = build_ring_lattice(lat, wake_dir=freestream_direction(np.radians(4.0), np.radians(2.0)))
    p_l = rings.te_left + 5.0 * rings.wake_dir
    p_r = rings.te_right + 5.0 * rings.wake_dir
    rc_w = rings.rc[rings.te_indices]

    v_tail = harmonic_tail_velocity(rings.cp, p_l, p_r, rings.wake_dir, 0.0, rc_w)
    v_steady = VR.steady_wake_velocity(rings.cp, p_l, p_r, rings.wake_dir, rc_w)
    assert np.max(np.abs(v_tail.imag)) == 0.0
    assert np.max(np.abs(v_tail.real - v_steady)) <= 1e-14 * np.max(np.abs(v_steady))


# ---------------------------------------------------------------------------
# 6. High aspect ratio limit: AR 40 mid-span strip lift approaches Theodorsen
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_high_aspect_ratio_midspan_approaches_theodorsen():
    """Mid-span strip lift of a rectangular AR 40 wing approaches Theodorsen C(k) at k = 0.1 and 0.5."""
    ac = _rect_wing(semi_span=20.0, chord=1.0, n_panels=20)
    lat = build_lattice(ac, n_chord=8, chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=0.0)
    k_vals = [0.1, 0.5]

    resp = _solve_harmonic_rings(
        lat,
        cond,
        k_vals,
        S_ref=ac.S_ref,
        b_ref=ac.b_ref,
        c_ref=ac.c_ref,
        ref_point=ac.ref_point,
        wake_chords=10.0,
        use_tail=True,
    )

    # Mid-span strip is closest to y = 0.
    mid_strip = int(np.argmin(np.abs(lat.qc_mid[:, 1])))
    for i, k in enumerate(k_vals):
        # 2D Theodorsen heave lift per unit z / c = 1 (upward positive):
        # cl_2d(k) = 2 * pi * k^2 - 4 * pi * i * k * C(k).
        ck = _theodorsen_c(k)
        cl_theo = 2.0 * np.pi * k * k - 4.0j * np.pi * k * ck
        cl_mid = resp.heave.strip_cl[i, mid_strip]
        rel_err = abs(cl_mid - cl_theo) / abs(cl_theo)
        # Measured relative differences on AR 40 (8 x 40): 0.33 % at k = 0.1 and 0.17 % at k = 0.5.
        assert rel_err < 0.01, f"k={k}: mid-span cl={cl_mid}, Theodorsen={cl_theo}, rel_err={rel_err:.4f}"


def test_non_unit_chord_and_tapered_wing_k0_and_metadata(monkeypatch):
    """Non-unit chord and tapered wing keep exact k -> 0 scaling and record per-strip dx_te_ratio."""
    ac2 = _rect_wing(semi_span=6.0, chord=2.0, n_panels=6)
    st = vt.SolverSettings(
        solver_type="vlm", n_panels=6, n_chord=4, spacing="half-cosine", chord_spacing="uniform",
    )
    cond2 = vt.FlightCondition(V_inf=30.0, rho=1.225, alpha=np.radians(5.0), h=1.0)
    monkeypatch.setattr(lattice_mod, "CORE_RADIUS_FRACTION", 0.0)
    lat2 = build_lattice(ac2, st, collocation="vlm", n_chord=4, chord_spacing="uniform")
    g2 = make_ground_plane(lat2, 1.0, cond2.alpha, cond2.beta, ref_point=ac2.ref_point, height_ref="ref")

    hd2 = vt.height_derivatives(ac2, cond2, st, h_step=1e-4, alpha_step_deg=1e-3, axes="body")
    rd2 = vt.rate_derivatives(ac2, cond2, st, step=0.01, axes="body")
    r2 = _solve_harmonic_rings(
        lat2, cond2, [0.0, 0.2], S_ref=ac2.S_ref, b_ref=ac2.b_ref, c_ref=ac2.c_ref,
        ground=g2, ref_point=ac2.ref_point, te_extension_fraction=0.0, rc=0.0,
    )
    assert abs(r2.k0_heave.CL - hd2.derivatives["CL_h"]) <= 1e-8 * max(abs(hd2.derivatives["CL_h"]), 1.0)
    assert abs(r2.k0_heave.Cm - hd2.derivatives["Cm_h"]) <= 1e-8 * max(abs(hd2.derivatives["Cm_h"]), 1.0)
    assert abs(r2.k0_pitch.CL - hd2.derivatives["CL_alpha"]) <= 1e-8 * max(abs(hd2.derivatives["CL_alpha"]), 1.0)
    assert abs(r2.k0_pitch_rate.CL - rd2.derivatives["CL_q"]) <= 1e-8 * max(abs(rd2.derivatives["CL_q"]), 1.0)
    assert np.allclose(r2.dx_te_ratio, 1.0, atol=1e-14)
    assert abs(r2.dt - r2.dx_w / cond2.V_inf) <= 1e-15

    # Tapered wing records non-uniform dx_te_ratio across the span and stays private (not in __all__).
    monkeypatch.setattr(lattice_mod, "CORE_RADIUS_FRACTION", 1e-4)
    ac_sw = _swept_tapered_wing(n_panels=6)
    cond_sw = vt.FlightCondition(V_inf=30.0, rho=1.225, alpha=np.radians(5.0), h=0.8)
    lat_sw = build_lattice(ac_sw, st, collocation="vlm", n_chord=4, chord_spacing="uniform")
    g_sw = make_ground_plane(lat_sw, 0.8, cond_sw.alpha, cond_sw.beta, ref_point=ac_sw.ref_point, height_ref="ref")
    hd_sw = vt.height_derivatives(ac_sw, cond_sw, st, h_step=1e-4, alpha_step_deg=1e-3, axes="body")
    r_sw = _solve_harmonic_rings(
        lat_sw, cond_sw, [0.2], S_ref=ac_sw.S_ref, b_ref=ac_sw.b_ref, c_ref=ac_sw.c_ref,
        ground=g_sw, ref_point=ac_sw.ref_point, te_extension_fraction=0.0,
    )
    assert r_sw.dx_te_ratio.shape == (lat_sw.n_strips,)
    assert float(np.max(r_sw.dx_te_ratio)) > 1.05
    assert float(np.min(r_sw.dx_te_ratio)) < 0.95
    assert abs(r_sw.k0_heave.CL - hd_sw.derivatives["CL_h"]) <= 1e-8 * max(abs(hd_sw.derivatives["CL_h"]), 1.0)
    assert abs(r_sw.k0_pitch.CL - hd_sw.derivatives["CL_alpha"]) <= 1e-8 * max(abs(hd_sw.derivatives["CL_alpha"]), 1.0)
    assert "_HarmonicResponse" not in getattr(vt, "__all__", [])
    assert "_solve_harmonic_rings" not in getattr(vt, "__all__", [])
    assert vt.unsteady.__all__ == []


# ---------------------------------------------------------------------------
# 7. Refusal tests for invalid inputs and refused lattices
# ---------------------------------------------------------------------------


def test_refuses_invalid_k_and_wake_length():
    """Negative or non-finite k, empty k, non-positive wake length and bad solve_method raise InputError."""
    ac = _rect_wing(n_panels=4)
    lat = build_lattice(ac, n_chord=2, chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=0.0)
    common = {"S_ref": ac.S_ref, "b_ref": ac.b_ref, "c_ref": ac.c_ref, "ref_point": ac.ref_point}

    for bad_k in ([-0.1], [0.1, -0.05], [np.nan], [np.inf], [-np.inf], []):
        with pytest.raises(InputError, match="k"):
            _solve_harmonic_rings(lat, cond, bad_k, **common)

    for bad_wake in (0.0, -2.0, np.nan, np.inf):
        with pytest.raises(InputError, match="wake_chords"):
            _solve_harmonic_rings(lat, cond, [0.1], wake_chords=bad_wake, **common)

    with pytest.raises(InputError, match="solve_method"):
        _solve_harmonic_rings(lat, cond, [0.1], solve_method="invalid", **common)


def test_refuses_unsupported_lattices_and_ground_strike():
    """Lifting-line collocation raises InputError; ground strike raises GroundStrikeError."""
    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=0.0)

    # Lifting-line collocation remains refused.
    ac_rect = _rect_wing(n_panels=4)
    lat_llt = build_lattice(ac_rect, collocation="llt", n_chord=1)
    with pytest.raises(InputError, match="collocation"):
        _solve_harmonic_rings(
            lat_llt, cond, [0.1], S_ref=ac_rect.S_ref, b_ref=ac_rect.b_ref, c_ref=ac_rect.c_ref,
        )

    # Ground strike: large alpha at very low height lowers the trailing edge below ground.
    lat_ok = build_lattice(ac_rect, n_chord=4)
    cond_strike = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=np.radians(20.0), h=0.05)
    ground_strike = make_ground_plane(
        lat_ok, 0.05, cond_strike.alpha, cond_strike.beta, ref_point=ac_rect.ref_point, height_ref="ref",
    )
    with pytest.raises(GroundStrikeError):
        _solve_harmonic_rings(
            lat_ok, cond_strike, [0.1],
            S_ref=ac_rect.S_ref, b_ref=ac_rect.b_ref, c_ref=ac_rect.c_ref,
            ground=ground_strike, ref_point=ac_rect.ref_point,
        )


# ---------------------------------------------------------------------------
# 8. Cambered wings, deflected flaps and surface junctions
# ---------------------------------------------------------------------------


def _cambered_naca4412_wing(n_panels: int = 6) -> vt.Aircraft:
    """Build a rectangular NACA 4412 wing with quarter-chord moment reference."""
    wing = vt.LiftingSurface(
        name="cambered",
        semi_span=3.0,
        n_panels=n_panels,
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.0, shape="4412"),
            vt.WingSection(y_frac=1.0, chord=1.0, shape="4412"),
        ],
    )
    ac = vt.Aircraft(surfaces=[wing], ref_point=np.array([0.25, 0.0, 0.0]))
    ac.compute_reference_values()
    return ac


def _deflected_flap_wing(deflection_deg: float = 20.0, n_panels: int = 6) -> vt.Aircraft:
    """Build a rectangular wing with a full-span chordwise deflected flap."""
    fl = vt.ControlSurface(
        name="flap",
        eta_start=0.0,
        eta_end=1.0,
        hinge_x_c=0.75,
        deflection=np.radians(deflection_deg),
    )
    wing = vt.LiftingSurface(
        name="flapped",
        semi_span=3.0,
        n_panels=n_panels,
        controls=[fl],
        sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=1.0)],
    )
    ac = vt.Aircraft(surfaces=[wing], ref_point=np.array([0.25, 0.0, 0.0]))
    ac.compute_reference_values()
    return ac


def _wing_with_winglet(n_wing: int = 6, n_winglet: int = 4) -> vt.Aircraft:
    """Build a rectangular wing with a vertical winglet joined at the right tip."""
    wing = vt.LiftingSurface(
        name="wing",
        semi_span=3.0,
        n_panels=n_wing,
        sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=1.0)],
    )
    winglet = vt.LiftingSurface(
        name="winglet",
        semi_span=0.5,
        n_panels=n_winglet,
        is_symmetric=False,
        dihedral=np.radians(90.0),
        position=np.array([0.0, 3.0, 0.0]),
        sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=0.5)],
    )
    ac = vt.Aircraft(surfaces=[wing, winglet], ref_point=np.array([0.25, 0.0, 0.0]))
    ac.compute_reference_values()
    return ac


@pytest.mark.parametrize("case_kind", ["camber", "flap", "winglet"])
def test_harmonic_runs_and_woodbury_matches_direct_on_camber_flap_winglet(case_kind):
    """Harmonic solver runs on cambered, flapped and joined-winglet lattices; Woodbury matches direct LU."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if case_kind == "camber":
            ac = _cambered_naca4412_wing(n_panels=4)
        elif case_kind == "flap":
            ac = _deflected_flap_wing(deflection_deg=20.0, n_panels=4)
        else:
            ac = _wing_with_winglet(n_wing=4, n_winglet=2)
        lat = build_lattice(ac, n_chord=4, chord_spacing="uniform")

    cond = vt.FlightCondition(V_inf=25.0, rho=1.225, alpha=np.radians(4.0), h=0.5)
    ground = make_ground_plane(lat, 0.5, cond.alpha, cond.beta, ref_point=ac.ref_point, height_ref="ref")
    k_vals = [0.0, 0.05, 0.2]

    r_wb = _solve_harmonic_rings(
        lat, cond, k_vals, S_ref=ac.S_ref, b_ref=ac.b_ref, c_ref=ac.c_ref,
        ground=ground, ref_point=ac.ref_point, solve_method="woodbury",
    )
    r_dir = _solve_harmonic_rings(
        lat, cond, k_vals, S_ref=ac.S_ref, b_ref=ac.b_ref, c_ref=ac.c_ref,
        ground=ground, ref_point=ac.ref_point, solve_method="direct",
    )

    for motion in ("heave", "pitch"):
        m_wb = getattr(r_wb, motion)
        m_dir = getattr(r_dir, motion)
        for name in ("CL", "CD", "CY", "Cl", "Cm", "Cn"):
            a = getattr(m_wb, name)
            b = getattr(m_dir, name)
            assert np.all(np.isfinite(a))
            scale = max(float(np.max(np.abs(b))), 1.0)
            assert np.max(np.abs(a - b)) <= 1e-12 * scale


@pytest.mark.parametrize("case_kind", ["camber", "flap", "winglet"])
def test_mirror_wing_parity_bent_and_junction(case_kind):
    """Ground-image harmonic system equals explicit antiphase two-wing model on camber, flap and winglet."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if case_kind == "camber":
            ac = _cambered_naca4412_wing(n_panels=4)
        elif case_kind == "flap":
            ac = _deflected_flap_wing(deflection_deg=20.0, n_panels=4)
        else:
            ac = _wing_with_winglet(n_wing=4, n_winglet=2)
        lat = build_lattice(ac, n_chord=4, chord_spacing="uniform")

    h_ground = 0.5
    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=np.radians(4.0), h=h_ground)
    ground = make_ground_plane(lat, h_ground, cond.alpha, cond.beta, ref_point=ac.ref_point, height_ref="ref")
    k = 0.2

    sys_ge = _HarmonicRingSystem(
        lat, cond, S_ref=ac.S_ref, b_ref=ac.b_ref, c_ref=ac.c_ref,
        ground=ground, ref_point=ac.ref_point, wake_chords=5.0, use_tail=True,
    )
    rhs_ge, G_ge = sys_ge.solve_circulations(k, solve_method="woodbury")

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

    phase0 = np.exp(-1j * omega * (n_rows + 0.5) * dt)
    tail_l = te_l_2 + (n_rows * dx_w) * wd_2
    tail_r = te_r_2 + (n_rows * dx_w) * wd_2
    W_2 += phase0 * harmonic_tail_normal_influence(cp_2, n_2, tail_l, tail_r, wd_2, kappa, rc_te_2)
    A_2[:, te_idx_2] += W_2

    rhs_2 = np.concatenate([rhs_ge, rhs_ge], axis=0)
    G_2 = np.linalg.solve(A_2, rhs_2)

    for col in range(G_ge.shape[1]):
        scale = max(float(np.max(np.abs(G_ge[:, col]))), 1.0)
        assert np.max(np.abs(G_2[:n, col] - G_ge[:, col])) <= 1e-12 * scale
        assert np.max(np.abs(G_2[n:, col] + G_ge[:, col])) <= 1e-12 * scale


@pytest.mark.parametrize(
    "case_kind,h_ground",
    [
        ("camber", None),
        ("camber", 0.5),
        ("flap", None),
        ("flap", 0.5),
        ("winglet", None),
        ("winglet", 0.5),
    ],
)
def test_k0_limits_match_steady_solvers_camber_flap_winglet(monkeypatch, case_kind, h_ground):
    """k -> 0 limits match analyze, height_derivatives and rate_derivatives on camber, flap and winglet."""
    from ventorum import gpu

    old_prec = gpu.get_precision()
    gpu.set_precision("float64")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if case_kind == "camber":
                ac = _cambered_naca4412_wing(n_panels=6)
            elif case_kind == "flap":
                ac = _deflected_flap_wing(deflection_deg=20.0, n_panels=6)
            else:
                ac = _wing_with_winglet(n_wing=6, n_winglet=4)

            st = vt.SolverSettings(
                solver_type="vlm",
                n_panels=6,
                n_chord=8,
                spacing="half-cosine",
                chord_spacing="uniform",
            )
            cond = vt.FlightCondition(V_inf=30.0, rho=1.225, alpha=np.radians(4.0), h=h_ground)

            # Use CORE_RADIUS_FRACTION = 1e-4 (as in the 3D swept-tapered check above)
            # so 3D non-planar leg midpoints do not hit 1e-16 round-off singularities.
            monkeypatch.setattr(lattice_mod, "CORE_RADIUS_FRACTION", 1e-4)
            lat = build_lattice(ac, st, collocation="vlm", n_chord=8, chord_spacing="uniform")
            ground = (
                make_ground_plane(lat, h_ground, cond.alpha, cond.beta, ref_point=ac.ref_point, height_ref="ref")
                if h_ground is not None
                else None
            )

            cl_a_ref, cm_a_ref = _steady_vlm_alpha_slopes(ac, cond, st, d_alpha=1e-5)
            rd = vt.rate_derivatives(ac, cond, st, step=0.01, axes="body")
            hd = (
                vt.height_derivatives(ac, cond, st, h_step=1e-4, alpha_step_deg=1e-3, axes="body")
                if h_ground is not None
                else None
            )

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
        )

        assert abs(resp.k0_pitch.CL - cl_a_ref) <= 1e-8 * max(abs(cl_a_ref), 1.0)
        assert abs(resp.k0_pitch.Cm - cm_a_ref) <= 1e-8 * max(abs(cm_a_ref), 1.0)
        assert abs(resp.pitch.CL[0] - resp.k0_pitch.CL) <= 1e-12 * max(abs(cl_a_ref), 1.0)
        assert abs(resp.pitch.Cm[0] - resp.k0_pitch.Cm) <= 1e-12 * max(abs(cm_a_ref), 1.0)
        assert abs(resp.k0_pitch_rate.CL - rd.derivatives["CL_q"]) <= 1e-8 * max(abs(rd.derivatives["CL_q"]), 1.0)
        assert abs(resp.k0_pitch_rate.Cm - rd.derivatives["Cm_q"]) <= 1e-8 * max(abs(rd.derivatives["Cm_q"]), 1.0)

        if hd is not None:
            assert abs(resp.k0_heave.CL - hd.derivatives["CL_h"]) <= 1e-8 * max(abs(hd.derivatives["CL_h"]), 1.0)
            assert abs(resp.k0_heave.Cm - hd.derivatives["Cm_h"]) <= 1e-8 * max(abs(hd.derivatives["Cm_h"]), 1.0)
            assert abs(resp.mean_state.CL - hd.base.totals.CL) <= 1e-8 * max(abs(hd.base.totals.CL), 1.0)
            assert abs(resp.mean_state.Cm - hd.base.totals.Cm) <= 1e-8 * max(abs(hd.base.totals.Cm), 1.0)
    finally:
        gpu.set_precision(old_prec)


def test_bent_ring_apparent_mass_area_follows_hinge():
    """Apparent-mass area and moment on a flap-hinge ring integrate the bent sub-quadrilaterals."""
    fl = vt.ControlSurface(name="flap", eta_start=0.0, eta_end=1.0, hinge_x_c=0.75, deflection=np.radians(20.0))
    wing = vt.LiftingSurface(
        name="tapered_flapped",
        semi_span=3.0,
        n_panels=4,
        controls=[fl],
        sections=[vt.WingSection(y_frac=0.0, chord=1.2), vt.WingSection(y_frac=1.0, chord=0.6)],
    )
    ac = vt.Aircraft(surfaces=[wing], ref_point=np.array([0.3, 0.0, 0.0]))
    ac.compute_reference_values()
    lat = build_lattice(ac, n_chord=4, chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=0.0)
    sys = _HarmonicRingSystem(
        lat, cond, S_ref=ac.S_ref, b_ref=ac.b_ref, c_ref=ac.c_ref, ref_point=ac.ref_point,
    )
    # On a 4-chord uniform mesh with hinge at 0.75, panel k = 2 crosses the 20 deg hinge.
    p = 2
    a0, b0 = lat.a[p], lat.b[p]
    hl, hr = lat.a_kink[p], lat.b_kink[p]
    a1, b1 = lat.a[p + 1], lat.b[p + 1]
    rp = ac.ref_point
    tris = (
        (a0, hr, b0),
        (a0, hl, hr),
        (hl, b1, hr),
        (hl, a1, b1),
    )
    expected_vec = sum(0.5 * np.cross(t1 - t0, t2 - t0) for t0, t1, t2 in tris)
    expected_mom = sum(
        np.cross((t0 + t1 + t2) / 3.0 - rp, 0.5 * np.cross(t1 - t0, t2 - t0))
        for t0, t1, t2 in tris
    )
    flat_a1 = 0.5 * np.cross(b1 - a0, b0 - a0)
    flat_a2 = 0.5 * np.cross(a1 - a0, b1 - a0)
    flat_c = (
        np.linalg.norm(flat_a1) * (a0 + b0 + b1) / 3.0
        + np.linalg.norm(flat_a2) * (a0 + b1 + a1) / 3.0
    ) / (np.linalg.norm(flat_a1) + np.linalg.norm(flat_a2))
    flat_mom = np.cross(flat_c - rp, flat_a1 + flat_a2)
    assert sys.area_moment_vec is not None
    assert np.allclose(sys.area_vec[p], expected_vec, atol=1e-14, rtol=1e-14)
    assert np.allclose(sys.area_moment_vec[p], expected_mom, atol=1e-14, rtol=1e-14)
    assert np.linalg.norm(sys.area_moment_vec[p] - flat_mom) > 1e-4 * np.linalg.norm(expected_mom)
