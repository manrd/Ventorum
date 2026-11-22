# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for the vortex-ring lattice, ground images and steady-solver parity."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import ventorum as vt
from ventorum.aero import vortex_ring as VR
from ventorum.aero.system import (
    freestream_direction,
    make_ground_plane,
    make_unknown_map,
)
from ventorum.core.errors import InputError
from ventorum.geometry import lattice as lattice_mod
from ventorum.geometry.lattice import build_lattice
from ventorum.solvers.core import _vlm_rhs, solve_vlm
from ventorum.unsteady.rings import (
    RingLattice,
    _solve_steady_rings,
    _steady_ring_influence,
    build_ring_lattice,
    ground_image_rings,
)


def _planar_rect_wing(semi_span: float = 3.0, chord: float = 1.0, n_panels: int = 12) -> vt.Aircraft:
    wing = vt.LiftingSurface(
        name="wing",
        semi_span=semi_span,
        n_panels=n_panels,
        sections=[
            vt.WingSection(y_frac=0.0, chord=chord),
            vt.WingSection(y_frac=1.0, chord=chord),
        ],
    )
    return vt.Aircraft(surfaces=[wing], ref_point=np.array([0.25 * chord, 0.0, 0.0]))


def _swept_tapered_dihedral_wing(n_panels: int = 12) -> vt.Aircraft:
    wing = vt.LiftingSurface(
        name="swept",
        semi_span=3.0,
        n_panels=n_panels,
        sweep_le=np.radians(25.0),
        dihedral=np.radians(8.0),
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.2, twist=np.radians(1.0)),
            vt.WingSection(y_frac=1.0, chord=0.6, twist=np.radians(-2.0)),
        ],
    )
    return vt.Aircraft(surfaces=[wing], ref_point=np.array([0.3, 0.0, 0.0]))


def _wing_and_tail() -> vt.Aircraft:
    wing = vt.LiftingSurface(
        name="wing",
        semi_span=3.0,
        n_panels=6,
        dihedral=np.radians(5.0),
        sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=0.6)],
    )
    tail = vt.LiftingSurface(
        name="tail",
        semi_span=1.2,
        n_panels=4,
        position=np.array([3.5, 0.0, 0.2]),
        sections=[vt.WingSection(y_frac=0.0, chord=0.5), vt.WingSection(y_frac=1.0, chord=0.4)],
    )
    return vt.Aircraft(surfaces=[wing, tail])


def _assert_ring_topology(lat, rings: RingLattice, te_ext: float = 0.25) -> None:
    """Check corner layout, shared edges, TE closure rule and lattice consistency."""
    n_p = lat.n_panels
    n_s = lat.n_strips
    n_c = lat.n_chord

    assert rings.n_panels == n_p
    assert rings.n_strips == n_s
    assert rings.n_chord == n_c
    assert rings.corners.shape == (n_p, 4, 3)
    assert rings.cp.shape == (n_p, 3)
    assert rings.normal_bc.shape == (n_p, 3)
    assert rings.panel_strip.shape == (n_p,)
    assert rings.panel_chord_index.shape == (n_p,)
    assert rings.te_mask.shape == (n_p,)
    assert rings.te_left.shape == (n_s, 3)
    assert rings.te_right.shape == (n_s, 3)
    assert rings.rc.shape == (n_p,)
    assert rings.wake_dir.shape == (n_s, 3)

    assert np.array_equal(rings.cp, lat.cp)
    assert np.array_equal(rings.normal_bc, lat.normal_bc)
    assert np.array_equal(rings.rc, lat.rc)
    assert np.array_equal(rings.corners[:, 0], lat.a)
    assert np.array_equal(rings.corners[:, 1], lat.b)
    assert int(np.count_nonzero(rings.te_mask)) == n_s

    for s in range(n_s):
        for k in range(n_c):
            p = s * n_c + k
            assert rings.panel_strip[p] == s
            assert rings.panel_chord_index[p] == k
            if k < n_c - 1:
                assert not rings.te_mask[p]
                assert np.array_equal(rings.corners[p, 3], lat.a[p + 1])
                assert np.array_equal(rings.corners[p, 2], lat.b[p + 1])
            else:
                assert rings.te_mask[p]
                expected_l = lat.te_left[s] + (te_ext / 0.75) * (lat.te_left[s] - lat.a[p])
                expected_r = lat.te_right[s] + (te_ext / 0.75) * (lat.te_right[s] - lat.b[p])
                assert np.allclose(rings.corners[p, 3], expected_l, atol=1e-14, rtol=1e-14)
                assert np.allclose(rings.corners[p, 2], expected_r, atol=1e-14, rtol=1e-14)
                assert np.array_equal(rings.te_left[s], rings.corners[p, 3])
                assert np.array_equal(rings.te_right[s], rings.corners[p, 2])

    # Adjacent strips within the same surface share their common side-edge corners.
    for s in range(n_s - 1):
        if lat.strip_surface[s] != lat.strip_surface[s + 1]:
            continue
        for k in range(n_c):
            p_l = s * n_c + k
            p_r = (s + 1) * n_c + k
            assert np.allclose(rings.corners[p_l, 1], rings.corners[p_r, 0], atol=1e-13)
            assert np.allclose(rings.corners[p_l, 2], rings.corners[p_r, 3], atol=1e-13)


def test_build_ring_lattice_supported_geometries():
    """Ring lattice maps all non-kinked, non-joined geometries from build_lattice."""
    cases = []

    # Planar rectangular symmetric wing (n_chord=4 and n_chord=1).
    ac_rect = _planar_rect_wing(n_panels=6)
    cases.append(build_lattice(ac_rect, n_chord=4))
    cases.append(build_lattice(ac_rect, n_chord=1))

    # Swept, tapered wing with dihedral and twist.
    cases.append(build_lattice(_swept_tapered_dihedral_wing(n_panels=6), n_chord=4))

    # Asymmetric surface plus explicit mirror_y copy (twin vertical fins).
    fin_r = vt.LiftingSurface(
        name="fin_r", semi_span=1.0, n_panels=4, is_symmetric=False,
        dihedral=np.radians(90.0), position=np.array([2.0, 1.0, 0.0]),
        sections=[vt.WingSection(y_frac=0.0, chord=0.8), vt.WingSection(y_frac=1.0, chord=0.5)],
    )
    cases.append(build_lattice(vt.Aircraft(surfaces=[fin_r, fin_r.mirrored()]), n_chord=3))

    # Multiple separate surfaces (wing + tail).
    cases.append(build_lattice(_wing_and_tail(), n_chord=3))

    # Turned-normal cambered wing (LinearAirfoil alpha_L0 != 0).
    af_cam = vt.LinearAirfoil(name="cam", a0=2.0 * np.pi, alpha_L0=np.radians(-4.0))
    w_cam_lin = vt.LiftingSurface(
        name="w", semi_span=3.0, n_panels=4,
        sections=[vt.WingSection(y_frac=0.0, chord=1.0, airfoil=af_cam),
                  vt.WingSection(y_frac=1.0, chord=1.0, airfoil=af_cam)],
    )
    cases.append(build_lattice(vt.Aircraft(surfaces=[w_cam_lin]), n_chord=4))

    # Symmetric section shape (NACA 0012, zero camber offset).
    w_0012 = vt.LiftingSurface(
        name="w", semi_span=3.0, n_panels=4,
        sections=[vt.WingSection(y_frac=0.0, chord=1.0, shape="0012"),
                  vt.WingSection(y_frac=1.0, chord=1.0, shape="0012")],
    )
    cases.append(build_lattice(vt.Aircraft(surfaces=[w_0012]), n_chord=4))

    # Cambered shape with n_chord=1 (turned-normal fallback).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        w_4412 = vt.LiftingSurface(
            name="w", semi_span=3.0, n_panels=4,
            sections=[vt.WingSection(y_frac=0.0, chord=1.0, shape="4412"),
                      vt.WingSection(y_frac=1.0, chord=1.0, shape="4412")],
        )
        cases.append(build_lattice(vt.Aircraft(surfaces=[w_4412]), n_chord=1))

    # Section-mode flap (n_chord=1) and undeflected chordwise flap (n_chord=4).
    fl10 = vt.ControlSurface(name="flap", eta_start=0.0, eta_end=1.0, hinge_x_c=0.75, deflection=np.radians(10.0))
    fl0 = vt.ControlSurface(name="flap0", eta_start=0.0, eta_end=1.0, hinge_x_c=0.75, deflection=0.0)
    w_fl10 = vt.LiftingSurface(
        name="w", semi_span=3.0, n_panels=4, controls=[fl10],
        sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=1.0)],
    )
    w_fl0 = vt.LiftingSurface(
        name="w", semi_span=3.0, n_panels=4, controls=[fl0],
        sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=1.0)],
    )
    cases.append(build_lattice(vt.Aircraft(surfaces=[w_fl10]), n_chord=1))
    cases.append(build_lattice(vt.Aircraft(surfaces=[w_fl0]), n_chord=4))

    for lat in cases:
        rings = build_ring_lattice(lat)
        _assert_ring_topology(lat, rings, te_ext=0.25)


@pytest.mark.parametrize(
    "wing_kind,h,alpha_deg,beta_deg,phi_deg",
    [
        ("rect", 0.5, 5.0, 0.0, 0.0),
        ("swept", 0.8, 4.0, 3.0, 6.0),
    ],
)
def test_ground_image_mirror_parity(wing_kind, h, alpha_deg, beta_deg, phi_deg):
    """Ring lattice with GroundPlane matches an explicit two-lattice free-air mirror model to round-off."""
    ac = _planar_rect_wing(n_panels=6) if wing_kind == "rect" else _swept_tapered_dihedral_wing(n_panels=6)
    lat = build_lattice(ac, n_chord=4, chord_spacing="uniform")
    cond = vt.FlightCondition(
        V_inf=30.0,
        alpha=np.radians(alpha_deg),
        beta=np.radians(beta_deg),
        phi=np.radians(phi_deg),
        h=h,
    )
    ground = make_ground_plane(lat, h, cond.alpha, cond.beta, cond.phi, ref_point=ac.ref_point, height_ref="ref")
    wd = freestream_direction(cond.alpha, cond.beta)
    rings = build_ring_lattice(lat, wake_dir=wd)
    img = ground_image_rings(rings, ground)
    assert img.sign == -1.0

    # Solve N x N system with ground images.
    A_ge = _steady_ring_influence(rings, ground=ground, wake_dir=wd)
    umap = make_unknown_map(lat, False)
    rhs_real = _vlm_rhs(lat, umap, cond, ac.ref_point)
    g_ge = np.linalg.solve(A_ge, rhs_real)

    # Explicit 2N x 2N free-air two-lattice model: real rings + reflected rings, no ground kernel.
    n = rings.n_panels
    te_idx = rings.te_indices
    corners_2 = np.concatenate([rings.corners, img.corners], axis=0)
    cp_2 = np.concatenate([rings.cp, ground.reflect_points(rings.cp)], axis=0)
    n_2 = np.concatenate([rings.normal_bc, ground.reflect_directions(rings.normal_bc)], axis=0)
    rc_2 = np.concatenate([rings.rc, rings.rc], axis=0)

    A_2 = VR.ring_normal_influence(cp_2, n_2, corners_2, rc_2)
    te_l_2 = np.concatenate([rings.te_left, img.te_left], axis=0)
    te_r_2 = np.concatenate([rings.te_right, img.te_right], axis=0)
    wd_2 = np.concatenate([rings.wake_dir, img.wake_dir], axis=0)
    rc_te_2 = np.concatenate([rings.rc[te_idx], rings.rc[te_idx]], axis=0)
    te_idx_2 = np.concatenate([te_idx, te_idx + n])
    A_2[:, te_idx_2] += VR.steady_wake_normal_influence(cp_2, n_2, te_l_2, te_r_2, wd_2, rc_te_2)

    rhs_2 = -cond.V_inf * (n_2 @ wd)
    g_2 = np.linalg.solve(A_2, rhs_2)

    assert np.max(np.abs(g_2[:n] - g_ge)) <= 1e-12 * np.max(np.abs(g_ge))
    assert np.max(np.abs(g_2[n:] + g_ge)) <= 1e-12 * np.max(np.abs(g_ge))

    # Induced normal velocity on points lying on the ground plane is zero to round-off.
    P_ground = rings.cp - ground.height(rings.cp)[:, None] * ground.normal[None, :]
    n_ground = np.broadcast_to(ground.normal, P_ground.shape)
    v_n_real = (
        VR.ring_normal_influence(P_ground, n_ground, rings.corners, rings.rc) @ g_ge
        + VR.steady_wake_normal_influence(P_ground, n_ground, rings.te_left, rings.te_right,
                                          rings.wake_dir, rings.rc[te_idx]) @ g_ge[te_idx]
    )
    v_n_img = img.sign * (
        VR.ring_normal_influence(P_ground, n_ground, img.corners, rings.rc) @ g_ge
        + VR.steady_wake_normal_influence(P_ground, n_ground, img.te_left, img.te_right,
                                          img.wake_dir, rings.rc[te_idx]) @ g_ge[te_idx]
    )
    assert np.max(np.abs(v_n_real + v_n_img)) <= 1e-13 * np.max(np.abs(v_n_real))


@pytest.mark.parametrize("h_over_c", [None, 1.0, 0.5])
def test_steady_check_exact_equivalence_planar_ar6(monkeypatch, h_over_c):
    """Planar AR 6 wing (8 x 24): rings equal existing VLM to <= 1e-10 in free air and ground effect."""
    ac = _planar_rect_wing(semi_span=3.0, chord=1.0, n_panels=12)
    st = vt.SolverSettings(n_panels=12, n_chord=8, chord_spacing="uniform")
    lat = build_lattice(ac, st, collocation="vlm", n_chord=8, chord_spacing="uniform")
    assert lat.n_panels == 192

    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(5.0), h=h_over_c)
    ground = (
        make_ground_plane(lat, h_over_c, cond.alpha, cond.beta, ref_point=ac.ref_point, height_ref="ref")
        if h_over_c is not None
        else None
    )
    wd = freestream_direction(cond.alpha, cond.beta)

    # Exact-equivalence settings: zero core radius, TE closure at the trailing edge, same wake_dir.
    monkeypatch.setattr(lattice_mod, "CORE_RADIUS_FRACTION", 0.0)
    g_vlm, _ = solve_vlm(lat, cond, ground=ground, use_symmetry=False, wake_dir=wd, ref_point=ac.ref_point)
    _, g_ring_hs = _solve_steady_rings(
        lat, cond, ground=ground, wake_dir=wd, ref_point=ac.ref_point,
        te_extension_fraction=0.0, rc=0.0,
    )
    rel = np.max(np.abs(g_ring_hs - g_vlm)) / np.max(np.abs(g_vlm))
    assert rel <= 1e-10


def test_steady_check_exact_equivalence_body_wake_with_default_te_offset(monkeypatch):
    """In free air with body-axis wake along the untwisted chord and rc=0, 0.25 TE offset also matches VLM."""
    ac = _planar_rect_wing(semi_span=3.0, chord=1.0, n_panels=12)
    st = vt.SolverSettings(n_panels=12, n_chord=8, chord_spacing="uniform")
    lat = build_lattice(ac, st, collocation="vlm", n_chord=8, chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(5.0))
    wd = np.array([1.0, 0.0, 0.0])

    monkeypatch.setattr(lattice_mod, "CORE_RADIUS_FRACTION", 0.0)
    g_vlm, _ = solve_vlm(lat, cond, ground=None, use_symmetry=False, wake_dir=wd, ref_point=ac.ref_point)
    _, g_ring_hs = _solve_steady_rings(
        lat, cond, ground=None, wake_dir=wd, ref_point=ac.ref_point,
        te_extension_fraction=0.25, rc=0.0,
    )
    rel = np.max(np.abs(g_ring_hs - g_vlm)) / np.max(np.abs(g_vlm))
    assert rel <= 1e-10


@pytest.mark.parametrize("h_ground", [None, 1.0])
def test_steady_check_exact_equivalence_swept_tapered_dihedral(monkeypatch, h_ground):
    """Swept, tapered wing with dihedral and twist: rings equal existing VLM to <= 1e-10."""
    ac = _swept_tapered_dihedral_wing(n_panels=12)
    st = vt.SolverSettings(n_panels=12, n_chord=8, chord_spacing="uniform")
    lat = build_lattice(ac, st, collocation="vlm", n_chord=8, chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(5.0), beta=np.radians(3.0), h=h_ground)
    ground = (
        make_ground_plane(lat, h_ground, cond.alpha, cond.beta, ref_point=ac.ref_point, height_ref="ref")
        if h_ground is not None
        else None
    )
    wd = freestream_direction(cond.alpha, cond.beta)

    monkeypatch.setattr(lattice_mod, "CORE_RADIUS_FRACTION", 0.0)
    g_vlm, _ = solve_vlm(lat, cond, ground=ground, use_symmetry=False, wake_dir=wd, ref_point=ac.ref_point)
    _, g_ring_hs = _solve_steady_rings(
        lat, cond, ground=ground, wake_dir=wd, ref_point=ac.ref_point,
        te_extension_fraction=0.0, rc=0.0,
    )
    rel = np.max(np.abs(g_ring_hs - g_vlm)) / np.max(np.abs(g_vlm))
    assert rel <= 1e-10


def test_steady_check_single_chord_panel_with_default_core_and_cross_surface():
    """With n_chord=1 and TE closure at the TE, rings match VLM to <= 1e-10 even with default cores."""
    ac = _wing_and_tail()
    lat = build_lattice(ac, n_chord=1)
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(5.0))
    wd = freestream_direction(cond.alpha, cond.beta)

    g_vlm, _ = solve_vlm(lat, cond, ground=None, use_symmetry=False, wake_dir=wd)
    _, g_ring_hs = _solve_steady_rings(
        lat, cond, ground=None, wake_dir=wd, te_extension_fraction=0.0,
    )
    rel = np.max(np.abs(g_ring_hs - g_vlm)) / np.max(np.abs(g_vlm))
    assert rel <= 1e-10


def test_steady_check_default_settings_small_documented_difference():
    """At default settings (0.25 TE offset and Ventorum core), rings stay within 1e-3 of VLM."""
    ac = _planar_rect_wing(semi_span=3.0, chord=1.0, n_panels=12)
    st = vt.SolverSettings(n_panels=12, n_chord=8, chord_spacing="uniform")
    lat = build_lattice(ac, st, collocation="vlm", n_chord=8, chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(5.0))
    wd = freestream_direction(cond.alpha, cond.beta)

    g_vlm, _ = solve_vlm(lat, cond, ground=None, use_symmetry=False, wake_dir=wd, ref_point=ac.ref_point)
    _, g_ring_hs = _solve_steady_rings(lat, cond, ground=None, wake_dir=wd, ref_point=ac.ref_point)
    rel = np.max(np.abs(g_ring_hs - g_vlm)) / np.max(np.abs(g_vlm))
    assert 1e-6 < rel < 1e-3


def test_refuses_llt_collocation():
    """build_ring_lattice refuses lifting-line collocation with InputError."""
    ac = _planar_rect_wing(n_panels=4)
    lat = build_lattice(ac, collocation="llt", n_chord=1)
    with pytest.raises(InputError, match="collocation"):
        build_ring_lattice(lat)


def _cambered_naca4412_wing(semi_span: float = 3.0, chord: float = 1.0, n_panels: int = 8) -> vt.Aircraft:
    """Build a rectangular NACA 4412 cambered wing with quarter-chord moment reference."""
    wing = vt.LiftingSurface(
        name="cambered",
        semi_span=semi_span,
        n_panels=n_panels,
        sections=[
            vt.WingSection(y_frac=0.0, chord=chord, shape="4412"),
            vt.WingSection(y_frac=1.0, chord=chord, shape="4412"),
        ],
    )
    return vt.Aircraft(surfaces=[wing], ref_point=np.array([0.25 * chord, 0.0, 0.0]))


def _flapped_wing(
    deflection_deg: float = 20.0,
    semi_span: float = 3.0,
    chord: float = 1.0,
    n_panels: int = 8,
    eta_start: float = 0.0,
    eta_end: float = 1.0,
    hinge_x_c: float = 0.75,
    shape: str | None = None,
) -> vt.Aircraft:
    """Build a rectangular wing with a chordwise deflected flap."""
    fl = vt.ControlSurface(
        name="flap",
        eta_start=eta_start,
        eta_end=eta_end,
        hinge_x_c=hinge_x_c,
        deflection=np.radians(deflection_deg),
    )
    wing = vt.LiftingSurface(
        name="flapped",
        semi_span=semi_span,
        n_panels=n_panels,
        controls=[fl],
        sections=[
            vt.WingSection(y_frac=0.0, chord=chord, shape=shape),
            vt.WingSection(y_frac=1.0, chord=chord, shape=shape),
        ],
    )
    return vt.Aircraft(surfaces=[wing], ref_point=np.array([0.25 * chord, 0.0, 0.0]))


def _wing_with_winglet(n_wing: int = 6, n_winglet: int = 3, *, symmetric_winglets: bool = False) -> vt.Aircraft:
    """Build a rectangular wing with a vertical winglet joined at the tip."""
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
    surfaces = [wing, winglet, winglet.mirrored()] if symmetric_winglets else [wing, winglet]
    return vt.Aircraft(surfaces=surfaces, ref_point=np.array([0.25, 0.0, 0.0]))


def test_build_ring_lattice_camber_leg_chain():
    """Cambered mean-line leg chains map to rings with intermediate camber side nodes."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ac = _cambered_naca4412_wing(n_panels=4)
        lat = build_lattice(ac, n_chord=4, chord_spacing="uniform")

    rings_te0 = build_ring_lattice(lat, te_extension_fraction=0.0)
    _assert_ring_topology(lat, rings_te0, te_ext=0.0)
    assert rings_te0.has_side_nodes
    assert rings_te0.side_nodes is not None and rings_te0.side_counts is not None
    assert rings_te0.side_nodes.shape == (lat.n_panels, 2, 1, 3)
    assert np.all(rings_te0.side_counts == 1)
    assert np.allclose(rings_te0.side_nodes[:, 0, 0, :], lat.b_kink, atol=1e-15)
    assert np.allclose(rings_te0.side_nodes[:, 1, 0, :], lat.a_kink, atol=1e-15)

    # With default TE extension (0.25), TE rings carry both the camber node and the physical TE node.
    rings_def = build_ring_lattice(lat, te_extension_fraction=0.25)
    _assert_ring_topology(lat, rings_def, te_ext=0.25)
    assert rings_def.side_nodes is not None and rings_def.side_counts is not None
    assert rings_def.side_nodes.shape == (lat.n_panels, 2, 2, 3)
    interior = ~rings_def.te_mask
    te_idx = rings_def.te_indices
    assert np.all(rings_def.side_counts[interior] == 1)
    assert np.all(rings_def.side_counts[te_idx] == 2)
    assert np.allclose(rings_def.side_nodes[te_idx, 0, 0, :], lat.b_kink[te_idx], atol=1e-15)
    assert np.allclose(rings_def.side_nodes[te_idx, 0, 1, :], lat.b_te[te_idx], atol=1e-15)
    assert np.allclose(rings_def.side_nodes[te_idx, 1, 0, :], lat.a_te[te_idx], atol=1e-15)
    assert np.allclose(rings_def.side_nodes[te_idx, 1, 1, :], lat.a_kink[te_idx], atol=1e-15)

    # Adjacent strips share their bent side polylines identically.
    for s in range(lat.n_strips - 1):
        for k in range(lat.n_chord):
            p_l = s * lat.n_chord + k
            p_r = (s + 1) * lat.n_chord + k
            cr = int(rings_def.side_counts[p_l, 0])
            cl = int(rings_def.side_counts[p_r, 1])
            assert cr == cl
            assert np.allclose(rings_def.side_nodes[p_l, 0, :cr], rings_def.side_nodes[p_r, 1, :cl][::-1], atol=1e-13)


def test_build_ring_lattice_deflected_chordwise_flap():
    """Deflected chordwise flaps map to rings with hinge nodes on the hinge-crossing rings."""
    ac = _flapped_wing(deflection_deg=20.0, n_panels=4)
    lat = build_lattice(ac, n_chord=4, chord_spacing="uniform")
    rings = build_ring_lattice(lat, te_extension_fraction=0.25)
    _assert_ring_topology(lat, rings, te_ext=0.25)
    assert rings.has_side_nodes
    assert rings.side_nodes is not None and rings.side_counts is not None

    # Hinge is at x/c = 0.75 (between panels k=2 and k=3): only k=2 rings carry hinge side nodes.
    k_idx = rings.panel_chord_index
    assert np.all(rings.side_counts[k_idx == 2] == 1)
    assert np.all(rings.side_counts[k_idx != 2] == 0)
    hinge_panels = np.flatnonzero(k_idx == 2)
    assert np.allclose(rings.side_nodes[hinge_panels, 0, 0, :], lat.b_kink[hinge_panels], atol=1e-15)
    assert np.allclose(rings.side_nodes[hinge_panels, 1, 0, :], lat.a_kink[hinge_panels], atol=1e-15)

    # Partial-span flap on a cambered wing also maps cleanly.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ac_cf = _flapped_wing(deflection_deg=15.0, n_panels=6, eta_start=0.0, eta_end=0.5, hinge_x_c=0.7, shape="4412")
        lat_cf = build_lattice(ac_cf, n_chord=4, chord_spacing="uniform")
    rings_cf = build_ring_lattice(lat_cf, te_extension_fraction=0.25)
    assert rings_cf.has_side_nodes
    assert rings_cf.n_panels == lat_cf.n_panels


def test_build_ring_lattice_surface_junction():
    """Joined surfaces (wing + winglet and T-tail) map with continuous TE closure and shared core group."""
    ac = _wing_with_winglet(n_wing=6, n_winglet=3, symmetric_winglets=True)
    lat = build_lattice(ac, n_chord=4, chord_spacing="uniform")
    rings = build_ring_lattice(lat, te_extension_fraction=0.25)
    _assert_ring_topology(lat, rings, te_ext=0.25)
    assert not rings.has_side_nodes
    assert np.all(rings.strip_core_group == rings.strip_core_group[0])

    # Right wing tip (strip 11) joins right winglet root (strip 12); left wing tip (0) joins left winglet (17).
    s_w_r = lat.surfaces[0].strips.stop - 1
    s_wl_r = lat.surfaces[1].strips.start
    s_w_l = lat.surfaces[0].strips.start
    s_wl_l = lat.surfaces[2].strips.stop - 1
    assert np.allclose(rings.te_right[s_w_r], rings.te_left[s_wl_r], atol=1e-13)
    assert np.allclose(rings.te_left[s_w_l], rings.te_right[s_wl_l], atol=1e-13)
    for k in range(lat.n_chord):
        assert np.allclose(rings.corners[s_w_r * lat.n_chord + k, 1], rings.corners[s_wl_r * lat.n_chord + k, 0],
                           atol=1e-13)
        assert np.allclose(rings.corners[s_w_r * lat.n_chord + k, 2], rings.corners[s_wl_r * lat.n_chord + k, 3],
                           atol=1e-13)

    # T-tail three-way junction: vertical fin top edge meets horizontal tail root edge.
    fin = vt.LiftingSurface(
        name="fin", semi_span=1.0, n_panels=3, is_symmetric=False,
        dihedral=np.radians(90.0), position=np.array([0.0, 0.0, 0.0]),
        sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=0.8)],
    )
    htail = vt.LiftingSurface(
        name="htail", semi_span=1.2, n_panels=4, position=np.array([0.0, 0.0, 1.0]),
        sections=[vt.WingSection(y_frac=0.0, chord=0.8), vt.WingSection(y_frac=1.0, chord=0.6)],
    )
    lat_t = build_lattice(vt.Aircraft(surfaces=[fin, htail]), n_chord=4, chord_spacing="uniform")
    rings_t = build_ring_lattice(lat_t, te_extension_fraction=0.25)
    _assert_ring_topology(lat_t, rings_t, te_ext=0.25)
    assert np.all(rings_t.strip_core_group == rings_t.strip_core_group[0])
    s_fin_top = lat_t.surfaces[0].strips.stop - 1
    s_ht_l = lat_t.surfaces[1].strips.start + 3
    s_ht_r = lat_t.surfaces[1].strips.start + 4
    assert np.allclose(rings_t.te_right[s_fin_top], rings_t.te_right[s_ht_l], atol=1e-13)
    assert np.allclose(rings_t.te_right[s_fin_top], rings_t.te_left[s_ht_r], atol=1e-13)


@pytest.mark.parametrize("h_over_c", [None, 0.5])
def test_steady_check_exact_equivalence_cambered_naca4412(monkeypatch, h_over_c):
    """Cambered NACA 4412 wing (n_chord = 8): rings equal VLM to <= 1e-10 in free air and at h/c = 0.5."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ac = _cambered_naca4412_wing(semi_span=3.0, chord=1.0, n_panels=8)
        st = vt.SolverSettings(n_panels=8, n_chord=8, chord_spacing="uniform")
        lat = build_lattice(ac, st, collocation="vlm", n_chord=8, chord_spacing="uniform")

    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), h=h_over_c)
    ground = (
        make_ground_plane(lat, h_over_c, cond.alpha, cond.beta, ref_point=ac.ref_point, height_ref="ref")
        if h_over_c is not None
        else None
    )
    wd = freestream_direction(cond.alpha, cond.beta)

    monkeypatch.setattr(lattice_mod, "CORE_RADIUS_FRACTION", 0.0)
    g_vlm, _ = solve_vlm(lat, cond, ground=ground, use_symmetry=False, wake_dir=wd, ref_point=ac.ref_point)
    _, g_ring_hs = _solve_steady_rings(
        lat, cond, ground=ground, wake_dir=wd, ref_point=ac.ref_point,
        te_extension_fraction=0.0, rc=0.0,
    )
    rel = np.max(np.abs(g_ring_hs - g_vlm)) / np.max(np.abs(g_vlm))
    assert rel <= 1e-10


@pytest.mark.parametrize("h_over_c", [None, 0.5])
def test_steady_check_exact_equivalence_deflected_flap(monkeypatch, h_over_c):
    """Wing with 20 deg deflected chordwise flap (n_chord = 8): rings equal VLM to <= 1e-10."""
    ac = _flapped_wing(deflection_deg=20.0, semi_span=3.0, chord=1.0, n_panels=8)
    st = vt.SolverSettings(n_panels=8, n_chord=8, chord_spacing="uniform")
    lat = build_lattice(ac, st, collocation="vlm", n_chord=8, chord_spacing="uniform")

    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), h=h_over_c)
    ground = (
        make_ground_plane(lat, h_over_c, cond.alpha, cond.beta, ref_point=ac.ref_point, height_ref="ref")
        if h_over_c is not None
        else None
    )
    wd = freestream_direction(cond.alpha, cond.beta)

    monkeypatch.setattr(lattice_mod, "CORE_RADIUS_FRACTION", 0.0)
    g_vlm, _ = solve_vlm(lat, cond, ground=ground, use_symmetry=False, wake_dir=wd, ref_point=ac.ref_point)
    _, g_ring_hs = _solve_steady_rings(
        lat, cond, ground=ground, wake_dir=wd, ref_point=ac.ref_point,
        te_extension_fraction=0.0, rc=0.0,
    )
    rel = np.max(np.abs(g_ring_hs - g_vlm)) / np.max(np.abs(g_vlm))
    assert rel <= 1e-10


def test_steady_check_exact_equivalence_cambered_with_flap(monkeypatch):
    """Cambered NACA 4412 wing with 20 deg deflected flap (n_chord = 8): rings equal VLM to <= 1e-10."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ac = _flapped_wing(deflection_deg=20.0, semi_span=3.0, chord=1.0, n_panels=6, shape="4412")
        st = vt.SolverSettings(n_panels=6, n_chord=8, chord_spacing="uniform")
        lat = build_lattice(ac, st, collocation="vlm", n_chord=8, chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    wd = freestream_direction(cond.alpha, cond.beta)

    monkeypatch.setattr(lattice_mod, "CORE_RADIUS_FRACTION", 0.0)
    g_vlm, _ = solve_vlm(lat, cond, ground=None, use_symmetry=False, wake_dir=wd, ref_point=ac.ref_point)
    _, g_ring_hs = _solve_steady_rings(
        lat, cond, ground=None, wake_dir=wd, ref_point=ac.ref_point,
        te_extension_fraction=0.0, rc=0.0,
    )
    rel = np.max(np.abs(g_ring_hs - g_vlm)) / np.max(np.abs(g_vlm))
    assert rel <= 1e-10


@pytest.mark.parametrize("h_over_c", [None, 0.5])
def test_steady_check_exact_equivalence_wing_with_winglet(monkeypatch, h_over_c):
    """Wing with winglet joined at the tip (n_chord = 8): rings equal VLM to <= 1e-10."""
    ac = _wing_with_winglet(n_wing=6, n_winglet=3)
    st = vt.SolverSettings(n_panels=6, n_chord=8, chord_spacing="uniform")
    lat = build_lattice(ac, st, collocation="vlm", n_chord=8, chord_spacing="uniform")

    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=np.radians(2.0), h=h_over_c)
    ground = (
        make_ground_plane(lat, h_over_c, cond.alpha, cond.beta, ref_point=ac.ref_point, height_ref="ref")
        if h_over_c is not None
        else None
    )
    wd = freestream_direction(cond.alpha, cond.beta)

    monkeypatch.setattr(lattice_mod, "CORE_RADIUS_FRACTION", 0.0)
    g_vlm, _ = solve_vlm(lat, cond, ground=ground, use_symmetry=False, wake_dir=wd, ref_point=ac.ref_point)
    _, g_ring_hs = _solve_steady_rings(
        lat, cond, ground=ground, wake_dir=wd, ref_point=ac.ref_point,
        te_extension_fraction=0.0, rc=0.0,
    )
    rel = np.max(np.abs(g_ring_hs - g_vlm)) / np.max(np.abs(g_vlm))
    assert rel <= 1e-10


@pytest.mark.parametrize("case_kind", ["camber", "flap", "winglet"])
def test_ground_image_mirror_parity_bent_and_junction(case_kind):
    """Bent rings and joined surfaces match an explicit two-lattice free-air mirror model to round-off."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if case_kind == "camber":
            ac = _cambered_naca4412_wing(n_panels=4)
        elif case_kind == "flap":
            ac = _flapped_wing(deflection_deg=20.0, n_panels=4)
        else:
            ac = _wing_with_winglet(n_wing=4, n_winglet=2)
        lat = build_lattice(ac, n_chord=4, chord_spacing="uniform")

    h = 0.5
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=np.radians(2.0), h=h)
    ground = make_ground_plane(lat, h, cond.alpha, cond.beta, ref_point=ac.ref_point, height_ref="ref")
    wd = freestream_direction(cond.alpha, cond.beta)
    rings = build_ring_lattice(lat, wake_dir=wd)
    img = ground_image_rings(rings, ground)

    A_ge = _steady_ring_influence(rings, ground=ground, wake_dir=wd)
    umap = make_unknown_map(lat, False)
    rhs_real = _vlm_rhs(lat, umap, cond, ac.ref_point)
    g_ge = np.linalg.solve(A_ge, rhs_real)

    n = rings.n_panels
    te_idx = rings.te_indices
    corners_2 = np.concatenate([rings.corners, img.corners], axis=0)
    sn_2 = None if rings.side_nodes is None else np.concatenate([rings.side_nodes, img.side_nodes], axis=0)
    sc_2 = None if rings.side_counts is None else np.concatenate([rings.side_counts, img.side_counts], axis=0)
    cp_2 = np.concatenate([rings.cp, ground.reflect_points(rings.cp)], axis=0)
    n_2 = np.concatenate([rings.normal_bc, ground.reflect_directions(rings.normal_bc)], axis=0)
    rc_2 = np.concatenate([rings.rc, rings.rc], axis=0)

    A_2 = VR.ring_normal_influence(cp_2, n_2, corners_2, rc_2, side_nodes=sn_2, side_counts=sc_2)
    te_l_2 = np.concatenate([rings.te_left, img.te_left], axis=0)
    te_r_2 = np.concatenate([rings.te_right, img.te_right], axis=0)
    wd_2 = np.concatenate([rings.wake_dir, img.wake_dir], axis=0)
    rc_te_2 = np.concatenate([rings.rc[te_idx], rings.rc[te_idx]], axis=0)
    te_idx_2 = np.concatenate([te_idx, te_idx + n])
    A_2[:, te_idx_2] += VR.steady_wake_normal_influence(cp_2, n_2, te_l_2, te_r_2, wd_2, rc_te_2)

    rhs_2 = -cond.V_inf * (n_2 @ wd)
    g_2 = np.linalg.solve(A_2, rhs_2)

    assert np.max(np.abs(g_2[:n] - g_ge)) <= 1e-12 * np.max(np.abs(g_ge))
    assert np.max(np.abs(g_2[n:] + g_ge)) <= 1e-12 * np.max(np.abs(g_ge))


def test_steady_check_default_settings_camber_flap_winglet():
    """At default settings, cambered chains match VLM to round-off when te_ext=0 and within 3e-3 at te_ext=0.25."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ac_cam = _cambered_naca4412_wing(semi_span=3.0, chord=1.0, n_panels=8)
        st8 = vt.SolverSettings(n_panels=8, n_chord=8, chord_spacing="uniform")
        lat_cam = build_lattice(ac_cam, st8, collocation="vlm", n_chord=8, chord_spacing="uniform")

    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    wd = freestream_direction(cond.alpha, cond.beta)

    # On a cambered leg-chain lattice, even with default nonzero core, te_ext=0 matches VLM to round-off.
    g_vlm_cam, _ = solve_vlm(lat_cam, cond, ground=None, use_symmetry=False, wake_dir=wd, ref_point=ac_cam.ref_point)
    _, g_cam_te0 = _solve_steady_rings(
        lat_cam, cond, ground=None, wake_dir=wd, ref_point=ac_cam.ref_point, te_extension_fraction=0.0,
    )
    _, g_cam_def = _solve_steady_rings(
        lat_cam, cond, ground=None, wake_dir=wd, ref_point=ac_cam.ref_point,
    )
    assert np.max(np.abs(g_cam_te0 - g_vlm_cam)) / np.max(np.abs(g_vlm_cam)) <= 1e-12
    assert np.max(np.abs(g_cam_def - g_vlm_cam)) / np.max(np.abs(g_vlm_cam)) <= 1e-3

    # Deflected 20 deg flap and wing + winglet stay within 3e-3 at default settings.
    ac_fl = _flapped_wing(deflection_deg=20.0, semi_span=3.0, chord=1.0, n_panels=8)
    lat_fl = build_lattice(ac_fl, st8, collocation="vlm", n_chord=8, chord_spacing="uniform")
    g_vlm_fl, _ = solve_vlm(lat_fl, cond, ground=None, use_symmetry=False, wake_dir=wd, ref_point=ac_fl.ref_point)
    _, g_fl_def = _solve_steady_rings(lat_fl, cond, ground=None, wake_dir=wd, ref_point=ac_fl.ref_point)
    assert np.max(np.abs(g_fl_def - g_vlm_fl)) / np.max(np.abs(g_vlm_fl)) <= 3e-3

    ac_wl = _wing_with_winglet(n_wing=6, n_winglet=3)
    st6 = vt.SolverSettings(n_panels=6, n_chord=8, chord_spacing="uniform")
    lat_wl = build_lattice(ac_wl, st6, collocation="vlm", n_chord=8, chord_spacing="uniform")
    cond_wl = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=np.radians(2.0))
    wd_wl = freestream_direction(cond_wl.alpha, cond_wl.beta)
    g_vlm_wl, _ = solve_vlm(lat_wl, cond_wl, ground=None, use_symmetry=False, wake_dir=wd_wl, ref_point=ac_wl.ref_point)
    _, g_wl_def = _solve_steady_rings(lat_wl, cond_wl, ground=None, wake_dir=wd_wl, ref_point=ac_wl.ref_point)
    assert np.max(np.abs(g_wl_def - g_vlm_wl)) / np.max(np.abs(g_vlm_wl)) <= 3e-3


def test_refuses_invalid_ring_parameters():
    """Negative TE extension or zero wake direction raises InputError."""
    lat = build_lattice(_planar_rect_wing(n_panels=4), n_chord=2)
    with pytest.raises(InputError, match="te_extension_fraction"):
        build_ring_lattice(lat, te_extension_fraction=-0.1)
    with pytest.raises(InputError, match="wake_dir"):
        build_ring_lattice(lat, wake_dir=np.zeros(3))
