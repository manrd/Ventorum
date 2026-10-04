# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for vortex lattice geometry transformation utilities.

Verifies 3D rotation matrix construction, lattice rigid-body transformation,
preservation of input objects, symmetry checks, and force invariance in own axes.
"""

from __future__ import annotations

import numpy as np
import pytest

import ventorum as vt
from ventorum.aero.system import (
    freestream_direction,
    lift_direction,
    side_direction,
)
from ventorum.geometry.transform import (
    rotation_matrix,
    transform_lattice,
)


def _wing_and_tail():
    wing = vt.LiftingSurface(
        name="wing",
        semi_span=4.0,
        n_panels=8,
        dihedral=np.radians(5.0),
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.2),
            vt.WingSection(y_frac=1.0, chord=0.6),
        ],
    )
    tail = vt.LiftingSurface(
        name="tail",
        semi_span=1.2,
        n_panels=4,
        position=np.array([3.5, 0.0, 0.2]),
        sections=[
            vt.WingSection(y_frac=0.0, chord=0.5),
            vt.WingSection(y_frac=1.0, chord=0.4),
        ],
    )
    return vt.Aircraft(surfaces=[wing, tail])


def test_rotation_order():
    """Verify rotation order (yaw, then pitch, then roll) and sign conventions."""
    roll = np.radians(12.0)
    pitch = np.radians(7.0)
    yaw = np.radians(-15.0)

    R = rotation_matrix(roll=roll, pitch=pitch, yaw=yaw)

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cr, sr],
        [0.0, -sr, cr],
    ])
    Ry = np.array([
        [cp, 0.0, sp],
        [0.0, 1.0, 0.0],
        [-sp, 0.0, cp],
    ])
    Rz = np.array([
        [cy, sy, 0.0],
        [-sy, cy, 0.0],
        [0.0, 0.0, 1.0],
    ])

    R_expected = Rx @ Ry @ Rz
    assert np.allclose(R, R_expected, atol=1e-15)

    # Positive pitch (nose up) moves a point on the +x axis (aft) down (-z)
    p_x = np.array([1.0, 0.0, 0.0])
    p_pitched = rotation_matrix(pitch=np.radians(10.0)) @ p_x
    assert p_pitched[2] < 0.0
    assert p_pitched[0] > 0.0

    # Positive roll moves a point on the right wing (+y) down (-z)
    p_y = np.array([0.0, 1.0, 0.0])
    p_rolled = rotation_matrix(roll=np.radians(10.0)) @ p_y
    assert p_rolled[2] < 0.0
    assert p_rolled[1] > 0.0

    # Positive yaw (nose right) moves a point on the +x axis (aft) left (-y)
    p_yawed = rotation_matrix(yaw=np.radians(10.0)) @ p_x
    assert p_yawed[1] < 0.0
    assert p_yawed[0] > 0.0


def test_input_not_changed():
    """Verify input lattice is strictly unmodified after transform_lattice."""
    ac = _wing_and_tail()
    solver = vt.HorseshoeSolver()
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    st = vt.SolverSettings(n_panels=8, n_chord=3)
    lattice = solver.build(ac, st, cond)

    array_attr_names = [
        "a", "b", "a_te", "b_te", "cp", "normal_bc", "panel_strip",
        "panel_mirror", "panel_length", "strip_surface", "le_left",
        "le_right", "te_left", "te_right", "qc_mid", "dl", "chord",
        "twist", "width", "area", "chord_dir", "normal", "normal_bc_strip",
        "a0", "alpha_L0", "Cd0", "Cm0", "strip_mirror", "strip_is_right",
        "eta", "cp_frac",
    ]
    orig_arrays = {attr: getattr(lattice, attr).copy() for attr in array_attr_names}
    orig_surface_arrays = [
        (s.edge_le.copy(), s.edge_te.copy(), s.edge_chord.copy(), s.edge_twist.copy())
        for s in lattice.surfaces
    ]

    R = rotation_matrix(roll=np.radians(10.0), pitch=np.radians(5.0), yaw=np.radians(-7.0))
    about = np.array([1.0, 0.5, 0.2])
    trans = np.array([0.3, -0.2, 0.1])

    new_lattice = transform_lattice(lattice, rotation=R, about=about, translation=trans)
    assert new_lattice is not lattice

    for attr in array_attr_names:
        current = getattr(lattice, attr)
        original = orig_arrays[attr]
        assert np.array_equal(current, original), f"Input lattice array {attr} was modified"

    for s, (orig_le, orig_te, orig_c, orig_tw) in zip(lattice.surfaces, orig_surface_arrays):
        assert np.array_equal(s.edge_le, orig_le)
        assert np.array_equal(s.edge_te, orig_te)
        assert np.array_equal(s.edge_chord, orig_c)
        assert np.array_equal(s.edge_twist, orig_tw)


def test_refuse_non_rotation():
    """Verify non-orthonormal and reflection matrices raise ValueError with informative messages."""
    ac = _wing_and_tail()
    solver = vt.HorseshoeSolver()
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    lattice = solver.build(ac, vt.SolverSettings(n_panels=8, n_chord=2), cond)

    # Reflection matrix (det = -1)
    refl = np.diag([1.0, 1.0, -1.0])
    with pytest.raises(ValueError, match="determinant"):
        transform_lattice(lattice, rotation=refl)

    # Scaled matrix (not orthonormal)
    scaled = 2.0 * np.eye(3)
    with pytest.raises(ValueError, match="orthonormal"):
        transform_lattice(lattice, rotation=scaled)

    # Non-orthogonal matrix
    sheared = np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(ValueError):
        transform_lattice(lattice, rotation=sheared)

    # Wrong shape
    with pytest.raises(ValueError):
        transform_lattice(lattice, rotation=np.eye(4))


def test_same_force_in_own_axes():
    """Verify force and moment vectors transform identically under 3D rigid transform."""
    ac = _wing_and_tail()
    ac.compute_reference_values()
    settings = vt.SolverSettings(n_panels=8, n_chord=3)
    solver = vt.HorseshoeSolver()

    alpha_orig = np.radians(4.0)
    beta_orig = 0.0
    V_inf = 30.0
    cond_orig = vt.FlightCondition(V_inf=V_inf, alpha=alpha_orig, beta=beta_orig)

    rp_orig = ac.moment_reference()
    lat_orig = solver.build(ac, settings, cond_orig, ref_point=rp_orig)

    roll = np.radians(10.0)
    pitch = np.radians(5.0)
    yaw = np.radians(-7.0)
    about = np.array([1.0, 0.5, 0.2])
    translation = np.array([0.3, -0.2, 0.1])
    R = rotation_matrix(roll=roll, pitch=pitch, yaw=yaw)

    lat_moved = transform_lattice(lat_orig, rotation=R, about=about, translation=translation)

    v_orig = freestream_direction(alpha_orig, beta_orig)
    v_moved = R @ v_orig

    beta_moved = float(np.arcsin(-v_moved[1]))
    alpha_moved = float(np.arctan2(v_moved[2], v_moved[0]))
    cond_moved = vt.FlightCondition(V_inf=V_inf, alpha=alpha_moved, beta=beta_moved)

    rp_moved = R @ (rp_orig - about) + about + translation

    res_orig = solver.solve_lattice(
        lat_orig, cond_orig, settings,
        ac.S_ref, ac.b_ref, ac.c_ref,
        ref_point=rp_orig, main_surface=ac.main_surface_index(),
    )
    res_moved = solver.solve_lattice(
        lat_moved, cond_moved, settings,
        ac.S_ref, ac.b_ref, ac.c_ref,
        ref_point=rp_moved, main_surface=ac.main_surface_index(),
    )

    qS_orig = 0.5 * cond_orig.rho * (V_inf ** 2) * ac.S_ref
    qS_moved = 0.5 * cond_moved.rho * (V_inf ** 2) * ac.S_ref

    d_orig = freestream_direction(cond_orig.alpha, cond_orig.beta)
    l_orig = lift_direction(cond_orig.alpha)
    s_orig = side_direction(cond_orig.alpha, cond_orig.beta)
    F_orig = qS_orig * (res_orig.totals.CDi * d_orig + res_orig.totals.CL * l_orig + res_orig.totals.CY * s_orig)

    d_moved = freestream_direction(cond_moved.alpha, cond_moved.beta)
    l_moved = lift_direction(cond_moved.alpha)
    s_moved = side_direction(cond_moved.alpha, cond_moved.beta)
    F_moved = qS_moved * (res_moved.totals.CDi * d_moved + res_moved.totals.CL * l_moved + res_moved.totals.CY * s_moved)

    assert np.linalg.norm(F_moved - R @ F_orig) <= 1e-10 * np.linalg.norm(F_orig)

    M_orig = qS_orig * np.array([
        -res_orig.totals.Cl * ac.b_ref,
        res_orig.totals.Cm * ac.c_ref,
        -res_orig.totals.Cn * ac.b_ref,
    ])
    M_moved = qS_moved * np.array([
        -res_moved.totals.Cl * ac.b_ref,
        res_moved.totals.Cm * ac.c_ref,
        -res_moved.totals.Cn * ac.b_ref,
    ])

    assert np.linalg.norm(M_moved - R @ M_orig) <= 1e-10 * np.linalg.norm(M_orig)


def test_pitch_keeps_symmetry():
    """Verify pure pitch preserves mirror symmetry maps and yields identical CL."""
    ac = _wing_and_tail()
    ac.compute_reference_values()
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(5.0))
    solver = vt.HorseshoeSolver()

    lat = solver.build(ac, vt.SolverSettings(n_panels=8, n_chord=3), cond)
    assert lat.can_fold_symmetry() is True

    R = rotation_matrix(pitch=np.radians(6.0))
    lat_pitched = transform_lattice(lat, rotation=R)

    assert lat_pitched.can_fold_symmetry() is True
    assert np.all(lat_pitched.panel_mirror >= 0)
    assert np.all(lat_pitched.strip_mirror >= 0)

    st_sym = vt.SolverSettings(n_panels=8, n_chord=3, use_symmetry=True)
    res_sym = solver.solve_lattice(
        lat_pitched, cond, st_sym,
        ac.S_ref, ac.b_ref, ac.c_ref,
        main_surface=ac.main_surface_index(),
    )

    st_full = vt.SolverSettings(n_panels=8, n_chord=3, use_symmetry=False)
    res_full = solver.solve_lattice(
        lat_pitched, cond, st_full,
        ac.S_ref, ac.b_ref, ac.c_ref,
        main_surface=ac.main_surface_index(),
    )

    assert abs(res_sym.totals.CL - res_full.totals.CL) < 1e-12


def test_roll_breaks_symmetry():
    """Verify non-pitch rotations or y-translations deactivate symmetry folding."""
    ac = _wing_and_tail()
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(5.0))
    solver = vt.HorseshoeSolver()
    lat = solver.build(ac, vt.SolverSettings(n_panels=8, n_chord=3), cond)
    assert lat.can_fold_symmetry() is True

    # Roll breaks symmetry
    R_roll = rotation_matrix(roll=np.radians(5.0))
    lat_rolled = transform_lattice(lat, rotation=R_roll)
    assert lat_rolled.can_fold_symmetry() is False
    assert np.all(lat_rolled.panel_mirror == -1)
    assert np.all(lat_rolled.strip_mirror == -1)

    # Yaw breaks symmetry
    R_yaw = rotation_matrix(yaw=np.radians(5.0))
    lat_yawed = transform_lattice(lat, rotation=R_yaw)
    assert lat_yawed.can_fold_symmetry() is False
    assert np.all(lat_yawed.panel_mirror == -1)
    assert np.all(lat_yawed.strip_mirror == -1)

    # Translation with y-component breaks symmetry
    lat_trans = transform_lattice(lat, translation=np.array([0.0, 0.5, 0.0]))
    assert lat_trans.can_fold_symmetry() is False
    assert np.all(lat_trans.panel_mirror == -1)
    assert np.all(lat_trans.strip_mirror == -1)


def test_signs_match_the_flow_angles():
    """Nose-up pitch equals a positive alpha, and nose-right yaw equals a negative beta.

    Reviewer test (T-0031): the signs are anchored to the flow angles of the
    solver, not only to the matrix product. A flow at alpha 0, beta 0 seen
    in the axes of the rotated geometry is R^T times (1, 0, 0).
    """
    from ventorum.aero.system import freestream_direction

    t = np.radians(6.0)
    level = np.array([1.0, 0.0, 0.0])
    R = rotation_matrix(pitch=t)
    assert np.allclose(R.T @ level, freestream_direction(t, 0.0), atol=1e-15)
    R = rotation_matrix(yaw=t)
    assert np.allclose(R.T @ level, freestream_direction(0.0, -t), atol=1e-15)


def test_pitched_lattice_equals_alpha():
    """A wing pitched nose up by 4 deg at alpha 0 gives the CL of the original wing at alpha 4 deg."""
    import ventorum as vt
    from ventorum.solvers.factory import make_solver

    wing = vt.LiftingSurface(name="wing", semi_span=3.0, n_panels=12,
                              sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=0.6)])
    ac = vt.Aircraft(surfaces=[wing])
    ac.compute_reference_values()
    settings = vt.SolverSettings(solver_type="vlm", wake_alignment="freestream")
    solver = make_solver("vlm")
    ref = ac.moment_reference()
    a = np.radians(4.0)
    base = solver.build(ac, settings, vt.FlightCondition(V_inf=30.0, alpha=0.0), None, ref)
    pitched = transform_lattice(base, rotation=rotation_matrix(pitch=a))
    args = (settings, float(ac.S_ref), float(ac.b_ref), float(ac.c_ref))
    cl_alpha = solver.solve_lattice(base, vt.FlightCondition(V_inf=30.0, alpha=a), *args,
                                    ref_point=ref).totals.CL
    cl_pitch = solver.solve_lattice(pitched, vt.FlightCondition(V_inf=30.0, alpha=0.0), *args,
                                    ref_point=ref).totals.CL
    assert cl_alpha > 0.0
    assert abs(cl_pitch - cl_alpha) <= 1e-10 * abs(cl_alpha)
