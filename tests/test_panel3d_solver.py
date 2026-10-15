# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for the 3D source-doublet panel solver (Ventorum-3DP).

Published sources
-----------------
* Morino, L. and Kuo, C. C., 1974, Subsonic potential aerodynamics for
  complex configurations: A general theory, AIAA Journal, Vol. 12, No. 2,
  pp. 191-197.
* Katz, J. and Plotkin, A., 2001, Low-Speed Aerodynamics, 2nd ed.,
  Cambridge University Press, Chapters 9 and 12.
* Maskew, B., 1987, Program VSAERO Theory Document, NASA CR-4023.
"""

from __future__ import annotations

import numpy as np

from ventorum.core.datatypes import Aircraft, FlightCondition, LiftingSurface, SolverSettings, WingSection
from ventorum.panel3d.mesh import PanelMesh, mesh_lifting_surface
from ventorum.panel3d.solver import PanelResult, PanelSolver
from ventorum.reference.analytic3d import (
    ellipsoid_quad_mesh,
    ellipsoid_surface_velocity,
    sphere_cp,
    sphere_quad_mesh,
)
from ventorum.solvers.horseshoe import VortexLatticeSolver


def _naca0012_coordinates(n: int = 401) -> np.ndarray:
    """Selig ordered unit-chord coordinates of a NACA 0012 section."""
    beta = np.linspace(0.0, np.pi, n)
    x = 0.5 * (1.0 - np.cos(beta))
    yt = 5.0 * 0.12 * (
        0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4
    )
    upper = np.column_stack([x, yt])[::-1]
    lower = np.column_stack([x, -yt])[1:]
    return np.vstack([upper, lower])


def _rectangular_wing_surface(semi_span: float = 4.0, chord: float = 1.0) -> LiftingSurface:
    """Symmetric rectangular wing surface definition."""
    return LiftingSurface(
        name="Wing",
        semi_span=semi_span,
        sections=[
            WingSection(y_frac=0.0, chord=chord),
            WingSection(y_frac=1.0, chord=chord),
        ],
        is_symmetric=True,
    )


def test_sphere_cp_convergence_and_dalembert():
    """Verify sphere pressure coefficient error and net d'Alembert force."""
    errors = []
    for n_lat, n_lon in [(16, 32), (32, 64)]:
        nodes, panels = sphere_quad_mesh(1.0, n_lat, n_lon)
        # Orient the sphere so its polar axis is along x (the flow direction)
        nodes_x = np.column_stack([nodes[:, 2], nodes[:, 1], -nodes[:, 0]])
        mesh = PanelMesh(
            nodes=nodes_x,
            panels=panels,
            component=np.zeros(len(panels), dtype=int),
            kind=np.full(len(panels), "surface"),
            wake_edges=np.empty((0, 4), dtype=int),
        )
        solver = PanelSolver(mesh, far_ratio=5.0)
        cond = FlightCondition(V_inf=1.0, alpha=0.0)
        res = solver.solve(cond)

        c = solver.centroids
        cp_exact = sphere_cp(c, 1.0, np.array([1.0, 0.0, 0.0]))
        max_err = float(np.max(np.abs(res.cp - cp_exact)))
        errors.append(max_err)

        # Net force vector
        rho = float(cond.rho)
        q_inf = 0.5 * rho * cond.V_inf**2
        f_vec = -np.sum(res.cp[:, None] * solver.areas[:, None] * solver.normals, axis=0) * q_inf
        net_force_norm = float(np.linalg.norm(f_vec))
        assert net_force_norm < 1.0e-3, f"Sphere {n_lat}x{n_lon} force exceeds 1e-3: {net_force_norm}"

    # Error decreases with refinement and is below 0.03 at 32x64
    assert errors[1] < errors[0], f"Error did not decrease: {errors[0]} to {errors[1]}"
    assert errors[1] < 0.03, f"Error at 32x64 not below 0.03: {errors[1]}"


def test_ellipsoid_oblique_surface_velocity():
    """Verify surface speed on an ellipsoid at oblique onset away from poles."""
    a, b, c = 1.0, 0.5, 0.25
    nodes, panels = ellipsoid_quad_mesh(a, b, c, 32, 64, spacing="cosine")
    mesh = PanelMesh(
        nodes=nodes,
        panels=panels,
        component=np.zeros(len(panels), dtype=int),
        kind=np.full(len(panels), "surface"),
        wake_edges=np.empty((0, 4), dtype=int),
    )
    solver = PanelSolver(mesh, far_ratio=5.0)

    # Oblique onset flow
    V_stream = np.array([10.0, 3.0, -2.0])
    v_mag = float(np.linalg.norm(V_stream))
    alpha = float(np.arctan2(V_stream[2], V_stream[0]))
    beta = float(-np.arcsin(V_stream[1] / v_mag))
    cond = FlightCondition(V_inf=v_mag, alpha=alpha, beta=beta)
    res = solver.solve(cond)

    speed_num = np.linalg.norm(res.velocity, axis=-1)
    v_exact = ellipsoid_surface_velocity(solver.centroids, a, b, c, V_stream)
    speed_exact = np.linalg.norm(v_exact, axis=-1)

    # Away from the poles along the major axis (|x| < 0.85 * a)
    away_mask = np.abs(solver.centroids[:, 0]) < 0.85 * a
    rel_err = np.abs(speed_num[away_mask] - speed_exact[away_mask]) / speed_exact[away_mask]
    max_rel_err = float(np.max(rel_err))
    assert max_rel_err < 0.02, f"Ellipsoid relative speed error exceeds 2 %: {max_rel_err:.4f}"


def test_superposition_exact_match():
    """Verify superposition of unit solutions equals direct solve to 1e-10."""
    surf = _rectangular_wing_surface(semi_span=4.0, chord=1.0)
    coords = _naca0012_coordinates(101)
    mesh = mesh_lifting_surface(surf, coords, n_chord=10, n_span=10)
    solver = PanelSolver(mesh, far_ratio=np.inf)

    cond = FlightCondition(
        V_inf=35.0,
        alpha=np.radians(3.5),
        beta=np.radians(-2.0),
        p=0.1,
        q=-0.05,
        r=0.08,
    )
    res_super = solver.solve(cond)

    # Direct solve with condition onset flow
    v_dir = solver._condition_velocity(cond, solver.centroids)
    sigma_direct = -np.sum(solver.normals * v_dir, axis=-1)
    rhs_direct = -solver.B @ sigma_direct
    mu_direct = np.linalg.solve(solver.A, rhs_direct)

    max_mu_diff = float(np.max(np.abs(res_super.mu - mu_direct)))
    assert max_mu_diff < 1.0e-10, f"Superposition mu difference exceeds 1e-10: {max_mu_diff}"


def test_rectangular_wing_ar8_naca0012():
    """Verify rectangular wing lift, linearity, symmetry, and Trefftz induced drag."""
    surf = _rectangular_wing_surface(semi_span=4.0, chord=1.0)
    coords = _naca0012_coordinates(201)
    mesh = mesh_lifting_surface(surf, coords, n_chord=20, n_span=20)
    solver = PanelSolver(mesh, far_ratio=np.inf)

    # 1. Zero lift at zero angle of attack
    cond0 = FlightCondition(V_inf=50.0, alpha=0.0)
    res0 = solver.solve(cond0)
    assert abs(res0.CL) < 1.0e-10, f"CL(0) is not below 1e-10: {res0.CL}"

    # 2. Lift at 4 degrees compared against VLM of the same planform
    cond4 = FlightCondition(V_inf=50.0, alpha=np.radians(4.0))
    res4 = solver.solve(cond4)

    ac = Aircraft(surfaces=[surf])
    st = SolverSettings(n_panels=20, n_chord=4)
    vlm_res = VortexLatticeSolver().solve(ac, cond4, st)
    cl_vlm = float(vlm_res.totals.CL)
    cdi_vlm = float(vlm_res.totals.CDi)

    print(f"Panel CL(4): {res4.CL:.6f}, VLM CL(4): {cl_vlm:.6f}")
    # Thickness raises lift slope slightly (0 to 10 % above VLM)
    assert res4.CL > cl_vlm, f"Panel CL ({res4.CL}) not above VLM CL ({cl_vlm})"
    assert res4.CL < 1.10 * cl_vlm, f"Panel CL ({res4.CL}) exceeds 10 % over VLM CL ({cl_vlm})"

    # 3. Equal lift on left and right halves
    q_inf = 0.5 * cond4.rho * cond4.V_inf**2
    dF = -res4.cp[:, None] * solver.areas[:, None] * solver.normals * q_inf
    l_dir = np.array([-np.sin(cond4.alpha), 0.0, np.cos(cond4.alpha)])
    lift_per_panel = dF @ l_dir
    right_mask = solver.centroids[:, 1] > 0
    left_mask = solver.centroids[:, 1] < 0
    l_right = float(np.sum(lift_per_panel[right_mask]))
    l_left = float(np.sum(lift_per_panel[left_mask]))
    half_sref = 0.5 * solver.S_ref
    cl_right = l_right / (q_inf * half_sref)
    cl_left = l_left / (q_inf * half_sref)
    assert abs(cl_right - cl_left) < 1.0e-10, f"Half wing lift mismatch: {abs(cl_right - cl_left)}"

    # 4. Trefftz plane induced drag positive and within 15 % of VLM CDi
    assert res4.CDi_trefftz > 0.0, f"Trefftz drag is not positive: {res4.CDi_trefftz}"
    assert abs(res4.CDi_trefftz - cdi_vlm) / cdi_vlm < 0.15, (
        f"CDi_trefftz ({res4.CDi_trefftz}) not within 15 % of VLM CDi ({cdi_vlm})"
    )

    # 5. Unit onset linearity: unit-w solution is strictly linear
    cond2 = FlightCondition(V_inf=50.0, alpha=np.radians(2.0))
    cond6 = FlightCondition(V_inf=50.0, alpha=np.radians(6.0))
    res2 = solver.solve(cond2)
    res6 = solver.solve(cond6)
    # The doublet and circulation solutions are strictly linear in w = V_inf * sin(alpha)
    w2 = cond2.V_inf * np.sin(cond2.alpha)
    w4 = cond4.V_inf * np.sin(cond4.alpha)
    w6 = cond6.V_inf * np.sin(cond6.alpha)
    # Scaled circulation per unit w matches to machine precision
    gamma_w2 = (res2.mu[solver.wake.wake_up] - res2.mu[solver.wake.wake_lo]) / w2
    gamma_w4 = (res4.mu[solver.wake.wake_up] - res4.mu[solver.wake.wake_lo]) / w4
    gamma_w6 = (res6.mu[solver.wake.wake_up] - res6.mu[solver.wake.wake_lo]) / w6
    max_circ_diff = float(np.max(np.abs(gamma_w6 - 2.0 * gamma_w4 + gamma_w2)))
    assert max_circ_diff < 1.0e-10, f"Circulation second difference exceeds 1e-10: {max_circ_diff}"


def test_sideslip_sign_change():
    """Verify lateral coefficients CY, Cl, Cn change sign with beta."""
    surf = _rectangular_wing_surface(semi_span=4.0, chord=1.0)
    coords = _naca0012_coordinates(101)
    mesh = mesh_lifting_surface(surf, coords, n_chord=10, n_span=10)
    solver = PanelSolver(mesh, far_ratio=np.inf)

    cond_pos = FlightCondition(V_inf=50.0, alpha=np.radians(2.0), beta=np.radians(5.0))
    cond_neg = FlightCondition(V_inf=50.0, alpha=np.radians(2.0), beta=np.radians(-5.0))

    res_pos = solver.solve(cond_pos)
    res_neg = solver.solve(cond_neg)

    assert abs(res_pos.CY + res_neg.CY) < 1.0e-10, f"CY sign change failed: {res_pos.CY + res_neg.CY}"
    assert abs(res_pos.Cl + res_neg.Cl) < 1.0e-10, f"Cl sign change failed: {res_pos.Cl + res_neg.Cl}"
    assert abs(res_pos.Cn + res_neg.Cn) < 1.0e-10, f"Cn sign change failed: {res_pos.Cn + res_neg.Cn}"
    assert abs(res_pos.Cm - res_neg.Cm) < 1.0e-10, f"Cm symmetric check failed: {res_pos.Cm - res_neg.Cm}"


def test_solve_conditions_batch():
    """Verify solve_conditions returns consistent list of PanelResult."""
    surf = _rectangular_wing_surface(semi_span=3.0, chord=1.0)
    coords = _naca0012_coordinates(101)
    mesh = mesh_lifting_surface(surf, coords, n_chord=8, n_span=8)
    solver = PanelSolver(mesh, far_ratio=np.inf)

    conditions = [
        FlightCondition(V_inf=40.0, alpha=np.radians(1.0)),
        FlightCondition(V_inf=40.0, alpha=np.radians(3.0)),
        FlightCondition(V_inf=40.0, alpha=np.radians(5.0)),
    ]
    results = solver.solve_conditions(conditions)
    assert len(results) == 3
    for i, res in enumerate(results):
        assert isinstance(res, PanelResult)
        res_single = solver.solve(conditions[i])
        assert np.isclose(res.CL, res_single.CL, atol=1.0e-12)
        assert np.isclose(res.CD_pressure, res_single.CD_pressure, atol=1.0e-12)
        assert np.isclose(res.CDi_trefftz, res_single.CDi_trefftz, atol=1.0e-12)
