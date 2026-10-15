# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for 3D panel symmetry folding and ground effect by images (Ventorum-3DP).

Published sources
-----------------
* Katz, J. and Plotkin, A., 2001, Low-Speed Aerodynamics, 2nd ed.,
  Cambridge University Press, section 9.4 and chapter 12.
* Morino, L. and Kuo, C. C., 1974, Subsonic potential aerodynamics for
  complex configurations: A general theory, AIAA Journal, Vol. 12, No. 2,
  pp. 191-197.
* Wieselsberger, C., 1921, Ueber den Fluegelwiderstand in der Naehe des Bodens,
  Zeitschrift fuer Flugtechnik und Motorluftschiffahrt, Vol. 12, No. 10.
"""

from __future__ import annotations

import numpy as np
import pytest

from ventorum.aero.system import ground_normal
from ventorum.core.datatypes import Aircraft, FlightCondition, LiftingSurface, SolverSettings, WingSection
from ventorum.panel3d.kernels import _precompute_geometry, doublet_potential, influence_matrices
from ventorum.panel3d.mesh import mesh_lifting_surface
from ventorum.panel3d.solver import PanelSolver, _standardize_panels
from ventorum.panel3d.wake import build_wake
from ventorum.solvers.horseshoe import VortexLatticeSolver


def _naca0012_coordinates(n: int = 101) -> np.ndarray:
    """Selig ordered unit-chord coordinates of a NACA 0012 section."""
    beta = np.linspace(0.0, np.pi, n)
    x = 0.5 * (1.0 - np.cos(beta))
    yt = 5.0 * 0.12 * (
        0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4
    )
    upper = np.column_stack([x, yt])[::-1]
    lower = np.column_stack([x, -yt])[1:]
    return np.vstack([upper, lower])


def _rectangular_wing_surface(semi_span: float = 3.0, chord: float = 1.0) -> LiftingSurface:
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


def test_half_mesh_symmetry_equals_full_mesh():
    """Verify that half mesh with symmetry equals full mesh solve to 1e-10."""
    surf = _rectangular_wing_surface(semi_span=3.0, chord=1.0)
    coords = _naca0012_coordinates(51)
    mesh_full = mesh_lifting_surface(surf, coords, n_chord=8, n_span=8, half=False)
    mesh_half = mesh_lifting_surface(surf, coords, n_chord=8, n_span=8, half=True)

    solver_full = PanelSolver(mesh_full, far_ratio=np.inf)
    solver_half = PanelSolver(mesh_half, far_ratio=np.inf)

    # 1. Symmetric condition: alpha = 4 deg, beta = 0
    cond_sym = FlightCondition(V_inf=25.0, alpha=np.radians(4.0), beta=0.0)
    res_f_sym = solver_full.solve(cond_sym)
    res_h_sym = solver_half.solve(cond_sym)

    n_half = len(mesh_half.panels)
    assert abs(res_h_sym.CL - res_f_sym.CL) < 1.0e-10
    assert abs(res_h_sym.CD_pressure - res_f_sym.CD_pressure) < 1.0e-10
    assert abs(res_h_sym.Cm - res_f_sym.Cm) < 1.0e-10
    assert np.max(np.abs(res_h_sym.cp - res_f_sym.cp[:n_half])) < 1.0e-10
    assert np.max(np.abs(res_h_sym.mu - res_f_sym.mu[:n_half])) < 1.0e-10

    # 2. Asymmetric condition: alpha = 4 deg, beta = 5 deg (sideslip)
    cond_asym = FlightCondition(V_inf=25.0, alpha=np.radians(4.0), beta=np.radians(5.0))
    res_f_asym = solver_full.solve(cond_asym)
    res_h_asym = solver_half.solve(cond_asym)

    assert abs(res_h_asym.CL - res_f_asym.CL) < 1.0e-10
    assert abs(res_h_asym.CD_pressure - res_f_asym.CD_pressure) < 1.0e-10
    assert abs(res_h_asym.CY - res_f_asym.CY) < 1.0e-10
    assert abs(res_h_asym.Cl - res_f_asym.Cl) < 1.0e-10
    assert abs(res_h_asym.Cm - res_f_asym.Cm) < 1.0e-10
    assert abs(res_h_asym.Cn - res_f_asym.Cn) < 1.0e-10
    assert np.max(np.abs(res_h_asym.cp - res_f_asym.cp[:n_half])) < 1.0e-10
    assert np.max(np.abs(res_h_asym.mu - res_f_asym.mu[:n_half])) < 1.0e-10


def test_ground_effect_equals_explicit_mirror():
    """Verify that image solve equals free-air solve of body plus explicit mirror to 1e-10."""
    surf = _rectangular_wing_surface(semi_span=2.0, chord=1.0)
    coords = _naca0012_coordinates(41)
    mesh = mesh_lifting_surface(surf, coords, n_chord=6, n_span=6, half=False)

    alpha = np.radians(4.0)
    h_val = 0.5  # chord is 1.0, so h/c = 0.5
    cond = FlightCondition(V_inf=15.0, alpha=alpha, h=h_val)

    solver_image = PanelSolver(mesh, far_ratio=np.inf)
    res_image = solver_image.solve(cond)

    # Build explicit mirror body in free air
    k_g = ground_normal(alpha, 0.0, 0.0)
    ref_pt = solver_image.ref_point
    offset = float(np.dot(ref_pt, k_g) - h_val)

    nodes_real = mesh.nodes
    h_pts = np.dot(nodes_real, k_g) - offset
    nodes_img = nodes_real - 2.0 * h_pts[:, None] * k_g[None, :]

    panels_real = _standardize_panels(mesh.panels, mesh.nodes)
    panels_img = _standardize_panels(panels_real[:, [0, 3, 2, 1]], nodes_img) + len(nodes_real)

    nodes_both = np.vstack([nodes_real, nodes_img])
    panels_both = np.vstack([panels_real, panels_img])

    geom_real = _precompute_geometry(nodes_real[panels_real])
    geom_img = _precompute_geometry(nodes_img[panels_img - len(nodes_real)])
    c_both = np.vstack([geom_real.centroid, geom_img.centroid])

    wake_real = build_wake(nodes_real, panels_real, mesh.wake_edges, geom_real.normal)
    wake_panels_real = wake_real.panels
    h_wake = np.dot(wake_panels_real, k_g) - offset
    wake_panels_img = wake_panels_real - 2.0 * h_wake[..., None] * k_g[None, None, :]
    wake_panels_img = wake_panels_img[:, [0, 3, 2, 1], :]

    wake_both = np.vstack([wake_panels_real, wake_panels_img])
    Cw_both = doublet_potential(c_both, wake_both, far_ratio=np.inf)

    verts_both = nodes_both[panels_both]
    B_both, C_both = influence_matrices(c_both, verts_both, far_ratio=np.inf)

    A_both = C_both.copy()
    n_real = len(panels_real)
    nw = len(wake_panels_real)
    for s, strip_p_idx in enumerate(wake_real.strip_panel_indices):
        up_idx = wake_real.wake_up[s]
        lo_idx = wake_real.wake_lo[s]
        cs_r = np.sum(Cw_both[:, strip_p_idx], axis=1)
        cs_i = np.sum(Cw_both[:, nw + np.array(strip_p_idx)], axis=1)
        A_both[:, up_idx] += cs_r
        A_both[:, lo_idx] -= cs_r
        A_both[:, n_real + up_idx] += cs_i
        A_both[:, n_real + lo_idx] -= cs_i

    d = np.array([np.cos(alpha), 0.0, np.sin(alpha)])
    v_inf_vec = float(cond.V_inf) * d
    geom_both = _precompute_geometry(verts_both)
    sigma_both = -np.sum(geom_both.normal * v_inf_vec, axis=-1)
    rhs = -B_both @ sigma_both
    mu_both = np.linalg.solve(A_both, rhs)

    # Compare doublet strengths on real body
    diff_mu = np.max(np.abs(res_image.mu - mu_both[:n_real]))
    assert diff_mu < 1.0e-10, f"diff_mu {diff_mu} exceeds 1e-10"

    diff_sigma = np.max(np.abs(res_image.sigma - sigma_both[:n_real]))
    assert diff_sigma < 1.0e-10, f"diff_sigma {diff_sigma} exceeds 1e-10"

    # Compare pressure coefficients on real body
    from ventorum.panel3d.post import build_edge_neighbourhood, compute_surface_gradient, compute_surface_velocity_and_cp, integrate_forces_and_moments
    nbrs = build_edge_neighbourhood(panels_real, mesh.wake_edges)
    grad_both = compute_surface_gradient(geom_real.centroid, geom_real.normal, mu_both[:n_real], nbrs)
    v_surf_both, cp_both = compute_surface_velocity_and_cp(v_inf_vec, geom_real.normal, grad_both, float(cond.V_inf))

    diff_cp = np.max(np.abs(res_image.cp - cp_both))
    assert diff_cp < 1.0e-10, f"diff_cp {diff_cp} exceeds 1e-10"

    forces_exp = integrate_forces_and_moments(
        cp=cp_both,
        areas=geom_real.area,
        normals=geom_real.normal,
        centroids=geom_real.centroid,
        condition=cond,
        S_ref=solver_image.S_ref,
        b_ref=solver_image.b_ref,
        c_ref=solver_image.c_ref,
        ref_point=solver_image.ref_point,
    )
    assert abs(res_image.CL - forces_exp["CL"]) < 1.0e-10
    assert abs(res_image.CD_pressure - forces_exp["CD_pressure"]) < 1.0e-10
    assert abs(res_image.Cm - forces_exp["Cm"]) < 1.0e-10


def test_ground_effect_asymptotic_and_height_sweep(capsys):
    """Verify free-air asymptote at h/c=1000 and CL increase as h/c goes from 1.0 to 0.3."""
    surf = _rectangular_wing_surface(semi_span=3.0, chord=1.0)
    coords = _naca0012_coordinates(51)
    mesh = mesh_lifting_surface(surf, coords, n_chord=8, n_span=8, half=False)

    alpha_deg = 4.0
    alpha = np.radians(alpha_deg)
    cond_free = FlightCondition(V_inf=30.0, alpha=alpha)

    solver = PanelSolver(mesh, far_ratio=np.inf)
    res_free = solver.solve(cond_free)

    # 1. Asymptote at h/c = 1000
    cond_h1000 = FlightCondition(V_inf=30.0, alpha=alpha, h=1000.0)
    res_h1000 = solver.solve(cond_h1000)

    rel_diff_cl = abs(res_h1000.CL - res_free.CL) / abs(res_free.CL)
    assert rel_diff_cl < 1.0e-5, f"Relative CL difference at h=1000 is {rel_diff_cl}, exceeds 1e-5"

    # 2. Height sweep: CL increases as h/c goes from 1.0 to 0.3
    # Compare with VLM
    aircraft = Aircraft(surfaces=[surf])
    settings = SolverSettings()
    vlm = VortexLatticeSolver()

    cl_3dp_list = []
    cl_vlm_list = []
    heights = [1.0, 0.7, 0.5, 0.3]

    for h_val in heights:
        c_ge = FlightCondition(V_inf=30.0, alpha=alpha, h=h_val)
        res_3dp = solver.solve(c_ge)
        res_vlm = vlm.solve(aircraft, c_ge, settings)

        cl_3dp_list.append(res_3dp.CL)
        cl_vlm_list.append(float(res_vlm.totals.CL))

    with capsys.disabled():
        cl_vlm_free = float(vlm.solve(aircraft, cond_free, settings).totals.CL)
        print("\nHeight sweep comparison (AR 6 NACA 0012, alpha=4 deg):")
        print(f"Free-air: 3DP CL={res_free.CL:.5f}, VLM CL={cl_vlm_free:.5f}")
        for h_val, c3, cv in zip(heights, cl_3dp_list, cl_vlm_list):
            print(f"  h/c={h_val:.2f}: 3DP CL={c3:.5f}, VLM CL={cv:.5f}")

    # Verify monotonic increase of CL as h/c decreases
    for i in range(len(heights) - 1):
        assert cl_3dp_list[i + 1] > cl_3dp_list[i], (
            f"CL did not increase from h={heights[i]} ({cl_3dp_list[i]}) to "
            f"h={heights[i+1]} ({cl_3dp_list[i+1]})"
        )


def test_half_mesh_plus_ground_equals_full_mesh_plus_ground():
    """Verify that half mesh plus ground equals full mesh plus ground to 1e-10."""
    surf = _rectangular_wing_surface(semi_span=3.0, chord=1.0)
    coords = _naca0012_coordinates(51)
    mesh_full = mesh_lifting_surface(surf, coords, n_chord=6, n_span=6, half=False)
    mesh_half = mesh_lifting_surface(surf, coords, n_chord=6, n_span=6, half=True)

    solver_full = PanelSolver(mesh_full, far_ratio=np.inf)
    solver_half = PanelSolver(mesh_half, far_ratio=np.inf)

    # Condition in ground effect with sideslip
    cond = FlightCondition(
        V_inf=25.0,
        alpha=np.radians(3.5),
        beta=np.radians(5.0),
        h=0.6,
    )

    res_full = solver_full.solve(cond)
    res_half = solver_half.solve(cond)

    n_half = len(mesh_half.panels)
    assert abs(res_half.CL - res_full.CL) < 1.0e-10
    assert abs(res_half.CD_pressure - res_full.CD_pressure) < 1.0e-10
    assert abs(res_half.CY - res_full.CY) < 1.0e-10
    assert abs(res_half.Cl - res_full.Cl) < 1.0e-10
    assert abs(res_half.Cm - res_full.Cm) < 1.0e-10
    assert abs(res_half.Cn - res_full.Cn) < 1.0e-10
    assert np.max(np.abs(res_half.cp - res_full.cp[:n_half])) < 1.0e-10
    assert np.max(np.abs(res_half.mu - res_full.mu[:n_half])) < 1.0e-10


def test_ground_touching_or_under_raises_value_error():
    """Verify that geometry touching or under the ground raises ValueError."""
    surf = _rectangular_wing_surface(semi_span=3.0, chord=1.0)
    coords = _naca0012_coordinates(31)
    mesh = mesh_lifting_surface(surf, coords, n_chord=6, n_span=6, half=False)
    solver = PanelSolver(mesh, far_ratio=np.inf)

    # Negative height
    with pytest.raises(ValueError):
        solver.solve(FlightCondition(V_inf=20.0, alpha=0.0, h=-0.5))

    # Zero height
    with pytest.raises(ValueError):
        solver.solve(FlightCondition(V_inf=20.0, alpha=0.0, h=0.0))

    # Ground strike: height smaller than thickness/2 (NACA 0012 half thickness is 0.06 m)
    with pytest.raises(ValueError):
        solver.solve(FlightCondition(V_inf=20.0, alpha=0.0, h=0.03))


def test_solve_conditions_grouping_by_attitude_and_height():
    """Verify that solve_conditions groups by (attitude, height) and matches single solves."""
    surf = _rectangular_wing_surface(semi_span=2.5, chord=1.0)
    coords = _naca0012_coordinates(31)
    mesh = mesh_lifting_surface(surf, coords, n_chord=6, n_span=6, half=True)
    solver = PanelSolver(mesh, far_ratio=np.inf)

    cond1 = FlightCondition(V_inf=20.0, alpha=np.radians(2.0), h=0.8)
    cond2 = FlightCondition(V_inf=30.0, alpha=np.radians(2.0), h=0.8)  # Same group as cond1
    cond3 = FlightCondition(V_inf=25.0, alpha=np.radians(4.0), h=0.5)  # Different group
    cond4 = FlightCondition(V_inf=20.0, alpha=np.radians(4.0))         # Free air

    conditions = [cond1, cond2, cond3, cond4]
    results_batch = solver.solve_conditions(conditions)

    assert len(results_batch) == 4
    for cond, res_batch in zip(conditions, results_batch):
        res_single = solver.solve(cond)
        assert abs(res_batch.CL - res_single.CL) < 1.0e-12
        assert abs(res_batch.Cm - res_single.Cm) < 1.0e-12
        assert abs(res_batch.CD_pressure - res_single.CD_pressure) < 1.0e-12
