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
import scipy.linalg

from ventorum.aero.system import freestream_direction, ground_normal
from ventorum.core.datatypes import Aircraft, FlightCondition, LiftingSurface, SolverSettings, WingSection
from ventorum.core.errors import GroundStrikeError
from ventorum.panel3d.kernels import _precompute_geometry, doublet_potential, influence_matrices
from ventorum.panel3d.mesh import mesh_lifting_surface
from ventorum.panel3d.post import (
    build_surface_gradient_operator,
    compute_surface_velocity_and_cp,
    compute_trefftz_induced_drag,
    integrate_forces_and_moments,
    surface_gradient,
)
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


def _ground_config(solver: PanelSolver):
    """Return the ground-effect configurations of a solver."""
    return [cfg for key, cfg in solver._configs.items() if key[0] == "ground"]


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
    """Verify that image solve equals free-air solve of body plus explicit mirror to 1e-10.

    In ground effect the wake follows the free stream, so the explicit
    model sheds the wake of the body and of its mirror along the free
    stream too. The Trefftz drag of the image solve is half the free-air
    Trefftz drag of the body and mirror pair (the mirror wake carries the
    same drag by symmetry).
    """
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
    n_real = len(panels_real)
    wake_edges_both = np.vstack([
        mesh.wake_edges,
        mesh.wake_edges + np.array([n_real, n_real, len(nodes_real), len(nodes_real)]),
    ])

    verts_both = nodes_both[panels_both]
    geom_both = _precompute_geometry(verts_both)
    c_both = geom_both.centroid
    wake_dir = freestream_direction(alpha, 0.0)
    wake_both = build_wake(
        nodes_both,
        panels_both,
        wake_edges_both,
        geom_both.normal,
        wake_length=solver_image.wake_length,
        wake_dir=wake_dir,
    )
    Cw_both = doublet_potential(c_both, wake_both.panels, far_ratio=np.inf)
    B_both, C_both = influence_matrices(c_both, verts_both, far_ratio=np.inf)

    A_both = C_both.copy()
    for s, strip_p_idx in enumerate(wake_both.strip_panel_indices):
        cs = np.sum(Cw_both[:, strip_p_idx], axis=1)
        A_both[:, wake_both.wake_up[s]] += cs
        A_both[:, wake_both.wake_lo[s]] -= cs

    v_inf_vec = float(cond.V_inf) * wake_dir
    sigma_both = -np.sum(geom_both.normal * v_inf_vec, axis=-1)
    rhs = -B_both @ sigma_both
    mu_both = np.linalg.solve(A_both, rhs)

    # Compare doublet strengths on real body
    diff_mu = np.max(np.abs(res_image.mu - mu_both[:n_real]))
    assert diff_mu < 1.0e-10, f"diff_mu {diff_mu} exceeds 1e-10"

    diff_sigma = np.max(np.abs(res_image.sigma - sigma_both[:n_real]))
    assert diff_sigma < 1.0e-10, f"diff_sigma {diff_sigma} exceeds 1e-10"

    # Compare pressure coefficients on real body
    geom_real = _precompute_geometry(nodes_real[panels_real])
    grad_op = build_surface_gradient_operator(
        nodes_real, panels_real, geom_real.centroid, geom_real.normal, geom_real.area, mesh.wake_edges,
        kinds=mesh.kind,
    )
    grad_both = surface_gradient(grad_op, mu_both[:n_real])
    v_surf_both, cp_both = compute_surface_velocity_and_cp(
        np.broadcast_to(v_inf_vec, geom_real.centroid.shape), geom_real.normal, grad_both, float(cond.V_inf)
    )

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

    cdi_pair = compute_trefftz_induced_drag(
        wake_both, mu_both, float(cond.V_inf), float(cond.rho), solver_image.S_ref, wake_dir=wake_dir
    )
    assert res_image.CDi_trefftz == pytest.approx(0.5 * cdi_pair, rel=1e-9)


def test_ground_effect_asymptotic_and_height_sweep(capsys):
    """Verify the far-ground asymptote at h/c=1000 and CL increase as h/c goes from 1.0 to 0.3.

    In ground effect the wake follows the free stream; in free air it
    follows the body x axis (owner decision: one factorization per sweep
    in free air). Far from the ground the image solve therefore tends to a
    free-air solve with a free-stream wake. The test compares h/c 1000
    with h/c 1e6 (the images are then negligible) to 1e-5 relative, and
    prints the difference to the body-axis free-air result, which is the
    effect of the wake direction.
    """
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
    res_far = solver.solve(FlightCondition(V_inf=30.0, alpha=alpha, h=1.0e6))

    rel_diff_cl = abs(res_h1000.CL - res_far.CL) / abs(res_far.CL)
    assert rel_diff_cl < 1.0e-5, f"Relative CL difference at h=1000 is {rel_diff_cl}, exceeds 1e-5"
    with capsys.disabled():
        print(
            f"\nFar ground (free-stream wake) CL={res_far.CL:.6f}, free air (body-axis wake) "
            f"CL={res_free.CL:.6f}, relative difference {abs(res_far.CL / res_free.CL - 1.0):.2e}"
        )

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


def test_ground_reduces_trefftz_induced_drag(capsys):
    """Verify that the Trefftz drag falls near the ground (images in the Trefftz plane).

    Near the ground the image vortices cancel part of the downwash, so the
    induced drag falls (Wieselsberger 1921). The VLM values are printed
    for information.
    """
    surf = _rectangular_wing_surface(semi_span=3.0, chord=1.0)
    mesh = mesh_lifting_surface(surf, _naca0012_coordinates(51), n_chord=8, n_span=8, half=True)
    solver = PanelSolver(mesh, far_ratio=np.inf)
    alpha = np.radians(4.0)
    conds = [FlightCondition(V_inf=30.0, alpha=alpha, h=h) for h in (None, 1.0, 0.3)]
    res = solver.solve_conditions(conds)
    ratio = [r.CDi_trefftz / r.CL**2 for r in res]

    aircraft = Aircraft(surfaces=[surf])
    vlm = [VortexLatticeSolver().solve(aircraft, c, SolverSettings()) for c in conds]
    ratio_vlm = [float(v.totals.CDi) / float(v.totals.CL) ** 2 for v in vlm]
    with capsys.disabled():
        for c, r, rv in zip(conds, ratio, ratio_vlm):
            print(f"h={c.h}: 3DP CDi/CL^2={r:.5f}, VLM CDi/CL^2={rv:.5f}")
    assert res[2].CDi_trefftz < res[1].CDi_trefftz < res[0].CDi_trefftz
    assert ratio[2] < ratio[1] < ratio[0]


def test_half_mesh_plus_ground_equals_full_mesh_plus_ground():
    """Verify that half mesh plus ground equals full mesh plus ground to 1e-10."""
    surf = _rectangular_wing_surface(semi_span=3.0, chord=1.0)
    coords = _naca0012_coordinates(51)
    mesh_full = mesh_lifting_surface(surf, coords, n_chord=6, n_span=6, half=False)
    mesh_half = mesh_lifting_surface(surf, coords, n_chord=6, n_span=6, half=True)

    solver_full = PanelSolver(mesh_full, far_ratio=np.inf)
    solver_half = PanelSolver(mesh_half, far_ratio=np.inf)

    n_half = len(mesh_half.panels)
    conds = [
        # Symmetric condition: solved with the four-fold images.
        FlightCondition(V_inf=25.0, alpha=np.radians(3.5), h=0.6, q=0.2),
        # Sideslip: the wake follows the free stream, so the solver uses
        # the full mesh.
        FlightCondition(V_inf=25.0, alpha=np.radians(3.5), beta=np.radians(5.0), h=0.6),
    ]
    for cond in conds:
        res_full = solver_full.solve(cond)
        res_half = solver_half.solve(cond)
        assert abs(res_half.CL - res_full.CL) < 1.0e-10
        assert abs(res_half.CD_pressure - res_full.CD_pressure) < 1.0e-10
        assert abs(res_half.CDi_trefftz - res_full.CDi_trefftz) < 1.0e-10
        assert abs(res_half.CY - res_full.CY) < 1.0e-10
        assert abs(res_half.Cl - res_full.Cl) < 1.0e-10
        assert abs(res_half.Cm - res_full.Cm) < 1.0e-10
        assert abs(res_half.Cn - res_full.Cn) < 1.0e-10
        assert np.max(np.abs(res_half.cp - res_full.cp[:n_half])) < 1.0e-10
        assert np.max(np.abs(res_half.mu - res_full.mu[:n_half])) < 1.0e-10


def test_half_mesh_ground_with_bank_equals_full_mesh():
    """Verify that a half mesh with bank in ground effect equals the full mesh.

    With bank the ground normal has a y component, so the y = 0 mirror of
    the ground image is not an image of the mirror body: four-fold images
    are wrong. The solver must solve the full mesh instead.
    """
    surf = _rectangular_wing_surface(semi_span=3.0, chord=1.0)
    coords = _naca0012_coordinates(51)
    mesh_full = mesh_lifting_surface(surf, coords, n_chord=6, n_span=6, half=False)
    mesh_half = mesh_lifting_surface(surf, coords, n_chord=6, n_span=6, half=True)
    cond = FlightCondition(V_inf=25.0, alpha=np.radians(4.0), phi=np.radians(5.0), h=0.6)

    res_full = PanelSolver(mesh_full, far_ratio=np.inf).solve(cond)
    res_half = PanelSolver(mesh_half, far_ratio=np.inf).solve(cond)
    n_half = len(mesh_half.panels)
    for name in ("CL", "CD_pressure", "CDi_trefftz", "CY", "Cl", "Cm", "Cn"):
        assert abs(getattr(res_half, name) - getattr(res_full, name)) < 1.0e-10, name
    assert np.max(np.abs(res_half.cp - res_full.cp[:n_half])) < 1.0e-10
    assert np.max(np.abs(res_half.mu - res_full.mu[:n_half])) < 1.0e-10


def test_ground_wake_follows_free_stream_above_ground():
    """Verify that the ground-effect wake follows the free stream and stays above the ground.

    At alpha 8 deg and h/c 0.25 a wake along the body x axis would go
    through the ground plane. The free-stream wake keeps the height of
    the trailing edge. The free-air wake stays on the body x axis.
    """
    surf = _rectangular_wing_surface(semi_span=2.0, chord=1.0)
    mesh = mesh_lifting_surface(surf, _naca0012_coordinates(41), n_chord=6, n_span=6, half=True)
    solver = PanelSolver(mesh, far_ratio=np.inf)
    alpha = np.radians(8.0)
    cond = FlightCondition(V_inf=20.0, alpha=alpha, h=0.25)
    res = solver.solve(cond)
    assert np.isfinite(res.CL)

    (cfg,) = _ground_config(solver)
    normal, offset = cfg.ground
    wake_dir = freestream_direction(alpha, 0.0)
    for wake in (cfg.wake, cfg.full_wake):
        corners = wake.panels.reshape(-1, 3)
        clearance = corners @ normal - offset
        assert float(np.min(clearance)) > 0.0
        # The streamwise edge of a panel is q1 -> q2 or q1 -> q4 (the order
        # follows the normal of the upper trailing-edge panel).
        for quad in wake.panels:
            edges = [quad[1] - quad[0], quad[3] - quad[0]]
            stream = max(edges, key=lambda e: float(np.linalg.norm(e)))
            assert np.allclose(stream / np.linalg.norm(stream), wake_dir, atol=1e-12)
    te_nodes = mesh.nodes[mesh.wake_edges[:, 2:4].ravel()]
    assert float(np.min(clearance)) == pytest.approx(
        float(np.min(te_nodes @ normal - offset)), abs=1e-12
    )
    # A body-axis wake of the same length would cross the ground plane.
    tips = te_nodes + solver.wake_length * np.array([1.0, 0.0, 0.0])
    assert float(np.min(tips @ normal - offset)) < 0.0
    # Free air keeps the body-axis wake.
    free_edge = solver.wake.panels[:, 1] - solver.wake.panels[:, 0]
    assert np.allclose(free_edge[:, 1:], 0.0) and np.all(free_edge[:, 0] > 0.0)


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
    with pytest.raises(GroundStrikeError):
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


@pytest.mark.parametrize("half", [False, True])
def test_ground_factorizations_one_per_group(monkeypatch, half: bool):
    """Verify one factorization per (attitude, height) group, counted.

    Two ground groups and free air give three factorizations on a full
    mesh and on a half mesh (symmetric conditions use only the symmetric
    system). The count of the solver equals the count of a wrapper around
    the LU routine. A sideslip condition in ground effect on a half mesh
    goes to the full mesh: one more factorization.
    """
    calls = {"n": 0}
    original = scipy.linalg.lu_factor

    def counting(a, *args, **kwargs):
        calls["n"] += 1
        return original(a, *args, **kwargs)

    monkeypatch.setattr(scipy.linalg, "lu_factor", counting)
    surf = _rectangular_wing_surface(semi_span=2.5, chord=1.0)
    mesh = mesh_lifting_surface(surf, _naca0012_coordinates(31), n_chord=6, n_span=6, half=half)
    solver = PanelSolver(mesh, far_ratio=np.inf)
    conds = [
        FlightCondition(V_inf=20.0, alpha=np.radians(2.0), h=0.8),
        FlightCondition(V_inf=30.0, alpha=np.radians(2.0), h=0.8, q=0.1),
        FlightCondition(V_inf=25.0, alpha=np.radians(4.0), h=0.5),
        FlightCondition(V_inf=20.0, alpha=np.radians(4.0)),
        FlightCondition(V_inf=22.0, alpha=np.radians(6.0)),
    ]
    solver.solve_conditions(conds)
    assert calls["n"] == 3
    assert solver.factorization_count == 3
    solver.solve(FlightCondition(V_inf=20.0, alpha=np.radians(2.0), beta=np.radians(3.0), h=0.8))
    assert calls["n"] == 4
    assert solver.factorization_count == 4


def test_explicit_ground_plane_equals_condition_height():
    """Verify that an explicit ground plane gives the same result as condition.h."""
    surf = _rectangular_wing_surface(semi_span=2.5, chord=1.0)
    mesh = mesh_lifting_surface(surf, _naca0012_coordinates(31), n_chord=6, n_span=6, half=True)
    solver = PanelSolver(mesh, far_ratio=np.inf)
    alpha = np.radians(3.0)
    res_h = solver.solve(FlightCondition(V_inf=20.0, alpha=alpha, h=0.7))
    k_g = ground_normal(alpha, 0.0, 0.0)
    offset = float(np.dot(solver.ref_point, k_g) - 0.7)
    res_g = solver.solve(FlightCondition(V_inf=20.0, alpha=alpha), ground=(k_g, offset))
    for name in ("CL", "CD_pressure", "CDi_trefftz", "Cm"):
        assert getattr(res_g, name) == getattr(res_h, name)
    assert solver.factorization_count == 1
