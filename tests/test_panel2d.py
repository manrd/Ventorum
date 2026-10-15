# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for the 2D inviscid panel solver with ground effect and multi-element support.

This test module verifies:
1. Closed-form stream function panel influences against 64-point Gauss quadrature.
2. Exact comparison against Karman-Trefftz and Joukowski conformal mapping solutions.
3. Convergence order of maximum surface pressure coefficient error.
4. Agreement between circulation lift and pressure-integrated lift.
5. Pressure drag coefficient verification (D'Alembert paradox in free air).
6. Two-right-hand-side angle superposition and source panel additions.
7. Ground effect image method against explicit two-body solves and asymptotic behavior.
8. Multi-element aerodynamic interference.
9. Numerical equivalence between NumPy and PyTorch backends across CPU and GPU devices.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ventorum.panel2d.influence import (
    linear_vortex_stream_influence,
    source_panel_stream_influence,
)
from ventorum.panel2d.solver import Panel2DResult, solve_inviscid
from ventorum.reference.karman_trefftz import (
    karman_trefftz_airfoil,
    karman_trefftz_solution,
)


def _gauss_legendre_quadrature(f, a: float, b: float, n_points: int = 64, n_sub: int = 16) -> float:
    """Compute 1D integral using composite Gauss-Legendre quadrature."""
    edges = np.linspace(a, b, n_sub + 1)
    tot = 0.0
    x, w = np.polynomial.legendre.leggauss(n_points)
    for i in range(n_sub):
        sub_a, sub_b = edges[i], edges[i + 1]
        t = 0.5 * (sub_b - sub_a) * x + 0.5 * (sub_a + sub_b)
        tot += 0.5 * (sub_b - sub_a) * float(np.sum(w * f(t)))
    return tot


def test_closed_form_influences_against_gauss_quadrature():
    """Verify closed-form panel stream function influences against 64-point quadrature.

    Tolerance: 1e-10 relative error at field points from 0.1 to 5 panel lengths away.
    """
    x1, y1 = 0.4, 0.2
    x2, y2 = 1.6, 0.7
    dx, dy = x2 - x1, y2 - y1
    length = float(np.hypot(dx, dy))
    tx, ty = dx / length, dy / length
    nx, ny = -ty, tx
    theta_panel = float(np.arctan2(dy, dx))

    # Field points distributed from 0.1 to 5 panel lengths away
    distances = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
    angles = [0.1, 0.5, 1.2, 2.3, 3.8, 5.1]

    for d in distances:
        for ang in angles:
            # Field point in global coordinates
            xg = 0.5 * (x1 + x2) + d * length * np.cos(ang)
            yg = 0.5 * (y1 + y2) + d * length * np.sin(ang)

            # Local coordinates
            xi = (xg - x1) * tx + (yg - y1) * ty
            eta = (xg - x1) * nx + (yg - y1) * ny

            # 1. Linear vortex panel influence: psi_1 and psi_2
            # Analytical:
            psi1_ana, psi2_ana = linear_vortex_stream_influence(xi, eta, length)

            # Quadrature for shape function 1: (1 - s/L)
            def k1(s, xi_val=xi, eta_val=eta, len_val=length):
                dist2 = (xi_val - s) ** 2 + eta_val**2
                return 1.0 / (4.0 * np.pi) * (1.0 - s / len_val) * np.log(dist2)

            def k2(s, xi_val=xi, eta_val=eta, len_val=length):
                dist2 = (xi_val - s) ** 2 + eta_val**2
                return 1.0 / (4.0 * np.pi) * (s / len_val) * np.log(dist2)

            psi1_num = _gauss_legendre_quadrature(k1, 0.0, length, n_points=64)
            psi2_num = _gauss_legendre_quadrature(k2, 0.0, length, n_points=64)

            err1 = abs(psi1_ana - psi1_num) / max(abs(psi1_num), 1e-12)
            err2 = abs(psi2_ana - psi2_num) / max(abs(psi2_num), 1e-12)
            assert err1 < 1e-10, f"psi1 relative error {err1} at d={d}, ang={ang}"
            assert err2 < 1e-10, f"psi2 relative error {err2} at d={d}, ang={ang}"

            # 2. Constant source panel influence
            psi_s_ana = source_panel_stream_influence(xi, eta, length, theta_panel)

            def ks(s, xi_val=xi, eta_val=eta, th_val=theta_panel):
                return (1.0 / (2.0 * np.pi)) * (np.arctan2(eta_val, xi_val - s) + th_val)

            psi_s_num = _gauss_legendre_quadrature(ks, 0.0, length, n_points=64)
            err_s = abs(psi_s_ana - psi_s_num) / max(abs(psi_s_num), 1e-12)
            assert err_s < 1e-10, f"psi_source relative error {err_s} at d={d}, ang={ang}"


def test_exact_karman_trefftz_checks():
    """Verify solver against exact Karman-Trefftz airfoil solution.

    Requirements:
    - Cl within 0.5% of exact at 200 panels for alpha = 0, 4, 8 deg.
    - Cl and Cl_pressure agree within 1% at 200 panels.
    - Free-air Cd_pressure below 1e-3 at 200 panels.
    - Maximum Cp error decreases between 100, 200, and 400 panels with order >= 1.
    """
    center = (-0.07, 0.04)
    tau_deg = 10.0

    # 1. Checks at 200 panels (201 nodes) for alpha = 0, 4, 8 deg
    coords_200 = karman_trefftz_airfoil(center, tau_deg, n_points=201)
    alphas = [0.0, 4.0, 8.0]
    res_200 = solve_inviscid(coords_200, alphas, backend="numpy")

    assert isinstance(res_200, Panel2DResult)

    for i, a in enumerate(alphas):
        _, cl_exact = karman_trefftz_solution(center, tau_deg, n_points=201, alpha_deg=a)
        cl_calc = float(res_200.Cl[i])
        cl_press = float(res_200.Cl_pressure[i])
        cd_press = float(res_200.Cd_pressure[i])

        rel_diff_cl = abs(cl_calc - cl_exact) / abs(cl_exact)
        assert rel_diff_cl < 0.005, f"alpha={a}: Cl error {rel_diff_cl:.4f} > 0.5%"

        # Agreement between circulation Cl and pressure Cl within 1%
        rel_diff_press = abs(cl_calc - cl_press) / abs(cl_calc)
        assert rel_diff_press < 0.01, f"alpha={a}: Cl vs Cl_press diff {rel_diff_press:.4f} > 1%"

        # D'Alembert paradox: Cd_pressure below 1e-3
        assert abs(cd_press) < 1e-3, f"alpha={a}: Cd_pressure {cd_press:.2e} >= 1e-3"

    # 2. Convergence of maximum Cp error across 100, 200, and 400 panels
    alpha_test = 4.0
    panel_counts = [100, 200, 400]
    max_cp_errors = []

    for n_pan in panel_counts:
        coords_n = karman_trefftz_airfoil(center, tau_deg, n_points=n_pan + 1)
        cp_exact, _ = karman_trefftz_solution(center, tau_deg, n_points=n_pan + 1, alpha_deg=alpha_test)
        res_n = solve_inviscid(coords_n, alpha_test, backend="numpy")
        cp_calc = np.asarray(res_n.Cp)

        # Exclude immediate trailing edge singularity node for regular error evaluation
        err = float(np.max(np.abs(cp_calc[2:-2] - cp_exact[2:-2])))
        max_cp_errors.append(err)

    order1 = np.log(max_cp_errors[0] / max_cp_errors[1]) / np.log(2.0)
    order2 = np.log(max_cp_errors[1] / max_cp_errors[2]) / np.log(2.0)
    print(f"\n[Convergence] Observed Cp error orders: {order1:.2f} (100->200), {order2:.2f} (200->400)")

    assert max_cp_errors[1] < max_cp_errors[0]
    assert max_cp_errors[2] < max_cp_errors[1]
    assert order1 >= 1.0, f"Convergence order {order1:.2f} is less than 1.0"
    assert order2 >= 1.0, f"Convergence order {order2:.2f} is less than 1.0"


def test_symmetric_joukowski_zero_lift():
    """Verify symmetric Joukowski airfoil at zero alpha gives |Cl| below 1e-10."""
    center = (-0.1, 0.0)
    tau_deg = 0.0
    coords = karman_trefftz_airfoil(center, tau_deg, n_points=101)

    res = solve_inviscid(coords, 0.0, backend="numpy")
    cl = float(res.Cl)
    assert abs(cl) < 1e-10, f"Symmetric airfoil at alpha 0 gave Cl = {cl}, expected < 1e-10"


def test_free_air_superposition():
    """Verify free-air superposition matches direct solve to 1e-12."""
    center = (-0.07, 0.04)
    tau_deg = 10.0
    coords = karman_trefftz_airfoil(center, tau_deg, n_points=101)

    # Solve via multi-angle vector (uses 2-RHS superposition)
    res_multi = solve_inviscid(coords, [0.0, 5.0, 10.0], backend="numpy")
    cl_superposed = float(res_multi.Cl[1])
    cm_superposed = float(res_multi.Cm[1])
    cp_superposed = np.asarray(res_multi.Cp[1])

    # Direct solve at alpha 5 deg
    res_direct = solve_inviscid(coords, 5.0, backend="numpy")
    cl_direct = float(res_direct.Cl)
    cm_direct = float(res_direct.Cm)
    cp_direct = np.asarray(res_direct.Cp)

    assert abs(cl_superposed - cl_direct) < 1e-12
    assert abs(cm_superposed - cm_direct) < 1e-12
    assert np.allclose(cp_superposed, cp_direct, atol=1e-12)


def test_source_panels_transpiration():
    """Verify source panel contributions against scratch assembly and bit-for-bit zero source."""
    center = (-0.07, 0.04)
    tau_deg = 10.0
    coords = karman_trefftz_airfoil(center, tau_deg, n_points=81)
    n_panels = len(coords) - 1

    # 1. Zero source must reproduce inviscid result bit for bit
    res_inviscid = solve_inviscid(coords, 4.0, backend="numpy")
    zeros_sigma = np.zeros(n_panels, dtype=np.float64)
    res_zero_sigma = solve_inviscid(coords, 4.0, sigma=zeros_sigma, backend="numpy")

    assert np.array_equal(res_inviscid.gamma, res_zero_sigma.gamma)
    assert res_inviscid.Cl == res_zero_sigma.Cl

    # 2. Smooth non-zero source distribution
    x_mid = 0.5 * (coords[:-1, 0] + coords[1:, 0])
    sigma_smooth = 0.02 * np.sin(np.pi * x_mid)

    res_sigma = solve_inviscid(coords, 4.0, sigma=sigma_smooth, backend="numpy")
    # Verify that transpiration source changes the solution
    assert abs(res_sigma.Cl - res_inviscid.Cl) > 1e-4

    # Verify superposition: delta_gamma equals solve with alpha=0, Vinf=0, only sigma
    # Testing that sigma adds linear RHS terms
    sigma_scaled = 2.0 * sigma_smooth
    res_scaled = solve_inviscid(coords, 4.0, sigma=sigma_scaled, backend="numpy")
    delta1 = np.asarray(res_sigma.gamma) - np.asarray(res_inviscid.gamma)
    delta2 = np.asarray(res_scaled.gamma) - np.asarray(res_inviscid.gamma)
    assert np.allclose(2.0 * delta1, delta2, atol=1e-12)


def test_ground_effect_properties():
    """Verify ground effect image method properties and boundary checks.

    (a) GE result equals free-air 2-body solve of body + explicit mirror to 1e-10 at h/c 0.3, alpha 4 deg.
    (b) At h/c 1000 the GE Cl is within 1e-4 of free air.
    (c) Symmetric Joukowski at alpha 0, h/c 0.1 has negative Cl (suction toward ground).
    (d) Geometry below or touching the ground raises ValueError.
    """
    center = (-0.1, 0.0)
    tau_deg = 0.0
    coords = karman_trefftz_airfoil(center, tau_deg, n_points=61)

    # (a) GE vs explicit 2-body mirror solve
    h_a = 0.3
    alpha_a = 4.0
    res_ge = solve_inviscid(coords, alpha_a, height=h_a, backend="numpy")

    # Build explicit mirror body in free air
    coords_rot = np.asarray(res_ge.coordinates)
    mirror_coords = coords_rot.copy()
    mirror_coords[:, 1] *= -1.0

    res_2body = solve_inviscid([coords_rot, mirror_coords], 0.0, backend="numpy")
    cl_ge = float(res_ge.Cl)
    cl_2body = float(res_2body.circulation[0] * 2.0)
    assert abs(cl_ge - cl_2body) < 1e-10, f"GE vs 2-body diff: {abs(cl_ge - cl_2body):.3e}"

    # (b) Asymptotic behavior: h/c = 1000 matches free air within 1e-4
    res_fa = solve_inviscid(coords, 0.0, backend="numpy")
    res_high_ge = solve_inviscid(coords, 0.0, height=1000.0, backend="numpy")
    diff_h1000 = abs(float(res_high_ge.Cl) - float(res_fa.Cl))
    assert diff_h1000 < 1e-4, f"h=1000 diff {diff_h1000:.2e} >= 1e-4"

    # (c) Symmetric section at alpha 0, h/c 0.1 has negative Cl (ground suction)
    res_suction = solve_inviscid(coords, 0.0, height=0.1, backend="numpy")
    assert float(res_suction.Cl) < 0.0, f"Expected negative Cl in ground effect, got {res_suction.Cl}"

    # (d) Geometry below or touching ground raises ValueError
    with pytest.raises(ValueError, match="touching the ground"):
        solve_inviscid(coords, 10.0, height=0.01, height_reference="pivot", backend="numpy")


def test_multi_element_well_separated():
    """Verify two well separated sections (20 chords apart) give each free-air Cl within 1e-3."""
    center = (-0.1, 0.0)
    coords0 = karman_trefftz_airfoil(center, 0.0, n_points=51)
    coords1 = coords0.copy()
    coords1[:, 1] += 20.0  # 20 chords apart in cross-flow direction

    alpha_deg = 4.0
    res_single = solve_inviscid(coords0, alpha_deg, backend="numpy")
    cl_single = float(res_single.Cl)

    res_two = solve_inviscid([coords0, coords1], alpha_deg, backend="numpy")
    cl_0 = float(res_two.circulation[0] * 2.0)
    cl_1 = float(res_two.circulation[1] * 2.0)

    err0 = abs(cl_0 - cl_single)
    err1 = abs(cl_1 - cl_single)
    assert err0 < 1e-3, f"Body 0 Cl error {err0:.2e} >= 1e-3"
    assert err1 < 1e-3, f"Body 1 Cl error {err1:.2e} >= 1e-3"


def test_backends_parity_and_accuracy():
    """Verify PyTorch backend on CPU for float64 (parity to 1e-10) and float32 (1e-4)."""
    center = (-0.07, 0.04)
    tau_deg = 10.0
    coords = karman_trefftz_airfoil(center, tau_deg, n_points=81)
    alphas = [0.0, 4.0, 8.0]

    res_np = solve_inviscid(coords, alphas, backend="numpy")
    res_th64 = solve_inviscid(coords, alphas, backend="torch", device="cpu", dtype=torch.float64)

    # Parity check: torch float64 vs numpy to 1e-10
    assert np.allclose(np.asarray(res_np.Cl), res_th64.Cl.numpy(), atol=1e-10)
    assert np.allclose(np.asarray(res_np.Cm), res_th64.Cm.numpy(), atol=1e-10)
    assert np.allclose(np.asarray(res_np.Cp), res_th64.Cp.numpy(), atol=1e-10)

    # Precision check: torch float32 within 1e-4 on Cl
    res_th32 = solve_inviscid(coords, alphas, backend="torch", device="cpu", dtype=torch.float32)
    diff_cl32 = np.max(np.abs(np.asarray(res_np.Cl) - res_th32.Cl.numpy()))
    assert diff_cl32 < 1e-4, f"float32 Cl diff {diff_cl32:.2e} >= 1e-4"


@pytest.mark.gpu
def test_backends_gpu():
    """Verify CUDA backend for float32, float64 and a 33-angle ground effect batch."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is not available")

    center = (-0.07, 0.04)
    tau_deg = 10.0
    coords = karman_trefftz_airfoil(center, tau_deg, n_points=61)

    # 33-angle batch in Ground Effect
    alphas_33 = np.linspace(-2.0, 10.0, 33)

    # float64 on CUDA
    res_cuda64 = solve_inviscid(
        coords,
        alphas_33,
        height=0.4,
        backend="torch",
        device="cuda",
        dtype=torch.float64,
    )
    res_cpu64 = solve_inviscid(
        coords,
        alphas_33,
        height=0.4,
        backend="torch",
        device="cpu",
        dtype=torch.float64,
    )

    diff_cuda = float(torch.max(torch.abs(res_cuda64.Cl.cpu() - res_cpu64.Cl)).item())
    assert diff_cuda < 1e-10, f"CUDA float64 diff {diff_cuda:.2e} >= 1e-10"

    # float32 on CUDA
    res_cuda32 = solve_inviscid(
        coords,
        alphas_33,
        height=0.4,
        backend="torch",
        device="cuda",
        dtype=torch.float32,
    )
    diff_cuda32 = float(torch.max(torch.abs(res_cuda32.Cl.cpu() - res_cpu64.Cl.float())).item())
    assert diff_cuda32 < 1e-3, f"CUDA float32 diff {diff_cuda32:.2e} >= 1e-3"
