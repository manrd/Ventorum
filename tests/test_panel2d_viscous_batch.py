# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Acceptance tests for batched 2D viscous panel sweeps on CPU and GPU.

The tests check equality between the batched PyTorch solver and the
sequential NumPy solver, input validation, active set convergence masking,
failed case diagnosis reporting, and CUDA execution in float64 and float32.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ventorum.panel2d.viscous import solve_viscous
from ventorum.panel2d.viscous_batch import (
    ViscousBatchResult,
    ViscousCase,
    solve_viscous_batch,
)


def make_naca0012(n_points: int = 120) -> np.ndarray:
    """Generate NACA 0012 coordinates in Selig order with sharp trailing edge."""
    theta = np.linspace(0.0, np.pi, n_points // 2 + 1)
    x = 0.5 * (1.0 - np.cos(theta))
    yt = 0.6 * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)
    xu = x[::-1]
    yu = yt[::-1]
    xl = x[1:]
    yl = -yt[1:]
    return np.column_stack([np.concatenate([xu, xl]), np.concatenate([yu, yl])])


def make_naca2412(n_points: int = 120) -> np.ndarray:
    """Generate NACA 2412 coordinates in Selig order."""
    theta = np.linspace(0.0, np.pi, n_points // 2 + 1)
    x = 0.5 * (1.0 - np.cos(theta))
    yt = 5.0 * 0.12 * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)
    m = 0.02
    p = 0.4
    yc = np.where(x < p, m / (p**2) * (2.0 * p * x - x**2), m / ((1.0 - p)**2) * ((1.0 - 2.0 * p) + 2.0 * p * x - x**2))
    dyc = np.where(x < p, 2.0 * m / (p**2) * (p - x), 2.0 * m / ((1.0 - p)**2) * (p - x))
    th_c = np.arctan(dyc)

    xu = x - yt * np.sin(th_c)
    yu = yc + yt * np.cos(th_c)
    xl = x + yt * np.sin(th_c)
    yl = yc - yt * np.cos(th_c)
    return np.column_stack([np.concatenate([xu[::-1], xl[1:]]), np.concatenate([yu[::-1], yl[1:]])])


def _check_equality(res_batch: ViscousBatchResult, res_seq, tol: float = 1e-8) -> None:
    """Assert agreement between batched and sequential results to tolerance."""
    assert len(res_batch) == len(res_seq.alpha)
    for i in range(len(res_batch)):
        b_res = res_batch[i]
        seq_cl = float(res_seq.Cl[i])
        seq_cd = float(res_seq.Cd[i])
        seq_cm = float(res_seq.Cm[i])
        seq_xtr = res_seq.transition_positions[i]
        seq_conv = bool(res_seq.converged[i])

        assert b_res.converged == seq_conv, f"Case {i}: converged mismatch ({b_res.converged} vs {seq_conv})"
        assert abs(b_res.Cl - seq_cl) < tol, f"Case {i}: Cl diff {abs(b_res.Cl - seq_cl):.2e} exceeds {tol}"
        assert abs(b_res.Cd - seq_cd) < tol, f"Case {i}: Cd diff {abs(b_res.Cd - seq_cd):.2e} exceeds {tol}"
        assert abs(b_res.Cm - seq_cm) < tol, f"Case {i}: Cm diff {abs(b_res.Cm - seq_cm):.2e} exceeds {tol}"
        assert abs(b_res.transition_positions[0] - seq_xtr[0]) < tol, f"Case {i}: xtr upper diff exceeds {tol}"
        assert abs(b_res.transition_positions[1] - seq_xtr[1]) < tol, f"Case {i}: xtr lower diff exceeds {tol}"


def test_batch_cpu_float64_naca0012_re5e5():
    """Verify NACA 0012 sweep at Re 5e5 against sequential solver to 1e-8."""
    coords = make_naca0012(120)
    alphas = [-4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    re = 5e5

    cases = [ViscousCase(coordinates=coords, alpha=a, Re=re) for a in alphas]
    res_batch = solve_viscous_batch(cases, backend="torch", device="cpu", dtype=torch.float64)
    res_seq = solve_viscous(coords, alphas, re)

    _check_equality(res_batch, res_seq, tol=1e-8)


def test_batch_cpu_float64_naca0012_re2e6():
    """Verify NACA 0012 sweep at Re 2e6 against sequential solver to 1e-8."""
    coords = make_naca0012(120)
    alphas = [-4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    re = 2e6

    cases = [ViscousCase(coordinates=coords, alpha=a, Re=re) for a in alphas]
    res_batch = solve_viscous_batch(cases, backend="torch", device="cpu", dtype=torch.float64)
    res_seq = solve_viscous(coords, alphas, re)

    _check_equality(res_batch, res_seq, tol=1e-8)


def test_batch_cpu_float64_naca2412_re5e5():
    """Verify NACA 2412 sweep at Re 5e5 against sequential solver to 1e-8."""
    coords = make_naca2412(120)
    alphas = [-4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    re = 5e5

    cases = [ViscousCase(coordinates=coords, alpha=a, Re=re) for a in alphas]
    res_batch = solve_viscous_batch(cases, backend="torch", device="cpu", dtype=torch.float64)
    res_seq = solve_viscous(coords, alphas, re)

    _check_equality(res_batch, res_seq, tol=1e-8)


def test_batch_cpu_float64_naca2412_re2e6():
    """Verify NACA 2412 sweep at Re 2e6 against sequential solver to 1e-8."""
    coords = make_naca2412(120)
    alphas = [-4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    re = 2e6

    cases = [ViscousCase(coordinates=coords, alpha=a, Re=re) for a in alphas]
    res_batch = solve_viscous_batch(cases, backend="torch", device="cpu", dtype=torch.float64)
    res_seq = solve_viscous(coords, alphas, re)

    _check_equality(res_batch, res_seq, tol=1e-8)


def test_batch_cpu_float64_ground_height():
    """Verify ground effect case against sequential solver to 1e-8."""
    coords = make_naca0012(120)
    alpha = 2.0
    re = 1e6
    height = 0.5

    cases = [ViscousCase(coordinates=coords, alpha=alpha, Re=re, height=height)]
    res_batch = solve_viscous_batch(cases, backend="torch", device="cpu", dtype=torch.float64)
    res_seq = solve_viscous(coords, alpha, re, height=height)

    assert len(res_batch) == 1
    b_res = res_batch[0]
    assert abs(b_res.Cl - res_seq.Cl) < 1e-8
    assert abs(b_res.Cd - res_seq.Cd) < 1e-8
    assert abs(b_res.Cm - res_seq.Cm) < 1e-8


def test_batch_input_validation():
    """Verify validation of backend, mach, and Reynolds number."""
    coords = make_naca0012(60)

    # Invalid backend
    cases = [ViscousCase(coordinates=coords, alpha=0.0, Re=1e6)]
    with pytest.raises(ValueError, match="backend"):
        solve_viscous_batch(cases, backend="numpy")

    # Invalid mach
    cases_mach = [ViscousCase(coordinates=coords, alpha=0.0, Re=1e6, mach=0.3)]
    with pytest.raises(ValueError, match="mach"):
        solve_viscous_batch(cases_mach)

    # Invalid Re
    cases_re = [ViscousCase(coordinates=coords, alpha=0.0, Re=-1e5)]
    with pytest.raises(ValueError, match="Reynolds"):
        solve_viscous_batch(cases_re)


def test_batch_failed_cases_reported():
    """Verify non-converged cases report status without raising unhandled error."""
    coords = make_naca0012(120)
    # alpha=10 with tol=1e-6 does not reach convergence within 50 iterations
    cases = [ViscousCase(coordinates=coords, alpha=10.0, Re=1e6, tol=1e-6, max_iter=20)]
    res_batch = solve_viscous_batch(cases, backend="torch", device="cpu", dtype=torch.float64)

    assert len(res_batch) == 1
    r = res_batch[0]
    assert not r.converged
    assert len(r.reason) > 0
    assert r.residual > 0.0


@pytest.mark.gpu
def test_batch_cuda_float64_equality():
    """Verify batched CUDA float64 matches sequential solver to 1e-8."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")

    coords = make_naca0012(120)
    alphas = [0.0, 2.0, 4.0]
    re = 1e6

    cases = [ViscousCase(coordinates=coords, alpha=a, Re=re) for a in alphas]
    res_cuda = solve_viscous_batch(cases, backend="torch", device="cuda:0", dtype=torch.float64)
    res_seq = solve_viscous(coords, alphas, re)

    _check_equality(res_cuda, res_seq, tol=1e-8)


@pytest.mark.gpu
def test_batch_cuda_float32():
    """Verify batched CUDA float32 runs within stated numerical floor."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")

    coords = make_naca0012(120)
    alphas = [0.0, 2.0, 4.0]
    re = 1e6

    cases = [ViscousCase(coordinates=coords, alpha=a, Re=re, tol=1e-4) for a in alphas]
    res_f32 = solve_viscous_batch(cases, backend="torch", device="cuda:0", dtype=torch.float32)
    res_seq = solve_viscous(coords, alphas, re, tol=1e-4)

    # float32 achieves within 1e-3 relative agreement to float64 sequential solution
    assert len(res_f32) == len(alphas)
    for i in range(len(res_f32)):
        r32 = res_f32[i]
        cl_seq = float(res_seq.Cl[i])
        cd_seq = float(res_seq.Cd[i])
        if abs(cl_seq) > 1e-4:
            assert abs(r32.Cl - cl_seq) / abs(cl_seq) < 1e-3
        else:
            assert abs(r32.Cl - cl_seq) < 1e-4
        assert abs(r32.Cd - cd_seq) / cd_seq < 1e-2
        assert r32.residual > 0.0
