# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Acceptance tests of the 2D viscous panel solver with boundary-layer coupling.

The tests check convergence, physical validity, transition movement,
Squire-Young drag consistency, and ground effect against analytical
and published benchmarks.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import numpy as np
import pytest

from ventorum.panel2d.viscous import solve_viscous


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


def test_naca0012_alpha_sweep_re1e6():
    """Verify convergence and physics of NACA 0012 alpha sweep at Re 1e6."""
    coords = make_naca0012(120)
    alphas = np.array([-4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0])
    res = solve_viscous(coords, alphas, 1e6, tol=1e-4, max_iter=40)

    # 1. Convergence
    assert np.all(res.converged), f"Convergence failed at alphas: {alphas[~res.converged]}"
    assert np.all(res.residual < 1e-4)

    # 2. Cl at alpha 0 below 1e-6 in magnitude
    idx_0 = np.where(alphas == 0.0)[0][0]
    assert abs(res.Cl[idx_0]) < 1e-6

    # 3. Cl slope between 0.09 and 0.115 per deg
    idx_4 = np.where(alphas == 4.0)[0][0]
    idx_m4 = np.where(alphas == -4.0)[0][0]
    cl_slope = (res.Cl[idx_4] - res.Cl[idx_m4]) / 8.0
    assert 0.09 <= cl_slope <= 0.115, f"Cl slope {cl_slope:.4f} outside [0.09, 0.115]"

    # 4. Cd at alpha 0 between 0.004 and 0.008
    cd_0 = res.Cd[idx_0]
    assert 0.004 <= cd_0 <= 0.008, f"Cd(0) {cd_0:.5f} outside [0.004, 0.008]"

    # 5. Transition moves forward on upper surface as alpha increases
    idx_8 = np.where(alphas == 8.0)[0][0]
    xtr_upper_0 = res.transition_positions[idx_0][0]
    xtr_upper_8 = res.transition_positions[idx_8][0]
    assert xtr_upper_8 < xtr_upper_0, f"Transition did not move forward: {xtr_upper_0} -> {xtr_upper_8}"

    # 6. Squire-Young Cd and friction plus pressure Cd within 10%
    cd_total_nearfield = res.Cd_friction[idx_0] + res.Cd_pressure[idx_0]
    rel_diff = abs(cd_0 - cd_total_nearfield) / cd_0
    assert rel_diff <= 0.10, f"Drag discrepancy {rel_diff * 100:.2f}% exceeds 10%"


def test_naca0012_re3e6():
    """Verify NACA 0012 Cd at Re 3e6 is lower than at Re 1e6."""
    coords = make_naca0012(120)
    res_1e6 = solve_viscous(coords, 0.0, 1e6, tol=1e-4)
    res_3e6 = solve_viscous(coords, 0.0, 3e6, tol=1e-4)

    assert res_3e6.converged
    assert 0.003 <= res_3e6.Cd <= 0.008
    assert res_3e6.Cd < res_1e6.Cd, f"Cd at 3e6 ({res_3e6.Cd:.5f}) not lower than at 1e6 ({res_1e6.Cd:.5f})"


def test_ground_effect():
    """Verify ground effect pass-through and asymptotic limit."""
    coords = make_naca0012(120)
    alpha = 2.0
    res_free = solve_viscous(coords, alpha, 1e6, tol=1e-4)
    res_h1000 = solve_viscous(coords, alpha, 1e6, height=1000.0, tol=1e-4)
    res_h02 = solve_viscous(coords, alpha, 1e6, height=0.2, tol=1e-4)

    # At h/c 1000 equals free air within 1e-4 on Cl
    assert abs(res_h1000.Cl - res_free.Cl) < 1e-4

    # Runs and converges at h/c 0.2
    assert res_h02.converged
    assert np.isfinite(res_h02.Cl)
    assert np.isfinite(res_h02.Cd)


def test_forced_transition():
    """Verify forced transition trip specification."""
    coords = make_naca0012(120)
    res = solve_viscous(coords, 0.0, 1e6, xtr=(0.10, 0.10), tol=1e-4)
    assert res.converged
    assert res.transition_positions[0] <= 0.12
    assert res.transition_positions[1] <= 0.12


def test_invalid_parameters():
    """Verify input validation for mach, backend and Reynolds number."""
    coords = make_naca0012(60)
    with pytest.raises(ValueError, match="mach"):
        solve_viscous(coords, 0.0, 1e6, mach=0.2)

    with pytest.raises(ValueError, match="(?i)backend"):
        solve_viscous(coords, 0.0, 1e6, backend="unknown")

    with pytest.raises(ValueError, match="Reynolds"):
        solve_viscous(coords, 0.0, -1e6)


def test_xfoil_comparison(capsys):
    """Print side-by-side table against XFOIL if installed."""
    xfoil_path = shutil.which("xfoil") or (
        "C:\\Users\\M\\Tools\\xfoil-6.99\\xfoil.exe"
        if os.path.exists("C:\\Users\\M\\Tools\\xfoil-6.99\\xfoil.exe")
        else None
    )
    if not xfoil_path:
        pytest.skip("XFOIL not found on PATH.")

    coords = make_naca0012(120)
    alphas = [0.0, 2.0, 4.0]
    re = 1e6

    # Run XFOIL
    xfoil_cmd = (
        "NACA 0012\n"
        "OPER\n"
        f"VISC {re:.0e}\n"
        + "".join(f"ALFA {a}\n" for a in alphas)
        + "QUIT\n"
    )

    try:
        proc = subprocess.run(
            [xfoil_path],
            input=xfoil_cmd,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        xfoil_results = {}
        for line in proc.stdout.splitlines():
            if "a =" in line and "CL =" in line:
                parts = line.split()
                a_val = float(parts[2])
                cl_val = float(parts[5])
                xfoil_results[a_val] = {"cl": cl_val}
            elif "CD =" in line and "CDf =" in line:
                parts = line.split()
                cd_val = float(parts[2])
                last_a = sorted(xfoil_results.keys())[-1]
                xfoil_results[last_a]["cd"] = cd_val
    except Exception as exc:
        pytest.skip(f"XFOIL run failed: {exc}")

    # Run Ventorum-2D
    res = solve_viscous(coords, alphas, re, tol=1e-5)

    print("\n" + "=" * 65)
    print("Side-by-Side Comparison: Ventorum-2D vs XFOIL (NACA 0012, Re=1e6)")
    print("=" * 65)
    print(f"{'Alpha':>6} | {'Cl (Ventorum)':>13} {'Cl (XFOIL)':>11} | {'Cd (Ventorum)':>13} {'Cd (XFOIL)':>11}")
    print("-" * 65)

    for i, a in enumerate(alphas):
        v_cl = res.Cl[i]
        v_cd = res.Cd[i]
        x_res = xfoil_results.get(a, {"cl": float("nan"), "cd": float("nan")})
        x_cl = x_res.get("cl", float("nan"))
        x_cd = x_res.get("cd", float("nan"))
        print(f"{a:6.1f} | {v_cl:13.4f} {x_cl:11.4f} | {v_cd:13.5f} {x_cd:11.5f}")
    print("=" * 65)
