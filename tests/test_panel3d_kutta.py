# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for the Ventorum-3DP pressure Kutta condition option.

Published sources
-----------------
* Kinnas, S. A. and Hsin, C. Y., 1992, Boundary element method for the
  analysis of the unsteady flow around extreme propeller geometries,
  AIAA Journal, Vol. 30, No. 3 (iterative pressure Kutta).
* Katz, J. and Plotkin, A., 2001, Low-Speed Aerodynamics, 2nd ed.,
  Cambridge University Press, section 12.4 (the Kutta condition).
* Maskew, B., 1987, Program VSAERO Theory Document, NASA CR-4023.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg

from ventorum.core.datatypes import FlightCondition, LiftingSurface, WingSection
from ventorum.panel3d.mesh import mesh_lifting_surface
from ventorum.panel3d.solver import PanelSolver


def _naca0012_coordinates(n: int = 61) -> np.ndarray:
    """Selig ordered unit-chord coordinates of a NACA 0012 section."""
    beta = np.linspace(0.0, np.pi, n)
    x = 0.5 * (1.0 - np.cos(beta))
    yt = 5.0 * 0.12 * (
        0.2969 * np.sqrt(np.maximum(x, 1e-12))
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1036 * x**4
    )
    upper = np.column_stack([x, yt])[::-1]
    lower = np.column_stack([x, -yt])[1:]
    return np.vstack([upper, lower])


def _naca2412_coordinates(n: int = 61) -> np.ndarray:
    """Selig ordered unit-chord coordinates of a NACA 2412 section.

    Thickness follows the NACA 4-digit thickness equation. Camber follows
    the NACA 4-digit camber line with maximum camber 2 percent at 40 percent
    chord. The section is the sum of camber and thickness on the x stations.
    """
    beta = np.linspace(0.0, np.pi, n)
    x = 0.5 * (1.0 - np.cos(beta))
    yt = 5.0 * 0.12 * (
        0.2969 * np.sqrt(np.maximum(x, 1e-12))
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1036 * x**4
    )
    m_camber = 0.02
    p_camber = 0.4
    yc = np.where(
        x < p_camber,
        m_camber / p_camber**2 * (2.0 * p_camber * x - x**2),
        m_camber / (1.0 - p_camber) ** 2 * ((1.0 - 2.0 * p_camber) + 2.0 * p_camber * x - x**2),
    )
    upper = np.column_stack([x, yc + yt])[::-1]
    lower = np.column_stack([x, yc - yt])[1:]
    return np.vstack([upper, lower])


def _rectangular_wing_surface(semi_span: float = 4.0, chord: float = 1.0) -> LiftingSurface:
    """Symmetric rectangular wing surface definition (AR 8 with defaults)."""
    return LiftingSurface(
        name="Wing",
        semi_span=semi_span,
        sections=[
            WingSection(y_frac=0.0, chord=chord),
            WingSection(y_frac=1.0, chord=chord),
        ],
        is_symmetric=True,
    )


def _te_jump(solver: PanelSolver, cp: np.ndarray) -> float:
    """Maximum trailing-edge pressure jump over the solved mesh strips."""
    up_idx = np.asarray(solver.wake.wake_up)
    lo_idx = np.asarray(solver.wake.wake_lo)
    return float(np.max(np.abs(cp[up_idx] - cp[lo_idx])))


def test_default_morino_bit_for_bit():
    """Verify the default option equals an explicit Morino run bit for bit."""
    surf = _rectangular_wing_surface()
    coords = _naca0012_coordinates(61)
    mesh = mesh_lifting_surface(surf, coords, n_chord=8, n_span=8)
    cond = FlightCondition(V_inf=50.0, alpha=np.radians(4.0))

    solver_default = PanelSolver(mesh, far_ratio=np.inf)
    res_default = solver_default.solve(cond)

    solver_morino = PanelSolver(mesh, far_ratio=np.inf, kutta="morino")
    res_morino = solver_morino.solve(cond)

    assert res_default.CL == res_morino.CL
    assert res_default.CD_pressure == res_morino.CD_pressure
    assert res_default.CDi_trefftz == res_morino.CDi_trefftz
    assert np.array_equal(res_default.cp, res_morino.cp)
    assert np.array_equal(res_default.mu, res_morino.mu)
    assert np.array_equal(res_default.sigma, res_morino.sigma)
    assert np.array_equal(res_default.velocity, res_morino.velocity)
    assert res_default.kutta == "morino"
    assert res_morino.kutta_iters == 0


def _run_pressure_case(coords: np.ndarray, label: str) -> None:
    """Run one pressure Kutta case with factorization counting."""
    surf = _rectangular_wing_surface()
    mesh = mesh_lifting_surface(surf, coords, n_chord=10, n_span=10)
    cond = FlightCondition(V_inf=50.0, alpha=np.radians(4.0))

    solver_morino = PanelSolver(mesh, far_ratio=np.inf, kutta="morino")
    res_morino = solver_morino.solve(cond)

    calls = {"n": 0}
    orig = scipy.linalg.lu_factor

    def counting(a, *args, **kwargs):
        calls["n"] += 1
        return orig(a, *args, **kwargs)

    scipy.linalg.lu_factor = counting
    try:
        solver = PanelSolver(mesh, far_ratio=np.inf, kutta="pressure")
        res = solver.solve(cond)
    finally:
        scipy.linalg.lu_factor = orig

    jump = _te_jump(solver, res.cp)
    print(f"{label}: Morino CL={res_morino.CL:.6f}, pressure CL={res.CL:.6f}")
    print(f"{label}: iters={res.kutta_iters}, residual={res.kutta_residual:.3e}, jump={jump:.3e}")
    assert calls["n"] == 1, f"factorization count is {calls['n']}, expected 1"
    assert res.kutta_iters <= 20
    assert jump < 1e-6
    assert res.kutta_residual < 1e-6
    assert abs(res.CL - res_morino.CL) / abs(res_morino.CL) < 0.05


def test_pressure_kutta_naca0012():
    """Verify pressure Kutta removes the trailing-edge jump on NACA 0012."""
    _run_pressure_case(_naca0012_coordinates(61), "NACA0012")


def test_pressure_kutta_naca2412():
    """Verify pressure Kutta removes the trailing-edge jump on NACA 2412."""
    _run_pressure_case(_naca2412_coordinates(61), "NACA2412")


def test_symmetric_alpha_zero_matches_morino():
    """Verify pressure equals Morino to 1e-10 on a symmetric wing at alpha 0."""
    surf = _rectangular_wing_surface()
    coords = _naca0012_coordinates(61)
    mesh = mesh_lifting_surface(surf, coords, n_chord=8, n_span=8)
    cond = FlightCondition(V_inf=50.0, alpha=0.0)

    solver_morino = PanelSolver(mesh, far_ratio=np.inf, kutta="morino")
    res_morino = solver_morino.solve(cond)

    solver = PanelSolver(mesh, far_ratio=np.inf, kutta="pressure")
    res = solver.solve(cond)

    assert abs(res.CL - res_morino.CL) < 1e-10
    assert np.max(np.abs(res.cp - res_morino.cp)) < 1e-10
    assert np.max(np.abs(res.mu - res_morino.mu)) < 1e-10


def test_pressure_kutta_half_mesh():
    """Verify pressure Kutta converges on the solved half mesh."""
    surf = _rectangular_wing_surface()
    coords = _naca0012_coordinates(51)
    mesh = mesh_lifting_surface(surf, coords, n_chord=8, n_span=6, half=True)
    cond = FlightCondition(V_inf=50.0, alpha=np.radians(4.0))

    solver = PanelSolver(mesh, far_ratio=np.inf, kutta="pressure")
    res = solver.solve(cond)

    jump = _te_jump(solver, res.cp)
    assert res.kutta_iters <= 20
    assert jump < 1e-6
    assert res.kutta_residual < 1e-6


def test_pressure_kutta_ground():
    """Verify pressure Kutta converges in ground effect on the solved mesh."""
    surf = _rectangular_wing_surface()
    coords = _naca0012_coordinates(51)
    mesh = mesh_lifting_surface(surf, coords, n_chord=8, n_span=6)
    cond = FlightCondition(V_inf=30.0, alpha=np.radians(4.0), h=1.0)

    solver = PanelSolver(mesh, far_ratio=np.inf, kutta="pressure")
    res = solver.solve(cond)

    jump = _te_jump(solver, res.cp)
    assert res.kutta_iters <= 20
    assert jump < 1e-6
    assert res.kutta_residual < 1e-6
