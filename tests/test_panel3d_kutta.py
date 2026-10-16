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
import pytest
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


# Default (Morino) run stored after the change to the node-based surface
# gradient. The doublets and the Trefftz drag equal the run of the first
# solver version; the pressure values follow the present gradient method.
_STORED_MORINO = {
    "CL": 0.32913829884375495,
    "CD_pressure": -0.0022952942097927286,
    "CDi_trefftz": 0.004233828611102339,
    "Cm": -0.07665012280304392,
    "cp_sum": -271.1126355930909,
    "cp_abs_sum": 305.16602098296994,
    "cp_0": -1.3715636687742108,
    "cp_100": -0.4000422007027551,
    "mu_abs_sum": 783.0904197027365,
    "velocity_abs_sum": 17887.834801496563,
}


def test_default_morino_matches_stored_run():
    """Verify the default option against a stored Morino run.

    The default and an explicit ``kutta="morino"`` give the same result bit
    for bit, and both equal the stored run. On the machine that stored the
    values they are equal bit for bit; the test allows 1e-12 relative
    (1e-15 absolute for the coefficients near zero), because other LAPACK
    and BLAS builds can round the last bits differently.
    """
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

    stored = _STORED_MORINO
    for name in ("CL", "CD_pressure", "CDi_trefftz", "Cm"):
        assert getattr(res_default, name) == pytest.approx(stored[name], rel=1e-12), name
    for name in ("CY", "Cl", "Cn"):
        assert abs(getattr(res_default, name)) < 1e-13, name
    assert float(np.sum(res_default.cp)) == pytest.approx(stored["cp_sum"], rel=1e-12)
    assert float(np.sum(np.abs(res_default.cp))) == pytest.approx(stored["cp_abs_sum"], rel=1e-12)
    assert float(res_default.cp[0]) == pytest.approx(stored["cp_0"], rel=1e-12)
    assert float(res_default.cp[100]) == pytest.approx(stored["cp_100"], rel=1e-12)
    assert float(np.sum(np.abs(res_default.mu))) == pytest.approx(stored["mu_abs_sum"], rel=1e-12)
    assert float(np.sum(np.abs(res_default.velocity))) == pytest.approx(
        stored["velocity_abs_sum"], rel=1e-12
    )


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
    assert solver.factorization_count == 1
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


def test_pressure_kutta_half_mesh_asymmetric_equals_full_mesh():
    """Verify the pressure Kutta on a half mesh with sideslip and roll rate.

    The flow is not symmetric, so the wake corrections of the left and the
    right strips differ. The half mesh solves them with its symmetric and
    antisymmetric factorizations (no new factorization) and must equal the
    full-mesh pressure Kutta solve to 1e-10.
    """
    surf = _rectangular_wing_surface()
    coords = _naca0012_coordinates(51)
    mesh_half = mesh_lifting_surface(surf, coords, n_chord=8, n_span=6, half=True)
    mesh_full = mesh_lifting_surface(surf, coords, n_chord=8, n_span=6, half=False)
    cond = FlightCondition(V_inf=50.0, alpha=np.radians(4.0), beta=np.radians(5.0), p=0.5)

    solver_half = PanelSolver(mesh_half, far_ratio=np.inf, kutta="pressure")
    res_half = solver_half.solve(cond)
    res_full = PanelSolver(mesh_full, far_ratio=np.inf, kutta="pressure").solve(cond)

    assert res_half.kutta_converged and res_full.kutta_converged
    assert solver_half.factorization_count == 2
    n_half = len(mesh_half.panels)
    for name in ("CL", "CD_pressure", "CDi_trefftz", "CY", "Cl", "Cm", "Cn"):
        assert abs(getattr(res_half, name) - getattr(res_full, name)) < 1e-10, name
    assert np.max(np.abs(res_half.cp - res_full.cp[:n_half])) < 1e-10
    assert np.max(np.abs(res_half.mu - res_full.mu[:n_half])) < 1e-10
    n_strips = len(res_half.mu_w)
    assert np.max(np.abs(res_half.mu_w - res_full.mu_w[:n_strips])) < 1e-10
    # The right and left trailing edges of the full mesh carry no pressure jump.
    full_jump = np.abs(res_full.cp[mesh_full.wake_edges[:, 0]] - res_full.cp[mesh_full.wake_edges[:, 1]])
    assert float(np.max(full_jump)) < 1e-6
