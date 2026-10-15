# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for the Ventorum-3DP GPU solve path and the numba backend parity.

The numpy backend is the reference. The numba backend must match it to
1e-10 relative on CL, CD, Cm and Cp. The gpu backend (float64) must match
it to 1e-10. The gpu backend (float32) must stay within a measured floor
stated in the test, never above 1e-4 on CL.

Published sources
-----------------
* Morino, L. and Kuo, C. C., 1974, Subsonic potential aerodynamics for
  complex configurations: A general theory, AIAA Journal, Vol. 12, No. 2,
  pp. 191-197.
* Katz, J. and Plotkin, A., 2001, Low-Speed Aerodynamics, 2nd ed.,
  Cambridge University Press, section 9.4 and chapter 12.
* Hess, J. L. and Smith, A. M. O., 1964, Calculation of potential flow about
  arbitrary bodies, Journal of Ship Research, Vol. 8.
"""

from __future__ import annotations

import numpy as np
import pytest

from ventorum.core.datatypes import FlightCondition, LiftingSurface, WingSection
from ventorum.panel3d.mesh import PanelMesh, mesh_lifting_surface
from ventorum.panel3d.solver import PanelSolver
from ventorum.reference.analytic3d import sphere_quad_mesh


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


def _need_cuda():
    """Skip the test when a CUDA GPU, PyTorch or Warp is missing."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("warp")
    if not torch.cuda.is_available():
        pytest.skip("A CUDA GPU is not available.")
    return torch


def _assert_close(new: float | np.ndarray, ref: float | np.ndarray, tol: float, name: str) -> None:
    """Assert relative agreement to tol, or 1e-10 absolute near zero."""
    n = np.asarray(new, dtype=float)
    r = np.asarray(ref, dtype=float)
    denom = np.maximum(np.abs(r), 1e-8)
    err = float(np.max(np.abs(n - r) / denom))
    abs_err = float(np.max(np.abs(n - r)))
    assert err <= tol or abs_err <= 1e-10, f"{name}: rel {err:.3e} abs {abs_err:.3e} above {tol:.1e}"


def _sphere_mesh() -> PanelMesh:
    """Small sphere mesh for the backend parity test in m."""
    nodes, panels = sphere_quad_mesh(1.0, 12, 24)
    nodes_x = np.column_stack([nodes[:, 2], nodes[:, 1], -nodes[:, 0]])
    return PanelMesh(
        nodes=nodes_x,
        panels=panels,
        component=np.zeros(len(panels), dtype=int),
        kind=np.full(len(panels), "surface"),
        wake_edges=np.empty((0, 4), dtype=int),
    )


def _check_solver_parity(res_new, res_ref, tol: float, label: str) -> None:
    """Compare CL, CD, Cm and Cp of two panel results."""
    _assert_close(res_new.CL, res_ref.CL, tol, f"{label} CL")
    _assert_close(res_new.CD_pressure, res_ref.CD_pressure, tol, f"{label} CD")
    _assert_close(res_new.Cm, res_ref.Cm, tol, f"{label} Cm")
    _assert_close(np.asarray(res_new.cp), np.asarray(res_ref.cp), tol, f"{label} Cp")


def test_numba_parity_sphere() -> None:
    """Numba matches numpy to 1e-10 on a sphere."""
    mesh = _sphere_mesh()
    cond = FlightCondition(V_inf=1.0, alpha=0.0)
    res_ref = PanelSolver(mesh, far_ratio=5.0, backend="numpy").solve(cond)
    res_new = PanelSolver(mesh, far_ratio=5.0, backend="numba").solve(cond)
    _check_solver_parity(res_new, res_ref, 1e-10, "sphere")
    print(f"sphere CL numpy={res_ref.CL:.6e} numba={res_new.CL:.6e}")


def test_numba_parity_wing() -> None:
    """Numba matches numpy to 1e-10 on a wing."""
    surf = _rectangular_wing_surface()
    coords = _naca0012_coordinates(61)
    mesh = mesh_lifting_surface(surf, coords, n_chord=8, n_span=8)
    cond = FlightCondition(V_inf=50.0, alpha=np.radians(4.0))
    res_ref = PanelSolver(mesh, far_ratio=np.inf, backend="numpy").solve(cond)
    res_new = PanelSolver(mesh, far_ratio=np.inf, backend="numba").solve(cond)
    _check_solver_parity(res_new, res_ref, 1e-10, "wing")
    print(f"wing CL numpy={res_ref.CL:.6f} numba={res_new.CL:.6f}")


def test_numba_parity_half_mesh_ground() -> None:
    """Numba matches numpy to 1e-10 on a half mesh wing in ground effect."""
    surf = _rectangular_wing_surface()
    coords = _naca0012_coordinates(51)
    mesh = mesh_lifting_surface(surf, coords, n_chord=6, n_span=6, half=True)
    cond = FlightCondition(V_inf=30.0, alpha=np.radians(4.0), h=0.6)
    res_ref = PanelSolver(mesh, far_ratio=np.inf, backend="numpy").solve(cond)
    res_new = PanelSolver(mesh, far_ratio=np.inf, backend="numba").solve(cond)
    _check_solver_parity(res_new, res_ref, 1e-10, "half ground")
    print(f"half ground CL numpy={res_ref.CL:.6f} numba={res_new.CL:.6f}")


def test_numba_parity_pressure_kutta() -> None:
    """Numba matches numpy to 1e-10 with the pressure Kutta option."""
    surf = _rectangular_wing_surface()
    coords = _naca0012_coordinates(51)
    mesh = mesh_lifting_surface(surf, coords, n_chord=6, n_span=6)
    cond = FlightCondition(V_inf=50.0, alpha=np.radians(4.0))
    res_ref = PanelSolver(mesh, far_ratio=np.inf, backend="numpy", kutta="pressure").solve(cond)
    res_new = PanelSolver(mesh, far_ratio=np.inf, backend="numba", kutta="pressure").solve(cond)
    _check_solver_parity(res_new, res_ref, 1e-10, "pressure")
    assert res_new.kutta_converged


def test_unknown_backend_raises() -> None:
    """An unknown backend name raises a clear error."""
    mesh = _sphere_mesh()
    with pytest.raises(ValueError, match="Unknown panel backend"):
        PanelSolver(mesh, backend="cuda-please")


def test_required_bytes_reports_size() -> None:
    """The memory estimate matches the documented dense footprint."""
    from ventorum.panel3d.gpu import required_bytes

    # float64, N=100: 8 bytes x (3 x N x N + 2 x N x 6 onsets) = 249600.
    assert required_bytes(100, "float64") == 249600
    # float32, N=100: float64 assembly pair + float32 solve triple = 284800.
    assert required_bytes(100, "float32") == 284800
    # Nearly quadratic growth: doubling N nearly quadruples the bytes.
    assert required_bytes(200, "float64") == 979200
    assert required_bytes(200, "float64") > 3.5 * required_bytes(100, "float64")
    assert required_bytes(0, "float64") == 0


def test_gpu_backend_without_cuda_raises(monkeypatch) -> None:
    """Backend gpu without a CUDA GPU raises a clear error, never a silent fallback."""
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    mesh = _sphere_mesh()
    with pytest.raises(RuntimeError, match="CUDA"):
        PanelSolver(mesh, backend="gpu")


@pytest.mark.gpu
def test_gpu_float64_parity_wing() -> None:
    """Gpu float64 matches numpy to 1e-10 on a wing."""
    torch = _need_cuda()
    surf = _rectangular_wing_surface()
    coords = _naca0012_coordinates(61)
    mesh = mesh_lifting_surface(surf, coords, n_chord=8, n_span=8)
    cond = FlightCondition(V_inf=50.0, alpha=np.radians(4.0))
    res_ref = PanelSolver(mesh, far_ratio=np.inf, backend="numpy").solve(cond)
    res_new = PanelSolver(
        mesh, far_ratio=np.inf, backend="gpu", dtype=torch.float64
    ).solve(cond)
    _check_solver_parity(res_new, res_ref, 1e-10, "gpu wing")
    print(f"gpu wing CL numpy={res_ref.CL:.6f} gpu={res_new.CL:.6f}")


@pytest.mark.gpu
def test_gpu_float64_parity_half_mesh_ground() -> None:
    """Gpu float64 matches numpy to 1e-10 on a half mesh wing in ground effect."""
    torch = _need_cuda()
    surf = _rectangular_wing_surface()
    coords = _naca0012_coordinates(51)
    mesh = mesh_lifting_surface(surf, coords, n_chord=6, n_span=6, half=True)
    cond = FlightCondition(V_inf=30.0, alpha=np.radians(4.0), h=0.6)
    res_ref = PanelSolver(mesh, far_ratio=np.inf, backend="numpy").solve(cond)
    res_new = PanelSolver(
        mesh, far_ratio=np.inf, backend="gpu", dtype=torch.float64
    ).solve(cond)
    _check_solver_parity(res_new, res_ref, 1e-10, "gpu half ground")
    print(f"gpu half ground CL numpy={res_ref.CL:.6f} gpu={res_new.CL:.6f}")


@pytest.mark.gpu
def test_gpu_float32_floor_wing() -> None:
    """Gpu float32 stays within float32 round-off of numpy on a wing.

    The test measures the error floor and states it. The gate is 1e-4 on
    CL; the expected floor is about 1e-5 relative.
    """
    torch = _need_cuda()
    surf = _rectangular_wing_surface()
    coords = _naca0012_coordinates(61)
    mesh = mesh_lifting_surface(surf, coords, n_chord=8, n_span=8)
    cond = FlightCondition(V_inf=50.0, alpha=np.radians(4.0))
    res_ref = PanelSolver(mesh, far_ratio=np.inf, backend="numpy").solve(cond)
    res_new = PanelSolver(
        mesh, far_ratio=np.inf, backend="gpu", dtype=torch.float32
    ).solve(cond)
    floors = {
        "CL": abs(res_new.CL - res_ref.CL) / max(abs(res_ref.CL), 1e-8),
        "CD": abs(res_new.CD_pressure - res_ref.CD_pressure) / max(abs(res_ref.CD_pressure), 1e-8),
        "Cm": abs(res_new.Cm - res_ref.Cm) / max(abs(res_ref.Cm), 1e-8),
        "Cp": float(np.max(np.abs(res_new.cp - res_ref.cp)) / max(float(np.max(np.abs(res_ref.cp))), 1e-8)),
    }
    print(f"gpu float32 floors: {floors}")
    assert floors["CL"] <= 1e-4, f"gpu float32 CL floor {floors['CL']:.3e} above 1e-4"


@pytest.mark.gpu
def test_gpu_pressure_kutta_half_mesh_ground() -> None:
    """Gpu pressure Kutta converges on a half mesh wing in ground effect."""
    torch = _need_cuda()
    surf = _rectangular_wing_surface()
    coords = _naca0012_coordinates(51)
    mesh = mesh_lifting_surface(surf, coords, n_chord=6, n_span=6, half=True)
    cond = FlightCondition(V_inf=30.0, alpha=np.radians(4.0), h=0.6)
    solver = PanelSolver(
        mesh, far_ratio=np.inf, backend="gpu", dtype=torch.float64, kutta="pressure"
    )
    res = solver.solve(cond)
    up_idx = np.asarray(solver.wake.wake_up)
    lo_idx = np.asarray(solver.wake.wake_lo)
    jump = float(np.max(np.abs(res.cp[up_idx] - res.cp[lo_idx])))
    assert res.kutta_iters <= 20
    assert jump < 1e-6
    assert res.kutta_residual < 1e-6


@pytest.mark.gpu
def test_gpu_memory_error_reports_size() -> None:
    """An oversize dense matrix is refused with its size in the message."""
    _need_cuda()
    from ventorum.panel3d.gpu import check_device_memory

    with pytest.raises(MemoryError, match="MiB"):
        check_device_memory(10_000_000, "float64")
