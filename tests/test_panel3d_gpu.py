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
    """The memory estimate matches the documented terms of one configuration.

    For N = 100 panels: the float64 assembly pair 2 N^2 8 = 160000 bytes;
    the kept body matrices (2 on a full mesh, 4 on a half mesh); the
    ground image matrices (as many as the body matrices); the two summed
    self and mirror matrices of a half mesh; A, B and LU of the system that
    is factorized, plus the LU of the other system; 2 N 6 per system for
    the right-hand sides and solutions.
    """
    from ventorum.panel3d.gpu import required_bytes

    nn = 100 * 100
    # Full mesh, free air, float64.
    assert required_bytes(100, "float64") == 2 * nn * 8 + 2 * nn * 8 + 3 * nn * 8 + 2 * 100 * 6 * 8
    # Full mesh, free air, float32 (float64 assembly pair, float32 solve).
    assert required_bytes(100, "float32") == 2 * nn * 8 + 2 * nn * 4 + 3 * nn * 4 + 2 * 100 * 6 * 4
    # Half mesh in ground effect with both systems, float64.
    half_ground = required_bytes(100, "float64", n_systems=2, half=True, ground=True)
    assert half_ground == (
        2 * nn * 8 + 4 * nn * 8 + 4 * nn * 8 + 2 * nn * 8 + 3 * nn * 8 + 1 * nn * 8 + 2 * 2 * 100 * 6 * 8
    )
    half_free = required_bytes(100, "float64", n_systems=2, half=True)
    full_free = required_bytes(100, "float64")
    assert half_ground > half_free > full_free
    # Body matrices already on the device for a later configuration.
    assert required_bytes(100, "float64", ground=True, body_cached=True) == (
        2 * nn * 8 + 2 * nn * 8 + 3 * nn * 8 + 2 * 100 * 6 * 8
    )
    # Nearly quadratic growth: doubling N nearly quadruples the bytes.
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


@pytest.mark.gpu
def test_gpu_path_does_no_cpu_factorization(monkeypatch) -> None:
    """The gpu backend factorizes on the device only and counts it.

    A half mesh in free air (symmetric and antisymmetric systems), a ground
    configuration and the pressure Kutta option run with a wrapper around
    the SciPy LU routine (no call allowed) and around the torch LU routine
    (the count must equal the factorization count of the solver).
    """
    torch = _need_cuda()
    import scipy.linalg

    cpu_calls = {"n": 0}
    gpu_calls = {"n": 0}
    scipy_lu = scipy.linalg.lu_factor
    torch_lu = torch.linalg.lu_factor

    def counting_cpu(a, *args, **kwargs):
        cpu_calls["n"] += 1
        return scipy_lu(a, *args, **kwargs)

    def counting_gpu(a, *args, **kwargs):
        gpu_calls["n"] += 1
        return torch_lu(a, *args, **kwargs)

    monkeypatch.setattr(scipy.linalg, "lu_factor", counting_cpu)
    monkeypatch.setattr(torch.linalg, "lu_factor", counting_gpu)
    surf = _rectangular_wing_surface()
    mesh = mesh_lifting_surface(surf, _naca0012_coordinates(51), n_chord=6, n_span=6, half=True)
    solver = PanelSolver(
        mesh, far_ratio=np.inf, backend="gpu", dtype=torch.float64, kutta="pressure"
    )
    assert gpu_calls["n"] == 0
    solver.solve(FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=np.radians(3.0)))
    solver.solve(FlightCondition(V_inf=30.0, alpha=np.radians(4.0), h=0.6))
    assert cpu_calls["n"] == 0
    assert gpu_calls["n"] == 3
    assert solver.factorization_count == 3


@pytest.mark.gpu
def test_gpu_float64_parity_ground_sideslip_and_pressure() -> None:
    """Gpu float64 matches numpy to 1e-10 in ground effect with the free-stream wake.

    The sideslip case on a half mesh goes to the full mesh; the pressure
    Kutta case on a half mesh with sideslip uses both factorizations.
    """
    torch = _need_cuda()
    surf = _rectangular_wing_surface()
    mesh = mesh_lifting_surface(surf, _naca0012_coordinates(51), n_chord=6, n_span=6, half=True)
    conds = [
        FlightCondition(V_inf=30.0, alpha=np.radians(6.0), beta=np.radians(4.0), h=0.5),
        FlightCondition(V_inf=30.0, alpha=np.radians(4.0), phi=np.radians(3.0), h=0.6),
    ]
    for kutta in ("morino", "pressure"):
        ref = PanelSolver(mesh, far_ratio=np.inf, kutta=kutta).solve_conditions(conds)
        new = PanelSolver(
            mesh, far_ratio=np.inf, kutta=kutta, backend="gpu", dtype=torch.float64
        ).solve_conditions(conds)
        for r_new, r_ref in zip(new, ref):
            _check_solver_parity(r_new, r_ref, 1e-10, f"gpu ground {kutta}")
            _assert_close(r_new.CDi_trefftz, r_ref.CDi_trefftz, 1e-10, f"gpu ground {kutta} CDi")
    asym = FlightCondition(V_inf=50.0, alpha=np.radians(4.0), beta=np.radians(5.0), p=0.5)
    ref = PanelSolver(mesh, far_ratio=np.inf, kutta="pressure").solve(asym)
    new = PanelSolver(
        mesh, far_ratio=np.inf, kutta="pressure", backend="gpu", dtype=torch.float64
    ).solve(asym)
    _check_solver_parity(new, ref, 1e-10, "gpu pressure asymmetric")
