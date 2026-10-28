# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests of the CPU LU factorization backend of the 3D panel solver."""

from __future__ import annotations

import numpy as np
import pytest

from ventorum import FlightCondition, LiftingSurface, WingSection, gpu
from ventorum.airfoil import airfoil
from ventorum.hardware import profile as prof
from ventorum.hardware.detector import get_machine_fingerprint
from ventorum.panel3d.mesh import mesh_lifting_surface
from ventorum.panel3d.solver import PanelSolver


def _small_mesh(*, half: bool = True):
    """Return a small lifting surface mesh for unit tests."""
    wing = LiftingSurface(
        name="TestWing",
        semi_span=3.0,
        sections=[
            WingSection(y_frac=0.0, chord=1.0, shape="0012"),
            WingSection(y_frac=1.0, chord=0.8, shape="0012"),
        ],
    )
    coords = airfoil("0012").coords
    return mesh_lifting_surface(wing, coords, n_chord=8, n_span=6, half=half)


def _assert_parity(res_scipy, res_torch, *, rtol: float = 1e-10, atol: float = 1e-12):
    """Assert parity between SciPy and PyTorch solver results."""
    for key in ("CL", "Cm", "CD_pressure", "CDi_trefftz"):
        v_s = float(getattr(res_scipy, key))
        v_t = float(getattr(res_torch, key))
        scale = max(abs(v_s), abs(v_t), 1.0)
        diff = abs(v_s - v_t)
        assert diff <= max(atol, rtol * scale), f"{key}: scipy={v_s}, torch={v_t}, diff={diff}"

    cp_s = np.asarray(res_scipy.cp, dtype=float)
    cp_t = np.asarray(res_torch.cp, dtype=float)
    scale_cp = max(float(np.max(np.abs(cp_s))), 1.0)
    max_cp_diff = float(np.max(np.abs(cp_s - cp_t)))
    assert max_cp_diff <= max(atol, rtol * scale_cp), f"cp: max diff={max_cp_diff}"


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    """Provide an isolated configuration folder for machine profiles."""
    monkeypatch.setenv("VENTORUM_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("VENTORUM_DISABLE_AUTOTUNE", raising=False)
    prof.reset_profile_cache()
    yield tmp_path
    prof.reset_profile_cache()


def test_cpu_lu_direct_parity():
    """Verify parity of direct Morino solve between SciPy and PyTorch CPU LU."""
    pytest.importorskip("torch")
    mesh = _small_mesh(half=True)
    cond = FlightCondition(V_inf=35.0, alpha=np.radians(3.0))

    solver_scipy = PanelSolver(mesh, backend="numpy", cpu_lu="scipy")
    res_scipy = solver_scipy.solve(cond)

    solver_torch = PanelSolver(mesh, backend="numpy", cpu_lu="torch")
    res_torch = solver_torch.solve(cond)

    assert solver_scipy.cpu_lu == "scipy"
    assert solver_torch.cpu_lu == "torch"
    _assert_parity(res_scipy, res_torch)


def test_cpu_lu_direct_parity_numba():
    """Verify parity of direct Morino solve with the Numba backend."""
    pytest.importorskip("torch")
    pytest.importorskip("numba")
    mesh = _small_mesh(half=True)
    cond = FlightCondition(V_inf=30.0, alpha=np.radians(4.0))

    solver_scipy = PanelSolver(mesh, backend="numba", cpu_lu="scipy")
    res_scipy = solver_scipy.solve(cond)

    solver_torch = PanelSolver(mesh, backend="numba", cpu_lu="torch")
    res_torch = solver_torch.solve(cond)

    _assert_parity(res_scipy, res_torch)


def test_cpu_lu_pressure_kutta_parity():
    """Verify parity of pressure Kutta solve between SciPy and PyTorch CPU LU."""
    pytest.importorskip("torch")
    mesh = _small_mesh(half=True)
    cond = FlightCondition(V_inf=30.0, alpha=np.radians(4.0))

    solver_scipy = PanelSolver(mesh, backend="numpy", kutta="pressure", cpu_lu="scipy")
    res_scipy = solver_scipy.solve(cond)

    solver_torch = PanelSolver(mesh, backend="numpy", kutta="pressure", cpu_lu="torch")
    res_torch = solver_torch.solve(cond)

    assert res_scipy.kutta_converged
    assert res_torch.kutta_converged
    _assert_parity(res_scipy, res_torch)


def test_cpu_lu_krylov_ground_parity():
    """Verify parity of Krylov ground solve between SciPy and PyTorch CPU LU."""
    pytest.importorskip("torch")
    mesh = _small_mesh(half=True)
    cond1 = FlightCondition(V_inf=30.0, alpha=np.radians(2.0), h=1.5)
    cond2 = FlightCondition(V_inf=30.0, alpha=np.radians(4.0), h=1.5)

    solver_scipy = PanelSolver(mesh, backend="numpy", ground_solver="krylov", cpu_lu="scipy")
    res_scipy = solver_scipy.solve_conditions([cond1, cond2])

    solver_torch = PanelSolver(mesh, backend="numpy", ground_solver="krylov", cpu_lu="torch")
    res_torch = solver_torch.solve_conditions([cond1, cond2])

    assert len(res_scipy) == 2
    assert len(res_torch) == 2
    for r_s, r_t in zip(res_scipy, res_torch):
        _assert_parity(r_s, r_t)


def test_cpu_lu_krylov_preconditioner_accepts_torch_factors():
    """Verify that lockstep Krylov ground solver accepts PyTorch anchor factors."""
    pytest.importorskip("torch")
    mesh = _small_mesh(half=True)
    solver = PanelSolver(mesh, backend="numpy", ground_solver="krylov", cpu_lu="torch")
    conds = [
        FlightCondition(V_inf=30.0, alpha=np.radians(1.0), h=1.2),
        FlightCondition(V_inf=30.0, alpha=np.radians(3.0), h=1.2),
        FlightCondition(V_inf=30.0, alpha=np.radians(5.0), h=1.2),
    ]
    results = solver.solve_conditions(conds)
    assert len(results) == 3
    # Check that second and third conditions used Krylov with anchor factors
    cfg_list = list(solver._configs.values())
    assert len(cfg_list) == 3
    anchor_cfg = cfg_list[0]
    assert anchor_cfg.ground_anchor
    assert anchor_cfg.ground_solver == "direct"
    factors = anchor_cfg.systems.get("sym")
    assert factors is not None
    import torch

    assert isinstance(factors[0], torch.Tensor)
    iter_cfg = cfg_list[1]
    assert iter_cfg.ground_solver == "krylov"


def test_cpu_lu_validation():
    """Verify input validation for the private cpu_lu argument."""
    mesh = _small_mesh(half=True)
    with pytest.raises(ValueError, match="Unknown cpu_lu option"):
        PanelSolver(mesh, cpu_lu="invalid_option")


def test_cpu_lu_default_without_profile(config_dir):
    """Verify that SciPy is the default CPU LU backend without a profile."""
    mesh = _small_mesh(half=True)
    solver = PanelSolver(mesh)
    assert solver.cpu_lu == "scipy"


def test_cpu_lu_profile_read(config_dir):
    """Verify reading the cpu_lu key from machine profile."""
    pytest.importorskip("torch")
    data = {
        "schema": 4,
        "fingerprint": get_machine_fingerprint(),
        "settings": {"gpu": {"panel3d": {"cpu_lu": "torch"}}},
    }
    prof.save_profile(data)
    assert gpu.cpu_lu() == "torch"
    mesh = _small_mesh(half=True)
    solver = PanelSolver(mesh)
    assert solver.cpu_lu == "torch"


def test_cpu_lu_profile_read_flat_panel3d(config_dir):
    """Verify reading cpu_lu from settings.panel3d."""
    pytest.importorskip("torch")
    data = {
        "schema": 4,
        "fingerprint": get_machine_fingerprint(),
        "settings": {"panel3d": {"cpu_lu": "torch"}},
    }
    prof.save_profile(data)
    assert gpu.cpu_lu() == "torch"
    mesh = _small_mesh(half=True)
    solver = PanelSolver(mesh)
    assert solver.cpu_lu == "torch"


def test_cpu_lu_profile_fallback_on_invalid(config_dir):
    """Verify fallback to SciPy when profile has an unknown cpu_lu key."""
    data = {
        "schema": 4,
        "fingerprint": get_machine_fingerprint(),
        "settings": {"gpu": {"panel3d": {"cpu_lu": "unknown_backend"}}},
    }
    prof.save_profile(data)
    assert gpu.cpu_lu() == "scipy"
    mesh = _small_mesh(half=True)
    solver = PanelSolver(mesh)
    assert solver.cpu_lu == "scipy"


def test_cpu_lu_fallback_when_torch_unavailable(config_dir, monkeypatch):
    """Verify fallback to SciPy when PyTorch is not available."""
    import importlib.util

    real_find_spec = importlib.util.find_spec

    def mock_find_spec(name, *args, **kwargs):
        if name == "torch":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", mock_find_spec)
    data = {
        "schema": 4,
        "fingerprint": get_machine_fingerprint(),
        "settings": {"gpu": {"panel3d": {"cpu_lu": "torch"}}},
    }
    prof.save_profile(data)
    assert gpu.cpu_lu() == "scipy"


def test_cpu_lu_explicit_overrides_profile(config_dir):
    """Verify that an explicit cpu_lu argument overrides the profile."""
    data = {
        "schema": 4,
        "fingerprint": get_machine_fingerprint(),
        "settings": {"gpu": {"panel3d": {"cpu_lu": "torch"}}},
    }
    prof.save_profile(data)
    mesh = _small_mesh(half=True)
    solver = PanelSolver(mesh, cpu_lu="scipy")
    assert solver.cpu_lu == "scipy"


def test_cpu_lu_factor_budget_counts_torch_factors():
    """Verify that factor budget accurately counts PyTorch CPU factors."""
    pytest.importorskip("torch")
    mesh = _small_mesh(half=True)
    n_panels = len(mesh.panels)
    cond = FlightCondition(V_inf=30.0, alpha=np.radians(3.0))

    solver = PanelSolver(mesh, backend="numpy", kutta="pressure", cpu_lu="torch")
    solver.solve(cond)

    cfg = next(iter(solver._configs.values()))
    host_bytes, device_bytes = solver._factor_bytes(cfg)
    assert device_bytes == 0
    # Expected bytes for one parity system factor (LU float64 matrix + int32 pivots)
    expected_bytes = n_panels * n_panels * 8 + n_panels * 4
    assert host_bytes == expected_bytes


def test_cpu_lu_factor_trimming_drops_torch_factors(monkeypatch):
    """Verify that factor trimming drops PyTorch CPU factors under memory limit."""
    pytest.importorskip("torch")
    from ventorum.panel3d import solver as solver_mod

    mesh = _small_mesh(half=True)
    cond1 = FlightCondition(V_inf=30.0, alpha=np.radians(2.0), h=1.5)
    cond2 = FlightCondition(V_inf=30.0, alpha=np.radians(4.0), h=2.5)

    monkeypatch.setattr(solver_mod, "_FACTOR_BUDGET_HOST_FRACTION", 0.0)
    solver = PanelSolver(mesh, backend="numpy", kutta="pressure", cpu_lu="torch")
    solver.solve(cond1)
    solver.solve(cond2)

    assert solver._factors_dropped > 0


def test_parity_assertion_fails_on_mutation():
    """Verify that _assert_parity fails on a mutated result (mutation check)."""
    import dataclasses

    mesh = _small_mesh(half=True)
    cond = FlightCondition(V_inf=30.0, alpha=np.radians(3.0))
    solver = PanelSolver(mesh)
    res = solver.solve(cond)
    mutated = dataclasses.replace(res, CL=res.CL + 1e-5)
    with pytest.raises(AssertionError, match="CL"):
        _assert_parity(res, mutated)
