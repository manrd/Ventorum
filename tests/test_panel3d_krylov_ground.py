# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Check the opt-in ground GMRES solve and its direct fallback."""

from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import pytest
import scipy.linalg
import scipy.sparse.linalg

from ventorum.airfoil import airfoil
from ventorum.core.datatypes import Aircraft, FlightCondition, LiftingSurface, SolverSettings, WingSection
from ventorum.panel3d import cache
from ventorum.panel3d.mesh import mesh_lifting_surface
from ventorum.panel3d.solver import PanelSolver
from ventorum.solvers.panel3d import PanelMethodSolver


def _wing():
    return Aircraft(surfaces=[LiftingSurface(
        name="wing", semi_span=2.0, sections=[
            WingSection(y_frac=0.0, chord=1.0, shape="2412"),
            WingSection(y_frac=1.0, chord=0.7, shape="2412"),
        ], is_symmetric=True,
    )])


def _mesh():
    return mesh_lifting_surface(
        _wing().surfaces[0], airfoil("2412", closed_te=True).coords,
        n_chord=6, n_span=5, half=True,
    )


def _conditions():
    return [FlightCondition(V_inf=25.0, alpha=float(a), h=h)
            for h in (0.5, 1.0) for a in np.radians(np.arange(1.0, 10.0))]


def _parity(actual, expected, *, bits=False):
    if bits:
        assert np.array_equal(actual, expected, equal_nan=True)
    else:
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12)


def _panel_parity(actual, expected, *, bits=False):
    for name in ("CL", "CD_pressure", "CDi_trefftz", "CY", "Cl", "Cm", "Cn",
                 "cp", "mu", "sigma", "velocity", "mu_w"):
        _parity(getattr(actual, name), getattr(expected, name), bits=bits)


@pytest.fixture(autouse=True)
def _clear_cache():
    cache.clear_cache()
    yield
    cache.clear_cache()


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_ground_sweep_parity_and_counters(backend, monkeypatch):
    """Nine angles at two heights reuse two LUs and preserve all result fields."""
    calls = []
    original = scipy.linalg.lu_factor

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(scipy.linalg, "lu_factor", counted)
    settings = SolverSettings(n_panels=5, n_chord=6, panel_backend=backend, panel_cache=False)
    method = PanelMethodSolver()
    direct = method.solve(_wing(), _conditions(), settings)
    assert len(calls) == 18
    settings.panel_ground_solver = "krylov"
    calls.clear()
    krylov = method.solve(_wing(), _conditions(), settings)
    assert len(calls) == 2
    repeat = method.solve(_wing(), _conditions(), settings)
    for i, (a, b, c) in enumerate(zip(krylov, direct, repeat, strict=True)):
        anchor = i in (0, 9)
        _panel_parity(a.details["panel_result"], b.details["panel_result"], bits=anchor)
        _panel_parity(a.details["panel_result"], c.details["panel_result"], bits=True)
        for sa, sb, sc in zip(a.spanwise, b.spanwise, c.spanwise, strict=True):
            for f in dataclasses.fields(sb):
                if isinstance(getattr(sb, f.name), np.ndarray):
                    _parity(getattr(sa, f.name), getattr(sb, f.name), bits=anchor)
                    _parity(getattr(sa, f.name), getattr(sc, f.name), bits=True)
        disp = a.details["dispatch"]
        assert disp["ground_solver"] == ("direct" if anchor else "krylov")
        assert disp["factorization_count"] == 2
        assert b.details["dispatch"]["factorization_count"] == 18
        assert disp["fallbacks"] == []
        assert all(r <= 1e-10 for r in disp["relative_residuals"].values())
        assert all(0 <= n <= 100 for n in disp["iterations"].values())
        if not anchor:
            assert max(disp["iterations"].values()) > 0
        print(backend, i, disp["ground_solver"], disp["iterations"])
    print(backend, "factorizations: direct=18 krylov=2")


@pytest.mark.parametrize("failure", ["cap", "true_residual"])
def test_fallback_replaces_anchor_and_releases_old_lu(monkeypatch, failure):
    """Failure of any onset makes the whole configuration a direct anchor."""
    import ventorum.panel3d.solver as module

    solver = PanelSolver(_mesh(), ground_solver="krylov")
    conditions = _conditions()[:3]
    solver.solve(conditions[0])
    if failure == "cap":
        monkeypatch.setattr(module, "_GROUND_GMRES_MAXITER", 1)
    else:
        monkeypatch.setattr(scipy.sparse.linalg, "gmres",
                            lambda op, rhs, **kw: (np.zeros_like(rhs), 0))
    result = solver.solve(conditions[1])
    reference = PanelSolver(_mesh()).solve(conditions[1])
    _panel_parity(result, reference, bits=True)
    first, second = solver._configs.values()
    assert second.ground_solver == "direct"
    assert second.fallbacks
    assert solver.factorization_count == 2
    assert first.systems == {"sym": None}
    assert second.systems["sym"] is not None
    assert not first.strips and not second.strips
    monkeypatch.undo()
    solver.solve(conditions[2])
    assert solver.factorization_count == 2
    assert list(solver._configs.values())[-1].ground_solver == "krylov"
    # Cached unit solutions still work after their factors are dropped.
    _panel_parity(solver.solve(conditions[0]), PanelSolver(_mesh()).solve(conditions[0]), bits=True)


def test_full_mesh_sideslip_and_incremental_parities():
    """Ground sideslip reaches the full sub-solver with the same option."""
    conditions = [dataclasses.replace(c, beta=0.04, p=0.01, r=0.02) for c in _conditions()[:3]]
    direct = PanelSolver(_mesh())
    solver = PanelSolver(_mesh(), ground_solver="krylov")
    for a, b in zip(solver.solve_conditions(conditions), direct.solve_conditions(conditions), strict=True):
        _panel_parity(a, b)
    assert solver._full_solver.ground_solver == "krylov"
    assert solver.factorization_count == 1
    assert list(solver._full_solver._configs.values())[-1].ground_solver == "krylov"
    # Add an antisymmetric unit onset to a known symmetric configuration.
    solver = PanelSolver(_mesh(), ground_solver="krylov")
    solver.solve_conditions(_conditions()[:2])
    cond = dataclasses.replace(_conditions()[1], p=0.01)
    _panel_parity(solver.solve(cond), PanelSolver(_mesh()).solve(cond))
    assert solver.factorization_count == 2


def test_defaults_cache_clone_and_free_air():
    """The option is validated, cloned, keyed, and inactive in free air."""
    settings = SolverSettings(n_panels=5, n_chord=6, panel_backend="numpy")
    assert settings.panel_ground_solver == "direct"
    settings.panel_ground_solver = "krylov"
    assert settings.clone().panel_ground_solver == "krylov"

    @dataclasses.dataclass
    class Settings(SolverSettings):
        extra: int = 1

    assert Settings(panel_ground_solver="krylov").clone().panel_ground_solver == "krylov"
    cond = _conditions()[0]
    method = PanelMethodSolver()
    a = method.solve(_wing(), cond, settings)
    settings.panel_ground_solver = "direct"
    b = method.solve(_wing(), cond, settings)
    assert not b.details["dispatch"]["cached_solver"]
    _panel_parity(a.details["panel_result"], b.details["panel_result"], bits=True)
    assert method.solve(_wing(), cond, settings).details["dispatch"]["cached_solver"]
    free = [dataclasses.replace(c, h=None) for c in _conditions()[:3]]
    solver = PanelSolver(_mesh(), ground_solver="krylov")
    reference = PanelSolver(_mesh())
    for x, y in zip(solver.solve_conditions(free), reference.solve_conditions(free), strict=True):
        _panel_parity(x, y, bits=True)
    assert solver.factorization_count == 1
    assert next(iter(solver._configs.values())).ground_solver == "direct"
    with pytest.raises(ValueError, match="ground_solver"):
        PanelSolver(_mesh(), ground_solver="bad")
    settings.panel_ground_solver = "bad"
    with pytest.raises(ValueError, match="panel_ground_solver"):
        method.solve(_wing(), cond, settings)


def test_pressure_kutta_keeps_direct():
    """Pressure Kutta records why it uses direct LU."""
    settings = SolverSettings(n_panels=5, n_chord=6, panel_backend="numpy",
                              panel_kutta="pressure", panel_ground_solver="krylov")
    result = PanelMethodSolver().solve(_wing(), _conditions()[0], settings)
    assert result.details["dispatch"]["ground_solver"] == "direct"
    assert result.details["dispatch"]["ground_solver_reason"] == "krylov not used: pressure Kutta"
    settings.panel_ground_solver = "direct"
    reference = PanelMethodSolver().solve(_wing(), _conditions()[0], settings)
    _panel_parity(result.details["panel_result"], reference.details["panel_result"], bits=True)


def test_right_preconditioner_zero_rhs_and_legacy_api(monkeypatch):
    """The operator applies the inverse on the right, including older SciPy APIs."""
    solver = PanelSolver(_mesh(), ground_solver="krylov")
    matrix = np.array([[3.0, 1.0], [0.5, 2.0]])
    anchor = np.array([[2.0, 0.0], [1.0, 1.0]])
    lu = scipy.linalg.lu_factor(anchor)
    rhs = np.array([1.0, 2.0])
    original = scipy.sparse.linalg.gmres
    calls = []

    def legacy(op, b, *, tol, **kwargs):
        vector = np.array([2.0, -1.0])
        np.testing.assert_allclose(op @ vector, matrix @ np.linalg.solve(anchor, vector))
        assert kwargs["restart"] == 50 and kwargs["maxiter"] == 100
        assert kwargs["callback_type"] == "legacy"
        calls.append(1)
        tolerance = "rtol" if "rtol" in inspect.signature(original).parameters else "tol"
        return original(op, b, **{tolerance: tol}, **kwargs)

    monkeypatch.setattr(scipy.sparse.linalg, "gmres", legacy)
    solution, count, residual, ok = solver._ground_gmres(matrix, lu, rhs, np.ones(2))
    np.testing.assert_allclose(solution, np.linalg.solve(matrix, rhs), rtol=1e-10)
    assert ok and count > 0 and residual <= 1e-10
    solution, count, residual, ok = solver._ground_gmres(matrix, lu, np.zeros(2), np.ones(2))
    assert np.array_equal(solution, np.zeros(2))
    assert ok and count == 0 and residual == 0.0
    assert calls == [1]


@pytest.mark.gpu
def test_gpu_keeps_direct():
    """The GPU keeps its device LU and records the reason."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    settings = SolverSettings(n_panels=5, n_chord=6, panel_backend="gpu",
                              panel_ground_solver="krylov")
    results = PanelMethodSolver().solve(_wing(), _conditions()[:2], settings)
    settings.panel_ground_solver = "direct"
    references = PanelMethodSolver().solve(_wing(), _conditions()[:2], settings)
    for result, reference in zip(results, references, strict=True):
        _panel_parity(result.details["panel_result"], reference.details["panel_result"], bits=True)
        disp = result.details["dispatch"]
        assert disp["ground_solver"] == "direct"
        assert disp["ground_solver_reason"] == "krylov not used: gpu backend"
        assert disp["factorization_count"] == 2


@pytest.mark.parametrize("ref_point", [None, np.array([0.25, 0.0, 0.0])])
def test_one_anchor_per_height_with_any_reference_point(ref_point):
    """Configurations at one height share one anchor LU, also with a moment point off the origin.

    The plane offset of a condition-based ground changes with the attitude
    when the moment reference point is not at the origin; the height of that
    point above the ground does not. A sweep of 9 angles at 2 heights needs 2
    factorizations, no fallback, and stays within 1e-9 of the direct solve.
    """
    aircraft = _wing()
    conds = [FlightCondition(V_inf=30.0, alpha=float(np.radians(a)), h=h)
             for h in (0.6, 1.0) for a in np.linspace(0.0, 8.0, 9)]
    out = {}
    for solver in ("direct", "krylov"):
        settings = SolverSettings(n_chord=8, n_panels=8, panel_backend="numpy",
                                  panel_ground_solver=solver, panel_cache=False)
        out[solver] = PanelMethodSolver().solve(aircraft, conds, settings, ref_point=ref_point)
    krylov = out["krylov"]
    assert krylov[0].details["dispatch"]["factorization_count"] == 2
    assert not any(r.details["dispatch"]["fallbacks"] for r in krylov)
    assert [r.details["dispatch"]["ground_solver"] for r in krylov].count("krylov") == 16
    for a, b in zip(krylov, out["direct"]):
        assert abs(a.totals.CL - b.totals.CL) <= 1e-9 * max(abs(b.totals.CL), 1e-3)
        cp_scale = max(float(np.max(np.abs(b.details["cp"]))), 1.0)
        assert float(np.max(np.abs(a.details["cp"] - b.details["cp"]))) <= 1e-9 * cp_scale


@pytest.mark.slow
def test_converged_gmres_does_not_fall_back():
    """A converged GMRES solve passes the true-residual check (no needless fallback).

    GMRES stops on its preconditioned residual; with the inner limit equal to
    the true-residual limit, the farthest angle of each height failed the
    final check by a hair and fell back to a direct factorization on this
    10 by 12 mesh. The inner limit is a decade lower.
    """
    aircraft = Aircraft(name="AR8", surfaces=[LiftingSurface(
        name="Wing", semi_span=4.0, is_symmetric=True,
        sections=[WingSection(y_frac=0.0, chord=1.0, shape="0012"),
                  WingSection(y_frac=1.0, chord=1.0, shape="0012")],
    )])
    aircraft.compute_reference_values()
    conds = [FlightCondition(V_inf=30.0, alpha=float(np.radians(a)), h=h)
             for h in (0.6, 1.0) for a in np.linspace(0.0, 8.0, 9)]
    settings = SolverSettings(n_chord=10, n_panels=12, panel_backend="numba",
                              panel_ground_solver="krylov", panel_cache=False)
    results = PanelMethodSolver().solve(aircraft, conds, settings)
    assert results[0].details["dispatch"]["factorization_count"] == 2
    assert not any(r.details["dispatch"]["fallbacks"] for r in results)
