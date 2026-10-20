# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Check automatic ground solves, direct exclusions and anchor work."""

import dataclasses

import numpy as np
import pytest
import scipy.linalg

from ventorum.core.datatypes import SolverSettings
from ventorum.panel3d.solver import PanelSolver
from ventorum.solvers.panel3d import PanelMethodSolver

from test_panel3d_krylov_ground import _conditions, _mesh, _panel_parity, _wing


def _settings(**options):
    return SolverSettings(n_panels=5, n_chord=6, panel_cache=False, **options)


def _sweep():
    return [c for c in _conditions() if c.alpha in np.radians([1.0, 2.0, 3.0])]


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_default_ground_sweep_dispatch_parity_and_counters(backend, monkeypatch):
    """Three angles at two heights use two anchors and four iterative solves."""
    assert SolverSettings().panel_ground_solver == "auto"
    assert SolverSettings().clone().panel_ground_solver == "auto"
    assert PanelSolver(_mesh()).ground_solver == "auto"
    calls = []
    original = scipy.linalg.lu_factor

    def factor(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(scipy.linalg, "lu_factor", factor)
    method = PanelMethodSolver()
    direct = method.solve(_wing(), _sweep(), _settings(
        panel_backend=backend, panel_ground_solver="direct",
    ))
    assert len(calls) == 6
    calls.clear()
    results = method.solve(_wing(), _sweep(), _settings(panel_backend=backend))
    assert len(calls) == 2
    for i, (actual, reference) in enumerate(zip(results, direct, strict=True)):
        anchor = i in (0, 3)
        _panel_parity(actual.details["panel_result"], reference.details["panel_result"],
                      bits=anchor)
        for a, b in zip(actual.spanwise, reference.spanwise, strict=True):
            for field in dataclasses.fields(b):
                expected = getattr(b, field.name)
                if isinstance(expected, np.ndarray):
                    np.testing.assert_allclose(getattr(a, field.name), expected,
                                               rtol=1e-9, atol=1e-12)
        dispatch = actual.details["dispatch"]
        assert dispatch["ground_solver"] == ("direct" if anchor else "krylov")
        assert dispatch["ground_solver_reason"] == (
            "auto: direct anchor" if anchor else "auto: krylov in ground effect"
        )
        assert dispatch["factorization_count"] == 2
        assert not dispatch["fallbacks"]
        if anchor:
            assert set(dispatch["iterations"].values()) == {0}
            assert set(dispatch["relative_residuals"].values()) == {None}
            assert dispatch["matrix_passes"] == 0
        else:
            assert max(dispatch["iterations"].values()) > 0
            assert all(r <= 1e-10 for r in dispatch["relative_residuals"].values())
            assert dispatch["matrix_passes"] == max(dispatch["iterations"].values()) + 2
    for mode, rows in (("direct", direct), ("auto", results)):
        records = [r.details["dispatch"] for r in rows]
        print(backend, mode, "factorizations", records[0]["factorization_count"],
              "iterations", [list(r["iterations"].values()) for r in records],
              "matrix_passes", [r["matrix_passes"] for r in records])


@pytest.mark.parametrize("backend", ["numpy", "numba"])
@pytest.mark.parametrize("exclusion, reason", [
    ("free", "auto: direct in free air"),
    ("pressure", "auto: direct with pressure Kutta"),
])
def test_auto_cpu_exclusions_are_direct(backend, exclusion, reason):
    """Excluded configurations keep the direct result bits and a neutral reason."""
    conditions = _sweep()[:2]
    options = {"panel_backend": backend}
    if exclusion == "free":
        conditions = [dataclasses.replace(c, h=None) for c in conditions]
    else:
        options["panel_kutta"] = "pressure"
    method = PanelMethodSolver()
    results = method.solve(_wing(), conditions, _settings(**options))
    references = method.solve(_wing(), conditions, _settings(
        panel_ground_solver="direct", **options,
    ))
    for actual, reference in zip(results, references, strict=True):
        _panel_parity(actual.details["panel_result"], reference.details["panel_result"], bits=True)
        assert actual.details["dispatch"]["ground_solver"] == "direct"
        assert actual.details["dispatch"]["ground_solver_reason"] == reason


@pytest.mark.gpu
def test_auto_gpu_exclusion_is_direct():
    """The automatic GPU solve keeps direct LU and its result bits."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    method = PanelMethodSolver()
    results = method.solve(_wing(), _sweep()[:2], _settings(panel_backend="gpu"))
    references = method.solve(_wing(), _sweep()[:2], _settings(
        panel_backend="gpu", panel_ground_solver="direct",
    ))
    for actual, reference in zip(results, references, strict=True):
        _panel_parity(actual.details["panel_result"], reference.details["panel_result"], bits=True)
        assert actual.details["dispatch"]["ground_solver"] == "direct"
        assert actual.details["dispatch"]["ground_solver_reason"] == "auto: direct on gpu backend"


class _WorkMatrix(np.ndarray):
    """Count products with the assembled work matrix, including a copied view."""

    def __array_finalize__(self, source):
        self.products = getattr(source, "products", [])

    def __matmul__(self, other):
        self.products.append(1)
        return np.asarray(self) @ other


@pytest.mark.parametrize("mode", ["auto", "direct", "krylov"])
@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_ground_anchor_has_no_copy_or_residual_product(mode, backend, monkeypatch):
    """The LU overwrites the assembled work matrix without diagnostic products."""
    solver = PanelSolver(_mesh(), backend=backend, ground_solver=mode)
    matrices, copies, factors = [], [], []
    add, copy, factor = solver._add, solver._copy, solver._factorize

    def counted_add(*args):
        matrix = add(*args).view(_WorkMatrix)
        matrices.append(matrix)
        return matrix

    def counted_copy(matrix):
        copies.append(1)
        return copy(matrix)

    def counted_factor(matrix):
        assert matrix is matrices[-1]
        lu = factor(matrix)
        assert np.shares_memory(lu[0], matrix)
        factors.append(1)
        return lu

    monkeypatch.setattr(solver, "_add", counted_add)
    monkeypatch.setattr(solver, "_copy", counted_copy)
    monkeypatch.setattr(solver, "_factorize", counted_factor)
    solver.solve(_sweep()[0])
    assert len(matrices) == len(factors) == 1
    assert copies == []
    assert matrices[0].products == []
    cfg = next(iter(solver._configs.values()))
    assert cfg.ground_solver == "direct"
    assert set(cfg.iterations.values()) == {0}
    assert set(cfg.relative_residuals.values()) == {None}


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_auto_height_map_has_only_direct_anchors(backend):
    """One angle at each height needs no GMRES and keeps direct result bits."""
    conditions = [dataclasses.replace(_sweep()[0], h=h) for h in (0.5, 0.75, 1.0)]
    solver = PanelSolver(_mesh(), backend=backend)
    results = solver.solve_conditions(conditions)
    references = PanelSolver(_mesh(), backend=backend, ground_solver="direct").solve_conditions(conditions)
    assert solver.factorization_count == 3
    for actual, reference, cfg in zip(results, references, solver._configs.values(), strict=True):
        _panel_parity(actual, reference, bits=True)
        assert cfg.ground_anchor and cfg.ground_solver == "direct"
        assert set(cfg.iterations.values()) == {0}
        assert set(cfg.relative_residuals.values()) == {None}
        assert cfg.matrix_passes == 0 and not cfg.fallbacks


def test_unknown_ground_solver_names_all_values():
    """Both public entry points list all allowed values on an invalid option."""
    with pytest.raises(ValueError, match="auto.*direct.*krylov"):
        PanelSolver(_mesh(), ground_solver="unknown")
    with pytest.raises(ValueError, match="auto.*direct.*krylov"):
        PanelMethodSolver().solve(_wing(), _sweep()[0], _settings(
            panel_backend="numpy", panel_ground_solver="unknown",
        ))
