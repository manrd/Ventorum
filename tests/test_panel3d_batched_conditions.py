# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Parity checks for batched panel conditions against the retained scalar path."""

from dataclasses import fields, replace

import numpy as np
import pytest

from ventorum.aero.system import freestream_direction, lift_direction
from ventorum.airfoil import airfoil
from ventorum.core.datatypes import Aircraft, FlightCondition, LiftingSurface, SolverSettings, WingSection
from ventorum.panel3d import solver as solver_module
from ventorum.panel3d.mesh import mesh_lifting_surface
from ventorum.panel3d.solver import PanelSolver
from ventorum.solvers.panel3d import PanelMethodSolver
from ventorum.solvers import panel3d as factory_module


def _wing():
    """Return a wing with lengths in m."""
    return LiftingSurface(
        name="Wing", semi_span=3.0, is_symmetric=True,
        sections=[WingSection(y_frac=e, chord=1.0, shape="2412") for e in (0.0, 1.0)],
    )


def _solver(**kwargs):
    """Return a small half-mesh solver."""
    mesh = mesh_lifting_surface(
        _wing(), airfoil("2412").coords, n_chord=6, n_span=5, half=True,
    )
    return PanelSolver(mesh, **kwargs)


def _conditions():
    """Return nine conditions with angles in rad and speed in m/s."""
    return [FlightCondition(V_inf=30.0, alpha=float(a)) for a in np.radians(np.arange(-3, 6))]


def _close(new, old, tol=1e-12):
    """Check the array relative rule with an absolute floor near zero."""
    a, b = np.asarray(new), np.asarray(old)
    assert a.shape == b.shape
    if not b.size:
        return
    scale = float(np.max(np.abs(b)))
    assert np.max(np.abs(a - b)) <= max(tol * scale, 1e-14)


def _panel_equal(new, old, tol=1e-12, bits=False):
    """Check every field, including the condition and Kutta metadata."""
    for f in fields(old):
        a, b = getattr(new, f.name), getattr(old, f.name)
        if isinstance(b, (float, np.ndarray)):
            if bits:
                assert np.array_equal(a, b), f.name
            else:
                try:
                    _close(a, b, tol)
                except AssertionError as exc:
                    raise AssertionError(f.name) from exc
        else:
            assert a == b, f.name


def _scalar_results(solver, conditions):
    """Evaluate each prepared configuration with the old helpers."""
    results = []
    for c in conditions:
        key, direction, ground = solver._config_key(c, None)
        target = solver
        if solver.mesh.half and not solver._symmetric_config(direction, ground):
            target = solver._full_mesh_solver()
            key, direction, ground = target._config_key(c, None)
        cfg = target._configs.get(key)
        if cfg is None:
            cfg = target._new_config(key, direction, ground)
            target._configs[key] = cfg
        target._ensure(cfg, target._parities_needed([target._condition_coeffs(c)]))
        evaluate = target._evaluate_result if target.kutta == "morino" else target._evaluate_pressure_result
        result = evaluate(cfg, c)
        results.append(solver._to_half(result) if target is not solver else result)
    return results


@pytest.mark.slow
@pytest.mark.parametrize("backend", ["numpy", "numba"])
@pytest.mark.parametrize("half", [True, False])
def test_sweep_matches_single_conditions_cpu(backend, half, monkeypatch):
    """Check panel fields, factory totals and each strip array to round-off."""
    conditions = _conditions()
    solver = _solver(backend=backend)
    batch = solver.solve_conditions(conditions)
    reference = _scalar_results(solver, conditions)
    for a, b in zip(batch, reference):
        _panel_equal(a, b)

    aircraft = Aircraft(surfaces=[_wing()])
    settings = SolverSettings(
        n_chord=6, n_panels=5, panel_backend=backend, use_symmetry=half,
    )
    factory = PanelMethodSolver()
    results = factory.solve(aircraft, conditions, settings)

    def scalar_batch(self, cfg, conds, coeffs):
        return [self._evaluate_result(cfg, c) for c in conds]

    monkeypatch.setattr(PanelSolver, "_evaluate_results_batch", scalar_batch)

    def scalar_strips(solver, results, conditions, maps, meta):
        return [factory_module._spanwise_reference(solver, r, c, maps, meta)
                for r, c in zip(results, conditions)]

    monkeypatch.setattr(factory_module, "_spanwise_results_batch", scalar_strips)
    singles = [factory.solve(aircraft, c, settings) for c in conditions]
    for a, b in zip(results, singles):
        for f in fields(b.totals):
            if f.name != "trust" and getattr(b.totals, f.name) is not None:
                _close(getattr(a.totals, f.name), getattr(b.totals, f.name))
        assert a.totals.trust.to_dict() == b.totals.trust.to_dict()
        for x, y in zip(a.spanwise, b.spanwise):
            for f in fields(y):
                if isinstance(getattr(y, f.name), np.ndarray):
                    _close(getattr(x, f.name), getattr(y, f.name))
        _check_scalar_strips(a)


def _check_scalar_strips(result):
    """Integrate each strip independently with the former force and arm formulas."""
    from ventorum.geometry.lattice import _surface_eta, resolve_spacing, surface_reference_line
    from ventorum.panel3d.kernels import _precompute_geometry
    from ventorum.solvers.panel3d import _surface_strip_maps

    mesh, c = result.details["mesh"], result.condition
    surf = _wing()
    eta, mid = _surface_eta(surf, 5, resolve_spacing("auto", surf, "vlm"))
    right, left = _surface_strip_maps(mesh, surf, 5, eta, 0, mesh.half)
    geom = _precompute_geometry(mesh.nodes[mesh.panels])
    q = 0.5 * c.rho * c.V_inf**2
    force = -result.details["cp"][:, None] * geom.area[:, None] * geom.normal * q
    x, y, z = surface_reference_line(surf, mid)
    pos = np.column_stack([x, y, z])
    width = np.diff(eta) * surf.semi_span
    sw = result.spanwise[0]
    for side, maps in enumerate((right, left)):
        if maps is None:
            continue
        for k, ids in enumerate(maps):
            j = k if side == 0 else 4 - k
            qc = pos[j].copy()
            qc[0] += 0.25
            if side:
                qc[1] *= -1.0
            f = np.sum(force[ids], axis=0)
            m = np.sum(np.cross(geom.centroid[ids] - qc, force[ids]), axis=0)
            idx = 5 + k if side == 0 else k
            _close(sw.Cl[idx], np.dot(f, lift_direction(c.alpha)) / (q * width[j]))
            _close(sw.Cd_i[idx], np.dot(f, freestream_direction(c.alpha, c.beta)) / (q * width[j]))
            _close(sw.Cm_section[idx], m[1] / (q * width[j]))


def test_mixed_rates_and_sideslip_in_one_batch():
    """Different rates in rad/s and sideslip angles in rad share one batch."""
    solver = _solver(ref_point=np.array([0.2, 0.0, 0.1]))
    conditions = [replace(c, beta=0.01 * i, p=0.02 * i, q=-0.03 * i, r=0.01 * i,
                          V_inf=20.0 + i, rho=1.0 + i * 0.01)
                  for i, c in enumerate(_conditions())]
    results = solver.solve_conditions(conditions)
    assert solver.factorization_count == 2
    for a, b in zip(results, _scalar_results(solver, conditions)):
        _panel_equal(a, b)


def test_ground_configurations_keep_order():
    """Shuffled free-air and ground configurations preserve condition order."""
    conditions = _conditions()
    conditions[1] = replace(conditions[1], h=0.8)
    conditions[3] = replace(conditions[3], h=0.6, beta=0.02)
    conditions[5] = replace(conditions[1], V_inf=35.0, q=0.1)
    conditions = [conditions[i] for i in (5, 0, 3, 7, 1, 8, 2, 6, 4)]
    solver = _solver()
    batch = solver.solve_conditions(conditions)
    for c, a, b in zip(conditions, batch, _scalar_results(solver, conditions)):
        assert a.condition is c
        _panel_equal(a, b)


def test_chunking_gives_same_results(monkeypatch):
    """A small memory budget processes nine conditions in three chunks."""
    solver = _solver()
    conditions = _conditions()
    reference = solver.solve_conditions(conditions)
    monkeypatch.setattr(solver_module, "_CONDITION_BATCH_BYTES", 3 * solver._condition_bytes())
    calls = []
    original = solver._evaluate_results_batch

    def counted(cfg, conds, coeffs):
        calls.append(len(conds))
        return original(cfg, conds, coeffs)

    monkeypatch.setattr(solver, "_evaluate_results_batch", counted)
    batch = solver.solve_conditions(conditions)
    assert calls == [3, 3, 3]
    for a, b in zip(batch, reference):
        _panel_equal(a, b)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_batched_gpu_matches_old_gpu(dtype, monkeypatch):
    """Compare both GPU precisions with the old device evaluation."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("warp")
    if not torch.cuda.is_available():
        pytest.skip("A CUDA GPU is not available.")
    solver = _solver(backend="gpu", dtype=dtype)
    conditions = [replace(c, beta=0.02, p=0.03, q=-0.02, r=0.01) for c in _conditions()]
    reference = _scalar_results(solver, conditions)
    for a, b in zip(solver.solve_conditions(conditions), reference):
        _panel_equal(a, b, 1e-5 if dtype == "float32" else 1e-12)
    monkeypatch.setattr(solver_module, "_CONDITION_BATCH_BYTES", 3 * solver._condition_bytes())
    for a, b in zip(solver.solve_conditions(conditions), reference):
        _panel_equal(a, b, 1e-5 if dtype == "float32" else 1e-12)


def test_pressure_kutta_unchanged(monkeypatch):
    """Pressure Kutta keeps the per-condition iteration and every result bit."""
    solver = _solver(kutta="pressure")
    conditions = _conditions()[:3]
    reference = _scalar_results(solver, conditions)

    def forbidden(*args):
        raise AssertionError("Pressure Kutta must keep its scalar iteration.")

    monkeypatch.setattr(solver, "_evaluate_results_batch", forbidden)
    for a, b in zip(solver.solve_conditions(conditions), reference):
        _panel_equal(a, b, bits=True)


@pytest.mark.parametrize("backend", ["numpy", pytest.param("gpu", marks=pytest.mark.gpu)])
def test_batch_without_wake_and_with_zero_speed(backend):
    """A closed body has no wake; zero speed keeps zero pressure and loads."""
    if backend == "gpu":
        torch = pytest.importorskip("torch")
        pytest.importorskip("warp")
        if not torch.cuda.is_available():
            pytest.skip("A CUDA GPU is not available.")
    from ventorum.panel3d.mesh import PanelMesh
    from ventorum.reference.analytic3d import sphere_quad_mesh

    nodes, panels = sphere_quad_mesh(1.0, 6, 8)
    mesh = PanelMesh(nodes=nodes, panels=panels, component=np.zeros(len(panels), dtype=int),
                     kind=np.full(len(panels), "surface"), wake_edges=np.empty((0, 4), dtype=int))
    solver = PanelSolver(mesh, backend=backend, dtype="float64", S_ref=0.0)
    conditions = [FlightCondition(V_inf=v, alpha=0.1) for v in (0.0, 10.0)]
    for a, b in zip(solver.solve_conditions(conditions), _scalar_results(solver, conditions)):
        _panel_equal(a, b)
        assert a.CDi_trefftz == 0.0


@pytest.mark.parametrize("backend, dtype", [
    ("numpy", None), ("numba", None),
    pytest.param("gpu", "float32", marks=pytest.mark.gpu),
    pytest.param("gpu", "float64", marks=pytest.mark.gpu),
])
def test_batch_size_keeps_each_result_bit(backend, dtype):
    """Regression: batch-size round-off changed exact mesh-error study reports."""
    if backend == "gpu":
        torch = pytest.importorskip("torch")
        pytest.importorskip("warp")
        if not torch.cuda.is_available():
            pytest.skip("A CUDA GPU is not available.")
    solver = _solver(backend=backend, dtype=dtype)
    conditions = [replace(c, V_inf=22.0 + i, beta=0.02, p=0.03, q=-0.02, r=0.01)
                  for i, c in enumerate(_conditions())]
    batch = solver.solve_conditions(conditions)
    for result, c in zip(batch, conditions):
        _panel_equal(result, solver.solve(c), bits=True)
