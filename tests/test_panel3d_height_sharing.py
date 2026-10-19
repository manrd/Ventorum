"""Exact wake reuse between ground heights, with deterministic work counts."""

from dataclasses import replace

import numpy as np
import pytest

from ventorum.airfoil import airfoil
from ventorum.core.datatypes import Aircraft, FlightCondition, LiftingSurface, SolverSettings, WingSection
from ventorum.core.errors import GroundStrikeError
from ventorum.panel3d import solver as module
from ventorum.panel3d.cache import solver_bytes
from ventorum.panel3d.mesh import mesh_lifting_surface
from ventorum.panel3d.solver import PanelSolver
from ventorum.solvers.panel3d import PanelMethodSolver


def _wing():
    """Return an aspect-ratio-eight wing with lengths in m."""
    return LiftingSurface(
        name="Wing", semi_span=4.0, is_symmetric=True,
        sections=[WingSection(y_frac=y, chord=1.0, shape="0012") for y in (0.0, 1.0)],
    )


def _solver(**kwargs):
    mesh = mesh_lifting_surface(
        _wing(), airfoil("0012").coords, n_chord=6, n_span=5, half=True,
    )
    return PanelSolver(mesh, backend=kwargs.pop("backend", "numba"), **kwargs)


def _map():
    """Return a height-major map with heights in m and angles in rad."""
    return [FlightCondition(V_inf=30.0, alpha=float(np.radians(a)), h=h)
            for h in (0.6, 0.8, 1.0) for a in (0.0, 2.0, 4.0)]


def _bits(actual, expected):
    for name in ("CL", "CD_pressure", "CDi_trefftz", "CY", "Cl", "Cm", "Cn",
                 "cp", "mu", "sigma", "velocity", "mu_w"):
        assert np.array_equal(getattr(actual, name), getattr(expected, name)), name


@pytest.mark.slow
@pytest.mark.parametrize("ground_solver", ["direct", "krylov"])
def test_height_map_bits(ground_solver):
    """Keep the Krylov anchor order by using one reference per height."""
    conditions = _map()
    solver = _solver(ground_solver=ground_solver)
    actual = solver.solve_conditions(conditions)
    if ground_solver == "direct":
        expected = [_solver().solve(c) for c in conditions]
    else:
        expected = []
        for start in range(0, len(conditions), 3):
            expected.extend(_solver(ground_solver="krylov").solve_conditions(
                conditions[start:start + 3],
            ))
        assert solver.factorization_count == 3
        assert not any(cfg.fallbacks for cfg in solver._configs.values())
    for a, b in zip(actual, expected, strict=True):
        _bits(a, b)
    assert [cfg.wake_shared for cfg in solver._configs.values()] == [False] * 3 + [True] * 6


@pytest.mark.slow
def test_height_map_counts_and_shared_arrays(monkeypatch):
    """Half meshes need 2*A wake builds and 2*A + 2*C strip calls.

    A is the number of distinct attitudes. C is the number of ground
    configurations. Without sharing the rules are 2*C and 4*C. Counts
    exclude the two body-axis wakes built by solver construction.
    """
    counts = {}
    results = {}
    for shared in (False, True):
        solver = _solver()
        calls = {"wake": 0, "strip": 0}
        snapshots = {}
        build, strip, config_strips = solver._build_wake, solver._strip_matrix, solver._config_strips

        def counted_build(*args, calls=calls, build=build):
            calls["wake"] += 1
            return build(*args)

        def counted_strip(*args, calls=calls, strip=strip):
            calls["strip"] += 1
            return strip(*args)

        def capture(cfg, config_strips=config_strips, solver=solver, snapshots=snapshots):
            config_strips(cfg)
            for entry in solver._attitude_cache.values():
                for a in entry.strips.values():
                    if id(a) not in snapshots:
                        snapshots[id(a)] = (a, a.copy())

        with monkeypatch.context() as patch:
            if not shared:
                patch.setattr(module, "_ATTITUDE_CACHE_BYTES", 0, raising=False)
            patch.setattr(solver, "_build_wake", counted_build)
            patch.setattr(solver, "_strip_matrix", counted_strip)
            if shared:
                patch.setattr(solver, "_config_strips", capture)
            results[shared] = solver.solve_conditions(_map())
        counts[shared] = calls
        if shared:
            assert len(snapshots) == 6
            for array, copy in snapshots.values():
                assert np.array_equal(array, copy)
            configs = list(solver._configs.values())
            for i in range(3):
                for j in (i + 3, i + 6):
                    assert configs[j].wake is configs[i].wake
                    assert configs[j].full_wake is configs[i].full_wake
                    assert configs[j].wake_panels_m is configs[i].wake_panels_m
            assert all(not cfg.strips for cfg in configs)
    assert counts[False] == {"wake": 18, "strip": 36}
    assert counts[True] == {"wake": 6, "strip": 24}
    print("height map counts before", counts[False], "after", counts[True])
    for a, b in zip(results[True], results[False], strict=True):
        _bits(a, b)


def test_lru_eviction_and_rebuild(monkeypatch):
    """A hit refreshes the LRU order; an evicted attitude is rebuilt."""
    solver = _solver()
    builds = []
    build = solver._build_wake

    def counted_build(*args):
        builds.append(1)
        return build(*args)

    monkeypatch.setattr(solver, "_build_wake", counted_build)
    one = 2 * len(solver.centroids) * len(solver.wake.wake_up) * 8
    monkeypatch.setattr(module, "_ATTITUDE_CACHE_BYTES", 2 * one)
    conditions = [_map()[i] for i in (0, 1, 3, 2, 7)]
    actual = []
    for c in conditions:
        actual.append(solver.solve(c))
        assert sum(a.nbytes for e in solver._attitude_cache.values()
                   for a in e.strips.values()) <= 2 * one
    assert len(solver._attitude_cache) == 2
    assert len(builds) == 8
    assert [cfg.wake_shared for cfg in solver._configs.values()] == [False, False, True, False, False]
    for a, c in zip(actual, conditions, strict=True):
        _bits(a, _solver().solve(c))
    # An old configuration can need its strips again for a new parity.
    c = replace(conditions[0], p=0.01)
    _bits(solver.solve(c), _solver().solve(c))


def test_oversized_entry_is_not_stored(monkeypatch):
    solver = _solver()
    monkeypatch.setattr(module, "_ATTITUDE_CACHE_BYTES", 1)
    conditions = [_map()[i] for i in (0, 3)]
    actual = solver.solve_conditions(conditions)
    assert not solver._attitude_cache
    assert not any(cfg.wake_shared for cfg in solver._configs.values())
    first, second = solver._configs.values()
    assert first.wake is not second.wake and first.full_wake is not second.full_wake
    for a, c in zip(actual, conditions, strict=True):
        _bits(a, _solver().solve(c))


def test_shared_wake_clearance_at_each_height(monkeypatch):
    """A synthetic low wake tip isolates the wake check from the body check."""
    original = PanelSolver._build_wake

    def low_tip(self, *args):
        wake = original(self, *args)
        if args[-1] is not None:
            wake.panels[0, 1, 2] = -0.5
        return wake

    monkeypatch.setattr(PanelSolver, "_build_wake", low_tip)
    solver = _solver()
    solver.solve(FlightCondition(V_inf=30.0, alpha=0.0, h=1.0))
    condition = FlightCondition(V_inf=30.0, alpha=0.0, h=0.3)
    with pytest.raises(GroundStrikeError, match="The wake goes below the ground plane") as shared:
        solver.solve(condition)
    with pytest.raises(GroundStrikeError) as fresh:
        _solver().solve(condition)
    assert str(shared.value) == str(fresh.value)


@pytest.mark.parametrize("kutta", ["morino", "pressure"])
def test_full_mesh_subsolver_shares_its_own_wake(kutta):
    conditions = [replace(_map()[i], beta=0.02) for i in (1, 4)]
    solver = _solver(kutta=kutta)
    actual = solver.solve_conditions(conditions)
    assert not solver._attitude_cache
    sub = solver._full_solver
    assert [cfg.wake_shared for cfg in sub._configs.values()] == [False, True]
    assert len(sub._attitude_cache) == 1
    assert set(next(iter(sub._attitude_cache.values())).strips) == {"self"}
    for a, c in zip(actual, conditions, strict=True):
        _bits(a, _solver(kutta=kutta).solve(c))


def test_pressure_shared_strips_stay_unchanged_on_later_solves():
    solver = _solver(kutta="pressure")
    conditions = [replace(_map()[i], p=0.01) for i in (1, 4)]
    first = solver.solve(conditions[0])
    entry = next(iter(solver._attitude_cache.values()))
    saved = {name: a.copy() for name, a in entry.strips.items()}
    second = solver.solve(conditions[1])
    repeat = solver.solve(conditions[0])
    for name, a in entry.strips.items():
        assert np.array_equal(a, saved[name])
    assert all(not cfg.strips for cfg in solver._configs.values())
    _bits(first, repeat)
    _bits(second, _solver(kutta="pressure").solve(conditions[1]))


def test_free_air_and_different_normals_do_not_share():
    solver = _solver()
    solver.solve(FlightCondition(V_inf=30.0))
    assert not solver._attitude_cache
    assert not next(iter(solver._configs.values())).wake_shared
    condition = FlightCondition(V_inf=30.0, alpha=0.0)
    grounds = [(np.array([0.0, 0.0, 1.0]), -1.0),
               (np.array([0.0, 0.1, 1.0]), -1.0)]
    full = PanelSolver(solver._full_mesh, backend="numba")
    full.solve_conditions([condition, condition], grounds=grounds)
    assert len(full._attitude_cache) == 2
    assert not any(cfg.wake_shared for cfg in full._configs.values())


def test_no_wake_panels_share_geometry_without_strips():
    from ventorum.panel3d.mesh import PanelMesh
    from ventorum.reference.analytic3d import sphere_quad_mesh

    nodes, panels = sphere_quad_mesh(0.1, 6, 8)
    mesh = PanelMesh(nodes=nodes, panels=panels, component=np.zeros(len(panels), dtype=int),
                     kind=np.full(len(panels), "surface"), wake_edges=np.empty((0, 4), dtype=int))
    solver = PanelSolver(mesh, backend="numba")
    conditions = [FlightCondition(V_inf=30.0, h=h) for h in (0.6, 0.8)]
    actual = solver.solve_conditions(conditions)
    first, second = solver._configs.values()
    assert not first.wake_shared and second.wake_shared
    assert first.wake is second.wake and first.full_wake is second.full_wake
    assert not next(iter(solver._attitude_cache.values())).strips
    for a, c in zip(actual, conditions, strict=True):
        _bits(a, PanelSolver(mesh, backend="numba").solve(c))


def test_solver_bytes_counts_shared_strips_once():
    solver = _solver()
    solver.solve_conditions(_map())
    entries = list(solver._attitude_cache.values())
    saved = [dict(e.strips) for e in entries]
    size = sum(a.nbytes for strips in saved for a in strips.values())
    assert size > 0 and all(not cfg.strips for cfg in solver._configs.values())
    after, device = solver_bytes(solver)
    for e in entries:
        e.strips.clear()
    before, _ = solver_bytes(solver)
    assert after - before == size and device == 0
    for e, strips in zip(entries, saved, strict=True):
        e.strips.update(strips)
    for cfg in solver._configs.values():
        cfg.strips.update(saved[0])
    assert solver_bytes(solver) == (after, device)


def test_factory_dispatch_reports_wake_sharing():
    settings = SolverSettings(n_chord=6, n_panels=5, panel_backend="numba", panel_cache=False)
    results = PanelMethodSolver().solve(Aircraft(surfaces=[_wing()]), _map(), settings)
    assert [r.details["dispatch"]["wake_shared"] for r in results] == [False] * 3 + [True] * 6


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_gpu_height_sharing_bits(dtype):
    torch = pytest.importorskip("torch")
    pytest.importorskip("warp")
    if not torch.cuda.is_available():
        pytest.skip("A CUDA GPU is not available.")
    solver = _solver(backend="gpu", dtype=dtype)
    conditions = [_map()[i] for i in (0, 1, 3, 4)]
    actual = solver.solve_conditions(conditions)
    assert [cfg.wake_shared for cfg in solver._configs.values()] == [False, False, True, True]
    for a, c in zip(actual, conditions, strict=True):
        _bits(a, _solver(backend="gpu", dtype=dtype).solve(c))


def test_pressure_group_computes_its_strips_once(monkeypatch):
    # Regression: the pressure Kutta must not drop and rebuild the strips per condition.
    solver = _solver(kutta="pressure")
    calls = []
    strip_matrix = solver._strip_matrix

    def counted(*args, **kwargs):
        calls.append(1)
        return strip_matrix(*args, **kwargs)

    monkeypatch.setattr(solver, "_strip_matrix", counted)
    base = _map()[1]
    group = [replace(base, p=p) for p in (0.0, 0.01, 0.02)]
    results = solver.solve_conditions(group)
    assert len(solver._configs) == 1
    # One set for the configuration: wake, mirror, ground image, mirror image.
    assert len(calls) == 4
    assert all(not cfg.strips for cfg in solver._configs.values())
    for res, cond in zip(results, group, strict=True):
        _bits(res, _solver(kutta="pressure").solve(cond))
