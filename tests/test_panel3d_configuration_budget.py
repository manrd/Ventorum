"""Check bounded panel configurations and factor storage without timing."""

from dataclasses import replace

import numpy as np
import pytest

from ventorum.airfoil import airfoil
from ventorum.core.datatypes import Aircraft, FlightCondition, LiftingSurface, SolverSettings, WingSection
from ventorum.hardware import detector
from ventorum.panel3d import cache
from ventorum.panel3d import solver as module
from ventorum.panel3d.mesh import mesh_lifting_surface
from ventorum.panel3d.solver import PanelSolver
from ventorum.solvers.panel3d import PanelMethodSolver


def _wing():
    return LiftingSurface(name="Wing", semi_span=2.0, is_symmetric=True, sections=[
        WingSection(y_frac=0.0, chord=1.0, shape="2412"),
        WingSection(y_frac=1.0, chord=0.7, shape="2412"),
    ])


@pytest.fixture
def mesh():
    return mesh_lifting_surface(
        _wing(), airfoil("2412", closed_te=True).coords, n_chord=6, n_span=5, half=True,
    )


@pytest.fixture(autouse=True)
def empty_cache():
    cache.clear_cache()
    yield
    cache.clear_cache()


def _map():
    return [FlightCondition(V_inf=25.0, alpha=a, h=h)
            for h in (0.6, 0.9, 1.2) for a in (0.02, 0.04, 0.06)]


def _equal(actual, expected, *, bits=True):
    for name in ("CL", "CD_pressure", "CDi_trefftz", "CY", "Cl", "Cm", "Cn",
                 "cp", "mu", "sigma", "velocity", "mu_w", "kutta_residual"):
        a, b = getattr(actual, name), getattr(expected, name)
        if bits:
            assert np.array_equal(a, b), name
        else:
            # The same Krylov rule as the ground solver regression tests.
            np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12, err_msg=name)


def _pool(solver):
    host = device = 0
    for cfg in solver._configs.values():
        for factors in cfg.systems.values():
            for array in factors or ():
                if isinstance(array, np.ndarray):
                    host += array.nbytes
                elif array.device.type == "cpu":
                    host += array.numel() * array.element_size()
                else:
                    device += array.numel() * array.element_size()
    return host, device


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_direct_map_bits_and_repeat(mesh, backend):
    solver = PanelSolver(mesh, backend=backend, ground_solver="direct")
    conditions = _map()
    actual = solver.solve_conditions(conditions)
    assert len(solver._configs) == 9
    assert solver.factorization_count == 9
    assert solver._factors_dropped == 9
    assert _pool(solver) == (0, 0)
    for a, c in zip(actual, conditions, strict=True):
        _equal(a, PanelSolver(mesh, backend=backend, ground_solver="direct").solve(c))
    _equal(solver.solve(conditions[0]), actual[0])
    assert solver.factorization_count == 9
    assert solver._factors_dropped == 0
    assert all(v is None for cfg in solver._configs.values() for v in cfg.systems.values())
    # Free air also drops factors. A later antisymmetric onset adds only one LU.
    free = replace(conditions[0], h=None)
    solver.solve(free)
    asymmetric = replace(free, p=0.01)
    _equal(solver.solve(asymmetric), PanelSolver(mesh, backend=backend, ground_solver="direct").solve(asymmetric))
    assert solver.factorization_count == 11
    assert _pool(solver) == (0, 0)


def test_map_memory_and_solver_cache(mesh, monkeypatch):
    """The removed LU and pivot buffers reduce solver_bytes by their exact sizes."""
    solver = PanelSolver(mesh, ground_solver="direct")
    retained = PanelSolver(mesh, kutta="pressure")
    # Keep the same Morino evaluation and arrays, but retain factors as before.
    monkeypatch.setattr(retained, "_evaluate_pressure_result", lambda cfg, c: retained._evaluate_results_batch(
        cfg, [c], module._condition_arrays([c])[0],
    )[0])
    retained.solve_conditions(_map())
    solver.solve_conditions(_map())
    before_pool = _pool(retained)
    after_pool = _pool(solver)
    before = cache.solver_bytes(retained)
    after = cache.solver_bytes(solver)
    assert before_pool[0] > 0 and before_pool[1] == 0
    assert after_pool == (0, 0)
    assert before[0] - after[0] == before_pool[0]
    assert before[1] == after[1] == 0
    print("map pool bytes before/after:", before_pool, after_pool)
    print("map solver_bytes before/after:", before, after)
    # Put the cache bound between the two measured sizes. The new solver fits.
    monkeypatch.setattr(cache, "MAX_BYTES", (before[0] + after[0]) // 2)
    geometry = cache.CachedGeometry(mesh, [], [], len(mesh.panels), {})
    lease = cache.acquire(("budget",), None, geometry, lambda: solver, gpu=False)
    cache.release(lease, ok=True)
    assert cache.cache_info()["entries"] == 1


def test_anchors_keep_factors_and_protect_running_anchor(mesh, monkeypatch):
    solver = PanelSolver(mesh, ground_solver="krylov")
    conditions = _map()[:3]
    solver.solve(conditions[0])
    anchor = next(iter(solver._configs.values()))
    assert anchor.ground_anchor and _pool(solver)[0] > 0
    # No allocation is needed for the next angle. Its running anchor must
    # survive a zero budget until the group has used the preconditioner.
    monkeypatch.setattr(module, "_FACTOR_BUDGET_HOST_FRACTION", 0.0)
    original = solver._ground_gmres_lockstep

    def checked(matrix, factors, *args):
        assert anchor.ground_anchor and factors is anchor.systems["sym"]
        return original(matrix, factors, *args)

    monkeypatch.setattr(solver, "_ground_gmres_lockstep", checked)
    actual = solver.solve(conditions[1])
    _equal(actual, PanelSolver(mesh).solve(conditions[1]), bits=False)
    assert solver.factorization_count == 1
    assert not anchor.ground_anchor and _pool(solver) == (0, 0)
    monkeypatch.setattr(solver, "_ground_gmres_lockstep", original)
    actual = solver.solve(conditions[2])
    _equal(actual, PanelSolver(mesh).solve(conditions[2]))
    assert solver.factorization_count == 2


def test_factor_lru_replaces_evicted_height_anchor(mesh, monkeypatch):
    solver = PanelSolver(mesh, ground_solver="krylov")
    conditions = [_map()[i] for i in (0, 3, 6)]
    solver.solve(conditions[0])
    one_factor = _pool(solver)[0]
    monkeypatch.setattr(detector, "_get_total_ram_gb", lambda: 2 * one_factor / (0.25 * 2**30))
    solver.solve(conditions[1])
    first, second = solver._configs.values()
    solver.solve(conditions[0])  # Refresh the first height in the factor LRU.
    solver.solve(conditions[2])
    assert first.ground_anchor and first.systems["sym"] is not None
    assert not second.ground_anchor and second.systems["sym"] is None
    assert solver._factors_dropped == 1
    next_angle = replace(conditions[1], alpha=0.04)
    _equal(solver.solve(next_angle), PanelSolver(mesh).solve(next_angle))
    assert solver._configs[solver._config_key(next_angle, None)[0]].ground_anchor
    assert solver.factorization_count == 4
    # A transient free-air LU does not enter the retained factor pool.
    held = _pool(solver)
    solver.solve(replace(conditions[0], h=None))
    assert _pool(solver) == held and solver._factors_dropped == 1


def test_running_anchor_survives_incremental_parity_at_zero_budget(mesh, monkeypatch):
    solver = PanelSolver(mesh, ground_solver="krylov")
    solver.solve_conditions(_map()[:2])
    anchor = next(iter(solver._configs.values()))
    original = solver._ground_gmres_lockstep
    monkeypatch.setattr(module, "_FACTOR_BUDGET_HOST_FRACTION", 0.0)

    def checked(matrix, factors, *args):
        assert anchor.ground_anchor
        assert all(anchor.systems[p] is not None for p in ("sym", "asym"))
        assert factors is anchor.systems["asym"]
        return original(matrix, factors, *args)

    monkeypatch.setattr(solver, "_ground_gmres_lockstep", checked)
    condition = replace(_map()[1], p=0.01)
    _equal(solver.solve(condition), PanelSolver(mesh).solve(condition), bits=False)
    assert solver.factorization_count == 2
    assert solver._factors_dropped == 2 and _pool(solver) == (0, 0)
    assert not anchor.ground_anchor


def test_pressure_factors_refactorize_with_identical_bits(mesh, monkeypatch):
    solver = PanelSolver(mesh, kutta="pressure")
    reference = PanelSolver(mesh, kutta="pressure")
    conditions = _map()[:2]
    expected = reference.solve_conditions(conditions)
    one_factor = _pool(reference)[0] // 2
    monkeypatch.setattr(detector, "_get_total_ram_gb", lambda: one_factor / (0.25 * 2**30))
    original = solver._factorize

    def checked(matrix):
        # The previous factors must be gone before the next LU allocation.
        assert _pool(solver) == (0, 0)
        return original(matrix)

    monkeypatch.setattr(solver, "_factorize", checked)
    actual = solver.solve_conditions(conditions)
    for a, b in zip(actual, expected, strict=True):
        _equal(a, b)
    assert solver.factorization_count == 2 and solver._factors_dropped == 1
    _equal(solver.solve(conditions[0]), expected[0])
    assert solver.factorization_count == 3 and solver._factors_dropped == 1
    monkeypatch.setattr(solver, "_factorize", original)
    # A group needs both parities together, even when they exceed the budget.
    monkeypatch.setattr(module, "_FACTOR_BUDGET_HOST_FRACTION", 0.0)
    free = FlightCondition(V_inf=25.0, alpha=0.04, p=0.01)
    _equal(solver.solve(free), reference.solve(free))
    assert solver.factorization_count == 5 and solver._factors_dropped == 3
    assert _pool(solver) == (0, 0)
    _equal(solver.solve(free), reference.solve(free))
    assert solver.factorization_count == 7 and solver._factors_dropped == 2


def test_cached_krylov_units_can_become_a_new_direct_anchor(mesh, monkeypatch):
    solver = PanelSolver(mesh, ground_solver="krylov")
    conditions = _map()[:3]
    solver.solve_conditions(conditions[:2])
    cfg = solver._configs[solver._config_key(conditions[1], None)[0]]
    assert cfg.ground_solver == "krylov"
    monkeypatch.setattr(module, "_FACTOR_BUDGET_HOST_FRACTION", 0.0)
    solver.solve(conditions[0])  # Release the old anchor after its group.
    monkeypatch.setattr(module, "_FACTOR_BUDGET_HOST_FRACTION", 0.25)
    asymmetric = replace(conditions[1], p=0.01)
    actual = solver.solve(asymmetric)
    _equal(actual, PanelSolver(mesh).solve(asymmetric))
    assert cfg.ground_anchor and cfg.ground_solver == "direct"
    assert cfg.systems["sym"] is not None and cfg.systems["asym"] is not None
    assert all(n == 0 for n in cfg.iterations.values())
    assert solver.factorization_count == 3


def test_configuration_lru_bound_and_rebuild(mesh, monkeypatch):
    monkeypatch.setattr(module, "_MAX_CONFIGURATIONS", 2)
    solver = PanelSolver(mesh, ground_solver="direct")
    conditions = [replace(_map()[0], h=None), *_map()[:3]]
    expected = [PanelSolver(mesh, ground_solver="direct").solve(c) for c in conditions]
    actual = solver.solve_conditions(conditions)
    # The bound applies at the start of the next call: the running call keeps its 4.
    assert len(solver._configs) == 4 and solver._configurations_evicted == 0
    for a, b in zip(actual, expected, strict=True):
        _equal(a, b)
    solver.solve(conditions[2])  # trims to the 2 most recent (conditions 2 and 3), then reuses 2
    assert solver._configurations_evicted == 2 and len(solver._configs) == 2
    _equal(solver.solve(conditions[0]), expected[0])  # rebuilt; 2 kept, so no trim at its start
    assert solver._configurations_evicted == 0 and len(solver._configs) == 3
    assert solver.factorization_count == 5
    _equal(solver.solve(conditions[2]), expected[2])  # trims the least recent (condition 3)
    assert solver._configurations_evicted == 1
    assert solver._config_key(conditions[3], None)[0] not in solver._configs
    assert solver.factorization_count == 5


def test_host_budget_is_read_for_each_call(mesh, monkeypatch):
    values = iter((4.0, 0.0, 1.0))
    monkeypatch.setattr(detector, "_get_total_ram_gb", lambda: next(values))
    solver = PanelSolver(mesh, kutta="pressure")
    condition = _map()[0]
    first = solver.solve(condition)
    assert solver._factor_budget_host == int(4 * 2**30 * 0.25)
    _equal(solver.solve(condition), first)
    assert solver._factor_budget_host == 0
    assert solver.factorization_count == 1 and solver._factors_dropped == 1
    assert _pool(solver) == (0, 0)
    assert solver.solve_conditions([]) == []
    assert solver._factor_budget_host == int(2**30 * 0.25)
    assert solver._factors_dropped == solver._configurations_evicted == 0


def test_factor_byte_accounting(mesh):
    cfg = PanelSolver(mesh)._new_config(("free",), None, None)
    cfg.systems["sym"] = (np.zeros((3, 3)), np.zeros(3, dtype=np.int32))
    expected = 3 * 3 * 8 + 3 * 4
    assert PanelSolver._factor_bytes(cfg) == (expected, 0)
    torch = pytest.importorskip("torch")
    cfg.systems["asym"] = (torch.zeros((3, 3), dtype=torch.float64),
                           torch.zeros(3, dtype=torch.int32))
    assert PanelSolver._factor_bytes(cfg) == (2 * expected, 0)


@pytest.mark.parametrize("full", [False, True])
def test_factory_dispatch_counters(monkeypatch, full):
    monkeypatch.setattr(module, "_MAX_CONFIGURATIONS", 2)
    conditions = _map()[:2]
    if full:
        conditions = [replace(c, beta=0.02) for c in conditions]
    settings = SolverSettings(n_chord=6, n_panels=5, panel_backend="numpy", panel_ground_solver="direct")
    method, aircraft = PanelMethodSolver(), Aircraft(surfaces=[_wing()])
    first = method.solve(aircraft, conditions, settings)
    for result in first:
        assert result.details["dispatch"]["factors_dropped"] == 2
        assert result.details["dispatch"]["configurations_evicted"] == 0
    repeat = method.solve(aircraft, conditions[0], settings)
    assert repeat.details["dispatch"]["factors_dropped"] == 0
    assert repeat.details["dispatch"]["factorization_count"] == 0
    third = replace(conditions[0], alpha=0.06)
    added = method.solve(aircraft, third, settings)
    # The bound applies at the start of the next call.
    assert added.details["dispatch"]["configurations_evicted"] == 0
    evicted = method.solve(aircraft, conditions[1], settings)
    assert evicted.details["dispatch"]["configurations_evicted"] == 1


def test_factory_metadata_survives_configuration_eviction(mesh, monkeypatch):
    """Factory dispatch stays available when a call has more configurations than the bound.

    The bound applies at the start of the next call, so the factory can read
    the records of every configuration of the running call.
    """
    monkeypatch.setattr(module, "_MAX_CONFIGURATIONS", 2)
    settings = SolverSettings(n_chord=6, n_panels=5, panel_backend="numpy", panel_cache=False)
    actual = PanelMethodSolver().solve(Aircraft(surfaces=[_wing()]), _map()[:4], settings)
    assert len(actual) == 4
    assert all(r.details["dispatch"]["configurations_evicted"] == 0 for r in actual)
    assert all("ground_solver" in r.details["dispatch"] for r in actual)
    solver = PanelSolver(mesh, backend="numpy")
    solver.solve_conditions(_map()[:4])
    assert len(solver._configs) == 4
    # The next call first trims to the bound (4 - 2 = 2 evictions), then adds its own configuration.
    solver.solve_conditions(_map()[4:5])
    assert solver._configurations_evicted == 2
    assert len(solver._configs) == 3


@pytest.mark.gpu
def test_gpu_map_memory_counters(mesh, monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    solver = PanelSolver(mesh, backend="gpu")
    retained = PanelSolver(mesh, backend="gpu", kutta="pressure")
    monkeypatch.setattr(retained, "_evaluate_pressure_result", lambda cfg, c: retained._evaluate_results_batch(
        cfg, [c], module._condition_arrays([c])[0],
    )[0])
    retained.solve_conditions(_map())
    solver.solve_conditions(_map())
    before_pool, after_pool = _pool(retained), _pool(solver)
    before, after = cache.solver_bytes(retained), cache.solver_bytes(solver)
    assert before_pool[0] == 0 and before_pool[1] > 0
    assert after_pool == (0, 0)
    assert tuple(a - b for a, b in zip(before, after, strict=True)) == before_pool
    print("gpu map pool bytes before/after:", before_pool, after_pool)
    print("gpu map solver_bytes before/after:", before, after)


@pytest.mark.gpu
def test_gpu_map_bits_drops_and_budget_reads(mesh, monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    solver = PanelSolver(mesh, backend="gpu")
    actual = solver.solve_conditions(_map())
    for a, c in zip(actual, _map(), strict=True):
        _equal(a, PanelSolver(mesh, backend="gpu").solve(c))
    assert solver._factors_dropped == 9 and _pool(solver) == (0, 0)
    assert all(cfg.unit_mu_t.is_cuda for cfg in solver._configs.values())
    before = solver.factorization_count
    _equal(solver.solve(_map()[0]), actual[0])
    assert solver.factorization_count == before
    # Query the pool budget independently of the unchanged GPU fit check.
    retained = PanelSolver(mesh, backend="gpu", kutta="pressure")
    retained.solve(_map()[0])
    held = _pool(retained)[1]
    assert held > 0
    memory_query = torch.cuda.mem_get_info
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (10000, 20000))
    retained.solve(_map()[0])
    assert retained._factor_budget_device == 5000 + held
    monkeypatch.setattr(torch.cuda, "mem_get_info", memory_query)
    monkeypatch.setattr(module, "_FACTOR_BUDGET_DEVICE_FRACTION", 0.0)
    # A new solver starts with no pool. Zero still permits a running group.
    zero = PanelSolver(mesh, backend="gpu", kutta="pressure")
    expected = retained.solve(_map()[0])
    _equal(zero.solve(_map()[0]), expected)
    assert zero._factor_budget_device == 0 and _pool(zero) == (0, 0)
    assert zero._factors_dropped == 1
    _equal(zero.solve(_map()[0]), expected)
    assert zero.factorization_count == 2 and zero._factors_dropped == 1
    assert _pool(zero) == (0, 0)
