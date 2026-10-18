# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Cache of panel solvers across solve calls of the 3D panel method.

A repeated solve of the same geometry with the same settings must reuse the
mesh, the solver and its factorizations, and give the same bits as a solve
without the cache. Any change of the geometry or of the settings must miss.
"""

from __future__ import annotations

import copy
import dataclasses
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import scipy.linalg

import ventorum.panel3d
from ventorum.core.datatypes import (
    Aircraft,
    ControlSurface,
    FlightCondition,
    LiftingSurface,
    NodeDisplacements,
    SolverSettings,
    WingSection,
)
from ventorum.core.errors import GroundStrikeError
from ventorum.panel3d import cache
from ventorum.panel3d.solver import PanelSolver
from ventorum.solvers.panel3d import PanelMethodSolver

_N_SPAN = 5


def _wing(twist: float = 0.0, deflection: float = 0.0, shape: str = "0012") -> Aircraft:
    """Return a tapered wing with a flap (about 150 half-mesh panels)."""
    surf = LiftingSurface(
        name="wing",
        semi_span=2.0,
        sections=[
            WingSection(y_frac=0.0, chord=0.8, shape=shape),
            WingSection(y_frac=1.0, chord=0.5, twist=twist, shape=shape),
        ],
        controls=[ControlSurface(name="flap", eta_start=0.3, eta_end=0.7, hinge_x_c=0.75,
                                 deflection=deflection)],
        is_symmetric=True,
    )
    return Aircraft(name="cache wing", surfaces=[surf])


def _settings(**kw) -> SolverSettings:
    base = dict(n_panels=_N_SPAN, n_chord=6, panel_backend="numpy")
    base.update(kw)
    return SolverSettings(**base)


def _uncached(settings: SolverSettings) -> SolverSettings:
    out = settings.clone()
    out.panel_cache = False
    return out


def _cond(alpha_deg: float, **kw) -> FlightCondition:
    return FlightCondition(V_inf=30.0, alpha=float(np.radians(alpha_deg)), **kw)


def _solve(ac, cond, settings, **kw):
    return PanelMethodSolver().solve(ac, cond, settings, **kw)


def _assert_same_bits(a, b) -> None:
    """Assert that two factory results have the same bits (totals, arrays, strips, trust)."""
    for name in ("CL", "CDi", "CY", "Cl", "Cm", "Cn", "CDi_nearfield", "e"):
        x, y = getattr(a.totals, name), getattr(b.totals, name)
        assert np.array_equal(np.asarray(x), np.asarray(y), equal_nan=True), name
    pa, pb = a.details["panel_result"], b.details["panel_result"]
    for name in ("cp", "mu", "sigma", "velocity", "mu_w"):
        assert np.array_equal(getattr(pa, name), getattr(pb, name)), name
    assert np.array_equal(a.details["cp"], b.details["cp"])
    for sa, sb in zip(a.spanwise, b.spanwise, strict=True):
        for f in dataclasses.fields(sb):
            va, vb = getattr(sa, f.name), getattr(sb, f.name)
            if isinstance(vb, np.ndarray):
                assert np.array_equal(va, vb), f.name
            else:
                assert va == vb, f.name
    assert a.totals.trust.to_dict() == b.totals.trust.to_dict()
    assert a.details["notes"] == b.details["notes"]
    assert np.array_equal(a.details["mesh"].nodes, b.details["mesh"].nodes)
    assert np.array_equal(a.details["mesh"].panels, b.details["mesh"].panels)


@pytest.fixture(autouse=True)
def _empty_cache():
    cache.clear_cache()
    yield
    cache.clear_cache()


@pytest.fixture
def lu_calls(monkeypatch):
    """Count the dense LU factorizations of the CPU backends."""
    calls = []
    original = scipy.linalg.lu_factor

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(scipy.linalg, "lu_factor", counting)
    return calls


@pytest.mark.parametrize("kutta", ["morino", "pressure"])
def test_second_angle_reuses_the_solver(kutta, lu_calls):
    """A second solve at another angle reuses the solver: no new factorization, same bits."""
    st = _settings(panel_kutta=kutta)
    first = _solve(_wing(), _cond(2.0), st)
    assert first.details["dispatch"]["cached_solver"] is False
    assert first.details["dispatch"]["factorization_count"] == 1
    assert len(lu_calls) == 1
    assert cache.cache_info()["entries"] == 1

    # A new Aircraft object with the same geometry hits too.
    second = _solve(_wing(), _cond(5.0), st)
    assert second.details["dispatch"]["cached_solver"] is True
    assert second.details["dispatch"]["factorization_count"] == 0
    assert len(lu_calls) == 1

    reference = _solve(_wing(), _cond(5.0), _uncached(st))
    assert reference.details["dispatch"]["cached_solver"] is False
    assert reference.details["dispatch"]["factorization_count"] == 1
    assert len(lu_calls) == 2
    _assert_same_bits(second, reference)
    assert cache.cache_info()["entries"] == 1


def test_cached_sweep_with_rates_and_ground_equals_uncached(lu_calls):
    """Free-air rates reuse the factorization; a new ground configuration adds only its own."""
    st = _settings()
    _solve(_wing(), [_cond(0.0), _cond(3.0)], st)
    assert len(lu_calls) == 1
    conds = [_cond(1.0, q=0.3), _cond(4.0), _cond(2.0, h=0.6), _cond(2.0, h=0.9)]
    cached = _solve(_wing(), conds, st)
    assert all(r.details["dispatch"]["cached_solver"] for r in cached)
    assert all(r.details["dispatch"]["factorization_count"] == 2 for r in cached)
    assert len(lu_calls) == 3
    again = _solve(_wing(), conds[2:], st)
    assert all(r.details["dispatch"]["factorization_count"] == 0 for r in again)
    assert len(lu_calls) == 3
    reference = _solve(_wing(), conds, _uncached(st))
    for a, b in zip(cached, reference, strict=True):
        _assert_same_bits(a, b)
    for a, b in zip(again, reference[2:], strict=True):
        _assert_same_bits(a, b)


def test_sideslip_uses_its_own_full_mesh_solver():
    """Sideslip needs the full mesh: another key; the half-mesh solver stays cached."""
    st = _settings()
    _solve(_wing(), _cond(2.0), st)
    side = _solve(_wing(), _cond(2.0, beta=float(np.radians(3.0))), st)
    assert side.details["dispatch"]["cached_solver"] is False
    assert not side.symmetry_used
    assert cache.cache_info()["entries"] == 2
    side2 = _solve(_wing(), _cond(4.0, beta=float(np.radians(-2.0))), st)
    assert side2.details["dispatch"]["cached_solver"] is True
    assert side2.details["dispatch"]["factorization_count"] == 0
    _assert_same_bits(side2, _solve(_wing(), _cond(4.0, beta=float(np.radians(-2.0))), _uncached(st)))
    half = _solve(_wing(), _cond(6.0), st)
    assert half.details["dispatch"]["cached_solver"] is True


def _set_twist(ac):
    ac.surfaces[0].sections[1].twist = float(np.radians(-2.0))


def _set_flap(ac):
    ac.surfaces[0].controls[0].deflection = float(np.radians(5.0))


def _set_displacement(ac):
    n_edges = _N_SPAN + 1
    te = np.zeros((n_edges, 3))
    te[:, 2] = np.linspace(0.0, 0.01, n_edges)
    ac.surfaces[0].node_displacements = NodeDisplacements(le=np.zeros((n_edges, 3)), te=te)


def _set_shape(ac):
    for sec in ac.surfaces[0].sections:
        sec.shape = "2412"


def _set_shape_coords(ac):
    # Same NACA 0012, as coordinates: a section shape of another type.
    from ventorum.airfoil import airfoil

    coords = np.array(airfoil("0012").coords, copy=True)
    coords[:, 1] *= 1.05
    for sec in ac.surfaces[0].sections:
        sec.shape = coords


def _set_chord(ac):
    ac.surfaces[0].sections[0].chord = 0.85


@pytest.mark.parametrize("change", [_set_twist, _set_flap, _set_displacement, _set_shape, _set_shape_coords,
                                    _set_chord])
def test_geometry_change_misses(change):
    """A change of the geometry (also in place on the same object) misses the cache."""
    st = _settings()
    ac = _wing()
    _solve(ac, _cond(3.0), st)
    change(ac)
    changed = _solve(ac, _cond(3.0), st)
    assert changed.details["dispatch"]["cached_solver"] is False
    assert changed.details["dispatch"]["factorization_count"] == 1
    fresh = _wing()
    change(fresh)
    _assert_same_bits(changed, _solve(fresh, _cond(3.0), _uncached(st)))


@pytest.mark.parametrize("change", [
    dict(n_chord=8), dict(n_panels=6), dict(panel_chord_spacing="uniform"), dict(panel_kutta="pressure"),
    dict(panel_backend="numba"), dict(panel_wake_length=20.0), dict(panel_scale_root_thickness=False),
    dict(use_symmetry=False), dict(spacing="uniform"),
])
def test_settings_change_misses(change):
    """A change of a mesh or solver setting misses the cache."""
    _solve(_wing(), _cond(3.0), _settings())
    changed = _solve(_wing(), _cond(3.0), _settings(**change))
    assert changed.details["dispatch"]["cached_solver"] is False
    assert changed.details["dispatch"]["factorization_count"] == 1
    _assert_same_bits(changed, _solve(_wing(), _cond(3.0), _uncached(_settings(**change))))


def test_reference_point_and_values_miss():
    """A new moment reference point or new reference values miss the cache."""
    st = _settings()
    _solve(_wing(), _cond(3.0), st)
    moved = _solve(_wing(), _cond(3.0), st, ref_point=np.array([0.2, 0.0, 0.0]))
    assert moved.details["dispatch"]["cached_solver"] is False
    ac = _wing()
    ac.S_ref = 2.0
    scaled = _solve(ac, _cond(3.0), st)
    assert scaled.details["dispatch"]["cached_solver"] is False
    ref = _wing()
    ref.S_ref = 2.0
    _assert_same_bits(scaled, _solve(ref, _cond(3.0), _uncached(st)))


def test_replaced_mesh_builder_misses(monkeypatch):
    """A replaced module helper of the mesh data is part of the key (as in the lattice cache)."""
    from ventorum.solvers import panel3d as factory

    st = _settings()
    _solve(_wing(), _cond(3.0), st)
    original = factory.check_mesh
    calls = []

    def counted(mesh):
        calls.append(1)
        return original(mesh)

    monkeypatch.setattr(factory, "check_mesh", counted)
    res = _solve(_wing(), _cond(3.0), st)
    assert calls == [1]
    assert res.details["dispatch"]["cached_solver"] is False


def test_replaced_factorization_misses(monkeypatch):
    """A replaced factorization function is part of the key: a test that counts it sees a new solver."""
    st = _settings()
    _solve(_wing(), _cond(3.0), st)
    original = scipy.linalg.lu_factor
    calls = []

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(scipy.linalg, "lu_factor", counting)
    res = _solve(_wing(), _cond(3.0), st)
    assert calls == [1]
    assert res.details["dispatch"]["cached_solver"] is False
    again = _solve(_wing(), _cond(4.0), st)
    assert calls == [1]
    assert again.details["dispatch"]["cached_solver"] is True


def test_least_recently_used_solver_is_evicted():
    """At most MAX_SOLVERS solvers stay; the least recently used goes first."""
    assert cache.MAX_SOLVERS == 2
    st = _settings()
    a, b, c = _wing(), _wing(twist=0.02), _wing(twist=0.04)
    _solve(a, _cond(1.0), st)
    _solve(b, _cond(1.0), st)
    assert _solve(a, _cond(2.0), st).details["dispatch"]["cached_solver"] is True
    _solve(c, _cond(1.0), st)  # evicts b (a was used after b)
    assert cache.cache_info()["entries"] == 2
    assert _solve(a, _cond(3.0), st).details["dispatch"]["cached_solver"] is True
    assert _solve(c, _cond(3.0), st).details["dispatch"]["cached_solver"] is True
    assert _solve(b, _cond(3.0), st).details["dispatch"]["cached_solver"] is False


def test_byte_bound_evicts_and_frees(monkeypatch):
    """The host byte budget evicts solvers; an evicted solver drops its matrices."""
    st = _settings()
    _solve(_wing(), _cond(1.0), st)
    info = cache.cache_info()
    one = info["host_bytes"]
    assert info["entries"] == 1 and one > 0
    with cache._lock:
        first = next(iter(cache._entries.values())).solver
    assert first._configs and first._body_parts is not None
    # Room for one solver only: the second evicts the first.
    monkeypatch.setattr(cache, "MAX_BYTES", int(1.5 * one))
    _solve(_wing(twist=0.02), _cond(1.0), st)
    assert cache.cache_info()["entries"] == 1
    assert first._configs == {} and first._body_parts is None
    assert _solve(_wing(), _cond(1.0), st).details["dispatch"]["cached_solver"] is False
    # A budget smaller than one solver keeps nothing.
    monkeypatch.setattr(cache, "MAX_BYTES", 1)
    cache.clear_cache()
    _solve(_wing(), _cond(1.0), st)
    assert cache.cache_info()["entries"] == 0
    assert _solve(_wing(), _cond(2.0), st).details["dispatch"]["cached_solver"] is False


def test_solver_bytes_counts_the_matrices():
    """The byte count holds at least the dense matrices and the LU factors of the solver."""
    st = _settings()
    res = _solve(_wing(), _cond(1.0), st)
    with cache._lock:
        solver = next(iter(cache._entries.values())).solver
    n = len(res.details["cp"])
    host, device = cache.solver_bytes(solver)
    # B and C of the solved mesh and of its mirror, one LU factor.
    assert host >= 5 * n * n * 8
    assert device == 0


def test_clear_cache_empties_it():
    """ventorum.panel3d.clear_cache() removes every solver."""
    st = _settings()
    _solve(_wing(), _cond(1.0), st)
    _solve(_wing(twist=0.02), _cond(1.0), st)
    assert cache.cache_info()["entries"] == 2
    ventorum.panel3d.clear_cache()
    assert cache.cache_info() == {"entries": 0, "gpu_entries": 0, "in_use": 0, "host_bytes": 0, "device_bytes": 0}
    assert _solve(_wing(), _cond(1.0), st).details["dispatch"]["cached_solver"] is False


def test_cache_off_stores_nothing():
    """panel_cache=False builds a new solver each call; clone keeps the setting."""
    st = _settings(panel_cache=False)
    assert st.clone().panel_cache is False
    assert SolverSettings().panel_cache is True
    for _ in range(2):
        res = _solve(_wing(), _cond(1.0), st)
        assert res.details["dispatch"]["cached_solver"] is False
        assert res.details["dispatch"]["factorization_count"] == 1
    assert cache.cache_info()["entries"] == 0
    assert res.details["mesh"].nodes.flags.writeable


def test_returned_arrays_do_not_reach_the_cache():
    """Arrays of a result are copies or read-only: a change does not move the next result."""
    st = _settings()
    first = _solve(_wing(), _cond(3.0), st)
    mesh = first.details["mesh"]
    with pytest.raises(ValueError):
        mesh.nodes[0, 0] = 1.0
    with pytest.raises(ValueError):
        mesh.panels[0, 0] = 0
    # Change every writable array of the first result.
    first.details["cp"][:] = 7.0
    pr = first.details["panel_result"]
    for name in ("mu", "sigma", "velocity", "mu_w"):
        getattr(pr, name)[...] = 7.0
    for sw in first.spanwise:
        for f in dataclasses.fields(sw):
            val = getattr(sw, f.name)
            if isinstance(val, np.ndarray) and val.flags.writeable:
                val[...] = 7.0
    stats = first.details["mesh_stats"]
    for key, val in list(stats.items()):
        if isinstance(val, np.ndarray):
            val[...] = 7
        stats[key] = None
    first.details["notes"].append("changed")
    second = _solve(_wing(), _cond(3.0), st)
    assert second.details["dispatch"]["cached_solver"] is True
    reference = _solve(_wing(), _cond(3.0), _uncached(st))
    _assert_same_bits(second, reference)
    assert second.details["mesh_stats"].keys() == reference.details["mesh_stats"].keys()
    for key, val in reference.details["mesh_stats"].items():
        assert np.array_equal(np.asarray(second.details["mesh_stats"][key]), np.asarray(val)), key


def test_one_sided_surface_strip_arrays_are_copies():
    """A one-sided surface returns the cached strip chords: each call gets its own copy."""
    surf = LiftingSurface(name="fin", semi_span=1.5, is_symmetric=False, sections=[
        WingSection(y_frac=0.0, chord=0.7, shape="0012"), WingSection(y_frac=1.0, chord=0.4, shape="0012")])
    ac = Aircraft(surfaces=[surf])
    st = _settings()
    first = _solve(ac, _cond(3.0), st)
    chord = np.array(first.spanwise[0].chord, copy=True)
    first.spanwise[0].chord[...] = 9.0
    first.spanwise[0].y[...] = 9.0
    second = _solve(ac, _cond(3.0), st)
    assert second.details["dispatch"]["cached_solver"] is True
    assert np.array_equal(second.spanwise[0].chord, chord)
    _assert_same_bits(second, _solve(ac, _cond(3.0), _uncached(st)))


def test_failed_solve_drops_the_solver():
    """A solve that raises drops the cached solver it used."""
    st = _settings()
    _solve(_wing(), _cond(2.0), st)
    with pytest.raises(GroundStrikeError):
        _solve(_wing(), _cond(2.0, h=0.01), st)
    assert cache.cache_info()["entries"] == 0
    assert _solve(_wing(), _cond(2.0), st).details["dispatch"]["cached_solver"] is False


def test_two_threads_give_correct_results():
    """Two threads that solve the same geometry at the same time get correct bits."""
    st = _settings()
    alphas = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    reference = {a: _solve(_wing(), _cond(a), _uncached(st)) for a in alphas}
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda a: (a, _solve(_wing(), _cond(a), st)), alphas))
    for a, res in results:
        _assert_same_bits(res, reference[a])
    assert cache.cache_info()["entries"] == 1
    assert cache.cache_info()["in_use"] == 0


def test_solver_in_use_is_not_shared(monkeypatch):
    """While one thread uses the cached solver, the other builds its own (and does not store it)."""
    st = _settings()
    _solve(_wing(), _cond(1.0), st)
    barrier = threading.Barrier(2, timeout=60)
    original = PanelSolver.solve_conditions
    used = []

    def waiting(self, conditions, grounds=None):
        used.append(self)
        barrier.wait()
        return original(self, conditions, grounds)

    monkeypatch.setattr(PanelSolver, "solve_conditions", waiting)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda a: _solve(_wing(), _cond(a), st), [2.0, 3.0]))
    monkeypatch.setattr(PanelSolver, "solve_conditions", original)
    assert len({id(s) for s in used}) == 2
    assert sorted(r.details["dispatch"]["cached_solver"] for r in results) == [False, True]
    assert cache.cache_info()["entries"] == 1
    for a, res in zip([2.0, 3.0], results, strict=True):
        _assert_same_bits(res, _solve(_wing(), _cond(a), _uncached(st)))


def test_mesh_error_study_gets_correct_results():
    """The mesh-error study with the cache gives the bits of an uncached study."""
    st = _settings(panel_estimate_mesh_error=True)
    cond = _cond(3.0)
    _solve(_wing(), cond, _settings())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cached = _solve(_wing(), cond, st, ref_point=np.array([0.1, 0.0, 0.0]))
        reference = _solve(_wing(), cond, _uncached(st), ref_point=np.array([0.1, 0.0, 0.0]))
    _assert_same_bits(cached, reference)
    assert cached.details["mesh_convergence"] == reference.details["mesh_convergence"]


def test_cached_result_copies_are_deep():
    """Two results of one cached solver share no writable array."""
    st = _settings()
    a = _solve(_wing(), _cond(3.0), st)
    b = _solve(_wing(), _cond(3.0), st)
    saved = copy.deepcopy(b.details["panel_result"].cp)
    a.details["panel_result"].cp[...] = 0.0
    assert np.array_equal(b.details["panel_result"].cp, saved)
    assert b.details["mesh"] is a.details["mesh"]


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


@pytest.mark.gpu
@pytest.mark.skipif(not _cuda_available(), reason="needs a CUDA GPU")
def test_gpu_keeps_one_solver_and_frees_device_memory(monkeypatch):
    """At most one cached solver holds device arrays; eviction frees them."""
    st = _settings(panel_backend="gpu")
    reference = _solve(_wing(), _cond(5.0), _uncached(st))
    first = _solve(_wing(), _cond(2.0), st)
    assert first.details["dispatch"]["device"] == "gpu"
    info = cache.cache_info()
    assert info["gpu_entries"] == 1 and info["device_bytes"] > 0
    with cache._lock:
        gpu_solver = next(iter(cache._entries.values())).solver
    second = _solve(_wing(), _cond(5.0), st)
    assert second.details["dispatch"]["cached_solver"] is True
    assert second.details["dispatch"]["factorization_count"] == 0
    _assert_same_bits(second, reference)
    # A CPU solver can stay next to the GPU solver.
    _solve(_wing(), _cond(2.0), _settings())
    assert cache.cache_info()["entries"] == 2
    # A second GPU geometry evicts the first GPU solver and frees its arrays
    # before the new solve starts (the new solve gets the device memory).
    original = PanelSolver.solve_conditions
    state_at_solve = []

    def recording(self, conditions, grounds=None):
        state_at_solve.append((dict(gpu_solver._configs), gpu_solver._body_parts))
        return original(self, conditions, grounds)

    monkeypatch.setattr(PanelSolver, "solve_conditions", recording)
    _solve(_wing(twist=0.02), _cond(2.0), st)
    monkeypatch.setattr(PanelSolver, "solve_conditions", original)
    assert state_at_solve == [({}, None)]
    info = cache.cache_info()
    assert info["gpu_entries"] == 1 and info["entries"] == 2
    assert gpu_solver._configs == {} and gpu_solver._body_parts is None
    # A GPU solve without the cache frees the cached GPU solver too.
    _solve(_wing(twist=0.04), _cond(2.0), _uncached(st))
    assert cache.cache_info()["gpu_entries"] == 0
