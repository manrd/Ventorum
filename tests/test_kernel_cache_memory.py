# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests of the memory of the kernel cache of a sweep.

The results of a sweep keep the lattice (``details["lattice"]``), but not
the kernel cache of the sweep (:func:`ventorum.aero.vortex.kernel_cache_scope`).
The checks use weak references and ``gc.collect``, not timing. The values
of the results do not change: a test compares the bits of the coefficients
with those of the sweep that keeps the kernel cache on its lattice.
"""

from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest

import ventorum as vt
from ventorum.aero import vortex
from ventorum.geometry import lattice_cache
from ventorum.solvers.factory import make_solver

ALPHAS = np.arange(-2.0, 8.0, 2.0)


def _wing(span: float = 5.0) -> vt.LiftingSurface:
    return vt.LiftingSurface(name="w", semi_span=span,
                             sections=[vt.WingSection(y_frac=0.0, chord=2.0),
                                       vt.WingSection(y_frac=1.0, chord=1.0, x_le=0.8)])


def _bits(results) -> list:
    return [(float(r.totals.CL).hex(), float(r.totals.CDi).hex(), float(r.totals.Cm).hex()) for r in results]


@pytest.fixture
def tracked_kernel_caches(monkeypatch):
    """Record a weak reference to each kernel cache that kernel_cache_scope makes, and whether it was filled."""
    refs: list = []

    class Tracked(vortex.KernelCache):
        __slots__ = ("filled",)

        def __init__(self):
            super().__init__()
            self.filled = False
            refs.append(weakref.ref(self))

        def __setitem__(self, key, value):
            self.filled = True
            super().__setitem__(key, value)

    monkeypatch.setattr(vortex, "KernelCache", Tracked)
    return refs


def _old_sweep_bits(span: float = 5.0) -> list:
    """Bits of the sweep with the kernel cache on the lattice that the results keep (the path before the change)."""
    ac = vt.Aircraft(surfaces=[_wing(span)])
    ac.compute_reference_values()
    settings = vt.SolverSettings(solver_type="vlm", n_panels=20)
    solver = make_solver("vlm")
    rp = ac.moment_reference()
    lattice = solver.build(ac, settings, vt.FlightCondition(V_inf=50.0), None, rp)
    lattice.kernel_cache = {}
    conds = [vt.FlightCondition(V_inf=50.0, alpha=float(np.radians(a))) for a in ALPHAS]
    res = solver.solve_batch(lattice, conds, settings, ac.S_ref, ac.b_ref, ac.c_ref, ref_point=rp,
                             main_surface=ac.main_surface_index())
    return _bits(res)


# ---------------------------------------------------------------------------
# Kernel cache of a sweep
# ---------------------------------------------------------------------------


def test_kernel_cache_scope_restores_the_old_value():
    holder = type("Holder", (), {})()   # any object with a kernel_cache attribute
    holder.kernel_cache = None
    with vortex.kernel_cache_scope(holder) as cache:
        assert holder.kernel_cache is cache
        assert isinstance(cache, vortex.KernelCache)
        weakref.ref(cache)   # the cache accepts weak references
    assert holder.kernel_cache is None
    old = {"x": 1}
    holder.kernel_cache = old
    with pytest.raises(RuntimeError), vortex.kernel_cache_scope(holder):
        raise RuntimeError("stop")
    assert holder.kernel_cache is old


def test_sweep_results_do_not_keep_the_kernel_cache(tracked_kernel_caches):
    """A kept sweep result does not keep the kernel cache alive; its values and its lattice stay."""
    lattice_cache.clear()
    res = vt.analyze_sweep(_wing(), ALPHAS, solver="vlm", n_panels=20, n_jobs=1)
    gc.collect()
    assert len(tracked_kernel_caches) == 1
    ref = tracked_kernel_caches[0]
    assert ref() is None, "the results keep the kernel cache of the sweep alive"
    lat = res[0].details["lattice"]
    assert lat.kernel_cache is None
    assert all(r.details["lattice"] is lat for r in res)
    # The result outlives the cache entry: values and lattice geometry stay.
    before = _bits(res)
    n_panels, cp = lat.n_panels, np.array(lat.cp)
    lattice_cache.clear()
    gc.collect()
    assert _bits(res) == before
    assert res[0].details["lattice"].n_panels == n_panels
    np.testing.assert_array_equal(res[0].details["lattice"].cp, cp)
    # The same bits as the sweep with the kernel cache kept on the lattice.
    assert before == _old_sweep_bits()


def test_sweep_uses_the_kernel_cache(tracked_kernel_caches):
    """The sweep still fills its kernel cache (the fixed part is computed once)."""
    seen = []
    orig = vortex.influence_matrix

    def spy(*args, **kwargs):
        cache = kwargs.get("cache")
        seen.append(cache is not None and len(cache) >= 0)
        return orig(*args, **kwargs)

    import ventorum.solvers.core as core

    lattice_cache.clear()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(core, "influence_matrix", spy)
        vt.analyze_sweep(_wing(), ALPHAS, solver="vlm", n_panels=20, n_jobs=1)
    assert seen and all(seen)
    gc.collect()
    assert tracked_kernel_caches[0]() is None


def test_parallel_alpha_sweep_does_not_keep_the_kernel_cache(tracked_kernel_caches):
    """The parallel path of alpha_sweep (pool of workers) drops its kernel cache after the sweep."""
    from ventorum.visualization.drag_polar import alpha_sweep

    lattice_cache.clear()
    settings = vt.SolverSettings(solver_type="vlm", n_panels=24)
    res = alpha_sweep(vt.Aircraft(surfaces=[_wing()]), vt.FlightCondition(V_inf=50.0), settings, np.radians(ALPHAS),
                      n_jobs=2)
    gc.collect()
    assert len(res) == ALPHAS.size
    assert tracked_kernel_caches, "the sweep made no kernel cache"
    assert all(r() is None for r in tracked_kernel_caches)
    for r in res:
        assert r.details["lattice"].kernel_cache is None


def test_rate_derivatives_do_not_keep_the_kernel_cache(tracked_kernel_caches):
    lattice_cache.clear()
    cond = vt.FlightCondition(V_inf=50.0, alpha=float(np.radians(3.0)))
    out = vt.rate_derivatives(_wing(), cond, settings=vt.SolverSettings(solver_type="vlm", n_panels=16))
    gc.collect()
    assert len(tracked_kernel_caches) == 1
    assert tracked_kernel_caches[0]() is None
    assert np.isfinite(out.derivatives["CL_q"])
