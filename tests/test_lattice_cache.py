# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Cache of built lattices (owner decision 2026-10-03)."""

from __future__ import annotations

import numpy as np
import pytest

import ventorum as vt
from ventorum.geometry import lattice as lattice_module
from ventorum.geometry import lattice_cache


def _wing():
    return vt.LiftingSurface(name="wing", semi_span=4.0, n_panels=8, sections=[
        vt.WingSection(y_frac=0.0, chord=1.2), vt.WingSection(y_frac=1.0, chord=0.6)])


def _lattice(surface):
    solver = vt.LinearLLTSolver()
    ac = vt.Aircraft(surfaces=[surface])
    ac.compute_reference_values()
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    return solver.build(ac, vt.SolverSettings(solver_type="linear"), cond)


def test_equal_geometry_reuses_the_lattice():
    """Two objects with the same geometry share the arrays of one cached lattice."""
    lattice_cache.clear()
    a, b = _lattice(_wing()), _lattice(_wing())
    assert a is not b
    assert a.cp is b.cp
    assert a.kernel_cache is None


def test_geometry_change_gives_a_new_lattice():
    """A changed chord gives another lattice."""
    lattice_cache.clear()
    w = _wing()
    a = _lattice(w)
    w.sections[1].chord = 0.5
    b = _lattice(w)
    assert a.cp is not b.cp
    assert not np.array_equal(a.chord, b.chord)


def test_replaced_builder_function_gives_a_new_lattice(monkeypatch):
    """Regression (verification case V10): a replaced core_groups is part of the key."""
    lattice_cache.clear()
    a = _lattice(_wing())
    monkeypatch.setattr(lattice_module, "core_groups", lambda surfaces: np.arange(len(surfaces)) + 7)
    b = _lattice(_wing())
    assert a.cp is not b.cp
    assert np.all(b.strip_core_group == 7)


def test_cached_arrays_are_read_only():
    """A write into a shared lattice raises instead of changing later solves."""
    lattice_cache.clear()
    lat = _lattice(_wing())
    with pytest.raises(ValueError):
        lat.cp[0, 0] = 1.0


def test_repeated_solve_gives_the_same_bits():
    """The second solve (cached lattice) equals the first (new lattice) bit for bit."""
    lattice_cache.clear()
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    st = vt.SolverSettings(solver_type="nonlinear")
    r1 = vt.NonlinearSolver().solve(_wing(), cond, st)
    r2 = vt.NonlinearSolver().solve(_wing(), cond, st)
    assert r1.totals.CL == r2.totals.CL and r1.totals.CDi == r2.totals.CDi and r1.totals.Cm == r2.totals.Cm
