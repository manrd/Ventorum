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
    r1 = vt.NonlinearLLTSolver().solve(_wing(), cond, st)
    r2 = vt.NonlinearLLTSolver().solve(_wing(), cond, st)
    assert r1.totals.CL == r2.totals.CL and r1.totals.CDi == r2.totals.CDi and r1.totals.Cm == r2.totals.Cm


def test_user_airfoil_arrays_stay_writable():
    """A solve must not make the arrays of the user's polar read-only (review round 5, C2)."""
    import numpy as np

    import ventorum as vt

    alpha = np.radians(np.arange(-10.0, 15.1, 1.0))
    cl = 2 * np.pi * alpha
    cd = 0.01 + 0.0 * alpha
    tab = vt.TabulatedAirfoil(name="user", alpha=alpha, Cl_data=cl, Cd_data=cd)
    wing = vt.LiftingSurface(semi_span=4.0, sections=[vt.WingSection(y_frac=0.0, chord=1.0, airfoil=tab),
                                                      vt.WingSection(y_frac=1.0, chord=1.0, airfoil=tab)])
    for solver in ("vlm", "linear", "nonlinear"):
        vt.analyze(wing, alpha_deg=4.0, solver=solver, n_panels=10)
    assert alpha.flags.writeable and cl.flags.writeable and cd.flags.writeable
    cl[0] = cl[0]  # the user can still write into the array


def test_cached_none_value_is_computed_once():
    """Regression test: a lattice-only value of None was computed again on every case.

    Linear airfoils have no polar limits (None), so the limits were recomputed
    for each case of a sweep.
    """
    from types import SimpleNamespace

    from ventorum.solvers.lattice_base import LatticeSolver

    lattice = SimpleNamespace(geom_cache={})
    calls = []

    def compute():
        calls.append(1)
        return None

    assert LatticeSolver._cached(lattice, "polar_limits", compute) is None
    assert LatticeSolver._cached(lattice, "polar_limits", compute) is None
    assert len(calls) == 1


@pytest.mark.parametrize("shape,hinge,n_chord,text", [
    ("0012", 0.70, 8, "snaps to the nearest node"),
    ("6409", 0.70, 4, "moves a chordwise node"),
])
def test_mesh_warnings_on_every_solve(shape, hinge, n_chord, text):
    """The hinge warnings of the mesh come on every solve, also from a cached lattice, and go to the trust score.

    build_lattice issued them only when it built the lattice: a second solve
    of the same geometry gave no warning.
    """
    import warnings

    lattice_cache.clear()
    wing = vt.LiftingSurface(
        name="w", semi_span=4.0,
        sections=[vt.WingSection(y_frac=e, chord=1.0, shape=shape) for e in (0.0, 1.0)],
        controls=[vt.ControlSurface("flap", 0.1, 0.4, hinge, np.radians(10.0))],
    )
    settings = vt.SolverSettings(n_panels=32, n_chord=n_chord)
    for alpha in (2.0, 3.0, 2.0):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = vt.VortexLatticeSolver().solve(
                vt.Aircraft(surfaces=[wing]), vt.FlightCondition(V_inf=30.0, alpha=np.radians(alpha)), settings)
        hinge_msgs = [str(w.message) for w in caught if text in str(w.message)]
        assert len(hinge_msgs) == 1, hinge_msgs
        assert hinge_msgs[0] in res.details["lattice"].mesh_warnings
        assert hinge_msgs[0] in res.totals.trust.warnings
    # The note does not change the score: the same lattice without its mesh
    # warnings gives the same score.
    import dataclasses

    ac = vt.Aircraft(surfaces=[wing])
    ac.compute_reference_values()
    bare = dataclasses.replace(res.details["lattice"], mesh_warnings=[], geom_cache={})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res0 = vt.VortexLatticeSolver().solve_lattice(
            bare, vt.FlightCondition(V_inf=30.0, alpha=np.radians(2.0)), settings, ac.S_ref, ac.b_ref, ac.c_ref)
    assert not any(text in w for w in res0.totals.trust.warnings)
    assert res0.totals.trust.score == res.totals.trust.score
    lattice_cache.clear()


def _airfoil_aircraft(af):
    """Return a rectangular wing (semi-span 4 m, chord 1 m) with *af* at both sections."""
    return vt.Aircraft(surfaces=[vt.LiftingSurface(name="W", semi_span=4.0, sections=[
        vt.WingSection(y_frac=0.0, chord=1.0, airfoil=af), vt.WingSection(y_frac=1.0, chord=1.0, airfoil=af)])])


def _tab_airfoil(cd0):
    alpha = np.radians(np.arange(-10.0, 12.5, 2.5))
    return vt.TabulatedAirfoil(alpha=alpha, Cl_data=2.0 * np.pi * alpha, Cd_data=cd0 + 0.01 * alpha**2,
                               Cm_data=np.full(alpha.size, -0.05))


@pytest.mark.parametrize("solver", ["vlm", "linear", "nonlinear"])
@pytest.mark.parametrize("kind", ["linear", "tabulated"])
def test_in_place_airfoil_change_does_not_reach_a_cached_lattice(solver, kind):
    """Regression: a cached lattice kept the airfoil objects of its first caller.

    After an in-place change of that airfoil, a new aircraft with the old
    airfoil values hit the old cache entry and computed with the changed
    values (CDp 0.03 instead of 0.01), with no warning. The cached lattice
    now holds its own copies. The changed airfoil gives its new value, and
    an equal aircraft still hits the cache with the old value.
    """
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    st = vt.SolverSettings(solver_type=solver, n_panels=8)

    def make(cd0):
        return vt.LinearAirfoil(Cd0=cd0, Cm0=-0.05) if kind == "linear" else _tab_airfoil(cd0)

    lattice_cache.clear()
    cold = vt.analyze(_airfoil_aircraft(make(0.01)), cond, st.clone()).totals
    lattice_cache.clear()
    af = make(0.01)
    first = vt.analyze(_airfoil_aircraft(af), cond, st.clone())
    if kind == "linear":
        af.Cd0 = 0.03
        af.Cm0 = -0.10
    else:
        af.Cd_data = af.Cd_data + 0.02
        af.Cm_data = np.full(af.alpha.size, -0.10)
    changed = vt.analyze(_airfoil_aircraft(af), cond, st.clone()).totals
    assert changed.CDp > cold.CDp + 0.015
    fresh = vt.analyze(_airfoil_aircraft(make(0.01)), cond, st.clone())
    # An equal aircraft is a cache hit: it shares the arrays of the first lattice.
    assert fresh.details["lattice"].cp is first.details["lattice"].cp
    for name in ("CL", "CDp", "Cm"):
        assert getattr(fresh.totals, name) == pytest.approx(getattr(cold, name), rel=1e-12, abs=1e-15), name
    lattice_cache.clear()


def test_cached_lattice_keeps_the_airfoil_sharing():
    """The cache copies keep one copy per user airfoil, not one per strip."""
    lattice_cache.clear()
    af = vt.LinearAirfoil(Cd0=0.01)
    lat = _lattice(vt.LiftingSurface(name="wing", semi_span=4.0, n_panels=8, sections=[
        vt.WingSection(y_frac=0.0, chord=1.2, airfoil=af), vt.WingSection(y_frac=1.0, chord=0.6, airfoil=af)]))
    assert all(a is not af for a in lat.airfoils)
    assert len({id(a) for a in lat.airfoils}) == 1
    lattice_cache.clear()
