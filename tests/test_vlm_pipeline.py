# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Vortex-lattice single-solve and sweep pipelines (T-0039).

Every check compares bits (``numpy.array_equal``), never with a tolerance,
except the torch backend in ground effect, which is not reproducible to the
bit from one run to the next (also before T-0039): there each angle of the
sweep is compared with a relative tolerance of 1e-12:

* A vortex-lattice sweep out of ground effect gives each angle the result of
  the kernel-cache path that existed before (``solve_lattice`` on a lattice
  with a kernel cache); in ground effect each angle equals a single solve.
* The loads evaluate a repeated point once (``induced_velocity(repeat=...)``)
  with the same bits, with and without the kernel cache.
* ``solve_vlm_batch`` equals ``solve_vlm`` per case.
* ``alpha_sweep`` and the agent polar tool give the results of the paths
  they replace.
"""

from __future__ import annotations

import importlib.util
import warnings

import numpy as np
import pytest

import ventorum as vt
from ventorum.aero import loads as loads_mod
from ventorum.aero import vortex as V
from ventorum.aero.system import UnknownMap, build_sources, ground_plane_from_condition, wake_direction
from ventorum.solvers import core
from ventorum.solvers.factory import make_solver

HAVE_CYTHON = importlib.util.find_spec("ventorum.aero.vortex_cython") is not None
HAVE_TORCH = importlib.util.find_spec("torch") is not None
BACKENDS = ["numpy", "numba"] + (["cython"] if HAVE_CYTHON else []) + (["torch"] if HAVE_TORCH else [])

ALPHAS = np.radians([-2.0, 0.0, 3.0, 6.0, 10.0])
TOTALS = ("CL", "CDi", "CDp", "CD_total", "Cl", "Cm", "Cn", "CY", "CDi_nearfield", "e")


@pytest.fixture(autouse=True)
def _restore_kernel_backend():
    old = V._kernel_backend
    yield
    V._kernel_backend = old


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return bool(np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True))


def _assert_same_result(r, s):
    for name in TOTALS:
        assert _same(getattr(r.totals, name), getattr(s.totals, name)), name
    assert r.totals.trust.score == s.totals.trust.score
    assert np.array_equal(r.details["gamma"], s.details["gamma"])
    for sw_r, sw_s in zip(r.spanwise, s.spanwise):
        for name in ("gamma", "Cl", "Cd_i", "alpha_eff", "alpha_i", "local_lift", "Cm_section", "Cd_profile"):
            assert _same(getattr(sw_r, name), getattr(sw_s, name)), name


def _assert_close_result(r, s, rtol=1e-12, atol=1e-14):
    """Compare two results with a relative tolerance (torch in ground effect).

    The torch backend is not reproducible to the bit from one run to the
    next, so its sweep angles are compared with a tolerance instead of
    with ``numpy.array_equal``.
    """
    for name in TOTALS:
        a, b = getattr(r.totals, name), getattr(s.totals, name)
        if a is None or b is None:
            assert a is None and b is None, name
        else:
            np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=name)
    assert r.totals.trust.score == s.totals.trust.score
    np.testing.assert_allclose(r.details["gamma"], s.details["gamma"], rtol=rtol, atol=atol,
                               err_msg="gamma")
    assert r.converged == s.converged
    assert r.symmetry_used == s.symmetry_used
    for sw_r, sw_s in zip(r.spanwise, s.spanwise):
        for name in ("gamma", "Cl", "Cd_i", "alpha_eff", "alpha_i", "local_lift", "Cm_section", "Cd_profile"):
            a, b = getattr(sw_r, name), getattr(sw_s, name)
            if a is None or b is None:
                assert a is None and b is None, name
            else:
                np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=name)


def _wing(**kw):
    af = vt.LinearAirfoil(Cd0=0.008, Cm0=-0.03)
    return vt.LiftingSurface(name="wing", semi_span=5.0, n_panels=8, sections=[
        vt.WingSection(y_frac=0.0, chord=1.25, airfoil=af),
        vt.WingSection(y_frac=1.0, chord=0.8, airfoil=af, twist=np.radians(-2.0))], **kw)


def _tail():
    return vt.LiftingSurface(name="tail", semi_span=1.5, n_panels=4, position=np.array([3.5, 0.0, 0.5]),
                             sections=[vt.WingSection(y_frac=0.0, chord=0.6), vt.WingSection(y_frac=1.0, chord=0.4)])


CASES = {
    "free air": (lambda: vt.Aircraft(surfaces=[_wing(sweep_le=np.radians(10.0))]), {}),
    "sideslip": (lambda: vt.Aircraft(surfaces=[_wing(dihedral=np.radians(4.0))]), {"beta": np.radians(3.0)}),
    "wing and tail": (lambda: vt.Aircraft(surfaces=[_wing(), _tail()]), {}),
    "ground": (lambda: vt.Aircraft(surfaces=[_wing()]), {"h": 1.0}),
    "ground wing and tail": (lambda: vt.Aircraft(surfaces=[_wing(), _tail()]), {"h": 1.5}),
}


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("n_chord", [1, 4, None])
def test_vlm_sweep_angles_equal_reference(backend, case, n_chord):
    """Each angle of solve_sweep equals the kernel-cache solve (free air) or the single solve (ground)."""
    make_aircraft, kw = CASES[case]
    in_ground = "h" in kw
    close = backend == "torch" and in_ground
    V.set_kernel_backend(backend)
    solver = make_solver("vlm")
    st = vt.SolverSettings(solver_type="vlm", n_chord=n_chord, spacing="cosine")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sweep = solver.solve_sweep(make_aircraft(), vt.FlightCondition(V_inf=30.0, **kw), st, ALPHAS)
        ac = make_aircraft()
        ac.compute_reference_values()
        rp = ac.moment_reference()
        cache_lattice = None
        for a, r in zip(ALPHAS, sweep):
            cond = vt.FlightCondition(V_inf=30.0, alpha=float(a), **kw)
            if in_ground:
                ref = solver.solve(ac, cond, st)
            else:
                if cache_lattice is None:
                    cache_lattice = solver.build(ac, st, cond, None, rp)
                    cache_lattice.kernel_cache = {}
                ref = solver.solve_lattice(cache_lattice, cond, st, ac.S_ref, ac.b_ref, ac.c_ref,
                                           ref_point=rp, main_surface=ac.main_surface_index())
            if close:
                _assert_close_result(r, ref)
            else:
                _assert_same_result(r, ref)


def _vlm_case(ac, cond):
    solver = make_solver("vlm")
    ac.compute_reference_values()
    rp = ac.moment_reference()
    st = vt.SolverSettings(solver_type="vlm", n_chord=4)
    lat = solver.build(ac, st, cond, None, rp)
    gp = ground_plane_from_condition(lat, cond, rp) if cond.h is not None else None
    wd = wake_direction(cond, gp)
    g, _, _ = solver.solve_circulation(lat, cond, st, gp, wd)
    return lat, g, gp, wd


@pytest.mark.parametrize("backend", ["numpy", "numba"] + (["cython"] if HAVE_CYTHON else []))
@pytest.mark.parametrize("with_cache", [False, True])
def test_repeated_points_give_the_same_bits(backend, with_cache):
    """induced_velocity with repeat equals the evaluation of every point, with and without the kernel cache."""
    V.set_kernel_backend(backend)
    ac = vt.Aircraft(surfaces=[_wing(sweep_le=np.radians(10.0))])
    lat, g, gp, wd = _vlm_case(ac, vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0)))
    full = UnknownMap(unknown_panels=np.arange(lat.n_panels), panel_column=np.arange(lat.n_panels), symmetric=False)
    src = build_sources(lat, wd, full, None)
    pts = [lat.force_points, 0.5 * (lat.a_te + lat.a), 0.5 * (lat.b + lat.b_te)]
    P, tgt, repeat = loads_mod._load_points(lat, np.arange(lat.n_panels), pts)
    assert repeat is not None and repeat[0].size < P.shape[0]
    assert np.array_equal(P[repeat[0]][repeat[1]], P)
    ref_cache = {} if with_cache else None
    new_cache = {} if with_cache else None
    key = "k" if with_cache else None
    for _ in range(2):   # the second call uses the cached part
        ref = V.induced_velocity(P, src, g, tgt, cache=ref_cache, cache_key=key)
        new = V.induced_velocity(P, src, g, tgt, cache=new_cache, cache_key=key, repeat=repeat)
        assert np.array_equal(ref, new)


def test_no_repeats_with_two_core_groups():
    """With two core groups the target core radius acts: no point is shared."""
    lat, *_ = _vlm_case(vt.Aircraft(surfaces=[_wing(), _tail()]), vt.FlightCondition(V_inf=30.0))
    pts = [lat.force_points, 0.5 * (lat.a_te + lat.a), 0.5 * (lat.b + lat.b_te)]
    assert loads_mod._load_points(lat, np.arange(lat.n_panels), pts)[2] is None


@pytest.mark.parametrize("kw", [{}, {"beta": np.radians(3.0)}, {"h": 1.0}])
def test_solve_vlm_batch_equals_solve_vlm(kw):
    """solve_vlm_batch gives each case the circulation of solve_vlm."""
    ac = vt.Aircraft(surfaces=[_wing(dihedral=np.radians(3.0))])
    conds = [vt.FlightCondition(V_inf=30.0, alpha=np.radians(a), **kw) for a in (1.0, 5.0)]
    items = [_vlm_case(ac, c) for c in conds]
    if items[0][0].n_panels != items[1][0].n_panels:
        pytest.skip("the two attitudes have different chordwise counts")
    batch = core.solve_vlm_batch(items[0][0], conds, [it[2] for it in items], True,
                                 np.array([it[3] for it in items]))
    for (lat_k, _, gp, wd), cond, (g_b, _) in zip(items, conds, batch):
        g_1, _ = core.solve_vlm(lat_k, cond, gp, True, wd)
        assert np.array_equal(g_b, g_1)


@pytest.mark.parametrize("n_jobs", ["auto", 2])
def test_alpha_sweep_of_vlm_keeps_its_results(n_jobs):
    """alpha_sweep of the vortex lattice gives the results of solve_sweep (batch or pool of workers)."""
    from ventorum.visualization.drag_polar import alpha_sweep

    ac = vt.Aircraft(surfaces=[_wing()])
    st = vt.SolverSettings(solver_type="vlm", n_chord=4)
    cond = vt.FlightCondition(V_inf=30.0)
    for r, s in zip(alpha_sweep(ac, cond, st, ALPHAS, n_jobs=n_jobs),
                    vt.HorseshoeSolver().solve_sweep(ac, cond, st, ALPHAS)):
        _assert_same_result(r, s)


def test_agent_polar_of_vlm_equals_single_solves():
    """The agent polar tool of a small vortex lattice equals vt.analyze per angle, to the bit."""
    from ventorum.agent import tools
    from ventorum.agent.schemas import build_aircraft_from_spec

    spec = {"span_m": 8.0, "root_chord_m": 1.2, "tip_chord_m": 0.7, "tip_twist_deg": -2.0}
    sett = tools.parse_settings({"solver": "vlm", "n_panels": 8, "n_chord": 4})
    ac = build_aircraft_from_spec(spec)
    alphas = [-2.0, 2.0, 6.0]
    conds = [tools.make_flight_condition(tools.parse_condition({"V_inf_m_s": 30.0}), alpha_deg=a) for a in alphas]
    rows = tools._batched_polar(ac, sett, conds, alphas)
    for (_, res, err, _), fc in zip(rows, conds):
        assert err is None
        _assert_same_result(res, vt.analyze(ac.clone(), fc, sett))
    out = tools.polar_sweep(spec, -2.0, 6.0, 4.0, flight_condition={"V_inf_m_s": 30.0},
                            settings={"solver": "vlm", "n_panels": 8, "n_chord": 4})
    assert out["status"] == "success"
    for row, (_, res, _, _) in zip(out["polar_table"], rows):
        assert row["CL"] == tools.metrics_payload(res)["CL"]


def test_probe_lattices_come_from_the_cache():
    """The probe lattice of resolve_n_chord and the lattice of estimate_panels are built once (lattice cache)."""
    from ventorum.geometry import lattice_cache
    from ventorum.utils.parallel import estimate_panels

    lattice_cache.clear()
    ac = vt.Aircraft(surfaces=[_wing()])
    ac.compute_reference_values()
    st = vt.SolverSettings(solver_type="vlm")
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), h=1.0)
    solver = vt.HorseshoeSolver()
    assert solver.resolve_n_chord(ac, st, cond, None) == solver.resolve_n_chord(ac, st, cond, None)
    assert estimate_panels(ac, st) == estimate_panels(ac, st) == 16 * 4
    assert len(lattice_cache._cache) == 2   # the probe (one chordwise panel) and the four-panel lattice
