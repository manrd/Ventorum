# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Batched lifting-line sweeps.

A lifting-line sweep solves its angles as one batch: one kernel call gives
the velocity tensors of all angles, one call of the dense solver solves all
systems, and the loads of all angles are computed together. Each angle must
give the same result, to the last bit, as a single solve.
"""

from __future__ import annotations

import importlib.util
import warnings

import numpy as np
import pytest

import ventorum as vt
from ventorum.aero import loads as loads_mod
from ventorum.aero import vortex as V
from ventorum.aero.system import build_sources, ground_plane_from_condition, panel_targets, wake_direction
from ventorum.solvers import core
from ventorum.solvers.factory import make_solver
from ventorum.utils import parallel as par

HAVE_CYTHON = importlib.util.find_spec("ventorum.aero.vortex_cython") is not None
HAVE_TORCH = importlib.util.find_spec("torch") is not None
BACKENDS = ["numpy", "numba"] + (["cython"] if HAVE_CYTHON else []) + (["torch"] if HAVE_TORCH else [])

ALPHAS = np.radians([-4.0, 0.0, 3.0, 7.0, 11.0, 15.0])
TOTALS = ("CL", "CDi", "CDp", "CD_total", "Cl", "Cm", "Cn", "CY", "CDi_nearfield", "e")


@pytest.fixture(autouse=True)
def _restore_kernel_backend():
    old = V._kernel_backend
    yield
    V._kernel_backend = old


def _tabulated():
    a = np.radians(np.arange(-10.0, 21.0))
    return vt.TabulatedAirfoil(alpha=a, Cl_data=2 * np.pi * np.sin(a) * np.cos(a) ** 2,
                               Cd_data=0.01 + 0.02 * a ** 2, Cm_data=-0.05 + 0.01 * a)


def _surface(airfoil=None, name="wing", position=(0.0, 0.0, 0.0), semi_span=4.0, tip_chord=0.6, dihedral=0.0):
    af = airfoil if airfoil is not None else vt.LinearAirfoil(Cd0=0.008, Cm0=-0.03)
    return vt.LiftingSurface(name=name, semi_span=semi_span, n_panels=8, position=np.array(position),
                             dihedral=np.radians(dihedral), sections=[
                                 vt.WingSection(y_frac=0.0, chord=1.2, airfoil=af),
                                 vt.WingSection(y_frac=1.0, chord=tip_chord, airfoil=af, twist=np.radians(-2.0))])


def _wing_and_tail():
    return vt.Aircraft(surfaces=[_surface(dihedral=5.0),
                                 _surface(name="tail", position=(3.5, 0.0, 0.2), semi_span=1.2, tip_chord=0.4)])


CASES = {
    "linear": ("linear", lambda: vt.Aircraft(surfaces=[_surface()]), {}),
    "linear wing and tail": ("linear", _wing_and_tail, {}),
    "linear ground": ("linear", lambda: vt.Aircraft(surfaces=[_surface(dihedral=3.0)]), {"h": 3.0}),
    "linear sideslip": ("linear", lambda: vt.Aircraft(surfaces=[_surface()]), {"beta": np.radians(3.0)}),
    "nonlinear": ("nonlinear", lambda: vt.Aircraft(surfaces=[_surface(_tabulated())]), {}),
    "nonlinear ground": ("nonlinear", lambda: vt.Aircraft(surfaces=[_surface(_tabulated())]), {"h": 3.0}),
}


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return bool(np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True))


def _assert_same_result(r, s):
    for name in TOTALS:
        assert _same(getattr(r.totals, name), getattr(s.totals, name)), name
    assert r.totals.trust.score == s.totals.trust.score
    assert r.converged == s.converged and r.iterations == s.iterations
    assert np.array_equal(r.details["gamma"], s.details["gamma"])
    for sw_r, sw_s in zip(r.spanwise, s.spanwise):
        for name in ("gamma", "Cl", "Cd_i", "alpha_eff", "alpha_i", "local_lift", "Cm_section", "Cd_profile"):
            assert _same(getattr(sw_r, name), getattr(sw_s, name)), name


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", list(CASES))
def test_sweep_angles_equal_single_solves(backend, case):
    """Each angle of a batched sweep has the same bits as a single solve (the nonlinear one with continuation)."""
    V.set_kernel_backend(backend)
    solver_name, make_aircraft, kw = CASES[case]
    solver = make_solver(solver_name)
    ac = make_aircraft()
    st = vt.SolverSettings(solver_type=solver_name, spacing="cosine")
    base = vt.FlightCondition(V_inf=30.0, alpha=0.0, beta=kw.get("beta", 0.0), h=kw.get("h"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sweep = solver.solve_sweep(ac, base, st, ALPHAS)
        ac.compute_reference_values()
        rp = ac.moment_reference()
        g_prev = None
        for a, r in zip(ALPHAS, sweep):
            cond = vt.FlightCondition(V_inf=30.0, alpha=float(a), beta=base.beta, h=base.h)
            lat = solver.build(ac, st, cond, None, rp)
            # The sweep starts each angle from the converged circulation of the
            # angle before it (out of ground effect); give the single solve the same start.
            g0 = g_prev if (solver_name == "nonlinear" and base.h is None) else None
            single = solver.solve_lattice(lat, cond, st, ac.S_ref, ac.b_ref, ac.c_ref, ref_point=rp,
                                          gamma0=g0, main_surface=ac.main_surface_index())
            g_prev = single.details["gamma"] if single.converged else None
            _assert_same_result(r, single)


def test_linear_sweep_equals_package_single_solves():
    """The linear sweep equals vt.analyze at each angle, also for the result conditions."""
    ac = _wing_and_tail()
    st = vt.SolverSettings(solver_type="linear")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # the taper gives a small quarter-chord sweep
        sweep = vt.LinearLLTSolver().solve_sweep(ac, vt.FlightCondition(V_inf=30.0), st, ALPHAS)
        singles = [vt.analyze(ac, vt.FlightCondition(V_inf=30.0, alpha=float(a)), st) for a in ALPHAS]
    for a, r, single in zip(ALPHAS, sweep, singles):
        _assert_same_result(r, single)
        assert r.condition.alpha == float(a)
        assert r.execution_time > 0.0


def _solved(solver_name, ac, cond):
    solver = make_solver(solver_name)
    ac.compute_reference_values()
    rp = ac.moment_reference()
    st = vt.SolverSettings(solver_type=solver_name)
    lat = solver.build(ac, st, cond, None, rp)
    gp = ground_plane_from_condition(lat, cond, rp) if cond.h is not None else None
    wd = wake_direction(cond, gp)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g, ae, info = solver.solve_circulation(lat, cond, st, gp, wd)
    return lat, g, ae, info, gp, wd, rp


@pytest.mark.parametrize("solver_name", ["vlm", "linear", "nonlinear"])
@pytest.mark.parametrize("kw", [{}, {"h": 2.0}, {"beta": np.radians(4.0)}])
def test_loads_batch_rows_equal_single_loads(solver_name, kw):
    """compute_loads_batch gives each case the bits of compute_loads (leg forces, ground, sideslip, polars)."""
    ac = vt.Aircraft(surfaces=[_surface(_tabulated() if solver_name == "nonlinear" else None, dihedral=4.0)])
    conds = [vt.FlightCondition(V_inf=30.0, alpha=np.radians(a), beta=kw.get("beta", 0.0), h=kw.get("h"))
             for a in (2.0, 6.0, 10.0)]
    items = [_solved(solver_name, ac, c) for c in conds]
    lat = items[0][0]
    for use_vc in (False, True):
        vcs = [it[3].v_control for it in items]
        if use_vc and any(v is None for v in vcs):
            continue
        batch = loads_mod.compute_loads_batch(
            lat, np.array([it[1] for it in items]), conds, ac.S_ref, ac.b_ref, ac.c_ref,
            grounds=[it[4] for it in items], ref_point=items[0][6],
            alpha_eff_strips=None if items[0][2] is None else np.array([it[2] for it in items]),
            wake_dirs=np.array([it[5] for it in items]), v_controls=np.array(vcs) if use_vc else None)
        for it, cond, b in zip(items, conds, batch):
            one = loads_mod.compute_loads(lat, it[1], cond, ac.S_ref, ac.b_ref, ac.c_ref, ground=it[4],
                                          ref_point=it[6], alpha_eff_strip=it[2], wake_dir=it[5],
                                          v_control=it[3].v_control if use_vc else None)
            for name in TOTALS:
                assert _same(getattr(b.totals, name), getattr(one.totals, name)), name
            assert np.array_equal(b.strip_force, one.strip_force)
            assert np.array_equal(b.extras["trefftz_normalwash"], one.extras["trefftz_normalwash"])
            assert np.array_equal(b.extras["moment_total_geometry_axes"], one.extras["moment_total_geometry_axes"])


@pytest.mark.parametrize("with_ground", [False, True])
def test_trefftz_batch_rows_equal_single(with_ground):
    """trefftz_induced_drag_batch gives each case the bits of trefftz_induced_drag."""
    ac = _wing_and_tail()
    conds = [vt.FlightCondition(V_inf=30.0, alpha=np.radians(a), h=3.0 if with_ground else None) for a in (1.0, 5.0)]
    items = [_solved("linear", ac, c) for c in conds]
    lat = items[0][0]
    sg = np.array([it[1].reshape(lat.n_strips, lat.n_chord).sum(axis=1) for it in items])
    D, w, s = loads_mod.trefftz_induced_drag_batch(lat, sg, np.array([it[5] for it in items]),
                                                   np.array([1.2, 1.1]), [it[4] for it in items])
    for k, (it, rho) in enumerate(zip(items, (1.2, 1.1))):
        d1, w1, s1 = loads_mod.trefftz_induced_drag(lat, sg[k], it[5], rho, it[4])
        assert D[k] == d1 and np.array_equal(w[k], w1) and np.array_equal(s[k], s1)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("with_ground", [False, True])
def test_batch_tensors_equal_single_tensors(backend, with_ground, monkeypatch):
    """One kernel call for K cases gives each case the bits of its own call, also when the batch is split."""
    V.set_kernel_backend(backend)
    ac = _wing_and_tail()
    conds = [vt.FlightCondition(V_inf=30.0, alpha=np.radians(a), h=3.0 if with_ground else None)
             for a in (0.0, 4.0, 8.0, 12.0)]
    items = [_solved("linear", ac, c) for c in conds]
    lat = items[0][0]
    umap = core._unknown_map(lat, conds[0], items[0][4], True)
    wds = np.array([it[5] for it in items])
    grounds = [it[4] for it in items]
    P = lat.cp[umap.unknown_panels]
    tg = panel_targets(lat, umap.unknown_panels)
    single = [V.velocity_tensor_unknowns(P, build_sources(lat, wds[k], umap, grounds[k]), umap.n, tg)
              for k in range(len(conds))]
    for limit in (core._BATCH_TENSOR_BYTES, 1):   # one call, and one call per case
        monkeypatch.setattr(core, "_BATCH_TENSOR_BYTES", limit)
        batch = core.llt_velocity_tensors(lat, umap, wds, grounds)
        for b, s in zip(batch, single):
            assert np.array_equal(b, s)


def test_threaded_contractions_equal_serial(monkeypatch):
    """The per-case work of a batch on several threads gives the bits of the serial loop."""
    ac = vt.Aircraft(surfaces=[_surface()])
    st = vt.SolverSettings(solver_type="linear")
    solver = vt.LinearLLTSolver()
    serial = solver.solve_sweep(ac, vt.FlightCondition(V_inf=30.0), st, ALPHAS)
    monkeypatch.setattr(core, "_THREADED_MIN_UNKNOWNS", 1)
    with par.forced_single_threads(4):
        threaded = solver.solve_sweep(ac, vt.FlightCondition(V_inf=30.0), st, ALPHAS)
    for r, s in zip(threaded, serial):
        _assert_same_result(r, s)


def test_alpha_sweep_of_lifting_line_uses_the_batch():
    """alpha_sweep (Ventorum.analyze_sweep) of a lifting line gives the results of solve_sweep."""
    from ventorum.visualization.drag_polar import alpha_sweep

    ac = vt.Aircraft(surfaces=[_surface()])
    st = vt.SolverSettings(solver_type="linear")
    cond = vt.FlightCondition(V_inf=30.0)
    for r, s in zip(alpha_sweep(ac, cond, st, ALPHAS, n_jobs=4),
                    vt.LinearLLTSolver().solve_sweep(ac, cond, st, ALPHAS)):
        _assert_same_result(r, s)


@pytest.mark.parametrize("solver_name", ["linear", "nonlinear"])
def test_agent_polar_of_lifting_line_equals_single_solves(solver_name):
    """The agent polar tool solves a lifting-line polar as one batch; each angle equals vt.analyze."""
    from ventorum.agent import tools
    from ventorum.agent.schemas import build_aircraft_from_spec

    spec = {"span_m": 8.0, "root_chord_m": 1.2, "tip_chord_m": 0.7, "tip_twist_deg": -2.0}
    out = tools.polar_sweep(spec, -2.0, 8.0, 2.0, flight_condition={"V_inf_m_s": 30.0},
                            settings={"solver": solver_name, "n_panels": 12})
    assert out["status"] == "success", out
    ac = build_aircraft_from_spec(spec)
    sett = tools.parse_settings({"solver": solver_name, "n_panels": 12})
    for row in out["polar_table"]:
        fc = tools.make_flight_condition(tools.parse_condition({"V_inf_m_s": 30.0}), alpha_deg=row["alpha_deg"])
        single = vt.analyze(ac.clone(), fc, sett)
        assert row["CL"] == tools.metrics_payload(single)["CL"]
        assert row["CDi"] == tools.metrics_payload(single)["CDi"]


def test_empty_batch_and_batch_threads():
    """An empty batch gives no results; a batch that runs alone uses all cores."""
    ac = vt.Aircraft(surfaces=[_surface()])
    assert vt.LinearLLTSolver().solve_sweep(ac, vt.FlightCondition(V_inf=30.0), vt.SolverSettings(), []) == []
    assert par.single_case_threads(40, batch=8) == par.cpu_cores()
    with par.forced_single_threads(2):
        assert par.single_case_threads(40, batch=8) == 2


def test_cached_solver_data_are_read_only():
    """The unknown map and the strip data cached on a lattice cannot be changed by a caller."""
    lat, *_ = _solved("linear", vt.Aircraft(surfaces=[_surface()]), vt.FlightCondition(V_inf=30.0))
    umap = core._unknown_map(lat, vt.FlightCondition(V_inf=30.0), None, True)
    assert core._unknown_map(lat, vt.FlightCondition(V_inf=30.0), None, True) is umap
    with pytest.raises(ValueError):
        umap.panel_column[0] = 3
    with pytest.raises(ValueError):
        core._llt_strip_data(lat, umap)["normal"][0, 0] = 1.0


def test_trust_names_the_first_non_finite_array():
    """The fast all-finite test keeps the old label of the first non-finite array."""
    from ventorum.core.trust import evaluate_aerodynamic_trust

    r = vt.analyze(_surface(), alpha_deg=4.0, solver="linear")
    sw = r.spanwise[0]
    sw.alpha_i = sw.alpha_i.copy()
    sw.alpha_i[2] = np.nan
    sw.local_lift = sw.local_lift.copy()
    sw.local_lift[0] = np.inf
    t = evaluate_aerodynamic_trust(AR=r.totals.AR, spanwise_list=[sw], CL=r.totals.CL, CDi=r.totals.CDi)
    assert t.factors.get("non_finite_result") == 0.6
    assert any(f"{sw.surface_name}.alpha_i" in w for w in t.warnings)
    assert not any("local_lift" in w for w in t.warnings)
