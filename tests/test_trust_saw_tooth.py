# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Saw-tooth rule of the trust score (factor ``numerical``).

The rule flags a spanwise circulation whose strip-to-strip step changes
sign on three or more consecutive steps (the strips alternate up and down).
This is the odd-even oscillation of an ill-conditioned system, for example
the former mean-line lattice with straight horseshoe legs (an AR 8 NACA 6409
wing with the default 80 spanwise panels gave CL 7.74 at alpha 0).

A former rule counted all local extrema of the circulation. Each part-span
control adds physical extrema at its edges, so a smooth, mesh-converged
solution with a flap and an aileron got the penalty, and the rating changed
with the mesh. The tests below check that such solutions are not flagged.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import ventorum as vt
from ventorum.core.datatypes import FlightCondition
from ventorum.core.trust import (
    _SAW_TOOTH_RUN,
    _sign_change_run,
    _sign_change_run_rows,
    evaluate_aerodynamic_trust,
    spanwise_stats_batch,
)

_KEY = "Spanwise circulation oscillates"


def _flagged(res) -> bool:
    return any(_KEY in w for w in res.totals.trust.warnings)


def _ctrl(name, e0, e1, hinge, deg, symmetric=True):
    return vt.ControlSurface(name=name, eta_start=e0, eta_end=e1, hinge_x_c=hinge, deflection=np.radians(deg),
                             symmetric=symmetric)


def _wing(controls=(), shape=None):
    w = vt.LiftingSurface(name="w", semi_span=4.0, controls=list(controls),
                          sections=[vt.WingSection(0.0, 1.0, shape=shape), vt.WingSection(1.0, 1.0, shape=shape)])
    ac = vt.Aircraft(surfaces=[w], ref_point=np.array([0.25, 0.0, 0.0]))
    ac.compute_reference_values()
    return ac


def _flap_aileron(flap_deg, ail_deg):
    return [_ctrl("flap", 0.1, 0.4, 0.7, flap_deg), _ctrl("ail", 0.6, 0.95, 0.75, ail_deg, symmetric=False)]


def _multi_controls():
    return [_ctrl("f1", 0.05, 0.2, 0.7, 15.0), _ctrl("f2", 0.2, 0.35, 0.7, -10.0), _ctrl("f3", 0.35, 0.5, 0.7, 20.0),
            _ctrl("f4", 0.55, 0.7, 0.75, 5.0), _ctrl("ail", 0.75, 0.97, 0.75, 12.0, symmetric=False)]


def _solve(ac, solver, n, alpha_deg=3.0, n_chord=None, h=None):
    fc = FlightCondition(V_inf=30.0, alpha=np.radians(alpha_deg), h=h)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return vt.analyze(ac, fc, vt.SolverSettings(solver_type=solver, n_panels=n, n_chord=n_chord))


def _elliptic(n):
    y = np.cos(np.linspace(np.pi, 0.0, 2 * n + 1))
    yc = 0.5 * (y[1:] + y[:-1])
    return yc, np.sqrt(1.0 - yc ** 2)


# ----------------------------------------------------------------------------- the measure
def test_smooth_load_has_a_run_of_one():
    """A smooth load with one maximum away from the root has one sign change."""
    yc, g = _elliptic(40)
    assert _sign_change_run(g * (1.0 + 0.3 * yc)) == 1


@pytest.mark.parametrize("n", [24, 48, 80])
def test_synthetic_saw_tooth_is_a_long_run(n):
    """A load with an odd-even oscillation over the whole span gives a run of nearly all steps."""
    _, g = _elliptic(n)
    g = g * (1.0 + 0.2 * (-1.0) ** np.arange(g.size))
    assert _sign_change_run(g) >= g.size - 6
    assert _sign_change_run(g) >= _SAW_TOOTH_RUN


def test_synthetic_saw_tooth_at_the_tips_is_flagged():
    """An oscillation on a few tip strips only (as on the former mean-line lattice) gives a long run."""
    yc, g = _elliptic(40)
    g = g + 0.05 * np.where(np.abs(yc) > 0.9, (-1.0) ** np.arange(g.size), 0.0)
    assert _sign_change_run(g) >= _SAW_TOOTH_RUN


def test_features_one_strip_wide_are_not_a_saw_tooth():
    """A bump and a dip one strip wide each give two consecutive sign changes, not more."""
    _, g = _elliptic(40)
    g = g.copy()
    g[20] += 0.3
    g[50] -= 0.3
    assert _sign_change_run(g) == 2


def test_a_step_without_sign_stops_the_run():
    """A step below the tolerance (for example the symmetric root step) is not a sign change."""
    g = np.array([0.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 0.0])
    assert _sign_change_run(g) == 1


def test_run_of_rows_equals_the_run_per_row():
    rng = np.random.default_rng(7)
    for _ in range(300):
        m = int(rng.integers(1, 25))
        G = rng.normal(size=(4, m)).cumsum(axis=1)
        if rng.random() < 0.5:
            G = np.round(G, 1)
        if rng.random() < 0.3:
            G = G + 0.5 * (-1.0) ** np.arange(m)
        assert np.array_equal(_sign_change_run_rows(G), [_sign_change_run(g) for g in G])


def test_trust_rule_uses_the_threshold():
    """The penalty 0.40 and the warning come at the threshold run, not below it."""
    kw = dict(AR=8.0, CL=0.4, CDi=0.007, n_panels=20, solver_type="vlm", n_chord=4)
    for run, flag in ((_SAW_TOOTH_RUN - 1, False), (_SAW_TOOTH_RUN, True)):
        stats = {"max_local_cl": 0.5, "max_alpha_eff_deg": 3.0, "sign_change_run": run}
        t = evaluate_aerodynamic_trust(spanwise_stats=stats, **kw)
        assert any(_KEY in w for w in t.warnings) is flag
        assert t.factors.get("numerical", 0.0) == pytest.approx(0.40 if flag else 0.0)


# ----------------------------------------------------------------------------- smooth solutions
@pytest.mark.parametrize("solver,n_chord", [("vlm", 4), ("linear", None)])
@pytest.mark.parametrize("n", [24, 48])
@pytest.mark.parametrize("flap,ail", [(10.0, 10.0), (20.0, -15.0)])
def test_flap_and_aileron_are_not_flagged(solver, n_chord, n, flap, ail):
    """A smooth solution with a flap and an aileron (seven physical extrema) gets no penalty at 24 and 48 panels."""
    res = _solve(_wing(_flap_aileron(flap, ail)), solver, n, n_chord=n_chord)
    assert max(_sign_change_run(sw.gamma) for sw in res.spanwise) < _SAW_TOOTH_RUN
    assert not _flagged(res)
    assert res.totals.trust.factors.get("numerical", 0.0) == 0.0


@pytest.mark.parametrize("solver,n_chord", [("vlm", 4), ("linear", None)])
@pytest.mark.parametrize("n", [24, 48])
def test_wing_with_five_controls_is_not_flagged(solver, n_chord, n):
    """Four flaps (two adjacent pairs with opposite deflections) and an aileron: no penalty."""
    res = _solve(_wing(_multi_controls()), solver, n, n_chord=n_chord)
    assert not _flagged(res)


def test_flap_and_aileron_in_ground_effect_are_not_flagged():
    res = _solve(_wing(_flap_aileron(10.0, 10.0)), "vlm", 48, n_chord=4, h=0.3)
    assert not _flagged(res)


# ----------------------------------------------------------------------------- saw-tooth solutions
@pytest.fixture
def straight_legs(monkeypatch):
    """Build mean-line lattices with the former straight horseshoe legs (no node chains)."""
    from ventorum.geometry import lattice as mesh_mod
    from ventorum.geometry import lattice_cache

    orig = mesh_mod._leg_nodes

    def straight(a, b, a_te, b_te, hinge_nodes, chain_strips, n_chord, *nodes):
        return orig(a, b, a_te, b_te, hinge_nodes, chain_strips[:0], n_chord)

    monkeypatch.setattr(mesh_mod, "_leg_nodes", straight)
    lattice_cache.clear()
    yield
    lattice_cache.clear()


@pytest.mark.parametrize("shape,alpha", [("6409", 0.0), ("4412", 0.0), ("6409", 2.0)])
def test_straight_legs_saw_tooth_is_flagged(straight_legs, shape, alpha):
    """The former ill-conditioned mean-line lattice (default 80 panels) oscillates at the tips: flagged."""
    res = _solve(_wing(shape=shape), "vlm", 80, alpha_deg=alpha)
    assert res.details["lattice"].leg_next is None
    assert _flagged(res)
    assert res.totals.trust.factors["numerical"] >= 0.40


def test_mean_surface_legs_are_not_flagged():
    """The same wing with the legs on the mean surface is smooth."""
    res = _solve(_wing(shape="6409"), "vlm", 80, alpha_deg=0.0)
    assert not _flagged(res)


# ----------------------------------------------------------------------------- batch and single paths
def _batch_and_single(ac, settings, alphas):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = vt.VortexLatticeSolver().solve_sweep(ac, FlightCondition(V_inf=30.0), settings, np.radians(alphas))
    lat = res[0].details["lattice"]
    arrays = {}
    for name in ("gamma", "Cl", "Cd_i", "alpha_eff", "alpha_i", "local_lift", "Cm_section"):
        arrays[name] = np.array([np.concatenate([getattr(sw, name) for sw in r.spanwise]) for r in res])
    stats = spanwise_stats_batch([surf.strips for surf in lat.surfaces], arrays)
    out = []
    for r, s in zip(res, stats):
        t = r.totals
        kw = dict(AR=t.AR, condition=r.condition, CL=t.CL, CDi=t.CDi, CD_total=t.CD_total,
                  n_panels=settings.n_panels, solver_type="vlm", n_chord=settings.n_chord)
        a = evaluate_aerodynamic_trust(spanwise_list=r.spanwise, **kw)
        b = evaluate_aerodynamic_trust(spanwise_stats=s, **kw)
        assert a.to_dict() == b.to_dict()
        assert s["sign_change_run"] == max(_sign_change_run(sw.gamma) for sw in r.spanwise)
        out.append(a)
    return out


def test_batch_and_single_paths_agree_on_controls():
    out = _batch_and_single(_wing(_flap_aileron(10.0, 10.0)), vt.SolverSettings(solver_type="vlm", n_panels=48,
                                                                                   n_chord=4), [-2.0, 3.0, 8.0])
    assert not any(_KEY in w for t in out for w in t.warnings)


def test_batch_and_single_paths_agree_on_a_saw_tooth(straight_legs):
    out = _batch_and_single(_wing(shape="6409"), vt.SolverSettings(solver_type="vlm", n_panels=80), [0.0, 2.0])
    assert all(any(_KEY in w for w in t.warnings) for t in out)


# ----------------------------------------------------------------------------- GPU kernel of the statistics
@pytest.mark.slow
def test_warp_statistics_kernel_equals_the_cpu_rule():
    """The trust statistics of the Warp loads kernel give the same run as the CPU rule (Warp CPU device)."""
    wp = pytest.importorskip("warp")
    from ventorum.gpu.kernels import kernels

    k = kernels("float64")
    rng = np.random.default_rng(11)
    K, m, cut = 200, 40, 25
    G = rng.normal(size=(K, m)).cumsum(axis=1)
    G[::2] = np.round(G[::2], 1)
    G[::3] += 0.5 * (-1.0) ** np.arange(m)
    span = np.zeros((K, 8, m))
    span[:, 0, :] = G
    dev = "cpu"
    out = wp.zeros((K, 22), dtype=wp.float64, device=dev)
    ones = wp.array(np.ones(K), dtype=wp.float64, device=dev)
    wp.launch(k.loads_totals, dim=(K,), device=dev, inputs=[
        wp.array(np.zeros((K, 2, 12)), dtype=wp.float64, device=dev),
        wp.array(np.tile([1.0, 0.0, 0.0], (K, 1)), dtype=wp.vec3d, device=dev),
        ones, ones, wp.array(np.zeros(K), dtype=wp.float64, device=dev), 1.0, 1.0, 1.0,
        wp.array(span, dtype=wp.float64, device=dev),
        wp.array(np.array([0, cut], np.int32), dtype=wp.int32, device=dev),
        wp.array(np.array([cut, m], np.int32), dtype=wp.int32, device=dev), out])
    ref = np.maximum(_sign_change_run_rows(G[:, :cut]), _sign_change_run_rows(G[:, cut:]))
    assert np.array_equal(out.numpy()[:, 21].astype(int), ref)


@pytest.mark.gpu
def test_gpu_sweep_gives_the_cpu_saw_tooth_result(straight_legs):
    """On the GPU the straight-leg saw-tooth and the flap and aileron wing get the CPU rating and warning."""
    from ventorum import gpu

    if not gpu.available():
        pytest.skip(f"The GPU pipelines cannot run: {gpu.unavailable_reason()}")
    old_dev, old_prec = gpu.get_device(), gpu._precision
    try:
        for ac, n in ((_wing(shape="6409"), 80), (_wing(_flap_aileron(10.0, 10.0)), 48)):
            st = vt.SolverSettings(solver_type="vlm", n_panels=n, n_chord=4)
            runs = {}
            for device in ("cpu", "gpu"):
                gpu.set_device(device)
                gpu.set_precision("float64")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    runs[device] = vt.VortexLatticeSolver().solve_sweep(
                        ac, FlightCondition(V_inf=30.0), st, np.radians([0.0, 2.0]))
            for r, s in zip(runs["cpu"], runs["gpu"]):
                assert s.details["device"] == "gpu"
                assert r.totals.trust.rating == s.totals.trust.rating
                assert _flagged(r) == _flagged(s)
    finally:
        gpu.set_device(old_dev)
        gpu.set_precision(old_prec)
