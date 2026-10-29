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
    lattice_spanwise_stats,
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
def _warp_run(dev, G, cut, edge):
    """Return the sign-change run of the Warp loads kernel on device *dev* for the rows of *G*."""
    import warp as wp

    from ventorum.gpu.kernels import kernels

    k = kernels("float64")
    K, m = G.shape
    span = np.zeros((K, 8, m))
    span[:, 0, :] = G
    out = wp.zeros((K, 22), dtype=wp.float64, device=dev)
    ones = wp.array(np.ones(K), dtype=wp.float64, device=dev)
    wp.launch(k.loads_totals, dim=(K,), device=dev, inputs=[
        wp.array(np.zeros((K, 2, 12)), dtype=wp.float64, device=dev),
        wp.array(np.tile([1.0, 0.0, 0.0], (K, 1)), dtype=wp.vec3d, device=dev),
        ones, ones, wp.array(np.zeros(K), dtype=wp.float64, device=dev), 1.0, 1.0, 1.0,
        wp.array(span, dtype=wp.float64, device=dev),
        wp.array(np.array([0, cut], np.int32), dtype=wp.int32, device=dev),
        wp.array(np.array([cut, m], np.int32), dtype=wp.int32, device=dev),
        wp.array(np.asarray(edge, dtype=np.int32), dtype=wp.int32, device=dev), out])
    return out.numpy()[:, 21].astype(int)


def _random_rows(seed=11, K=200, m=40):
    rng = np.random.default_rng(seed)
    G = rng.normal(size=(K, m)).cumsum(axis=1)
    G[::2] = np.round(G[::2], 1)
    G[::3] += 0.5 * (-1.0) ** np.arange(m)
    return G


def _cpu_run(G, cut, edge):
    edge = np.asarray(edge, dtype=bool)
    return np.maximum(_sign_change_run_rows(G[:, :cut], skip=edge[:cut - 1]),
                      _sign_change_run_rows(G[:, cut:], skip=edge[cut:-1]))


@pytest.mark.slow
def test_warp_statistics_kernel_equals_the_cpu_rule():
    """The trust statistics of the Warp loads kernel give the same run as the CPU rule (Warp CPU device)."""
    pytest.importorskip("warp")
    G, cut = _random_rows(), 25
    edge = np.zeros(G.shape[1], dtype=bool)
    assert np.array_equal(_warp_run("cpu", G, cut, edge), _cpu_run(G, cut, edge))


@pytest.mark.slow
def test_warp_statistics_kernel_with_control_edges_equals_the_cpu_rule():
    """With steps near control edges masked, the Warp kernel (CPU device) gives the CPU run."""
    pytest.importorskip("warp")
    G, cut = _random_rows(13), 25
    edge = np.zeros(G.shape[1], dtype=bool)
    edge[[3, 4, 5, 10, 11, 12, 22, 23, 30, 31, 32]] = True
    ref = _cpu_run(G, cut, edge)
    assert not np.array_equal(ref, _cpu_run(G, cut, np.zeros_like(edge)))
    assert np.array_equal(_warp_run("cpu", G, cut, edge), ref)


@pytest.mark.gpu
def test_warp_statistics_kernel_with_control_edges_on_the_gpu():
    """CPU/GPU parity of the masked sign-change run (Warp kernel on the CUDA device)."""
    wp = pytest.importorskip("warp")
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("No CUDA device for Warp.")
    G, cut = _random_rows(17), 25
    edge = np.zeros(G.shape[1], dtype=bool)
    edge[[3, 4, 5, 10, 11, 12, 22, 23, 30, 31, 32]] = True
    assert np.array_equal(_warp_run("cuda:0", G, cut, edge), _cpu_run(G, cut, edge))


@pytest.mark.gpu
def test_gpu_sweep_with_a_flap_edge_gives_the_cpu_statistics():
    """On the GPU the sweep of a flapped wing passes the lattice edge mask: same run, rating and warning as the CPU."""
    from ventorum import gpu

    if not gpu.available():
        pytest.skip(f"The GPU pipelines cannot run: {gpu.unavailable_reason()}")
    old_dev, old_prec = gpu.get_device(), gpu._precision
    try:
        st = vt.SolverSettings(solver_type="vlm", n_panels=48, n_chord=8)
        runs = {}
        for device in ("cpu", "gpu"):
            gpu.set_device(device)
            gpu.set_precision("float64")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                runs[device] = vt.VortexLatticeSolver().solve_sweep(
                    _swept_flap(), FlightCondition(V_inf=30.0), st, np.radians([0.0, 4.0, 8.0]))
        for r, s in zip(runs["cpu"], runs["gpu"]):
            assert s.details["device"] == "gpu"
            lat = s.details["lattice"]
            assert lat.control_edge_step is not None
            assert r.totals.trust.rating == s.totals.trust.rating
            assert _flagged(r) == _flagged(s)
    finally:
        gpu.set_device(old_dev)
        gpu.set_precision(old_prec)


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


# ----------------------------------------------------------------------------- control edges
# A jump of the deflection at a side edge of a control gives a jump of the
# spanwise slope of the load with a logarithmic shape. When the mesh resolves
# the edge, this physical feature can give a peak, a dip one strip wide and a
# bump (about 5e-4 of the peak circulation), that is three consecutive sign
# changes. The steps whose strips are both within two strips of a control
# edge therefore have no sign in the saw-tooth measure.
def _swept_flap(controls=None):
    if controls is None:
        controls = [_ctrl("flap", 0.2, 0.6, 0.75, 10.0)]
    w = vt.LiftingSurface(name="w", semi_span=4.0, controls=list(controls), sweep_le=np.radians(30.0),
                          sections=[vt.WingSection(0.0, 1.0), vt.WingSection(1.0, 0.6)])
    ac = vt.Aircraft(surfaces=[w], ref_point=np.array([0.25, 0.0, 0.0]))
    ac.compute_reference_values()
    return ac


@pytest.fixture(scope="module")
def flap_case():
    """Swept, tapered wing with a part-span flap (10 deg): the solved result and its lattice."""
    res = _solve(_swept_flap(), "vlm", 48, alpha_deg=4.0, n_chord=8)
    return res, res.details["lattice"]


def _edges(lat):
    """Return the strip index e of each control edge (between strips e - 1 and e) of the lattice."""
    step = lat.control_edge_step
    idx = np.flatnonzero(step)
    # Each edge gives the three steps e - 2, e - 1 and e.
    return [int(i) for i in idx if i >= 2 and step[i - 1] and step[i - 2] and not (i + 1 < step.size and step[i + 1])]


def _edge_feature(g, e, rel=5e-4):
    """Put a peak (strip e - 1), a dip one strip wide (strip e) and a bump (strip e + 1) at the edge e."""
    g = g.copy()
    h = rel * np.abs(g).max()
    m = max(g[e - 2], g[e + 2])
    g[e - 1], g[e], g[e + 1] = m + 2.0 * h, m + h, m + 2.0 * h
    return g


def test_control_edge_steps_of_the_lattice(flap_case):
    """The flap has four edges (two per half); each gives the three steps whose strips are near it."""
    _, lat = flap_case
    step = lat.control_edge_step
    assert step is not None and step.shape == (lat.n_strips,)
    edges = _edges(lat)
    assert len(edges) == 4
    assert np.flatnonzero(step).tolist() == sorted(e + k for e in edges for k in (-2, -1, 0))
    for e in edges:
        assert lat.eta[e - 1] * lat.eta[e] > 0.0  # not at the root


def test_no_control_edge_without_deflection():
    """A control with zero deflection and a symmetric flap through the root give no edge there."""
    res = _solve(_swept_flap([_ctrl("flap", 0.2, 0.6, 0.75, 0.0)]), "vlm", 24, n_chord=4)
    assert res.details["lattice"].control_edge_step is None
    lat = _solve(_swept_flap([_ctrl("flap", 0.0, 0.5, 0.75, 10.0)]), "vlm", 24, n_chord=4).details["lattice"]
    assert len(_edges(lat)) == 2


def test_antisymmetric_control_through_the_root_has_a_root_edge():
    lat = _solve(_swept_flap([_ctrl("ail", 0.0, 0.5, 0.75, 10.0, symmetric=False)]), "vlm", 24,
                 n_chord=4).details["lattice"]
    root = lat.n_strips // 2
    assert root in _edges(lat)


def test_two_controls_that_meet_have_an_edge():
    """Two adjacent controls with different deflections, or with different hinges, have an edge between them."""
    for second in (_ctrl("f2", 0.4, 0.7, 0.75, -5.0), _ctrl("f2", 0.4, 0.7, 0.8, 10.0)):
        lat = _solve(_swept_flap([_ctrl("f1", 0.1, 0.4, 0.75, 10.0), second]), "vlm", 24,
                     n_chord=4).details["lattice"]
        assert len(_edges(lat)) == 6


def test_flap_edge_feature_is_not_flagged(flap_case):
    """The peak, dip and bump at a flap edge give a run of 3 without the exemption, and are not flagged with it."""
    res, lat = flap_case
    sw = res.spanwise[0]
    assert _sign_change_run(sw.gamma) < _SAW_TOOTH_RUN
    for e in _edges(lat):
        g = _edge_feature(sw.gamma, e)
        skip = lat.control_edge_step[:-1]
        assert _sign_change_run(g) >= _SAW_TOOTH_RUN
        assert _sign_change_run(g, skip=skip) < _SAW_TOOTH_RUN
        # The same feature one strip further out is also not flagged.
        assert _sign_change_run(_edge_feature(sw.gamma, e + 1), skip=skip) < _SAW_TOOTH_RUN
        # Through the trust score: the spanwise results alone flag it, the lattice statistics do not.
        new = sw.clone()
        new.gamma = g
        t = res.totals
        kw = dict(AR=t.AR, condition=res.condition, CL=t.CL, CDi=t.CDi, CD_total=t.CD_total, n_panels=48,
                  solver_type="vlm", n_chord=8)
        assert any(_KEY in w for w in evaluate_aerodynamic_trust(spanwise_list=[new], **kw).warnings)
        stats = lattice_spanwise_stats(lat, [new])
        assert stats["sign_change_run"] < _SAW_TOOTH_RUN
        t2 = evaluate_aerodynamic_trust(spanwise_list=[new], spanwise_stats=stats, **kw)
        assert not any(_KEY in w for w in t2.warnings)


def test_saw_tooth_from_a_control_edge_is_flagged(flap_case):
    """An odd-even oscillation that starts at a control edge and runs well beyond two strips is flagged."""
    res, lat = flap_case
    sw = res.spanwise[0]
    for e in _edges(lat):
        g = sw.gamma.copy()
        h = 0.02 * np.abs(g).max()
        j = np.arange(e - 2, min(e + 10, g.size))
        g[j] += h * (-1.0) ** j
        assert _sign_change_run(g, skip=lat.control_edge_step[:-1]) >= _SAW_TOOTH_RUN
        new = sw.clone()
        new.gamma = g
        assert lattice_spanwise_stats(lat, [new])["sign_change_run"] >= _SAW_TOOTH_RUN


def test_synthetic_saw_tooth_is_flagged_with_an_edge_mask():
    """The odd-even case over the whole span stays flagged with steps near two edges masked."""
    _, g = _elliptic(40)
    g = g * (1.0 + 0.2 * (-1.0) ** np.arange(g.size))
    skip = np.zeros(g.size - 1, dtype=bool)
    skip[[18, 19, 20, 58, 59, 60]] = True
    assert _sign_change_run(g, skip=skip) >= _SAW_TOOTH_RUN


@pytest.mark.parametrize("n", [64, 160])
def test_straight_legs_saw_tooth_with_a_flap_is_flagged(straight_legs, n):
    """The ill-conditioned straight-leg lattice of a cambered wing with a flap is still flagged."""
    res = _solve(_wing([_ctrl("flap", 0.2, 0.5, 0.75, 10.0)], shape="6409"), "vlm", n, alpha_deg=0.0)
    assert res.details["lattice"].control_edge_step is not None
    assert _flagged(res)


def test_batch_and_single_paths_agree_with_an_edge_mask(flap_case):
    """spanwise_stats_batch with the edge mask gives the run of _sign_change_run with the same mask."""
    res, lat = flap_case
    rng = np.random.default_rng(3)
    G = np.array([_edge_feature(res.spanwise[0].gamma, e) for e in _edges(lat)]
                 + [res.spanwise[0].gamma + 0.01 * rng.normal(size=lat.n_strips) for _ in range(4)])
    arrays = {n: np.repeat(getattr(res.spanwise[0], n)[None, :], G.shape[0], axis=0)
              for n in ("Cl", "Cd_i", "alpha_eff", "alpha_i", "local_lift")}
    arrays["gamma"] = G
    arrays["control_edge_step"] = lat.control_edge_step
    stats = spanwise_stats_batch([surf.strips for surf in lat.surfaces], arrays)
    skip = lat.control_edge_step[:-1]
    assert [s["sign_change_run"] for s in stats] == [_sign_change_run(g, skip=skip) for g in G]
