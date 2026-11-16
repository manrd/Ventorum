"""Case-wide scale of the saw-tooth rule of the trust score (factor ``numerical``).

A surface without load (a vertical fin at zero sideslip and zero rates) has
a round-off circulation. Its strip-to-strip steps have random signs. When
the rule measured the steps of each surface against the largest circulation
of that surface, this noise gave a false saw-tooth warning and the rating
LOW, although the forces were correct. The rule now measures the steps of
all surfaces against the largest circulation of the case, and it does not
measure a surface below the threshold of that scale. With one surface the
two scales are the same, so the results do not change.
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
_ALPHAS = [-4.0, 0.0, 1.0, 4.0, 8.0, 12.0]
_SETTINGS = dict(solver_type="vlm", n_panels=24, n_chord=4)


def _old_run_rows(G, rel_tol=1e-6, skip=None):
    """The former rule: the steps of each surface against the largest |Gamma| of that surface."""
    G = np.asarray(G, dtype=float)
    K, m = G.shape
    if m < 4:
        return np.zeros(K, dtype=int)
    d = G[:, 1:] - G[:, :-1]
    scale = np.maximum(np.abs(G).max(axis=1), 1e-300)
    keep = np.abs(d) > rel_tol * scale[:, None]
    if skip is not None:
        keep &= ~np.broadcast_to(np.asarray(skip, dtype=bool), d.shape)
    s = np.where(keep, np.sign(d), 0.0)
    change = s[:, 1:] * s[:, :-1] < 0.0
    j = np.arange(change.shape[1])
    last_break = np.maximum.accumulate(np.where(change, -1, j[None, :]), axis=1)
    run = np.where(change, j[None, :] - last_break, 0)
    return run.max(axis=1).astype(int)


def _flagged(res) -> bool:
    return any(_KEY in w for w in res.totals.trust.warnings)


def _aircraft():
    """Wing, horizontal tail and vertical fin (all flat plates, no twist)."""
    w = vt.LiftingSurface(name="wing", semi_span=5.0, sections=[vt.WingSection(0.0, 1.2), vt.WingSection(1.0, 0.8)])
    t = vt.LiftingSurface(name="htail", semi_span=1.6, position=np.array([4.0, 0.0, 0.0]),
                          sections=[vt.WingSection(0.0, 0.7), vt.WingSection(1.0, 0.5)])
    f = vt.LiftingSurface(name="fin", semi_span=1.2, is_symmetric=False, dihedral=np.pi / 2,
                          position=np.array([4.0, 0.0, 0.0]),
                          sections=[vt.WingSection(0.0, 0.9), vt.WingSection(1.0, 0.5)])
    ac = vt.Aircraft(surfaces=[w, t, f], ref_point=np.array([0.3, 0.0, 0.0]))
    ac.compute_reference_values()
    return ac


def _wing():
    w = vt.LiftingSurface(name="w", semi_span=4.0, sections=[vt.WingSection(0.0, 1.0), vt.WingSection(1.0, 0.6)])
    ac = vt.Aircraft(surfaces=[w], ref_point=np.array([0.25, 0.0, 0.0]))
    ac.compute_reference_values()
    return ac


@pytest.fixture
def cpu_device():
    """Run the lattice solvers on the CPU."""
    from ventorum import gpu

    old = gpu.get_device()
    gpu.set_device("cpu")
    yield
    gpu.set_device(old)


def _old_rule_fires(res) -> bool:
    return max(_sign_change_run(sw.gamma, case_scale=None) for sw in res.spanwise) >= _SAW_TOOTH_RUN


def _trust_kw(res):
    t = res.totals
    return dict(AR=t.AR, condition=res.condition, CL=t.CL, CDi=t.CDi, CD_total=t.CD_total,
                n_panels=_SETTINGS["n_panels"], solver_type="vlm", n_chord=_SETTINGS["n_chord"])


# ----------------------------------------------------------------------------- unloaded fin
def test_unloaded_fin_does_not_trigger_the_rule_single_case(cpu_device):
    """vt.analyze of the wing, tail and fin at beta 0: no saw-tooth warning at any angle."""
    old_fired = 0
    for a in _ALPHAS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = vt.analyze(_aircraft(), FlightCondition(V_inf=30.0, alpha=np.radians(a)),
                             vt.SolverSettings(**_SETTINGS))
        assert not _flagged(res), a
        assert res.totals.trust.factors.get("numerical", 0.0) == 0.0
        old_fired += _old_rule_fires(res)
    # The former per-surface scale fired on the round-off of the fin at most angles.
    assert old_fired >= 2


def test_unloaded_fin_does_not_trigger_the_rule_sweep(cpu_device):
    """vt.analyze_sweep (the batch statistics) of the same aircraft: no saw-tooth warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = vt.analyze_sweep(_aircraft(), np.array(_ALPHAS), V_inf=30.0, settings=vt.SolverSettings(**_SETTINGS))
    assert len(res) == len(_ALPHAS)
    old_fired = 0
    for r in res:
        assert not _flagged(r)
        assert r.totals.trust.factors.get("numerical", 0.0) == 0.0
        # The spanwise results alone (the single-case rule) also give no warning.
        t = evaluate_aerodynamic_trust(spanwise_list=r.spanwise, **_trust_kw(r))
        assert not any(_KEY in w for w in t.warnings)
        old_fired += _old_rule_fires(r)
    assert old_fired >= 2


def test_batch_statistics_skip_the_unloaded_surface():
    """spanwise_stats_batch: a noise surface next to a loaded one gives no run; alone it keeps the old rule."""
    rng = np.random.default_rng(5)
    y = np.linspace(-1.0, 1.0, 30)
    wing = np.sqrt(1.0 - (0.98 * y) ** 2)
    # Round-off whose steps alternate in sign (the worst case of random noise).
    noise = 1e-14 * (-1.0) ** np.arange(12) * (1.0 + 0.2 * rng.random(12))
    G = np.concatenate([wing, noise])[None, :]
    arrays = {n: np.zeros_like(G) for n in ("Cl", "Cd_i", "alpha_eff", "alpha_i", "local_lift")}
    arrays["gamma"] = G
    strips = [slice(0, 30), slice(30, 42)]
    assert _old_run_rows(noise[None, :])[0] >= _SAW_TOOTH_RUN
    assert spanwise_stats_batch(strips, arrays)[0]["sign_change_run"] < _SAW_TOOTH_RUN
    # A case with only the noise surface: the case scale is its own maximum (former rule).
    arrays1 = {n: v[:, 30:] for n, v in arrays.items()}
    assert spanwise_stats_batch([slice(0, 12)], arrays1)[0]["sign_change_run"] == _old_run_rows(noise[None, :])[0]


# ----------------------------------------------------------------------------- loaded fin
def test_loaded_fin_with_a_saw_tooth_is_flagged(cpu_device):
    """At beta 5 deg the fin has load; an odd-even oscillation on it still triggers the rule."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = vt.analyze(_aircraft(), FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=np.radians(5.0)),
                         vt.SolverSettings(**_SETTINGS))
    assert not _flagged(res)
    fin = res.spanwise[2]
    gmax_case = max(float(np.abs(sw.gamma).max()) for sw in res.spanwise)
    assert float(np.abs(fin.gamma).max()) > 1e-3 * gmax_case
    new = fin.clone()
    j = np.arange(new.gamma.size)
    new.gamma = fin.gamma + 0.05 * float(np.abs(fin.gamma).max()) * (-1.0) ** j
    spanwise = [res.spanwise[0], res.spanwise[1], new]
    t = evaluate_aerodynamic_trust(spanwise_list=spanwise, **_trust_kw(res))
    assert any(_KEY in w for w in t.warnings)
    assert t.factors["numerical"] >= 0.40
    # The batch statistics give the same run.
    lat = res.details["lattice"]
    arrays = {n: np.concatenate([getattr(sw, n) for sw in spanwise])[None, :]
              for n in ("gamma", "Cl", "Cd_i", "alpha_eff", "alpha_i", "local_lift")}
    stats = spanwise_stats_batch([sf.strips for sf in lat.surfaces], arrays)[0]
    assert stats["sign_change_run"] >= _SAW_TOOTH_RUN
    assert stats["sign_change_run"] == max(
        _sign_change_run(sw.gamma, case_scale=gmax_case) for sw in spanwise)


def test_saw_tooth_on_the_wing_is_flagged_next_to_an_unloaded_fin():
    """The case scale does not hide a saw-tooth of the loaded surface."""
    y = np.linspace(-1.0, 1.0, 30)
    wing = np.sqrt(1.0 - (0.98 * y) ** 2) * (1.0 + 0.1 * (-1.0) ** np.arange(30))
    G = np.concatenate([wing, 1e-15 * np.random.default_rng(1).normal(size=10)])[None, :]
    arrays = {n: np.zeros_like(G) for n in ("Cl", "Cd_i", "alpha_eff", "alpha_i", "local_lift")}
    arrays["gamma"] = G
    assert spanwise_stats_batch([slice(0, 30), slice(30, 40)], arrays)[0]["sign_change_run"] >= _SAW_TOOTH_RUN


# ----------------------------------------------------------------------------- one surface: no change
def test_single_surface_run_is_the_former_run_bit_for_bit():
    """With one surface the case scale is the surface maximum: the run equals the former rule on all rows."""
    rng = np.random.default_rng(23)
    rows = []
    for _ in range(400):
        m = 40
        g = rng.normal(size=m).cumsum() * 10.0 ** rng.integers(-20, 3)
        if rng.random() < 0.3:
            g = g + 0.5 * np.abs(g).max() * (-1.0) ** np.arange(m)
        if rng.random() < 0.3:
            g = np.round(g, 1)
        rows.append(g)
    rows += [np.zeros(40), np.full(40, 1e-310), np.where(np.arange(40) % 2, 1e-305, 2e-305)]
    G = np.array(rows)
    skip = rng.random(39) < 0.1
    for sk in (None, skip):
        old = _old_run_rows(G, skip=sk)
        assert np.array_equal(_sign_change_run_rows(G, skip=sk, case_scale=np.abs(G).max(axis=1)), old)
        assert np.array_equal(_sign_change_run_rows(G, skip=sk), old)
        arrays = {n: np.zeros_like(G) for n in ("Cl", "Cd_i", "alpha_eff", "alpha_i", "local_lift")}
        arrays["gamma"] = G
        if sk is not None:
            arrays["control_edge_step"] = np.append(sk, False)
        stats = spanwise_stats_batch([slice(0, 40)], arrays)
        assert [s["sign_change_run"] for s in stats] == old.tolist()
        assert [_sign_change_run(g, skip=sk, case_scale=float(np.abs(g).max())) for g in G] == old.tolist()


def test_case_scale_zero_or_not_finite_keeps_the_former_rule():
    g = np.array([0.0, 1e-14, -1e-14, 1e-14, -1e-14, 1e-14, 0.0])
    old = int(_old_run_rows(g[None, :])[0])
    assert old >= _SAW_TOOTH_RUN
    for cs in (None, 0.0, np.nan, np.inf):
        assert _sign_change_run(g, case_scale=cs) == old
    assert _sign_change_run(g, case_scale=1.0) == 0


@pytest.mark.parametrize("solver,n_chord", [("vlm", 4), ("linear", None)])
def test_single_surface_trust_result_is_unchanged(cpu_device, solver, n_chord):
    """One surface: the trust result of the solver equals the result of the former rule, bit for bit."""
    for a in (-2.0, 0.0, 5.0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = vt.analyze(_wing(), FlightCondition(V_inf=30.0, alpha=np.radians(a)),
                             vt.SolverSettings(solver_type=solver, n_panels=24, n_chord=n_chord))
        sw = res.spanwise[0]
        t = res.totals
        old_stats = {"max_local_cl": float(np.abs(sw.Cl).max()),
                     "max_alpha_eff_deg": float(np.degrees(np.abs(sw.alpha_eff).max())),
                     "sign_change_run": int(_old_run_rows(sw.gamma[None, :])[0])}
        kw = dict(AR=t.AR, condition=res.condition, CL=t.CL, CDi=t.CDi, CD_total=t.CD_total, n_panels=24,
                  solver_type=solver, n_chord=n_chord)
        old = evaluate_aerodynamic_trust(spanwise_stats=old_stats, **kw).to_dict()
        assert evaluate_aerodynamic_trust(spanwise_list=res.spanwise, **kw).to_dict() == old


# ----------------------------------------------------------------------------- GPU kernel
def _warp_run(dev, G, cut):
    """Return the sign-change run of the Warp loads kernel on device *dev* (two surfaces split at *cut*)."""
    import warp as wp

    from ventorum.gpu.kernels import kernels

    out_runs = []
    for k in (kernels("float64"), kernels("float64", kinks=False)):
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
            wp.array(np.zeros(m, dtype=np.int32), dtype=wp.int32, device=dev), out])
        out_runs.append(out.numpy()[:, 21].astype(int))
    assert np.array_equal(out_runs[0], out_runs[1])
    return out_runs[0]


def _noise_rows(seed=29, K=200, m=40, cut=25):
    """Rows with a loaded first surface and, in some rows, a second surface of round-off only."""
    rng = np.random.default_rng(seed)
    G = rng.normal(size=(K, m)).cumsum(axis=1)
    G[::3] += 0.5 * (-1.0) ** np.arange(m)
    G[::2, cut:] = 10.0 ** rng.integers(-16, -3, size=(K // 2, 1)) * rng.normal(size=(K // 2, m - cut))
    G[1::4] = 0.0
    G[5::8, :cut] = 0.0
    return G


def _cpu_run(G, cut):
    arrays = {n: np.zeros_like(G) for n in ("Cl", "Cd_i", "alpha_eff", "alpha_i", "local_lift")}
    arrays["gamma"] = G
    return np.array([s["sign_change_run"] for s in spanwise_stats_batch([slice(0, cut), slice(cut, None)], arrays)])


@pytest.mark.slow
def test_warp_statistics_kernel_uses_the_case_scale():
    """The Warp loads kernel (CPU device, both kernel sets) gives the CPU run with the case-wide scale."""
    pytest.importorskip("warp")
    G, cut = _noise_rows(), 25
    ref = _cpu_run(G, cut)
    assert not np.array_equal(ref, np.maximum(_old_run_rows(G[:, :cut]), _old_run_rows(G[:, cut:])))
    assert np.array_equal(_warp_run("cpu", G, cut), ref)


@pytest.mark.gpu
def test_warp_statistics_kernel_uses_the_case_scale_on_the_gpu():
    wp = pytest.importorskip("warp")
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("No CUDA device for Warp.")
    G, cut = _noise_rows(31), 25
    assert np.array_equal(_warp_run("cuda:0", G, cut), _cpu_run(G, cut))


@pytest.mark.gpu
def test_gpu_sweep_of_wing_tail_fin_gives_the_cpu_trust_result():
    """The wing, tail and fin sweep on the GPU: the CPU warnings, scores and ratings, and no saw-tooth warning."""
    from ventorum import gpu

    if not gpu.available():
        pytest.skip(f"The GPU pipelines cannot run: {gpu.unavailable_reason()}")
    old_dev, old_prec = gpu.get_device(), gpu._precision
    try:
        st = vt.SolverSettings(**_SETTINGS)
        runs = {}
        for device in ("cpu", "gpu"):
            gpu.set_device(device)
            gpu.set_precision("float64")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                runs[device] = vt.VortexLatticeSolver().solve_sweep(
                    _aircraft(), FlightCondition(V_inf=30.0), st, np.radians(_ALPHAS))
        for r, s in zip(runs["cpu"], runs["gpu"]):
            assert s.details["device"] == "gpu"
            assert not _flagged(r) and not _flagged(s)
            assert r.totals.trust.rating == s.totals.trust.rating
            assert r.totals.trust.score == pytest.approx(s.totals.trust.score, rel=0.0, abs=1e-12)
            assert sorted(r.totals.trust.warnings) == sorted(s.totals.trust.warnings)
    finally:
        gpu.set_device(old_dev)
        gpu.set_precision(old_prec)
