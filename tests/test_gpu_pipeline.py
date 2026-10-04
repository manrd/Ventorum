# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests of the GPU pipelines (:mod:`ventorum.gpu`).

The parity tests need an NVIDIA CUDA GPU with PyTorch and NVIDIA Warp; they
are marked ``gpu`` and are skipped without one. The tests of the selection
logic and of the batch helpers run everywhere.

Tolerances: float64 results equal the CPU results to round-off (the order of
the sums differs). float32 results are within the float32 error of the
kernels (owner direction D-19 accepts float32 for the GPU paths).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import ventorum as vt
from ventorum import gpu

TOL = {"float64": 1e-11, "float32": 2e-5}


def _need_gpu():
    if not gpu.available():
        pytest.skip(f"The GPU pipelines cannot run: {gpu.unavailable_reason()}")


@pytest.fixture
def gpu_device():
    """Run the test with the GPU device selected; restore the settings after it."""
    _need_gpu()
    old_dev, old_prec = gpu.get_device(), gpu._precision
    yield
    gpu.set_device(old_dev)
    gpu.set_precision(old_prec)


def _polar(stall: bool = False):
    a = np.radians(np.arange(-10.0, 25.0, 1.0))
    cl = 2 * np.pi * a
    if stall:
        cl = np.where(a < np.radians(12), cl, 2 * np.pi * np.radians(12) - 3.0 * (a - np.radians(12)))
    return vt.TabulatedAirfoil(alpha=a, Cl_data=cl, Cd_data=0.008 + 0.01 * (a / np.radians(10)) ** 2,
                               Cm_data=-0.05 + 0.0 * a)


def _aircraft(tail: bool, airfoil=None):
    af = airfoil if airfoil is not None else vt.LinearAirfoil(Cd0=0.008, Cm0=-0.04)
    s = [vt.LiftingSurface(name="wing", semi_span=5.0, sweep_le=np.radians(2.0),
                           sections=[vt.WingSection(y_frac=0.0, chord=1.25, airfoil=af),
                                     vt.WingSection(y_frac=1.0, chord=0.9, airfoil=af, twist=np.radians(-2.0))])]
    if tail:
        s.append(vt.LiftingSurface(name="tail", semi_span=1.5, position=np.array([3.5, 0.0, 0.5]),
                                   sections=[vt.WingSection(y_frac=0.0, chord=0.6, airfoil=af),
                                             vt.WingSection(y_frac=1.0, chord=0.4, airfoil=af)]))
    return vt.Aircraft(surfaces=s)


def _sweep(solver, ac, st, alphas, device, precision="float64", h=None, beta=0.0, phi=0.0):
    gpu.set_device(device)
    gpu.set_precision(precision)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return solver.solve_sweep(ac, vt.FlightCondition(V_inf=25.0, h=h, beta=beta, phi=phi), st, alphas)


def _assert_close(ref, new, tol):
    # The coefficients are compared on the scale of the lift coefficient (a
    # small moment, for example the roll moment of a case with a small
    # sideslip, is a near-cancelling sum of large terms).
    cl_scale = max(float(np.max(np.abs([r.totals.CL for r in ref]))), 1e-3)
    for name in ("CL", "CDi", "Cm", "CY", "Cl", "Cn", "CDi_nearfield", "CD_total"):
        a = np.array([getattr(r.totals, name) or 0.0 for r in ref])
        b = np.array([getattr(r.totals, name) or 0.0 for r in new])
        scale = max(float(np.max(np.abs(a))), cl_scale if name not in ("CDi", "CD_total") else 1e-3)
        assert np.max(np.abs(a - b)) <= tol * scale, (name, np.max(np.abs(a - b)) / scale)
    ga = np.concatenate([r.details["gamma"] for r in ref])
    gb = np.concatenate([r.details["gamma"] for r in new])
    assert np.max(np.abs(ga - gb)) <= 20 * tol * np.max(np.abs(ga))
    for r, s in zip(ref, new):
        assert s.details["device"] == "gpu"
        assert r.converged == s.converged
        assert r.totals.trust.rating == s.totals.trust.rating
        assert len(r.spanwise) == len(s.spanwise)
        for sw_r, sw_s in zip(r.spanwise, s.spanwise):
            assert sw_r.surface_name == sw_s.surface_name
            assert np.max(np.abs(sw_r.Cl - sw_s.Cl)) <= 20 * tol * max(np.max(np.abs(sw_r.Cl)), 1e-6)


# ----------------------------------------------------------------------------- selection (no GPU needed)
def test_device_and_precision_settings_are_checked():
    old_dev, old_prec = gpu.get_device(), gpu._precision
    try:
        with pytest.raises(ValueError):
            gpu.set_device("tpu")
        with pytest.raises(ValueError):
            gpu.set_precision("float16")
        gpu.set_device("cpu")
        assert gpu.get_device() == "cpu"
        assert not gpu.use_gpu(1e12)
        gpu.set_precision("float64")
        assert gpu.get_precision() == "float64"
        gpu.set_precision("auto")
        assert gpu.get_precision() in ("float32", "float64")
        if not gpu.available():
            with pytest.raises(ValueError):
                gpu.set_device("gpu")
    finally:
        gpu.set_device(old_dev)
        gpu.set_precision(old_prec)


def test_cpu_device_never_uses_the_gpu():
    old = gpu.get_device()
    try:
        gpu.set_device("cpu")
        ac = _aircraft(False)
        st = vt.SolverSettings(solver_type="linear", n_panels=10)
        res = vt.LinearLLTSolver().solve_sweep(ac, vt.FlightCondition(V_inf=25.0), st, np.radians([0.0, 4.0]))
        assert all("device" not in r.details for r in res)
    finally:
        gpu.set_device(old)


def test_auto_device_keeps_small_solves_on_the_cpu():
    old = gpu.get_device()
    try:
        gpu.set_device("auto")
        assert not gpu.use_gpu(gpu.min_work() * 0.5)
    finally:
        gpu.set_device(old)


def test_auto_rule_without_a_cost_model(monkeypatch):
    monkeypatch.setattr(gpu, "available", lambda: True)
    monkeypatch.setattr(gpu, "cost_model", lambda family: None)
    old = gpu.get_device()
    try:
        gpu.set_device("auto")
        for family in gpu.FAMILIES:
            assert not gpu.use_gpu(gpu.min_work(family) * 0.5, family, gpu.min_cases(family) - 1, 10)
            assert gpu.use_gpu(gpu.min_work(family), family, 1, 10)
            assert gpu.use_gpu(1.0, family, gpu.min_cases(family), 10)
    finally:
        gpu.set_device(old)


def _model():
    # CPU: 1 ms per case at 100 panels, 100 ms per case at 1000 panels. GPU:
    # 10 ms for one case, 1 ms more for 33 cases, 2 ms more for 257 cases.
    return {"n": [100, 1000],
            "cpu": {"k": [1, 33], "t": [[1e-3, 33e-3], [1e-1, 3.3]]},
            "gpu": {"k": [1, 33, 257], "t": [[1e-2, 1.1e-2, 1.3e-2], [1e-2, 1.1e-2, 1.3e-2]]}}


def test_auto_rule_with_a_cost_model(monkeypatch):
    model = _model()
    monkeypatch.setattr(gpu, "available", lambda: True)
    monkeypatch.setattr(gpu, "cost_model", lambda family: model)
    old = gpu.get_device()
    try:
        gpu.set_device("auto")
        assert not gpu.use_gpu(1e12, "linear", 1, 100)   # 1 ms on the CPU, 10 ms on the GPU
        assert gpu.use_gpu(0.0, "linear", 1, 1000)       # 100 ms on the CPU, 10 ms on the GPU
        assert gpu.use_gpu(0.0, "vlm", 64, 100)          # 64 ms on the CPU, 11.3 ms on the GPU
        # Power law between the panel counts, end slopes outside them.
        assert gpu.estimate_time(model, "cpu", 316.227766, 1) == pytest.approx(1e-2, rel=1e-6)
        assert gpu.estimate_time(model, "cpu", 10000, 1) == pytest.approx(10.0, rel=1e-6)
        # Linear in the cases between the batch sizes, the last slope after them.
        assert gpu.estimate_time(model, "cpu", 100, 1001) == pytest.approx(1.001, rel=1e-9)
        assert gpu.estimate_time(model, "gpu", 100, 17) == pytest.approx(1.05e-2, rel=1e-9)
        assert gpu.estimate_time(model, "gpu", 100, 145) == pytest.approx(1.2e-2, rel=1e-9)
        assert gpu.estimate_time(model, "gpu", 100, 481) == pytest.approx(1.5e-2, rel=1e-9)
    finally:
        gpu.set_device(old)


def test_cost_model_ignores_a_bad_profile_entry(monkeypatch):
    from ventorum.hardware import profile

    good = _model()
    bad_k = {"k": [2, 33], "t": [[1e-3, 33e-3], [1e-1, 3.3]]}
    bad_t = {"k": [1, 33], "t": [[1e-3, 33e-3]]}
    for entry, ok in ((good, True), (dict(good, cpu=bad_k), False), (dict(good, cpu=bad_t), False),
                      (dict(good, n=[]), False), (dict(good, gpu=None), False), ("x", False), (None, False)):
        monkeypatch.setattr(profile, "tuned_setting", lambda *keys, default=None, e=entry: e)
        assert (gpu.cost_model("linear") is not None) == ok


# ----------------------------------------------------------------------------- GPU parity
@pytest.mark.gpu
@pytest.mark.parametrize("precision", ["float64", "float32"])
@pytest.mark.parametrize("tail", [False, True])
@pytest.mark.parametrize("h,beta", [(None, 0.0), (3.0, 0.0), (None, 0.05)])
def test_gpu_linear_lifting_line_equals_cpu(gpu_device, precision, tail, h, beta):
    ac = _aircraft(tail)
    st = vt.SolverSettings(solver_type="linear", n_panels=24, spacing="cosine")
    alphas = np.radians(np.linspace(-4.0, 12.0, 9))
    ref = _sweep(vt.LinearLLTSolver(), ac, st, alphas, "cpu", h=h, beta=beta)
    new = _sweep(vt.LinearLLTSolver(), ac, st, alphas, "gpu", precision, h=h, beta=beta)
    _assert_close(ref, new, TOL[precision])


@pytest.mark.gpu
@pytest.mark.parametrize("precision", ["float64", "float32"])
@pytest.mark.parametrize("tail", [False, True])
@pytest.mark.parametrize("h,beta", [(None, 0.0), (3.0, 0.0), (None, 0.05)])
def test_gpu_nonlinear_lifting_line_equals_cpu(gpu_device, precision, tail, h, beta):
    ac = _aircraft(tail, _polar())
    st = vt.SolverSettings(solver_type="nonlinear", n_panels=24, spacing="cosine", tolerance=1e-9)
    alphas = np.radians(np.linspace(-4.0, 12.0, 9))
    ref = _sweep(vt.NonlinearSolver(), ac, st, alphas, "cpu", h=h, beta=beta)
    new = _sweep(vt.NonlinearSolver(), ac, st, alphas, "gpu", precision, h=h, beta=beta)
    _assert_close(ref, new, max(TOL[precision], 1e-8))
    assert [r.iterations for r in ref] == [r.iterations for r in new] or precision == "float32"


@pytest.mark.gpu
def test_gpu_nonlinear_lifting_line_past_stall_follows_the_cpu_branch(gpu_device):
    ac = _aircraft(False, _polar(stall=True))
    st = vt.SolverSettings(solver_type="nonlinear", n_panels=20, spacing="cosine")
    alphas = np.radians(np.linspace(6.0, 18.0, 13))
    ref = _sweep(vt.NonlinearSolver(), ac, st, alphas, "cpu")
    new = _sweep(vt.NonlinearSolver(), ac, st, alphas, "gpu", "float64")
    a = np.array([r.totals.CL for r in ref])
    b = np.array([r.totals.CL for r in new])
    assert np.max(np.abs(a - b)) <= 1e-5 * np.max(np.abs(a))
    assert [r.converged for r in ref] == [r.converged for r in new]


@pytest.mark.gpu
@pytest.mark.parametrize("precision", ["float64", "float32"])
@pytest.mark.parametrize("tail", [False, True])
@pytest.mark.parametrize("h,beta", [(None, 0.0), (1.0, 0.0), (None, 0.05)])
def test_gpu_vortex_lattice_equals_cpu(gpu_device, precision, tail, h, beta):
    ac = _aircraft(tail)
    st = vt.SolverSettings(solver_type="vlm", n_panels=16, n_chord=3, spacing="cosine")
    alphas = np.radians(np.linspace(-4.0, 12.0, 7))
    ref = _sweep(vt.HorseshoeSolver(), ac, st, alphas, "cpu", h=h, beta=beta)
    new = _sweep(vt.HorseshoeSolver(), ac, st, alphas, "gpu", precision, h=h, beta=beta)
    _assert_close(ref, new, TOL[precision])


@pytest.mark.gpu
@pytest.mark.parametrize("precision", ["float64", "float32"])
@pytest.mark.parametrize("kind", ["linear", "nonlinear", "vlm"])
def test_gpu_solver_options_equal_cpu(gpu_device, precision, kind):
    """Parity with the body-axis wake, no symmetry folding, bank in ground effect and one chordwise panel.

    The VLM case uses a single chordwise panel; the lifting-line solvers
    ignore the chordwise count. The bank (phi) tilts the ground plane, so
    the case is asymmetric and in ground effect.
    """
    solver = {"linear": vt.LinearLLTSolver, "nonlinear": vt.NonlinearSolver, "vlm": vt.HorseshoeSolver}[kind]()
    ac = _aircraft(True, _polar() if kind == "nonlinear" else None)
    st = vt.SolverSettings(solver_type=kind, n_panels=16, n_chord=1 if kind == "vlm" else None,
                           wake_alignment="body", use_symmetry=False,
                           tolerance=1e-9 if kind == "nonlinear" else 1e-6)
    alphas = np.radians(np.linspace(-4.0, 12.0, 7))
    ref = _sweep(solver, ac, st, alphas, "cpu", h=3.0, phi=0.05)
    new = _sweep(solver, ac, st, alphas, "gpu", precision, h=3.0, phi=0.05)
    tol = max(TOL[precision], 1e-8) if kind == "nonlinear" else TOL[precision]
    _assert_close(ref, new, tol)
    if kind == "nonlinear":
        assert [r.iterations for r in ref] == [r.iterations for r in new] or precision == "float32"


@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["linear", "nonlinear", "vlm"])
def test_gpu_single_solve_equals_cpu(gpu_device, kind):
    solver = {"linear": vt.LinearLLTSolver, "nonlinear": vt.NonlinearSolver, "vlm": vt.HorseshoeSolver}[kind]()
    ac = _aircraft(True, _polar() if kind == "nonlinear" else None)
    st = vt.SolverSettings(solver_type=kind, n_panels=20, n_chord=3 if kind == "vlm" else None)
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(5.0))
    gpu.set_device("cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = solver.solve(ac, cond, st)
        gpu.set_device("gpu")
        gpu.set_precision("float64")
        new = solver.solve(ac, cond, st)
    assert new.details["device"] == "gpu"
    _assert_close([ref], [new], TOL["float64"])


@pytest.mark.gpu
def test_gpu_batch_is_split_into_chunks(gpu_device, monkeypatch):
    from ventorum.gpu import engine

    ac = _aircraft(False)
    st = vt.SolverSettings(solver_type="linear", n_panels=16)
    alphas = np.radians(np.linspace(-4.0, 12.0, 11))
    ref = _sweep(vt.LinearLLTSolver(), ac, st, alphas, "gpu", "float64")
    monkeypatch.setattr(engine, "CHUNK_BYTES", 1)   # one case per chunk
    new = _sweep(vt.LinearLLTSolver(), ac, st, alphas, "gpu", "float64")
    _assert_close(ref, new, 1e-12)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("min_n", [0, 128])
def test_gpu_batch_solve_equals_a_direct_solve(gpu_device, monkeypatch, dtype, min_n):
    import torch

    from ventorum.gpu import engine
    from ventorum.gpu.kernels import kernels

    # min_n 0: the iterative solve (reference inverse and defect correction);
    # 128: the direct solve of small systems.
    monkeypatch.setattr(engine, "REF_MIN_N", min_n)
    rng = np.random.default_rng(0)
    n, K = 40, 24
    base = rng.normal(size=(n, n)) + n * np.eye(n)
    M = np.stack([base + 0.05 * k / K * rng.normal(size=(n, n)) for k in range(K)]).astype(dtype)
    diag = n + rng.random((K, n))
    b = rng.normal(size=(K, n))
    MT = torch.as_tensor(np.ascontiguousarray(M.transpose(0, 2, 1)), device="cuda")
    for d in (None, diag):
        x = engine.batched_solve(MT, torch.as_tensor(b, device="cuda"),
                                 diag64=None if d is None else torch.as_tensor(d, device="cuda"),
                                 kern=kernels(dtype)).cpu().numpy()
        A = M.astype(float) if d is None else d[:, :, None] * np.eye(n) - M.astype(float)
        ref = np.linalg.solve(A, b[..., None])[..., 0]
        tol = 1e-12 if dtype == "float64" else 1e-9
        assert np.max(np.abs(x - ref)) <= tol * np.max(np.abs(ref))


@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["linear", "nonlinear", "vlm"])
@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_gpu_iterative_solves_equal_cpu(gpu_device, monkeypatch, kind, precision):
    from ventorum.gpu import engine

    monkeypatch.setattr(engine, "REF_MIN_N", 0)   # the iterative solve also for small systems
    solver = {"linear": vt.LinearLLTSolver, "nonlinear": vt.NonlinearSolver, "vlm": vt.HorseshoeSolver}[kind]()
    ac = _aircraft(True, _polar() if kind == "nonlinear" else None)
    st = vt.SolverSettings(solver_type=kind, n_panels=20, n_chord=3 if kind == "vlm" else None, tolerance=1e-9)
    alphas = np.radians(np.linspace(-4.0, 12.0, 9))
    ref = _sweep(solver, ac, st, alphas, "cpu")
    new = _sweep(solver, ac, st, alphas, "gpu", precision)
    _assert_close(ref, new, max(TOL[precision], 1e-8))


@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["linear", "nonlinear"])
def test_gpu_polars_follow_a_change_of_the_airfoils(gpu_device, kind):
    # The GPU keeps the polars of a lattice; a change of an airfoil (on the
    # same lattice object) must reach the next solve, as on the CPU.
    solver = {"linear": vt.LinearLLTSolver, "nonlinear": vt.NonlinearSolver}[kind]()
    lin = vt.LinearAirfoil(Cd0=0.008, Cm0=-0.04)
    tab = _polar()
    ac = _aircraft(False, tab if kind == "nonlinear" else lin)
    st = vt.SolverSettings(solver_type=kind, n_panels=12)
    conds = [vt.FlightCondition(V_inf=25.0, alpha=np.radians(a)) for a in (0.0, 4.0, 8.0)]
    lattice = solver.build(ac, st, conds[0])
    ac.compute_reference_values()

    def batch(device):
        gpu.set_device(device)
        gpu.set_precision("float64")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return solver.solve_batch(lattice, conds, st, ac.S_ref, ac.b_ref, ac.c_ref)

    _assert_close(batch("cpu"), batch("gpu"), TOL["float64"])
    if kind == "linear":
        lin.Cd0 = 0.02
        lin.Cm0 = -0.08
    else:
        tab.Cd_data = tab.Cd_data * 1.5
        tab.Cm_data = tab.Cm_data - 0.03
    ref, new = batch("cpu"), batch("gpu")
    _assert_close(ref, new, TOL["float64"])
    assert ref[1].totals.CD_total == pytest.approx(new[1].totals.CD_total, rel=1e-11)


@pytest.mark.gpu
@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_gpu_ground_effect_sweep_equals_cpu(gpu_device, precision):
    from ventorum.ground_effect import GroundEffectSweep

    wing = vt.LiftingSurface(name="wing", semi_span=3.0, sections=[vt.WingSection(y_frac=0.0, chord=1.0),
                                                                   vt.WingSection(y_frac=1.0, chord=0.7)])
    st = vt.SolverSettings(solver_type="vlm", n_panels=10, n_chord=4)
    heights = [0.15, 0.4, 1.0]
    out = {}
    for device in ("cpu", "gpu"):
        gpu.set_device(device)
        gpu.set_precision(precision)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out[device] = GroundEffectSweep(wing, settings=st).run_sweep(
                heights, alphas_deg=[-2.0, 4.0, 9.0], phis_deg=[0.0, 3.0], V_inf=30.0, height_ref="min")
    ref, new = out["cpu"], out["gpu"]
    assert np.array_equal(ref.is_strike, new.is_strike)
    tol = TOL[precision]
    for name in ("CL", "CDi", "Cm", "Cl", "Cn", "CY"):
        a, b = getattr(ref, name), getattr(new, name)
        ok = np.isfinite(a)
        assert np.array_equal(ok, np.isfinite(b))
        assert np.max(np.abs(a[ok] - b[ok])) <= tol * max(np.max(np.abs(ref.CL[ok])), 1e-3), name
    assert all(r.solver_result.details.get("device") == "gpu" for r in new.results if r is not None)


@pytest.mark.gpu
def test_no_nonwritable_warning(gpu_device):
    """The first GPU solve gives no torch warning about read-only arrays (review round 5, C4)."""
    gpu.set_device("gpu")
    wing = vt.LiftingSurface(semi_span=5.0, sections=[vt.WingSection(y_frac=0.0, chord=1.0),
                                                      vt.WingSection(y_frac=1.0, chord=0.5)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for solver in ("vlm", "linear", "nonlinear"):
            for h in (None, 2.0):
                cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), h=h)
                r = vt.analyze(wing.clone(), cond, vt.SolverSettings(solver_type=solver, n_panels=24))
                assert r.details.get("device") == "gpu"
    assert not [w for w in caught if "not writable" in str(w.message)]
