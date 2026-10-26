# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Parity tests of the compiled vector-Jacobian-product (VJP) kernels of the adjoint gradients.

The PyTorch reference copy (:mod:`ventorum.sensitivity.reference`) is the
reference. Every VJP backend (numpy, Numba, Cython, NVIDIA Warp) must give
the same products and the same input gradients to round-off: 1e-12 in
float64 and 1e-5 in float32, relative to the largest component of each
gradient.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import ventorum as vt
from ventorum.sensitivity import kernels as K

_TOL = {"float64": 1e-12, "float32": 1e-5}


@pytest.fixture(autouse=True)
def _restore_vjp_backend():
    """Restore the forced VJP backend after every test of this file."""
    old = K.get_forced_backend()
    yield
    K.set_vjp_backend(old)


def _need_cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or not K.available("warp", "cuda"):
        pytest.skip("The Warp VJP kernels need a CUDA GPU and the warp-lang package.")
    return torch


def _section() -> vt.LinearAirfoil:
    return vt.LinearAirfoil(name="Sec", a0=2.0 * np.pi, alpha_L0=np.radians(-2.0), Cd0=0.008, Cm0=-0.05)


def _wing(name: str = "Wing", span: float = 5.0, position=None, sweep: float = 15.0) -> vt.LiftingSurface:
    return vt.LiftingSurface(
        name=name, semi_span=span, sweep_le=np.radians(sweep), dihedral=np.radians(5.0),
        position=np.zeros(3) if position is None else np.asarray(position, dtype=float),
        sections=[vt.WingSection(y_frac=0.0, chord=2.0 * span / 5.0, airfoil=_section()),
                  vt.WingSection(y_frac=1.0, chord=span / 5.0, airfoil=_section())])


def _model_cases() -> list[tuple]:
    """Return (id, aircraft, condition, settings, wrt, outputs) of the captured kernel calls.

    * ``vlm-tail-ground``: wing and tail in ground effect with sideslip
      (images, cross-surface cores, leg edges, the full system).
    * ``vlm-fold``: symmetric state with the fold (two sources per unknown,
      folded loads).
    * ``linear``: lifting line (collocation points on their own bound
      segments).
    """
    tail = _wing("Tail", 2.0, position=[4.0, 0.0, 0.5], sweep=0.0)
    ac = vt.Aircraft(surfaces=[_wing(), tail])
    ac.compute_reference_values()
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=np.radians(3.0), h=2.0)
    st = vt.SolverSettings(solver_type="vlm", n_panels=8, n_chord=3)
    cases = [("vlm-tail-ground", ac, cond, st, ("nodes", "alpha", "h"), ("CL", "CY", "CDi", "Cm"))]
    ac = vt.Aircraft(surfaces=[_wing()])
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    st = vt.SolverSettings(solver_type="vlm", n_panels=8, n_chord=3, use_symmetry=True)
    cases.append(("vlm-fold", ac, cond, st, ("nodes", "alpha"), ("CL", "CDi", "Cm")))
    ac = vt.Aircraft(surfaces=[_wing(sweep=0.0)])
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=np.radians(3.0))
    st = vt.SolverSettings(solver_type="linear", n_panels=12)
    cases.append(("linear", ac, cond, st, ("nodes", "alpha"), ("CL", "CY", "CDi", "Cm")))
    # Leg chains of a cambered mean-line lattice, with the hinge node of a
    # deflected part-span flap on the chain, in ground effect.
    flap = vt.ControlSurface("flap", 0.25, 0.75, 0.7, np.radians(8.0), symmetric=False)
    ac = vt.Aircraft(surfaces=[vt.LiftingSurface(
        name="Wing", semi_span=4.0, dihedral=np.radians(3.0), controls=[flap],
        sections=[vt.WingSection(y_frac=0.0, chord=1.0, shape="4412"),
                  vt.WingSection(y_frac=1.0, chord=0.8, shape="6409")])])
    ac.compute_reference_values()
    st = vt.SolverSettings(solver_type="vlm", n_panels=6, n_chord=4)
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(3.0), beta=np.radians(2.0))
    cases.append(("vlm-chain", ac, cond, st, ("nodes", "alpha"), ("CL", "CY", "CDi", "Cm")))
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(3.0), beta=np.radians(2.0), h=2.0)
    cases.append(("vlm-chain-ground", ac, cond, st, ("nodes", "alpha", "h"), ("CL", "CY", "CDi", "Cm")))
    return cases


_CASE_IDS = [c[0] for c in _model_cases()]

# Cases checked in float64 only. In the ground case of this wing the float32
# PyTorch reference itself is 6 % from the float64 reference in the
# wake-direction gradient at the leg points (the same without camber and
# without the chains, so a property of the geometry in float32); the float32
# kernels and the float32 reference then differ by more than the float32
# tolerance. The float64 products agree to about 1e-15.
_FLOAT64_ONLY = {"vlm-chain-ground"}


def _captured_calls(case_id: str, device: str, precision: str):  # type: ignore[no-untyped-def]
    """Return (model, calls) with the O(n^2) calls of one reference forward pass.

    Each call is (name, args, kwargs) of ``_normal_matrix``,
    ``_tensor_unknowns`` (the system rows of the lifting line), ``_induced``
    or ``_trefftz_wash`` of the reference model, with its tensors on
    *device* in *precision*.
    """
    import torch

    from ventorum.sensitivity import adjoint

    case = {c[0]: c for c in _model_cases()}[case_id]
    _, ac, cond, st, wrt, outs = case
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prep = adjoint._prepare_case(ac, cond, st, "body", wrt, outs)
    model, base = adjoint.build_reference(prep["aircraft"], cond, prep["settings"], prep["lattice"], "body",
                                          device, precision, prep["symmetric"])
    calls = []
    for name in _CAPTURED:
        orig = getattr(model, name)

        def wrap(*args, _orig=orig, _name=name, **kwargs):  # type: ignore[no-untyped-def]
            calls.append((_name, args, kwargs))
            return _orig(*args, **kwargs)

        setattr(model, name, wrap)
    t = adjoint.make_input_tensors(model, base, wrt, device, precision)
    g = torch.as_tensor(prep["g_unknown"], dtype=model.b["dtype"], device=model.b["device"])
    with torch.no_grad():
        model.forward(t, g)
    for name in _CAPTURED:
        delattr(model, name)
    return model, calls


_CAPTURED = ("_normal_matrix", "_tensor_unknowns", "_induced", "_trefftz_wash")


# Floating-point arguments that the reference copy keeps constant: the source
# signs of the system rows (+1, or -1 for ground images).
_CONSTANT_ARGS = {"_normal_matrix": (9,), "_tensor_unknowns": (8,)}


def _grads(model, name: str, args: tuple, kwargs: dict, backend: str | None, seed: int):  # type: ignore[no-untyped-def]
    """Return (output, input gradients) of one captured call for random output weights."""
    import torch

    model.b["vjp_backend"] = backend
    leaves = []
    call_args = []
    for pos, a in enumerate(args):
        if isinstance(a, torch.Tensor) and a.is_floating_point() and pos not in _CONSTANT_ARGS.get(name, ()):
            leaf = a.detach().clone().requires_grad_(True)
            leaves.append(leaf)
            call_args.append(leaf)
        else:
            call_args.append(a)
    if name == "_tensor_unknowns":
        # Lifting line: the reference contracts the velocity tensor with the
        # strip normals; the kernels give the normal component per source.
        N = call_args.pop()
        if backend is None:
            out = torch.einsum("ijk,ik->ij", model._tensor_unknowns(*call_args, **kwargs), N)
        else:
            out = model._normal_matrix(call_args[0], N, *call_args[1:], **kwargs)
    else:
        out = getattr(model, name)(*call_args, **kwargs)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    w = torch.randn(out.shape, generator=gen, dtype=torch.float64).to(dtype=out.dtype, device=out.device)
    grads = torch.autograd.grad((out * w).sum(), leaves, allow_unused=True)
    model.b["vjp_backend"] = None
    return out.detach().cpu().double().numpy(), [None if gr is None else gr.detach().cpu().double().numpy()
                                                 for gr in grads]


def _check_parity(case_id: str, backend: str, device: str, precision: str) -> None:
    import torch

    model, calls = _captured_calls(case_id, device, precision)
    names = {c[0] for c in calls}
    rows = "_tensor_unknowns" if case_id == "linear" else "_normal_matrix"
    assert names == {rows, "_induced", "_trefftz_wash"}, names
    for k, (name, args, kwargs) in enumerate(calls):
        if name == "_tensor_unknowns":
            # Random unit normals of the strips (the last argument).
            gen = torch.Generator(device="cpu").manual_seed(100 + k)
            N = torch.randn((args[0].shape[0], 3), generator=gen, dtype=torch.float64)
            N = (N / torch.linalg.norm(N, dim=1, keepdim=True)).to(dtype=args[0].dtype, device=args[0].device)
            calls[k] = (name, tuple(args) + (N,), kwargs)
    tol = _TOL[precision]
    for k, (name, args, kwargs) in enumerate(calls):
        ref_out, ref_g = _grads(model, name, args, kwargs, None, seed=k)
        new_out, new_g = _grads(model, name, args, kwargs, backend, seed=k)
        scale = float(np.max(np.abs(ref_out)))
        assert float(np.max(np.abs(new_out - ref_out))) <= tol * scale, (case_id, name, "output")
        assert len(ref_g) == len(new_g)
        for q, (gr, gn) in enumerate(zip(ref_g, new_g)):
            if gr is None or not np.any(gr):
                # No dependence in the reference: the kernel gives zero (or nothing).
                assert gn is None or float(np.max(np.abs(gn))) == 0.0, (case_id, name, q)
                continue
            assert gn is not None, (case_id, name, q)
            err = float(np.max(np.abs(gn - gr))) / float(np.max(np.abs(gr)))
            assert err <= tol, (case_id, name, q, err)


@pytest.mark.parametrize("precision", ["float64", "float32"])
@pytest.mark.parametrize("backend", ["numpy", "numba", "cython"])
@pytest.mark.parametrize("case_id", _CASE_IDS)
def test_vjp_parity(case_id: str, backend: str, precision: str) -> None:
    """Each CPU VJP backend equals the PyTorch reference VJP (products and all input gradients)."""
    pytest.importorskip("torch")
    if case_id in _FLOAT64_ONLY and precision == "float32":
        pytest.skip("float64 only (see _FLOAT64_ONLY)")
    if backend == "cython":
        pytest.importorskip("ventorum.aero.vortex_vjp_cython")
    _check_parity(case_id, backend, "cpu", precision)


@pytest.mark.gpu
@pytest.mark.parametrize("precision", ["float64", "float32"])
@pytest.mark.parametrize("case_id", _CASE_IDS)
def test_vjp_parity_warp(case_id: str, precision: str) -> None:
    """The Warp VJP kernels equal the PyTorch reference VJP on the GPU (float64 and float32)."""
    _need_cuda()
    if case_id in _FLOAT64_ONLY and precision == "float32":
        pytest.skip("float64 only (see _FLOAT64_ONLY)")
    _check_parity(case_id, "warp", "cuda", precision)


def _synthetic(m: int, n: int, n_u: int, seed: int) -> tuple:
    """Return random horseshoe arrays with masked pairs, zero-length legs and two core groups."""
    rng = np.random.default_rng(seed)
    P = rng.normal(size=(m, 3))
    X = rng.normal(size=(m, 3))
    a = rng.normal(size=(n, 3))
    b = a + 0.5 * rng.normal(size=(n, 3))
    ate = a + np.array([1.0, 0.0, 0.1])
    bte = b + np.array([1.0, 0.0, 0.1])
    ate[:3] = a[:3]
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    rc2s = rng.uniform(1e-3, 1e-2, n)
    tgrc2 = rng.uniform(1e-3, 1e-2, m)
    sgrp = rng.integers(0, 2, n).astype(np.int64)
    tgrp = rng.integers(0, 2, m).astype(np.int64)
    own = np.full(m, -1, dtype=np.int64)
    edge = np.full(m, -1, dtype=np.int64)
    spanel = np.arange(n, dtype=np.int64)
    sea = rng.integers(0, n, n).astype(np.int64)
    seb = rng.integers(0, n, n).astype(np.int64)
    for i in range(5):
        own[i] = i
        P[i] = a[i] + 0.3 * (b[i] - a[i])
    for i in range(5, 10):
        edge[i] = sea[i]
        P[i] = ate[i] + 0.5 * (a[i] - ate[i])
    s = rng.choice([-1.0, 1.0], n)
    col = rng.integers(0, n_u, n).astype(np.int64)
    W = rng.normal(size=(m, n_u))
    return P, X, a, b, ate, bte, d, rc2s, tgrc2, sgrp, tgrp, own, spanel, edge, sea, seb, s, col, W


@pytest.mark.parametrize("use_w", [True, False])
def test_numba_vjp_blocks_give_the_same_result(use_w: bool) -> None:
    """The Numba VJP gives the same derivatives for 1 and 7 row blocks (round-off of the block sums)."""
    from ventorum.aero import vortex_vjp_numba as nb

    P, X, a, b, ate, bte, d, rc2s, tgrc2, sg, tg, own, sp, edge, sea, seb, s, col, W = _synthetic(41, 33, 20, 5)
    Wc = W if use_w else np.zeros((1, 1))
    one = nb.horseshoe_vjp_kernel(P, X, a, b, ate, bte, d, rc2s, tgrc2, sg, tg, own, sp, edge, sea, seb, s, col,
                                  Wc, use_w, True, 1)
    many = nb.horseshoe_vjp_kernel(P, X, a, b, ate, bte, d, rc2s, tgrc2, sg, tg, own, sp, edge, sea, seb, s, col,
                                   Wc, use_w, True, 7)
    for x, y in zip(one, many):
        assert float(np.max(np.abs(x - y))) <= 1e-13 * max(float(np.max(np.abs(x))), 1e-300)


def test_cython_python_threads_equal_openmp(monkeypatch) -> None:
    """The Cython VJP on Python threads (the macOS build) equals the OpenMP path to round-off."""
    cy = pytest.importorskip("ventorum.aero.vortex_vjp_cython")
    P, X, a, b, ate, bte, d, rc2s, tgrc2, sg, tg, own, sp, edge, sea, seb, s, col, W = _synthetic(83, 47, 30, 6)
    args = (P, X, a, b, ate, bte, d, rc2s, tgrc2, sg, tg, own, sp, edge, sea, seb, s, col)
    fwd, vjp, _, _ = K._cython_impl()
    ref_f = fwd(0, *args, 30, True)
    ref_g = vjp(*args, W, True)
    monkeypatch.setattr(cy, "openmp_enabled", lambda: False)
    monkeypatch.setattr("ventorum.utils.parallel.kernel_threads", lambda: 4)
    fwd2, vjp2, _, _ = K._cython_impl()
    assert np.array_equal(fwd2(0, *args, 30, True), ref_f)
    for x, y in zip(vjp2(*args, W, True), ref_g):
        assert float(np.max(np.abs(x - y))) <= 1e-13 * max(float(np.max(np.abs(x))), 1e-300)


def test_trefftz_vjp_backends_agree() -> None:
    """The Trefftz VJP of Numba and Cython equals the numpy reference VJP (float64)."""
    from ventorum.aero import vortex_vjp as ref

    rng = np.random.default_rng(7)
    m, n = 23, 46
    q, qn, p = rng.normal(size=(m, 3)), rng.normal(size=(m, 3)), rng.normal(size=(n, 3))
    gamma = rng.normal(size=n)
    d = np.array([0.9, 0.1, 0.2]) / np.linalg.norm([0.9, 0.1, 0.2])
    rc2, tgrc2 = rng.uniform(1e-3, 1e-2, n), rng.uniform(1e-3, 1e-2, m)
    sg, tg = rng.integers(0, 2, n).astype(np.int64), rng.integers(0, 2, m).astype(np.int64)
    gw = rng.normal(size=m)
    args = (q, qn, p, gamma, d, rc2, tgrc2, sg, tg, True)
    want_f = ref.trefftz_forward(*args)
    want_g = ref.trefftz_vjp(*args, gw)
    backends = ["numba"] + (["cython"] if K.available("cython", "cpu") else [])
    for backend in backends:
        _, _, tfwd, tvjp = K._cpu_impl(backend)
        assert float(np.max(np.abs(tfwd(*args) - want_f))) <= 1e-12 * float(np.max(np.abs(want_f)))
        for x, y in zip(tvjp(*args, gw), want_g):
            assert float(np.max(np.abs(x - y))) <= 1e-12 * float(np.max(np.abs(y))), backend


def test_resolve_rules() -> None:
    """The method keyword and the forced backend select the path of the partials."""
    old = K.get_forced_backend()
    try:
        assert K.resolve("reference", "cpu") == ("reference", "torch")
        assert K.resolve("kernels", "cpu") == ("kernels", K.default_backend("cpu"))
        K.set_vjp_backend("numpy")
        assert K.resolve("auto", "cpu") == ("kernels", "numpy")
        assert K.resolve("kernels", "cpu") == ("kernels", "numpy")
        assert K.resolve("reference", "cpu") == ("reference", "torch")
        K.set_vjp_backend("torch")
        assert K.resolve("auto", "cpu") == ("reference", "torch")
        assert K.resolve("kernels", "cpu") == ("kernels", K.default_backend("cpu"))
        # A forced backend of another device is not used.
        K.set_vjp_backend("warp")
        assert K.resolve("auto", "cpu") == ("kernels", K.default_backend("cpu"))
        K.set_vjp_backend(None)
        with pytest.raises(ValueError, match="method"):
            K.resolve("fast", "cpu")
        with pytest.raises(ValueError, match="VJP backend"):
            K.set_vjp_backend("fortran")
    finally:
        K.set_vjp_backend(old)


def test_resolve_rules_restores_the_forced_backend(monkeypatch) -> None:
    """Regression: test_resolve_rules restores the forced backend also when it fails part way."""
    K.set_vjp_backend(None)
    real = K.resolve
    calls = []

    def failing(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(args)
        if K.get_forced_backend() == "numpy":
            raise AssertionError("failure after the backend is forced")
        return real(*args, **kwargs)

    monkeypatch.setattr(K, "resolve", failing)
    with pytest.raises(AssertionError, match="forced"):
        test_resolve_rules()
    assert len(calls) == 3
    assert K.get_forced_backend() is None


def test_row_blocks_depend_only_on_the_row_count() -> None:
    """The VJP row partition is a function of the row count only (fixed block size, capped count)."""
    from ventorum.aero import vortex_vjp as ref

    assert ref.row_blocks(1) == 1
    assert ref.row_blocks(ref.VJP_BLOCK_ROWS) == 1
    assert ref.row_blocks(ref.VJP_BLOCK_ROWS + 1) == 2
    assert ref.row_blocks(10**7) == ref.VJP_MAX_BLOCKS
    edges = ref.row_block_edges(1000)
    assert edges[0] == 0 and edges[-1] == 1000 and len(edges) == ref.row_blocks(1000) + 1
    assert np.all(np.diff(edges) > 0)


def _vjp_at_threads(backend: str, threads: int, horseshoe: tuple, trefftz: tuple) -> list[np.ndarray]:
    """Return the horseshoe and Trefftz VJP arrays of *backend* with *threads* Numba threads."""
    import numba

    old = numba.get_num_threads()
    numba.set_num_threads(threads)
    try:
        _, vjp, _, tvjp = K._cpu_impl(backend)
        out = list(vjp(*horseshoe, True)) + list(tvjp(*trefftz))
    finally:
        numba.set_num_threads(old)
    return [np.array(x, copy=True) for x in out]


def _determinism_inputs() -> tuple[tuple, tuple]:
    P, X, a, b, ate, bte, d, rc2s, tgrc2, sg, tg, own, sp, edge, sea, seb, s, col, W = _synthetic(301, 97, 40, 11)
    horseshoe = (P, X, a, b, ate, bte, d, rc2s, tgrc2, sg, tg, own, sp, edge, sea, seb, s, col, W)
    rng = np.random.default_rng(12)
    m, n = 263, 71
    q, qn, p = rng.normal(size=(m, 3)), rng.normal(size=(m, 3)), rng.normal(size=(n, 3))
    dd = np.array([0.9, 0.1, 0.2]) / np.linalg.norm([0.9, 0.1, 0.2])
    trefftz = (q, qn, p, rng.normal(size=n), dd, rng.uniform(1e-3, 1e-2, n), rng.uniform(1e-3, 1e-2, m),
               rng.integers(0, 2, n).astype(np.int64), rng.integers(0, 2, m).astype(np.int64), True,
               rng.normal(size=m))
    return horseshoe, trefftz


@pytest.mark.parametrize("backend", ["numba", "cython"])
def test_vjp_bitwise_equal_for_any_thread_count(backend: str) -> None:
    """Regression: the CPU VJP gives bitwise the same arrays with 1 and N kernel threads."""
    import numba

    if not K.available(backend, "cpu"):
        pytest.skip(f"The {backend} VJP backend is not available.")
    n_max = int(numba.config.NUMBA_NUM_THREADS)
    if n_max < 2:
        pytest.skip("Numba has only one thread on this machine.")
    horseshoe, trefftz = _determinism_inputs()
    one = _vjp_at_threads(backend, 1, horseshoe, trefftz)
    for threads in sorted({2, 3, n_max}):
        if threads > n_max:
            continue
        many = _vjp_at_threads(backend, threads, horseshoe, trefftz)
        for x, y in zip(one, many):
            assert np.array_equal(x, y), (backend, threads)


def test_cython_python_threads_bitwise_equal_openmp(monkeypatch) -> None:
    """Regression: the Cython VJP on Python threads gives bitwise the OpenMP result (same partition)."""
    cy = pytest.importorskip("ventorum.aero.vortex_vjp_cython")
    horseshoe, trefftz = _determinism_inputs()
    _, vjp, _, tvjp = K._cython_impl()
    want = list(vjp(*horseshoe, True)) + list(tvjp(*trefftz))
    monkeypatch.setattr(cy, "openmp_enabled", lambda: False)
    for threads in (1, 5):
        monkeypatch.setattr("ventorum.utils.parallel.kernel_threads", lambda t=threads: t)
        _, vjp2, _, tvjp2 = K._cython_impl()
        got = list(vjp2(*horseshoe, True)) + list(tvjp2(*trefftz))
        for x, y in zip(want, got):
            assert np.array_equal(x, y), threads


def _write_profile(monkeypatch, tmp_path, settings: dict) -> None:
    """Write a profile file of this machine (schema of this version) and make it the active profile."""
    import json

    from ventorum.hardware import profile as prof
    from ventorum.hardware.detector import get_machine_fingerprint

    monkeypatch.setenv("VENTORUM_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("VENTORUM_DISABLE_AUTOTUNE", raising=False)
    data = {"schema": prof.SCHEMA_VERSION, "fingerprint": get_machine_fingerprint(), "settings": settings}
    prof.profile_path().write_text(json.dumps(data), encoding="utf-8")
    prof.reset_profile_cache()


@pytest.fixture
def _profile_cache():
    """Forget the profile of the test after it (the next call reads the real file again)."""
    from ventorum.hardware import profile as prof

    yield
    prof.reset_profile_cache()


def test_profile_old_schema_loads(monkeypatch, tmp_path, _profile_cache) -> None:
    """A profile file without the grad_vjp entry (an older profile) loads; the built-in defaults apply.

    A profile file with the entry selects the backend per size class and device.
    """
    from ventorum.hardware.profile import active_profile, tuned_setting

    _write_profile(monkeypatch, tmp_path, {"kernels": {"small": {"tensor": "numba"}}, "single": {}, "batch": {}})
    assert active_profile() is not None
    assert tuned_setting("kernels", "small", "tensor") == "numba"
    assert K.tuned_backend("cpu", 160) is None
    assert K.resolve("auto", "cpu", 160) == ("kernels", K.default_backend("cpu"))
    _write_profile(monkeypatch, tmp_path, {"grad_vjp": {"small": {"cpu": "numpy", "gpu": "torch"},
                                                        "medium": {"cpu": "torch", "gpu": "warp"}}})
    assert K.tuned_backend("cpu", 160) == "numpy"
    assert K.resolve("auto", "cpu", 160) == ("kernels", "numpy")
    assert K.resolve("auto", "cpu", 960) == ("reference", "torch")
    # "kernels" ignores a profile choice of the reference path.
    assert K.resolve("kernels", "cpu", 960) == ("kernels", K.default_backend("cpu"))
    # A backend that is not available falls back to the default.
    _write_profile(monkeypatch, tmp_path, {"grad_vjp": {"small": {"cpu": "warp"}}})
    assert K.resolve("auto", "cpu", 160) == ("kernels", K.default_backend("cpu"))


def test_default_backends() -> None:
    """Without a profile: Numba on the CPU; Warp on the GPU when it is available."""
    assert K.default_backend("cpu") == "numba"
    expected = "warp" if K.available("warp", "cuda") else "torch"
    assert K.default_backend("cuda") == expected


@pytest.mark.slow
def test_quick_tuner_writes_grad_vjp(monkeypatch, tmp_path, _profile_cache) -> None:
    """The quick tuner writes the CPU grad_vjp entry per size class; the gradients then use it.

    A quick run times the CPU device only (it must stay short), so a GPU
    keeps its default backend. The full-mode GPU measurement is checked in
    :func:`test_full_grad_vjp_measures_gpu`.
    """
    pytest.importorskip("torch")
    import json

    from ventorum.hardware import profile as prof
    from ventorum.hardware import tuner

    monkeypatch.setenv("VENTORUM_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("VENTORUM_DISABLE_AUTOTUNE", raising=False)
    monkeypatch.setattr(tuner, "_CASES", {"small": (6, 2), "medium": (8, 2), "large": (10, 2)})
    monkeypatch.setattr(tuner, "_REPEATS", {"small": 1, "medium": 1, "large": 1})
    tuner.tune_machine(quick=True, verbose=False)
    data = json.loads(prof.profile_path().read_text(encoding="utf-8"))
    entry = data["settings"]["grad_vjp"]
    assert set(entry) == {"small", "medium"}
    for cls in ("small", "medium"):
        assert set(entry[cls]) == {"cpu"}
        assert entry[cls]["cpu"] in K.CPU_BACKENDS + ("torch",)
        assert set(data["measurements"]["grad_vjp"][cls]["cpu"]) >= {"numba"}
    prof.reset_profile_cache()
    assert K.tuned_backend("cpu", 10) == entry["small"]["cpu"]
    assert K.tuned_backend("cuda", 10) is None


@pytest.mark.slow
@pytest.mark.gpu
def test_full_grad_vjp_measures_gpu(monkeypatch) -> None:
    """The full-mode gradient step times the GPU backends and chooses Warp or PyTorch."""
    pytest.importorskip("torch")
    from ventorum import gpu
    from ventorum.hardware import tuner

    if not (K.available("warp", "cuda") and gpu.available()):
        pytest.skip("needs the Warp CUDA backend")
    monkeypatch.setattr(tuner, "_REPEATS", {"small": 1, "medium": 1, "large": 1})
    ac, st = tuner._wing(6, 2)
    choice, times = tuner._measure_grad_vjp("small", ac, st, 2, lambda _msg: None, quick=False)
    assert set(choice) == {"cpu", "gpu"}
    assert choice["gpu"] in ("warp", "torch")
    assert set(times["gpu"]) >= {"warp"}


def _collinear_targets(A: np.ndarray, B: np.ndarray, seed: int) -> np.ndarray:
    """Return targets on and near the extended line A -> B outside the filament.

    The targets before A and after B lie on the line (exactly collinear in
    binary floating point when A and B have dyadic coordinates) and at small
    offsets from it, plus some general targets.
    """
    rng = np.random.default_rng(seed)
    pts = []
    for t in (-2.0, -0.5, -0.125, 1.125, 1.5, 3.0):
        base = A + t * (B - A)
        pts.append(base)
        for off in (1e-9, 1e-6, 1e-3, 1e-1):
            pts.append(base + off * rng.normal(size=3))
    pts.extend(rng.normal(size=(8, 3)))
    return np.array(pts)


@pytest.mark.parametrize("kind", ["segment", "ray"])
def test_filament_vjp_parity_on_extended_lines(kind: str) -> None:
    """The numpy and Numba filament VJP equal torch autograd of the reference near and on the extended lines.

    Targets lie before and after the segments (behind the rays), exactly
    collinear and at offsets from 1e-9 m to 0.1 m. There the core uses the
    distance to the nearest end point, and the derivative of the velocity
    at a collinear point is not zero. Float64, 1e-12 relative to the
    largest component of each gradient.
    """
    torch = pytest.importorskip("torch")
    from ventorum.aero import vortex_vjp as ref
    from ventorum.aero import vortex_vjp_numba as nb
    from ventorum.sensitivity.reference import ray_vel, seg_vel

    A = np.array([[0.0, 0.0, 0.0], [0.5, -0.25, 0.125], [1.0, 2.0, -0.5]])
    if kind == "segment":
        E = np.array([[1.0, 0.5, 0.25], [0.5, 0.75, 0.125], [1.0, 2.0, -0.5]])  # last: zero length
    else:
        E = np.array([[1.0, 0.0, 0.0], [0.6, 0.0, -0.8], [0.0, 0.0, 1.0]])
    lines = [(A[j], A[j] + E[j]) if kind == "ray" else (A[j], E[j]) for j in range(2)]
    P = np.vstack([_collinear_targets(a, b, 11 + j) for j, (a, b) in enumerate(lines)])
    m, n = P.shape[0], A.shape[0]
    rng = np.random.default_rng(3)
    rc2 = rng.uniform(1e-3, 0.3, (m, n))
    u = rng.normal(size=(m, n, 3))

    tP, tA, tE, tr = (torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in (P, A, E, rc2))
    fn = seg_vel if kind == "segment" else ray_vel
    v_t = fn(tP, tA, tE, tr)
    g_t = torch.autograd.grad((v_t * torch.tensor(u)).sum(), [tP, tA, tE, tr])
    v_t = v_t.detach().numpy()
    g_t = [g.numpy() for g in g_t]

    tol = _TOL["float64"]

    def check(v, grads, name):
        assert float(np.max(np.abs(v - v_t))) <= tol * float(np.max(np.abs(v_t))), (name, "velocity")
        for q, (gn, gr) in enumerate(zip(grads, g_t)):
            err = float(np.max(np.abs(gn - gr))) / float(np.max(np.abs(gr)))
            assert err <= tol, (name, q, err)

    # numpy reference VJP.
    if kind == "segment":
        v, st = ref._seg_forward(P[:, None, :], A[None], E[None], rc2, ref.FLOORS["float64"])
        gP, gA, gE, gr = ref._seg_backward(u, st)
    else:
        v, st = ref._ray_forward(P[:, None, :], A[None], E[None], rc2, ref.FLOORS["float64"])
        gP, gA, gE, gr = ref._ray_backward(u, st)
    check(v, [gP.sum(axis=1), gA.sum(axis=0), gE.sum(axis=0), gr], "numpy")

    # Numba VJP (one pair per call).
    v = np.zeros((m, n, 3))
    gP = np.zeros((m, 3))
    gA = np.zeros((n, 3))
    gE = np.zeros((n, 3))
    gr = np.zeros((m, n))
    v2 = np.zeros((m, n, 3))
    vel = nb._seg_v if kind == "segment" else nb._ray_v
    for i in range(m):
        for j in range(n):
            if kind == "segment":
                o = nb._seg_fb(*P[i], *A[j], *E[j], rc2[i, j], *u[i, j])
                gA[j] += o[6:9]
                gE[j] += o[9:12]
                gr[i, j] = o[12]
            else:
                o = nb._ray_fb(*P[i], *A[j], *E[j], rc2[i, j], *u[i, j])
                gA[j] -= o[3:6]
                gE[j] += o[6:9]
                gr[i, j] = o[9]
            v[i, j] = o[:3]
            v2[i, j] = vel(*P[i], *A[j], *E[j], rc2[i, j])
            gP[i] += o[3:6]
    check(v, [gP, gA, gE, gr], "numba")
    assert float(np.max(np.abs(v2 - v_t))) <= tol * float(np.max(np.abs(v_t))), "numba forward"


def _warp_line_case(seed: int) -> tuple:
    """Return horseshoes and targets on and near the extended lines of their legs (outside the legs)."""
    rng = np.random.default_rng(seed)
    a = np.array([[0.25, -0.5, 0.0], [0.5, 0.625, 0.0625]])
    b = np.array([[0.25, 0.5, 0.0], [0.75, 1.625, 0.125]])
    ate = a + np.array([1.0, 0.0, 0.0])
    bte = b + np.array([1.0, 0.0, 0.125])
    d = np.array([[1.0, 0.0, 0.0], [0.6, 0.0, 0.8]])
    pts = []
    for s in range(2):
        for p0, p1 in ((a[s], b[s]), (ate[s], a[s]), (b[s], bte[s])):
            for t in (-0.5, -0.125, 1.125, 1.5):
                pts.append(p0 + t * (p1 - p0))
        for p0 in (ate[s], bte[s]):
            for t in (0.125, 1.0):
                pts.append(p0 - t * d[s])
    pts = np.array(pts)
    off = np.array([1e-6, 1e-3, 1e-1])[rng.integers(0, 3, len(pts))][:, None]
    P = np.vstack([pts, pts + off * rng.normal(size=pts.shape)])
    return P, a, b, ate, bte, d


@pytest.mark.gpu
@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_warp_vjp_parity_on_extended_leg_lines(precision: str) -> None:
    """The Warp horseshoe VJP equals torch autograd of the reference on and near the extended leg lines.

    The targets lie before and after the finite legs and behind the start
    points of the wake legs, exactly collinear and at offsets from 1e-6 m to
    0.1 m. There the core uses the distance to the nearest end point, and
    the derivative at a collinear point is not zero. The tolerances are
    those of the other parity tests of this file.
    """
    torch = _need_cuda()
    from types import SimpleNamespace

    from ventorum.gpu import vjp_kernels as wk
    from ventorum.sensitivity.reference import horseshoe_vel

    P, a, b, ate, bte, d = _warp_line_case(21)
    m, n = P.shape[0], a.shape[0]
    rng = np.random.default_rng(4)
    X = rng.normal(size=(m, 3))
    rc2 = np.array([4e-4, 2.5e-3])
    s = np.array([1.0, -1.0])
    dt = torch.float64 if precision == "float64" else torch.float32
    dev = "cuda"
    ins = [torch.tensor(x, dtype=dt, device=dev, requires_grad=True) for x in (P, X, a, b, ate, bte, d, rc2, s)]
    tP, tX, ta, tb, tat, tbt, td, tr, ts = ins
    v = horseshoe_vel(tP, ta, tb, tat, tbt, td, tr)
    loss = ((v * tX[:, None, :]).sum(-1) * ts[None, :]).sum()
    want = [g.detach().double().cpu().numpy() for g in torch.autograd.grad(loss, ins)]

    i64 = torch.int64
    topo = SimpleNamespace(src_group=torch.zeros(n, dtype=i64), tg_group=torch.zeros(m, dtype=i64),
                           own=torch.full((m,), -1, dtype=i64), src_panel=torch.arange(n, dtype=i64),
                           edge=torch.full((m,), -1, dtype=i64), src_edge_a=torch.arange(n, dtype=i64),
                           src_edge_b=torch.arange(n, dtype=i64), col=torch.arange(n, dtype=i64),
                           n_u=n, use_tg=False)
    det = [x.detach() for x in ins]
    out = wk.horseshoe_vjp(det[0], det[1], det[2], det[3], det[4], det[5], det[6], det[7], None, det[8], None, topo)
    gP, gX, _, ga, gb, gat, gbt, gd, gr, gs = (None if x is None else x.double().cpu().numpy() for x in out)
    tol = _TOL[precision]
    for q, (gn, gw) in enumerate(zip((gP, gX, ga, gb, gat, gbt, gd, gr, gs), want)):
        assert np.all(np.isfinite(gn)), q
        err = float(np.max(np.abs(gn - gw))) / float(np.max(np.abs(gw)))
        assert err <= tol, (precision, q, err)
