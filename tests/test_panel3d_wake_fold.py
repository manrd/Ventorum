"""Parity tests for wake strip reductions, folds and full-mesh reuse."""

from dataclasses import fields

import numpy as np
import pytest

from ventorum.core.datatypes import FlightCondition, LiftingSurface, WingSection
from ventorum.panel3d import gpu
from ventorum.panel3d.kutta import strip_influence_matrix
from ventorum.panel3d.mesh import mesh_lifting_surface
from ventorum.panel3d.solver import PanelResult, PanelSolver, _fold_wake
from ventorum.panel3d.wake import PanelWake


def _need_cuda():
    torch = pytest.importorskip("torch")
    pytest.importorskip("warp")
    if not torch.cuda.is_available():
        pytest.skip("A CUDA GPU is not available.")
    return torch


def _strip_loop(cw, strips):
    c = np.asarray(cw, dtype=float)
    out = np.zeros((c.shape[0], len(strips)), dtype=float)
    for s, idx in enumerate(strips):
        cols = np.asarray(idx, dtype=int)
        if cols.size:
            out[:, s] = np.sum(c[:, cols], axis=1)
    return out


def _gpu_strip_loop(cw, strips):
    import torch

    out = torch.zeros((cw.shape[0], len(strips)), dtype=cw.dtype, device=cw.device)
    for s, cols in enumerate(strips):
        idx = torch.as_tensor(np.asarray(cols, dtype=np.int64), device=cw.device)
        if idx.numel():
            out[:, s] = cw.index_select(1, idx).sum(dim=1)
    return out


def _fold_loop(mat_a, d_strip, wake):
    for s in range(len(wake.strip_panel_indices)):
        cs = d_strip[:, s]
        mat_a[:, int(wake.wake_up[s])] += cs
        mat_a[:, int(wake.wake_lo[s])] -= cs
    return mat_a


def _roundoff(actual, expected, tol=1e-12):
    scale = max(float(np.max(np.abs(expected), initial=0.0)), 1.0)
    assert np.max(np.abs(actual - expected), initial=0.0) <= tol * scale


_DEVICES = ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)]


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_strip_sum_bits_one_panel_per_strip(device, dtype):
    rng = np.random.default_rng(27)
    cw = rng.normal(size=(13, 11)).astype(dtype)
    strips = [[int(i)] for i in rng.permutation(11)]
    if device == "cpu":
        assert np.array_equal(strip_influence_matrix(cw, strips), _strip_loop(cw, strips))
    else:
        torch = _need_cuda()
        tensor = torch.as_tensor(cw, device="cuda")
        actual = gpu.strip_sum(tensor, strips)
        expected = _gpu_strip_loop(tensor, strips)
        assert np.array_equal(actual.cpu().numpy(), expected.cpu().numpy())


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_gpu_strip_sum_is_deterministic(dtype):
    """Repeated GPU strip sums give the same bits.

    The strip sum must not use atomic additions (index_add_ on the GPU), whose
    order can change the float result between runs. Atomic-order effects do
    not show in every run, so this test is a smoke check; the implementation
    is deterministic by construction (gather, reshape and sum).
    """
    torch = _need_cuda()
    rng = np.random.default_rng(29)
    cw = torch.as_tensor(rng.random(size=(500, 4000)).astype(dtype), device="cuda")
    strips = [list(range(40 * s, 40 * s + 40)) for s in range(100)]
    first = gpu.strip_sum(cw, strips)
    for _ in range(60):
        assert torch.equal(gpu.strip_sum(cw, strips), first)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_strip_sum_roundoff_many_panels_per_strip(device, dtype):
    rng = np.random.default_rng(28)
    cw = rng.normal(size=(17, 36)).astype(dtype)
    strips = rng.permutation(36).reshape(9, 4).tolist()
    if device == "cpu":
        _roundoff(strip_influence_matrix(cw, strips), _strip_loop(cw, strips))
    else:
        torch = _need_cuda()
        tensor = torch.as_tensor(cw, device="cuda")
        actual = gpu.strip_sum(tensor, strips).cpu().numpy()
        expected = _gpu_strip_loop(tensor, strips).cpu().numpy()
        _roundoff(actual, expected, 1e-5 if dtype == np.float32 else 1e-12)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("strips", [[], [[], [], []], [[3, 1], [], [0], [1, 2, 1], []]])
def test_strip_sum_empty_and_ragged_strips(device, strips):
    cw = np.random.default_rng(29).normal(size=(7, 4))
    if device == "cpu":
        _roundoff(strip_influence_matrix(cw, strips), _strip_loop(cw, strips))
    else:
        torch = _need_cuda()
        tensor = torch.as_tensor(cw, device="cuda")
        _roundoff(gpu.strip_sum(tensor, strips).cpu().numpy(), _strip_loop(cw, strips))


def _synthetic_wake(repeated):
    up = np.array([0, 0, 1, 2]) if repeated else np.arange(4)
    lo = np.array([1, 2, 3, 0]) if repeated else np.arange(4, 8)
    return PanelWake(
        panels=np.zeros((4, 4, 3)), wake_up=up, wake_lo=lo,
        strip_panel_indices=[[s] for s in range(4)],
        p_left=np.zeros((4, 3)), p_right=np.zeros((4, 3)),
        wake_edges=np.zeros((4, 4), dtype=int),
    )


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("repeated", [False, True])
def test_fold_repeated_panels_matches_loop(device, dtype, repeated):
    rng = np.random.default_rng(30)
    wake = _synthetic_wake(repeated)
    mat = rng.normal(size=(8, 8)).astype(dtype)
    strips = rng.normal(size=(8, 4)).astype(dtype)
    if device == "cpu":
        actual = mat.copy()
        expected = _fold_loop(mat.copy(), strips, wake)
        assert _fold_wake(actual, strips, wake) is actual
    else:
        torch = _need_cuda()
        actual_t = torch.as_tensor(mat, device="cuda")
        strips_t = torch.as_tensor(strips, device="cuda")
        expected = _fold_loop(actual_t.clone(), strips_t, wake).cpu().numpy()
        assert _fold_wake(actual_t, strips_t, wake) is actual_t
        actual = actual_t.cpu().numpy()
    if repeated:
        _roundoff(actual, expected, 1e-5 if dtype == np.float32 else 1e-12)
    else:
        assert np.array_equal(actual, expected)


@pytest.mark.gpu
@pytest.mark.parametrize("sign", [1.0, -1.0, 0.375])
@pytest.mark.parametrize("n_wake", [1, 4])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_add_wake_influence_matches_loop(sign, n_wake, dtype):
    torch = _need_cuda()
    rng = np.random.default_rng(31)
    for repeated in (False, True):
        wake = _synthetic_wake(repeated)
        strips = np.arange(4 * n_wake).reshape(4, n_wake).tolist()
        cw = torch.as_tensor(rng.normal(size=(8, 4 * n_wake)).astype(dtype), device="cuda")
        mat = torch.as_tensor(rng.normal(size=(8, 8)).astype(dtype), device="cuda")
        expected = _fold_loop(mat.clone(), _gpu_strip_loop(cw, strips) * sign, wake)
        actual = mat.clone()
        assert gpu.add_wake_influence(
            actual, cw, wake.wake_up, wake.wake_lo, strips, sign
        ) is actual
        if n_wake == 1 and not repeated:
            assert np.array_equal(actual.cpu().numpy(), expected.cpu().numpy())
        else:
            _roundoff(actual.cpu().numpy(), expected.cpu().numpy(),
                      1e-5 if dtype == np.float32 else 1e-12)
    empty = mat.clone()
    assert gpu.add_wake_influence(empty, cw[:, :0], [], [], []) is empty
    assert np.array_equal(empty.cpu().numpy(), mat.cpu().numpy())


def _half_mesh():
    surf = LiftingSurface(
        name="Wing", semi_span=3.0, is_symmetric=True,
        sections=[WingSection(y_frac=0.0, chord=1.0), WingSection(y_frac=1.0, chord=0.8)],
    )
    x = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, 51)))
    yt = 0.6 * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2
                + 0.2843 * x**3 - 0.1036 * x**4)
    coords = np.vstack([np.column_stack([x, yt])[::-1], np.column_stack([x, -yt])[1:]])
    return mesh_lifting_surface(surf, coords, n_chord=6, n_span=5, half=True)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("kutta", ["morino", "pressure"])
def test_full_mesh_sub_solver_reuses_parent_pieces(device, kutta, monkeypatch):
    options = {"backend": "numpy"}
    if device == "gpu":
        torch = _need_cuda()
        options = {"backend": "gpu", "dtype": torch.float64}
    parent = PanelSolver(_half_mesh(), far_ratio=np.inf, kutta=kutta, **options)
    old = PanelSolver(
        parent._full_mesh, wake_length=parent.wake_length, n_wake=parent.n_wake,
        far_ratio=parent.far_ratio, S_ref=parent.S_ref, b_ref=parent.b_ref,
        c_ref=parent.c_ref, ref_point=parent.ref_point, kutta=kutta, **options,
    )
    # No full-mesh preparation may run while the child is constructed.
    import ventorum.panel3d.solver as solver_module

    def forbidden(*args, **kwargs):
        pytest.fail("The sub-solver rebuilt a full-mesh piece.")

    with monkeypatch.context() as patch:
        for name in ("_standardize_panels", "_precompute_geometry", "build_surface_gradient_operator", "build_wake"):
            patch.setattr(solver_module, name, forbidden)
        sub = parent._full_mesh_solver()
    assert parent._full_mesh_solver() is sub
    assert sub._grad_op is parent._grad_op
    assert sub._full_geom is parent._full_geom
    assert sub._full_wake is parent._full_wake
    assert sub.wake is parent._full_wake
    assert sub.standardized_panels is parent._standardized_full_panels
    assert sub._standardized_full_panels is parent._standardized_full_panels
    assert sub._configs is not parent._configs
    assert sub._body_parts is None
    assert sub.factorization_count == parent.factorization_count == 0
    condition = FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=np.radians(3.0), h=0.8)
    result = parent.solve(condition)
    expected = parent._to_half(old.solve(condition))
    for item in fields(PanelResult):
        assert np.array_equal(getattr(result, item.name), getattr(expected, item.name)), item.name
    assert sub._own_factorizations == 1
    assert parent._own_factorizations == 0
    assert sub.factorization_count == parent.factorization_count == 1
    assert parent._configs == {}
    assert len(sub._configs) == 1


@pytest.mark.gpu
def test_gpu_fold_matches_cpu(monkeypatch):
    torch = _need_cuda()
    mesh = _half_mesh()
    condition = FlightCondition(V_inf=30.0, alpha=np.radians(4.0), h=0.8)
    systems = []
    for backend in ("numpy", "gpu"):
        solver = PanelSolver(mesh, backend=backend, dtype=torch.float64, far_ratio=np.inf)
        captured = []
        factorize = solver._factorize

        def capture(mat_a, factorize=factorize, backend=backend, captured=captured):
            captured.append(mat_a.cpu().numpy().copy() if backend == "gpu" else mat_a.copy())
            return factorize(mat_a)

        monkeypatch.setattr(solver, "_factorize", capture)
        solver.solve(condition)
        assert len(captured) == 1
        systems.append(captured[0])
    _roundoff(systems[1], systems[0])
