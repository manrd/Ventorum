# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests of the fused assembly of the 3D panel parity systems.

A half mesh with symmetry about y = 0 solves ``C_self + sign * C_mirror``
and the ground images add ``C_image + sign * C_mirror_image``. The pair
kernels return these sums and the source right-hand sides
``(B_a + sign * B_b) @ sigma`` without the separate matrices or the source
matrix. The tests check:

* each pair kernel of every backend against the numpy reference built from
  :func:`ventorum.panel3d.kernels.influence_matrices` (near-field and
  far-field pairs, targets on the panel planes, both signs);
* the solver systems and unit solutions against a copy of the old
  assembly (four separate matrices, sums after the wake fold), in free
  air, in ground effect and with sideslip and rates;
* the dense arrays that a half-mesh free-air solve keeps;
* the device memory estimate of the fused assembly.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg

from ventorum.core.datatypes import FlightCondition, LiftingSurface, WingSection
from ventorum.panel3d import kernels as ref
from ventorum.panel3d import kernels_numba as nb
from ventorum.panel3d.images import mirror_panels_y, reflect_panels_plane
from ventorum.panel3d.mesh import mesh_lifting_surface
from ventorum.panel3d.solver import _ASYM_COLS, _SYM_COLS, PanelSolver, _fold_wake

_GROUND = (np.array([0.0, 0.0, 1.0]), -0.7)


def _need_cuda():
    """Skip the test when a CUDA GPU, PyTorch or Warp is missing."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("warp")
    if not torch.cuda.is_available():
        pytest.skip("A CUDA GPU is not available.")
    return torch


def _assert_roundoff(new, expected, tol: float, name: str, floor_scale: float = 0.0) -> None:
    """Assert max |new - expected| <= tol * max(max |expected|, floor_scale) or 1e-14.

    *floor_scale* is the smallest scale of a dimensionless array (1 for
    the doublet potentials). The absolute floor 1e-14 covers arrays whose
    values are all near zero.
    """
    new = np.asarray(new, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    assert new.shape == expected.shape, f"{name}: shape {new.shape} != {expected.shape}"
    err = float(np.max(np.abs(new - expected), initial=0.0))
    scale = max(float(np.max(np.abs(expected), initial=0.0)), floor_scale)
    assert err <= tol * scale or err <= 1e-14, f"{name}: error {err:.3e} above {tol:.1e} x {scale:.3e}"


# ----------------------------------------------------------------------
# Kernel inputs
# ----------------------------------------------------------------------


def _panel_set(seed: int = 5) -> np.ndarray:
    """Return quads, a non-planar quad and triangles near y = 0 in m.

    The panels lie close to the plane of symmetry, so many targets are in
    the near field of both a panel and its mirror.
    """
    rng = np.random.default_rng(seed)
    panels = []
    for _ in range(5):
        base = rng.uniform(-0.6, 0.6, size=(4, 3))
        base[:, 1] = np.abs(base[:, 1]) + 0.05
        panels.append(base)
    quad = np.array([[0.0, 0.1, 0.0], [1.0, 0.1, 0.1], [1.0, 0.9, -0.1], [0.0, 0.9, 0.0]])
    panels.append(quad)
    for _ in range(2):
        tri = rng.uniform(-0.6, 0.6, size=(3, 3))
        tri[:, 1] = np.abs(tri[:, 1]) + 0.05
        panels.append(np.vstack([tri, tri[2]]))
    return np.ascontiguousarray(np.array(panels, dtype=np.float64))


def _pair_sets(kind: str):
    """Return (panels_a, panels_b or None) of one pair kind."""
    a = _panel_set()
    if kind == "single":
        return a, None
    if kind == "mirror":
        return a, mirror_panels_y(a)
    return a, reflect_panels_plane(a, *_GROUND)


def _targets(panels: np.ndarray, seed: int = 9) -> np.ndarray:
    """Return near-field, on-plane, far-field and collocation targets in m."""
    rng = np.random.default_rng(seed)
    geom = ref._precompute_geometry(panels)
    pts = [geom.centroid]
    for scale in (0.05, 0.3, 1.0, 8.0, 40.0):
        direction = rng.normal(size=(len(panels), 3))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        pts.append(geom.centroid + scale * geom.diameter[:, None] * direction)
    # On the panel plane, inside and outside the panel.
    pts.append(geom.centroid + 0.2 * geom.diameter[:, None] * geom.l_axis)
    pts.append(geom.centroid + 2.0 * geom.diameter[:, None] * geom.m_axis)
    # On the plane of symmetry and on the ground plane.
    on_sym = geom.centroid.copy()
    on_sym[:, 1] = 0.0
    pts.append(on_sym)
    return np.ascontiguousarray(np.vstack(pts))


def _sigma(n_panels: int, n_cols: int = 6, seed: int = 3) -> np.ndarray:
    """Return source strengths with shape (M, K) in m/s."""
    return np.random.default_rng(seed).normal(size=(n_panels, n_cols))


def _reference(points, panels_a, panels_b, sign, sigma, far_ratio):
    """Return the pair results from the separate numpy matrices."""
    b_a, c_a = ref.influence_matrices(points, panels_a, far_ratio=far_ratio)
    if panels_b is None:
        return c_a, b_a @ sigma
    b_b, c_b = ref.influence_matrices(points, panels_b, far_ratio=far_ratio)
    return c_a + sign * c_b, (b_a + sign * b_b) @ sigma


_KINDS = ["single", "mirror", "image"]


@pytest.mark.parametrize("far_ratio", [5.0, np.inf])
@pytest.mark.parametrize("sign", [1.0, -1.0])
@pytest.mark.parametrize("kind", _KINDS)
def test_numpy_pair_kernels_match_separate_matrices(kind, sign, far_ratio):
    """The numpy pair kernels equal the sums of the separate reference matrices."""
    panels_a, panels_b = _pair_sets(kind)
    points = _targets(panels_a)
    sigma = _sigma(len(panels_a))
    exp_c, exp_r = _reference(points, panels_a, panels_b, sign, sigma, far_ratio)
    got_c = ref.doublet_pair_matrix(points, panels_a, panels_b, sign, far_ratio=far_ratio)
    # The same doublet blocks and the same addition: bit-identical.
    assert np.array_equal(got_c, exp_c)
    got_r = ref.source_pair_rhs(points, panels_a, panels_b, sign, sigma, far_ratio=far_ratio)
    _assert_roundoff(got_r, exp_r, 1e-12, "numpy source_pair_rhs")
    both_c, both_r = ref.pair_system(points, panels_a, panels_b, sign, sigma, far_ratio=far_ratio)
    assert np.array_equal(both_c, got_c)
    assert np.array_equal(both_r, got_r)
    if panels_b is None:
        single = ref.source_rhs(points, panels_a, sigma, far_ratio=far_ratio)
        assert np.array_equal(single, got_r)


def test_numpy_pair_kernels_chunk_and_shapes(monkeypatch):
    """Small target chunks, a one-dimensional sigma and the input checks."""
    panels_a, panels_b = _pair_sets("mirror")
    points = _targets(panels_a)
    sigma = _sigma(len(panels_a), 3)
    whole_c, whole_r = ref.pair_system(points, panels_a, panels_b, -1.0, sigma)
    monkeypatch.setattr(ref, "_CHUNK_BUDGET", 3 * len(panels_a))
    part_c, part_r = ref.pair_system(points, panels_a, panels_b, -1.0, sigma)
    assert np.array_equal(part_c, whole_c)
    _assert_roundoff(part_r, whole_r, 1e-13, "chunked rhs")
    flat = ref.source_pair_rhs(points, panels_a, panels_b, -1.0, sigma[:, 1])
    assert flat.shape == (len(points),)
    _assert_roundoff(flat, whole_r[:, 1], 1e-13, "one-dimensional sigma")
    with pytest.raises(ValueError, match="same shape"):
        ref.doublet_pair_matrix(points, panels_a, panels_b[:-1])
    with pytest.raises(ValueError, match="sigma"):
        ref.source_rhs(points, panels_a, sigma[:-1])
    with pytest.raises(ValueError, match="at most 6"):
        ref.source_rhs(points, panels_a, _sigma(len(panels_a), 7))
    empty_c, empty_r = ref.pair_system(points[:0], panels_a, panels_b, 1.0, sigma)
    assert empty_c.shape == (0, len(panels_a)) and empty_r.shape == (0, 3)


@pytest.mark.parametrize("far_ratio", [5.0, np.inf])
@pytest.mark.parametrize("sign", [1.0, -1.0])
@pytest.mark.parametrize("kind", _KINDS)
def test_numba_pair_kernels_match_numpy(kind, sign, far_ratio):
    """The Numba pair kernels equal the numpy reference to round-off."""
    panels_a, panels_b = _pair_sets(kind)
    points = _targets(panels_a)
    sigma = _sigma(len(panels_a))
    exp_c, exp_r = _reference(points, panels_a, panels_b, sign, sigma, far_ratio)
    got_c = nb.doublet_pair_matrix_numba(points, panels_a, panels_b, sign, far_ratio=far_ratio)
    _assert_roundoff(got_c, exp_c, 1e-12, "numba doublet_pair_matrix", floor_scale=1.0)
    got_r = nb.source_pair_rhs_numba(points, panels_a, panels_b, sign, sigma, far_ratio=far_ratio)
    _assert_roundoff(got_r, exp_r, 1e-12, "numba source_pair_rhs")
    both_c, both_r = nb.pair_system_numba(points, panels_a, panels_b, sign, sigma, far_ratio=far_ratio)
    assert np.array_equal(both_c, got_c)
    assert np.array_equal(both_r, got_r)
    if panels_b is None:
        assert np.array_equal(nb.source_rhs_numba(points, panels_a, sigma, far_ratio=far_ratio), got_r)


def test_numba_pair_doublet_equals_influence_doublet():
    """The doublet-only Numba pair kernel gives the same bits as the C of the B, C kernel."""
    panels_a, _ = _pair_sets("single")
    points = _targets(panels_a)
    _, c_old = nb.influence_matrices_numba(points, panels_a)
    assert np.array_equal(nb.doublet_pair_matrix_numba(points, panels_a, None), c_old)


def test_numba_pair_kernels_shapes(unsafe_float32_kernels):
    """A one-dimensional sigma, empty targets and the float32 output dtype."""
    panels_a, panels_b = _pair_sets("mirror")
    points = _targets(panels_a)
    sigma = _sigma(len(panels_a), 2)
    full = nb.source_pair_rhs_numba(points, panels_a, panels_b, 1.0, sigma)
    flat = nb.source_pair_rhs_numba(points, panels_a, panels_b, 1.0, sigma[:, 0])
    assert flat.shape == (len(points),)
    assert np.array_equal(flat, full[:, 0])
    empty_c, empty_r = nb.pair_system_numba(points[:0], panels_a, panels_b, 1.0, sigma)
    assert empty_c.shape == (0, len(panels_a)) and empty_r.shape == (0, 2)
    c32 = nb.doublet_pair_matrix_numba(points.astype(np.float32), panels_a.astype(np.float32), None)
    assert c32.dtype == np.float32


@pytest.mark.gpu
@pytest.mark.parametrize("far_ratio", [5.0, np.inf])
@pytest.mark.parametrize("sign", [1.0, -1.0])
@pytest.mark.parametrize("kind", _KINDS)
def test_warp_pair_kernels_match_numpy(kind, sign, far_ratio):
    """The Warp float64 pair kernels equal the numpy reference to round-off."""
    torch = _need_cuda()
    from ventorum.panel3d import kernels_warp as wpk

    panels_a, panels_b = _pair_sets(kind)
    points = _targets(panels_a)
    sigma = _sigma(len(panels_a))
    exp_c, exp_r = _reference(points, panels_a, panels_b, sign, sigma, far_ratio)
    got_c = wpk.doublet_pair_matrix_warp(points, panels_a, panels_b, sign, far_ratio=far_ratio,
                                         dtype=torch.float64)
    assert got_c.is_cuda and got_c.dtype == torch.float64
    _assert_roundoff(got_c.cpu().numpy(), exp_c, 1e-12, "warp doublet_pair_matrix", floor_scale=1.0)
    got_r = wpk.source_pair_rhs_warp(points, panels_a, panels_b, sign, sigma, far_ratio=far_ratio,
                                     dtype=torch.float64)
    _assert_roundoff(got_r.cpu().numpy(), exp_r, 1e-12, "warp source_pair_rhs")
    both_c, both_r = wpk.pair_system_warp(points, panels_a, panels_b, sign, sigma, far_ratio=far_ratio,
                                          dtype=torch.float64)
    _assert_roundoff(both_c.cpu().numpy(), exp_c, 1e-12, "warp pair_system C", floor_scale=1.0)
    _assert_roundoff(both_r.cpu().numpy(), exp_r, 1e-12, "warp pair_system R")
    if panels_b is None:
        single = wpk.source_rhs_warp(points, panels_a, sigma, far_ratio=far_ratio, dtype=torch.float64)
        _assert_roundoff(single.cpu().numpy(), exp_r, 1e-12, "warp source_rhs")


@pytest.mark.gpu
def test_warp_pair_blocks_round_float64_to_float32():
    """Float64 kernels with a float32 output round each block of rows once.

    Several blocks of rows (block_rows=7) must give the float32 rounding of
    the float64 result, and the source sums keep float64.
    """
    torch = _need_cuda()
    from ventorum.panel3d import kernels_warp as wpk

    panels_a, panels_b = _pair_sets("mirror")
    points = _targets(panels_a)
    sigma = _sigma(len(panels_a), 3)
    full = wpk.doublet_pair_matrix_warp(points, panels_a, panels_b, -1.0, dtype=torch.float64)
    blocks, rhs = wpk.pair_system_warp(points, panels_a, panels_b, -1.0, sigma, dtype=torch.float64,
                                       out_dtype=torch.float32, block_rows=7)
    assert blocks.dtype == torch.float32 and rhs.dtype == torch.float64
    assert torch.equal(blocks, full.to(torch.float32))
    exp_c, exp_r = _reference(points, panels_a, panels_b, -1.0, sigma, 5.0)
    _assert_roundoff(rhs.cpu().numpy(), exp_r, 1e-12, "warp float64 rhs")
    one_block = wpk.doublet_pair_matrix_warp(points, panels_a, panels_b, -1.0, dtype=torch.float64,
                                             out_dtype=torch.float32)
    assert torch.equal(one_block, blocks)
    flat = wpk.source_pair_rhs_warp(points, panels_a, panels_b, -1.0, sigma[:, 2], dtype=torch.float64)
    assert flat.shape == (len(points),)
    _assert_roundoff(flat.cpu().numpy(), exp_r[:, 2], 1e-12, "warp one-dimensional sigma")
    empty_c, empty_r = wpk.pair_system_warp(points[:0], panels_a, panels_b, 1.0, sigma, dtype=torch.float64)
    assert tuple(empty_c.shape) == (0, len(panels_a)) and tuple(empty_r.shape) == (0, 3)


@pytest.mark.gpu
def test_warp_source_pair_rhs_is_deterministic():
    """Repeated source sums on the GPU give the same bits (no atomic additions)."""
    torch = _need_cuda()
    from ventorum.panel3d import kernels_warp as wpk

    panels_a, panels_b = _pair_sets("image")
    points = _targets(panels_a)
    sigma = _sigma(len(panels_a))
    first = wpk.source_pair_rhs_warp(points, panels_a, panels_b, 1.0, sigma, dtype=torch.float64)
    for _ in range(5):
        again = wpk.source_pair_rhs_warp(points, panels_a, panels_b, 1.0, sigma, dtype=torch.float64)
        assert torch.equal(again, first)


# ----------------------------------------------------------------------
# Device wrappers
# ----------------------------------------------------------------------


@pytest.mark.gpu
def test_device_pair_wrappers_and_memory_refusal(monkeypatch):
    """The gpu module wrappers round the float64 assembly and refuse a matrix that does not fit."""
    torch = _need_cuda()
    from ventorum.panel3d import gpu

    panels_a, panels_b = _pair_sets("image")
    points = _targets(panels_a)
    sigma = _sigma(len(panels_a), 3)
    exp_c, exp_r = _reference(points, panels_a, panels_b, 1.0, sigma, 5.0)
    mat, rhs = gpu.pair_system_device(points, panels_a, panels_b, 1.0, sigma, dtype=torch.float32)
    assert mat.dtype == torch.float32 and rhs.dtype == torch.float64
    _assert_roundoff(mat.cpu().numpy(), exp_c, 1e-6, "device pair matrix float32", floor_scale=1.0)
    _assert_roundoff(rhs.cpu().numpy(), exp_r, 1e-12, "device pair rhs float64")
    pair64 = gpu.doublet_pair_matrix_device(points, panels_a, panels_b, 1.0, dtype=torch.float64)
    _assert_roundoff(pair64.cpu().numpy(), exp_c, 1e-12, "device pair matrix float64", floor_scale=1.0)
    single = gpu.doublet_potential_device(points, panels_a, dtype=torch.float64)
    _assert_roundoff(single.cpu().numpy(), ref.doublet_potential(points, panels_a), 1e-12,
                     "device doublet potential", floor_scale=1.0)
    monkeypatch.setattr(gpu, "free_device_bytes", lambda device=None: 0)
    with pytest.raises(MemoryError, match="MiB"):
        gpu.pair_system_device(points, panels_a, panels_b, 1.0, sigma, dtype=torch.float32)
    with pytest.raises(MemoryError, match="MiB"):
        gpu.check_device_memory(1000, "float32", n_systems=2, half=True, assembly="fused")


def test_required_bytes_of_the_fused_assembly():
    """The fused estimate matches its documented terms and is below the parts estimate."""
    from ventorum.panel3d.gpu import N_ONSETS, required_bytes
    from ventorum.panel3d.kernels_warp import BLOCK_BYTES, RHS_CHUNK

    n = 1000
    nn = n * n
    partial = 8 * n * N_ONSETS * (-(-n // RHS_CHUNK))
    # Half mesh, free air, symmetric system only, float32: body parity
    # matrix, work matrix and LU, one float64 block, partial sums.
    assert required_bytes(n, "float32", n_systems=1, half=True, assembly="fused") == (
        nn * 4 + 2 * nn * 4 + min(BLOCK_BYTES, 8 * nn) + partial + 2 * n * 6 * 4
    )
    # Both systems in ground effect with the body cached, float64 (no block).
    assert required_bytes(n, "float64", n_systems=2, half=True, ground=True, body_cached=True,
                          assembly="fused") == (
        nn * 8 + 2 * nn * 8 + nn * 8 + partial + 2 * 2 * n * 6 * 8
    )
    cases = ({"n_systems": 1}, {"n_systems": 2, "half": True},
             {"n_systems": 2, "half": True, "ground": True})
    for kwargs in cases:
        for dtype in ("float32", "float64"):
            fused = required_bytes(n, dtype, assembly="fused", **kwargs)
            assert fused < 0.6 * required_bytes(n, dtype, **kwargs)
    assert required_bytes(0, "float64", assembly="fused") == 0
    with pytest.raises(ValueError, match="assembly"):
        required_bytes(n, "float64", assembly="dense")


# ----------------------------------------------------------------------
# Solver systems against a copy of the old assembly
# ----------------------------------------------------------------------


def _mesh(half: bool):
    """Return a small tapered wing mesh (about 100 panels per half)."""
    surf = LiftingSurface(
        name="Wing", semi_span=3.0, is_symmetric=True,
        sections=[WingSection(y_frac=0.0, chord=1.0), WingSection(y_frac=1.0, chord=0.7)],
    )
    x = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, 41)))
    yt = 0.6 * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)
    coords = np.vstack([np.column_stack([x, yt])[::-1], np.column_stack([x, -yt])[1:]])
    return mesh_lifting_surface(surf, coords, n_chord=7, n_span=5, half=half)


def _old_assembly(solver: PanelSolver, cfg) -> dict:
    """Copy of the old assembly: separate B and C parts, each with its wake fold.

    Returns ``{parity: (A, unit_mu)}`` with the system matrix [-] and the
    unit solutions of its onset columns (zeros elsewhere) in m^2/s.
    """
    from ventorum.panel3d.kutta import strip_influence_matrix

    pts = solver.centroids
    fr = solver.far_ratio
    strips = cfg.wake.strip_panel_indices
    has_wake = len(cfg.wake.panels) > 0

    def part(panels, wake_panels):
        b, c = ref.influence_matrices(pts, panels, far_ratio=fr)
        a = c.copy()
        if has_wake:
            d = strip_influence_matrix(ref.doublet_potential(pts, wake_panels, far_ratio=fr), strips)
            _fold_wake(a, d, cfg.wake)
        return a, b

    a_s, b_s = part(solver.body_verts, cfg.wake.panels)
    if solver.mesh.half:
        a_m, b_m = part(solver._body_verts_m, cfg.wake_panels_m)
    if cfg.ground is not None:
        k, h = cfg.ground
        a_g, b_g = part(reflect_panels_plane(solver.body_verts, k, h),
                        reflect_panels_plane(cfg.wake.panels, k, h) if has_wake else None)
        if solver.mesh.half:
            a_yg, b_yg = part(reflect_panels_plane(solver._body_verts_m, k, h),
                              reflect_panels_plane(cfg.wake_panels_m, k, h) if has_wake else None)
    out = {}
    for parity in cfg.systems:
        if parity == "full":
            mat_a, mat_b, cols = a_s, b_s, list(range(6))
            if cfg.ground is not None:
                mat_a, mat_b = a_s + a_g, b_s + b_g
        else:
            sign = 1.0 if parity == "sym" else -1.0
            mat_a, mat_b = a_s + sign * a_m, b_s + sign * b_m
            if cfg.ground is not None:
                mat_a, mat_b = mat_a + a_g + sign * a_yg, mat_b + b_g + sign * b_yg
            cols = list(_SYM_COLS if parity == "sym" else _ASYM_COLS)
        unit = np.zeros((len(pts), 6))
        lu = scipy.linalg.lu_factor(mat_a)
        unit[:, cols] = scipy.linalg.lu_solve(lu, -mat_b @ solver._unit_sigma[:, cols])
        out[parity] = (mat_a, unit)
    return out


def _capture_systems(monkeypatch, solver: PanelSolver) -> list:
    """Record a float64 host copy of each matrix that the solver factorizes."""
    captured = []
    factorize = solver._factorize

    def capture(mat_a):
        host = mat_a.cpu().numpy() if solver.backend == "gpu" else mat_a
        captured.append(np.array(host, dtype=np.float64, copy=True))
        return factorize(mat_a)

    monkeypatch.setattr(solver, "_factorize", capture)
    return captured


_CONDITIONS = [
    # Free air with sideslip and rates: symmetric and antisymmetric systems.
    FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=np.radians(3.0), p=0.2, r=0.05),
    # Ground effect with a roll rate: both systems with the images.
    FlightCondition(V_inf=30.0, alpha=np.radians(3.0), p=0.3, h=0.6),
    # Ground effect, symmetric flow at a second attitude and height.
    FlightCondition(V_inf=30.0, alpha=np.radians(6.0), q=0.1, h=0.8),
]


def _check_against_old(solver, captured, tol_a, tol_mu, label):
    """Compare the factorized systems and unit solutions with the old assembly."""
    index = 0
    for key, cfg in solver._configs.items():
        old = _old_assembly(solver, cfg)
        for parity, (mat_old, unit_old) in old.items():
            _assert_roundoff(captured[index], mat_old, tol_a, f"{label} {key[0]} {parity} A", floor_scale=1.0)
            index += 1
            _, cols = solver._parity_sign_cols(parity)
            for k in cols:
                _assert_roundoff(cfg.unit_mu[:, k], unit_old[:, k], tol_mu, f"{label} {key[0]} {parity} mu {k}")
    assert index == len(captured)


@pytest.mark.parametrize("backend", ["numpy", "numba"])
@pytest.mark.parametrize("half", [True, False])
def test_solver_systems_match_old_assembly(monkeypatch, backend, half):
    """Free air, ground effect, sideslip and rates: systems and unit solutions to round-off."""
    solver = PanelSolver(_mesh(half), backend=backend, far_ratio=5.0)
    captured = _capture_systems(monkeypatch, solver)
    results = solver.solve_conditions(_CONDITIONS)
    assert len(results) == len(_CONDITIONS)
    keys = [key[0] for key in solver._configs]
    assert keys.count("free") == 1 and keys.count("ground") == 2
    _check_against_old(solver, captured, 1e-12, 1e-12, f"{backend} half={half}")
    # The body cache holds the free-air parity systems only: no source matrix.
    expected = {"A_sym", "rhs_sym", "A_asym", "rhs_asym"} if half else {"A_full", "rhs_full"}
    assert set(solver._body_parts) == expected


def test_pressure_kutta_and_full_mesh_properties():
    """The pressure Kutta uses the strip systems; the A and B properties keep their results."""
    from ventorum.panel3d.kutta import strip_influence_matrix

    half = PanelSolver(_mesh(True), kutta="pressure", far_ratio=5.0)
    res = half.solve(FlightCondition(V_inf=30.0, alpha=np.radians(4.0), h=0.7))
    assert res.kutta == "pressure" and res.kutta_converged
    full = PanelSolver(_mesh(False), far_ratio=5.0)
    mat_b, mat_c = ref.influence_matrices(full.centroids, full.body_verts, far_ratio=5.0)
    assert np.array_equal(full.B, mat_b)
    d = strip_influence_matrix(ref.doublet_potential(full.centroids, full.wake.panels, far_ratio=5.0),
                               full.wake.strip_panel_indices)
    assert np.array_equal(full.A, _fold_wake(mat_c.copy(), d, full.wake))
    # The properties do not fill the solver caches.
    assert full._body_parts is None
    with pytest.raises(AttributeError):
        _ = half.A


def _dense_bytes(solver: PanelSolver, n: int) -> tuple[int, int]:
    """Return (count, bytes) of the (N, N) arrays and tensors that a solver keeps."""
    seen, found = set(), {}

    def walk(obj, depth=0):
        if id(obj) in seen or depth > 6:
            return
        seen.add(id(obj))
        if isinstance(obj, np.ndarray):
            if obj.shape == (n, n):
                found[id(obj)] = int(obj.nbytes)
            return
        if hasattr(obj, "element_size") and hasattr(obj, "shape"):
            if tuple(obj.shape) == (n, n):
                found[id(obj)] = int(obj.nelement() * obj.element_size())
            return
        if isinstance(obj, dict):
            for value in obj.values():
                walk(value, depth + 1)
        elif isinstance(obj, (list, tuple)):
            for value in obj:
                walk(value, depth + 1)
        elif hasattr(obj, "__dict__"):
            for value in vars(obj).values():
                walk(value, depth + 1)

    walk(solver)
    return len(found), sum(found.values())


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_half_mesh_free_air_keeps_one_matrix_and_the_lu(backend):
    """A symmetric free-air solve keeps A_sym and the LU factors, nothing else of size N x N.

    The pressure Kutta condition keeps the LU for its iterations; with the
    Morino condition the solver drops it after the unit solutions (see
    tests/test_panel3d_configuration_budget.py).
    """
    solver = PanelSolver(_mesh(True), backend=backend, far_ratio=5.0, kutta="pressure")
    solver.solve(FlightCondition(V_inf=30.0, alpha=np.radians(4.0)))
    n = len(solver.centroids)
    lu, _ = solver._configs[("free",)].systems["sym"]
    count, kept = _dense_bytes(solver, n)
    assert count == 2
    assert kept <= n * n * 8 + lu.nbytes
    assert set(solver._body_parts) == {"A_sym", "rhs_sym"}


@pytest.mark.gpu
@pytest.mark.parametrize("half", [True, False])
def test_gpu_solver_systems_match_old_assembly(monkeypatch, half):
    """The gpu backend in float64: systems and unit solutions against the old assembly."""
    torch = _need_cuda()
    solver = PanelSolver(_mesh(half), backend="gpu", dtype=torch.float64, far_ratio=5.0)
    captured = _capture_systems(monkeypatch, solver)
    solver.solve_conditions(_CONDITIONS)
    _check_against_old(solver, captured, 1e-12, 1e-10, f"gpu half={half}")


@pytest.mark.gpu
def test_gpu_float32_keeps_one_matrix_and_the_lu(monkeypatch):
    """The float32 gpu solve keeps A_sym and the LU on the device, at the float32 floor.

    The pressure Kutta condition keeps the LU (see the CPU test above).
    """
    torch = _need_cuda()
    solver = PanelSolver(_mesh(True), backend="gpu", dtype=torch.float32, far_ratio=5.0, kutta="pressure")
    captured = _capture_systems(monkeypatch, solver)
    solver.solve(FlightCondition(V_inf=30.0, alpha=np.radians(4.0)))
    n = len(solver.centroids)
    count, kept = _dense_bytes(solver, n)
    assert count == 2 and kept <= 2 * n * n * 4
    cfg = solver._configs[("free",)]
    mat_old, unit_old = _old_assembly(solver, cfg)["sym"]
    _assert_roundoff(captured[0], mat_old, 1e-6, "gpu float32 A_sym", floor_scale=1.0)
    for k in _SYM_COLS:
        _assert_roundoff(cfg.unit_mu[:, k], unit_old[:, k], 1e-4, f"gpu float32 mu {k}")
