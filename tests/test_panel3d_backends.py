# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Parity tests of the compiled 3D panel backends against the numpy reference.

The numpy kernels in :mod:`ventorum.panel3d.kernels` are the reference.
The Numba CPU kernels and the Warp GPU kernels must give the same
influence matrices and velocities to round-off in float64, and within
float32 round-off in float32.
"""

from __future__ import annotations

import numpy as np
import pytest

from ventorum.panel3d import kernels as ref
from ventorum.panel3d import kernels_numba as nb


def _need_cuda():
    """Skip the test when a CUDA GPU, PyTorch or Warp is missing."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("warp")
    if not torch.cuda.is_available():
        pytest.skip("A CUDA GPU is not available.")
    return torch


def _random_panels(seed: int = 7) -> np.ndarray:
    """Return random non-planar quadrilaterals and triangles in m.

    The set mixes non-planar quadrilaterals (out-of-plane corner offsets),
    triangles (fourth corner equal to the third), skewed quads and a thin
    high aspect ratio quad.
    """
    rng = np.random.default_rng(seed)
    panels = []
    for _ in range(4):
        base = rng.uniform(-1.0, 1.0, size=(4, 3))
        base[:, 2] += rng.uniform(-0.3, 0.3, size=4)
        panels.append(base)
    for _ in range(2):
        tri = rng.uniform(-1.0, 1.0, size=(3, 3))
        panels.append(np.vstack([tri, tri[2]]))
    panels.append(np.array([
        [0.0, 0.0, 0.0], [10.0, 0.1, 0.05], [10.0, 1.0, -0.05], [0.0, 1.0, 0.0],
    ]))
    out = np.array(panels, dtype=np.float64)
    return np.ascontiguousarray(out)


def _panel_frame(panel: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return centroid, normal and in-plane axes of one panel in m."""
    p1, p2, p3, p4 = panel
    n_raw = np.cross(p3 - p1, p4 - p2)
    normal = n_raw / np.linalg.norm(n_raw)
    if np.linalg.norm(p4 - p3) < 1e-12:
        centroid = (p1 + p2 + p3) / 3.0
    else:
        centroid = 0.25 * (p1 + p2 + p3 + p4)
    axis = p2 - p1
    if np.linalg.norm(axis) < 1e-12:
        axis = p3 - p1
    axis = axis / np.linalg.norm(axis)
    return centroid, normal, axis, np.cross(normal, axis)


def _targets_for(panels: np.ndarray, seed: int = 11) -> np.ndarray:
    """Return near-field, on-plane, far-field and centroid targets in m."""
    rng = np.random.default_rng(seed)
    pts = []
    for panel in panels:
        centroid, normal, lax, maxis = _panel_frame(panel)
        diam = max(float(np.linalg.norm(panel[i] - panel[j])) for i in range(4) for j in range(4))
        for scale in (0.05, 0.2, 0.7, 2.0):
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            pts.append(centroid + scale * diam * direction)
        pts.append(centroid + 0.1 * diam * normal)
        pts.append(centroid - 0.1 * diam * normal)
        pts.append(centroid + 0.3 * diam * lax + 0.2 * diam * maxis)
        pts.append(centroid + 3.0 * diam * lax - 2.0 * diam * maxis)
        pts.append(centroid + 12.0 * diam * normal)
        pts.append(centroid)
    return np.ascontiguousarray(np.array(pts, dtype=np.float64))


def _targets_off_plane(panels: np.ndarray, seed: int = 23) -> np.ndarray:
    """Return targets with a strictly positive height above every panel in m.

    The doublet potential jumps across the panel plane (body-side limit
    +1/2, fluid-side limit -1/2), so float32 input rounding alone can flip
    an exactly on-plane target to the other side. The float32 floor tests
    use this set, where every target keeps a height above 1e-4 diameters.
    """
    rng = np.random.default_rng(seed)
    pts = []
    for panel in panels:
        centroid, normal, lax, maxis = _panel_frame(panel)
        diam = max(float(np.linalg.norm(panel[i] - panel[j])) for i in range(4) for j in range(4))
        for scale in (0.05, 0.2, 0.7, 2.0):
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            while abs(float(direction @ normal)) < 0.3:
                direction = rng.normal(size=3)
                direction /= np.linalg.norm(direction)
            pts.append(centroid + scale * diam * direction)
        pts.append(centroid + 0.1 * diam * normal)
        pts.append(centroid - 0.1 * diam * normal)
        pts.append(centroid + 0.3 * diam * lax + 0.2 * diam * maxis + 1e-3 * diam * normal)
        pts.append(centroid + 12.0 * diam * normal)
    return np.ascontiguousarray(np.array(pts, dtype=np.float64))


def _rel_err(new: np.ndarray, reference: np.ndarray) -> float:
    """Return the relative error of *new* against *reference*."""
    denom = float(np.max(np.abs(reference)))
    if denom == 0.0:
        return float(np.max(np.abs(new)))
    return float(np.max(np.abs(new - reference)) / denom)


def _assert_parity(new: np.ndarray, reference: np.ndarray, tol: float, name: str) -> None:
    """Assert agreement to *tol* relative or 1e-14 absolute."""
    err = _rel_err(new, reference)
    abs_err = float(np.max(np.abs(new - reference)))
    assert err <= tol or abs_err <= 1e-14, f"{name}: rel {err:.3e} abs {abs_err:.3e} above {tol:.1e}"


@pytest.mark.parametrize("far_ratio", [5.0, np.inf])
def test_numba_matrices_equal_numpy_float64(far_ratio: float) -> None:
    """Numba B and C equal numpy to 1e-12 on quads, triangles and all ranges."""
    panels = _random_panels()
    points = _targets_for(panels)
    ref_b, ref_c = ref.influence_matrices(points, panels, far_ratio=far_ratio)
    new_b, new_c = nb.influence_matrices_numba(points, panels, far_ratio=far_ratio)
    assert new_b.shape == ref_b.shape
    assert new_c.shape == ref_c.shape
    _assert_parity(new_b, ref_b, 1e-12, "B")
    _assert_parity(new_c, ref_c, 1e-12, "C")


@pytest.mark.parametrize("far_ratio", [5.0, np.inf])
def test_numba_velocities_equal_numpy_float64(far_ratio: float) -> None:
    """Numba source and doublet velocities equal numpy to 1e-12."""
    panels = _random_panels()
    points = _targets_for(panels)
    ref_vs = ref.source_velocity(points, panels, far_ratio=far_ratio)
    new_vs = nb.source_velocity_numba(points, panels, far_ratio=far_ratio)
    _assert_parity(new_vs, ref_vs, 1e-12, "source velocity")
    ref_vd = ref.doublet_velocity(points, panels, far_ratio=far_ratio)
    new_vd = nb.doublet_velocity_numba(points, panels, far_ratio=far_ratio)
    _assert_parity(new_vd, ref_vd, 1e-12, "doublet velocity")


def test_numba_self_terms_equal_numpy_float64() -> None:
    """Numba self influence at the own centroids equals numpy, limit 1/2."""
    panels = np.array([
        [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]],
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.5, 0.0], [1.0, 1.5, 0.0]],
    ], dtype=np.float64)
    centroids = np.array([
        [0.0, 0.0, 0.0],
        [(0.0 + 2.0 + 1.0) / 3.0, (0.0 + 0.0 + 1.5) / 3.0, 0.0],
    ])
    ref_b, ref_c = ref.influence_matrices(centroids, panels, far_ratio=np.inf)
    new_b, new_c = nb.influence_matrices_numba(centroids, panels, far_ratio=np.inf)
    _assert_parity(new_b, ref_b, 1e-12, "self B")
    _assert_parity(new_c, ref_c, 1e-12, "self C")
    assert np.allclose(np.diag(new_c), 0.5, atol=1e-12)


@pytest.mark.parametrize("far_ratio", [5.0, np.inf])
def test_numba_float32_matches_numpy_on_float32_inputs(far_ratio: float, unsafe_float32_kernels) -> None:
    """Numba on float32 inputs equals numpy on the same float32 inputs to 1e-12.

    This covers the full target set (near-field, on-plane, far-field, own
    centroids): both paths run identical float64 arithmetic on identical
    bits, so every branch decides the same way.
    """
    panels = _random_panels()
    points = _targets_for(panels)
    panels32 = panels.astype(np.float32)
    points32 = points.astype(np.float32)
    ref_b, ref_c = ref.influence_matrices(points32, panels32, far_ratio=far_ratio)
    new_b, new_c = nb.influence_matrices_numba(points32, panels32, far_ratio=far_ratio)
    assert new_b.dtype == np.float32
    assert new_c.dtype == np.float32
    _assert_parity(new_b, ref_b, 1e-12, "B f32 inputs")
    _assert_parity(new_c, ref_c, 1e-12, "C f32 inputs")
    ref_vs = ref.source_velocity(points32, panels32, far_ratio=far_ratio)
    new_vs = nb.source_velocity_numba(points32, panels32, far_ratio=far_ratio)
    _assert_parity(new_vs, ref_vs, 1e-12, "Vs f32 inputs")
    ref_vd = ref.doublet_velocity(points32, panels32, far_ratio=far_ratio)
    new_vd = nb.doublet_velocity_numba(points32, panels32, far_ratio=far_ratio)
    _assert_parity(new_vd, ref_vd, 1e-12, "Vd f32 inputs")


def test_numba_float32_floor(unsafe_float32_kernels) -> None:
    """Numba float32 stays within float32 round-off of the float64 reference.

    The test measures the error floor and states it. The gate is 1e-4;
    the expected floor is about 1e-6 relative. Exactly on-plane targets
    are excluded: the doublet jumps across the plane, so float32 input
    rounding alone can flip the side (the numpy reference disagrees with
    itself across input precisions there).
    """
    panels = _random_panels()
    points = _targets_off_plane(panels)
    ref_b, ref_c = ref.influence_matrices(points, panels, far_ratio=5.0)
    ref_vs = ref.source_velocity(points, panels, far_ratio=5.0)
    ref_vd = ref.doublet_velocity(points, panels, far_ratio=5.0)
    panels32 = panels.astype(np.float32)
    points32 = points.astype(np.float32)
    new_b, new_c = nb.influence_matrices_numba(points32, panels32, far_ratio=5.0)
    new_vs = nb.source_velocity_numba(points32, panels32, far_ratio=5.0)
    new_vd = nb.doublet_velocity_numba(points32, panels32, far_ratio=5.0)
    assert new_b.dtype == np.float32
    assert new_c.dtype == np.float32
    assert new_vs.dtype == np.float32
    assert new_vd.dtype == np.float32
    floors = {
        "B": _rel_err(new_b.astype(np.float64), ref_b),
        "C": _rel_err(new_c.astype(np.float64), ref_c),
        "Vs": _rel_err(new_vs.astype(np.float64), ref_vs),
        "Vd": _rel_err(new_vd.astype(np.float64), ref_vd),
    }
    print(f"numba float32 floors: {floors}")
    for name, err in floors.items():
        assert err <= 1e-4, f"numba float32 {name} floor {err:.3e} above 1e-4"


def test_numba_empty_inputs() -> None:
    """Empty targets or panels give empty outputs with the right shape."""
    panels = _random_panels()[:2]
    points = _targets_for(panels)[:3]
    empty_pts = np.empty((0, 3), dtype=np.float64)
    empty_pans = np.empty((0, 4, 3), dtype=np.float64)
    new_b, new_c = nb.influence_matrices_numba(empty_pts, panels)
    assert new_b.shape == (0, 2)
    assert new_c.shape == (0, 2)
    new_b, new_c = nb.influence_matrices_numba(points, empty_pans)
    assert new_b.shape == (3, 0)
    assert new_c.shape == (3, 0)
    assert nb.source_velocity_numba(empty_pts, panels).shape == (0, 2, 3)
    assert nb.doublet_velocity_numba(points, empty_pans).shape == (3, 0, 3)


def test_backend_selector() -> None:
    """The selector routes to numpy and numba with equal results."""
    panels = _random_panels()[:3]
    points = _targets_for(panels)[:12]
    direct = ref.influence_matrices(points, panels)
    for backend in ("numpy", "numba"):
        got_b, got_c = ref.influence_matrices_backend(points, panels, backend=backend)
        _assert_parity(np.asarray(got_b), direct[0], 1e-12, f"B {backend}")
        _assert_parity(np.asarray(got_c), direct[1], 1e-12, f"C {backend}")
    with pytest.raises(ValueError):
        ref.influence_matrices_backend(points, panels, backend="cuda-please")


@pytest.mark.gpu
@pytest.mark.parametrize("far_ratio", [5.0, np.inf])
def test_warp_matrices_equal_numpy_float64(far_ratio: float) -> None:
    """Warp float64 B and C equal numpy to 1e-12 on CUDA."""
    torch = _need_cuda()
    from ventorum.panel3d import kernels_warp as wpk

    panels = _random_panels()
    points = _targets_for(panels)
    ref_b, ref_c = ref.influence_matrices(points, panels, far_ratio=far_ratio)
    new_b, new_c = wpk.influence_matrices_warp(points, panels, far_ratio=far_ratio, dtype=torch.float64)
    assert isinstance(new_b, torch.Tensor) and new_b.is_cuda
    assert isinstance(new_c, torch.Tensor) and new_c.is_cuda
    _assert_parity(new_b.cpu().numpy(), ref_b, 1e-12, "warp B")
    _assert_parity(new_c.cpu().numpy(), ref_c, 1e-12, "warp C")


@pytest.mark.gpu
def test_warp_float32_floor(unsafe_float32_kernels) -> None:
    """Warp float32 stays within float32 round-off of the float64 reference.

    The test measures the error floor and states it. The gate is 1e-4;
    the expected floor is about 1e-6 relative. Exactly on-plane targets
    are excluded (see test_numba_float32_floor).
    """
    torch = _need_cuda()
    from ventorum.panel3d import kernels_warp as wpk

    panels = _random_panels()
    points = _targets_off_plane(panels)
    ref_b, ref_c = ref.influence_matrices(points, panels, far_ratio=5.0)
    new_b, new_c = wpk.influence_matrices_warp(points, panels, far_ratio=5.0, dtype=torch.float32)
    assert new_b.dtype == torch.float32
    floors = {
        "B": _rel_err(new_b.cpu().numpy().astype(np.float64), ref_b),
        "C": _rel_err(new_c.cpu().numpy().astype(np.float64), ref_c),
    }
    print(f"warp float32 floors: {floors}")
    for name, err in floors.items():
        assert err <= 1e-4, f"warp float32 {name} floor {err:.3e} above 1e-4"


def _rotated_cube(n: int = 4, seed: int = 4, offset=(7.3, -2.1, 4.4)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a rotated and moved closed cube with outward panels [m].

    Returns the panels (6 n^2, 4, 3), the collocation points (panel corner
    averages) and the edge midpoints, all in float64 [m]. The collocation
    points come from the unrotated cube and are rotated and moved on their
    own, so float32 rounding of the inputs moves them off the panel planes.
    """
    u = np.linspace(-1.0, 1.0, n + 1)
    panels = []
    for axis in range(3):
        for sign in (1.0, -1.0):
            a_u, a_v = (axis + 1) % 3, (axis + 2) % 3
            if sign < 0.0:
                a_u, a_v = a_v, a_u
            for i in range(n):
                for j in range(n):
                    quad = []
                    for du, dv in ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)):
                        p = np.zeros(3)
                        p[axis] = sign
                        p[a_u] = u[du]
                        p[a_v] = u[dv]
                        quad.append(p)
                    panels.append(quad)
    cube = np.array(panels)
    colloc = cube.mean(axis=1)
    mids = 0.5 * (cube + np.roll(cube, -1, axis=1)).reshape(-1, 3)
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0.0:
        q[:, 0] = -q[:, 0]
    off = np.asarray(offset, dtype=float)
    return cube @ q.T + off, colloc @ q.T + off, mids @ q.T + off


def test_float32_on_plane_and_self_terms_numpy_numba(unsafe_float32_kernels) -> None:
    """float32 inputs keep the on-plane terms and the self terms +1/2.

    The targets are the own collocation points (self terms), the coplanar
    neighbour collocation points (on-plane, outside) and the edge midpoints
    (on-plane, on an edge: 1/4 per panel) of a rotated and moved closed
    cube. The collocation points are rounded to float32 on their own, not
    recomputed from the rounded corners. Numpy and Numba on float32 inputs
    stay within the float32 floor of the float64 reference (measured and
    printed, gate 1e-4), and every self term is +1/2.
    """
    panels, colloc, mids = _rotated_cube()
    targets = np.vstack([colloc, mids])
    n = panels.shape[0]
    ref_b, ref_c = ref.influence_matrices(targets, panels, far_ratio=np.inf)
    ref_vs = ref.source_velocity(colloc, panels, far_ratio=np.inf)
    ref_vd = ref.doublet_velocity(colloc, panels, far_ratio=np.inf)
    assert np.allclose(np.diag(ref_c[:n]), 0.5, rtol=0.0, atol=1e-12)
    pans32 = panels.astype(np.float32)
    tg32 = targets.astype(np.float32)
    col32 = colloc.astype(np.float32)
    floors = {}
    for label, matrices, vs_fn, vd_fn in (
        ("numpy", ref.influence_matrices, ref.source_velocity, ref.doublet_velocity),
        ("numba", nb.influence_matrices_numba, nb.source_velocity_numba, nb.doublet_velocity_numba),
    ):
        new_b, new_c = matrices(tg32, pans32, far_ratio=np.inf)
        assert new_c.dtype == np.float32
        assert np.allclose(np.diag(new_c[:n]), 0.5, rtol=0.0, atol=1e-6), f"{label} self terms flipped"
        floors[f"{label} B"] = _rel_err(new_b.astype(np.float64), ref_b)
        floors[f"{label} C"] = _rel_err(new_c.astype(np.float64), ref_c)
        floors[f"{label} Vs"] = _rel_err(vs_fn(col32, pans32, far_ratio=np.inf).astype(np.float64), ref_vs)
        floors[f"{label} Vd"] = _rel_err(vd_fn(col32, pans32, far_ratio=np.inf).astype(np.float64), ref_vd)
    print(f"float32 on-plane floors: {floors}")
    for name, err in floors.items():
        assert err <= 1e-4, f"float32 {name} floor {err:.3e} above 1e-4"


def test_numba_float32_full_target_set(unsafe_float32_kernels) -> None:
    """Numba float32 matches the float64 reference on all targets.

    This set has exactly on-plane targets and the own centroids of
    non-planar quads and triangles (excluded from the older floor test).
    """
    panels = _random_panels()
    points = _targets_for(panels)
    ref_b, ref_c = ref.influence_matrices(points, panels, far_ratio=5.0)
    new_b, new_c = nb.influence_matrices_numba(points.astype(np.float32), panels.astype(np.float32), far_ratio=5.0)
    floors = {
        "B": _rel_err(new_b.astype(np.float64), ref_b),
        "C": _rel_err(new_c.astype(np.float64), ref_c),
    }
    print(f"numba float32 full-set floors: {floors}")
    for name, err in floors.items():
        assert err <= 1e-4, f"numba float32 full-set {name} floor {err:.3e} above 1e-4"


@pytest.mark.gpu
def test_warp_float32_on_plane_and_self_terms(unsafe_float32_kernels) -> None:
    """Warp float32 keeps the on-plane terms and the self terms +1/2 on CUDA.

    Same targets as the numpy and Numba float32 test (rotated closed cube,
    collocation points rounded to float32 on their own). The floor is
    measured and printed, gate 1e-4.
    """
    torch = _need_cuda()
    from ventorum.panel3d import kernels_warp as wpk

    panels, colloc, mids = _rotated_cube()
    targets = np.vstack([colloc, mids])
    n = panels.shape[0]
    ref_b, ref_c = ref.influence_matrices(targets, panels, far_ratio=np.inf)
    for inputs in ("float32", "float64"):
        pts = targets.astype(inputs)
        pans = panels.astype(inputs)
        new_b, new_c = wpk.influence_matrices_warp(pts, pans, far_ratio=np.inf, dtype=torch.float32)
        got_c = new_c.cpu().numpy().astype(np.float64)
        got_b = new_b.cpu().numpy().astype(np.float64)
        assert np.allclose(np.diag(got_c[:n]), 0.5, rtol=0.0, atol=1e-6), f"warp self terms flipped ({inputs} inputs)"
        floors = {"B": _rel_err(got_b, ref_b), "C": _rel_err(got_c, ref_c)}
        print(f"warp float32 on-plane floors ({inputs} inputs): {floors}")
        for name, err in floors.items():
            assert err <= 1e-4, f"warp float32 {name} floor {err:.3e} above 1e-4"
    new_b, new_c = wpk.influence_matrices_warp(targets, panels, far_ratio=np.inf, dtype=torch.float64)
    _assert_parity(new_b.cpu().numpy(), ref_b, 1e-12, "warp B cube")
    _assert_parity(new_c.cpu().numpy(), ref_c, 1e-12, "warp C cube")


@pytest.mark.gpu
def test_backend_selector_warp() -> None:
    """The selector routes to warp on CUDA with results equal to numpy."""
    torch = _need_cuda()
    panels = _random_panels()[:3]
    points = _targets_for(panels)[:12]
    direct = ref.influence_matrices(points, panels)
    got_b, got_c = ref.influence_matrices_backend(points, panels, backend="warp", dtype=torch.float64)
    assert isinstance(got_b, torch.Tensor) and got_b.is_cuda
    _assert_parity(got_b.cpu().numpy(), direct[0], 1e-12, "warp selector B")
    _assert_parity(got_c.cpu().numpy(), direct[1], 1e-12, "warp selector C")
