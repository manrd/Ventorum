# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for the vortex-ring Biot-Savart kernels and steady wake closure.

The numpy implementation in :mod:`ventorum.aero.vortex_ring` is the
reference. The tests check it against direct sums of ``segment_velocity``
and ``ray_velocity``, verify Numba parity, check closed-ring dipole decay
and shared-edge cancellation, and test boundary and chunking cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from ventorum.aero import vortex as V
from ventorum.aero import vortex_ring as VR
from ventorum.aero import vortex_ring_numba as VR_NB


@pytest.fixture(autouse=True)
def _restore_kernel_backend():
    """Restore the global vortex kernel backend after every test."""
    old = V._kernel_backend
    yield
    V._kernel_backend = old


def _random_rings(n: int, seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return random evaluation points, unit normals, 3D rings, wake dirs and cores."""
    rng = np.random.default_rng(seed)
    m = 37
    P = rng.normal(scale=2.0, size=(m, 3))
    normals = rng.normal(size=(m, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    centers = rng.normal(scale=2.0, size=(n, 3))
    dx = rng.uniform(0.2, 0.8, size=(n, 1))
    dy = rng.uniform(0.2, 0.8, size=(n, 1))
    jitter = rng.normal(scale=0.05, size=(n, 4, 3))
    corners = np.empty((n, 4, 3), dtype=float)
    corners[:, 0, :] = centers + np.hstack([-0.5 * dx, -0.5 * dy, np.zeros((n, 1))])
    corners[:, 1, :] = centers + np.hstack([-0.5 * dx, +0.5 * dy, np.zeros((n, 1))])
    corners[:, 2, :] = centers + np.hstack([+0.5 * dx, +0.5 * dy, np.zeros((n, 1))])
    corners[:, 3, :] = centers + np.hstack([+0.5 * dx, -0.5 * dy, np.zeros((n, 1))])
    corners += jitter

    # Place a few points on extended lines and close to corners to exercise both core branches.
    P[0] = corners[0, 0] + 1.5 * (corners[0, 1] - corners[0, 0])
    P[1] = 0.5 * (corners[1, 0] + corners[1, 1]) + np.array([0.0, 0.0, 1e-4])
    P[2] = corners[2, 2] + 1e-5 * np.array([1.0, -1.0, 0.5])

    wake_dir = rng.normal(size=(n, 3))
    wake_dir[:, 0] = np.abs(wake_dir[:, 0]) + 1.0
    wake_dir /= np.linalg.norm(wake_dir, axis=1, keepdims=True)
    rc = rng.uniform(1e-4, 5e-2, size=n)
    return P, normals, corners, wake_dir, rc


@pytest.mark.parametrize("rc_mode", ["scalar", "1d", "2d", "zero"])
def test_numpy_ring_kernel_matches_four_segments(rc_mode):
    """Numpy ring velocity and normal influence match four direct segment_velocity calls."""
    P, normals, corners, _, rc_1d = _random_rings(n=25, seed=11)
    m, n = P.shape[0], corners.shape[0]
    if rc_mode == "scalar":
        rc = 0.015
    elif rc_mode == "1d":
        rc = rc_1d
    elif rc_mode == "2d":
        rng = np.random.default_rng(19)
        rc = rng.uniform(1e-4, 5e-2, size=(m, n))
    else:
        rc = 0.0

    V.set_kernel_backend("numpy")
    got_v = VR.ring_velocity(P, corners, rc)
    got_a = VR.ring_normal_influence(P, normals, corners, rc)

    ref_v = (
        V.segment_velocity(P, corners[:, 0], corners[:, 1], rc)
        + V.segment_velocity(P, corners[:, 1], corners[:, 2], rc)
        + V.segment_velocity(P, corners[:, 2], corners[:, 3], rc)
        + V.segment_velocity(P, corners[:, 3], corners[:, 0], rc)
    )
    ref_a = np.sum(ref_v * normals[:, None, :], axis=2)

    assert got_v.shape == (m, n, 3)
    assert got_a.shape == (m, n)
    assert np.max(np.abs(got_v - ref_v)) <= 1e-13 * np.max(np.abs(ref_v))
    assert np.max(np.abs(got_a - ref_a)) <= 1e-13 * np.max(np.abs(ref_a))


@pytest.mark.parametrize("dir_mode", ["1d", "2d"])
@pytest.mark.parametrize("rc_mode", ["scalar", "1d", "2d", "zero"])
def test_numpy_steady_wake_matches_segment_and_rays(dir_mode, rc_mode):
    """Steady wake closure equals the canceling TE segment plus two semi-infinite rays."""
    P, normals, corners, wake_dir_2d, rc_1d = _random_rings(n=18, seed=23)
    m, n = P.shape[0], corners.shape[0]
    te_left = corners[:, 3]
    te_right = corners[:, 2]
    if dir_mode == "1d":
        wake_dir = np.array([np.cos(0.1), 0.0, np.sin(0.1)])
        wd_2d = np.broadcast_to(wake_dir, (n, 3))
    else:
        wake_dir = wake_dir_2d
        wd_2d = wake_dir_2d
    if rc_mode == "scalar":
        rc = 0.01
    elif rc_mode == "1d":
        rc = rc_1d
    elif rc_mode == "2d":
        rc = np.random.default_rng(29).uniform(1e-4, 5e-2, size=(m, n))
    else:
        rc = 0.0

    V.set_kernel_backend("numpy")
    got_v = VR.steady_wake_velocity(P, te_left, te_right, wake_dir, rc)
    got_a = VR.steady_wake_normal_influence(P, normals, te_left, te_right, wake_dir, rc)

    ref_v = (
        V.segment_velocity(P, te_left, te_right, rc)
        + V.ray_velocity(P, te_right, wd_2d, rc)
        - V.ray_velocity(P, te_left, wd_2d, rc)
    )
    ref_a = np.sum(ref_v * normals[:, None, :], axis=2)

    assert got_v.shape == (m, n, 3)
    assert got_a.shape == (m, n)
    assert np.max(np.abs(got_v - ref_v)) <= 1e-13 * np.max(np.abs(ref_v))
    assert np.max(np.abs(got_a - ref_a)) <= 1e-13 * np.max(np.abs(ref_a))

    # Closed TE ring plus steady wake closure equals an open bent horseshoe.
    ring_plus_wake = VR.ring_velocity(P, corners, rc) + got_v
    open_horseshoe = (
        V.segment_velocity(P, corners[:, 0], corners[:, 1], rc)
        + V.segment_velocity(P, corners[:, 1], te_right, rc)
        + V.segment_velocity(P, te_left, corners[:, 0], rc)
        + V.ray_velocity(P, te_right, wd_2d, rc)
        - V.ray_velocity(P, te_left, wd_2d, rc)
    )
    assert np.max(np.abs(ring_plus_wake - open_horseshoe)) <= 1e-13 * np.max(np.abs(open_horseshoe))


@pytest.mark.parametrize("rc_mode", ["scalar", "1d", "2d", "zero"])
def test_numba_ring_kernels_equal_numpy(rc_mode):
    """Numba ring and steady-wake kernels agree with numpy to round-off."""
    if not VR_NB.HAVE_NUMBA:
        pytest.skip("Numba is not installed.")
    P, normals, corners, wake_dir, rc_1d = _random_rings(n=24, seed=31)
    m, n = P.shape[0], corners.shape[0]
    te_left = corners[:, 3]
    te_right = corners[:, 2]
    if rc_mode == "scalar":
        rc = 0.012
    elif rc_mode == "1d":
        rc = rc_1d
    elif rc_mode == "2d":
        rc = np.random.default_rng(37).uniform(1e-4, 5e-2, size=(m, n))
    else:
        rc = 0.0

    out = {}
    for backend in ("numpy", "numba"):
        V.set_kernel_backend(backend)
        out[backend] = (
            VR.ring_velocity(P, corners, rc),
            VR.ring_normal_influence(P, normals, corners, rc),
            VR.steady_wake_velocity(P, te_left, te_right, wake_dir, rc),
            VR.steady_wake_normal_influence(P, normals, te_left, te_right, wake_dir, rc),
        )
    for ref, new in zip(out["numpy"], out["numba"]):
        assert np.max(np.abs(new - ref)) <= 1e-12 * np.max(np.abs(ref))


def test_closed_ring_dipole_decay():
    """A closed vortex ring decays as 1/R^3 in the far field, faster than a 1/R^2 segment."""
    corners = np.array([[
        [-0.5, -0.5, 0.0],
        [-0.5, +0.5, 0.0],
        [+0.5, +0.5, 0.0],
        [+0.5, -0.5, 0.0],
    ]])
    radii = np.array([20.0, 40.0, 80.0, 160.0])
    direction = np.array([0.3, -0.4, 0.8660254])
    direction /= np.linalg.norm(direction)
    P = radii[:, None] * direction[None, :]

    v_ring = VR.ring_velocity(P, corners, 0.0)[:, 0, :]
    v_seg = V.segment_velocity(P, corners[:, 0], corners[:, 1], 0.0)[:, 0, :]

    mag_ring = np.linalg.norm(v_ring, axis=1)
    mag_seg = np.linalg.norm(v_seg, axis=1)

    # Doubling distance divides ring velocity by ~8 (1/R^3) and segment velocity by ~4 (1/R^2).
    ring_ratios = mag_ring[:-1] / mag_ring[1:]
    seg_ratios = mag_seg[:-1] / mag_seg[1:]
    assert np.all((ring_ratios > 7.9) & (ring_ratios < 8.1))
    assert np.all((seg_ratios > 3.95) & (seg_ratios < 4.05))

    # Ratio |v_ring| / |v_seg| decays as 1/R (halves when R doubles).
    rel_ratios = (mag_ring / mag_seg)[:-1] / (mag_ring / mag_seg)[1:]
    assert np.all((rel_ratios > 1.98) & (rel_ratios < 2.02))


@pytest.mark.parametrize("orientation", ["spanwise", "chordwise"])
@pytest.mark.parametrize("rc", [0.0, 0.01])
def test_adjacent_equal_strength_rings_cancel_shared_edge(orientation, rc):
    """Two adjacent rings of equal strength induce the same velocity as one outer ring."""
    if orientation == "spanwise":
        # Two side-by-side rings sharing edge x in [0, 1], y = 0.
        r1 = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, -1.0, 0.0]])
        r2 = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        outer = np.array([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, -1.0, 0.0]])
    else:
        # Two tandem rings sharing edge x = 0.5, y in [-0.5, 0.5].
        r1 = np.array([[0.0, -0.5, 0.0], [0.0, 0.5, 0.0], [0.5, 0.5, 0.0], [0.5, -0.5, 0.0]])
        r2 = np.array([[0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [1.0, 0.5, 0.0], [1.0, -0.5, 0.0]])
        outer = np.array([[0.0, -0.5, 0.0], [0.0, 0.5, 0.0], [1.0, 0.5, 0.0], [1.0, -0.5, 0.0]])

    # Include points very close to the shared interior edge.
    rng = np.random.default_rng(41)
    P = rng.normal(scale=1.5, size=(20, 3))
    if orientation == "spanwise":
        P[0] = np.array([0.5, 1e-4, 1e-4])
        P[1] = np.array([0.3, -1e-4, 2e-4])
    else:
        P[0] = np.array([0.5, 0.1, 1e-4])
        P[1] = np.array([0.5, -0.2, 2e-4])

    both = np.stack([r1, r2], axis=0)
    v_each = VR.ring_velocity(P, both, rc)
    v_sum = v_each.sum(axis=1)
    scale = max(float(np.max(np.abs(v_each))), 1.0)
    if rc == 0.0:
        # With exact Biot-Savart, collinear outer edges also add identically.
        # Points at 1e-4 m from the shared edge cancel O(1e3) velocities down to O(1),
        # so the round-off residual scales with the individual ring velocity.
        v_outer = VR.ring_velocity(P, outer[None, :, :], rc)[:, 0, :]
        assert np.max(np.abs(v_sum - v_outer)) <= 1e-14 * scale
        assert np.max(np.abs(v_sum - v_outer)) <= 1e-12 * np.max(np.abs(v_outer))
    else:
        # With nonzero finite-segment core, the shared interior edge still cancels exactly,
        # while the six remaining perimeter segments stay intact.
        if orientation == "spanwise":
            segs = [
                (r1[0], r1[1]), (r2[0], r2[1]), (r2[1], r2[2]),
                (r2[2], r2[3]), (r1[2], r1[3]), (r1[3], r1[0]),
            ]
        else:
            segs = [
                (r1[0], r1[1]), (r1[1], r1[2]), (r2[1], r2[2]),
                (r2[2], r2[3]), (r2[3], r2[0]), (r1[3], r1[0]),
            ]
        v_perim = sum(V.segment_velocity(P, a[None, :], b[None, :], rc)[:, 0, :] for a, b in segs)
        assert np.max(np.abs(v_sum - v_perim)) <= 1e-14 * scale
        assert np.max(np.abs(v_sum - v_perim)) <= 1e-12 * np.max(np.abs(v_perim))


def test_numpy_chunking_gives_same_result(monkeypatch):
    """Small row chunks agree with one large chunk to floating-point round-off."""
    P, normals, corners, wake_dir, rc = _random_rings(n=16, seed=53)
    V.set_kernel_backend("numpy")
    v_big = VR.ring_velocity(P, corners, rc)
    a_big = VR.ring_normal_influence(P, normals, corners, rc)
    w_big = VR.steady_wake_velocity(P, corners[:, 3], corners[:, 2], wake_dir, rc)
    monkeypatch.setattr(VR, "_CHUNK_BUDGET", 500)
    v_small = VR.ring_velocity(P, corners, rc)
    a_small = VR.ring_normal_influence(P, normals, corners, rc)
    w_small = VR.steady_wake_velocity(P, corners[:, 3], corners[:, 2], wake_dir, rc)
    assert np.max(np.abs(v_small - v_big)) <= 1e-14 * np.max(np.abs(v_big))
    assert np.max(np.abs(a_small - a_big)) <= 1e-14 * np.max(np.abs(a_big))
    assert np.max(np.abs(w_small - w_big)) <= 1e-14 * np.max(np.abs(w_big))


def test_empty_inputs_and_invalid_backend():
    """Empty point or ring arrays return matching empty shapes; unknown backend raises ValueError."""
    P0 = np.zeros((0, 3))
    N0 = np.zeros((0, 3))
    C0 = np.zeros((0, 4, 3))
    P2 = np.zeros((2, 3))
    N2 = np.tile([0.0, 0.0, 1.0], (2, 1))
    C2 = np.zeros((2, 4, 3))

    for backend in ("numpy", "numba") if VR_NB.HAVE_NUMBA else ("numpy",):
        assert VR.ring_velocity(P0, C2, 0.0, backend=backend).shape == (0, 2, 3)
        assert VR.ring_velocity(P2, C0, 0.0, backend=backend).shape == (2, 0, 3)
        assert VR.ring_normal_influence(P0, N0, C2, 0.0, backend=backend).shape == (0, 2)
        assert VR.ring_normal_influence(P2, N2, C0, 0.0, backend=backend).shape == (2, 0)
        assert VR.steady_wake_velocity(P2, np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), 0.0,
                                       backend=backend).shape == (2, 0, 3)
        assert VR.steady_wake_normal_influence(P2, N2, np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), 0.0,
                                               backend=backend).shape == (2, 0)

    with pytest.raises(ValueError, match="Unknown ring kernel backend"):
        VR.ring_velocity(P2, C2, 0.0, backend="invalid")


def _random_bent_rings(
    n: int,
    max_nodes: int = 2,
    seed: int = 71,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return random points, normals, 3D rings, bent side nodes, side counts and cores."""
    P, normals, corners, _, rc = _random_rings(n=n, seed=seed)
    rng = np.random.default_rng(seed + 100)
    side_counts = np.zeros((n, 2), dtype=np.int64)
    side_nodes = np.zeros((n, 2, max_nodes, 3), dtype=np.float64)

    for j in range(n):
        cr = int(rng.integers(0, max_nodes + 1))
        cl = int(rng.integers(0, max_nodes + 1))
        side_counts[j, 0] = cr
        side_counts[j, 1] = cl
        # Right side edge: c1 -> c2.
        for k in range(cr):
            frac = (k + 1) / (cr + 1)
            base = (1.0 - frac) * corners[j, 1] + frac * corners[j, 2]
            side_nodes[j, 0, k] = base + rng.normal(scale=0.06, size=3)
        for k in range(cr, max_nodes):
            side_nodes[j, 0, k] = corners[j, 2]
        # Left side edge: c3 -> c0.
        for k in range(cl):
            frac = (k + 1) / (cl + 1)
            base = (1.0 - frac) * corners[j, 3] + frac * corners[j, 0]
            side_nodes[j, 1, k] = base + rng.normal(scale=0.06, size=3)
        for k in range(cl, max_nodes):
            side_nodes[j, 1, k] = corners[j, 0]

    # Ensure at least one ring has zero, one, and max_nodes intermediate nodes.
    if n >= 3 and max_nodes >= 2:
        side_counts[0] = (0, 0)
        side_counts[1] = (1, 0)
        side_nodes[1, 0, 0] = 0.5 * (corners[1, 1] + corners[1, 2]) + np.array([0.0, 0.0, 0.05])
        side_counts[2] = (2, 2)
        side_nodes[2, 0, 0] = (2.0 * corners[2, 1] + corners[2, 2]) / 3.0 + np.array([0.0, 0.0, 0.04])
        side_nodes[2, 0, 1] = (corners[2, 1] + 2.0 * corners[2, 2]) / 3.0 + np.array([0.0, 0.0, 0.03])
        side_nodes[2, 1, 0] = (2.0 * corners[2, 3] + corners[2, 0]) / 3.0 + np.array([0.0, 0.0, 0.03])
        side_nodes[2, 1, 1] = (corners[2, 3] + 2.0 * corners[2, 0]) / 3.0 + np.array([0.0, 0.0, 0.04])
        # Put an evaluation point near a bent intermediate node.
        P[3] = side_nodes[2, 0, 0] + np.array([0.0, 0.0, 1e-4])

    return P, normals, corners, side_nodes, side_counts, rc


def _direct_poly_ring_velocity(
    P: np.ndarray,
    corners: np.ndarray,
    side_nodes: np.ndarray,
    side_counts: np.ndarray,
    rc: np.ndarray | float,
) -> np.ndarray:
    """Reference bent-ring velocity computed directly from segment_velocity."""
    m, n = P.shape[0], corners.shape[0]
    out = np.zeros((m, n, 3), dtype=np.float64)
    for j in range(n):
        rc_j = rc[:, j : j + 1] if np.ndim(rc) == 2 else (rc[j : j + 1] if np.ndim(rc) == 1 else rc)
        cr = int(side_counts[j, 0])
        cl = int(side_counts[j, 1])
        pts = [corners[j, 0], corners[j, 1]]
        for k in range(cr):
            pts.append(side_nodes[j, 0, k])
        pts.extend([corners[j, 2], corners[j, 3]])
        for k in range(cl):
            pts.append(side_nodes[j, 1, k])
        vj = np.zeros((m, 3), dtype=np.float64)
        for s in range(len(pts)):
            a = pts[s][None, :]
            b = pts[(s + 1) % len(pts)][None, :]
            vj += V.segment_velocity(P, a, b, rc_j)[:, 0, :]
        out[:, j, :] = vj
    return out


@pytest.mark.parametrize("rc_mode", ["scalar", "1d", "2d", "zero"])
def test_numpy_bent_ring_kernel_matches_segment_sum(rc_mode):
    """Numpy bent-ring velocity and normal influence match direct segment_velocity sums."""
    P, normals, corners, side_nodes, side_counts, rc_1d = _random_bent_rings(n=20, max_nodes=2, seed=73)
    m, n = P.shape[0], corners.shape[0]
    if rc_mode == "scalar":
        rc = 0.015
    elif rc_mode == "1d":
        rc = rc_1d
    elif rc_mode == "2d":
        rc = np.random.default_rng(79).uniform(1e-4, 5e-2, size=(m, n))
    else:
        rc = 0.0

    V.set_kernel_backend("numpy")
    got_v = VR.ring_velocity(P, corners, rc, side_nodes=side_nodes, side_counts=side_counts)
    got_a = VR.ring_normal_influence(P, normals, corners, rc, side_nodes=side_nodes, side_counts=side_counts)

    ref_v = _direct_poly_ring_velocity(P, corners, side_nodes, side_counts, rc)
    ref_a = np.sum(ref_v * normals[:, None, :], axis=2)

    assert got_v.shape == (m, n, 3)
    assert got_a.shape == (m, n)
    assert np.max(np.abs(got_v - ref_v)) <= 1e-13 * np.max(np.abs(ref_v))
    assert np.max(np.abs(got_a - ref_a)) <= 1e-13 * np.max(np.abs(ref_a))


@pytest.mark.parametrize("rc_mode", ["scalar", "1d", "2d", "zero"])
def test_numba_bent_ring_kernels_equal_numpy(rc_mode):
    """Numba bent-ring velocity and normal influence agree with numpy to round-off."""
    if not VR_NB.HAVE_NUMBA:
        pytest.skip("Numba is not installed.")
    P, normals, corners, side_nodes, side_counts, rc_1d = _random_bent_rings(n=24, max_nodes=2, seed=83)
    m, n = P.shape[0], corners.shape[0]
    if rc_mode == "scalar":
        rc = 0.012
    elif rc_mode == "1d":
        rc = rc_1d
    elif rc_mode == "2d":
        rc = np.random.default_rng(89).uniform(1e-4, 5e-2, size=(m, n))
    else:
        rc = 0.0

    out = {}
    for backend in ("numpy", "numba"):
        V.set_kernel_backend(backend)
        out[backend] = (
            VR.ring_velocity(P, corners, rc, side_nodes=side_nodes, side_counts=side_counts),
            VR.ring_normal_influence(P, normals, corners, rc, side_nodes=side_nodes, side_counts=side_counts),
        )
    for ref, new in zip(out["numpy"], out["numba"]):
        assert np.max(np.abs(new - ref)) <= 1e-12 * np.max(np.abs(ref))


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_straight_line_intermediate_nodes_equal_four_corner_ring(backend):
    """A ring with intermediate nodes on straight lines equals the four-corner ring to round-off."""
    if backend == "numba" and not VR_NB.HAVE_NUMBA:
        pytest.skip("Numba is not installed.")
    P, normals, corners, _, rc_1d = _random_rings(n=18, seed=97)
    n = corners.shape[0]

    side_counts = np.zeros((n, 2), dtype=np.int64)
    side_nodes = np.zeros((n, 2, 2, 3), dtype=np.float64)
    for j in range(n):
        cr = j % 3
        cl = (j + 1) % 3
        side_counts[j, 0] = cr
        side_counts[j, 1] = cl
        for k in range(cr):
            t = (k + 1) / (cr + 1)
            side_nodes[j, 0, k] = (1.0 - t) * corners[j, 1] + t * corners[j, 2]
        for k in range(cl):
            t = (k + 1) / (cl + 1)
            side_nodes[j, 1, k] = (1.0 - t) * corners[j, 3] + t * corners[j, 0]

    # Exact Biot-Savart (rc = 0): subdivided straight side segments equal single segments to round-off.
    v_quad = VR.ring_velocity(P, corners, 0.0, backend=backend)
    v_poly = VR.ring_velocity(P, corners, 0.0, side_nodes=side_nodes, side_counts=side_counts, backend=backend)
    a_quad = VR.ring_normal_influence(P, normals, corners, 0.0, backend=backend)
    a_poly = VR.ring_normal_influence(
        P, normals, corners, 0.0, side_nodes=side_nodes, side_counts=side_counts, backend=backend,
    )
    assert np.max(np.abs(v_poly - v_quad)) <= 1e-13 * np.max(np.abs(v_quad))
    assert np.max(np.abs(a_poly - a_quad)) <= 1e-13 * np.max(np.abs(a_quad))

    # Zero side counts take the four-corner fast path and match identically even with nonzero core.
    zero_counts = np.zeros((n, 2), dtype=np.int64)
    v_quad_rc = VR.ring_velocity(P, corners, rc_1d, backend=backend)
    v_zero_rc = VR.ring_velocity(P, corners, rc_1d, side_nodes=side_nodes, side_counts=zero_counts, backend=backend)
    assert np.array_equal(v_zero_rc, v_quad_rc)


@pytest.mark.parametrize("rc", [0.0, 0.01])
def test_bent_rings_shared_edge_cancellation_and_dipole_decay(rc):
    """Two adjacent bent rings cancel their shared bent side edge; one bent ring decays as 1/R^3."""
    r1 = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, -1.0, 0.0]])
    r2 = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    shared_mid = np.array([0.5, 0.0, 0.12])
    outer_l = np.array([0.5, -1.0, 0.12])
    outer_r = np.array([0.5, 1.0, 0.12])

    both = np.stack([r1, r2], axis=0)
    sn_both = np.zeros((2, 2, 1, 3), dtype=np.float64)
    sc_both = np.ones((2, 2), dtype=np.int64)
    sn_both[0, 0, 0] = shared_mid
    sn_both[0, 1, 0] = outer_l
    sn_both[1, 0, 0] = outer_r
    sn_both[1, 1, 0] = shared_mid

    outer = np.array([[[0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, -1.0, 0.0]]])
    sn_outer = np.zeros((1, 2, 1, 3), dtype=np.float64)
    sc_outer = np.ones((1, 2), dtype=np.int64)
    sn_outer[0, 0, 0] = outer_r
    sn_outer[0, 1, 0] = outer_l

    rng = np.random.default_rng(101)
    P = rng.normal(scale=1.5, size=(16, 3))
    P[0] = shared_mid + np.array([0.0, 1e-4, 1e-4])

    v_each = VR.ring_velocity(P, both, rc, side_nodes=sn_both, side_counts=sc_both)
    v_sum = v_each.sum(axis=1)
    scale = max(float(np.max(np.abs(v_each))), 1.0)
    if rc == 0.0:
        v_outer = VR.ring_velocity(P, outer, rc, side_nodes=sn_outer, side_counts=sc_outer)[:, 0, :]
        assert np.max(np.abs(v_sum - v_outer)) <= 1e-14 * scale
        assert np.max(np.abs(v_sum - v_outer)) <= 1e-12 * np.max(np.abs(v_outer))
    else:
        segs = [
            (r1[0], r1[1]), (r2[0], r2[1]),
            (r2[1], outer_r), (outer_r, r2[2]),
            (r2[2], r2[3]), (r1[2], r1[3]),
            (r1[3], outer_l), (outer_l, r1[0]),
        ]
        v_perim = sum(V.segment_velocity(P, a[None, :], b[None, :], rc)[:, 0, :] for a, b in segs)
        assert np.max(np.abs(v_sum - v_perim)) <= 1e-14 * scale
        assert np.max(np.abs(v_sum - v_perim)) <= 1e-12 * np.max(np.abs(v_perim))

    # Closed bent ring centred at the origin decays as 1/R^3 in the far field.
    outer_centered = np.array([[[-0.5, -0.5, 0.0], [-0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.5, -0.5, 0.0]]])
    sn_centered = np.array([[[[0.0, 0.5, 0.12]], [[0.0, -0.5, 0.12]]]], dtype=np.float64)
    radii = np.array([20.0, 40.0, 80.0, 160.0])
    direction = np.array([0.3, -0.4, 0.8660254])
    direction /= np.linalg.norm(direction)
    P_far = radii[:, None] * direction[None, :]
    mag = np.linalg.norm(
        VR.ring_velocity(P_far, outer_centered, rc, side_nodes=sn_centered, side_counts=sc_outer)[:, 0, :],
        axis=1,
    )
    ratios = mag[:-1] / mag[1:]
    assert np.all((ratios > 7.9) & (ratios < 8.1))


def test_bent_ring_chunking_and_validation(monkeypatch):
    """Chunking matches unchunked bent rings; invalid side_nodes or side_counts raise ValueError."""
    P, normals, corners, side_nodes, side_counts, rc = _random_bent_rings(n=14, max_nodes=2, seed=107)
    V.set_kernel_backend("numpy")
    v_big = VR.ring_velocity(P, corners, rc, side_nodes=side_nodes, side_counts=side_counts)
    a_big = VR.ring_normal_influence(P, normals, corners, rc, side_nodes=side_nodes, side_counts=side_counts)
    monkeypatch.setattr(VR, "_CHUNK_BUDGET", 400)
    v_small = VR.ring_velocity(P, corners, rc, side_nodes=side_nodes, side_counts=side_counts)
    a_small = VR.ring_normal_influence(P, normals, corners, rc, side_nodes=side_nodes, side_counts=side_counts)
    assert np.max(np.abs(v_small - v_big)) <= 1e-14 * np.max(np.abs(v_big))
    assert np.max(np.abs(a_small - a_big)) <= 1e-14 * np.max(np.abs(a_big))

    with pytest.raises(ValueError, match="side_nodes"):
        VR.ring_velocity(P, corners, rc, side_nodes=side_nodes)
    with pytest.raises(ValueError, match="side_nodes"):
        VR.ring_velocity(P, corners, rc, side_counts=side_counts)
    bad_counts = side_counts.copy()
    bad_counts[0, 0] = 3
    with pytest.raises(ValueError, match="side_counts"):
        VR.ring_velocity(P, corners, rc, side_nodes=side_nodes, side_counts=bad_counts)
    bad_neg = side_counts.copy()
    bad_neg[0, 1] = -1
    with pytest.raises(ValueError, match="side_counts"):
        VR.ring_normal_influence(P, normals, corners, rc, side_nodes=side_nodes, side_counts=bad_neg)
