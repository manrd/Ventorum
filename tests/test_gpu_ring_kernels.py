# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests of the GPU vortex-ring kernels and harmonic solver GPU path.

Covers Warp ring velocity, ring normal influence (four-corner and bent side
polylines), steady wake closure and far-wake tail against the CPU reference
kernels in float64 (round-off) and float32 (the float32 tolerance rule of
:mod:`tests.test_gpu_pipeline`), with and without ground images; GPU against
CPU parity for :func:`ventorum.unsteady.harmonic._solve_harmonic_rings` and
:func:`ventorum.dynamic_derivatives` (free air, ground effect ``h/c = 0.5`` and
a cambered wing); ``ventorum.gpu.release()`` freeing registered ring arrays;
and the CPU fallback note when CUDA is hidden in a new process.
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import ventorum as vt
from ventorum import gpu
from ventorum.aero import vortex as V
from ventorum.aero import vortex_ring as VR
from ventorum.aero.system import GroundPlane, make_ground_plane
from ventorum.geometry.lattice import build_lattice
from ventorum.unsteady.harmonic import (
    NOTE_GPU_OUT_OF_MEMORY,
    _HarmonicRingSystem,
    _solve_harmonic_rings,
)
from ventorum.unsteady.rings import build_ring_lattice, ground_image_rings
from ventorum.unsteady.tail import harmonic_tail_normal_influence, harmonic_tail_velocity

# Same tolerance rule as tests/test_gpu_pipeline.py.
TOL = {"float64": 1e-11, "float32": 2e-5}
ROOT = Path(__file__).resolve().parents[1]


def _need_gpu():
    if not gpu.available():
        pytest.skip(f"The GPU pipelines cannot run: {gpu.unavailable_reason()}")


@pytest.fixture
def gpu_device():
    """Run the test with the GPU device selected; restore settings afterwards."""
    _need_gpu()
    old_dev, old_prec = gpu.get_device(), gpu._precision
    yield
    gpu.release()
    gpu.set_device(old_dev)
    gpu.set_precision(old_prec)


def _random_rings(n: int, seed: int = 7, offset: float = 0.0, include_on_segment: bool = True):
    """Return random points, unit normals, 3D rings, wake directions, cores and ground."""
    rng = np.random.default_rng(seed)
    m = 37
    shift = np.array([offset, -0.5 * offset, 0.25 * offset], dtype=np.float64)
    P = rng.normal(scale=2.0, size=(m, 3)) + shift
    normals = rng.normal(size=(m, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    centers = rng.normal(scale=2.0, size=(n, 3)) + shift
    dx = rng.uniform(0.2, 0.8, size=(n, 1))
    dy = rng.uniform(0.2, 0.8, size=(n, 1))
    jitter = rng.normal(scale=0.05, size=(n, 4, 3))
    corners = np.empty((n, 4, 3), dtype=np.float64)
    corners[:, 0, :] = centers + np.hstack([-0.5 * dx, -0.5 * dy, np.zeros((n, 1))])
    corners[:, 1, :] = centers + np.hstack([-0.5 * dx, +0.5 * dy, np.zeros((n, 1))])
    corners[:, 2, :] = centers + np.hstack([+0.5 * dx, +0.5 * dy, np.zeros((n, 1))])
    corners[:, 3, :] = centers + np.hstack([+0.5 * dx, -0.5 * dy, np.zeros((n, 1))])
    corners += jitter

    # Points on extended lines and close to corners exercise both core branches
    # and float32 pair-difference rules. Points directly on a segment are
    # regular only with nonzero core radius (exact Biot-Savart has a 1/r singularity).
    P[0] = corners[0, 0] + 1.5 * (corners[0, 1] - corners[0, 0])
    P[1] = 0.5 * (corners[1, 0] + corners[1, 1]) + np.array([0.0, 0.0, 1e-4])
    P[2] = corners[2, 2] + 1e-5 * np.array([1.0, -1.0, 0.5])
    if include_on_segment:
        P[3] = 0.5 * (corners[3, 0] + corners[3, 1])
        P[4] = 0.5 * (corners[3, 1] + corners[3, 2])

    wake_dir = rng.normal(size=(n, 3))
    wake_dir[:, 0] = np.abs(wake_dir[:, 0]) + 1.0
    wake_dir /= np.linalg.norm(wake_dir, axis=1, keepdims=True)
    rc = rng.uniform(1e-4, 5e-2, size=n)

    gk = np.array([0.05, -0.02, 1.0], dtype=np.float64)
    gk /= np.linalg.norm(gk)
    go = float(np.min(corners.reshape(-1, 3) @ gk) - 1.5)
    ground = GroundPlane(normal=gk, offset=go)
    return P, normals, corners, wake_dir, rc, ground


def _random_bent_rings(
    n: int,
    max_nodes: int = 2,
    seed: int = 71,
    offset: float = 0.0,
    include_on_segment: bool = True,
):
    """Return random points, normals, 3D rings, bent side nodes, side counts, cores and ground."""
    P, normals, corners, _, rc, ground = _random_rings(
        n=n, seed=seed, offset=offset, include_on_segment=include_on_segment,
    )
    rng = np.random.default_rng(seed + 100)
    side_counts = np.zeros((n, 2), dtype=np.int64)
    side_nodes = np.zeros((n, 2, max_nodes, 3), dtype=np.float64)

    for j in range(n):
        cr = int(rng.integers(0, max_nodes + 1))
        cl = int(rng.integers(0, max_nodes + 1))
        side_counts[j, 0] = cr
        side_counts[j, 1] = cl
        for k in range(cr):
            frac = (k + 1) / (cr + 1)
            base = (1.0 - frac) * corners[j, 1] + frac * corners[j, 2]
            side_nodes[j, 0, k] = base + rng.normal(scale=0.06, size=3)
        for k in range(cr, max_nodes):
            side_nodes[j, 0, k] = corners[j, 2]
        for k in range(cl):
            frac = (k + 1) / (cl + 1)
            base = (1.0 - frac) * corners[j, 3] + frac * corners[j, 0]
            side_nodes[j, 1, k] = base + rng.normal(scale=0.06, size=3)
        for k in range(cl, max_nodes):
            side_nodes[j, 1, k] = corners[j, 0]

    if n >= 3 and max_nodes >= 2:
        side_counts[0] = (0, 0)
        side_counts[1] = (1, 0)
        side_nodes[1, 0, 0] = 0.5 * (corners[1, 1] + corners[1, 2]) + np.array([0.0, 0.0, 0.05])
        side_counts[2] = (2, 2)
        side_nodes[2, 0, 0] = (2.0 * corners[2, 1] + corners[2, 2]) / 3.0 + np.array([0.0, 0.0, 0.04])
        side_nodes[2, 0, 1] = (corners[2, 1] + 2.0 * corners[2, 2]) / 3.0 + np.array([0.0, 0.0, 0.03])
        side_nodes[2, 1, 0] = (2.0 * corners[2, 3] + corners[2, 0]) / 3.0 + np.array([0.0, 0.0, 0.03])
        side_nodes[2, 1, 1] = (corners[2, 3] + 2.0 * corners[2, 0]) / 3.0 + np.array([0.0, 0.0, 0.04])
        P[5] = side_nodes[2, 0, 0] + np.array([0.0, 0.0, 1e-4])
        if include_on_segment:
            P[6] = 0.5 * (corners[2, 1] + side_nodes[2, 0, 0])

    return P, normals, corners, side_nodes, side_counts, rc, ground


def _make_rc(rc_mode: str, rc_1d: np.ndarray, m: int, n: int, seed: int = 19):
    if rc_mode == "scalar":
        return 0.015
    if rc_mode == "1d":
        return rc_1d
    if rc_mode == "2d":
        return np.random.default_rng(seed).uniform(1e-4, 5e-2, size=(m, n))
    return 0.0


def _rect_wing(semi_span: float = 3.0, chord: float = 1.0, n_panels: int = 6) -> vt.Aircraft:
    wing = vt.LiftingSurface(
        name="wing",
        semi_span=semi_span,
        n_panels=n_panels,
        sections=[
            vt.WingSection(y_frac=0.0, chord=chord),
            vt.WingSection(y_frac=1.0, chord=chord),
        ],
    )
    ac = vt.Aircraft(surfaces=[wing], ref_point=np.array([0.25 * chord, 0.0, 0.0]))
    ac.compute_reference_values()
    return ac


def _cambered_naca4412_wing(semi_span: float = 3.0, chord: float = 1.0, n_panels: int = 6) -> vt.Aircraft:
    wing = vt.LiftingSurface(
        name="cambered_wing",
        semi_span=semi_span,
        n_panels=n_panels,
        sections=[
            vt.WingSection(y_frac=0.0, chord=chord, shape="4412"),
            vt.WingSection(y_frac=1.0, chord=chord, shape="4412"),
        ],
    )
    ac = vt.Aircraft(surfaces=[wing], ref_point=np.array([0.25 * chord, 0.0, 0.0]))
    ac.compute_reference_values()
    return ac


# ---------------------------------------------------------------------------
# 1. Warp ring and steady-wake kernels against numpy reference
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize("precision", ["float64", "float32"])
@pytest.mark.parametrize("rc_mode", ["scalar", "1d", "2d", "zero"])
@pytest.mark.parametrize("use_ground", [False, True])
def test_warp_four_corner_ring_equals_numpy(gpu_device, precision, rc_mode, use_ground):
    """Warp four-corner ring velocity and normal influence match numpy reference."""
    from ventorum.gpu import ring_kernels as RK

    P, normals, corners, _, rc_1d, ground = _random_rings(
        n=25, seed=11, offset=50.0, include_on_segment=rc_mode != "zero",
    )
    m, n = P.shape[0], corners.shape[0]
    rc = _make_rc(rc_mode, rc_1d, m, n, seed=19)

    old_be = V._kernel_backend
    try:
        V.set_kernel_backend("numpy")
        ref_v = VR.ring_velocity(P, corners, rc, backend="numpy")
        ref_a = VR.ring_normal_influence(P, normals, corners, rc, backend="numpy")
        if use_ground:
            img_c = ground.reflect_points(corners)
            ref_v_img = -VR.ring_velocity(P, img_c, rc, backend="numpy")
            ref_a_img = -VR.ring_normal_influence(P, normals, img_c, rc, backend="numpy")
            ref_v = ref_v + ref_v_img
            ref_a = ref_a + ref_a_img
    finally:
        V._kernel_backend = old_be

    g = ground if use_ground else None
    got_v = RK.ring_velocity_gpu(P, corners, rc, ground=g, precision=precision)
    got_a = RK.ring_normal_influence_gpu(P, normals, corners, rc, ground=g, precision=precision)

    tol = TOL[precision]
    assert got_v.shape == (m, n, 3)
    assert got_a.shape == (m, n)
    assert np.all(np.isfinite(got_v)) and np.all(np.isfinite(got_a))
    assert np.max(np.abs(got_v - ref_v)) <= tol * max(float(np.max(np.abs(ref_v))), 1.0)
    assert np.max(np.abs(got_a - ref_a)) <= tol * max(float(np.max(np.abs(ref_a))), 1.0)

    if use_ground:
        got_v_img = RK.ring_velocity_gpu(P, corners, rc, ground=g, img_mode=2, precision=precision)
        got_a_img = RK.ring_normal_influence_gpu(P, normals, corners, rc, ground=g, img_mode=2, precision=precision)
        assert np.max(np.abs(got_v_img - ref_v_img)) <= tol * max(float(np.max(np.abs(ref_v_img))), 1.0)
        assert np.max(np.abs(got_a_img - ref_a_img)) <= tol * max(float(np.max(np.abs(ref_a_img))), 1.0)


@pytest.mark.gpu
@pytest.mark.parametrize("precision", ["float64", "float32"])
@pytest.mark.parametrize("rc_mode", ["scalar", "1d", "2d", "zero"])
@pytest.mark.parametrize("use_ground", [False, True])
def test_warp_bent_ring_equals_numpy(gpu_device, precision, rc_mode, use_ground):
    """Warp bent-side-polyline ring kernels match numpy reference with and without ground images."""
    from ventorum.gpu import ring_kernels as RK

    P, normals, corners, sn, sc, rc_1d, ground = _random_bent_rings(
        n=22, max_nodes=2, seed=73, offset=40.0, include_on_segment=rc_mode != "zero",
    )
    m, n = P.shape[0], corners.shape[0]
    rc = _make_rc(rc_mode, rc_1d, m, n, seed=79)

    ref_v = VR.ring_velocity(P, corners, rc, side_nodes=sn, side_counts=sc, backend="numpy")
    ref_a = VR.ring_normal_influence(P, normals, corners, rc, side_nodes=sn, side_counts=sc, backend="numpy")
    if use_ground:
        img_c = ground.reflect_points(corners)
        img_sn = ground.reflect_points(sn)
        ref_v = ref_v - VR.ring_velocity(P, img_c, rc, side_nodes=img_sn, side_counts=sc, backend="numpy")
        ref_a = ref_a - VR.ring_normal_influence(
            P, normals, img_c, rc, side_nodes=img_sn, side_counts=sc, backend="numpy",
        )

    g = ground if use_ground else None
    got_v = RK.ring_velocity_gpu(P, corners, rc, side_nodes=sn, side_counts=sc, ground=g, precision=precision)
    got_a = RK.ring_normal_influence_gpu(
        P, normals, corners, rc, side_nodes=sn, side_counts=sc, ground=g, precision=precision,
    )

    tol = TOL[precision]
    assert got_v.shape == (m, n, 3)
    assert got_a.shape == (m, n)
    assert np.all(np.isfinite(got_v)) and np.all(np.isfinite(got_a))
    assert np.max(np.abs(got_v - ref_v)) <= tol * max(float(np.max(np.abs(ref_v))), 1.0)
    assert np.max(np.abs(got_a - ref_a)) <= tol * max(float(np.max(np.abs(ref_a))), 1.0)


@pytest.mark.gpu
@pytest.mark.parametrize("precision", ["float64", "float32"])
@pytest.mark.parametrize("dir_mode", ["1d", "2d"])
@pytest.mark.parametrize("rc_mode", ["scalar", "1d", "2d", "zero"])
@pytest.mark.parametrize("use_ground", [False, True])
def test_warp_steady_wake_equals_numpy(gpu_device, precision, dir_mode, rc_mode, use_ground):
    """Warp steady-wake closure velocity and normal influence match numpy reference."""
    from ventorum.gpu import ring_kernels as RK

    P, normals, corners, wd_2d, rc_1d, ground = _random_rings(n=18, seed=23, offset=30.0)
    m, n = P.shape[0], corners.shape[0]
    te_left = corners[:, 3]
    te_right = corners[:, 2]
    if dir_mode == "1d":
        wake_dir = np.array([np.cos(0.1), 0.0, np.sin(0.1)], dtype=np.float64)
    else:
        wake_dir = wd_2d
    rc = _make_rc(rc_mode, rc_1d, m, n, seed=29)

    ref_v = VR.steady_wake_velocity(P, te_left, te_right, wake_dir, rc, backend="numpy")
    ref_a = VR.steady_wake_normal_influence(P, normals, te_left, te_right, wake_dir, rc, backend="numpy")
    if use_ground:
        img_tl = ground.reflect_points(te_left)
        img_tr = ground.reflect_points(te_right)
        img_wd = ground.reflect_directions(wake_dir)
        ref_v = ref_v - VR.steady_wake_velocity(P, img_tl, img_tr, img_wd, rc, backend="numpy")
        ref_a = ref_a - VR.steady_wake_normal_influence(P, normals, img_tl, img_tr, img_wd, rc, backend="numpy")

    g = ground if use_ground else None
    got_v = RK.steady_wake_velocity_gpu(P, te_left, te_right, wake_dir, rc, ground=g, precision=precision)
    got_a = RK.steady_wake_normal_influence_gpu(
        P, normals, te_left, te_right, wake_dir, rc, ground=g, precision=precision,
    )

    tol = TOL[precision]
    assert got_v.shape == (m, n, 3)
    assert got_a.shape == (m, n)
    assert np.all(np.isfinite(got_v)) and np.all(np.isfinite(got_a))
    assert np.max(np.abs(got_v - ref_v)) <= tol * max(float(np.max(np.abs(ref_v))), 1.0)
    assert np.max(np.abs(got_a - ref_a)) <= tol * max(float(np.max(np.abs(ref_a))), 1.0)


@pytest.mark.gpu
@pytest.mark.parametrize("use_ground", [False, True])
@pytest.mark.parametrize("kappa", [0.0, 0.002, 0.25, 1.5])
def test_gpu_harmonic_tail_equals_cpu(gpu_device, use_ground, kappa):
    """GPU far-wake harmonic tail matches CPU tail to round-off, upstream and downstream."""
    from ventorum.gpu import ring_kernels as RK

    ac = _rect_wing(n_panels=6)
    lat = build_lattice(ac, n_chord=4, chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), h=0.5 if use_ground else None)
    ground = (
        make_ground_plane(lat, 0.5, cond.alpha, cond.beta, ref_point=ac.ref_point, height_ref="ref")
        if use_ground
        else None
    )
    rings = build_ring_lattice(lat, wake_dir=np.array([np.cos(cond.alpha), 0.0, np.sin(cond.alpha)]))
    tail_l = rings.te_left + 3.0 * rings.wake_dir
    tail_r = rings.te_right + 3.0 * rings.wake_dir
    rc_w = rings.rc[rings.te_indices]

    # Include evaluation points both upstream (a_min > 0) and aft (a_min <= 0) of the tail start.
    P = np.vstack([rings.cp, tail_l[:2] + 0.5 * rings.wake_dir[:2] + np.array([[0.0, 0.1, 0.2], [0.0, -0.1, 0.3]])])
    normals = np.vstack([rings.normal_bc, np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])])

    ref_v = harmonic_tail_velocity(P, tail_l, tail_r, rings.wake_dir, kappa, rc_w)
    ref_a = harmonic_tail_normal_influence(P, normals, tail_l, tail_r, rings.wake_dir, kappa, rc_w)
    if use_ground:
        img = ground_image_rings(rings, ground)
        img_tl = img.te_left + 3.0 * img.wake_dir
        img_tr = img.te_right + 3.0 * img.wake_dir
        ref_v = ref_v + img.sign * harmonic_tail_velocity(P, img_tl, img_tr, img.wake_dir, kappa, rc_w)
        ref_a = ref_a + img.sign * harmonic_tail_normal_influence(P, normals, img_tl, img_tr, img.wake_dir, kappa, rc_w)

    got_v = RK.harmonic_tail_velocity_gpu(P, tail_l, tail_r, rings.wake_dir, kappa, rc_w, ground=ground)
    got_a = RK.harmonic_tail_normal_influence_gpu(P, normals, tail_l, tail_r, rings.wake_dir, kappa, rc_w, ground=ground)

    assert np.max(np.abs(got_v - ref_v)) <= 1e-12 * max(float(np.max(np.abs(ref_v))), 1.0)
    assert np.max(np.abs(got_a - ref_a)) <= 1e-12 * max(float(np.max(np.abs(ref_a))), 1.0)


# ---------------------------------------------------------------------------
# 2. Harmonic solver and dynamic_derivatives parity (GPU vs CPU)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize("precision", ["float64", "float32"])
@pytest.mark.parametrize("h_ground", [None, 0.5])
@pytest.mark.parametrize("cambered", [False, True])
def test_gpu_harmonic_response_equals_cpu(gpu_device, precision, h_ground, cambered):
    """Harmonic ring responses on the GPU match the CPU reference in free air, h/c=0.5 and camber."""
    ac = _cambered_naca4412_wing(n_panels=6) if cambered else _rect_wing(n_panels=6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lat = build_lattice(ac, n_chord=4, chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=30.0, rho=1.225, alpha=np.radians(4.0), h=h_ground)
    ground = (
        make_ground_plane(lat, h_ground, cond.alpha, cond.beta, ref_point=ac.ref_point, height_ref="ref")
        if h_ground is not None
        else None
    )
    k_vals = [0.0, 0.05, 0.2, 0.5]

    gpu.set_device("cpu")
    ref = _solve_harmonic_rings(
        lat, cond, k_vals, S_ref=ac.S_ref, b_ref=ac.b_ref, c_ref=ac.c_ref, ground=ground, ref_point=ac.ref_point,
    )
    assert ref.device == "cpu"
    assert ref.precision == "float64"

    gpu.set_device("gpu")
    gpu.set_precision(precision)
    got = _solve_harmonic_rings(
        lat, cond, k_vals, S_ref=ac.S_ref, b_ref=ac.b_ref, c_ref=ac.c_ref, ground=ground, ref_point=ac.ref_point,
    )
    assert got.device == "gpu"
    assert got.precision == precision

    tol = TOL[precision]
    cl_scale = max(float(np.max(np.abs(ref.pitch.CL))), 1.0)
    for motion in ("heave", "pitch"):
        rm = getattr(ref, motion)
        gm = getattr(got, motion)
        for name in ("CL", "CD", "CY", "Cl", "Cm", "Cn"):
            a = getattr(rm, name)
            b = getattr(gm, name)
            scale = max(float(np.max(np.abs(a))), cl_scale)
            assert np.max(np.abs(a - b)) <= tol * scale, (motion, name, np.max(np.abs(a - b)) / scale)
        s_scale = max(float(np.max(np.abs(rm.strip_cl))), cl_scale)
        assert np.max(np.abs(rm.strip_cl - gm.strip_cl)) <= 5 * tol * s_scale

    for attr in ("k0_heave", "k0_pitch", "k0_pitch_rate", "mean_state"):
        rs = getattr(ref, attr)
        gs = getattr(got, attr)
        for name in ("CL", "CD", "CY", "Cl", "Cm", "Cn"):
            va = getattr(rs, name)
            vb = getattr(gs, name)
            scale = max(abs(va), cl_scale)
            assert abs(va - vb) <= tol * scale, (attr, name, abs(va - vb) / scale)


@pytest.mark.gpu
def test_gpu_harmonic_woodbury_equals_direct(gpu_device):
    """GPU batched Woodbury solve equals GPU batched direct complex LU to round-off."""
    ac = _rect_wing(n_panels=6)
    lat = build_lattice(ac, n_chord=4, chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=30.0, rho=1.225, alpha=np.radians(4.0), h=0.5)
    ground = make_ground_plane(lat, 0.5, cond.alpha, cond.beta, ref_point=ac.ref_point, height_ref="ref")
    k_vals = [0.0, 0.05, 0.2, 0.5]

    gpu.set_device("gpu")
    gpu.set_precision("float64")
    r_wb = _solve_harmonic_rings(
        lat, cond, k_vals, S_ref=ac.S_ref, b_ref=ac.b_ref, c_ref=ac.c_ref,
        ground=ground, ref_point=ac.ref_point, solve_method="woodbury",
    )
    r_dir = _solve_harmonic_rings(
        lat, cond, k_vals, S_ref=ac.S_ref, b_ref=ac.b_ref, c_ref=ac.c_ref,
        ground=ground, ref_point=ac.ref_point, solve_method="direct",
    )
    assert r_wb.device == "gpu" and r_dir.device == "gpu"
    for motion in ("heave", "pitch"):
        for name in ("CL", "CD", "Cm"):
            a = getattr(getattr(r_wb, motion), name)
            b = getattr(getattr(r_dir, motion), name)
            assert np.max(np.abs(a - b)) <= 1e-12 * max(float(np.max(np.abs(b))), 1.0)


@pytest.mark.gpu
@pytest.mark.parametrize("precision", ["float64", "float32"])
@pytest.mark.parametrize("h_ground", [None, 0.5])
@pytest.mark.parametrize("cambered", [False, True])
def test_gpu_dynamic_derivatives_equals_cpu(gpu_device, precision, h_ground, cambered):
    """dynamic_derivatives on GPU agrees with CPU in free air, h/c = 0.5 and on a cambered wing."""
    ac = _cambered_naca4412_wing(n_panels=6) if cambered else _rect_wing(n_panels=6)
    st = vt.SolverSettings(solver_type="vlm", n_panels=6, n_chord=8, spacing="half-cosine", chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=30.0, rho=1.225, alpha=np.radians(4.0), h=h_ground)
    k_vals = (0.0, 0.05, 0.2, 0.5)

    gpu.set_device("cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = vt.dynamic_derivatives(ac, cond, st, k=k_vals)
    assert ref.device == "cpu"
    assert ref.precision == "float64"

    gpu.set_device("gpu")
    gpu.set_precision(precision)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = vt.dynamic_derivatives(ac, cond, st, k=k_vals)
    assert got.device == "gpu"
    assert got.precision == precision
    d = got.to_dict()
    assert d["device"] == "gpu"
    assert d["precision"] == precision

    tol = TOL[precision]
    base_scale = max(abs(ref.k0["CL_alpha"]), 1.0)
    for key in ref.k0:
        scale = max(abs(ref.k0[key]), base_scale)
        assert abs(ref.k0[key] - got.k0[key]) <= tol * scale, (key, abs(ref.k0[key] - got.k0[key]) / scale)

    for key in ref.derivatives:
        a = ref.derivatives[key]
        b = got.derivatives[key]
        scale = max(float(np.max(np.abs(a))), base_scale)
        assert np.max(np.abs(a - b)) <= tol * scale, (key, np.max(np.abs(a - b)) / scale)

    for attr in ("H_theta", "H_z", "H_pureq"):
        rm = getattr(ref, attr)
        gm = getattr(got, attr)
        for coeff in ("CL", "CD", "CY", "Cl", "Cm", "Cn"):
            scale = max(float(np.max(np.abs(rm[coeff]))), base_scale)
            assert np.max(np.abs(rm[coeff] - gm[coeff])) <= tol * scale


# ---------------------------------------------------------------------------
# 3. Memory management: gpu.release() frees registered ring arrays
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_gpu_release_frees_ring_arrays(gpu_device, precision):
    """gpu.release() frees device arrays held by the ring kernels and harmonic system."""
    import torch

    from ventorum.gpu import ring_kernels as RK

    gpu.set_device("gpu")
    gpu.set_precision(precision)
    gpu.release()
    gc.collect()
    assert RK.device_bytes() == 0

    ac = _cambered_naca4412_wing(n_panels=8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lat = build_lattice(ac, n_chord=4, chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=30.0, rho=1.225, alpha=np.radians(4.0), h=0.5)
    ground = make_ground_plane(lat, 0.5, cond.alpha, cond.beta, ref_point=ac.ref_point, height_ref="ref")

    system = _HarmonicRingSystem(
        lat, cond, S_ref=ac.S_ref, b_ref=ac.b_ref, c_ref=ac.c_ref, ground=ground, ref_point=ac.ref_point,
    )
    assert system.device == "gpu"
    ref_resp = system.solve([0.0, 0.1, 0.3])

    held = RK.device_bytes()
    assert held > 0
    allocated = torch.cuda.memory_allocated()
    freed = gpu.release()
    gc.collect()
    assert freed >= held
    assert RK.device_bytes() == 0
    assert torch.cuda.memory_allocated() <= allocated - held

    # Solving again after release still works cleanly and reproduces the response.
    after_resp = system.solve([0.0, 0.1, 0.3])
    assert np.max(np.abs(after_resp.pitch.CL - ref_resp.pitch.CL)) <= 1e-12 * max(
        float(np.max(np.abs(ref_resp.pitch.CL))), 1.0,
    )


# ---------------------------------------------------------------------------
# 4. Out-of-memory fallback on GPU and CUDA-hidden fallback in a subprocess
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_gpu_harmonic_oom_fallback_adds_note(gpu_device, monkeypatch):
    """Two GPU out-of-memory errors fall back to CPU with one warning and the fallback note."""
    import torch

    from ventorum.gpu import ring_kernels as RK

    ac = _rect_wing(n_panels=6)
    st = vt.SolverSettings(solver_type="vlm", n_panels=6, n_chord=4, chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=30.0, rho=1.225, alpha=np.radians(4.0))

    gpu.set_device("cpu")
    ref = vt.dynamic_derivatives(ac, cond, st, k=(0.0, 0.1))

    oom_cls = getattr(torch, "OutOfMemoryError", None) or torch.cuda.OutOfMemoryError

    def raise_oom(*args, **kwargs):
        raise oom_cls("CUDA out of memory. Tried to allocate 20.00 GiB (test)")

    gpu.set_device("gpu")
    monkeypatch.setattr(RK, "ring_normal_influence_gpu", raise_oom)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = vt.dynamic_derivatives(ac, cond, st, k=(0.0, 0.1))

    assert any("out of memory" in str(w.message).lower() for w in caught)
    assert got.device == "cpu"
    assert got.precision == "float64"
    assert NOTE_GPU_OUT_OF_MEMORY in got.notes
    assert got.trust is not None and NOTE_GPU_OUT_OF_MEMORY in got.trust.warnings
    assert got.trust.rating == ref.trust.rating
    assert got.trust.score == ref.trust.score
    assert float(got.k0["CL_alpha"]).hex() == float(ref.k0["CL_alpha"]).hex()


@pytest.mark.slow
def test_cuda_hidden_cpu_fallback_note_subprocess():
    """With CUDA_VISIBLE_DEVICES=-1 and VENTORUM_DEVICE=gpu, dynamic_derivatives falls back to CPU with a note."""
    script = """
import json, warnings
import numpy as np
import ventorum as vt
from ventorum import gpu

wing = vt.LiftingSurface(
    name="wing",
    semi_span=3.0,
    n_panels=6,
    sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=1.0)],
)
ac = vt.Aircraft(surfaces=[wing], ref_point=np.array([0.25, 0.0, 0.0]))
cond = vt.FlightCondition(V_inf=30.0, rho=1.225, alpha=np.radians(4.0))
st = vt.SolverSettings(solver_type="vlm", n_panels=6, n_chord=8, chord_spacing="uniform")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    res = vt.dynamic_derivatives(ac, cond, st, k=(0.0, 0.1))

print(json.dumps({
    "gpu_available": gpu.available(),
    "gpu_setting": gpu.get_device(),
    "device": res.device,
    "precision": res.precision,
    "notes": res.notes,
    "trust_warnings": [] if res.trust is None else list(res.trust.warnings),
    "CL_alpha": res.k0["CL_alpha"],
    "Cm_q": res.k0["Cm_q"],
}))
"""
    base_env = os.environ.copy()
    base_env["PYTHONPATH"] = str(ROOT)
    base_env["MPLBACKEND"] = "Agg"
    base_env["NUMBA_NUM_THREADS"] = "4"
    base_env["OMP_NUM_THREADS"] = "4"
    base_env["MKL_NUM_THREADS"] = "4"
    base_env["OPENBLAS_NUM_THREADS"] = "4"
    base_env["CUDA_VISIBLE_DEVICES"] = "-1"

    env_gpu = dict(base_env, VENTORUM_DEVICE="gpu")
    proc_gpu = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(ROOT),
        env=env_gpu,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    out_gpu = json.loads(proc_gpu.stdout.strip().splitlines()[-1])
    assert out_gpu["gpu_available"] is False
    assert out_gpu["gpu_setting"] == "gpu"
    assert out_gpu["device"] == "cpu"
    assert out_gpu["precision"] == "float64"
    assert any("GPU unavailable" in n and "CPU" in n for n in out_gpu["notes"])
    assert any("GPU unavailable" in w and "CPU" in w for w in out_gpu["trust_warnings"])

    env_cpu = dict(base_env, VENTORUM_DEVICE="cpu")
    proc_cpu = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(ROOT),
        env=env_cpu,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    out_cpu = json.loads(proc_cpu.stdout.strip().splitlines()[-1])
    assert out_cpu["device"] == "cpu"
    assert out_cpu["precision"] == "float64"
    assert not any("GPU unavailable" in n for n in out_cpu["notes"])
    assert out_gpu["CL_alpha"] == pytest.approx(out_cpu["CL_alpha"], rel=1e-14, abs=1e-14)
    assert out_gpu["Cm_q"] == pytest.approx(out_cpu["Cm_q"], rel=1e-14, abs=1e-14)


@pytest.mark.gpu
@pytest.mark.parametrize("use_ground", [False, True])
@pytest.mark.parametrize("kappa", [0.0, 0.002, 0.25, 1.5])
def test_gpu_displaced_steady_tail_equals_cpu(gpu_device, use_ground, kappa):
    """GPU displaced steady far-wake tail matches CPU to round-off, upstream and downstream."""
    from ventorum.gpu import ring_kernels as RK
    from ventorum.unsteady.harmonic import _ey_cross
    from ventorum.unsteady.tail import (
        displaced_steady_tail_normal_influence,
        displaced_steady_tail_velocity,
    )

    ac = _rect_wing(n_panels=6)
    lat = build_lattice(ac, n_chord=4, chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), h=0.5 if use_ground else None)
    ground = (
        make_ground_plane(lat, 0.5, cond.alpha, cond.beta, ref_point=ac.ref_point, height_ref="ref")
        if use_ground
        else None
    )
    rings = build_ring_lattice(lat, wake_dir=np.array([np.cos(cond.alpha), 0.0, np.sin(cond.alpha)]))
    wake_length = 3.0
    tail_l = rings.te_left + wake_length * rings.wake_dir
    tail_r = rings.te_right + wake_length * rings.wake_dir
    rc_w = rings.rc[rings.te_indices]

    # Non-uniform left/right pitch-like displacement vectors plus upstream and downstream points.
    u_l = _ey_cross(rings.te_left - ac.ref_point[None, :])
    u_r = _ey_cross(rings.te_right - ac.ref_point[None, :])
    P = np.vstack([rings.cp, tail_l[:2] + 0.5 * rings.wake_dir[:2] + np.array([[0.0, 0.1, 0.2], [0.0, -0.1, 0.3]])])
    normals = np.vstack([rings.normal_bc, np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])])

    ref_v = displaced_steady_tail_velocity(
        P, tail_l, tail_r, rings.wake_dir, u_l, u_r, kappa, wake_length, rc_w,
    )
    ref_a = displaced_steady_tail_normal_influence(
        P, normals, tail_l, tail_r, rings.wake_dir, u_l, u_r, kappa, wake_length, rc_w,
    )
    if use_ground:
        img = ground_image_rings(rings, ground)
        img_tl = img.te_left + wake_length * img.wake_dir
        img_tr = img.te_right + wake_length * img.wake_dir
        u_l_img = ground.reflect_directions(u_l)
        u_r_img = ground.reflect_directions(u_r)
        ref_v = ref_v + img.sign * displaced_steady_tail_velocity(
            P, img_tl, img_tr, img.wake_dir, u_l_img, u_r_img, kappa, wake_length, rc_w,
        )
        ref_a = ref_a + img.sign * displaced_steady_tail_normal_influence(
            P, normals, img_tl, img_tr, img.wake_dir, u_l_img, u_r_img, kappa, wake_length, rc_w,
        )

    got_v = RK.displaced_steady_tail_velocity_gpu(
        P, tail_l, tail_r, rings.wake_dir, u_l, u_r, kappa, wake_length, rc_w, ground=ground,
    )
    got_a = RK.displaced_steady_tail_normal_influence_gpu(
        P, normals, tail_l, tail_r, rings.wake_dir, u_l, u_r, kappa, wake_length, rc_w, ground=ground,
    )

    assert np.max(np.abs(got_v - ref_v)) <= 1e-12 * max(float(np.max(np.abs(ref_v))), 1.0)
    assert np.max(np.abs(got_a - ref_a)) <= 1e-12 * max(float(np.max(np.abs(ref_a))), 1.0)


def _flapped_wing_gpu(*, cambered: bool = False, n_panels: int = 6) -> vt.Aircraft:
    """Build a rectangular wing (flat or NACA 4412) with a full-span chordwise flap."""
    fl = vt.ControlSurface(name="flap", eta_start=0.0, eta_end=1.0, hinge_x_c=0.75, deflection=0.0)
    shape = "4412" if cambered else None
    wing = vt.LiftingSurface(
        name="flapped_wing",
        semi_span=3.0,
        n_panels=n_panels,
        controls=[fl],
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.0, shape=shape),
            vt.WingSection(y_frac=1.0, chord=1.0, shape=shape),
        ],
    )
    ac = vt.Aircraft(surfaces=[wing], ref_point=np.array([0.25, 0.0, 0.0]))
    ac.compute_reference_values()
    return ac


@pytest.mark.gpu
@pytest.mark.parametrize("precision", ["float64", "float32"])
@pytest.mark.parametrize("h_ground", [None, 0.5])
@pytest.mark.parametrize("cambered", [False, True])
def test_gpu_harmonic_flap_mode_equals_cpu(gpu_device, precision, h_ground, cambered):
    """GPU harmonic flap mode agrees with CPU in float64 and float32 (free air, h/c = 0.5 and camber)."""
    ac = _flapped_wing_gpu(cambered=cambered, n_panels=6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lat = build_lattice(ac, n_chord=4, chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=30.0, rho=1.225, alpha=np.radians(4.0), h=h_ground)
    ground = (
        make_ground_plane(lat, h_ground, cond.alpha, cond.beta, ref_point=ac.ref_point, height_ref="ref")
        if h_ground is not None
        else None
    )
    k_vals = [0.0, 0.05, 0.2, 0.5]

    gpu.set_device("cpu")
    ref = _solve_harmonic_rings(
        lat, cond, k_vals, S_ref=ac.S_ref, b_ref=ac.b_ref, c_ref=ac.c_ref,
        ground=ground, ref_point=ac.ref_point, controls=("flap",),
    )

    gpu.set_device("gpu")
    gpu.set_precision(precision)
    sys_gpu = _HarmonicRingSystem(
        lat, cond, S_ref=ac.S_ref, b_ref=ac.b_ref, c_ref=ac.c_ref,
        ground=ground, ref_point=ac.ref_point, controls=("flap",),
    )
    got = sys_gpu.solve(k_vals)
    assert got.device == "gpu"
    assert got.precision == precision

    tol = TOL[precision]
    cl_scale = max(float(np.max(np.abs(ref.controls["flap"].CL))), 1.0)
    rf = ref.controls["flap"]
    gf = got.controls["flap"]
    for name in ("CL", "CD", "CY", "Cl", "Cm", "Cn"):
        a = getattr(rf, name)
        b = getattr(gf, name)
        scale = max(float(np.max(np.abs(a))), cl_scale)
        assert np.max(np.abs(a - b)) <= tol * scale, (name, np.max(np.abs(a - b)) / scale)
    s_scale = max(float(np.max(np.abs(rf.strip_cl))), cl_scale)
    assert np.max(np.abs(rf.strip_cl - gf.strip_cl)) <= 5 * tol * s_scale

    rk0 = ref.k0_controls["flap"]
    gk0 = got.k0_controls["flap"]
    for name in ("CL", "CD", "CY", "Cl", "Cm", "Cn"):
        va = getattr(rk0, name)
        vb = getattr(gk0, name)
        scale = max(abs(va), cl_scale)
        assert abs(va - vb) <= tol * scale, ("k0", name, abs(va - vb) / scale)

    # Re-solving after gpu.release() reproduces the flap mode on the GPU.
    gpu.release()
    after = sys_gpu.solve(k_vals)
    assert np.max(np.abs(after.controls["flap"].CL - gf.CL)) <= 1e-12 * cl_scale


@pytest.mark.gpu
@pytest.mark.parametrize("precision", ["float64", "float32"])
@pytest.mark.parametrize("h_ground", [None, 0.5])
def test_gpu_dynamic_derivatives_controls_equals_cpu(gpu_device, precision, h_ground):
    """dynamic_derivatives(..., controls=('flap',)) on GPU agrees with CPU in float64 and float32."""
    ac = _flapped_wing_gpu(cambered=False, n_panels=6)
    st = vt.SolverSettings(solver_type="vlm", n_panels=6, n_chord=8, spacing="half-cosine", chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=30.0, rho=1.225, alpha=np.radians(4.0), h=h_ground)
    k_vals = (0.0, 0.05, 0.2, 0.5)

    gpu.set_device("cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = vt.dynamic_derivatives(ac, cond, st, k=k_vals, controls=("flap",))

    gpu.set_device("gpu")
    gpu.set_precision(precision)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = vt.dynamic_derivatives(ac, cond, st, k=k_vals, controls=("flap",))
    assert got.device == "gpu"
    assert got.precision == precision

    tol = TOL[precision]
    base_scale = max(abs(ref.k0["CL_delta_flap"]), 1.0)
    for key in ref.k0:
        if "_delta_" not in key:
            continue
        scale = max(abs(ref.k0[key]), base_scale)
        assert abs(ref.k0[key] - got.k0[key]) <= tol * scale, (key, abs(ref.k0[key] - got.k0[key]) / scale)
        a = ref.derivatives[key]
        b = got.derivatives[key]
        d_scale = max(float(np.max(np.abs(a))), base_scale)
        assert np.max(np.abs(a - b)) <= tol * d_scale, (key, np.max(np.abs(a - b)) / d_scale)

    for coeff in ("CL", "CD", "CY", "Cl", "Cm", "Cn"):
        a = ref.H_delta["flap"][coeff]
        b = got.H_delta["flap"][coeff]
        scale = max(float(np.max(np.abs(a))), base_scale)
        assert np.max(np.abs(a - b)) <= tol * scale


