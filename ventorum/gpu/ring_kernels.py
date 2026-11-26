# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""CUDA vortex-ring, steady-wake and far-wake tail kernels (NVIDIA Warp and PyTorch).

This module provides the GPU counterparts of :mod:`ventorum.aero.vortex_ring`
and :mod:`ventorum.unsteady.tail`:

* ``ring_velocity_gpu`` and ``ring_normal_influence_gpu``: unit-circulation
  velocity [1/m] and normal-velocity influence [1/m] of four-corner vortex
  rings and of rings with bent side polylines (camber or flap hinge nodes).
* ``steady_wake_velocity_gpu`` and ``steady_wake_normal_influence_gpu``:
  unit-circulation velocity [1/m] and normal-velocity influence [1/m] of the
  semi-infinite steady trailing-edge wake closure.
* ``harmonic_tail_velocity_gpu`` and ``harmonic_tail_normal_influence_gpu``:
  closed-form 3D semi-infinite harmonic far-wake tail on the GPU in
  complex128.

Ground images are computed inside the kernels using the exact reflection
identity of the Biot-Savart law: for a flat ground plane with unit normal
``gk`` and offset ``go``, the reflected point is ``R(P) = P - 2 (P . gk - go)
gk`` and the reflected vector is ``R_0(v) = v - 2 (v . gk) gk``. Because
``R_0`` is an improper orthogonal reflection (``det(R_0) = -1``), the
cross product transforms as ``R_0(a) x R_0(b) = -R_0(a x b)``, so a mirror
vortex element with circulation sign ``-1`` induces
``(-1) * (-R_0(v(R(P)))) = +R_0(v(R(P)))``. The same identity holds for
four-corner rings, bent side polylines, steady wake rays and the harmonic
far-wake tail.

Precision
---------
The Warp ring and steady-wake kernels are compiled for ``"float64"`` and
``"float32"`` using the same Biot-Savart formulas, Scully finite-filament
core regularisation and precision guards as :mod:`ventorum.gpu.kernels`:

* ``"float64"`` matches the numpy reference kernels to round-off.
* ``"float32"`` uses Dekker float32 pairs ``(hi, lo)`` for point-corner
  differences (Dekker 1971) and float64 cross products ``(b - a) x (P - a)``
  so points on segment lines, extended lines and at large coordinate offsets
  retain single-precision accuracy.

Memory
------
Persistent device-array holders (:class:`DeviceRingSet` and registered
harmonic solver states) register with this module so that
:func:`ventorum.gpu.release` frees all ring GPU arrays and returns the freed
byte count.

References
----------
* J. Katz and A. Plotkin, "Low-Speed Aerodynamics", 2nd ed., Cambridge
  University Press, 2001, chapters 10, 12 and 13 (Biot-Savart law, vortex
  rings, steady wake closure and method of images).
* M. P. Scully, "Computation of helicopter rotor wake geometry and its
  influence on rotor harmonic airloads", ASRL TR 178-1, MIT, 1975.
* T. J. Dekker, "A floating-point technique for extending the available
  precision", Numerische Mathematik 18, 1971, pp. 224-242.
"""

# No "from __future__ import annotations": Warp reads the type annotations of the
# kernels at definition time, with the types of the closure (T, vec).
import math
import threading
import weakref
from types import SimpleNamespace

import numpy as np
import torch
import warp as wp

from ventorum.gpu import engine
from ventorum.gpu import memory as gpu_memory

_INV_4PI = 1.0 / (4.0 * math.pi)
_FLOORS = {"float64": 1e-300, "float32": 1e-30}
_TAIL_CHUNK_BYTES = 4_000_000

_lock = threading.Lock()
_built: dict = {}

_registry_lock = threading.Lock()
_holders: weakref.WeakSet = weakref.WeakSet()
_tail_quad: dict = {}

# 32-point Gauss-Legendre quadrature tables on host (same nodes as ventorum.unsteady.tail).
_GL_X, _GL_W = np.polynomial.legendre.leggauss(32)
_TH_32 = 0.25 * np.pi * (_GL_X + 1.0)
_W_TH_32 = 0.25 * np.pi * _GL_W
_TAN_32_NP = np.ascontiguousarray(np.tan(_TH_32), dtype=np.float64)
_SEC2_W_32_NP = np.ascontiguousarray(_W_TH_32 / (np.cos(_TH_32) ** 2), dtype=np.float64)
_U01_32_NP = np.ascontiguousarray(0.5 * (_GL_X + 1.0), dtype=np.float64)
_W01_32_NP = np.ascontiguousarray(0.5 * _GL_W, dtype=np.float64)

# 64-point Gauss-Legendre tables for the 45-degree contour of the displaced steady far wake.
_GL64_X, _GL64_W = np.polynomial.legendre.leggauss(64)
_TH_64 = 0.25 * np.pi * (_GL64_X + 1.0)
_W_TH_64 = 0.25 * np.pi * _GL64_W
_T_64 = np.tan(_TH_64)
_TAN2_64_NP = np.ascontiguousarray(_T_64 * _T_64, dtype=np.float64)
_DTAN2_W_64_NP = np.ascontiguousarray((2.0 * _T_64 / (np.cos(_TH_64) ** 2)) * _W_TH_64, dtype=np.float64)


# ---------------------------------------------------------------------------
# Device-array registry for ventorum.gpu.release()
# ---------------------------------------------------------------------------


def register_device_holder(holder) -> None:
    """Register *holder* (must implement ``device_bytes()`` and ``release_gpu()``) for :func:`release_gpu`."""
    with _registry_lock:
        _holders.add(holder)


def unregister_device_holder(holder) -> None:
    """Remove *holder* from the active device-array registry."""
    with _registry_lock:
        _holders.discard(holder)


def device_bytes() -> int:
    """Return total CUDA memory [B] held by registered ring holders and tail tables."""
    seen: set = set()
    with _registry_lock:
        holders = list(_holders)
        quad = dict(_tail_quad)
    total = gpu_memory._tensor_bytes(quad, seen)
    for h in holders:
        tensors = getattr(h, "tensors", None)
        if tensors is not None:
            total += gpu_memory._tensor_bytes(tensors, seen)
        elif hasattr(h, "device_bytes"):
            total += int(h.device_bytes())
    return total


def release_gpu() -> int:
    """Release all registered vortex-ring device arrays and return freed bytes [B]."""
    with _registry_lock:
        holders = list(_holders)
        _holders.clear()
        quad = dict(_tail_quad)
        _tail_quad.clear()
    seen: set = set()
    freed = gpu_memory._tensor_bytes(quad, seen)
    quad.clear()
    for h in holders:
        if hasattr(h, "release_gpu"):
            freed += int(h.release_gpu())
    return freed


def _get_tail_quad() -> dict:
    """Return cached CUDA float64 Gauss-Legendre quadrature tensors for the harmonic tail."""
    with _registry_lock:
        if not _tail_quad:
            _tail_quad["tan"] = engine._f64(_TAN_32_NP)
            _tail_quad["sec2_w"] = engine._f64(_SEC2_W_32_NP)
            _tail_quad["u01"] = engine._f64(_U01_32_NP)
            _tail_quad["w01"] = engine._f64(_W01_32_NP)
            _tail_quad["tan2_64"] = engine._f64(_TAN2_64_NP)
            _tail_quad["dtan2_w_64"] = engine._f64(_DTAN2_W_64_NP)
        return _tail_quad


class DeviceRingSet:
    """Registered container for vortex-ring geometry tensors on the GPU.

    Attributes
    ----------
    precision : str
        Kernel precision (``"float64"`` or ``"float32"``).
    tensors : dict[str, torch.Tensor]
        Device tensors for corners, optional side polylines, float32 pairs and
        optional trailing-edge closure arrays.
    has_poly : bool
        True when bent side polylines are active.
    """

    def __init__(
        self,
        corners: np.ndarray,
        *,
        side_nodes: np.ndarray | None = None,
        side_counts: np.ndarray | None = None,
        te_left: np.ndarray | None = None,
        te_right: np.ndarray | None = None,
        wake_dir: np.ndarray | None = None,
        precision: str = "float64",
    ) -> None:
        self.precision = precision
        self.n_rings = int(corners.shape[0])
        self.tensors: dict[str, torch.Tensor] = {}
        c64 = engine._f64(corners)
        self.tensors["corners"] = c64
        if precision == "float32":
            self.tensors["Ch"], self.tensors["Cl"] = engine.split2(c64)
        else:
            self.tensors["Ch"] = torch.empty((1, 1, 3), dtype=torch.float32, device=engine._dev())
            self.tensors["Cl"] = self.tensors["Ch"]

        has_poly = (
            side_nodes is not None
            and side_counts is not None
            and side_nodes.shape[2] > 0
            and self.n_rings > 0
            and bool(np.any(np.asarray(side_counts) > 0))
        )
        self.has_poly = has_poly
        if has_poly:
            sn64 = engine._f64(side_nodes)
            self.tensors["side_nodes"] = sn64
            self.tensors["side_counts"] = engine._i32(side_counts)
            if precision == "float32":
                self.tensors["SNh"], self.tensors["SNl"] = engine.split2(sn64)
            else:
                self.tensors["SNh"] = torch.empty((1, 1, 1, 3), dtype=torch.float32, device=engine._dev())
                self.tensors["SNl"] = self.tensors["SNh"]

        if te_left is not None and te_right is not None:
            tl64 = engine._f64(te_left)
            tr64 = engine._f64(te_right)
            self.tensors["te_left"] = tl64
            self.tensors["te_right"] = tr64
            if precision == "float32":
                self.tensors["tlh"], self.tensors["tll"] = engine.split2(tl64)
                self.tensors["trh"], self.tensors["trl"] = engine.split2(tr64)
            else:
                z = torch.empty((1, 3), dtype=torch.float32, device=engine._dev())
                self.tensors["tlh"] = self.tensors["tll"] = z
                self.tensors["trh"] = self.tensors["trl"] = z
            if wake_dir is not None:
                self.tensors["wake_dir"] = engine._f64(_broadcast_wake_dir(wake_dir, int(te_left.shape[0])))

        register_device_holder(self)

    def device_bytes(self) -> int:
        """Return CUDA storage bytes held by this ring set [B]."""
        return gpu_memory._tensor_bytes(self.tensors, set())

    def release_gpu(self) -> int:
        """Release all CUDA tensors held by this ring set and return freed bytes [B]."""
        freed = self.device_bytes()
        self.tensors.clear()
        return freed


# ---------------------------------------------------------------------------
# NVIDIA Warp ring and steady-wake kernels
# ---------------------------------------------------------------------------


def kernels(precision: str) -> SimpleNamespace:
    """Return the Warp ring kernels for *precision* (``"float32"`` or ``"float64"``).

    Parameters
    ----------
    precision : str
        Kernel arithmetic precision, ``"float64"`` or ``"float32"``.

    Returns
    -------
    types.SimpleNamespace
        Compiled Warp kernel functions and type constants.
    """
    with _lock:
        ns = _built.get(precision)
        if ns is None:
            if precision not in _FLOORS:
                raise ValueError(f"Unknown precision {precision!r}; use 'float32' or 'float64'.")
            wp.config.log_level = getattr(wp, "LOG_WARNING", wp.config.log_level)
            wp.init()
            ns = _build(precision)
            _built[precision] = ns
        return ns


def _build(precision: str) -> SimpleNamespace:  # noqa: C901 - one Warp closure per precision
    T = wp.float64 if precision == "float64" else wp.float32
    vec = wp.vec3d if precision == "float64" else wp.vec3f
    floor = _FLOORS[precision]
    is64 = precision == "float64"
    module = f"ventorum_gpu_ring_{precision}"
    wp.set_module_options({"enable_backward": False}, wp.get_module(module))

    @wp.func(module=module)
    def tvec(x: wp.vec3d):
        return vec(T(x[0]), T(x[1]), T(x[2]))

    @wp.func(module=module)
    def rsqrt(x: T):
        # Same fast float64 inverse square root as ventorum.gpu.kernels.
        if wp.static(is64):
            if x > T(1.0e-30) and x < T(1.0e30):
                y = T(wp.float32(1.0) / wp.sqrt(wp.float32(x)))
                y = y * (T(1.5) - T(0.5) * x * y * y)
                y = y * (T(1.5) - T(0.5) * x * y * y)
                return y
            return T(1.0) / wp.sqrt(x)
        else:
            return T(1.0) / wp.sqrt(x)

    @wp.func(module=module)
    def rcp(x: T):
        if wp.static(is64):
            ax = wp.abs(x)
            if ax > T(1.0e-30) and ax < T(1.0e30):
                y = T(wp.float32(1.0) / wp.float32(x))
                y = y * (T(2.0) - x * y)
                y = y * (T(2.0) - x * y)
                return y
            return T(1.0) / x
        else:
            return T(1.0) / x

    @wp.func(module=module)
    def inv_norm(r: vec):
        return rsqrt(wp.dot(r, r) + T(floor))

    @wp.func(module=module)
    def seg_c(c: vec, r1: vec, r2: vec, r0: vec, i1: T, i2: T, rc2: T):
        # Same finite-filament Scully Biot-Savart formula as ventorum.gpu.kernels.seg_c.
        a1 = wp.dot(r0, r1)
        a2 = wp.dot(r0, r2)
        k = T(0.0)
        if a1 < T(0.0) or a2 > T(0.0):
            D2 = T(0.0)
            if a1 < T(0.0):
                D2 = wp.dot(r1, r1) + T(floor)
            else:
                D2 = wp.dot(r2, r2) + T(floor)
            e = rcp(i1 * i2) + wp.dot(r1, r2)
            if e > T(floor):
                k = T(_INV_4PI) * (i1 + i2) * rcp(e) * D2 * rcp(D2 + rc2)
        else:
            proj = wp.dot(r0, r1 * i1 - r2 * i2)
            den = wp.dot(c, c) + rc2 * wp.dot(r0, r0)
            if den > T(floor):
                k = T(_INV_4PI) * proj * rcp(den)
        return c * k

    @wp.func(module=module)
    def seg(r1: vec, r2: vec, r0: vec, i1: T, i2: T, rc2: T):
        return seg_c(wp.cross(r1, r2), r1, r2, r0, i1, i2, rc2)

    @wp.func(module=module)
    def ray(r: vec, ir: T, d: vec, rc2: T):
        # Same semi-infinite filament formula as ventorum.gpu.kernels.ray.
        c = wp.cross(d, r)
        dr = wp.dot(d, r)
        k = T(0.0)
        if dr < T(0.0):
            D2 = wp.dot(r, r) + T(floor)
            rn = D2 * ir
            e = rn * (rn - dr)
            if e > T(floor):
                k = T(_INV_4PI) * rcp(e) * D2 * rcp(D2 + rc2)
        else:
            den = wp.dot(c, c) + rc2
            if den > T(floor):
                k = T(_INV_4PI) * (T(1.0) + dr * ir) * rcp(den)
        return c * k

    @wp.func(module=module)
    def ray_pre(c: vec, dr: T, rn: T, rc2: T):
        # Same float32 semi-infinite filament formula as ventorum.gpu.kernels.ray_pre.
        k = T(0.0)
        if dr < T(0.0):
            D2 = rn * rn
            e = rn * (rn - dr)
            if e > T(floor):
                k = T(_INV_4PI) / e * D2 / (D2 + rc2)
        else:
            den = wp.dot(c, c) + rc2
            if den > T(floor):
                k = T(_INV_4PI) * (T(1.0) + dr / rn) / den
        return c * k

    @wp.func(module=module)
    def mirror_point(x: wp.vec3d, gk: wp.vec3d, go: wp.float64):
        return x - (wp.float64(2.0) * (wp.dot(x, gk) - go)) * gk

    @wp.func(module=module)
    def mirror_dir(v: vec, gk: vec):
        return v - (T(2.0) * wp.dot(v, gk)) * gk

    @wp.func(module=module)
    def diff2(ah: wp.vec3f, al: wp.vec3f, bh: wp.vec3f, bl: wp.vec3f):
        return (ah - bh) + (al - bl)

    @wp.func(module=module)
    def ring_seg(
        P64: wp.vec3d,
        a64: wp.vec3d,
        b64: wp.vec3d,
        r1: vec,
        r2: vec,
        i1: T,
        i2: T,
        rc2: T,
    ):
        # Finite segment a64 -> b64. Guard zero-length segments; in float32,
        # compute r0 and the collinearity-sensitive cross product (b - a) x (P - a)
        # in float64 (matching ventorum.gpu.kernels).
        r0_64 = b64 - a64
        if wp.dot(r0_64, r0_64) == wp.float64(0.0):
            return vec()
        if wp.static(is64):
            return seg(r1, r2, r0_64, i1, i2, rc2)
        else:
            c = tvec(wp.cross(r0_64, P64 - a64))
            return seg_c(c, r1, r2, tvec(r0_64), i1, i2, rc2)

    @wp.func(module=module)
    def ring_vel_quad_one(
        P64: wp.vec3d,
        Ph: wp.vec3f,
        Pl: wp.vec3f,
        corners: wp.array2d(dtype=wp.vec3d),
        Ch: wp.array2d(dtype=wp.vec3f),
        Cl: wp.array2d(dtype=wp.vec3f),
        j: int,
        rc2: T,
    ):
        c0 = corners[j, 0]
        c1 = corners[j, 1]
        c2 = corners[j, 2]
        c3 = corners[j, 3]
        r0 = vec()
        r1 = vec()
        r2 = vec()
        r3 = vec()
        if wp.static(is64):
            r0 = P64 - c0
            r1 = P64 - c1
            r2 = P64 - c2
            r3 = P64 - c3
        else:
            r0 = diff2(Ph, Pl, Ch[j, 0], Cl[j, 0])
            r1 = diff2(Ph, Pl, Ch[j, 1], Cl[j, 1])
            r2 = diff2(Ph, Pl, Ch[j, 2], Cl[j, 2])
            r3 = diff2(Ph, Pl, Ch[j, 3], Cl[j, 3])
        i0 = inv_norm(r0)
        i1 = inv_norm(r1)
        i2 = inv_norm(r2)
        i3 = inv_norm(r3)
        v = ring_seg(P64, c0, c1, r0, r1, i0, i1, rc2)
        v = v + ring_seg(P64, c1, c2, r1, r2, i1, i2, rc2)
        v = v + ring_seg(P64, c2, c3, r2, r3, i2, i3, rc2)
        v = v + ring_seg(P64, c3, c0, r3, r0, i3, i0, rc2)
        return v

    @wp.func(module=module)
    def ring_vel_poly_one(
        P64: wp.vec3d,
        Ph: wp.vec3f,
        Pl: wp.vec3f,
        corners: wp.array2d(dtype=wp.vec3d),
        Ch: wp.array2d(dtype=wp.vec3f),
        Cl: wp.array2d(dtype=wp.vec3f),
        sn: wp.array3d(dtype=wp.vec3d),
        SNh: wp.array3d(dtype=wp.vec3f),
        SNl: wp.array3d(dtype=wp.vec3f),
        sc: wp.array2d(dtype=wp.int32),
        j: int,
        rc2: T,
    ):
        c0 = corners[j, 0]
        c1 = corners[j, 1]
        c2 = corners[j, 2]
        c3 = corners[j, 3]
        r0 = vec()
        r1 = vec()
        r2 = vec()
        r3 = vec()
        if wp.static(is64):
            r0 = P64 - c0
            r1 = P64 - c1
            r2 = P64 - c2
            r3 = P64 - c3
        else:
            r0 = diff2(Ph, Pl, Ch[j, 0], Cl[j, 0])
            r1 = diff2(Ph, Pl, Ch[j, 1], Cl[j, 1])
            r2 = diff2(Ph, Pl, Ch[j, 2], Cl[j, 2])
            r3 = diff2(Ph, Pl, Ch[j, 3], Cl[j, 3])
        i0 = inv_norm(r0)
        i1 = inv_norm(r1)
        i2 = inv_norm(r2)
        i3 = inv_norm(r3)

        # Leading bound segment: c0 -> c1.
        v = ring_seg(P64, c0, c1, r0, r1, i0, i1, rc2)

        # Right side polyline: c1 -> sn[j, 0, 0..cr-1] -> c2.
        cr = sc[j, 0]
        if cr == 0:
            v = v + ring_seg(P64, c1, c2, r1, r2, i1, i2, rc2)
        else:
            pa = c1
            ra = r1
            ia = i1
            for k in range(cr):
                pb = sn[j, 0, k]
                rb = vec()
                if wp.static(is64):
                    rb = P64 - pb
                else:
                    rb = diff2(Ph, Pl, SNh[j, 0, k], SNl[j, 0, k])
                ib = inv_norm(rb)
                v = v + ring_seg(P64, pa, pb, ra, rb, ia, ib, rc2)
                pa = pb
                ra = rb
                ia = ib
            v = v + ring_seg(P64, pa, c2, ra, r2, ia, i2, rc2)

        # Trailing spanwise segment: c2 -> c3.
        v = v + ring_seg(P64, c2, c3, r2, r3, i2, i3, rc2)

        # Left side polyline: c3 -> sn[j, 1, 0..cl-1] -> c0.
        cl = sc[j, 1]
        if cl == 0:
            v = v + ring_seg(P64, c3, c0, r3, r0, i3, i0, rc2)
        else:
            pa = c3
            ra = r3
            ia = i3
            for k in range(cl):
                pb = sn[j, 1, k]
                rb = vec()
                if wp.static(is64):
                    rb = P64 - pb
                else:
                    rb = diff2(Ph, Pl, SNh[j, 1, k], SNl[j, 1, k])
                ib = inv_norm(rb)
                v = v + ring_seg(P64, pa, pb, ra, rb, ia, ib, rc2)
                pa = pb
                ra = rb
                ia = ib
            v = v + ring_seg(P64, pa, c0, ra, r0, ia, i0, rc2)
        return v

    @wp.func(module=module)
    def steady_wake_vel_one(
        P64: wp.vec3d,
        Ph: wp.vec3f,
        Pl: wp.vec3f,
        tl: wp.array(dtype=wp.vec3d),
        tlh: wp.array(dtype=wp.vec3f),
        tll: wp.array(dtype=wp.vec3f),
        tr: wp.array(dtype=wp.vec3d),
        trh: wp.array(dtype=wp.vec3f),
        trl: wp.array(dtype=wp.vec3f),
        wd: wp.array(dtype=wp.vec3d),
        j: int,
        rc2: T,
    ):
        pl64 = tl[j]
        pr64 = tr[j]
        d64 = wd[j]
        rl = vec()
        rr = vec()
        if wp.static(is64):
            rl = P64 - pl64
            rr = P64 - pr64
        else:
            rl = diff2(Ph, Pl, tlh[j], tll[j])
            rr = diff2(Ph, Pl, trh[j], trl[j])
        il = inv_norm(rl)
        ir = inv_norm(rr)
        v = ring_seg(P64, pl64, pr64, rl, rr, il, ir, rc2)
        if wp.static(is64):
            v = v + ray(rr, ir, d64, rc2) - ray(rl, il, d64, rc2)
        else:
            r64_r = P64 - pr64
            r64_l = P64 - pl64
            cr = tvec(wp.cross(d64, r64_r))
            cl = tvec(wp.cross(d64, r64_l))
            drr = T(wp.dot(d64, r64_r))
            drl = T(wp.dot(d64, r64_l))
            v = v + ray_pre(cr, drr, T(1.0) / ir, rc2) - ray_pre(cl, drl, T(1.0) / il, rc2)
        return v

    @wp.func(module=module)
    def eval_quad_with_img(
        P64: wp.vec3d,
        Ph: wp.vec3f,
        Pl: wp.vec3f,
        corners: wp.array2d(dtype=wp.vec3d),
        Ch: wp.array2d(dtype=wp.vec3f),
        Cl: wp.array2d(dtype=wp.vec3f),
        j: int,
        rc2: T,
        img_mode: int,
        gk: wp.vec3d,
        go: wp.float64,
    ):
        v = vec()
        if img_mode != 2:
            v = ring_vel_quad_one(P64, Ph, Pl, corners, Ch, Cl, j, rc2)
        if img_mode != 0:
            P_img = mirror_point(P64, gk, go)
            if wp.static(is64):
                vi = ring_vel_quad_one(P_img, Ph, Pl, corners, Ch, Cl, j, rc2)
            else:
                Pih = wp.vec3f(P_img)
                Pil = wp.vec3f(P_img - wp.vec3d(Pih))
                vi = ring_vel_quad_one(P_img, Pih, Pil, corners, Ch, Cl, j, rc2)
            v = v + mirror_dir(vi, tvec(gk))
        return v

    @wp.func(module=module)
    def eval_poly_with_img(
        P64: wp.vec3d,
        Ph: wp.vec3f,
        Pl: wp.vec3f,
        corners: wp.array2d(dtype=wp.vec3d),
        Ch: wp.array2d(dtype=wp.vec3f),
        Cl: wp.array2d(dtype=wp.vec3f),
        sn: wp.array3d(dtype=wp.vec3d),
        SNh: wp.array3d(dtype=wp.vec3f),
        SNl: wp.array3d(dtype=wp.vec3f),
        sc: wp.array2d(dtype=wp.int32),
        j: int,
        rc2: T,
        img_mode: int,
        gk: wp.vec3d,
        go: wp.float64,
    ):
        v = vec()
        if img_mode != 2:
            v = ring_vel_poly_one(P64, Ph, Pl, corners, Ch, Cl, sn, SNh, SNl, sc, j, rc2)
        if img_mode != 0:
            P_img = mirror_point(P64, gk, go)
            if wp.static(is64):
                vi = ring_vel_poly_one(P_img, Ph, Pl, corners, Ch, Cl, sn, SNh, SNl, sc, j, rc2)
            else:
                Pih = wp.vec3f(P_img)
                Pil = wp.vec3f(P_img - wp.vec3d(Pih))
                vi = ring_vel_poly_one(P_img, Pih, Pil, corners, Ch, Cl, sn, SNh, SNl, sc, j, rc2)
            v = v + mirror_dir(vi, tvec(gk))
        return v

    @wp.func(module=module)
    def eval_wake_with_img(
        P64: wp.vec3d,
        Ph: wp.vec3f,
        Pl: wp.vec3f,
        tl: wp.array(dtype=wp.vec3d),
        tlh: wp.array(dtype=wp.vec3f),
        tll: wp.array(dtype=wp.vec3f),
        tr: wp.array(dtype=wp.vec3d),
        trh: wp.array(dtype=wp.vec3f),
        trl: wp.array(dtype=wp.vec3f),
        wd: wp.array(dtype=wp.vec3d),
        j: int,
        rc2: T,
        img_mode: int,
        gk: wp.vec3d,
        go: wp.float64,
    ):
        v = vec()
        if img_mode != 2:
            v = steady_wake_vel_one(P64, Ph, Pl, tl, tlh, tll, tr, trh, trl, wd, j, rc2)
        if img_mode != 0:
            P_img = mirror_point(P64, gk, go)
            if wp.static(is64):
                vi = steady_wake_vel_one(P_img, Ph, Pl, tl, tlh, tll, tr, trh, trl, wd, j, rc2)
            else:
                Pih = wp.vec3f(P_img)
                Pil = wp.vec3f(P_img - wp.vec3d(Pih))
                vi = steady_wake_vel_one(P_img, Pih, Pil, tl, tlh, tll, tr, trh, trl, wd, j, rc2)
            v = v + mirror_dir(vi, tvec(gk))
        return v

    @wp.kernel(module=module)
    def ring_vel_quad(
        P: wp.array(dtype=wp.vec3d),
        Ph: wp.array(dtype=wp.vec3f),
        Pl: wp.array(dtype=wp.vec3f),
        corners: wp.array2d(dtype=wp.vec3d),
        Ch: wp.array2d(dtype=wp.vec3f),
        Cl: wp.array2d(dtype=wp.vec3f),
        rc2: wp.array2d(dtype=T),
        rc_is_2d: int,
        img_mode: int,
        gk: wp.vec3d,
        go: wp.float64,
        outT: wp.array2d(dtype=vec),
    ):
        j, i = wp.tid()
        c2 = rc2[j, i] if rc_is_2d != 0 else rc2[j, 0]
        ph = wp.vec3f()
        pl = wp.vec3f()
        if wp.static(not is64):
            ph = Ph[i]
            pl = Pl[i]
        outT[j, i] = eval_quad_with_img(P[i], ph, pl, corners, Ch, Cl, j, c2, img_mode, gk, go)

    @wp.kernel(module=module)
    def ring_normal_quad(
        P: wp.array(dtype=wp.vec3d),
        Ph: wp.array(dtype=wp.vec3f),
        Pl: wp.array(dtype=wp.vec3f),
        N: wp.array(dtype=vec),
        corners: wp.array2d(dtype=wp.vec3d),
        Ch: wp.array2d(dtype=wp.vec3f),
        Cl: wp.array2d(dtype=wp.vec3f),
        rc2: wp.array2d(dtype=T),
        rc_is_2d: int,
        img_mode: int,
        gk: wp.vec3d,
        go: wp.float64,
        outT: wp.array2d(dtype=T),
    ):
        j, i = wp.tid()
        c2 = rc2[j, i] if rc_is_2d != 0 else rc2[j, 0]
        ph = wp.vec3f()
        pl = wp.vec3f()
        if wp.static(not is64):
            ph = Ph[i]
            pl = Pl[i]
        v = eval_quad_with_img(P[i], ph, pl, corners, Ch, Cl, j, c2, img_mode, gk, go)
        outT[j, i] = wp.dot(v, N[i])

    @wp.kernel(module=module)
    def ring_vel_poly(
        P: wp.array(dtype=wp.vec3d),
        Ph: wp.array(dtype=wp.vec3f),
        Pl: wp.array(dtype=wp.vec3f),
        corners: wp.array2d(dtype=wp.vec3d),
        Ch: wp.array2d(dtype=wp.vec3f),
        Cl: wp.array2d(dtype=wp.vec3f),
        sn: wp.array3d(dtype=wp.vec3d),
        SNh: wp.array3d(dtype=wp.vec3f),
        SNl: wp.array3d(dtype=wp.vec3f),
        sc: wp.array2d(dtype=wp.int32),
        rc2: wp.array2d(dtype=T),
        rc_is_2d: int,
        img_mode: int,
        gk: wp.vec3d,
        go: wp.float64,
        outT: wp.array2d(dtype=vec),
    ):
        j, i = wp.tid()
        c2 = rc2[j, i] if rc_is_2d != 0 else rc2[j, 0]
        ph = wp.vec3f()
        pl = wp.vec3f()
        if wp.static(not is64):
            ph = Ph[i]
            pl = Pl[i]
        outT[j, i] = eval_poly_with_img(
            P[i], ph, pl, corners, Ch, Cl, sn, SNh, SNl, sc, j, c2, img_mode, gk, go,
        )

    @wp.kernel(module=module)
    def ring_normal_poly(
        P: wp.array(dtype=wp.vec3d),
        Ph: wp.array(dtype=wp.vec3f),
        Pl: wp.array(dtype=wp.vec3f),
        N: wp.array(dtype=vec),
        corners: wp.array2d(dtype=wp.vec3d),
        Ch: wp.array2d(dtype=wp.vec3f),
        Cl: wp.array2d(dtype=wp.vec3f),
        sn: wp.array3d(dtype=wp.vec3d),
        SNh: wp.array3d(dtype=wp.vec3f),
        SNl: wp.array3d(dtype=wp.vec3f),
        sc: wp.array2d(dtype=wp.int32),
        rc2: wp.array2d(dtype=T),
        rc_is_2d: int,
        img_mode: int,
        gk: wp.vec3d,
        go: wp.float64,
        outT: wp.array2d(dtype=T),
    ):
        j, i = wp.tid()
        c2 = rc2[j, i] if rc_is_2d != 0 else rc2[j, 0]
        ph = wp.vec3f()
        pl = wp.vec3f()
        if wp.static(not is64):
            ph = Ph[i]
            pl = Pl[i]
        v = eval_poly_with_img(
            P[i], ph, pl, corners, Ch, Cl, sn, SNh, SNl, sc, j, c2, img_mode, gk, go,
        )
        outT[j, i] = wp.dot(v, N[i])

    @wp.kernel(module=module)
    def steady_wake_vel(
        P: wp.array(dtype=wp.vec3d),
        Ph: wp.array(dtype=wp.vec3f),
        Pl: wp.array(dtype=wp.vec3f),
        tl: wp.array(dtype=wp.vec3d),
        tlh: wp.array(dtype=wp.vec3f),
        tll: wp.array(dtype=wp.vec3f),
        tr: wp.array(dtype=wp.vec3d),
        trh: wp.array(dtype=wp.vec3f),
        trl: wp.array(dtype=wp.vec3f),
        wd: wp.array(dtype=wp.vec3d),
        rc2: wp.array2d(dtype=T),
        rc_is_2d: int,
        img_mode: int,
        gk: wp.vec3d,
        go: wp.float64,
        outT: wp.array2d(dtype=vec),
    ):
        j, i = wp.tid()
        c2 = rc2[j, i] if rc_is_2d != 0 else rc2[j, 0]
        ph = wp.vec3f()
        pl = wp.vec3f()
        if wp.static(not is64):
            ph = Ph[i]
            pl = Pl[i]
        outT[j, i] = eval_wake_with_img(
            P[i], ph, pl, tl, tlh, tll, tr, trh, trl, wd, j, c2, img_mode, gk, go,
        )

    @wp.kernel(module=module)
    def steady_wake_normal(
        P: wp.array(dtype=wp.vec3d),
        Ph: wp.array(dtype=wp.vec3f),
        Pl: wp.array(dtype=wp.vec3f),
        N: wp.array(dtype=vec),
        tl: wp.array(dtype=wp.vec3d),
        tlh: wp.array(dtype=wp.vec3f),
        tll: wp.array(dtype=wp.vec3f),
        tr: wp.array(dtype=wp.vec3d),
        trh: wp.array(dtype=wp.vec3f),
        trl: wp.array(dtype=wp.vec3f),
        wd: wp.array(dtype=wp.vec3d),
        rc2: wp.array2d(dtype=T),
        rc_is_2d: int,
        img_mode: int,
        gk: wp.vec3d,
        go: wp.float64,
        outT: wp.array2d(dtype=T),
    ):
        j, i = wp.tid()
        c2 = rc2[j, i] if rc_is_2d != 0 else rc2[j, 0]
        ph = wp.vec3f()
        pl = wp.vec3f()
        if wp.static(not is64):
            ph = Ph[i]
            pl = Pl[i]
        v = eval_wake_with_img(
            P[i], ph, pl, tl, tlh, tll, tr, trh, trl, wd, j, c2, img_mode, gk, go,
        )
        outT[j, i] = wp.dot(v, N[i])

    @wp.kernel(module=module)
    def segment_vel(
        P: wp.array(dtype=wp.vec3d),
        A: wp.array(dtype=wp.vec3d),
        B: wp.array(dtype=wp.vec3d),
        rc2: wp.array2d(dtype=T),
        rc_is_2d: int,
        outT: wp.array2d(dtype=vec),
    ):
        # Single finite segment A[j] -> B[j] evaluated at P[i] (used by GPU tail Part 1).
        j, i = wp.tid()
        c2 = rc2[j, i] if rc_is_2d != 0 else rc2[j, 0]
        P64 = P[i]
        a64 = A[j]
        b64 = B[j]
        r1 = tvec(P64 - a64)
        r2 = tvec(P64 - b64)
        i1 = inv_norm(r1)
        i2 = inv_norm(r2)
        outT[j, i] = ring_seg(P64, a64, b64, r1, r2, i1, i2, c2)

    return SimpleNamespace(
        T=T,
        vec=vec,
        precision=precision,
        ring_vel_quad=ring_vel_quad,
        ring_normal_quad=ring_normal_quad,
        ring_vel_poly=ring_vel_poly,
        ring_normal_poly=ring_normal_poly,
        steady_wake_vel=steady_wake_vel,
        steady_wake_normal=steady_wake_normal,
        segment_vel=segment_vel,
    )


# ---------------------------------------------------------------------------
# Tensor packing helpers
# ---------------------------------------------------------------------------


def _as_cuda_f64(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=engine._dev(), dtype=torch.float64).contiguous()
    return engine._f64(x)


def _as_cuda_i32(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=engine._dev(), dtype=torch.int32).contiguous()
    return engine._i32(x)


def _broadcast_wake_dir(wake_dir, n: int) -> np.ndarray:
    wd = np.asarray(wake_dir, dtype=np.float64)
    if wd.ndim == 1:
        if wd.shape != (3,):
            raise ValueError(f"Expected wake_dir of shape (3,) or ({n}, 3), got {wd.shape}.")
        return np.ascontiguousarray(np.broadcast_to(wd, (n, 3)))
    if wd.ndim == 2 and wd.shape == (n, 3):
        return np.ascontiguousarray(wd)
    raise ValueError(f"Expected wake_dir of shape (3,) or ({n}, 3), got {wd.shape}.")


def _pack_rc2_gpu(rc, m: int, n: int, t_dtype: torch.dtype) -> tuple[torch.Tensor, int]:
    """Pack squared core radius [m^2] on GPU as shape ``(n, 1)`` or ``(n, m)``."""
    dev = engine._dev()
    if isinstance(rc, torch.Tensor):
        t = rc.to(device=dev, dtype=torch.float64)
        if t.ndim == 0:
            return torch.full((n, 1), float(t.item()) ** 2, dtype=t_dtype, device=dev), 0
        if t.ndim == 1:
            if t.shape[0] != n:
                raise ValueError(f"Expected rc of shape ({n},), got {tuple(t.shape)}.")
            return (t * t).to(t_dtype).reshape(n, 1).contiguous(), 0
        if t.ndim == 2:
            if tuple(t.shape) != (m, n):
                raise ValueError(f"Expected rc of shape ({m}, {n}), got {tuple(t.shape)}.")
            return (t * t).transpose(0, 1).to(t_dtype).contiguous(), 1
        raise ValueError(f"Expected rc to be scalar, 1-D or 2-D, got shape {tuple(t.shape)}.")
    arr = np.asarray(rc, dtype=np.float64)
    if arr.ndim == 0:
        return torch.full((n, 1), float(arr) ** 2, dtype=t_dtype, device=dev), 0
    if arr.ndim == 1:
        if arr.shape[0] != n:
            raise ValueError(f"Expected rc of shape ({n},), got {arr.shape}.")
        return engine._f64(arr ** 2).to(t_dtype).reshape(n, 1).contiguous(), 0
    if arr.ndim == 2:
        if arr.shape != (m, n):
            raise ValueError(f"Expected rc of shape ({m}, {n}), got {arr.shape}.")
        return engine._f64(np.ascontiguousarray((arr ** 2).T)).to(t_dtype).contiguous(), 1
    raise ValueError(f"Expected rc to be scalar, 1-D or 2-D, got shape {arr.shape}.")


def _ground_params(ground, img_mode: int | None) -> tuple[int, wp.vec3d, float]:
    """Return ``(img_mode, gk_vec3d, go_float)`` for a Warp kernel launch."""
    if ground is None:
        if img_mode is not None and img_mode != 0:
            raise ValueError("img_mode != 0 requires a ground plane.")
        return 0, wp.vec3d(0.0, 0.0, 1.0), 0.0
    mode = 1 if img_mode is None else int(img_mode)
    if isinstance(ground, tuple):
        gk_arr, go = ground
        gk_np = np.asarray(gk_arr, dtype=np.float64)
        go_f = float(go)
    else:
        gk_np = np.asarray(ground.normal, dtype=np.float64)
        go_f = float(ground.offset)
    return mode, wp.vec3d(float(gk_np[0]), float(gk_np[1]), float(gk_np[2])), go_f


def _pairs_or_dummy(t64: torch.Tensor, is32: bool) -> tuple[torch.Tensor, torch.Tensor]:
    if is32:
        return engine.split2(t64)
    shape = tuple(1 for _ in range(t64.ndim - 1)) + (3,)
    z = torch.empty(shape, dtype=torch.float32, device=engine._dev())
    return z, z


def _has_active_poly(n: int, side_nodes, side_counts) -> bool:
    if side_nodes is None and side_counts is None:
        return False
    if side_nodes is None or side_counts is None:
        raise ValueError("side_nodes and side_counts must both be provided or both be None.")
    if n == 0 or int(side_nodes.shape[2]) == 0:
        return False
    if isinstance(side_counts, torch.Tensor):
        return bool((side_counts > 0).any().item())
    return bool(np.any(np.asarray(side_counts) > 0))


# ---------------------------------------------------------------------------
# High-level GPU ring and steady-wake functions
# ---------------------------------------------------------------------------


def ring_velocity_gpu(
    P,
    corners,
    rc,
    *,
    side_nodes=None,
    side_counts=None,
    ground=None,
    img_mode: int | None = None,
    precision: str = "float64",
    out_tensor: bool = False,
):
    """Compute unit-circulation vortex-ring velocities [1/m] on the GPU.

    Parameters
    ----------
    P : numpy.ndarray or torch.Tensor
        Evaluation points [m], shape ``(M, 3)``.
    corners : numpy.ndarray or torch.Tensor
        Ring corner coordinates [m], shape ``(N, 4, 3)``.
    rc : float, numpy.ndarray or torch.Tensor
        Core radius [m], scalar, shape ``(N,)`` or shape ``(M, N)``.
    side_nodes : numpy.ndarray, torch.Tensor or None, optional
        Padded intermediate bent side-edge nodes [m], shape ``(N, 2, K, 3)``.
    side_counts : numpy.ndarray, torch.Tensor or None, optional
        Active intermediate node counts per side edge [-], shape ``(N, 2)``.
    ground : GroundPlane, tuple or None, optional
        Flat ground plane (or ``(normal, offset)`` tuple) for ground images
        inside the kernel.
    img_mode : int or None, optional
        ``0`` real only, ``1`` real plus ground image (default when *ground*
        is given), ``2`` ground image only (with circulation sign ``-1``).
    precision : {'float64', 'float32'}, optional
        Warp kernel arithmetic precision (default ``"float64"``).
    out_tensor : bool, optional
        When True return a CUDA :class:`torch.Tensor` of shape ``(M, N, 3)``;
        when False (default) return a host float64 :class:`numpy.ndarray`.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Induced velocity per unit circulation [1/m], shape ``(M, N, 3)``.
    """
    P64 = _as_cuda_f64(P)
    C64 = _as_cuda_f64(corners)
    m, n = int(P64.shape[0]), int(C64.shape[0])
    t_dtype = engine.torch_dtype(precision)
    if m == 0 or n == 0:
        out = torch.empty((m, n, 3), dtype=t_dtype, device=engine._dev())
        return out if out_tensor else out.double().cpu().numpy()

    k = kernels(precision)
    is32 = precision == "float32"
    Ph, Pl = _pairs_or_dummy(P64, is32)
    Ch, Cl = _pairs_or_dummy(C64, is32)
    rc2, rc_is_2d = _pack_rc2_gpu(rc, m, n, t_dtype)
    mode, gk, go = _ground_params(ground, img_mode)
    outT = torch.empty((n, m, 3), dtype=t_dtype, device=engine._dev())

    if _has_active_poly(n, side_nodes, side_counts):
        SN64 = _as_cuda_f64(side_nodes)
        SC32 = _as_cuda_i32(side_counts)
        SNh, SNl = _pairs_or_dummy(SN64, is32)
        engine.launch(
            k.ring_vel_poly,
            (n, m),
            [
                engine.view(P64, wp.vec3d), engine.view(Ph, wp.vec3f), engine.view(Pl, wp.vec3f),
                engine.view(C64, wp.vec3d), engine.view(Ch, wp.vec3f), engine.view(Cl, wp.vec3f),
                engine.view(SN64, wp.vec3d), engine.view(SNh, wp.vec3f), engine.view(SNl, wp.vec3f),
                engine.view(SC32, wp.int32),
                engine.view(rc2, k.T), rc_is_2d, mode, gk, go,
                engine.view(outT, k.vec),
            ],
        )
    else:
        engine.launch(
            k.ring_vel_quad,
            (n, m),
            [
                engine.view(P64, wp.vec3d), engine.view(Ph, wp.vec3f), engine.view(Pl, wp.vec3f),
                engine.view(C64, wp.vec3d), engine.view(Ch, wp.vec3f), engine.view(Cl, wp.vec3f),
                engine.view(rc2, k.T), rc_is_2d, mode, gk, go,
                engine.view(outT, k.vec),
            ],
        )

    out = outT.transpose(0, 1)
    return out if out_tensor else out.double().cpu().numpy()


def ring_normal_influence_gpu(
    P,
    normals,
    corners,
    rc,
    *,
    side_nodes=None,
    side_counts=None,
    ground=None,
    img_mode: int | None = None,
    precision: str = "float64",
    out_tensor: bool = False,
):
    """Compute unit-circulation vortex-ring normal-velocity influence [1/m] on the GPU.

    Parameters
    ----------
    P : numpy.ndarray or torch.Tensor
        Evaluation points [m], shape ``(M, 3)``.
    normals : numpy.ndarray or torch.Tensor
        Unit normal vectors at *P* [-], shape ``(M, 3)``.
    corners : numpy.ndarray or torch.Tensor
        Ring corner coordinates [m], shape ``(N, 4, 3)``.
    rc : float, numpy.ndarray or torch.Tensor
        Core radius [m], scalar, shape ``(N,)`` or shape ``(M, N)``.
    side_nodes : numpy.ndarray, torch.Tensor or None, optional
        Padded intermediate bent side-edge nodes [m], shape ``(N, 2, K, 3)``.
    side_counts : numpy.ndarray, torch.Tensor or None, optional
        Active intermediate node counts per side edge [-], shape ``(N, 2)``.
    ground : GroundPlane, tuple or None, optional
        Flat ground plane (or ``(normal, offset)`` tuple) for ground images.
    img_mode : int or None, optional
        ``0`` real only, ``1`` real plus ground image, ``2`` ground image only.
    precision : {'float64', 'float32'}, optional
        Warp kernel arithmetic precision (default ``"float64"``).
    out_tensor : bool, optional
        When True return a CUDA :class:`torch.Tensor` of shape ``(M, N)``;
        when False (default) return a host float64 :class:`numpy.ndarray`.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Normal-velocity influence matrix [1/m], shape ``(M, N)``.
    """
    P64 = _as_cuda_f64(P)
    C64 = _as_cuda_f64(corners)
    m, n = int(P64.shape[0]), int(C64.shape[0])
    t_dtype = engine.torch_dtype(precision)
    if m == 0 or n == 0:
        out = torch.empty((m, n), dtype=t_dtype, device=engine._dev())
        return out if out_tensor else out.double().cpu().numpy()

    N_t = _as_cuda_f64(normals).to(t_dtype).contiguous()
    k = kernels(precision)
    is32 = precision == "float32"
    Ph, Pl = _pairs_or_dummy(P64, is32)
    Ch, Cl = _pairs_or_dummy(C64, is32)
    rc2, rc_is_2d = _pack_rc2_gpu(rc, m, n, t_dtype)
    mode, gk, go = _ground_params(ground, img_mode)
    outT = torch.empty((n, m), dtype=t_dtype, device=engine._dev())

    if _has_active_poly(n, side_nodes, side_counts):
        SN64 = _as_cuda_f64(side_nodes)
        SC32 = _as_cuda_i32(side_counts)
        SNh, SNl = _pairs_or_dummy(SN64, is32)
        engine.launch(
            k.ring_normal_poly,
            (n, m),
            [
                engine.view(P64, wp.vec3d), engine.view(Ph, wp.vec3f), engine.view(Pl, wp.vec3f),
                engine.view(N_t, k.vec),
                engine.view(C64, wp.vec3d), engine.view(Ch, wp.vec3f), engine.view(Cl, wp.vec3f),
                engine.view(SN64, wp.vec3d), engine.view(SNh, wp.vec3f), engine.view(SNl, wp.vec3f),
                engine.view(SC32, wp.int32),
                engine.view(rc2, k.T), rc_is_2d, mode, gk, go,
                engine.view(outT, k.T),
            ],
        )
    else:
        engine.launch(
            k.ring_normal_quad,
            (n, m),
            [
                engine.view(P64, wp.vec3d), engine.view(Ph, wp.vec3f), engine.view(Pl, wp.vec3f),
                engine.view(N_t, k.vec),
                engine.view(C64, wp.vec3d), engine.view(Ch, wp.vec3f), engine.view(Cl, wp.vec3f),
                engine.view(rc2, k.T), rc_is_2d, mode, gk, go,
                engine.view(outT, k.T),
            ],
        )

    out = outT.transpose(0, 1)
    return out if out_tensor else out.double().cpu().numpy()


def steady_wake_velocity_gpu(
    P,
    te_left,
    te_right,
    wake_dir,
    rc,
    *,
    ground=None,
    img_mode: int | None = None,
    precision: str = "float64",
    out_tensor: bool = False,
):
    """Compute unit-circulation steady wake closure velocity [1/m] on the GPU.

    Parameters
    ----------
    P : numpy.ndarray or torch.Tensor
        Evaluation points [m], shape ``(M, 3)``.
    te_left, te_right : numpy.ndarray or torch.Tensor
        Left and right trailing-edge closure corners [m], shape ``(N_s, 3)``.
    wake_dir : numpy.ndarray or torch.Tensor
        Unit wake direction [-], shape ``(3,)`` or ``(N_s, 3)``.
    rc : float, numpy.ndarray or torch.Tensor
        Core radius [m], scalar, shape ``(N_s,)`` or shape ``(M, N_s)``.
    ground : GroundPlane, tuple or None, optional
        Flat ground plane (or ``(normal, offset)`` tuple) for ground images.
    img_mode : int or None, optional
        ``0`` real only, ``1`` real plus ground image, ``2`` ground image only.
    precision : {'float64', 'float32'}, optional
        Warp kernel arithmetic precision (default ``"float64"``).
    out_tensor : bool, optional
        When True return a CUDA :class:`torch.Tensor` of shape ``(M, N_s, 3)``;
        when False (default) return a host float64 :class:`numpy.ndarray`.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Induced velocity per unit circulation [1/m], shape ``(M, N_s, 3)``.
    """
    P64 = _as_cuda_f64(P)
    tl64 = _as_cuda_f64(te_left)
    tr64 = _as_cuda_f64(te_right)
    m, n = int(P64.shape[0]), int(tl64.shape[0])
    t_dtype = engine.torch_dtype(precision)
    if m == 0 or n == 0:
        out = torch.empty((m, n, 3), dtype=t_dtype, device=engine._dev())
        return out if out_tensor else out.double().cpu().numpy()

    if isinstance(wake_dir, torch.Tensor):
        wd64 = wake_dir.to(device=engine._dev(), dtype=torch.float64)
        if wd64.ndim == 1:
            wd64 = wd64.unsqueeze(0).expand(n, 3).contiguous()
        else:
            wd64 = wd64.contiguous()
    else:
        wd64 = engine._f64(_broadcast_wake_dir(wake_dir, n))

    k = kernels(precision)
    is32 = precision == "float32"
    Ph, Pl = _pairs_or_dummy(P64, is32)
    tlh, tll = _pairs_or_dummy(tl64, is32)
    trh, trl = _pairs_or_dummy(tr64, is32)
    rc2, rc_is_2d = _pack_rc2_gpu(rc, m, n, t_dtype)
    mode, gk, go = _ground_params(ground, img_mode)
    outT = torch.empty((n, m, 3), dtype=t_dtype, device=engine._dev())

    engine.launch(
        k.steady_wake_vel,
        (n, m),
        [
            engine.view(P64, wp.vec3d), engine.view(Ph, wp.vec3f), engine.view(Pl, wp.vec3f),
            engine.view(tl64, wp.vec3d), engine.view(tlh, wp.vec3f), engine.view(tll, wp.vec3f),
            engine.view(tr64, wp.vec3d), engine.view(trh, wp.vec3f), engine.view(trl, wp.vec3f),
            engine.view(wd64, wp.vec3d),
            engine.view(rc2, k.T), rc_is_2d, mode, gk, go,
            engine.view(outT, k.vec),
        ],
    )
    out = outT.transpose(0, 1)
    return out if out_tensor else out.double().cpu().numpy()


def steady_wake_normal_influence_gpu(
    P,
    normals,
    te_left,
    te_right,
    wake_dir,
    rc,
    *,
    ground=None,
    img_mode: int | None = None,
    precision: str = "float64",
    out_tensor: bool = False,
):
    """Compute unit-circulation steady wake closure normal influence [1/m] on the GPU.

    Parameters
    ----------
    P : numpy.ndarray or torch.Tensor
        Evaluation points [m], shape ``(M, 3)``.
    normals : numpy.ndarray or torch.Tensor
        Unit normal vectors at *P* [-], shape ``(M, 3)``.
    te_left, te_right : numpy.ndarray or torch.Tensor
        Left and right trailing-edge closure corners [m], shape ``(N_s, 3)``.
    wake_dir : numpy.ndarray or torch.Tensor
        Unit wake direction [-], shape ``(3,)`` or ``(N_s, 3)``.
    rc : float, numpy.ndarray or torch.Tensor
        Core radius [m], scalar, shape ``(N_s,)`` or shape ``(M, N_s)``.
    ground : GroundPlane, tuple or None, optional
        Flat ground plane (or ``(normal, offset)`` tuple) for ground images.
    img_mode : int or None, optional
        ``0`` real only, ``1`` real plus ground image, ``2`` ground image only.
    precision : {'float64', 'float32'}, optional
        Warp kernel arithmetic precision (default ``"float64"``).
    out_tensor : bool, optional
        When True return a CUDA :class:`torch.Tensor` of shape ``(M, N_s)``;
        when False (default) return a host float64 :class:`numpy.ndarray`.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Normal-velocity influence matrix [1/m], shape ``(M, N_s)``.
    """
    P64 = _as_cuda_f64(P)
    tl64 = _as_cuda_f64(te_left)
    tr64 = _as_cuda_f64(te_right)
    m, n = int(P64.shape[0]), int(tl64.shape[0])
    t_dtype = engine.torch_dtype(precision)
    if m == 0 or n == 0:
        out = torch.empty((m, n), dtype=t_dtype, device=engine._dev())
        return out if out_tensor else out.double().cpu().numpy()

    N_t = _as_cuda_f64(normals).to(t_dtype).contiguous()
    if isinstance(wake_dir, torch.Tensor):
        wd64 = wake_dir.to(device=engine._dev(), dtype=torch.float64)
        if wd64.ndim == 1:
            wd64 = wd64.unsqueeze(0).expand(n, 3).contiguous()
        else:
            wd64 = wd64.contiguous()
    else:
        wd64 = engine._f64(_broadcast_wake_dir(wake_dir, n))

    k = kernels(precision)
    is32 = precision == "float32"
    Ph, Pl = _pairs_or_dummy(P64, is32)
    tlh, tll = _pairs_or_dummy(tl64, is32)
    trh, trl = _pairs_or_dummy(tr64, is32)
    rc2, rc_is_2d = _pack_rc2_gpu(rc, m, n, t_dtype)
    mode, gk, go = _ground_params(ground, img_mode)
    outT = torch.empty((n, m), dtype=t_dtype, device=engine._dev())

    engine.launch(
        k.steady_wake_normal,
        (n, m),
        [
            engine.view(P64, wp.vec3d), engine.view(Ph, wp.vec3f), engine.view(Pl, wp.vec3f),
            engine.view(N_t, k.vec),
            engine.view(tl64, wp.vec3d), engine.view(tlh, wp.vec3f), engine.view(tll, wp.vec3f),
            engine.view(tr64, wp.vec3d), engine.view(trh, wp.vec3f), engine.view(trl, wp.vec3f),
            engine.view(wd64, wp.vec3d),
            engine.view(rc2, k.T), rc_is_2d, mode, gk, go,
            engine.view(outT, k.T),
        ],
    )
    out = outT.transpose(0, 1)
    return out if out_tensor else out.double().cpu().numpy()


# ---------------------------------------------------------------------------
# Closed-form 3D semi-infinite harmonic far-wake tail on the GPU (complex128)
# ---------------------------------------------------------------------------


def _segment_vel_gpu_f64(P64: torch.Tensor, A64: torch.Tensor, B64: torch.Tensor, rc2_T: torch.Tensor, rc_is_2d: int) -> torch.Tensor:
    """Return float64 segment velocity ``(M, N, 3)`` on CUDA via Warp."""
    m, n = int(P64.shape[0]), int(A64.shape[0])
    k = kernels("float64")
    outT = torch.empty((n, m, 3), dtype=torch.float64, device=engine._dev())
    engine.launch(
        k.segment_vel,
        (n, m),
        [
            engine.view(P64, wp.vec3d), engine.view(A64, wp.vec3d), engine.view(B64, wp.vec3d),
            engine.view(rc2_T, k.T), rc_is_2d, engine.view(outT, k.vec),
        ],
    )
    return outT.transpose(0, 1)


def _contour_integrals_gpu(
    a_L: torch.Tensor,
    a_R: torch.Tensor,
    rho2_L: torch.Tensor,
    rho2_R: torch.Tensor,
    v0_sq: torch.Tensor,
    v0_dot_v1: torch.Tensor,
    v1_sq: torch.Tensor,
    tL_0: torch.Tensor,
    tR_0: torch.Tensor,
    d_dot_us: torch.Tensor,
    L_scale: torch.Tensor,
    kappa: float,
    quad: dict,
    xi_shift: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate the four scalar contour-rotated tail integrals on CUDA in complex128."""
    u = L_scale.unsqueeze(-1) * quad["tan"]
    du_w = L_scale.unsqueeze(-1) * quad["sec2_w"]
    exp_arg = torch.clamp(kappa * u, max=700.0)
    wt = (-1j) * du_w * torch.exp(-exp_arg)
    if xi_shift is not None:
        xi_s = xi_shift.unsqueeze(-1)
        wt = wt * torch.exp((-1j * kappa) * xi_s)
        xi = xi_s - 1j * u
    else:
        xi = -1j * u

    sL = xi + a_L.unsqueeze(-1)
    sR = xi + a_R.unsqueeze(-1)
    r2L = sL * sL + rho2_L.unsqueeze(-1)
    r2R = sR * sR + rho2_R.unsqueeze(-1)
    radL = torch.sqrt(r2L)
    radR = torch.sqrt(r2R)

    I_tr_R = torch.sum(wt / (radR * r2R), dim=-1)
    I_tr_L = torch.sum(wt / (radL * r2L), dim=-1)

    q_perp2 = v0_sq.unsqueeze(-1) - 2.0 * xi * v0_dot_v1.unsqueeze(-1) + (xi * xi) * v1_sq.unsqueeze(-1)
    xi_dus = xi * d_dot_us.unsqueeze(-1)
    num_R = -tR_0.unsqueeze(-1) + xi_dus
    num_L = -tL_0.unsqueeze(-1) + xi_dus
    span_br = (num_R / radR - num_L / radL) / q_perp2

    I_sp0 = torch.sum(wt * span_br, dim=-1)
    I_sp1 = torch.sum((wt * xi) * span_br, dim=-1)
    return I_tr_R, I_tr_L, I_sp0, I_sp1


def _graded_segment_nodes_gpu(
    xi_0: torch.Tensor,
    centers: torch.Tensor,
    scales: torch.Tensor,
    quad: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return graded nodes and weights [m] on ``[0, xi_0]`` on CUDA.

    This is the CUDA twin of :func:`ventorum.unsteady.tail._graded_segment_nodes`.
    """
    n = int(xi_0.shape[0])
    end = xi_0.unsqueeze(-1)
    clipped = torch.minimum(torch.clamp(centers, min=0.0), end)
    bp, _ = torch.sort(torch.cat([torch.zeros_like(end), clipped, end], dim=1), dim=1)
    lo = bp[:, :-1]
    hi = bp[:, 1:]
    half = 0.5 * (hi - lo)
    anchors = torch.cat([lo, hi], dim=1)
    n_pan = int(lo.shape[1])
    sign = torch.cat([
        torch.ones(n_pan, dtype=torch.float64, device=xi_0.device),
        -torch.ones(n_pan, dtype=torch.float64, device=xi_0.device),
    ])
    length = torch.cat([half, half], dim=1)
    dist = torch.sqrt((anchors.unsqueeze(-1) - centers.unsqueeze(1)) ** 2 + scales.unsqueeze(1) ** 2)
    b = torch.maximum(torch.amin(dist, dim=2), 1e-12 * end + 1e-300)
    lam = torch.log1p(length / b)
    delta = b.unsqueeze(-1) * torch.expm1(lam.unsqueeze(-1) * quad["u01"])
    w = (delta + b.unsqueeze(-1)) * (lam.unsqueeze(-1) * quad["w01"])
    xi = anchors.unsqueeze(-1) + sign[None, :, None] * delta
    return xi.reshape(n, -1), w.reshape(n, -1)


def _segment_layout_gpu(
    a_L: torch.Tensor,
    a_R: torch.Tensor,
    rho2_L: torch.Tensor,
    rho2_R: torch.Tensor,
    v0_sq: torch.Tensor,
    v0_dot_v1: torch.Tensor,
    v1_sq: torch.Tensor,
    rc2: torch.Tensor,
    span_len: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(xi_0, pad, centers, scales)`` [m] of the tail real segment on CUDA.

    This is the CUDA twin of :func:`ventorum.unsteady.tail._segment_layout`.
    """
    v1s = torch.clamp(v1_sq, min=1e-300)
    x_min = v0_dot_v1 / v1s
    q_min = torch.clamp(v0_sq - v0_dot_v1 * x_min, min=0.0)
    centers = torch.stack([-a_L, -a_R, x_min], dim=1)
    scales = torch.sqrt(torch.stack([rho2_L + rc2, rho2_R + rc2, (q_min + rc2) / v1s], dim=1))
    pad = torch.maximum(torch.amax(scales, dim=1), span_len)
    xi_0 = -torch.minimum(torch.minimum(a_L, a_R), -x_min) + pad
    return xi_0, pad, centers, scales


def _real_segment_integrals_gpu(
    a_L: torch.Tensor,
    a_R: torch.Tensor,
    rho2_L: torch.Tensor,
    rho2_R: torch.Tensor,
    v0_sq: torch.Tensor,
    v0_dot_v1: torch.Tensor,
    v1_sq: torch.Tensor,
    tL_0: torch.Tensor,
    tR_0: torch.Tensor,
    d_dot_us: torch.Tensor,
    xi: torch.Tensor,
    w: torch.Tensor,
    kappa: float,
    rc2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate the four scalar tail integrals on ``[0, xi_0]`` on CUDA in complex128.

    *xi* and *w* are the graded nodes and weights [m] of :func:`_graded_segment_nodes_gpu`.
    """
    wt = w * torch.exp((-1j * kappa) * xi)

    sL = xi + a_L.unsqueeze(-1)
    sR = xi + a_R.unsqueeze(-1)
    r2L = sL * sL + rho2_L.unsqueeze(-1) + rc2.unsqueeze(-1)
    r2R = sR * sR + rho2_R.unsqueeze(-1) + rc2.unsqueeze(-1)
    radL = torch.sqrt(r2L)
    radR = torch.sqrt(r2R)

    I_tr_R = torch.sum(wt / (radR * r2R), dim=-1)
    I_tr_L = torch.sum(wt / (radL * r2L), dim=-1)

    q_perp2 = (
        v0_sq.unsqueeze(-1)
        - 2.0 * xi * v0_dot_v1.unsqueeze(-1)
        + (xi * xi) * v1_sq.unsqueeze(-1)
        + rc2.unsqueeze(-1)
    )
    xi_dus = xi * d_dot_us.unsqueeze(-1)
    num_R = -tR_0.unsqueeze(-1) + xi_dus
    num_L = -tL_0.unsqueeze(-1) + xi_dus
    span_br = (num_R / radR - num_L / radL) / q_perp2

    I_sp0 = torch.sum(wt * span_br, dim=-1)
    I_sp1 = torch.sum((wt * xi) * span_br, dim=-1)
    return I_tr_R, I_tr_L, I_sp0, I_sp1


def _tail_velocity_single_gpu(
    P64: torch.Tensor,
    pl64: torch.Tensor,
    pr64: torch.Tensor,
    wd64: torch.Tensor,
    kappa: float,
    rc2_T: torch.Tensor,
    rc_is_2d: int,
) -> torch.Tensor:
    """Evaluate real-side harmonic tail velocity [1/m] on CUDA, shape ``(M, N_s, 3)`` complex128."""
    dev = engine._dev()
    m, n = int(P64.shape[0]), int(pl64.shape[0])
    quad = _get_tail_quad()

    l_span = pr64 - pl64
    span_len = torch.linalg.vector_norm(l_span, dim=1, keepdim=True)
    u_s = l_span / torch.clamp(span_len, min=1e-300)
    v1 = torch.linalg.cross(u_s, wd64, dim=-1)
    v1_sq = torch.sum(v1 * v1, dim=-1)
    d_dot_us = torch.sum(wd64 * u_s, dim=-1)

    out = torch.empty((m, n, 3), dtype=torch.complex128, device=dev)
    step = max(1, min(m, _TAIL_CHUNK_BYTES // max(1, n * 32 * 16)))
    span_pref = (-1j * kappa) * _INV_4PI

    for i0 in range(0, m, step):
        i1 = min(m, i0 + step)
        Pc = P64[i0:i1].contiguous()
        rc2_c = rc2_T[:, i0:i1].contiguous() if rc_is_2d else rc2_T

        # Part 1: closing spanwise segment at xi = 0 (p_left -> p_right).
        v_tot = _segment_vel_gpu_f64(Pc, pl64, pr64, rc2_c, rc_is_2d).to(torch.complex128)

        r_L = Pc[:, None, :] - pl64[None, :, :]
        r_R = Pc[:, None, :] - pr64[None, :, :]
        a_L = -torch.sum(r_L * wd64[None, :, :], dim=-1)
        a_R = -torch.sum(r_R * wd64[None, :, :], dim=-1)

        wd_b = wd64[None, :, :].expand_as(r_L)
        d_x_rL = torch.linalg.cross(wd_b, r_L, dim=-1)
        d_x_rR = torch.linalg.cross(wd_b, r_R, dim=-1)
        rho2_L = torch.sum(d_x_rL * d_x_rL, dim=-1)
        rho2_R = torch.sum(d_x_rR * d_x_rR, dim=-1)

        us_b = u_s[None, :, :].expand_as(r_L)
        v0 = torch.linalg.cross(us_b, r_L, dim=-1)
        v0_sq = torch.sum(v0 * v0, dim=-1)
        v0_dot_v1 = torch.sum(v0 * v1[None, :, :], dim=-1)
        xi_min = v0_dot_v1 / torch.clamp(v1_sq[None, :], min=1e-300)
        tL_0 = torch.sum(r_L * u_s[None, :, :], dim=-1)
        tR_0 = torch.sum(r_R * u_s[None, :, :], dim=-1)

        a_min = torch.minimum(torch.minimum(a_L, a_R), -xi_min)
        v1_sq_2d = v1_sq[None, :].expand_as(a_L)
        dus_2d = d_dot_us[None, :].expand_as(a_L)

        pos = a_min > 0.0
        if bool(pos.all().item()):
            L_scale = torch.clamp(a_min, max=2.0 / kappa)
            I_tr_R, I_tr_L, I_sp0, I_sp1 = _contour_integrals_gpu(
                a_L, a_R, rho2_L, rho2_R, v0_sq, v0_dot_v1, v1_sq_2d,
                tL_0, tR_0, dus_2d, L_scale, kappa, quad,
            )
        else:
            I_tr_R = torch.empty(a_L.shape, dtype=torch.complex128, device=dev)
            I_tr_L = torch.empty(a_L.shape, dtype=torch.complex128, device=dev)
            I_sp0 = torch.empty(a_L.shape, dtype=torch.complex128, device=dev)
            I_sp1 = torch.empty(a_L.shape, dtype=torch.complex128, device=dev)
            if bool(pos.any().item()):
                L_p = torch.clamp(a_min[pos], max=2.0 / kappa)
                rR, rL, s0, s1 = _contour_integrals_gpu(
                    a_L[pos], a_R[pos], rho2_L[pos], rho2_R[pos],
                    v0_sq[pos], v0_dot_v1[pos], v1_sq_2d[pos],
                    tL_0[pos], tR_0[pos], dus_2d[pos], L_p, kappa, quad,
                )
                I_tr_R[pos], I_tr_L[pos], I_sp0[pos], I_sp1[pos] = rR, rL, s0, s1
            neg = ~pos
            if bool(neg.any().item()):
                rc2_mat = rc2_c.transpose(0, 1).expand_as(a_L)
                fields = (
                    a_L[neg], a_R[neg], rho2_L[neg], rho2_R[neg],
                    v0_sq[neg], v0_dot_v1[neg], v1_sq_2d[neg],
                    tL_0[neg], tR_0[neg], dus_2d[neg],
                    torch.clamp(rc2_mat[neg], min=1e-24),
                    span_len.reshape(1, -1).expand_as(a_L)[neg],
                )
                n_neg = int(fields[0].shape[0])
                res = [torch.empty(n_neg, dtype=torch.complex128, device=dev) for _ in range(4)]
                batch = max(1, _TAIL_CHUNK_BYTES // (4 * 64 * 16))
                for j0 in range(0, n_neg, batch):
                    j1 = min(n_neg, j0 + batch)
                    aL, aR, r2L, r2R, v0s, v01, v1s, tL, tR, dus, rc2_n, sp = (
                        f[j0:j1] for f in fields
                    )
                    xi_0, pad, centers, scales = _segment_layout_gpu(
                        aL, aR, r2L, r2R, v0s, v01, v1s, rc2_n, sp,
                    )
                    xi_q, w_q = _graded_segment_nodes_gpu(xi_0, centers, scales, quad)
                    part1 = _real_segment_integrals_gpu(
                        aL, aR, r2L, r2R, v0s, v01, v1s, tL, tR, dus, xi_q, w_q, kappa, rc2_n,
                    )
                    part2 = _contour_integrals_gpu(
                        aL, aR, r2L, r2R, v0s, v01, v1s, tL, tR, dus,
                        torch.clamp(pad, max=2.0 / kappa), kappa, quad, xi_shift=xi_0,
                    )
                    for out_k, p1, p2 in zip(res, part1, part2, strict=True):
                        out_k[j0:j1] = p1 + p2
                I_tr_R[neg], I_tr_L[neg], I_sp0[neg], I_sp1[neg] = res

        v_tot = v_tot + _INV_4PI * (I_tr_R.unsqueeze(-1) * d_x_rR - I_tr_L.unsqueeze(-1) * d_x_rL)
        v_tot = v_tot + span_pref * (I_sp0.unsqueeze(-1) * v0 - I_sp1.unsqueeze(-1) * v1[None, :, :])
        out[i0:i1] = v_tot

    return out


def harmonic_tail_velocity_gpu(
    P,
    p_left,
    p_right,
    wake_dir,
    kappa: float,
    rc=0.0,
    *,
    ground=None,
    img_mode: int | None = None,
    out_tensor: bool = False,
):
    """Compute unit-strength 3D harmonic far-wake tail velocity [1/m] on the GPU.

    Parameters
    ----------
    P : numpy.ndarray or torch.Tensor
        Evaluation points [m], shape ``(M, 3)``.
    p_left, p_right : numpy.ndarray or torch.Tensor
        Left and right tail start corners [m], shape ``(N_s, 3)``.
    wake_dir : numpy.ndarray or torch.Tensor
        Unit wake direction [-], shape ``(3,)`` or ``(N_s, 3)``.
    kappa : float
        Spatial wavenumber ``omega / V_inf`` [1/m], must be ``>= 0``.
    rc : float, numpy.ndarray or torch.Tensor, optional
        Core radius [m], scalar, shape ``(N_s,)`` or shape ``(M, N_s)``.
    ground : GroundPlane, tuple or None, optional
        Flat ground plane for ground images (exact reflection identity).
    img_mode : int or None, optional
        ``0`` real only, ``1`` real plus ground image, ``2`` ground image only.
    out_tensor : bool, optional
        When True return a CUDA :class:`torch.Tensor` (complex128);
        when False (default) return a host complex128 :class:`numpy.ndarray`.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Complex induced velocity per unit circulation at ``xi = 0`` [1/m],
        shape ``(M, N_s, 3)``.
    """
    kap = float(kappa)
    if not math.isfinite(kap) or kap < 0.0:
        raise ValueError(f"kappa must be finite and >= 0, got {kappa!r}.")
    P64 = _as_cuda_f64(P)
    pl64 = _as_cuda_f64(p_left)
    pr64 = _as_cuda_f64(p_right)
    m, n = int(P64.shape[0]), int(pl64.shape[0])
    if m == 0 or n == 0:
        out = torch.empty((m, n, 3), dtype=torch.complex128, device=engine._dev())
        return out if out_tensor else out.cpu().numpy()

    if isinstance(wake_dir, torch.Tensor):
        wd64 = wake_dir.to(device=engine._dev(), dtype=torch.float64)
        if wd64.ndim == 1:
            wd64 = wd64.unsqueeze(0).expand(n, 3).contiguous()
        else:
            wd64 = wd64.contiguous()
    else:
        wd64 = engine._f64(_broadcast_wake_dir(wake_dir, n))
    wd_norms = torch.linalg.vector_norm(wd64, dim=1, keepdim=True)
    if bool((wd_norms <= 1e-12).any().item()):
        raise ValueError("wake_dir must have non-zero length for every strip.")
    wd64 = (wd64 / wd_norms).contiguous()

    if kap == 0.0:
        out = steady_wake_velocity_gpu(
            P64, pl64, pr64, wd64, rc, ground=ground, img_mode=img_mode, precision="float64", out_tensor=True,
        ).to(torch.complex128)
        return out if out_tensor else out.cpu().numpy()

    rc2_T, rc_is_2d = _pack_rc2_gpu(rc, m, n, torch.float64)
    mode, gk_wp, go = _ground_params(ground, img_mode)

    out = None
    if mode != 2:
        out = _tail_velocity_single_gpu(P64, pl64, pr64, wd64, kap, rc2_T, rc_is_2d)
    if mode != 0:
        gk = torch.tensor([gk_wp[0], gk_wp[1], gk_wp[2]], dtype=torch.float64, device=engine._dev())
        P_img = P64 - (2.0 * (P64 @ gk - go)).unsqueeze(-1) * gk[None, :]
        vi = _tail_velocity_single_gpu(P_img, pl64, pr64, wd64, kap, rc2_T, rc_is_2d)
        # Apply real reflection R_0(v) = v - 2 (v . gk) gk to complex vector vi.
        gk_c = gk.to(torch.complex128)
        vi_ref = vi - (2.0 * torch.sum(vi * gk_c[None, None, :], dim=-1, keepdim=True)) * gk_c[None, None, :]
        out = vi_ref if out is None else (out + vi_ref)

    return out if out_tensor else out.cpu().numpy()


def harmonic_tail_normal_influence_gpu(
    P,
    normals,
    p_left,
    p_right,
    wake_dir,
    kappa: float,
    rc=0.0,
    *,
    ground=None,
    img_mode: int | None = None,
    out_tensor: bool = False,
):
    """Compute unit-strength 3D harmonic far-wake tail normal influence [1/m] on the GPU.

    Parameters
    ----------
    P : numpy.ndarray or torch.Tensor
        Evaluation points [m], shape ``(M, 3)``.
    normals : numpy.ndarray or torch.Tensor
        Unit normal vectors at *P* [-], shape ``(M, 3)``.
    p_left, p_right : numpy.ndarray or torch.Tensor
        Left and right tail start corners [m], shape ``(N_s, 3)``.
    wake_dir : numpy.ndarray or torch.Tensor
        Unit wake direction [-], shape ``(3,)`` or ``(N_s, 3)``.
    kappa : float
        Spatial wavenumber ``omega / V_inf`` [1/m], must be ``>= 0``.
    rc : float, numpy.ndarray or torch.Tensor, optional
        Core radius [m], scalar, shape ``(N_s,)`` or shape ``(M, N_s)``.
    ground : GroundPlane, tuple or None, optional
        Flat ground plane for ground images.
    img_mode : int or None, optional
        ``0`` real only, ``1`` real plus ground image, ``2`` ground image only.
    out_tensor : bool, optional
        When True return a CUDA :class:`torch.Tensor` (complex128);
        when False (default) return a host complex128 :class:`numpy.ndarray`.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Complex normal-velocity influence matrix [1/m], shape ``(M, N_s)``.
    """
    N_c = _as_cuda_f64(normals).to(torch.complex128)
    v = harmonic_tail_velocity_gpu(
        P, p_left, p_right, wake_dir, kappa, rc, ground=ground, img_mode=img_mode, out_tensor=True,
    )
    out = torch.einsum("msi,mi->ms", v, N_c)
    return out if out_tensor else out.cpu().numpy()


def segment_velocity_gpu(
    P,
    A,
    B,
    rc,
    *,
    ground=None,
    img_mode: int | None = None,
    out_tensor: bool = False,
):
    """Compute unit-circulation finite-segment velocity [1/m] on the GPU in float64.

    Parameters
    ----------
    P : numpy.ndarray or torch.Tensor
        Evaluation points [m], shape ``(M, 3)``.
    A, B : numpy.ndarray or torch.Tensor
        Segment start and end points [m], shape ``(N, 3)``.
    rc : float, numpy.ndarray or torch.Tensor
        Core radius [m], scalar, shape ``(N,)`` or shape ``(M, N)``.
    ground : GroundPlane, tuple or None, optional
        Flat ground plane for ground images (exact reflection identity).
    img_mode : int or None, optional
        ``0`` real only, ``1`` real plus ground image, ``2`` ground image only.
    out_tensor : bool, optional
        When True return a CUDA :class:`torch.Tensor` (float64); when False
        return a host float64 :class:`numpy.ndarray`.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Induced velocity per unit circulation [1/m], shape ``(M, N, 3)``.
    """
    P64 = _as_cuda_f64(P)
    A64 = _as_cuda_f64(A)
    B64 = _as_cuda_f64(B)
    m, n = int(P64.shape[0]), int(A64.shape[0])
    if m == 0 or n == 0:
        out = torch.empty((m, n, 3), dtype=torch.float64, device=engine._dev())
        return out if out_tensor else out.cpu().numpy()
    rc2_T, rc_is_2d = _pack_rc2_gpu(rc, m, n, torch.float64)
    mode, gk_wp, go = _ground_params(ground, img_mode)
    out = None
    if mode != 2:
        out = _segment_vel_gpu_f64(P64, A64, B64, rc2_T, rc_is_2d)
    if mode != 0:
        gk = torch.tensor([gk_wp[0], gk_wp[1], gk_wp[2]], dtype=torch.float64, device=engine._dev())
        P_img = P64 - (2.0 * (P64 @ gk - go)).unsqueeze(-1) * gk[None, :]
        vi = _segment_vel_gpu_f64(P_img, A64, B64, rc2_T, rc_is_2d)
        vi_ref = vi - (2.0 * torch.sum(vi * gk[None, None, :], dim=-1, keepdim=True)) * gk[None, None, :]
        out = vi_ref if out is None else (out + vi_ref)
    return out if out_tensor else out.cpu().numpy()


def _as_cuda_disp(u, n: int, name: str) -> torch.Tensor:
    """Validate and convert displacement *u* to contiguous CUDA float64 of shape ``(n, 3)``."""
    if isinstance(u, torch.Tensor):
        t = u.to(device=engine._dev(), dtype=torch.float64)
        if t.ndim == 1:
            if tuple(t.shape) != (3,):
                raise ValueError(f"Expected {name} of shape (3,) or ({n}, 3), got {tuple(t.shape)}.")
            return t.unsqueeze(0).expand(n, 3).contiguous()
        if t.ndim == 2 and tuple(t.shape) == (n, 3):
            return t.contiguous()
        raise ValueError(f"Expected {name} of shape (3,) or ({n}, 3), got {tuple(t.shape)}.")
    return engine._f64(_broadcast_wake_dir(u, n))


def _displaced_ray_contour_gpu(
    r: torch.Tensor,
    a: torch.Tensor,
    rho2: torch.Tensor,
    d: torch.Tensor,
    u: torch.Tensor,
    kappa: float,
    L_w,
    quad: dict,
) -> torch.Tensor:
    """Evaluate displaced semi-infinite filament integral for ``a > 0`` on CUDA in complex128."""
    r_norm = torch.sqrt(a * a + rho2)
    d_x_r = torch.linalg.cross(d, r, dim=-1)
    d_x_u = torch.linalg.cross(d, u, dim=-1)
    u_x_r = torch.linalg.cross(u, r, dim=-1)
    u_x_d = -d_x_u
    r_dot_u = torch.sum(r * u, dim=-1)
    d_dot_u = torch.sum(d * u, dim=-1)

    D = r_norm * (r_norm + a)
    dD = -(2.0 + a / r_norm) * r_dot_u + r_norm * d_dot_u
    I_static = _INV_4PI * (-d_x_u / D.unsqueeze(-1) - d_x_r * (dD / (D * D)).unsqueeze(-1))

    L_scale = torch.clamp(r_norm, max=2.0 / kappa)
    uv = L_scale.unsqueeze(-1) * quad["tan2_64"]
    du_w = L_scale.unsqueeze(-1) * quad["dtan2_w_64"]
    xi = torch.complex(uv, -uv)
    decay = torch.exp(-torch.clamp(kappa * uv, max=700.0))
    ang = -kappa * uv
    osc = torch.complex(torch.cos(ang), torch.sin(ang))
    wt = (1.0 - 1.0j) * du_w * decay * osc

    s = xi + a.unsqueeze(-1)
    r2 = s * s + rho2.unsqueeze(-1)
    rad = torch.sqrt(r2)
    inv_r3 = 1.0 / (rad * r2)
    inv_r5 = inv_r3 / r2

    J30 = torch.sum(wt * inv_r3, dim=-1)
    J31 = torch.sum((wt * xi) * inv_r3, dim=-1)
    J50 = torch.sum(wt * inv_r5, dim=-1)
    J51 = torch.sum((wt * xi) * inv_r5, dim=-1)

    I_osc = _INV_4PI * (
        ((-1j * kappa) * u_x_r - d_x_u) * J30.unsqueeze(-1)
        + (1j * kappa) * u_x_d * J31.unsqueeze(-1)
        + 3.0 * d_x_r * (r_dot_u * J50 - d_dot_u * J51).unsqueeze(-1)
    )
    if isinstance(L_w, torch.Tensor):
        ang_w = -kappa * L_w
        phi_tail = torch.complex(torch.cos(ang_w), torch.sin(ang_w)).unsqueeze(-1)
    else:
        phi_tail = complex(math.cos(-kappa * float(L_w)), math.sin(-kappa * float(L_w)))
    return phi_tail * I_osc - I_static


def _displaced_ray_real_seg_gpu(
    r: torch.Tensor,
    a: torch.Tensor,
    rho2: torch.Tensor,
    d: torch.Tensor,
    u: torch.Tensor,
    xi: torch.Tensor,
    w: torch.Tensor,
    kappa: float,
    L_w: float,
    rc2: torch.Tensor,
) -> torch.Tensor:
    """Evaluate displaced filament integral on real ``[0, xi_0]`` on CUDA in complex128.

    *xi* and *w* are the graded nodes and weights [m] of :func:`_graded_segment_nodes_gpu`.
    """
    d_x_r = torch.linalg.cross(d, r, dim=-1)
    d_x_u = torch.linalg.cross(d, u, dim=-1)
    u_x_r = torch.linalg.cross(u, r, dim=-1)
    u_x_d = -d_x_u
    r_dot_u = torch.sum(r * u, dim=-1)
    d_dot_u = torch.sum(d * u, dim=-1)

    b = -kappa * (L_w + xi)
    # Exact cancellation-free exp(i b) - 1 = -2 sin^2(b / 2) + i sin(b).
    psi = torch.complex(-2.0 * (torch.sin(0.5 * b) ** 2), torch.sin(b))
    dpsi = (-1j * kappa) * torch.complex(torch.cos(b), torch.sin(b))

    s = xi + a.unsqueeze(-1)
    r2 = s * s + rho2.unsqueeze(-1) + rc2.unsqueeze(-1)
    rad = torch.sqrt(r2)
    inv_r3 = 1.0 / (rad * r2)
    inv_r5 = inv_r3 / r2

    K30_p = torch.sum((w * dpsi) * inv_r3, dim=-1)
    K31_p = torch.sum((w * dpsi * xi) * inv_r3, dim=-1)
    K30 = torch.sum((w * psi) * inv_r3, dim=-1)
    K50 = torch.sum((w * psi) * inv_r5, dim=-1)
    K51 = torch.sum((w * psi * xi) * inv_r5, dim=-1)

    return _INV_4PI * (
        u_x_r * K30_p.unsqueeze(-1)
        - u_x_d * K31_p.unsqueeze(-1)
        - d_x_u * K30.unsqueeze(-1)
        + 3.0 * d_x_r * (r_dot_u * K50 - d_dot_u * K51).unsqueeze(-1)
    )


def _displaced_tail_velocity_single_gpu(
    P64: torch.Tensor,
    pl64: torch.Tensor,
    pr64: torch.Tensor,
    wd64: torch.Tensor,
    ul64: torch.Tensor,
    ur64: torch.Tensor,
    kappa: float,
    L_w: float,
    rc2_T: torch.Tensor,
    rc_is_2d: int,
) -> torch.Tensor:
    """Evaluate real-side displaced steady far-wake velocity on CUDA, shape ``(M, N_s, 3)`` complex128."""
    dev = engine._dev()
    m, n = int(P64.shape[0]), int(pl64.shape[0])
    quad = _get_tail_quad()
    span_len = torch.linalg.vector_norm(pr64 - pl64, dim=1)
    out = torch.zeros((m, n, 3), dtype=torch.complex128, device=dev)
    step = max(1, min(m, _TAIL_CHUNK_BYTES // max(1, n * 64 * 16)))

    for i0 in range(0, m, step):
        i1 = min(m, i0 + step)
        Pc = P64[i0:i1].contiguous()
        rc2_c = rc2_T[:, i0:i1].contiguous() if rc_is_2d else rc2_T
        v_tot = torch.zeros((i1 - i0, n, 3), dtype=torch.complex128, device=dev)
        wd_b = wd64[None, :, :].expand(i1 - i0, n, 3)

        for sign, pe, ue in ((+1.0, pr64, ur64), (-1.0, pl64, ul64)):
            r = Pc[:, None, :] - pe[None, :, :]
            a = -torch.sum(r * wd64[None, :, :], dim=-1)
            d_x_r = torch.linalg.cross(wd_b, r, dim=-1)
            rho2 = torch.sum(d_x_r * d_x_r, dim=-1)
            u_b = ue[None, :, :].expand_as(r)

            pos = a > 0.0
            if bool(pos.all().item()):
                v_ray = _displaced_ray_contour_gpu(r, a, rho2, wd_b, u_b, kappa, L_w, quad)
            else:
                v_ray = torch.empty(r.shape, dtype=torch.complex128, device=dev)
                if bool(pos.any().item()):
                    v_ray[pos] = _displaced_ray_contour_gpu(
                        r[pos], a[pos], rho2[pos], wd_b[pos], u_b[pos], kappa, L_w, quad,
                    )
                neg = ~pos
                if bool(neg.any().item()):
                    rc2_mat = rc2_c.transpose(0, 1).expand_as(a)
                    fields = (
                        r[neg], a[neg], rho2[neg], wd_b[neg], u_b[neg],
                        torch.clamp(rc2_mat[neg], min=1e-24),
                        span_len.reshape(1, -1).expand_as(a)[neg],
                    )
                    n_neg = int(fields[1].shape[0])
                    v_n = torch.empty((n_neg, 3), dtype=torch.complex128, device=dev)
                    batch = max(1, _TAIL_CHUNK_BYTES // (4 * 64 * 16))
                    for j0 in range(0, n_neg, batch):
                        j1 = min(n_neg, j0 + batch)
                        r_j, a_j, rho2_j, d_j, u_j, rc2_j, sp_j = (f[j0:j1] for f in fields)
                        scale = torch.sqrt(rho2_j + rc2_j)
                        pad = torch.maximum(scale, sp_j)
                        xi_0 = -a_j + pad
                        xi_q, w_q = _graded_segment_nodes_gpu(
                            xi_0, -a_j.unsqueeze(-1), scale.unsqueeze(-1), quad,
                        )
                        v_seg = _displaced_ray_real_seg_gpu(
                            r_j, a_j, rho2_j, d_j, u_j, xi_q, w_q, kappa, L_w, rc2_j,
                        )
                        v_tail = _displaced_ray_contour_gpu(
                            r_j - xi_0.unsqueeze(-1) * d_j, a_j + xi_0, rho2_j, d_j, u_j,
                            kappa, L_w + xi_0, quad,
                        )
                        v_n[j0:j1] = v_seg + v_tail
                    v_ray[neg] = v_n

            v_tot = v_tot + sign * v_ray

        out[i0:i1] = v_tot

    return out


def displaced_steady_tail_velocity_gpu(
    P,
    p_left,
    p_right,
    wake_dir,
    u_left,
    u_right,
    kappa: float,
    wake_length: float,
    rc=0.0,
    *,
    ground=None,
    img_mode: int | None = None,
    out_tensor: bool = False,
):
    """Compute velocity perturbation from displaced steady far-wake filaments on the GPU.

    Parameters
    ----------
    P : numpy.ndarray or torch.Tensor
        Evaluation points [m], shape ``(M, 3)``.
    p_left, p_right : numpy.ndarray or torch.Tensor
        Left and right tail start corners [m], shape ``(N_s, 3)``.
    wake_dir : numpy.ndarray or torch.Tensor
        Unit wake direction [-], shape ``(3,)`` or ``(N_s, 3)``.
    u_left, u_right : numpy.ndarray or torch.Tensor
        Trailing-edge corner displacement vectors [m] (heave) or [m/rad]
        (pitch), shape ``(3,)`` or ``(N_s, 3)``.
    kappa : float
        Spatial wavenumber ``omega / V_inf`` [1/m], must be ``>= 0``.
    wake_length : float
        Discrete wake length ``L_w`` [m] ahead of the tail start, must be ``>= 0``.
    rc : float, numpy.ndarray or torch.Tensor, optional
        Core radius [m], scalar, shape ``(N_s,)`` or shape ``(M, N_s)``.
    ground : GroundPlane, tuple or None, optional
        Flat ground plane for ground images (exact reflection identity).
    img_mode : int or None, optional
        ``0`` real only, ``1`` real plus ground image, ``2`` ground image only.
    out_tensor : bool, optional
        When True return a CUDA :class:`torch.Tensor` (complex128); when False
        return a host complex128 :class:`numpy.ndarray`.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Complex linearised velocity perturbation per unit steady trailing-edge
        circulation [1/m] (heave) or [1/(m rad)] (pitch), shape ``(M, N_s, 3)``.
    """
    kap = float(kappa)
    if not math.isfinite(kap) or kap < 0.0:
        raise ValueError(f"kappa must be finite and >= 0, got {kappa!r}.")
    L_w = float(wake_length)
    if not math.isfinite(L_w) or L_w < 0.0:
        raise ValueError(f"wake_length must be finite and >= 0, got {wake_length!r}.")
    P64 = _as_cuda_f64(P)
    pl64 = _as_cuda_f64(p_left)
    pr64 = _as_cuda_f64(p_right)
    m, n = int(P64.shape[0]), int(pl64.shape[0])
    if m == 0 or n == 0 or kap == 0.0:
        out = torch.zeros((m, n, 3), dtype=torch.complex128, device=engine._dev())
        return out if out_tensor else out.cpu().numpy()

    if isinstance(wake_dir, torch.Tensor):
        wd64 = wake_dir.to(device=engine._dev(), dtype=torch.float64)
        if wd64.ndim == 1:
            wd64 = wd64.unsqueeze(0).expand(n, 3).contiguous()
        else:
            wd64 = wd64.contiguous()
    else:
        wd64 = engine._f64(_broadcast_wake_dir(wake_dir, n))
    wd_norms = torch.linalg.vector_norm(wd64, dim=1, keepdim=True)
    if bool((wd_norms <= 1e-12).any().item()):
        raise ValueError("wake_dir must have non-zero length for every strip.")
    wd64 = (wd64 / wd_norms).contiguous()

    ul64 = _as_cuda_disp(u_left, n, "u_left")
    ur64 = _as_cuda_disp(u_right, n, "u_right")
    rc2_T, rc_is_2d = _pack_rc2_gpu(rc, m, n, torch.float64)
    mode, gk_wp, go = _ground_params(ground, img_mode)

    out = None
    if mode != 2:
        out = _displaced_tail_velocity_single_gpu(P64, pl64, pr64, wd64, ul64, ur64, kap, L_w, rc2_T, rc_is_2d)
    if mode != 0:
        gk = torch.tensor([gk_wp[0], gk_wp[1], gk_wp[2]], dtype=torch.float64, device=engine._dev())
        P_img = P64 - (2.0 * (P64 @ gk - go)).unsqueeze(-1) * gk[None, :]
        vi = _displaced_tail_velocity_single_gpu(P_img, pl64, pr64, wd64, ul64, ur64, kap, L_w, rc2_T, rc_is_2d)
        gk_c = gk.to(torch.complex128)
        vi_ref = vi - (2.0 * torch.sum(vi * gk_c[None, None, :], dim=-1, keepdim=True)) * gk_c[None, None, :]
        out = vi_ref if out is None else (out + vi_ref)

    return out if out_tensor else out.cpu().numpy()


def displaced_steady_tail_normal_influence_gpu(
    P,
    normals,
    p_left,
    p_right,
    wake_dir,
    u_left,
    u_right,
    kappa: float,
    wake_length: float,
    rc=0.0,
    *,
    ground=None,
    img_mode: int | None = None,
    out_tensor: bool = False,
):
    """Compute normal-velocity perturbation from displaced steady far-wake filaments on the GPU.

    Parameters
    ----------
    P : numpy.ndarray or torch.Tensor
        Evaluation points [m], shape ``(M, 3)``.
    normals : numpy.ndarray or torch.Tensor
        Unit normal vectors at *P* [-], shape ``(M, 3)``.
    p_left, p_right : numpy.ndarray or torch.Tensor
        Left and right tail start corners [m], shape ``(N_s, 3)``.
    wake_dir : numpy.ndarray or torch.Tensor
        Unit wake direction [-], shape ``(3,)`` or ``(N_s, 3)``.
    u_left, u_right : numpy.ndarray or torch.Tensor
        Trailing-edge corner displacement vectors [m] (heave) or [m/rad]
        (pitch), shape ``(3,)`` or ``(N_s, 3)``.
    kappa : float
        Spatial wavenumber ``omega / V_inf`` [1/m], must be ``>= 0``.
    wake_length : float
        Discrete wake length ``L_w`` [m] ahead of the tail start, must be ``>= 0``.
    rc : float, numpy.ndarray or torch.Tensor, optional
        Core radius [m], scalar, shape ``(N_s,)`` or shape ``(M, N_s)``.
    ground : GroundPlane, tuple or None, optional
        Flat ground plane for ground images.
    img_mode : int or None, optional
        ``0`` real only, ``1`` real plus ground image, ``2`` ground image only.
    out_tensor : bool, optional
        When True return a CUDA :class:`torch.Tensor` (complex128); when False
        return a host complex128 :class:`numpy.ndarray`.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Complex normal-velocity perturbation matrix [1/m] (heave) or
        [1/(m rad)] (pitch), shape ``(M, N_s)``.
    """
    N_c = _as_cuda_f64(normals).to(torch.complex128)
    v = displaced_steady_tail_velocity_gpu(
        P, p_left, p_right, wake_dir, u_left, u_right, kappa, wake_length, rc,
        ground=ground, img_mode=img_mode, out_tensor=True,
    )
    out = torch.einsum("msi,mi->ms", v, N_c)
    return out if out_tensor else out.cpu().numpy()

