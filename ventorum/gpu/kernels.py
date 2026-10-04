# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Fused CUDA kernels of the batched lattice pipelines (NVIDIA Warp).

The kernels compute, for K flight conditions on one lattice in one launch:

* ``tensor``: the velocity per unit circulation of each unknown at each
  evaluation point (the velocity tensor) and its component along a normal
  (the influence matrix);
* ``normal``: the influence matrix only;
* ``induced``: the induced velocity at a set of points for a given
  circulation of every panel;
* ``trefftz``: the normal wash of the 2-D point vortices in the Trefftz plane;
* ``case_te``, ``case_points``, ``trefftz_prep``, ``llt_rhs``: the per-case
  data of a batch (wake-leg data, Trefftz-plane points, right-hand sides) in
  one launch each.

Physics and regularisation are the same as in the reference kernels of
:mod:`ventorum.aero.vortex`: the Biot-Savart law of the bent horseshoe (finite
segments and semi-infinite wake legs), the core radius ``rc`` with the
denominators ``|r1 x r2|^2 + rc^2 |r0|^2`` and ``|d x r|^2 + rc^2``, and the
cross-surface core of :class:`ventorum.aero.vortex.Targets`. The ground images
are made inside the kernels from the ground plane of each case (method of
images: the source points are mirrored in the plane, the circulation changes
sign).

Method: Biot-Savart law for straight vortex filaments (Katz, J. and Plotkin,
A., "Low-Speed Aerodynamics", 2nd ed., Cambridge University Press, 2001,
section 2.11; Anderson, J. D., "Fundamentals of Aerodynamics", 5th ed.,
McGraw-Hill, 2011). One GPU thread computes one matrix entry (or one point)
with all its sources, so no tensor of point-source pairs is stored.

Precision
---------
Each kernel is made for float32 or float64 (the type ``T``).

* float64: the formulas of the reference kernels.
* float32: three steps keep the digits that plain float32 arithmetic loses
  when the coordinates are large compared with the distances.

  1. A float64 value x is stored as a pair of float32 values (hi, lo) with
     ``x = hi + lo`` to about 1e-15. A difference ``x - y`` is made as
     ``(hi_x - hi_y) + (lo_x - lo_y)``: a float32 subtraction is exact to
     round-off relative to its result, so the difference keeps its digits
     also when x and y are almost equal (Dekker, T. J., "A floating-point
     technique for extending the available precision", Numerische
     Mathematik 18, 1971, pp. 224-242). The point-source vectors use this.
  2. The cross product ``d x r`` and the product ``d . r`` of a wake leg are
     differences of ``d x P``, ``d . P`` (per case and point) and ``d x TE``,
     ``d . TE`` (per case and trailing-edge point), made in float64 per case
     and stored as pairs; ``1 + cos(theta)`` uses a form without
     cancellation. A wake leg near an evaluation point (the wing wake near a
     tail) then keeps its distance to the point.
  3. The cross product ``r1 x r2`` of a bound vortex is ``(b - a) x (P - a)``
     in float64, so a point on the line of the bound vortex (the
     lifting-line control points, the force points of the loads) gives zero
     and not float32 rounding noise. The kernel cache computes this part once
     per lattice.

  Sums over many sources use float64 accumulators.

The module needs the ``warp-lang`` package (optional dependency ``gpu``).
"""

# No "from __future__ import annotations": Warp reads the type annotations of the
# kernels at definition time, with the types of the closure (T, vec).
import math
import threading
from types import SimpleNamespace

import warp as wp

_INV_4PI = 1.0 / (4.0 * math.pi)
_INV_2PI = 1.0 / (2.0 * math.pi)

# Floors of the denominators and of the squared norms, per type. float64 uses
# the values of the reference kernels; float32 uses a value that is
# representable (1e-300 is zero in float32).
_FLOORS = {"float64": 1e-300, "float32": 1e-30}

_lock = threading.Lock()
_built: dict[str, SimpleNamespace] = {}


def kernels(precision: str) -> SimpleNamespace:
    """Return the kernels and structs for *precision* (``"float32"`` or ``"float64"``); build them on the first call.

    Warp compiles the kernels on their first launch and keeps the binaries
    in its kernel cache on disk.
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


def _build(precision: str) -> SimpleNamespace:  # noqa: C901 - one closure per kernel type
    T = wp.float64 if precision == "float64" else wp.float32
    vec = wp.vec3d if precision == "float64" else wp.vec3f
    floor = _FLOORS[precision]
    is64 = precision == "float64"
    module = f"ventorum_gpu_{precision}"
    # No adjoint kernels (faster compile, smaller binaries).
    wp.set_module_options({"enable_backward": False}, wp.get_module(module))

    class Sources:
        """The panels as sources (the float32 pair arrays are used only by the float32 kernels)."""

        A: wp.array(dtype=wp.vec3d)      # bound vortex start a [m]
        B: wp.array(dtype=wp.vec3d)      # bound vortex end b [m]
        ATE: wp.array(dtype=wp.vec3d)    # trailing-edge point of the a leg [m]
        BTE: wp.array(dtype=wp.vec3d)    # trailing-edge point of the b leg [m]
        Ah: wp.array(dtype=wp.vec3f)     # a as a float32 pair (high part)
        Al: wp.array(dtype=wp.vec3f)     # a as a float32 pair (low part)
        r0b: wp.array(dtype=vec)         # b - a
        r0a: wp.array(dtype=vec)         # a - a_te
        r0t: wp.array(dtype=vec)         # b_te - b
        rc2: wp.array(dtype=T)           # squared core radius [m^2]
        group: wp.array(dtype=wp.int32)  # core group

    class Points:
        """Evaluation points (float64 and float32 pair), their normals, and their cross-surface core data."""

        P: wp.array(dtype=wp.vec3d)
        Ph: wp.array(dtype=wp.vec3f)
        Pl: wp.array(dtype=wp.vec3f)
        N: wp.array(dtype=vec)
        group: wp.array(dtype=wp.int32)
        rc2: wp.array(dtype=T)
        use_tg: int

    class Cases:
        """Per-case data: wake direction, ground plane, and the wake-leg data of the float32 kernels.

        The wake-leg data (float32 pairs) are ``d x P`` and ``d . P`` per
        point (K, m), and ``d x TE`` and ``d . TE`` per panel and
        trailing-edge point (K, n, 2), for the real horseshoes and for the
        images (``*_i``); ``Aih``, ``Ail`` are the image points of a (K, n).
        """

        D: wp.array(dtype=vec)
        GK: wp.array(dtype=wp.vec3d)
        GO: wp.array(dtype=wp.float64)
        n_img: int
        dxPh: wp.array2d(dtype=wp.vec3f)
        dxPl: wp.array2d(dtype=wp.vec3f)
        dPh: wp.array2d(dtype=wp.float32)
        dPl: wp.array2d(dtype=wp.float32)
        dxTh: wp.array3d(dtype=wp.vec3f)
        dxTl: wp.array3d(dtype=wp.vec3f)
        dTh: wp.array3d(dtype=wp.float32)
        dTl: wp.array3d(dtype=wp.float32)
        dxPh_i: wp.array2d(dtype=wp.vec3f)
        dxPl_i: wp.array2d(dtype=wp.vec3f)
        dPh_i: wp.array2d(dtype=wp.float32)
        dPl_i: wp.array2d(dtype=wp.float32)
        dxTh_i: wp.array3d(dtype=wp.vec3f)
        dxTl_i: wp.array3d(dtype=wp.vec3f)
        dTh_i: wp.array3d(dtype=wp.float32)
        dTl_i: wp.array3d(dtype=wp.float32)
        Aih: wp.array2d(dtype=wp.vec3f)
        Ail: wp.array2d(dtype=wp.vec3f)

    class Polars:
        """Section polars of the strips of the unknowns (the nonlinear lifting line).

        ``kind`` is 0 for a linear airfoil (``a0``, ``aL0``) and 1 for a
        tabulated polar (table ``tab``): breakpoints ``x`` (padded), their
        count ``npts``, and the cubic pieces of Cl and of dCl/dalpha
        (highest power first; the pieces of the slope have a leading zero).
        """

        kind: wp.array(dtype=wp.int32)
        tab: wp.array(dtype=wp.int32)
        a0: wp.array(dtype=wp.float64)
        aL0: wp.array(dtype=wp.float64)
        cd0: wp.array(dtype=wp.float64)
        cm0: wp.array(dtype=wp.float64)
        x: wp.array2d(dtype=wp.float64)
        npts: wp.array(dtype=wp.int32)
        has_cm: wp.array(dtype=wp.int32)
        ccl: wp.array3d(dtype=wp.float64)
        cdcl: wp.array3d(dtype=wp.float64)
        ccd: wp.array3d(dtype=wp.float64)
        ccm: wp.array3d(dtype=wp.float64)

    # One struct type per precision: the names carry the precision.
    Polars.__name__ = Polars.__qualname__ = f"Polars_{precision}"
    Polars = wp.struct(Polars)
    Sources.__name__ = Sources.__qualname__ = f"Sources_{precision}"
    Points.__name__ = Points.__qualname__ = f"Points_{precision}"
    Cases.__name__ = Cases.__qualname__ = f"Cases_{precision}"
    Sources = wp.struct(Sources)
    Points = wp.struct(Points)
    Cases = wp.struct(Cases)

    @wp.func(module=module)
    def tvec(x: wp.vec3d):
        return vec(T(x[0]), T(x[1]), T(x[2]))

    @wp.func(module=module)
    def rsqrt(x: T):
        # 1 / sqrt(x). float64: a float32 estimate and two Newton steps in
        # float64 (each step doubles the correct bits: 24, 48, 96), which
        # avoids the slow float64 square root and division of GPUs with a low
        # float64 rate; the result is within one or two units in the last
        # place. Outside the float32 range the exact formula is used.
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
        # 1 / x. float64: a float32 estimate and two Newton steps (see rsqrt).
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
    def seg(r1: vec, r2: vec, r0: vec, i1: T, i2: T, rc2: T):
        # Finite segment from the vectors to its two end points (r1, r2), its
        # direction r0 and the inverse norms of r1 and r2.
        c = wp.cross(r1, r2)
        proj = wp.dot(r0, r1 * i1 - r2 * i2)
        den = wp.dot(c, c) + rc2 * wp.dot(r0, r0)
        k = T(0.0)
        if den > T(floor):
            k = T(_INV_4PI) * proj * rcp(den)
        return c * k

    @wp.func(module=module)
    def seg_c(c: vec, r1: vec, r2: vec, r0: vec, i1: T, i2: T, rc2: T):
        # The same segment with the cross product c = r1 x r2 given.
        proj = wp.dot(r0, r1 * i1 - r2 * i2)
        den = wp.dot(c, c) + rc2 * wp.dot(r0, r0)
        k = T(0.0)
        if den > T(floor):
            k = T(_INV_4PI) * proj * rcp(den)
        return c * k

    @wp.func(module=module)
    def ray(r: vec, ir: T, d: vec, rc2: T):
        # Semi-infinite line from the point at -r along the unit vector d
        # (float64 formula of the reference kernels).
        c = wp.cross(d, r)
        den = wp.dot(c, c) + rc2
        k = T(0.0)
        if den > T(floor):
            k = T(_INV_4PI) * (T(1.0) + wp.dot(d, r) * ir) * rcp(den)
        return c * k

    @wp.func(module=module)
    def ray_pre(c: vec, dr: T, rn: T, rc2: T):
        # Semi-infinite line from c = d x r and dr = d . r (float32 kernels).
        # 1 + cos(theta) = |c|^2 / (|r| (|r| - d.r)) when d.r < 0 (no cancellation).
        cc = wp.dot(c, c)
        den = cc + rc2
        opc = T(1.0) + dr / rn
        if dr < T(0.0):
            opc = cc / (rn * (rn - dr))
        k = T(0.0)
        if den > T(floor):
            k = T(_INV_4PI) * opc / den
        return c * k

    @wp.func(module=module)
    def mirror_point(x: wp.vec3d, gk: wp.vec3d, go: wp.float64):
        return x - (wp.float64(2.0) * (wp.dot(x, gk) - go)) * gk

    @wp.func(module=module)
    def mirror_dir(v: vec, gk: vec):
        return v - (T(2.0) * wp.dot(v, gk)) * gk

    @wp.func(module=module)
    def diff2(ah: wp.vec3f, al: wp.vec3f, bh: wp.vec3f, bl: wp.vec3f):
        # (a - b) from float32 pairs.
        return (ah - bh) + (al - bl)

    @wp.func(module=module)
    def split3(x: wp.vec3d):
        # The float32 pair (hi, lo) of x: x = hi + lo to about 1e-15 relative.
        h = wp.vec3f(x)
        return h, wp.vec3f(x - wp.vec3d(h))

    @wp.func(module=module)
    def split1(x: wp.float64):
        h = wp.float32(x)
        return h, wp.float32(x - wp.float64(h))

    def make_horseshoe(PARTS: int, IMG: int):
        """Return the horseshoe function for the legs PARTS (0 all, 1 bound vortex and chordwise legs, 2 wake legs).

        IMG = 1 gives the ground image of the horseshoe (without the sign
        change). Both are compile-time constants: each variant computes only
        what it needs.
        """

        def hs(src: Sources, pts: Points, cs: Cases, k: int, i: int, s: int, rc2: T):
            d = cs.D[k]
            r1 = vec()
            r2 = vec()
            q1 = vec()
            w2 = vec()
            vb = vec()
            va = vec()
            vt = vec()
            cbound = vec()
            P = pts.P[i]
            if wp.static(is64):
                pa = src.A[s]
                pb = src.B[s]
                pat = src.ATE[s]
                pbt = src.BTE[s]
                if wp.static(IMG == 1):
                    gk = cs.GK[k]
                    go = cs.GO[k]
                    pa = mirror_point(pa, gk, go)
                    pb = mirror_point(pb, gk, go)
                    pat = mirror_point(pat, gk, go)
                    pbt = mirror_point(pbt, gk, go)
                    d = mirror_dir(d, gk)
                r1 = P - pa
                r2 = P - pb
                q1 = P - pat
                w2 = P - pbt
                vb = pb - pa
                va = pa - pat
                vt = pbt - pb
            else:
                vb = src.r0b[s]
                va = src.r0a[s]
                vt = src.r0t[s]
                if wp.static(IMG == 1):
                    gkt = tvec(cs.GK[k])
                    vb = mirror_dir(vb, gkt)
                    va = mirror_dir(va, gkt)
                    vt = mirror_dir(vt, gkt)
                    r1 = diff2(pts.Ph[i], pts.Pl[i], cs.Aih[k, s], cs.Ail[k, s])
                else:
                    r1 = diff2(pts.Ph[i], pts.Pl[i], src.Ah[s], src.Al[s])
                    if wp.static(PARTS != 2):
                        # (b - a) x (P - a) in float64 (see the module notes).
                        pa = src.A[s]
                        cbound = tvec(wp.cross(src.B[s] - pa, P - pa))
                r2 = r1 - vb
                q1 = r1 + va
                w2 = r2 - vt
            iq = inv_norm(q1)
            iw = inv_norm(w2)
            v = vec()
            if wp.static(PARTS != 2):
                i1 = inv_norm(r1)
                i2 = inv_norm(r2)
                if wp.static(is64 or IMG == 1):
                    v = seg(r1, r2, vb, i1, i2, rc2)
                else:
                    v = seg_c(cbound, r1, r2, vb, i1, i2, rc2)
                v = v + seg(q1, r1, va, iq, i1, rc2) + seg(r2, w2, vt, i2, iw, rc2)
            if wp.static(PARTS != 1):
                if wp.static(is64):
                    v = v + ray(w2, iw, d, rc2) - ray(q1, iq, d, rc2)
                else:
                    cb = vec()
                    ca = vec()
                    drb = T(0.0)
                    dra = T(0.0)
                    if wp.static(IMG == 0):
                        cb = diff2(cs.dxPh[k, i], cs.dxPl[k, i], cs.dxTh[k, s, 1], cs.dxTl[k, s, 1])
                        ca = diff2(cs.dxPh[k, i], cs.dxPl[k, i], cs.dxTh[k, s, 0], cs.dxTl[k, s, 0])
                        drb = (cs.dPh[k, i] - cs.dTh[k, s, 1]) + (cs.dPl[k, i] - cs.dTl[k, s, 1])
                        dra = (cs.dPh[k, i] - cs.dTh[k, s, 0]) + (cs.dPl[k, i] - cs.dTl[k, s, 0])
                    else:
                        cb = diff2(cs.dxPh_i[k, i], cs.dxPl_i[k, i], cs.dxTh_i[k, s, 1], cs.dxTl_i[k, s, 1])
                        ca = diff2(cs.dxPh_i[k, i], cs.dxPl_i[k, i], cs.dxTh_i[k, s, 0], cs.dxTl_i[k, s, 0])
                        drb = (cs.dPh_i[k, i] - cs.dTh_i[k, s, 1]) + (cs.dPl_i[k, i] - cs.dTl_i[k, s, 1])
                        dra = (cs.dPh_i[k, i] - cs.dTh_i[k, s, 0]) + (cs.dPl_i[k, i] - cs.dTl_i[k, s, 0])
                    v = v + ray_pre(cb, drb, T(1.0) / iw, rc2) - ray_pre(ca, dra, T(1.0) / iq, rc2)
            return v

        return wp.func(hs, name=f"hs_{precision}_{PARTS}{IMG}", module=module)

    hs_full = make_horseshoe(0, 0)
    hs_fixed = make_horseshoe(1, 0)
    hs_wake = make_horseshoe(2, 0)
    hs_img = make_horseshoe(0, 1)
    hs_img_fixed = make_horseshoe(1, 1)
    hs_img_wake = make_horseshoe(2, 1)

    @wp.func(module=module)
    def core2(src: Sources, s: int, pts: Points, i: int):
        c = src.rc2[s]
        if pts.use_tg != 0:
            if pts.group[i] != src.group[s]:
                c = c + pts.rc2[i]
        return c

    def make_entry(MODE: int):
        """Return the sum over the sources of unknown j at point i (case k) for MODE.

        MODE 0: all legs of the real horseshoes and of the images; 1: the
        bound vortices and chordwise legs of the real horseshoes (the kernel
        cache); 2: the wake legs of the real horseshoes and all legs of the
        images (the part that changes with the case).
        """

        def entry(src: Sources, pts: Points, cs: Cases, src_of: wp.array2d(dtype=wp.int32), k: int, i: int, j: int):
            acc = vec()
            for q in range(src_of.shape[1]):
                s = src_of[j, q]
                c2 = core2(src, s, pts, i)
                if wp.static(MODE == 0):
                    acc = acc + hs_full(src, pts, cs, k, i, s, c2)
                elif wp.static(MODE == 1):
                    acc = acc + hs_fixed(src, pts, cs, k, i, s, c2)
                else:
                    acc = acc + hs_wake(src, pts, cs, k, i, s, c2)
            if wp.static(MODE != 1):
                if cs.n_img != 0:
                    for q in range(src_of.shape[1]):
                        s = src_of[j, q]
                        acc = acc - hs_img(src, pts, cs, k, i, s, core2(src, s, pts, i))
            return acc

        return wp.func(entry, name=f"entry_{precision}_{MODE}", module=module)

    entries = {mode: make_entry(mode) for mode in (0, 1, 2)}

    # The matrices are stored transposed (column-major per case): entry (i, j)
    # of case k is at [k, j, i], and the thread index is (k, j, i). The
    # threads of one warp then have the same unknown j (the same sources: one
    # read for the warp) and consecutive points i (contiguous reads and
    # writes), and the residual kernel reads the matrix contiguously.
    def make_tensor(MODE: int):
        f = entries[MODE]

        def tensor(src: Sources, pts: Points, cs: Cases, src_of: wp.array2d(dtype=wp.int32),
                   fixed_vT: wp.array2d(dtype=vec), store_v: int, out_vT: wp.array3d(dtype=vec),
                   out_nT: wp.array3d(dtype=T)):
            # Velocity per unit circulation of unknown j at point i (case k), and
            # its component along pts.N[i]. MODE 2 adds the cached fixed part.
            k, j, i = wp.tid()
            acc = f(src, pts, cs, src_of, k, i, j)
            if wp.static(MODE == 2):
                acc = fixed_vT[j, i] + acc
            if store_v != 0:
                out_vT[k, j, i] = acc
            out_nT[k, j, i] = wp.dot(acc, pts.N[i])

        return wp.kernel(tensor, name=f"tensor_{precision}_{MODE}", module=module)

    def make_normal(MODE: int):
        def normal(src: Sources, pts: Points, cs: Cases, src_of: wp.array2d(dtype=wp.int32),
                   fixed_nT: wp.array2d(dtype=T), out_nT: wp.array3d(dtype=T)):
            # Normal-velocity influence of unknown j at point i (case k), source by
            # source (the same sum as `tensor`, projected on pts.N[i] per source).
            k, j, i = wp.tid()
            n = pts.N[i]
            acc = T(0.0)
            for q in range(src_of.shape[1]):
                s = src_of[j, q]
                c2 = core2(src, s, pts, i)
                if wp.static(MODE == 0):
                    acc = acc + wp.dot(hs_full(src, pts, cs, k, i, s, c2), n)
                elif wp.static(MODE == 1):
                    acc = acc + wp.dot(hs_fixed(src, pts, cs, k, i, s, c2), n)
                else:
                    acc = acc + wp.dot(hs_wake(src, pts, cs, k, i, s, c2), n)
            if wp.static(MODE != 1):
                if cs.n_img != 0:
                    for q in range(src_of.shape[1]):
                        s = src_of[j, q]
                        acc = acc - wp.dot(hs_img(src, pts, cs, k, i, s, core2(src, s, pts, i)), n)
            if wp.static(MODE == 2):
                acc = fixed_nT[j, i] + acc
            out_nT[k, j, i] = acc

        return wp.kernel(normal, name=f"normal_{precision}_{MODE}", module=module)

    tensors = {mode: make_tensor(mode) for mode in (0, 1, 2)}
    normals = {mode: make_normal(mode) for mode in (0, 1, 2)}

    @wp.kernel(module=module)
    def induced(src: Sources, pts: Points, cs: Cases, G: wp.array2d(dtype=wp.float64),
                out: wp.array2d(dtype=wp.vec3d)):
        # Induced velocity at point i of case k from all panels s with the
        # circulation G[k, s] (and their ground images with -G[k, s]).
        k, i = wp.tid()
        acc = wp.vec3d()
        n_src = src.A.shape[0]
        for s in range(n_src):
            g = G[k, s]
            if g != wp.float64(0.0):
                v = hs_full(src, pts, cs, k, i, s, core2(src, s, pts, i))
                acc = acc + wp.vec3d(wp.float64(v[0]) * g, wp.float64(v[1]) * g, wp.float64(v[2]) * g)
        if cs.n_img != 0:
            for s in range(n_src):
                g = G[k, s]
                if g != wp.float64(0.0):
                    v = hs_img(src, pts, cs, k, i, s, core2(src, s, pts, i))
                    acc = acc - wp.vec3d(wp.float64(v[0]) * g, wp.float64(v[1]) * g, wp.float64(v[2]) * g)
        out[k, i] = acc

    @wp.kernel(module=module)
    def wake_table(src: Sources, pts: Points, cs: Cases, rep_panel: wp.array(dtype=wp.int32),
                   img: int, out: wp.array3d(dtype=T)):
        # Normal velocity at point i (case k) of the wake legs of wake group w
        # per unit circulation (out[k, w, i]; img = 1: the ground image legs).
        # The panels of a wake group (one strip, one core radius) share their
        # wake legs, so the matrix needs them once per group, not per panel.
        k, w, i = wp.tid()
        s = rep_panel[w]
        v = vec()
        if img == 0:
            v = hs_wake(src, pts, cs, k, i, s, core2(src, s, pts, i))
        else:
            v = hs_img_wake(src, pts, cs, k, i, s, core2(src, s, pts, i))
        out[k, w, i] = wp.dot(v, pts.N[i])

    @wp.kernel(module=module)
    def normal_from_table(src: Sources, pts: Points, cs: Cases, src_of: wp.array2d(dtype=wp.int32),
                          wgroup: wp.array(dtype=wp.int32), fixed_nT: wp.array2d(dtype=T),
                          table: wp.array3d(dtype=T), table_img: wp.array3d(dtype=T), out_nT: wp.array3d(dtype=T)):
        # The influence matrix of mode 2 (transposed, out_nT[k, j, i]): the
        # cached fixed legs, the wake legs from the wake table, and the ground
        # images (their bound vortices and chordwise legs per panel, their wake
        # legs from the image wake table).
        k, j, i = wp.tid()
        acc = T(0.0)
        for q in range(src_of.shape[1]):
            acc = acc + table[k, wgroup[src_of[j, q]], i]
        if cs.n_img != 0:
            n = pts.N[i]
            for q in range(src_of.shape[1]):
                s = src_of[j, q]
                acc = acc - (wp.dot(hs_img_fixed(src, pts, cs, k, i, s, core2(src, s, pts, i)), n)
                             + table_img[k, wgroup[s], i])
        out_nT[k, j, i] = fixed_nT[j, i] + acc

    @wp.kernel(module=module)
    def fixed_points(src: Sources, pts: Points, cs: Cases, cols: wp.array2d(dtype=wp.int32),
                     out: wp.array3d(dtype=wp.float64)):
        # Velocity of the bound vortices and chordwise legs at point i per unit
        # circulation of column c (the sum of the panels cols[c, :]), stored as
        # out[i, component, c] (the matrix of a product with the circulation).
        i, c = wp.tid()
        acc = vec()
        for q in range(cols.shape[1]):
            s = cols[c, q]
            acc = acc + hs_fixed(src, pts, cs, 0, i, s, core2(src, s, pts, i))
        out[i, 0, c] = wp.float64(acc[0])
        out[i, 1, c] = wp.float64(acc[1])
        out[i, 2, c] = wp.float64(acc[2])

    @wp.kernel(module=module)
    def induced_wakes(src: Sources, pts: Points, cs: Cases, rep_panel: wp.array(dtype=wp.int32),
                      Gw: wp.array2d(dtype=wp.float64), out: wp.array2d(dtype=wp.vec3d)):
        # Add the wake legs of all wake groups to out[k, i]: the panels of one
        # group (one strip, one core radius) share their trailing-edge points,
        # so one panel (rep_panel[w]) with the summed circulation Gw[k, w]
        # stands for the group (ground images with -Gw).
        k, i = wp.tid()
        acc = out[k, i]
        for w in range(rep_panel.shape[0]):
            g = Gw[k, w]
            if g != wp.float64(0.0):
                s = rep_panel[w]
                v = hs_wake(src, pts, cs, k, i, s, core2(src, s, pts, i))
                acc = acc + wp.vec3d(wp.float64(v[0]) * g, wp.float64(v[1]) * g, wp.float64(v[2]) * g)
        if cs.n_img != 0:
            for w in range(rep_panel.shape[0]):
                g = Gw[k, w]
                if g != wp.float64(0.0):
                    s = rep_panel[w]
                    v = hs_img_wake(src, pts, cs, k, i, s, core2(src, s, pts, i))
                    acc = acc - wp.vec3d(wp.float64(v[0]) * g, wp.float64(v[1]) * g, wp.float64(v[2]) * g)
        out[k, i] = acc

    @wp.kernel(module=module)
    def induced_img_fixed(src: Sources, pts: Points, cs: Cases, G: wp.array2d(dtype=wp.float64),
                          out: wp.array2d(dtype=wp.vec3d)):
        # Add the bound vortices and chordwise legs of the ground images (-G[k, s]) to out[k, i].
        k, i = wp.tid()
        acc = out[k, i]
        for s in range(src.A.shape[0]):
            g = G[k, s]
            if g != wp.float64(0.0):
                v = hs_img_fixed(src, pts, cs, k, i, s, core2(src, s, pts, i))
                acc = acc - wp.vec3d(wp.float64(v[0]) * g, wp.float64(v[1]) * g, wp.float64(v[2]) * g)
        out[k, i] = acc

    @wp.func(module=module)
    def polar_cl(pol: Polars, i: int, a: wp.float64):
        # Section lift coefficient and its slope at the angle a [rad] (strip i):
        # the formulas of the CPU path (linear airfoil, or the cubic pieces of
        # the shape-preserving interpolation with the end values outside the
        # table and a zero slope there).
        if pol.kind[i] == 0:
            return wp.vec2d(pol.a0[i] * (a - pol.aL0[i]), pol.a0[i])
        t = pol.tab[i]
        n = pol.npts[t]
        lo = pol.x[t, 0]
        hi = pol.x[t, n - 1]
        ac = wp.min(wp.max(a, lo), hi)
        il = wp.int32(0)
        ih = n - 1
        while ih - il > 1:
            mid = (il + ih) // 2
            if pol.x[t, mid] <= ac:
                il = mid
            else:
                ih = mid
        sx = ac - pol.x[t, il]
        cl = ((pol.ccl[t, il, 0] * sx + pol.ccl[t, il, 1]) * sx + pol.ccl[t, il, 2]) * sx + pol.ccl[t, il, 3]
        dcl = wp.float64(0.0)
        if a >= lo and a <= hi:
            dcl = ((pol.cdcl[t, il, 0] * sx + pol.cdcl[t, il, 1]) * sx + pol.cdcl[t, il, 2]) * sx + pol.cdcl[t, il, 3]
        return wp.vec2d(cl, dcl)

    @wp.func(module=module)
    def polar_cd_cm(pol: Polars, i: int, a: wp.float64):
        # Section drag and moment coefficients at the angle a [rad] (strip i):
        # Cd0 and Cm0 of a linear airfoil, or the cubic pieces of the table
        # with the end values outside it (no Cm data gives zero).
        if pol.kind[i] == 0:
            return wp.vec2d(pol.cd0[i], pol.cm0[i])
        t = pol.tab[i]
        n = pol.npts[t]
        ac = wp.min(wp.max(a, pol.x[t, 0]), pol.x[t, n - 1])
        il = wp.int32(0)
        ih = n - 1
        while ih - il > 1:
            mid = (il + ih) // 2
            if pol.x[t, mid] <= ac:
                il = mid
            else:
                ih = mid
        sx = ac - pol.x[t, il]
        cd = ((pol.ccd[t, il, 0] * sx + pol.ccd[t, il, 1]) * sx + pol.ccd[t, il, 2]) * sx + pol.ccd[t, il, 3]
        cm = wp.float64(0.0)
        if pol.has_cm[t] != 0:
            cm = ((pol.ccm[t, il, 0] * sx + pol.ccm[t, il, 1]) * sx + pol.ccm[t, il, 2]) * sx + pol.ccm[t, il, 3]
        return wp.vec2d(cd, cm)

    @wp.func(module=module)
    def nl_residual(W: wp.vec3d, g: wp.float64, sc: wp.float64, dlv: wp.vec3d, nv: wp.vec3d, av: wp.vec3d,
                    pol: Polars, i: int):
        # Residual R_i = 2 |W x dl| G - V^2 dA Cl(alpha) of one strip.
        wx = wp.cross(W, dlv)
        al = wp.atan2(wp.dot(W, nv), wp.dot(W, av))
        c = polar_cl(pol, i, al)
        return wp.float64(2.0) * wp.length(wx) * g - sc * c[0]

    @wp.kernel(module=module)
    def nl_state(W: wp.array2d(dtype=wp.vec3d), g: wp.array2d(dtype=wp.float64), dl: wp.array(dtype=wp.vec3d),
                 nrm: wp.array(dtype=wp.vec3d), adir: wp.array(dtype=wp.vec3d), scale: wp.array2d(dtype=wp.float64),
                 pol: Polars, R: wp.array2d(dtype=wp.float64), wxn: wp.array2d(dtype=wp.float64),
                 alpha: wp.array2d(dtype=wp.float64), c_an: wp.array2d(dtype=wp.float64),
                 c_va: wp.array2d(dtype=wp.float64), d2: wp.array2d(dtype=wp.float64),
                 wvec: wp.array2d(dtype=wp.vec3d)):
        # The state of strip i of case k (solve_llt_nonlinear, evaluate):
        # |W x dl|, alpha, the residual R = 2 |W x dl| G - V^2 dA Cl, the
        # Jacobian terms c_an = V^2 dA dCl (W . a) / |W|^2, c_va = V^2 dA dCl
        # (W . n) / |W|^2 and w = dl x u with u = (W x dl) / |W x dl|, and the
        # damped fixed-point direction d2 = V^2 dA Cl / (2 |W x dl|) - G.
        k, i = wp.tid()
        Wk = W[k, i]
        wx = wp.cross(Wk, dl[i])
        n_ = wp.length(wx)
        a_n = wp.dot(Wk, nrm[i])
        a_a = wp.dot(Wk, adir[i])
        al = wp.atan2(a_n, a_a)
        c = polar_cl(pol, i, al)
        sc = scale[k, i]
        gi = g[k, i]
        wxn[k, i] = n_
        alpha[k, i] = al
        R[k, i] = wp.float64(2.0) * n_ * gi - sc * c[0]
        fac = sc * c[1] / wp.max(a_a * a_a + a_n * a_n, wp.float64(1e-300))
        c_an[k, i] = fac * a_a
        c_va[k, i] = fac * a_n
        d2[k, i] = sc * c[0] / (wp.float64(2.0) * wp.max(n_, wp.float64(1e-300))) - gi
        u = wx / wp.max(n_, wp.float64(1e-300))
        wvec[k, i] = wp.cross(dl[i], u)

    @wp.kernel(module=module)
    def nl_reduce(R: wp.array2d(dtype=wp.float64), scale: wp.array2d(dtype=wp.float64), tol: wp.float64,
                  first: int, accepted: wp.array(dtype=wp.int32), active0: wp.array(dtype=wp.int32),
                  conv: wp.array(dtype=wp.int32), active: wp.array(dtype=wp.int32),
                  f0: wp.array(dtype=wp.float64), res: wp.array(dtype=wp.float64)):
        # The merit 0.5 sum (R / (V^2 dA))^2 and the residual max |R| / (V^2 dA)
        # of case k, and its flags: converged, and still iterating (a case with
        # no accepted trial stops; its state did not change).
        k = wp.tid()
        m = wp.float64(0.0)
        r = wp.float64(0.0)
        for i in range(R.shape[1]):
            q = R[k, i] / scale[k, i]
            m = m + q * q
            r = wp.max(r, wp.abs(q))
        f0[k] = wp.float64(0.5) * m
        res[k] = r
        if first != 0:
            c = wp.int32(0)
            if r < tol:
                c = wp.int32(1)
            conv[k] = c
            if c == 0 and active0[k] != 0:
                active[k] = 1
            else:
                active[k] = 0
        else:
            if accepted[k] != 0 and r < tol:
                conv[k] = 1
            if accepted[k] != 0 and conv[k] == 0:
                active[k] = 1
            else:
                active[k] = 0

    @wp.kernel(module=module)
    def nl_update(mer: wp.array2d(dtype=wp.float64), f0: wp.array(dtype=wp.float64),
                  lam: wp.array(dtype=wp.float64), sel: wp.array(dtype=wp.int32), n_ls: int,
                  has_step: wp.array(dtype=wp.int32), active: wp.array(dtype=wp.int32),
                  step: wp.array2d(dtype=wp.float64), d2: wp.array2d(dtype=wp.float64),
                  U1: wp.array2d(dtype=wp.vec3d), U2: wp.array2d(dtype=wp.vec3d), g: wp.array2d(dtype=wp.float64),
                  W: wp.array2d(dtype=wp.vec3d), accepted: wp.array(dtype=wp.int32),
                  iters: wp.array(dtype=wp.int32)):
        # The first trial of case k that the rules of solve_llt_nonlinear accept
        # (the Newton step lengths with the sufficient-decrease test, then the
        # damped fixed-point steps with a lower merit), and the new circulation
        # and velocity of strip i.
        k, i = wp.tid()
        ok = wp.int32(0)
        t_step = wp.float64(0.0)
        t_fp = wp.float64(0.0)
        if active[k] != 0:
            f = f0[k]
            for l in range(mer.shape[1]):
                if ok == 0:
                    acc = wp.int32(0)
                    if l < n_ls:
                        if has_step[k] != 0 and mer[k, l] <= (wp.float64(1.0) - wp.float64(1.0e-4) * lam[l]) * f:
                            acc = wp.int32(1)
                    else:
                        if mer[k, l] < f:
                            acc = wp.int32(1)
                    if acc != 0:
                        ok = wp.int32(1)
                        if sel[l] == 0:
                            t_step = lam[l]
                        else:
                            t_fp = lam[l]
        if ok != 0:
            g[k, i] = g[k, i] + t_step * step[k, i] + t_fp * d2[k, i]
            W[k, i] = W[k, i] + t_step * U1[k, i] + t_fp * U2[k, i]
        if i == 0:
            accepted[k] = ok
            if active[k] != 0:
                iters[k] = iters[k] + 1

    @wp.kernel(module=module)
    def nl_trials(W0: wp.array2d(dtype=wp.vec3d), U1: wp.array2d(dtype=wp.vec3d), U2: wp.array2d(dtype=wp.vec3d),
                  g: wp.array2d(dtype=wp.float64), d1: wp.array2d(dtype=wp.float64), d2: wp.array2d(dtype=wp.float64),
                  lam: wp.array(dtype=wp.float64), sel: wp.array(dtype=wp.int32),
                  dl: wp.array(dtype=wp.vec3d), nrm: wp.array(dtype=wp.vec3d), adir: wp.array(dtype=wp.vec3d),
                  scale: wp.array2d(dtype=wp.float64), pol: Polars, rr: wp.array3d(dtype=wp.float64)):
        # The squared scaled residual (R / (V^2 dA))^2 of strip i at the trial
        # point l of case k: the circulation g + lam d and the velocity
        # W0 + lam U, with (d, U) the Newton step (sel 0) or the fixed-point
        # direction (sel 1). The velocity is linear in the circulation, so one
        # product per direction serves all trials.
        k, l, i = wp.tid()
        t = lam[l]
        Wt = W0[k, i]
        gt = g[k, i]
        if sel[l] == 0:
            Wt = Wt + t * U1[k, i]
            gt = gt + t * d1[k, i]
        else:
            Wt = Wt + t * U2[k, i]
            gt = gt + t * d2[k, i]
        r = nl_residual(Wt, gt, scale[k, i], dl[i], nrm[i], adir[i], pol, i) / scale[k, i]
        rr[k, l, i] = r * r

    @wp.kernel(module=module)
    def nl_merit(rr: wp.array3d(dtype=wp.float64), merit: wp.array2d(dtype=wp.float64)):
        # Merit 0.5 sum_i (R_i / (V^2 dA_i))^2 of the trial point l of case k
        # (the strips in order, so the sum does not depend on the launch).
        k, l = wp.tid()
        m = wp.float64(0.0)
        for i in range(rr.shape[2]):
            m = m + rr[k, l, i]
        merit[k, l] = wp.float64(0.5) * m

    @wp.kernel(module=module)
    def nl_jacobian(cases: wp.array(dtype=wp.int32), VtT: wp.array3d(dtype=vec), AnT: wp.array3d(dtype=T),
                    VaT: wp.array3d(dtype=T), g: wp.array2d(dtype=wp.float64), wxn: wp.array2d(dtype=wp.float64),
                    c_an: wp.array2d(dtype=wp.float64), c_va: wp.array2d(dtype=wp.float64),
                    wvec: wp.array2d(dtype=wp.vec3d), JT: wp.array3d(dtype=T)):
        # Transposed Jacobian of the case cases[c]: J[i, j] = 2 G_i (v_ij . w_i)
        # - c_an[i] (v_ij . n_i) + c_va[i] (v_ij . a_i) + diag(2 |W x dl|) (the
        # terms of solve_llt_nonlinear; c_an, c_va and w from nl_state).
        c, j, i = wp.tid()
        k = cases[c]
        v = (T(wp.float64(2.0) * g[k, i]) * wp.dot(VtT[k, j, i], tvec(wvec[k, i])) - T(c_an[k, i]) * AnT[k, j, i]
             + T(c_va[k, i]) * VaT[k, j, i])
        if i == j:
            v = v + T(wp.float64(2.0) * wxn[k, i])
        JT[c, j, i] = v

    @wp.kernel(module=module)
    def loads_strip(G: wp.array2d(dtype=wp.float64), vmid: wp.array2d(dtype=wp.vec3d),
                    vla: wp.array2d(dtype=wp.vec3d), vlb: wp.array2d(dtype=wp.vec3d), use_leg: int,
                    D: wp.array(dtype=wp.vec3d), V: wp.array(dtype=wp.float64), rho: wp.array(dtype=wp.float64),
                    rp: wp.vec3d, lvec: wp.array(dtype=wp.vec3d), mid: wp.array(dtype=wp.vec3d),
                    lega: wp.array(dtype=wp.vec3d), legb: wp.array(dtype=wp.vec3d),
                    lmida: wp.array(dtype=wp.vec3d), lmidb: wp.array(dtype=wp.vec3d), n_c: int,
                    chord: wp.array(dtype=wp.float64), width: wp.array(dtype=wp.float64),
                    nrm: wp.array(dtype=wp.vec3d), cdir: wp.array(dtype=wp.vec3d), dl: wp.array(dtype=wp.vec3d),
                    qc: wp.array(dtype=wp.vec3d), a0: wp.array(dtype=wp.float64), aL0: wp.array(dtype=wp.float64),
                    ae_in: wp.array2d(dtype=wp.float64), has_ae: int, pol: Polars,
                    wn: wp.array2d(dtype=wp.float64), segl: wp.array2d(dtype=wp.float64),
                    span: wp.array3d(dtype=wp.float64), sforce: wp.array2d(dtype=wp.vec3d),
                    part: wp.array3d(dtype=wp.float64)):
        # The loads of strip s of case k (the equations of compute_loads_batch):
        # Kutta-Joukowski forces on the bound vortices (and on the chordwise
        # legs with use_leg), their moments about rp, the section values, the
        # profile drag and the section moment, and the Trefftz-plane drag of
        # the strip. span[k, :, s] gets the 8 spanwise arrays; part[k, s, :]
        # the parts of the totals (force 3, moment 3, profile force 3,
        # profile drag, induced drag term, profile flag).
        k, s = wp.tid()
        d = D[k]
        Vk = V[k]
        rk = rho[k]
        q = wp.float64(0.5) * rk * Vk * Vk
        Vd = Vk * d
        sg = wp.float64(0.0)
        F = wp.vec3d()
        M = wp.vec3d()
        for r in range(n_c):
            p = s * n_c + r
            g = G[k, p]
            sg = sg + g
            Fp = (rk * g) * wp.cross(Vd + vmid[k, p], lvec[p])
            M = M + wp.cross(mid[p] - rp, Fp)
            if use_leg != 0:
                Fa = (rk * g) * wp.cross(Vd + vla[k, p], lega[p])
                Fb = (rk * g) * wp.cross(Vd + vlb[k, p], legb[p])
                M = M + wp.cross(lmida[p] - rp, Fa) + wp.cross(lmidb[p] - rp, Fb)
                Fp = Fp + Fa + Fb
            F = F + Fp
        c = chord[s]
        w = width[s]
        cl = wp.float64(2.0) * sg / (Vk * c)
        ag = wp.atan2(wp.dot(nrm[s], d), wp.dot(cdir[s], d))
        ae = aL0[s] + cl / wp.max(a0[s], wp.float64(1e-9))
        if has_ae != 0:
            ae = ae_in[k, s]
        cdm = polar_cd_cm(pol, s, ae)
        Dp = q * c * w * cdm[0]
        Fprof = Dp * d
        M = M + wp.cross(qc[s] - rp, Fprof)
        nl = wp.cross(d, dl[s])
        nl = nl / wp.max(wp.length(nl), wp.float64(1e-300))
        ax = wp.cross(nl, d)
        ax = ax / wp.max(wp.length(ax), wp.float64(1e-300))
        M = M + (q * c * c * w * cdm[1]) * ax
        di = sg * wn[k, s] * segl[k, s]
        span[k, 0, s] = sg
        span[k, 1, s] = cl
        span[k, 2, s] = -di / (Vk * Vk * c * w)
        span[k, 3, s] = cdm[0]
        span[k, 4, s] = ae
        span[k, 5, s] = ag - ae
        span[k, 6, s] = rk * Vk * sg
        span[k, 7, s] = cdm[1]
        sforce[k, s] = F
        for j in range(3):
            part[k, s, j] = F[j]
            part[k, s, 3 + j] = M[j]
            part[k, s, 6 + j] = Fprof[j]
        part[k, s, 9] = Dp
        part[k, s, 10] = di
        flag = wp.float64(0.0)
        if cdm[0] != wp.float64(0.0):
            flag = wp.float64(1.0)
        part[k, s, 11] = flag

    @wp.kernel(module=module)
    def loads_totals(part: wp.array3d(dtype=wp.float64), D: wp.array(dtype=wp.vec3d),
                     V: wp.array(dtype=wp.float64), rho: wp.array(dtype=wp.float64),
                     alpha: wp.array(dtype=wp.float64), S_ref: wp.float64, b_ref: wp.float64, c_ref: wp.float64,
                     span: wp.array3d(dtype=wp.float64), surf0: wp.array(dtype=wp.int32),
                     surf1: wp.array(dtype=wp.int32), out: wp.array2d(dtype=wp.float64)):
        # Totals of case k: the sums of the strip parts, the coefficients of
        # compute_loads_batch, and the statistics of the trust score (all
        # spanwise values finite, peak |Cl|, peak |alpha_eff| [deg], most
        # extrema of the circulation on one surface). out[k] = CL, CDi, CDp,
        # CD_total, e, Cl, Cm, Cn, CY, CDi_nearfield, profile flag, AR,
        # F_total (3), M (3), finite, max |Cl|, max |alpha_eff| [deg], extrema.
        k = wp.tid()
        n_s = part.shape[1]
        F = wp.vec3d()
        M = wp.vec3d()
        Fp = wp.vec3d()
        Dp = wp.float64(0.0)
        Di = wp.float64(0.0)
        hp = wp.float64(0.0)
        for s in range(n_s):
            F = F + wp.vec3d(part[k, s, 0], part[k, s, 1], part[k, s, 2])
            M = M + wp.vec3d(part[k, s, 3], part[k, s, 4], part[k, s, 5])
            Fp = Fp + wp.vec3d(part[k, s, 6], part[k, s, 7], part[k, s, 8])
            Dp = Dp + part[k, s, 9]
            Di = Di + part[k, s, 10]
            hp = wp.max(hp, part[k, s, 11])
        d = D[k]
        Vk = V[k]
        qS = wp.float64(0.5) * rho[k] * Vk * Vk * S_ref
        sa = wp.sin(alpha[k])
        ca = wp.cos(alpha[k])
        L_dir = wp.vec3d(-sa, wp.float64(0.0), ca)
        S_dir = wp.cross(L_dir, d)
        AR = b_ref * b_ref / S_ref
        Ft = F + Fp
        CL = wp.dot(Ft, L_dir) / qS
        CDi = (wp.float64(-0.5) * rho[k] * Di) / qS
        CDp = Dp / qS
        e = wp.float64(0.0)
        if CDi > wp.float64(1e-12):
            e = CL * CL / (wp.float64(3.141592653589793) * AR * CDi)
        else:
            e = wp.float64(0.0) / wp.float64(0.0)
        out[k, 0] = CL
        out[k, 1] = CDi
        out[k, 2] = CDp
        out[k, 3] = CDi + hp * CDp
        out[k, 4] = e
        out[k, 5] = -M[0] / (qS * b_ref)
        out[k, 6] = M[1] / (qS * c_ref)
        out[k, 7] = -M[2] / (qS * b_ref)
        out[k, 8] = wp.dot(Ft, S_dir) / qS
        out[k, 9] = wp.dot(F, d) / qS
        out[k, 10] = hp
        out[k, 11] = AR
        for j in range(3):
            out[k, 12 + j] = Ft[j]
            out[k, 15 + j] = M[j]
        # Statistics of the trust score.
        finite = wp.float64(1.0)
        mcl = wp.float64(0.0)
        mae = wp.float64(0.0)
        n_ext = wp.int32(0)
        for f in range(surf0.shape[0]):
            s0 = surf0[f]
            s1 = surf1[f]
            gmax = wp.float64(0.0)
            for s in range(s0, s1):
                for j in range(8):
                    v = span[k, j, s]
                    if not wp.isfinite(v):
                        finite = wp.float64(0.0)
                mcl = wp.max(mcl, wp.abs(span[k, 1, s]))
                mae = wp.max(mae, wp.abs(span[k, 4, s]))
                gmax = wp.max(gmax, wp.abs(span[k, 0, s]))
            if s1 - s0 >= 4:
                tol = wp.float64(1e-6) * wp.max(gmax, wp.float64(1e-300))
                last = wp.int32(0)
                cnt = wp.int32(0)
                for s in range(s0, s1 - 1):
                    dg = span[k, 0, s + 1] - span[k, 0, s]
                    if wp.abs(dg) > tol:
                        sgn = wp.int32(1)
                        if dg < wp.float64(0.0):
                            sgn = -1
                        if last != 0 and sgn != last:
                            cnt = cnt + 1
                        last = sgn
                n_ext = wp.max(n_ext, cnt)
        out[k, 18] = finite
        out[k, 19] = mcl
        out[k, 20] = mae * wp.float64(57.29577951308232)
        out[k, 21] = wp.float64(n_ext)

    @wp.kernel(module=module)
    def residual(MT: wp.array3d(dtype=T), x: wp.array2d(dtype=wp.float64), b: wp.array2d(dtype=wp.float64),
                 diag: wp.array2d(dtype=wp.float64), mode: int, out: wp.array2d(dtype=wp.float64)):
        # r = b - A x for A = M (mode 0) or A = diag(diag) - M (mode 1), with a
        # float64 accumulator (the entries of M are exact in their type). MT is
        # the transpose of M (MT[k, j, i] = M[k, i, j]): the threads of one warp
        # (consecutive i) then read consecutive memory.
        k, i = wp.tid()
        acc = wp.float64(0.0)
        for j in range(MT.shape[1]):
            acc = acc + wp.float64(MT[k, j, i]) * x[k, j]
        if mode == 0:
            out[k, i] = b[k, i] - acc
        else:
            out[k, i] = b[k, i] - (diag[k, i] * x[k, i] - acc)

    @wp.kernel(module=module)
    def contract2(VtT: wp.array3d(dtype=vec), g1: wp.array2d(dtype=wp.float64), g2: wp.array2d(dtype=wp.float64),
                  out1: wp.array2d(dtype=wp.vec3d), out2: wp.array2d(dtype=wp.vec3d)):
        # The contraction of contract for two vectors with one read of the tensor.
        k, i = wp.tid()
        a1 = wp.vec3d()
        a2 = wp.vec3d()
        for j in range(VtT.shape[1]):
            v = VtT[k, j, i]
            vd = wp.vec3d(wp.float64(v[0]), wp.float64(v[1]), wp.float64(v[2]))
            a1 = a1 + vd * g1[k, j]
            a2 = a2 + vd * g2[k, j]
        out1[k, i] = a1
        out2[k, i] = a2

    @wp.kernel(module=module)
    def contract(VtT: wp.array3d(dtype=vec), g: wp.array2d(dtype=wp.float64), out: wp.array2d(dtype=wp.vec3d)):
        # out[k, i] = sum_j Vt[k, i, j] g[k, j] with a float64 accumulator; VtT is
        # the transposed tensor (VtT[k, j, i] = Vt[k, i, j]).
        k, i = wp.tid()
        acc = wp.vec3d()
        for j in range(VtT.shape[1]):
            v = VtT[k, j, i]
            gj = g[k, j]
            acc = acc + wp.vec3d(wp.float64(v[0]) * gj, wp.float64(v[1]) * gj, wp.float64(v[2]) * gj)
        out[k, i] = acc

    @wp.kernel(module=module)
    def trefftz(Qh: wp.array2d(dtype=wp.vec3f), Ql: wp.array2d(dtype=wp.vec3f), Q: wp.array2d(dtype=wp.vec3d),
                QN: wp.array2d(dtype=vec),
                PVh: wp.array2d(dtype=wp.vec3f), PVl: wp.array2d(dtype=wp.vec3f), PV: wp.array2d(dtype=wp.vec3d),
                GV: wp.array2d(dtype=wp.float64), D: wp.array(dtype=vec),
                rc2: wp.array(dtype=T), sgroup: wp.array(dtype=wp.int32),
                tgroup: wp.array(dtype=wp.int32), trc2: wp.array(dtype=T), use_tg: int,
                out: wp.array2d(dtype=wp.float64)):
        # Normal wash at Trefftz point i of case k from the 2-D point vortices
        # PV[k, :] with the circulation GV[k, :] about the wake direction D[k].
        k, i = wp.tid()
        qn = QN[k, i]
        d = D[k]
        acc = wp.float64(0.0)
        for j in range(GV.shape[1]):
            r = vec()
            if wp.static(is64):
                r = Q[k, i] - PV[k, j]
            else:
                r = diff2(Qh[k, i], Ql[k, i], PVh[k, j], PVl[k, j])
            c2 = rc2[j]
            if use_tg != 0:
                if tgroup[i] != sgroup[j]:
                    c2 = c2 + trc2[i]
            w = T(_INV_2PI) * wp.dot(wp.cross(d, r), qn) / (wp.dot(r, r) + c2)
            acc = acc + wp.float64(w) * GV[k, j]
        out[k, i] = acc

    @wp.kernel(module=module)
    def case_te(D: wp.array(dtype=wp.vec3d), GK: wp.array(dtype=wp.vec3d), GO: wp.array(dtype=wp.float64),
                ATE: wp.array(dtype=wp.vec3d), BTE: wp.array(dtype=wp.vec3d), A: wp.array(dtype=wp.vec3d),
                n_img: int, dxTh: wp.array3d(dtype=wp.vec3f), dxTl: wp.array3d(dtype=wp.vec3f),
                dTh: wp.array3d(dtype=wp.float32), dTl: wp.array3d(dtype=wp.float32),
                dxTh_i: wp.array3d(dtype=wp.vec3f), dxTl_i: wp.array3d(dtype=wp.vec3f),
                dTh_i: wp.array3d(dtype=wp.float32), dTl_i: wp.array3d(dtype=wp.float32),
                Aih: wp.array2d(dtype=wp.vec3f), Ail: wp.array2d(dtype=wp.vec3f)):
        # The wake-leg data of panel j for case k (see Cases), as float32
        # pairs: d x TE and d . TE of the two trailing-edge points; with ground
        # images also those of the mirrored points with the mirrored
        # direction, and the mirrored bound-vortex point a.
        k, j = wp.tid()
        d = D[k]
        for e in range(2):
            te = ATE[j]
            if e == 1:
                te = BTE[j]
            h, lo = split3(wp.cross(d, te))
            dxTh[k, j, e] = h
            dxTl[k, j, e] = lo
            sh, sl = split1(wp.dot(te, d))
            dTh[k, j, e] = sh
            dTl[k, j, e] = sl
            if n_img != 0:
                gk = GK[k]
                ti = te - (wp.float64(2.0) * (wp.dot(te, gk) - GO[k])) * gk
                di = d - (wp.float64(2.0) * wp.dot(d, gk)) * gk
                h, lo = split3(wp.cross(di, ti))
                dxTh_i[k, j, e] = h
                dxTl_i[k, j, e] = lo
                sh, sl = split1(wp.dot(ti, di))
                dTh_i[k, j, e] = sh
                dTl_i[k, j, e] = sl
        if n_img != 0:
            gk = GK[k]
            a = A[j]
            h, lo = split3(a - (wp.float64(2.0) * (wp.dot(a, gk) - GO[k])) * gk)
            Aih[k, j] = h
            Ail[k, j] = lo

    @wp.kernel(module=module)
    def case_points(D: wp.array(dtype=wp.vec3d), GK: wp.array(dtype=wp.vec3d), P: wp.array(dtype=wp.vec3d),
                    n_img: int, dxPh: wp.array2d(dtype=wp.vec3f), dxPl: wp.array2d(dtype=wp.vec3f),
                    dPh: wp.array2d(dtype=wp.float32), dPl: wp.array2d(dtype=wp.float32),
                    dxPh_i: wp.array2d(dtype=wp.vec3f), dxPl_i: wp.array2d(dtype=wp.vec3f),
                    dPh_i: wp.array2d(dtype=wp.float32), dPl_i: wp.array2d(dtype=wp.float32)):
        # The wake-leg data of evaluation point i for case k (see Cases), as
        # float32 pairs: d x P and d . P, and with ground images the same with
        # the mirrored direction.
        k, i = wp.tid()
        d = D[k]
        p = P[i]
        h, lo = split3(wp.cross(d, p))
        dxPh[k, i] = h
        dxPl[k, i] = lo
        sh, sl = split1(wp.dot(p, d))
        dPh[k, i] = sh
        dPl[k, i] = sl
        if n_img != 0:
            gk = GK[k]
            di = d - (wp.float64(2.0) * wp.dot(d, gk)) * gk
            h, lo = split3(wp.cross(di, p))
            dxPh_i[k, i] = h
            dxPl_i[k, i] = lo
            sh, sl = split1(wp.dot(p, di))
            dPh_i[k, i] = sh
            dPl_i[k, i] = sl

    @wp.kernel(module=module)
    def trefftz_prep(WD: wp.array(dtype=wp.vec3d), te_l: wp.array(dtype=wp.vec3d), te_r: wp.array(dtype=wp.vec3d),
                     cp_frac: wp.array(dtype=wp.float64), GK: wp.array(dtype=wp.vec3d),
                     GO: wp.array(dtype=wp.float64), n_img: int, sg: wp.array2d(dtype=wp.float64),
                     eval_pos: wp.array(dtype=wp.int32), Q: wp.array2d(dtype=wp.vec3d),
                     Qh: wp.array2d(dtype=wp.vec3f), Ql: wp.array2d(dtype=wp.vec3f), QN: wp.array2d(dtype=vec),
                     seg_len: wp.array2d(dtype=wp.float64), PV: wp.array2d(dtype=wp.vec3d),
                     PVh: wp.array2d(dtype=wp.vec3f), PVl: wp.array2d(dtype=wp.vec3f),
                     GV: wp.array2d(dtype=wp.float64), Dn: wp.array(dtype=vec)):
        # The Trefftz-plane geometry of strip s for case k (the geometry of
        # trefftz_induced_drag_batch): the trailing-edge points projected on
        # the plane normal to the wake direction d, the evaluation point at
        # the span fraction of the control point and its normal (written at
        # eval_pos[s] when it is not negative), the projected strip length,
        # and the 2-D point vortices (right and left points, then their
        # ground images) with their circulation.
        k, s = wp.tid()
        n_s = te_l.shape[0]
        w = WD[k]
        d = w / wp.length(w)
        tl = te_l[s]
        tr = te_r[s]
        pl = tl - wp.dot(tl, d) * d
        pr = tr - wp.dot(tr, d) * d
        sv = pr - pl
        seg_len[k, s] = wp.length(sv)
        g = sg[k, s]
        PV[k, s] = pr
        PV[k, n_s + s] = pl
        GV[k, s] = g
        GV[k, n_s + s] = -g
        if wp.static(not is64):
            h, lo = split3(pr)
            PVh[k, s] = h
            PVl[k, s] = lo
            h, lo = split3(pl)
            PVh[k, n_s + s] = h
            PVl[k, n_s + s] = lo
        if n_img != 0:
            gk = GK[k]
            go = GO[k]
            for e in range(2):
                t = tr
                if e == 1:
                    t = tl
                refl = t - (wp.float64(2.0) * (wp.dot(t, gk) - go)) * gk
                pi = refl - wp.dot(refl, d) * d
                c = 2 * n_s + e * n_s + s
                PV[k, c] = pi
                if e == 0:
                    GV[k, c] = -g
                else:
                    GV[k, c] = g
                if wp.static(not is64):
                    h, lo = split3(pi)
                    PVh[k, c] = h
                    PVl[k, c] = lo
        m = eval_pos[s]
        if m >= 0:
            q = pl + cp_frac[s] * sv
            Q[k, m] = q
            if wp.static(not is64):
                h, lo = split3(q)
                Qh[k, m] = h
                Ql[k, m] = lo
            qn = wp.cross(d, sv)
            qn = qn / wp.max(wp.length(qn), wp.float64(1e-300))
            QN[k, m] = tvec(qn)
        if s == 0:
            Dn[k] = tvec(d)

    @wp.kernel(module=module)
    def llt_rhs(D: wp.array(dtype=wp.vec3d), V: wp.array(dtype=wp.float64), dl: wp.array(dtype=wp.vec3d),
                nrm: wp.array(dtype=wp.vec3d), area: wp.array(dtype=wp.float64), a0: wp.array(dtype=wp.float64),
                aL0: wp.array(dtype=wp.float64), diag: wp.array2d(dtype=wp.float64),
                rhs: wp.array2d(dtype=wp.float64), scale: wp.array2d(dtype=wp.float64),
                vinf: wp.array(dtype=wp.vec3d)):
        # The terms of the linear lifting line of strip i for case k
        # (solve_llt_linear): the diagonal 2 |u x dl| / (a0 dA), the
        # right-hand side V (u . n - alpha_L0), the scale V^2 dA of the
        # nonlinear residual, and the free-stream velocity V u.
        k, i = wp.tid()
        d = D[k]
        v = V[k]
        diag[k, i] = wp.float64(2.0) * wp.length(wp.cross(d, dl[i])) / (a0[i] * area[i])
        rhs[k, i] = v * (wp.dot(d, nrm[i]) - aL0[i])
        scale[k, i] = (v * v) * area[i]
        if i == 0:
            vinf[k] = v * d

    return SimpleNamespace(T=T, vec=vec, precision=precision, Sources=Sources, Points=Points, Cases=Cases,
                           tensors=tensors, normals=normals, induced=induced, contract=contract, residual=residual,
                           fixed_points=fixed_points, induced_wakes=induced_wakes, induced_img_fixed=induced_img_fixed,
                           Polars=Polars, nl_state=nl_state, nl_trials=nl_trials, nl_jacobian=nl_jacobian,
                           nl_reduce=nl_reduce, nl_update=nl_update, contract2=contract2, nl_merit=nl_merit,
                           wake_table=wake_table, normal_from_table=normal_from_table,
                           loads_strip=loads_strip, loads_totals=loads_totals,
                           trefftz=trefftz, case_te=case_te, case_points=case_points, trefftz_prep=trefftz_prep,
                           llt_rhs=llt_rhs)
