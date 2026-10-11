# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Vector-Jacobian products of the bent-horseshoe kernel on the GPU (NVIDIA Warp).

The kernels compute the same quantities as the numpy reference
:mod:`ventorum.aero.vortex_vjp` (same kernel, same stable forms, same
topology masks), for the adjoint gradients of :mod:`ventorum.sensitivity`.
They take PyTorch CUDA tensors and run on the current PyTorch stream.

Layout
------
* The VJP runs in two launches. The row launch has one thread per target
  point and block of sources; it adds the target derivatives of its block.
  The column launch has one thread per source and block of targets; it
  adds the source derivatives of its block. Each launch writes one partial
  sum per block into a buffer, and PyTorch adds the buffers in float64. So
  no thread adds into a shared value (no atomic operations), and the result
  does not depend on the thread timing. The two launches compute each pair
  twice; that costs less than the atomic additions of one launch.
* The system rows (forward) have one thread per matrix entry; it adds the
  sources of its unknown in source order (the order of the reference copy).
* Each kernel is made for float32 or float64 (the type ``T``). In float32
  each pair is computed in float32 (the speed of consumer GPUs); the block
  sums have at most ``BLOCK`` terms and the sums of the blocks are made in
  float64.

The derivation of the derivatives is in :mod:`ventorum.aero.vortex_vjp` and
in the theory manual (``docs/theory/gradients.md``). Sources: J. Katz and
A. Plotkin, "Low-Speed Aerodynamics", 2nd ed., Cambridge University Press,
2001 (the kernel); A. Griewank and A. Walther, "Evaluating Derivatives",
2nd ed., SIAM, 2008 (the reverse mode).

The module needs the ``warp-lang`` package (optional dependency ``gpu``).
"""

# No "from __future__ import annotations": Warp reads the type annotations of the
# kernels at definition time, with the types of the closure (T, vec).
import math
import threading
from types import SimpleNamespace

import torch
import warp as wp

_INV_4PI = 1.0 / (4.0 * math.pi)
_INV_2PI = 1.0 / (2.0 * math.pi)
_FLOORS = {"float64": 1e-300, "float32": 1e-30}

# Sources (or targets) per thread of the row (column) launch.
BLOCK = 32

_lock = threading.Lock()
_built: dict = {}


def kernels(precision: str) -> SimpleNamespace:
    """Return the VJP kernels for *precision* (``"float32"`` or ``"float64"``); build them on the first call."""
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
    module = f"ventorum_vjp_{precision}"
    wp.set_module_options({"enable_backward": False}, wp.get_module(module))
    BJ = BLOCK

    @wp.func(module=module)
    def seg_v(p: vec, A: vec, B: vec, rc2: T):
        # Velocity of a unit finite segment A -> B at p (stable projection).
        r1 = p - A
        r2 = p - B
        r0 = B - A
        c = wp.cross(r1, r2)
        cr2 = wp.dot(c, c)
        n1 = wp.sqrt(wp.dot(r1, r1) + T(floor))
        n2 = wp.sqrt(wp.dot(r2, r2) + T(floor))
        dot12 = wp.dot(r1, r2)
        proj = T(0.0)
        if dot12 > T(0.0):
            n12 = n1 * n2
            proj = cr2 * (n1 + n2) / (n12 * (n12 + dot12))
        else:
            proj = wp.dot(r0, r1 / n1 - r2 / n2)
        den = cr2 + rc2 * wp.dot(r0, r0)
        k = T(0.0)
        if den > T(floor):
            k = T(_INV_4PI) * proj / den
        return c * k

    @wp.func(module=module)
    def ray_v(p: vec, A: vec, d: vec, rc2: T):
        # Velocity of a unit semi-infinite line from A along d at p (stable 1 + cos).
        r = p - A
        c = wp.cross(d, r)
        cr2 = wp.dot(c, c)
        rn = wp.sqrt(wp.dot(r, r) + T(floor))
        dr = wp.dot(d, r)
        oc = T(1.0) + dr / rn
        if dr < T(0.0):
            oc = cr2 / (rn * (rn - dr))
        den = cr2 + rc2
        k = T(0.0)
        if den > T(floor):
            k = T(_INV_4PI) * oc / den
        return c * k

    @wp.func(module=module)
    def seg_fb(p: vec, A: vec, B: vec, rc2: T, u: vec):
        # Velocity and VJP of a unit finite segment A -> B at p for the adjoint u.
        # Returns v, gP, gA, gB, grc2. A pair with a zero denominator gives zeros.
        r1 = p - A
        r2 = p - B
        r0 = B - A
        c = wp.cross(r1, r2)
        cr2 = wp.dot(c, c)
        r0sq = wp.dot(r0, r0)
        n1 = wp.sqrt(wp.dot(r1, r1) + T(floor))
        n2 = wp.sqrt(wp.dot(r2, r2) + T(floor))
        dot12 = wp.dot(r1, r2)
        n12 = n1 * n2
        t1 = r1 / n1
        t2 = r2 / n2
        out = dot12 > T(0.0)
        den_out = T(1.0)
        proj = T(0.0)
        if out:
            den_out = n12 * (n12 + dot12)
            proj = cr2 * (n1 + n2) / den_out
        else:
            proj = wp.dot(r0, t1 - t2)
        den = cr2 + rc2 * r0sq
        ok = T(0.0)
        safe = T(1.0)
        if den > T(floor):
            ok = T(1.0)
            safe = den
        k = ok * T(_INV_4PI) * proj / safe
        v = c * k
        gc = u * k
        gk = wp.dot(u, c)
        gproj = ok * gk * T(_INV_4PI) / safe
        gden = -ok * gk * k / safe
        gcr2 = gden
        grc2 = gden * r0sq
        gr0sq = gden * rc2
        gr1 = vec()
        gr2 = vec()
        gr0 = vec()
        gn1 = T(0.0)
        gn2 = T(0.0)
        if out:
            gcr2 = gcr2 + gproj * (n1 + n2) / den_out
            gsum = gproj * cr2 / den_out
            gD = -gproj * proj / den_out
            gn12 = gD * (T(2.0) * n12 + dot12)
            gdot12 = gD * n12
            gn1 = gsum + gn12 * n2
            gn2 = gsum + gn12 * n1
            gr1 = r2 * gdot12
            gr2 = r1 * gdot12
        else:
            gr0 = (t1 - t2) * gproj
            gt1 = r0 * gproj
            gr1 = gt1 / n1
            gr2 = -gt1 / n2
            gn1 = -wp.dot(gt1, t1) / n1
            gn2 = wp.dot(gt1, t2) / n2
        gr1 = gr1 + r1 * (gn1 / n1)
        gr2 = gr2 + r2 * (gn2 / n2)
        gc = gc + c * (T(2.0) * gcr2)
        gr1 = gr1 + wp.cross(r2, gc)
        gr2 = gr2 + wp.cross(gc, r1)
        gr0 = gr0 + r0 * (T(2.0) * gr0sq)
        return v, gr1 + gr2, -gr1 - gr0, gr0 - gr2, grc2

    @wp.func(module=module)
    def ray_fb(p: vec, A: vec, d: vec, rc2: T, u: vec):
        # Velocity and VJP of a unit semi-infinite line from A along d at p.
        # Returns v, gP, gd, grc2; gA is -gP.
        r = p - A
        c = wp.cross(d, r)
        cr2 = wp.dot(c, c)
        rn = wp.sqrt(wp.dot(r, r) + T(floor))
        dr = wp.dot(d, r)
        behind = dr < T(0.0)
        den_b = T(1.0)
        oc = T(1.0) + dr / rn
        if behind:
            den_b = rn * (rn - dr)
            oc = cr2 / den_b
        den = cr2 + rc2
        ok = T(0.0)
        safe = T(1.0)
        if den > T(floor):
            ok = T(1.0)
            safe = den
        k = ok * T(_INV_4PI) * oc / safe
        v = c * k
        gc = u * k
        gk = wp.dot(u, c)
        goc = ok * gk * T(_INV_4PI) / safe
        gden = -ok * gk * k / safe
        gcr2 = gden
        grc2 = gden
        grn = T(0.0)
        gdr = T(0.0)
        if behind:
            gcr2 = gcr2 + goc / den_b
            gE = -goc * oc / den_b
            grn = gE * (T(2.0) * rn - dr)
            gdr = -gE * rn
        else:
            gdr = goc / rn
            grn = -goc * dr / (rn * rn)
        gr = r * (grn / rn) + d * gdr
        gd = r * gdr
        gc = gc + c * (T(2.0) * gcr2)
        gd = gd + wp.cross(r, gc)
        gr = gr + wp.cross(gc, d)
        return v, gr, gd, grc2

    @wp.func(module=module)
    def pair_rc2(i: int, j: int, rc2s: wp.array(dtype=T), tgrc2: wp.array(dtype=T),
                 sgrp: wp.array(dtype=wp.int64), tgrp: wp.array(dtype=wp.int64), use_tg: int):
        r = rc2s[j]
        if use_tg != 0:
            if tgrp[i] != sgrp[j]:
                r = rc2s[j] + tgrc2[i]
        return r

    @wp.func(module=module)
    def is_other(i: int, j: int, sgrp: wp.array(dtype=wp.int64), tgrp: wp.array(dtype=wp.int64), use_tg: int):
        r = T(0.0)
        if use_tg != 0:
            if tgrp[i] != sgrp[j]:
                r = T(1.0)
        return r

    @wp.func(module=module)
    def hs_v(i: int, j: int, p: vec, a: wp.array(dtype=vec), b: wp.array(dtype=vec),
             ate: wp.array(dtype=vec), bte: wp.array(dtype=vec), d: wp.array(dtype=vec), rc2: T,
             own: wp.array(dtype=wp.int64), spanel: wp.array(dtype=wp.int64), edge: wp.array(dtype=wp.int64),
             sea: wp.array(dtype=wp.int64), seb: wp.array(dtype=wp.int64)):
        v = vec()
        if own[i] != spanel[j]:
            v = v + seg_v(p, a[j], b[j], rc2)
        if edge[i] != sea[j]:
            v = v + seg_v(p, ate[j], a[j], rc2)
        if edge[i] != seb[j]:
            v = v + seg_v(p, b[j], bte[j], rc2)
        return v + ray_v(p, bte[j], d[j], rc2) - ray_v(p, ate[j], d[j], rc2)

    @wp.kernel(module=module)
    def fwd_rows(P: wp.array(dtype=vec), X: wp.array(dtype=vec), a: wp.array(dtype=vec),
                 b: wp.array(dtype=vec), ate: wp.array(dtype=vec), bte: wp.array(dtype=vec),
                 d: wp.array(dtype=vec), rc2s: wp.array(dtype=T), tgrc2: wp.array(dtype=T),
                 sgrp: wp.array(dtype=wp.int64), tgrp: wp.array(dtype=wp.int64), own: wp.array(dtype=wp.int64),
                 spanel: wp.array(dtype=wp.int64), edge: wp.array(dtype=wp.int64),
                 sea: wp.array(dtype=wp.int64), seb: wp.array(dtype=wp.int64), s: wp.array(dtype=T),
                 off: wp.array(dtype=wp.int64), order: wp.array(dtype=wp.int64), use_tg: int,
                 out: wp.array2d(dtype=T)):
        # System rows: one thread per (target, unknown); the sources of the unknown in source order.
        i, u = wp.tid()
        p = P[i]
        x = X[i]
        acc = T(0.0)
        for k in range(int(off[u]), int(off[u + 1])):
            j = int(order[k])
            rc2 = pair_rc2(i, j, rc2s, tgrc2, sgrp, tgrp, use_tg)
            v = hs_v(i, j, p, a, b, ate, bte, d, rc2, own, spanel, edge, sea, seb)
            acc = acc + s[j] * wp.dot(v, x)
        out[i, u] = acc

    @wp.kernel(module=module)
    def fwd_induced(P: wp.array(dtype=vec), a: wp.array(dtype=vec), b: wp.array(dtype=vec),
                    ate: wp.array(dtype=vec), bte: wp.array(dtype=vec), d: wp.array(dtype=vec),
                    rc2s: wp.array(dtype=T), tgrc2: wp.array(dtype=T), sgrp: wp.array(dtype=wp.int64),
                    tgrp: wp.array(dtype=wp.int64), own: wp.array(dtype=wp.int64),
                    spanel: wp.array(dtype=wp.int64), edge: wp.array(dtype=wp.int64),
                    sea: wp.array(dtype=wp.int64), seb: wp.array(dtype=wp.int64), s: wp.array(dtype=T),
                    use_tg: int, buf: wp.array3d(dtype=T)):
        # Induced velocity: one thread per (target, source block); one partial sum per block.
        i, jb = wp.tid()
        n = a.shape[0]
        p = P[i]
        acc = vec()
        for j in range(jb * BJ, wp.min((jb + 1) * BJ, n)):
            rc2 = pair_rc2(i, j, rc2s, tgrc2, sgrp, tgrp, use_tg)
            acc = acc + hs_v(i, j, p, a, b, ate, bte, d, rc2, own, spanel, edge, sea, seb) * s[j]
        buf[jb, i, 0] = acc[0]
        buf[jb, i, 1] = acc[1]
        buf[jb, i, 2] = acc[2]

    @wp.func(module=module)
    def hs_fb(i: int, j: int, p: vec, u: vec, a: wp.array(dtype=vec), b: wp.array(dtype=vec),
              ate: wp.array(dtype=vec), bte: wp.array(dtype=vec), d: wp.array(dtype=vec), rc2: T,
              own: wp.array(dtype=wp.int64), spanel: wp.array(dtype=wp.int64), edge: wp.array(dtype=wp.int64),
              sea: wp.array(dtype=wp.int64), seb: wp.array(dtype=wp.int64)):
        # Velocity and VJP of horseshoe j at point i. Returns v, gP, ga, gb, gat, gbt, gd, grc2.
        aj = a[j]
        bj = b[j]
        tj = ate[j]
        uj = bte[j]
        dj = d[j]
        v = vec()
        gP = vec()
        ga = vec()
        gb = vec()
        gat = vec()
        gbt = vec()
        grc = T(0.0)
        if own[i] != spanel[j]:
            v1, p1, A1, B1, r1 = seg_fb(p, aj, bj, rc2, u)
            v = v + v1
            gP = gP + p1
            ga = ga + A1
            gb = gb + B1
            grc = grc + r1
        if edge[i] != sea[j]:
            v2, p2, A2, B2, r2 = seg_fb(p, tj, aj, rc2, u)
            v = v + v2
            gP = gP + p2
            gat = gat + A2
            ga = ga + B2
            grc = grc + r2
        if edge[i] != seb[j]:
            v3, p3, A3, B3, r3 = seg_fb(p, bj, uj, rc2, u)
            v = v + v3
            gP = gP + p3
            gb = gb + A3
            gbt = gbt + B3
            grc = grc + r3
        v4, p4, d4, r4 = ray_fb(p, uj, dj, rc2, u)
        v5, p5, d5, r5 = ray_fb(p, tj, dj, rc2, -u)
        v = v + v4 - v5
        gP = gP + p4 + p5
        gbt = gbt - p4
        gat = gat - p5
        grc = grc + r4 + r5
        return v, gP, ga, gb, gat, gbt, d4 + d5, grc

    @wp.kernel(module=module)
    def vjp_rows(P: wp.array(dtype=vec), X: wp.array(dtype=vec), a: wp.array(dtype=vec),
                 b: wp.array(dtype=vec), ate: wp.array(dtype=vec), bte: wp.array(dtype=vec),
                 d: wp.array(dtype=vec), rc2s: wp.array(dtype=T), tgrc2: wp.array(dtype=T),
                 sgrp: wp.array(dtype=wp.int64), tgrp: wp.array(dtype=wp.int64), own: wp.array(dtype=wp.int64),
                 spanel: wp.array(dtype=wp.int64), edge: wp.array(dtype=wp.int64),
                 sea: wp.array(dtype=wp.int64), seb: wp.array(dtype=wp.int64), s: wp.array(dtype=T),
                 col: wp.array(dtype=wp.int64), W: wp.array2d(dtype=T), use_W: int, use_tg: int,
                 buf: wp.array3d(dtype=T)):
        # Target derivatives: one thread per (target, source block). buf[jb, i] = gP, gX, g_tg_rc2.
        i, jb = wp.tid()
        n = a.shape[0]
        p = P[i]
        x = X[i]
        aP = vec()
        aX = vec()
        atg = T(0.0)
        for j in range(jb * BJ, wp.min((jb + 1) * BJ, n)):
            wf = T(1.0)
            if use_W != 0:
                wf = W[i, int(col[j])]
            cij = wf * s[j]
            rc2 = pair_rc2(i, j, rc2s, tgrc2, sgrp, tgrp, use_tg)
            v, gP, ga, gb, gat, gbt, gd, grc = hs_fb(i, j, p, x * cij, a, b, ate, bte, d, rc2,
                                                    own, spanel, edge, sea, seb)
            aP = aP + gP
            aX = aX + v * cij
            atg = atg + grc * is_other(i, j, sgrp, tgrp, use_tg)
        buf[jb, i, 0] = aP[0]
        buf[jb, i, 1] = aP[1]
        buf[jb, i, 2] = aP[2]
        buf[jb, i, 3] = aX[0]
        buf[jb, i, 4] = aX[1]
        buf[jb, i, 5] = aX[2]
        buf[jb, i, 6] = atg

    @wp.kernel(module=module)
    def vjp_cols(P: wp.array(dtype=vec), X: wp.array(dtype=vec), a: wp.array(dtype=vec),
                 b: wp.array(dtype=vec), ate: wp.array(dtype=vec), bte: wp.array(dtype=vec),
                 d: wp.array(dtype=vec), rc2s: wp.array(dtype=T), tgrc2: wp.array(dtype=T),
                 sgrp: wp.array(dtype=wp.int64), tgrp: wp.array(dtype=wp.int64), own: wp.array(dtype=wp.int64),
                 spanel: wp.array(dtype=wp.int64), edge: wp.array(dtype=wp.int64),
                 sea: wp.array(dtype=wp.int64), seb: wp.array(dtype=wp.int64), s: wp.array(dtype=T),
                 col: wp.array(dtype=wp.int64), W: wp.array2d(dtype=T), use_W: int, use_tg: int,
                 buf: wp.array3d(dtype=T)):
        # Source derivatives: one thread per (source, target block).
        # buf[ib, j] = ga, gb, ga_te, gb_te, gd, g_rc2_src, gs.
        j, ib = wp.tid()
        m = P.shape[0]
        sj = s[j]
        cj = int(col[j])
        aa = vec()
        ab = vec()
        aat = vec()
        abt = vec()
        ad = vec()
        arc = T(0.0)
        asg = T(0.0)
        for i in range(ib * BJ, wp.min((ib + 1) * BJ, m)):
            wf = T(1.0)
            if use_W != 0:
                wf = W[i, cj]
            cij = wf * sj
            x = X[i]
            rc2 = pair_rc2(i, j, rc2s, tgrc2, sgrp, tgrp, use_tg)
            v, gP, ga, gb, gat, gbt, gd, grc = hs_fb(i, j, P[i], x * cij, a, b, ate, bte, d, rc2,
                                                    own, spanel, edge, sea, seb)
            aa = aa + ga
            ab = ab + gb
            aat = aat + gat
            abt = abt + gbt
            ad = ad + gd
            arc = arc + grc
            asg = asg + wf * wp.dot(x, v)
        for c in range(3):
            buf[ib, j, c] = aa[c]
            buf[ib, j, 3 + c] = ab[c]
            buf[ib, j, 6 + c] = aat[c]
            buf[ib, j, 9 + c] = abt[c]
            buf[ib, j, 12 + c] = ad[c]
        buf[ib, j, 15] = arc
        buf[ib, j, 16] = asg

    @wp.kernel(module=module)
    def tz_rows(q: wp.array(dtype=vec), qn: wp.array(dtype=vec), p: wp.array(dtype=vec),
                gamma: wp.array(dtype=T), dd: wp.array(dtype=vec), rc2: wp.array(dtype=T), tgrc2: wp.array(dtype=T),
                sgrp: wp.array(dtype=wp.int64), tgrp: wp.array(dtype=wp.int64), use_tg: int,
                gw: wp.array(dtype=T), buf: wp.array3d(dtype=T)):
        # Trefftz wash and target derivatives: buf[jb, i] = w, gq, gqn, g_tg_rc2.
        i, jb = wp.tid()
        n = p.shape[0]
        d = dd[0]
        qi = q[i]
        qq = qn[i]
        g = gw[i]
        aw = T(0.0)
        aq = vec()
        an = vec()
        atg = T(0.0)
        for j in range(jb * BJ, wp.min((jb + 1) * BJ, n)):
            c2 = pair_rc2(i, j, rc2, tgrc2, sgrp, tgrp, use_tg)
            r = qi - p[j]
            v = wp.cross(d, r)
            D = wp.dot(r, r) + c2
            k = T(_INV_2PI) * gamma[j] / D
            e = wp.dot(v, qq)
            aw = aw + e * k
            ge = g * k
            gD = -(g * e) * k / D
            gv = qq * ge
            aq = aq + r * (T(2.0) * gD) + wp.cross(gv, d)
            an = an + v * ge
            atg = atg + gD * is_other(i, j, sgrp, tgrp, use_tg)
        buf[jb, i, 0] = aw
        for c in range(3):
            buf[jb, i, 1 + c] = aq[c]
            buf[jb, i, 4 + c] = an[c]
        buf[jb, i, 7] = atg

    @wp.kernel(module=module)
    def tz_cols(q: wp.array(dtype=vec), qn: wp.array(dtype=vec), p: wp.array(dtype=vec),
                gamma: wp.array(dtype=T), dd: wp.array(dtype=vec), rc2: wp.array(dtype=T), tgrc2: wp.array(dtype=T),
                sgrp: wp.array(dtype=wp.int64), tgrp: wp.array(dtype=wp.int64), use_tg: int,
                gw: wp.array(dtype=T), buf: wp.array3d(dtype=T)):
        # Trefftz vortex derivatives: buf[ib, j] = gp, ggamma, g_rc2, gd.
        j, ib = wp.tid()
        m = q.shape[0]
        d = dd[0]
        pj = p[j]
        gj = gamma[j]
        ap = vec()
        ag = T(0.0)
        arc = T(0.0)
        ad = vec()
        for i in range(ib * BJ, wp.min((ib + 1) * BJ, m)):
            c2 = pair_rc2(i, j, rc2, tgrc2, sgrp, tgrp, use_tg)
            r = q[i] - pj
            v = wp.cross(d, r)
            D = wp.dot(r, r) + c2
            k = T(_INV_2PI) * gj / D
            qq = qn[i]
            g = gw[i]
            e = wp.dot(v, qq)
            ge = g * k
            gk = g * e
            gD = -gk * k / D
            gv = qq * ge
            ap = ap - (r * (T(2.0) * gD) + wp.cross(gv, d))
            ag = ag + gk * T(_INV_2PI) / D
            arc = arc + gD
            ad = ad + wp.cross(r, gv)
        for c in range(3):
            buf[ib, j, c] = ap[c]
            buf[ib, j, 5 + c] = ad[c]
        buf[ib, j, 3] = ag
        buf[ib, j, 4] = arc

    return SimpleNamespace(T=T, vec=vec, fwd_rows=fwd_rows, fwd_induced=fwd_induced, vjp_rows=vjp_rows,
                           vjp_cols=vjp_cols, tz_rows=tz_rows, tz_cols=tz_cols)


# --------------------------------------------------------------------------
# Host functions: torch CUDA tensors in, torch CUDA tensors out.
# --------------------------------------------------------------------------

def _prec(t: torch.Tensor) -> str:
    return "float64" if t.dtype == torch.float64 else "float32"


def _launch(kernel, dim, inputs) -> None:  # type: ignore[no-untyped-def]
    from ventorum.gpu.engine import launch

    launch(kernel, dim, inputs)


def _v(t: torch.Tensor, ns):  # type: ignore[no-untyped-def]
    return wp.from_torch(t.detach().contiguous(), dtype=ns.vec)


def _s(t: torch.Tensor):  # type: ignore[no-untyped-def]
    return wp.from_torch(t.detach().contiguous())


def _i(t: torch.Tensor):  # type: ignore[no-untyped-def]
    return wp.from_torch(t.to(torch.int64).contiguous())


def _blocks(n: int) -> int:
    return max(1, (int(n) + BLOCK - 1) // BLOCK)


def _sum_blocks(buf: torch.Tensor, dtype) -> torch.Tensor:  # type: ignore[no-untyped-def]
    """Add the block partial sums (axis 0) in float64; return them in *dtype*."""
    return buf.to(torch.float64).sum(dim=0).to(dtype)


def _topo_arrays(topo, m: int, dev) -> tuple:  # type: ignore[no-untyped-def]
    """Return the Warp int64 arrays (sgrp, tgrp, own, spanel, edge, sea, seb, col) of *topo*."""
    cache = getattr(topo, "_wp", None)
    if cache is None:
        cache = tuple(_i(x.to(dev)) for x in (topo.src_group, topo.tg_group, topo.own, topo.src_panel,
                                              topo.edge, topo.src_edge_a, topo.src_edge_b, topo.col))
        topo._wp = cache
    return cache


def _csr(topo, dev) -> tuple:  # type: ignore[no-untyped-def]
    """Return the (offsets, order) of the sources per unknown, sources in source order."""
    cache = getattr(topo, "_csr", None)
    if cache is None:
        col = topo.col.to(dev)
        order = torch.argsort(col, stable=True)
        counts = torch.bincount(col, minlength=topo.n_u)
        off = torch.zeros(topo.n_u + 1, dtype=torch.int64, device=dev)
        off[1:] = torch.cumsum(counts, dim=0)
        cache = (_i(off), _i(order))
        topo._csr = cache
    return cache


def horseshoe_forward(mode, P, X, a, b, a_te, b_te, d, rc2_src, tg_rc2, s, topo):  # type: ignore[no-untyped-def]
    """Return the system rows (mode 0, (m, n_u)) or the induced velocity (mode 1, (m, 3)) on the GPU."""
    ns = kernels(_prec(P))
    m, n = int(P.shape[0]), int(a.shape[0])
    dev = P.device
    use_tg = 1 if (topo.use_tg and tg_rc2 is not None) else 0
    tg2 = tg_rc2 if tg_rc2 is not None else torch.zeros(m, dtype=P.dtype, device=dev)
    sg, tg, own, sp, edge, sea, seb, _col = _topo_arrays(topo, m, dev)
    geo = (_v(a, ns), _v(b, ns), _v(a_te, ns), _v(b_te, ns), _v(d, ns), _s(rc2_src), _s(tg2))
    if mode == 0:
        out = torch.zeros((m, topo.n_u), dtype=P.dtype, device=dev)
        off, order = _csr(topo, dev)
        _launch(ns.fwd_rows, (m, topo.n_u),
                [_v(P, ns), _v(X, ns), *geo, sg, tg, own, sp, edge, sea, seb, _s(s), off, order, use_tg,
                 wp.from_torch(out)])
        return out
    nb = _blocks(n)
    buf = torch.empty((nb, m, 3), dtype=P.dtype, device=dev)
    _launch(ns.fwd_induced, (m, nb),
            [_v(P, ns), *geo, sg, tg, own, sp, edge, sea, seb, _s(s), use_tg, wp.from_torch(buf)])
    return _sum_blocks(buf, P.dtype)


def horseshoe_vjp(P, X, a, b, a_te, b_te, d, rc2_src, tg_rc2, s, W, topo):  # type: ignore[no-untyped-def]
    """Return the VJP (gP, gX, g_tg_rc2, ga, gb, ga_te, gb_te, gd, g_rc2_src, gs) on the GPU."""
    ns = kernels(_prec(P))
    m, n = int(P.shape[0]), int(a.shape[0])
    dev = P.device
    use_tg = 1 if (topo.use_tg and tg_rc2 is not None) else 0
    tg2 = tg_rc2 if tg_rc2 is not None else torch.zeros(m, dtype=P.dtype, device=dev)
    sg, tg, own, sp, edge, sea, seb, col = _topo_arrays(topo, m, dev)
    use_w = 1 if W is not None else 0
    Wt = W.contiguous() if W is not None else torch.zeros((1, 1), dtype=P.dtype, device=dev)
    args = [_v(P, ns), _v(X, ns), _v(a, ns), _v(b, ns), _v(a_te, ns), _v(b_te, ns), _v(d, ns), _s(rc2_src),
            _s(tg2), sg, tg, own, sp, edge, sea, seb, _s(s), col, wp.from_torch(Wt), use_w, use_tg]
    nbj = _blocks(n)
    rbuf = torch.empty((nbj, m, 7), dtype=P.dtype, device=dev)
    _launch(ns.vjp_rows, (m, nbj), args + [wp.from_torch(rbuf)])
    nbi = _blocks(m)
    cbuf = torch.empty((nbi, n, 17), dtype=P.dtype, device=dev)
    _launch(ns.vjp_cols, (n, nbi), args + [wp.from_torch(cbuf)])
    r = _sum_blocks(rbuf, P.dtype)
    c = _sum_blocks(cbuf, P.dtype)
    return (r[:, 0:3], r[:, 3:6], r[:, 6], c[:, 0:3], c[:, 3:6], c[:, 6:9], c[:, 9:12], c[:, 12:15],
            c[:, 15], c[:, 16])


def _tz_args(q, qn, p, gamma, d, rc2, tg_rc2, groups, ns):  # type: ignore[no-untyped-def]
    sg, tg, use_tg = groups
    return [_v(q, ns), _v(qn, ns), _v(p, ns), _s(gamma), _v(d.reshape(1, 3), ns), _s(rc2), _s(tg_rc2),
            _i(sg.to(q.device)), _i(tg.to(q.device)), 1 if use_tg else 0]


def trefftz_forward(q, qn, p, gamma, d, rc2, tg_rc2, groups):  # type: ignore[no-untyped-def]
    """Return the Trefftz normal wash (m,) on the GPU."""
    ns = kernels(_prec(q))
    m, n = int(q.shape[0]), int(p.shape[0])
    nb = _blocks(n)
    buf = torch.empty((nb, m, 8), dtype=q.dtype, device=q.device)
    gw = torch.zeros(m, dtype=q.dtype, device=q.device)
    _launch(ns.tz_rows, (m, nb), _tz_args(q, qn, p, gamma, d, rc2, tg_rc2, groups, ns) + [_s(gw),
                                                                                        wp.from_torch(buf)])
    return _sum_blocks(buf[:, :, 0], q.dtype)


def trefftz_vjp(q, qn, p, gamma, d, rc2, tg_rc2, groups, gw):  # type: ignore[no-untyped-def]
    """Return the VJP (gq, gqn, g_tg_rc2, gp, ggamma, g_rc2, gd) of the Trefftz normal wash on the GPU."""
    ns = kernels(_prec(q))
    m, n = int(q.shape[0]), int(p.shape[0])
    args = _tz_args(q, qn, p, gamma, d, rc2, tg_rc2, groups, ns) + [_s(gw)]
    nbj = _blocks(n)
    rbuf = torch.empty((nbj, m, 8), dtype=q.dtype, device=q.device)
    _launch(ns.tz_rows, (m, nbj), args + [wp.from_torch(rbuf)])
    nbi = _blocks(m)
    cbuf = torch.empty((nbi, n, 8), dtype=q.dtype, device=q.device)
    _launch(ns.tz_cols, (n, nbi), args + [wp.from_torch(cbuf)])
    r = _sum_blocks(rbuf, q.dtype)
    c = _sum_blocks(cbuf, q.dtype)
    gd = cbuf[:, :, 5:8].to(torch.float64).sum(dim=(0, 1)).to(q.dtype)
    return r[:, 1:4], r[:, 4:7], r[:, 7], c[:, 0:3], c[:, 3], c[:, 4], gd
