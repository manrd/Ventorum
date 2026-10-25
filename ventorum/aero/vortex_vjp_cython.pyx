# Author: Manuel Alejandro Rodriguez Diaz, PhD
#cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""Compiled (Cython) vector-Jacobian products of the bent-horseshoe kernel.

The functions here compute the same quantities as the numpy reference
:mod:`ventorum.aero.vortex_vjp` and the Numba kernels
:mod:`ventorum.aero.vortex_vjp_numba` (same kernel, same stable forms, same
topology masks, same floors), in one pass over each (point, source) pair.
The points run in parallel (OpenMP ``prange``) in blocks of rows; each
block adds its source derivatives into its own buffer, and the buffers are
added in block order at the end, so the result does not depend on the
thread timing. Without OpenMP (the default macOS build, see setup.py), the
caller splits the rows into blocks on Python threads; the functions release
the GIL.

Each pair is computed in float64. Each function takes the arguments of its
Numba counterpart, in the same order, plus one last argument
``num_threads``.

The derivation of the derivatives is in :mod:`ventorum.aero.vortex_vjp` and
in the theory manual (``docs/theory/gradients.md``). Sources: J. Katz and
A. Plotkin, "Low-Speed Aerodynamics", 2nd ed., Cambridge University Press,
2001 (the kernel); A. Griewank and A. Walther, "Evaluating Derivatives",
2nd ed., SIAM, 2008 (the reverse mode).
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt, pi
from cython.parallel cimport prange

cnp.import_array()

cdef extern from *:
    """
    #ifdef _OPENMP
    #define VENTORUM_VJP_HAVE_OPENMP 1
    #else
    #define VENTORUM_VJP_HAVE_OPENMP 0
    #endif
    """
    int VENTORUM_VJP_HAVE_OPENMP


def openmp_enabled():
    """Return True if this extension was compiled with OpenMP.

    Returns
    -------
    bool
        True when the C compiler defined ``_OPENMP`` for this build.
    """
    return bool(VENTORUM_VJP_HAVE_OPENMP)


cdef double _INV_4PI = 1.0 / (4.0 * pi)
cdef double _INV_2PI = 1.0 / (2.0 * pi)
cdef double _FLOOR = 1e-300
# Values per source in the block buffers: ga, gb, ga_te, gb_te, gd (3 each), g_rc2_src, gs,
# and the leg nodes ga_k, gb_k (3 each).
# Values per vortex in the Trefftz block buffers: gp (3), ggamma, g_rc2.
cdef enum:
    NSRC = 23
    NVTX = 5


cdef inline void _seg_v(double px, double py, double pz, double ax, double ay, double az,
                        double bx, double by, double bz, double rc2, double* v) noexcept nogil:
    # Velocity of a unit finite segment a -> b at p (stable projection).
    # Outside the segment (before a or after b) the exact law is multiplied by
    # D^2 / (D^2 + rc2), D the distance to the nearest end point (see ventorum.aero.vortex).
    cdef double r1x = px - ax, r1y = py - ay, r1z = pz - az
    cdef double r2x = px - bx, r2y = py - by, r2z = pz - bz
    cdef double r0x = bx - ax, r0y = by - ay, r0z = bz - az
    cdef double cx = r1y * r2z - r1z * r2y
    cdef double cy = r1z * r2x - r1x * r2z
    cdef double cz = r1x * r2y - r1y * r2x
    cdef double cr2 = cx * cx + cy * cy + cz * cz
    cdef double r0sq = r0x * r0x + r0y * r0y + r0z * r0z
    cdef double d1 = r1x * r1x + r1y * r1y + r1z * r1z + _FLOOR
    cdef double d2 = r2x * r2x + r2y * r2y + r2z * r2z + _FLOOR
    cdef double n1 = sqrt(d1)
    cdef double n2 = sqrt(d2)
    cdef double dot12 = r1x * r2x + r1y * r2y + r1z * r2z
    cdef bint before = r0x * r1x + r0y * r1y + r0z * r1z < 0.0
    cdef bint after = r0x * r2x + r0y * r2y + r0z * r2z > 0.0
    cdef double n12, proj, den, k, e, dd
    k = 0.0
    if before or after:
        dd = d1 if before else d2
        n12 = n1 * n2
        e = n12 + dot12
        if e > _FLOOR:
            k = _INV_4PI * (n1 + n2) / (n12 * e) * (dd / (dd + rc2))
    else:
        if dot12 > 0.0:
            n12 = n1 * n2
            proj = cr2 * (n1 + n2) / (n12 * (n12 + dot12))
        else:
            proj = (r0x * (r1x / n1 - r2x / n2) + r0y * (r1y / n1 - r2y / n2)
                    + r0z * (r1z / n1 - r2z / n2))
        den = cr2 + rc2 * r0sq
        if den > _FLOOR:
            k = _INV_4PI * proj / den
    v[0] = cx * k
    v[1] = cy * k
    v[2] = cz * k


cdef inline void _ray_v(double px, double py, double pz, double ax, double ay, double az,
                        double dx, double dy, double dz, double rc2, double* v) noexcept nogil:
    # Velocity of a unit semi-infinite line from a along d at p.
    # Behind the start point the exact law is multiplied by D^2 / (D^2 + rc2), D = |r|.
    cdef double rx = px - ax, ry = py - ay, rz = pz - az
    cdef double cx = dy * rz - dz * ry
    cdef double cy = dz * rx - dx * rz
    cdef double cz = dx * ry - dy * rx
    cdef double cr2 = cx * cx + cy * cy + cz * cz
    cdef double dd = rx * rx + ry * ry + rz * rz + _FLOOR
    cdef double rn = sqrt(dd)
    cdef double dr = dx * rx + dy * ry + dz * rz
    cdef double e, den, k
    k = 0.0
    if dr < 0.0:
        e = rn * (rn - dr)
        if e > _FLOOR:
            k = _INV_4PI / e * (dd / (dd + rc2))
    else:
        den = cr2 + rc2
        if den > _FLOOR:
            k = _INV_4PI * (1.0 + dr / rn) / den
    v[0] = cx * k
    v[1] = cy * k
    v[2] = cz * k


cdef inline void _seg_fb(double px, double py, double pz, double ax, double ay, double az,
                         double bx, double by, double bz, double rc2,
                         double ux, double uy, double uz, double* o) noexcept nogil:
    # Velocity and VJP of a unit finite segment a -> b at p for the adjoint u.
    # o: v (3), gP (3), gA (3), gB (3), grc2.
    cdef double r1x = px - ax, r1y = py - ay, r1z = pz - az
    cdef double r2x = px - bx, r2y = py - by, r2z = pz - bz
    cdef double r0x = bx - ax, r0y = by - ay, r0z = bz - az
    cdef double cx = r1y * r2z - r1z * r2y
    cdef double cy = r1z * r2x - r1x * r2z
    cdef double cz = r1x * r2y - r1y * r2x
    cdef double cr2 = cx * cx + cy * cy + cz * cz
    cdef double r0sq = r0x * r0x + r0y * r0y + r0z * r0z
    cdef double d1 = r1x * r1x + r1y * r1y + r1z * r1z + _FLOOR
    cdef double d2 = r2x * r2x + r2y * r2y + r2z * r2z + _FLOOR
    cdef double n1 = sqrt(d1)
    cdef double n2 = sqrt(d2)
    cdef double dot12 = r1x * r2x + r1y * r2y + r1z * r2z
    cdef bint before = r0x * r1x + r0y * r1y + r0z * r1z < 0.0
    cdef bint after = r0x * r2x + r0y * r2y + r0z * r2z > 0.0
    cdef bint stab = dot12 > 0.0
    cdef double n12 = n1 * n2
    cdef double t1x = r1x / n1, t1y = r1y / n1, t1z = r1z / n1
    cdef double t2x = r2x / n2, t2y = r2y / n2, t2z = r2z / n2
    cdef double den_out, proj, den, k, gcx, gcy, gcz, gk, gproj, gden, gcr2, grc2, gr0sq
    cdef double gr1x = 0.0, gr1y = 0.0, gr1z = 0.0, gr2x = 0.0, gr2y = 0.0, gr2z = 0.0
    cdef double gr0x = 0.0, gr0y = 0.0, gr0z = 0.0
    cdef double gsum, gD, gn12, gdot12, gn1, gn2, gt1x, gt1y, gt1z, f1, f2
    cdef double dd, dsum, fac, hh, gh, gfac, gdd
    cdef int q
    if before or after:
        # Exact law times the core factor:
        # k = (n1 + n2) / (4 pi n1 n2 e) * D2 / (D2 + rc2), e = n1 n2 + r1 . r2.
        if not (n12 + dot12) > _FLOOR:
            for q in range(13):
                o[q] = 0.0
            return
        den_out = n12 * (n12 + dot12)
        dd = d1 if before else d2
        dsum = dd + rc2
        fac = dd / dsum
        hh = (n1 + n2) / den_out
        k = _INV_4PI * hh * fac
        o[0] = cx * k
        o[1] = cy * k
        o[2] = cz * k
        gcx = k * ux
        gcy = k * uy
        gcz = k * uz
        gk = ux * cx + uy * cy + uz * cz
        gh = gk * _INV_4PI * fac
        gfac = gk * _INV_4PI * hh
        # dF/dD2 = rc2 / (D2 + rc2)^2, dF/drc2 = -D2 / (D2 + rc2)^2.
        gdd = gfac * rc2 / (dsum * dsum)
        grc2 = -gfac * dd / (dsum * dsum)
        gcr2 = 0.0
        gr0sq = 0.0
        gsum = gh / den_out
        gD = -gh * hh / den_out
        gn12 = gD * (2.0 * n12 + dot12)
        gdot12 = gD * n12
        gn1 = gsum + gn12 * n2
        gn2 = gsum + gn12 * n1
        gr1x = gdot12 * r2x
        gr1y = gdot12 * r2y
        gr1z = gdot12 * r2z
        gr2x = gdot12 * r1x
        gr2y = gdot12 * r1y
        gr2z = gdot12 * r1z
        # dD2/dr1 = 2 r1 (before), dD2/dr2 = 2 r2 (after).
        if before:
            gr1x = gr1x + 2.0 * gdd * r1x
            gr1y = gr1y + 2.0 * gdd * r1y
            gr1z = gr1z + 2.0 * gdd * r1z
        else:
            gr2x = gr2x + 2.0 * gdd * r2x
            gr2y = gr2y + 2.0 * gdd * r2y
            gr2z = gr2z + 2.0 * gdd * r2z
    else:
        if stab:
            den_out = n12 * (n12 + dot12)
            proj = cr2 * (n1 + n2) / den_out
        else:
            den_out = 1.0
            proj = r0x * (t1x - t2x) + r0y * (t1y - t2y) + r0z * (t1z - t2z)
        den = cr2 + rc2 * r0sq
        if not den > _FLOOR:
            for q in range(13):
                o[q] = 0.0
            return
        k = _INV_4PI * proj / den
        o[0] = cx * k
        o[1] = cy * k
        o[2] = cz * k
        gcx = k * ux
        gcy = k * uy
        gcz = k * uz
        gk = ux * cx + uy * cy + uz * cz
        gproj = gk * _INV_4PI / den
        gden = -gk * k / den
        gcr2 = gden
        grc2 = gden * r0sq
        gr0sq = gden * rc2
        if stab:
            gcr2 = gcr2 + gproj * (n1 + n2) / den_out
            gsum = gproj * cr2 / den_out
            gD = -gproj * proj / den_out
            gn12 = gD * (2.0 * n12 + dot12)
            gdot12 = gD * n12
            gn1 = gsum + gn12 * n2
            gn2 = gsum + gn12 * n1
            gr1x = gdot12 * r2x
            gr1y = gdot12 * r2y
            gr1z = gdot12 * r2z
            gr2x = gdot12 * r1x
            gr2y = gdot12 * r1y
            gr2z = gdot12 * r1z
        else:
            gr0x = gproj * (t1x - t2x)
            gr0y = gproj * (t1y - t2y)
            gr0z = gproj * (t1z - t2z)
            gt1x = gproj * r0x
            gt1y = gproj * r0y
            gt1z = gproj * r0z
            gr1x = gt1x / n1
            gr1y = gt1y / n1
            gr1z = gt1z / n1
            gr2x = -gt1x / n2
            gr2y = -gt1y / n2
            gr2z = -gt1z / n2
            gn1 = -(gt1x * t1x + gt1y * t1y + gt1z * t1z) / n1
            gn2 = (gt1x * t2x + gt1y * t2y + gt1z * t2z) / n2
    f1 = gn1 / n1
    f2 = gn2 / n2
    gr1x = gr1x + f1 * r1x
    gr1y = gr1y + f1 * r1y
    gr1z = gr1z + f1 * r1z
    gr2x = gr2x + f2 * r2x
    gr2y = gr2y + f2 * r2y
    gr2z = gr2z + f2 * r2z
    gcx = gcx + 2.0 * gcr2 * cx
    gcy = gcy + 2.0 * gcr2 * cy
    gcz = gcz + 2.0 * gcr2 * cz
    # c = r1 x r2: g_r1 += r2 x gc, g_r2 += gc x r1.
    gr1x = gr1x + (r2y * gcz - r2z * gcy)
    gr1y = gr1y + (r2z * gcx - r2x * gcz)
    gr1z = gr1z + (r2x * gcy - r2y * gcx)
    gr2x = gr2x + (gcy * r1z - gcz * r1y)
    gr2y = gr2y + (gcz * r1x - gcx * r1z)
    gr2z = gr2z + (gcx * r1y - gcy * r1x)
    gr0x = gr0x + 2.0 * gr0sq * r0x
    gr0y = gr0y + 2.0 * gr0sq * r0y
    gr0z = gr0z + 2.0 * gr0sq * r0z
    o[3] = gr1x + gr2x
    o[4] = gr1y + gr2y
    o[5] = gr1z + gr2z
    o[6] = -gr1x - gr0x
    o[7] = -gr1y - gr0y
    o[8] = -gr1z - gr0z
    o[9] = gr0x - gr2x
    o[10] = gr0y - gr2y
    o[11] = gr0z - gr2z
    o[12] = grc2


cdef inline void _ray_fb(double px, double py, double pz, double ax, double ay, double az,
                         double dx, double dy, double dz, double rc2,
                         double ux, double uy, double uz, double* o) noexcept nogil:
    # Velocity and VJP of a unit semi-infinite line from a along d at p.
    # o: v (3), gP (3), gd (3), grc2; gA is -gP.
    cdef double rx = px - ax, ry = py - ay, rz = pz - az
    cdef double cx = dy * rz - dz * ry
    cdef double cy = dz * rx - dx * rz
    cdef double cz = dx * ry - dy * rx
    cdef double cr2 = cx * cx + cy * cy + cz * cz
    cdef double dd = rx * rx + ry * ry + rz * rz + _FLOOR
    cdef double rn = sqrt(dd)
    cdef double dr = dx * rx + dy * ry + dz * rz
    cdef double den_b, oc, den, k, gcx, gcy, gcz, gk, goc, gden, gcr2, grc2, gE, grn, gdr, f
    cdef double grx, gry, grz, gdx, gdy, gdz
    cdef double dsum, fac, gfac
    cdef double gdd = 0.0
    cdef int q
    if dr < 0.0:
        # Exact law times the core factor: k = 1 / (4 pi E) * D2 / (D2 + rc2), E = rn (rn - dr).
        den_b = rn * (rn - dr)
        if not den_b > _FLOOR:
            for q in range(10):
                o[q] = 0.0
            return
        dsum = dd + rc2
        fac = dd / dsum
        k = _INV_4PI / den_b * fac
        o[0] = cx * k
        o[1] = cy * k
        o[2] = cz * k
        gcx = k * ux
        gcy = k * uy
        gcz = k * uz
        gk = ux * cx + uy * cy + uz * cz
        gfac = gk * _INV_4PI / den_b
        gE = -gk * k / den_b
        # dF/dD2 = rc2 / (D2 + rc2)^2, dF/drc2 = -D2 / (D2 + rc2)^2, dD2/dr = 2 r.
        gdd = gfac * rc2 / (dsum * dsum)
        grc2 = -gfac * dd / (dsum * dsum)
        gcr2 = 0.0
        grn = gE * (2.0 * rn - dr)
        gdr = -gE * rn
    else:
        oc = 1.0 + dr / rn
        den = cr2 + rc2
        if not den > _FLOOR:
            for q in range(10):
                o[q] = 0.0
            return
        k = _INV_4PI * oc / den
        o[0] = cx * k
        o[1] = cy * k
        o[2] = cz * k
        gcx = k * ux
        gcy = k * uy
        gcz = k * uz
        gk = ux * cx + uy * cy + uz * cz
        goc = gk * _INV_4PI / den
        gden = -gk * k / den
        gcr2 = gden
        grc2 = gden
        gdr = goc / rn
        grn = -goc * dr / (rn * rn)
    f = grn / rn + 2.0 * gdd
    grx = f * rx + gdr * dx
    gry = f * ry + gdr * dy
    grz = f * rz + gdr * dz
    gdx = gdr * rx
    gdy = gdr * ry
    gdz = gdr * rz
    gcx = gcx + 2.0 * gcr2 * cx
    gcy = gcy + 2.0 * gcr2 * cy
    gcz = gcz + 2.0 * gcr2 * cz
    # c = d x r: g_d += r x gc, g_r += gc x d.
    gdx = gdx + (ry * gcz - rz * gcy)
    gdy = gdy + (rz * gcx - rx * gcz)
    gdz = gdz + (rx * gcy - ry * gcx)
    grx = grx + (gcy * dz - gcz * dy)
    gry = gry + (gcz * dx - gcx * dz)
    grz = grz + (gcx * dy - gcy * dx)
    o[3] = grx
    o[4] = gry
    o[5] = grz
    o[6] = gdx
    o[7] = gdy
    o[8] = gdz
    o[9] = grc2


cdef void _forward_row(Py_ssize_t i, int mode, const double[:, ::1] P, const double[:, ::1] X,
                       const double[:, ::1] a, const double[:, ::1] b, const double[:, ::1] ate,
                       const double[:, ::1] bte, const double[:, ::1] d, const double[::1] rc2s,
                       const double[::1] tgrc2, const cnp.int64_t[::1] sgrp, const cnp.int64_t[::1] tgrp,
                       const cnp.int64_t[::1] own, const cnp.int64_t[::1] spanel, const cnp.int64_t[::1] edge,
                       const cnp.int64_t[::1] sea, const cnp.int64_t[::1] seb, const double[::1] s,
                       const cnp.int64_t[::1] col, bint use_tg,
                       const double[:, ::1] ak, const double[:, ::1] bk, const signed char[::1] kfl,
                       const cnp.int64_t[::1] seka, const cnp.int64_t[::1] sekb,
                       double[:, ::1] out) noexcept nogil:
    # One row of the forward product (mode 0: system rows, mode 1: induced velocity).
    # A leg with a node (kfl[j] != 0) runs ate -> ak -> a and b -> bk -> bte;
    # on the other sources ak is ate and bk is bte.
    cdef Py_ssize_t n = a.shape[0]
    cdef Py_ssize_t j
    cdef double v[3]
    cdef double w[3]
    cdef double vx, vy, vz, rc2, px = P[i, 0], py = P[i, 1], pz = P[i, 2]
    cdef double ox = 0.0, oy = 0.0, oz = 0.0
    cdef bint oth
    for j in range(n):
        oth = use_tg and tgrp[i] != sgrp[j]
        rc2 = rc2s[j] + tgrc2[i] if oth else rc2s[j]
        vx = 0.0
        vy = 0.0
        vz = 0.0
        if own[i] != spanel[j]:
            _seg_v(px, py, pz, a[j, 0], a[j, 1], a[j, 2], b[j, 0], b[j, 1], b[j, 2], rc2, v)
            vx = v[0]
            vy = v[1]
            vz = v[2]
        if edge[i] != sea[j]:
            _seg_v(px, py, pz, ak[j, 0], ak[j, 1], ak[j, 2], a[j, 0], a[j, 1], a[j, 2], rc2, v)
            vx = vx + v[0]
            vy = vy + v[1]
            vz = vz + v[2]
        if kfl[j] != 0 and edge[i] != seka[j]:
            _seg_v(px, py, pz, ate[j, 0], ate[j, 1], ate[j, 2], ak[j, 0], ak[j, 1], ak[j, 2], rc2, v)
            vx = vx + v[0]
            vy = vy + v[1]
            vz = vz + v[2]
        if edge[i] != seb[j]:
            _seg_v(px, py, pz, b[j, 0], b[j, 1], b[j, 2], bk[j, 0], bk[j, 1], bk[j, 2], rc2, v)
            vx = vx + v[0]
            vy = vy + v[1]
            vz = vz + v[2]
        if kfl[j] != 0 and edge[i] != sekb[j]:
            _seg_v(px, py, pz, bk[j, 0], bk[j, 1], bk[j, 2], bte[j, 0], bte[j, 1], bte[j, 2], rc2, v)
            vx = vx + v[0]
            vy = vy + v[1]
            vz = vz + v[2]
        _ray_v(px, py, pz, bte[j, 0], bte[j, 1], bte[j, 2], d[j, 0], d[j, 1], d[j, 2], rc2, v)
        _ray_v(px, py, pz, ate[j, 0], ate[j, 1], ate[j, 2], d[j, 0], d[j, 1], d[j, 2], rc2, w)
        vx = vx + v[0] - w[0]
        vy = vy + v[1] - w[1]
        vz = vz + v[2] - w[2]
        if mode == 0:
            out[i, col[j]] = out[i, col[j]] + s[j] * (vx * X[i, 0] + vy * X[i, 1] + vz * X[i, 2])
        else:
            ox = ox + s[j] * vx
            oy = oy + s[j] * vy
            oz = oz + s[j] * vz
    if mode != 0:
        out[i, 0] = ox
        out[i, 1] = oy
        out[i, 2] = oz


def horseshoe_forward(int mode, const double[:, ::1] P, const double[:, ::1] X, const double[:, ::1] a,
                      const double[:, ::1] b, const double[:, ::1] ate, const double[:, ::1] bte,
                      const double[:, ::1] d, const double[::1] rc2s, const double[::1] tgrc2,
                      const cnp.int64_t[::1] sgrp, const cnp.int64_t[::1] tgrp, const cnp.int64_t[::1] own,
                      const cnp.int64_t[::1] spanel, const cnp.int64_t[::1] edge, const cnp.int64_t[::1] sea,
                      const cnp.int64_t[::1] seb, const double[::1] s, const cnp.int64_t[::1] col,
                      int n_u, bint use_tg, int num_threads, ak=None, bk=None, kfl=None, seka=None, sekb=None):
    """Return the system rows (mode 0, (m, n_u)) or the induced velocity (mode 1, (m, 3)).

    The arguments are those of :func:`ventorum.aero.vortex_vjp.horseshoe_forward`
    (with *mode* first), plus the OpenMP thread count *num_threads*. The
    leg nodes *ak*, *bk* (n, 3), their flags *kfl* (int8, n) and the edge
    ids *seka*, *sekb* of the aft segments are optional (None: no nodes).

    Returns
    -------
    numpy.ndarray
        The product in float64.
    """
    cdef Py_ssize_t m = P.shape[0]
    cdef Py_ssize_t i
    cdef Py_ssize_t ncol = n_u if mode == 0 else 3
    cdef const double[:, ::1] ak_v = ate if ak is None else ak
    cdef const double[:, ::1] bk_v = bte if bk is None else bk
    cdef const signed char[::1] kfl_v = np.zeros(a.shape[0], dtype=np.int8) if kfl is None else kfl
    cdef const cnp.int64_t[::1] seka_v = sea if seka is None else seka
    cdef const cnp.int64_t[::1] sekb_v = seb if sekb is None else sekb
    out = np.zeros((m, ncol), dtype=np.float64)
    cdef double[:, ::1] out_v = out
    for i in prange(m, num_threads=num_threads, nogil=True, schedule="static"):
        _forward_row(i, mode, P, X, a, b, ate, bte, d, rc2s, tgrc2, sgrp, tgrp, own, spanel, edge,
                     sea, seb, s, col, use_tg, ak_v, bk_v, kfl_v, seka_v, sekb_v, out_v)
    return out


cdef void _vjp_block(Py_ssize_t t, Py_ssize_t nblocks, const double[:, ::1] P, const double[:, ::1] X,
                     const double[:, ::1] a, const double[:, ::1] b, const double[:, ::1] ate,
                     const double[:, ::1] bte, const double[:, ::1] d, const double[::1] rc2s,
                     const double[::1] tgrc2, const cnp.int64_t[::1] sgrp, const cnp.int64_t[::1] tgrp,
                     const cnp.int64_t[::1] own, const cnp.int64_t[::1] spanel, const cnp.int64_t[::1] edge,
                     const cnp.int64_t[::1] sea, const cnp.int64_t[::1] seb, const double[::1] s,
                     const cnp.int64_t[::1] col, const double[:, ::1] W, bint use_W, bint use_tg,
                     const double[:, ::1] ak, const double[:, ::1] bk, const signed char[::1] kfl,
                     const cnp.int64_t[::1] seka, const cnp.int64_t[::1] sekb,
                     double[:, ::1] gP, double[:, ::1] gX, double[::1] gtg,
                     double[:, :, ::1] buf) noexcept nogil:
    # The rows of block t: target derivatives into gP, gX, gtg; source derivatives into buf[t].
    cdef Py_ssize_t m = P.shape[0]
    cdef Py_ssize_t n = a.shape[0]
    cdef Py_ssize_t i0 = (t * m) // nblocks
    cdef Py_ssize_t i1 = ((t + 1) * m) // nblocks
    cdef Py_ssize_t i, j
    cdef double r[13]
    cdef double px, py, pz, xx, xy, xz, aPx, aPy, aPz, aXx, aXy, aXz, atg
    cdef double wf, cij, rc2, ux, uy, uz, vx, vy, vz, grc
    cdef double ax, ay, az, bx, by, bz, tax, tay, taz, tbx, tby, tbz, dx, dy, dz
    cdef bint oth, kj
    cdef int ka, kb
    for i in range(i0, i1):
        px = P[i, 0]
        py = P[i, 1]
        pz = P[i, 2]
        xx = X[i, 0]
        xy = X[i, 1]
        xz = X[i, 2]
        aPx = 0.0
        aPy = 0.0
        aPz = 0.0
        aXx = 0.0
        aXy = 0.0
        aXz = 0.0
        atg = 0.0
        for j in range(n):
            wf = W[i, col[j]] if use_W else 1.0
            cij = wf * s[j]
            oth = use_tg and tgrp[i] != sgrp[j]
            rc2 = rc2s[j] + tgrc2[i] if oth else rc2s[j]
            ux = cij * xx
            uy = cij * xy
            uz = cij * xz
            ax = a[j, 0]
            ay = a[j, 1]
            az = a[j, 2]
            bx = b[j, 0]
            by = b[j, 1]
            bz = b[j, 2]
            tax = ate[j, 0]
            tay = ate[j, 1]
            taz = ate[j, 2]
            tbx = bte[j, 0]
            tby = bte[j, 1]
            tbz = bte[j, 2]
            dx = d[j, 0]
            dy = d[j, 1]
            dz = d[j, 2]
            vx = 0.0
            vy = 0.0
            vz = 0.0
            grc = 0.0
            if own[i] != spanel[j]:
                _seg_fb(px, py, pz, ax, ay, az, bx, by, bz, rc2, ux, uy, uz, r)
                vx = vx + r[0]
                vy = vy + r[1]
                vz = vz + r[2]
                aPx = aPx + r[3]
                aPy = aPy + r[4]
                aPz = aPz + r[5]
                buf[t, j, 0] += r[6]
                buf[t, j, 1] += r[7]
                buf[t, j, 2] += r[8]
                buf[t, j, 3] += r[9]
                buf[t, j, 4] += r[10]
                buf[t, j, 5] += r[11]
                grc = grc + r[12]
            kj = kfl[j] != 0
            # Start of the forward a-leg segment (the node, or a_te) and end
            # of the forward b-leg segment (the node, or b_te) in the buffer.
            ka = 17 if kj else 6
            kb = 20 if kj else 9
            if edge[i] != sea[j]:
                _seg_fb(px, py, pz, ak[j, 0], ak[j, 1], ak[j, 2], ax, ay, az, rc2, ux, uy, uz, r)
                vx = vx + r[0]
                vy = vy + r[1]
                vz = vz + r[2]
                aPx = aPx + r[3]
                aPy = aPy + r[4]
                aPz = aPz + r[5]
                buf[t, j, ka] += r[6]
                buf[t, j, ka + 1] += r[7]
                buf[t, j, ka + 2] += r[8]
                buf[t, j, 0] += r[9]
                buf[t, j, 1] += r[10]
                buf[t, j, 2] += r[11]
                grc = grc + r[12]
            if kj and edge[i] != seka[j]:
                _seg_fb(px, py, pz, tax, tay, taz, ak[j, 0], ak[j, 1], ak[j, 2], rc2, ux, uy, uz, r)
                vx = vx + r[0]
                vy = vy + r[1]
                vz = vz + r[2]
                aPx = aPx + r[3]
                aPy = aPy + r[4]
                aPz = aPz + r[5]
                buf[t, j, 6] += r[6]
                buf[t, j, 7] += r[7]
                buf[t, j, 8] += r[8]
                buf[t, j, 17] += r[9]
                buf[t, j, 18] += r[10]
                buf[t, j, 19] += r[11]
                grc = grc + r[12]
            if edge[i] != seb[j]:
                _seg_fb(px, py, pz, bx, by, bz, bk[j, 0], bk[j, 1], bk[j, 2], rc2, ux, uy, uz, r)
                vx = vx + r[0]
                vy = vy + r[1]
                vz = vz + r[2]
                aPx = aPx + r[3]
                aPy = aPy + r[4]
                aPz = aPz + r[5]
                buf[t, j, 3] += r[6]
                buf[t, j, 4] += r[7]
                buf[t, j, 5] += r[8]
                buf[t, j, kb] += r[9]
                buf[t, j, kb + 1] += r[10]
                buf[t, j, kb + 2] += r[11]
                grc = grc + r[12]
            if kj and edge[i] != sekb[j]:
                _seg_fb(px, py, pz, bk[j, 0], bk[j, 1], bk[j, 2], tbx, tby, tbz, rc2, ux, uy, uz, r)
                vx = vx + r[0]
                vy = vy + r[1]
                vz = vz + r[2]
                aPx = aPx + r[3]
                aPy = aPy + r[4]
                aPz = aPz + r[5]
                buf[t, j, 20] += r[6]
                buf[t, j, 21] += r[7]
                buf[t, j, 22] += r[8]
                buf[t, j, 9] += r[9]
                buf[t, j, 10] += r[10]
                buf[t, j, 11] += r[11]
                grc = grc + r[12]
            _ray_fb(px, py, pz, tbx, tby, tbz, dx, dy, dz, rc2, ux, uy, uz, r)
            vx = vx + r[0]
            vy = vy + r[1]
            vz = vz + r[2]
            aPx = aPx + r[3]
            aPy = aPy + r[4]
            aPz = aPz + r[5]
            buf[t, j, 9] -= r[3]
            buf[t, j, 10] -= r[4]
            buf[t, j, 11] -= r[5]
            buf[t, j, 12] += r[6]
            buf[t, j, 13] += r[7]
            buf[t, j, 14] += r[8]
            grc = grc + r[9]
            _ray_fb(px, py, pz, tax, tay, taz, dx, dy, dz, rc2, -ux, -uy, -uz, r)
            vx = vx - r[0]
            vy = vy - r[1]
            vz = vz - r[2]
            aPx = aPx + r[3]
            aPy = aPy + r[4]
            aPz = aPz + r[5]
            buf[t, j, 6] -= r[3]
            buf[t, j, 7] -= r[4]
            buf[t, j, 8] -= r[5]
            buf[t, j, 12] += r[6]
            buf[t, j, 13] += r[7]
            buf[t, j, 14] += r[8]
            grc = grc + r[9]
            aXx = aXx + cij * vx
            aXy = aXy + cij * vy
            aXz = aXz + cij * vz
            buf[t, j, 15] += grc
            buf[t, j, 16] += wf * (xx * vx + xy * vy + xz * vz)
            if oth:
                atg = atg + grc
        gP[i, 0] = aPx
        gP[i, 1] = aPy
        gP[i, 2] = aPz
        gX[i, 0] = aXx
        gX[i, 1] = aXy
        gX[i, 2] = aXz
        gtg[i] = atg


def horseshoe_vjp(const double[:, ::1] P, const double[:, ::1] X, const double[:, ::1] a,
                  const double[:, ::1] b, const double[:, ::1] ate, const double[:, ::1] bte,
                  const double[:, ::1] d, const double[::1] rc2s, const double[::1] tgrc2,
                  const cnp.int64_t[::1] sgrp, const cnp.int64_t[::1] tgrp, const cnp.int64_t[::1] own,
                  const cnp.int64_t[::1] spanel, const cnp.int64_t[::1] edge, const cnp.int64_t[::1] sea,
                  const cnp.int64_t[::1] seb, const double[::1] s, const cnp.int64_t[::1] col,
                  const double[:, ::1] W, bint use_W, bint use_tg, int nblocks, int num_threads,
                  ak=None, bk=None, kfl=None, seka=None, sekb=None):
    """Return the VJP (gP, gX, g_tg_rc2, ga, gb, ga_te, gb_te, gd, g_rc2_src, gs) in float64.

    The arguments are those of :func:`ventorum.aero.vortex_vjp.horseshoe_vjp`,
    with *W* a (1, 1) dummy and *use_W* False for the induced velocity,
    plus the number of row blocks *nblocks* and the OpenMP thread count
    *num_threads*. The leg nodes are optional, as in :func:`horseshoe_forward`;
    with them the tuple has two more items, ``ga_k`` and ``gb_k`` (zero on a
    source without a node).

    Returns
    -------
    tuple of numpy.ndarray
        The derivatives, as in :func:`ventorum.aero.vortex_vjp.horseshoe_vjp`.
    """
    cdef Py_ssize_t m = P.shape[0]
    cdef Py_ssize_t n = a.shape[0]
    cdef Py_ssize_t t, j, k
    gP = np.zeros((m, 3))
    gX = np.zeros((m, 3))
    gtg = np.zeros(m)
    buf = np.zeros((nblocks, n, NSRC))
    cdef const double[:, ::1] ak_v = ate if ak is None else ak
    cdef const double[:, ::1] bk_v = bte if bk is None else bk
    cdef const signed char[::1] kfl_v = np.zeros(n, dtype=np.int8) if kfl is None else kfl
    cdef const cnp.int64_t[::1] seka_v = sea if seka is None else seka
    cdef const cnp.int64_t[::1] sekb_v = seb if sekb is None else sekb
    cdef double[:, ::1] gP_v = gP
    cdef double[:, ::1] gX_v = gX
    cdef double[::1] gtg_v = gtg
    cdef double[:, :, ::1] buf_v = buf
    for t in prange(nblocks, num_threads=num_threads, nogil=True, schedule="static", chunksize=1):
        _vjp_block(t, nblocks, P, X, a, b, ate, bte, d, rc2s, tgrc2, sgrp, tgrp, own, spanel, edge, sea, seb,
                   s, col, W, use_W, use_tg, ak_v, bk_v, kfl_v, seka_v, sekb_v, gP_v, gX_v, gtg_v, buf_v)
    acc = np.zeros((n, NSRC))
    cdef double[:, ::1] acc_v = acc
    for t in range(nblocks):
        for j in range(n):
            for k in range(NSRC):
                acc_v[j, k] += buf_v[t, j, k]
    out = (gP, gX, gtg, acc[:, 0:3].copy(), acc[:, 3:6].copy(), acc[:, 6:9].copy(),
           acc[:, 9:12].copy(), acc[:, 12:15].copy(), acc[:, 15].copy(), acc[:, 16].copy())
    if ak is None:
        return out
    return out + (acc[:, 17:20].copy(), acc[:, 20:23].copy())


cdef void _trefftz_row(Py_ssize_t i, const double[:, ::1] q, const double[:, ::1] qn, const double[:, ::1] p,
                       const double[::1] gamma, const double[::1] d, const double[::1] rc2,
                       const double[::1] tgrc2, const cnp.int64_t[::1] sgrp, const cnp.int64_t[::1] tgrp,
                       bint use_tg, double[::1] out) noexcept nogil:
    cdef Py_ssize_t n = p.shape[0]
    cdef Py_ssize_t j
    cdef double acc = 0.0, c2, rx, ry, rz, vx, vy, vz, k
    cdef bint oth
    for j in range(n):
        oth = use_tg and tgrp[i] != sgrp[j]
        c2 = rc2[j] + tgrc2[i] if oth else rc2[j]
        rx = q[i, 0] - p[j, 0]
        ry = q[i, 1] - p[j, 1]
        rz = q[i, 2] - p[j, 2]
        vx = d[1] * rz - d[2] * ry
        vy = d[2] * rx - d[0] * rz
        vz = d[0] * ry - d[1] * rx
        k = _INV_2PI * gamma[j] / (rx * rx + ry * ry + rz * rz + c2)
        acc = acc + (vx * qn[i, 0] + vy * qn[i, 1] + vz * qn[i, 2]) * k
    out[i] = acc


def trefftz_forward(const double[:, ::1] q, const double[:, ::1] qn, const double[:, ::1] p,
                    const double[::1] gamma, const double[::1] d, const double[::1] rc2,
                    const double[::1] tgrc2, const cnp.int64_t[::1] sgrp, const cnp.int64_t[::1] tgrp,
                    bint use_tg, int num_threads):
    """Return the normal wash (m,) of 2-D point vortices at Trefftz-plane points.

    The arguments are those of :func:`ventorum.aero.vortex_vjp.trefftz_forward`,
    plus the OpenMP thread count *num_threads*.

    Returns
    -------
    numpy.ndarray
        The normal wash [m/s] in float64.
    """
    cdef Py_ssize_t m = q.shape[0]
    cdef Py_ssize_t i
    out = np.zeros(m)
    cdef double[::1] out_v = out
    for i in prange(m, num_threads=num_threads, nogil=True, schedule="static"):
        _trefftz_row(i, q, qn, p, gamma, d, rc2, tgrc2, sgrp, tgrp, use_tg, out_v)
    return out


cdef void _trefftz_block(Py_ssize_t t, Py_ssize_t nblocks, const double[:, ::1] q, const double[:, ::1] qn,
                         const double[:, ::1] p, const double[::1] gamma, const double[::1] d,
                         const double[::1] rc2, const double[::1] tgrc2, const cnp.int64_t[::1] sgrp,
                         const cnp.int64_t[::1] tgrp, bint use_tg, const double[::1] gw,
                         double[:, ::1] gq, double[:, ::1] gqn, double[::1] gtg,
                         double[:, :, ::1] buf, double[:, ::1] dbuf) noexcept nogil:
    cdef Py_ssize_t m = q.shape[0]
    cdef Py_ssize_t n = p.shape[0]
    cdef Py_ssize_t i0 = (t * m) // nblocks
    cdef Py_ssize_t i1 = ((t + 1) * m) // nblocks
    cdef Py_ssize_t i, j
    cdef double dx = d[0], dy = d[1], dz = d[2]
    cdef double g, qnx, qny, qnz, aqx, aqy, aqz, anx, any_, anz, atg
    cdef double c2, rx, ry, rz, vx, vy, vz, D, k, e, ge, gk, gD, gvx, gvy, gvz, grx, gry, grz
    cdef bint oth
    for i in range(i0, i1):
        g = gw[i]
        qnx = qn[i, 0]
        qny = qn[i, 1]
        qnz = qn[i, 2]
        aqx = 0.0
        aqy = 0.0
        aqz = 0.0
        anx = 0.0
        any_ = 0.0
        anz = 0.0
        atg = 0.0
        for j in range(n):
            oth = use_tg and tgrp[i] != sgrp[j]
            c2 = rc2[j] + tgrc2[i] if oth else rc2[j]
            rx = q[i, 0] - p[j, 0]
            ry = q[i, 1] - p[j, 1]
            rz = q[i, 2] - p[j, 2]
            vx = dy * rz - dz * ry
            vy = dz * rx - dx * rz
            vz = dx * ry - dy * rx
            D = rx * rx + ry * ry + rz * rz + c2
            k = _INV_2PI * gamma[j] / D
            e = vx * qnx + vy * qny + vz * qnz
            ge = g * k
            gk = g * e
            gD = -gk * k / D
            gvx = ge * qnx
            gvy = ge * qny
            gvz = ge * qnz
            grx = 2.0 * gD * rx + (gvy * dz - gvz * dy)
            gry = 2.0 * gD * ry + (gvz * dx - gvx * dz)
            grz = 2.0 * gD * rz + (gvx * dy - gvy * dx)
            aqx = aqx + grx
            aqy = aqy + gry
            aqz = aqz + grz
            anx = anx + ge * vx
            any_ = any_ + ge * vy
            anz = anz + ge * vz
            if oth:
                atg = atg + gD
            buf[t, j, 0] -= grx
            buf[t, j, 1] -= gry
            buf[t, j, 2] -= grz
            buf[t, j, 3] += gk * _INV_2PI / D
            buf[t, j, 4] += gD
            dbuf[t, 0] += ry * gvz - rz * gvy
            dbuf[t, 1] += rz * gvx - rx * gvz
            dbuf[t, 2] += rx * gvy - ry * gvx
        gq[i, 0] = aqx
        gq[i, 1] = aqy
        gq[i, 2] = aqz
        gqn[i, 0] = anx
        gqn[i, 1] = any_
        gqn[i, 2] = anz
        gtg[i] = atg


def trefftz_vjp(const double[:, ::1] q, const double[:, ::1] qn, const double[:, ::1] p,
                const double[::1] gamma, const double[::1] d, const double[::1] rc2,
                const double[::1] tgrc2, const cnp.int64_t[::1] sgrp, const cnp.int64_t[::1] tgrp,
                bint use_tg, const double[::1] gw, int nblocks, int num_threads):
    """Return the VJP (gq, gqn, g_tg_rc2, gp, ggamma, g_rc2, gd) of the Trefftz normal wash.

    The arguments are those of :func:`ventorum.aero.vortex_vjp.trefftz_vjp`,
    plus the number of row blocks *nblocks* and the OpenMP thread count
    *num_threads*.

    Returns
    -------
    tuple of numpy.ndarray
        The derivatives, as in :func:`ventorum.aero.vortex_vjp.trefftz_vjp`.
    """
    cdef Py_ssize_t m = q.shape[0]
    cdef Py_ssize_t n = p.shape[0]
    cdef Py_ssize_t t, j, k
    gq = np.zeros((m, 3))
    gqn = np.zeros((m, 3))
    gtg = np.zeros(m)
    buf = np.zeros((nblocks, n, NVTX))
    dbuf = np.zeros((nblocks, 3))
    cdef double[:, ::1] gq_v = gq
    cdef double[:, ::1] gqn_v = gqn
    cdef double[::1] gtg_v = gtg
    cdef double[:, :, ::1] buf_v = buf
    cdef double[:, ::1] dbuf_v = dbuf
    for t in prange(nblocks, num_threads=num_threads, nogil=True, schedule="static", chunksize=1):
        _trefftz_block(t, nblocks, q, qn, p, gamma, d, rc2, tgrc2, sgrp, tgrp, use_tg, gw,
                       gq_v, gqn_v, gtg_v, buf_v, dbuf_v)
    acc = np.zeros((n, NVTX))
    gd = np.zeros(3)
    cdef double[:, ::1] acc_v = acc
    cdef double[::1] gd_v = gd
    for t in range(nblocks):
        for j in range(n):
            for k in range(NVTX):
                acc_v[j, k] += buf_v[t, j, k]
        for k in range(3):
            gd_v[k] += dbuf_v[t, k]
    return gq, gqn, gtg, acc[:, 0:3].copy(), acc[:, 3].copy(), acc[:, 4].copy(), gd
