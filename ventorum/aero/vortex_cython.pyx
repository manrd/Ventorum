# Author: Manuel Alejandro Rodriguez Diaz, PhD
#cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""Compiled (Cython) versions of the bent-horseshoe kernels of :mod:`ventorum.aero.vortex`.

The functions here compute the same quantities as the numpy kernels, with
the same regularisation and the same floors, in one pass over each
(point, source) pair. They do not build ``(m, n, 3)`` temporary arrays, and
they run the points in parallel (OpenMP ``prange``). The numpy kernels stay
the reference: the tests require both paths to agree to round-off.

Inside one horseshoe the legs share their vectors and norms, and
zero-length chordwise legs add nothing. Straight horseshoes with unit signs
take a short path with no fold of columns.

The kernels need flat arrays. :func:`ventorum.aero.vortex_numba.pack_sources`
and :func:`ventorum.aero.vortex_numba.pack_targets` prepare them from a
:class:`~ventorum.aero.vortex.HorseshoeSet` and
:class:`~ventorum.aero.vortex.Targets`. Each kernel takes the same
arguments as its Numba counterpart, in the same order, plus one last
argument ``num_threads``.

The formulas and their sources are in :mod:`ventorum.aero.vortex`
(Biot-Savart law: J. Katz and A. Plotkin, "Low-Speed Aerodynamics", 2nd ed., Cambridge University
Press, 2001).
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt, pi
from cython.parallel cimport prange

cnp.import_array()

cdef extern from *:
    """
    #ifdef _OPENMP
    #define VENTORUM_HAVE_OPENMP 1
    #else
    #define VENTORUM_HAVE_OPENMP 0
    #endif
    """
    int VENTORUM_HAVE_OPENMP


def openmp_enabled():
    """Return True if this extension was compiled with OpenMP.

    Without OpenMP the kernels give the same results, but each call runs
    in one thread and ``num_threads`` has no effect.

    Returns
    -------
    bool
        True when the C compiler defined ``_OPENMP`` for this build.
    """
    return bool(VENTORUM_HAVE_OPENMP)

def has_leg_kinks():
    """Return True: the kernels of this build take the leg nodes ``a_kink`` and ``b_kink``.

    The leg of a source with a node runs ``a_te -> a_kink -> a`` (and
    ``b -> b_kink -> b_te``), see :class:`ventorum.aero.vortex.HorseshoeSet`.

    Returns
    -------
    bool
        Always True for this source.
    """
    return True


cdef double _INV_4PI = 1.0 / (4.0 * pi)
cdef double _INV_2PI = 1.0 / (2.0 * pi)


cdef inline bint _is_direct(const cnp.int64_t[::1] column, const double[::1] sign,
                            int n_unknowns, Py_ssize_t n) noexcept nogil:
    # True when each source maps to its own unknown with unit sign.
    cdef Py_ssize_t j
    if n_unknowns != n:
        return False
    for j in range(n):
        if column[j] != j or sign[j] != 1.0:
            return False
    return True


cdef inline bint _is_straight(const double[:, ::1] a, const double[:, ::1] b,
                              const double[:, ::1] a_te, const double[:, ::1] b_te,
                              Py_ssize_t n) noexcept nogil:
    # True when every chordwise leg has zero length.
    cdef Py_ssize_t j
    for j in range(n):
        if (a_te[j, 0] != a[j, 0] or a_te[j, 1] != a[j, 1] or a_te[j, 2] != a[j, 2]
                or b_te[j, 0] != b[j, 0] or b_te[j, 1] != b[j, 1]
                or b_te[j, 2] != b[j, 2]):
            return False
    return True


cdef inline double _seg_k(double r1x, double r1y, double r1z, double r2x, double r2y, double r2z,
                          double r0x, double r0y, double r0z, double n1, double n2,
                          double cr2, double r0sq, double rc2) noexcept nogil:
    # Factor k of one finite segment (velocity = (r1 x r2) k), as in the numpy kernel.
    # Over the segment the core term is rc2 |r0|^2. Outside it (before the start
    # or after the end) the exact law is multiplied by D^2 / (D^2 + rc2), with D
    # the distance to the nearest end point.
    cdef double dd, e, proj, denom
    cdef bint before = r0x * r1x + r0y * r1y + r0z * r1z < 0.0
    cdef bint after = r0x * r2x + r0y * r2y + r0z * r2z > 0.0
    if before or after:
        if before:
            dd = r1x * r1x + r1y * r1y + r1z * r1z + 1e-300
        else:
            dd = r2x * r2x + r2y * r2y + r2z * r2z + 1e-300
        e = n1 * n2 + r1x * r2x + r1y * r2y + r1z * r2z
        if e > 1e-300:
            return _INV_4PI * (n1 + n2) / (n1 * n2 * e) * (dd / (dd + rc2))
        return 0.0
    proj = (r0x * (r1x / n1 - r2x / n2)
            + r0y * (r1y / n1 - r2y / n2)
            + r0z * (r1z / n1 - r2z / n2))
    denom = cr2 + rc2 * r0sq
    if denom > 1e-300:
        return _INV_4PI * proj / denom
    return 0.0


cdef inline double _ray_k(double rx, double ry, double rz, double rn, double dr,
                          double cr2, double rc2) noexcept nogil:
    # Factor k of one semi-infinite line (velocity = (d x r) k), as in the numpy
    # kernel; dr = d . r. Behind the start point the exact law is multiplied by
    # D^2 / (D^2 + rc2), with D = |r|.
    cdef double dd, e, denom
    if dr < 0.0:
        dd = rx * rx + ry * ry + rz * rz + 1e-300
        e = rn * (rn - dr)
        if e > 1e-300:
            return _INV_4PI / e * (dd / (dd + rc2))
        return 0.0
    denom = cr2 + rc2
    if denom > 1e-300:
        return _INV_4PI * (1.0 + dr / rn) / denom
    return 0.0


def induced_velocity_kernel(const double[:, ::1] P, const double[:, ::1] a, const double[:, ::1] b,
                            const double[:, ::1] a_te, const double[:, ::1] b_te, const double[:, ::1] d,
                            const double[::1] rc2_src, const cnp.int64_t[::1] src_group, const double[::1] g_src,
                            const cnp.int64_t[::1] tg_group, const double[::1] tg_rc2, bint use_tg,
                            int parts, int num_threads, a_kink=None, b_kink=None):
    """Return the total induced velocity at the evaluation points.

    Parameters
    ----------
    P : (m, 3)
        Evaluation points [m].
    a, b : (n, 3)
        Bound-vortex end points [m]. Positive circulation runs from ``a`` to ``b``.
    a_te, b_te : (n, 3)
        Trailing-edge points where the legs turn into the wake [m].
    d : (n, 3)
        Unit wake direction.
    rc2_src : (n,)
        Squared core radius of each source [m^2].
    src_group : (n,)
        Surface index of each source.
    g_src : (n,)
        Circulation of each source, sign included [m^2/s].
    tg_group : (m,)
        Surface index of each evaluation point.
    tg_rc2 : (m,)
        Squared core radius added for sources of other surfaces [m^2].
    use_tg : bool
        Whether the target core applies.
    parts : int
        0: all legs, 1: bound vortex and chordwise legs, 2: wake legs.
    num_threads : int
        Number of OpenMP threads.
    a_kink, b_kink : (n, 3) or None
        Nodes of the chordwise legs [m] (see ``a_kink`` in
        :class:`ventorum.aero.vortex.HorseshoeSet`), or None for straight legs.

    Returns
    -------
    (m, 3) induced velocity [m/s].
    """
    cdef Py_ssize_t m = P.shape[0]
    cdef Py_ssize_t n = a.shape[0]
    cdef cnp.ndarray[double, ndim=2] out
    cdef double[:, ::1] out_v
    cdef Py_ssize_t i, j
    cdef double rc2, g, vx, vy, vz, sx, sy, sz
    cdef double px, py, pz, ax, ay, az, bx, by, bz, tax, tay, taz, tbx, tby, tbz
    cdef bint straight_left, straight_right
    cdef double r1x, r1y, r1z, r2x, r2y, r2z, r0x, r0y, r0z, n1, n2
    cdef double cx, cy, cz, cr2, r0sq, k
    cdef double q1x, q1y, q1z, nq, w2x, w2y, w2z, nw
    cdef double kax, kay, kaz, kbx, kby, kbz, t1x, t1y, t1z, t2x, t2y, t2z, nt
    # Leg nodes (see ventorum.aero.vortex.HorseshoeSet); without them the
    # node arrays are a_te and b_te and the legs are computed as before.
    cdef bint kink = a_kink is not None
    cdef const double[:, ::1] ak = a_te if a_kink is None else a_kink
    cdef const double[:, ::1] bk = b_te if b_kink is None else b_kink
    cdef double dx, dy, dz
    cdef double rbrx, rbry, rbrz, rbrn, larx, lary, larz, larn
    cdef double qx, qy, qz, qcr2, qdr, qk
    cdef double ex, ey, ez, ecr2, edr, ek
    cdef double ux, uy, uz
    if parts == 0 and not use_tg and not kink and _is_straight(a, b, a_te, b_te, n):
        out = np.empty((m, 3), dtype=np.float64)
        out_v = out
        for i in prange(m, num_threads=num_threads, nogil=True, schedule="static"):
            # Assign the per-pair results here so each thread gets a private copy.
            vx = 0.0
            vy = 0.0
            vz = 0.0
            sx = 0.0
            sy = 0.0
            sz = 0.0
            px = P[i, 0]
            py = P[i, 1]
            pz = P[i, 2]
            for j in range(n):
                ax = a[j, 0]
                ay = a[j, 1]
                az = a[j, 2]
                bx = b[j, 0]
                by = b[j, 1]
                bz = b[j, 2]
                r1x = px - ax
                r1y = py - ay
                r1z = pz - az
                r2x = px - bx
                r2y = py - by
                r2z = pz - bz
                n1 = sqrt(r1x * r1x + r1y * r1y + r1z * r1z + 1e-300)
                n2 = sqrt(r2x * r2x + r2y * r2y + r2z * r2z + 1e-300)
                r0x = bx - ax
                r0y = by - ay
                r0z = bz - az
                cx = r1y * r2z - r1z * r2y
                cy = r1z * r2x - r1x * r2z
                cz = r1x * r2y - r1y * r2x
                cr2 = cx * cx + cy * cy + cz * cz
                r0sq = r0x * r0x + r0y * r0y + r0z * r0z
                k = _seg_k(r1x, r1y, r1z, r2x, r2y, r2z, r0x, r0y, r0z,
                           n1, n2, cr2, r0sq, rc2_src[j])
                vx = cx * k
                vy = cy * k
                vz = cz * k
                dx = d[j, 0]
                dy = d[j, 1]
                dz = d[j, 2]
                if dy == 0.0:
                    qx = -dz * r2y
                    qy = dz * r2x - dx * r2z
                    qz = dx * r2y
                    qdr = dx * r2x + dz * r2z
                    ex = -dz * r1y
                    ey = dz * r1x - dx * r1z
                    ez = dx * r1y
                    edr = dx * r1x + dz * r1z
                else:
                    qx = dy * r2z - dz * r2y
                    qy = dz * r2x - dx * r2z
                    qz = dx * r2y - dy * r2x
                    qdr = dx * r2x + dy * r2y + dz * r2z
                    ex = dy * r1z - dz * r1y
                    ey = dz * r1x - dx * r1z
                    ez = dx * r1y - dy * r1x
                    edr = dx * r1x + dy * r1y + dz * r1z
                qcr2 = qx * qx + qy * qy + qz * qz
                qk = _ray_k(r2x, r2y, r2z, n2, qdr, qcr2, rc2_src[j])
                vx = vx + qx * qk
                vy = vy + qy * qk
                vz = vz + qz * qk
                ecr2 = ex * ex + ey * ey + ez * ez
                ek = _ray_k(r1x, r1y, r1z, n1, edr, ecr2, rc2_src[j])
                vx = vx - ex * ek
                vy = vy - ey * ek
                vz = vz - ez * ek
                g = g_src[j]
                sx = sx + vx * g
                sy = sy + vy * g
                sz = sz + vz * g
            out_v[i, 0] = sx
            out_v[i, 1] = sy
            out_v[i, 2] = sz
        return out
    out = np.empty((m, 3), dtype=np.float64)
    out_v = out
    for i in prange(m, num_threads=num_threads, nogil=True, schedule="static"):
        # Assign the per-pair results here so each thread gets a private copy.
        vx = 0.0
        vy = 0.0
        vz = 0.0
        sx = 0.0
        sy = 0.0
        sz = 0.0
        px = P[i, 0]
        py = P[i, 1]
        pz = P[i, 2]
        for j in range(n):
            rc2 = rc2_src[j]
            if use_tg and tg_group[i] != src_group[j]:
                rc2 = rc2 + tg_rc2[i]
            ax = a[j, 0]
            ay = a[j, 1]
            az = a[j, 2]
            bx = b[j, 0]
            by = b[j, 1]
            bz = b[j, 2]
            tax = a_te[j, 0]
            tay = a_te[j, 1]
            taz = a_te[j, 2]
            tbx = b_te[j, 0]
            tby = b_te[j, 1]
            tbz = b_te[j, 2]
            straight_left = tax == ax and tay == ay and taz == az
            straight_right = tbx == bx and tby == by and tbz == bz
            r1x = 0.0
            r1y = 0.0
            r1z = 0.0
            n1 = 0.0
            r2x = 0.0
            r2y = 0.0
            r2z = 0.0
            n2 = 0.0
            if parts == 2:
                vx = 0.0
                vy = 0.0
                vz = 0.0
                if straight_left or straight_right:
                    r1x = px - ax
                    r1y = py - ay
                    r1z = pz - az
                    r2x = px - bx
                    r2y = py - by
                    r2z = pz - bz
                    n1 = sqrt(r1x * r1x + r1y * r1y + r1z * r1z + 1e-300)
                    n2 = sqrt(r2x * r2x + r2y * r2y + r2z * r2z + 1e-300)
            else:
                r1x = px - ax
                r1y = py - ay
                r1z = pz - az
                r2x = px - bx
                r2y = py - by
                r2z = pz - bz
                n1 = sqrt(r1x * r1x + r1y * r1y + r1z * r1z + 1e-300)
                n2 = sqrt(r2x * r2x + r2y * r2y + r2z * r2z + 1e-300)
                r0x = bx - ax
                r0y = by - ay
                r0z = bz - az
                if r0x == 0.0 and r0y == 0.0 and r0z == 0.0:
                    vx = 0.0
                    vy = 0.0
                    vz = 0.0
                else:
                    cx = r1y * r2z - r1z * r2y
                    cy = r1z * r2x - r1x * r2z
                    cz = r1x * r2y - r1y * r2x
                    cr2 = cx * cx + cy * cy + cz * cz
                    r0sq = r0x * r0x + r0y * r0y + r0z * r0z
                    k = _seg_k(r1x, r1y, r1z, r2x, r2y, r2z, r0x, r0y, r0z,
                               n1, n2, cr2, r0sq, rc2)
                    vx = cx * k
                    vy = cy * k
                    vz = cz * k
                if not straight_left:
                    kax = tax
                    kay = tay
                    kaz = taz
                    if kink:
                        kax = ak[j, 0]
                        kay = ak[j, 1]
                        kaz = ak[j, 2]
                    q1x = px - kax
                    q1y = py - kay
                    q1z = pz - kaz
                    nq = sqrt(q1x * q1x + q1y * q1y + q1z * q1z + 1e-300)
                    r0x = ax - kax
                    r0y = ay - kay
                    r0z = az - kaz
                    cx = q1y * r1z - q1z * r1y
                    cy = q1z * r1x - q1x * r1z
                    cz = q1x * r1y - q1y * r1x
                    cr2 = cx * cx + cy * cy + cz * cz
                    r0sq = r0x * r0x + r0y * r0y + r0z * r0z
                    k = _seg_k(q1x, q1y, q1z, r1x, r1y, r1z, r0x, r0y, r0z,
                               nq, n1, cr2, r0sq, rc2)
                    vx = vx + cx * k
                    vy = vy + cy * k
                    vz = vz + cz * k
                    if kink and (kax != tax or kay != tay or kaz != taz):
                        # Aft segment a_te -> node of a leg with a node.
                        t1x = px - tax
                        t1y = py - tay
                        t1z = pz - taz
                        nt = sqrt(t1x * t1x + t1y * t1y + t1z * t1z + 1e-300)
                        r0x = kax - tax
                        r0y = kay - tay
                        r0z = kaz - taz
                        cx = t1y * q1z - t1z * q1y
                        cy = t1z * q1x - t1x * q1z
                        cz = t1x * q1y - t1y * q1x
                        cr2 = cx * cx + cy * cy + cz * cz
                        r0sq = r0x * r0x + r0y * r0y + r0z * r0z
                        k = _seg_k(t1x, t1y, t1z, q1x, q1y, q1z, r0x, r0y, r0z,
                                   nt, nq, cr2, r0sq, rc2)
                        vx = vx + cx * k
                        vy = vy + cy * k
                        vz = vz + cz * k
                if not straight_right:
                    kbx = tbx
                    kby = tby
                    kbz = tbz
                    if kink:
                        kbx = bk[j, 0]
                        kby = bk[j, 1]
                        kbz = bk[j, 2]
                    w2x = px - kbx
                    w2y = py - kby
                    w2z = pz - kbz
                    nw = sqrt(w2x * w2x + w2y * w2y + w2z * w2z + 1e-300)
                    r0x = kbx - bx
                    r0y = kby - by
                    r0z = kbz - bz
                    cx = r2y * w2z - r2z * w2y
                    cy = r2z * w2x - r2x * w2z
                    cz = r2x * w2y - r2y * w2x
                    cr2 = cx * cx + cy * cy + cz * cz
                    r0sq = r0x * r0x + r0y * r0y + r0z * r0z
                    k = _seg_k(r2x, r2y, r2z, w2x, w2y, w2z, r0x, r0y, r0z,
                               n2, nw, cr2, r0sq, rc2)
                    vx = vx + cx * k
                    vy = vy + cy * k
                    vz = vz + cz * k
                    if kink and (kbx != tbx or kby != tby or kbz != tbz):
                        # Aft segment node -> b_te of a leg with a node.
                        t2x = px - tbx
                        t2y = py - tby
                        t2z = pz - tbz
                        nt = sqrt(t2x * t2x + t2y * t2y + t2z * t2z + 1e-300)
                        r0x = tbx - kbx
                        r0y = tby - kby
                        r0z = tbz - kbz
                        cx = w2y * t2z - w2z * t2y
                        cy = w2z * t2x - w2x * t2z
                        cz = w2x * t2y - w2y * t2x
                        cr2 = cx * cx + cy * cy + cz * cz
                        r0sq = r0x * r0x + r0y * r0y + r0z * r0z
                        k = _seg_k(w2x, w2y, w2z, t2x, t2y, t2z, r0x, r0y, r0z,
                                   nw, nt, cr2, r0sq, rc2)
                        vx = vx + cx * k
                        vy = vy + cy * k
                        vz = vz + cz * k
            if parts != 1:
                if straight_right:
                    rbrx = r2x
                    rbry = r2y
                    rbrz = r2z
                    rbrn = n2
                else:
                    rbrx = px - tbx
                    rbry = py - tby
                    rbrz = pz - tbz
                    rbrn = sqrt(rbrx * rbrx + rbry * rbry + rbrz * rbrz + 1e-300)
                if straight_left:
                    larx = r1x
                    lary = r1y
                    larz = r1z
                    larn = n1
                else:
                    larx = px - tax
                    lary = py - tay
                    larz = pz - taz
                    larn = sqrt(larx * larx + lary * lary + larz * larz + 1e-300)
                dx = d[j, 0]
                dy = d[j, 1]
                dz = d[j, 2]
                if dy == 0.0:
                    qx = -dz * rbry
                    qy = dz * rbrx - dx * rbrz
                    qz = dx * rbry
                    qdr = dx * rbrx + dz * rbrz
                    ex = -dz * lary
                    ey = dz * larx - dx * larz
                    ez = dx * lary
                    edr = dx * larx + dz * larz
                else:
                    qx = dy * rbrz - dz * rbry
                    qy = dz * rbrx - dx * rbrz
                    qz = dx * rbry - dy * rbrx
                    qdr = dx * rbrx + dy * rbry + dz * rbrz
                    ex = dy * larz - dz * lary
                    ey = dz * larx - dx * larz
                    ez = dx * lary - dy * larx
                    edr = dx * larx + dy * lary + dz * larz
                qcr2 = qx * qx + qy * qy + qz * qz
                qk = _ray_k(rbrx, rbry, rbrz, rbrn, qdr, qcr2, rc2)
                ux = qx * qk
                uy = qy * qk
                uz = qz * qk
                ecr2 = ex * ex + ey * ey + ez * ez
                ek = _ray_k(larx, lary, larz, larn, edr, ecr2, rc2)
                ex = ex * ek
                ey = ey * ek
                ez = ez * ek
                if parts == 2:
                    vx = ux - ex
                    vy = uy - ey
                    vz = uz - ez
                else:
                    vx = vx + ux
                    vy = vy + uy
                    vz = vz + uz
                    vx = vx - ex
                    vy = vy - ey
                    vz = vz - ez
            g = g_src[j]
            sx = sx + vx * g
            sy = sy + vy * g
            sz = sz + vz * g
        out_v[i, 0] = sx
        out_v[i, 1] = sy
        out_v[i, 2] = sz
    return out


def normal_influence_kernel(const double[:, ::1] P, const double[:, ::1] normals,
                            const double[:, ::1] a, const double[:, ::1] b,
                            const double[:, ::1] a_te, const double[:, ::1] b_te, const double[:, ::1] d,
                            const double[::1] rc2_src, const cnp.int64_t[::1] src_group, const double[::1] sign,
                            const cnp.int64_t[::1] column, int n_unknowns,
                            const cnp.int64_t[::1] tg_group, const double[::1] tg_rc2, bint use_tg,
                            int parts, int num_threads, a_kink=None, b_kink=None):
    """Return the normal-velocity influence matrix.

    ``A[i, j]`` is the velocity normal to ``normals[i]`` at ``P[i]`` for unit
    circulation of unknown ``j``. The sources that map to the same unknown
    (mirror and ground images) are added in source order. Each thread writes
    only its own row ``i``, so the addition over ``column`` has no race.

    Parameters
    ----------
    P : (m, 3)
        Evaluation points [m].
    normals : (m, 3)
        Unit normals at the evaluation points.
    a, b : (n, 3)
        Bound-vortex end points [m]. Positive circulation runs from ``a`` to ``b``.
    a_te, b_te : (n, 3)
        Trailing-edge points where the legs turn into the wake [m].
    d : (n, 3)
        Unit wake direction.
    rc2_src : (n,)
        Squared core radius of each source [m^2].
    src_group : (n,)
        Surface index of each source.
    sign : (n,)
        Multiplier on the circulation (+1 real, -1 ground image).
    column : (n,)
        Index of the unknown that gives each source its circulation.
    n_unknowns : int
        Number of unknowns (columns of the result).
    tg_group : (m,)
        Surface index of each evaluation point.
    tg_rc2 : (m,)
        Squared core radius added for sources of other surfaces [m^2].
    use_tg : bool
        Whether the target core applies.
    parts : int
        0: all legs, 1: bound vortex and chordwise legs, 2: wake legs.
    num_threads : int
        Number of OpenMP threads.
    a_kink, b_kink : (n, 3) or None
        Nodes of the chordwise legs [m] (see ``a_kink`` in
        :class:`ventorum.aero.vortex.HorseshoeSet`), or None for straight legs.

    Returns
    -------
    (m, n_unknowns) normal velocity per unit circulation [1/m].
    """
    cdef Py_ssize_t m = P.shape[0]
    cdef Py_ssize_t n = a.shape[0]
    cdef cnp.ndarray[double, ndim=2] out
    cdef double[:, ::1] out_v
    cdef Py_ssize_t i, j
    cdef double rc2, vx, vy, vz, nx, ny, nz
    cdef double px, py, pz, ax, ay, az, bx, by, bz, tax, tay, taz, tbx, tby, tbz
    cdef bint straight_left, straight_right
    cdef double r1x, r1y, r1z, r2x, r2y, r2z, r0x, r0y, r0z, n1, n2
    cdef double cx, cy, cz, cr2, r0sq, k
    cdef double q1x, q1y, q1z, nq, w2x, w2y, w2z, nw
    cdef double kax, kay, kaz, kbx, kby, kbz, t1x, t1y, t1z, t2x, t2y, t2z, nt
    # Leg nodes (see ventorum.aero.vortex.HorseshoeSet); without them the
    # node arrays are a_te and b_te and the legs are computed as before.
    cdef bint kink = a_kink is not None
    cdef const double[:, ::1] ak = a_te if a_kink is None else a_kink
    cdef const double[:, ::1] bk = b_te if b_kink is None else b_kink
    cdef double dx, dy, dz
    cdef double rbrx, rbry, rbrz, rbrn, larx, lary, larz, larn
    cdef double qx, qy, qz, qcr2, qdr, qk
    cdef double ex, ey, ez, ecr2, edr, ek
    cdef double ux, uy, uz
    if parts == 0 and not use_tg and not kink and _is_direct(column, sign, n_unknowns, n) \
            and _is_straight(a, b, a_te, b_te, n):
        out = np.empty((m, n_unknowns), dtype=np.float64)
        out_v = out
        for i in prange(m, num_threads=num_threads, nogil=True, schedule="static"):
            # Assign the per-pair results here so each thread gets a private copy.
            vx = 0.0
            vy = 0.0
            vz = 0.0
            nx = normals[i, 0]
            ny = normals[i, 1]
            nz = normals[i, 2]
            px = P[i, 0]
            py = P[i, 1]
            pz = P[i, 2]
            for j in range(n):
                ax = a[j, 0]
                ay = a[j, 1]
                az = a[j, 2]
                bx = b[j, 0]
                by = b[j, 1]
                bz = b[j, 2]
                r1x = px - ax
                r1y = py - ay
                r1z = pz - az
                r2x = px - bx
                r2y = py - by
                r2z = pz - bz
                r0x = bx - ax
                r0y = by - ay
                r0z = bz - az
                n1 = sqrt(r1x * r1x + r1y * r1y + r1z * r1z + 1e-300)
                n2 = sqrt(r2x * r2x + r2y * r2y + r2z * r2z + 1e-300)
                # Bound segment a -> b (same formula as the numpy kernel).
                cx = r1y * r2z - r1z * r2y
                cy = r1z * r2x - r1x * r2z
                cz = r1x * r2y - r1y * r2x
                cr2 = cx * cx + cy * cy + cz * cz
                r0sq = r0x * r0x + r0y * r0y + r0z * r0z
                k = _seg_k(r1x, r1y, r1z, r2x, r2y, r2z, r0x, r0y, r0z,
                           n1, n2, cr2, r0sq, rc2_src[j])
                vx = cx * k
                vy = cy * k
                vz = cz * k
                # Wake rays from b (+) and a (-), shared norms.
                dx = d[j, 0]
                dy = d[j, 1]
                dz = d[j, 2]
                if dy == 0.0:
                    qx = -dz * r2y
                    qy = dz * r2x - dx * r2z
                    qz = dx * r2y
                    qdr = dx * r2x + dz * r2z
                    ex = -dz * r1y
                    ey = dz * r1x - dx * r1z
                    ez = dx * r1y
                    edr = dx * r1x + dz * r1z
                else:
                    qx = dy * r2z - dz * r2y
                    qy = dz * r2x - dx * r2z
                    qz = dx * r2y - dy * r2x
                    qdr = dx * r2x + dy * r2y + dz * r2z
                    ex = dy * r1z - dz * r1y
                    ey = dz * r1x - dx * r1z
                    ez = dx * r1y - dy * r1x
                    edr = dx * r1x + dy * r1y + dz * r1z
                qcr2 = qx * qx + qy * qy + qz * qz
                qk = _ray_k(r2x, r2y, r2z, n2, qdr, qcr2, rc2_src[j])
                vx = vx + qx * qk
                vy = vy + qy * qk
                vz = vz + qz * qk
                ecr2 = ex * ex + ey * ey + ez * ez
                ek = _ray_k(r1x, r1y, r1z, n1, edr, ecr2, rc2_src[j])
                vx = vx - ex * ek
                vy = vy - ey * ek
                vz = vz - ez * ek
                out_v[i, j] = vx * nx + vy * ny + vz * nz
        return out
    out = np.zeros((m, n_unknowns), dtype=np.float64)
    out_v = out
    for i in prange(m, num_threads=num_threads, nogil=True, schedule="static"):
        # Assign the per-pair results here so each thread gets a private copy.
        vx = 0.0
        vy = 0.0
        vz = 0.0
        nx = normals[i, 0]
        ny = normals[i, 1]
        nz = normals[i, 2]
        px = P[i, 0]
        py = P[i, 1]
        pz = P[i, 2]
        for j in range(n):
            rc2 = rc2_src[j]
            if use_tg and tg_group[i] != src_group[j]:
                rc2 = rc2 + tg_rc2[i]
            ax = a[j, 0]
            ay = a[j, 1]
            az = a[j, 2]
            bx = b[j, 0]
            by = b[j, 1]
            bz = b[j, 2]
            tax = a_te[j, 0]
            tay = a_te[j, 1]
            taz = a_te[j, 2]
            tbx = b_te[j, 0]
            tby = b_te[j, 1]
            tbz = b_te[j, 2]
            straight_left = tax == ax and tay == ay and taz == az
            straight_right = tbx == bx and tby == by and tbz == bz
            r1x = 0.0
            r1y = 0.0
            r1z = 0.0
            n1 = 0.0
            r2x = 0.0
            r2y = 0.0
            r2z = 0.0
            n2 = 0.0
            if parts == 2:
                vx = 0.0
                vy = 0.0
                vz = 0.0
                if straight_left or straight_right:
                    r1x = px - ax
                    r1y = py - ay
                    r1z = pz - az
                    r2x = px - bx
                    r2y = py - by
                    r2z = pz - bz
                    n1 = sqrt(r1x * r1x + r1y * r1y + r1z * r1z + 1e-300)
                    n2 = sqrt(r2x * r2x + r2y * r2y + r2z * r2z + 1e-300)
            else:
                r1x = px - ax
                r1y = py - ay
                r1z = pz - az
                r2x = px - bx
                r2y = py - by
                r2z = pz - bz
                n1 = sqrt(r1x * r1x + r1y * r1y + r1z * r1z + 1e-300)
                n2 = sqrt(r2x * r2x + r2y * r2y + r2z * r2z + 1e-300)
                r0x = bx - ax
                r0y = by - ay
                r0z = bz - az
                if r0x == 0.0 and r0y == 0.0 and r0z == 0.0:
                    vx = 0.0
                    vy = 0.0
                    vz = 0.0
                else:
                    cx = r1y * r2z - r1z * r2y
                    cy = r1z * r2x - r1x * r2z
                    cz = r1x * r2y - r1y * r2x
                    cr2 = cx * cx + cy * cy + cz * cz
                    r0sq = r0x * r0x + r0y * r0y + r0z * r0z
                    k = _seg_k(r1x, r1y, r1z, r2x, r2y, r2z, r0x, r0y, r0z,
                               n1, n2, cr2, r0sq, rc2)
                    vx = cx * k
                    vy = cy * k
                    vz = cz * k
                if not straight_left:
                    kax = tax
                    kay = tay
                    kaz = taz
                    if kink:
                        kax = ak[j, 0]
                        kay = ak[j, 1]
                        kaz = ak[j, 2]
                    q1x = px - kax
                    q1y = py - kay
                    q1z = pz - kaz
                    nq = sqrt(q1x * q1x + q1y * q1y + q1z * q1z + 1e-300)
                    r0x = ax - kax
                    r0y = ay - kay
                    r0z = az - kaz
                    cx = q1y * r1z - q1z * r1y
                    cy = q1z * r1x - q1x * r1z
                    cz = q1x * r1y - q1y * r1x
                    cr2 = cx * cx + cy * cy + cz * cz
                    r0sq = r0x * r0x + r0y * r0y + r0z * r0z
                    k = _seg_k(q1x, q1y, q1z, r1x, r1y, r1z, r0x, r0y, r0z,
                               nq, n1, cr2, r0sq, rc2)
                    vx = vx + cx * k
                    vy = vy + cy * k
                    vz = vz + cz * k
                    if kink and (kax != tax or kay != tay or kaz != taz):
                        # Aft segment a_te -> node of a leg with a node.
                        t1x = px - tax
                        t1y = py - tay
                        t1z = pz - taz
                        nt = sqrt(t1x * t1x + t1y * t1y + t1z * t1z + 1e-300)
                        r0x = kax - tax
                        r0y = kay - tay
                        r0z = kaz - taz
                        cx = t1y * q1z - t1z * q1y
                        cy = t1z * q1x - t1x * q1z
                        cz = t1x * q1y - t1y * q1x
                        cr2 = cx * cx + cy * cy + cz * cz
                        r0sq = r0x * r0x + r0y * r0y + r0z * r0z
                        k = _seg_k(t1x, t1y, t1z, q1x, q1y, q1z, r0x, r0y, r0z,
                                   nt, nq, cr2, r0sq, rc2)
                        vx = vx + cx * k
                        vy = vy + cy * k
                        vz = vz + cz * k
                if not straight_right:
                    kbx = tbx
                    kby = tby
                    kbz = tbz
                    if kink:
                        kbx = bk[j, 0]
                        kby = bk[j, 1]
                        kbz = bk[j, 2]
                    w2x = px - kbx
                    w2y = py - kby
                    w2z = pz - kbz
                    nw = sqrt(w2x * w2x + w2y * w2y + w2z * w2z + 1e-300)
                    r0x = kbx - bx
                    r0y = kby - by
                    r0z = kbz - bz
                    cx = r2y * w2z - r2z * w2y
                    cy = r2z * w2x - r2x * w2z
                    cz = r2x * w2y - r2y * w2x
                    cr2 = cx * cx + cy * cy + cz * cz
                    r0sq = r0x * r0x + r0y * r0y + r0z * r0z
                    k = _seg_k(r2x, r2y, r2z, w2x, w2y, w2z, r0x, r0y, r0z,
                               n2, nw, cr2, r0sq, rc2)
                    vx = vx + cx * k
                    vy = vy + cy * k
                    vz = vz + cz * k
                    if kink and (kbx != tbx or kby != tby or kbz != tbz):
                        # Aft segment node -> b_te of a leg with a node.
                        t2x = px - tbx
                        t2y = py - tby
                        t2z = pz - tbz
                        nt = sqrt(t2x * t2x + t2y * t2y + t2z * t2z + 1e-300)
                        r0x = tbx - kbx
                        r0y = tby - kby
                        r0z = tbz - kbz
                        cx = w2y * t2z - w2z * t2y
                        cy = w2z * t2x - w2x * t2z
                        cz = w2x * t2y - w2y * t2x
                        cr2 = cx * cx + cy * cy + cz * cz
                        r0sq = r0x * r0x + r0y * r0y + r0z * r0z
                        k = _seg_k(w2x, w2y, w2z, t2x, t2y, t2z, r0x, r0y, r0z,
                                   nw, nt, cr2, r0sq, rc2)
                        vx = vx + cx * k
                        vy = vy + cy * k
                        vz = vz + cz * k
            if parts != 1:
                if straight_right:
                    rbrx = r2x
                    rbry = r2y
                    rbrz = r2z
                    rbrn = n2
                else:
                    rbrx = px - tbx
                    rbry = py - tby
                    rbrz = pz - tbz
                    rbrn = sqrt(rbrx * rbrx + rbry * rbry + rbrz * rbrz + 1e-300)
                if straight_left:
                    larx = r1x
                    lary = r1y
                    larz = r1z
                    larn = n1
                else:
                    larx = px - tax
                    lary = py - tay
                    larz = pz - taz
                    larn = sqrt(larx * larx + lary * lary + larz * larz + 1e-300)
                dx = d[j, 0]
                dy = d[j, 1]
                dz = d[j, 2]
                if dy == 0.0:
                    qx = -dz * rbry
                    qy = dz * rbrx - dx * rbrz
                    qz = dx * rbry
                    qdr = dx * rbrx + dz * rbrz
                    ex = -dz * lary
                    ey = dz * larx - dx * larz
                    ez = dx * lary
                    edr = dx * larx + dz * larz
                else:
                    qx = dy * rbrz - dz * rbry
                    qy = dz * rbrx - dx * rbrz
                    qz = dx * rbry - dy * rbrx
                    qdr = dx * rbrx + dy * rbry + dz * rbrz
                    ex = dy * larz - dz * lary
                    ey = dz * larx - dx * larz
                    ez = dx * lary - dy * larx
                    edr = dx * larx + dy * lary + dz * larz
                qcr2 = qx * qx + qy * qy + qz * qz
                qk = _ray_k(rbrx, rbry, rbrz, rbrn, qdr, qcr2, rc2)
                ux = qx * qk
                uy = qy * qk
                uz = qz * qk
                ecr2 = ex * ex + ey * ey + ez * ez
                ek = _ray_k(larx, lary, larz, larn, edr, ecr2, rc2)
                ex = ex * ek
                ey = ey * ek
                ez = ez * ek
                if parts == 2:
                    vx = ux - ex
                    vy = uy - ey
                    vz = uz - ez
                else:
                    vx = vx + ux
                    vy = vy + uy
                    vz = vz + uz
                    vx = vx - ex
                    vy = vy - ey
                    vz = vz - ez
            out_v[i, column[j]] += (vx * nx + vy * ny + vz * nz) * sign[j]
    return out


def velocity_tensor_kernel(const double[:, ::1] P, const double[:, ::1] a, const double[:, ::1] b,
                           const double[:, ::1] a_te, const double[:, ::1] b_te, const double[:, ::1] d,
                           const double[::1] rc2_src, const cnp.int64_t[::1] src_group, const double[::1] sign,
                           const cnp.int64_t[::1] tg_group, const double[::1] tg_rc2, bint use_tg,
                           int parts, int num_threads, a_kink=None, b_kink=None):
    """Return the velocity per source horseshoe, with the source sign applied.

    Parameters
    ----------
    P : (m, 3)
        Evaluation points [m].
    a, b : (n, 3)
        Bound-vortex end points [m]. Positive circulation runs from ``a`` to ``b``.
    a_te, b_te : (n, 3)
        Trailing-edge points where the legs turn into the wake [m].
    d : (n, 3)
        Unit wake direction.
    rc2_src : (n,)
        Squared core radius of each source [m^2].
    src_group : (n,)
        Surface index of each source.
    sign : (n,)
        Multiplier on the circulation (+1 real, -1 ground image).
    tg_group : (m,)
        Surface index of each evaluation point.
    tg_rc2 : (m,)
        Squared core radius added for sources of other surfaces [m^2].
    use_tg : bool
        Whether the target core applies.
    parts : int
        0: all legs, 1: bound vortex and chordwise legs, 2: wake legs.
    num_threads : int
        Number of OpenMP threads.
    a_kink, b_kink : (n, 3) or None
        Nodes of the chordwise legs [m] (see ``a_kink`` in
        :class:`ventorum.aero.vortex.HorseshoeSet`), or None for straight legs.

    Returns
    -------
    (m, n, 3) velocity per unit circulation [1/m], sign applied.
    """
    cdef Py_ssize_t m = P.shape[0]
    cdef Py_ssize_t n = a.shape[0]
    cdef cnp.ndarray[double, ndim=3] out = np.empty((m, n, 3), dtype=np.float64)
    cdef double[:, :, ::1] out_v = out
    cdef Py_ssize_t i, j
    cdef double rc2, s, vx, vy, vz
    cdef double px, py, pz, ax, ay, az, bx, by, bz, tax, tay, taz, tbx, tby, tbz
    cdef bint straight_left, straight_right
    cdef double r1x, r1y, r1z, r2x, r2y, r2z, r0x, r0y, r0z, n1, n2
    cdef double cx, cy, cz, cr2, r0sq, k
    cdef double q1x, q1y, q1z, nq, w2x, w2y, w2z, nw
    cdef double kax, kay, kaz, kbx, kby, kbz, t1x, t1y, t1z, t2x, t2y, t2z, nt
    # Leg nodes (see ventorum.aero.vortex.HorseshoeSet); without them the
    # node arrays are a_te and b_te and the legs are computed as before.
    cdef bint kink = a_kink is not None
    cdef const double[:, ::1] ak = a_te if a_kink is None else a_kink
    cdef const double[:, ::1] bk = b_te if b_kink is None else b_kink
    cdef double dx, dy, dz
    cdef double rbrx, rbry, rbrz, rbrn, larx, lary, larz, larn
    cdef double qx, qy, qz, qcr2, qdr, qk
    cdef double ex, ey, ez, ecr2, edr, ek
    cdef double ux, uy, uz
    for i in prange(m, num_threads=num_threads, nogil=True, schedule="static"):
        # Assign the per-pair results here so each thread gets a private copy.
        vx = 0.0
        vy = 0.0
        vz = 0.0
        px = P[i, 0]
        py = P[i, 1]
        pz = P[i, 2]
        for j in range(n):
            rc2 = rc2_src[j]
            if use_tg and tg_group[i] != src_group[j]:
                rc2 = rc2 + tg_rc2[i]
            ax = a[j, 0]
            ay = a[j, 1]
            az = a[j, 2]
            bx = b[j, 0]
            by = b[j, 1]
            bz = b[j, 2]
            tax = a_te[j, 0]
            tay = a_te[j, 1]
            taz = a_te[j, 2]
            tbx = b_te[j, 0]
            tby = b_te[j, 1]
            tbz = b_te[j, 2]
            straight_left = tax == ax and tay == ay and taz == az
            straight_right = tbx == bx and tby == by and tbz == bz
            r1x = 0.0
            r1y = 0.0
            r1z = 0.0
            n1 = 0.0
            r2x = 0.0
            r2y = 0.0
            r2z = 0.0
            n2 = 0.0
            if parts == 2:
                vx = 0.0
                vy = 0.0
                vz = 0.0
                if straight_left or straight_right:
                    r1x = px - ax
                    r1y = py - ay
                    r1z = pz - az
                    r2x = px - bx
                    r2y = py - by
                    r2z = pz - bz
                    n1 = sqrt(r1x * r1x + r1y * r1y + r1z * r1z + 1e-300)
                    n2 = sqrt(r2x * r2x + r2y * r2y + r2z * r2z + 1e-300)
            else:
                r1x = px - ax
                r1y = py - ay
                r1z = pz - az
                r2x = px - bx
                r2y = py - by
                r2z = pz - bz
                n1 = sqrt(r1x * r1x + r1y * r1y + r1z * r1z + 1e-300)
                n2 = sqrt(r2x * r2x + r2y * r2y + r2z * r2z + 1e-300)
                r0x = bx - ax
                r0y = by - ay
                r0z = bz - az
                if r0x == 0.0 and r0y == 0.0 and r0z == 0.0:
                    vx = 0.0
                    vy = 0.0
                    vz = 0.0
                else:
                    cx = r1y * r2z - r1z * r2y
                    cy = r1z * r2x - r1x * r2z
                    cz = r1x * r2y - r1y * r2x
                    cr2 = cx * cx + cy * cy + cz * cz
                    r0sq = r0x * r0x + r0y * r0y + r0z * r0z
                    k = _seg_k(r1x, r1y, r1z, r2x, r2y, r2z, r0x, r0y, r0z,
                               n1, n2, cr2, r0sq, rc2)
                    vx = cx * k
                    vy = cy * k
                    vz = cz * k
                if not straight_left:
                    kax = tax
                    kay = tay
                    kaz = taz
                    if kink:
                        kax = ak[j, 0]
                        kay = ak[j, 1]
                        kaz = ak[j, 2]
                    q1x = px - kax
                    q1y = py - kay
                    q1z = pz - kaz
                    nq = sqrt(q1x * q1x + q1y * q1y + q1z * q1z + 1e-300)
                    r0x = ax - kax
                    r0y = ay - kay
                    r0z = az - kaz
                    cx = q1y * r1z - q1z * r1y
                    cy = q1z * r1x - q1x * r1z
                    cz = q1x * r1y - q1y * r1x
                    cr2 = cx * cx + cy * cy + cz * cz
                    r0sq = r0x * r0x + r0y * r0y + r0z * r0z
                    k = _seg_k(q1x, q1y, q1z, r1x, r1y, r1z, r0x, r0y, r0z,
                               nq, n1, cr2, r0sq, rc2)
                    vx = vx + cx * k
                    vy = vy + cy * k
                    vz = vz + cz * k
                    if kink and (kax != tax or kay != tay or kaz != taz):
                        # Aft segment a_te -> node of a leg with a node.
                        t1x = px - tax
                        t1y = py - tay
                        t1z = pz - taz
                        nt = sqrt(t1x * t1x + t1y * t1y + t1z * t1z + 1e-300)
                        r0x = kax - tax
                        r0y = kay - tay
                        r0z = kaz - taz
                        cx = t1y * q1z - t1z * q1y
                        cy = t1z * q1x - t1x * q1z
                        cz = t1x * q1y - t1y * q1x
                        cr2 = cx * cx + cy * cy + cz * cz
                        r0sq = r0x * r0x + r0y * r0y + r0z * r0z
                        k = _seg_k(t1x, t1y, t1z, q1x, q1y, q1z, r0x, r0y, r0z,
                                   nt, nq, cr2, r0sq, rc2)
                        vx = vx + cx * k
                        vy = vy + cy * k
                        vz = vz + cz * k
                if not straight_right:
                    kbx = tbx
                    kby = tby
                    kbz = tbz
                    if kink:
                        kbx = bk[j, 0]
                        kby = bk[j, 1]
                        kbz = bk[j, 2]
                    w2x = px - kbx
                    w2y = py - kby
                    w2z = pz - kbz
                    nw = sqrt(w2x * w2x + w2y * w2y + w2z * w2z + 1e-300)
                    r0x = kbx - bx
                    r0y = kby - by
                    r0z = kbz - bz
                    cx = r2y * w2z - r2z * w2y
                    cy = r2z * w2x - r2x * w2z
                    cz = r2x * w2y - r2y * w2x
                    cr2 = cx * cx + cy * cy + cz * cz
                    r0sq = r0x * r0x + r0y * r0y + r0z * r0z
                    k = _seg_k(r2x, r2y, r2z, w2x, w2y, w2z, r0x, r0y, r0z,
                               n2, nw, cr2, r0sq, rc2)
                    vx = vx + cx * k
                    vy = vy + cy * k
                    vz = vz + cz * k
                    if kink and (kbx != tbx or kby != tby or kbz != tbz):
                        # Aft segment node -> b_te of a leg with a node.
                        t2x = px - tbx
                        t2y = py - tby
                        t2z = pz - tbz
                        nt = sqrt(t2x * t2x + t2y * t2y + t2z * t2z + 1e-300)
                        r0x = tbx - kbx
                        r0y = tby - kby
                        r0z = tbz - kbz
                        cx = w2y * t2z - w2z * t2y
                        cy = w2z * t2x - w2x * t2z
                        cz = w2x * t2y - w2y * t2x
                        cr2 = cx * cx + cy * cy + cz * cz
                        r0sq = r0x * r0x + r0y * r0y + r0z * r0z
                        k = _seg_k(w2x, w2y, w2z, t2x, t2y, t2z, r0x, r0y, r0z,
                                   nw, nt, cr2, r0sq, rc2)
                        vx = vx + cx * k
                        vy = vy + cy * k
                        vz = vz + cz * k
            if parts != 1:
                if straight_right:
                    rbrx = r2x
                    rbry = r2y
                    rbrz = r2z
                    rbrn = n2
                else:
                    rbrx = px - tbx
                    rbry = py - tby
                    rbrz = pz - tbz
                    rbrn = sqrt(rbrx * rbrx + rbry * rbry + rbrz * rbrz + 1e-300)
                if straight_left:
                    larx = r1x
                    lary = r1y
                    larz = r1z
                    larn = n1
                else:
                    larx = px - tax
                    lary = py - tay
                    larz = pz - taz
                    larn = sqrt(larx * larx + lary * lary + larz * larz + 1e-300)
                dx = d[j, 0]
                dy = d[j, 1]
                dz = d[j, 2]
                if dy == 0.0:
                    qx = -dz * rbry
                    qy = dz * rbrx - dx * rbrz
                    qz = dx * rbry
                    qdr = dx * rbrx + dz * rbrz
                    ex = -dz * lary
                    ey = dz * larx - dx * larz
                    ez = dx * lary
                    edr = dx * larx + dz * larz
                else:
                    qx = dy * rbrz - dz * rbry
                    qy = dz * rbrx - dx * rbrz
                    qz = dx * rbry - dy * rbrx
                    qdr = dx * rbrx + dy * rbry + dz * rbrz
                    ex = dy * larz - dz * lary
                    ey = dz * larx - dx * larz
                    ez = dx * lary - dy * larx
                    edr = dx * larx + dy * lary + dz * larz
                qcr2 = qx * qx + qy * qy + qz * qz
                qk = _ray_k(rbrx, rbry, rbrz, rbrn, qdr, qcr2, rc2)
                ux = qx * qk
                uy = qy * qk
                uz = qz * qk
                ecr2 = ex * ex + ey * ey + ez * ez
                ek = _ray_k(larx, lary, larz, larn, edr, ecr2, rc2)
                ex = ex * ek
                ey = ey * ek
                ez = ez * ek
                if parts == 2:
                    vx = ux - ex
                    vy = uy - ey
                    vz = uz - ez
                else:
                    vx = vx + ux
                    vy = vy + uy
                    vz = vz + uz
                    vx = vx - ex
                    vy = vy - ey
                    vz = vz - ez
            s = sign[j]
            out_v[i, j, 0] = vx * s
            out_v[i, j, 1] = vy * s
            out_v[i, j, 2] = vz * s
    return out



def velocity_unknowns_kernel(const double[:, ::1] P, const double[:, ::1] a, const double[:, ::1] b,
                           const double[:, ::1] a_te, const double[:, ::1] b_te, const double[:, ::1] d,
                           const double[::1] rc2_src, const cnp.int64_t[::1] src_group, const double[::1] sign,
                           const cnp.int64_t[::1] column, Py_ssize_t n_unknowns,
                           const cnp.int64_t[::1] tg_group, const double[::1] tg_rc2, bint use_tg,
                           int parts, int num_threads, a_kink=None, b_kink=None):
    """Return the velocity per unknown: the sources of each unknown added in source order (sign applied).

    The same velocities as ``velocity_tensor_kernel``, accumulated into
    ``out[i, column[j]]`` while they are computed, so the (m, n, 3) tensor per
    source is never stored. Each thread owns one row i.

    Parameters
    ----------
    P : (m, 3)
        Evaluation points [m].
    a, b : (n, 3)
        Bound-vortex end points [m]. Positive circulation runs from ``a`` to ``b``.
    a_te, b_te : (n, 3)
        Trailing-edge points where the legs turn into the wake [m].
    d : (n, 3)
        Unit wake direction.
    rc2_src : (n,)
        Squared core radius of each source [m^2].
    src_group : (n,)
        Surface index of each source.
    sign : (n,)
        Multiplier on the circulation (+1 real, -1 ground image).
    column : (n,)
        Unknown of each source.
    n_unknowns : int
        Number of unknowns.
    tg_group : (m,)
        Surface index of each evaluation point.
    tg_rc2 : (m,)
        Squared core radius added for sources of other surfaces [m^2].
    use_tg : bool
        Whether the target core applies.
    parts : int
        0: all legs, 1: bound vortex and chordwise legs, 2: wake legs.
    num_threads : int
        Number of OpenMP threads.
    a_kink, b_kink : (n, 3) or None
        Nodes of the chordwise legs [m] (see ``a_kink`` in
        :class:`ventorum.aero.vortex.HorseshoeSet`), or None for straight legs.

    Returns
    -------
    (m, n_unknowns, 3) velocity per unit circulation of each unknown [1/m].
    """
    cdef Py_ssize_t m = P.shape[0]
    cdef Py_ssize_t n = a.shape[0]
    cdef cnp.ndarray[double, ndim=3] out = np.zeros((m, n_unknowns, 3), dtype=np.float64)
    cdef double[:, :, ::1] out_v = out
    cdef Py_ssize_t i, j
    cdef double rc2, s, vx, vy, vz
    cdef double px, py, pz, ax, ay, az, bx, by, bz, tax, tay, taz, tbx, tby, tbz
    cdef bint straight_left, straight_right
    cdef double r1x, r1y, r1z, r2x, r2y, r2z, r0x, r0y, r0z, n1, n2
    cdef double cx, cy, cz, cr2, r0sq, k
    cdef double q1x, q1y, q1z, nq, w2x, w2y, w2z, nw
    cdef double kax, kay, kaz, kbx, kby, kbz, t1x, t1y, t1z, t2x, t2y, t2z, nt
    # Leg nodes (see ventorum.aero.vortex.HorseshoeSet); without them the
    # node arrays are a_te and b_te and the legs are computed as before.
    cdef bint kink = a_kink is not None
    cdef const double[:, ::1] ak = a_te if a_kink is None else a_kink
    cdef const double[:, ::1] bk = b_te if b_kink is None else b_kink
    cdef double dx, dy, dz
    cdef double rbrx, rbry, rbrz, rbrn, larx, lary, larz, larn
    cdef double qx, qy, qz, qcr2, qdr, qk
    cdef double ex, ey, ez, ecr2, edr, ek
    cdef double ux, uy, uz
    for i in prange(m, num_threads=num_threads, nogil=True, schedule="static"):
        # Assign the per-pair results here so each thread gets a private copy.
        vx = 0.0
        vy = 0.0
        vz = 0.0
        px = P[i, 0]
        py = P[i, 1]
        pz = P[i, 2]
        for j in range(n):
            rc2 = rc2_src[j]
            if use_tg and tg_group[i] != src_group[j]:
                rc2 = rc2 + tg_rc2[i]
            ax = a[j, 0]
            ay = a[j, 1]
            az = a[j, 2]
            bx = b[j, 0]
            by = b[j, 1]
            bz = b[j, 2]
            tax = a_te[j, 0]
            tay = a_te[j, 1]
            taz = a_te[j, 2]
            tbx = b_te[j, 0]
            tby = b_te[j, 1]
            tbz = b_te[j, 2]
            straight_left = tax == ax and tay == ay and taz == az
            straight_right = tbx == bx and tby == by and tbz == bz
            r1x = 0.0
            r1y = 0.0
            r1z = 0.0
            n1 = 0.0
            r2x = 0.0
            r2y = 0.0
            r2z = 0.0
            n2 = 0.0
            if parts == 2:
                vx = 0.0
                vy = 0.0
                vz = 0.0
                if straight_left or straight_right:
                    r1x = px - ax
                    r1y = py - ay
                    r1z = pz - az
                    r2x = px - bx
                    r2y = py - by
                    r2z = pz - bz
                    n1 = sqrt(r1x * r1x + r1y * r1y + r1z * r1z + 1e-300)
                    n2 = sqrt(r2x * r2x + r2y * r2y + r2z * r2z + 1e-300)
            else:
                r1x = px - ax
                r1y = py - ay
                r1z = pz - az
                r2x = px - bx
                r2y = py - by
                r2z = pz - bz
                n1 = sqrt(r1x * r1x + r1y * r1y + r1z * r1z + 1e-300)
                n2 = sqrt(r2x * r2x + r2y * r2y + r2z * r2z + 1e-300)
                r0x = bx - ax
                r0y = by - ay
                r0z = bz - az
                if r0x == 0.0 and r0y == 0.0 and r0z == 0.0:
                    vx = 0.0
                    vy = 0.0
                    vz = 0.0
                else:
                    cx = r1y * r2z - r1z * r2y
                    cy = r1z * r2x - r1x * r2z
                    cz = r1x * r2y - r1y * r2x
                    cr2 = cx * cx + cy * cy + cz * cz
                    r0sq = r0x * r0x + r0y * r0y + r0z * r0z
                    k = _seg_k(r1x, r1y, r1z, r2x, r2y, r2z, r0x, r0y, r0z,
                               n1, n2, cr2, r0sq, rc2)
                    vx = cx * k
                    vy = cy * k
                    vz = cz * k
                if not straight_left:
                    kax = tax
                    kay = tay
                    kaz = taz
                    if kink:
                        kax = ak[j, 0]
                        kay = ak[j, 1]
                        kaz = ak[j, 2]
                    q1x = px - kax
                    q1y = py - kay
                    q1z = pz - kaz
                    nq = sqrt(q1x * q1x + q1y * q1y + q1z * q1z + 1e-300)
                    r0x = ax - kax
                    r0y = ay - kay
                    r0z = az - kaz
                    cx = q1y * r1z - q1z * r1y
                    cy = q1z * r1x - q1x * r1z
                    cz = q1x * r1y - q1y * r1x
                    cr2 = cx * cx + cy * cy + cz * cz
                    r0sq = r0x * r0x + r0y * r0y + r0z * r0z
                    k = _seg_k(q1x, q1y, q1z, r1x, r1y, r1z, r0x, r0y, r0z,
                               nq, n1, cr2, r0sq, rc2)
                    vx = vx + cx * k
                    vy = vy + cy * k
                    vz = vz + cz * k
                    if kink and (kax != tax or kay != tay or kaz != taz):
                        # Aft segment a_te -> node of a leg with a node.
                        t1x = px - tax
                        t1y = py - tay
                        t1z = pz - taz
                        nt = sqrt(t1x * t1x + t1y * t1y + t1z * t1z + 1e-300)
                        r0x = kax - tax
                        r0y = kay - tay
                        r0z = kaz - taz
                        cx = t1y * q1z - t1z * q1y
                        cy = t1z * q1x - t1x * q1z
                        cz = t1x * q1y - t1y * q1x
                        cr2 = cx * cx + cy * cy + cz * cz
                        r0sq = r0x * r0x + r0y * r0y + r0z * r0z
                        k = _seg_k(t1x, t1y, t1z, q1x, q1y, q1z, r0x, r0y, r0z,
                                   nt, nq, cr2, r0sq, rc2)
                        vx = vx + cx * k
                        vy = vy + cy * k
                        vz = vz + cz * k
                if not straight_right:
                    kbx = tbx
                    kby = tby
                    kbz = tbz
                    if kink:
                        kbx = bk[j, 0]
                        kby = bk[j, 1]
                        kbz = bk[j, 2]
                    w2x = px - kbx
                    w2y = py - kby
                    w2z = pz - kbz
                    nw = sqrt(w2x * w2x + w2y * w2y + w2z * w2z + 1e-300)
                    r0x = kbx - bx
                    r0y = kby - by
                    r0z = kbz - bz
                    cx = r2y * w2z - r2z * w2y
                    cy = r2z * w2x - r2x * w2z
                    cz = r2x * w2y - r2y * w2x
                    cr2 = cx * cx + cy * cy + cz * cz
                    r0sq = r0x * r0x + r0y * r0y + r0z * r0z
                    k = _seg_k(r2x, r2y, r2z, w2x, w2y, w2z, r0x, r0y, r0z,
                               n2, nw, cr2, r0sq, rc2)
                    vx = vx + cx * k
                    vy = vy + cy * k
                    vz = vz + cz * k
                    if kink and (kbx != tbx or kby != tby or kbz != tbz):
                        # Aft segment node -> b_te of a leg with a node.
                        t2x = px - tbx
                        t2y = py - tby
                        t2z = pz - tbz
                        nt = sqrt(t2x * t2x + t2y * t2y + t2z * t2z + 1e-300)
                        r0x = tbx - kbx
                        r0y = tby - kby
                        r0z = tbz - kbz
                        cx = w2y * t2z - w2z * t2y
                        cy = w2z * t2x - w2x * t2z
                        cz = w2x * t2y - w2y * t2x
                        cr2 = cx * cx + cy * cy + cz * cz
                        r0sq = r0x * r0x + r0y * r0y + r0z * r0z
                        k = _seg_k(w2x, w2y, w2z, t2x, t2y, t2z, r0x, r0y, r0z,
                                   nw, nt, cr2, r0sq, rc2)
                        vx = vx + cx * k
                        vy = vy + cy * k
                        vz = vz + cz * k
            if parts != 1:
                if straight_right:
                    rbrx = r2x
                    rbry = r2y
                    rbrz = r2z
                    rbrn = n2
                else:
                    rbrx = px - tbx
                    rbry = py - tby
                    rbrz = pz - tbz
                    rbrn = sqrt(rbrx * rbrx + rbry * rbry + rbrz * rbrz + 1e-300)
                if straight_left:
                    larx = r1x
                    lary = r1y
                    larz = r1z
                    larn = n1
                else:
                    larx = px - tax
                    lary = py - tay
                    larz = pz - taz
                    larn = sqrt(larx * larx + lary * lary + larz * larz + 1e-300)
                dx = d[j, 0]
                dy = d[j, 1]
                dz = d[j, 2]
                if dy == 0.0:
                    qx = -dz * rbry
                    qy = dz * rbrx - dx * rbrz
                    qz = dx * rbry
                    qdr = dx * rbrx + dz * rbrz
                    ex = -dz * lary
                    ey = dz * larx - dx * larz
                    ez = dx * lary
                    edr = dx * larx + dz * larz
                else:
                    qx = dy * rbrz - dz * rbry
                    qy = dz * rbrx - dx * rbrz
                    qz = dx * rbry - dy * rbrx
                    qdr = dx * rbrx + dy * rbry + dz * rbrz
                    ex = dy * larz - dz * lary
                    ey = dz * larx - dx * larz
                    ez = dx * lary - dy * larx
                    edr = dx * larx + dy * lary + dz * larz
                qcr2 = qx * qx + qy * qy + qz * qz
                qk = _ray_k(rbrx, rbry, rbrz, rbrn, qdr, qcr2, rc2)
                ux = qx * qk
                uy = qy * qk
                uz = qz * qk
                ecr2 = ex * ex + ey * ey + ez * ez
                ek = _ray_k(larx, lary, larz, larn, edr, ecr2, rc2)
                ex = ex * ek
                ey = ey * ek
                ez = ez * ek
                if parts == 2:
                    vx = ux - ex
                    vy = uy - ey
                    vz = uz - ez
                else:
                    vx = vx + ux
                    vy = vy + uy
                    vz = vz + uz
                    vx = vx - ex
                    vy = vy - ey
                    vz = vz - ez
            s = sign[j]
            out_v[i, column[j], 0] += vx * s
            out_v[i, column[j], 1] += vy * s
            out_v[i, column[j], 2] += vz * s
    return out

def trefftz_normalwash_kernel(const double[:, ::1] q, const double[:, ::1] q_normal,
                              const double[:, ::1] p, const double[::1] gamma,
                              const double[::1] d, const double[::1] rc2,
                              const cnp.int64_t[::1] src_group, const cnp.int64_t[::1] tg_group,
                              const double[::1] tg_rc2, bint use_tg, int num_threads):
    """Return the normal wash at Trefftz-plane points from 2-D point vortices.

    Parameters
    ----------
    q : (m, 3)
        Evaluation points [m], projected onto the plane.
    q_normal : (m, 3)
        Unit normals at the evaluation points, in the plane.
    p : (n, 3)
        Vortex positions [m], projected onto the plane.
    gamma : (n,)
        Circulation about the axis *d* [m^2/s]. The sign follows the
        right-hand rule.
    d : (3,)
        Unit normal of the Trefftz plane (the wake direction).
    rc2 : (n,)
        Squared 2-D core radius [m^2].
    src_group : (n,)
        Core group of each vortex.
    tg_group : (m,)
        Core group of each evaluation point.
    tg_rc2 : (m,)
        Added squared core radius [m^2] between a point and a vortex of a
        different core group.
    use_tg : bool
        Apply the added core.
    num_threads : int
        Number of OpenMP threads.

    Returns
    -------
    (m,) velocity component along ``q_normal`` [m/s].
    """
    cdef Py_ssize_t m = q.shape[0]
    cdef Py_ssize_t n = p.shape[0]
    cdef cnp.ndarray[double, ndim=1] out = np.empty(m, dtype=np.float64)
    cdef double[::1] out_v = out
    cdef Py_ssize_t i, j
    cdef double dx = d[0], dy = d[1], dz = d[2]
    cdef double qx, qy, qz, nx, ny, nz, rx, ry, rz, vx, vy, vz, k, acc, c2
    for i in prange(m, num_threads=num_threads, nogil=True, schedule="static"):
        qx = q[i, 0]
        qy = q[i, 1]
        qz = q[i, 2]
        nx = q_normal[i, 0]
        ny = q_normal[i, 1]
        nz = q_normal[i, 2]
        acc = 0.0
        for j in range(n):
            rx = qx - p[j, 0]
            ry = qy - p[j, 1]
            rz = qz - p[j, 2]
            vx = dy * rz - dz * ry
            vy = dz * rx - dx * rz
            vz = dx * ry - dy * rx
            c2 = rc2[j]
            if use_tg and tg_group[i] != src_group[j]:
                c2 = c2 + tg_rc2[i]
            k = _INV_2PI * gamma[j] / (rx * rx + ry * ry + rz * rz + c2)
            acc = acc + (vx * nx + vy * ny + vz * nz) * k
        out_v[i] = acc
    return out
