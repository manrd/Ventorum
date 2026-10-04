# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Compiled (Numba) versions of the bent-horseshoe kernels of :mod:`ventorum.aero.vortex`.

The functions here compute the same quantities as the numpy kernels, with
the same regularisation and the same floors, in one pass over each
(point, source) pair. They do not build ``(m, n, 3)`` temporary arrays, and
they run the points in parallel (``prange``). The numpy kernels stay the
reference: the tests require both paths to agree to round-off.

Inside one horseshoe the legs share their vectors and norms, and
zero-length chordwise legs add nothing. Straight horseshoes with unit signs
take a short path with no fold of columns.

The kernels need flat arrays. :func:`pack_sources` and
:func:`pack_targets` prepare them from a :class:`~ventorum.aero.vortex.HorseshoeSet`
and :class:`~ventorum.aero.vortex.Targets`.
"""

from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit, prange

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - numba is a dependency
    HAVE_NUMBA = False

_INV_4PI = 1.0 / (4.0 * math.pi)
_INV_2PI = 1.0 / (2.0 * math.pi)


def pack_sources(hs) -> tuple:
    """Return the source arrays of a horseshoe set in the layout of the kernels.

    The order is: a, b, a_te, b_te, wake_dir, squared core radius, group, sign.
    """
    group = hs.group if hs.group is not None else np.zeros(hs.n, dtype=np.int64)
    return (
        np.ascontiguousarray(hs.a, dtype=np.float64),
        np.ascontiguousarray(hs.b, dtype=np.float64),
        np.ascontiguousarray(hs.a_te, dtype=np.float64),
        np.ascontiguousarray(hs.b_te, dtype=np.float64),
        np.ascontiguousarray(hs.wake_dir, dtype=np.float64),
        np.ascontiguousarray(np.asarray(hs.rc, dtype=np.float64) ** 2),
        np.ascontiguousarray(group, dtype=np.int64),
        np.ascontiguousarray(hs.sign, dtype=np.float64),
    )


def pack_targets(m: int, hs, targets) -> tuple:
    """Return the target arrays: group, added squared core, and a flag that says if they apply."""
    if targets is None or hs.group is None:
        return np.zeros(m, dtype=np.int64), np.zeros(m), False
    return (np.ascontiguousarray(targets.group, dtype=np.int64),
            np.ascontiguousarray(np.asarray(targets.rc, dtype=np.float64) ** 2), True)


if HAVE_NUMBA:

    @njit(cache=True, inline="always")
    def _seg_terms(r1x, r1y, r1z, r2x, r2y, r2z, r0x, r0y, r0z, n1, n2, rc2):
        # Velocity of one finite segment from ready vectors and norms.
        # Same formula as the numpy kernel; a zero-length segment adds nothing.
        if r0x == 0.0 and r0y == 0.0 and r0z == 0.0:
            return 0.0, 0.0, 0.0
        cx = r1y * r2z - r1z * r2y
        cy = r1z * r2x - r1x * r2z
        cz = r1x * r2y - r1y * r2x
        cr2 = cx * cx + cy * cy + cz * cz
        r0sq = r0x * r0x + r0y * r0y + r0z * r0z
        proj = (r0x * (r1x / n1 - r2x / n2) + r0y * (r1y / n1 - r2y / n2)
                + r0z * (r1z / n1 - r2z / n2))
        denom = cr2 + rc2 * r0sq
        if denom > 1e-300:
            k = _INV_4PI * proj / denom
        else:
            k = 0.0
        return cx * k, cy * k, cz * k

    @njit(cache=True, inline="always")
    def _ray_terms(rx, ry, rz, rn, dx, dy, dz, rc2):
        # Velocity of one semi-infinite line from a ready vector and norm.
        # Same formula as the numpy kernel; dy == 0 uses the shorter form.
        if dy == 0.0:
            cx = -dz * ry
            cy = dz * rx - dx * rz
            cz = dx * ry
            cos_t = (dx * rx + dz * rz) / rn
        else:
            cx = dy * rz - dz * ry
            cy = dz * rx - dx * rz
            cz = dx * ry - dy * rx
            cos_t = (dx * rx + dy * ry + dz * rz) / rn
        cr2 = cx * cx + cy * cy + cz * cz
        denom = cr2 + rc2
        if denom > 1e-300:
            k = _INV_4PI * (1.0 + cos_t) / denom
        else:
            k = 0.0
        return cx * k, cy * k, cz * k

    @njit(cache=True, inline="always")
    def _horseshoe_straight(P, i, a, b, d, j, rc2):
        # Full horseshoe with zero-length chordwise legs (a_te is a, b_te is b).
        # The bound norms serve the wake legs too, as in the numpy kernel.
        px = P[i, 0]
        py = P[i, 1]
        pz = P[i, 2]
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
        n1 = math.sqrt(r1x * r1x + r1y * r1y + r1z * r1z + 1e-300)
        n2 = math.sqrt(r2x * r2x + r2y * r2y + r2z * r2z + 1e-300)
        vx, vy, vz = _seg_terms(r1x, r1y, r1z, r2x, r2y, r2z,
                                bx - ax, by - ay, bz - az, n1, n2, rc2)
        dx = d[j, 0]
        dy = d[j, 1]
        dz = d[j, 2]
        ux, uy, uz = _ray_terms(r2x, r2y, r2z, n2, dx, dy, dz, rc2)
        ex, ey, ez = _ray_terms(r1x, r1y, r1z, n1, dx, dy, dz, rc2)
        return vx + ux - ex, vy + uy - ey, vz + uz - ez

    @njit(cache=True, inline="always")
    def _horseshoe_generic(P, i, a, b, a_te, b_te, d, j, rc2, parts):
        # Bent horseshoe, part 0 all, 1 fixed legs, 2 wake legs. Vectors and
        # norms are shared between the legs; straight chordwise legs add nothing.
        px = P[i, 0]
        py = P[i, 1]
        pz = P[i, 2]
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
        r1x, r1y, r1z, n1 = 0.0, 0.0, 0.0, 0.0
        r2x, r2y, r2z, n2 = 0.0, 0.0, 0.0, 0.0
        if parts == 2:
            vx, vy, vz = 0.0, 0.0, 0.0
            if straight_left or straight_right:
                r1x = px - ax
                r1y = py - ay
                r1z = pz - az
                r2x = px - bx
                r2y = py - by
                r2z = pz - bz
                n1 = math.sqrt(r1x * r1x + r1y * r1y + r1z * r1z + 1e-300)
                n2 = math.sqrt(r2x * r2x + r2y * r2y + r2z * r2z + 1e-300)
        else:
            r1x = px - ax
            r1y = py - ay
            r1z = pz - az
            r2x = px - bx
            r2y = py - by
            r2z = pz - bz
            n1 = math.sqrt(r1x * r1x + r1y * r1y + r1z * r1z + 1e-300)
            n2 = math.sqrt(r2x * r2x + r2y * r2y + r2z * r2z + 1e-300)
            vx, vy, vz = _seg_terms(r1x, r1y, r1z, r2x, r2y, r2z,
                                    bx - ax, by - ay, bz - az, n1, n2, rc2)
            if not straight_left:
                q1x = px - tax
                q1y = py - tay
                q1z = pz - taz
                nq = math.sqrt(q1x * q1x + q1y * q1y + q1z * q1z + 1e-300)
                ux, uy, uz = _seg_terms(q1x, q1y, q1z, r1x, r1y, r1z,
                                        ax - tax, ay - tay, az - taz, nq, n1, rc2)
                vx += ux
                vy += uy
                vz += uz
            if not straight_right:
                w2x = px - tbx
                w2y = py - tby
                w2z = pz - tbz
                nw = math.sqrt(w2x * w2x + w2y * w2y + w2z * w2z + 1e-300)
                ux, uy, uz = _seg_terms(r2x, r2y, r2z, w2x, w2y, w2z,
                                        tbx - bx, tby - by, tbz - bz, n2, nw, rc2)
                vx += ux
                vy += uy
                vz += uz
        if parts == 1:
            return vx, vy, vz
        dx = d[j, 0]
        dy = d[j, 1]
        dz = d[j, 2]
        if straight_right:
            rbrx, rbry, rbrz, rbrn = r2x, r2y, r2z, n2
        else:
            rbrx = px - tbx
            rbry = py - tby
            rbrz = pz - tbz
            rbrn = math.sqrt(rbrx * rbrx + rbry * rbry + rbrz * rbrz + 1e-300)
        if straight_left:
            larx, lary, larz, larn = r1x, r1y, r1z, n1
        else:
            larx = px - tax
            lary = py - tay
            larz = pz - taz
            larn = math.sqrt(larx * larx + lary * lary + larz * larz + 1e-300)
        ux, uy, uz = _ray_terms(rbrx, rbry, rbrz, rbrn, dx, dy, dz, rc2)
        ex, ey, ez = _ray_terms(larx, lary, larz, larn, dx, dy, dz, rc2)
        if parts == 2:
            return ux - ex, uy - ey, uz - ez
        return vx + ux - ex, vy + uy - ey, vz + uz - ez

    @njit(cache=True)
    def _is_direct(column, sign, n_unknowns, n):
        # True when each source maps to its own unknown with unit sign.
        # Then out[i, j] = ... stores the result with no fold.
        if n_unknowns != n:
            return False
        for j in range(n):
            if column[j] != j or sign[j] != 1.0:
                return False
        return True

    @njit(cache=True)
    def _is_straight(a, b, a_te, b_te, n):
        # True when every chordwise leg has zero length (a_te is a, b_te is b).
        for j in range(n):
            if (a_te[j, 0] != a[j, 0] or a_te[j, 1] != a[j, 1] or a_te[j, 2] != a[j, 2]
                    or b_te[j, 0] != b[j, 0] or b_te[j, 1] != b[j, 1]
                    or b_te[j, 2] != b[j, 2]):
                return False
        return True

    @njit(cache=True, parallel=True)
    def induced_velocity_kernel(P, a, b, a_te, b_te, d, rc2_src, src_group, g_src, tg_group, tg_rc2, use_tg, parts=0):
        """Return the total induced velocity (m, 3) for the source strengths *g_src* (sign included).

        Straight horseshoes with one unknown per source take a short path with
        no fold; all other inputs use the general path. Both give the same
        result as the numpy kernel to round-off.
        """
        m = P.shape[0]
        n = a.shape[0]
        if parts == 0 and not use_tg and _is_straight(a, b, a_te, b_te, n):
            out = np.empty((m, 3))
            for i in prange(m):
                sx = 0.0
                sy = 0.0
                sz = 0.0
                for j in range(n):
                    vx, vy, vz = _horseshoe_straight(P, i, a, b, d, j, rc2_src[j])
                    g = g_src[j]
                    sx += vx * g
                    sy += vy * g
                    sz += vz * g
                out[i, 0] = sx
                out[i, 1] = sy
                out[i, 2] = sz
            return out
        out = np.empty((m, 3))
        for i in prange(m):
            sx = 0.0
            sy = 0.0
            sz = 0.0
            for j in range(n):
                rc2 = rc2_src[j]
                if use_tg and tg_group[i] != src_group[j]:
                    rc2 += tg_rc2[i]
                vx, vy, vz = _horseshoe_generic(P, i, a, b, a_te, b_te, d, j, rc2, parts)
                g = g_src[j]
                sx += vx * g
                sy += vy * g
                sz += vz * g
            out[i, 0] = sx
            out[i, 1] = sy
            out[i, 2] = sz
        return out

    @njit(cache=True, parallel=True)
    def normal_influence_kernel(P, normals, a, b, a_te, b_te, d, rc2_src, src_group, sign, column, n_unknowns,
                                tg_group, tg_rc2, use_tg, parts=0):
        """Return the normal-velocity influence matrix (m, n_unknowns).

        The sources that map to the same unknown (mirror and ground images)
        are added in source order. Straight horseshoes with one unknown per
        source and unit sign take a short path with no fold; all other inputs
        use the general path. Both give the same result as the numpy kernel
        to round-off.
        """
        m = P.shape[0]
        n = a.shape[0]
        if parts == 0 and not use_tg and _is_direct(column, sign, n_unknowns, n) \
                and _is_straight(a, b, a_te, b_te, n):
            out = np.empty((m, n_unknowns))
            for i in prange(m):
                nx = normals[i, 0]
                ny = normals[i, 1]
                nz = normals[i, 2]
                for j in range(n):
                    vx, vy, vz = _horseshoe_straight(P, i, a, b, d, j, rc2_src[j])
                    out[i, j] = vx * nx + vy * ny + vz * nz
            return out
        out = np.zeros((m, n_unknowns))
        for i in prange(m):
            nx = normals[i, 0]
            ny = normals[i, 1]
            nz = normals[i, 2]
            for j in range(n):
                rc2 = rc2_src[j]
                if use_tg and tg_group[i] != src_group[j]:
                    rc2 += tg_rc2[i]
                vx, vy, vz = _horseshoe_generic(P, i, a, b, a_te, b_te, d, j, rc2, parts)
                out[i, column[j]] += (vx * nx + vy * ny + vz * nz) * sign[j]
        return out

    @njit(cache=True, parallel=True)
    def velocity_tensor_kernel(P, a, b, a_te, b_te, d, rc2_src, src_group, sign, tg_group, tg_rc2, use_tg, parts=0):
        """Return the velocity (m, n_sources, 3) per unit source circulation (sign included)."""
        m = P.shape[0]
        n = a.shape[0]
        out = np.empty((m, n, 3))
        for i in prange(m):
            for j in range(n):
                rc2 = rc2_src[j]
                if use_tg and tg_group[i] != src_group[j]:
                    rc2 += tg_rc2[i]
                vx, vy, vz = _horseshoe_generic(P, i, a, b, a_te, b_te, d, j, rc2, parts)
                s = sign[j]
                out[i, j, 0] = vx * s
                out[i, j, 1] = vy * s
                out[i, j, 2] = vz * s
        return out

    @njit(cache=True, parallel=True)
    def velocity_unknowns_kernel(P, a, b, a_te, b_te, d, rc2_src, src_group, sign, column, n_unknowns,
                                 tg_group, tg_rc2, use_tg, parts=0):
        """Return the velocity (m, n_unknowns, 3) per unit circulation of each unknown.

        The same velocities as ``velocity_tensor_kernel``, added into the
        unknown of each source in source order while they are computed (each
        thread owns one row), so the tensor per source is never stored.
        """
        m = P.shape[0]
        n = a.shape[0]
        out = np.zeros((m, n_unknowns, 3))
        for i in prange(m):
            for j in range(n):
                rc2 = rc2_src[j]
                if use_tg and tg_group[i] != src_group[j]:
                    rc2 += tg_rc2[i]
                vx, vy, vz = _horseshoe_generic(P, i, a, b, a_te, b_te, d, j, rc2, parts)
                s = sign[j]
                c = column[j]
                out[i, c, 0] += vx * s
                out[i, c, 1] += vy * s
                out[i, c, 2] += vz * s
        return out

    @njit(cache=True, parallel=True)
    def trefftz_normalwash_kernel(q, q_normal, p, gamma, d, rc2, src_group, tg_group, tg_rc2, use_tg):
        """Return the normal wash (m,) of 2-D point vortices in the Trefftz plane.

        If *use_tg* is True, the squared core radius between a point and a
        vortex of a different core group gets the added term *tg_rc2* of the point.
        """
        m = q.shape[0]
        n = p.shape[0]
        out = np.empty(m)
        dx = d[0]
        dy = d[1]
        dz = d[2]
        for i in prange(m):
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
                    c2 += tg_rc2[i]
                k = _INV_2PI * gamma[j] / (rx * rx + ry * ry + rz * rz + c2)
                acc += (vx * nx + vy * ny + vz * nz) * k
            out[i] = acc
        return out
