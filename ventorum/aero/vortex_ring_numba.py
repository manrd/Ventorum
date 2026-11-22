# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Compiled (Numba) vortex-ring kernels for :mod:`ventorum.aero.vortex_ring`.

Each kernel computes the same Biot-Savart formulas, regularisation and floors
as the numpy reference kernels in :mod:`ventorum.aero.vortex_ring`, in one
parallel pass (``prange``) over the evaluation points with no ``(M, N, 3)``
intermediate arrays. Each corner displacement vector and norm of a ring is
computed once per pair and shared between its two adjacent segments.

Cython and GPU (NVIDIA Warp) backends for the ring kernels are planned for a
later step.

References
----------
* J. Katz and A. Plotkin, "Low-Speed Aerodynamics", 2nd ed., Cambridge
  University Press, 2001, chapters 10, 12 and 13 (Biot-Savart law for straight
  vortex segments, semi-infinite lines and vortex rings).
* M. P. Scully, "Computation of helicopter rotor wake geometry and its
  influence on rotor harmonic airloads", ASRL TR 178-1, MIT, 1975.
* G. H. Vatistas, V. Kozel and W. C. Mih, "A simpler model for concentrated
  vortices", Experiments in Fluids 11 (1991) 73-76.
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


def pack_rc2(rc: np.ndarray | float, m: int, n: int) -> tuple[np.ndarray, bool]:
    """Pack the squared core radius [m^2] into a contiguous 2-D array.

    Parameters
    ----------
    rc : numpy.ndarray or float
        Core radius [m], scalar, shape ``(n,)`` or shape ``(m, n)``.
    m : int
        Number of evaluation points.
    n : int
        Number of vortex elements.

    Returns
    -------
    rc2 : numpy.ndarray
        Contiguous float64 squared core radius [m^2], shape ``(m, n)`` when
        *rc* is 2-D or shape ``(1, n)`` otherwise.
    rc_is_2d : bool
        True when *rc2* varies with the evaluation point row.
    """
    arr = np.asarray(rc, dtype=np.float64)
    if arr.ndim == 0:
        return np.ascontiguousarray(np.full((1, n), float(arr) ** 2, dtype=np.float64)), False
    if arr.ndim == 1:
        if arr.shape[0] != n:
            raise ValueError(f"Expected rc of shape ({n},), got {arr.shape}.")
        return np.ascontiguousarray((arr ** 2).reshape(1, n)), False
    if arr.ndim == 2:
        if arr.shape != (m, n):
            raise ValueError(f"Expected rc of shape ({m}, {n}), got {arr.shape}.")
        return np.ascontiguousarray(arr ** 2), True
    raise ValueError(f"Expected rc to be scalar, 1-D or 2-D, got shape {arr.shape}.")


if HAVE_NUMBA:

    @njit(cache=True, inline="always")
    def _seg_terms(r1x, r1y, r1z, r2x, r2y, r2z, r0x, r0y, r0z, n1, n2, rc2):
        # Unit-circulation velocity of one straight finite segment.
        # Matches ventorum.aero.vortex.segment_velocity and vortex_numba._seg_terms.
        if r0x == 0.0 and r0y == 0.0 and r0z == 0.0:
            return 0.0, 0.0, 0.0
        cx = r1y * r2z - r1z * r2y
        cy = r1z * r2x - r1x * r2z
        cz = r1x * r2y - r1y * r2x
        before = r0x * r1x + r0y * r1y + r0z * r1z < 0.0
        after = r0x * r2x + r0y * r2y + r0z * r2z > 0.0
        if before or after:
            if before:
                dd = r1x * r1x + r1y * r1y + r1z * r1z + 1e-300
            else:
                dd = r2x * r2x + r2y * r2y + r2z * r2z + 1e-300
            e = n1 * n2 + r1x * r2x + r1y * r2y + r1z * r2z
            if e > 1e-300:
                k = _INV_4PI * (n1 + n2) / (n1 * n2 * e) * (dd / (dd + rc2))
            else:
                k = 0.0
            return cx * k, cy * k, cz * k
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
        # Unit-circulation velocity of one semi-infinite line.
        # Matches ventorum.aero.vortex.ray_velocity and vortex_numba._ray_terms.
        if dy == 0.0:
            cx = -dz * ry
            cy = dz * rx - dx * rz
            cz = dx * ry
            dr = dx * rx + dz * rz
        else:
            cx = dy * rz - dz * ry
            cy = dz * rx - dx * rz
            cz = dx * ry - dy * rx
            dr = dx * rx + dy * ry + dz * rz
        if dr < 0.0:
            dd = rx * rx + ry * ry + rz * rz + 1e-300
            e = rn * (rn - dr)
            if e > 1e-300:
                k = _INV_4PI / e * (dd / (dd + rc2))
            else:
                k = 0.0
            return cx * k, cy * k, cz * k
        cr2 = cx * cx + cy * cy + cz * cz
        denom = cr2 + rc2
        if denom > 1e-300:
            k = _INV_4PI * (1.0 + dr / rn) / denom
        else:
            k = 0.0
        return cx * k, cy * k, cz * k

    @njit(cache=True, inline="always")
    def _ring_vel_one(px, py, pz, corners, j, rc2):
        # Unit-circulation velocity at (px, py, pz) from ring j (four segments).
        c0x = corners[j, 0, 0]
        c0y = corners[j, 0, 1]
        c0z = corners[j, 0, 2]
        c1x = corners[j, 1, 0]
        c1y = corners[j, 1, 1]
        c1z = corners[j, 1, 2]
        c2x = corners[j, 2, 0]
        c2y = corners[j, 2, 1]
        c2z = corners[j, 2, 2]
        c3x = corners[j, 3, 0]
        c3y = corners[j, 3, 1]
        c3z = corners[j, 3, 2]

        r0x = px - c0x
        r0y = py - c0y
        r0z = pz - c0z
        r1x = px - c1x
        r1y = py - c1y
        r1z = pz - c1z
        r2x = px - c2x
        r2y = py - c2y
        r2z = pz - c2z
        r3x = px - c3x
        r3y = py - c3y
        r3z = pz - c3z

        n0 = math.sqrt(r0x * r0x + r0y * r0y + r0z * r0z + 1e-300)
        n1 = math.sqrt(r1x * r1x + r1y * r1y + r1z * r1z + 1e-300)
        n2 = math.sqrt(r2x * r2x + r2y * r2y + r2z * r2z + 1e-300)
        n3 = math.sqrt(r3x * r3x + r3y * r3y + r3z * r3z + 1e-300)

        vx, vy, vz = _seg_terms(r0x, r0y, r0z, r1x, r1y, r1z, c1x - c0x, c1y - c0y, c1z - c0z, n0, n1, rc2)
        ux, uy, uz = _seg_terms(r1x, r1y, r1z, r2x, r2y, r2z, c2x - c1x, c2y - c1y, c2z - c1z, n1, n2, rc2)
        vx += ux
        vy += uy
        vz += uz
        ux, uy, uz = _seg_terms(r2x, r2y, r2z, r3x, r3y, r3z, c3x - c2x, c3y - c2y, c3z - c2z, n2, n3, rc2)
        vx += ux
        vy += uy
        vz += uz
        ux, uy, uz = _seg_terms(r3x, r3y, r3z, r0x, r0y, r0z, c0x - c3x, c0y - c3y, c0z - c3z, n3, n0, rc2)
        return vx + ux, vy + uy, vz + uz

    @njit(cache=True, inline="always")
    def _ring_vel_poly_one(px, py, pz, corners, side_nodes, side_counts, j, rc2):
        # Unit-circulation velocity at (px, py, pz) from ring j with bent side polylines.
        c0x = corners[j, 0, 0]
        c0y = corners[j, 0, 1]
        c0z = corners[j, 0, 2]
        c1x = corners[j, 1, 0]
        c1y = corners[j, 1, 1]
        c1z = corners[j, 1, 2]
        c2x = corners[j, 2, 0]
        c2y = corners[j, 2, 1]
        c2z = corners[j, 2, 2]
        c3x = corners[j, 3, 0]
        c3y = corners[j, 3, 1]
        c3z = corners[j, 3, 2]

        r0x = px - c0x
        r0y = py - c0y
        r0z = pz - c0z
        r1x = px - c1x
        r1y = py - c1y
        r1z = pz - c1z
        r2x = px - c2x
        r2y = py - c2y
        r2z = pz - c2z
        r3x = px - c3x
        r3y = py - c3y
        r3z = pz - c3z

        n0 = math.sqrt(r0x * r0x + r0y * r0y + r0z * r0z + 1e-300)
        n1 = math.sqrt(r1x * r1x + r1y * r1y + r1z * r1z + 1e-300)
        n2 = math.sqrt(r2x * r2x + r2y * r2y + r2z * r2z + 1e-300)
        n3 = math.sqrt(r3x * r3x + r3y * r3y + r3z * r3z + 1e-300)

        # Leading bound segment: c0 -> c1.
        vx, vy, vz = _seg_terms(r0x, r0y, r0z, r1x, r1y, r1z, c1x - c0x, c1y - c0y, c1z - c0z, n0, n1, rc2)

        # Right side polyline: c1 -> side_nodes[j, 0, 0..cr-1] -> c2.
        cr = side_counts[j, 0]
        if cr == 0:
            ux, uy, uz = _seg_terms(r1x, r1y, r1z, r2x, r2y, r2z, c2x - c1x, c2y - c1y, c2z - c1z, n1, n2, rc2)
            vx += ux
            vy += uy
            vz += uz
        else:
            ax, ay, az = c1x, c1y, c1z
            rax, ray, raz, na = r1x, r1y, r1z, n1
            for k in range(cr):
                bx = side_nodes[j, 0, k, 0]
                by = side_nodes[j, 0, k, 1]
                bz = side_nodes[j, 0, k, 2]
                rbx = px - bx
                rby = py - by
                rbz = pz - bz
                nb = math.sqrt(rbx * rbx + rby * rby + rbz * rbz + 1e-300)
                ux, uy, uz = _seg_terms(rax, ray, raz, rbx, rby, rbz, bx - ax, by - ay, bz - az, na, nb, rc2)
                vx += ux
                vy += uy
                vz += uz
                ax, ay, az = bx, by, bz
                rax, ray, raz, na = rbx, rby, rbz, nb
            ux, uy, uz = _seg_terms(rax, ray, raz, r2x, r2y, r2z, c2x - ax, c2y - ay, c2z - az, na, n2, rc2)
            vx += ux
            vy += uy
            vz += uz

        # Trailing spanwise segment: c2 -> c3.
        ux, uy, uz = _seg_terms(r2x, r2y, r2z, r3x, r3y, r3z, c3x - c2x, c3y - c2y, c3z - c2z, n2, n3, rc2)
        vx += ux
        vy += uy
        vz += uz

        # Left side polyline: c3 -> side_nodes[j, 1, 0..cl-1] -> c0.
        cl = side_counts[j, 1]
        if cl == 0:
            ux, uy, uz = _seg_terms(r3x, r3y, r3z, r0x, r0y, r0z, c0x - c3x, c0y - c3y, c0z - c3z, n3, n0, rc2)
            vx += ux
            vy += uy
            vz += uz
        else:
            ax, ay, az = c3x, c3y, c3z
            rax, ray, raz, na = r3x, r3y, r3z, n3
            for k in range(cl):
                bx = side_nodes[j, 1, k, 0]
                by = side_nodes[j, 1, k, 1]
                bz = side_nodes[j, 1, k, 2]
                rbx = px - bx
                rby = py - by
                rbz = pz - bz
                nb = math.sqrt(rbx * rbx + rby * rby + rbz * rbz + 1e-300)
                ux, uy, uz = _seg_terms(rax, ray, raz, rbx, rby, rbz, bx - ax, by - ay, bz - az, na, nb, rc2)
                vx += ux
                vy += uy
                vz += uz
                ax, ay, az = bx, by, bz
                rax, ray, raz, na = rbx, rby, rbz, nb
            ux, uy, uz = _seg_terms(rax, ray, raz, r0x, r0y, r0z, c0x - ax, c0y - ay, c0z - az, na, n0, rc2)
            vx += ux
            vy += uy
            vz += uz

        return vx, vy, vz

    @njit(cache=True, inline="always")
    def _steady_wake_vel_one(px, py, pz, te_left, te_right, wake_dir, j, rc2):
        # Unit-circulation velocity of steady wake closure j:
        # segment te_left -> te_right plus ray(te_right, d) minus ray(te_left, d).
        lx = te_left[j, 0]
        ly = te_left[j, 1]
        lz = te_left[j, 2]
        rx = te_right[j, 0]
        ry = te_right[j, 1]
        rz = te_right[j, 2]

        rlx = px - lx
        rly = py - ly
        rlz = pz - lz
        rrx = px - rx
        rry = py - ry
        rrz = pz - rz

        nl = math.sqrt(rlx * rlx + rly * rly + rlz * rlz + 1e-300)
        nr = math.sqrt(rrx * rrx + rry * rry + rrz * rrz + 1e-300)

        vx, vy, vz = _seg_terms(rlx, rly, rlz, rrx, rry, rrz, rx - lx, ry - ly, rz - lz, nl, nr, rc2)
        dx = wake_dir[j, 0]
        dy = wake_dir[j, 1]
        dz = wake_dir[j, 2]
        ux, uy, uz = _ray_terms(rrx, rry, rrz, nr, dx, dy, dz, rc2)
        ex, ey, ez = _ray_terms(rlx, rly, rlz, nl, dx, dy, dz, rc2)
        return vx + ux - ex, vy + uy - ey, vz + uz - ez

    @njit(cache=True, parallel=True)
    def ring_velocity_kernel(P, corners, rc2, rc_is_2d):
        """Return unit-circulation ring velocities [1/m], shape ``(M, N, 3)``."""
        m = P.shape[0]
        n = corners.shape[0]
        out = np.empty((m, n, 3), dtype=np.float64)
        for i in prange(m):
            px = P[i, 0]
            py = P[i, 1]
            pz = P[i, 2]
            for j in range(n):
                c2 = rc2[i, j] if rc_is_2d else rc2[0, j]
                vx, vy, vz = _ring_vel_one(px, py, pz, corners, j, c2)
                out[i, j, 0] = vx
                out[i, j, 1] = vy
                out[i, j, 2] = vz
        return out

    @njit(cache=True, parallel=True)
    def ring_normal_influence_kernel(P, normals, corners, rc2, rc_is_2d):
        """Return unit-circulation ring normal influence [1/m], shape ``(M, N)``."""
        m = P.shape[0]
        n = corners.shape[0]
        out = np.empty((m, n), dtype=np.float64)
        for i in prange(m):
            px = P[i, 0]
            py = P[i, 1]
            pz = P[i, 2]
            nx = normals[i, 0]
            ny = normals[i, 1]
            nz = normals[i, 2]
            for j in range(n):
                c2 = rc2[i, j] if rc_is_2d else rc2[0, j]
                vx, vy, vz = _ring_vel_one(px, py, pz, corners, j, c2)
                out[i, j] = vx * nx + vy * ny + vz * nz
        return out

    @njit(cache=True, parallel=True)
    def ring_poly_velocity_kernel(P, corners, side_nodes, side_counts, rc2, rc_is_2d):
        """Return unit-circulation bent-ring velocities [1/m], shape ``(M, N, 3)``."""
        m = P.shape[0]
        n = corners.shape[0]
        out = np.empty((m, n, 3), dtype=np.float64)
        for i in prange(m):
            px = P[i, 0]
            py = P[i, 1]
            pz = P[i, 2]
            for j in range(n):
                c2 = rc2[i, j] if rc_is_2d else rc2[0, j]
                vx, vy, vz = _ring_vel_poly_one(px, py, pz, corners, side_nodes, side_counts, j, c2)
                out[i, j, 0] = vx
                out[i, j, 1] = vy
                out[i, j, 2] = vz
        return out

    @njit(cache=True, parallel=True)
    def ring_poly_normal_influence_kernel(P, normals, corners, side_nodes, side_counts, rc2, rc_is_2d):
        """Return unit-circulation bent-ring normal influence [1/m], shape ``(M, N)``."""
        m = P.shape[0]
        n = corners.shape[0]
        out = np.empty((m, n), dtype=np.float64)
        for i in prange(m):
            px = P[i, 0]
            py = P[i, 1]
            pz = P[i, 2]
            nx = normals[i, 0]
            ny = normals[i, 1]
            nz = normals[i, 2]
            for j in range(n):
                c2 = rc2[i, j] if rc_is_2d else rc2[0, j]
                vx, vy, vz = _ring_vel_poly_one(px, py, pz, corners, side_nodes, side_counts, j, c2)
                out[i, j] = vx * nx + vy * ny + vz * nz
        return out

    @njit(cache=True, parallel=True)
    def steady_wake_velocity_kernel(P, te_left, te_right, wake_dir, rc2, rc_is_2d):
        """Return unit-circulation steady-wake closure velocities [1/m], shape ``(M, N_s, 3)``."""
        m = P.shape[0]
        n = te_left.shape[0]
        out = np.empty((m, n, 3), dtype=np.float64)
        for i in prange(m):
            px = P[i, 0]
            py = P[i, 1]
            pz = P[i, 2]
            for j in range(n):
                c2 = rc2[i, j] if rc_is_2d else rc2[0, j]
                vx, vy, vz = _steady_wake_vel_one(px, py, pz, te_left, te_right, wake_dir, j, c2)
                out[i, j, 0] = vx
                out[i, j, 1] = vy
                out[i, j, 2] = vz
        return out

    @njit(cache=True, parallel=True)
    def steady_wake_normal_influence_kernel(P, normals, te_left, te_right, wake_dir, rc2, rc_is_2d):
        """Return unit-circulation steady-wake closure normal influence [1/m], shape ``(M, N_s)``."""
        m = P.shape[0]
        n = te_left.shape[0]
        out = np.empty((m, n), dtype=np.float64)
        for i in prange(m):
            px = P[i, 0]
            py = P[i, 1]
            pz = P[i, 2]
            nx = normals[i, 0]
            ny = normals[i, 1]
            nz = normals[i, 2]
            for j in range(n):
                c2 = rc2[i, j] if rc_is_2d else rc2[0, j]
                vx, vy, vz = _steady_wake_vel_one(px, py, pz, te_left, te_right, wake_dir, j, c2)
                out[i, j] = vx * nx + vy * ny + vz * nz
        return out
