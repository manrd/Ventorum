# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Compiled (Numba) versions of the 3D panel kernels of :mod:`ventorum.panel3d.kernels`.

The functions here compute the same quantities as the numpy reference
kernels, with the same formulas and the same branches: planar projection of
the corners, the van Oosterom and Strackee solid angle of the two panel
triangles, the body-side limit in the on-plane band, the analytic edge sums,
the vortex-ring velocity of the doublet, and the point far-field expansions
beyond ``far_ratio`` panel diameters. The outer loop runs over the target
points in parallel (``prange``), and no temporary array of target-panel
pairs is built. The numpy kernels stay the reference: the tests require
both paths to agree to round-off.

The pair kernels (:func:`doublet_pair_matrix_numba`,
:func:`source_pair_rhs_numba`, :func:`pair_system_numba`) evaluate a panel
and its mirror (or image) for the same target in one pass. They return the
parity system ``C_a + sign * C_b`` and the source right-hand sides
``(B_a + sign * B_b) @ sigma`` without storing the separate matrices or
the source matrix.

As in the reference, the computation runs in float64 and the outputs take
the dtype of the inputs. The on-plane band uses the machine epsilon of the
input dtype, so float32 inputs (for example collocation points rounded to
float32) keep the body-side self term +1/2. The velocities are the gradient
of the potential, V = +grad(Phi).

References
----------
* J. L. Hess and A. M. O. Smith, "Calculation of non-lifting potential flow
  about arbitrary three-dimensional bodies", Journal of Ship Research 8 (1964)
  22-44.
* J. L. Hess and A. M. O. Smith, "Calculation of potential flow about arbitrary
  bodies", Progress in Aeronautical Sciences 8 (1967) 1-138, Pergamon Press.
* J. Katz and A. Plotkin, "Low-Speed Aerodynamics", 2nd ed., Cambridge University
  Press, 2001, section 10.4 and chapter 12.
* J. N. Newman, "Distributions of sources and normal dipoles over a quadrilateral
  panel", Journal of Engineering Mathematics 20 (1986) 113-126.
* A. van Oosterom and J. Strackee, "The solid angle of a plane triangle", IEEE
  Transactions on Biomedical Engineering 30 (1983) 125-126.
"""

from __future__ import annotations

import math

import numpy as np

from ventorum.panel3d.kernels import _DEGENERATE_EDGE, _PLANE_TOL_FACTOR, _input_eps

try:
    from numba import njit, prange

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - numba is an allowed dependency
    HAVE_NUMBA = False

__all__ = [
    "source_velocity_numba",
    "doublet_velocity_numba",
    "influence_matrices_numba",
    "doublet_pair_matrix_numba",
    "source_rhs_numba",
    "source_pair_rhs_numba",
    "pair_system_numba",
]

_INV_4PI = 1.0 / (4.0 * math.pi)
_FLOOR = 1e-300


if HAVE_NUMBA:

    @njit(cache=True, inline="always")
    def _tri_omega(x, y, z, x0, y0, x1, y1, x2, y2, r0, r1, r2):
        # Solid angle of one triangle from the van Oosterom and Strackee formula.
        area2 = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
        num = -area2 * z
        d01 = (x0 - x) * (x1 - x) + (y0 - y) * (y1 - y) + z * z
        d02 = (x0 - x) * (x2 - x) + (y0 - y) * (y2 - y) + z * z
        d12 = (x1 - x) * (x2 - x) + (y1 - y) * (y2 - y) + z * z
        den = r0 * r1 * r2 + d01 * r2 + d02 * r1 + d12 * r0
        return 2.0 * math.atan2(num, den)

    @njit(cache=True, inline="always")
    def _plane_winding(x, y, xl, yl, elen, act, tol):
        # Body-side limit in the on-plane band: in-plane angle of the panel.
        total = 0.0
        for k in range(4):
            if act[k]:
                kn = k + 1 if k < 3 else 0
                ax = xl[k] - x
                ay = yl[k] - y
                bx = xl[kn] - x
                by = yl[kn] - y
                cr = ax * by - ay * bx
                dt = ax * bx + ay * by
                ra = math.hypot(ax, ay)
                rb = math.hypot(bx, by)
                on_edge = (abs(cr) <= tol * elen[k] and dt <= 0.0) or ra <= tol or rb <= tol
                if not on_edge:
                    total += math.atan2(cr, dt)
        return total

    @njit(cache=True, inline="always")
    def _solid_angle(x, y, z, xl, yl, elen, act, tol):
        # Solid angle of the panel: triangles (0, 1, 2) and (0, 2, 3).
        # In the on-plane band the body-side limit holds.
        if abs(z) <= tol:
            return _plane_winding(x, y, xl, yl, elen, act, tol)
        r0 = math.sqrt((x - xl[0]) ** 2 + (y - yl[0]) ** 2 + z * z)
        r1 = math.sqrt((x - xl[1]) ** 2 + (y - yl[1]) ** 2 + z * z)
        r2 = math.sqrt((x - xl[2]) ** 2 + (y - yl[2]) ** 2 + z * z)
        r3 = math.sqrt((x - xl[3]) ** 2 + (y - yl[3]) ** 2 + z * z)
        om = _tri_omega(x, y, z, xl[0], yl[0], xl[1], yl[1], xl[2], yl[2], r0, r1, r2)
        if act[3]:
            om += _tri_omega(x, y, z, xl[0], yl[0], xl[2], yl[2], xl[3], yl[3], r0, r2, r3)
        return om

    @njit(cache=True, inline="always")
    def _geom_row(pan, cen, nor, lax, maxis, xl, yl, dx, dy, elen, act, cor, eps):
        # Geometry of one panel: frame, projection, edges, area, diameter.
        # Same steps as the reference precompute. Returns (area, diameter, tol).
        diam = 0.0
        for a in range(4):
            for b in range(a + 1, 4):
                dd = math.sqrt((pan[a, 0] - pan[b, 0]) ** 2 + (pan[a, 1] - pan[b, 1]) ** 2
                               + (pan[a, 2] - pan[b, 2]) ** 2)
                if dd > diam:
                    diam = dd
        tiny = _DEGENERATE_EDGE * diam
        d13x = pan[2, 0] - pan[0, 0]
        d13y = pan[2, 1] - pan[0, 1]
        d13z = pan[2, 2] - pan[0, 2]
        d24x = pan[3, 0] - pan[1, 0]
        d24y = pan[3, 1] - pan[1, 1]
        d24z = pan[3, 2] - pan[1, 2]
        nrx = d13y * d24z - d13z * d24y
        nry = d13z * d24x - d13x * d24z
        nrz = d13x * d24y - d13y * d24x
        nn = math.sqrt(nrx * nrx + nry * nry + nrz * nrz)
        area = 0.5 * nn
        inv = 1.0 / (nn if nn > _FLOOR else _FLOOR)
        nor[0] = nrx * inv
        nor[1] = nry * inv
        nor[2] = nrz * inv
        pax = 0.25 * (pan[0, 0] + pan[1, 0] + pan[2, 0] + pan[3, 0])
        pay = 0.25 * (pan[0, 1] + pan[1, 1] + pan[2, 1] + pan[3, 1])
        paz = 0.25 * (pan[0, 2] + pan[1, 2] + pan[2, 2] + pan[3, 2])
        for k in range(4):
            dist = (pan[k, 0] - pax) * nor[0] + (pan[k, 1] - pay) * nor[1] + (pan[k, 2] - paz) * nor[2]
            cor[k, 0] = pan[k, 0] - dist * nor[0]
            cor[k, 1] = pan[k, 1] - dist * nor[1]
            cor[k, 2] = pan[k, 2] - dist * nor[2]
        t43 = math.sqrt((pan[3, 0] - pan[2, 0]) ** 2 + (pan[3, 1] - pan[2, 1]) ** 2
                        + (pan[3, 2] - pan[2, 2]) ** 2)
        if t43 <= tiny:
            cen[0] = (cor[0, 0] + cor[1, 0] + cor[2, 0]) / 3.0
            cen[1] = (cor[0, 1] + cor[1, 1] + cor[2, 1]) / 3.0
            cen[2] = (cor[0, 2] + cor[1, 2] + cor[2, 2]) / 3.0
        else:
            cen[0] = 0.25 * (cor[0, 0] + cor[1, 0] + cor[2, 0] + cor[3, 0])
            cen[1] = 0.25 * (cor[0, 1] + cor[1, 1] + cor[2, 1] + cor[3, 1])
            cen[2] = 0.25 * (cor[0, 2] + cor[1, 2] + cor[2, 2] + cor[3, 2])
        v12x = cor[1, 0] - cor[0, 0]
        v12y = cor[1, 1] - cor[0, 1]
        v12z = cor[1, 2] - cor[0, 2]
        l12 = math.sqrt(v12x * v12x + v12y * v12y + v12z * v12z)
        v13x = cor[2, 0] - cor[0, 0]
        v13y = cor[2, 1] - cor[0, 1]
        v13z = cor[2, 2] - cor[0, 2]
        l13 = math.sqrt(v13x * v13x + v13y * v13y + v13z * v13z)
        if l12 <= tiny:
            lrx, lry, lrz, ln = v13x, v13y, v13z, l13
        else:
            lrx, lry, lrz, ln = v12x, v12y, v12z, l12
        linv = 1.0 / (ln if ln > _FLOOR else _FLOOR)
        lax[0] = lrx * linv
        lax[1] = lry * linv
        lax[2] = lrz * linv
        maxis[0] = nor[1] * lax[2] - nor[2] * lax[1]
        maxis[1] = nor[2] * lax[0] - nor[0] * lax[2]
        maxis[2] = nor[0] * lax[1] - nor[1] * lax[0]
        for k in range(4):
            ddx = cor[k, 0] - cen[0]
            ddy = cor[k, 1] - cen[1]
            ddz = cor[k, 2] - cen[2]
            xl[k] = ddx * lax[0] + ddy * lax[1] + ddz * lax[2]
            yl[k] = ddx * maxis[0] + ddy * maxis[1] + ddz * maxis[2]
        for k in range(4):
            kn = k + 1 if k < 3 else 0
            dx[k] = xl[kn] - xl[k]
            dy[k] = yl[kn] - yl[k]
            elen[k] = math.hypot(dx[k], dy[k])
            act[k] = elen[k] > tiny
        cnorm = math.sqrt(cen[0] * cen[0] + cen[1] * cen[1] + cen[2] * cen[2])
        tol = _PLANE_TOL_FACTOR * eps * (cnorm + diam)
        return area, diam, tol

    @njit(cache=True, inline="always")
    def _edge_log(x, y, z, xa, ya, xb, yb, dk):
        # Edge logarithm ln((ra + rb + dk) / (ra + rb - dk)) without cancellation.
        ax = xa - x
        ay = ya - y
        bx = xb - x
        by = yb - y
        z2 = z * z
        ra = math.sqrt(ax * ax + ay * ay + z2)
        rb = math.sqrt(bx * bx + by * by + z2)
        dot = ax * bx + ay * by + z2
        rr = ra * rb
        if dot >= 0.0:
            s = rr + dot
        else:
            cr = ax * by - ay * bx
            dd = rr - dot
            s = (cr * cr + z2 * dk * dk) / (dd if dd > _FLOOR else _FLOOR)
        if s > 0.0:
            return 2.0 * math.log(ra + rb + dk) - math.log(2.0 * s)
        return 0.0

    @njit(cache=True, inline="always")
    def _pair_bc(tx, ty, tz, cen, nor, lax, maxis, xl, yl, dx, dy, elen, act, area, diam, tol,
                 far_ratio, use_far):
        # Source (B) and doublet (C) potentials of one target-panel pair.
        rx = tx - cen[0]
        ry = ty - cen[1]
        rz = tz - cen[2]
        r = math.sqrt(rx * rx + ry * ry + rz * rz)
        if use_far and r > far_ratio * diam:
            r_safe = r if r > _FLOOR else _FLOOR
            b = -_INV_4PI * area / r_safe
            rdotn = rx * nor[0] + ry * nor[1] + rz * nor[2]
            r3 = r_safe * r_safe * r_safe
            r3 = r3 if r3 > _FLOOR else _FLOOR
            c = -_INV_4PI * area * rdotn / r3
            return b, c
        x = rx * lax[0] + ry * lax[1] + rz * lax[2]
        y = rx * maxis[0] + ry * maxis[1] + rz * maxis[2]
        z = rx * nor[0] + ry * nor[1] + rz * nor[2]
        om = _solid_angle(x, y, z, xl, yl, elen, act, tol)
        s = 0.0
        for k in range(4):
            if act[k]:
                kn = k + 1 if k < 3 else 0
                dk = elen[k]
                dk_safe = dk if dk > _FLOOR else _FLOOR
                lk = _edge_log(x, y, z, xl[k], yl[k], xl[kn], yl[kn], dk)
                rperp = ((x - xl[k]) * dy[k] - (y - yl[k]) * dx[k]) / dk_safe
                s += rperp * lk
        b = _INV_4PI * (s - z * om)
        c = _INV_4PI * om
        return b, c

    @njit(cache=True, inline="always")
    def _pair_source_vel(tx, ty, tz, cen, nor, lax, maxis, xl, yl, dx, dy, elen, act, area, diam,
                         tol, far_ratio, use_far):
        # Induced velocity (+grad Phi) per unit source strength of one pair in m/s.
        rx = tx - cen[0]
        ry = ty - cen[1]
        rz = tz - cen[2]
        r = math.sqrt(rx * rx + ry * ry + rz * rz)
        if use_far and r > far_ratio * diam:
            r3 = r * r * r
            r3 = r3 if r3 > _FLOOR else _FLOOR
            f = _INV_4PI * area / r3
            return f * rx, f * ry, f * rz
        x = rx * lax[0] + ry * lax[1] + rz * lax[2]
        y = rx * maxis[0] + ry * maxis[1] + rz * maxis[2]
        z = rx * nor[0] + ry * nor[1] + rz * nor[2]
        om = _solid_angle(x, y, z, xl, yl, elen, act, tol)
        svx = 0.0
        svy = 0.0
        for k in range(4):
            if act[k]:
                kn = k + 1 if k < 3 else 0
                dk = elen[k]
                inv_dk = 1.0 / (dk if dk > _FLOOR else _FLOOR)
                lk = _edge_log(x, y, z, xl[k], yl[k], xl[kn], yl[kn], dk)
                svx += dy[k] * inv_dk * lk
                svy += -dx[k] * inv_dk * lk
        vx = _INV_4PI * svx
        vy = _INV_4PI * svy
        vz = -_INV_4PI * om
        gx = vx * lax[0] + vy * maxis[0] + vz * nor[0]
        gy = vx * lax[1] + vy * maxis[1] + vz * nor[1]
        gz = vx * lax[2] + vy * maxis[2] + vz * nor[2]
        return gx, gy, gz

    @njit(cache=True, inline="always")
    def _seg_vel(tx, ty, tz, sax, say, saz, eax, eay, eaz, tol):
        # Biot-Savart velocity of one straight vortex segment with unit circulation.
        r1x = tx - sax
        r1y = ty - say
        r1z = tz - saz
        r2x = tx - eax
        r2y = ty - eay
        r2z = tz - eaz
        r0x = eax - sax
        r0y = eay - say
        r0z = eaz - saz
        cx = r1y * r2z - r1z * r2y
        cy = r1z * r2x - r1x * r2z
        cz = r1x * r2y - r1y * r2x
        cr2 = cx * cx + cy * cy + cz * cz
        n1 = math.sqrt(r1x * r1x + r1y * r1y + r1z * r1z)
        n2 = math.sqrt(r2x * r2x + r2y * r2y + r2z * r2z)
        dot = r1x * r2x + r1y * r2y + r1z * r2z
        nn = n1 * n2
        r0_sq = r0x * r0x + r0y * r0y + r0z * r0z
        if n1 <= tol or n2 <= tol:
            return 0.0, 0.0, 0.0
        if dot < 0.0 and cr2 <= tol * tol * r0_sq:
            return 0.0, 0.0, 0.0
        if dot >= 0.0:
            den = nn + dot
        else:
            dd = nn - dot
            den = cr2 / (dd if dd > _FLOOR else _FLOOR)
        q = nn * den
        if not q > 0.0:
            return 0.0, 0.0, 0.0
        f = _INV_4PI * (n1 + n2) / q
        if not math.isfinite(f):
            return 0.0, 0.0, 0.0
        return cx * f, cy * f, cz * f

    @njit(cache=True, inline="always")
    def _pair_doublet_vel(tx, ty, tz, cen, nor, act, cor, area, diam, tol, far_ratio, use_far):
        # Induced velocity (+grad Phi) per unit doublet strength of one pair in m/s.
        rx = tx - cen[0]
        ry = ty - cen[1]
        rz = tz - cen[2]
        r = math.sqrt(rx * rx + ry * ry + rz * rz)
        if use_far and r > far_ratio * diam:
            rdotn = rx * nor[0] + ry * nor[1] + rz * nor[2]
            r3 = r * r * r
            r3 = r3 if r3 > _FLOOR else _FLOOR
            r5 = r3 * r * r
            r5 = r5 if r5 > _FLOOR else _FLOOR
            f = -_INV_4PI * area
            gx = f * (nor[0] / r3 - 3.0 * rdotn * rx / r5)
            gy = f * (nor[1] / r3 - 3.0 * rdotn * ry / r5)
            gz = f * (nor[2] / r3 - 3.0 * rdotn * rz / r5)
            return gx, gy, gz
        ringx = 0.0
        ringy = 0.0
        ringz = 0.0
        for k in range(4):
            if act[k]:
                kn = k + 1 if k < 3 else 0
                sx, sy, sz = _seg_vel(tx, ty, tz, cor[k, 0], cor[k, 1], cor[k, 2],
                                      cor[kn, 0], cor[kn, 1], cor[kn, 2], tol)
                ringx += sx
                ringy += sy
                ringz += sz
        return ringx, ringy, ringz

    @njit(cache=True)
    def _geometry_nb(panels, eps):
        # Geometry of all panels (serial loop, cheap next to the pair loop).
        nM = panels.shape[0]
        cen = np.empty((nM, 3), dtype=np.float64)
        nor = np.empty((nM, 3), dtype=np.float64)
        lax = np.empty((nM, 3), dtype=np.float64)
        maxis = np.empty((nM, 3), dtype=np.float64)
        xl = np.empty((nM, 4), dtype=np.float64)
        yl = np.empty((nM, 4), dtype=np.float64)
        dx = np.empty((nM, 4), dtype=np.float64)
        dy = np.empty((nM, 4), dtype=np.float64)
        elen = np.empty((nM, 4), dtype=np.float64)
        act = np.empty((nM, 4), dtype=np.bool_)
        cor = np.empty((nM, 4, 3), dtype=np.float64)
        area = np.empty(nM, dtype=np.float64)
        diam = np.empty(nM, dtype=np.float64)
        tol = np.empty(nM, dtype=np.float64)
        for j in range(nM):
            area[j], diam[j], tol[j] = _geom_row(panels[j], cen[j], nor[j], lax[j], maxis[j], xl[j],
                                                 yl[j], dx[j], dy[j], elen[j], act[j], cor[j], eps)
        return cen, nor, lax, maxis, xl, yl, dx, dy, elen, act, cor, area, diam, tol

    @njit(cache=True, parallel=True, fastmath=False)
    def _influence_nb(P, panels, B, C, far_ratio, use_far, eps):
        # Influence matrices: prange over the target points.
        cen, nor, lax, maxis, xl, yl, dx, dy, elen, act, cor, area, diam, tol = _geometry_nb(panels, eps)
        nM = panels.shape[0]
        for i in prange(P.shape[0]):
            for j in range(nM):
                b, c = _pair_bc(P[i, 0], P[i, 1], P[i, 2], cen[j], nor[j], lax[j], maxis[j],
                                xl[j], yl[j], dx[j], dy[j], elen[j], act[j], area[j], diam[j],
                                tol[j], far_ratio, use_far)
                B[i, j] = b
                C[i, j] = c

    @njit(cache=True, parallel=True, fastmath=False)
    def _source_vel_nb(P, panels, V, far_ratio, use_far, eps):
        # Source velocities: prange over the target points.
        cen, nor, lax, maxis, xl, yl, dx, dy, elen, act, cor, area, diam, tol = _geometry_nb(panels, eps)
        nM = panels.shape[0]
        for i in prange(P.shape[0]):
            for j in range(nM):
                gx, gy, gz = _pair_source_vel(P[i, 0], P[i, 1], P[i, 2], cen[j], nor[j], lax[j],
                                              maxis[j], xl[j], yl[j], dx[j], dy[j], elen[j], act[j],
                                              area[j], diam[j], tol[j], far_ratio, use_far)
                V[i, j, 0] = gx
                V[i, j, 1] = gy
                V[i, j, 2] = gz

    @njit(cache=True, parallel=True, fastmath=False)
    def _doublet_vel_nb(P, panels, V, far_ratio, use_far, eps):
        # Doublet velocities: prange over the target points.
        cen, nor, lax, maxis, xl, yl, dx, dy, elen, act, cor, area, diam, tol = _geometry_nb(panels, eps)
        nM = panels.shape[0]
        for i in prange(P.shape[0]):
            for j in range(nM):
                gx, gy, gz = _pair_doublet_vel(P[i, 0], P[i, 1], P[i, 2], cen[j], nor[j], act[j],
                                               cor[j], area[j], diam[j], tol[j], far_ratio, use_far)
                V[i, j, 0] = gx
                V[i, j, 1] = gy
                V[i, j, 2] = gz

    @njit(cache=True, inline="always")
    def _pair_c(tx, ty, tz, cen, nor, lax, maxis, xl, yl, elen, act, area, diam, tol, far_ratio,
                use_far):
        # Doublet (C) potential of one target-panel pair: the same steps as the
        # C part of _pair_bc, without the edge sums of the source potential.
        rx = tx - cen[0]
        ry = ty - cen[1]
        rz = tz - cen[2]
        r = math.sqrt(rx * rx + ry * ry + rz * rz)
        if use_far and r > far_ratio * diam:
            r_safe = r if r > _FLOOR else _FLOOR
            rdotn = rx * nor[0] + ry * nor[1] + rz * nor[2]
            r3 = r_safe * r_safe * r_safe
            r3 = r3 if r3 > _FLOOR else _FLOOR
            return -_INV_4PI * area * rdotn / r3
        x = rx * lax[0] + ry * lax[1] + rz * lax[2]
        y = rx * maxis[0] + ry * maxis[1] + rz * maxis[2]
        z = rx * nor[0] + ry * nor[1] + rz * nor[2]
        om = _solid_angle(x, y, z, xl, yl, elen, act, tol)
        return _INV_4PI * om

    @njit(cache=True, parallel=True, fastmath=False)
    def _pair_nb(P, pan_a, pan_b, sign, has_b, S, C, R, want_c, want_rhs, far_ratio, use_far, eps):
        # Pair kernel: C = C_a + sign * C_b and R = (B_a + sign * B_b) @ S.
        # prange over the target points; each row of R is private to one
        # target, so the sums need no atomic additions. With has_b False
        # the second panel set is not read.
        ga = _geometry_nb(pan_a, eps)
        gb = _geometry_nb(pan_b, eps)
        cen_a, nor_a, lax_a, max_a, xl_a, yl_a, dx_a, dy_a, el_a, act_a, cor_a, ar_a, di_a, tol_a = ga
        cen_b, nor_b, lax_b, max_b, xl_b, yl_b, dx_b, dy_b, el_b, act_b, cor_b, ar_b, di_b, tol_b = gb
        nM = pan_a.shape[0]
        nK = S.shape[1]
        for i in prange(P.shape[0]):
            px = P[i, 0]
            py = P[i, 1]
            pz = P[i, 2]
            acc = np.zeros(nK, dtype=np.float64)
            for j in range(nM):
                if want_rhs:
                    b, c = _pair_bc(px, py, pz, cen_a[j], nor_a[j], lax_a[j], max_a[j], xl_a[j],
                                    yl_a[j], dx_a[j], dy_a[j], el_a[j], act_a[j], ar_a[j], di_a[j],
                                    tol_a[j], far_ratio, use_far)
                    if has_b:
                        b2, c2 = _pair_bc(px, py, pz, cen_b[j], nor_b[j], lax_b[j], max_b[j],
                                          xl_b[j], yl_b[j], dx_b[j], dy_b[j], el_b[j], act_b[j],
                                          ar_b[j], di_b[j], tol_b[j], far_ratio, use_far)
                        b = b + sign * b2
                        c = c + sign * c2
                    for k in range(nK):
                        acc[k] += b * S[j, k]
                else:
                    c = _pair_c(px, py, pz, cen_a[j], nor_a[j], lax_a[j], max_a[j], xl_a[j],
                                yl_a[j], el_a[j], act_a[j], ar_a[j], di_a[j], tol_a[j], far_ratio,
                                use_far)
                    if has_b:
                        c2 = _pair_c(px, py, pz, cen_b[j], nor_b[j], lax_b[j], max_b[j], xl_b[j],
                                     yl_b[j], el_b[j], act_b[j], ar_b[j], di_b[j], tol_b[j],
                                     far_ratio, use_far)
                        c = c + sign * c2
                if want_c:
                    C[i, j] = c
            if want_rhs:
                for k in range(nK):
                    R[i, k] = acc[k]


def _work_arrays(collocation, panels):
    """Return the target dtype, the float64 work arrays and the input epsilon [-]."""
    pts = np.asarray(collocation)
    pans = np.asarray(panels)
    target_dtype = np.result_type(pts.dtype, pans.dtype)
    eps = _input_eps(pts, pans)
    return (target_dtype,
            np.ascontiguousarray(pts, dtype=np.float64),
            np.ascontiguousarray(pans, dtype=np.float64),
            eps)


def _require_numba():
    """Raise a clear error when Numba is missing."""
    if not HAVE_NUMBA:
        raise RuntimeError("Numba is not installed, so the Numba panel kernels cannot run.")


def source_velocity_numba(
    points: np.ndarray,
    panels: np.ndarray,
    far_ratio: float = 5.0,
) -> np.ndarray:
    """Compute induced velocity per unit source strength at target points.

    Same results as :func:`ventorum.panel3d.kernels.source_velocity`,
    in compiled code on the CPU. The velocity is +grad(Phi).

    Parameters
    ----------
    points : numpy.ndarray
        Target coordinates with shape (P, 3) in m.
    panels : numpy.ndarray
        Panel corner coordinates with shape (M, 4, 3) in m.
    far_ratio : float, default=5.0
        Distance-to-diameter ratio beyond which point source expansion is used.
        Setting to numpy.inf forces exact integration everywhere.

    Returns
    -------
    numpy.ndarray
        Induced velocity vectors with shape (P, M, 3) in m/s per unit source
        strength [m/s]. Dtype matches input.
    """
    _require_numba()
    target_dtype, pts, pans, eps = _work_arrays(points, panels)
    out = np.empty((pts.shape[0], pans.shape[0], 3), dtype=np.float64)
    if pts.shape[0] == 0 or pans.shape[0] == 0:
        return out.astype(target_dtype)
    use_far = not np.isinf(far_ratio)
    _source_vel_nb(pts, pans, out, float(far_ratio), use_far, eps)
    return out.astype(target_dtype)


def doublet_velocity_numba(
    points: np.ndarray,
    panels: np.ndarray,
    far_ratio: float = 5.0,
) -> np.ndarray:
    """Compute induced velocity per unit doublet strength at target points.

    Same results as :func:`ventorum.panel3d.kernels.doublet_velocity`,
    in compiled code on the CPU. The velocity is +grad(Phi).

    Parameters
    ----------
    points : numpy.ndarray
        Target coordinates with shape (P, 3) in m.
    panels : numpy.ndarray
        Panel corner coordinates with shape (M, 4, 3) in m.
    far_ratio : float, default=5.0
        Distance-to-diameter ratio beyond which point doublet expansion is used.
        Setting to numpy.inf forces exact integration everywhere.

    Returns
    -------
    numpy.ndarray
        Induced velocity vectors with shape (P, M, 3) in m/s per unit doublet
        strength [m^2/s]. Dtype matches input.
    """
    _require_numba()
    target_dtype, pts, pans, eps = _work_arrays(points, panels)
    out = np.empty((pts.shape[0], pans.shape[0], 3), dtype=np.float64)
    if pts.shape[0] == 0 or pans.shape[0] == 0:
        return out.astype(target_dtype)
    use_far = not np.isinf(far_ratio)
    _doublet_vel_nb(pts, pans, out, float(far_ratio), use_far, eps)
    return out.astype(target_dtype)


def influence_matrices_numba(
    collocation: np.ndarray,
    panels: np.ndarray,
    far_ratio: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute aerodynamic influence matrices B and C in compiled code.

    Same results as :func:`ventorum.panel3d.kernels.influence_matrices`,
    in compiled code on the CPU.

    Parameters
    ----------
    collocation : numpy.ndarray
        Collocation point coordinates with shape (P, 3) in m.
    panels : numpy.ndarray
        Panel corner coordinates with shape (M, 4, 3) in m.
    far_ratio : float, default=5.0
        Distance-to-diameter ratio for point far-field transition.

    Returns
    -------
    tuple of numpy.ndarray
        (B, C) where B has shape (P, M) for source potentials and C has shape
        (P, M) for doublet potentials. Dtype matches input.
    """
    _require_numba()
    target_dtype, pts, pans, eps = _work_arrays(collocation, panels)
    mat_b = np.empty((pts.shape[0], pans.shape[0]), dtype=np.float64)
    mat_c = np.empty((pts.shape[0], pans.shape[0]), dtype=np.float64)
    if pts.shape[0] == 0 or pans.shape[0] == 0:
        return mat_b.astype(target_dtype), mat_c.astype(target_dtype)
    use_far = not np.isinf(far_ratio)
    _influence_nb(pts, pans, mat_b, mat_c, float(far_ratio), use_far, eps)
    return mat_b.astype(target_dtype), mat_c.astype(target_dtype)


def _run_pair_numba(points, panels_a, panels_b, sign, sigma, far_ratio, want_c, want_rhs):
    """Run the compiled pair kernel and return (C, R, target dtype, R dtype)."""
    from ventorum.panel3d.kernels import _pair_inputs

    _require_numba()
    target_dtype, pts, pa, pb, sig, flat, rhs_dtype, eps = _pair_inputs(
        points, panels_a, panels_b, sigma if want_rhs else None
    )
    pts = np.ascontiguousarray(pts)
    pa = np.ascontiguousarray(pa)
    has_b = pb is not None
    pb = np.ascontiguousarray(pb) if has_b else np.empty((0, 4, 3), dtype=np.float64)
    n_p, n_m = pts.shape[0], pa.shape[0]
    mat = np.empty((n_p, n_m) if want_c else (1, 1), dtype=np.float64)
    if want_rhs:
        sig = np.ascontiguousarray(sig)
        rhs = np.zeros((n_p, sig.shape[1]), dtype=np.float64)
    else:
        sig = np.zeros((1, 0), dtype=np.float64)
        rhs = np.zeros((1, 0), dtype=np.float64)
    if n_p > 0 and n_m > 0:
        use_far = not np.isinf(far_ratio)
        _pair_nb(pts, pa, pb, float(sign), has_b, sig, mat, rhs, bool(want_c), bool(want_rhs),
                 float(far_ratio), use_far, eps)
    if want_rhs and flat:
        rhs = rhs.reshape(-1)
    return (mat if want_c else None), (rhs if want_rhs else None), target_dtype, rhs_dtype


def doublet_pair_matrix_numba(
    points: np.ndarray,
    panels_a: np.ndarray,
    panels_b: np.ndarray | None,
    sign: float = 1.0,
    far_ratio: float = 5.0,
) -> np.ndarray:
    """Compute ``C(points, panels_a) + sign * C(points, panels_b)`` in compiled code.

    Same results as :func:`ventorum.panel3d.kernels.doublet_pair_matrix`,
    in compiled code on the CPU. Each target evaluates a panel and its
    mirror (or image) in the same pass; the two separate matrices are
    never stored.

    Parameters
    ----------
    points : numpy.ndarray
        Target coordinates with shape (P, 3) in m.
    panels_a : numpy.ndarray
        Panel corner coordinates with shape (M, 4, 3) in m.
    panels_b : numpy.ndarray or None
        Second panel set with shape (M, 4, 3) in m. None gives
        ``C(points, panels_a)``.
    sign : float, default=1.0
        Factor of the second panel set [-] (+1 or -1).
    far_ratio : float, default=5.0
        Distance-to-diameter ratio beyond which point doublet expansion is used.

    Returns
    -------
    numpy.ndarray
        Doublet potential sum [-] with shape (P, M). Dtype matches the inputs.
    """
    mat, _, target_dtype, _ = _run_pair_numba(
        points, panels_a, panels_b, sign, None, far_ratio, True, False
    )
    return mat.astype(target_dtype, copy=False)


def source_pair_rhs_numba(
    points: np.ndarray,
    panels_a: np.ndarray,
    panels_b: np.ndarray | None,
    sign: float,
    sigma: np.ndarray,
    far_ratio: float = 5.0,
) -> np.ndarray:
    """Compute ``(B(points, panels_a) + sign * B(points, panels_b)) @ sigma`` in compiled code.

    Same results as :func:`ventorum.panel3d.kernels.source_pair_rhs` to
    round-off (the sum over the panels runs in panel order). The source
    matrix is never stored.

    Parameters
    ----------
    points : numpy.ndarray
        Target coordinates with shape (P, 3) in m.
    panels_a : numpy.ndarray
        Panel corner coordinates with shape (M, 4, 3) in m.
    panels_b : numpy.ndarray or None
        Second panel set with shape (M, 4, 3) in m, or None.
    sign : float
        Factor of the second panel set [-] (+1 or -1).
    sigma : numpy.ndarray
        Source strengths with shape (M, K), K at most 6, or (M,) in m/s.
    far_ratio : float, default=5.0
        Distance-to-diameter ratio beyond which point source expansion is used.

    Returns
    -------
    numpy.ndarray
        Source potentials with shape (P, K), or (P,) for a one-dimensional
        *sigma*, in m^2/s.
    """
    _, rhs, _, rhs_dtype = _run_pair_numba(
        points, panels_a, panels_b, sign, sigma, far_ratio, False, True
    )
    return rhs.astype(rhs_dtype, copy=False)


def source_rhs_numba(
    points: np.ndarray,
    panels: np.ndarray,
    sigma: np.ndarray,
    far_ratio: float = 5.0,
) -> np.ndarray:
    """Compute ``B(points, panels) @ sigma`` in compiled code.

    Same results as :func:`ventorum.panel3d.kernels.source_rhs` to
    round-off. The source matrix is never stored.

    Parameters
    ----------
    points : numpy.ndarray
        Target coordinates with shape (P, 3) in m.
    panels : numpy.ndarray
        Panel corner coordinates with shape (M, 4, 3) in m.
    sigma : numpy.ndarray
        Source strengths with shape (M, K), K at most 6, or (M,) in m/s.
    far_ratio : float, default=5.0
        Distance-to-diameter ratio beyond which point source expansion is used.

    Returns
    -------
    numpy.ndarray
        Source potentials with shape (P, K), or (P,) for a one-dimensional
        *sigma*, in m^2/s.
    """
    return source_pair_rhs_numba(points, panels, None, 1.0, sigma, far_ratio=far_ratio)


def pair_system_numba(
    points: np.ndarray,
    panels_a: np.ndarray,
    panels_b: np.ndarray | None,
    sign: float,
    sigma: np.ndarray,
    far_ratio: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the doublet pair matrix and the source pair right-hand sides in one pass.

    Same results as :func:`ventorum.panel3d.kernels.pair_system` to
    round-off. Each target-panel pair computes the source and the doublet
    potential together (they share the solid angle), so one pass gives
    the system matrix and the right-hand sides of a parity system.

    Parameters
    ----------
    points : numpy.ndarray
        Target coordinates with shape (P, 3) in m.
    panels_a : numpy.ndarray
        Panel corner coordinates with shape (M, 4, 3) in m.
    panels_b : numpy.ndarray or None
        Second panel set with shape (M, 4, 3) in m, or None.
    sign : float
        Factor of the second panel set [-] (+1 or -1).
    sigma : numpy.ndarray
        Source strengths with shape (M, K), K at most 6, or (M,) in m/s.
    far_ratio : float, default=5.0
        Distance-to-diameter ratio of the point far-field expansions.

    Returns
    -------
    tuple of numpy.ndarray
        (C, R): C with shape (P, M) [-] is ``C_a + sign * C_b`` and R with
        shape (P, K) in m^2/s is ``(B_a + sign * B_b) @ sigma``.
    """
    mat, rhs, target_dtype, rhs_dtype = _run_pair_numba(
        points, panels_a, panels_b, sign, sigma, far_ratio, True, True
    )
    return mat.astype(target_dtype, copy=False), rhs.astype(rhs_dtype, copy=False)
