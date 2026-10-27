# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Compiled (Numba) vector-Jacobian products of the bent-horseshoe kernel.

The functions here compute the same quantities as the numpy reference
:mod:`ventorum.aero.vortex_vjp` (same kernel, same stable forms, same
topology masks, same floors), in one pass over each (point, source) pair,
without (rows, sources) temporary arrays. The points run in parallel
(``prange``) in blocks of rows; each block adds its source derivatives into
its own buffer, and the buffers are added in block order at the end. The
block partition depends only on the row count, so the result is bitwise
the same for every thread count and thread timing.

Each pair is computed in float64 (scalar CPU code runs at the same speed in
float32 and float64); the caller casts the result to the precision of the
request.

The vortex core acts on the distance from the target to the finite
filament: outside a segment and behind the start point of a ray, the
exact law is multiplied by the Scully factor ``D^2 / (D^2 + rc^2)`` (see
:mod:`ventorum.aero.vortex`).

The derivation of the derivatives is in :mod:`ventorum.aero.vortex_vjp` and
in the theory manual (``docs/theory/gradients.md``). Sources: J. Katz and
A. Plotkin, "Low-Speed Aerodynamics", 2nd ed., Cambridge University Press,
2001 (the kernel); A. Griewank and A. Walther, "Evaluating Derivatives",
2nd ed., SIAM, 2008 (the reverse mode).
"""

from __future__ import annotations

import math

import numpy as np

from ventorum.aero.vortex_vjp import row_blocks

try:
    from numba import njit, prange

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - numba is a dependency
    HAVE_NUMBA = False

_INV_4PI = 1.0 / (4.0 * math.pi)
_INV_2PI = 1.0 / (2.0 * math.pi)
_FLOOR = 1e-300

# Values per source in the block buffers of the horseshoe VJP:
# ga (3), gb (3), ga_te (3), gb_te (3), gd (3), g_rc2_src, gs, the leg
# nodes ga_k (3), gb_k (3), and the ends of the own legs of a chain
# ga_e (3), gb_e (3).
_NSRC = 29
# Values per vortex in the block buffers of the Trefftz VJP: gp (3), ggamma, g_rc2.
_NVTX = 5


def blocks(m: int) -> int:
    """Return the number of row blocks of one call.

    The count depends only on the row count *m* (see
    :func:`ventorum.aero.vortex_vjp.row_blocks`), so the result is bitwise
    the same for every Numba thread count.
    """
    return row_blocks(m)


if HAVE_NUMBA:

    @njit(cache=True, inline="always")
    def _seg_v(px, py, pz, ax, ay, az, bx, by, bz, rc2):
        # Velocity of a unit finite segment a -> b at p. The core acts on the
        # distance to the finite segment (see ventorum.aero.vortex).
        r1x = px - ax
        r1y = py - ay
        r1z = pz - az
        r2x = px - bx
        r2y = py - by
        r2z = pz - bz
        r0x = bx - ax
        r0y = by - ay
        r0z = bz - az
        cx = r1y * r2z - r1z * r2y
        cy = r1z * r2x - r1x * r2z
        cz = r1x * r2y - r1y * r2x
        cr2 = cx * cx + cy * cy + cz * cz
        r0sq = r0x * r0x + r0y * r0y + r0z * r0z
        d1 = r1x * r1x + r1y * r1y + r1z * r1z + _FLOOR
        d2 = r2x * r2x + r2y * r2y + r2z * r2z + _FLOOR
        n1 = math.sqrt(d1)
        n2 = math.sqrt(d2)
        dot12 = r1x * r2x + r1y * r2y + r1z * r2z
        before = r0x * r1x + r0y * r1y + r0z * r1z < 0.0
        after = r0x * r2x + r0y * r2y + r0z * r2z > 0.0
        if before or after:
            # Outside: exact law times D2 / (D2 + rc2), D2 to the nearest end point.
            n12 = n1 * n2
            e = n12 + dot12
            if e > _FLOOR:
                D2 = d1 if before else d2
                k = _INV_4PI * (n1 + n2) / (n12 * e) * (D2 * (1.0 / (D2 + rc2)))
            else:
                k = 0.0
            return cx * k, cy * k, cz * k
        if dot12 > 0.0:
            n12 = n1 * n2
            proj = cr2 * (n1 + n2) / (n12 * (n12 + dot12))
        else:
            proj = (r0x * (r1x / n1 - r2x / n2) + r0y * (r1y / n1 - r2y / n2)
                    + r0z * (r1z / n1 - r2z / n2))
        den = cr2 + rc2 * r0sq
        if den > _FLOOR:
            k = _INV_4PI * proj / den
        else:
            k = 0.0
        return cx * k, cy * k, cz * k

    @njit(cache=True, inline="always")
    def _ray_v(px, py, pz, ax, ay, az, dx, dy, dz, rc2):
        # Velocity of a unit semi-infinite line from a along d at p. The core
        # acts on the distance to the line (see ventorum.aero.vortex).
        rx = px - ax
        ry = py - ay
        rz = pz - az
        cx = dy * rz - dz * ry
        cy = dz * rx - dx * rz
        cz = dx * ry - dy * rx
        cr2 = cx * cx + cy * cy + cz * cz
        D2 = rx * rx + ry * ry + rz * rz + _FLOOR
        rn = math.sqrt(D2)
        dr = dx * rx + dy * ry + dz * rz
        if dr < 0.0:
            # Behind the start point: exact law times D2 / (D2 + rc2).
            e = rn * (rn - dr)
            if e > _FLOOR:
                k = _INV_4PI / e * (D2 * (1.0 / (D2 + rc2)))
            else:
                k = 0.0
            return cx * k, cy * k, cz * k
        den = cr2 + rc2
        if den > _FLOOR:
            k = _INV_4PI * (1.0 + dr / rn) / den
        else:
            k = 0.0
        return cx * k, cy * k, cz * k

    @njit(cache=True, inline="always")
    def _seg_fb(px, py, pz, ax, ay, az, bx, by, bz, rc2, ux, uy, uz):
        # Velocity and VJP of a unit finite segment a -> b at p for the adjoint u.
        # Returns v (3), gP (3), gA (3), gB (3), grc2.
        r1x = px - ax
        r1y = py - ay
        r1z = pz - az
        r2x = px - bx
        r2y = py - by
        r2z = pz - bz
        r0x = bx - ax
        r0y = by - ay
        r0z = bz - az
        cx = r1y * r2z - r1z * r2y
        cy = r1z * r2x - r1x * r2z
        cz = r1x * r2y - r1y * r2x
        cr2 = cx * cx + cy * cy + cz * cz
        r0sq = r0x * r0x + r0y * r0y + r0z * r0z
        d1 = r1x * r1x + r1y * r1y + r1z * r1z + _FLOOR
        d2 = r2x * r2x + r2y * r2y + r2z * r2z + _FLOOR
        n1 = math.sqrt(d1)
        n2 = math.sqrt(d2)
        dot12 = r1x * r2x + r1y * r2y + r1z * r2z
        n12 = n1 * n2
        before = r0x * r1x + r0y * r1y + r0z * r1z < 0.0
        after = r0x * r2x + r0y * r2y + r0z * r2z > 0.0
        if before or after:
            # Outside: k = (n1 + n2) / (4 pi n12 e) F, e = n12 + dot12,
            # F = D2 / (D2 + rc2), D2 = |r1|^2 before a, |r2|^2 after b.
            e = n12 + dot12
            if not e > _FLOOR:
                return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            D2 = d1 if before else d2
            q = 1.0 / (D2 + rc2)
            F = D2 * q
            den_o = n12 * e
            k = _INV_4PI * (n1 + n2) / den_o * F
            vx = cx * k
            vy = cy * k
            vz = cz * k
            gcx = k * ux
            gcy = k * uy
            gcz = k * uz
            gk = ux * cx + uy * cy + uz * cz
            gsum = gk * _INV_4PI * F / den_o
            gG = -gk * k / den_o
            ge = gG * n12
            gn12 = gG * e + ge
            gdot12 = ge
            gF = gk * _INV_4PI * (n1 + n2) / den_o
            # dF/dD2 = rc2 / (D2 + rc2)^2, dF/drc2 = -D2 / (D2 + rc2)^2.
            gD2 = gF * (rc2 * q) * q
            grc2 = -gF * F * q
            f1 = (gsum + gn12 * n2) / n1
            f2 = (gsum + gn12 * n1) / n2
            if before:
                f1 += 2.0 * gD2
            else:
                f2 += 2.0 * gD2
            gr1x = gdot12 * r2x + f1 * r1x
            gr1y = gdot12 * r2y + f1 * r1y
            gr1z = gdot12 * r2z + f1 * r1z
            gr2x = gdot12 * r1x + f2 * r2x
            gr2y = gdot12 * r1y + f2 * r2y
            gr2z = gdot12 * r1z + f2 * r2z
            # c = r1 x r2: g_r1 += r2 x gc, g_r2 += gc x r1.
            gr1x += r2y * gcz - r2z * gcy
            gr1y += r2z * gcx - r2x * gcz
            gr1z += r2x * gcy - r2y * gcx
            gr2x += gcy * r1z - gcz * r1y
            gr2y += gcz * r1x - gcx * r1z
            gr2z += gcx * r1y - gcy * r1x
            return (vx, vy, vz,
                    gr1x + gr2x, gr1y + gr2y, gr1z + gr2z,
                    -gr1x, -gr1y, -gr1z,
                    -gr2x, -gr2y, -gr2z,
                    grc2)
        # Over the segment: the production formula.
        out = dot12 > 0.0
        t1x = r1x / n1
        t1y = r1y / n1
        t1z = r1z / n1
        t2x = r2x / n2
        t2y = r2y / n2
        t2z = r2z / n2
        if out:
            den_out = n12 * (n12 + dot12)
            proj = cr2 * (n1 + n2) / den_out
        else:
            den_out = 1.0
            proj = r0x * (t1x - t2x) + r0y * (t1y - t2y) + r0z * (t1z - t2z)
        den = cr2 + rc2 * r0sq
        if not den > _FLOOR:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        k = _INV_4PI * proj / den
        vx = cx * k
        vy = cy * k
        vz = cz * k
        gcx = k * ux
        gcy = k * uy
        gcz = k * uz
        gk = ux * cx + uy * cy + uz * cz
        gproj = gk * _INV_4PI / den
        gden = -gk * k / den
        gcr2 = gden
        grc2 = gden * r0sq
        gr0sq = gden * rc2
        gr1x = 0.0
        gr1y = 0.0
        gr1z = 0.0
        gr2x = 0.0
        gr2y = 0.0
        gr2z = 0.0
        gr0x = 0.0
        gr0y = 0.0
        gr0z = 0.0
        if out:
            gcr2 += gproj * (n1 + n2) / den_out
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
        gr1x += f1 * r1x
        gr1y += f1 * r1y
        gr1z += f1 * r1z
        gr2x += f2 * r2x
        gr2y += f2 * r2y
        gr2z += f2 * r2z
        gcx += 2.0 * gcr2 * cx
        gcy += 2.0 * gcr2 * cy
        gcz += 2.0 * gcr2 * cz
        # c = r1 x r2: g_r1 += r2 x gc, g_r2 += gc x r1.
        gr1x += r2y * gcz - r2z * gcy
        gr1y += r2z * gcx - r2x * gcz
        gr1z += r2x * gcy - r2y * gcx
        gr2x += gcy * r1z - gcz * r1y
        gr2y += gcz * r1x - gcx * r1z
        gr2z += gcx * r1y - gcy * r1x
        gr0x += 2.0 * gr0sq * r0x
        gr0y += 2.0 * gr0sq * r0y
        gr0z += 2.0 * gr0sq * r0z
        return (vx, vy, vz,
                gr1x + gr2x, gr1y + gr2y, gr1z + gr2z,
                -gr1x - gr0x, -gr1y - gr0y, -gr1z - gr0z,
                gr0x - gr2x, gr0y - gr2y, gr0z - gr2z,
                grc2)

    @njit(cache=True, inline="always")
    def _ray_fb(px, py, pz, ax, ay, az, dx, dy, dz, rc2, ux, uy, uz):
        # Velocity and VJP of a unit semi-infinite line from a along d at p.
        # Returns v (3), gP (3), gd (3), grc2; gA is -gP.
        rx = px - ax
        ry = py - ay
        rz = pz - az
        cx = dy * rz - dz * ry
        cy = dz * rx - dx * rz
        cz = dx * ry - dy * rx
        cr2 = cx * cx + cy * cy + cz * cz
        D2 = rx * rx + ry * ry + rz * rz + _FLOOR
        rn = math.sqrt(D2)
        dr = dx * rx + dy * ry + dz * rz
        if dr < 0.0:
            # Behind: k = F / (4 pi E), E = rn (rn - dr), F = D2 / (D2 + rc2).
            e = rn * (rn - dr)
            if not e > _FLOOR:
                return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            q = 1.0 / (D2 + rc2)
            F = D2 * q
            k = _INV_4PI / e * F
            gcx = k * ux
            gcy = k * uy
            gcz = k * uz
            gk = ux * cx + uy * cy + uz * cz
            gE = -gk * k / e
            grn = gE * (2.0 * rn - dr)
            gdr = -gE * rn
            gF = gk * _INV_4PI / e
            gD2 = gF * (rc2 * q) * q
            grc2 = -gF * F * q
            f = grn / rn + 2.0 * gD2
        else:
            den = cr2 + rc2
            if not den > _FLOOR:
                return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            oc = 1.0 + dr / rn
            k = _INV_4PI * oc / den
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
            f = grn / rn
            gcx += 2.0 * gcr2 * cx
            gcy += 2.0 * gcr2 * cy
            gcz += 2.0 * gcr2 * cz
        vx = cx * k
        vy = cy * k
        vz = cz * k
        grx = f * rx + gdr * dx
        gry = f * ry + gdr * dy
        grz = f * rz + gdr * dz
        gdx = gdr * rx
        gdy = gdr * ry
        gdz = gdr * rz
        # c = d x r: g_d += r x gc, g_r += gc x d.
        gdx += ry * gcz - rz * gcy
        gdy += rz * gcx - rx * gcz
        gdz += rx * gcy - ry * gcx
        grx += gcy * dz - gcz * dy
        gry += gcz * dx - gcx * dz
        grz += gcx * dy - gcy * dx
        return vx, vy, vz, grx, gry, grz, gdx, gdy, gdz, grc2

    @njit(cache=True, inline="always")
    def _pair_rc2(i, j, rc2s, tgrc2, sgrp, tgrp, use_tg):
        if use_tg and tgrp[i] != sgrp[j]:
            return rc2s[j] + tgrc2[i], True
        return rc2s[j], False

    @njit(cache=True, inline="always")
    def _hs_v(i, j, P, a, b, ate, bte, d, rc2, own, spanel, edge, sea, seb, ak, bk, kfl, seka, sekb, ae, be):
        # Unit-circulation velocity of horseshoe j at point i (masked pairs zero).
        # A leg with a node (kfl[j] != 0) runs ae -> ak -> a and b -> bk -> be;
        # on the other sources ak is ate and bk is bte. ae, be are the
        # trailing-edge points, or on a leg chain the bound points of the
        # next source. The last three values are the own leg segments alone.
        px = P[i, 0]
        py = P[i, 1]
        pz = P[i, 2]
        vbx = 0.0
        vby = 0.0
        vbz = 0.0
        if own[i] != spanel[j]:
            vbx, vby, vbz = _seg_v(px, py, pz, a[j, 0], a[j, 1], a[j, 2], b[j, 0], b[j, 1], b[j, 2], rc2)
        vax = 0.0
        vay = 0.0
        vaz = 0.0
        if edge[i] != sea[j]:
            vax, vay, vaz = _seg_v(px, py, pz, ak[j, 0], ak[j, 1], ak[j, 2], a[j, 0], a[j, 1], a[j, 2], rc2)
        if kfl[j] != 0 and edge[i] != seka[j]:
            kx, ky, kz = _seg_v(px, py, pz, ae[j, 0], ae[j, 1], ae[j, 2], ak[j, 0], ak[j, 1], ak[j, 2], rc2)
            vax += kx
            vay += ky
            vaz += kz
        vlx = 0.0
        vly = 0.0
        vlz = 0.0
        if edge[i] != seb[j]:
            vlx, vly, vlz = _seg_v(px, py, pz, b[j, 0], b[j, 1], b[j, 2], bk[j, 0], bk[j, 1], bk[j, 2], rc2)
        if kfl[j] != 0 and edge[i] != sekb[j]:
            kx, ky, kz = _seg_v(px, py, pz, bk[j, 0], bk[j, 1], bk[j, 2], be[j, 0], be[j, 1], be[j, 2], rc2)
            vlx += kx
            vly += ky
            vlz += kz
        wbx, wby, wbz = _ray_v(px, py, pz, bte[j, 0], bte[j, 1], bte[j, 2], d[j, 0], d[j, 1], d[j, 2], rc2)
        wax, way, waz = _ray_v(px, py, pz, ate[j, 0], ate[j, 1], ate[j, 2], d[j, 0], d[j, 1], d[j, 2], rc2)
        return ((vbx + vax + vlx) + wbx - wax, (vby + vay + vly) + wby - way,
                (vbz + vaz + vlz) + wbz - waz, vax + vlx, vay + vly, vaz + vlz)

    @njit(cache=True, parallel=True)
    def horseshoe_forward_chain_kernel(mode, P, X, a, b, ate, bte, d, rc2s, tgrc2, sgrp, tgrp, own, spanel,
                                       edge, sea, seb, s, col, n_u, use_tg, ak, bk, kfl, seka, sekb, ae, be, nxt,
                                       chain):
        """Return the system rows (mode 0, (m, n_u)) or the induced velocity (mode 1, (m, 3)).

        With *chain* the sources of each row run from the last to the first:
        a source adds the legs of the next source of its chain (``nxt``).
        """
        m = P.shape[0]
        n = a.shape[0]
        if mode == 0:
            out = np.zeros((m, n_u))
        else:
            out = np.zeros((m, 3))
        for i in prange(m):
            ox = 0.0
            oy = 0.0
            oz = 0.0
            cx = 0.0
            cy = 0.0
            cz = 0.0
            for jj in range(n):
                j = n - 1 - jj if chain else jj
                rc2, _ = _pair_rc2(i, j, rc2s, tgrc2, sgrp, tgrp, use_tg)
                vx, vy, vz, lx, ly, lz = _hs_v(i, j, P, a, b, ate, bte, d, rc2, own, spanel, edge, sea, seb,
                                               ak, bk, kfl, seka, sekb, ae, be)
                if chain:
                    if nxt[j] != 0:
                        vx += cx
                        vy += cy
                        vz += cz
                        lx += cx
                        ly += cy
                        lz += cz
                    cx = lx
                    cy = ly
                    cz = lz
                if mode == 0:
                    out[i, col[j]] += s[j] * (vx * X[i, 0] + vy * X[i, 1] + vz * X[i, 2])
                else:
                    ox += s[j] * vx
                    oy += s[j] * vy
                    oz += s[j] * vz
            if mode != 0:
                out[i, 0] = ox
                out[i, 1] = oy
                out[i, 2] = oz
        return out

    @njit(cache=True, parallel=True)
    def horseshoe_vjp_chain_kernel(P, X, a, b, ate, bte, d, rc2s, tgrc2, sgrp, tgrp, own, spanel, edge, sea, seb,
                                   s, col, W, use_W, use_tg, nblocks, ak, bk, kfl, seka, sekb, ae, be, nxt, chain):
        """Return the VJP (gP, gX, g_tg_rc2, ga, gb, ga_te, gb_te, gd, g_rc2_src, gs, ga_k, gb_k, ga_e, gb_e).

        ga_k and gb_k are zero on a source without a leg node (kfl[j] == 0):
        its leg derivative is in ga_te and gb_te. With *chain* the aft
        segments end at ``ae``, ``be`` (their derivatives are ga_e, gb_e;
        zero without chains), and the own leg segments of a source take the
        adjoint of every source before them on their chain (``nxt``).
        """
        m = P.shape[0]
        n = a.shape[0]
        gP = np.zeros((m, 3))
        gX = np.zeros((m, 3))
        gtg = np.zeros(m)
        buf = np.zeros((nblocks, n, _NSRC))
        # Slots of the aft-segment ends: a_te and b_te, or the chain ends.
        sa = 23 if chain else 6
        sb = 26 if chain else 9
        for t in prange(nblocks):
            i0 = (t * m) // nblocks
            i1 = ((t + 1) * m) // nblocks
            # X . (own legs) of each source of the row (chains only).
            xo = np.zeros(n if chain else 1)
            for i in range(i0, i1):
                cw = 0.0
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
                    rc2, oth = _pair_rc2(i, j, rc2s, tgrc2, sgrp, tgrp, use_tg)
                    ux = cij * xx
                    uy = cij * xy
                    uz = cij * xz
                    # Adjoint of the own leg segments: the sum over the chain.
                    if chain and j > 0 and nxt[j - 1] != 0:
                        cw = cij + cw
                    else:
                        cw = cij
                    lux = cw * xx
                    luy = cw * xy
                    luz = cw * xz
                    ox = 0.0
                    oy = 0.0
                    oz = 0.0
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
                    bu = buf[t, j]
                    vx = 0.0
                    vy = 0.0
                    vz = 0.0
                    grc = 0.0
                    if own[i] != spanel[j]:
                        r = _seg_fb(px, py, pz, ax, ay, az, bx, by, bz, rc2, ux, uy, uz)
                        vx += r[0]
                        vy += r[1]
                        vz += r[2]
                        aPx += r[3]
                        aPy += r[4]
                        aPz += r[5]
                        bu[0] += r[6]
                        bu[1] += r[7]
                        bu[2] += r[8]
                        bu[3] += r[9]
                        bu[4] += r[10]
                        bu[5] += r[11]
                        grc += r[12]
                    kj = kfl[j] != 0
                    # Start of the forward a-leg segment: the node, or a_te.
                    ka = 17 if kj else 6
                    # End of the forward b-leg segment: the node, or b_te.
                    kb = 20 if kj else 9
                    if edge[i] != sea[j]:
                        r = _seg_fb(px, py, pz, ak[j, 0], ak[j, 1], ak[j, 2], ax, ay, az, rc2, lux, luy, luz)
                        if chain:
                            ox += r[0]
                            oy += r[1]
                            oz += r[2]
                        else:
                            vx += r[0]
                            vy += r[1]
                            vz += r[2]
                        aPx += r[3]
                        aPy += r[4]
                        aPz += r[5]
                        bu[ka] += r[6]
                        bu[ka + 1] += r[7]
                        bu[ka + 2] += r[8]
                        bu[0] += r[9]
                        bu[1] += r[10]
                        bu[2] += r[11]
                        grc += r[12]
                    if kj and edge[i] != seka[j]:
                        r = _seg_fb(px, py, pz, ae[j, 0], ae[j, 1], ae[j, 2], ak[j, 0], ak[j, 1], ak[j, 2], rc2,
                                    lux, luy, luz)
                        if chain:
                            ox += r[0]
                            oy += r[1]
                            oz += r[2]
                        else:
                            vx += r[0]
                            vy += r[1]
                            vz += r[2]
                        aPx += r[3]
                        aPy += r[4]
                        aPz += r[5]
                        bu[sa] += r[6]
                        bu[sa + 1] += r[7]
                        bu[sa + 2] += r[8]
                        bu[17] += r[9]
                        bu[18] += r[10]
                        bu[19] += r[11]
                        grc += r[12]
                    if edge[i] != seb[j]:
                        r = _seg_fb(px, py, pz, bx, by, bz, bk[j, 0], bk[j, 1], bk[j, 2], rc2, lux, luy, luz)
                        if chain:
                            ox += r[0]
                            oy += r[1]
                            oz += r[2]
                        else:
                            vx += r[0]
                            vy += r[1]
                            vz += r[2]
                        aPx += r[3]
                        aPy += r[4]
                        aPz += r[5]
                        bu[3] += r[6]
                        bu[4] += r[7]
                        bu[5] += r[8]
                        bu[kb] += r[9]
                        bu[kb + 1] += r[10]
                        bu[kb + 2] += r[11]
                        grc += r[12]
                    if kj and edge[i] != sekb[j]:
                        r = _seg_fb(px, py, pz, bk[j, 0], bk[j, 1], bk[j, 2], be[j, 0], be[j, 1], be[j, 2], rc2,
                                    lux, luy, luz)
                        if chain:
                            ox += r[0]
                            oy += r[1]
                            oz += r[2]
                        else:
                            vx += r[0]
                            vy += r[1]
                            vz += r[2]
                        aPx += r[3]
                        aPy += r[4]
                        aPz += r[5]
                        bu[20] += r[6]
                        bu[21] += r[7]
                        bu[22] += r[8]
                        bu[sb] += r[9]
                        bu[sb + 1] += r[10]
                        bu[sb + 2] += r[11]
                        grc += r[12]
                    q = _ray_fb(px, py, pz, tbx, tby, tbz, dx, dy, dz, rc2, ux, uy, uz)
                    vx += q[0]
                    vy += q[1]
                    vz += q[2]
                    aPx += q[3]
                    aPy += q[4]
                    aPz += q[5]
                    bu[9] -= q[3]
                    bu[10] -= q[4]
                    bu[11] -= q[5]
                    bu[12] += q[6]
                    bu[13] += q[7]
                    bu[14] += q[8]
                    grc += q[9]
                    q = _ray_fb(px, py, pz, tax, tay, taz, dx, dy, dz, rc2, -ux, -uy, -uz)
                    vx -= q[0]
                    vy -= q[1]
                    vz -= q[2]
                    aPx += q[3]
                    aPy += q[4]
                    aPz += q[5]
                    bu[6] -= q[3]
                    bu[7] -= q[4]
                    bu[8] -= q[5]
                    bu[12] += q[6]
                    bu[13] += q[7]
                    bu[14] += q[8]
                    grc += q[9]
                    # The legs of the sources after j on its chain are in the
                    # chain sum cw of their own segments (gX) and in the
                    # suffix pass below (gs).
                    # Without chains ox is zero and the own legs are in v.
                    aXx += cij * vx + cw * ox
                    aXy += cij * vy + cw * oy
                    aXz += cij * vz + cw * oz
                    bu[15] += grc
                    bu[16] += wf * (xx * (vx + ox) + xy * (vy + oy) + xz * (vz + oz))
                    if chain:
                        xo[j] = xx * ox + xy * oy + xz * oz
                    if oth:
                        atg += grc
                if chain:
                    # gs: the own legs of the sources after each source on its chain.
                    tail = 0.0
                    for jj in range(n):
                        j = n - 1 - jj
                        if nxt[j] != 0:
                            tail = xo[j + 1] + tail
                            wf = W[i, col[j]] if use_W else 1.0
                            buf[t, j, 16] += wf * tail
                        else:
                            tail = 0.0
                gP[i, 0] = aPx
                gP[i, 1] = aPy
                gP[i, 2] = aPz
                gX[i, 0] = aXx
                gX[i, 1] = aXy
                gX[i, 2] = aXz
                gtg[i] = atg
        acc = np.zeros((n, _NSRC))
        for t in range(nblocks):
            for j in range(n):
                for k in range(_NSRC):
                    acc[j, k] += buf[t, j, k]
        return (gP, gX, gtg, acc[:, 0:3].copy(), acc[:, 3:6].copy(), acc[:, 6:9].copy(),
                acc[:, 9:12].copy(), acc[:, 12:15].copy(), acc[:, 15].copy(), acc[:, 16].copy(),
                acc[:, 17:20].copy(), acc[:, 20:23].copy(), acc[:, 23:26].copy(), acc[:, 26:29].copy())

    _NO_CHAIN = np.zeros(1, dtype=np.int8)

    def horseshoe_forward_kink_kernel(mode, P, X, a, b, ate, bte, d, rc2s, tgrc2, sgrp, tgrp, own, spanel,
                                      edge, sea, seb, s, col, n_u, use_tg, ak, bk, kfl, seka, sekb,
                                      ae=None, be=None, nxt=None):
        """Return the system rows (mode 0) or the induced velocity (mode 1) of sources with leg nodes.

        *ae*, *be*, *nxt* are the leg chains (see
        :func:`horseshoe_forward_chain_kernel`), or None.
        """
        if nxt is None:
            return horseshoe_forward_chain_kernel(mode, P, X, a, b, ate, bte, d, rc2s, tgrc2, sgrp, tgrp, own,
                                                  spanel, edge, sea, seb, s, col, n_u, use_tg, ak, bk, kfl, seka,
                                                  sekb, ate, bte, _NO_CHAIN, False)
        return horseshoe_forward_chain_kernel(mode, P, X, a, b, ate, bte, d, rc2s, tgrc2, sgrp, tgrp, own, spanel,
                                              edge, sea, seb, s, col, n_u, use_tg, ak, bk, kfl, seka, sekb,
                                              ae, be, nxt, True)

    def horseshoe_vjp_kink_kernel(P, X, a, b, ate, bte, d, rc2s, tgrc2, sgrp, tgrp, own, spanel, edge, sea, seb,
                                  s, col, W, use_W, use_tg, nblocks, ak, bk, kfl, seka, sekb,
                                  ae=None, be=None, nxt=None):
        """Return the VJP of sources with leg nodes: 12 items, or 14 with the leg chains *ae*, *be*, *nxt*."""
        if nxt is None:
            return horseshoe_vjp_chain_kernel(P, X, a, b, ate, bte, d, rc2s, tgrc2, sgrp, tgrp, own, spanel, edge,
                                              sea, seb, s, col, W, use_W, use_tg, nblocks, ak, bk, kfl, seka, sekb,
                                              ate, bte, _NO_CHAIN, False)[:12]
        return horseshoe_vjp_chain_kernel(P, X, a, b, ate, bte, d, rc2s, tgrc2, sgrp, tgrp, own, spanel, edge,
                                          sea, seb, s, col, W, use_W, use_tg, nblocks, ak, bk, kfl, seka, sekb,
                                          ae, be, nxt, True)

    def horseshoe_forward_kernel(mode, P, X, a, b, ate, bte, d, rc2s, tgrc2, sgrp, tgrp, own, spanel,
                                 edge, sea, seb, s, col, n_u, use_tg):
        """Return the product of :func:`horseshoe_forward_kink_kernel` for sources without leg nodes."""
        return horseshoe_forward_kink_kernel(mode, P, X, a, b, ate, bte, d, rc2s, tgrc2, sgrp, tgrp, own, spanel,
                                             edge, sea, seb, s, col, n_u, use_tg, ate, bte,
                                             np.zeros(a.shape[0], dtype=np.int8), sea, seb)

    def horseshoe_vjp_kernel(P, X, a, b, ate, bte, d, rc2s, tgrc2, sgrp, tgrp, own, spanel, edge, sea, seb,
                             s, col, W, use_W, use_tg, nblocks):
        """Return the first ten items of :func:`horseshoe_vjp_kink_kernel` for sources without leg nodes."""
        return horseshoe_vjp_kink_kernel(P, X, a, b, ate, bte, d, rc2s, tgrc2, sgrp, tgrp, own, spanel, edge,
                                         sea, seb, s, col, W, use_W, use_tg, nblocks, ate, bte,
                                         np.zeros(a.shape[0], dtype=np.int8), sea, seb)[:10]

    @njit(cache=True, parallel=True)
    def trefftz_forward_kernel(q, qn, p, gamma, d, rc2, tgrc2, sgrp, tgrp, use_tg):
        """Return the normal wash (m,) of 2-D point vortices at Trefftz-plane points."""
        m = q.shape[0]
        n = p.shape[0]
        out = np.zeros(m)
        dx = d[0]
        dy = d[1]
        dz = d[2]
        for i in prange(m):
            acc = 0.0
            for j in range(n):
                c2, _ = _pair_rc2(i, j, rc2, tgrc2, sgrp, tgrp, use_tg)
                rx = q[i, 0] - p[j, 0]
                ry = q[i, 1] - p[j, 1]
                rz = q[i, 2] - p[j, 2]
                vx = dy * rz - dz * ry
                vy = dz * rx - dx * rz
                vz = dx * ry - dy * rx
                k = _INV_2PI * gamma[j] / (rx * rx + ry * ry + rz * rz + c2)
                acc += (vx * qn[i, 0] + vy * qn[i, 1] + vz * qn[i, 2]) * k
            out[i] = acc
        return out

    @njit(cache=True, parallel=True)
    def trefftz_vjp_kernel(q, qn, p, gamma, d, rc2, tgrc2, sgrp, tgrp, use_tg, gw, nblocks):
        """Return the VJP (gq, gqn, g_tg_rc2, gp, ggamma, g_rc2, gd) of the Trefftz normal wash."""
        m = q.shape[0]
        n = p.shape[0]
        gq = np.zeros((m, 3))
        gqn = np.zeros((m, 3))
        gtg = np.zeros(m)
        buf = np.zeros((nblocks, n, _NVTX))
        dbuf = np.zeros((nblocks, 3))
        dx = d[0]
        dy = d[1]
        dz = d[2]
        for t in prange(nblocks):
            i0 = (t * m) // nblocks
            i1 = ((t + 1) * m) // nblocks
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
                    c2, oth = _pair_rc2(i, j, rc2, tgrc2, sgrp, tgrp, use_tg)
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
                    aqx += grx
                    aqy += gry
                    aqz += grz
                    anx += ge * vx
                    any_ += ge * vy
                    anz += ge * vz
                    if oth:
                        atg += gD
                    bu = buf[t, j]
                    bu[0] -= grx
                    bu[1] -= gry
                    bu[2] -= grz
                    bu[3] += gk * _INV_2PI / D
                    bu[4] += gD
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
        acc = np.zeros((n, _NVTX))
        gd = np.zeros(3)
        for t in range(nblocks):
            for j in range(n):
                for k in range(_NVTX):
                    acc[j, k] += buf[t, j, k]
            for k in range(3):
                gd[k] += dbuf[t, k]
        return gq, gqn, gtg, acc[:, 0:3].copy(), acc[:, 3].copy(), acc[:, 4].copy(), gd
