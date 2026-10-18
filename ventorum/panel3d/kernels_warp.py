# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Warp (CUDA) versions of the 3D panel influence kernels of :mod:`ventorum.panel3d.kernels`.

The kernels compute the same matrices as the numpy reference: the source
potentials ``B`` and the doublet potentials ``C``, with the same formulas
and the same branches (planar projection done on the host, the van
Oosterom and Strackee solid angle, the body-side limit in the on-plane
band, the analytic edge sums, and the point far-field expansions beyond
``far_ratio`` panel diameters).

The on-plane band (the body-side limit of the doublet potential) uses a
tolerance relative to the panel size and the coordinate magnitude, with the
machine epsilon of the coarser of the input dtype and the kernel precision.
So a float32 run keeps the self term +1/2 at the collocation points.

The pair kernels evaluate a panel and its mirror (or image) for the same
target: :func:`doublet_pair_matrix_warp` returns ``C_a + sign * C_b`` and
:func:`source_pair_rhs_warp` returns ``(B_a + sign * B_b) @ sigma``
without the source matrix. The source sums use a fixed chunk of panels
per thread and a plain reduction over the chunks (no atomic additions),
so the result does not change from run to run.

Thread map: one thread computes one (target, panel) pair, so the launch
has ``(P, M)`` threads. Each precision (float32, float64) builds its own
kernels once per process. Warp and PyTorch are imported lazily inside the
functions, so importing this module never loads them. The outputs are
PyTorch tensors on the CUDA device (``wp.to_torch``).

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
import threading

import numpy as np

from ventorum.panel3d.kernels import MAX_RHS_COLUMNS

__all__ = [
    "influence_matrices_warp",
    "doublet_pair_matrix_warp",
    "source_rhs_warp",
    "source_pair_rhs_warp",
    "pair_system_warp",
]

_INV_4PI = 1.0 / (4.0 * math.pi)

#: Device bytes of one block of rows of the doublet pair kernel when the
#: output dtype differs from the kernel precision.
BLOCK_BYTES = 64 * 2**20

#: Panels summed by one GPU thread of the source pair kernel [-].
RHS_CHUNK = 128

_lock = threading.Lock()
_built: dict[str, object] = {}
_initialised = False


def _kernels(precision: str):
    """Build and cache the Warp kernels of one precision.

    Float32 arithmetic is refused outside the developer path
    :func:`ventorum.panel3d.kernels.unsafe_float32_kernels` (the float32
    on-plane band zeroes real influences near long panels).
    """
    import warp as wp

    from ventorum.panel3d.kernels import _require_float64

    if precision == "float32":
        _require_float64("float32 arithmetic of the Warp panel kernels")

    global _initialised
    with _lock:
        found = _built.get(precision)
        if found is not None:
            return found
        if precision not in ("float32", "float64"):
            raise ValueError(f"Unknown precision {precision!r}; use 'float32' or 'float64'.")
        if not _initialised:
            wp.init()
            _initialised = True
        T = wp.float64 if precision == "float64" else wp.float32
        floor = 1e-300 if precision == "float64" else 1e-30
        module = f"ventorum_panel3d_{precision}"
        wp.set_module_options({"enable_backward": False}, wp.get_module(module))

        @wp.func(module=module)
        def tri_omega(
            x: T,
            y: T,
            z: T,
            x0: T,
            y0: T,
            x1: T,
            y1: T,
            x2: T,
            y2: T,
            r0: T,
            r1: T,
            r2: T,
        ):
            area2 = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
            num = T(0.0) - area2 * z
            d01 = (x0 - x) * (x1 - x) + (y0 - y) * (y1 - y) + z * z
            d02 = (x0 - x) * (x2 - x) + (y0 - y) * (y2 - y) + z * z
            d12 = (x1 - x) * (x2 - x) + (y1 - y) * (y2 - y) + z * z
            den = r0 * r1 * r2 + d01 * r2 + d02 * r1 + d12 * r0
            return T(2.0) * wp.atan2(num, den)

        @wp.func(module=module)
        def plane_winding(
            x: T,
            y: T,
            j: int,
            xloc: wp.array2d(dtype=T),
            yloc: wp.array2d(dtype=T),
            elen: wp.array2d(dtype=T),
            act: wp.array2d(dtype=int),
            tol: T,
        ):
            # Body-side limit in the on-plane band: in-plane angle of the panel.
            total = T(0.0)
            for k in range(4):
                if act[j, k] != 0:
                    kn = k + 1
                    if kn == 4:
                        kn = 0
                    ax = xloc[j, k] - x
                    ay = yloc[j, k] - y
                    bx = xloc[j, kn] - x
                    by = yloc[j, kn] - y
                    cr = ax * by - ay * bx
                    dt = ax * bx + ay * by
                    ra = wp.sqrt(ax * ax + ay * ay)
                    rb = wp.sqrt(bx * bx + by * by)
                    on_edge = False
                    if wp.abs(cr) <= tol * elen[j, k] and dt <= T(0.0):
                        on_edge = True
                    if ra <= tol or rb <= tol:
                        on_edge = True
                    if not on_edge:
                        total = total + wp.atan2(cr, dt)
            return total

        @wp.func(module=module)
        def solid_angle(
            x: T,
            y: T,
            z: T,
            j: int,
            xloc: wp.array2d(dtype=T),
            yloc: wp.array2d(dtype=T),
            elen: wp.array2d(dtype=T),
            act: wp.array2d(dtype=int),
            tol: T,
        ):
            om = T(0.0)
            if wp.abs(z) <= tol:
                om = plane_winding(x, y, j, xloc, yloc, elen, act, tol)
            else:
                r0 = wp.sqrt((x - xloc[j, 0]) * (x - xloc[j, 0]) + (y - yloc[j, 0]) * (y - yloc[j, 0]) + z * z)
                r1 = wp.sqrt((x - xloc[j, 1]) * (x - xloc[j, 1]) + (y - yloc[j, 1]) * (y - yloc[j, 1]) + z * z)
                r2 = wp.sqrt((x - xloc[j, 2]) * (x - xloc[j, 2]) + (y - yloc[j, 2]) * (y - yloc[j, 2]) + z * z)
                r3 = wp.sqrt((x - xloc[j, 3]) * (x - xloc[j, 3]) + (y - yloc[j, 3]) * (y - yloc[j, 3]) + z * z)
                om = tri_omega(x, y, z, xloc[j, 0], yloc[j, 0], xloc[j, 1], yloc[j, 1],
                               xloc[j, 2], yloc[j, 2], r0, r1, r2)
                if act[j, 3] != 0:
                    om = om + tri_omega(x, y, z, xloc[j, 0], yloc[j, 0], xloc[j, 2], yloc[j, 2],
                                        xloc[j, 3], yloc[j, 3], r0, r2, r3)
            return om

        @wp.func(module=module)
        def edge_log(x: T, y: T, z: T, xa: T, ya: T, xb: T, yb: T, dk: T):
            # Edge logarithm ln((ra + rb + dk) / (ra + rb - dk)) without cancellation.
            ax = xa - x
            ay = ya - y
            bx = xb - x
            by = yb - y
            z2 = z * z
            ra = wp.sqrt(ax * ax + ay * ay + z2)
            rb = wp.sqrt(bx * bx + by * by + z2)
            dot = ax * bx + ay * by + z2
            rr = ra * rb
            s = T(0.0)
            if dot >= T(0.0):
                s = rr + dot
            else:
                cr = ax * by - ay * bx
                dd = rr - dot
                if dd < T(floor):
                    dd = T(floor)
                s = (cr * cr + z2 * dk * dk) / dd
            lk = T(0.0)
            if s > T(0.0):
                lk = T(2.0) * wp.log(ra + rb + dk) - wp.log(T(2.0) * s)
            return lk

        @wp.kernel(module=module)
        def bc_kernel(
            P: wp.array2d(dtype=T),
            cen: wp.array2d(dtype=T),
            nor: wp.array2d(dtype=T),
            lax: wp.array2d(dtype=T),
            maxis: wp.array2d(dtype=T),
            xloc: wp.array2d(dtype=T),
            yloc: wp.array2d(dtype=T),
            dxe: wp.array2d(dtype=T),
            dye: wp.array2d(dtype=T),
            elen: wp.array2d(dtype=T),
            act: wp.array2d(dtype=int),
            area: wp.array(dtype=T),
            diam: wp.array(dtype=T),
            ptol: wp.array(dtype=T),
            far_ratio: T,
            use_far: int,
            B: wp.array2d(dtype=T),
            C: wp.array2d(dtype=T),
        ):
            i, j = wp.tid()
            rx = P[i, 0] - cen[j, 0]
            ry = P[i, 1] - cen[j, 1]
            rz = P[i, 2] - cen[j, 2]
            r = wp.sqrt(rx * rx + ry * ry + rz * rz)
            is_far = False
            if use_far != 0:
                if r > far_ratio * diam[j]:
                    is_far = True
            if is_far:
                rs = r
                if rs < T(floor):
                    rs = T(floor)
                b = T(0.0) - T(_INV_4PI) * area[j] / rs
                rdotn = rx * nor[j, 0] + ry * nor[j, 1] + rz * nor[j, 2]
                r3 = rs * rs * rs
                if r3 < T(floor):
                    r3 = T(floor)
                c = T(0.0) - T(_INV_4PI) * area[j] * rdotn / r3
            else:
                x = rx * lax[j, 0] + ry * lax[j, 1] + rz * lax[j, 2]
                y = rx * maxis[j, 0] + ry * maxis[j, 1] + rz * maxis[j, 2]
                z = rx * nor[j, 0] + ry * nor[j, 1] + rz * nor[j, 2]
                om = solid_angle(x, y, z, j, xloc, yloc, elen, act, ptol[j])
                s = T(0.0)
                for k in range(4):
                    if act[j, k] != 0:
                        kn = k + 1
                        if kn == 4:
                            kn = 0
                        dk = elen[j, k]
                        dk_safe = dk
                        if dk_safe < T(floor):
                            dk_safe = T(floor)
                        lk = edge_log(x, y, z, xloc[j, k], yloc[j, k], xloc[j, kn], yloc[j, kn], dk)
                        rperp = ((x - xloc[j, k]) * dye[j, k] - (y - yloc[j, k]) * dxe[j, k]) / dk_safe
                        s = s + rperp * lk
                b = T(_INV_4PI) * (s - z * om)
                c = T(_INV_4PI) * om
            B[i, j] = b
            C[i, j] = c

        @wp.func(module=module)
        def pair_c(
            px: T,
            py: T,
            pz: T,
            j: int,
            cen: wp.array2d(dtype=T),
            nor: wp.array2d(dtype=T),
            lax: wp.array2d(dtype=T),
            maxis: wp.array2d(dtype=T),
            xloc: wp.array2d(dtype=T),
            yloc: wp.array2d(dtype=T),
            elen: wp.array2d(dtype=T),
            act: wp.array2d(dtype=int),
            area: wp.array(dtype=T),
            diam: wp.array(dtype=T),
            ptol: wp.array(dtype=T),
            far_ratio: T,
            use_far: int,
        ):
            # Doublet potential of one target-panel pair (the C part of bc_kernel).
            rx = px - cen[j, 0]
            ry = py - cen[j, 1]
            rz = pz - cen[j, 2]
            r = wp.sqrt(rx * rx + ry * ry + rz * rz)
            is_far = False
            if use_far != 0:
                if r > far_ratio * diam[j]:
                    is_far = True
            c = T(0.0)
            if is_far:
                rs = r
                if rs < T(floor):
                    rs = T(floor)
                rdotn = rx * nor[j, 0] + ry * nor[j, 1] + rz * nor[j, 2]
                r3 = rs * rs * rs
                if r3 < T(floor):
                    r3 = T(floor)
                c = T(0.0) - T(_INV_4PI) * area[j] * rdotn / r3
            else:
                x = rx * lax[j, 0] + ry * lax[j, 1] + rz * lax[j, 2]
                y = rx * maxis[j, 0] + ry * maxis[j, 1] + rz * maxis[j, 2]
                z = rx * nor[j, 0] + ry * nor[j, 1] + rz * nor[j, 2]
                om = solid_angle(x, y, z, j, xloc, yloc, elen, act, ptol[j])
                c = T(_INV_4PI) * om
            return c

        @wp.func(module=module)
        def pair_b(
            px: T,
            py: T,
            pz: T,
            j: int,
            cen: wp.array2d(dtype=T),
            nor: wp.array2d(dtype=T),
            lax: wp.array2d(dtype=T),
            maxis: wp.array2d(dtype=T),
            xloc: wp.array2d(dtype=T),
            yloc: wp.array2d(dtype=T),
            dxe: wp.array2d(dtype=T),
            dye: wp.array2d(dtype=T),
            elen: wp.array2d(dtype=T),
            act: wp.array2d(dtype=int),
            area: wp.array(dtype=T),
            diam: wp.array(dtype=T),
            ptol: wp.array(dtype=T),
            far_ratio: T,
            use_far: int,
        ):
            # Source potential of one target-panel pair (the B part of bc_kernel).
            rx = px - cen[j, 0]
            ry = py - cen[j, 1]
            rz = pz - cen[j, 2]
            r = wp.sqrt(rx * rx + ry * ry + rz * rz)
            is_far = False
            if use_far != 0:
                if r > far_ratio * diam[j]:
                    is_far = True
            b = T(0.0)
            if is_far:
                rs = r
                if rs < T(floor):
                    rs = T(floor)
                b = T(0.0) - T(_INV_4PI) * area[j] / rs
            else:
                x = rx * lax[j, 0] + ry * lax[j, 1] + rz * lax[j, 2]
                y = rx * maxis[j, 0] + ry * maxis[j, 1] + rz * maxis[j, 2]
                z = rx * nor[j, 0] + ry * nor[j, 1] + rz * nor[j, 2]
                om = solid_angle(x, y, z, j, xloc, yloc, elen, act, ptol[j])
                s = T(0.0)
                for k in range(4):
                    if act[j, k] != 0:
                        kn = k + 1
                        if kn == 4:
                            kn = 0
                        dk = elen[j, k]
                        dk_safe = dk
                        if dk_safe < T(floor):
                            dk_safe = T(floor)
                        lk = edge_log(x, y, z, xloc[j, k], yloc[j, k], xloc[j, kn], yloc[j, kn], dk)
                        rperp = ((x - xloc[j, k]) * dye[j, k] - (y - yloc[j, k]) * dxe[j, k]) / dk_safe
                        s = s + rperp * lk
                b = T(_INV_4PI) * (s - z * om)
            return b

        @wp.kernel(module=module)
        def doublet_pair_kernel(
            P: wp.array2d(dtype=T),
            row0: int,
            cen_a: wp.array2d(dtype=T),
            nor_a: wp.array2d(dtype=T),
            lax_a: wp.array2d(dtype=T),
            max_a: wp.array2d(dtype=T),
            xl_a: wp.array2d(dtype=T),
            yl_a: wp.array2d(dtype=T),
            el_a: wp.array2d(dtype=T),
            act_a: wp.array2d(dtype=int),
            ar_a: wp.array(dtype=T),
            di_a: wp.array(dtype=T),
            tol_a: wp.array(dtype=T),
            cen_b: wp.array2d(dtype=T),
            nor_b: wp.array2d(dtype=T),
            lax_b: wp.array2d(dtype=T),
            max_b: wp.array2d(dtype=T),
            xl_b: wp.array2d(dtype=T),
            yl_b: wp.array2d(dtype=T),
            el_b: wp.array2d(dtype=T),
            act_b: wp.array2d(dtype=int),
            ar_b: wp.array(dtype=T),
            di_b: wp.array(dtype=T),
            tol_b: wp.array(dtype=T),
            has_b: int,
            sign: T,
            far_ratio: T,
            use_far: int,
            C: wp.array2d(dtype=T),
        ):
            # One thread per (target row of the block, panel) pair.
            i, j = wp.tid()
            t = row0 + i
            px = P[t, 0]
            py = P[t, 1]
            pz = P[t, 2]
            c = pair_c(px, py, pz, j, cen_a, nor_a, lax_a, max_a, xl_a, yl_a, el_a, act_a, ar_a, di_a,
                       tol_a, far_ratio, use_far)
            c2 = T(0.0)
            if has_b != 0:
                c2 = pair_c(px, py, pz, j, cen_b, nor_b, lax_b, max_b, xl_b, yl_b, el_b, act_b, ar_b,
                            di_b, tol_b, far_ratio, use_far)
                c = c + sign * c2
            C[i, j] = c

        @wp.kernel(module=module)
        def source_pair_kernel(
            P: wp.array2d(dtype=T),
            cen_a: wp.array2d(dtype=T),
            nor_a: wp.array2d(dtype=T),
            lax_a: wp.array2d(dtype=T),
            max_a: wp.array2d(dtype=T),
            xl_a: wp.array2d(dtype=T),
            yl_a: wp.array2d(dtype=T),
            dx_a: wp.array2d(dtype=T),
            dy_a: wp.array2d(dtype=T),
            el_a: wp.array2d(dtype=T),
            act_a: wp.array2d(dtype=int),
            ar_a: wp.array(dtype=T),
            di_a: wp.array(dtype=T),
            tol_a: wp.array(dtype=T),
            cen_b: wp.array2d(dtype=T),
            nor_b: wp.array2d(dtype=T),
            lax_b: wp.array2d(dtype=T),
            max_b: wp.array2d(dtype=T),
            xl_b: wp.array2d(dtype=T),
            yl_b: wp.array2d(dtype=T),
            dx_b: wp.array2d(dtype=T),
            dy_b: wp.array2d(dtype=T),
            el_b: wp.array2d(dtype=T),
            act_b: wp.array2d(dtype=int),
            ar_b: wp.array(dtype=T),
            di_b: wp.array(dtype=T),
            tol_b: wp.array(dtype=T),
            has_b: int,
            sign: T,
            sigma: wp.array2d(dtype=T),
            chunk: int,
            n_panels: int,
            far_ratio: T,
            use_far: int,
            partial: wp.array3d(dtype=T),
        ):
            # One thread per (panel chunk, target): the partial sum over the
            # panels of one chunk. The chunk index is the slow index, so the
            # threads of a warp read the same panel at the same time. A
            # deterministic sum over the chunks follows on the device.
            q, i = wp.tid()
            px = P[i, 0]
            py = P[i, 1]
            pz = P[i, 2]
            j0 = q * chunk
            j1 = j0 + chunk
            if j1 > n_panels:
                j1 = n_panels
            s0 = T(0.0)
            s1 = T(0.0)
            s2 = T(0.0)
            s3 = T(0.0)
            s4 = T(0.0)
            s5 = T(0.0)
            for j in range(j0, j1):
                b = pair_b(px, py, pz, j, cen_a, nor_a, lax_a, max_a, xl_a, yl_a, dx_a, dy_a, el_a, act_a,
                           ar_a, di_a, tol_a, far_ratio, use_far)
                if has_b != 0:
                    b2 = pair_b(px, py, pz, j, cen_b, nor_b, lax_b, max_b, xl_b, yl_b, dx_b, dy_b, el_b,
                                act_b, ar_b, di_b, tol_b, far_ratio, use_far)
                    b = b + sign * b2
                s0 = s0 + b * sigma[j, 0]
                s1 = s1 + b * sigma[j, 1]
                s2 = s2 + b * sigma[j, 2]
                s3 = s3 + b * sigma[j, 3]
                s4 = s4 + b * sigma[j, 4]
                s5 = s5 + b * sigma[j, 5]
            partial[q, i, 0] = s0
            partial[q, i, 1] = s1
            partial[q, i, 2] = s2
            partial[q, i, 3] = s3
            partial[q, i, 4] = s4
            partial[q, i, 5] = s5

        found = {
            "bc": bc_kernel,
            "doublet_pair": doublet_pair_kernel,
            "source_pair": source_pair_kernel,
        }
        _built[precision] = found
        return found


def _resolve_precision(dtype, collocation, panels) -> str:
    """Return 'float32' or 'float64' for the requested *dtype* (None reads the inputs)."""
    import torch

    if dtype is None:
        target = np.result_type(np.asarray(collocation).dtype, np.asarray(panels).dtype)
        return "float32" if target == np.dtype(np.float32) else "float64"
    if isinstance(dtype, torch.dtype):
        if dtype == torch.float32:
            return "float32"
        if dtype == torch.float64:
            return "float64"
        raise ValueError(f"The Warp panel kernels use float32 or float64, not {dtype}.")
    if isinstance(dtype, str):
        key = dtype.strip().lower()
        if key in ("float32", "float64"):
            return key
        raise ValueError(f"The Warp panel kernels use float32 or float64, not {dtype!r}.")
    resolved = np.dtype(dtype)
    return "float32" if resolved == np.dtype(np.float32) else "float64"


def influence_matrices_warp(
    collocation,
    panels,
    far_ratio: float = 5.0,
    device=None,
    dtype=None,
):
    """Compute aerodynamic influence matrices B and C on a CUDA GPU.

    Same results as :func:`ventorum.panel3d.kernels.influence_matrices`:
    ``B`` holds the source potentials and ``C`` the doublet potentials.
    One GPU thread computes one (target, panel) pair.

    Parameters
    ----------
    collocation : array_like
        Collocation point coordinates with shape (P, 3) in m.
    panels : array_like
        Panel corner coordinates with shape (M, 4, 3) in m.
    far_ratio : float, default=5.0
        Distance-to-diameter ratio for point far-field transition.
    device : optional
        CUDA device of the computation (for example ``"cuda:0"``).
        None selects the current CUDA device.
    dtype : optional
        Arithmetic of the kernels: ``"float32"`` or ``"float64"``
        (also as torch or numpy dtypes). None reads the inputs:
        float32 only when both inputs are float32, else float64.

    Returns
    -------
    tuple of torch.Tensor
        (B, C) on the CUDA device: B with shape (P, M) holds the
        source potentials and C with shape (P, M) the doublet potentials.

    Raises
    ------
    RuntimeError
        If no CUDA GPU is available.
    ValueError
        If *device* is not a CUDA device or *dtype* is not supported.
    """
    import torch
    import warp as wp

    from ventorum.panel3d.kernels import _input_eps, _precompute_geometry

    dev, dev_str = _cuda_device(device)

    precision = _resolve_precision(dtype, collocation, panels)
    torch_dtype = torch.float32 if precision == "float32" else torch.float64
    numpy_dtype = np.float32 if precision == "float32" else np.float64
    kernel = _kernels(precision)["bc"]
    warp_type = wp.float32 if precision == "float32" else wp.float64

    pts = np.ascontiguousarray(np.asarray(collocation, dtype=np.float64))
    pans = np.ascontiguousarray(np.asarray(panels, dtype=np.float64))
    n_targets, n_panels = pts.shape[0], pans.shape[0]
    if n_targets == 0 or n_panels == 0:
        empty_b = torch.empty((n_targets, n_panels), dtype=torch_dtype, device=dev)
        empty_c = torch.empty((n_targets, n_panels), dtype=torch_dtype, device=dev)
        return empty_b, empty_c

    # The on-plane band follows the coarser of the input and the kernel precision.
    eps = max(_input_eps(collocation, panels), float(np.finfo(numpy_dtype).eps))
    geom = _precompute_geometry(pans, eps)

    def _move(data: np.ndarray):
        return wp.array(np.ascontiguousarray(data, dtype=numpy_dtype), dtype=warp_type, device=dev_str)

    w_pts = _move(pts)
    w_cen = _move(geom.centroid)
    w_nor = _move(geom.normal)
    w_lax = _move(geom.l_axis)
    w_max = _move(geom.m_axis)
    w_xl = _move(geom.x_loc)
    w_yl = _move(geom.y_loc)
    w_dx = _move(geom.dx)
    w_dy = _move(geom.dy)
    w_el = _move(geom.edge_len)
    w_act = wp.array(np.ascontiguousarray(geom.edge_active, dtype=np.int32), dtype=wp.int32, device=dev_str)
    w_area = _move(geom.area)
    w_diam = _move(geom.diameter)
    w_tol = _move(geom.plane_tol)
    w_b = wp.empty((n_targets, n_panels), dtype=warp_type, device=dev_str)
    w_c = wp.empty((n_targets, n_panels), dtype=warp_type, device=dev_str)

    use_far = 0 if np.isinf(far_ratio) else 1
    wp.launch(
        kernel,
        dim=(n_targets, n_panels),
        inputs=[w_pts, w_cen, w_nor, w_lax, w_max, w_xl, w_yl, w_dx, w_dy, w_el, w_act, w_area,
                w_diam, w_tol, float(far_ratio), use_far, w_b, w_c],
        device=dev_str,
    )
    if hasattr(wp, "synchronize"):
        wp.synchronize()
    return wp.to_torch(w_b), wp.to_torch(w_c)


def _cuda_device(device):
    """Return (torch device, Warp device string) of the CUDA device to use.

    Raises
    ------
    RuntimeError
        If no CUDA GPU is available.
    ValueError
        If *device* is not a CUDA device.
    """
    import torch

    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("The Warp panel kernels need a CUDA GPU, but PyTorch finds none.")
        dev = torch.device("cuda", torch.cuda.current_device())
    else:
        dev = device if isinstance(device, torch.device) else torch.device(device)
        if dev.type != "cuda":
            raise ValueError(f"The Warp panel kernels run on a CUDA device, not on {dev}.")
        if not torch.cuda.is_available():
            raise RuntimeError("The Warp panel kernels need a CUDA GPU, but PyTorch finds none.")
    dev_str = f"cuda:{dev.index}" if dev.index is not None else "cuda"
    return dev, dev_str


def _geometry_arrays(pans: np.ndarray, eps: float, numpy_dtype, dev_str: str) -> dict:
    """Move the precomputed geometry of a panel set to the device.

    Parameters
    ----------
    pans : numpy.ndarray
        Float64 panel corners with shape (M, 4, 3) in m.
    eps : float
        Machine epsilon [-] of the on-plane band.
    numpy_dtype
        Kernel precision (numpy.float32 or numpy.float64).
    dev_str : str
        Warp device string.

    Returns
    -------
    dict
        Warp arrays of the panel geometry, keyed by the field name.
    """
    import warp as wp

    from ventorum.panel3d.kernels import _precompute_geometry

    warp_type = wp.float32 if numpy_dtype == np.float32 else wp.float64
    geom = _precompute_geometry(pans, eps)

    def _move(data: np.ndarray):
        return wp.array(np.ascontiguousarray(data, dtype=numpy_dtype), dtype=warp_type, device=dev_str)

    return {
        "cen": _move(geom.centroid),
        "nor": _move(geom.normal),
        "lax": _move(geom.l_axis),
        "max": _move(geom.m_axis),
        "xl": _move(geom.x_loc),
        "yl": _move(geom.y_loc),
        "dx": _move(geom.dx),
        "dy": _move(geom.dy),
        "el": _move(geom.edge_len),
        "act": wp.array(np.ascontiguousarray(geom.edge_active, dtype=np.int32), dtype=wp.int32,
                        device=dev_str),
        "area": _move(geom.area),
        "diam": _move(geom.diameter),
        "tol": _move(geom.plane_tol),
    }


def _doublet_args(g: dict) -> list:
    """Return the geometry arguments of the doublet pair kernel."""
    return [g["cen"], g["nor"], g["lax"], g["max"], g["xl"], g["yl"], g["el"], g["act"], g["area"],
            g["diam"], g["tol"]]


def _source_args(g: dict) -> list:
    """Return the geometry arguments of the source pair kernel."""
    return [g["cen"], g["nor"], g["lax"], g["max"], g["xl"], g["yl"], g["dx"], g["dy"], g["el"],
            g["act"], g["area"], g["diam"], g["tol"]]


def _torch_dtype(dtype, default):
    """Return the torch dtype for a torch, numpy or string *dtype* (None gives *default*)."""
    import torch

    if dtype is None:
        return default
    return torch.float32 if _resolve_precision(dtype, None, None) == "float32" else torch.float64


class _PairSetup:
    """Device, precision and uploaded geometry of one pair kernel call."""

    def __init__(self, points, panels_a, panels_b, sigma, device, dtype):
        import torch

        from ventorum.panel3d.kernels import _pair_inputs

        _, pts, pa, pb, sig, flat, _, in_eps = _pair_inputs(points, panels_a, panels_b, sigma)
        self.dev, self.dev_str = _cuda_device(device)
        inputs = [points, panels_a] + ([] if panels_b is None else [panels_b])
        if dtype is None:
            target = np.result_type(*[np.asarray(a).dtype for a in inputs])
            self.precision = "float32" if target == np.dtype(np.float32) else "float64"
        else:
            self.precision = _resolve_precision(dtype, None, None)
        self.torch_dtype = torch.float32 if self.precision == "float32" else torch.float64
        self.numpy_dtype = np.float32 if self.precision == "float32" else np.float64
        self.kernels = _kernels(self.precision)
        self.pts = np.ascontiguousarray(pts)
        self.n_targets = int(pts.shape[0])
        self.n_panels = int(pa.shape[0])
        self.has_b = pb is not None
        self.sigma = sig
        self.flat = flat
        self.empty = self.n_targets == 0 or self.n_panels == 0
        if self.empty:
            return
        import warp as wp

        warp_type = wp.float32 if self.precision == "float32" else wp.float64
        # The on-plane band follows the coarser of the input and the kernel precision.
        eps = max(in_eps, float(np.finfo(self.numpy_dtype).eps))
        self.w_pts = wp.array(self.pts.astype(self.numpy_dtype), dtype=warp_type, device=self.dev_str)
        self.geom_a = _geometry_arrays(pa, eps, self.numpy_dtype, self.dev_str)
        self.geom_b = _geometry_arrays(pb, eps, self.numpy_dtype, self.dev_str) if self.has_b else self.geom_a


def _doublet_pair(setup: _PairSetup, sign, far_ratio, out_dtype, block_rows):
    """Run the doublet pair kernel of one set-up; return a torch tensor (P, M)."""
    import torch
    import warp as wp

    out_t = _torch_dtype(out_dtype, setup.torch_dtype)
    n_p, n_m = setup.n_targets, setup.n_panels
    if setup.empty:
        return torch.empty((n_p, n_m), dtype=out_t, device=setup.dev)
    kernel = setup.kernels["doublet_pair"]
    warp_type = wp.float32 if setup.precision == "float32" else wp.float64
    use_far = 0 if np.isinf(far_ratio) else 1
    tail = [1 if setup.has_b else 0, float(sign), float(far_ratio), use_far]
    geo = _doublet_args(setup.geom_a) + _doublet_args(setup.geom_b)
    if out_t == setup.torch_dtype:
        w_c = wp.empty((n_p, n_m), dtype=warp_type, device=setup.dev_str)
        wp.launch(kernel, dim=(n_p, n_m), inputs=[setup.w_pts, 0, *geo, *tail, w_c],
                  device=setup.dev_str)
        if hasattr(wp, "synchronize"):
            wp.synchronize()
        return wp.to_torch(w_c)
    # Kernel precision differs from the output dtype: compute blocks of
    # target rows in the kernel precision, then round each block to the
    # output dtype. Only one block of the kernel precision is on the device.
    item = 4 if setup.precision == "float32" else 8
    rows = int(block_rows) if block_rows is not None else max(1, BLOCK_BYTES // (n_m * item))
    rows = max(1, min(rows, n_p))
    out = torch.empty((n_p, n_m), dtype=out_t, device=setup.dev)
    w_block = wp.empty((rows, n_m), dtype=warp_type, device=setup.dev_str)
    block_t = wp.to_torch(w_block)
    for r0 in range(0, n_p, rows):
        r1 = min(r0 + rows, n_p)
        wp.launch(kernel, dim=(r1 - r0, n_m), inputs=[setup.w_pts, r0, *geo, *tail, w_block],
                  device=setup.dev_str)
        wp.synchronize()
        out[r0:r1] = block_t[: r1 - r0].to(out_t)
        torch.cuda.synchronize(setup.dev)
    del block_t, w_block
    return out


def _source_pair(setup: _PairSetup, sign, far_ratio):
    """Run the source pair kernel of one set-up; return a torch tensor (P, K)."""
    import torch
    import warp as wp

    n_p, n_m = setup.n_targets, setup.n_panels
    n_k = setup.sigma.shape[1]
    if setup.empty:
        out = torch.zeros((n_p, n_k), dtype=setup.torch_dtype, device=setup.dev)
        return out.reshape(-1) if setup.flat else out
    warp_type = wp.float32 if setup.precision == "float32" else wp.float64
    padded = np.zeros((n_m, MAX_RHS_COLUMNS), dtype=setup.numpy_dtype)
    padded[:, :n_k] = setup.sigma
    w_sigma = wp.array(padded, dtype=warp_type, device=setup.dev_str)
    n_chunks = -(-n_m // RHS_CHUNK)
    w_part = wp.zeros((n_chunks, n_p, MAX_RHS_COLUMNS), dtype=warp_type, device=setup.dev_str)
    use_far = 0 if np.isinf(far_ratio) else 1
    geo = _source_args(setup.geom_a) + _source_args(setup.geom_b)
    wp.launch(
        setup.kernels["source_pair"],
        dim=(n_chunks, n_p),
        inputs=[setup.w_pts, *geo, 1 if setup.has_b else 0, float(sign), w_sigma, RHS_CHUNK, n_m,
                float(far_ratio), use_far, w_part],
        device=setup.dev_str,
    )
    wp.synchronize()
    # A plain reduction over the chunk axis (no atomic additions): the
    # result does not change from run to run.
    out = wp.to_torch(w_part).sum(dim=0)[:, :n_k].contiguous()
    torch.cuda.synchronize(setup.dev)
    del w_part
    return out.reshape(-1) if setup.flat else out


def doublet_pair_matrix_warp(
    points,
    panels_a,
    panels_b,
    sign: float = 1.0,
    far_ratio: float = 5.0,
    device=None,
    dtype=None,
    out_dtype=None,
    block_rows: int | None = None,
):
    """Compute ``C(points, panels_a) + sign * C(points, panels_b)`` on a CUDA GPU.

    Same results as :func:`ventorum.panel3d.kernels.doublet_pair_matrix`.
    One GPU thread computes one target, one panel and its mirror (or
    image); the two separate matrices are never stored.

    Parameters
    ----------
    points : array_like
        Target coordinates with shape (P, 3) in m.
    panels_a : array_like
        Panel corner coordinates with shape (M, 4, 3) in m.
    panels_b : array_like or None
        Second panel set with shape (M, 4, 3) in m. None gives
        ``C(points, panels_a)``.
    sign : float, default=1.0
        Factor of the second panel set [-] (+1 or -1).
    far_ratio : float, default=5.0
        Distance-to-diameter ratio for point far-field transition.
    device : optional
        CUDA device (for example ``"cuda:0"``). None selects the current
        CUDA device.
    dtype : optional
        Arithmetic of the kernel: ``"float32"`` or ``"float64"`` (also as
        torch or numpy dtypes). None reads the inputs.
    out_dtype : optional
        Dtype of the output. None keeps the kernel precision. A different
        dtype (for example float64 arithmetic and a float32 output) rounds
        blocks of rows; only one block of the kernel precision is kept.
    block_rows : int or None, optional
        Rows of one block [-] when *out_dtype* differs from the kernel
        precision. None uses blocks of about 64 MiB.

    Returns
    -------
    torch.Tensor
        Doublet potential sum [-] on the CUDA device with shape (P, M).
    """
    setup = _PairSetup(points, panels_a, panels_b, None, device, dtype)
    return _doublet_pair(setup, sign, far_ratio, out_dtype, block_rows)


def source_pair_rhs_warp(
    points,
    panels_a,
    panels_b,
    sign: float,
    sigma,
    far_ratio: float = 5.0,
    device=None,
    dtype=None,
):
    """Compute ``(B(points, panels_a) + sign * B(points, panels_b)) @ sigma`` on a CUDA GPU.

    Same results as :func:`ventorum.panel3d.kernels.source_pair_rhs` to
    round-off. One GPU thread sums one target over a chunk of panels;
    a reduction over the chunks follows. The source matrix is never
    stored.

    Parameters
    ----------
    points : array_like
        Target coordinates with shape (P, 3) in m.
    panels_a : array_like
        Panel corner coordinates with shape (M, 4, 3) in m.
    panels_b : array_like or None
        Second panel set with shape (M, 4, 3) in m, or None.
    sign : float
        Factor of the second panel set [-] (+1 or -1).
    sigma : array_like
        Source strengths with shape (M, K), K at most 6, or (M,) in m/s.
    far_ratio : float, default=5.0
        Distance-to-diameter ratio for point far-field transition.
    device : optional
        CUDA device. None selects the current CUDA device.
    dtype : optional
        Arithmetic of the kernel. None reads the inputs.

    Returns
    -------
    torch.Tensor
        Source potentials on the CUDA device with shape (P, K), or (P,)
        for a one-dimensional *sigma*, in m^2/s.
    """
    setup = _PairSetup(points, panels_a, panels_b, sigma, device, dtype)
    return _source_pair(setup, sign, far_ratio)


def source_rhs_warp(points, panels, sigma, far_ratio: float = 5.0, device=None, dtype=None):
    """Compute ``B(points, panels) @ sigma`` on a CUDA GPU.

    Same results as :func:`ventorum.panel3d.kernels.source_rhs` to
    round-off (see :func:`source_pair_rhs_warp`).

    Parameters
    ----------
    points : array_like
        Target coordinates with shape (P, 3) in m.
    panels : array_like
        Panel corner coordinates with shape (M, 4, 3) in m.
    sigma : array_like
        Source strengths with shape (M, K), K at most 6, or (M,) in m/s.
    far_ratio : float, default=5.0
        Distance-to-diameter ratio for point far-field transition.
    device : optional
        CUDA device. None selects the current CUDA device.
    dtype : optional
        Arithmetic of the kernel. None reads the inputs.

    Returns
    -------
    torch.Tensor
        Source potentials on the CUDA device with shape (P, K), or (P,),
        in m^2/s.
    """
    return source_pair_rhs_warp(points, panels, None, 1.0, sigma, far_ratio=far_ratio, device=device,
                                dtype=dtype)


def pair_system_warp(
    points,
    panels_a,
    panels_b,
    sign: float,
    sigma,
    far_ratio: float = 5.0,
    device=None,
    dtype=None,
    out_dtype=None,
    block_rows: int | None = None,
):
    """Compute the doublet pair matrix and the source pair right-hand sides on a CUDA GPU.

    Same results as :func:`ventorum.panel3d.kernels.pair_system` to
    round-off. The geometry of both panel sets goes to the device once.

    Parameters
    ----------
    points : array_like
        Target coordinates with shape (P, 3) in m.
    panels_a : array_like
        Panel corner coordinates with shape (M, 4, 3) in m.
    panels_b : array_like or None
        Second panel set with shape (M, 4, 3) in m, or None.
    sign : float
        Factor of the second panel set [-] (+1 or -1).
    sigma : array_like
        Source strengths with shape (M, K), K at most 6, or (M,) in m/s.
    far_ratio : float, default=5.0
        Distance-to-diameter ratio for point far-field transition.
    device : optional
        CUDA device. None selects the current CUDA device.
    dtype : optional
        Arithmetic of the kernels. None reads the inputs.
    out_dtype : optional
        Dtype of the doublet matrix (see :func:`doublet_pair_matrix_warp`).
        The right-hand sides keep the kernel precision.
    block_rows : int or None, optional
        Rows of one block of the doublet matrix [-] (see
        :func:`doublet_pair_matrix_warp`).

    Returns
    -------
    tuple of torch.Tensor
        (C, R) on the CUDA device: C with shape (P, M) [-] and R with
        shape (P, K) in m^2/s.
    """
    setup = _PairSetup(points, panels_a, panels_b, sigma, device, dtype)
    return (_doublet_pair(setup, sign, far_ratio, out_dtype, block_rows),
            _source_pair(setup, sign, far_ratio))
