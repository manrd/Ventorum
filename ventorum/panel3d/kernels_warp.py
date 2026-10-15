# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Warp (CUDA) versions of the 3D panel influence kernels of :mod:`ventorum.panel3d.kernels`.

The kernels compute the same matrices as the numpy reference: the source
potentials ``B`` and the doublet potentials ``C``, with the same formulas
and the same branches (planar projection done on the host, the van
Oosterom and Strackee solid angle, the analytic edge sums, and the point
far-field expansions beyond ``far_ratio`` panel diameters).

Thread map: one thread computes one (target, panel) pair, so the launch
has ``(P, M)`` threads. Each precision (float32, float64) builds its own
kernels once per process. Warp and PyTorch are imported lazily inside the
functions, so importing this module never loads them. The outputs are
PyTorch tensors on the CUDA device (``wp.to_torch``).

References
----------
* J. L. Hess and A. M. O. Smith, "Calculation of Nonlifting Potential Flow About
  Arbitrary Three-Dimensional Bodies", Progress in Aeronautical Sciences 8 (1967) 1-138.
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

__all__ = ["influence_matrices_warp"]

_INV_4PI = 1.0 / (4.0 * math.pi)

_lock = threading.Lock()
_built: dict[str, object] = {}
_initialised = False


def _kernels(precision: str):
    """Build and cache the Warp kernels of one precision."""
    import warp as wp

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
        def solid_angle(
            x: T,
            y: T,
            z: T,
            j: int,
            xloc: wp.array2d(dtype=T),
            yloc: wp.array2d(dtype=T),
            elen: wp.array2d(dtype=T),
            area: wp.array(dtype=T),
        ):
            r0 = wp.sqrt((x - xloc[j, 0]) * (x - xloc[j, 0]) + (y - yloc[j, 0]) * (y - yloc[j, 0]) + z * z)
            r1 = wp.sqrt((x - xloc[j, 1]) * (x - xloc[j, 1]) + (y - yloc[j, 1]) * (y - yloc[j, 1]) + z * z)
            r2 = wp.sqrt((x - xloc[j, 2]) * (x - xloc[j, 2]) + (y - yloc[j, 2]) * (y - yloc[j, 2]) + z * z)
            r3 = wp.sqrt((x - xloc[j, 3]) * (x - xloc[j, 3]) + (y - yloc[j, 3]) * (y - yloc[j, 3]) + z * z)
            om = tri_omega(x, y, z, xloc[j, 0], yloc[j, 0], xloc[j, 1], yloc[j, 1],
                           xloc[j, 2], yloc[j, 2], r0, r1, r2)
            if elen[j, 3] > T(1e-14):
                om = om + tri_omega(x, y, z, xloc[j, 0], yloc[j, 0], xloc[j, 2], yloc[j, 2],
                                    xloc[j, 3], yloc[j, 3], r0, r2, r3)
            az = z
            if az < T(0.0):
                az = T(0.0) - az
            if az < T(1e-14):
                inside = True
                for k in range(4):
                    if elen[j, k] > T(1e-14):
                        kn = k + 1
                        if kn == 4:
                            kn = 0
                        cross = (xloc[j, kn] - xloc[j, k]) * (y - yloc[j, k]) - (
                            yloc[j, kn] - yloc[j, k]) * (x - xloc[j, k])
                        if cross < T(-1e-12) * area[j]:
                            inside = False
                if inside:
                    om = T(2.0) * T(3.141592653589793)
                else:
                    om = T(0.0)
            return om

        @wp.func(module=module)
        def edge_log(x: T, y: T, z: T, xa: T, ya: T, xb: T, yb: T, dk: T):
            rk = wp.sqrt((x - xa) * (x - xa) + (y - ya) * (y - ya) + z * z)
            rkp1 = wp.sqrt((x - xb) * (x - xb) + (y - yb) * (y - yb) + z * z)
            den = rk + rkp1 - dk
            lk = T(0.0)
            if den > T(1e-15):
                lk = wp.log((rk + rkp1 + dk) / den)
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
            area: wp.array(dtype=T),
            diam: wp.array(dtype=T),
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
                om = solid_angle(x, y, z, j, xloc, yloc, elen, area)
                s = T(0.0)
                for k in range(4):
                    if elen[j, k] > T(1e-14):
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

        _built[precision] = bc_kernel
        return bc_kernel


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

    from ventorum.panel3d.kernels import _precompute_geometry

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

    precision = _resolve_precision(dtype, collocation, panels)
    torch_dtype = torch.float32 if precision == "float32" else torch.float64
    numpy_dtype = np.float32 if precision == "float32" else np.float64
    kernel = _kernels(precision)
    warp_type = wp.float32 if precision == "float32" else wp.float64

    pts = np.ascontiguousarray(np.asarray(collocation, dtype=np.float64))
    pans = np.ascontiguousarray(np.asarray(panels, dtype=np.float64))
    n_targets, n_panels = pts.shape[0], pans.shape[0]
    if n_targets == 0 or n_panels == 0:
        empty_b = torch.empty((n_targets, n_panels), dtype=torch_dtype, device=dev)
        empty_c = torch.empty((n_targets, n_panels), dtype=torch_dtype, device=dev)
        return empty_b, empty_c

    geom = _precompute_geometry(pans)

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
    w_area = _move(geom.area)
    w_diam = _move(geom.diameter)
    w_b = wp.empty((n_targets, n_panels), dtype=warp_type, device=dev_str)
    w_c = wp.empty((n_targets, n_panels), dtype=warp_type, device=dev_str)

    use_far = 0 if np.isinf(far_ratio) else 1
    wp.launch(
        kernel,
        dim=(n_targets, n_panels),
        inputs=[w_pts, w_cen, w_nor, w_lax, w_max, w_xl, w_yl, w_dx, w_dy, w_el, w_area,
                w_diam, float(far_ratio), use_far, w_b, w_c],
        device=dev_str,
    )
    if hasattr(wp, "synchronize"):
        wp.synchronize()
    return wp.to_torch(w_b), wp.to_torch(w_c)
