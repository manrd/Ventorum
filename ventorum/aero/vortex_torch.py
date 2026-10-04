# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""PyTorch backend of the bent-horseshoe kernels of :mod:`ventorum.aero.vortex`.

The functions here compute the same quantities as the numpy reference kernels
and the Numba kernels: the velocity of the bent horseshoe vortices (bound
segment, chordwise legs and semi-infinite wake legs) from the Biot-Savart law,
with the same regularisation and the same floors. They take the same packed
arrays as :mod:`ventorum.aero.vortex_numba` (see
:func:`~ventorum.aero.vortex_numba.pack_sources` and
:func:`~ventorum.aero.vortex_numba.pack_targets`), the same part selector
(0 all legs, 1 fixed legs, 2 wake legs), and they return numpy arrays on the
CPU.

Method: vectorised PyTorch tensors in float64 only, on the CPU or on a CUDA
GPU (Katz, J. and Plotkin, A., "Low-Speed Aerodynamics", 2nd ed., Cambridge
University Press, 2001; Anderson, J. D., "Fundamentals of Aerodynamics", 5th
ed., McGraw-Hill, 2011).

Device: the environment variable ``VENTORUM_TORCH_DEVICE`` (``cpu``, ``cuda``
or ``cuda:N``); when it is not set, ``cuda`` if a CUDA GPU is available, else
``cpu``. Apple ``mps`` is not used: it has no float64 support, so the kernels
always run on ``cpu`` or ``cuda``.

Memory: the evaluation points run in chunks. One chunk holds at most
``VENTORUM_TORCH_CHUNK`` float64 values in its largest temporary (the
``(rows, sources, 3)`` pair tensor; default 2.0e7). The results do not depend
on the chunk size.

CPU threads: on the CPU device each call uses
:func:`ventorum.utils.parallel.kernel_threads` torch threads, and restores
the previous count after it.
"""

from __future__ import annotations

import contextlib
import math
import os

import numpy as np

try:
    import torch

    HAVE_TORCH = True
except ImportError:  # pragma: no cover - torch is optional
    torch = None  # type: ignore[assignment]
    HAVE_TORCH = False

_INV_4PI = 1.0 / (4.0 * math.pi)
_INV_2PI = 1.0 / (2.0 * math.pi)

# Default target number of float64 values in the largest temporary of a chunk.
_DEFAULT_CHUNK_VALUES = 2.0e7


def torch_device() -> str:
    """Return the torch device of the kernels: ``"cpu"``, ``"cuda"`` or ``"cuda:N"``.

    The environment variable ``VENTORUM_TORCH_DEVICE`` wins when it is set.
    Else the device is ``"cuda"`` if a CUDA GPU is available, else ``"cpu"``.

    Returns
    -------
    str
        The device string.
    """
    env = os.environ.get("VENTORUM_TORCH_DEVICE", "").strip()
    if env:
        return env
    if HAVE_TORCH and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _chunk_budget() -> float:
    """Return the target number of float64 values in the largest temporary of a chunk."""
    raw = os.environ.get("VENTORUM_TORCH_CHUNK", "").strip()
    if raw:
        try:
            value = float(raw)
        except ValueError:
            return _DEFAULT_CHUNK_VALUES
        if value > 0.0:
            return value
    return _DEFAULT_CHUNK_VALUES


def _chunk_rows(m: int, n: int) -> int:
    """Return the rows of evaluation points of one chunk for *m* points and *n* sources."""
    per_row = 3 * max(1, int(n))
    return max(1, min(int(m), int(_chunk_budget() // per_row)))


def _require_torch() -> None:
    """Raise when PyTorch is not available."""
    if not HAVE_TORCH:
        raise RuntimeError("The torch kernel backend needs the torch package.")


def _f64(values: np.ndarray, device) -> torch.Tensor:
    """Return *values* as a float64 tensor on *device*."""
    arr = np.ascontiguousarray(values, dtype=np.float64)
    if not arr.flags.writeable:
        arr = arr.copy()  # torch.as_tensor needs a writable buffer
    return torch.as_tensor(arr, dtype=torch.float64, device=device)


def _i64(values: np.ndarray, device) -> torch.Tensor:
    """Return *values* as an int64 tensor on *device*."""
    arr = np.ascontiguousarray(values, dtype=np.int64)
    if not arr.flags.writeable:
        arr = arr.copy()  # torch.as_tensor needs a writable buffer
    return torch.as_tensor(arr, dtype=torch.int64, device=device)


@contextlib.contextmanager
def _torch_threads(device):
    """Set the torch thread count for one call on the CPU device, and restore it after it."""
    if device.type == "cpu":
        from ventorum.utils.parallel import kernel_threads

        prev = torch.get_num_threads()
        torch.set_num_threads(kernel_threads())
        try:
            yield
        finally:
            torch.set_num_threads(prev)
    else:
        yield


def _total_rc2(rc2_src, src_group, tg_group, tg_rc2, use_tg):
    """Return the squared core radius per (point, source): (m, n) or (1, n)."""
    if use_tg:
        other = tg_group[:, None] != src_group[None, :]
        return rc2_src[None, :] + torch.where(other, tg_rc2[:, None], 0.0)
    return rc2_src[None, :]


def _segment(P, A, B, rc2):
    """Return the velocity of unit-strength finite segments *A* -> *B*: (m, n, 3)."""
    r1 = P[:, None, :] - A[None, :, :]
    r2 = P[:, None, :] - B[None, :, :]
    r0 = B[None, :, :] - A[None, :, :]
    cx = r1[..., 1] * r2[..., 2] - r1[..., 2] * r2[..., 1]
    cy = r1[..., 2] * r2[..., 0] - r1[..., 0] * r2[..., 2]
    cz = r1[..., 0] * r2[..., 1] - r1[..., 1] * r2[..., 0]
    cr2 = cx * cx + cy * cy + cz * cz
    r0sq = r0[..., 0] * r0[..., 0] + r0[..., 1] * r0[..., 1] + r0[..., 2] * r0[..., 2]
    # A tiny floor keeps the unit vectors finite when P is on an end point.
    n1 = torch.sqrt((r1 * r1).sum(-1) + 1e-300)
    n2 = torch.sqrt((r2 * r2).sum(-1) + 1e-300)
    t1 = r1 / n1[..., None]
    t2 = r2 / n2[..., None]
    proj = (r0[..., 0] * (t1[..., 0] - t2[..., 0])
            + r0[..., 1] * (t1[..., 1] - t2[..., 1])
            + r0[..., 2] * (t1[..., 2] - t2[..., 2]))
    denom = cr2 + rc2 * r0sq
    ok = denom > 1e-300
    safe = torch.where(ok, denom, torch.ones_like(denom))
    k = torch.where(ok, _INV_4PI * proj / safe, torch.zeros_like(proj))
    return torch.stack((cx * k, cy * k, cz * k), dim=-1)


def _ray(P, A, d, rc2):
    """Return the velocity of unit-strength semi-infinite lines from *A* along *d*: (m, n, 3)."""
    r = P[:, None, :] - A[None, :, :]
    dd = d[None, :, :]
    cx = dd[..., 1] * r[..., 2] - dd[..., 2] * r[..., 1]
    cy = dd[..., 2] * r[..., 0] - dd[..., 0] * r[..., 2]
    cz = dd[..., 0] * r[..., 1] - dd[..., 1] * r[..., 0]
    cr2 = cx * cx + cy * cy + cz * cz
    rn = torch.sqrt((r * r).sum(-1) + 1e-300)
    cos_t = (dd * r).sum(-1) / rn
    denom = cr2 + rc2
    ok = denom > 1e-300
    safe = torch.where(ok, denom, torch.ones_like(denom))
    k = torch.where(ok, _INV_4PI * (1.0 + cos_t) / safe, torch.zeros_like(cos_t))
    return torch.stack((cx * k, cy * k, cz * k), dim=-1)


def _horseshoe(P, a, b, a_te, b_te, d, rc2, parts: int):
    """Return the unit-circulation velocity of the horseshoes: (m, n, 3).

    *parts* selects the legs: 0 all, 1 fixed (bound vortex and chordwise
    legs), 2 wake legs.
    """
    if parts == 2:
        return _ray(P, b_te, d, rc2) - _ray(P, a_te, d, rc2)
    fixed = (_segment(P, a, b, rc2) + _segment(P, a_te, a, rc2) + _segment(P, b, b_te, rc2))
    if parts == 1:
        return fixed
    return fixed + (_ray(P, b_te, d, rc2) - _ray(P, a_te, d, rc2))


def induced_velocity_kernel(P, a, b, a_te, b_te, d, rc2_src, src_group, g_src,
                            tg_group, tg_rc2, use_tg, parts=0) -> np.ndarray:
    """Return the total induced velocity [m/s] for the source strengths *g_src* (sign included).

    Parameters
    ----------
    P : (m, 3) array
        Evaluation points [m].
    a, b : (n, 3) arrays
        Bound-vortex end points [m].
    a_te, b_te : (n, 3) arrays
        Trailing-edge points where the legs turn into the wake [m].
    d : (n, 3) array
        Unit wake direction of each horseshoe.
    rc2_src : (n,) array
        Squared core radius of each source [m^2].
    src_group : (n,) array
        Surface index of each source.
    g_src : (n,) array
        Source strength (circulation times sign) [m^2/s].
    tg_group : (m,) array
        Surface index of each point.
    tg_rc2 : (m,) array
        Added squared core radius of each point [m^2].
    use_tg : bool
        Apply the target core when the flag is True.
    parts : int
        Legs to use: 0 all, 1 fixed, 2 wake.

    Returns
    -------
    (m, 3) array
        Induced velocity [m/s], on the CPU.
    """
    _require_torch()
    device = torch.device(torch_device())
    with _torch_threads(device), torch.inference_mode():
        tP = _f64(P, device)
        ta, tb, ta_te, tb_te, td = (_f64(a, device), _f64(b, device), _f64(a_te, device),
                                   _f64(b_te, device), _f64(d, device))
        trc2 = _f64(rc2_src, device)
        tgrp = _i64(src_group, device)
        tg = _f64(g_src, device)
        ttg = _i64(tg_group, device)
        ttrc2 = _f64(tg_rc2, device)
        m, n = tP.shape[0], ta.shape[0]
        out = np.empty((m, 3))
        for i0 in range(0, m, _chunk_rows(m, n)):
            i1 = min(m, i0 + _chunk_rows(m, n))
            rc2 = _total_rc2(trc2, tgrp, ttg[i0:i1], ttrc2[i0:i1], use_tg)
            v = _horseshoe(tP[i0:i1], ta, tb, ta_te, tb_te, td, rc2, parts)
            out[i0:i1] = (v * tg[None, :, None]).sum(1).cpu().numpy()
    return out


def normal_influence_kernel(P, normals, a, b, a_te, b_te, d, rc2_src, src_group, sign,
                            column, n_unknowns, tg_group, tg_rc2, use_tg, parts=0) -> np.ndarray:
    """Return the normal-velocity influence matrix [1/m].

    Sources that map to the same unknown (mirror and ground images) are added
    in source order.

    Parameters
    ----------
    P : (m, 3) array
        Evaluation points [m].
    normals : (m, 3) array
        Unit normals at the evaluation points.
    a, b : (n, 3) arrays
        Bound-vortex end points [m].
    a_te, b_te : (n, 3) arrays
        Trailing-edge points where the legs turn into the wake [m].
    d : (n, 3) array
        Unit wake direction of each horseshoe.
    rc2_src : (n,) array
        Squared core radius of each source [m^2].
    src_group : (n,) array
        Surface index of each source.
    sign : (n,) array
        Multiplier on the circulation (+1 real, -1 ground image).
    column : (n,) array
        Index of the unknown that gives each horseshoe its circulation.
    n_unknowns : int
        Number of unknowns.
    tg_group : (m,) array
        Surface index of each point.
    tg_rc2 : (m,) array
        Added squared core radius of each point [m^2].
    use_tg : bool
        Apply the target core when the flag is True.
    parts : int
        Legs to use: 0 all, 1 fixed, 2 wake.

    Returns
    -------
    (m, n_unknowns) array
        Normal velocity per unit circulation [1/m], on the CPU.
    """
    _require_torch()
    device = torch.device(torch_device())
    with _torch_threads(device), torch.inference_mode():
        tP = _f64(P, device)
        tN = _f64(normals, device)
        ta, tb, ta_te, tb_te, td = (_f64(a, device), _f64(b, device), _f64(a_te, device),
                                   _f64(b_te, device), _f64(d, device))
        trc2 = _f64(rc2_src, device)
        tgrp = _i64(src_group, device)
        tsign = _f64(sign, device)
        tcol = _i64(column, device)
        ttg = _i64(tg_group, device)
        ttrc2 = _f64(tg_rc2, device)
        m, n = tP.shape[0], ta.shape[0]
        out = np.empty((m, int(n_unknowns)))
        for i0 in range(0, m, _chunk_rows(m, n)):
            i1 = min(m, i0 + _chunk_rows(m, n))
            rc2 = _total_rc2(trc2, tgrp, ttg[i0:i1], ttrc2[i0:i1], use_tg)
            v = _horseshoe(tP[i0:i1], ta, tb, ta_te, tb_te, td, rc2, parts)
            a_src = (v * tN[i0:i1, None, :]).sum(-1) * tsign[None, :]
            acc = torch.zeros((i1 - i0, int(n_unknowns)), dtype=torch.float64, device=device)
            acc.index_add_(1, tcol, a_src)
            out[i0:i1] = acc.cpu().numpy()
    return out


def velocity_tensor_kernel(P, a, b, a_te, b_te, d, rc2_src, src_group, sign,
                           tg_group, tg_rc2, use_tg, parts=0) -> np.ndarray:
    """Return the velocity per unit source circulation [1/m], with the sign applied.

    Parameters
    ----------
    P : (m, 3) array
        Evaluation points [m].
    a, b : (n, 3) arrays
        Bound-vortex end points [m].
    a_te, b_te : (n, 3) arrays
        Trailing-edge points where the legs turn into the wake [m].
    d : (n, 3) array
        Unit wake direction of each horseshoe.
    rc2_src : (n,) array
        Squared core radius of each source [m^2].
    src_group : (n,) array
        Surface index of each source.
    sign : (n,) array
        Multiplier on the circulation (+1 real, -1 ground image).
    tg_group : (m,) array
        Surface index of each point.
    tg_rc2 : (m,) array
        Added squared core radius of each point [m^2].
    use_tg : bool
        Apply the target core when the flag is True.
    parts : int
        Legs to use: 0 all, 1 fixed, 2 wake.

    Returns
    -------
    (m, n, 3) array
        Velocity per unit circulation [1/m], on the CPU.
    """
    _require_torch()
    device = torch.device(torch_device())
    with _torch_threads(device), torch.inference_mode():
        tP = _f64(P, device)
        ta, tb, ta_te, tb_te, td = (_f64(a, device), _f64(b, device), _f64(a_te, device),
                                   _f64(b_te, device), _f64(d, device))
        trc2 = _f64(rc2_src, device)
        tgrp = _i64(src_group, device)
        tsign = _f64(sign, device)
        ttg = _i64(tg_group, device)
        ttrc2 = _f64(tg_rc2, device)
        m, n = tP.shape[0], ta.shape[0]
        out = np.empty((m, n, 3))
        for i0 in range(0, m, _chunk_rows(m, n)):
            i1 = min(m, i0 + _chunk_rows(m, n))
            rc2 = _total_rc2(trc2, tgrp, ttg[i0:i1], ttrc2[i0:i1], use_tg)
            v = _horseshoe(tP[i0:i1], ta, tb, ta_te, tb_te, td, rc2, parts)
            out[i0:i1] = (v * tsign[None, :, None]).cpu().numpy()
    return out


def trefftz_normalwash_kernel(q, q_normal, p, gamma, d, rc2, src_group, tg_group, tg_rc2,
                              use_tg) -> np.ndarray:
    """Return the normal wash [m/s] of 2-D point vortices in the Trefftz plane.

    Parameters
    ----------
    q : (m, 3) array
        Evaluation points [m], already projected onto the plane.
    q_normal : (m, 3) array
        Unit normals at the evaluation points, in the plane.
    p : (n, 3) array
        Vortex positions [m], projected onto the plane.
    gamma : (n,) array
        Circulation about the axis *d* [m^2/s]. The sign follows the
        right-hand rule.
    d : (3,) array
        Unit normal of the Trefftz plane (the wake direction).
    rc2 : (n,) array
        Squared 2-D core radius [m^2].
    src_group : (n,) int array
        Core group of each vortex.
    tg_group : (m,) int array
        Core group of each evaluation point.
    tg_rc2 : (m,) array
        Added squared core radius [m^2] between a point and a vortex of a
        different core group.
    use_tg : bool
        Apply the added core.

    Returns
    -------
    (m,) array
        Velocity component along ``q_normal`` [m/s], on the CPU.
    """
    _require_torch()
    device = torch.device(torch_device())
    with _torch_threads(device), torch.inference_mode():
        tq = _f64(q, device)
        tqn = _f64(q_normal, device)
        tp = _f64(p, device)
        tg = _f64(gamma, device)
        td = _f64(d, device)
        trc2 = _f64(rc2, device)
        tsg = torch.as_tensor(np.asarray(src_group), dtype=torch.int64, device=device)
        ttg = torch.as_tensor(np.asarray(tg_group), dtype=torch.int64, device=device)
        ttrc2 = _f64(tg_rc2, device)
        m, n = tq.shape[0], tp.shape[0]
        out = np.empty(m)
        for i0 in range(0, m, _chunk_rows(m, n)):
            i1 = min(m, i0 + _chunk_rows(m, n))
            r = tq[i0:i1, None, :] - tp[None, :, :]
            vx = td[1] * r[..., 2] - td[2] * r[..., 1]
            vy = td[2] * r[..., 0] - td[0] * r[..., 2]
            vz = td[0] * r[..., 1] - td[1] * r[..., 0]
            c2 = _total_rc2(trc2, tsg, ttg[i0:i1], ttrc2[i0:i1], use_tg)
            k = _INV_2PI * tg[None, :] / ((r * r).sum(-1) + c2)
            qn = tqn[i0:i1, None, :]
            wn = (vx * qn[..., 0] + vy * qn[..., 1] + vz * qn[..., 2]) * k
            out[i0:i1] = wn.sum(1).cpu().numpy()
    return out
