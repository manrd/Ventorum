# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Vortex-ring Biot-Savart kernels and steady trailing-edge wake closure.

This module provides the numpy reference kernels and dispatches to the
compiled Numba kernels in :mod:`ventorum.aero.vortex_ring_numba` when Numba
is available and the active backend is not forced to ``"numpy"``.

Each quadrilateral vortex ring has four corners ``(c0, c1, c2, c3)`` [m] in
cyclic order. Positive circulation follows the right-hand rule around the
oriented segments ``c0 -> c1 -> c2 -> c3 -> c0``:

* ``c0 -> c1``: leading bound segment (on the panel quarter-chord line).
* ``c1 -> c2``: right chordwise side segment.
* ``c2 -> c3``: trailing spanwise segment.
* ``c3 -> c0``: left chordwise side segment.

The steady wake closure of a trailing-edge strip is a semi-infinite steady
wake horseshoe starting at the trailing-edge ring corners ``(te_left,
te_right)``:

* ``te_left -> te_right``: leading spanwise segment (cancels the ``te_right
  -> te_left`` trailing segment of the closed trailing-edge ring).
* ``te_right -> infinity`` along ``wake_dir`` (positive semi-infinite leg).
* ``infinity -> te_left`` along ``-wake_dir`` (negative semi-infinite leg).

All segments and semi-infinite rays use the same finite-filament Scully core
regularisation as :mod:`ventorum.aero.vortex` (``segment_velocity`` and
``ray_velocity``).

Cython and GPU (NVIDIA Warp) backends for the ring kernels are planned for a
later step.

References
----------
* J. Katz and A. Plotkin, "Low-Speed Aerodynamics", 2nd ed., Cambridge
  University Press, 2001, chapters 10, 12 and 13 (Biot-Savart law, vortex
  rings and steady closure of a vortex-ring lattice).
* M. P. Scully, "Computation of helicopter rotor wake geometry and its
  influence on rotor harmonic airloads", ASRL TR 178-1, MIT, 1975.
* G. H. Vatistas, V. Kozel and W. C. Mih, "A simpler model for concentrated
  vortices", Experiments in Fluids 11 (1991) 73-76.
"""

from __future__ import annotations

import numpy as np

from ventorum.aero import vortex as _vortex
from ventorum.aero import vortex_ring_numba as _nb
from ventorum.aero.vortex import ray_velocity, segment_velocity

# Target number of float64 entries per temporary array in a numpy row chunk.
_CHUNK_BUDGET = 2_000_000


def _ring_backend(backend: str | None = None) -> str:
    """Return ``"numba"`` or ``"numpy"`` for the ring kernels.

    When *backend* is None, follow :data:`ventorum.aero.vortex._kernel_backend`:
    use ``"numpy"`` when numpy is forced, and ``"numba"`` whenever Numba is
    available otherwise.
    """
    if backend is not None:
        key = str(backend).lower()
        if key not in ("auto", "numba", "numpy"):
            raise ValueError(f"Unknown ring kernel backend {backend!r}; use 'auto', 'numba' or 'numpy'.")
        if key == "numba" and not _nb.HAVE_NUMBA:
            raise ValueError("The Numba ring kernel backend needs the numba package.")
        if key == "numpy":
            return "numpy"
        return "numba" if _nb.HAVE_NUMBA else "numpy"
    if _vortex._kernel_backend == "numpy":
        return "numpy"
    return "numba" if _nb.HAVE_NUMBA else "numpy"


def _chunk_rows(m: int, n: int) -> int:
    """Return the row count per chunk for numpy ring evaluations."""
    return max(1, min(m, _CHUNK_BUDGET // max(1, 36 * n)))


def _slice_rc(rc: np.ndarray | float, i0: int, i1: int) -> np.ndarray | float:
    """Slice *rc* along the evaluation-point axis when it is 2-D."""
    if np.ndim(rc) == 2:
        return np.asarray(rc, dtype=float)[i0:i1]
    return rc


def _select_rc_cols(rc: np.ndarray | float, idx: np.ndarray) -> np.ndarray | float:
    """Select ring columns *idx* from *rc* (scalar, 1-D or 2-D)."""
    nd = np.ndim(rc)
    if nd == 2:
        return np.asarray(rc, dtype=np.float64)[:, idx]
    if nd == 1:
        return np.asarray(rc, dtype=np.float64)[idx]
    return rc


def _validate_side_polylines(
    n: int,
    side_nodes: np.ndarray | None,
    side_counts: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None, bool]:
    """Validate optional bent side-edge polyline arrays for *n* rings.

    Returns ``(sn, sc, has_poly)`` where *has_poly* is True only when at least
    one ring side edge carries an intermediate node.
    """
    if side_nodes is None and side_counts is None:
        return None, None, False
    if side_nodes is None or side_counts is None:
        raise ValueError("side_nodes and side_counts must both be provided or both be None.")
    sn = np.asarray(side_nodes, dtype=np.float64)
    sc = np.asarray(side_counts, dtype=np.int64)
    if sn.ndim != 4 or sn.shape[0] != n or sn.shape[1] != 2 or sn.shape[3] != 3:
        raise ValueError(f"Expected side_nodes of shape ({n}, 2, K, 3), got {sn.shape}.")
    if sc.shape != (n, 2):
        raise ValueError(f"Expected side_counts of shape ({n}, 2), got {sc.shape}.")
    k_max = int(sn.shape[2])
    if n > 0 and (bool(np.any(sc < 0)) or bool(np.any(sc > k_max))):
        raise ValueError(f"All side_counts entries must lie in [0, {k_max}].")
    has_poly = (k_max > 0) and (n > 0) and bool(np.any(sc > 0))
    return sn, sc, has_poly


def _numpy_poly_ring_chunk(
    Pc: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
    c2: np.ndarray,
    c3: np.ndarray,
    sn: np.ndarray,
    sc: np.ndarray,
    rc_c: np.ndarray | float,
) -> np.ndarray:
    """Evaluate bent-ring velocities [1/m] on one row chunk *Pc* in numpy."""
    k_max = int(sn.shape[2])
    cnt_r = sc[:, 0]
    cnt_l = sc[:, 1]

    # Leading bound segment c0 -> c1 and trailing spanwise segment c2 -> c3.
    v = segment_velocity(Pc, c0, c1, rc_c)
    v += segment_velocity(Pc, c2, c3, rc_c)

    # Right side polyline: c1 -> sn[:, 0, 0..cnt_r-1] -> c2.
    end0_r = np.where((cnt_r > 0)[:, None], sn[:, 0, 0, :], c2)
    v += segment_velocity(Pc, c1, end0_r, rc_c)
    for s in range(1, k_max + 1):
        idx = np.flatnonzero(cnt_r >= s)
        if idx.size == 0:
            break
        seg_a = sn[idx, 0, s - 1, :]
        if s < k_max:
            seg_b = np.where((cnt_r[idx] > s)[:, None], sn[idx, 0, s, :], c2[idx])
        else:
            seg_b = c2[idx]
        v[:, idx, :] += segment_velocity(Pc, seg_a, seg_b, _select_rc_cols(rc_c, idx))

    # Left side polyline: c3 -> sn[:, 1, 0..cnt_l-1] -> c0.
    end0_l = np.where((cnt_l > 0)[:, None], sn[:, 1, 0, :], c0)
    v += segment_velocity(Pc, c3, end0_l, rc_c)
    for s in range(1, k_max + 1):
        idx = np.flatnonzero(cnt_l >= s)
        if idx.size == 0:
            break
        seg_a = sn[idx, 1, s - 1, :]
        if s < k_max:
            seg_b = np.where((cnt_l[idx] > s)[:, None], sn[idx, 1, s, :], c0[idx])
        else:
            seg_b = c0[idx]
        v[:, idx, :] += segment_velocity(Pc, seg_a, seg_b, _select_rc_cols(rc_c, idx))

    return v


def _broadcast_wake_dir(wake_dir: np.ndarray, n: int) -> np.ndarray:
    """Validate and broadcast *wake_dir* to shape ``(n, 3)``."""
    wd = np.asarray(wake_dir, dtype=np.float64)
    if wd.ndim == 1:
        if wd.shape != (3,):
            raise ValueError(f"Expected wake_dir of shape (3,) or ({n}, 3), got {wd.shape}.")
        return np.ascontiguousarray(np.broadcast_to(wd, (n, 3)))
    if wd.ndim == 2 and wd.shape == (n, 3):
        return np.ascontiguousarray(wd)
    raise ValueError(f"Expected wake_dir of shape (3,) or ({n}, 3), got {wd.shape}.")


def ring_velocity(
    P: np.ndarray,
    corners: np.ndarray,
    rc: np.ndarray | float,
    *,
    side_nodes: np.ndarray | None = None,
    side_counts: np.ndarray | None = None,
    backend: str | None = None,
) -> np.ndarray:
    """Compute velocity at *P* from unit-circulation vortex rings.

    Each ring ``j`` has four end corners ``corners[j, 0..3]`` [m] ordered
    cyclically. When *side_nodes* and *side_counts* are None (or all counts
    are zero), each ring uses the four-corner fast path ``0 -> 1 -> 2 -> 3 ->
    0``. Otherwise the right side edge ``1 -> 2`` passes through
    ``side_nodes[j, 0, :side_counts[j, 0]]`` and the left side edge ``3 -> 0``
    passes through ``side_nodes[j, 1, :side_counts[j, 1]]``.

    Parameters
    ----------
    P : numpy.ndarray
        Evaluation points [m], shape ``(M, 3)``.
    corners : numpy.ndarray
        Ring corner coordinates [m], shape ``(N, 4, 3)``.
    rc : numpy.ndarray or float
        Core radius [m], scalar, shape ``(N,)`` or shape ``(M, N)``.
    side_nodes : numpy.ndarray or None, optional
        Padded intermediate nodes along the right (axis-1 index 0) and left
        (axis-1 index 1) side polylines [m], shape ``(N, 2, K, 3)``.
    side_counts : numpy.ndarray or None, optional
        Number of active intermediate nodes per ring side edge, shape
        ``(N, 2)`` int64.
    backend : str or None, optional
        Kernel backend (``"auto"``, ``"numba"`` or ``"numpy"``). None follows
        :mod:`ventorum.aero.vortex`.

    Returns
    -------
    numpy.ndarray
        Induced velocity per unit circulation [m/s per m^2/s], shape
        ``(M, N, 3)``.
    """
    P_arr = np.asarray(P, dtype=np.float64)
    c_arr = np.asarray(corners, dtype=np.float64)
    if P_arr.ndim != 2 or P_arr.shape[1] != 3:
        raise ValueError(f"Expected P of shape (M, 3), got {P_arr.shape}.")
    if c_arr.ndim != 3 or c_arr.shape[1:] != (4, 3):
        raise ValueError(f"Expected corners of shape (N, 4, 3), got {c_arr.shape}.")
    m, n = P_arr.shape[0], c_arr.shape[0]
    sn, sc, has_poly = _validate_side_polylines(n, side_nodes, side_counts)
    chosen = _ring_backend(backend)
    if m == 0 or n == 0:
        return np.empty((m, n, 3), dtype=np.float64)

    if chosen == "numba":
        rc2, rc_is_2d = _nb.pack_rc2(rc, m, n)
        if has_poly:
            return _nb.ring_poly_velocity_kernel(
                np.ascontiguousarray(P_arr),
                np.ascontiguousarray(c_arr),
                np.ascontiguousarray(sn),
                np.ascontiguousarray(sc),
                rc2,
                rc_is_2d,
            )
        return _nb.ring_velocity_kernel(
            np.ascontiguousarray(P_arr),
            np.ascontiguousarray(c_arr),
            rc2,
            rc_is_2d,
        )

    c0, c1, c2, c3 = c_arr[:, 0], c_arr[:, 1], c_arr[:, 2], c_arr[:, 3]
    out = np.empty((m, n, 3), dtype=np.float64)
    step = _chunk_rows(m, n)
    for i0 in range(0, m, step):
        i1 = min(m, i0 + step)
        Pc = P_arr[i0:i1]
        rc_c = _slice_rc(rc, i0, i1)
        if has_poly:
            out[i0:i1] = _numpy_poly_ring_chunk(Pc, c0, c1, c2, c3, sn, sc, rc_c)
        else:
            v = segment_velocity(Pc, c0, c1, rc_c)
            v += segment_velocity(Pc, c1, c2, rc_c)
            v += segment_velocity(Pc, c2, c3, rc_c)
            v += segment_velocity(Pc, c3, c0, rc_c)
            out[i0:i1] = v
    return out


def ring_normal_influence(
    P: np.ndarray,
    normals: np.ndarray,
    corners: np.ndarray,
    rc: np.ndarray | float,
    *,
    side_nodes: np.ndarray | None = None,
    side_counts: np.ndarray | None = None,
    backend: str | None = None,
) -> np.ndarray:
    """Compute normal-velocity influence of unit-circulation vortex rings.

    Entry ``(i, j)`` is the dot product of the velocity induced by ring ``j``
    at ``P[i]`` with ``normals[i]``. Supports bent side polylines via
    *side_nodes* and *side_counts* while retaining the four-corner fast path
    when no side edge has intermediate nodes.

    Parameters
    ----------
    P : numpy.ndarray
        Evaluation points [m], shape ``(M, 3)``.
    normals : numpy.ndarray
        Unit normal vectors at *P* [-], shape ``(M, 3)``.
    corners : numpy.ndarray
        Ring corner coordinates [m], shape ``(N, 4, 3)``.
    rc : numpy.ndarray or float
        Core radius [m], scalar, shape ``(N,)`` or shape ``(M, N)``.
    side_nodes : numpy.ndarray or None, optional
        Padded intermediate nodes along the right (axis-1 index 0) and left
        (axis-1 index 1) side polylines [m], shape ``(N, 2, K, 3)``.
    side_counts : numpy.ndarray or None, optional
        Number of active intermediate nodes per ring side edge, shape
        ``(N, 2)`` int64.
    backend : str or None, optional
        Kernel backend (``"auto"``, ``"numba"`` or ``"numpy"``). None follows
        :mod:`ventorum.aero.vortex`.

    Returns
    -------
    numpy.ndarray
        Normal-velocity influence matrix [m/s per m^2/s], shape ``(M, N)``.
    """
    P_arr = np.asarray(P, dtype=np.float64)
    n_arr = np.asarray(normals, dtype=np.float64)
    c_arr = np.asarray(corners, dtype=np.float64)
    if P_arr.ndim != 2 or P_arr.shape[1] != 3:
        raise ValueError(f"Expected P of shape (M, 3), got {P_arr.shape}.")
    if n_arr.shape != P_arr.shape:
        raise ValueError(f"Expected normals of shape {P_arr.shape}, got {n_arr.shape}.")
    if c_arr.ndim != 3 or c_arr.shape[1:] != (4, 3):
        raise ValueError(f"Expected corners of shape (N, 4, 3), got {c_arr.shape}.")
    m, n = P_arr.shape[0], c_arr.shape[0]
    sn, sc, has_poly = _validate_side_polylines(n, side_nodes, side_counts)
    chosen = _ring_backend(backend)
    if m == 0 or n == 0:
        return np.empty((m, n), dtype=np.float64)

    if chosen == "numba":
        rc2, rc_is_2d = _nb.pack_rc2(rc, m, n)
        if has_poly:
            return _nb.ring_poly_normal_influence_kernel(
                np.ascontiguousarray(P_arr),
                np.ascontiguousarray(n_arr),
                np.ascontiguousarray(c_arr),
                np.ascontiguousarray(sn),
                np.ascontiguousarray(sc),
                rc2,
                rc_is_2d,
            )
        return _nb.ring_normal_influence_kernel(
            np.ascontiguousarray(P_arr),
            np.ascontiguousarray(n_arr),
            np.ascontiguousarray(c_arr),
            rc2,
            rc_is_2d,
        )

    c0, c1, c2, c3 = c_arr[:, 0], c_arr[:, 1], c_arr[:, 2], c_arr[:, 3]
    out = np.empty((m, n), dtype=np.float64)
    step = _chunk_rows(m, n)
    for i0 in range(0, m, step):
        i1 = min(m, i0 + step)
        Pc = P_arr[i0:i1]
        nc = n_arr[i0:i1, None, :]
        rc_c = _slice_rc(rc, i0, i1)
        if has_poly:
            v = _numpy_poly_ring_chunk(Pc, c0, c1, c2, c3, sn, sc, rc_c)
        else:
            v = segment_velocity(Pc, c0, c1, rc_c)
            v += segment_velocity(Pc, c1, c2, rc_c)
            v += segment_velocity(Pc, c2, c3, rc_c)
            v += segment_velocity(Pc, c3, c0, rc_c)
        out[i0:i1] = (
            v[..., 0] * nc[..., 0]
            + v[..., 1] * nc[..., 1]
            + v[..., 2] * nc[..., 2]
        )
    return out


def steady_wake_velocity(
    P: np.ndarray,
    te_left: np.ndarray,
    te_right: np.ndarray,
    wake_dir: np.ndarray,
    rc: np.ndarray | float,
    *,
    backend: str | None = None,
) -> np.ndarray:
    """Compute velocity at *P* from the steady semi-infinite wake closure.

    Each strip ``j`` has a semi-infinite steady wake horseshoe with leading
    spanwise segment ``te_left[j] -> te_right[j]`` (canceling the trailing
    segment ``te_right[j] -> te_left[j]`` of the closed trailing-edge ring),
    a positive semi-infinite leg from ``te_right[j]`` along ``wake_dir[j]``,
    and a negative semi-infinite leg from ``te_left[j]`` along ``wake_dir[j]``.

    Parameters
    ----------
    P : numpy.ndarray
        Evaluation points [m], shape ``(M, 3)``.
    te_left, te_right : numpy.ndarray
        Left and right trailing-edge closure corners [m], shape ``(N_s, 3)``.
    wake_dir : numpy.ndarray
        Unit wake direction [-], shape ``(3,)`` or ``(N_s, 3)``.
    rc : numpy.ndarray or float
        Core radius [m], scalar, shape ``(N_s,)`` or shape ``(M, N_s)``.
    backend : str or None, optional
        Kernel backend (``"auto"``, ``"numba"`` or ``"numpy"``). None follows
        :mod:`ventorum.aero.vortex`.

    Returns
    -------
    numpy.ndarray
        Induced velocity per unit circulation [m/s per m^2/s], shape
        ``(M, N_s, 3)``.
    """
    P_arr = np.asarray(P, dtype=np.float64)
    tl = np.asarray(te_left, dtype=np.float64)
    tr = np.asarray(te_right, dtype=np.float64)
    if P_arr.ndim != 2 or P_arr.shape[1] != 3:
        raise ValueError(f"Expected P of shape (M, 3), got {P_arr.shape}.")
    if tl.ndim != 2 or tl.shape[1] != 3 or tr.shape != tl.shape:
        raise ValueError(f"Expected te_left and te_right of shape (N_s, 3), got {tl.shape} and {tr.shape}.")
    m, n = P_arr.shape[0], tl.shape[0]
    wd = _broadcast_wake_dir(wake_dir, n)
    chosen = _ring_backend(backend)
    if m == 0 or n == 0:
        return np.empty((m, n, 3), dtype=np.float64)

    if chosen == "numba":
        rc2, rc_is_2d = _nb.pack_rc2(rc, m, n)
        return _nb.steady_wake_velocity_kernel(
            np.ascontiguousarray(P_arr),
            np.ascontiguousarray(tl),
            np.ascontiguousarray(tr),
            wd,
            rc2,
            rc_is_2d,
        )

    out = np.empty((m, n, 3), dtype=np.float64)
    step = _chunk_rows(m, n)
    for i0 in range(0, m, step):
        i1 = min(m, i0 + step)
        Pc = P_arr[i0:i1]
        rc_c = _slice_rc(rc, i0, i1)
        v = segment_velocity(Pc, tl, tr, rc_c)
        v += ray_velocity(Pc, tr, wd, rc_c)
        v -= ray_velocity(Pc, tl, wd, rc_c)
        out[i0:i1] = v
    return out


def steady_wake_normal_influence(
    P: np.ndarray,
    normals: np.ndarray,
    te_left: np.ndarray,
    te_right: np.ndarray,
    wake_dir: np.ndarray,
    rc: np.ndarray | float,
    *,
    backend: str | None = None,
) -> np.ndarray:
    """Compute normal-velocity influence of the steady semi-infinite wake closure.

    Parameters
    ----------
    P : numpy.ndarray
        Evaluation points [m], shape ``(M, 3)``.
    normals : numpy.ndarray
        Unit normal vectors at *P* [-], shape ``(M, 3)``.
    te_left, te_right : numpy.ndarray
        Left and right trailing-edge closure corners [m], shape ``(N_s, 3)``.
    wake_dir : numpy.ndarray
        Unit wake direction [-], shape ``(3,)`` or ``(N_s, 3)``.
    rc : numpy.ndarray or float
        Core radius [m], scalar, shape ``(N_s,)`` or shape ``(M, N_s)``.
    backend : str or None, optional
        Kernel backend (``"auto"``, ``"numba"`` or ``"numpy"``). None follows
        :mod:`ventorum.aero.vortex`.

    Returns
    -------
    numpy.ndarray
        Normal-velocity influence matrix [m/s per m^2/s], shape ``(M, N_s)``.
    """
    P_arr = np.asarray(P, dtype=np.float64)
    n_arr = np.asarray(normals, dtype=np.float64)
    tl = np.asarray(te_left, dtype=np.float64)
    tr = np.asarray(te_right, dtype=np.float64)
    if P_arr.ndim != 2 or P_arr.shape[1] != 3:
        raise ValueError(f"Expected P of shape (M, 3), got {P_arr.shape}.")
    if n_arr.shape != P_arr.shape:
        raise ValueError(f"Expected normals of shape {P_arr.shape}, got {n_arr.shape}.")
    if tl.ndim != 2 or tl.shape[1] != 3 or tr.shape != tl.shape:
        raise ValueError(f"Expected te_left and te_right of shape (N_s, 3), got {tl.shape} and {tr.shape}.")
    m, n = P_arr.shape[0], tl.shape[0]
    wd = _broadcast_wake_dir(wake_dir, n)
    chosen = _ring_backend(backend)
    if m == 0 or n == 0:
        return np.empty((m, n), dtype=np.float64)

    if chosen == "numba":
        rc2, rc_is_2d = _nb.pack_rc2(rc, m, n)
        return _nb.steady_wake_normal_influence_kernel(
            np.ascontiguousarray(P_arr),
            np.ascontiguousarray(n_arr),
            np.ascontiguousarray(tl),
            np.ascontiguousarray(tr),
            wd,
            rc2,
            rc_is_2d,
        )

    out = np.empty((m, n), dtype=np.float64)
    step = _chunk_rows(m, n)
    for i0 in range(0, m, step):
        i1 = min(m, i0 + step)
        Pc = P_arr[i0:i1]
        nc = n_arr[i0:i1, None, :]
        rc_c = _slice_rc(rc, i0, i1)
        v = segment_velocity(Pc, tl, tr, rc_c)
        v += ray_velocity(Pc, tr, wd, rc_c)
        v -= ray_velocity(Pc, tl, wd, rc_c)
        out[i0:i1] = (
            v[..., 0] * nc[..., 0]
            + v[..., 1] * nc[..., 1]
            + v[..., 2] * nc[..., 2]
        )
    return out
