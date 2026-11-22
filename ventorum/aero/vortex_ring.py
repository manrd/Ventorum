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
    backend: str | None = None,
) -> np.ndarray:
    """Compute velocity at *P* from unit-circulation quadrilateral vortex rings.

    Each ring ``j`` has four corners ``corners[j, 0..3]`` [m] ordered cyclically
    so the four straight segments are ``0 -> 1``, ``1 -> 2``, ``2 -> 3`` and
    ``3 -> 0``.

    Parameters
    ----------
    P : numpy.ndarray
        Evaluation points [m], shape ``(M, 3)``.
    corners : numpy.ndarray
        Ring corner coordinates [m], shape ``(N, 4, 3)``.
    rc : numpy.ndarray or float
        Core radius [m], scalar, shape ``(N,)`` or shape ``(M, N)``.
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
    chosen = _ring_backend(backend)
    if m == 0 or n == 0:
        return np.empty((m, n, 3), dtype=np.float64)

    if chosen == "numba":
        rc2, rc_is_2d = _nb.pack_rc2(rc, m, n)
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
    backend: str | None = None,
) -> np.ndarray:
    """Compute normal-velocity influence of unit-circulation vortex rings.

    Entry ``(i, j)`` is the dot product of the velocity induced by ring ``j``
    at ``P[i]`` with ``normals[i]``.

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
    chosen = _ring_backend(backend)
    if m == 0 or n == 0:
        return np.empty((m, n), dtype=np.float64)

    if chosen == "numba":
        rc2, rc_is_2d = _nb.pack_rc2(rc, m, n)
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
