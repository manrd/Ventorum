# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Unified acceleration manager and dispatcher for Ventorum.

Coordinates between:
1. Pure NumPy (baseline vectorized execution).
2. Numba JIT (multi-threaded AVX parallel CPU execution).
3. Cython (compiled C extension).
4. Numba + Cython (hybrid cooperative execution).

Allows seamless runtime switching, benchmarking, and automatic best-available dispatch.
"""

from __future__ import annotations

import os
from typing import Any
import numpy as np

from ventorum.legacy.core.constants import VORTEX_CORE_RADIUS
from ventorum.legacy.aero.numba_kernels import (
    has_numba,
    is_numba_enabled,
    set_numba_enabled,
    _numba_horseshoe_velocity_matrix,
    _numba_aic_matrix,
    _numba_trailing_velocity_kernel,
    _numba_bound_velocity_kernel,
    _numba_nonlinear_relaxation_loop,
)
from ventorum.legacy.aero.cython_accel import (
    has_cython,
    is_cython_enabled,
    set_cython_enabled,
    cython_horseshoe_velocity_matrix,
    cython_aic_matrix,
    cython_trailing_velocity_kernel,
    cython_bound_velocity_kernel,
    cython_nonlinear_relaxation_loop,
)

# Active acceleration preference: 'auto', 'numba', 'cython', 'numba+cython', 'numpy'
_ACTIVE_ACCELERATION: str = "auto"


def get_active_acceleration() -> str:
    """Return the current acceleration mode string."""
    global _ACTIVE_ACCELERATION
    if _ACTIVE_ACCELERATION == "auto":
        nb = is_numba_enabled()
        cy = is_cython_enabled()
        if nb and cy:
            return "numba+cython"
        elif nb:
            return "numba"
        elif cy:
            return "cython"
        return "numpy"
    return _ACTIVE_ACCELERATION


def set_acceleration(mode: str) -> None:
    """Set the active acceleration backend.

    Parameters
    ----------
    mode : str
        One of:
        - 'auto': Enable all available accelerators.
        - 'numba': Enable Numba JIT only.
        - 'cython': Enable Cython compiled C extension only.
        - 'numba+cython' or 'hybrid': Enable both Numba and Cython.
        - 'numpy' or 'none': Disable compiled acceleration and run pure NumPy.
    """
    global _ACTIVE_ACCELERATION
    m = mode.lower().strip()

    if m in ("numpy", "none", "baseline"):
        _ACTIVE_ACCELERATION = "numpy"
        set_numba_enabled(False)
        set_cython_enabled(False)
    elif m == "numba":
        _ACTIVE_ACCELERATION = "numba"
        set_numba_enabled(True)
        set_cython_enabled(False)
    elif m == "cython":
        _ACTIVE_ACCELERATION = "cython"
        set_numba_enabled(False)
        set_cython_enabled(True)
    elif m in ("numba+cython", "cython+numba", "hybrid", "both"):
        _ACTIVE_ACCELERATION = "numba+cython"
        set_numba_enabled(True)
        set_cython_enabled(True)
    elif m == "auto":
        _ACTIVE_ACCELERATION = "auto"
        set_numba_enabled(has_numba())
        set_cython_enabled(has_cython())
    else:
        raise ValueError(
            f"Unknown acceleration mode '{mode}'. Choose from 'auto', 'numba', 'cython', 'numba+cython', or 'numpy'."
        )


def dispatch_horseshoe_velocity_matrix(
    cp: np.ndarray,
    nl: np.ndarray,
    nr: np.ndarray,
    trailing_dir: np.ndarray,
    gamma: float = 1.0,
    rc: float = VORTEX_CORE_RADIUS,
    return_trailing: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray] | None:
    """Dispatch horseshoe vortex velocity matrix calculation to the optimal accelerator.

    In auto/hybrid mode, Numba is prioritized because multi-threaded parallel AVX2 execution
    is up to 7.6x faster than single-threaded Cython and up to 100x faster than NumPy.
    """
    mode = get_active_acceleration()

    # Numba multi-core parallel execution is fastest for full NxM tensor assembly
    if mode in ("numba", "numba+cython", "auto") and is_numba_enabled():
        cp_arr = np.ascontiguousarray(cp, dtype=np.float64)
        nl_arr = np.ascontiguousarray(nl, dtype=np.float64)
        nr_arr = np.ascontiguousarray(nr, dtype=np.float64)
        td_arr = np.ascontiguousarray(trailing_dir, dtype=np.float64)
        v_tot, v_trail = _numba_horseshoe_velocity_matrix(
            cp_arr, nl_arr, nr_arr, td_arr, float(gamma), float(rc)
        )
        if return_trailing:
            return v_tot, v_trail
        return v_tot

    if mode in ("cython", "numba+cython", "auto") and is_cython_enabled():
        return cython_horseshoe_velocity_matrix(
            cp, nl, nr, trailing_dir, gamma=gamma, rc=rc, return_trailing=return_trailing
        )

    return None


def dispatch_aic_matrix(
    cp: np.ndarray,
    nl: np.ndarray,
    nr: np.ndarray,
    normals: np.ndarray,
    trailing_dir: np.ndarray,
    gamma: float = 1.0,
    rc: float = VORTEX_CORE_RADIUS,
) -> np.ndarray | None:
    """Dispatch direct AIC normalwash evaluation to the optimal accelerator."""
    mode = get_active_acceleration()

    # Numba multi-core parallel execution is fastest for NxM grid evaluation
    if mode in ("numba", "numba+cython", "auto") and is_numba_enabled():
        cp_arr = np.ascontiguousarray(cp, dtype=np.float64)
        nl_arr = np.ascontiguousarray(nl, dtype=np.float64)
        nr_arr = np.ascontiguousarray(nr, dtype=np.float64)
        norm_arr = np.ascontiguousarray(normals, dtype=np.float64)
        td_arr = np.ascontiguousarray(trailing_dir, dtype=np.float64)
        return _numba_aic_matrix(cp_arr, nl_arr, nr_arr, norm_arr, td_arr, float(gamma), float(rc))

    if mode in ("cython", "numba+cython", "auto") and is_cython_enabled():
        return cython_aic_matrix(
            cp, nl, nr, normals, trailing_dir, gamma=gamma, rc=rc
        )

    return None


def dispatch_trailing_velocity_kernel(
    r1: np.ndarray,
    r2: np.ndarray,
    r1_norm_reg: np.ndarray,
    r2_norm_reg: np.ndarray,
    trailing_dir: np.ndarray,
    gamma: float = 1.0,
    rc: float = VORTEX_CORE_RADIUS,
) -> np.ndarray | None:
    """Dispatch trailing-vortex downwash tensor evaluation to active accelerator."""
    mode = get_active_acceleration()

    # In hybrid mode ('numba+cython'), delegate kernel evaluation to Cython's compiled C routines
    if mode in ("cython", "numba+cython") and is_cython_enabled():
        return cython_trailing_velocity_kernel(
            r1, r2, r1_norm_reg, r2_norm_reg, trailing_dir, gamma=gamma, rc=rc
        )

    if mode in ("numba", "auto") and is_numba_enabled():
        r1_arr = np.ascontiguousarray(r1, dtype=np.float64)
        r2_arr = np.ascontiguousarray(r2, dtype=np.float64)
        r1_reg = np.ascontiguousarray(r1_norm_reg, dtype=np.float64)
        r2_reg = np.ascontiguousarray(r2_norm_reg, dtype=np.float64)
        td_arr = np.ascontiguousarray(trailing_dir, dtype=np.float64)
        return _numba_trailing_velocity_kernel(
            r1_arr, r2_arr, r1_reg, r2_reg, td_arr, float(gamma), float(rc)
        )

    if is_cython_enabled():
        return cython_trailing_velocity_kernel(
            r1, r2, r1_norm_reg, r2_norm_reg, trailing_dir, gamma=gamma, rc=rc
        )

    return None


def dispatch_bound_velocity_kernel(
    r1: np.ndarray,
    r2: np.ndarray,
    r0: np.ndarray,
    rc: float = VORTEX_CORE_RADIUS,
    gamma: float = 1.0,
) -> np.ndarray | None:
    """Dispatch finite bound-vortex kernel evaluation to active accelerator."""
    mode = get_active_acceleration()

    # In hybrid mode ('numba+cython'), delegate bound vortex evaluation to Cython's compiled C routines
    if mode in ("cython", "numba+cython") and is_cython_enabled():
        return cython_bound_velocity_kernel(r1, r2, r0, rc=rc, gamma=gamma)

    if mode in ("numba", "auto") and is_numba_enabled():
        r1_arr = np.ascontiguousarray(r1, dtype=np.float64)
        r2_arr = np.ascontiguousarray(r2, dtype=np.float64)
        r0_arr = np.ascontiguousarray(r0, dtype=np.float64)
        return _numba_bound_velocity_kernel(
            r1_arr, r2_arr, r0_arr, float(rc), float(gamma)
        )

    if is_cython_enabled():
        return cython_bound_velocity_kernel(r1, r2, r0, rc=rc, gamma=gamma)

    return None


def dispatch_nonlinear_relaxation_loop(
    V_z: np.ndarray,
    Gamma_init: np.ndarray,
    inv_vinf: float,
    alpha_geom: np.ndarray,
    half_vinf_chords: np.ndarray,
    alpha_table: np.ndarray,
    cl_table: np.ndarray,
    omega: float,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, int, bool, list[float]] | None:
    """Dispatch nonlinear polar relaxation loop to active accelerator.

    In auto/hybrid mode, Cython is prioritized over Numba because raw C scalar loops
    with direct pointer table lookups are 1.45x faster than Numba JIT (2.32 ms vs 3.36 ms).
    """
    mode = get_active_acceleration()

    # Prioritize Cython for sequential relaxation loops
    if mode in ("cython", "numba+cython", "auto") and is_cython_enabled():
        return cython_nonlinear_relaxation_loop(
            V_z, Gamma_init, inv_vinf, alpha_geom, half_vinf_chords,
            alpha_table, cl_table, omega, max_iter, tol
        )

    if mode in ("numba", "numba+cython", "auto") and is_numba_enabled():
        vz_arr = np.ascontiguousarray(V_z, dtype=np.float64)
        gam_arr = np.ascontiguousarray(Gamma_init, dtype=np.float64)
        ag_arr = np.ascontiguousarray(alpha_geom, dtype=np.float64)
        hc_arr = np.ascontiguousarray(half_vinf_chords, dtype=np.float64)
        alpha_tbl = np.ascontiguousarray(alpha_table, dtype=np.float64)
        cl_tbl = np.ascontiguousarray(cl_table, dtype=np.float64)

        return _numba_nonlinear_relaxation_loop(
            vz_arr, gam_arr, float(inv_vinf), ag_arr, hc_arr,
            alpha_tbl, cl_tbl, float(omega), int(max_iter), float(tol)
        )

    return None


class acceleration_context:
    """Context manager for temporarily executing under a specified acceleration mode.

    Example
    -------
    >>> with acceleration_context('cython'):
    ...     res = vt.analyze(wing)
    """

    def __init__(self, mode: str):
        self.target_mode = mode
        self.prev_mode = get_active_acceleration()
        self.prev_numba = is_numba_enabled()
        self.prev_cython = is_cython_enabled()

    def __enter__(self):
        set_acceleration(self.target_mode)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _ACTIVE_ACCELERATION
        _ACTIVE_ACCELERATION = self.prev_mode
        set_numba_enabled(self.prev_numba)
        set_cython_enabled(self.prev_cython)

