# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Cython acceleration interface and runtime management for Ventorum.

Manages Cython C-extension module loading, runtime enable/disable toggles,
and safe Python wrappers with contiguous memory layout guarantees.
"""

from __future__ import annotations

import os
import warnings
from typing import Any
import numpy as np

from ventorum.legacy.core.constants import VORTEX_CORE_RADIUS

try:
    from ventorum.legacy.aero import cython_kernels
    HAS_CYTHON = True
except ImportError:
    cython_kernels = None
    HAS_CYTHON = False

# Global runtime toggle (can be overridden via environment variable VENTORUM_USE_CYTHON)
_CYTHON_ENABLED = HAS_CYTHON
if os.environ.get("VENTORUM_USE_CYTHON", "1").lower() in ("0", "false", "no", "off"):
    _CYTHON_ENABLED = False


def has_cython() -> bool:
    """Return True if Cython compiled C extensions are available."""
    return HAS_CYTHON


def is_cython_enabled() -> bool:
    """Return True if Cython acceleration is currently active."""
    return _CYTHON_ENABLED and HAS_CYTHON


def set_cython_enabled(enabled: bool) -> None:
    """Globally enable or disable Cython acceleration in Ventorum."""
    global _CYTHON_ENABLED
    if enabled and not HAS_CYTHON:
        warnings.warn(
            "Cannot enable Cython: compiled C extension ventorum.legacy.aero.cython_kernels is not available.",
            RuntimeWarning,
            stacklevel=2,
        )
        _CYTHON_ENABLED = False
    else:
        _CYTHON_ENABLED = bool(enabled)


# ═══════════════════════════════════════════════════════════════════════════════
# High-Level Safe Wrappers
# ═══════════════════════════════════════════════════════════════════════════════

def cython_horseshoe_velocity_matrix(
    cp: np.ndarray,
    nl: np.ndarray,
    nr: np.ndarray,
    trailing_dir: np.ndarray,
    gamma: float = 1.0,
    rc: float = VORTEX_CORE_RADIUS,
    return_trailing: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compute 3D horseshoe vortex induced velocities using compiled Cython C extension."""
    if not is_cython_enabled():
        raise RuntimeError("Cython is not enabled or available.")

    cp_arr = np.ascontiguousarray(cp, dtype=np.float64)
    nl_arr = np.ascontiguousarray(nl, dtype=np.float64)
    nr_arr = np.ascontiguousarray(nr, dtype=np.float64)
    td_arr = np.ascontiguousarray(trailing_dir, dtype=np.float64)

    v_tot, v_trail = cython_kernels.cy_horseshoe_velocity_matrix(
        cp_arr, nl_arr, nr_arr, td_arr, float(gamma), float(rc)
    )
    if return_trailing:
        return v_tot, v_trail
    return v_tot


def cython_aic_matrix(
    cp: np.ndarray,
    nl: np.ndarray,
    nr: np.ndarray,
    normals: np.ndarray,
    trailing_dir: np.ndarray,
    gamma: float = 1.0,
    rc: float = VORTEX_CORE_RADIUS,
) -> np.ndarray:
    """Compute normalwash AIC matrix directly in C via Cython."""
    if not is_cython_enabled():
        raise RuntimeError("Cython is not enabled or available.")

    cp_arr = np.ascontiguousarray(cp, dtype=np.float64)
    nl_arr = np.ascontiguousarray(nl, dtype=np.float64)
    nr_arr = np.ascontiguousarray(nr, dtype=np.float64)
    norm_arr = np.ascontiguousarray(normals, dtype=np.float64)
    td_arr = np.ascontiguousarray(trailing_dir, dtype=np.float64)

    return cython_kernels.cy_aic_matrix(
        cp_arr, nl_arr, nr_arr, norm_arr, td_arr, float(gamma), float(rc)
    )


def cython_trailing_velocity_kernel(
    r1: np.ndarray,
    r2: np.ndarray,
    r1_norm_reg: np.ndarray,
    r2_norm_reg: np.ndarray,
    trailing_dir: np.ndarray,
    gamma: float = 1.0,
    rc: float = VORTEX_CORE_RADIUS,
) -> np.ndarray:
    """Evaluate trailing-vortex downwash tensor from precomputed relative distances via Cython."""
    if not is_cython_enabled():
        raise RuntimeError("Cython is not enabled or available.")

    r1_arr = np.ascontiguousarray(r1, dtype=np.float64)
    r2_arr = np.ascontiguousarray(r2, dtype=np.float64)
    r1_reg = np.ascontiguousarray(r1_norm_reg, dtype=np.float64)
    r2_reg = np.ascontiguousarray(r2_norm_reg, dtype=np.float64)
    td_arr = np.ascontiguousarray(trailing_dir, dtype=np.float64)

    return cython_kernels.cy_trailing_velocity_kernel(
        r1_arr, r2_arr, r1_reg, r2_reg, td_arr, float(gamma), float(rc)
    )


def cython_bound_velocity_kernel(
    r1: np.ndarray,
    r2: np.ndarray,
    r0: np.ndarray,
    rc: float = VORTEX_CORE_RADIUS,
    gamma: float = 1.0,
) -> np.ndarray:
    """Evaluate finite bound-vortex kernel between nodes A and B via Cython."""
    if not is_cython_enabled():
        raise RuntimeError("Cython is not enabled or available.")

    r1_arr = np.ascontiguousarray(r1, dtype=np.float64)
    r2_arr = np.ascontiguousarray(r2, dtype=np.float64)
    r0_arr = np.ascontiguousarray(r0, dtype=np.float64)

    return cython_kernels.cy_bound_velocity_kernel(
        r1_arr, r2_arr, r0_arr, float(rc), float(gamma)
    )


def cython_nonlinear_relaxation_loop(
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
) -> tuple[np.ndarray, int, bool, list[float]]:
    """Execute nonlinear polar relaxation loop in compiled Cython C code."""
    if not is_cython_enabled():
        raise RuntimeError("Cython is not enabled or available.")

    vz_arr = np.ascontiguousarray(V_z, dtype=np.float64)
    gam_arr = np.ascontiguousarray(Gamma_init, dtype=np.float64)
    ag_arr = np.ascontiguousarray(alpha_geom, dtype=np.float64)
    hc_arr = np.ascontiguousarray(half_vinf_chords, dtype=np.float64)
    alpha_tbl = np.ascontiguousarray(alpha_table, dtype=np.float64)
    cl_tbl = np.ascontiguousarray(cl_table, dtype=np.float64)

    return cython_kernels.cy_nonlinear_relaxation_loop(
        vz_arr, gam_arr, float(inv_vinf), ag_arr, hc_arr,
        alpha_tbl, cl_tbl, float(omega), int(max_iter), float(tol)
    )
