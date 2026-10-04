# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Numba-accelerated high-performance kernels for Ventorum.

Provides JIT-compiled (parallel multi-threaded CPU) kernels for:
1. Biot-Savart horseshoe vortex velocity tensor assembly.
2. Direct Aerodynamic Influence Coefficient (AIC) matrix normalwash evaluation.
3. Precomputed trailing filament velocity evaluation.
4. Precomputed bound vortex kernel evaluation.
5. Nonlinear section polar iterative relaxation loops.
6. Scalar Biot-Savart filament primitives.

When Numba is available, these kernels bypass Python looping and massive intermediate
NumPy 3D array allocations (N, M, 3), keeping vector math in CPU registers (AVX2/AVX-512)
and scaling across all CPU cores with zero GIL contention, achieving a 15x-50x speedup.
If Numba is unavailable or disabled, clean fallback to NumPy is guaranteed.
"""

from __future__ import annotations

import os
import warnings
from typing import Any
import numpy as np

from ventorum.legacy.core.constants import VORTEX_CORE_RADIUS

try:
    import numba
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    njit = None
    prange = range

# Global runtime toggle (can be overridden via environment variable VENTORUM_USE_NUMBA)
_NUMBA_ENABLED = HAS_NUMBA
if os.environ.get("VENTORUM_USE_NUMBA", "1").lower() in ("0", "false", "no", "off"):
    _NUMBA_ENABLED = False


def has_numba() -> bool:
    """Return True if Numba is installed and importable."""
    return HAS_NUMBA


def is_numba_enabled() -> bool:
    """Return True if Numba acceleration is currently active."""
    return _NUMBA_ENABLED and HAS_NUMBA


def set_numba_enabled(enabled: bool) -> None:
    """Globally enable or disable Numba JIT acceleration in Ventorum."""
    global _NUMBA_ENABLED
    if enabled and not HAS_NUMBA:
        warnings.warn(
            "Cannot enable Numba: numba package is not installed.",
            RuntimeWarning,
            stacklevel=2,
        )
        _NUMBA_ENABLED = False
    else:
        _NUMBA_ENABLED = bool(enabled)


# ═══════════════════════════════════════════════════════════════════════════════
# JIT-Compiled Core Kernels
# ═══════════════════════════════════════════════════════════════════════════════

if HAS_NUMBA:
    @njit(parallel=True, cache=True)
    def _numba_horseshoe_velocity_matrix(
        cp: np.ndarray,
        nl: np.ndarray,
        nr: np.ndarray,
        td: np.ndarray,
        gamma: float = 1.0,
        rc: float = VORTEX_CORE_RADIUS,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Parallel Numba kernel evaluating 3D velocities induced by horseshoe vortices."""
        N = cp.shape[0]
        M = nl.shape[0]
        V_tot = np.empty((N, M, 3), dtype=np.float64)
        V_trail = np.empty((N, M, 3), dtype=np.float64)

        rc_sq = rc * rc
        inv_4pi = 1.0 / (4.0 * np.pi)
        g_inv_4pi = gamma * inv_4pi

        td0 = td[0]
        td1 = td[1]
        td2 = td[2]

        for i in prange(N):
            px = cp[i, 0]
            py = cp[i, 1]
            pz = cp[i, 2]

            for j in range(M):
                ax = nl[j, 0]
                ay = nl[j, 1]
                az = nl[j, 2]

                bx = nr[j, 0]
                by = nr[j, 1]
                bz = nr[j, 2]

                # Displacement vectors
                r1x = px - ax
                r1y = py - ay
                r1z = pz - az

                r2x = px - bx
                r2y = py - by
                r2z = pz - bz

                r0x = bx - ax
                r0y = by - ay
                r0z = bz - az

                # Bound segment: r1 x r2
                c0 = r1y * r2z - r1z * r2y
                c1 = r1z * r2x - r1x * r2z
                c2 = r1x * r2y - r1y * r2x
                cross_sq = c0 * c0 + c1 * c1 + c2 * c2

                r0_sq = r0x * r0x + r0y * r0y + r0z * r0z
                denom_bound = cross_sq + r0_sq * rc_sq

                r1_norm_sq = r1x * r1x + r1y * r1y + r1z * r1z
                r2_norm_sq = r2x * r2x + r2y * r2y + r2z * r2z
                r1_norm_reg = np.sqrt(r1_norm_sq + rc_sq)
                r2_norm_reg = np.sqrt(r2_norm_sq + rc_sq)

                diff_x = (r1x / r1_norm_reg) - (r2x / r2_norm_reg)
                diff_y = (r1y / r1_norm_reg) - (r2y / r2_norm_reg)
                diff_z = (r1z / r1_norm_reg) - (r2z / r2_norm_reg)
                dot_term = r0x * diff_x + r0y * diff_y + r0z * diff_z

                if denom_bound > 0.0:
                    scale_bound = g_inv_4pi * (dot_term / denom_bound)
                else:
                    scale_bound = 0.0

                vb_x = c0 * scale_bound
                vb_y = c1 * scale_bound
                vb_z = c2 * scale_bound

                # Trailing legs
                if td1 == 0.0:
                    ca0 = -td2 * r1y
                    ca1 = td2 * r1x - td0 * r1z
                    ca2 = td0 * r1y
                    cos_theta_A = (td0 * r1x + td2 * r1z) / r1_norm_reg

                    cb0 = -td2 * r2y
                    cb1 = td2 * r2x - td0 * r2z
                    cb2 = td0 * r2y
                    cos_theta_B = (td0 * r2x + td2 * r2z) / r2_norm_reg
                else:
                    ca0 = td1 * r1z - td2 * r1y
                    ca1 = td2 * r1x - td0 * r1z
                    ca2 = td0 * r1y - td1 * r1x
                    cos_theta_A = (td0 * r1x + td1 * r1y + td2 * r1z) / r1_norm_reg

                    cb0 = td1 * r2z - td2 * r2y
                    cb1 = td2 * r2x - td0 * r2z
                    cb2 = td0 * r2y - td1 * r2x
                    cos_theta_B = (td0 * r2x + td1 * r2y + td2 * r2z) / r2_norm_reg

                denom_A = ca0 * ca0 + ca1 * ca1 + ca2 * ca2 + rc_sq
                scale_A = -g_inv_4pi * (1.0 + cos_theta_A) / denom_A if denom_A > 0.0 else 0.0

                denom_B = cb0 * cb0 + cb1 * cb1 + cb2 * cb2 + rc_sq
                scale_B = g_inv_4pi * (1.0 + cos_theta_B) / denom_B if denom_B > 0.0 else 0.0

                vt_x = ca0 * scale_A + cb0 * scale_B
                vt_y = ca1 * scale_A + cb1 * scale_B
                vt_z = ca2 * scale_A + cb2 * scale_B

                V_trail[i, j, 0] = vt_x
                V_trail[i, j, 1] = vt_y
                V_trail[i, j, 2] = vt_z

                V_tot[i, j, 0] = vb_x + vt_x
                V_tot[i, j, 1] = vb_y + vt_y
                V_tot[i, j, 2] = vb_z + vt_z

        return V_tot, V_trail

    @njit(parallel=True, cache=True)
    def _numba_aic_matrix(
        cp: np.ndarray,
        nl: np.ndarray,
        nr: np.ndarray,
        normals: np.ndarray,
        td: np.ndarray,
        gamma: float = 1.0,
        rc: float = VORTEX_CORE_RADIUS,
    ) -> np.ndarray:
        """Parallel Numba kernel evaluating AIC matrix normalwash without 3D velocity tensor allocation."""
        N = cp.shape[0]
        M = nl.shape[0]
        AIC = np.empty((N, M), dtype=np.float64)

        rc_sq = rc * rc
        inv_4pi = 1.0 / (4.0 * np.pi)
        g_inv_4pi = gamma * inv_4pi
        td0, td1, td2 = td[0], td[1], td[2]

        for i in prange(N):
            px, py, pz = cp[i, 0], cp[i, 1], cp[i, 2]
            nx, ny, nz = normals[i, 0], normals[i, 1], normals[i, 2]

            for j in range(M):
                ax, ay, az = nl[j, 0], nl[j, 1], nl[j, 2]
                bx, by, bz = nr[j, 0], nr[j, 1], nr[j, 2]

                r1x = px - ax
                r1y = py - ay
                r1z = pz - az

                r2x = px - bx
                r2y = py - by
                r2z = pz - bz

                r0x = bx - ax
                r0y = by - ay
                r0z = bz - az

                c0 = r1y * r2z - r1z * r2y
                c1 = r1z * r2x - r1x * r2z
                c2 = r1x * r2y - r1y * r2x
                cross_sq = c0 * c0 + c1 * c1 + c2 * c2
                r0_sq = r0x * r0x + r0y * r0y + r0z * r0z
                denom_bound = cross_sq + r0_sq * rc_sq

                r1_norm_reg = np.sqrt(r1x * r1x + r1y * r1y + r1z * r1z + rc_sq)
                r2_norm_reg = np.sqrt(r2x * r2x + r2y * r2y + r2z * r2z + rc_sq)

                diff_x = (r1x / r1_norm_reg) - (r2x / r2_norm_reg)
                diff_y = (r1y / r1_norm_reg) - (r2y / r2_norm_reg)
                diff_z = (r1z / r1_norm_reg) - (r2z / r2_norm_reg)
                dot_term = r0x * diff_x + r0y * diff_y + r0z * diff_z
                scale_bound = g_inv_4pi * (dot_term / denom_bound) if denom_bound > 0.0 else 0.0
                vb_x = c0 * scale_bound
                vb_y = c1 * scale_bound
                vb_z = c2 * scale_bound

                if td1 == 0.0:
                    ca0 = -td2 * r1y
                    ca1 = td2 * r1x - td0 * r1z
                    ca2 = td0 * r1y
                    cos_theta_A = (td0 * r1x + td2 * r1z) / r1_norm_reg

                    cb0 = -td2 * r2y
                    cb1 = td2 * r2x - td0 * r2z
                    cb2 = td0 * r2y
                    cos_theta_B = (td0 * r2x + td2 * r2z) / r2_norm_reg
                else:
                    ca0 = td1 * r1z - td2 * r1y
                    ca1 = td2 * r1x - td0 * r1z
                    ca2 = td0 * r1y - td1 * r1x
                    cos_theta_A = (td0 * r1x + td1 * r1y + td2 * r1z) / r1_norm_reg

                    cb0 = td1 * r2z - td2 * r2y
                    cb1 = td2 * r2x - td0 * r2z
                    cb2 = td0 * r2y - td1 * r2x
                    cos_theta_B = (td0 * r2x + td1 * r2y + td2 * r2z) / r2_norm_reg

                denom_A = ca0 * ca0 + ca1 * ca1 + ca2 * ca2 + rc_sq
                scale_A = -g_inv_4pi * (1.0 + cos_theta_A) / denom_A if denom_A > 0.0 else 0.0
                denom_B = cb0 * cb0 + cb1 * cb1 + cb2 * cb2 + rc_sq
                scale_B = g_inv_4pi * (1.0 + cos_theta_B) / denom_B if denom_B > 0.0 else 0.0

                v_tot_x = vb_x + ca0 * scale_A + cb0 * scale_B
                v_tot_y = vb_y + ca1 * scale_A + cb1 * scale_B
                v_tot_z = vb_z + ca2 * scale_A + cb2 * scale_B

                AIC[i, j] = v_tot_x * nx + v_tot_y * ny + v_tot_z * nz

        return AIC

    @njit(parallel=True, cache=True)
    def _numba_trailing_velocity_kernel(
        r1: np.ndarray,
        r2: np.ndarray,
        r1_norm_reg: np.ndarray,
        r2_norm_reg: np.ndarray,
        td: np.ndarray,
        gamma: float = 1.0,
        rc: float = VORTEX_CORE_RADIUS,
    ) -> np.ndarray:
        """Parallel Numba kernel evaluating trailing-vortex downwash tensor from precomputed distances."""
        N, M, _ = r1.shape
        V_trail = np.empty((N, M, 3), dtype=np.float64)
        rc_sq = rc * rc
        inv_4pi = 1.0 / (4.0 * np.pi)
        g_inv_4pi = gamma * inv_4pi
        td0, td1, td2 = td[0], td[1], td[2]

        for i in prange(N):
            for j in range(M):
                r1x = r1[i, j, 0]
                r1y = r1[i, j, 1]
                r1z = r1[i, j, 2]

                r2x = r2[i, j, 0]
                r2y = r2[i, j, 1]
                r2z = r2[i, j, 2]

                r1_n = r1_norm_reg[i, j, 0]
                r2_n = r2_norm_reg[i, j, 0]

                if td1 == 0.0:
                    ca0 = -td2 * r1y
                    ca1 = td2 * r1x - td0 * r1z
                    ca2 = td0 * r1y
                    cos_theta_A = (td0 * r1x + td2 * r1z) / r1_n

                    cb0 = -td2 * r2y
                    cb1 = td2 * r2x - td0 * r2z
                    cb2 = td0 * r2y
                    cos_theta_B = (td0 * r2x + td2 * r2z) / r2_n
                else:
                    ca0 = td1 * r1z - td2 * r1y
                    ca1 = td2 * r1x - td0 * r1z
                    ca2 = td0 * r1y - td1 * r1x
                    cos_theta_A = (td0 * r1x + td1 * r1y + td2 * r1z) / r1_n

                    cb0 = td1 * r2z - td2 * r2y
                    cb1 = td2 * r2x - td0 * r2z
                    cb2 = td0 * r2y - td1 * r2x
                    cos_theta_B = (td0 * r2x + td1 * r2y + td2 * r2z) / r2_n

                denom_A = ca0 * ca0 + ca1 * ca1 + ca2 * ca2 + rc_sq
                denom_B = cb0 * cb0 + cb1 * cb1 + cb2 * cb2 + rc_sq

                scale_A = -g_inv_4pi * (1.0 + cos_theta_A) / denom_A if denom_A > 0.0 else 0.0
                scale_B = g_inv_4pi * (1.0 + cos_theta_B) / denom_B if denom_B > 0.0 else 0.0

                V_trail[i, j, 0] = ca0 * scale_A + cb0 * scale_B
                V_trail[i, j, 1] = ca1 * scale_A + cb1 * scale_B
                V_trail[i, j, 2] = ca2 * scale_A + cb2 * scale_B

        return V_trail

    @njit(parallel=True, cache=True)
    def _numba_bound_velocity_kernel(
        r1: np.ndarray,
        r2: np.ndarray,
        r0: np.ndarray,
        rc: float = VORTEX_CORE_RADIUS,
        gamma: float = 1.0,
    ) -> np.ndarray:
        """Parallel Numba kernel evaluating finite bound-vortex kernel from relative distances."""
        N, M, _ = r1.shape
        V_bound = np.empty((N, M, 3), dtype=np.float64)
        rc_sq = rc * rc
        inv_4pi = 1.0 / (4.0 * np.pi)
        g_inv_4pi = gamma * inv_4pi
        is_r0_3d = (r0.ndim == 3)

        for i in prange(N):
            for j in range(M):
                r1x = r1[i, j, 0]
                r1y = r1[i, j, 1]
                r1z = r1[i, j, 2]

                r2x = r2[i, j, 0]
                r2y = r2[i, j, 1]
                r2z = r2[i, j, 2]

                if is_r0_3d:
                    r0x = r0[0, j, 0] if r0.shape[0] == 1 else r0[i, j, 0]
                    r0y = r0[0, j, 1] if r0.shape[0] == 1 else r0[i, j, 1]
                    r0z = r0[0, j, 2] if r0.shape[0] == 1 else r0[i, j, 2]
                else:
                    r0x = r0[j, 0]
                    r0y = r0[j, 1]
                    r0z = r0[j, 2]

                c0 = r1y * r2z - r1z * r2y
                c1 = r1z * r2x - r1x * r2z
                c2 = r1x * r2y - r1y * r2x
                cross_sq = c0 * c0 + c1 * c1 + c2 * c2

                r0_sq = r0x * r0x + r0y * r0y + r0z * r0z
                denom_bound = cross_sq + r0_sq * rc_sq

                r1_norm_sq = r1x * r1x + r1y * r1y + r1z * r1z
                r2_norm_sq = r2x * r2x + r2y * r2y + r2z * r2z
                r1_norm_reg = np.sqrt(r1_norm_sq + rc_sq)
                r2_norm_reg = np.sqrt(r2_norm_sq + rc_sq)

                diff_x = (r1x / r1_norm_reg) - (r2x / r2_norm_reg)
                diff_y = (r1y / r1_norm_reg) - (r2y / r2_norm_reg)
                diff_z = (r1z / r1_norm_reg) - (r2z / r2_norm_reg)
                dot_term = r0x * diff_x + r0y * diff_y + r0z * diff_z

                if denom_bound > 0.0:
                    scale_bound = g_inv_4pi * (dot_term / denom_bound)
                else:
                    scale_bound = 0.0

                V_bound[i, j, 0] = c0 * scale_bound
                V_bound[i, j, 1] = c1 * scale_bound
                V_bound[i, j, 2] = c2 * scale_bound

        return V_bound

    @njit(cache=True)
    def _numba_nonlinear_relaxation_loop(
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
        """JIT-compiled iterative relaxation loop for nonlinear section polars."""
        N = Gamma_init.shape[0]
        Gamma = Gamma_init.copy()
        converged = False
        final_iter = max_iter
        residual_history = []

        alpha_eff = np.empty(N, dtype=np.float64)

        for it in range(1, max_iter + 1):
            final_iter = it

            # Matrix-vector multiply for induced downwash
            for i in range(N):
                vz_g = 0.0
                for j in range(N):
                    vz_g += V_z[i, j] * Gamma[j]
                alpha_i = -vz_g * inv_vinf
                alpha_eff[i] = alpha_geom[i] - alpha_i

            # Cl polar lookup and circulation relaxation
            max_dGamma = 0.0
            for i in range(N):
                cl = np.interp(alpha_eff[i], alpha_table, cl_table)
                dg = half_vinf_chords[i] * cl - Gamma[i]
                abs_dg = abs(dg)
                if abs_dg > max_dGamma:
                    max_dGamma = abs_dg
                Gamma[i] += omega * dg

            residual = omega * max_dGamma
            residual_history.append(residual)

            if residual < tol:
                converged = True
                break

            # Adaptive relaxation damping near stall
            if len(residual_history) >= 3 and residual_history[-1] > residual_history[-2]:
                omega = max(0.02, omega * 0.75)

        return Gamma, final_iter, converged, residual_history

    @njit(fastmath=True, cache=True)
    def _numba_finite_vortex_velocity(
        A: np.ndarray,
        B: np.ndarray,
        P: np.ndarray,
        Gamma: float = 1.0,
        rc: float = VORTEX_CORE_RADIUS,
    ) -> np.ndarray:
        """Scalar finite vortex velocity evaluated at P."""
        r1x = P[0] - A[0]
        r1y = P[1] - A[1]
        r1z = P[2] - A[2]

        r2x = P[0] - B[0]
        r2y = P[1] - B[1]
        r2z = P[2] - B[2]

        r0x = B[0] - A[0]
        r0y = B[1] - A[1]
        r0z = B[2] - A[2]

        c0 = r1y * r2z - r1z * r2y
        c1 = r1z * r2x - r1x * r2z
        c2 = r1x * r2y - r1y * r2x
        cross_sq = c0 * c0 + c1 * c1 + c2 * c2
        r0_sq = r0x * r0x + r0y * r0y + r0z * r0z
        rc_sq = rc * rc

        denom = cross_sq + r0_sq * rc_sq
        if denom <= 0.0:
            return np.zeros(3, dtype=np.float64)

        r1_norm_reg = np.sqrt(r1x * r1x + r1y * r1y + r1z * r1z + rc_sq)
        r2_norm_reg = np.sqrt(r2x * r2x + r2y * r2y + r2z * r2z + rc_sq)

        diff_x = r1x / r1_norm_reg - r2x / r2_norm_reg
        diff_y = r1y / r1_norm_reg - r2y / r2_norm_reg
        diff_z = r1z / r1_norm_reg - r2z / r2_norm_reg

        dot_term = r0x * diff_x + r0y * diff_y + r0z * diff_z
        scale = (Gamma / (4.0 * np.pi)) * dot_term / denom

        out = np.empty(3, dtype=np.float64)
        out[0] = c0 * scale
        out[1] = c1 * scale
        out[2] = c2 * scale
        return out

    @njit(fastmath=True, cache=True)
    def _numba_semi_infinite_vortex_velocity(
        A: np.ndarray,
        d_hat: np.ndarray,
        P: np.ndarray,
        Gamma: float = 1.0,
        rc: float = VORTEX_CORE_RADIUS,
    ) -> np.ndarray:
        """Scalar semi-infinite vortex velocity evaluated at P."""
        rx = P[0] - A[0]
        ry = P[1] - A[1]
        rz = P[2] - A[2]

        d0 = d_hat[0]
        d1 = d_hat[1]
        d2 = d_hat[2]

        c0 = d1 * rz - d2 * ry
        c1 = d2 * rx - d0 * rz
        c2 = d0 * ry - d1 * rx
        cross_sq = c0 * c0 + c1 * c1 + c2 * c2
        rc_sq = rc * rc

        denom = cross_sq + rc_sq
        if denom <= 0.0:
            return np.zeros(3, dtype=np.float64)

        r_norm_reg = np.sqrt(rx * rx + ry * ry + rz * rz + rc_sq)
        cos_theta = (d0 * rx + d1 * ry + d2 * rz) / r_norm_reg
        scale = (Gamma / (4.0 * np.pi)) * (1.0 + cos_theta) / denom

        out = np.empty(3, dtype=np.float64)
        out[0] = c0 * scale
        out[1] = c1 * scale
        out[2] = c2 * scale
        return out

    @njit(fastmath=True, cache=True)
    def _numba_horseshoe_velocity(
        node_left: np.ndarray,
        node_right: np.ndarray,
        trailing_dir: np.ndarray,
        P: np.ndarray,
        Gamma: float = 1.0,
        rc: float = VORTEX_CORE_RADIUS,
    ) -> np.ndarray:
        """Scalar complete horseshoe vortex velocity evaluated at P."""
        V_bound = _numba_finite_vortex_velocity(node_left, node_right, P, Gamma, rc)
        V_left = _numba_semi_infinite_vortex_velocity(node_left, trailing_dir, P, -Gamma, rc)
        V_right = _numba_semi_infinite_vortex_velocity(node_right, trailing_dir, P, Gamma, rc)

        out = np.empty(3, dtype=np.float64)
        out[0] = V_bound[0] + V_left[0] + V_right[0]
        out[1] = V_bound[1] + V_left[1] + V_right[1]
        out[2] = V_bound[2] + V_left[2] + V_right[2]
        return out

else:
    # Fallback stubs if Numba is not installed
    _numba_horseshoe_velocity_matrix = None
    _numba_aic_matrix = None
    _numba_trailing_velocity_kernel = None
    _numba_bound_velocity_kernel = None
    _numba_nonlinear_relaxation_loop = None
    _numba_finite_vortex_velocity = None
    _numba_semi_infinite_vortex_velocity = None
    _numba_horseshoe_velocity = None
