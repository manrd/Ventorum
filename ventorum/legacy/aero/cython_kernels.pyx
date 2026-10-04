# Author: Manuel Alejandro Rodriguez Diaz, PhD
#cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""
Cython-compiled high-performance C kernels for Ventorum.

Provides C-level compiled routines for:
1. `cy_horseshoe_velocity_matrix`: Biot-Savart horseshoe vortex velocity tensor assembly.
2. `cy_aic_matrix`: Direct AIC normalwash projection without 3D tensor allocation.
3. `cy_trailing_velocity_kernel`: Precomputed trailing filaments velocity evaluation.
4. `cy_bound_velocity_kernel`: Precomputed bound vortex kernel evaluation.
5. `cy_nonlinear_relaxation_loop`: Fast C loop for nonlinear polar relaxation.
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt, pi, fabs
cimport cython

cnp.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def cy_horseshoe_velocity_matrix(
    const double[:, :] cp,
    const double[:, :] nl,
    const double[:, :] nr,
    const double[:] td,
    double gamma=1.0,
    double rc=1e-6
):
    """Compute 3D velocity induced at evaluation points `cp` by horseshoe vortices `nl`->`nr`."""
    cdef Py_ssize_t N = cp.shape[0]
    cdef Py_ssize_t M = nl.shape[0]
    cdef cnp.ndarray[double, ndim=3] V_tot = np.empty((N, M, 3), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=3] V_trail = np.empty((N, M, 3), dtype=np.float64)

    cdef double rc_sq = rc * rc
    cdef double inv_4pi = 1.0 / (4.0 * pi)
    cdef double g_inv_4pi = gamma * inv_4pi

    cdef double td0 = td[0], td1 = td[1], td2 = td[2]
    cdef Py_ssize_t i, j
    cdef double px, py, pz, ax, ay, az, bx, by, bz
    cdef double r1x, r1y, r1z, r2x, r2y, r2z, r0x, r0y, r0z
    cdef double c0, c1, c2, cross_sq, r0_sq, denom_bound
    cdef double r1_norm_sq, r2_norm_sq, r1_norm_reg, r2_norm_reg
    cdef double diff_x, diff_y, diff_z, dot_term, scale_bound
    cdef double vb_x, vb_y, vb_z, ca0, ca1, ca2, cb0, cb1, cb2
    cdef double cos_theta_A, cos_theta_B, denom_A, denom_B, scale_A, scale_B
    cdef double vl_x, vl_y, vl_z, vr_x, vr_y, vr_z, vt_x, vt_y, vt_z

    for i in range(N):
        px = cp[i, 0]
        py = cp[i, 1]
        pz = cp[i, 2]

        for j in range(M):
            ax = nl[j, 0]; ay = nl[j, 1]; az = nl[j, 2]
            bx = nr[j, 0]; by = nr[j, 1]; bz = nr[j, 2]

            r1x = px - ax; r1y = py - ay; r1z = pz - az
            r2x = px - bx; r2y = py - by; r2z = pz - bz
            r0x = bx - ax; r0y = by - ay; r0z = bz - az

            c0 = r1y * r2z - r1z * r2y
            c1 = r1z * r2x - r1x * r2z
            c2 = r1x * r2y - r1y * r2x
            cross_sq = c0 * c0 + c1 * c1 + c2 * c2

            r0_sq = r0x * r0x + r0y * r0y + r0z * r0z
            denom_bound = cross_sq + r0_sq * rc_sq

            r1_norm_sq = r1x * r1x + r1y * r1y + r1z * r1z
            r2_norm_sq = r2x * r2x + r2y * r2y + r2z * r2z
            r1_norm_reg = sqrt(r1_norm_sq + rc_sq)
            r2_norm_reg = sqrt(r2_norm_sq + rc_sq)

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
            denom_B = cb0 * cb0 + cb1 * cb1 + cb2 * cb2 + rc_sq

            if denom_A > 0.0:
                scale_A = -g_inv_4pi * (1.0 + cos_theta_A) / denom_A
            else:
                scale_A = 0.0

            if denom_B > 0.0:
                scale_B = g_inv_4pi * (1.0 + cos_theta_B) / denom_B
            else:
                scale_B = 0.0

            vl_x = ca0 * scale_A; vl_y = ca1 * scale_A; vl_z = ca2 * scale_A
            vr_x = cb0 * scale_B; vr_y = cb1 * scale_B; vr_z = cb2 * scale_B

            vt_x = vl_x + vr_x; vt_y = vl_y + vr_y; vt_z = vl_z + vr_z

            V_trail[i, j, 0] = vt_x
            V_trail[i, j, 1] = vt_y
            V_trail[i, j, 2] = vt_z

            V_tot[i, j, 0] = vb_x + vt_x
            V_tot[i, j, 1] = vb_y + vt_y
            V_tot[i, j, 2] = vb_z + vt_z

    return V_tot, V_trail


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def cy_aic_matrix(
    const double[:, :] cp,
    const double[:, :] nl,
    const double[:, :] nr,
    const double[:, :] normals,
    const double[:] td,
    double gamma=1.0,
    double rc=1e-6
):
    """Compute normalwash AIC matrix directly in C without 3D velocity tensor allocation."""
    cdef Py_ssize_t N = cp.shape[0]
    cdef Py_ssize_t M = nl.shape[0]
    cdef cnp.ndarray[double, ndim=2] AIC = np.empty((N, M), dtype=np.float64)

    cdef double rc_sq = rc * rc
    cdef double inv_4pi = 1.0 / (4.0 * pi)
    cdef double g_inv_4pi = gamma * inv_4pi

    cdef double td0 = td[0], td1 = td[1], td2 = td[2]
    cdef Py_ssize_t i, j
    cdef double px, py, pz, nx, ny, nz, ax, ay, az, bx, by, bz
    cdef double r1x, r1y, r1z, r2x, r2y, r2z, r0x, r0y, r0z
    cdef double c0, c1, c2, cross_sq, r0_sq, denom_bound
    cdef double r1_norm_sq, r2_norm_sq, r1_norm_reg, r2_norm_reg
    cdef double diff_x, diff_y, diff_z, dot_term, scale_bound
    cdef double vb_x, vb_y, vb_z, ca0, ca1, ca2, cb0, cb1, cb2
    cdef double cos_theta_A, cos_theta_B, denom_A, denom_B, scale_A, scale_B
    cdef double v_tot_x, v_tot_y, v_tot_z

    for i in range(N):
        px = cp[i, 0]; py = cp[i, 1]; pz = cp[i, 2]
        nx = normals[i, 0]; ny = normals[i, 1]; nz = normals[i, 2]

        for j in range(M):
            ax = nl[j, 0]; ay = nl[j, 1]; az = nl[j, 2]
            bx = nr[j, 0]; by = nr[j, 1]; bz = nr[j, 2]

            r1x = px - ax; r1y = py - ay; r1z = pz - az
            r2x = px - bx; r2y = py - by; r2z = pz - bz
            r0x = bx - ax; r0y = by - ay; r0z = bz - az

            c0 = r1y * r2z - r1z * r2y
            c1 = r1z * r2x - r1x * r2z
            c2 = r1x * r2y - r1y * r2x
            cross_sq = c0 * c0 + c1 * c1 + c2 * c2

            r0_sq = r0x * r0x + r0y * r0y + r0z * r0z
            denom_bound = cross_sq + r0_sq * rc_sq

            r1_norm_sq = r1x * r1x + r1y * r1y + r1z * r1z
            r2_norm_sq = r2x * r2x + r2y * r2y + r2z * r2z
            r1_norm_reg = sqrt(r1_norm_sq + rc_sq)
            r2_norm_reg = sqrt(r2_norm_sq + rc_sq)

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
            denom_B = cb0 * cb0 + cb1 * cb1 + cb2 * cb2 + rc_sq

            if denom_A > 0.0:
                scale_A = -g_inv_4pi * (1.0 + cos_theta_A) / denom_A
            else:
                scale_A = 0.0

            if denom_B > 0.0:
                scale_B = g_inv_4pi * (1.0 + cos_theta_B) / denom_B
            else:
                scale_B = 0.0

            v_tot_x = vb_x + ca0 * scale_A + cb0 * scale_B
            v_tot_y = vb_y + ca1 * scale_A + cb1 * scale_B
            v_tot_z = vb_z + ca2 * scale_A + cb2 * scale_B

            AIC[i, j] = v_tot_x * nx + v_tot_y * ny + v_tot_z * nz

    return AIC


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def cy_trailing_velocity_kernel(
    const double[:, :, :] r1,
    const double[:, :, :] r2,
    const double[:, :, :] r1_norm_reg,
    const double[:, :, :] r2_norm_reg,
    const double[:] td,
    double gamma=1.0,
    double rc=1e-6
):
    """Evaluate trailing-vortex downwash tensor from precomputed relative distances."""
    cdef Py_ssize_t N = r1.shape[0]
    cdef Py_ssize_t M = r1.shape[1]
    cdef cnp.ndarray[double, ndim=3] V_trail = np.empty((N, M, 3), dtype=np.float64)

    cdef double rc_sq = rc * rc
    cdef double inv_4pi = 1.0 / (4.0 * pi)
    cdef double g_inv_4pi = gamma * inv_4pi
    cdef double td0 = td[0], td1 = td[1], td2 = td[2]

    cdef Py_ssize_t i, j
    cdef double r1x, r1y, r1z, r2x, r2y, r2z, r1_n, r2_n
    cdef double ca0, ca1, ca2, cb0, cb1, cb2
    cdef double cos_theta_A, cos_theta_B, denom_A, denom_B, scale_A, scale_B

    for i in range(N):
        for j in range(M):
            r1x = r1[i, j, 0]; r1y = r1[i, j, 1]; r1z = r1[i, j, 2]
            r2x = r2[i, j, 0]; r2y = r2[i, j, 1]; r2z = r2[i, j, 2]
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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def cy_bound_velocity_kernel(
    const double[:, :, :] r1,
    const double[:, :, :] r2,
    const double[:, :, :] r0,
    double rc=1e-6,
    double gamma=1.0
):
    """Evaluate finite bound-vortex kernel between nodes A and B."""
    cdef Py_ssize_t N = r1.shape[0]
    cdef Py_ssize_t M = r1.shape[1]
    cdef cnp.ndarray[double, ndim=3] V_bound = np.empty((N, M, 3), dtype=np.float64)

    cdef double rc_sq = rc * rc
    cdef double inv_4pi = 1.0 / (4.0 * pi)
    cdef double g_inv_4pi = gamma * inv_4pi
    cdef bint is_r0_single_row = (r0.shape[0] == 1)

    cdef Py_ssize_t i, j
    cdef double r1x, r1y, r1z, r2x, r2y, r2z, r0x, r0y, r0z
    cdef double c0, c1, c2, cross_sq, r0_sq, denom_bound
    cdef double r1_norm_sq, r2_norm_sq, r1_norm_reg, r2_norm_reg
    cdef double diff_x, diff_y, diff_z, dot_term, scale_bound

    for i in range(N):
        for j in range(M):
            r1x = r1[i, j, 0]; r1y = r1[i, j, 1]; r1z = r1[i, j, 2]
            r2x = r2[i, j, 0]; r2y = r2[i, j, 1]; r2z = r2[i, j, 2]

            if is_r0_single_row:
                r0x = r0[0, j, 0]; r0y = r0[0, j, 1]; r0z = r0[0, j, 2]
            else:
                r0x = r0[i, j, 0]; r0y = r0[i, j, 1]; r0z = r0[i, j, 2]

            c0 = r1y * r2z - r1z * r2y
            c1 = r1z * r2x - r1x * r2z
            c2 = r1x * r2y - r1y * r2x
            cross_sq = c0 * c0 + c1 * c1 + c2 * c2

            r0_sq = r0x * r0x + r0y * r0y + r0z * r0z
            denom_bound = cross_sq + r0_sq * rc_sq

            r1_norm_sq = r1x * r1x + r1y * r1y + r1z * r1z
            r2_norm_sq = r2x * r2x + r2y * r2y + r2z * r2z
            r1_norm_reg = sqrt(r1_norm_sq + rc_sq)
            r2_norm_reg = sqrt(r2_norm_sq + rc_sq)

            diff_x = (r1x / r1_norm_reg) - (r2x / r2_norm_reg)
            diff_y = (r1y / r1_norm_reg) - (r2y / r2_norm_reg)
            diff_z = (r1z / r1_norm_reg) - (r2z / r2_norm_reg)
            dot_term = r0x * diff_x + r0y * diff_y + r0z * diff_z

            scale_bound = g_inv_4pi * (dot_term / denom_bound) if denom_bound > 0.0 else 0.0

            V_bound[i, j, 0] = c0 * scale_bound
            V_bound[i, j, 1] = c1 * scale_bound
            V_bound[i, j, 2] = c2 * scale_bound

    return V_bound


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def cy_nonlinear_relaxation_loop(
    const double[:, :] V_z,
    const double[:] Gamma_init,
    double inv_vinf,
    const double[:] alpha_geom,
    const double[:] half_vinf_chords,
    const double[:] alpha_table,
    const double[:] cl_table,
    double omega,
    int max_iter,
    double tol
):
    """Cython-compiled iterative relaxation loop for nonlinear section polars."""
    cdef Py_ssize_t N = Gamma_init.shape[0]
    cdef Py_ssize_t n_polar = alpha_table.shape[0]
    cdef cnp.ndarray[double, ndim=1] gamma_arr = np.empty(N, dtype=np.float64)
    cdef double[:] Gamma = gamma_arr
    cdef cnp.ndarray[double, ndim=1] alpha_eff_arr = np.empty(N, dtype=np.float64)
    cdef double[:] alpha_eff = alpha_eff_arr
    cdef list residual_history = []
    cdef bint converged = False
    cdef int it, final_iter = max_iter
    cdef Py_ssize_t i, j, k, n_res
    cdef double vz_g, alpha_i, alpha_eff_i, cl_i, dg, abs_dg, max_dGamma, residual, d_alpha

    for i in range(N):
        Gamma[i] = Gamma_init[i]

    for it in range(1, max_iter + 1):
        final_iter = it
        max_dGamma = 0.0

        # Pass 1: Induced downwash evaluation
        for i in range(N):
            vz_g = 0.0
            for j in range(N):
                vz_g += V_z[i, j] * Gamma[j]
            alpha_i = -vz_g * inv_vinf
            alpha_eff[i] = alpha_geom[i] - alpha_i

        # Pass 2: Cl polar lookup & circulation relaxation
        for i in range(N):
            alpha_eff_i = alpha_eff[i]
            if alpha_eff_i <= alpha_table[0]:
                cl_i = cl_table[0]
            elif alpha_eff_i >= alpha_table[n_polar - 1]:
                cl_i = cl_table[n_polar - 1]
            else:
                k = 0
                while k < n_polar - 1 and alpha_table[k + 1] < alpha_eff_i:
                    k += 1
                d_alpha = alpha_table[k + 1] - alpha_table[k]
                if d_alpha != 0.0:
                    cl_i = cl_table[k] + (alpha_eff_i - alpha_table[k]) * (cl_table[k + 1] - cl_table[k]) / d_alpha
                else:
                    cl_i = cl_table[k]

            dg = half_vinf_chords[i] * cl_i - Gamma[i]
            abs_dg = fabs(dg)
            if abs_dg > max_dGamma:
                max_dGamma = abs_dg
            Gamma[i] += omega * dg

        residual = omega * max_dGamma
        residual_history.append(residual)

        if residual < tol:
            converged = True
            break

        n_res = len(residual_history)
        if n_res >= 3 and residual_history[n_res - 1] > residual_history[n_res - 2]:
            omega = max(0.02, omega * 0.75)

    return gamma_arr, final_iter, converged, residual_history


