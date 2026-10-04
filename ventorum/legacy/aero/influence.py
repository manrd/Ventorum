# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Aerodynamic Influence Coefficient (AIC) matrix assembly.

The AIC matrix relates unit-circulation horseshoe vortices to the normal-wash
at each control point.  For *N* total panels the system is:

    AIC · Γ = b

where  ``AIC[i, j]``  is the normal velocity induced at control point *i* by
horseshoe vortex *j* with Γ = 1, and  ``b[i] = −V∞ · n̂_i``.
"""

from __future__ import annotations

import numpy as np

from ventorum.legacy.aero.biot_savart import horseshoe_velocity
from ventorum.legacy.core.constants import VORTEX_CORE_RADIUS
from ventorum.legacy.core.datatypes import DiscretizedSurface, FlightCondition
from ventorum.legacy.aero.acceleration import (
    dispatch_horseshoe_velocity_matrix,
    dispatch_aic_matrix,
    dispatch_trailing_velocity_kernel,
    dispatch_bound_velocity_kernel,
)


def _trailing_direction(condition: FlightCondition) -> np.ndarray:
    """Unit vector of trailing vortex filaments in body-fixed coordinates.

    In standard lifting-line and vortex-lattice formulations (e.g. Drela's AVL,
    Katz & Plotkin), trailing vortex filaments shed in the longitudinal plane
    (alpha-aligned, beta=0) to prevent numerical wake-control-point collisions
    during sideslip.
    """
    ca, sa = np.cos(condition.alpha), np.sin(condition.alpha)
    return np.array([ca, 0.0, sa])


def _freestream_direction(condition: FlightCondition) -> np.ndarray:
    """Unit vector of the free-stream in body-fixed coordinates.

    Convention: x forward/aft, y starboard, z up.
    At angle of attack α and sideslip β, the free stream velocity direction
    in body axes is [cos α cos β, −sin β, sin α cos β].
    """
    ca, sa = np.cos(condition.alpha), np.sin(condition.alpha)
    cb, sb = np.cos(condition.beta), np.sin(condition.beta)
    return np.array([ca * cb, -sb, sa * cb])


def _freestream_vector(condition: FlightCondition) -> np.ndarray:
    """Full free-stream velocity vector [m/s]."""
    return condition.V_inf * _freestream_direction(condition)


def compute_horseshoe_velocity_matrix(
    cp: np.ndarray,
    nl: np.ndarray,
    nr: np.ndarray,
    trailing_dir: np.ndarray,
    gamma: float = 1.0,
    rc: float = VORTEX_CORE_RADIUS,
    return_trailing: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compute the 3D velocity induced at evaluation points `cp` by horseshoe vortices `nl`->`nr`.
    
    Parameters
    ----------
    cp : np.ndarray, shape (N, 3)
        Evaluation (control) points.
    nl, nr : np.ndarray, shape (M, 3)
        Left and right bound vortex nodes.
    trailing_dir : np.ndarray, shape (3,)
        Unit vector in the freestream trailing direction.
    gamma : float
        Horseshoe circulation multiplier.
    rc : float
        Regularization radius.
    return_trailing : bool
        If True, returns a tuple `(V_tot, V_trail)`.
        
    Returns
    -------
    V : np.ndarray, shape (N, M, 3) or tuple of np.ndarray
        Velocity induced at each of the N points by each of the M horseshoe vortices.
    """
    v_res = dispatch_horseshoe_velocity_matrix(
        cp, nl, nr, trailing_dir, gamma=gamma, rc=rc, return_trailing=return_trailing
    )
    if v_res is not None:
        return v_res

    P = cp[:, np.newaxis, :]  # (N, 1, 3)
    A = nl[np.newaxis, :, :]  # (1, M, 3)
    B = nr[np.newaxis, :, :]  # (1, M, 3)

    # Relative vectors
    r1 = P - A  # (N, M, 3)
    r2 = P - B  # (N, M, 3)
    r0 = B - A  # (1, M, 3)

    rc_sq = rc * rc
    inv_4pi = 1.0 / (4.0 * np.pi)

    # 1. Finite bound vortex segment: A -> B
    # Direct 3D cross product of r1 x r2 avoiding temporary stack allocations
    cross_bound = np.empty_like(r1)
    cross_bound[..., 0] = r1[..., 1] * r2[..., 2] - r1[..., 2] * r2[..., 1]
    cross_bound[..., 1] = r1[..., 2] * r2[..., 0] - r1[..., 0] * r2[..., 2]
    cross_bound[..., 2] = r1[..., 0] * r2[..., 1] - r1[..., 1] * r2[..., 0]

    cross_sq = (cross_bound[..., 0]**2 + cross_bound[..., 1]**2 + cross_bound[..., 2]**2)[..., None]
    r0_sq = (r0[..., 0]**2 + r0[..., 1]**2 + r0[..., 2]**2)[..., None]
    denom_bound = cross_sq + r0_sq * rc_sq

    r1_norm_sq = (r1[..., 0]**2 + r1[..., 1]**2 + r1[..., 2]**2)[..., None]
    r2_norm_sq = (r2[..., 0]**2 + r2[..., 1]**2 + r2[..., 2]**2)[..., None]
    r1_norm_reg = np.sqrt(r1_norm_sq + rc_sq)
    r2_norm_reg = np.sqrt(r2_norm_sq + rc_sq)

    diff = (r1 / r1_norm_reg) - (r2 / r2_norm_reg)
    dot_term = (r0[..., 0] * diff[..., 0] + r0[..., 1] * diff[..., 1] + r0[..., 2] * diff[..., 2])[..., None]

    safe_denom_bound = np.maximum(denom_bound, 1e-30)
    scale_bound = np.where(denom_bound > 0.0, (gamma * inv_4pi) * (dot_term / safe_denom_bound), 0.0)
    V_bound = cross_bound * scale_bound

    # 2. Left trailing leg: arrives from infinity to A along trailing_dir
    td0, td1, td2 = trailing_dir[0], trailing_dir[1], trailing_dir[2]
    cross_A = np.empty_like(r1)
    cross_B = np.empty_like(r2)

    if td1 == 0.0:
        cross_A[..., 0] = -td2 * r1[..., 1]
        cross_A[..., 1] = td2 * r1[..., 0] - td0 * r1[..., 2]
        cross_A[..., 2] = td0 * r1[..., 1]
        cos_theta_A = (td0 * r1[..., 0] + td2 * r1[..., 2])[..., None] / r1_norm_reg

        cross_B[..., 0] = -td2 * r2[..., 1]
        cross_B[..., 1] = td2 * r2[..., 0] - td0 * r2[..., 2]
        cross_B[..., 2] = td0 * r2[..., 1]
        cos_theta_B = (td0 * r2[..., 0] + td2 * r2[..., 2])[..., None] / r2_norm_reg
    else:
        cross_A[..., 0] = td1 * r1[..., 2] - td2 * r1[..., 1]
        cross_A[..., 1] = td2 * r1[..., 0] - td0 * r1[..., 2]
        cross_A[..., 2] = td0 * r1[..., 1] - td1 * r1[..., 0]
        cos_theta_A = (td0 * r1[..., 0] + td1 * r1[..., 1] + td2 * r1[..., 2])[..., None] / r1_norm_reg

        cross_B[..., 0] = td1 * r2[..., 2] - td2 * r2[..., 1]
        cross_B[..., 1] = td2 * r2[..., 0] - td0 * r2[..., 2]
        cross_B[..., 2] = td0 * r2[..., 1] - td1 * r2[..., 0]
        cos_theta_B = (td0 * r2[..., 0] + td1 * r2[..., 1] + td2 * r2[..., 2])[..., None] / r2_norm_reg

    denom_A = (cross_A[..., 0]**2 + cross_A[..., 1]**2 + cross_A[..., 2]**2)[..., None] + rc_sq
    safe_denom_A = np.maximum(denom_A, 1e-30)
    scale_A = np.where(denom_A > 0.0, -(gamma * inv_4pi) * (1.0 + cos_theta_A) / safe_denom_A, 0.0)
    V_left = cross_A * scale_A

    denom_B = (cross_B[..., 0]**2 + cross_B[..., 1]**2 + cross_B[..., 2]**2)[..., None] + rc_sq
    safe_denom_B = np.maximum(denom_B, 1e-30)
    scale_B = np.where(denom_B > 0.0, (gamma * inv_4pi) * (1.0 + cos_theta_B) / safe_denom_B, 0.0)
    V_right = cross_B * scale_B

    V_trail = V_left + V_right
    V_tot = V_bound + V_trail

    if return_trailing:
        return V_tot, V_trail
    return V_tot


def _eval_trailing_kernel(
    r1: np.ndarray,
    r2: np.ndarray,
    r1_norm_reg: np.ndarray,
    r2_norm_reg: np.ndarray,
    trailing_dir: np.ndarray,
    gamma: float = 1.0,
    rc: float = VORTEX_CORE_RADIUS,
) -> np.ndarray:
    """Vectorized Biot-Savart trailing filaments evaluation kernel."""
    v_trail_accel = dispatch_trailing_velocity_kernel(
        r1, r2, r1_norm_reg, r2_norm_reg, trailing_dir, gamma=gamma, rc=rc
    )
    if v_trail_accel is not None:
        return v_trail_accel

    rc_sq = rc * rc
    inv_4pi = 1.0 / (4.0 * np.pi)
    td0, td1, td2 = trailing_dir[0], trailing_dir[1], trailing_dir[2]

    cross_A = np.empty_like(r1)
    cross_B = np.empty_like(r2)

    if td1 == 0.0:
        cross_A[..., 0] = -td2 * r1[..., 1]
        cross_A[..., 1] = td2 * r1[..., 0] - td0 * r1[..., 2]
        cross_A[..., 2] = td0 * r1[..., 1]
        cos_theta_A = (td0 * r1[..., 0] + td2 * r1[..., 2])[..., None] / r1_norm_reg

        cross_B[..., 0] = -td2 * r2[..., 1]
        cross_B[..., 1] = td2 * r2[..., 0] - td0 * r2[..., 2]
        cross_B[..., 2] = td0 * r2[..., 1]
        cos_theta_B = (td0 * r2[..., 0] + td2 * r2[..., 2])[..., None] / r2_norm_reg
    else:
        cross_A[..., 0] = td1 * r1[..., 2] - td2 * r1[..., 1]
        cross_A[..., 1] = td2 * r1[..., 0] - td0 * r1[..., 2]
        cross_A[..., 2] = td0 * r1[..., 1] - td1 * r1[..., 0]
        cos_theta_A = (td0 * r1[..., 0] + td1 * r1[..., 1] + td2 * r1[..., 2])[..., None] / r1_norm_reg

        cross_B[..., 0] = td1 * r2[..., 2] - td2 * r2[..., 1]
        cross_B[..., 1] = td2 * r2[..., 0] - td0 * r2[..., 2]
        cross_B[..., 2] = td0 * r2[..., 1] - td1 * r2[..., 0]
        cos_theta_B = (td0 * r2[..., 0] + td1 * r2[..., 1] + td2 * r2[..., 2])[..., None] / r2_norm_reg

    denom_A = (cross_A[..., 0]**2 + cross_A[..., 1]**2 + cross_A[..., 2]**2)[..., None] + rc_sq
    safe_denom_A = np.maximum(denom_A, 1e-30)
    scale_A = np.where(denom_A > 0.0, -(gamma * inv_4pi) * (1.0 + cos_theta_A) / safe_denom_A, 0.0)

    denom_B = (cross_B[..., 0]**2 + cross_B[..., 1]**2 + cross_B[..., 2]**2)[..., None] + rc_sq
    safe_denom_B = np.maximum(denom_B, 1e-30)
    scale_B = np.where(denom_B > 0.0, (gamma * inv_4pi) * (1.0 + cos_theta_B) / safe_denom_B, 0.0)

    return cross_A * scale_A + cross_B * scale_B


def compute_trailing_velocity_from_cache(
    geo_cache: HorseshoeGeometryCache,
    trailing_dir: np.ndarray,
    gamma: float = 1.0,
    rc: float = VORTEX_CORE_RADIUS,
) -> np.ndarray:
    """Evaluate trailing-vortex downwash tensor from precomputed geometry cache."""
    # Primary trailing legs
    V_trail = _eval_trailing_kernel(
        geo_cache.r1,
        geo_cache.r2,
        geo_cache.r1_norm_reg,
        geo_cache.r2_norm_reg,
        trailing_dir,
        gamma=gamma,
        rc=rc,
    )

    # Symmetric mirrored trailing legs (if symmetry plane is enabled)
    if geo_cache.use_symmetry and geo_cache.r1_sym is not None and geo_cache.r2_sym is not None:
        V_trail_sym = _eval_trailing_kernel(
            geo_cache.r1_sym,
            geo_cache.r2_sym,
            geo_cache.r1_sym_norm_reg,
            geo_cache.r2_sym_norm_reg,
            trailing_dir,
            gamma=gamma,
            rc=rc,
        )
        V_trail = V_trail + V_trail_sym

    # Ground Effect (Method of Images)
    if geo_cache.h is not None and geo_cache.r1_img is not None:
        td_img = np.array([trailing_dir[0], trailing_dir[1], -trailing_dir[2]])
        # Primary image legs
        V_trail_img = _eval_trailing_kernel(
            geo_cache.r1_img,
            geo_cache.r2_img,
            geo_cache.r1_img_reg,
            geo_cache.r2_img_reg,
            td_img,
            gamma=-gamma,
            rc=rc,
        )
        V_trail = V_trail + V_trail_img

        # Mirrored image legs
        if geo_cache.use_symmetry and geo_cache.r1_img_sym is not None and geo_cache.r2_img_sym is not None:
            V_trail_img_sym = _eval_trailing_kernel(
                geo_cache.r1_img_sym,
                geo_cache.r2_img_sym,
                geo_cache.r1_img_sym_reg,
                geo_cache.r2_img_sym_reg,
                td_img,
                gamma=-gamma,
                rc=rc,
            )
            V_trail = V_trail + V_trail_img_sym

    return V_trail


def build_aic_and_rhs(
    disc_surfaces: list[DiscretizedSurface],
    condition: FlightCondition,
    rc: float = VORTEX_CORE_RADIUS,
    return_details: bool = False,
    use_symmetry: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assemble the global AIC matrix and RHS vector.

    Parameters
    ----------
    disc_surfaces : list[DiscretizedSurface]
        One entry per discretised surface (may include mirrored halves, or half-mesh if use_symmetry).
    condition : FlightCondition
        Free-stream conditions.
    rc : float
        Vortex-core regularisation radius.
    return_details : bool
        If True, also returns (V_tot, V_trail).
    use_symmetry : bool
        If True, solves the problem on the half-mesh (y >= 0) with mathematical reflection across Y=0.

    Returns
    -------
    AIC : np.ndarray, shape (N, N)
    rhs : np.ndarray, shape (N,)
    (optional) V_tot : np.ndarray, shape (N, N, 3)
    (optional) V_trail : np.ndarray, shape (N, N, 3)
    """
    # Auto-detect if half-mesh surfaces were passed
    if any(getattr(ds, "is_half_mesh", False) for ds in disc_surfaces):
        use_symmetry = True

    # --- flatten all panels into global arrays --------------------------------
    if len(disc_surfaces) == 1:
        ds0 = disc_surfaces[0]
        cp = ds0.control_points
        normals = ds0.normals
        nl = ds0.nodes_qc[:-1]
        nr = ds0.nodes_qc[1:]
    else:
        all_cp: list[np.ndarray] = []
        all_normals: list[np.ndarray] = []
        all_nodes_left: list[np.ndarray] = []
        all_nodes_right: list[np.ndarray] = []

        for ds in disc_surfaces:
            all_cp.append(ds.control_points)          # [n, 3]
            all_normals.append(ds.normals)            # [n, 3]
            all_nodes_left.append(ds.nodes_qc[:-1])    # [n, 3]
            all_nodes_right.append(ds.nodes_qc[1:])    # [n, 3]

        cp = np.vstack(all_cp)             # [N, 3]
        normals = np.vstack(all_normals)   # [N, 3]
        nl = np.vstack(all_nodes_left)     # [N, 3]
        nr = np.vstack(all_nodes_right)    # [N, 3]

    trailing_dir = _trailing_direction(condition)
    V_inf_vec = _freestream_vector(condition)

    if not return_details:
        AIC = dispatch_aic_matrix(cp, nl, nr, normals, trailing_dir, gamma=1.0, rc=rc)
        if AIC is not None:
            if use_symmetry:
                nl_sym = nr.copy(); nl_sym[:, 1] *= -1.0
                nr_sym = nl.copy(); nr_sym[:, 1] *= -1.0
                AIC_sym = dispatch_aic_matrix(cp, nl_sym, nr_sym, normals, trailing_dir, gamma=1.0, rc=rc)
                if AIC_sym is not None:
                    AIC += AIC_sym
                h = condition.h
                if h is not None:
                    td_img = np.array([trailing_dir[0], trailing_dir[1], -trailing_dir[2]])
                    nl_img = nl.copy(); nl_img[:, 2] = -2.0 * h - nl_img[:, 2]
                    nr_img = nr.copy(); nr_img[:, 2] = -2.0 * h - nr_img[:, 2]
                    nl_img_sym = nl_sym.copy(); nl_img_sym[:, 2] = -2.0 * h - nl_img_sym[:, 2]
                    nr_img_sym = nr_sym.copy(); nr_img_sym[:, 2] = -2.0 * h - nr_img_sym[:, 2]

                    AIC_img = dispatch_aic_matrix(cp, nl_img, nr_img, normals, td_img, gamma=-1.0, rc=rc)
                    AIC_img_sym = dispatch_aic_matrix(cp, nl_img_sym, nr_img_sym, normals, td_img, gamma=-1.0, rc=rc)
                    if AIC_img is not None and AIC_img_sym is not None:
                        AIC += (AIC_img + AIC_img_sym)
            else:
                h = condition.h
                if h is not None:
                    td_img = np.array([trailing_dir[0], trailing_dir[1], -trailing_dir[2]])
                    nl_img = nl.copy(); nl_img[:, 2] = -2.0 * h - nl_img[:, 2]
                    nr_img = nr.copy(); nr_img[:, 2] = -2.0 * h - nr_img[:, 2]
                    AIC_img = dispatch_aic_matrix(cp, nl_img, nr_img, normals, td_img, gamma=-1.0, rc=rc)
                    if AIC_img is not None:
                        AIC += AIC_img

            rhs = -np.dot(normals, V_inf_vec)
            return AIC, rhs

    if use_symmetry:
        # Evaluate primary right-hand semi-span
        if return_details:
            V_R, V_tr_R = compute_horseshoe_velocity_matrix(
                cp, nl, nr, trailing_dir, gamma=1.0, rc=rc, return_trailing=True
            )
        else:
            V_R = compute_horseshoe_velocity_matrix(
                cp, nl, nr, trailing_dir, gamma=1.0, rc=rc, return_trailing=False
            )
            V_tr_R = None

        # Mirrored left-hand semi-span: reflection across xz-plane (y -> -y)
        nl_sym = nr.copy()
        nl_sym[:, 1] *= -1.0
        nr_sym = nl.copy()
        nr_sym[:, 1] *= -1.0

        if return_details:
            V_L, V_tr_L = compute_horseshoe_velocity_matrix(
                cp, nl_sym, nr_sym, trailing_dir, gamma=1.0, rc=rc, return_trailing=True
            )
            V_tot = V_R + V_L
            V_trail = V_tr_R + V_tr_L
        else:
            V_L = compute_horseshoe_velocity_matrix(
                cp, nl_sym, nr_sym, trailing_dir, gamma=1.0, rc=rc, return_trailing=False
            )
            V_tot = V_R + V_L
            V_trail = None

        # Ground Effect with symmetry (Method of Images across ground plane z = -h)
        h = condition.h
        if h is not None:
            td_img = np.array([trailing_dir[0], trailing_dir[1], -trailing_dir[2]])

            nl_img = nl.copy()
            nl_img[:, 2] = -2.0 * h - nl_img[:, 2]
            nr_img = nr.copy()
            nr_img[:, 2] = -2.0 * h - nr_img[:, 2]

            nl_img_sym = nl_sym.copy()
            nl_img_sym[:, 2] = -2.0 * h - nl_img_sym[:, 2]
            nr_img_sym = nr_sym.copy()
            nr_img_sym[:, 2] = -2.0 * h - nr_img_sym[:, 2]

            if return_details:
                V_img_R, V_tr_img_R = compute_horseshoe_velocity_matrix(
                    cp, nl_img, nr_img, td_img, gamma=-1.0, rc=rc, return_trailing=True
                )
                V_img_L, V_tr_img_L = compute_horseshoe_velocity_matrix(
                    cp, nl_img_sym, nr_img_sym, td_img, gamma=-1.0, rc=rc, return_trailing=True
                )
                V_tot += (V_img_R + V_img_L)
                V_trail += (V_tr_img_R + V_tr_img_L)
            else:
                V_img_R = compute_horseshoe_velocity_matrix(
                    cp, nl_img, nr_img, td_img, gamma=-1.0, rc=rc, return_trailing=False
                )
                V_img_L = compute_horseshoe_velocity_matrix(
                    cp, nl_img_sym, nr_img_sym, td_img, gamma=-1.0, rc=rc, return_trailing=False
                )
                V_tot += (V_img_R + V_img_L)

    else:
        # Standard full solve without symmetry plane
        if return_details:
            V_tot, V_trail = compute_horseshoe_velocity_matrix(
                cp, nl, nr, trailing_dir, gamma=1.0, rc=rc, return_trailing=True
            )
        else:
            V_tot = compute_horseshoe_velocity_matrix(
                cp, nl, nr, trailing_dir, gamma=1.0, rc=rc, return_trailing=False
            )
            V_trail = None

        # Ground Effect (Method of Images)
        h = condition.h
        if h is not None:
            nl_img = nl.copy()
            nl_img[:, 2] = -2.0 * h - nl_img[:, 2]
            nr_img = nr.copy()
            nr_img[:, 2] = -2.0 * h - nr_img[:, 2]
            td_img = np.array([trailing_dir[0], trailing_dir[1], -trailing_dir[2]])
            if return_details:
                V_img, V_img_trail = compute_horseshoe_velocity_matrix(
                    cp, nl_img, nr_img, td_img, gamma=-1.0, rc=rc, return_trailing=True
                )
                V_tot += V_img
                V_trail += V_img_trail
            else:
                V_img = compute_horseshoe_velocity_matrix(
                    cp, nl_img, nr_img, td_img, gamma=-1.0, rc=rc, return_trailing=False
                )
                V_tot += V_img

    # Normalwash: AIC[i, j] = V_tot[i, j] · normals[i]
    AIC = np.einsum('ijk,ik->ij', V_tot, normals)

    # RHS: flow-tangency condition
    rhs = -np.dot(normals, V_inf_vec)

    if return_details:
        return AIC, rhs, V_tot, V_trail
    return AIC, rhs


class HorseshoeGeometryCache:
    """Precomputed bound-vortex geometry and relative distance vectors.
    
    Accelerates multi-point flight condition solves and angle-of-attack sweeps
    by caching geometry-dependent quantities that are invariant to free-stream direction.
    Supports Y=0 symmetry plane acceleration.
    """
    def __init__(
        self,
        cp: np.ndarray,
        normals: np.ndarray,
        nl: np.ndarray,
        nr: np.ndarray,
        r1: np.ndarray,
        r2: np.ndarray,
        r1_norm_reg: np.ndarray,
        r2_norm_reg: np.ndarray,
        V_bound: np.ndarray,
        disc_surfaces: list[DiscretizedSurface],
        h: float | None = None,
        r1_img: np.ndarray | None = None,
        r2_img: np.ndarray | None = None,
        r1_img_reg: np.ndarray | None = None,
        r2_img_reg: np.ndarray | None = None,
        V_bound_img: np.ndarray | None = None,
        AIC_bound: np.ndarray | None = None,
        use_symmetry: bool = False,
        nl_sym: np.ndarray | None = None,
        nr_sym: np.ndarray | None = None,
        r1_sym: np.ndarray | None = None,
        r2_sym: np.ndarray | None = None,
        r1_sym_norm_reg: np.ndarray | None = None,
        r2_sym_norm_reg: np.ndarray | None = None,
        r1_img_sym: np.ndarray | None = None,
        r2_img_sym: np.ndarray | None = None,
        r1_img_sym_reg: np.ndarray | None = None,
        r2_img_sym_reg: np.ndarray | None = None,
    ):
        self.cp = cp
        self.normals = normals
        self.nl = nl
        self.nr = nr
        self.r1 = r1
        self.r2 = r2
        self.r1_norm_reg = r1_norm_reg
        self.r2_norm_reg = r2_norm_reg
        self.V_bound = V_bound
        self.disc_surfaces = disc_surfaces
        self.h = h
        self.r1_img = r1_img
        self.r2_img = r2_img
        self.r1_img_reg = r1_img_reg
        self.r2_img_reg = r2_img_reg
        self.V_bound_img = V_bound_img
        self.AIC_bound = AIC_bound
        self.use_symmetry = use_symmetry
        self.nl_sym = nl_sym
        self.nr_sym = nr_sym
        self.r1_sym = r1_sym
        self.r2_sym = r2_sym
        self.r1_sym_norm_reg = r1_sym_norm_reg
        self.r2_sym_norm_reg = r2_sym_norm_reg
        self.r1_img_sym = r1_img_sym
        self.r2_img_sym = r2_img_sym
        self.r1_img_sym_reg = r1_img_sym_reg
        self.r2_img_sym_reg = r2_img_sym_reg


def _eval_bound_kernel(
    r1: np.ndarray,
    r2: np.ndarray,
    r0: np.ndarray,
    rc: float = VORTEX_CORE_RADIUS,
    gamma: float = 1.0,
) -> np.ndarray:
    """Helper to evaluate finite bound-vortex kernel between nodes A and B."""
    v_bound_accel = dispatch_bound_velocity_kernel(r1, r2, r0, rc=rc, gamma=gamma)
    if v_bound_accel is not None:
        return v_bound_accel

    rc_sq = rc * rc
    inv_4pi = 1.0 / (4.0 * np.pi)

    cross_bound = np.empty_like(r1)
    cross_bound[..., 0] = r1[..., 1] * r2[..., 2] - r1[..., 2] * r2[..., 1]
    cross_bound[..., 1] = r1[..., 2] * r2[..., 0] - r1[..., 0] * r2[..., 2]
    cross_bound[..., 2] = r1[..., 0] * r2[..., 1] - r1[..., 1] * r2[..., 0]

    cross_sq = (cross_bound[..., 0]**2 + cross_bound[..., 1]**2 + cross_bound[..., 2]**2)[..., None]
    r0_sq = (r0[..., 0]**2 + r0[..., 1]**2 + r0[..., 2]**2)[..., None]
    denom_bound = cross_sq + r0_sq * rc_sq

    r1_norm_sq = (r1[..., 0]**2 + r1[..., 1]**2 + r1[..., 2]**2)[..., None]
    r2_norm_sq = (r2[..., 0]**2 + r2[..., 1]**2 + r2[..., 2]**2)[..., None]
    r1_norm_reg = np.sqrt(r1_norm_sq + rc_sq)
    r2_norm_reg = np.sqrt(r2_norm_sq + rc_sq)

    diff = (r1 / r1_norm_reg) - (r2 / r2_norm_reg)
    dot_term = (r0[..., 0] * diff[..., 0] + r0[..., 1] * diff[..., 1] + r0[..., 2] * diff[..., 2])[..., None]
    safe_denom_bound = np.maximum(denom_bound, 1e-30)
    scale_bound = np.where(denom_bound > 0.0, (gamma * inv_4pi) * (dot_term / safe_denom_bound), 0.0)
    return cross_bound * scale_bound


def precompute_horseshoe_geometry(
    disc_surfaces: list[DiscretizedSurface],
    h: float | None = None,
    rc: float = VORTEX_CORE_RADIUS,
    use_symmetry: bool = False,
) -> HorseshoeGeometryCache:
    """Precompute bound-vortex kernel and relative position vectors for horseshoe solver."""
    if any(getattr(ds, "is_half_mesh", False) for ds in disc_surfaces):
        use_symmetry = True

    if len(disc_surfaces) == 1:
        ds0 = disc_surfaces[0]
        cp = ds0.control_points
        normals = ds0.normals
        nl = ds0.nodes_qc[:-1]
        nr = ds0.nodes_qc[1:]
    else:
        all_cp: list[np.ndarray] = []
        all_normals: list[np.ndarray] = []
        all_nodes_left: list[np.ndarray] = []
        all_nodes_right: list[np.ndarray] = []

        for ds in disc_surfaces:
            all_cp.append(ds.control_points)
            all_normals.append(ds.normals)
            all_nodes_left.append(ds.nodes_qc[:-1])
            all_nodes_right.append(ds.nodes_qc[1:])

        cp = np.vstack(all_cp)
        normals = np.vstack(all_normals)
        nl = np.vstack(all_nodes_left)
        nr = np.vstack(all_nodes_right)

    P = cp[:, np.newaxis, :]
    A = nl[np.newaxis, :, :]
    B = nr[np.newaxis, :, :]
    r1 = P - A
    r2 = P - B
    r0 = B - A
    rc_sq = rc * rc

    r1_norm_sq = (r1[..., 0]**2 + r1[..., 1]**2 + r1[..., 2]**2)[..., None]
    r2_norm_sq = (r2[..., 0]**2 + r2[..., 1]**2 + r2[..., 2]**2)[..., None]
    r1_norm_reg = np.sqrt(r1_norm_sq + rc_sq)
    r2_norm_reg = np.sqrt(r2_norm_sq + rc_sq)

    V_bound = _eval_bound_kernel(r1, r2, r0, rc=rc, gamma=1.0)

    # Symmetry components
    nl_sym = nr_sym = r1_sym = r2_sym = r1_sym_norm_reg = r2_sym_norm_reg = None
    if use_symmetry:
        nl_sym = nr.copy(); nl_sym[:, 1] *= -1.0
        nr_sym = nl.copy(); nr_sym[:, 1] *= -1.0
        A_sym = nl_sym[np.newaxis, :, :]
        B_sym = nr_sym[np.newaxis, :, :]
        r1_sym = P - A_sym
        r2_sym = P - B_sym
        r0_sym = B_sym - A_sym
        r1_s_sq = (r1_sym[..., 0]**2 + r1_sym[..., 1]**2 + r1_sym[..., 2]**2)[..., None]
        r2_s_sq = (r2_sym[..., 0]**2 + r2_sym[..., 1]**2 + r2_sym[..., 2]**2)[..., None]
        r1_sym_norm_reg = np.sqrt(r1_s_sq + rc_sq)
        r2_sym_norm_reg = np.sqrt(r2_s_sq + rc_sq)
        V_bound_sym = _eval_bound_kernel(r1_sym, r2_sym, r0_sym, rc=rc, gamma=1.0)
        V_bound = V_bound + V_bound_sym

    # Ground Effect (Method of Images)
    r1_img = r2_img = r1_img_reg = r2_img_reg = V_bound_img = None
    r1_img_sym = r2_img_sym = r1_img_sym_reg = r2_img_sym_reg = None
    if h is not None:
        nl_img = nl.copy()
        nl_img[:, 2] = -2.0 * h - nl_img[:, 2]
        nr_img = nr.copy()
        nr_img[:, 2] = -2.0 * h - nr_img[:, 2]
        A_img = nl_img[np.newaxis, :, :]
        B_img = nr_img[np.newaxis, :, :]
        r1_img = P - A_img
        r2_img = P - B_img
        r0_img = B_img - A_img
        r1s_img = (r1_img[..., 0]**2 + r1_img[..., 1]**2 + r1_img[..., 2]**2)[..., None]
        r2s_img = (r2_img[..., 0]**2 + r2_img[..., 1]**2 + r2_img[..., 2]**2)[..., None]
        r1_img_reg = np.sqrt(r1s_img + rc_sq)
        r2_img_reg = np.sqrt(r2s_img + rc_sq)
        V_bound_img = _eval_bound_kernel(r1_img, r2_img, r0_img, rc=rc, gamma=-1.0)

        if use_symmetry and nl_sym is not None and nr_sym is not None:
            nl_img_sym = nl_sym.copy()
            nl_img_sym[:, 2] = -2.0 * h - nl_img_sym[:, 2]
            nr_img_sym = nr_sym.copy()
            nr_img_sym[:, 2] = -2.0 * h - nr_img_sym[:, 2]
            A_img_s = nl_img_sym[np.newaxis, :, :]
            B_img_s = nr_img_sym[np.newaxis, :, :]
            r1_img_sym = P - A_img_s
            r2_img_sym = P - B_img_s
            r0_img_s = B_img_s - A_img_s
            r1s_img_s = (r1_img_sym[..., 0]**2 + r1_img_sym[..., 1]**2 + r1_img_sym[..., 2]**2)[..., None]
            r2s_img_s = (r2_img_sym[..., 0]**2 + r2_img_sym[..., 1]**2 + r2_img_sym[..., 2]**2)[..., None]
            r1_img_sym_reg = np.sqrt(r1s_img_s + rc_sq)
            r2_img_sym_reg = np.sqrt(r2s_img_s + rc_sq)
            V_bound_img_sym = _eval_bound_kernel(r1_img_sym, r2_img_sym, r0_img_s, rc=rc, gamma=-1.0)
            V_bound_img = V_bound_img + V_bound_img_sym

    # Precompute invariant bound-vortex contribution to AIC matrix
    AIC_bound = np.einsum('ijk,ik->ij', V_bound, normals)
    if V_bound_img is not None:
        AIC_bound += np.einsum('ijk,ik->ij', V_bound_img, normals)

    return HorseshoeGeometryCache(
        cp, normals, nl, nr,
        r1, r2, r1_norm_reg, r2_norm_reg, V_bound, disc_surfaces,
        h=h, r1_img=r1_img, r2_img=r2_img, r1_img_reg=r1_img_reg, r2_img_reg=r2_img_reg, V_bound_img=V_bound_img,
        AIC_bound=AIC_bound,
        use_symmetry=use_symmetry,
        nl_sym=nl_sym, nr_sym=nr_sym, r1_sym=r1_sym, r2_sym=r2_sym,
        r1_sym_norm_reg=r1_sym_norm_reg, r2_sym_norm_reg=r2_sym_norm_reg,
        r1_img_sym=r1_img_sym, r2_img_sym=r2_img_sym,
        r1_img_sym_reg=r1_img_sym_reg, r2_img_sym_reg=r2_img_sym_reg,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Linear LLT System Assembly (Phillips & Snyder Modern Lifting-Line Theory)
# ═══════════════════════════════════════════════════════════════════════════════

def build_linear_llt_system(
    disc_surfaces: list[DiscretizedSurface],
    condition: FlightCondition,
    rc: float = VORTEX_CORE_RADIUS,
    return_details: bool = False,
    use_symmetry: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assemble the system matrix and RHS for the Modern Linear LLT solver.

    Follows Phillips & Snyder (2000) / Phillips (2004) Section 1.9 Modern Lifting-Line Theory:
    evaluates bound vortex filaments along the quarter-chord line with collocation/control points
    at the quarter-chord panel centers. Equates the 3D vortex force to 2D section lift:

        [ diag(2 * |v_inf x dl_i| / (a0_i * dA_i)) - AIC_norm ] · Gamma = V_inf * (v_inf · n_i - alpha_L0_i)

    Parameters
    ----------
    disc_surfaces : list[DiscretizedSurface]
        Discretized surfaces (half-mesh if use_symmetry=True).
    condition : FlightCondition
        Operating flight condition.
    rc : float
        Regularization radius.
    return_details : bool
        If True, also returns (V_tot, V_trail).
    use_symmetry : bool
        If True, solves on right semi-span with lateral symmetry reflection.

    Returns
    -------
    A : np.ndarray, shape (N, N)
    rhs : np.ndarray, shape (N,)
    (optional) V_tot : np.ndarray, shape (N, N, 3)
    (optional) V_trail : np.ndarray, shape (N, N, 3)
    """
    if any(getattr(ds, "is_half_mesh", False) for ds in disc_surfaces):
        use_symmetry = True

    # Flatten panel geometry into global arrays
    if len(disc_surfaces) == 1:
        ds0 = disc_surfaces[0]
        if hasattr(ds0, "panel_centers_qc") and len(ds0.panel_centers_qc) == len(ds0.dy_panels):
            cp = ds0.panel_centers_qc
        else:
            cp = 0.5 * (ds0.nodes_qc[:-1] + ds0.nodes_qc[1:])
        normals = ds0.normals
        nl = ds0.nodes_qc[:-1]
        nr = ds0.nodes_qc[1:]
        chords = ds0.chords
        dy = ds0.dy_panels
        a0 = np.array([getattr(af, "a0", 2.0 * np.pi) for af in ds0.airfoils], dtype=float)
        alpha_L0 = np.array([getattr(af, "alpha_L0", 0.0) for af in ds0.airfoils], dtype=float)
    else:
        all_cp: list[np.ndarray] = []
        all_normals: list[np.ndarray] = []
        all_nodes_left: list[np.ndarray] = []
        all_nodes_right: list[np.ndarray] = []
        all_chords: list[np.ndarray] = []
        all_dy: list[np.ndarray] = []
        all_a0: list[float] = []
        all_alpha_L0: list[float] = []

        for ds in disc_surfaces:
            if hasattr(ds, "panel_centers_qc") and len(ds.panel_centers_qc) == len(ds.dy_panels):
                all_cp.append(ds.panel_centers_qc)
            else:
                all_cp.append(0.5 * (ds.nodes_qc[:-1] + ds.nodes_qc[1:]))
            all_normals.append(ds.normals)
            all_nodes_left.append(ds.nodes_qc[:-1])
            all_nodes_right.append(ds.nodes_qc[1:])
            all_chords.append(ds.chords)
            all_dy.append(ds.dy_panels)
            all_a0.extend([getattr(af, "a0", 2.0 * np.pi) for af in ds.airfoils])
            all_alpha_L0.extend([getattr(af, "alpha_L0", 0.0) for af in ds.airfoils])

        cp = np.vstack(all_cp)
        normals = np.vstack(all_normals)
        nl = np.vstack(all_nodes_left)
        nr = np.vstack(all_nodes_right)
        chords = np.concatenate(all_chords)
        dy = np.concatenate(all_dy)
        a0 = np.array(all_a0, dtype=float)
        alpha_L0 = np.array(all_alpha_L0, dtype=float)

    dl = nr - nl
    dA = np.maximum(chords * dy, 1e-12)
    a0 = np.maximum(a0, 1e-6)

    trailing_dir = _trailing_direction(condition)
    v_dir = _freestream_direction(condition)
    V_inf = condition.V_inf

    # Bound vortex cross-product term: |v_inf x dl_i|
    v_cross_dl = np.cross(v_dir, dl)
    mag_v_cross_dl = np.linalg.norm(v_cross_dl, axis=1)
    diag_term = (2.0 * mag_v_cross_dl) / (a0 * dA)

    # Influence tensor evaluation
    # Modern Linear LLT control points are located at the quarter-chord line (collinear with
    # the bound vortex filaments). Analytically, a straight filament induces zero velocity on itself.
    # We enforce V_bound[i, i] = 0 to eliminate numerical floating-point singular roundoff.
    if use_symmetry:
        V_R, V_tr_R = compute_horseshoe_velocity_matrix(
            cp, nl, nr, trailing_dir, gamma=1.0, rc=rc, return_trailing=True
        )
        for i in range(len(cp)):
            V_R[i, i, :] = V_tr_R[i, i, :]

        nl_sym = nr.copy(); nl_sym[:, 1] *= -1.0
        nr_sym = nl.copy(); nr_sym[:, 1] *= -1.0

        V_L, V_tr_L = compute_horseshoe_velocity_matrix(
            cp, nl_sym, nr_sym, trailing_dir, gamma=1.0, rc=rc, return_trailing=True
        )
        V_tot = V_R + V_L
        V_trail = V_tr_R + V_tr_L

        h = condition.h
        if h is not None:
            td_img = np.array([trailing_dir[0], trailing_dir[1], -trailing_dir[2]])
            nl_img = nl.copy(); nl_img[:, 2] = -2.0 * h - nl_img[:, 2]
            nr_img = nr.copy(); nr_img[:, 2] = -2.0 * h - nr_img[:, 2]
            nl_img_sym = nl_sym.copy(); nl_img_sym[:, 2] = -2.0 * h - nl_img_sym[:, 2]
            nr_img_sym = nr_sym.copy(); nr_img_sym[:, 2] = -2.0 * h - nr_img_sym[:, 2]

            V_img_R, V_tr_img_R = compute_horseshoe_velocity_matrix(
                cp, nl_img, nr_img, td_img, gamma=-1.0, rc=rc, return_trailing=True
            )
            V_img_L, V_tr_img_L = compute_horseshoe_velocity_matrix(
                cp, nl_img_sym, nr_img_sym, td_img, gamma=-1.0, rc=rc, return_trailing=True
            )
            V_tot += (V_img_R + V_img_L)
            V_trail += (V_tr_img_R + V_tr_img_L)

    else:
        V_tot, V_trail = compute_horseshoe_velocity_matrix(
            cp, nl, nr, trailing_dir, gamma=1.0, rc=rc, return_trailing=True
        )
        for i in range(len(cp)):
            V_tot[i, i, :] = V_trail[i, i, :]

        h = condition.h
        if h is not None:
            nl_img = nl.copy(); nl_img[:, 2] = -2.0 * h - nl_img[:, 2]
            nr_img = nr.copy(); nr_img[:, 2] = -2.0 * h - nr_img[:, 2]
            td_img = np.array([trailing_dir[0], trailing_dir[1], -trailing_dir[2]])
            V_img, V_img_trail = compute_horseshoe_velocity_matrix(
                cp, nl_img, nr_img, td_img, gamma=-1.0, rc=rc, return_trailing=True
            )
            V_tot += V_img
            V_trail += V_img_trail

    AIC = np.einsum('ijk,ik->ij', V_tot, normals)
    A = np.diag(diag_term) - AIC
    rhs = V_inf * (np.dot(normals, v_dir) - alpha_L0)

    if return_details:
        return A, rhs, V_tot, V_trail
    return A, rhs


class LinearGeometryCache:
    """Precomputed bound-vortex geometry and aerodynamic properties for Linear LLT."""
    def __init__(
        self,
        cp: np.ndarray,
        normals: np.ndarray,
        nl: np.ndarray,
        nr: np.ndarray,
        dl: np.ndarray,
        dA: np.ndarray,
        a0: np.ndarray,
        alpha_L0: np.ndarray,
        r1: np.ndarray,
        r2: np.ndarray,
        r1_norm_reg: np.ndarray,
        r2_norm_reg: np.ndarray,
        V_bound: np.ndarray,
        disc_surfaces: list[DiscretizedSurface],
        h: float | None = None,
        r1_img: np.ndarray | None = None,
        r2_img: np.ndarray | None = None,
        r1_img_reg: np.ndarray | None = None,
        r2_img_reg: np.ndarray | None = None,
        V_bound_img: np.ndarray | None = None,
        AIC_bound: np.ndarray | None = None,
        use_symmetry: bool = False,
        nl_sym: np.ndarray | None = None,
        nr_sym: np.ndarray | None = None,
        r1_sym: np.ndarray | None = None,
        r2_sym: np.ndarray | None = None,
        r1_sym_norm_reg: np.ndarray | None = None,
        r2_sym_norm_reg: np.ndarray | None = None,
        r1_img_sym: np.ndarray | None = None,
        r2_img_sym: np.ndarray | None = None,
        r1_img_sym_reg: np.ndarray | None = None,
        r2_img_sym_reg: np.ndarray | None = None,
    ):
        self.cp = cp
        self.normals = normals
        self.nl = nl
        self.nr = nr
        self.dl = dl
        self.dA = dA
        self.a0 = a0
        self.alpha_L0 = alpha_L0
        self.r1 = r1
        self.r2 = r2
        self.r1_norm_reg = r1_norm_reg
        self.r2_norm_reg = r2_norm_reg
        self.V_bound = V_bound
        self.disc_surfaces = disc_surfaces
        self.h = h
        self.r1_img = r1_img
        self.r2_img = r2_img
        self.r1_img_reg = r1_img_reg
        self.r2_img_reg = r2_img_reg
        self.V_bound_img = V_bound_img
        self.AIC_bound = AIC_bound
        self.use_symmetry = use_symmetry
        self.nl_sym = nl_sym
        self.nr_sym = nr_sym
        self.r1_sym = r1_sym
        self.r2_sym = r2_sym
        self.r1_sym_norm_reg = r1_sym_norm_reg
        self.r2_sym_norm_reg = r2_sym_norm_reg
        self.r1_img_sym = r1_img_sym
        self.r2_img_sym = r2_img_sym
        self.r1_img_sym_reg = r1_img_sym_reg
        self.r2_img_sym_reg = r2_img_sym_reg


def precompute_linear_geometry(
    disc_surfaces: list[DiscretizedSurface],
    h: float | None = None,
    rc: float = VORTEX_CORE_RADIUS,
    use_symmetry: bool = False,
) -> LinearGeometryCache:
    """Precompute bound-vortex kernel and relative position vectors for Linear LLT."""
    if any(getattr(ds, "is_half_mesh", False) for ds in disc_surfaces):
        use_symmetry = True

    if len(disc_surfaces) == 1:
        ds0 = disc_surfaces[0]
        if hasattr(ds0, "panel_centers_qc") and len(ds0.panel_centers_qc) == len(ds0.dy_panels):
            cp = ds0.panel_centers_qc
        else:
            cp = 0.5 * (ds0.nodes_qc[:-1] + ds0.nodes_qc[1:])
        normals = ds0.normals
        nl = ds0.nodes_qc[:-1]
        nr = ds0.nodes_qc[1:]
        chords = ds0.chords
        dy = ds0.dy_panels
        a0 = np.array([getattr(af, "a0", 2.0 * np.pi) for af in ds0.airfoils], dtype=float)
        alpha_L0 = np.array([getattr(af, "alpha_L0", 0.0) for af in ds0.airfoils], dtype=float)
    else:
        all_cp: list[np.ndarray] = []
        all_normals: list[np.ndarray] = []
        all_nodes_left: list[np.ndarray] = []
        all_nodes_right: list[np.ndarray] = []
        all_chords: list[np.ndarray] = []
        all_dy: list[np.ndarray] = []
        all_a0: list[float] = []
        all_alpha_L0: list[float] = []

        for ds in disc_surfaces:
            if hasattr(ds, "panel_centers_qc") and len(ds.panel_centers_qc) == len(ds.dy_panels):
                all_cp.append(ds.panel_centers_qc)
            else:
                all_cp.append(0.5 * (ds.nodes_qc[:-1] + ds.nodes_qc[1:]))
            all_normals.append(ds.normals)
            all_nodes_left.append(ds.nodes_qc[:-1])
            all_nodes_right.append(ds.nodes_qc[1:])
            all_chords.append(ds.chords)
            all_dy.append(ds.dy_panels)
            all_a0.extend([getattr(af, "a0", 2.0 * np.pi) for af in ds.airfoils])
            all_alpha_L0.extend([getattr(af, "alpha_L0", 0.0) for af in ds.airfoils])

        cp = np.vstack(all_cp)
        normals = np.vstack(all_normals)
        nl = np.vstack(all_nodes_left)
        nr = np.vstack(all_nodes_right)
        chords = np.concatenate(all_chords)
        dy = np.concatenate(all_dy)
        a0 = np.array(all_a0, dtype=float)
        alpha_L0 = np.array(all_alpha_L0, dtype=float)

    dl = nr - nl
    dA = np.maximum(chords * dy, 1e-12)
    a0 = np.maximum(a0, 1e-6)

    P = cp[:, np.newaxis, :]
    A = nl[np.newaxis, :, :]
    B = nr[np.newaxis, :, :]
    r1 = P - A
    r2 = P - B
    r0 = B - A
    rc_sq = rc * rc

    r1_norm_sq = (r1[..., 0]**2 + r1[..., 1]**2 + r1[..., 2]**2)[..., None]
    r2_norm_sq = (r2[..., 0]**2 + r2[..., 1]**2 + r2[..., 2]**2)[..., None]
    r1_norm_reg = np.sqrt(r1_norm_sq + rc_sq)
    r2_norm_reg = np.sqrt(r2_norm_sq + rc_sq)

    V_bound = _eval_bound_kernel(r1, r2, r0, rc=rc, gamma=1.0)
    for i in range(len(cp)):
        V_bound[i, i, :] = 0.0

    # Symmetry components
    nl_sym = nr_sym = r1_sym = r2_sym = r1_sym_norm_reg = r2_sym_norm_reg = None
    if use_symmetry:
        nl_sym = nr.copy(); nl_sym[:, 1] *= -1.0
        nr_sym = nl.copy(); nr_sym[:, 1] *= -1.0
        A_sym = nl_sym[np.newaxis, :, :]
        B_sym = nr_sym[np.newaxis, :, :]
        r1_sym = P - A_sym
        r2_sym = P - B_sym
        r0_sym = B_sym - A_sym
        r1s_sq = (r1_sym[..., 0]**2 + r1_sym[..., 1]**2 + r1_sym[..., 2]**2)[..., None]
        r2s_sq = (r2_sym[..., 0]**2 + r2_sym[..., 1]**2 + r2_sym[..., 2]**2)[..., None]
        r1_sym_norm_reg = np.sqrt(r1s_sq + rc_sq)
        r2_sym_norm_reg = np.sqrt(r2s_sq + rc_sq)

        V_bound_sym = _eval_bound_kernel(r1_sym, r2_sym, r0_sym, rc=rc, gamma=1.0)
        V_bound = V_bound + V_bound_sym

    # Ground effect image components
    r1_img = r2_img = r1_img_reg = r2_img_reg = V_bound_img = None
    r1_img_sym = r2_img_sym = r1_img_sym_reg = r2_img_sym_reg = None
    if h is not None:
        nl_img = nl.copy(); nl_img[:, 2] = -2.0 * h - nl_img[:, 2]
        nr_img = nr.copy(); nr_img[:, 2] = -2.0 * h - nr_img[:, 2]
        A_img = nl_img[np.newaxis, :, :]
        B_img = nr_img[np.newaxis, :, :]
        r1_img = P - A_img
        r2_img = P - B_img
        r0_img = B_img - A_img

        r1_img_sq = (r1_img[..., 0]**2 + r1_img[..., 1]**2 + r1_img[..., 2]**2)[..., None]
        r2_img_sq = (r2_img[..., 0]**2 + r2_img[..., 1]**2 + r2_img[..., 2]**2)[..., None]
        r1_img_reg = np.sqrt(r1_img_sq + rc_sq)
        r2_img_reg = np.sqrt(r2_img_sq + rc_sq)
        V_bound_img = _eval_bound_kernel(r1_img, r2_img, r0_img, rc=rc, gamma=-1.0)

        if use_symmetry and nl_sym is not None and nr_sym is not None:
            nl_img_sym = nl_sym.copy(); nl_img_sym[:, 2] = -2.0 * h - nl_img_sym[:, 2]
            nr_img_sym = nr_sym.copy(); nr_img_sym[:, 2] = -2.0 * h - nr_img_sym[:, 2]
            A_img_s = nl_img_sym[np.newaxis, :, :]
            B_img_s = nr_img_sym[np.newaxis, :, :]
            r1_img_sym = P - A_img_s
            r2_img_sym = P - B_img_s
            r0_img_s = B_img_s - A_img_s
            r1s_img_s = (r1_img_sym[..., 0]**2 + r1_img_sym[..., 1]**2 + r1_img_sym[..., 2]**2)[..., None]
            r2s_img_s = (r2_img_sym[..., 0]**2 + r2_img_sym[..., 1]**2 + r2_img_sym[..., 2]**2)[..., None]
            r1_img_sym_reg = np.sqrt(r1s_img_s + rc_sq)
            r2_img_sym_reg = np.sqrt(r2s_img_s + rc_sq)
            V_bound_img_sym = _eval_bound_kernel(r1_img_sym, r2_img_sym, r0_img_s, rc=rc, gamma=-1.0)
            V_bound_img = V_bound_img + V_bound_img_sym

    AIC_bound = np.einsum('ijk,ik->ij', V_bound, normals)
    if V_bound_img is not None:
        AIC_bound += np.einsum('ijk,ik->ij', V_bound_img, normals)

    return LinearGeometryCache(
        cp, normals, nl, nr, dl, dA, a0, alpha_L0,
        r1, r2, r1_norm_reg, r2_norm_reg, V_bound, disc_surfaces,
        h=h, r1_img=r1_img, r2_img=r2_img, r1_img_reg=r1_img_reg, r2_img_reg=r2_img_reg, V_bound_img=V_bound_img,
        AIC_bound=AIC_bound,
        use_symmetry=use_symmetry,
        nl_sym=nl_sym, nr_sym=nr_sym, r1_sym=r1_sym, r2_sym=r2_sym,
        r1_sym_norm_reg=r1_sym_norm_reg, r2_sym_norm_reg=r2_sym_norm_reg,
        r1_img_sym=r1_img_sym, r2_img_sym=r2_img_sym,
        r1_img_sym_reg=r1_img_sym_reg, r2_img_sym_reg=r2_img_sym_reg,
    )






