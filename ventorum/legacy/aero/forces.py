# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Force and moment integration from the circulation distribution.
"""

from __future__ import annotations

from typing import Any
import numpy as np

from ventorum.legacy.aero.biot_savart import horseshoe_velocity, semi_infinite_vortex_velocity
from ventorum.legacy.core.constants import VORTEX_CORE_RADIUS
from ventorum.legacy.core.datatypes import (
    DiscretizedSurface,
    FlightCondition,
    IntegratedResult,
    SpanwiseResult,
)
from ventorum.legacy.core.trust import evaluate_aerodynamic_trust


def _trailing_direction(condition: FlightCondition) -> np.ndarray:
    ca, sa = np.cos(condition.alpha), np.sin(condition.alpha)
    return np.array([ca, 0.0, sa])


def _freestream_direction(condition: FlightCondition) -> np.ndarray:
    ca, sa = np.cos(condition.alpha), np.sin(condition.alpha)
    cb, sb = np.cos(condition.beta), np.sin(condition.beta)
    return np.array([ca * cb, -sb, sa * cb])


def compute_spanwise_horseshoe(
    disc_surfaces: list[DiscretizedSurface],
    Gamma_global: np.ndarray,
    condition: FlightCondition,
    rc: float = VORTEX_CORE_RADIUS,
    V_trail: np.ndarray | None = None,
    w_ind_z: np.ndarray | None = None,
) -> list[SpanwiseResult]:
    """Compute per-surface spanwise distributions from the horseshoe solver
    circulation vector.

    Parameters
    ----------
    disc_surfaces : list[DiscretizedSurface]
    Gamma_global : np.ndarray, shape (N_total,)
        Circulation at each panel (global ordering).
    condition : FlightCondition
    rc : float
        Regularization radius.
    V_trail : np.ndarray or None
        Optional precomputed trailing-leg velocity influence matrix of shape (N_total, N_total, 3).
        If provided, avoids recalculating Biot-Savart trailing filaments.
    w_ind_z : np.ndarray or None
        Optional precomputed vertical induced downwash vector of shape (N_total,).
        If provided, eliminates PCIe tensor copies from GPU solvers.

    Returns
    -------
    list[SpanwiseResult]
        One entry per surface.
    """
    V_inf = condition.V_inf
    rho = condition.rho

    # Compute downwash at all control points
    all_n_panels = [len(ds.y_panels) for ds in disc_surfaces]
    N_tot = sum(all_n_panels)

    if w_ind_z is not None:
        w_ind_z_all = w_ind_z
    elif V_trail is not None:
        # Fast BLAS matrix-vector multiply for induced downwash (vertical z-component only)
        w_ind_z_all = V_trail[:, :, 2] @ Gamma_global
    else:
        # Fallback: compute trailing downwash directly using fast kernel
        all_cp = np.vstack([ds.control_points for ds in disc_surfaces])
        all_nl = np.vstack([ds.nodes_qc[:-1] for ds in disc_surfaces])
        all_nr = np.vstack([ds.nodes_qc[1:] for ds in disc_surfaces])
        trailing_dir = _trailing_direction(condition)

        P = all_cp[:, np.newaxis, :]
        r1 = P - all_nl[np.newaxis, :, :]
        r2 = P - all_nr[np.newaxis, :, :]
        rc_sq = rc * rc
        inv_4pi = 1.0 / (4.0 * np.pi)

        td0, td1, td2 = trailing_dir[0], trailing_dir[1], trailing_dir[2]
        cross_A = np.stack([
            td1 * r1[..., 2] - td2 * r1[..., 1],
            td2 * r1[..., 0] - td0 * r1[..., 2],
            td0 * r1[..., 1] - td1 * r1[..., 0],
        ], axis=-1)
        denom_A = (cross_A[..., 0]**2 + cross_A[..., 1]**2 + cross_A[..., 2]**2)[..., None] + rc_sq
        r1_norm_reg = np.sqrt((r1[..., 0]**2 + r1[..., 1]**2 + r1[..., 2]**2)[..., None] + rc_sq)
        cos_theta_A = (td0 * r1[..., 0] + td1 * r1[..., 1] + td2 * r1[..., 2])[..., None] / r1_norm_reg
        scale_A = np.where(denom_A > 0.0, -(inv_4pi) * (1.0 + cos_theta_A) / denom_A, 0.0)

        cross_B = np.stack([
            td1 * r2[..., 2] - td2 * r2[..., 1],
            td2 * r2[..., 0] - td0 * r2[..., 2],
            td0 * r2[..., 1] - td1 * r2[..., 0],
        ], axis=-1)
        denom_B = (cross_B[..., 0]**2 + cross_B[..., 1]**2 + cross_B[..., 2]**2)[..., None] + rc_sq
        r2_norm_reg = np.sqrt((r2[..., 0]**2 + r2[..., 1]**2 + r2[..., 2]**2)[..., None] + rc_sq)
        cos_theta_B = (td0 * r2[..., 0] + td1 * r2[..., 1] + td2 * r2[..., 2])[..., None] / r2_norm_reg
        scale_B = np.where(denom_B > 0.0, inv_4pi * (1.0 + cos_theta_B) / denom_B, 0.0)

        V_trail_z = cross_A[:, :, 2] * scale_A[:, :, 0] + cross_B[:, :, 2] * scale_B[:, :, 0]
        w_ind_z_all = V_trail_z @ Gamma_global

    results: list[SpanwiseResult] = []
    offset = 0

    for ds, n_panels in zip(disc_surfaces, all_n_panels):
        Gamma_surf = Gamma_global[offset:offset + n_panels]
        alpha_i = -w_ind_z_all[offset:offset + n_panels] / V_inf  # small-angle approx
        v_dir = _freestream_direction(condition)
        alpha_geom = np.arcsin(np.clip(np.dot(ds.normals, v_dir), -1.0, 1.0))
        alpha_eff = alpha_geom - alpha_i

        # Local aerodynamic coefficients
        Cl = 2.0 * Gamma_surf / (V_inf * ds.chords)
        Cd_i = Cl * alpha_i  # Trefftz-plane decomposition

        # Profile drag from section polars (if available)
        Cd_profile = None
        has_prof = getattr(ds, "has_profile_drag", None)
        if has_prof is None:
            has_prof = any(
                getattr(af, "Cd0", None) is None or getattr(af, "Cd0", 0.0) > 0.0
                for af in ds.airfoils
            )
            groups = None
        else:
            groups = getattr(ds, "airfoil_groups", None)

        if has_prof:
            Cd_profile = np.zeros(n_panels, dtype=float)
            if groups is not None:
                for uaf, idx in groups:
                    cd_vals = uaf.Cd(alpha_eff[idx])
                    Cd_profile[idx] = np.atleast_1d(cd_vals)
            else:
                unique_airfoils: list[Any] = []
                for af in ds.airfoils:
                    if not any(af is u for u in unique_airfoils):
                        unique_airfoils.append(af)
                for uaf in unique_airfoils:
                    idx = np.where([af is uaf for af in ds.airfoils])[0]
                    cd_vals = uaf.Cd(alpha_eff[idx])
                    Cd_profile[idx] = np.atleast_1d(cd_vals)
            if not np.any(Cd_profile > 0):
                Cd_profile = None

        local_lift = rho * V_inf * Gamma_surf  # L'(y) = ρ V∞ Γ

        results.append(SpanwiseResult(
            y=ds.y_panels.copy(),
            gamma=Gamma_surf.copy(),
            Cl=Cl,
            Cd_i=Cd_i,
            Cd_profile=Cd_profile,
            alpha_eff=alpha_eff,
            alpha_i=alpha_i,
            local_lift=local_lift,
            surface_name=ds.surface_name,
        ))
        offset += n_panels

    return results


def integrate_results(
    spanwise_list: list[SpanwiseResult],
    disc_surfaces: list[DiscretizedSurface],
    condition: FlightCondition,
    S_ref: float,
    b_ref: float,
    c_ref: float | None = None,
    ref_point: np.ndarray | None = None,
) -> IntegratedResult:
    """Integrate spanwise data into total aerodynamic coefficients.

    Parameters
    ----------
    spanwise_list : list[SpanwiseResult]
    disc_surfaces : list[DiscretizedSurface]
    condition : FlightCondition
    S_ref : float
        Reference wing area [m²].
    b_ref : float
        Reference span [m].
    c_ref : float | None
        Reference chord (Mean Aerodynamic Chord) [m].
    ref_point : np.ndarray | None
        Center of moments [x_ref, y_ref, z_ref]. Defaults to aircraft origin [0, 0, 0].

    Returns
    -------
    IntegratedResult
    """
    q_inf = 0.5 * condition.rho * condition.V_inf ** 2

    total_lift = 0.0
    total_Di = 0.0
    total_Dp = 0.0
    has_profile = False

    total_Mx = 0.0
    total_My = 0.0
    total_Mz = 0.0

    if c_ref is None:
        c_ref = S_ref / b_ref if b_ref > 0 else 1.0

    ca, sa = np.cos(condition.alpha), np.sin(condition.alpha)
    rp = np.zeros(3, dtype=float) if ref_point is None else np.asarray(ref_point, dtype=float)

    for sw, ds in zip(spanwise_list, disc_surfaces):
        L_vec = sw.local_lift * ds.dy_panels
        total_lift += np.sum(L_vec)

        Di_vec = q_inf * ds.chords * sw.Cd_i * ds.dy_panels
        total_Di += np.sum(Di_vec)

        if sw.Cd_profile is not None:
            Dp_vec = q_inf * ds.chords * sw.Cd_profile * ds.dy_panels
            total_Dp += np.sum(Dp_vec)
            has_profile = True

        # Moments about specified reference point (quarter-chord panel centers)
        if hasattr(ds, "panel_centers_qc") and len(ds.panel_centers_qc) == len(ds.dy_panels):
            pt = ds.panel_centers_qc
        else:
            pt = 0.5 * (ds.nodes_qc[:-1] + ds.nodes_qc[1:])
        x = pt[:, 0] - rp[0]
        y = pt[:, 1] - rp[1]
        z = pt[:, 2] - rp[2]

        # Bound-vortex filament vector for each panel
        d_nodes = np.diff(ds.nodes_qc, axis=0)
        dy_seg = d_nodes[:, 1]
        dz_seg = d_nodes[:, 2]
        has_tilt = np.any(np.abs(dz_seg) > 1e-12)

        if has_tilt:
            # 3D force decomposition: vertical and lateral lift components
            L_vert = condition.rho * condition.V_inf * sw.gamma * dy_seg
            L_lat = -condition.rho * condition.V_inf * sw.gamma * dz_seg

            Fx = -Di_vec * ca + L_vert * sa
            Fz = L_vert * ca + Di_vec * sa
            Fy = L_lat
        else:
            Fx = -Di_vec * ca + L_vec * sa
            Fz = L_vec * ca + Di_vec * sa
            Fy = 0.0

        total_Mx += np.sum(y * Fz - z * Fy)
        total_My += np.sum(z * Fx - x * Fz)
        total_Mz += np.sum(x * Fy - y * Fx)

    CL = total_lift / (q_inf * S_ref)
    CDi = total_Di / (q_inf * S_ref)
    CDp = total_Dp / (q_inf * S_ref) if has_profile else None
    CD_total = CDi + (CDp if CDp is not None else 0.0)

    Cl_mom = total_Mx / (q_inf * S_ref * b_ref)
    Cm_mom = total_My / (q_inf * S_ref * c_ref)
    Cn_mom = total_Mz / (q_inf * S_ref * b_ref)

    AR = b_ref ** 2 / S_ref
    # Span efficiency factor (uncapped in ground effect where e naturally exceeds 1.5)
    e_denom = np.pi * AR * CDi if CDi > 1e-14 else 1.0
    e = CL ** 2 / e_denom if CDi > 1e-14 else 1.0
    if condition.h is None:
        e = min(e, 1.5)

    n_p = len(disc_surfaces[0].y_panels) if disc_surfaces and len(disc_surfaces[0].y_panels) > 0 else 80
    trust_eval = evaluate_aerodynamic_trust(
        AR=AR,
        surfaces=disc_surfaces,
        condition=condition,
        spanwise_list=spanwise_list,
        CL=CL,
        CDi=CDi,
        CD_total=CD_total if has_profile else None,
        converged=True,
        n_panels=n_p,
        b_ref=b_ref,
    )

    return IntegratedResult(
        CL=CL,
        CDi=CDi,
        CDp=CDp,
        CD_total=CD_total if has_profile else None,
        e=e,
        AR=AR,
        Cl=Cl_mom,
        Cm=Cm_mom,
        Cn=Cn_mom,
        trust=trust_eval,
    )
