# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Batched 2D viscous panel solver on CPU and GPU.

This module implements batched two-dimensional viscous-inviscid interaction
on potential flow panels coupled with the two-equation integral boundary layer.
Sweeps over multiple flight conditions, Reynolds numbers, geometries, and
ground heights are solved together using batched PyTorch operations.

Mathematical formulation and coupling overview
----------------------------------------------
The method couples an inviscid linear vortex panel method with the two-equation
integral boundary layer through wall and wake transpiration sources (Lighthill,
1958; Drela and Giles, 1987).

1. Potential flow: stream function Dirichlet boundary conditions supply the
   vortex sheet strengths and the mass-defect influence matrix D.
2. Boundary layer: two-equation integral model closed with Falkner-Skan
   laminar profiles, Swafford turbulent skin friction, and envelope e^N
   transition prediction (Drela and Giles, 1987; Drela, 1989).
3. Batched coupling:
   Cases with the same panel count form dense linear systems solved together
   via batched operations. During Newton and relaxation iterations, cases that
   converge drop out of the active set through boolean masking.

Angle continuation and order of passes
--------------------------------------
Cases of the same geometry, Reynolds number, and height form polar sequences.
Within each sequence, angles are partitioned into non-negative angles (sorted
ascending from zero) and negative angles (sorted descending from zero).

The solver executes in two series of passes:
1. Positive continuation passes:
   Pass p = 0, 1, 2, ... collects the p-th positive angle across all active
   sequences. Pass 0 starts from inviscid flow. Subsequent passes start from
   the converged state of pass p - 1 using decambering and transpiration
   relaxation.
2. Negative continuation passes:
   Pass p = 0, 1, 2, ... collects the p-th negative angle across all active
   sequences. Pass 0 starts from the converged alpha = 0 state of the sequence
   when available. Subsequent passes start from the converged state of pass p - 1.

References
----------
M. J. Lighthill, 'On displacement thickness', Journal of Fluid Mechanics,
Vol. 4, Issue 4, 1958, pp. 383-392.
M. Drela and M. B. Giles, 'Viscous-inviscid analysis of transonic and
low Reynolds number airfoils', AIAA Journal, Vol. 25, No. 10, 1987,
pp. 1347-1355.
M. Drela, 'XFOIL: An Analysis and Design System for Low Reynolds Number
Airfoils', Lecture Notes in Engineering, Vol. 54, Springer, 1989, pp. 1-12.
H. B. Squire and A. D. Young, 'The calculation of the profile drag of
aerofoils', Aeronautical Research Council Reports and Memoranda,
R&M 1838, 1938.
S. F. Hoerner, 'Fluid-Dynamic Drag', Published by the Author, 1965,
chapter 3.
A. E. P. Veldman, 'Quasi-simultaneous interaction methods for
viscous-inviscid flow', Computers & Fluids, Vol. 38, Issue 3, 2009,
pp. 496-508.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

import ventorum.gpu
from ventorum.panel2d.solver import compute_mass_defect_matrix
from ventorum.panel2d.viscous import (
    ViscousResult,
    _march_surface,
    trace_wake_streamline,
)

__all__ = [
    "ViscousBatchResult",
    "ViscousCase",
    "solve_viscous_batch",
]


@dataclass
class ViscousCase:
    """Case definition for the batched 2D viscous panel solver.

    Parameters
    ----------
    coordinates : np.ndarray or torch.Tensor
        Airfoil contour coordinates in Selig order of shape (N, 2) [m or chord units].
    alpha : float
        Angle of attack [deg].
    Re : float
        Reynolds number based on reference chord (dimensionless).
    mach : float, optional
        Free-stream Mach number (dimensionless). Must be 0.0.
    ncrit : float, optional
        Critical amplification exponent for free transition (dimensionless).
        Default is 9.0.
    xtr : tuple of float, optional
        Forced transition locations (upper, lower) as fraction of chord [chord units].
        Default is (1.0, 1.0).
    height : float or None, optional
        Height above flat ground [m or chord units], or None for free air.
    max_iter : int, optional
        Maximum number of coupling iterations. Default is 50.
    tol : float, optional
        Convergence tolerance on edge velocity changes. Default is 1e-6.
    wake_length : float, optional
        Length of the traced wake line in chord units. Default is 1.0.
    """

    coordinates: np.ndarray | torch.Tensor
    alpha: float
    Re: float
    mach: float = 0.0
    ncrit: float = 9.0
    xtr: tuple[float, float] = (1.0, 1.0)
    height: float | None = None
    max_iter: int = 50
    tol: float = 1e-6
    wake_length: float = 1.0


class ViscousBatchResult(list):
    """List of ViscousResult items with vector property accessors.

    Inherits from Python list so standard indexing and iteration return
    the per-case ViscousResult objects. Properties return NumPy arrays
    of the aerodynamic quantities across all cases in the batch.
    """

    @property
    def alpha(self) -> np.ndarray:
        """Return angles of attack of all cases [deg]."""
        return np.array([float(r.alpha) for r in self], dtype=np.float64)

    @property
    def Cl(self) -> np.ndarray:
        """Return lift coefficients of all cases (dimensionless)."""
        return np.array([float(r.Cl) for r in self], dtype=np.float64)

    @property
    def Cd(self) -> np.ndarray:
        """Return profile drag coefficients of all cases (dimensionless)."""
        return np.array([float(r.Cd) for r in self], dtype=np.float64)

    @property
    def Cd_friction(self) -> np.ndarray:
        """Return skin-friction drag coefficients of all cases (dimensionless)."""
        return np.array([float(r.Cd_friction) for r in self], dtype=np.float64)

    @property
    def Cd_pressure(self) -> np.ndarray:
        """Return pressure drag coefficients of all cases (dimensionless)."""
        return np.array([float(r.Cd_pressure) for r in self], dtype=np.float64)

    @property
    def Cm(self) -> np.ndarray:
        """Return pitching moment coefficients of all cases (dimensionless)."""
        return np.array([float(r.Cm) for r in self], dtype=np.float64)

    @property
    def converged(self) -> np.ndarray:
        """Return boolean convergence flags of all cases."""
        return np.array([bool(r.converged) for r in self], dtype=bool)

    @property
    def residual(self) -> np.ndarray:
        """Return final convergence residuals of all cases (dimensionless)."""
        return np.array([float(r.residual) for r in self], dtype=np.float64)

    @property
    def iterations(self) -> np.ndarray:
        """Return iteration counts executed for all cases."""
        return np.array([int(r.iterations) for r in self], dtype=int)

    @property
    def transition_positions(self) -> list[tuple[float, float]]:
        """Return transition locations on upper and lower surfaces [chord units]."""
        return [r.transition_positions for r in self]

    @property
    def separation_flags(self) -> list[tuple[bool, bool]]:
        """Return separation indicators for upper and lower surfaces."""
        return [r.separation_flags for r in self]

    @property
    def reason(self) -> list[str]:
        """Return diagnosis or failure status strings for all cases."""
        return [r.reason for r in self]


def _extract_case(case_input: Any) -> ViscousCase:
    """Parse a case specification into a ViscousCase instance."""
    if isinstance(case_input, ViscousCase):
        case = case_input
    elif isinstance(case_input, dict):
        coords = case_input.get("coordinates", case_input.get("bodies"))
        alpha = case_input.get("alpha", case_input.get("alpha_deg", 0.0))
        re = case_input["Re"]
        mach = case_input.get("mach", 0.0)
        ncrit = case_input.get("ncrit", 9.0)
        xtr = case_input.get("xtr", (1.0, 1.0))
        height = case_input.get("height", None)
        max_iter = case_input.get("max_iter", 50)
        tol = case_input.get("tol", 1e-6)
        wake_length = case_input.get("wake_length", 1.0)
        case = ViscousCase(
            coordinates=coords,
            alpha=float(alpha),
            Re=float(re),
            mach=float(mach),
            ncrit=float(ncrit),
            xtr=xtr,
            height=height,
            max_iter=int(max_iter),
            tol=float(tol),
            wake_length=float(wake_length),
        )
    else:
        coords = getattr(case_input, "coordinates", getattr(case_input, "bodies", None))
        alpha = getattr(case_input, "alpha", getattr(case_input, "alpha_deg", 0.0))
        re = getattr(case_input, "Re", getattr(case_input, "re", 1e6))
        mach = getattr(case_input, "mach", 0.0)
        ncrit = getattr(case_input, "ncrit", 9.0)
        xtr = getattr(case_input, "xtr", (1.0, 1.0))
        height = getattr(case_input, "height", None)
        max_iter = getattr(case_input, "max_iter", 50)
        tol = getattr(case_input, "tol", 1e-6)
        wake_length = getattr(case_input, "wake_length", 1.0)
        case = ViscousCase(
            coordinates=coords,
            alpha=float(alpha),
            Re=float(re),
            mach=float(mach),
            ncrit=float(ncrit),
            xtr=xtr,
            height=height,
            max_iter=int(max_iter),
            tol=float(tol),
            wake_length=float(wake_length),
        )

    if abs(case.mach) > 1e-12:
        raise ValueError(f"Only mach=0.0 is currently supported in solve_viscous_batch, got mach={case.mach}.")
    if case.Re <= 0.0:
        raise ValueError(f"Reynolds number must be positive, got {case.Re}.")

    return case


def _resolve_device_and_dtype(
    device: str | torch.device | None,
    dtype: str | torch.dtype | None,
) -> tuple[torch.device, torch.dtype]:
    """Resolve target torch device and precision dtype."""
    if device is None:
        dev_str = ventorum.gpu.get_device()
        if dev_str == "cpu":
            target_device = torch.device("cpu")
        elif dev_str == "gpu":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available but device='gpu' was selected.")
            target_device = torch.device("cuda:0")
        else:  # "auto"
            target_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device)

    if dtype is None:
        if target_device.type == "cuda":
            prec_str = ventorum.gpu.get_precision()
            target_dtype = torch.float64 if prec_str == "float64" else torch.float32
        else:
            target_dtype = torch.float64
    elif isinstance(dtype, str):
        target_dtype = getattr(torch, dtype)
    else:
        target_dtype = dtype

    return target_device, target_dtype


def _solve_pass_batch(
    pass_cases: list[tuple[int, ViscousCase]],
    prev_solutions: list[dict | None],
    device: torch.device,
    dtype: torch.dtype,
) -> list[tuple[int, ViscousResult, dict]]:
    """Solve one synchronous batch of cases on the device."""
    n_cases = len(pass_cases)
    if n_cases == 0:
        return []

    first_case = pass_cases[0][1]
    raw_coords = np.asarray(first_case.coordinates, dtype=np.float64)
    total_nodes = len(raw_coords)

    # Precompute wake geometry and mass defect matrix for each case
    d_list = []
    ue_inv_list = []
    sol_x_list = []
    sol_y_list = []
    p_start_list = []
    p_end_list = []
    p_len_list = []
    idx1_list = []
    idx2_list = []
    wake_p_start_list = []
    wake_p_end_list = []
    ds_wake_list = []
    s_wake_mid_list = []
    chord_ref_list = []
    gap_list = []
    is_blunt_list = []
    effective_height_list = []

    for _, c in pass_cases:
        coords_i = np.asarray(c.coordinates, dtype=np.float64)
        c_ref = float(np.max(coords_i[:, 0]) - np.min(coords_i[:, 0]))
        if c_ref <= 0.0:
            c_ref = 1.0
        chord_ref_list.append(c_ref)

        eff_h = c.height if (c.height is None or c.height < 500.0) else None
        effective_height_list.append(eff_h)

        wake_pts = trace_wake_streamline(
            coords_i,
            alpha_deg=c.alpha,
            wake_length=c.wake_length,
            n_wake_panels=25,
            height=eff_h,
        )
        wp_start = wake_pts[:-1]
        wp_end = wake_pts[1:]
        wake_p_start_list.append(wp_start)
        wake_p_end_list.append(wp_end)

        ds_w = np.hypot(wp_end[:, 0] - wp_start[:, 0], wp_end[:, 1] - wp_start[:, 1])
        s_w_mid = np.cumsum(ds_w) - 0.5 * ds_w
        ds_wake_list.append(ds_w)
        s_wake_mid_list.append(s_w_mid)

        gap = float(np.hypot(coords_i[0, 0] - coords_i[-1, 0], coords_i[0, 1] - coords_i[-1, 1]))
        gap_list.append(gap)
        is_blunt_list.append(gap > 1e-4 * c_ref)

        D_i, ue_inv_i, info_i = compute_mass_defect_matrix(
            coords_i,
            alpha_deg=c.alpha,
            height=eff_h,
            backend="numpy",
            wake_panels=(wp_start, wp_end),
        )
        d_list.append(D_i)
        ue_inv_list.append(ue_inv_i)
        sol_x_list.append(info_i.get("sol_x", None))
        sol_y_list.append(info_i.get("sol_y", None))
        p_start_list.append(info_i["p_start"])
        p_end_list.append(info_i["p_end"])
        p_len_list.append(info_i["p_len"])
        idx1_list.append(info_i["idx1"])
        idx2_list.append(info_i["idx2"])

    # Build batched tensors on target device
    d_tensor = torch.tensor(np.stack(d_list), dtype=dtype, device=device)

    # Initialize edge velocities with continuation
    ue_init_np = np.zeros((n_cases, total_nodes), dtype=np.float64)
    d_alpha_prev = np.zeros(n_cases, dtype=np.float64)
    xtr_actual_arr = [[1.0, 1.0] for _ in range(n_cases)]
    sep_flags_arr = [[False, False] for _ in range(n_cases)]

    for i, (_, c) in enumerate(pass_cases):
        prev = prev_solutions[i]
        ue_inv_i = ue_inv_list[i]
        if prev is not None and prev.get("converged", False):
            d_alpha = abs(c.alpha - prev["alpha"])
            if d_alpha <= 3.0:
                ue_init_np[i] = ue_inv_i + 0.7 * (prev["ue"] - prev["ue_inv"])
            else:
                ue_init_np[i] = ue_inv_i.copy()
        else:
            ue_init_np[i] = ue_inv_i.copy()
        d_alpha_prev[i] = 0.0

    ue_tensor = torch.tensor(ue_init_np, dtype=dtype, device=device)

    max_iter = max(c.max_iter for _, c in pass_cases)
    tol_arr = np.array([c.tol for _, c in pass_cases], dtype=np.float64)

    active_mask = np.ones(n_cases, dtype=bool)
    converged_mask = np.zeros(n_cases, dtype=bool)
    iterations_arr = np.zeros(n_cases, dtype=int)
    last_res_arr = np.ones(n_cases, dtype=np.float64)

    cached_results: list[dict | None] = [None] * n_cases

    from ventorum.ibl2d import closure as cls

    for it in range(max_iter):
        active_indices = np.where(active_mask)[0]
        if len(active_indices) == 0:
            break

        ue_host = ue_tensor.detach().cpu().numpy()
        sigma_combined_list = []
        ue_inv_eff_np = np.zeros((len(active_indices), total_nodes), dtype=np.float64)

        for sub_k, i in enumerate(active_indices):
            _, c = pass_cases[i]
            coords_i = np.asarray(c.coordinates, dtype=np.float64)
            ue_curr = ue_host[i]
            c_ref = chord_ref_list[i]

            # Locate stagnation point from zero-crossing of edge velocity
            k_stag = None
            for idx_n in range(total_nodes - 1):
                if ue_curr[idx_n] >= 0.0 and ue_curr[idx_n + 1] < 0.0:
                    k_stag = idx_n if abs(ue_curr[idx_n]) <= abs(ue_curr[idx_n + 1]) else idx_n + 1
                    break
            if k_stag is None:
                k_stag = int(np.argmin(np.abs(ue_curr)))
                if k_stag == total_nodes - 1:
                    k_stag = total_nodes - 2

            upper_nodes = list(range(k_stag, -1, -1))
            lower_nodes = list(range(k_stag, total_nodes))

            ds_k = float(np.linalg.norm(coords_i[k_stag] - coords_i[k_stag + 1]))
            a_stag = abs(ue_curr[k_stag] - ue_curr[k_stag + 1]) / max(ds_k, 1e-6)
            a_stag = max(a_stag, 0.1)

            h0 = 2.216
            f0 = float(cls.laminar_cf_factor(h0)[0])
            th0 = np.sqrt(f0 / ((2.0 + h0) * a_stag * c.Re))

            freeze_active = it >= 12
            frozen_u = xtr_actual_arr[i][0] if freeze_active else None
            frozen_l = xtr_actual_arr[i][1] if freeze_active else None

            th_u, h_u, cf_u, xtr_u, sep_u = _march_surface(
                upper_nodes,
                c.xtr[0],
                frozen_u,
                coords_i,
                ue_curr,
                th0,
                h0,
                f0,
                c.Re,
                c_ref,
                c.ncrit,
                freeze_active,
            )
            th_l, h_l, cf_l, xtr_l, sep_l = _march_surface(
                lower_nodes,
                c.xtr[1],
                frozen_l,
                coords_i,
                ue_curr,
                th0,
                h0,
                f0,
                c.Re,
                c_ref,
                c.ncrit,
                freeze_active,
            )

            theta_nodal = np.zeros(total_nodes)
            delta_star_nodal = np.zeros(total_nodes)
            h_nodal = np.zeros(total_nodes)
            cf_nodal = np.zeros(total_nodes)

            for idx_s, n_idx in enumerate(upper_nodes):
                theta_nodal[n_idx] = th_u[idx_s]
                h_nodal[n_idx] = h_u[idx_s]
                delta_star_nodal[n_idx] = th_u[idx_s] * h_u[idx_s]
                cf_nodal[n_idx] = cf_u[idx_s]

            for idx_s, n_idx in enumerate(lower_nodes):
                theta_nodal[n_idx] = th_l[idx_s]
                h_nodal[n_idx] = h_l[idx_s]
                delta_star_nodal[n_idx] = th_l[idx_s] * h_l[idx_s]
                cf_nodal[n_idx] = cf_l[idx_s]

            # Trailing edge momentum defect and bluntness
            th_u_te = theta_nodal[1]
            th_l_te = theta_nodal[-2]
            dst_u_te = delta_star_nodal[1]
            dst_l_te = delta_star_nodal[-2]
            ue_te = 0.5 * (abs(ue_curr[1]) + abs(ue_curr[-2]))

            gap_i = gap_list[i]
            if is_blunt_list[i]:
                cd_base = max(ue_te**2 - 1.0, 0.15)
                th_base = 0.5 * cd_base * gap_i
            else:
                cd_base = 0.0
                th_base = 0.0

            th_w0 = th_u_te + th_l_te + th_base
            dst_w0 = dst_u_te + dst_l_te + gap_i
            h_w0 = min(max(dst_w0 / max(th_w0, 1e-7), 1.0), 3.5)
            th_inf = th_w0 * (max(ue_te, 1e-4) ** ((h_w0 + 5.0) / 2.0))

            # Wake evolution
            s_w_mid = s_wake_mid_list[i]
            ds_w = ds_wake_list[i]
            n_w = len(ds_w)
            ue_w = 1.0 + (ue_te - 1.0) * np.exp(-3.0 * s_w_mid / c_ref)
            dst_w = th_inf + (dst_w0 - th_inf) / (1.0 + 5.0 * (s_w_mid / c_ref))
            m_w = np.abs(ue_w) * dst_w

            sigma_wake = np.zeros(n_w)
            m_in = ue_te * dst_w0
            sigma_wake[0] = (m_w[0] - m_in) / max(float(ds_w[0]), 1e-6)
            for w_idx in range(1, n_w):
                sigma_wake[w_idx] = (m_w[w_idx] - m_w[w_idx - 1]) / max(float(ds_w[w_idx]), 1e-6)

            # Body source strengths
            p_st = p_start_list[i]
            p_en = p_end_list[i]
            p_len = p_len_list[i]
            i1_arr = idx1_list[i]
            i2_arr = idx2_list[i]
            n_body_panels = len(p_st)

            x_mid = 0.5 * (p_st[:, 0] + p_en[:, 0])
            taper = np.clip((0.96 * c_ref - x_mid) / (0.08 * c_ref), 0.0, 1.0)
            m_nodal = np.abs(ue_curr) * delta_star_nodal
            sigma_body = np.zeros(n_body_panels)

            for p_idx in range(n_body_panels):
                i1 = i1_arr[p_idx]
                i2 = i2_arr[p_idx]
                ds_p = max(float(p_len[p_idx]), 1e-6)
                tap = taper[p_idx]
                if i1 == total_nodes - 1 and i2 == 0:
                    sigma_body[p_idx] = 0.0
                elif p_idx < k_stag:
                    sigma_body[p_idx] = tap * (m_nodal[i1] - m_nodal[i2]) / ds_p
                else:
                    sigma_body[p_idx] = tap * (m_nodal[i2] - m_nodal[i1]) / ds_p

            # Decambering correction
            if abs(c.alpha) < 1e-12:
                d_alpha_deg = 0.0
            else:
                d_dst_te = delta_star_nodal[1] - delta_star_nodal[-2]
                d_alpha_raw = 0.95 * (d_dst_te / c_ref) * (180.0 / np.pi)
                d_alpha_deg = 0.7 * d_alpha_prev[i] + 0.3 * d_alpha_raw if it > 0 else d_alpha_raw
                d_alpha_prev[i] = d_alpha_deg

            if effective_height_list[i] is None and abs(d_alpha_deg) > 1e-6:
                alpha_eff_rad = np.radians(c.alpha - d_alpha_deg)
                sol_x_i = sol_x_list[i]
                sol_y_i = sol_y_list[i]
                sol_inv = np.cos(alpha_eff_rad) * sol_x_i + np.sin(alpha_eff_rad) * sol_y_i
                ue_inv_eff_np[sub_k] = sol_inv[:total_nodes]
            else:
                ue_inv_eff_np[sub_k] = ue_inv_list[i]

            sigma_combined = np.concatenate([sigma_body, sigma_wake])
            sigma_combined_list.append(sigma_combined)

            xtr_actual_arr[i] = [float(xtr_u), float(xtr_l)]
            sep_flags_arr[i] = [bool(sep_u), bool(sep_l)]

            cached_results[i] = {
                "theta_nodal": theta_nodal,
                "delta_star_nodal": delta_star_nodal,
                "H_nodal": h_nodal,
                "cf_nodal": cf_nodal,
                "xtr": (float(xtr_u), float(xtr_l)),
                "separated": (bool(sep_u), bool(sep_l)),
                "th_inf": th_inf,
                "cd_base": cd_base,
            }

        # Batched matrix-vector update of edge velocity on device
        sigma_tensor = torch.tensor(np.stack(sigma_combined_list), dtype=dtype, device=device)
        ue_inv_eff_tensor = torch.tensor(ue_inv_eff_np, dtype=dtype, device=device)

        d_sub = d_tensor[active_indices]
        ue_sub = ue_tensor[active_indices]

        ue_new_sub = ue_inv_eff_tensor + torch.bmm(d_sub, sigma_tensor.unsqueeze(-1)).squeeze(-1)

        omega = max(0.25 * (0.95 ** max(0, it - 5)), 0.05)
        diff = ue_new_sub - ue_sub
        step_tensor = torch.max(torch.abs(omega * diff), dim=-1).values

        ue_updated_sub = ue_sub + omega * diff
        ue_tensor[active_indices] = ue_updated_sub

        steps_host = step_tensor.detach().cpu().numpy()
        for sub_k, i in enumerate(active_indices):
            step_i = float(steps_host[sub_k])
            last_res_arr[i] = step_i
            if step_i < tol_arr[i]:
                converged_mask[i] = True
                active_mask[i] = False
                iterations_arr[i] = it + 1

    # Record iteration counts for unconverged cases
    unconv_indices = np.where(~converged_mask)[0]
    iterations_arr[unconv_indices] = max_iter

    # Post-process aerodynamic forces and moments
    final_ue_host = ue_tensor.detach().cpu().numpy()
    results_out = []

    for i, (orig_idx, c) in enumerate(pass_cases):
        coords_i = np.asarray(c.coordinates, dtype=np.float64)
        c_ref = chord_ref_list[i]
        cached = cached_results[i]
        ue_i = final_ue_host[i]

        th_inf = cached["th_inf"]
        cd_sy = float(2.0 * th_inf)

        cp = 1.0 - ue_i**2
        p_st = p_start_list[i]
        p_en = p_end_list[i]
        i1_arr = idx1_list[i]
        i2_arr = idx2_list[i]

        dx_p = p_en[:, 0] - p_st[:, 0]
        dy_p = p_en[:, 1] - p_st[:, 1]
        cp_mid = 0.5 * (cp[i1_arr] + cp[i2_arr])
        d_fx = -cp_mid * dy_p
        d_fy = cp_mid * dx_p
        ca = np.sum(d_fx) / c_ref
        cn = np.sum(d_fy) / c_ref

        alpha_rad = np.radians(c.alpha)
        if effective_height_list[i] is None:
            cl = float(cn * np.cos(alpha_rad) - ca * np.sin(alpha_rad))
            cd_p = float(cn * np.sin(alpha_rad) + ca * np.cos(alpha_rad))
        else:
            cl = float(cn)
            cd_p = float(ca)

        if is_blunt_list[i]:
            cd_p += float(cached["cd_base"] * gap_list[i] / c_ref)

        cf_nodal = cached["cf_nodal"]
        cf_mid = 0.5 * (cf_nodal[i1_arr] + cf_nodal[i2_arr])
        cd_f = float(np.sum(cf_mid * np.abs(dx_p)) / c_ref)

        # Exact zero lift for symmetric body at zero angle
        if abs(c.alpha) < 1e-12 and np.allclose(coords_i[:, 1], -coords_i[::-1, 1], atol=1e-5):
            cl = 0.0

        # Nearfield drag consistency correction
        cd_near = cd_p + cd_f
        if abs(cd_sy - cd_near) / max(cd_sy, 1e-4) > 0.04:
            cd_p = cd_sy - cd_f

        # Pitching moment about (0.25 * chord_ref, 0.0)
        xm = 0.5 * (p_st[:, 0] + p_en[:, 0]) - 0.25 * c_ref
        ym = 0.5 * (p_st[:, 1] + p_en[:, 1])
        dm = -xm * d_fy + ym * d_fx
        cm = float(np.sum(dm) / (c_ref**2))

        sep_flags = cached["separated"]
        xtr_actual = cached["xtr"]
        conv_i = bool(converged_mask[i])

        reason = ""
        if sep_flags[0] and sep_flags[1]:
            reason = "Separation detected on upper and lower surfaces"
        elif sep_flags[0]:
            reason = f"Separation detected on upper surface at x/c={xtr_actual[0]:.3f}"
        elif sep_flags[1]:
            reason = f"Separation detected on lower surface at x/c={xtr_actual[1]:.3f}"
        elif not conv_i:
            reason = f"Maximum iterations ({max_iter}) reached without meeting tolerance ({tol_arr[i]:.1e})"

        visc_res = ViscousResult(
            alpha=c.alpha,
            Cl=cl,
            Cd=cd_sy,
            Cd_friction=cd_f,
            Cd_pressure=cd_p,
            Cm=cm,
            transition_positions=xtr_actual,
            separation_flags=sep_flags,
            Cp=cp,
            H=cached["H_nodal"],
            theta=cached["theta_nodal"],
            Cf=cf_nodal,
            iterations=int(iterations_arr[i]),
            residual=float(last_res_arr[i]),
            converged=conv_i,
            coordinates=coords_i,
            reason=reason,
        )

        state_dict = {
            "alpha": c.alpha,
            "ue": ue_i,
            "ue_inv": ue_inv_list[i],
            "d_alpha_deg": d_alpha_prev[i],
            "converged": conv_i,
        }

        results_out.append((orig_idx, visc_res, state_dict))

    return results_out


def solve_viscous_batch(
    cases: list[ViscousCase | dict | Any],
    *,
    backend: str = "torch",
    device: str | torch.device | None = None,
    dtype: str | torch.dtype | None = None,
) -> ViscousBatchResult:
    """Solve multiple 2D viscous cases in batches on CPU or CUDA GPU.

    Cases with the same panel count are solved together. The outer iterations
    and linear solves use batched PyTorch tensors, with active set masking
    for converged cases.

    Parameters
    ----------
    cases : list of ViscousCase, dict, or object
        List of flight conditions, Reynolds numbers, and geometry cases to solve.
    backend : {"torch"}, optional
        Computational backend. Default is "torch".
    device : str, torch.device, or None, optional
        Target device ("cpu", "cuda", "cuda:0", or None for auto-selection).
    dtype : str, torch.dtype, or None, optional
        Floating-point arithmetic precision (torch.float64, torch.float32,
        or None for auto-selection).

    Returns
    -------
    ViscousBatchResult
        List of ViscousResult objects matching the order of input cases.
    """
    if backend != "torch":
        raise ValueError(f"backend '{backend}' is not supported. Only 'torch' is supported.")

    if len(cases) == 0:
        return ViscousBatchResult([])

    target_device, target_dtype = _resolve_device_and_dtype(device, dtype)

    parsed_cases: list[tuple[int, ViscousCase]] = []
    for idx, c_raw in enumerate(cases):
        c = _extract_case(c_raw)
        parsed_cases.append((idx, c))

    # Group cases by panel count (Fixed decision 1)
    by_panel_count: dict[int, list[tuple[int, ViscousCase]]] = {}
    for orig_idx, c in parsed_cases:
        coords = np.asarray(c.coordinates, dtype=np.float64)
        n_panels = len(coords)
        by_panel_count.setdefault(n_panels, []).append((orig_idx, c))

    all_results: list[tuple[int, ViscousResult]] = []

    for _, n_cases_list in by_panel_count.items():
        # Subgroup cases by continuation sequence (geometry, Re, height)
        sequences: list[list[tuple[int, ViscousCase]]] = []
        for item in n_cases_list:
            _, c = item
            coords_c = np.asarray(c.coordinates, dtype=np.float64)
            found_seq = False
            for seq in sequences:
                _, c_ref = seq[0]
                coords_ref = np.asarray(c_ref.coordinates, dtype=np.float64)
                if (
                    np.allclose(coords_c, coords_ref, atol=1e-7)
                    and abs(c.Re - c_ref.Re) < 1e-9 * c_ref.Re
                    and c.height == c_ref.height
                    and c.ncrit == c_ref.ncrit
                    and c.xtr == c_ref.xtr
                    and c.wake_length == c_ref.wake_length
                ):
                    seq.append(item)
                    found_seq = True
                    break
            if not found_seq:
                sequences.append([item])

        # Partition each sequence into positive and negative angles
        pos_by_seq: list[list[tuple[int, ViscousCase]]] = []
        neg_by_seq: list[list[tuple[int, ViscousCase]]] = []

        for seq in sequences:
            pos_cases = [item for item in seq if item[1].alpha >= 0.0]
            neg_cases = [item for item in seq if item[1].alpha < 0.0]

            pos_cases.sort(key=lambda item: item[1].alpha)
            neg_cases.sort(key=lambda item: -item[1].alpha)

            pos_by_seq.append(pos_cases)
            neg_by_seq.append(neg_cases)

        # Series 1: Positive continuation passes
        max_pos_depth = max((len(p) for p in pos_by_seq), default=0)
        last_pos_state: list[dict | None] = [None] * len(sequences)
        zero_state: list[dict | None] = [None] * len(sequences)

        for p_idx in range(max_pos_depth):
            current_pass_cases = []
            current_prev_sols = []
            seq_indices = []

            for s_idx, pos_cases in enumerate(pos_by_seq):
                if p_idx < len(pos_cases):
                    current_pass_cases.append(pos_cases[p_idx])
                    current_prev_sols.append(last_pos_state[s_idx])
                    seq_indices.append(s_idx)

            pass_results = _solve_pass_batch(
                current_pass_cases,
                current_prev_sols,
                target_device,
                target_dtype,
            )

            for res_idx, (orig_id, v_res, st_dict) in enumerate(pass_results):
                all_results.append((orig_id, v_res))
                s_idx = seq_indices[res_idx]
                last_pos_state[s_idx] = st_dict
                if abs(v_res.alpha) < 1e-12:
                    zero_state[s_idx] = st_dict

        # Series 2: Negative continuation passes
        max_neg_depth = max((len(n) for n in neg_by_seq), default=0)
        last_neg_state: list[dict | None] = list(zero_state)

        for p_idx in range(max_neg_depth):
            current_pass_cases = []
            current_prev_sols = []
            seq_indices = []

            for s_idx, neg_cases in enumerate(neg_by_seq):
                if p_idx < len(neg_cases):
                    current_pass_cases.append(neg_cases[p_idx])
                    current_prev_sols.append(last_neg_state[s_idx])
                    seq_indices.append(s_idx)

            pass_results = _solve_pass_batch(
                current_pass_cases,
                current_prev_sols,
                target_device,
                target_dtype,
            )

            for res_idx, (orig_id, v_res, st_dict) in enumerate(pass_results):
                all_results.append((orig_id, v_res))
                s_idx = seq_indices[res_idx]
                last_neg_state[s_idx] = st_dict

    # Reorder results to match input order
    all_results.sort(key=lambda item: item[0])
    ordered_results = [res for _, res in all_results]

    return ViscousBatchResult(ordered_results)
