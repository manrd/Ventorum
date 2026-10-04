# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
GPU-accelerated Lifting Line Theory solver using horseshoe vortices
and PyTorch / CUDA batched linear algebra.

Provides extreme acceleration for high-discretization wings and
massive batched angle-of-attack / parameter sweeps.
"""

from __future__ import annotations

import copy
from typing import Sequence, Any
import numpy as np

try:
    import torch
    from ventorum.legacy.aero.gpu_influence import (
        compute_batched_aic_and_rhs,
        precompute_gpu_geometry_cache,
        get_device,
        has_cuda,
        GPUGeometryCache,
    )
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from ventorum.legacy.aero.forces import compute_spanwise_horseshoe, integrate_results
from ventorum.legacy.core.constants import VORTEX_CORE_RADIUS
from ventorum.legacy.core.datatypes import (
    Aircraft,
    DiscretizedSurface,
    FlightCondition,
    SolverResult,
    SolverSettings,
    SpanwiseResult,
    IntegratedResult,
)
from ventorum.legacy.geometry.processing import discretize_surface, discretize_aircraft_surfaces
from ventorum.legacy.core.symmetry import (
    can_use_symmetry,
    reconstruct_full_circulation,
    reconstruct_full_downwash,
    mirror_discretized_surface,
)
from ventorum.legacy.solvers.base import BaseSolver
from ventorum.legacy.utils.validation import validate_aircraft


class GPUHorseshoeSolver(BaseSolver):
    """GPU-accelerated Horseshoe LLT solver supporting single and batched execution."""

    def __init__(
        self,
        device: str | None = None,
        dtype: str = "float64",
    ):
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required to use GPUHorseshoeSolver.")
        self.device = get_device(device)
        self.dtype = torch.float64 if dtype == "float64" else torch.float32

    def solve_discretized(
        self,
        disc_surfaces: list[DiscretizedSurface],
        condition: FlightCondition,
        S_ref: float,
        b_ref: float,
        c_ref: float | None = None,
        rc: float = VORTEX_CORE_RADIUS,
        use_symmetry: bool = False,
    ) -> SolverResult:
        """Solve a pre-discretized aircraft system on GPU."""
        is_half = use_symmetry or any(getattr(ds, "is_half_mesh", False) for ds in disc_surfaces)

        if len(disc_surfaces) == 1:
            ds0 = disc_surfaces[0]
            cp = ds0.control_points
            normals = ds0.normals
            nl = ds0.nodes_qc[:-1]
            nr = ds0.nodes_qc[1:]
        else:
            cp = np.vstack([ds.control_points for ds in disc_surfaces])
            normals = np.vstack([ds.normals for ds in disc_surfaces])
            nl = np.vstack([ds.nodes_qc[:-1] for ds in disc_surfaces])
            nr = np.vstack([ds.nodes_qc[1:] for ds in disc_surfaces])

        ca, sa = np.cos(condition.alpha), np.sin(condition.alpha)
        cb, sb = np.cos(condition.beta), np.sin(condition.beta)
        trailing_dir = np.array([ca, 0.0, sa])
        V_inf_vec = condition.V_inf * np.array([ca * cb, -sb, sa * cb])

        # Transfer to GPU
        cp_t = torch.from_numpy(cp).to(device=self.device, dtype=self.dtype)
        nl_t = torch.from_numpy(nl).to(device=self.device, dtype=self.dtype)
        nr_t = torch.from_numpy(nr).to(device=self.device, dtype=self.dtype)
        norm_t = torch.from_numpy(normals).to(device=self.device, dtype=self.dtype)
        td_t = torch.from_numpy(trailing_dir).to(device=self.device, dtype=self.dtype)
        vinf_t = torch.from_numpy(V_inf_vec).to(device=self.device, dtype=self.dtype)

        AIC, rhs, _, V_trail = compute_batched_aic_and_rhs(
            cp=cp_t,
            nl=nl_t,
            nr=nr_t,
            normals=norm_t,
            trailing_dir=td_t,
            V_inf_vec=vinf_t,
            rc=rc,
            h=condition.h,
            return_details=True,
            use_symmetry=is_half,
        )

        Gamma_t = torch.linalg.solve(AIC, rhs)
        w_ind_z_t = torch.mv(V_trail[:, :, 2], Gamma_t)
        Gamma_raw = Gamma_t.cpu().numpy()
        w_ind_z_raw = w_ind_z_t.cpu().numpy()

        if is_half:
            disc_surfaces_eval = [mirror_discretized_surface(ds) for ds in disc_surfaces]
            Gamma = reconstruct_full_circulation(Gamma_raw, disc_surfaces)
            w_ind_z = reconstruct_full_downwash(w_ind_z_raw, disc_surfaces)
        else:
            disc_surfaces_eval = disc_surfaces
            Gamma = Gamma_raw
            w_ind_z = w_ind_z_raw

        spanwise_list = compute_spanwise_horseshoe(
            disc_surfaces_eval, Gamma, condition, w_ind_z=w_ind_z
        )
        totals = integrate_results(
            spanwise_list,
            disc_surfaces_eval,
            condition,
            S_ref=S_ref,
            b_ref=b_ref,
            c_ref=c_ref,
        )

        return SolverResult(
            spanwise=spanwise_list,
            totals=totals,
            solver_type="horseshoe_gpu",
            converged=True,
            iterations=1,
            symmetry_used=is_half,
        )

    def solve(
        self,
        aircraft: Aircraft | Any,
        condition: FlightCondition,
        settings: SolverSettings,
    ) -> SolverResult:
        """Standard solve entry point executing on GPU."""
        if hasattr(aircraft, "sections") and not hasattr(aircraft, "surfaces"):
            aircraft = Aircraft(name=getattr(aircraft, "name", "SingleWing"), surfaces=[aircraft])
        validate_aircraft(aircraft)
        aircraft.compute_reference_values()

        use_sym = can_use_symmetry(aircraft, condition, settings)
        disc_surfaces: list[DiscretizedSurface] = discretize_aircraft_surfaces(
            aircraft, settings, half_mesh=use_sym
        )

        return self.solve_discretized(
            disc_surfaces,
            condition,
            aircraft.S_ref,
            aircraft.b_ref,
            aircraft.c_ref,
            use_symmetry=use_sym,
        )

    def solve_sweep(
        self,
        aircraft: Aircraft | Any,
        condition: FlightCondition,
        settings: SolverSettings,
        alpha_range: np.ndarray,
        rc: float = VORTEX_CORE_RADIUS,
    ) -> list[SolverResult]:
        """Solve an entire angle-of-attack sweep in a single batched GPU kernel execution.

        Precomputes the invariant bound vortex kernel once, then evaluates trailing
        downwash and inverts all B linear systems concurrently in GPU memory.
        """
        if hasattr(aircraft, "sections") and not hasattr(aircraft, "surfaces"):
            aircraft = Aircraft(name=getattr(aircraft, "name", "SingleWing"), surfaces=[aircraft])
        validate_aircraft(aircraft)
        aircraft.compute_reference_values()

        disc_surfaces = discretize_aircraft_surfaces(aircraft, settings)

        if len(disc_surfaces) == 1:
            ds0 = disc_surfaces[0]
            cp = ds0.control_points
            normals = ds0.normals
            nl = ds0.nodes_qc[:-1]
            nr = ds0.nodes_qc[1:]
        else:
            cp = np.vstack([ds.control_points for ds in disc_surfaces])
            normals = np.vstack([ds.normals for ds in disc_surfaces])
            nl = np.vstack([ds.nodes_qc[:-1] for ds in disc_surfaces])
            nr = np.vstack([ds.nodes_qc[1:] for ds in disc_surfaces])

        cache = precompute_gpu_geometry_cache(
            cp_np=cp,
            normals_np=normals,
            nl_np=nl,
            nr_np=nr,
            h=condition.h,
            rc=rc,
            device=self.device,
            dtype=self.dtype,
        )

        B = len(alpha_range)
        alphas = np.asarray(alpha_range, dtype=float)
        ca = np.cos(alphas)
        sa = np.sin(alphas)
        cb = np.cos(condition.beta)
        sb = np.sin(condition.beta)

        td_np = np.stack([ca, np.zeros_like(ca), sa], axis=-1)
        vinf_np = condition.V_inf * np.stack([ca * cb, np.full_like(ca, -sb), sa * cb], axis=-1)

        td_t = torch.from_numpy(td_np).to(device=self.device, dtype=self.dtype)       # (B, 3)
        vinf_t = torch.from_numpy(vinf_np).to(device=self.device, dtype=self.dtype)   # (B, 3)

        rc_sq = rc * rc
        inv_4pi = 1.0 / (4.0 * np.pi)

        # Invariant bound & control point tensors
        r1 = cache.r1.unsqueeze(0)  # (1, N, N, 3)
        r2 = cache.r2.unsqueeze(0)  # (1, N, N, 3)
        r1_reg = cache.r1_norm_reg.unsqueeze(0)
        r2_reg = cache.r2_norm_reg.unsqueeze(0)

        chunk_size = 2000
        Gamma_chunks = []
        w_ind_z_chunks = []

        for start_idx in range(0, B, chunk_size):
            end_idx = min(start_idx + chunk_size, B)
            b_chunk = end_idx - start_idx

            td_c = td_t[start_idx:end_idx].unsqueeze(1).unsqueeze(1) # (b_chunk, 1, 1, 3)
            vinf_c = vinf_t[start_idx:end_idx]                        # (b_chunk, 3)

            cross_A = torch.linalg.cross(td_c.expand(b_chunk, *cache.r1.shape), r1.expand(b_chunk, *cache.r1.shape))
            denom_A = torch.sum(cross_A**2, dim=-1, keepdim=True) + rc_sq
            cos_theta_A = torch.sum(td_c * r1, dim=-1, keepdim=True) / r1_reg
            scale_A = torch.where(
                denom_A > 0.0,
                -inv_4pi * (1.0 + cos_theta_A) / torch.clamp(denom_A, min=1e-30),
                torch.zeros_like(denom_A),
            )
            V_left = cross_A * scale_A

            cross_B = torch.linalg.cross(td_c.expand(b_chunk, *cache.r2.shape), r2.expand(b_chunk, *cache.r2.shape))
            denom_B = torch.sum(cross_B**2, dim=-1, keepdim=True) + rc_sq
            cos_theta_B = torch.sum(td_c * r2, dim=-1, keepdim=True) / r2_reg
            scale_B = torch.where(
                denom_B > 0.0,
                inv_4pi * (1.0 + cos_theta_B) / torch.clamp(denom_B, min=1e-30),
                torch.zeros_like(denom_B),
            )
            V_right = cross_B * scale_B
            V_trail = V_left + V_right

            if cache.h is not None and cache.r1_img is not None:
                td_img = td_t[start_idx:end_idx].clone()
                td_img[:, 2] = -td_img[:, 2]
                td_i = td_img.unsqueeze(1).unsqueeze(1)
                r1_i = cache.r1_img.unsqueeze(0)
                r2_i = cache.r2_img.unsqueeze(0)
                r1_ir = cache.r1_img_reg.unsqueeze(0)
                r2_ir = cache.r2_img_reg.unsqueeze(0)

                cA_i = torch.linalg.cross(td_i.expand(b_chunk, *cache.r1_img.shape), r1_i.expand(b_chunk, *cache.r1_img.shape))
                dA_i = torch.sum(cA_i**2, dim=-1, keepdim=True) + rc_sq
                ctA_i = torch.sum(td_i * r1_i, dim=-1, keepdim=True) / r1_ir
                sA_i = torch.where(
                    dA_i > 0.0,
                    -(-1.0 * inv_4pi) * (1.0 + ctA_i) / torch.clamp(dA_i, min=1e-30),
                    torch.zeros_like(dA_i),
                )

                cB_i = torch.linalg.cross(td_i.expand(b_chunk, *cache.r2_img.shape), r2_i.expand(b_chunk, *cache.r2_img.shape))
                dB_i = torch.sum(cB_i**2, dim=-1, keepdim=True) + rc_sq
                ctB_i = torch.sum(td_i * r2_i, dim=-1, keepdim=True) / r2_ir
                sB_i = torch.where(
                    dB_i > 0.0,
                    (-1.0 * inv_4pi) * (1.0 + ctB_i) / torch.clamp(dB_i, min=1e-30),
                    torch.zeros_like(dB_i),
                )
                V_trail = V_trail + cA_i * sA_i + cB_i * sB_i

            AIC_trail = torch.einsum('bijk,ik->bij', V_trail, cache.normals)
            AIC = cache.AIC_bound.unsqueeze(0) + AIC_trail
            rhs = -torch.einsum('ik,bk->bi', cache.normals, vinf_c)

            Gamma_chunk = torch.linalg.solve(AIC, rhs)
            w_ind_z_chunk = torch.einsum('bij,bj->bi', V_trail[..., 2], Gamma_chunk)

            Gamma_chunks.append(Gamma_chunk.cpu())
            w_ind_z_chunks.append(w_ind_z_chunk.cpu())

        Gamma_all = torch.cat(Gamma_chunks, dim=0).numpy()
        w_ind_z_all = torch.cat(w_ind_z_chunks, dim=0).numpy()

        # Vectorized post-processing
        q_inf = 0.5 * condition.rho * (condition.V_inf ** 2)
        inv_vinf = 1.0 / condition.V_inf
        all_n_panels = [len(ds.y_panels) for ds in disc_surfaces]

        surface_spanwise = []
        total_lift = np.zeros(B, dtype=float)
        total_Di = np.zeros(B, dtype=float)
        total_Dp = np.zeros(B, dtype=float)
        has_profile = False
        total_Mx = np.zeros(B, dtype=float)
        total_My = np.zeros(B, dtype=float)
        total_Mz = np.zeros(B, dtype=float)

        offset = 0
        for ds, n_p in zip(disc_surfaces, all_n_panels):
            g_surf = Gamma_all[:, offset:offset + n_p]
            w_surf = w_ind_z_all[:, offset:offset + n_p]
            a_i = -w_surf * inv_vinf
            a_eff = alphas[:, None] + ds.twists[None, :] - a_i
            cl = 2.0 * g_surf * inv_vinf / ds.chords[None, :]
            cdi = cl * a_i
            local_l = condition.rho * condition.V_inf * g_surf

            L_vec = local_l * ds.dy_panels[None, :]
            total_lift += np.sum(L_vec, axis=-1)

            Di_vec = q_inf * ds.chords[None, :] * cdi * ds.dy_panels[None, :]
            total_Di += np.sum(Di_vec, axis=-1)

            # Profile drag if polar tables present
            cd_prof = None
            has_prof = getattr(ds, "has_profile_drag", None)
            if has_prof is None:
                has_prof = any(
                    getattr(af, "Cd0", None) is None or getattr(af, "Cd0", 0.0) > 0.0
                    for af in ds.airfoils
                )
            if has_prof:
                has_profile = True
                cd_prof = np.zeros((B, n_p), dtype=float)
                groups = getattr(ds, "airfoil_groups", None)
                if groups is not None:
                    for uaf, p_indices in groups:
                        cd_prof[:, p_indices] = uaf.Cd(a_eff[:, p_indices])
                else:
                    unique_airfoils = []
                    for af in ds.airfoils:
                        if not any(af is u for u in unique_airfoils):
                            unique_airfoils.append(af)
                    for uaf in unique_airfoils:
                        p_indices = np.where([af is uaf for af in ds.airfoils])[0]
                        cd_prof[:, p_indices] = uaf.Cd(a_eff[:, p_indices])
                Dp_vec = q_inf * ds.chords[None, :] * cd_prof * ds.dy_panels[None, :]
                total_Dp += np.sum(Dp_vec, axis=-1)

            if hasattr(ds, "panel_centers_qc") and len(ds.panel_centers_qc) == len(ds.dy_panels):
                pt = ds.panel_centers_qc
            else:
                pt = 0.5 * (ds.nodes_qc[:-1] + ds.nodes_qc[1:])
            x = pt[:, 0][None, :]
            y = pt[:, 1][None, :]
            z = pt[:, 2][None, :]

            Fx = -Di_vec * ca[:, None] + L_vec * sa[:, None]
            Fz = L_vec * ca[:, None] + Di_vec * sa[:, None]

            total_Mx += np.sum(y * Fz, axis=-1)
            total_My += np.sum(z * Fx - x * Fz, axis=-1)
            total_Mz += np.sum(-y * Fx, axis=-1)

            surface_spanwise.append({
                "ds": ds,
                "gamma": g_surf,
                "cl": cl,
                "cdi": cdi,
                "cd_prof": cd_prof,
                "a_eff": a_eff,
                "a_i": a_i,
                "local_lift": local_l,
            })
            offset += n_p

        denom_qS = q_inf * aircraft.S_ref
        CL_all = total_lift / denom_qS
        CDi_all = total_Di / denom_qS
        CDp_all = total_Dp / denom_qS if has_profile else None
        CD_total_all = CDi_all + (CDp_all if CDp_all is not None else 0.0)

        c_ref = aircraft.c_ref if aircraft.c_ref is not None else (aircraft.S_ref / aircraft.b_ref if aircraft.b_ref > 0 else 1.0)
        Cl_mom_all = total_Mx / (denom_qS * aircraft.b_ref)
        Cm_mom_all = total_My / (denom_qS * c_ref)
        Cn_mom_all = total_Mz / (denom_qS * aircraft.b_ref)

        AR = aircraft.b_ref ** 2 / aircraft.S_ref
        e_all = np.where(
            CDi_all > 1e-14,
            np.minimum(CL_all ** 2 / (np.pi * AR * np.maximum(CDi_all, 1e-14)), 1.5),
            1.0,
        )

        results: list[SolverResult] = []
        for b_idx in range(B):
            tot = IntegratedResult(
                CL=float(CL_all[b_idx]),
                CDi=float(CDi_all[b_idx]),
                CDp=float(CDp_all[b_idx]) if CDp_all is not None else None,
                CD_total=float(CD_total_all[b_idx]) if has_profile else None,
                e=float(e_all[b_idx]),
                AR=float(AR),
                Cl=float(Cl_mom_all[b_idx]),
                Cm=float(Cm_mom_all[b_idx]),
                Cn=float(Cn_mom_all[b_idx]),
            )
            sp_list = [
                SpanwiseResult(
                    y=s_info["ds"].y_panels,
                    gamma=s_info["gamma"][b_idx],
                    Cl=s_info["cl"][b_idx],
                    Cd_i=s_info["cdi"][b_idx],
                    Cd_profile=s_info["cd_prof"][b_idx] if s_info["cd_prof"] is not None else None,
                    alpha_eff=s_info["a_eff"][b_idx],
                    alpha_i=s_info["a_i"][b_idx],
                    local_lift=s_info["local_lift"][b_idx],
                    surface_name=s_info["ds"].surface_name,
                )
                for s_info in surface_spanwise
            ]
            results.append(SolverResult(
                spanwise=sp_list,
                totals=tot,
                solver_type="horseshoe_gpu_batched",
                converged=True,
                iterations=1,
            ))

        return results
