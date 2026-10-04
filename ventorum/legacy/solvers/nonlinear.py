# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Nonlinear iterative solver wrapper.

This module wraps the horseshoe solver to handle **tabulated (nonlinear)**
section aerodynamic data.  Instead of a single matrix solve, it iterates:

1. Compute an initial Γ distribution from a linearised solve.
2. Compute induced velocities → effective angle of attack at each panel.
3. Look up Cl(α_eff) from the section polar tables.
4. Update Γ = ½ V∞ c Cl(α_eff).
5. Blend old and new Γ with a relaxation factor ω.
6. Repeat until ‖ΔΓ‖ < tolerance.
"""

from __future__ import annotations

import warnings

import numpy as np

from ventorum.legacy.utils.linalg import fast_linear_solve
from ventorum.legacy.aero.biot_savart import horseshoe_velocity
from ventorum.legacy.aero.forces import compute_spanwise_horseshoe, integrate_results
from ventorum.legacy.aero.acceleration import (
    dispatch_nonlinear_relaxation_loop,
)
from ventorum.legacy.aero.influence import (
    build_aic_and_rhs,
    compute_horseshoe_velocity_matrix,
    _trailing_direction,
    _freestream_direction,
    _freestream_vector,
    compute_trailing_velocity_from_cache,
)
from ventorum.legacy.aero.polars import section_Cl, section_Cd
from ventorum.legacy.core.constants import VORTEX_CORE_RADIUS
from ventorum.legacy.core.datatypes import (
    Aircraft,
    DiscretizedSurface,
    FlightCondition,
    SolverResult,
    SolverSettings,
)
from ventorum.legacy.geometry.processing import discretize_surface, discretize_aircraft_surfaces
from ventorum.legacy.solvers.base import BaseSolver
from ventorum.legacy.utils.validation import validate_aircraft


class NonlinearSolver(BaseSolver):
    """Iterative LLT solver for nonlinear (tabulated) section polars."""

    def solve_discretized(
        self,
        disc_surfaces: list[DiscretizedSurface],
        condition: FlightCondition,
        settings: SolverSettings,
        S_ref: float,
        b_ref: float,
        c_ref: float | None = None,
        use_symmetry: bool | None = None,
    ) -> SolverResult:
        """Solve a pre-discretized aircraft system with nonlinear section polars."""
        from ventorum.legacy.core.symmetry import (
            can_use_symmetry,
            extract_half_mesh_surface,
            mirror_discretized_surface,
            reconstruct_full_circulation,
            reconstruct_full_downwash,
        )

        if use_symmetry is None:
            use_symmetry = can_use_symmetry(disc_surfaces, condition, settings)

        if use_symmetry:
            # Ensure half-mesh surfaces
            disc_surfaces_half = [
                ds if ds.is_half_mesh else extract_half_mesh_surface(ds)
                for ds in disc_surfaces
            ]
            surfs_to_solve = disc_surfaces_half
        else:
            surfs_to_solve = disc_surfaces

        # --- flatten geometry for fast iteration ------------------------------
        if len(surfs_to_solve) == 1:
            ds0 = surfs_to_solve[0]
            all_chords = ds0.chords
            all_twists = ds0.twists
            airfoil_groups = ds0.airfoil_groups
            is_single_airfoil = (len(airfoil_groups) == 1)
            if is_single_airfoil:
                single_af = airfoil_groups[0][0]
        else:
            all_chords = np.concatenate([ds.chords for ds in surfs_to_solve])
            all_twists = np.concatenate([ds.twists for ds in surfs_to_solve])
            all_airfoils = []
            for ds in surfs_to_solve:
                all_airfoils.extend(ds.airfoils)

            unique_airfoils = []
            for af in all_airfoils:
                if not any(af is u for u in unique_airfoils):
                    unique_airfoils.append(af)

            is_single_airfoil = (len(unique_airfoils) == 1)
            if is_single_airfoil:
                single_af = unique_airfoils[0]
            else:
                airfoil_groups = [
                    (uaf, np.where([af is uaf for af in all_airfoils])[0])
                    for uaf in unique_airfoils
                ]

        N = len(all_chords)
        V_inf = condition.V_inf

        # --- initial guess: linear solve & precomputed unit velocity tensor ---
        AIC, rhs, V_tot, V_trail = build_aic_and_rhs(
            surfs_to_solve, condition, return_details=True, use_symmetry=use_symmetry
        )
        Gamma = fast_linear_solve(AIC, rhs)
        V_z = V_tot[:, :, 2]  # shape (N, N)

        # --- iterative loop ---------------------------------------------------
        omega = settings.relaxation
        residual_history: list[float] = []
        converged = False

        # Precompute invariant terms before relaxation loop
        all_normals = np.vstack([ds.normals for ds in surfs_to_solve])
        v_dir = _freestream_direction(condition)
        alpha_geom = np.arcsin(np.clip(np.dot(all_normals, v_dir), -1.0, 1.0))
        half_vinf_chords = 0.5 * V_inf * all_chords
        inv_vinf = 1.0 / V_inf

        accel_loop_res = None
        if is_single_airfoil and hasattr(single_af, "alpha") and hasattr(single_af, "Cl_data"):
            accel_loop_res = dispatch_nonlinear_relaxation_loop(
                V_z, Gamma, inv_vinf, alpha_geom, half_vinf_chords,
                single_af.alpha, single_af.Cl_data, float(omega),
                int(settings.max_iterations), float(settings.tolerance)
            )

        if accel_loop_res is not None:
            Gamma, iteration, converged, residual_history = accel_loop_res
        else:
            for iteration in range(1, settings.max_iterations + 1):
                # Fast matrix-vector multiply for induced downwash
                alpha_i = -(V_z @ Gamma) * inv_vinf
                alpha_eff = alpha_geom - alpha_i

                # Vectorized look up of Cl from section polars
                if is_single_airfoil:
                    Cl_local = single_af.Cl(alpha_eff)
                else:
                    Cl_local = np.empty(N, dtype=float)
                    for uaf, idx in airfoil_groups:
                        Cl_local[idx] = uaf.Cl(alpha_eff[idx])

                # New circulation from Kutta-Joukowski & relaxation update
                dGamma = half_vinf_chords * Cl_local - Gamma
                residual = float(omega * np.max(np.abs(dGamma)))
                Gamma += omega * dGamma
                residual_history.append(residual)

                if residual < settings.tolerance:
                    converged = True
                    break

                # Adaptive relaxation: dynamically damp omega if residual is growing (near stall)
                if len(residual_history) >= 3 and residual_history[-1] > residual_history[-2]:
                    omega = max(0.02, omega * 0.75)

        if not converged:
            warnings.warn(
                f"NonlinearSolver did not converge after {settings.max_iterations} "
                f"iterations. Final residual: {residual_history[-1]:.2e}. "
                f"Consider reducing relaxation factor (currently {omega}).",
                stacklevel=2,
            )

        # --- post-process and reconstruct -------------------------------------
        if use_symmetry:
            w_ind_z_semi = V_trail[:, :, 2] @ Gamma
            disc_surfaces_full = [mirror_discretized_surface(ds) for ds in disc_surfaces_half]
            Gamma_full = reconstruct_full_circulation(Gamma, disc_surfaces_half)
            w_ind_z_full = reconstruct_full_downwash(w_ind_z_semi, disc_surfaces_half)

            spanwise_list = compute_spanwise_horseshoe(
                disc_surfaces_full, Gamma_full, condition, w_ind_z=w_ind_z_full
            )
            totals = integrate_results(
                spanwise_list,
                disc_surfaces_full,
                condition,
                S_ref=S_ref,
                b_ref=b_ref,
                c_ref=c_ref,
            )
            totals.Cl = 0.0
            totals.Cn = 0.0

            return SolverResult(
                spanwise=spanwise_list,
                totals=totals,
                solver_type="horseshoe_nonlinear",
                converged=converged,
                iterations=iteration,
                residual_history=residual_history,
                symmetry_used=True,
            )

        spanwise_list = compute_spanwise_horseshoe(
            disc_surfaces, Gamma, condition, V_trail=V_trail
        )
        totals = integrate_results(
            spanwise_list,
            disc_surfaces,
            condition,
            S_ref=S_ref,
            b_ref=b_ref,
            c_ref=c_ref,
        )

        return SolverResult(
            spanwise=spanwise_list,
            totals=totals,
            solver_type="horseshoe_nonlinear",
            converged=converged,
            iterations=iteration,
            residual_history=residual_history,
            symmetry_used=False,
        )

    def solve_precomputed(
        self,
        geo_cache: Any,
        condition: FlightCondition,
        settings: SolverSettings,
        S_ref: float,
        b_ref: float,
        c_ref: float | None = None,
        rc: float = VORTEX_CORE_RADIUS,
    ) -> SolverResult:
        """Solve using precomputed bound-vortex geometry cache."""
        disc_surfaces = geo_cache.disc_surfaces
        if len(disc_surfaces) == 1:
            ds0 = disc_surfaces[0]
            all_chords = ds0.chords
            all_twists = ds0.twists
            airfoil_groups = ds0.airfoil_groups
            is_single_airfoil = (len(airfoil_groups) == 1)
            if is_single_airfoil:
                single_af = airfoil_groups[0][0]
        else:
            all_chords = np.concatenate([ds.chords for ds in disc_surfaces])
            all_twists = np.concatenate([ds.twists for ds in disc_surfaces])
            all_airfoils = []
            for ds in disc_surfaces:
                all_airfoils.extend(ds.airfoils)

            unique_airfoils = []
            for af in all_airfoils:
                if not any(af is u for u in unique_airfoils):
                    unique_airfoils.append(af)

            is_single_airfoil = (len(unique_airfoils) == 1)
            if is_single_airfoil:
                single_af = unique_airfoils[0]
            else:
                airfoil_groups = [
                    (uaf, np.where([af is uaf for af in all_airfoils])[0])
                    for uaf in unique_airfoils
                ]

        N = len(all_chords)
        V_inf = condition.V_inf

        trailing_dir = _trailing_direction(condition)
        V_inf_vec = _freestream_vector(condition)

        V_trail = compute_trailing_velocity_from_cache(geo_cache, trailing_dir, rc=rc)
        V_tot = geo_cache.V_bound + V_trail
        if geo_cache.h is not None and geo_cache.V_bound_img is not None:
            V_tot = V_tot + geo_cache.V_bound_img

        if getattr(geo_cache, "AIC_bound", None) is not None:
            AIC = geo_cache.AIC_bound + np.einsum('ijk,ik->ij', V_trail, geo_cache.normals)
        else:
            AIC = np.einsum('ijk,ik->ij', V_tot, geo_cache.normals)

        rhs = -np.dot(geo_cache.normals, V_inf_vec)
        Gamma = fast_linear_solve(AIC, rhs)
        V_z = V_tot[:, :, 2]

        omega = settings.relaxation
        residual_history: list[float] = []
        converged = False

        v_dir = _freestream_direction(condition)
        alpha_geom = np.arcsin(np.clip(np.dot(geo_cache.normals, v_dir), -1.0, 1.0))
        half_vinf_chords = 0.5 * V_inf * all_chords
        inv_vinf = 1.0 / V_inf

        accel_loop_res = None
        if is_single_airfoil:
            accel_loop_res = dispatch_nonlinear_relaxation_loop(
                V_z, Gamma, inv_vinf, alpha_geom, half_vinf_chords,
                single_af.alpha, single_af.Cl_data, float(omega),
                int(settings.max_iterations), float(settings.tolerance)
            )

        if accel_loop_res is not None:
            Gamma, iteration, converged, residual_history = accel_loop_res
        else:
            for iteration in range(1, settings.max_iterations + 1):
                alpha_i = -(V_z @ Gamma) * inv_vinf
                alpha_eff = alpha_geom - alpha_i

                if is_single_airfoil:
                    Cl_local = single_af.Cl(alpha_eff)
                else:
                    Cl_local = np.empty(N, dtype=float)
                    for uaf, idx in airfoil_groups:
                        Cl_local[idx] = uaf.Cl(alpha_eff[idx])

                dGamma = half_vinf_chords * Cl_local - Gamma
                residual = float(omega * np.max(np.abs(dGamma)))
                Gamma += omega * dGamma
                residual_history.append(residual)

                if residual < settings.tolerance:
                    converged = True
                    break

                if len(residual_history) >= 3 and residual_history[-1] > residual_history[-2]:
                    omega = max(0.02, omega * 0.75)

        if not converged:
            warnings.warn(
                f"NonlinearSolver did not converge after {settings.max_iterations} "
                f"iterations. Final residual: {residual_history[-1]:.2e}. "
                f"Consider reducing relaxation factor (currently {omega}).",
                stacklevel=2,
            )

        if geo_cache.use_symmetry:
            from ventorum.legacy.core.symmetry import (
                mirror_discretized_surface,
                reconstruct_full_circulation,
                reconstruct_full_downwash,
            )
            w_ind_z_semi = V_trail[:, :, 2] @ Gamma
            disc_surfaces_half = geo_cache.disc_surfaces
            disc_surfaces_full = [mirror_discretized_surface(ds) for ds in disc_surfaces_half]
            Gamma_full = reconstruct_full_circulation(Gamma, disc_surfaces_half)
            w_ind_z_full = reconstruct_full_downwash(w_ind_z_semi, disc_surfaces_half)

            spanwise_list = compute_spanwise_horseshoe(
                disc_surfaces_full, Gamma_full, condition, w_ind_z=w_ind_z_full
            )
            totals = integrate_results(
                spanwise_list,
                disc_surfaces_full,
                condition,
                S_ref=S_ref,
                b_ref=b_ref,
                c_ref=c_ref,
            )
            totals.Cl = 0.0
            totals.Cn = 0.0

            return SolverResult(
                spanwise=spanwise_list,
                totals=totals,
                solver_type="horseshoe_nonlinear",
                converged=converged,
                iterations=iteration,
                residual_history=residual_history,
                symmetry_used=True,
            )

        spanwise_list = compute_spanwise_horseshoe(
            disc_surfaces, Gamma, condition, V_trail=V_trail
        )
        totals = integrate_results(
            spanwise_list,
            disc_surfaces,
            condition,
            S_ref=S_ref,
            b_ref=b_ref,
            c_ref=c_ref,
        )

        return SolverResult(
            spanwise=spanwise_list,
            totals=totals,
            solver_type="horseshoe_nonlinear",
            converged=converged,
            iterations=iteration,
            residual_history=residual_history,
            symmetry_used=False,
        )

    def solve(
        self,
        aircraft: Aircraft,
        condition: FlightCondition,
        settings: SolverSettings,
    ) -> SolverResult:
        # --- validate & prepare ------------------------------------------------
        validate_aircraft(aircraft)
        aircraft.compute_reference_values()

        from ventorum.legacy.core.symmetry import can_use_symmetry
        use_sym = can_use_symmetry(aircraft, condition, settings)

        # --- discretise surfaces (half-mesh if symmetry enabled) ---------------
        disc_surfaces = discretize_aircraft_surfaces(aircraft, settings, half_mesh=use_sym)

        return self.solve_discretized(
            disc_surfaces, condition, settings, aircraft.S_ref, aircraft.b_ref, aircraft.c_ref, use_symmetry=use_sym
        )

    def solve_sweep(
        self,
        aircraft: Aircraft,
        condition: FlightCondition,
        settings: SolverSettings,
        alpha_range: np.ndarray,
    ) -> list[SolverResult]:
        """Solve an angle-of-attack sweep reusing precomputed bound-vortex geometry.

        Eliminates redundant aircraft validation, reference computation, surface
        meshing, and invariant Biot-Savart tensor assembly across all angles of attack.
        """
        validate_aircraft(aircraft)
        aircraft.compute_reference_values()

        from ventorum.legacy.core.symmetry import can_use_symmetry
        use_sym = can_use_symmetry(aircraft, condition, settings)

        disc_surfaces = discretize_aircraft_surfaces(aircraft, settings, half_mesh=use_sym)
        from ventorum.legacy.aero.influence import precompute_horseshoe_geometry
        geo_cache = precompute_horseshoe_geometry(disc_surfaces, h=condition.h, use_symmetry=use_sym)
        results: list[SolverResult] = []
        for a in alpha_range:
            cond = FlightCondition(
                V_inf=condition.V_inf,
                alpha=float(a),
                beta=condition.beta,
                rho=condition.rho,
                h=condition.h,
                phi=getattr(condition, "phi", 0.0),
            )
            res = self.solve_precomputed(
                geo_cache, cond, settings, aircraft.S_ref, aircraft.b_ref, aircraft.c_ref
            )
            results.append(res)
        return results
