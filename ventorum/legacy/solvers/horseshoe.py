# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
General numerical Lifting Line Theory solver using horseshoe vortices
and the Biot-Savart law.

This solver handles arbitrary wing geometry: sweep, dihedral, twist, taper,
and multiple interacting lifting surfaces.  It assembles an Aerodynamic
Influence Coefficient (AIC) matrix and solves:

    AIC · Γ  =  b

where *Γ* is the circulation at each panel and *b* encodes the free-stream
boundary condition.
"""

from __future__ import annotations

import numpy as np

from ventorum.legacy.utils.linalg import fast_linear_solve
from ventorum.legacy.aero.forces import compute_spanwise_horseshoe, integrate_results
from ventorum.legacy.aero.influence import (
    build_aic_and_rhs,
    _trailing_direction,
    _freestream_vector,
    HorseshoeGeometryCache,
    precompute_horseshoe_geometry,
    compute_trailing_velocity_from_cache,
)
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


class HorseshoeSolver(BaseSolver):
    """Horseshoe-vortex / Biot-Savart LLT solver (linear section data)."""

    def solve_discretized(
        self,
        disc_surfaces: list[DiscretizedSurface],
        condition: FlightCondition,
        S_ref: float,
        b_ref: float,
        c_ref: float | None = None,
        use_symmetry: bool | None = None,
    ) -> SolverResult:
        """Solve a pre-discretized aircraft system without re-meshing overhead."""
        from ventorum.legacy.core.symmetry import (
            can_use_symmetry,
            extract_half_mesh_surface,
            mirror_discretized_surface,
            reconstruct_full_circulation,
            reconstruct_full_downwash,
        )

        if use_symmetry is None:
            use_symmetry = can_use_symmetry(disc_surfaces, condition)

        if use_symmetry:
            # Ensure surfaces are half-mesh
            disc_surfaces_half = [
                ds if ds.is_half_mesh else extract_half_mesh_surface(ds)
                for ds in disc_surfaces
            ]

            AIC, rhs, V_tot, V_trail = build_aic_and_rhs(
                disc_surfaces_half, condition, return_details=True, use_symmetry=True
            )
            Gamma_semi = fast_linear_solve(AIC, rhs)
            w_ind_z_semi = V_trail[:, :, 2] @ Gamma_semi

            # Reconstruct full-aircraft representations
            disc_surfaces_full = [mirror_discretized_surface(ds) for ds in disc_surfaces_half]
            Gamma_full = reconstruct_full_circulation(Gamma_semi, disc_surfaces_half)
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
            # Symmetric aerodynamics guarantees zero lateral moment coefficients
            totals.Cl = 0.0
            totals.Cn = 0.0

            return SolverResult(
                spanwise=spanwise_list,
                totals=totals,
                solver_type="horseshoe",
                converged=True,
                iterations=1,
                symmetry_used=True,
            )

        # Standard full solve without symmetry plane
        AIC, rhs, V_tot, V_trail = build_aic_and_rhs(
            disc_surfaces, condition, return_details=True, use_symmetry=False
        )
        Gamma = fast_linear_solve(AIC, rhs)

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
            solver_type="horseshoe",
            converged=True,
            iterations=1,
            symmetry_used=False,
        )

    def solve_precomputed(
        self,
        geo_cache: HorseshoeGeometryCache,
        condition: FlightCondition,
        S_ref: float,
        b_ref: float,
        c_ref: float | None = None,
        rc: float = VORTEX_CORE_RADIUS,
    ) -> SolverResult:
        """Solve using precomputed bound-vortex geometry (blazing fast for alpha sweeps)."""
        trailing_dir = _trailing_direction(condition)
        V_inf_vec = _freestream_vector(condition)

        V_trail = compute_trailing_velocity_from_cache(geo_cache, trailing_dir, rc=rc)

        if getattr(geo_cache, "AIC_bound", None) is not None:
            AIC = geo_cache.AIC_bound + np.einsum('ijk,ik->ij', V_trail, geo_cache.normals)
        else:
            V_tot = geo_cache.V_bound + V_trail
            if geo_cache.h is not None and geo_cache.V_bound_img is not None:
                V_tot += geo_cache.V_bound_img
            AIC = np.einsum('ijk,ik->ij', V_tot, geo_cache.normals)

        rhs = -np.dot(geo_cache.normals, V_inf_vec)
        Gamma = fast_linear_solve(AIC, rhs)

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
                solver_type="horseshoe",
                converged=True,
                iterations=1,
                symmetry_used=True,
            )

        spanwise_list = compute_spanwise_horseshoe(
            geo_cache.disc_surfaces, Gamma, condition, V_trail=V_trail
        )
        totals = integrate_results(
            spanwise_list,
            geo_cache.disc_surfaces,
            condition,
            S_ref=S_ref,
            b_ref=b_ref,
            c_ref=c_ref,
        )

        return SolverResult(
            spanwise=spanwise_list,
            totals=totals,
            solver_type="horseshoe",
            converged=True,
            iterations=1,
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
            disc_surfaces, condition, aircraft.S_ref, aircraft.b_ref, aircraft.c_ref, use_symmetry=use_sym
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
                geo_cache, cond, aircraft.S_ref, aircraft.b_ref, aircraft.c_ref
            )
            results.append(res)
        return results

