# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Modern Linear Lifting Line Theory solver based on Phillips & Snyder (2000)
and Phillips (2004) Section 1.9.

This solver handles general 3D aircraft configurations with arbitrary sweep,
dihedral, geometric twist, taper, and multi-surface aerodynamic interactions.
Unlike pure Weissinger / 3/4-chord vortex lattice methods, this formulation
directly couples the 3D Biot-Savart vortex lifting law (Kutta-Joukowski)
with 2D airfoil section lift curve slope a0(y) and zero-lift angle alpha_L0(y):

    [ diag(2 * |v_inf x dl_i| / (a0_i * dA_i)) - AIC_norm ] · Gamma = V_inf * (v_inf · n_i - alpha_L0_i)

This yields a direct, non-iterative linear system that captures the exact
classical Prandtl lifting-line solution for straight wings, while generalizing
to swept, dihedral, banked, ground-effect, and multi-wing aircraft.
"""

from __future__ import annotations

import numpy as np

from ventorum.legacy.utils.linalg import fast_linear_solve
from ventorum.legacy.aero.forces import compute_spanwise_horseshoe, integrate_results
from ventorum.legacy.aero.influence import (
    build_linear_llt_system,
    _trailing_direction,
    _freestream_direction,
    _freestream_vector,
    LinearGeometryCache,
    precompute_linear_geometry,
    compute_trailing_velocity_from_cache,
)
from ventorum.legacy.core.constants import VORTEX_CORE_RADIUS
from ventorum.legacy.core.datatypes import (
    Aircraft,
    DiscretizedSurface,
    FlightCondition,
    LiftingSurface,
    SolverResult,
    SolverSettings,
)
from ventorum.legacy.geometry.processing import discretize_aircraft_surfaces
from ventorum.legacy.solvers.base import BaseSolver
from ventorum.legacy.utils.validation import validate_aircraft


class LinearLLTSolver(BaseSolver):
    """Modern Numerical Linear Lifting Line Theory (LLT) solver.

    Formulated on the 3D vortex lifting law of Phillips & Snyder (2000),
    providing a non-iterative linear matrix solution for arbitrary 3D geometry
    with full support for section lift curve slope a0 and camber / zero-lift AoA alpha_L0.
    """

    def solve_discretized(
        self,
        disc_surfaces: list[DiscretizedSurface],
        condition: FlightCondition,
        S_ref: float,
        b_ref: float,
        c_ref: float | None = None,
        use_symmetry: bool | None = None,
        ref_point: np.ndarray | None = None,
    ) -> SolverResult:
        """Solve a pre-discretized aircraft system using Modern Linear LLT."""
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
            # Ensure surfaces are half-mesh (y >= 0)
            disc_surfaces_half = [
                ds if ds.is_half_mesh else extract_half_mesh_surface(ds)
                for ds in disc_surfaces
            ]

            A, rhs, V_tot, V_trail = build_linear_llt_system(
                disc_surfaces_half, condition, return_details=True, use_symmetry=True
            )
            Gamma_semi = fast_linear_solve(A, rhs)
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
                ref_point=ref_point,
            )
            totals.Cl = 0.0
            totals.Cn = 0.0

            return SolverResult(
                spanwise=spanwise_list,
                totals=totals,
                solver_type="linear",
                converged=True,
                iterations=1,
                symmetry_used=True,
            )

        # Standard full solve without symmetry plane
        A, rhs, V_tot, V_trail = build_linear_llt_system(
            disc_surfaces, condition, return_details=True, use_symmetry=False
        )
        Gamma = fast_linear_solve(A, rhs)

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
            ref_point=ref_point,
        )

        return SolverResult(
            spanwise=spanwise_list,
            totals=totals,
            solver_type="linear",
            converged=True,
            iterations=1,
            symmetry_used=False,
        )

    def solve_precomputed(
        self,
        geo_cache: LinearGeometryCache,
        condition: FlightCondition,
        S_ref: float,
        b_ref: float,
        c_ref: float | None = None,
        rc: float = VORTEX_CORE_RADIUS,
        ref_point: np.ndarray | None = None,
    ) -> SolverResult:
        """Solve using precomputed bound-vortex geometry cache for fast sweeps."""
        trailing_dir = _trailing_direction(condition)
        v_dir = _freestream_direction(condition)
        V_inf = condition.V_inf

        V_trail = compute_trailing_velocity_from_cache(geo_cache, trailing_dir, rc=rc)

        if getattr(geo_cache, "AIC_bound", None) is not None:
            AIC = geo_cache.AIC_bound + np.einsum('ijk,ik->ij', V_trail, geo_cache.normals)
        else:
            V_tot = geo_cache.V_bound + V_trail
            if geo_cache.h is not None and geo_cache.V_bound_img is not None:
                V_tot += geo_cache.V_bound_img
            AIC = np.einsum('ijk,ik->ij', V_tot, geo_cache.normals)

        v_cross_dl = np.cross(v_dir, geo_cache.dl)
        mag_v_cross_dl = np.linalg.norm(v_cross_dl, axis=1)
        diag_term = (2.0 * mag_v_cross_dl) / (geo_cache.a0 * geo_cache.dA)

        A = np.diag(diag_term) - AIC
        rhs = V_inf * (np.dot(geo_cache.normals, v_dir) - geo_cache.alpha_L0)
        Gamma = fast_linear_solve(A, rhs)

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
                ref_point=ref_point,
            )
            totals.Cl = 0.0
            totals.Cn = 0.0

            return SolverResult(
                spanwise=spanwise_list,
                totals=totals,
                solver_type="linear",
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
            ref_point=ref_point,
        )

        return SolverResult(
            spanwise=spanwise_list,
            totals=totals,
            solver_type="linear",
            converged=True,
            iterations=1,
            symmetry_used=False,
        )

    def solve(
        self,
        aircraft: Aircraft | LiftingSurface,
        condition: FlightCondition,
        settings: SolverSettings,
    ) -> SolverResult:
        """Run the Modern Linear LLT analysis on the specified aircraft."""
        if isinstance(aircraft, LiftingSurface):
            aircraft = Aircraft(surfaces=[aircraft])
        validate_aircraft(aircraft)
        aircraft.compute_reference_values()

        from ventorum.legacy.core.symmetry import can_use_symmetry
        use_sym = can_use_symmetry(aircraft, condition, settings)

        disc_surfaces = discretize_aircraft_surfaces(aircraft, settings, half_mesh=use_sym)

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
        aircraft: Aircraft | LiftingSurface,
        condition: FlightCondition,
        settings: SolverSettings,
        alpha_range: np.ndarray,
    ) -> list[SolverResult]:
        """Solve an angle-of-attack sweep using precomputed bound-vortex geometry."""
        if isinstance(aircraft, LiftingSurface):
            aircraft = Aircraft(surfaces=[aircraft])
        validate_aircraft(aircraft)
        aircraft.compute_reference_values()

        from ventorum.legacy.core.symmetry import can_use_symmetry
        use_sym = can_use_symmetry(aircraft, condition, settings)

        disc_surfaces = discretize_aircraft_surfaces(aircraft, settings, half_mesh=use_sym)
        geo_cache = precompute_linear_geometry(disc_surfaces, h=condition.h, use_symmetry=use_sym)
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


# Alias for convenience
LinearSolver = LinearLLTSolver
