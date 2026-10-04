# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Classical Fourier-series Lifting Line Theory solver.

This solver expands the circulation distribution as

    Γ(θ) = 2 b V∞  Σ  Aₙ sin(nθ)

and solves the resulting N×N linear system for the Fourier coefficients Aₙ.
From these coefficients, total CL, CDi, and the span efficiency factor are
computed analytically.

**Limitations:** single, symmetric, unswept, planar wing with linear section
data only.  For swept / dihedral / multi-surface cases, use the horseshoe
solver.
"""

from __future__ import annotations

import numpy as np

from ventorum.legacy.utils.linalg import fast_linear_solve
from ventorum.legacy.core.datatypes import (
    Aircraft,
    FlightCondition,
    IntegratedResult,
    SolverResult,
    SolverSettings,
    SpanwiseResult,
)
from ventorum.legacy.geometry.processing import build_fourier_stations
from ventorum.legacy.core.trust import evaluate_aerodynamic_trust
from ventorum.legacy.solvers.base import BaseSolver
from ventorum.legacy.utils.validation import validate_fourier_applicability


class FourierSolver(BaseSolver):
    """Classical Fourier-series LLT solver."""

    def solve(
        self,
        aircraft: Aircraft,
        condition: FlightCondition,
        settings: SolverSettings,
    ) -> SolverResult:
        # --- pre-checks -------------------------------------------------------
        validate_fourier_applicability(aircraft)
        aircraft.compute_reference_values()
        surf = aircraft.surfaces[0]

        N = settings.n_panels  # number of Fourier terms / collocation points
        b = surf.semi_span * 2.0  # full span
        V_inf = condition.V_inf
        alpha = condition.alpha

        # --- build collocation stations ----------------------------------------
        stations = build_fourier_stations(surf, N)
        theta = stations.theta          # [N]
        chords = stations.chords        # [N]
        a0 = stations.a0                # [N]
        alpha_L0 = stations.alpha_L0    # [N]
        twists = stations.twists        # [N]

        # μ_k = c_k · a0_k / (4b)
        mu = chords * a0 / (4.0 * b)

        # --- assemble the linear system (vectorized) ----------------------------
        n_indices = np.arange(1, N + 1)
        sin_theta = np.sin(theta)

        n_theta = theta[:, np.newaxis] * n_indices[np.newaxis, :]
        lhs = np.sin(n_theta) * (
            1.0 + (mu / sin_theta)[:, np.newaxis] * n_indices[np.newaxis, :]
        )
        rhs = mu * (alpha + twists - alpha_L0)

        # --- solve for Fourier coefficients ------------------------------------
        A = fast_linear_solve(lhs, rhs)

        # --- aerodynamic results from Fourier coefficients ---------------------
        AR = b ** 2 / aircraft.S_ref

        CL = np.pi * AR * A[0]

        # CDi = π AR Σ n Aₙ²
        CDi = np.pi * AR * np.sum(n_indices * A ** 2)

        # Span efficiency factor: guard against zero-lift condition (A[0] == 0)
        if abs(A[0]) > 1e-12:
            delta = np.sum(n_indices[1:] * A[1:] ** 2) / (A[0] ** 2)
            e = 1.0 / (1.0 + delta) if delta >= 0.0 else 1.0
        else:
            e = 1.0
        e = min(max(e, 0.0), 1.5)

        # --- spanwise distributions (vectorized reconstruction) ----------------
        N_out = 2 * N + 1  # finer output grid
        theta_out = np.linspace(0.01, np.pi - 0.01, N_out)
        y_out = surf.semi_span * np.cos(theta_out)  # from tip (+) to tip (−)
        sin_theta_out = np.sin(theta_out)

        sin_mat_out = np.sin(theta_out[:, np.newaxis] * n_indices[np.newaxis, :])
        gamma_out = (sin_mat_out @ A) * (2.0 * b * V_inf)
        alpha_i_out = (sin_mat_out @ (n_indices * A)) / np.maximum(sin_theta_out, 1e-14)

        alpha_eff_out = alpha + np.interp(
            np.abs(y_out), stations.y[::-1], twists[::-1]
        ) - alpha_i_out

        # Interpolate chord at output stations
        y_frac_out = np.abs(y_out) / surf.semi_span
        chords_out = np.interp(y_frac_out, stations.y / surf.semi_span, chords)

        with np.errstate(divide="ignore", invalid="ignore"):
            Cl_out = np.where(chords_out > 1e-12, 2.0 * gamma_out / (V_inf * np.maximum(chords_out, 1e-12)), 0.0)
            Cd_i_out = Cl_out * alpha_i_out
        local_lift_out = condition.rho * V_inf * gamma_out

        # Profile drag from section data
        Cd0_out = np.interp(y_frac_out, stations.y / surf.semi_span, stations.Cd0)
        has_profile = np.any(Cd0_out > 0)

        spanwise = SpanwiseResult(
            y=y_out,
            gamma=gamma_out,
            Cl=Cl_out,
            Cd_i=Cd_i_out,
            Cd_profile=Cd0_out if has_profile else None,
            alpha_eff=alpha_eff_out,
            alpha_i=alpha_i_out,
            local_lift=local_lift_out,
            surface_name=surf.name,
        )

        # Profile drag integration
        CDp = None
        if has_profile:
            # Simple integration using midpoint rule
            dy = np.abs(np.gradient(y_out))
            s_ref = aircraft.S_ref if aircraft.S_ref and aircraft.S_ref > 0 else 1.0
            CDp = np.sum(Cd0_out * chords_out * dy) / s_ref

        trust_eval = evaluate_aerodynamic_trust(
            AR=AR,
            surfaces=[surf],
            condition=condition,
            spanwise_list=[spanwise],
            CL=CL,
            CDi=CDi,
            CD_total=CDi + (CDp if CDp is not None else 0.0),
            converged=True,
            n_panels=N,
        )

        totals = IntegratedResult(
            CL=CL,
            CDi=CDi,
            CDp=CDp,
            CD_total=CDi + (CDp if CDp is not None else 0.0),
            e=e,
            CL_alpha=np.pi * AR * mu.mean() / (1.0 + mu.mean()),  # approx
            AR=AR,
            trust=trust_eval,
        )

        return SolverResult(
            spanwise=[spanwise],
            totals=totals,
            solver_type="fourier",
            converged=True,
            iterations=1,
            fourier_coefficients=A,
        )

    def solve_sweep(
        self,
        aircraft: Aircraft,
        condition: FlightCondition,
        settings: SolverSettings,
        alpha_range: np.ndarray,
    ) -> list[SolverResult]:
        """Solve an angle-of-attack sweep in a single vectorized multi-RHS operation.

        Exploits the mathematical invariance of the Fourier collocation LHS matrix
        to angle of attack, solving L · A = B for all alphas simultaneously with
        zero precision loss.
        """
        validate_fourier_applicability(aircraft)
        aircraft.compute_reference_values()
        surf = aircraft.surfaces[0]

        N = settings.n_panels
        b = surf.semi_span * 2.0
        V_inf = condition.V_inf

        stations = build_fourier_stations(surf, N)
        theta = stations.theta
        chords = stations.chords
        a0 = stations.a0
        alpha_L0 = stations.alpha_L0
        twists = stations.twists

        mu = chords * a0 / (4.0 * b)
        n_indices = np.arange(1, N + 1)
        sin_theta = np.sin(theta)

        n_theta = theta[:, np.newaxis] * n_indices[np.newaxis, :]
        lhs = np.sin(n_theta) * (
            1.0 + (mu / sin_theta)[:, np.newaxis] * n_indices[np.newaxis, :]
        )

        alphas_arr = np.asarray(alpha_range, dtype=float)
        # Vectorized RHS matrix of shape (N, M_alpha)
        rhs_mat = mu[:, np.newaxis] * (
            alphas_arr[np.newaxis, :] + twists[:, np.newaxis] - alpha_L0[:, np.newaxis]
        )
        A_mat = fast_linear_solve(lhs, rhs_mat)

        AR = b ** 2 / aircraft.S_ref
        CL_arr = np.pi * AR * A_mat[0, :]
        CDi_arr = np.pi * AR * np.sum(n_indices[:, np.newaxis] * A_mat ** 2, axis=0)

        # Efficiency factors
        e_arr = np.empty(len(alphas_arr), dtype=float)
        for i in range(len(alphas_arr)):
            A0 = A_mat[0, i]
            if abs(A0) > 1e-12:
                delta = np.sum(n_indices[1:] * A_mat[1:, i] ** 2) / (A0 ** 2)
                e_val = 1.0 / (1.0 + delta) if delta >= 0.0 else 1.0
            else:
                e_val = 1.0
            e_arr[i] = min(max(e_val, 0.0), 1.5)

        # Output grid reconstruction (invariant components precomputed once)
        N_out = 2 * N + 1
        theta_out = np.linspace(0.01, np.pi - 0.01, N_out)
        y_out = surf.semi_span * np.cos(theta_out)
        sin_theta_out = np.sin(theta_out)
        sin_mat_out = np.sin(theta_out[:, np.newaxis] * n_indices[np.newaxis, :])
        inv_sin_theta_out = 1.0 / np.maximum(sin_theta_out, 1e-14)

        twists_interp = np.interp(np.abs(y_out), stations.y[::-1], twists[::-1])
        y_frac_out = np.abs(y_out) / surf.semi_span
        chords_out = np.interp(y_frac_out, stations.y / surf.semi_span, chords)
        Cd0_out = np.interp(y_frac_out, stations.y / surf.semi_span, stations.Cd0)
        has_profile = np.any(Cd0_out > 0)

        CDp = None
        if has_profile:
            dy = np.abs(np.gradient(y_out))
            s_ref = aircraft.S_ref if aircraft.S_ref and aircraft.S_ref > 0 else 1.0
            CDp = np.sum(Cd0_out * chords_out * dy) / s_ref

        CL_alpha_approx = np.pi * AR * mu.mean() / (1.0 + mu.mean())
        chord_safe = np.maximum(chords_out, 1e-12)
        chord_valid = chords_out > 1e-12

        results: list[SolverResult] = []
        for i, a in enumerate(alphas_arr):
            A = A_mat[:, i]
            gamma_out = (sin_mat_out @ A) * (2.0 * b * V_inf)
            alpha_i_out = (sin_mat_out @ (n_indices * A)) * inv_sin_theta_out
            alpha_eff_out = a + twists_interp - alpha_i_out

            with np.errstate(divide="ignore", invalid="ignore"):
                Cl_out = np.where(chord_valid, 2.0 * gamma_out / (V_inf * chord_safe), 0.0)
                Cd_i_out = Cl_out * alpha_i_out
            local_lift_out = condition.rho * V_inf * gamma_out

            spanwise = SpanwiseResult(
                y=y_out,
                gamma=gamma_out,
                Cl=Cl_out,
                Cd_i=Cd_i_out,
                Cd_profile=Cd0_out if has_profile else None,
                alpha_eff=alpha_eff_out,
                alpha_i=alpha_i_out,
                local_lift=local_lift_out,
                surface_name=surf.name,
            )
            cond_i = FlightCondition(V_inf=V_inf, alpha=float(a), rho=condition.rho)
            trust_i = evaluate_aerodynamic_trust(
                AR=AR,
                surfaces=[surf],
                condition=cond_i,
                spanwise_list=[spanwise],
                CL=float(CL_arr[i]),
                CDi=float(CDi_arr[i]),
                CD_total=float(CDi_arr[i] + (CDp if CDp is not None else 0.0)),
                converged=True,
                n_panels=N,
            )
            totals = IntegratedResult(
                CL=CL_arr[i],
                CDi=CDi_arr[i],
                CDp=CDp,
                CD_total=CDi_arr[i] + (CDp if CDp is not None else 0.0),
                e=e_arr[i],
                CL_alpha=CL_alpha_approx,
                AR=AR,
                trust=trust_i,
            )
            results.append(SolverResult(
                spanwise=[spanwise],
                totals=totals,
                solver_type="fourier",
                converged=True,
                iterations=1,
                fourier_coefficients=A,
                condition=cond_i,
            ))

        return results
