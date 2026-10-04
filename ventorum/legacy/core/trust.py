# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Aerodynamic validity, trust scoring, and uncertainty quantification engine for Ventorum.

Quantifies the degree to which a flight condition and wing geometry satisfy
Prandtl's Lifting Line Theory assumptions. Reports a normalized Trust Score (0 to 1),
qualitative confidence rating, uncertainty bounds (error bars on CL, CDi, and L/D),
penalty breakdown, and actionable engineering recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Any, Sequence
import numpy as np

from ventorum.legacy.core.datatypes import (
    DiscretizedSurface,
    FlightCondition,
    LiftingSurface,
    SpanwiseResult,
    TrustScore,
)


def evaluate_aerodynamic_trust(
    AR: float,
    surfaces: Sequence[DiscretizedSurface | LiftingSurface] | None = None,
    condition: FlightCondition | None = None,
    spanwise_list: Sequence[SpanwiseResult] | None = None,
    CL: float = 0.0,
    CDi: float = 0.0,
    CD_total: float | None = None,
    converged: bool = True,
    n_panels: int = 80,
    max_sweep_rad: float | None = None,
    b_ref: float | None = None,
) -> TrustScore:
    """Evaluate the physical validity and numerical uncertainty of an Ventorum solution.

    Parameters
    ----------
    AR : float
        Wing aspect ratio (b^2 / S).
    surfaces : sequence of DiscretizedSurface or LiftingSurface, optional
        Lifting surface geometry models.
    condition : FlightCondition, optional
        Flight operating conditions.
    spanwise_list : sequence of SpanwiseResult, optional
        Solved spanwise distributions (for local Cl and stall proximity inspection).
    CL : float
        Integrated lift coefficient.
    CDi : float
        Integrated induced drag coefficient.
    CD_total : float or None
        Total drag coefficient (CDi + CDp).
    converged : bool
        Whether the solver achieved mathematical convergence.
    n_panels : int
        Number of panels per semi-span.
    max_sweep_rad : float or None
        Maximum leading-edge sweep angle [rad]. If None, inferred from surfaces.

    Returns
    -------
    TrustScore
    """
    penalties: dict[str, float] = {}
    warnings: list[str] = []
    recommendations: list[str] = []

    # 1. Aspect Ratio Penalty (Classical LLT requires AR >> 1, ideally AR >= 5)
    ar_pen = 0.0
    if AR <= 0:
        ar_pen = 1.0
        warnings.append("Aspect ratio is non-positive or undefined.")
    elif AR < 3.0:
        # Severe breakdown: slender-wing or delta-wing flow dominates
        ar_pen = 0.45 + 0.35 * min(1.0, (3.0 - AR) / 2.0)
        warnings.append(
            f"Very low aspect ratio (AR={AR:.2f} < 3.0). Classical LLT breaks down "
            f"due to dominant 3D spanwise flow and tip vortex interaction."
        )
        recommendations.append("Use 3D Vortex Lattice Method (VLM) or 3D Panel/CFD code for AR < 3.5.")
    elif AR < 5.0:
        # Mild to moderate 3D tip error
        ar_pen = 0.20 * ((5.0 - AR) / 2.0)
        warnings.append(
            f"Moderate aspect ratio (AR={AR:.2f} < 5.0). 3D tip roll-up introduces "
            f"mild discrepancies in lift curve slope and induced drag."
        )
        recommendations.append("Consider VLM verification for aspect ratios below 5.0.")
    penalties["aspect_ratio"] = ar_pen

    # 2. Leading Edge Sweep Angle Penalty
    sweep_pen = 0.0
    max_sweep = 0.0
    if max_sweep_rad is not None:
        max_sweep = float(abs(max_sweep_rad))
    elif surfaces:
        for s in surfaces:
            if hasattr(s, "sweep_le"):
                max_sweep = max(max_sweep, abs(float(getattr(s, "sweep_le", 0.0))))
            elif hasattr(s, "nodes_qc") and len(s.nodes_qc) > 1:
                # In symmetric discretizations, nodes run left-tip -> root -> right-tip
                mid_idx = len(s.nodes_qc) // 2
                dx = abs(s.nodes_qc[-1, 0] - s.nodes_qc[mid_idx, 0])
                dy = abs(s.nodes_qc[-1, 1] - s.nodes_qc[mid_idx, 1])
                if dy > 1e-6:
                    max_sweep = max(max_sweep, abs(float(np.arctan2(dx, dy))))

    sweep_deg = float(np.degrees(max_sweep))
    if sweep_deg > 30.0:
        sweep_pen = 0.45 + 0.35 * min(1.0, (sweep_deg - 30.0) / 15.0)
        warnings.append(
            f"High sweep angle (Lambda={sweep_deg:.1f} deg > 30 deg). Strong spanwise boundary "
            f"layer drift and crossflow invalidate straight bound-vortex assumptions."
        )
        recommendations.append("Use VLM or Euler/Navier-Stokes CFD for wings with sweep > 25 deg.")
    elif sweep_deg > 15.0:
        sweep_pen = 0.25 * ((sweep_deg - 15.0) / 15.0)
        warnings.append(
            f"Moderate sweep angle (Lambda={sweep_deg:.1f} deg > 15 deg). Induced downwash and "
            f"stall inception have increased uncertainty in classical LLT."
        )
        recommendations.append("VLM is recommended to capture 3D sweep-induced skew.")
    penalties["sweep"] = sweep_pen

    # 3. Stall & Flow Separation Proximity
    stall_pen = 0.0
    max_local_cl = 0.0
    max_alpha_eff_deg = 0.0

    if spanwise_list:
        for sw in spanwise_list:
            if len(sw.Cl) > 0:
                # Exclude outermost 2 boundary panels on each tip to filter singular tip Biot-Savart filament artifacts
                cl_interior = sw.Cl[2:-2] if len(sw.Cl) > 8 else (sw.Cl[1:-1] if len(sw.Cl) > 4 else sw.Cl)
                if len(cl_interior) > 0:
                    max_local_cl = max(max_local_cl, float(np.max(np.abs(cl_interior))))
            if len(sw.alpha_eff) > 0:
                aeff_interior = sw.alpha_eff[2:-2] if len(sw.alpha_eff) > 8 else (sw.alpha_eff[1:-1] if len(sw.alpha_eff) > 4 else sw.alpha_eff)
                if len(aeff_interior) > 0:
                    max_alpha_eff_deg = max(max_alpha_eff_deg, float(np.max(np.abs(np.degrees(aeff_interior)))))

    # If spanwise data was not provided, approximate using global CL
    if max_local_cl == 0.0 and abs(CL) > 0:
        max_local_cl = abs(CL) * 1.25  # typical peak Cl / CL ratio for elliptic/tapered wings

    if max_local_cl > 1.55 or max_alpha_eff_deg > 16.0:
        stall_pen = 0.50 + 0.35 * min(1.0, (max_local_cl - 1.55) / 0.5)
        warnings.append(
            f"Severe stall/separation risk: peak sectional Cl={max_local_cl:.2f} "
            f"(eff AoA={max_alpha_eff_deg:.1f} deg). Linear LLT overpredicts lift post-stall."
        )
        recommendations.append("Enable nonlinear tabulated polars with XFOIL data or use RANS CFD.")
    elif max_local_cl > 1.20 or max_alpha_eff_deg > 12.0:
        stall_pen = 0.25 * ((max_local_cl - 1.20) / 0.35)
        warnings.append(
            f"Approaching section stall: peak sectional Cl={max_local_cl:.2f} "
            f"(eff AoA={max_alpha_eff_deg:.1f} deg). Approaching typical Cl_max boundary."
        )
    penalties["stall_proximity"] = stall_pen

    # 4. Compressibility / Mach Number Penalty
    mach_pen = 0.0
    if condition is not None:
        v_inf = float(condition.V_inf)
        # Approximate speed of sound at standard sea-level / low altitude
        a_sound = 340.3
        mach = v_inf / a_sound
        if mach > 0.65:
            mach_pen = 0.60 + 0.30 * min(1.0, (mach - 0.65) / 0.25)
            warnings.append(
                f"Transonic regime (Mach={mach:.2f} > 0.65). Local shock wave formation "
                f"and compressibility invalidate incompressible LLT."
            )
            recommendations.append("Compressible Euler or Transonic RANS CFD required.")
        elif mach > 0.30:
            mach_pen = 0.30 * ((mach - 0.30) / 0.35)
            warnings.append(
                f"Subsonic compressibility effects present (Mach={mach:.2f} > 0.30). "
                f"Prandtl-Glauert compressibility correction may be needed."
            )
    penalties["compressibility"] = mach_pen

    # 5. Ground Effect Proximity Penalty
    ground_pen = 0.0
    if condition is not None and condition.h is not None:
        h = float(condition.h)
        b_span = b_ref
        if b_span is None and surfaces and len(surfaces) > 0:
            s0 = surfaces[0]
            if hasattr(s0, "semi_span"):
                b_span = float(s0.semi_span) * (2.0 if getattr(s0, "is_symmetric", True) else 1.0)
            elif hasattr(s0, "nodes_qc") and len(s0.nodes_qc) > 1:
                b_span = float(np.ptp(s0.nodes_qc[:, 1]))
        if b_span is None or b_span <= 0:
            b_span = AR if AR > 0 else 10.0

        h_over_b = h / b_span

        if h_over_b < 0.04:
            ground_pen = 0.40 + 0.35 * min(1.0, (0.04 - h_over_b) / 0.04)
            warnings.append(
                f"Extreme ground proximity (h/b={h_over_b:.3f} < 0.04). Near-field ground boundary "
                f"layer and ram-pressure cushions depart from planar image vortex model."
            )
            recommendations.append("Consider surface panel methods with ground plane discretization.")
        elif h_over_b < 0.10:
            ground_pen = 0.15 * ((0.10 - h_over_b) / 0.06)
            warnings.append(
                f"Strong ground effect (h/b={h_over_b:.3f} < 0.10). Wingtip vortex deformation is significant."
            )
    penalties["ground_proximity"] = ground_pen

    # 6. Numerical Discretization and Convergence Penalty
    num_pen = 0.0
    if not converged:
        num_pen += 0.55
        warnings.append("Solver failed to converge to specified tolerance residual.")
        recommendations.append("Increase relaxation factor or max_iterations in SolverSettings.")

    if n_panels < 16:
        num_pen += 0.15 * ((16 - n_panels) / 12.0)
        warnings.append(
            f"Coarse spanwise mesh (n_panels={n_panels} < 16). Wingtip downwash singularity "
            f"may be under-resolved."
        )
    penalties["numerical"] = num_pen

    # Composite Score Calculation
    total_penalty = min(1.0, sum(penalties.values()))
    score = max(0.0, 1.0 - total_penalty)

    # Assign qualitative rating tier
    if score >= 0.85:
        rating = "HIGH"
    elif score >= 0.65:
        rating = "MODERATE"
    elif score >= 0.40:
        rating = "LOW"
    else:
        rating = "UNRELIABLE"

    # Physics-grounded Uncertainty Bounds
    # Base error: ~2% on CL and ~4% on CDi under ideal LLT conditions
    fidelity_factor = 1.0 - score
    unc_cl = max(0.015, abs(CL) * (0.025 + 0.20 * fidelity_factor) + 0.01 * fidelity_factor)
    unc_cdi = max(0.0004, CDi * (0.04 + 0.35 * fidelity_factor) + 0.0003 * fidelity_factor)

    cd_tot = CD_total if CD_total is not None and CD_total > 0 else (CDi if CDi > 1e-6 else 0.02)
    ld_ratio = CL / cd_tot if cd_tot > 0 else 0.0
    unc_ld = max(0.2, abs(ld_ratio) * (0.05 + 0.30 * fidelity_factor))

    return TrustScore(
        score=score,
        rating=rating,
        uncertainty_CL=unc_cl,
        uncertainty_CDi=unc_cdi,
        uncertainty_LD=unc_ld,
        factors=penalties,
        warnings=warnings,
        recommendations=recommendations,
    )
