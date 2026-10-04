# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Run mesh convergence studies and find the best discretisation.

The module does these tasks:

1. Run a mesh convergence analysis over candidate panel counts and spacing
   schemes.
2. Compare the results with a high-resolution reference solution or with
   the Richardson limit.
3. Calculate the Grid Convergence Index (GCI, Roache 1998).
4. Find the **minimal mesh** that meets the error tolerance.
5. Recommend a **general mesh** with a margin, for similar geometries,
   sweep angles and flight conditions.
6. Give dimensionless scaling laws (panels per unit aspect ratio, span
   density, spacing rules).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Literal
from collections.abc import Sequence

import numpy as np

from ventorum.core.datatypes import (
    Aircraft,
    FlightCondition,
    LiftingSurface,
    SolverResult,
    SolverSettings,
)
from ventorum.geometry.processing import discretize_aircraft_surfaces


# ═══════════════════════════════════════════════════════════════════════════════
# Data structures for Mesh Convergence Study
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class MeshConvergencePoint:
    """A single evaluation point in a mesh convergence study.

    Parameters
    ----------
    n_panels : int
        Nominal panels per semi-span on the primary reference surface.
    spacing : str
        Spacing scheme applied ('auto', 'half-cosine', 'cosine', 'uniform', etc.).
    total_panels : int
        Total aerodynamic panels across all surfaces of the aircraft.
    panels_solved : int
        Total number of panels of the lattice that was solved, all surfaces.
        This count holds both halves and all chordwise panels, so it is
        larger than ``n_panels`` when the mesh has several surfaces, a
        mirrored half, or more than one chordwise panel.
    CL : float
        Lift coefficient at the nominal/cruise condition.
    CDi : float
        Induced drag coefficient.
    CD_total : float
        Total drag coefficient (including profile drag if polar data available).
    Cm : float
        Pitching moment coefficient.
    error_cl_pct : float
        Relative percentage error in CL with respect to the reference solution.
    error_cdi_pct : float
        Relative percentage error in CDi with respect to the reference solution.
    error_gamma_l2_pct : float
        Relative L2-norm percentage error of the spanwise circulation distribution Γ(y).
    max_sweep_error_pct : float
        Maximum relative error across all evaluated angles of attack in a sweep.
    mean_sweep_error_pct : float
        Mean relative error across all evaluated angles of attack in a sweep.
    solve_time_ms : float
        Median solve execution time in milliseconds.
    cond_num : float
        Condition number of the aerodynamic influence coefficient (AIC) matrix.
    min_panel_ar : float
        Minimum panel aspect ratio (Δy / chord) across all panels.
    is_accurate : bool
        Whether this mesh meets the user-defined accuracy tolerance.
    is_safe : bool
        Whether the mesh avoids numerical ill-conditioning and aspect ratio collapse.
    efficiency_score : float
        Composite score balancing accuracy, panel count, and execution speed.
    gci_cl : float | None
        Grid Convergence Index (%) for CL if part of a monotonic refinement triplet.
    gci_cdi : float | None
        Grid Convergence Index (%) for CDi if part of a monotonic refinement triplet.
    result : SolverResult | None
        Full solver result for this mesh configuration.
    """

    n_panels: int
    spacing: str
    total_panels: int
    panels_solved: int
    CL: float
    CDi: float
    CD_total: float | None = None
    Cm: float = 0.0
    error_cl_pct: float = 0.0
    error_cdi_pct: float = 0.0
    error_gamma_l2_pct: float = 0.0
    max_sweep_error_pct: float = 0.0
    mean_sweep_error_pct: float = 0.0
    solve_time_ms: float = 0.0
    cond_num: float = 1.0
    min_panel_ar: float = 0.1
    is_accurate: bool = True
    is_safe: bool = True
    efficiency_score: float = 1.0
    gci_cl: float | None = None
    gci_cdi: float | None = None
    result: SolverResult | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert convergence point to serializable dictionary."""
        cd_tot = float(self.CD_total) if self.CD_total is not None else float(self.CDi)
        return {
            "n_panels": self.n_panels,
            "spacing": self.spacing,
            "total_panels": self.total_panels,
            "panels_solved": self.panels_solved,
            "CL": float(self.CL),
            "CDi": float(self.CDi),
            "CD_total": cd_tot,
            "Cm": float(self.Cm),
            "error_cl_pct": float(self.error_cl_pct),
            "error_cdi_pct": float(self.error_cdi_pct),
            "error_gamma_l2_pct": float(self.error_gamma_l2_pct),
            "max_sweep_error_pct": float(self.max_sweep_error_pct),
            "mean_sweep_error_pct": float(self.mean_sweep_error_pct),
            "solve_time_ms": float(self.solve_time_ms),
            "cond_num": float(self.cond_num),
            "min_panel_ar": float(self.min_panel_ar),
            "is_accurate": bool(self.is_accurate),
            "is_safe": bool(self.is_safe),
            "efficiency_score": float(self.efficiency_score),
            "gci_cl": float(self.gci_cl) if self.gci_cl is not None else None,
            "gci_cdi": float(self.gci_cdi) if self.gci_cdi is not None else None,
        }


@dataclass(slots=True)
class GeneralizationGuideline:
    """Design guidelines and dimensionless scaling laws extracted from the convergence study.

    Provides rules to transfer the optimal mesh parameters to similar geometries,
    aspect ratios, and operating conditions.
    """

    recommended_n_panels: int
    recommended_spacing: str
    proportional_panels: bool
    min_panels: int
    aspect_ratio: float
    semi_span: float
    panels_per_ar: float
    panels_per_semi_span: float
    min_dy_over_chord: float
    dihedral_threshold_deg: float
    sweep_threshold_deg: float
    scaling_formula: str
    multi_surface_scaling_rule: str
    spacing_selection_rule: str
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        """Convert generalization guidelines to dictionary."""
        return {
            "recommended_n_panels": self.recommended_n_panels,
            "recommended_spacing": self.recommended_spacing,
            "proportional_panels": self.proportional_panels,
            "min_panels": self.min_panels,
            "aspect_ratio": float(self.aspect_ratio),
            "semi_span": float(self.semi_span),
            "panels_per_ar": float(self.panels_per_ar),
            "panels_per_semi_span": float(self.panels_per_semi_span),
            "min_dy_over_chord": float(self.min_dy_over_chord),
            "scaling_formula": self.scaling_formula,
            "multi_surface_scaling_rule": self.multi_surface_scaling_rule,
            "spacing_selection_rule": self.spacing_selection_rule,
            "explanation": self.explanation,
        }

    def to_markdown(self) -> str:
        """Format guidelines as markdown documentation."""
        return (
            f"### Mesh Generalization Guidelines for Similar Geometries\n\n"
            f"- **Primary Wing Resolution**: `N = max({self.min_panels}, round({self.panels_per_ar:.2f} * AR))` panels per semi-span.\n"
            f"- **Span Density**: `{self.panels_per_semi_span:.1f}` panels/meter of semi-span.\n"
            f"- **Recommended Spacing**: `{self.recommended_spacing}` (see selection criteria below).\n"
            f"- **Multi-Surface Scaling**: `{self.multi_surface_scaling_rule}`\n"
            f"- **Spacing Selection Rule**: {self.spacing_selection_rule}\n\n"
            f"**Rationale**:\n{self.explanation}\n"
        )


@dataclass
class MeshConvergenceResult:
    """Comprehensive output of an Ventorum mesh convergence study.

    Contains all evaluated mesh points, the identified minimal mesh, the recommended
    production mesh, the reference solution, and generalization rules.

    Parameters
    ----------
    converged : bool
        True only when at least one evaluated level meets the tolerance on
        every checked quantity. When False, the minimal and recommended
        meshes are the finest evaluated level.
    """

    case_name: str
    points: list[MeshConvergencePoint]
    minimal_mesh: MeshConvergencePoint
    recommended_mesh: MeshConvergencePoint
    recommended_settings: SolverSettings
    reference_point: MeshConvergencePoint
    reference_result: SolverResult
    tolerance_pct: float
    target_metric: str
    generalization: GeneralizationGuideline
    alpha_tested_deg: list[float]
    geometry_summary: dict[str, Any]
    profiling: dict[str, Any] = field(default_factory=dict)
    # Panel count of each surface that has its own ``n_panels`` (None for the
    # other surfaces), scaled to the recommended mesh.
    recommended_surface_n_panels: list[int | None] = field(default_factory=list)
    converged: bool = False

    def summary(self, as_markdown: bool = False) -> str:
        """Format a human-readable or Markdown summary table."""
        ref_cl = self.reference_point.CL
        ref_cdi = self.reference_point.CDi
        ref_panels = self.reference_point.total_panels
        do_sweep = len(self.alpha_tested_deg) > 1
        smallest_err = smallest_reached_error_pct(self.points, self.target_metric, do_sweep)

        lines: list[str] = []
        if as_markdown:
            lines.append(f"## Ventorum Mesh Convergence Study: {self.case_name}")
            lines.append("")
            lines.append(f"- **Target Tolerance**: `{self.tolerance_pct:.2f}%` on `{self.target_metric}`")
            if not self.converged:
                lines.append(
                    f"- **Convergence**: not converged within tolerance `{self.tolerance_pct:.2f}%`; "
                    f"smallest error reached `{smallest_err:.3f}%`."
                )
            lines.append(f"- **Reference Solution**: `N={self.reference_point.n_panels} {self.reference_point.spacing}` ({ref_panels} total panels)")
            ref_tot_drag = f"{self.reference_point.CD_total:.6f}" if self.reference_point.CD_total is not None else f"{ref_cdi:.6f}"
            lines.append(f"  - $C_L = {ref_cl:.6f}$, $C_{{Di}} = {ref_cdi:.6f}$, $C_{{D,tot}} = {ref_tot_drag}$")
            lines.append("")
            lines.append("| N_ref | Spacing | N_tot | N_sol | CL | CL Err% | CDi Err% | Γ L2 Err% | Max Swp% | Time (ms) | Cond # | Status |")
            lines.append("|:-----:|:-------:|:-----:|:-----:|:--:|:-------:|:--------:|:---------:|:--------:|:---------:|:------:|:------:|")
            for pt in self.points:
                status = "Pass" if pt.is_accurate else "Coarse"
                if pt == self.minimal_mesh:
                    status = "**[MINIMAL]**"
                if pt == self.recommended_mesh:
                    status = "**[RECOMMENDED]**"
                lines.append(
                    f"| {pt.n_panels} | {pt.spacing} | {pt.total_panels} | {pt.panels_solved} | {pt.CL:.5f} | "
                    f"{pt.error_cl_pct:.3f}% | {pt.error_cdi_pct:.3f}% | {pt.error_gamma_l2_pct:.3f}% | "
                    f"{pt.max_sweep_error_pct:.3f}% | {pt.solve_time_ms:.2f} | {pt.cond_num:.1e} | {status} |"
                )
            lines.append("")
            lines.append("### Optimal Mesh Selection")
            if not self.converged:
                lines.append(
                    f"- **Convergence**: not converged within tolerance `{self.tolerance_pct:.2f}%` on "
                    f"`{self.target_metric}`; smallest error reached `{smallest_err:.3f}%`. "
                    f"The finest evaluated mesh is recommended."
                )
            lines.append(
                f"- **Minimal Mesh**: `N={self.minimal_mesh.n_panels} {self.minimal_mesh.spacing}` "
                f"({self.minimal_mesh.total_panels} total panels, {self.minimal_mesh.panels_solved} panels solved) -> $C_L$ err: `{self.minimal_mesh.error_cl_pct:.3f}%`, "
                f"$C_{{Di}}$ err: `{self.minimal_mesh.error_cdi_pct:.3f}%`, Solve time: `{self.minimal_mesh.solve_time_ms:.2f} ms`"
            )
            lines.append(
                f"- **Recommended Mesh**: `N={self.recommended_mesh.n_panels} {self.recommended_mesh.spacing}` "
                f"({self.recommended_mesh.total_panels} total panels, {self.recommended_mesh.panels_solved} panels solved) -> $C_L$ err: `{self.recommended_mesh.error_cl_pct:.3f}%`, "
                f"$C_{{Di}}$ err: `{self.recommended_mesh.error_cdi_pct:.3f}%`, Solve time: `{self.recommended_mesh.solve_time_ms:.2f} ms`"
            )
            lines.append("")
            lines.append(self.generalization.to_markdown())
        else:
            w = 115
            ref_tot_drag = f"{self.reference_point.CD_total:.6f}" if self.reference_point.CD_total is not None else f"{ref_cdi:.6f}"
            lines.append("=" * w)
            lines.append(f"Ventorum MESH CONVERGENCE STUDY: {self.case_name.upper()}".center(w))
            lines.append("=" * w)
            lines.append(
                f"Target Tolerance: {self.tolerance_pct:.2f}% ({self.target_metric}) | "
                f"Reference Mesh: N={self.reference_point.n_panels} {self.reference_point.spacing} ({ref_panels} panels)"
            )
            if not self.converged:
                lines.append(
                    f"Convergence: not converged within tolerance {self.tolerance_pct:.2f}% "
                    f"({self.target_metric}); smallest error reached {smallest_err:.3f}%."
                )
            lines.append(
                f"Reference Solution: CL = {ref_cl:.6f}, CDi = {ref_cdi:.6f}, CD_total = {ref_tot_drag} "
                f"(solve time: {self.reference_point.solve_time_ms:.2f} ms)"
            )
            lines.append("-" * w)
            header = f"{'N_ref':<6} | {'Spacing':<12} | {'N_tot':<6} | {'N_sol':<6} | {'CL':<9} | {'CL Err%':<8} | {'CDi Err%':<9} | {'Gamma L2%':<9} | {'Max Swp%':<9} | {'Time (ms)':<9} | {'Cond #':<9} | {'Status'}"
            lines.append(header)
            lines.append("-" * w)
            for pt in self.points:
                status = "Accurate" if pt.is_accurate else "Coarse"
                if pt == self.minimal_mesh:
                    status = "[MINIMAL]"
                if pt == self.recommended_mesh:
                    status = "[RECOMMENDED]"
                row = (
                    f"{pt.n_panels:<6d} | {pt.spacing:<12s} | {pt.total_panels:<6d} | {pt.panels_solved:<6d} | {pt.CL:<9.5f} | "
                    f"{pt.error_cl_pct:<7.3f}% | {pt.error_cdi_pct:<8.3f}% | {pt.error_gamma_l2_pct:<8.3f}% | "
                    f"{pt.max_sweep_error_pct:<8.3f}% | {pt.solve_time_ms:<9.2f} | {pt.cond_num:<9.1e} | {status}"
                )
                lines.append(row)
            lines.append("-" * w)
            spd_min = self.reference_point.solve_time_ms / max(self.minimal_mesh.solve_time_ms, 1e-6)
            spd_rec = self.reference_point.solve_time_ms / max(self.recommended_mesh.solve_time_ms, 1e-6)
            if not self.converged:
                lines.append(
                    f"[!] NOT CONVERGED: not converged within tolerance {self.tolerance_pct:.2f}% ({self.target_metric}); "
                    f"smallest error reached {smallest_err:.3f}%. The finest evaluated mesh is recommended."
                )
            lines.append(
                f"[+] MINIMAL MESH:     N={self.minimal_mesh.n_panels:<2d} {self.minimal_mesh.spacing:<11s} "
                f"({self.minimal_mesh.total_panels} panels, {self.minimal_mesh.panels_solved} solved) | CL err: {self.minimal_mesh.error_cl_pct:.3f}% | "
                f"CDi err: {self.minimal_mesh.error_cdi_pct:.3f}% | Time: {self.minimal_mesh.solve_time_ms:.2f}ms ({spd_min:.1f}x speedup)"
            )
            lines.append(
                f"[+] RECOMMENDED MESH: N={self.recommended_mesh.n_panels:<2d} {self.recommended_mesh.spacing:<11s} "
                f"({self.recommended_mesh.total_panels} panels, {self.recommended_mesh.panels_solved} solved) | CL err: {self.recommended_mesh.error_cl_pct:.3f}% | "
                f"CDi err: {self.recommended_mesh.error_cdi_pct:.3f}% | Time: {self.recommended_mesh.solve_time_ms:.2f}ms ({spd_rec:.1f}x speedup)"
            )
            lines.append(
                f"    -> Scaling Guideline: k_AR = {self.generalization.panels_per_ar:.2f} panels/AR. "
                f"For similar geometries: N ~= max({self.generalization.min_panels}, round({self.generalization.panels_per_ar:.2f} * AR))."
            )
            if self.profiling:
                t_tot = self.profiling.get("t_total_ms", 0.0)
                t_ref = self.profiling.get("t_ref_ms", 0.0)
                t_cands = self.profiling.get("t_cands_ms", 0.0)
                t_ana = self.profiling.get("t_analysis_ms", 0.0)
                th_p = self.profiling.get("throughput_panels_sec", 0.0)
                th_s = self.profiling.get("throughput_solves_sec", 0.0)
                lines.append("-" * w)
                lines.append(
                    f"[*] Study Execution Profile: Total={t_tot:.1f}ms | Ref Solve={t_ref:.1f}ms ({(t_ref/max(t_tot,1e-6))*100:.1f}%) | "
                    f"Candidate Sweep={t_cands:.1f}ms ({(t_cands/max(t_tot,1e-6))*100:.1f}%) | Analytics={t_ana:.1f}ms | "
                    f"Throughput: {th_p:.0f} panels/s, {th_s:.1f} solves/s"
                )
            lines.append("=" * w)

        return "\n".join(lines)

    def plot(self, save_path: str | Path | None = None, show: bool = False) -> Any:
        """Generate a 4-panel publication-grade mesh convergence plot."""
        from ventorum.visualization.convergence import plot_mesh_convergence
        return plot_mesh_convergence(self, save_path=save_path, show=show)

    def to_dict(self) -> dict[str, Any]:
        """Convert entire study to a JSON-serializable dictionary."""
        return {
            "case_name": self.case_name,
            "converged": bool(self.converged),
            "tolerance_pct": float(self.tolerance_pct),
            "target_metric": self.target_metric,
            "alpha_tested_deg": [float(a) for a in self.alpha_tested_deg],
            "geometry_summary": self.geometry_summary,
            "reference": self.reference_point.to_dict(),
            "minimal_mesh": self.minimal_mesh.to_dict(),
            "recommended_mesh": self.recommended_mesh.to_dict(),
            "recommended_settings": {
                "n_panels": self.recommended_settings.n_panels,
                "spacing": self.recommended_settings.spacing,
                "proportional_panels": self.recommended_settings.proportional_panels,
                "min_panels": self.recommended_settings.min_panels,
                "use_symmetry": self.recommended_settings.use_symmetry,
                "n_chord": self.recommended_settings.n_chord,
                "surface_n_panels": list(self.recommended_surface_n_panels),
            },
            "generalization": self.generalization.to_dict(),
            "points": [p.to_dict() for p in self.points],
            "profiling": self.profiling,
        }

    def to_dataframe(self) -> Any:
        """Export convergence points to a pandas DataFrame (if pandas is available)."""
        try:
            import pandas as pd
            records = [p.to_dict() for p in self.points]
            return pd.DataFrame(records)
        except ImportError as e:
            raise ImportError("pandas is required for to_dataframe(). Install with 'pip install pandas'.") from e


# ═══════════════════════════════════════════════════════════════════════════════
# Core Study Engine
# ═══════════════════════════════════════════════════════════════════════════════

def _level_aircraft(aircraft: Aircraft, n_level: int, n_base: int) -> Aircraft:
    """Aircraft for one mesh level.

    A surface with its own ``n_panels`` keeps it for every level of the
    solver settings, so the study would not refine it. Here that count is
    taken as the count for *n_base* panels and is scaled by
    ``n_level / n_base`` (at least 4, the smallest count that the input
    check accepts). A surface with its own ``spacing`` keeps it.
    """
    if all(s.n_panels is None for s in aircraft.surfaces):
        return aircraft
    out = aircraft.clone()
    scale = n_level / max(int(n_base), 1)
    for surf in out.surfaces:
        if surf.n_panels is not None:
            surf.n_panels = max(4, int(round(surf.n_panels * scale)))
    return out


def _level_key(aircraft: Aircraft, settings: SolverSettings, n_level: int) -> tuple:
    """Key of the mesh that a level gives (two levels with the same key give the same mesh)."""
    from ventorum.geometry.lattice import resolve_spacing
    from ventorum.solvers.factory import resolve_solver_type

    canonical = resolve_solver_type(settings.solver_type)
    collocation = "vlm" if canonical == "vlm" else "llt"
    spacings = tuple(
        resolve_spacing(s.spacing if s.spacing is not None else settings.spacing, s, collocation)
        for s in aircraft.surfaces
    )
    return (int(n_level), spacings)


def _mesh_stats(
    res: SolverResult, aircraft: Aircraft, settings: SolverSettings, use_symmetry: bool,
) -> tuple[int, float, int]:
    """Total spanwise panels, smallest panel width / chord, and panels solved.

    The third value is the total number of panels of the lattice that was
    solved, all surfaces (both halves and all chordwise panels).
    """
    lat = res.details.get("lattice") if res.details else None
    if lat is not None:
        # With symmetry, the left half of a symmetric surface is not counted.
        counted = (lat.strip_is_right | (lat.strip_mirror < 0)) if use_symmetry else np.ones(lat.n_strips, bool)
        min_ar = float(np.min(lat.width / np.maximum(lat.chord, 1e-12)))
        return int(np.sum(counted)), min_ar, int(lat.n_panels)
    ds = discretize_aircraft_surfaces(aircraft, settings, half_mesh=use_symmetry)
    min_ar = min(float(np.min(d.dy_panels / np.maximum(d.chords, 1e-12))) for d in ds)
    total = sum(len(d.dy_panels) for d in ds)
    return total, min_ar, total


def _span_coordinate(res: SolverResult, s_idx: int) -> np.ndarray:
    """Span fraction (eta, signed) of the strips of surface *s_idx*.

    It is measured along the surface, so it is correct also for a vertical
    or a highly dihedral surface. Without a lattice (Fourier solver) the
    coordinate is y divided by its largest value.
    """
    sw = res.spanwise[s_idx]
    lat = res.details.get("lattice") if res.details else None
    if lat is not None and s_idx < len(lat.surfaces):
        eta = np.asarray(lat.eta[lat.surfaces[s_idx].strips], dtype=float)
        if eta.size == len(sw.gamma):
            return eta
    y = np.asarray(sw.y, dtype=float)
    return y / max(float(np.max(np.abs(y))), 1e-12) if y.size else y


def _gamma_l2_error_pct(res: SolverResult, ref_res: SolverResult) -> float:
    """Mean over the surfaces of the relative L2 error of the circulation [%].

    Each surface is compared along its span fraction, over the inner 95 % of
    the span.
    """
    l2_errs = []
    for s_idx in range(min(len(res.spanwise), len(ref_res.spanwise))):
        g_cand = np.asarray(res.spanwise[s_idx].gamma, dtype=float)
        g_ref = np.asarray(ref_res.spanwise[s_idx].gamma, dtype=float)
        if g_cand.size == 0 or g_ref.size == 0:
            continue
        e_cand = _span_coordinate(res, s_idx)
        e_ref = _span_coordinate(ref_res, s_idx)
        o_c = np.argsort(e_cand, kind="stable")
        e_cand, g_cand = e_cand[o_c], g_cand[o_c]
        b_s = max(float(np.max(np.abs(e_ref))), 1e-12)
        mask = np.abs(e_ref) <= 0.95 * b_s
        e_eval = e_ref[mask] if np.any(mask) else e_ref
        g_ref_eval = g_ref[mask] if np.any(mask) else g_ref
        gamma_interp = np.interp(e_eval, e_cand, g_cand)
        norm_diff = np.linalg.norm(gamma_interp - g_ref_eval)
        norm_ref = np.linalg.norm(g_ref_eval)
        l2_errs.append((norm_diff / max(norm_ref, 1e-12)) * 100.0)
    return float(np.mean(l2_errs)) if l2_errs else 0.0


def _solve_sweep(
    aircraft: Aircraft,
    condition: FlightCondition,
    settings: SolverSettings,
    alpha_sweep_deg: Sequence[float],
    primary: SolverResult,
) -> list[SolverResult]:
    """Solve each angle of the sweep; an angle equal to the primary condition uses *primary*."""
    from ventorum import analyze

    out: list[SolverResult] = []
    for alpha in alpha_sweep_deg:
        a_rad = float(np.radians(alpha))
        if np.isclose(a_rad, condition.alpha, rtol=0.0, atol=1e-12):
            out.append(primary)
            continue
        cond_a = condition.clone()
        cond_a.alpha = a_rad
        out.append(analyze(aircraft, condition=cond_a, settings=settings))
    return out


def _evaluate_candidate(
    aircraft: Aircraft,
    condition: FlightCondition,
    settings: SolverSettings,
    alpha_sweep_deg: Sequence[float] | None,
    ref_res: SolverResult,
    ref_sweep_res: list[SolverResult] | None,
    n_repeats: int = 1,
) -> tuple[SolverResult, float, float, float, float, float, float, int, int]:
    """Execute evaluation of a candidate mesh, computing errors, time, and conditioning.

    Returns ``(result, solve_ms, cond_num, min_panel_ar, max_sweep_err, mean_sweep_err,
    gamma_l2_err, total_panels, panels_solved)``.
    """
    from ventorum import analyze

    try:
        from ventorum.solvers.lattice_base import assemble_system_matrix
        cond_num = float(np.linalg.cond(assemble_system_matrix(aircraft, condition, settings)))
    except Exception:
        cond_num = 1e12

    # Measure timing on the primary flight condition
    times: list[float] = []
    res_primary = None
    for _ in range(max(1, n_repeats)):
        t0 = time.perf_counter()
        res_primary = analyze(aircraft, condition=condition, settings=settings)
        times.append((time.perf_counter() - t0) * 1000.0)
    t_solve_ms = float(np.median(times)) if times else 0.0

    tot_panels, min_ar, panels_solved = _mesh_stats(res_primary, aircraft, settings, settings.use_symmetry)

    # If alpha sweep is evaluated, evaluate across all angles
    sweep_errors: list[float] = []
    if alpha_sweep_deg is not None and len(alpha_sweep_deg) > 1 and ref_sweep_res is not None:
        cl_scale = max(float(max(abs(r.totals.CL) for r in ref_sweep_res)), 0.2)
        cdi_scale = max(float(max(abs(r.totals.CDi) for r in ref_sweep_res)), 0.005)
        sweep_res = _solve_sweep(aircraft, condition, settings, alpha_sweep_deg, res_primary)
        for res_a, ref_a in zip(sweep_res, ref_sweep_res):
            # Standard aerodynamic flight envelope error normalization:
            # Scaled relative to the envelope maximum to prevent zero-lift singularities
            err_cl = abs(res_a.totals.CL - ref_a.totals.CL) / cl_scale * 100.0
            err_cdi = abs(res_a.totals.CDi - ref_a.totals.CDi) / cdi_scale * 100.0
            sweep_errors.append(max(err_cl, err_cdi))

    max_swp_err = float(np.max(sweep_errors)) if sweep_errors else 0.0
    mean_swp_err = float(np.mean(sweep_errors)) if sweep_errors else 0.0

    # L2 circulation error compared to reference, per surface along its span.
    l2_gamma_err = _gamma_l2_error_pct(res_primary, ref_res)

    return res_primary, t_solve_ms, cond_num, min_ar, max_swp_err, mean_swp_err, l2_gamma_err, tot_panels, panels_solved


def _compute_gci_triplets(points_by_spacing: dict[str, list[MeshConvergencePoint]]) -> None:
    """Compute apparent order and Grid Convergence Index (GCI) for monotonic triplets."""
    for _sp, pts in points_by_spacing.items():
        if len(pts) < 3:
            continue
        # Sort by panel count
        sorted_pts = sorted(pts, key=lambda p: p.n_panels)
        for i in range(len(sorted_pts) - 2):
            p1, p2, p3 = sorted_pts[i], sorted_pts[i + 1], sorted_pts[i + 2]
            n1, n2, n3 = p1.n_panels, p2.n_panels, p3.n_panels
            r12 = n2 / max(n1, 1)
            r23 = n3 / max(n2, 1)
            if r12 < 1.15 or r23 < 1.15:
                continue

            # GCI for CL
            e23 = p3.CL - p2.CL
            e12 = p2.CL - p1.CL
            if (e23 * e12) > 0:  # Monotonic convergence
                r_avg = 0.5 * (r12 + r23)
                if r_avg > 1.0 and abs(e23) > 0.0:
                    p_order = np.log(abs(e12 / e23)) / np.log(r_avg)
                    if p_order > 0.1:
                        denom = (r23 ** p_order) - 1.0
                        if denom > 1e-6:
                            gci = (1.25 * abs(e23 / max(abs(p3.CL), 1e-6)) / denom) * 100.0
                            p3.gci_cl = float(gci)

            # GCI for CDi
            e23_cd = p3.CDi - p2.CDi
            e12_cd = p2.CDi - p1.CDi
            if (e23_cd * e12_cd) > 0:
                r_avg = 0.5 * (r12 + r23)
                if r_avg > 1.0 and abs(e23_cd) > 0.0:
                    p_order_cd = np.log(abs(e12_cd / e23_cd)) / np.log(r_avg)
                    if p_order_cd > 0.1:
                        denom_cd = (r23 ** p_order_cd) - 1.0
                        if denom_cd > 1e-6:
                            gci_cd = (1.25 * abs(e23_cd / max(abs(p3.CDi), 1e-6)) / denom_cd) * 100.0
                            p3.gci_cdi = float(gci_cd)


def _checked_error_pct(
    pt: MeshConvergencePoint, target_metric: str, do_sweep: bool,
) -> float:
    """Largest error of one level over every checked quantity [%].

    The checked quantities are the ones that ``is_accurate`` tests: the
    target metric, plus the sweep error when a sweep was evaluated.
    """
    if target_metric == "CL":
        err = float(pt.error_cl_pct)
    elif target_metric == "CDi":
        err = float(pt.error_cdi_pct)
    elif target_metric == "circulation":
        err = float(pt.error_gamma_l2_pct)
    else:
        err = float(max(pt.error_cl_pct, pt.error_cdi_pct))
    if do_sweep:
        err = float(max(err, pt.max_sweep_error_pct))
    return err


def smallest_reached_error_pct(
    points: Sequence[MeshConvergencePoint], target_metric: str, do_sweep: bool,
) -> float:
    """Smallest checked error reached by any evaluated level [%]."""
    if not points:
        return float("nan")
    return float(min(_checked_error_pct(p, target_metric, do_sweep) for p in points))


def run_mesh_convergence_study(
    case: Any,
    condition: FlightCondition | None = None,
    *,
    tolerance_pct: float = 0.5,
    cl_tolerance_pct: float | None = None,
    cdi_tolerance_pct: float | None = None,
    target_metric: Literal["both", "CL", "CDi", "circulation"] = "both",
    spacing_schemes: Sequence[str] | str = ("auto", "half-cosine", "cosine", "uniform"),
    panel_counts: Sequence[int] | None = None,
    ref_n_panels: int = 160,
    ref_spacing: str = "auto",
    reference_result: SolverResult | None = None,
    proportional_panels: bool = True,
    robustness_margin: float = 0.8,
    evaluate_sweep: bool | None = None,
    alpha_sweep_deg: Sequence[float] | None = None,
    solver_type: str | None = None,
    use_symmetry: bool = True,
    progress: bool = False,
    apply_to_case: bool = False,
) -> MeshConvergenceResult:
    """Execute a comprehensive aerodynamic mesh convergence study on a case.

    Determines the minimal and recommended mesh configurations that deliver accurate
    aerodynamic forces within tolerance across the studied flight envelope, and extracts
    generalization guidelines that apply to similar geometries and conditions.

    Parameters
    ----------
    case : Ventorum | Aircraft | LiftingSurface
        The case or geometry to evaluate. If an `Ventorum` instance is supplied, its
        configured geometry, flight condition, solver settings, and alpha sweep are
        automatically utilized.
    condition : FlightCondition or None
        Flight condition override. If None and case is an instance, uses instance condition.
    tolerance_pct : float
        Overall target percentage error relative to reference (default 0.5%, i.e. 0.005).
    cl_tolerance_pct : float or None
        Explicit tolerance for CL (defaults to `tolerance_pct`).
    cdi_tolerance_pct : float or None
        Explicit tolerance for CDi (defaults to `tolerance_pct`).
    target_metric : 'both', 'CL', 'CDi', or 'circulation'
        Quantity of interest used to determine convergence.
    spacing_schemes : Sequence[str] or str
        Spanwise node spacing schemes to test (e.g. 'auto', 'half-cosine', 'cosine', 'uniform').
    panel_counts : Sequence[int] or None
        Panel counts per semi-span to evaluate. Default: (10, 15, 20, 30, 40, 60, 80).
    ref_n_panels : int
        Panels per semi-span for the high-fidelity reference solution (default 160).
    ref_spacing : str
        Spacing scheme for the reference solution (default 'auto').
    reference_result : SolverResult or None
        Optional user-provided reference solution.
    proportional_panels : bool
        If True, scales panel counts proportionally across surfaces based on relative span.
        A surface with its own ``n_panels`` is refined too: its count is taken as the
        count for the ``n_panels`` of the case settings (default 80) and is scaled with
        each level. A surface with its own ``spacing`` keeps that spacing.
    robustness_margin : float
        Safety factor applied to the tolerance to determine the recommended mesh
        (default 0.8, targeting 0.8 * tolerance_pct to ensure robustness for similar cases).
    evaluate_sweep : bool or None
        Whether to evaluate accuracy across an angle-of-attack sweep. If None, automatically
        detected if case has an alpha sweep configured.
    alpha_sweep_deg : Sequence[float] or None
        Array of angles of attack [deg] for the sweep evaluation.
    solver_type : str or None
        Solver algorithm to use (default from case settings or 'horseshoe').
    use_symmetry : bool
        Whether to utilize symmetry plane acceleration.
    progress : bool
        If True, prints progress updates during execution.
    apply_to_case : bool
        If True and case is an `Ventorum` instance, automatically applies the recommended
        solver settings to the instance.

    Returns
    -------
    MeshConvergenceResult
        Comprehensive result container with points, minimal/recommended meshes, and guidelines.
    """
    from ventorum import analyze

    # 1. Unpack case geometry, flight condition, and settings
    is_instance = hasattr(case, "aircraft") and hasattr(case, "settings")
    if is_instance:
        aircraft = case.aircraft.clone()
        base_cond = condition.clone() if condition is not None else case.condition.clone()
        base_settings = case.settings.clone()
        case_name = getattr(case, "name", "Ventorum_Case")
        sweep = case.alpha_sweep_deg if alpha_sweep_deg is None else alpha_sweep_deg
    elif isinstance(case, LiftingSurface):
        aircraft = Aircraft(name=f"{getattr(case, 'name', 'Wing')}_Aircraft", surfaces=[case.clone()])
        aircraft.compute_reference_values()
        base_cond = condition.clone() if condition is not None else FlightCondition(alpha=np.radians(5.0))
        base_settings = SolverSettings()
        case_name = getattr(case, "name", "LiftingSurface")
        sweep = alpha_sweep_deg
    elif isinstance(case, Aircraft):
        aircraft = case.clone()
        aircraft.compute_reference_values()
        base_cond = condition.clone() if condition is not None else FlightCondition(alpha=np.radians(5.0))
        base_settings = SolverSettings()
        case_name = getattr(case, "name", "Aircraft")
        sweep = alpha_sweep_deg
    else:
        raise TypeError(f"Unsupported case type: {type(case)}. Expected Ventorum, Aircraft, or LiftingSurface.")

    # A surface's own n_panels is scaled with the level relative to this count.
    n_base = int(base_settings.n_panels)

    # Override solver settings if requested
    if solver_type is not None:
        base_settings.solver_type = solver_type
    base_settings.use_symmetry = use_symmetry
    base_settings.proportional_panels = proportional_panels

    # Tolerances
    tol_cl = float(cl_tolerance_pct if cl_tolerance_pct is not None else tolerance_pct)
    tol_cdi = float(cdi_tolerance_pct if cdi_tolerance_pct is not None else tolerance_pct)

    # Spacing schemes
    if isinstance(spacing_schemes, str):
        schemes = [spacing_schemes]
    else:
        schemes = list(spacing_schemes)

    # Panel counts to test
    if panel_counts is None:
        default_counts = [10, 15, 20, 30, 40, 60, 80]
        # Keep only counts strictly below ref_n_panels
        p_counts = [n for n in default_counts if n < ref_n_panels]
        if not p_counts:
            p_counts = [10, 20, 40]
    else:
        p_counts = [int(n) for n in panel_counts if int(n) < ref_n_panels]
        if not p_counts:
            p_counts = [int(n) for n in panel_counts]
    p_counts = list(dict.fromkeys(p_counts))  # each level once

    # Angle of attack sweep setup
    if evaluate_sweep is None:
        do_sweep = bool(sweep is not None and len(sweep) > 1)
    else:
        do_sweep = bool(evaluate_sweep)

    if do_sweep and (sweep is None or len(sweep) <= 1):
        sweep = np.array([-2.0, 0.0, 2.0, 5.0, 8.0, 10.0])
    sweep_angles = [float(a) for a in sweep] if (do_sweep and sweep is not None) else [float(np.degrees(base_cond.alpha))]

    t_study_start = time.perf_counter()

    if progress:
        print(f"[*] Initiating Mesh Convergence Study for '{case_name}'...")
        print(f"    Target Tolerance: {tolerance_pct:.2f}% | Evaluated Schemes: {schemes}")

    # 2. Establish high-resolution Reference Solution
    ref_settings = base_settings.clone()
    ref_settings.n_panels = ref_n_panels
    ref_settings.spacing = ref_spacing
    ref_settings.proportional_panels = proportional_panels

    ref_aircraft = _level_aircraft(aircraft, ref_n_panels, n_base)
    ref_sweep_res: list[SolverResult] | None = None
    if reference_result is not None:
        ref_res = reference_result
        ref_t_ms = 0.0
        ref_cond = 1.0
        ref_min_ar = 0.1
        ref_tot_panels, _, ref_panels_solved = _mesh_stats(ref_res, ref_aircraft, ref_settings, use_symmetry)
    else:
        # Measure reference solve
        t_ref0 = time.perf_counter()
        ref_res = analyze(ref_aircraft, condition=base_cond, settings=ref_settings)
        ref_t_ms = (time.perf_counter() - t_ref0) * 1000.0

        ref_tot_panels, ref_min_ar, ref_panels_solved = _mesh_stats(ref_res, ref_aircraft, ref_settings, use_symmetry)
        try:
            from ventorum.solvers.lattice_base import assemble_system_matrix
            ref_cond = float(np.linalg.cond(assemble_system_matrix(ref_aircraft, base_cond, ref_settings)))
        except Exception:
            ref_cond = 1e6

        if do_sweep:
            ref_sweep_res = _solve_sweep(ref_aircraft, base_cond, ref_settings, sweep_angles, ref_res)

    t_ref_done = time.perf_counter()

    ref_point = MeshConvergencePoint(
        n_panels=ref_n_panels,
        spacing=ref_spacing,
        total_panels=ref_tot_panels,
        panels_solved=ref_panels_solved,
        CL=ref_res.totals.CL,
        CDi=ref_res.totals.CDi,
        CD_total=ref_res.totals.CD_total,
        Cm=ref_res.totals.Cm,
        error_cl_pct=0.0,
        error_cdi_pct=0.0,
        error_gamma_l2_pct=0.0,
        max_sweep_error_pct=0.0,
        mean_sweep_error_pct=0.0,
        solve_time_ms=ref_t_ms,
        cond_num=ref_cond,
        min_panel_ar=ref_min_ar,
        is_accurate=True,
        is_safe=True,
        efficiency_score=1.0,
        result=ref_res,
    )

    # 3. Evaluate candidate mesh combinations
    points: list[MeshConvergencePoint] = []
    points_by_spacing: dict[str, list[MeshConvergencePoint]] = {sp: [] for sp in schemes}
    # Levels that give the same mesh (for example 'auto' and the spacing it
    # selects) are solved once.
    solved_levels: dict[tuple, tuple] = {}

    for sp in schemes:
        for n_p in p_counts:
            cand_settings = base_settings.clone()
            cand_settings.n_panels = n_p
            cand_settings.spacing = sp
            cand_settings.proportional_panels = proportional_panels
            cand_aircraft = _level_aircraft(aircraft, n_p, n_base)

            key = _level_key(cand_aircraft, cand_settings, n_p)
            if key not in solved_levels:
                solved_levels[key] = _evaluate_candidate(
                    aircraft=cand_aircraft,
                    condition=base_cond,
                    settings=cand_settings,
                    alpha_sweep_deg=sweep_angles if do_sweep else None,
                    ref_res=ref_res,
                    ref_sweep_res=ref_sweep_res,
                )
            (
                res,
                t_ms,
                cond_num,
                min_ar,
                max_swp_err,
                mean_swp_err,
                l2_gamma_err,
                tot_panels,
                panels_solved,
            ) = solved_levels[key]

            # Calculate relative percentage errors on primary condition
            err_cl = abs(res.totals.CL - ref_res.totals.CL) / max(abs(ref_res.totals.CL), 1e-4) * 100.0
            err_cdi = abs(res.totals.CDi - ref_res.totals.CDi) / max(abs(ref_res.totals.CDi), 1e-5) * 100.0

            # Determine accuracy pass/fail based on target_metric
            if target_metric == "CL":
                metric_pass = (err_cl <= tol_cl)
            elif target_metric == "CDi":
                metric_pass = (err_cdi <= tol_cdi)
            elif target_metric == "circulation":
                metric_pass = (l2_gamma_err <= tolerance_pct)
            else:  # "both"
                metric_pass = (err_cl <= tol_cl) and (err_cdi <= tol_cdi)

            if do_sweep:
                metric_pass = metric_pass and (max_swp_err <= max(tol_cl, tol_cdi))

            # Numerical safety checks
            is_safe = (cond_num < 1e8) and (min_ar > 0.001)

            # Efficiency score: rewards speedup and accuracy, penalizes unnecessary panels
            spd = ref_t_ms / max(t_ms, 1e-4)
            avg_err = 0.5 * (err_cl + err_cdi)
            eff_score = float(spd / (1.0 + avg_err))

            pt = MeshConvergencePoint(
                n_panels=n_p,
                spacing=sp,
                total_panels=tot_panels,
                panels_solved=panels_solved,
                CL=res.totals.CL,
                CDi=res.totals.CDi,
                CD_total=res.totals.CD_total,
                Cm=res.totals.Cm,
                error_cl_pct=err_cl,
                error_cdi_pct=err_cdi,
                error_gamma_l2_pct=l2_gamma_err,
                max_sweep_error_pct=max_swp_err,
                mean_sweep_error_pct=mean_swp_err,
                solve_time_ms=t_ms,
                cond_num=cond_num,
                min_panel_ar=min_ar,
                is_accurate=bool(metric_pass),
                is_safe=bool(is_safe),
                efficiency_score=eff_score,
                result=res,
            )
            points.append(pt)
            points_by_spacing[sp].append(pt)

            if progress:
                status_str = "PASS" if (metric_pass and is_safe) else "FAIL"
                print(f"    -> N={n_p:2d} {sp:<11s} | CL err: {err_cl:6.3f}% | CDi err: {err_cdi:6.3f}% | {status_str}")

    t_cands_done = time.perf_counter()

    # 4. Compute GCI for triplets
    _compute_gci_triplets(points_by_spacing)

    # 5. Identify Minimal Mesh
    # Must be accurate and safe. Among those, minimum total_panels, then solve_time_ms.
    # ``converged`` is True only when at least one level meets the tolerance
    # on every checked quantity.
    converged = bool(any(p.is_accurate for p in points))
    accurate_safe = [p for p in points if p.is_accurate and p.is_safe]
    if accurate_safe:
        minimal_mesh = min(accurate_safe, key=lambda p: (p.total_panels, p.solve_time_ms))
    else:
        # Fallback to candidate with minimum combined error
        minimal_mesh = min(points, key=lambda p: p.error_cl_pct + p.error_cdi_pct)

    # 6. Identify Recommended Production Mesh
    # The recommended mesh must provide a robustness margin (error <= robustness_margin * tolerance)
    # and favor well-behaved spacing schemes ('auto' or tip-clustered) to ensure it reliably
    # generalizes to similar geometries, sweeps, and conditions.
    robust_tol_cl = robustness_margin * tol_cl
    robust_tol_cdi = robustness_margin * tol_cdi
    robust_tol_swp = robustness_margin * max(tol_cl, tol_cdi)

    def is_robust(p: MeshConvergencePoint) -> bool:
        if not p.is_safe:
            return False
        if target_metric == "CL":
            pass_acc = p.error_cl_pct <= robust_tol_cl
        elif target_metric == "CDi":
            pass_acc = p.error_cdi_pct <= robust_tol_cdi
        elif target_metric == "circulation":
            pass_acc = p.error_gamma_l2_pct <= (robustness_margin * tolerance_pct)
        else:
            pass_acc = (p.error_cl_pct <= robust_tol_cl) and (p.error_cdi_pct <= robust_tol_cdi)
        if do_sweep:
            pass_acc = pass_acc and (p.max_sweep_error_pct <= robust_tol_swp)
        return pass_acc

    robust_candidates = [p for p in points if is_robust(p) and p.n_panels >= 12]

    if robust_candidates:
        # Prefer 'auto' spacing if present and robust, otherwise pick highest efficiency score
        auto_robust = [p for p in robust_candidates if p.spacing == "auto"]
        if auto_robust:
            recommended_mesh = min(auto_robust, key=lambda p: (p.total_panels, p.solve_time_ms))
        else:
            recommended_mesh = min(robust_candidates, key=lambda p: (p.total_panels, p.solve_time_ms))
    elif accurate_safe:
        # If no mesh reached the tighter robustness margin, use the minimal mesh or next highest
        recommended_mesh = minimal_mesh
    else:
        recommended_mesh = min(points, key=lambda p: p.error_cl_pct + p.error_cdi_pct)

    if not converged:
        # No level meets the tolerance: recommend the finest evaluated level.
        finest = max(points, key=lambda p: (p.panels_solved, p.total_panels))
        minimal_mesh = finest
        recommended_mesh = finest

    # 7. Formulate Dimensionless Generalization Guidelines
    b_ref = aircraft.b_ref
    s_ref = aircraft.S_ref
    ar_ref = (b_ref ** 2) / max(s_ref, 1e-6)
    semi_span_ref = b_ref / 2.0

    n_rec = recommended_mesh.n_panels
    k_ar = n_rec / max(ar_ref, 1e-4)
    dens_b = n_rec / max(semi_span_ref, 1e-4)

    # Spacing rationale based on surface features
    has_angled_junction = any(
        abs(getattr(s, "dihedral", 0.0)) > np.radians(5.0) or abs(getattr(s, "sweep_le", 0.0)) > np.radians(15.0)
        for s in aircraft.surfaces
    )

    spacing_rule = (
        "Use 'half-cosine' (tip-clustered) for planar wings (|dihedral| <= 5 deg, |sweep| <= 15 deg) "
        "because dGamma/dy = 0 at the root plane by mirror symmetry, saving ~40% panels over uniform meshing. "
        "Use 'cosine' (both root and tip clustered) for surfaces with dihedral kinks (e.g. V-tails) "
        "or sweep > 15 deg where crossflow and apex vortex induction create root-plane gradients."
    )

    multi_surf_rule = (
        "N_secondary = clamp(round(N_main * sqrt(b_secondary / b_main)), min=8, max=500)"
    )

    scaling_formula = f"N = max(12, int(round({k_ar:.2f} * AR)))"

    smallest_err = smallest_reached_error_pct(points, target_metric, do_sweep)
    if converged:
        explanation = (
            f"The studied case '{case_name}' (AR = {ar_ref:.2f}, b = {b_ref:.2f} m) converged with high fidelity "
            f"at N = {n_rec} panels/semi-span using '{recommended_mesh.spacing}' spacing (delivering "
            f"{recommended_mesh.error_cl_pct:.3f}% CL error and {recommended_mesh.error_cdi_pct:.3f}% CDi error "
            f"with a solve time of {recommended_mesh.solve_time_ms:.2f} ms). "
            f"This corresponds to {k_ar:.2f} panels per unit aspect ratio. "
            f"For similar geometries or extended flight envelopes, applying this scaling ensures equivalent "
            f"tip-vortex resolution and circulation gradient capture without numerical ill-conditioning."
        )
    else:
        explanation = (
            f"The studied case '{case_name}' (AR = {ar_ref:.2f}, b = {b_ref:.2f} m) is not converged "
            f"within tolerance {tolerance_pct:.2f}% on '{target_metric}'; the smallest error reached is "
            f"{smallest_err:.3f}%. The finest evaluated mesh (N = {n_rec} panels/semi-span, "
            f"{recommended_mesh.panels_solved} panels solved, '{recommended_mesh.spacing}' spacing) gives "
            f"{recommended_mesh.error_cl_pct:.3f}% CL error and {recommended_mesh.error_cdi_pct:.3f}% CDi error. "
            f"The scaling N = max(12, int(round({k_ar:.2f} * AR))) is given for reference but it is not "
            f"verified to meet the tolerance."
        )

    generalization = GeneralizationGuideline(
        recommended_n_panels=n_rec,
        recommended_spacing=recommended_mesh.spacing,
        proportional_panels=proportional_panels,
        min_panels=8,
        aspect_ratio=ar_ref,
        semi_span=semi_span_ref,
        panels_per_ar=k_ar,
        panels_per_semi_span=dens_b,
        min_dy_over_chord=recommended_mesh.min_panel_ar,
        dihedral_threshold_deg=5.0,
        sweep_threshold_deg=15.0,
        scaling_formula=scaling_formula,
        multi_surface_scaling_rule=multi_surf_rule,
        spacing_selection_rule=spacing_rule,
        explanation=explanation,
    )

    # 8. Build Recommended SolverSettings object: the settings of the case
    # with only the refined values changed.
    recommended_settings = base_settings.clone()
    recommended_settings.n_panels = n_rec
    recommended_settings.spacing = recommended_mesh.spacing
    recommended_settings.proportional_panels = proportional_panels
    recommended_settings.use_symmetry = use_symmetry
    rec_surface_n = [s.n_panels for s in _level_aircraft(aircraft, n_rec, n_base).surfaces]

    # Apply to case instance if requested
    if apply_to_case and is_instance:
        case.settings = recommended_settings.clone()
        for surf, n_s in zip(case.aircraft.surfaces, rec_surface_n):
            if n_s is not None:
                surf.n_panels = n_s

    geom_summary = {
        "aircraft_name": aircraft.name,
        "n_surfaces": len(aircraft.surfaces),
        "b_ref": float(aircraft.b_ref),
        "S_ref": float(aircraft.S_ref),
        "c_ref": float(aircraft.c_ref),
        "aspect_ratio": float(ar_ref),
        "has_angled_junction": bool(has_angled_junction),
    }

    t_study_end = time.perf_counter()
    t_total_ms = (t_study_end - t_study_start) * 1000.0
    t_ref_ms = (t_ref_done - t_study_start) * 1000.0
    t_cands_ms = (t_cands_done - t_ref_done) * 1000.0
    t_analysis_ms = (t_study_end - t_cands_done) * 1000.0

    n_sweep_pts = len(sweep_angles) if do_sweep else 1
    total_solves = (len(solved_levels) + 1) * n_sweep_pts
    total_panels_evaluated = (sum(v[7] for v in solved_levels.values()) + ref_point.total_panels) * n_sweep_pts

    throughput_panels_sec = total_panels_evaluated / max(t_total_ms / 1000.0, 1e-6)
    throughput_solves_sec = total_solves / max(t_total_ms / 1000.0, 1e-6)

    profiling_data = {
        "t_total_ms": t_total_ms,
        "t_ref_ms": t_ref_ms,
        "t_cands_ms": t_cands_ms,
        "t_analysis_ms": t_analysis_ms,
        "total_solves": total_solves,
        "total_panels_evaluated": total_panels_evaluated,
        "throughput_panels_sec": throughput_panels_sec,
        "throughput_solves_sec": throughput_solves_sec,
    }

    result = MeshConvergenceResult(
        case_name=case_name,
        points=points,
        minimal_mesh=minimal_mesh,
        recommended_mesh=recommended_mesh,
        recommended_settings=recommended_settings,
        reference_point=ref_point,
        reference_result=ref_res,
        tolerance_pct=tolerance_pct,
        target_metric=target_metric,
        generalization=generalization,
        alpha_tested_deg=sweep_angles,
        geometry_summary=geom_summary,
        profiling=profiling_data,
        recommended_surface_n_panels=rec_surface_n,
        converged=converged,
    )

    if progress:
        print("\n" + result.summary())

    return result
