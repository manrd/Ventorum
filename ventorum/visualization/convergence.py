# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Plot the convergence history and the mesh convergence of a solution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ventorum.core.datatypes import SolverResult


def plot_convergence(
    result: SolverResult,
    ax: plt.Axes | None = None,
) -> Figure:
    """Plot the iteration residual history for the nonlinear iterative solver.

    Parameters
    ----------
    result : SolverResult
        Must have been produced by the nonlinear solver (otherwise the
        residual history will be empty).
    ax : plt.Axes or None

    Returns
    -------
    Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    if not result.residual_history:
        ax.text(0.5, 0.5, "No iteration data\n(linear solver was used)",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        return fig

    iterations = np.arange(1, len(result.residual_history) + 1)
    residuals = np.array(result.residual_history)

    ax.semilogy(iterations, residuals, "o-", markersize=3, color="#2563eb",
                linewidth=1.5)

    # Mark convergence
    if result.converged:
        ax.axhline(y=residuals[-1], color="green", linestyle="--", alpha=0.5,
                   label=f"Converged: {residuals[-1]:.2e}")
    else:
        ax.axhline(y=residuals[-1], color="red", linestyle="--", alpha=0.5,
                   label=f"NOT converged: {residuals[-1]:.2e}")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("max |ΔΓ| [m²/s]")
    ax.set_title(
        f"Nonlinear Solver Convergence - {result.iterations} iterations"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_mesh_convergence(
    study: Any,
    save_path: str | Path | None = None,
    show: bool = False,
    figsize: tuple[float, float] = (16, 11),
) -> Figure:
    r"""Plot a mesh convergence report with four panels.

    The panels are:

    1. Convergence of :math:`C_L` and :math:`C_{Di}` with the panel count,
       with the reference values and the error tolerance bands.
    2. Decay of the relative error (log scale) for each spacing scheme.
    3. Pareto front of the error and the solve time [ms]. The minimal and
       the recommended meshes are marked.
    4. Spanwise circulation :math:`\Gamma(y)` of the coarse, minimal,
       recommended and reference solutions.

    Parameters
    ----------
    study : MeshConvergenceResult
        Output of ``run_mesh_convergence_study``.
    save_path : str, Path or None, optional
        File path for the image (.png, .pdf, .svg). If None, no file is
        written.
    show : bool, optional
        If True, call ``plt.show()``.
    figsize : tuple of float, optional
        Figure size [in].

    Returns
    -------
    matplotlib.figure.Figure
        The figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    plt.subplots_adjust(hspace=0.32, wspace=0.28)

    ref = study.reference_point
    rec = study.recommended_mesh
    min_pt = study.minimal_mesh
    tol = study.tolerance_pct

    # Color palette
    c_blue = "#2563eb"
    c_emerald = "#059669"
    c_amber = "#d97706"
    c_red = "#dc2626"
    c_purple = "#7c3aed"
    scheme_colors = {
        "auto": "#059669",
        "half-cosine": "#2563eb",
        "cosine": "#7c3aed",
        "uniform": "#dc2626",
        "root-cosine": "#d97706",
        "power": "#0891b2",
    }

    # -------------------------------------------------------------------------
    # Panel 1: CL and CDi Convergence vs Panels (Dual y-axis)
    # -------------------------------------------------------------------------
    ax1 = axes[0, 0]
    ax1_twin = ax1.twinx()

    # Use auto or primary spacing points for clean dual axis
    pts_for_conv = [p for p in study.points if p.spacing == rec.spacing]
    if not pts_for_conv:
        pts_for_conv = study.points

    pts_sorted = sorted(pts_for_conv, key=lambda p: p.n_panels)
    n_vals = [p.n_panels for p in pts_sorted]
    cl_vals = [p.CL for p in pts_sorted]
    cdi_vals = [p.CDi for p in pts_sorted]

    l1 = ax1.plot(n_vals, cl_vals, "o-", color=c_blue, linewidth=1.8, markersize=5, label=f"CL ({rec.spacing})")
    ax1.axhline(ref.CL, color=c_blue, linestyle="--", alpha=0.6, label=f"CL Reference ({ref.CL:.4f})")
    ax1.fill_between(
        [min(n_vals) - 5, max(n_vals) + 10],
        ref.CL * (1.0 - tol / 100.0),
        ref.CL * (1.0 + tol / 100.0),
        color=c_blue, alpha=0.10, label=f"CL ±{tol:.1f}% Tol",
    )

    l2 = ax1_twin.plot(n_vals, cdi_vals, "s-", color=c_amber, linewidth=1.8, markersize=5, label=f"CDi ({rec.spacing})")
    ax1_twin.axhline(ref.CDi, color=c_amber, linestyle="--", alpha=0.6, label=f"CDi Reference ({ref.CDi:.5f})")
    ax1_twin.fill_between(
        [min(n_vals) - 5, max(n_vals) + 10],
        ref.CDi * (1.0 - tol / 100.0),
        ref.CDi * (1.0 + tol / 100.0),
        color=c_amber, alpha=0.10, label=f"CDi ±{tol:.1f}% Tol",
    )

    # Vertical markers for minimal and recommended
    ax1.axvline(min_pt.n_panels, color=c_purple, linestyle=":", alpha=0.7, label=f"Minimal (N={min_pt.n_panels})")
    ax1.axvline(rec.n_panels, color=c_emerald, linestyle="-.", linewidth=1.5, alpha=0.8, label=f"Recommended (N={rec.n_panels})")

    ax1.set_xlabel("Panels per Semi-Span N", fontweight="bold")
    ax1.set_ylabel("Lift Coefficient CL", color=c_blue, fontweight="bold")
    ax1_twin.set_ylabel("Induced Drag Coefficient CDi", color=c_amber, fontweight="bold")
    ax1.set_title("Aerodynamic Coefficients Convergence", fontweight="bold")
    ax1.set_xlim(min(n_vals) - 2, max(n_vals) + 5)
    ax1.grid(True, alpha=0.3)

    # Combined legend
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower right", fontsize=8)

    # -------------------------------------------------------------------------
    # Panel 2: Relative Error Decay vs N (Log scale, comparing spacing schemes)
    # -------------------------------------------------------------------------
    ax2 = axes[0, 1]
    schemes_present = sorted(list({p.spacing for p in study.points}))
    for sp in schemes_present:
        pts_sp = sorted([p for p in study.points if p.spacing == sp], key=lambda p: p.n_panels)
        n_sp = [p.n_panels for p in pts_sp]
        max_err_sp = [max(p.error_cl_pct, p.error_cdi_pct) for p in pts_sp]
        col = scheme_colors.get(sp, "#64748b")
        ax2.semilogy(n_sp, max_err_sp, "o-", color=col, linewidth=1.8, markersize=5, label=f"{sp}")

    ax2.axhline(tol, color=c_red, linestyle="--", linewidth=1.5, label=f"Target Tolerance ({tol:.2f}%)")
    ax2.axhline(study.generalization.recommended_n_panels, color=c_emerald, linestyle=":", alpha=0.0)
    ax2.axvline(rec.n_panels, color=c_emerald, linestyle="-.", alpha=0.7, label=f"Recommended N={rec.n_panels}")

    ax2.set_xlabel("Panels per Semi-Span N", fontweight="bold")
    ax2.set_ylabel("Max Relative Error [%] (log scale)", fontweight="bold")
    ax2.set_title("Error Decay Across Spacing Schemes", fontweight="bold")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(loc="upper right", fontsize=8)

    # -------------------------------------------------------------------------
    # Panel 3: Pareto Frontier (Accuracy vs Solve Time)
    # -------------------------------------------------------------------------
    ax3 = axes[1, 0]
    for sp in schemes_present:
        pts_sp = [p for p in study.points if p.spacing == sp]
        t_sp = [p.solve_time_ms for p in pts_sp]
        err_sp = [max(p.error_cl_pct, p.error_cdi_pct) for p in pts_sp]
        col = scheme_colors.get(sp, "#64748b")
        ax3.scatter(t_sp, err_sp, color=col, s=40, alpha=0.7, label=f"{sp}" if len(schemes_present) > 1 else None)

    # Highlight Minimal Mesh
    ax3.scatter(
        [min_pt.solve_time_ms], [max(min_pt.error_cl_pct, min_pt.error_cdi_pct)],
        color=c_amber, marker="D", s=140, edgecolors="black", linewidth=1.5,
        label=f"Minimal Mesh (N={min_pt.n_panels}, {min_pt.solve_time_ms:.1f}ms)",
        zorder=5,
    )

    # Highlight Recommended Mesh
    ax3.scatter(
        [rec.solve_time_ms], [max(rec.error_cl_pct, rec.error_cdi_pct)],
        color=c_emerald, marker="*", s=220, edgecolors="black", linewidth=1.5,
        label=f"Recommended Mesh (N={rec.n_panels}, {rec.solve_time_ms:.1f}ms)",
        zorder=6,
    )

    # Tolerance line
    ax3.axhline(tol, color=c_red, linestyle="--", linewidth=1.2, label=f"Tolerance ({tol:.2f}%)")

    # Speedup annotation
    spd = ref.solve_time_ms / max(rec.solve_time_ms, 1e-6)
    ax3.annotate(
        f"Speedup: {spd:.1f}x\nTotal Panels: {rec.total_panels}",
        xy=(rec.solve_time_ms, max(rec.error_cl_pct, rec.error_cdi_pct)),
        xytext=(rec.solve_time_ms * 1.25, max(rec.error_cl_pct, rec.error_cdi_pct) * 1.5 + 0.05),
        arrowprops=dict(facecolor="black", arrowstyle="->", shrinkA=4, shrinkB=4),
        fontweight="bold", fontsize=9,
    )

    ax3.set_xlabel("Solve Latency [ms]", fontweight="bold")
    ax3.set_ylabel("Max Aerodynamic Error [%]", fontweight="bold")
    ax3.set_title("Pareto Frontier: Accuracy vs Computational Speed", fontweight="bold")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper right", fontsize=8)

    # -------------------------------------------------------------------------
    # Panel 4: Spanwise Circulation Distribution Γ(y) Fidelity
    # -------------------------------------------------------------------------
    ax4 = axes[1, 1]
    # Identify coarse point, minimal, recommended, and reference
    pts_sorted_all = sorted(study.points, key=lambda p: p.total_panels)
    coarse_pt = pts_sorted_all[0]

    # Plot on primary surface (surface 0)
    sw_ref = ref.result.spanwise[0] if ref.result else None
    sw_rec = rec.result.spanwise[0] if rec.result else None
    sw_min = min_pt.result.spanwise[0] if min_pt.result else None
    sw_coarse = coarse_pt.result.spanwise[0] if coarse_pt.result else None

    if sw_ref is not None:
        ax4.plot(sw_ref.y, sw_ref.gamma, "k-", linewidth=2.0, alpha=0.85, label=f"Reference (N={ref.n_panels})")
    if sw_coarse is not None:
        ax4.plot(sw_coarse.y, sw_coarse.gamma, "r--", linewidth=1.2, alpha=0.7, label=f"Coarse (N={coarse_pt.n_panels}, {coarse_pt.error_cl_pct:.2f}% err)")
    if sw_min is not None and sw_min is not sw_coarse:
        ax4.plot(sw_min.y, sw_min.gamma, ":", color=c_amber, linewidth=1.5, alpha=0.8, label=f"Minimal (N={min_pt.n_panels}, {min_pt.error_cl_pct:.2f}% err)")
    if sw_rec is not None:
        ax4.plot(sw_rec.y, sw_rec.gamma, "o-", color=c_emerald, linewidth=1.6, markersize=3, label=f"Recommended (N={rec.n_panels}, {rec.error_cl_pct:.2f}% err)")

    if sw_ref is not None:
        g_max = float(np.max(sw_ref.gamma))
        g_min = float(min(0.0, np.min(sw_ref.gamma)))
        margin = max(0.12 * abs(g_max), 0.5)
        ax4.set_ylim(g_min - 0.05 * margin, g_max + margin)

    ax4.set_xlabel("Spanwise Position y [m]", fontweight="bold")
    ax4.set_ylabel("Circulation Γ [m²/s]", fontweight="bold")
    ax4.set_title("Spanwise Circulation Γ(y) Profile Fidelity", fontweight="bold")
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc="lower center", fontsize=8)

    # Overall figure title
    fig.suptitle(
        f"Ventorum Mesh Convergence Study - {study.case_name}\n"
        f"Recommended: N={rec.n_panels} {rec.spacing} ({rec.total_panels} panels, err={max(rec.error_cl_pct, rec.error_cdi_pct):.3f}%, {rec.solve_time_ms:.2f}ms)",
        fontsize=13, fontweight="bold", y=0.98,
    )

    if save_path:
        out_p = Path(save_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    return fig
