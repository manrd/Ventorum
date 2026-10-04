# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Plot the performance, the efficiency (L/D), the pitching moment and the section polars."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ventorum.core.datatypes import (
    SolverResult,
    TabulatedAirfoil,
    LinearAirfoil,
)


def plot_efficiency_curves(
    results: list[SolverResult],
    alpha_range: np.ndarray | None = None,
    ax: plt.Axes | None = None,
) -> Figure:
    """Plot aerodynamic glide efficiency (L/D) and propeller UAV endurance parameter (CL^1.5 / CD).

    Parameters
    ----------
    results : list[SolverResult]
        Sweep results across angles of attack.
    alpha_range : np.ndarray or None
        Angles of attack in radians.
    ax : plt.Axes or None

    Returns
    -------
    Figure
    """
    if ax is None:
        fig, ax1 = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
        ax1 = ax

    CLs = np.array([r.totals.CL for r in results])
    # Determine drag array (total drag if available, else induced drag)
    if results[0].totals.CD_total is not None:
        CDs = np.array([r.totals.CD_total for r in results])
        drag_label = "$C_D$ (Total)"
    else:
        CDs = np.array([r.totals.CDi for r in results])
        drag_label = "$C_{Di}$ (Induced)"

    safe_CDs = np.maximum(CDs, 1e-6)
    L_over_D = CLs / safe_CDs

    # Propeller endurance parameter: CL^1.5 / CD (for positive CL)
    pos_mask = CLs > 0
    endurance_param = np.zeros_like(CLs)
    endurance_param[pos_mask] = (CLs[pos_mask] ** 1.5) / safe_CDs[pos_mask]

    if alpha_range is not None:
        alphas_deg = np.degrees(alpha_range)
    else:
        alphas_deg = np.arange(len(CLs))

    # Primary axis: L/D
    color_ld = "#2563eb"
    line1 = ax1.plot(
        alphas_deg, L_over_D, "o-", color=color_ld, linewidth=2.2, label="Aerodynamic Efficiency ($L/D$)"
    )
    ax1.set_xlabel(r"Angle of Attack $\alpha$ [°]", fontsize=10, fontweight="bold")
    ax1.set_ylabel(r"Glide Ratio $L/D$", color=color_ld, fontsize=10, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color_ld)
    ax1.grid(True, linestyle="--", alpha=0.35)

    # Annotate max L/D
    if len(L_over_D) > 0:
        max_idx = int(np.argmax(L_over_D))
        max_ld = L_over_D[max_idx]
        alpha_max_ld = alphas_deg[max_idx]
        cl_at_max = CLs[max_idx]
        ax1.annotate(
            f"$(L/D)_{{max}} = {max_ld:.1f}$\nat $\\alpha = {alpha_max_ld:.1f}°$\n($C_L = {cl_at_max:.2f}$)",
            xy=(alpha_max_ld, max_ld),
            xytext=(alpha_max_ld + 1.5, max_ld * 0.82),
            arrowprops=dict(arrowstyle="->", color=color_ld, lw=1.5),
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#eff6ff", edgecolor=color_ld, alpha=0.9),
        )

    # Secondary axis: Endurance parameter CL^1.5 / CD
    ax2 = ax1.twinx()
    color_end = "#059669"
    line2 = ax2.plot(
        alphas_deg,
        endurance_param,
        "s--",
        color=color_end,
        linewidth=1.8,
        label=r"Endurance Factor ($C_L^{1.5}/C_D$)",
    )
    ax2.set_ylabel(
        r"Endurance Factor $C_L^{1.5} / C_D$", color=color_end, fontsize=10, fontweight="bold"
    )
    ax2.tick_params(axis="y", labelcolor=color_end)

    if np.any(pos_mask):
        end_idx = int(np.argmax(endurance_param))
        max_end = endurance_param[end_idx]
        alpha_end = alphas_deg[end_idx]
        ax2.annotate(
            f"$(C_L^{{1.5}}/C_D)_{{max}} = {max_end:.1f}$\nat $\\alpha = {alpha_end:.1f}°$",
            xy=(alpha_end, max_end),
            xytext=(alpha_end + 1.5, max_end * 0.85),
            arrowprops=dict(arrowstyle="->", color=color_end, lw=1.5),
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#ecfdf5", edgecolor=color_end, alpha=0.9),
        )

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", fontsize=9)

    ax1.set_title(
        f"Aerodynamic Efficiency & Flight Endurance Curves (Based on {drag_label})",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def plot_pitching_moment(
    results: list[SolverResult],
    alpha_range: np.ndarray | None = None,
    axes: tuple[plt.Axes, plt.Axes] | None = None,
) -> Figure:
    """Plot longitudinal pitching moment stability (Cm vs alpha and Cm vs CL).

    Calculates pitch stiffness (dCm/dalpha), static margin (SM = -dCm/dCL), and trim angle.

    Parameters
    ----------
    results : list[SolverResult]
    alpha_range : np.ndarray or None
    axes : tuple[plt.Axes, plt.Axes] or None
        Optional (ax1, ax2) to draw upon.

    Returns
    -------
    Figure
    """
    if axes is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    else:
        ax1, ax2 = axes
        fig = ax1.figure

    CLs = np.array([r.totals.CL for r in results])
    Cms = np.array([r.totals.Cm for r in results])

    if alpha_range is not None:
        alphas_deg = np.degrees(alpha_range)
        alphas_rad = alpha_range
    else:
        alphas_deg = np.arange(len(CLs))
        alphas_rad = np.radians(alphas_deg)

    # 1. Cm vs Alpha
    ax1.plot(alphas_deg, Cms, "o-", color="#dc2626", linewidth=2.0)
    ax1.axhline(0, color="gray", linestyle="--", alpha=0.6)
    ax1.set_xlabel(r"Angle of Attack $\alpha$ [°]", fontsize=10, fontweight="bold")
    ax1.set_ylabel(r"Pitching Moment Coefficient $C_m$", fontsize=10, fontweight="bold")
    ax1.set_title(r"Longitudinal Pitch Stiffness ($C_m$ vs $\alpha$)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.35)

    # Linear slope dCm/dalpha and trim angle
    if len(Cms) >= 3:
        mid = len(Cms) // 2
        lo, hi = max(0, mid - 2), min(len(Cms), mid + 3)
        if hi - lo >= 2:
            poly_alpha = np.polyfit(alphas_rad[lo:hi], Cms[lo:hi], 1)
            cm_alpha = poly_alpha[0]
            cm0 = poly_alpha[1]
            alpha_trim_deg = np.degrees(-cm0 / cm_alpha) if abs(cm_alpha) > 1e-6 else np.nan

            is_stable = cm_alpha < 0
            stability_str = "STABLE ($C_{m_\\alpha} < 0$)" if is_stable else "UNSTABLE ($C_{m_\\alpha} > 0$)"

            stat_txt = (
                f"$C_{{m_\\alpha}} = {cm_alpha:.3f}$ /rad ({cm_alpha * np.pi / 180:.4f} /°)\n"
                f"Status: {stability_str}\n"
                f"Trim $\\alpha_{{trim}} \\approx {alpha_trim_deg:.1f}°$"
            )
            ax1.text(
                0.04,
                0.06,
                stat_txt,
                transform=ax1.transAxes,
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff1f2", edgecolor="#dc2626", alpha=0.85),
            )

    # 2. Cm vs CL
    ax2.plot(CLs, Cms, "s-", color="#7c3aed", linewidth=2.0)
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.6)
    ax2.set_xlabel(r"Total Lift Coefficient $C_L$", fontsize=10, fontweight="bold")
    ax2.set_ylabel(r"Pitching Moment Coefficient $C_m$", fontsize=10, fontweight="bold")
    ax2.set_title(r"Static Margin & Neutral Point ($C_m$ vs $C_L$)", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.35)

    if len(Cms) >= 3:
        mid = len(Cms) // 2
        lo, hi = max(0, mid - 2), min(len(Cms), mid + 3)
        if hi - lo >= 2:
            poly_cl = np.polyfit(CLs[lo:hi], Cms[lo:hi], 1)
            dCm_dCL = poly_cl[0]
            static_margin = -dCm_dCL

            sm_txt = (
                f"$dC_m/dC_L = {dCm_dCL:.3f}$\n"
                f"Static Margin (SM) = {static_margin * 100:.1f}%\n"
                f"Longitudinal Balance: {'Stable' if static_margin > 0 else 'Unstable'}"
            )
            ax2.text(
                0.04,
                0.06,
                sm_txt,
                transform=ax2.transAxes,
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#f5f3ff", edgecolor="#7c3aed", alpha=0.85),
            )

    fig.suptitle("Aircraft Pitching Moment & Longitudinal Stability Analysis", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_airfoil_polar(
    airfoil: TabulatedAirfoil | LinearAirfoil,
    alpha_deg_range: np.ndarray | None = None,
) -> Figure:
    """Plot 2D section aerodynamic polar curves (Cl-alpha, Cd-alpha, Cl-Cd, Cm-alpha).

    Parameters
    ----------
    airfoil : TabulatedAirfoil or LinearAirfoil
    alpha_deg_range : np.ndarray or None
        Range of angles in degrees to evaluate.

    Returns
    -------
    Figure
    """
    if alpha_deg_range is not None:
        a_deg = alpha_deg_range
    elif isinstance(airfoil, TabulatedAirfoil) and len(airfoil.alpha) > 0:
        a_deg = np.degrees(airfoil.tables()[0])  # ascending (the attribute keeps the given order)
    else:
        a_deg = np.linspace(-6.0, 18.0, 100)

    a_rad = np.radians(a_deg)
    Cls = airfoil.Cl(a_rad)
    Cds = airfoil.Cd(a_rad)
    Cms = airfoil.Cm(a_rad)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # 1. Cl vs alpha
    ax = axes[0, 0]
    ax.plot(a_deg, Cls, color="#2563eb", linewidth=2.0)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.4)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel(r"$\alpha$ [°]", fontsize=10, fontweight="bold")
    ax.set_ylabel(r"$C_l$", fontsize=10, fontweight="bold")
    ax.set_title(f"Section Lift Curve ($a_0 \\approx {airfoil.a0:.2f}$ /rad, $\\alpha_{{L0}} \\approx {np.degrees(airfoil.alpha_L0):.1f}°$)", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)

    # 2. Cd vs alpha
    ax = axes[0, 1]
    ax.plot(a_deg, Cds, color="#dc2626", linewidth=2.0)
    ax.set_xlabel(r"$\alpha$ [°]", fontsize=10, fontweight="bold")
    ax.set_ylabel(r"$C_d$", fontsize=10, fontweight="bold")
    ax.set_title("Section Profile Drag Rise", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)

    # 3. Section Drag Polar (Cl vs Cd)
    ax = axes[1, 0]
    ax.plot(Cds, Cls, color="#059669", linewidth=2.0)
    ax.set_xlabel(r"$C_d$", fontsize=10, fontweight="bold")
    ax.set_ylabel(r"$C_l$", fontsize=10, fontweight="bold")
    ax.set_title("Airfoil Drag Polar ($C_l$ vs $C_d$)", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)

    # 4. Cm vs alpha
    ax = axes[1, 1]
    ax.plot(a_deg, Cms, color="#7c3aed", linewidth=2.0)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel(r"$\alpha$ [°]", fontsize=10, fontweight="bold")
    ax.set_ylabel(r"$C_m$", fontsize=10, fontweight="bold")
    ax.set_title("Quarter-Chord Pitching Moment ($C_m$ vs $\\alpha$)", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)

    fig.suptitle(f"2D Airfoil Characteristics - {airfoil.name}", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig
