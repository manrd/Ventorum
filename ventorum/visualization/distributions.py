# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Plot the spanwise distributions of a solution."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ventorum.core.datatypes import SolverResult
from ventorum.core.constants import RHO_SL


def plot_lift_distribution(
    result: SolverResult,
    show_elliptic: bool = True,
    ax: plt.Axes | None = None,
) -> Figure:
    """Plot normalised circulation Γ(y)/Γ_max vs spanwise station.

    Parameters
    ----------
    result : SolverResult
    show_elliptic : bool
        If *True*, overlay the ideal elliptic distribution.
    ax : plt.Axes or None
        Axes to plot on.  If *None*, a new figure is created.

    Returns
    -------
    Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    for sw in result.spanwise:
        y_norm = sw.y / np.max(np.abs(sw.y)) if np.max(np.abs(sw.y)) > 0 else sw.y
        gamma_norm = sw.gamma / np.max(np.abs(sw.gamma)) if np.max(np.abs(sw.gamma)) > 0 else sw.gamma
        ax.plot(y_norm, gamma_norm, "o-", markersize=2, label=sw.surface_name)

    if show_elliptic:
        y_ell = np.linspace(-1, 1, 200)
        gamma_ell = np.sqrt(1.0 - y_ell ** 2)
        ax.plot(y_ell, gamma_ell, "k--", alpha=0.5, label="Elliptic (ideal)")

    ax.set_xlabel("y / (b/2)")
    ax.set_ylabel("Γ / Γ_max")
    ax.set_title("Spanwise Lift (Circulation) Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_cl_distribution(
    result: SolverResult,
    ax: plt.Axes | None = None,
) -> Figure:
    """Plot local Cl vs spanwise station."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    for sw in result.spanwise:
        y_norm = sw.y / np.max(np.abs(sw.y)) if np.max(np.abs(sw.y)) > 0 else sw.y
        ax.plot(y_norm, sw.Cl, "o-", markersize=2, label=sw.surface_name)

    ax.set_xlabel("y / (b/2)")
    ax.set_ylabel("Cl")
    ax.set_title("Spanwise Cl Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_alpha_distribution(
    result: SolverResult,
    ax: plt.Axes | None = None,
) -> Figure:
    """Plot effective and induced angles of attack along the span."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    for sw in result.spanwise:
        y_norm = sw.y / np.max(np.abs(sw.y)) if np.max(np.abs(sw.y)) > 0 else sw.y
        ax.plot(y_norm, np.degrees(sw.alpha_eff), "o-", markersize=2,
                label=f"{sw.surface_name} α_eff")
        ax.plot(y_norm, np.degrees(sw.alpha_i), "s--", markersize=2,
                label=f"{sw.surface_name} α_i")

    ax.set_xlabel("y / (b/2)")
    ax.set_ylabel("Angle [°]")
    ax.set_title("Spanwise Angle-of-Attack Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_induced_drag_distribution(
    result: SolverResult,
    ax: plt.Axes | None = None,
) -> Figure:
    """Plot local Cd_i vs spanwise station."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    for sw in result.spanwise:
        y_norm = sw.y / np.max(np.abs(sw.y)) if np.max(np.abs(sw.y)) > 0 else sw.y
        ax.plot(y_norm, sw.Cd_i, "o-", markersize=2, label=sw.surface_name)

    ax.set_xlabel("y / (b/2)")
    ax.set_ylabel("Cd_i")
    ax.set_title("Spanwise Induced Drag Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_all_distributions(result: SolverResult) -> Figure:
    """Create a 2×2 subplot of all spanwise distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    plot_lift_distribution(result, ax=axes[0, 0])
    plot_cl_distribution(result, ax=axes[0, 1])
    plot_alpha_distribution(result, ax=axes[1, 0])
    plot_induced_drag_distribution(result, ax=axes[1, 1])

    fig.suptitle(
        f"LLT Results - CL={result.totals.CL:.4f}  CDi={result.totals.CDi:.6f}  "
        f"e={result.totals.e:.4f}",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_fourier_spectrum(
    result: SolverResult,
    n_harmonics: int = 15,
    surface_index: int = 0,
    ax: plt.Axes | None = None,
) -> Figure:
    """Plot the Fourier sine series circulation spectrum (A_n coefficients).

    In the classical Lanchester–Prandtl lifting-line theory, the circulation is expanded as:
    Γ(θ) = 2·b·V_∞ · Σ A_n sin(n·θ), where θ = arccos(-2y/b).
    A_1 produces the equivalent elliptic lift, while higher odd harmonics
    (A_3, A_5, ...) represent departures from elliptic loading and produce
    induced drag penalty factor δ = Σ n·(A_n / A_1)².

    Parameters
    ----------
    result : SolverResult
    n_harmonics : int
        Number of Fourier harmonics to display (default: 15).
    surface_index : int
        Index of the surface to analyze (default: 0, primary wing).
    ax : plt.Axes or None

    Returns
    -------
    Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5.5))
    else:
        fig = ax.figure

    if surface_index >= len(result.spanwise):
        raise IndexError(f"Surface index {surface_index} out of range ({len(result.spanwise)} available).")

    sw = result.spanwise[surface_index]
    b = 2.0 * float(np.max(np.abs(sw.y)))
    if b <= 0:
        b = 2.0

    cond = getattr(result, "condition", None)
    V_inf = cond.V_inf if cond is not None else 25.0

    # Map spanwise station y to Trefftz circle angle theta in [0, pi]
    eta = np.clip(sw.y / (b / 2.0), -1.0, 1.0)
    theta = np.arccos(-eta)
    gamma_star = sw.gamma / (2.0 * b * V_inf)

    # Sort theta in ascending order for numerical integration
    sort_idx = np.argsort(theta)
    th_sorted = theta[sort_idx]
    gs_sorted = gamma_star[sort_idx]

    _trap = getattr(np, "trapezoid", getattr(np, "trapz", None))
    harmonics = np.arange(1, n_harmonics + 1)
    A_coeffs = np.zeros(n_harmonics)

    for i, n in enumerate(harmonics):
        # Orthogonal sine projection: A_n = (2/pi) * integral_0^pi gamma*(theta) * sin(n*theta) dtheta
        sin_basis = np.sin(n * th_sorted)
        A_coeffs[i] = (2.0 / np.pi) * float(_trap(gs_sorted * sin_basis, th_sorted))

    A1 = A_coeffs[0]
    A1_safe = A1 if abs(A1) > 1e-12 else 1e-12
    rel_coeffs = A_coeffs / A1_safe

    # Induced drag factor delta = sum_{n=2}^N n * (A_n / A_1)^2
    delta = float(np.sum([n * (rel_coeffs[n - 1] ** 2) for n in range(2, n_harmonics + 1)]))
    e_fourier = 1.0 / (1.0 + delta) if delta >= 0 else 1.0

    # Plot bars: A1 in primary blue, higher odd harmonics in amber/red, even harmonics in gray
    colors = []
    for n in harmonics:
        if n == 1:
            colors.append("#2563eb")  # Primary blue
        elif n % 2 != 0:
            colors.append("#dc2626")  # Odd harmonics (drag penalty)
        else:
            colors.append("#94a3b8")  # Even harmonics (asymmetry)

    ax.bar(harmonics, np.abs(rel_coeffs) * 100.0, color=colors, width=0.6, edgecolor="#1e293b", alpha=0.9)
    ax.set_xticks(harmonics)
    ax.set_xlabel("Harmonic Number $n$", fontsize=10, fontweight="bold")
    ax.set_ylabel(r"Relative Amplitude $|A_n / A_1|$ [%]", fontsize=10, fontweight="bold")

    stat_box = (
        f"Fundamental $A_1$: {A1:.5f}\n"
        f"Elliptic Deviation $\\delta$: {delta:.4f}\n"
        f"Fourier Span Efficiency $e$: {e_fourier:.4f}\n"
        f"Solver Oswald $e$: {result.totals.e:.4f}"
    )
    ax.text(
        0.72,
        0.95,
        stat_box,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.9),
    )

    ax.set_title(
        f"Lifting-line Fourier circulation spectrum - {sw.surface_name}\n"
        f"($A_1$: Elliptic Lift, $A_3, A_5, \\dots$: Non-Elliptic Distortion & Drag)",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, linestyle="--", alpha=0.35, axis="y")
    fig.tight_layout()
    return fig


def plot_component_breakdown(
    result: SolverResult,
    ax: plt.Axes | None = None,
) -> Figure:
    """Plot multi-surface aerodynamic force breakdown (Lift, Induced Drag, Total Drag).

    Compares the relative aerodynamic contributions of each lifting surface
    (e.g., Main Wing vs V-Tail / Horizontal Tail / Canard).

    Parameters
    ----------
    result : SolverResult
    ax : plt.Axes or None

    Returns
    -------
    Figure
    """
    if ax is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    else:
        fig = ax.figure
        ax1 = ax
        ax2 = None

    names = [sw.surface_name for sw in result.spanwise]
    n_surfs = len(names)
    indices = np.arange(n_surfs)

    # Calculate dimensional forces per surface
    cond = getattr(result, "condition", None)
    rho = cond.rho if cond is not None else RHO_SL
    V_inf = cond.V_inf if cond is not None else 25.0
    q_inf = 0.5 * rho * (V_inf ** 2)

    lifts = []
    cdis = []
    cd_tots = []
    _trap = getattr(np, "trapezoid", getattr(np, "trapz", None))

    for sw in result.spanwise:
        # Sort by y for trapezoidal integration
        sort_y = np.argsort(sw.y)
        y_s = sw.y[sort_y]
        L_prime = sw.local_lift[sort_y] if len(sw.local_lift) > 0 else (rho * V_inf * sw.gamma[sort_y])
        Di_prime = np.abs(L_prime * sw.alpha_i[sort_y])

        L_surf = float(_trap(L_prime, y_s))
        Di_surf = float(_trap(Di_prime, y_s))
        lifts.append(L_surf)
        cdis.append(Di_surf)

        if sw.Cd_profile is not None:
            safe_cl = np.where(np.abs(sw.Cl[sort_y]) > 1e-4, np.abs(sw.Cl[sort_y]), 1.0)
            c_local = np.where(np.abs(sw.Cl[sort_y]) > 1e-4, 2.0 * sw.gamma[sort_y] / (V_inf * safe_cl), 0.1)
            D_prof_prime = q_inf * c_local * sw.Cd_profile[sort_y]
            cd_tots.append(Di_surf + float(_trap(D_prof_prime, y_s)))
        else:
            cd_tots.append(Di_surf)

    lifts = np.array(lifts)
    cdis = np.array(cdis)
    cd_tots = np.array(cd_tots)
    total_lift = np.sum(lifts)
    total_drag = np.sum(cd_tots)

    lift_pcts = (lifts / total_lift * 100.0) if abs(total_lift) > 1e-9 else np.zeros_like(lifts)
    drag_pcts = (cd_tots / total_drag * 100.0) if abs(total_drag) > 1e-9 else np.zeros_like(cd_tots)

    # Subplot 1: Force Values
    width = 0.35
    ax1.bar(indices - width / 2, lifts, width=width, label="Lift $L$ [N]", color="#2563eb", edgecolor="#1e293b")
    ax1.bar(indices + width / 2, cd_tots, width=width, label="Total Drag $D$ [N]", color="#dc2626", edgecolor="#1e293b")
    ax1.set_xticks(indices)
    ax1.set_xticklabels(names, fontsize=10, fontweight="bold")
    ax1.set_ylabel("Aerodynamic Force [N]", fontsize=10, fontweight="bold")
    ax1.set_title("Surface Forces [N]", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.35, axis="y")
    ax1.legend(loc="upper right", fontsize=9)

    # Subplot 2: Percentage Share
    if ax2 is not None:
        ax2.bar(indices - width / 2, lift_pcts, width=width, label="Lift Share [%]", color="#3b82f6", edgecolor="#1e293b")
        ax2.bar(indices + width / 2, drag_pcts, width=width, label="Drag Share [%]", color="#ef4444", edgecolor="#1e293b")
        ax2.set_xticks(indices)
        ax2.set_xticklabels(names, fontsize=10, fontweight="bold")
        ax2.set_ylabel("Share of Total Force [%]", fontsize=10, fontweight="bold")
        ax2.set_title("Relative Contribution [%]", fontsize=11, fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.35, axis="y")
        ax2.legend(loc="upper right", fontsize=9)

    fig.suptitle("Multi-Surface Aerodynamic Force & Share Breakdown", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig

