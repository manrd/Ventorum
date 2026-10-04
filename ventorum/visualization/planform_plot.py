# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Plot the 2-D planform of an aircraft and its dimensional span loading."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.patches as patches

from ventorum.core.datatypes import Aircraft, LiftingSurface, SolverResult
from ventorum.core.constants import RHO_SL
from ventorum.geometry.processing import discretize_surface


def plot_planform_2d(
    aircraft: Aircraft | LiftingSurface,
    show_mac: bool = True,
    show_quarter_chord: bool = True,
    show_dimensions: bool = True,
    ax: plt.Axes | None = None,
) -> Figure:
    """Plot an engineering planform blueprint (X-Y top-down view) with dimensions.

    Parameters
    ----------
    aircraft : Aircraft or LiftingSurface
    show_mac : bool
        If True, annotates the Mean Aerodynamic Chord (MAC) position and length.
    show_quarter_chord : bool
        If True, traces the 1/4-chord aerodynamic line.
    show_dimensions : bool
        If True, displays annotations for wingspan, chords, area, and aspect ratio.
    ax : plt.Axes or None

    Returns
    -------
    Figure
    """
    if isinstance(aircraft, LiftingSurface):
        aircraft = Aircraft(name=aircraft.name, surfaces=[aircraft])

    aircraft.compute_reference_values()

    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 7))
    else:
        fig = ax.figure

    colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706"]

    for idx, surf in enumerate(aircraft.surfaces):
        col = colors[idx % len(colors)]
        ds = discretize_surface(surf, n_panels=surf.n_panels or 40, spacing=surf.spacing or "auto", surface_index=idx)
        n_panels = len(ds.chords)

        # Reconstruct LE and TE points
        le_pts = []
        te_pts = []
        for i in range(n_panels):
            node_A = ds.nodes_qc[i]
            node_B = ds.nodes_qc[i + 1]
            vec_c = ds.control_points[i] - 0.5 * (node_A + node_B)
            if i == 0:
                le_pts.append(node_A[:2] - 0.5 * vec_c[:2])
                te_pts.append(node_A[:2] + 1.5 * vec_c[:2])
            le_pts.append(node_B[:2] - 0.5 * vec_c[:2])
            te_pts.append(node_B[:2] + 1.5 * vec_c[:2])

        le_pts = np.array(le_pts)
        te_pts = np.array(te_pts)

        # Closed boundary polygon: LE (left to right) -> TE (right to left) -> close
        poly_pts = np.vstack([le_pts, te_pts[::-1]])
        poly = patches.Polygon(
            np.column_stack([poly_pts[:, 1], poly_pts[:, 0]]),  # Plot Y on horizontal, X on vertical
            closed=True,
            facecolor=col,
            alpha=0.15,
            edgecolor=col,
            linewidth=2.0,
            label=f"{surf.name} (Planform)",
        )
        ax.add_patch(poly)

        # Quarter-chord line
        if show_quarter_chord:
            ax.plot(
                ds.nodes_qc[:, 1],
                ds.nodes_qc[:, 0],
                "--",
                color=col,
                linewidth=1.5,
                alpha=0.8,
                label=f"{surf.name} (1/4c)",
            )

    # Annotate Mean Aerodynamic Chord (MAC)
    if show_mac and aircraft.c_ref is not None:
        main_surf = aircraft.surfaces[0]
        # Position of MAC along span for a trapezoidal half-wing: y_mac ≈ (b/6) * (1 + 2*lambda)/(1 + lambda)
        secs = sorted(main_surf.sections, key=lambda s: s.y_frac)
        cr = secs[0].chord
        ct = secs[-1].chord
        taper = ct / cr if cr > 0 else 1.0
        b_semi = main_surf.semi_span
        y_mac = (b_semi / 3.0) * (1.0 + 2.0 * taper) / (1.0 + taper)

        # Leading edge x at y_mac
        x_mac_le = main_surf.position[0] + y_mac * np.tan(main_surf.sweep_le)
        c_mac = aircraft.c_ref

        # Draw MAC line on positive Y side
        ax.plot(
            [y_mac, y_mac],
            [x_mac_le, x_mac_le + c_mac],
            color="#b91c1c",
            linewidth=2.5,
            linestyle="-",
            label=rf"MAC ($\overline{{c}}$ = {c_mac:.3f} m)",
        )
        # Symmetrical side
        if main_surf.is_symmetric:
            ax.plot(
                [-y_mac, -y_mac],
                [x_mac_le, x_mac_le + c_mac],
                color="#b91c1c",
                linewidth=2.5,
                linestyle="-",
            )
            ax.plot(
                [-y_mac, y_mac],
                [x_mac_le + 0.25 * c_mac, x_mac_le + 0.25 * c_mac],
                color="#b91c1c",
                linewidth=1.0,
                linestyle=":",
                alpha=0.6,
            )

    # Invert X axis so flow is from top (upstream) to bottom (downstream)
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Spanwise Coordinate Y [m]", fontsize=10, fontweight="bold")
    ax.set_ylabel("Chordwise Coordinate X [m]", fontsize=10, fontweight="bold")

    title_lines = [f"{aircraft.name} - 2D Planform Blueprint Layout"]
    if show_dimensions and aircraft.S_ref and aircraft.b_ref:
        AR = aircraft.b_ref**2 / aircraft.S_ref
        title_lines.append(
            rf"$S_{{ref}}$ = {aircraft.S_ref:.3f} m²,  $b$ = {aircraft.b_ref:.3f} m,  "
            rf"AR = {AR:.2f},  $\overline{{c}}$ = {aircraft.c_ref:.3f} m"
        )
    ax.set_title("\n".join(title_lines), fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig


def plot_span_loading(
    aircraft: Aircraft | LiftingSurface,
    result: SolverResult,
    show_elliptic: bool = True,
    ax: plt.Axes | None = None,
) -> Figure:
    """Plot dimensional span loading L'(y) [N/m] vs ideal elliptic load distribution.

    Calculates and displays the wing root bending moment and aerodynamic center of pressure.

    Parameters
    ----------
    aircraft : Aircraft or LiftingSurface
    result : SolverResult
    show_elliptic : bool
        If True, overlays equivalent elliptic loading producing the exact same total lift.
    ax : plt.Axes or None

    Returns
    -------
    Figure
    """
    if isinstance(aircraft, LiftingSurface):
        aircraft = Aircraft(name=aircraft.name, surfaces=[aircraft])

    aircraft.compute_reference_values()
    b_ref = aircraft.b_ref if aircraft.b_ref else 2.0
    cond = getattr(result, "condition", None)
    rho = cond.rho if cond is not None else RHO_SL
    V_inf = cond.V_inf if cond is not None else 25.0

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5.5))
    else:
        fig = ax.figure

    total_lift_force = 0.0
    root_bending_moment = 0.0

    # Plot each surface
    for sw in result.spanwise:
        # L'(y) = rho * V_inf * gamma
        L_prime = rho * V_inf * sw.gamma
        ax.plot(sw.y, L_prime, "o-", markersize=3, linewidth=2, label=f"{sw.surface_name} $L'(y)$")

        # Accumulate lift and bending moment on positive semi-span (y >= 0)
        pos_mask = sw.y >= 0
        if np.any(pos_mask):
            y_pos = sw.y[pos_mask]
            L_pos = L_prime[pos_mask]
            if len(y_pos) > 1:
                _trap = getattr(np, "trapezoid", getattr(np, "trapz", None))
                total_lift_force += 2.0 * float(_trap(L_pos, y_pos))
                root_bending_moment += float(_trap(y_pos * L_pos, y_pos))

    # Equivalent elliptic loading: L_ell(y) = (4 * L_tot) / (pi * b) * sqrt(1 - (2y/b)^2)
    if show_elliptic and total_lift_force > 0:
        y_ell = np.linspace(-b_ref / 2.0, b_ref / 2.0, 250)
        y_norm = np.clip(2.0 * y_ell / b_ref, -1.0, 1.0)
        L_ell = (4.0 * total_lift_force / (np.pi * b_ref)) * np.sqrt(np.maximum(0.0, 1.0 - y_norm**2))
        ax.plot(y_ell, L_ell, "k--", linewidth=1.8, alpha=0.75, label="Elliptic (Ideal Min Drag)")
        ax.fill_between(y_ell, L_ell, color="gray", alpha=0.10)

    # Centroid of lift on half-wing
    y_cp = (root_bending_moment / (0.5 * total_lift_force)) if total_lift_force > 0 else 0.0

    ax.set_xlabel("Spanwise Station Y [m]", fontsize=10, fontweight="bold")
    ax.set_ylabel("Sectional Lift Loading $L'(y)$ [N/m]", fontsize=10, fontweight="bold")

    stat_box = (
        f"Total Lift ($L$): {total_lift_force:.1f} N\n"
        f"Root Bending Moment ($M_{{root}}$): {root_bending_moment:.1f} N·m\n"
        f"Lift Centroid ($y_{{cp}}$): {y_cp:.3f} m ({y_cp/(b_ref/2.0)*100:.1f}% semi-span)"
    )
    ax.text(
        0.03,
        0.95,
        stat_box,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.9),
    )

    ax.set_title(
        f"Spanwise Lift Loading Distribution & Structural Wing Bending\n"
        f"($C_L$ = {result.totals.CL:.4f}, $C_{{Di}}$ = {result.totals.CDi:.5f}, $e$ = {result.totals.e:.4f})",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig
