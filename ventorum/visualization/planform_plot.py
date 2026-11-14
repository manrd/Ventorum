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
        main_surf, y_mac, x_mac_le = _main_surface_mac_geometry(aircraft)
        c_mac = aircraft.c_ref

        # Draw MAC line on main surface side
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


def _main_surface_mac_geometry(aircraft: Aircraft) -> tuple[LiftingSurface, float, float]:
    """Return the main surface, its MAC spanwise station y_mac [m] and leading-edge x_mac_le [m]."""
    from ventorum.geometry.lattice import surface_reference_line

    main_surf = aircraft.surfaces[aircraft.main_surface_index()]
    secs = sorted(main_surf.sections, key=lambda s: s.y_frac)
    eta = np.array([s.y_frac for s in secs], dtype=float)
    chords = np.array([s.chord for s in secs], dtype=float)
    x_le, y, _ = surface_reference_line(main_surf, eta)
    dy = np.abs(np.diff(y))
    if float(np.sum(dy)) <= 1e-9 * max(main_surf.semi_span, 1e-12):
        dy = np.diff(eta) * main_surf.semi_span
    y_abs = y + float(main_surf.position[1])
    x_abs = x_le + float(main_surf.position[0])
    y0, y1 = y_abs[:-1], y_abs[1:]
    x0, x1 = x_abs[:-1], x_abs[1:]
    c0, c1 = chords[:-1], chords[1:]
    area_half = float(np.sum(0.5 * (c0 + c1) * dy))
    if area_half > 0.0:
        y_mac = float(np.sum((dy / 6.0) * (y0 * (2.0 * c0 + c1) + y1 * (c0 + 2.0 * c1)))) / area_half
        x_mac_le = float(np.sum((dy / 6.0) * (x0 * (2.0 * c0 + c1) + x1 * (c0 + 2.0 * c1)))) / area_half
    else:
        y_mac = float(y_abs[0])
        x_mac_le = float(x_abs[0])
    if main_surf.mirror_y and not main_surf.is_symmetric:
        y_mac = -y_mac
    return main_surf, y_mac, x_mac_le


def _span_loading_surface_stats(
    aircraft: Aircraft,
    result: SolverResult,
    rho: float,
    v_inf: float,
) -> tuple[float, list[float], float]:
    """Return total lift [N], per-surface root bending moments [N*m] and main-surface y_cp [m]."""
    lattice = (result.details or {}).get("lattice")
    total_lift_force = 0.0
    surface_moments: list[float] = []
    half_lifts: list[float] = []

    for idx, sw in enumerate(result.spanwise):
        surf = aircraft.surfaces[idx] if idx < len(aircraft.surfaces) else aircraft.surfaces[-1]
        sym_factor = 2.0 if surf.is_symmetric else 1.0
        L_prime = rho * v_inf * np.asarray(sw.gamma, dtype=float)
        y_arr = np.asarray(sw.y, dtype=float)
        y_root = float(surf.position[1])

        if lattice is not None and idx < len(lattice.surfaces):
            sl = lattice.surfaces[idx].strips
            w_all = np.asarray(lattice.width[sl], dtype=float)
            if surf.is_symmetric:
                right = np.asarray(lattice.strip_is_right[sl], dtype=bool)
                w_half = w_all[right]
                L_half = L_prime[right]
                y_half = y_arr[right]
            else:
                w_half = w_all
                L_half = L_prime
                y_half = y_arr
        elif (
            getattr(result, "solver_type", None) == "fourier"
            or getattr(result, "fourier_coefficients", None) is not None
        ):
            b = 2.0 * surf.semi_span if surf.is_symmetric else surf.semi_span
            y_s = np.sort(y_arr)
            y_min = -0.5 * b if surf.is_symmetric else y_root
            y_max = 0.5 * b if surf.is_symmetric else (y_root + surf.semi_span)
            edges = np.concatenate([[y_min], 0.5 * (y_s[:-1] + y_s[1:]), [y_max]])
            if surf.is_symmetric:
                w_half = np.maximum(y_root, edges[1:]) - np.maximum(y_root, edges[:-1])
                right = w_half > 0.0
                w_half = w_half[right]
                L_half = L_prime[right]
                y_half = y_arr[right]
            else:
                w_half = np.diff(edges)
                L_half = L_prime
                y_half = y_arr
        else:
            if surf.is_symmetric and np.any(y_arr < y_root):
                right = y_arr >= y_root
            else:
                right = np.ones(len(y_arr), dtype=bool)
            L_half = L_prime[right]
            y_half = y_arr[right]
            n_half = max(1, int(np.sum(right)))
            ds = discretize_surface(
                surf,
                n_panels=n_half,
                spacing=surf.spacing or "auto",
                surface_index=idx,
                half_mesh=True,
            )
            w_half = np.asarray(ds.dy_panels, dtype=float)

        l_half_sum = float(np.sum(L_half * w_half))
        m_surf = float(np.sum(np.abs(y_half - y_root) * L_half * w_half))
        total_lift_force += sym_factor * l_half_sum
        surface_moments.append(m_surf)
        half_lifts.append(l_half_sum)

    main_idx = aircraft.main_surface_index()
    if main_idx < len(surface_moments) and half_lifts[main_idx] > 0.0:
        y_cp = surface_moments[main_idx] / half_lifts[main_idx]
    else:
        y_cp = 0.0
    return total_lift_force, surface_moments, y_cp


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

    # Plot each surface
    for sw in result.spanwise:
        # L'(y) = rho * V_inf * gamma
        L_prime = rho * V_inf * sw.gamma
        ax.plot(sw.y, L_prime, "o-", markersize=3, linewidth=2, label=f"{sw.surface_name} $L'(y)$")

    total_lift_force, surface_moments, y_cp = _span_loading_surface_stats(aircraft, result, rho, V_inf)

    # Equivalent elliptic loading: L_ell(y) = (4 * L_tot) / (pi * b) * sqrt(1 - (2y/b)^2)
    if show_elliptic and total_lift_force > 0:
        y_ell = np.linspace(-b_ref / 2.0, b_ref / 2.0, 250)
        y_norm = np.clip(2.0 * y_ell / b_ref, -1.0, 1.0)
        L_ell = (4.0 * total_lift_force / (np.pi * b_ref)) * np.sqrt(np.maximum(0.0, 1.0 - y_norm**2))
        ax.plot(y_ell, L_ell, "k--", linewidth=1.8, alpha=0.75, label="Elliptic (Ideal Min Drag)")
        ax.fill_between(y_ell, L_ell, color="gray", alpha=0.10)

    ax.set_xlabel("Spanwise Station Y [m]", fontsize=10, fontweight="bold")
    ax.set_ylabel("Sectional Lift Loading $L'(y)$ [N/m]", fontsize=10, fontweight="bold")

    if len(surface_moments) == 1:
        moment_lines = f"Root Bending Moment ($M_{{root}}$): {surface_moments[0]:.1f} N·m"
    else:
        moment_lines = "\n".join(
            f"Root Bending Moment ($M_{{root}}$, {sw.surface_name}): {m:.1f} N·m"
            for sw, m in zip(result.spanwise, surface_moments)
        )
    stat_box = (
        f"Total Lift ($L$): {total_lift_force:.1f} N\n"
        f"{moment_lines}\n"
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
