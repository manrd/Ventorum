# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Plot a one-page aerodynamic dashboard of an aircraft.

The dashboard shows the 3-D geometry, the 2-D planform, the spanwise
loading, the flow angles, the drag polar, the aerodynamic efficiency, the
longitudinal stability and a table of the key values.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ventorum.core.datatypes import Aircraft, LiftingSurface, SolverResult, SolverSettings
from ventorum.core.constants import RHO_SL
from ventorum.geometry.processing import discretize_surface


def plot_aircraft_dashboard(
    aircraft: Aircraft | LiftingSurface,
    result: SolverResult,
    sweep_results: list[SolverResult] | None = None,
    alpha_sweep_deg: np.ndarray | None = None,
    settings: SolverSettings | None = None,
    figsize: tuple[float, float] = (22, 11),
    elev: float = 24.0,
    azim: float = -125.0,
    cmap: str = "viridis",
) -> Figure:
    """Generate a publication-grade 8-panel Executive Aerodynamic Aircraft Dashboard.

    Panels included:
    1. 3D Surface Geometry colored by local lift coefficient Cl.
    2. 2D Top-Down Planform Blueprint Layout with MAC and quarter-chord line.
    3. Dimensional Spanwise Lift Loading L'(y) [N/m] vs Elliptic and Root Bending Moment.
    4. Spanwise Angle-of-Attack Breakdown (Geometric, Induced, Effective).
    5. Aircraft Drag Polar (CL vs CD) with Parabolic Model Fit.
    6. Aerodynamic Glide Ratio (L/D) and Propeller Endurance (CL^1.5 / CD).
    7. Longitudinal Pitching Moment Stability (Cm vs alpha & Cm vs CL).
    8. Executive Reference Specifications & Performance Summary Table.

    Parameters
    ----------
    aircraft : Aircraft or LiftingSurface
    result : SolverResult
        Current operating flight point result.
    sweep_results : list[SolverResult] or None
        Optional sweep results for polars and stability curves.
    alpha_sweep_deg : np.ndarray or None
        Angles of attack corresponding to sweep_results in degrees.
    settings : SolverSettings or None
    figsize : tuple[float, float]
    elev : float
        3D camera elevation angle.
    azim : float
        3D camera azimuth angle.
    cmap : str
        Colormap for 3D surface panels.

    Returns
    -------
    Figure
    """
    if isinstance(aircraft, LiftingSurface):
        aircraft = Aircraft(name=aircraft.name, surfaces=[aircraft])

    aircraft.compute_reference_values()
    b_ref = aircraft.b_ref if aircraft.b_ref else 2.0
    S_ref = aircraft.S_ref if aircraft.S_ref else 1.0
    c_ref = aircraft.c_ref if aircraft.c_ref else 1.0
    AR = (b_ref ** 2) / S_ref if S_ref > 0 else 10.0

    cond = getattr(result, "condition", None)
    rho = cond.rho if cond is not None else RHO_SL
    V_inf = cond.V_inf if cond is not None else 25.0
    alpha_cur_deg = np.degrees(cond.alpha) if cond is not None else 5.0
    q_inf = 0.5 * rho * (V_inf ** 2)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 4, hspace=0.32, wspace=0.28)

    # ═══════════════════════════════════════════════════════════════════════════
    # PANEL 1: 3D Surface Panel Results (Cl)
    # ═══════════════════════════════════════════════════════════════════════════
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    all_polys = []
    all_vals = []
    all_pts_list = []

    for idx, surf in enumerate(aircraft.surfaces):
        if idx >= len(result.spanwise):
            continue
        sw = result.spanwise[idx]
        n_semi = (
            settings.n_panels
            if settings is not None
            else (len(sw.gamma) // 2 if surf.is_symmetric else len(sw.gamma))
        )
        spacing = settings.spacing if settings is not None else "cosine"
        ds = discretize_surface(surf, n_semi, spacing, idx)
        n_panels = min(len(ds.chords), len(sw.Cl))

        for i in range(n_panels):
            node_A = ds.nodes_qc[i]
            node_B = ds.nodes_qc[i + 1]
            vec_c = ds.control_points[i] - 0.5 * (node_A + node_B)
            p_le_A = node_A - 0.5 * vec_c
            p_le_B = node_B - 0.5 * vec_c
            p_te_B = node_B + 1.5 * vec_c
            p_te_A = node_A + 1.5 * vec_c

            poly = [p_le_A, p_le_B, p_te_B, p_te_A]
            all_polys.append(poly)
            all_vals.append(sw.Cl[i])
            all_pts_list.extend(poly)

    if all_polys:
        vals = np.array(all_vals)
        vmin = float(np.min(vals))
        vmax = float(np.max(vals))
        if abs(vmax - vmin) < 1e-4:
            vmax += 0.1
            vmin -= 0.1

        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap_obj = plt.get_cmap(cmap)
        facecolors = cmap_obj(norm(vals))

        poly_col = Poly3DCollection(
            all_polys,
            facecolors=facecolors,
            edgecolors="#1e293b",
            linewidths=0.4,
            alpha=0.92,
        )
        ax1.add_collection3d(poly_col)

        pts = np.array(all_pts_list)
        mid = pts.mean(axis=0)
        max_r = np.ptp(pts, axis=0).max() / 2.0
        ax1.set_xlim(mid[0] - max_r, mid[0] + max_r)
        ax1.set_ylim(mid[1] - max_r, mid[1] + max_r)
        ax1.set_zlim(mid[2] - max_r, mid[2] + max_r)

        sm = cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax1, shrink=0.55, pad=0.08, aspect=15)
        cb.set_label(r"$C_l$", fontsize=9, fontweight="bold")

    ax1.set_xlabel("X [m]", fontsize=8)
    ax1.set_ylabel("Y [m]", fontsize=8)
    ax1.set_zlabel("Z [m]", fontsize=8)
    ax1.tick_params(labelsize=8)
    ax1.view_init(elev=elev, azim=azim)
    ax1.set_title(r"1. 3D Surface Lift Coefficient $C_l$", fontsize=10, fontweight="bold")

    # ═══════════════════════════════════════════════════════════════════════════
    # PANEL 2: 2D Planform Blueprint & MAC
    # ═══════════════════════════════════════════════════════════════════════════
    ax2 = fig.add_subplot(gs[0, 1])
    colors_surf = ["#2563eb", "#dc2626", "#059669", "#7c3aed"]

    for idx, surf in enumerate(aircraft.surfaces):
        col = colors_surf[idx % len(colors_surf)]
        ds = discretize_surface(surf, n_panels=surf.n_panels or 35, spacing=surf.spacing or "auto", surface_index=idx)
        n_p = len(ds.chords)
        le_pts = []
        te_pts = []
        for i in range(n_p):
            nA = ds.nodes_qc[i]
            nB = ds.nodes_qc[i + 1]
            vc = ds.control_points[i] - 0.5 * (nA + nB)
            if i == 0:
                le_pts.append(nA[:2] - 0.5 * vc[:2])
                te_pts.append(nA[:2] + 1.5 * vc[:2])
            le_pts.append(nB[:2] - 0.5 * vc[:2])
            te_pts.append(nB[:2] + 1.5 * vc[:2])

        le_pts = np.array(le_pts)
        te_pts = np.array(te_pts)
        poly_pts = np.vstack([le_pts, te_pts[::-1]])
        poly = patches.Polygon(
            np.column_stack([poly_pts[:, 1], poly_pts[:, 0]]),
            closed=True,
            facecolor=col,
            alpha=0.15,
            edgecolor=col,
            linewidth=1.8,
            label=f"{surf.name}",
        )
        ax2.add_patch(poly)
        ax2.plot(ds.nodes_qc[:, 1], ds.nodes_qc[:, 0], "--", color=col, linewidth=1.2, alpha=0.8)

    # MAC line
    main_surf = aircraft.surfaces[0]
    secs = sorted(main_surf.sections, key=lambda s: s.y_frac)
    cr = secs[0].chord
    ct = secs[-1].chord
    taper = ct / cr if cr > 0 else 1.0
    b_semi = main_surf.semi_span
    y_mac = (b_semi / 3.0) * (1.0 + 2.0 * taper) / (1.0 + taper)
    x_mac_le = main_surf.position[0] + y_mac * np.tan(main_surf.sweep_le)

    ax2.plot([y_mac, y_mac], [x_mac_le, x_mac_le + c_ref], color="#b91c1c", linewidth=2.2, label=r"MAC ($\overline{c}$)")
    if main_surf.is_symmetric:
        ax2.plot([-y_mac, -y_mac], [x_mac_le, x_mac_le + c_ref], color="#b91c1c", linewidth=2.2)

    ax2.invert_yaxis()
    ax2.set_aspect("equal", adjustable="datalim")
    ax2.set_xlabel("Spanwise Y [m]", fontsize=8, fontweight="bold")
    ax2.set_ylabel("Chordwise X [m]", fontsize=8, fontweight="bold")
    ax2.tick_params(labelsize=8)
    ax2.set_title(r"2. 2D Planform Blueprint & MAC", fontsize=10, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend(loc="upper right", fontsize=7)

    # ═══════════════════════════════════════════════════════════════════════════
    # PANEL 3: Dimensional Span Loading L'(y) vs Elliptic & Root Bending Moment
    # ═══════════════════════════════════════════════════════════════════════════
    ax3 = fig.add_subplot(gs[0, 2])
    _trap = getattr(np, "trapezoid", getattr(np, "trapz", None))
    total_lift_force = 0.0
    root_bending_moment = 0.0

    for sw in result.spanwise:
        L_prime = rho * V_inf * sw.gamma
        ax3.plot(sw.y, L_prime, "o-", markersize=2.5, linewidth=1.8, label=f"{sw.surface_name}")

        pos_mask = sw.y >= 0
        if np.any(pos_mask):
            y_p = sw.y[pos_mask]
            L_p = L_prime[pos_mask]
            if len(y_p) > 1:
                total_lift_force += 2.0 * float(_trap(L_p, y_p))
                root_bending_moment += float(_trap(y_p * L_p, y_p))

    if total_lift_force > 0:
        y_ell = np.linspace(-b_ref / 2.0, b_ref / 2.0, 200)
        y_norm = np.clip(2.0 * y_ell / b_ref, -1.0, 1.0)
        L_ell = (4.0 * total_lift_force / (np.pi * b_ref)) * np.sqrt(np.maximum(0.0, 1.0 - y_norm ** 2))
        ax3.plot(y_ell, L_ell, "k--", linewidth=1.5, alpha=0.7, label="Elliptic (Ideal)")
        ax3.fill_between(y_ell, L_ell, color="gray", alpha=0.08)

    y_cp = (root_bending_moment / (0.5 * total_lift_force)) if total_lift_force > 0 else 0.0
    stat_txt3 = (
        f"$L_{{tot}}$: {total_lift_force:.1f} N\n"
        f"$M_{{root}}$: {root_bending_moment:.1f} N·m\n"
        f"$y_{{cp}}$: {y_cp:.2f} m"
    )
    ax3.text(
        0.03, 0.95, stat_txt3, transform=ax3.transAxes, verticalalignment="top", fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.85),
    )

    ax3.set_xlabel("Spanwise Y [m]", fontsize=8, fontweight="bold")
    ax3.set_ylabel(r"$L'(y)$ [N/m]", fontsize=8, fontweight="bold")
    ax3.set_title(r"3. Spanwise Lift Loading & $M_{root}$", fontsize=10, fontweight="bold")
    ax3.tick_params(labelsize=8)
    ax3.grid(True, linestyle="--", alpha=0.3)
    ax3.legend(loc="upper right", fontsize=7)

    # ═══════════════════════════════════════════════════════════════════════════
    # PANEL 4: Flowfield Angles (alpha_geo, alpha_i, alpha_eff)
    # ═══════════════════════════════════════════════════════════════════════════
    ax4 = fig.add_subplot(gs[0, 3])
    for sw in result.spanwise:
        y_n = sw.y / (b_ref / 2.0)
        ax4.plot(y_n, np.degrees(sw.alpha_eff), "o-", markersize=2.5, linewidth=1.6, label=r"$\alpha_{eff}$ (" + sw.surface_name + ")")
        ax4.plot(y_n, np.degrees(sw.alpha_i), "s--", markersize=2.0, linewidth=1.2, label=r"$\alpha_i$ (" + sw.surface_name + ")")

    ax4.set_xlabel("Normalized Station $2y/b$", fontsize=8, fontweight="bold")
    ax4.set_ylabel("Angle [°]", fontsize=8, fontweight="bold")
    ax4.set_title(r"4. Aerodynamic Angles ($\alpha_{eff}, \alpha_i$)", fontsize=10, fontweight="bold")
    ax4.tick_params(labelsize=8)
    ax4.grid(True, linestyle="--", alpha=0.3)
    ax4.legend(loc="lower right", fontsize=7)

    # ═══════════════════════════════════════════════════════════════════════════
    # PANEL 5: Drag Polar (CL vs CD) with Parabolic Fit
    # ═══════════════════════════════════════════════════════════════════════════
    ax5 = fig.add_subplot(gs[1, 0])
    if sweep_results and len(sweep_results) > 1:
        cls_sw = np.array([r.totals.CL for r in sweep_results])
        if sweep_results[0].totals.CD_total is not None:
            cds_sw = np.array([r.totals.CD_total for r in sweep_results])
            lbl_drag = r"$C_D$ (Total)"
        else:
            cds_sw = np.array([r.totals.CDi for r in sweep_results])
            lbl_drag = r"$C_{Di}$ (Induced)"

        ax5.plot(cds_sw, cls_sw, "o-", color="#2563eb", linewidth=1.8, label=lbl_drag)

        # Highlight current point
        cd_cur = result.totals.CD_total if result.totals.CD_total is not None else result.totals.CDi
        ax5.plot(cd_cur, result.totals.CL, "r*", markersize=11, label=r"Current $\alpha$")

        # Parabolic fit: CD = CD0 + CL^2 / (pi * AR * e)
        if len(cls_sw) >= 3:
            p = np.polyfit(cls_sw**2, cds_sw, 1)
            k_fit = p[0]
            cd0_fit = p[1]
            e_fit = 1.0 / (np.pi * AR * k_fit) if k_fit > 0 else 0.8
            cl_fit_line = np.linspace(np.min(cls_sw), np.max(cls_sw), 100)
            cd_fit_line = cd0_fit + k_fit * (cl_fit_line ** 2)
            ax5.plot(cd_fit_line, cl_fit_line, "k--", alpha=0.7, label=f"Fit ($e = {e_fit:.2f}$)")
    else:
        # Single point representation with analytical parabolic curve
        e_val = result.totals.e if result.totals.e else 0.85
        cd_cur = result.totals.CD_total if result.totals.CD_total is not None else result.totals.CDi
        cd0_est = max(0.005, cd_cur - (result.totals.CL**2) / (np.pi * AR * e_val))
        cl_pts = np.linspace(-0.2, max(1.4, result.totals.CL * 1.3), 100)
        cd_pts = cd0_est + (cl_pts ** 2) / (np.pi * AR * e_val)
        ax5.plot(cd_pts, cl_pts, "--", color="#2563eb", linewidth=1.8, label="Model Polar")
        ax5.plot(cd_cur, result.totals.CL, "r*", markersize=11, label=f"Point ($\\alpha={alpha_cur_deg:.1f}°$)")

    ax5.set_xlabel(r"Drag Coefficient $C_D$", fontsize=8, fontweight="bold")
    ax5.set_ylabel(r"Lift Coefficient $C_L$", fontsize=8, fontweight="bold")
    ax5.set_title(r"5. Aircraft Drag Polar ($C_L$ vs $C_D$)", fontsize=10, fontweight="bold")
    ax5.tick_params(labelsize=8)
    ax5.grid(True, linestyle="--", alpha=0.3)
    ax5.legend(loc="lower right", fontsize=7)

    # ═══════════════════════════════════════════════════════════════════════════
    # PANEL 6: Aerodynamic Efficiency & Endurance
    # ═══════════════════════════════════════════════════════════════════════════
    ax6_1 = fig.add_subplot(gs[1, 1])
    if sweep_results and len(sweep_results) > 1:
        cls_sw = np.array([r.totals.CL for r in sweep_results])
        cds_sw = np.array([
            r.totals.CD_total if r.totals.CD_total is not None else r.totals.CDi
            for r in sweep_results
        ])
        safe_cd = np.maximum(cds_sw, 1e-6)
        ld_sw = cls_sw / safe_cd
        a_deg = alpha_sweep_deg if alpha_sweep_deg is not None else np.arange(len(cls_sw))

        c_ld = "#2563eb"
        ax6_1.plot(a_deg, ld_sw, "o-", color=c_ld, linewidth=1.8, label=r"$L/D$")
        ax6_1.set_xlabel(r"$\alpha$ [°]", fontsize=8, fontweight="bold")
        ax6_1.set_ylabel(r"Glide Ratio $L/D$", color=c_ld, fontsize=8, fontweight="bold")
        ax6_1.tick_params(axis="y", labelcolor=c_ld, labelsize=8)

        # Max L/D annotation
        max_idx = int(np.argmax(ld_sw))
        ax6_1.annotate(
            f"$(L/D)_{{max}}={ld_sw[max_idx]:.1f}$\n$\\alpha={a_deg[max_idx]:.1f}°$",
            xy=(a_deg[max_idx], ld_sw[max_idx]),
            xytext=(a_deg[max_idx] + 1.0, ld_sw[max_idx] * 0.8),
            arrowprops=dict(arrowstyle="->", color=c_ld, lw=1.2),
            fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#eff6ff", edgecolor=c_ld, alpha=0.85),
        )

        # Secondary axis for endurance CL^1.5 / CD
        ax6_2 = ax6_1.twinx()
        c_end = "#059669"
        end_param = np.zeros_like(cls_sw)
        pos = cls_sw > 0
        end_param[pos] = (cls_sw[pos] ** 1.5) / safe_cd[pos]
        ax6_2.plot(a_deg, end_param, "s--", color=c_end, linewidth=1.4, label=r"$C_L^{1.5}/C_D$")
        ax6_2.set_ylabel(r"Endurance $C_L^{1.5}/C_D$", color=c_end, fontsize=8, fontweight="bold")
        ax6_2.tick_params(axis="y", labelcolor=c_end, labelsize=8)
    else:
        cd_cur = result.totals.CD_total if result.totals.CD_total is not None else result.totals.CDi
        safe_cd = max(cd_cur, 1e-6)
        ld_pt = result.totals.CL / safe_cd
        end_pt = (result.totals.CL ** 1.5) / safe_cd if result.totals.CL > 0 else 0.0

        ax6_1.bar(["$L/D$", "$C_L^{1.5}/C_D$"], [ld_pt, end_pt], color=["#2563eb", "#059669"], width=0.5)
        ax6_1.set_ylabel("Efficiency Metric Value", fontsize=8, fontweight="bold")

    ax6_1.set_title(r"6. Aerodynamic Efficiency ($L/D$)", fontsize=10, fontweight="bold")
    ax6_1.grid(True, linestyle="--", alpha=0.3)

    # ═══════════════════════════════════════════════════════════════════════════
    # PANEL 7: Longitudinal Pitching Moment Stability
    # ═══════════════════════════════════════════════════════════════════════════
    ax7 = fig.add_subplot(gs[1, 2])
    if sweep_results and len(sweep_results) > 1:
        cms = np.array([r.totals.Cm for r in sweep_results])
        a_deg = alpha_sweep_deg if alpha_sweep_deg is not None else np.arange(len(cms))
        a_rad = np.radians(a_deg)
        cls_sw = np.array([r.totals.CL for r in sweep_results])

        ax7.plot(a_deg, cms, "o-", color="#dc2626", linewidth=1.8, label=r"$C_m$ vs $\alpha$")
        ax7.axhline(0, color="gray", linestyle="--", alpha=0.5)

        if len(cms) >= 3:
            p_m = np.polyfit(a_rad, cms, 1)
            cm_alpha = p_m[0]
            cm0 = p_m[1]
            a_trim = np.degrees(-cm0 / cm_alpha) if abs(cm_alpha) > 1e-6 else np.nan

            p_cl = np.polyfit(cls_sw, cms, 1)
            sm = -p_cl[0]

            stat7 = (
                f"$C_{{m_\\alpha}}$: {cm_alpha:.3f} /rad\n"
                f"Static Margin: {sm * 100:.1f}%\n"
                f"Trim $\\alpha$: {a_trim:.1f}°"
            )
            ax7.text(
                0.04, 0.06, stat7, transform=ax7.transAxes, fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff1f2", edgecolor="#dc2626", alpha=0.85),
            )
    else:
        ax7.bar(["$C_m$"], [result.totals.Cm], color="#dc2626", width=0.4)
        ax7.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax7.set_ylabel(r"$C_m$", fontsize=8, fontweight="bold")

    ax7.set_xlabel(r"$\alpha$ [°]", fontsize=8, fontweight="bold")
    ax7.set_ylabel(r"$C_m$", fontsize=8, fontweight="bold")
    ax7.set_title(r"7. Pitching Moment & Stability", fontsize=10, fontweight="bold")
    ax7.tick_params(labelsize=8)
    ax7.grid(True, linestyle="--", alpha=0.3)

    # ═══════════════════════════════════════════════════════════════════════════
    # PANEL 8: Executive Summary Table & Aircraft KPI Card
    # ═══════════════════════════════════════════════════════════════════════════
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.axis("off")

    cd_tot_val = result.totals.CD_total if result.totals.CD_total is not None else result.totals.CDi
    ld_val = result.totals.CL / max(cd_tot_val, 1e-6)
    d_tot_force = cd_tot_val * q_inf * S_ref
    lift_force = result.totals.CL * q_inf * S_ref

    table_data = [
        ["Configuration", aircraft.name],
        ["Wingspan (b)", f"{b_ref:.3f} m"],
        ["Wing Area (S)", f"{S_ref:.3f} m²"],
        ["Aspect Ratio (AR)", f"{AR:.2f}"],
        ["MAC (c_ref)", f"{c_ref:.3f} m"],
        ["Airspeed (V_inf)", f"{V_inf:.1f} m/s"],
        ["Angle of Attack (α)", f"{alpha_cur_deg:.2f}°"],
        ["Lift Coeff (CL)", f"{result.totals.CL:.4f}"],
        ["Induced Drag (CDi)", f"{result.totals.CDi:.5f}"],
        ["Total Drag (CD)", f"{cd_tot_val:.5f}"],
        ["Glide Ratio (L/D)", f"{ld_val:.2f}"],
        ["Span Efficiency (e)", f"{result.totals.e:.4f}"],
        ["Total Lift Force (L)", f"{lift_force:.1f} N"],
        ["Total Drag Force (D)", f"{d_tot_force:.2f} N"],
        ["Root Bending (M_root)", f"{root_bending_moment:.1f} N·m"],
        ["Pitching Moment (Cm)", f"{result.totals.Cm:.4f}"],
    ]

    table = ax8.table(
        cellText=table_data,
        colLabels=["Parameter", "Value"],
        loc="center",
        cellLoc="left",
        colWidths=[0.55, 0.45],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.8)
    table.scale(1.0, 1.15)

    # Style table header and cells
    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1e293b")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            if row % 2 == 0:
                cell.set_facecolor("#f8fafc")
            else:
                cell.set_facecolor("#ffffff")
        cell.set_edgecolor("#e2e8f0")

    ax8.set_title(r"8. Executive Aerodynamic KPI Summary", fontsize=10, fontweight="bold")

    fig.suptitle(
        f"{aircraft.name} - Executive Aerodynamic Performance & Design Dashboard",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    return fig
