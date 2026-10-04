# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Visualization and plotting suite for ground-effect aerodynamics.

Generates publication-quality figures for:
- Height sweeps: CL, CDi, L/D, and span efficiency e vs h/c
- Roll effects: Rolling moment Cl vs bank angle phi, roll stability Cl_phi vs h/c, and induced yaw Cn
- Pitch stability: Cm vs alpha, pitch and height aerodynamic centres, Irodov margin
- Spanwise distributions: Asymmetric circulation Gamma(y), cl(y), and cdi(y) on banked wings
- 2D contour maps: Carpet plots of CL(h/c, alpha), L/D(h/c, alpha), Cl(h/c, phi)
- Clearance & strike envelopes: Safe bank angle limits phi_max vs h/c
"""

from __future__ import annotations

import functools
import os
import numpy as np
import matplotlib.pyplot as plt

from ventorum.ground_effect.state import GroundEffectResult
from ventorum.ground_effect.sweep import GroundEffectSweepResult


# Styling constants for high-contrast, publication-quality figures
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
def _line_color(i: int, n: int):
    """Palette colour for line i of n; a continuous colour map if n is larger than the palette."""
    if n <= len(PALETTE):
        return PALETTE[i]
    return plt.cm.viridis(i / max(1, n - 1))


STYLE_CONFIG = {
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
}

# Metrics of a GroundEffectSweepResult that plot_ground_effect_matrix can show.
MATRIX_METRICS = ("CL", "CDi", "CD", "CY", "Cl", "Cm", "Cn", "L_over_D", "e", "h_min")


def _styled(func):
    """Apply STYLE_CONFIG only while *func* runs (the global matplotlib settings do not change)."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with plt.rc_context(STYLE_CONFIG):
            return func(*args, **kwargs)
    return wrapper


@_styled
def plot_height_sweep(
    sweep_result: GroundEffectSweepResult,
    alpha_deg: float | None = None,
    phi_deg: float = 0.0,
    save_path: str | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot aerodynamic augmentation curves (CL, CDi, L/D, e) vs normalized height h/c.

    Parameters
    ----------
    sweep_result : GroundEffectSweepResult
        Sweep data.
    alpha_deg : float or None
        Angle of attack to plot [deg]. If None, chooses the middle alpha.
    phi_deg : float
        Bank angle to isolate [deg] (default 0.0).
    save_path : str or None
        If provided, saves figure to this path.
    dpi : int
        Figure resolution.

    Returns
    -------
    plt.Figure
    """
    if alpha_deg is None:
        a_idx = len(sweep_result.alphas_deg) // 2
        alpha_val = float(sweep_result.alphas_deg[a_idx])
    else:
        alpha_val = float(alpha_deg)

    h_vals, cl_vals = sweep_result.get_slice_1d("CL", alpha_val=alpha_val, phi_val=phi_deg)
    _, cdi_vals = sweep_result.get_slice_1d("CDi", alpha_val=alpha_val, phi_val=phi_deg)
    _, ld_vals = sweep_result.get_slice_1d("L_over_D", alpha_val=alpha_val, phi_val=phi_deg)
    _, e_vals = sweep_result.get_slice_1d("e", alpha_val=alpha_val, phi_val=phi_deg)

    h_over_c = h_vals / sweep_result.c_ref if sweep_result.c_ref > 0 else h_vals

    fig, axs = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    fig.suptitle(f"Ventorum Ground Effect Augmentation Curves ($\\alpha = {alpha_val:.1f}^\\circ$, $\\phi = {phi_deg:.1f}^\\circ$)", fontweight="bold")

    # 1. Lift coefficient CL vs h/c
    axs[0, 0].plot(h_over_c, cl_vals, "o-", color="#1f77b4", linewidth=2.0, markersize=5)
    axs[0, 0].set_xlabel("$h / c$ (Height / Mean Chord)")
    axs[0, 0].set_ylabel("Lift Coefficient $C_L$")
    axs[0, 0].set_title("Lift Augmentation (Ground Cushion)")
    axs[0, 0].grid(True)

    # 2. Induced drag coefficient CDi vs h/c
    axs[0, 1].plot(h_over_c, cdi_vals, "s-", color="#d62728", linewidth=2.0, markersize=5)
    axs[0, 1].set_xlabel("$h / c$ (Height / Mean Chord)")
    axs[0, 1].set_ylabel("Induced Drag Coefficient $C_{Di}$")
    axs[0, 1].set_title("Induced Drag Suppression")
    axs[0, 1].grid(True)

    # 3. Aerodynamic efficiency L/D vs h/c
    axs[1, 0].plot(h_over_c, ld_vals, "^-", color="#2ca02c", linewidth=2.0, markersize=5)
    axs[1, 0].set_xlabel("$h / c$ (Height / Mean Chord)")
    axs[1, 0].set_ylabel("Lift-to-Drag Ratio $L/D$")
    axs[1, 0].set_title("Aerodynamic Efficiency")
    axs[1, 0].grid(True)

    # 4. Oswald span efficiency e vs h/c
    axs[1, 1].plot(h_over_c, e_vals, "d-", color="#9467bd", linewidth=2.0, markersize=5)
    axs[1, 1].axhline(1.0, color="gray", linestyle=":", label="Free-Air Elliptic ($e=1.0$)")
    axs[1, 1].set_xlabel("$h / c$ (Height / Mean Chord)")
    axs[1, 1].set_ylabel("Oswald Span Efficiency $e$")
    axs[1, 1].set_title("Effective Span Efficiency")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=dpi)

    return fig


@_styled
def plot_roll_effect(
    sweep_result: GroundEffectSweepResult,
    alpha_deg: float | None = None,
    save_path: str | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot rolling moment Cl vs bank angle phi across various heights, showcasing roll stability.

    Standard sign convention: ``Cl > 0`` is right wing down, and ``phi > 0``
    puts the right wing nearer to the ground, so a bank-restoring moment has
    ``Cl * phi < 0`` and the roll stiffness ``-dCl/dphi`` is positive.

    Panels: Cl(phi) per height, the roll stiffness ``-dCl/dphi`` against h/c
    (log scale), and the yawing moment Cn(phi).

    Parameters
    ----------
    sweep_result : GroundEffectSweepResult
        Sweep data.
    alpha_deg : float or None
        Angle of attack of the plotted slice [deg]. If None, the middle angle.
    save_path : str or None
        If given, the figure is saved to this path.
    dpi : int
        Resolution of the saved figure.
    """
    if alpha_deg is None:
        a_idx = int(np.argmin(np.abs(sweep_result.alphas_deg - 4.0))) if len(sweep_result.alphas_deg) > 0 else 0
        alpha_val = float(sweep_result.alphas_deg[a_idx])
    else:
        alpha_val = float(alpha_deg)
        a_idx = int(np.argmin(np.abs(sweep_result.alphas_deg - alpha_val)))

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    fig.suptitle(f"Ground-Effect Roll Coupling & Aerodynamic Restoring Moments ($\\alpha = {alpha_val:.1f}^\\circ$)", fontweight="bold")

    phis = sweep_result.phis_deg
    h_over_c = sweep_result.heights / sweep_result.c_ref if sweep_result.c_ref > 0 else sweep_result.heights

    # Subplot 1: Cl vs phi for each height
    for hi, h_c in enumerate(h_over_c):
        cl_roll = sweep_result.Cl[hi, a_idx, :]
        axs[0].plot(phis, cl_roll, "o-", color=_line_color(hi, len(h_over_c)), label=f"$h/c = {h_c:.2f}$", linewidth=1.8, markersize=4)

    axs[0].axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    axs[0].axvline(0.0, color="black", linestyle="--", linewidth=0.8)
    axs[0].set_xlabel("Roll / Bank Angle $\\phi$ [deg]")
    axs[0].set_ylabel("Rolling Moment Coefficient $C_l$")
    axs[0].set_title("Restoring Rolling Moment $C_l(\\phi)$")
    axs[0].legend(loc="upper left")
    axs[0].grid(True)

    # Subplot 2: Roll stability derivative Cl_phi vs h/c
    derivs = sweep_result.compute_stability_derivatives()
    if "Cl_phi_deg" in derivs:
        stiff = -np.asarray(derivs["Cl_phi_deg"], dtype=float)
        ok = np.isfinite(stiff) & (stiff > 0)
        if np.any(ok):
            axs[1].semilogy(h_over_c[ok], stiff[ok], "o-", color="#d62728", linewidth=2.0, markersize=6)
        axs[1].set_xlabel("$h / c$ (Height / Mean Chord)")
        axs[1].set_ylabel("Roll Stiffness $-\\partial C_l / \\partial \\phi$ [1/deg] (Log)")
        axs[1].set_title("Bank-Restoring Roll Stiffness (positive = restoring)")
        axs[1].grid(True, which="both")

    # Subplot 3: Yawing moment Cn vs phi (induced yaw due to asymmetric drag)
    for hi, h_c in enumerate(h_over_c):
        cn_yaw = sweep_result.Cn[hi, a_idx, :]
        axs[2].plot(phis, cn_yaw, "s-", color=_line_color(hi, len(h_over_c)), label=f"$h/c = {h_c:.2f}$", linewidth=1.8, markersize=4)

    axs[2].axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    axs[2].set_xlabel("Roll / Bank Angle $\\phi$ [deg]")
    axs[2].set_ylabel("Yawing Moment Coefficient $C_n$")
    axs[2].set_title("Roll-Induced Yawing Moment $C_n(\\phi)$")
    axs[2].grid(True)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=dpi)

    return fig


@_styled
def plot_pitch_stability(
    sweep_result: GroundEffectSweepResult,
    phi_deg: float = 0.0,
    save_path: str | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot pitching moment Cm vs alpha and neutral point / Aerodynamic Center migration.

    Parameters
    ----------
    sweep_result : GroundEffectSweepResult
        Sweep data.
    phi_deg : float
        Bank angle of the plotted slice [deg].
    save_path : str or None
        If given, the figure is saved to this path.
    dpi : int
        Resolution of the saved figure.
    """
    p0_idx = int(np.argmin(np.abs(sweep_result.phis_deg - phi_deg)))
    alphas = sweep_result.alphas_deg
    h_over_c = sweep_result.heights / sweep_result.c_ref if sweep_result.c_ref > 0 else sweep_result.heights

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    fig.suptitle(f"Longitudinal Static Stability in Ground Effect ($\\phi = {phi_deg:.1f}^\\circ$)", fontweight="bold")

    # 1. Cm vs alpha at each height
    for hi, h_c in enumerate(h_over_c):
        cm_vals = sweep_result.Cm[hi, :, p0_idx]
        axs[0].plot(alphas, cm_vals, "o-", color=_line_color(hi, len(h_over_c)), label=f"$h/c = {h_c:.2f}$", linewidth=1.8, markersize=4)

    axs[0].axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    axs[0].set_xlabel("Angle of Attack $\\alpha$ [deg]")
    axs[0].set_ylabel("Pitching Moment Coefficient $C_m$")
    axs[0].set_title("Pitching Moment $C_m(\\alpha)$")
    axs[0].legend()
    axs[0].grid(True)

    # 2. Pitch and height aerodynamic centres (aft of the reference point) vs h/c
    derivs = sweep_result.compute_stability_derivatives()
    c = sweep_result.c_ref if sweep_result.c_ref > 0 else 1.0
    a_mid = len(alphas) // 2
    if "x_alpha" in derivs and "x_h" in derivs:
        axs[1].plot(h_over_c, derivs["x_alpha"][:, a_mid] / c, "s-", color="#1f77b4", linewidth=2.0, label="$x_\\alpha / c$ (pitch)")
        axs[1].plot(h_over_c, derivs["x_h"][:, a_mid] / c, "o-", color="#d62728", linewidth=2.0, label="$x_h / c$ (height)")
        axs[1].legend()
    elif "x_ac" in derivs:
        axs[1].plot(h_over_c, derivs["x_ac"] / c, "s-", color="#1f77b4", linewidth=2.0, label="$x_\\alpha / c$ (pitch)")
        axs[1].legend()
    axs[1].set_xlabel("$h / c$ (Height / Mean Chord)")
    axs[1].set_ylabel("Aerodynamic centre aft of reference point / c")
    axs[1].set_title(f"Aerodynamic Centres ($\\alpha = {alphas[a_mid]:.1f}^\\circ$)")
    axs[1].grid(True)

    # 3. Irodov margin (x_alpha - x_h)/c: must be positive for static stability
    if "irodov_margin" in derivs:
        axs[2].plot(h_over_c, derivs["irodov_margin"][:, a_mid], "d-", color="#2ca02c", linewidth=2.0)
        axs[2].axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        axs[2].set_xlabel("$h / c$ (Height / Mean Chord)")
        axs[2].set_ylabel("$(x_\\alpha - x_h) / c$")
        axs[2].set_title("Irodov Margin (stable if > 0)")
        axs[2].grid(True)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=dpi)

    return fig


def _reference_distribution(result: GroundEffectResult) -> dict[str, np.ndarray]:
    """Spanwise data of the main surface and its mirror copies, sorted by y.

    Keys: ``y``, ``gamma``, ``Cl``, ``Cd_i`` and ``z`` (height of the
    quarter-chord panel centres in the ground frame of the plots).
    """
    idx = [i for i in result.reference_surface_indices if i < len(result.spanwise)] or [0]
    sws = [result.spanwise[i] for i in idx]
    out = {k: np.concatenate([np.asarray(getattr(sw, k), dtype=float) for sw in sws])
           for k in ("y", "gamma", "Cl", "Cd_i")}
    if len(result.transformed_surfaces) > max(idx):
        out["z"] = np.concatenate([result.transformed_surfaces[i].panel_centers_qc[:, 2] for i in idx])
    if out.get("z") is None or len(out["z"]) != len(out["y"]):
        out["z"] = np.full(len(out["y"]), np.nan)
    order = np.argsort(out["y"], kind="stable")
    return {k: v[order] for k, v in out.items()}


@_styled
def plot_asymmetric_distributions(
    result_banked: GroundEffectResult,
    result_level: GroundEffectResult | None = None,
    save_path: str | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot spanwise circulation, section lift, and downwash comparing a banked wing to a level wing.

    The curves show the main surface and its mirror copies (for example two
    mirrored half wings), so that the two halves of the span show.
    """
    b_half = result_banked.b_ref / 2.0
    sw_banked = _reference_distribution(result_banked)
    y_norm = sw_banked["y"] / b_half
    sw_level = _reference_distribution(result_level) if result_level is not None else None
    y_norm_level = sw_level["y"] / b_half if sw_level is not None else None

    fig, axs = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    phi_val = result_banked.condition.phi_deg
    h_val = result_banked.condition.h
    fig.suptitle(f"Spanwise Asymmetric Aerodynamics: Banked Wing in Ground Effect ($h = {h_val:.2f}$ m, $\\phi = {phi_val:.1f}^\\circ$)", fontweight="bold")

    # 1. Circulation Gamma(y)
    axs[0, 0].plot(y_norm, sw_banked["gamma"], color="#d62728", linewidth=2.2, label=f"Banked ($\\phi={phi_val:.1f}^\\circ$)")
    if sw_level is not None:
        axs[0, 0].plot(y_norm_level, sw_level["gamma"], color="gray", linestyle="--", linewidth=1.8, label="Level ($\\phi=0^\\circ$)")
    axs[0, 0].set_xlabel("Normalized Span $2y / b$ (Negative = Port, Positive = Starboard)")
    axs[0, 0].set_ylabel("Circulation $\\Gamma(y)$ [m$^2$/s]")
    axs[0, 0].set_title("Circulation Distribution (Lift Cushion on Dipped Wing)")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # 2. Sectional lift coefficient cl(y)
    axs[0, 1].plot(y_norm, sw_banked["Cl"], color="#1f77b4", linewidth=2.2, label=f"Banked ($\\phi={phi_val:.1f}^\\circ$)")
    if sw_level is not None:
        axs[0, 1].plot(y_norm_level, sw_level["Cl"], color="gray", linestyle="--", linewidth=1.8, label="Level ($\\phi=0^\\circ$)")
    axs[0, 1].set_xlabel("Normalized Span $2y / b$")
    axs[0, 1].set_ylabel("Section Lift Coefficient $c_l(y)$")
    axs[0, 1].set_title("Section Lift Distribution")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # 3. Sectional induced drag coefficient cd_i(y)
    axs[1, 0].plot(y_norm, sw_banked["Cd_i"], color="#ff7f0e", linewidth=2.2, label=f"Banked ($\\phi={phi_val:.1f}^\\circ$)")
    if sw_level is not None:
        axs[1, 0].plot(y_norm_level, sw_level["Cd_i"], color="gray", linestyle="--", linewidth=1.8, label="Level ($\\phi=0^\\circ$)")
    axs[1, 0].set_xlabel("Normalized Span $2y / b$")
    axs[1, 0].set_ylabel("Section Induced Drag $c_{di}(y)$")
    axs[1, 0].set_title("Induced Drag Distribution (Asymmetry Drives Yaw)")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # 4. Vertical clearance profile across the span
    z_clearance = sw_banked["z"] - (-result_banked.condition.h)
    axs[1, 1].plot(y_norm, z_clearance, color="#2ca02c", linewidth=2.2, label="Clearance to Ground")
    axs[1, 1].axhline(0.0, color="brown", linestyle="-", linewidth=2.5, label="Ground Plane ($z=0$)")
    axs[1, 1].set_xlabel("Normalized Span $2y / b$")
    axs[1, 1].set_ylabel("Height Above Ground $h(y)$ [m]")
    axs[1, 1].set_title("Physical Wing Clearance Across Span")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=dpi)

    return fig


@_styled
def plot_ground_effect_matrix(
    sweep_result: GroundEffectSweepResult,
    metric: str = "CL",
    phi_deg: float = 0.0,
    save_path: str | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot 2D contour carpet plot of an aerodynamic parameter as a function of h/c and alpha.

    *metric* is one of ``MATRIX_METRICS``. The sweep must have at least two
    heights and two angles of attack.

    Parameters
    ----------
    sweep_result : GroundEffectSweepResult
        Sweep data.
    metric : str
        Name of the plotted quantity.
    phi_deg : float
        Bank angle of the plotted slice [deg].
    save_path : str or None
        If given, the figure is saved to this path.
    dpi : int
        Resolution of the saved figure.

    Raises
    ------
    ValueError
        If *metric* is not a valid name, or if the grid is smaller than 2 x 2.
    """
    if metric not in MATRIX_METRICS:
        raise ValueError(f"metric={metric!r} is not valid. Use one of: {', '.join(MATRIX_METRICS)}.")
    n_h, n_a = len(sweep_result.heights), len(sweep_result.alphas_deg)
    if n_h < 2 or n_a < 2:
        raise ValueError(
            f"A contour map needs at least 2 heights and 2 angles of attack (the sweep has {n_h} and {n_a}). "
            "Use plot_height_sweep or plot_pitch_stability for a single height or angle."
        )
    p0_idx = int(np.argmin(np.abs(sweep_result.phis_deg - phi_deg)))
    h_over_c = sweep_result.heights / sweep_result.c_ref if sweep_result.c_ref > 0 else sweep_result.heights
    alphas = sweep_result.alphas_deg

    H, A = np.meshgrid(h_over_c, alphas, indexing="ij")
    Z = np.asarray(getattr(sweep_result, metric), dtype=float)[:, :, p0_idx]
    if not np.any(np.isfinite(Z)):
        raise ValueError(f"All values of {metric} are NaN at phi = {phi_deg} deg (every case failed).")

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    cf = ax.contourf(H, A, Z, levels=25, cmap="viridis")
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(f"{metric} Value")

    cs = ax.contour(H, A, Z, levels=12, colors="white", linewidths=0.7, alpha=0.6)
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.2f")

    ax.set_xlabel("Normalized Ground Height $h / c$")
    ax.set_ylabel("Angle of Attack $\\alpha$ [deg]")
    ax.set_title(f"Ground Effect Contour Map: {metric}($h/c$, $\\alpha$) at $\\phi = {phi_deg:.1f}^\\circ$", fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.4)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=dpi)

    return fig


@_styled
def plot_clearance_envelope(
    sweep_result: GroundEffectSweepResult,
    save_path: str | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot the allowable bank angle corridor (strike boundary phi_max vs h/c).

    Heights without a limit (NaN) show as gaps. A limit equal to the search
    limit (``strike_limit_found`` False) means no contact up to that bank
    angle; these points have an open marker.

    Raises
    ------
    ValueError
        If the sweep has no bank strike limits (for example a sweep with
        ``compute_strike_limit=False``).
    """
    h_over_c = sweep_result.heights / sweep_result.c_ref if sweep_result.c_ref > 0 else sweep_result.heights
    # Take middle alpha
    a_idx = len(sweep_result.alphas_deg) // 2
    phi_limits = np.asarray(sweep_result.phi_strike_limit, dtype=float)[:, a_idx, 0]
    ok = np.isfinite(phi_limits)
    if not np.any(ok):
        raise ValueError(
            "The sweep has no bank strike limits (all NaN). Run it with compute_strike_limit=True."
        )
    found = np.asarray(sweep_result.strike_limit_found, dtype=bool)
    found = found[:, a_idx, 0] if found.shape == sweep_result.phi_strike_limit.shape else ok
    top = float(np.nanmax(phi_limits)) * 1.25

    fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    ax.plot(h_over_c, phi_limits, "o-", color="#d62728", linewidth=2.2, markersize=5, label="Bank angle at first contact $\\phi_{\\max}$")
    open_pts = ok & ~found
    if np.any(open_pts):
        ax.plot(h_over_c[open_pts], phi_limits[open_pts], "o", color="#d62728", markerfacecolor="white",
                markersize=7, linestyle="none", label="No contact up to the search limit")
    ax.fill_between(h_over_c, 0, phi_limits, where=ok, color="#2ca02c", alpha=0.18, label="No contact (geometry only)")
    ax.fill_between(h_over_c, phi_limits, top, where=ok & found, color="#d62728", alpha=0.15, label="Contact")

    ax.set_xlabel("$h / c$ (Height / Mean Chord)")
    ax.set_ylabel("Critical Bank Angle $\\phi_{\\max}$ [deg]")
    ax.set_title("Bank Angle at Ground Contact (geometry only, flat ground)", fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=dpi)

    return fig
