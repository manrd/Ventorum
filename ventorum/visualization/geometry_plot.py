# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Plot the 3-D geometry of an aircraft.

The plots show the planform wireframe, the quarter-chord line, the control
points and the panel normals.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from ventorum.core.datatypes import Aircraft, SolverResult, SolverSettings
from ventorum.geometry.processing import discretize_surface


def plot_geometry(
    aircraft: Aircraft,
    settings: SolverSettings | None = None,
    show_normals: bool = False,
    show_control_points: bool = True,
    elev: float | None = None,
    azim: float | None = None,
) -> Figure:
    """Plot the 3-D wing geometry.

    Parameters
    ----------
    aircraft : Aircraft
    settings : SolverSettings or None
        Panel count and spacing of the shown strips. None gives the
        defaults of the solvers (:class:`SolverSettings`).
    show_normals : bool
        If *True*, draw short arrows for panel normals.
    show_control_points : bool
        If *True*, mark the 3/4-chord control points.
    elev : float or None
        3D view elevation angle in degrees.
    azim : float or None
        3D view azimuth angle in degrees.

    Returns
    -------
    Figure
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # None gives the solver defaults (see discretize_surface).
    n_panels = settings.n_panels if settings else None
    spacing = settings.spacing if settings else None

    for idx, surf in enumerate(aircraft.surfaces):
        ds = discretize_surface(surf, n_panels, spacing, idx)

        # Quarter-chord line
        ax.plot(
            ds.nodes_qc[:, 0], ds.nodes_qc[:, 1], ds.nodes_qc[:, 2],
            "-", linewidth=2, label=f"{surf.name} (1/4c)",
        )

        # Control points
        if show_control_points:
            ax.scatter(
                ds.control_points[:, 0],
                ds.control_points[:, 1],
                ds.control_points[:, 2],
                s=10, marker="x", alpha=0.6,
            )

        # Panel normals
        if show_normals:
            scale = surf.semi_span * 0.05
            for i in range(len(ds.y_panels)):
                cp = ds.control_points[i]
                n = ds.normals[i]
                ax.quiver(
                    cp[0], cp[1], cp[2],
                    n[0] * scale, n[1] * scale, n[2] * scale,
                    color="red", alpha=0.4, arrow_length_ratio=0.3,
                )

        # Leading and trailing edges (approximate from node data and chords)
        # LE = node_qc - 0.25*chord_hat, TE = node_qc + 0.75*chord_hat
        # For simplicity, approximate chord direction as x-axis
        n_pts = len(ds.y_panels)
        le_pts = np.zeros((n_pts, 3))
        te_pts = np.zeros((n_pts, 3))
        for i in range(n_pts):
            mid_node = 0.5 * (ds.nodes_qc[i] + ds.nodes_qc[i + 1])
            le_pts[i] = mid_node.copy()
            le_pts[i, 0] -= 0.25 * ds.chords[i]
            te_pts[i] = mid_node.copy()
            te_pts[i, 0] += 0.75 * ds.chords[i]

        ax.plot(le_pts[:, 0], le_pts[:, 1], le_pts[:, 2],
                "--", alpha=0.5, label=f"{surf.name} LE")
        ax.plot(te_pts[:, 0], te_pts[:, 1], te_pts[:, 2],
                ":", alpha=0.5, label=f"{surf.name} TE")

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("Wing Geometry")
    ax.legend(fontsize=8)

    # Equal aspect ratio
    all_nodes = np.vstack([
        discretize_surface(s, n_panels, spacing, i).nodes_qc
        for i, s in enumerate(aircraft.surfaces)
    ])
    max_range = np.ptp(all_nodes, axis=0).max() / 2.0
    mid = all_nodes.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    if elev is not None or azim is not None:
        ax.view_init(elev=elev, azim=azim)

    fig.tight_layout()
    return fig


def result_strip_polygons(
    aircraft: Aircraft,
    result: SolverResult,
    settings: SolverSettings | None = None,
) -> list[np.ndarray]:
    """Return the strip outlines of a solved result, one array per surface.

    The outlines come from the lattice of the solve
    (``result.details["lattice"]``), so each outline is at the spanwise
    position of the solved strip. A result without a lattice (for
    example the 3D panel solver) gets the lattice that
    :func:`ventorum.geometry.lattice.build_lattice` makes from
    ``aircraft`` and ``settings`` (the solver defaults when None).

    Parameters
    ----------
    aircraft : Aircraft
        Aircraft geometry model.
    result : SolverResult
        Solved result; ``result.spanwise`` holds one entry per surface.
    settings : SolverSettings or None
        Settings of the solve. They are used only for a result without a
        lattice.

    Returns
    -------
    list of numpy.ndarray
        For each entry of ``result.spanwise``, the outlines of its strips,
        shape (n_strips, 4, 3) [m]: leading edge left, leading edge right,
        trailing edge right, trailing edge left.

    Raises
    ------
    ValueError
        If the number of surfaces or strips of the layout is not the number
        in the result, or if a solved strip position ``y`` [m] is not in
        its outline (the layout is not the layout of the solve).
    """
    from ventorum.geometry.lattice import build_lattice

    lattice = (result.details or {}).get("lattice")
    if lattice is None:
        lattice = build_lattice(aircraft, settings or SolverSettings(), collocation="vlm", n_chord=1)
        origin = "the lattice rebuilt from the aircraft and the settings"
    else:
        origin = "the lattice of the result"
    if len(lattice.surfaces) != len(result.spanwise):
        raise ValueError(
            f"The result has {len(result.spanwise)} surfaces, but {origin} has "
            f"{len(lattice.surfaces)}."
        )
    out = []
    for sl, sw in zip(lattice.surfaces, result.spanwise):
        idx = np.arange(sl.strips.start, sl.strips.stop)
        polys = np.stack(
            [lattice.le_left[idx], lattice.le_right[idx], lattice.te_right[idx], lattice.te_left[idx]],
            axis=1,
        )
        n_vals = len(sw.Cl)
        if len(polys) != n_vals:
            raise ValueError(
                f"Surface {sl.name!r}: the result has {n_vals} strips, but {origin} has "
                f"{len(polys)}. Give the settings of the solve."
            )
        if sw.y is not None and n_vals:
            y_lo = np.minimum(polys[:, 0, 1], polys[:, 1, 1])
            y_hi = np.maximum(polys[:, 0, 1], polys[:, 1, 1])
            tol = 1e-9 * max(1.0, float(np.max(np.abs(polys[..., 1]))))
            y = np.asarray(sw.y, dtype=float)
            if np.any(y < y_lo - tol) or np.any(y > y_hi + tol):
                raise ValueError(
                    f"Surface {sl.name!r}: a solved strip position y is not in its strip of "
                    f"{origin}. Give the settings of the solve."
                )
        out.append(polys)
    return out


def plot_surface_results(
    aircraft: Aircraft,
    result: SolverResult,
    metric: str = "Cl",
    settings: SolverSettings | None = None,
    cmap: str = "viridis",
    clim: tuple[float, float] | None = None,
    edge_color: str = "black",
    edge_width: float = 0.3,
    alpha: float = 0.92,
    elev: float | None = 22.0,
    azim: float | None = -125.0,
    ax: plt.Axes | None = None,
) -> Figure:
    """Plot aerodynamic result values directly on the 3-D aircraft geometry as colored panels.

    Parameters
    ----------
    aircraft : Aircraft
        Aircraft geometry model.
    result : SolverResult
        LLT analysis result containing spanwise distribution data.
    metric : str
        Aerodynamic quantity to color the surface panels with:
        - "Cl": Local section lift coefficient (default)
        - "gamma": Circulation Γ [m²/s]
        - "alpha_eff": Effective angle of attack α_eff [°]
        - "alpha_i": Induced downwash angle α_i [°]
        - "Cd_i": Local induced drag coefficient
        - "lift": Local section lift force per unit span L' [N/m]
    settings : SolverSettings or None
        Settings of the solve. The strips come from the lattice of the
        result; the settings are used only for a result without a lattice
        (see :func:`result_strip_polygons`).
    cmap : str
        Matplotlib colormap name (e.g. 'viridis', 'plasma', 'turbo', 'coolwarm').
    clim : tuple[float, float] or None
        Explicit (min, max) limits for the colorbar. If None, auto-scaled.
    edge_color : str
        Color of panel outline borders.
    edge_width : float
        Line width of panel borders.
    alpha : float
        Transparency of surface panels [-] (0 = invisible, 1 = opaque); not an angle.
    elev : float or None
        3D view elevation angle in degrees.
    azim : float or None
        3D view azimuth angle in degrees.
    ax : plt.Axes or None
        Existing 3D axes to draw on. If None, a new figure is created.

    Returns
    -------
    Figure
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm

    if ax is None:
        fig = plt.figure(figsize=(13, 8))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    metric_key = metric.lower().strip()
    if metric_key in ("cl", "c_l"):
        def extract_fn(sw):
            return sw.Cl
        metric_label = "Section Lift Coefficient $C_l$"
        title_metric = "$C_l$"
    elif metric_key in ("gamma", "circulation"):
        def extract_fn(sw):
            return sw.gamma
        metric_label = r"Circulation $\Gamma$ [m$^2$/s]"
        title_metric = r"$\Gamma$"
    elif metric_key in ("alpha_eff", "alphaeff", "a_eff"):
        def extract_fn(sw):
            return np.degrees(sw.alpha_eff)
        metric_label = r"Effective Angle of Attack $\alpha_{eff}$ [°]"
        title_metric = r"$\alpha_{eff}$"
    elif metric_key in ("alpha_i", "alphai", "a_i", "downwash"):
        def extract_fn(sw):
            return np.degrees(sw.alpha_i)
        metric_label = r"Induced Downwash Angle $\alpha_i$ [°]"
        title_metric = r"$\alpha_i$"
    elif metric_key in ("cd_i", "cdi", "c_di"):
        def extract_fn(sw):
            return sw.Cd_i
        metric_label = r"Section Induced Drag $C_{di}$"
        title_metric = "$C_{di}$"
    elif metric_key in ("lift", "local_lift"):
        def extract_fn(sw):
            return sw.local_lift
        metric_label = r"Section Lift $L'$ [N/m]"
        title_metric = r"Section Lift"
    else:
        raise ValueError(
            f"Unknown metric '{metric}'. Expected 'Cl', 'gamma', 'alpha_eff', "
            f"'alpha_i', 'Cd_i', or 'lift'."
        )

    all_polys = []
    all_vals = []

    # The strips of the solve (see result_strip_polygons): each value is at its solved strip.
    for polys, sw in zip(result_strip_polygons(aircraft, result, settings), result.spanwise):
        all_polys.extend(list(p) for p in polys)
        all_vals.extend(np.asarray(extract_fn(sw)))

    all_vals_arr = np.asarray(all_vals)
    if len(all_vals_arr) == 0:
        return fig

    vmin = clim[0] if clim is not None else float(np.min(all_vals_arr))
    vmax = clim[1] if clim is not None else float(np.max(all_vals_arr))
    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 0.1

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap)
    facecolors = cmap_obj(norm(all_vals_arr))

    poly_col = Poly3DCollection(
        all_polys,
        facecolors=facecolors,
        edgecolors=edge_color,
        linewidths=edge_width,
        alpha=alpha,
    )
    ax.add_collection3d(poly_col)

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.08, aspect=20)
    cbar.set_label(metric_label, fontsize=11, fontweight="bold")

    # Set axes limits for equal aspect ratio
    all_pts = np.vstack([np.array(p) for p in all_polys])
    max_range = np.ptp(all_pts, axis=0).max() / 2.0
    mid = all_pts.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title(
        f"{aircraft.name} - 3D Surface {title_metric} Distribution\n"
        f"($C_L$ = {result.totals.CL:.4f}, $C_{{Di}}$ = {result.totals.CDi:.5f}, e = {result.totals.e:.3f})",
        fontsize=12,
        fontweight="bold",
    )

    if elev is not None or azim is not None:
        ax.view_init(elev=elev, azim=azim)

    fig.tight_layout()
    return fig


# Convenience alias
plot_panel_results = plot_surface_results
