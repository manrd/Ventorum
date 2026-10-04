# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Plot the vortex system and the cross-flow behind the aircraft.

The cross-flow plot shows the flow in a plane behind the aircraft.

Both plots use the lattice, the circulation, the wake direction and the
ground plane that the solver used (``result.details``), so they show the
same vortex system that gave the forces.
"""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ventorum.aero.system import UnknownMap, build_sources
from ventorum.aero.vortex import induced_velocity
from ventorum.core.datatypes import Aircraft, LiftingSurface, SolverResult


def _solution(result: SolverResult):
    d = getattr(result, "details", None) or {}
    needed = ("lattice", "gamma", "wake_dir")
    if not all(k in d for k in needed):
        raise ValueError(
            "The result has no lattice data. Use a result from the vortex-lattice or lifting-line "
            "solvers (the Fourier solver has no vortex lattice)."
        )
    return d["lattice"], np.asarray(d["gamma"], dtype=float), np.asarray(d["wake_dir"], dtype=float), d.get("ground")


def plot_vortex_wake(
    aircraft: Aircraft | LiftingSurface,
    result: SolverResult,
    wake_length: float | None = None,
    elev: float = 24.0,
    azim: float = -125.0,
    cmap: str = "viridis",
    ax: plt.Axes | None = None,
) -> Figure:
    """Plot a 3-D view of the vortex system.

    The plot shows the bound vortices, the trailing legs on the surface and
    the wake legs. The colour shows the strength of the trailing vorticity
    shed at each strip edge.

    Parameters
    ----------
    aircraft : Aircraft or LiftingSurface
        Used for the title only.
    result : SolverResult
        Result of a lattice solver.
    wake_length : float or None
        Length of the drawn wake legs [m]. Default 1.5 times the reference span.
    elev : float, optional
        Elevation angle of the view [deg].
    azim : float, optional
        Azimuth angle of the view [deg].
    cmap : str, optional
        Name of the Matplotlib colour map.
    ax : matplotlib.axes.Axes or None, optional
        3-D axes to draw on. If None, a new figure is made.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.
    """
    lattice, gamma, wake_dir, _ = _solution(result)
    name = aircraft.name if isinstance(aircraft, (Aircraft, LiftingSurface)) else "Aircraft"
    if ax is None:
        fig = plt.figure(figsize=(13, 8))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    span = float(np.ptp(lattice.all_points()[:, 1])) or 1.0
    L = wake_length if wake_length is not None else 1.5 * span
    n_c = lattice.n_chord
    strip_gamma = gamma.reshape(lattice.n_strips, n_c).sum(axis=1)

    # Bound vortices of all panels.
    for a, b in zip(lattice.a, lattice.b):
        ax.plot(*np.c_[a, b], color="#334155", linewidth=1.0, alpha=0.8)

    # Trailing vorticity at each strip edge: the difference of the strip
    # circulations on both sides (the full circulation at a free edge).
    edges, strengths = [], []
    for surf in lattice.surfaces:
        idx = np.arange(surf.strips.start, surf.strips.stop)
        g = strip_gamma[idx]
        shed = np.concatenate([[g[0]], np.diff(g) * -1.0, [-g[-1]]]) if len(g) else np.array([])
        te_nodes = np.vstack([lattice.te_left[idx], lattice.te_right[idx][-1:]])
        for p, s_val in zip(te_nodes, shed):
            edges.append(p)
            strengths.append(s_val)
    edges = np.asarray(edges)
    strengths = np.asarray(strengths)
    vmax = float(np.max(np.abs(strengths))) if strengths.size else 1.0
    norm = mcolors.Normalize(vmin=0.0, vmax=max(vmax, 1e-12))
    cmap_obj = plt.get_cmap(cmap)
    for p, s_val in zip(edges, strengths):
        q = p + L * wake_dir
        ax.plot(*np.c_[p, q], color=cmap_obj(norm(abs(s_val))), linewidth=1.0, alpha=0.75)

    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.08, aspect=20)
    cbar.set_label(r"Shed vorticity $|\Delta\Gamma|$ [m$^2$/s]", fontsize=10)

    pts = np.vstack([lattice.all_points(), edges + L * wake_dir]) if edges.size else lattice.all_points()
    r = np.ptp(pts, axis=0).max() / 2.0
    m = pts.mean(axis=0)
    ax.set_xlim(m[0] - r, m[0] + r)
    ax.set_ylim(m[1] - r, m[1] + r)
    ax.set_zlim(m[2] - r, m[2] + r)
    ax.set_xlabel("x [m] (aft)")
    ax.set_ylabel("y [m] (right)")
    ax.set_zlabel("z [m] (up)")
    ax.set_title(f"{name}: vortex system ({result.solver_type}, {n_c} chordwise panels)", fontsize=12)
    ax.view_init(elev=elev, azim=azim)
    fig.tight_layout()
    return fig


def plot_trefftz_plane(
    aircraft: Aircraft | LiftingSurface,
    result: SolverResult,
    x_slice: float | None = None,
    grid_res: tuple[int, int] = (35, 25),
    y_limits: tuple[float, float] | None = None,
    z_limits: tuple[float, float] | None = None,
    cmap: str = "coolwarm",
    streamlines: bool = True,
    ax: plt.Axes | None = None,
) -> Figure:
    """Induced cross-flow in a plane x = x_slice behind the aircraft.

    The velocity is induced by the solved vortex system, including the ground
    images in ground effect; points below the ground are masked. The colour
    is the downwash angle ``-w / V`` [deg] (body axes).

    Parameters
    ----------
    x_slice : float or None
        Position of the plane [m]. Default: 1.5 reference spans behind the
        most aft trailing edge.
    """
    lattice, gamma, wake_dir, ground = _solution(result)
    cond = result.condition
    V = float(cond.V_inf) if cond is not None else 1.0
    pts = lattice.all_points()
    span = float(np.ptp(pts[:, 1])) or 1.0
    if x_slice is None:
        x_slice = float(np.max(pts[:, 0]) + 1.5 * span)
    y_min, y_max = y_limits if y_limits is not None else (pts[:, 1].min() - 0.15 * span, pts[:, 1].max() + 0.15 * span)
    zc = float(np.mean(pts[:, 2]))
    z_min, z_max = z_limits if z_limits is not None else (zc - 0.35 * span, zc + 0.35 * span)

    Ny, Nz = grid_res
    y_vals = np.linspace(y_min, y_max, Ny)
    z_vals = np.linspace(z_min, z_max, Nz)
    Y, Z = np.meshgrid(y_vals, z_vals)
    P = np.column_stack([np.full(Y.size, x_slice), Y.ravel(), Z.ravel()])

    n = lattice.n_panels
    src = build_sources(lattice, wake_dir, UnknownMap(np.arange(n), np.arange(n), False), ground)
    v = induced_velocity(P, src, gamma)
    v_field = v[:, 1].reshape(Z.shape)
    w_field = v[:, 2].reshape(Z.shape)
    eps = -(w_field / V) * (180.0 / np.pi)
    if ground is not None:
        below = (ground.height(P) < 0.0).reshape(Z.shape)
        eps = np.ma.masked_where(below, eps)
        v_field = np.where(below, 0.0, v_field)
        w_field = np.where(below, 0.0, w_field)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6.5))
    else:
        fig = ax.figure
    vmax = max(float(np.nanpercentile(np.abs(np.ma.filled(eps, 0.0)), 98)), 0.5)
    cf = ax.contourf(Y, Z, eps, levels=np.linspace(-vmax, vmax, 31), cmap=cmap, extend="both", alpha=0.88)
    cbar = fig.colorbar(cf, ax=ax, shrink=0.75, pad=0.04)
    cbar.set_label(r"Downwash angle $-w/V$ [deg]", fontsize=10)
    if streamlines:
        ax.streamplot(y_vals, z_vals, v_field, w_field, color="#0f172a", density=1.1, linewidth=0.8, arrowsize=1.0)
    for surf in lattice.surfaces:
        ax.plot(surf.edge_te[:, 1], surf.edge_te[:, 2], color="black", linewidth=2.0, linestyle="--")
    if ground is not None:
        # Trace of the ground plane in the cutting plane.
        k, off = ground.normal, ground.offset
        if abs(k[2]) > 1e-9:
            z_g = (off - k[0] * x_slice - k[1] * y_vals) / k[2]
            ax.plot(y_vals, z_g, color="#7c2d12", linewidth=2.5, label="ground")
            ax.legend(loc="upper right", fontsize=9)
    ax.set_xlabel("y [m]")
    ax.set_ylabel("z [m]")
    ax.set_title(f"Induced cross-flow at x = {x_slice:.2f} m (V = {V:.1f} m/s)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return fig
