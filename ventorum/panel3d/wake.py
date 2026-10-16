# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Wake shedding and wake geometry for the 3D panel method.

This module builds the flat wake panels that shed from the trailing edges
of lifting surfaces in Ventorum-3DP. In free air the wake follows the body
+x axis, so the influence matrix does not depend on the flight condition.
In ground effect the solver sheds the wake along the free stream (parallel
to the ground), so that no wake panel goes below the ground plane.

Published sources
-----------------
* Morino, L. and Kuo, C. C., 1974, Subsonic potential aerodynamics for
  complex configurations: A general theory, AIAA Journal, Vol. 12, No. 2,
  pp. 191-197.
* Katz, J. and Plotkin, A., 2001, Low-Speed Aerodynamics, 2nd ed.,
  Cambridge University Press, section 9.4 and chapter 12.
* Maskew, B., 1987, Program VSAERO Theory Document, NASA CR-4023.

Conventions
-----------
* Wake panels extend along the wake direction (default the body +x axis,
  [1.0, 0.0, 0.0]).
* The panel normal vector is oriented to point in the same direction as the
  outward normal of the upper trailing-edge panel.
* Each wake strip carries doublet strength mu_w = mu_upper - mu_lower.
* The default wake length is 50 times the largest streamwise chord.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "DEFAULT_WAKE_CHORDS",
    "PanelWake",
    "build_wake",
    "largest_streamwise_chord",
]

#: Default wake length in multiples of the largest streamwise chord [-].
DEFAULT_WAKE_CHORDS = 50.0


@dataclass(slots=True)
class PanelWake:
    """Precomputed wake geometry and connectivity for the 3D panel solver.

    Attributes
    ----------
    panels : numpy.ndarray
        Coordinates of wake panel corners with shape (Nw_total, 4, 3) in m.
    wake_up : numpy.ndarray
        Indices of upper trailing-edge body panels for each wake strip,
        shape (Nw_strips,).
    wake_lo : numpy.ndarray
        Indices of lower trailing-edge body panels for each wake strip,
        shape (Nw_strips,).
    strip_panel_indices : list of list of int
        List of wake panel indices belonging to each wake strip.
    p_left : numpy.ndarray
        Left endpoint of each wake strip in the Trefftz plane with shape
        (Nw_strips, 3) in m (smaller y coordinate).
    p_right : numpy.ndarray
        Right endpoint of each wake strip in the Trefftz plane with shape
        (Nw_strips, 3) in m (larger y coordinate).
    wake_edges : numpy.ndarray
        Original wake edge array with shape (Nw_strips, 4).
    """

    panels: np.ndarray
    wake_up: np.ndarray
    wake_lo: np.ndarray
    strip_panel_indices: list[list[int]]
    p_left: np.ndarray
    p_right: np.ndarray
    wake_edges: np.ndarray


def largest_streamwise_chord(
    nodes: np.ndarray,
    panels: np.ndarray,
    wake_edges: np.ndarray | None,
    component: np.ndarray | None = None,
) -> float:
    """Return the largest streamwise chord of the lifting surfaces of a mesh.

    For each trailing-edge node of a wake edge, the section nodes are the
    nodes of the same component that lie in the station plane through the
    node (the plane normal to the span direction of the wake strip in the
    y-z plane, within a quarter of the strip width). The streamwise chord
    of the section is the x extent of these nodes.

    Parameters
    ----------
    nodes : numpy.ndarray
        Mesh node coordinates with shape (Nn, 3) in m.
    panels : numpy.ndarray
        Panel corner node indices with shape (Np, 4).
    wake_edges : numpy.ndarray or None
        Wake shedding edges with shape (Ns, 4): upper panel, lower panel,
        node a, node b.
    component : numpy.ndarray or None, optional
        Component index of each panel with shape (Np,). None puts all
        panels in one component.

    Returns
    -------
    float
        Largest streamwise chord in m. Without wake edges, or when no
        section is found, the x extent of the mesh in m.
    """
    pts = np.asarray(nodes, dtype=float)
    pan = np.asarray(panels, dtype=int)
    x_extent = float(np.max(pts[:, 0]) - np.min(pts[:, 0])) if pts.size else 0.0
    w_edges = (
        np.empty((0, 4), dtype=int)
        if wake_edges is None
        else np.asarray(wake_edges, dtype=int).reshape(-1, 4)
    )
    if w_edges.shape[0] == 0 or pan.shape[0] == 0:
        return x_extent
    comp = (
        np.zeros(pan.shape[0], dtype=int)
        if component is None or np.asarray(component).size != pan.shape[0]
        else np.asarray(component, dtype=int).reshape(-1)
    )
    comp_nodes: dict[int, np.ndarray] = {}
    for c in np.unique(comp):
        comp_nodes[int(c)] = np.unique(pan[comp == c].reshape(-1))
    best = 0.0
    for up_idx, _lo_idx, na, nb in w_edges:
        ids = comp_nodes[int(comp[int(up_idx)])]
        sec = pts[ids]
        pa = pts[int(na)]
        pb = pts[int(nb)]
        span = np.array([0.0, pb[1] - pa[1], pb[2] - pa[2]])
        width = float(np.linalg.norm(span))
        if width <= 1.0e-12:
            continue
        s_hat = span / width
        for te in (pa, pb):
            dist = (sec - te[None, :]) @ s_hat
            on_plane = sec[np.abs(dist) <= 0.25 * width]
            if on_plane.shape[0] >= 2:
                best = max(best, float(np.max(on_plane[:, 0]) - np.min(on_plane[:, 0])))
    return best if best > 0.0 else x_extent


def build_wake(
    nodes: np.ndarray,
    panels: np.ndarray,
    wake_edges: np.ndarray | None,
    normals: np.ndarray,
    *,
    wake_length: float | None = None,
    n_wake: int = 1,
    wake_dir: np.ndarray | None = None,
    component: np.ndarray | None = None,
) -> PanelWake:
    """Build flat wake panels shed along a given direction from trailing edges.

    Parameters
    ----------
    nodes : numpy.ndarray
        Mesh node coordinates with shape (Nn, 3) in m.
    panels : numpy.ndarray
        Mesh panel corner node indices with shape (Np, 4).
    wake_edges : numpy.ndarray or None
        Wake edge definitions with shape (Nw_strips, 4): upper panel index,
        lower panel index, node a, node b.
    normals : numpy.ndarray
        Unit normal vectors of body panels with shape (Np, 3).
    wake_length : float or None, optional
        Length of the wake in m. If None, defaults to 50 times the largest
        streamwise chord of the mesh (see :func:`largest_streamwise_chord`).
    n_wake : int, default=1
        Number of panels along each wake strip. Must be at least 1.
    wake_dir : numpy.ndarray or None, optional
        Unit direction vector along which wake panels extend. If None,
        defaults to the body +x axis ([1.0, 0.0, 0.0]). The solver gives
        the free-stream direction in ground effect.
    component : numpy.ndarray or None, optional
        Component index of each panel with shape (Np,), used only for the
        default wake length.

    Returns
    -------
    PanelWake
        Precomputed wake panels and strip connectivity.
    """
    if n_wake < 1:
        raise ValueError(f"n_wake must be at least 1, got {n_wake}.")

    w_edges = (
        np.empty((0, 4), dtype=int)
        if wake_edges is None
        else np.asarray(wake_edges, dtype=int).reshape(-1, 4)
    )
    n_strips = w_edges.shape[0]

    if n_strips == 0:
        return PanelWake(
            panels=np.empty((0, 4, 3), dtype=float),
            wake_up=np.empty(0, dtype=int),
            wake_lo=np.empty(0, dtype=int),
            strip_panel_indices=[],
            p_left=np.empty((0, 3), dtype=float),
            p_right=np.empty((0, 3), dtype=float),
            wake_edges=w_edges,
        )

    # Determine wake length if not given
    if wake_length is None:
        chord_estimate = max(largest_streamwise_chord(nodes, panels, w_edges, component), 1.0e-3)
        wake_len = DEFAULT_WAKE_CHORDS * chord_estimate
    else:
        wake_len = float(wake_length)
        if not np.isfinite(wake_len) or wake_len <= 0.0:
            raise ValueError(f"wake_length must be positive and finite, got {wake_len}.")

    x_stations = np.linspace(0.0, wake_len, n_wake + 1)
    if wake_dir is None:
        x_hat = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        d_arr = np.asarray(wake_dir, dtype=float).reshape(3)
        d_norm = float(np.linalg.norm(d_arr))
        if d_norm <= 1.0e-12:
            raise ValueError(f"wake_dir must be a non-zero vector, got {wake_dir}.")
        x_hat = d_arr / d_norm

    wake_panels_list = []
    strip_indices = []
    p_left_list = []
    p_right_list = []
    wake_up = np.empty(n_strips, dtype=int)
    wake_lo = np.empty(n_strips, dtype=int)

    panel_counter = 0
    for k in range(n_strips):
        up_idx, lo_idx, na, nb = w_edges[k]
        wake_up[k] = up_idx
        wake_lo[k] = lo_idx

        pa = np.asarray(nodes[na], dtype=float)
        pb = np.asarray(nodes[nb], dtype=float)
        n_up = normals[up_idx]

        # Order endpoints so that p_left has smaller y coordinate
        if pa[1] < pb[1]:
            p_l, p_r = pa, pb
        else:
            p_l, p_r = pb, pa
        p_left_list.append(p_l)
        p_right_list.append(p_r)

        strip_p_idx = []
        for m in range(n_wake):
            x0 = x_stations[m]
            x1 = x_stations[m + 1]

            q1 = pa + x0 * x_hat
            q2 = pa + x1 * x_hat
            q3 = pb + x1 * x_hat
            q4 = pb + x0 * x_hat

            # Ensure wake normal points in the same direction as upper panel normal
            n_raw = np.cross(q3 - q1, q4 - q2)
            if np.dot(n_raw, n_up) < 0.0:
                q1, q2, q3, q4 = q1, q4, q3, q2

            wake_panels_list.append([q1, q2, q3, q4])
            strip_p_idx.append(panel_counter)
            panel_counter += 1

        strip_indices.append(strip_p_idx)

    return PanelWake(
        panels=np.array(wake_panels_list, dtype=float),
        wake_up=wake_up,
        wake_lo=wake_lo,
        strip_panel_indices=strip_indices,
        p_left=np.array(p_left_list, dtype=float),
        p_right=np.array(p_right_list, dtype=float),
        wake_edges=w_edges,
    )
