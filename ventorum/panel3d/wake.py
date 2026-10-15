# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Wake shedding and wake geometry for the 3D panel method.

This module builds the flat body-axis wake panels that shed from the trailing
edges of lifting surfaces in Ventorum-3DP.

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
* Wake panels extend along the body +x axis ([1.0, 0.0, 0.0]).
* The panel normal vector is oriented to point in the same direction as the
  outward normal of the upper trailing-edge panel.
* Each wake strip carries doublet strength mu_w = mu_upper - mu_lower.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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


def build_wake(
    nodes: np.ndarray,
    panels: np.ndarray,
    wake_edges: np.ndarray | None,
    normals: np.ndarray,
    *,
    wake_length: float | None = None,
    n_wake: int = 1,
) -> PanelWake:
    """Build flat wake panels shed along the body +x axis from trailing edges.

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
        chord of the mesh.
    n_wake : int, default=1
        Number of panels along each wake strip. Must be at least 1.

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
        x_min = float(np.min(nodes[:, 0]))
        x_max = float(np.max(nodes[:, 0]))
        chord_estimate = max(x_max - x_min, 1.0e-3)
        wake_len = 50.0 * chord_estimate
    else:
        wake_len = float(wake_length)
        if wake_len <= 0.0:
            raise ValueError(f"wake_length must be positive, got {wake_len}.")

    x_stations = np.linspace(0.0, wake_len, n_wake + 1)
    x_hat = np.array([1.0, 0.0, 0.0], dtype=float)

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
