# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Geometric reflection and image methods for Ventorum-3DP.

This module provides symmetry folding about the y = 0 plane and ground-effect
image representations for constant-strength source-doublet panels.

Published sources
-----------------
* Katz, J. and Plotkin, A., 2001, Low-Speed Aerodynamics, 2nd ed.,
  Cambridge University Press, section 9.4 and chapter 12.
* Morino, L. and Kuo, C. C., 1974, Subsonic potential aerodynamics for
  complex configurations: A general theory, AIAA Journal, Vol. 12, No. 2,
  pp. 191-197.
* Wieselsberger, C., 1921, Ueber den Fluegelwiderstand in der Naehe des Bodens,
  Zeitschrift fuer Flugtechnik und Motorluftschiffahrt, Vol. 12, No. 10.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from ventorum.aero.system import ground_normal
from ventorum.core.datatypes import FlightCondition
from ventorum.core.errors import GroundStrikeError
from ventorum.panel3d.mesh import PanelMesh

__all__ = [
    "check_ground_clearance",
    "ground_plane_parameters",
    "mirror_panels_y",
    "mirror_points_y",
    "reconstruct_full_mesh",
    "reflect_panels_plane",
    "reflect_points_plane",
    "standardize_panel_corners",
]


def mirror_points_y(pts: np.ndarray) -> np.ndarray:
    """Reflect 3D coordinates across the y = 0 plane.

    Parameters
    ----------
    pts : numpy.ndarray
        Array of coordinates with shape (..., 3) in m.

    Returns
    -------
    numpy.ndarray
        Reflected coordinates with y inverted in m.
    """
    out = np.array(pts, dtype=float, copy=True)
    out[..., 1] = -out[..., 1]
    return out


def reflect_points_plane(
    pts: np.ndarray,
    normal: np.ndarray,
    offset: float,
) -> np.ndarray:
    """Reflect 3D coordinates across an arbitrary plane.

    The plane equation is dot(x, normal) = offset, where normal is a unit vector.

    Parameters
    ----------
    pts : numpy.ndarray
        Array of coordinates with shape (..., 3) in m.
    normal : numpy.ndarray
        Unit normal vector of the plane with shape (3,).
    offset : float
        Plane offset constant in m.

    Returns
    -------
    numpy.ndarray
        Reflected coordinates in m.
    """
    p = np.asarray(pts, dtype=float)
    n = np.asarray(normal, dtype=float)
    h = np.sum(p * n, axis=-1, keepdims=True) - float(offset)
    return p - 2.0 * h * n


def standardize_panel_corners(panels: np.ndarray) -> np.ndarray:
    """Standardize triangle panels to repeat the third vertex in the fourth position.

    Parameters
    ----------
    panels : numpy.ndarray
        Panel corner node coordinates with shape (M, 4, 3) in m.

    Returns
    -------
    numpy.ndarray
        Standardized panel corner coordinates with shape (M, 4, 3) in m.
    """
    out = np.array(panels, dtype=float, copy=True)
    m = out.shape[0]
    for i in range(m):
        row = out[i]
        # Check if already standardized (corner 2 close to corner 3)
        if np.allclose(row[2], row[3], atol=1e-12):
            continue
        # Search which adjacent pair is identical
        for shift in range(1, 4):
            shifted = np.roll(row, -shift, axis=0)
            if np.allclose(shifted[2], shifted[3], atol=1e-12):
                out[i] = shifted
                break
    return out


def mirror_panels_y(panels: np.ndarray) -> np.ndarray:
    """Reflect panel coordinates across y = 0 with outward normal preservation.

    Parameters
    ----------
    panels : numpy.ndarray
        Panel corner coordinates with shape (M, 4, 3) in m.

    Returns
    -------
    numpy.ndarray
        Mirrored panel coordinates with reversed vertex order in m.
    """
    p_mirr = mirror_points_y(panels)
    # Reverse vertex order [0, 3, 2, 1] to keep the outward normal pointing outward
    p_rev = p_mirr[:, [0, 3, 2, 1], :]
    return standardize_panel_corners(p_rev)


def reflect_panels_plane(
    panels: np.ndarray,
    normal: np.ndarray,
    offset: float,
) -> np.ndarray:
    """Reflect panel coordinates across an arbitrary plane with outward normal preservation.

    Parameters
    ----------
    panels : numpy.ndarray
        Panel corner coordinates with shape (M, 4, 3) in m.
    normal : numpy.ndarray
        Unit normal vector of the plane with shape (3,).
    offset : float
        Plane offset constant in m.

    Returns
    -------
    numpy.ndarray
        Reflected panel coordinates with reversed vertex order in m.
    """
    p_ref = reflect_points_plane(panels, normal, offset)
    p_rev = p_ref[:, [0, 3, 2, 1], :]
    return standardize_panel_corners(p_rev)


def ground_plane_parameters(
    condition: FlightCondition,
    ref_point: np.ndarray | None = None,
) -> tuple[np.ndarray, float] | None:
    """Compute ground plane unit normal and offset for a flight condition.

    Parameters
    ----------
    condition : FlightCondition
        Flight condition with optional height condition.h.
    ref_point : numpy.ndarray or None, optional
        Moment reference center in geometry axes with shape (3,) in m.

    Returns
    -------
    tuple of (numpy.ndarray, float) or None
        (normal, offset) of the ground plane in m, or None if condition.h is None.

    Raises
    ------
    ValueError
        If condition.h is not strictly positive.
    """
    if condition.h is None:
        return None
    h = float(condition.h)
    if h <= 0.0:
        raise ValueError(f"Ground height h={h} must be strictly positive.")
    alpha = float(condition.alpha)
    beta = float(condition.beta)
    phi = float(getattr(condition, "phi", 0.0))
    k = ground_normal(alpha, beta, phi)
    rp = np.zeros(3, dtype=float) if ref_point is None else np.asarray(ref_point, dtype=float)
    offset = float(np.dot(rp, k) - h)
    return k, offset


def check_ground_clearance(
    nodes: np.ndarray,
    normal: np.ndarray,
    offset: float,
) -> float:
    """Verify that all nodes remain strictly above the ground plane.

    Parameters
    ----------
    nodes : numpy.ndarray
        Node coordinates with shape (N, 3) in m.
    normal : numpy.ndarray
        Ground unit normal vector with shape (3,).
    offset : float
        Ground plane offset in m.

    Returns
    -------
    float
        Minimum clearance of any node above the ground in m.

    Raises
    ------
    GroundStrikeError
        If any node touches or crosses the ground plane (a ValueError).
    """
    h = np.asarray(nodes, dtype=float).reshape(-1, 3) @ np.asarray(normal, dtype=float) - float(offset)
    min_h = float(np.min(h))
    if min_h <= 0.0:
        raise GroundStrikeError(
            f"Geometry touches or crosses the ground plane (minimum clearance {min_h:.4e} m). "
            "No solution is computed for a ground strike."
        )
    return min_h


def reconstruct_full_mesh(half_mesh: PanelMesh) -> PanelMesh:
    """Reconstruct full symmetric PanelMesh from a half mesh.

    Parameters
    ----------
    half_mesh : PanelMesh
        Right-half surface mesh.

    Returns
    -------
    PanelMesh
        Full mesh with both halves and half=False.
    """
    if not half_mesh.half:
        return half_mesh

    nodes = list(half_mesh.nodes)
    right_n = len(nodes)
    index_map = np.full(right_n, -1, dtype=int)

    for i in range(right_n):
        if abs(nodes[i][1]) < 1.0e-12:
            index_map[i] = i
        else:
            p = np.array(nodes[i], dtype=float)
            p[1] = -p[1]
            nodes.append(p)
            index_map[i] = len(nodes) - 1

    panels = list(half_mesh.panels)
    shift_pan = len(half_mesh.panels)
    for quad in half_mesh.panels:
        p1, p2, p3, p4 = quad
        panels.append([
            int(index_map[p1]),
            int(index_map[p4]),
            int(index_map[p3]),
            int(index_map[p2]),
        ])

    wake_edges = list(half_mesh.wake_edges)
    if len(half_mesh.wake_edges) > 0:
        for w in half_mesh.wake_edges:
            up_idx, lo_idx, na, nb = w
            wake_edges.append([
                int(up_idx + shift_pan),
                int(lo_idx + shift_pan),
                int(index_map[na]),
                int(index_map[nb]),
            ])

    comp = np.concatenate([half_mesh.component, half_mesh.component])
    kind = np.concatenate([half_mesh.kind, half_mesh.kind])

    # Other per-panel metadata of the mesh (for example a strip index) is
    # repeated for the mirror panels.
    extra = {}
    n_half = len(half_mesh.panels)
    known = {"nodes", "panels", "component", "kind", "wake_edges", "half"}
    for fld in dataclasses.fields(half_mesh):
        if fld.name in known or not fld.init:
            continue
        value = getattr(half_mesh, fld.name)
        arr = np.asarray(value) if value is not None else None
        if arr is not None and arr.ndim >= 1 and arr.shape[0] == n_half:
            extra[fld.name] = np.concatenate([arr, arr])
        else:
            extra[fld.name] = value

    return PanelMesh(
        nodes=np.array(nodes, dtype=float),
        panels=np.array(panels, dtype=int),
        component=comp,
        kind=kind,
        wake_edges=np.array(wake_edges, dtype=int).reshape(-1, 4),
        half=False,
        **extra,
    )
