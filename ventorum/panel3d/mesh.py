# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Closed quadrilateral meshes of thick lifting surfaces for the 3D panel method.

The mesh wraps a thick section around each lifting surface. Each spanwise
station carries the true airfoil shape (upper surface, lower surface, blunt
base at an open trailing edge, flat tip cap). The panels are plane
quadrilaterals with outward corner order, and the trailing-edge strips carry
the wake shedding edges for the solver.

Published sources
-----------------
* Hess, J. L. and Smith, A. M. O., 1964, Calculation of potential flow about
  arbitrary bodies, Journal of Ship Research, Vol. 8 (quadrilateral
  source-doublet panels on a closed body, outward normal convention).
* Katz, J. and Plotkin, A., 1991, Low-Speed Aerodynamics, McGraw-Hill,
  Chapters 11 and 12 (panel corner order, plane quadrilateral panels, signed
  volume of a closed mesh).
* Phillips, W. F. and Snyder, D. O., 2000, Modern adaptation of Prandtl's
  classic lifting-line theory, Journal of Aircraft, Vol. 37 (twist rotation
  about the quarter chord and the leading-edge reference line, shared here
  with the lattice builder so the thin-surface limit matches the VLM exactly).

Coordinate conventions
----------------------
Geometry axes: x aft [m], y right [m], z up [m]. Airfoil input arrays use the
Selig order: from the trailing edge along the upper surface to the leading
edge, then along the lower surface back to the trailing edge. Unit chord with
the leading edge at (0, 0) and the trailing-edge midpoint at (1, 0). A closed
trailing edge repeats the same point at the start and the end.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ventorum.geometry.lattice import _surface_eta, resolve_spacing, surface_reference_line

__all__ = ["PanelMesh", "check_mesh", "mesh_aircraft", "mesh_lifting_surface"]

#: Trailing-edge gap over local chord below which the edge counts as sharp.
_TE_GAP_TOL = 1e-6
_XHAT = np.array([1.0, 0.0, 0.0])
_ZHAT = np.array([0.0, 0.0, 1.0])


@dataclass(slots=True)
class PanelMesh:
    """Closed quadrilateral mesh of one or more thick lifting surfaces.

    Attributes
    ----------
    nodes : numpy.ndarray
        Node positions [m], shape (Nn, 3).
    panels : numpy.ndarray
        Panel corner node indices, shape (Np, 4). A triangle repeats its
        third node in the fourth entry.
    component : numpy.ndarray
        Component index of each panel, shape (Np,).
    kind : numpy.ndarray
        Panel kind codes, shape (Np,): "surface", "tip" or "base".
    wake_edges : numpy.ndarray
        Wake shedding edges, shape (Nw, 4): upper trailing-edge panel index,
        lower trailing-edge panel index, node a, node b. Nodes a and b span
        one spanwise strip at the trailing edge (the upper edge when the
        trailing edge is blunt).
    half : bool
        True for a right-half mesh only, open at y = 0 [m].
    """

    nodes: np.ndarray
    panels: np.ndarray
    component: np.ndarray
    kind: np.ndarray
    wake_edges: np.ndarray
    half: bool = False


def _target_x(n_chord: int, chord_spacing: str) -> np.ndarray:
    """Chordwise stations in unit chord for one surface, from 0 to 1."""
    m = str(chord_spacing).lower().replace("_", "-")
    if m in ("cosine", "full-cosine"):
        k = np.arange(n_chord + 1, dtype=float)
        return 0.5 * (1.0 - np.cos(np.pi * k / float(n_chord)))
    if m == "uniform":
        return np.linspace(0.0, 1.0, n_chord + 1)
    raise ValueError(f"Unknown chord_spacing {chord_spacing!r}; use 'cosine' or 'uniform'.")


def _split_selig(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a Selig array into upper (ascending x) and lower (ascending x) parts."""
    a = np.asarray(coords, dtype=float)
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError("Each section_coords array must have shape (N, 2).")
    if a.shape[0] < 4:
        raise ValueError("Each section_coords array needs at least 4 points.")
    if not np.all(np.isfinite(a)):
        raise ValueError("section_coords holds non-finite values.")
    i_le = int(np.argmin(a[:, 0]))
    if i_le == 0 or i_le == a.shape[0] - 1:
        raise ValueError("section_coords has no leading-edge minimum inside the array.")
    return a[: i_le + 1][::-1], a[i_le:]


def _resample_section(coords: np.ndarray, x_target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resample one section to the target x stations; interpolate y on x per surface."""
    up, lo = _split_selig(coords)
    return np.interp(x_target, up[:, 0], up[:, 1]), np.interp(x_target, lo[:, 0], lo[:, 1])


def _rotate_vecs(v: np.ndarray, axis: np.ndarray, angle: np.ndarray | float) -> np.ndarray:
    """Rodrigues rotation of row vectors *v* about unit axes *axis*."""
    v = np.asarray(v, dtype=float)
    axis = np.asarray(axis, dtype=float)
    ang = np.broadcast_to(np.asarray(angle, dtype=float), (v.shape[0],))
    c = np.cos(ang)[:, None]
    s = np.sin(ang)[:, None]
    kdv = np.sum(axis * v, axis=1)[:, None]
    return v * c + np.cross(axis, v) * s + axis * kdv * (1.0 - c)


def _interp_rows(fr: np.ndarray, mat: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """Linear interpolation of each column of *mat* (sections by rows) at *eta*."""
    fr = np.asarray(fr, dtype=float)
    out = np.empty((len(eta), mat.shape[1]), dtype=float)
    for k in range(mat.shape[1]):
        out[:, k] = np.interp(eta, fr, mat[:, k])
    return out


def _as_coord_list(section_coords: np.ndarray | Sequence[np.ndarray], n_sec: int) -> list[np.ndarray]:
    """Return one coordinate array per section."""
    if isinstance(section_coords, np.ndarray):
        return [np.asarray(section_coords, dtype=float) for _ in range(n_sec)]
    items = list(section_coords)
    if len(items) != n_sec:
        raise ValueError(f"section_coords has {len(items)} arrays for {n_sec} sections.")
    return [np.asarray(c, dtype=float) for c in items]


def mesh_lifting_surface(
    surface,
    section_coords,
    n_chord: int = 40,
    chord_spacing: str = "cosine",
    tip: str = "flat",
    half: bool = False,
    n_span=None,
) -> PanelMesh:
    """Build a closed quadrilateral mesh of one thick lifting surface.

    Parameters
    ----------
    surface : LiftingSurface
        Surface definition (planform, twist, sweep, dihedral [rad]).
    section_coords : numpy.ndarray or list of numpy.ndarray
        One (N, 2) unit-chord Selig array for all sections, or one array
        per section in the order of ``surface.sections``.
    n_chord : int
        Chordwise panels per surface (upper and lower).
    chord_spacing : str
        "cosine" (clustered at leading and trailing edge) or "uniform" in x.
    tip : str
        "flat" (flat tip cap) or "none" (open tip, only for tests).
    half : bool
        True keeps only the right half, open at y = 0 [m]. Without effect
        on a non-symmetric surface (it is already one side).
    n_span : int or None
        Spanwise panels per semi-span. None uses the surface value when set,
        else 20. The stations match the lattice builder for that count.

    Returns
    -------
    PanelMesh
        Closed mesh with outward panels and wake shedding edges.
    """
    surf = surface
    n_chord = int(n_chord)
    if n_chord < 1:
        raise ValueError(f"n_chord must be at least 1, got {n_chord}.")
    if tip not in ("flat", "none"):
        raise ValueError(f"Unknown tip {tip!r}; use 'flat' or 'none'.")
    secs = list(surf.sections)
    if len(secs) < 2:
        raise ValueError("A lifting surface needs at least two sections.")
    if n_span is None:
        n_span_eff = int(surf.n_panels) if surf.n_panels is not None else 20
    else:
        n_span_eff = int(n_span)
    if n_span_eff < 1:
        raise ValueError(f"n_span must be at least 1, got {n_span_eff}.")

    order = sorted(range(len(secs)), key=lambda i: float(secs[i].y_frac))
    secs_sorted = [secs[i] for i in order]
    coord_list = _as_coord_list(section_coords, len(secs))
    coord_sorted = [coord_list[i] for i in order]
    fr = np.array([float(s.y_frac) for s in secs_sorted])
    chords = np.array([float(s.chord) for s in secs_sorted])
    twists = np.array([float(s.twist) for s in secs_sorted])

    xt = _target_x(n_chord, chord_spacing)
    nc1 = n_chord + 1
    yu_rows = []
    yl_rows = []
    for c in coord_sorted:
        yu, yl = _resample_section(c, xt)
        yu_rows.append(yu)
        yl_rows.append(yl)
    yu_mat = np.vstack(yu_rows)
    yl_mat = np.vstack(yl_rows)

    spacing = resolve_spacing(surf.spacing, surf, "vlm")
    eta_e, _ = _surface_eta(surf, n_span_eff, spacing)
    eta_e = np.asarray(eta_e, dtype=float)
    ns = len(eta_e)
    chord_e = np.interp(eta_e, fr, chords)
    twist_e = np.interp(eta_e, fr, twists) + float(surf.incidence)
    x_ref, y_ref, z_ref = surface_reference_line(surf, eta_e)
    offset = np.asarray(surf.position, dtype=float).reshape(3)
    le_flat = np.column_stack([x_ref, y_ref, z_ref]) + offset
    if ns >= 2:
        dy = np.gradient(y_ref, eta_e)
        dz = np.gradient(z_ref, eta_e)
    else:  # pragma: no cover - one spanwise panel still has two edge stations
        dy = np.zeros_like(eta_e)
        dz = np.zeros_like(eta_e)
    axis = np.column_stack([np.zeros_like(eta_e), dy, dz])
    axis /= np.maximum(np.linalg.norm(axis, axis=1, keepdims=True), 1e-300)
    if surf.is_symmetric:
        axis[eta_e <= 1e-12] = np.array([0.0, 1.0, 0.0])
    cdir = _rotate_vecs(np.tile(_XHAT, (ns, 1)), axis, twist_e)
    zdir = _rotate_vecs(np.tile(_ZHAT, (ns, 1)), axis, twist_e)
    qc = le_flat + 0.25 * chord_e[:, None] * _XHAT
    yu_e = _interp_rows(fr, yu_mat, eta_e)
    yl_e = _interp_rows(fr, yl_mat, eta_e)

    upper = np.empty((ns, nc1, 3))
    lower = np.empty((ns, nc1, 3))
    for j in range(ns):
        upper[j] = qc[j] + chord_e[j] * ((xt - 0.25)[:, None] * cdir[j] + yu_e[j][:, None] * zdir[j])
        lower[j] = qc[j] + chord_e[j] * ((xt - 0.25)[:, None] * cdir[j] + yl_e[j][:, None] * zdir[j])
    sharp = np.abs(yu_e[:, -1] - yl_e[:, -1]) <= _TE_GAP_TOL

    nodes: list[np.ndarray] = []
    station_of: list[int] = []

    def add_node(p: np.ndarray, station: int) -> int:
        nodes.append(np.asarray(p, dtype=float).reshape(3))
        station_of.append(int(station))
        return len(nodes) - 1

    uid = np.full((ns, nc1), -1, dtype=int)
    lid = np.full((ns, nc1), -1, dtype=int)
    for j in range(ns):
        uid[j, 0] = lid[j, 0] = add_node(0.5 * (upper[j, 0] + lower[j, 0]), j)
        for k in range(1, n_chord):
            uid[j, k] = add_node(upper[j, k], j)
            lid[j, k] = add_node(lower[j, k], j)
        if sharp[j]:
            uid[j, n_chord] = lid[j, n_chord] = add_node(0.5 * (upper[j, n_chord] + lower[j, n_chord]), j)
        else:
            uid[j, n_chord] = add_node(upper[j, n_chord], j)
            lid[j, n_chord] = add_node(lower[j, n_chord], j)

    panels: list[list[int]] = []
    kinds: list[str] = []
    up_pan = np.full((ns - 1, n_chord), -1, dtype=int)
    lo_pan = np.full((ns - 1, n_chord), -1, dtype=int)

    def add_panel(p1: int, p2: int, p3: int, p4: int, kind: str) -> int:
        panels.append([int(p1), int(p2), int(p3), int(p4)])
        kinds.append(kind)
        return len(panels) - 1

    for j in range(ns - 1):
        for k in range(n_chord):
            up_pan[j, k] = add_panel(uid[j, k], uid[j, k + 1], uid[j + 1, k + 1], uid[j + 1, k], "surface")
            lo_pan[j, k] = add_panel(lid[j, k], lid[j + 1, k], lid[j + 1, k + 1], lid[j, k + 1], "surface")
        if sharp[j] and sharp[j + 1]:
            pass
        else:
            add_panel(uid[j, n_chord], lid[j, n_chord], lid[j + 1, n_chord], uid[j + 1, n_chord], "base")

    def add_cap(station: int, flip: bool) -> None:
        """Flat cap of the airfoil loop at one spanwise station."""
        u = uid[station]
        v = lid[station]
        if n_chord == 1:
            q: list[tuple[int, int, int, int]] = [(u[0], u[1], v[1], v[1])]
        else:
            q = [(u[0], u[1], v[1], v[1])]
            for k in range(1, n_chord - 1):
                q.append((u[k], u[k + 1], v[k + 1], v[k]))
            if sharp[station]:
                q.append((u[n_chord - 1], u[n_chord], v[n_chord - 1], v[n_chord - 1]))
            else:
                q.append((u[n_chord - 1], u[n_chord], v[n_chord], v[n_chord - 1]))
        for p1, p2, p3, p4 in q:
            if flip:
                add_panel(p1, p4, p3, p2, "tip")
            else:
                add_panel(p1, p2, p3, p4, "tip")

    symmetric = bool(surf.is_symmetric)
    if tip == "flat":
        add_cap(ns - 1, flip=False)
        if not symmetric:
            add_cap(0, flip=True)

    wake: list[list[int]] = []
    for j in range(ns - 1):
        wake.append([int(up_pan[j, n_chord - 1]), int(lo_pan[j, n_chord - 1]), int(uid[j, n_chord]), int(uid[j + 1, n_chord])])

    right_n = len(nodes)
    right_panels = list(panels)
    right_kinds = list(kinds)
    right_station = list(station_of)
    up_left = np.full_like(up_pan, -1)
    lo_left = np.full_like(lo_pan, -1)

    mirror = bool(getattr(surf, "mirror_y", False))
    if symmetric and not half:
        index_map = np.full(right_n, -1, dtype=int)
        for i in range(right_n):
            if right_station[i] == 0:
                index_map[i] = i
            else:
                p = np.asarray(nodes[i], dtype=float).copy()
                p[1] = -p[1]
                nodes.append(p)
                station_of.append(right_station[i])
                index_map[i] = len(nodes) - 1
        for pi, quad in enumerate(right_panels):
            p1, p2, p3, p4 = quad
            panels.append([int(index_map[p1]), int(index_map[p4]), int(index_map[p3]), int(index_map[p2])])
            kinds.append(right_kinds[pi])
        # Map left panel indices by rebuilding the same order: upper row then lower row per strip.
        left_of = {}
        shift = len(right_panels)
        for offset, pi in enumerate(range(len(right_panels))):
            left_of[pi] = shift + offset
        for j in range(ns - 1):
            for k in range(n_chord):
                up_left[j, k] = left_of[int(up_pan[j, k])]
                lo_left[j, k] = left_of[int(lo_pan[j, k])]
        for j in range(ns - 1):
            wake.append([
                int(up_left[j, n_chord - 1]),
                int(lo_left[j, n_chord - 1]),
                int(index_map[int(uid[j, n_chord])]),
                int(index_map[int(uid[j + 1, n_chord])]),
            ])
        half_flag = False
    elif mirror and not symmetric:
        for i in range(right_n):
            p = np.asarray(nodes[i], dtype=float).copy()
            p[1] = -p[1]
            nodes[i] = p
        panels = [[q[0], q[3], q[2], q[1]] for q in panels]
        half_flag = False
    else:
        half_flag = bool(half and symmetric)

    node_arr = np.vstack(nodes)
    panel_arr = np.array(panels, dtype=int)
    wake_arr = np.array(wake, dtype=int).reshape(-1, 4)
    return PanelMesh(
        nodes=node_arr,
        panels=panel_arr,
        component=np.zeros(panel_arr.shape[0], dtype=int),
        kind=np.array(kinds, dtype=str),
        wake_edges=wake_arr,
        half=half_flag,
    )


def mesh_aircraft(aircraft, section_coords_by_surface, **kw) -> PanelMesh:
    """Build one panel mesh for all surfaces of an aircraft.

    Parameters
    ----------
    aircraft : Aircraft
        Aircraft with the lifting surfaces.
    section_coords_by_surface : dict or list
        Dict from surface name (or index) to the section coordinates of
        :func:`mesh_lifting_surface`, or a list in surface order.
    **kw
        Extra arguments of :func:`mesh_lifting_surface` (n_chord, n_span,
        chord_spacing, tip, half).

    Returns
    -------
    PanelMesh
        Combined mesh with one component index per surface.
    """
    surfs = list(aircraft.surfaces)
    if isinstance(section_coords_by_surface, dict):
        coord_list = []
        for i, s in enumerate(surfs):
            if s.name in section_coords_by_surface:
                coord_list.append(section_coords_by_surface[s.name])
            elif i in section_coords_by_surface:
                coord_list.append(section_coords_by_surface[i])
            elif str(i) in section_coords_by_surface:
                coord_list.append(section_coords_by_surface[str(i)])
            else:
                raise KeyError(f"No section coordinates for surface {s.name!r}.")
    else:
        coord_list = list(section_coords_by_surface)
        if len(coord_list) != len(surfs):
            raise ValueError(f"Got {len(coord_list)} coordinate sets for {len(surfs)} surfaces.")
    node_parts = []
    panel_parts = []
    kind_parts = []
    comp_parts = []
    wake_parts = []
    n_nodes = 0
    n_panels = 0
    half_all = True
    for i, (s, c) in enumerate(zip(surfs, coord_list)):
        sub = mesh_lifting_surface(s, c, **kw)
        node_parts.append(np.asarray(sub.nodes, dtype=float))
        panel_parts.append(np.asarray(sub.panels, dtype=int) + n_nodes)
        kind_parts.append(np.asarray(sub.kind))
        comp_parts.append(np.full(np.asarray(sub.panels).shape[0], i, dtype=int))
        w = np.asarray(sub.wake_edges, dtype=int).copy()
        if w.size:
            w[:, 0:2] += n_panels
            w[:, 2:4] += n_nodes
        wake_parts.append(w.reshape(-1, 4))
        n_nodes += int(np.asarray(sub.nodes).shape[0])
        n_panels += int(np.asarray(sub.panels).shape[0])
        half_all = half_all and bool(sub.half)
    return PanelMesh(
        nodes=np.vstack(node_parts),
        panels=np.vstack(panel_parts),
        component=np.concatenate(comp_parts),
        kind=np.concatenate(kind_parts),
        wake_edges=np.vstack(wake_parts).reshape(-1, 4),
        half=bool(half_all and len(surfs) > 0),
    )


def _panel_geometry(p: np.ndarray) -> tuple[float, np.ndarray, float, float, float]:
    """Area [m^2], unit normal, aspect ratio, skew and warp of one panel."""
    p1, p2, p3, p4 = (np.asarray(q, dtype=float).reshape(3) for q in p)
    n = np.cross(p3 - p1, p4 - p2)
    n_norm = float(np.linalg.norm(n))
    if n_norm <= 1e-300:
        n = np.cross(p2 - p1, p3 - p1)
        n_norm = float(np.linalg.norm(n))
    if n_norm <= 1e-300:
        n = np.cross(p3 - p1, p4 - p1)
        n_norm = float(np.linalg.norm(n))
    unit = n / n_norm if n_norm > 1e-300 else np.zeros(3)
    area = 0.5 * float(np.linalg.norm(np.cross(p2 - p1, p3 - p1)))
    area += 0.5 * float(np.linalg.norm(np.cross(p3 - p1, p4 - p1)))
    uniq = [p1, p2, p3, p4]
    loop = []
    for q in uniq:
        if not loop or float(np.linalg.norm(q - loop[-1])) > 0.0:
            loop.append(q)
    if len(loop) > 1 and float(np.linalg.norm(loop[-1] - loop[0])) == 0.0:
        loop.pop()
    edges = []
    for i in range(len(loop)):
        edges.append(float(np.linalg.norm(loop[(i + 1) % len(loop)] - loop[i])))
    edges = [e for e in edges if e > 0.0]
    aspect = float(max(edges) / min(edges)) if len(edges) >= 2 else 1.0
    skew = 0.0
    m = len(loop)
    for i in range(m):
        a = loop[i - 1] - loop[i]
        b = loop[(i + 1) % m] - loop[i]
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na > 0.0 and nb > 0.0:
            skew = max(skew, abs(float(np.dot(a, b)) / (na * nb)))
    pts = np.vstack(loop)
    diam = 0.0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            diam = max(diam, float(np.linalg.norm(pts[i] - pts[j])))
    if diam > 0.0 and float(np.linalg.norm(unit)) > 0.0:
        mean = np.mean(pts, axis=0)
        warp = float(np.max(np.abs((pts - mean) @ unit)) / diam)
    else:
        warp = 0.0
    return area, unit, aspect, float(skew), float(warp)


def check_mesh(mesh: PanelMesh) -> dict:
    """Check closure, orientation and quality of a panel mesh.

    Parameters
    ----------
    mesh : PanelMesh
        Mesh to check.

    Returns
    -------
    dict
        watertight (every edge is shared by exactly two panels, except the
        root edges of a half mesh and the open tip of tip "none"), outward
        (positive signed volume), volume [m^3], min_area [m^2], per-panel
        aspect_ratio, skew and warp arrays with their max values, and
        n_open_edges.
    """
    nodes = np.asarray(mesh.nodes, dtype=float)
    panels = np.asarray(mesh.panels, dtype=int)
    n_p = panels.shape[0]
    areas = np.empty(n_p)
    aspects = np.empty(n_p)
    skews = np.empty(n_p)
    warps = np.empty(n_p)
    volume = 0.0
    for i in range(n_p):
        p = nodes[panels[i]]
        area, _, aspects[i], skews[i], warps[i] = _panel_geometry(p)
        areas[i] = area
        p1, p2, p3, p4 = (np.asarray(q, dtype=float).reshape(3) for q in p)
        volume += float(np.dot(p1, np.cross(p2, p3)) + np.dot(p1, np.cross(p3, p4)))
    volume /= 6.0
    counts: dict[tuple[int, int], int] = {}
    owners: dict[tuple[int, int], int] = {}
    for i in range(n_p):
        row = panels[i]
        for a, b in ((row[0], row[1]), (row[1], row[2]), (row[2], row[3]), (row[3], row[0])):
            a, b = int(a), int(b)
            if a == b:
                continue
            key = (min(a, b), max(a, b))
            counts[key] = counts.get(key, 0) + 1
            owners.setdefault(key, i)
    comp_of = np.asarray(mesh.component, dtype=int).reshape(-1) if np.asarray(mesh.component).size else np.zeros(n_p, dtype=int)
    comp_nodes: dict[int, list[int]] = {}
    for i in range(n_p):
        c = int(comp_of[i]) if comp_of.shape[0] == n_p else 0
        comp_nodes.setdefault(c, []).extend(int(q) for q in panels[i])
    comp_min_y: dict[int, float] = {}
    comp_max_y: dict[int, float] = {}
    for c, ids in comp_nodes.items():
        ya = np.abs(nodes[np.array(sorted(set(ids)), dtype=int), 1])
        comp_min_y[c] = float(np.min(ya))
        comp_max_y[c] = float(np.max(ya))
    watertight = True
    n_open = 0
    for key, cnt in counts.items():
        if cnt == 2:
            continue
        if cnt > 2:
            watertight = False
            continue
        n_open += 1
        a, b = key
        ya, yb = abs(float(nodes[a, 1])), abs(float(nodes[b, 1]))
        c = int(comp_of[owners[key]]) if comp_of.shape[0] == n_p else 0
        lo = comp_min_y.get(c, 0.0)
        hi = comp_max_y.get(c, 0.0)
        span = max(1.0, hi)
        root_ok = bool(mesh.half) and max(ya, yb) <= lo + max(1e-9, 1e-3 * span)
        tip_ok = min(ya, yb) >= hi - max(1e-9, 1e-3 * span)
        if not (root_ok or tip_ok):
            watertight = False
    return {
        "watertight": bool(watertight),
        "outward": bool(volume > 0.0),
        "volume": float(volume),
        "min_area": float(np.min(areas)) if n_p else 0.0,
        "aspect_ratio": aspects,
        "skew": skews,
        "warp": warps,
        "max_aspect_ratio": float(np.max(aspects)) if n_p else 1.0,
        "max_skew": float(np.max(skews)) if n_p else 0.0,
        "max_warp": float(np.max(warps)) if n_p else 0.0,
        "n_open_edges": int(n_open),
    }
