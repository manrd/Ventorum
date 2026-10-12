# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Differentiable planform geometry for the adjoint gradients.

A PyTorch port of :func:`ventorum.geometry.lattice.surface_edge_geometry` and of the
spanwise station rule for a fixed panel count. The adjoint gradients contract the node
gradients with the derivatives of the nodes with respect to the planform parameters
through this port (reverse mode), not by finite differences.

The mesh topology stays fixed: the panel count, the spacing choice, the section-break
and control-break assignment of each snapped edge, the section order and the mirror
structure do not change. The spanwise stations move with the parameters (a snapped
edge tracks the span fraction of its section); their count does not change.

References
----------
* J. Katz and A. Plotkin, "Low-Speed Aerodynamics", 2nd ed., Cambridge University
  Press, 2001: the horseshoe geometry (leading edge, trailing edge, quarter chord).
* The spacing rules are the rules of :mod:`ventorum.geometry.discretization`
  (uniform and cosine families with control points at the mid parameter).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ventorum.sensitivity.reference import require_torch, rod_rot


@dataclass
class SurfaceTrace:
    """Frozen topology and base values of one surface for the torch port.

    Attributes
    ----------
    name : str
        Surface name.
    is_symmetric : bool
        True for a surface mirrored about y = 0.
    mirror_y : bool
        True for a mirror copy (y -> -y).
    n_sp : int
        Spanwise panels of the defining half (fixed count).
    spacing : str
        Resolved spacing name (fixed choice).
    n_sections : int
        Number of sections (fixed order, root first).
    eta_base : np.ndarray
        Base edge stations (n_edges,) [-].
    mid_base : np.ndarray
        Base control-point stations (n_edges - 1,) [-] (mid parameter, with
        the snap updates of the numpy builder).
    snap_sec : list
        Per edge the section index whose span fraction it tracks, or None.
    snap_moved : list
        Per edge True when the edge moved to its break at the base point.
        An edge on its break keeps the spacing midpoints until the break
        moves away (same rule as the numpy builder).
    root_mask : np.ndarray
        Per edge True where the twist axis is the y axis (frozen).
    sec_has_x : list of bool
        Per section True when x_le is set explicitly.
    has_z : bool
        True when every section sets z_le explicitly.
    sec_base : list of dict
        Per section the base values (chord [m], twist [rad], x_le [m] or None,
        z_le [m] or None, y_frac [-], alpha_L0 [rad]).
    surf_base : dict
        Surface base values (semi_span [m], sweep_le [rad], dihedral [rad],
        incidence [rad], position [m] shape (3,)).
    undef : dict
        Undeformed defining-half nodes from the numpy builder
        (keys 'eta', 'le', 'te').
    n_full : int
        Strips of the full surface in the lattice.
    full_perm : np.ndarray
        Defining-half strip index of each full strip.
    full_complement : np.ndarray
        Per full strip True when the fraction is one minus the defining value.
    """

    name: str = ""
    is_symmetric: bool = True
    mirror_y: bool = False
    n_sp: int = 0
    spacing: str = "half-cosine"
    n_sections: int = 0
    eta_base: np.ndarray = field(default_factory=lambda: np.zeros(0))
    mid_base: np.ndarray = field(default_factory=lambda: np.zeros(0))
    snap_sec: list = field(default_factory=list)
    snap_moved: list = field(default_factory=list)
    root_mask: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    sec_has_x: list = field(default_factory=list)
    has_z: bool = False
    sec_base: list = field(default_factory=list)
    surf_base: dict = field(default_factory=dict)
    undef: dict = field(default_factory=dict)
    n_full: int = 0
    full_perm: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    full_complement: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))


@dataclass
class PlanformGeometry:
    """Frozen planform topology of an aircraft for the torch port."""

    surfaces: dict = field(default_factory=dict)
    order: list = field(default_factory=list)
    main_index: int = 0
    ref_surfaces: list = field(default_factory=list)
    ref_factors: dict = field(default_factory=dict)


@dataclass
class PlanformTensors:
    """Torch tensors of the planform port (nodes, stations, strips, references)."""

    params: dict = field(default_factory=dict)
    nodes: dict = field(default_factory=dict)
    strips: dict = field(default_factory=dict)
    refs: dict = field(default_factory=dict)


def _eta_trace(surf: Any, n_sp: int, spacing: str) -> tuple:
    """Return (edges, mids, snap, moved) of the spanwise stations with provenance.

    The algorithm is the station rule of the numpy builder (section breaks then
    control limits, nearest free edge, quarter-width rule): *snap[i]* is the section
    index whose span fraction edge *i* tracks, or None, and *moved[i]* tells whether
    the edge moved at the base point. Control snaps are constants (control limits
    are not planform parameters), so they need no provenance.
    """
    from ventorum.geometry.discretization import get_spacing
    from ventorum.geometry.lattice import _section_breaks

    edges, mids = get_spacing(spacing, int(n_sp), surf=surf)
    edges = np.asarray(edges, dtype=float).copy()
    mids = np.asarray(mids, dtype=float).copy()
    edges[0], edges[-1] = 0.0, 1.0
    snap: list = [None] * len(edges)
    snap_moved: list = [False] * len(edges)
    secs = sorted(surf.sections, key=lambda s: s.y_frac)
    fr = [float(s.y_frac) for s in secs]
    breaks = [(float(eb), fr.index(float(eb))) for eb in _section_breaks(surf)]
    moved: set = set()

    def _snap_to(value: float, sec_idx: int | None) -> None:
        cand = [i for i in range(1, len(edges) - 1) if i not in moved]
        if not cand:
            return
        i = min(cand, key=lambda j: abs(edges[j] - value))
        if abs(edges[i] - value) < 1e-12:
            moved.add(i)
            if sec_idx is not None:
                snap[i] = sec_idx
            return
        w_left, w_right = edges[i] - edges[i - 1], edges[i + 1] - edges[i]
        if value - edges[i - 1] < 0.25 * w_left or edges[i + 1] - value < 0.25 * w_right:
            return
        edges[i] = value
        mids[i - 1] = 0.5 * (edges[i - 1] + edges[i])
        mids[i] = 0.5 * (edges[i] + edges[i + 1])
        moved.add(i)
        if sec_idx is not None:
            snap[i] = sec_idx
            snap_moved[i] = True

    for eb, sec_idx in breaks:
        _snap_to(eb, sec_idx)
    for c in getattr(surf, "controls", []):
        for lim in (c.eta_start, c.eta_end):
            if 1e-12 < lim < 1.0 - 1e-12:
                _snap_to(float(lim), None)
    return edges, mids, snap, snap_moved


def build_planform(aircraft: Any, settings: Any, collocation: str) -> PlanformGeometry:
    """Freeze the planform topology of *aircraft* for the torch port.

    Parameters
    ----------
    aircraft : Aircraft
        Aircraft model (not changed).
    settings : SolverSettings
        Discretisation settings (panel counts, spacing).
    collocation : str
        ``"vlm"`` or ``"llt"`` (selects the spacing rule of ``"auto"``).

    Returns
    -------
    PlanformGeometry
        Frozen traces per surface, main-surface index and reference surfaces.
    """
    from ventorum.geometry.deformation import undeformed_nodes
    from ventorum.geometry.discretization import compute_surface_n_panels
    from ventorum.geometry.lattice import _surface_mesh_eta, resolve_spacing

    solver = "vlm" if collocation == "vlm" else "linear"
    ref_semi = max((float(s.semi_span) for s in aircraft.surfaces), default=1.0)
    undef = undeformed_nodes(aircraft, settings, solver=solver)
    geom = PlanformGeometry()
    geom.main_index = aircraft.main_surface_index()
    geom.ref_surfaces = [s.name for s in aircraft.reference_surfaces()]
    for surf in aircraft.surfaces:
        tr = SurfaceTrace()
        tr.name = surf.name
        tr.is_symmetric = bool(surf.is_symmetric)
        tr.mirror_y = bool(getattr(surf, "mirror_y", False))
        if getattr(settings, "proportional_panels", False) and surf.n_panels is None:
            n_sp = compute_surface_n_panels(
                surf, base_n_panels=settings.n_panels, reference_semi_span=ref_semi,
                min_panels=getattr(settings, "min_panels", 8))
        else:
            n_sp = surf.n_panels if surf.n_panels is not None else settings.n_panels
        tr.n_sp = int(n_sp)
        spacing = resolve_spacing(
            surf.spacing if surf.spacing is not None else settings.spacing, surf, collocation)
        tr.spacing = str(spacing)
        eta_e, eta_mid = _surface_mesh_eta(surf, settings, collocation, ref_semi)
        edges, mids, snap, snap_moved = _eta_trace(surf, tr.n_sp, tr.spacing)
        if edges.shape != eta_e.shape or float(np.max(np.abs(edges - eta_e))) != 0.0:
            raise RuntimeError(
                f"[{surf.name}] the station trace does not reproduce the mesh stations; "
                "the planform chain cannot run.")
        if mids.shape != eta_mid.shape or float(np.max(np.abs(mids - eta_mid))) != 0.0:
            raise RuntimeError(
                f"[{surf.name}] the station trace does not reproduce the control stations; "
                "the planform chain cannot run.")
        tr.eta_base = np.asarray(eta_e, dtype=float).copy()
        tr.mid_base = np.asarray(eta_mid, dtype=float).copy()
        tr.snap_sec = list(snap)
        tr.snap_moved = list(snap_moved)
        tr.n_sections = len(surf.sections)
        secs = list(surf.sections)
        tr.sec_has_x = [bool(s.x_le is not None) for s in secs]
        tr.has_z = bool(any(s.z_le is not None for s in secs))
        for s in secs:
            tr.sec_base.append({
                "chord": float(s.chord),
                "twist": float(s.twist),
                "x_le": None if s.x_le is None else float(s.x_le),
                "z_le": None if s.z_le is None else float(s.z_le),
                "y_frac": float(s.y_frac),
                "alpha_L0": float(s.airfoil.alpha_L0),
                "a0": float(s.airfoil.a0),
                "Cd0": float(s.airfoil.Cd0),
                "Cm0": float(s.airfoil.Cm0),
            })
        tr.surf_base = {
            "semi_span": float(surf.semi_span),
            "sweep_le": float(surf.sweep_le),
            "dihedral": float(surf.dihedral),
            "incidence": float(surf.incidence),
            "position": np.asarray(surf.position, dtype=float).reshape(3).copy(),
        }
        tr.undef = {"eta": np.asarray(undef[surf.name]["eta"], dtype=float).copy(),
                    "le": np.asarray(undef[surf.name]["le"], dtype=float).copy(),
                    "te": np.asarray(undef[surf.name]["te"], dtype=float).copy()}
        tr.root_mask = tr.eta_base <= 1e-12
        n_r = tr.n_sp
        if tr.is_symmetric:
            tr.n_full = 2 * n_r
            tr.full_perm = np.concatenate([np.arange(n_r)[::-1], np.arange(n_r)]).astype(int)
            tr.full_complement = np.concatenate([np.ones(n_r, bool), np.zeros(n_r, bool)])
        elif tr.mirror_y:
            tr.n_full = n_r
            tr.full_perm = np.arange(n_r)[::-1].astype(int)
            tr.full_complement = np.ones(n_r, bool)
        else:
            tr.n_full = n_r
            tr.full_perm = np.arange(n_r).astype(int)
            tr.full_complement = np.zeros(n_r, bool)
        geom.surfaces[surf.name] = tr
        geom.order.append(surf.name)
        geom.ref_factors[surf.name] = 2.0 if tr.is_symmetric else 1.0
    return geom


def _interp_torch(x: Any, xp: Any, fp: Any) -> Any:
    """Linear interpolation of *fp* at *x* over ascending knots *xp* (torch).

    Matches ``np.interp`` (clamped ends) in value; differentiable in the values,
    the queries and the knots except at the knots.
    """
    import torch

    n = int(xp.shape[0])
    if n == 1:
        return fp[0].expand_as(x).clone()
    idx = torch.searchsorted(xp, x)
    j = torch.clamp(idx - 1, 0, n - 2)
    x0, x1 = xp[j], xp[j + 1]
    f0, f1 = fp[j], fp[j + 1]
    denom = x1 - x0
    tiny = float(torch.finfo(x.dtype).tiny)
    w = torch.where(denom > 0, (x - x0) / torch.clamp_min(denom, tiny), torch.zeros_like(x))
    out = (1.0 - w) * f0 + w * f1
    out = torch.where(x <= xp[0], fp[0], out)
    out = torch.where(x >= xp[-1], fp[-1], out)
    return out


def _gradient_torch(y: Any, x: Any) -> Any:
    """Spanwise derivative dy/dx on a non-uniform grid (torch).

    Lagrange three-point central differences inside, first differences at the
    ends: the rule of ``np.gradient`` with the default edge order.
    """
    import torch

    n = int(x.shape[0])
    if n == 1:
        return torch.zeros_like(y)
    h = x[1:] - x[:-1]
    d = (y[1:] - y[:-1]) / h
    out = torch.empty_like(y)
    out[0] = d[0]
    out[-1] = d[-1]
    if n > 2:
        hm, hp = h[:-1], h[1:]
        denom = hm * hp * (hm + hp)
        a = -hp * hp / denom
        b = (hp - hm) * (hm + hp) / denom
        c = hm * hm / denom
        out[1:-1] = a * y[:-2] + b * y[1:-1] + c * y[2:]
    return out


def _surface_params_torch(tr: SurfaceTrace, device: Any, dtype: Any,
                          differentiable: bool) -> dict:
    """Return the torch parameters of one surface (leaves when *differentiable*)."""
    import torch

    dev = torch.device(device)

    def _leaf(value: float) -> Any:
        t = torch.tensor(float(value), dtype=dtype, device=dev)
        if differentiable:
            t.requires_grad_(True)
        return t

    def _const(value: float) -> Any:
        return torch.tensor(float(value), dtype=dtype, device=dev)

    p: dict = {}
    for key in ("semi_span", "sweep_le", "dihedral", "incidence"):
        p[key] = _leaf(tr.surf_base[key])
    pos = torch.as_tensor(np.ascontiguousarray(tr.surf_base["position"]),
                          dtype=dtype, device=dev)
    if differentiable:
        pos.requires_grad_(True)
    p["position"] = pos
    secs = []
    n = tr.n_sections
    for j, sb in enumerate(tr.sec_base):
        d: dict = {"chord": _leaf(sb["chord"]), "twist": _leaf(sb["twist"])}
        if j == 0 or j == n - 1:
            d["y_frac"] = _const(sb["y_frac"])
        else:
            d["y_frac"] = _leaf(sb["y_frac"])
        d["x_le"] = _leaf(sb["x_le"]) if sb["x_le"] is not None else None
        d["z_le"] = _leaf(sb["z_le"]) if sb["z_le"] is not None else None
        d["alpha_L0"] = _leaf(sb["alpha_L0"])
        secs.append(d)
    p["sec"] = secs
    return p


def _eta_torch(tr: SurfaceTrace, p: dict, device: Any, dtype: Any) -> Any:
    """Return the torch edge stations of one surface (snapped edges track y_frac)."""
    import torch

    eta = torch.as_tensor(np.ascontiguousarray(tr.eta_base), dtype=dtype,
                          device=torch.device(device)).clone()
    if any(s is not None for s in tr.snap_sec):
        yf_base = torch.as_tensor(np.array([sb["y_frac"] for sb in tr.sec_base]),
                                  dtype=dtype, device=torch.device(device))
        yf_now = torch.stack([p["sec"][j]["y_frac"] for j in range(tr.n_sections)])
        shift = yf_now - yf_base
        for i, sec_idx in enumerate(tr.snap_sec):
            if sec_idx is not None:
                step = torch.zeros_like(eta)
                step[i] = shift[sec_idx]
                eta = eta + step
    return eta


def edge_nodes_torch(tr: SurfaceTrace, p: dict, device: Any, dtype: Any) -> tuple:
    """Return (le, te, eta) of the defining half of one surface (torch port).

    Parameters
    ----------
    tr : SurfaceTrace
        Frozen trace of the surface.
    p : dict
        Torch parameters from :func:`planform_tensors` (leaves or constants).
    device : str
        Torch device.
    dtype : torch.dtype
        Torch dtype.

    Returns
    -------
    tuple
        ``(le, te, eta)``: leading edge [m], trailing edge [m] and stations [-].
    """
    import torch

    require_torch()
    dev = torch.device(device)
    eta = _eta_torch(tr, p, device, dtype)
    fr = torch.stack([d["y_frac"] for d in p["sec"]])
    b = p["semi_span"]
    sweep = p["sweep_le"]
    dih = p["dihedral"]
    inc = p["incidence"]
    if any(tr.sec_has_x):
        xv = torch.stack([
            d["x_le"] if d["x_le"] is not None
            else d["y_frac"] * b * torch.tan(sweep) for d in p["sec"]])
        x_le = _interp_torch(eta, fr, xv)
    else:
        x_le = eta * b * torch.tan(sweep)
    if tr.has_z:
        y = eta * b
        zv = torch.stack([d["z_le"] for d in p["sec"]])
        z = _interp_torch(eta, fr, zv)
    else:
        y = eta * b * torch.cos(dih)
        z = eta * b * torch.sin(dih)
    chord = _interp_torch(eta, fr, torch.stack([d["chord"] for d in p["sec"]]))
    twist = _interp_torch(eta, fr, torch.stack([d["twist"] for d in p["sec"]])) + inc
    dy = _gradient_torch(y, eta)
    dz = _gradient_torch(z, eta)
    axis = torch.stack([torch.zeros_like(eta), dy, dz], dim=1)
    axis = axis / torch.clamp_min(torch.linalg.norm(axis, dim=1, keepdim=True), 1e-300)
    if tr.is_symmetric:
        mask = torch.as_tensor(np.ascontiguousarray(tr.root_mask), dtype=torch.bool, device=dev)
        yaxis = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=dev).expand_as(axis)
        axis = torch.where(mask[:, None], yaxis, axis)
    n = int(eta.shape[0])
    x_hat = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=dev).expand(n, 3)
    chord_dir = rod_rot(x_hat, axis, twist)
    offset = p["position"]
    le_flat = torch.stack([x_le, y, z], dim=1) + offset
    qc = le_flat + 0.25 * chord[:, None] * x_hat
    le = qc - 0.25 * chord[:, None] * chord_dir
    te = qc + 0.75 * chord[:, None] * chord_dir
    return le, te, eta


def planform_tensors(geom: PlanformGeometry, device: Any, dtype: Any,
                     differentiable: bool = True) -> PlanformTensors:
    """Build all torch tensors of the planform port.

    Parameters
    ----------
    geom : PlanformGeometry
        Frozen topology from :func:`build_planform`.
    device : str
        Torch device.
    dtype : torch.dtype
        Torch dtype.
    differentiable : bool
        True makes the parameters leaves (gradient path); False makes them
        constants (parity checks). The airfoil incidence (``alpha_L0``) always
        comes from the section parameters; the other section properties
        (``a0``, ``Cd0``, ``Cm0``) come from the frozen base values.

    Returns
    -------
    PlanformTensors
        Parameters, defining-half nodes, defining-half strips and references.
    """
    require_torch()
    out = PlanformTensors()
    for name in geom.order:
        tr = geom.surfaces[name]
        p = _surface_params_torch(tr, device, dtype, differentiable)
        out.params[name] = p
        le, te, eta = edge_nodes_torch(tr, p, device, dtype)
        out.nodes[name] = {"le": le, "te": te, "eta": eta}
        out.strips[name] = _strip_tensors_with_consts(tr, p, eta, device, dtype)
    out.refs = _auto_refs_torch(geom, out.params, device, dtype)
    return out


def _strip_tensors_with_consts(tr: SurfaceTrace, p: dict, eta: Any,
                               device: Any, dtype: Any) -> dict:
    """Return the defining-half strip tensors with frozen non-planform properties."""
    import torch

    dev = torch.device(device)
    n_r = tr.n_sp
    base = torch.as_tensor(np.ascontiguousarray(tr.mid_base), dtype=dtype, device=dev)
    means = 0.5 * (eta[:-1] + eta[1:])
    yf_now = torch.stack([d["y_frac"] for d in p["sec"]])
    forced = [False] * n_r
    switches: dict = {}
    for i, sec_idx in enumerate(tr.snap_sec):
        if sec_idx is None:
            continue
        if bool(tr.snap_moved[i]):
            for k in (i - 1, i):
                if 0 <= k < n_r:
                    forced[k] = True
        else:
            coincident = torch.abs(
                torch.tensor(float(tr.eta_base[i]), dtype=dtype, device=dev)
                - yf_now[sec_idx]) < 1e-12
            for k in (i - 1, i):
                if 0 <= k < n_r:
                    switches.setdefault(k, []).append(coincident)
    act = torch.as_tensor(np.array(forced, dtype=float), dtype=dtype, device=dev)
    for k, group in switches.items():
        all_c = group[0]
        for c in group[1:]:
            all_c = all_c & c
        gate = torch.where(all_c, 0.0, 1.0)
        one = torch.where(torch.arange(n_r, device=dev) == k, gate, 0.0)
        act = act + one
    mid = torch.where(act > 0.5, means, base)
    span = torch.clamp_min(eta[1:] - eta[:-1], 1e-300)
    frac = (mid - eta[:-1]) / span
    fr = torch.stack([d["y_frac"] for d in p["sec"]])
    out: dict = {"eta": mid, "frac": frac}
    for prop in ("a0", "alpha_L0", "Cd0", "Cm0"):
        if prop == "alpha_L0":
            vals = torch.stack([d["alpha_L0"] for d in p["sec"]])
        else:
            base = np.array([sb[prop] for sb in tr.sec_base], dtype=float)
            vals = torch.as_tensor(np.ascontiguousarray(base), dtype=dtype, device=dev)
        out[prop] = _interp_torch(mid, fr, vals)
    return out


def _auto_refs_torch(geom: PlanformGeometry, params: dict, device: Any, dtype: Any) -> dict:
    """Return the automatic reference values as torch functions of the parameters."""
    import torch

    dev = torch.device(device)
    span = torch.zeros((), dtype=dtype, device=dev)
    area = torch.zeros((), dtype=dtype, device=dev)
    int_c2 = torch.zeros((), dtype=dtype, device=dev)
    for name in geom.ref_surfaces:
        tr = geom.surfaces[name]
        p = params[name]
        fr = torch.stack([d["y_frac"] for d in p["sec"]])
        chords = torch.stack([d["chord"] for d in p["sec"]])
        b = p["semi_span"]
        if tr.has_z:
            yk = fr * b
        else:
            yk = fr * b * torch.cos(p["dihedral"])
        dy = torch.abs(yk[1:] - yk[:-1])
        tiny = max(float(tr.surf_base["semi_span"]), 1e-12) * 1e-9
        use_proj = float(torch.sum(dy).detach().cpu()) > tiny
        if not use_proj:
            dy = (fr[1:] - fr[:-1]) * b
        f = float(geom.ref_factors[name])
        span = span + f * torch.sum(dy)
        area = area + f * torch.sum(0.5 * (chords[:-1] + chords[1:]) * dy)
        int_c2 = int_c2 + f * torch.sum(
            dy / 3.0 * (chords[:-1] ** 2 + chords[:-1] * chords[1:] + chords[1:] ** 2))
    out = {"b_ref": span, "S_ref": area}
    out["c_ref"] = torch.where(area > 0, int_c2 / torch.clamp_min(area, 1e-300),
                               torch.ones((), dtype=dtype, device=dev))
    return out


def full_strip_tensor(geom: PlanformGeometry, per_surf: dict) -> Any:
    """Concatenate defining-half strip tensors to the full-strip order of the lattice."""
    import torch

    parts = []
    for name in geom.order:
        tr = geom.surfaces[name]
        d = per_surf[name]
        perm = torch.as_tensor(np.ascontiguousarray(tr.full_perm), dtype=torch.int64,
                               device=d.device)
        parts.append(d[perm])
    return torch.cat(parts, dim=0)


def full_frac_tensor(geom: PlanformGeometry, per_surf: dict) -> Any:
    """Concatenate defining-half fractions with the mirror complement on left strips."""
    import torch

    parts = []
    for name in geom.order:
        tr = geom.surfaces[name]
        d = per_surf[name]
        perm = torch.as_tensor(np.ascontiguousarray(tr.full_perm), dtype=torch.int64,
                               device=d.device)
        comp = torch.as_tensor(np.ascontiguousarray(tr.full_complement), dtype=torch.bool,
                               device=d.device)
        g = d[perm]
        parts.append(torch.where(comp, 1.0 - g, g))
    return torch.cat(parts, dim=0)
