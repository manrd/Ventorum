# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Geometry processing: interpolate section properties and build the 3-D panel
geometry used by the horseshoe solver and the 1-D station data used by the
Fourier solver.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from ventorum.legacy.core.datatypes import (
    Aircraft,
    AirfoilType,
    DiscretizedSurface,
    FourierStations,
    LinearAirfoil,
    LiftingSurface,
    SolverSettings,
    TabulatedAirfoil,
    WingSection,
)
from ventorum.legacy.geometry.discretization import (
    compute_surface_n_panels,
    fourier_collocation_angles,
    get_spacing,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Section interpolation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _sorted_sections(surf: LiftingSurface) -> list[WingSection]:
    return sorted(surf.sections, key=lambda s: s.y_frac)


def _interp_scalar(
    y_frac: float | np.ndarray,
    sections: list[WingSection],
    attr: str,
) -> float | np.ndarray:
    """Linearly interpolate a scalar section attribute at *y_frac*."""
    fracs = np.array([s.y_frac for s in sections])
    vals = np.array([getattr(s, attr) for s in sections])
    return np.interp(y_frac, fracs, vals)


def _compute_x_le(
    y_frac: float | np.ndarray,
    sections: list[WingSection],
    semi_span: float,
    sweep_le: float,
) -> float | np.ndarray:
    """Compute leading-edge *x*-offset at *y_frac*.

    If any section specifies ``x_le`` explicitly, interpolation uses those
    values.  Otherwise the surface-level *sweep_le* is applied uniformly.
    """
    has_explicit = any(s.x_le is not None for s in sections)
    if has_explicit:
        # Fill missing with sweep-based value
        fracs = np.array([s.y_frac for s in sections])
        vals = np.array([
            s.x_le if s.x_le is not None else s.y_frac * semi_span * np.tan(sweep_le)
            for s in sections
        ])
        return np.interp(y_frac, fracs, vals)
    return np.asarray(y_frac) * semi_span * np.tan(sweep_le)


def _compute_z_le(
    y_frac: float | np.ndarray,
    sections: list[WingSection],
    semi_span: float,
    dihedral: float,
) -> float | np.ndarray:
    """Compute leading-edge *z*-offset at *y_frac*."""
    has_explicit = any(s.z_le is not None for s in sections)
    if has_explicit:
        fracs = np.array([s.y_frac for s in sections])
        vals = np.array([
            s.z_le if s.z_le is not None else s.y_frac * semi_span * np.sin(dihedral)
            for s in sections
        ])
        return np.interp(y_frac, fracs, vals)
    return np.asarray(y_frac) * semi_span * np.sin(dihedral)


def _nearest_airfoil(
    y_frac: float,
    sections: list[WingSection],
) -> AirfoilType:
    """Return the airfoil of the section nearest to *y_frac*."""
    dists = [abs(s.y_frac - y_frac) for s in sections]
    return sections[int(np.argmin(dists))].airfoil


def _interpolate_linear_airfoil(
    y_frac: float,
    sections: list[WingSection],
) -> LinearAirfoil:
    """Linearly interpolate ``a0``, ``alpha_L0``, ``Cd0``, ``Cm0``
    between the two bounding sections that carry :class:`LinearAirfoil`.
    Falls back to thin-airfoil defaults if no linear airfoils are found.
    """
    lin_secs = [(s.y_frac, s.airfoil) for s in sections
                if isinstance(s.airfoil, LinearAirfoil)]
    if len(lin_secs) == 0:
        # Fall back: extract linear properties from tabulated polars
        fracs = np.array([s.y_frac for s in sections])
        a0_vals = np.array([s.airfoil.a0 for s in sections])
        aL0_vals = np.array([s.airfoil.alpha_L0 for s in sections])
        a0 = float(np.interp(y_frac, fracs, a0_vals))
        aL0 = float(np.interp(y_frac, fracs, aL0_vals))
        return LinearAirfoil(a0=a0, alpha_L0=aL0)

    fracs = np.array([f for f, _ in lin_secs])
    a0s = np.array([af.a0 for _, af in lin_secs])
    aL0s = np.array([af.alpha_L0 for _, af in lin_secs])
    Cd0s = np.array([af.Cd0 for _, af in lin_secs])
    Cm0s = np.array([af.Cm0 for _, af in lin_secs])
    return LinearAirfoil(
        a0=float(np.interp(y_frac, fracs, a0s)),
        alpha_L0=float(np.interp(y_frac, fracs, aL0s)),
        Cd0=float(np.interp(y_frac, fracs, Cd0s)),
        Cm0=float(np.interp(y_frac, fracs, Cm0s)),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Discretise for the Fourier solver
# ═══════════════════════════════════════════════════════════════════════════════

def build_fourier_stations(
    surf: LiftingSurface,
    n_terms: int | None = None,
) -> FourierStations:
    """Create 1-D collocation stations for the Fourier solver.

    Parameters
    ----------
    surf : LiftingSurface
        The (single, symmetric, unswept) surface.
    n_terms : int or None
        Number of Fourier terms = number of collocation stations.
        If None, inherits from surf.n_panels or default 80.

    Returns
    -------
    FourierStations
    """
    eff_n_terms = surf.n_panels if surf.n_panels is not None else (n_terms if n_terms is not None else 80)
    sections = _sorted_sections(surf)
    sec_fracs = np.array([s.y_frac for s in sections])
    sec_chords = np.array([s.chord for s in sections])
    sec_twists = np.array([s.twist for s in sections])

    theta = fourier_collocation_angles(eff_n_terms)
    # Map θ → y_frac: y = (b/2)·cos(θ) → y_frac = |cos(θ)| (ranges 1→0→1 as θ 0→π/2→π)
    # Since y_frac is 0 at root and 1 at tip for symmetric lifting surfaces,
    # the fraction of semi-span from root to tip is |cos(θ)|.
    y = np.cos(theta) * surf.semi_span
    y_frac = np.abs(np.cos(theta))

    chords = np.interp(y_frac, sec_fracs, sec_chords)
    twists = np.interp(y_frac, sec_fracs, sec_twists)

    lin_secs = [(s.y_frac, s.airfoil) for s in sections
                if isinstance(s.airfoil, LinearAirfoil)]
    if len(lin_secs) == 0:
        a0_vals = np.array([s.airfoil.a0 for s in sections])
        aL0_vals = np.array([s.airfoil.alpha_L0 for s in sections])
        a0 = np.interp(y_frac, sec_fracs, a0_vals)
        alpha_L0 = np.interp(y_frac, sec_fracs, aL0_vals)
        Cd0 = np.zeros_like(y_frac)
    else:
        fracs = np.array([f for f, _ in lin_secs])
        a0s = np.array([af.a0 for _, af in lin_secs])
        aL0s = np.array([af.alpha_L0 for _, af in lin_secs])
        Cd0s = np.array([af.Cd0 for _, af in lin_secs])
        a0 = np.interp(y_frac, fracs, a0s)
        alpha_L0 = np.interp(y_frac, fracs, aL0s)
        Cd0 = np.interp(y_frac, fracs, Cd0s)

    return FourierStations(
        theta=theta,
        y=y,
        chords=chords,
        a0=a0,
        alpha_L0=alpha_L0,
        twists=twists,
        Cd0=Cd0,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Discretise for the horseshoe solver
# ═══════════════════════════════════════════════════════════════════════════════

def discretize_surface(
    surf: LiftingSurface,
    n_panels: int | None = None,
    spacing: str | None = None,
    surface_index: int = 0,
    half_mesh: bool = False,
) -> DiscretizedSurface:
    """Discretise a :class:`LiftingSurface` into horseshoe-vortex panels.

    For a symmetric surface with half_mesh=False the right semi-span is discretised first, then
    mirrored to the left.  Panel ordering is **left-tip → root → right-tip**.
    When half_mesh=True, only the right semi-span (y >= 0) is discretised.

    Parameters
    ----------
    surf : LiftingSurface
    n_panels : int or None
        Panels per semi-span. If None, inherits from surf.n_panels (or default 80).
    spacing : str or None
        ``"auto"``, ``"cosine"``, ``"half-cosine"``, ``"root"``, or ``"uniform"``.
        If None, inherits from surf.spacing (or default "cosine").
    surface_index : int
        Index of this surface within the parent :class:`Aircraft`.

    Returns
    -------
    DiscretizedSurface
    """
    sections = _sorted_sections(surf)
    sec_fracs = np.array([s.y_frac for s in sections])
    sec_chords = np.array([s.chord for s in sections])
    sec_twists = np.array([s.twist for s in sections])
    has_explicit_x = any(s.x_le is not None for s in sections)
    has_explicit_z = any(s.z_le is not None for s in sections)
    b_semi = surf.semi_span
    sweep_le = surf.sweep_le
    dihedral = surf.dihedral

    # Resolve panel count and spacing method (surface-level overrides take precedence)
    eff_n_panels = surf.n_panels if surf.n_panels is not None else (n_panels if n_panels is not None else 80)
    eff_spacing = surf.spacing if surf.spacing is not None else (spacing if spacing is not None else "cosine")

    y_edges_frac, y_mids_frac = get_spacing(eff_spacing, eff_n_panels, surf=surf)

    # --- interpolate at panel edges (for node positions) ----------------------
    chords_edges = np.interp(y_edges_frac, sec_fracs, sec_chords)
    twists_edges = np.interp(y_edges_frac, sec_fracs, sec_twists)

    if has_explicit_x:
        x_vals = np.array([
            s.x_le if s.x_le is not None else s.y_frac * b_semi * np.tan(sweep_le)
            for s in sections
        ])
        x_le_edges = np.interp(y_edges_frac, sec_fracs, x_vals)
        x_le_mid = np.interp(y_mids_frac, sec_fracs, x_vals)
    else:
        tan_sweep = b_semi * np.tan(sweep_le)
        x_le_edges = y_edges_frac * tan_sweep
        x_le_mid = y_mids_frac * tan_sweep

    if has_explicit_z:
        z_vals = np.array([
            s.z_le if s.z_le is not None else s.y_frac * b_semi * np.sin(dihedral)
            for s in sections
        ])
        z_le_edges = np.interp(y_edges_frac, sec_fracs, z_vals)
        z_le_mid = np.interp(y_mids_frac, sec_fracs, z_vals)
    else:
        sin_dih = b_semi * np.sin(dihedral)
        z_le_edges = y_edges_frac * sin_dih
        z_le_mid = y_mids_frac * sin_dih

    # --- interpolate at panel centres (for control points & properties) -------
    chords_mid = np.interp(y_mids_frac, sec_fracs, sec_chords)
    twists_mid = np.interp(y_mids_frac, sec_fracs, sec_twists)

    # --- build right-semi-span panel geometry ---------------------------------
    # Node positions at 1/4-chord for panel edges
    n_edges = len(y_edges_frac)
    right_nodes = np.empty((n_edges, 3))
    right_nodes[:, 0] = x_le_edges + 0.25 * chords_edges  # x_qc
    right_nodes[:, 1] = y_edges_frac * b_semi              # y
    right_nodes[:, 2] = z_le_edges                          # z

    # Control points at 3/4-chord for panel centres
    n_mid = len(y_mids_frac)
    right_cp = np.empty((n_mid, 3))
    right_cp[:, 0] = x_le_mid + 0.75 * chords_mid
    right_cp[:, 1] = y_mids_frac * b_semi
    right_cp[:, 2] = z_le_mid

    # Normal vectors - account for local twist and dihedral (vectorized)
    dx = 0.5 * chords_mid
    tw = twists_mid + surf.incidence
    cos_tw, sin_tw = np.cos(tw), np.sin(tw)
    
    cv_x = dx * cos_tw
    cv_z = -dx * sin_tw
    
    span_vec = np.diff(right_nodes, axis=0)
    
    nx = -cv_z * span_vec[:, 1]
    ny = cv_z * span_vec[:, 0] - cv_x * span_vec[:, 2]
    nz = cv_x * span_vec[:, 1]
    
    norm_sq = nx**2 + ny**2 + nz**2
    norm_mag = np.sqrt(norm_sq)
    valid = norm_mag > 1e-14
    
    inv_mag = np.where(valid, 1.0 / np.maximum(norm_mag, 1e-14), 0.0)
    right_normals = np.column_stack([nx * inv_mag, ny * inv_mag, nz * inv_mag])
    if not np.all(valid):
        right_normals[~valid] = [0.0, 0.0, 1.0]

    # Airfoils at each panel (vectorized nearest-station lookup)
    if len(sections) == 1:
        right_airfoils = [sections[0].airfoil] * n_mid
    else:
        sec_fracs = np.array([s.y_frac for s in sections])
        nearest_idx = np.argmin(np.abs(y_mids_frac[:, None] - sec_fracs[None, :]), axis=1)
        right_airfoils = [sections[idx].airfoil for idx in nearest_idx]

    # --- mirror to left semi-span if symmetric and not half_mesh --------------
    is_half = bool(surf.is_symmetric and half_mesh)
    if surf.is_symmetric and not half_mesh:
        # Left nodes: mirror y, reverse order (left tip first)
        left_nodes = right_nodes[::-1].copy()
        left_nodes[:, 1] *= -1.0

        left_cp = right_cp[::-1].copy()
        left_cp[:, 1] *= -1.0

        left_normals = right_normals[::-1].copy()
        # Normal vector y-component flips under reflection across y=0 (xz-plane):
        # A dihedral wing with normals pointing inward-up on the right (+y) wing
        # must have normals pointing inward-up on the left (-y) wing.
        left_normals[:, 1] *= -1.0

        left_airfoils = right_airfoils[::-1]
        left_y_mids = -(y_mids_frac[::-1] * b_semi)
        left_dy = np.abs(np.diff(y_edges_frac[::-1])) * b_semi  # strictly positive
        right_y_mids_abs = y_mids_frac * b_semi
        right_dy = np.diff(y_edges_frac) * b_semi

        # Concatenate: left (negative y) then right (positive y)
        all_nodes = np.vstack([left_nodes, right_nodes[1:]])  # shared root node
        all_cp = np.vstack([left_cp, right_cp])
        all_normals = np.vstack([left_normals, right_normals])
        all_y = np.concatenate([left_y_mids, right_y_mids_abs])
        all_dy = np.concatenate([left_dy, right_dy])
        all_chords = np.concatenate([chords_mid[::-1], chords_mid])
        all_twists = np.concatenate([twists_mid[::-1], twists_mid])
        all_airfoils = left_airfoils + right_airfoils
    else:
        all_nodes = right_nodes
        all_cp = right_cp
        all_normals = right_normals
        all_y = y_mids_frac * b_semi
        all_dy = np.diff(y_edges_frac) * b_semi
        all_chords = chords_mid
        all_twists = twists_mid
        all_airfoils = right_airfoils

    # --- apply surface position offset ----------------------------------------
    offset = np.asarray(surf.position, dtype=float)
    all_nodes = all_nodes + offset
    all_cp = all_cp + offset

    # --- precompute panel quarter-chord centers and airfoil groups ------------
    panel_centers = 0.5 * (all_nodes[:-1] + all_nodes[1:])
    unique_airfoils = []
    for af in all_airfoils:
        if not any(af is u for u in unique_airfoils):
            unique_airfoils.append(af)
    airfoil_groups = [
        (uaf, np.where([af is uaf for af in all_airfoils])[0])
        for uaf in unique_airfoils
    ]
    has_profile_drag = any(
        getattr(af, "Cd0", None) is None or getattr(af, "Cd0", 0.0) > 0.0
        for af in unique_airfoils
    )

    return DiscretizedSurface(
        nodes_qc=all_nodes,
        control_points=all_cp,
        normals=all_normals,
        y_panels=all_y,
        dy_panels=all_dy,
        chords=all_chords,
        twists=all_twists,
        airfoils=all_airfoils,
        surface_name=surf.name,
        surface_index=surface_index,
        panel_centers_qc=panel_centers,
        has_profile_drag=has_profile_drag,
        airfoil_groups=airfoil_groups,
        is_half_mesh=is_half,
    )


def discretize_aircraft_surfaces(
    aircraft: Aircraft,
    settings: SolverSettings,
    half_mesh: bool = False,
) -> list[DiscretizedSurface]:
    """Discretise all lifting surfaces in an aircraft according to solver settings.

    Respects per-surface overrides (``surf.n_panels``, ``surf.spacing``), applies
    geometry-adaptive optimal spacing when ``spacing == "auto"``, and scales panel counts
    proportionally when ``settings.proportional_panels is True``.

    Parameters
    ----------
    aircraft : Aircraft
    settings : SolverSettings
    half_mesh : bool
        If True, returns half-mesh surfaces for symmetric solver acceleration.

    Returns
    -------
    list[DiscretizedSurface]
    """
    ref_semi = max((s.semi_span for s in aircraft.surfaces), default=1.0)
    disc_surfaces: list[DiscretizedSurface] = []
    for idx, surf in enumerate(aircraft.surfaces):
        if getattr(settings, "proportional_panels", False) and getattr(surf, "n_panels", None) is None:
            n_p = compute_surface_n_panels(
                surf,
                base_n_panels=settings.n_panels,
                reference_semi_span=ref_semi,
                min_panels=getattr(settings, "min_panels", 8),
            )
        else:
            n_p = settings.n_panels

        ds = discretize_surface(
            surf,
            n_panels=n_p,
            spacing=settings.spacing,
            surface_index=idx,
            half_mesh=half_mesh,
        )
        disc_surfaces.append(ds)
    return disc_surfaces
