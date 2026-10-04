# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Symmetry plane (Y=0) acceleration and mirror management for Ventorum.

Mathematical Foundation
-----------------------
In aerodynamic lifting-line and vortex-lattice formulations, when an aircraft
exhibits geometric symmetry across the lateral mid-plane (Y = 0) and the oncoming
flow exhibits zero lateral asymmetry (sideslip angle beta = 0, bank angle phi = 0,
symmetric control surface deflections), the spanwise circulation distribution is
strictly even across the span:

    Gamma(-y) = Gamma(y)

Under this condition, the complete (2N x 2N) aerodynamic influence system:

    | AIC_LL   AIC_LR | | Gamma_L |   | rhs_L |
    | AIC_RL   AIC_RR | | Gamma_R | = | rhs_R |

collapses by spatial parity (Gamma_L = J . Gamma_R, where J is the coordinate-reversal
permutation) into a half-mesh (N x N) system evaluated only over the right semi-span
(y >= 0):

    (AIC_RR + AIC_RL . J) . Gamma_R = rhs_R

This yields:
1. Matrix dimension reduction: N x N instead of 2N x 2N.
2. Biot-Savart kernel evaluations: 2 * N^2 instead of 4 * N^2 (50% reduction).
3. Direct linear solve operations: O(N^3) instead of O(8 N^3) (8x reduction).
4. Nonlinear relaxation iterations: N x N downwash mat-vecs and half the polar lookups.

Once Gamma_R is solved, the full aircraft circulation and induced downwash are
losslessly reconstructed by mirroring:
    Gamma_full = [Gamma_R[::-1], Gamma_R]
    w_ind_z_full = [w_ind_z_R[::-1], w_ind_z_R]

yielding results identical to the full (2N x 2N) solve down to machine epsilon.
"""

from __future__ import annotations

from typing import Sequence, Any
import numpy as np

from ventorum.legacy.core.datatypes import (
    Aircraft,
    DiscretizedSurface,
    FlightCondition,
    LiftingSurface,
    SolverSettings,
)


def is_symmetry_compatible(
    aircraft: Aircraft | LiftingSurface | Sequence[DiscretizedSurface] | Any,
    condition: FlightCondition,
    settings: SolverSettings | None = None,
) -> tuple[bool, str]:
    """Determine whether an aerodynamic analysis case is compatible with Y=0 symmetry.

    Parameters
    ----------
    aircraft : Aircraft, LiftingSurface, or list of DiscretizedSurface
        Geometry representation.
    condition : FlightCondition
        Operating flight condition.
    settings : SolverSettings or None
        Solver knobs (checked for user toggle `use_symmetry`).

    Returns
    -------
    is_compatible : bool
        True if the problem can be solved on the half-mesh and mirrored.
    reason : str
        Detailed diagnostic explanation for compatibility or incompatibility.
    """
    # 1. User toggle in solver settings
    if settings is not None and not getattr(settings, "use_symmetry", True):
        return False, "Symmetry disabled by user (settings.use_symmetry is False)"

    # 2. Asymmetric flow: sideslip (beta)
    beta = getattr(condition, "beta", 0.0)
    if abs(beta) > 1e-12:
        return False, f"Asymmetric flow condition: sideslip beta={beta:.4f} rad ({np.degrees(beta):.2f} deg) != 0"

    beta_deg = getattr(condition, "beta_deg", 0.0)
    if abs(beta_deg) > 1e-12:
        return False, f"Asymmetric flow condition: sideslip beta_deg={beta_deg:.2f} deg != 0"

    # 3. Asymmetric flow/attitude: bank/roll angle (phi)
    phi = getattr(condition, "phi", 0.0)
    if abs(phi) > 1e-12:
        return False, f"Asymmetric attitude: roll/bank angle phi={phi:.4f} rad ({np.degrees(phi):.2f} deg) != 0"

    phi_deg = getattr(condition, "phi_deg", 0.0)
    if abs(phi_deg) > 1e-12:
        return False, f"Asymmetric attitude: roll/bank angle phi_deg={phi_deg:.2f} deg != 0"

    # 4. Geometry inspection
    if isinstance(aircraft, LiftingSurface):
        surfaces = [aircraft]
    elif isinstance(aircraft, Aircraft):
        surfaces = aircraft.surfaces
    elif isinstance(aircraft, (list, tuple)):
        # List of DiscretizedSurface
        for ds in aircraft:
            if not isinstance(ds, DiscretizedSurface):
                continue
            if getattr(ds, "is_half_mesh", False):
                continue
            n_p = len(ds.y_panels)
            if n_p % 2 != 0:
                return False, f"DiscretizedSurface '{ds.surface_name}' panel count ({n_p}) is not even"
            n_h = n_p // 2
            # Verify left and right panels mirror across y = 0
            y_left = ds.y_panels[:n_h]
            y_right = ds.y_panels[n_h:]
            y_asym = np.max(np.abs(y_left + y_right[::-1]))
            if y_asym > 1e-4:
                return False, f"DiscretizedSurface '{ds.surface_name}' y-coordinates are not symmetric across Y=0"
            z_left = ds.control_points[:n_h, 2]
            z_right = ds.control_points[n_h:, 2]
            z_asym = np.max(np.abs(z_left - z_right[::-1]))
            if z_asym > 1e-4:
                return False, f"DiscretizedSurface '{ds.surface_name}' z-coordinates are not symmetric across Y=0"
            norm_y_left = ds.normals[:n_h, 1]
            norm_y_right = ds.normals[n_h:, 1]
            norm_y_asym = np.max(np.abs(norm_y_left + norm_y_right[::-1]))
            if norm_y_asym > 1e-4:
                return False, f"DiscretizedSurface '{ds.surface_name}' normal vectors are not symmetric across Y=0"
        return True, "Pre-discretized surfaces are verified symmetric across Y=0"
    else:
        return False, f"Unrecognized geometry type: {type(aircraft)}"

    for surf in surfaces:
        # Surface must be marked symmetric
        if not getattr(surf, "is_symmetric", True):
            return False, f"Surface '{surf.name}' is asymmetric (is_symmetric is False)"

        # Root position must lie on Y = 0
        pos = getattr(surf, "position", np.zeros(3))
        if abs(pos[1]) > 1e-12:
            return False, f"Surface '{surf.name}' root position y={pos[1]:.4f} is not centered on Y=0"

        # Check for asymmetric control deflections if present
        if getattr(surf, "has_asymmetric_controls", False) or getattr(surf, "is_symmetric_controls", True) is False:
            return False, f"Surface '{surf.name}' has asymmetric control deflections"

        ctrls = getattr(surf, "control_deflections", None)
        if isinstance(ctrls, dict):
            # Asymmetric controls like ailerons produce differential deflections
            for k, val in ctrls.items():
                if any(w in k.lower() for w in ("aileron", "roll", "asym")) and abs(val) > 1e-12:
                    return False, f"Surface '{surf.name}' has non-zero asymmetric control: {k}={val}"

    return True, "Case is compatible with Y=0 symmetry plane"


def can_use_symmetry(
    aircraft: Aircraft | LiftingSurface | Sequence[DiscretizedSurface] | Any,
    condition: FlightCondition,
    settings: SolverSettings | None = None,
) -> bool:
    """Convenience boolean check for Y=0 symmetry plane eligibility."""
    compat, _ = is_symmetry_compatible(aircraft, condition, settings)
    return compat


def mirror_discretized_surface(ds_half: DiscretizedSurface) -> DiscretizedSurface:
    """Mirror a right-semi-span half mesh to reconstruct the full 3D discretized surface.

    Parameters
    ----------
    ds_half : DiscretizedSurface
        Half-mesh containing right semi-span panels (y >= 0).

    Returns
    -------
    DiscretizedSurface
        Complete mirrored surface ordered from left-tip to right-tip.
    """
    if not ds_half.is_half_mesh:
        return ds_half

    # Left nodes: mirror y, reverse order (left tip first, root last)
    left_nodes = ds_half.nodes_qc[::-1].copy()
    left_nodes[:, 1] *= -1.0

    left_cp = ds_half.control_points[::-1].copy()
    left_cp[:, 1] *= -1.0

    left_normals = ds_half.normals[::-1].copy()
    left_normals[:, 1] *= -1.0

    left_airfoils = ds_half.airfoils[::-1]
    left_y = -ds_half.y_panels[::-1]
    left_dy = ds_half.dy_panels[::-1]
    left_chords = ds_half.chords[::-1]
    left_twists = ds_half.twists[::-1]

    # Combine: left (negative y) + right (positive y, skipping shared root node)
    all_nodes = np.vstack([left_nodes, ds_half.nodes_qc[1:]])
    all_cp = np.vstack([left_cp, ds_half.control_points])
    all_normals = np.vstack([left_normals, ds_half.normals])
    all_y = np.concatenate([left_y, ds_half.y_panels])
    all_dy = np.concatenate([left_dy, ds_half.dy_panels])
    all_chords = np.concatenate([left_chords, ds_half.chords])
    all_twists = np.concatenate([left_twists, ds_half.twists])
    all_airfoils = left_airfoils + ds_half.airfoils

    panel_centers = 0.5 * (all_nodes[:-1] + all_nodes[1:])

    unique_airfoils = []
    for af in all_airfoils:
        if not any(af is u for u in unique_airfoils):
            unique_airfoils.append(af)
    airfoil_groups = [
        (uaf, np.where([af is uaf for af in all_airfoils])[0])
        for uaf in unique_airfoils
    ]

    return DiscretizedSurface(
        nodes_qc=all_nodes,
        control_points=all_cp,
        normals=all_normals,
        y_panels=all_y,
        dy_panels=all_dy,
        chords=all_chords,
        twists=all_twists,
        airfoils=all_airfoils,
        surface_name=ds_half.surface_name,
        surface_index=ds_half.surface_index,
        panel_centers_qc=panel_centers,
        has_profile_drag=ds_half.has_profile_drag,
        airfoil_groups=airfoil_groups,
        is_half_mesh=False,
    )


def extract_half_mesh_surface(ds_full: DiscretizedSurface) -> DiscretizedSurface:
    """Extract the right semi-span (y >= 0) half-mesh from a full discretized surface.

    Parameters
    ----------
    ds_full : DiscretizedSurface
        Full symmetric surface with 2N panels.

    Returns
    -------
    DiscretizedSurface
        Half-mesh with N panels (right semi-span).
    """
    if ds_full.is_half_mesh:
        return ds_full

    n_tot = len(ds_full.y_panels)
    n_semi = n_tot // 2

    # In standard left-tip -> root -> right-tip layout, right semi-span is the second half
    nodes_qc = ds_full.nodes_qc[n_semi:].copy()
    control_points = ds_full.control_points[n_semi:].copy()
    normals = ds_full.normals[n_semi:].copy()
    y_panels = ds_full.y_panels[n_semi:].copy()
    dy_panels = ds_full.dy_panels[n_semi:].copy()
    chords = ds_full.chords[n_semi:].copy()
    twists = ds_full.twists[n_semi:].copy()
    airfoils = list(ds_full.airfoils[n_semi:])
    panel_centers_qc = ds_full.panel_centers_qc[n_semi:].copy() if len(ds_full.panel_centers_qc) == n_tot else 0.5 * (nodes_qc[:-1] + nodes_qc[1:])

    unique_airfoils = []
    for af in airfoils:
        if not any(af is u for u in unique_airfoils):
            unique_airfoils.append(af)
    airfoil_groups = [
        (uaf, np.where([af is uaf for af in airfoils])[0])
        for uaf in unique_airfoils
    ]

    return DiscretizedSurface(
        nodes_qc=nodes_qc,
        control_points=control_points,
        normals=normals,
        y_panels=y_panels,
        dy_panels=dy_panels,
        chords=chords,
        twists=twists,
        airfoils=airfoils,
        surface_name=ds_full.surface_name,
        surface_index=ds_full.surface_index,
        panel_centers_qc=panel_centers_qc,
        has_profile_drag=ds_full.has_profile_drag,
        airfoil_groups=airfoil_groups,
        is_half_mesh=True,
    )


def reconstruct_full_circulation(
    Gamma_semi: np.ndarray,
    disc_surfaces_half: Sequence[DiscretizedSurface],
) -> np.ndarray:
    """Reconstruct global full-aircraft circulation vector from semi-span solution.

    Parameters
    ----------
    Gamma_semi : np.ndarray, shape (N_semi_total,)
        Circulation on right semi-span panels across all surfaces.
    disc_surfaces_half : list of DiscretizedSurface
        Half-mesh surfaces.

    Returns
    -------
    Gamma_full : np.ndarray, shape (2 * N_semi_total,)
        Full mirrored circulation vector.
    """
    offset = 0
    full_parts = []
    for ds in disc_surfaces_half:
        n_p = len(ds.y_panels)
        g_surf = Gamma_semi[offset:offset + n_p]
        # Reconstruct: left (reversed) then right
        full_parts.append(g_surf[::-1])
        full_parts.append(g_surf)
        offset += n_p
    return np.concatenate(full_parts)


def reconstruct_full_downwash(
    w_ind_z_semi: np.ndarray,
    disc_surfaces_half: Sequence[DiscretizedSurface],
) -> np.ndarray:
    """Reconstruct global full-aircraft downwash vector from semi-span solution.

    Parameters
    ----------
    w_ind_z_semi : np.ndarray, shape (N_semi_total,)
        Induced downwash vertical component on right semi-span control points.
    disc_surfaces_half : list of DiscretizedSurface
        Half-mesh surfaces.

    Returns
    -------
    w_ind_z_full : np.ndarray, shape (2 * N_semi_total,)
        Full mirrored vertical downwash vector.
    """
    offset = 0
    full_parts = []
    for ds in disc_surfaces_half:
        n_p = len(ds.y_panels)
        w_surf = w_ind_z_semi[offset:offset + n_p]
        # Downwash is symmetric: w(-y) = w(y)
        full_parts.append(w_surf[::-1])
        full_parts.append(w_surf)
        offset += n_p
    return np.concatenate(full_parts)
