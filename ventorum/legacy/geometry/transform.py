# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
3D Kinematics, rigid-body rotations, and ground-effect clearance engine for Ventorum.

Provides geometry transformations ensuring that for ground-effect analysis:
1. Undisturbed flow remains strictly parallel to the ground plane (V_z = 0).
2. The aircraft geometry is rotated in 3D (pitch theta/alpha, roll phi, yaw psi)
   about an arbitrary reference point (CG, quarter-chord root, or aircraft origin).
3. Heights and ground clearances are tracked precisely (reference height, minimum
   clearance, tip clearances, and ground strike detection).
"""

from __future__ import annotations

from typing import Literal, Sequence, Any
import numpy as np

from ventorum.legacy.core.datatypes import (
    Aircraft,
    DiscretizedSurface,
    FlightCondition,
    LiftingSurface,
)
from ventorum.legacy.geometry.processing import discretize_surface


def rotation_matrix_body(
    roll_rad: float = 0.0,
    pitch_rad: float = 0.0,
    yaw_rad: float = 0.0,
) -> np.ndarray:
    """Compute standard 3-D aeronautical rotation matrix.

    Rotations are applied in aerospace order: Yaw (psi) -> Roll (phi) -> Pitch (theta).
    
    Coordinates in Ventorum:
      - x: chordwise aft (downstream)
      - y: spanwise starboard (right wing)
      - z: vertical upward

    Sign conventions:
      - Pitch theta > 0: nose up (leading edge up, trailing edge down, normal tilts forward)
      - Roll phi > 0: right wing down (starboard wing dips, port wing rises, normal tilts right)
      - Yaw psi > 0: nose right (starboard yaw)

    Parameters
    ----------
    roll_rad : float
        Roll angle phi [rad].
    pitch_rad : float
        Pitch angle theta [rad] (geometric angle of attack relative to horizontal flow).
    yaw_rad : float
        Yaw / heading angle psi [rad].

    Returns
    -------
    np.ndarray
        3x3 proper orthonormal rotation matrix R.
        For row vectors v: v_rot = v @ R.T
        For column vectors v: v_rot = R @ v
    """
    cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
    cr, sr = np.cos(roll_rad), np.sin(roll_rad)
    cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)

    # Pitch about Y (nose up: x -> +z, z -> -x)
    # [x', y', z'] = [x*cp + z*sp, y, -x*sp + z*cp]
    # Normal [0, 0, 1] becomes [sp, 0, cp], so normal.dot(V_inf_x) > 0 for pitch up
    Ry = np.array([
        [cp, 0.0, sp],
        [0.0, 1.0, 0.0],
        [-sp, 0.0, cp],
    ], dtype=float)

    # Roll about X (right wing down: y -> -z, z -> +y)
    # [x', y', z'] = [x, y*cr + z*sr, -y*sr + z*cr]
    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cr, sr],
        [0.0, -sr, cr],
    ], dtype=float)

    # Yaw about Z (nose right: x -> +y, y -> -x)
    Rz = np.array([
        [cy, -sy, 0.0],
        [sy, cy, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)

    # Combined rotation: first pitch, then roll, then yaw
    # v_intermediate = Ry @ v
    # v_rolled = Rx @ v_intermediate
    # v_final = Rz @ v_rolled
    R = Rz @ (Rx @ Ry)
    return R


def transform_discretized_surface(
    ds: DiscretizedSurface,
    R: np.ndarray,
    translation: np.ndarray | Sequence[float] = (0.0, 0.0, 0.0),
    ref_point: np.ndarray | Sequence[float] = (0.0, 0.0, 0.0),
) -> DiscretizedSurface:
    """Apply a 3-D rigid-body rotation about *ref_point* and *translation* to a panel mesh.

    Parameters
    ----------
    ds : DiscretizedSurface
        Original panel-discretized surface.
    R : np.ndarray
        3x3 rotation matrix.
    translation : array-like of float, shape (3,)
        Rigid translation vector [dx, dy, dz].
    ref_point : array-like of float, shape (3,)
        Center of rotation [x_ref, y_ref, z_ref].

    Returns
    -------
    DiscretizedSurface
        Transformed discretized surface with updated nodes, control points,
        normals, and panel geometries.
    """
    ref = np.asarray(ref_point, dtype=float).reshape(1, 3)
    trans = np.asarray(translation, dtype=float).reshape(1, 3)

    # Rotate coordinates about ref_point, then translate
    nodes_rot = ref + (ds.nodes_qc - ref) @ R.T + trans
    cp_rot = ref + (ds.control_points - ref) @ R.T + trans

    if hasattr(ds, "panel_centers_qc") and len(ds.panel_centers_qc) == len(ds.dy_panels):
        centers_rot = ref + (ds.panel_centers_qc - ref) @ R.T + trans
    else:
        centers_rot = 0.5 * (nodes_rot[:-1] + nodes_rot[1:])

    # Normals only rotate (pure direction vectors, unaffected by translation/ref_point)
    normals_rot = ds.normals @ R.T
    norm_mags = np.linalg.norm(normals_rot, axis=1, keepdims=True)
    norm_mags = np.where(norm_mags > 1e-14, norm_mags, 1.0)
    normals_rot = normals_rot / norm_mags

    # Transformed panel widths and projected lengths
    d_nodes = np.diff(nodes_rot, axis=0)
    dy_rot = d_nodes[:, 1]
    dz_rot = d_nodes[:, 2]
    # Keep panel length (arc length) positive
    ds_arc = np.sqrt(dy_rot**2 + dz_rot**2 + d_nodes[:, 0]**2)

    return DiscretizedSurface(
        nodes_qc=nodes_rot,
        control_points=cp_rot,
        normals=normals_rot,
        y_panels=centers_rot[:, 1],
        dy_panels=ds.dy_panels.copy(),  # retain nominal spanwise strip width for integration
        chords=ds.chords.copy(),
        twists=ds.twists.copy(),
        airfoils=list(ds.airfoils),
        surface_name=ds.surface_name,
        surface_index=ds.surface_index,
        panel_centers_qc=centers_rot,
        has_profile_drag=ds.has_profile_drag,
        airfoil_groups=list(ds.airfoil_groups) if hasattr(ds, "airfoil_groups") else [],
    )


def compute_ground_clearance(
    disc_surfaces: list[DiscretizedSurface],
    ground_z: float = 0.0,
    include_edges: bool = True,
) -> dict[str, Any]:
    """Compute ground clearance and strike metrics for a set of discretized surfaces.

    Parameters
    ----------
    disc_surfaces : list[DiscretizedSurface]
        List of panel-discretized surfaces in world coordinates.
    ground_z : float
        Z-coordinate of the horizontal ground plane (default 0.0).
    include_edges : bool
        If True, estimates leading and trailing edge points from chords and normals.

    Returns
    -------
    dict
        Dictionary containing:
        - 'h_min': minimum vertical clearance above ground [m]
        - 'is_strike': True if h_min <= 0 (collision/penetration)
        - 'h_tip_left': clearance of leftmost wingtip [m]
        - 'h_tip_right': clearance of rightmost wingtip [m]
        - 'min_point': 3D coordinates [x, y, z] of the lowest point [m]
        - 'clearances': array of clearances at all evaluation points [m]
    """
    all_points: list[np.ndarray] = []

    for ds in disc_surfaces:
        all_points.append(ds.nodes_qc)
        all_points.append(ds.control_points)

        if include_edges and len(ds.chords) > 0:
            # Estimate leading edge: quarter-chord minus 0.25*chord along chord line
            # Chord direction is perpendicular to normal in the x-z local plane
            centers = ds.panel_centers_qc if hasattr(ds, "panel_centers_qc") else 0.5 * (ds.nodes_qc[:-1] + ds.nodes_qc[1:])
            nx = ds.normals[:, 0]
            nz = ds.normals[:, 2]
            # In-plane tangent: cx = nz, cz = -nx
            c_tan = np.column_stack([nz, np.zeros_like(nz), -nx])
            c_tan_norm = np.linalg.norm(c_tan, axis=1, keepdims=True)
            c_tan_norm = np.where(c_tan_norm > 1e-14, c_tan_norm, 1.0)
            c_dir = c_tan / c_tan_norm

            le_pts = centers - 0.25 * ds.chords[:, None] * c_dir
            te_pts = centers + 0.75 * ds.chords[:, None] * c_dir
            all_points.append(le_pts)
            all_points.append(te_pts)

    pts = np.vstack(all_points)
    clearances = pts[:, 2] - ground_z
    min_idx = int(np.argmin(clearances))
    h_min = float(clearances[min_idx])
    min_pt = pts[min_idx]

    # Wingtip clearances from the first primary lifting surface
    primary = disc_surfaces[0]
    h_tip_left = float(primary.nodes_qc[0, 2] - ground_z)
    h_tip_right = float(primary.nodes_qc[-1, 2] - ground_z)

    return {
        "h_min": h_min,
        "is_strike": h_min <= 0.0,
        "h_tip_left": h_tip_left,
        "h_tip_right": h_tip_right,
        "min_point": min_pt,
        "clearances": clearances,
    }


def find_roll_strike_limit(
    aircraft_or_surfaces: Aircraft | LiftingSurface | list[DiscretizedSurface],
    h: float,
    alpha_deg: float = 0.0,
    ref_point: np.ndarray | Sequence[float] | None = None,
    height_ref: Literal["ref", "min", "qc", "te"] = "ref",
    max_roll_deg: float = 60.0,
    tolerance_deg: float = 0.05,
    n_panels: int = 40,
) -> float:
    """Find the critical bank/roll angle phi_strike at which a wingtip contacts the ground.

    Parameters
    ----------
    aircraft_or_surfaces : Aircraft or LiftingSurface or list[DiscretizedSurface]
        Geometry to analyze.
    h : float
        Height parameter [m].
    alpha_deg : float
        Angle of attack / pitch angle [deg].
    ref_point : array-like or None
        Center of rotation.
    height_ref : str
        Height reference definition ('ref', 'min', 'qc', 'te').
    max_roll_deg : float
        Upper search bound for roll angle in degrees.
    tolerance_deg : float
        Convergence tolerance on roll angle.
    n_panels : int
        Panels per semi-span used for discretisation if not pre-discretised.

    Returns
    -------
    float
        Critical roll angle in degrees (phi_strike).
    """
    if isinstance(aircraft_or_surfaces, (Aircraft, LiftingSurface)):
        ac = aircraft_or_surfaces if isinstance(aircraft_or_surfaces, Aircraft) else Aircraft(surfaces=[aircraft_or_surfaces])
        ac.compute_reference_values()
        base_surfaces = [
            discretize_surface(surf, n_panels=n_panels, spacing="cosine", surface_index=idx)
            for idx, surf in enumerate(ac.surfaces)
        ]
    else:
        base_surfaces = aircraft_or_surfaces

    if ref_point is None:
        ref_pt = np.array([base_surfaces[0].nodes_qc[len(base_surfaces[0].nodes_qc)//2, 0], 0.0, 0.0])
    else:
        ref_pt = np.asarray(ref_point, dtype=float)

    # Extract all critical surface points once upfront
    all_pts: list[np.ndarray] = []
    for ds in base_surfaces:
        all_pts.append(ds.nodes_qc)
        all_pts.append(ds.control_points)
        if len(ds.chords) > 0:
            centers = (
                ds.panel_centers_qc
                if hasattr(ds, "panel_centers_qc") and len(ds.panel_centers_qc) == len(ds.dy_panels)
                else 0.5 * (ds.nodes_qc[:-1] + ds.nodes_qc[1:])
            )
            nx = ds.normals[:, 0]
            nz = ds.normals[:, 2]
            c_tan = np.column_stack([nz, np.zeros_like(nz), -nx])
            c_norm = np.maximum(np.linalg.norm(c_tan, axis=1, keepdims=True), 1e-14)
            c_dir = c_tan / c_norm
            all_pts.append(centers - 0.25 * ds.chords[:, None] * c_dir)
            all_pts.append(centers + 0.75 * ds.chords[:, None] * c_dir)

    pts = np.vstack(all_pts)
    rel_pts = pts - ref_pt

    # Pitch rotation Ry(alpha) applied once
    alpha_rad = np.radians(alpha_deg)
    cp, sp = np.cos(alpha_rad), np.sin(alpha_rad)
    Ry = np.array([
        [cp, 0.0, sp],
        [0.0, 1.0, 0.0],
        [-sp, 0.0, cp],
    ], dtype=float)

    p_pitch = rel_pts @ Ry.T
    y_p = p_pitch[:, 1]
    z_p = p_pitch[:, 2]

    # Determine ground_z in the pitched zero-roll frame
    if height_ref == "ref":
        z_ground = float(ref_pt[2] - h)
    elif height_ref == "min":
        z_ground = float(ref_pt[2] + np.min(z_p) - h)
    elif height_ref == "qc":
        mid_idx = len(base_surfaces[0].nodes_qc) // 2
        qc_pitched = (base_surfaces[0].nodes_qc[mid_idx] - ref_pt) @ Ry.T
        z_ground = float(ref_pt[2] + qc_pitched[2] - h)
    elif height_ref == "te":
        mid_idx = len(base_surfaces[0].nodes_qc) // 2
        qc_pitched = (base_surfaces[0].nodes_qc[mid_idx] - ref_pt) @ Ry.T
        c_root = base_surfaces[0].chords[len(base_surfaces[0].chords)//2]
        z_ground = float(ref_pt[2] + qc_pitched[2] - 0.75 * c_root * sp - h)
    else:
        z_ground = float(ref_pt[2] - h)

    def _get_min_clearance(phi_deg_val: float) -> float:
        phi_rad = np.radians(phi_deg_val)
        sm = np.sin(phi_rad)
        cm = np.cos(phi_rad)
        # In roll rotation Rx(phi) applied after Ry(alpha):
        # z' = ref_pt[2] - y_p * sin(phi) + z_p * cos(phi) for right-wing down
        # Negative sign corresponds to right wing down (y > 0 -> z' decreases)
        z_new = ref_pt[2] - y_p * sm + z_p * cm
        return float(np.min(z_new - z_ground))

    h0 = _get_min_clearance(0.0)
    if h0 <= 0.0:
        return 0.0

    h_max = _get_min_clearance(max_roll_deg)
    if h_max > 0.0:
        return float(max_roll_deg)

    # Ultra-fast bisection (evaluates in microseconds without mesh allocations)
    low = 0.0
    high = float(max_roll_deg)
    while (high - low) > tolerance_deg:
        mid = 0.5 * (low + high)
        if _get_min_clearance(mid) <= 0.0:
            high = mid
        else:
            low = mid

    return float(0.5 * (low + high))


def prepare_ground_effect_geometry(
    geometry: Aircraft | LiftingSurface | list[DiscretizedSurface],
    h: float,
    alpha_deg: float = 0.0,
    phi_deg: float = 0.0,
    beta_deg: float = 0.0,
    ref_point: np.ndarray | Sequence[float] | None = None,
    height_ref: Literal["ref", "min", "qc", "te"] = "ref",
    n_panels: int = 80,
    spacing: str = "cosine",
) -> tuple[list[DiscretizedSurface], dict[str, Any], np.ndarray]:
    """Position and rotate an aircraft geometry for ground-effect analysis.

    Ensures that:
    1. The aircraft is rotated by pitch angle `alpha_deg`, roll angle `phi_deg`,
       and sideslip `beta_deg` about `ref_point`.
    2. The ground plane is positioned exactly at z = -h relative to the transformed
       coordinate system, allowing standard Method of Images reflection across z = -h.
    3. The undisturbed free-stream flow will be horizontal along [V_inf, 0, 0].

    Parameters
    ----------
    geometry : Aircraft or LiftingSurface or list[DiscretizedSurface]
        Aircraft configuration.
    h : float
        Height above ground [m].
    alpha_deg : float
        Pitch angle / angle of attack relative to horizontal ground [deg].
    phi_deg : float
        Roll / bank angle relative to ground [deg] (positive = right wing down).
    beta_deg : float
        Sideslip / yaw angle [deg].
    ref_point : array-like or None
        Center of rotation and moment reference point [x, y, z].
        If None, automatically chosen as root quarter-chord line.
    height_ref : {'ref', 'min', 'qc', 'te'}
        Convention for height `h`:
        - 'ref': height of *ref_point* above ground
        - 'min': minimum clearance of any point on wing above ground
        - 'qc': height of root quarter-chord above ground
        - 'te': height of root trailing-edge above ground
    n_panels : int
        Panels per semi-span.
    spacing : str
        Panel spacing rule ('cosine' or 'uniform').

    Returns
    -------
    surfaces : list[DiscretizedSurface]
        Transformed discretized surfaces ready for solver.
    clearance_info : dict
        Ground clearance, tip heights, and strike status.
    actual_ref_point : np.ndarray
        Effective reference point [x, y, z] used for moments.
    """
    if h <= 0.0:
        raise ValueError(f"Ground height h={h} must be strictly positive (h > 0).")

    # 1. Obtain baseline discretized surfaces
    if isinstance(geometry, (Aircraft, LiftingSurface)):
        ac = geometry if isinstance(geometry, Aircraft) else Aircraft(surfaces=[geometry])
        ac.compute_reference_values()
        base_surfaces = [
            discretize_surface(surf, n_panels=n_panels, spacing=spacing, surface_index=idx)
            for idx, surf in enumerate(ac.surfaces)
        ]
    else:
        base_surfaces = geometry

    # 2. Determine reference point
    if ref_point is None:
        # Default: root quarter-chord point of primary surface
        prim = base_surfaces[0]
        mid_idx = len(prim.nodes_qc) // 2
        ref_pt = np.array([prim.nodes_qc[mid_idx, 0], 0.0, prim.nodes_qc[mid_idx, 2]], dtype=float)
    else:
        ref_pt = np.asarray(ref_point, dtype=float).copy()

    # 3. Compute 3D rotation matrix
    R = rotation_matrix_body(
        roll_rad=np.radians(phi_deg),
        pitch_rad=np.radians(alpha_deg),
        yaw_rad=np.radians(beta_deg),
    )

    # 4. Single-pass transformation and ground plane positioning
    if height_ref == "ref":
        # Analytical fast path: z_ground = ref_pt[2] - h, so dz = -ref_pt[2]
        dz = -float(ref_pt[2])
        trans = np.array([0.0, 0.0, dz], dtype=float)
        final_surfaces = [
            transform_discretized_surface(ds, R, translation=trans, ref_point=ref_pt)
            for ds in base_surfaces
        ]
    else:
        # Evaluate rotated geometry directly
        rot_temp = [
            transform_discretized_surface(ds, R, translation=(0.0, 0.0, 0.0), ref_point=ref_pt)
            for ds in base_surfaces
        ]
        if height_ref == "min":
            all_z = np.concatenate([s.nodes_qc[:, 2] for s in rot_temp])
            z_ground = float(np.min(all_z) - h)
        elif height_ref == "qc":
            mid_idx = len(rot_temp[0].nodes_qc) // 2
            z_ground = float(rot_temp[0].nodes_qc[mid_idx, 2] - h)
        elif height_ref == "te":
            mid_idx = len(rot_temp[0].nodes_qc) // 2
            z_qc = rot_temp[0].nodes_qc[mid_idx, 2]
            c_root = rot_temp[0].chords[len(rot_temp[0].chords) // 2]
            z_ground = float(z_qc - 0.75 * c_root * np.sin(np.radians(alpha_deg)) - h)
        else:
            z_ground = float(ref_pt[2] - h)

        dz = -h - z_ground
        trans = np.array([0.0, 0.0, dz], dtype=float)
        if abs(dz) > 1e-14:
            for s in rot_temp:
                s.nodes_qc[:, 2] += dz
                s.control_points[:, 2] += dz
                if hasattr(s, "panel_centers_qc") and len(s.panel_centers_qc) > 0:
                    s.panel_centers_qc[:, 2] += dz
        final_surfaces = rot_temp

    ref_pt_shifted = ref_pt + trans

    # 7. Evaluate clearance metrics (ground is at z = -h)
    clearance_info = compute_ground_clearance(final_surfaces, ground_z=-h)
    clearance_info["h_specified"] = h
    clearance_info["height_ref"] = height_ref
    clearance_info["z_ground"] = -h
    clearance_info["ref_point_height"] = float(ref_pt_shifted[2] - (-h))

    return final_surfaces, clearance_info, ref_pt_shifted
