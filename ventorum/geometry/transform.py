# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Geometric transformations for vortex lattices.

Provides 3D rotation matrix construction and rigid-body transformation of
vortex lattices (panels, strips, and surface slices).
"""

from __future__ import annotations

import numpy as np

from ventorum.geometry.lattice import SurfaceSlice, VortexLattice


def rotation_matrix(
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
) -> np.ndarray:
    """Compute the 3D rotation matrix for roll, pitch, and yaw.

    Axes follow the geometry conventions (x aft, y right, z up).
    Positive roll is right wing down.
    Positive pitch is nose up: a point on the +x axis (aft) moves down (-z).
    Positive yaw is nose right: a point on the +x axis (aft) moves left (-y).
    So a lattice pitched by *pitch* in a flow at alpha 0 is the original
    lattice at alpha = *pitch*, and a lattice yawed by *yaw* in a flow at
    beta 0 is the original lattice at beta = -*yaw* (positive beta is wind
    from the right, see :func:`ventorum.aero.system.freestream_direction`).

    The order of the rotations is yaw, then pitch, then roll.
    Each rotation is about the axes of the geometry:

        v_new = R_x(roll) @ R_y(pitch) @ R_z(yaw) @ v

    Elementary matrices
    -------------------
    Yaw about z by yaw:

        R_z = [[ cos(yaw), sin(yaw), 0],
               [-sin(yaw), cos(yaw), 0],
               [ 0,        0,        1]]

    Pitch about y by pitch:

        R_y = [[ cos(pitch), 0, sin(pitch)],
               [ 0,          1, 0         ],
               [-sin(pitch), 0, cos(pitch)]]

    Roll about x by roll:

        R_x = [[1,  0,         0        ],
               [0,  cos(roll), sin(roll)],
               [0, -sin(roll), cos(roll)]]

    The complete product is:

        R = R_x @ R_y @ R_z

    Parameters
    ----------
    roll : float, optional
        Roll angle [rad]. Default is 0.0.
    pitch : float, optional
        Pitch angle [rad]. Default is 0.0.
    yaw : float, optional
        Yaw angle [rad]. Default is 0.0.

    Returns
    -------
    numpy.ndarray
        Orthonormal rotation matrix of shape (3, 3) with determinant +1.
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cr, sr],
        [0.0, -sr, cr],
    ], dtype=float)

    Ry = np.array([
        [cp, 0.0, sp],
        [0.0, 1.0, 0.0],
        [-sp, 0.0, cp],
    ], dtype=float)

    Rz = np.array([
        [cy, sy, 0.0],
        [-sy, cy, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)

    return Rx @ (Ry @ Rz)


def transform_lattice(
    lattice: VortexLattice,
    rotation: np.ndarray | None = None,
    about: np.ndarray | None = None,
    translation: np.ndarray | None = None,
) -> VortexLattice:
    """Transform a vortex lattice by rotation and translation.

    Applies the rigid-body transformation:

        x_new = R @ (x - about) + about + translation

    to every point field, and applies R to every direction field.
    Scalar fields (chords, widths, areas, fractions, indices) do not change.
    The input lattice and its arrays are not modified.

    Parameters
    ----------
    lattice : VortexLattice
        The input vortex lattice.
    rotation : numpy.ndarray or None, optional
        Orthonormal rotation matrix of shape (3, 3) with determinant +1.
        If None, the identity matrix is used.
    about : numpy.ndarray or None, optional
        Point about which the rotation is applied [m], shape (3,).
        If None, the origin [0, 0, 0] is used.
    translation : numpy.ndarray or None, optional
        Translation vector [m], shape (3,).
        If None, zero translation [0, 0, 0] is used.

    Returns
    -------
    VortexLattice
        New vortex lattice with transformed geometry.

    Raises
    ------
    ValueError
        If rotation is not a 3x3 orthonormal matrix with determinant +1
        (tolerance 1e-10).
    ValueError
        If about or translation cannot be cast to shape (3,).
    """
    if rotation is None:
        R = np.eye(3, dtype=float)
    else:
        R = np.asarray(rotation, dtype=float)
        if R.shape != (3, 3):
            raise ValueError(f"Rotation matrix must have shape (3, 3), got {R.shape}.")
        if not np.all(np.isfinite(R)):
            raise ValueError("Rotation matrix contains non-finite values.")
        ortho_err = float(np.max(np.abs(R @ R.T - np.eye(3))))
        if ortho_err > 1e-10:
            raise ValueError(
                f"Rotation matrix is not orthonormal (max error {ortho_err:.3e} > 1e-10)."
            )
        det = float(np.linalg.det(R))
        if abs(det - 1.0) > 1e-10:
            raise ValueError(
                f"Rotation matrix determinant must be +1, got {det:.6f} "
                f"(error {abs(det - 1.0):.3e} > 1e-10)."
            )

    if about is None:
        about_vec = np.zeros(3, dtype=float)
    else:
        about_vec = np.asarray(about, dtype=float).ravel()
        if about_vec.shape != (3,):
            raise ValueError(f"about must have shape (3,), got {about_vec.shape}.")

    if translation is None:
        trans_vec = np.zeros(3, dtype=float)
    else:
        trans_vec = np.asarray(translation, dtype=float).ravel()
        if trans_vec.shape != (3,):
            raise ValueError(f"translation must have shape (3,), got {trans_vec.shape}.")

    def _transform_points(pts: np.ndarray) -> np.ndarray:
        return (pts - about_vec) @ R.T + about_vec + trans_vec

    def _transform_directions(dirs: np.ndarray) -> np.ndarray:
        return dirs @ R.T

    # Check whether mirror symmetry across y = 0 is preserved.
    # Pure pitch about the y-axis keeps y untouched: R_y maps y to y.
    is_pure_pitch = (
        abs(R[1, 1] - 1.0) <= 1e-12
        and abs(R[0, 1]) <= 1e-12
        and abs(R[2, 1]) <= 1e-12
        and abs(R[1, 0]) <= 1e-12
        and abs(R[1, 2]) <= 1e-12
    )
    no_y_translation = abs(trans_vec[1]) <= 1e-12

    if is_pure_pitch and no_y_translation:
        panel_mirror = lattice.panel_mirror.copy()
        strip_mirror = lattice.strip_mirror.copy()
    else:
        panel_mirror = np.full_like(lattice.panel_mirror, -1)
        strip_mirror = np.full_like(lattice.strip_mirror, -1)

    new_surfaces: list[SurfaceSlice] = []
    for s in lattice.surfaces:
        if isinstance(s, SurfaceSlice):
            new_surfaces.append(SurfaceSlice(
                name=s.name,
                index=s.index,
                is_symmetric=s.is_symmetric,
                strips=s.strips,
                edge_le=_transform_points(s.edge_le),
                edge_te=_transform_points(s.edge_te),
                edge_chord=s.edge_chord.copy(),
                edge_twist=s.edge_twist.copy(),
            ))
        elif hasattr(s, "nodes_qc"):
            from ventorum.core.datatypes import DiscretizedSurface
            new_surfaces.append(DiscretizedSurface(
                nodes_qc=_transform_points(s.nodes_qc),
                control_points=_transform_points(s.control_points),
                normals=_transform_directions(s.normals),
                y_panels=s.y_panels.copy(),
                dy_panels=s.dy_panels.copy(),
                chords=s.chords.copy(),
                twists=s.twists.copy(),
                airfoils=list(s.airfoils),
                surface_name=s.surface_name,
                surface_index=s.surface_index,
                panel_centers_qc=_transform_points(s.panel_centers_qc),
                has_profile_drag=s.has_profile_drag,
                airfoil_groups=list(s.airfoil_groups),
                is_half_mesh=s.is_half_mesh,
                nodes_le=_transform_points(s.nodes_le) if s.nodes_le is not None else None,
                nodes_te=_transform_points(s.nodes_te) if s.nodes_te is not None else None,
            ))

    return VortexLattice(
        collocation=lattice.collocation,
        n_chord=lattice.n_chord,
        a=_transform_points(lattice.a),
        b=_transform_points(lattice.b),
        a_te=_transform_points(lattice.a_te),
        b_te=_transform_points(lattice.b_te),
        cp=_transform_points(lattice.cp),
        normal_bc=_transform_directions(lattice.normal_bc),
        panel_strip=lattice.panel_strip.copy(),
        panel_mirror=panel_mirror,
        panel_length=lattice.panel_length.copy(),
        strip_surface=lattice.strip_surface.copy(),
        le_left=_transform_points(lattice.le_left),
        le_right=_transform_points(lattice.le_right),
        te_left=_transform_points(lattice.te_left),
        te_right=_transform_points(lattice.te_right),
        qc_mid=_transform_points(lattice.qc_mid),
        dl=_transform_directions(lattice.dl),
        chord=lattice.chord.copy(),
        twist=lattice.twist.copy(),
        width=lattice.width.copy(),
        area=lattice.area.copy(),
        chord_dir=_transform_directions(lattice.chord_dir),
        normal=_transform_directions(lattice.normal),
        normal_bc_strip=_transform_directions(lattice.normal_bc_strip),
        a0=lattice.a0.copy(),
        alpha_L0=lattice.alpha_L0.copy(),
        Cd0=lattice.Cd0.copy(),
        Cm0=lattice.Cm0.copy(),
        airfoils=list(lattice.airfoils),
        strip_mirror=strip_mirror,
        strip_is_right=lattice.strip_is_right.copy(),
        eta=lattice.eta.copy(),
        cp_frac=lattice.cp_frac.copy(),
        surfaces=new_surfaces,
        strip_core_group=(
            lattice.strip_core_group.copy()
            if lattice.strip_core_group is not None
            else None
        ),
        join_warnings=list(lattice.join_warnings),
        kernel_cache=None,
    )
