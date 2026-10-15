# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Post-processing for the 3D source-doublet panel method.

This module computes surface tangential perturbation velocities from the doublet
gradient, evaluates surface pressure coefficients, integrates near-field forces
and moments, and calculates far-field Trefftz-plane induced drag.

Published sources
-----------------
* Katz, J. and Plotkin, A., 2001, Low-Speed Aerodynamics, 2nd ed.,
  Cambridge University Press, section 9.4 and chapter 12.
* Drela, M., 2014, Flight Vehicle Aerodynamics, MIT Press, chapter 5.
* Stevens, B. L., Lewis, F. L., and Johnson, E. N., 2016, Aircraft Control
  and Simulation, 3rd ed., Wiley, chapter 2.

Conventions
-----------
* Surface velocity: V = V_onset_tangential - grad(mu).
* Pressure coefficient: Cp = 1 - |V|^2 / V_inf^2.
* Pressure force on panel: dF = -Cp * q_inf * Area * n.
* Wind axes: Lift along L_hat = [-sin(alpha), 0, cos(alpha)], side force
  along Y_hat = L_hat x u_hat, drag along u_hat.
* Moments: Flight-dynamics body axes (AVL convention):
  Cl = -Mx / (q_inf * S_ref * b_ref)  (right wing down positive)
  Cm = +My / (q_inf * S_ref * c_ref)  (nose up positive)
  Cn = -Mz / (q_inf * S_ref * b_ref)  (nose right positive).
"""

from __future__ import annotations

import numpy as np

from ventorum.aero.system import freestream_direction, lift_direction
from ventorum.core.datatypes import FlightCondition
from ventorum.panel3d.wake import PanelWake


def build_edge_neighbourhood(
    panels: np.ndarray,
    wake_edges: np.ndarray | None = None,
) -> list[list[int]]:
    """Find neighbour panels that share an edge, excluding wake shedding edges.

    Parameters
    ----------
    panels : numpy.ndarray
        Corner node indices of panels with shape (Np, 4).
    wake_edges : numpy.ndarray or None, optional
        Wake shedding edges with shape (Nw, 4).

    Returns
    -------
    list of list of int
        List of neighbour panel indices for each panel.
    """
    wake_set = set()
    if wake_edges is not None and len(wake_edges) > 0:
        w_arr = np.asarray(wake_edges, dtype=int).reshape(-1, 4)
        for row in w_arr:
            na, nb = int(row[2]), int(row[3])
            wake_set.add((min(na, nb), max(na, nb)))

    edge_to_panels: dict[tuple[int, int], list[int]] = {}
    n_panels = panels.shape[0]

    for i in range(n_panels):
        row = panels[i]
        for k in range(4):
            a, b = int(row[k]), int(row[(k + 1) % 4])
            if a == b:
                continue
            edge = (min(a, b), max(a, b))
            if edge in wake_set:
                # Do not connect across the wake shedding boundary
                continue
            edge_to_panels.setdefault(edge, []).append(i)

    neighbour_sets = [set() for _ in range(n_panels)]
    for p_list in edge_to_panels.values():
        if len(p_list) == 2:
            p0, p1 = p_list[0], p_list[1]
            neighbour_sets[p0].add(p1)
            neighbour_sets[p1].add(p0)
        elif len(p_list) > 2:
            for p in p_list:
                for q in p_list:
                    if p != q:
                        neighbour_sets[p].add(q)

    return [sorted(s) for s in neighbour_sets]


def compute_surface_gradient(
    centroids: np.ndarray,
    normals: np.ndarray,
    mu: np.ndarray,
    neighbour_indices: list[list[int]],
) -> np.ndarray:
    """Compute the tangential surface gradient of doublet strength on each panel.

    A least-squares fit over neighbour panels that share an edge is solved in
    the tangent plane of the panel.

    Parameters
    ----------
    centroids : numpy.ndarray
        Panel centroid coordinates with shape (Np, 3) in m.
    normals : numpy.ndarray
        Panel unit normal vectors with shape (Np, 3).
    mu : numpy.ndarray
        Doublet strengths on panels with shape (Np,).
    neighbour_indices : list of list of int
        List of neighbour indices for each panel.

    Returns
    -------
    numpy.ndarray
        Surface gradient vectors with shape (Np, 3) in 1/m.
    """
    n_panels = centroids.shape[0]
    grad_mu = np.zeros((n_panels, 3), dtype=np.float64)

    for i in range(n_panels):
        nbrs = neighbour_indices[i]
        if len(nbrs) == 0:
            continue

        c_i = centroids[i]
        n_i = normals[i]
        dr = centroids[nbrs] - c_i[None, :]

        # Project neighbour displacement vectors onto the panel tangent plane
        dr_normal = np.sum(dr * n_i[None, :], axis=-1, keepdims=True)
        dr_tangent = dr - dr_normal * n_i[None, :]
        dmu = mu[nbrs] - mu[i]

        # Solve least-squares problem: dr_tangent @ g = dmu
        g, _, _, _ = np.linalg.lstsq(dr_tangent, dmu, rcond=None)

        # Enforce exact tangency to panel normal
        g_proj = g - np.dot(g, n_i) * n_i
        grad_mu[i] = g_proj

    return grad_mu


def compute_surface_velocity_and_cp(
    V_onset: np.ndarray,
    normals: np.ndarray,
    grad_mu: np.ndarray,
    V_inf: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute local surface velocity and pressure coefficient on each panel.

    Parameters
    ----------
    V_onset : numpy.ndarray
        Onset velocity vector at panel centroids with shape (Np, 3) in m/s.
    normals : numpy.ndarray
        Panel unit normal vectors with shape (Np, 3).
    grad_mu : numpy.ndarray
        Tangential gradient of doublet potential with shape (Np, 3) in m/s.
    V_inf : float
        Free-stream speed in m/s.

    Returns
    -------
    V_surf : numpy.ndarray
        Surface velocity vectors with shape (Np, 3) in m/s.
    cp : numpy.ndarray
        Pressure coefficients on panels with shape (Np,).
    """
    # Tangential component of onset flow: V_onset - (V_onset . n) * n
    v_norm = np.sum(V_onset * normals, axis=-1, keepdims=True)
    V_onset_t = V_onset - v_norm * normals

    # Surface velocity: V = V_onset_t - grad(mu)
    V_surf = V_onset_t - grad_mu

    speed_sq = np.sum(V_surf**2, axis=-1)
    if V_inf > 1.0e-12:
        cp = 1.0 - speed_sq / (V_inf**2)
    else:
        cp = np.zeros(V_onset.shape[0], dtype=float)

    return V_surf, cp


def integrate_forces_and_moments(
    cp: np.ndarray,
    areas: np.ndarray,
    normals: np.ndarray,
    centroids: np.ndarray,
    condition: FlightCondition,
    S_ref: float,
    b_ref: float,
    c_ref: float,
    ref_point: np.ndarray | None = None,
) -> dict[str, float]:
    """Integrate surface pressures into force and moment coefficients.

    Parameters
    ----------
    cp : numpy.ndarray
        Pressure coefficients on panels with shape (Np,).
    areas : numpy.ndarray
        Panel surface areas with shape (Np,) in m^2.
    normals : numpy.ndarray
        Panel outward unit normals with shape (Np, 3).
    centroids : numpy.ndarray
        Panel centroid coordinates with shape (Np, 3) in m.
    condition : FlightCondition
        Flight condition with alpha [rad], beta [rad], V_inf [m/s], rho [kg/m^3].
    S_ref : float
        Reference planform area in m^2.
    b_ref : float
        Reference span in m.
    c_ref : float
        Reference chord in m.
    ref_point : numpy.ndarray or None, optional
        Moment reference point with shape (3,) in m. Default is the origin.

    Returns
    -------
    dict
        Dictionary with CL, CD_pressure, CY, Cl, Cm, Cn.
    """
    rho = float(condition.rho)
    v_inf = float(condition.V_inf)
    q_inf = 0.5 * rho * v_inf**2

    # Panel force vector in geometry axes [N]: dF = -Cp * q_inf * Area * n
    panel_forces = -cp[:, None] * areas[:, None] * normals * q_inf
    total_force = np.sum(panel_forces, axis=0)

    # Reference directions
    alpha = float(condition.alpha)
    beta = float(condition.beta)
    u_hat = freestream_direction(alpha, beta)
    l_hat = lift_direction(alpha)
    y_hat = np.cross(l_hat, u_hat)

    # Aerodynamic force coefficients
    ref_force = q_inf * S_ref
    if ref_force > 1.0e-14:
        cl_val = float(np.dot(total_force, l_hat) / ref_force)
        cy_val = float(np.dot(total_force, y_hat) / ref_force)
        cd_val = float(np.dot(total_force, u_hat) / ref_force)
    else:
        cl_val = 0.0
        cy_val = 0.0
        cd_val = 0.0

    # Moments about reference point in geometry axes [N m]
    rp = np.zeros(3, dtype=float) if ref_point is None else np.asarray(ref_point, dtype=float)
    r_arm = centroids - rp[None, :]
    panel_moments = np.cross(r_arm, panel_forces)
    total_moment = np.sum(panel_moments, axis=0)

    # Convert to AVL flight-dynamics body axis moment coefficients
    if ref_force > 1.0e-14 and b_ref > 1.0e-14 and c_ref > 1.0e-14:
        c_roll = float(-total_moment[0] / (ref_force * b_ref))
        c_pitch = float(+total_moment[1] / (ref_force * c_ref))
        c_yaw = float(-total_moment[2] / (ref_force * b_ref))
    else:
        c_roll = 0.0
        c_pitch = 0.0
        c_yaw = 0.0

    return {
        "CL": cl_val,
        "CD_pressure": cd_val,
        "CY": cy_val,
        "Cl": c_roll,
        "Cm": c_pitch,
        "Cn": c_yaw,
    }


def compute_trefftz_induced_drag(
    wake: PanelWake,
    mu: np.ndarray,
    V_inf: float,
    rho: float,
    S_ref: float,
) -> float:
    """Compute far-field induced drag from trailing vortex sheet in Trefftz plane.

    Parameters
    ----------
    wake : PanelWake
        Precomputed wake geometry.
    mu : numpy.ndarray
        Doublet strengths on body panels with shape (Np,).
    V_inf : float
        Free-stream speed in m/s.
    rho : float
        Air density in kg/m^3.
    S_ref : float
        Reference area in m^2.

    Returns
    -------
    float
        Far-field induced drag coefficient CDi_trefftz.
    """
    n_strips = len(wake.wake_up)
    if n_strips == 0 or V_inf <= 1.0e-12 or S_ref <= 1.0e-12:
        return 0.0

    # Circulation of each wake strip: Gamma_s = mu_upper - mu_lower
    gamma_w = mu[wake.wake_up] - mu[wake.wake_lo]

    # Endpoints of strip segments in the Trefftz plane
    pl = wake.p_left
    pr = wake.p_right

    seg_vec = pr - pl
    seg_len = np.linalg.norm(seg_vec, axis=-1)
    safe_len = np.maximum(seg_len, 1.0e-12)

    # Segment midpoint evaluation points
    q_eval = 0.5 * (pl + pr)

    # In-plane normal vector: n = d_wake x seg_vec / seg_len
    # For body-axis wake d_wake = [1, 0, 0]:
    # [1, 0, 0] x [dx, dy, dz] = [0, -dz, dy]
    dy = seg_vec[:, 1]
    dz = seg_vec[:, 2]
    qn = np.column_stack([np.zeros_like(dy), -dz, dy]) / safe_len[:, None]

    # Trailing vortices at right endpoints (+gamma_w) and left endpoints (-gamma_w)
    pv = np.vstack([pr, pl])
    gv = np.concatenate([gamma_w, -gamma_w])

    # Core radius regularization to prevent zero-distance divisions
    rc = 0.01 * float(np.mean(safe_len))
    rc_sq = rc**2

    w_n = np.zeros(n_strips, dtype=np.float64)
    for s in range(n_strips):
        qs = q_eval[s]
        rq = qs[None, :] - pv
        r2 = rq[:, 1] ** 2 + rq[:, 2] ** 2
        denom = 2.0 * np.pi * (r2 + rc_sq)

        vy = -rq[:, 2] / denom
        vz = rq[:, 1] / denom

        v_y_ind = float(np.sum(gv * vy))
        v_z_ind = float(np.sum(gv * vz))
        w_n[s] = v_y_ind * qn[s, 1] + v_z_ind * qn[s, 2]

    # Induced drag: D_i = -0.5 * rho * sum(Gamma_s * w_n,s * l_s)
    d_ind = -0.5 * rho * float(np.sum(gamma_w * w_n * seg_len))
    q_inf = 0.5 * rho * V_inf**2
    return float(d_ind / (q_inf * S_ref))
