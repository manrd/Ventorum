# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Ventorum-2D: two-dimensional inviscid panel solver with ground effect.

This module implements the 2D linear-vorticity stream-function panel method
supporting sharp and blunt trailing edges, multiple bodies, source transpiration,
and ground effect through the method of images.

References
----------
J. Katz and A. Plotkin, 'Low-Speed Aerodynamics', 2nd ed., Cambridge University
Press, 2001, chapter 11.
M. Drela, 'XFOIL: An Analysis and Design System for Low Reynolds Number Airfoils',
Lecture Notes in Engineering, Vol. 54, Springer, 1989, pp. 1-12.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg
import torch

from ventorum.panel2d.influence import (
    compute_panel_local_coordinates_numpy,
    compute_panel_local_coordinates_torch,
    linear_vortex_stream_influence,
    source_panel_stream_influence,
)


@dataclass
class Panel2DResult:
    """Aerodynamic results of the 2D panel method.

    Attributes
    ----------
    alpha : float, np.ndarray, or torch.Tensor
        Angle of attack [deg].
    Cl : float, np.ndarray, or torch.Tensor
        Total lift coefficient calculated from circulation (Kutta-Joukowski).
    Cl_pressure : float, np.ndarray, or torch.Tensor
        Total lift coefficient calculated from surface pressure integration.
    Cd_pressure : float, np.ndarray, or torch.Tensor
        Total pressure drag coefficient from surface pressure integration.
    Cm : float, np.ndarray, or torch.Tensor
        Pitching moment coefficient about x_ref (nose up positive, in the
        body-0 frame before rotation).
    Cp : np.ndarray, torch.Tensor, or list
        Surface pressure coefficients at the nodes.
    ue : np.ndarray, torch.Tensor, or list
        Surface tangential velocity normalized by free-stream speed at the nodes.
    gamma : np.ndarray, torch.Tensor, or list
        Nodal vortex strengths normalized by free-stream speed.
    coordinates : np.ndarray, torch.Tensor, or list
        Node coordinates as solved [m] (rotated and placed in ground effect).
    circulation : float, np.ndarray, or torch.Tensor
        Circulation per body [m^2/s] normalized by V_inf.
    """

    alpha: float | np.ndarray | torch.Tensor
    Cl: float | np.ndarray | torch.Tensor
    Cl_pressure: float | np.ndarray | torch.Tensor
    Cd_pressure: float | np.ndarray | torch.Tensor
    Cm: float | np.ndarray | torch.Tensor
    Cp: np.ndarray | torch.Tensor | list
    ue: np.ndarray | torch.Tensor | list
    gamma: np.ndarray | torch.Tensor | list
    coordinates: np.ndarray | torch.Tensor | list
    circulation: float | np.ndarray | torch.Tensor


def _build_body_panels_numpy(
    body_nodes: np.ndarray,
    start_node_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build panels and collocation points for a body in NumPy."""
    n_nodes = len(body_nodes)
    chord = float(np.max(body_nodes[:, 0]) - np.min(body_nodes[:, 0]))
    if chord <= 0.0:
        chord = 1.0

    gap = float(np.hypot(body_nodes[0, 0] - body_nodes[-1, 0], body_nodes[0, 1] - body_nodes[-1, 1]))
    is_blunt = gap > 1e-4 * chord

    p_starts = [body_nodes[:-1]]
    p_ends = [body_nodes[1:]]
    idx_1 = [np.arange(start_node_idx, start_node_idx + n_nodes - 1)]
    idx_2 = [np.arange(start_node_idx + 1, start_node_idx + n_nodes)]

    colloc = body_nodes.copy()
    if is_blunt:
        # Close blunt trailing edge with base panel from last node to first node
        p_starts.append(body_nodes[-1:, :])
        p_ends.append(body_nodes[0:1, :])
        idx_1.append(np.array([start_node_idx + n_nodes - 1]))
        idx_2.append(np.array([start_node_idx]))
    else:
        # Sharp trailing edge: if endpoints coincide, perturb collocation slightly
        if np.allclose(body_nodes[0], body_nodes[-1], atol=1e-12):
            colloc[0] = body_nodes[0] + 1e-2 * (body_nodes[1] - body_nodes[0])
            colloc[-1] = body_nodes[-1] + 1e-2 * (body_nodes[-2] - body_nodes[-1])

    all_p_start = np.vstack(p_starts)
    all_p_end = np.vstack(p_ends)
    all_idx1 = np.concatenate(idx_1)
    all_idx2 = np.concatenate(idx_2)

    return colloc, all_p_start, all_p_end, all_idx1, all_idx2


def _build_body_panels_torch(
    body_nodes: torch.Tensor,
    start_node_idx: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build panels and collocation points for a body in PyTorch."""
    sample = body_nodes if body_nodes.ndim == 2 else body_nodes[0]
    n_nodes = sample.shape[0]
    chord = float(torch.max(sample[:, 0]).item() - torch.min(sample[:, 0]).item())
    if chord <= 0.0:
        chord = 1.0

    gap = float(torch.hypot(sample[0, 0] - sample[-1, 0], sample[0, 1] - sample[-1, 1]).item())
    is_blunt = gap > 1e-4 * chord

    p_starts = [body_nodes[..., :-1, :]]
    p_ends = [body_nodes[..., 1:, :]]
    idx_1 = [torch.arange(start_node_idx, start_node_idx + n_nodes - 1, device=body_nodes.device)]
    idx_2 = [torch.arange(start_node_idx + 1, start_node_idx + n_nodes, device=body_nodes.device)]

    colloc = body_nodes.clone()
    if is_blunt:
        p_starts.append(body_nodes[..., -1:, :])
        p_ends.append(body_nodes[..., 0:1, :])
        idx_1.append(torch.tensor([start_node_idx + n_nodes - 1], device=body_nodes.device))
        idx_2.append(torch.tensor([start_node_idx], device=body_nodes.device))
    else:
        if torch.allclose(sample[0, :], sample[-1, :], atol=1e-12):
            colloc[..., 0, :] = body_nodes[..., 0, :] + 1e-2 * (body_nodes[..., 1, :] - body_nodes[..., 0, :])
            colloc[..., -1, :] = body_nodes[..., -1, :] + 1e-2 * (body_nodes[..., -2, :] - body_nodes[..., -1, :])

    all_p_start = torch.cat(p_starts, dim=-2)
    all_p_end = torch.cat(p_ends, dim=-2)
    all_idx1 = torch.cat(idx_1, dim=-1)
    all_idx2 = torch.cat(idx_2, dim=-1)

    return colloc, all_p_start, all_p_end, all_idx1, all_idx2


def solve_inviscid(
    bodies: np.ndarray | list[np.ndarray] | torch.Tensor | list[torch.Tensor],
    alpha_deg: float | list[float] | np.ndarray | torch.Tensor,
    *,
    height: float | None = None,
    height_reference: str = "pivot",
    pivot: tuple[float, float] | None = None,
    sigma: np.ndarray | list[np.ndarray] | torch.Tensor | None = None,
    x_ref: tuple[float, float] = (0.25, 0.0),
    chord_ref: float | None = None,
    backend: str = "numpy",
    device: str | torch.device | None = None,
    dtype: np.dtype | torch.dtype | None = None,
) -> Panel2DResult:
    """Solve 2D inviscid flow over one or more bodies, with ground effect.

    Parameters
    ----------
    bodies : np.ndarray, list of np.ndarray, torch.Tensor, or list of torch.Tensor
        One or more bodies, each an (N, 2) array in Selig order.
    alpha_deg : float, list of float, np.ndarray, or torch.Tensor
        Angle of attack [deg].
    height : float, optional
        Height above flat ground at y=0 [m]. If None, solves in free air.
    height_reference : {"pivot", "trailing_edge"}, optional
        Height reference point. Default is "pivot".
    pivot : tuple of float, optional
        Pivot point (x, y) [m] for geometry rotation. Defaults to quarter chord of body 0.
    sigma : np.ndarray, list of np.ndarray, or torch.Tensor, optional
        Prescribed constant source strengths on panels. Default is None (zero).
    x_ref : tuple of float, optional
        Moment reference point [m] in body-0 frame before rotation. Default is (0.25, 0.0).
    chord_ref : float, optional
        Reference chord [m]. Defaults to chord of body 0.
    backend : {"numpy", "torch"}, optional
        Computational backend. Default is "numpy".
    device : str or torch.device, optional
        Device for PyTorch backend ("cpu" or "cuda").
    dtype : np.dtype or torch.dtype, optional
        Data type. Defaults to float64.

    Returns
    -------
    Panel2DResult
        Aerodynamic results and node distributions.
    """
    if backend not in ("numpy", "torch"):
        raise ValueError(f"Unknown backend '{backend}'. Supported backends: 'numpy', 'torch'.")

    # Normalize bodies into a list
    if isinstance(bodies, (np.ndarray, torch.Tensor)):
        bodies_list = [bodies]
    elif isinstance(bodies, (list, tuple)):
        bodies_list = list(bodies)
    else:
        raise TypeError("bodies must be an (N, 2) array or list of (N, 2) arrays.")

    is_scalar_alpha = np.ndim(alpha_deg) == 0
    if backend == "numpy":
        alphas = np.atleast_1d(np.asarray(alpha_deg, dtype=np.float64))
        return _solve_numpy(
            bodies_list,
            alphas,
            height=height,
            height_reference=height_reference,
            pivot=pivot,
            sigma=sigma,
            x_ref=x_ref,
            chord_ref=chord_ref,
            is_scalar_alpha=is_scalar_alpha,
        )

    # PyTorch backend
    if device is None:
        device = "cpu"
    if dtype is None:
        dtype = torch.float64

    alphas_th = torch.as_tensor(alpha_deg, dtype=dtype, device=device).reshape(-1)
    return _solve_torch(
        bodies_list,
        alphas_th,
        height=height,
        height_reference=height_reference,
        pivot=pivot,
        sigma=sigma,
        x_ref=x_ref,
        chord_ref=chord_ref,
        device=device,
        dtype=dtype,
        is_scalar_alpha=is_scalar_alpha,
    )


def _wake_source_stream_influence(
    xi: np.ndarray,
    eta: np.ndarray,
    length: np.ndarray,
) -> np.ndarray:
    """Calculate stream function influence of wake source panels without branch cut.

    The branch cut of the arctan integral is directed downstream along +xi so
    that the stream function upstream across the airfoil remains continuous.
    """
    t1 = xi
    t2 = xi - length
    r1_sq = np.maximum(t1 * t1 + eta * eta, 1e-30)
    r2_sq = np.maximum(t2 * t2 + eta * eta, 1e-30)
    safe_t1 = np.where(np.abs(t1) < 1e-15, -1e-15, t1)
    safe_t2 = np.where(np.abs(t2) < 1e-15, -1e-15, t2)
    at1 = np.where(t1 < 0.0, np.arctan(eta / safe_t1), np.arctan2(eta, t1))
    at2 = np.where(t2 < 0.0, np.arctan(eta / safe_t2), np.arctan2(eta, t2))
    val1 = t1 * at1 + 0.5 * eta * np.log(r1_sq)
    val2 = t2 * at2 + 0.5 * eta * np.log(r2_sq)
    return (val1 - val2) / (2.0 * np.pi)


def compute_mass_defect_matrix(
    bodies: np.ndarray | list[np.ndarray],
    *,
    alpha_deg: float = 0.0,
    height: float | None = None,
    height_reference: str = "pivot",
    pivot: tuple[float, float] | None = None,
    chord_ref: float | None = None,
    backend: str = "numpy",
    wake_panels: np.ndarray | tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Calculate the mass-defect influence matrix D and inviscid edge velocity.

    Parameters
    ----------
    bodies : np.ndarray or list of np.ndarray
        Airfoil contour coordinates in Selig order [m].
    alpha_deg : float, optional
        Angle of attack [deg]. Default is 0.0.
    height : float, optional
        Height above flat ground at y=0 [m], or None for free air.
    height_reference : {"pivot", "trailing_edge"}, optional
        Height reference point. Default is "pivot".
    pivot : tuple of float, optional
        Pivot point (x, y) [m] for geometry rotation. Defaults to quarter chord.
    chord_ref : float, optional
        Reference chord [m]. Defaults to chord of body 0.
    backend : {"numpy"}, optional
        Computational backend. Default is "numpy".
    wake_panels : np.ndarray or tuple of np.ndarray, optional
        Wake panel node coordinates of shape (Nw + 1, 2) or tuple of (start, end)
        points of shape (Nw, 2) for wake displacement source modeling.

    Returns
    -------
    tuple of (D, ue_inv, info)
        D: mass-defect matrix of shape (total_nodes, total_panels).
        ue_inv: inviscid surface velocity at nodes [dimensionless].
        info: dictionary containing geometric panels, collocation points, and solutions.
    """
    if backend != "numpy":
        raise ValueError(f"Only backend='numpy' is currently supported in compute_mass_defect_matrix, got '{backend}'.")

    if isinstance(bodies, np.ndarray):
        bodies_np = [np.asarray(bodies, dtype=np.float64)]
    elif isinstance(bodies, (list, tuple)):
        bodies_np = [np.asarray(b, dtype=np.float64) for b in bodies]
    else:
        raise TypeError("bodies must be an (N, 2) array or list of (N, 2) arrays.")

    n_bodies = len(bodies_np)
    b0_xle = float(np.min(bodies_np[0][:, 0]))
    b0_xte = float(np.max(bodies_np[0][:, 0]))
    b0_chord = b0_xte - b0_xle
    if chord_ref is None:
        chord_ref = b0_chord if b0_chord > 0.0 else 1.0

    if pivot is None:
        pivot = (b0_xle + 0.25 * chord_ref, 0.0)
    px, py = pivot

    node_counts = [len(b) for b in bodies_np]
    total_nodes = sum(node_counts)

    body_node_ranges = []
    curr = 0
    for nc in node_counts:
        body_node_ranges.append((curr, curr + nc))
        curr += nc

    if height is None:
        colloc_list, p_start_list, p_end_list, idx1_list, idx2_list = [], [], [], [], []
        panel_body_ids = []
        curr_node = 0
        for b_id, b in enumerate(bodies_np):
            c, ps, pe, i1, i2 = _build_body_panels_numpy(b, curr_node)
            colloc_list.append(c)
            p_start_list.append(ps)
            p_end_list.append(pe)
            idx1_list.append(i1)
            idx2_list.append(i2)
            panel_body_ids.extend([b_id] * len(ps))
            curr_node += len(b)

        colloc_all = np.vstack(colloc_list)
        p_start_all = np.vstack(p_start_list)
        p_end_all = np.vstack(p_end_list)
        idx1_all = np.concatenate(idx1_list)
        idx2_all = np.concatenate(idx2_list)
        n_unknowns = total_nodes + n_bodies

        xi, eta, p_len, p_th = compute_panel_local_coordinates_numpy(colloc_all, p_start_all, p_end_all)
        eta_clean = np.where(np.abs(eta) < 1e-12, 0.0, eta)
        p_th_unwrapped = np.unwrap(p_th.ravel())[np.newaxis, :]
        psi1, psi2 = linear_vortex_stream_influence(xi, eta_clean, p_len)
        psi_s_all = source_panel_stream_influence(xi, eta_clean, p_len, p_th_unwrapped)

        A = np.zeros((n_unknowns, n_unknowns), dtype=np.float64)
        np.add.at(A[:total_nodes, :total_nodes], (slice(None), idx1_all), psi1)
        np.add.at(A[:total_nodes, :total_nodes], (slice(None), idx2_all), psi2)

        for b_id, (n_start, n_end) in enumerate(body_node_ranges):
            A[n_start:n_end, total_nodes + b_id] = -1.0
            kutta_row = total_nodes + b_id
            A[kutta_row, n_start] = 1.0
            A[kutta_row, n_end - 1] = 1.0

        lu, piv = scipy.linalg.lu_factor(A)

        rhs_x = np.zeros(n_unknowns, dtype=np.float64)
        rhs_y = np.zeros(n_unknowns, dtype=np.float64)
        rhs_x[:total_nodes] = -colloc_all[:, 1]
        rhs_y[:total_nodes] = colloc_all[:, 0]

        sol_x = scipy.linalg.lu_solve((lu, piv), rhs_x)
        sol_y = scipy.linalg.lu_solve((lu, piv), rhs_y)

        p_start_w, p_end_w = None, None
        if wake_panels is not None:
            if isinstance(wake_panels, tuple):
                p_start_w, p_end_w = wake_panels
            else:
                p_start_w = wake_panels[:-1]
                p_end_w = wake_panels[1:]
            xi_w, eta_w, p_len_w, _ = compute_panel_local_coordinates_numpy(colloc_all, p_start_w, p_end_w)
            eta_w_clean = np.where(np.abs(eta_w) < 1e-12, 0.0, eta_w)
            psi_s_wake = _wake_source_stream_influence(xi_w, eta_w_clean, p_len_w)
            psi_s_combined = np.hstack([psi_s_all, psi_s_wake])
        else:
            psi_s_combined = psi_s_all

        rhs_S = np.zeros((n_unknowns, psi_s_combined.shape[1]), dtype=np.float64)
        rhs_S[:total_nodes, :] = -psi_s_combined
        sol_S = scipy.linalg.lu_solve((lu, piv), rhs_S)
        D = sol_S[:total_nodes, :]

        alpha_rad = np.radians(alpha_deg)
        sol_inv = np.cos(alpha_rad) * sol_x + np.sin(alpha_rad) * sol_y
        ue_inv = sol_inv[:total_nodes]

        info = {
            "sol_x": sol_x,
            "sol_y": sol_y,
            "colloc": colloc_all,
            "p_start": p_start_all,
            "p_end": p_end_all,
            "p_len": p_len.ravel(),
            "p_th": p_th.ravel(),
            "idx1": idx1_all,
            "idx2": idx2_all,
            "chord_ref": chord_ref,
            "pivot": pivot,
            "body_node_ranges": body_node_ranges,
            "panel_body_ids": panel_body_ids,
            "height": None,
            "wake_p_start": p_start_w,
            "wake_p_end": p_end_w,
            "n_wake_panels": len(p_start_w) if p_start_w is not None else 0,
        }
        return D, ue_inv, info

    # Ground effect
    a_rad = np.radians(alpha_deg)
    cos_a = np.cos(a_rad)
    sin_a = np.sin(a_rad)

    rot_bodies = []
    for b in bodies_np:
        dx = b[:, 0] - px
        dy = b[:, 1] - py
        xr = px + dx * cos_a + dy * sin_a
        yr = py - dx * sin_a + dy * cos_a
        rot_bodies.append(np.column_stack([xr, yr]))

    if height_reference == "pivot":
        delta_y = height - py
    elif height_reference == "trailing_edge":
        te_y_vals = [b[0, 1] for b in rot_bodies] + [b[-1, 1] for b in rot_bodies]
        min_te_y = float(min(te_y_vals))
        delta_y = height - min_te_y
    else:
        raise ValueError(f"Unknown height_reference '{height_reference}'.")

    ge_bodies = [np.column_stack([b[:, 0], b[:, 1] + delta_y]) for b in rot_bodies]
    for b in ge_bodies:
        if np.min(b[:, 1]) <= 0.0:
            raise ValueError("Geometry under or touching the ground")

    colloc_list, p_start_list, p_end_list, idx1_list, idx2_list = [], [], [], [], []
    panel_body_ids = []
    curr_node = 0
    for b_id, b in enumerate(ge_bodies):
        c, ps, pe, i1, i2 = _build_body_panels_numpy(b, curr_node)
        colloc_list.append(c)
        p_start_list.append(ps)
        p_end_list.append(pe)
        idx1_list.append(i1)
        idx2_list.append(i2)
        panel_body_ids.extend([b_id] * len(ps))
        curr_node += len(b)

    colloc_ge = np.vstack(colloc_list)
    p_start_ge = np.vstack(p_start_list)
    p_end_ge = np.vstack(p_end_list)
    idx1_ge = np.concatenate(idx1_list)
    idx2_ge = np.concatenate(idx2_list)
    n_unknowns = total_nodes + n_bodies

    p_start_img = p_start_ge.copy()
    p_start_img[:, 1] *= -1.0
    p_end_img = p_end_ge.copy()
    p_end_img[:, 1] *= -1.0

    xi_r, eta_r, len_r, th_r = compute_panel_local_coordinates_numpy(colloc_ge, p_start_ge, p_end_ge)
    eta_r_clean = np.where(np.abs(eta_r) < 1e-12, 0.0, eta_r)
    th_r_unwrapped = np.unwrap(th_r.ravel())[np.newaxis, :]
    psi1_r, psi2_r = linear_vortex_stream_influence(xi_r, eta_r_clean, len_r)

    xi_i, eta_i, len_i, th_i = compute_panel_local_coordinates_numpy(colloc_ge, p_start_img, p_end_img)
    eta_i_clean = np.where(np.abs(eta_i) < 1e-12, 0.0, eta_i)
    th_i_unwrapped = np.unwrap(th_i.ravel())[np.newaxis, :]
    psi1_i, psi2_i = linear_vortex_stream_influence(xi_i, eta_i_clean, len_i)

    A_ge = np.zeros((n_unknowns, n_unknowns), dtype=np.float64)
    np.add.at(A_ge[:total_nodes, :total_nodes], (slice(None), idx1_ge), psi1_r - psi1_i)
    np.add.at(A_ge[:total_nodes, :total_nodes], (slice(None), idx2_ge), psi2_r - psi2_i)

    for b_id, (n_start, n_end) in enumerate(body_node_ranges):
        A_ge[n_start:n_end, total_nodes + b_id] = -1.0
        kutta_row = total_nodes + b_id
        A_ge[kutta_row, n_start] = 1.0
        A_ge[kutta_row, n_end - 1] = 1.0

    rhs_ge = np.zeros(n_unknowns, dtype=np.float64)
    rhs_ge[:total_nodes] = -colloc_ge[:, 1]

    lu, piv = scipy.linalg.lu_factor(A_ge)
    sol_ge = scipy.linalg.lu_solve((lu, piv), rhs_ge)
    ue_inv = sol_ge[:total_nodes]

    psi_s_r = source_panel_stream_influence(xi_r, eta_r_clean, len_r, th_r_unwrapped)
    psi_s_i = source_panel_stream_influence(xi_i, eta_i_clean, len_i, th_i_unwrapped)

    p_start_w, p_end_w = None, None
    if wake_panels is not None:
        if isinstance(wake_panels, tuple):
            p_start_w, p_end_w = wake_panels
        else:
            p_start_w = wake_panels[:-1]
            p_end_w = wake_panels[1:]
        dx_w = p_start_w[:, 0] - px
        dy_w = p_start_w[:, 1] - py
        psw_r = np.column_stack([px + dx_w * cos_a + dy_w * sin_a, py - dx_w * sin_a + dy_w * cos_a + delta_y])
        dx_we = p_end_w[:, 0] - px
        dy_we = p_end_w[:, 1] - py
        pew_r = np.column_stack([px + dx_we * cos_a + dy_we * sin_a, py - dx_we * sin_a + dy_we * cos_a + delta_y])

        psw_img = psw_r.copy()
        psw_img[:, 1] *= -1.0
        pew_img = pew_r.copy()
        pew_img[:, 1] *= -1.0

        xi_wr, eta_wr, len_wr, _ = compute_panel_local_coordinates_numpy(colloc_ge, psw_r, pew_r)
        psi_s_wr = _wake_source_stream_influence(xi_wr, np.where(np.abs(eta_wr) < 1e-12, 0.0, eta_wr), len_wr)

        xi_wi, eta_wi, len_wi, _ = compute_panel_local_coordinates_numpy(colloc_ge, psw_img, pew_img)
        psi_s_wi = _wake_source_stream_influence(xi_wi, np.where(np.abs(eta_wi) < 1e-12, 0.0, eta_wi), len_wi)

        psi_s_wake_ge = psi_s_wr + psi_s_wi
        psi_s_ge_combined = np.hstack([psi_s_r + psi_s_i, psi_s_wake_ge])
    else:
        psi_s_ge_combined = psi_s_r + psi_s_i

    rhs_ge_S = np.zeros((n_unknowns, psi_s_ge_combined.shape[1]), dtype=np.float64)
    rhs_ge_S[:total_nodes, :] = -psi_s_ge_combined
    sol_S = scipy.linalg.lu_solve((lu, piv), rhs_ge_S)
    D = sol_S[:total_nodes, :]

    info = {
        "colloc": colloc_ge,
        "p_start": p_start_ge,
        "p_end": p_end_ge,
        "p_len": len_r.ravel(),
        "p_th": th_r.ravel(),
        "idx1": idx1_ge,
        "idx2": idx2_ge,
        "chord_ref": chord_ref,
        "pivot": pivot,
        "body_node_ranges": body_node_ranges,
        "panel_body_ids": panel_body_ids,
        "height": height,
        "wake_p_start": p_start_w,
        "wake_p_end": p_end_w,
        "n_wake_panels": len(p_start_w) if p_start_w is not None else 0,
    }
    return D, ue_inv, info


def _solve_numpy(
    bodies: list[np.ndarray],
    alphas: np.ndarray,
    *,
    height: float | None,
    height_reference: str,
    pivot: tuple[float, float] | None,
    sigma: np.ndarray | list[np.ndarray] | None,
    x_ref: tuple[float, float],
    chord_ref: float | None,
    is_scalar_alpha: bool,
) -> Panel2DResult:
    """Implement NumPy float64 reference solver."""
    bodies_np = [np.asarray(b, dtype=np.float64) for b in bodies]
    n_bodies = len(bodies_np)

    # Reference chord from body 0
    b0_xle = float(np.min(bodies_np[0][:, 0]))
    b0_xte = float(np.max(bodies_np[0][:, 0]))
    b0_chord = b0_xte - b0_xle
    if chord_ref is None:
        chord_ref = b0_chord if b0_chord > 0.0 else 1.0

    if pivot is None:
        pivot = (b0_xle + 0.25 * chord_ref, 0.0)
    px, py = pivot

    # Build geometry and index structure
    node_counts = [len(b) for b in bodies_np]
    total_nodes = sum(node_counts)

    body_node_ranges = []
    curr = 0
    for nc in node_counts:
        body_node_ranges.append((curr, curr + nc))
        curr += nc

    colloc_list, p_start_list, p_end_list, idx1_list, idx2_list = [], [], [], [], []
    panel_body_ids = []
    curr_node = 0
    for b_id, b in enumerate(bodies_np):
        c, ps, pe, i1, i2 = _build_body_panels_numpy(b, curr_node)
        colloc_list.append(c)
        p_start_list.append(ps)
        p_end_list.append(pe)
        idx1_list.append(i1)
        idx2_list.append(i2)
        panel_body_ids.extend([b_id] * len(ps))
        curr_node += len(b)

    colloc_all = np.vstack(colloc_list)
    p_start_all = np.vstack(p_start_list)
    p_end_all = np.vstack(p_end_list)
    idx1_all = np.concatenate(idx1_list)
    idx2_all = np.concatenate(idx2_list)
    total_panels = len(p_start_all)
    n_unknowns = total_nodes + n_bodies

    # Check prescribed source strengths
    sigma_vec = np.zeros(total_panels, dtype=np.float64)
    has_sigma = False
    if sigma is not None:
        if isinstance(sigma, (list, tuple)):
            sigma_vec = np.concatenate([np.asarray(s, dtype=np.float64) for s in sigma])
        else:
            sigma_vec = np.asarray(sigma, dtype=np.float64)
        has_sigma = not np.all(sigma_vec == 0.0)

    # 1. Free-air solve
    if height is None:
        # Build matrix A
        xi, eta, p_len, p_th = compute_panel_local_coordinates_numpy(colloc_all, p_start_all, p_end_all)
        psi1, psi2 = linear_vortex_stream_influence(xi, eta, p_len)

        A = np.zeros((n_unknowns, n_unknowns), dtype=np.float64)
        # Vectorized assembly into A
        np.add.at(A[:total_nodes, :total_nodes], (slice(None), idx1_all), psi1)
        np.add.at(A[:total_nodes, :total_nodes], (slice(None), idx2_all), psi2)

        # Stream function constants per body
        for b_id, (n_start, n_end) in enumerate(body_node_ranges):
            A[n_start:n_end, total_nodes + b_id] = -1.0

        # Kutta conditions per body
        for b_id, (n_start, n_end) in enumerate(body_node_ranges):
            kutta_row = total_nodes + b_id
            A[kutta_row, n_start] = 1.0
            A[kutta_row, n_end - 1] = 1.0

        # Factorize once
        lu, piv = scipy.linalg.lu_factor(A)

        # Right hand sides for x and y free stream
        rhs_x = np.zeros(n_unknowns, dtype=np.float64)
        rhs_y = np.zeros(n_unknowns, dtype=np.float64)
        rhs_x[:total_nodes] = -colloc_all[:, 1]
        rhs_y[:total_nodes] = colloc_all[:, 0]

        sol_x = scipy.linalg.lu_solve((lu, piv), rhs_x)
        sol_y = scipy.linalg.lu_solve((lu, piv), rhs_y)

        # Source term contribution
        sol_sigma = np.zeros(n_unknowns, dtype=np.float64)
        if has_sigma:
            psi_s_all = source_panel_stream_influence(xi, eta, p_len, p_th)
            rhs_sigma = np.zeros(n_unknowns, dtype=np.float64)
            rhs_sigma[:total_nodes] = -psi_s_all @ sigma_vec
            sol_sigma = scipy.linalg.lu_solve((lu, piv), rhs_sigma)

        # Superposition across angles
        n_alphas = len(alphas)
        alpha_rad = np.radians(alphas)
        cos_a = np.cos(alpha_rad)
        sin_a = np.sin(alpha_rad)

        sols = np.outer(cos_a, sol_x) + np.outer(sin_a, sol_y)
        if has_sigma:
            sols += sol_sigma[np.newaxis, :]

        # Unrotated geometry for force and moment calculations
        dx_p = p_end_all[:, 0] - p_start_all[:, 0]
        dy_p = p_end_all[:, 1] - p_start_all[:, 1]
        p_lengths = np.hypot(dx_p, dy_p)
        x_mids = 0.5 * (p_start_all[:, 0] + p_end_all[:, 0])
        y_mids = 0.5 * (p_start_all[:, 1] + p_end_all[:, 1])

        cl_list, cl_press_list, cd_press_list, cm_list = [], [], [], []
        cp_list, ue_list, gamma_list, circ_list = [], [], [], []

        for i in range(n_alphas):
            a_deg = alphas[i]
            sol_i = sols[i]
            gamma_i = sol_i[:total_nodes]

            # Cp and ue
            ue_i = gamma_i.copy()
            cp_i = 1.0 - ue_i**2

            # Circulation per body
            gamma_p1 = gamma_i[idx1_all]
            gamma_p2 = gamma_i[idx2_all]
            gamma_p_avg = 0.5 * (gamma_p1 + gamma_p2)
            d_circ = gamma_p_avg * p_lengths

            circ_bodies = np.zeros(n_bodies, dtype=np.float64)
            for p_idx, b_id in enumerate(panel_body_ids):
                circ_bodies[b_id] += d_circ[p_idx]
            total_circ = float(np.sum(circ_bodies))

            cl_circ = float(2.0 * total_circ / chord_ref)

            # Pressure integration
            cp_p1 = cp_i[idx1_all]
            cp_p2 = cp_i[idx2_all]
            cp_p_avg = 0.5 * (cp_p1 + cp_p2)

            d_fx = -cp_p_avg * dy_p
            d_fy = cp_p_avg * dx_p
            ca = float(np.sum(d_fx) / chord_ref)
            cn = float(np.sum(d_fy) / chord_ref)

            a_r = alpha_rad[i]
            cl_press = float(cn * np.cos(a_r) - ca * np.sin(a_r))
            cd_press = float(cn * np.sin(a_r) + ca * np.cos(a_r))

            # Pitching moment about x_ref
            rx = x_mids - x_ref[0]
            ry = y_mids - x_ref[1]
            dm = -rx * d_fy + ry * d_fx
            cm = float(np.sum(dm) / (chord_ref**2))

            cl_list.append(cl_circ)
            cl_press_list.append(cl_press)
            cd_press_list.append(cd_press)
            cm_list.append(cm)
            cp_list.append(cp_i)
            ue_list.append(ue_i)
            gamma_list.append(gamma_i)
            circ_list.append(circ_bodies)

        out_coords = bodies_np if n_bodies > 1 else bodies_np[0]

        if is_scalar_alpha:
            res_cp = cp_list[0] if n_bodies == 1 else [cp_list[0][s:e] for s, e in body_node_ranges]
            res_ue = ue_list[0] if n_bodies == 1 else [ue_list[0][s:e] for s, e in body_node_ranges]
            res_gamma = gamma_list[0] if n_bodies == 1 else [gamma_list[0][s:e] for s, e in body_node_ranges]
            res_circ = circ_list[0] if n_bodies > 1 else float(circ_list[0][0])
            return Panel2DResult(
                alpha=float(alphas[0]),
                Cl=float(cl_list[0]),
                Cl_pressure=float(cl_press_list[0]),
                Cd_pressure=float(cd_press_list[0]),
                Cm=float(cm_list[0]),
                Cp=res_cp,
                ue=res_ue,
                gamma=res_gamma,
                coordinates=out_coords,
                circulation=res_circ,
            )

        res_cp = np.array(cp_list) if n_bodies == 1 else [np.array([c[s:e] for c in cp_list]) for s, e in body_node_ranges]
        res_ue = np.array(ue_list) if n_bodies == 1 else [np.array([u[s:e] for u in ue_list]) for s, e in body_node_ranges]
        res_gamma = np.array(gamma_list) if n_bodies == 1 else [np.array([g[s:e] for g in gamma_list]) for s, e in body_node_ranges]
        res_circ = np.array(circ_list) if n_bodies > 1 else np.array([c[0] for c in circ_list])

        return Panel2DResult(
            alpha=alphas,
            Cl=np.array(cl_list),
            Cl_pressure=np.array(cl_press_list),
            Cd_pressure=np.array(cd_press_list),
            Cm=np.array(cm_list),
            Cp=res_cp,
            ue=res_ue,
            gamma=res_gamma,
            coordinates=out_coords,
            circulation=res_circ,
        )

    # 2. Ground Effect solve
    cl_list, cl_press_list, cd_press_list, cm_list = [], [], [], []
    cp_list, ue_list, gamma_list, circ_list, coords_solved = [], [], [], [], []

    for a_deg in alphas:
        a_rad = np.radians(a_deg)
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)

        # Rotate clockwise by alpha about pivot
        rot_bodies = []
        for b in bodies_np:
            dx = b[:, 0] - px
            dy = b[:, 1] - py
            xr = px + dx * cos_a + dy * sin_a
            yr = py - dx * sin_a + dy * cos_a
            rot_bodies.append(np.column_stack([xr, yr]))

        # Height adjustment
        if height_reference == "pivot":
            delta_y = height - py
        elif height_reference == "trailing_edge":
            te_y_vals = [b[0, 1] for b in rot_bodies] + [b[-1, 1] for b in rot_bodies]
            min_te_y = float(min(te_y_vals))
            delta_y = height - min_te_y
        else:
            raise ValueError(f"Unknown height_reference '{height_reference}'.")

        ge_bodies = [np.column_stack([b[:, 0], b[:, 1] + delta_y]) for b in rot_bodies]

        # Geometry check: touching or below ground
        for b in ge_bodies:
            if np.min(b[:, 1]) <= 0.0:
                raise ValueError("Geometry under or touching the ground")

        # Build GE panels
        colloc_list, p_start_list, p_end_list, idx1_list, idx2_list = [], [], [], [], []
        curr_node = 0
        for b in ge_bodies:
            c, ps, pe, i1, i2 = _build_body_panels_numpy(b, curr_node)
            colloc_list.append(c)
            p_start_list.append(ps)
            p_end_list.append(pe)
            idx1_list.append(i1)
            idx2_list.append(i2)
            curr_node += len(b)

        colloc_ge = np.vstack(colloc_list)
        p_start_ge = np.vstack(p_start_list)
        p_end_ge = np.vstack(p_end_list)
        idx1_ge = np.concatenate(idx1_list)
        idx2_ge = np.concatenate(idx2_list)

        # Mirror image panels (y -> -y)
        p_start_img = p_start_ge.copy()
        p_start_img[:, 1] *= -1.0
        p_end_img = p_end_ge.copy()
        p_end_img[:, 1] *= -1.0

        xi_r, eta_r, len_r, th_r = compute_panel_local_coordinates_numpy(colloc_ge, p_start_ge, p_end_ge)
        psi1_r, psi2_r = linear_vortex_stream_influence(xi_r, eta_r, len_r)

        xi_i, eta_i, len_i, th_i = compute_panel_local_coordinates_numpy(colloc_ge, p_start_img, p_end_img)
        psi1_i, psi2_i = linear_vortex_stream_influence(xi_i, eta_i, len_i)

        A_ge = np.zeros((n_unknowns, n_unknowns), dtype=np.float64)
        np.add.at(A_ge[:total_nodes, :total_nodes], (slice(None), idx1_ge), psi1_r - psi1_i)
        np.add.at(A_ge[:total_nodes, :total_nodes], (slice(None), idx2_ge), psi2_r - psi2_i)

        for b_id, (n_start, n_end) in enumerate(body_node_ranges):
            A_ge[n_start:n_end, total_nodes + b_id] = -1.0
            kutta_row = total_nodes + b_id
            A_ge[kutta_row, n_start] = 1.0
            A_ge[kutta_row, n_end - 1] = 1.0

        # RHS in GE: free stream along +x: psi = y
        rhs_ge = np.zeros(n_unknowns, dtype=np.float64)
        rhs_ge[:total_nodes] = -colloc_ge[:, 1]

        if has_sigma:
            psi_s_r = source_panel_stream_influence(xi_r, eta_r, len_r, th_r)
            psi_s_i = source_panel_stream_influence(xi_i, eta_i, len_i, th_i)
            rhs_ge[:total_nodes] -= (psi_s_r + psi_s_i) @ sigma_vec

        sol_ge = np.linalg.solve(A_ge, rhs_ge)
        gamma_ge = sol_ge[:total_nodes]

        ue_ge = gamma_ge.copy()
        cp_ge = 1.0 - ue_ge**2

        # Circulation
        dx_p = p_end_ge[:, 0] - p_start_ge[:, 0]
        dy_p = p_end_ge[:, 1] - p_start_ge[:, 1]
        p_len_ge = np.hypot(dx_p, dy_p)
        gamma_p_avg = 0.5 * (gamma_ge[idx1_ge] + gamma_ge[idx2_ge])
        d_circ = gamma_p_avg * p_len_ge

        circ_bodies = np.zeros(n_bodies, dtype=np.float64)
        for p_idx, b_id in enumerate(panel_body_ids):
            circ_bodies[b_id] += d_circ[p_idx]
        total_circ = float(np.sum(circ_bodies))
        cl_circ = float(2.0 * total_circ / chord_ref)

        # In GE, free stream is along +x: lift is total Fy, drag is total Fx
        cp_p_avg = 0.5 * (cp_ge[idx1_ge] + cp_ge[idx2_ge])
        d_fx = -cp_p_avg * dy_p
        d_fy = cp_p_avg * dx_p
        cd_press = float(np.sum(d_fx) / chord_ref)
        cl_press = float(np.sum(d_fy) / chord_ref)

        # Moment in body-0 frame before rotation
        dx_unrot = p_end_all[:, 0] - p_start_all[:, 0]
        dy_unrot = p_end_all[:, 1] - p_start_all[:, 1]
        xm_unrot = 0.5 * (p_start_all[:, 0] + p_end_all[:, 0])
        ym_unrot = 0.5 * (p_start_all[:, 1] + p_end_all[:, 1])
        dfx_unrot = -cp_p_avg * dy_unrot
        dfy_unrot = cp_p_avg * dx_unrot
        dm = -(xm_unrot - x_ref[0]) * dfy_unrot + (ym_unrot - x_ref[1]) * dfx_unrot
        cm = float(np.sum(dm) / (chord_ref**2))

        cl_list.append(cl_circ)
        cl_press_list.append(cl_press)
        cd_press_list.append(cd_press)
        cm_list.append(cm)
        cp_list.append(cp_ge)
        ue_list.append(ue_ge)
        gamma_list.append(gamma_ge)
        circ_list.append(circ_bodies)
        coords_solved.append(ge_bodies if n_bodies > 1 else ge_bodies[0])

    if is_scalar_alpha:
        res_cp = cp_list[0] if n_bodies == 1 else [cp_list[0][s:e] for s, e in body_node_ranges]
        res_ue = ue_list[0] if n_bodies == 1 else [ue_list[0][s:e] for s, e in body_node_ranges]
        res_gamma = gamma_list[0] if n_bodies == 1 else [gamma_list[0][s:e] for s, e in body_node_ranges]
        res_circ = circ_list[0] if n_bodies > 1 else float(circ_list[0][0])
        return Panel2DResult(
            alpha=float(alphas[0]),
            Cl=float(cl_list[0]),
            Cl_pressure=float(cl_press_list[0]),
            Cd_pressure=float(cd_press_list[0]),
            Cm=float(cm_list[0]),
            Cp=res_cp,
            ue=res_ue,
            gamma=res_gamma,
            coordinates=coords_solved[0],
            circulation=res_circ,
        )

    res_cp = np.array(cp_list) if n_bodies == 1 else [np.array([c[s:e] for c in cp_list]) for s, e in body_node_ranges]
    res_ue = np.array(ue_list) if n_bodies == 1 else [np.array([u[s:e] for u in ue_list]) for s, e in body_node_ranges]
    res_gamma = np.array(gamma_list) if n_bodies == 1 else [np.array([g[s:e] for g in gamma_list]) for s, e in body_node_ranges]
    res_circ = np.array(circ_list) if n_bodies > 1 else np.array([c[0] for c in circ_list])

    return Panel2DResult(
        alpha=alphas,
        Cl=np.array(cl_list),
        Cl_pressure=np.array(cl_press_list),
        Cd_pressure=np.array(cd_press_list),
        Cm=np.array(cm_list),
        Cp=res_cp,
        ue=res_ue,
        gamma=res_gamma,
        coordinates=coords_solved,
        circulation=res_circ,
    )


def _solve_torch(
    bodies: list[torch.Tensor | np.ndarray],
    alphas: torch.Tensor,
    *,
    height: float | None,
    height_reference: str,
    pivot: tuple[float, float] | None,
    sigma: np.ndarray | list[np.ndarray] | torch.Tensor | None,
    x_ref: tuple[float, float],
    chord_ref: float | None,
    device: str | torch.device,
    dtype: torch.dtype,
    is_scalar_alpha: bool,
) -> Panel2DResult:
    """Implement PyTorch batched CPU and CUDA solver."""
    bodies_th = [torch.as_tensor(b, dtype=dtype, device=device) for b in bodies]
    n_bodies = len(bodies_th)

    b0_xle = float(torch.min(bodies_th[0][:, 0]).item())
    b0_xte = float(torch.max(bodies_th[0][:, 0]).item())
    b0_chord = b0_xte - b0_xle
    if chord_ref is None:
        chord_ref = b0_chord if b0_chord > 0.0 else 1.0

    if pivot is None:
        pivot = (b0_xle + 0.25 * chord_ref, 0.0)
    px, py = pivot

    node_counts = [b.shape[-2] for b in bodies_th]
    total_nodes = sum(node_counts)

    body_node_ranges = []
    curr = 0
    for nc in node_counts:
        body_node_ranges.append((curr, curr + nc))
        curr += nc

    colloc_list, p_start_list, p_end_list, idx1_list, idx2_list = [], [], [], [], []
    panel_body_ids = []
    curr_node = 0
    for b_id, b in enumerate(bodies_th):
        c, ps, pe, i1, i2 = _build_body_panels_torch(b, curr_node)
        colloc_list.append(c)
        p_start_list.append(ps)
        p_end_list.append(pe)
        idx1_list.append(i1)
        idx2_list.append(i2)
        panel_body_ids.extend([b_id] * ps.shape[-2])
        curr_node += b.shape[-2]

    colloc_all = torch.cat(colloc_list, dim=-2)
    p_start_all = torch.cat(p_start_list, dim=-2)
    p_end_all = torch.cat(p_end_list, dim=-2)
    idx1_all = torch.cat(idx1_list, dim=-1)
    idx2_all = torch.cat(idx2_list, dim=-1)
    total_panels = p_start_all.shape[-2]
    n_unknowns = total_nodes + n_bodies

    # Sources
    sigma_vec = torch.zeros(total_panels, dtype=dtype, device=device)
    has_sigma = False
    if sigma is not None:
        if isinstance(sigma, (list, tuple)):
            sigma_vec = torch.cat([torch.as_tensor(s, dtype=dtype, device=device) for s in sigma])
        else:
            sigma_vec = torch.as_tensor(sigma, dtype=dtype, device=device)
        has_sigma = bool(torch.any(sigma_vec != 0.0).item())

    # Free-air solve
    if height is None:
        xi, eta, p_len, p_th = compute_panel_local_coordinates_torch(colloc_all, p_start_all, p_end_all)
        psi1, psi2 = linear_vortex_stream_influence(xi, eta, p_len)

        A = torch.zeros((n_unknowns, n_unknowns), dtype=dtype, device=device)
        A[:total_nodes, :total_nodes].index_add_(-1, idx1_all, psi1)
        A[:total_nodes, :total_nodes].index_add_(-1, idx2_all, psi2)

        for b_id, (n_start, n_end) in enumerate(body_node_ranges):
            A[n_start:n_end, total_nodes + b_id] = -1.0
            kutta_row = total_nodes + b_id
            A[kutta_row, n_start] = 1.0
            A[kutta_row, n_end - 1] = 1.0

        rhs_x = torch.zeros(n_unknowns, dtype=dtype, device=device)
        rhs_y = torch.zeros(n_unknowns, dtype=dtype, device=device)
        rhs_x[:total_nodes] = -colloc_all[:, 1]
        rhs_y[:total_nodes] = colloc_all[:, 0]

        rhs_stack = torch.stack([rhs_x, rhs_y], dim=-1)
        sols_xy = torch.linalg.solve(A, rhs_stack)
        sol_x = sols_xy[:, 0]
        sol_y = sols_xy[:, 1]

        sol_sigma = torch.zeros(n_unknowns, dtype=dtype, device=device)
        if has_sigma:
            psi_s_all = source_panel_stream_influence(xi, eta, p_len, p_th)
            rhs_sigma = torch.zeros(n_unknowns, dtype=dtype, device=device)
            rhs_sigma[:total_nodes] = -torch.matmul(psi_s_all, sigma_vec)
            sol_sigma = torch.linalg.solve(A, rhs_sigma)

        alpha_rad = torch.deg2rad(alphas)
        cos_a = torch.cos(alpha_rad)
        sin_a = torch.sin(alpha_rad)

        sols = torch.outer(cos_a, sol_x) + torch.outer(sin_a, sol_y)
        if has_sigma:
            sols += sol_sigma.unsqueeze(0)

        dx_p = p_end_all[:, 0] - p_start_all[:, 0]
        dy_p = p_end_all[:, 1] - p_start_all[:, 1]
        p_lengths = torch.hypot(dx_p, dy_p)
        x_mids = 0.5 * (p_start_all[:, 0] + p_end_all[:, 0])
        y_mids = 0.5 * (p_start_all[:, 1] + p_end_all[:, 1])

        gamma_all = sols[:, :total_nodes]
        ue_all = gamma_all.clone()
        cp_all = 1.0 - ue_all**2

        gamma_p1 = gamma_all[:, idx1_all]
        gamma_p2 = gamma_all[:, idx2_all]
        gamma_p_avg = 0.5 * (gamma_p1 + gamma_p2)
        d_circ = gamma_p_avg * p_lengths.unsqueeze(0)

        circ_bodies = torch.zeros((len(alphas), n_bodies), dtype=dtype, device=device)
        for p_idx, b_id in enumerate(panel_body_ids):
            circ_bodies[:, b_id] += d_circ[:, p_idx]
        total_circ = torch.sum(circ_bodies, dim=-1)
        cl_circ = 2.0 * total_circ / chord_ref

        cp_p1 = cp_all[:, idx1_all]
        cp_p2 = cp_all[:, idx2_all]
        cp_p_avg = 0.5 * (cp_p1 + cp_p2)

        d_fx = -cp_p_avg * dy_p.unsqueeze(0)
        d_fy = cp_p_avg * dx_p.unsqueeze(0)
        ca = torch.sum(d_fx, dim=-1) / chord_ref
        cn = torch.sum(d_fy, dim=-1) / chord_ref

        cl_press = cn * cos_a - ca * sin_a
        cd_press = cn * sin_a + ca * cos_a

        rx = x_mids - x_ref[0]
        ry = y_mids - x_ref[1]
        dm = -rx.unsqueeze(0) * d_fy + ry.unsqueeze(0) * d_fx
        cm = torch.sum(dm, dim=-1) / (chord_ref**2)

        out_coords = bodies_th if n_bodies > 1 else bodies_th[0]

        if is_scalar_alpha:
            res_cp = cp_all[0] if n_bodies == 1 else [cp_all[0, s:e] for s, e in body_node_ranges]
            res_ue = ue_all[0] if n_bodies == 1 else [ue_all[0, s:e] for s, e in body_node_ranges]
            res_gamma = gamma_all[0] if n_bodies == 1 else [gamma_all[0, s:e] for s, e in body_node_ranges]
            res_circ = circ_bodies[0] if n_bodies > 1 else circ_bodies[0, 0]
            return Panel2DResult(
                alpha=alphas[0],
                Cl=cl_circ[0],
                Cl_pressure=cl_press[0],
                Cd_pressure=cd_press[0],
                Cm=cm[0],
                Cp=res_cp,
                ue=res_ue,
                gamma=res_gamma,
                coordinates=out_coords,
                circulation=res_circ,
            )

        res_cp = cp_all if n_bodies == 1 else [cp_all[:, s:e] for s, e in body_node_ranges]
        res_ue = ue_all if n_bodies == 1 else [ue_all[:, s:e] for s, e in body_node_ranges]
        res_gamma = gamma_all if n_bodies == 1 else [gamma_all[:, s:e] for s, e in body_node_ranges]
        res_circ = circ_bodies if n_bodies > 1 else circ_bodies[:, 0]

        return Panel2DResult(
            alpha=alphas,
            Cl=cl_circ,
            Cl_pressure=cl_press,
            Cd_pressure=cd_press,
            Cm=cm,
            Cp=res_cp,
            ue=res_ue,
            gamma=res_gamma,
            coordinates=out_coords,
            circulation=res_circ,
        )

    # Ground Effect solve in PyTorch (batched)
    n_a = len(alphas)
    a_rad = torch.deg2rad(alphas)
    cos_a = torch.cos(a_rad)
    sin_a = torch.sin(a_rad)

    # Batch rotate bodies: (n_a, N, 2)
    rot_bodies_all = []
    for b in bodies_th:
        dx = b[:, 0] - px
        dy = b[:, 1] - py
        xr = px + torch.outer(cos_a, dx) + torch.outer(sin_a, dy)
        yr = py - torch.outer(sin_a, dx) + torch.outer(cos_a, dy)
        rot_b = torch.stack([xr, yr], dim=-1)
        rot_bodies_all.append(rot_b)

    if height_reference == "pivot":
        delta_y = height - py
    elif height_reference == "trailing_edge":
        te_ys = torch.cat([b[:, 0:1, 1] for b in rot_bodies_all] + [b[:, -1:, 1] for b in rot_bodies_all], dim=-1)
        min_te_y = torch.min(te_ys, dim=-1).values
        delta_y = height - min_te_y
    else:
        raise ValueError(f"Unknown height_reference '{height_reference}'.")

    ge_bodies_all = []
    for b in rot_bodies_all:
        if isinstance(delta_y, torch.Tensor):
            y_shifted = b[:, :, 1] + delta_y.unsqueeze(-1)
        else:
            y_shifted = b[:, :, 1] + delta_y
        ge_b = torch.stack([b[:, :, 0], y_shifted], dim=-1)
        if torch.min(ge_b[:, :, 1]).item() <= 0.0:
            raise ValueError("Geometry under or touching the ground")
        ge_bodies_all.append(ge_b)

    # Build batched GE panels: (n_a, ...)
    colloc_list, p_start_list, p_end_list = [], [], []
    for b in ge_bodies_all:
        c, ps, pe, _, _ = _build_body_panels_torch(b, 0)
        colloc_list.append(c)
        p_start_list.append(ps)
        p_end_list.append(pe)

    colloc_ge = torch.cat(colloc_list, dim=1)
    p_start_ge = torch.cat(p_start_list, dim=1)
    p_end_ge = torch.cat(p_end_list, dim=1)

    p_start_img = p_start_ge.clone()
    p_start_img[..., 1] *= -1.0
    p_end_img = p_end_ge.clone()
    p_end_img[..., 1] *= -1.0

    xi_r, eta_r, len_r, th_r = compute_panel_local_coordinates_torch(colloc_ge, p_start_ge, p_end_ge)
    psi1_r, psi2_r = linear_vortex_stream_influence(xi_r, eta_r, len_r)

    xi_i, eta_i, len_i, th_i = compute_panel_local_coordinates_torch(colloc_ge, p_start_img, p_end_img)
    psi1_i, psi2_i = linear_vortex_stream_influence(xi_i, eta_i, len_i)

    # Assemble batched A: (n_a, n_unknowns, n_unknowns)
    A_ge = torch.zeros((n_a, n_unknowns, n_unknowns), dtype=dtype, device=device)
    A_ge[:, :total_nodes, :total_nodes].index_add_(-1, idx1_all, psi1_r - psi1_i)
    A_ge[:, :total_nodes, :total_nodes].index_add_(-1, idx2_all, psi2_r - psi2_i)

    for b_id, (n_start, n_end) in enumerate(body_node_ranges):
        A_ge[:, n_start:n_end, total_nodes + b_id] = -1.0
        kutta_row = total_nodes + b_id
        A_ge[:, kutta_row, n_start] = 1.0
        A_ge[:, kutta_row, n_end - 1] = 1.0

    rhs_ge = torch.zeros((n_a, n_unknowns), dtype=dtype, device=device)
    rhs_ge[:, :total_nodes] = -colloc_ge[..., 1]

    if has_sigma:
        psi_s_r = source_panel_stream_influence(xi_r, eta_r, len_r, th_r)
        psi_s_i = source_panel_stream_influence(xi_i, eta_i, len_i, th_i)
        src_rhs = torch.matmul(psi_s_r + psi_s_i, sigma_vec)
        rhs_ge[:, :total_nodes] -= src_rhs

    sol_ge = torch.linalg.solve(A_ge, rhs_ge.unsqueeze(-1)).squeeze(-1)
    gamma_ge = sol_ge[:, :total_nodes]
    ue_ge = gamma_ge.clone()
    cp_ge = 1.0 - ue_ge**2

    dx_p = p_end_ge[..., 0] - p_start_ge[..., 0]
    dy_p = p_end_ge[..., 1] - p_start_ge[..., 1]
    p_len_ge = torch.hypot(dx_p, dy_p)

    gamma_p1 = gamma_ge[:, idx1_all]
    gamma_p2 = gamma_ge[:, idx2_all]
    gamma_p_avg = 0.5 * (gamma_p1 + gamma_p2)
    d_circ = gamma_p_avg * p_len_ge

    circ_bodies = torch.zeros((n_a, n_bodies), dtype=dtype, device=device)
    for p_idx, b_id in enumerate(panel_body_ids):
        circ_bodies[:, b_id] += d_circ[:, p_idx]
    total_circ = torch.sum(circ_bodies, dim=-1)
    cl_circ = 2.0 * total_circ / chord_ref

    cp_p1 = cp_ge[:, idx1_all]
    cp_p2 = cp_ge[:, idx2_all]
    cp_p_avg = 0.5 * (cp_p1 + cp_p2)

    d_fx = -cp_p_avg * dy_p
    d_fy = cp_p_avg * dx_p
    cd_press = torch.sum(d_fx, dim=-1) / chord_ref
    cl_press = torch.sum(d_fy, dim=-1) / chord_ref

    dx_unrot = p_end_all[:, 0] - p_start_all[:, 0]
    dy_unrot = p_end_all[:, 1] - p_start_all[:, 1]
    xm_unrot = 0.5 * (p_start_all[:, 0] + p_end_all[:, 0])
    ym_unrot = 0.5 * (p_start_all[:, 1] + p_end_all[:, 1])
    dfx_unrot = -cp_p_avg * dy_unrot.unsqueeze(0)
    dfy_unrot = cp_p_avg * dx_unrot.unsqueeze(0)
    dm = -(xm_unrot.unsqueeze(0) - x_ref[0]) * dfy_unrot + (ym_unrot.unsqueeze(0) - x_ref[1]) * dfx_unrot
    cm = torch.sum(dm, dim=-1) / (chord_ref**2)

    coords_out = ge_bodies_all if n_bodies > 1 else ge_bodies_all[0]

    if is_scalar_alpha:
        res_cp = cp_ge[0] if n_bodies == 1 else [cp_ge[0, s:e] for s, e in body_node_ranges]
        res_ue = ue_ge[0] if n_bodies == 1 else [ue_ge[0, s:e] for s, e in body_node_ranges]
        res_gamma = gamma_ge[0] if n_bodies == 1 else [gamma_ge[0, s:e] for s, e in body_node_ranges]
        res_circ = circ_bodies[0] if n_bodies > 1 else circ_bodies[0, 0]
        res_coords = coords_out[0] if not isinstance(coords_out, list) else [b[0] for b in coords_out]
        return Panel2DResult(
            alpha=alphas[0],
            Cl=cl_circ[0],
            Cl_pressure=cl_press[0],
            Cd_pressure=cd_press[0],
            Cm=cm[0],
            Cp=res_cp,
            ue=res_ue,
            gamma=res_gamma,
            coordinates=res_coords,
            circulation=res_circ,
        )

    res_cp = cp_ge if n_bodies == 1 else [cp_ge[:, s:e] for s, e in body_node_ranges]
    res_ue = ue_ge if n_bodies == 1 else [ue_ge[:, s:e] for s, e in body_node_ranges]
    res_gamma = gamma_ge if n_bodies == 1 else [gamma_ge[:, s:e] for s, e in body_node_ranges]
    res_circ = circ_bodies if n_bodies > 1 else circ_bodies[:, 0]

    return Panel2DResult(
        alpha=alphas,
        Cl=cl_circ,
        Cl_pressure=cl_press,
        Cd_pressure=cd_press,
        Cm=cm,
        Cp=res_cp,
        ue=res_ue,
        gamma=res_gamma,
        coordinates=coords_out,
        circulation=res_circ,
    )
