# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Closed-form stream function panel influence coefficients.

This module implements closed-form stream function influences for linear-strength
vortex panels and constant-strength source panels. It provides vectorized
implementations for NumPy arrays and PyTorch tensors.

References
----------
J. Katz and A. Plotkin, 'Low-Speed Aerodynamics', 2nd ed., Cambridge University
Press, 2001, chapter 11.
M. Drela, 'XFOIL: An Analysis and Design System for Low Reynolds Number Airfoils',
Lecture Notes in Engineering, Vol. 54, Springer, 1989, pp. 1-12.
"""

from __future__ import annotations

import numpy as np
import torch


def _f0_numpy(t: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """Evaluate indefinite integral int ln(t^2 + eta^2) dt in NumPy."""
    dist2 = t * t + eta * eta
    safe_dist2 = np.maximum(dist2, 1e-30)
    term1 = t * np.log(safe_dist2) - 2.0 * t
    safe_eta = np.where(np.abs(eta) < 1e-15, 1.0, eta)
    term2 = np.where(np.abs(eta) < 1e-15, 0.0, 2.0 * eta * np.arctan(t / safe_eta))
    return term1 + term2


def _f1_numpy(t: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """Evaluate indefinite integral int t ln(t^2 + eta^2) dt in NumPy."""
    dist2 = t * t + eta * eta
    safe_dist2 = np.maximum(dist2, 1e-30)
    return 0.5 * dist2 * (np.log(safe_dist2) - 1.0)


def _f0_torch(t: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    """Evaluate indefinite integral int ln(t^2 + eta^2) dt in PyTorch."""
    dist2 = t * t + eta * eta
    safe_dist2 = torch.clamp(dist2, min=1e-30)
    term1 = t * torch.log(safe_dist2) - 2.0 * t
    safe_eta = torch.where(torch.abs(eta) < 1e-15, torch.ones_like(eta), eta)
    term2 = torch.where(
        torch.abs(eta) < 1e-15,
        torch.zeros_like(eta),
        2.0 * eta * torch.atan(t / safe_eta),
    )
    return term1 + term2


def _f1_torch(t: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    """Evaluate indefinite integral int t ln(t^2 + eta^2) dt in PyTorch."""
    dist2 = t * t + eta * eta
    safe_dist2 = torch.clamp(dist2, min=1e-30)
    return 0.5 * dist2 * (torch.log(safe_dist2) - 1.0)


def linear_vortex_stream_influence(
    xi: float | np.ndarray | torch.Tensor,
    eta: float | np.ndarray | torch.Tensor,
    length: float | np.ndarray | torch.Tensor,
) -> tuple[float | np.ndarray | torch.Tensor, float | np.ndarray | torch.Tensor]:
    """Calculate stream function influences of a linear-strength vortex panel.

    The vortex strength varies linearly on the panel from gamma_1 at local
    xi=0 to gamma_2 at local xi=length:
    gamma(s) = gamma_1 * (1 - s/length) + gamma_2 * (s/length).

    Parameters
    ----------
    xi : float, np.ndarray, or torch.Tensor
        Chordwise distance in panel local coordinate system [m].
    eta : float, np.ndarray, or torch.Tensor
        Normal distance in panel local coordinate system [m].
    length : float, np.ndarray, or torch.Tensor
        Panel length [m].

    Returns
    -------
    tuple of (psi_1, psi_2)
        Stream function influences of unit nodal vortex strengths gamma_1
        and gamma_2 [m].
    """
    if isinstance(xi, torch.Tensor) or isinstance(eta, torch.Tensor):
        if not isinstance(xi, torch.Tensor):
            xi = torch.as_tensor(xi, dtype=torch.float64)
        if not isinstance(eta, torch.Tensor):
            eta = torch.as_tensor(eta, dtype=xi.dtype, device=xi.device)
        if not isinstance(length, torch.Tensor):
            length = torch.as_tensor(length, dtype=xi.dtype, device=xi.device)

        i0 = _f0_torch(xi, eta) - _f0_torch(xi - length, eta)
        df1 = _f1_torch(xi, eta) - _f1_torch(xi - length, eta)

        psi1 = (1.0 / (4.0 * np.pi)) * ((1.0 - xi / length) * i0 + (1.0 / length) * df1)
        psi2 = (1.0 / (4.0 * np.pi)) * ((xi / length) * i0 - (1.0 / length) * df1)
        return psi1, psi2

    # NumPy implementation
    xi_np = np.asarray(xi, dtype=np.float64)
    eta_np = np.asarray(eta, dtype=np.float64)
    len_np = np.asarray(length, dtype=np.float64)

    i0 = _f0_numpy(xi_np, eta_np) - _f0_numpy(xi_np - len_np, eta_np)
    df1 = _f1_numpy(xi_np, eta_np) - _f1_numpy(xi_np - len_np, eta_np)

    psi1 = (1.0 / (4.0 * np.pi)) * ((1.0 - xi_np / len_np) * i0 + (1.0 / len_np) * df1)
    psi2 = (1.0 / (4.0 * np.pi)) * ((xi_np / len_np) * i0 - (1.0 / len_np) * df1)

    if np.ndim(xi) == 0 and np.ndim(eta) == 0 and np.ndim(length) == 0:
        return float(psi1), float(psi2)
    return psi1, psi2


def source_panel_stream_influence(
    xi: float | np.ndarray | torch.Tensor,
    eta: float | np.ndarray | torch.Tensor,
    length: float | np.ndarray | torch.Tensor,
    theta_panel: float | np.ndarray | torch.Tensor = 0.0,
) -> float | np.ndarray | torch.Tensor:
    """Calculate stream function influence of a constant-strength source panel.

    The source strength sigma is constant over the panel.

    Parameters
    ----------
    xi : float, np.ndarray, or torch.Tensor
        Chordwise distance in panel local coordinate system [m].
    eta : float, np.ndarray, or torch.Tensor
        Normal distance in panel local coordinate system [m].
    length : float, np.ndarray, or torch.Tensor
        Panel length [m].
    theta_panel : float, np.ndarray, or torch.Tensor, optional
        Panel orientation angle with respect to global x-axis [rad]. Default is 0.0.

    Returns
    -------
    float, np.ndarray, or torch.Tensor
        Stream function influence for unit source strength [m].
    """
    if isinstance(xi, torch.Tensor) or isinstance(eta, torch.Tensor):
        if not isinstance(xi, torch.Tensor):
            xi = torch.as_tensor(xi, dtype=torch.float64)
        if not isinstance(eta, torch.Tensor):
            eta = torch.as_tensor(eta, dtype=xi.dtype, device=xi.device)
        if not isinstance(length, torch.Tensor):
            length = torch.as_tensor(length, dtype=xi.dtype, device=xi.device)
        if not isinstance(theta_panel, torch.Tensor):
            theta_panel = torch.as_tensor(theta_panel, dtype=xi.dtype, device=xi.device)

        t1 = xi
        t2 = xi - length
        r1_sq = torch.clamp(t1 * t1 + eta * eta, min=1e-30)
        r2_sq = torch.clamp(t2 * t2 + eta * eta, min=1e-30)

        val1 = t1 * torch.atan2(eta, t1) + 0.5 * eta * torch.log(r1_sq)
        val2 = t2 * torch.atan2(eta, t2) + 0.5 * eta * torch.log(r2_sq)
        return (val1 - val2 + theta_panel * length) / (2.0 * np.pi)

    # NumPy implementation
    xi_np = np.asarray(xi, dtype=np.float64)
    eta_np = np.asarray(eta, dtype=np.float64)
    len_np = np.asarray(length, dtype=np.float64)
    th_np = np.asarray(theta_panel, dtype=np.float64)

    t1 = xi_np
    t2 = xi_np - len_np
    r1_sq = np.maximum(t1 * t1 + eta_np * eta_np, 1e-30)
    r2_sq = np.maximum(t2 * t2 + eta_np * eta_np, 1e-30)

    val1 = t1 * np.arctan2(eta_np, t1) + 0.5 * eta_np * np.log(r1_sq)
    val2 = t2 * np.arctan2(eta_np, t2) + 0.5 * eta_np * np.log(r2_sq)
    psi_s = (val1 - val2 + th_np * len_np) / (2.0 * np.pi)

    if np.ndim(xi) == 0 and np.ndim(eta) == 0 and np.ndim(length) == 0:
        return float(psi_s)
    return psi_s


def compute_panel_local_coordinates_numpy(
    colloc: np.ndarray,
    p_start: np.ndarray,
    p_end: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Transform collocation points into local panel coordinates in NumPy.

    Parameters
    ----------
    colloc : np.ndarray
        Array of collocation points with shape (Nc, 2) [m].
    p_start : np.ndarray
        Array of panel start points with shape (Np, 2) [m].
    p_end : np.ndarray
        Array of panel end points with shape (Np, 2) [m].

    Returns
    -------
    tuple of (xi, eta, length, theta_panel)
        xi of shape (Nc, Np), eta of shape (Nc, Np), length of shape (1, Np),
        theta_panel of shape (1, Np).
    """
    dx = p_end[:, 0] - p_start[:, 0]
    dy = p_end[:, 1] - p_start[:, 1]
    length = np.hypot(dx, dy)
    tx = dx / length
    ty = dy / length
    nx = -ty
    ny = tx
    theta_panel = np.arctan2(dy, dx)

    # Broadcast (Nc, 1) and (1, Np)
    rx = colloc[:, 0:1] - p_start[np.newaxis, :, 0]
    ry = colloc[:, 1:2] - p_start[np.newaxis, :, 1]

    xi = rx * tx[np.newaxis, :] + ry * ty[np.newaxis, :]
    eta = rx * nx[np.newaxis, :] + ry * ny[np.newaxis, :]

    return xi, eta, length[np.newaxis, :], theta_panel[np.newaxis, :]


def compute_panel_local_coordinates_torch(
    colloc: torch.Tensor,
    p_start: torch.Tensor,
    p_end: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Transform collocation points into local panel coordinates in PyTorch.

    Supports optional leading batch dimensions (e.g., across angles of attack).

    Parameters
    ----------
    colloc : torch.Tensor
        Tensor of collocation points with shape (..., Nc, 2) [m].
    p_start : torch.Tensor
        Tensor of panel start points with shape (..., Np, 2) [m].
    p_end : torch.Tensor
        Tensor of panel end points with shape (..., Np, 2) [m].

    Returns
    -------
    tuple of (xi, eta, length, theta_panel)
        xi of shape (..., Nc, Np), eta of shape (..., Nc, Np),
        length of shape (..., 1, Np), theta_panel of shape (..., 1, Np).
    """
    dx = p_end[..., :, 0] - p_start[..., :, 0]
    dy = p_end[..., :, 1] - p_start[..., :, 1]
    length = torch.hypot(dx, dy)
    tx = dx / length
    ty = dy / length
    nx = -ty
    ny = tx
    theta_panel = torch.atan2(dy, dx)

    rx = colloc[..., :, 0:1] - p_start[..., None, :, 0]
    ry = colloc[..., :, 1:2] - p_start[..., None, :, 1]

    xi = rx * tx[..., None, :] + ry * ty[..., None, :]
    eta = rx * nx[..., None, :] + ry * ny[..., None, :]

    return xi, eta, length[..., None, :], theta_panel[..., None, :]
