# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
PyTorch / CUDA-accelerated Aerodynamic Influence Coefficient (AIC) assembly
and Biot-Savart tensor kernels for Ventorum.

Supports single-surface, multi-surface, ground effect (Method of Images),
batched multi-angle sweeps, and massive parallel case evaluations on GPU.
"""

from __future__ import annotations

import warnings
from typing import Sequence, Any
import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def has_cuda() -> bool:
    """Return True if PyTorch is installed and CUDA GPU is available."""
    if not _TORCH_AVAILABLE:
        return False
    return bool(torch.cuda.is_available())


def get_device(preferred: str | None = None) -> Any:
    """Resolve compute device (defaults to 'cuda' if available, otherwise 'cpu')."""
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for GPU acceleration in Ventorum.")
    if preferred is not None:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TensorPolarTable:
    """On-device tabulated airfoil polar lookup table for GPU-resident nonlinear LLT.

    Evaluates Cl(alpha) and Cd(alpha) via vectorized piecewise-linear interpolation
    directly in GPU VRAM, eliminating host-device synchronization during iterative loops.
    """

    def __init__(
        self,
        alpha_rad: np.ndarray | torch.Tensor,
        cl_data: np.ndarray | torch.Tensor,
        cd_data: np.ndarray | torch.Tensor | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        dev = device or get_device()
        dtype = dtype if dtype is not None else torch.float64
        if isinstance(alpha_rad, np.ndarray):
            self.alpha = torch.from_numpy(alpha_rad).to(device=dev, dtype=dtype)
        else:
            self.alpha = alpha_rad.to(device=dev, dtype=dtype)

        if isinstance(cl_data, np.ndarray):
            self.cl = torch.from_numpy(cl_data).to(device=dev, dtype=dtype)
        else:
            self.cl = cl_data.to(device=dev, dtype=dtype)

        if cd_data is not None:
            if isinstance(cd_data, np.ndarray):
                self.cd = torch.from_numpy(cd_data).to(device=dev, dtype=dtype)
            else:
                self.cd = cd_data.to(device=dev, dtype=dtype)
        else:
            self.cd = None

        self.device = dev
        self.dtype = dtype

    def evaluate_cl(self, alpha_eff: torch.Tensor) -> torch.Tensor:
        """Vectorized piecewise-linear Cl lookup on GPU matching np.interp (clamping to polar bounds)."""
        clamped = alpha_eff.clamp(self.alpha[0], self.alpha[-1])
        idx = torch.searchsorted(self.alpha, clamped).clamp(1, len(self.alpha) - 1)
        x0 = self.alpha[idx - 1]
        x1 = self.alpha[idx]
        y0 = self.cl[idx - 1]
        y1 = self.cl[idx]
        slope = (y1 - y0) / (x1 - x0)
        return y0 + slope * (clamped - x0)

    def evaluate_cd(self, alpha_eff: torch.Tensor) -> torch.Tensor | None:
        """Vectorized piecewise-linear Cd lookup on GPU matching np.interp (clamping to polar bounds)."""
        if self.cd is None:
            return None
        clamped = alpha_eff.clamp(self.alpha[0], self.alpha[-1])
        idx = torch.searchsorted(self.alpha, clamped).clamp(1, len(self.alpha) - 1)
        x0 = self.alpha[idx - 1]
        x1 = self.alpha[idx]
        y0 = self.cd[idx - 1]
        y1 = self.cd[idx]
        slope = (y1 - y0) / (x1 - x0)
        return y0 + slope * (clamped - x0)


def compute_batched_aic_and_rhs(
    cp: torch.Tensor,
    nl: torch.Tensor,
    nr: torch.Tensor,
    normals: torch.Tensor,
    trailing_dir: torch.Tensor,
    V_inf_vec: torch.Tensor,
    rc: float = 1.0e-10,
    h: float | None = None,
    return_details: bool = False,
    use_symmetry: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assemble global AIC matrix and RHS vector on GPU with batched tensor broadcasting.

    Parameters
    ----------
    cp : torch.Tensor, shape (B, N, 3) or (N, 3)
        Evaluation (control) points at 3/4-chord.
    nl, nr : torch.Tensor, shape (B, N, 3) or (N, 3)
        Left and right bound vortex nodes at 1/4-chord.
    normals : torch.Tensor, shape (B, N, 3) or (N, 3)
        Outward surface normals at control points.
    trailing_dir : torch.Tensor, shape (B, 3) or (3,)
        Unit trailing direction vector in body coordinates.
    V_inf_vec : torch.Tensor, shape (B, 3) or (3,)
        Freestream velocity vector [m/s].
    rc : float
        Regularization vortex core radius.
    h : float or None
        Height above ground plane for Method of Images.
    return_details : bool
        If True, returns (AIC, rhs, V_tot, V_trail).
    use_symmetry : bool
        If True, adds the influence of symmetric mirrored panels (reflection across Y=0).

    Returns
    -------
    AIC : torch.Tensor, shape (B, N, N) or (N, N)
    rhs : torch.Tensor, shape (B, N) or (N,)
    """
    is_batched = cp.ndim == 3
    if not is_batched:
        cp = cp.unsqueeze(0)
        nl = nl.unsqueeze(0)
        nr = nr.unsqueeze(0)
        normals = normals.unsqueeze(0)
        trailing_dir = trailing_dir.unsqueeze(0)
        V_inf_vec = V_inf_vec.unsqueeze(0)

    B, N, _ = cp.shape
    device = cp.device
    dtype = cp.dtype

    P = cp.unsqueeze(2)   # (B, N, 1, 3)
    A = nl.unsqueeze(1)   # (B, 1, N, 3)
    B_node = nr.unsqueeze(1) # (B, 1, N, 3)

    r1 = P - A  # (B, N, N, 3)
    r2 = P - B_node  # (B, N, N, 3)
    r0 = B_node - A  # (B, 1, N, 3)

    rc_sq = rc * rc
    inv_4pi = 1.0 / (4.0 * np.pi)

    # 1. Bound vortex segment A -> B
    cross_bound = torch.linalg.cross(r1, r2)
    cross_sq = torch.sum(cross_bound**2, dim=-1, keepdim=True)
    r0_sq = torch.sum(r0**2, dim=-1, keepdim=True)
    denom_bound = cross_sq + r0_sq * rc_sq

    r1_norm_reg = torch.sqrt(torch.sum(r1**2, dim=-1, keepdim=True) + rc_sq)
    r2_norm_reg = torch.sqrt(torch.sum(r2**2, dim=-1, keepdim=True) + rc_sq)
    diff = (r1 / r1_norm_reg) - (r2 / r2_norm_reg)
    dot_term = torch.sum(r0 * diff, dim=-1, keepdim=True)

    safe_denom_bound = torch.clamp(denom_bound, min=1e-30)
    scale_bound = torch.where(
        denom_bound > 0.0,
        inv_4pi * (dot_term / safe_denom_bound),
        torch.zeros_like(dot_term),
    )
    V_bound = cross_bound * scale_bound

    # 2. Trailing legs
    td = trailing_dir.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, 3)
    cross_A = torch.linalg.cross(td.expand_as(r1), r1)
    denom_A = torch.sum(cross_A**2, dim=-1, keepdim=True) + rc_sq
    cos_theta_A = torch.sum(td * r1, dim=-1, keepdim=True) / r1_norm_reg
    scale_A = torch.where(
        denom_A > 0.0,
        -inv_4pi * (1.0 + cos_theta_A) / torch.clamp(denom_A, min=1e-30),
        torch.zeros_like(denom_A),
    )
    V_left = cross_A * scale_A

    cross_B = torch.linalg.cross(td.expand_as(r2), r2)
    denom_B = torch.sum(cross_B**2, dim=-1, keepdim=True) + rc_sq
    cos_theta_B = torch.sum(td * r2, dim=-1, keepdim=True) / r2_norm_reg
    scale_B = torch.where(
        denom_B > 0.0,
        inv_4pi * (1.0 + cos_theta_B) / torch.clamp(denom_B, min=1e-30),
        torch.zeros_like(denom_B),
    )
    V_right = cross_B * scale_B

    V_trail = V_left + V_right
    V_tot = V_bound + V_trail

    # Mirrored symmetric panels (Y=0 reflection)
    if use_symmetry:
        nl_sym = nr.clone()
        nl_sym[..., 1] = -nl_sym[..., 1]
        nr_sym = nl.clone()
        nr_sym[..., 1] = -nr_sym[..., 1]

        A_s = nl_sym.unsqueeze(1)
        B_s = nr_sym.unsqueeze(1)
        r1_s = P - A_s
        r2_s = P - B_s
        r0_s = B_s - A_s

        cross_bound_s = torch.linalg.cross(r1_s, r2_s)
        cross_sq_s = torch.sum(cross_bound_s**2, dim=-1, keepdim=True)
        r0_sq_s = torch.sum(r0_s**2, dim=-1, keepdim=True)
        denom_bound_s = cross_sq_s + r0_sq_s * rc_sq

        r1_norm_reg_s = torch.sqrt(torch.sum(r1_s**2, dim=-1, keepdim=True) + rc_sq)
        r2_norm_reg_s = torch.sqrt(torch.sum(r2_s**2, dim=-1, keepdim=True) + rc_sq)
        diff_s = (r1_s / r1_norm_reg_s) - (r2_s / r2_norm_reg_s)
        dot_term_s = torch.sum(r0_s * diff_s, dim=-1, keepdim=True)

        safe_denom_bound_s = torch.clamp(denom_bound_s, min=1e-30)
        scale_bound_s = torch.where(
            denom_bound_s > 0.0,
            inv_4pi * (dot_term_s / safe_denom_bound_s),
            torch.zeros_like(dot_term_s),
        )
        V_bound_s = cross_bound_s * scale_bound_s

        cross_A_s = torch.linalg.cross(td.expand_as(r1_s), r1_s)
        denom_A_s = torch.sum(cross_A_s**2, dim=-1, keepdim=True) + rc_sq
        cos_theta_A_s = torch.sum(td * r1_s, dim=-1, keepdim=True) / r1_norm_reg_s
        scale_A_s = torch.where(
            denom_A_s > 0.0,
            -inv_4pi * (1.0 + cos_theta_A_s) / torch.clamp(denom_A_s, min=1e-30),
            torch.zeros_like(denom_A_s),
        )
        V_left_s = cross_A_s * scale_A_s

        cross_B_s = torch.linalg.cross(td.expand_as(r2_s), r2_s)
        denom_B_s = torch.sum(cross_B_s**2, dim=-1, keepdim=True) + rc_sq
        cos_theta_B_s = torch.sum(td * r2_s, dim=-1, keepdim=True) / r2_norm_reg_s
        scale_B_s = torch.where(
            denom_B_s > 0.0,
            inv_4pi * (1.0 + cos_theta_B_s) / torch.clamp(denom_B_s, min=1e-30),
            torch.zeros_like(denom_B_s),
        )
        V_right_s = cross_B_s * scale_B_s

        V_tot = V_tot + V_bound_s + V_left_s + V_right_s
        V_trail = V_trail + V_left_s + V_right_s

    # 3. Ground Effect (Method of Images)
    if h is not None:
        nl_img = nl.clone()
        nl_img[..., 2] = -2.0 * h - nl_img[..., 2]
        nr_img = nr.clone()
        nr_img[..., 2] = -2.0 * h - nr_img[..., 2]
        td_img = trailing_dir.clone()
        td_img[..., 2] = -td_img[..., 2]

        A_img = nl_img.unsqueeze(1)
        B_img = nr_img.unsqueeze(1)
        r1_img = P - A_img
        r2_img = P - B_img
        r0_img = B_img - A_img

        cross_bound_img = torch.linalg.cross(r1_img, r2_img)
        cross_sq_img = torch.sum(cross_bound_img**2, dim=-1, keepdim=True)
        r0_sq_img = torch.sum(r0_img**2, dim=-1, keepdim=True)
        denom_bound_img = cross_sq_img + r0_sq_img * rc_sq
        r1_norm_reg_img = torch.sqrt(torch.sum(r1_img**2, dim=-1, keepdim=True) + rc_sq)
        r2_norm_reg_img = torch.sqrt(torch.sum(r2_img**2, dim=-1, keepdim=True) + rc_sq)
        diff_img = (r1_img / r1_norm_reg_img) - (r2_img / r2_norm_reg_img)
        dot_term_img = torch.sum(r0_img * diff_img, dim=-1, keepdim=True)
        scale_bound_img = torch.where(
            denom_bound_img > 0.0,
            (-1.0 * inv_4pi) * (dot_term_img / torch.clamp(denom_bound_img, min=1e-30)),
            torch.zeros_like(dot_term_img),
        )
        V_bound_img = cross_bound_img * scale_bound_img

        td_i = td_img.unsqueeze(1).unsqueeze(1)
        cross_A_img = torch.linalg.cross(td_i.expand_as(r1_img), r1_img)
        denom_A_img = torch.sum(cross_A_img**2, dim=-1, keepdim=True) + rc_sq
        cos_theta_A_img = torch.sum(td_i * r1_img, dim=-1, keepdim=True) / r1_norm_reg_img
        scale_A_img = torch.where(
            denom_A_img > 0.0,
            -(-1.0 * inv_4pi) * (1.0 + cos_theta_A_img) / torch.clamp(denom_A_img, min=1e-30),
            torch.zeros_like(denom_A_img),
        )
        V_left_img = cross_A_img * scale_A_img

        cross_B_img = torch.linalg.cross(td_i.expand_as(r2_img), r2_img)
        denom_B_img = torch.sum(cross_B_img**2, dim=-1, keepdim=True) + rc_sq
        cos_theta_B_img = torch.sum(td_i * r2_img, dim=-1, keepdim=True) / r2_norm_reg_img
        scale_B_img = torch.where(
            denom_B_img > 0.0,
            (-1.0 * inv_4pi) * (1.0 + cos_theta_B_img) / torch.clamp(denom_B_img, min=1e-30),
            torch.zeros_like(denom_B_img),
        )
        V_right_img = cross_B_img * scale_B_img

        V_tot = V_tot + V_bound_img + V_left_img + V_right_img
        V_trail = V_trail + V_left_img + V_right_img

        if use_symmetry:
            nl_img_s = nl_sym.clone()
            nl_img_s[..., 2] = -2.0 * h - nl_img_s[..., 2]
            nr_img_s = nr_sym.clone()
            nr_img_s[..., 2] = -2.0 * h - nr_img_s[..., 2]

            A_img_s = nl_img_s.unsqueeze(1)
            B_img_s = nr_img_s.unsqueeze(1)
            r1_img_s = P - A_img_s
            r2_img_s = P - B_img_s
            r0_img_s = B_img_s - A_img_s

            cross_bound_img_s = torch.linalg.cross(r1_img_s, r2_img_s)
            cross_sq_img_s = torch.sum(cross_bound_img_s**2, dim=-1, keepdim=True)
            r0_sq_img_s = torch.sum(r0_img_s**2, dim=-1, keepdim=True)
            denom_bound_img_s = cross_sq_img_s + r0_sq_img_s * rc_sq
            r1_norm_reg_img_s = torch.sqrt(torch.sum(r1_img_s**2, dim=-1, keepdim=True) + rc_sq)
            r2_norm_reg_img_s = torch.sqrt(torch.sum(r2_img_s**2, dim=-1, keepdim=True) + rc_sq)
            diff_img_s = (r1_img_s / r1_norm_reg_img_s) - (r2_img_s / r2_norm_reg_img_s)
            dot_term_img_s = torch.sum(r0_img_s * diff_img_s, dim=-1, keepdim=True)
            scale_bound_img_s = torch.where(
                denom_bound_img_s > 0.0,
                (-1.0 * inv_4pi) * (dot_term_img_s / torch.clamp(denom_bound_img_s, min=1e-30)),
                torch.zeros_like(dot_term_img_s),
            )
            V_bound_img_s = cross_bound_img_s * scale_bound_img_s

            cross_A_img_s = torch.linalg.cross(td_i.expand_as(r1_img_s), r1_img_s)
            denom_A_img_s = torch.sum(cross_A_img_s**2, dim=-1, keepdim=True) + rc_sq
            cos_theta_A_img_s = torch.sum(td_i * r1_img_s, dim=-1, keepdim=True) / r1_norm_reg_img_s
            scale_A_img_s = torch.where(
                denom_A_img_s > 0.0,
                -(-1.0 * inv_4pi) * (1.0 + cos_theta_A_img_s) / torch.clamp(denom_A_img_s, min=1e-30),
                torch.zeros_like(denom_A_img_s),
            )
            V_left_img_s = cross_A_img_s * scale_A_img_s

            cross_B_img_s = torch.linalg.cross(td_i.expand_as(r2_img_s), r2_img_s)
            denom_B_img_s = torch.sum(cross_B_img_s**2, dim=-1, keepdim=True) + rc_sq
            cos_theta_B_img_s = torch.sum(td_i * r2_img_s, dim=-1, keepdim=True) / r2_norm_reg_img_s
            scale_B_img_s = torch.where(
                denom_B_img_s > 0.0,
                (-1.0 * inv_4pi) * (1.0 + cos_theta_B_img_s) / torch.clamp(denom_B_img_s, min=1e-30),
                torch.zeros_like(denom_B_img_s),
            )
            V_right_img_s = cross_B_img_s * scale_B_img_s

            V_tot = V_tot + V_bound_img_s + V_left_img_s + V_right_img_s
            V_trail = V_trail + V_left_img_s + V_right_img_s

    # 4. Normalwash projection & RHS
    AIC = torch.einsum('bijk,bik->bij', V_tot, normals)
    rhs = -torch.einsum('bik,bk->bi', normals, V_inf_vec)

    if not is_batched:
        AIC = AIC.squeeze(0)
        rhs = rhs.squeeze(0)
        V_tot = V_tot.squeeze(0)
        V_trail = V_trail.squeeze(0)

    if return_details:
        return AIC, rhs, V_tot, V_trail
    return AIC, rhs


class GPUGeometryCache:
    """Precomputed bound-vortex geometry tensors resident in GPU memory."""

    def __init__(
        self,
        cp: torch.Tensor,
        normals: torch.Tensor,
        nl: torch.Tensor,
        nr: torch.Tensor,
        r1: torch.Tensor,
        r2: torch.Tensor,
        r1_norm_reg: torch.Tensor,
        r2_norm_reg: torch.Tensor,
        V_bound: torch.Tensor,
        AIC_bound: torch.Tensor,
        h: float | None = None,
        V_bound_img: torch.Tensor | None = None,
        r1_img: torch.Tensor | None = None,
        r2_img: torch.Tensor | None = None,
        r1_img_reg: torch.Tensor | None = None,
        r2_img_reg: torch.Tensor | None = None,
    ):
        self.cp = cp
        self.normals = normals
        self.nl = nl
        self.nr = nr
        self.r1 = r1
        self.r2 = r2
        self.r1_norm_reg = r1_norm_reg
        self.r2_norm_reg = r2_norm_reg
        self.V_bound = V_bound
        self.AIC_bound = AIC_bound
        self.h = h
        self.V_bound_img = V_bound_img
        self.r1_img = r1_img
        self.r2_img = r2_img
        self.r1_img_reg = r1_img_reg
        self.r2_img_reg = r2_img_reg


def precompute_gpu_geometry_cache(
    cp_np: np.ndarray,
    normals_np: np.ndarray,
    nl_np: np.ndarray,
    nr_np: np.ndarray,
    h: float | None = None,
    rc: float = 1.0e-10,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> GPUGeometryCache:
    """Precompute bound-vortex kernel and distance vectors directly in GPU memory."""
    dev = device or get_device()
    dtype = dtype if dtype is not None else torch.float64
    cp = torch.from_numpy(cp_np).to(device=dev, dtype=dtype)
    normals = torch.from_numpy(normals_np).to(device=dev, dtype=dtype)
    nl = torch.from_numpy(nl_np).to(device=dev, dtype=dtype)
    nr = torch.from_numpy(nr_np).to(device=dev, dtype=dtype)

    P = cp.unsqueeze(1)
    A = nl.unsqueeze(0)
    B = nr.unsqueeze(0)

    r1 = P - A
    r2 = P - B
    r0 = B - A
    rc_sq = rc * rc
    inv_4pi = 1.0 / (4.0 * np.pi)

    cross_bound = torch.linalg.cross(r1, r2)
    cross_sq = torch.sum(cross_bound**2, dim=-1, keepdim=True)
    r0_sq = torch.sum(r0**2, dim=-1, keepdim=True)
    denom_bound = cross_sq + r0_sq * rc_sq

    r1_norm_reg = torch.sqrt(torch.sum(r1**2, dim=-1, keepdim=True) + rc_sq)
    r2_norm_reg = torch.sqrt(torch.sum(r2**2, dim=-1, keepdim=True) + rc_sq)
    diff = (r1 / r1_norm_reg) - (r2 / r2_norm_reg)
    dot_term = torch.sum(r0 * diff, dim=-1, keepdim=True)

    safe_denom_bound = torch.clamp(denom_bound, min=1e-30)
    scale_bound = torch.where(
        denom_bound > 0.0,
        inv_4pi * (dot_term / safe_denom_bound),
        torch.zeros_like(dot_term),
    )
    V_bound = cross_bound * scale_bound
    AIC_bound = torch.einsum('ijk,ik->ij', V_bound, normals)

    V_bound_img = None
    r1_img = None
    r2_img = None
    r1_img_reg = None
    r2_img_reg = None

    if h is not None:
        nl_img = nl.clone()
        nl_img[:, 2] = -2.0 * h - nl_img[:, 2]
        nr_img = nr.clone()
        nr_img[:, 2] = -2.0 * h - nr_img[:, 2]

        A_img = nl_img.unsqueeze(0)
        B_img = nr_img.unsqueeze(0)
        r1_img = P - A_img
        r2_img = P - B_img
        r0_img = B_img - A_img

        cross_bound_img = torch.linalg.cross(r1_img, r2_img)
        cross_sq_img = torch.sum(cross_bound_img**2, dim=-1, keepdim=True)
        r0_sq_img = torch.sum(r0_img**2, dim=-1, keepdim=True)
        denom_bound_img = cross_sq_img + r0_sq_img * rc_sq

        r1_img_reg = torch.sqrt(torch.sum(r1_img**2, dim=-1, keepdim=True) + rc_sq)
        r2_img_reg = torch.sqrt(torch.sum(r2_img**2, dim=-1, keepdim=True) + rc_sq)
        diff_img = (r1_img / r1_img_reg) - (r2_img / r2_img_reg)
        dot_term_img = torch.sum(r0_img * diff_img, dim=-1, keepdim=True)

        safe_denom_img = torch.clamp(denom_bound_img, min=1e-30)
        scale_bound_img = torch.where(
            denom_bound_img > 0.0,
            (-1.0 * inv_4pi) * (dot_term_img / safe_denom_img),
            torch.zeros_like(dot_term_img),
        )
        V_bound_img = cross_bound_img * scale_bound_img
        AIC_bound = AIC_bound + torch.einsum('ijk,ik->ij', V_bound_img, normals)

    return GPUGeometryCache(
        cp=cp,
        normals=normals,
        nl=nl,
        nr=nr,
        r1=r1,
        r2=r2,
        r1_norm_reg=r1_norm_reg,
        r2_norm_reg=r2_norm_reg,
        V_bound=V_bound,
        AIC_bound=AIC_bound,
        h=h,
        V_bound_img=V_bound_img,
        r1_img=r1_img,
        r2_img=r2_img,
        r1_img_reg=r1_img_reg,
        r2_img_reg=r2_img_reg,
    )
