# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""GPU solve path of the 3D source-doublet panel method.

This module runs the dense linear algebra of :mod:`ventorum.panel3d.solver`
on a CUDA GPU. The influence matrices come from the Warp kernels of
:mod:`ventorum.panel3d.kernels_warp` (one thread per target-panel pair).
The factorization uses ``torch.linalg.lu_factor`` on the device, and all
unit onset right-hand sides are solved in one ``lu_solve`` call per system.
The influence matrices are assembled in float64 (the on-plane self terms
need float64 coordinates); the factorization, the solve and the array
post-processing run in the requested dtype. Array work of the
post-processing (superposition of the unit solutions, surface velocity
and pressure coefficient) runs on the device with PyTorch. Results
return to the host as numpy arrays. Warp and PyTorch are imported lazily
inside the functions, so importing this module never loads them.

The precision follows the Ventorum device and precision selection (see
:mod:`ventorum.gpu`): ``dtype=None`` uses :func:`ventorum.gpu.get_precision`
(float32 on consumer GPUs is accepted, no forced float64). The dense matrix
must fit in the free device memory; otherwise the solve is refused with a
clear error that reports the size (no silent fallback to the CPU).

Published sources
-----------------
* Hess, J. L. and Smith, A. M. O., 1967, Calculation of potential flow
  about arbitrary bodies, Progress in Aeronautical Sciences, Vol. 8,
  pp. 1-138.
* Katz, J. and Plotkin, A., 2001, Low-Speed Aerodynamics, 2nd ed.,
  Cambridge University Press, section 10.4 and chapter 12.
* Newman, J. N., 1986, Distributions of sources and normal dipoles over a
  quadrilateral panel, Journal of Engineering Mathematics, Vol. 20,
  pp. 113-126.
* van Oosterom, A. and Strackee, J., 1983, The solid angle of a plane
  triangle, IEEE Transactions on Biomedical Engineering, Vol. 30,
  pp. 125-126.

Conventions
-----------
* Geometry axes: x aft [m], y right [m], z up [m].
* Doublet strengths in [m^2/s]. Source strengths in [m/s].
* Pressure coefficient [-].
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "add_wake_influence",
    "check_device_memory",
    "doublet_potential_device",
    "evaluate_condition_on_device",
    "factorize_solve",
    "free_device_bytes",
    "influence_matrices_device",
    "lu_factor_device",
    "lu_solve_device",
    "required_bytes",
    "resolve_device",
    "resolve_dtype",
    "strip_sum",
]

#: Number of unit onset right-hand sides solved together (u, v, w, p, q, r).
N_ONSETS = 6


def _precision_name(dtype) -> str:
    """Return 'float32' or 'float64' for a torch, numpy or string dtype."""
    if dtype is None:
        raise ValueError("A dtype is needed to name the precision.")
    try:
        import torch

        if isinstance(dtype, torch.dtype):
            if dtype == torch.float32:
                return "float32"
            if dtype == torch.float64:
                return "float64"
            raise ValueError(
                f"The gpu panel backend uses float32 or float64, not {dtype}."
            )
    except ImportError:
        pass
    key = str(dtype).strip().lower()
    if key in ("float32", "float64"):
        return key
    try:
        resolved = np.dtype(dtype)
    except TypeError:
        raise ValueError(
            f"The gpu panel backend uses float32 or float64, not {dtype!r}."
        ) from None
    if resolved == np.dtype(np.float32):
        return "float32"
    if resolved == np.dtype(np.float64):
        return "float64"
    raise ValueError(
        f"The gpu panel backend uses float32 or float64, not {dtype!r}."
    )


def required_bytes(
    n_panels: int,
    dtype,
    n_systems: int = 1,
    *,
    half: bool | None = None,
    ground: bool = False,
    body_cached: bool = False,
) -> int:
    """Return the device bytes that one configuration of the solver allocates.

    The estimate is the peak of the new allocations while the solver
    assembles and factorizes the systems of one configuration (free air,
    or one attitude and height in ground effect). Let ``N`` be the number
    of solved panels, ``s`` the bytes of the solve dtype (4 or 8) and
    ``n_body`` the body matrices that the solver keeps for later
    configurations (B and C, two for a full mesh and four for a half mesh
    with the mirror pair). The terms are:

    * the float64 assembly pair of one kernel launch, ``2 N^2 8`` (the
      on-plane self terms need float64 coordinates);
    * the kept body matrices, ``n_body N^2 s`` (only for the first
      configuration, see *body_cached*);
    * the ground image matrices in ground effect, ``n_body N^2 s``;
    * the two summed matrices of the self and mirror parts on a half mesh,
      ``2 N^2 s``;
    * the system matrix, the source matrix and the LU factors of the system
      that is factorized, ``3 N^2 s``, plus the LU factors that stay from
      the other systems of the configuration, ``(n_systems - 1) N^2 s``;
    * the right-hand sides and solutions, ``2 N 6 s`` per system.

    Parameters
    ----------
    n_panels : int
        Number of solved body panels [-].
    dtype
        Arithmetic of the factorization and solve (torch, numpy or string
        dtype).
    n_systems : int, default=1
        Number of dense systems of the configuration [-] (2 for the
        symmetric and antisymmetric half mesh systems).
    half : bool or None, optional
        True for a half mesh (mirror matrices). None takes
        ``n_systems == 2``.
    ground : bool, default=False
        True in ground effect (image matrices).
    body_cached : bool, default=False
        True when the body matrices are already on the device.

    Returns
    -------
    int
        Required device memory in bytes.
    """
    n = int(n_panels)
    n_sys = int(n_systems)
    if n <= 0 or n_sys <= 0:
        return 0
    s = 4 if _precision_name(dtype) == "float32" else 8
    is_half = (n_sys == 2) if half is None else bool(half)
    n_body = 4 if is_half else 2
    nn = n * n
    total = 2 * nn * 8
    if not body_cached:
        total += n_body * nn * s
    if ground:
        total += n_body * nn * s
    if is_half:
        total += 2 * nn * s
    total += 3 * nn * s + (n_sys - 1) * nn * s
    total += n_sys * 2 * n * N_ONSETS * s
    return int(total)


def resolve_device(device=None):
    """Return the CUDA device of the gpu panel backend.

    Parameters
    ----------
    device : optional
        CUDA device (for example ``"cuda:0"``). None selects the current
        CUDA device.

    Returns
    -------
    torch.device
        The selected CUDA device.

    Raises
    ------
    RuntimeError
        If no CUDA GPU is available, or if *device* is not a CUDA device.
    """
    import torch

    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "The gpu panel backend needs a CUDA GPU, but PyTorch finds "
                "none. Check the CUDA driver, or solve with backend 'numpy'."
            )
        return torch.device("cuda", torch.cuda.current_device())
    dev = device if isinstance(device, torch.device) else torch.device(device)
    if dev.type != "cuda":
        raise ValueError(
            f"The gpu panel backend runs on a CUDA device, not on {dev}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "The gpu panel backend needs a CUDA GPU, but PyTorch finds "
            "none. Check the CUDA driver, or solve with backend 'numpy'."
        )
    return dev


def resolve_dtype(dtype=None):
    """Return the torch dtype of the gpu panel backend.

    Parameters
    ----------
    dtype : optional
        Requested arithmetic (torch, numpy or string dtype). None follows
        the Ventorum precision selection
        (:func:`ventorum.gpu.get_precision`).

    Returns
    -------
    torch.dtype
        ``torch.float32`` or ``torch.float64``.
    """
    import torch

    if dtype is None:
        from ventorum import gpu as gpu_sel

        selected = gpu_sel.get_precision()
        return torch.float32 if str(selected) == "float32" else torch.float64
    name = _precision_name(dtype)
    return torch.float32 if name == "float32" else torch.float64


def free_device_bytes(device=None) -> int:
    """Return the device bytes that PyTorch can still use.

    The value is the free memory of the driver plus the memory that the
    PyTorch caching allocator holds but does not use.

    Parameters
    ----------
    device : optional
        CUDA device. None selects the current CUDA device.

    Returns
    -------
    int
        Usable device memory in bytes.
    """
    import torch

    dev = resolve_device(device)
    free, _ = torch.cuda.mem_get_info(dev)
    cached = int(torch.cuda.memory_reserved(dev)) - int(torch.cuda.memory_allocated(dev))
    return int(free) + max(cached, 0)


def check_device_memory(
    n_panels: int,
    dtype,
    device=None,
    n_systems: int = 1,
    *,
    half: bool | None = None,
    ground: bool = False,
    body_cached: bool = False,
) -> None:
    """Refuse the solve when the dense systems do not fit in free memory.

    Parameters
    ----------
    n_panels : int
        Number of solved body panels [-].
    dtype
        Arithmetic of the solve (torch, numpy or string dtype).
    device : optional
        CUDA device. None selects the current CUDA device.
    n_systems : int, default=1
        Number of dense systems of the configuration [-].
    half : bool or None, optional
        True for a half mesh (see :func:`required_bytes`).
    ground : bool, default=False
        True in ground effect.
    body_cached : bool, default=False
        True when the body matrices are already on the device.

    Raises
    ------
    RuntimeError
        If no CUDA GPU is available.
    MemoryError
        If the dense systems do not fit in the free device memory. The
        message reports the required and the free size.
    """
    dev = resolve_device(device)
    need = required_bytes(
        n_panels, dtype, n_systems=n_systems, half=half, ground=ground, body_cached=body_cached
    )
    free = free_device_bytes(dev)
    if need > int(free):
        raise MemoryError(
            f"The gpu panel backend needs about {need / 2**20:.1f} MiB for "
            f"the dense {int(n_panels)} x {int(n_panels)} system, but only "
            f"{int(free) / 2**20:.1f} MiB is free on {dev}. Use a coarser "
            "mesh or a GPU with more memory."
        )


def influence_matrices_device(
    collocation,
    panels,
    *,
    device=None,
    dtype=None,
    far_ratio: float = 5.0,
):
    """Compute B and C on the device with the Warp kernels.

    Same results as :func:`ventorum.panel3d.kernels.influence_matrices`:
    ``B`` holds the source potentials and ``C`` the doublet potentials.

    Parameters
    ----------
    collocation : numpy.ndarray
        Collocation point coordinates with shape (P, 3) in m.
    panels : numpy.ndarray
        Panel corner coordinates with shape (M, 4, 3) in m.
    device : optional
        CUDA device. None selects the current CUDA device.
    dtype : optional
        Arithmetic of the kernels. None follows the Ventorum precision
        selection.
    far_ratio : float, default=5.0
        Distance-to-diameter ratio for point far-field transition.

    Returns
    -------
    tuple of torch.Tensor
        (B, C) on the CUDA device with shapes (P, M).
    """
    from ventorum.panel3d.kernels_warp import influence_matrices_warp

    dev = resolve_device(device)
    torch_dtype = resolve_dtype(dtype)
    n_targets = int(np.asarray(collocation).shape[0])
    n_sources = int(np.asarray(panels).shape[0])
    item = 4 if _precision_name(torch_dtype) == "float32" else 8
    need = 2 * n_targets * n_sources * item
    free = free_device_bytes(dev)
    if need > free:
        raise MemoryError(
            f"The gpu panel backend needs about {need / 2**20:.1f} MiB for "
            f"the {n_targets} x {n_sources} influence matrices, but only "
            f"{free / 2**20:.1f} MiB is free on {dev}. Use a coarser mesh "
            "or a GPU with more memory."
        )
    return influence_matrices_warp(
        collocation, panels, far_ratio=far_ratio, device=dev, dtype=torch_dtype
    )


def doublet_potential_device(
    points,
    panels,
    *,
    device=None,
    dtype=None,
    far_ratio: float = 5.0,
):
    """Compute the doublet potentials C on the device with the Warp kernels.

    Same results as :func:`ventorum.panel3d.kernels.doublet_potential`.
    The source potentials of the same kernel launch are discarded.

    Parameters
    ----------
    points : numpy.ndarray
        Target coordinates with shape (P, 3) in m.
    panels : numpy.ndarray
        Panel corner coordinates with shape (M, 4, 3) in m.
    device : optional
        CUDA device. None selects the current CUDA device.
    dtype : optional
        Arithmetic of the kernels. None follows the Ventorum precision
        selection.
    far_ratio : float, default=5.0
        Distance-to-diameter ratio for point far-field transition.

    Returns
    -------
    torch.Tensor
        Doublet potentials on the CUDA device with shape (P, M).
    """
    _, mat_c = influence_matrices_device(
        points, panels, device=device, dtype=dtype, far_ratio=far_ratio
    )
    return mat_c


def add_wake_influence(
    mat_a,
    mat_cw,
    wake_up,
    wake_lo,
    strip_panel_indices,
    sign: float = 1.0,
):
    """Add wake strip doublet influences to matrix A on the device.

    Each wake strip enters the solved system through the doublet jump of
    its trailing-edge pair: the summed wake potentials are added to the
    upper panel column and subtracted from the lower panel column.

    Parameters
    ----------
    mat_a : torch.Tensor
        Doublet system matrix with shape (N, N).
    mat_cw : torch.Tensor
        Doublet potentials of the wake panels with shape (N, Nw).
    wake_up : array_like
        Upper trailing-edge panel indices with shape (Ns,).
    wake_lo : array_like
        Lower trailing-edge panel indices with shape (Ns,).
    strip_panel_indices : list of list of int
        Wake panel indices of each strip.
    sign : float, default=1.0
        Sign of the image contribution (always 1.0; images go through the
        same kernels with preserved outward normals).

    Returns
    -------
    torch.Tensor
        The same matrix *mat_a*, updated in place.
    """
    import torch

    up_idx = torch.as_tensor(np.asarray(wake_up, dtype=np.int64), device=mat_a.device)
    lo_idx = torch.as_tensor(np.asarray(wake_lo, dtype=np.int64), device=mat_a.device)
    cs = strip_sum(mat_cw, strip_panel_indices) * float(sign)
    # Accumulate repeated panel indices, including upper/lower overlap.
    mat_a.index_add_(1, up_idx, cs)
    mat_a.index_add_(1, lo_idx, -cs)
    return mat_a


def strip_sum(mat_cw, strip_panel_indices):
    """Sum wake panel potentials into one column per wake strip on the device.

    Parameters
    ----------
    mat_cw : torch.Tensor
        Doublet potentials of the wake panels with shape (N, Nw).
    strip_panel_indices : list of list of int
        Wake panel indices of each strip.

    Returns
    -------
    torch.Tensor
        Strip influence matrix with shape (N, Ns) on the same device.
    """
    import torch

    n_targets = int(mat_cw.shape[0])
    n_strips = len(strip_panel_indices)
    out = torch.zeros(
        (n_targets, n_strips), dtype=mat_cw.dtype, device=mat_cw.device
    )
    lengths = np.fromiter(map(len, strip_panel_indices), dtype=np.int64, count=n_strips)
    if not np.any(lengths):
        return out
    # Deterministic segment sums: index_add_ on the GPU uses atomic additions,
    # whose order (and so the float result) can change from run to run.
    if np.all(lengths == lengths[0]):
        cols = torch.as_tensor(
            np.concatenate(strip_panel_indices).astype(np.int64, copy=False), device=mat_cw.device
        )
        gathered = mat_cw.index_select(1, cols)
        return gathered.reshape(n_targets, n_strips, int(lengths[0])).sum(dim=2)
    for s, cols in enumerate(strip_panel_indices):
        if len(cols):
            col_idx = torch.as_tensor(np.asarray(cols, dtype=np.int64), device=mat_cw.device)
            out[:, s] = mat_cw.index_select(1, col_idx).sum(dim=1)
    return out


def lu_factor_device(mat_a):
    """Return the LU factors of A on the device.

    Parameters
    ----------
    mat_a : torch.Tensor
        System matrix with shape (N, N) on the CUDA device.

    Returns
    -------
    tuple
        ``(LU, piv)`` of ``torch.linalg.lu_factor``.
    """
    import torch

    return torch.linalg.lu_factor(mat_a)


def lu_solve_device(lu_info, rhs):
    """Solve A x = rhs with the LU factors of :func:`lu_factor_device`.

    Parameters
    ----------
    lu_info : tuple
        ``(LU, piv)`` on the CUDA device.
    rhs : torch.Tensor
        Right-hand sides with shape (N, K) on the same device.

    Returns
    -------
    torch.Tensor
        Solution with shape (N, K). One ``lu_solve`` call solves all
        columns.
    """
    import torch

    lu, piv = lu_info
    return torch.linalg.lu_solve(lu, piv, rhs)


def factorize_solve(mat_a, rhs):
    """Factorize A on the device and solve all right-hand sides at once.

    Parameters
    ----------
    mat_a : torch.Tensor
        System matrix with shape (N, N) on the CUDA device.
    rhs : torch.Tensor
        Right-hand sides with shape (N, K) on the same device.

    Returns
    -------
    tuple
        (solution, (LU, piv)): the solution with shape (N, K) and the
        factorization for reuse (for example pressure Kutta sensitivities).
        One ``lu_solve`` call solves all K columns.
    """
    lu_info = lu_factor_device(mat_a)
    return lu_solve_device(lu_info, rhs), lu_info


def evaluate_condition_on_device(
    unit_mu,
    unit_sigma,
    unit_grad,
    coeffs,
    v_onset,
    normals,
    v_inf: float,
):
    """Superpose the unit solutions and the surface Cp on the device.

    The doublet strengths, the source strengths and the surface gradient
    field are linear in the six unit onset weights, so one matrix product
    per field combines them. The surface velocity and the pressure
    coefficient follow the same equations as
    :func:`ventorum.panel3d.post.compute_surface_velocity_and_cp`.

    Parameters
    ----------
    unit_mu : torch.Tensor
        Unit onset doublet strengths with shape (N, 6) in m^2/s.
    unit_sigma : torch.Tensor
        Unit onset source strengths with shape (N, 6) in m/s.
    unit_grad : torch.Tensor
        Unit onset surface gradients on the full mesh with shape (Nf, 6, 3)
        in m/s per unit onset.
    coeffs : array_like
        Six unit onset weights (u, v, w in m/s, p, q, r in rad/s).
    v_onset : array_like
        Onset velocity at the full mesh centroids with shape (Nf, 3) in m/s.
    normals : array_like
        Full mesh unit normals with shape (Nf, 3).
    v_inf : float
        Free-stream speed in m/s.

    Returns
    -------
    tuple of numpy.ndarray
        (mu, sigma, grad, velocity, cp): body doublets (N,) in m^2/s,
        body sources (N,) in m/s, full surface gradient (Nf, 3) in m/s,
        full surface velocity (Nf, 3) in m/s, full pressure coefficients
        (Nf,) [-], all in float64 on the host.
    """
    import torch

    dev = unit_mu.device
    dt = unit_mu.dtype
    c = torch.as_tensor(np.asarray(coeffs, dtype=np.float64), dtype=dt, device=dev)
    mu_t = unit_mu @ c
    sigma_t = unit_sigma @ c
    grad_t = torch.einsum("k,nkd->nd", c, unit_grad)

    def _host_copy(a):
        arr = np.ascontiguousarray(a, dtype=np.float64)
        return arr if arr.flags.writeable else arr.copy()

    von_t = torch.as_tensor(_host_copy(v_onset), dtype=dt, device=dev)
    nrm_t = torch.as_tensor(_host_copy(normals), dtype=dt, device=dev)
    v_norm = (von_t * nrm_t).sum(dim=-1, keepdim=True)
    v_surf_t = (von_t - v_norm * nrm_t) - grad_t
    v_inf_f = float(v_inf)
    if v_inf_f > 1.0e-12:
        cp_t = 1.0 - (v_surf_t**2).sum(dim=-1) / (v_inf_f**2)
    else:
        cp_t = torch.zeros(
            (int(nrm_t.shape[0]),), dtype=dt, device=dev
        )
    def _to_host(t):
        return t.to(torch.float64).cpu().numpy()

    return (
        _to_host(mu_t),
        _to_host(sigma_t),
        _to_host(grad_t),
        _to_host(v_surf_t),
        _to_host(cp_t),
    )
