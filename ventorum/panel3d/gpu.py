# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""GPU solve path of the 3D source-doublet panel method.

This module runs the dense linear algebra of :mod:`ventorum.panel3d.solver`
on a CUDA GPU. The influence matrices come from the Warp kernels of
:mod:`ventorum.panel3d.kernels_warp` (one thread per target-panel pair).
The factorization uses ``torch.linalg.lu_factor`` on the device, and all
unit onset right-hand sides are solved in one ``lu_solve`` call per system.
The influence matrices are assembled in float64 (the on-plane self terms
need float64 coordinates); the factorization, the solve and the array
post-processing run in the requested dtype. The solver assembles only the
systems that it factorizes: the pair kernels give the parity matrix
``C_a + sign * C_b`` of a panel set and its mirror (or image) in one pass,
rounded to the solve dtype block by block, and the source right-hand
sides ``(B_a + sign * B_b) @ sigma`` in float64 without the source matrix. Array work of the
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
    "doublet_pair_matrix_device",
    "doublet_potential_device",
    "evaluate_condition_on_device",
    "factorize_solve",
    "free_device_bytes",
    "influence_matrices_device",
    "lu_factor_device",
    "lu_solve_device",
    "pair_system_device",
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
    assembly: str = "parts",
) -> int:
    """Return the device bytes that one configuration of the solver allocates.

    The estimate is the peak of the new allocations while the solver
    assembles and factorizes the systems of one configuration (free air,
    or one attitude and height in ground effect). Let ``N`` be the number
    of solved panels and ``s`` the bytes of the solve dtype (4 or 8).

    With ``assembly="fused"`` (the assembly of
    :class:`ventorum.panel3d.solver.PanelSolver`) the pair kernels give
    only the systems that are factorized. The terms are:

    * the parity matrices of the body that the solver keeps for later
      configurations, ``n_systems N^2 s`` (only when they are not on the
      device yet, see *body_cached*);
    * the image parity matrix in ground effect, ``N^2 s``;
    * the system matrix that is factorized and its LU factors,
      ``2 N^2 s``, plus the LU factors that stay from the other systems
      of the configuration, ``(n_systems - 1) N^2 s``;
    * one float64 block of rows of the doublet pair kernel,
      ``min(BLOCK_BYTES, 8 N^2)``, when the solve dtype is float32;
    * the float64 partial sums of the source kernel,
      ``8 N 6 ceil(N / RHS_CHUNK)``, and the right-hand sides and
      solutions, ``2 N 6 s`` per system.

    With ``assembly="parts"`` (the default; separate source and doublet
    matrices B and C of the body, ``n_body`` of them: two for a full mesh
    and four for a half mesh with the mirror pair) the terms are:

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
    assembly : str, default="parts"
        ``"fused"`` for the parity systems of the pair kernels, or
        ``"parts"`` for separate B and C matrices of the body.

    Returns
    -------
    int
        Required device memory in bytes.

    Raises
    ------
    ValueError
        If *assembly* is not ``"parts"`` or ``"fused"``.
    """
    if assembly not in ("parts", "fused"):
        raise ValueError(f"Unknown assembly {assembly!r}; use 'parts' or 'fused'.")
    n = int(n_panels)
    n_sys = int(n_systems)
    if n <= 0 or n_sys <= 0:
        return 0
    s = 4 if _precision_name(dtype) == "float32" else 8
    nn = n * n
    if assembly == "fused":
        from ventorum.panel3d.kernels_warp import BLOCK_BYTES, RHS_CHUNK

        total = 0 if body_cached else n_sys * nn * s
        if ground:
            total += nn * s
        total += 2 * nn * s + (n_sys - 1) * nn * s
        if s != 8:
            total += min(BLOCK_BYTES, 8 * nn)
        total += 8 * n * N_ONSETS * (-(-n // RHS_CHUNK))
        total += n_sys * 2 * n * N_ONSETS * s
        return int(total)
    is_half = (n_sys == 2) if half is None else bool(half)
    n_body = 4 if is_half else 2
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
    assembly: str = "parts",
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
    assembly : str, default="parts"
        Storage layout of the estimate (see :func:`required_bytes`).

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
        n_panels,
        dtype,
        n_systems=n_systems,
        half=half,
        ground=ground,
        body_cached=body_cached,
        assembly=assembly,
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
    The doublet pair kernel runs with one panel set, so no source
    potential is computed.

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
    torch_dtype = resolve_dtype(dtype)
    return doublet_pair_matrix_device(
        points, panels, None, device=device, dtype=torch_dtype, assembly_dtype=torch_dtype,
        far_ratio=far_ratio,
    )


def _check_pair_memory(dev, n_targets: int, n_sources: int, item: int, item_assembly: int) -> None:
    """Refuse a pair kernel call whose output does not fit in free device memory."""
    from ventorum.panel3d.kernels_warp import BLOCK_BYTES, RHS_CHUNK

    need = n_targets * n_sources * item
    if item_assembly != item:
        need += min(BLOCK_BYTES, n_targets * n_sources * item_assembly)
    need += 8 * n_targets * N_ONSETS * (-(-n_sources // RHS_CHUNK))
    free = free_device_bytes(dev)
    if need > free:
        raise MemoryError(
            f"The gpu panel backend needs about {need / 2**20:.1f} MiB for "
            f"the {n_targets} x {n_sources} pair matrix, but only "
            f"{free / 2**20:.1f} MiB is free on {dev}. Use a coarser mesh "
            "or a GPU with more memory."
        )


def doublet_pair_matrix_device(
    points,
    panels_a,
    panels_b,
    sign: float = 1.0,
    *,
    device=None,
    dtype=None,
    assembly_dtype=None,
    far_ratio: float = 5.0,
):
    """Compute ``C(points, panels_a) + sign * C(points, panels_b)`` on the device.

    Same results as :func:`ventorum.panel3d.kernels.doublet_pair_matrix`.
    The Warp kernel runs in the assembly precision (float64 by default)
    and the output is rounded to *dtype* block by block, so the device
    never holds the full matrix in both precisions.

    Parameters
    ----------
    points : numpy.ndarray
        Target coordinates with shape (P, 3) in m.
    panels_a : numpy.ndarray
        Panel corner coordinates with shape (M, 4, 3) in m.
    panels_b : numpy.ndarray or None
        Second panel set with shape (M, 4, 3) in m, or None.
    sign : float, default=1.0
        Factor of the second panel set [-] (+1 or -1).
    device : optional
        CUDA device. None selects the current CUDA device.
    dtype : optional
        Dtype of the output. None follows the Ventorum precision selection.
    assembly_dtype : optional
        Arithmetic of the kernel. None gives float64.
    far_ratio : float, default=5.0
        Distance-to-diameter ratio for point far-field transition.

    Returns
    -------
    torch.Tensor
        Doublet potential sum [-] on the CUDA device with shape (P, M).
    """
    import torch

    from ventorum.panel3d.kernels_warp import doublet_pair_matrix_warp

    dev = resolve_device(device)
    torch_dtype = resolve_dtype(dtype)
    kernel_dtype = torch.float64 if assembly_dtype is None else resolve_dtype(assembly_dtype)
    item = 4 if torch_dtype == torch.float32 else 8
    item_k = 4 if kernel_dtype == torch.float32 else 8
    _check_pair_memory(dev, int(np.asarray(points).shape[0]), int(np.asarray(panels_a).shape[0]),
                       item, item_k)
    return doublet_pair_matrix_warp(
        points, panels_a, panels_b, sign, far_ratio=far_ratio, device=dev, dtype=kernel_dtype,
        out_dtype=torch_dtype,
    )


def pair_system_device(
    points,
    panels_a,
    panels_b,
    sign: float,
    sigma,
    *,
    device=None,
    dtype=None,
    far_ratio: float = 5.0,
):
    """Compute one parity system on the device: the doublet pair matrix and the source sums.

    The Warp kernels run in float64 (the on-plane self terms need float64
    coordinates). The doublet matrix ``C_a + sign * C_b`` is rounded to
    *dtype* block by block; the source sums ``(B_a + sign * B_b) @ sigma``
    stay in float64. The source matrix is never stored.

    Parameters
    ----------
    points : numpy.ndarray
        Target coordinates with shape (P, 3) in m.
    panels_a : numpy.ndarray
        Panel corner coordinates with shape (M, 4, 3) in m.
    panels_b : numpy.ndarray or None
        Second panel set with shape (M, 4, 3) in m, or None.
    sign : float
        Factor of the second panel set [-] (+1 or -1).
    sigma : numpy.ndarray
        Source strengths with shape (M, K), K at most 6, in m/s.
    device : optional
        CUDA device. None selects the current CUDA device.
    dtype : optional
        Dtype of the doublet matrix. None follows the Ventorum precision
        selection.
    far_ratio : float, default=5.0
        Distance-to-diameter ratio for point far-field transition.

    Returns
    -------
    tuple of torch.Tensor
        (C, R) on the CUDA device: C with shape (P, M) [-] in *dtype* and
        R with shape (P, K) in m^2/s in float64.
    """
    import torch

    from ventorum.panel3d.kernels_warp import pair_system_warp

    dev = resolve_device(device)
    torch_dtype = resolve_dtype(dtype)
    item = 4 if torch_dtype == torch.float32 else 8
    _check_pair_memory(dev, int(np.asarray(points).shape[0]), int(np.asarray(panels_a).shape[0]),
                       item, 8)
    return pair_system_warp(
        points, panels_a, panels_b, sign, sigma, far_ratio=far_ratio, device=dev,
        dtype=torch.float64, out_dtype=torch_dtype,
    )


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


def _sum_batch_axis(values, axis):
    """Sum an axis with a fixed binary tree; output units equal input units.

    The addition order does not depend on the other dimensions, including
    the condition count. This keeps mesh-error reports bit-identical for
    scalar and batched studies.
    """
    import torch

    work = values.movedim(axis, -1)
    while work.shape[-1] > 1:
        count = work.shape[-1]
        paired = count - count % 2
        reduced = work[..., :paired:2] + work[..., 1:paired:2]
        work = torch.cat([reduced, work[..., -1:]], dim=-1) if count % 2 else reduced
    return work[..., 0]


def _evaluate_conditions_on_device(
    unit_mu, unit_sigma, unit_grad, coeffs, speed, rho, geometry, trefftz,
):
    """Evaluate a chunk on the device and copy the required outputs to the host.

    Unit fields have the units of :func:`evaluate_condition_on_device`.
    Coeffs has shape (K, 6), with translations in m/s and rates in rad/s.
    Speed and rho have shape (K,) in m/s and kg/m^3. Geometry holds cached
    full-mesh normals [-], areas [m^2], relative centroids [m] and the host
    unit doublets [m^2/s per unit onset] as device tensors. Trefftz holds
    velocity factors [1/m], strip normals [-], lengths [m], panel indices
    and the reference area [m^2], or None for a body without a wake.

    Return host float64 arrays: mu (K, N) [m^2/s], sigma (K, N) [m/s],
    velocity (K, Nf, 3) [m/s], Cp (K, Nf) [-], total force (K, 3) [N],
    total moment (K, 3) [N m] and Trefftz drag (K,) [-]. Pressure loads
    and Trefftz drag use float64, as in the old host post-processing.
    """
    import torch

    dev, dt = unit_mu.device, unit_mu.dtype
    c64 = torch.as_tensor(coeffs, dtype=torch.float64, device=dev)
    c = c64.to(dt)
    speeds = torch.as_tensor(speed, dtype=torch.float64, device=dev)
    densities = torch.as_tensor(rho, dtype=torch.float64, device=dev)
    # Keep the six-term multiply-add order of the old device matvec.
    # Loop over unit onsets, with all conditions and panels in each step.
    mu = torch.zeros((len(coeffs), unit_mu.shape[0]), dtype=dt, device=dev)
    sigma = torch.zeros_like(mu)
    for k in range(N_ONSETS):
        mu = torch.addcmul(mu, c[:, None, k], unit_mu[None, :, k])
        sigma = torch.addcmul(sigma, c[:, None, k], unit_sigma[None, :, k])
    if dt == torch.float32:
        # Match the reference contraction: four terms, then the two-term
        # tail. Round each multiply-add once to float32. Float64 holds
        # each float32 product and sum before this rounding step.
        gradient = torch.zeros((len(coeffs), unit_grad.shape[0], 3), dtype=dt, device=dev)
        for first, stop in ((0, 4), (4, N_ONSETS)):
            partial = torch.zeros_like(gradient)
            for k in range(first, stop):
                partial = (partial.to(torch.float64)
                           + c[:, None, None, k].to(torch.float64)
                           * unit_grad[None, :, k, :].to(torch.float64)).to(dt)
            gradient = gradient + partial
    else:
        gradient = torch.zeros((len(coeffs), unit_grad.shape[0], 3), dtype=dt, device=dev)
        for k in range(N_ONSETS):
            gradient = torch.addcmul(gradient, c[:, None, None, k], unit_grad[None, :, k, :])
    # Compute the onset in float64 before conversion, as in the old helper.
    omega = c64[:, 3:] * torch.tensor([-1.0, 1.0, -1.0], device=dev)
    onset = (c64[:, None, :3] - torch.linalg.cross(
        omega[:, None, :], geometry["arm"][None, :, :], dim=-1,
    )).to(dt)
    normals = geometry["normal"].to(dt)
    velocity = (onset - (onset * normals).sum(dim=-1, keepdim=True) * normals) - gradient
    valid = speeds > 1e-12
    # Tensor/scalar division in the reference uses a rounded reciprocal.
    # Tensor/tensor division rounds differently on CUDA.
    inverse_speed_sq = np.divide(
        1.0, speed**2, out=np.zeros_like(speed), where=speed > 1e-12,
    )
    inverse_speed_sq = torch.as_tensor(inverse_speed_sq, dtype=dt, device=dev)
    cp = torch.where(valid[:, None],
                     1.0 - (velocity**2).sum(dim=-1) * inverse_speed_sq[:, None], 0.0)
    q = 0.5 * densities * speeds**2
    force = -cp.to(torch.float64)[..., None] * geometry["area"][:, None]
    force = force * geometry["normal"] * q[:, None, None]
    moment = torch.linalg.cross(geometry["arm"][None, :, :], force, dim=-1)
    cdi = torch.zeros_like(speeds)
    if trefftz is not None:
        factors, up, lo, s_ref, ground = trefftz
        mu_full = mu.to(torch.float64)
        if geometry["half"]:
            host_mu = geometry["unit_mu"]
            symmetric = torch.zeros_like(mu_full)
            asymmetric = torch.zeros_like(mu_full)
            for k in (0, 2, 4):
                symmetric = torch.addcmul(symmetric, c64[:, None, k], host_mu[None, :, k])
            for k in (1, 3, 5):
                asymmetric = torch.addcmul(asymmetric, c64[:, None, k], host_mu[None, :, k])
            left = symmetric - asymmetric
            mu_full = torch.cat([mu_full, left], dim=1)
        gamma = mu_full[:, up] - mu_full[:, lo]
        gv = torch.cat([gamma, -gamma], dim=1)
        if ground:
            gv = torch.cat([gv, -gv], dim=1)
        vy, vz, ny, nz, length = factors
        wash = (_sum_batch_axis(gv[:, None, :] * vy[None], -1) * ny
                + _sum_batch_axis(gv[:, None, :] * vz[None], -1) * nz)
        drag = -0.5 * densities * _sum_batch_axis(gamma * wash * length, 1)
        if s_ref > 1e-12:
            cdi = torch.where(valid, drag / (q * s_ref), 0.0)
    return tuple(t.to(torch.float64).cpu().numpy() for t in (
        mu, sigma, velocity, cp, _sum_batch_axis(force, 1), _sum_batch_axis(moment, 1), cdi,
    ))
