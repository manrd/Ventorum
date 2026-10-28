# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Inviscid three-dimensional source-doublet panel solver for Ventorum-3DP.

This module provides the PanelSolver class and the PanelResult dataclass.
It solves the potential flow around arbitrary three-dimensional bodies
and lifting configurations with an internal Dirichlet boundary condition.
It supports symmetry folding about the y = 0 plane on half meshes and
ground effect using the method of images.

Published sources
-----------------
* Morino, L. and Kuo, C. C., 1974, Subsonic potential aerodynamics for
  complex configurations: A general theory, AIAA Journal, Vol. 12, No. 2,
  pp. 191-197.
* Maskew, B., 1987, Program VSAERO Theory Document, NASA CR-4023.
* Katz, J. and Plotkin, A., 2001, Low-Speed Aerodynamics, 2nd ed.,
  Cambridge University Press, section 9.4 and chapter 12.
* Wieselsberger, C., 1921, Ueber den Fluegelwiderstand in der Naehe des Bodens,
  Zeitschrift fuer Flugtechnik und Motorluftschiffahrt, Vol. 12, No. 10.
* Kinnas, S. A. and Hsin, C. Y., 1992, Boundary element method for the
  analysis of the unsteady flow around extreme propeller geometries,
  AIAA Journal, Vol. 30, No. 3 (iterative pressure Kutta).

Conventions
-----------
* Geometry axes: x aft [m], y right [m], z up [m].
* Flight dynamics body axes moments (AVL convention):
  Cl > 0 right wing down, Cm > 0 nose up, Cn > 0 nose right.
* Boundary condition: Zero total perturbation potential inside the body
  (internal Dirichlet boundary condition).
* Kutta condition: Constant doublet wake strips. Option "morino" sets
  mu_w = mu_upper - mu_lower. Option "pressure" adjusts the wake strengths
  to zero the trailing-edge pressure jump.

Configurations and factorizations
---------------------------------
A configuration is the wake and the image system of a set of conditions.
In free air the wake follows the body x axis, so one configuration (and one
factorization of each system) serves all free-air conditions. In ground
effect the wake follows the free stream, parallel to the ground, and the
ground images depend on the attitude and the height, so each (attitude,
height) group is one configuration with its own factorization. The solver
factorizes a system only when a condition needs it. On a half mesh the
symmetric system serves the symmetric unit onsets (u, w, q) and the
antisymmetric system the antisymmetric ones (v, p, r); the antisymmetric
system is factorized only when a condition has sideslip, roll rate or yaw
rate. The solver assembles only the systems that it factorizes: one pass
of the pair kernels of :mod:`ventorum.panel3d.kernels` gives the parity
matrix ``C_self + sign * C_mirror`` and the unit onset right-hand sides
``-(B_self + sign * B_mirror) @ sigma_unit``; the source matrix is never
stored. The free-air parity matrices of the body are kept (without the
wake) for the later configurations; a ground configuration adds its image
parity matrix ``C_image + sign * C_mirror_image`` and the wake strips. A
half mesh needs a flow configuration that is symmetric about
y = 0: in ground effect with sideslip (the wake follows the free stream)
or with bank (the ground normal has a y component) the solver solves the
full mesh instead. ``factorization_count`` counts the factorizations.

Kutta options
-------------
* "morino" (default): linear Morino condition.
* "pressure": Newton iteration on the wake strip strengths with residual
  Cp_upper - Cp_lower at each trailing-edge pair, starting from Morino.

Backends
--------
* "numpy" (default): reference kernels and SciPy dense solves on the CPU.
* "numba": compiled CPU kernels with the same formulas; SciPy solves.
* "gpu": Warp kernel assembly on the CUDA device, ``torch.linalg``
  factorization on the device, all unit onset right-hand sides in one
  ``lu_solve`` call, array post-processing on the device, results
  returned as numpy. ``dtype=None`` follows the Ventorum precision
  selection (float32 on consumer GPUs is accepted). Assembly runs in
  float64; the matrices are kept, factorized and solved in the requested
  dtype. The dense matrices must fit in the free device memory;
  otherwise the solve is refused with a clear error that reports the
  size. The gpu backend does no factorization on the CPU and keeps no
  copy of the matrices on the host.
"""

from __future__ import annotations

import dataclasses
import inspect
import warnings
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import scipy.linalg
import scipy.sparse.linalg

from ventorum.aero.system import freestream_direction, has_rates, rotation_velocity
from ventorum.core.datatypes import FlightCondition
from ventorum.core.errors import GroundStrikeError, ValidityError
from ventorum.panel3d.images import (
    check_ground_clearance,
    ground_plane_parameters,
    mirror_panels_y,
    reconstruct_full_mesh,
    reflect_panels_plane,
)
from ventorum.panel3d.kernels import (
    _precompute_geometry,
    doublet_potential,
    influence_matrices,
    pair_system,
)
from ventorum.panel3d.kutta import solve_pressure_kutta, strip_influence_matrix
from ventorum.panel3d.mesh import PanelMesh
from ventorum.panel3d.post import (
    _condition_arrays,
    _force_coefficients_batch,
    _integrate_batch,
    _surface_velocity_batch,
    _trefftz_batch,
    _trefftz_factors,
    build_surface_gradient_operator,
    compute_surface_velocity_and_cp,
    compute_trefftz_induced_drag,
    integrate_forces_and_moments,
    surface_gradient,
)
from ventorum.panel3d.wake import (
    DEFAULT_WAKE_CHORDS,
    PanelWake,
    build_wake,
    check_wake_crossing,
    largest_streamwise_chord,
)

__all__ = ["PanelResult", "PanelSolver"]

#: Unit onsets that are symmetric (u, w, q) and antisymmetric (v, p, r)
#: about the y = 0 plane.
_SYM_COLS = (0, 2, 4)
_ASYM_COLS = (1, 3, 5)

#: Tolerance on the y component of the ground normal and of the wake
#: direction for symmetry folding [-] (the same as
#: :func:`ventorum.aero.system.is_symmetric_condition`).
_SYMMETRY_TOL = 1e-12

#: Temporary array budget per host or device condition chunk [bytes].
_CONDITION_BATCH_BYTES = 256 * 2**20

#: Host strip budget of the ground attitude LRU cache [bytes].
_ATTITUDE_CACHE_BYTES = 512 * 2**20

#: Retained factor budgets as fractions of device free and host physical memory [-].
_FACTOR_BUDGET_DEVICE_FRACTION = 0.5
_FACTOR_BUDGET_HOST_FRACTION = 0.25
#: Maximum number of solved configurations, including free air [-].
_MAX_CONFIGURATIONS = 256

#: Ground GMRES relative true-residual limit [-] and inner iteration cap [-].
_GROUND_GMRES_RTOL = 1e-10
_GROUND_GMRES_MAXITER = 100
#: GMRES stops on its own (preconditioned) residual. This inner limit is a
#: decade below the true-residual limit, so that a converged solve passes the
#: final true-residual check instead of falling back to a direct factorization.
_GROUND_GMRES_INNER_RTOL = 0.1 * _GROUND_GMRES_RTOL


@dataclass(slots=True)
class PanelResult:
    """Aerodynamic result of the 3D source-doublet panel solver.

    Attributes
    ----------
    CL : float
        Lift coefficient in wind axes [-].
    CD_pressure : float
        Near-field pressure drag coefficient in wind axes [-].
    CDi_trefftz : float
        Far-field Trefftz plane induced drag coefficient [-].
    CY : float
        Side force coefficient in wind axes [-].
    Cl : float
        Rolling moment coefficient in body axes [-].
    Cm : float
        Pitching moment coefficient in body axes [-].
    Cn : float
        Yawing moment coefficient in body axes [-].
    cp : numpy.ndarray
        Surface pressure coefficients on body panels with shape (Np,) [-].
    mu : numpy.ndarray
        Doublet strengths on body panels with shape (Np,) [m^2/s].
    sigma : numpy.ndarray
        Source strengths on body panels with shape (Np,) [m/s].
    velocity : numpy.ndarray
        Surface velocity vectors with shape (Np, 3) [m/s].
    condition : FlightCondition
        The solved flight condition.
    kutta : str
        Kutta option used for this result ("morino" or "pressure").
    kutta_iters : int
        Number of pressure Kutta Newton updates performed.
    kutta_residual : float
        Final max |delta Cp| at the trailing-edge pairs [-].
    kutta_converged : bool
        True when the pressure residual falls below tolerance.
    trust_note : str
        Note on the trust of the result. Empty when converged.
    mu_w : numpy.ndarray or None
        Wake strip strengths on the solved mesh with shape (Ns,) in m^2/s.
    """

    CL: float
    CD_pressure: float
    CDi_trefftz: float
    CY: float
    Cl: float
    Cm: float
    Cn: float
    cp: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    velocity: np.ndarray
    condition: FlightCondition
    kutta: str = "morino"
    kutta_iters: int = 0
    kutta_residual: float = 0.0
    kutta_converged: bool = True
    trust_note: str = ""
    mu_w: np.ndarray | None = None


@dataclass
class _AttitudeWake:
    """Wake geometry in m and host float64 strips [-] shared between heights."""

    wake: PanelWake
    full_wake: PanelWake
    wake_panels_m: np.ndarray | None
    strips: dict = field(default_factory=dict)


@dataclass
class _Configuration:
    """Wake, images and solved systems of one set of conditions.

    Iteration counts [-] and relative true residuals [-] use unit-onset
    indices (u, v, w, p, q, r). Direct solves have zero iterations.
    Residuals are checked for eligible opt-in ground solves. Fallbacks
    keep the failed iterative attempt before the direct solve replaces it.
    Matrix passes [-] count lockstep products across all parity systems,
    including residual checks and failed attempts. Direct anchors use none.
    Wake sharing records a geometry hit in the ground attitude cache.
    A system value of None marks solved units whose matrix was dropped.
    """

    key: tuple
    wake_dir: np.ndarray | None
    ground: tuple[np.ndarray, float] | None
    wake: PanelWake
    full_wake: PanelWake
    wake_panels_m: np.ndarray | None
    systems: dict = field(default_factory=dict)
    unit_mu: np.ndarray | None = None
    unit_grad: np.ndarray | None = None
    unit_mu_t: object = None
    unit_grad_t: object = None
    strips: dict = field(default_factory=dict)
    trefftz_factors: object = None
    gpu_post: object = None
    ground_solver: str = "direct"
    ground_solver_reason: str = ""
    iterations: dict = field(default_factory=dict)
    relative_residuals: dict = field(default_factory=dict)
    fallbacks: list = field(default_factory=list)
    matrix_passes: int = 0
    wake_shared: bool = False
    ground_anchor: bool = False
    factor_use: int = 0
    viscous_units: np.ndarray | None = None
    viscous_wake: tuple | None = None


def _standardize_panels(panels: np.ndarray, nodes: np.ndarray | None = None) -> np.ndarray:
    """Standardize triangle panels to repeat the third vertex in the fourth position.

    Parameters
    ----------
    panels : numpy.ndarray
        Panel corner node indices with shape (Np, 4).
    nodes : numpy.ndarray or None, optional
        Mesh node coordinates with shape (Nn, 3) in m.

    Returns
    -------
    numpy.ndarray
        Standardized panel corner node indices with shape (Np, 4).
    """
    pans = np.asarray(panels, dtype=int).copy()
    shifts = np.zeros(len(pans), dtype=int)
    found = np.zeros(len(pans), dtype=bool)
    # Search shifts in order. Do not read coordinates after an index match
    # or after an earlier shift matched, as in the loop reference.
    for shift in range(4):
        a, b = (2 + shift) % 4, (3 + shift) % 4
        match = pans[:, a] == pans[:, b]
        if nodes is not None:
            active = ~found & ~match
            coords = np.asarray(nodes)
            match[active] = np.all(
                np.isclose(coords[pans[active, a]], coords[pans[active, b]], atol=1e-12),
                axis=-1,
            )
        choose = ~found & match
        shifts[choose] = shift
        found |= match
    order = (np.arange(4)[None, :] + shifts[:, None]) % 4
    return pans[np.arange(len(pans))[:, None], order]


def _standardize_panels_reference(panels: np.ndarray, nodes: np.ndarray | None = None) -> np.ndarray:
    """Provide the loop reference of the panel index parity tests.

    Parameters
    ----------
    panels : numpy.ndarray
        Panel corner node indices with shape (Np, 4).
    nodes : numpy.ndarray or None, optional
        Mesh node coordinates with shape (Nn, 3) in m.

    Returns
    -------
    numpy.ndarray
        Standardized panel corner node indices with shape (Np, 4).
    """
    pans = np.asarray(panels, dtype=int).copy()
    for i in range(len(pans)):
        row = pans[i]
        for shift in range(4):
            shifted = np.roll(row, -shift)
            if shifted[2] == shifted[3]:
                pans[i] = shifted
                break
            if nodes is not None and np.allclose(
                nodes[shifted[2]], nodes[shifted[3]], atol=1e-12
            ):
                pans[i] = shifted
                break
    return pans


def _fold_wake(mat_a, d_strip, wake: PanelWake):
    """Add the wake strip influences to the trailing-edge columns of A.

    Each wake strip carries mu_upper - mu_lower (Morino), so its summed
    potential column adds to the upper panel column and subtracts from the
    lower panel column. Works on numpy arrays and torch tensors.
    """
    if isinstance(mat_a, np.ndarray):
        np.add.at(mat_a.T, wake.wake_up, d_strip.T)
        np.subtract.at(mat_a.T, wake.wake_lo, d_strip.T)
    else:
        import torch

        up_idx = torch.as_tensor(wake.wake_up, dtype=torch.int64, device=mat_a.device)
        lo_idx = torch.as_tensor(wake.wake_lo, dtype=torch.int64, device=mat_a.device)
        mat_a.index_add_(1, up_idx, d_strip)
        mat_a.index_add_(1, lo_idx, -d_strip)
    return mat_a


def _as_ground(ground) -> tuple[np.ndarray, float] | None:
    """Return (unit normal, offset [m]) of a ground plane object or tuple, or None."""
    if ground is None:
        return None
    if hasattr(ground, "normal") and hasattr(ground, "offset"):
        normal, offset = ground.normal, ground.offset
    else:
        normal, offset = ground
    k = np.asarray(normal, dtype=float).reshape(3)
    k_norm = float(np.linalg.norm(k))
    if not np.isfinite(k_norm) or k_norm <= 0.0:
        raise ValueError(f"The ground normal must be a non-zero vector, got {normal}.")
    return k / k_norm, float(offset)


class PanelSolver:
    """Three-dimensional source-doublet potential flow panel solver.

    This solver uses constant-strength planar source and doublet panels with an
    internal Dirichlet boundary condition. It supports symmetry folding on half
    meshes (PanelMesh.half=True) and ground effect using the method of images.
    The construction builds the geometry only; each system is assembled
    and factorized when a condition first needs it (see the module notes).

    Parameters
    ----------
    mesh : PanelMesh
        Surface mesh containing nodes, panels, and wake shedding edges.
    wake_length : float or None, optional
        Length of the wake in m. If None, defaults to 50 times the largest
        streamwise chord of the mesh (see
        :func:`ventorum.panel3d.wake.largest_streamwise_chord`).
    n_wake : int, default=1
        Number of panels along each wake strip.
    far_ratio : float, default=5.0
        Distance-to-diameter ratio for point-singularity far-field transitions.
    S_ref : float or None, optional
        Reference area in m^2. If None, estimated from the mesh planform.
    b_ref : float or None, optional
        Reference span in m. If None, estimated from the mesh span.
    c_ref : float or None, optional
        Reference chord in m. If None, estimated as S_ref / b_ref.
    ref_point : numpy.ndarray or None, optional
        Moment reference center with shape (3,) in m. Default is the origin.
    kutta : str, default="morino"
        Kutta condition option: "morino" (linear, default) or "pressure"
        (iterative, removes the trailing-edge pressure jump).
    backend : str, default="numpy"
        Kernel backend: "numpy" (reference), "numba" (compiled CPU) or
        "gpu" (Warp assembly and torch solve on the CUDA device).
    device : optional
        CUDA device of the "gpu" backend (for example ``"cuda:0"``).
        None selects the current CUDA device. Other backends ignore it.
    dtype : optional
        Arithmetic of the "gpu" backend. None follows the Ventorum
        precision selection. Other backends ignore it.
    ground_solver : str, default="auto"
        Ground solve: "auto", "direct" or "krylov". Auto uses Krylov
        for CPU ground configurations with Morino Kutta and direct LU
        elsewhere. Krylov reuses an anchor LU per ground height [m].
        Direct unit onsets, including anchors, record zero iterations and
        a relative residual of None: direct LU, residual not computed.

    Attributes
    ----------
    factorization_count : int
        Number of dense LU factorizations done so far (CPU or device),
        including the factorizations of the full-mesh solve that a half
        mesh uses for asymmetric configurations in ground effect.
    wake_length : float
        Wake length in m.
    """

    def __init__(
        self,
        mesh: PanelMesh,
        *,
        wake_length: float | None = None,
        n_wake: int = 1,
        far_ratio: float = 5.0,
        S_ref: float | None = None,
        b_ref: float | None = None,
        c_ref: float | None = None,
        ref_point: np.ndarray | None = None,
        kutta: str = "morino",
        backend: str = "numpy",
        device=None,
        dtype=None,
        ground_solver: str = "auto",
        cpu_lu: str | None = None,
        _reuse_full_from: PanelSolver | None = None,
    ) -> None:
        if kutta not in ("morino", "pressure"):
            raise ValueError(f"Unknown kutta option {kutta!r}; use 'morino' or 'pressure'.")
        key = str(backend).strip().lower()
        if key not in ("numpy", "numba", "gpu"):
            raise ValueError(
                f"Unknown panel backend {backend!r}; use 'numpy', 'numba' or 'gpu'."
            )
        if int(n_wake) < 1:
            raise ValueError(f"n_wake must be at least 1, got {n_wake}.")
        if ground_solver not in ("auto", "direct", "krylov"):
            raise ValueError(f"Unknown ground_solver {ground_solver!r}; use 'auto', 'direct' or 'krylov'.")
        self.ground_solver = ground_solver
        self.backend = key
        self.kutta = str(kutta)
        if cpu_lu is None:
            from ventorum import gpu

            cpu_lu = gpu.cpu_lu()
        cpu_lu_key = str(cpu_lu).strip().lower()
        if cpu_lu_key not in ("scipy", "torch"):
            raise ValueError(f"Unknown cpu_lu option {cpu_lu!r}; use 'scipy' or 'torch'.")
        self.cpu_lu = cpu_lu_key
        self._cpu_lu = cpu_lu_key
        self.mesh = mesh
        self.far_ratio = float(far_ratio)
        self.n_wake = int(n_wake)
        self._device_arg = device
        self._dtype_arg = dtype
        self._own_factorizations = 0
        self._full_solver: PanelSolver | None = None

        # Standardize triangular panels on given mesh
        if _reuse_full_from is None:
            self.standardized_panels = _standardize_panels(mesh.panels, mesh.nodes)
        else:
            self.standardized_panels = _reuse_full_from._standardized_full_panels
        self.body_verts = mesh.nodes[self.standardized_panels]

        # Precompute body geometry on given mesh
        geom = (
            _precompute_geometry(self.body_verts)
            if _reuse_full_from is None else _reuse_full_from._full_geom
        )
        self.centroids = geom.centroid
        self.normals = geom.normal
        self.areas = geom.area

        # Wake length: one value for the solved mesh, the full mesh and
        # every configuration.
        if wake_length is None:
            chord = largest_streamwise_chord(
                mesh.nodes, self.standardized_panels, mesh.wake_edges, mesh.component
            )
            self.wake_length = DEFAULT_WAKE_CHORDS * max(chord, 1.0e-3)
        else:
            self.wake_length = float(wake_length)
            if not np.isfinite(self.wake_length) or self.wake_length <= 0.0:
                raise ValueError(f"wake_length must be positive and finite, got {wake_length}.")

        # Body-axis wake of free air on the solved mesh
        self.wake: PanelWake = (
            self._build_wake(
                mesh.nodes, self.standardized_panels, mesh.wake_edges, self.normals, None
            )
            if _reuse_full_from is None else _reuse_full_from._full_wake
        )

        # Reconstruct full mesh for post-processing and reference geometry if half mesh
        if mesh.half:
            self._full_mesh = reconstruct_full_mesh(mesh)
        else:
            self._full_mesh = mesh

        # Full mesh geometry and wake for global post-processing
        if _reuse_full_from is None:
            self._standardized_full_panels = _standardize_panels(
                self._full_mesh.panels, self._full_mesh.nodes
            )
            self._full_geom = _precompute_geometry(
                self._full_mesh.nodes[self._standardized_full_panels]
            )
            self._full_wake = self._build_wake(
                self._full_mesh.nodes,
                self._standardized_full_panels,
                self._full_mesh.wake_edges,
                self._full_geom.normal,
                None,
            )
            self._grad_op = build_surface_gradient_operator(
                self._full_mesh.nodes,
                self._standardized_full_panels,
                self._full_geom.centroid,
                self._full_geom.normal,
                self._full_geom.area,
                self._full_mesh.wake_edges,
                kinds=self._full_mesh.kind,
            )
        else:
            # The parent already prepared these full-mesh pieces.
            self._standardized_full_panels = _reuse_full_from._standardized_full_panels
            self._full_geom = _reuse_full_from._full_geom
            self._full_wake = _reuse_full_from._full_wake
            self._grad_op = _reuse_full_from._grad_op

        # Reference center
        if ref_point is None:
            self.ref_point = np.zeros(3, dtype=np.float64)
        else:
            self.ref_point = np.asarray(ref_point, dtype=np.float64).reshape(3)

        # Reference geometry: estimated from full aircraft configuration
        if S_ref is not None:
            self.S_ref = float(S_ref)
        else:
            s_proj = float(
                np.sum(np.maximum(0.0, self._full_geom.normal[:, 2]) * self._full_geom.area)
            )
            if s_proj > 1e-6:
                self.S_ref = s_proj
            else:
                y_span = float(
                    np.max(self._full_mesh.nodes[:, 1]) - np.min(self._full_mesh.nodes[:, 1])
                )
                x_chord = float(
                    np.max(self._full_mesh.nodes[:, 0]) - np.min(self._full_mesh.nodes[:, 0])
                )
                self.S_ref = max(np.pi * 0.25 * y_span * x_chord, 1e-6)

        if b_ref is not None:
            self.b_ref = float(b_ref)
        else:
            y_span = float(
                np.max(self._full_mesh.nodes[:, 1]) - np.min(self._full_mesh.nodes[:, 1])
            )
            self.b_ref = max(y_span, 1e-3)

        if c_ref is not None:
            self.c_ref = float(c_ref)
        else:
            self.c_ref = self.S_ref / self.b_ref

        # Unit onset flow definitions: 6 onsets (u, v, w, p, q, r)
        n_panels = len(self.centroids)
        r_rel = self.centroids - self.ref_point[None, :]
        rx = r_rel[:, 0]
        ry = r_rel[:, 1]
        rz = r_rel[:, 2]

        v_unit = np.zeros((6, n_panels, 3), dtype=np.float64)
        v_unit[0, :, 0] = 1.0
        v_unit[1, :, 1] = 1.0
        v_unit[2, :, 2] = 1.0
        v_unit[3, :, 1] = -rz
        v_unit[3, :, 2] = ry
        v_unit[4, :, 0] = -rz
        v_unit[4, :, 2] = rx
        v_unit[5, :, 0] = -ry
        v_unit[5, :, 1] = rx
        self._v_unit = v_unit

        # Unit onset source strengths on body panels: sigma_j = -n_j . V_onset
        self._unit_sigma = -np.sum(self.normals[None, :, :] * v_unit, axis=-1).T

        # Mirrored body geometry for the half mesh (backend independent)
        if self.mesh.half:
            self._body_verts_m = mirror_panels_y(self.body_verts)

        # Device and precision of the gpu backend (no assembly here)
        self._body_parts: dict | None = None
        self._configs: OrderedDict[tuple, _Configuration] = OrderedDict()
        self._attitude_cache: OrderedDict[tuple, _AttitudeWake] = OrderedDict()
        self._factor_budget_host = float("inf")
        self._factor_budget_device = float("inf")
        self._factor_clock = 0
        self._protected_factors: set[tuple] = set()
        self._factors_dropped = 0
        self._configurations_evicted = 0
        self._viscous_strips = None
        self._viscous_components = (0, 2)
        self._viscous_evaluator = None
        self._viscous_solve_count = 0
        self._viscous_rhs_count = 0
        if self.backend == "gpu":
            import torch

            from ventorum.panel3d import gpu as gpu_mod

            self._gpu_device = gpu_mod.resolve_device(device)
            self._gpu_dtype = gpu_mod.resolve_dtype(dtype)
            self._gpu_unit_sigma = torch.as_tensor(
                np.asarray(self._unit_sigma, dtype=np.float64),
                dtype=self._gpu_dtype,
                device=self._gpu_device,
            )

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    @property
    def factorization_count(self) -> int:
        """Number of dense LU factorizations done so far [-]."""
        sub = self._full_solver.factorization_count if self._full_solver is not None else 0
        return int(self._own_factorizations + sub)

    @staticmethod
    def _factor_bytes(cfg: _Configuration) -> tuple[int, int]:
        """Return bytes held by factors and optional viscous responses [bytes]."""
        host = device = 0
        for factors in cfg.systems.values():
            for array in factors or ():
                if isinstance(array, np.ndarray):
                    host += int(array.nbytes)
                elif array.device.type == "cpu":
                    host += array.numel() * array.element_size()
                else:
                    device += array.numel() * array.element_size()
        arrays = [cfg.viscous_units, *(cfg.viscous_wake or ())]
        seen = set()
        for array in arrays:
            if isinstance(array, np.ndarray) and id(array) not in seen:
                host += int(array.nbytes)
                seen.add(id(array))
        return host, device

    def _read_factor_budgets(self) -> None:
        """Read the factor pool limits [bytes] at the start of a solve call."""
        from ventorum.hardware.detector import _get_total_ram_gb

        self._factor_budget_host = int(_FACTOR_BUDGET_HOST_FRACTION * _get_total_ram_gb() * 2**30)
        if self.backend == "gpu":
            import torch

            free, _ = torch.cuda.mem_get_info(self._gpu_device)
            held = sum(self._factor_bytes(c)[1] for c in self._configs.values())
            self._factor_budget_device = int(_FACTOR_BUDGET_DEVICE_FRACTION * free) + held

    def _touch_factors(self, cfg: _Configuration) -> None:
        """Record the most recent group or preconditioner use of factors."""
        self._factor_clock += 1
        cfg.factor_use = self._factor_clock

    def _drop_factors(self, cfg: _Configuration) -> None:
        """Drop each retained parity factor and release the anchor role."""
        for parity, factors in cfg.systems.items():
            if factors is not None:
                cfg.systems[parity] = None
                self._factors_dropped += 1
        cfg.ground_anchor = False
        if self._viscous_strips is not None:
            cfg.viscous_units = None
            cfg.viscous_wake = None
            cfg.systems.clear()

    def _trim_factors(self, incoming: tuple[int, int] = (0, 0)) -> None:
        """Make room for incoming host and device factors [bytes], in LRU order.

        The running group and its anchor can exceed a budget. Trim again
        after the group releases them, so a zero budget still permits a solve.
        """
        pool = [(cfg, self._factor_bytes(cfg)) for cfg in self._configs.values()]
        host = incoming[0] + sum(size[0] for _, size in pool)
        if self._viscous_strips is not None:
            # The free-air source sums belong to the shared body geometry.
            columns = len(self._viscous_components) * len(self._viscous_strips)
            host += sum(a[:, -columns:].nbytes
                        for key, a in (self._body_parts or {}).items() if key.startswith("rhs_"))
        device = incoming[1] + sum(size[1] for _, size in pool)
        for cfg, size in sorted(pool, key=lambda item: item[0].factor_use):
            if host <= self._factor_budget_host and device <= self._factor_budget_device:
                break
            if cfg.key in self._protected_factors or size == (0, 0):
                continue
            self._drop_factors(cfg)
            host -= size[0]
            device -= size[1]

    def _bound_configurations(self) -> None:
        """Evict the least recently used configurations at the start of a call.

        The configurations of the running call stay until the next call:
        the factory reads their dispatch records after the call.
        """
        while len(self._configs) > _MAX_CONFIGURATIONS:
            _, cfg = self._configs.popitem(last=False)
            self._drop_factors(cfg)
            self._configurations_evicted += 1

    def _build_wake(self, nodes, panels, wake_edges, normals, wake_dir) -> PanelWake:
        """Build one wake with the wake length and panel count of the solver."""
        return build_wake(
            nodes,
            panels,
            wake_edges,
            normals,
            wake_length=self.wake_length,
            n_wake=self.n_wake,
            wake_dir=wake_dir,
        )

    def _factorize(self, mat_a):
        """Factorize one dense system (the only place of a factorization).

        The solver passes its own work copy of the system, so the CPU
        factorization overwrites it (no second copy of the matrix).
        """
        self._own_factorizations += 1
        if self.backend == "gpu":
            from ventorum.panel3d import gpu as gpu_mod

            return gpu_mod.lu_factor_device(mat_a)
        if self.cpu_lu == "torch":
            import torch

            if isinstance(mat_a, torch.Tensor):
                t = mat_a
            else:
                try:
                    t = torch.from_numpy(mat_a)
                except ValueError:
                    t = torch.as_tensor(mat_a, dtype=torch.float64, device="cpu")
            piv = torch.empty(t.shape[0], dtype=torch.int32, device=t.device)
            torch.linalg.lu_factor(t, out=(t, piv))
            return (t, piv)
        return scipy.linalg.lu_factor(mat_a, overwrite_a=True)

    def _lu_solve(self, lu_info, rhs):
        """Solve with stored LU factors (numpy or torch right-hand sides)."""
        import torch

        if isinstance(lu_info[0], torch.Tensor):
            lu, piv = lu_info
            if isinstance(rhs, np.ndarray):
                is_1d = rhs.ndim == 1
                rhs_arr = rhs if rhs.flags.c_contiguous else np.ascontiguousarray(rhs)
                rhs_t = torch.from_numpy(rhs_arr)
                if is_1d:
                    rhs_t = rhs_t.unsqueeze(-1)
                if lu.device.type != "cpu" or rhs_t.dtype != lu.dtype:
                    rhs_t = rhs_t.to(device=lu.device, dtype=lu.dtype)
                sol_t = torch.linalg.lu_solve(lu, piv, rhs_t)
                if lu.device.type != "cpu":
                    sol_t = sol_t.to(torch.float64).cpu()
                return sol_t.squeeze(-1).numpy() if is_1d else sol_t.numpy()
            is_1d = rhs.ndim == 1
            rhs_t = rhs.unsqueeze(-1) if is_1d else rhs
            if rhs_t.device != lu.device or rhs_t.dtype != lu.dtype:
                rhs_t = rhs_t.to(device=lu.device, dtype=lu.dtype)
            sol_t = torch.linalg.lu_solve(lu, piv, rhs_t)
            return sol_t.squeeze(-1) if is_1d else sol_t
        return scipy.linalg.lu_solve(lu_info, rhs)

    def _solve_host(self, cfg: _Configuration, parity: str, rhs: np.ndarray) -> np.ndarray:
        """Solve one stored system for host right-hand sides; return float64 numpy."""
        lu_info = cfg.systems[parity]
        if self.backend != "gpu":
            if isinstance(lu_info[0], np.ndarray):
                return scipy.linalg.lu_solve(lu_info, rhs)
            return self._lu_solve(lu_info, rhs)
        import torch

        rhs_t = torch.as_tensor(
            np.ascontiguousarray(rhs, dtype=np.float64),
            dtype=self._gpu_dtype,
            device=self._gpu_device,
        )
        squeeze = rhs_t.ndim == 1
        if squeeze:
            rhs_t = rhs_t.reshape(-1, 1)
        sol = self._lu_solve(lu_info, rhs_t).to(torch.float64).cpu().numpy()
        return sol.reshape(-1) if squeeze else sol

    def _matrices(
        self, collocation: np.ndarray, panels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (B, C) with the selected backend (device tensors for gpu)."""
        if self.backend == "gpu":
            import torch

            from ventorum.panel3d import gpu as gpu_mod

            mat_b, mat_c = gpu_mod.influence_matrices_device(
                collocation,
                panels,
                device=self._gpu_device,
                dtype=torch.float64,
                far_ratio=self.far_ratio,
            )
            if self._gpu_dtype != torch.float64:
                mat_b = mat_b.to(self._gpu_dtype)
                mat_c = mat_c.to(self._gpu_dtype)
            return mat_b, mat_c
        if self.backend == "numba":
            from ventorum.panel3d.kernels_numba import influence_matrices_numba

            return influence_matrices_numba(
                np.asarray(collocation),
                np.asarray(panels),
                far_ratio=self.far_ratio,
            )
        return influence_matrices(
            collocation, panels, far_ratio=self.far_ratio
        )

    def _pair_system(self, panels_a: np.ndarray, panels_b: np.ndarray | None, sign: float, cols) -> tuple:
        """Return one parity system at the collocation points with the selected backend.

        Returns ``(C_a + sign * C_b, (B_a + sign * B_b) @ sigma_unit[:, cols])``:
        the doublet matrix with shape (N, N) [-] and the source sums with
        shape (N, K) in m^2/s per unit onset. The gpu backend returns
        device tensors: the matrix in the solve dtype and the sums in
        float64.
        """
        sigma = np.ascontiguousarray(self._unit_sigma[:, list(cols)])
        if self._viscous_strips is not None:
            return self._viscous_pair_system(panels_a, panels_b, sign, sigma)
        if self.backend == "gpu":
            from ventorum.panel3d import gpu as gpu_mod

            return gpu_mod.pair_system_device(
                self.centroids,
                panels_a,
                panels_b,
                sign,
                sigma,
                device=self._gpu_device,
                dtype=self._gpu_dtype,
                far_ratio=self.far_ratio,
            )
        if self.backend == "numba":
            from ventorum.panel3d.kernels_numba import pair_system_numba

            return pair_system_numba(
                self.centroids, panels_a, panels_b, sign, sigma, far_ratio=self.far_ratio
            )
        return pair_system(self.centroids, panels_a, panels_b, sign, sigma, far_ratio=self.far_ratio)

    def _set_viscous_strips(self, strips, lateral: bool = False) -> None:
        """Enable strip source sums before assembly (panel indices, no units).

        The strip source sums hold the x and z unit onsets. With *lateral*
        they also hold the y unit onset, for strips whose span axis is not
        the y axis (dihedral, fins): their section angle turns the onset
        about another axis.
        """
        if self.backend == "gpu":
            raise ValueError("Viscous panel coupling requires a CPU backend; GPU is not supported.")
        if self.ground_solver == "krylov":
            raise ValueError("Viscous panel coupling requires one direct LU per configuration.")
        if self._configs or self._body_parts is not None:
            raise ValueError("Set viscous strips before the first panel solve.")
        maps = [np.asarray(ids, dtype=int) for ids in strips]
        if not maps or not np.array_equal(np.sort(np.concatenate(maps)), np.arange(len(self.centroids))):
            raise ValueError("Viscous strips must partition every body panel exactly once.")
        self._viscous_strips = maps
        self._viscous_components = (0, 2, 1) if lateral else (0, 2)

    def _viscous_pair_system(self, panels_a, panels_b, sign, sigma):
        """Accumulate unit and strip source sums from each bounded kernel chunk.

        Each source/doublet pair is evaluated once. The source matrix is
        discarded after its target chunk. The strip column blocks hold the
        x, z and (for lateral strips) y unit source sums (m^2/s per unit
        onset).
        """
        from ventorum.panel3d.kernels import _CHUNK_BUDGET

        n = len(self.centroids)
        ns = len(self._viscous_strips)
        k = sigma.shape[1]
        matrix = np.empty((n, n))
        components = self._viscous_components
        rhs = np.empty((n, k + len(components) * ns))
        size = max(1, _CHUNK_BUDGET // n)
        for start in range(0, n, size):
            stop = min(n, start + size)
            b, c = self._matrices(self.centroids[start:stop], panels_a)
            if panels_b is not None:
                bm, cm = self._matrices(self.centroids[start:stop], panels_b)
                b += sign * bm
                c += sign * cm
                del bm, cm
            matrix[start:stop] = c
            rhs[start:stop, :k] = b @ sigma
            for j, ids in enumerate(self._viscous_strips):
                for block, component in enumerate(components):
                    rhs[start:stop, k + block * ns + j] = b[:, ids] @ self._unit_sigma[ids, component]
        return matrix, rhs

    def _viscous_response(self, cfg, condition, directions=None):
        """Return body doublet sensitivity (m^2/s per rad) to strip corrections.

        A correction of strip k changes the onset of its panels by
        ``-V u_k``. The source change is ``V n dot u_k``. With
        ``U = -A^-1 R`` per unit onset component,
        ``G_k = -V (u_kx U_x + u_ky U_y + u_kz U_z)``. Without *directions*,
        ``u_k = cos(beta) [-sin(alpha), 0, cos(alpha)]`` for every strip, so
        ``G = V cos(beta) (sin(alpha) U_x - cos(alpha) U_z)``. No angle needs
        an LU solve.

        Parameters
        ----------
        cfg : _Configuration
            Configuration with the strip responses ``viscous_units``.
        condition : FlightCondition
            Flight condition (speed V_inf [m/s], alpha and beta [rad]).
        directions : numpy.ndarray or None
            Onset change per unit strip angle ``u_k`` [1/rad], shape
            (n_strips, 3). A y component needs lateral strip responses.

        Returns
        -------
        numpy.ndarray
            Doublet change per unit strip angle [m^2/s per rad], shape
            (n_panels, n_strips).
        """
        ns = len(self._viscous_strips)
        units = cfg.viscous_units
        if directions is None:
            return float(condition.V_inf) * np.cos(condition.beta) * (
                np.sin(condition.alpha) * units[:, :ns]
                - np.cos(condition.alpha) * units[:, ns:2 * ns]
            )
        directions = np.asarray(directions, dtype=float)
        response = units[:, :ns] * directions[:, 0] + units[:, ns:2 * ns] * directions[:, 2]
        if len(self._viscous_components) > 2:
            response = response + units[:, 2 * ns:3 * ns] * directions[:, 1]
        elif np.any(directions[:, 1] != 0.0):
            raise ValueError("Strip corrections with a y onset need lateral strip responses.")
        return -float(condition.V_inf) * response

    def _viscous_pressure_response(self, cfg, coeffs):
        """Cache the wake response once per configuration (m^2/s per m^2/s)."""
        if self._viscous_strips is not None and self.mesh.half and np.any(coeffs[list(_ASYM_COLS)] != 0.0):
            raise ValueError(
                "Viscous panel coupling does not support asymmetric flow on a half mesh; use a full mesh."
            )
        if cfg.viscous_wake is None:
            cfg.viscous_wake = self._pressure_sensitivity(cfg, coeffs)
            self._viscous_solve_count += 1
            self._viscous_rhs_count += cfg.viscous_wake[0].shape[1]
        return cfg.viscous_wake

    def _doublet(self, points: np.ndarray, panels: np.ndarray) -> np.ndarray:
        """Return the doublet potentials with the selected CPU backend."""
        if self.backend == "numba":
            from ventorum.panel3d.kernels_numba import doublet_pair_matrix_numba

            return doublet_pair_matrix_numba(
                np.asarray(points),
                np.asarray(panels),
                None,
                far_ratio=self.far_ratio,
            )
        return doublet_potential(points, panels, far_ratio=self.far_ratio)

    def _strip_matrix(self, wake_panels: np.ndarray, wake: PanelWake) -> tuple:
        """Return the summed wake strip potentials (host float64, backend array)."""
        strips = wake.strip_panel_indices
        n_solve = len(self.centroids)
        if len(strips) == 0:
            host = np.zeros((n_solve, 0), dtype=float)
            if self.backend == "gpu":
                import torch

                return host, torch.zeros((n_solve, 0), dtype=self._gpu_dtype, device=self._gpu_device)
            return host, host
        if self.backend == "gpu":
            import torch

            from ventorum.panel3d import gpu as gpu_mod

            mat_cw = gpu_mod.doublet_potential_device(
                self.centroids,
                wake_panels,
                device=self._gpu_device,
                dtype=torch.float64,
                far_ratio=self.far_ratio,
            )
            d64 = gpu_mod.strip_sum(mat_cw, strips)
            del mat_cw
            host = d64.cpu().numpy()
            return host, d64.to(self._gpu_dtype)
        host = strip_influence_matrix(self._doublet(self.centroids, wake_panels), strips)
        return host, host

    def _copy(self, mat):
        """Return a work copy of a numpy array (Fortran order) or torch tensor.

        The Fortran order lets the CPU factorization overwrite the copy.
        """
        if self.backend == "gpu":
            return mat.clone()
        if self.backend == "numba":
            from ventorum.panel3d.kernels_numba import copy_to_fortran_numba

            return copy_to_fortran_numba(mat)
        return np.array(mat, order="F", copy=True)

    def _add(self, mat_a, mat_b):
        """Return the sum of two matrices as a new work matrix (see :meth:`_copy`)."""
        if self.backend == "gpu":
            return mat_a + mat_b
        if self.backend == "numba":
            from ventorum.panel3d.kernels_numba import add_to_fortran_numba

            return add_to_fortran_numba(mat_a, mat_b)
        return np.add(mat_a, mat_b, order="F")

    def _check_gpu_memory(self, half: bool, ground: bool, n_systems: int, body_cached: bool) -> None:
        """Refuse a configuration that does not fit in the device memory."""
        if self.backend != "gpu":
            return
        from ventorum.panel3d import gpu as gpu_mod

        gpu_mod.check_device_memory(
            len(self.centroids),
            self._gpu_dtype,
            self._gpu_device,
            n_systems=n_systems,
            half=half,
            ground=ground,
            body_cached=body_cached,
            assembly="fused",
        )

    @staticmethod
    def _parity_sign_cols(parity: str) -> tuple[float, tuple[int, ...]]:
        """Return the mirror sign [-] and the unit onset columns of one system."""
        if parity == "full":
            return 1.0, tuple(range(6))
        if parity == "sym":
            return 1.0, _SYM_COLS
        return -1.0, _ASYM_COLS

    def _body_system(self, parity: str) -> tuple:
        """Return the free-air body system of one parity (computed once, then kept).

        Returns ``(A, R)``: ``A = C_self + sign * C_mirror`` without the
        wake (``C_self`` on a full mesh) and ``R = (B_self + sign *
        B_mirror) @ sigma_unit`` for the unit onset columns of the system.
        """
        parts = self._body_parts if self._body_parts is not None else {}
        if f"A_{parity}" not in parts:
            sign, cols = self._parity_sign_cols(parity)
            panels_m = self._body_verts_m if self.mesh.half else None
            mat_a, rhs = self._pair_system(self.body_verts, panels_m, sign, cols)
            parts[f"A_{parity}"] = mat_a
            parts[f"rhs_{parity}"] = rhs
            self._body_parts = parts
        return parts[f"A_{parity}"], parts[f"rhs_{parity}"]

    # ------------------------------------------------------------------
    # Free-air matrices on the host (inspection and tests)
    # ------------------------------------------------------------------

    def _free_matrices_host(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the free-air full-mesh system (A, B) on the host."""
        if self.mesh.half:
            raise AttributeError(
                "A and B are defined for a full mesh; a half mesh has a symmetric "
                "and an antisymmetric system."
            )
        # The solver keeps no source matrix, so the separate kernels give B and C.
        mat_b, mat_a = self._matrices(self.centroids, self.body_verts)
        if len(self.wake.panels) > 0:
            _, d_dev = self._strip_matrix(self.wake.panels, self.wake)
            _fold_wake(mat_a, d_dev, self.wake)
        if self.backend == "gpu":
            import torch

            return (
                mat_a.to(torch.float64).cpu().numpy(),
                mat_b.to(torch.float64).cpu().numpy(),
            )
        return mat_a, mat_b

    @property
    def A(self) -> np.ndarray:
        """Free-air doublet system matrix of a full mesh with shape (N, N) [-]."""
        return self._free_matrices_host()[0]

    @property
    def B(self) -> np.ndarray:
        """Source influence matrix of a full mesh with shape (N, N) in m."""
        return self._free_matrices_host()[1]

    # ------------------------------------------------------------------
    # Configurations
    # ------------------------------------------------------------------

    def _ground_of(self, condition: FlightCondition, ground) -> tuple[np.ndarray, float] | None:
        """Return the ground plane of one condition: explicit plane, else condition.h."""
        if ground is not None:
            return _as_ground(ground)
        gp = ground_plane_parameters(condition, self.ref_point)
        if gp is None:
            return None
        return np.asarray(gp[0], dtype=float), float(gp[1])

    @staticmethod
    def _ground_wake_dir(condition: FlightCondition, normal: np.ndarray) -> np.ndarray:
        """Return the free-stream direction projected on the ground plane.

        The ground of ``condition.h`` is parallel to the free stream, so the
        projection keeps the free-stream direction. An explicit ground plane
        that is not parallel to the free stream gets the wake along the
        projection of the free stream on the ground.
        """
        d = freestream_direction(float(condition.alpha), float(condition.beta))
        dot = float(np.dot(d, normal))
        if abs(dot) <= 1.0e-12:
            return d
        d_t = d - dot * normal
        n_t = float(np.linalg.norm(d_t))
        if n_t <= 1.0e-9:
            raise ValueError("The free stream is normal to the ground plane; no wake direction is defined.")
        return d_t / n_t

    def _config_key(self, condition: FlightCondition, ground) -> tuple:
        """Return the configuration key and its data for one condition."""
        gp = self._ground_of(condition, ground)
        if gp is None:
            return ("free",), None, None
        normal, offset = gp
        wake_dir = self._ground_wake_dir(condition, normal)
        key = (
            "ground",
            tuple(np.round(normal, 12).tolist()),
            round(offset, 12),
            tuple(np.round(wake_dir, 12).tolist()),
        )
        return key, wake_dir, (normal, offset)

    def _symmetric_config(self, wake_dir, ground) -> bool:
        """Return True if a configuration is symmetric about y = 0."""
        if wake_dir is not None and abs(float(wake_dir[1])) > _SYMMETRY_TOL:
            return False
        if ground is not None and abs(float(ground[0][1])) > _SYMMETRY_TOL:
            return False
        return True

    def _check_wake_crossing(self, full_wake: PanelWake) -> None:
        """Refuse a wake sheet that touches or crosses another closed surface.

        The rigid flat wake of constant doublet strength makes the
        perturbation potential jump inside a closed surface that it
        crosses, and the internal Dirichlet condition then has no solution
        (see :func:`ventorum.panel3d.wake.check_wake_crossing`). A tandem
        wing at the height of the front wing, 2 m behind it, gave CL of
        about -2.5 at alpha 4 deg instead of about 0.54.

        Parameters
        ----------
        full_wake : PanelWake
            Wake of the full mesh of the configuration.

        Raises
        ------
        ValidityError
            If a wake sheet touches or crosses another component.
        """
        crossing = check_wake_crossing(
            self._full_mesh.nodes,
            self._standardized_full_panels,
            self._full_mesh.component,
            full_wake,
        )
        if not crossing["crossing_pairs"]:
            return
        src, dst, n_edges = crossing["crossing_pairs"][0]
        raise ValidityError(
            f"The wake of 3D panel component {src} touches or passes through component {dst} "
            f"({n_edges} panel edges of component {dst} touch or cross the wake sheet; the "
            "component index is the order of the surfaces in the aircraft). The 3D panel "
            "method sheds a rigid flat wake (along the body x axis in free air, along the free "
            "stream in ground effect). A wake sheet inside a closed surface makes the potential "
            "jump inside that surface, and the loads are then wrong. Move the surface above or "
            "below the wake of the surface in front of it."
        )

    def _new_config(self, key, wake_dir, ground) -> _Configuration:
        """Build the wake geometry of a configuration (no matrices yet)."""
        attitude_key = (key[1], key[3]) if ground is not None else None
        shared = self._attitude_cache.get(attitude_key)
        if shared is not None:
            self._attitude_cache.move_to_end(attitude_key)
            wake, full_wake = shared.wake, shared.full_wake
        elif wake_dir is None:
            wake = self.wake
            full_wake = self._full_wake
        else:
            wake = self._build_wake(
                self.mesh.nodes, self.standardized_panels, self.mesh.wake_edges, self.normals, wake_dir
            )
            full_wake = self._build_wake(
                self._full_mesh.nodes,
                self._standardized_full_panels,
                self._full_mesh.wake_edges,
                self._full_geom.normal,
                wake_dir,
            )
        if shared is None:
            self._check_wake_crossing(full_wake)
        if ground is not None:
            check_ground_clearance(self._full_mesh.nodes, ground[0], ground[1])
            if len(full_wake.panels) > 0:
                try:
                    check_ground_clearance(full_wake.panels.reshape(-1, 3), ground[0], ground[1])
                except GroundStrikeError as exc:
                    raise GroundStrikeError(f"The wake goes below the ground plane: {exc}") from None
        wake_m = shared.wake_panels_m if shared is not None else None
        if self.mesh.half and shared is None:
            wake_m = (
                mirror_panels_y(wake.panels)
                if len(wake.panels) > 0
                else np.empty((0, 4, 3), dtype=float)
            )
        if ground is not None and shared is None:
            # The strip shapes are known before assembly. Do not retain
            # geometry for an entry whose strips cannot fit by themselves.
            strip_bytes = len(self.centroids) * len(wake.wake_up) * 8 * (2 if self.mesh.half else 1)
            if strip_bytes <= _ATTITUDE_CACHE_BYTES:
                self._attitude_cache[attitude_key] = _AttitudeWake(wake, full_wake, wake_m)
        return _Configuration(
            key=key,
            wake_dir=wake_dir,
            ground=ground,
            wake=wake,
            full_wake=full_wake,
            wake_panels_m=wake_m,
            wake_shared=shared is not None,
            ground_solver_reason=(
                "auto: direct on gpu backend" if self.ground_solver == "auto" and self.backend == "gpu"
                else "auto: direct with pressure Kutta" if self.ground_solver == "auto" and self.kutta == "pressure"
                else "auto: direct in free air" if self.ground_solver == "auto" and ground is None
                else "krylov not used: gpu backend" if self.ground_solver == "krylov" and self.backend == "gpu"
                else "krylov not used: pressure Kutta" if self.ground_solver == "krylov" and self.kutta == "pressure"
                else "krylov not used: free air" if self.ground_solver == "krylov" and ground is None
                else ""
            ),
        )

    def _parities_needed(self, coeff_rows: list[np.ndarray]) -> list[str]:
        """Return the systems that a set of conditions needs."""
        if not self.mesh.half:
            return ["full"]
        needed = ["sym"]
        if any(np.any(c[list(_ASYM_COLS)] != 0.0) for c in coeff_rows):
            needed.append("asym")
        return needed

    def _config_strips(self, cfg: _Configuration) -> None:
        """Compute the wake strip matrices of a configuration once (host float64).

        The keys are "self" (the wake), "m" (its mirror on a half mesh),
        "g" (its ground image) and "yg" (the ground image of the mirror).
        """
        if len(cfg.wake.panels) == 0 or "self" in cfg.strips:
            return
        attitude_key = (cfg.key[1], cfg.key[3]) if cfg.ground is not None else None
        shared = self._attitude_cache.get(attitude_key)
        if shared is not None:
            self._attitude_cache.move_to_end(attitude_key)
        if shared is not None and shared.strips:
            cfg.strips.update(shared.strips)
        else:
            cfg.strips["self"] = self._strip_matrix(cfg.wake.panels, cfg.wake)[0]
            if self.mesh.half:
                cfg.strips["m"] = self._strip_matrix(cfg.wake_panels_m, cfg.wake)[0]
            if shared is not None:
                shared.strips.update(cfg.strips)
                total = sum(a.nbytes for entry in self._attitude_cache.values()
                            for a in entry.strips.values())
                while total > _ATTITUDE_CACHE_BYTES:
                    _, oldest = self._attitude_cache.popitem(last=False)
                    total -= sum(a.nbytes for a in oldest.strips.values())
        if cfg.ground is not None:
            normal, offset = cfg.ground
            cfg.strips["g"] = self._strip_matrix(
                reflect_panels_plane(cfg.wake.panels, normal, offset), cfg.wake
            )[0]
            if self.mesh.half:
                cfg.strips["yg"] = self._strip_matrix(
                    reflect_panels_plane(cfg.wake_panels_m, normal, offset), cfg.wake
                )[0]

    def _uses_ground_krylov(self, cfg: _Configuration) -> bool:
        """Return whether this configuration can use CPU float64 GMRES."""
        return (self.ground_solver in ("auto", "krylov") and cfg.ground is not None
                and self.backend != "gpu" and self.kutta == "morino"
                and self._viscous_strips is None)

    def _ground_height(self, cfg: _Configuration) -> float:
        """Return the height [m] of the moment reference point above the ground of *cfg*.

        Ground configurations with the same height share one anchor LU. The
        plane offset itself is not a height: with a reference point away
        from the origin it changes with the attitude at a fixed height.
        """
        normal, offset = cfg.ground
        return round(float(np.dot(self.ref_point, normal)) - float(offset), 12)

    def _ground_matrix(self, cfg, parity):
        """Assemble one ground system [-] and source sums [m^2/s per onset]."""
        sign, cols = self._parity_sign_cols(parity)
        a_body, rhs = self._body_system(parity)
        normal, offset = cfg.ground
        image = reflect_panels_plane(self.body_verts, normal, offset)
        image_m = (reflect_panels_plane(self._body_verts_m, normal, offset)
                   if self.mesh.half else None)
        a_img, rhs_img = self._pair_system(image, image_m, sign, cols)
        mat_a = self._add(a_body, a_img)
        self._config_strips(cfg)
        if len(cfg.wake.panels):
            _fold_wake(mat_a, self._strip_system(cfg, parity), cfg.wake)
        return mat_a, rhs + rhs_img

    def _ground_gmres(self, mat_a, lu_info, rhs, initial):
        """Retain the single-column SciPy reference for parity tests only.

        Use Saad and Schultz (1986), SIAM J. Sci. Stat. Comput. 7(3),
        pp. 856-869, and Saad (2003), Iterative Methods for Sparse Linear
        Systems, second edition, chapters 6 and 9. Specific page and
        equation numbers for right preconditioning: to be checked.
        Solve A M^-1 y = b - A x_anchor, then x = x_anchor + M^-1 y.
        Inputs are A [-] and unit doublets and right-hand sides in
        m^2/s per unit onset. Return x, inner iterations [-], relative
        true residual [-], and convergence status.
        """
        norm_b = float(np.linalg.norm(rhs))
        if norm_b == 0.0:
            return np.zeros_like(rhs), 0, 0.0, True

        def inverse(v):
            return self._lu_solve(lu_info, v)

        operator = scipy.sparse.linalg.LinearOperator(
            mat_a.shape, matvec=lambda v: mat_a @ inverse(v), dtype=np.float64,
        )
        count = 0

        def iteration(_):
            nonlocal count
            count += 1

        # SciPy before 1.12 calls the relative tolerance "tol".
        parameters = inspect.signature(scipy.sparse.linalg.gmres).parameters
        tolerance = "tol" if "tol" in parameters and "rtol" not in parameters else "rtol"
        correction, info = scipy.sparse.linalg.gmres(
            operator, rhs - mat_a @ initial, restart=50,
            maxiter=_GROUND_GMRES_MAXITER, atol=0.0,
            callback=iteration, callback_type="legacy", **{tolerance: _GROUND_GMRES_INNER_RTOL},
        )
        solution = initial + inverse(correction)
        residual = float(np.linalg.norm(rhs - mat_a @ solution) / norm_b)
        return solution, count, residual, info == 0 and np.isfinite(residual) and residual <= _GROUND_GMRES_RTOL

    def _ground_gmres_lockstep(self, mat_a, lu_info, rhs, initial):
        """Solve independent right-preconditioned GMRES columns in lockstep.

        Follow Saad and Schultz (1986), SIAM J. Sci. Stat. Comput. 7(3),
        pp. 856-869, and Saad (2003), Iterative Methods for Sparse Linear
        Systems, second edition, chapters 6 and 9. Use modified
        Gram-Schmidt and Givens rotations. Algorithm and equation
        numbers: to be checked. Each column has its own Krylov space.

        Parameters
        ----------
        mat_a : numpy.ndarray
            System matrix with shape (N, N) [-].
        lu_info : tuple
            Anchor LU factors and pivots of the right preconditioner [-].
        rhs : numpy.ndarray
            Right-hand sides with shape (N, K) in m^2/s per unit onset.
        initial : numpy.ndarray
            Anchor solutions with shape (N, K) in m^2/s per unit onset.

        Returns
        -------
        solution : numpy.ndarray
            Solutions with shape (N, K) in m^2/s per unit onset.
        counts : numpy.ndarray
            Inner iteration counts per column [-].
        residuals : numpy.ndarray
            Relative true residuals per column [-].
        accepted : numpy.ndarray
            Inner convergence and true-residual acceptance per column.
        matrix_passes : int
            Products with mat_a [-]. Count each initial, restart and final
            residual pass, and each shared Arnoldi pass. All-zero right-hand
            sides need no passes. Direct solves do not use this counter.
        """
        n, k = rhs.shape
        solution = np.array(initial, dtype=np.float64, copy=True)
        counts = np.zeros(k, dtype=int)
        residuals = np.zeros(k)
        inner_done = np.zeros(k, dtype=bool)
        norm_b = np.linalg.norm(rhs, axis=0)
        nonzero = np.flatnonzero(norm_b != 0.0)
        solution[:, norm_b == 0.0] = 0.0
        inner_done[norm_b == 0.0] = True
        if not len(nonzero):
            return solution, counts, residuals, inner_done, 0

        residual = np.zeros_like(solution)
        residual[:, nonzero] = rhs[:, nonzero] - mat_a @ solution[:, nonzero]
        matrix_passes = 1
        # Match the reference correction solve: its RHS is b - A x_anchor.
        limit = _GROUND_GMRES_INNER_RTOL * np.linalg.norm(residual, axis=0)
        inner_done[nonzero] = np.linalg.norm(residual[:, nonzero], axis=0) == 0.0
        active = np.flatnonzero(~inner_done)
        restart = min(50, n)
        for start in range(0, _GROUND_GMRES_MAXITER, restart):
            if not len(active):
                break
            steps = min(restart, _GROUND_GMRES_MAXITER - start)
            basis = np.zeros((k, n, steps + 1))
            preconditioned = np.zeros((k, n, steps))
            hessenberg = np.zeros((k, steps + 1, steps))
            cosine = np.zeros((k, steps))
            sine = np.zeros((k, steps))
            projected = np.zeros((k, steps + 1))
            for col in active:
                beta = float(np.linalg.norm(residual[:, col]))
                basis[col, :, 0] = residual[:, col] / beta
                projected[col, 0] = beta
            remaining = []
            for step in range(steps):
                # The only dense inverse and matrix passes of this iteration.
                vectors = basis[active, :, step].T
                z = self._lu_solve(lu_info, vectors)
                work = mat_a @ z
                matrix_passes += 1
                next_active = []
                for j, col in enumerate(active):
                    counts[col] += 1
                    preconditioned[col, :, step] = z[:, j]
                    w = work[:, j].copy()
                    before = float(np.linalg.norm(w))
                    h = hessenberg[col]
                    for row in range(step + 1):
                        h[row, step] = np.dot(basis[col, :, row], w)
                        w -= h[row, step] * basis[col, :, row]
                    h[step + 1, step] = np.linalg.norm(w)
                    breakdown = h[step + 1, step] <= np.finfo(float).eps * before
                    if not breakdown:
                        basis[col, :, step + 1] = w / h[step + 1, step]
                    for row in range(step):
                        upper, lower = h[row:row + 2, step].copy()
                        h[row, step] = cosine[col, row] * upper + sine[col, row] * lower
                        h[row + 1, step] = -sine[col, row] * upper + cosine[col, row] * lower
                    diagonal = float(np.hypot(h[step, step], h[step + 1, step]))
                    if diagonal == 0.0 or not np.isfinite(diagonal):
                        # No valid correction exists in this column's space.
                        continue
                    cosine[col, step] = h[step, step] / diagonal
                    sine[col, step] = h[step + 1, step] / diagonal
                    h[step, step] = diagonal
                    h[step + 1, step] = 0.0
                    projected[col, step + 1] = -sine[col, step] * projected[col, step]
                    projected[col, step] *= cosine[col, step]
                    inner_done[col] = abs(projected[col, step + 1]) <= limit[col]
                    if inner_done[col] or breakdown or step + 1 == steps:
                        size = step + 1
                        weights = scipy.linalg.solve_triangular(h[:size, :size], projected[col, :size])
                        solution[:, col] += preconditioned[col, :, :size] @ weights
                        if not inner_done[col] and not breakdown:
                            remaining.append(col)
                    else:
                        next_active.append(col)
                active = np.asarray(next_active, dtype=int)
                if not len(active):
                    break
            active = np.asarray(remaining, dtype=int)
            if len(active) and start + steps < _GROUND_GMRES_MAXITER:
                residual[:, active] = rhs[:, active] - mat_a @ solution[:, active]
                matrix_passes += 1
                inner_done[active] = np.linalg.norm(residual[:, active], axis=0) <= limit[active]
                active = active[~inner_done[active]]

        final = rhs[:, nonzero] - mat_a @ solution[:, nonzero]
        matrix_passes += 1
        residuals[nonzero] = np.linalg.norm(final, axis=0) / norm_b[nonzero]
        accepted = inner_done & np.isfinite(residuals) & (residuals <= _GROUND_GMRES_RTOL)
        return solution, counts, residuals, accepted, matrix_passes

    def _ensure(self, cfg: _Configuration, parities: list[str]) -> None:
        """Solve missing unit systems, with an optional ground LU anchor."""
        if not self._uses_ground_krylov(cfg):
            if self._viscous_strips is not None and cfg.ground is not None:
                cfg.ground_solver_reason = "viscous coupling: one direct LU per configuration"
            self._ensure_direct(cfg, parities)
            return
        missing = [p for p in parities if p not in cfg.systems]
        if not missing:
            return
        height = self._ground_height(cfg)
        anchor = next((c for c in self._configs.values()
                       if c.ground_anchor and self._ground_height(c) == height), None)
        if anchor is None or anchor is cfg:
            cfg.ground_anchor = True
            if cfg.ground_solver == "krylov":
                # A cached iterative configuration needs direct units before
                # it can supply the initial guess of a new anchor.
                missing = list(dict.fromkeys([*cfg.systems, *parities]))
                cfg.systems.clear()
                cfg.ground_solver = "direct"
            self._ensure_direct(cfg, missing)
            cfg.strips.clear()
            return
        self._protected_factors.add(anchor.key)
        self._touch_factors(anchor)
        # An onset added on a later call also needs its anchor unit solution.
        self._ensure_direct(anchor, missing)
        anchor.strips.clear()
        if cfg.unit_mu is None:
            cfg.unit_mu = np.zeros((len(self.centroids), 6), dtype=np.float64)
        cfg.ground_solver = "krylov"
        if self.ground_solver == "auto":
            cfg.ground_solver_reason = "auto: krylov in ground effect"
        for parity in missing:
            _, cols = self._parity_sign_cols(parity)
            mat_a, rhs = self._ground_matrix(cfg, parity)
            sol, counts, residuals, ok, passes = self._ground_gmres_lockstep(
                mat_a, anchor.systems[parity], -rhs, anchor.unit_mu[:, list(cols)],
            )
            cfg.unit_mu[:, list(cols)] = sol
            cfg.matrix_passes += passes
            for idx, col in enumerate(cols):
                cfg.iterations[col] = int(counts[idx])
                cfg.relative_residuals[col] = float(residuals[idx])
            del mat_a
            if not np.all(ok):
                cfg.fallbacks.append({"parity": parity, "iterations": dict(cfg.iterations),
                                      "relative_residuals": dict(cfg.relative_residuals)})
                # All unit solutions of this configuration must become direct.
                needed = list(dict.fromkeys([*cfg.systems, *parities]))
                cfg.systems.clear()
                cfg.ground_solver = "direct"
                cfg.ground_anchor = True
                self._ensure_direct(cfg, needed)
                self._drop_factors(anchor)
                cfg.strips.clear()
                return
            # The unit solutions suffice for later conditions. Drop the matrix.
            cfg.systems[parity] = None
        self._update_unit_gradients(cfg)
        cfg.strips.clear()

    def _ensure_direct(self, cfg: _Configuration, parities: list[str]) -> None:
        """Assemble and factorize the missing systems of a configuration.

        Each system is ``A = A_body + A_image + fold(D)``: the kept
        free-air body parity matrix, the image parity matrix of the pair
        kernel in ground effect, and the summed wake strips of the system
        on the trailing-edge columns. Only the factorized systems, the
        body parity matrices and the strip matrices are kept.
        """
        if self._viscous_strips is not None and "asym" in parities:
            raise ValueError(
                "Viscous panel coupling does not support asymmetric flow on a half mesh; use a full mesh."
            )
        keep_factors = self.kutta == "pressure" or cfg.ground_anchor
        missing = [p for p in parities if p not in cfg.systems
                   or (keep_factors and cfg.systems[p] is None)]
        if not missing:
            return
        if self.ground_solver == "auto" and cfg.ground_anchor:
            cfg.ground_solver_reason = "auto: direct anchor"
        half = bool(self.mesh.half)
        in_ground = cfg.ground is not None
        parts = self._body_parts or {}
        body_cached = all(f"A_{p}" in parts for p in missing)
        n_panels = len(self.centroids)
        itemsize = self._gpu_dtype.itemsize if self.backend == "gpu" else 8
        incoming = (len(missing) * (n_panels * n_panels * itemsize + n_panels * 4)
                    if keep_factors else 0)
        if self._viscous_strips is not None:
            incoming += n_panels * len(self._viscous_components) * len(self._viscous_strips) * 8
        self._trim_factors((0, incoming) if self.backend == "gpu" else (incoming, 0))
        self._check_gpu_memory(half, in_ground, len(missing), body_cached)
        has_wake = len(cfg.wake.panels) > 0
        self._config_strips(cfg)
        if in_ground:
            normal, offset = cfg.ground
            image = reflect_panels_plane(self.body_verts, normal, offset)
            image_m = (
                reflect_panels_plane(self._body_verts_m, normal, offset) if half else None
            )

        if cfg.unit_mu is None:
            cfg.unit_mu = np.zeros((n_panels, 6), dtype=np.float64)
        for parity in missing:
            sign, cols = self._parity_sign_cols(parity)
            a_body, rhs = self._body_system(parity)
            if in_ground:
                # Ground images: equal source and doublet strengths, reflected panels.
                a_img, rhs_img = self._pair_system(image, image_m, sign, cols)
                mat_a = self._add(a_body, a_img)
                del a_img
                rhs = rhs + rhs_img
            else:
                mat_a = self._copy(a_body)
            if has_wake:
                d_mat = self._strip_system(cfg, parity)
                if self.backend == "gpu":
                    import torch

                    d_mat = torch.as_tensor(d_mat, dtype=self._gpu_dtype, device=self._gpu_device)
                _fold_wake(mat_a, d_mat, cfg.wake)
            lu_info = self._factorize(mat_a)
            del mat_a
            cfg.systems[parity] = lu_info
            if self._viscous_strips is None:
                self._unit_solutions(cfg, lu_info, rhs, cols, parity)
            elif parity in ("sym", "full"):
                self._unit_solutions(cfg, lu_info, rhs[:, :len(cols)], cols, parity)
                cfg.viscous_units = self._lu_solve(lu_info, -rhs[:, len(cols):])
                self._viscous_solve_count += (1 if parity == "full" else len(cols)) + 1
                self._viscous_rhs_count += rhs.shape[1]
            cfg.iterations.update(dict.fromkeys(cols, 0))
            cfg.relative_residuals.update(dict.fromkeys(cols))
            if not keep_factors:
                cfg.systems[parity] = None
                self._factors_dropped += 1
            del lu_info
        self._update_unit_gradients(cfg)
        if self.kutta == "morino":
            cfg.strips.clear()

    def _unit_solutions(self, cfg, lu_info, rhs, cols, parity) -> None:
        """Solve the unit onset right-hand sides of one system.

        *rhs* holds the source sums ``B @ sigma_unit[:, cols]`` with shape
        (N, K) in m^2/s per unit onset; the right-hand sides are ``-rhs``.
        """
        if self.backend == "gpu":
            import torch

            sol = self._lu_solve(lu_info, (-rhs).to(self._gpu_dtype))
            if cfg.unit_mu_t is None:
                cfg.unit_mu_t = torch.zeros(
                    (len(self.centroids), 6), dtype=self._gpu_dtype, device=self._gpu_device
                )
            cfg.unit_mu_t[:, list(cols)] = sol
            cfg.unit_mu[:, list(cols)] = sol.to(torch.float64).cpu().numpy()
            return
        if parity == "full":
            # Keep the memory layout of the solve output: the superposition
            # products then round as in every earlier version.
            cfg.unit_mu = self._lu_solve(lu_info, -np.ascontiguousarray(rhs))
            return
        for idx, k in enumerate(cols):
            cfg.unit_mu[:, k] = self._lu_solve(lu_info, -rhs[:, idx])

    def _full_unit_mu(self, unit_mu: np.ndarray) -> np.ndarray:
        """Return the unit doublets on the full mesh with shape (Nf, 6)."""
        if not self.mesh.half:
            return unit_mu
        left = unit_mu.copy()
        left[:, list(_ASYM_COLS)] *= -1.0
        return np.vstack([unit_mu, left])

    def _update_unit_gradients(self, cfg: _Configuration) -> None:
        """Compute the surface gradients of the six unit solutions on the full mesh."""
        grad = surface_gradient(self._grad_op, self._full_unit_mu(cfg.unit_mu))
        cfg.unit_grad = np.ascontiguousarray(np.transpose(grad, (0, 2, 1)))
        if self.backend == "gpu":
            import torch

            cfg.unit_grad_t = torch.as_tensor(
                cfg.unit_grad, dtype=self._gpu_dtype, device=self._gpu_device
            )

    # ------------------------------------------------------------------
    # Conditions
    # ------------------------------------------------------------------

    def _condition_velocity(
        self,
        condition: FlightCondition,
        points: np.ndarray,
    ) -> np.ndarray:
        """Return local onset flow velocity at points for the condition."""
        alpha = float(condition.alpha)
        beta = float(condition.beta)
        v_inf = float(condition.V_inf)
        d = freestream_direction(alpha, beta)
        v_trans = v_inf * d[None, :]

        w_rot = rotation_velocity(condition, points, self.ref_point)
        if w_rot is not None:
            return v_trans + w_rot
        return np.broadcast_to(v_trans, points.shape)

    def _condition_coeffs(self, condition: FlightCondition) -> np.ndarray:
        """Return the six unit-onset weights for a flight condition."""
        alpha = float(condition.alpha)
        beta = float(condition.beta)
        v_inf = float(condition.V_inf)
        d = freestream_direction(alpha, beta)
        u = v_inf * d[0]
        v = v_inf * d[1]
        w = v_inf * d[2]
        p = float(getattr(condition, "p", 0.0))
        q = float(getattr(condition, "q", 0.0))
        r = float(getattr(condition, "r", 0.0))
        return np.array([u, v, w, p, q, r], dtype=np.float64)

    def _reconstruct_full_mu(self, unit_mu: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        """Reconstruct the full mesh Morino doublets from half or full solve."""
        mu_body = unit_mu @ coeffs
        if not self.mesh.half:
            return np.array(mu_body, dtype=float, copy=True)
        mu_sym = unit_mu[:, list(_SYM_COLS)] @ coeffs[list(_SYM_COLS)]
        mu_asym = unit_mu[:, list(_ASYM_COLS)] @ coeffs[list(_ASYM_COLS)]
        mu_left = mu_sym - mu_asym
        return np.concatenate([mu_body, mu_left])

    def _trefftz(self, cfg: _Configuration, mu_full, condition, wake_gamma=None) -> float:
        """Trefftz-plane induced drag of a configuration (images in ground effect)."""
        return compute_trefftz_induced_drag(
            wake=cfg.full_wake,
            mu=mu_full,
            V_inf=float(condition.V_inf),
            rho=float(condition.rho),
            S_ref=self.S_ref,
            wake_dir=cfg.wake_dir,
            ground=cfg.ground,
            wake_gamma=wake_gamma,
        )

    def _evaluate_result(self, cfg: _Configuration, condition: FlightCondition) -> PanelResult:
        """Evaluate coefficients and distributions of one condition (Morino Kutta)."""
        v_inf = float(condition.V_inf)
        coeffs = self._condition_coeffs(condition)
        unit_mu = cfg.unit_mu
        v_onset_full = self._condition_velocity(condition, self._full_geom.centroid)
        if self.backend == "gpu":
            from ventorum.panel3d import gpu as gpu_mod

            mu_body, sigma_body, _, v_surf_full, cp_full = gpu_mod.evaluate_condition_on_device(
                cfg.unit_mu_t,
                self._gpu_unit_sigma,
                cfg.unit_grad_t,
                coeffs,
                v_onset_full,
                self._full_geom.normal,
                v_inf,
                has_rates(condition),
            )
        else:
            mu_body = unit_mu @ coeffs
            sigma_body = self._unit_sigma @ coeffs
            grad_full = np.einsum("k,nkd->nd", coeffs, cfg.unit_grad)
            v_surf_full, cp_full = compute_surface_velocity_and_cp(
                v_onset_full, self._full_geom.normal, grad_full, v_inf, has_rates(condition)
            )

        forces = integrate_forces_and_moments(
            cp=cp_full,
            areas=self._full_geom.area,
            normals=self._full_geom.normal,
            centroids=self._full_geom.centroid,
            condition=condition,
            S_ref=self.S_ref,
            b_ref=self.b_ref,
            c_ref=self.c_ref,
            ref_point=self.ref_point,
        )

        if not self.mesh.half:
            mu_full = mu_body
        else:
            mu_sym = unit_mu[:, list(_SYM_COLS)] @ coeffs[list(_SYM_COLS)]
            mu_asym = unit_mu[:, list(_ASYM_COLS)] @ coeffs[list(_ASYM_COLS)]
            mu_left = mu_sym - mu_asym
            mu_full = np.concatenate([mu_body, mu_left])
        cdi_trefftz = self._trefftz(cfg, mu_full, condition)

        n_panels = len(self.centroids)
        cp_out = cp_full[:n_panels]
        v_surf_out = v_surf_full[:n_panels]

        up_te = np.asarray(cfg.wake.wake_up, dtype=int)
        lo_te = np.asarray(cfg.wake.wake_lo, dtype=int)
        if up_te.size:
            te_resid = float(np.max(np.abs(cp_out[up_te] - cp_out[lo_te])))
            mu_wake = mu_body[up_te] - mu_body[lo_te]
        else:
            te_resid = 0.0
            mu_wake = np.zeros(0, dtype=float)

        return PanelResult(
            CL=forces["CL"],
            CD_pressure=forces["CD_pressure"],
            CDi_trefftz=cdi_trefftz,
            CY=forces["CY"],
            Cl=forces["Cl"],
            Cm=forces["Cm"],
            Cn=forces["Cn"],
            cp=cp_out,
            mu=mu_body,
            sigma=sigma_body,
            velocity=v_surf_out,
            condition=condition,
            kutta="morino",
            kutta_iters=0,
            kutta_residual=te_resid,
            kutta_converged=True,
            trust_note="",
            mu_w=mu_wake,
        )

    def _condition_bytes(self) -> int:
        """Return a conservative temporary array size per condition [bytes].

        Count float64 host and device fields, pressure force work arrays,
        superposition work arrays and the endpoint circulation vectors.
        Returned result arrays are not temporary storage.
        """
        strips = len(self._full_wake.wake_up)
        return 8 * (48 * len(self._full_geom.area) + 16 * strips + 12 * strips**2 + 256)

    def _evaluate_results_batch(self, cfg, conditions, coeffs):
        """Evaluate one Morino chunk with onset weights in m/s and rad/s.

        Conditions have one wake and image configuration. Coeffs has
        shape (K, 6). Return PanelResult objects in condition order.
        """
        _, speed, rho, lift, drag = _condition_arrays(conditions)
        n = len(self.centroids)
        wake = cfg.full_wake
        if len(wake.wake_up) and cfg.trefftz_factors is None:
            cfg.trefftz_factors = _trefftz_factors(wake, cfg.wake_dir, cfg.ground)
        if self.backend == "gpu":
            import torch

            from ventorum.panel3d.gpu import _evaluate_conditions_on_device

            def device_array(a, dtype=torch.float64):
                return torch.as_tensor(a, dtype=dtype, device=self._gpu_device)

            if cfg.gpu_post is None:
                geometry = {
                    "normal": device_array(self._full_geom.normal),
                    "area": device_array(self._full_geom.area),
                    "arm": device_array(self._full_geom.centroid - self.ref_point),
                    "unit_mu": device_array(cfg.unit_mu),
                    "half": self.mesh.half,
                }
                trefftz = None
                if len(wake.wake_up):
                    trefftz = (
                        tuple(device_array(a) for a in cfg.trefftz_factors),
                        device_array(wake.wake_up, torch.int64),
                        device_array(wake.wake_lo, torch.int64), self.S_ref, cfg.ground is not None,
                    )
                cfg.gpu_post = geometry, trefftz
            else:
                # An antisymmetric solve can fill new unit columns on a
                # later call. Refresh only this small linear field.
                cfg.gpu_post[0]["unit_mu"] = device_array(cfg.unit_mu)
            mu, sigma, velocity, cp, force, moment, cdi = _evaluate_conditions_on_device(
                cfg.unit_mu_t, self._gpu_unit_sigma, cfg.unit_grad_t,
                coeffs, speed, rho, *cfg.gpu_post,
            )
            forces = _force_coefficients_batch(
                force, moment, speed, rho, lift, drag, self.S_ref, self.b_ref, self.c_ref,
            )
        else:
            # Fixed contractions keep each result independent of chunk size.
            mu = np.einsum("nk,ck->cn", cfg.unit_mu, coeffs, order="C")
            sigma = np.einsum("nk,ck->cn", self._unit_sigma, coeffs, order="C")
            gradient = np.einsum("ck,nkd->cnd", coeffs, cfg.unit_grad, order="C")
            omega = coeffs[:, 3:] * np.array([-1.0, 1.0, -1.0])
            onset = coeffs[:, None, :3] - np.cross(
                omega[:, None, :], self._full_geom.centroid - self.ref_point,
            )
            rates = np.any(coeffs[:, 3:] != 0.0, axis=1)
            velocity, cp = _surface_velocity_batch(
                onset, self._full_geom.normal, gradient, speed, rates,
            )
            forces = _integrate_batch(
                cp, self._full_geom, speed, rho, lift, drag,
                self.S_ref, self.b_ref, self.c_ref, self.ref_point,
            )
            mu_full = mu
            if self.mesh.half:
                left = (np.einsum("nk,ck->cn", cfg.unit_mu[:, _SYM_COLS],
                                  coeffs[:, _SYM_COLS], order="C")
                        - np.einsum("nk,ck->cn", cfg.unit_mu[:, _ASYM_COLS],
                                    coeffs[:, _ASYM_COLS], order="C"))
                mu_full = np.concatenate([mu, left], axis=1)
            gamma = mu_full[:, wake.wake_up] - mu_full[:, wake.wake_lo]
            cdi = _trefftz_batch(
                gamma, speed, rho, self.S_ref, cfg.trefftz_factors, cfg.ground is not None,
            )
        up, lo = cfg.wake.wake_up, cfg.wake.wake_lo
        mu_w = mu[:, up] - mu[:, lo]
        residual = (np.max(np.abs(cp[:, up] - cp[:, lo]), axis=1)
                    if len(up) else np.zeros(len(conditions)))
        return [PanelResult(
            **{name: float(values[i]) for name, values in forces.items()},
            CDi_trefftz=float(cdi[i]), cp=cp[i, :n], mu=mu[i], sigma=sigma[i],
            velocity=velocity[i, :n], condition=condition,
            kutta_residual=float(residual[i]), mu_w=mu_w[i],
        ) for i, condition in enumerate(conditions)]

    def _strip_system(self, cfg: _Configuration, parity: str) -> np.ndarray:
        """Return the strip influence matrix of one system (host float64)."""
        sign = -1.0 if parity == "asym" else 1.0
        d_mat = np.array(cfg.strips["self"], copy=True)
        if "m" in cfg.strips:
            d_mat = d_mat + sign * cfg.strips["m"]
        if "g" in cfg.strips:
            d_mat = d_mat + cfg.strips["g"]
        if "yg" in cfg.strips:
            d_mat = d_mat + sign * cfg.strips["yg"]
        return d_mat

    def _pressure_sensitivity(self, cfg: _Configuration, coeffs: np.ndarray):
        """Return (z_solve, z_full, wake_up, wake_lo, expand) of the pressure Kutta.

        *expand* maps the unknown corrections to the corrections of all
        strips of the full wake.
        """
        self._config_strips(cfg)
        n_strips = len(cfg.wake.wake_up)
        if not self.mesh.half:
            z = self._solve_host(cfg, "full", -self._strip_system(cfg, "full"))
            up = np.asarray(cfg.wake.wake_up, dtype=int)
            lo = np.asarray(cfg.wake.wake_lo, dtype=int)
            return z, z, up, lo, lambda d: d
        z_sym = self._solve_host(cfg, "sym", -self._strip_system(cfg, "sym"))
        if not np.any(coeffs[list(_ASYM_COLS)] != 0.0):
            # Symmetric flow: one correction per right strip and its mirror.
            up = np.asarray(cfg.wake.wake_up, dtype=int)
            lo = np.asarray(cfg.wake.wake_lo, dtype=int)
            return z_sym, np.vstack([z_sym, z_sym]), up, lo, lambda d: np.concatenate([d, d])
        # Asymmetric flow: one correction per strip of the full wake.
        z_asym = self._solve_host(cfg, "asym", -self._strip_system(cfg, "asym"))
        z_same = 0.5 * (z_sym + z_asym)
        z_other = 0.5 * (z_sym - z_asym)
        z_solve = np.hstack([z_same, z_other])
        z_full = np.block([[z_same, z_other], [z_other, z_same]])
        up = np.asarray(cfg.full_wake.wake_up, dtype=int)
        lo = np.asarray(cfg.full_wake.wake_lo, dtype=int)
        if up.shape[0] != 2 * n_strips:
            raise RuntimeError("The full wake of a half mesh must hold two strips per half strip.")
        return z_solve, z_full, up, lo, lambda d: d

    def _evaluate_pressure_result(
        self, cfg: _Configuration, condition: FlightCondition
    ) -> PanelResult:
        """Evaluate one flight condition with the iterative pressure Kutta."""
        coeffs = self._condition_coeffs(condition)
        mu_solve0 = cfg.unit_mu @ coeffs
        sigma_body = self._unit_sigma @ coeffs
        mu_full0 = self._reconstruct_full_mu(cfg.unit_mu, coeffs)
        v_inf = float(condition.V_inf)
        v_onset_full = self._condition_velocity(condition, self._full_geom.centroid)

        if len(cfg.wake.wake_up) == 0:
            z_solve = np.zeros((len(mu_solve0), 0))
            z_full = np.zeros((len(mu_full0), 0))
            up = lo = np.zeros(0, dtype=int)

            def expand(d):
                return d
        else:
            z_solve, z_full, up, lo, expand = self._pressure_sensitivity(cfg, coeffs)

        kres = solve_pressure_kutta(
            mu_full0,
            z_full,
            self._grad_op,
            up,
            lo,
            self._full_geom.normal,
            v_onset_full,
            v_inf,
            rates=has_rates(condition),
        )
        mu_solve = mu_solve0 + z_solve @ kres.delta
        mu_full = kres.mu_full

        trust_note = ""
        if not kres.converged:
            msg = (
                "Pressure Kutta did not converge in 20 iterations; "
                f"final max|delta Cp|={kres.residual:.3e}. Treat the trailing edge with care."
            )
            warnings.warn(msg, RuntimeWarning, stacklevel=3)
            trust_note = msg

        grad_full = surface_gradient(self._grad_op, mu_full)
        v_surf_full, cp_full = compute_surface_velocity_and_cp(
            v_onset_full, self._full_geom.normal, grad_full, v_inf, has_rates(condition)
        )
        forces = integrate_forces_and_moments(
            cp=cp_full,
            areas=self._full_geom.area,
            normals=self._full_geom.normal,
            centroids=self._full_geom.centroid,
            condition=condition,
            S_ref=self.S_ref,
            b_ref=self.b_ref,
            c_ref=self.c_ref,
            ref_point=self.ref_point,
        )

        # Trefftz drag with the corrected wake strengths of all strips.
        up_f = np.asarray(cfg.full_wake.wake_up, dtype=int)
        lo_f = np.asarray(cfg.full_wake.wake_lo, dtype=int)
        if up_f.size:
            gamma_full = (mu_full[up_f] - mu_full[lo_f]) + expand(kres.delta)
        else:
            gamma_full = np.zeros(0, dtype=float)
        cdi_trefftz = self._trefftz(cfg, mu_full, condition, wake_gamma=gamma_full)

        n_panels = len(self.centroids)
        n_strips = len(cfg.wake.wake_up)
        return PanelResult(
            CL=forces["CL"],
            CD_pressure=forces["CD_pressure"],
            CDi_trefftz=cdi_trefftz,
            CY=forces["CY"],
            Cl=forces["Cl"],
            Cm=forces["Cm"],
            Cn=forces["Cn"],
            cp=cp_full[:n_panels],
            mu=np.array(mu_solve, dtype=float, copy=True),
            sigma=sigma_body,
            velocity=v_surf_full[:n_panels],
            condition=condition,
            kutta="pressure",
            kutta_iters=int(kres.n_iters),
            kutta_residual=float(kres.residual),
            kutta_converged=bool(kres.converged),
            trust_note=trust_note,
            mu_w=np.array(gamma_full[:n_strips], dtype=float, copy=True),
        )

    def _full_mesh_solver(self) -> PanelSolver:
        """Return the full-mesh solver of a half mesh (built once, when needed)."""
        if self._full_solver is None:
            self._full_solver = PanelSolver(
                self._full_mesh,
                wake_length=self.wake_length,
                n_wake=self.n_wake,
                far_ratio=self.far_ratio,
                S_ref=self.S_ref,
                b_ref=self.b_ref,
                c_ref=self.c_ref,
                ref_point=self.ref_point,
                kutta=self.kutta,
                backend=self.backend,
                device=self._device_arg,
                dtype=self._dtype_arg,
                ground_solver=self.ground_solver,
                cpu_lu=self.cpu_lu,
                _reuse_full_from=self,
            )
        return self._full_solver

    def _to_half(self, res: PanelResult) -> PanelResult:
        """Return a full-mesh result with the distributions of the right half."""
        n_panels = len(self.centroids)
        n_strips = len(self.wake.wake_up)
        return dataclasses.replace(
            res,
            cp=np.array(res.cp[:n_panels], copy=True),
            mu=np.array(res.mu[:n_panels], copy=True),
            sigma=np.array(res.sigma[:n_panels], copy=True),
            velocity=np.array(res.velocity[:n_panels], copy=True),
            mu_w=None if res.mu_w is None else np.array(res.mu_w[:n_strips], copy=True),
        )

    def solve(self, condition: FlightCondition, ground=None) -> PanelResult:
        """Solve potential flow for a given flight condition.

        Parameters
        ----------
        condition : FlightCondition
            Flight condition to evaluate.
        ground : GroundPlane or tuple or None, optional
            Explicit ground plane (an object with ``normal`` and ``offset``
            [m], or a tuple ``(normal, offset)``). None uses ``condition.h``.

        Returns
        -------
        PanelResult
            Forces, moments, pressure, doublet and source distributions.
        """
        return self.solve_conditions([condition], grounds=None if ground is None else [ground])[0]

    def solve_conditions(
        self,
        conditions: Sequence[FlightCondition],
        grounds: Sequence | None = None,
    ) -> list[PanelResult]:
        """Solve potential flow for multiple flight conditions.

        Conditions are grouped by configuration: all free-air conditions
        share one configuration; in ground effect each (attitude, height)
        group is one configuration. Each system of a configuration is
        factorized once with the default direct option. The opt-in Krylov
        ground option reuses one anchor per ground offset [m].

        Parameters
        ----------
        conditions : sequence of FlightCondition
            Flight conditions to solve.
        grounds : sequence or None, optional
            Explicit ground plane of each condition (an object with
            ``normal`` and ``offset`` [m], a tuple ``(normal, offset)``, or
            None). None for all conditions uses ``condition.h``. An explicit
            plane replaces ``condition.h``.

        Returns
        -------
        list of PanelResult
            List of solution results corresponding to each condition.
        """
        conds = list(conditions)
        self._factors_dropped = 0
        self._configurations_evicted = 0
        self._bound_configurations()
        self._read_factor_budgets()
        if not conds:
            return []
        if grounds is None:
            ground_list = [None] * len(conds)
        else:
            ground_list = list(grounds)
            if len(ground_list) != len(conds):
                raise ValueError(
                    f"grounds holds {len(ground_list)} entries for {len(conds)} conditions."
                )

        # Group the conditions by configuration (checks the clearances).
        groups: dict[tuple, list[int]] = {}
        meta: dict[tuple, tuple] = {}
        for idx, (cond, gnd) in enumerate(zip(conds, ground_list)):
            key, wake_dir, gp = self._config_key(cond, gnd)
            if gp is not None and key not in groups:
                check_ground_clearance(self._full_mesh.nodes, gp[0], gp[1])
            groups.setdefault(key, []).append(idx)
            meta[key] = (wake_dir, gp)

        results: list[PanelResult | None] = [None] * len(conds)
        for key, indices in groups.items():
            wake_dir, gp = meta[key]
            if self.mesh.half and not self._symmetric_config(wake_dir, gp):
                # The wake or the ground breaks the y = 0 symmetry: full mesh.
                sub = self._full_mesh_solver()
                sub_res = sub.solve_conditions(
                    [conds[i] for i in indices], grounds=[gp] * len(indices)
                )
                self._factors_dropped += sub._factors_dropped
                self._configurations_evicted += sub._configurations_evicted
                for idx, res in zip(indices, sub_res):
                    results[idx] = self._to_half(res)
                continue
            cfg = self._configs.get(key)
            if cfg is None:
                cfg = self._new_config(key, wake_dir, gp)
                self._configs[key] = cfg
            self._configs.move_to_end(key)
            self._protected_factors = {key}
            self._touch_factors(cfg)
            group_conditions = [conds[i] for i in indices]
            coeff_rows = _condition_arrays(group_conditions)[0]
            self._ensure(cfg, self._parities_needed(coeff_rows))
            if self._viscous_evaluator is not None:
                for idx in indices:
                    results[idx] = self._viscous_evaluator(self, cfg, conds[idx])
                cfg.strips.clear()
            elif self.kutta == "morino":
                size = max(1, _CONDITION_BATCH_BYTES // self._condition_bytes())
                for start in range(0, len(indices), size):
                    chunk = indices[start:start + size]
                    evaluated = self._evaluate_results_batch(
                        cfg, group_conditions[start:start + size], coeff_rows[start:start + size],
                    )
                    for idx, res in zip(chunk, evaluated):
                        results[idx] = res
            else:
                for idx in indices:
                    results[idx] = self._evaluate_pressure_result(cfg, conds[idx])
                # Keep the strips for all conditions of the group, then drop them.
                cfg.strips.clear()

            self._protected_factors.clear()
            self._trim_factors()

        return [r for r in results if r is not None]
