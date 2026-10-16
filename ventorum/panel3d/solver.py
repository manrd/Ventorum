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
rate. A half mesh needs a flow configuration that is symmetric about
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
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import scipy.linalg

from ventorum.aero.system import freestream_direction, rotation_velocity
from ventorum.core.datatypes import FlightCondition
from ventorum.core.errors import GroundStrikeError
from ventorum.panel3d.images import (
    check_ground_clearance,
    ground_plane_parameters,
    mirror_panels_y,
    reconstruct_full_mesh,
    reflect_panels_plane,
)
from ventorum.panel3d.kernels import _precompute_geometry, doublet_potential, influence_matrices
from ventorum.panel3d.kutta import solve_pressure_kutta, strip_influence_matrix
from ventorum.panel3d.mesh import PanelMesh
from ventorum.panel3d.post import (
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
class _Configuration:
    """Wake, images and factorized systems of one set of conditions."""

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
    for s in range(len(wake.strip_panel_indices)):
        up_idx = int(wake.wake_up[s])
        lo_idx = int(wake.wake_lo[s])
        cs = d_strip[:, s]
        mat_a[:, up_idx] += cs
        mat_a[:, lo_idx] -= cs
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
        self.backend = key
        self.kutta = str(kutta)
        self.mesh = mesh
        self.far_ratio = float(far_ratio)
        self.n_wake = int(n_wake)
        self._device_arg = device
        self._dtype_arg = dtype
        self._own_factorizations = 0
        self._full_solver: PanelSolver | None = None

        # Standardize triangular panels on given mesh
        self.standardized_panels = _standardize_panels(mesh.panels, mesh.nodes)
        self.body_verts = mesh.nodes[self.standardized_panels]

        # Precompute body geometry on given mesh
        geom = _precompute_geometry(self.body_verts)
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
        self.wake: PanelWake = self._build_wake(
            mesh.nodes, self.standardized_panels, mesh.wake_edges, self.normals, None
        )

        # Reconstruct full mesh for post-processing and reference geometry if half mesh
        if mesh.half:
            self._full_mesh = reconstruct_full_mesh(mesh)
        else:
            self._full_mesh = mesh

        # Full mesh geometry and wake for global post-processing
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
        )

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
        self._configs: dict[tuple, _Configuration] = {}
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
        """Factorize one dense system (the only place of a factorization)."""
        self._own_factorizations += 1
        if self.backend == "gpu":
            from ventorum.panel3d import gpu as gpu_mod

            return gpu_mod.lu_factor_device(mat_a)
        return scipy.linalg.lu_factor(mat_a)

    def _lu_solve(self, lu_info, rhs):
        """Solve with stored LU factors (numpy or torch right-hand sides)."""
        if self.backend == "gpu":
            from ventorum.panel3d import gpu as gpu_mod

            return gpu_mod.lu_solve_device(lu_info, rhs)
        return scipy.linalg.lu_solve(lu_info, rhs)

    def _solve_host(self, cfg: _Configuration, parity: str, rhs: np.ndarray) -> np.ndarray:
        """Solve one stored system for host right-hand sides; return float64 numpy."""
        lu_info = cfg.systems[parity]
        if self.backend != "gpu":
            return scipy.linalg.lu_solve(lu_info, rhs)
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

    def _doublet(self, points: np.ndarray, panels: np.ndarray) -> np.ndarray:
        """Return the doublet potentials with the selected CPU backend."""
        if self.backend == "numba":
            from ventorum.panel3d.kernels_numba import influence_matrices_numba

            _, mat_c = influence_matrices_numba(
                np.asarray(points),
                np.asarray(panels),
                far_ratio=self.far_ratio,
            )
            return mat_c
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
        """Return a copy of a numpy array or torch tensor."""
        if self.backend == "gpu":
            return mat.clone()
        return np.array(mat, copy=True)

    def _check_gpu_memory(self, half: bool, ground: bool, n_systems: int) -> None:
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
            body_cached=self._body_parts is not None,
        )

    def _body(self) -> dict:
        """Return the body influence matrices (computed once, then kept)."""
        if self._body_parts is None:
            parts = {}
            b_self, c_self = self._matrices(self.centroids, self.body_verts)
            parts["B_self"] = b_self
            parts["C_self"] = c_self
            if self.mesh.half:
                b_m, c_m = self._matrices(self.centroids, self._body_verts_m)
                parts["B_m"] = b_m
                parts["C_m"] = c_m
            self._body_parts = parts
        return self._body_parts

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
        body = self._body()
        mat_a = self._copy(body["C_self"])
        if len(self.wake.panels) > 0:
            _, d_dev = self._strip_matrix(self.wake.panels, self.wake)
            _fold_wake(mat_a, d_dev, self.wake)
        mat_b = body["B_self"]
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

    def _new_config(self, key, wake_dir, ground) -> _Configuration:
        """Build the wake geometry of a configuration (no matrices yet)."""
        if wake_dir is None:
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
        if ground is not None:
            check_ground_clearance(self._full_mesh.nodes, ground[0], ground[1])
            if len(full_wake.panels) > 0:
                try:
                    check_ground_clearance(full_wake.panels.reshape(-1, 3), ground[0], ground[1])
                except GroundStrikeError as exc:
                    raise GroundStrikeError(f"The wake goes below the ground plane: {exc}") from None
        wake_m = None
        if self.mesh.half:
            wake_m = (
                mirror_panels_y(wake.panels)
                if len(wake.panels) > 0
                else np.empty((0, 4, 3), dtype=float)
            )
        return _Configuration(
            key=key,
            wake_dir=wake_dir,
            ground=ground,
            wake=wake,
            full_wake=full_wake,
            wake_panels_m=wake_m,
        )

    def _parities_needed(self, coeff_rows: list[np.ndarray]) -> list[str]:
        """Return the systems that a set of conditions needs."""
        if not self.mesh.half:
            return ["full"]
        needed = ["sym"]
        if any(np.any(c[list(_ASYM_COLS)] != 0.0) for c in coeff_rows):
            needed.append("asym")
        return needed

    def _ensure(self, cfg: _Configuration, parities: list[str]) -> None:
        """Assemble and factorize the missing systems of a configuration."""
        missing = [p for p in parities if p not in cfg.systems]
        if not missing:
            return
        half = bool(self.mesh.half)
        in_ground = cfg.ground is not None
        self._check_gpu_memory(half, in_ground, len(missing))
        body = self._body()
        has_wake = len(cfg.wake.panels) > 0
        n_panels = len(self.centroids)

        # Self (and mirror) parts with the wake of the configuration.
        a_self = self._copy(body["C_self"])
        if has_wake:
            d_host, d_dev = self._strip_matrix(cfg.wake.panels, cfg.wake)
            cfg.strips["self"] = d_host
            _fold_wake(a_self, d_dev, cfg.wake)
        b_self = body["B_self"]
        if half:
            a_m = self._copy(body["C_m"])
            if has_wake:
                d_host, d_dev = self._strip_matrix(cfg.wake_panels_m, cfg.wake)
                cfg.strips["m"] = d_host
                _fold_wake(a_m, d_dev, cfg.wake)
            b_m = body["B_m"]

        # Ground images (equal source and doublet strengths, reflected panels).
        if in_ground:
            normal, offset = cfg.ground
            b_g, a_g = self._matrices(
                self.centroids, reflect_panels_plane(self.body_verts, normal, offset)
            )
            if has_wake:
                d_host, d_dev = self._strip_matrix(
                    reflect_panels_plane(cfg.wake.panels, normal, offset), cfg.wake
                )
                cfg.strips["g"] = d_host
                _fold_wake(a_g, d_dev, cfg.wake)
            if half:
                b_yg, a_yg = self._matrices(
                    self.centroids, reflect_panels_plane(self._body_verts_m, normal, offset)
                )
                if has_wake:
                    d_host, d_dev = self._strip_matrix(
                        reflect_panels_plane(cfg.wake_panels_m, normal, offset), cfg.wake
                    )
                    cfg.strips["yg"] = d_host
                    _fold_wake(a_yg, d_dev, cfg.wake)

        if cfg.unit_mu is None:
            cfg.unit_mu = np.zeros((n_panels, 6), dtype=np.float64)
        for parity in missing:
            if parity == "full":
                mat_a = a_self if not in_ground else a_self + a_g
                mat_b = b_self if not in_ground else b_self + b_g
                cols = list(range(6))
            else:
                sign = 1.0 if parity == "sym" else -1.0
                if sign > 0.0:
                    mat_a = a_self + a_m
                    mat_b = b_self + b_m
                else:
                    mat_a = a_self - a_m
                    mat_b = b_self - b_m
                if in_ground:
                    if sign > 0.0:
                        mat_a = mat_a + a_g + a_yg
                        mat_b = mat_b + b_g + b_yg
                    else:
                        mat_a = mat_a + a_g - a_yg
                        mat_b = mat_b + b_g - b_yg
                cols = list(_SYM_COLS if parity == "sym" else _ASYM_COLS)
            lu_info = self._factorize(mat_a)
            del mat_a
            cfg.systems[parity] = lu_info
            self._unit_solutions(cfg, lu_info, mat_b, cols, parity)
            del mat_b
        self._update_unit_gradients(cfg)

    def _unit_solutions(self, cfg, lu_info, mat_b, cols, parity) -> None:
        """Solve the unit onset right-hand sides of one system."""
        if self.backend == "gpu":
            import torch

            sigma_t = self._gpu_unit_sigma[:, cols]
            sol = self._lu_solve(lu_info, -mat_b @ sigma_t)
            if cfg.unit_mu_t is None:
                cfg.unit_mu_t = torch.zeros(
                    (len(self.centroids), 6), dtype=self._gpu_dtype, device=self._gpu_device
                )
            cfg.unit_mu_t[:, cols] = sol
            cfg.unit_mu[:, cols] = sol.to(torch.float64).cpu().numpy()
            return
        if parity == "full":
            # Keep the memory layout of the solve output: the superposition
            # products then round as in every earlier version.
            cfg.unit_mu = scipy.linalg.lu_solve(lu_info, -mat_b @ self._unit_sigma)
            return
        for k in cols:
            cfg.unit_mu[:, k] = scipy.linalg.lu_solve(lu_info, -mat_b @ self._unit_sigma[:, k])

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
            )
        else:
            mu_body = unit_mu @ coeffs
            sigma_body = self._unit_sigma @ coeffs
            grad_full = np.einsum("k,nkd->nd", coeffs, cfg.unit_grad)
            v_surf_full, cp_full = compute_surface_velocity_and_cp(
                v_onset_full, self._full_geom.normal, grad_full, v_inf
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
            v_onset_full, self._full_geom.normal, grad_full, v_inf
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
        factorized once.

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
            if gp is not None:
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
                for idx, res in zip(indices, sub_res):
                    results[idx] = self._to_half(res)
                continue
            cfg = self._configs.get(key)
            if cfg is None:
                cfg = self._new_config(key, wake_dir, gp)
                self._configs[key] = cfg
            coeff_rows = [self._condition_coeffs(conds[i]) for i in indices]
            self._ensure(cfg, self._parities_needed(coeff_rows))
            for idx in indices:
                if self.kutta == "morino":
                    results[idx] = self._evaluate_result(cfg, conds[idx])
                else:
                    results[idx] = self._evaluate_pressure_result(cfg, conds[idx])

        return [r for r in results if r is not None]
