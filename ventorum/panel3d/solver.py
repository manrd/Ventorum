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

Kutta options
-------------
* "morino" (default): linear Morino condition, unchanged results.
* "pressure": Newton iteration on the wake strip strengths with residual
  Cp_upper - Cp_lower at each trailing-edge pair, starting from Morino.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import warnings

import numpy as np
import scipy.linalg

from ventorum.aero.system import freestream_direction, rotation_velocity
from ventorum.core.datatypes import FlightCondition
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
    build_edge_neighbourhood,
    compute_surface_gradient,
    compute_surface_velocity_and_cp,
    compute_trefftz_induced_drag,
    integrate_forces_and_moments,
)
from ventorum.panel3d.wake import PanelWake, build_wake

__all__ = ["PanelResult", "PanelSolver"]


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


def _add_wake_influence(
    A: np.ndarray,
    Cw: np.ndarray,
    wake: PanelWake,
    sign: float = 1.0,
) -> np.ndarray:
    """Add wake strip doublet influences to matrix A."""
    for s, strip_p_idx in enumerate(wake.strip_panel_indices):
        up_idx = wake.wake_up[s]
        lo_idx = wake.wake_lo[s]
        cs = np.sum(Cw[:, strip_p_idx], axis=1) * sign
        A[:, up_idx] += cs
        A[:, lo_idx] -= cs
    return A


class PanelSolver:
    """Three-dimensional source-doublet potential flow panel solver.

    This solver uses constant-strength planar source and doublet panels with an
    internal Dirichlet boundary condition. It supports symmetry folding on half
    meshes (PanelMesh.half=True) and ground effect using the method of images.

    Parameters
    ----------
    mesh : PanelMesh
        Surface mesh containing nodes, panels, and wake shedding edges.
    wake_length : float or None, optional
        Length of the downstream body-axis wake in m. If None, defaults to
        50 times the maximum chord of the mesh.
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
    ) -> None:
        if kutta not in ("morino", "pressure"):
            raise ValueError(f"Unknown kutta option {kutta!r}; use 'morino' or 'pressure'.")
        self.kutta = str(kutta)
        self.mesh = mesh
        self.far_ratio = float(far_ratio)

        # Standardize triangular panels on given mesh
        self.standardized_panels = _standardize_panels(mesh.panels, mesh.nodes)
        self.body_verts = mesh.nodes[self.standardized_panels]

        # Precompute body geometry on given mesh
        geom = _precompute_geometry(self.body_verts)
        self.centroids = geom.centroid
        self.normals = geom.normal
        self.areas = geom.area

        # Build downstream wake on given mesh
        self.wake: PanelWake = build_wake(
            mesh.nodes,
            self.standardized_panels,
            mesh.wake_edges,
            self.normals,
            wake_length=wake_length,
            n_wake=n_wake,
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
        self._full_wake = build_wake(
            self._full_mesh.nodes,
            self._standardized_full_panels,
            self._full_mesh.wake_edges,
            self._full_geom.normal,
            wake_length=wake_length,
            n_wake=n_wake,
        )
        self._full_neighbours = build_edge_neighbourhood(
            self._standardized_full_panels, self._full_mesh.wake_edges
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

        # Influence matrices for free air
        if not self.mesh.half:
            # Full mesh solve
            B, C = influence_matrices(self.centroids, self.body_verts, far_ratio=self.far_ratio)
            self.B = B
            A = np.array(C, copy=True)
            if len(self.wake.panels) > 0:
                Cw = doublet_potential(self.centroids, self.wake.panels, far_ratio=self.far_ratio)
                _add_wake_influence(A, Cw, self.wake, sign=1.0)
            self.A = A
            self._free_lu, self._free_piv = scipy.linalg.lu_factor(self.A)
            self._free_unit_mu = scipy.linalg.lu_solve(
                (self._free_lu, self._free_piv), -self.B @ self._unit_sigma
            )
            self._free_unit_grad = self._compute_full_unit_gradients(self._free_unit_mu)
        else:
            # Half mesh with symmetry folding
            self._body_verts_m = mirror_panels_y(self.body_verts)
            B_self, C_self = influence_matrices(
                self.centroids, self.body_verts, far_ratio=self.far_ratio
            )
            B_m, C_m = influence_matrices(
                self.centroids, self._body_verts_m, far_ratio=self.far_ratio
            )

            A_self = np.array(C_self, copy=True)
            A_m = np.array(C_m, copy=True)
            if len(self.wake.panels) > 0:
                self._wake_panels_m = mirror_panels_y(self.wake.panels)
                Cw_self = doublet_potential(
                    self.centroids, self.wake.panels, far_ratio=self.far_ratio
                )
                Cw_m = doublet_potential(
                    self.centroids, self._wake_panels_m, far_ratio=self.far_ratio
                )
                _add_wake_influence(A_self, Cw_self, self.wake, sign=1.0)
                _add_wake_influence(A_m, Cw_m, self.wake, sign=1.0)
            else:
                self._wake_panels_m = np.empty((0, 4, 3), dtype=float)

            self._A_self = A_self
            self._B_self = B_self
            self._A_m = A_m
            self._B_m = B_m

            # Free-air factorizations (symmetric and antisymmetric)
            A_sym = A_self + A_m
            B_sym = B_self + B_m
            A_asym = A_self - A_m
            B_asym = B_self - B_m

            self._free_lu_sym, self._free_piv_sym = scipy.linalg.lu_factor(A_sym)
            self._free_lu_asym, self._free_piv_asym = scipy.linalg.lu_factor(A_asym)

            unit_mu = np.zeros((n_panels, 6), dtype=np.float64)
            for k in [0, 2, 4]:  # Symmetric: u, w, q
                unit_mu[:, k] = scipy.linalg.lu_solve(
                    (self._free_lu_sym, self._free_piv_sym), -B_sym @ self._unit_sigma[:, k]
                )
            for k in [1, 3, 5]:  # Asymmetric: v, p, r
                unit_mu[:, k] = scipy.linalg.lu_solve(
                    (self._free_lu_asym, self._free_piv_asym), -B_asym @ self._unit_sigma[:, k]
                )
            self._free_unit_mu = unit_mu
            self._free_unit_grad = self._compute_full_unit_gradients(self._free_unit_mu)

    def _compute_full_unit_gradients(self, unit_mu_half: np.ndarray) -> np.ndarray:
        """Precompute surface gradient fields on the full mesh for the 6 unit onsets."""
        n_full = len(self._full_geom.centroid)
        unit_grad = np.zeros((n_full, 6, 3), dtype=np.float64)
        if not self.mesh.half:
            for k in range(6):
                unit_grad[:, k, :] = compute_surface_gradient(
                    self._full_geom.centroid,
                    self._full_geom.normal,
                    unit_mu_half[:, k],
                    self._full_neighbours,
                )
        else:
            # Reconstruct full mu for each unit onset
            for k in range(6):
                is_sym = k in (0, 2, 4)
                mu_right = unit_mu_half[:, k]
                mu_left = mu_right if is_sym else -mu_right
                mu_full_k = np.concatenate([mu_right, mu_left])
                unit_grad[:, k, :] = compute_surface_gradient(
                    self._full_geom.centroid,
                    self._full_geom.normal,
                    mu_full_k,
                    self._full_neighbours,
                )
        return unit_grad

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

    def _factorize_ground_group(
        self,
        normal: np.ndarray,
        offset: float,
    ) -> tuple[tuple, np.ndarray]:
        """Compute factorization and unit doublet strengths for a ground configuration."""
        if not self.mesh.half:
            body_verts_g = reflect_panels_plane(self.body_verts, normal, offset)
            B_g, C_g = influence_matrices(
                self.centroids, body_verts_g, far_ratio=self.far_ratio
            )
            A_g = np.array(C_g, copy=True)
            if len(self.wake.panels) > 0:
                wake_panels_g = reflect_panels_plane(self.wake.panels, normal, offset)
                Cw_g = doublet_potential(
                    self.centroids, wake_panels_g, far_ratio=self.far_ratio
                )
                _add_wake_influence(A_g, Cw_g, self.wake, sign=1.0)

            A_tot = self.A + A_g
            B_tot = self.B + B_g
            lu, piv = scipy.linalg.lu_factor(A_tot)
            unit_mu = scipy.linalg.lu_solve((lu, piv), -B_tot @ self._unit_sigma)
            unit_grad = self._compute_full_unit_gradients(unit_mu)
            return (lu, piv), unit_mu, unit_grad

        # Half mesh: four-fold images
        body_verts_g = reflect_panels_plane(self.body_verts, normal, offset)
        body_verts_yg = reflect_panels_plane(self._body_verts_m, normal, offset)

        B_g, C_g = influence_matrices(self.centroids, body_verts_g, far_ratio=self.far_ratio)
        B_yg, C_yg = influence_matrices(
            self.centroids, body_verts_yg, far_ratio=self.far_ratio
        )

        A_g = np.array(C_g, copy=True)
        A_yg = np.array(C_yg, copy=True)
        if len(self.wake.panels) > 0:
            wake_panels_g = reflect_panels_plane(self.wake.panels, normal, offset)
            wake_panels_yg = reflect_panels_plane(self._wake_panels_m, normal, offset)
            Cw_g = doublet_potential(self.centroids, wake_panels_g, far_ratio=self.far_ratio)
            Cw_yg = doublet_potential(
                self.centroids, wake_panels_yg, far_ratio=self.far_ratio
            )
            _add_wake_influence(A_g, Cw_g, self.wake, sign=1.0)
            _add_wake_influence(A_yg, Cw_yg, self.wake, sign=1.0)

        A_sym = self._A_self + self._A_m + A_g + A_yg
        B_sym = self._B_self + self._B_m + B_g + B_yg

        A_asym = self._A_self - self._A_m + A_g - A_yg
        B_asym = self._B_self - self._B_m + B_g - B_yg

        lu_sym, piv_sym = scipy.linalg.lu_factor(A_sym)
        lu_asym, piv_asym = scipy.linalg.lu_factor(A_asym)

        n_panels = len(self.centroids)
        unit_mu = np.zeros((n_panels, 6), dtype=np.float64)
        for k in [0, 2, 4]:
            unit_mu[:, k] = scipy.linalg.lu_solve(
                (lu_sym, piv_sym), -B_sym @ self._unit_sigma[:, k]
            )
        for k in [1, 3, 5]:
            unit_mu[:, k] = scipy.linalg.lu_solve(
                (lu_asym, piv_asym), -B_asym @ self._unit_sigma[:, k]
            )

        unit_grad = self._compute_full_unit_gradients(unit_mu)
        return (lu_sym, piv_sym, lu_asym, piv_asym), unit_mu, unit_grad

    def _evaluate_result(
        self,
        condition: FlightCondition,
        unit_mu: np.ndarray,
        unit_grad: np.ndarray,
    ) -> PanelResult:
        """Evaluate aerodynamic coefficients and distributions for a condition."""
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

        coeffs = np.array([u, v, w, p, q, r], dtype=np.float64)
        mu_body = unit_mu @ coeffs
        sigma_body = self._unit_sigma @ coeffs

        # Superpose gradient field on the full mesh
        grad_full = np.einsum("k,nkd->nd", coeffs, unit_grad)
        v_onset_full = self._condition_velocity(condition, self._full_geom.centroid)
        v_surf_full, cp_full = compute_surface_velocity_and_cp(
            v_onset_full, self._full_geom.normal, grad_full, v_inf
        )

        # Full mesh forces and moments
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

        # Trefftz plane induced drag on full wake
        if not self.mesh.half:
            mu_full = mu_body
        else:
            mu_sym = unit_mu[:, [0, 2, 4]] @ coeffs[[0, 2, 4]]
            mu_asym = unit_mu[:, [1, 3, 5]] @ coeffs[[1, 3, 5]]
            mu_left = mu_sym - mu_asym
            mu_full = np.concatenate([mu_body, mu_left])

        cdi_trefftz = compute_trefftz_induced_drag(
            wake=self._full_wake,
            mu=mu_full,
            V_inf=v_inf,
            rho=float(condition.rho),
            S_ref=self.S_ref,
        )

        n_panels = len(self.centroids)
        cp_out = cp_full[:n_panels]
        v_surf_out = v_surf_full[:n_panels]

        up_te = np.asarray(self.wake.wake_up, dtype=int)
        lo_te = np.asarray(self.wake.wake_lo, dtype=int)
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
        mu_sym = unit_mu[:, [0, 2, 4]] @ coeffs[[0, 2, 4]]
        mu_asym = unit_mu[:, [1, 3, 5]] @ coeffs[[1, 3, 5]]
        mu_left = mu_sym - mu_asym
        return np.concatenate([mu_body, mu_left])

    def _strip_matrix_free(self) -> np.ndarray:
        """Sum wake doublet influences into one column per solved strip in free air."""
        n_solve = len(self.centroids)
        strips = self.wake.strip_panel_indices
        n_strips = len(strips)
        if n_strips == 0:
            return np.zeros((n_solve, 0), dtype=float)
        if not self.mesh.half:
            cw = doublet_potential(
                self.centroids, self.wake.panels, far_ratio=self.far_ratio
            )
            return strip_influence_matrix(cw, strips)
        cw_self = doublet_potential(
            self.centroids, self.wake.panels, far_ratio=self.far_ratio
        )
        cw_m = doublet_potential(
            self.centroids, self._wake_panels_m, far_ratio=self.far_ratio
        )
        d_self = strip_influence_matrix(cw_self, strips)
        d_m = strip_influence_matrix(cw_m, strips)
        return d_self + d_m

    def _strip_matrix_ground(self, normal: np.ndarray, offset: float) -> np.ndarray:
        """Sum wake and image influences into one column per solved strip."""
        n_solve = len(self.centroids)
        strips = self.wake.strip_panel_indices
        n_strips = len(strips)
        if n_strips == 0:
            return np.zeros((n_solve, 0), dtype=float)
        if not self.mesh.half:
            cw = doublet_potential(
                self.centroids, self.wake.panels, far_ratio=self.far_ratio
            )
            wake_g = reflect_panels_plane(self.wake.panels, normal, offset)
            cw_g = doublet_potential(self.centroids, wake_g, far_ratio=self.far_ratio)
            return strip_influence_matrix(cw, strips) + strip_influence_matrix(cw_g, strips)
        cw_self = doublet_potential(
            self.centroids, self.wake.panels, far_ratio=self.far_ratio
        )
        cw_m = doublet_potential(
            self.centroids, self._wake_panels_m, far_ratio=self.far_ratio
        )
        wake_g = reflect_panels_plane(self.wake.panels, normal, offset)
        wake_yg = reflect_panels_plane(self._wake_panels_m, normal, offset)
        cw_g = doublet_potential(self.centroids, wake_g, far_ratio=self.far_ratio)
        cw_yg = doublet_potential(self.centroids, wake_yg, far_ratio=self.far_ratio)
        d_mat = strip_influence_matrix(cw_self, strips)
        d_mat = d_mat + strip_influence_matrix(cw_m, strips)
        d_mat = d_mat + strip_influence_matrix(cw_g, strips)
        d_mat = d_mat + strip_influence_matrix(cw_yg, strips)
        return d_mat

    def _pressure_lu(self, lu_info) -> tuple:
        """Select the reused factorization for the pressure update."""
        if not self.mesh.half:
            return lu_info
        lu_sym, piv_sym, _, _ = lu_info
        return (lu_sym, piv_sym)

    def _evaluate_pressure_result(
        self,
        condition: FlightCondition,
        unit_mu: np.ndarray,
        lu_info,
        d_solve: np.ndarray,
    ) -> PanelResult:
        """Evaluate one flight condition with the iterative pressure Kutta."""
        coeffs = self._condition_coeffs(condition)
        mu_solve0 = unit_mu @ coeffs
        sigma_body = self._unit_sigma @ coeffs
        mu_full0 = self._reconstruct_full_mu(unit_mu, coeffs)
        v_inf = float(condition.V_inf)
        v_onset_full = self._condition_velocity(condition, self._full_geom.centroid)

        lu = self._pressure_lu(lu_info)
        up_te = np.asarray(self.wake.wake_up, dtype=int)
        lo_te = np.asarray(self.wake.wake_lo, dtype=int)

        mu_solve, mu_full, delta, n_iters, resid, converged = solve_pressure_kutta(
            mu_solve0,
            mu_full0,
            lu,
            d_solve,
            up_te,
            lo_te,
            self._full_geom.centroid,
            self._full_geom.normal,
            self._full_neighbours,
            v_onset_full,
            v_inf,
        )

        if up_te.size:
            mu_wake = (mu_solve[up_te] - mu_solve[lo_te]) + delta
        else:
            mu_wake = np.zeros(0, dtype=float)

        trust_note = ""
        if not converged:
            msg = (
                "Pressure Kutta did not converge in 20 iterations; "
                f"final max|delta Cp|={resid:.3e}. Treat the trailing edge with care."
            )
            warnings.warn(msg, RuntimeWarning, stacklevel=3)
            trust_note = msg

        grad_full = compute_surface_gradient(
            self._full_geom.centroid, self._full_geom.normal, mu_full, self._full_neighbours
        )
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

        # Trefftz drag uses the corrected wake strengths through an adjusted
        # doublet vector that keeps the surface state unchanged elsewhere.
        mu_treff = np.array(mu_full, dtype=float, copy=True)
        if delta.size and len(self._full_wake.wake_up):
            up_full = np.asarray(self._full_wake.wake_up, dtype=int)
            lo_full = np.asarray(self._full_wake.wake_lo, dtype=int)
            n_half_strips = int(delta.shape[0])
            for s in range(n_half_strips):
                d_half = float(delta[s]) / 2.0
                mu_treff[up_full[s]] += d_half
                mu_treff[lo_full[s]] -= d_half
                mirror = s + n_half_strips
                if mirror < up_full.shape[0]:
                    mu_treff[up_full[mirror]] += d_half
                    mu_treff[lo_full[mirror]] -= d_half
        cdi_trefftz = compute_trefftz_induced_drag(
            wake=self._full_wake,
            mu=mu_treff,
            V_inf=v_inf,
            rho=float(condition.rho),
            S_ref=self.S_ref,
        )

        n_panels = len(self.centroids)
        cp_out = cp_full[:n_panels]
        v_surf_out = v_surf_full[:n_panels]

        return PanelResult(
            CL=forces["CL"],
            CD_pressure=forces["CD_pressure"],
            CDi_trefftz=cdi_trefftz,
            CY=forces["CY"],
            Cl=forces["Cl"],
            Cm=forces["Cm"],
            Cn=forces["Cn"],
            cp=cp_out,
            mu=np.array(mu_solve, dtype=float, copy=True),
            sigma=sigma_body,
            velocity=v_surf_out,
            condition=condition,
            kutta="pressure",
            kutta_iters=int(n_iters),
            kutta_residual=float(resid),
            kutta_converged=bool(converged),
            trust_note=trust_note,
            mu_w=np.array(mu_wake, dtype=float, copy=True),
        )

    def solve(self, condition: FlightCondition) -> PanelResult:
        """Solve potential flow for a given flight condition.

        Parameters
        ----------
        condition : FlightCondition
            Flight condition to evaluate.

        Returns
        -------
        PanelResult
            Forces, moments, pressure, doublet and source distributions.
        """
        return self.solve_conditions([condition])[0]

    def solve_conditions(self, conditions: Sequence[FlightCondition]) -> list[PanelResult]:
        """Solve potential flow for multiple flight conditions.

        In ground effect, conditions are grouped by (attitude, height) and
        factorized once per group.

        Parameters
        ----------
        conditions : sequence of FlightCondition
            Flight conditions to solve.

        Returns
        -------
        list of PanelResult
            List of solution results corresponding to each condition.
        """
        if not conditions:
            return []

        # Validate clearances and build grouping keys
        groups: dict[tuple, list[int]] = {}
        for idx, cond in enumerate(conditions):
            if cond.h is None:
                key = ("free",)
            else:
                gp = ground_plane_parameters(cond, self.ref_point)
                normal, offset = gp
                check_ground_clearance(self._full_mesh.nodes, normal, offset)
                a_round = round(float(cond.alpha), 11)
                b_round = round(float(cond.beta), 11)
                p_round = round(float(getattr(cond, "phi", 0.0)), 11)
                h_round = round(float(cond.h), 11)
                key = ("ground", a_round, b_round, p_round, h_round)
            groups.setdefault(key, []).append(idx)

        results: list[PanelResult | None] = [None] * len(conditions)

        for key, indices in groups.items():
            if key[0] == "free":
                if self.kutta == "morino":
                    unit_mu = self._free_unit_mu
                    unit_grad = self._free_unit_grad
                    for idx in indices:
                        results[idx] = self._evaluate_result(conditions[idx], unit_mu, unit_grad)
                    continue
                unit_mu = self._free_unit_mu
                if not self.mesh.half:
                    lu_info = (self._free_lu, self._free_piv)
                else:
                    lu_info = (
                        self._free_lu_sym,
                        self._free_piv_sym,
                        self._free_lu_asym,
                        self._free_piv_asym,
                    )
                d_free = self._strip_matrix_free()
                for idx in indices:
                    results[idx] = self._evaluate_pressure_result(
                        conditions[idx], unit_mu, lu_info, d_free
                    )
                continue

            first_cond = conditions[indices[0]]
            normal, offset = ground_plane_parameters(first_cond, self.ref_point)
            lu_info, unit_mu, unit_grad = self._factorize_ground_group(normal, offset)
            if self.kutta == "morino":
                for idx in indices:
                    results[idx] = self._evaluate_result(conditions[idx], unit_mu, unit_grad)
                continue
            d_ground = self._strip_matrix_ground(np.asarray(normal, dtype=float), float(offset))
            for idx in indices:
                results[idx] = self._evaluate_pressure_result(
                    conditions[idx], unit_mu, lu_info, d_ground
                )

        return [r for r in results if r is not None]
