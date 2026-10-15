# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Inviscid three-dimensional source-doublet panel solver for Ventorum-3DP.

This module provides the PanelSolver class and the PanelResult dataclass.
It solves the potential flow around arbitrary three-dimensional bodies
and lifting configurations with an internal Dirichlet boundary condition.

Published sources
-----------------
* Morino, L. and Kuo, C. C., 1974, Subsonic potential aerodynamics for
  complex configurations: A general theory, AIAA Journal, Vol. 12, No. 2,
  pp. 191-197.
* Maskew, B., 1987, Program VSAERO Theory Document, NASA CR-4023.
* Katz, J. and Plotkin, A., 2001, Low-Speed Aerodynamics, 2nd ed.,
  Cambridge University Press, section 9.4 and chapter 12.

Conventions
-----------
* Geometry axes: x aft [m], y right [m], z up [m].
* Flight dynamics body axes moments (AVL convention):
  Cl > 0 right wing down, Cm > 0 nose up, Cn > 0 nose right.
* Boundary condition: Zero total perturbation potential inside the body
  (internal Dirichlet boundary condition).
* Kutta condition: Constant doublet wake strips with strength
  mu_w = mu_upper - mu_lower.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import scipy.linalg

from ventorum.aero.system import freestream_direction, rotation_velocity
from ventorum.core.datatypes import FlightCondition
from ventorum.panel3d.kernels import _precompute_geometry, doublet_potential, influence_matrices
from ventorum.panel3d.mesh import PanelMesh
from ventorum.panel3d.post import (
    build_edge_neighbourhood,
    compute_surface_gradient,
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


class PanelSolver:
    """Three-dimensional source-doublet potential flow panel solver.

    This solver uses constant-strength planar source and doublet panels with an
    internal Dirichlet boundary condition. It solves for six unit onset flows
    once. Any flight condition is then evaluated by linear superposition.

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
    ) -> None:
        self.mesh = mesh
        self.far_ratio = float(far_ratio)

        # Standardize triangular panels
        standardized_panels = _standardize_panels(mesh.panels, mesh.nodes)
        body_verts = mesh.nodes[standardized_panels]

        # Precompute body geometry
        geom = _precompute_geometry(body_verts)
        self.centroids = geom.centroid
        self.normals = geom.normal
        self.areas = geom.area

        # Build downstream wake
        self.wake: PanelWake = build_wake(
            mesh.nodes,
            standardized_panels,
            mesh.wake_edges,
            self.normals,
            wake_length=wake_length,
            n_wake=n_wake,
        )

        # Build panel edge neighbourhood for least-squares surface gradient
        self.neighbours = build_edge_neighbourhood(standardized_panels, mesh.wake_edges)

        # Determine reference geometry and moment reference point
        if ref_point is None:
            self.ref_point = np.zeros(3, dtype=np.float64)
        else:
            self.ref_point = np.asarray(ref_point, dtype=np.float64).reshape(3)

        if S_ref is not None:
            self.S_ref = float(S_ref)
        else:
            s_proj = float(np.sum(np.maximum(0.0, self.normals[:, 2]) * self.areas))
            if s_proj > 1e-6:
                self.S_ref = s_proj
            else:
                y_span = float(np.max(mesh.nodes[:, 1]) - np.min(mesh.nodes[:, 1]))
                x_chord = float(np.max(mesh.nodes[:, 0]) - np.min(mesh.nodes[:, 0]))
                self.S_ref = max(np.pi * 0.25 * y_span * x_chord, 1e-6)

        if b_ref is not None:
            self.b_ref = float(b_ref)
        else:
            y_span = float(np.max(mesh.nodes[:, 1]) - np.min(mesh.nodes[:, 1]))
            self.b_ref = max(y_span, 1e-3)

        if c_ref is not None:
            self.c_ref = float(c_ref)
        else:
            self.c_ref = self.S_ref / self.b_ref

        # Assemble aerodynamic influence matrices B and C
        B, C = influence_matrices(self.centroids, body_verts, far_ratio=self.far_ratio)
        self.B = B

        # Incorporate wake doublet influence into matrix A via Morino Kutta condition
        A = np.array(C, copy=True)
        if len(self.wake.panels) > 0:
            c_wake = doublet_potential(self.centroids, self.wake.panels, far_ratio=self.far_ratio)
            for s, strip_p_idx in enumerate(self.wake.strip_panel_indices):
                up_idx = self.wake.wake_up[s]
                lo_idx = self.wake.wake_lo[s]
                c_strip = np.sum(c_wake[:, strip_p_idx], axis=1)
                A[:, up_idx] += c_strip
                A[:, lo_idx] -= c_strip
        self.A = A

        # LU-factorize matrix A once for all solves
        self._lu, self._piv = scipy.linalg.lu_factor(self.A)

        # Solve six unit right-hand sides: u, v, w, p, q, r
        n_panels = len(self.centroids)
        r_rel = self.centroids - self.ref_point[None, :]
        rx = r_rel[:, 0]
        ry = r_rel[:, 1]
        rz = r_rel[:, 2]

        # Unit onset flow velocity fields: shape (6, Np, 3)
        v_unit = np.zeros((6, n_panels, 3), dtype=np.float64)
        # 0: u = 1
        v_unit[0, :, 0] = 1.0
        # 1: v = 1
        v_unit[1, :, 1] = 1.0
        # 2: w = 1
        v_unit[2, :, 2] = 1.0
        # 3: p = 1: omega = [-1, 0, 0], -omega x r = [0, -rz, ry]
        v_unit[3, :, 1] = -rz
        v_unit[3, :, 2] = ry
        # 4: q = 1: omega = [0, 1, 0], -omega x r = [-rz, 0, rx]
        v_unit[4, :, 0] = -rz
        v_unit[4, :, 2] = rx
        # 5: r = 1: omega = [0, 0, -1], -omega x r = [-ry, rx, 0]
        v_unit[5, :, 0] = -ry
        v_unit[5, :, 1] = rx

        # Source strengths for unit onsets: sigma_j = -n_j . V_onset
        self._unit_sigma = -np.sum(self.normals[None, :, :] * v_unit, axis=-1).T

        # Solve for unit doublet strengths: A @ mu = -B @ sigma
        rhs = -self.B @ self._unit_sigma
        self._unit_mu = scipy.linalg.lu_solve((self._lu, self._piv), rhs)

        # Precompute unit surface velocities
        self._unit_v_surf = np.zeros((n_panels, 6, 3), dtype=np.float64)
        for k in range(6):
            grad_k = compute_surface_gradient(
                self.centroids, self.normals, self._unit_mu[:, k], self.neighbours
            )
            v_norm_k = np.sum(v_unit[k] * self.normals, axis=-1, keepdims=True)
            v_tangent_k = v_unit[k] - v_norm_k * self.normals
            self._unit_v_surf[:, k, :] = v_tangent_k - grad_k

    def _condition_velocity(
        self,
        condition: FlightCondition,
        points: np.ndarray,
    ) -> np.ndarray:
        """Return local onset flow velocity at points for the condition.

        Parameters
        ----------
        condition : FlightCondition
            Flight condition.
        points : numpy.ndarray
            Coordinates of evaluation points with shape (M, 3) in m.

        Returns
        -------
        numpy.ndarray
            Local onset flow velocities with shape (M, 3) in m/s.
        """
        alpha = float(condition.alpha)
        beta = float(condition.beta)
        v_inf = float(condition.V_inf)
        d = freestream_direction(alpha, beta)
        v_trans = v_inf * d[None, :]

        w_rot = rotation_velocity(condition, points, self.ref_point)
        if w_rot is not None:
            return v_trans + w_rot
        return np.broadcast_to(v_trans, points.shape)

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

        # Linear superposition coefficients
        coeffs = np.array([u, v, w, p, q, r], dtype=np.float64)

        mu = self._unit_mu @ coeffs
        sigma = self._unit_sigma @ coeffs
        v_surf = np.einsum("k,nkd->nd", coeffs, self._unit_v_surf)

        # Compute surface pressure coefficient
        if v_inf > 1e-12:
            speed_sq = np.sum(v_surf**2, axis=-1)
            cp = 1.0 - speed_sq / (v_inf**2)
        else:
            cp = np.zeros(len(self.centroids), dtype=np.float64)

        # Integrate forces and moments
        forces = integrate_forces_and_moments(
            cp=cp,
            areas=self.areas,
            normals=self.normals,
            centroids=self.centroids,
            condition=condition,
            S_ref=self.S_ref,
            b_ref=self.b_ref,
            c_ref=self.c_ref,
            ref_point=self.ref_point,
        )

        # Evaluate far-field Trefftz plane induced drag
        cdi_trefftz = compute_trefftz_induced_drag(
            wake=self.wake,
            mu=mu,
            V_inf=v_inf,
            rho=float(condition.rho),
            S_ref=self.S_ref,
        )

        return PanelResult(
            CL=forces["CL"],
            CD_pressure=forces["CD_pressure"],
            CDi_trefftz=cdi_trefftz,
            CY=forces["CY"],
            Cl=forces["Cl"],
            Cm=forces["Cm"],
            Cn=forces["Cn"],
            cp=cp,
            mu=mu,
            sigma=sigma,
            velocity=v_surf,
            condition=condition,
        )

    def solve_conditions(self, conditions: Sequence[FlightCondition]) -> list[PanelResult]:
        """Solve potential flow for multiple flight conditions.

        Parameters
        ----------
        conditions : sequence of FlightCondition
            Flight conditions to solve.

        Returns
        -------
        list of PanelResult
            List of solution results corresponding to each condition.
        """
        return [self.solve(cond) for cond in conditions]
