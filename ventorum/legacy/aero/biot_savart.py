# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Biot-Savart law implementations for vortex-filament induced velocities.

Three primitives are provided:

1. **finite_vortex_velocity** - straight segment from A to B.
2. **semi_infinite_vortex_velocity** - ray from A to infinity along d̂.
3. **horseshoe_velocity** - bound segment + two trailing legs.

All functions accept and return 3-D vectors as ``np.ndarray`` of shape (3,).
A small vortex-core radius is used to regularise the singular Biot-Savart
kernel when the evaluation point is very close to a filament.
"""

from __future__ import annotations

import numpy as np

from ventorum.legacy.core.constants import VORTEX_CORE_RADIUS


# ═══════════════════════════════════════════════════════════════════════════════
# Primitive: finite vortex segment
# ═══════════════════════════════════════════════════════════════════════════════

def finite_vortex_velocity(
    A: np.ndarray,
    B: np.ndarray,
    P: np.ndarray,
    Gamma: float = 1.0,
    rc: float = VORTEX_CORE_RADIUS,
) -> np.ndarray:
    """Velocity induced at *P* by a finite vortex segment from *A* to *B*.

    The circulation *Γ* is positive in the direction from *A* to *B* (by
    the right-hand rule, induced velocity curls around the segment
    accordingly).

    Parameters
    ----------
    A, B : np.ndarray, shape (3,)
        Segment endpoints.
    P : np.ndarray, shape (3,)
        Evaluation point.
    Gamma : float
        Segment circulation [m²/s].
    rc : float
        Vortex-core regularisation radius [m].

    Returns
    -------
    V : np.ndarray, shape (3,)
        Induced velocity [m/s].
    """
    r1 = P - A
    r2 = P - B
    r0 = B - A

    cross = np.array([
        r1[1] * r2[2] - r1[2] * r2[1],
        r1[2] * r2[0] - r1[0] * r2[2],
        r1[0] * r2[1] - r1[1] * r2[0],
    ])
    cross_sq = cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]
    r0_sq = r0[0] * r0[0] + r0[1] * r0[1] + r0[2] * r0[2]
    rc_sq = rc * rc

    # Vatistas/Scully regularisation: prevents discontinuous jumps and 1/r singularity
    denom = cross_sq + r0_sq * rc_sq
    if denom <= 0.0:
        return np.zeros(3)

    r1_norm_reg = np.sqrt(r1[0] * r1[0] + r1[1] * r1[1] + r1[2] * r1[2] + rc_sq)
    r2_norm_reg = np.sqrt(r2[0] * r2[0] + r2[1] * r2[1] + r2[2] * r2[2] + rc_sq)

    diff = r1 / r1_norm_reg - r2 / r2_norm_reg
    dot_term = r0[0] * diff[0] + r0[1] * diff[1] + r0[2] * diff[2]

    return cross * ((Gamma / (4.0 * np.pi)) * dot_term / denom)


# ═══════════════════════════════════════════════════════════════════════════════
# Primitive: semi-infinite vortex ray
# ═══════════════════════════════════════════════════════════════════════════════

def semi_infinite_vortex_velocity(
    A: np.ndarray,
    d_hat: np.ndarray,
    P: np.ndarray,
    Gamma: float = 1.0,
    rc: float = VORTEX_CORE_RADIUS,
) -> np.ndarray:
    """Velocity induced at *P* by a semi-infinite vortex starting at *A* and
    extending to infinity along direction *d̂*.

    Parameters
    ----------
    A : np.ndarray, shape (3,)
        Starting point of the ray.
    d_hat : np.ndarray, shape (3,)
        Unit direction toward infinity (must be normalised).
    P : np.ndarray, shape (3,)
        Evaluation point.
    Gamma : float
        Circulation [m²/s].
    rc : float
        Regularisation radius.

    Returns
    -------
    V : np.ndarray, shape (3,)
        Induced velocity [m/s].
    """
    r = P - A
    cross = np.array([
        d_hat[1] * r[2] - d_hat[2] * r[1],
        d_hat[2] * r[0] - d_hat[0] * r[2],
        d_hat[0] * r[1] - d_hat[1] * r[0],
    ])
    cross_sq = cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]
    rc_sq = rc * rc

    denom = cross_sq + rc_sq
    if denom <= 0.0:
        return np.zeros(3)

    r_norm_reg = np.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + rc_sq)
    cos_theta = (d_hat[0] * r[0] + d_hat[1] * r[1] + d_hat[2] * r[2]) / r_norm_reg

    return cross * ((Gamma / (4.0 * np.pi)) * (1.0 + cos_theta) / denom)


# ═══════════════════════════════════════════════════════════════════════════════
# Composite: horseshoe vortex
# ═══════════════════════════════════════════════════════════════════════════════

def horseshoe_velocity(
    node_left: np.ndarray,
    node_right: np.ndarray,
    trailing_dir: np.ndarray,
    P: np.ndarray,
    Gamma: float = 1.0,
    rc: float = VORTEX_CORE_RADIUS,
) -> np.ndarray:
    """Velocity induced at *P* by a complete horseshoe vortex.

    The horseshoe consists of:

    * A **bound segment** from *node_left* to *node_right* (circulation in
      the left → right direction).
    * A **left trailing leg** from far downstream to *node_left* (arriving
      from infinity along *trailing_dir*).
    * A **right trailing leg** from *node_right* to far downstream (departing
      along *trailing_dir*).

    The loop direction is:  left_∞ → node_left → node_right → right_∞.

    Parameters
    ----------
    node_left, node_right : np.ndarray, shape (3,)
        Bound-segment endpoints (at the 1/4-chord).
    trailing_dir : np.ndarray, shape (3,)
        Unit vector in the free-stream direction (downstream).
    P : np.ndarray, shape (3,)
        Evaluation point.
    Gamma : float
        Horseshoe circulation [m²/s].
    rc : float
        Regularisation radius.

    Returns
    -------
    V : np.ndarray, shape (3,)
        Total induced velocity [m/s].
    """
    # Bound segment: node_left → node_right
    V_bound = finite_vortex_velocity(node_left, node_right, P, Gamma, rc)

    # Left trailing leg: from ∞ to node_left along trailing_dir.
    # Equivalent to *minus* the semi-infinite ray from node_left in +trailing_dir.
    V_left = -semi_infinite_vortex_velocity(node_left, trailing_dir, P, Gamma, rc)

    # Right trailing leg: from node_right to ∞ along trailing_dir.
    V_right = semi_infinite_vortex_velocity(node_right, trailing_dir, P, Gamma, rc)

    return V_bound + V_left + V_right


# ═══════════════════════════════════════════════════════════════════════════════
# Vectorised convenience
# ═══════════════════════════════════════════════════════════════════════════════

def horseshoe_velocity_batch(
    nodes_left: np.ndarray,
    nodes_right: np.ndarray,
    trailing_dir: np.ndarray,
    P: np.ndarray,
    rc: float = VORTEX_CORE_RADIUS,
) -> np.ndarray:
    """Induced velocity at *P* from a batch of *M* unit-strength horseshoe
    vortices.

    Parameters
    ----------
    nodes_left : np.ndarray, shape (M, 3)
    nodes_right : np.ndarray, shape (M, 3)
    trailing_dir : np.ndarray, shape (3,)
    P : np.ndarray, shape (3,)
    rc : float

    Returns
    -------
    V : np.ndarray, shape (M, 3)
        Induced velocity from each horseshoe (Γ = 1).
    """
    from ventorum.legacy.aero.influence import compute_horseshoe_velocity_matrix
    return compute_horseshoe_velocity_matrix(
        P[np.newaxis, :], nodes_left, nodes_right, trailing_dir, gamma=1.0, rc=rc
    )[0]
