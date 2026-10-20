"""Two-dimensional discrete vortices on a real mean line.

Sources: J. Katz and A. Plotkin, Low-Speed Aerodynamics, 2nd ed., 2001
(lumped vortex); C. Coulliette and A. Plotkin, The Aeronautical Journal
100, 1996 (cambered plate with images); S. E. Widnall and T. M. Barrows,
J. Fluid Mech. 41, 1970 (limits of the mean-plane approximation).
This independent reference uses unit chord [m], speed [m/s] and density
[kg/m^3]. It does not use the production vortex kernels.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def cambered_plate(
    mean_line: Callable, slope: Callable, *, alpha: float = 0.0,
    h: float | None = None, n_panels: int = 64, explicit_mirror: bool = False,
) -> dict:
    """Solve a unit-chord cambered plate with lumped point vortices.

    Parameters
    ----------
    mean_line : callable
        Mean-line height [chord units] at chord fractions [-].
    slope : callable
        Mean-line slope [-] at chord fractions [-].
    alpha : float
        Angle of attack [rad].
    h : float or None
        Quarter-chord height [m] above a plane parallel to the flow.
        None selects free air.
    n_panels : int
        Number of uniform chordwise panels [-].
    explicit_mirror : bool
        Solve both real and reflected plates instead of image reduction.

    Returns
    -------
    dict
        ``CL`` and quarter-chord ``Cm`` [-], ``gamma`` [m^2/s],
        vortex ``points`` [m] and ``control_points`` [m], in (x, z) axes.
    """
    if int(n_panels) != n_panels or n_panels < 1:
        raise ValueError("n_panels must be a positive integer.")
    if h is not None and (not np.isfinite(h) or h <= 0.0):
        raise ValueError("h must be positive and finite [m].")
    edges = np.linspace(0.0, 1.0, n_panels + 1)
    dx = np.diff(edges)
    xv = edges[:-1] + 0.25 * dx
    xc = edges[:-1] + 0.75 * dx
    q = np.column_stack([xv, mean_line(xv)])
    p = np.column_stack([xc, mean_line(xc)])
    normals = np.column_stack([-np.asarray(slope(xc)), np.ones(n_panels)])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    d = np.array([np.cos(alpha), np.sin(alpha)])
    k = np.array([-np.sin(alpha), np.cos(alpha)])
    rp = np.array([0.25, 0.0])

    def velocity(target, source):
        r = target[:, None, :] - source[None, :, :]
        r2 = np.sum(r * r, axis=-1)
        inv = np.divide(1.0, 2.0 * np.pi * r2, out=np.zeros_like(r2), where=r2 > 0.0)
        return np.stack([r[..., 1] * inv, -r[..., 0] * inv], axis=-1)

    def reflect(points):
        height = (points - rp) @ k + h
        return points - 2.0 * height[:, None] * k

    def reflect_vectors(vectors):
        return vectors - 2.0 * (vectors @ k)[:, None] * k

    if h is not None and np.min((np.vstack([p, q]) - rp) @ k + h) <= 0.0:
        raise ValueError("The cambered plate touches the ground.")
    if h is not None and explicit_mirror:
        pp, qq = np.vstack([p, reflect(p)]), np.vstack([q, reflect(q)])
        nn = np.vstack([normals, reflect_vectors(normals)])
        A = np.einsum("ijk,ik->ij", velocity(pp, qq), nn)
        all_gamma = np.linalg.solve(A, -nn @ d)
        gamma = all_gamma[:n_panels]
        v = np.einsum("ijk,j->ik", velocity(q, qq), all_gamma)
    else:
        v_cp = velocity(p, q)
        v_q = velocity(q, q)
        if h is not None:
            v_cp -= velocity(p, reflect(q))
            v_q -= velocity(q, reflect(q))
        A = np.einsum("ijk,ik->ij", v_cp, normals)
        gamma = np.linalg.solve(A, -normals @ d)
        v = np.einsum("ijk,j->ik", v_q, gamma)
    onset = d + v
    force = gamma[:, None] * np.column_stack([-onset[:, 1], onset[:, 0]])
    moment = q[:, 1] * force[:, 0] - (q[:, 0] - 0.25) * force[:, 1]
    return {"CL": float(2.0 * np.sum(force @ k)), "Cm": float(2.0 * np.sum(moment)),
            "gamma": gamma, "points": q, "control_points": p}
