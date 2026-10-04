# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Moment-coefficient axes: body, stability and wind.

The three right-handed axis systems share the geometry axes of ``AGENTS.md``
(x aft, y right, z up) and the AVL moment signs (Cl > 0 right wing down,
Cm > 0 nose up, Cn > 0 nose right):

* ``body``: fixed to the aircraft. This is the default of every result.
* ``stability``: the body axes turned by the angle of attack about the y
  axis, so that x lies along the projection of the free stream on the plane
  of symmetry.
* ``wind``: the stability axes turned by the sideslip angle about the z
  axis, so that x lies along the free stream.

With these signs the triple (Cl, Cm, Cn) is the moment vector written in
the standard flight-dynamics body axes (x forward, y right, z down), so the
standard rotations below apply. The definitions follow the standard flight-dynamics axes (Stevens, Lewis
and Johnson, Aircraft Control and Simulation, 3rd ed., Wiley, 2016,
Chap. 2). The force coefficients CL, CD and CY keep their present
definitions (relative to the free stream) in every set; only the moment
coefficients Cl, Cm and Cn change with the axes.

Rotation matrices (alpha and beta in rad, ca = cos(alpha) and so on).
A moment vector M = (Cl, Cm, Cn) transforms as M_out = R M_in.

Body to stability (rotation by alpha about y)::

    R_bs = [[ca, 0, sa],
            [ 0, 1,  0],
            [-sa, 0, ca]]

Stability to wind (rotation by beta about z, so that x meets the flight
velocity, whose stability-frame components are (cos(beta), sin(beta), 0)
for the sideslip convention of the free stream in
:func:`ventorum.aero.system.freestream_direction`)::

    R_sw = [[cb, sb, 0],
            [-sb, cb, 0],
            [ 0,  0,  1]]

Body to wind::

    R_bw = R_sw R_bs

The inverse maps use the transpose, since the matrices are orthogonal.
"""

from __future__ import annotations

import numpy as np

AXES_VALUES = ("body", "stability", "wind")
AXES_ALL = ("body", "stability", "wind", "all")


def _check_axes(name: str, value: str) -> str:
    """Return *value* if it names a moment axis system."""
    if value not in AXES_VALUES:
        raise ValueError(f"'{name}' must be one of {list(AXES_VALUES)}, got {value!r}.")
    return value


def rotation_body_to_stability(alpha: float) -> np.ndarray:
    """Rotation matrix from body to stability axes for *alpha* [rad].

    Parameters
    ----------
    alpha : float
        Angle of attack [rad].

    Returns
    -------
    numpy.ndarray
        Orthogonal matrix of shape (3, 3).
    """
    ca, sa = float(np.cos(alpha)), float(np.sin(alpha))
    return np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]])


def rotation_stability_to_wind(beta: float) -> np.ndarray:
    """Rotation matrix from stability to wind axes for *beta* [rad].

    Parameters
    ----------
    beta : float
        Sideslip angle [rad], positive for wind from the right.

    Returns
    -------
    numpy.ndarray
        Orthogonal matrix of shape (3, 3).
    """
    cb, sb = float(np.cos(beta)), float(np.sin(beta))
    return np.array([[cb, sb, 0.0], [-sb, cb, 0.0], [0.0, 0.0, 1.0]])


def rotation_body_to_wind(alpha: float, beta: float) -> np.ndarray:
    """Rotation matrix from body to wind axes.

    Parameters
    ----------
    alpha : float
        Angle of attack [rad].
    beta : float
        Sideslip angle [rad], positive for wind from the right.

    Returns
    -------
    numpy.ndarray
        Orthogonal matrix of shape (3, 3), the product of the stability
        map and the wind map.
    """
    return rotation_stability_to_wind(beta) @ rotation_body_to_stability(alpha)


def transform_moments(
    Cl: float,
    Cm: float,
    Cn: float,
    alpha: float,
    beta: float,
    to: str,
    from_axes: str = "body",
) -> dict[str, float]:
    """Express moment coefficients in another axis system.

    Parameters
    ----------
    Cl : float
        Rolling moment coefficient (Cl > 0 right wing down).
    Cm : float
        Pitching moment coefficient (Cm > 0 nose up).
    Cn : float
        Yawing moment coefficient (Cn > 0 nose right).
    alpha : float
        Angle of attack [rad].
    beta : float
        Sideslip angle [rad], positive for wind from the right.
    to : str
        Target system: ``"body"``, ``"stability"`` or ``"wind"``.
    from_axes : str, optional
        Source system (default ``"body"``).

    Returns
    -------
    dict
        ``{"Cl", "Cm", "Cn"}`` in the target system.

    Raises
    ------
    ValueError
        If *to* or *from_axes* is not a known axis system.
    """
    _check_axes("to", to)
    _check_axes("from_axes", from_axes)
    vec = np.array([float(Cl), float(Cm), float(Cn)], dtype=float)
    r_bs = rotation_body_to_stability(alpha)
    r_bw = rotation_body_to_wind(alpha, beta)
    to_body = {"body": np.eye(3), "stability": r_bs.T, "wind": r_bw.T}
    from_body = {"body": np.eye(3), "stability": r_bs, "wind": r_bw}
    out = from_body[to] @ (to_body[from_axes] @ vec)
    return {"Cl": float(out[0]), "Cm": float(out[1]), "Cn": float(out[2])}
