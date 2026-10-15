# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Exact low-speed potential-flow references for 3D bodies and their quad meshes.

This module gives the exact surface velocity and pressure coefficient of a
sphere and of a tri-axial ellipsoid in steady incompressible flow, plus
latitude-longitude quadrilateral meshes of both bodies. The solvers use these
values to verify 3D panel results (stagnation pressure, suction peak, area
and volume of the meshes).

Sources:

* H. Lamb, Hydrodynamics, 6th edition, Cambridge University Press, 1932,
  article 92 (flow past a sphere) and article 114 (flow past an ellipsoid).
* H. Lamb, Hydrodynamics, 6th edition, article 105 (flow past a spheroid,
  prolate and oblate closed forms used here as a check).
* J. Katz and A. Plotkin, Low-Speed Aerodynamics, 2nd edition, Cambridge
  University Press, 2001, section 3.12 (pressure distribution on a sphere).

Ellipsoid surface velocity: for a uniform stream along one body axis, Lamb
(article 114) shows that the exact potential on the body surface is a linear
function of the streamwise coordinate plus a constant. Its surface gradient
is therefore the tangential projection of a scaled stream vector. For a
stream ``U`` along the x axis the scale factor is ``2 / (2 - alpha0)``, where
``alpha0`` is the axis coefficient defined below. Potential flow is linear,
so an oblique stream is the sum of the three axial solutions: the surface
velocity is the tangential projection of the vector ``W = (2 Ux / (2 -
alpha0), 2 Uy / (2 - beta0), 2 Uz / (2 - gamma0))``. The sphere is the
special case ``alpha0 = beta0 = gamma0 = 2 / 3``, which gives ``W = 1.5 U``.

Meshes: latitude-longitude grids with one node at each pole and shared nodes
at the longitude seam, so no two nodes coincide. Polar cells are triangles
stored as degenerate quadrilaterals (the last entry repeats the third).
Corner order is counterclockwise seen from outside the body, so the diagonal
rule ``(p3 - p1) x (p4 - p2)`` points out of the body for every panel.
"""

from __future__ import annotations

import operator
from typing import Literal

import numpy as np
from scipy.special import elliprd


def _check_points(points: np.ndarray) -> np.ndarray:
    """Return points as a validated float array of shape (P, 3)."""
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
        raise ValueError("points must be an array of shape (P, 3) with P >= 1")
    if not np.all(np.isfinite(arr)):
        raise ValueError("points must hold finite values")
    return arr


def _stream_vector(value: float | np.ndarray, name: str) -> np.ndarray:
    """Return the free-stream vector [m/s] for a scalar or a 3-vector input."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        arr = np.array([float(arr), 0.0, 0.0])
    if arr.shape != (3,):
        raise ValueError(f"{name} must be a scalar or an array of shape (3,)")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must hold finite values")
    return arr


def _positive(value: float, name: str) -> float:
    """Return a validated positive length [m]."""
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return number


def _check_mesh_sizes(n_lat: int, n_lon: int) -> tuple[int, int]:
    """Return validated latitude and longitude counts (dimensionless)."""
    try:
        n_lat = operator.index(n_lat)
        n_lon = operator.index(n_lon)
    except TypeError as exc:
        raise ValueError("n_lat and n_lon must be integers") from exc
    if n_lat < 2:
        raise ValueError("n_lat must be at least 2")
    if n_lon < 3:
        raise ValueError("n_lon must be at least 3")
    return n_lat, n_lon


def sphere_surface_velocity(
    points: np.ndarray,
    radius: float,
    V_inf: float | np.ndarray,
    center: np.ndarray = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Exact surface velocity on a sphere in steady incompressible flow [m/s].

    The flow is the classical potential flow past a sphere (Lamb 1932,
    article 92; Katz and Plotkin 2001, section 3.12). The velocity at each
    surface point is ``1.5`` times the tangential part of the free stream.

    Parameters
    ----------
    points : np.ndarray
        Evaluation points on the sphere surface, shape (P, 3) [m].
    radius : float
        Sphere radius [m]. It must be positive.
    V_inf : float or np.ndarray
        Free-stream speed [m/s] along the x axis when scalar, else the
        free-stream vector of shape (3,) [m/s].
    center : np.ndarray
        Sphere center of shape (3,) [m].

    Returns
    -------
    np.ndarray
        Surface velocity vectors, shape (P, 3) [m/s].
    """
    pts = _check_points(points)
    radius = _positive(radius, "radius")
    stream = _stream_vector(V_inf, "V_inf")
    origin = np.asarray(center, dtype=float)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("center must be an array of shape (3,) with finite values [m]")
    relative = pts - origin[None, :]
    dist = np.linalg.norm(relative, axis=1)
    if np.any(dist == 0.0):
        raise ValueError("points must not coincide with the sphere center")
    normal = relative / dist[:, None]
    normal_stream = normal @ stream
    return 1.5 * (stream[None, :] - normal_stream[:, None] * normal)


def sphere_cp(
    points: np.ndarray,
    radius: float,
    V_inf: float | np.ndarray,
    center: np.ndarray = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Exact pressure coefficient on a sphere (dimensionless).

    ``Cp = 1 - |V|^2 / |U|^2 = 1 - (9/4) sin^2(theta)``, where ``theta`` is
    the angle from the free-stream direction (Lamb 1932, article 92).

    Parameters
    ----------
    points : np.ndarray
        Evaluation points on the sphere surface, shape (P, 3) [m].
    radius : float
        Sphere radius [m]. It must be positive.
    V_inf : float or np.ndarray
        Free-stream speed [m/s] along the x axis when scalar, else the
        free-stream vector of shape (3,) [m/s].
    center : np.ndarray
        Sphere center of shape (3,) [m].

    Returns
    -------
    np.ndarray
        Pressure coefficient at each point, shape (P,) (dimensionless).
    """
    stream = _stream_vector(V_inf, "V_inf")
    speed_squared = float(stream @ stream)
    if speed_squared == 0.0:
        raise ValueError("V_inf must be nonzero to compute Cp [m/s]")
    velocity = sphere_surface_velocity(points, radius, stream, center)
    return 1.0 - np.sum(velocity**2, axis=1) / speed_squared


def ellipsoid_coefficients(a: float, b: float, c: float) -> tuple[float, float, float]:
    """Axis coefficients of a tri-axial ellipsoid (dimensionless).

    The coefficients are the integrals of Lamb (1932), article 114::

        alpha0 = a b c ∫_0^∞ dλ / ((a^2 + λ) Δ),

    with ``Δ = sqrt((a^2 + λ) (b^2 + λ) (c^2 + λ))``, and likewise for
    ``beta0`` and ``gamma0`` with the factors ``(b^2 + λ)`` and ``(c^2 +
    λ)``. They satisfy ``alpha0 + beta0 + gamma0 = 2``. Each integral is a
    Carlson symmetric elliptic integral: ``alpha0 = (2/3) a b c
    R_D(b^2, c^2, a^2)``, evaluated with ``scipy.special.elliprd``.

    Parameters
    ----------
    a : float
        Semi-axis along x [m]. It must be positive.
    b : float
        Semi-axis along y [m]. It must be positive.
    c : float
        Semi-axis along z [m]. It must be positive.

    Returns
    -------
    tuple[float, float, float]
        The coefficients ``(alpha0, beta0, gamma0)`` (dimensionless).
    """
    a = _positive(a, "a")
    b = _positive(b, "b")
    c = _positive(c, "c")
    alpha0 = a * b * c * (2.0 / 3.0) * float(elliprd(b * b, c * c, a * a))
    beta0 = a * b * c * (2.0 / 3.0) * float(elliprd(a * a, c * c, b * b))
    gamma0 = a * b * c * (2.0 / 3.0) * float(elliprd(a * a, b * b, c * c))
    return alpha0, beta0, gamma0


def ellipsoid_surface_velocity(
    points: np.ndarray,
    a: float,
    b: float,
    c: float,
    V_inf_vec: float | np.ndarray,
) -> np.ndarray:
    """Exact surface velocity on a tri-axial ellipsoid [m/s].

    The ellipsoid is ``x^2/a^2 + y^2/b^2 + z^2/c^2 = 1``, centered at the
    origin. The velocity is the tangential projection of the scaled stream
    ``W = (2 Ux/(2-alpha0), 2 Uy/(2-beta0), 2 Uz/(2-gamma0))`` (Lamb 1932,
    article 114; see the module docstring for the derivation).

    Parameters
    ----------
    points : np.ndarray
        Evaluation points on the ellipsoid surface, shape (P, 3) [m].
    a : float
        Semi-axis along x [m]. It must be positive.
    b : float
        Semi-axis along y [m]. It must be positive.
    c : float
        Semi-axis along z [m]. It must be positive.
    V_inf_vec : float or np.ndarray
        Free-stream vector of shape (3,) [m/s], or a scalar speed [m/s]
        along the x axis.

    Returns
    -------
    np.ndarray
        Surface velocity vectors, shape (P, 3) [m/s].
    """
    pts = _check_points(points)
    a = _positive(a, "a")
    b = _positive(b, "b")
    c = _positive(c, "c")
    stream = _stream_vector(V_inf_vec, "V_inf_vec")
    alpha0, beta0, gamma0 = ellipsoid_coefficients(a, b, c)
    scaled = np.array(
        [
            2.0 * stream[0] / (2.0 - alpha0),
            2.0 * stream[1] / (2.0 - beta0),
            2.0 * stream[2] / (2.0 - gamma0),
        ]
    )
    gradient = np.column_stack(
        [pts[:, 0] / a**2, pts[:, 1] / b**2, pts[:, 2] / c**2]
    )
    magnitude = np.linalg.norm(gradient, axis=1)
    if np.any(magnitude == 0.0):
        raise ValueError("points must lie on the ellipsoid surface")
    normal = gradient / magnitude[:, None]
    normal_scaled = normal @ scaled
    return scaled[None, :] - normal_scaled[:, None] * normal


def ellipsoid_cp(
    points: np.ndarray,
    a: float,
    b: float,
    c: float,
    V_inf_vec: float | np.ndarray,
) -> np.ndarray:
    """Exact pressure coefficient on a tri-axial ellipsoid (dimensionless).

    ``Cp = 1 - |V|^2 / |U|^2`` with the exact surface velocity ``V``.

    Parameters
    ----------
    points : np.ndarray
        Evaluation points on the ellipsoid surface, shape (P, 3) [m].
    a : float
        Semi-axis along x [m]. It must be positive.
    b : float
        Semi-axis along y [m]. It must be positive.
    c : float
        Semi-axis along z [m]. It must be positive.
    V_inf_vec : float or np.ndarray
        Free-stream vector of shape (3,) [m/s], or a scalar speed [m/s]
        along the x axis.

    Returns
    -------
    np.ndarray
        Pressure coefficient at each point, shape (P,) (dimensionless).
    """
    stream = _stream_vector(V_inf_vec, "V_inf_vec")
    speed_squared = float(stream @ stream)
    if speed_squared == 0.0:
        raise ValueError("V_inf_vec must be nonzero to compute Cp [m/s]")
    velocity = ellipsoid_surface_velocity(points, a, b, c, stream)
    return 1.0 - np.sum(velocity**2, axis=1) / speed_squared


def prolate_spheroid_coefficients(a: float, b: float) -> tuple[float, float, float]:
    """Closed-form axis coefficients of a spheroid (dimensionless).

    The spheroid has semi-axis ``a`` along x and equal transverse
    semi-axes ``b`` (Lamb 1932, article 105). The same article gives both
    the prolate form (``a > b``) and the oblate form (``a < b``); the
    sphere (``a == b``) gives ``2/3`` thrice. The transverse pair follows
    from ``beta0 = gamma0 = (2 - alpha0) / 2``.

    Parameters
    ----------
    a : float
        Semi-axis along the symmetry (x) axis [m]. It must be positive.
    b : float
        Transverse semi-axis [m]. It must be positive.

    Returns
    -------
    tuple[float, float, float]
        The coefficients ``(alpha0, beta0, gamma0)`` (dimensionless).
    """
    a = _positive(a, "a")
    b = _positive(b, "b")
    if a == b:
        return 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0
    if a > b:
        eccentricity = float(np.sqrt(1.0 - (b / a) ** 2))
        log_term = float(np.log((1.0 + eccentricity) / (1.0 - eccentricity)))
        alpha0 = (1.0 - eccentricity**2) / eccentricity**3 * (log_term - 2.0 * eccentricity)
    else:
        eccentricity = float(np.sqrt(1.0 - (a / b) ** 2))
        root = float(np.sqrt(1.0 - eccentricity**2))
        alpha0 = 2.0 / eccentricity**3 * (eccentricity - root * float(np.arcsin(eccentricity)))
    transverse = (2.0 - alpha0) / 2.0
    return float(alpha0), float(transverse), float(transverse)


def _latitude_angles(n_lat: int, spacing: str) -> np.ndarray:
    """Polar angles of the rings from the north pole, shape (n_lat + 1,) [rad]."""
    index = np.arange(n_lat + 1)
    if spacing == "uniform":
        return np.pi * index / n_lat
    return 0.5 * np.pi * (1.0 - np.cos(np.pi * index / n_lat))


def _latlon_nodes(
    semi_axes: tuple[float, float, float], n_lat: int, n_lon: int, theta: np.ndarray
) -> np.ndarray:
    """Nodes of a latitude-longitude grid, shape (N, 3) [m]."""
    a, b, c = semi_axes
    count = 2 + (n_lat - 1) * n_lon
    nodes = np.zeros((count, 3))
    nodes[0, :] = (0.0, 0.0, c)
    nodes[count - 1, :] = (0.0, 0.0, -c)
    phi = 2.0 * np.pi * np.arange(n_lon) / n_lon
    for ring in range(1, n_lat):
        angle = theta[ring]
        first = 1 + (ring - 1) * n_lon
        nodes[first : first + n_lon, 0] = a * np.sin(angle) * np.cos(phi)
        nodes[first : first + n_lon, 1] = b * np.sin(angle) * np.sin(phi)
        nodes[first : first + n_lon, 2] = c * np.cos(angle)
    return nodes


def _latlon_panels(n_lat: int, n_lon: int) -> np.ndarray:
    """Panels of a latitude-longitude grid, shape (M, 4), int.

    Corner order is counterclockwise seen from outside, so the diagonal
    rule ``(p3 - p1) x (p4 - p2)`` points out of the body. Polar cells
    repeat the third node in the fourth entry.
    """
    quads: list[list[int]] = []
    for side in range(n_lon):
        nxt = (side + 1) % n_lon
        quads.append([0, 1 + side, 1 + nxt, 1 + nxt])
    for ring in range(1, n_lat - 1):
        upper = 1 + (ring - 1) * n_lon
        lower = 1 + ring * n_lon
        for side in range(n_lon):
            nxt = (side + 1) % n_lon
            quads.append([upper + side, lower + side, lower + nxt, upper + nxt])
    last = 1 + (n_lat - 2) * n_lon
    south = 2 + (n_lat - 1) * n_lon - 1
    for side in range(n_lon):
        nxt = (side + 1) % n_lon
        quads.append([last + side, south, south, last + nxt])
    return np.array(quads, dtype=np.int64)


def sphere_quad_mesh(radius: float, n_lat: int, n_lon: int) -> tuple[np.ndarray, np.ndarray]:
    """Latitude-longitude quadrilateral mesh of a sphere centered at the origin.

    Parameters
    ----------
    radius : float
        Sphere radius [m]. It must be positive.
    n_lat : int
        Count of latitude bands (dimensionless). It must be at least 2.
    n_lon : int
        Count of longitude divisions (dimensionless). It must be at least 3.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Nodes of shape (N, 3) [m] and panels of shape (M, 4) (int indices),
        with ``N = 2 + (n_lat - 1) * n_lon`` and ``M = n_lat * n_lon``.
    """
    radius = _positive(radius, "radius")
    n_lat, n_lon = _check_mesh_sizes(n_lat, n_lon)
    theta = _latitude_angles(n_lat, "uniform")
    nodes = _latlon_nodes((radius, radius, radius), n_lat, n_lon, theta)
    return nodes, _latlon_panels(n_lat, n_lon)


def ellipsoid_quad_mesh(
    a: float,
    b: float,
    c: float,
    n_lat: int,
    n_lon: int,
    spacing: Literal["cosine", "uniform"] = "cosine",
) -> tuple[np.ndarray, np.ndarray]:
    """Latitude-longitude quadrilateral mesh of an ellipsoid at the origin.

    Parameters
    ----------
    a : float
        Semi-axis along x [m]. It must be positive.
    b : float
        Semi-axis along y [m]. It must be positive.
    c : float
        Semi-axis along z [m]. It must be positive.
    n_lat : int
        Count of latitude bands (dimensionless). It must be at least 2.
    n_lon : int
        Count of longitude divisions (dimensionless). It must be at least 3.
    spacing : str
        Ring layout (dimensionless): ``"cosine"`` clusters rings near the
        poles, ``"uniform"`` spaces the polar angle evenly.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Nodes of shape (N, 3) [m] and panels of shape (M, 4) (int indices),
        with ``N = 2 + (n_lat - 1) * n_lon`` and ``M = n_lat * n_lon``.
    """
    a = _positive(a, "a")
    b = _positive(b, "b")
    c = _positive(c, "c")
    n_lat, n_lon = _check_mesh_sizes(n_lat, n_lon)
    if spacing not in ("cosine", "uniform"):
        raise ValueError('spacing must be "cosine" or "uniform" (dimensionless)')
    theta = _latitude_angles(n_lat, spacing)
    nodes = _latlon_nodes((a, b, c), n_lat, n_lon, theta)
    return nodes, _latlon_panels(n_lat, n_lon)
