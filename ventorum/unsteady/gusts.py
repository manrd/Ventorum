# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Convected gust and wind-shear velocity fields for time-domain unsteady flow.

Every gust constructor returns a callable ``gust(points, t) -> velocities``
where ``points`` has shape ``(n, 3)`` [m] in geometry axes (x aft, y right,
z up), ``t`` is time [s], and the returned array has shape ``(n, 3)`` [m/s].
The returned velocity is added to the mean free stream ``V_0`` at those
points.

When a gust constructed by this module is passed to
:func:`ventorum.time_response` (or
:func:`ventorum.unsteady.timedomain._solve_time_rings`), the solver binds the
mean flight condition and ground plane automatically via ``_bind_flow``
without modifying the original gust object. To evaluate a gust directly
outside the solver, pass ``V_inf`` or ``condition`` (and optional ``ground``)
to the constructor.

Direction conventions
---------------------
* ``"vertical"``: along ``lift_direction(alpha)`` in free air or
  ``ground.normal`` in ground effect. A positive amplitude is an upward gust
  that increases the effective angle of attack.
* ``"head"``: along the mean free-stream direction ``d_0`` (opposite to the
  mean flight path). A positive amplitude is a headwind that adds airspeed.
* ``"side"``: along the lateral axis ``e_y`` (towards the right wingtip). A
  positive amplitude blows from left to right.
* Unit 3-vector ``(ux, uy, uz)``: fixed direction in geometry axes (x aft,
  y right, z up).

References
----------
* R. L. Bisplinghoff, H. Ashley and R. L. Halfman, "Aeroelasticity",
  Addison-Wesley, 1955, chapter 5 (sharp-edged and discrete gust models).
* F. M. Hoblit, "Gust Loads on Aircraft: Concepts and Applications", AIAA,
  1988, chapter 2 (1-cosine discrete gust profile).
* R. B. Stull, "An Introduction to Boundary Layer Meteorology", Kluwer
  Academic Publishers, 1988, chapter 9 (logarithmic wind profile of the
  neutral atmospheric surface layer).
* J. R. Garratt, "The Atmospheric Boundary Layer", Cambridge University
  Press, 1992, chapter 3 (surface-layer logarithmic wind profile).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from ventorum.aero.system import (
    GroundPlane,
    freestream_direction,
    lift_direction,
    side_direction,
)
from ventorum.core.datatypes import FlightCondition
from ventorum.core.errors import InputError
from ventorum.utils.vec import cross3

__all__ = [
    "linear_shear",
    "logarithmic_shear",
    "one_minus_cosine",
    "sharp_edged",
    "wind_shear",
]

_ALL_DIRECTIONS = ("vertical", "head", "side")
_HORIZONTAL_DIRECTIONS = ("head", "side")


def _validate_finite_float(value: Any, name: str, *, positive: bool = False) -> float:
    """Validate *value* as a finite float (not bool), optionally strictly positive."""
    if isinstance(value, bool):
        raise InputError(f"{name} must be a finite number, got {value!r}.")
    try:
        val = float(value)
    except (TypeError, ValueError) as exc:
        raise InputError(f"{name} must be a finite number, got {value!r}.") from exc
    if not math.isfinite(val):
        raise InputError(f"{name} must be finite, got {value!r}.")
    if positive and val <= 0.0:
        raise InputError(f"{name} must be > 0, got {val:g}.")
    return val


def _validate_direction(
    direction: str | Sequence[float] | np.ndarray,
    *,
    allowed_names: tuple[str, ...] = _ALL_DIRECTIONS,
) -> str | np.ndarray:
    """Validate *direction* as a named axis or a unit 3-vector in geometry axes."""
    if isinstance(direction, str):
        if direction not in allowed_names:
            raise InputError(
                f"Unknown gust direction {direction!r}; allowed names are {allowed_names} "
                "or a unit 3-vector in geometry axes."
            )
        return direction
    if isinstance(direction, bool):
        raise InputError(f"Invalid gust direction {direction!r}.")
    try:
        vec = np.asarray(direction, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InputError(
            f"direction must be one of {allowed_names} or a unit 3-vector, got {direction!r}."
        ) from exc
    if vec.shape != (3,) or not np.all(np.isfinite(vec)):
        raise InputError(f"direction vector must be a finite 3-vector, got {direction!r}.")
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12 or abs(norm - 1.0) > 1e-6:
        raise InputError(
            f"direction vector must be a unit 3-vector (norm 1), got norm {norm:.6g}."
        )
    return (vec / norm).copy()


def _flow_basis(
    condition: FlightCondition | None,
    ground: GroundPlane | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mean-flight unit basis ``(e_x, e_y, e_z)`` in geometry axes."""
    alpha = 0.0 if condition is None else float(condition.alpha)
    beta = 0.0 if condition is None else float(condition.beta)
    d_0 = freestream_direction(alpha, beta)
    L_0 = lift_direction(alpha)
    S_0 = side_direction(alpha, beta)
    e_x = d_0
    e_z = L_0 if ground is None else np.asarray(ground.normal, dtype=np.float64)
    if ground is None:
        e_y = S_0
    else:
        e_y_raw = cross3(e_z, e_x)
        e_y_norm = float(np.linalg.norm(e_y_raw))
        e_y = e_y_raw / e_y_norm if e_y_norm > 1e-12 else S_0
    return e_x, e_y, e_z


def _resolve_direction_vector(
    direction: str | np.ndarray,
    condition: FlightCondition | None,
    ground: GroundPlane | None,
) -> np.ndarray:
    """Resolve a validated *direction* to a unit 3-vector in geometry axes."""
    if isinstance(direction, np.ndarray):
        return direction
    e_x, e_y, e_z = _flow_basis(condition, ground)
    if direction == "vertical":
        return e_z
    if direction == "head":
        return e_x
    return e_y


def _check_points(points: np.ndarray) -> np.ndarray:
    """Validate and return *points* as a 2-D float64 array of shape ``(n, 3)``."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise InputError(f"points must have shape (n, 3), got {pts.shape}.")
    return pts


class _SharpEdgedGust:
    """Callable sharp-edged convected gust field."""

    _is_shear = False

    def __init__(
        self,
        amplitude: float,
        direction: str | np.ndarray,
        x_front: float,
        *,
        V_inf: float | None = None,
        condition: FlightCondition | None = None,
        ground: GroundPlane | None = None,
    ) -> None:
        self.amplitude = amplitude
        self.direction = direction
        self.x_front = x_front
        self._V_inf = V_inf
        self._condition = condition
        self._ground = ground

    def _bind_flow(
        self,
        condition: FlightCondition,
        ground: GroundPlane | None,
    ) -> _SharpEdgedGust:
        cond = condition if self._condition is None else self._condition
        gnd = ground if self._ground is None else self._ground
        v_inf = float(cond.V_inf) if self._V_inf is None else self._V_inf
        return _SharpEdgedGust(
            self.amplitude,
            self.direction,
            self.x_front,
            V_inf=v_inf,
            condition=cond,
            ground=gnd,
        )

    def __call__(self, points: np.ndarray, t: float) -> np.ndarray:
        pts = _check_points(points)
        t_val = _validate_finite_float(t, "t")
        if self._V_inf is not None:
            v_inf = self._V_inf
        elif self._condition is not None:
            v_inf = float(self._condition.V_inf)
        else:
            raise InputError(
                "sharp_edged needs V_inf or condition when called outside time_response."
            )
        e_x, _, _ = _flow_basis(self._condition, self._ground)
        u_dir = _resolve_direction_vector(self.direction, self._condition, self._ground)
        xi = (self.x_front + v_inf * t_val) - (pts @ e_x)
        out = np.zeros_like(pts, dtype=np.float64)
        mask = xi >= 0.0
        if np.any(mask):
            out[mask] = self.amplitude * u_dir[None, :]
        return out


class _OneMinusCosineGust:
    """Callable 1-cosine convected gust field."""

    _is_shear = False

    def __init__(
        self,
        amplitude: float,
        length: float,
        direction: str | np.ndarray,
        x_front: float,
        *,
        V_inf: float | None = None,
        condition: FlightCondition | None = None,
        ground: GroundPlane | None = None,
    ) -> None:
        self.amplitude = amplitude
        self.length = length
        self.direction = direction
        self.x_front = x_front
        self._V_inf = V_inf
        self._condition = condition
        self._ground = ground

    def _bind_flow(
        self,
        condition: FlightCondition,
        ground: GroundPlane | None,
    ) -> _OneMinusCosineGust:
        cond = condition if self._condition is None else self._condition
        gnd = ground if self._ground is None else self._ground
        v_inf = float(cond.V_inf) if self._V_inf is None else self._V_inf
        return _OneMinusCosineGust(
            self.amplitude,
            self.length,
            self.direction,
            self.x_front,
            V_inf=v_inf,
            condition=cond,
            ground=gnd,
        )

    def __call__(self, points: np.ndarray, t: float) -> np.ndarray:
        pts = _check_points(points)
        t_val = _validate_finite_float(t, "t")
        if self._V_inf is not None:
            v_inf = self._V_inf
        elif self._condition is not None:
            v_inf = float(self._condition.V_inf)
        else:
            raise InputError(
                "one_minus_cosine needs V_inf or condition when called outside time_response."
            )
        e_x, _, _ = _flow_basis(self._condition, self._ground)
        u_dir = _resolve_direction_vector(self.direction, self._condition, self._ground)
        xi = (self.x_front + v_inf * t_val) - (pts @ e_x)
        out = np.zeros_like(pts, dtype=np.float64)
        mask = (xi >= 0.0) & (xi <= self.length)
        if np.any(mask):
            mag = 0.5 * self.amplitude * (1.0 - np.cos((2.0 * math.pi / self.length) * xi[mask]))
            out[mask] = mag[:, None] * u_dir[None, :]
        return out


class _WindShearGust:
    """Callable horizontal wind-shear velocity field."""

    _is_shear = True

    def __init__(
        self,
        profile: Callable[[np.ndarray], np.ndarray] | Callable[[float], float],
        direction: str | np.ndarray,
        time_factor: Callable[[float], float] | None = None,
        *,
        condition: FlightCondition | None = None,
        ground: GroundPlane | None = None,
    ) -> None:
        self.profile = profile
        self.direction = direction
        self.time_factor = time_factor
        self._condition = condition
        self._ground = ground

    def _bind_flow(
        self,
        condition: FlightCondition,
        ground: GroundPlane | None,
    ) -> _WindShearGust:
        cond = condition if self._condition is None else self._condition
        gnd = ground if self._ground is None else self._ground
        return _WindShearGust(
            self.profile,
            self.direction,
            self.time_factor,
            condition=cond,
            ground=gnd,
        )

    def __call__(self, points: np.ndarray, t: float) -> np.ndarray:
        pts = _check_points(points)
        t_val = _validate_finite_float(t, "t")
        factor = 1.0 if self.time_factor is None else _validate_finite_float(self.time_factor(t_val), "time_factor(t)")
        if self._ground is None:
            # Height along the mean-flight vertical e_z = lift_direction(alpha),
            # measured from the geometry origin. The wind blows along the mean
            # flight path, so points on one line along d_0 see one wind speed.
            _, _, e_z = _flow_basis(self._condition, None)
            heights = pts @ e_z
        else:
            heights = self._ground.height(pts)
        try:
            raw = self.profile(heights)
            speeds = np.asarray(raw, dtype=np.float64)
            if speeds.ndim == 0:
                speeds = np.full(heights.shape, float(speeds), dtype=np.float64)
            elif speeds.shape != heights.shape:
                raise TypeError("profile shape mismatch")
        except TypeError:
            speeds = np.array([float(self.profile(float(h))) for h in heights], dtype=np.float64)
        if not np.all(np.isfinite(speeds)):
            raise InputError("wind_shear profile returned non-finite speeds.")
        u_dir = _resolve_direction_vector(self.direction, self._condition, self._ground)
        return (factor * speeds)[:, None] * u_dir[None, :]


def sharp_edged(
    amplitude: float,
    direction: str | Sequence[float] | np.ndarray = "vertical",
    x_front: float = 0.0,
    *,
    V_inf: float | None = None,
    condition: FlightCondition | None = None,
    ground: GroundPlane | None = None,
) -> Callable[[np.ndarray, float], np.ndarray]:
    """Build a sharp-edged convected step gust callable ``gust(points, t)``.

    The gust front is frozen in the air and convects aft along the mean flight
    path at ``V_inf``: its position at time ``t`` [s] is ``x_front + V_inf * t``
    [m] along ``d_0``. Points behind the front (``points . d_0 <= x_front +
    V_inf * t``) receive velocity ``amplitude * u_dir`` [m/s]; points ahead
    receive zero.

    Parameters
    ----------
    amplitude : float
        Gust velocity magnitude [m/s]. Sign follows *direction*: positive
        ``"vertical"`` is upward, positive ``"head"`` is a headwind (adds
        airspeed), positive ``"side"`` blows towards the right wingtip.
    direction : {'vertical', 'head', 'side'} or array-like of shape (3,), optional
        Gust velocity direction. ``"vertical"`` uses ``lift_direction(alpha)``
        in free air or ``ground.normal`` in ground effect; ``"head"`` is along
        the free stream ``d_0`` (against the flight path); ``"side"`` is along
        ``e_y``; or pass a unit 3-vector in geometry axes. Default
        ``"vertical"``.
    x_front : float, optional
        Initial streamwise position of the gust front along ``d_0`` at
        ``t = 0`` [m]. Default ``0.0``.
    V_inf : float or None, optional
        Convection speed [m/s] for direct evaluation outside
        :func:`ventorum.time_response`. None binds automatically in the solver.
    condition : FlightCondition or None, optional
        Mean flight condition for direct evaluation outside the solver.
    ground : GroundPlane or None, optional
        Ground plane for direct evaluation outside the solver.

    Returns
    -------
    callable
        Callable ``gust(points, t) -> velocities`` [m/s] of shape ``(n, 3)``.

    Raises
    ------
    InputError
        If *amplitude*, *x_front* or *V_inf* is non-finite (or ``V_inf <= 0``),
        or if *direction* is invalid.
    """
    amp = _validate_finite_float(amplitude, "amplitude")
    xf = _validate_finite_float(x_front, "x_front")
    v_inf_val = None if V_inf is None else _validate_finite_float(V_inf, "V_inf", positive=True)
    dir_spec = _validate_direction(direction, allowed_names=_ALL_DIRECTIONS)
    return _SharpEdgedGust(amp, dir_spec, xf, V_inf=v_inf_val, condition=condition, ground=ground)


def one_minus_cosine(
    amplitude: float,
    length: float,
    direction: str | Sequence[float] | np.ndarray = "vertical",
    x_front: float = 0.0,
    *,
    V_inf: float | None = None,
    condition: FlightCondition | None = None,
    ground: GroundPlane | None = None,
) -> Callable[[np.ndarray, float], np.ndarray]:
    """Build a 1-cosine discrete convected gust callable ``gust(points, t)``.

    For distance behind the gust front ``xi = x_front + V_inf * t - points . d_0``
    [m], the velocity magnitude is
    ``0.5 * amplitude * (1 - cos(2 * pi * xi / length))`` [m/s] for
    ``0 <= xi <= length`` and zero outside (Hoblit 1988; Bisplinghoff, Ashley
    and Halfman 1955). Peak velocity equals *amplitude* at ``xi = length / 2``.

    Parameters
    ----------
    amplitude : float
        Peak gust velocity [m/s]. Positive ``"vertical"`` is upward, positive
        ``"head"`` adds airspeed, positive ``"side"`` blows towards the right
        wingtip.
    length : float
        Total gust wavelength [m] (must be ``> 0``). Equal to ``2 * H`` where
        ``H`` is the gust gradient distance to peak velocity.
    direction : {'vertical', 'head', 'side'} or array-like of shape (3,), optional
        Gust velocity direction (named axis or unit 3-vector in geometry axes).
        Default ``"vertical"``.
    x_front : float, optional
        Initial streamwise position of the gust front along ``d_0`` at
        ``t = 0`` [m]. Default ``0.0``.
    V_inf : float or None, optional
        Convection speed [m/s] for direct evaluation outside
        :func:`ventorum.time_response`. None binds automatically in the solver.
    condition : FlightCondition or None, optional
        Mean flight condition for direct evaluation outside the solver.
    ground : GroundPlane or None, optional
        Ground plane for direct evaluation outside the solver.

    Returns
    -------
    callable
        Callable ``gust(points, t) -> velocities`` [m/s] of shape ``(n, 3)``.

    Raises
    ------
    InputError
        If *amplitude*, *x_front* or *V_inf* is invalid, if *length* is not
        finite and ``> 0``, or if *direction* is invalid.
    """
    amp = _validate_finite_float(amplitude, "amplitude")
    glen = _validate_finite_float(length, "length", positive=True)
    xf = _validate_finite_float(x_front, "x_front")
    v_inf_val = None if V_inf is None else _validate_finite_float(V_inf, "V_inf", positive=True)
    dir_spec = _validate_direction(direction, allowed_names=_ALL_DIRECTIONS)
    return _OneMinusCosineGust(amp, glen, dir_spec, xf, V_inf=v_inf_val, condition=condition, ground=ground)


def linear_shear(
    u0: float,
    du_dz: float,
    z_ref: float = 0.0,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a linear height-dependent wind speed profile ``profile(height) -> speed`` [m/s].

    Evaluates ``u(h) = u0 + du_dz * (h - z_ref)`` [m/s], where *h* is the
    height above the ground [m] (or geometry ``z`` [m] in free air).

    Parameters
    ----------
    u0 : float
        Wind speed at *z_ref* [m/s].
    du_dz : float
        Vertical wind-speed gradient [1/s] (m/s per m of height).
    z_ref : float, optional
        Reference height [m]. Default ``0.0``.

    Returns
    -------
    callable
        Profile callable ``profile(height) -> speed`` [m/s].
    """
    u0_v = _validate_finite_float(u0, "u0")
    grad = _validate_finite_float(du_dz, "du_dz")
    zr = _validate_finite_float(z_ref, "z_ref")

    def _profile(height: float | np.ndarray) -> np.ndarray | float:
        if np.ndim(height) == 0:
            return u0_v + grad * (float(height) - zr)
        h_arr = np.asarray(height, dtype=np.float64)
        return u0_v + grad * (h_arr - zr)

    return _profile


def logarithmic_shear(
    u_ref: float,
    z_ref: float,
    z0: float,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build the neutral atmospheric surface-layer logarithmic wind profile [m/s].

    Evaluates the logarithmic wind profile of the neutral atmospheric surface
    layer (Stull 1988, chapter 9; Garratt 1992, chapter 3):

    ``u(h) = u_ref * ln(h / z0) / ln(z_ref / z0)`` for ``h > z0``, and
    ``u(h) = 0`` for ``h <= z0``.

    Parameters
    ----------
    u_ref : float
        Wind speed at reference height *z_ref* [m/s].
    z_ref : float
        Reference height above the ground [m] (must be ``> z0``).
    z0 : float
        Aerodynamic roughness length [m] (must be ``> 0`` and ``< z_ref``).

    Returns
    -------
    callable
        Profile callable ``profile(height) -> speed`` [m/s].

    Raises
    ------
    InputError
        If *z0* is not ``> 0``, if *z_ref* is not ``> z0``, or if any input is
        non-finite.
    """
    u_v = _validate_finite_float(u_ref, "u_ref")
    z0_v = _validate_finite_float(z0, "z0", positive=True)
    zr_v = _validate_finite_float(z_ref, "z_ref", positive=True)
    if zr_v <= z0_v:
        raise InputError(f"z_ref ({zr_v:g} m) must be strictly larger than z0 ({z0_v:g} m).")
    inv_denom = u_v / math.log(zr_v / z0_v)

    def _profile(height: float | np.ndarray) -> np.ndarray | float:
        if np.ndim(height) == 0:
            hf = float(height)
            return inv_denom * math.log(hf / z0_v) if hf > z0_v else 0.0
        h_arr = np.asarray(height, dtype=np.float64)
        out = np.zeros_like(h_arr, dtype=np.float64)
        mask = h_arr > z0_v
        if np.any(mask):
            out[mask] = inv_denom * np.log(h_arr[mask] / z0_v)
        return out

    return _profile


def wind_shear(
    profile: Callable[[Any], Any],
    direction: str | Sequence[float] | np.ndarray = "head",
    *,
    time_factor: Callable[[float], float] | None = None,
    condition: FlightCondition | None = None,
    ground: GroundPlane | None = None,
) -> Callable[[np.ndarray, float], np.ndarray]:
    """Build a height-dependent horizontal wind-shear callable ``gust(points, t)``.

    Evaluates ``time_factor(t) * profile(height) * u_dir`` [m/s], where
    ``height`` is ``ground.height(points)`` [m] in ground effect or, in
    free air, the coordinate along the mean-flight vertical
    ``e_z = lift_direction(alpha)`` from the geometry origin [m]
    (``points[:, 2]`` at ``alpha = 0``).

    Parameters
    ----------
    profile : callable
        Callable ``profile(height) -> speed`` [m/s] giving the wind speed as a
        function of height [m] (see :func:`linear_shear` and
        :func:`logarithmic_shear`).
    direction : {'head', 'side'} or array-like of shape (3,), optional
        Horizontal wind direction. ``"head"`` (default) adds airspeed along
        the mean free stream ``d_0``; ``"side"`` blows towards the right
        wingtip along ``e_y``; or pass a unit 3-vector in geometry axes.
        ``"vertical"`` is refused because wind shear is a horizontal wind.
    time_factor : callable or None, optional
        Optional scalar modulation ``time_factor(t) -> factor`` [-] of time
        ``t`` [s]. Default None (constant factor ``1.0``).
    condition : FlightCondition or None, optional
        Mean flight condition for direct evaluation outside the solver.
    ground : GroundPlane or None, optional
        Ground plane for direct evaluation outside the solver.

    Returns
    -------
    callable
        Callable ``gust(points, t) -> velocities`` [m/s] of shape ``(n, 3)``.

    Raises
    ------
    InputError
        If *profile* is not callable, *time_factor* is neither None nor
        callable, or *direction* is invalid.
    """
    if not callable(profile):
        raise InputError("wind_shear profile must be a callable profile(height) -> speed.")
    if time_factor is not None and not callable(time_factor):
        raise InputError("wind_shear time_factor must be a callable time_factor(t) -> factor or None.")
    dir_spec = _validate_direction(direction, allowed_names=_HORIZONTAL_DIRECTIONS)
    return _WindShearGust(profile, dir_spec, time_factor, condition=condition, ground=ground)
