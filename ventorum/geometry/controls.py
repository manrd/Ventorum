# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Hinged control surfaces: thin-airfoil theory and input validation.

This module provides the aerodynamic effectiveness factor and pitching-moment
derivative of a hinged flap from thin-airfoil theory, constructs deflected
airfoil models for lifting-line solvers, and validates control surface inputs.

References
----------
H. Glauert, The Elements of Aerofoil and Airscrew Theory, Cambridge
University Press, 1926, chapter VII.
H. Glauert, "Theoretical relationships for an aerofoil with hinged flap",
Aeronautical Research Committee, R&M 1095, 1927.
J. Katz and A. Plotkin, Low-Speed Aerodynamics, 2nd ed., Cambridge
University Press, 2001.
"""

from __future__ import annotations

import numpy as np

from ventorum.core.datatypes import (
    AirfoilType,
    LiftingSurface,
    LinearAirfoil,
    TabulatedAirfoil,
)

__all__ = [
    "flap_effectiveness",
    "flap_moment_derivative",
    "hinge_moment_derivatives",
    "deflected_airfoil",
    "validate_controls",
    "CONTROL_JOIN",
    "ControlJoinField",
]

#: Spanwise width of the join of a chordwise control at its side edge, as a
#: fraction of the local chord [-]. The value is the default transition of
#: the 3D panel method (``ventorum.panel3d.mesh.CONTROL_TRANSITION``), so
#: both models seal the side edge with the same geometry.
CONTROL_JOIN = 0.2


def flap_effectiveness(hinge_x_c: float) -> float:
    """Compute the flap effectiveness factor tau from thin-airfoil theory.

    The factor gives the change in zero-lift angle of attack per unit flap
    deflection: tau = -d(alpha_L0) / d(delta) = (dCl/ddelta) / (dCl/dalpha).

    Parameters
    ----------
    hinge_x_c : float
        Hinge chordwise position as a fraction of local chord from the
        leading edge [-], in [0, 1].

    Returns
    -------
    float
        Flap effectiveness factor tau [-].

    References
    ----------
    H. Glauert, The Elements of Aerofoil and Airscrew Theory, Cambridge
    University Press, 1926.
    J. Katz and A. Plotkin, Low-Speed Aerodynamics, 2nd ed., Cambridge
    University Press, 2001.
    """
    h = float(hinge_x_c)
    if h <= 0.0:
        return 1.0
    if h >= 1.0:
        return 0.0
    theta_h = float(np.arccos(np.clip(1.0 - 2.0 * h, -1.0, 1.0)))
    return float(1.0 - (theta_h - np.sin(theta_h)) / np.pi)


def flap_moment_derivative(hinge_x_c: float) -> float:
    """Compute the section pitching-moment derivative dCm_c/4 / d(delta).

    The derivative gives the rate of change of section pitching moment about
    the quarter chord with flap deflection from thin-airfoil theory [1/rad].

    Parameters
    ----------
    hinge_x_c : float
        Hinge chordwise position as a fraction of local chord from the
        leading edge [-], in [0, 1].

    Returns
    -------
    float
        Pitching-moment derivative dCm/ddelta about the quarter chord [1/rad]
        (negative).

    References
    ----------
    H. Glauert, The Elements of Aerofoil and Airscrew Theory, Cambridge
    University Press, 1926.
    J. Katz and A. Plotkin, Low-Speed Aerodynamics, 2nd ed., Cambridge
    University Press, 2001.
    """
    h = float(hinge_x_c)
    if h <= 0.0 or h >= 1.0:
        return 0.0
    theta_h = float(np.arccos(np.clip(1.0 - 2.0 * h, -1.0, 1.0)))
    return float(-0.5 * np.sin(theta_h) * (1.0 - np.cos(theta_h)))


def hinge_moment_derivatives(hinge_x_c: float) -> tuple[float, float]:
    """Compute the section hinge-moment derivatives of a plain flap from thin-airfoil theory.

    The section hinge moment of a flap aft of ``x/c = hinge_x_c`` is
    ``Ch = Ch_alpha * alpha + Ch_delta * delta``, with ``alpha`` the angle of
    attack of the flat plate ahead of the hinge and ``delta`` the flap
    deflection. The hinge moment is about the hinge line, positive when it
    tends to increase the deflection (trailing edge down). The coefficient is
    ``Ch = H' / (q c_f^2)``, with ``H'`` the hinge moment per unit span
    [N m/m], ``q`` the dynamic pressure [Pa] and ``c_f = (1 - hinge_x_c) c``
    the flap chord [m].

    With ``x/c = (1 - cos(theta)) / 2``, ``theta_h = arccos(1 - 2 hinge_x_c)``
    and ``E = 1 - hinge_x_c``, the integral of the thin-airfoil load
    (Glauert's series) times the arm aft of the hinge gives::

        Ch_alpha = -((pi - theta_h) (cos(theta_h) - 1/2)
                     + sin(theta_h) (1 - cos(theta_h) / 2)) / E^2
        Ch_delta = (1 - theta_h / pi) Ch_alpha
                   - sin(theta_h) ((pi - theta_h) cos(theta_h) + sin(theta_h)) / (2 pi E^2)

    The first term of ``Ch_delta`` is the load of the flat plate at the
    angle ``(1 - theta_h / pi) delta``; the second is the load of the
    flap series terms (a logarithmic load at the hinge). At ``hinge_x_c -> 0``
    both derivatives go to ``-pi/2`` (moment of a flat plate about its
    leading edge). At ``hinge_x_c -> 1``, ``Ch_alpha -> 0`` and
    ``Ch_delta -> -8 / (3 pi)``.

    Parameters
    ----------
    hinge_x_c : float
        Hinge chordwise position as a fraction of local chord from the
        leading edge [-], in (0, 1).

    Returns
    -------
    tuple of float
        ``(Ch_alpha, Ch_delta)`` [1/rad], both negative.

    Raises
    ------
    ValueError
        If *hinge_x_c* is not in the open interval (0, 1).

    References
    ----------
    H. Glauert, The Elements of Aerofoil and Airscrew Theory, Cambridge
    University Press, 1926, chapter VII: thin-airfoil theory and the Fourier
    series of the load of a cambered (flapped) section.
    H. Glauert, "Theoretical relationships for an aerofoil with hinged
    flap", Aeronautical Research Committee, R&M 1095, 1927: the hinge
    moment of a hinged flap (``Ch_alpha`` is twice his ``b_1``, equation
    14, because his coefficients use ``rho V^2``).
    """
    h = float(hinge_x_c)
    if not (0.0 < h < 1.0):
        raise ValueError(f"hinge_x_c must satisfy 0 < hinge_x_c < 1, got {hinge_x_c}.")
    theta_h = float(np.arccos(1.0 - 2.0 * h))
    c, s = float(np.cos(theta_h)), float(np.sin(theta_h))
    e2 = (1.0 - h) ** 2
    ch_alpha = -((np.pi - theta_h) * (c - 0.5) + s * (1.0 - 0.5 * c)) / e2
    ch_delta = (1.0 - theta_h / np.pi) * ch_alpha - s * ((np.pi - theta_h) * c + s) / (2.0 * np.pi * e2)
    return float(ch_alpha), float(ch_delta)


def deflected_airfoil(airfoil: AirfoilType, delta: float, hinge_x_c: float) -> AirfoilType:
    """Construct an airfoil model for a strip with a deflected flap.

    The thin-airfoil theory of a hinged flap gives the lift change as a
    shift of the angle of attack by ``tau * delta`` and a pitching-moment
    change ``dCm_ddelta * delta`` about the quarter chord. The deflected
    airfoil is

    - ``Cl_new(alpha) = Cl(alpha + tau * delta)``,
    - ``Cd_new(alpha) = Cd(alpha + tau * delta)``,
    - ``Cm_new(alpha) = Cm(alpha + tau * delta) + dCm_ddelta * delta``,

    with ``tau = flap_effectiveness(hinge_x_c)`` and
    ``dCm_ddelta = flap_moment_derivative(hinge_x_c)``.

    For a ``TabulatedAirfoil`` the new table has the angles
    ``alpha - tau * delta``, the same Cl and Cd rows, and the Cm rows plus
    ``dCm_ddelta * delta``. Thus the Cm of the section follows the shifted
    lift curve, also in stall: at a given angle of the base table, the
    deflected section has the same Cl and its Cm changes by
    ``dCm_ddelta * delta`` only. The angle range of the table also moves by
    ``-tau * delta``. A table without Cm data gets
    ``Cm_new = dCm_ddelta * delta`` at every angle. The shift keeps the
    order of the angles, so the new table keeps the gap mask
    (:attr:`ventorum.core.datatypes.TabulatedAirfoil.gap_mask`) of the
    base table.

    For a ``LinearAirfoil``, Cm and Cd do not change with the angle, so the
    same rule gives ``alpha_L0_new = alpha_L0 - tau * delta`` and
    ``Cm0_new = Cm0 + dCm_ddelta * delta``. The lift slope ``a0`` and
    ``Cd0`` do not change.

    Parameters
    ----------
    airfoil : AirfoilType
        Base airfoil model (LinearAirfoil or TabulatedAirfoil).
    delta : float
        Flap deflection angle [rad], positive trailing edge down.
    hinge_x_c : float
        Hinge chordwise position as a fraction of local chord [-].

    Returns
    -------
    AirfoilType
        Deflected airfoil model, a new object. The function returns the
        same object if ``delta == 0.0``. It does not change *airfoil*.

    References
    ----------
    H. Glauert, The Elements of Aerofoil and Airscrew Theory, Cambridge
    University Press, 1926.
    J. Katz and A. Plotkin, Low-Speed Aerodynamics, 2nd ed., Cambridge
    University Press, 2001.
    """
    if delta == 0.0:
        return airfoil

    tau = flap_effectiveness(hinge_x_c)
    dCm_ddelta = flap_moment_derivative(hinge_x_c)

    if isinstance(airfoil, LinearAirfoil):
        return LinearAirfoil(
            name=f"{airfoil.name} (deflected)",
            a0=airfoil.a0,
            alpha_L0=float(airfoil.alpha_L0 - tau * delta),
            Cd0=airfoil.Cd0,
            Cm0=float(airfoil.Cm0 + dCm_ddelta * delta),
        )
    if isinstance(airfoil, TabulatedAirfoil):
        alpha = np.asarray(airfoil.alpha, dtype=float)
        cl = np.asarray(airfoil.Cl_data, dtype=float).copy()
        cd = np.asarray(airfoil.Cd_data, dtype=float).copy()
        alpha_new = alpha - tau * delta
        if airfoil.Cm_data is not None:
            cm_new = np.asarray(airfoil.Cm_data, dtype=float) + dCm_ddelta * delta
        else:
            cm_new = np.full_like(alpha, dCm_ddelta * delta)
        return TabulatedAirfoil(
            name=f"{airfoil.name} (deflected)",
            alpha=alpha_new,
            Cl_data=cl,
            Cd_data=cd,
            Cm_data=cm_new,
            Re=airfoil.Re,
            gap_mask=None if airfoil.gap_mask is None else np.array(airfoil.gap_mask, dtype=bool),
        )
    raise TypeError(f"Unsupported airfoil type: {type(airfoil)}")


def validate_controls(surface: LiftingSurface) -> None:
    """Refuse invalid controls on a lifting surface.

    Parameters
    ----------
    surface : LiftingSurface
        Surface whose controls are validated.

    Raises
    ------
    ValueError
        If any control on *surface* violates the physical or numerical limits,
        naming the surface, the control, and the offending field.
    """
    controls = getattr(surface, "controls", [])
    if not controls:
        return

    for c in controls:
        # 1. Names not empty
        if not isinstance(c.name, str) or not c.name.strip():
            raise ValueError(
                f"Surface {surface.name!r}, control {c.name!r}: field 'name' must not be empty."
            )

        # 2. Every value finite
        for field_name in ("eta_start", "eta_end", "hinge_x_c", "deflection"):
            val = getattr(c, field_name)
            if not np.isfinite(val):
                raise ValueError(
                    f"Surface {surface.name!r}, control {c.name!r}: field '{field_name}' must be finite, got {val}."
                )

        # 3. 0 <= eta_start < eta_end <= 1
        if c.eta_start < 0.0 or c.eta_start >= c.eta_end:
            raise ValueError(
                f"Surface {surface.name!r}, control {c.name!r}: field 'eta_start' must satisfy 0 <= eta_start < eta_end, got {c.eta_start}."
            )
        if c.eta_end > 1.0 or c.eta_end <= c.eta_start:
            raise ValueError(
                f"Surface {surface.name!r}, control {c.name!r}: field 'eta_end' must satisfy eta_start < eta_end <= 1, got {c.eta_end}."
            )

        # 4. 0 < hinge_x_c < 1
        if c.hinge_x_c <= 0.0 or c.hinge_x_c >= 1.0:
            raise ValueError(
                f"Surface {surface.name!r}, control {c.name!r}: field 'hinge_x_c' must satisfy 0 < hinge_x_c < 1, got {c.hinge_x_c}."
            )

        # 5. abs(deflection) <= radians(30)
        if abs(c.deflection) > np.radians(30.0) + 1e-12:
            raise ValueError(
                f"Surface {surface.name!r}, control {c.name!r}: field 'deflection' magnitude must be <= 30 deg, got {np.degrees(c.deflection):.2f} deg."
            )

    # 6. Two controls of one surface must not overlap in span (they may touch)
    if len(controls) > 1:
        sorted_ctrls = sorted(controls, key=lambda ctrl: ctrl.eta_start)
        for i in range(len(sorted_ctrls) - 1):
            c1, c2 = sorted_ctrls[i], sorted_ctrls[i + 1]
            if c1.eta_end > c2.eta_start + 1e-9:
                raise ValueError(
                    f"Surface {surface.name!r}: controls {c1.name!r} and {c2.name!r} overlap in span "
                    f"({c1.eta_start}..{c1.eta_end} and {c2.eta_start}..{c2.eta_end}); "
                    f"field 'eta_end' of {c1.name!r} exceeds field 'eta_start' of {c2.name!r}."
                )


class ControlJoinField:
    """Spanwise deflection field of the chordwise controls of one surface.

    The vortex lattice deflects a chordwise control as geometry: the mean
    surface aft of the hinge turns about the hinge line. At a side edge of a
    control the deflection changes linearly over a join of spanwise length
    ``CONTROL_JOIN`` times the local chord, centred on the edge, so the
    deflected surface stays continuous (a sealed edge, as the transition of
    the 3D panel method). The join is a property of the geometry, not of
    the mesh: a node at span parameter ``eta`` gets the same deflection on
    every mesh.

    The rules of the join follow the 3D panel method
    (:func:`ventorum.panel3d.mesh._transition_edge_lists`): the join is not
    longer than the control; it does not reach past the root or the tip;
    and it shares the room to the next control edge with the join of that
    edge. A join also lies between two controls with different deflections
    (linear from one to the other), and at the root of a symmetric surface
    under an antisymmetric control (from -delta on the left to +delta on the
    right). The deflection is linear in the control deflections, so its
    derivative is a fixed coefficient per control.

    Parameters
    ----------
    controls : list of ControlSurface
        Controls of the surface (they do not overlap in span).
    kind : str
        ``"plain"`` (span parameter 0 to 1), ``"symmetric"`` (-1 to 1, the
        left half negative) or ``"mirror"`` (a left copy, -1 to 0).
    chord_of : callable
        Local chord [m] at a span parameter of the right half.
    rate_of : callable
        Length of the reference line [m] per unit span parameter.
    width : float
        Join length over the local chord [-].
    """

    _TOL = 1e-9

    def __init__(self, controls: list, kind: str, chord_of, rate_of, width: float = CONTROL_JOIN) -> None:
        self.pieces = []
        for c in controls:
            e0, e1 = float(c.eta_start), float(c.eta_end)
            sgn_left = 1.0 if c.symmetric else -1.0
            if kind in ("plain", "symmetric"):
                self.pieces.append((e0, e1, c, 1.0))
            if kind in ("symmetric", "mirror"):
                self.pieces.append((-e1, -e0, c, sgn_left))
        self.lo = 0.0 if kind == "plain" else -1.0
        self.hi = 0.0 if kind == "mirror" else 1.0
        tol = self._TOL
        cuts = sorted({b for p in self.pieces for b in p[:2] if self.lo + tol < b < self.hi - tol})
        merged: list[float] = []
        for b in cuts:
            if not merged or b - merged[-1] > tol:
                merged.append(b)
        joins = []
        for i, b in enumerate(merged):
            left, right = self.piece(b - 10.0 * tol), self.piece(b + 10.0 * tol)
            if (left is None and right is None) or (
                    left is not None and right is not None and left[2] is right[2] and left[3] == right[3]):
                continue
            w = float(width) * float(chord_of(abs(b))) / max(float(rate_of(abs(b))), 1e-300)
            half = 0.5 * w
            # Not longer than each control next to the edge.
            for side in (left, right):
                if side is not None:
                    half = min(half, 0.5 * (side[1] - side[0]))
            # Up to the root or the tip the whole room is free; the room to
            # another edge is shared with the join of that edge.
            prev = merged[i - 1] if i > 0 else None
            nxt = merged[i + 1] if i + 1 < len(merged) else None
            half = min(half, (b - self.lo) if prev is None else 0.5 * (b - prev))
            half = min(half, (self.hi - b) if nxt is None else 0.5 * (nxt - b))
            if half > tol:
                joins.append((b, half, left, right))
        self.joins = joins

    def piece(self, eta: float):
        """Return ``(start, end, control, side sign)`` of the control at signed span parameter *eta*, or None."""
        for p in self.pieces:
            if p[0] - self._TOL <= eta <= p[1] + self._TOL:
                return p
        return None

    def at(self, eta: float) -> tuple[list, object]:
        """Return the deflection terms and the hinge control at signed span parameter *eta*.

        Parameters
        ----------
        eta : float
            Signed span parameter [-] (negative on the left half).

        Returns
        -------
        terms : list of (ControlSurface, float)
            The deflection at *eta* is ``sum(coef * control.deflection)``
            [rad]; the coefficient holds the side sign and the join factor.
        hinge : ControlSurface or None
            The control whose hinge line the point uses (None: no deflected
            geometry at *eta*).
        """
        eta = float(eta)
        for b, half, left, right in self.joins:
            if b - half < eta < b + half:
                t = (eta - (b - half)) / (2.0 * half)
                terms: list = []
                for side, f in ((left, 1.0 - t), (right, t)):
                    if side is None:
                        continue
                    for k, (c, coef) in enumerate(terms):
                        if c is side[2]:
                            terms[k] = (c, coef + side[3] * f)
                            break
                    else:
                        terms.append((side[2], side[3] * f))
                own = left if (eta < b and left is not None) or right is None else right
                return terms, own[2]
        p = self.piece(eta)
        if p is None:
            return [], None
        return [(p[2], p[3])], p[2]
