# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Hinged control surfaces: thin-airfoil theory and input validation.

This module provides the aerodynamic effectiveness factor and pitching-moment
derivative of a hinged flap from thin-airfoil theory, constructs deflected
airfoil models for lifting-line solvers, and validates control surface inputs.

References
----------
H. Glauert, The Elements of Aerofoil and Airscrew Theory, Cambridge
University Press, 1926.
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
    "deflected_airfoil",
    "validate_controls",
]


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
    University Press, 1926, chapter V.
    J. Katz and A. Plotkin, Low-Speed Aerodynamics, 2nd ed., Cambridge
    University Press, 2001, chapter 5.
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
    University Press, 1926, chapter V.
    J. Katz and A. Plotkin, Low-Speed Aerodynamics, 2nd ed., Cambridge
    University Press, 2001, chapter 5.
    """
    h = float(hinge_x_c)
    if h <= 0.0 or h >= 1.0:
        return 0.0
    theta_h = float(np.arccos(np.clip(1.0 - 2.0 * h, -1.0, 1.0)))
    return float(-0.5 * np.sin(theta_h) * (1.0 - np.cos(theta_h)))


def deflected_airfoil(airfoil: AirfoilType, delta: float, hinge_x_c: float) -> AirfoilType:
    """Construct an airfoil model for a strip with a deflected flap.

    Uses thin-airfoil theory of a hinged flap:
    Cl_new(alpha) = Cl(alpha + tau * delta),
    Cm_new(alpha) = Cm(alpha) + dCm_ddelta * delta,
    Cd_new(alpha) = Cd(alpha + tau * delta).

    Parameters
    ----------
    airfoil : AirfoilType
        Base airfoil model (LinearAirfoil or TabulatedAirfoil).
    delta : float
        Flap deflection angle [rad].
    hinge_x_c : float
        Hinge chordwise position as a fraction of local chord [-].

    Returns
    -------
    AirfoilType
        Deflected airfoil model. Returns the exact same object if delta == 0.0.

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
