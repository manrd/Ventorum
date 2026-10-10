# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Input-validation helpers.

Each function raises ``ValueError`` with a descriptive message if the input
is malformed.
"""

from __future__ import annotations

import math
import numbers

import numpy as np

from ventorum.core.datatypes import (
    Aircraft,
    FlightCondition,
    LiftingSurface,
    SolverSettings,
    TabulatedAirfoil,
    WingSection,
)

#: Largest absolute value of a length or a position [m]. Only absurd values
#: are refused; this is not a flight envelope.
LENGTH_LIMIT_M = 1.0e5
#: Smallest semi-span and section chord [m] accepted. Smaller positive
#: values underflow to zero in products of lengths (area, q*S*b) and give
#: a division by zero in the loads. A wing below one micrometre is absurd.
MIN_SPAN_M = 1.0e-6
MIN_CHORD_M = 1.0e-6
#: Smallest planform area [m^2] accepted (unprojected, both halves).
MIN_AREA_M2 = 1.0e-12
#: Largest absolute value of an angle [rad].
ANGLE_LIMIT_RAD = math.pi
#: Largest free-stream speed [m/s].
SPEED_LIMIT_M_S = 1.0e4
#: Smallest free-stream speed [m/s] accepted. A smaller speed makes the
#: dynamic pressure underflow and gives a division by zero in the loads.
MIN_SPEED_M_S = 0.1


def _is_finite_number(value: object) -> bool:
    """Return True when *value* is a finite number."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _finite_value(label: str, value: float) -> float:
    """Return ``float(value)``; raise ``ValueError`` naming *label* when it is not finite."""
    if not _is_finite_number(value):
        raise ValueError(f"{label}: value={value!r} is not a finite number.")
    return float(value)


def _check_length(label: str, value: float) -> float:
    """Return ``float(value)`` [m]; refuse non-finite and absurd lengths, naming *label*."""
    number = _finite_value(label, value)
    if abs(number) > LENGTH_LIMIT_M:
        raise ValueError(
            f"{label}: |value|={abs(number)!r} m is above the size limit {LENGTH_LIMIT_M:g} m."
        )
    return number


def _check_angle(label: str, value: float) -> float:
    """Return ``float(value)`` [rad]; refuse non-finite and absurd angles, naming *label*."""
    number = _finite_value(label, value)
    if abs(number) > ANGLE_LIMIT_RAD:
        raise ValueError(
            f"{label}: |value|={abs(number)!r} rad is above the size limit pi rad."
        )
    return number


def _check_int(label: str, value: int) -> int:
    """Return ``int(value)``; refuse ``bool`` and non-integers, naming *label*."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{label}: value={value!r} must be an integer (bool is refused).")
    return int(value)


def validate_section(sec: WingSection, idx: int, surface_name: str) -> None:
    """Check a single :class:`WingSection` for obvious errors."""
    tag = f"[{surface_name}] Section {idx}"
    _finite_value(f"{tag}: y_frac", sec.y_frac)
    if not (0.0 <= sec.y_frac <= 1.0):
        raise ValueError(
            f"[{surface_name}] Section {idx}: y_frac={sec.y_frac} "
            f"must be in [0, 1]."
        )
    _check_length(f"{tag}: chord", sec.chord)
    if sec.chord < MIN_CHORD_M:
        raise ValueError(
            f"[{surface_name}] Section {idx}: chord={sec.chord} m must be >= {MIN_CHORD_M:g} m."
        )
    _check_angle(f"{tag}: twist", sec.twist)
    if sec.x_le is not None:
        _check_length(f"{tag}: x_le", sec.x_le)
    if sec.z_le is not None:
        _check_length(f"{tag}: z_le", sec.z_le)
    if isinstance(sec.airfoil, TabulatedAirfoil):
        n = len(sec.airfoil.alpha)
        if n < 2:
            raise ValueError(
                f"[{surface_name}] Section {idx}: tabulated airfoil "
                f"'{sec.airfoil.name}' needs at least 2 data points."
            )
        if len(sec.airfoil.Cl_data) != n or len(sec.airfoil.Cd_data) != n:
            raise ValueError(
                f"[{surface_name}] Section {idx}: tabulated airfoil "
                f"'{sec.airfoil.name}' has mismatched array lengths."
            )


def validate_surface(surf: LiftingSurface) -> None:
    """Check a :class:`LiftingSurface` for completeness and consistency."""
    _check_length(f"[{surf.name}] semi_span", surf.semi_span)
    if surf.semi_span < MIN_SPAN_M:
        raise ValueError(
            f"[{surf.name}] semi_span={surf.semi_span} m must be >= {MIN_SPAN_M:g} m."
        )
    try:
        position = np.asarray(surf.position, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"[{surf.name}] position={surf.position!r} is not a number array."
        ) from exc
    for k, comp in enumerate(np.ravel(position)):
        _check_length(f"[{surf.name}] position[{k}]", float(comp))
    _check_angle(f"[{surf.name}] incidence", surf.incidence)
    _check_angle(f"[{surf.name}] sweep_le", surf.sweep_le)
    _check_angle(f"[{surf.name}] dihedral", surf.dihedral)
    if len(surf.sections) < 2:
        raise ValueError(
            f"[{surf.name}] At least 2 sections (root + tip) are required. "
            f"Got {len(surf.sections)}."
        )
    # Sections must be sorted by y_frac
    fracs = [s.y_frac for s in surf.sections]
    if fracs != sorted(fracs):
        raise ValueError(
            f"[{surf.name}] Sections must be ordered by y_frac. "
            f"Got: {fracs}."
        )
    # Root at 0, tip at 1
    if surf.sections[0].y_frac != 0.0:
        raise ValueError(
            f"[{surf.name}] First section y_frac must be 0.0 (root). "
            f"Got {surf.sections[0].y_frac}."
        )
    if surf.sections[-1].y_frac != 1.0:
        raise ValueError(
            f"[{surf.name}] Last section y_frac must be 1.0 (tip). "
            f"Got {surf.sections[-1].y_frac}."
        )
    for i, sec in enumerate(surf.sections):
        validate_section(sec, i, surf.name)

    # Explicit z_le changes the meaning of semi_span (projected span instead
    # of the length along the dihedral line), so it must be given for every
    # section or for none.
    n_z = sum(sec.z_le is not None for sec in surf.sections)
    if 0 < n_z < len(surf.sections):
        raise ValueError(
            f"[{surf.name}] z_le is given for {n_z} of {len(surf.sections)} sections. Give z_le for every "
            "section (semi_span is then the projected span) or for none (semi_span is then the length "
            "along the dihedral line)."
        )
    if n_z and abs(surf.dihedral) > 0.0:
        raise ValueError(
            f"[{surf.name}] z_le is given for the sections and dihedral={surf.dihedral} is also set; "
            "use one of them."
        )
    if getattr(surf, "mirror_y", False) and surf.is_symmetric:
        raise ValueError(f"[{surf.name}] mirror_y=True needs is_symmetric=False.")

    if surf.n_panels is not None:
        surf.n_panels = _check_int(f"[{surf.name}] n_panels", surf.n_panels)
        if surf.n_panels < 4:
            raise ValueError(f"[{surf.name}] n_panels={surf.n_panels} must be an integer >= 4.")
    if surf.spacing is not None:
        valid_spacings = (
            "auto", "cosine", "full-cosine", "full_cosine", "both",
            "half-cosine", "half_cosine", "tip", "tip-clustered",
            "root", "root-cosine", "root_cosine", "root-clustered",
            "uniform", "power", "stretched",
        )
        if surf.spacing.lower() not in valid_spacings:
            raise ValueError(
                f"[{surf.name}] Unknown spacing={surf.spacing!r}. Options: 'auto', 'cosine', 'half-cosine', 'root', 'uniform', 'power'."
            )
    # Planform area (unprojected: semi_span is the length along the
    # dihedral line unless z_le is given). The unprojected area is used
    # so that a vertical fin (no projected area) still passes.
    fracs = [s.y_frac for s in surf.sections]
    chords = [s.chord for s in surf.sections]
    area = surf.semi_span * sum(
        0.5 * (a + b) * (g - f)
        for a, b, f, g in zip(chords[:-1], chords[1:], fracs[:-1], fracs[1:])
    )
    if surf.is_symmetric:
        area *= 2.0
    if area < MIN_AREA_M2:
        raise ValueError(
            f"[{surf.name}] planform area={area} m^2 must be >= {MIN_AREA_M2:g} m^2."
        )


def validate_aircraft(ac: Aircraft) -> None:
    """Validate all surfaces in an :class:`Aircraft`."""
    if not ac.surfaces:
        raise ValueError("Aircraft must have at least one LiftingSurface.")
    for surf in ac.surfaces:
        validate_surface(surf)
    # Lower limits on the reference values. A smaller positive value
    # underflows or overflows in the coefficients (q*S, q*S*b, q*S*c).
    for key, floor, unit in (("S_ref", MIN_AREA_M2, "m^2"), ("b_ref", MIN_SPAN_M, "m"),
                             ("c_ref", MIN_CHORD_M, "m")):
        value = getattr(ac, key)
        if value is not None:
            _finite_value(f"Aircraft '{ac.name}': {key}", value)
            if value < floor:
                raise ValueError(
                    f"Aircraft '{ac.name}': {key}={value} {unit} must be >= {floor:g} {unit}."
                )
    if ac.ref_point is not None:
        try:
            ref = np.asarray(ac.ref_point, dtype=float).reshape(3)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Aircraft '{ac.name}': ref_point={ac.ref_point!r} must be 3 numbers [m]."
            ) from exc
        for k, comp in enumerate(ref):
            _check_length(f"Aircraft '{ac.name}': ref_point[{k}]", float(comp))


def validate_flight_condition(fc: FlightCondition) -> None:
    """Check :class:`FlightCondition` values."""
    _finite_value("V_inf", fc.V_inf)
    if fc.V_inf < MIN_SPEED_M_S:
        raise ValueError(f"V_inf={fc.V_inf} m/s must be >= {MIN_SPEED_M_S:g} m/s.")
    if fc.V_inf > SPEED_LIMIT_M_S:
        raise ValueError(
            f"V_inf={fc.V_inf} m/s is above the size limit {SPEED_LIMIT_M_S:g} m/s."
        )
    _check_angle("alpha", fc.alpha)
    _check_angle("beta", fc.beta)
    _check_angle("phi", fc.phi)
    for rate in ("p", "q", "r"):
        value = getattr(fc, rate, None)
        if value is not None:
            _finite_value(rate, value)
    _finite_value("rho", fc.rho)
    if fc.rho <= 0:
        raise ValueError(f"rho={fc.rho} must be > 0.")
    if fc.h is not None:
        if not _is_finite_number(fc.h):
            raise ValueError(f"Height h={fc.h} is not a finite number.")
        if fc.h <= 0:
            raise ValueError(f"Height h={fc.h} must be > 0 (distance above the ground plane).")
        if abs(float(fc.h)) > LENGTH_LIMIT_M:
            raise ValueError(
                f"Height h={fc.h} m is above the size limit {LENGTH_LIMIT_M:g} m."
            )
    from ventorum.core.constants import A_SL, MACH_LIMIT_INCOMPRESSIBLE
    mach = fc.V_inf / A_SL
    if mach > MACH_LIMIT_INCOMPRESSIBLE:
        import warnings
        warnings.warn(
            f"Mach {mach:.2f} (V_inf={fc.V_inf} m/s, sea-level speed of sound) is above the incompressible "
            f"limit {MACH_LIMIT_INCOMPRESSIBLE}: the case is out of the valid envelope.",
            RuntimeWarning, stacklevel=3,
        )


def validate_solver_settings(ss: SolverSettings) -> None:
    """Check :class:`SolverSettings` values."""
    from ventorum.solvers.factory import VALID_SOLVER_NAMES as valid_solvers
    if str(ss.solver_type).lower() not in valid_solvers:
        raise ValueError(
            f"solver_type='{ss.solver_type}' must be one of {valid_solvers}."
        )
    if ss.n_panels is None:
        raise ValueError("n_panels=None is not valid; give an integer >= 4.")
    ss.n_panels = _check_int("n_panels", ss.n_panels)
    if ss.n_panels < 4:
        raise ValueError(f"n_panels={ss.n_panels} must be >= 4.")
    n_chord = getattr(ss, "n_chord", None)
    if n_chord is not None:
        ss.n_chord = _check_int("n_chord", n_chord)
        if ss.n_chord < 1:
            raise ValueError(f"n_chord={ss.n_chord} must be a positive integer or None (automatic).")
    min_panels = getattr(ss, "min_panels", None)
    if min_panels is not None:
        ss.min_panels = _check_int("min_panels", min_panels)
        if ss.min_panels < 1:
            raise ValueError(f"min_panels={ss.min_panels} must be a positive integer.")
    if getattr(ss, "chord_spacing", "uniform") not in ("uniform", "cosine"):
        raise ValueError(f"chord_spacing={ss.chord_spacing!r} must be 'uniform' or 'cosine'.")
    if getattr(ss, "wake_alignment", "freestream") not in ("freestream", "body"):
        raise ValueError(f"wake_alignment={ss.wake_alignment!r} must be 'freestream' or 'body'.")
    valid_spacings = (
        "auto", "cosine", "full-cosine", "full_cosine", "both",
        "half-cosine", "half_cosine", "tip", "tip-clustered",
        "root", "root-cosine", "root_cosine", "root-clustered",
        "uniform", "power", "stretched",
    )
    if ss.spacing.lower() not in valid_spacings:
        raise ValueError(
            f"spacing='{ss.spacing}' is invalid. Options: 'auto', 'cosine', 'half-cosine', 'root', 'uniform', 'power'."
        )
    if ss.tolerance is None:
        raise ValueError("tolerance=None is not valid; give a finite number > 0.")
    _finite_value("tolerance", ss.tolerance)
    if ss.tolerance <= 0:
        raise ValueError(f"tolerance={ss.tolerance} must be > 0.")


def validate_fourier_applicability(ac: Aircraft) -> None:
    """Warn / raise if the Fourier solver is used on incompatible geometry.

    The Fourier solver requires a single, symmetric, unswept, planar surface.
    """
    if len(ac.surfaces) > 1:
        raise ValueError(
            "The Fourier solver supports only a single lifting surface. "
            f"Got {len(ac.surfaces)} surfaces. Use 'vlm' instead."
        )
    surf = ac.surfaces[0]
    if not surf.is_symmetric:
        raise ValueError(
            "The Fourier solver requires a symmetric surface "
            f"(is_symmetric=True). Surface '{surf.name}' is not symmetric."
        )
    if abs(surf.sweep_le) > np.radians(0.5):
        has_explicit_sweep = any(
            s.x_le is not None and abs(s.x_le) > 1e-6 for s in surf.sections
        )
        if has_explicit_sweep or abs(surf.sweep_le) > np.radians(0.5):
            raise ValueError(
                "The Fourier solver does not support swept wings "
                f"(sweep_le={np.degrees(surf.sweep_le):.1f}°). "
                "Use 'vlm' instead."
            )
    if abs(surf.dihedral) > np.radians(0.5):
        raise ValueError(
            "The Fourier solver does not support dihedral "
            f"(dihedral={np.degrees(surf.dihedral):.1f}°). "
            "Use 'vlm' instead."
        )
