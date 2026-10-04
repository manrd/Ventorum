# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Input-validation helpers.

Each function raises ``ValueError`` with a descriptive message if the input
is malformed.
"""

from __future__ import annotations

import numpy as np

from ventorum.legacy.core.datatypes import (
    Aircraft,
    FlightCondition,
    LiftingSurface,
    SolverSettings,
    TabulatedAirfoil,
    WingSection,
)


def validate_section(sec: WingSection, idx: int, surface_name: str) -> None:
    """Check a single :class:`WingSection` for obvious errors."""
    if not (0.0 <= sec.y_frac <= 1.0):
        raise ValueError(
            f"[{surface_name}] Section {idx}: y_frac={sec.y_frac} "
            f"must be in [0, 1]."
        )
    if sec.chord <= 0:
        raise ValueError(
            f"[{surface_name}] Section {idx}: chord={sec.chord} must be > 0."
        )
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
    if surf.semi_span <= 0:
        raise ValueError(
            f"[{surf.name}] semi_span={surf.semi_span} must be > 0."
        )
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

    if surf.n_panels is not None:
        if not isinstance(surf.n_panels, int) or surf.n_panels < 4:
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


def validate_aircraft(ac: Aircraft) -> None:
    """Validate all surfaces in an :class:`Aircraft`."""
    if not ac.surfaces:
        raise ValueError("Aircraft must have at least one LiftingSurface.")
    for surf in ac.surfaces:
        validate_surface(surf)


def validate_flight_condition(fc: FlightCondition) -> None:
    """Check :class:`FlightCondition` values."""
    if fc.V_inf <= 0:
        raise ValueError(f"V_inf={fc.V_inf} must be > 0.")
    if fc.rho <= 0:
        raise ValueError(f"rho={fc.rho} must be > 0.")
    if fc.h is not None and fc.h <= 0:
        raise ValueError(f"Altitude h={fc.h} must be > 0 (distance above ground plane).")


def validate_solver_settings(ss: SolverSettings) -> None:
    """Check :class:`SolverSettings` values."""
    valid_solvers = (
        "fourier", "horseshoe", "linear", "linear_llt", "nonlinear",
        "horseshoe_nonlinear", "gpu", "gpu_horseshoe", "gpu_nonlinear"
    )
    if ss.solver_type not in valid_solvers:
        raise ValueError(
            f"solver_type='{ss.solver_type}' must be one of {valid_solvers}."
        )
    if ss.n_panels < 4:
        raise ValueError(f"n_panels={ss.n_panels} must be >= 4.")
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
    if ss.tolerance <= 0:
        raise ValueError(f"tolerance={ss.tolerance} must be > 0.")
    if not (0 < ss.relaxation <= 1):
        raise ValueError(f"relaxation={ss.relaxation} must be in (0, 1].")


def validate_fourier_applicability(ac: Aircraft) -> None:
    """Warn / raise if the Fourier solver is used on incompatible geometry.

    The Fourier solver requires a single, symmetric, unswept, planar surface.
    """
    if len(ac.surfaces) > 1:
        raise ValueError(
            "The Fourier solver supports only a single lifting surface. "
            f"Got {len(ac.surfaces)} surfaces. Use 'horseshoe' instead."
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
                "Use 'horseshoe' instead."
            )
    if abs(surf.dihedral) > np.radians(0.5):
        raise ValueError(
            "The Fourier solver does not support dihedral "
            f"(dihedral={np.degrees(surf.dihedral):.1f}°). "
            "Use 'horseshoe' instead."
        )
