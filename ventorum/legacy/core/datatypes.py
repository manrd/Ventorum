# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
All dataclasses for Ventorum: airfoil models, geometry, flight conditions, solver
settings, and result containers.

Design philosophy
-----------------
Every piece of data that flows between modules is a frozen or mutable dataclass.
Mutable arrays (numpy) live inside dataclasses that are *not* frozen so that
solvers can populate result fields incrementally.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal, Any

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# Airfoil section models
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class LinearAirfoil:
    """Linear (thin-airfoil) aerodynamic model for a 2D section.

    Parameters
    ----------
    name : str
        Descriptive label (e.g. ``"NACA0012"``).
    a0 : float
        Lift-curve slope [1/rad].  Default is 2π (thin-airfoil theory).
    alpha_L0 : float
        Zero-lift angle of attack [rad].  Positive for cambered airfoils with
        negative zero-lift AoA convention (i.e. ``alpha_L0 < 0`` lifts at
        ``alpha = 0``).
    Cd0 : float
        Zero-lift profile drag coefficient.
    Cm0 : float
        Zero-lift pitching moment coefficient.
    """

    name: str = "Thin Airfoil"
    a0: float = 2.0 * np.pi
    alpha_L0: float = 0.0
    Cd0: float = 0.0
    Cm0: float = 0.0

    def clone(self) -> LinearAirfoil:
        """Fast explicit clone avoiding reflection."""
        return LinearAirfoil(
            name=self.name,
            a0=self.a0,
            alpha_L0=self.alpha_L0,
            Cd0=self.Cd0,
            Cm0=self.Cm0,
        )

    def __copy__(self) -> LinearAirfoil:
        return self.clone()

    def __deepcopy__(self, memo: dict) -> LinearAirfoil:
        return self.clone()

    # -- convenience ------------------------------------------------------------
    def Cl(self, alpha: float | np.ndarray) -> float | np.ndarray:
        """Section lift coefficient at angle of attack *alpha* [rad]."""
        return self.a0 * (alpha - self.alpha_L0)

    def Cd(self, alpha: float | np.ndarray) -> float | np.ndarray:  # noqa: ARG002
        """Section drag coefficient (constant for linear model)."""
        if isinstance(alpha, (int, float)):
            return float(self.Cd0)
        return np.full(np.shape(alpha), self.Cd0, dtype=float)

    def Cm(self, alpha: float | np.ndarray) -> float | np.ndarray:  # noqa: ARG002
        """Section pitching moment coefficient (constant for linear model)."""
        if isinstance(alpha, (int, float)):
            return float(self.Cm0)
        return np.full(np.shape(alpha), self.Cm0, dtype=float)


@dataclass(slots=True)
class TabulatedAirfoil:
    """Nonlinear aerodynamic model from polar-data tables.

    The arrays must be sorted in ascending *alpha* order so that interpolation
    works correctly.

    Parameters
    ----------
    name : str
        Descriptive label (e.g. ``"NACA2412_Re1e6"``).
    alpha : np.ndarray
        Angle-of-attack array [rad].
    Cl_data : np.ndarray
        Lift-coefficient array (same length as *alpha*).
    Cd_data : np.ndarray
        Drag-coefficient array.
    Cm_data : np.ndarray | None
        Pitching-moment array (optional).
    Re : float | None
        Reynolds number this polar was generated / measured at.
    """

    name: str = "Tabulated Airfoil"
    alpha: np.ndarray = field(default_factory=lambda: np.array([]))
    Cl_data: np.ndarray = field(default_factory=lambda: np.array([]))
    Cd_data: np.ndarray = field(default_factory=lambda: np.array([]))
    Cm_data: np.ndarray | None = None
    Re: float | None = None
    _cached_a0: float | None = field(default=None, init=False, repr=False)
    _cached_alpha_L0: float | None = field(default=None, init=False, repr=False)

    def clone(self) -> TabulatedAirfoil:
        """Fast explicit clone avoiding reflection."""
        inst = TabulatedAirfoil(
            name=self.name,
            alpha=self.alpha.copy(),
            Cl_data=self.Cl_data.copy(),
            Cd_data=self.Cd_data.copy(),
            Cm_data=self.Cm_data.copy() if self.Cm_data is not None else None,
            Re=self.Re,
        )
        inst._cached_a0 = self._cached_a0
        inst._cached_alpha_L0 = self._cached_alpha_L0
        return inst

    def __copy__(self) -> TabulatedAirfoil:
        return self.clone()

    def __deepcopy__(self, memo: dict) -> TabulatedAirfoil:
        return self.clone()

    # -- convenience (linear interpolation) ------------------------------------
    def Cl(self, alpha: float | np.ndarray) -> float | np.ndarray:
        return np.interp(alpha, self.alpha, self.Cl_data)

    def Cd(self, alpha: float | np.ndarray) -> float | np.ndarray:
        return np.interp(alpha, self.alpha, self.Cd_data)

    def Cm(self, alpha: float | np.ndarray) -> float | np.ndarray:
        if self.Cm_data is not None:
            return np.interp(alpha, self.alpha, self.Cm_data)
        return np.zeros_like(np.atleast_1d(alpha), dtype=float).squeeze()

    def _compute_linear_fit(self) -> None:
        if len(self.alpha) < 2:
            self._cached_a0 = 2.0 * np.pi
            self._cached_alpha_L0 = 0.0
            return
        mask = np.abs(self.alpha) < np.radians(8.0)
        if mask.sum() < 2:
            mask = np.ones_like(self.alpha, dtype=bool)
        coeffs = np.polyfit(self.alpha[mask], self.Cl_data[mask], 1)
        self._cached_a0 = float(coeffs[0])
        self._cached_alpha_L0 = float(-coeffs[1] / coeffs[0]) if abs(coeffs[0]) > 1e-12 else 0.0

    @property
    def a0(self) -> float:
        """Approximate lift-curve slope from the linear region of the polar."""
        if self._cached_a0 is None:
            self._compute_linear_fit()
        return self._cached_a0

    @property
    def alpha_L0(self) -> float:
        """Approximate zero-lift angle of attack [rad]."""
        if self._cached_alpha_L0 is None:
            self._compute_linear_fit()
        return self._cached_alpha_L0


# Type alias for convenience
AirfoilType = LinearAirfoil | TabulatedAirfoil


# ═══════════════════════════════════════════════════════════════════════════════
# Wing geometry
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class WingSection:
    """Properties at a single spanwise station of a lifting surface.

    The user defines a small number of *sections* (at least root + tip);
    the solver interpolates between them when discretising into panels.

    Parameters
    ----------
    y_frac : float
        Spanwise position as a fraction of the semi-span, in ``[0, 1]``
        (0 = root, 1 = tip).
    chord : float
        Local chord length [m].
    twist : float
        Geometric twist angle [rad].  Positive → leading edge up (wash-in).
    x_le : float | None
        Leading-edge *x*-offset from the root LE [m].  If *None*, computed
        from the surface-level ``sweep_le`` parameter during discretisation.
    z_le : float | None
        Leading-edge *z*-offset from the root LE [m].  If *None*, computed
        from the surface-level ``dihedral`` parameter during discretisation.
    airfoil : AirfoilType
        Aerodynamic model for this section.
    """

    y_frac: float = 0.0
    chord: float = 1.0
    twist: float = 0.0
    x_le: float | None = None
    z_le: float | None = None
    airfoil: AirfoilType = field(default_factory=LinearAirfoil)

    def clone(self) -> WingSection:
        """Fast explicit clone."""
        af = self.airfoil.clone() if hasattr(self.airfoil, "clone") else self.airfoil
        return WingSection(
            y_frac=self.y_frac,
            chord=self.chord,
            twist=self.twist,
            x_le=self.x_le,
            z_le=self.z_le,
            airfoil=af,
        )

    def __copy__(self) -> WingSection:
        return self.clone()

    def __deepcopy__(self, memo: dict) -> WingSection:
        return self.clone()


@dataclass(slots=True)
class LiftingSurface:
    """A complete lifting surface (wing, horizontal tail, canard, etc.).

    Parameters
    ----------
    name : str
        Human-readable label.
    semi_span : float
        Semi-span length [m].
    sections : list[WingSection]
        Ordered list of cross-section definitions from root (``y_frac=0``)
        to tip (``y_frac=1``).  At least two sections are required.
    is_symmetric : bool
        If *True*, the surface is mirrored about the aircraft's xz-plane.
    position : np.ndarray
        ``[x, y, z]`` offset of the surface root LE from the aircraft
        reference point [m].
    incidence : float
        Surface incidence angle relative to the aircraft reference [rad].
    sweep_le : float
        Leading-edge sweep angle [rad].  Used to compute ``x_le`` at each
        section when the section's own ``x_le`` is *None*.
    dihedral : float
        Dihedral angle [rad].  Used to compute ``z_le`` at each section when
        the section's own ``z_le`` is *None*.
    """

    name: str = "Wing"
    semi_span: float = 5.0
    sections: list[WingSection] = field(default_factory=lambda: [
        WingSection(y_frac=0.0, chord=2.0),
        WingSection(y_frac=1.0, chord=2.0),
    ])
    is_symmetric: bool = True
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    incidence: float = 0.0
    sweep_le: float = 0.0
    dihedral: float = 0.0
    n_panels: int | None = None
    spacing: str | None = None
    control_deflections: dict[str, float] | None = None

    def clone(self) -> LiftingSurface:
        """Fast explicit clone."""
        return LiftingSurface(
            name=self.name,
            semi_span=self.semi_span,
            sections=[sec.clone() for sec in self.sections],
            is_symmetric=self.is_symmetric,
            position=self.position.copy(),
            incidence=self.incidence,
            sweep_le=self.sweep_le,
            dihedral=self.dihedral,
            n_panels=self.n_panels,
            spacing=self.spacing,
            control_deflections=dict(self.control_deflections) if self.control_deflections is not None else None,
        )

    def __copy__(self) -> LiftingSurface:
        return self.clone()

    def __deepcopy__(self, memo: dict) -> LiftingSurface:
        return self.clone()


@dataclass(slots=True)
class Aircraft:
    """Collection of lifting surfaces.

    Parameters
    ----------
    name : str
        Aircraft identifier.
    surfaces : list[LiftingSurface]
        All lifting surfaces to include in the analysis.
    S_ref : float | None
        Reference wing area [m²].  If *None*, computed from the first surface.
    b_ref : float | None
        Reference span [m].  If *None*, computed from the first surface.
    c_ref : float | None
        Reference chord (mean aerodynamic chord) [m].  If *None*, estimated
        from the first surface.
    """

    name: str = "Aircraft"
    surfaces: list[LiftingSurface] = field(default_factory=lambda: [
        LiftingSurface(),
    ])
    S_ref: float | None = None
    b_ref: float | None = None
    c_ref: float | None = None

    def clone(self) -> Aircraft:
        """Fast explicit clone."""
        return Aircraft(
            name=self.name,
            surfaces=[surf.clone() for surf in self.surfaces],
            S_ref=self.S_ref,
            b_ref=self.b_ref,
            c_ref=self.c_ref,
        )

    def __copy__(self) -> Aircraft:
        return self.clone()

    def __deepcopy__(self, memo: dict) -> Aircraft:
        return self.clone()

    def compute_reference_values(self) -> None:
        """Fill *S_ref*, *b_ref*, *c_ref* from the first surface if missing."""
        surf = self.surfaces[0]
        if self.b_ref is None:
            self.b_ref = surf.semi_span * (2.0 if surf.is_symmetric else 1.0)
        if self.S_ref is None:
            # Trapezoidal approximation
            secs = sorted(surf.sections, key=lambda s: s.y_frac)
            area = 0.0
            for i in range(len(secs) - 1):
                dy = (secs[i + 1].y_frac - secs[i].y_frac) * surf.semi_span
                area += 0.5 * (secs[i].chord + secs[i + 1].chord) * dy
            if surf.is_symmetric:
                area *= 2.0
            self.S_ref = area
        if self.c_ref is None:
            # Exact Mean Aerodynamic Chord (MAC) integration: (1/S_ref) * integral(c(y)^2 dy)
            secs = sorted(surf.sections, key=lambda s: s.y_frac)
            int_c2 = 0.0
            for i in range(len(secs) - 1):
                dy = (secs[i + 1].y_frac - secs[i].y_frac) * surf.semi_span
                c1, c2 = secs[i].chord, secs[i + 1].chord
                int_c2 += (dy / 3.0) * (c1**2 + c1 * c2 + c2**2)
            if surf.is_symmetric:
                int_c2 *= 2.0
            if self.S_ref and self.S_ref > 0:
                self.c_ref = int_c2 / self.S_ref
            else:
                self.c_ref = 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Flight conditions & solver settings
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class FlightCondition:
    """Free-stream conditions for the analysis.

    Parameters
    ----------
    V_inf : float
        Free-stream velocity [m/s].
    alpha : float
        Angle of attack [rad].
    beta : float
        Sideslip angle [rad] (only affects horseshoe solver).
    rho : float
        Air density [kg/m³].
    """

    V_inf: float = 50.0
    alpha: float = np.radians(5.0)
    beta: float = 0.0
    rho: float = 1.225
    h: float | None = None  # Altitude of the aircraft origin above the ground plane [m]
    phi: float = 0.0  # Bank / roll angle [rad] (default 0.0)

    @property
    def phi_deg(self) -> float:
        """Roll / bank angle in degrees."""
        return float(np.degrees(self.phi))

    @property
    def beta_deg(self) -> float:
        """Sideslip angle in degrees."""
        return float(np.degrees(self.beta))

    @property
    def alpha_deg(self) -> float:
        """Angle of attack in degrees."""
        return float(np.degrees(self.alpha))

    def clone(self) -> FlightCondition:
        """Fast explicit clone."""
        return FlightCondition(
            V_inf=self.V_inf,
            alpha=self.alpha,
            beta=self.beta,
            rho=self.rho,
            h=self.h,
            phi=self.phi,
        )

    def __copy__(self) -> FlightCondition:
        return self.clone()

    def __deepcopy__(self, memo: dict) -> FlightCondition:
        return self.clone()


@dataclass(slots=True)
class SolverSettings:
    """Configuration knobs for the LLT solvers.

    Parameters
    ----------
    solver_type : str
        ``"horseshoe"`` (default), ``"linear"`` (Modern Linear LLT), ``"fourier"``, or ``"nonlinear"``.
    n_panels : int
        Number of spanwise panels **per semi-span**.
    spacing : str
        ``"cosine"`` (default), ``"half-cosine"`` (tip-clustered), ``"root"``,
        ``"uniform"``, or ``"auto"`` (geometry-adaptive automatic spacing).
    max_iterations : int
        Maximum iterations for the nonlinear solver.
    tolerance : float
        Convergence threshold on ``max(|ΔΓ|)``.
    relaxation : float
        Blending factor ω for the nonlinear update:
        ``Γ_{new} = (1-ω)·Γ_{old} + ω·Γ_{computed}``.
    proportional_panels : bool
        If True, scales panel counts proportionally across surfaces based on span.
    min_panels : int
        Minimum panels per semi-span when proportional scaling is enabled.
    """

    solver_type: Literal["fourier", "horseshoe", "linear", "linear_llt", "nonlinear"] | str = "horseshoe"
    n_panels: int = 80
    spacing: Literal["auto", "cosine", "half-cosine", "root", "uniform"] | str = "cosine"
    max_iterations: int = 200
    tolerance: float = 1.0e-6
    relaxation: float = 0.3
    proportional_panels: bool = False
    min_panels: int = 8
    use_symmetry: bool = True  # Toggle symmetry plane on Y=0 (default True)

    def clone(self) -> SolverSettings:
        """Fast explicit clone."""
        return SolverSettings(
            solver_type=self.solver_type,
            n_panels=self.n_panels,
            spacing=self.spacing,
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
            relaxation=self.relaxation,
            proportional_panels=self.proportional_panels,
            min_panels=self.min_panels,
            use_symmetry=self.use_symmetry,
        )

    def __copy__(self) -> SolverSettings:
        return self.clone()

    def __deepcopy__(self, memo: dict) -> SolverSettings:
        return self.clone()


# ═══════════════════════════════════════════════════════════════════════════════
# Discretised geometry (internal, produced by the geometry processor)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class DiscretizedSurface:
    """Panel-level representation of a single lifting surface.

    Created by :func:`ventorum.legacy.geometry.processing.discretize_surface`.

    Convention: arrays indexed ``[i]`` where ``i = 0 … N-1`` for *N* panels.
    """

    # Bound-vortex node positions at the 1/4-chord line  [N+1, 3]
    nodes_qc: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    # Control-point positions at the 3/4-chord              [N, 3]
    control_points: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    # Outward surface normals at the control points          [N, 3]
    normals: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    # Panel-centre spanwise coordinate                       [N]
    y_panels: np.ndarray = field(default_factory=lambda: np.empty(0))
    # Panel widths                                           [N]
    dy_panels: np.ndarray = field(default_factory=lambda: np.empty(0))
    # Interpolated chord at each panel centre                [N]
    chords: np.ndarray = field(default_factory=lambda: np.empty(0))
    # Interpolated twist at each panel centre                [N]
    twists: np.ndarray = field(default_factory=lambda: np.empty(0))
    # Airfoil objects at each panel (nearest-section rule)
    airfoils: list[AirfoilType] = field(default_factory=list)
    # Metadata
    surface_name: str = ""
    surface_index: int = 0
    # Precomputed cached properties for force integration & polars
    panel_centers_qc: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    has_profile_drag: bool = False
    airfoil_groups: list[tuple[Any, np.ndarray]] = field(default_factory=list)
    is_half_mesh: bool = False  # True if this surface only contains the right semi-span (y >= 0)


@dataclass(slots=True)
class FourierStations:
    """1-D discretisation used exclusively by the Fourier solver.

    The Fourier solver does not need 3-D panel geometry; it only requires
    spanwise distributions of section properties at the collocation angles.
    """

    theta: np.ndarray = field(default_factory=lambda: np.empty(0))   # [N]
    y: np.ndarray = field(default_factory=lambda: np.empty(0))       # [N]
    chords: np.ndarray = field(default_factory=lambda: np.empty(0))  # [N]
    a0: np.ndarray = field(default_factory=lambda: np.empty(0))      # [N]
    alpha_L0: np.ndarray = field(default_factory=lambda: np.empty(0))  # [N]
    twists: np.ndarray = field(default_factory=lambda: np.empty(0))  # [N]
    Cd0: np.ndarray = field(default_factory=lambda: np.empty(0))     # [N]


# ═══════════════════════════════════════════════════════════════════════════════
# Results
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class SpanwiseResult:
    """Spanwise distributions for a **single** lifting surface."""

    y: np.ndarray = field(default_factory=lambda: np.empty(0))
    gamma: np.ndarray = field(default_factory=lambda: np.empty(0))
    Cl: np.ndarray = field(default_factory=lambda: np.empty(0))
    Cd_i: np.ndarray = field(default_factory=lambda: np.empty(0))
    Cd_profile: np.ndarray | None = None
    alpha_eff: np.ndarray = field(default_factory=lambda: np.empty(0))
    alpha_i: np.ndarray = field(default_factory=lambda: np.empty(0))
    local_lift: np.ndarray = field(default_factory=lambda: np.empty(0))
    surface_name: str = ""

    def clone(self) -> SpanwiseResult:
        """Fast explicit clone."""
        return SpanwiseResult(
            y=self.y.copy(),
            gamma=self.gamma.copy(),
            Cl=self.Cl.copy(),
            Cd_i=self.Cd_i.copy(),
            Cd_profile=self.Cd_profile.copy() if self.Cd_profile is not None else None,
            alpha_eff=self.alpha_eff.copy(),
            alpha_i=self.alpha_i.copy(),
            local_lift=self.local_lift.copy(),
            surface_name=self.surface_name,
        )

    def __copy__(self) -> SpanwiseResult:
        return self.clone()

    def __deepcopy__(self, memo: dict) -> SpanwiseResult:
        return self.clone()


@dataclass(slots=True)
class TrustScore:
    """Quantitative validity score and physical uncertainty assessment.

    Parameters
    ----------
    score : float
        Composite trust score in [0.0, 1.0]. 1.0 indicates perfect adherence
        to classical lifting-line assumptions; values < 0.4 indicate the flow
        physics severely violate LLT assumptions.
    rating : Literal["HIGH", "MODERATE", "LOW", "UNRELIABLE"]
        Qualitative categorization of analysis validity.
    uncertainty_CL : float
        Estimated absolute uncertainty margin (+/- Delta CL).
    uncertainty_CDi : float
        Estimated absolute uncertainty margin (+/- Delta CDi).
    uncertainty_LD : float
        Estimated absolute uncertainty margin (+/- Delta L/D).
    factors : dict[str, float]
        Individual penalty factor contributions (0.0 = no penalty).
    warnings : list[str]
        Specific physics alerts explaining what aerodynamic limits were approached.
    recommendations : list[str]
        Engineering advice on alternative higher-order methods (e.g. VLM, CFD).
    """

    score: float = 1.0
    rating: Literal["HIGH", "MODERATE", "LOW", "UNRELIABLE"] = "HIGH"
    uncertainty_CL: float = 0.02
    uncertainty_CDi: float = 0.001
    uncertainty_LD: float = 0.5
    factors: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def clone(self) -> TrustScore:
        """Fast explicit clone."""
        return TrustScore(
            score=self.score,
            rating=self.rating,
            uncertainty_CL=self.uncertainty_CL,
            uncertainty_CDi=self.uncertainty_CDi,
            uncertainty_LD=self.uncertainty_LD,
            factors=dict(self.factors),
            warnings=list(self.warnings),
            recommendations=list(self.recommendations),
        )

    def __copy__(self) -> TrustScore:
        return self.clone()

    def __deepcopy__(self, memo: dict) -> TrustScore:
        return self.clone()

    def to_dict(self) -> dict[str, Any]:
        """Convert the trust assessment to a JSON-serializable dictionary."""
        return {
            "score": round(float(self.score), 4),
            "rating": self.rating,
            "uncertainty_CL": round(float(self.uncertainty_CL), 4),
            "uncertainty_CDi": round(float(self.uncertainty_CDi), 6),
            "uncertainty_LD": round(float(self.uncertainty_LD), 2),
            "factors": {k: round(float(v), 4) for k, v in self.factors.items()},
            "warnings": list(self.warnings),
            "recommendations": list(self.recommendations),
        }

    def summary_str(self) -> str:
        """Concise one-line summary string for user interfaces and agent reasoning."""
        warn_str = f" | Warnings: {len(self.warnings)}" if self.warnings else " | Flow fully valid"
        return (
            f"Trust: {self.score:.2f} ({self.rating}) | "
            f"CL +/- {self.uncertainty_CL:.3f}, CDi +/- {self.uncertainty_CDi:.5f}"
            f"{warn_str}"
        )


@dataclass(slots=True)
class IntegratedResult:
    """Wing-level (or aircraft-level) integrated coefficients."""

    CL: float = 0.0
    CDi: float = 0.0
    CDp: float | None = None
    CD_total: float | None = None
    e: float = 0.0             # span efficiency factor
    CL_alpha: float | None = None
    AR: float = 0.0
    Cl: float = 0.0            # Rolling moment coefficient
    Cm: float = 0.0            # Pitching moment coefficient
    Cn: float = 0.0            # Yawing moment coefficient
    trust: TrustScore | None = None

    def clone(self) -> IntegratedResult:
        """Fast explicit clone."""
        return IntegratedResult(
            CL=self.CL,
            CDi=self.CDi,
            CDp=self.CDp,
            CD_total=self.CD_total,
            e=self.e,
            CL_alpha=self.CL_alpha,
            AR=self.AR,
            Cl=self.Cl,
            Cm=self.Cm,
            Cn=self.Cn,
            trust=self.trust.clone() if self.trust is not None else None,
        )

    def __copy__(self) -> IntegratedResult:
        return self.clone()

    def __deepcopy__(self, memo: dict) -> IntegratedResult:
        return self.clone()


@dataclass(slots=True)
class SolverResult:
    """Top-level container returned by every solver.

    Carries both per-surface spanwise data and aircraft-level totals, plus
    solver metadata (iteration count, residual history, etc.).
    """

    # Per-surface spanwise distributions (one entry per LiftingSurface)
    spanwise: list[SpanwiseResult] = field(default_factory=list)
    # Aircraft-level integrated coefficients
    totals: IntegratedResult = field(default_factory=IntegratedResult)
    # Solver metadata
    solver_type: str = ""
    converged: bool = True
    iterations: int = 0
    residual_history: list[float] = field(default_factory=list)
    # Fourier coefficients (only populated by the Fourier solver)
    fourier_coefficients: np.ndarray | None = None
    # Operating flight condition (optional)
    condition: FlightCondition | None = None
    execution_time: float = 0.0
    symmetry_used: bool = False  # Records if Y=0 symmetry plane was utilized

    def clone(self) -> SolverResult:
        """Fast explicit clone."""
        return SolverResult(
            spanwise=[sw.clone() for sw in self.spanwise],
            totals=self.totals.clone(),
            solver_type=self.solver_type,
            converged=self.converged,
            iterations=self.iterations,
            residual_history=list(self.residual_history),
            fourier_coefficients=self.fourier_coefficients.copy() if self.fourier_coefficients is not None else None,
            condition=self.condition.clone() if self.condition is not None else None,
            execution_time=self.execution_time,
            symmetry_used=self.symmetry_used,
        )

    def __copy__(self) -> SolverResult:
        return self.clone()

    def __deepcopy__(self, memo: dict) -> SolverResult:
        return self.clone()


# ═══════════════════════════════════════════════════════════════════════════════
# JSON serialisation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _airfoil_to_dict(af: AirfoilType) -> dict:
    """Serialise an airfoil to a plain dict."""
    if isinstance(af, LinearAirfoil):
        return {
            "type": "linear",
            "name": af.name,
            "a0": af.a0,
            "alpha_L0": af.alpha_L0,
            "Cd0": af.Cd0,
            "Cm0": af.Cm0,
        }
    return {
        "type": "tabulated",
        "name": af.name,
        "alpha": af.alpha.tolist(),
        "Cl": af.Cl_data.tolist(),
        "Cd": af.Cd_data.tolist(),
        "Cm": af.Cm_data.tolist() if af.Cm_data is not None else None,
        "Re": af.Re,
    }


def _airfoil_from_dict(d: dict) -> AirfoilType:
    """Deserialise an airfoil from a plain dict."""
    if d["type"] == "linear":
        return LinearAirfoil(
            name=d.get("name", ""),
            a0=d.get("a0", 2.0 * np.pi),
            alpha_L0=d.get("alpha_L0", 0.0),
            Cd0=d.get("Cd0", 0.0),
            Cm0=d.get("Cm0", 0.0),
        )
    return TabulatedAirfoil(
        name=d.get("name", ""),
        alpha=np.asarray(d["alpha"]),
        Cl_data=np.asarray(d["Cl"]),
        Cd_data=np.asarray(d["Cd"]),
        Cm_data=np.asarray(d["Cm"]) if d.get("Cm") is not None else None,
        Re=d.get("Re"),
    )


def aircraft_to_json(ac: Aircraft) -> str:
    """Serialise an :class:`Aircraft` to a JSON string."""
    surfaces = []
    for surf in ac.surfaces:
        sections = []
        for sec in surf.sections:
            sections.append({
                "y_frac": sec.y_frac,
                "chord": sec.chord,
                "twist": sec.twist,
                "x_le": sec.x_le,
                "z_le": sec.z_le,
                "airfoil": _airfoil_to_dict(sec.airfoil),
            })
        surfaces.append({
            "name": surf.name,
            "semi_span": surf.semi_span,
            "sections": sections,
            "is_symmetric": surf.is_symmetric,
            "position": surf.position.tolist(),
            "incidence": surf.incidence,
            "sweep_le": surf.sweep_le,
            "dihedral": surf.dihedral,
            "n_panels": surf.n_panels,
            "spacing": surf.spacing,
        })
    data = {
        "name": ac.name,
        "surfaces": surfaces,
        "S_ref": ac.S_ref,
        "b_ref": ac.b_ref,
        "c_ref": ac.c_ref,
    }
    return json.dumps(data, indent=2)


def aircraft_from_json(text: str) -> Aircraft:
    """Deserialise an :class:`Aircraft` from a JSON string."""
    data = json.loads(text)
    surfaces = []
    for sd in data["surfaces"]:
        sections = []
        for sec_d in sd["sections"]:
            sections.append(WingSection(
                y_frac=sec_d["y_frac"],
                chord=sec_d["chord"],
                twist=sec_d.get("twist", 0.0),
                x_le=sec_d.get("x_le"),
                z_le=sec_d.get("z_le"),
                airfoil=_airfoil_from_dict(sec_d["airfoil"]),
            ))
        surfaces.append(LiftingSurface(
            name=sd.get("name", "Surface"),
            semi_span=sd["semi_span"],
            sections=sections,
            is_symmetric=sd.get("is_symmetric", True),
            position=np.asarray(sd.get("position", [0, 0, 0])),
            incidence=sd.get("incidence", 0.0),
            sweep_le=sd.get("sweep_le", 0.0),
            dihedral=sd.get("dihedral", 0.0),
            n_panels=sd.get("n_panels", None),
            spacing=sd.get("spacing", None),
        ))
    return Aircraft(
        name=data.get("name", "Aircraft"),
        surfaces=surfaces,
        S_ref=data.get("S_ref"),
        b_ref=data.get("b_ref"),
        c_ref=data.get("c_ref"),
    )
