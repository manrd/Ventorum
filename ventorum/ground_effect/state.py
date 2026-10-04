# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Define the state, condition and result datatypes for ground-effect analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Any
import numpy as np

from ventorum.core.datatypes import (
    SpanwiseResult,
    SolverResult,
    DiscretizedSurface,
)
from ventorum.core.constants import RHO_SL


@dataclass(slots=True)
class GroundEffectCondition:
    """Operational parameters for a wing or aircraft operating in ground proximity.

    The free stream is parallel to the flat ground (level flight). The
    aircraft keeps its body axes; the ground plane is tilted in body axes by
    the angle of attack and turned about the free-stream direction by the
    bank angle (see :func:`ventorum.aero.system.ground_normal`). With
    ``phi_deg = 0`` the wings are level, also in sideslip.

    Parameters
    ----------
    h : float
        Height [m] above the ground (finite and > 0) in the convention of
        *height_ref*. With ``height_ref='ref'``, the ground is at the
        distance h below *ref_point*.
    alpha_deg : float
        Pitch angle / angle of attack relative to horizontal ground [deg].
    phi_deg : float
        Roll / bank angle relative to ground [deg].
        Positive = right wing (starboard) dips down, left wing (port) rises up.
    beta_deg : float
        Sideslip / yaw angle relative to oncoming flow [deg].
    V_inf : float
        Free-stream velocity [m/s] (default 50.0).
    rho : float
        Air density [kg/m³] (default 1.225).
    ref_point : np.ndarray | None
        Moment reference point and the point for ``height_ref='ref'`` [x, y, z] [m].
        If None, the analysis uses ``Aircraft.moment_reference()``
        (``Aircraft.ref_point``, or the origin of the geometry axes if that
        is None).
    height_ref : {'ref', 'min', 'qc', 'te'}
        Convention determining how height *h* is defined:
        - ``'ref'``: Height of *ref_point* above the ground plane.
        - ``'min'``: Minimum clearance of any point on the aircraft to the ground.
        - ``'qc'``: Height of the root quarter-chord point above the ground.
        - ``'te'``: Height of the root trailing-edge point above the ground.
    """

    h: float = 1.0
    alpha_deg: float = 4.0
    phi_deg: float = 0.0
    beta_deg: float = 0.0
    V_inf: float = 50.0
    rho: float = RHO_SL
    ref_point: np.ndarray | None = None
    height_ref: Literal["ref", "min", "qc", "te"] = "ref"

    def clone(self) -> GroundEffectCondition:
        """Fast explicit clone."""
        return GroundEffectCondition(
            h=self.h,
            alpha_deg=self.alpha_deg,
            phi_deg=self.phi_deg,
            beta_deg=self.beta_deg,
            V_inf=self.V_inf,
            rho=self.rho,
            ref_point=self.ref_point.copy() if self.ref_point is not None else None,
            height_ref=self.height_ref,
        )


@dataclass(slots=True)
class GroundEffectResult:
    """Comprehensive aerodynamic and kinematic outputs for ground-effect analysis.

    Contains wind-axis and body-axis integrated force and moment coefficients,
    clearance metrics, strike limits, spanwise aerodynamic distributions, and
    underlying solver results.

    Attributes
    ----------
    S_ref : float
        Reference area [m^2].
    b_ref : float
        Reference span [m].
    c_ref : float
        Reference chord [m].
    q_inf : float
        Dynamic pressure [Pa].
    h_ref : float
        Height of the moment reference point above the ground [m].
    """

    condition: GroundEffectCondition
    # Reference values
    S_ref: float = 0.0
    b_ref: float = 0.0
    c_ref: float = 0.0
    q_inf: float = 0.0

    # Wind-axis forces: CL normal to the free stream in the body x-z plane,
    # CD along the free stream, CY positive to the right.
    CL: float = 0.0
    CDi: float = 0.0
    CDp: float | None = None
    CD: float = 0.0
    CY: float = 0.0
    # Moments about ref_point, standard convention (Cl > 0 right wing down,
    # Cm > 0 nose up, Cn > 0 nose right). Cl and Cn are in stability axes.
    # A bank-restoring rolling moment has Cl < 0 for phi > 0 (Cl_phi < 0).
    Cl: float = 0.0
    Cm: float = 0.0
    Cn: float = 0.0
    L_over_D: float = float("nan")  # L/D with the profile drag when polars give it (NaN at zero drag)
    e: float = 0.0             # span efficiency CL^2 / (pi AR CDi), > 1 in ground effect

    # Standard body axes (x forward, y right, z down) and body-axis moments.
    CX_body: float = 0.0
    CY_body: float = 0.0
    CZ_body: float = 0.0
    Cl_body: float = 0.0
    Cm_body: float = 0.0
    Cn_body: float = 0.0

    # Dimensional forces and moments [N, N·m]
    L: float = 0.0
    D: float = 0.0
    Y: float = 0.0
    Mx: float = 0.0
    My: float = 0.0
    Mz: float = 0.0

    # Clearance and kinematic metrics
    h_min: float = 0.0         # Minimum clearance of any leading/trailing-edge node [m]
    h_ref: float = 0.0         # Height of reference point above ground [m]
    h_tip_left: float = 0.0    # Left wingtip clearance [m]
    h_tip_right: float = 0.0   # Right wingtip clearance [m]
    # Bank angle [deg] at the first ground contact, None if not computed. If
    # no point touches up to the search limit (60 deg), the value is that
    # limit and strike_limit_found is False: the true limit is larger.
    phi_strike_limit: float | None = None
    strike_limit_found: bool | None = None  # None if the limit was not computed
    h_over_c: float = 0.0      # Height normalized by mean chord
    h_over_b: float = 0.0      # Height normalized by span

    # Spanwise distributions and geometry
    spanwise: list[SpanwiseResult] = field(default_factory=list)
    # Indices in spanwise / transformed_surfaces of the main surface and its
    # mirror copies (see Aircraft.reference_surfaces).
    reference_surface_indices: list[int] = field(default_factory=lambda: [0])
    transformed_surfaces: list[DiscretizedSurface] = field(default_factory=list)
    solver_result: SolverResult | None = None
    execution_time: float = 0.0
    h_min_over_c: float = 0.0  # Smallest clearance / c_ref (the validity parameter)
    n_chord: int = 1           # Chordwise panels used by the lattice
    solver_type: str = ""
    vertical_force_coefficient: float = 0.0  # Force normal to the ground / (q S)

    def summary(self) -> str:
        """Return formatted multi-line summary of ground-effect analysis."""
        cond = self.condition
        lines = [
            "=" * 70,
            " Ventorum GROUND EFFECT AERODYNAMICS SUMMARY",
            "=" * 70,
            f" Flight Condition:  V_inf = {cond.V_inf:.1f} m/s | rho = {cond.rho:.3f} kg/m³ | q_inf = {self.q_inf:.1f} Pa",
            f" Attitude:          Alpha(Pitch) = {cond.alpha_deg:.2f}° | Roll(Phi) = {cond.phi_deg:.2f}° | Beta(Yaw) = {cond.beta_deg:.2f}°",
            f" Heights:           h = {cond.h:.3f} m ({cond.height_ref}) | h/c = {self.h_over_c:.3f} | h_min/c = {self.h_min_over_c:.3f} | h/b = {self.h_over_b:.4f}",
            f" Model:             solver = {self.solver_type} | chordwise panels = {self.n_chord}",
            f" Clearances:        h_min = {self.h_min:.3f} m | Tip Left = {self.h_tip_left:.3f} m | Tip Right = {self.h_tip_right:.3f} m",
        ]
        if self.phi_strike_limit is not None:
            if self.strike_limit_found is False:
                lines.append(f" Strike Bank Limit: no contact up to ±{self.phi_strike_limit:.2f}° (search limit)")
            else:
                lines.append(f" Strike Bank Limit: Phi_max = ±{self.phi_strike_limit:.2f}°")
        lines.extend([
            "-" * 70,
            " Wind-Axis Aerodynamic Coefficients:",
            f"   Lift (CL)         : {self.CL:9.4f}   (L = {self.L:10.1f} N)",
            f"   Induced Drag (CDi): {self.CDi:9.6f}   (Di = {self.q_inf * self.S_ref * self.CDi:9.1f} N)",
            f"   Total Drag (CD)   : {self.CD:9.6f}   (D = {self.D:10.1f} N)",
            f"   Sideforce (CY)    : {self.CY:9.6f}   (Y = {self.Y:10.1f} N)",
            f"   Efficiency (L/D)  : {self.L_over_D:9.2f}   (e = {self.e:6.3f})",
            "-" * 70,
            " Ground-Effect Moments (about ref_point):",
            f"   Rolling Moment (Cl) : {self.Cl:9.6f}   (Mx = {self.Mx:10.2f} N·m)  {'[Restoring]' if self.Cl * cond.phi_deg < 0 else ''}",
            f"   Pitching Moment (Cm): {self.Cm:9.6f}   (My = {self.My:10.2f} N·m)",
            f"   Yawing Moment (Cn)  : {self.Cn:9.6f}   (Mz = {self.Mz:10.2f} N·m)",
            "=" * 70,
        ])
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize key metrics to dictionary."""
        return {
            "h": self.condition.h,
            "alpha_deg": self.condition.alpha_deg,
            "phi_deg": self.condition.phi_deg,
            "beta_deg": self.condition.beta_deg,
            "h_over_c": self.h_over_c,
            "h_over_b": self.h_over_b,
            "h_min": self.h_min,
            "h_tip_left": self.h_tip_left,
            "h_tip_right": self.h_tip_right,
            "phi_strike_limit": self.phi_strike_limit,
            "strike_limit_found": self.strike_limit_found,
            "CL": self.CL,
            "CDi": self.CDi,
            "CDp": self.CDp,
            "CD": self.CD,
            "CY": self.CY,
            "Cl": self.Cl,
            "Cm": self.Cm,
            "Cn": self.Cn,
            "L_over_D": self.L_over_D,
            "e": self.e,
            "L": self.L,
            "D": self.D,
            "Y": self.Y,
            "Mx": self.Mx,
            "My": self.My,
            "Mz": self.Mz,
            "CX_body": self.CX_body,
            "CY_body": self.CY_body,
            "CZ_body": self.CZ_body,
            "Cl_body": self.Cl_body,
            "Cm_body": self.Cm_body,
            "Cn_body": self.Cn_body,
            "execution_time": self.execution_time,
            "h_min_over_c": self.h_min_over_c,
            "n_chord": self.n_chord,
            "solver_type": self.solver_type,
        }
