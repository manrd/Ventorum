# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Validity checks and trust score for a solution.

The score (0 to 1) adds penalties for conditions where the method in use is
known to be weak. The rules depend on the method:

* Lifting-line methods (``linear``, ``nonlinear``, ``fourier``) need a high
  aspect ratio and no sweep: with sweep the numerical lifting line is not
  grid convergent.
* The vortex-lattice method (``vlm``) handles sweep and low aspect ratio,
  but not the leading-edge vortex of slender, highly swept wings.
* All methods are inviscid and incompressible and use thin-wing theory. In
  ground effect, thickness and viscous effects grow when the gap is small
  compared with the chord; the limits below are stated in h/c.
* With tabulated section polars, the stall limits come from each polar
  (its maximum Cl and the angle of that maximum), and a section angle
  outside the table (where the end values are used) lowers the score.
* A wing of low aspect ratio at a high angle of attack gets vortex lift
  from its side edges, which the potential-flow methods do not model:
  E. C. Polhamus, "A concept of the vortex lift of sharp-edge delta wings
  based on a leading-edge-suction analogy", NASA TN D-3767, 1966;
  J. E. Lamar, "Extension of leading-edge-suction analogy to wings with
  separated flow around the side edges at subsonic speeds", NASA TR R-428,
  1974.
* The 3D panel method (``panel3d``) adds mesh rules: the panel shape and
  validity, and two indicators of the discretization error (the
  near-field/far-field drag difference and the chordwise panel count). A
  mesh-error study of the method calibrated their thresholds (see
  :func:`panel3d_mesh_checks` and the ``PANEL3D_*`` constants).

The uncertainty values (``uncertainty_CL`` and the others) are heuristic
bands. They are NOT calibrated against experimental data. Do not use them as
error bars in a report.
"""

from __future__ import annotations

import math

from ventorum.core.constants import A_SL, MACH_LIMIT_INCOMPRESSIBLE, RHO_SL

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ventorum.aero.system import GroundPlane
    from ventorum.geometry.lattice import VortexLattice

from ventorum.core.datatypes import (
    DiscretizedSurface,
    FlightCondition,
    LiftingSurface,
    SpanwiseResult,
    TabulatedAirfoil,
    TrustScore,
)

LIFTING_LINE_SOLVERS = ("linear", "linear_llt", "llt", "nonlinear", "fourier")

# ISA constants for the speed of sound from the air density.
_ISA_T0 = 288.15                 # Sea-level temperature [K]
_ISA_RHO_TROPOPAUSE = 0.36392    # Density at 11 000 m [kg/m^3]
_ISA_T_TROPOPAUSE = 216.65       # Temperature at and above 11 000 m [K]
_ISA_DENSITY_EXPONENT = 4.25588  # rho/rho0 = (T/T0)^4.25588 in the troposphere
_GAMMA_R_AIR = 1.4 * 287.05287   # Ratio of specific heats times the gas constant of air [J/(kg K)]

#: Threshold for the trailing-edge clearance trust warning on Trefftz CDi [-].
_CDI_TE_CLEARANCE = 0.15

#: Largest change of the strip lift between neighbouring strips of a mean-line lattice before the
#: conditioning warning [-]. Converged cambered wings stay below about 0.25 (NACA 2412 to 6409).
_MEAN_LINE_STRIP_JUMP = 0.3

#: Thresholds of the low-aspect-ratio side-edge vortex rule (vortex lattice).
LOW_AR_SIDE_EDGE_AR = 2.0
LOW_AR_SIDE_EDGE_ALPHA_DEG = 10.0

#: Near-field and far-field lateral comparison: relative limit [-] and the
#: absolute floors of CY and Cn [-]. A warning needs both limits passed.
NEAR_FAR_RELATIVE = 0.05
NEAR_FAR_FLOOR_CY = 0.001
NEAR_FAR_FLOOR_CN = 0.0002

#: Mesh-adequacy thresholds of the 3D panel method (Ventorum-3DP).
#:
#: These thresholds were calibrated by a mesh-error study of the method. Wings
#: of different aspect ratio, taper, sweep, dihedral, section (symmetric and
#: cambered), ground clearance and flap deflection were solved on a grid of
#: chordwise and spanwise panel counts with the default spacings. The error of
#: each mesh is the relative difference of CL and of the Trefftz-plane CDi from
#: a reference value: the Richardson extrapolation of separate chordwise and
#: spanwise refinements (``ventorum.panel3d.convergence.order_and_extrapolation``),
#: checked on a sphere and an ellipsoid against the exact solutions
#: (``ventorum.reference.analytic3d``). Each indicator below was compared with
#: that error:
#:
#: * A *warning* threshold marks the meshes with an error above 2 % in CL or
#:   4 % in CDi. A *strong* threshold marks an error above 5 % in CL or 10 % in
#:   CDi. The CDi limits are twice the CL limits because the induced drag goes
#:   with the square of the lift: one circulation error gives about twice the
#:   relative error in CDi.
#: * The chordwise panel count and the near-field/far-field drag difference
#:   predicted the error. The largest panel aspect ratio and the largest
#:   skew did not: on a normal wing mesh they come from the trailing-edge
#:   base panels, the tip caps, the triangles at the edges of a control
#:   surface and the short panels of the cosine spacing, not from a mesh
#:   error. The shape rules therefore use the median panel. At a fixed
#:   chordwise count, a large median aspect ratio means few spanwise strips
#:   and a larger error, so the aspect-ratio limit is a calibrated warning
#:   that adds to the drag rule. The skew limit and the warp limit were not
#:   calibrated: the median skew of the study meshes stayed well below the
#:   limit (the swept wing had the largest), and no study mesh had warped
#:   panels.
#: * Panels that are not convex (a folded strip) gave results with no value.
#:
#: Recalibration: solve the same kind of wing set on a grid of panel counts,
#: compute the reference values in the same way, and set each threshold to
#: the value that best separates the meshes above the error limit from the
#: meshes below it (the largest balanced accuracy; on a flat optimum, a
#: round value inside it). Keep the error limits of the two levels, so that
#: the meaning of a warning does not change.
PANEL3D_ASPECT_RATIO_PERCENTILE = 50.0
PANEL3D_MAX_ASPECT_RATIO = 30.0
PANEL3D_SKEW_PERCENTILE = 50.0
PANEL3D_MAX_SKEW = 0.85
PANEL3D_MAX_WARP = 0.10
PANEL3D_PENALTY_ASPECT_RATIO = 0.15
PANEL3D_PENALTY_SKEW = 0.20
PANEL3D_PENALTY_WARP = 0.20
PANEL3D_PENALTY_WATERTIGHT = 0.35
PANEL3D_PENALTY_NONCONVEX = 0.60

#: Discretization-error indicators of the 3D panel method (calibrated as
#: above). Near-field pressure drag against far-field Trefftz induced drag:
#: relative difference ``|CD_nearfield - CDi| / |CDi|`` of the warning and of
#: the strong warning, and the absolute floor of the difference [-] below
#: which the rule does not act (at zero lift both drags are zero). Chordwise
#: panel count per surface: the smallest count without a warning and without
#: a strong warning [-]. The penalty of the worse of the two indicators is
#: the factor ``mesh_resolution``.
PANEL3D_DRAG_WARN_REL = 0.15
PANEL3D_DRAG_STRONG_REL = 0.45
PANEL3D_DRAG_ABSOLUTE_FLOOR = 0.0005
PANEL3D_N_CHORD_WARN = 28
PANEL3D_N_CHORD_STRONG = 14
PANEL3D_PENALTY_RESOLUTION_WARN = 0.20
PANEL3D_PENALTY_RESOLUTION_STRONG = 0.40

#: Far-field kernel trust note in ground effect [-].
PANEL3D_FAR_FIELD_GROUND_NOTE = (
    "The default far-field kernel (second-moment expansion about the panel centroid) agreed with "
    "exact integration within about 0.01 % in lift, moment and induced drag in the tested ground "
    "cases; use far_ratio=inf for reference values."
)

#: Far-field kernel trust note in free air with fine spanwise refinement [-].
PANEL3D_FAR_FIELD_FREE_AIR_NOTE = (
    "The default far-field kernel (second-moment expansion about the panel centroid) agreed with "
    "exact integration within about 0.01 % in the tested free-air cases with more than 80 spanwise "
    "panels; use far_ratio=inf for reference values."
)

#: Outer-strip section load trust note in ground effect [-].
PANEL3D_OUTER_STRIP_GROUND_NOTE = (
    "In ground effect, the strip loads of the outermost strip of a closed tip are not "
    "resolved (sharp-edge feature); CL and Cm are not affected by more than the mesh error."
)

#: Dihedral root thickness scaling trust warning [-].
PANEL3D_ROOT_SCALE_WARNING = (
    "The root thickness scaling is not verified above 75 deg dihedral (lift can change by more than 10 %)."
)

#: Trust warning of a steep root dihedral without the root thickness scaling
#: [-]. The limit is ``ventorum.panel3d.mesh.STEEP_ROOT_DIHEDRAL_DEG``.
PANEL3D_STEEP_ROOT_UNSCALED_WARNING = (
    "Steep root without the root thickness scaling: the first panel row of a symmetric surface has a "
    "dihedral of 60 deg or more and scale_root_thickness is False. The pressure coefficient at the root is "
    "singular, so the loads are not reliable."
)

#: Trust note of a steep root dihedral with the root thickness scaling [-].
#: The limit is ``ventorum.panel3d.mesh.STEEP_ROOT_DIHEDRAL_DEG``.
PANEL3D_STEEP_ROOT_CONVERGENCE_NOTE = (
    "Steep root: the first panel row of a symmetric surface has a dihedral of 60 deg or more. The lift "
    "converges slowly with the spanwise panel count, so the result needs a spanwise convergence study."
)

#: Penalty for non-unique post-stall solution branches in viscous coupling.
POST_STALL_BRANCH_PENALTY = 0.35


def control_chord_note(n_chord: int) -> str:
    """Return the trust note of a deflected control with too few chordwise panels.

    The flap effectiveness of the vortex lattice converges at first order
    with the chordwise panel count: about 40/n_chord % below the converged
    value (rectangular wing of aspect ratio 40, full-span flap with the
    hinge at 0.75 c, against Glauert's tau). The note does not change the
    score.

    Parameters
    ----------
    n_chord : int
        Chordwise panels of the vortex lattice [-], 2 or more.

    Returns
    -------
    str
        The note, with the expected error in percent.
    """
    error_pct = 40.0 / float(n_chord)
    return (
        f"Deflected control with n_chord = {int(n_chord)}: flap effectiveness about {error_pct:.3g} % low "
        "(first-order chordwise convergence). The automatic n_chord uses 16 or more chordwise panels "
        "when a surface has a control."
    )


def _negligible_side_force(CY: float | None, CY_farfield: float | None) -> bool:
    """Return True when the near-field and far-field side forces are below the CY floor."""
    values = [v for v in (CY, CY_farfield) if v is not None and _is_finite(v)]
    return all(abs(float(v)) <= NEAR_FAR_FLOOR_CY for v in values)


def near_far_lateral_warnings(
    CY: float | None = None,
    CY_farfield: float | None = None,
    Cn: float | None = None,
    Cn_trefftz: float | None = None,
) -> list[str]:
    """Return the warnings of the near-field and far-field lateral comparison.

    The near-field value must be in the axes of the far-field value: ``CY``
    in wind axes (the side force normal to the free stream) and ``Cn`` in
    stability axes. For each coefficient, the rule warns when
    ``|near - far| > 0.05 |far|`` and ``|near - far|`` is larger than the
    floor (0.001 for CY, 0.0002 for Cn). A pair with a None or a non-finite
    value gives no warning.

    The Cn pair is compared only when each available (finite) side force,
    near and far field, is at or below the CY floor. The Trefftz plane has no chordwise position, so
    ``Cn_trefftz`` does not contain the yawing moment of a side force that
    acts ahead of or behind the reference point; with a side force the two
    Cn values differ for that reason alone.

    Parameters
    ----------
    CY : float or None
        Near-field side-force coefficient in wind axes [-].
    CY_farfield : float or None
        Side-force coefficient of the Trefftz plane in wind axes [-].
    Cn : float or None
        Near-field yawing-moment coefficient in stability axes [-].
    Cn_trefftz : float or None
        Yawing-moment coefficient of the Trefftz-plane induced drag in
        stability axes [-].

    Returns
    -------
    list of str
        One warning per coefficient that fails the rule (empty if none).
    """
    out: list[str] = []
    pairs = (("CY", "CY_farfield", "wind axes", CY, CY_farfield, NEAR_FAR_FLOOR_CY),
             ("Cn", "Cn_trefftz", "stability axes", Cn, Cn_trefftz, NEAR_FAR_FLOOR_CN))
    for name, far_name, axes, near, far, floor in pairs:
        if near is None or far is None or not (_is_finite(near) and _is_finite(far)):
            continue
        if name == "Cn" and not _negligible_side_force(CY, CY_farfield):
            continue
        near_v, far_v = float(near), float(far)
        diff = abs(near_v - far_v)
        if diff > NEAR_FAR_RELATIVE * abs(far_v) and diff > floor:
            out.append(
                f"Near field and far field do not agree for {name} ({axes}): near field {near_v:.6g}, "
                f"{far_name} {far_v:.6g}, difference {diff:.3g}. Probable causes: a junction of surfaces that "
                "the mesh does not resolve (for example a fin on a tail tip, a winglet, or joined edges with "
                "chordwise grids or strip widths that do not match), or a deflected control, whose trailing-edge "
                "displacement the Trefftz-plane trace does not include (see the theory manual, forces and moments)."
            )
    return out


def trailing_edge_clearance(lattice: VortexLattice | None, ground: GroundPlane | None) -> float | None:
    """Return the smallest trailing-edge height divided by the local chord.

    Parameters
    ----------
    lattice : VortexLattice or None
        Vortex lattice geometry.
    ground : GroundPlane or None
        Ground plane, or None for free air.

    Returns
    -------
    float or None
        Smallest trailing-edge clearance as a fraction of the local chord [-],
        or None in free air (when *ground* is None).
    """
    if ground is None or lattice is None or getattr(lattice, "n_strips", 0) == 0:
        return None
    if not (hasattr(lattice, "te_left") and hasattr(lattice, "te_right") and hasattr(lattice, "chord")):
        return None
    h_l = ground.height(lattice.te_left)
    h_r = ground.height(lattice.te_right)
    h_min = np.minimum(h_l, h_r)
    chord = np.maximum(np.asarray(lattice.chord, dtype=float), 1e-300)
    return float(np.min(h_min / chord))


def mean_line_strip_jump(lattice: VortexLattice | None, spanwise_list: Sequence | None) -> float | None:
    """Return the largest change of the strip lift between neighbouring strips on a mean-line lattice.

    Only surfaces whose lattice lies on a cambered section mean line count,
    and strips at a deflected control (and their neighbours) are left out,
    because a deflection changes the strip lift by design. Straight
    horseshoe legs on such a lattice left the curved mean surface; with
    narrow strips they passed close to the control points of other strips,
    the linear system was near-singular, and the strip lift oscillated from
    strip to strip. The legs now follow the mean surface
    (``VortexLattice.leg_next``); the value stays as a diagnostic of an
    ill-conditioned mean-line solve.

    Parameters
    ----------
    lattice : VortexLattice or None
        Vortex lattice geometry.
    spanwise_list : sequence of SpanwiseResult or None
        Spanwise results, one per surface of the lattice.

    Returns
    -------
    float or None
        Largest absolute difference of the strip lift coefficient between
        neighbouring strips of one surface [-], or None when no surface lies
        on a cambered mean line.
    """
    camber = getattr(lattice, "geometric_camber", None)
    if lattice is None or spanwise_list is None or camber is None or not np.any(camber):
        return None
    if not getattr(lattice, "has_camber_offset", True):
        return None
    control = np.zeros(lattice.n_strips, dtype=bool)
    eta = np.abs(np.asarray(lattice.eta, dtype=float))
    names = [s.name for s in lattice.surfaces]
    for info in getattr(lattice, "control_info", []) or []:
        if float(info.get("deflection_rad", 0.0)) == 0.0 or info.get("surface") not in names:
            continue
        sf = lattice.surfaces[names.index(info["surface"])]
        idx = np.arange(sf.strips.start, sf.strips.stop)
        lo = float(info.get("eta_start_eff", 0.0)) - 0.05
        hi = float(info.get("eta_end_eff", 1.0)) + 0.05
        control[idx[(eta[idx] >= lo) & (eta[idx] <= hi)]] = True
    jump = 0.0
    for sf, sw in zip(lattice.surfaces, spanwise_list):
        cl = np.asarray(getattr(sw, "Cl", []), dtype=float)
        idx = np.arange(sf.strips.start, sf.strips.stop)
        if cl.shape[0] != idx.shape[0] or idx.shape[0] < 2:
            continue
        use = camber[idx[:-1]] & camber[idx[1:]] & ~control[idx[:-1]] & ~control[idx[1:]]
        if np.any(use):
            jump = max(jump, float(np.max(np.abs(np.diff(cl))[use])))
    return jump


def _robust_value(stats: dict, array_key: str, max_key: str, percentile: float, default: float) -> tuple[float, bool]:
    """Return a percentile of a per-panel array of *stats*, or its maximum value.

    Parameters
    ----------
    stats : dict
        Mesh statistics (:func:`ventorum.panel3d.mesh.check_mesh`).
    array_key : str
        Key of the per-panel array (for example ``"aspect_ratio"``).
    max_key : str
        Key of the maximum value, used when the array is absent or empty.
    percentile : float
        Percentile of the array [%], from 0 to 100.
    default : float
        Value when the statistics have neither key.

    Returns
    -------
    tuple of (float, bool)
        The value and True when it is the percentile of the array (False
        when it is the maximum value or the default).
    """
    arr = stats.get(array_key)
    if arr is not None:
        values = np.asarray(arr, dtype=float).ravel()
        values = values[np.isfinite(values)]
        if values.size:
            return float(np.percentile(values, percentile)), True
    return float(stats.get(max_key, default)), False


#: Texts of the two levels of the 3D panel discretization-error indicators.
_RESOLUTION_TEXT = {
    1: "the mesh error is probably above 2 % in CL or 4 % in CDi.",
    2: "the mesh error is probably above 5 % in CL or 10 % in CDi.",
}


def _percentile_name(percentile: float, quantity: str) -> str:
    """Return "the median <quantity>" or "the <q>th percentile of the <quantity>"."""
    if float(percentile) == 50.0:
        return f"the median {quantity}"
    return f"the {percentile:g}th percentile of the {quantity}"


def panel3d_mesh_checks(
    mesh_stats: dict | None,
    CDi: float | None = None,
    CD_nearfield: float | None = None,
    n_chord: int | None = None,
) -> tuple[dict[str, float], list[str], list[str]]:
    """Return the mesh rules of the 3D panel method: penalties, warnings and recommendations.

    The thresholds are the ``PANEL3D_*`` constants of this module, calibrated
    by a mesh-error study of the method (see the notes of the constants). Two
    groups of rules give two factors:

    * ``mesh_adequacy`` (needs *mesh_stats*): the shape and the validity of
      the panels. A percentile (the median) of the panel aspect ratio and of
      the panel skew, not the largest value; the largest warp; a mesh that is not
      watertight, panels that are not convex (a folded strip), and a surface
      that intersects itself or its mirror image (``n_self_intersections``,
      same penalty as the folded panels). The penalties add. A steep root
      dihedral (``steep_root_dihedral``) gives a warning without a
      penalty: without the root thickness scaling
      (``scale_root_thickness`` False) the root pressure coefficient is
      singular; with it the lift needs a spanwise convergence study.
    * ``mesh_resolution``: the discretization-error indicators. The relative
      difference of the near-field pressure drag and the far-field Trefftz
      induced drag, and the chordwise panel count when it is given. Each
      indicator gives a level: none, warning (probable error above 2 % in CL
      or 4 % in CDi) or strong warning (above 5 % in CL or 10 % in CDi). The
      penalty is that of the worse level.

    Parameters
    ----------
    mesh_stats : dict or None
        Mesh statistics of :func:`ventorum.panel3d.mesh.check_mesh`. The
        per-panel arrays ``aspect_ratio`` [-] and ``skew`` [-] give the
        percentiles; without them the largest values are used.
    CDi : float or None, optional
        Far-field Trefftz-plane induced drag coefficient [-].
    CD_nearfield : float or None, optional
        Near-field pressure drag coefficient [-].
    n_chord : int or None, optional
        Chordwise panels per surface [-]. None skips the chordwise rule.

    Returns
    -------
    tuple of (dict, list of str, list of str)
        The penalties by factor name, the warnings and the recommendations.
    """
    penalties: dict[str, float] = {}
    warnings: list[str] = []
    recommendations: list[str] = []

    if mesh_stats is not None:
        mesh_pen = 0.0
        q_ar = PANEL3D_ASPECT_RATIO_PERCENTILE
        ar, ar_is_pct = _robust_value(mesh_stats, "aspect_ratio", "max_aspect_ratio", q_ar, 1.0)
        if ar > PANEL3D_MAX_ASPECT_RATIO:
            mesh_pen += PANEL3D_PENALTY_ASPECT_RATIO
            what = _percentile_name(q_ar, "panel aspect ratio") if ar_is_pct else "the panel aspect ratio"
            warnings.append(
                f"Elongated panels: {what} is {ar:.1f} (above {PANEL3D_MAX_ASPECT_RATIO:g}). The spanwise "
                "panel count is probably low for the chordwise count; the mesh error is probably above 2 % "
                "in CL or 4 % in CDi."
            )
            recommendations.append("Increase the spanwise panel count (n_panels).")
        q_sk = PANEL3D_SKEW_PERCENTILE
        sk, sk_is_pct = _robust_value(mesh_stats, "skew", "max_skew", q_sk, 0.0)
        if sk > PANEL3D_MAX_SKEW:
            mesh_pen += PANEL3D_PENALTY_SKEW
            what = _percentile_name(q_sk, "panel skew") if sk_is_pct else "the panel skew"
            warnings.append(f"Skewed mesh: {what} is {sk:.2f} (above {PANEL3D_MAX_SKEW:g}).")
        max_wp = float(mesh_stats.get("max_warp", 0.0))
        if max_wp > PANEL3D_MAX_WARP:
            mesh_pen += PANEL3D_PENALTY_WARP
            warnings.append(
                f"Warped panels: maximum panel warp reaches {max_wp:.3f} (above {PANEL3D_MAX_WARP:g})."
            )
        if not bool(mesh_stats.get("watertight", True)):
            mesh_pen += PANEL3D_PENALTY_WATERTIGHT
            warnings.append("Panel mesh is not watertight.")
        n_nonconvex = int(mesh_stats.get("n_nonconvex", 0) or 0)
        if n_nonconvex > 0:
            mesh_pen += PANEL3D_PENALTY_NONCONVEX
            warnings.append(
                f"Folded panels: {n_nonconvex} panels are not convex. A folded strip makes the "
                "result not valid."
            )
            recommendations.append(
                "Change the spanwise panel count or spacing so that no strip folds (for example a "
                "root strip narrower than the section thickness on a wing with dihedral)."
            )
        n_self_x = int(mesh_stats.get("n_self_intersections", 0) or 0)
        if n_self_x > 0:
            mesh_pen += PANEL3D_PENALTY_NONCONVEX
            warnings.append(
                f"Self-intersecting surface: {n_self_x} panel edges cross panels of the same closed "
                "surface or of its mirror image. A surface that passes through itself makes the result "
                "not valid."
            )
            recommendations.append(
                "Use the root thickness scaling (panel_scale_root_thickness=True, the default), or change "
                "the spanwise panel count or the dihedral, so that the surface does not pass through "
                "itself or its mirror image."
            )
        if bool(mesh_stats.get("root_scale_warning", False)):
            warnings.append(PANEL3D_ROOT_SCALE_WARNING)
        if bool(mesh_stats.get("steep_root_dihedral", False)):
            if bool(mesh_stats.get("scale_root_thickness", True)):
                warnings.append(PANEL3D_STEEP_ROOT_CONVERGENCE_NOTE)
                recommendations.append(
                    "Solve with at least two spanwise panel counts and compare the lift."
                )
            else:
                warnings.append(PANEL3D_STEEP_ROOT_UNSCALED_WARNING)
                recommendations.append(
                    "Use the root thickness scaling (panel_scale_root_thickness=True, the default)."
                )
        penalties["mesh_adequacy"] = mesh_pen

    levels: list[int] = []
    if CD_nearfield is not None and CDi is not None and _is_finite(CD_nearfield) and _is_finite(CDi):
        drag_diff = abs(float(CD_nearfield) - float(CDi))
        drag_ref = max(abs(float(CDi)), 1e-4)
        rel = drag_diff / drag_ref
        level = 0
        if drag_diff > PANEL3D_DRAG_ABSOLUTE_FLOOR:
            if rel > PANEL3D_DRAG_STRONG_REL:
                level = 2
            elif rel > PANEL3D_DRAG_WARN_REL:
                level = 1
        if level:
            limit = PANEL3D_DRAG_STRONG_REL if level == 2 else PANEL3D_DRAG_WARN_REL
            warnings.append(
                f"Near-field pressure drag ({float(CD_nearfield):.4f}) and far-field Trefftz induced drag "
                f"({float(CDi):.4f}) differ by {drag_diff:.4f} (relative difference {rel:.1%}, above "
                f"{limit:.0%}): {_RESOLUTION_TEXT[level]}"
            )
        levels.append(level)
    if n_chord is not None and int(n_chord) > 0:
        n_c = int(n_chord)
        level = 2 if n_c < PANEL3D_N_CHORD_STRONG else (1 if n_c < PANEL3D_N_CHORD_WARN else 0)
        if level:
            limit = PANEL3D_N_CHORD_STRONG if level == 2 else PANEL3D_N_CHORD_WARN
            warnings.append(
                f"Coarse chordwise mesh: {n_c} chordwise panels (below {limit}): {_RESOLUTION_TEXT[level]}"
            )
        levels.append(level)
    if levels:
        worst = max(levels)
        penalties["mesh_resolution"] = (0.0, PANEL3D_PENALTY_RESOLUTION_WARN, PANEL3D_PENALTY_RESOLUTION_STRONG)[worst]
        if worst:
            recommendations.append(
                "Increase n_chord: the chordwise panel count controls most of the 3D panel mesh error. "
                "Set panel_estimate_mesh_error=True for an error estimate of this mesh."
            )
    return penalties, warnings, recommendations



def isa_speed_of_sound(rho: float) -> float:
    """Return the ISA speed of sound [m/s] at the air density *rho* [kg/m^3].

    The ISA temperature follows from the density: in the troposphere
    ``T = T0 (rho/rho0)^(1/4.25588)``, and 216.65 K above 11 000 m. At the
    sea-level density the value is the sea-level speed of sound. The result
    is exact for the ISA atmosphere and approximate for a non-standard day.
    """
    rho = float(rho)
    if not np.isfinite(rho) or rho <= 0.0 or rho == RHO_SL:
        return A_SL
    if rho <= _ISA_RHO_TROPOPAUSE:
        T = _ISA_T_TROPOPAUSE
    else:
        T = _ISA_T0 * (rho / RHO_SL) ** (1.0 / _ISA_DENSITY_EXPONENT)
    return float(np.sqrt(_GAMMA_R_AIR * T))


def polar_limits(airfoils: Sequence) -> dict[str, np.ndarray] | None:
    """Return the limits of the tabulated section data of each strip.

    Parameters
    ----------
    airfoils : sequence
        The airfoil of each strip (``VortexLattice.airfoils``).

    Returns
    -------
    dict of numpy.ndarray or None
        Arrays with one value per strip: ``alpha_lo`` and ``alpha_hi``
        [rad] (the alpha range of the table), ``cl_max`` and
        ``alpha_cl_max`` [rad] (the largest Cl of the table and its angle),
        ``cl_min`` and ``alpha_cl_min`` [rad]. A maximum (or minimum) at an
        end of the table is not a stall: its values are NaN. A strip with a
        linear airfoil has NaN everywhere. None if no strip is tabulated.
    """
    n = len(airfoils)
    keys = ("alpha_lo", "alpha_hi", "cl_max", "alpha_cl_max", "cl_min", "alpha_cl_min")
    out = {k: np.full(n, np.nan) for k in keys}
    seen: dict[int, tuple] = {}
    any_tab = False
    for i, af in enumerate(airfoils):
        if not isinstance(af, TabulatedAirfoil):
            continue
        any_tab = True
        row = seen.get(id(af))
        if row is None:
            a, cl = af._prepare_arrays()[:2]
            k_max, k_min = int(np.argmax(cl)), int(np.argmin(cl))
            in_max = 0 < k_max < a.size - 1
            in_min = 0 < k_min < a.size - 1
            row = (float(a[0]), float(a[-1]),
                   float(cl[k_max]) if in_max else np.nan, float(a[k_max]) if in_max else np.nan,
                   float(cl[k_min]) if in_min else np.nan, float(a[k_min]) if in_min else np.nan)
            seen[id(af)] = row
        for key, v in zip(keys, row):
            out[key][i] = v
    return out if any_tab else None


def polar_stats(limits: dict[str, np.ndarray] | None, cl: np.ndarray, alpha_eff: np.ndarray) -> dict | None:
    """Compare the section results with the limits of their polars.

    Parameters
    ----------
    limits : dict or None
        The result of :func:`polar_limits`.
    cl : numpy.ndarray
        Section lift coefficient of each strip.
    alpha_eff : numpy.ndarray
        Effective angle of attack of each strip [rad].

    Returns
    -------
    dict or None
        ``n_strips``, ``n_tabulated``, ``n_out_of_table``,
        ``max_excess_deg`` (the largest angle outside a table [deg]),
        ``n_past_stall``, ``stall_ratio`` (the largest Cl/Cl_max of the
        strips whose polar has a stall, positive and negative side),
        ``peak_cl`` and ``cl_max_at_peak`` (the strip of that ratio), and
        ``all_with_stall`` (every strip has a polar with a stall). None if
        *limits* is None or the arrays are not finite.
    """
    if limits is None:
        return None
    cl = np.asarray(cl, dtype=float)
    ae = np.asarray(alpha_eff, dtype=float)
    if cl.shape != limits["alpha_lo"].shape or ae.shape != cl.shape or cl.size == 0:
        return None
    if not (np.isfinite(cl).all() and np.isfinite(ae).all()):
        return None
    tab = np.isfinite(limits["alpha_lo"])
    excess = np.where(tab, np.maximum(limits["alpha_lo"] - ae, ae - limits["alpha_hi"]), -np.inf)
    out_mask = excess > 1e-9
    has_max = np.isfinite(limits["cl_max"]) & (limits["cl_max"] > 0.0)
    has_min = np.isfinite(limits["cl_min"]) & (limits["cl_min"] < 0.0)
    cl_max = np.where(has_max, limits["cl_max"], 1.0)
    cl_min = np.where(has_min, limits["cl_min"], -1.0)
    r_pos = np.where(has_max & (cl > 0.0), cl / cl_max, 0.0)
    r_neg = np.where(has_min & (cl < 0.0), cl / cl_min, 0.0)
    past = ((has_max & (ae > np.where(has_max, limits["alpha_cl_max"], np.inf)))
            | (has_min & (ae < np.where(has_min, limits["alpha_cl_min"], -np.inf)))
            | (r_pos > 1.0) | (r_neg > 1.0))
    ratio = np.maximum(r_pos, r_neg)
    k = int(np.argmax(ratio))
    limit_at_k = limits["cl_max"][k] if r_pos[k] >= r_neg[k] else limits["cl_min"][k]
    return {
        "n_strips": int(cl.size),
        "n_tabulated": int(tab.sum()),
        "n_out_of_table": int(out_mask.sum()),
        "max_excess_deg": float(np.degrees(excess[out_mask].max())) if out_mask.any() else 0.0,
        "n_past_stall": int(past.sum()),
        "stall_ratio": float(ratio[k]),
        "peak_cl": float(cl[k]),
        "cl_max_at_peak": float(limit_at_k),
        "all_with_stall": bool(tab.all() and has_max.all()),
    }


def _is_finite(value: object) -> bool:
    """Return True when *value* is a finite number."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _count_extrema(g: np.ndarray, rel_tol: float = 1e-6) -> int:
    """Return the number of local extrema of a sequence (tiny steps are ignored)."""
    g = np.asarray(g, dtype=float)
    if g.size < 4:
        return 0
    d = g[1:] - g[:-1]
    scale = max(float(np.abs(g).max()), 1e-300)
    d = d[np.abs(d) > rel_tol * scale]
    if d.size < 2:
        return 0
    s = np.sign(d)
    return int(np.count_nonzero(s[1:] != s[:-1]))


def _count_extrema_rows(G: np.ndarray, rel_tol: float = 1e-6) -> np.ndarray:
    """Return :func:`_count_extrema` of each row of *G* (shape (K, m)), with the same rules."""
    G = np.asarray(G, dtype=float)
    K, m = G.shape
    if m < 4:
        return np.zeros(K, dtype=int)
    d = G[:, 1:] - G[:, :-1]
    scale = np.maximum(np.abs(G).max(axis=1), 1e-300)
    keep = np.abs(d) > rel_tol * scale[:, None]
    s = np.sign(d)
    # The sign of the last kept step before each step (forward fill).
    idx = np.where(keep, np.arange(m - 1)[None, :], -1)
    last = np.maximum.accumulate(idx, axis=1)
    prev = np.concatenate([np.full((K, 1), -1), last[:, :-1]], axis=1)
    prev_sign = np.take_along_axis(s, np.maximum(prev, 0), axis=1)
    change = keep & (prev >= 0) & (s != prev_sign)
    return change.sum(axis=1).astype(int)


def spanwise_stats_batch(surface_strips: Sequence, arrays: dict[str, np.ndarray]) -> list[dict | None]:
    """Return the spanwise statistics of the trust score for K cases at once.

    Parameters
    ----------
    surface_strips : sequence of slices or index arrays
        The strips of each surface (the split of the spanwise results).
    arrays : dict of (K, n_strips) arrays
        ``gamma``, ``Cl``, ``Cd_i``, ``alpha_eff``, ``alpha_i``,
        ``local_lift`` and, if present, ``Cd_profile`` and ``Cm_section``
        (a value of None means "absent" for all cases).

    Returns
    -------
    list of dict or None
        For each case the values that :func:`evaluate_aerodynamic_trust`
        computes from the spanwise results (``max_local_cl``,
        ``max_alpha_eff_deg``, ``n_extrema``), or None when an array of the
        case is not finite (the caller then passes the spanwise results, so
        that the warning names the array).
    """
    names = [n for n in ("gamma", "Cl", "Cd_i", "alpha_eff", "alpha_i", "local_lift", "Cd_profile", "Cm_section")
             if arrays.get(n) is not None]
    K = arrays["gamma"].shape[0]
    finite = np.ones(K, dtype=bool)
    for n in names:
        finite &= np.isfinite(arrays[n]).all(axis=1)
    cl = np.abs(arrays["Cl"])
    ae = np.abs(arrays["alpha_eff"])
    max_cl = np.zeros(K)
    max_ae = np.zeros(K)
    n_ext = np.zeros(K, dtype=int)
    for sl in surface_strips:
        g = arrays["gamma"][:, sl]
        if g.shape[1] == 0:
            continue
        max_cl = np.maximum(max_cl, cl[:, sl].max(axis=1))
        max_ae = np.maximum(max_ae, np.degrees(ae[:, sl].max(axis=1)))
        n_ext = np.maximum(n_ext, _count_extrema_rows(g))
    return [{"max_local_cl": float(max_cl[k]), "max_alpha_eff_deg": float(max_ae[k]), "n_extrema": int(n_ext[k])}
            if finite[k] else None for k in range(K)]


def evaluate_aerodynamic_trust(
    AR: float,
    surfaces: Sequence[DiscretizedSurface | LiftingSurface] | None = None,
    condition: FlightCondition | None = None,
    spanwise_list: Sequence[SpanwiseResult] | None = None,
    CL: float = 0.0,
    CDi: float = 0.0,
    CD_total: float | None = None,
    converged: bool = True,
    n_panels: int = 80,
    max_sweep_rad: float | None = None,
    *,
    solver_type: str | None = None,
    h_over_c: float | None = None,
    n_chord: int | None = None,
    notes: Sequence[str] | None = None,
    max_chord_over_c: float = 1.0,
    spanwise_stats: dict | None = None,
    polar: dict | None = None,
    CY: float | None = None,
    CY_farfield: float | None = None,
    Cn_stability: float | None = None,
    Cn_trefftz: float | None = None,
    mesh_stats: dict | None = None,
    CD_nearfield: float | None = None,
    post_stall_branch: bool = False,
    te_clearance: float | None = None,
    mean_line_jump: float | None = None,
) -> TrustScore:
    """Evaluate the validity of a solution.

    Parameters
    ----------
    AR : float
        Reference aspect ratio ``b^2 / S``.
    surfaces : optional
        Used only to find the sweep when *max_sweep_rad* is not given.
    condition : FlightCondition, optional
        Speed (for the Mach number check).
    spanwise_list : optional
        Spanwise results (section lift and circulation checks).
    CL, CDi, CD_total : float
        Integrated coefficients.
    converged : bool
        Whether the solver met its tolerance.
    n_panels : int
        Spanwise panels per semi-span of the main surface.
    max_sweep_rad : float, optional
        Representative sweep of the lifting line [rad].
    solver_type : str, optional
        ``"vlm"``, ``"linear"``, ``"nonlinear"`` or ``"fourier"``. If not
        given, the lifting-line rules are used.
    h_over_c : float, optional
        Smallest ground clearance divided by the reference chord.
    n_chord : int, optional
        Chordwise panels of the vortex lattice, or chordwise panels per
        surface of the 3D panel method (then also the chordwise rule of
        :func:`panel3d_mesh_checks`).
    notes : list of str, optional
        Extra warnings from the solver.
    max_chord_over_c : float
        Largest strip chord divided by the reference chord (for the
        chordwise panel check in ground effect).
    spanwise_stats : dict or None
        The statistics of the spanwise results from
        :func:`spanwise_stats_batch` (all arrays finite). A batch computes
        them for all its cases at once; the result is the same as from
        *spanwise_list*.
    polar : dict or None
        The comparison of the section results with their tabulated polars
        (:func:`polar_stats`), or None if no strip has a tabulated polar.
        With it, a section angle outside a table gives the penalty
        ``polar_range``, and the stall rule also uses the maximum Cl and the
        stall angle of each polar. When every strip has a polar with a
        stall, the fixed limits of linear sections (Cl 1.55 and 16 deg) are
        not used.
    CY, CY_farfield : float or None
        Near-field side-force coefficient and the far-field value of the
        Trefftz plane, both in wind axes [-].
    Cn_stability, Cn_trefftz : float or None
        Near-field yawing-moment coefficient in stability axes and the
        yawing moment of the Trefftz-plane induced drag (stability axes)
        [-]. When the near field and the far field do not agree
        (:func:`near_far_lateral_warnings`), the result gets a warning. The
        comparison is a diagnostic: it does not change the score.
    mesh_stats : dict or None, optional
        3D panel method only: the mesh statistics of
        :func:`ventorum.panel3d.mesh.check_mesh` (see
        :func:`panel3d_mesh_checks`).
    CD_nearfield : float or None, optional
        3D panel method only: the near-field pressure drag coefficient [-],
        compared with the far-field *CDi* (see :func:`panel3d_mesh_checks`).
    post_stall_branch : bool, optional
        True if multiple post-stall equilibrium solution branches exist.
    te_clearance : float or None, optional
        Smallest trailing-edge height above the ground divided by the local
        chord [-]. None in free air or when not evaluated.
    mean_line_jump : float or None, optional
        Largest change of the strip lift coefficient between neighbouring
        strips of a mean-line lattice [-] (see :func:`mean_line_strip_jump`).
        None when no surface lies on a cambered mean line.

    A result with a non-finite coefficient never keeps a trust level above
    LOW: it gets the penalty ``non_finite_result`` and the warning
    "non-finite result". This is a second safety line; the input checks
    must stop the known cases first.
    """
    # Non-finite result guard: any total coefficient that is not a finite
    # number caps the trust level at LOW.
    bad_totals: list[str] = []
    for label, value in (("CL", CL), ("CDi", CDi), ("CD_total", CD_total)):
        if value is not None and not _is_finite(value):
            bad_totals.append(f"{label}={value}")
    for sw in (spanwise_list or ()) if spanwise_stats is None else ():
        arrays = [
            ("gamma", sw.gamma), ("Cl", sw.Cl), ("Cd_i", sw.Cd_i),
            ("alpha_eff", sw.alpha_eff), ("alpha_i", sw.alpha_i),
            ("local_lift", sw.local_lift),
        ]
        if sw.Cd_profile is not None:
            arrays.append(("Cd_profile", sw.Cd_profile))
        if sw.Cm_section is not None:
            arrays.append(("Cm_section", sw.Cm_section))
        # One test for all arrays first (the usual case: all finite); the
        # array-by-array test below only finds the label of the first bad one.
        try:
            all_finite = bool(np.isfinite(np.concatenate(
                [np.asarray(arr, dtype=float).ravel() for _, arr in arrays])).all())
        except ValueError:
            all_finite = False
        if all_finite:
            continue
        for label, arr in arrays:
            values = np.asarray(arr, dtype=float)
            if values.size and not bool(np.all(np.isfinite(values))):
                bad_totals.append(f"{sw.surface_name}.{label}")
                break
    non_finite_result = bool(bad_totals)
    penalties: dict[str, float] = {}
    warnings: list[str] = []
    recommendations: list[str] = []
    if non_finite_result:
        penalties["non_finite_result"] = 0.6
        warnings.append(
            "Non-finite result (non-finite result coefficients: "
            + ", ".join(bad_totals)
            + "): the coefficients are not valid numbers; check the inputs."
        )
    solver = (solver_type or "linear").lower()
    is_llt = solver in LIFTING_LINE_SOLVERS

    # 1. Aspect ratio -----------------------------------------------------------
    ar_pen = 0.0
    if AR <= 0:
        ar_pen = 1.0
        warnings.append("Aspect ratio is non-positive or undefined.")
    elif is_llt and AR < 3.0:
        ar_pen = 0.45 + 0.35 * min(1.0, (3.0 - AR) / 2.0)
        warnings.append(
            f"Very low aspect ratio (AR={AR:.2f} < 3) for a lifting-line method: "
            "chordwise effects are large."
        )
        recommendations.append("Use the VLM solver (solver='vlm') for AR < 3.")
    elif is_llt and AR < 5.0:
        ar_pen = 0.20 * ((5.0 - AR) / 2.0)
        warnings.append(f"Moderate aspect ratio (AR={AR:.2f} < 5) for a lifting-line method.")
        recommendations.append("Compare with the VLM solver for AR < 5.")
    penalties["aspect_ratio"] = ar_pen

    # 2. Sweep ------------------------------------------------------------------
    sweep_pen = 0.0
    max_sweep = 0.0
    if max_sweep_rad is not None:
        max_sweep = float(abs(max_sweep_rad))
    elif surfaces:
        for s in surfaces:
            if hasattr(s, "sweep_le"):
                max_sweep = max(max_sweep, abs(float(getattr(s, "sweep_le", 0.0))))
    sweep_deg = float(np.degrees(max_sweep))
    if is_llt and sweep_deg > 2.5:
        sweep_pen = min(0.8, 0.20 + 0.60 * (sweep_deg - 2.5) / 27.5)
        warnings.append(
            f"Lifting line with sweep angle {sweep_deg:.1f} deg: the numerical lifting line is not "
            "grid convergent with sweep (lift falls as panels are added)."
        )
        recommendations.append("Use the VLM solver (solver='vlm') for swept wings.")
    elif not is_llt and sweep_deg > 55.0 and AR < 2.5:
        sweep_pen = 0.35
        warnings.append(
            f"Slender wing (sweep {sweep_deg:.0f} deg, AR={AR:.2f}): leading-edge vortex lift is not modelled."
        )
    penalties["sweep"] = sweep_pen

    # 3. Section lift and stall ---------------------------------------------------
    stall_pen = 0.0
    max_local_cl = 0.0
    max_alpha_eff_deg = 0.0
    if spanwise_stats is not None:
        max_local_cl = spanwise_stats["max_local_cl"]
        max_alpha_eff_deg = spanwise_stats["max_alpha_eff_deg"]
    elif spanwise_list:
        for sw in spanwise_list:
            if len(sw.Cl) > 0:
                max_local_cl = max(max_local_cl, float(np.abs(sw.Cl).max()))
            if len(sw.alpha_eff) > 0:
                # degrees(max |a|) equals max |degrees(a)|: the scaling is monotonic.
                max_alpha_eff_deg = max(max_alpha_eff_deg, float(np.degrees(np.abs(sw.alpha_eff).max())))
    if max_local_cl == 0.0 and abs(CL) > 0:
        max_local_cl = abs(CL) * 1.25
    linear_model = solver != "nonlinear"
    use_fixed = not (polar is not None and polar["all_with_stall"])
    if use_fixed and (max_local_cl > 1.55 or max_alpha_eff_deg > 16.0):
        stall_pen = 0.50 + 0.35 * min(1.0, max(0.0, (max_local_cl - 1.55) / 0.5))
        if linear_model:
            warnings.append(
                f"Severe stall risk: peak section Cl={max_local_cl:.2f} (alpha_eff={max_alpha_eff_deg:.1f} deg). "
                "The linear section model has no stall and over-predicts lift here."
            )
            recommendations.append("Use solver='nonlinear' with section polars (e.g. from XFOIL).")
        else:
            warnings.append(
                f"Post-stall sections: peak section Cl={max_local_cl:.2f} (alpha_eff={max_alpha_eff_deg:.1f} deg). "
                "The lifting-line solution past maximum lift is not unique and not validated."
            )
    elif use_fixed and (max_local_cl > 1.20 or max_alpha_eff_deg > 12.0):
        stall_pen = 0.25 * max(0.0, (max_local_cl - 1.20) / 0.35)
        warnings.append(
            f"Approaching section stall: peak section Cl={max_local_cl:.2f} (alpha_eff={max_alpha_eff_deg:.1f} deg)."
        )
    if polar is not None:
        # Stall of the tabulated polars: their own maximum Cl and stall angle.
        ratio = polar["stall_ratio"]
        if polar["n_past_stall"] > 0:
            stall_pen = max(stall_pen, 0.50 + 0.35 * min(1.0, max(0.0, (ratio - 1.0) / 0.3)))
            where = (f"{polar['n_past_stall']} of {polar['n_strips']} strips are past the stall of their "
                     f"polar (peak section Cl={polar['peak_cl']:.2f}, polar Cl_max={polar['cl_max_at_peak']:.2f})")
            if linear_model:
                warnings.append(
                    f"Severe stall risk: {where}. This solver uses the linear part of the polars and "
                    "over-predicts lift here."
                )
                recommendations.append("Use solver='nonlinear' with the same section polars.")
            else:
                warnings.append(
                    f"Post-stall sections: {where}. The lifting-line solution past maximum lift is not "
                    "unique and not validated."
                )
        elif ratio > 0.9:
            stall_pen = max(stall_pen, 0.25 * (ratio - 0.9) / 0.1)
            warnings.append(
                f"Approaching section stall: peak section Cl={polar['peak_cl']:.2f} is "
                f"{100.0 * ratio:.0f} % of the polar Cl_max ({polar['cl_max_at_peak']:.2f})."
            )
    penalties["stall_proximity"] = stall_pen

    # 3b. Section angles outside the polar tables --------------------------------------
    range_pen = 0.0
    if polar is not None and polar["n_out_of_table"] > 0:
        range_pen = 0.30
        warnings.append(
            f"Outside the polar data: {polar['n_out_of_table']} of {polar['n_strips']} strips have an "
            f"effective angle up to {polar['max_excess_deg']:.1f} deg outside their polar table; the end "
            "values of the table (Cl, Cd, Cm) are used there."
        )
        recommendations.append("Extend the section polars over the range of alpha_eff.")
    penalties["polar_range"] = range_pen

    # 3c. Side-edge vortex lift of low-aspect-ratio wings (vortex lattice) ------------
    side_pen = 0.0
    if (not is_llt and condition is not None and 0.0 < AR < LOW_AR_SIDE_EDGE_AR
            and sweep_deg <= 55.0 and abs(np.degrees(condition.alpha)) > LOW_AR_SIDE_EDGE_ALPHA_DEG):
        side_pen = 0.25
        warnings.append(
            f"Low aspect ratio (AR={AR:.2f} < {LOW_AR_SIDE_EDGE_AR:g}) at alpha="
            f"{np.degrees(condition.alpha):.1f} deg: the vortex lift of the side edges is not modelled, "
            "so CL is under-predicted (Polhamus 1966; Lamar 1974)."
        )
    penalties["side_edge_vortex"] = side_pen

    # 4. Compressibility (outside the valid envelope) ---------------------------------
    mach_pen = 0.0
    if condition is not None:
        correction_applied = getattr(condition, "mach", None) is not None and condition.mach > 0.0
        if correction_applied:
            mach = float(condition.mach)
        else:
            a_sound = isa_speed_of_sound(getattr(condition, "rho", RHO_SL))
            mach = float(condition.V_inf) / a_sound

        if mach > MACH_LIMIT_INCOMPRESSIBLE:
            mach_pen = 0.60 + 0.30 * min(1.0, (mach - MACH_LIMIT_INCOMPRESSIBLE) / 0.35)
            if correction_applied:
                warnings.append(
                    f"Out of the valid envelope: Mach={mach:.2f} > {MACH_LIMIT_INCOMPRESSIBLE}. "
                    "The Prandtl-Glauert correction is applied and is not validated above Mach 0.3."
                )
            else:
                warnings.append(
                    f"Out of the valid envelope: Mach={mach:.2f} > {MACH_LIMIT_INCOMPRESSIBLE} (ISA speed of "
                    f"sound {a_sound:.1f} m/s at the air density). The methods are incompressible and no "
                    "compressibility correction is applied."
                )
                recommendations.append("Reduce the speed, or use a compressible method.")
        if correction_applied and mach > 0.7:
            warnings.append(
                f"Mach={mach:.2f} > 0.7: linear theory loses validity near the critical Mach number."
            )
    penalties["compressibility"] = mach_pen

    # 5. Ground proximity (stated in h/c) ------------------------------------------------
    ground_pen = 0.0
    if h_over_c is not None:
        hc = float(h_over_c)
        if hc < 0.1:
            ground_pen = 0.45
            warnings.append(
                f"Extreme ground proximity (h_min/c={hc:.3f} < 0.1): thin-wing theory without thickness "
                "or viscous effects; real thick sections can lose lift here."
            )
        elif hc < 0.3:
            ground_pen = 0.20
            warnings.append(
                f"Strong ground effect (h_min/c={hc:.3f} < 0.3): thickness and viscous effects are not modelled."
            )
        if not is_llt and n_chord is not None and n_chord > 0 and (max_chord_over_c / n_chord) > hc:
            ground_pen += 0.20
            warnings.append(
                f"Chordwise panels ({n_chord} on the largest chord, {max_chord_over_c:.2f} c_ref) are longer "
                f"than the ground gap (h_min/c={hc:.3f})."
            )
            recommendations.append("Increase n_chord so that c/n_chord < h_min.")
    penalties["ground_proximity"] = ground_pen

    # 5b. Trailing-edge ground clearance (Trefftz CDi far-wake height dependence)
    if te_clearance is not None and _is_finite(te_clearance) and float(te_clearance) < _CDI_TE_CLEARANCE:
        warnings.append(
            f"Small trailing-edge clearance (h_te/c={float(te_clearance):.3f} < {_CDI_TE_CLEARANCE:g}): "
            "Trefftz CDi (and drag-derived outputs) depends on the height of the far wake."
        )

    # 5c. Mean-line lattice conditioning (oscillating strip loads on the mean surface)
    if mean_line_jump is not None and _is_finite(mean_line_jump) and float(mean_line_jump) > _MEAN_LINE_STRIP_JUMP:
        warnings.append(
            f"Cambered mean-line lattice: the strip lift changes by up to {float(mean_line_jump):.3g} between "
            "neighbouring strips. The solve on the curved mean surface can be ill-conditioned, so the loads "
            "(also the totals) can be wrong. Use fewer, wider spanwise panels or a larger n_chord, and compare."
        )

    # 6. Numerical health -------------------------------------------------------------------
    num_pen = 0.0
    if not converged:
        num_pen += 0.55
        warnings.append("Solver did not converge to the specified tolerance.")
        recommendations.append("Increase max_iterations, or check for post-stall conditions.")
    if n_panels < 8:
        num_pen += 0.15 * ((8 - n_panels) / 4.0)
        warnings.append(f"Coarse spanwise mesh (n_panels={n_panels} < 8).")
    if spanwise_list or spanwise_stats is not None:
        if spanwise_stats is not None:
            n_extrema = spanwise_stats["n_extrema"]
        else:
            n_extrema = max((_count_extrema(sw.gamma) for sw in spanwise_list), default=0)
        if n_extrema > 6:
            num_pen += 0.40
            warnings.append(
                f"Spanwise circulation oscillates ({n_extrema} local extrema): probable numerical problem."
            )
    penalties["numerical"] = num_pen

    # 7. Near-field and far-field lateral values (a warning only, no penalty) -------------------
    warnings.extend(near_far_lateral_warnings(CY, CY_farfield, Cn_stability, Cn_trefftz))

    # 8. 3D panel method: mesh adequacy and discretization-error indicators -------------------
    if solver in ("panel3d", "3dp"):
        p3_pen, p3_warn, p3_rec = panel3d_mesh_checks(mesh_stats, CDi, CD_nearfield, n_chord)
        penalties.update(p3_pen)
        warnings.extend(p3_warn)
        recommendations.extend(r for r in p3_rec if r not in recommendations)

    # 8b. Viscous coupling post-stall non-unique branches --------------------------------------
    branch_pen = 0.0
    if post_stall_branch:
        branch_pen = POST_STALL_BRANCH_PENALTY
        warnings.append(
            "Non-unique post-stall solution branches: section slope dCl/dalpha is at or below "
            "zero and multiple solutions exist."
        )
        recommendations.append("Treat post-stall aerodynamic forces with caution; compare both solution branches.")
    penalties["post_stall_branch"] = branch_pen

    for msg in notes or ():
        if msg not in warnings:
            warnings.append(msg)

    # Composite score ------------------------------------------------------------------------
    total_penalty = min(1.0, sum(penalties.values()))
    score = max(0.0, 1.0 - total_penalty)
    if score >= 0.85:
        rating = "HIGH"
    elif score >= 0.65:
        rating = "MODERATE"
    elif score >= 0.40:
        rating = "LOW"
    else:
        rating = "UNRELIABLE"
    if non_finite_result and rating in ("HIGH", "MODERATE"):
        rating = "LOW"  # a cap: an UNRELIABLE rating stays UNRELIABLE

    # Heuristic bands (not calibrated against data).
    fidelity_factor = 1.0 - score
    unc_cl = max(0.015, abs(CL) * (0.03 + 0.20 * fidelity_factor) + 0.01 * fidelity_factor)
    unc_cdi = max(0.0004, CDi * (0.05 + 0.35 * fidelity_factor) + 0.0003 * fidelity_factor)
    cd_tot = CD_total if CD_total is not None and CD_total > 0 else (CDi if CDi > 1e-6 else 0.02)
    ld_ratio = CL / cd_tot if cd_tot > 0 else 0.0
    unc_ld = max(0.2, abs(ld_ratio) * (0.05 + 0.30 * fidelity_factor))

    return TrustScore(
        score=score,
        rating=rating,
        uncertainty_CL=unc_cl,
        uncertainty_CDi=unc_cdi,
        uncertainty_LD=unc_ld,
        factors=penalties,
        warnings=warnings,
        recommendations=recommendations,
    )
