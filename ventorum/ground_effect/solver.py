# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Single-point ground-effect analysis.

Model
-----
* The ground is a flat plane parallel to the free stream (level flight). The
  aircraft keeps its body axes; the plane is tilted in body axes by the
  angle of attack, turned about the free stream by the bank angle, and placed
  at the requested height (see :func:`ventorum.aero.system.make_ground_plane`).
* The image method makes the velocity normal to the ground zero. The wake
  leaves the trailing edge parallel to the ground.
* Lifting surfaces are thin (no thickness) and inviscid. At small gaps a real
  thick section can lose lift (suction under the wing); this model cannot
  show that. The trust score warns below h_min/c = 0.3.

Solver limits
-------------
* Vortex-lattice method (default): the number of chordwise panels is chosen
  so that a panel is not longer than the smallest gap (``n_chord=None``).
  :class:`~ventorum.ground_effect.GroundEffectSweep` uses the count of
  the lowest height for all cases.
* Lifting-line solvers (``"linear"``, ``"nonlinear"``): one horseshoe per
  strip cannot follow the chordwise variation of the image flow. Against the
  vortex lattice they under-predict the lift increment: by 3 to 5 % of CL at
  h_min/c = 1 and by 13 to 18 % at h_min/c = 0.5 (rectangular wings, AR 4 to
  8, alpha 2 to 6 deg). They are refused below h_min/c = 1.
* The Fourier solver has no ground effect.
"""

from __future__ import annotations

import time
from typing import Any, Literal
from collections.abc import Sequence

import numpy as np

from ventorum.aero.system import (
    GroundPlane,
    freestream_direction,
    make_ground_plane,
)
from ventorum.core.datatypes import (
    Aircraft,
    DiscretizedSurface,
    FlightCondition,
    LiftingSurface,
    SolverSettings,
)
from ventorum.core.constants import RHO_SL
from ventorum.geometry.lattice import VortexLattice, build_lattice, lattice_to_discretized
from ventorum.ground_effect.state import GroundEffectCondition, GroundEffectResult
from ventorum.solvers.factory import make_solver, resolve_solver_type
from ventorum.core.errors import GroundStrikeError, ValidityError
from ventorum.utils.validation import validate_aircraft

from ventorum.solvers.lattice_base import LLT_GE_MIN_H_OVER_C  # noqa: E402


def _as_aircraft(geometry: Aircraft | LiftingSurface) -> Aircraft:
    if isinstance(geometry, Aircraft):
        return geometry
    if isinstance(geometry, LiftingSurface):
        return Aircraft(name="GroundEffectWing", surfaces=[geometry])
    if isinstance(geometry, (list, tuple)) and geometry and isinstance(geometry[0], DiscretizedSurface):
        raise TypeError(
            "analyze_ground_effect needs the Aircraft or LiftingSurface definition (section data and "
            "chordwise geometry); a list of DiscretizedSurface is no longer accepted."
        )
    raise TypeError(f"Unsupported geometry type {type(geometry).__name__}.")


def plane_reference(aircraft: Aircraft, ref_point: np.ndarray, height_ref: str) -> tuple[np.ndarray, str]:
    """Return the reference point [m] and the height mode for :func:`make_ground_plane`.

    For ``'qc'`` and ``'te'`` the point is the root of the main surface
    (:meth:`Aircraft.root_point`) and the mode is ``'ref'``.

    Raises
    ------
    ValueError
        If *height_ref* is not ``'ref'``, ``'min'``, ``'qc'`` or ``'te'``.
    """
    if height_ref in ("qc", "te"):
        return aircraft.root_point(height_ref), "ref"
    if height_ref not in ("ref", "min"):
        raise ValueError(f"Unknown height_ref={height_ref!r}; use 'ref', 'min', 'qc' or 'te'.")
    return np.asarray(ref_point, dtype=float), height_ref


def place_ground(
    aircraft: Aircraft,
    settings: SolverSettings,
    h: float,
    alpha: float,
    beta: float,
    phi: float,
    ref_point: np.ndarray,
    height_ref: str,
) -> tuple[GroundPlane, VortexLattice]:
    """Make the ground plane for one attitude and height.

    Parameters
    ----------
    aircraft : Aircraft
        The aircraft.
    settings : SolverSettings
        Solver settings for the probe lattice.
    h : float
        Height above the ground [m] of the point that *height_ref* selects.
    alpha : float
        Angle of attack [rad].
    beta : float
        Sideslip angle [rad].
    phi : float
        Bank angle [rad].
    ref_point : numpy.ndarray
        Moment reference point [m], shape (3,).
    height_ref : {'ref', 'min', 'qc', 'te'}
        Point that has the height *h*. See :func:`plane_reference`.

    Returns
    -------
    gp : GroundPlane
        The ground plane in body axes.
    probe : VortexLattice
        Lattice with one chordwise panel that was used to place the ground.
    """
    probe = build_lattice(aircraft, settings, collocation="vlm", n_chord=1)
    p, mode = plane_reference(aircraft, ref_point, height_ref)
    gp = make_ground_plane(probe, h, alpha, beta, phi, ref_point=p, height_ref=mode)
    return gp, probe


def reference_surface_indices(aircraft: Aircraft) -> list[int]:
    """Return the indices in ``aircraft.surfaces`` of the main surface and its mirror copies."""
    refs = aircraft.reference_surfaces()
    return [i for i, s in enumerate(aircraft.surfaces) if any(s is r for r in refs)]


def clearance_info(
    probe: VortexLattice,
    gp: GroundPlane,
    ref_point: np.ndarray,
    surface_indices: Sequence[int] = (0,),
) -> dict[str, float]:
    """Clearances [m] to the ground.

    The tip clearances are at the spanwise edges with the smallest and the
    largest y of the reference surfaces (*surface_indices*, the main surface
    and its mirror copies), so that a pair of mirrored half wings gives the
    two real tips.
    """
    pts = probe.all_points()
    heights = gp.height(pts)
    slices = [s for s in probe.surfaces if s.index in set(surface_indices)] or probe.surfaces[:1]
    le = np.vstack([s.edge_le for s in slices])
    te = np.vstack([s.edge_te for s in slices])
    i_l, i_r = int(np.argmin(le[:, 1])), int(np.argmax(le[:, 1]))
    tip_l = float(min(gp.height(le[[i_l]])[0], gp.height(te[[i_l]])[0]))
    tip_r = float(min(gp.height(le[[i_r]])[0], gp.height(te[[i_r]])[0]))
    return {
        "h_min": float(np.min(heights)),
        "h_ref": float(gp.height(np.asarray(ref_point, dtype=float)[None, :])[0]),
        "h_tip_left": tip_l,
        "h_tip_right": tip_r,
    }


MAX_BANK_DEG = 60.0  # Largest bank angle that the strike-limit search examines [deg]


def check_height(h: float) -> float:
    """Return *h* as a float. Raise ValueError if it is not finite or not larger than 0."""
    try:
        hv = float(h)
    except (TypeError, ValueError):
        raise ValueError(f"Ground height h={h!r} must be a finite number larger than 0.") from None
    if not (np.isfinite(hv) and hv > 0.0):
        raise ValueError(f"Ground height h={h} must be a finite number larger than 0 (0 < h < inf).")
    return hv


def find_bank_strike_limit(
    aircraft: Aircraft,
    settings: SolverSettings,
    h: float,
    alpha_deg: float,
    beta_deg: float,
    ref_point: np.ndarray,
    height_ref: str,
    max_bank_deg: float = MAX_BANK_DEG,
    tol_deg: float = 0.02,
) -> float:
    """Bank angle [deg] at which the first point of the aircraft touches the ground.

    All angles are in degrees. The height *h* (in the convention of
    *height_ref*) fixes the attitude at zero bank; the aircraft then banks
    about the free-stream direction with the reference point at a constant
    height. Returns *max_bank_deg* if no point touches below it (no contact
    up to the search limit, so the true limit is larger), and 0 if the level
    attitude already touches.
    """
    probe = build_lattice(aircraft, settings, collocation="vlm", n_chord=1)
    pts = probe.all_points()
    alpha, beta = np.radians(alpha_deg), np.radians(beta_deg)
    rp = np.asarray(ref_point, dtype=float)
    # Height of the reference point at zero bank, in this height convention.
    p, mode = plane_reference(aircraft, rp, height_ref)
    gp0 = make_ground_plane(probe, h, alpha, beta, 0.0, ref_point=p, height_ref=mode)
    h_rp = float(gp0.height(rp[None, :])[0])

    def margin(phi_deg: float) -> float:
        best = np.inf
        for sgn in (1.0, -1.0):  # either wing down
            gp = make_ground_plane(probe, h_rp, alpha, beta, sgn * np.radians(phi_deg),
                                   ref_point=rp, height_ref="ref")
            best = min(best, float(np.min(gp.height(pts))))
        return best

    if margin(0.0) <= 0.0:
        return 0.0
    if margin(max_bank_deg) > 0.0:
        return float(max_bank_deg)
    lo, hi = 0.0, float(max_bank_deg)
    while hi - lo > tol_deg:
        mid = 0.5 * (lo + hi)
        if margin(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def ground_frame_maps(gp: GroundPlane, condition: FlightCondition, h: float):
    """Return the maps from body axes to a ground frame for plots.

    Ground frame: x along the free stream, y = z x x, z normal to the ground,
    with the ground at ``z = -h`` (the older plotting convention).

    Returns
    -------
    pm : callable
        Map for points [m], shape (n, 3) to shape (n, 3).
    vm : callable
        Map for vectors, shape (n, 3) to shape (n, 3).
    """
    d = freestream_direction(condition.alpha, condition.beta)
    k = gp.normal
    y_ax = np.cross(k, d)

    def pm(p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        return np.column_stack([p @ d, p @ y_ax, gp.height(p) - h])

    def vm(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        return np.column_stack([v @ d, v @ y_ax, v @ k])

    return pm, vm


def analyze_ground_effect(
    geometry: Aircraft | LiftingSurface,
    h: float,
    alpha_deg: float = 5.0,
    phi_deg: float = 0.0,
    beta_deg: float = 0.0,
    *,
    V_inf: float = 50.0,
    rho: float = RHO_SL,
    ref_point: np.ndarray | Sequence[float] | None = None,
    height_ref: Literal["ref", "min", "qc", "te"] = "ref",
    solver: str | None = None,
    settings: SolverSettings | None = None,
    n_panels: int = 80,
    spacing: str = "auto",
    compute_strike_limit: bool = True,
) -> GroundEffectResult:
    """Aerodynamics of an aircraft near flat ground.

    Parameters
    ----------
    geometry : Aircraft or LiftingSurface
    h : float
        Height [m] above the ground (finite, > 0) in the convention of
        *height_ref*. With the default ``height_ref='ref'``, the ground is
        at the distance h below *ref_point*.
    alpha_deg : float
        Angle of attack = pitch attitude relative to the ground [deg] (default 5.0).
    phi_deg : float
        Bank angle [deg]; positive puts the right wing nearer to the ground.
    beta_deg : float
        Sideslip [deg]; positive is wind from the right.
    V_inf : float
        Free-stream speed [m/s].
    rho : float
        Air density [kg/m^3].
    ref_point : (3,) or None
        Moment reference point [m] and the point for ``height_ref='ref'``.
        Default: ``Aircraft.ref_point`` (the origin of the geometry axes if
        None), the same point that :func:`ventorum.analyze` uses.
    height_ref : {'ref', 'min', 'qc', 'te'}
        * ``'ref'``: height of *ref_point*.
        * ``'min'``: smallest clearance of any leading- or trailing-edge node.
        * ``'qc'`` / ``'te'``: root quarter-chord / trailing-edge point of the main surface
          (:meth:`Aircraft.root_point`).
    solver : str or None
        ``None``/``"auto"``: VLM. The lifting-line solvers are refused below
        h_min/c = 1.
    settings : SolverSettings or None
        Discretisation; ``n_chord=None`` resolves the chordwise panels from the gap.

    Returns
    -------
    GroundEffectResult

    Raises
    ------
    ValueError
        If *h* is not a finite number larger than 0, or an angle is not finite.
    GroundStrikeError
        If the aircraft touches the ground (no result is computed).
    ValidityError
        If a lifting-line solver is used below its validity limit, or the
        Fourier solver is used (it has no ground effect).
    """
    t0 = time.perf_counter()
    case = prepare_ground_case(geometry, h, alpha_deg, phi_deg, beta_deg, V_inf=V_inf, rho=rho,
                               ref_point=ref_point, height_ref=height_ref, solver=solver, settings=settings,
                               n_panels=n_panels, spacing=spacing)
    sol = make_solver(case["canonical"])
    res = sol.solve(case["aircraft"], case["condition"], case["settings"], ground=case["ground"],
                    ref_point=case["ref_point"])
    return ground_case_result(case, res, compute_strike_limit, t0)


def prepare_ground_case(
    geometry: Aircraft | LiftingSurface,
    h: float,
    alpha_deg: float = 5.0,
    phi_deg: float = 0.0,
    beta_deg: float = 0.0,
    *,
    V_inf: float = 50.0,
    rho: float = RHO_SL,
    ref_point: np.ndarray | Sequence[float] | None = None,
    height_ref: Literal["ref", "min", "qc", "te"] = "ref",
    solver: str | None = None,
    settings: SolverSettings | None = None,
    n_panels: int = 80,
    spacing: str = "auto",
    probe: VortexLattice | None = None,
) -> dict[str, Any]:
    """Check one case of :func:`analyze_ground_effect` and place its ground; return the case data.

    The arguments are those of :func:`analyze_ground_effect`; *probe* is
    the lattice with one chordwise panel that places the ground, when the
    caller has it (a sweep builds it once). The checks and the errors are
    those of :func:`analyze_ground_effect`.
    """
    h = check_height(h)
    for name, val in (("alpha_deg", alpha_deg), ("phi_deg", phi_deg), ("beta_deg", beta_deg)):
        if not np.isfinite(val):
            raise ValueError(f"{name}={val} must be a finite number.")

    aircraft = _as_aircraft(geometry)
    validate_aircraft(aircraft)
    aircraft.compute_reference_values()
    c_ref = aircraft.c_ref

    if settings is None:
        sett = SolverSettings(solver_type=solver or "auto", n_panels=n_panels, spacing=spacing)
    else:
        sett = settings.clone()
        if solver is not None:
            sett.solver_type = solver
    canonical = resolve_solver_type(sett.solver_type)
    if canonical == "fourier":
        raise ValidityError("The Fourier solver has no ground effect. Use solver='vlm'.")

    alpha, beta, phi = np.radians(alpha_deg), np.radians(beta_deg), np.radians(phi_deg)
    rp = aircraft.moment_reference() if ref_point is None else np.asarray(ref_point, dtype=float)
    if probe is None:
        gp, probe = place_ground(aircraft, sett, h, alpha, beta, phi, rp, height_ref)
    else:
        p, mode = plane_reference(aircraft, rp, height_ref)
        gp = make_ground_plane(probe, h, alpha, beta, phi, ref_point=p, height_ref=mode)
    clear = clearance_info(probe, gp, rp, reference_surface_indices(aircraft))
    if clear["h_min"] <= 0.0:
        raise GroundStrikeError(
            f"Ground strike: minimum clearance {clear['h_min']:.4g} m at h={h}, alpha={alpha_deg} deg, "
            f"phi={phi_deg} deg. No result is computed for a strike."
        )
    h_min_over_c = clear["h_min"] / c_ref
    if canonical in ("linear", "nonlinear") and h_min_over_c < LLT_GE_MIN_H_OVER_C:
        raise ValidityError(
            f"Lifting-line solver '{canonical}' is not valid in ground effect below "
            f"h_min/c = {LLT_GE_MIN_H_OVER_C} (here {h_min_over_c:.3f}). Use solver='vlm'."
        )
    return {
        "h": h, "alpha_deg": alpha_deg, "phi_deg": phi_deg, "beta_deg": beta_deg, "V_inf": V_inf, "rho": rho,
        "height_ref": height_ref, "aircraft": aircraft, "settings": sett, "canonical": canonical,
        "condition": FlightCondition(V_inf=V_inf, alpha=alpha, beta=beta, rho=rho, phi=phi),
        "ground": gp, "ref_point": rp, "clear": clear, "h_min_over_c": h_min_over_c,
    }


def ground_case_result(case: dict[str, Any], res, compute_strike_limit: bool, t0: float) -> GroundEffectResult:
    """Return the :class:`GroundEffectResult` of a case of :func:`prepare_ground_case` and its solver result."""
    aircraft, sett, gp, rp = case["aircraft"], case["settings"], case["ground"], case["ref_point"]
    h, alpha_deg, phi_deg, beta_deg = case["h"], case["alpha_deg"], case["phi_deg"], case["beta_deg"]
    V_inf, rho, height_ref, condition = case["V_inf"], case["rho"], case["height_ref"], case["condition"]
    clear, h_min_over_c = case["clear"], case["h_min_over_c"]
    S_ref, b_ref, c_ref = aircraft.S_ref, aircraft.b_ref, aircraft.c_ref
    alpha = condition.alpha
    tot = res.totals
    lattice: VortexLattice = res.details["lattice"]

    q_inf = 0.5 * rho * V_inf ** 2
    qS = q_inf * S_ref
    F = res.details["loads"].extras["force_total_body"]
    k = gp.normal
    CD = tot.CD_total if tot.CD_total is not None else tot.CDi
    L_over_D = tot.CL / CD if CD > 1e-12 else float("nan")

    # Standard body axes (x forward, y right, z down).
    CX_body = -float(F[0]) / qS
    CY_body = float(F[1]) / qS
    CZ_body = -float(F[2]) / qS
    # Stability-axis rolling and yawing moments (rotation by alpha about y).
    ca, sa = np.cos(alpha), np.sin(alpha)
    Cl_stab = tot.Cl * ca + tot.Cn * sa
    Cn_stab = tot.Cn * ca - tot.Cl * sa

    phi_limit = None
    limit_found = None
    if compute_strike_limit:
        phi_limit = find_bank_strike_limit(aircraft, sett, h, alpha_deg, beta_deg, rp, height_ref,
                                           max_bank_deg=MAX_BANK_DEG)
        limit_found = phi_limit < MAX_BANK_DEG

    ge_cond = GroundEffectCondition(
        h=h, alpha_deg=alpha_deg, phi_deg=phi_deg, beta_deg=beta_deg,
        V_inf=V_inf, rho=rho, ref_point=rp, height_ref=height_ref,
    )
    pm, vm = ground_frame_maps(gp, condition, h)

    return GroundEffectResult(
        condition=ge_cond,
        S_ref=float(S_ref), b_ref=float(b_ref), c_ref=float(c_ref), q_inf=float(q_inf),
        CL=float(tot.CL), CDi=float(tot.CDi), CDp=tot.CDp, CD=float(CD), CY=float(tot.CY),
        Cl=float(Cl_stab), Cm=float(tot.Cm), Cn=float(Cn_stab),
        L_over_D=float(L_over_D), e=float(tot.e),
        CX_body=CX_body, CY_body=CY_body, CZ_body=CZ_body,
        Cl_body=float(tot.Cl), Cm_body=float(tot.Cm), Cn_body=float(tot.Cn),
        L=float(tot.CL * qS), D=float(CD * qS), Y=float(tot.CY * qS),
        Mx=float(Cl_stab * qS * b_ref), My=float(tot.Cm * qS * c_ref), Mz=float(Cn_stab * qS * b_ref),
        h_min=clear["h_min"], h_ref=clear["h_ref"],
        h_tip_left=clear["h_tip_left"], h_tip_right=clear["h_tip_right"],
        phi_strike_limit=phi_limit, strike_limit_found=limit_found,
        h_over_c=float(h / c_ref), h_over_b=float(h / b_ref),
        spanwise=res.spanwise,
        reference_surface_indices=reference_surface_indices(aircraft),
        transformed_surfaces=lattice_to_discretized(lattice, point_map=pm, vector_map=vm),
        solver_result=res,
        execution_time=float(time.perf_counter() - t0),
        h_min_over_c=float(h_min_over_c),
        n_chord=int(lattice.n_chord),
        solver_type=res.solver_type,
        vertical_force_coefficient=float(F @ k) / qS,
    )
