# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Ventorum-2D viscous panel solver (fully simultaneous viscous-inviscid coupling).

The solver couples the inviscid panel method of
:mod:`ventorum.panel2d.solver` with the integral boundary layer of
:mod:`ventorum.ibl2d` on both surfaces and along the wake, through wall
and wake transpiration sources, and solves the whole system with one global
Newton iteration (see :mod:`ventorum.panel2d.vii`).

* Wake: the boundary layer is marched along a wake line that follows the
  inviscid streamline from the trailing edge (see
  :mod:`ventorum.panel2d.coupling`), with the wake closures (no wall
  friction, doubled dissipation) and a start from the sum of the two
  trailing-edge layers plus the trailing-edge gap.
* Drag: ``Cd`` is the Squire-Young drag at the wake end (Squire and Young
  1938); ``Cd_friction`` (skin friction along the surface) and
  ``Cd_pressure`` (surface pressure integration) are independent checks; no
  value is replaced by another.
* Ground effect: the image method of the inviscid solver; the image body
  carries the mirror image of the real boundary layer and wake (equal
  source strengths), and there is no boundary layer on the ground. There is
  no height limit: a large height gives the free-air result by computation.
* Alpha sweeps: the angles are solved from the angle closest to zero
  upwards and downwards; each solve starts from the converged solution of
  the previous angle (warm start), with the angle step halved (up to
  SUBSTEP_DEPTH times) when the warm start does not converge. A cold start
  begins from direct marches on the inviscid edge velocity at the inviscid
  zero-lift angle and continues to the angle in steps of at most
  CONTINUATION_STEP (start values only).
* A case that does not converge returns ``converged=False`` with a reason
  and a warning; its values are those of the last iterate.

Only one body is supported (a plain flap is part of the single-element
geometry); more bodies raise ValueError.

References
----------
M. Drela and M. B. Giles, Viscous-inviscid analysis of transonic and low
Reynolds number airfoils, AIAA Journal 25(10), 1987, pp. 1347-1355.
M. Drela, XFOIL: An analysis and design system for low Reynolds number
airfoils, Lecture Notes in Engineering 54, Springer, 1989, pp. 1-12.
M. J. Lighthill, On displacement thickness, Journal of Fluid Mechanics 4,
1958, pp. 383-392.
H. B. Squire and A. D. Young, The calculation of the profile drag of
aerofoils, ARC R and M 1838, 1938.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from ventorum.panel2d import coupling as cp
from ventorum.panel2d import vii
from ventorum.utils.parallel import blas_single_thread

__all__ = ["ViscousResult", "solve_viscous", "trace_wake_streamline", "prepare_case", "warm_state", "finish_angle",
           "row_from_result", "sweep_order"]

#: Number of times the angle step of a warm start may be halved.
SUBSTEP_DEPTH = 3
#: Largest angle step of the continuation from the zero-lift angle [deg].
CONTINUATION_STEP = 2.0


@dataclass
class ViscousResult:
    """Results of the 2D viscous panel solver.

    For an array of angles every field is an array (or a list) with one
    entry per angle, in the input order.

    Attributes
    ----------
    alpha : float or np.ndarray
        Angle of attack [deg].
    Cl : float or np.ndarray
        Lift coefficient from the circulation (Kutta-Joukowski) [-].
    Cd : float or np.ndarray
        Profile drag coefficient, Squire-Young at the wake end [-].
    Cd_friction : float or np.ndarray
        Skin-friction drag coefficient from the surface integration [-].
    Cd_pressure : float or np.ndarray
        Pressure drag coefficient from the surface pressure integration [-].
    Cm : float or np.ndarray
        Pitching moment coefficient about x_ref (pressure only) [-].
    transition_positions : tuple of float or list of tuple
        Transition x/c on the upper and lower side [-] (1.0 when the layer
        stays laminar to the trailing edge).
    separation_flags : tuple of bool or list of tuple
        True when the skin friction is negative at a station of the side.
    Cp : np.ndarray or list of np.ndarray
        Pressure coefficient at the nodes [-].
    H : np.ndarray or list of np.ndarray
        Shape factor at the nodes [-].
    theta : np.ndarray or list of np.ndarray
        Momentum thickness at the nodes [m].
    Cf : np.ndarray or list of np.ndarray
        Skin-friction coefficient (based on ue) at the nodes [-].
    iterations : int or np.ndarray
        Newton iterations [-].
    residual : float or np.ndarray
        Largest absolute residual of the final iterate [-].
    converged : bool or np.ndarray
        True when the residual is below the tolerance.
    coordinates : np.ndarray or list of np.ndarray
        Node coordinates as solved [m].
    reason : str or list of str
        Convergence status or failure reason.
    Cl_pressure : float or np.ndarray
        Lift coefficient from the surface pressure integration [-].
    delta_star : np.ndarray or list of np.ndarray
        Displacement thickness at the nodes [m].
    ue : np.ndarray or list of np.ndarray
        Edge velocity over the free-stream speed at the nodes [-].
    wake : dict or list of dict
        Wake line coordinates [m], theta [m], H [-] and ue [-].
    """

    alpha: float | np.ndarray
    Cl: float | np.ndarray
    Cd: float | np.ndarray
    Cd_friction: float | np.ndarray
    Cd_pressure: float | np.ndarray
    Cm: float | np.ndarray
    transition_positions: tuple | list
    separation_flags: tuple | list
    Cp: np.ndarray | list
    H: np.ndarray | list
    theta: np.ndarray | list
    Cf: np.ndarray | list
    iterations: int | np.ndarray
    residual: float | np.ndarray
    converged: bool | np.ndarray
    coordinates: np.ndarray | list
    reason: str | list = ""
    Cl_pressure: float | np.ndarray = 0.0
    delta_star: np.ndarray | list = field(default_factory=list)
    ue: np.ndarray | list = field(default_factory=list)
    wake: dict | list = field(default_factory=dict)

    @property
    def x_transition(self):
        """Return the transition positions (alias of transition_positions)."""
        return self.transition_positions

    @property
    def separated(self):
        """Return the separation flags (alias of separation_flags)."""
        return self.separation_flags


def trace_wake_streamline(bodies, alpha_deg: float, wake_length: float = 1.0, n_wake_panels: int = 30,
                          height=None) -> np.ndarray:
    """Return the wake line of the viscous solver for one body.

    The line starts at the trailing edge along the bisector of the two
    trailing-edge panels and follows the inviscid streamline (see
    :mod:`ventorum.panel2d.coupling`).

    Parameters
    ----------
    bodies : np.ndarray or list of np.ndarray
        Airfoil coordinates in Selig order [m] (one body).
    alpha_deg : float
        Angle of attack [deg].
    wake_length : float
        Wake length over the chord [-].
    n_wake_panels : int
        Number of wake panels.
    height : float or None
        Height above the ground [m], or None for free air.

    Returns
    -------
    np.ndarray
        Wake node coordinates, shape (n_wake_panels + 1, 2) [m].
    """
    body = _single_body(bodies)
    return cp.build_coupling_system(body, alpha_deg, height=height, wake_length=wake_length,
                                    n_wake=n_wake_panels).wake_nodes


def _single_body(bodies) -> np.ndarray:
    """Return the only body, or raise ValueError for several bodies."""
    if isinstance(bodies, (list, tuple)):
        if len(bodies) != 1:
            raise ValueError(
                f"solve_viscous supports one body (a plain flap is part of the single-element geometry); "
                f"got {len(bodies)} bodies."
            )
        bodies = bodies[0]
    arr = np.asarray(bodies, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"The body must be an (N, 2) array, got shape {arr.shape}.")
    return arr


def prepare_case(body, alpha, re, *, ncrit, xtr, height, height_reference, wake_length, n_wake):
    """Return the CaseSetup of one angle, or raise a ValueError with the reason."""
    sys_ = cp.build_coupling_system(body, alpha, height=height, height_reference=height_reference,
                                    wake_length=wake_length, n_wake=n_wake)
    return vii.CaseSetup(sys_, float(re), float(ncrit), (float(xtr[0]), float(xtr[1])))


def warm_state(setup: vii.CaseSetup, prev_setup: vii.CaseSetup, prev: vii.CaseState) -> vii.CaseState:
    """Return a first estimate for a case from the converged state of a neighbouring angle.

    theta, delta_star and c are taken node by node and wake station by wake
    station (they change smoothly with the angle; the mass defect does not
    near the stagnation point). The mass defects follow from the edge
    velocity of the new case with the old sources. The structure
    (stagnation panel, transition intervals) is then updated.
    """
    _, ds_prev, _ = vii.host_edge(prev_setup, prev)
    st = prev.copy()
    st.frozen = [False, False]
    xd = st.x.copy()
    xd[:, 1] = ds_prev
    ue_est, _, _ = vii.host_edge(setup, prev)
    vii.to_mixed(setup, st, xd, ue_est)
    vii._update_structure(setup, st)
    return st


_NUMPY = {"xp": np, "device": None, "dtype": None}


def _solve_one(setup, start, tol, max_iter, omega0, bk=None):
    """Run the Newton solve of one case from a start state."""
    bk = _NUMPY if bk is None else bk
    return vii.newton_solve([setup], [start], tol=tol, max_iter=max_iter, omega0=[omega0], **bk)[0]


def _cold_solve(setup, tol, max_iter, bk=None):
    """Solve a case from a cold start.

    Three first estimates are tried in turn until one converges:

    1. direct marches on the inviscid edge velocity, with the coupling
       factor omega starting at 1/64 (see
       :func:`ventorum.panel2d.vii.newton_solve`);
    2. quasi-simultaneous marches (:data:`ventorum.panel2d.vii.QS_SWEEPS`
       sweeps, see :func:`ventorum.panel2d.vii.initial_state`), which pass
       a separation point, with omega = 1;
    3. the direct marches with omega = 1.

    The attempt with the smallest residual is returned when none converges.
    """
    best = None
    for sweeps, omega0 in ((0, 1.0 / 64.0), (vii.QS_SWEEPS, 1.0), (0, 1.0)):
        try:
            init = vii.initial_state(setup, sweeps=sweeps)
        except np.linalg.LinAlgError:
            if sweeps == 0:
                raise
            continue
        res = _solve_one(setup, init, tol, max_iter, omega0, bk)
        if res.converged:
            return res
        if best is None or res.residual < best.residual:
            best = res
    return best


def solve_viscous(
    bodies,
    alpha_deg,
    Re: float,
    *,
    mach: float = 0.0,
    ncrit: float = 9.0,
    xtr: tuple = (1.0, 1.0),
    height=None,
    max_iter: int = 50,
    tol: float = 1e-6,
    wake_length: float = 1.0,
    n_wake: int = 30,
    height_reference: str = "pivot",
    backend: str = "numpy",
) -> ViscousResult:
    """Solve 2D viscous flow with fully simultaneous boundary-layer coupling.

    Parameters
    ----------
    bodies : np.ndarray or list of one np.ndarray
        Airfoil coordinates in Selig order [m or chord units].
    alpha_deg : float or array_like
        Angle of attack [deg].
    Re : float
        Reynolds number of the chord [-].
    mach : float
        Free-stream Mach number [-]; only 0 is supported.
    ncrit : float
        Critical amplification exponent of the e^N method [-].
    xtr : tuple of float
        Forced transition x/c on the upper and lower side [-]; 1.0 (or
        more) means free transition only.
    height : float or None
        Height above the ground [m] (of the pivot, the quarter chord, by
        default), or None for free air.
    max_iter : int
        Largest number of Newton iterations per angle.
    tol : float
        Convergence limit of the largest absolute residual [-].
    wake_length : float
        Wake length over the chord [-].
    n_wake : int
        Number of wake panels.
    height_reference : {"pivot", "trailing_edge"}
        Reference point of the height.
    backend : {"numpy"}
        Backend of the sequential solver (the batch solver
        :func:`ventorum.panel2d.viscous_batch.solve_viscous_batch` runs on
        torch).

    Returns
    -------
    ViscousResult
        Loads, distributions and convergence data per angle.
    """
    if abs(mach) > 1e-12:
        raise ValueError(f"Only mach=0.0 is supported in solve_viscous, got mach={mach}.")
    if backend != "numpy":
        raise ValueError(f"backend '{backend}' is not supported by solve_viscous; use 'numpy' "
                         "(solve_viscous_batch runs on torch).")
    if not np.isfinite(Re) or Re <= 0.0:
        raise ValueError(f"Reynolds number must be positive, got {Re}.")
    body = _single_body(bodies)
    is_scalar = np.ndim(alpha_deg) == 0
    alphas = np.atleast_1d(np.asarray(alpha_deg, dtype=np.float64))
    order_up, order_dn = sweep_order(alphas)
    out: dict[float, dict] = {}
    kw = dict(ncrit=ncrit, xtr=xtr, height=height, height_reference=height_reference,
              wake_length=wake_length, n_wake=n_wake)
    with blas_single_thread():
        first_state = None
        prev = None
        for a in order_up:
            out[a], state = _solve_angle(body, a, Re, prev, tol, max_iter, kw)
            if out[a]["converged"]:
                prev = state
                if first_state is None:
                    first_state = state
        prev = first_state
        for a in order_dn:
            out[a], state = _solve_angle(body, a, Re, prev, tol, max_iter, kw)
            if out[a]["converged"]:
                prev = state
    rows = [out[float(a)] for a in alphas]
    bad = [r for r in rows if not r["converged"]]
    for r in bad:
        warnings.warn(f"solve_viscous: alpha = {r['alpha']:g} deg did not converge: {r['reason']}",
                      UserWarning, stacklevel=2)
    if is_scalar:
        r = rows[0]
        return ViscousResult(**{k: r[k] for k in _FIELDS})
    return ViscousResult(
        alpha=alphas,
        Cl=np.array([r["Cl"] for r in rows]),
        Cd=np.array([r["Cd"] for r in rows]),
        Cd_friction=np.array([r["Cd_friction"] for r in rows]),
        Cd_pressure=np.array([r["Cd_pressure"] for r in rows]),
        Cm=np.array([r["Cm"] for r in rows]),
        transition_positions=[r["transition_positions"] for r in rows],
        separation_flags=[r["separation_flags"] for r in rows],
        Cp=[r["Cp"] for r in rows],
        H=[r["H"] for r in rows],
        theta=[r["theta"] for r in rows],
        Cf=[r["Cf"] for r in rows],
        iterations=np.array([r["iterations"] for r in rows], dtype=int),
        residual=np.array([r["residual"] for r in rows]),
        converged=np.array([r["converged"] for r in rows], dtype=bool),
        coordinates=[r["coordinates"] for r in rows],
        reason=[r["reason"] for r in rows],
        Cl_pressure=np.array([r["Cl_pressure"] for r in rows]),
        delta_star=[r["delta_star"] for r in rows],
        ue=[r["ue"] for r in rows],
        wake=[r["wake"] for r in rows],
    )


def sweep_order(alphas):
    """Return the solve order of a sweep: angles >= 0 upwards, then angles < 0 downwards.

    Parameters
    ----------
    alphas : array_like
        Angles of attack [deg].

    Returns
    -------
    tuple of list
        The upward and the downward sequence [deg] (unique values).
    """
    a = np.atleast_1d(np.asarray(alphas, dtype=np.float64))
    return sorted({float(v) for v in a if v >= 0.0}), sorted({float(v) for v in a if v < 0.0}, reverse=True)


_FIELDS = ("alpha", "Cl", "Cd", "Cd_friction", "Cd_pressure", "Cm", "transition_positions", "separation_flags",
           "Cp", "H", "theta", "Cf", "iterations", "residual", "converged", "coordinates", "reason",
           "Cl_pressure", "delta_star", "ue", "wake")


def _warm_path(body, alpha, re, prev, tol, max_iter, kw, depth, bk=None, first=None):
    """Solve one angle from the converged state of another angle.

    When the solve from ``prev`` does not converge, the angle step is halved
    (intermediate angles solved first), up to SUBSTEP_DEPTH times. The
    intermediate solutions only serve as start values. ``first`` holds the
    first attempt (setup, result) when the caller already ran it.
    """
    if first is None:
        setup = prepare_case(body, alpha, re, **kw)
        res = _solve_one(setup, warm_state(setup, prev[0], prev[1]), tol, max_iter, 1.0, bk)
    else:
        setup, res = first
    if res.converged or depth >= SUBSTEP_DEPTH:
        return setup, res
    mid = 0.5 * (prev[0].sys.alpha + alpha)
    s_mid, r_mid = _warm_path(body, mid, re, prev, tol, max_iter, kw, depth + 1, bk)
    if not r_mid.converged:
        return setup, res
    s_end, r_end = _warm_path(body, alpha, re, (s_mid, r_mid.state), tol, max_iter, kw, depth + 1, bk)
    return (s_end, r_end) if r_end.converged or r_end.residual < res.residual else (setup, res)


def finish_angle(body, alpha, re, prev, tol, max_iter, kw, bk=None, first=None):
    """Complete the solve of one angle; return the result row and (setup, final state).

    Start values: the converged state of the previous angle (with halved
    angle steps when needed, see :func:`_warm_path`); a cold start from
    direct marches (see :func:`_cold_path`) when there is no previous state
    or the warm path fails. ``first`` (setup, result) is the first warm
    attempt when the caller already ran it (batched solver); the remaining
    steps are then the same as in the sequential solver.
    """
    try:
        if prev is not None:
            setup, res = _warm_path(body, alpha, re, prev, tol, max_iter, kw, 0, bk, first)
            if not res.converged:
                cold_setup, cold = _cold_path(body, alpha, re, tol, max_iter, kw, bk)
                if cold.converged or cold.residual < res.residual:
                    setup, res = cold_setup, cold
        else:
            setup, res = _cold_path(body, alpha, re, tol, max_iter, kw, bk)
    except cp.WakeGroundError as exc:
        return _failed_row(alpha, body, str(exc)), None
    return row_from_result(setup, res, alpha), (setup, res.state)


def zero_lift_angle(body, kw) -> float:
    """Return the inviscid zero-lift angle of a section [deg] (free air or ground effect).

    Parameters
    ----------
    body : np.ndarray
        Airfoil coordinates [m].
    kw : dict
        Case settings (height and height reference are used).

    Returns
    -------
    float
        Angle where the inviscid lift is zero [deg], from two inviscid
        solves (0 and 4 deg).
    """
    from ventorum.panel2d.solver import solve_inviscid

    h = kw.get("height")
    opts = {} if h is None else {"height": h, "height_reference": kw.get("height_reference", "pivot")}
    cl = solve_inviscid(body, [0.0, 4.0], **opts).Cl
    return float(-4.0 * cl[0] / (cl[1] - cl[0]))


def _cold_path(body, alpha, re, tol, max_iter, kw, bk=None):
    """Solve an angle without a converged neighbour.

    A cold start (first estimate by direct marches) is reliable near the
    zero-lift angle, where the edge velocity has no strong suction peak.
    When the angle is more than CONTINUATION_STEP from the inviscid
    zero-lift angle, the solver starts cold at the zero-lift angle and
    continues to the angle in steps of at most CONTINUATION_STEP (each with
    the halving of :func:`_warm_path`); the intermediate solutions serve as
    start values only. When that start fails, a cold start at the angle
    itself is the last attempt.
    """
    a0 = zero_lift_angle(body, kw)
    if abs(alpha - a0) > CONTINUATION_STEP:
        try:
            s0 = prepare_case(body, a0, re, **kw)
            r0 = _cold_solve(s0, tol, max_iter, bk)
        except cp.WakeGroundError:
            r0 = None
        if r0 is not None and r0.converged:
            n = int(np.ceil(abs(alpha - a0) / CONTINUATION_STEP))
            prev = (s0, r0.state)
            setup, res = s0, r0
            for k in range(1, n + 1):
                a = a0 + (alpha - a0) * k / n
                setup, res = _warm_path(body, a, re, prev, tol, max_iter, kw, 0, bk)
                if not res.converged:
                    break
                prev = (setup, res.state)
            if res.converged:
                return setup, res
    setup = prepare_case(body, alpha, re, **kw)
    return setup, _cold_solve(setup, tol, max_iter, bk)


def _solve_angle(body, alpha, re, prev, tol, max_iter, kw):
    """Solve one angle with the numpy backend (see :func:`finish_angle`)."""
    return finish_angle(body, alpha, re, prev, tol, max_iter, kw)


def row_from_result(setup, res, alpha) -> dict:
    """Return the output row of a Newton result."""
    o = vii.case_outputs(setup, res.state, res.ue)
    return {
        "alpha": float(alpha),
        "Cl": o["Cl"],
        "Cd": o["Cd"],
        "Cd_friction": o["Cd_friction"],
        "Cd_pressure": o["Cd_pressure"],
        "Cm": o["Cm"],
        "transition_positions": o["xtr"],
        "separation_flags": o["separated"],
        "Cp": o["Cp"],
        "H": o["H"],
        "theta": o["theta"],
        "Cf": o["Cf"],
        "iterations": int(res.iterations),
        "residual": float(res.residual),
        "converged": bool(res.converged),
        "coordinates": setup.sys.nodes,
        "reason": res.reason,
        "Cl_pressure": o["Cl_pressure"],
        "delta_star": o["delta_star"],
        "ue": o["ue"],
        "wake": {"x": o["wake_x"], "theta": o["wake_theta"], "H": o["wake_H"], "ue": o["wake_ue"]},
    }


def _failed_row(alpha, body, reason) -> dict:
    """Return the output row of a case that could not be set up."""
    nan = float("nan")
    n = len(body)
    arr = np.full(n, nan)
    return {
        "alpha": float(alpha), "Cl": nan, "Cd": nan, "Cd_friction": nan, "Cd_pressure": nan, "Cm": nan,
        "transition_positions": (nan, nan), "separation_flags": (False, False), "Cp": arr, "H": arr,
        "theta": arr, "Cf": arr, "iterations": 0, "residual": nan, "converged": False,
        "coordinates": body, "reason": reason, "Cl_pressure": nan, "delta_star": arr, "ue": arr, "wake": {},
    }
