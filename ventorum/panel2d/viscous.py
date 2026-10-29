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
  ``Cl`` is the surface pressure lift in ground effect (Coulliette and
  Plotkin 1996, eq. 30). ``Cl_circulation`` keeps 2 Gamma / c. In free air
  ``Cl`` stays equal to ``Cl_circulation``.
* Alpha sweeps: the angles are solved from the angle closest to zero
  upwards and downwards; each solve starts from the converged solution of
  the previous angle (warm start), with the angle step halved (up to
  SUBSTEP_DEPTH times) when the warm start does not converge. A cold start
  begins from direct marches on the inviscid edge velocity at the inviscid
  zero-lift angle and continues to the angle in steps of at most
  CONTINUATION_STEP (start values only).
* A case that does not converge returns ``converged=False`` with a reason
  and a warning; its loads and distributions are NaN (the last iterate is
  not a solution), its residual and work counters are given. A singular
  linear system never raises an exception: the angle fails with a reason
  that names it.
* Fail fast: a Newton solve stops early by stagnation or divergence (see
  :func:`ventorum.panel2d.vii.newton_solve`); an angle has a work budget
  in Newton iterations over all its attempts (WORK_BUDGET); a failed
  quasi-simultaneous cold-start attempt far from convergence ends the cold
  start; a stagnation stop at an intermediate angle ends the halving of
  the angle step; the cold start at the zero-lift angle is solved once per
  sweep. Each early stop is named in the reason. The result counts the
  work of each angle (Newton iterations, linear solves and Newton solves).

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
C. Coulliette and A. Plotkin, Aerofoil ground effect revisited,
The Aeronautical Journal 100, 1996.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from ventorum.airfoil.geometry import AirfoilShape
from ventorum.panel2d import coupling as cp
from ventorum.panel2d import vii
from ventorum.panel2d.solver import resolve_reference
from ventorum.utils.parallel import blas_single_thread

__all__ = ["ViscousResult", "StopRules", "solve_viscous", "solve_viscous_flapped", "trace_wake_streamline", "prepare_case", "warm_state",
           "finish_angle", "row_from_result", "sweep_order", "check_chord_ref", "resolve_body", "wake_length_note",
           "warn_long_wake", "WAKE_LENGTH_LIMIT", "WORK_BUDGET", "COLD_START_BUDGET", "STEP_BUDGET"]

#: Number of times the angle step of a warm start may be halved.
SUBSTEP_DEPTH = 3
#: Largest angle step of the continuation from the zero-lift angle [deg].
CONTINUATION_STEP = 2.0
#: Work budget of one angle: largest sum of Newton iterations of the attempts at the angle itself (warm
#: start, halved angle steps from a converged neighbour, the solve at the angle in the last continuation
#: step and the cold start at the angle) [-] (0: no budget, also no budget of the continuation path).
WORK_BUDGET = 250
#: Work budget of the cold start at the zero-lift angle of the continuation path [-]. Measured: at most
#: 65 Newton iterations (NACA 4412, Re 1e6, 4 deg, 161 to 401 points, free air and h/c 2). The cold start
#: tries at most three first estimates, each up to its stage limit (max_iter, 50 by default), so it uses
#: at most 150 Newton iterations at the default limit; the budget is above that value, so it never cuts
#: a cold start at the default limit.
COLD_START_BUDGET = 200
#: Work budget of one continuation step (from one angle of the path to the next, halved steps
#: included, without the solves at the angle itself) [-]. Measured: at most 225 Newton iterations in
#: the sample of COLD_START_BUDGET (whole cases up to 378 in free air and 490 at h/c 2). The budget is
#: that value with a margin of about 1.3. It is also above the former budget of the whole angle (250),
#: so a step that the former budget allowed is not cut.
STEP_BUDGET = 300
#: Largest wake length over the chord [-] without a warning. The far-wake model drives the wake shape
#: factor H to 1 about 1.0 to 1.2 chords behind the trailing edge, so no state exists beyond, and a
#: longer wake does not converge in the measured sample. The Squire-Young drag is already converged at
#: a wake length of 0.5 chords (within 0.1 % up to 1.5 chords at 4 deg).
WAKE_LENGTH_LIMIT = 1.5
#: Two angles closer than this are the same angle [deg]: for the work budgets, the best attempt at the angle
#: and the number of continuation steps (a round-off tolerance).
ANGLE_MATCH_DEG = 1e-9
#: A failed quasi-simultaneous cold-start attempt that stops above this residual ends the cold start [-].
COLD_SKIP_RESIDUAL = 1e-2


@dataclass(frozen=True)
class StopRules:
    """Early-stop rules of the viscous solver (fail fast).

    Attributes
    ----------
    stall_window : int
        Stagnation stop of a Newton solve: the solve stops when its
        smallest residual has not fallen by the factor ``stall_factor`` in
        the last ``stall_window`` iterations (window restarted when omega
        changes and, at omega = 1, when the structure changes) [-]. 0
        switches the stagnation stops off, and also the end of the cold
        start after a failed quasi-simultaneous first estimate.
    stall_factor : float
        Factor of the stagnation stop [-], in (0, 1).
    divergence_factor : float
        Divergence stop of a Newton solve: the solve stops when, at
        omega = 1, the residual exceeds this factor times its smallest
        earlier value [-]. 0 switches the rule off.
    work_budget : int
        Largest sum of Newton iterations of the attempts at one angle
        (warm start, halved angle steps from a converged neighbour, the
        solve at the angle in the last step of the continuation path and
        the cold start at the angle) [-]. 0 means no budget: then the
        continuation path has no budget either.
    cold_start_budget : int
        Largest sum of Newton iterations of the cold start at the
        zero-lift angle of the continuation path [-] (see
        :data:`COLD_START_BUDGET`). 0 means no budget.
    step_budget : int
        Largest sum of Newton iterations of one step of the continuation
        path, halved steps included, without the solves at the angle
        itself [-] (see :data:`STEP_BUDGET`). 0 means no budget.
    lm_trials : int
        Largest number of damping values (one linear solve each) of a
        Levenberg-Marquardt fallback step of a Newton iteration after a
        fallback step of the same Newton solve found no decrease of the
        residual above the stagnation floor [-], at least 1 (the fallback
        steps before that try :data:`ventorum.panel2d.vii.LM_TRIALS`
        values; see :func:`ventorum.panel2d.vii.newton_solve`).
    """

    stall_window: int = vii.STALL_WINDOW
    stall_factor: float = vii.STALL_FACTOR
    divergence_factor: float = vii.DIVERGENCE_FACTOR
    work_budget: int = WORK_BUDGET
    lm_trials: int = vii.LM_RETRY_TRIALS
    cold_start_budget: int = COLD_START_BUDGET
    step_budget: int = STEP_BUDGET

    def __post_init__(self):
        if int(self.stall_window) != self.stall_window or self.stall_window < 0:
            raise ValueError(f"stall_window must be an integer >= 0, got {self.stall_window}.")
        if not 0.0 < self.stall_factor < 1.0:
            raise ValueError(f"stall_factor must lie in (0, 1), got {self.stall_factor}.")
        if not (self.divergence_factor == 0.0 or self.divergence_factor > 1.0):
            raise ValueError(f"divergence_factor must be 0 (off) or larger than 1, got {self.divergence_factor}.")
        if int(self.work_budget) != self.work_budget or self.work_budget < 0:
            raise ValueError(f"work_budget must be an integer >= 0, got {self.work_budget}.")
        if int(self.lm_trials) != self.lm_trials or self.lm_trials < 1:
            raise ValueError(f"lm_trials must be an integer >= 1, got {self.lm_trials}.")
        for name in ("cold_start_budget", "step_budget"):
            value = getattr(self, name)
            if int(value) != value or value < 0:
                raise ValueError(f"{name} must be an integer >= 0, got {value}.")


class _BudgetUsedUp(Exception):
    """The work budget of an angle is used up."""


class _StageBudgetUsedUp(Exception):
    """The work budget of a stage of the continuation path is used up."""


class _Stage:
    """A stage of the continuation path: its name, its work budget and its work."""

    def __init__(self, name: str, kind: str, budget: int):
        self.name = name
        self.kind = kind
        self.budget = int(budget)
        self.iterations = 0

    def label(self) -> str:
        """Return the name of the budget of the stage."""
        return f"{self.kind} budget of {self.budget} Newton iterations"


class _AngleWork:
    """Work counters, work budgets and early-stop notes of one angle.

    ``iterations`` counts the Newton iterations charged to the work
    budget of the angle (the solves at the angle itself, and all solves
    outside the continuation path). ``path_iterations`` counts the Newton
    iterations of the continuation path at other angles (the cold start at
    the zero-lift angle and the solves of its steps), which are charged to
    the budget of their stage. The work counters of the result give the
    sum of the two.
    """

    def __init__(self, alpha, rules: StopRules):
        self.alpha = float(alpha)
        self.rules = rules
        self.iterations = 0
        self.path_iterations = 0
        self.linear_solves = 0
        self.solves = 0
        self.notes = []
        self.retry_notes = []
        self.best = None
        self.stage = None

    def _stage_of(self, alpha):
        """Return the stage that a solve at ``alpha`` [deg] is charged to, or None (the angle budget).

        The last continuation step ends at the angle up to round-off (``a0 + (alpha - a0) * n / n``), so
        a solve within ANGLE_MATCH_DEG of the angle is a solve at the angle.
        """
        if self.stage is None or (alpha is not None and abs(float(alpha) - self.alpha) <= ANGLE_MATCH_DEG):
            return None
        return self.stage

    def cap(self, max_iter: int, alpha=None) -> int:
        """Return the iteration limit of the next Newton solve (at ``alpha`` [deg], None: the angle)."""
        if self.rules.work_budget <= 0:
            return int(max_iter)
        stage = self._stage_of(alpha)
        if stage is not None:
            if stage.budget <= 0:
                return int(max_iter)
            return int(min(max_iter, stage.budget - stage.iterations))
        return int(min(max_iter, self.rules.work_budget - self.iterations))

    def check(self, max_iter: int, alpha=None) -> int:
        """Return the iteration limit of the next Newton solve; raise when its budget is used up."""
        cap = self.cap(max_iter, alpha)
        if cap <= 0:
            raise _StageBudgetUsedUp() if self._stage_of(alpha) is not None else _BudgetUsedUp()
        return cap

    def add(self, setup, res, cap: int, max_iter: int):
        """Count a Newton solve; mark a stop at a limit below max_iter as a budget stop."""
        stage = self._stage_of(setup.sys.alpha)
        if res.status == "max_iter" and cap < max_iter:
            retry_reason = res.reason if res._first_stop else ""
            res.status = "budget"
            if stage is None:
                label = f"work budget of {self.rules.work_budget} Newton iterations per angle"
            else:
                label = f"{stage.label()} ({stage.name})"
            res.reason = f"{label} used up (largest residual {res.residual:.3e})"
            if retry_reason:
                res.reason += "; " + retry_reason
            if stage is None:
                self.note(f"{label} used up")
        if stage is None:
            self.iterations += int(res.iterations)
        else:
            stage.iterations += int(res.iterations)
            self.path_iterations += int(res.iterations)
        self.linear_solves += int(res.linear_solves)
        self.solves += 1 + bool(res._first_stop)
        if res._first_stop:
            self.retry_notes.append(f"stage at alpha = {setup.sys.alpha:g} deg: {res.reason}")
        if (abs(float(setup.sys.alpha) - self.alpha) <= ANGLE_MATCH_DEG
                and (self.best is None or res.residual < self.best[1].residual)):
            self.best = (setup, res)

    def note(self, text: str):
        """Record an early stop of the chain of attempts."""
        if text not in self.notes:
            self.notes.append(text)

    def counters(self) -> dict:
        """Return the work counters of the angle (continuation path included)."""
        return {"newton_iterations": self.iterations + self.path_iterations, "linear_solves": self.linear_solves,
                "newton_solves": self.solves}


@dataclass
class ViscousResult:
    """Results of the 2D viscous panel solver.

    For an array of angles every field is an array (or a list) with one
    entry per angle, in the input order. The coefficients use the
    reference chord (see :func:`solve_viscous`, ``chord_ref``). The loads
    and distributions of an angle that did not converge are NaN.

    Attributes
    ----------
    alpha : float or np.ndarray
        Angle of attack [deg].
    Cl : float or np.ndarray
        Lift coefficient [-]: circulation lift in free air, surface
        pressure lift in ground effect.
    Cl_circulation : float or np.ndarray
        Lift coefficient from the total circulation (2 Gamma / c) [-].
    Cd : float or np.ndarray
        Profile drag coefficient, Squire-Young at the wake end [-].
    Cd_friction : float or np.ndarray
        Skin-friction drag coefficient from the surface integration [-].
    Cd_pressure : float or np.ndarray
        Pressure drag coefficient from the surface pressure integration [-].
    Cm : float or np.ndarray
        Pitching moment coefficient about the moment reference point
        (pressure only) [-]: ``moment_ref`` of :func:`solve_viscous`, by
        default the quarter-chord point on the chord line of the
        reference geometry.
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
    newton_iterations : int or np.ndarray
        Work of the angle: sum of the Newton iterations of all its Newton
        solves (warm start, halved angle steps, cold start and continuation,
        failed attempts included) [-].
    linear_solves : int or np.ndarray
        Work of the angle: solves of the Newton or Levenberg-Marquardt
        linear system over all its Newton solves [-].
    newton_solves : int or np.ndarray
        Number of Newton solves (attempts) of the angle [-].
    details : dict or list of dict
        Continuation details (reached deflection and step history).
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
    newton_iterations: int | np.ndarray = 0
    linear_solves: int | np.ndarray = 0
    newton_solves: int | np.ndarray = 0
    Cl_circulation: float | np.ndarray = 0.0
    details: dict | list = field(default_factory=dict)

    @property
    def reached_deflection(self):
        """Return the reached flap deflection [deg]."""
        if isinstance(self.details, list):
            return [d.get("reached_deflection") if isinstance(d, dict) else None for d in self.details]
        return self.details.get("reached_deflection") if isinstance(self.details, dict) else None

    @property
    def step_history(self):
        """Return the continuation step history (deflection, converged, iterations)."""
        if isinstance(self.details, list):
            return [d.get("step_history") if isinstance(d, dict) else None for d in self.details]
        return self.details.get("step_history") if isinstance(self.details, dict) else None

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
    bodies : np.ndarray, AirfoilShape, or list of one of them
        Airfoil coordinates in Selig order [m] (one body), or a shape.
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


def _single_body(bodies):
    """Return the only body (an (N, 2) array or an AirfoilShape), or raise ValueError for several bodies."""
    if isinstance(bodies, (list, tuple)):
        if len(bodies) != 1:
            raise ValueError(
                f"solve_viscous supports one body (a plain flap is part of the single-element geometry); "
                f"got {len(bodies)} bodies."
            )
        bodies = bodies[0]
    if isinstance(bodies, AirfoilShape):
        return bodies
    arr = np.asarray(bodies, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"The body must be an (N, 2) array, got shape {arr.shape}.")
    return arr


def resolve_body(bodies, chord_ref=None, moment_ref=None):
    """Return the coordinates and the reference values of the one body of a viscous solve.

    Parameters
    ----------
    bodies : np.ndarray, AirfoilShape, or list of one of them
        Airfoil coordinates in Selig order [m], or a shape.
    chord_ref : float or None
        Explicit reference chord [m], or None for the default.
    moment_ref : tuple of float or None
        Explicit moment reference point (x, y) [m], or None for the
        default.

    Returns
    -------
    body : np.ndarray
        Coordinates, shape (N, 2) [m].
    chord_ref : float
        Reference chord [m] (see
        :func:`ventorum.panel2d.solver.resolve_reference`).
    moment_ref : tuple of float
        Moment reference point (x, y) [m].
    pivot : tuple of float
        Default pivot of the ground-effect placement [m]: the quarter-chord
        point on the chord line of the reference geometry.
    """
    body = _single_body(bodies)
    chord, moment, pivot = resolve_reference(body, check_chord_ref(chord_ref), moment_ref)
    coords = body.coords if isinstance(body, AirfoilShape) else body
    return coords, chord, moment, pivot


def prepare_case(body, alpha, re, *, ncrit, xtr, height, height_reference, wake_length, n_wake, chord_ref=None,
                 moment_ref=None, pivot=None):
    """Return the CaseSetup of one angle, or raise a ValueError with the reason."""
    sys_ = cp.build_coupling_system(body, alpha, height=height, height_reference=height_reference, pivot=pivot,
                                    chord_ref=chord_ref, moment_ref=moment_ref, wake_length=wake_length,
                                    n_wake=n_wake)
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


def _solve_one(setup, start, tol, max_iter, omega0, bk=None, ctl=None, *, retry_window=False):
    """Run the Newton solve of one case from a start state.

    With ``ctl`` (the work record of the angle) the solve uses its stop
    rules, its iteration limit is cut to the work budget that is left (the
    budget of the angle, or the budget of the stage of the continuation
    path), and its work is counted. _BudgetUsedUp (_StageBudgetUsedUp for
    a stage) is raised when no budget is left.
    """
    bk = _NUMPY if bk is None else bk
    rules = StopRules() if ctl is None else ctl.rules
    cap = int(max_iter) if ctl is None else ctl.check(max_iter, setup.sys.alpha)
    res = vii.newton_solve([setup], [start], tol=tol, max_iter=cap, omega0=[omega0], stall_window=rules.stall_window,
                           stall_factor=rules.stall_factor, divergence_factor=rules.divergence_factor,
                           lm_trials=rules.lm_trials, retry_window=retry_window, **bk)[0]
    if ctl is not None:
        ctl.add(setup, res, cap, max_iter)
    return res


def _cold_solve(setup, tol, max_iter, bk=None, ctl=None, *, retry=True):
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
    A first estimate whose marches meet a singular linear system is
    skipped; when no first estimate exists, the result has the status
    "singular". With ``ctl`` (the work record of the angle) and the stagnation stops
    on, a failed attempt from the quasi-simultaneous marches that stopped
    by stagnation, divergence or a non-finite residual with a residual
    above COLD_SKIP_RESIDUAL ends the cold start: the third first estimate
    did not rescue such a case in the test sample. A failure of the first
    estimate does not end the cold start (the quasi-simultaneous marches
    often rescue it).
    Only the quasi-simultaneous estimate may retry once after a
    residual-window stop, within the same stage limit and angle budget.
    ``retry=False`` removes this retry. The cold start at the zero-lift
    angle of a continuation uses it: in the test sample no retry there
    converged, and each one used its full stage limit.
    """
    best = None
    for sweeps, omega0 in ((0, 1.0 / 64.0), (vii.QS_SWEEPS, 1.0), (0, 1.0)):
        try:
            init = vii.initial_state(setup, sweeps=sweeps)
        except np.linalg.LinAlgError:
            continue
        try:
            res = _solve_one(setup, init, tol, max_iter, omega0, bk, ctl, retry_window=retry and sweeps > 0)
        except _StageBudgetUsedUp:
            # The budget of the stage of the continuation path is used up: keep the best attempt.
            if best is None:
                raise
            break
        if res.converged:
            return res
        if best is None or res.residual < best.residual:
            best = res
        if (ctl is not None and ctl.rules.stall_window > 0 and sweeps > 0
                and res.status in ("stagnation", "divergence", "not_finite") and not res.residual <= COLD_SKIP_RESIDUAL):
            ctl.note(f"cold start ended after a stop ({res.status.replace('_', '-')}) of the quasi-simultaneous "
                     f"first estimate with a residual above {COLD_SKIP_RESIDUAL:g}")
            break
    if best is None:
        best = vii.NewtonResult(None, None, False, 0, float("inf"),
                                "singular: the marches of every first estimate of the cold start meet a singular "
                                "linear system", 0, "singular")
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
    stall_window: int = vii.STALL_WINDOW,
    stall_factor: float = vii.STALL_FACTOR,
    divergence_factor: float = vii.DIVERGENCE_FACTOR,
    work_budget: int = WORK_BUDGET,
    lm_trials: int = vii.LM_RETRY_TRIALS,
    chord_ref: float | None = None,
    moment_ref: tuple[float, float] | None = None,
) -> ViscousResult:
    """Solve 2D viscous flow with fully simultaneous boundary-layer coupling.

    Parameters
    ----------
    bodies : np.ndarray, AirfoilShape, or list of one of them
        Airfoil coordinates in Selig order [m or chord units], or an
        :class:`ventorum.airfoil.AirfoilShape`. The reference geometry of
        the section (:func:`ventorum.panel2d.solver.section_reference`)
        gives the default reference chord, moment reference point and
        pivot; a shape with a deflected plain flap holds the geometry of
        the section before the deflection.
    alpha_deg : float or array_like
        Angle of attack [deg].
    Re : float
        Reynolds number of the reference chord [-].
    mach : float
        Free-stream Mach number [-]; only 0 is supported.
    ncrit : float
        Critical amplification exponent of the e^N method [-]. The default
        9 is the usual value of the e^N method for a low-disturbance wind
        tunnel. It is a code setting: no source in this repository
        confirms it yet.
    xtr : tuple of float
        Forced transition x/c on the upper and lower side [-] (x from the
        most forward point, over the reference chord); 1.0 (or more) means
        free transition only.
    height : float or None
        Height above the ground [m] (of the pivot, the quarter-chord point
        on the chord line of the reference geometry, by default), or None
        for free air.
    max_iter : int
        Largest number of Newton iterations per angle.
    tol : float
        Convergence limit of the largest absolute residual [-].
    wake_length : float
        Wake length over the chord [-]. The default 1 is a code setting,
        not from a source. Measured reason: the Squire-Young drag changes
        by less than 0.1 % from 0.5 to 1.5 chords (NACA 4412, Re 1e6,
        4 deg; tests/test_panel2d_viscous_wake.py). A wake longer than
        :data:`WAKE_LENGTH_LIMIT` (1.5) gives a UserWarning and a note in
        the reason of each angle: it does not converge, because the wake
        shape factor of the wake model reaches 1 at a finite distance
        behind the trailing edge (see ``docs/theory/panel2d_viscous.md``).
    n_wake : int
        Number of wake panels.
    height_reference : {"pivot", "trailing_edge"}
        Reference point of the height.
    backend : {"numpy"}
        Backend of the sequential solver (the batch solver
        :func:`ventorum.panel2d.viscous_batch.solve_viscous_batch` runs on
        torch).
    stall_window : int
        Stagnation stop of each Newton solve: it stops when its smallest
        residual has not fallen by the factor ``stall_factor`` in the last
        ``stall_window`` iterations (window restarted when omega changes
        and, at omega = 1, when the structure changes), or when, at
        omega = 1, the line search and the Levenberg-Marquardt fallback
        both fail twice on the same state [-]. 0 switches both rules off,
        and also the end of the cold start after a failed
        quasi-simultaneous first estimate.
    stall_factor : float
        Factor of the stagnation stop [-], in (0, 1).
    divergence_factor : float
        Divergence stop of each Newton solve: it stops when, at omega = 1,
        the residual exceeds this factor times its smallest earlier value
        [-]. 0 switches the rule off.
    work_budget : int
        Largest sum of Newton iterations of the attempts at one angle
        [-]. The continuation path from the zero-lift angle has its own
        budgets (:data:`COLD_START_BUDGET` for its cold start and
        :data:`STEP_BUDGET` for each step, see :class:`StopRules`). 0
        means no budget, also on the continuation path.
    lm_trials : int
        Largest number of damping values (one linear solve each) of a
        Levenberg-Marquardt fallback step of a Newton iteration after a
        fallback step of the same Newton solve found no decrease of the
        residual above the stagnation floor [-], at least 1 (the fallback
        steps before that try :data:`ventorum.panel2d.vii.LM_TRIALS`
        values; see :func:`ventorum.panel2d.vii.newton_solve`). The angles
        that converge do not depend on it in the test sample; it cuts the
        linear solves of failed attempts.
    chord_ref : float or None
        Reference chord [m] of the coefficients, of the Reynolds number,
        of x/c and of the wake length, positive. It wins over the
        reference geometry. Default: the reference chord of the section
        (see :func:`ventorum.panel2d.solver.section_reference`): for a
        flapped shape the chord of the section before the deflection, as
        in thin-airfoil theory (Glauert 1927); else the chord of the
        section as given, the distance from the leading edge to the
        trailing-edge midpoint (see
        :func:`ventorum.panel2d.solver.section_chord`). A flapped section
        given as an array has a shorter chord than the undeflected
        section: give ``chord_ref``, or give the flapped shape.
    moment_ref : tuple of float or None
        Moment reference point (x, y) [m] in the section frame. It wins
        over the reference geometry. Default: the quarter-chord point on
        the chord line of the reference geometry, 0.25 of the way from
        its leading edge to its trailing-edge midpoint (see
        :func:`ventorum.panel2d.solver.reference_point`). It does not move
        the pivot.

    Returns
    -------
    ViscousResult
        Loads, distributions, convergence data and work counters per angle.
        The loads and distributions of an angle that did not converge are
        NaN.
    """
    if abs(mach) > 1e-12:
        raise ValueError(f"Only mach=0.0 is supported in solve_viscous, got mach={mach}.")
    if backend != "numpy":
        raise ValueError(f"backend '{backend}' is not supported by solve_viscous; use 'numpy' "
                         "(solve_viscous_batch runs on torch).")
    if not np.isfinite(Re) or Re <= 0.0:
        raise ValueError(f"Reynolds number must be positive, got {Re}.")
    rules = StopRules(stall_window=stall_window, stall_factor=stall_factor, divergence_factor=divergence_factor,
                      work_budget=work_budget, lm_trials=lm_trials)
    body, chord_ref, moment_ref, pivot = resolve_body(bodies, chord_ref, moment_ref)
    wake_note = warn_long_wake(wake_length, "solve_viscous")
    is_scalar = np.ndim(alpha_deg) == 0
    alphas = np.atleast_1d(np.asarray(alpha_deg, dtype=np.float64))
    order_up, order_dn = sweep_order(alphas)
    out: dict[float, dict] = {}
    kw = dict(ncrit=ncrit, xtr=xtr, height=height, height_reference=height_reference,
              wake_length=wake_length, n_wake=n_wake, chord_ref=chord_ref, moment_ref=moment_ref, pivot=pivot)
    cache: dict = {}
    with blas_single_thread():
        first_state = None
        prev = None
        for a in order_up:
            out[a], state = _solve_angle(body, a, Re, prev, tol, max_iter, kw, rules, cache)
            if out[a]["converged"]:
                prev = state
                if first_state is None:
                    first_state = state
        prev = first_state
        for a in order_dn:
            out[a], state = _solve_angle(body, a, Re, prev, tol, max_iter, kw, rules, cache)
            if out[a]["converged"]:
                prev = state
    rows = [out[float(a)] for a in alphas]
    add_note(rows, wake_note)
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
        Cl_circulation=np.array([r["Cl_circulation"] for r in rows]),
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
        newton_iterations=np.array([r["newton_iterations"] for r in rows], dtype=int),
        linear_solves=np.array([r["linear_solves"] for r in rows], dtype=int),
        newton_solves=np.array([r["newton_solves"] for r in rows], dtype=int),
    )


def check_chord_ref(chord_ref):
    """Return the reference chord as a float, or None; raise ValueError when it is not positive and finite.

    Parameters
    ----------
    chord_ref : float or None
        Reference chord [m], or None for the chord of the section.

    Returns
    -------
    float or None
        The reference chord [m].
    """
    if chord_ref is None:
        return None
    c = float(chord_ref)
    if not np.isfinite(c) or c <= 0.0:
        raise ValueError(f"chord_ref must be a positive length, got {chord_ref}.")
    return c


def wake_length_note(wake_length) -> str | None:
    """Return the note of a wake longer than :data:`WAKE_LENGTH_LIMIT`, or None.

    Parameters
    ----------
    wake_length : float
        Wake length over the reference chord [-].

    Returns
    -------
    str or None
        The reason text of a long wake, None for a wake length up to the
        limit.
    """
    if not float(wake_length) > WAKE_LENGTH_LIMIT:
        return None
    return (f"wake length {float(wake_length):g} chords is above {WAKE_LENGTH_LIMIT:g} chords: the far-wake model "
            "drives the wake shape factor H to 1 about 1.0 to 1.2 chords behind the trailing edge, so a longer "
            "wake does not converge; the Squire-Young drag is converged at a wake length of 0.5 chords")


def warn_long_wake(wake_length, solver: str, stacklevel: int = 3) -> str | None:
    """Give a UserWarning for a wake longer than :data:`WAKE_LENGTH_LIMIT`; return its note or None.

    Parameters
    ----------
    wake_length : float
        Wake length over the reference chord [-].
    solver : str
        Name of the calling solver (for the message).
    stacklevel : int
        Stack level of the warning.

    Returns
    -------
    str or None
        The note (see :func:`wake_length_note`).
    """
    note = wake_length_note(wake_length)
    if note is not None:
        warnings.warn(f"{solver}: {note}.", UserWarning, stacklevel=stacklevel)
    return note


def add_note(rows, note: str | None):
    """Append ``note`` to the reason of each result row (no change when ``note`` is None)."""
    if note is None:
        return
    for r in rows:
        if note not in r["reason"]:
            r["reason"] = f"{r['reason']}; {note}" if r["reason"] else note


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
           "Cl_pressure", "delta_star", "ue", "wake", "newton_iterations", "linear_solves", "newton_solves",
           "Cl_circulation")


def _warm_path(body, alpha, re, prev, tol, max_iter, kw, depth, bk=None, first=None, ctl=None, mid_step=False):
    """Solve one angle from the converged state of another angle.

    When the solve from ``prev`` does not converge, the angle step is halved
    (intermediate angles solved first), up to SUBSTEP_DEPTH times. The
    intermediate solutions only serve as start values. ``first`` holds the
    first attempt (setup, result) when the caller already ran it. With
    ``ctl`` (the work record of the angle), the solve at an intermediate
    angle (``mid_step``) that stops by stagnation is not halved again; the
    halving at that level then ends. The second half-step (from the
    intermediate angle to the angle) is always halved again: in the test
    sample such halvings rescued angles down to the last depth.
    """
    if first is None:
        setup = prepare_case(body, alpha, re, **kw)
        res = _solve_one(setup, warm_state(setup, prev[0], prev[1]), tol, max_iter, 1.0, bk, ctl)
    else:
        setup, res = first
    if res.converged or depth >= SUBSTEP_DEPTH:
        return setup, res
    if ctl is not None and mid_step and res.status == "stagnation":
        ctl.note("halving of the angle step ended after a stagnation stop at an intermediate angle")
        return setup, res
    mid = 0.5 * (prev[0].sys.alpha + alpha)
    s_mid, r_mid = _warm_path(body, mid, re, prev, tol, max_iter, kw, depth + 1, bk, ctl=ctl, mid_step=True)
    if not r_mid.converged:
        return setup, res
    s_end, r_end = _warm_path(body, alpha, re, (s_mid, r_mid.state), tol, max_iter, kw, depth + 1, bk, ctl=ctl)
    return (s_end, r_end) if r_end.converged or r_end.residual < res.residual else (setup, res)


def finish_angle(body, alpha, re, prev, tol, max_iter, kw, bk=None, first=None, rules=None, cache=None):
    """Complete the solve of one angle; return the result row and (setup, final state).

    Start values: the converged state of the previous angle (with halved
    angle steps when needed, see :func:`_warm_path`); a cold start from
    direct marches (see :func:`_cold_path`) when there is no previous state
    or the warm path fails. ``first`` (setup, result) is the first warm
    attempt when the caller already ran it (batched solver); the remaining
    steps are then the same as in the sequential solver. ``rules``
    (:class:`StopRules`, default values when None) sets the early stops
    and the work budget of the angle. When the budget is used up (or a
    linear system outside the Newton solves is singular), the attempt at
    the angle with the smallest residual gives the convergence data of the
    row. The loads and distributions of a failed angle are NaN (see
    :func:`row_from_result`). The row holds the work counters, and its
    reason names each early stop of a failed angle. ``cache`` (a dict, one
    per sweep) keeps the cold start at the zero-lift angle for the next
    angles (see :func:`_cold_path`).
    """
    ctl = _AngleWork(alpha, StopRules() if rules is None else rules)
    if first is not None:
        ctl.add(first[0], first[1], ctl.cap(max_iter), max_iter)
    try:
        if prev is not None:
            setup, res = _warm_path(body, alpha, re, prev, tol, max_iter, kw, 0, bk, first, ctl=ctl)
            if not res.converged:
                cold_setup, cold = _cold_path(body, alpha, re, tol, max_iter, kw, bk, ctl=ctl, cache=cache)
                if cold.converged or cold.residual < res.residual:
                    setup, res = cold_setup, cold
        else:
            setup, res = _cold_path(body, alpha, re, tol, max_iter, kw, bk, ctl=ctl, cache=cache)
    except (_BudgetUsedUp, *vii.linalg_errors((bk or _NUMPY)["xp"])) as exc:
        if isinstance(exc, _BudgetUsedUp):
            ctl.note(f"work budget of {ctl.rules.work_budget} Newton iterations per angle used up")
        else:
            ctl.note(f"singular: a linear system of the angle setup is singular ({exc})")
        if ctl.best is None:
            row = _failed_row(alpha, body, "; ".join(ctl.notes + ctl.retry_notes) + " before a solve at this angle")
            row.update(ctl.counters())
            return row, None
        setup, res = ctl.best
    except cp.WakeGroundError as exc:
        row = _failed_row(alpha, body, str(exc))
        row.update(ctl.counters())
        return row, None
    row = row_from_result(setup, res, alpha)
    row.update(ctl.counters())
    retry_notes = [n for n in ctl.retry_notes if res.reason not in n or not res._first_stop]
    if retry_notes:
        row["reason"] += "; " + "; ".join(retry_notes)
    if not res.converged and ctl.notes:
        notes = [n for n in ctl.notes if n not in row["reason"]]
        if notes:
            row["reason"] += "; " + "; ".join(notes)
    return row, (setup, res.state)


def zero_lift_angle(body, kw) -> float:
    """Return the inviscid zero-lift angle of a section [deg] (free air or ground effect).

    Parameters
    ----------
    body : np.ndarray
        Airfoil coordinates [m].
    kw : dict
        Case settings (height, height reference, reference chord and
        pivot are used).

    Returns
    -------
    float
        Angle where the inviscid lift is zero [deg], from two inviscid
        solves (0 and 4 deg).
    """
    from ventorum.panel2d.solver import solve_inviscid

    h = kw.get("height")
    opts = {} if h is None else {"height": h, "height_reference": kw.get("height_reference", "pivot"),
                                 "pivot": kw.get("pivot")}
    cl = solve_inviscid(body, [0.0, 4.0], chord_ref=kw.get("chord_ref"), **opts).Cl
    return float(-4.0 * cl[0] / (cl[1] - cl[0]))


def _cold_path(body, alpha, re, tol, max_iter, kw, bk=None, ctl=None, cache=None):
    """Solve an angle without a converged neighbour.

    A cold start (first estimate by direct marches) is reliable near the
    zero-lift angle, where the edge velocity has no strong suction peak.
    When the angle is more than CONTINUATION_STEP from the inviscid
    zero-lift angle, the solver starts cold at the zero-lift angle and
    continues to the angle in steps of at most CONTINUATION_STEP (each with
    the halving of :func:`_warm_path`; both tests have the round-off
    tolerance ANGLE_MATCH_DEG, see :func:`_continuation_steps`); the intermediate solutions serve as
    start values only. When that start fails, a cold start at the angle
    itself is the last attempt. ``cache`` (a dict per sweep) keeps the cold
    start at the zero-lift angle, which is the same for every angle of a
    sweep; a later angle reuses it instead of solving it again.

    Work budgets (with ``ctl``): the cold start at the zero-lift angle has
    the budget ``cold_start_budget`` and each continuation step the budget
    ``step_budget`` of the stop rules; the solves at the angle itself use
    the budget of the angle. A stage of the path that fails (budget used
    up, or no convergence) is named in a note of the angle, and the cold
    start at the angle follows.
    """
    a0 = zero_lift_angle(body, kw)
    n = _continuation_steps(alpha, a0)
    if n > 0:
        if ctl is not None:
            # No budget left at the angle: the continuation path cannot end in a solve at the angle.
            ctl.check(max_iter, alpha)
        rules = StopRules() if ctl is None else ctl.rules
        cold_name = f"cold start at the zero-lift angle {a0:.4g} deg"
        if cache is not None and a0 in cache:
            s0, r0 = cache[a0]
        else:
            s0, r0 = None, None
            stage = _enter_stage(ctl, cold_name, "cold-start", rules.cold_start_budget)
            try:
                s0 = prepare_case(body, a0, re, **kw)
                r0 = _cold_solve(s0, tol, max_iter, bk, ctl, retry=False)
            except cp.WakeGroundError:
                s0, r0 = None, None
            except _StageBudgetUsedUp:
                r0 = None
                _path_note(ctl, f"{cold_name} stopped: {stage.label()} used up")
            finally:
                _leave_stage(ctl)
            # The cold start at the zero-lift angle has its own budget: it is the same for every angle.
            if cache is not None:
                cache[a0] = (s0, r0)
        if r0 is not None and not r0.converged:
            _path_note(ctl, f"{cold_name} did not converge: {r0.reason}")
        if r0 is not None and r0.converged:
            prev = (s0, r0.state)
            setup, res = s0, r0
            for k in range(1, n + 1):
                a = a0 + (alpha - a0) * k / n
                step_name = f"continuation step {k} of {n}, from {prev[0].sys.alpha:.4g} deg to {a:.4g} deg"
                stage = _enter_stage(ctl, step_name, "continuation-step", rules.step_budget)
                try:
                    setup, res = _warm_path(body, a, re, prev, tol, max_iter, kw, 0, bk, ctl=ctl)
                except _StageBudgetUsedUp:
                    res = None
                finally:
                    _leave_stage(ctl)
                if res is None or not res.converged:
                    if stage is not None and ctl.rules.work_budget > 0 and 0 < stage.budget <= stage.iterations:
                        _path_note(ctl, f"{step_name} stopped: {stage.label()} used up")
                    else:
                        _path_note(ctl, f"{step_name} did not converge: {res.reason}")
                    break
                prev = (setup, res.state)
            if res is not None and res.converged:
                return setup, res
    setup = prepare_case(body, alpha, re, **kw)
    return setup, _cold_solve(setup, tol, max_iter, bk, ctl)


def _continuation_steps(alpha, a0) -> int:
    """Return the number of continuation steps from the zero-lift angle to an angle (0: no continuation path).

    The path is used when the angle is more than CONTINUATION_STEP from the zero-lift angle, and its
    steps are at most CONTINUATION_STEP long. Both tests use the round-off tolerance ANGLE_MATCH_DEG, so
    an angle a whole number of steps from the zero-lift angle gives the same steps on every platform
    (for a symmetric section the zero-lift angle is 0 deg only up to a round-off of either sign).

    Parameters
    ----------
    alpha : float
        Angle of attack [deg].
    a0 : float
        Inviscid zero-lift angle [deg] (see :func:`zero_lift_angle`).

    Returns
    -------
    int
        Number of continuation steps, 0 when the angle is within CONTINUATION_STEP (plus the tolerance)
        of the zero-lift angle.
    """
    distance = abs(float(alpha) - float(a0))
    if distance <= CONTINUATION_STEP + ANGLE_MATCH_DEG:
        return 0
    return int(np.ceil((distance - ANGLE_MATCH_DEG) / CONTINUATION_STEP))


def _enter_stage(ctl, name: str, kind: str, budget: int):
    """Start a stage of the continuation path in the work record ``ctl``; return the stage (None without ctl)."""
    if ctl is None:
        return None
    ctl.stage = _Stage(name, kind, budget)
    return ctl.stage


def _leave_stage(ctl):
    """End the stage of the continuation path in the work record ``ctl``."""
    if ctl is not None:
        ctl.stage = None


def _path_note(ctl, text: str):
    """Record the failed stage of the continuation path in the notes of the angle."""
    if ctl is not None:
        ctl.note("continuation path: " + text)


def _solve_angle(body, alpha, re, prev, tol, max_iter, kw, rules=None, cache=None):
    """Solve one angle with the numpy backend (see :func:`finish_angle`)."""
    return finish_angle(body, alpha, re, prev, tol, max_iter, kw, rules=rules, cache=cache)


def row_from_result(setup, res, alpha) -> dict:
    """Return the output row of a Newton result.

    The last iterate of a result that did not converge is not a solution:
    its loads (Cl, Cl_circulation, Cl_pressure, Cd, Cd_friction, Cd_pressure, Cm), its
    transition positions and its node and wake distributions are NaN. The
    convergence flag, the reason, the residual and the iteration count
    stay.
    """
    if not res.converged:
        row = _failed_row(alpha, setup.sys.nodes, res.reason, wake_x=setup.sys.wake_nodes)
        row.update(iterations=int(res.iterations), residual=float(res.residual))
        return row
    o = vii.case_outputs(setup, res.state, res.ue)
    return {
        "alpha": float(alpha),
        "Cl": o["Cl_pressure"] if setup.sys.ground else o["Cl"],
        "Cl_circulation": o["Cl"],
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


def _failed_row(alpha, body, reason, wake_x=None) -> dict:
    """Return the output row of a failed angle: NaN loads and distributions.

    ``wake_x`` holds the wake line [m] when the case was set up; the wake
    distributions are then NaN arrays, else the wake dict is empty.
    """
    nan = float("nan")
    coords = body.coords if hasattr(body, "coords") else body
    n = len(coords)

    def arr(k=n):
        return np.full(k, nan)

    wake = {} if wake_x is None else {"x": wake_x, "theta": arr(len(wake_x)), "H": arr(len(wake_x)),
                                      "ue": arr(len(wake_x))}
    return {
        "alpha": float(alpha), "Cl": nan, "Cl_circulation": nan, "Cd": nan, "Cd_friction": nan,
        "Cd_pressure": nan, "Cm": nan,
        "transition_positions": (nan, nan), "separation_flags": (False, False), "Cp": arr(), "H": arr(),
        "theta": arr(), "Cf": arr(), "iterations": 0, "residual": nan, "converged": False,
        "coordinates": coords, "reason": reason, "Cl_pressure": nan, "delta_star": arr(), "ue": arr(), "wake": wake,
        "newton_iterations": 0, "linear_solves": 0, "newton_solves": 0,
    }


def solve_viscous_flapped(
    shape,
    alpha_deg,
    Re: float,
    *,
    x_hinge: float,
    deflection_deg: float,
    y_hinge: float | None = None,
    step_deg: float = 2.5,
    mach: float = 0.0,
    **options,
) -> ViscousResult:
    """Solve 2D viscous flow of a flapped airfoil by deflection continuation.

    The solver steps the flap deflection from 0 deg to ``deflection_deg``
    in steps of at most ``step_deg`` [deg]. Each step starts from the
    converged state of the previous deflection (warm start). A step that
    fails is halved (down to 0.25 deg). When that also fails, the solver
    returns a result with ``converged=False`` and the reached deflection
    in its details (loads set to NaN).

    Parameters
    ----------
    shape : AirfoilShape or array_like
        Airfoil section coordinates [m or chord units] in Selig order, or
        an :class:`ventorum.airfoil.AirfoilShape`.
    alpha_deg : float or array_like
        Angle of attack [deg].
    Re : float
        Reynolds number of the reference chord [-].
    x_hinge : float
        Hinge chord station [chord units], in (0.05, 0.98).
    deflection_deg : float
        Final flap deflection [deg], positive when the trailing edge moves
        down.
    y_hinge : float or None
        Hinge height as a fraction [-] of local thickness at ``x_hinge``,
        from lower (0) to upper (1) surface. Default None selects 0.5
        (mid-thickness).
    step_deg : float
        Largest deflection step of the continuation [deg], positive.
        Default 2.5.
    mach : float
        Free-stream Mach number [-]; only 0 is supported.
    **options
        Options passed to :func:`solve_viscous` (such as ``tol``,
        ``max_iter``, ``ncrit``, ``xtr``, ``height``, ``chord_ref``,
        ``moment_ref``, etc.).
        ``work_budget`` (int, default :data:`WORK_BUDGET`) sets the
        iteration budget per continuation step [-]. The whole continuation
        has a total cap of 20 times the per-step budget.

    Returns
    -------
    ViscousResult
        Loads, distributions, convergence status and work counters at the
        final deflection.
    """
    if abs(mach) > 1e-12:
        raise ValueError(f"Only mach=0.0 is supported in solve_viscous_flapped, got mach={mach}.")
    if not np.isfinite(Re) or Re <= 0.0:
        raise ValueError(f"Reynolds number must be positive, got {Re}.")
    if not np.isfinite(step_deg) or step_deg <= 0.0:
        raise ValueError(f"step_deg must be positive, got {step_deg}.")

    hinge_treatment = options.pop("hinge_treatment", "repanel")
    n_arc = options.pop("n_arc", None)
    n_panels = options.pop("n_panels", None)

    if abs(deflection_deg) < 1e-12:
        res = solve_viscous(shape, alpha_deg, Re, mach=mach, **options)  # it gives the long-wake warning
        is_scalar = np.ndim(alpha_deg) == 0
        if is_scalar:
            res.details = {
                "reached_deflection": 0.0,
                "step_history": [(0.0, bool(res.converged), int(res.iterations))],
            }
        else:
            res.details = [
                {
                    "reached_deflection": 0.0,
                    "step_history": [(0.0, bool(c), int(it))],
                }
                for c, it in zip(res.converged, res.iterations)
            ]
        return res

    if isinstance(shape, AirfoilShape):
        base_shape = shape
    else:
        arr = np.asarray(shape, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"shape must be an (N, 2) array or AirfoilShape, got shape {arr.shape}.")
        base_shape = AirfoilShape(coords=arr)

    chord_ref_opt = options.pop("chord_ref", None)
    moment_ref_opt = options.pop("moment_ref", None)
    _, chord_ref, moment_ref, pivot = resolve_body(base_shape, chord_ref_opt, moment_ref_opt)

    # Default ncrit 9 and wake length 1 c: the origins are in solve_viscous.
    ncrit = options.pop("ncrit", 9.0)
    xtr = options.pop("xtr", (1.0, 1.0))
    height = options.pop("height", None)
    max_iter = options.pop("max_iter", 50)
    tol = options.pop("tol", 1e-6)
    wake_length = options.pop("wake_length", 1.0)
    n_wake = options.pop("n_wake", 30)
    height_reference = options.pop("height_reference", "pivot")
    backend = options.pop("backend", "numpy")
    stall_window = options.pop("stall_window", vii.STALL_WINDOW)
    stall_factor = options.pop("stall_factor", vii.STALL_FACTOR)
    divergence_factor = options.pop("divergence_factor", vii.DIVERGENCE_FACTOR)
    work_budget = options.pop("work_budget", WORK_BUDGET)
    lm_trials = options.pop("lm_trials", vii.LM_RETRY_TRIALS)

    if backend != "numpy":
        raise ValueError(f"backend '{backend}' is not supported by solve_viscous_flapped; use 'numpy'.")
    if options:
        raise TypeError(f"solve_viscous_flapped got unexpected keyword argument(s): {list(options.keys())}")

    rules = StopRules(stall_window=stall_window, stall_factor=stall_factor, divergence_factor=divergence_factor,
                      work_budget=work_budget, lm_trials=lm_trials)
    wake_note = warn_long_wake(wake_length, "solve_viscous_flapped")
    # Work cap of the whole deflection path: 20 angle budgets. A code
    # setting, not from a source; no measurement in the code or its tests
    # supports the factor 20.
    total_cap = 20 * work_budget if work_budget > 0 else 0
    kw = dict(ncrit=ncrit, xtr=xtr, height=height, height_reference=height_reference,
              wake_length=wake_length, n_wake=n_wake, chord_ref=chord_ref, moment_ref=moment_ref, pivot=pivot)

    yh = 0.5 if y_hinge is None else float(y_hinge)

    def _deflect(d: float) -> AirfoilShape:
        if abs(d) < 1e-12:
            return base_shape
        return base_shape.with_flap(
            x_hinge=x_hinge,
            deflection=d,
            y_hinge=yh,
            hinge_treatment=hinge_treatment,
            n_arc=n_arc,
            n_panels=n_panels,
        )

    def _continue_deflection(alpha: float) -> tuple[dict, tuple | None]:
        total_iterations = 0
        total_linear_solves = 0
        total_solves = 0
        step_history: list[tuple[float, bool, int]] = []

        step0_budget = work_budget if work_budget > 0 else 0
        if total_cap > 0 and step0_budget > 0:
            step0_budget = min(step0_budget, total_cap)
        step0_rules = StopRules(
            stall_window=rules.stall_window,
            stall_factor=rules.stall_factor,
            divergence_factor=rules.divergence_factor,
            work_budget=step0_budget,
            lm_trials=rules.lm_trials,
        )
        try:
            row0, state0 = finish_angle(base_shape, alpha, Re, None, tol, max_iter, kw, rules=step0_rules)
        except (_BudgetUsedUp, *vii.linalg_errors(np), cp.WakeGroundError, ValueError) as exc:
            if isinstance(exc, _BudgetUsedUp):
                reason = "work budget used up at 0 deg (reached 0 deg)"
            else:
                reason = f"continuation failed at deflection 0 deg (reached 0 deg): {exc}"
            row = _failed_row(alpha, base_shape.coords, reason)
            row["details"] = {
                "reached_deflection": 0.0,
                "step_history": [(0.0, False, 0)],
            }
            return row, None

        iters0 = int(row0["newton_iterations"])
        total_iterations += iters0
        total_linear_solves += int(row0["linear_solves"])
        total_solves += int(row0["newton_solves"])
        step_history.append((0.0, bool(row0["converged"]), iters0))

        if not row0["converged"]:
            reason_0 = row0["reason"]
            if "work budget" in reason_0 or row0.get("reason") == "budget":
                row0["reason"] = "work budget used up at 0 deg (reached 0 deg)"
            else:
                row0["reason"] = f"continuation did not converge at deflection 0 deg (reached 0 deg): {reason_0}"
            row0["details"] = {
                "reached_deflection": 0.0,
                "step_history": step_history,
            }
            return row0, None

        target_d = float(deflection_deg)
        step_max = float(step_deg)
        # Smallest deflection step [deg] of the halving. A code setting, not
        # from a source; no measurement in the code or its tests supports it.
        min_step = 0.25
        sgn = 1.0 if target_d > 0.0 else -1.0

        current_d = 0.0
        prev = state0
        current_step = step_max
        last_setup = prev[0]
        last_res = None

        while abs(target_d - current_d) > 1e-9:
            step_to_try = min(current_step, abs(target_d - current_d))
            next_d = current_d + sgn * step_to_try

            if total_cap > 0 and total_iterations >= total_cap:
                reason = f"work budget used up at {current_d:g} deg (reached {current_d:g} deg)"
                nodes = last_setup.sys.nodes if last_setup is not None and hasattr(last_setup, "sys") else base_shape.coords
                wake_nodes = last_setup.sys.wake_nodes if last_setup is not None and hasattr(last_setup, "sys") else None
                row = _failed_row(alpha, nodes, reason, wake_x=wake_nodes)
                row["newton_iterations"] = total_iterations
                row["linear_solves"] = total_linear_solves
                row["newton_solves"] = total_solves
                row["details"] = {
                    "reached_deflection": float(current_d),
                    "step_history": step_history,
                }
                return row, None

            step_budget = work_budget if work_budget > 0 else 0
            if total_cap > 0 and step_budget > 0:
                step_budget = min(step_budget, total_cap - total_iterations)

            step_rules = StopRules(
                stall_window=rules.stall_window,
                stall_factor=rules.stall_factor,
                divergence_factor=rules.divergence_factor,
                work_budget=step_budget,
                lm_trials=rules.lm_trials,
            )
            step_ctl = _AngleWork(alpha, step_rules)

            setup = None
            try:
                shape_next = _deflect(next_d)
                setup = prepare_case(shape_next, alpha, Re, **kw)
                if prev is not None and setup.n_st == prev[0].n_st and len(setup.sys.nodes) == len(prev[0].sys.nodes):
                    start = warm_state(setup, prev[0], prev[1])
                    res = _solve_one(setup, start, tol, max_iter, 1.0, bk=_NUMPY, ctl=step_ctl)
                else:
                    res = _cold_solve(setup, tol, max_iter, bk=_NUMPY, ctl=step_ctl)
            except _BudgetUsedUp:
                res = vii.NewtonResult(
                    None, None, False, step_ctl.iterations, float("nan"),
                    f"work budget used up at {next_d:g} deg", step_ctl.linear_solves, "budget"
                )
            except (*vii.linalg_errors(np), cp.WakeGroundError, ValueError) as exc:
                res = vii.NewtonResult(
                    None, None, False, step_ctl.iterations, float("nan"),
                    f"exception during solve ({exc})", step_ctl.linear_solves, "error"
                )

            step_iters = int(res.iterations) if res is not None else step_ctl.iterations
            total_iterations += step_iters
            total_linear_solves += int(res.linear_solves) if res is not None else step_ctl.linear_solves
            total_solves += 1
            step_history.append((float(next_d), bool(res.converged) if res is not None else False, step_iters))

            if res is not None and res.converged:
                current_d = next_d
                prev = (setup, res.state)
                last_setup = setup
                last_res = res
                current_step = step_max
            else:
                half = step_to_try * 0.5
                if half >= min_step - 1e-9:
                    current_step = half
                elif step_to_try > min_step + 1e-9:
                    current_step = min_step
                else:
                    res_reason = res.reason if res is not None else "step failed"
                    if res is not None and (res.status == "budget" or "work budget" in res_reason):
                        reason = (
                            f"continuation did not converge at deflection {next_d:g} deg "
                            f"(reached {current_d:g} deg): work budget used up at {next_d:g} deg"
                        )
                    else:
                        reason = (
                            f"continuation did not converge at deflection {next_d:g} deg "
                            f"(reached {current_d:g} deg): {res_reason}"
                        )
                    wake_nodes = setup.sys.wake_nodes if setup is not None and hasattr(setup, "sys") else None
                    nodes = setup.sys.nodes if setup is not None and hasattr(setup, "sys") else base_shape.coords
                    row = _failed_row(alpha, nodes, reason, wake_x=wake_nodes)
                    row["newton_iterations"] = total_iterations
                    row["linear_solves"] = total_linear_solves
                    row["newton_solves"] = total_solves
                    row["residual"] = float(res.residual) if res is not None else float("nan")
                    row["details"] = {
                        "reached_deflection": float(current_d),
                        "step_history": step_history,
                    }
                    return row, None

        row = row_from_result(last_setup, last_res, alpha)
        row["newton_iterations"] = total_iterations
        row["linear_solves"] = total_linear_solves
        row["newton_solves"] = total_solves
        row["iterations"] = int(last_res.iterations)
        row["residual"] = float(last_res.residual)
        row["converged"] = True
        row["details"] = {
            "reached_deflection": float(current_d),
            "step_history": step_history,
        }
        return row, (last_setup, last_res.state)

    is_scalar = np.ndim(alpha_deg) == 0
    alphas = np.atleast_1d(np.asarray(alpha_deg, dtype=np.float64))
    order_up, order_dn = sweep_order(alphas)
    out: dict[float, dict] = {}

    with blas_single_thread():
        first_state = None
        prev = None
        target_shape = _deflect(float(deflection_deg))
        for a in order_up:
            if prev is not None:
                row, state = finish_angle(target_shape, a, Re, prev, tol, max_iter, kw, rules=rules)
                if not row["converged"]:
                    row, state = _continue_deflection(a)
                else:
                    row["details"] = {
                        "reached_deflection": float(deflection_deg),
                        "step_history": [(float(deflection_deg), True, int(row["iterations"]))],
                    }
            else:
                row, state = _continue_deflection(a)
            out[a] = row
            if row["converged"]:
                prev = state
                if first_state is None:
                    first_state = state

        prev = first_state
        for a in order_dn:
            if prev is not None:
                row, state = finish_angle(target_shape, a, Re, prev, tol, max_iter, kw, rules=rules)
                if not row["converged"]:
                    row, state = _continue_deflection(a)
                else:
                    row["details"] = {
                        "reached_deflection": float(deflection_deg),
                        "step_history": [(float(deflection_deg), True, int(row["iterations"]))],
                    }
            else:
                row, state = _continue_deflection(a)
            out[a] = row
            if row["converged"]:
                prev = state

    rows = [out[float(a)] for a in alphas]
    add_note(rows, wake_note)
    bad = [r for r in rows if not r["converged"]]
    for r in bad:
        warnings.warn(f"solve_viscous_flapped: alpha = {r['alpha']:g} deg did not converge: {r['reason']}",
                      UserWarning, stacklevel=2)
    if is_scalar:
        r = rows[0]
        return ViscousResult(**{k: r[k] for k in _FIELDS}, details=r.get("details", {}))
    return ViscousResult(
        alpha=alphas,
        Cl=np.array([r["Cl"] for r in rows]),
        Cl_circulation=np.array([r["Cl_circulation"] for r in rows]),
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
        newton_iterations=np.array([r["newton_iterations"] for r in rows], dtype=int),
        linear_solves=np.array([r["linear_solves"] for r in rows], dtype=int),
        newton_solves=np.array([r["newton_solves"] for r in rows], dtype=int),
        details=[r.get("details", {}) for r in rows],
    )
