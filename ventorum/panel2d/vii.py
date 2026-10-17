# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Fully simultaneous viscous-inviscid Newton solver of Ventorum-2D.

One global Newton system holds the boundary-layer variables of the upper
surface, the lower surface and the wake: the momentum thickness theta, the
mass defect m = ue delta_star and the third variable c (the amplification
exponent N in laminar flow, sqrt(Ctau) in turbulent flow and in the wake).
The edge velocity is eliminated through the mass-defect influence of the
panel method (fully simultaneous coupling; Drela and Giles 1987; Drela
1989)::

    ue = ue_inv + D m,    delta_star = m / ue.

D maps the station mass defects to the station edge velocities through the
wall and wake transpiration sources (sigma = dm/dxi on each source panel;
Lighthill 1958); the factorization of the inviscid matrix is reused (see
:mod:`ventorum.panel2d.coupling`). The station equations are those of
:mod:`ventorum.ibl2d.equations`, the same as in the direct march, with their
exact Jacobians; the chain rule through ue and delta_star gives the exact
Jacobian of the coupled system.

Structure updates between Newton iterations (documented in
``docs/theory/panel2d_viscous.md``):

* Stagnation point: between the two nodes where the total gamma changes
  sign, by linear interpolation; it follows alpha and the viscous solution.
  The first station of each side holds the stagnation-point similarity
  equations with the velocity gradient a = (ue_upper + ue_lower) / ds of the
  stagnation panel. A node closer to the stagnation point than
  STAGNATION_DROP times its next interval is not a station (its mass defect
  is zero); the side then starts at the next node.
* Transition: the transition interval of each side is the first interval
  where N reaches ncrit (or the forced position); when it moves, the third
  variable of the stations that change regime is set from the state
  (sqrt(TRANSITION_CTAU_FACTOR Ctau_eq) or the integrated N).

Newton step control (path only, not the converged solution): the step is
limited so that theta, m and c fall by at most 50 % and rise by at most
200 %, H - 1 falls by at most 50 % and ue stays positive; a backtracking
line search then halves the step until the 2-norm of the residual
decreases (Armijo condition with coefficient 1e-4). When the accepted Newton step factor is below LM_SWITCH
(a near-singular Jacobian, for example near laminar separation), a
Levenberg-Marquardt step with adaptive damping is taken instead (Levenberg
1944; Marquardt 1963). A first estimate far from the solution can give a
negative edge velocity; a cold start therefore begins with the coupling
term D m scaled by omega = 1/64 and doubles omega whenever the largest
residual is below OMEGA_STEP_RESIDUAL, until omega is 1 (continuation in
the coupling strength). Converged means omega = 1, the
largest absolute residual of all equations below ``tol`` and no structure
change in the last two evaluations. The residuals are relative (logarithmic
differences and ratios), except the amplification equation, which is
absolute in N.

Early stops (fail fast; they change the work of a failed solve, not a
converged solution): a case stops with status "stagnation" when its
smallest residual has not fallen by the factor STALL_FACTOR in the last
STALL_WINDOW iterations (the window starts again when omega changes and,
at omega = 1, when the structure changes), or when, at omega = 1, the line
search and the Levenberg-Marquardt fallback both fail twice on the same
state and structure (the iteration would then repeat itself exactly);
both stagnation stops apply only while the smallest residual is above
STALL_FLOOR times the tolerance (near the tolerance the residual can stall
at the round-off level of float32 and still reach it); it
stops with status "divergence" when, at omega = 1, the residual exceeds
DIVERGENCE_FACTOR times its smallest earlier value. After a
Levenberg-Marquardt step of a solve found no decrease while the residual
was above STALL_FLOOR times the tolerance, the later ones of that solve
try at most ``lm_trials`` damping values instead of LM_TRIALS.
A singular Newton matrix is a failed step (see :func:`newton_solve`).

The same code runs with numpy (one case) and with torch (a batch of cases
with the same node and wake counts, on the CPU or on CUDA).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ventorum.ibl2d import closure as cl
from ventorum.ibl2d import equations as eq
from ventorum.ibl2d import march as mr
from ventorum.ibl2d import transition as tr
from ventorum.ibl2d.dual import variables
from ventorum.panel2d.coupling import CouplingSystem

__all__ = ["CaseSetup", "CaseState", "NewtonResult", "newton_solve", "initial_state", "case_outputs"]

#: A node closer to the stagnation point than this fraction of its next interval is not a station [-].
STAGNATION_DROP = 0.1
#: Largest residual at which the coupling factor omega of a cold start is doubled (path only) [-].
OMEGA_STEP_RESIDUAL = 0.1
#: Iterations without halving the residual after which omega is doubled anyway (path only).
OMEGA_STALL = 4
#: Largest residual at which the transition intervals are updated (path only) [-].
STRUCTURE_RESIDUAL = 1e-3
#: Iterations without halving the residual after which the structure is updated anyway (path only).
STALL_ITERATIONS = 8
TRANSITION_EVERY_ITERATION = True
#: Newton step factor below which a Levenberg-Marquardt step is tried instead (path only) [-].
LM_SWITCH = 1e-2
#: Largest number of damping values (trials, one linear solve each) of one Levenberg-Marquardt step [-].
LM_TRIALS = 12
#: Default trials of a Levenberg-Marquardt step after a step of the same Newton solve found no decrease [-].
LM_RETRY_TRIALS = 2
#: Number of quasi-simultaneous sweeps of the second first estimate of a cold start (path only).
QS_SWEEPS = 3
#: Window of the stagnation stop [Newton iterations] (0 switches the stagnation stops off).
STALL_WINDOW = 12
#: Factor by which the smallest residual must fall within STALL_WINDOW iterations [-].
STALL_FACTOR = 0.5
#: The stagnation stops apply only while the smallest residual is above this factor times tol [-].
STALL_FLOOR = 1e3
#: Growth of the residual over its smallest value at omega = 1 that stops a solve as divergent [-] (0: off).
DIVERGENCE_FACTOR = 1e4


@dataclass
class CaseSetup:
    """Fixed data of one coupled case (host side, numpy).

    Attributes
    ----------
    sys : CouplingSystem
        Inviscid data.
    re : float
        Reynolds number of the reference chord [-].
    ncrit : float
        Critical amplification exponent [-].
    xtr : tuple of float
        Forced transition x/c on the upper and lower side [-]; values of 1
        or more mean free transition.
    """

    sys: CouplingSystem
    re: float
    ncrit: float
    xtr: tuple

    def __post_init__(self):
        s = self.sys
        self.n = s.n_nodes
        self.nw = s.n_wake
        self.n_st = self.n + self.nw
        self.n_src = s.gamma_src.shape[1]
        self.sens = np.concatenate([s.gamma_src, s.q_src], 0)
        self.base = np.concatenate([s.gamma_inv, s.q_inv])
        self.re_l = self.re / s.chord_ref
        P = np.zeros((self.n_st, self.n + self.nw - 1))
        n, nw = self.n, self.nw
        P[n, 0] = 1.0
        for j in range(1, nw - 1):
            P[n + j, n + j - 1] = 0.5
            P[n + j, n + j] = 0.5
        P[n + nw - 1, n + nw - 2] = 1.5
        P[n + nw - 1, n + nw - 3] = -0.5
        self._p_wake = P


@dataclass
class CaseState:
    """Variables and structure of one case.

    Attributes
    ----------
    x : np.ndarray
        Station variables (theta [m], m [m], c [-]), shape (n_st, 3).
    i_s : int
        Stagnation panel (upper node i_s, lower node i_s + 1).
    drop : int
        Node at the stagnation point that is not a station, or -1.
    itr : list of int
        Position (in the side list) of the last laminar station, per side.
    panel : int
        Panel that holds the stagnation point (gamma changes sign on it).
    frozen : list of bool
        Sides whose transition interval no longer moves (see
        :func:`newton_solve`).
    """

    x: np.ndarray
    i_s: int
    drop: int = -1
    itr: list = field(default_factory=lambda: [0, 0])
    panel: int = -1
    frozen: list = field(default_factory=lambda: [False, False])

    def copy(self):
        """Return a deep copy."""
        return CaseState(self.x.copy(), self.i_s, self.drop, list(self.itr), self.panel, list(self.frozen))


@dataclass
class NewtonResult:
    """Outcome of a Newton solve of one case.

    Attributes
    ----------
    state : CaseState
        Final state.
    ue : np.ndarray
        Station edge velocities of the final state [-].
    converged : bool
        True when the convergence test is met.
    iterations : int
        Newton iterations (evaluations of the residual and the Jacobian
        after the first one).
    residual : float
        Largest absolute residual of the final iterate [-].
    reason : str
        Convergence status or failure reason.
    linear_solves : int
        Solves of the Newton or Levenberg-Marquardt linear system (size
        three times the station count).
    status : str
        "converged", "stagnation", "divergence", "not_finite", "singular"
        or "max_iter".
    """

    state: CaseState
    ue: np.ndarray
    converged: bool
    iterations: int
    residual: float
    reason: str
    linear_solves: int = 0
    status: str = ""


# ---------------------------------------------------------------------------
# Structure helpers
# ---------------------------------------------------------------------------


def _sides(setup: CaseSetup, st: CaseState):
    """Return the station lists (node indices) of the upper and lower sides."""
    up = np.arange(st.i_s, -1, -1)
    lo = np.arange(st.i_s + 1, setup.n)
    if st.drop >= 0:
        up = up[up != st.drop]
        lo = lo[lo != st.drop]
    return up, lo


def _signs(setup: CaseSetup, st: CaseState):
    """Return the edge-velocity sign of each node (+1 upper, -1 lower, 0 dropped)."""
    sg = np.ones(setup.n)
    sg[st.i_s + 1:] = -1.0
    if st.drop >= 0:
        sg[st.drop] = 0.0
    return sg


def _station_mask(setup: CaseSetup, st: CaseState):
    """Return a mask of the stations that carry boundary-layer variables."""
    m = np.ones(setup.n_st, dtype=bool)
    if st.drop >= 0:
        m[st.drop] = False
    return m


def _dstar_mask(setup: CaseSetup, st: CaseState):
    """Return a mask of the stations whose second variable is delta_star.

    These are the first station of each side: next to the stagnation point
    ue can be very small, and delta_star = m / ue would be ill-conditioned.
    All other stations use the mass defect m.
    """
    up, lo = _sides(setup, st)
    m = np.zeros(setup.n_st, dtype=bool)
    m[up[0]] = True
    m[lo[0]] = True
    return m


def host_edge(setup: CaseSetup, st: CaseState, omega: float = 1.0):
    """Return (ue, delta_star, m) at the stations of a state (numpy).

    ue solves (I - omega D diag(delta_star_S)) ue = ue_inv + omega D m_N,
    where S are the delta_star stations and N the mass-defect stations.
    """
    ue_inv, D = _coupling_matrices(setup, st)
    S = _dstar_mask(setup, st)
    x2 = st.x[:, 1]
    A = np.eye(setup.n_st) - omega * D * np.where(S, x2, 0.0)[None, :]
    ue = np.linalg.solve(A, ue_inv + omega * (D @ np.where(S, 0.0, x2)))
    mask = _station_mask(setup, st)
    safe = np.where(mask, ue, 1.0)
    ds = np.where(S, x2, x2 / safe)
    m = np.where(S, ue * x2, x2)
    return ue, ds, m


def _source_matrix(setup: CaseSetup, st: CaseState):
    """Return the source strengths per unit station mass defect, shape (n_src, n_st).

    On an airfoil panel sigma L = (m where the layer leaves the panel) - (m
    where it enters); on the stagnation panel both ends are exits. The
    trailing-edge gap panel carries no viscous source: the gap thickness of
    the wake start is the displacement of its inviscid gap source. On a wake
    panel sigma L = m_downstream - m_upstream.
    """
    n, nw = setup.n, setup.nw
    s = setup.sys
    sg = _signs(setup, st)
    lay = s.layout
    B = np.zeros((setup.n_src, setup.n_st))
    for p in range(lay.n_panels):
        if p in lay.te_panel:
            continue
        i1, i2 = lay.idx1[p], lay.idx2[p]
        L = s.panel_len[p]
        if sg[i1] != 0.0:
            B[p, i1] = (1.0 if sg[i1] > 0.0 else -1.0) / L
        if sg[i2] != 0.0:
            B[p, i2] = (-1.0 if sg[i2] > 0.0 else 1.0) / L
    npn = lay.n_panels
    for j in range(nw - 1):
        B[npn + j, n + j + 1] = 1.0 / s.wake_len[j]
        B[npn + j, n + j] = -1.0 / s.wake_len[j]
    return B


def _coupling_matrices(setup: CaseSetup, st: CaseState):
    """Return (ue_inv, D) at the stations for the current structure.

    Station edge velocities: sign * gamma at the airfoil nodes, gamma of
    the upper trailing-edge node at the first wake station (equal to the
    lower one by the Kutta condition), the mean of the two adjacent
    wake-panel midpoint velocities at the other wake stations and a linear
    extrapolation at the last one.
    """
    P, B = _p_and_b(setup, st)
    return P @ setup.base, P @ setup.sens @ B


def _p_and_b(setup: CaseSetup, st: CaseState):
    """Return the station-velocity map P and the source matrix B of a structure."""
    P = setup._p_wake.copy()
    n = setup.n
    P[np.arange(n), np.arange(n)] = _signs(setup, st)
    return P, _source_matrix(setup, st)


def _laminar_rate(setup, th, ds, ue):
    """Return the laminar growth rate dN/dxi of states (values)."""
    return tr.transition_growth_rate(ds / th, th, ue * th * setup.re_l)[0]


def _ctau_start(setup, th, ds, ue):
    """Return sqrt(TRANSITION_CTAU_FACTOR Ctau_eq) of states (values)."""
    th = np.atleast_1d(np.asarray(th, dtype=float))
    return mr._turbulent_ceq_sqrt(np, th, np.atleast_1d(ds), np.atleast_1d(ue), np.full(th.size, setup.re_l))


def _side_geometry(setup: CaseSetup, nodes_idx):
    """Return the interval lengths and the x/c of a side."""
    s = setup.sys.s
    return np.abs(np.diff(s[nodes_idx])), setup.sys.x_over_c[nodes_idx]


def _forced_interval(setup: CaseSetup, side: int, nodes_idx):
    """Return (j, w) of the interval that holds the forced transition point, or (None, 2)."""
    xf = setup.xtr[side]
    if xf >= 1.0:
        return None, 2.0
    x = setup.sys.x_over_c[nodes_idx]
    if xf <= x[0]:
        return 0, 0.0
    for j in range(len(x) - 1):
        if x[j] < xf <= x[j + 1]:
            return j, (xf - x[j]) / (x[j + 1] - x[j])
    return None, 2.0


def _third_turbulent(setup: CaseSetup, st: CaseState):
    """Return the stations whose third variable is sqrt(Ctau) (turbulent and wake)."""
    up, lo = _sides(setup, st)
    idx = [np.arange(setup.n, setup.n_st)]
    for side, nodes_idx in enumerate((up, lo)):
        idx.append(nodes_idx[st.itr[side] + 1:])
    return np.concatenate(idx).astype(int)


# ---------------------------------------------------------------------------
# Residuals and Jacobians (batched over cases)
# ---------------------------------------------------------------------------


@dataclass
class _Lists:
    """Index lists of all equations of a batch (host numpy arrays)."""

    kinds: dict
    sim: dict
    te: dict
    drop: dict


def _build_lists(setups, states):
    """Return the equation index lists of a batch."""
    kinds = {k: {"b": [], "g0": [], "g1": [], "g2": [], "dxi": [], "dxi0": [], "wf": [], "nc": []}
             for k in (eq.LAMINAR, eq.TURBULENT, eq.WAKE, eq.TRANSITION, eq.POST)}
    sim = {"b": [], "g": [], "gu": [], "gl": [], "cu": [], "cl": [], "ds": []}
    te = {"b": [], "gu": [], "gl": [], "gw": [], "gap": [], "tu": [], "tl": []}
    drop = {"b": [], "g": []}
    for b, (su, st) in enumerate(zip(setups, states)):
        up, lo = _sides(su, st)
        s = su.sys.s
        n = su.n
        if st.drop >= 0:
            d = st.drop
            rows = [(up[0], up[0], up[0], 1.0, 0.0, s[d] - s[up[0]]),
                    (lo[0], lo[0], lo[0], 0.0, 1.0, s[lo[0]] - s[d])]
            drop["b"].append(b)
            drop["g"].append(d)
        else:
            dsg = s[st.i_s + 1] - s[st.i_s]
            rows = [(up[0], up[0], lo[0], 1.0, 1.0, dsg), (lo[0], up[0], lo[0], 1.0, 1.0, dsg)]
        for g, gu, gl, cu, cl_, dd in rows:
            for key, val in zip(("b", "g", "gu", "gl", "cu", "cl", "ds"), (b, g, gu, gl, cu, cl_, dd)):
                sim[key].append(val)
        turb_te = []
        for side, nodes_idx in enumerate((up, lo)):
            dxi, _ = _side_geometry(su, nodes_idx)
            itr = st.itr[side]
            jf, wfv = _forced_interval(su, side, nodes_idx)
            for j in range(len(nodes_idx) - 1):
                if j < itr:
                    k = eq.LAMINAR
                elif j == itr:
                    k = eq.TRANSITION
                elif j == itr + 1:
                    k = eq.POST
                else:
                    k = eq.TURBULENT
                lst = kinds[k]
                jw = itr if k == eq.POST else j
                lst["b"].append(b)
                lst["g0"].append(nodes_idx[max(j - 1, 0)])
                lst["g1"].append(nodes_idx[j])
                lst["g2"].append(nodes_idx[j + 1])
                lst["dxi"].append(dxi[j])
                lst["dxi0"].append(dxi[max(j - 1, 0)])
                lst["wf"].append(wfv if (jf is not None and jf == jw) else 2.0)
                lst["nc"].append(su.ncrit)
            turb_te.append(itr < len(nodes_idx) - 1)
        for key, val in zip(("b", "gu", "gl", "gw", "gap", "tu", "tl"),
                            (b, up[-1], lo[-1], n, su.sys.te_gap, turb_te[0], turb_te[1])):
            te[key].append(val)
        for j in range(su.nw - 1):
            lst = kinds[eq.WAKE]
            lst["b"].append(b)
            lst["g0"].append(n + j)
            lst["g1"].append(n + j)
            lst["g2"].append(n + j + 1)
            lst["dxi"].append(su.sys.wake_len[j])
            lst["dxi0"].append(su.sys.wake_len[j])
            lst["wf"].append(2.0)
            lst["nc"].append(su.ncrit)
    conv = {k: {key: np.asarray(v) for key, v in lst.items()} for k, lst in kinds.items()}
    return _Lists(conv, {k: np.asarray(v) for k, v in sim.items()},
                  {k: np.asarray(v) for k, v in te.items()}, {k: np.asarray(v, dtype=int) for k, v in drop.items()})


def linalg_errors(xp=np) -> tuple:
    """Return the exception types of a singular linear system (numpy, and torch for the torch backend).

    Parameters
    ----------
    xp : module
        numpy or torch.

    Returns
    -------
    tuple of type
        numpy.linalg.LinAlgError, and torch.linalg.LinAlgError when ``xp``
        is torch.
    """
    if xp is np:
        return (np.linalg.LinAlgError,)
    return (np.linalg.LinAlgError, xp.linalg.LinAlgError)


def _newton_steps(be, J, R):
    """Return the Newton steps of a batch and the mask of the cases with a singular matrix.

    The batch is solved at once. When that solve fails with a singular
    matrix, the cases are solved one at a time; a case whose matrix is
    singular gets a zero step and a True entry in the mask.

    Parameters
    ----------
    be : _Backend
        Array backend.
    J : array
        Jacobians, shape (B, 3 n_st, 3 n_st).
    R : array
        Residuals, shape (B, 3 n_st).

    Returns
    -------
    tuple
        Host steps, shape (B, 3 n_st), and a boolean mask of shape (B,).
    """
    nb = int(R.shape[0])
    try:
        return be.host(be.solve(J, -R)), np.zeros(nb, dtype=bool)
    except be.singular_errors:
        pass
    out = np.zeros((nb, int(R.shape[1])))
    bad = np.zeros(nb, dtype=bool)
    for k in range(nb):
        try:
            out[k] = be.host(be.solve(J[k:k + 1], -R[k:k + 1]))[0]
        except be.singular_errors:
            bad[k] = True
    return out, bad


class _Backend:
    """Array-module helper (numpy or torch)."""

    def __init__(self, xp, device=None, dtype=None):
        self.xp = xp
        self.torch = xp is not np
        self.device = device
        self.dtype = dtype if dtype is not None else (np.float64 if xp is np else xp.float64)
        #: Exceptions of a singular linear system on this backend.
        self.singular_errors = linalg_errors(xp)

    def arr(self, a, integer=False):
        """Return a host array as an array of this backend."""
        if not self.torch:
            return np.asarray(a, dtype=int if integer else np.float64)
        import torch

        if integer:
            return torch.as_tensor(np.asarray(a, dtype=np.int64), device=self.device)
        return torch.as_tensor(np.asarray(a, dtype=np.float64), dtype=self.dtype, device=self.device)

    def bool(self, a):
        """Return a host boolean array on this backend."""
        if not self.torch:
            return np.asarray(a, dtype=bool)
        import torch

        return torch.as_tensor(np.asarray(a, dtype=bool), device=self.device)

    def zeros(self, shape):
        """Return zeros."""
        if not self.torch:
            return np.zeros(shape)
        return self.xp.zeros(shape, dtype=self.dtype, device=self.device)

    def add_at(self, a, idx, vals):
        """Add values at an index tuple (accumulating)."""
        if not self.torch:
            np.add.at(a, idx, vals)
        else:
            a.index_put_(idx, vals, accumulate=True)

    def solve(self, A, b):
        """Solve batched linear systems A x = b (b of shape (B, n))."""
        if not self.torch:
            return np.linalg.solve(A, b[..., None])[..., 0]
        return self.xp.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)

    def solve_mat(self, A, b):
        """Solve batched linear systems A X = B (B of shape (B, n, k))."""
        if not self.torch:
            return np.linalg.solve(A, b)
        return self.xp.linalg.solve(A, b)

    def stack(self, arrays):
        """Stack arrays of this backend along a new first axis."""
        if not self.torch:
            return np.stack(arrays)
        return self.xp.stack(arrays)

    def host(self, a):
        """Return a host numpy array (float64)."""
        if not self.torch:
            return a
        return a.detach().double().cpu().numpy()


def _evaluate(be: _Backend, lists: _Lists, Xd, UE, re_b, need_jac: bool):
    """Return the residuals (B, 3n) and the Jacobians in the (theta, delta_star, c, ue) variables.

    J (B, 3n, 3n) holds the derivatives with respect to the station values
    (theta, delta_star, c) at fixed ue; Ju (B, 3n, n) the derivatives with
    respect to the station edge velocities.
    """
    xp = be.xp
    nb, nst, _ = Xd.shape
    R = be.zeros((nb, 3 * nst))
    J = be.zeros((nb, 3 * nst, 3 * nst)) if need_jac else None
    Ju = be.zeros((nb, 3 * nst, nst)) if need_jac else None
    for kind, lst in lists.kinds.items():
        if lst["b"].size == 0:
            continue
        bi = be.arr(lst["b"], True)
        g1 = be.arr(lst["g1"], True)
        g2 = be.arr(lst["g2"], True)
        dxi = be.arr(lst["dxi"])
        v = variables([Xd[bi, g1, 0], Xd[bi, g1, 1], Xd[bi, g1, 2], UE[bi, g1],
                       Xd[bi, g2, 0], Xd[bi, g2, 1], Xd[bi, g2, 2], UE[bi, g2]], xp=xp)
        re = re_b[bi]
        if kind == eq.POST:
            g0 = be.arr(lst["g0"], True)
            v = variables([Xd[bi, g1, 0], Xd[bi, g1, 1], Xd[bi, g1, 2], UE[bi, g1],
                           Xd[bi, g2, 0], Xd[bi, g2, 1], Xd[bi, g2, 2], UE[bi, g2],
                           Xd[bi, g0, 0], Xd[bi, g0, 1], Xd[bi, g0, 2], UE[bi, g0]], xp=xp)
            r1, r2, r3 = eq.post_transition_residuals(tuple(v[8:]), tuple(v[:4]), tuple(v[4:8]),
                                                      be.arr(lst["dxi0"]), dxi, re, be.arr(lst["nc"]),
                                                      be.arr(lst["wf"]))
        elif kind == eq.TRANSITION:
            r1, r2, r3, _ = eq.transition_residuals(tuple(v[:4]), tuple(v[4:]), dxi, re,
                                                     be.arr(lst["nc"]), be.arr(lst["wf"]))
        else:
            r1, r2, r3 = eq.interval_residuals(kind, tuple(v[:4]), tuple(v[4:]), dxi, re)
        cols = ((0, g1, 0), (1, g1, 1), (2, g1, 2), (4, g2, 0), (5, g2, 1), (6, g2, 2))
        if kind == eq.POST:
            cols = cols + ((8, g0, 0), (9, g0, 1), (10, g0, 2))
        for k, r in enumerate((r1, r2, r3)):
            row = 3 * g2 + k
            R[bi, row] = r.val
            if need_jac:
                for j, gc, jc in cols:
                    be.add_at(J, (bi, row, 3 * gc + jc), r.der[..., j])
                be.add_at(Ju, (bi, row, g1), r.der[..., 3])
                be.add_at(Ju, (bi, row, g2), r.der[..., 7])
                if kind == eq.POST:
                    be.add_at(Ju, (bi, row, g0), r.der[..., 11])
    sm = lists.sim
    if sm["b"].size:
        bi = be.arr(sm["b"], True)
        g = be.arr(sm["g"], True)
        gu = be.arr(sm["gu"], True)
        gl = be.arr(sm["gl"], True)
        v = variables([Xd[bi, g, 0], Xd[bi, g, 1], Xd[bi, g, 2], UE[bi, gu], UE[bi, gl]], xp=xp)
        a = (be.arr(sm["cu"]) * v[3] + be.arr(sm["cl"]) * v[4]) / be.arr(sm["ds"])
        for k, r in enumerate(eq.similarity_residuals(v[0], v[1], v[2], a, re_b[bi])):
            row = 3 * g + k
            R[bi, row] = r.val
            if need_jac:
                for j in range(3):
                    be.add_at(J, (bi, row, 3 * g + j), r.der[..., j])
                be.add_at(Ju, (bi, row, gu), r.der[..., 3])
                be.add_at(Ju, (bi, row, gl), r.der[..., 4])
    te = lists.te
    if te["b"].size:
        bi = be.arr(te["b"], True)
        gu = be.arr(te["gu"], True)
        gl = be.arr(te["gl"], True)
        gw = be.arr(te["gw"], True)
        v = variables([Xd[bi, gu, 0], Xd[bi, gu, 1], Xd[bi, gu, 2], UE[bi, gu],
                       Xd[bi, gl, 0], Xd[bi, gl, 1], Xd[bi, gl, 2], UE[bi, gl],
                       Xd[bi, gw, 0], Xd[bi, gw, 1], Xd[bi, gw, 2]], xp=xp)
        res = eq.trailing_edge_residuals(tuple(v[0:4]), tuple(v[4:8]), tuple(v[8:11]), be.arr(te["gap"]),
                                         re_b[bi], be.bool(te["tu"]), be.bool(te["tl"]))
        cols = ((0, gu, 0), (1, gu, 1), (2, gu, 2), (4, gl, 0), (5, gl, 1), (6, gl, 2),
                (8, gw, 0), (9, gw, 1), (10, gw, 2))
        for k, r in enumerate(res):
            row = 3 * gw + k
            R[bi, row] = r.val
            if need_jac:
                for j, gc, jc in cols:
                    be.add_at(J, (bi, row, 3 * gc + jc), r.der[..., j])
                be.add_at(Ju, (bi, row, gu), r.der[..., 3])
                be.add_at(Ju, (bi, row, gl), r.der[..., 7])
    return R, J, Ju


def _edge(be: _Backend, ue_inv, D, X, S):
    """Return the station edge velocities and the matrix A of the mixed variables."""
    x2 = X[:, :, 1]
    n = x2.shape[-1]
    eye = be.zeros((n, n))
    idx = be.arr(np.arange(n), True)
    eye[idx, idx] = 1.0
    A = eye[None] - D * (S * x2)[:, None, :]
    rhs = ue_inv + (D @ ((1.0 - S) * x2)[:, :, None])[:, :, 0]
    return be.solve(A, rhs), A


def _residual_m(be: _Backend, lists: _Lists, X, ue_inv, D, re_b, S, drop_b, drop_g, need_jac: bool):
    """Return R, the Jacobian in the station variables and ue.

    The second variable is m (mass defect) or, where S is 1, delta_star.
    ue solves (I - D diag(S delta_star)) ue = ue_inv + D ((1 - S) m) and
    delta_star = m / ue at the mass-defect stations. The rows of a dropped
    stagnation node hold the identity (its variables do not change).
    """
    x2 = X[:, :, 1]
    UE, A = _edge(be, ue_inv, D, X, S)
    safe = UE * 1.0
    if drop_b.size:
        db = be.arr(drop_b, True)
        dg = be.arr(drop_g, True)
        safe[db, dg] = 1.0
    Xd = X * 1.0
    Xd[:, :, 1] = S * x2 + (1.0 - S) * x2 / safe
    if drop_b.size:
        Xd[db, dg, 1] = 2.0 * X[db, dg, 0]
    R, J, Ju = _evaluate(be, lists, Xd, UE, re_b, need_jac)
    if not need_jac:
        return R, None, UE
    coef = S * UE + (1.0 - S)
    M = be.solve_mat(A, D * coef[:, None, :])
    dd_dx = S + (1.0 - S) / safe
    dd_du = -(1.0 - S) * x2 / (safe * safe)
    Jd = J[:, :, 1::3]
    Jm = J * 1.0
    Jm[:, :, 1::3] = Jd * dd_dx[:, None, :] + (Ju + Jd * dd_du[:, None, :]) @ M
    if drop_b.size:
        for k in range(3):
            Jm[db, 3 * dg + k, :] = 0.0
            Jm[db, 3 * dg + k, 3 * dg + k] = 1.0
    return R, Jm, UE


# ---------------------------------------------------------------------------
# First estimate by a direct march
# ---------------------------------------------------------------------------


def _law(coupling, g):
    """Return the local interaction law (a, d) of station g, or None (direct mode).

    ue_g = a + d ue_g delta_star_g, with the mass defects of the other
    stations held at their current values.
    """
    if coupling is None:
        return None
    ui, D, mm = coupling
    a = ui[g] + D[g] @ mm - D[g, g] * mm[g]
    return np.array([a]), np.array([D[g, g]])


def _march_side(setup: CaseSetup, nodes_idx, ue, a, side: int, coupling=None):
    """March one side for the first estimate.

    Returns the station values (theta, delta_star, c), the position of the
    last laminar station and the edge velocities of the march. In direct
    mode (``coupling`` None) the edge velocity is ``ue``. With ``coupling``
    = (ue_inv, D, m) each station follows the local interaction law of a
    quasi-simultaneous method (Veldman 2009): its edge velocity is
    ue_inv + D m with the mass defects of the other stations held at their
    current values (``m`` is updated in place as the march proceeds). The
    local law removes the singularity of the direct mode at separation.
    Past a separation point (or a failed station) the march continues with
    H frozen; this is a first estimate only.
    """
    re = setup.re_l
    m = len(nodes_idx)
    ue = np.array(ue, dtype=float)
    out = np.zeros((m, 3))
    hs = mr.stagnation_shape_factor()
    f0 = float(cl.laminar_cf_factor(hs)[0])
    th0 = np.sqrt(f0 / ((2.0 + hs) * re * a))
    out[0] = (th0, hs * th0, 0.0)
    if coupling is not None:
        g0 = nodes_idx[0]
        _march_edge(coupling, _law(coupling, g0), ue, 0, g0, hs * th0)
    dxi, _ = _side_geometry(setup, nodes_idx)
    jf, wfv = _forced_interval(setup, side, nodes_idx)
    turb = False
    itr = m - 1
    h_frozen = None
    one = np.array([re])
    for j in range(m - 1):
        g = nodes_idx[j + 1]
        law = _law(coupling, g)
        st1 = tuple(np.array([v]) for v in (out[j, 0], out[j, 1], out[j, 2], ue[j]))
        u2 = np.array([ue[j + 1]])
        dx = np.array([dxi[j]])
        if not turb:
            r1 = float(_laminar_rate(setup, out[j, 0], out[j, 1], ue[j]))
            forced = jf is not None and jf == j
            if out[j, 2] + dxi[j] * r1 >= setup.ncrit or forced:
                g_c = _ctau_start(setup, out[j, 0], out[j, 1], ue[j])
                t, d, c, ok, _ = mr.station_newton(eq.TRANSITION, st1, u2, dx, one, (st1[0], st1[1], g_c),
                                                   ncrit=np.array([setup.ncrit]),
                                                   w_forced=np.array([wfv if forced else 2.0]), law=law)
                turb = True
                itr = j
                h_frozen = None
                if not (ok[0] and d[0] / t[0] < 4.0):
                    h_frozen = min(out[j, 1] / out[j, 0], 2.5)
                    st1t = (st1[0], st1[1], g_c, st1[3])
                    t, d, c, ok, _ = mr.station_newton(eq.TURBULENT, st1t, u2, dx, one, (st1[0], st1[1], g_c),
                                                       h_fix=np.array([h_frozen]), law=law)
                out[j + 1] = (t[0], d[0], c[0]) if ok[0] else (out[j, 0], out[j, 1], g_c[0])
                _march_edge(coupling, law, ue, j + 1, g, out[j + 1, 1])
                continue
            kind = eq.LAMINAR
        else:
            kind = eq.TURBULENT
        hf = None if h_frozen is None else np.array([h_frozen])
        t, d, c, ok, _ = mr.station_newton(kind, st1, u2, dx, one, (st1[0], st1[1], st1[2]), h_fix=hf, law=law)
        u_new = ue[j + 1] if law is None else law[0][0] / max(1.0 - law[1][0] * d[0], 0.1)
        hsep = 4.0 if kind == eq.LAMINAR else float(cl.separation_shape_factor(u_new * t[0] * re)[0])
        if hf is None and not (ok[0] and d[0] / t[0] < hsep):
            h_frozen = min(out[j, 1] / out[j, 0], hsep - 0.2)
            t, d, c, ok, _ = mr.station_newton(kind, st1, u2, dx, one, (st1[0], st1[1], st1[2]),
                                               h_fix=np.array([h_frozen]), law=law)
        out[j + 1] = (t[0], d[0], c[0]) if ok[0] else (out[j, 0] * 1.01, out[j, 1] * 1.01, out[j, 2])
        _march_edge(coupling, law, ue, j + 1, g, out[j + 1, 1])
    return out, itr, ue


def _march_edge(coupling, law, ue, k, g, ds):
    """Set the edge velocity and the mass defect of a marched station (interaction law only)."""
    if coupling is None:
        return
    den = 1.0 - law[1][0] * ds
    if law[0][0] > 0.0 and den > 0.1:
        ue[k] = law[0][0] / den
    coupling[2][g] = ue[k] * ds


def initial_state(setup: CaseSetup, sweeps: int = 0) -> CaseState:
    """Return the first estimate of a case from marches on the inviscid edge velocity.

    Parameters
    ----------
    setup : CaseSetup
        Case data.
    sweeps : int
        0: direct marches on the inviscid edge velocity. k > 0: k sweeps of
        quasi-simultaneous marches (see :func:`_march_side`) over both
        sides and the wake, each sweep with the mass defects of the
        previous one; the edge velocities then include the displacement
        effect and the march passes separation points.

    Returns
    -------
    CaseState
        Station variables (theta, m, c) and structure.
    """
    s = setup.sys
    panel = _find_stagnation(s.gamma_inv, None)
    i_s, drop = _canonical(panel, _drop_node(s.gamma_inv, panel, s.s))
    st = CaseState(np.zeros((setup.n_st, 3)), i_s, drop, [0, 0], panel)
    ue_inv, D = _coupling_matrices(setup, st)
    up, lo = _sides(setup, st)
    if st.drop >= 0:
        a_u = ue_inv[up[0]] / (s.s[st.drop] - s.s[up[0]])
        a_l = ue_inv[lo[0]] / (s.s[lo[0]] - s.s[st.drop])
    else:
        a_u = a_l = (ue_inv[up[0]] + ue_inv[lo[0]]) / (s.s[st.i_s + 1] - s.s[st.i_s])
    n = setup.n
    mm = np.zeros(setup.n_st)
    coupling = None if sweeps <= 0 else (ue_inv, D, mm)
    one = np.array([setup.re_l])
    for _ in range(max(sweeps, 1)):
        xu, itr_u, uu = _march_side(setup, up, ue_inv[up], a_u, 0, coupling)
        xl, itr_l, ul = _march_side(setup, lo, ue_inv[lo], a_l, 1, coupling)
        xd = np.zeros((setup.n_st, 3))
        ue_all = ue_inv.copy()
        xd[up] = xu
        xd[lo] = xl
        ue_all[up] = uu
        ue_all[lo] = ul
        st.itr = [itr_u, itr_l]
        if st.drop >= 0:
            xd[st.drop] = xu[0]
        tu, tl = xu[-1], xl[-1]

        def ctau(x, ue_, turb):
            return x[2] ** 2 if turb else float(_ctau_start(setup, x[0], x[1], ue_)[0]) ** 2

        cu = ctau(tu, uu[-1], itr_u < len(up) - 1)
        cl_ = ctau(tl, ul[-1], itr_l < len(lo) - 1)
        thw = tu[0] + tl[0]
        xd[n] = (thw, tu[1] + tl[1] + s.te_gap, np.sqrt((tu[0] * cu + tl[0] * cl_) / thw))
        if coupling is not None:
            _march_edge(coupling, _law(coupling, n), ue_all, n, n, xd[n, 1])
        for j in range(setup.nw - 1):
            g1, g2 = n + j, n + j + 1
            law = _law(coupling, g2)
            st1 = (np.array([xd[g1, 0]]), np.array([xd[g1, 1]]), np.array([xd[g1, 2]]), np.array([ue_all[g1]]))
            t, d, c, ok, _ = mr.station_newton(eq.WAKE, st1, np.array([ue_all[g2]]), np.array([s.wake_len[j]]),
                                               one, st1[:3], law=law)
            xd[g2] = (t[0], d[0], c[0]) if ok[0] else xd[g1]
            _march_edge(coupling, law, ue_all, g2, g2, xd[g2, 1])
    to_mixed(setup, st, xd, ue_all)
    return st


# ---------------------------------------------------------------------------
# Structure updates
# ---------------------------------------------------------------------------


def _find_stagnation(gamma, i_old):
    """Return the stagnation panel index i: gamma[i] > 0 >= gamma[i + 1]."""
    cand = np.nonzero((gamma[:-1] > 0.0) & (gamma[1:] <= 0.0))[0]
    if cand.size == 0:
        return int(np.argmin(np.abs(gamma[:-1])))
    ref = 0.5 * len(gamma) if i_old is None else i_old
    return int(cand[np.argmin(np.abs(cand - ref))])


def _canonical(i_s, drop):
    """Return the stagnation panel index in canonical form (i_s = drop - 1 when a node is dropped)."""
    return (drop - 1, drop) if drop >= 0 else (i_s, drop)


def _drop_node(gamma, i_s, s):
    """Return the node that is too close to the stagnation point to be a station, or -1.

    The stagnation point divides the panel (i_s, i_s + 1) at the fraction
    w = gamma[i_s] / (gamma[i_s] - gamma[i_s + 1]). A first station closer
    to the stagnation point than STAGNATION_DROP times its next interval
    would make the logarithmic differences of the next interval
    ill-conditioned (ue nearly zero); that node is then not a station (its
    mass defect is zero) and the side starts at the next node.
    """
    w = gamma[i_s] / (gamma[i_s] - gamma[i_s + 1])
    ds = s[i_s + 1] - s[i_s]
    if i_s >= 1 and w * ds < STAGNATION_DROP * (s[i_s] - s[i_s - 1]):
        return int(i_s)
    if i_s + 2 < len(s) and (1.0 - w) * ds < STAGNATION_DROP * (s[i_s + 2] - s[i_s + 1]):
        return int(i_s + 1)
    return -1


def _gamma_total(setup: CaseSetup, st: CaseState, omega: float = 1.0):
    """Return the total nodal gamma of the current state."""
    _, _, m = host_edge(setup, st, omega)
    sigma = _source_matrix(setup, st) @ m
    return setup.sys.gamma_inv + omega * (setup.sys.gamma_src @ sigma)


def _n_increase(setup, xa, xb, dsa, dsb, ua, ub, dxi):
    """Return the laminar increase of N between two station states (values)."""
    va = variables([np.array([xa[0]]), np.array([dsa]), np.array([xa[2]]), np.array([ua])])
    vb = variables([np.array([xb[0]]), np.array([dsb]), np.array([0.0]), np.array([ub])])
    nb = eq.laminar_n_prediction(tuple(va), tuple(vb), np.array([dxi]), setup.re_l)
    return float(nb.val[0]) - xa[2]


def to_mixed(setup: CaseSetup, st: CaseState, xd, ue_est):
    """Set the station variables of a state from (theta, delta_star, c) values.

    The second variable is delta_star at the first station of each side and
    the mass defect delta_star * ue_est elsewhere (zero at a dropped node).
    """
    S = _dstar_mask(setup, st)
    mask = _station_mask(setup, st)
    x = np.array(xd, dtype=float)
    x[:, 1] = np.where(S, xd[:, 1], np.where(mask, xd[:, 1] * np.abs(ue_est), 0.0))
    st.x = x


def _update_structure(setup: CaseSetup, st: CaseState, omega: float = 1.0, transition: bool = True,
                      pre=None) -> bool:
    """Update the stagnation point and (when asked) the transition intervals; return True on a change.

    ``pre`` holds (ue, delta_star, gamma) of the state when the caller has
    them (the batched solver computes them on its device).
    """
    changed = False
    if pre is None:
        ue, ds, _ = host_edge(setup, st, omega)
        gam = _gamma_total(setup, st, omega)
    else:
        ue, ds, gam = pre
    panel = _find_stagnation(gam, st.panel)
    # The node-drop decision is made again when the stagnation point moves
    # to another panel. On the same panel a node can be dropped (the
    # stagnation point came too close to it) but not restored, so that the
    # decision cannot oscillate within a solve.
    if panel != st.panel:
        i_new, d_new = _canonical(panel, _drop_node(gam, panel, setup.sys.s))
        st.panel = panel
    elif st.drop < 0:
        i_new, d_new = _canonical(panel, _drop_node(gam, panel, setup.sys.s))
    else:
        i_new, d_new = st.i_s, st.drop
    if i_new != st.i_s or d_new != st.drop:
        xd = st.x.copy()
        xd[:, 1] = ds
        old_up, old_lo = _sides(setup, st)
        st.i_s, st.drop = i_new, d_new
        up, lo = _sides(setup, st)
        for side, (new, old) in enumerate(((up, old_up), (lo, old_lo))):
            moved = new[~np.isin(new, old)]
            if moved.size:
                stay = new[np.isin(new, old)]
                ref = stay[0] if stay.size else new[0]
                xd[moved] = xd[ref]
                xd[moved, 2] = 0.0
            st.itr[side] = int(np.clip(st.itr[side] + moved.size - np.sum(~np.isin(old, new)), 0, len(new) - 1))
        if st.drop >= 0:
            xd[st.drop] = xd[up[0]]
        # Edge-velocity estimate for the new structure: |gamma| at the nodes
        # (continuous when a node changes side), the old values in the wake.
        ue_est = np.concatenate([np.abs(gam), ue[setup.n:]])
        to_mixed(setup, st, xd, ue_est)
        ue, ds, _ = host_edge(setup, st, omega)
        changed = True
    up, lo = _sides(setup, st)
    for side, nodes_idx in enumerate((up, lo)):
        if st.frozen[side] or not transition:
            continue
        m = len(nodes_idx)
        itr = st.itr[side]
        dxi, _ = _side_geometry(setup, nodes_idx)
        xs = st.x[nodes_idx]
        us = ue[nodes_idx]
        rate = _laminar_rate(setup, xs[:, 0], ds[nodes_idx], us)
        jf, _ = _forced_interval(setup, side, nodes_idx)
        new = m - 1
        for j in range(min(itr + 1, m - 1)):
            if xs[j, 2] + dxi[j] * rate[j] >= setup.ncrit or (jf is not None and j == jf):
                new = j
                break
        if new == m - 1 and itr < m - 1:
            # The transition point left its interval downstream. The new
            # interval is estimated with the growth rate of the last laminar
            # station held constant (the stations downstream are turbulent,
            # so their laminar rate is not known); the N of the stations that
            # become laminar follow the same estimate.
            need = setup.ncrit - xs[itr, 2]
            xi = np.concatenate([[0.0], np.cumsum(dxi[itr:])])
            if rate[itr] > 0.0:
                reach = np.nonzero(xi[1:] * rate[itr] >= need)[0]
                new = itr + 1 + (int(reach[0]) - 1 if reach.size else m)
                new = int(min(max(new, itr + 1), m - 1))
            else:
                new = itr + 1
            for k in range(itr + 1, new + 1):
                st.x[nodes_idx[k], 2] = (min(xs[itr, 2] + rate[itr] * xi[k - itr], setup.ncrit - 1e-9)
                                         if rate[itr] > 0.0 else xs[itr, 2])
        if new != itr:
            if new < itr:
                g = nodes_idx[new + 1:itr + 1]
                st.x[g, 2] = _ctau_start(setup, st.x[g, 0], ds[g], ue[g])
            st.itr[side] = new
            changed = True
    return changed


# ---------------------------------------------------------------------------
# Newton iteration
# ---------------------------------------------------------------------------


def _shape_and_edge(x, ui, D, S, mask):
    """Return the edge velocities and shape factors of station values (numpy)."""
    A = np.eye(len(x)) - D * (S * x[:, 1])[None, :]
    u = np.linalg.solve(A, ui + D @ ((1.0 - S) * x[:, 1]))
    us = np.where(mask, u, 1.0)
    return u, (S * x[:, 1] + (1.0 - S) * x[:, 1] / us) / x[:, 0]


def _limit_step(setup, st, x0, d, ui, D, S, mask):
    """Return the step factor of a Newton or Levenberg-Marquardt step.

    theta, m (or delta_star) and sqrt(Ctau) fall by at most 50 % and rise
    by at most 200 %, H - 1 falls by at most 50 % and ue stays positive.
    """
    if not np.all(np.isfinite(d)):
        return 0.0
    third = _third_turbulent(setup, st)
    rel = np.concatenate([d[mask, 0] / x0[mask, 0], d[mask, 1] / x0[mask, 1], d[third, 2] / x0[third, 2]])
    dec = np.max(-rel, initial=0.0)
    inc = np.max(rel, initial=0.0)
    lam = min(1.0, 0.5 / dec if dec > 0.0 else 1.0, 2.0 / inc if inc > 0.0 else 1.0)
    _, h0 = _shape_and_edge(x0, ui, D, S, mask)
    for _ in range(60):
        xt = x0 + lam * d
        if np.all(xt[mask, :2] > 0.0):
            ut, ht = _shape_and_edge(xt, ui, D, S, mask)
            if np.all(ut[mask] > 0.0) and np.all(ht[mask] > 1.0 + 0.5 * (h0[mask] - 1.0)):
                return lam
        lam *= 0.5
    return 0.0


def _lm_step(be, setup, st, x0, J, R, r0, ui, D, S, mask, mu, trials=LM_TRIALS):
    """Return a Levenberg-Marquardt step that decreases the residual, and the new damping.

    Solves (J^T J + mu diag(J^T J)) dx = -J^T R (Levenberg 1944; Marquardt
    1963) and raises mu by 10 until the 2-norm of the residual decreases,
    with at most ``trials`` values of mu (one linear solve each). Returns
    (None, mu) when no step decreases it. The third value is the number of
    linear solves.
    """
    G = J.T @ J
    g = J.T @ R
    dg = np.diag(G).copy()
    dg = np.where(dg > 0.0, dg, 1.0)
    lists = _build_lists([setup], [st])
    drop_b = np.array([0] if st.drop >= 0 else [], dtype=int)
    drop_g = np.array([st.drop] if st.drop >= 0 else [], dtype=int)
    one = be.arr(np.array([setup.re_l]))
    n_solves = 0
    for _ in range(int(trials)):
        n_solves += 1
        try:
            d = np.linalg.solve(G + mu * np.diag(dg), -g).reshape(-1, 3)
        except np.linalg.LinAlgError:
            mu *= 10.0
            continue
        lam = _limit_step(setup, st, x0, d, ui, D, S, mask)
        if lam > 0.0:
            xt = x0 + lam * d
            Rt, _, UEt = _residual_m(be, lists, be.arr(xt[None]), be.arr(ui[None]), be.arr(D[None]), one,
                                     be.arr(S[None]), drop_b, drop_g, False)
            rn = float(np.sqrt(np.sum(be.host(Rt) ** 2)))
            ue_t = be.host(UEt)[0]
            if np.isfinite(rn) and rn < r0 and np.all(ue_t[mask] > 0.0):
                return xt, max(mu * 0.1, 1e-12), n_solves
        mu *= 10.0
    return None, 1e-3, n_solves


def _freeze_cycles(hist, setup: CaseSetup, st: CaseState):
    """Stop a transition interval that alternates between two positions.

    The converged states of two adjacent transition intervals can each
    place the transition point in the other interval (the laminar and the
    turbulent closures differ at the station between them). The positions
    are updated only near convergence (see :func:`newton_solve`); when the
    position of a side returns to its value of two updates before,
    the downstream position is kept for the rest of the solve; the
    transition point then lies at the node between the two intervals,
    within one interval of the point where N = ncrit.
    """
    hist.append((st.i_s, st.drop, tuple(st.itr)))
    moved = False
    if len(hist) < 3:
        return moved
    a, b, c = hist[-3], hist[-2], hist[-1]
    if a[:2] != c[:2] or b[:2] != c[:2]:
        return moved
    for side in (0, 1):
        if not st.frozen[side] and a[2][side] == c[2][side] != b[2][side]:
            keep = max(b[2][side], c[2][side])
            if keep != st.itr[side]:
                up, lo = _sides(setup, st)
                nodes_idx = (up, lo)[side]
                ue, ds, _ = host_edge(setup, st)
                itr = st.itr[side]
                dxi, _ = _side_geometry(setup, nodes_idx)
                for j in range(itr, keep):
                    ga, gb = nodes_idx[j], nodes_idx[j + 1]
                    st.x[gb, 2] = st.x[ga, 2] + _n_increase(setup, st.x[ga], st.x[gb], ds[ga], ds[gb], ue[ga],
                                                            ue[gb], dxi[j])
                st.itr[side] = keep
                moved = True
            st.frozen[side] = True
    return moved


def _early_stop(res, level_min, smallest, no_step, at_one, stall_window, stall_factor, divergence_factor,
                floor=0.0):
    """Return (status, reason) of an early stop of a Newton solve, or None.

    Parameters
    ----------
    res : float
        Largest absolute residual of the current iterate [-].
    level_min : list of float
        Running minimum of the residual since the last change of omega (or,
        at omega = 1, of the structure), the current evaluation last [-].
    smallest : float
        Smallest residual of the earlier evaluations at omega = 1 [-]
        (inf when there is none).
    no_step : int
        Consecutive iterations without a step (the line search and the
        Levenberg-Marquardt fallback both failed) on the same structure.
    at_one : bool
        True when omega = 1.
    stall_window : int
        Window of the stagnation rule [iterations] (0: both stagnation
        rules off).
    stall_factor : float
        Factor by which the smallest residual must fall in the window [-].
    divergence_factor : float
        Growth over the smallest earlier residual at omega = 1 that is
        divergence [-] (0: off).
    floor : float
        The stagnation rules apply only while the smallest residual of the
        window is above this value [-]. Near the tolerance the residual can
        stall at the round-off level (float32) and still reach it.

    Returns
    -------
    tuple of str or None
        The status ("stagnation" or "divergence") and the reason text.
    """
    stall = stall_window > 0 and level_min[-1] > floor
    if stall and at_one and no_step >= 2:
        # Two failed steps on the same state and structure: the next
        # iterations repeat the last one exactly (fixed point of the loop).
        return "stagnation", (f"stagnation: neither the Newton line search nor the Levenberg-Marquardt fallback "
                              f"reduces the residual (largest residual {res:.3e})")
    if divergence_factor > 0.0 and at_one and res > divergence_factor * smallest:
        return "divergence", (f"divergence: the residual grew to more than {divergence_factor:g} times its "
                              f"smallest value (largest residual {res:.3e})")
    if stall and len(level_min) > stall_window and level_min[-1] > stall_factor * level_min[-1 - stall_window]:
        where = "on the same structure" if at_one else "at a coupling factor omega below 1"
        return "stagnation", (f"stagnation: the smallest residual did not fall by the factor {stall_factor:g} in "
                              f"{stall_window} Newton iterations {where} (largest residual {res:.3e})")
    return None


def newton_solve(setups, states, *, tol=1e-6, max_iter=50, xp=np, device=None, dtype=None, verbose=False,
                 omega0=None, stall_window=STALL_WINDOW, stall_factor=STALL_FACTOR,
                 divergence_factor=DIVERGENCE_FACTOR, lm_trials=LM_RETRY_TRIALS):
    """Solve a batch of coupled cases by the global Newton method.

    Parameters
    ----------
    setups : list of CaseSetup
        Case data (same node and wake counts in a batch).
    states : list of CaseState
        First estimates (changed in place).
    tol : float
        Convergence limit of the largest absolute residual [-].
    max_iter : int
        Largest number of Newton iterations.
    xp : module
        numpy or torch.
    device, dtype : optional
        Torch device and dtype.
    verbose : bool
        Print the residual history.
    omega0 : list of float, optional
        Start value of the coupling factor per case: 1/64 for a first
        estimate from a direct march (cold start), 1 for a converged
        neighbouring solution (warm start). Default: 1/64 for all.
    stall_window : int
        Stagnation stop: a case stops with status "stagnation" when its
        smallest residual has not fallen by the factor ``stall_factor`` in
        the last ``stall_window`` iterations [-]. The window starts again
        when omega changes and, at omega = 1, when the structure changes. 0
        switches the stagnation stops off.
    stall_factor : float
        Factor of the stagnation stop [-], in (0, 1). The stagnation stops
        apply only while the smallest residual is above STALL_FLOOR times
        ``tol``.
    divergence_factor : float
        Divergence stop: a case stops with status "divergence" when, at
        omega = 1, the residual exceeds this factor times its smallest
        earlier value [-]. 0 switches the rule off.
    lm_trials : int
        Largest number of damping values (one linear solve each) of a
        Levenberg-Marquardt fallback step after a fallback step of the same
        solve found no decrease of the residual while the residual was above
        STALL_FLOOR times ``tol`` [-], at least 1. The fallback steps before
        that try LM_TRIALS values. Near the tolerance (the round-off level of
        float32) a failed fallback step does not set the cap.

    Returns
    -------
    list of NewtonResult
        One result per case, in input order.

    Notes
    -----
    A case also stops with status "stagnation" when the line search and
    the Levenberg-Marquardt fallback both fail twice in a row on the same
    structure: the following iterations would repeat the last one exactly.
    The stop rules change the work of a failed solve, not a converged
    solution.

    A singular Newton matrix (a linear-algebra error of its solve) gives
    no Newton step: the Levenberg-Marquardt fallback is tried instead. When
    the fallback also fails at omega = 1, the case stops at once with status
    "singular" (the next iterations would repeat the last one exactly; this
    stop does not depend on the stop rules). Below omega = 1 it counts as a
    failed step. A singular linear system at another place of the solve
    stops the cases that are not finished with status "singular". A
    singular matrix never raises an exception.
    """
    if int(lm_trials) != lm_trials or lm_trials < 1:
        raise ValueError(f"lm_trials must be an integer >= 1, got {lm_trials}.")
    be = _Backend(xp, device, dtype)
    nb = len(setups)
    results = [None] * nb
    omega = np.full(nb, 1.0 / 64.0) if omega0 is None else np.asarray(omega0, dtype=float).copy()
    last_res = np.full(nb, np.inf)
    changed_prev = np.ones(nb, dtype=bool)
    active = list(range(nb))
    mats = {}
    history = [[] for _ in range(nb)]
    best_res = np.full(nb, np.inf)
    best_it = np.zeros(nb, dtype=int)
    lm_mu = np.full(nb, 1e-3)
    lm_failed = np.zeros(nb, dtype=bool)
    level_min = [[] for _ in range(nb)]
    level_omega = np.full(nb, np.nan)
    smallest = np.full(nb, np.inf)
    no_step = np.zeros(nb, dtype=int)
    n_lin = np.zeros(nb, dtype=int)

    sens = {}

    def assemble(idx, refresh=True):
        for i in idx:
            if i not in sens:
                sens[i] = (be.arr(setups[i].sens), be.arr(setups[i].base))
            if refresh or i not in mats:
                mats[i] = _p_and_b(setups[i], states[i])
        P = be.arr(np.stack([mats[i][0] for i in idx]))
        B = be.arr(np.stack([mats[i][1] for i in idx]))
        sens_b = be.stack([sens[i][0] for i in idx])
        base_b = be.stack([sens[i][1] for i in idx])
        om = be.arr(omega[list(idx)])
        ue_inv = (P @ base_b[:, :, None])[:, :, 0]
        D = (P @ (sens_b @ B)) * om[:, None, None]
        X = be.arr(np.stack([states[i].x for i in idx]))
        re_b = be.arr(np.array([setups[i].re_l for i in idx]))
        masks = np.stack([_station_mask(setups[i], states[i]) for i in idx])
        drop_b = np.array([k for k, i in enumerate(idx) if states[i].drop >= 0], dtype=int)
        drop_g = np.array([states[i].drop for i in idx if states[i].drop >= 0], dtype=int)
        S = be.arr(np.stack([_dstar_mask(setups[i], states[i]) for i in idx]).astype(float))
        return ue_inv, D, X, re_b, masks, drop_b, drop_g, S

    def edge(ue_inv, D, X, S):
        return _edge(be, ue_inv, D, X, S)[0]

    def edge_ok(ue_h, masks):
        return np.array([bool(np.all(u[m] > 0.0) and np.all(np.isfinite(u))) for u, m in zip(ue_h, masks)])

    it = 0
    try:
        # Start: every edge velocity must be positive. A negative edge velocity
        # next to the stagnation point means that the coupling moves the
        # stagnation point: the structure is updated first; omega is halved only
        # when that does not help.
        for _ in range(20):
            ue_inv, D, X, re_b, masks, _, _, S = assemble(active, refresh=True)
            ok = edge_ok(be.host(edge(ue_inv, D, X, S)), masks)
            if np.all(ok):
                break
            for k, i in enumerate(active):
                if not ok[k] and not _update_structure(setups[i], states[i], omega[i], transition=False):
                    omega[i] *= 0.5
        for it in range(max_iter + 1):
            if not active:
                break
            if it > 0 and np.any(omega[active] < 1.0):
                keep = omega.copy()
                for i in active:
                    # A cold start can stall at small omega where the boundary layer
                    # separates: the nearly direct problem has no solution there
                    # (Goldstein singularity). The coupling then grows anyway.
                    if last_res[i] < OMEGA_STEP_RESIDUAL or it - best_it[i] > OMEGA_STALL:
                        omega[i] = min(1.0, 2.0 * omega[i])
                        best_it[i] = it
                        best_res[i] = np.inf
                for _ in range(3):
                    ue_inv, D, X, re_b, masks, _, _, S = assemble(active, refresh=True)
                    ok = edge_ok(be.host(edge(ue_inv, D, X, S)), masks)
                    moved = False
                    for k, i in enumerate(active):
                        if not ok[k] and omega[i] != keep[i]:
                            moved = _update_structure(setups[i], states[i], omega[i], transition=False) or moved
                    if not moved:
                        break
                for k, i in enumerate(active):
                    if not ok[k]:
                        omega[i] = keep[i]
                    if keep[i] < 1.0:
                        # omega or the structure can change here: the failed-step count starts again.
                        no_step[i] = 0
            changed = np.zeros(len(active), dtype=bool)
            if it > 0:
                ue_inv, D, X, re_b, masks, _, _, S = assemble(active, refresh=False)
                UE0, _ = _edge(be, ue_inv, D, X, S)
                x2 = X[:, :, 1]
                # A dropped stagnation node has no edge velocity (zero) and no
                # mass defect: divide by 1 there, as host_edge does.
                mk = be.arr(masks.astype(float))
                ds0 = S * x2 + (1.0 - S) * x2 / (UE0 * mk + (1.0 - mk))
                mfull = S * UE0 * x2 + (1.0 - S) * x2
                Bm = be.arr(np.stack([mats[i][1] for i in active]))
                sens_b = be.stack([sens[i][0] for i in active])
                base_b = be.stack([sens[i][1] for i in active])
                om = be.arr(omega[active])
                nn = setups[active[0]].n
                gam0 = base_b[:, :nn] + om[:, None] * (sens_b[:, :nn] @ (Bm @ mfull[:, :, None]))[:, :, 0]
                pre_ue, pre_ds, pre_gam = be.host(UE0), be.host(ds0), be.host(gam0)
                for k, i in enumerate(active):
                    # The transition intervals move only when the equations of the
                    # current structure are nearly solved (or the iteration stalls),
                    # so that each move compares converged states.
                    stalled = it - best_it[i] > STALL_ITERATIONS
                    allow = TRANSITION_EVERY_ITERATION or last_res[i] < STRUCTURE_RESIDUAL or stalled
                    changed[k] = _update_structure(setups[i], states[i], omega[i], transition=allow,
                                                   pre=(pre_ue[k], pre_ds[k], pre_gam[k]))
                    if allow and omega[i] == 1.0:
                        changed[k] = _freeze_cycles(history[i], setups[i], states[i]) or changed[k]
                    if changed[k]:
                        best_it[i] = it
                        best_res[i] = np.inf
            ue_inv, D, X, re_b, masks, drop_b, drop_g, S = assemble(active, refresh=bool(np.any(changed)) or it == 0)
            lists = _build_lists([setups[i] for i in active], [states[i] for i in active])
            R, J, UE = _residual_m(be, lists, X, ue_inv, D, re_b, S, drop_b, drop_g, True)
            R_h = be.host(R)
            ue_h = be.host(UE)
            res = np.max(np.abs(R_h), axis=1)
            res = np.where(np.isfinite(res), res, np.inf)
            if verbose:
                for k, i in enumerate(active):
                    arg = int(np.argmax(np.abs(np.where(np.isfinite(R_h[k]), R_h[k], 1e300))))
                    print(f"it {it} case {i} res {res[k]:.3e} at station {arg // 3} eq {arg % 3} "
                          f"omega {omega[i]:g} i_s {states[i].i_s} drop {states[i].drop} itr {states[i].itr} "
                          f"changed {bool(changed[k])}")
            done = []
            for k, i in enumerate(active):
                last_res[i] = res[k]
                if res[k] < 0.5 * best_res[i]:
                    best_res[i] = res[k]
                    best_it[i] = it
                # Stop-rule bookkeeping: the stagnation window starts again when
                # omega changes, and at omega = 1 also when the structure changes
                # (a different discrete problem). Below omega = 1 the transition
                # intervals are not frozen and can alternate without end, so a
                # structure change does not restart the window there.
                if omega[i] != level_omega[i] or (changed[k] and omega[i] == 1.0):
                    level_min[i] = []
                    level_omega[i] = omega[i]
                    no_step[i] = 0
                lm = level_min[i]
                lm.append(min(float(res[k]), lm[-1]) if lm else float(res[k]))
                at_one = omega[i] == 1.0
                stop = _early_stop(float(res[k]), lm, smallest[i], int(no_step[i]), at_one, stall_window, stall_factor,
                                   divergence_factor, STALL_FLOOR * tol)
                if at_one:
                    smallest[i] = min(smallest[i], float(res[k]))
                if omega[i] == 1.0 and res[k] < tol and not changed[k] and not changed_prev[i]:
                    results[i] = NewtonResult(states[i], ue_h[k], True, it, float(res[k]), "converged",
                                              int(n_lin[i]), "converged")
                    done.append(k)
                elif not np.isfinite(res[k]):
                    results[i] = NewtonResult(states[i], ue_h[k], False, it, float(res[k]),
                                              "the residual is not finite (a closure left its range)", int(n_lin[i]),
                                              "not_finite")
                    done.append(k)
                elif stop is not None and it < max_iter:
                    results[i] = NewtonResult(states[i], ue_h[k], False, it, float(res[k]), stop[1], int(n_lin[i]),
                                              stop[0])
                    done.append(k)
                elif it == max_iter:
                    results[i] = NewtonResult(states[i], ue_h[k], False, it, float(res[k]),
                                              f"no convergence to tol={tol:g} in {max_iter} Newton iterations "
                                              f"(largest residual {res[k]:.3e})", int(n_lin[i]), "max_iter")
                    done.append(k)
                changed_prev[i] = changed[k]
            keep = [k for k in range(len(active)) if k not in done]
            if not keep:
                break
            if len(keep) < len(active):
                kk = be.arr(np.array(keep), True)
                R, J = R[kk], J[kk]
                active = [active[k] for k in keep]
                ue_inv, D, X, re_b, masks, drop_b, drop_g, S = assemble(active, refresh=False)
                lists = _build_lists([setups[i] for i in active], [states[i] for i in active])
            dx_h, singular = _newton_steps(be, J, R)
            dx_h = dx_h.reshape(len(active), -1, 3)
            res_k = res[keep]
            ue_k = ue_h[keep]
            X_h = be.host(X)
            ui_h = be.host(ue_inv)
            D_h = be.host(D)
            S_h = be.host(S)
            lam = np.ones(len(active))
            for k, i in enumerate(active):
                # A singular Newton matrix gives no direction (zero step, no line search).
                lam[k] = 0.0 if singular[k] else _limit_step(setups[i], states[i], X_h[k], dx_h[k], ui_h[k], D_h[k],
                                                             S_h[k], masks[k])
            r0 = np.sqrt(np.sum(be.host(R) ** 2, axis=1))
            accept = np.zeros(len(active), dtype=bool)
            for _ in range(20):
                if np.all(accept):
                    break
                Xt = be.arr(X_h + lam[:, None, None] * dx_h)
                Rt, _, UEt = _residual_m(be, lists, Xt, ue_inv, D, re_b, S, drop_b, drop_g, False)
                rn = np.sqrt(np.sum(be.host(Rt) ** 2, axis=1))
                ok = np.isfinite(rn) & (rn <= (1.0 - 1e-4 * lam) * r0) & edge_ok(be.host(UEt), masks)
                for k in range(len(active)):
                    if not accept[k]:
                        if ok[k]:
                            accept[k] = True
                        else:
                            lam[k] *= 0.5
            J_h = None
            for k, i in enumerate(active):
                n_lin[i] += 1
                if accept[k] and lam[k] >= LM_SWITCH and not singular[k]:
                    states[i].x = X_h[k] + lam[k] * dx_h[k]
                    lm_mu[i] = max(lm_mu[i] * 0.1, 1e-12)
                    no_step[i] = 0
                else:
                    # The Newton direction makes little progress (a near-singular
                    # Jacobian): Levenberg-Marquardt steps instead.
                    if J_h is None:
                        J_h = be.host(J)
                        R_hk = be.host(R)
                    xn, lm_mu[i], n_lm = _lm_step(be, setups[i], states[i], X_h[k], J_h[k], R_hk[k], r0[k], ui_h[k],
                                                  D_h[k], S_h[k], masks[k], lm_mu[i],
                                                  lm_trials if lm_failed[i] else LM_TRIALS)
                    n_lin[i] += n_lm
                    # Near the tolerance (round-off level of float32) a failed step does not set the cap.
                    lm_failed[i] = lm_failed[i] or (xn is None and res_k[k] > STALL_FLOOR * tol)
                    if xn is not None:
                        states[i].x = xn
                        no_step[i] = 0
                    elif accept[k] and not singular[k]:
                        states[i].x = X_h[k] + lam[k] * dx_h[k]
                        no_step[i] = 0
                    else:
                        no_step[i] += 1
                        if singular[k] and omega[i] == 1.0:
                            # No Newton step exists and the fallback fails: the
                            # next iterations would repeat this one exactly.
                            results[i] = NewtonResult(
                                states[i], ue_k[k], False, it, float(res_k[k]),
                                f"singular: the Newton matrix is singular and the Levenberg-Marquardt fallback does "
                                f"not reduce the residual (largest residual {res_k[k]:.3e})", int(n_lin[i]), "singular")
                if verbose:
                    print(f"   case {i} step {lam[k]:.3e} accepted {bool(accept[k])} mu {lm_mu[i]:.1e}")
            active = [i for i in active if results[i] is None]
    except be.singular_errors as exc:
        # A singular linear system outside the Newton step (edge velocities,
        # structure update): the cases that are not finished stop here.
        for i in active:
            if results[i] is None:
                results[i] = NewtonResult(states[i], np.zeros(setups[i].n_st), False, it, float(last_res[i]),
                                          f"singular: a linear system of the solve is singular ({exc})",
                                          int(n_lin[i]), "singular")
    for i in range(nb):
        if results[i] is None:
            results[i] = NewtonResult(states[i], np.zeros(setups[i].n_st), False, max_iter, float(last_res[i]),
                                      "no convergence", int(n_lin[i]), "max_iter")
    return results


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def case_outputs(setup: CaseSetup, st: CaseState, ue: np.ndarray, x_ref=None) -> dict:
    """Return the loads and the distributions of a solved case.

    Parameters
    ----------
    setup : CaseSetup
        Case data.
    st : CaseState
        Final state.
    ue : np.ndarray
        Station edge velocities [-].
    x_ref : tuple of float, optional
        Moment reference point in the body frame [m]. Default: the
        quarter-chord point of the reference chord (``setup.sys.x_ref``).

    Returns
    -------
    dict
        Cl, Cl_pressure, Cd (Squire-Young at the wake end), Cd_friction,
        Cd_pressure, Cm, transition x/c per side, separation flags (Cf < 0
        at a station of the side) and node and wake distributions.
    """
    from ventorum.panel2d.solver import _loads

    s = setup.sys
    n = setup.n
    c = s.chord_ref
    re = setup.re_l
    if x_ref is None:
        x_ref = s.x_ref
    gam = _gamma_total(setup, st)
    alpha = None if s.ground else np.array([np.radians(s.alpha)])
    cl_, clp, cdp, cm, _, cp = _loads(np, gam[None, :], s.nodes, s.nodes_ref, s.layout, c, x_ref, alpha, s.ground)
    up, lo = _sides(setup, st)
    mask = _station_mask(setup, st)
    th = st.x[:n, 0].copy()
    _, ds_all, _ = host_edge(setup, st)
    ds = np.where(mask[:n], ds_all[:n], np.nan)
    H = ds / th
    cf = np.full(n, np.nan)
    ctau = np.full(n, np.nan)
    amp = np.full(n, np.nan)
    xtr = [1.0, 1.0]
    sep = [False, False]
    cdf = 0.0
    if st.drop >= 0:
        p0 = s.nodes[st.drop]
    else:
        w0 = gam[st.i_s] / (gam[st.i_s] - gam[st.i_s + 1])
        p0 = s.nodes[st.i_s] + w0 * (s.nodes[st.i_s + 1] - s.nodes[st.i_s])
    for side, g in enumerate((up, lo)):
        itr = st.itr[side]
        m = len(g)
        lam = np.arange(m) <= itr
        rt = ue[g] * th[g] * re
        cfl = 2.0 * cl.laminar_cf_factor(H[g])[0] / rt
        cft = cl.turbulent_cf(H[g], np.maximum(rt, 1.5))[0]
        cf[g] = np.where(lam, cfl, cft)
        ctau[g] = np.where(lam, np.nan, st.x[g, 2] ** 2)
        amp[g] = np.where(lam, st.x[g, 2], np.nan)
        sep[side] = bool(np.any(cf[g] < 0.0))
        if itr < m - 1:
            dxi, xoc = _side_geometry(setup, g)
            r1 = float(_laminar_rate(setup, th[g[itr]], ds[g[itr]], ue[g[itr]]))
            w = (setup.ncrit - st.x[g[itr], 2]) / (dxi[itr] * r1) if r1 > 0.0 else 2.0
            jf, wfv = _forced_interval(setup, side, g)
            if jf is not None and jf == itr:
                w = min(w, wfv)
            w = min(max(w, 0.0), 1.0)
            xtr[side] = float(xoc[itr] + w * (xoc[itr + 1] - xoc[itr]))
        tau = cf[g] * ue[g] ** 2
        pts = s.nodes[g]
        seg = pts[1:] - pts[:-1]
        cdf += float(np.sum(0.5 * (tau[1:] + tau[:-1]) * (seg @ s.e_drag))) / c
        cdf += float(0.5 * tau[0] * ((pts[0] - p0) @ s.e_drag)) / c
    wk = slice(n, setup.n_st)
    thw = st.x[wk, 0]
    uew = ue[wk]
    hw = st.x[wk, 1] / (uew * thw)
    cd_sy = float(2.0 * thw[-1] * uew[-1] ** (0.5 * (hw[-1] + 5.0)) / c)
    return {
        "Cl": float(cl_[0]),
        "Cl_pressure": float(clp[0]),
        "Cd": cd_sy,
        "Cd_friction": cdf,
        "Cd_pressure": float(cdp[0]),
        "Cm": float(cm[0]),
        "xtr": (xtr[0], xtr[1]),
        "separated": (sep[0], sep[1]),
        "Cp": cp[0],
        "H": H,
        "theta": th,
        "delta_star": ds,
        "Cf": cf,
        "Ctau": ctau,
        "N": amp,
        "ue": np.abs(gam),
        "gamma": gam,
        "wake_x": s.wake_nodes,
        "wake_theta": thw,
        "wake_H": hw,
        "wake_ue": uew,
    }
