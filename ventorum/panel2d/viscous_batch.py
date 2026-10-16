# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Batched Ventorum-2D viscous solves on the CPU or on a CUDA GPU (torch).

The batch solver runs the same algorithm as the sequential solver
:func:`ventorum.panel2d.viscous.solve_viscous`, with the same residual and
Jacobian code (:mod:`ventorum.panel2d.vii` and
:mod:`ventorum.ibl2d.equations`):

* Cases with the same geometry, Reynolds number, height and transition
  settings form a sequence; its angles are solved in the order of the
  sequential solver (the angles >= 0 upwards, then the angles < 0 downwards,
  each from the converged state of the previous angle).
* Pass p takes the p-th angle of every sequence. Within a pass, the cases
  with the same node and wake counts are solved together: one batched
  Newton iteration whose residuals, Jacobians, edge velocities, mass-defect
  influence matrices and linear solves (``torch.linalg.solve`` on the batch)
  run on the chosen device. Converged cases leave the active set.
* A case whose first attempt fails continues with the fallback steps of
  the sequential solver (halved angle steps, then a cold start), one case
  at a time on the same device. The first angle of each sequence has no
  converged neighbour; it follows the cold path of the sequential solver
  (continuation from the zero-lift angle), one case at a time.

Host work per case: the inviscid setup of each angle (panel matrices and
their factorization, numpy float64), the first estimate of a cold start
(direct march, numpy float64), the bookkeeping of the structure
(stagnation point and transition intervals, from vectors copied from the
device) and the rare Levenberg-Marquardt fallback steps.

Precision: float64 gives the results of the sequential solver. float32 is
allowed on the device; the residual of the equations then stops near the
float32 round-off level, and a case that does not reach the tolerance is
reported with ``converged=False`` and its reached residual (there is no
switch to float64).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ventorum.panel2d import coupling as cp
from ventorum.panel2d import vii
from ventorum.panel2d import viscous as vs
from ventorum.utils.parallel import blas_single_thread

__all__ = ["ViscousCase", "ViscousBatchResult", "solve_viscous_batch"]


@dataclass
class ViscousCase:
    """One case of the batched viscous solver.

    Attributes
    ----------
    coordinates : np.ndarray
        Airfoil coordinates in Selig order, shape (N, 2) [m or chord units].
    alpha : float
        Angle of attack [deg].
    Re : float
        Reynolds number of the chord [-].
    mach : float
        Mach number [-]; only 0 is supported.
    ncrit : float
        Critical amplification exponent [-].
    xtr : tuple of float
        Forced transition x/c on the upper and lower side [-] (1.0: free).
    height : float or None
        Height above the ground [m], or None for free air.
    max_iter : int
        Largest number of Newton iterations.
    tol : float
        Convergence limit of the largest absolute residual [-].
    wake_length : float
        Wake length over the chord [-].
    n_wake : int
        Number of wake panels.
    """

    coordinates: np.ndarray
    alpha: float
    Re: float
    mach: float = 0.0
    ncrit: float = 9.0
    xtr: tuple = (1.0, 1.0)
    height: float | None = None
    max_iter: int = 50
    tol: float = 1e-6
    wake_length: float = 1.0
    n_wake: int = 30


class ViscousBatchResult(list):
    """List of :class:`ventorum.panel2d.viscous.ViscousResult`, one per case.

    The properties return arrays over the cases, in input order.
    """

    def _arr(self, name, dtype=float):
        return np.array([getattr(r, name) for r in self], dtype=dtype)

    @property
    def alpha(self) -> np.ndarray:
        """Return the angles of attack [deg]."""
        return self._arr("alpha")

    @property
    def Cl(self) -> np.ndarray:
        """Return the lift coefficients [-]."""
        return self._arr("Cl")

    @property
    def Cd(self) -> np.ndarray:
        """Return the Squire-Young drag coefficients [-]."""
        return self._arr("Cd")

    @property
    def Cd_friction(self) -> np.ndarray:
        """Return the friction drag coefficients [-]."""
        return self._arr("Cd_friction")

    @property
    def Cd_pressure(self) -> np.ndarray:
        """Return the pressure drag coefficients [-]."""
        return self._arr("Cd_pressure")

    @property
    def Cm(self) -> np.ndarray:
        """Return the pitching moment coefficients [-]."""
        return self._arr("Cm")

    @property
    def converged(self) -> np.ndarray:
        """Return the convergence flags."""
        return self._arr("converged", bool)

    @property
    def residual(self) -> np.ndarray:
        """Return the final residuals [-]."""
        return self._arr("residual")

    @property
    def iterations(self) -> np.ndarray:
        """Return the Newton iteration counts."""
        return self._arr("iterations", int)

    @property
    def transition_positions(self) -> list:
        """Return the transition x/c pairs [-]."""
        return [r.transition_positions for r in self]

    @property
    def separation_flags(self) -> list:
        """Return the separation flag pairs."""
        return [r.separation_flags for r in self]

    @property
    def reason(self) -> list:
        """Return the convergence status texts."""
        return [r.reason for r in self]


def _case(c) -> ViscousCase:
    """Return a ViscousCase from a ViscousCase, a dict or an object with the same fields."""
    if isinstance(c, ViscousCase):
        out = c
    elif isinstance(c, dict):
        out = ViscousCase(**c)
    else:
        out = ViscousCase(**{k: getattr(c, k) for k in ViscousCase.__dataclass_fields__ if hasattr(c, k)})
    if abs(out.mach) > 1e-12:
        raise ValueError(f"Only mach=0.0 is supported, got mach={out.mach}.")
    if not np.isfinite(out.Re) or out.Re <= 0.0:
        raise ValueError(f"Reynolds number must be positive, got {out.Re}.")
    return out


def _resolve(device, dtype):
    """Return the torch device and dtype (CUDA when available, else the CPU; float64 by default)."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if dtype is None:
        dtype = torch.float64
    elif isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    return device, dtype


def _seq_key(c: ViscousCase):
    """Return the key of the sequence of a case."""
    coords = np.ascontiguousarray(np.asarray(c.coordinates, dtype=np.float64))
    xtr = (float(c.xtr[0]), float(c.xtr[1]))
    return (coords.tobytes(), coords.shape, float(c.Re), c.height, float(c.ncrit), xtr, float(c.wake_length),
            int(c.n_wake), int(c.max_iter), float(c.tol))


def solve_viscous_batch(cases, *, backend: str = "torch", device=None, dtype=None) -> ViscousBatchResult:
    """Solve many viscous cases with batched Newton iterations (CPU or CUDA).

    Parameters
    ----------
    cases : list of ViscousCase or dict
        The cases (geometry, angle, Reynolds number, height, transition and
        solver settings).
    backend : {"torch"}
        Array backend.
    device : str or torch.device, optional
        "cpu", "cuda" or "cuda:k". Default: CUDA when available, else the
        CPU.
    dtype : torch.dtype or str, optional
        torch.float64 (default) or torch.float32.

    Returns
    -------
    ViscousBatchResult
        One :class:`ventorum.panel2d.viscous.ViscousResult` per case, in
        input order.
    """
    if backend != "torch":
        raise ValueError(f"backend '{backend}' is not supported; solve_viscous_batch runs on 'torch'.")
    parsed = [_case(c) for c in cases]
    if not parsed:
        return ViscousBatchResult([])
    dev, dt = _resolve(device, dtype)
    bk = {"xp": torch, "device": dev, "dtype": dt}
    seqs: dict = {}
    for i, c in enumerate(parsed):
        seqs.setdefault(_seq_key(c), []).append(i)
    plans = []
    for idx in seqs.values():
        c0 = parsed[idx[0]]
        up, dn = vs.sweep_order([parsed[i].alpha for i in idx])
        kw = dict(ncrit=c0.ncrit, xtr=tuple(c0.xtr), height=c0.height, height_reference="pivot",
                  wake_length=c0.wake_length, n_wake=c0.n_wake)
        body = vs._single_body(np.asarray(c0.coordinates, dtype=np.float64))
        plans.append({"idx": idx, "up": up, "dn": dn, "kw": kw, "body": body, "case": c0, "prev": None,
                      "first": None, "rows": {}})
    with blas_single_thread():
        n_up = max(len(p["up"]) for p in plans)
        for k in range(n_up):
            _pass([(p, p["up"][k]) for p in plans if k < len(p["up"])], bk, upward=True)
        for p in plans:
            p["prev"] = p["first"]
        n_dn = max(len(p["dn"]) for p in plans)
        for k in range(n_dn):
            _pass([(p, p["dn"][k]) for p in plans if k < len(p["dn"])], bk, upward=False)
    out = [None] * len(parsed)
    for p in plans:
        for i in p["idx"]:
            r = p["rows"][float(parsed[i].alpha)]
            out[i] = vs.ViscousResult(**{f: r[f] for f in vs._FIELDS})
    return ViscousBatchResult(out)


def _pass(items, bk, upward):
    """Solve one angle of each sequence: a batched first attempt, then the fallbacks per case."""
    work = []
    for plan, alpha in items:
        c = plan["case"]
        if plan["prev"] is None:
            # No converged neighbour: the cold path of the sequential solver, one case at a time.
            row, state = vs.finish_angle(plan["body"], alpha, c.Re, None, c.tol, c.max_iter, plan["kw"], bk)
            _store(plan, alpha, row, state, upward)
            continue
        try:
            setup = vs.prepare_case(plan["body"], alpha, c.Re, **plan["kw"])
        except cp.WakeGroundError as exc:
            plan["rows"][float(alpha)] = vs._failed_row(alpha, plan["body"], str(exc))
            continue
        start = vs.warm_state(setup, plan["prev"][0], plan["prev"][1])
        work.append((plan, alpha, setup, start))
    groups: dict = {}
    for w in work:
        key = (w[2].n, w[2].nw, w[0]["case"].tol, w[0]["case"].max_iter)
        groups.setdefault(key, []).append(w)
    for (_, _, tol, max_iter), grp in groups.items():
        results = vii.newton_solve([w[2] for w in grp], [w[3] for w in grp], tol=tol, max_iter=max_iter,
                                   omega0=[1.0] * len(grp), **bk)
        for w, res in zip(grp, results):
            plan, alpha, setup, _ = w
            c = plan["case"]
            row, state = vs.finish_angle(plan["body"], alpha, c.Re, plan["prev"], tol, max_iter, plan["kw"], bk,
                                         first=(setup, res))
            _store(plan, alpha, row, state, upward)


def _store(plan, alpha, row, state, upward):
    """Store the result of one angle and keep a converged state as the next start."""
    plan["rows"][float(alpha)] = row
    if row["converged"]:
        plan["prev"] = state
        if upward and plan["first"] is None:
            plan["first"] = state
