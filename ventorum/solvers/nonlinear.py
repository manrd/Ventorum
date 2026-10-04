# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Nonlinear numerical lifting line with section polars.

The equations are those of the linear lifting line (Phillips & Snyder 2000),
but the section lift comes from the section polar at the local angle of
attack, and the local velocity includes the induced velocity::

    R_i = 2 |W_i x dl_i| G_i - V_inf^2 dA_i Cl_i(alpha_i) = 0
    W_i = V_inf + sum_j v_ij G_j,   alpha_i = atan2(W_i . n_i, W_i . a_i)

The system is solved by Newton's method with a backtracking line search,
starting from the linear solution. Convergence is tested on the error of
the section lift coefficient (``SolverSettings.tolerance``).

The induced velocity is taken on the lifting line (the quarter chord), not
at a 3/4-chord control point. A 3/4-chord point already contains the
two-dimensional effect of the bound vortex, which the section polar also
contains; using it would count that effect twice and give about half the
lift slope.

Limitations
-----------
* The same sweep limitation as the linear lifting line.
* Past the maximum lift the solution may not be unique; the solver reports
  ``converged=False`` when Newton's method does not reach the tolerance.
"""

from __future__ import annotations

import warnings

import numpy as np

from ventorum.aero.system import GroundPlane, freestream_direction
from ventorum.core.datatypes import FlightCondition, SolverSettings
from ventorum.geometry.lattice import VortexLattice
from ventorum.solvers.core import (
    SolveInfo,
    _unknown_map,
    linear_llt_from_tensor,
    llt_velocity_tensors,
    solve_llt_nonlinear,
)
from ventorum.solvers.lattice_base import LatticeSolver


class NonlinearSolver(LatticeSolver):
    """Nonlinear lifting line with tabulated (or linear) section polars."""

    collocation = "llt"
    name = "nonlinear"

    def solve_circulation(
        self,
        lattice: VortexLattice,
        condition: FlightCondition,
        settings: SolverSettings,
        ground: GroundPlane | None,
        wake_dir: np.ndarray | None = None,
        gamma0: np.ndarray | None = None,
        Vt: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, SolveInfo]:
        """Solve for the panel circulation with Newton's method.

        If the solve does not converge, the method starts again from the
        linear solution and from scaled linear solutions. It keeps the
        result with the smallest residual. The velocity tensor per unknown
        (*Vt*, computed here if None) is shared by all starts.

        Returns
        -------
        gamma : numpy.ndarray
            Circulation of each panel [m^2/s].
        alpha_eff : numpy.ndarray
            Effective angle of attack of each strip [rad].
        info : SolveInfo
            Convergence information.
        """
        use_sym = getattr(settings, "use_symmetry", True)
        umap = _unknown_map(lattice, condition, ground, use_sym)
        if wake_dir is None:
            wake_dir = freestream_direction(condition.alpha, condition.beta)
        if Vt is None:
            Vt = llt_velocity_tensors(lattice, umap, np.asarray(wake_dir, dtype=float)[None, :], [ground])[0]
        kw = dict(
            use_symmetry=use_sym,
            max_iterations=int(settings.max_iterations),
            tolerance=float(settings.tolerance),
            wake_dir=wake_dir,
            Vt=Vt,
        )
        gamma, alpha_eff, info = solve_llt_nonlinear(lattice, condition, ground, gamma0=gamma0, **kw)
        if not info.converged:
            # Past the maximum lift Newton's method can stop in a local
            # minimum of the residual. Start again from the linear solution
            # (if a start value was given) and from scaled linear solutions,
            # and keep the best result.
            g_lin = linear_llt_from_tensor(lattice, condition, umap, Vt)[umap.panel_column]
            starts = ([None] if gamma0 is not None else []) + [f * g_lin for f in (0.85, 0.7, 0.55, 1.15)]
            for g_start in starts:
                g2, a2, i2 = solve_llt_nonlinear(lattice, condition, ground, gamma0=g_start, **kw)
                if i2.residual_history[-1] < info.residual_history[-1]:
                    gamma, alpha_eff, info = g2, a2, i2
                if info.converged:
                    break
        if not info.converged:
            last = info.residual_history[-1] if info.residual_history else float("nan")
            warnings.warn(
                f"NonlinearSolver did not converge in {info.iterations} Newton iterations "
                f"(section Cl error {last:.2e}). The result is not reliable; this is common "
                "past the maximum lift.",
                RuntimeWarning,
                stacklevel=4,
            )
        return gamma, alpha_eff, info

    def solve_circulation_batch(
        self,
        lattice: VortexLattice,
        conditions: list[FlightCondition],
        settings: SolverSettings,
        grounds: list[GroundPlane | None],
        wake_dirs: np.ndarray,
        continuation: bool = True,
    ) -> list[tuple[np.ndarray, np.ndarray, SolveInfo]]:
        """Solve the cases in order; one kernel call gives the velocity tensors of all cases.

        Each case gets the same result, to the last bit, as
        :meth:`solve_circulation` with the same start value.
        """
        use_sym = getattr(settings, "use_symmetry", True)
        umaps = [_unknown_map(lattice, c, g, use_sym) for c, g in zip(conditions, grounds)]
        if any(u is not umaps[0] for u in umaps):
            return super().solve_circulation_batch(lattice, conditions, settings, grounds, wake_dirs, continuation)
        Vts = llt_velocity_tensors(lattice, umaps[0], wake_dirs, grounds)
        out = []
        g_prev = None
        for k, (cond, ground) in enumerate(zip(conditions, grounds)):
            sol = self.solve_circulation(lattice, cond, settings, ground, wake_dirs[k],
                                         gamma0=g_prev if continuation else None, Vt=Vts[k])
            g_prev = sol[0] if sol[2].converged else None
            out.append(sol)
        return out
