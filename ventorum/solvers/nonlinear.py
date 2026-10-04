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

from ventorum.aero.system import GroundPlane
from ventorum.core.datatypes import FlightCondition, SolverSettings
from ventorum.geometry.lattice import VortexLattice
from ventorum.solvers.core import SolveInfo, solve_llt_linear, solve_llt_nonlinear
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
    ) -> tuple[np.ndarray, np.ndarray, SolveInfo]:
        """Solve for the panel circulation with Newton's method.

        If the solve does not converge, the method starts again from the
        linear solution and from scaled linear solutions. It keeps the
        result with the smallest residual.

        Returns
        -------
        gamma : numpy.ndarray
            Circulation of each panel [m^2/s].
        alpha_eff : numpy.ndarray
            Effective angle of attack of each strip [rad].
        info : SolveInfo
            Convergence information.
        """
        kw = dict(
            use_symmetry=getattr(settings, "use_symmetry", True),
            max_iterations=int(settings.max_iterations),
            tolerance=float(settings.tolerance),
            wake_dir=wake_dir,
        )
        gamma, alpha_eff, info = solve_llt_nonlinear(lattice, condition, ground, gamma0=gamma0, **kw)
        if not info.converged:
            # Past the maximum lift Newton's method can stop in a local
            # minimum of the residual. Start again from the linear solution
            # (if a start value was given) and from scaled linear solutions,
            # and keep the best result.
            g_lin, _ = solve_llt_linear(lattice, condition, ground, kw["use_symmetry"], wake_dir)
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
