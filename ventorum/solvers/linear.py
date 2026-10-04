# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Linear numerical lifting line (Phillips & Snyder 2000).

One horseshoe per spanwise strip, on the quarter-chord line. At a point on
each bound vortex the Kutta-Joukowski force is set equal to the section lift
``a0 (alpha - alpha_L0)``::

    [2 |u x dl_i| / (a0_i dA_i)] G_i - sum_j (v_ij . n_i) G_j = V (u . n_i - alpha_L0_i)

For straight (unswept) lifting lines this is the Lanchester–Prandtl lifting-line theory
for arbitrary planform, twist, camber, dihedral and several surfaces. With
cosine spacing and control points at the mid parameter it gives ``e = 1``
for the elliptic wing with 10 to 20 panels per semi-span.

Limitation: with sweep the method is not grid convergent (the lift falls as
panels are added, because of the kink of the lifting line at the root).
Use the vortex-lattice solver for swept wings.

Reference: W. F. Phillips and D. O. Snyder, "Modern adaptation of Prandtl's
classic lifting-line theory", Journal of Aircraft 37(4), 2000, pp. 662-670.
"""

from __future__ import annotations

import numpy as np

from ventorum.aero.system import GroundPlane
from ventorum.core.datatypes import FlightCondition, SolverSettings
from ventorum.geometry.lattice import VortexLattice
from ventorum.solvers.core import SolveInfo, solve_llt_linear, solve_llt_linear_batch
from ventorum.solvers.lattice_base import LatticeSolver


class LinearLLTSolver(LatticeSolver):
    """Linear numerical lifting line (Phillips & Snyder)."""

    collocation = "llt"
    name = "linear"

    def solve_circulation(
        self,
        lattice: VortexLattice,
        condition: FlightCondition,
        settings: SolverSettings,
        ground: GroundPlane | None,
        wake_dir: np.ndarray | None = None,
        gamma0: np.ndarray | None = None,
    ) -> tuple[np.ndarray, None, SolveInfo]:
        """Solve for the panel circulation.

        Returns
        -------
        gamma : numpy.ndarray
            Circulation of each panel [m^2/s].
        alpha_eff : None
            This solver gives no effective angle of attack.
        info : SolveInfo
            Convergence information.
        """
        gamma, info = solve_llt_linear(lattice, condition, ground,
                                       use_symmetry=getattr(settings, "use_symmetry", True), wake_dir=wake_dir)
        return gamma, None, info

    def solve_circulation_batch(
        self,
        lattice: VortexLattice,
        conditions: list[FlightCondition],
        settings: SolverSettings,
        grounds: list[GroundPlane | None],
        wake_dirs: np.ndarray,
        continuation: bool = True,
    ) -> list[tuple[np.ndarray, None, SolveInfo]]:
        """Solve all cases together: one kernel call and one call of the dense solver.

        Each case gets the same circulation, to the last bit, as
        :meth:`solve_circulation`. *continuation* has no effect (the system
        is linear).
        """
        sols = solve_llt_linear_batch(lattice, conditions, grounds,
                                      getattr(settings, "use_symmetry", True), wake_dirs)
        if sols is None:
            return super().solve_circulation_batch(lattice, conditions, settings, grounds, wake_dirs, False)
        return [(g, None, info) for g, info in sols]


# Alias for convenience
LinearSolver = LinearLLTSolver
