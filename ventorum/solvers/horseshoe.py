# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Vortex-lattice method (VLM): the default linear solver.

Each lifting surface is divided into spanwise strips and ``n_chord``
chordwise panels. Each panel carries a horseshoe vortex: the bound vortex on
the 1/4-chord line of the panel, the legs along the strip edges to the
trailing edge, and then into the wake along the free stream. Flow tangency
is enforced at one control point per panel.

What the method models
----------------------
* Arbitrary planform: sweep, taper, dihedral, twist, several surfaces.
* Section lift slope ``a0`` (by the control-point position) and camber
  (by the zero-lift line) of linear section data.
* Ground effect by the image method (ground parallel to the free stream).

What it does not model
----------------------
* Thickness, viscosity and stall: the section lift stays linear. Use the
  nonlinear solver with tabulated polars for stall.
* Compressibility (no Prandtl-Glauert correction).
* Leading-edge vortex lift of slender wings.
"""

from __future__ import annotations

import warnings

import numpy as np

from ventorum.aero.system import GroundPlane
from ventorum.core.datatypes import FlightCondition, SolverSettings
from ventorum.geometry.lattice import VortexLattice
from ventorum.solvers.core import SolveInfo, solve_vlm, solve_vlm_batch
from ventorum.solvers.lattice_base import LatticeSolver


class HorseshoeSolver(LatticeSolver):
    """Vortex-lattice method with linear section data (default solver)."""

    collocation = "vlm"
    name = "vlm"

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
        if _has_tabulated(lattice):
            warnings.warn(
                "The VLM uses the linear part of the tabulated polars (lift slope and zero-lift "
                "angle from a fit). Stall is not modelled; for the start of stall on an unswept wing use solver='nonlinear'.",
                RuntimeWarning,
                stacklevel=4,
            )
        gamma, info = solve_vlm(lattice, condition, ground,
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
        """Solve all cases together: the systems one by one, the dense solves in one call.

        Each case gets the same circulation, to the last bit, as
        :meth:`solve_circulation`. *continuation* has no effect (the system
        is linear).
        """
        if _has_tabulated(lattice):
            warnings.warn(
                "The VLM uses the linear part of the tabulated polars (lift slope and zero-lift "
                "angle from a fit). Stall is not modelled; for the start of stall on an unswept wing use solver='nonlinear'.",
                RuntimeWarning,
                stacklevel=4,
            )
        sols = solve_vlm_batch(lattice, conditions, grounds,
                               getattr(settings, "use_symmetry", True), wake_dirs)
        if sols is None:
            return super().solve_circulation_batch(lattice, conditions, settings, grounds, wake_dirs, False)
        return [(g, None, info) for g, info in sols]


def _has_tabulated(lattice: VortexLattice) -> bool:
    """Return ``lattice.has_tabulated``, cached on the lattice (the test walks every strip)."""
    value = lattice.geom_cache.get("has_tabulated")
    if value is None:
        value = lattice.has_tabulated
        lattice.geom_cache["has_tabulated"] = value
    return value


# Clearer name for the same solver.
VortexLatticeSolver = HorseshoeSolver
