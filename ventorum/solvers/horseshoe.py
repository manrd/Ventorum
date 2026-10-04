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
from ventorum.solvers.core import SolveInfo, solve_vlm
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
        if lattice.has_tabulated:
            warnings.warn(
                "The VLM uses the linear part of the tabulated polars (lift slope and zero-lift "
                "angle from a fit). Stall is not modelled; for the start of stall on an unswept wing use solver='nonlinear'.",
                RuntimeWarning,
                stacklevel=4,
            )
        gamma, info = solve_vlm(lattice, condition, ground,
                                use_symmetry=getattr(settings, "use_symmetry", True), wake_dir=wake_dir)
        return gamma, None, info


# Clearer name for the same solver.
VortexLatticeSolver = HorseshoeSolver
