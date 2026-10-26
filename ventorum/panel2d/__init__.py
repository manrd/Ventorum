# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Ventorum-2D: two-dimensional panel method for airfoils, with ground effect."""

from ventorum.panel2d.viscous import solve_viscous, solve_viscous_flapped

#: Revision of the 2D inviscid and viscous solvers. It changes whenever a
#: change of the solvers can change results (polar caches use it in their key).
SOLVER_REVISION = "2026-10-17.3"

__all__ = ["SOLVER_REVISION", "solve_viscous", "solve_viscous_flapped"]
