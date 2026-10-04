# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Solvers: vortex-lattice method, numerical lifting line and Fourier lifting line."""

from ventorum.solvers.base import BaseSolver
from ventorum.solvers.fourier import FourierSolver
from ventorum.solvers.horseshoe import HorseshoeSolver, VortexLatticeSolver
from ventorum.solvers.linear import LinearLLTSolver, LinearSolver
from ventorum.solvers.nonlinear import NonlinearSolver

__all__ = [
    "BaseSolver",
    "FourierSolver",
    "HorseshoeSolver",
    "VortexLatticeSolver",
    "LinearLLTSolver",
    "LinearSolver",
    "NonlinearSolver",
]
