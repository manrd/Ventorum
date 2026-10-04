# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""LLT solvers for Ventorum."""

from ventorum.legacy.solvers.base import BaseSolver
from ventorum.legacy.solvers.fourier import FourierSolver
from ventorum.legacy.solvers.horseshoe import HorseshoeSolver
from ventorum.legacy.solvers.linear import LinearLLTSolver, LinearSolver
from ventorum.legacy.solvers.nonlinear import NonlinearSolver
from ventorum.legacy.aero.gpu_influence import has_cuda

try:
    from ventorum.legacy.solvers.gpu_horseshoe import GPUHorseshoeSolver
    from ventorum.legacy.solvers.gpu_nonlinear import GPUNonlinearSolver
except ImportError:
    GPUHorseshoeSolver = None
    GPUNonlinearSolver = None

__all__ = [
    "BaseSolver",
    "FourierSolver",
    "HorseshoeSolver",
    "LinearLLTSolver",
    "LinearSolver",
    "NonlinearSolver",
    "GPUHorseshoeSolver",
    "GPUNonlinearSolver",
    "has_cuda",
]

