# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Solvers: vortex-lattice method, numerical lifting line and Fourier lifting line."""

from ventorum.solvers.base import BaseSolver
from ventorum.solvers.fourier import FourierSolver
from ventorum.solvers.horseshoe import VortexLatticeSolver
from ventorum.solvers.linear import LinearLLTSolver
from ventorum.solvers.nonlinear import NonlinearLLTSolver

__all__ = [
    "BaseSolver",
    "FourierSolver",
    "VortexLatticeSolver",
    "LinearLLTSolver",
    "NonlinearLLTSolver",
]


def __getattr__(name: str):
    """Return an old solver class name with a deprecation warning."""
    if name == "HorseshoeSolver":
        from ventorum.solvers.horseshoe import VortexLatticeSolver
        from ventorum.utils.deprecation import warn_deprecated_class

        warn_deprecated_class("HorseshoeSolver", "VortexLatticeSolver")
        return VortexLatticeSolver
    if name == "LinearSolver":
        from ventorum.solvers.linear import LinearLLTSolver
        from ventorum.utils.deprecation import warn_deprecated_class

        warn_deprecated_class("LinearSolver", "LinearLLTSolver")
        return LinearLLTSolver
    if name == "NonlinearSolver":
        from ventorum.solvers.nonlinear import NonlinearLLTSolver
        from ventorum.utils.deprecation import warn_deprecated_class

        warn_deprecated_class("NonlinearSolver", "NonlinearLLTSolver")
        return NonlinearLLTSolver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
