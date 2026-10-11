# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Solvers: vortex-lattice method, numerical lifting line and Fourier lifting line.

The derivative functions (for example ``derivative_matrix``) live in their own
modules of this package, which have the same names; they are exported from
``ventorum`` only, so that the package attribute names the module.
"""

from ventorum.solvers.base import BaseSolver
from ventorum.solvers.fourier import FourierSolver
from ventorum.solvers.horseshoe import VortexLatticeSolver
from ventorum.solvers.linear import LinearLLTSolver
from ventorum.solvers.nonlinear import NonlinearLLTSolver

__all__ = [
    "BaseSolver",
    "FourierSolver",
    "LinearLLTSolver",
    "NonlinearLLTSolver",
    "VortexLatticeSolver",
]


def __getattr__(name: str):
    """Return an old solver class name with a deprecation warning."""
    from ventorum.utils.deprecation import DEPRECATED_CLASS_NAMES, warn_deprecated_class

    if name in DEPRECATED_CLASS_NAMES:
        import ventorum.solvers as solvers

        warn_deprecated_class(name, DEPRECATED_CLASS_NAMES[name])
        return getattr(solvers, DEPRECATED_CLASS_NAMES[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
