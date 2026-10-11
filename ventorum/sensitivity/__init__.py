# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Adjoint gradients of the lattice solvers (phase 1).

:func:`gradients` gives exact gradients of the scalar outputs with respect
to the lattice nodes, the flow variables, the linear section properties
and the reference values, for the vortex lattice and the linear lifting
line. :func:`gradient_vjp` covers the vector outputs. :func:`check_gradients`
checks the gradients against Richardson finite differences.
"""

from ventorum.sensitivity.adjoint import (
    GradientResult,
    GradientScopeError,
    gradients,
    gradient_vjp,
)
from ventorum.sensitivity.check import check_gradients

__all__ = [
    "GradientResult",
    "GradientScopeError",
    "gradients",
    "gradient_vjp",
    "check_gradients",
]
