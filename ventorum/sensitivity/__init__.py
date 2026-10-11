# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Adjoint gradients of the lattice solvers (phase 1).

:func:`gradients` gives exact gradients of the scalar outputs with respect
to the lattice nodes, the flow variables, the linear section properties
and the reference values, for the vortex lattice and the linear lifting
line. :func:`gradient_vjp` covers the vector outputs. :func:`check_gradients`
checks the gradients against Richardson finite differences.

The gradient functions load PyTorch. This package loads them on first
access (PEP 562), so that ``import ventorum`` does not load PyTorch.
"""

from ventorum.sensitivity.errors import GradientScopeError

__all__ = [
    "GradientResult",
    "GradientScopeError",
    "gradients",
    "gradient_vjp",
    "check_gradients",
]

# Public name -> module that defines it (loaded on first access).
_LAZY = {
    "GradientResult": "ventorum.sensitivity.adjoint",
    "gradients": "ventorum.sensitivity.adjoint",
    "gradient_vjp": "ventorum.sensitivity.adjoint",
    "check_gradients": "ventorum.sensitivity.check",
}


def __getattr__(name: str):
    """Return a gradient function or class, and import its module on first access."""
    if name in _LAZY:
        import importlib

        value = getattr(importlib.import_module(_LAZY[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return the module names, with the lazy names included."""
    return sorted(set(globals()) | set(__all__))
