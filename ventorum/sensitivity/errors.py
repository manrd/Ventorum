# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Errors of the adjoint gradients.

This module imports nothing heavy, so that other modules can catch
:class:`GradientScopeError` without loading PyTorch.
"""


class GradientScopeError(ValueError):
    """A configuration that the adjoint gradients do not cover.

    The Fourier solver (a closed form), a nonlinear lifting-line solve that
    did not converge, and the planform input with tabulated airfoils have
    no adjoint gradients. Callers can catch this error to use another
    method (for example central differences of the production solve).
    """
