# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Errors of the adjoint gradients.

This module imports nothing heavy, so that other modules can catch
:class:`GradientScopeError` without loading PyTorch.
"""


class GradientScopeError(ValueError):
    """A configuration that the adjoint gradients do not cover yet.

    The nonlinear lifting line, tabulated airfoils, deflected controls, body
    rates and the Fourier solver have no adjoint gradients in this version.
    A later version adds them. Callers can catch this error to use another
    method (for example central differences of the production solve).
    """
