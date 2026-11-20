# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Exceptions with a clear meaning for callers (scripts, sweeps, agents)."""


class VentorumError(ValueError):
    """Base class of the package errors (a ValueError for backward compatibility)."""


class InputError(VentorumError, ValueError):
    """The user input is not valid: a value out of range, a wrong type or a missing item.

    It is a :class:`VentorumError` and a ``ValueError``, so code that
    catches ``ValueError`` also catches it. The input checks of
    :mod:`ventorum.utils.validation` and the agent tools
    (:mod:`ventorum.agent`, which re-exports this class) raise it.
    """


class GroundStrikeError(VentorumError):
    """The aircraft touches or crosses the ground plane; no result is computed."""


class ValidityError(VentorumError):
    """The selected method is not valid for this case (for example a lifting line too near the ground)."""
