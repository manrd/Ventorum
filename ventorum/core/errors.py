# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Exceptions with a clear meaning for callers (scripts, sweeps, agents)."""


class VentorumError(ValueError):
    """Base class of the package errors (a ValueError for backward compatibility)."""


class GroundStrikeError(VentorumError):
    """The aircraft touches or crosses the ground plane; no result is computed."""


class ValidityError(VentorumError):
    """The selected method is not valid for this case (for example a lifting line too near the ground)."""
