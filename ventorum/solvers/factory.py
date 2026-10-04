# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Solver selection by name."""

from __future__ import annotations

_ALIASES = {
    "vlm": "vlm",
    "horseshoe": "vlm",
    "lattice": "vlm",
    "linear": "linear",
    "linear_llt": "linear",
    "llt": "linear",
    "nonlinear": "nonlinear",
    "fourier": "fourier",
}

VALID_SOLVER_NAMES = ("auto",) + tuple(_ALIASES)


def resolve_solver_type(solver_type: str | None) -> str:
    """Canonical solver name: ``"vlm"``, ``"linear"``, ``"nonlinear"`` or ``"fourier"``.

    ``"auto"`` (or *None*) always gives the vortex-lattice method, so the
    method does not change when a section changes from a linear to a
    tabulated airfoil. The VLM uses the linear part of a tabulated polar and
    gives a warning; for the start of stall on an unswept wing select
    ``"nonlinear"`` explicitly.
    """
    s = (solver_type or "auto").lower()
    if s == "auto":
        return "vlm"
    if s not in _ALIASES:
        raise ValueError(f"Unknown solver {solver_type!r}. Options: {', '.join(VALID_SOLVER_NAMES)}.")
    return _ALIASES[s]


def make_solver(canonical: str):
    """Instance of the solver class for a canonical name."""
    if canonical == "vlm":
        from ventorum.solvers.horseshoe import HorseshoeSolver
        return HorseshoeSolver()
    if canonical == "linear":
        from ventorum.solvers.linear import LinearLLTSolver
        return LinearLLTSolver()
    if canonical == "nonlinear":
        from ventorum.solvers.nonlinear import NonlinearSolver
        return NonlinearSolver()
    if canonical == "fourier":
        from ventorum.solvers.fourier import FourierSolver
        return FourierSolver()
    raise ValueError(f"Unknown canonical solver {canonical!r}.")

