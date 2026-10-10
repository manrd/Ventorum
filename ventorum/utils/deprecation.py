# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Deprecation helpers for renamed public API names."""

from __future__ import annotations

import traceback
import warnings

#: Canonical solver names. Use these in new code.
CANONICAL_SOLVERS = ("auto", "vlm", "linear", "nonlinear", "fourier")

#: Old solver names that still work. Each maps to its canonical name.
SOLVER_ALIASES = {
    "horseshoe": "vlm",
    "lattice": "vlm",
    "llt": "linear",
    "linear_llt": "linear",
}


def warn_solver_alias(solver_name: str | None) -> str:
    """Warn once for an old solver name and return the canonical name.

    Parameters
    ----------
    solver_name : str or None
        Solver name given by the user. None means the default.

    Returns
    -------
    str
        Canonical solver name (``"vlm"`` for None and ``"auto"``).
    """
    if solver_name is None:
        return "vlm"
    lowered = str(solver_name).lower()
    if lowered == "auto":
        return "vlm"
    canonical = SOLVER_ALIASES.get(lowered)
    if canonical is None:
        return lowered
    message = (
        f"solver '{solver_name}' is deprecated; use '{canonical}' instead. "
        "The old name still works for now and gives the same result."
    )
    level = 2
    frames = traceback.extract_stack()
    for depth, frame in enumerate(reversed(frames[:-1]), start=2):
        name = frame.filename.replace("\\", "/")
        if "ventorum/utils/deprecation" in name:
            continue
        if "/ventorum/" in name or name.endswith("/ventorum.py"):
            continue
        level = depth
        break
    warnings.warn(message, FutureWarning, stacklevel=level)
    return canonical


def canonical_settings(settings):
    """Warn for an old solver name and return settings with the canonical name.

    A public entry point calls this once. When ``settings.solver_type`` is an
    old name, the function warns (see :func:`warn_solver_alias`) and returns
    a copy with the canonical name, so the inner calls see only canonical
    names and give no second warning. Otherwise it returns *settings*
    unchanged. The function does not change the warning filters, so it is
    safe in threads.

    Parameters
    ----------
    settings : SolverSettings
        Settings given by the user.

    Returns
    -------
    SolverSettings
        *settings*, or a copy with the canonical solver name.
    """
    name = getattr(settings, "solver_type", None)
    if name is None or str(name).lower() not in SOLVER_ALIASES:
        return settings
    out = settings.clone()
    out.solver_type = warn_solver_alias(name)
    return out


def canonical_solver(name: str | None) -> str | None:
    """Warn for an old solver name and return the canonical name.

    Parameters
    ----------
    name : str or None
        Solver name given by the user.

    Returns
    -------
    str or None
        The canonical name for an old name; else *name* unchanged.
    """
    if name is None or str(name).lower() not in SOLVER_ALIASES:
        return name
    return warn_solver_alias(name)
