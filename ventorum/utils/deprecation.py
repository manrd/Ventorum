# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Deprecation helpers for renamed public API names."""

from __future__ import annotations

import os
import traceback
import warnings

_PACKAGE_DIR = os.path.normcase(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + os.sep

#: Canonical solver names. Use these in new code.
CANONICAL_SOLVERS = ("auto", "vlm", "linear", "nonlinear", "fourier")

#: Old solver names that still work. Each maps to its canonical name.
SOLVER_ALIASES = {
    "horseshoe": "vlm",
    "lattice": "vlm",
    "llt": "linear",
    "linear_llt": "linear",
}


def _user_stacklevel() -> int:
    """Return the ``stacklevel`` of the first frame outside the package.

    The caller gives this value to :func:`warnings.warn` so that the warning
    points at the user's code.

    Returns
    -------
    int
        Stack level for :func:`warnings.warn` called by the caller of this
        function, or -1 when importlib probes a name for the statement
        ``from package import name``.
    """
    frames = traceback.extract_stack()[:-2]
    for depth, frame in enumerate(reversed(frames), start=2):
        if frame.name == "_handle_fromlist" and frame.filename.startswith("<frozen importlib"):
            return -1
        if os.path.normcase(os.path.abspath(frame.filename)).startswith(_PACKAGE_DIR):
            continue
        return depth
    return 2


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
    warnings.warn(message, FutureWarning, stacklevel=_user_stacklevel())
    return canonical


#: Old solver class names that still work. Each maps to its canonical class name.
DEPRECATED_CLASS_NAMES = {
    "HorseshoeSolver": "VortexLatticeSolver",
    "LinearSolver": "LinearLLTSolver",
    "NonlinearSolver": "NonlinearLLTSolver",
}


def warn_deprecated_class(old_name: str, new_name: str) -> None:
    """Warn for an old solver class name.

    Parameters
    ----------
    old_name : str
        Old class name used by the caller.
    new_name : str
        Canonical class name to use instead.
    """
    message = (
        f"{old_name} is deprecated; use {new_name}. "
        "The old name will be removed in version 1.0."
    )
    level = _user_stacklevel()
    if level < 0:
        # importlib probes the name of a package with hasattr before the
        # statement "from package import name" reads it. Warn only once, at
        # the read.
        return
    warnings.warn(message, FutureWarning, stacklevel=level)


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
