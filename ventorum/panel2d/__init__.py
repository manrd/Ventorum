# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Ventorum-2D: two-dimensional panel method for airfoils, with ground effect."""

#: Revision of the 2D inviscid and viscous solvers. It changes whenever a
#: change of the solvers can change results (polar caches use it in their key).
SOLVER_REVISION = "2026-10-17.3"

__all__ = ["SOLVER_REVISION", "solve_viscous_flapped"]


def __getattr__(name: str):
    """Return lazy exports of ventorum.panel2d."""
    if name == "solve_viscous_flapped":
        from ventorum.panel2d.viscous import solve_viscous_flapped

        globals()["solve_viscous_flapped"] = solve_viscous_flapped
        return solve_viscous_flapped
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return module attribute names, including lazy exports."""
    return sorted(set(globals().keys()) | set(__all__))

