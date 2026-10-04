# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Strip-level geometry of lifting surfaces for plots and mesh studies.

The geometry is the lattice the solvers use (see
:mod:`ventorum.geometry.lattice`), one panel per strip.
"""

from __future__ import annotations

from ventorum.core.datatypes import (
    Aircraft,
    DiscretizedSurface,
    LiftingSurface,
    SolverSettings,
)


def discretize_surface(
    surf: LiftingSurface,
    n_panels: int | None = None,
    spacing: str | None = None,
    surface_index: int = 0,
    half_mesh: bool = False,
) -> DiscretizedSurface:
    """Strip-level geometry of one :class:`LiftingSurface` (one panel per strip).

    The geometry is the same as the lattice the solvers use (see
    :mod:`ventorum.geometry.lattice` for the conventions). For a
    symmetric surface the strips run from the left tip to the right tip; with
    ``half_mesh=True`` only the right half is returned.

    Parameters
    ----------
    surf : LiftingSurface
    n_panels : int or None
        Panels per semi-span. The surface's own ``n_panels`` takes precedence;
        default 80.
    spacing : str or None
        The surface's own ``spacing`` takes precedence; default ``"auto"``.
    surface_index : int
        Index of this surface within the parent :class:`Aircraft`.
    """
    from ventorum.geometry.lattice import build_lattice, lattice_to_discretized

    settings = SolverSettings(
        n_panels=int(n_panels) if n_panels is not None else 80,
        spacing=spacing if spacing is not None else "auto",
    )
    lat = build_lattice(Aircraft(surfaces=[surf]), settings, collocation="vlm", n_chord=1)
    ds = lattice_to_discretized(lat, half_mesh=half_mesh)[0]
    ds.surface_index = surface_index
    return ds


def discretize_aircraft_surfaces(
    aircraft: Aircraft,
    settings: SolverSettings,
    half_mesh: bool = False,
) -> list[DiscretizedSurface]:
    """Return the strip-level geometry of all surfaces.

    The geometry comes from the same lattice that the solvers build
    (:func:`ventorum.geometry.lattice.build_lattice`). The surface
    overrides (``surf.n_panels``, ``surf.spacing``), the ``"auto"``
    spacing and ``settings.proportional_panels`` apply in the same way.

    Parameters
    ----------
    aircraft : Aircraft
    settings : SolverSettings
    half_mesh : bool
        If True, keep only the right half of symmetric surfaces.

    Returns
    -------
    list[DiscretizedSurface]
    """
    from ventorum.geometry.lattice import build_lattice, lattice_to_discretized

    lat = build_lattice(aircraft, settings, collocation="vlm", n_chord=1)
    out = lattice_to_discretized(lat, half_mesh=half_mesh)
    for idx, ds in enumerate(out):
        ds.surface_index = idx
    return out
