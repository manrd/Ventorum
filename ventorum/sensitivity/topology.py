# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Shared-hinge topology of lattice geometries.

This module provides :class:`SharedHingeTopology` and
:func:`shared_hinge_topology` to determine which strips and surfaces share
flap hinge nodes in a vortex lattice, using the matching rule and tolerance
of the production lattice.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np


@dataclasses.dataclass(frozen=True)
class SharedHingeTopology:
    """Shared-hinge topology across surfaces in a lattice.

    Parameters
    ----------
    partner : np.ndarray
        Array with shape (n_strips, 2, 2) and integer dtype. For each strip
        and side (0 = left, 1 = right), (strip, side) of the coincident hinge
        node, or -1 where there is none.
    shared_surfaces : frozenset of tuple[str, str]
        Pairs of distinct surface names (surf_a, surf_b) with surf_a < surf_b
        that share at least one hinge node across their boundary.
    surface_names : tuple of str
        Names of the surfaces in the lattice.
    strip_surface : np.ndarray
        Surface index of each strip, with shape (n_strips,) and integer dtype.
    """

    partner: np.ndarray
    shared_surfaces: frozenset[tuple[str, str]]
    surface_names: tuple[str, ...]
    strip_surface: np.ndarray

    def __eq__(self, other: object) -> bool:
        """Compare equality with another topology or directly with partner ndarray."""
        if isinstance(other, np.ndarray):
            return bool(np.array_equal(self.partner, other))
        if isinstance(other, SharedHingeTopology):
            return bool(np.array_equal(self.partner, other.partner))
        return False

    def diff_surfaces(self, other: SharedHingeTopology) -> set[str]:
        """Return the names of surfaces whose shared hinge topology differs from other.

        Parameters
        ----------
        other : SharedHingeTopology
            Another topology to compare against.

        Returns
        -------
        set of str
            Names of surfaces whose shared-hinge pairings differed.
        """
        if not isinstance(other, SharedHingeTopology):
            raise TypeError(f"Cannot compare SharedHingeTopology with {type(other).__name__}")
        if self.partner.shape != other.partner.shape:
            return set(self.surface_names) | set(other.surface_names)

        diff_mask = np.any(self.partner != other.partner, axis=(1, 2))
        changed_strips = np.flatnonzero(diff_mask)
        surfs: set[str] = set()

        for s in changed_strips:
            for topo in (self, other):
                if s < len(topo.strip_surface):
                    idx_s = int(topo.strip_surface[s])
                    if 0 <= idx_s < len(topo.surface_names):
                        surfs.add(topo.surface_names[idx_s])
                for side in (0, 1):
                    p_strip = int(topo.partner[s, side, 0])
                    if 0 <= p_strip < len(topo.strip_surface):
                        idx_p = int(topo.strip_surface[p_strip])
                        if 0 <= idx_p < len(topo.surface_names):
                            surfs.add(topo.surface_names[idx_p])
        return surfs


def shared_hinge_topology(lattice: Any) -> SharedHingeTopology:
    """Return the shared-hinge topology of a lattice.

    The topology records which strips and surfaces share each flap hinge
    node, using the matching rule and tolerance of the production lattice
    (:func:`ventorum.geometry.lattice.build_lattice`).

    Parameters
    ----------
    lattice : VortexLattice
        Built lattice geometry.

    Returns
    -------
    SharedHingeTopology
        Shared-hinge topology.
    """
    n_strips = int(getattr(lattice, "n_strips", len(getattr(lattice, "chord", []))))
    surfaces = getattr(lattice, "surfaces", [])
    surface_names = tuple(getattr(s, "name", f"surf_{i}") for i, s in enumerate(surfaces))
    strip_surf = np.asarray(
        getattr(lattice, "strip_surface", np.zeros(n_strips, dtype=np.int64)), dtype=np.int64
    )

    flap_field = (
        getattr(lattice, "geom_cache", {}).get("flap_field")
        if hasattr(lattice, "geom_cache")
        else None
    )

    # The production lattice stores its hinge-node partners (its own
    # matching rule and tolerance) in the flap field. A lattice without a
    # flap field has no hinge, so no strip shares a hinge node.
    if flap_field is not None:
        partner = np.array(flap_field["partner"], dtype=np.int64, copy=True)
    else:
        partner = np.full((n_strips, 2, 2), -1, dtype=np.int64)

    # Find pairs of distinct surfaces that share at least one hinge node
    shared_surfs: set[tuple[str, str]] = set()
    for s in range(n_strips):
        if s >= len(strip_surf):
            continue
        surf_idx_s = int(strip_surf[s])
        if not (0 <= surf_idx_s < len(surface_names)):
            continue
        surf_s = surface_names[surf_idx_s]
        for side in (0, 1):
            q = int(partner[s, side, 0])
            if 0 <= q < len(strip_surf):
                surf_idx_q = int(strip_surf[q])
                if surf_idx_q != surf_idx_s and 0 <= surf_idx_q < len(surface_names):
                    surf_q = surface_names[surf_idx_q]
                    pair = (surf_s, surf_q) if surf_s < surf_q else (surf_q, surf_s)
                    shared_surfs.add(pair)

    return SharedHingeTopology(
        partner=partner,
        shared_surfaces=frozenset(shared_surfs),
        surface_names=surface_names,
        strip_surface=strip_surf,
    )
