# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Cache of built vortex lattices.

A lattice depends only on the geometry of the aircraft and on the mesh
settings. Repeated solves of the same geometry (at other flight conditions,
or with a new ``Aircraft`` object that describes the same geometry) reuse the
lattice instead of building it again.

* The key is a fingerprint of every public field of the surfaces, sections
  and airfoils (walked generically, so a new field is covered), plus the mesh
  settings, the collocation rule and the chordwise panel count. Any change of
  the geometry or the settings gives a new key.
* The cache keeps the last :data:`MAX_ENTRIES` lattices (least recently used
  out). The arrays of a cached lattice are read-only, so that a write into a
  shared lattice raises an error instead of changing later solves.
* :func:`get_or_build` returns a shallow copy: the arrays are shared, but each
  caller has its own ``kernel_cache`` (set to None), so a single solve never
  uses the kernel cache of an earlier sweep and gives the same bits as a solve
  with a new lattice.
* The key holds the values of the airfoil data (polar coefficients,
  tabulated polars and the section shapes of the mean line). The cached
  lattice keeps its own copies of the strip airfoils, made when it is
  built. The loads read the polar values from these copies. A later change
  of a user airfoil in place therefore never reaches a cached lattice: the
  changed airfoil gives a new key, and an aircraft with the old values
  gets a cache hit with the old values.
* The GPU pipelines keep the GPU data of a lattice in its ``geom_cache``
  (shared by the copies). A lattice that leaves the cache (least recently
  used out, or :func:`clear`) loses its GPU data, so a result that keeps
  the lattice does not keep GPU memory. The byte limit of the GPU data of
  the cached lattices is in :mod:`ventorum.gpu.memory`.
"""

from __future__ import annotations

import copy
import dataclasses
import sys
import threading
import warnings
from collections import OrderedDict
from typing import Any
from collections.abc import Callable

import numpy as np

#: Number of lattices kept.
MAX_ENTRIES = 8

_cache: OrderedDict[tuple, Any] = OrderedDict()
_lock = threading.Lock()


def _fingerprint(obj: Any) -> Any:
    """Return a hashable value that changes when any public field of *obj* changes."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, np.ndarray):
        return ("nd", obj.dtype.str, obj.shape, obj.tobytes())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (list, tuple)):
        return tuple(_fingerprint(x) for x in obj)
    if dataclasses.is_dataclass(obj):
        return (type(obj).__qualname__,) + tuple(
            _fingerprint(getattr(obj, f.name)) for f in dataclasses.fields(obj) if not f.name.startswith("_")
        )
    # An object of another type: its identity (a change inside it is not seen).
    return (type(obj).__qualname__, id(obj))


#: Module functions that build_lattice calls. Their identity is part of the
#: key, so that code which replaces one (for example the verification of
#: forced core groups) never gets a lattice built with the original.
_BUILD_HELPERS = ("core_groups", "near_miss_warnings", "surface_edge_geometry", "surface_reference_line",
                  "resolve_spacing", "_surface_eta", "_surface_mesh_eta", "airfoil_linear_properties", "check_overlaps",
                  "_chordwise_fractions", "compute_surface_n_panels", "build_lattice",
                  "validate_controls", "deflected_airfoil", "flap_effectiveness", "flap_moment_derivative")


def _helpers() -> tuple:
    """Return the helper function objects themselves.

    Not their id: the key keeps the objects alive, so a freed replacement
    cannot hand its id to a new one.
    """
    from ventorum.geometry import lattice as lattice_module

    return tuple(getattr(lattice_module, name, None) for name in _BUILD_HELPERS)


def surfaces_fingerprint(aircraft: Any) -> Any:
    """Fingerprint of the surfaces of *aircraft* (every public field, see the module docstring)."""
    return _fingerprint(aircraft.surfaces)


_info: OrderedDict[Any, dict] = OrderedDict()


def geometry_info(fingerprint: Any, aircraft: Any) -> dict:
    """Main-surface index and automatic reference values of the surfaces, cached by *fingerprint*.

    Both depend only on the surfaces. Returns ``{"main": int, "auto_ref": dict}``.
    """
    with _lock:
        hit = _info.get(fingerprint)
        if hit is not None:
            _info.move_to_end(fingerprint)
            return hit
    hit = {"main": aircraft.main_surface_index(), "auto_ref": aircraft._auto_reference_values()}
    with _lock:
        _info[fingerprint] = hit
        while len(_info) > 4 * MAX_ENTRIES:
            _info.popitem(last=False)
    return hit


def lattice_key(aircraft: Any, settings: Any, collocation: str, n_chord: int, chord_spacing: str,
                fingerprint: Any = None) -> tuple:
    """Key of the lattice of *aircraft* for the given mesh settings (*fingerprint* if computed before)."""
    return (
        _helpers(),
        surfaces_fingerprint(aircraft) if fingerprint is None else fingerprint,
        collocation, int(n_chord), str(chord_spacing),
        getattr(settings, "n_panels", None), getattr(settings, "spacing", None),
        bool(getattr(settings, "proportional_panels", False)), getattr(settings, "min_panels", None),
    )


def _own_airfoils(lattice: Any) -> None:
    """Replace the strip airfoils of a new lattice by copies that the cache owns.

    *lattice* comes from the build and has no cached derived data yet. The
    copies keep the sharing of the originals: strips that share one airfoil
    object share one copy, so the grouping of the strips by airfoil does not
    change. An object without a ``clone`` method is kept as it is.
    """
    afs = getattr(lattice, "airfoils", None)
    if not afs:
        return
    copies: dict[int, Any] = {}
    owned = []
    for af in afs:
        key = id(af)
        if key not in copies:
            copies[key] = af.clone() if hasattr(af, "clone") else af
        owned.append(copies[key])
    lattice.airfoils = owned
    for slc in getattr(lattice, "surfaces", None) or ():
        slice_afs = getattr(slc, "airfoils", None)
        if slice_afs:
            slc.airfoils = [copies.get(id(af), af) for af in slice_afs]


def _freeze(obj: Any) -> None:
    """Make every array of a lattice (and of its surface slices) read-only.

    The airfoils of the strips are not frozen: they are the copies that the
    cache owns (see :func:`_own_airfoils`), and the polar tables keep their
    lazy interpolation data. A change of the user airfoil data changes the
    surfaces fingerprint, so the cache then builds a new lattice.
    """
    for f in dataclasses.fields(obj):
        if f.name == "airfoils":
            continue
        val = getattr(obj, f.name)
        if isinstance(val, np.ndarray):
            val.flags.writeable = False
        elif isinstance(val, list):
            for item in val:
                if dataclasses.is_dataclass(item):
                    _freeze(item)


def get_or_build(key: tuple, build: Callable[[], Any]) -> Any:
    """Return a shallow copy of the cached lattice for *key*, building it with *build* if needed.

    *build* issues the warnings of the mesh construction. A cached lattice
    issues its ``mesh_warnings`` again, so that each solve of the same
    geometry gives the same warnings.
    """
    with _lock:
        hit = _cache.get(key)
        if hit is not None:
            _cache.move_to_end(key)
    if hit is not None:
        for msg in getattr(hit, "mesh_warnings", None) or ():
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
    else:
        hit = build()
        _own_airfoils(hit)
        _freeze(hit)
        evicted = []
        with _lock:
            _cache[key] = hit
            _cache.move_to_end(key)
            while len(_cache) > MAX_ENTRIES:
                evicted.append(_cache.popitem(last=False)[1])
        _drop_device_data(evicted)
    out = copy.copy(hit)
    out.kernel_cache = None
    # The control information is a list of dicts: give each copy its own, so
    # a change by one caller does not reach the cached lattice or other copies.
    if getattr(hit, "control_info", None):
        out.control_info = [dict(c) for c in hit.control_info]
    return out


def _drop_device_data(lattices: list) -> None:
    """Remove the GPU data of *lattices* (see :func:`ventorum.gpu.memory.drop`).

    Only when the GPU memory module is loaded: without it, no lattice has
    GPU data.
    """
    memory = sys.modules.get("ventorum.gpu.memory")
    if memory is None:
        return
    for lat in lattices:
        memory.drop(lat)


def entries() -> list:
    """Return the cached lattices, the least recently used first.

    The lattices are the cached objects, not copies. Do not change them.
    """
    with _lock:
        return list(_cache.values())


def clear() -> None:
    """Remove every cached lattice (and its GPU data)."""
    with _lock:
        evicted = list(_cache.values())
        _cache.clear()
        _info.clear()
    _drop_device_data(evicted)
