# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Cache of built vortex lattices (owner decision, 2026-10-03).

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
"""

from __future__ import annotations

import copy
import dataclasses
import threading
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
                  "resolve_spacing", "_surface_eta", "airfoil_linear_properties", "check_overlaps",
                  "_chordwise_fractions", "compute_surface_n_panels", "build_lattice")


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


def _freeze(obj: Any) -> None:
    """Make every array of a lattice (and of its surface slices) read-only.

    The airfoils of the strips are not frozen: they are the objects of the
    user (their tables are the arrays the user gave), and a solve must not
    change them. A change of their data changes the surfaces fingerprint,
    so the cache then builds a new lattice.
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
    """Return a shallow copy of the cached lattice for *key*, building it with *build* if needed."""
    with _lock:
        hit = _cache.get(key)
        if hit is not None:
            _cache.move_to_end(key)
    if hit is None:
        hit = build()
        _freeze(hit)
        with _lock:
            _cache[key] = hit
            _cache.move_to_end(key)
            while len(_cache) > MAX_ENTRIES:
                _cache.popitem(last=False)
    out = copy.copy(hit)
    out.kernel_cache = None
    return out


def clear() -> None:
    """Remove every cached lattice."""
    with _lock:
        _cache.clear()
        _info.clear()
