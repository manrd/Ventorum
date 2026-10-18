# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Cache of panel solvers across solve calls of the 3D panel method.

Every call of :meth:`ventorum.solvers.panel3d.PanelMethodSolver.solve`
builds the panel mesh, checks it, builds a
:class:`ventorum.panel3d.solver.PanelSolver` (geometry, wakes, surface
gradient operator) and factorizes its systems. When only the flight
condition changes (angle of attack, sideslip, body rates, speed, density,
or the ground of a new configuration), the mesh and the solver stay the
same. This module keeps the last solvers, so that a repeated solve skips
the setup and, for a known configuration, the factorization (one
factorization serves all free-air conditions of a mesh: unit-onset
superposition).

Rules (see ``docs/design/performance_architecture.md``, section "Caches"):

* The key has two parts. The geometry key (:func:`geometry_key`) holds the
  surfaces fingerprint of :func:`ventorum.geometry.lattice_cache.surfaces_fingerprint`,
  the section coordinates that the mesh uses, the objects of unknown type
  of the surfaces, the mesh settings, the half-mesh flag and the module
  functions that build the mesh. The solver key (:func:`solver_key`) adds
  the solver class, the wake length, the Kutta option, the backend, the
  device and the precision, the reference values, the moment reference
  point, the thread settings that can change the bits of a
  factorization and the functions whose output the solver keeps (kernels,
  factorization, gradient operator).
* At most :data:`MAX_SOLVERS` solvers, at most :data:`MAX_BYTES` bytes of
  host memory in all cached solvers, and at most :data:`MAX_GPU_SOLVERS`
  solver with device arrays. The least recently used solver goes first.
  An evicted solver drops its matrices, factorizations and device arrays.
  A call that builds a new device solver first evicts the cached device
  solvers, so that the new solve has the device memory.
* A solver that one thread uses is not given to another thread at the
  same time: the other thread builds a second solver for its call and
  does not wait. The second solver is not cached.
* The arrays of a cached mesh are read-only. The factory gives each call
  its own copies of the strip data and of the mesh statistics. The
  results of a cached solve are bit-identical to those of an uncached
  solve with the same thread settings.
* A solve that raises an error drops the solver that it used.

:func:`clear_cache` (also ``ventorum.panel3d.clear_cache``) removes all
cached solvers. The setting ``SolverSettings.panel_cache=False`` solves
without the cache.
"""

from __future__ import annotations

import dataclasses
import sys
import threading
import types
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = [
    "MAX_BYTES",
    "MAX_GPU_SOLVERS",
    "MAX_SOLVERS",
    "CachedGeometry",
    "Lease",
    "acquire",
    "cache_info",
    "clear_cache",
    "find_geometry",
    "geometry_key",
    "release",
    "solver_bytes",
    "solver_key",
]

#: Largest number of cached solvers [-].
MAX_SOLVERS = 2

#: Largest host memory of all cached solvers together [bytes] (2 GiB).
MAX_BYTES = 2 * 2**30

#: Largest number of cached solvers with device (GPU) arrays [-].
MAX_GPU_SOLVERS = 1


@dataclasses.dataclass
class CachedGeometry:
    """Mesh data of one geometry key, shared by the solvers of that key.

    Attributes
    ----------
    mesh : PanelMesh
        Solved panel mesh (half or full). Its arrays are read-only after
        the first store in the cache.
    strip_maps : list
        Per surface, the panel indices of each lattice strip on the right
        side and on the left side (None when the left side is not meshed).
    geom_meta : list of dict
        Per surface, the strip stations and properties of the strip
        integration (positions in m, chords, widths in m, areas in m^2,
        lift slopes in 1/rad and zero-lift angles in rad).
    n_panels : int
        Number of solved panels [-].
    mesh_stats : dict
        Output of :func:`ventorum.panel3d.mesh.check_mesh` on the mesh.
    """

    mesh: Any
    strip_maps: list
    geom_meta: list
    n_panels: int
    mesh_stats: dict

    def freeze(self) -> None:
        """Make the arrays of the mesh, the strip data and the statistics read-only."""
        for f in dataclasses.fields(self.mesh):
            _freeze_value(getattr(self.mesh, f.name))
        for meta in self.geom_meta:
            _freeze_value(meta)
        _freeze_value(self.mesh_stats)


def _freeze_value(value: Any) -> None:
    """Set ``writeable = False`` on the arrays in *value* (arrays, dicts, lists, tuples)."""
    if isinstance(value, np.ndarray):
        value.flags.writeable = False
    elif isinstance(value, dict):
        for item in value.values():
            _freeze_value(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _freeze_value(item)


class _Held:
    """Key part for an object of unknown type: equal only to the same object.

    The key keeps the object itself, not only its ``id``: Python can give
    the ``id`` of a freed object to a new one, so an ``id`` alone could give
    a false hit (the lattice cache had this defect for its helper functions).
    """

    __slots__ = ("obj",)

    def __init__(self, obj: Any) -> None:
        self.obj = obj

    def __hash__(self) -> int:
        return id(self.obj)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Held) and other.obj is self.obj


def _unknown_objects(obj: Any, out: list) -> None:
    """Collect the objects that the surfaces fingerprint enters by identity only."""
    if obj is None or isinstance(obj, (bool, int, float, str, np.ndarray, np.generic)):
        return
    if isinstance(obj, (list, tuple)):
        for item in obj:
            _unknown_objects(item, out)
        return
    if dataclasses.is_dataclass(obj):
        for f in dataclasses.fields(obj):
            if not f.name.startswith("_"):
                _unknown_objects(getattr(obj, f.name), out)
        return
    out.append(_Held(obj))


def _array_key(arr: Any) -> tuple:
    """Return a hashable key of the dtype, the shape and the bytes of an array."""
    a = np.asarray(arr)
    return (a.dtype.str, a.shape, a.tobytes())


def geometry_key(
    aircraft: Any,
    section_coords: list,
    mesh_settings: tuple,
    helpers: tuple,
) -> tuple:
    """Return the key of the mesh data of *aircraft* for the given mesh settings.

    Parameters
    ----------
    aircraft : Aircraft
        Geometry. Every public field of its surfaces enters the key
        (:func:`ventorum.geometry.lattice_cache.surfaces_fingerprint`):
        sections, twist, controls and their deflections, node
        displacements, airfoils, positions and panel counts.
    section_coords : list of list of numpy.ndarray
        Unit-chord section coordinates of each section of each surface,
        as the mesh uses them [-]. They cover a section shape that the
        fingerprint cannot see (for example a file name whose file
        changed).
    mesh_settings : tuple
        Mesh settings and the half-mesh flag (hashable values).
    helpers : tuple
        The module functions that build the mesh data (the objects
        themselves), so that code that replaces one never gets mesh data
        built with the original.

    Returns
    -------
    tuple
        Hashable key.
    """
    from ventorum.geometry.lattice_cache import surfaces_fingerprint

    held: list = []
    _unknown_objects(aircraft.surfaces, held)
    coords = tuple(tuple(_array_key(c) for c in surf) for surf in section_coords)
    return ("panel3d-geometry", helpers, surfaces_fingerprint(aircraft), tuple(held), coords,
            tuple(mesh_settings))


def solver_key(geom_key: tuple, solver_settings: tuple) -> tuple:
    """Return the key of a solver: the geometry key and the solver settings (hashable values)."""
    return ("panel3d-solver", geom_key, tuple(solver_settings))


@dataclasses.dataclass(eq=False)
class _Entry:
    """One cached solver."""

    key: tuple
    geom_key: tuple
    geometry: CachedGeometry
    solver: Any
    gpu: bool
    in_use: bool = False
    evicted: bool = False
    host_bytes: int = 0
    device_bytes: int = 0


@dataclasses.dataclass(eq=False)
class Lease:
    """A solver given to one solve call.

    Attributes
    ----------
    solver : PanelSolver
        The solver of the call.
    cached : bool
        True when the solver came from the cache (built by an earlier call).
    """

    solver: Any
    cached: bool
    _entry: _Entry | None = None
    _key: tuple | None = None
    _geom_key: tuple | None = None
    _geometry: CachedGeometry | None = None
    _store: bool = False


_entries: OrderedDict[tuple, _Entry] = OrderedDict()
_lock = threading.Lock()


def find_geometry(geom_key: tuple) -> CachedGeometry | None:
    """Return the cached mesh data of *geom_key*, or None."""
    with _lock:
        for entry in reversed(_entries.values()):
            if entry.geom_key == geom_key:
                return entry.geometry
    return None


def acquire(
    key: tuple | None,
    geom_key: tuple | None,
    geometry: CachedGeometry,
    build: Callable[[], Any],
    gpu: bool = False,
) -> Lease:
    """Return the solver of one solve call, from the cache or new.

    Parameters
    ----------
    key : tuple or None
        Solver key (:func:`solver_key`), or None to solve without the
        cache (a new solver that is not stored).
    geom_key : tuple or None
        Geometry key of the solver.
    geometry : CachedGeometry
        Mesh data of the solver.
    build : callable
        Function without arguments that builds a new solver.
    gpu : bool, default=False
        True when the solver keeps device arrays (gpu backend). Before a
        new device solver is built (also with *key* None), the cached
        device solvers that no call uses are evicted, so that their device
        memory is free for the new solve.

    Returns
    -------
    Lease
        The solver and its cache state. Give it back with :func:`release`.

    Notes
    -----
    When another thread uses the cached solver of *key*, this call does
    not wait: it builds a second solver, uses it for this call only, and
    does not store it.
    """
    busy = False
    to_free: list[_Entry] = []
    with _lock:
        entry = None if key is None else _entries.get(key)
        if entry is not None:
            if not entry.in_use:
                entry.in_use = True
                _entries.move_to_end(key)
                return Lease(solver=entry.solver, cached=True, _entry=entry)
            busy = True
        if gpu:
            for other in list(_entries.values()):
                if other.gpu and not other.in_use:
                    _remove(other)
                    to_free.append(other)
    _free(to_free)
    solver = build()
    return Lease(solver=solver, cached=False, _key=key, _geom_key=geom_key, _geometry=geometry,
                 _store=key is not None and not busy)


def release(lease: Lease, ok: bool = True) -> None:
    """Give back the solver of a call and apply the bounds of the cache.

    Parameters
    ----------
    lease : Lease
        Lease from :func:`acquire`.
    ok : bool, default=True
        False when the solve raised an error: the solver is then dropped
        (its state can be incomplete).
    """
    entry = lease._entry
    if entry is None and not (lease._store and ok):
        return
    host, device = solver_bytes(lease.solver)
    to_free: list[_Entry] = []
    with _lock:
        if entry is not None:
            entry.in_use = False
            entry.host_bytes, entry.device_bytes = host, device
            if not ok or entry.evicted:
                if _entries.get(entry.key) is entry:
                    del _entries[entry.key]
                to_free.append(entry)
            else:
                to_free.extend(_enforce_bounds(entry))
        elif lease._key not in _entries:
            lease._geometry.freeze()
            entry = _Entry(key=lease._key, geom_key=lease._geom_key, geometry=lease._geometry,
                           solver=lease.solver, gpu=getattr(lease.solver, "backend", "") == "gpu",
                           host_bytes=host, device_bytes=device)
            _entries[entry.key] = entry
            to_free.extend(_enforce_bounds(entry))
    _free(to_free)


def _remove(entry: _Entry) -> None:
    """Remove *entry* from the cache (the lock is held)."""
    if _entries.get(entry.key) is entry:
        del _entries[entry.key]
    entry.evicted = True


def _enforce_bounds(latest: _Entry) -> list[_Entry]:
    """Evict entries until the bounds hold; return the evicted entries that are not in use.

    The lock is held. An entry in use is never evicted here; its call
    applies the bounds again when it ends.
    """
    evicted: list[_Entry] = []
    if latest.host_bytes > MAX_BYTES:
        # A solver larger than the whole budget is not kept.
        _remove(latest)
        return [latest]
    if latest.gpu:
        others = [e for e in _entries.values() if e.gpu and e is not latest]
        while others and len(others) + 1 > MAX_GPU_SOLVERS:
            victim = next((e for e in others if not e.in_use), None)
            if victim is None:
                # The other device solvers are in use: do not keep this one.
                _remove(latest)
                return [latest]
            _remove(victim)
            others.remove(victim)
            evicted.append(victim)
    while len(_entries) > MAX_SOLVERS or sum(e.host_bytes for e in _entries.values()) > MAX_BYTES:
        victim = next((e for e in _entries.values() if not e.in_use), None)
        if victim is None:
            break
        _remove(victim)
        evicted.append(victim)
    return evicted


def _free(entries: list[_Entry]) -> None:
    """Drop the matrices, factorizations and device arrays of evicted solvers."""
    gpu_freed = False
    for entry in entries:
        solver = entry.solver
        for s in (solver, getattr(solver, "_full_solver", None)):
            if s is None:
                continue
            s._configs = {}
            s._body_parts = None
        solver._full_solver = None
        gpu_freed = gpu_freed or entry.gpu
    if gpu_freed:
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available() and torch.cuda.is_initialized():
            # Give the freed blocks back to the driver, so that the Warp
            # kernels and other programs can use them.
            torch.cuda.empty_cache()


def clear_cache() -> None:
    """Remove every cached panel solver and free its memory.

    A solver that a running solve uses is freed when that solve ends.
    """
    with _lock:
        entries = list(_entries.values())
        _entries.clear()
        for entry in entries:
            entry.evicted = True
        to_free = [e for e in entries if not e.in_use]
    _free(to_free)


def cache_info() -> dict:
    """Return the state of the cache.

    Returns
    -------
    dict
        ``entries`` (number of cached solvers), ``gpu_entries`` (solvers
        with device arrays), ``in_use`` (solvers that a running solve
        uses), ``host_bytes`` and ``device_bytes`` (memory of the cached
        solvers at their last release, in bytes).
    """
    with _lock:
        entries = list(_entries.values())
    return {
        "entries": len(entries),
        "gpu_entries": sum(1 for e in entries if e.gpu),
        "in_use": sum(1 for e in entries if e.in_use),
        "host_bytes": int(sum(e.host_bytes for e in entries)),
        "device_bytes": int(sum(e.device_bytes for e in entries)),
    }


_SKIP_TYPES = (type, types.ModuleType, types.FunctionType, types.BuiltinFunctionType, types.MethodType,
               bool, int, float, complex, str, bytes)


def _is_tensor(obj: Any) -> bool:
    """Return True for a torch tensor (without an import of torch)."""
    return type(obj).__module__.startswith("torch") and hasattr(obj, "untyped_storage")


def solver_bytes(solver: Any) -> tuple[int, int]:
    """Return the host and device memory of the arrays that *solver* holds [bytes].

    The walk follows the attributes, dicts, lists, tuples and dataclass
    fields of the solver (also its full-mesh solver and its
    configurations). Each numpy buffer and each torch storage counts once.

    Parameters
    ----------
    solver : PanelSolver
        Solver to measure.

    Returns
    -------
    tuple of int
        (host bytes, device bytes).
    """
    host = 0
    device = 0
    seen: set = set()
    stack = [solver]
    while stack:
        obj = stack.pop()
        if obj is None or isinstance(obj, _SKIP_TYPES):
            continue
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, np.ndarray):
            base = obj
            while isinstance(base.base, np.ndarray):
                base = base.base
            if base is not obj:
                if id(base) in seen:
                    continue
                seen.add(id(base))
            host += int(base.nbytes)
            continue
        if _is_tensor(obj):
            storage = obj.untyped_storage()
            tag = ("storage", obj.device.type, storage.data_ptr())
            if tag in seen:
                continue
            seen.add(tag)
            if obj.device.type == "cpu":
                host += int(storage.nbytes())
            else:
                device += int(storage.nbytes())
            continue
        if isinstance(obj, dict):
            stack.extend(obj.values())
            continue
        if isinstance(obj, (list, tuple, set, frozenset)):
            stack.extend(obj)
            continue
        if dataclasses.is_dataclass(obj):
            stack.extend(getattr(obj, f.name, None) for f in dataclasses.fields(obj))
            continue
        attrs = getattr(obj, "__dict__", None)
        if isinstance(attrs, dict):
            stack.extend(attrs.values())
        for cls in type(obj).__mro__:
            slots = getattr(cls, "__slots__", ())
            for name in (slots,) if isinstance(slots, str) else slots:
                stack.append(getattr(obj, name, None))
    return host, device
