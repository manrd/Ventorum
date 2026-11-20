# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""GPU memory of the lattice cache.

The GPU pipelines keep the GPU data of a lattice (:class:`ventorum.gpu.engine.DeviceLattice`:
the lattice arrays, the unknown maps and the GPU kernel cache of the fixed
legs) in ``lattice.geom_cache``. The lattice cache
(:mod:`ventorum.geometry.lattice_cache`) keeps up to
:data:`ventorum.geometry.lattice_cache.MAX_ENTRIES` lattices, so without a
limit their GPU data stay on the GPU for the life of the process: about
300 MiB for a lattice of 4000 panels in float32, twice that in float64.

Rules:

* The GPU data of the cached lattices have a byte limit
  (:func:`get_cache_limit`). The default is :data:`CACHE_FRACTION` of the
  total memory of the GPU. The environment variable
  ``VENTORUM_GPU_CACHE_BYTES`` or :func:`set_cache_limit` change it.
* After each GPU batch, :func:`trim` releases the GPU data of the least
  recently used cached lattices until the total is below the limit. The
  most recent lattice is released only when it alone is above the limit.
  The CPU lattice stays in the lattice cache: the next GPU solve of that
  geometry copies its data to the GPU again.
* A lattice that the lattice cache drops also loses its GPU data, so a
  result that keeps the lattice (``details["lattice"]``) does not keep GPU
  memory.
* :func:`release` (also ``ventorum.gpu.release``) releases the GPU data of
  all cached lattices and gives the cached blocks of the PyTorch memory
  pool back to the driver.

A solve that runs while its GPU data are released keeps its own reference
to them, so it ends normally; the memory is freed when it ends. Releasing
GPU data never changes a result: the data are computed again with the same
arithmetic.
"""

from __future__ import annotations

import os
import sys
import threading
import warnings
from typing import Any

#: Default limit of the GPU data of the cached lattices, as a fraction of the total GPU memory.
CACHE_FRACTION = 0.25

#: Environment variable of the limit [bytes]. The value is an integer number of
#: bytes, or a number with one of the units KiB, MiB or GiB (for example ``"1.5GiB"``).
ENV_CACHE_BYTES = "VENTORUM_GPU_CACHE_BYTES"

#: First element of the ``geom_cache`` keys of the GPU data (see :func:`ventorum.gpu.engine.device_lattice`).
GPU_KEYS = ("gpu_lattice", "gpu_lattice_mach")

_UNITS = {"": 1, "b": 1, "kib": 1024, "mib": 1024 ** 2, "gib": 1024 ** 3}

_lock = threading.Lock()
_limit: int | None = None


def parse_bytes(text: str) -> int:
    """Return the number of bytes that *text* gives.

    Parameters
    ----------
    text : str
        An integer number of bytes, or a number followed by ``B``, ``KiB``,
        ``MiB`` or ``GiB`` (case does not matter), for example ``"800MiB"``.

    Returns
    -------
    int
        The number of bytes [B].

    Raises
    ------
    ValueError
        If *text* is not a non-negative number with a known unit.
    """
    raw = str(text).strip().lower().replace(" ", "")
    i = len(raw)
    while i > 0 and raw[i - 1].isalpha():
        i -= 1
    number, unit = raw[:i], raw[i:]
    if unit not in _UNITS:
        raise ValueError(f"Unknown unit {unit!r} in {text!r}; use B, KiB, MiB or GiB.")
    try:
        value = float(number)
    except ValueError:
        raise ValueError(f"{text!r} is not a number of bytes.") from None
    if not (value >= 0.0 and value != float("inf")):
        raise ValueError(f"The number of bytes must be finite and not negative, got {text!r}.")
    return int(value * _UNITS[unit])


def set_cache_limit(nbytes: int | str | None) -> None:
    """Set the byte limit of the GPU data of the cached lattices.

    Parameters
    ----------
    nbytes : int, str or None
        The limit [B], or a text for :func:`parse_bytes` (for example
        ``"1GiB"``). 0 keeps no GPU data after a batch. None removes the
        setting: the limit then comes from ``VENTORUM_GPU_CACHE_BYTES`` or
        the default (:data:`CACHE_FRACTION` of the total GPU memory).

    Raises
    ------
    ValueError
        If *nbytes* is negative or not a number of bytes.
    """
    global _limit
    if nbytes is None:
        value = None
    elif isinstance(nbytes, str):
        value = parse_bytes(nbytes)
    else:
        value = int(nbytes)
        if value < 0:
            raise ValueError(f"The cache limit must not be negative, got {nbytes!r}.")
    with _lock:
        _limit = value
    trim()


def _total_memory() -> int:
    """Return the total memory of the GPU [B], or 0 when PyTorch has no CUDA GPU."""
    torch = sys.modules.get("torch")
    if torch is None:
        from ventorum import gpu

        if not gpu.available():
            return 0
        import torch
    try:
        if not torch.cuda.is_available():
            return 0
        return int(torch.cuda.get_device_properties(0).total_memory)
    except Exception:  # noqa: BLE001 - a broken CUDA installation: no GPU data
        return 0


def get_cache_limit() -> int:
    """Return the byte limit of the GPU data of the cached lattices [B].

    The value of :func:`set_cache_limit`, else the value of the environment
    variable ``VENTORUM_GPU_CACHE_BYTES``, else :data:`CACHE_FRACTION` of
    the total memory of the GPU (0 without a GPU). An invalid environment
    value gives one warning, and the default is used.

    Returns
    -------
    int
        The limit [B].
    """
    with _lock:
        value = _limit
    if value is not None:
        return value
    raw = os.environ.get(ENV_CACHE_BYTES)
    if raw is not None and raw.strip():
        try:
            return parse_bytes(raw)
        except ValueError as exc:
            from ventorum import gpu

            gpu._warn_once(f"env:{ENV_CACHE_BYTES}:{raw}",
                           f"Invalid value {raw!r} of the environment variable {ENV_CACHE_BYTES} ({exc}). "
                           f"Ventorum uses {CACHE_FRACTION:g} of the total GPU memory.")
    return int(CACHE_FRACTION * _total_memory())


def _is_gpu_key(key: Any) -> bool:
    """Return True for a ``geom_cache`` key of GPU data."""
    return isinstance(key, tuple) and bool(key) and key[0] in GPU_KEYS


def _tensor_bytes(obj: Any, seen: set, depth: int = 0) -> int:
    """Return the bytes of the CUDA tensors that *obj* holds, each storage counted once [B]."""
    if depth > 6 or obj is None or isinstance(obj, (bool, int, float, str, bytes)):
        return 0
    module = type(obj).__module__
    if module.startswith("torch"):
        if not getattr(obj, "is_cuda", False) or not hasattr(obj, "untyped_storage"):
            return 0
        storage = obj.untyped_storage()
        key = ("storage", storage.data_ptr())
        if key in seen:
            return 0
        seen.add(key)
        return int(storage.nbytes())
    if module.startswith(("warp", "numpy")):
        return 0   # Warp arrays here are views of torch tensors; numpy arrays are host memory.
    key = ("object", id(obj))
    if key in seen:
        return 0
    seen.add(key)
    if isinstance(obj, dict):
        return sum(_tensor_bytes(v, seen, depth + 1) for v in list(obj.values()))
    if isinstance(obj, (list, tuple)):
        return sum(_tensor_bytes(v, seen, depth + 1) for v in obj)
    attrs = getattr(obj, "__dict__", None)
    if attrs:
        return sum(_tensor_bytes(v, seen, depth + 1) for v in list(attrs.values()))
    return 0


def lattice_bytes(lattice: Any) -> int:
    """Return the bytes of the GPU data that *lattice* holds in its ``geom_cache`` [B].

    Parameters
    ----------
    lattice : VortexLattice
        A lattice (cached or not).

    Returns
    -------
    int
        The bytes of the CUDA tensors of its GPU data [B].
    """
    cache = getattr(lattice, "geom_cache", None) or {}
    seen: set = set()
    return sum(_tensor_bytes(v, seen) for k, v in list(cache.items()) if _is_gpu_key(k))


def drop(lattice: Any) -> bool:
    """Remove the GPU data of *lattice* from its ``geom_cache``.

    The shallow copies of a cached lattice share one ``geom_cache``, so the
    data leave all copies. The CPU data stay.

    Parameters
    ----------
    lattice : VortexLattice
        A lattice (cached or not).

    Returns
    -------
    bool
        True if the lattice had GPU data.
    """
    cache = getattr(lattice, "geom_cache", None)
    if not cache:
        return False
    keys = [k for k in list(cache) if _is_gpu_key(k)]
    for k in keys:
        cache.pop(k, None)
    return bool(keys)


def _empty_pool() -> None:
    """Give the cached blocks of the PyTorch memory pool back to the driver (if CUDA is in use)."""
    torch = sys.modules.get("torch")
    if torch is None:
        return
    try:
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001 - the pool stays as it is
        warnings.warn(f"The PyTorch GPU memory pool was not emptied: {exc}", RuntimeWarning, stacklevel=3)


def _cached_lattices() -> list:
    """Return the cached lattices, the least recently used first."""
    from ventorum.geometry import lattice_cache

    return lattice_cache.entries()


def cache_bytes() -> int:
    """Return the bytes of the GPU data of all cached lattices [B]."""
    seen: set = set()
    total = 0
    for lat in _cached_lattices():
        cache = getattr(lat, "geom_cache", None) or {}
        total += sum(_tensor_bytes(v, seen) for k, v in list(cache.items()) if _is_gpu_key(k))
    return total


def trim(limit: int | None = None) -> int:
    """Release the GPU data of the least recently used cached lattices until the total is at or below the limit.

    The most recently used lattice is released only when it alone is above
    the limit. The pipelines call this function after each GPU batch.

    Parameters
    ----------
    limit : int or None, optional
        The limit [B]. None (default) uses :func:`get_cache_limit`.

    Returns
    -------
    int
        The bytes of the released GPU data [B].
    """
    held = [(lat, lattice_bytes(lat)) for lat in _cached_lattices()]
    held = [(lat, size) for lat, size in held if size > 0]
    if not held:
        return 0
    limit = get_cache_limit() if limit is None else int(limit)
    total = sum(size for _, size in held)
    freed = 0
    for lat, size in held:
        if total <= limit:
            break
        if drop(lat):
            total -= size
            freed += size
    if freed:
        _empty_pool()
    return freed


def release() -> int:
    """Release the GPU data of all cached lattices and empty the PyTorch memory pool.

    The CPU lattices stay in the lattice cache. A later GPU solve copies
    its lattice to the GPU again; its results do not change. Without a
    GPU, the function does nothing.

    Returns
    -------
    int
        The bytes of the released GPU data [B].
    """
    freed = 0
    for lat in _cached_lattices():
        size = lattice_bytes(lat)
        if drop(lat):
            freed += size
    _empty_pool()
    return freed
