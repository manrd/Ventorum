# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Vortex-filament kernels for the lattice solvers.

All solvers use one horseshoe shape (a "bent" horseshoe)::

    infinity --(along d)--> TE_a --(along chord)--> A --(bound)--> B
             --(along chord)--> TE_b --(along d)--> infinity

* ``A -> B`` is the bound vortex (on the 1/4-chord line of the panel).
* ``A -> TE_a`` and ``B -> TE_b`` follow the local chord to the trailing edge.
  They stay in the plane of the wing, so a control point never sits just
  below or above a trailing leg. The old straight legs, which left the bound
  vortex along the free stream, did that at angle of attack and made the
  matrix near-singular for narrow panels.
* From the trailing edge the legs go to infinity along the wake direction
  ``d`` (the free-stream direction). In ground effect the free stream is
  parallel to the ground, so the wake is parallel to the ground too.

Image systems (ground plane) and y-symmetry are handled by the caller, which
gives extra "source" horseshoes with a sign of +1 or -1.

The kernels are plain numpy. They process the evaluation points in chunks,
so memory stays bounded for large lattices.

Regularisation: each source has a core radius ``rc``. The finite-segment
kernel uses the denominator ``|r1 x r2|^2 + (rc |r0|)^2`` and the
semi-infinite kernel uses ``|d x r|^2 + rc^2``. When ``rc`` is much smaller
than the distance, the kernels are the exact Biot-Savart law.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from ventorum.aero import vortex_numba as _nb
from ventorum.hardware.profile import size_class as _size_class
from ventorum.hardware.profile import tuned_setting as _tuned_setting
from ventorum.utils.parallel import _local as _parallel_local

try:
    from ventorum.aero import vortex_cython as _cy
    _HAVE_CYTHON = True
except ImportError:  # pragma: no cover - the extension is optional
    _cy = None
    _HAVE_CYTHON = False

# PyTorch is optional and slow to import (seconds), so the torch backend
# loads it on its first use only (see _torch). _HAVE_TORCH only says that the
# package is installed; _torch() sets it to False if the import fails.
_HAVE_TORCH = importlib.util.find_spec("torch") is not None
_th = None


def _torch():
    """Return the module :mod:`ventorum.aero.vortex_torch`; import it on the first call.

    Raises
    ------
    ValueError
        If PyTorch is not installed or cannot be imported.
    """
    global _th, _HAVE_TORCH
    if _th is None:
        mod = None
        if _HAVE_TORCH:
            try:
                from ventorum.aero import vortex_torch as mod
            except ImportError:  # pragma: no cover - a broken torch installation
                mod = None
        if mod is None or not mod.HAVE_TORCH:
            _HAVE_TORCH = False
            raise ValueError("The torch kernel backend needs the torch package.")
        _th = mod
    return _th

# How the Cython kernels use threads. With OpenMP, the kernel splits its
# rows itself (num_threads). Without OpenMP (the default macOS build, see
# setup.py), Ventorum splits the rows into blocks and runs one kernel call per
# block on Python threads; the kernels release the GIL, so the blocks run in
# parallel. Each row is computed alone, so both paths give the same bits.
_CY_OPENMP = bool(_HAVE_CYTHON and _cy.openmp_enabled())

# Arguments of each Cython kernel that have one row per evaluation point.
_CY_ROW_ARGS = {
    "induced_velocity_kernel": (0, 9, 10),
    "normal_influence_kernel": (0, 1, 12, 13),
    "velocity_tensor_kernel": (0, 9, 10),
    "velocity_unknowns_kernel": (0, 11, 12),
    "trefftz_normalwash_kernel": (0, 1, 7, 8),
}
_CY_MIN_ROWS = 8  # smallest block of rows for one Python thread
_cy_pool: ThreadPoolExecutor | None = None
_cy_pool_lock = threading.Lock()

_INV_4PI = 1.0 / (4.0 * np.pi)
_INV_2PI = 1.0 / (2.0 * np.pi)

# Target number of float64 values per temporary array in a chunk.
_CHUNK_BUDGET = 2_000_000

# Kernel backend: "numba" (compiled, parallel), "cython" (compiled C with
# OpenMP threads), "torch" (PyTorch tensors on the CPU or on a CUDA GPU) or
# "numpy" (the reference). "auto" selects the platform default or the machine
# profile. The environment variable VENTORUM_KERNEL sets the start value.
_KERNEL_BACKENDS = ("auto", "numba", "numpy", "cython", "torch")
_kernel_backend = os.environ.get("VENTORUM_KERNEL", "auto").lower()
if _kernel_backend not in _KERNEL_BACKENDS:
    _kernel_backend = "auto"


def set_kernel_backend(name: str) -> None:
    """Select the vortex kernel backend: ``"auto"``, ``"numba"``, ``"numpy"``, ``"cython"`` or ``"torch"``."""
    key = str(name).lower()
    if key not in _KERNEL_BACKENDS:
        raise ValueError(f"Unknown kernel backend {name!r}; use one of {_KERNEL_BACKENDS}.")
    if key == "numba" and not _nb.HAVE_NUMBA:
        raise ValueError("The Numba kernel backend needs the numba package.")
    if key == "cython" and not _HAVE_CYTHON:
        raise ValueError("The Cython kernel backend needs the compiled ventorum.aero.vortex_cython extension.")
    if key == "torch":
        if not _HAVE_TORCH:
            raise ValueError("The torch kernel backend needs the torch package.")
        _torch()  # import it now, so that a broken installation fails here
    global _kernel_backend
    _kernel_backend = key


def get_kernel_backend() -> str:
    """Return the kernel backend in use.

    For a forced backend (``set_kernel_backend``), it returns that backend.
    With ``"auto"``, it returns the default of the platform (see
    :func:`_default_backend`), not the profile.
    """
    if _kernel_backend == "numpy":
        return "numpy"
    if _kernel_backend == "cython" and _HAVE_CYTHON:
        return "cython"
    if _kernel_backend == "torch" and _HAVE_TORCH:
        return "torch"
    if _kernel_backend == "numba" and _nb.HAVE_NUMBA:
        return "numba"
    return _default_backend()


# The four kernel functions and their keys in the machine profile.
KERNELS = ("tensor", "influence", "induced", "trefftz")


def _is_macos() -> bool:
    """Return True when this process runs on macOS."""
    return sys.platform == "darwin"


def _default_backend() -> str:
    """Return the default backend when no profile matches.

    On macOS, Numba comes first (the Cython build has no OpenMP there, see
    docs/user/performance_limits.md). On other systems, Cython comes first
    (it was about 2 times faster than Numba on the one machine measured).
    """
    if _is_macos():
        if _nb.HAVE_NUMBA:
            return "numba"
        if _HAVE_CYTHON:
            return "cython"
        return "numpy"
    if _HAVE_CYTHON:
        return "cython"
    if _nb.HAVE_NUMBA:
        return "numba"
    return "numpy"


def kernel_backend_for(kernel: str, n_panels: int | None = None) -> str:
    """Return the backend ("numba", "cython", "torch" or "numpy") for one kernel call.

    A forced backend (``set_kernel_backend``) applies to all kernels. With
    ``"auto"``, the machine profile gives the backend for the kernel and the
    size class. Without a profile, or when the profile names a backend that
    is not available, the default of the platform is used.

    Parameters
    ----------
    kernel : str
        One of ``"tensor"``, ``"influence"``, ``"induced"``, ``"trefftz"``.
    n_panels : int or None
        Panel count that selects the size class. None means the thread-local
        value of ``ventorum.utils.parallel.solve_threads``, else small.

    Returns
    -------
    str
        The backend name.
    """
    if kernel not in KERNELS:
        raise ValueError(f"Unknown kernel {kernel!r}; use one of {KERNELS}.")
    if _kernel_backend != "auto":
        # A forced backend that is not available (for example VENTORUM_KERNEL=cython
        # without the compiled extension) falls back as in get_kernel_backend.
        return get_kernel_backend()
    hint = getattr(_parallel_local, "n_panels", None)
    cls = _size_class(hint if hint is not None else n_panels)
    backend = _tuned_setting("kernels", cls, kernel, default=None)
    if (backend not in ("numba", "cython", "torch", "numpy")
            or (backend == "cython" and not _HAVE_CYTHON)
            or (backend == "torch" and not _HAVE_TORCH)
            or (backend == "numba" and not _nb.HAVE_NUMBA)):
        return _default_backend()
    return backend


def _kernel_threads() -> int:
    """Return the kernel thread count of the calling thread.

    The count is the thread-local Numba setting that
    :func:`ventorum.utils.parallel.solve_threads` and
    :func:`ventorum.utils.parallel.case_executor` apply. The Cython
    kernels take it as their ``num_threads`` argument.
    """
    from ventorum.utils.parallel import kernel_threads
    return kernel_threads()


def _cython_pool() -> ThreadPoolExecutor:
    """Return the shared thread pool for the Cython row blocks (created on first use)."""
    global _cy_pool
    if _cy_pool is None:
        with _cy_pool_lock:
            if _cy_pool is None:
                from ventorum.utils.parallel import cpu_cores
                _cy_pool = ThreadPoolExecutor(max_workers=cpu_cores(), thread_name_prefix="ventorum-cython")
    return _cy_pool


def _cy_call(name: str, *args) -> np.ndarray:
    """Call the Cython kernel *name* with the kernel threads of the calling thread.

    With OpenMP the kernel gets the thread count. Without OpenMP the rows
    are split into blocks that run on Python threads (see ``_CY_OPENMP``).
    """
    kernel = getattr(_cy, name)
    threads = _kernel_threads()
    if _CY_OPENMP:
        return kernel(*args, threads)
    rows = _CY_ROW_ARGS[name]
    m = args[rows[0]].shape[0]
    blocks = min(threads, m // _CY_MIN_ROWS)
    if blocks <= 1:
        return kernel(*args, 1)
    edges = np.linspace(0, m, blocks + 1).astype(int)

    def run(k: int) -> np.ndarray:
        i0, i1 = edges[k], edges[k + 1]
        part = list(args)
        for r in rows:
            part[r] = args[r][i0:i1]
        return kernel(*part, 1)

    return np.concatenate(list(_cython_pool().map(run, range(blocks))), axis=0)


def _normal_influence(backend: str, args: tuple, parts: int) -> np.ndarray:
    """Call the normal-influence kernel of *backend* on the packed *args* with the part selector *parts*."""
    if backend == "torch":
        return _torch().normal_influence_kernel(*args, parts)
    if backend == "cython":
        return _cy_call("normal_influence_kernel", *args, parts)
    return _nb.normal_influence_kernel(*args, parts)


@dataclass(slots=True)
class HorseshoeSet:
    """A set of bent horseshoe vortices (the "sources").

    Attributes
    ----------
    a, b : (n, 3)
        Bound-vortex end points. Positive circulation runs from ``a`` to ``b``.
    a_te, b_te : (n, 3)
        Trailing-edge points where the legs turn into the wake.
    wake_dir : (n, 3)
        Unit wake direction for each horseshoe (image horseshoes in a ground
        plane have the reflected direction).
    rc : (n,)
        Core radius of each horseshoe [m].
    sign : (n,)
        Multiplier on the circulation (+1 real, -1 ground image).
    column : (n,)
        Index of the unknown that gives this horseshoe its circulation.
    group : (n,) or None
        Surface index of each horseshoe (images keep the index of their
        surface). Used for the cross-surface core, see :class:`Targets`.
    """

    a: np.ndarray
    b: np.ndarray
    a_te: np.ndarray
    b_te: np.ndarray
    wake_dir: np.ndarray
    rc: np.ndarray
    sign: np.ndarray
    column: np.ndarray
    group: np.ndarray | None = None

    @property
    def n(self) -> int:
        """Return the number of horseshoes."""
        return int(self.a.shape[0])

    def concatenate(self, other: HorseshoeSet) -> HorseshoeSet:
        """Return a new set with the horseshoes of this set, then of *other*."""
        return HorseshoeSet(
            a=np.vstack([self.a, other.a]),
            b=np.vstack([self.b, other.b]),
            a_te=np.vstack([self.a_te, other.a_te]),
            b_te=np.vstack([self.b_te, other.b_te]),
            wake_dir=np.vstack([self.wake_dir, other.wake_dir]),
            rc=np.concatenate([self.rc, other.rc]),
            sign=np.concatenate([self.sign, other.sign]),
            column=np.concatenate([self.column, other.column]),
            group=(None if self.group is None or other.group is None
                   else np.concatenate([self.group, other.group])),
        )


@dataclass(slots=True)
class Targets:
    """Evaluation points that belong to a surface.

    A vortex of one surface that acts on a point of another surface (for
    example a wing trailing vortex that passes through the tail) gets the
    extra core ``rc`` of the target point: ``rc_total^2 = rc_source^2 +
    rc_target^2``. The target core is a fraction of the target strip width,
    so the result does not depend on where a filament falls between two
    control points. Points of the same surface use the source core only.
    """

    group: np.ndarray   # (m,) group index of each point (surface index)
    rc: np.ndarray      # (m,) core radius added for sources of other surfaces [m]

    def rows(self, i0: int, i1: int) -> Targets:
        """Return the targets from index *i0* up to (not including) *i1*."""
        return Targets(group=self.group[i0:i1], rc=self.rc[i0:i1])


def _dot(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return u[..., 0] * v[..., 0] + u[..., 1] * v[..., 1] + u[..., 2] * v[..., 2]


def _cross(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    out = np.empty(np.broadcast_shapes(u.shape, v.shape))
    out[..., 0] = u[..., 1] * v[..., 2] - u[..., 2] * v[..., 1]
    out[..., 1] = u[..., 2] * v[..., 0] - u[..., 0] * v[..., 2]
    out[..., 2] = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    return out


def segment_velocity(
    P: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    rc: np.ndarray | float,
) -> np.ndarray:
    """Velocity at points *P* from unit-strength finite segments *A* -> *B*.

    Parameters
    ----------
    P : (m, 3) evaluation points.
    A, B : (n, 3) segment end points.
    rc : (n,) or float, core radius per segment.

    Returns
    -------
    (m, n, 3) induced velocity for unit circulation.
    """
    P = P[:, None, :]
    r1 = P - A[None, :, :]
    r2 = P - B[None, :, :]
    r0 = (B - A)[None, :, :]
    cr = _cross(r1, r2)
    cr2 = _dot(cr, cr)
    r0sq = _dot(r0, r0)
    rc2 = _rc2(rc, P.shape[0], A.shape[0])
    # A tiny floor keeps the unit vectors finite when P is on an end point.
    n1 = np.sqrt(_dot(r1, r1) + 1e-300)
    n2 = np.sqrt(_dot(r2, r2) + 1e-300)
    proj = (
        r0[..., 0] * (r1[..., 0] / n1 - r2[..., 0] / n2)
        + r0[..., 1] * (r1[..., 1] / n1 - r2[..., 1] / n2)
        + r0[..., 2] * (r1[..., 2] / n1 - r2[..., 2] / n2)
    )
    denom = cr2 + rc2 * r0sq
    ok = denom > 1e-300
    k = np.where(ok, _INV_4PI * proj / np.where(ok, denom, 1.0), 0.0)
    return cr * k[..., None]


def ray_velocity(
    P: np.ndarray,
    A: np.ndarray,
    d: np.ndarray,
    rc: np.ndarray | float,
) -> np.ndarray:
    """Velocity at *P* from unit-strength semi-infinite lines that start at *A*.

    The vorticity points along the unit vector *d* (from *A* to infinity).

    Parameters
    ----------
    P : (m, 3) evaluation points.
    A : (n, 3) start points.
    d : (n, 3) unit directions.
    rc : (n,) or float, core radius.

    Returns
    -------
    (m, n, 3) induced velocity for unit circulation.
    """
    r = P[:, None, :] - A[None, :, :]
    dd = d[None, :, :]
    cr = _cross(dd, r)
    cr2 = _dot(cr, cr)
    rn = np.sqrt(_dot(r, r) + 1e-300)
    cos_t = _dot(dd, r) / rn
    rc2 = _rc2(rc, P.shape[0], A.shape[0])
    denom = cr2 + rc2
    ok = denom > 1e-300
    k = np.where(ok, _INV_4PI * (1.0 + cos_t) / np.where(ok, denom, 1.0), 0.0)
    return cr * k[..., None]


def _rc2(rc, m: int, n: int) -> np.ndarray:
    """Squared core radius as an array that broadcasts to (m, n)."""
    rc = np.asarray(rc, dtype=float)
    if rc.ndim == 2:
        return rc ** 2
    return np.broadcast_to(rc ** 2, (n,))[None, :]


def _core(hs: HorseshoeSet, tg: Targets | None) -> np.ndarray:
    """Core radius per (point, source): (n,) or (m, n)."""
    if tg is None or hs.group is None:
        return hs.rc
    other = tg.group[:, None] != hs.group[None, :]
    return np.sqrt(hs.rc[None, :] ** 2 + np.where(other, tg.rc[:, None] ** 2, 0.0))


def _horseshoe_block(P: np.ndarray, hs: HorseshoeSet, tg: Targets | None = None) -> np.ndarray:
    """Unit-circulation velocity of every horseshoe at every point: (m, n, 3)."""
    a, b, a_te, b_te, d = hs.a, hs.b, hs.a_te, hs.b_te, hs.wake_dir
    rc = _core(hs, tg)
    v = segment_velocity(P, a, b, rc)
    v += segment_velocity(P, a_te, a, rc)
    v += segment_velocity(P, b, b_te, rc)
    v += ray_velocity(P, b_te, d, rc)
    v -= ray_velocity(P, a_te, d, rc)
    return v


def _chunk_rows(m: int, n: int) -> int:
    # About 12 temporaries of shape (rows, n, 3) live at the same time.
    return max(1, min(m, _CHUNK_BUDGET // max(1, 36 * n)))


def horseshoe_velocity_tensor(P: np.ndarray, hs: HorseshoeSet, targets: Targets | None = None) -> np.ndarray:
    """Velocity at *P* per source horseshoe, with the source sign applied.

    Returns an array of shape ``(m, n_sources, 3)``. Use this only for small
    problems; the other functions do not store the full tensor.
    """
    m, n = P.shape[0], hs.n
    backend = kernel_backend_for("tensor", n)
    if backend in ("numba", "cython", "torch"):
        a, b, a_te, b_te, d, rc2, grp, sign = _nb.pack_sources(hs)
        tg_group, tg_rc2, use_tg = _nb.pack_targets(m, hs, targets)
        Pc = np.ascontiguousarray(P, dtype=float)
        if backend == "torch":
            return _torch().velocity_tensor_kernel(Pc, a, b, a_te, b_te, d, rc2, grp, sign,
                                              tg_group, tg_rc2, use_tg, 0)
        if backend == "cython":
            return _cy_call("velocity_tensor_kernel", Pc, a, b, a_te, b_te, d, rc2, grp, sign,
                                              tg_group, tg_rc2, use_tg, 0)
        return _nb.velocity_tensor_kernel(Pc, a, b, a_te, b_te, d, rc2, grp, sign, tg_group, tg_rc2, use_tg)
    out = np.empty((m, n, 3))
    step = _chunk_rows(m, n)
    for i0 in range(0, m, step):
        i1 = min(m, i0 + step)
        tg = targets.rows(i0, i1) if targets is not None else None
        out[i0:i1] = _horseshoe_block(P[i0:i1], hs, tg) * hs.sign[None, :, None]
    return out


# Largest cached tensor [bytes]. A larger problem is solved without the cache.
_CACHE_LIMIT_BYTES = 512 * 1024 ** 2


def influence_matrix(
    P: np.ndarray,
    normals: np.ndarray,
    hs: HorseshoeSet,
    n_unknowns: int,
    targets: Targets | None = None,
    cache: dict | None = None,
    cache_key: object = None,
) -> np.ndarray:
    """Normal-velocity influence matrix.

    ``A[i, j]`` is the velocity normal to ``normals[i]`` at ``P[i]`` for unit
    circulation of unknown ``j``. Several sources can map to one unknown
    (mirror images, ground images); their contributions are added.

    With a *cache* (a dict) and a *cache_key*, the part of the matrix from
    the bound vortices and the chordwise legs is computed once and kept;
    only the wake legs are computed again. The caller must use the same key
    only for the same points, normals, sources and targets (the wake
    direction can change).
    """
    m, n = P.shape[0], hs.n
    backend = kernel_backend_for("influence", n)
    if backend in ("numba", "cython", "torch"):
        a, b, a_te, b_te, d, rc2, grp, sign = _nb.pack_sources(hs)
        tg_group, tg_rc2, use_tg = _nb.pack_targets(m, hs, targets)
        args = (np.ascontiguousarray(P, dtype=float), np.ascontiguousarray(normals, dtype=float),
                a, b, a_te, b_te, d, rc2, grp, sign,
                np.ascontiguousarray(hs.column, dtype=np.int64), int(n_unknowns), tg_group, tg_rc2, use_tg)
        if cache is not None and cache_key is not None and m * n_unknowns * 8 <= _CACHE_LIMIT_BYTES:
            fixed = cache.get(("A", cache_key))
            if fixed is None:
                fixed = _normal_influence(backend, args, 1)
                cache[("A", cache_key)] = fixed
            return fixed + _normal_influence(backend, args, 2)
        return _normal_influence(backend, args, 0)
    A_src = np.empty((m, n))
    step = _chunk_rows(m, n)
    for i0 in range(0, m, step):
        i1 = min(m, i0 + step)
        tg = targets.rows(i0, i1) if targets is not None else None
        v = _horseshoe_block(P[i0:i1], hs, tg)
        A_src[i0:i1] = _dot(v, normals[i0:i1, None, :]) * hs.sign[None, :]
    return _fold_columns(A_src, hs.column, n_unknowns)


def velocity_tensor_unknowns(
    P: np.ndarray, hs: HorseshoeSet, n_unknowns: int, targets: Targets | None = None,
) -> np.ndarray:
    """Velocity at *P* per unknown: shape ``(m, n_unknowns, 3)``.

    The Numba and Cython backends add each source into its unknown inside the
    kernel (no tensor per source); the other backends fold the tensor per
    source. Both add the sources of an unknown in source order.
    """
    m, n = P.shape[0], hs.n
    backend = kernel_backend_for("tensor", n)
    if backend in ("numba", "cython"):
        a, b, a_te, b_te, d, rc2, grp, sign = _nb.pack_sources(hs)
        tg_group, tg_rc2, use_tg = _nb.pack_targets(m, hs, targets)
        Pc = np.ascontiguousarray(P, dtype=float)
        col = np.ascontiguousarray(hs.column, dtype=np.int64)
        if backend == "cython":
            return _cy_call("velocity_unknowns_kernel", Pc, a, b, a_te, b_te, d, rc2, grp, sign, col,
                            int(n_unknowns), tg_group, tg_rc2, use_tg, 0)
        return _nb.velocity_unknowns_kernel(Pc, a, b, a_te, b_te, d, rc2, grp, sign, col, int(n_unknowns),
                                            tg_group, tg_rc2, use_tg)
    V = horseshoe_velocity_tensor(P, hs, targets)
    return _fold_columns(V, hs.column, n_unknowns)


def induced_velocity(
    P: np.ndarray, hs: HorseshoeSet, gamma_unknowns: np.ndarray, targets: Targets | None = None,
    cache: dict | None = None, cache_key: object = None,
    repeat: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Return the total induced velocity at *P* for the circulation of the unknowns: (m, 3).

    *cache* and *cache_key* work as in :func:`influence_matrix`: the
    velocity per source from the bound vortices and chordwise legs is kept,
    and only the wake legs are computed again.

    *repeat* = ``(keep, inverse)`` tells that the points repeat:
    ``P[keep][inverse]`` equals *P*, also for the targets. The kernel then
    runs only on ``P[keep]``. Each kernel row is computed alone, so the result
    has the same bits as without *repeat*. With a cache, the cached part keeps
    all rows (its matrix-vector product can round differently when the rows
    change); only the wake legs use the points without repeats.
    """
    if repeat is None:
        return _induced(P, hs, gamma_unknowns, targets, cache, cache_key, False)
    keep, inverse = repeat
    tg_keep = None if targets is None else Targets(group=targets.group[keep], rc=targets.rc[keep])
    if cache is not None and cache_key is not None:
        fixed = _induced(P, hs, gamma_unknowns, targets, cache, cache_key, True)
        if fixed is None:   # no cache for this case (numpy backend or too large)
            return _induced(P, hs, gamma_unknowns, targets, cache, cache_key, False)
        return fixed + _induced_parts(P[keep], hs, gamma_unknowns, tg_keep, 2)[inverse]
    return _induced(P[keep], hs, gamma_unknowns, tg_keep, None, None, False)[inverse]


def _induced_parts(P, hs, gamma_unknowns, targets, parts: int) -> np.ndarray:
    """Return one part (1 fixed legs, 2 wake legs) of the induced velocity with a compiled backend."""
    m, n = P.shape[0], hs.n
    g_src = gamma_unknowns[hs.column] * hs.sign
    backend = kernel_backend_for("induced", n)
    a, b, a_te, b_te, d, rc2, grp, sign = _nb.pack_sources(hs)
    tg_group, tg_rc2, use_tg = _nb.pack_targets(m, hs, targets)
    Pc = np.ascontiguousarray(P, dtype=float)
    gc = np.ascontiguousarray(g_src, dtype=float)
    if backend == "torch":
        return _torch().induced_velocity_kernel(Pc, a, b, a_te, b_te, d, rc2, grp, gc,
                                                tg_group, tg_rc2, use_tg, parts)
    if backend == "cython":
        return _cy_call("induced_velocity_kernel", Pc, a, b, a_te, b_te, d, rc2, grp, gc,
                        tg_group, tg_rc2, use_tg, parts)
    return _nb.induced_velocity_kernel(Pc, a, b, a_te, b_te, d, rc2, grp, gc, tg_group, tg_rc2, use_tg, parts)


def _induced(P, hs, gamma_unknowns, targets, cache, cache_key, fixed_only: bool) -> np.ndarray | None:
    """Body of :func:`induced_velocity`. With *fixed_only*, return only the cached part (None if there is none)."""
    m, n = P.shape[0], hs.n
    g_src = gamma_unknowns[hs.column] * hs.sign
    backend = kernel_backend_for("induced", n)
    if backend in ("numba", "cython", "torch"):
        a, b, a_te, b_te, d, rc2, grp, sign = _nb.pack_sources(hs)
        tg_group, tg_rc2, use_tg = _nb.pack_targets(m, hs, targets)
        Pc = np.ascontiguousarray(P, dtype=float)
        gc = np.ascontiguousarray(g_src, dtype=float)
        if cache is not None and cache_key is not None and m * n * 3 * 8 <= _CACHE_LIMIT_BYTES:
            T = cache.get(("T", cache_key))
            if T is None:
                if backend == "torch":
                    T3 = _torch().velocity_tensor_kernel(Pc, a, b, a_te, b_te, d, rc2, grp, sign,
                                                    tg_group, tg_rc2, use_tg, 1)
                elif backend == "cython":
                    T3 = _cy_call("velocity_tensor_kernel", Pc, a, b, a_te, b_te, d, rc2, grp, sign,
                                                    tg_group, tg_rc2, use_tg, 1)
                else:
                    T3 = _nb.velocity_tensor_kernel(Pc, a, b, a_te, b_te, d, rc2, grp, sign,
                                                    tg_group, tg_rc2, use_tg, 1)
                # Rows (point, component), columns source: one matrix-vector product per call.
                T = np.ascontiguousarray(T3.transpose(0, 2, 1).reshape(m * 3, n))
                cache[("T", cache_key)] = T
            g_col = np.ascontiguousarray(gamma_unknowns[hs.column], dtype=float)
            v = (T @ g_col).reshape(m, 3)
            if fixed_only:
                return v
            if backend == "torch":
                return v + _torch().induced_velocity_kernel(Pc, a, b, a_te, b_te, d, rc2, grp, gc,
                                                       tg_group, tg_rc2, use_tg, 2)
            if backend == "cython":
                return v + _cy_call("induced_velocity_kernel", Pc, a, b, a_te, b_te, d, rc2, grp, gc,
                                                       tg_group, tg_rc2, use_tg, 2)
            return v + _nb.induced_velocity_kernel(Pc, a, b, a_te, b_te, d, rc2, grp, gc,
                                                   tg_group, tg_rc2, use_tg, 2)
        if fixed_only:
            return None
        if backend == "torch":
            return _torch().induced_velocity_kernel(Pc, a, b, a_te, b_te, d, rc2, grp, gc,
                                               tg_group, tg_rc2, use_tg, 0)
        if backend == "cython":
            return _cy_call("induced_velocity_kernel", Pc, a, b, a_te, b_te, d, rc2, grp, gc,
                                               tg_group, tg_rc2, use_tg, 0)
        return _nb.induced_velocity_kernel(Pc, a, b, a_te, b_te, d, rc2, grp, gc, tg_group, tg_rc2, use_tg)
    if fixed_only:
        return None
    out = np.zeros((m, 3))
    step = _chunk_rows(m, n)
    for i0 in range(0, m, step):
        i1 = min(m, i0 + step)
        tg = targets.rows(i0, i1) if targets is not None else None
        v = _horseshoe_block(P[i0:i1], hs, tg)
        out[i0:i1] = np.einsum("mnk,n->mk", v, g_src)
    return out


def _fold_columns(A_src: np.ndarray, column: np.ndarray, n_unknowns: int) -> np.ndarray:
    """Add the columns (axis 1) of *A_src* that belong to the same unknown.

    *A_src* has shape ``(m, n_sources)`` or ``(m, n_sources, k)``; the result
    has ``n_unknowns`` in place of ``n_sources``. The sources of one unknown
    are added in the order in which they appear in *column*, starting from
    zero: the same operations as ``np.add.at``, so the result is the same to
    the last bit, but one vectorised addition per occurrence rank replaces
    the element-by-element loop of ``np.add.at``.
    """
    column = np.asarray(column)
    if A_src.shape[1] == n_unknowns and np.array_equal(column, np.arange(n_unknowns)):
        return A_src
    order = np.argsort(column, kind="stable")
    col_sorted = column[order]
    rank = np.arange(order.size) - np.searchsorted(col_sorted, col_sorted, side="left")
    counts = np.bincount(column, minlength=n_unknowns)
    if counts.size == n_unknowns and np.all(counts == counts[0]) and counts[0] > 0:
        # Every unknown has the same number of sources (for example the two
        # halves of a symmetric surface): gather and add, no scatter. The
        # sources are added in the same order; only the sign of an exact
        # zero can differ from np.add.at (0.0 + -0.0 = 0.0).
        per = order.reshape(n_unknowns, int(counts[0]))  # sources of each unknown, in column order
        out = np.take(A_src, per[:, 0], axis=1)
        for r in range(1, int(counts[0])):
            out = out + np.take(A_src, per[:, r], axis=1)
        return np.ascontiguousarray(out)
    out = np.zeros((A_src.shape[0], n_unknowns) + A_src.shape[2:])
    for r in range(int(rank.max()) + 1 if rank.size else 0):
        sel = order[rank == r]  # at most one source per unknown at each rank
        out[:, column[sel]] += A_src[:, sel]
    return out


def trefftz_normalwash(
    q: np.ndarray,
    q_normal: np.ndarray,
    p: np.ndarray,
    gamma: np.ndarray,
    d: np.ndarray,
    rc: np.ndarray | float,
    group: np.ndarray | None = None,
    targets: Targets | None = None,
) -> np.ndarray:
    """Compute the normal wash at Trefftz-plane points from 2-D point vortices.

    Parameters
    ----------
    q : numpy.ndarray
        Evaluation points [m], shape (m, 3). The points are already
        projected onto the plane.
    q_normal : numpy.ndarray
        Unit normals at the evaluation points, shape (m, 3). The normals are
        in the plane.
    p : numpy.ndarray
        Vortex positions [m], shape (n, 3). The positions are projected
        onto the plane.
    gamma : numpy.ndarray
        Circulation about the axis *d* [m^2/s], shape (n,). The sign follows
        the right-hand rule.
    d : numpy.ndarray
        Unit normal of the Trefftz plane (the wake direction), shape (3,).
    rc : numpy.ndarray or float
        2-D core radius [m], shape (n,) or a scalar.
    group : numpy.ndarray or None
        Core group of each vortex, shape (n,). Use it with *targets*.
    targets : Targets or None
        Core group and cross-surface core radius [m] of each evaluation
        point, shape (m,). Between a point and a vortex of a different core
        group, the squared core radius is ``rc**2 + targets.rc**2``: the
        same rule as the near field (see :func:`horseshoe_velocity_tensor`).
        None (or *group* None) uses *rc* only.

    Returns
    -------
    numpy.ndarray
        Velocity component along ``q_normal`` [m/s], shape (m,).
    """
    prep = trefftz_prepare(q.shape[0], p.shape[0], rc, group, targets)
    return trefftz_normalwash_prepared(q, q_normal, p, gamma, d, prep)


def trefftz_prepare(
    m: int,
    n: int,
    rc: np.ndarray | float,
    group: np.ndarray | None = None,
    targets: Targets | None = None,
) -> tuple:
    """Return the core data of :func:`trefftz_normalwash` as contiguous arrays.

    The data depend only on the vortices and the points, not on their
    positions: a caller that evaluates the same vortex set many times (for
    example the angles of a sweep) prepares it once. The arguments are those
    of :func:`trefftz_normalwash`; *m* and *n* are the numbers of points and
    vortices.
    """
    use_tg = targets is not None and group is not None
    src_group = np.ascontiguousarray(group, dtype=np.int64) if use_tg else np.zeros(n, dtype=np.int64)
    if use_tg:
        tg_group = np.ascontiguousarray(targets.group, dtype=np.int64)
        tg_rc2 = np.ascontiguousarray(np.broadcast_to(np.asarray(targets.rc, dtype=float) ** 2, (m,)))
    else:
        tg_group, tg_rc2 = np.zeros(m, dtype=np.int64), np.zeros(m)
    rc2_arr = np.ascontiguousarray(np.broadcast_to(np.asarray(rc, dtype=float) ** 2, (n,)))
    return rc2_arr, src_group, tg_group, tg_rc2, bool(use_tg)


def trefftz_normalwash_prepared(
    q: np.ndarray,
    q_normal: np.ndarray,
    p: np.ndarray,
    gamma: np.ndarray,
    d: np.ndarray,
    prep: tuple,
) -> np.ndarray:
    """Compute :func:`trefftz_normalwash` with the core data *prep* of :func:`trefftz_prepare`."""
    rc2_arr, src_group, tg_group, tg_rc2, use_tg = prep
    n = p.shape[0]
    backend = kernel_backend_for("trefftz", n)
    if backend in ("numba", "cython", "torch"):
        args = (np.ascontiguousarray(q, dtype=float), np.ascontiguousarray(q_normal, dtype=float),
                np.ascontiguousarray(p, dtype=float), np.ascontiguousarray(gamma, dtype=float),
                np.ascontiguousarray(d, dtype=float), rc2_arr, src_group, tg_group, tg_rc2, use_tg)
        if backend == "torch":
            return _torch().trefftz_normalwash_kernel(*args)
        if backend == "cython":
            return _cy_call("trefftz_normalwash_kernel", *args)
        return _nb.trefftz_normalwash_kernel(*args)
    r = q[:, None, :] - p[None, :, :]
    v = _cross(np.broadcast_to(d, r.shape), r)
    rc2 = rc2_arr[None, :]
    if use_tg:
        rc2 = rc2 + np.where(tg_group[:, None] != src_group[None, :], tg_rc2[:, None], 0.0)
    k = _INV_2PI * gamma[None, :] / (_dot(r, r) + rc2)
    wn = _dot(v, q_normal[:, None, :]) * k
    return wn.sum(axis=1)
