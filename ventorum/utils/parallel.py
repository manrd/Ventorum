# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Parallel execution: two levels, chosen by Ventorum.

* **Inside one case**: the compiled (Numba) vortex kernels run on several
  threads. BLAS runs on one thread inside a solve, so that the two thread
  pools do not compete for the same cores.
* **Across cases**: independent cases (the angles of a sweep, the cases of
  a batch or of a ground-effect grid) run in parallel worker threads.

:func:`plan_parallel` decides how many workers to use and how many kernel
threads each worker gets. It uses the tuned values of the machine profile
(see :mod:`ventorum.hardware`) when one matches this machine, and the
built-in defaults otherwise. The product of workers and threads per worker
never exceeds the number of cores.
"""

from __future__ import annotations

import contextlib
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

_controller = None
_controller_lock = threading.Lock()
_local = threading.local()


def cpu_cores() -> int:
    """Return the number of CPU cores that Ventorum may use."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:  # not available on Windows and macOS
        return max(1, os.cpu_count() or 1)


def _blas_controller():
    """Return a cached threadpoolctl controller, or None if threadpoolctl is not installed."""
    global _controller
    if _controller is None:
        with _controller_lock:
            if _controller is None:
                try:
                    from threadpoolctl import ThreadpoolController
                except ImportError:  # pragma: no cover - threadpoolctl is a dependency
                    _controller = False
                else:
                    _controller = ThreadpoolController()
    return _controller or None


def _set_kernel_threads(n: int) -> int | None:
    """Set the Numba thread count of the calling thread. Return the old value, or None."""
    try:
        import numba
    except ImportError:  # pragma: no cover - numba is a dependency
        return None
    n = max(1, min(int(n), numba.config.NUMBA_NUM_THREADS))
    old = numba.get_num_threads()
    numba.set_num_threads(n)
    return old


def kernel_threads() -> int:
    """Return the number of kernel threads of the calling thread.

    This is the thread-local Numba count that :func:`solve_threads` and
    :func:`case_executor` set, or ``cpu_cores()`` when Numba is not
    installed. The Cython kernels take it as their ``num_threads`` argument.
    """
    try:
        import numba
    except ImportError:  # pragma: no cover - numba is a dependency
        return cpu_cores()
    return int(numba.get_num_threads())


_forced_single: list[int | None] = [None]


@contextlib.contextmanager
def forced_single_threads(n: int | None):
    """Force the kernel threads of a case that runs alone (used by the tuner)."""
    old = _forced_single[0]
    _forced_single[0] = n
    try:
        yield
    finally:
        _forced_single[0] = old


def single_case_threads(n_panels: int | None, batch: int = 1) -> int:
    """Return the number of kernel threads for one case (or one batch of cases) that runs alone.

    A batch of more than one case (for example the angles of a lifting-line
    sweep, computed in one kernel call) uses all cores: its kernel calls do
    the work of many cases.
    """
    if _forced_single[0] is not None:
        return int(_forced_single[0])
    if batch > 1:
        return cpu_cores()
    from ventorum.hardware.profile import size_class, tuned_setting

    tuned = tuned_setting("single", size_class(n_panels), "threads")
    return int(tuned) if tuned else cpu_cores()


_blas_lock = threading.Lock()
_blas_depth = 0  # Number of threads inside blas_single_thread
_blas_limiter = None  # The threadpoolctl limit of the first entry


@contextlib.contextmanager
def blas_single_thread():
    """Run the BLAS and LAPACK calls inside the context on one thread.

    A dense solve of the size that Ventorum uses (up to a few hundred
    unknowns) is fastest on one thread: on a 12-core machine, OpenBLAS with
    all threads was 5 to 23 times slower for 100 to 400 unknowns. Without
    threadpoolctl the context does nothing.

    The BLAS thread count is a setting of the whole process, and the
    workers of :func:`case_executor` enter and leave the context at
    different times. The context therefore counts the threads that are
    inside it: the first entry sets one BLAS thread, and the last exit
    restores the count of the process from before the first entry. A
    limit per worker would restore the counts in the wrong order (a worker
    would get all BLAS threads again while it still runs, and the process
    would keep one BLAS thread after the run).
    """
    global _blas_depth, _blas_limiter
    ctl = _blas_controller()
    if ctl is None:
        yield
        return
    with _blas_lock:
        if _blas_depth == 0:
            _blas_limiter = ctl.limit(limits=1, user_api="blas")
        _blas_depth += 1
    try:
        yield
    finally:
        with _blas_lock:
            _blas_depth -= 1
            if _blas_depth == 0 and _blas_limiter is not None:
                _blas_limiter.restore_original_limits()
                _blas_limiter = None


@contextlib.contextmanager
def solve_threads(n_panels: int | None = None, batch: int = 1):
    """Set the thread policy inside one solve (or one batch of *batch* cases on the same lattice).

    * BLAS runs on one thread while the compiled kernels run (on a 4-core
      machine a vortex-lattice solve took 64 ms with both on 4 threads, and
      24 ms with BLAS on 1 thread).
    * In a worker of :func:`case_executor` the kernels keep the thread count
      of the worker. A case that runs alone uses the tuned thread count for
      its size (all cores by default); a batch uses all cores (see
      :func:`single_case_threads`). The thread count does not change the
      results (each kernel row is computed by one thread).
    * The panel count is stored in a thread-local variable for the duration
      of the context, so that the kernel backend resolver can use the size
      class of the solve. The old value is restored after the context.

    The settings apply only inside the context; the caller's settings are
    restored after it.
    """
    from ventorum.aero.vortex import get_kernel_backend

    old_n_panels = getattr(_local, "n_panels", None)
    _local.n_panels = n_panels
    try:
        old_threads = None
        if get_kernel_backend() in ("numba", "cython") and getattr(_local, "worker_threads", None) is None:
            old_threads = _set_kernel_threads(single_case_threads(n_panels, batch))
        try:
            # BLAS on one thread for every kernel backend: the dense solves of
            # Ventorum are small, and a multi-threaded OpenBLAS solve of 100 to
            # 400 unknowns was 5 to 23 times slower than on one thread.
            with blas_single_thread():
                yield
        finally:
            if old_threads is not None:
                _set_kernel_threads(old_threads)
    finally:
        _local.n_panels = old_n_panels


@dataclass(frozen=True)
class ParallelPlan:
    """How to run a set of independent cases."""

    workers: int              # cases that run at the same time
    threads_per_worker: int   # kernel threads of each case
    source: str               # "user", "profile" or "default"


def _default_plan(n_tasks: int, n_panels: int | None, cores: int) -> tuple[int, int]:
    """Return the built-in plan (workers, threads per worker) for a machine with no profile.

    Small cases cannot use many kernel threads well, so they run side by
    side; large cases run one at a time on all cores.
    """
    from ventorum.hardware.profile import size_class

    cls = size_class(n_panels)
    if cls == "small":
        workers = min(n_tasks, cores)
    elif cls == "medium":
        workers = min(n_tasks, max(1, cores // 2))
    else:
        workers = 1
    return workers, max(1, cores // max(1, workers))


def plan_parallel(n_tasks: int, n_panels: int | None = None, n_jobs: int | str | None = "auto") -> ParallelPlan:
    """Return the parallel plan for *n_tasks* independent cases of about *n_panels* panels each.

    Parameters
    ----------
    n_tasks : int
        Number of independent cases.
    n_panels : int or None
        Panels of one case (it selects the size class); None means small.
    n_jobs : int, str or None
        ``"auto"`` (or None): the tuned plan of this machine, else the
        built-in default. A positive integer: that many workers. ``-1`` or
        ``"max"``: one worker per core. ``1``: one case at a time. In every
        case the kernel threads per worker are the cores divided by the
        workers.
    """
    from ventorum.hardware.profile import size_class, tuned_setting

    cores = cpu_cores()
    inner = getattr(_local, "worker_threads", None)
    if inner is not None:
        # Already inside a worker: no nested pool, the worker's threads only.
        return ParallelPlan(1, inner, "nested")
    if n_tasks <= 1:
        return ParallelPlan(1, cores, "default")
    if n_jobs is None or (isinstance(n_jobs, str) and n_jobs.lower() == "auto"):
        tuned = tuned_setting("batch", size_class(n_panels))
        if isinstance(tuned, dict) and tuned.get("workers"):
            w = max(1, min(int(tuned["workers"]), n_tasks, cores))
            t = max(1, min(int(tuned.get("threads", cores // w)), cores // w))
            return ParallelPlan(w, t, "profile")
        w, t = _default_plan(n_tasks, n_panels, cores)
        return ParallelPlan(w, t, "default")
    if isinstance(n_jobs, str):
        if n_jobs.lower() != "max":
            raise ValueError(f"n_jobs={n_jobs!r} must be a positive integer, -1, 'auto' or 'max'.")
        w = min(n_tasks, cores)
    else:
        n = int(n_jobs)
        if n == -1:
            w = min(n_tasks, cores)
        elif n < 1:
            raise ValueError(f"n_jobs={n_jobs!r} must be a positive integer, -1, 'auto' or 'max'.")
        else:
            w = min(n, n_tasks)
    return ParallelPlan(w, max(1, cores // w), "user")


def estimate_panels(aircraft, settings) -> int | None:
    """Return the panel count of the vortex lattice of *aircraft* (an estimate for the parallel plan)."""
    try:
        from ventorum.geometry import lattice_cache
        from ventorum.geometry.lattice import build_lattice

        n_chord = settings.n_chord if settings.n_chord else 4
        chord_spacing = getattr(settings, "chord_spacing", "uniform")
        key = lattice_cache.lattice_key(aircraft, settings, "vlm", n_chord, chord_spacing)
        lat = lattice_cache.get_or_build(key, lambda: build_lattice(
            aircraft, settings, collocation="vlm", n_chord=n_chord, chord_spacing=chord_spacing,
        ))
        return int(lat.n_panels)
    except Exception:  # noqa: BLE001 - only an estimate
        return None


def run_cases(func, items, n_panels: int | None = None, n_jobs: int | str | None = "auto") -> list:
    """Run *func* on each item with the parallel plan, and return the results in input order."""
    items = list(items)
    plan = plan_parallel(len(items), n_panels, n_jobs)
    if plan.workers <= 1:
        return [func(it) for it in items]
    with case_executor(plan) as ex:
        return list(ex.map(func, items))


def _worker_init(threads: int) -> None:
    _local.worker_threads = threads
    _set_kernel_threads(threads)


@contextlib.contextmanager
def case_executor(plan: ParallelPlan):
    """Yield a thread pool that runs cases with the thread counts of *plan*.

    Each worker thread sets its own Numba thread count. Use ``executor.map``
    or ``executor.submit`` as with any ``concurrent.futures`` executor.
    """
    with ThreadPoolExecutor(max_workers=plan.workers, initializer=_worker_init,
                            initargs=(plan.threads_per_worker,)) as ex:
        yield ex


def resolve_workers(n_tasks: int, n_jobs: int | str | None = "auto", max_cap: int = 16) -> int:
    """Return the number of workers for *n_tasks* independent cases.

    Kept for callers that need only the worker count; new code should use
    :func:`plan_parallel` and :func:`case_executor`, which also set the
    kernel threads of each worker.
    """
    return min(plan_parallel(n_tasks, None, n_jobs).workers, max(1, max_cap))
