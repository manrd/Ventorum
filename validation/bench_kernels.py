# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Benchmark the vortex kernels: numpy against numba against cython against torch.

The script prints a table with the median time (ms) of
:func:`ventorum.aero.vortex.induced_velocity` and
:func:`ventorum.aero.vortex.influence_matrix` for the four backends,
for the three case sizes of the tuner (160, 960 and 3200 panels), with one
thread and with all cores. The numbers change only the speed, never the
results; the tuner uses them to pick a backend per case size.

Usage:
    python validation/bench_kernels.py
    python validation/bench_kernels.py --cython-threads

The option ``--cython-threads`` compares the two thread paths of the
Cython kernels instead: OpenMP and row blocks on Python threads (the macOS
build, see docs/user/performance_limits.md). It needs an OpenMP build.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from ventorum.aero import vortex as V
from ventorum.aero.system import UnknownMap, build_sources, panel_targets
from ventorum.geometry.lattice import build_lattice
from ventorum.hardware import tuner
from ventorum.utils import parallel as par


def _median_time(fn, repeats: int) -> float:
    """Return the median wall time (s) of *fn*, with one warm-up call."""
    fn()  # warm-up (compilation, caches)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def _case(n_panels: int, n_chord: int):
    """Build the lattice, sources and targets of one tuner case."""
    ac, st = tuner._wing(n_panels, n_chord)
    lat = build_lattice(ac, st, collocation="vlm", n_chord=st.n_chord)
    n = lat.n_panels
    src = build_sources(lat, np.array([1.0, 0.0, 0.05]), UnknownMap(np.arange(n), np.arange(n), False), None)
    tg = panel_targets(lat, np.arange(n))
    gamma = np.linspace(0.5, 2.0, n)
    return lat, src, tg, gamma, n


def main() -> None:
    cores = par.cpu_cores()
    try:
        import ventorum.aero.vortex_cython  # noqa: F401
    except ImportError:
        have_cython = False
    else:
        have_cython = True
    backends = ["numpy", "numba"] + (["cython"] if have_cython else [])
    if not have_cython:
        print("ventorum.aero.vortex_cython is not compiled: the cython column is omitted.")
    if V._HAVE_TORCH:
        from ventorum.aero import vortex_torch as vtorch

        backends.append("torch")
        print(f"The torch column uses device '{vtorch.torch_device()}'.")
    else:
        print("torch is not installed: the torch column is omitted.")
    print(f"Kernel benchmark: median time in ms (this machine has {cores} cores).")
    print()
    print(f"{'panels':>8s} {'threads':>8s} {'backend':>8s} {'induced_velocity':>18s} {'influence_matrix':>18s}")
    for name, (n_panels, n_chord) in tuner._CASES.items():
        lat, src, tg, gamma, n = _case(n_panels, n_chord)
        for threads in (1, cores):
            for backend in backends:
                # Select the backend before solve_threads: it reads the backend on entry.
                with tuner._kernel_backend(backend), par.forced_single_threads(threads), par.solve_threads(n):
                    t_ind = _median_time(
                        lambda lat=lat, src=src, tg=tg, gamma=gamma:
                        V.induced_velocity(lat.force_points, src, gamma, tg),
                        tuner._REPEATS[name])
                    t_mat = _median_time(
                        lambda lat=lat, src=src, tg=tg, n=n:
                        V.influence_matrix(lat.cp, lat.normal_bc, src, n, tg),
                        tuner._REPEATS[name])
                print(f"{n:>8d} {threads:>8d} {backend:>8s} {1e3 * t_ind:>18.3f} {1e3 * t_mat:>18.3f}")
        print()


def cython_threads() -> None:
    """Print the time of the Cython kernels with OpenMP and with Python threads."""
    if not (V._HAVE_CYTHON and V._CY_OPENMP):
        print("This needs the Cython kernels built with OpenMP.")
        return
    cores = par.cpu_cores()
    print(f"Cython kernels, {cores} threads: median time in ms, OpenMP / Python threads.")
    print()
    print(f"{'panels':>8s} {'influence_matrix':>22s} {'induced_velocity':>22s}")
    old = V._kernel_backend
    V.set_kernel_backend("cython")
    try:
        for name, (n_panels, n_chord) in tuner._CASES.items():
            lat, src, tg, gamma, n = _case(n_panels, n_chord)
            res = {}
            for omp in (True, False):
                V._CY_OPENMP = omp
                with par.forced_single_threads(cores), par.solve_threads(n):
                    res[omp] = (
                        _median_time(lambda lat=lat, src=src, tg=tg, n=n:
                                     V.influence_matrix(lat.cp, lat.normal_bc, src, n, tg), tuner._REPEATS[name]),
                        _median_time(lambda lat=lat, src=src, tg=tg, gamma=gamma:
                                     V.induced_velocity(lat.force_points, src, gamma, tg), tuner._REPEATS[name]))
            cols = [f"{1e3 * res[True][k]:.2f} / {1e3 * res[False][k]:.2f}" for k in (0, 1)]
            print(f"{n:>8d} {cols[0]:>22s} {cols[1]:>22s}")
    finally:
        V._CY_OPENMP = True
        V.set_kernel_backend(old)


if __name__ == "__main__":
    if "--cython-threads" in sys.argv[1:]:
        cython_threads()
    else:
        main()
