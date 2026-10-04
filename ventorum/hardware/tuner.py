# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Machine tuner: measure this machine and write its profile.

The tuner runs only when it is called: at setup (``ventorum-tune``), or when a
user or an AI agent asks for it (:func:`tune_machine`). It never runs before
an ordinary analysis. A normal run takes about half a minute; ``quick=True``
skips the large cases.

It measures, for each case size (see :data:`ventorum.hardware.profile.SIZE_CLASSES`):

* the kernel backend for each kernel function (tensor, influence, induced,
  trefftz), among the available compiled backends and numpy;
* the best number of kernel threads for one case that runs alone;
* the best split of the cores between cases in parallel and threads per
  case, for batches (sweeps, design studies);
* with a CUDA GPU and the GPU pipelines (:mod:`ventorum.gpu`), a cost
  model of the CPU and GPU paths per solver family (the rule of the
  ``"auto"`` device).

It also records the GPUs that it finds. The results change only the speed,
never the results of an analysis.

The design (fingerprint, stored profile, safe defaults) follows the original
Ventorum tuner (ventorum.legacy.hardware).
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import time
from typing import Any

import numpy as np

# Lattice sizes of the test cases: (panels per semi-span, chordwise panels).
_CASES = {
    "small": (20, 4),     # 160 panels
    "medium": (60, 8),    # 960 panels
    "large": (100, 16),   # 3200 panels
}
_REPEATS = {"small": 7, "medium": 3, "large": 2}
_TIE = 0.03  # a setting with fewer threads wins if it is less than 3 % slower


def _wing(n_panels: int, n_chord: int):
    import ventorum as vt

    wing = vt.LiftingSurface(name="TuningWing", semi_span=5.0, sections=[
        vt.WingSection(y_frac=0.0, chord=1.6), vt.WingSection(y_frac=1.0, chord=0.9, twist=np.radians(-2.0)),
    ])
    st = vt.SolverSettings(solver_type="vlm", n_panels=n_panels, n_chord=n_chord)
    return vt.Aircraft(name="TuningWing", surfaces=[wing]), st


def _time(fn, repeats: int) -> float:
    fn()  # warm-up (compilation, caches)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def _time_warm(fn, repeats: int, warm_seconds: float = 0.5) -> float:
    """Return the median time of *fn* after it ran for *warm_seconds* (a GPU comes out of its idle clocks)."""
    t_end = time.perf_counter() + warm_seconds
    fn()
    while time.perf_counter() < t_end:
        fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def _thread_candidates(cores: int) -> list[int]:
    c = {1, cores}
    k = 2
    while k < cores:
        c.add(k)
        k *= 2
    return sorted(c)


def _pick(timings: dict[int, float]) -> int:
    """Return the key with the smallest time; fewer threads win a tie."""
    best = min(timings.values())
    return min(k for k, t in timings.items() if t <= best * (1.0 + _TIE))


@contextlib.contextmanager
def _kernel_backend(name: str):
    from ventorum.aero import vortex

    old = vortex._kernel_backend
    vortex.set_kernel_backend(name)
    try:
        yield
    finally:
        vortex._kernel_backend = old


def _available_backends() -> list[str]:
    """Return the available kernel backends in preference order."""
    from ventorum.aero import vortex as V

    backends = []
    if V._nb.HAVE_NUMBA:
        backends.append("numba")
    if V._HAVE_CYTHON:
        backends.append("cython")
    if V._HAVE_TORCH:
        backends.append("torch")
    backends.append("numpy")
    return backends


def _pick_backend(timings: dict[str, float]) -> str:
    """Return the fastest backend; numba wins a tie, then cython, then torch, then numpy.

    Numba comes first because it needs no C compiler on reinstall.
    """
    best = min(timings.values())
    close = [b for b, t in timings.items() if t <= best * (1.0 + _TIE)]
    for pref in ("numba", "cython", "torch", "numpy"):
        if pref in close:
            return pref
    return min(timings, key=timings.get)


def _measure_kernels(cls: str, ac, st, cores: int) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
    """Time each kernel function for each backend of one case size.

    Returns the chosen backend per kernel and the raw times in seconds.
    """
    from ventorum.aero import vortex as V
    from ventorum.aero.system import UnknownMap, build_sources, panel_targets
    from ventorum.geometry.lattice import build_lattice
    from ventorum.utils import parallel as par

    lat = build_lattice(ac, st, collocation="vlm", n_chord=st.n_chord)
    n = lat.n_panels
    src = build_sources(lat, np.array([1.0, 0.0, 0.05]), UnknownMap(np.arange(n), np.arange(n), False), None)
    tg = panel_targets(lat, np.arange(n))
    gamma = np.linspace(0.5, 2.0, n)

    backends = _available_backends()
    if cls != "small" and len(backends) > 1:
        # Numpy is about 10 times slower at medium and large; skip it when a
        # compiled backend is available.
        backends = [b for b in backends if b != "numpy"]

    choices: dict[str, str] = {}
    times: dict[str, dict[str, float]] = {}
    for kernel in V.KERNELS:
        times[kernel] = {}
        for backend in backends:
            with _kernel_backend(backend), par.forced_single_threads(cores), par.solve_threads(n):
                times[kernel][backend] = _time(_kernel_fn(kernel, lat, src, tg, gamma, n), _REPEATS[cls])
        choices[kernel] = _pick_backend(times[kernel])
    return choices, times


def _kernel_fn(kernel: str, lat, src, tg, gamma, n: int):
    """Return a zero-argument function that calls one kernel of the lattice."""
    from ventorum.aero import vortex as V

    if kernel == "tensor":
        def fn():
            return V.horseshoe_velocity_tensor(lat.cp, src, tg)
    elif kernel == "influence":
        def fn():
            return V.influence_matrix(lat.cp, lat.normal_bc, src, n, tg)
    elif kernel == "induced":
        def fn():
            return V.induced_velocity(lat.force_points, src, gamma, tg)
    else:  # trefftz
        q = lat.force_points
        q_normal = np.tile([0.0, 0.0, 1.0], (q.shape[0], 1))

        def fn():
            return V.trefftz_normalwash(q, q_normal, q[::-1].copy(), gamma[: q.shape[0]],
                                        np.array([1.0, 0.0, 0.0]), 0.01)
    return fn


def _cython_threads() -> str | None:
    """Return the Cython thread mode: "openmp", "python" or None."""
    from ventorum.aero import vortex as V

    if not V._HAVE_CYTHON:
        return None
    return "openmp" if V._cy.openmp_enabled() else "python"


def _torch_device() -> str | None:
    """Return the torch device string of the tuner run, or None without torch."""
    from ventorum.aero import vortex as V

    if not V._HAVE_TORCH:
        return None
    return V._torch().torch_device()


def _make_profile(prof, settings, meas, hw, quick, tuning_seconds, vt) -> dict[str, Any]:
    """Return the profile dict."""
    return {
        "schema": prof.SCHEMA_VERSION,
        "ventorum_version": vt.__version__,
        "created_utc": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "fingerprint": hw.fingerprint,
        "hardware": hw.public_dict(),
        "quick": bool(quick),
        "tuning_seconds": round(tuning_seconds, 2),
        "settings": settings,
        "measurements": meas,
    }


@contextlib.contextmanager
def _profile_in_process(profile: dict[str, Any]):
    """Make *profile* the active profile of this process, without a file.

    The thread and batch steps of the tuner must run with the kernel
    choices of step 1. This context applies them in memory, also when
    ``save=False``, and restores the previous state after it.
    """
    from ventorum.hardware import profile as prof

    with prof._lock:
        old = (prof._cached, prof._cached_loaded)
        prof._cached, prof._cached_loaded = profile, True
    try:
        yield
    finally:
        with prof._lock:
            prof._cached, prof._cached_loaded = old


def _gpu_info() -> dict[str, Any]:
    from ventorum.hardware.detector import _detect_gpu_capabilities

    info = _detect_gpu_capabilities()
    info["compute_capability"] = list(info["compute_capability"]) if info.get("compute_capability") else None
    return info


# Lattices of the GPU step, per solver family: (panels per semi-span,
# chordwise panels of the vortex lattice).
_GPU_CASES = {
    True: {"vlm": [(10, 4), (40, 4)], "linear": [(10, None), (40, None), (160, None)],
           "nonlinear": [(10, None), (40, None), (160, None)]},
    False: {"vlm": [(10, 4), (40, 4), (100, 4)], "linear": [(10, None), (40, None), (160, None), (400, None)],
            "nonlinear": [(10, None), (40, None), (160, None), (400, None)]},
}
# Batch sizes (cases) of the GPU step, per device. The CPU time of a batch
# grows linearly with its cases; the GPU time does not (a small batch does
# not fill the GPU), so the GPU has a third, large batch.
_GPU_BATCHES = {"cpu": (1, 33), "gpu": (1, 33, 257)}


def _tuning_airfoil():
    """Return the tabulated airfoil of the nonlinear GPU step (a smooth lift curve with its maximum at 16 deg)."""
    import ventorum as vt

    a = np.radians(np.arange(-12.0, 22.0, 1.0))
    a_max = np.radians(16.0)
    cl = 2.0 * np.pi * (a - a ** 3 / (3.0 * a_max ** 2))
    cd = 0.008 + 0.01 * (a / np.radians(10.0)) ** 2
    return vt.TabulatedAirfoil(alpha=a, Cl_data=cl, Cd_data=cd, Cm_data=np.full_like(a, -0.05))


def _measure_gpu(quick: bool, say) -> tuple[dict[str, Any], dict[str, Any]]:
    """Measure the cost model of the GPU pipelines; return the GPU settings and the measurements.

    For each solver family (``"vlm"``, ``"linear"``, ``"nonlinear"``) and a
    few lattices, the time of sweeps of a few batch sizes on the CPU and on
    the GPU (the precision of ``"auto"``; the GPU times after a warm-up, at
    the clocks of a GPU in use). The model (key ``"model"``, see
    :func:`ventorum.gpu.cost_model`) lets the ``"auto"`` device compare the
    estimated times (:func:`ventorum.gpu.use_gpu`).
    """
    import ventorum as vt
    from ventorum import gpu

    if not gpu.available():
        return {"available": False, "reason": gpu.unavailable_reason()}, {}
    old_dev, old_prec = gpu.get_device(), gpu._precision
    precision = gpu.get_precision()
    solvers = {"vlm": vt.HorseshoeSolver, "linear": vt.LinearLLTSolver, "nonlinear": vt.NonlinearSolver}
    rows = []
    model: dict[str, dict[str, Any]] = {}
    try:
        for family, lattices in _GPU_CASES[bool(quick)].items():
            sec = {"airfoil": _tuning_airfoil()} if family == "nonlinear" else {}
            entry: dict[str, Any] = {"n": []}
            entry.update({dev: {"k": list(ks), "t": []} for dev, ks in _GPU_BATCHES.items()})
            for n, nc in lattices:
                wing = vt.LiftingSurface(name="TuningWing", semi_span=5.0, sections=[
                    vt.WingSection(y_frac=0.0, chord=1.6, **sec), vt.WingSection(y_frac=1.0, chord=0.9, **sec)])
                ac = vt.Aircraft(name="TuningWing", surfaces=[wing])
                st = vt.SolverSettings(solver_type=family, n_panels=n, n_chord=nc)
                solver = solvers[family]()
                cond = vt.FlightCondition(V_inf=30.0)
                n_p = int(solver.build(ac, st, cond).n_panels)
                entry["n"].append(n_p)
                seconds = {}
                for dev, ks in _GPU_BATCHES.items():
                    gpu.set_device(dev)
                    times = []
                    for K in ks:
                        alphas = np.radians(np.linspace(-4.0, 10.0, K))

                        def fn(solver=solver, ac=ac, cond=cond, st=st, alphas=alphas):
                            return solver.solve_sweep(ac, cond, st, alphas)

                        # A GPU after an idle time runs at low clocks: the
                        # model holds the times of a GPU in use.
                        times.append(_time_warm(fn, 5) if dev == "gpu" else _time(fn, 3))
                        seconds[f"{dev}_{K}"] = times[-1]
                    entry[dev]["t"].append(times)
                rows.append({"family": family, "panels": n_p, "seconds": seconds})
                say(f"gpu    {family:9s} {n_p:5d} panels: " + ", ".join(
                    f"{key.replace('_', ' ')} cases {1e3 * t:.1f} ms" for key, t in seconds.items()))
            model[family] = {precision: entry}
    finally:
        gpu.set_device(old_dev)
        gpu.set_precision(old_prec)
    return {"available": True, "precision": precision, "model": model}, {"lattices": rows}


def tune_machine(quick: bool = False, save: bool = True, verbose: bool = True) -> dict[str, Any]:
    """Measure this machine and return (and by default save) its profile.

    Parameters
    ----------
    quick : bool
        Skip the large cases (about 3 times faster).
    save : bool
        Write the profile to the user configuration folder (see
        :func:`ventorum.hardware.profile.profile_path`).
    verbose : bool
        Print the progress.

    Returns
    -------
    dict
        The profile (schema, version, fingerprint, hardware, settings and
        the raw measurements).
    """
    import warnings

    import ventorum as vt
    from ventorum.hardware import profile as prof
    from ventorum.hardware.detector import scan_hardware
    from ventorum.utils import parallel as par

    t_start = time.perf_counter()
    say = print if verbose else (lambda *a, **k: None)
    hw = scan_hardware(detect_gpu=False)
    cores = par.cpu_cores()
    classes = ["small", "medium"] if quick else ["small", "medium", "large"]
    meas: dict[str, Any] = {"single": {}, "batch": {}, "kernels": {}}
    settings: dict[str, Any] = {"single": {}, "batch": {}, "kernels": {}}

    from ventorum import gpu

    old_device = gpu.get_device()
    gpu.set_device("cpu")   # steps 1 to 3 measure the CPU paths
    try:
        profile = _tune_steps(quick, say, hw, cores, classes, meas, settings, t_start, prof, par, vt, warnings)
    finally:
        gpu.set_device(old_device)
    if save:
        path = prof.save_profile(profile)
        say(f"Profile saved: {path} ({profile['tuning_seconds']} s)")
    return profile


def _tune_steps(quick, say, hw, cores, classes, meas, settings, t_start, prof, par, vt, warnings) -> dict[str, Any]:
    """Run the steps of :func:`tune_machine` and return the profile."""
    with warnings.catch_warnings(), par.forced_single_threads(None):
        warnings.simplefilter("ignore")

        # 1. Kernel backend per class and kernel.
        for cls in classes:
            ac, st = _wing(*_CASES[cls])
            n_pan = par.estimate_panels(ac, st)
            choices, times = _measure_kernels(cls, ac, st, cores)
            settings["kernels"][cls] = choices
            meas["kernels"][cls] = times
            say(f"{cls:6s} ({n_pan} panels) kernels: " +
                ", ".join(f"{k} {v}" for k, v in choices.items()))

        # Steps 2 and 3 use the kernel choices of step 1. They are applied in
        # memory, also with save=False, and no partial profile file is written.
        step1 = _make_profile(prof, {"kernels": dict(settings["kernels"])}, {}, hw, quick, 0.0, vt)
        with _profile_in_process(step1):
            for cls in classes:
                ac, st = _wing(*_CASES[cls])
                n_pan = par.estimate_panels(ac, st)

                # 2. One case alone: kernel threads.
                single = {}
                for t in _thread_candidates(cores):
                    with par.forced_single_threads(t):
                        single[t] = _time(lambda ac=ac, st=st: vt.analyze(ac, alpha_deg=4.0, settings=st), _REPEATS[cls])
                best_t = _pick(single)
                settings["single"][cls] = {"threads": best_t}
                meas["single"][cls] = {"panels": n_pan, "seconds": {str(k): v for k, v in single.items()}}
                say(f"{cls:6s} ({n_pan} panels) one case: {best_t} threads, {1e3 * single[best_t]:.1f} ms")

                # 3. Batch: workers x threads per worker.
                n_tasks = {"small": 4 * cores, "medium": 2 * cores, "large": max(2, cores)}[cls]
                alphas = np.linspace(-4.0, 10.0, n_tasks)
                batch = {}
                for w in _thread_candidates(cores):
                    t0 = time.perf_counter()
                    vt.analyze_sweep(ac, alphas, settings=st, n_jobs=w)
                    batch[w] = n_tasks / (time.perf_counter() - t0)
                best_w = max(batch, key=batch.get)
                settings["batch"][cls] = {"workers": best_w, "threads": max(1, cores // best_w)}
                meas["batch"][cls] = {"panels": n_pan, "cases_per_second": {str(k): v for k, v in batch.items()}}
                say(f"{cls:6s} batch: {best_w} cases in parallel x {cores // best_w} threads, {batch[best_w]:.0f} cases/s")

            # 4. GPU pipelines: the cost model of the CPU and GPU paths. The
            # CPU times use the settings of steps 1 to 3.
            step3 = _make_profile(prof, {k: dict(settings[k]) for k in ("kernels", "single", "batch")}, {}, hw,
                                  quick, 0.0, vt)
            with _profile_in_process(step3):
                gpu_settings, gpu_meas = _measure_gpu(quick, say)
    settings["gpu"] = {**_gpu_info(), **gpu_settings}
    meas["gpu"] = gpu_meas
    settings["cython_threads"] = _cython_threads()
    settings["torch_device"] = _torch_device()
    return _make_profile(prof, settings, meas, hw, quick, time.perf_counter() - t_start, vt)
