# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Machine tuner: measure this machine and write its profile.

The tuner runs only when it is called: at setup (``ventorum-tune``), or when a
user or an AI agent asks for it (:func:`tune_machine`). It never runs before
an ordinary analysis. A quick run stays under 60 s: it skips the large cases
and trims the repeats, the GPU batches and the gradient devices. The full
mode measures everything and reports its time.

It measures, for each case size (see :data:`ventorum.hardware.profile.SIZE_CLASSES`):

* the kernel backend for each kernel function (tensor, influence, induced,
  trefftz), among the available compiled backends and numpy;
* the best number of kernel threads for one case that runs alone;
* the best split of the cores between cases in parallel and threads per
  case, for batches (sweeps, design studies);
* the kernel threads of one case inside a multi-case batch that runs alone
  (a 33-case batch of each solver family on one lattice, on the sweep
  path); a cell is written only when it beats all cores by more than 5 %,
  else the sweeps use all cores;
* with a CUDA GPU and the GPU pipelines (:mod:`ventorum.gpu`), a cost
  model of the CPU and GPU paths per solver family (the rule of the
  ``"auto"`` device);
* in full mode, cold panel solves on three half meshes, with Numba on the
  CPU and the default solve precision on the GPU; a separate gain gate
  and self-check keep only GPU winners that beat the CPU by more than 5 %;
* the backend of the vector-Jacobian products of the adjoint gradients
  (entry ``grad_vjp``: per size class and device, the fastest of the
  compiled VJP kernels and the PyTorch reference copy, see
  :mod:`ventorum.sensitivity.kernels`). A profile without this entry (an
  older profile) still loads; the gradients then use the built-in default.

It measures once per machine, from the rows of the GPU step (no extra
timing), the smallest work where the GPU median beats the CPU median per
solver family (``min_work`` at 1 case, ``min_cases`` at 33 cases). The full
mode also times the chunk caps of the CPU ground-effect batch and of the
adjoint gradients per path (a quick run keeps the defaults).

It also records the GPUs that it finds. Tuning changes the speed; the device
choice can change results at the float32 round-off level (about 1e-7 to 1e-6
relative).

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
# Repeats of a quick run (quick must stay under 60 s on the owner machine).
_QUICK_REPEATS = {"small": 3, "medium": 2}
_TIE = 0.03  # a setting with fewer threads wins if it is less than 3 % slower


def _repeats(cls: str, quick: bool) -> int:
    """Return the timed repeats of one timing for case class *cls*."""
    if quick:
        return _QUICK_REPEATS.get(cls, _REPEATS[cls])
    return _REPEATS[cls]


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


def _measure_kernels(cls: str, ac, st, cores: int, quick: bool = False) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
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
    reps = _repeats(cls, quick)

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
                times[kernel][backend] = _time(_kernel_fn(kernel, lat, src, tg, gamma, n), reps)
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


def _grad_available() -> bool:
    """Return True when the gradient step can run (PyTorch is installed)."""
    import importlib.util

    return importlib.util.find_spec("torch") is not None


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
# not fill the GPU), so the GPU has a third, large batch. A quick run keeps
# the first two GPU batches (the cost model needs at least two sizes).
_GPU_BATCHES = {"cpu": (1, 33), "gpu": (1, 33, 257)}

# Half meshes of the panel step: (spanwise panels, chordwise panels).
# The last mesh is the factory default. These are not vortex lattices.
_PANEL3D_CASES = [(25, 10), (40, 25), (80, 40)]
_PANEL3D_REPEATS = 3


def _panel3d_workload(n_span: int, n_chord: int, backend: str, precision: str):
    """Return a half mesh and a cold free-air panel solve callable.

    The rectangular wing has semi-span 4 m and chord 1 m. The condition is
    4 deg angle of attack at 30 m/s. Each call constructs a new PanelSolver:
    assembly, LU and evaluation all run, without the factory solver cache,
    and with all BLAS threads, as in the factory.
    """
    import ventorum as vt
    from ventorum.airfoil import airfoil
    from ventorum.panel3d.mesh import mesh_lifting_surface
    from ventorum.panel3d.solver import PanelSolver

    wing = vt.LiftingSurface(name="PanelTuningWing", semi_span=4.0, sections=[
        vt.WingSection(y_frac=0.0, chord=1.0, shape="0012"),
        vt.WingSection(y_frac=1.0, chord=1.0, shape="0012"),
    ])
    mesh = mesh_lifting_surface(wing, airfoil("0012").coords, n_chord=n_chord, n_span=n_span, half=True)
    cond = vt.FlightCondition(V_inf=30.0, alpha=float(np.radians(4.0)))

    def run():
        # Time the solve as the panel factory runs it: the dense LU of a panel
        # solve uses all BLAS threads (the one-thread BLAS rule is for many
        # small lattice solves, not for one large panel system).
        solver = PanelSolver(mesh, backend=backend, dtype=precision, S_ref=8.0, b_ref=8.0, c_ref=1.0)
        result = solver.solve(cond)
        if backend == "gpu":
            import torch

            torch.cuda.synchronize()
        return {"cp": result.cp, "factorizations": solver.factorization_count}

    return mesh, run


def _time_lu(fn, repeats: int) -> float:
    """Return the median time of LU factorization callable *fn*."""
    fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def _measure_cpu_lu(say) -> tuple[str, dict[str, Any]]:
    """Measure SciPy and PyTorch CPU LU at the panel3d workload sizes.

    Return the chosen backend ("scipy" or "torch") and the measurements dict.
    Both LUs run at each panel size for 3 repeats (median time).
    PyTorch wins if it is faster; ties and missing PyTorch fall back to SciPy.
    """
    import importlib.util
    import scipy.linalg

    if importlib.util.find_spec("torch") is None:
        say("cpu_lu: PyTorch is not available: scipy default")
        return "scipy", {"choice": "scipy", "reason": "PyTorch is not available"}

    import torch

    scipy_times = {}
    torch_times = {}
    sizes = []
    for n_span, n_chord in _PANEL3D_CASES:
        n = (2 * n_span + 1) * n_chord
        sizes.append(n)
        rng = np.random.default_rng(42)
        base = np.asfortranarray(rng.standard_normal((n, n), dtype=np.float64))
        base.flat[::n + 1] += float(n)

        def scipy_fn(mat=base):
            a = mat.copy(order="F")
            return scipy.linalg.lu_factor(a, overwrite_a=True)

        def torch_fn(mat=base, size=n):
            a = mat.copy(order="F")
            t = torch.from_numpy(a)
            piv = torch.empty(size, dtype=torch.int32, device="cpu")
            torch.linalg.lu_factor(t, out=(t, piv))
            return t, piv

        t_s = _time_lu(scipy_fn, _PANEL3D_REPEATS)
        t_t = _time_lu(torch_fn, _PANEL3D_REPEATS)
        scipy_times[str(n)] = t_s
        torch_times[str(n)] = t_t
        say(f"cpu_lu {n:5d} panels: SciPy {t_s:.4f} s, PyTorch {t_t:.4f} s")

    largest = str(sizes[-1])
    winner = "torch" if torch_times[largest] * (1.0 + _TIE) < scipy_times[largest] else "scipy"
    say(f"cpu_lu winner: {winner}")
    meas = {
        "sizes": sizes,
        "scipy_seconds": scipy_times,
        "torch_seconds": torch_times,
        "choice": winner,
    }
    return winner, meas


def _measure_panel3d(quick: bool, say) -> tuple[dict[str, Any], dict[str, Any]]:
    """Measure cold CPU and GPU panel solves and check each GPU winner again.

    Return the profile entries per precision and the raw times [s]. A GPU
    median must beat the CPU median by more than 5 % in both measurements.
    The table keeps the smaller CPU and larger GPU median of the two, so
    interpolation cannot turn a rejected measured GPU row into a winner.
    A quick run keeps the default crossover and does not time panel solves.
    """
    from ventorum import gpu

    if quick:
        return {}, {"skipped": "quick mode"}
    if not gpu.available():
        return {}, {"skipped": "GPU is not available"}
    try:
        import numba  # noqa: F401 - the measured CPU backend is required
    except ImportError:
        return {}, {"skipped": "Numba is not available"}
    start = time.perf_counter()
    precision = gpu.get_precision()
    entry: dict[str, Any] = {"n": [], "cpu": [], "gpu": [], "winner": []}
    rows = []
    for n_span, n_chord in _PANEL3D_CASES:
        mesh, cpu_fn = _panel3d_workload(n_span, n_chord, "numba", precision)
        _, gpu_fn = _panel3d_workload(n_span, n_chord, "gpu", precision)
        n = len(mesh.panels)
        cpu = _time(cpu_fn, _PANEL3D_REPEATS)
        device = _time_warm(gpu_fn, _PANEL3D_REPEATS)
        row = {"panels": n, "cpu_s": cpu, "gpu_s": device}
        if device * (1.0 + gpu.PANEL3D_MIN_GAIN) < cpu:
            # Reverse the order for the independent check. Each timed call
            # still constructs a cold solver; the GPU stays at active clocks.
            check_gpu = _time_warm(gpu_fn, _PANEL3D_REPEATS)
            check_cpu = _time(cpu_fn, _PANEL3D_REPEATS)
            cpu, device = min(cpu, check_cpu), max(device, check_gpu)
            kept = device * (1.0 + gpu.PANEL3D_MIN_GAIN) < cpu
            row["self_check"] = {"cpu_s": check_cpu, "gpu_s": check_gpu, "kept": kept}
        else:
            row["self_check"] = {"skipped": "CPU wins the gain gate"}
        winner = "gpu" if device * (1.0 + gpu.PANEL3D_MIN_GAIN) < cpu else "cpu"
        entry["n"].append(n)
        entry["cpu"].append(cpu)
        entry["gpu"].append(device)
        entry["winner"].append(winner)
        rows.append(row)
        say(f"panel3d {n:5d} panels: CPU {cpu:.4f} s, GPU {device:.4f} s; {winner} after gain gate and self-check")
    entry["crossover_panels"] = crossover_work(
        entry["n"], entry["cpu"], [t * (1.0 + gpu.PANEL3D_MIN_GAIN) for t in entry["gpu"]],
    )
    elapsed = time.perf_counter() - start
    say(f"panel3d crossover {entry['crossover_panels']} panels; tuner step {elapsed:.2f} s")
    cpu_lu_choice, cpu_lu_meas = _measure_cpu_lu(say)
    entry["cpu_lu"] = cpu_lu_choice
    return {precision: entry, "cpu_lu": cpu_lu_choice}, {"rows": rows, "seconds": elapsed, "cpu_lu": cpu_lu_meas}


def crossover_work(works: list[float], cpu_times: list[float], gpu_times: list[float]) -> float | None:
    """Return the smallest work where the GPU median beats the CPU median, or None.

    Parameters
    ----------
    works : list of float
        Work of each measured lattice (horseshoe evaluations, dimensionless
        count), in any order.
    cpu_times, gpu_times : list of float
        Median times [s] on the CPU and on the GPU at each work.

    Returns
    -------
    float or None
        The crossover work: log-linear interpolation of the log time ratio
        between the bracketing sizes. The smallest work when the GPU wins at
        every size; None when the CPU wins at every size.
    """
    import math

    if not (len(works) == len(cpu_times) == len(gpu_times)) or not works:
        raise ValueError("works, cpu_times and gpu_times must be non-empty lists of the same length.")
    prev: tuple[float, float, float] | None = None
    for w, c, g in sorted(zip(works, cpu_times, gpu_times), key=lambda p: p[0]):
        if g < c:
            if prev is None:
                return float(w)
            w0, c0, g0 = prev
            r0 = math.log(c0 / g0)
            r1 = math.log(c / g)
            f = r0 / (r0 - r1)
            return math.exp(math.log(w0) + f * (math.log(w) - math.log(w0)))
        prev = (w, c, g)
    return None


def crossover_cases(cpu_1: float, cpu_33: float, gpu_1: float, gpu_33: float) -> float | None:
    """Return the smallest case count where the GPU batch beats the CPU batch, or None.

    Linear interpolation of the time difference between 1 and 33 cases [s].
    1.0 when the GPU already wins at 1 case; None when the CPU still wins
    at 33 cases.
    """
    d1 = cpu_1 - gpu_1
    d33 = cpu_33 - gpu_33
    if d1 >= 0.0:
        return 1.0
    if d33 <= 0.0:
        return None
    return 1.0 + 32.0 * (-d1) / (d33 - d1)


def crossover_from_rows(rows: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, int], dict[str, Any]]:
    """Return the GPU crossover thresholds per solver family from the GPU step rows.

    For each family, ``min_work`` is the crossover work at 1 case (free-air
    work: cases times panels squared) and ``min_cases`` is the crossover
    case count at the smallest measured lattice. A family without a measured
    crossover keeps the built-in default (no entry). The third value holds
    the details for the measurements record.
    """
    min_work: dict[str, float] = {}
    min_cases: dict[str, int] = {}
    details: dict[str, Any] = {}
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_family.setdefault(row["family"], []).append(row)
    for family, fam_rows in by_family.items():
        fam_rows = sorted(fam_rows, key=lambda r: r["panels"])
        ordered = [(r["panels"] * r["panels"], r["seconds"]["cpu_1"], r["seconds"]["gpu_1"]) for r in fam_rows]
        w = crossover_work([p[0] for p in ordered], [p[1] for p in ordered], [p[2] for p in ordered])
        first = fam_rows[0]["seconds"]
        k = crossover_cases(first["cpu_1"], first["cpu_33"], first["gpu_1"], first["gpu_33"])
        details[family] = {"panels": [r["panels"] for r in fam_rows],
                           "crossover_work": w, "crossover_cases": k}
        if w is not None:
            min_work[family] = float(w)
        if k is not None:
            min_cases[family] = max(1, int(round(k)))
    return min_work, min_cases, details


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
    # A quick run keeps two batch sizes (the cost model needs at least two),
    # a shorter warm-up and fewer repeats, so that it stays under 60 s.
    batches = {"cpu": _GPU_BATCHES["cpu"], "gpu": _GPU_BATCHES["gpu"][:2] if quick else _GPU_BATCHES["gpu"]}
    warm = 0.3 if quick else 0.5
    gpu_reps = 3 if quick else 5
    old_dev, old_prec = gpu.get_device(), gpu._precision
    precision = gpu.get_precision()
    solvers = {"vlm": vt.VortexLatticeSolver, "linear": vt.LinearLLTSolver, "nonlinear": vt.NonlinearLLTSolver}
    rows = []
    model: dict[str, dict[str, Any]] = {}
    try:
        for family, lattices in _GPU_CASES[bool(quick)].items():
            sec = {"airfoil": _tuning_airfoil()} if family == "nonlinear" else {}
            entry: dict[str, Any] = {"n": []}
            entry.update({dev: {"k": list(ks), "t": []} for dev, ks in batches.items()})
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
                for dev, ks in batches.items():
                    gpu.set_device(dev)
                    times = []
                    for K in ks:
                        alphas = np.radians(np.linspace(-4.0, 10.0, K))

                        def fn(solver=solver, ac=ac, cond=cond, st=st, alphas=alphas):
                            return solver.solve_sweep(ac, cond, st, alphas)

                        # A GPU after an idle time runs at low clocks: the
                        # model holds the times of a GPU in use.
                        times.append(_time_warm(fn, gpu_reps, warm) if dev == "gpu" else _time(fn, 3))
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


# VJP backends of the gradient step that are not measured per size class and
# device: numpy and the PyTorch reference copy are 10 to 100 times slower than
# the compiled kernels on the CPU at medium and large sizes.
_GRAD_SKIP = {"cpu": {"medium": ("numpy", "torch"), "large": ("numpy", "torch")}, "gpu": {}}


def _grad_fn(ac, st, device: str):
    """Return fn(backend) that runs one forward and backward pass of the gradient partials.

    The pass is the one of :func:`ventorum.gradients` for the outputs CL,
    CD and Cm with respect to alpha and the nodes: the differentiable
    reference copy with its O(n^2) parts (system rows, load points,
    Trefftz points) in the VJP backend *backend* (``"torch"`` is the
    PyTorch reference copy).
    """
    import torch

    import ventorum as vt
    from ventorum.sensitivity import adjoint

    wrt, outs = ("alpha", "nodes"), ("CL", "CD", "Cm")
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    prep = adjoint._prepare_case(ac, cond, st, "body", wrt, outs, "reference")
    model, base = adjoint.build_reference(prep["aircraft"], cond, prep["settings"], prep["lattice"], "body",
                                          device, prep["precision"], prep["symmetric"])
    g = torch.as_tensor(prep["g_unknown"], dtype=model.b["dtype"], device=model.b["device"])
    t = adjoint.make_input_tensors(model, base, wrt, device, prep["precision"])
    leaves = [v for v in t.values() if isinstance(v, torch.Tensor) and v.requires_grad]

    def fn(backend: str):
        model.b["vjp_backend"] = None if backend == "torch" else backend
        out, A, rhs = model.forward(t, g)
        loss = sum(out[o] for o in outs) + (A @ g - rhs).sum()
        grads = torch.autograd.grad(loss, leaves, allow_unused=True)
        if device == "cuda":
            torch.cuda.synchronize()
        return grads

    return fn


def _measure_grad_vjp(cls: str, ac, st, cores: int, say, quick: bool = False) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
    """Time the VJP backends of the gradient partials for one case size; return the choices and the times.

    The choice per device (``"cpu"``, ``"gpu"``) is the fastest backend
    (a compiled backend wins a tie with the reference copy). A quick run
    measures the CPU device only.
    """
    from ventorum import gpu
    from ventorum.sensitivity import kernels as K
    from ventorum.utils import parallel as par

    choices: dict[str, str] = {}
    times: dict[str, dict[str, float]] = {}
    devices = ["cpu"] if quick else ["cpu"] + (["gpu"] if K.available("warp", "cuda") and gpu.available() else [])
    reps = _repeats(cls, quick)
    for dev in devices:
        torch_dev = "cuda" if dev == "gpu" else "cpu"
        names = list(K.GPU_BACKENDS if dev == "gpu" else K.CPU_BACKENDS) + ["torch"]
        names = [b for b in names if K.available(b, torch_dev) and b not in _GRAD_SKIP[dev].get(cls, ())]
        fn = _grad_fn(ac, st, torch_dev)
        times[dev] = {}
        for backend in names:
            with par.forced_single_threads(cores), par.solve_threads(None):
                if dev == "gpu":
                    times[dev][backend] = _time_warm(lambda b=backend, f=fn: f(b), reps)
                else:
                    times[dev][backend] = _time(lambda b=backend, f=fn: f(b), reps)
        best = min(times[dev].values())
        close = [b for b, t in times[dev].items() if t <= best * (1.0 + _TIE)]
        order = ("numba", "cython", "warp", "numpy", "torch")
        choices[dev] = next(b for b in order if b in close)
        say(f"{cls:6s} gradient VJP {dev}: {choices[dev]} (" + ", ".join(
            f"{b} {1e3 * t:.1f} ms" for b, t in times[dev].items()) + ")")
    return choices, times


# Cases of one batch in the batch-threads step: the sweep size of the fixed
# workloads, so the measurement uses the same path as the sweeps.
_BATCH_THREADS_CASES = 33
# Solver families measured in the batch-threads step.
_BATCH_THREADS_FAMILIES = ("vlm", "linear", "nonlinear")
# A thread count is written only when it beats all cores by more than
# this fraction (median times).
_BATCH_THREADS_MIN_GAIN = 0.05
_SINGLE_THREADS_MIN_GAIN = 0.05
# The tuner self-check drops the entries when a fixed workload with them is
# slower than without them by more than this fraction.
_SELF_CHECK_MAX_SLOWDOWN = 0.02
# Repeats of one timing of the thread steps and of the self-check.
_BATCH_THREADS_REPEATS = 3
_SINGLE_THREADS_REPEATS = 5
# Candidate chunk caps [MiB] of the ground-effect step.
_CHUNK_CANDIDATES_MIB = (64.0, 256.0, 1024.0)
_DEFAULT_CHUNK_MIB = 256.0


def _batch_threads_keep(timings: dict[int, float], cores: int) -> int | None:
    """Return the thread count to write, or None for all cores (no entry).

    The fastest count wins (fewer threads win a tie, see :func:`_pick`), but
    only when its median time beats all cores by more than
    ``_BATCH_THREADS_MIN_GAIN``. If no measured setting can meet this, the
    tuner writes no entry and the sweeps use all cores, the default.
    """
    best = _pick(timings)
    if best == cores:
        return None
    if timings[best] * (1.0 + _BATCH_THREADS_MIN_GAIN) < timings[cores]:
        return best
    return None


def _single_threads_keep(timings: dict[int, float], cores: int) -> int | None:
    """Return the thread count to write, or None for all cores (no entry).

    The fastest count wins (fewer threads win a tie, see :func:`_pick`), but
    only when its median time beats all cores by more than
    ``_SINGLE_THREADS_MIN_GAIN``. If no measured setting can meet this, the
    tuner writes no entry and single solves use all cores, the default.
    """
    best = _pick(timings)
    if best == cores:
        return None
    if timings[best] * (1.0 + _SINGLE_THREADS_MIN_GAIN) < timings[cores]:
        return best
    return None


def _self_check_drop(times: dict[str, tuple[float, float]], kind: str = "batch") -> str | None:
    """Return why the entries must be dropped, or None to keep them.

    *times* maps each workload name to ``(with_s, without_s)``: the median
    time [s] of the workload with the new entries and without them. A
    workload that is slower with the entries by more than
    ``_SELF_CHECK_MAX_SLOWDOWN`` drops all entries of that kind.
    """
    noun = "batch entries" if kind == "batch" else "single entries"
    for name, (with_s, without_s) in times.items():
        if with_s > without_s * (1.0 + _SELF_CHECK_MAX_SLOWDOWN):
            slower = 100.0 * (with_s - without_s) / without_s
            return (f"{name} is {slower:.1f} % slower with the tuned {kind} threads "
                    f"({with_s:.3f} s against {without_s:.3f} s): drop the {noun}")
    return None


def _batch_threads_solver(family: str):
    """Return a solver of the batch-threads family *family*."""
    from ventorum.solvers.horseshoe import VortexLatticeSolver
    from ventorum.solvers.linear import LinearLLTSolver
    from ventorum.solvers.nonlinear import NonlinearLLTSolver

    return {"vlm": VortexLatticeSolver, "linear": LinearLLTSolver,
            "nonlinear": NonlinearLLTSolver}[family]()


def _measure_batch_threads(family: str, cls: str, ac, st, cores: int, say) -> tuple[int | None, dict[str, float]]:
    """Time a 33-case batch of solver family *family* at each thread count.

    The batch runs alone on the CPU with the kernel choices of step 1, on
    the same path that the sweeps use (:meth:`solve_batch` with the CPU
    path). Return the thread count to write and the times [s]. The count is
    None when no setting beats all cores by more than 5 % (see
    :func:`_batch_threads_keep`); the caller then writes no entry and the
    sweeps use all cores.
    """
    import ventorum as vt
    from ventorum.utils import parallel as par

    solver = _batch_threads_solver(family)
    if family == "vlm":
        fam_st = vt.SolverSettings(solver_type="vlm", n_panels=st.n_panels, n_chord=st.n_chord)
    else:
        fam_st = vt.SolverSettings(solver_type=family, n_panels=st.n_panels)
    cond = vt.FlightCondition(V_inf=30.0)
    lat = solver.build(ac, fam_st, cond)
    conds = [vt.FlightCondition(V_inf=30.0, alpha=float(a))
             for a in np.radians(np.linspace(-4.0, 10.0, _BATCH_THREADS_CASES))]

    def run():
        return solver.solve_batch(lat, conds, fam_st, ac.S_ref, ac.b_ref, ac.c_ref, _cpu_only=True)

    run()  # warm-up (the first batch call builds the batch path)
    timings = {}
    for t in _thread_candidates(cores):
        with par.forced_single_threads(t):
            timings[t] = _time(run, _BATCH_THREADS_REPEATS)
    best = _batch_threads_keep(timings, cores)
    if best is None:
        say(f"{cls:6s} batch threads {family:9s}: no setting beats all cores by more than "
            f"{100.0 * _BATCH_THREADS_MIN_GAIN:.0f} % ({1e3 * timings[cores]:.1f} ms for "
            f"{_BATCH_THREADS_CASES} cases on all cores): no entry")
    else:
        say(f"{cls:6s} batch threads {family:9s}: {best} threads, {1e3 * timings[best]:.1f} ms for "
            f"{_BATCH_THREADS_CASES} cases (all cores {1e3 * timings[cores]:.1f} ms)")
    return best, {str(k): v for k, v in timings.items()}


def _ground_chunk_workload():
    """Return (sweep, heights, alphas) of the chunk timing workload.

    A serial ground-effect sweep of 220 linear cases on 100 strips: about
    70 MB of dense matrices, so the smallest candidate cap (64 MiB) solves
    it in two chunks and the larger caps in one.
    """
    import ventorum as vt
    from ventorum.ground_effect.sweep import GroundEffectSweep

    wing = vt.LiftingSurface(name="TuningWing", semi_span=5.0, sections=[
        vt.WingSection(y_frac=0.0, chord=1.6), vt.WingSection(y_frac=1.0, chord=0.9)])
    st = vt.SolverSettings(solver_type="linear", n_panels=100)
    sweep = GroundEffectSweep(wing, settings=st, backend="serial")
    return sweep, np.linspace(1.0, 5.0, 4), np.linspace(-4.0, 10.0, 55)


def _measure_chunks(quick: bool, say, grad_choice: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    """Return the tuned chunk caps per path [MiB] and the measurements.

    The ground-effect cap times a serial sweep at each candidate cap (the
    default wins a tie: larger caps only remove per-chunk overhead, and the
    default is the safe memory bound). The gradient budget times one small
    forward and backward pass at budgets around the default (same tie rule).
    A quick run keeps the defaults without timing them.
    """
    from ventorum.sensitivity import reference as ref

    meas: dict[str, Any] = {}
    if quick:
        say("chunks   quick mode keeps the default caps (the full mode measures them)")
        return {"ground_cpu": _DEFAULT_CHUNK_MIB,
                "gradient": float(ref._DEFAULT_CHUNK_VALUES * 8.0 / 1024 / 1024)}, \
            {"ground_cpu": {"choice": _DEFAULT_CHUNK_MIB, "reason": "quick mode: default, not measured"},
             "gradient": {"choice": "default", "reason": "quick mode: default, not measured"}}

    # Ground-effect CPU batch. One timing is noisy (tens of percent), so each
    # cap runs once warm and three times timed; the default wins a tie.
    sweep, heights, alphas = _ground_chunk_workload()
    ground_times: dict[str, float] = {}
    for cap in _CHUNK_CANDIDATES_MIB:
        with _profile_in_process({"settings": {"chunk_mib": {"ground_cpu": cap}}}):
            ground_times[str(cap)] = _time(lambda: sweep.run_sweep(heights, alphas, (0.0,), V_inf=30.0), 3)
    best_cap = min(ground_times, key=ground_times.get)
    close = [c for c in _CHUNK_CANDIDATES_MIB if ground_times[str(c)] <= ground_times[best_cap] * (1.0 + _TIE)]
    ground_choice = _DEFAULT_CHUNK_MIB if _DEFAULT_CHUNK_MIB in close else float(best_cap)
    say(f"chunks   ground CPU batch: {ground_choice:g} MiB (" + ", ".join(
        f"{c:g} MiB {ground_times[str(c)]:.2f} s" for c in _CHUNK_CANDIDATES_MIB) + ")")
    meas["ground_cpu"] = {"caps_mib": list(_CHUNK_CANDIDATES_MIB),
                          "seconds": ground_times, "choice_mib": ground_choice}

    # Adjoint gradient chunks.
    base_values = float(ref._DEFAULT_CHUNK_VALUES)
    grad_times: dict[str, float] = {}
    if not _grad_available():
        meas["gradient"] = {"choice_mib": base_values * 8.0 / 1024 / 1024,
                            "reason": "PyTorch is not installed: default, not measured"}
        return {"ground_cpu": float(ground_choice),
                "gradient": float(base_values * 8.0 / 1024 / 1024)}, meas
    ac, st = _wing(*_CASES["small"])
    fn = _grad_fn(ac, st, "cpu")
    backend = (grad_choice.get("small", {}).get("cpu") if isinstance(grad_choice, dict) else None)
    if backend not in ("numba", "cython", "warp", "numpy", "torch"):
        backend = "numba"
    import os

    old_env = os.environ.get("VENTORUM_GRAD_CHUNK")
    try:
        for factor in (0.25, 1.0, 4.0):
            os.environ["VENTORUM_GRAD_CHUNK"] = repr(base_values * factor)
            grad_times[str(factor)] = _time(lambda: fn(backend), 2)
    finally:
        if old_env is None:
            os.environ.pop("VENTORUM_GRAD_CHUNK", None)
        else:
            os.environ["VENTORUM_GRAD_CHUNK"] = old_env
    best_factor = min(grad_times, key=grad_times.get)
    close_f = [f for f in ("0.25", "1.0", "4.0") if grad_times[f] <= grad_times[best_factor] * (1.0 + _TIE)]
    grad_choice_mib = base_values * 8.0 / 1024 / 1024 if "1.0" in close_f else base_values * float(best_factor) * 8.0 / 1024 / 1024
    say(f"chunks   gradient: {grad_choice_mib:.1f} MiB (" + ", ".join(
        f"x{f} {1e3 * grad_times[f]:.0f} ms" for f in ("0.25", "1.0", "4.0")) + ")")
    meas["gradient"] = {"factors": ["0.25", "1.0", "4.0"], "seconds": grad_times, "choice_mib": grad_choice_mib}
    return {"ground_cpu": float(ground_choice), "gradient": float(grad_choice_mib)}, meas


def _self_check_workloads():
    """Return the fixed sweep workloads of the batch-threads self-check.

    The same sweeps as the fixed workloads ``sweep_vlm_n40_c4_33`` and
    ``sweep_linear_n80_33`` of ``validation/perf_history.py``: a rectangular
    wing (semi-span 5 m, chord 1.25 m), 33 angles from -4 to 12 deg.
    """
    import ventorum as vt

    wing = vt.LiftingSurface(
        name="SelfCheckWing",
        semi_span=5.0,
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.25, airfoil=vt.LinearAirfoil()),
            vt.WingSection(y_frac=1.0, chord=1.25, airfoil=vt.LinearAirfoil()),
        ],
    )
    alphas = np.linspace(-4.0, 12.0, 33)
    out = {}
    for name, solver, panels, n_chord in (("sweep_vlm_n40_c4_33", "vlm", 40, 4),
                                          ("sweep_linear_n80_33", "linear", 80, None)):
        st = vt.SolverSettings(solver_type=solver, n_panels=panels, n_chord=n_chord)

        def run(wing=wing, alphas=alphas, st=st):
            return vt.analyze_sweep(wing, alphas, settings=st)

        out[name] = run
    return out


def _self_check_single_workloads():
    """Return the fixed single workloads of the single-threads self-check.

    The same single workloads as ``single_vlm_n20_c4`` and ``single_vlm_n80_c4``
    of ``validation/perf_history.py``: a rectangular wing (semi-span 5 m, chord
    1.25 m) at 4 deg angle of attack.
    """
    import ventorum as vt

    wing = vt.LiftingSurface(
        name="SelfCheckWing",
        semi_span=5.0,
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.25, airfoil=vt.LinearAirfoil()),
            vt.WingSection(y_frac=1.0, chord=1.25, airfoil=vt.LinearAirfoil()),
        ],
    )
    out = {}
    for name, panels, n_chord in (("single_vlm_n20_c4", 20, 4),
                                  ("single_vlm_n80_c4", 80, 4)):
        st = vt.SolverSettings(solver_type="vlm", n_panels=panels, n_chord=n_chord)

        def run(wing=wing, st=st):
            return vt.analyze(wing, alpha_deg=4.0, settings=st)

        out[name] = run
    return out


def _self_check_single_threads(settings, hw, quick, say, prof, vt) -> dict[str, Any]:
    """Time two fixed single solves with and without the single entries; drop them if slower.

    A full tuner run ends with this check: each workload runs with the new
    ``single`` entries and without them (all cores). If either workload is
    slower with the entries by more than ``_SELF_CHECK_MAX_SLOWDOWN``, all
    single entries are dropped from *settings* and the reason is printed.
    Returns the measurements record. A quick run skips the check. It never
    changes the results, only the thread counts.
    """
    if quick or not settings.get("single"):
        return {"skipped": "quick mode" if quick else "no single entries measured"}
    workloads = _self_check_single_workloads()
    full = {k: dict(v) for k, v in settings.items() if k in ("kernels", "single", "batch", "batch_threads")}
    times: dict[str, tuple[float, float]] = {}
    with _profile_in_process(_make_profile(prof, full, {}, hw, quick, 0.0, vt)):
        with_times = {name: _time(fn, _SINGLE_THREADS_REPEATS) for name, fn in workloads.items()}
    stripped = {k: v for k, v in full.items() if k != "single"}
    with _profile_in_process(_make_profile(prof, stripped, {}, hw, quick, 0.0, vt)):
        without_times = {name: _time(fn, _SINGLE_THREADS_REPEATS) for name, fn in workloads.items()}
    for name in workloads:
        times[name] = (with_times[name], without_times[name])
    reason = _self_check_drop(times, kind="single")
    if reason is None:
        say("self-check single threads: the fixed single solves keep their speed "
            + ", ".join(f"{n} {w:.3f} s with, {wo:.3f} s without" for n, (w, wo) in times.items()))
        return {"workloads": {n: {"with_s": w, "without_s": wo} for n, (w, wo) in times.items()},
                "kept": True}
    settings["single"] = {}
    say(f"self-check single threads: {reason}")
    return {"workloads": {n: {"with_s": w, "without_s": wo} for n, (w, wo) in times.items()},
            "kept": False, "reason": reason}


def _self_check_batch_threads(settings, hw, quick, say, prof, vt) -> dict[str, Any]:
    """Time two fixed sweeps with and without the batch entries; drop them if slower.

    A full tuner run ends with this check: each workload runs with the new
    ``batch_threads`` entries and without them (all cores). If either
    workload is slower with the entries by more than
    ``_SELF_CHECK_MAX_SLOWDOWN``, all batch entries are dropped from
    *settings* and the reason is printed. Returns the measurements record.
    A quick run skips the check. It never changes the results, only the
    thread counts.
    """
    if quick or not settings.get("batch_threads"):
        return {"skipped": "quick mode" if quick else "no batch entries measured"}
    workloads = _self_check_workloads()
    full = {k: dict(v) for k, v in settings.items() if k in ("kernels", "single", "batch", "batch_threads")}
    times: dict[str, tuple[float, float]] = {}
    with _profile_in_process(_make_profile(prof, full, {}, hw, quick, 0.0, vt)):
        with_times = {name: _time(fn, _BATCH_THREADS_REPEATS) for name, fn in workloads.items()}
    stripped = {k: v for k, v in full.items() if k != "batch_threads"}
    with _profile_in_process(_make_profile(prof, stripped, {}, hw, quick, 0.0, vt)):
        without_times = {name: _time(fn, _BATCH_THREADS_REPEATS) for name, fn in workloads.items()}
    for name in workloads:
        times[name] = (with_times[name], without_times[name])
    reason = _self_check_drop(times)
    if reason is None:
        say("self-check batch threads: the fixed sweeps keep their speed "
            + ", ".join(f"{n} {w:.3f} s with, {wo:.3f} s without" for n, (w, wo) in times.items()))
        return {"workloads": {n: {"with_s": w, "without_s": wo} for n, (w, wo) in times.items()},
                "kept": True}
    settings["batch_threads"] = {}
    say(f"self-check batch threads: {reason}")
    return {"workloads": {n: {"with_s": w, "without_s": wo} for n, (w, wo) in times.items()},
            "kept": False, "reason": reason}


def tune_machine(quick: bool = False, save: bool = True, verbose: bool = True) -> dict[str, Any]:
    """Measure this machine and return (and by default save) its profile.

    Parameters
    ----------
    quick : bool
        Skip the large cases and trim the repeats, the GPU batches and the
        gradient devices, so that the run stays under 60 s. The full mode
        measures everything and reports its time.
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
    meas: dict[str, Any] = {"single": {}, "batch": {}, "kernels": {}, "grad_vjp": {},
                             "batch_threads": {}, "crossover": {}, "chunks": {}}
    settings: dict[str, Any] = {"single": {}, "batch": {}, "kernels": {}, "grad_vjp": {},
                                "batch_threads": {}}

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
                reps = _repeats(cls, quick) if quick else max(_SINGLE_THREADS_REPEATS, _repeats(cls, False))
                for t in _thread_candidates(cores):
                    cand_prof = _make_profile(prof, {
                        "kernels": dict(settings["kernels"]),
                        "single": {cls: {"threads": t}},
                    }, {}, hw, quick, 0.0, vt)
                    with _profile_in_process(cand_prof):
                        single[t] = _time(lambda ac=ac, st=st: vt.analyze(ac, alpha_deg=4.0, settings=st), reps)
                best_t = _single_threads_keep(single, cores)
                if best_t is not None:
                    settings["single"][cls] = {"threads": best_t}
                meas["single"][cls] = {
                    "panels": n_pan,
                    "seconds": {str(k): v for k, v in single.items()},
                    "choice": best_t,
                    "reason": ("beats all cores by more than 5 %" if best_t is not None
                               else "no setting beats all cores by more than 5 %: no entry"),
                }
                if best_t is None:
                    say(f"{cls:6s} ({n_pan} panels) one case: no setting beats all cores by more than "
                        f"{100.0 * _SINGLE_THREADS_MIN_GAIN:.0f} % ({1e3 * single[cores]:.1f} ms on all cores): no entry")
                else:
                    say(f"{cls:6s} ({n_pan} panels) one case: {best_t} threads, {1e3 * single[best_t]:.1f} ms "
                        f"(all cores {1e3 * single[cores]:.1f} ms)")

                # 3. Batch: workers x threads per worker.
                n_tasks = {"small": 4 * cores, "medium": 2 * cores, "large": max(2, cores)}[cls]
                alphas = np.linspace(-4.0, 10.0, n_tasks)
                # Warm-up: the first sweep builds the lattice and compiles
                # the batch path. Without it the first candidate pays that
                # one-time cost, which biases the choice and wastes time.
                vt.analyze_sweep(ac, alphas, settings=st, n_jobs=min(2, cores))
                batch = {}
                for w in _thread_candidates(cores):
                    t0 = time.perf_counter()
                    vt.analyze_sweep(ac, alphas, settings=st, n_jobs=w)
                    batch[w] = n_tasks / (time.perf_counter() - t0)
                best_w = max(batch, key=batch.get)
                settings["batch"][cls] = {"workers": best_w, "threads": max(1, cores // best_w)}
                meas["batch"][cls] = {"panels": n_pan, "cases_per_second": {str(k): v for k, v in batch.items()}}
                say(f"{cls:6s} batch: {best_w} cases in parallel x {cores // best_w} threads, {batch[best_w]:.0f} cases/s")

                # 3b. Threads of one case inside a multi-case batch that runs
                # alone (a 33-case batch of each solver family on one
                # lattice, on the sweep path). A cell is written only when
                # it beats all cores by more than 5 %; else the sweeps use
                # all cores.
                for family in _BATCH_THREADS_FAMILIES:
                    best_bt, bt_times = _measure_batch_threads(family, cls, ac, st, cores, say)
                    meas["batch_threads"].setdefault(family, {})[cls] = {
                        "panels": n_pan, "cases": _BATCH_THREADS_CASES,
                        "seconds": bt_times,
                        "choice": best_bt,
                        "reason": ("beats all cores by more than 5 %" if best_bt is not None
                                   else "no setting beats all cores by more than 5 %: no entry"),
                    }
                    if best_bt is not None:
                        settings["batch_threads"].setdefault(family, {})[cls] = best_bt

            # 4. GPU pipelines: the cost model of the CPU and GPU paths. The
            # CPU times use the settings of steps 1 to 3.
            step3 = _make_profile(prof, {k: dict(settings[k]) for k in ("kernels", "single", "batch")}, {}, hw,
                                  quick, 0.0, vt)
            with _profile_in_process(step3):
                gpu_settings, gpu_meas = _measure_gpu(quick, say)
                panel_settings, panel_meas = _measure_panel3d(quick, say)
                if panel_settings:
                    gpu_settings["panel3d"] = panel_settings
                meas["panel3d"] = panel_meas

                # 6. GPU thresholds from the measured crossover: the smallest
                # work where the GPU median beats the CPU median per family,
                # at 1 case (min_work) and at 33 cases (min_cases). No extra
                # timing: the values come from the rows of step 4.
                min_work, min_cases, cross_meas = crossover_from_rows(gpu_meas.get("lattices", []))
                gpu_settings["min_work"] = min_work
                gpu_settings["min_cases"] = min_cases
                meas["crossover"] = cross_meas
                for family in sorted({r["family"] for r in gpu_meas.get("lattices", [])}):
                    work = f"{min_work[family]:.3g}" if family in min_work else "default (CPU wins in range)"
                    cases = str(min_cases[family]) if family in min_cases else "default (CPU wins in range)"
                    say(f"crossover {family:9s} min_work {work}, min_cases {cases}")

                # 5. Backend of the vector-Jacobian products of the adjoint gradients.
                if _grad_available():
                    for cls in classes:
                        ac, st = _wing(*_CASES[cls])
                        choice, times = _measure_grad_vjp(cls, ac, st, cores, say, quick)
                        settings["grad_vjp"][cls] = choice
                        meas["grad_vjp"][cls] = times

                # 7. Chunk caps per path (the full mode measures them; a
                # quick run keeps the defaults).
                chunk_mib, chunk_meas = _measure_chunks(quick, say, settings["grad_vjp"])
                meas["chunks"] = chunk_meas
    # 8. Self-check of a full run: the fixed workloads with the new single and batch
    # entries must keep their speed against all cores.
    meas["single_threads_self_check"] = _self_check_single_threads(settings, hw, quick, say, prof, vt)
    meas["batch_threads_self_check"] = _self_check_batch_threads(settings, hw, quick, say, prof, vt)
    settings["gpu"] = {**_gpu_info(), **gpu_settings}
    meas["gpu"] = gpu_meas
    settings["chunk_mib"] = chunk_mib
    settings["cython_threads"] = _cython_threads()
    settings["torch_device"] = _torch_device()
    return _make_profile(prof, settings, meas, hw, quick, time.perf_counter() - t_start, vt)
