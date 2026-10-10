# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Speed of the original (legacy) code against the verified core.

The recovery audit (``docs/design/recovery_audit.md``) needs
measured speed numbers wherever a legacy component and a present component
do the same work. This script measures them. It prints Markdown tables with
the median wall time of each case; the audit copies them with the date and
the machine.

Two groups of cases:

1. Kernels. The influence matrix of N straight horseshoe vortices (bound
   segment and two semi-infinite legs) at N points, for each backend of the
   legacy code (numpy, Numba, Cython, PyTorch) and of the core (numpy,
   Numba, Cython). The core takes the same straight horseshoes (its
   chordwise segments have zero length). The script also checks that all
   paths give the same matrix.
2. Solves. One solve and one alpha sweep of a rectangular wing with each
   legacy solver and with its present equivalent. The legacy horseshoe and
   GPU solvers have known physics defects (``tests/test_legacy.py``), so a
   speed ratio there compares two different results; the table says so.

Usage:
    python validation/legacy_speed.py [--quick]

``--quick`` uses fewer sizes and repeats (for a test run).
"""

from __future__ import annotations

import importlib.util
import platform
import sys
import time
import warnings

import numpy as np

warnings.simplefilter("ignore")

QUICK = "--quick" in sys.argv
HAS_TORCH = importlib.util.find_spec("torch") is not None


def _median_time(fn, repeats: int) -> float:
    """Return the median wall time [s] of *fn*, after one warm-up call."""
    fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def _ms(t: float) -> str:
    return f"{1e3 * t:.2f}" if np.isfinite(t) else "n/a"


def _table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


# ── 1. kernels ───────────────────────────────────────────────────────────────

def _horseshoes(n: int, alpha: float = np.radians(5.0)):
    """Straight horseshoes on a rectangular wing of span 10 m and chord 1 m, with 3/4-chord points."""
    y = np.linspace(-5.0, 5.0, n + 1)
    nl = np.column_stack([np.full(n, 0.25), y[:-1], np.zeros(n)])
    nr = np.column_stack([np.full(n, 0.25), y[1:], np.zeros(n)])
    P = np.column_stack([np.full(n, 0.75), 0.5 * (y[:-1] + y[1:]), np.zeros(n)])
    nrm = np.tile([0.0, 0.0, 1.0], (n, 1))
    td = np.array([np.cos(alpha), 0.0, np.sin(alpha)])
    return nl, nr, P, nrm, td


def _kernel_paths(n: int, rc: float) -> dict:
    """Return the functions that compute the influence matrix of one size with each path."""
    from ventorum.aero import vortex as V
    from ventorum.legacy.aero import numba_kernels as lnb
    from ventorum.legacy.aero.acceleration import acceleration_context
    from ventorum.legacy.aero.cython_accel import cython_kernels as lcy
    from ventorum.legacy.aero.influence import compute_horseshoe_velocity_matrix
    from ventorum.utils import parallel as par

    nl, nr, P, nrm, td = _horseshoes(n)
    paths = {}

    def legacy_numpy():
        with acceleration_context("numpy"):
            return np.einsum("ijk,ik->ij", compute_horseshoe_velocity_matrix(P, nl, nr, td, gamma=1.0, rc=rc), nrm)

    paths["legacy numpy"] = legacy_numpy
    if lnb.HAS_NUMBA:
        paths["legacy numba"] = lambda: lnb._numba_aic_matrix(P, nl, nr, nrm, td, 1.0, rc)
    if lcy is not None:
        paths["legacy cython (1 thread)"] = lambda: lcy.cy_aic_matrix(P, nl, nr, nrm, td, 1.0, rc)
    if HAS_TORCH:
        import torch

        from ventorum.legacy.aero.gpu_influence import compute_batched_aic_and_rhs, get_device

        dev = get_device()
        tt = [torch.from_numpy(np.ascontiguousarray(x)).to(dev, torch.float64) for x in (P, nl, nr, nrm, td)]
        vinf = tt[4] * 25.0

        def legacy_torch():
            a, _ = compute_batched_aic_and_rhs(*tt, vinf, rc=rc)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            return a.cpu().numpy()

        paths[f"legacy torch ({dev.type})"] = legacy_torch

    hs = V.HorseshoeSet(a=nl, b=nr, a_te=nl.copy(), b_te=nr.copy(), wake_dir=np.tile(td, (n, 1)),
                        rc=np.full(n, rc), sign=np.ones(n), column=np.arange(n, dtype=np.int64), group=None)

    def core_path(backend: str):
        def run():
            V.set_kernel_backend(backend)
            # The same thread count as the legacy Numba kernel (all cores), not
            # the count of the machine profile: equal threads, a fair comparison.
            with par.forced_single_threads(par.cpu_cores()), par.solve_threads(n):
                return V.influence_matrix(P, nrm, hs, n_unknowns=n)
        return run

    backends = ["numpy", "numba"]
    if importlib.util.find_spec("ventorum.aero.vortex_cython") is not None:
        backends.append("cython")
    for backend in backends:
        paths[f"core {backend}"] = core_path(backend)
    return paths


def kernel_cases() -> tuple[str, dict]:
    from ventorum.aero import vortex as V

    rc = 1e-10
    sizes = (160, 640) if QUICK else (160, 640, 1600)
    repeats = 3 if QUICK else 7
    rows, worst = [], 0.0
    for n in sizes:
        paths = _kernel_paths(n, rc)
        ref = paths["legacy numpy"]()
        scale = float(np.max(np.abs(ref)))
        times = {}
        for name, fn in paths.items():
            worst = max(worst, float(np.max(np.abs(fn() - ref))) / scale)
            times[name] = _median_time(fn, repeats)
        V.set_kernel_backend("auto")
        best_legacy = min(t for k, t in times.items() if k.startswith("legacy"))
        best_core = min(t for k, t in times.items() if k.startswith("core"))
        for name, t in times.items():
            rows.append([str(n), name, _ms(t)])
        rows.append([str(n), "**best legacy / best core**", f"{best_legacy / best_core:.2f}"])
    text = "\n".join([
        "### Influence-matrix kernels (straight horseshoes, N vortices at N points)", "",
        f"Median time in ms. Largest difference of any path from the legacy numpy matrix: {worst:.1e} "
        "(relative to the largest entry). The legacy and core Numba and Cython paths use all CPU cores, "
        "except the legacy Cython kernel, which has one thread.", "",
        _table(["N", "path", "time [ms]"], rows), ""])
    return text, {"kernel_max_rel_diff": worst}


# ── 2. solves ────────────────────────────────────────────────────────────────

def _rect(lib):
    af = lib.LinearAirfoil()
    return lib.LiftingSurface(semi_span=5.0, sections=[lib.WingSection(y_frac=0.0, chord=1.25, airfoil=af),
                                                       lib.WingSection(y_frac=1.0, chord=1.25, airfoil=af)])


def _tab(lib):
    a = np.radians(np.arange(-10.0, 21.0))
    taf = lib.TabulatedAirfoil(alpha=a, Cl_data=2 * np.pi * a, Cd_data=np.zeros_like(a))
    return lib.LiftingSurface(semi_span=5.0, sections=[lib.WingSection(y_frac=0.0, chord=1.25, airfoil=taf),
                                                       lib.WingSection(y_frac=1.0, chord=1.25, airfoil=taf)])


def solve_cases() -> str:
    import ventorum as vt
    import ventorum.legacy as L

    n = 80
    repeats = 3 if QUICK else 5
    alphas = np.radians(np.linspace(-4.0, 12.0, 9 if QUICK else 33))
    lc = L.FlightCondition(V_inf=25.0, alpha=np.radians(5.0))
    cc = vt.FlightCondition(V_inf=25.0, alpha=np.radians(5.0))

    def legacy_case(solver_cls, settings_type, tab=False, **kw):
        ac = L.Aircraft(surfaces=[_tab(L) if tab else _rect(L)])
        st = L.SolverSettings(solver_type=settings_type, n_panels=n)
        s = solver_cls(**kw)
        return (lambda: s.solve(ac, lc, st)), (lambda: s.solve_sweep(ac, lc, st, alphas))

    def core_case(solver_cls, settings_type, tab=False, **kw):
        w = _tab(vt) if tab else _rect(vt)
        st = vt.SolverSettings(solver_type=settings_type, n_panels=n, spacing="cosine", **kw)
        s = solver_cls()
        return (lambda: s.solve(w, cc, st)), (lambda: s.solve_sweep(w, cc, st, alphas))

    pairs = [
        ("linear lifting line", legacy_case(L.LinearLLTSolver, "linear"),
         core_case(vt.LinearLLTSolver, "linear"), "same model"),
        ("Fourier lifting line", legacy_case(L.FourierSolver, "fourier"),
         core_case(vt.FourierSolver, "fourier"), "same model"),
        ("horseshoe (legacy) / VLM, 1 chordwise panel (core)", legacy_case(L.HorseshoeSolver, "horseshoe"),
         core_case(vt.HorseshoeSolver, "vlm", n_chord=1), "legacy has known defects"),
        ("horseshoe (legacy) / VLM, 4 chordwise panels (core)", legacy_case(L.HorseshoeSolver, "horseshoe"),
         core_case(vt.HorseshoeSolver, "vlm", n_chord=4), "legacy has known defects"),
        ("nonlinear lifting line (tabulated polar)", legacy_case(L.NonlinearSolver, "nonlinear", tab=True),
         core_case(vt.NonlinearSolver, "nonlinear", tab=True), "legacy gives about half the lift"),
    ]
    if HAS_TORCH:
        dev = "cuda" if L.has_cuda() else "cpu"
        pairs += [
            (f"GPU horseshoe on {dev} (legacy) / VLM, 1 chordwise panel (core)",
             legacy_case(L.GPUHorseshoeSolver, "horseshoe", device=dev),
             core_case(vt.HorseshoeSolver, "vlm", n_chord=1), "legacy has known defects"),
            (f"GPU nonlinear on {dev} (legacy) / nonlinear lifting line (core)",
             legacy_case(L.GPUNonlinearSolver, "nonlinear", tab=True, device=dev),
             core_case(vt.NonlinearSolver, "nonlinear", tab=True), "legacy gives about half the lift"),
        ]
    rows = []
    for name, (l1, ls), (c1, cs), note in pairs:
        cl_old = l1().totals.CL
        cl_new = c1().totals.CL
        t = [_median_time(f, repeats) for f in (l1, c1, ls, cs)]
        rows.append([name, f"{cl_old:.4f}", f"{cl_new:.4f}", _ms(t[0]), _ms(t[1]), f"{t[0] / t[1]:.2f}",
                     _ms(t[2]), _ms(t[3]), f"{t[2] / t[3]:.2f}", note])
    return "\n".join([
        "### Solves of a rectangular wing (AR 8, 80 spanwise panels, cosine spacing)", "",
        f"One solve at alpha = 5 deg, and an alpha sweep of {len(alphas)} angles from -4 to 12 deg with "
        "the `solve_sweep` method of each solver. Median time in ms. A ratio above 1 means that the core "
        "is faster.", "",
        _table(["case", "CL legacy", "CL core", "1 solve legacy", "1 solve core", "legacy / core",
                "sweep legacy", "sweep core", "legacy / core", "note"], rows), ""])


def linear_solve_cases() -> str:
    from ventorum.legacy.utils.linalg import fast_linear_solve

    rng = np.random.default_rng(1)
    rows = []
    for n in ((160, 640) if QUICK else (160, 640, 1600, 3200)):
        A = rng.standard_normal((n, n)) + n * np.eye(n)
        b = rng.standard_normal(n)
        t_old = _median_time(lambda A=A, b=b: fast_linear_solve(A, b), 5)
        t_new = _median_time(lambda A=A, b=b: np.linalg.solve(A, b), 5)
        rows.append([str(n), _ms(t_old), _ms(t_new), f"{t_new / t_old:.2f}"])
    return "\n".join([
        "### Dense linear solve", "",
        "`legacy.utils.linalg.fast_linear_solve` (SciPy LAPACK dgesv) against `numpy.linalg.solve`, "
        "one right-hand side, median time in ms.", "",
        _table(["N", "dgesv (legacy)", "numpy.linalg.solve", "numpy / dgesv"], rows), ""])


def main() -> None:
    import ventorum
    from ventorum.utils import parallel as par

    head = [
        "## Measured speed: legacy code against the verified core", "",
        f"Generated by `validation/legacy_speed.py` (Python {platform.python_version()}, numpy {np.__version__}, "
        f"ventorum {getattr(ventorum, '__version__', '?')}, {par.cpu_cores()} CPU cores"
        + (", PyTorch present" if HAS_TORCH else ", no PyTorch") + ").", ""]
    k, _ = kernel_cases()
    print("\n".join(head) + "\n" + k + "\n" + solve_cases() + "\n" + linear_solve_cases())


if __name__ == "__main__":
    main()
