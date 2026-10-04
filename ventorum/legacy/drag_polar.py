# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Drag polar and CL-alpha sweep plots.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ventorum.legacy.core.datatypes import (
    Aircraft,
    FlightCondition,
    SolverResult,
    SolverSettings,
)


def _sweep_worker_func(args: tuple) -> SolverResult:
    """General worker function for fourier and nonlinear solvers."""
    solver, aircraft, condition, settings, alpha_val = args
    cond = FlightCondition(
        V_inf=condition.V_inf,
        alpha=float(alpha_val),
        beta=condition.beta,
        rho=condition.rho,
        h=condition.h,
    )
    return solver.solve(aircraft, cond, settings)


def _sweep_worker_nonlinear_cached(args: tuple) -> SolverResult:
    """Worker function for nonlinear solver reusing precomputed geometry cache."""
    solver, geo_cache, condition, settings, S_ref, b_ref, c_ref, alpha_val = args
    cond = FlightCondition(
        V_inf=condition.V_inf,
        alpha=float(alpha_val),
        beta=condition.beta,
        rho=condition.rho,
        h=condition.h,
    )
    return solver.solve_precomputed(geo_cache, cond, settings, S_ref, b_ref, c_ref)


def _sweep_worker_horseshoe_cached(args: tuple) -> SolverResult:
    """Ultra-fast worker function using precomputed bound-vortex geometry."""
    solver, geo_cache, condition, S_ref, b_ref, c_ref, alpha_val = args
    cond = FlightCondition(
        V_inf=condition.V_inf,
        alpha=float(alpha_val),
        beta=condition.beta,
        rho=condition.rho,
        h=condition.h,
    )
    return solver.solve_precomputed(geo_cache, cond, S_ref, b_ref, c_ref)


def _sweep_chunk_worker_horseshoe(args: tuple) -> list[SolverResult]:
    """Execute a chunk of alpha angles in a single worker using precomputed geometry."""
    solver, geo_cache, condition, S_ref, b_ref, c_ref, alphas = args
    results = []
    for a in alphas:
        cond = FlightCondition(
            V_inf=condition.V_inf,
            alpha=float(a),
            beta=condition.beta,
            rho=condition.rho,
            h=condition.h,
        )
        results.append(solver.solve_precomputed(geo_cache, cond, S_ref, b_ref, c_ref))
    return results


def _sweep_chunk_worker_nonlinear(args: tuple) -> list[SolverResult]:
    """Execute a chunk of alpha angles in a single worker using precomputed geometry."""
    solver, geo_cache, condition, settings, S_ref, b_ref, c_ref, alphas = args
    results = []
    for a in alphas:
        cond = FlightCondition(
            V_inf=condition.V_inf,
            alpha=float(a),
            beta=condition.beta,
            rho=condition.rho,
            h=condition.h,
        )
        results.append(solver.solve_precomputed(geo_cache, cond, settings, S_ref, b_ref, c_ref))
    return results


def alpha_sweep(
    aircraft: Aircraft,
    condition: FlightCondition,
    settings: SolverSettings,
    alpha_range: np.ndarray | None = None,
    n_jobs: int | str = "auto",
    backend: str = "auto",
    progress: bool = False,
) -> list[SolverResult]:
    """Run the solver over a range of angles of attack with adaptive hardware acceleration.

    Parameters
    ----------
    aircraft : Aircraft
    condition : FlightCondition
        Base condition (alpha and h will be preserved, alpha overridden).
    settings : SolverSettings
    alpha_range : np.ndarray or None
        Angles of attack [rad]. Defaults to -2° to 12° in 1° steps.
    n_jobs : int or str
        Number of parallel workers/threads ('auto', 1 = serial, >1 = parallel,
        -1 = all available CPU cores). Defaults to 'auto'.
    backend : str
        Concurrency/hardware backend: 'auto' (GPU batched if CUDA available, CPU BLAS
        otherwise), 'gpu'/'cuda', 'thread', or 'process'. Defaults to 'auto'.
    progress : bool
        If True, displays a live terminal progress bar during the sweep.

    Returns
    -------
    list[SolverResult]
        One result per alpha.
    """
    import os
    # Import solvers here to avoid circular imports
    from ventorum.legacy.solvers.fourier import FourierSolver
    from ventorum.legacy.solvers.horseshoe import HorseshoeSolver
    from ventorum.legacy.solvers.nonlinear import NonlinearSolver
    from ventorum.legacy.core.datatypes import TabulatedAirfoil
    from ventorum.legacy.geometry.processing import discretize_surface
    from ventorum.legacy.aero.influence import precompute_horseshoe_geometry
    from ventorum.legacy.utils.validation import validate_aircraft
    from ventorum.legacy.utils.linalg import get_optimal_hardware_backend

    if alpha_range is None:
        alpha_range = np.radians(np.arange(-2, 13, 1.0))

    # Fast path: Fourier solver multi-RHS vectorized solve
    if settings.solver_type == "fourier":
        res = FourierSolver().solve_sweep(aircraft, condition, settings, alpha_range)
        if progress:
            from ventorum.legacy.utils.progress import ProgressBar
            pb = ProgressBar(total=len(alpha_range), title="Alpha Sweep (Fourier)", unit="pts")
            pb.update(len(alpha_range))
            pb.finish("Sweep complete")
        return res

    # Fast path: Modern Linear LLT solver precomputed sweep solve
    if settings.solver_type in ("linear", "linear_llt"):
        from ventorum.legacy.solvers.linear import LinearLLTSolver
        res = LinearLLTSolver().solve_sweep(aircraft, condition, settings, alpha_range)
        if progress:
            from ventorum.legacy.utils.progress import ProgressBar
            pb = ProgressBar(total=len(alpha_range), title="Alpha Sweep (Linear LLT)", unit="pts")
            pb.update(len(alpha_range))
            pb.finish("Sweep complete")
        return res

    # Decide which solver to use
    has_tabulated = any(
        isinstance(sec.airfoil, TabulatedAirfoil)
        for surf in aircraft.surfaces
        for sec in surf.sections
    )

    use_horseshoe = settings.solver_type in ("horseshoe", "vlm", "gpu_horseshoe") or (
        not has_tabulated and settings.solver_type not in ("nonlinear", "tabular", "gpu_nonlinear")
    )

    # Fast path: GPU batched acceleration via adaptive hardware selection
    from ventorum.legacy.aero.gpu_influence import has_cuda
    n_surfaces = len(aircraft.surfaces)
    b_len = len(alpha_range)
    opt_backend = get_optimal_hardware_backend(
        n_panels=settings.n_panels,
        n_surfaces=n_surfaces,
        is_sweep=True,
        n_cases=b_len,
        is_ground_effect=(condition.h is not None),
        is_nonlinear=(not use_horseshoe),
        user_backend=backend,
    )
    use_gpu = (settings.solver_type in ("gpu", "gpu_horseshoe", "gpu_nonlinear")) or (opt_backend == "gpu")
    if use_gpu and has_cuda():
        try:
            if not use_horseshoe:
                from ventorum.legacy.solvers.gpu_nonlinear import GPUNonlinearSolver
                gpu_solver = GPUNonlinearSolver()
            else:
                from ventorum.legacy.solvers.gpu_horseshoe import GPUHorseshoeSolver
                gpu_solver = GPUHorseshoeSolver()

            if progress:
                from ventorum.legacy.utils.progress import ProgressBar
                pb = ProgressBar(
                    total=len(alpha_range),
                    title=f"Alpha Sweep ({'GPU Batched' if use_horseshoe else 'GPU Nonlinear'})",
                    unit="pts",
                )
            results = gpu_solver.solve_sweep(aircraft, condition, settings, alpha_range)
            if progress and pb:
                pb.update(len(alpha_range))
                pb.finish("Sweep complete (GPU)")
            return results
        except Exception:
            # Fall back to CPU gracefully if GPU solve fails
            pass

    # Determine worker count on CPU using machine calibration
    from ventorum.legacy.hardware.config import get_active_setting
    if n_jobs == "auto" or n_jobs is None:
        workers = int(get_active_setting("optimal_sweep_workers", default=1))
    elif isinstance(n_jobs, int) and n_jobs < 0:
        workers = os.cpu_count() or 1
    else:
        workers = max(1, int(n_jobs))

    # Fast path: serial execution on CPU reusing precomputed geometry (fastest on CPU)
    if workers == 1 and not progress:
        if not use_horseshoe:
            return NonlinearSolver().solve_sweep(aircraft, condition, settings, alpha_range)
        else:
            return HorseshoeSolver().solve_sweep(aircraft, condition, settings, alpha_range)

    validate_aircraft(aircraft)
    aircraft.compute_reference_values()

    from ventorum.legacy.core.symmetry import can_use_symmetry
    from ventorum.legacy.geometry.processing import discretize_aircraft_surfaces
    use_sym = can_use_symmetry(aircraft, condition, settings)

    disc_surfaces = discretize_aircraft_surfaces(aircraft, settings, half_mesh=use_sym)
    geo_cache = precompute_horseshoe_geometry(disc_surfaces, h=condition.h, use_symmetry=use_sym)

    if not use_horseshoe:
        solver = NonlinearSolver()
        worker_func = _sweep_worker_nonlinear_cached
        task_args = [
            (solver, geo_cache, condition, settings, aircraft.S_ref, aircraft.b_ref, aircraft.c_ref, a)
            for a in alpha_range
        ]
    else:
        solver = HorseshoeSolver()
        worker_func = _sweep_worker_horseshoe_cached
        task_args = [
            (solver, geo_cache, condition, aircraft.S_ref, aircraft.b_ref, aircraft.c_ref, a)
            for a in alpha_range
        ]

    # Resolve execution backend for parallel workers
    effective_backend = backend.lower()
    if effective_backend == "auto":
        effective_backend = str(get_active_setting("optimal_sweep_backend", default="thread")).lower()

    pb = None
    if progress:
        from ventorum.legacy.utils.progress import ProgressBar
        mode_str = "serial" if workers == 1 else f"{workers} workers ({effective_backend})"
        pb = ProgressBar(
            total=len(alpha_range),
            title=f"Alpha Sweep ({mode_str})",
            unit="pts",
        )

    results: list[SolverResult] = []

    if workers == 1:

        # Serial execution with progress bar
        for arg in task_args:
            res = worker_func(arg)
            results.append(res)
            if pb:
                pb.update(1, status=f"a={np.degrees(arg[-1]):.1f} deg")
    else:
        # Multithreaded or multiprocess execution
        if effective_backend == "process":
            from concurrent.futures import ProcessPoolExecutor
            pool_cls = ProcessPoolExecutor
        else:
            from concurrent.futures import ThreadPoolExecutor
            pool_cls = ThreadPoolExecutor

        with pool_cls(max_workers=workers) as executor:
            if pb:
                futures = [executor.submit(worker_func, arg) for arg in task_args]
                for fut in futures:
                    res = fut.result()
                    results.append(res)
                    pb.update(1)
            else:
                n_chunks = min(workers, len(alpha_range))
                chunks = np.array_split(alpha_range, n_chunks)
                if has_tabulated:
                    chunk_args = [
                        (solver, geo_cache, condition, settings, aircraft.S_ref, aircraft.b_ref, aircraft.c_ref, c)
                        for c in chunks
                    ]
                    chunked_results = list(executor.map(_sweep_chunk_worker_nonlinear, chunk_args))
                else:
                    chunk_args = [
                        (solver, geo_cache, condition, aircraft.S_ref, aircraft.b_ref, aircraft.c_ref, c)
                        for c in chunks
                    ]
                    chunked_results = list(executor.map(_sweep_chunk_worker_horseshoe, chunk_args))
                results = [r for sub in chunked_results for r in sub]

    if pb:
        pb.finish("Sweep complete")

    return results


def plot_drag_polar(
    results: list[SolverResult],
    ax: plt.Axes | None = None,
) -> Figure:
    """Plot CL vs CDi (drag polar).

    Parameters
    ----------
    results : list[SolverResult]
        Results from an alpha sweep.
    ax : plt.Axes or None

    Returns
    -------
    Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    CLs = [r.totals.CL for r in results]
    CDis = [r.totals.CDi for r in results]

    ax.plot(CDis, CLs, "o-", markersize=4, color="#2563eb", linewidth=2,
            label="CL vs CDi")

    # Add total drag if available
    if results[0].totals.CD_total is not None:
        CDs = [r.totals.CD_total for r in results]
        ax.plot(CDs, CLs, "s--", markersize=4, color="#dc2626", linewidth=1.5,
                label="CL vs CD_total")

    ax.set_xlabel("CD")
    ax.set_ylabel("CL")
    ax.set_title("Drag Polar")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_cl_vs_alpha(
    results: list[SolverResult],
    alpha_range: np.ndarray | None = None,
    ax: plt.Axes | None = None,
) -> Figure:
    """Plot CL vs α.

    Parameters
    ----------
    results : list[SolverResult]
    alpha_range : np.ndarray or None
        α values [rad] corresponding to *results*.
    ax : plt.Axes or None

    Returns
    -------
    Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    CLs = [r.totals.CL for r in results]

    if alpha_range is not None:
        alphas_deg = np.degrees(alpha_range)
    else:
        alphas_deg = np.arange(len(CLs))

    ax.plot(alphas_deg, CLs, "o-", markersize=4, color="#2563eb", linewidth=2)

    # Annotate CL_alpha (slope from linear region)
    if len(CLs) >= 3:
        # Use central region
        mid = len(CLs) // 2
        lo, hi = max(0, mid - 2), min(len(CLs), mid + 3)
        if hi - lo >= 2:
            slope = np.polyfit(np.radians(alphas_deg[lo:hi]), CLs[lo:hi], 1)[0]
            ax.annotate(
                f"CL_α ≈ {slope:.3f} /rad\n({slope * np.pi / 180:.4f} /°)",
                xy=(alphas_deg[mid], CLs[mid]),
                xytext=(alphas_deg[mid] + 2, CLs[mid] - 0.15),
                arrowprops=dict(arrowstyle="->", color="gray"),
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8),
            )

    ax.set_xlabel("α [°]")
    ax.set_ylabel("CL")
    ax.set_title("Lift Curve")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_sweep_summary(
    results: list[SolverResult],
    alpha_range: np.ndarray,
) -> Figure:
    """Create a 1×2 summary: CL-α and drag polar side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_cl_vs_alpha(results, alpha_range, ax=axes[0])
    plot_drag_polar(results, ax=axes[1])
    fig.suptitle("Alpha Sweep Summary", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig
