# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Performance and parallel benchmarking utilities for Ventorum.
"""

from __future__ import annotations

import os
import time
import numpy as np

from ventorum.legacy.core.datatypes import LiftingSurface, WingSection, LinearAirfoil, TabulatedAirfoil
from ventorum.legacy.utils.progress import ProgressBar


def create_test_aircraft(n_surfaces: int = 1, nonlinear: bool = False) -> list[LiftingSurface]:
    """Create a generic geometry for benchmarking."""
    surfaces = []
    
    if nonlinear:
        # Create a dummy tabulated airfoil to force the nonlinear solver to iterate
        alpha_data = np.linspace(np.radians(-10), np.radians(20), 30)
        cl_data = 2 * np.pi * alpha_data
        cd_data = 0.01 + 0.05 * alpha_data**2
        airfoil = TabulatedAirfoil(name="Dummy", alpha=alpha_data, Cl_data=cl_data, Cd_data=cd_data)
    else:
        airfoil = LinearAirfoil()
        
    for i in range(n_surfaces):
        surf = LiftingSurface(
            name=f"Surface_{i}",
            semi_span=5.0,
            position=np.array([i * 2.0, 0.0, 0.0]),  # stagger them
            sections=[
                WingSection(y_frac=0.0, chord=2.0, airfoil=airfoil),
                WingSection(y_frac=1.0, chord=1.0, airfoil=airfoil),
            ],
        )
        surfaces.append(surf)
    return surfaces


def run_performance_benchmark(
    panel_counts: tuple[int, ...] = (10, 20, 40, 80, 160),
    solvers: tuple[str, ...] = ("fourier", "horseshoe", "nonlinear"),
    n_repeats: int = 3,
    verbose: bool = True,
    show_progress: bool = True,
    use_symmetry: bool | None = None,
) -> dict:
    """Run a standardized performance benchmark across solvers and discretizations.
    
    Parameters
    ----------
    panel_counts : tuple of int
        Number of spanwise panels per semi-span.
    solvers : tuple of str
        Solvers to benchmark.
    n_repeats : int
        Number of repeats for timing (minimum is reported).
    verbose : bool
        If True, prints formatted summary table.
    show_progress : bool
        If True, renders real-time progress bar in the terminal.
    use_symmetry : bool or None
        Whether to enforce symmetry plane (None = default solver behavior).
    """
    from ventorum.legacy import analyze

    results = {s: [] for s in solvers}
    total_runs = len(panel_counts) * len(solvers) * n_repeats
    pb = ProgressBar(total=total_runs, title="Solver Scaling", unit="runs") if show_progress else None

    for n in panel_counts:
        for solver in solvers:
            is_nonlinear = (solver == "nonlinear")
            geom = create_test_aircraft(n_surfaces=1, nonlinear=is_nonlinear)

            # Warm-up run
            try:
                analyze(geom[0], alpha_deg=5.0, solver=solver, n_panels=n, use_symmetry=use_symmetry)
            except Exception:
                pass

            times = []
            for r in range(n_repeats):
                t0 = time.perf_counter()
                analyze(geom[0], alpha_deg=5.0, solver=solver, n_panels=n, use_symmetry=use_symmetry)
                t1 = time.perf_counter()
                times.append(t1 - t0)
                if pb:
                    pb.update(1, status=f"{solver.capitalize()} N={n} (run {r+1}/{n_repeats})")

            best_time_ms = min(times) * 1000.0
            results[solver].append(best_time_ms)

    if pb:
        pb.finish("All discretization runs finished")

    if verbose:
        sym_str = f" [Symmetry: {use_symmetry}]" if use_symmetry is not None else ""
        print("\n" + "=" * 65)
        print(f" Ventorum Discretization Scaling Benchmark{sym_str} ")
        print("=" * 65)
        header = f"{'n_panels':>10} | " + " | ".join(f"{s.capitalize() + ' (ms)':>14}" for s in solvers)
        print(header)
        print("-" * 65)
        for i, n in enumerate(panel_counts):
            row = f"{n:>10} | " + " | ".join(f"{results[s][i]:>14.2f}" for s in solvers)
            print(row)
        print("=" * 65 + "\n")

    return {"panel_counts": list(panel_counts), "times_ms": results, "use_symmetry": use_symmetry}


def run_sweep_benchmark(
    alpha_range: tuple[float, float, int] = (-4.0, 12.0, 17),
    solvers: tuple[str, ...] = ("fourier", "horseshoe", "nonlinear"),
    n_panels: int = 40,
    n_surfaces: int = 1,
    n_repeats: int = 3,
    verbose: bool = True,
    show_progress: bool = True,
    use_symmetry: bool | None = None,
) -> dict:
    """Benchmark angle-of-attack sweeps across different solvers."""
    from ventorum.legacy import analyze_sweep

    alpha_array = np.linspace(alpha_range[0], alpha_range[1], alpha_range[2])
    results = {s: 0.0 for s in solvers}
    total_runs = len(solvers) * n_repeats
    pb = ProgressBar(total=total_runs, title="Alpha Sweep", unit="sweeps") if show_progress else None

    for solver in solvers:
        is_nonlinear = (solver == "nonlinear")
        geom = create_test_aircraft(n_surfaces=n_surfaces, nonlinear=is_nonlinear)

        # Warm-up
        try:
            analyze_sweep(
                geom[0],
                alpha_deg_range=np.array([0.0]),
                solver=solver,
                n_panels=n_panels,
                use_symmetry=use_symmetry,
            )
        except Exception:
            pass

        times = []
        for r in range(n_repeats):
            t0 = time.perf_counter()
            analyze_sweep(
                geom[0],
                alpha_deg_range=alpha_array,
                solver=solver,
                n_panels=n_panels,
                use_symmetry=use_symmetry,
            )
            t1 = time.perf_counter()
            times.append(t1 - t0)
            if pb:
                pb.update(1, status=f"{solver.capitalize()} (run {r+1}/{n_repeats})")

        best_time_ms = min(times) * 1000.0
        results[solver] = best_time_ms

    if pb:
        pb.finish("All sweep runs finished")

    if verbose:
        print("\n" + "=" * 65)
        print(f" Ventorum Alpha Sweep Benchmark ({alpha_range[2]} points, {n_surfaces} surface(s))")
        print("=" * 65)
        print(f"{'Solver':>15} | {'Total Time (ms)':>15} | {'Time per alpha (ms)':>19}")
        print("-" * 65)
        for solver in solvers:
            time_per_alpha = results[solver] / alpha_range[2]
            print(f"{solver:>15} | {results[solver]:>15.2f} | {time_per_alpha:>19.2f}")
        print("=" * 65 + "\n")

    return {
        "alpha_points": alpha_range[2],
        "n_panels": n_panels,
        "n_surfaces": n_surfaces,
        "times_ms": results,
    }


def run_ground_effect_benchmark(
    panel_counts: tuple[int, ...] = (10, 20, 40, 80),
    n_repeats: int = 3,
    verbose: bool = True,
    show_progress: bool = True,
) -> dict:
    """Benchmark horseshoe solver with and without ground effect."""
    from ventorum.legacy import analyze
    from ventorum.legacy.core.datatypes import FlightCondition, SolverSettings

    results = {"free_air": [], "ground_effect": []}
    total_runs = len(panel_counts) * 2 * n_repeats
    pb = ProgressBar(total=total_runs, title="Ground Effect", unit="runs") if show_progress else None

    for n in panel_counts:
        geom = create_test_aircraft(n_surfaces=1, nonlinear=False)

        # Free air
        times_free = []
        for r in range(n_repeats):
            t0 = time.perf_counter()
            analyze(geom[0], alpha_deg=5.0, solver="horseshoe", n_panels=n)
            t1 = time.perf_counter()
            times_free.append(t1 - t0)
            if pb:
                pb.update(1, status=f"Free-Air N={n} (run {r+1}/{n_repeats})")
        best_free = min(times_free) * 1000.0
        results["free_air"].append(best_free)

        # Ground effect (Method of Images)
        times_ge = []
        cond = FlightCondition(alpha=np.radians(5.0), h=0.2)
        settings = SolverSettings(solver_type="horseshoe", n_panels=n)
        for r in range(n_repeats):
            t0 = time.perf_counter()
            analyze(geom[0], condition=cond, settings=settings)
            t1 = time.perf_counter()
            times_ge.append(t1 - t0)
            if pb:
                pb.update(1, status=f"Ground-Effect N={n} (run {r+1}/{n_repeats})")
        best_ge = min(times_ge) * 1000.0
        results["ground_effect"].append(best_ge)

    if pb:
        pb.finish("All ground effect runs finished")

    if verbose:
        print("\n" + "=" * 65)
        print(" Ventorum Ground Effect Overhead Benchmark ")
        print("=" * 65)
        print(f"{'n_panels':>10} | {'Free Air (ms)':>15} | {'Ground Effect (ms)':>19} | {'Overhead':>10}")
        print("-" * 65)
        for i, n in enumerate(panel_counts):
            free = results["free_air"][i]
            ge = results["ground_effect"][i]
            mult = ge / free if free > 0 else 0
            print(f"{n:>10} | {free:>15.2f} | {ge:>19.2f} | {mult:>9.2f}x")
        print("=" * 65 + "\n")

    return {"panel_counts": list(panel_counts), "times_ms": results}


def run_multithread_benchmark(
    workers_list: tuple[int, ...] | None = None,
    alpha_range: tuple[float, float, int] = (-4.0, 12.0, 33),
    n_panels: int = 40,
    solver: str = "horseshoe",
    backend: str = "thread",
    n_repeats: int = 3,
    verbose: bool = True,
    show_progress: bool = True,
) -> dict:
    """Benchmark multithreaded scaling across worker configurations up to full machine capacity.
    
    Parameters
    ----------
    workers_list : tuple of int or None
        List of worker counts to benchmark. Defaults to [1, 2, 4, 8, max_cpus].
    alpha_range : tuple
        (start_alpha, end_alpha, num_points). Default is 33 points.
    n_panels : int
        Spanwise panels per semi-span.
    solver : str
        Solver to use ('horseshoe', 'nonlinear', or 'fourier').
    backend : str
        'thread' (ThreadPoolExecutor) or 'process' (ProcessPoolExecutor).
    n_repeats : int
        Number of test repetitions per worker count.
    verbose : bool
        If True, prints a formatted scaling report table.
    show_progress : bool
        If True, displays live progress in the terminal.
    """
    from ventorum.legacy import analyze_sweep

    cpu_max = os.cpu_count() or 1
    if workers_list is None:
        raw_list = [1, 2, 4, 8, cpu_max]
        # Deduplicate, keep sorted, and clamp to cpu_max
        workers_list = tuple(sorted(list({w for w in raw_list if w <= cpu_max} | {1, cpu_max})))

    alpha_array = np.linspace(alpha_range[0], alpha_range[1], alpha_range[2])
    geom = create_test_aircraft(n_surfaces=1, nonlinear=(solver == "nonlinear"))

    total_runs = len(workers_list) * n_repeats
    pb = ProgressBar(total=total_runs, title="Multithread Scaling", unit="runs") if show_progress else None

    # Warm-up run
    try:
        analyze_sweep(geom[0], alpha_deg_range=np.array([0.0, 1.0]), solver=solver, n_panels=n_panels, n_jobs=1)
    except Exception:
        pass

    times_ms = []
    mean_times_ms = []

    for w in workers_list:
        run_times = []
        for r in range(n_repeats):
            t0 = time.perf_counter()
            analyze_sweep(
                geom[0],
                alpha_deg_range=alpha_array,
                solver=solver,
                n_panels=n_panels,
                n_jobs=w,
                backend=backend,
            )
            t1 = time.perf_counter()
            run_times.append(t1 - t0)
            if pb:
                pb.update(1, status=f"workers={w} (run {r+1}/{n_repeats})")

        best_ms = min(run_times) * 1000.0
        mean_ms = (sum(run_times) / len(run_times)) * 1000.0
        times_ms.append(best_ms)
        mean_times_ms.append(mean_ms)

    if pb:
        pb.finish("Multithread benchmark complete")

    # Compute speedup and parallel efficiency relative to 1 worker (serial)
    t_serial = times_ms[0]
    speedups = [t_serial / t if t > 0 else 1.0 for t in times_ms]
    efficiencies = [(sp / w) * 100.0 for sp, w in zip(speedups, workers_list)]
    time_per_alpha = [t / alpha_range[2] for t in times_ms]

    if verbose:
        print("\n" + "=" * 88)
        print(f" Ventorum Multithread Scaling Benchmark ({alpha_range[2]} alphas, N={n_panels}, Max CPUs={cpu_max})")
        print("=" * 88)
        print(f"{'Workers':>8} | {'Total Time (ms)':>16} | {'Time/Alpha (ms)':>16} | {'Speedup':>9} | {'Efficiency':>11} | {'Scaling Note':<15}")
        print("-" * 88)
        for w, t_tot, t_alpha, sp, eff in zip(workers_list, times_ms, time_per_alpha, speedups, efficiencies):
            if w == 1:
                note = "Baseline (Serial)"
            elif eff >= 80.0:
                note = "Excellent"
            elif eff >= 60.0:
                note = "Good"
            elif eff >= 40.0:
                note = "Moderate"
            else:
                note = "Sub-linear"
            if w == cpu_max and w > 1:
                note += " [Full CPU]"
            print(f"{w:>8} | {t_tot:>16.2f} | {t_alpha:>16.2f} | {sp:>8.2f}x | {eff:>10.1f}% | {note:<15}")
        print("=" * 88 + "\n")

    return {
        "workers": list(workers_list),
        "cpu_max": cpu_max,
        "alpha_points": alpha_range[2],
        "n_panels": n_panels,
        "solver": solver,
        "times_ms": times_ms,
        "mean_times_ms": mean_times_ms,
        "time_per_alpha_ms": time_per_alpha,
        "speedup": speedups,
        "efficiency": efficiencies,
    }


def _create_benchmark_study_cases(
    n_cases: int = 8,
    alphas_deg: np.ndarray | None = None,
    n_panels: int = 40,
) -> list:
    """Generate a diverse suite of unique aerodynamic test cases for multi-instance benchmarking."""
    from ventorum.legacy.core.datatypes import Aircraft, LiftingSurface, WingSection, FlightCondition, SolverSettings
    from ventorum.legacy.instance import Ventorum

    if alphas_deg is None:
        alphas_deg = np.linspace(-2.0, 10.0, 13)

    cases_defs = [
        # (name, b, c_root, c_tip, sweep_deg, dihedral_deg, twist_deg, V_inf)
        ("Sailplane_HighAR", 15.0, 1.1, 0.5, 0.0, 2.0, -1.5, 35.0),
        ("Transport_Swept", 12.0, 2.5, 0.8, 25.0, 4.0, -2.5, 75.0),
        ("Cropped_Delta", 6.0, 3.2, 0.6, 42.0, 0.0, 0.0, 90.0),
        ("LowAR_Dihedral", 7.0, 1.8, 1.2, 5.0, 8.0, -0.5, 45.0),
        ("Agile_ForwardSwept", 8.5, 2.0, 0.9, -15.0, 1.0, 1.0, 60.0),
        ("HighLift_Tapered", 14.0, 2.8, 1.4, 8.0, 3.0, -3.0, 50.0),
        ("Commuter_Straight", 11.0, 1.9, 1.1, 2.0, 2.5, -1.0, 65.0),
        ("Supersonic_Arrow", 7.5, 3.0, 0.5, 50.0, -2.0, 0.0, 100.0),
    ]

    instances = []
    for i in range(n_cases):
        def_idx = i % len(cases_defs)
        name_base, b, cr, ct, sweep, dih, twist, vinf = cases_defs[def_idx]
        case_name = f"{name_base}_{i+1}" if i >= len(cases_defs) else name_base

        sections = [
            WingSection(y_frac=0.0, chord=cr, twist=0.0),
            WingSection(y_frac=1.0, chord=ct, twist=float(np.radians(twist))),
        ]
        surf = LiftingSurface(
            name=f"{case_name}_Wing",
            semi_span=b / 2.0,
            sweep_le=float(np.radians(sweep)),
            dihedral=float(np.radians(dih)),
            sections=sections,
        )
        cond = FlightCondition(V_inf=vinf, alpha=0.0)
        sett = SolverSettings(solver_type="horseshoe", n_panels=n_panels)

        inst = Ventorum(
            name=case_name,
            geometry=surf,
            condition=cond,
            settings=sett,
            alpha_sweep_deg=alphas_deg,
            n_workers=1,
            backend="thread",
        )
        instances.append(inst)

    return instances


def run_multi_instance_benchmark(
    n_cases: int = 8,
    alphas_per_case: int = 15,
    n_panels: int = 40,
    test_configs: list[tuple[int, int]] | None = None,
    n_repeats: int = 2,
    verbose: bool = True,
    show_progress: bool = True,
) -> dict:
    """Benchmark parallel execution across multiple independent Ventorum instances
    with dedicated internal workers, verifying zero inter-instance interference.

    Parameters
    ----------
    n_cases : int
        Number of independent Ventorum cases in the benchmark suite (default 8).
    alphas_per_case : int
        Number of sweep angles per case.
    n_panels : int
        Panels per semi-span.
    test_configs : list of (instances_concurrent, workers_per_instance) or None
        Configurations to benchmark. If None, automatically selects configs up
        to system CPU capacity.
    n_repeats : int
        Number of repetitions per configuration. Minimum wall-clock time is reported.
    verbose : bool
        If True, prints formatted scaling and non-interference verification table.
    show_progress : bool
        If True, displays live terminal progress.

    Returns
    -------
    dict
        Structured benchmark results, throughput metrics, speedup factors,
        and mathematical verification of zero interference.
    """
    from ventorum.legacy.instance import run_parallel_instances

    cpu_max = os.cpu_count() or 1
    alphas = np.linspace(-2.0, 10.0, alphas_per_case)

    if test_configs is None:
        # Hierarchical configurations: (concurrent_instances, workers_per_instance)
        raw_configs = [
            (1, 1),   # Pure Serial baseline (1 instance, 1 worker)
            (1, 4),   # Single-instance with internal multithreading (4 workers)
            (2, 1),   # 2 instances concurrent, 1 worker each
            (2, 2),   # 2 instances concurrent, 2 workers each (4 active threads)
            (4, 1),   # 4 instances concurrent, 1 worker each (4 active threads)
            (4, 2),   # 4 instances concurrent, 2 workers each (8 active threads)
            (4, 4),   # 4 instances concurrent, 4 workers each (16 active threads)
            (8, 1),   # 8 instances concurrent, 1 worker each (8 active threads)
            (8, 2),   # 8 instances concurrent, 2 workers each (16 active threads)
        ]
        # Filter configurations that make sense for machine capacity
        test_configs = []
        for inst_c, w_per_inst in raw_configs:
            if inst_c <= n_cases and (inst_c * w_per_inst) <= max(16, cpu_max * 2):
                test_configs.append((inst_c, w_per_inst))

    # --- Step 1: Establish Gold-Standard Serial Reference for All Cases ---
    if verbose:
        print("\n" + "=" * 94)
        print(f" Ventorum Multi-Instance & Nested Worker Benchmark ({n_cases} Cases, {alphas_per_case} Alphas, CPUs={cpu_max})")
        print("=" * 94)
        print(">> Establishing serial single-case reference baselines for non-interference verification...")

    baseline_instances = _create_benchmark_study_cases(n_cases, alphas, n_panels)
    serial_results_map = {}
    for inst in baseline_instances:
        inst.set_workers(1)
        res = inst.run(progress=False)
        serial_results_map[inst.name] = [
            (r.totals.CL, r.totals.CDi, r.totals.e, np.copy(r.spanwise[0].gamma))
            for r in inst.sweep_results
        ]

    # --- Step 2: Benchmark Concurrency Matrix ---
    total_solves = n_cases * alphas_per_case
    results_records = []
    baseline_time_s = None

    pb = None
    if show_progress:
        pb = ProgressBar(
            total=len(test_configs) * n_repeats,
            title="Multi-Instance Bench",
            unit="runs",
        )

    for cfg_idx, (inst_c, w_per_inst) in enumerate(test_configs):
        active_workers = inst_c * w_per_inst
        run_times = []
        max_discrepancy = 0.0

        for rep in range(n_repeats):
            # Create fresh, clean instances for each run
            cases = _create_benchmark_study_cases(n_cases, alphas, n_panels)
            for c in cases:
                c.set_workers(w_per_inst)

            t0 = time.perf_counter()
            run_parallel_instances(
                cases,
                max_concurrent_instances=inst_c,
                instance_backend="thread",
                show_progress=False,
            )
            t1 = time.perf_counter()
            run_times.append(t1 - t0)

            if pb:
                pb.update(1, status=f"inst={inst_c}, w={w_per_inst} (run {rep+1})")

            # Verify non-interference: compare every single result against serial baseline
            for c in cases:
                ser_data = serial_results_map[c.name]
                for alpha_idx, r in enumerate(c.sweep_results):
                    ser_cl, ser_cdi, ser_e, ser_gamma = ser_data[alpha_idx]
                    diff_cl = abs(r.totals.CL - ser_cl)
                    diff_cdi = abs(r.totals.CDi - ser_cdi)
                    diff_e = abs(r.totals.e - ser_e) if (abs(ser_cl) > 1e-3 and abs(r.totals.CL) > 1e-3) else 0.0
                    diff_gamma = float(np.max(np.abs(r.spanwise[0].gamma - ser_gamma)))
                    max_discrepancy = max(max_discrepancy, diff_cl, diff_cdi, diff_e, diff_gamma)

        best_time_s = min(run_times)
        mean_time_s = sum(run_times) / len(run_times)

        if cfg_idx == 0:
            baseline_time_s = best_time_s

        speedup = baseline_time_s / best_time_s if best_time_s > 0 else 1.0
        solves_sec = total_solves / best_time_s if best_time_s > 0 else 0.0
        cases_sec = n_cases / best_time_s if best_time_s > 0 else 0.0
        # Efficiency normalized to min(active_workers, cpu_max)
        denom_workers = min(active_workers, cpu_max)
        efficiency = (speedup / max(1, denom_workers)) * 100.0

        # Interference check
        zero_interference = max_discrepancy < 1e-10

        record = {
            "concurrent_instances": inst_c,
            "workers_per_instance": w_per_inst,
            "total_active_workers": active_workers,
            "best_time_s": best_time_s,
            "mean_time_s": mean_time_s,
            "speedup": speedup,
            "efficiency": efficiency,
            "solves_per_sec": solves_sec,
            "cases_per_sec": cases_sec,
            "max_discrepancy": max_discrepancy,
            "zero_interference_verified": zero_interference,
        }
        results_records.append(record)

    if pb:
        pb.finish("Multi-instance benchmark complete")

    if verbose:
        print("\n" + "=" * 98)
        print(f" Ventorum Multi-Instance Benchmark Results ({n_cases} Cases, {total_solves} Total Solves)")
        print("=" * 98)
        header = f"{'Config (Inst x Workers)':<25} | {'Active W':>8} | {'Time (s)':>9} | {'Throughput':>12} | {'Speedup':>8} | {'Eff (%)':>8} | {'Max Diff':>10} | {'Interference':<10}"
        print(header)
        print("-" * 98)
        for r in results_records:
            cfg_label = f"{r['concurrent_instances']} inst x {r['workers_per_instance']} workers"
            w_act = r["total_active_workers"]
            t_s = r["best_time_s"]
            thru = f"{r['solves_per_sec']:.1f} sol/s"
            sp = f"{r['speedup']:.2f}x"
            eff = f"{r['efficiency']:.1f}%"
            diff_str = f"{r['max_discrepancy']:.1e}"
            interf = "ZERO (PASS)" if r["zero_interference_verified"] else "FAIL"
            print(f"{cfg_label:<25} | {w_act:>8} | {t_s:>9.3f} | {thru:>12} | {sp:>8} | {eff:>8} | {diff_str:>10} | {interf:<10}")
        print("=" * 98 + "\n")

    return {
        "n_cases": n_cases,
        "alphas_per_case": alphas_per_case,
        "total_solves": total_solves,
        "n_panels": n_panels,
        "cpu_max": cpu_max,
        "records": results_records,
    }


def run_gpu_benchmark(
    panel_counts: tuple[int, ...] = (20, 40, 80, 160, 320),
    sweep_sizes: tuple[int, ...] = (10, 50, 200, 1000, 10000, 100000),
    n_repeats: int = 3,
    verbose: bool = True,
) -> dict[str, Any]:
    """Benchmark GPU vs CPU across discretization and batched sweep regimes."""
    from ventorum.legacy.aero.gpu_influence import has_cuda
    from ventorum.legacy.solvers.horseshoe import HorseshoeSolver
    from ventorum.legacy.solvers.gpu_horseshoe import GPUHorseshoeSolver
    from ventorum.legacy.core.datatypes import Aircraft, FlightCondition, SolverSettings

    if not has_cuda():
        raise RuntimeError("CUDA GPU is required to run run_gpu_benchmark.")

    geom = create_test_aircraft(n_surfaces=1, nonlinear=False)[0]
    ac = Aircraft(name="BenchmarkWing", surfaces=[geom])
    ac.compute_reference_values()
    cond = FlightCondition(V_inf=50.0, alpha=np.radians(5.0))

    cpu_solver = HorseshoeSolver()
    gpu_solver = GPUHorseshoeSolver()

    # 1. Discretization Benchmark
    disc_results = []
    for n in panel_counts:
        sett = SolverSettings(solver_type="horseshoe", n_panels=n)

        # Warmup
        cpu_solver.solve(ac, cond, sett)
        gpu_solver.solve(ac, cond, sett)

        t_cpu = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            cpu_solver.solve(ac, cond, sett)
            t_cpu.append((time.perf_counter() - t0) * 1000.0)

        t_gpu = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            gpu_solver.solve(ac, cond, sett)
            t_gpu.append((time.perf_counter() - t0) * 1000.0)

        best_cpu = min(t_cpu)
        best_gpu = min(t_gpu)
        sp = best_cpu / best_gpu if best_gpu > 0 else 0.0
        disc_results.append({
            "n_panels": n,
            "cpu_ms": best_cpu,
            "gpu_ms": best_gpu,
            "speedup": sp,
        })

    # 2. Batched Sweep Benchmark
    sweep_results = []
    sett_40 = SolverSettings(solver_type="horseshoe", n_panels=40)
    for B in sweep_sizes:
        alphas = np.radians(np.linspace(-4.0, 12.0, B))

        # Warmup
        if B >= 50000:
            gpu_solver.solve_sweep(ac, cond, sett_40, alphas[:1000])
        else:
            gpu_solver.solve_sweep(ac, cond, sett_40, alphas)

        repeats = 1 if B >= 100000 else (2 if B >= 10000 else n_repeats)
        t_gpu = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            gpu_solver.solve_sweep(ac, cond, sett_40, alphas)
            t_gpu.append((time.perf_counter() - t0) * 1000.0)

        best_t_ms = min(t_gpu)
        throughput = B / (best_t_ms / 1000.0)
        time_per_solve_us = (best_t_ms / B) * 1000.0
        sweep_results.append({
            "batch_size": B,
            "total_ms": best_t_ms,
            "throughput_solves_sec": throughput,
            "time_per_solve_us": time_per_solve_us,
        })

    if verbose:
        print("\n" + "=" * 70)
        print(" Ventorum GPU PERFORMANCE & SCALING BENCHMARK ")
        print("=" * 70)
        print("--- Discretization Scaling (Single Solve CPU vs GPU) ---")
        print(f"{'Panels (N)':<12} | {'CPU (ms)':>10} | {'GPU (ms)':>10} | {'Speedup':>10}")
        print("-" * 50)
        for r in disc_results:
            print(f"{r['n_panels']:<12} | {r['cpu_ms']:>10.2f} | {r['gpu_ms']:>10.2f} | {r['speedup']:>9.2f}x")

        print("\n--- Batched Alpha Sweep Scaling (GPU Tensor Acceleration) ---")
        print(f"{'Batch Size (B)':<15} | {'Total GPU (ms)':>15} | {'Throughput (sol/s)':>20} | {'Time/Solve':>12}")
        print("-" * 70)
        for r in sweep_results:
            print(f"{r['batch_size']:<15} | {r['total_ms']:>15.2f} | {r['throughput_solves_sec']:>20.1f} | {r['time_per_solve_us']:>9.2f} us")
        print("=" * 70 + "\n")

    return {
        "discretization": disc_results,
        "sweeps": sweep_results,
    }


def run_symmetry_benchmark(
    panel_counts: tuple[int, ...] = (10, 20, 40, 80, 160),
    solvers: tuple[str, ...] = ("horseshoe", "nonlinear"),
    backend: str = "auto",
    n_repeats: int = 3,
    verbose: bool = True,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Benchmark performance and mathematical precision with and without Y=0 symmetry plane.

    Parameters
    ----------
    panel_counts : tuple of int
        Panels per semi-span (total panels: 2*N for full mesh, N for half mesh).
    solvers : tuple of str
        Solvers to benchmark ('horseshoe', 'nonlinear', 'gpu', 'gpu_nonlinear').
    backend : str
        Execution backend ('auto', 'cpu', 'gpu').
    n_repeats : int
        Number of repeats for timing (minimum is reported).
    verbose : bool
        If True, prints formatted summary table.
    show_progress : bool
        If True, displays real-time progress bar.

    Returns
    -------
    dict
        Detailed comparison records containing runtimes, speedups, memory savings,
        and mathematical discrepancies.
    """
    from ventorum.legacy import analyze

    records = []
    total_runs = len(panel_counts) * len(solvers) * 2 * n_repeats
    pb = ProgressBar(total=total_runs, title="Symmetry Bench", unit="runs") if show_progress else None

    for solver in solvers:
        s_lower = solver.lower()
        if s_lower in ("gpu", "gpu_horseshoe"):
            solver_name = "horseshoe"
            b_end = "gpu"
        elif s_lower == "gpu_nonlinear":
            solver_name = "nonlinear"
            b_end = "gpu"
        else:
            solver_name = solver
            b_end = backend

        is_nonlinear = (solver_name == "nonlinear")
        geom = create_test_aircraft(n_surfaces=1, nonlinear=is_nonlinear)

        for n in panel_counts:
            # 1. Full mesh solve (use_symmetry=False)
            # Warm-up
            try:
                res_full = analyze(geom[0], alpha_deg=5.0, solver=solver_name, backend=b_end, n_panels=n, use_symmetry=False)
            except Exception:
                res_full = None

            t_full_list = []
            for r in range(n_repeats):
                t0 = time.perf_counter()
                res_full = analyze(geom[0], alpha_deg=5.0, solver=solver_name, backend=b_end, n_panels=n, use_symmetry=False)
                t_full_list.append((time.perf_counter() - t0) * 1000.0)
                if pb:
                    pb.update(1, status=f"{solver} N={n} Full (run {r+1})")

            best_t_full = min(t_full_list)

            # 2. Symmetric half-mesh solve (use_symmetry=True)
            # Warm-up
            try:
                res_sym = analyze(geom[0], alpha_deg=5.0, solver=solver_name, backend=b_end, n_panels=n, use_symmetry=True)
            except Exception:
                res_sym = None

            t_sym_list = []
            for r in range(n_repeats):
                t0 = time.perf_counter()
                res_sym = analyze(geom[0], alpha_deg=5.0, solver=solver_name, backend=b_end, n_panels=n, use_symmetry=True)
                t_sym_list.append((time.perf_counter() - t0) * 1000.0)
                if pb:
                    pb.update(1, status=f"{solver} N={n} Sym (run {r+1})")

            best_t_sym = min(t_sym_list)

            # Metrics
            speedup = best_t_full / best_t_sym if best_t_sym > 0 else 1.0
            time_reduction_pct = (1.0 - best_t_sym / best_t_full) * 100.0 if best_t_full > 0 else 0.0

            # Discrepancy check
            if res_full is not None and res_sym is not None:
                diff_cl = abs(res_full.totals.CL - res_sym.totals.CL)
                diff_cdi = abs(res_full.totals.CDi - res_sym.totals.CDi)
            else:
                diff_cl = 0.0
                diff_cdi = 0.0

            # Memory reduction (AIC entries: (2N)^2 vs N^2 = 4x reduction)
            aic_full_entries = (2 * n) ** 2
            aic_sym_entries = n ** 2
            mem_reduction_factor = float(aic_full_entries) / float(aic_sym_entries)

            rec = {
                "solver": solver,
                "n_panels_semi": n,
                "n_panels_total_full": 2 * n,
                "n_panels_half": n,
                "time_full_ms": best_t_full,
                "time_sym_ms": best_t_sym,
                "speedup": speedup,
                "time_reduction_pct": time_reduction_pct,
                "diff_cl": diff_cl,
                "diff_cdi": diff_cdi,
                "aic_mem_reduction": mem_reduction_factor,
            }
            records.append(rec)

    if pb:
        pb.finish("Symmetry benchmark finished")

    if verbose:
        print("\n" + "=" * 90)
        print(" Ventorum Symmetry Plane (Y=0) Acceleration & Accuracy Benchmark ")
        print("=" * 90)
        header = f"{'Solver':<12} | {'Panels (N)':<10} | {'Full (ms)':>10} | {'Sym (ms)':>10} | {'Speedup':>9} | {'Saved (%)':>10} | {'Delta CL':>10} | {'Delta CDi':>10}"
        print(header)
        print("-" * 90)
        for r in records:
            p_str = f"{r['n_panels_semi']} (tot {r['n_panels_total_full']})"
            sp_str = f"{r['speedup']:.2f}x"
            pct_str = f"{r['time_reduction_pct']:.1f}%"
            dcl_str = f"{r['diff_cl']:.2e}"
            dcdi_str = f"{r['diff_cdi']:.2e}"
            print(f"{r['solver']:<12} | {p_str:<10} | {r['time_full_ms']:>10.2f} | {r['time_sym_ms']:>10.2f} | {sp_str:>9} | {pct_str:>10} | {dcl_str:>10} | {dcdi_str:>10}")
        print("=" * 90)
        print(f" * AIC matrix size reduced by {records[0]['aic_mem_reduction']:.1f}x across all test configurations.")
        print(" * Default behavior: use_symmetry=True (automatically off when beta != 0, phi != 0, or asymmetric).\n")

    return {"records": records}


def run_mesh_convergence_benchmark(
    cases: tuple[str, ...] = ("planar_wing", "aerosonde_uav"),
    depths: tuple[str, ...] = ("fast", "standard", "deep_3x"),
    n_repeats: int = 1,
    backend: str = "cpu",
    verbose: bool = True,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Benchmark the automated mesh convergence study process itself.

    Evaluates execution wall-clock latency, sub-component profiling breakdown
    (reference solve, candidate grid evaluations, GCI/error analysis), panel throughput,
    and discretization evolution scaling across study depths (including 3x-4x higher panel counts).

    Parameters
    ----------
    cases : tuple of str
        Aircraft cases to test ('planar_wing', 'aerosonde_uav').
    depths : tuple of str
        Study depths ('fast', 'standard', 'deep_3x').
    n_repeats : int
        Number of repeats for timing (minimum is reported).
    backend : str
        Execution backend ('auto', 'cpu', 'gpu').
    verbose : bool
        If True, prints formatted summary tables and scaling analysis.
    show_progress : bool
        If True, displays progress bar in the terminal.

    Returns
    -------
    dict
        Benchmark metrics, timing records, profiling breakdown, and speedups.
    """
    from ventorum.legacy.geometry.mesh_convergence import run_mesh_convergence_study
    from ventorum.legacy.core.datatypes import Aircraft, FlightCondition, LiftingSurface, WingSection, LinearAirfoil

    # Pre-build benchmark geometries
    wing_planar = LiftingSurface(
        name="PlanarWing_AR8",
        semi_span=5.0,
        sections=[
            WingSection(y_frac=0.0, chord=1.25),
            WingSection(y_frac=1.0, chord=1.25),
        ],
        is_symmetric=True,
    )
    cond_planar = FlightCondition(V_inf=50.0, alpha=np.radians(5.0))

    af_4412 = LinearAirfoil(name="NACA 4412", a0=2 * np.pi, alpha_L0=np.radians(-4.0), Cd0=0.0075)
    af_0012 = LinearAirfoil(name="NACA 0012", a0=2 * np.pi, alpha_L0=0.0, Cd0=0.0065)
    main_w = LiftingSurface(
        name="MainWing",
        semi_span=1.45,
        sections=[
            WingSection(y_frac=0.0, chord=0.28, airfoil=af_4412, twist=np.radians(2.0)),
            WingSection(y_frac=0.6, chord=0.24, airfoil=af_4412, twist=np.radians(1.0)),
            WingSection(y_frac=1.0, chord=0.16, airfoil=af_4412, twist=0.0),
        ],
        sweep_le=np.radians(1.5),
        dihedral=np.radians(2.0),
        is_symmetric=True,
    )
    v_tail = LiftingSurface(
        name="InvertedVTail",
        semi_span=0.42,
        sections=[
            WingSection(y_frac=0.0, chord=0.15, airfoil=af_0012),
            WingSection(y_frac=1.0, chord=0.10, airfoil=af_0012),
        ],
        sweep_le=np.radians(8.0),
        dihedral=np.radians(-38.0),
        position=np.array([0.95, 0.0, 0.06]),
        is_symmetric=True,
    )
    ac_aerosonde = Aircraft(name="AerosondeUAV", surfaces=[main_w, v_tail])
    ac_aerosonde.compute_reference_values()
    cond_aero = FlightCondition(V_inf=25.0, alpha=np.radians(4.0))

    depth_configs = {
        "fast": {
            "panel_counts": (15, 30, 60),
            "ref_n_panels": 90,
            "spacing_schemes": ("auto", "cosine"),
            "sweep": None,
        },
        "standard": {
            "panel_counts": (15, 30, 50, 80),
            "ref_n_panels": 120,
            "spacing_schemes": ("auto", "half-cosine", "cosine"),
            "sweep_planar": None,
            "sweep_aero": [-2.0, 0.0, 2.0, 4.0, 6.0],
        },
        "deep_3x": {
            "panel_counts": (20, 40, 80, 120, 160, 200, 250),
            "ref_n_panels": 300,
            "spacing_schemes": ("uniform", "half-cosine", "cosine"),
            "sweep_planar": None,
            "sweep_aero": [-2.0, 0.0, 2.0, 4.0, 6.0],
        },
    }

    records: list[dict[str, Any]] = []
    total_runs = len(cases) * len(depths) * n_repeats
    pb = ProgressBar(total=total_runs, title="Study Benchmark", unit="studies") if show_progress else None

    for c_name in cases:
        if c_name == "planar_wing":
            geom = wing_planar
            cond = cond_planar
            is_multi = False
        else:
            geom = ac_aerosonde
            cond = cond_aero
            is_multi = True

        for d_name in depths:
            cfg = depth_configs[d_name]
            p_counts = cfg["panel_counts"]
            ref_n = cfg["ref_n_panels"]
            schemes = cfg["spacing_schemes"]
            if is_multi:
                swp = cfg.get("sweep_aero", None)
            else:
                swp = cfg.get("sweep_planar", None)

            study_times = []
            last_result = None

            for r in range(n_repeats):
                t0 = time.perf_counter()
                res = run_mesh_convergence_study(
                    case=geom,
                    condition=cond,
                    tolerance_pct=0.5,
                    panel_counts=p_counts,
                    spacing_schemes=schemes,
                    ref_n_panels=ref_n,
                    proportional_panels=is_multi,
                    evaluate_sweep=bool(swp is not None),
                    alpha_sweep_deg=swp,
                    backend=backend,
                    progress=False,
                )
                t1 = time.perf_counter()
                study_times.append((t1 - t0) * 1000.0)
                last_result = res
                if pb:
                    pb.update(1, status=f"{c_name} {d_name} (run {r+1}/{n_repeats})")

            best_time_ms = min(study_times)
            prof = last_result.profiling if last_result else {}

            rec = {
                "case": c_name,
                "depth": d_name,
                "n_candidates": len(last_result.points),
                "max_candidate_n": max(p.n_panels for p in last_result.points),
                "ref_n_panels": ref_n,
                "has_sweep": bool(swp is not None),
                "study_time_ms": best_time_ms,
                "t_ref_ms": prof.get("t_ref_ms", 0.0),
                "t_cands_ms": prof.get("t_cands_ms", 0.0),
                "t_analysis_ms": prof.get("t_analysis_ms", 0.0),
                "throughput_panels_sec": prof.get("throughput_panels_sec", 0.0),
                "throughput_solves_sec": prof.get("throughput_solves_sec", 0.0),
                "recommended_n": last_result.recommended_mesh.n_panels,
                "recommended_spacing": last_result.recommended_mesh.spacing,
                "minimal_n": last_result.minimal_mesh.n_panels,
                "rec_cl_err_pct": last_result.recommended_mesh.error_cl_pct,
                "rec_cdi_err_pct": last_result.recommended_mesh.error_cdi_pct,
                "rec_speedup": last_result.reference_point.solve_time_ms / max(last_result.recommended_mesh.solve_time_ms, 1e-6),
            }
            records.append(rec)

    if pb:
        pb.finish("Convergence study benchmarks finished")

    if verbose:
        w = 115
        print("\n" + "=" * w)
        print(" Ventorum Automated Mesh Convergence Process Benchmark ".center(w))
        print("=" * w)
        header = (
            f"{'Case':<14} | {'Depth':<9} | {'Max N':<6} | {'Ref N':<6} | {'Total (ms)':>10} | "
            f"{'Ref (ms)':>8} | {'Sweep (ms)':>10} | {'Throughput':>14} | {'Rec Mesh':<18} | {'Speedup':>7}"
        )
        print(header)
        print("-" * w)
        for r in records:
            tp_str = f"{r['throughput_panels_sec']:.0f} pan/s"
            rec_str = f"N={r['recommended_n']} {r['recommended_spacing']}"
            sp_str = f"{r['rec_speedup']:.1f}x"
            print(
                f"{r['case']:<14} | {r['depth']:<9} | {r['max_candidate_n']:<6d} | {r['ref_n_panels']:<6d} | "
                f"{r['study_time_ms']:>10.1f} | {r['t_ref_ms']:>8.1f} | {r['t_cands_ms']:>10.1f} | "
                f"{tp_str:>14} | {rec_str:<18} | {sp_str:>7}"
            )
        print("=" * w)
        print(" * Component Profiling: Reference solve scales as O(N_ref^2..N_ref^3); candidate evaluations scale with K.")
        print(" * Deep 3x Study: Evaluates high discretizations (N up to 250, Ref N=300) in ~1-2 seconds with extreme panel throughput.\n")

    return {"records": records}


def run_numba_comparison_benchmark(
    panel_counts: tuple[int, ...] = (20, 40, 80, 160, 320),
    solvers: tuple[str, ...] = ("horseshoe", "nonlinear"),
    n_repeats: int = 3,
    verbose: bool = True,
    use_symmetry: bool = False,
) -> dict[str, Any]:
    """Benchmark and compare runtime with Numba JIT enabled versus pure NumPy.

    Parameters
    ----------
    panel_counts : tuple of int
        Panels per semi-span to test.
    solvers : tuple of str
        Solvers to compare ('horseshoe', 'nonlinear').
    n_repeats : int
        Number of repeats for timing (minimum is reported).
    verbose : bool
        If True, prints formatted comparison tables.
    use_symmetry : bool
        Whether to enforce the Y=0 symmetry plane.

    Returns
    -------
    dict
        Dictionary containing comparative runtimes, speedup ratios, and accuracy verification.
    """
    from ventorum.legacy import analyze
    from ventorum.legacy.aero.numba_kernels import is_numba_enabled, set_numba_enabled, has_numba

    if not has_numba():
        raise RuntimeError("Numba is not installed; cannot run Numba comparison benchmark.")

    results = {s: [] for s in solvers}
    initial_numba_state = is_numba_enabled()

    try:
        for s in solvers:
            is_nonlin = (s == "nonlinear")
            geom = create_test_aircraft(n_surfaces=1, nonlinear=is_nonlin)[0]

            for n in panel_counts:
                # 1. Pure NumPy (Numba disabled)
                set_numba_enabled(False)
                analyze(geom, alpha_deg=5.0, solver=s, n_panels=n, use_symmetry=use_symmetry)
                times_np = []
                for _ in range(n_repeats):
                    t0 = time.perf_counter()
                    res_np = analyze(geom, alpha_deg=5.0, solver=s, n_panels=n, use_symmetry=use_symmetry)
                    times_np.append(time.perf_counter() - t0)
                t_np_ms = min(times_np) * 1000.0

                # 2. Numba JIT (Numba enabled)
                set_numba_enabled(True)
                analyze(geom, alpha_deg=5.0, solver=s, n_panels=n, use_symmetry=use_symmetry)
                times_nb = []
                for _ in range(n_repeats):
                    t0 = time.perf_counter()
                    res_nb = analyze(geom, alpha_deg=5.0, solver=s, n_panels=n, use_symmetry=use_symmetry)
                    times_nb.append(time.perf_counter() - t0)
                t_nb_ms = min(times_nb) * 1000.0

                speedup = t_np_ms / max(t_nb_ms, 1e-6)
                diff_cl = abs(res_nb.totals.CL - res_np.totals.CL)
                diff_cdi = abs(res_nb.totals.CDi - res_np.totals.CDi)

                results[s].append({
                    "n_panels": n,
                    "total_panels": n * 2,
                    "numpy_ms": t_np_ms,
                    "numba_ms": t_nb_ms,
                    "speedup": speedup,
                    "diff_cl": diff_cl,
                    "diff_cdi": diff_cdi,
                })
    finally:
        set_numba_enabled(initial_numba_state)

    if verbose:
        sym_str = f" [Symmetry: {use_symmetry}]" if use_symmetry is not None else ""
        print("\n" + "=" * 80)
        print(f" Ventorum Numba Acceleration vs Pure NumPy Benchmark{sym_str} ".center(80))
        print("=" * 80)
        for s in solvers:
            print(f"\n--- Solver: {s.upper()} ---")
            header = f"{'Panels (semi)':>13} | {'Total Panels':>12} | {'NumPy (ms)':>12} | {'Numba (ms)':>12} | {'Speedup':>9} | {'Max Err CL':>11}"
            print(header)
            print("-" * 80)
            for r in results[s]:
                row = f"{r['n_panels']:>13d} | {r['total_panels']:>12d} | {r['numpy_ms']:>12.2f} | {r['numba_ms']:>12.2f} | {r['speedup']:>8.1f}x | {r['diff_cl']:>11.1e}"
                print(row)
        print("=" * 80 + "\n")

    return results


def run_acceleration_comparison_benchmark(
    panel_counts: tuple[int, ...] = (20, 40, 80, 160, 320),
    solvers: tuple[str, ...] = ("horseshoe", "nonlinear"),
    n_repeats: int = 3,
    verbose: bool = True,
    use_symmetry: bool = False,
) -> dict[str, Any]:
    """Benchmark and compare runtime across Pure NumPy, Cython, Numba, and Numba+Cython.

    Evaluates:
    1. Baseline before either (Pure NumPy vectorized CPU)
    2. Cython compiled C extension
    3. Numba multi-threaded JIT
    4. Numba + Cython cooperative hybrid acceleration

    Parameters
    ----------
    panel_counts : tuple of int
        Panels per semi-span to test.
    solvers : tuple of str
        Solvers to compare ('horseshoe', 'nonlinear').
    n_repeats : int
        Number of repeats for timing (minimum is reported).
    verbose : bool
        If True, prints formatted comparison tables.
    use_symmetry : bool
        Whether to enforce the Y=0 symmetry plane.

    Returns
    -------
    dict
        Dictionary containing comparative runtimes, speedup ratios, and accuracy verification.
    """
    from ventorum.legacy import analyze
    from ventorum.legacy.aero.acceleration import (
        set_acceleration,
        get_active_acceleration,
        acceleration_context,
    )
    from ventorum.legacy.aero.cython_accel import has_cython
    from ventorum.legacy.aero.numba_kernels import has_numba

    modes = ["numpy"]
    if has_cython():
        modes.append("cython")
    if has_numba():
        modes.append("numba")
    if has_cython() and has_numba():
        modes.append("numba+cython")

    results: dict[str, list[dict[str, Any]]] = {s: [] for s in solvers}
    initial_accel = get_active_acceleration()

    try:
        for s in solvers:
            is_nonlin = (s == "nonlinear")
            geom = create_test_aircraft(n_surfaces=1, nonlinear=is_nonlin)[0]

            for n in panel_counts:
                entry: dict[str, Any] = {
                    "n_panels": n,
                    "total_panels": n * 2,
                    "times_ms": {},
                    "speedups": {},
                    "errors_cl": {},
                }

                ref_cl = None
                for mode in modes:
                    with acceleration_context(mode):
                        # Warmup
                        res_w = analyze(geom, alpha_deg=5.0, solver=s, n_panels=n, use_symmetry=use_symmetry)
                        times = []
                        for _ in range(n_repeats):
                            t0 = time.perf_counter()
                            res = analyze(geom, alpha_deg=5.0, solver=s, n_panels=n, use_symmetry=use_symmetry)
                            times.append(time.perf_counter() - t0)
                        t_ms = min(times) * 1000.0
                        entry["times_ms"][mode] = t_ms

                        if mode == "numpy":
                            ref_cl = res.totals.CL
                            entry["errors_cl"]["numpy"] = 0.0
                        else:
                            err_cl = abs(res.totals.CL - ref_cl) if ref_cl is not None else 0.0
                            entry["errors_cl"][mode] = err_cl

                # Compute speedups relative to Pure NumPy baseline
                t_base = entry["times_ms"].get("numpy", 1.0)
                for mode in modes:
                    t_m = entry["times_ms"][mode]
                    entry["speedups"][mode] = t_base / max(t_m, 1e-6)

                results[s].append(entry)
    finally:
        set_acceleration(initial_accel)

    if verbose:
        sym_str = f" [Symmetry: {use_symmetry}]" if use_symmetry else ""
        print("\n" + "=" * 90)
        print(f" Ventorum 4-Way Acceleration Benchmark: NumPy vs Cython vs Numba vs Hybrid{sym_str} ".center(90))
        print("=" * 90)

        for s in solvers:
            print(f"\n--- Solver: {s.upper()} ---")
            col_headers = [f"{'Panels':>7}", f"{'Total':>7}"]
            for m in modes:
                col_headers.append(f"{m + ' (ms)':>14}")
            for m in modes[1:]:
                col_headers.append(f"{'Spd ' + m:>12}")

            hdr_line = " | ".join(col_headers)
            print(hdr_line)
            print("-" * len(hdr_line))

            for r in results[s]:
                row_items = [f"{r['n_panels']:>7d}", f"{r['total_panels']:>7d}"]
                for m in modes:
                    row_items.append(f"{r['times_ms'][m]:>14.2f}")
                for m in modes[1:]:
                    row_items.append(f"{r['speedups'][m]:>11.1f}x")
                print(" | ".join(row_items))
        print("=" * 90 + "\n")

    return results





