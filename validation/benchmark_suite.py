# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Benchmark suite for Ventorum.

Measures the speed of Ventorum solvers across panel counts, sweeps,
ground effect, threading configurations, and symmetry usage.
Also verifies zero inter-instance interference in parallel execution.

Usage:
    python validation/benchmark_suite.py [--quick] [--output FILE]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

import ventorum as vt
from ventorum.instance import Ventorum, run_parallel_instances
from ventorum.utils.parallel import cpu_cores


# Suppress runtime warnings from the solvers
warnings.filterwarnings("ignore", category=RuntimeWarning)


def _median_time(fn, repeats: int) -> float:
    """Return the median wall time (s) of fn, with one warm-up call."""
    fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def _create_geometry(solver: str, n_panels: int) -> vt.LiftingSurface:
    """Create a test geometry appropriate for the solver."""
    is_nonlinear = solver == "nonlinear"
    if is_nonlinear:
        alpha_data = np.linspace(np.radians(-10), np.radians(20), 30)
        cl_data = 2 * np.pi * alpha_data
        cd_data = 0.01 + 0.05 * alpha_data**2
        airfoil = vt.TabulatedAirfoil(name="BenchAirfoil", alpha=alpha_data, Cl_data=cl_data, Cd_data=cd_data)
    else:
        airfoil = vt.LinearAirfoil()

    return vt.LiftingSurface(
        name="BenchWing",
        semi_span=5.0,
        sections=[
            vt.WingSection(y_frac=0.0, chord=2.0, airfoil=airfoil),
            vt.WingSection(y_frac=1.0, chord=1.0, airfoil=airfoil),
        ],
    )


def _create_sweep_geometry(solver: str) -> vt.LiftingSurface:
    """Create geometry for sweep tests."""
    return _create_geometry(solver, 40)


def _create_ground_effect_geometry() -> vt.LiftingSurface:
    """Create geometry for ground effect tests."""
    return vt.LiftingSurface(
        name="GEWing",
        semi_span=5.0,
        sections=[
            vt.WingSection(y_frac=0.0, chord=2.0),
            vt.WingSection(y_frac=1.0, chord=1.0),
        ],
    )


def benchmark_panel_scaling(
    panel_counts: tuple[int, ...] = (10, 20, 40, 80, 160),
    solvers: tuple[str, ...] = ("vlm", "linear", "nonlinear", "fourier"),
    n_repeats: int = 3,
    use_symmetry: bool | None = None,
) -> dict[str, Any]:
    """Benchmark single-point solve time vs panel count for each solver."""
    results = {s: [] for s in solvers}

    for n in panel_counts:
        for solver in solvers:
            geom = _create_geometry(solver, n)

            # Warm-up
            try:
                vt.analyze(geom, alpha_deg=5.0, solver=solver, n_panels=n, use_symmetry=use_symmetry)
            except Exception:
                results[solver].append(float("nan"))
                continue

            t_ms = _median_time(
                lambda g=geom, s=solver, p=n, sym=use_symmetry:
                vt.analyze(g, alpha_deg=5.0, solver=s, n_panels=p, use_symmetry=sym),
                n_repeats
            ) * 1000.0
            results[solver].append(t_ms)

    return {"panel_counts": list(panel_counts), "times_ms": results, "use_symmetry": use_symmetry}


def benchmark_alpha_sweep(
    alpha_range: tuple[float, float, int] = (-4.0, 12.0, 17),
    solvers: tuple[str, ...] = ("vlm", "linear", "nonlinear", "fourier"),
    n_panels: int = 40,
    n_repeats: int = 3,
    use_symmetry: bool | None = None,
) -> dict[str, Any]:
    """Benchmark alpha sweep time for each solver."""
    alpha_array = np.linspace(alpha_range[0], alpha_range[1], alpha_range[2])
    results = {s: 0.0 for s in solvers}

    for solver in solvers:
        geom = _create_sweep_geometry(solver)

        # Warm-up
        try:
            vt.analyze_sweep(
                geom, alpha_deg_range=np.array([0.0]),
                solver=solver, n_panels=n_panels, use_symmetry=use_symmetry
            )
        except Exception:
            results[solver] = float("nan")
            continue

        t_ms = _median_time(
            lambda g=geom, a=alpha_array, s=solver, p=n_panels, sym=use_symmetry:
            vt.analyze_sweep(g, alpha_deg_range=a, solver=s, n_panels=p, use_symmetry=sym),
            n_repeats
        ) * 1000.0
        results[solver] = t_ms

    return {
        "alpha_points": alpha_range[2],
        "n_panels": n_panels,
        "times_ms": results,
        "use_symmetry": use_symmetry,
    }


def benchmark_ground_effect(
    panel_counts: tuple[int, ...] = (10, 20, 40, 80),
    n_repeats: int = 3,
) -> dict[str, Any]:
    """Benchmark ground effect overhead vs free air."""
    results = {"free_air": [], "ground_effect": []}

    for n in panel_counts:
        geom = _create_ground_effect_geometry()

        # Free air
        t_free = _median_time(
            lambda g=geom, p=n: vt.analyze(g, alpha_deg=5.0, solver="vlm", n_panels=p),
            n_repeats
        ) * 1000.0
        results["free_air"].append(t_free)

        # Ground effect
        t_ge = _median_time(
            lambda g=geom, p=n: vt.analyze_ground_effect(
                g, h=0.2, alpha_deg=5.0, n_panels=p, compute_strike_limit=False
            ),
            n_repeats
        ) * 1000.0
        results["ground_effect"].append(t_ge)

    return {"panel_counts": list(panel_counts), "times_ms": results}


def benchmark_multithread(
    workers_list: tuple[int, ...] | None = None,
    alpha_range: tuple[float, float, int] = (-4.0, 12.0, 33),
    n_panels: int = 40,
    solver: str = "vlm",
    backend: str = "thread",
    n_repeats: int = 3,
) -> dict[str, Any]:
    """Benchmark multithreaded scaling across worker counts."""
    cpu_max = cpu_cores()
    if workers_list is None:
        raw = [1, 2, 4, 8, cpu_max]
        workers_list = tuple(sorted({w for w in raw if w <= cpu_max} | {1, cpu_max}))

    alpha_array = np.linspace(alpha_range[0], alpha_range[1], alpha_range[2])
    geom = _create_sweep_geometry(solver)

    # Warm-up
    try:
        vt.analyze_sweep(geom, alpha_deg_range=np.array([0.0, 1.0]),
                          solver=solver, n_panels=n_panels, n_jobs=1)
    except Exception:
        pass

    times_ms = []
    for w in workers_list:
        run_times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            vt.analyze_sweep(
                geom, alpha_deg_range=alpha_array, solver=solver,
                n_panels=n_panels, n_jobs=w, backend=backend
            )
            run_times.append(time.perf_counter() - t0)
        times_ms.append(min(run_times) * 1000.0)

    t_serial = times_ms[0]
    speedups = [t_serial / t if t > 0 else 1.0 for t in times_ms]
    efficiencies = [(sp / w) * 100.0 for sp, w in zip(speedups, workers_list)]
    time_per_alpha = [t / alpha_range[2] for t in times_ms]

    return {
        "workers": list(workers_list),
        "cpu_max": cpu_max,
        "alpha_points": alpha_range[2],
        "n_panels": n_panels,
        "solver": solver,
        "times_ms": times_ms,
        "time_per_alpha_ms": time_per_alpha,
        "speedup": speedups,
        "efficiency": efficiencies,
    }


def benchmark_symmetry(
    panel_counts: tuple[int, ...] = (10, 20, 40, 80, 160),
    solvers: tuple[str, ...] = ("vlm", "linear", "nonlinear"),
    n_repeats: int = 3,
) -> dict[str, Any]:
    """Benchmark with and without Y=0 symmetry plane."""
    records = []

    for solver in solvers:
        for n in panel_counts:
            geom = _create_geometry(solver, n)

            # Full mesh
            try:
                res_full = vt.analyze(geom, alpha_deg=5.0, solver=solver, n_panels=n, use_symmetry=False)
            except Exception:
                res_full = None

            t_full = _median_time(
                lambda g=geom, s=solver, p=n: vt.analyze(g, alpha_deg=5.0, solver=s, n_panels=p, use_symmetry=False),
                n_repeats
            ) * 1000.0

            # Symmetric half-mesh
            try:
                res_sym = vt.analyze(geom, alpha_deg=5.0, solver=solver, n_panels=n, use_symmetry=True)
            except Exception:
                res_sym = None

            t_sym = _median_time(
                lambda g=geom, s=solver, p=n: vt.analyze(g, alpha_deg=5.0, solver=s, n_panels=p, use_symmetry=True),
                n_repeats
            ) * 1000.0

            speedup = t_full / t_sym if t_sym > 0 else 1.0
            time_reduction = (1.0 - t_sym / t_full) * 100.0 if t_full > 0 else 0.0

            if res_full is not None and res_sym is not None:
                diff_cl = abs(res_full.totals.CL - res_sym.totals.CL)
                diff_cdi = abs(res_full.totals.CDi - res_sym.totals.CDi)
            else:
                diff_cl = 0.0
                diff_cdi = 0.0

            aic_full = (2 * n) ** 2
            aic_sym = n ** 2
            mem_reduction = float(aic_full) / float(aic_sym)

            records.append({
                "solver": solver,
                "n_panels_semi": n,
                "n_panels_total_full": 2 * n,
                "n_panels_half": n,
                "time_full_ms": t_full,
                "time_sym_ms": t_sym,
                "speedup": speedup,
                "time_reduction_pct": time_reduction,
                "diff_cl": diff_cl,
                "diff_cdi": diff_cdi,
                "aic_mem_reduction": mem_reduction,
            })

    return {"records": records}


def benchmark_multi_instance(
    n_cases: int = 8,
    alphas_per_case: int = 15,
    n_panels: int = 40,
    test_configs: list[tuple[int, int]] | None = None,
    n_repeats: int = 2,
) -> dict[str, Any]:
    """Benchmark parallel execution across multiple independent instances."""
    cpu_max = cpu_cores()
    alphas = np.linspace(-2.0, 10.0, alphas_per_case)

    def _create_cases():
        cases_defs = [
            ("Sailplane_HighAR", 15.0, 1.1, 0.5, 0.0, 2.0, -1.5, 35.0),
            ("Transport_Swept", 12.0, 2.5, 0.8, 25.0, 4.0, -2.5, 75.0),
            ("Cropped_Delta", 6.0, 3.2, 0.6, 42.0, 0.0, 0.0, 90.0),
            ("LowAR_Dihedral", 7.0, 1.8, 1.2, 5.0, 8.0, -0.5, 45.0),
            ("Agile_ForwardSwept", 8.5, 2.0, 0.9, -15.0, 1.0, 1.0, 60.0),
            ("HighLift_Tapered", 14.0, 2.8, 1.4, 8.0, 3.0, -3.0, 50.0),
            ("Commuter_Straight", 11.0, 1.9, 1.1, 2.0, 2.5, -1.0, 65.0),
            ("Arrow_Swept", 7.5, 3.0, 0.5, 50.0, -2.0, 0.0, 100.0),
        ]
        instances = []
        for i in range(n_cases):
            def_idx = i % len(cases_defs)
            name_base, b, cr, ct, sweep, dih, twist, vinf = cases_defs[def_idx]
            case_name = f"{name_base}_{i+1}" if i >= len(cases_defs) else name_base

            sections = [
                vt.WingSection(y_frac=0.0, chord=cr, twist=0.0),
                vt.WingSection(y_frac=1.0, chord=ct, twist=float(np.radians(twist))),
            ]
            surf = vt.LiftingSurface(
                name=f"{case_name}_Wing",
                semi_span=b / 2.0,
                sweep_le=float(np.radians(sweep)),
                dihedral=float(np.radians(dih)),
                sections=sections,
            )
            cond = vt.FlightCondition(V_inf=vinf, alpha=0.0)
            sett = vt.SolverSettings(solver_type="vlm", n_panels=n_panels)

            inst = Ventorum(
                name=case_name,
                geometry=surf,
                condition=cond,
                settings=sett,
                alpha_sweep_deg=alphas,
                n_workers=1,
                backend="thread",
            )
            instances.append(inst)
        return instances

    # Step 1: Serial baselines
    serial_results_map = {}
    baseline_instances = _create_cases()
    for inst in baseline_instances:
        inst.n_workers = 1
        _res = inst.run(progress=False)
        serial_results_map[inst.name] = [
            (r.totals.CL, r.totals.CDi, r.totals.e, np.copy(r.spanwise[0].gamma))
            for r in inst.sweep_results
        ]

    # Default configs
    if test_configs is None:
        raw_configs = [
            (1, 1), (1, 4), (2, 1), (2, 2), (4, 1), (4, 2), (4, 4), (8, 1), (8, 2),
        ]
        test_configs = []
        for inst_c, w_per_inst in raw_configs:
            if inst_c <= n_cases and (inst_c * w_per_inst) <= max(16, cpu_max * 2):
                test_configs.append((inst_c, w_per_inst))

    total_solves = n_cases * alphas_per_case
    results_records = []
    baseline_time_s = None

    for cfg_idx, (inst_c, w_per_inst) in enumerate(test_configs):
        active_workers = inst_c * w_per_inst
        run_times = []
        max_discrepancy = 0.0

        for _ in range(n_repeats):
            cases = _create_cases()
            for c in cases:
                c.n_workers = w_per_inst

            t0 = time.perf_counter()
            run_parallel_instances(
                cases, max_concurrent_instances=inst_c,
                instance_backend="thread", show_progress=False
            )
            t1 = time.perf_counter()
            run_times.append(t1 - t0)

            # Verify non-interference
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
        if cfg_idx == 0:
            baseline_time_s = best_time_s

        speedup = baseline_time_s / best_time_s if best_time_s > 0 else 1.0
        solves_sec = total_solves / best_time_s if best_time_s > 0 else 0.0
        denom_workers = min(active_workers, cpu_max)
        efficiency = (speedup / max(1, denom_workers)) * 100.0
        zero_interference = max_discrepancy < 1e-12  # zero interference means agreement to round-off

        results_records.append({
            "concurrent_instances": inst_c,
            "workers_per_instance": w_per_inst,
            "total_active_workers": active_workers,
            "best_time_s": best_time_s,
            "speedup": speedup,
            "efficiency": efficiency,
            "solves_per_sec": solves_sec,
            "max_discrepancy": max_discrepancy,
            "zero_interference_verified": zero_interference,
        })

    return {
        "n_cases": n_cases,
        "alphas_per_case": alphas_per_case,
        "total_solves": total_solves,
        "n_panels": n_panels,
        "cpu_max": cpu_max,
        "records": results_records,
    }


def _print_panel_scaling(res: dict[str, Any]) -> None:
    solvers = list(res["times_ms"].keys())
    panel_counts = res["panel_counts"]
    sym = res["use_symmetry"]
    sym_str = f" [Symmetry: {sym}]" if sym is not None else ""
    print("\n" + "=" * 65)
    print(f" Panel Scaling Benchmark{sym_str}")
    print("=" * 65)
    header = f"{'n_panels':>10} | " + " | ".join(f"{s.capitalize() + ' (ms)':>14}" for s in solvers)
    print(header)
    print("-" * 65)
    for i, n in enumerate(panel_counts):
        row = f"{n:>10} | " + " | ".join(f"{res['times_ms'][s][i]:>14.2f}" for s in solvers)
        print(row)
    print("=" * 65 + "\n")


def _print_sweep_benchmark(res: dict[str, Any]) -> None:
    solvers = list(res["times_ms"].keys())
    print("\n" + "=" * 65)
    print(f" Alpha Sweep Benchmark ({res['alpha_points']} points, {res['n_panels']} panels)")
    print("=" * 65)
    print(f"{'Solver':>15} | {'Total Time (ms)':>15} | {'Time per alpha (ms)':>19}")
    print("-" * 65)
    for solver in solvers:
        t = res["times_ms"][solver]
        tpa = t / res["alpha_points"]
        print(f"{solver:>15} | {t:>15.2f} | {tpa:>19.2f}")
    print("=" * 65 + "\n")


def _print_ground_effect(res: dict[str, Any]) -> None:
    panel_counts = res["panel_counts"]
    print("\n" + "=" * 65)
    print(" Ground Effect Overhead Benchmark")
    print("=" * 65)
    print(f"{'n_panels':>10} | {'Free Air (ms)':>15} | {'Ground Effect (ms)':>19} | {'Overhead':>10}")
    print("-" * 65)
    for i, n in enumerate(panel_counts):
        free = res["times_ms"]["free_air"][i]
        ge = res["times_ms"]["ground_effect"][i]
        mult = ge / free if free > 0 else 0
        print(f"{n:>10} | {free:>15.2f} | {ge:>19.2f} | {mult:>9.2f}x")
    print("=" * 65 + "\n")


def _print_multithread(res: dict[str, Any]) -> None:
    print("\n" + "=" * 88)
    print(f" Multithread Scaling Benchmark ({res['alpha_points']} alphas, N={res['n_panels']}, CPUs={res['cpu_max']})")
    print("=" * 88)
    print(f"{'Workers':>8} | {'Total (ms)':>12} | {'Time/Alpha (ms)':>16} | {'Speedup':>9} | {'Eff (%)':>8} | {'Note':<15}")
    print("-" * 88)
    for w, t_tot, t_alpha, sp, eff in zip(
        res["workers"], res["times_ms"], res["time_per_alpha_ms"],
        res["speedup"], res["efficiency"]
    ):
        if w == 1:
            note = "Baseline"
        elif eff >= 80:
            note = "Excellent"
        elif eff >= 60:
            note = "Good"
        elif eff >= 40:
            note = "Moderate"
        else:
            note = "Sub-linear"
        if w == res["cpu_max"] and w > 1:
            note += " [Full CPU]"
        print(f"{w:>8} | {t_tot:>12.2f} | {t_alpha:>16.2f} | {sp:>8.2f}x | {eff:>7.1f}% | {note:<15}")
    print("=" * 88 + "\n")


def _print_symmetry(res: dict[str, Any]) -> None:
    records = res["records"]
    print("\n" + "=" * 90)
    print(" Symmetry Plane (Y=0) Acceleration & Accuracy Benchmark")
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
    if records:
        print(f" * AIC matrix size reduced by {records[0]['aic_mem_reduction']:.1f}x across all configurations.")
    print(" * Default behavior: use_symmetry=True (automatically off when beta != 0, phi != 0, or asymmetric).\n")


def _print_multi_instance(res: dict[str, Any]) -> None:
    records = res["records"]
    print("\n" + "=" * 98)
    print(f" Multi-Instance Benchmark ({res['n_cases']} Cases, {res['total_solves']} Total Solves, CPUs={res['cpu_max']})")
    print("=" * 98)
    header = f"{'Config (Inst x Workers)':<25} | {'Active W':>8} | {'Time (s)':>9} | {'Throughput':>12} | {'Speedup':>8} | {'Eff (%)':>8} | {'Max Diff':>10} | {'Interference':<10}"
    print(header)
    print("-" * 98)
    for r in records:
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


def _write_markdown_report(
    panel_res: dict[str, Any],
    sweep_res: dict[str, Any],
    ge_res: dict[str, Any],
    mt_res: dict[str, Any],
    sym_res: dict[str, Any],
    mi_res: dict[str, Any],
    output_path: Path,
) -> None:
    """Write benchmark results as Markdown tables."""
    lines = [
        "# Ventorum Benchmark Report",
        "",
        f"Generated by `validation/benchmark_suite.py` on {time.strftime('%Y-%m-%d %H:%M:%S')}.",
        f"Machine: {os.cpu_count()} CPU cores.",
        "",
    ]

    # Panel scaling
    lines.append("## Panel Scaling (Single Point)")
    lines.append("")
    solvers = list(panel_res["times_ms"].keys())
    header = ["n_panels"] + [s.capitalize() + " (ms)" for s in solvers]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for i, n in enumerate(panel_res["panel_counts"]):
        row = [str(n)] + [f"{panel_res['times_ms'][s][i]:.2f}" for s in solvers]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Alpha sweep
    lines.append("## Alpha Sweep")
    lines.append("")
    header = ["Solver", "Total Time (ms)", "Time per Alpha (ms)"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for solver in sweep_res["times_ms"]:
        t = sweep_res["times_ms"][solver]
        tpa = t / sweep_res["alpha_points"]
        lines.append(f"| {solver} | {t:.2f} | {tpa:.2f} |")
    lines.append("")

    # Ground effect
    lines.append("## Ground Effect Overhead")
    lines.append("")
    header = ["n_panels", "Free Air (ms)", "Ground Effect (ms)", "Overhead"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for i, n in enumerate(ge_res["panel_counts"]):
        free = ge_res["times_ms"]["free_air"][i]
        ge = ge_res["times_ms"]["ground_effect"][i]
        mult = ge / free if free > 0 else 0
        lines.append(f"| {n} | {free:.2f} | {ge:.2f} | {mult:.2f}x |")
    lines.append("")

    # Multithread
    lines.append("## Multithread Scaling")
    lines.append("")
    header = ["Workers", "Total Time (ms)", "Time/Alpha (ms)", "Speedup", "Efficiency (%)", "Note"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for w, t_tot, t_alpha, sp, eff in zip(
        mt_res["workers"], mt_res["times_ms"], mt_res["time_per_alpha_ms"],
        mt_res["speedup"], mt_res["efficiency"]
    ):
        if w == 1:
            note = "Baseline"
        elif eff >= 80:
            note = "Excellent"
        elif eff >= 60:
            note = "Good"
        elif eff >= 40:
            note = "Moderate"
        else:
            note = "Sub-linear"
        if w == mt_res["cpu_max"] and w > 1:
            note += " [Full CPU]"
        lines.append(f"| {w} | {t_tot:.2f} | {t_alpha:.2f} | {sp:.2f}x | {eff:.1f}% | {note} |")
    lines.append("")

    # Symmetry
    lines.append("## Symmetry Plane Acceleration")
    lines.append("")
    header = ["Solver", "Panels (N)", "Full (ms)", "Sym (ms)", "Speedup", "Saved (%)", "Delta CL", "Delta CDi"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in sym_res["records"]:
        p_str = f"{r['n_panels_semi']} (tot {r['n_panels_total_full']})"
        lines.append(f"| {r['solver']} | {p_str} | {r['time_full_ms']:.2f} | {r['time_sym_ms']:.2f} | "
                     f"{r['speedup']:.2f}x | {r['time_reduction_pct']:.1f}% | {r['diff_cl']:.2e} | {r['diff_cdi']:.2e} |")
    lines.append("")

    # Multi-instance
    lines.append("## Multi-Instance Parallel Execution")
    lines.append("")
    header = ["Config (Inst x Workers)", "Active W", "Time (s)", "Throughput", "Speedup", "Eff (%)", "Max Diff", "Interference"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in mi_res["records"]:
        cfg_label = f"{r['concurrent_instances']} inst x {r['workers_per_instance']} workers"
        thru = f"{r['solves_per_sec']:.1f} sol/s"
        diff_str = f"{r['max_discrepancy']:.1e}"
        interf = "ZERO (PASS)" if r["zero_interference_verified"] else "FAIL"
        lines.append(f"| {cfg_label} | {r['total_active_workers']} | {r['best_time_s']:.3f} | "
                     f"{thru} | {r['speedup']:.2f}x | {r['efficiency']:.1f}% | {diff_str} | {interf} |")
    lines.append("")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_quick_benchmark() -> dict[str, Any]:
    """Run a quick benchmark with reduced parameters."""
    print("Running quick benchmark...")
    panel = benchmark_panel_scaling(panel_counts=(10, 40, 80), solvers=("vlm", "linear"), n_repeats=1)
    sweep = benchmark_alpha_sweep(alpha_range=(-4.0, 8.0, 9), solvers=("vlm", "linear"), n_repeats=1)
    ge = benchmark_ground_effect(panel_counts=(10, 40), n_repeats=1)
    mt = benchmark_multithread(workers_list=(1, 2, 4), n_repeats=1)
    sym = benchmark_symmetry(panel_counts=(10, 40), solvers=("vlm", "linear"), n_repeats=1)
    mi = benchmark_multi_instance(n_cases=4, alphas_per_case=5, test_configs=[(1, 1), (2, 1), (4, 1)], n_repeats=1)

    _print_panel_scaling(panel)
    _print_sweep_benchmark(sweep)
    _print_ground_effect(ge)
    _print_multithread(mt)
    _print_symmetry(sym)
    _print_multi_instance(mi)

    return {
        "panel": panel,
        "sweep": sweep,
        "ground_effect": ge,
        "multithread": mt,
        "symmetry": sym,
        "multi_instance": mi,
    }


def run_full_benchmark() -> dict[str, Any]:
    """Run the full benchmark suite."""
    print("Running full benchmark...")
    panel = benchmark_panel_scaling()
    sweep = benchmark_alpha_sweep()
    ge = benchmark_ground_effect()
    mt = benchmark_multithread()
    sym = benchmark_symmetry()
    mi = benchmark_multi_instance()

    _print_panel_scaling(panel)
    _print_sweep_benchmark(sweep)
    _print_ground_effect(ge)
    _print_multithread(mt)
    _print_symmetry(sym)
    _print_multi_instance(mi)

    return {
        "panel": panel,
        "sweep": sweep,
        "ground_effect": ge,
        "multithread": mt,
        "symmetry": sym,
        "multi_instance": mi,
    }


def main():
    parser = argparse.ArgumentParser(description="Ventorum Benchmark Suite")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark (reduced parameters)")
    parser.add_argument("--output", type=str, help="Write Markdown report to file")
    args = parser.parse_args()

    if args.quick:
        results = run_quick_benchmark()
    else:
        results = run_full_benchmark()

    if args.output:
        _write_markdown_report(
            results["panel"], results["sweep"], results["ground_effect"],
            results["multithread"], results["symmetry"], results["multi_instance"],
            Path(args.output)
        )
        print(f"Report written to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
