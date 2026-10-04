# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Automated hardware calibration and benchmark-driven auto-tuning engine for Ventorum.

Conducts micro-benchmarks on the active machine across hardware-dependent variables:
1. CPU vs GPU single-solve crossover discretization.
2. Batched sweep GPU crossover and tensor throughput.
3. CPU worker pool scaling for sweeps (threads vs processes, 1..cores).
4. Multi-instance concurrency allocation (concurrent cases x workers per case).

Generates and persists an optimized machine configuration bound to the local
hardware fingerprint.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np

from ventorum.legacy.core.datatypes import Aircraft, FlightCondition, SolverSettings
from ventorum.legacy.hardware.config import (
    MachineConfig,
    save_machine_config,
)
from ventorum.legacy.hardware.detector import (
    HardwareInfo,
    scan_hardware,
)
from ventorum.legacy.utils.benchmark import create_test_aircraft, _create_benchmark_study_cases
from ventorum.legacy.utils.progress import ProgressBar


def benchmark_cpu_vs_gpu_single(
    hw: HardwareInfo,
    quick: bool = False,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Benchmark single-solve execution to determine GPU vs CPU crossover panel count."""
    from ventorum.legacy.solvers.horseshoe import HorseshoeSolver

    if not hw.gpu_available:
        return {
            "gpu_tested": False,
            "crossover_panel_count": 999999,
            "notes": "No CUDA GPU detected; CPU BLAS is the sole single-point solver.",
            "data": [],
        }

    from ventorum.legacy.solvers.gpu_horseshoe import GPUHorseshoeSolver

    panel_counts = (20, 40, 80, 160) if quick else (20, 40, 80, 160, 320)
    repeats = 2 if quick else 3

    geom = create_test_aircraft(n_surfaces=1, nonlinear=False)[0]
    ac = Aircraft(name="BenchWing", surfaces=[geom])
    ac.compute_reference_values()
    cond = FlightCondition(V_inf=50.0, alpha=np.radians(5.0))

    cpu_solver = HorseshoeSolver()
    gpu_solver = GPUHorseshoeSolver()

    # Warmup
    sett_warm = SolverSettings(solver_type="horseshoe", n_panels=20)
    cpu_solver.solve(ac, cond, sett_warm)
    gpu_solver.solve(ac, cond, sett_warm)

    results = []
    crossover_n = None

    pb = (
        ProgressBar(
            total=len(panel_counts) * 2 * repeats,
            title="Tuning CPU vs GPU Single",
            unit="solves",
        )
        if show_progress
        else None
    )

    for n in panel_counts:
        sett = SolverSettings(solver_type="horseshoe", n_panels=n)

        # CPU runs
        t_cpu = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            cpu_solver.solve(ac, cond, sett)
            t_cpu.append((time.perf_counter() - t0) * 1000.0)
            if pb:
                pb.update(1, status=f"CPU N={n}")
        best_cpu = min(t_cpu)

        # GPU runs
        t_gpu = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            gpu_solver.solve(ac, cond, sett)
            t_gpu.append((time.perf_counter() - t0) * 1000.0)
            if pb:
                pb.update(1, status=f"GPU N={n}")
        best_gpu = min(t_gpu)

        sp = best_cpu / best_gpu if best_gpu > 0 else 0.0
        results.append({
            "n_panels": n,
            "cpu_ms": round(best_cpu, 3),
            "gpu_ms": round(best_gpu, 3),
            "speedup": round(sp, 2),
            "faster": "GPU" if best_gpu < best_cpu else "CPU",
        })

        if best_gpu < best_cpu and crossover_n is None:
            crossover_n = n

    if pb:
        pb.finish("CPU vs GPU Single Benchmark Complete")

    # If GPU is never faster up to 320 panels, set crossover conservatively
    if crossover_n is None:
        crossover_n = 400

    return {
        "gpu_tested": True,
        "crossover_panel_count": crossover_n,
        "data": results,
    }


def benchmark_gpu_sweep_acceleration(
    hw: HardwareInfo,
    quick: bool = False,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Benchmark GPU batched tensor kernels vs CPU sweeps across batch sizes."""
    from ventorum.legacy import analyze_sweep

    if not hw.gpu_available:
        return {
            "gpu_tested": False,
            "crossover_batch_size": 999999,
            "peak_speedup": 1.0,
            "notes": "No CUDA GPU detected.",
            "data": [],
        }

    sweep_sizes = (3, 9, 17, 33, 65) if quick else (3, 9, 17, 33, 65, 129)
    geom = create_test_aircraft(n_surfaces=1, nonlinear=False)[0]

    # Warmup
    alphas_warm = np.linspace(-4.0, 12.0, 5)
    analyze_sweep(geom, alpha_deg_range=alphas_warm, solver="horseshoe", n_panels=40, backend="gpu")

    results = []
    crossover_batch = None
    peak_sp = 1.0

    pb = (
        ProgressBar(
            total=len(sweep_sizes) * 2,
            title="Tuning GPU Sweeps",
            unit="sweeps",
        )
        if show_progress
        else None
    )

    for B in sweep_sizes:
        alphas = np.linspace(-4.0, 12.0, B)

        # CPU sweep (using 1 worker or optimal cpu worker)
        t0 = time.perf_counter()
        analyze_sweep(geom, alpha_deg_range=alphas, solver="horseshoe", n_panels=40, n_jobs=1, backend="thread")
        t_cpu_ms = (time.perf_counter() - t0) * 1000.0
        if pb:
            pb.update(1, status=f"CPU B={B}")

        # GPU batched sweep
        t0 = time.perf_counter()
        analyze_sweep(geom, alpha_deg_range=alphas, solver="horseshoe", n_panels=40, backend="gpu")
        t_gpu_ms = (time.perf_counter() - t0) * 1000.0
        if pb:
            pb.update(1, status=f"GPU B={B}")

        sp = t_cpu_ms / t_gpu_ms if t_gpu_ms > 0 else 1.0
        peak_sp = max(peak_sp, sp)
        results.append({
            "batch_size": B,
            "cpu_ms": round(t_cpu_ms, 2),
            "gpu_ms": round(t_gpu_ms, 2),
            "speedup": round(sp, 2),
            "faster": "GPU" if t_gpu_ms < t_cpu_ms else "CPU",
        })

        if t_gpu_ms < t_cpu_ms and crossover_batch is None:
            crossover_batch = B

    if pb:
        pb.finish("GPU Sweep Benchmark Complete")

    if crossover_batch is None:
        crossover_batch = 5

    return {
        "gpu_tested": True,
        "crossover_batch_size": crossover_batch,
        "peak_speedup": round(peak_sp, 2),
        "data": results,
    }


def benchmark_cpu_worker_scaling(
    hw: HardwareInfo,
    quick: bool = False,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Benchmark CPU worker scaling across worker counts and thread vs process backends."""
    from ventorum.legacy import analyze_sweep

    log_cores = hw.logical_cores
    raw_counts = [1, 2, 4, 6, 8, 12, log_cores]
    worker_counts = sorted(list({w for w in raw_counts if w <= log_cores} | {1, log_cores}))
    if quick and len(worker_counts) > 4:
        worker_counts = sorted(list({1, 2, min(4, log_cores), log_cores}))

    repeats = 2 if quick else 3
    geom = create_test_aircraft(n_surfaces=1, nonlinear=False)[0]
    alphas = np.linspace(-4.0, 12.0, 33)

    # 1. ThreadPoolExecutor scaling
    thread_times = {}
    pb = (
        ProgressBar(
            total=len(worker_counts) * repeats,
            title="Tuning CPU Workers",
            unit="runs",
        )
        if show_progress
        else None
    )

    for w in worker_counts:
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            analyze_sweep(
                geom,
                alpha_deg_range=alphas,
                solver="horseshoe",
                n_panels=40,
                n_jobs=w,
                backend="thread",
            )
            times.append((time.perf_counter() - t0) * 1000.0)
            if pb:
                pb.update(1, status=f"Threads w={w}")
        thread_times[w] = round(min(times), 2)

    if pb:
        pb.finish("CPU Worker Pool Tuning Complete")

    # Determine best thread worker count
    best_workers = min(thread_times, key=thread_times.get)
    best_thread_time = thread_times[best_workers]

    # Quick test comparing thread vs process backend at best worker count
    process_time = None
    if best_workers > 1 and not quick:
        try:
            t0 = time.perf_counter()
            analyze_sweep(
                geom,
                alpha_deg_range=alphas,
                solver="horseshoe",
                n_panels=40,
                n_jobs=best_workers,
                backend="process",
            )
            process_time = round((time.perf_counter() - t0) * 1000.0, 2)
        except Exception:
            process_time = None

    best_backend = "thread"
    if process_time is not None and process_time < best_thread_time:
        best_backend = "process"

    return {
        "optimal_workers": best_workers,
        "optimal_backend": best_backend,
        "thread_scaling_ms": thread_times,
        "best_thread_ms": best_thread_time,
        "process_ms": process_time,
    }


def benchmark_multi_instance_concurrency(
    hw: HardwareInfo,
    quick: bool = False,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Benchmark multi-instance concurrency allocations (concurrent cases x internal workers)."""
    from ventorum.legacy.instance import run_parallel_instances

    cpu_max = hw.logical_cores
    n_cases = 8
    alphas = np.linspace(-2.0, 10.0, 15)

    test_configs = [
        (1, 1),
        (2, 1),
        (min(4, cpu_max), 1),
        (min(cpu_max, 8), 1),
    ]
    if cpu_max >= 4 and not quick:
        test_configs.append((2, 2))
        if cpu_max >= 8:
            test_configs.append((4, 2))

    # Deduplicate configs
    seen = set()
    filtered_configs = []
    for c in test_configs:
        if c not in seen:
            seen.add(c)
            filtered_configs.append(c)

    results = []
    best_time = float("inf")
    best_cfg = (1, 1)

    pb = (
        ProgressBar(
            total=len(filtered_configs),
            title="Tuning Multi-Instance",
            unit="configs",
        )
        if show_progress
        else None
    )

    for inst_c, w_per_inst in filtered_configs:
        cases = _create_benchmark_study_cases(n_cases, alphas, n_panels=40)
        for c in cases:
            c.set_workers(w_per_inst)

        t0 = time.perf_counter()
        run_parallel_instances(
            cases,
            max_concurrent_instances=inst_c,
            instance_backend="thread",
            show_progress=False,
        )
        t_ms = (time.perf_counter() - t0) * 1000.0
        if pb:
            pb.update(1, status=f"inst={inst_c} w={w_per_inst}")

        throughput = n_cases / (t_ms / 1000.0) if t_ms > 0 else 0.0
        rec = {
            "instances": inst_c,
            "workers_per_instance": w_per_inst,
            "active_threads": inst_c * w_per_inst,
            "time_ms": round(t_ms, 2),
            "cases_per_sec": round(throughput, 2),
        }
        results.append(rec)

        if t_ms < best_time:
            best_time = t_ms
            best_cfg = (inst_c, w_per_inst)

    if pb:
        pb.finish("Multi-Instance Tuning Complete")

    return {
        "optimal_instances": best_cfg[0],
        "optimal_workers_per_instance": best_cfg[1],
        "optimal_backend": "thread",
        "data": results,
    }


def tune_machine(
    quick: bool = False,
    verbose: bool = True,
    save: bool = True,
    show_progress: bool = True,
) -> MachineConfig:
    """Execute complete machine scanning, benchmarking, and optimal configuration generation.

    Parameters
    ----------
    quick : bool
        If True, runs a streamlined calibration pass (~3-5 seconds).
        If False, executes full high-precision optimization matrix (~10-15 seconds).
    verbose : bool
        If True, prints formatted summary reports to the terminal.
    save : bool
        If True (default), automatically saves configuration to `.ventorum_machine_config.json`.
    show_progress : bool
        If True, displays live progress bars.

    Returns
    -------
    MachineConfig
        Validated configuration calibrated specifically for this machine.
    """
    hw = scan_hardware()

    if verbose:
        print("\n" + "=" * 80)
        print("        Ventorum HARDWARE SCAN & MACHINE-SPECIFIC AUTO-TUNER")
        print("=" * 80)
        print(f"Machine Name : {hw.node_name}")
        print(f"OS / Kernel  : {hw.system} {hw.release} ({hw.machine})")
        print(f"Processor    : {hw.cpu_processor}")
        print(f"CPU Topology : {hw.physical_cores} Physical Cores / {hw.logical_cores} Logical Cores")
        print(f"System RAM   : {hw.total_ram_gb:.2f} GB")
        if hw.gpu_available:
            print(f"Accelerator  : {hw.gpu_name} ({hw.gpu_vram_gb:.1f} GB VRAM, CUDA {hw.cuda_version})")
        else:
            print("Accelerator  : No CUDA GPU Detected (CPU Execution Only)")
        print(f"Fingerprint  : {hw.fingerprint}")
        print("=" * 80 + "\n")

    # 1. CPU vs GPU Single-Solve Crossover
    if verbose:
        print("[1/4] Calibrating Single-Solve CPU vs GPU Crossover...")
    res_single = benchmark_cpu_vs_gpu_single(hw, quick=quick, show_progress=show_progress)

    # 2. GPU Sweep Acceleration
    if verbose:
        print("\n[2/4] Calibrating Batched Sweep Tensor Acceleration...")
    res_sweep_gpu = benchmark_gpu_sweep_acceleration(hw, quick=quick, show_progress=show_progress)

    # 3. CPU Worker Scaling
    if verbose:
        print("\n[3/4] Calibrating CPU Worker Scaling for Sweeps...")
    res_cpu_workers = benchmark_cpu_worker_scaling(hw, quick=quick, show_progress=show_progress)

    # 4. Multi-Instance Concurrency
    if verbose:
        print("\n[4/4] Calibrating Multi-Instance Concurrency Matrix...")
    res_multi_inst = benchmark_multi_instance_concurrency(hw, quick=quick, show_progress=show_progress)

    # Synthesize optimal settings
    opt_backend = "auto"
    if hw.gpu_available:
        opt_backend = "auto"
    else:
        opt_backend = "cpu"

    gpu_chunk = 10000
    if hw.gpu_vram_gb:
        if hw.gpu_vram_gb >= 8.0:
            gpu_chunk = 50000
        elif hw.gpu_vram_gb >= 4.0:
            gpu_chunk = 20000

    optimized_settings = {
        "default_hardware_backend": opt_backend,
        "gpu_available": hw.gpu_available,
        "gpu_name": hw.gpu_name,
        "gpu_single_crossover_panels": res_single["crossover_panel_count"],
        "gpu_sweep_crossover_batch": res_sweep_gpu["crossover_batch_size"],
        "optimal_sweep_workers": res_cpu_workers["optimal_workers"],
        "optimal_sweep_backend": res_cpu_workers["optimal_backend"],
        "optimal_instance_concurrency": res_multi_inst["optimal_instances"],
        "optimal_workers_per_instance": res_multi_inst["optimal_workers_per_instance"],
        "optimal_instance_backend": res_multi_inst["optimal_backend"],
        "use_symmetry_default": True,
        "gpu_max_chunk_size": gpu_chunk,
    }

    benchmark_scores = {
        "cpu_vs_gpu_single": res_single,
        "gpu_sweep": res_sweep_gpu,
        "cpu_worker_scaling": res_cpu_workers,
        "multi_instance": res_multi_inst,
    }

    config = MachineConfig(
        fingerprint=hw.fingerprint,
        machine_name=hw.node_name,
        created_at=datetime.now(timezone.utc).isoformat(),
        ventorum_version="0.1.0",
        hardware_info=hw.to_dict(),
        settings=optimized_settings,
        benchmark_scores=benchmark_scores,
        is_active=True,
        status="matched",
    )

    if save:
        saved_path = save_machine_config(config)
        if verbose:
            print(f"\n>> Machine configuration saved to: {saved_path}")

    if verbose:
        print("\n" + "=" * 80)
        print("          OPTIMIZED MACHINE CONFIGURATION REPORT")
        print("=" * 80)
        print(f"Hardware Fingerprint          : {hw.fingerprint}")
        print(f"Default Hardware Backend      : {optimized_settings['default_hardware_backend'].upper()}")
        if hw.gpu_available:
            print(f"GPU Single-Solve Crossover    : N >= {optimized_settings['gpu_single_crossover_panels']} panels")
            print(f"GPU Sweep Batched Crossover   : B >= {optimized_settings['gpu_sweep_crossover_batch']} alphas (Peak Speedup: {res_sweep_gpu['peak_speedup']}x)")
        print(f"Optimal CPU Sweep Workers     : {optimized_settings['optimal_sweep_workers']} ({optimized_settings['optimal_sweep_backend']})")
        print(f"Optimal Multi-Instance Setup  : {optimized_settings['optimal_instance_concurrency']} concurrent cases x {optimized_settings['optimal_workers_per_instance']} workers/case")
        print(f"Symmetry Plane Default        : {optimized_settings['use_symmetry_default']} (Y=0 reflection)")
        print("=" * 80)
        print("Status: Active. This machine profile will be automatically loaded as default.")
        print("If copied to another machine, Ventorum will detect the change and safely revert.\n")

    return config
