# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Hardware capability scanning, machine fingerprinting, and auto-tuning package for Ventorum.
"""

from __future__ import annotations

from ventorum.legacy.hardware.detector import (
    HardwareInfo,
    scan_hardware,
    get_machine_fingerprint,
    generate_fingerprint,
)
from ventorum.legacy.hardware.config import (
    MachineConfig,
    STANDARD_DEFAULTS,
    CONFIG_FILENAME,
    load_machine_config,
    save_machine_config,
    clear_machine_config,
    is_current_machine_optimized,
    get_active_setting,
    is_autotune_disabled,
)
from ventorum.legacy.hardware.tuner import (
    tune_machine,
    benchmark_cpu_vs_gpu_single,
    benchmark_gpu_sweep_acceleration,
    benchmark_cpu_worker_scaling,
    benchmark_multi_instance_concurrency,
)

__all__ = [
    "HardwareInfo",
    "scan_hardware",
    "get_machine_fingerprint",
    "generate_fingerprint",
    "MachineConfig",
    "STANDARD_DEFAULTS",
    "CONFIG_FILENAME",
    "load_machine_config",
    "save_machine_config",
    "clear_machine_config",
    "is_current_machine_optimized",
    "get_active_setting",
    "is_autotune_disabled",
    "tune_machine",
    "benchmark_cpu_vs_gpu_single",
    "benchmark_gpu_sweep_acceleration",
    "benchmark_cpu_worker_scaling",
    "benchmark_multi_instance_concurrency",
]
