# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Configuration management and machine-fingerprint binding for Ventorum.

Stores, validates, and resolves machine-optimized configuration settings.
If the repository or virtual environment is copied/moved to a different machine,
this module detects the hardware fingerprint mismatch and automatically reverts
to standard baseline defaults to prevent sub-optimal or incompatible execution.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from ventorum.legacy.hardware.detector import (
    HardwareInfo,
    get_machine_fingerprint,
    scan_hardware,
)

# Standard baseline defaults when no machine profile exists or when mismatch occurs
STANDARD_DEFAULTS: dict[str, Any] = {
    "default_hardware_backend": "auto",
    "gpu_available": False,
    "gpu_name": None,
    "gpu_single_crossover_panels": 200,
    "gpu_sweep_crossover_batch": 5,
    "optimal_sweep_workers": 1,
    "optimal_sweep_backend": "thread",
    "optimal_instance_concurrency": 1,
    "optimal_workers_per_instance": 1,
    "optimal_instance_backend": "thread",
    "use_symmetry_default": True,
    "gpu_max_chunk_size": 10000,
}

CONFIG_FILENAME = ".ventorum_machine_config.json"


@dataclass
class MachineConfig:
    """Persistent machine-optimized configuration container."""

    fingerprint: str
    machine_name: str
    created_at: str
    ventorum_version: str
    hardware_info: dict[str, Any]
    settings: dict[str, Any]
    benchmark_scores: dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    status: str = "matched"  # 'matched', 'mismatch_reverted', 'missing', 'disabled'

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a setting value."""
        return self.settings.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to JSON dictionary."""
        return {
            "fingerprint": self.fingerprint,
            "machine_name": self.machine_name,
            "created_at": self.created_at,
            "ventorum_version": self.ventorum_version,
            "hardware_info": self.hardware_info,
            "settings": self.settings,
            "benchmark_scores": self.benchmark_scores,
            "is_active": self.is_active,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MachineConfig:
        """Instantiate MachineConfig from dictionary."""
        return cls(
            fingerprint=data.get("fingerprint", ""),
            machine_name=data.get("machine_name", "Unknown"),
            created_at=data.get("created_at", ""),
            ventorum_version=data.get("ventorum_version", "0.1.0"),
            hardware_info=data.get("hardware_info", {}),
            settings=data.get("settings", {}),
            benchmark_scores=data.get("benchmark_scores", {}),
            is_active=data.get("is_active", True),
            status=data.get("status", "matched"),
        )


def _get_config_search_paths() -> list[pathlib.Path]:
    """Return prioritized paths to search for the machine configuration file."""
    paths = [
        pathlib.Path.cwd() / CONFIG_FILENAME,
        pathlib.Path(__file__).resolve().parent.parent.parent / CONFIG_FILENAME,
        pathlib.Path.home() / ".ventorum" / CONFIG_FILENAME,
    ]
    # Deduplicate while preserving priority order
    seen: set[str] = set()
    unique_paths = []
    for p in paths:
        norm = str(p.resolve())
        if norm not in seen:
            seen.add(norm)
            unique_paths.append(p)
    return unique_paths


def is_autotune_disabled() -> bool:
    """Check if autotuning / machine configuration has been explicitly disabled via env vars."""
    val = os.environ.get("VENTORUM_DISABLE_AUTOTUNE", "").strip().lower()
    if val in ("1", "true", "yes", "on"):
        return True
    val2 = os.environ.get("VENTORUM_AUTOTUNE", "").strip().lower()
    if val2 in ("0", "false", "no", "off"):
        return True
    return False


# In-memory cached active config
_CACHED_CONFIG: MachineConfig | None = None
_CACHED_STATUS: str = "unloaded"


def load_machine_config(
    filepath: str | pathlib.Path | None = None,
    allow_mismatch: bool = False,
    force_reload: bool = False,
    verbose_mismatch: bool = True,
) -> tuple[MachineConfig | None, str]:
    """Load and validate the machine configuration against current machine hardware.

    Parameters
    ----------
    filepath : str, Path, or None
        Specific path to configuration file. If None, searches standard locations.
    allow_mismatch : bool
        If True, returns the config object even if fingerprint does not match,
        with status='mismatch_reverted'. If False (default), returns None on mismatch.
    force_reload : bool
        If True, bypasses in-memory cache and re-reads from disk.
    verbose_mismatch : bool
        If True, outputs an informative warning when a machine mismatch is detected.

    Returns
    -------
    tuple[MachineConfig | None, str]
        (config, status) where status is:
        - 'matched': Config exists and matches current machine fingerprint!
        - 'mismatch': Config exists but belongs to a different machine. Defaults restored!
        - 'missing': No configuration file found.
        - 'disabled': Machine optimization disabled via environment variable.
    """
    global _CACHED_CONFIG, _CACHED_STATUS

    if is_autotune_disabled():
        _CACHED_CONFIG = None
        _CACHED_STATUS = "disabled"
        return None, "disabled"

    if not force_reload and _CACHED_STATUS != "unloaded" and filepath is None:
        if _CACHED_STATUS == "matched":
            return _CACHED_CONFIG, _CACHED_STATUS
        elif _CACHED_STATUS == "mismatch":
            return (_CACHED_CONFIG if allow_mismatch else None), _CACHED_STATUS
        elif _CACHED_STATUS == "missing":
            return None, "missing"

    target_file: pathlib.Path | None = None
    if filepath is not None:
        p = pathlib.Path(filepath)
        if p.is_file():
            target_file = p
    else:
        for p in _get_config_search_paths():
            if p.is_file():
                target_file = p
                break

    if target_file is None:
        _CACHED_CONFIG = None
        _CACHED_STATUS = "missing"
        return None, "missing"

    try:
        with open(target_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = MachineConfig.from_dict(data)
    except Exception as exc:
        warnings.warn(f"Failed to load Ventorum machine config from {target_file}: {exc}")
        _CACHED_CONFIG = None
        _CACHED_STATUS = "missing"
        return None, "missing"

    current_fingerprint = get_machine_fingerprint()
    if cfg.fingerprint == current_fingerprint:
        cfg.is_active = True
        cfg.status = "matched"
        _CACHED_CONFIG = cfg
        _CACHED_STATUS = "matched"
        return cfg, "matched"
    else:
        # Machine mismatch! The software was copied or moved to a different machine.
        saved_name = cfg.machine_name or "Unknown Host"
        curr_hw = scan_hardware()
        cfg.is_active = False
        cfg.status = "mismatch_reverted"
        _CACHED_CONFIG = cfg
        _CACHED_STATUS = "mismatch"

        if verbose_mismatch:
            sys.stderr.write(
                f"\n[Ventorum Portability Notice] Machine hardware mismatch detected.\n"
                f" - Saved configuration was calibrated for: '{saved_name}' (Fingerprint: {cfg.fingerprint[:8]}...)\n"
                f" - Current machine identified as: '{curr_hw.node_name}' (Fingerprint: {current_fingerprint[:8]}...)\n"
                f" Automatically reverting to safe baseline defaults. "
                f"Run 'ventorum.legacy.tune_machine()' or 'python run_machine_tuning.py' to optimize for this machine.\n\n"
            )
            sys.stderr.flush()

        if allow_mismatch:
            return cfg, "mismatch"
        return None, "mismatch"


def save_machine_config(
    config: MachineConfig,
    filepath: str | pathlib.Path | None = None,
) -> pathlib.Path:
    """Save the machine configuration to disk.

    Parameters
    ----------
    config : MachineConfig
        Configuration object to persist.
    filepath : str, Path, or None
        Destination file. If None, saves to `.ventorum_machine_config.json` in cwd.

    Returns
    -------
    Path
        Absolute path to the saved file.
    """
    global _CACHED_CONFIG, _CACHED_STATUS

    if filepath is not None:
        target_path = pathlib.Path(filepath)
    else:
        target_path = pathlib.Path.cwd() / CONFIG_FILENAME

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)

    _CACHED_CONFIG = config
    _CACHED_STATUS = "matched"
    return target_path.resolve()


def clear_machine_config(filepath: str | pathlib.Path | None = None) -> bool:
    """Remove stored machine configuration file(s)."""
    global _CACHED_CONFIG, _CACHED_STATUS
    _CACHED_CONFIG = None
    _CACHED_STATUS = "unloaded"

    removed = False
    targets = [pathlib.Path(filepath)] if filepath else _get_config_search_paths()
    for t in targets:
        if t.is_file():
            try:
                t.unlink()
                removed = True
            except Exception:
                pass
    return removed


def is_current_machine_optimized() -> bool:
    """Check if the current machine has an active, validated optimization configuration."""
    cfg, status = load_machine_config(verbose_mismatch=False)
    return status == "matched" and cfg is not None and cfg.is_active


def get_active_setting(key: str, default: Any = None, user_override: Any = None) -> Any:
    """Resolve a hardware-dependent setting value according to priority rules.

    Resolution Priority:
    1. Explicit user/agent override (if provided and != 'auto').
    2. Machine-optimized configuration value (if current machine is verified).
    3. Caller-supplied default fallback.
    4. Global standard baseline default.

    Parameters
    ----------
    key : str
        Setting name (e.g. 'optimal_sweep_workers', 'default_hardware_backend').
    default : Any, optional
        Fallback value if not configured.
    user_override : Any, optional
        Value explicitly passed by caller (e.g. n_jobs=4, backend='gpu').
    """
    # 1. Explicit user override takes absolute precedence
    if user_override is not None and user_override != "auto":
        return user_override

    # 2. Check machine config
    cfg, status = load_machine_config(verbose_mismatch=False)
    if status == "matched" and cfg is not None and cfg.is_active:
        val = cfg.get(key)
        if val is not None:
            return val

    # 3. Caller fallback or standard baseline default
    if default is not None:
        return default
    return STANDARD_DEFAULTS.get(key)
