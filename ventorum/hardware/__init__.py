# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Hardware detection, machine profile and tuner.

Run ``ventorum-tune`` (or ``python -m ventorum.hardware tune``) once after
installation, and again after a hardware change. Ventorum works without it,
with built-in defaults.
"""

from ventorum.hardware.detector import HardwareInfo, get_machine_fingerprint, scan_hardware
from ventorum.hardware.profile import (
    active_profile,
    autotune_disabled,
    profile_path,
    read_profile,
    reset_profile_cache,
    size_class,
)
from ventorum.hardware.tuner import tune_machine

__all__ = [
    "HardwareInfo",
    "active_profile",
    "autotune_disabled",
    "get_machine_fingerprint",
    "profile_path",
    "read_profile",
    "reset_profile_cache",
    "scan_hardware",
    "size_class",
    "tune_machine",
]
