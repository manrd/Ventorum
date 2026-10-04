# Author: Manuel Alejandro Rodriguez Diaz, PhD
r"""Machine profile: the tuned execution settings of one machine.

The tuner (:mod:`ventorum.hardware.tuner`) measures the machine and writes
a profile. The dispatcher reads it, and uses it only when its fingerprint
matches this machine. Without a matching profile Ventorum uses built-in
defaults, which work on any machine. Ventorum never depends on the profile.

The profile is a JSON file in the user configuration folder:

* ``VENTORUM_CONFIG_DIR`` if this environment variable is set;
* Windows: ``%LOCALAPPDATA%\Ventorum``;
* macOS: ``~/Library/Application Support/Ventorum``;
* other systems: ``$XDG_CONFIG_HOME/ventorum`` or ``~/.config/ventorum``.

Set ``VENTORUM_DISABLE_AUTOTUNE=1`` to ignore the profile.
"""

from __future__ import annotations

import json
import os
import platform
import threading
from pathlib import Path
from typing import Any

PROFILE_FILENAME = "machine_profile.json"
SCHEMA_VERSION = 2

# Case classes by the number of panels of the lattice.
SIZE_CLASSES = ("small", "medium", "large")
SMALL_MAX_PANELS = 400
MEDIUM_MAX_PANELS = 2000

_lock = threading.Lock()
_cached: dict[str, Any] | None = None
_cached_loaded = False


def size_class(n_panels: int | None) -> str:
    """Return the case class of a lattice with *n_panels* panels."""
    if n_panels is None or n_panels <= SMALL_MAX_PANELS:
        return "small"
    if n_panels <= MEDIUM_MAX_PANELS:
        return "medium"
    return "large"


def config_dir() -> Path:
    """Return the folder of the machine profile."""
    env = os.environ.get("VENTORUM_CONFIG_DIR")
    if env:
        return Path(env)
    system = platform.system()
    if system == "Windows":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "Ventorum"
    if system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "Ventorum"
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "ventorum"


def profile_path() -> Path:
    """Return the path of the machine profile file."""
    return config_dir() / PROFILE_FILENAME


def autotune_disabled() -> bool:
    """Return True if the environment variable ``VENTORUM_DISABLE_AUTOTUNE`` switches the profile off."""
    return os.environ.get("VENTORUM_DISABLE_AUTOTUNE", "").strip().lower() in ("1", "true", "yes", "on")


def save_profile(profile: dict[str, Any], path: str | Path | None = None) -> Path:
    """Write *profile* to the profile file and make it the active profile of this process."""
    p = Path(path) if path is not None else profile_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(p)
    reset_profile_cache()
    return p


def read_profile(path: str | Path | None = None) -> dict[str, Any] | None:
    """Read the profile file. Return None if it does not exist or cannot be read."""
    p = Path(path) if path is not None else profile_path()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def reset_profile_cache() -> None:
    """Forget the profile that this process loaded (the next call reads the file again)."""
    global _cached, _cached_loaded
    with _lock:
        _cached = None
        _cached_loaded = False


def active_profile() -> dict[str, Any] | None:
    """Return the profile of this machine, or None.

    The profile is used only if it exists, its schema is known, autotuning is
    not disabled, and its fingerprint matches this machine. The result is
    kept for the life of the process.
    """
    global _cached, _cached_loaded
    if autotune_disabled():
        return None
    with _lock:
        if _cached_loaded:
            return _cached
        data = read_profile()
        if data is not None:
            from ventorum.hardware.detector import get_machine_fingerprint

            ok = data.get("schema") == SCHEMA_VERSION and data.get("fingerprint") == get_machine_fingerprint()
            data = data if ok else None
        _cached, _cached_loaded = data, True
        return _cached


def tuned_setting(*keys: str, default: Any = None) -> Any:
    """Return a value from the ``settings`` of the active profile, or *default*.

    Example: ``tuned_setting("batch", "small", "workers", default=None)``.
    """
    prof = active_profile()
    node: Any = prof.get("settings") if prof else None
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node
