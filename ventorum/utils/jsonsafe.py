# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Conversion of results to strict JSON (no NaN, no numpy types)."""

from __future__ import annotations

from typing import Any

import numpy as np


def json_safe(obj: Any) -> Any:
    """Convert numpy data to plain Python; NaN and infinity become None."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if np.isfinite(v) else None
    return obj
