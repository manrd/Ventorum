# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Small vector helpers for arrays of 3-vectors (no numpy dispatch overhead)."""

from __future__ import annotations

import numpy as np


def cross3(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Cross product ``u x v`` on the last axis of arrays of 3-vectors.

    Same formula and order of operations as ``numpy.cross`` (so the same
    result to the last bit), without the axis handling of ``numpy.cross``,
    which costs more than the arithmetic for small arrays. The inputs
    broadcast against each other.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    out = np.empty(np.broadcast_shapes(u.shape, v.shape))
    out[..., 0] = u[..., 1] * v[..., 2] - u[..., 2] * v[..., 1]
    out[..., 1] = u[..., 2] * v[..., 0] - u[..., 0] * v[..., 2]
    out[..., 2] = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    return out
