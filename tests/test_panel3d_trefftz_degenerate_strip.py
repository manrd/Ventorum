# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Core floor of the Trefftz trace nodes next to a zero-length strip.

Each trace node gets a core of a fraction of the shortest adjacent
strip. A strip of zero length would give a core of nearly zero. Such a
node now gets the core of the mean strip length of its trace. A trace
without a degenerate strip keeps its cores bit for bit.
"""

from __future__ import annotations

import numpy as np

from ventorum.panel3d.post import (
    TREFFTZ_CORE_FRACTION,
    TREFFTZ_DEGENERATE_STRIP_FRACTION,
    _trefftz_vortex_nodes,
)


def _trace(y_edges: np.ndarray, z: float = 0.0, x: float = 5.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return strip ends pl, pr [m] and lengths [m] of a straight trace with the given edges."""
    y = np.asarray(y_edges, dtype=float)
    pl = np.column_stack([np.full(len(y) - 1, x), y[:-1], np.full(len(y) - 1, z)])
    pr = np.column_stack([np.full(len(y) - 1, x), y[1:], np.full(len(y) - 1, z)])
    return pl, pr, np.linalg.norm(pr - pl, axis=1)


def _earlier_core(pl, pr, length):
    """Cores of the earlier rule: the fraction of the shortest adjacent strip (with the 1e-12 floor)."""
    ends = np.vstack([pr, pl])
    safe = np.maximum(np.concatenate([length, length]), 1.0e-12)
    core = np.empty(len(ends))
    for i, p in enumerate(ends):
        same = np.all(np.abs(ends - p) <= 1.0e-12, axis=1)
        core[i] = TREFFTZ_CORE_FRACTION * np.min(safe[same])
    return core


def test_valid_trace_keeps_its_cores_bit_for_bit():
    """Without a degenerate strip the cores are the earlier cores, bit for bit."""
    pl, pr, length = _trace(np.array([-1.0, -0.6, -0.1, 0.0, 0.05, 0.7, 1.0]))
    _, core = _trefftz_vortex_nodes(pl, pr, length)
    assert np.array_equal(core, _earlier_core(pl, pr, length))


def test_zero_length_strip_gets_the_floor_of_the_mean_strip():
    """The nodes next to a zero-length strip get the core of the mean strip length of the trace."""
    edges = np.array([-1.0, -0.5, 0.0, 0.0, 0.5, 1.0])
    pl, pr, length = _trace(edges)
    assert length.min() == 0.0
    position, core = _trefftz_vortex_nodes(pl, pr, length)
    assert np.all(np.isfinite(core))
    mean = float(np.mean(length))
    ends = np.vstack([pr, pl])
    at_zero = np.abs(ends[:, 1]) < 1e-12
    assert np.any(at_zero)
    np.testing.assert_allclose(core[at_zero], TREFFTZ_CORE_FRACTION * mean, rtol=1e-14)
    # The other nodes keep the earlier cores.
    earlier = _earlier_core(pl, pr, length)
    assert np.array_equal(core[~at_zero], earlier[~at_zero])
    assert earlier[at_zero].max() < 1e-12
    # The node merge is not changed: the ends at y = 0 have one position.
    assert len(np.unique(position[at_zero], axis=0)) == 1


def test_floor_is_per_trace():
    """Two separate traces each use their own mean strip length."""
    pl1, pr1, len1 = _trace(np.array([-1.0, -0.5, 0.0, 0.0, 0.5, 1.0]))
    pl2, pr2, len2 = _trace(np.array([3.0, 3.1, 3.2, 3.3]), z=2.0)
    pl, pr, length = np.vstack([pl1, pl2]), np.vstack([pr1, pr2]), np.concatenate([len1, len2])
    _, core = _trefftz_vortex_nodes(pl, pr, length)
    ends = np.vstack([pr, pl])
    at_zero = (np.abs(ends[:, 1]) < 1e-12) & (np.abs(ends[:, 2]) < 1e-12)
    np.testing.assert_allclose(core[at_zero], TREFFTZ_CORE_FRACTION * float(np.mean(len1)), rtol=1e-14)
    second = ends[:, 2] > 1.0
    np.testing.assert_allclose(core[second], TREFFTZ_CORE_FRACTION * 0.1, rtol=1e-9)


def test_tiny_but_valid_strip_is_not_floored():
    """A strip above the degenerate fraction keeps the core of its own length."""
    small = 10.0 * TREFFTZ_DEGENERATE_STRIP_FRACTION
    pl, pr, length = _trace(np.array([-1.0, -0.5, 0.0, small, 0.5, 1.0]))
    _, core = _trefftz_vortex_nodes(pl, pr, length)
    assert np.array_equal(core, _earlier_core(pl, pr, length))
