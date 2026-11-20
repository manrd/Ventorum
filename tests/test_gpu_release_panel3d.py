# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests of releasing the 3D panel cache GPU data with :func:`ventorum.gpu.release`.

``ventorum.gpu.release()`` frees the GPU data of both the lattice cache and the
3D panel cache in one call. CPU-only entries of the 3D panel cache stay cached,
and a later GPU panel solve gives the same result bit for bit.
"""

from __future__ import annotations

import gc
import subprocess
import sys

import numpy as np
import pytest

from ventorum import gpu
from ventorum.core.datatypes import (
    Aircraft,
    FlightCondition,
    LiftingSurface,
    SolverSettings,
    WingSection,
)
from ventorum.geometry import lattice_cache
from ventorum.panel3d import cache as panel_cache
from ventorum.solvers.panel3d import PanelMethodSolver


def _panel_wing() -> Aircraft:
    """Return a tapered symmetric wing for 3D panel cache tests."""
    surf = LiftingSurface(
        name="wing",
        semi_span=2.0,
        sections=[
            WingSection(y_frac=0.0, chord=0.8, shape="0012"),
            WingSection(y_frac=1.0, chord=0.5, shape="0012"),
        ],
        is_symmetric=True,
    )
    return Aircraft(name="panel release wing", surfaces=[surf])


def _cond(alpha_deg: float = 4.0) -> FlightCondition:
    return FlightCondition(V_inf=30.0, alpha=float(np.radians(alpha_deg)))


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


@pytest.fixture(autouse=True)
def _clean_caches():
    lattice_cache.clear()
    panel_cache.clear_cache()
    yield
    gpu.release()
    lattice_cache.clear()
    panel_cache.clear_cache()


def test_gpu_release_with_empty_and_cpu_only_panel_cache():
    """gpu.release() returns 0 on an empty or CPU-only panel cache and keeps CPU entries."""
    assert panel_cache.cache_info()["entries"] == 0
    assert gpu.release() == 0

    st_cpu = SolverSettings(n_panels=5, n_chord=6, panel_backend="numpy")
    first = PanelMethodSolver().solve(_panel_wing(), _cond(3.0), st_cpu)
    assert first.details["dispatch"]["device"] == "cpu"
    info_before = panel_cache.cache_info()
    assert info_before["entries"] == 1
    assert info_before["gpu_entries"] == 0
    assert info_before["device_bytes"] == 0

    assert gpu.release() == 0

    info_after = panel_cache.cache_info()
    assert info_after["entries"] == 1
    assert info_after["gpu_entries"] == 0
    assert info_after["host_bytes"] == info_before["host_bytes"]

    # The CPU entry is still cached and reused on the next solve.
    second = PanelMethodSolver().solve(_panel_wing(), _cond(5.0), st_cpu)
    assert second.details["dispatch"]["cached_solver"] is True
    assert second.details["dispatch"]["factorization_count"] == 0


def test_gpu_release_does_not_import_torch():
    """Calling gpu.release() with an empty cache does not import torch."""
    code = (
        "import sys; "
        "from ventorum import gpu; "
        "import ventorum.panel3d.cache; "
        "assert 'torch' not in sys.modules; "
        "assert gpu.release() == 0; "
        "assert 'torch' not in sys.modules"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout


@pytest.mark.gpu
@pytest.mark.skipif(not _cuda_available(), reason="needs a CUDA GPU")
def test_gpu_release_frees_panel3d_gpu_cache_and_keeps_bits():
    """gpu.release() frees the 3D panel GPU cache, keeps CPU entries, and preserves CL bits."""
    import torch

    st_gpu = SolverSettings(n_panels=5, n_chord=6, panel_backend="gpu")
    st_cpu = SolverSettings(n_panels=5, n_chord=6, panel_backend="numpy")

    # Populate both a CPU entry and a GPU entry in the 3D panel cache.
    PanelMethodSolver().solve(_panel_wing(), _cond(2.0), st_cpu)
    first = PanelMethodSolver().solve(_panel_wing(), _cond(4.0), st_gpu)
    assert first.details["dispatch"]["device"] == "gpu"

    info = panel_cache.cache_info()
    assert info["entries"] == 2
    assert info["gpu_entries"] == 1
    panel_gpu_bytes = info["device_bytes"]
    assert panel_gpu_bytes > 0
    assert gpu.cache_bytes() == 0

    allocated_before = torch.cuda.memory_allocated()
    freed = gpu.release()
    gc.collect()

    assert freed == panel_gpu_bytes
    info_after = panel_cache.cache_info()
    assert info_after["gpu_entries"] == 0
    assert info_after["device_bytes"] == 0
    # The CPU panel entry stays cached.
    assert info_after["entries"] == 1
    assert torch.cuda.memory_allocated() <= allocated_before - panel_gpu_bytes

    # Every remaining entry holds no GPU tensors.
    with panel_cache._lock:
        for entry in panel_cache._entries.values():
            assert not entry.gpu
            assert panel_cache.solver_bytes(entry.solver)[1] == 0

    # A second GPU solve builds the GPU solver again and gives the exact same CL bits.
    second = PanelMethodSolver().solve(_panel_wing(), _cond(4.0), st_gpu)
    assert second.details["dispatch"]["device"] == "gpu"
    assert second.details["dispatch"]["cached_solver"] is False
    assert float(second.totals.CL).hex() == float(first.totals.CL).hex()
    assert float(second.totals.CDi).hex() == float(first.totals.CDi).hex()
    assert float(second.totals.Cm).hex() == float(first.totals.Cm).hex()
