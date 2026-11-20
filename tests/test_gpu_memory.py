# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests of the GPU memory of the lattice cache (:mod:`ventorum.gpu.memory`).

* The GPU data of the cached lattices have a byte limit, and
  :func:`ventorum.gpu.release` frees them.
* A lattice that leaves the lattice cache loses its GPU data.
* A GPU solve that runs out of memory releases the GPU data, runs once
  more and then goes to the CPU with one warning.

The tests marked ``gpu`` need an NVIDIA CUDA GPU. No check changes a
result: the tests compare the bits of the coefficients with those of the
path without the release.
"""

from __future__ import annotations

import gc
import warnings

import numpy as np
import pytest

import ventorum as vt
from ventorum import gpu
from ventorum.geometry import lattice_cache
from ventorum.gpu import memory

ALPHAS = np.arange(-2.0, 8.0, 2.0)


def _wing(span: float = 5.0) -> vt.LiftingSurface:
    return vt.LiftingSurface(name="w", semi_span=span,
                             sections=[vt.WingSection(y_frac=0.0, chord=2.0),
                                       vt.WingSection(y_frac=1.0, chord=1.0, x_le=0.8)])


def _bits(results) -> list:
    return [(float(r.totals.CL).hex(), float(r.totals.CDi).hex(), float(r.totals.Cm).hex()) for r in results]


# ---------------------------------------------------------------------------
# GPU memory settings (no GPU needed)
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_limit(monkeypatch):
    monkeypatch.delenv(memory.ENV_CACHE_BYTES, raising=False)
    memory.set_cache_limit(None)
    yield
    memory.set_cache_limit(None)


@pytest.mark.parametrize("text, value", [("0", 0), ("1024", 1024), ("2KiB", 2048), ("1.5 MiB", 1572864),
                                         ("1gib", 1024 ** 3), ("12B", 12)])
def test_parse_bytes(text, value):
    assert memory.parse_bytes(text) == value


@pytest.mark.parametrize("text", ["", "abc", "-1", "5 GB", "inf"])
def test_parse_bytes_rejects_bad_values(text):
    with pytest.raises(ValueError):
        memory.parse_bytes(text)


def test_cache_limit_setting_and_environment(clean_limit, monkeypatch):
    monkeypatch.setattr(memory, "_total_memory", lambda: 8 * 1024 ** 3)
    assert memory.get_cache_limit() == int(memory.CACHE_FRACTION * 8 * 1024 ** 3)
    monkeypatch.setenv(memory.ENV_CACHE_BYTES, "800MiB")
    assert memory.get_cache_limit() == 800 * 1024 ** 2
    gpu.set_cache_limit("1GiB")
    assert gpu.get_cache_limit() == 1024 ** 3
    gpu.set_cache_limit(0)
    assert gpu.get_cache_limit() == 0
    gpu.set_cache_limit(None)
    assert gpu.get_cache_limit() == 800 * 1024 ** 2
    with pytest.raises(ValueError):
        gpu.set_cache_limit(-1)


def test_invalid_environment_limit_warns_and_uses_the_default(clean_limit, monkeypatch):
    monkeypatch.setattr(memory, "_total_memory", lambda: 4 * 1024 ** 3)
    monkeypatch.setenv(memory.ENV_CACHE_BYTES, "lots of memory")
    with pytest.warns(RuntimeWarning, match=memory.ENV_CACHE_BYTES):
        assert memory.get_cache_limit() == int(memory.CACHE_FRACTION * 4 * 1024 ** 3)


class _FakeLattice:
    def __init__(self, size: int):
        self.size = size
        self.geom_cache = {("gpu_lattice", "cuda:0", "float32"): object(), "airfoil_groups": "cpu data"}


def test_trim_releases_the_least_recently_used_first(monkeypatch):
    lats = [_FakeLattice(100), _FakeLattice(200), _FakeLattice(300)]   # the oldest first
    monkeypatch.setattr(memory, "_cached_lattices", lambda: list(lats))
    monkeypatch.setattr(memory, "lattice_bytes", lambda lat: lat.size if memory._is_gpu_key(
        next(iter(lat.geom_cache), None)) else 0)
    monkeypatch.setattr(memory, "_empty_pool", lambda: None)
    assert memory.trim(limit=600) == 0
    assert memory.trim(limit=500) == 100
    assert ("gpu_lattice", "cuda:0", "float32") not in lats[0].geom_cache
    assert lats[0].geom_cache == {"airfoil_groups": "cpu data"}   # the CPU data stay
    assert memory.trim(limit=300) == 200
    assert memory._is_gpu_key(next(iter(lats[2].geom_cache)))
    # The most recent lattice goes only when it alone is above the limit.
    assert memory.trim(limit=299) == 300
    assert lats[2].geom_cache == {"airfoil_groups": "cpu data"}


def test_lattice_cache_drops_gpu_data_of_lattices_that_leave_it():
    """A lattice that leaves the lattice cache loses its GPU data, also in the copies that results keep."""
    lattice_cache.clear()
    gpu_key = ("gpu_lattice", "cuda:0", "float32")
    first = vt.analyze_sweep(_wing(3.0), ALPHAS[:2], solver="vlm", n_panels=10, n_jobs=1)
    first[0].details["lattice"].geom_cache[gpu_key] = object()   # stands for a DeviceLattice
    for k in range(lattice_cache.MAX_ENTRIES):
        vt.analyze_sweep(_wing(3.5 + 0.5 * k), ALPHAS[:2], solver="vlm", n_panels=10, n_jobs=1)
    assert gpu_key not in first[0].details["lattice"].geom_cache
    kept = vt.analyze_sweep(_wing(9.0), ALPHAS[:2], solver="vlm", n_panels=10, n_jobs=1)
    kept[0].details["lattice"].geom_cache[gpu_key] = object()
    lattice_cache.clear()
    assert gpu_key not in kept[0].details["lattice"].geom_cache
    assert kept[0].details["lattice"].geom_cache   # the CPU data stay


def test_out_of_memory_errors_are_recognised():
    torch = pytest.importorskip("torch")
    from ventorum.gpu.pipeline import is_out_of_memory

    oom = getattr(torch, "OutOfMemoryError", None) or torch.cuda.OutOfMemoryError
    assert is_out_of_memory(oom("CUDA out of memory. Tried to allocate 2.00 GiB"))
    assert is_out_of_memory(RuntimeError("CUDA error: out of memory"))
    assert is_out_of_memory(RuntimeError("cusolver error: CUSOLVER_STATUS_ALLOC_FAILED"))
    assert not is_out_of_memory(RuntimeError("singular matrix"))
    assert not is_out_of_memory(ValueError("out of memory"))


def test_release_without_gpu_data_does_nothing():
    lattice_cache.clear()
    vt.analyze_sweep(_wing(), ALPHAS[:2], solver="vlm", n_panels=10, n_jobs=1)
    assert gpu.release() == 0
    assert gpu.cache_bytes() == 0


def test_out_of_memory_fallback_adds_note_and_keeps_trust_and_cl(monkeypatch):
    """Two GPU out-of-memory errors fall back to the CPU and add a note without changing trust or CL."""
    from ventorum.gpu import pipeline

    old_dev = gpu.get_device()
    try:
        gpu.set_device("cpu")
        ref_sweep = vt.analyze_sweep(_wing(), ALPHAS[:3], solver="vlm", n_panels=12, n_jobs=1)
        ref_single = vt.analyze(_wing(), vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0)),
                                vt.SolverSettings(solver_type="vlm", n_panels=12))
    finally:
        gpu.set_device(old_dev)

    def raise_oom(*args, **kwargs):
        raise RuntimeError("CUDA error: out of memory (test)")

    monkeypatch.setattr(gpu, "get_device", lambda: "gpu")
    monkeypatch.setattr(gpu, "use_gpu", lambda *args, return_reason=False, **kwargs: (
        (True, "device setting is gpu") if return_reason else True))
    monkeypatch.setattr(pipeline, "_solve_group", raise_oom)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res_sweep = vt.analyze_sweep(_wing(), ALPHAS[:3], solver="vlm", n_panels=12, n_jobs=1)
    oom_warnings = [w for w in caught if "out of memory" in str(w.message)]
    assert len(oom_warnings) == 1
    assert "out of memory" in pipeline.NOTE_OUT_OF_MEMORY.lower()
    assert "cpu" in pipeline.NOTE_OUT_OF_MEMORY.lower()
    for r, ref_r in zip(res_sweep, ref_sweep, strict=True):
        assert r.details["dispatch"]["device"] == "cpu"
        assert r.details["dispatch"]["reason"] == pipeline.REASON_OUT_OF_MEMORY
        assert pipeline.NOTE_OUT_OF_MEMORY in r.details["notes"]
        assert pipeline.NOTE_OUT_OF_MEMORY in r.totals.trust.warnings
        assert r.totals.trust.rating == ref_r.totals.trust.rating
        assert r.totals.trust.score == ref_r.totals.trust.score
        assert float(r.totals.CL).hex() == float(ref_r.totals.CL).hex()
    assert _bits(res_sweep) == _bits(ref_sweep)

    with warnings.catch_warnings(record=True) as caught_single:
        warnings.simplefilter("always")
        res_single = vt.analyze(_wing(), vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0)),
                                vt.SolverSettings(solver_type="vlm", n_panels=12))
    assert len([w for w in caught_single if "out of memory" in str(w.message)]) == 1
    assert res_single.details["dispatch"]["device"] == "cpu"
    assert res_single.details["dispatch"]["reason"] == pipeline.REASON_OUT_OF_MEMORY
    assert pipeline.NOTE_OUT_OF_MEMORY in res_single.details["notes"]
    assert pipeline.NOTE_OUT_OF_MEMORY in res_single.totals.trust.warnings
    assert res_single.totals.trust.rating == ref_single.totals.trust.rating
    assert res_single.totals.trust.score == ref_single.totals.trust.score
    assert float(res_single.totals.CL).hex() == float(ref_single.totals.CL).hex()


# ---------------------------------------------------------------------------
# GPU memory (need a CUDA GPU)
# ---------------------------------------------------------------------------


@pytest.fixture
def on_gpu(clean_limit):
    if not gpu.available():
        pytest.skip(f"The GPU pipelines cannot run: {gpu.unavailable_reason()}")
    old_dev, old_prec = gpu.get_device(), gpu._precision
    gpu.set_device("gpu")
    lattice_cache.clear()
    yield
    gpu.release()
    lattice_cache.clear()
    gpu.set_device(old_dev)
    gpu.set_precision(old_prec)


def _gpu_sweep(span: float = 5.0, n_panels: int = 20):
    res = vt.analyze_sweep(_wing(span), ALPHAS, solver="vlm", n_panels=n_panels, n_jobs=1)
    assert res[0].details["dispatch"]["device"] == "gpu"
    return res


@pytest.mark.gpu
@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_release_frees_the_gpu_data_and_keeps_the_bits(on_gpu, precision):
    import torch

    gpu.set_precision(precision)
    ref = _bits(_gpu_sweep())
    held = gpu.cache_bytes()
    assert held > 0
    allocated = torch.cuda.memory_allocated()
    assert gpu.release() == held
    gc.collect()
    assert gpu.cache_bytes() == 0
    assert torch.cuda.memory_allocated() <= allocated - held
    # The CPU lattice stays cached; the GPU data are made again with the same bits.
    assert _bits(_gpu_sweep()) == ref
    assert gpu.cache_bytes() == held


@pytest.mark.gpu
def test_zero_limit_keeps_no_gpu_data_and_the_bits(on_gpu):
    gpu.set_precision("float64")
    ref = _bits(_gpu_sweep())
    gpu.release()
    gpu.set_cache_limit(0)
    assert _bits(_gpu_sweep()) == ref
    assert gpu.cache_bytes() == 0


@pytest.mark.gpu
def test_limit_releases_the_least_recently_used_lattices(on_gpu):
    gpu.set_precision("float32")
    first = _gpu_sweep(4.0)
    one = gpu.cache_bytes()
    gpu.set_cache_limit(int(1.5 * one))
    _gpu_sweep(4.5)
    _gpu_sweep(5.0)
    assert 0 < gpu.cache_bytes() <= int(1.5 * one)
    latest = lattice_cache.entries()[-1]
    assert memory.lattice_bytes(latest) > 0
    assert memory.lattice_bytes(first[0].details["lattice"]) == 0
    # The result of the released lattice still gives its values.
    assert np.isfinite(first[0].totals.CL)


@pytest.mark.gpu
def test_result_does_not_keep_gpu_data_after_the_lattice_leaves_the_cache(on_gpu):
    gpu.set_precision("float32")
    res = _gpu_sweep()
    lat = res[0].details["lattice"]
    assert memory.lattice_bytes(lat) > 0
    lattice_cache.clear()
    assert memory.lattice_bytes(lat) == 0
    assert np.isfinite(res[0].totals.CL)


def _raise_oom(times: list):
    import torch

    oom = getattr(torch, "OutOfMemoryError", None) or torch.cuda.OutOfMemoryError
    from ventorum.gpu import pipeline

    orig = pipeline._solve_group

    def solve_group(*args, **kwargs):
        if times[0] > 0:
            times[0] -= 1
            raise oom("CUDA out of memory. Tried to allocate 20.00 GiB (test)")
        return orig(*args, **kwargs)

    return solve_group


@pytest.mark.gpu
def test_out_of_memory_once_releases_and_solves_again_on_the_gpu(on_gpu, monkeypatch):
    from ventorum.gpu import pipeline

    gpu.set_precision("float64")
    ref = _bits(_gpu_sweep())
    _gpu_sweep(6.0)   # a second cached lattice with GPU data
    assert gpu.cache_bytes() > 0
    times = [1]
    monkeypatch.setattr(pipeline, "_solve_group", _raise_oom(times))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        res = _gpu_sweep()
    assert times == [0]
    assert _bits(res) == ref
    # The other lattice lost its GPU data in the release.
    current = res[0].details["lattice"].geom_cache
    other = [lat for lat in lattice_cache.entries() if lat.geom_cache is not current]
    assert other
    assert all(memory.lattice_bytes(lat) == 0 for lat in other)


@pytest.mark.gpu
def test_out_of_memory_twice_solves_on_the_cpu_with_one_warning(on_gpu, monkeypatch):
    from ventorum.gpu import pipeline

    gpu.set_device("cpu")
    ref_res = vt.analyze_sweep(_wing(), ALPHAS, solver="vlm", n_panels=20, n_jobs=1)
    ref = _bits(ref_res)
    gpu.set_device("gpu")
    times = [2]
    monkeypatch.setattr(pipeline, "_solve_group", _raise_oom(times))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = vt.analyze_sweep(_wing(), ALPHAS, solver="vlm", n_panels=20, n_jobs=1)
    assert times == [0]
    oom_warnings = [w for w in caught if "out of memory" in str(w.message)]
    assert len(oom_warnings) == 1
    for r, ref_r in zip(res, ref_res, strict=True):
        assert r.details["dispatch"]["device"] == "cpu"
        assert r.details["dispatch"]["reason"] == pipeline.REASON_OUT_OF_MEMORY
        assert pipeline.NOTE_OUT_OF_MEMORY in r.details["notes"]
        assert pipeline.NOTE_OUT_OF_MEMORY in r.totals.trust.warnings
        assert r.totals.trust.rating == ref_r.totals.trust.rating
        assert r.totals.trust.score == ref_r.totals.trust.score
    assert _bits(res) == ref
    assert gpu.cache_bytes() == 0


@pytest.mark.gpu
def test_other_gpu_errors_are_not_caught(on_gpu, monkeypatch):
    from ventorum.gpu import pipeline

    def broken(*args, **kwargs):
        raise RuntimeError("a GPU defect that is not a memory error")

    monkeypatch.setattr(pipeline, "_solve_group", broken)
    with pytest.raises(RuntimeError, match="not a memory error"):
        _gpu_sweep()
