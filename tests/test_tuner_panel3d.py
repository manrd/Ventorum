# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests of the panel device cost model and its tuner measurements."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from ventorum import gpu
from ventorum.hardware import profile as prof
from ventorum.hardware import tuner
from ventorum.hardware.detector import get_machine_fingerprint


def _model():
    return {"n": [500, 2000, 6440], "cpu": [0.01, 0.1, 1.0],
            "gpu": [0.02, 0.05, 0.2], "winner": ["cpu", "gpu", "gpu"],
            "crossover_panels": 1000.0}


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("VENTORUM_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("VENTORUM_DISABLE_AUTOTUNE", raising=False)
    prof.reset_profile_cache()
    yield tmp_path
    prof.reset_profile_cache()


def test_panel_profile_round_trip(config_dir):
    entry = {"float32": _model()}
    data = {"schema": prof.SCHEMA_VERSION, "fingerprint": get_machine_fingerprint(),
            "settings": {"gpu": {"panel3d": entry}}}
    prof.save_profile(data)
    assert prof.SCHEMA_VERSION == 4
    assert prof.read_profile() == data
    assert prof.active_profile() == data
    assert prof.tuned_setting("gpu", "panel3d") == entry


@pytest.mark.parametrize("schema", [2, 3])
def test_old_profile_uses_panel_default(config_dir, monkeypatch, schema):
    prof.save_profile({"schema": schema, "fingerprint": get_machine_fingerprint(),
                       "settings": {"gpu": {"min_work": {"vlm": 1e99}}}})
    monkeypatch.setattr(gpu, "available", lambda: True)
    gpu.set_device("auto")
    assert prof.active_profile() is not None
    threshold = gpu.DEFAULT_PANEL3D_CROSSOVER
    for panels, expected in ((threshold - 1, False), (threshold, True)):
        decision, reason = gpu.use_gpu(panels ** 2, "panel3d", 1, panels, return_reason=True)
        assert decision is expected
        assert "panel3d default" in reason
        assert str(threshold) in reason


@pytest.mark.parametrize("panels, expected", [(500, False), (2000, True), (6440, True)])
def test_panel_dispatch_uses_its_profile(config_dir, monkeypatch, panels, expected):
    prof.save_profile({"schema": 4, "fingerprint": get_machine_fingerprint(),
                       "settings": {"gpu": {"panel3d": {"float32": _model()}}}})
    monkeypatch.setattr(gpu, "available", lambda: True)
    gpu.set_device("auto")
    gpu.set_precision("float32")
    try:
        decision, reason = gpu.use_gpu(panels ** 2, "panel3d", 1, panels, return_reason=True)
        assert decision is expected
        assert gpu.use_gpu(panels ** 2, "panel3d", 1, panels) is expected
        assert "profile gpu.panel3d.float32" in reason
        assert "1 configuration" in reason
        assert "5 %" in reason
        assert gpu.cost_model("panel3d") == _model()
    finally:
        gpu.set_precision("auto")


@pytest.mark.parametrize("gpu_time, expected", [(1.0 / 1.05, False), (0.96, False), (0.9, True)])
def test_panel_dispatch_requires_gain(monkeypatch, gpu_time, expected):
    model = _model()
    model.update(cpu=[1.0] * 3, gpu=[gpu_time] * 3)
    monkeypatch.setattr(gpu, "available", lambda: True)
    monkeypatch.setattr(gpu, "cost_model", lambda family: model)
    gpu.set_device("auto")
    assert gpu.use_gpu(500 ** 2, "panel3d", 1, 500) is expected
    assert gpu.estimate_time(model, "cpu", 500, 7) == 7.0
    assert gpu.estimate_time(model, "gpu", 500, 7) == pytest.approx(7.0 * gpu_time)


@pytest.mark.parametrize("key, value", [
    ("n", [500, 500, 6440]), ("n", [500, float("nan"), 6440]),
    ("cpu", [1.0]), ("gpu", [0.1, -1.0, 0.2]),
    ("gpu", [0.1, float("inf"), 0.2]), ("winner", ["gpu"]),
])
def test_invalid_panel_entry_uses_default(config_dir, monkeypatch, key, value):
    model = _model()
    model[key] = value
    prof.save_profile({"schema": 4, "fingerprint": get_machine_fingerprint(),
                       "settings": {"gpu": {"panel3d": {"float32": model}}}})
    monkeypatch.setattr(gpu, "available", lambda: True)
    gpu.set_device("auto")
    gpu.set_precision("float32")
    try:
        assert gpu.cost_model("panel3d") is None
        assert "panel3d default" in gpu.use_gpu(1, "panel3d", 1, 500, return_reason=True)[1]
    finally:
        gpu.set_precision("auto")


def test_panel_entry_does_not_cross_precision(config_dir):
    prof.save_profile({"schema": 4, "fingerprint": get_machine_fingerprint(),
                       "settings": {"gpu": {"panel3d": {"float32": _model()}}}})
    gpu.set_precision("float64")
    try:
        assert gpu.cost_model("panel3d") is None
    finally:
        gpu.set_precision("auto")


def test_panel_tuner_skips_without_cuda(monkeypatch):
    monkeypatch.setattr(gpu, "available", lambda: False)
    assert tuner._measure_panel3d(False, print)[0] == {}


def test_panel_tuner_quick_keeps_default():
    settings, measurements = tuner._measure_panel3d(True, print)
    assert settings == {}
    assert measurements == {"skipped": "quick mode"}


def test_panel_tuner_skips_without_numba(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def without_numba(name, *args, **kwargs):
        if name == "numba":
            raise ImportError("Numba is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(gpu, "available", lambda: True)
    monkeypatch.setattr(builtins, "__import__", without_numba)
    assert tuner._measure_panel3d(False, print) == ({}, {"skipped": "Numba is not available"})


def test_panel_default_without_explicit_count(config_dir, monkeypatch):
    monkeypatch.setattr(gpu, "available", lambda: True)
    gpu.set_device("auto")
    threshold = gpu.DEFAULT_PANEL3D_CROSSOVER
    assert gpu.use_gpu(3 * threshold ** 2, "panel3d", 3)
    assert not gpu.use_gpu(3 * (threshold - 1) ** 2, "panel3d", 3)
    assert gpu.use_gpu(3 * threshold ** 2, "panel3d", 3, return_reason=True)[0]


@pytest.mark.parametrize("check_gpu, expected", [(0.5, "gpu"), (1.03, "cpu"), (0.96, "cpu")])
def test_panel_tuner_self_check(monkeypatch, check_gpu, expected):
    monkeypatch.setattr(gpu, "available", lambda: True)
    monkeypatch.setattr(tuner, "_PANEL3D_CASES", [(2, 2)])
    cpu_times = iter([1.0, 1.0])
    gpu_times = iter([0.5, check_gpu])
    monkeypatch.setattr(tuner, "_time", lambda fn, repeats: next(cpu_times))
    monkeypatch.setattr(tuner, "_time_warm", lambda fn, repeats, warm_seconds=0.5: next(gpu_times))
    settings, measurements = tuner._measure_panel3d(False, lambda *args: None)
    entry = settings[gpu.get_precision()]
    assert entry["winner"] == [expected]
    assert measurements["rows"][0]["self_check"]["kept"] is (expected == "gpu")
    assert (entry["crossover_panels"] is not None) is (expected == "gpu")
    assert entry["gpu"][0] == max(0.5, check_gpu)


def test_panel_tuner_cpu_winner_needs_no_self_check(monkeypatch):
    monkeypatch.setattr(gpu, "available", lambda: True)
    monkeypatch.setattr(tuner, "_PANEL3D_CASES", [(2, 2)])
    monkeypatch.setattr(tuner, "_time", lambda fn, repeats: 1.0)
    monkeypatch.setattr(tuner, "_time_warm", lambda fn, repeats, warm_seconds=0.5: 2.0)
    settings, measurements = tuner._measure_panel3d(False, lambda *args: None)
    assert settings[gpu.get_precision()]["winner"] == ["cpu"]
    assert measurements["rows"][0]["self_check"] == {"skipped": "CPU wins the gain gate"}


@pytest.mark.parametrize("backend", ["numba", pytest.param("gpu", marks=pytest.mark.gpu)])
def test_panel_tuner_workload_is_cold(backend):
    if backend == "gpu" and not gpu.available():
        pytest.skip(gpu.unavailable_reason())
    mesh, run = tuner._panel3d_workload(3, 4, backend, "float32")
    first = run()
    second = run()
    assert len(mesh.panels) < 2000
    assert first["factorizations"] == second["factorizations"] == 1
    assert np.array_equal(first["cp"], second["cp"])
    assert np.isfinite(first["cp"]).all()


def test_panel_tuner_default_mesh_sizes():
    counts = [(2 * n_span + 1) * n_chord for n_span, n_chord in tuner._PANEL3D_CASES]
    assert counts == [510, 2025, 6440]


def test_panel_cost_model_does_not_change_lattice_model(config_dir):
    entry = _model()
    original = copy.deepcopy(entry)
    prof.save_profile({"schema": 4, "fingerprint": get_machine_fingerprint(),
                       "settings": {"gpu": {"panel3d": {"float32": entry}}}})
    assert gpu.cost_model("vlm") is None
    assert entry == original
