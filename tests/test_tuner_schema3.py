# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests of the schema-3 tuner additions: GPU crossover, batch threads and chunk sizes."""

from __future__ import annotations

import math

import pytest

from ventorum.hardware import profile as prof
from ventorum.hardware import tuner
from ventorum.hardware.detector import get_machine_fingerprint


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("VENTORUM_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("VENTORUM_DISABLE_AUTOTUNE", raising=False)
    prof.reset_profile_cache()
    yield tmp_path
    prof.reset_profile_cache()


def test_crossover_work_bracketed():
    """The work crossover sits where the CPU/GPU time ratio crosses one."""
    works = [100.0 * 100.0, 400.0 * 400.0]
    cpu = [0.010, 0.100]
    gpu = [0.020, 0.050]
    got = tuner.crossover_work(works, cpu, gpu)
    r0 = math.log(cpu[0] / gpu[0])
    r1 = math.log(cpu[1] / gpu[1])
    f = r0 / (r0 - r1)
    expected = math.exp(math.log(works[0]) + f * (math.log(works[1]) - math.log(works[0])))
    assert got == pytest.approx(expected)
    assert works[0] < got < works[1]


def test_crossover_work_gpu_wins_everywhere():
    """The crossover is the smallest work when the GPU wins at every size."""
    works = [10000.0, 160000.0]
    assert tuner.crossover_work(works, [0.010, 0.100], [0.005, 0.050]) == works[0]


def test_crossover_work_cpu_wins_everywhere():
    """No crossover exists when the CPU wins at every size."""
    works = [10000.0, 160000.0]
    assert tuner.crossover_work(works, [0.010, 0.100], [0.020, 0.200]) is None


def test_crossover_cases_interpolated():
    """The case crossover interpolates linearly between 1 and 33 cases."""
    assert tuner.crossover_cases(2.0, 10.0, 6.0, 8.0) == pytest.approx(1.0 + 32.0 * 4.0 / 6.0)


def test_crossover_cases_at_bounds():
    """The case crossover clamps to 1 and returns None past 33 cases."""
    assert tuner.crossover_cases(6.0, 10.0, 2.0, 8.0) == 1
    assert tuner.crossover_cases(2.0, 4.0, 6.0, 8.0) is None


def test_schema_3_profile_marks_new_keys(config_dir):
    """A schema-3 profile stores the new keys next to the old ones."""
    fp = get_machine_fingerprint()
    prof.save_profile({"schema": 3, "fingerprint": fp, "settings": {
        "single": {"small": {"threads": 1}},
        "batch": {"small": {"workers": 1, "threads": 1}},
        "batch_threads": {"vlm": {"small": 2}},
        "chunk_mib": {"ground_cpu": 256.0, "gradient": 152.0},
        "gpu": {"available": False, "min_work": {"vlm": 1.0e5}, "min_cases": {"vlm": 24}},
    }})
    active = prof.active_profile()
    assert active is not None and active["schema"] == 3
    assert prof.tuned_setting("batch_threads", "vlm", "small", default=None) == 2
    assert prof.tuned_setting("chunk_mib", "ground_cpu", default=None) == 256.0
    assert prof.tuned_setting("gpu", "min_work", "vlm", default=None) == 1.0e5


@pytest.mark.slow
def test_quick_tuner_writes_schema_3_keys(config_dir, monkeypatch):
    """A quick tuner run writes schema 3 with the new entries."""
    from ventorum.hardware import tuner as tuner_mod

    monkeypatch.setattr(tuner_mod, "_CASES", {"small": (6, 2), "medium": (8, 2), "large": (10, 2)})
    monkeypatch.setattr(tuner_mod, "_REPEATS", {"small": 1, "medium": 1, "large": 1})
    monkeypatch.setattr(tuner_mod, "_QUICK_REPEATS", {"small": 1, "medium": 1})
    monkeypatch.setattr(tuner_mod, "_GPU_CASES", {True: {"vlm": [(6, 2)]}, False: {"vlm": [(6, 2)]}})
    p = tuner_mod.tune_machine(quick=True, verbose=False)
    assert p["schema"] == prof.SCHEMA_VERSION == 3
    assert set(p["settings"]["batch_threads"]) <= {"vlm", "linear", "nonlinear"}
    for _family, cells in p["settings"]["batch_threads"].items():
        assert set(cells) <= {"small", "medium"}
    assert set(p["settings"]["chunk_mib"]) == {"ground_cpu", "gradient"}
    assert p["settings"]["chunk_mib"]["ground_cpu"] > 0.0
    assert p["settings"]["chunk_mib"]["gradient"] > 0.0
    assert isinstance(p["settings"]["gpu"].get("min_work", {}), dict)
    assert isinstance(p["settings"]["gpu"].get("min_cases", {}), dict)
    assert "batch_threads" in p["measurements"] and "chunks" in p["measurements"]
    assert prof.active_profile() is not None


def test_batch_threads_keep_needs_a_clear_win():
    """The tuner keeps a batch thread count only when it beats all cores by more than 5 %."""
    cores = 12
    all_cores = 1.00
    # A 10 % win is kept, and the fastest count wins.
    assert tuner._batch_threads_keep({1: 1.20, 2: 0.90, cores: all_cores}, cores) == 2
    # A 4 % win is not enough: no entry (all cores, the default).
    assert tuner._batch_threads_keep({1: 1.20, 2: 0.96, cores: all_cores}, cores) is None
    # All cores fastest: no entry.
    assert tuner._batch_threads_keep({1: 1.20, 2: 1.10, cores: all_cores}, cores) is None
    # Exactly 5 % is not more than 5 %: no entry.
    assert tuner._batch_threads_keep({2: all_cores / 1.05, cores: all_cores}, cores) is None


def test_single_threads_keep_needs_a_clear_win():
    """The tuner keeps a single-case thread count only when it beats all cores by more than 5 %."""
    cores = 12
    all_cores = 1.00
    # A 10 % win is kept, and the fastest count wins.
    assert tuner._single_threads_keep({1: 1.20, 2: 0.90, cores: all_cores}, cores) == 2
    # A 4 % win is not enough: no entry (all cores, the default).
    assert tuner._single_threads_keep({1: 1.20, 2: 0.96, cores: all_cores}, cores) is None
    # All cores fastest: no entry.
    assert tuner._single_threads_keep({1: 1.20, 2: 1.10, cores: all_cores}, cores) is None
    # Exactly 5 % is not more than 5 %: no entry.
    assert tuner._single_threads_keep({2: all_cores / 1.05, cores: all_cores}, cores) is None


def test_self_check_drop_needs_a_clear_loss():
    """The tuner self-check drops the batch entries when a fixed workload is more than 2 % slower."""
    assert tuner._self_check_drop({"sweep": (1.03, 1.00)}) is not None
    assert tuner._self_check_drop({"sweep": (1.02, 1.00)}) is None
    assert tuner._self_check_drop({"sweep": (0.99, 1.00)}) is None
    assert "sweep" in (tuner._self_check_drop({"sweep": (1.05, 1.00)}) or "")
    kept = tuner._self_check_drop({"a": (1.00, 1.00), "b": (1.01, 1.00)})
    assert kept is None


def test_self_check_single_threads_drop_needs_a_clear_loss():
    """The tuner self-check drops the single entries when a fixed workload is more than 2 % slower."""
    assert tuner._self_check_drop({"single": (1.03, 1.00)}, kind="single") is not None
    assert tuner._self_check_drop({"single": (1.02, 1.00)}, kind="single") is None
    assert tuner._self_check_drop({"single": (0.99, 1.00)}, kind="single") is None
    drop_msg = tuner._self_check_drop({"single": (1.05, 1.00)}, kind="single")
    assert drop_msg is not None and "single" in drop_msg and "drop the single entries" in drop_msg


def test_old_flat_batch_threads_form_is_ignored(config_dir):
    """The reader ignores the old flat batch_threads form and uses all cores."""
    from ventorum.utils import parallel as par

    fp = get_machine_fingerprint()
    prof.save_profile({"schema": 3, "fingerprint": fp,
                       "settings": {"batch_threads": {"small": 2, "medium": 8, "large": 12}}})
    try:
        for family in ("vlm", "linear", "nonlinear"):
            assert par._tuned_batch_threads(100, family) is None
            assert par.single_case_threads(100, batch=33, family=family) == par.cpu_cores()
    finally:
        prof.reset_profile_cache()
