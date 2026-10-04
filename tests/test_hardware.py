# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests of the machine profile, the tuner and the two-level parallel plan."""

from __future__ import annotations

import json
import threading

import numpy as np
import pytest

import ventorum as vt
from ventorum.hardware import profile as prof
from ventorum.hardware.detector import get_machine_fingerprint
from ventorum.utils import parallel as par


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("VENTORUM_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("VENTORUM_DISABLE_AUTOTUNE", raising=False)
    prof.reset_profile_cache()
    yield tmp_path
    prof.reset_profile_cache()


def _profile(fingerprint, workers=1, threads=1):
    return {"schema": prof.SCHEMA_VERSION, "fingerprint": fingerprint,
            "settings": {"single": {"small": {"threads": threads}},
                         "batch": {"small": {"workers": workers, "threads": threads}}}}


def test_profile_is_used_only_on_its_own_machine(config_dir):
    prof.save_profile(_profile(get_machine_fingerprint(), workers=1, threads=1))
    assert prof.active_profile() is not None
    assert par.plan_parallel(8, 100).source == "profile"
    prof.save_profile(_profile("another-machine"))
    assert prof.active_profile() is None
    assert par.plan_parallel(8, 100).source == "default"


def test_profile_can_be_disabled(config_dir, monkeypatch):
    prof.save_profile(_profile(get_machine_fingerprint()))
    monkeypatch.setenv("VENTORUM_DISABLE_AUTOTUNE", "1")
    prof.reset_profile_cache()
    assert prof.active_profile() is None


def test_profile_file_does_not_store_the_machine_identity(config_dir):
    from ventorum.hardware.detector import scan_hardware

    d = scan_hardware(detect_gpu=False).public_dict()
    assert "machine_guid" not in d and "node_name" not in d


def test_plan_never_oversubscribes_the_cores(config_dir):
    cores = par.cpu_cores()
    for n_jobs in ("auto", 1, 2, -1, "max", 64):
        for n_panels in (None, 100, 1000, 5000):
            p = par.plan_parallel(50, n_panels, n_jobs)
            assert 1 <= p.workers <= max(cores, 1) or n_jobs == 64
            assert p.workers * p.threads_per_worker <= max(cores, p.workers)
    with pytest.raises(ValueError):
        par.plan_parallel(4, 100, 0)


def test_no_nested_pools_inside_a_worker(config_dir):
    plan = par.ParallelPlan(2, 1, "user")
    with par.case_executor(plan) as ex:
        inner = list(ex.map(lambda _: par.plan_parallel(10, 100, "auto"), range(2)))
    assert all(p.workers == 1 and p.source == "nested" for p in inner)


def test_run_cases_keeps_the_input_order(config_dir):
    seen = set()

    def f(i):
        seen.add(threading.get_ident())
        return i * i

    assert par.run_cases(f, range(20), 100, 4) == [i * i for i in range(20)]


def test_parallel_sweep_equals_serial_sweep(config_dir):
    wing = vt.LiftingSurface(semi_span=4.0, sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=0.6)])
    st = vt.SolverSettings(n_panels=12, n_chord=3)
    alphas = np.linspace(-2.0, 8.0, 6)
    a = vt.analyze_sweep(wing, alphas, settings=st, n_jobs=1)
    b = vt.analyze_sweep(wing, alphas, settings=st, n_jobs=3)
    for ra, rb in zip(a, b):
        assert ra.totals.CL == pytest.approx(rb.totals.CL, rel=1e-12)


def test_quick_tuner_writes_a_matching_profile(config_dir, monkeypatch):
    from ventorum.aero import vortex as V
    from ventorum.hardware import tuner

    monkeypatch.setattr(tuner, "_CASES", {"small": (6, 2), "medium": (8, 2), "large": (10, 2)})
    monkeypatch.setattr(tuner, "_REPEATS", {"small": 1, "medium": 1, "large": 1})
    p = tuner.tune_machine(quick=True, verbose=False)
    data = json.loads((config_dir / prof.PROFILE_FILENAME).read_text())
    assert data["fingerprint"] == get_machine_fingerprint() == p["fingerprint"]
    assert set(data["settings"]["single"]) == {"small", "medium"}
    assert set(data["settings"]["kernels"]) == {"small", "medium"}
    for cls in ("small", "medium"):
        assert set(data["settings"]["kernels"][cls]) == set(V.KERNELS)
    assert set(data["measurements"]["kernels"]) == {"small", "medium"}
    for cls in ("small", "medium"):
        for kernel in V.KERNELS:
            assert data["measurements"]["kernels"][cls][kernel]
    assert "torch_device" in data["settings"]
    assert prof.active_profile() is not None
    assert "machine_guid" not in json.dumps(data)


def test_profile_selects_the_backend_per_kernel(config_dir):
    from ventorum.aero import vortex as V

    fp = get_machine_fingerprint()
    compiled = "cython" if V._HAVE_CYTHON else "numba"
    kernels = {"tensor": compiled, "influence": "numpy", "induced": "numba", "trefftz": compiled}
    prof.save_profile({"schema": prof.SCHEMA_VERSION, "fingerprint": fp,
                       "settings": {"kernels": {"small": kernels}}})
    try:
        for k, v in kernels.items():
            assert V.kernel_backend_for(k) == v

        wing = vt.LiftingSurface(semi_span=4.0, sections=[
            vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=0.6)])
        st = vt.SolverSettings(n_panels=12, n_chord=3)
        cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(5.0))
        old = V._kernel_backend  # the setting ("auto" stays "auto"), not the resolved backend
        try:
            V.set_kernel_backend("numpy")
            ref = vt.analyze(wing, condition=cond, settings=st).totals
            V.set_kernel_backend("auto")
            new = vt.analyze(wing, condition=cond, settings=st).totals
        finally:
            V.set_kernel_backend(old)
        for name in ("CL", "CDi", "Cm", "CY", "Cl", "Cn"):
            assert getattr(new, name) == pytest.approx(getattr(ref, name), rel=1e-12)
    finally:
        prof.reset_profile_cache()


def test_forced_backend_overrides_the_profile(config_dir):
    from ventorum.aero import vortex as V

    fp = get_machine_fingerprint()
    prof.save_profile({"schema": prof.SCHEMA_VERSION, "fingerprint": fp,
                       "settings": {"kernels": {"small": {"tensor": "numpy", "influence": "numpy",
                                                         "induced": "numpy", "trefftz": "numpy"}}}})
    old = V._kernel_backend  # the setting ("auto" stays "auto"), not the resolved backend
    try:
        V.set_kernel_backend("numba")
        for k in V.KERNELS:
            assert V.kernel_backend_for(k) == "numba"
    finally:
        V.set_kernel_backend(old)
        prof.reset_profile_cache()


def test_unavailable_backend_in_profile_falls_back(config_dir, monkeypatch):
    from ventorum.aero import vortex as V

    fp = get_machine_fingerprint()
    prof.save_profile({"schema": prof.SCHEMA_VERSION, "fingerprint": fp,
                       "settings": {"kernels": {"small": {"tensor": "cython", "influence": "cython",
                                                         "induced": "cython", "trefftz": "cython"}}}})
    monkeypatch.setattr(V, "_HAVE_CYTHON", False)
    try:
        # Without Cython, the default is Numba where it is installed, else numpy.
        expected = "numba" if V._nb.HAVE_NUMBA else "numpy"
        for k in V.KERNELS:
            assert V.kernel_backend_for(k) == expected
    finally:
        prof.reset_profile_cache()


def test_default_backend_without_profile(config_dir, monkeypatch):
    from ventorum.aero import vortex as V

    # No profile: the resolver follows the platform default.
    # macOS branch: numba, then cython, then numpy.
    monkeypatch.setattr(V, "_is_macos", lambda: True)
    monkeypatch.setattr(V._nb, "HAVE_NUMBA", True)
    monkeypatch.setattr(V, "_HAVE_CYTHON", True)
    assert V.kernel_backend_for("tensor") == "numba"
    monkeypatch.setattr(V._nb, "HAVE_NUMBA", False)
    assert V.kernel_backend_for("tensor") == "cython"
    monkeypatch.setattr(V, "_HAVE_CYTHON", False)
    assert V.kernel_backend_for("tensor") == "numpy"

    # Other systems: cython, then numba, then numpy.
    monkeypatch.setattr(V, "_is_macos", lambda: False)
    monkeypatch.setattr(V, "_HAVE_CYTHON", True)
    assert V.kernel_backend_for("tensor") == "cython"
    monkeypatch.setattr(V, "_HAVE_CYTHON", False)
    monkeypatch.setattr(V._nb, "HAVE_NUMBA", True)
    assert V.kernel_backend_for("tensor") == "numba"
    monkeypatch.setattr(V._nb, "HAVE_NUMBA", False)
    assert V.kernel_backend_for("tensor") == "numpy"


def test_schema_1_profile_is_ignored(config_dir):
    fp = get_machine_fingerprint()
    prof.save_profile({"schema": 1, "fingerprint": fp, "settings": {}})
    assert prof.active_profile() is None


def test_solve_threads_sets_the_size_hint(config_dir):
    from ventorum.aero import vortex as V

    fp = get_machine_fingerprint()
    large = "cython" if V._HAVE_CYTHON else "numpy"
    prof.save_profile({"schema": prof.SCHEMA_VERSION, "fingerprint": fp,
                       "settings": {"kernels": {
                           "small": {"tensor": "numba", "influence": "numba", "induced": "numba", "trefftz": "numba"},
                           "large": {"tensor": large, "influence": large, "induced": large, "trefftz": large}}}})
    try:
        # Outside a solve: small class (no thread-local value, no n_panels).
        assert V.kernel_backend_for("tensor") == "numba"
        with par.solve_threads(3000):
            assert V.kernel_backend_for("tensor") == large
        # After the context: the old value is back.
        assert V.kernel_backend_for("tensor") == "numba"

        # In a worker of case_executor.
        def run(_):
            with par.solve_threads(3000):
                return V.kernel_backend_for("tensor")

        plan = par.ParallelPlan(1, 1, "user")
        with par.case_executor(plan) as ex:
            results = list(ex.map(run, range(1)))
        assert results == [large]
    finally:
        prof.reset_profile_cache()


def test_unknown_kernel_key_raises():
    from ventorum.aero import vortex as V

    with pytest.raises(ValueError):
        V.kernel_backend_for("bad")


def test_forced_backend_that_is_not_available_falls_back(monkeypatch):
    """Regression test: VENTORUM_KERNEL=cython without the extension must not select it."""
    from ventorum.aero import vortex as V

    monkeypatch.setattr(V, "_kernel_backend", "cython")  # the value VENTORUM_KERNEL=cython sets
    monkeypatch.setattr(V, "_HAVE_CYTHON", False)
    for k in V.KERNELS:
        assert V.kernel_backend_for(k) == V.get_kernel_backend() != "cython"


def test_tuner_uses_its_kernel_choices_without_a_file(config_dir, monkeypatch):
    """Regression test: steps 2 and 3 use the kernel choices of step 1, also with save=False."""
    from ventorum.aero import vortex as V
    from ventorum.hardware import tuner

    monkeypatch.setattr(tuner, "_CASES", {"small": (6, 2), "medium": (8, 2), "large": (10, 2)})
    monkeypatch.setattr(tuner, "_REPEATS", {"small": 1, "medium": 1, "large": 1})
    numpy_everywhere = {k: "numpy" for k in V.KERNELS}
    monkeypatch.setattr(tuner, "_measure_kernels", lambda cls, ac, st, cores: (dict(numpy_everywhere), {}))
    seen = []
    real = V.kernel_backend_for

    def spy(kernel, n_panels=None):
        seen.append(real(kernel, n_panels))
        return seen[-1]

    monkeypatch.setattr(V, "kernel_backend_for", spy)
    prof.reset_profile_cache()
    try:
        tuner.tune_machine(quick=True, save=False, verbose=False)
    finally:
        prof.reset_profile_cache()
    assert seen and set(seen) == {"numpy"}
    assert not (config_dir / prof.PROFILE_FILENAME).exists()
    assert prof.active_profile() is None


def test_profile_can_select_torch(config_dir, monkeypatch):
    pytest.importorskip("torch")
    from ventorum.aero import vortex as V
    from ventorum.aero import vortex_torch as vtorch

    if not vtorch.HAVE_TORCH:
        pytest.skip("PyTorch is not installed.")
    monkeypatch.setenv("VENTORUM_TORCH_DEVICE", "cpu")
    fp = get_machine_fingerprint()
    kernels = {k: "torch" for k in V.KERNELS}
    prof.save_profile({"schema": prof.SCHEMA_VERSION, "fingerprint": fp,
                       "settings": {"kernels": {"small": kernels}}})
    try:
        for k in V.KERNELS:
            assert V.kernel_backend_for(k) == "torch"
    finally:
        prof.reset_profile_cache()
