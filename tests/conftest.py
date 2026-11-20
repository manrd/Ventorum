# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Shared test settings.

The tests compare the solve paths of the CPU to the bit. On a machine with a
GPU, the ``"auto"`` device would send large solves to the GPU pipelines
(:mod:`ventorum.gpu`), whose results differ at round-off (float64) or at the
float32 level. Every test therefore runs with the CPU device, except the
tests marked ``gpu``, which select the device themselves.

The tests also must not depend on the tuner profile of the machine
(:mod:`ventorum.hardware.profile`): the profile selects kernel backends,
thread counts and GPU thresholds. For the full test session,
``VENTORUM_CONFIG_DIR`` points to an empty temporary folder (set in
``pytest_configure``, before the test modules are imported), so every test
sees no profile and uses the built-in defaults. Subprocesses of the tests
get the same setting. A test of the profile behaviour opts in: it sets
``VENTORUM_CONFIG_DIR`` to its own folder with ``monkeypatch`` (the
``config_dir`` fixtures of the tuner and hardware tests), or it uses the
``machine_profile`` fixture to read the real profile of the machine.
"""

from __future__ import annotations

import os
import shutil
import tempfile

import pytest

#: Environment variable that sets the folder of the machine profile.
CONFIG_DIR_ENV = "VENTORUM_CONFIG_DIR"

#: Profile folder of the environment before the test session, or None.
_ORIGINAL_CONFIG_DIR = os.environ.get(CONFIG_DIR_ENV)


def pytest_configure(config):
    """Point the machine profile folder to an empty folder before collection.

    The hook runs before the test modules are imported, so no module can
    read the profile of the machine at import time.
    """
    from ventorum.hardware import profile

    folder = tempfile.mkdtemp(prefix="ventorum_no_machine_profile_")
    config._ventorum_no_profile_dir = folder
    os.environ[CONFIG_DIR_ENV] = folder
    profile.reset_profile_cache()


def pytest_unconfigure(config):
    """Restore the profile folder of the environment and remove the empty folder."""
    from ventorum.hardware import profile

    if _ORIGINAL_CONFIG_DIR is None:
        os.environ.pop(CONFIG_DIR_ENV, None)
    else:
        os.environ[CONFIG_DIR_ENV] = _ORIGINAL_CONFIG_DIR
    profile.reset_profile_cache()
    folder = getattr(config, "_ventorum_no_profile_dir", None)
    if folder is not None:
        shutil.rmtree(folder, ignore_errors=True)


@pytest.fixture
def machine_profile(monkeypatch):
    """Opt in to the real tuner profile of the machine for one test.

    The fixture restores the profile folder that was set before the test
    session (or the default folder of the system) and forgets the cached
    profile before and after the test.
    """
    from ventorum.hardware import profile

    monkeypatch.delenv(CONFIG_DIR_ENV, raising=False)
    original = _ORIGINAL_CONFIG_DIR
    if original is not None:
        monkeypatch.setenv(CONFIG_DIR_ENV, original)
    profile.reset_profile_cache()
    try:
        yield profile.profile_path()
    finally:
        profile.reset_profile_cache()


@pytest.fixture(autouse=True)
def _cpu_device_for_cpu_tests(request):
    from ventorum import gpu

    if request.node.get_closest_marker("gpu") is not None:
        yield
        return
    old = gpu.get_device()
    gpu.set_device("cpu")
    try:
        yield
    finally:
        gpu.set_device(old)


@pytest.fixture
def unsafe_float32_kernels(monkeypatch):
    """Open the developer path of the float32 panel kernels for one test.

    The panel kernels refuse float32 (the float32 on-plane band zeroes real
    influences near long panels). Tests of that float32 behaviour run through
    the developer path: the environment variable plus the context manager,
    whose entry warning the fixture checks.
    """
    from ventorum.panel3d.kernels import UNSAFE_FLOAT32_ENV, unsafe_float32_kernels

    monkeypatch.setenv(UNSAFE_FLOAT32_ENV, "1")
    with pytest.warns(RuntimeWarning, match="Unsafe float32 panel kernels"):
        cm = unsafe_float32_kernels("tests of the float32 on-plane band")
        cm.__enter__()
    try:
        yield
    finally:
        cm.__exit__(None, None, None)
