# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Shared test settings.

The tests compare the solve paths of the CPU to the bit. On a machine with a
GPU, the ``"auto"`` device would send large solves to the GPU pipelines
(:mod:`ventorum.gpu`), whose results differ at round-off (float64) or at the
float32 level. Every test therefore runs with the CPU device, except the
tests marked ``gpu``, which select the device themselves.
"""

from __future__ import annotations

import pytest


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
