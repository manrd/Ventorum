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
