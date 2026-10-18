# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""The 3D panel kernels refuse float32; the developer path keeps the float32 behaviour.

With float32 inputs the on-plane band of the kernels follows the float32
epsilon and the panel scale, so near a long panel (a 50 m wake panel) real
off-plane targets count as on-plane and their influence becomes zero. The
kernels refuse float32 for this reason. The developer path
``unsafe_float32_kernels`` (environment variable plus context manager) keeps
the float32 behaviour runnable for the investigation of that band.
"""

from __future__ import annotations

import numpy as np
import pytest

from ventorum.panel3d.kernels import (
    UNSAFE_FLOAT32_ENV,
    influence_matrices,
    unsafe_float32_kernels,
)
from ventorum.panel3d.kernels_numba import influence_matrices_numba

#: One long wake-like panel: 50 m streamwise, 0.05 m wide [m].
_LONG_PANEL = np.array([[[1.0, 0.0, 0.0], [51.0, 0.0, 0.0], [51.0, 0.05, 0.0], [1.0, 0.05, 0.0]]])


def _point(z: float) -> np.ndarray:
    """Target just upstream of the panel leading edge, z [m] above its plane."""
    return np.array([[0.9, 0.025, z]])


@pytest.mark.parametrize("kernel", [influence_matrices, influence_matrices_numba])
def test_cpu_kernels_refuse_float32_inputs(kernel, monkeypatch):
    """Float32 points or panels raise a ValueError that gives the reason."""
    monkeypatch.delenv(UNSAFE_FLOAT32_ENV, raising=False)
    for pts, pans in ((_point(1e-4).astype(np.float32), _LONG_PANEL),
                      (_point(1e-4), _LONG_PANEL.astype(np.float32))):
        with pytest.raises(ValueError, match="refuse float32"):
            kernel(pts, pans)


@pytest.mark.parametrize("kernel", [influence_matrices, influence_matrices_numba])
def test_cpu_kernels_accept_float64_and_integer_inputs(kernel):
    """Float64 and integer inputs keep working (integer inputs count as float64)."""
    _, c = kernel(_point(1e-4), _LONG_PANEL)
    assert float(c[0, 0]) != 0.0
    b, _ = kernel(np.array([[0, 0, 5]]), np.array([[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]]))
    assert np.isfinite(b).all()


@pytest.mark.gpu
def test_warp_kernels_refuse_float32_arithmetic(monkeypatch):
    """The Warp kernels refuse float32 arithmetic, also with float64 inputs."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("warp")
    if not torch.cuda.is_available():
        pytest.skip("A CUDA GPU is not available.")
    from ventorum.panel3d.kernels_warp import influence_matrices_warp

    monkeypatch.delenv(UNSAFE_FLOAT32_ENV, raising=False)
    with pytest.raises(ValueError, match="refuse float32 arithmetic"):
        influence_matrices_warp(_point(1e-4), _LONG_PANEL, dtype="float32")
    with pytest.raises(ValueError, match="refuse float32"):
        influence_matrices_warp(_point(1e-4).astype(np.float32), _LONG_PANEL.astype(np.float32))
    _, c = influence_matrices_warp(_point(1e-4), _LONG_PANEL, dtype="float64")
    assert float(c.cpu()[0, 0]) != 0.0


def test_developer_path_needs_the_environment_variable(monkeypatch):
    """The context manager alone does not open the path."""
    monkeypatch.delenv(UNSAFE_FLOAT32_ENV, raising=False)
    with pytest.raises(RuntimeError, match=UNSAFE_FLOAT32_ENV):
        with unsafe_float32_kernels("investigation"):
            pass
    monkeypatch.setenv(UNSAFE_FLOAT32_ENV, "yes")
    with pytest.raises(RuntimeError):
        with unsafe_float32_kernels("investigation"):
            pass


def test_developer_path_needs_a_reason_and_warns(monkeypatch):
    """An empty reason is refused; every entry warns; the path closes on exit."""
    monkeypatch.setenv(UNSAFE_FLOAT32_ENV, "1")
    with pytest.raises(ValueError, match="reason"):
        with unsafe_float32_kernels("  "):
            pass
    with pytest.warns(RuntimeWarning, match="Unsafe float32 panel kernels"):
        with unsafe_float32_kernels("band study"):
            influence_matrices(_point(1e-4).astype(np.float32), _LONG_PANEL.astype(np.float32))
    with pytest.raises(ValueError, match="refuse float32"):
        influence_matrices(_point(1e-4).astype(np.float32), _LONG_PANEL)


@pytest.mark.parametrize("kernel", [influence_matrices, influence_matrices_numba])
def test_developer_path_reproduces_the_float32_band_zeroing(kernel, monkeypatch):
    """Regression fixture of the defect under investigation (float32 on-plane band).

    Float64: the doublet potential of the 50 m panel at 0.1 mm above its plane
    is about -1.96e-5. Float32 inputs: the target falls in the float32 band and
    the potential is exactly zero; at 0.3 mm it agrees with float64. When the
    band rule changes, this test must change with it, on purpose.
    """
    _, c64 = kernel(_point(1e-4), _LONG_PANEL)
    _, c64_far = kernel(_point(3e-4), _LONG_PANEL)
    monkeypatch.setenv(UNSAFE_FLOAT32_ENV, "1")
    with pytest.warns(RuntimeWarning):
        with unsafe_float32_kernels("regression fixture of the float32 band"):
            _, c32 = kernel(_point(1e-4).astype(np.float32), _LONG_PANEL.astype(np.float32))
            _, c32_far = kernel(_point(3e-4).astype(np.float32), _LONG_PANEL.astype(np.float32))
    assert abs(float(c64[0, 0]) + 1.959278e-5) < 1e-10
    assert float(c32[0, 0]) == 0.0
    assert abs(float(c32_far[0, 0]) - float(c64_far[0, 0])) <= 1e-5 * abs(float(c64_far[0, 0]))
