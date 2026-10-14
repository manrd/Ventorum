# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Derivative convention of the clamped lift slope at its bounds."""

from __future__ import annotations

import math

import numpy as np
import pytest


@pytest.mark.parametrize("dtype_name", ["float64", "float32"])
def test_clamped_lift_slope_derivative_at_bounds(dtype_name: str) -> None:
    """The clamp derivative is 1 at its bounds and 0 outside them.

    Regression test: the lift-slope clamp used ``torch.clamp``, whose
    derivative at a bound is not documented. It is 1 in torch 2.6 and 0
    in torch 2.14 for float bounds, so the section gradients with respect
    to a0 at the clamp bounds were zero on newer torch. The clamp is now
    made from ``torch.where``, with derivative 1 on the closed interval
    [0.2, 2.0] (bounds included) and 0 outside it, the same convention as
    the table clamp. The values still match ``numpy.clip``.
    """
    import torch

    from ventorum.sensitivity.reference import _clamp_closed

    dtype = getattr(torch, dtype_name)
    a0 = torch.tensor(
        [0.4 * math.pi, 4.0 * math.pi, 0.3 * math.pi, 5.0 * math.pi],
        dtype=dtype,
        requires_grad=True,
    )
    claf = _clamp_closed(a0 / (2.0 * math.pi), 0.2, 2.0)
    (g,) = torch.autograd.grad(claf.sum(), a0)
    tol = 0.0 if dtype == torch.float64 else 1e-7
    assert np.all(
        np.abs(claf.detach().numpy() - np.clip(
            np.array([0.4 * math.pi, 4.0 * math.pi, 0.3 * math.pi, 5.0 * math.pi])
            / (2.0 * math.pi),
            0.2,
            2.0,
        )) <= tol
    )
    factor = g.detach().numpy() * (2.0 * math.pi)
    tol = 0.0 if dtype == torch.float64 else 1e-6
    assert np.max(np.abs(factor[:2] - 1.0)) <= tol
    assert np.all(factor[2:] == 0.0)
