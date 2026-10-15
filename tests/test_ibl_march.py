# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests of the direct integral boundary-layer march.

The tests check ``ventorum.ibl2d.march`` in direct mode: the laminar
flat-plate thickness and shape factor, the stagnation-point shape factor,
free transition with the e^N envelope, the turbulent flat-plate skin
friction, laminar separation in an adverse gradient, and the batched
variant (numpy, torch CPU, torch CUDA).
"""

from __future__ import annotations

import numpy as np
import pytest

from ventorum.ibl2d import march as mr


def _flat(s):
    """Return unit edge velocity on ``s``."""
    return np.ones_like(s)


def test_laminar_flat_plate():
    for re in (1e5, 1e6):
        s = np.linspace(0.0, 1.0, 401)
        ue = _flat(s)
        res = mr.march_direct(s, ue, re, ncrit=99.0)
        mask = s >= 0.05
        rex = re * s[mask]
        theta_ref = 0.664 * s[mask] / np.sqrt(rex)
        np.testing.assert_allclose(res.theta[mask], theta_ref, rtol=0.02)
        np.testing.assert_allclose(res.H[mask], 2.59, rtol=0.02)


def test_stagnation_flow():
    s = np.linspace(0.0, 1.0, 201)
    ue = s.copy()
    re = 1e5
    res = mr.march_direct(s, ue, re, ncrit=99.0)
    mid = (s > 0.1) & np.isfinite(res.H)
    assert np.all(np.isfinite(res.theta[mid]))
    np.testing.assert_allclose(res.H[mid], 2.216, rtol=0.03)
    scale = res.theta[mid] * np.sqrt(re)
    np.testing.assert_allclose(scale, np.mean(scale), rtol=0.02)


def test_free_transition_flat_plate():
    s = np.linspace(0.0, 1.0, 401)
    ue = _flat(s)
    re = 1e7
    res = mr.march_direct(s, ue, re, ncrit=9.0)
    assert res.x_transition is not None
    rex_tr = re * res.x_transition
    assert 1e6 <= rex_tr <= 5e6
    n = res.N[np.isfinite(res.N)]
    assert np.all(np.diff(n[: res.n_valid]) >= -1e-12)


def test_turbulent_flat_plate_cf():
    s = np.linspace(0.0, 1.0, 401)
    ue = _flat(s)
    re = 1e7
    res = mr.march_direct(s, ue, re, ncrit=9.0, forced_transition_s=0.01)
    mask = (s >= 0.1) & np.isfinite(res.Cf)
    rex = re * s[mask]
    cf_ref = 0.0592 * rex ** (-0.2)
    np.testing.assert_allclose(res.Cf[mask], cf_ref, rtol=0.10)


def test_adverse_gradient_separation():
    s = np.linspace(0.0, 1.0, 201)
    ue = 1.0 - 0.5 * s
    re = 1e5
    res = mr.march_direct(s, ue, re, ncrit=99.0)
    assert res.separated
    assert res.x_separation is not None
    assert res.n_valid < len(s)


def test_batch_numpy_equals_single():
    s = np.linspace(0.0, 1.0, 201)
    ue1 = np.ones_like(s)
    ue2 = 1.0 - 0.2 * s
    ue_batch = np.stack([ue1, ue2])
    re_batch = np.array([5e5, 2e5])
    outs = mr.march_direct_batch(s, ue_batch, re_batch, ncrit=99.0)
    ref0 = mr.march_direct(s, ue1, float(re_batch[0]), ncrit=99.0)
    ref1 = mr.march_direct(s, ue2, float(re_batch[1]), ncrit=99.0)
    for got, ref in zip(outs, (ref0, ref1)):
        np.testing.assert_allclose(got.theta, ref.theta, rtol=1e-12,
                                   atol=1e-14, equal_nan=True)
        np.testing.assert_allclose(got.H, ref.H, rtol=1e-12,
                                   atol=1e-14, equal_nan=True)


def test_batch_torch_cpu_parity():
    torch = pytest.importorskip("torch")
    s = np.linspace(0.0, 1.0, 201)
    ue1 = np.ones_like(s)
    ue2 = 1.0 - 0.2 * s
    ue_batch = np.stack([ue1, ue2])
    re_batch = np.array([5e5, 2e5])
    outs = mr.march_direct_batch(s, ue_batch, re_batch, xp=torch, ncrit=99.0)
    ref0 = mr.march_direct(s, ue1, float(re_batch[0]), ncrit=99.0)
    ref1 = mr.march_direct(s, ue2, float(re_batch[1]), ncrit=99.0)
    for got, ref in zip(outs, (ref0, ref1)):
        gth = np.asarray(got.theta.detach().cpu().numpy(), dtype=float)
        np.testing.assert_allclose(gth, ref.theta, rtol=1e-10,
                                   atol=1e-12, equal_nan=True)


@pytest.mark.gpu
def test_cuda_float32_theta():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("A CUDA GPU is not available.")
    s = np.linspace(0.0, 1.0, 201)
    ue_batch = np.stack([np.ones_like(s)])
    re_batch = np.array([5e5])
    outs = mr.march_direct_batch(s, ue_batch, re_batch, xp=torch,
                                 dtype="float32", device="cuda", ncrit=99.0)
    ref = mr.march_direct(s, np.ones_like(s), 5e5, ncrit=99.0)
    got = np.asarray(outs[0].theta.detach().cpu().numpy(), dtype=float)
    mask = np.isfinite(ref.theta) & np.isfinite(got)
    np.testing.assert_allclose(got[mask], ref.theta[mask], rtol=1e-4, atol=1e-6)
