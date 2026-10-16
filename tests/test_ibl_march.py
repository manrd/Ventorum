# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests of the direct integral boundary-layer march.

Reference solutions: Blasius (theta = 0.664 x / sqrt(Re_x)), Hiemenz
(H = 2.216), the envelope e^N transition on a flat plate, the turbulent
flat-plate skin friction Cf = 0.0592 Re_x^-0.2 (Schlichting), and a
decelerating flow that separates.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ventorum.ibl2d import march as mr


@pytest.mark.parametrize("re", [1e5, 1e6])
def test_laminar_flat_plate(re):
    """Laminar flat plate: theta within 2 % of Blasius and H within 2 % of 2.59."""
    s = np.linspace(0.0, 1.0, 101)
    r = mr.march_direct(s, np.ones_like(s), re, ncrit=1e9)
    m = s >= 0.05
    theta_ex = 0.664 * s[m] / np.sqrt(re * s[m])
    assert not r.separated
    assert np.max(np.abs(r.theta[m] / theta_ex - 1.0)) < 0.02
    assert np.max(np.abs(r.H[m] / 2.59 - 1.0)) < 0.02


def test_hiemenz_stagnation():
    """Stagnation flow ue = a s: H within 2 % of 2.216 and theta constant."""
    s = np.linspace(0.0, 0.2, 41)
    r = mr.march_direct(s, 5.0 * s, 1e6, ncrit=1e9)
    assert np.max(np.abs(r.H / 2.216 - 1.0)) < 0.02
    th = r.theta[1:]
    assert np.max(np.abs(th / th[0] - 1.0)) < 1e-6
    # The scaled thickness theta sqrt(a Re) of the closure solution, against the exact 0.2923.
    assert abs(th[0] * np.sqrt(5.0 * 1e6) / 0.2923 - 1.0) < 0.05


def test_free_transition_flat_plate():
    """Free transition at Re 1e7, ncrit 9 between Re_x 1e6 and 5e6; it moves downstream with ncrit."""
    s = np.concatenate([[0.0], np.geomspace(1e-4, 1.0, 250)])
    xtr = []
    for nc in (4.0, 9.0, 14.0):
        r = mr.march_direct(s, np.ones_like(s), 1e7, ncrit=nc)
        assert not r.separated
        lam = ~r.turbulent
        assert np.all(np.diff(r.N[lam]) >= 0.0)
        xtr.append(r.transition_s)
    print(f"\n[march] free transition Re_x at ncrit 4, 9, 14: {np.array(xtr) * 1e7}")
    assert 1e6 <= xtr[1] * 1e7 <= 5e6
    assert xtr[0] < xtr[1] < xtr[2]


def test_turbulent_flat_plate():
    """Forced transition at 1 % chord, Re 1e7: Cf within 10 % of 0.0592 Re_x^-0.2."""
    s = np.linspace(0.0, 1.0, 201)
    r = mr.march_direct(s, np.ones_like(s), 1e7, forced_transition_s=0.01)
    assert not r.separated
    assert abs(r.transition_s - 0.01) < 1e-12
    rex = s * 1e7
    m = (rex >= 1e6) & (rex <= 1e7)
    ref = 0.0592 * rex[m] ** -0.2
    err = np.max(np.abs(r.Cf[m] / ref - 1.0))
    print(f"\n[march] turbulent flat plate: largest Cf deviation {err:.3f}")
    assert err < 0.10


def test_adverse_gradient_separation():
    """ue = 1 - 0.5 s: the turbulent H grows and separation is flagged; laminar separation too."""
    s = np.linspace(0.0, 1.0, 101)
    ue = 1.0 - 0.5 * s
    r = mr.march_direct(s, ue, 1e6, forced_transition_s=0.05)
    assert r.separated
    ok = np.isfinite(r.H) & r.turbulent
    h = r.H[ok]
    assert h[-1] > h[len(h) // 2] > 1.4
    assert np.isfinite(r.separation_s)
    r2 = mr.march_direct(s, ue, 1e6)
    assert r2.separated and np.isnan(r2.transition_s)


def test_batch_equals_single():
    """numpy batch equals single runs to 1e-12; torch float64 equals numpy to 1e-10."""
    s = np.concatenate([[0.0], np.geomspace(1e-3, 1.0, 80)])
    ues = np.stack([np.ones_like(s), 1.0 + 0.1 * s, 1.0 - 0.1 * s])
    res = np.array([3e6, 5e6, 4e6])
    b = mr.march_direct_batch(s, ues, res, ncrit=9.0)
    for k in range(3):
        r = mr.march_direct(s, ues[k], res[k], ncrit=9.0)
        assert np.allclose(b.theta[k], r.theta, rtol=1e-12, atol=0.0, equal_nan=True)
        assert np.allclose(b.H[k], r.H, rtol=1e-12, atol=0.0, equal_nan=True)
    t = mr.march_direct_batch(s, ues, res, xp=torch, ncrit=9.0)
    assert np.allclose(t.theta.numpy(), b.theta, rtol=1e-10, atol=0.0, equal_nan=True)
    assert np.allclose(t.H.numpy(), b.H, rtol=1e-10, atol=0.0, equal_nan=True)


@pytest.mark.gpu
def test_batch_cuda_float32():
    """CUDA float32 batch within 1e-4 relative of numpy on theta."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    s = np.concatenate([[0.0], np.geomspace(1e-3, 1.0, 80)])
    ues = np.stack([np.ones_like(s), 1.0 + 0.1 * s])
    res = np.array([3e5, 5e5])
    b = mr.march_direct_batch(s, ues, res, ncrit=9.0)
    t = mr.march_direct_batch(s, ues, res, xp=torch, ncrit=9.0, device="cuda", dtype=torch.float32)
    th = t.theta.double().cpu().numpy()
    m = np.isfinite(b.theta) & (b.theta > 0.0)
    assert np.max(np.abs(th[m] / b.theta[m] - 1.0)) < 1e-4
