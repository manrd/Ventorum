# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests of the discrete boundary-layer equations (shared engine).

The Newton solvers use the exact Jacobians that the forward-mode
derivatives give. These tests compare them with central differences of the
residuals for every interval kind, the transition interval, the
stagnation-point start and the trailing-edge start of the wake.
"""

from __future__ import annotations

import numpy as np
import torch

from ventorum.ibl2d import equations as eq
from ventorum.ibl2d.dual import variables

RE = 1e6


def _states(kind):
    """Return station values (theta, delta_star, c, ue) for two stations."""
    rng = np.random.default_rng(3)
    n = 12
    th1 = 1e-3 * (1.0 + 0.3 * rng.random(n))
    th2 = th1 * (1.02 + 0.05 * rng.random(n))
    u1 = 1.0 + 0.2 * rng.random(n)
    u2 = u1 * (0.97 + 0.04 * rng.random(n))
    if kind == eq.LAMINAR:
        h1 = 2.3 + 1.0 * rng.random(n)
        h2 = h1 + 0.1 * rng.random(n)
        c1 = 2.0 + rng.random(n)
        c2 = c1 + 0.3 * rng.random(n)
    elif kind == eq.WAKE:
        h1 = 1.3 + 0.8 * rng.random(n)
        h2 = h1 - 0.05 * rng.random(n)
        c1 = 0.03 + 0.02 * rng.random(n)
        c2 = c1 * (0.95 + 0.1 * rng.random(n))
    else:
        h1 = 1.35 + 1.5 * rng.random(n)
        h2 = h1 + 0.1 * rng.random(n)
        c1 = 0.03 + 0.03 * rng.random(n)
        c2 = c1 * (0.95 + 0.1 * rng.random(n))
    return [th1, h1 * th1, c1, u1, th2, h2 * th2, c2, u2]


def _fd_check(fun, x, rtol=1e-6):
    """Compare forward-mode derivatives of ``fun`` with central differences."""
    duals = variables([np.asarray(a, dtype=np.float64) for a in x])
    res = fun(duals)
    for k in range(len(x)):
        h = 1e-6 * np.abs(x[k])
        xp_ = [a.copy() for a in x]
        xm_ = [a.copy() for a in x]
        xp_[k] = x[k] + h
        xm_[k] = x[k] - h
        rp = fun(variables(xp_))
        rm = fun(variables(xm_))
        for j, r in enumerate(res):
            fd = (rp[j].val - rm[j].val) / (2.0 * h)
            an = r.der[..., k]
            scale = np.maximum(np.abs(an), 1e-3 * np.max(np.abs(an)) + 1e-300)
            scale = np.maximum(scale, np.abs(r.val) / np.abs(x[k]) * 1e-4 + 1e-12 / np.abs(x[k]))
            err = np.max(np.abs(an - fd) / scale)
            assert err < rtol, f"residual {j}, variable {k}: relative error {err:.2e}"


def test_interval_jacobians():
    """Laminar, turbulent and wake interval Jacobians equal central differences."""
    dxi = np.full(12, 0.01)
    for kind in (eq.LAMINAR, eq.TURBULENT, eq.WAKE):
        x = _states(kind)

        def fun(v, kind=kind):
            return eq.interval_residuals(kind, tuple(v[:4]), tuple(v[4:]), dxi, RE)

        _fd_check(fun, x)


def test_transition_jacobian():
    """The transition-interval Jacobian (free and forced) equals central differences."""
    x = _states(eq.TURBULENT)
    lam = _states(eq.LAMINAR)
    x[0:4] = lam[0:4]
    x[1] = x[0] * np.linspace(2.8, 3.4, 12)
    dxi = np.full(12, 0.02)
    ncrit = np.full(12, 9.0)
    t1 = eq.laminar_terms(*variables(x[0:4]), RE)
    assert np.all(t1["exc"].val > 0.0)
    x[2] = ncrit - np.linspace(0.2, 0.8, 12) * dxi * t1["rtil"].val
    for wf in (np.full(12, 5.0), np.full(12, 0.3)):

        def fun(v, wf=wf):
            r1, r2, r3, _ = eq.transition_residuals(tuple(v[:4]), tuple(v[4:]), dxi, RE, ncrit, wf)
            return r1, r2, r3

        _fd_check(fun, x)
    r = eq.transition_residuals(tuple(variables(x)[:4]), tuple(variables(x)[4:]), dxi, RE, ncrit,
                                np.full(12, 5.0))
    w = r[3].val
    assert np.all((w > 0.0) & (w < 1.0)), w


def test_similarity_and_trailing_edge_jacobians():
    """Stagnation-point and trailing-edge residual Jacobians equal central differences."""
    rng = np.random.default_rng(5)
    n = 8
    th = 1e-4 * (1.0 + rng.random(n))
    ds = 2.3 * th
    nn = 0.1 * rng.random(n) + 0.01
    a = 5.0 + rng.random(n)

    def sim(v):
        return eq.similarity_residuals(v[0], v[1], v[2], v[3], RE)

    _fd_check(sim, [th, ds, nn, a])
    up = _states(eq.TURBULENT)[:4]
    lo = _states(eq.LAMINAR)[4:]
    wk = [up[0] + lo[0], up[1] + lo[1] + 1e-3, 0.05 * np.ones(12)]
    turb_up = np.ones(12, dtype=bool)
    turb_lo = np.arange(12) % 2 == 0

    def te(v):
        return eq.trailing_edge_residuals(tuple(v[:4]), tuple(v[4:8]), tuple(v[8:]), 1e-3, RE, turb_up,
                                          turb_lo)

    _fd_check(te, list(up) + list(lo) + wk)


def test_torch_equals_numpy():
    """The residuals and Jacobians agree between numpy and torch (float64)."""
    x = _states(eq.TURBULENT)
    dxi = np.full(12, 0.01)
    rn = eq.interval_residuals(eq.TURBULENT, tuple(variables(x)[:4]), tuple(variables(x)[4:]), dxi, RE)
    xt = [torch.tensor(a) for a in x]
    vt = variables(xt, xp=torch)
    rt = eq.interval_residuals(eq.TURBULENT, tuple(vt[:4]), tuple(vt[4:]), torch.tensor(dxi), RE)
    for a, b in zip(rn, rt):
        assert np.allclose(a.val, b.val.numpy(), rtol=1e-12, atol=1e-14)
        assert np.allclose(a.der, b.der.numpy(), rtol=1e-12, atol=1e-10)
