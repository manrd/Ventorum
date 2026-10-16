# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests of the integral boundary-layer closure and transition relations.

The checks: analytic derivatives against central differences, exact
flat-plate (Blasius) values, the Ludwieg-Tillmann turbulent skin friction,
the published closed forms of the transition relations (Drela and Giles
1987, written again in this file), continuity at the branch points, and
numpy/torch agreement.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from ventorum.ibl2d import closure as cl
from ventorum.ibl2d import transition as tr

H_GRID = np.linspace(1.05, 7.0, 61)
RT_GRID = np.geomspace(50.0, 1e5, 23)


def _grid():
    """Return a (H, Re_theta) grid without points close to branch switches."""
    hh, rr = np.meshgrid(H_GRID, RT_GRID)
    hh = hh.ravel()
    rr = rr.ravel()
    h0 = np.where(rr > 400.0, 3.0 + 400.0 / rr, 4.0)
    keep = (np.abs(hh - h0) > 1e-3) & (np.abs(hh - 4.0) > 1e-3) & (np.abs(hh - 7.4) > 1e-3)
    keep &= np.abs(rr - 400.0) > 1e-2
    return hh[keep], rr[keep]


def _check(name, value_fn, deriv, x, k):
    """Compare an analytic derivative with a central difference (step 1e-6 relative)."""
    x = [np.asarray(a, dtype=np.float64) for a in x]
    step = 1e-6 * np.abs(x[k])
    xp_ = [a.copy() for a in x]
    xm_ = [a.copy() for a in x]
    xp_[k] = x[k] + step
    xm_[k] = x[k] - step
    fd = (value_fn(*xp_) - value_fn(*xm_)) / (2.0 * step)
    scale = np.maximum(np.abs(deriv), np.abs(value_fn(*x)) / np.abs(x[k]) * 1e-3)
    err = np.max(np.abs(deriv - fd) / scale)
    assert err < 1e-6, f"{name}: derivative {k} relative error {err:.2e}"


def test_laminar_derivatives():
    """Laminar closures: analytic derivatives agree with central differences."""
    h = H_GRID[(np.abs(H_GRID - 4.0) > 1e-3) & (np.abs(H_GRID - 7.4) > 1e-3)]
    for fn in (cl.laminar_h_star, cl.laminar_cf_factor, cl.laminar_dissipation_factor):
        val, d = fn(h)
        _check(fn.__name__, lambda a, f=fn: f(a)[0], d, [h], 0)


def test_turbulent_derivatives():
    """Turbulent closures: analytic derivatives agree with central differences."""
    h, rt = _grid()
    _, dh, dr = cl.turbulent_h_star(h, rt)
    _check("H* (H)", lambda a, b: cl.turbulent_h_star(a, b)[0], dh, [h, rt], 0)
    _check("H* (Rt)", lambda a, b: cl.turbulent_h_star(a, b)[0], dr, [h, rt], 1)
    _, dh, dr = cl.turbulent_cf(h, rt)
    _check("Cf (H)", lambda a, b: cl.turbulent_cf(a, b)[0], dh, [h, rt], 0)
    _check("Cf (Rt)", lambda a, b: cl.turbulent_cf(a, b)[0], dr, [h, rt], 1)
    hs = cl.turbulent_h_star(h, rt)[0]
    _, d1, d2 = cl.slip_velocity(h, hs)
    _check("Us (H)", lambda a, b: cl.slip_velocity(a, b)[0], d1, [h, hs], 0)
    _check("Us (H*)", lambda a, b: cl.slip_velocity(a, b)[0], d2, [h, hs], 1)
    us = cl.slip_velocity(h, hs)[0]
    ok = us < 0.95
    h2, hs2, us2 = h[ok], hs[ok], us[ok]
    _, d1, d2, d3 = cl.equilibrium_shear_coefficient(h2, hs2, us2)
    fn = lambda a, b, c: cl.equilibrium_shear_coefficient(a, b, c)[0]  # noqa: E731
    _check("Ctau_eq (H)", fn, d1, [h2, hs2, us2], 0)
    _check("Ctau_eq (H*)", fn, d2, [h2, hs2, us2], 1)
    _check("Ctau_eq (Us)", fn, d3, [h2, hs2, us2], 2)
    cf = cl.turbulent_cf(h2, rt[ok])[0]
    ct = 0.8 * cl.equilibrium_shear_coefficient(h2, hs2, us2)[0] + 1e-4
    _, d1, d2, d3 = cl.turbulent_dissipation(cf, us2, ct)
    fn = lambda a, b, c: cl.turbulent_dissipation(a, b, c)[0]  # noqa: E731
    _check("CD (Cf)", fn, d1, [cf, us2, ct], 0)
    _check("CD (Us)", fn, d2, [cf, us2, ct], 1)
    _check("CD (Ctau)", fn, d3, [cf, us2, ct], 2)
    th = 1e-3 * np.ones_like(h)
    _, d1, d2 = cl.layer_thickness(th, h)
    _check("delta (theta)", lambda a, b: cl.layer_thickness(a, b)[0], d1, [th, h], 0)
    _check("delta (H)", lambda a, b: cl.layer_thickness(a, b)[0], d2, [th, h], 1)


def test_transition_derivatives():
    """Transition relations: analytic derivatives agree with central differences."""
    h = H_GRID
    for fn in (tr.amplification_rate, tr.growth_factor):
        _, d = fn(h)
        _check(fn.__name__, lambda a, f=fn: f(a)[0], d, [h], 0)
    # Re_theta0 changes by many decades near H = 1; the check uses ln(Re_theta0),
    # whose central difference has a small truncation error.
    r0, d = tr.critical_re_theta(h)
    _check("ln Re_theta0", lambda a: np.log(tr.critical_re_theta(a)[0]), d / r0, [h], 0)
    th = 2e-4 * np.ones_like(h)
    r0 = tr.critical_re_theta(h)[0]
    rt = 1.5 * r0
    _, dh, dth, drt = tr.transition_growth_rate(h, th, rt)
    _check("dN/dxi (H)", lambda a, b, c: tr.transition_growth_rate(a, b, c)[0], dh, [h, th, rt], 0)
    _check("dN/dxi (theta)", lambda a, b, c: tr.transition_growth_rate(a, b, c)[0], dth, [h, th, rt], 1)
    assert np.all(drt == 0.0)


def test_blasius_values():
    """Laminar closures at the Blasius shape factor 2.591 within 1 % of the exact values."""
    hs = float(cl.laminar_h_star(2.591)[0])
    f = float(cl.laminar_cf_factor(2.591)[0])
    assert abs(hs / 1.5725 - 1.0) < 0.01
    assert abs(f / 0.2205 - 1.0) < 0.01


def test_turbulent_cf_ludwieg_tillmann():
    """Turbulent Cf at H = 1.4, Re_theta = 1e4 within 15 % of Ludwieg-Tillmann."""
    cf = float(cl.turbulent_cf(1.4, 1e4)[0])
    lt = 0.246 * 10.0 ** (-0.678 * 1.4) * 1e4 ** (-0.268)
    assert abs(cf / lt - 1.0) < 0.15


def test_published_closed_forms():
    """Implemented relations equal the published closed forms (Drela and Giles 1987)."""
    for h in (2.216, 2.6, 3.0):
        u = 1.0 / (h - 1.0)
        log10_r0 = (1.415 * u - 0.489) * math.tanh(20.0 * u - 12.9) + 3.295 * u + 0.44
        r0 = 10.0**log10_r0
        got = float(tr.critical_re_theta(h)[0])
        assert abs(got / r0 - 1.0) < 1e-12, (h, got, r0)
        b = 2.4 * h - 3.7 + 2.5 * math.tanh(1.5 * h - 4.65)
        rate = 0.01 * math.sqrt(b * b + 0.25)
        assert abs(float(tr.amplification_rate(h)[0]) / rate - 1.0) < 1e-12
    # Turbulent skin friction and energy shape factor, both branches.
    for h, rt in ((1.4, 1e4), (2.5, 3e3), (5.0, 2e3), (3.5, 300.0)):
        cf = 0.3 * math.exp(-1.33 * h) * math.log10(rt) ** (-1.74 - 0.31 * h) + 0.00011 * (
            math.tanh(4.0 - h / 0.875) - 1.0
        )
        assert abs(float(cl.turbulent_cf(h, rt)[0]) / cf - 1.0) < 1e-12
        h0 = 3.0 + 400.0 / rt if rt > 400.0 else 4.0
        if h < h0:
            hs = 1.505 + 4.0 / rt + (0.165 - 1.6 / math.sqrt(rt)) * (h0 - h) ** 1.6 / h
        else:
            lr = math.log(rt)
            hs = 1.505 + 4.0 / rt + (h - h0) ** 2 * (0.04 / h + 0.007 * lr / (h - h0 + 4.0 / lr) ** 2)
        assert abs(float(cl.turbulent_h_star(h, rt)[0]) / hs - 1.0) < 1e-12


def test_transition_trends():
    """Re_theta0 rises as H falls from 3 to 2.2; the growth rate is zero below Re_theta0."""
    h = np.linspace(3.0, 2.2, 41)
    r0 = tr.critical_re_theta(h)[0]
    assert np.all(np.diff(r0) > 0.0)
    th = 1e-4 * np.ones_like(h)
    below = tr.transition_growth_rate(h, th, 0.99 * r0)[0]
    above = tr.transition_growth_rate(h, th, 1.01 * r0)[0]
    assert np.all(below == 0.0)
    assert np.all(above > 0.0)


def test_branch_continuity():
    """Values and first derivatives are continuous at all branch points."""
    eps = 1e-9
    for fn, hb in ((cl.laminar_h_star, 4.0), (cl.laminar_cf_factor, 7.4), (cl.laminar_dissipation_factor, 4.0)):
        lo = fn(np.array([hb - eps]))
        hi = fn(np.array([hb + eps]))
        assert abs(lo[0][0] - hi[0][0]) < 1e-6 * abs(lo[0][0])
        assert abs(lo[1][0] - hi[1][0]) < 1e-6
    for rt in (1e3, 1e4):
        h0 = 3.0 + 400.0 / rt
        lo = cl.turbulent_h_star(np.array([h0 - eps]), np.array([rt]))
        hi = cl.turbulent_h_star(np.array([h0 + eps]), np.array([rt]))
        for a, b in zip(lo, hi):
            assert abs(a[0] - b[0]) < 1e-6 * max(abs(a[0]), 1.0)
    for h in (2.0, 3.5, 5.0):
        lo = cl.turbulent_h_star(np.array([h]), np.array([400.0 - 1e-7]))
        hi = cl.turbulent_h_star(np.array([h]), np.array([400.0 + 1e-7]))
        assert abs(lo[0][0] - hi[0][0]) < 1e-6 * abs(lo[0][0])


def _all_values(xp, h, rt, th, values_only=False):
    """Return the outputs of every relation for one array module.

    With ``values_only`` only the value (first element) of each relation is
    returned.
    """
    out = []
    out += list(cl.laminar_h_star(h, xp=xp))
    out += list(cl.laminar_cf_factor(h, xp=xp))
    out += list(cl.laminar_dissipation_factor(h, xp=xp))
    hs = cl.turbulent_h_star(h, rt, xp=xp)
    out += list(hs)
    out += list(cl.turbulent_cf(h, rt, xp=xp))
    us = cl.slip_velocity(h, hs[0], xp=xp)
    out += list(us)
    ce = cl.equilibrium_shear_coefficient(h, hs[0], us[0], xp=xp)
    out += list(ce)
    out += list(cl.turbulent_dissipation(0.003 + 0.0 * h, us[0], ce[0], xp=xp))
    out += list(cl.layer_thickness(th, h, xp=xp))
    out += list(tr.critical_re_theta(h, xp=xp))
    out += list(tr.amplification_rate(h, xp=xp))
    out += list(tr.growth_factor(h, xp=xp))
    out += list(tr.transition_growth_rate(h, th, rt, xp=xp))
    if values_only:
        starts = [0, 2, 4, 6, 9, 12, 15, 19, 23, 26, 28, 30, 32]
        return [out[k] for k in starts]
    return out


def test_numpy_torch_parity():
    """numpy and torch (float64, CPU) give the same values to 1e-12."""
    h, rt = _grid()
    h = h[h < 6.0]
    rt = rt[: h.size]
    th = 1e-3 * np.ones_like(h)
    a = _all_values(np, h, rt, th)
    b = _all_values(torch, torch.tensor(h), torch.tensor(rt), torch.tensor(th))
    for u, v in zip(a, b):
        u = np.asarray(u)
        v = v.numpy()
        assert np.all(np.abs(u - v) <= 1e-12 * np.maximum(1.0, np.abs(u)))


@pytest.mark.gpu
def test_cuda_float32():
    """CUDA float32 values agree with numpy float64 to 1e-5 relative.

    The check covers the value of every relation (not the partial
    derivatives) on the attached-flow range H = 1.2 to 3.8. The relative
    error uses a scale floor of 5 % of the largest value of each relation on
    the grid, because some published relations cross zero (the growth
    factor near H = 2.1, Re_theta Cf/2 near H = 4.03, Us at H = 4) and
    there the float32 cancellation of their terms dominates.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    h = np.linspace(1.2, 3.8, 64)
    rt = np.geomspace(200.0, 5e4, 64)
    th = 1e-3 * np.ones_like(h)
    a = _all_values(np, h, rt, th, values_only=True)
    dev = torch.device("cuda")
    b = _all_values(
        torch,
        torch.tensor(h, dtype=torch.float32, device=dev),
        torch.tensor(rt, dtype=torch.float32, device=dev),
        torch.tensor(th, dtype=torch.float32, device=dev),
        values_only=True,
    )
    for k, (u, v) in enumerate(zip(a, b)):
        u = np.asarray(u)
        v = v.double().cpu().numpy()
        err = np.max(np.abs(u - v) / np.maximum(np.abs(u), 0.05 * np.max(np.abs(u))))
        assert err < 1e-5, (k, err)


def _falkner_skan(beta):
    """Return H, H*, Re_theta Cf/2 and Re_theta 2CD/H* of the exact Falkner-Skan profile."""
    from scipy.integrate import solve_bvp

    eta = np.linspace(0.0, 10.0, 400)
    guess = np.vstack([eta - 1.0 + np.exp(-eta), 1.0 - np.exp(-eta), np.exp(-eta)])

    def rhs(t, y):
        return np.vstack([y[1], y[2], -y[0] * y[2] - beta * (1.0 - y[1] ** 2)])

    def bc(ya, yb):
        return np.array([ya[0], ya[1], yb[1] - 1.0])

    sol = solve_bvp(rhs, bc, eta, guess, tol=1e-9, max_nodes=200000)
    assert sol.success
    e = np.linspace(0.0, 10.0, 20001)
    _, fp, fpp = sol.sol(e)
    ds = np.trapezoid(1.0 - fp, e)
    th = np.trapezoid(fp * (1.0 - fp), e)
    ths = np.trapezoid(fp * (1.0 - fp * fp), e)
    diss = np.trapezoid(fpp * fpp, e)
    hs = ths / th
    return ds / th, hs, fpp[0] * th, 2.0 * th * diss / hs


@pytest.mark.parametrize("beta", [1.0, 0.3, 0.0, -0.1, -0.16])
def test_laminar_closures_against_falkner_skan(beta):
    """The laminar closures reproduce exact Falkner-Skan profiles (solved in this test).

    The published relations are fits to the Falkner-Skan family; at the
    exact H of each profile, H* agrees to 0.2 %, the dissipation factor to
    1 % and the skin-friction factor to 0.01 (absolute).
    """
    h, hs, f, g = _falkner_skan(beta)
    assert abs(float(cl.laminar_h_star(h)[0]) / hs - 1.0) < 2e-3
    assert abs(float(cl.laminar_dissipation_factor(h)[0]) / g - 1.0) < 1e-2
    assert abs(float(cl.laminar_cf_factor(h)[0]) - f) < 1e-2
