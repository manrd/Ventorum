# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests of the two-dimensional integral boundary-layer closures.

The tests check the laminar and turbulent closures of
``ventorum.ibl2d.closure`` and the transition model of
``ventorum.ibl2d.transition``: analytic partial derivatives against central
finite differences, the Blasius flat-plate values, the Ludwieg-Tillmann skin
friction, the behaviour of the critical Reynolds number, numpy/torch parity
(CPU float64 and CUDA float32), and continuity at the switch points.
"""

from __future__ import annotations

import numpy as np
import pytest

from ventorum.ibl2d import closure as cls
from ventorum.ibl2d import transition as tr

STEP = 1e-6

H_GRID = np.array([1.05, 1.5, 2.0, 2.591, 3.0, 3.9, 4.1, 5.0, 7.0])
RE_GRID = np.array([50.0, 200.0, 1e3, 1e4, 1e5])
US_GRID = np.array([0.8, 1.2, 2.0, 3.5])
CT_GRID = np.array([1e-4, 1e-3, 5e-3])
DR_GRID = np.array([-50.0, 0.0, 100.0])


def _fd(func, args, index, step=STEP):
    """Central finite difference of ``func(*args)[0]`` wrt ``args[index]``."""
    work = [np.asarray(a, dtype=float) for a in args]
    x = work[index]
    dx = step * np.where(x != 0.0, np.abs(x), 1.0)
    up = [a.copy() for a in work]
    dn = [a.copy() for a in work]
    up[index] = x + dx
    dn[index] = x - dx
    hi = np.asarray(func(*up)[0], dtype=float)
    lo = np.asarray(func(*dn)[0], dtype=float)
    return (hi - lo) / (2.0 * dx)


def _s(x):
    """Return ``x`` as a Python float (``x`` holds one value)."""
    return float(np.asarray(x, dtype=float).item())


def _check_partials(func, args):
    """Check every analytic partial of ``func`` against finite differences."""
    out = func(*args)
    assert len(out) == 1 + len(args)
    val = np.asarray(out[0], dtype=float)
    for k in range(len(args)):
        num = _fd(func, args, k)
        ana = np.asarray(out[k + 1], dtype=float)
        assert num.shape == ana.shape == val.shape
        xk = np.asarray(args[k], dtype=float)
        atol = 1e-9 * float(np.max(np.abs(val) / np.maximum(np.abs(xk), 1.0)))
        np.testing.assert_allclose(num, ana, rtol=1e-6, atol=atol)


def _mesh2d():
    """Return the (H, Re) acceptance mesh."""
    return np.meshgrid(H_GRID, RE_GRID, indexing="ij")


def test_laminar_partials():
    for func in (cls.laminar_h_star, cls.laminar_cf_factor,
                 cls.laminar_dissipation_factor):
        _check_partials(func, [H_GRID])


def test_turbulent_partials():
    h, re = _mesh2d()
    _check_partials(cls.turbulent_h_star, [h, re])
    _check_partials(cls.turbulent_cf, [h, re])
    _check_partials(cls.turbulent_slip_velocity, [h, re])


def test_equilibrium_shear_partials():
    h, re = _mesh2d()
    hs = np.asarray(cls.turbulent_h_star(h, re)[0])
    for us in US_GRID:
        uu = np.full_like(h, float(us))
        _check_partials(cls.equilibrium_shear_coefficient, [h, hs, uu])


def test_turbulent_dissipation_partials():
    h, re = _mesh2d()
    for us in US_GRID:
        for ct in CT_GRID:
            uu = np.full_like(h, float(us))
            cc = np.full_like(h, float(ct))
            _check_partials(cls.turbulent_dissipation_coefficient, [h, re, uu, cc])


def test_lag_partials():
    ct = np.array([1e-4, 5e-4, 1e-3, 5e-3, 1e-2])
    _check_partials(cls.shear_lag_coefficient, [ct])


def test_transition_partials():
    h, re = _mesh2d()
    _check_partials(tr.critical_re_theta, [h])
    _check_partials(tr.amplification_rate, [h])
    re0 = np.asarray(tr.critical_re_theta(h)[0])
    gap = np.abs(re - re0)
    assert np.all(gap > 1e-3 * re)
    for dr in DR_GRID:
        rr = np.full_like(h, float(dr))
        _check_partials(tr.transition_growth_rate, [h, re, rr])


def test_laminar_blasius_values():
    hs = _s(cls.laminar_h_star(2.591)[0])
    assert hs == pytest.approx(1.5725, rel=0.02)
    fbf = _s(cls.laminar_cf_factor(2.591)[0])
    assert fbf == pytest.approx(0.2205, rel=0.02)
    dis = _s(cls.laminar_dissipation_factor(2.591)[0])
    assert np.isfinite(dis) and dis > 0.0


def test_turbulent_flat_plate_cf():
    cf = _s(cls.turbulent_cf(1.4, 1e4)[0])
    ludwieg_tillmann = 0.246 * 10.0 ** (-0.678 * 1.4) * 1e4 ** (-0.268)
    assert cf == pytest.approx(ludwieg_tillmann, rel=0.15)


def test_critical_re_theta_rises_in_favourable_gradient():
    r30 = _s(tr.critical_re_theta(3.0)[0])
    r26 = _s(tr.critical_re_theta(2.6)[0])
    r22 = _s(tr.critical_re_theta(2.2)[0])
    assert r22 > r26 > r30


def test_amplification_sign():
    h = np.array([2.6])
    re0 = _s(tr.critical_re_theta(h)[0])
    rate = np.array([50.0])
    below = _s(tr.transition_growth_rate(h, 0.5 * re0, rate)[0])
    assert below == 0.0
    above = _s(tr.transition_growth_rate(h, 1.5 * re0, rate)[0])
    assert above > 0.0
    slope = _s(tr.amplification_rate(h)[0])
    assert slope > 0.0


def test_values_finite_on_acceptance_grid():
    h4 = np.array([1.05, 2.0, 2.591, 4.0, 7.0])
    h, re = np.meshgrid(h4, RE_GRID, indexing="ij")
    outs = [cls.laminar_h_star(h), cls.laminar_cf_factor(h),
            cls.laminar_dissipation_factor(h), cls.turbulent_h_star(h, re),
            cls.turbulent_cf(h, re), cls.turbulent_slip_velocity(h, re)]
    hs = np.asarray(cls.turbulent_h_star(h, re)[0])
    uu = np.full_like(h, 1.5)
    cc = np.full_like(h, 1e-3)
    outs += [cls.equilibrium_shear_coefficient(h, hs, uu),
             cls.turbulent_dissipation_coefficient(h, re, uu, cc),
             cls.shear_lag_coefficient(cc), tr.critical_re_theta(h),
             tr.amplification_rate(h),
             tr.transition_growth_rate(h, re, np.full_like(h, 20.0))]
    for out in outs:
        for part in out:
            arr = np.asarray(part, dtype=float)
            assert np.all(np.isfinite(arr))


def test_default_array_module_is_numpy():
    out = cls.laminar_h_star(np.array([2.0, 3.0]))
    assert isinstance(out[0], np.ndarray)
    assert isinstance(out[1], np.ndarray)


def test_laminar_switches_are_continuous():
    for func in (cls.laminar_h_star, cls.laminar_dissipation_factor):
        for delta in (1e-7, 1e-4):
            lo = func(4.0 - delta)
            hi = func(4.0 + delta)
            val_lo = _s(lo[0])
            val_hi = _s(hi[0])
            assert abs(val_hi - val_lo) <= 1e-6 * abs(val_lo)
            der_lo = _s(lo[1])
            der_hi = _s(hi[1])
            assert abs(der_hi - der_lo) <= 0.5 * delta


def test_transition_switch_is_documented():
    # The envelope growth switches on at the critical Reynolds number: zero
    # below it, g(H) * dRe/ds above it. The jump is the source relation
    # itself, so this test records its exact size instead of bounding it.
    h = np.array([2.6])
    re0 = _s(tr.critical_re_theta(h)[0])
    rate = np.array([100.0])
    below = _s(tr.transition_growth_rate(h, 0.99 * re0, rate)[0])
    above = _s(tr.transition_growth_rate(h, 1.01 * re0, rate)[0])
    slope = _s(tr.amplification_rate(h)[0])
    assert below == 0.0
    assert above == pytest.approx(slope * 100.0, rel=1e-12)
    assert above - below == pytest.approx(slope * 100.0, rel=1e-12)


def _parity_cases():
    """Return (func, args) cases for the numpy/torch parity tests."""
    h, re = _mesh2d()
    hs = np.asarray(cls.turbulent_h_star(h, re)[0])
    uu = np.full_like(h, 1.5)
    cc = np.full_like(h, 1e-3)
    rr = np.full_like(h, 20.0)
    return [
        (cls.laminar_h_star, [H_GRID]),
        (cls.laminar_cf_factor, [H_GRID]),
        (cls.laminar_dissipation_factor, [H_GRID]),
        (cls.turbulent_h_star, [h, re]),
        (cls.turbulent_cf, [h, re]),
        (cls.turbulent_slip_velocity, [h, re]),
        (cls.equilibrium_shear_coefficient, [h, hs, uu]),
        (cls.turbulent_dissipation_coefficient, [h, re, uu, cc]),
        (cls.shear_lag_coefficient, [cc]),
        (tr.critical_re_theta, [h]),
        (tr.amplification_rate, [h]),
        (tr.transition_growth_rate, [h, re, rr]),
    ]


def _to_numpy(tensor):
    return np.asarray(tensor.detach().cpu().numpy(), dtype=float)


def test_numpy_torch_parity_cpu():
    torch = pytest.importorskip("torch")
    for func, args in _parity_cases():
        ref = [np.asarray(v, dtype=float) for v in func(*args)]
        targs = [torch.as_tensor(np.asarray(a, dtype=float), dtype=torch.float64)
                 for a in args]
        got = func(*targs, xp=torch)
        assert len(got) == len(ref)
        for r, g in zip(ref, got):
            gn = _to_numpy(g)
            assert gn.shape == r.shape
            scale = max(1.0, float(np.max(np.abs(r))))
            np.testing.assert_allclose(gn, r, rtol=1e-12, atol=1e-12 * scale,
                                       err_msg=f"cpu parity of {func.__name__}")


@pytest.mark.gpu
def test_cuda_float32_parity():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("A CUDA GPU is not available.")
    for func, args in _parity_cases():
        ref = [np.asarray(v, dtype=float) for v in func(*args)]
        targs = [torch.as_tensor(np.asarray(a, dtype=np.float32), device="cuda")
                 for a in args]
        got = func(*targs, xp=torch)
        assert len(got) == len(ref)
        for r, g in zip(ref, got):
            gn = _to_numpy(g).astype(float)
            assert gn.shape == r.shape
            scale = max(1.0, float(np.max(np.abs(r))))
            np.testing.assert_allclose(gn, r, rtol=1e-5, atol=1e-5 * scale,
                                       err_msg=f"cuda parity of {func.__name__}")
