# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Far-wake tail at points aft of the tail start.

A tailplane more than the discrete wake length behind a wing has its
collocation points aft of the start of the semi-infinite far-wake tail. The
tail integral then has a real segment with near-singular peaks. These tests
compare the tail with fine discrete ring sums at small and moderate reduced
frequency, and check that the wing-tail pitch damping does not depend on the
discrete wake length.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import ventorum as vt
from ventorum.aero.vortex_ring import ring_velocity
from ventorum.unsteady.derivatives import dynamic_derivatives
from ventorum.unsteady.tail import displaced_steady_tail_velocity, harmonic_tail_velocity

_PL = np.array([[3.75, -0.5, 0.0]])
_PR = np.array([[3.75, 0.5, 0.0]])
_WD = np.array([1.0, 0.0, 0.0])
_X_CUT = 9.0
# Points aft of the tail start: on the centre line, near a filament, near a
# corner, far aft and below, and outboard of the strip.
_PTS = np.array([
    [4.5, 0.0, 0.1],
    [4.5, 0.45, 0.02],
    [4.0, 0.52, 0.01],
    [8.0, -0.3, -0.2],
    [5.0, 2.0, 0.3],
])
_SHIFT = np.array([_X_CUT - 3.75, 0.0, 0.0])


def _harmonic_reference(kappa: float, n_fine: int = 8000) -> np.ndarray:
    """Return a fine ring sum on ``[3.75, X_CUT]`` plus the tail after ``X_CUT``."""
    dx = (_X_CUT - 3.75) / n_fine
    x0 = 3.75 + np.arange(n_fine) * dx
    x1 = x0 + dx
    zero = np.zeros(n_fine)
    corners = np.empty((n_fine, 4, 3))
    corners[:, 0] = np.column_stack([x0, zero - 0.5, zero])
    corners[:, 1] = np.column_stack([x0, zero + 0.5, zero])
    corners[:, 2] = np.column_stack([x1, zero + 0.5, zero])
    corners[:, 3] = np.column_stack([x1, zero - 0.5, zero])
    v_ring = ring_velocity(_PTS, corners, 1e-4)
    xm = 0.5 * (x0 + x1) - 3.75
    v_near = np.einsum("msi,s->mi", v_ring, np.exp(-1j * kappa * xm))
    v_far = harmonic_tail_velocity(_PTS, _PL + _SHIFT, _PR + _SHIFT, _WD, kappa, rc=1e-4)[:, 0]
    return v_near + v_far * np.exp(-1j * kappa * (_X_CUT - 3.75))


@pytest.mark.parametrize("k", [1e-3, 0.05, 0.2, 1.0])
def test_harmonic_tail_aft_points_match_fine_ring_sum(k):
    """The harmonic tail at aft points agrees with a fine ring sum to 1e-4."""
    kappa = 2.0 * k / 0.75
    got = harmonic_tail_velocity(_PTS, _PL, _PR, _WD, kappa, rc=1e-4)[:, 0]
    ref = _harmonic_reference(kappa)
    err = np.linalg.norm(got - ref, axis=1) / np.linalg.norm(ref, axis=1)
    assert np.all(err < 1e-4), err


def _displaced_reference(kappa: float, u: np.ndarray, n: int = 200_000) -> np.ndarray:
    """Return a fine midpoint sum on ``[0, X_CUT - 3.75]`` plus the tail after ``X_CUT``."""
    length = _X_CUT - 3.75
    xi = (np.arange(n) + 0.5) * (length / n)
    out = np.zeros((len(_PTS), 3), dtype=np.complex128)
    arg = -1j * kappa * (3.75 + xi)
    psi = np.expm1(arg)
    dpsi = (-1j * kappa) * np.exp(arg)
    d_x_u = np.cross(_WD, u)
    u_x_d = -d_x_u
    d_dot_u = float(_WD @ u)
    for sign, pe in ((1.0, _PR[0]), (-1.0, _PL[0])):
        for i, p in enumerate(_PTS):
            r = p - pe
            a = -float(r @ _WD)
            d_x_r = np.cross(_WD, r)
            rho2 = float(d_x_r @ d_x_r)
            u_x_r = np.cross(u, r)
            r_dot_u = float(r @ u)
            s = xi + a
            r2 = s * s + rho2 + 1e-8
            inv_r3 = 1.0 / (np.sqrt(r2) * r2)
            inv_r5 = inv_r3 / r2
            integrand = (
                np.outer(dpsi * inv_r3, u_x_r)
                - np.outer(dpsi * xi * inv_r3, u_x_d)
                - np.outer(psi * inv_r3, d_x_u)
                + 3.0 * np.outer(psi * (r_dot_u * inv_r5 - d_dot_u * xi * inv_r5), d_x_r)
            )
            out[i] += sign / (4.0 * math.pi) * integrand.sum(axis=0) * (length / n)
    out += displaced_steady_tail_velocity(
        _PTS, _PL + _SHIFT, _PR + _SHIFT, _WD, u, u, kappa, _X_CUT, rc=1e-4,
    )[:, 0]
    return out


@pytest.mark.parametrize("k", [1e-3, 0.05, 0.2])
def test_displaced_tail_aft_points_match_fine_sum(k):
    """The displaced steady tail at aft points agrees with a fine midpoint sum to 1e-6."""
    kappa = 2.0 * k / 0.75
    u = np.array([0.0, 0.0, 0.75])
    got = displaced_steady_tail_velocity(_PTS, _PL, _PR, _WD, u, u, kappa, 3.75, rc=1e-4)[:, 0]
    ref = _displaced_reference(kappa, u)
    err = np.linalg.norm(got - ref, axis=1) / np.linalg.norm(ref, axis=1)
    assert np.all(err < 1e-6), err


@pytest.mark.slow
def test_wing_tail_pitch_damping_does_not_depend_on_wake_length():
    """A tail aft of the discrete wake gives the long-wake k -> 0 derivatives.

    The tail is 6 reference chords behind the wing, so with ``wake_chords = 5``
    its collocation points are aft of the far-wake tail start. Before the
    graded tail quadrature, ``Cm_alpha_dot(0)`` was -0.53 here against -4.57
    with a long discrete wake.
    """
    wing = vt.LiftingSurface(
        name="wing", semi_span=6.0, n_panels=8,
        sections=[vt.WingSection(y_frac=0.0, chord=0.75), vt.WingSection(y_frac=1.0, chord=0.75)],
    )
    tail = vt.LiftingSurface(
        name="tail", semi_span=1.2, n_panels=4, position=np.array([4.5, 0.0, 0.1]),
        sections=[vt.WingSection(y_frac=0.0, chord=0.45), vt.WingSection(y_frac=1.0, chord=0.45)],
    )
    ac = vt.Aircraft(name="wing_tail", surfaces=[wing, tail], ref_point=np.array([0.1875, 0.0, 0.0]))
    cond = vt.FlightCondition(V_inf=30.0, alpha=math.radians(2.0))
    sett = vt.SolverSettings(n_chord=4)
    short = dynamic_derivatives(ac, cond, sett, k=(0.0, 0.05), wake_chords=5.0)
    long = dynamic_derivatives(ac, cond, sett, k=(0.0, 0.05), wake_chords=20.0)
    for key, tol in (("CL_alpha_dot", 0.01), ("Cm_alpha_dot", 0.03)):
        assert short.k0[key] == pytest.approx(long.k0[key], rel=tol)
        assert short.derivatives[key][1] == pytest.approx(long.derivatives[key][1], rel=tol)
