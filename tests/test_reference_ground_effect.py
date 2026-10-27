# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests of the ground-effect reference: induced drag of an elliptic loading with an image wake.

The tolerances are three to five times the measured differences. The
measured values are in the comments.
"""

from __future__ import annotations

import numpy as np
import pytest

from ventorum.reference.ground_effect import (
    elliptic_ground_drag_ratio,
    elliptic_ground_drag_ratio_asymptote,
    elliptic_ground_drag_ratio_spectral,
    wieselsberger_prandtl_drag_ratio,
)

pytestmark = pytest.mark.filterwarnings("error")


@pytest.mark.parametrize("h_over_b", [0.01, 0.05, 0.1, 0.2, 0.5, 2.0, 100.0])
def test_two_independent_evaluations_agree(h_over_b):
    """The complex-velocity form and the Fourier (Bessel) form are two
    derivations with two quadratures. Measured: largest difference 8.5e-16."""
    r1 = elliptic_ground_drag_ratio(h_over_b)
    r2 = elliptic_ground_drag_ratio_spectral(h_over_b)
    assert abs(r1 - r2) < 1e-13


@pytest.mark.parametrize("h_over_b", [0.001, 0.05, 0.2])
def test_midpoint_rule_is_converged(h_over_b):
    """Four times the default point count changes the value by round-off only.
    Measured: largest change 2.2e-16."""
    r = elliptic_ground_drag_ratio(h_over_b)
    n = int(200 + 60.0 / np.sqrt(4.0 * h_over_b))
    assert abs(elliptic_ground_drag_ratio(h_over_b, n_points=4 * n) - r) < 1e-14


def test_ratio_falls_monotonically_from_one_to_zero():
    h = np.geomspace(1e-4, 1e3, 60)
    r = np.array([elliptic_ground_drag_ratio(x) for x in h])
    assert np.all(np.diff(r) > 0.0)
    assert np.all((r > 0.0) & (r < 1.0))


@pytest.mark.parametrize("h_over_b", [10.0, 30.0, 100.0])
def test_free_air_limit(h_over_b):
    """Far from the ground, 1 - R = (b/h)^2 / 32 (vortex pair of the image).
    Measured: 32 (h/b)^2 (1 - R) - 1 = -0.094 (b/h)^2."""
    r = elliptic_ground_drag_ratio(h_over_b)
    assert abs(32.0 * h_over_b ** 2 * (1.0 - r) - 1.0) < 0.4 / h_over_b ** 2


@pytest.mark.parametrize("beta", [0.004, 0.02, 0.04])
def test_small_height_asymptote(beta):
    """R ~ (2 beta / pi) (ln(8 / beta) - 1) with beta = 4 h / b. The remainder
    is of higher order than beta. Measured: |remainder| / beta^3 = 0.30, 0.24, 0.21."""
    h = beta / 4.0
    r = elliptic_ground_drag_ratio(h)
    assert abs(r - elliptic_ground_drag_ratio_asymptote(h)) < beta ** 3


def test_prandtl_approximation_of_wieselsberger():
    """Wieselsberger (NACA TM 77, equation 4) gives Prandtl's approximation for
    h/b from 1/30 to 1/4. Measured: largest difference 0.0013 (at h/b = 0.053)."""
    h = np.linspace(1.0 / 30.0, 0.25, 50)
    d = [wieselsberger_prandtl_drag_ratio(x) - elliptic_ground_drag_ratio(x) for x in h]
    assert max(abs(x) for x in d) < 0.004


@pytest.mark.parametrize("bad", [0.0, -0.1, np.inf, np.nan])
def test_invalid_height_is_refused(bad):
    with pytest.raises(ValueError):
        elliptic_ground_drag_ratio(bad)
