# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Free-air wind shear uses the mean-flight vertical as the height axis.

The shear wind blows along the mean flight path ``d_0``. Points on one line
along ``d_0`` are at one mean-flight height and must see one wind speed, and
the gradient along ``e_z = lift_direction(alpha)`` must be the profile
gradient.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import ventorum as vt
from ventorum.aero.system import freestream_direction, lift_direction
from ventorum.unsteady.gusts import linear_shear, wind_shear


@pytest.mark.parametrize("alpha_deg", [0.0, 10.0])
def test_free_air_shear_height_is_along_the_mean_flight_vertical(alpha_deg):
    """At alpha 10 deg the speed is constant along d_0 and grows 1 m/s per m along e_z."""
    alpha = math.radians(alpha_deg)
    cond = vt.FlightCondition(V_inf=20.0, alpha=alpha)
    gust = wind_shear(linear_shear(u0=0.0, du_dz=1.0), "head")._bind_flow(cond, None)
    d_0 = freestream_direction(alpha, 0.0)
    e_z = lift_direction(alpha)
    along = np.array([s * d_0 for s in (0.0, 1.0, 5.0, 20.0)])
    speed_along = np.linalg.norm(gust(along, 0.0), axis=1)
    np.testing.assert_allclose(speed_along, 0.0, atol=1e-12)
    up = np.array([s * e_z for s in (0.5, 2.0)])
    speed_up = gust(up, 0.0) @ d_0
    np.testing.assert_allclose(speed_up, [0.5, 2.0], rtol=1e-12)


def test_free_air_shear_at_zero_alpha_uses_z():
    """At alpha 0 the height is the geometry z, as before."""
    cond = vt.FlightCondition(V_inf=20.0, alpha=0.0)
    gust = wind_shear(linear_shear(u0=1.0, du_dz=2.0), "head")._bind_flow(cond, None)
    pts = np.array([[0.3, -1.0, 0.25], [4.0, 2.0, -0.5]])
    np.testing.assert_array_equal(gust(pts, 0.0)[:, 0], 1.0 + 2.0 * pts[:, 2])
