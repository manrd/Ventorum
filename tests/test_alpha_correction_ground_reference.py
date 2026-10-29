"""Ground-effect section reference of the thick-surface (3DP) alpha correction.

The 2D section reference of a strip in ground effect is the pitch curve
``Cl(alpha; h)`` of the section at a fixed quarter-chord height, with the
onset parallel to the ground: the curve of a polar with a height axis.
The section line is its tangent at the attitude of the strip; the
coupling inverts the full curve (``tests/test_panel3d_v1_ground_reference.py``).
A line from
the onsets 0 and 1 deg with the ground tilted by the attitude lets the
onset cross the ground. Its value at the attitude was 10 % high at
h/c = 0.5 and 30 % high at h/c = 0.25 (10 deg).

The independent reference is the 2D linear-vortex panel code
``ventorum.panel2d.solver.solve_inviscid`` (method of images).
"""

import warnings

import numpy as np
import pytest

from ventorum.aero.polar_table import PolarTable
from ventorum.airfoil import airfoil
from ventorum.core.datatypes import (
    Aircraft, FlightCondition, LiftingSurface, SolverSettings, TabulatedAirfoil, WingSection,
)
from ventorum.panel2d.solver import solve_inviscid
from ventorum.panel3d.mesh import _close_trailing_edge, _resample_section, _target_x
from ventorum.solvers.factory import make_solver
from ventorum.solvers.panel3d import _morino_section_ground_line
from ventorum.viscous.alpha_correction import solve_viscous

#: Angles [deg] of the identity polars.
POLAR_ANGLES_DEG = np.arange(-14.0, 14.01, 0.5)


def _closed_section(n_points):
    """Return a NACA 0012 polygon with a closed trailing edge, as on the panel mesh."""
    x = _target_x(n_points, "cosine")
    yu, yl = _resample_section(airfoil("0012").coords, x)
    yu, yl = _close_trailing_edge(x, yu, yl)
    return np.vstack([np.column_stack([x[::-1], yu[::-1]]), np.column_stack([x[1:], yl[1:]])])


def _pitch_curve(height, angles_deg):
    """Return Cl and Cm [-] of NACA 0012 pitched about its quarter chord at the height [chord]."""
    result = solve_inviscid(_closed_section(100), angles_deg, height=height, pivot=(0.25, 0.0))
    return np.asarray(result.Cl, dtype=float).ravel(), np.asarray(result.Cm, dtype=float).ravel()


@pytest.mark.parametrize("height", [0.5, 0.25])
@pytest.mark.parametrize("attitude_deg", [2.0, 6.0, 10.0])
def test_ground_section_line_is_the_tangent_at_the_attitude(height, attitude_deg):
    """The 2D ground line matches the pitch curve of an independent code at the attitude.

    Tolerances from the data of a 100-point section: Cl within 0.9 %,
    slope within 0.6 % and Cm within 0.0021 of the panel2d values.
    """
    theta = np.radians(attitude_deg)
    normal = np.array([-np.sin(theta), np.cos(theta)])
    offset = -0.25 * np.sin(theta) - height
    slope, zero, moment = _morino_section_ground_line(_closed_section(100), (normal, offset))
    cl, cm = _pitch_curve(height, [attitude_deg - 0.5, attitude_deg, attitude_deg + 0.5])
    reference_slope = (cl[2] - cl[0]) / np.radians(1.0)
    co, si = np.cos(theta), np.sin(theta)
    assert slope * (theta - zero) == pytest.approx(cl[1], rel=0.015)
    assert slope == pytest.approx(reference_slope, rel=0.01)
    assert moment[0] + moment[1] * co**2 + moment[2] * co * si + moment[3] * si**2 == pytest.approx(
        cm[1], abs=0.003)


def _wing():
    secs = [WingSection(0.0, 1.0, x_le=-0.25, shape="0012"), WingSection(1.0, 1.0, x_le=-0.25, shape="0012")]
    return Aircraft(surfaces=[LiftingSurface(semi_span=4.0, sections=secs, spacing="uniform")])


def _identity_polars():
    """Return the free-air polar and the height-axis table of the NACA 0012 pitch curves."""
    cl, cm = _pitch_curve(None, POLAR_ANGLES_DEG)
    free = TabulatedAirfoil(alpha=np.radians(POLAR_ANGLES_DEG), Cl_data=cl, Cd_data=np.zeros_like(cl),
                            Cm_data=cm)
    heights = np.array([0.4, 0.5, 0.7])
    curves = [_pitch_curve(h, POLAR_ANGLES_DEG) for h in heights]
    cl3 = np.array([[[c[0] for c in curves]]] * 2)
    cm3 = np.array([[[c[1] for c in curves]]] * 2)
    table = PolarTable(Re=np.array([1e5, 1e8]), flap_deg=np.array([0.0]), alpha_deg=POLAR_ANGLES_DEG,
                       Cl=cl3, Cd=np.zeros_like(cl3), Cm=cm3, h_over_c=heights)
    return free, table


@pytest.mark.slow
def test_ground_identity_polar_residual_does_not_grow_with_the_mesh():
    """An identity polar with a height axis gives the free-air mesh residual, no more.

    The 3DP section reference is the cut polygon of the mesh, so even in
    free air an identity polar leaves a first-order mesh residual of the
    coupled CL. In ground effect (h/c 0.5) this residual must fall with
    refinement and stay near the free-air residual of the same mesh. The
    line from the onsets 0 and 1 deg gave +0.3, -1.0 and -1.9 % at 10 deg
    on these meshes, against +4.2, +3.2 and +2.5 % in free air. The
    tolerance 0.02 on the difference to free air comes from the data: the
    largest difference with the tangent of the pitch curve was 0.012, and
    with the inversion of the full pitch curve it is 0.019 (4 x 8 panels,
    10 deg: ground +6.0 %, free air +4.2 %). The tangent error at the tips
    partly cancelled the mesh residual; the full curve removes that error.
    """
    free_polar, table = _identity_polars()
    alphas = (2.0, 6.0, 10.0)
    residual = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for n_span, n_chord in ((4, 8), (6, 12), (8, 16)):
            settings = SolverSettings(solver_type="panel3d", n_panels=n_span, n_chord=n_chord,
                                      spacing="uniform", panel_backend="numpy", panel_cache=False,
                                      panel_ground_solver="direct")
            for h, polar in ((None, free_polar), (0.5, table)):
                conds = [FlightCondition(V_inf=30.0, alpha=np.radians(a), h=h) for a in alphas]
                visc = solve_viscous(_wing(), conds, settings, polar, model="panel3d", tol=1e-10)
                inv = make_solver("panel3d").solve(_wing(), conds, settings)
                for a, rv, ri in zip(alphas, visc, inv):
                    assert rv.converged
                    residual[n_span, h, a] = rv.totals.CL / ri.totals.CL - 1.0
    for a in alphas:
        ground = [residual[n, 0.5, a] for n in (4, 6, 8)]
        free = [residual[n, None, a] for n in (4, 6, 8)]
        print(f"alpha {a:g} deg: ground {np.round(100 * np.array(ground), 3)} %, "
              f"free {np.round(100 * np.array(free), 3)} %")
        assert 0.0 < ground[2] < ground[1] < ground[0]
        np.testing.assert_allclose(ground, free, rtol=0, atol=0.02)
