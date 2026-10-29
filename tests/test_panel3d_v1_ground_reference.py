"""Section reference of the thick-surface (3DP) alpha correction.

The effective angle of a strip is the angle at which the same-mesh 2D
lift curve of its cut section has the strip lift: the pitch curve
``Cl(alpha; h)`` in ground effect, the free-air curve otherwise. A
bracketed root finder on the curve that the tangent comes from solves
it, with the inversion of the tangent as the start value. The target
lift is the polar lift minus the converged inviscid 2D lift of the same
section plus the same-mesh 2D lift (defect correction, as in the
vortex-lattice coupling), so that the converged inviscid 2D polar is the
identity on any mesh.

The tests check the inversion against a brute-force scan of the pitch
curve, the converged 2D reference against a finer one, the analytic
Jacobian of the coupled residual, that the converged identity polar gives
a CL correction below 0.1 % in ground effect (72, 264 and 1056 panels)
and in free air (8 and 40 chordwise panels), and that a polar equal to
the pitch curve of the same cut section now gets the mesh-error
correction.
"""

import functools
import warnings

import numpy as np
import pytest

from ventorum.airfoil import airfoil
from ventorum.core.datatypes import (
    Aircraft, FlightCondition, LiftingSurface, LinearAirfoil, SolverSettings, WingSection,
)
from ventorum.panel3d.mesh import _close_trailing_edge, _resample_section, _target_x
from ventorum.solvers import panel3d
from ventorum.solvers.factory import make_solver
from ventorum.aero.polar_table import PolarTable
from ventorum.core.datatypes import TabulatedAirfoil
from ventorum.panel2d.solver import solve_inviscid
from ventorum.solvers.panel3d import (
    SECTION_REFERENCE_PANELS,
    _ConvergedSectionCurve,
    _GroundPitchCurve,
    _morino_section_ground_line,
)
from ventorum.viscous.alpha_correction import solve_viscous


def _closed_section(n_points):
    """Return a NACA 0012 polygon with a closed trailing edge, as on the panel mesh."""
    x = _target_x(n_points, "cosine")
    yu, yl = _resample_section(airfoil("0012").coords, x)
    yu, yl = _close_trailing_edge(x, yu, yl)
    return np.vstack([np.column_stack([x[::-1], yu[::-1]]), np.column_stack([x[1:], yl[1:]])])


def _ground_line(height, attitude):
    """Return the section ground line at the attitude [rad] and quarter-chord height [chord]."""
    normal = np.array([-np.sin(attitude), np.cos(attitude)])
    return normal, -0.25 * np.sin(attitude) - height


def _scan_root(curve, cl, lower, upper):
    """Return the attitude [rad] with Cl = cl from a brute-force scan of the pitch curve.

    A scan in steps of 0.05 deg finds the first sign change. A second scan
    in steps of 2.5e-4 deg inside that interval and a linear interpolation
    in the last step give the root. The interpolation error is of the
    order of the curvature times the square of the step, far below 1e-6 deg.
    """
    coarse = np.arange(lower, upper, np.radians(0.05))
    values = np.array([curve.lift(a) for a in coarse]) - cl
    k = int(np.flatnonzero(np.sign(values[:-1]) != np.sign(values[1:]))[0])
    fine = np.linspace(coarse[k], coarse[k + 1], 201)
    values = np.array([curve.lift(a) for a in fine]) - cl
    j = int(np.flatnonzero(np.sign(values[:-1]) != np.sign(values[1:]))[0])
    return fine[j] - values[j] * (fine[j + 1] - fine[j]) / (values[j + 1] - values[j])


@pytest.mark.parametrize("n_points", [9, 33])
@pytest.mark.parametrize("height", [0.5, 0.25])
def test_inversion_equals_brute_force_scan(n_points, height):
    """The root finder equals a brute-force scan of the pitch curve within 1e-6 deg.

    The lift values come from attitudes 3 deg below and 4 deg above the
    attitude of the tangent, where the tangent is not exact.
    """
    polygon = _closed_section(n_points)
    attitude = np.radians(8.0)
    slope, zero, _ = _morino_section_ground_line(polygon, _ground_line(height, attitude))
    curve = _GroundPitchCurve(polygon, height)
    for target_deg in (5.0, 12.0):
        cl = curve.lift(np.radians(target_deg)) + 0.01
        start = zero + cl / slope
        root, local, outside = curve.invert(cl, start, slope)
        assert not outside
        scan = _scan_root(_GroundPitchCurve(polygon, height), cl, np.radians(target_deg - 1.0),
                          np.radians(target_deg + 1.0))
        print(f"n {n_points}, h/c {height}, cl {cl:.6f}: tangent {np.degrees(start):.6f} deg, "
              f"root {np.degrees(root):.9f} deg, scan {np.degrees(scan):.9f} deg")
        assert abs(np.degrees(root - scan)) < 1e-6
        assert abs(np.degrees(start - root)) > 1e-3
        assert local == pytest.approx(
            (curve.lift(root + 1e-4) - curve.lift(root - 1e-4)) / 2e-4, rel=1e-6)


def test_inversion_outside_the_clear_attitudes_is_flagged():
    """A lift beyond the attitude of ground contact gives the tangent continuation and the flag."""
    polygon = _closed_section(17)
    height = 0.1
    curve = _GroundPitchCurve(polygon, height)
    attitude = np.radians(2.0)
    slope, zero, _ = _morino_section_ground_line(polygon, _ground_line(height, attitude))
    root, local, outside = curve.invert(50.0, zero + 50.0 / slope, slope)
    assert outside
    assert np.isfinite(root) and local > 0
    # The trailing edge touches the ground near asin(h / 0.75).
    assert root > np.arcsin(height / 0.75) - np.radians(1.0)
    # A lift inside the range is not flagged.
    cl = curve.lift(np.radians(4.0))
    root, _, outside = curve.invert(cl, zero + cl / slope, slope)
    assert not outside
    assert root == pytest.approx(np.radians(4.0), abs=1e-12)


def _wing():
    secs = [WingSection(0.0, 1.0, x_le=-0.25, shape="0012"), WingSection(1.0, 1.0, x_le=-0.25, shape="0012")]
    return Aircraft(surfaces=[LiftingSurface(semi_span=4.0, sections=secs, spacing="uniform")])


def _settings(n_span, n_chord):
    return SolverSettings(solver_type="panel3d", n_panels=n_span, n_chord=n_chord, spacing="uniform",
                          panel_backend="numpy", panel_cache=False, panel_ground_solver="direct")


def _cut_polygon(settings, monkeypatch):
    """Return the cut section polygon [chord] of the panel mesh (all strips have the same one)."""
    captured = []
    original = panel3d._morino_section_line

    def capture(coords, ground=None):
        captured.append(np.array(coords))
        return original(coords, ground)

    with monkeypatch.context() as patch:
        patch.setattr(panel3d, "_morino_section_line", capture)
        solve_viscous(_wing(), FlightCondition(V_inf=30.0, alpha=0.1), settings,
                      LinearAirfoil(a0=6.0), model="panel3d")
    assert len({c.tobytes() for c in captured}) == 1
    return captured[0]


class _PitchCurvePolar:
    """Section polar that is the exact pitch curve of a polygon (no table, no interpolation)."""

    name = "pitch curve"

    def __init__(self, curve):
        self.curve = curve
        self.alpha = np.radians(np.arange(-15.0, 25.01, 0.5))

    def _map(self, func, alpha):
        return np.vectorize(func, otypes=[float])(np.asarray(alpha, dtype=float))

    def Cl(self, alpha):
        return self._map(self.curve.lift, alpha)

    def Cl_slope(self, alpha):
        return self._map(self.curve.slope, alpha)

    def Cd(self, alpha):
        return np.zeros_like(np.asarray(alpha, dtype=float))

    def Cm(self, alpha):
        return np.zeros_like(np.asarray(alpha, dtype=float))


class _PitchCurveSource:
    """Polar source with ground support that gives the pitch curve of the cut polygon."""

    supports_ground = True

    def __init__(self, polygon):
        self.polygon = polygon
        self.heights = []

    def polar(self, *, section, Re, alphas_deg, flap=None, height=None):
        self.heights.append(height)
        return _PitchCurvePolar(_GroundPitchCurve(self.polygon, height))


def _keep_exact_polar(monkeypatch):
    """Do not tabulate a strip polar that blends one polar with itself.

    The root and the tip section of the wing are the same, so both give the
    same polar object. The coupling then reads the exact pitch curve, not a
    table of it.
    """
    from ventorum.viscous import alpha_correction as module

    original = module.blend_tabulated
    monkeypatch.setattr(module, "blend_tabulated",
                        lambda first, second, weight: first if first is second else original(first, second, weight))


#: Angles [deg] of the converged identity polars.
POLAR_ANGLES_DEG = np.arange(-14.0, 14.01, 0.5)


def _converged_curve(height, angles_deg):
    """Return Cl [-] of NACA 0012 (100 points) pitched about its quarter chord, from Ventorum-2D."""
    result = solve_inviscid(_closed_section(100), angles_deg, height=height, pivot=(0.25, 0.0))
    return np.asarray(result.Cl, dtype=float).ravel()


def _free_identity_polar():
    """Return the converged inviscid 2D polar in free air."""
    cl = _converged_curve(None, POLAR_ANGLES_DEG)
    return TabulatedAirfoil(alpha=np.radians(POLAR_ANGLES_DEG), Cl_data=cl, Cd_data=np.zeros_like(cl))


@functools.lru_cache(maxsize=1)
def _ground_identity_polar():
    """Return the converged inviscid 2D pitch curves as a table with a height axis."""
    heights = np.array([0.4, 0.5, 0.7])
    cl3 = np.array([[[_converged_curve(h, POLAR_ANGLES_DEG) for h in heights]]] * 2)
    table = PolarTable(Re=np.array([1e5, 1e8]), flap_deg=np.array([0.0]), alpha_deg=POLAR_ANGLES_DEG,
                       Cl=cl3, Cd=np.zeros_like(cl3), Cm=np.zeros_like(cl3), h_over_c=heights)
    return table


@pytest.mark.parametrize("height", [None, 0.5])
def test_converged_reference_resolution(height):
    """128 panels per side give Cl within 1e-4 of 256; the interpolation is within 1e-5.

    The reference is Ventorum-2D (second-order linear-vortex panels) on
    the cut section with ``SECTION_REFERENCE_PANELS`` cosine-spaced panels
    per side. The cubic Hermite interpolation between the 0.5 deg knots
    is checked against a direct solve between two knots.
    """
    assert SECTION_REFERENCE_PANELS == 128
    angle = np.radians(10.25)
    coarse = _ConvergedSectionCurve(_closed_section(SECTION_REFERENCE_PANELS + 1), height)
    fine = _ConvergedSectionCurve(_closed_section(2 * SECTION_REFERENCE_PANELS + 1), height)
    value = coarse.evaluate(angle)[0]
    direct = float(np.ravel(solve_inviscid(coarse.polygon, [10.25], height=height, pivot=(0.25, 0.0),
                                           chord_ref=1.0, moment_ref=(0.25, 0.0)).Cl)[0])
    print(f"h/c {height}: Cl 128 {value:.7f}, 256 {fine.evaluate(angle)[0]:.7f}, direct {direct:.7f}")
    assert abs(value - fine.evaluate(angle)[0]) < 1e-4
    assert abs(value - direct) < 1e-5
    step = 1e-6
    slope = (coarse.evaluate(angle + step)[0] - coarse.evaluate(angle - step)[0]) / (2 * step)
    assert coarse.evaluate(angle)[1] == pytest.approx(slope, rel=1e-7)


def _identity_case(mesh, height):
    settings = _settings(*mesh)
    condition = FlightCondition(V_inf=30.0, alpha=np.radians(10.0), h=height)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        polar = _free_identity_polar() if height is None else _ground_identity_polar()
        result = solve_viscous(_wing(), condition, settings, polar,
                               model="panel3d", tol=1e-12)
        inviscid = make_solver("panel3d").solve(_wing(), condition, settings)
    assert result.converged
    delta = np.degrees(np.max(np.abs(result.details["delta"])))
    change = result.totals.CL / inviscid.totals.CL - 1.0
    print(f"mesh {mesh}, h/c {height}: max |delta| {delta:.3e} deg, CL change {100 * change:+.4f} %")
    return change


@pytest.mark.slow
@pytest.mark.parametrize("mesh", [(4, 8), (8, 16), (16, 32)])
def test_converged_identity_polar_in_ground_effect(mesh):
    """The converged inviscid 2D polar gives a CL correction below 0.1 % in ground effect.

    AR 8 NACA 0012 at h/c 0.5 and 10 deg; 72, 264 and 1056 panels. With
    the same-mesh section as the only reference, this polar gave +6.0, +3.6
    and +1.7 % (the first-order mesh error of the cut section).
    """
    assert abs(_identity_case(mesh, 0.5)) < 1e-3


@pytest.mark.parametrize("mesh", [(4, 8), pytest.param((4, 40), marks=pytest.mark.slow)])
def test_converged_identity_polar_in_free_air(mesh):
    """The converged inviscid 2D polar gives a CL correction below 0.1 % in free air (nc 8 and 40).

    With the same-mesh section line as the only reference, this polar gave
    +4.2 % at nc 8 (10 deg).
    """
    assert abs(_identity_case(mesh, None)) < 1e-3


@pytest.mark.parametrize("mesh", [(4, 8), pytest.param((8, 16), marks=pytest.mark.slow)])
def test_same_mesh_pitch_curve_polar_gets_the_mesh_error_correction(mesh, monkeypatch):
    """The pitch curve of the cut section as the polar gets the mesh-error correction.

    Before the converged 2D reference, this polar was the identity (zero
    correction to round-off). Now the coupling reads it relative to the
    converged section, whose lift is higher than the lift of the coarse cut
    section; the coupled CL is lower than the inviscid CL. The target lift
    is ``cl_mesh(alpha_v) - cl_conv(alpha_v) + cl_mesh(alpha_v)``.
    """
    settings = _settings(*mesh)
    source = _PitchCurveSource(_cut_polygon(settings, monkeypatch))
    _keep_exact_polar(monkeypatch)
    condition = FlightCondition(V_inf=30.0, alpha=np.radians(10.0), h=0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = solve_viscous(_wing(), condition, settings, source, model="panel3d", tol=1e-12)
        inviscid = make_solver("panel3d").solve(_wing(), condition, settings)
    assert result.converged
    assert source.heights and all(h == pytest.approx(0.5) for h in source.heights)
    details = result.details
    np.testing.assert_allclose(details["cl_polar"], details["cl_section_mesh"], rtol=0, atol=1e-12)
    cl = np.concatenate([sw.Cl for sw in result.spanwise])
    np.testing.assert_allclose(cl, 2 * details["cl_section_mesh"] - details["cl_section_converged"],
                               rtol=0, atol=1e-10)
    change = result.totals.CL / inviscid.totals.CL - 1.0
    print(f"mesh {mesh}: CL change {100 * change:+.3f} %")
    assert change < -1e-3


def test_ground_residual_jacobian_with_the_pitch_curve(monkeypatch):
    """The analytic Jacobian with the inverted pitch curve equals a central difference."""
    from ventorum.viscous import alpha_correction as module

    original = module._panel_residual
    checked = False

    def check(x, *args):
        nonlocal checked
        state = original(x, *args)
        if not checked and args[-1] is not None:
            trial = x + np.linspace(-0.01, 0.02, len(x))
            jac = original(trial, *args)["jac"]
            fd = np.empty_like(jac)
            for k in range(len(x)):
                step = np.zeros_like(x)
                step[k] = 1e-5
                fd[:, k] = (original(trial + step, *args)["r"] - original(trial - step, *args)["r"]) / 2e-5
            np.testing.assert_allclose(jac, fd, rtol=1e-6, atol=1e-6 * np.max(np.abs(jac)))
            checked = True
        return state

    settings = _settings(3, 8)
    source = _PitchCurveSource(_cut_polygon(settings, monkeypatch))
    _keep_exact_polar(monkeypatch)
    monkeypatch.setattr(module, "_panel_residual", check)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = solve_viscous(_wing(), FlightCondition(V_inf=30.0, alpha=np.radians(6.0), h=0.4), settings,
                               source, model="panel3d", tol=1e-10)
    assert checked and result.converged
