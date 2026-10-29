# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Continuity of flapped geometry and results at zero deflection.

The hinge treatments "merge" and "repanel" of the plain-flap section
(:func:`ventorum.airfoil.flap.deflect_plain_flap`) and the chordwise
controls of the vortex lattice put a node at the hinge also at a zero
deflection. The node set at zero deflection is then the node set of a
small deflection, and the loads, the hinge moment and their derivatives
are continuous in the deflection.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import ventorum as vt
from ventorum import gpu
from ventorum.airfoil import AirfoilShape, airfoil
from ventorum.airfoil.flap import deflect_plain_flap, flap_hinge_point
from ventorum.geometry import lattice as mesh_mod
from ventorum.panel2d.solver import solve_inviscid

#: Small deflection of the continuity checks [deg].
_SMALL = 1e-6
#: Relative tolerance of the continuity checks [-].
_RTOL = 1e-8


def _section_hinge_moment(coords: np.ndarray, cp: np.ndarray, hinge: np.ndarray, x_hinge: float) -> float:
    """Return the hinge-moment coefficient [-] of the panels aft of the hinge.

    The pressure of a panel is the mean of the node values. The force on
    a panel is ``-Cp n ds`` (outward normal ``n``), and the moment is
    about the hinge point, nose up positive, per unit chord squared.
    """
    p0 = coords[:-1]
    p1 = coords[1:]
    mid = 0.5 * (p0 + p1)
    d = p1 - p0
    # Selig order runs counterclockwise round the body (x right, y up), so
    # (dy, -dx) is the outward normal times the panel length.
    n_ds = np.column_stack([d[:, 1], -d[:, 0]])
    cp_panel = 0.5 * (cp[:-1] + cp[1:])
    force = -cp_panel[:, None] * n_ds
    arm = mid - hinge
    moment = arm[:, 1] * force[:, 0] - arm[:, 0] * force[:, 1]
    aft = mid[:, 0] > x_hinge
    return float(np.sum(moment[aft]))


def _section_loads(base: np.ndarray, x_hinge: float, delta: float, treatment: str):
    """Return the node count and (CL, Cm, Ch) of the flapped section at alpha 0 deg."""
    flapped = AirfoilShape(name="2412", coords=base).with_flap(x_hinge=x_hinge, deflection=delta, hinge_treatment=treatment)
    res = solve_inviscid(flapped, 0.0)
    coords = np.asarray(res.coordinates, dtype=float)
    np.testing.assert_array_equal(coords, flapped.coords)
    hinge = flap_hinge_point(base, x_hinge, 0.5)
    ch = _section_hinge_moment(coords, np.asarray(res.Cp, dtype=float), hinge, x_hinge)
    return flapped.coords.shape[0], np.array([float(res.Cl), float(res.Cm), ch])


@pytest.mark.parametrize("treatment", ["merge", "repanel"])
@pytest.mark.parametrize("name", ["2412", "0012", "4415"])
@pytest.mark.parametrize("x_hinge", [0.7, 0.75])
def test_section_node_set_has_the_hinge_node_at_zero(treatment, name, x_hinge):
    """At zero deflection the node set is the limit of the small deflections."""
    base = airfoil(name, n_points=161).coords
    zero = deflect_plain_flap(base, x_hinge, 0.0, hinge_treatment=treatment)
    one = deflect_plain_flap(base, x_hinge, 1.0, hinge_treatment=treatment)
    assert zero.shape == one.shape
    for sign in (1.0, -1.0):
        small = deflect_plain_flap(base, x_hinge, sign * 1e-9, hinge_treatment=treatment)
        assert small.shape == zero.shape
        np.testing.assert_allclose(small, zero, rtol=0.0, atol=1e-9)
    if treatment == "repanel":
        # "repanel" keeps a node at the hinge station on each surface;
        # "merge" can remove it by its short-panel rule, as at a deflection.
        assert np.sum(np.abs(zero[:, 0] - x_hinge) < 1e-15) == 2


def test_section_none_keeps_the_input_at_zero():
    """The treatment "none" adds no node: zero deflection returns a copy of the input."""
    base = airfoil("2412", n_points=161).coords
    out = deflect_plain_flap(base, 0.7, 0.0, hinge_treatment="none")
    assert out is not base
    np.testing.assert_array_equal(out, base)


@pytest.mark.parametrize("treatment", ["merge", "repanel"])
@pytest.mark.parametrize("x_hinge", [0.7, 0.75])
def test_section_loads_continuous_at_zero(treatment, x_hinge):
    """CL, Cm and the hinge moment at zero are the mean of the values at +/-1e-6 deg."""
    base = airfoil("2412", n_points=161).coords
    n0, c0 = _section_loads(base, x_hinge, 0.0, treatment)
    n1, _ = _section_loads(base, x_hinge, 1.0, treatment)
    np_, cp = _section_loads(base, x_hinge, _SMALL, treatment)
    nm, cm = _section_loads(base, x_hinge, -_SMALL, treatment)
    assert n0 == n1 == np_ == nm
    np.testing.assert_allclose(c0, 0.5 * (cp + cm), rtol=_RTOL, atol=0.0)
    # No kink at zero: the one-sided slopes [1/deg] agree with the central slope.
    h = 1e-3
    _, ch_p = _section_loads(base, x_hinge, h, treatment)
    _, ch_m = _section_loads(base, x_hinge, -h, treatment)
    central = (ch_p - ch_m) / (2.0 * h)
    np.testing.assert_allclose((ch_p - c0) / h, central, rtol=1e-3, atol=0.0)
    np.testing.assert_allclose((c0 - ch_m) / h, central, rtol=1e-3, atol=0.0)


# ---------------------------------------------------------------------------
# Vortex lattice: chordwise control at zero deflection
# ---------------------------------------------------------------------------


@pytest.fixture
def _float64():
    old = gpu._precision
    gpu.set_precision("float64")
    yield
    gpu.set_precision(old)


def _aircraft(delta: float) -> vt.Aircraft:
    """Return a cambered wing with a part-span flap, deflection *delta* [rad]."""
    return vt.Aircraft(surfaces=[vt.LiftingSurface(
        name="wing", semi_span=4.0,
        sections=[vt.WingSection(y_frac=e, chord=1.0, shape="2412") for e in (0.0, 1.0)],
        controls=[vt.ControlSurface("flap", 0.25, 0.75, 0.7, delta)],
    )], ref_point=np.array([0.25, 0.0, 0.0]))


def _settings() -> vt.SolverSettings:
    return vt.SolverSettings(n_chord=10, n_panels=8, chord_spacing="uniform")


_CONDITION = vt.FlightCondition(V_inf=30.0, alpha=0.05)


def _vlm_loads(delta: float) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = vt.VortexLatticeSolver().solve(_aircraft(delta), _CONDITION, _settings())
    ch = [m.Ch for m in vt.hinge_moments(res)]
    return np.array([float(res.totals.CL), float(res.totals.Cm), *ch])


def test_vlm_node_set_at_zero_equals_one_degree():
    """The chordwise control has its hinge nodes at zero deflection, as at 1 deg."""
    zero = mesh_mod.build_lattice(_aircraft(0.0), _settings(), n_chord=10)
    one = mesh_mod.build_lattice(_aircraft(math.radians(1.0)), _settings(), n_chord=10)
    assert zero.a.shape == one.a.shape
    assert zero.a_kink is not None and one.a_kink is not None
    kinked_zero = np.any(zero.a_kink != zero.a_te, axis=1)
    kinked_one = np.any(one.a_kink != one.a_te, axis=1)
    np.testing.assert_array_equal(kinked_zero, kinked_one)
    assert sorted(zero.geom_cache["flap_hinges"]) == sorted(one.geom_cache["flap_hinges"])


@pytest.mark.usefixtures("_float64")
def test_vlm_loads_continuous_at_zero():
    """CL, Cm and the hinge moments at zero are the mean of the values at +/-1e-6 deg."""
    c0 = _vlm_loads(0.0)
    cp = _vlm_loads(math.radians(_SMALL))
    cm = _vlm_loads(-math.radians(_SMALL))
    np.testing.assert_allclose(c0, 0.5 * (cp + cm), rtol=_RTOL, atol=0.0)


@pytest.mark.usefixtures("_float64")
def test_vlm_cl_derivative_at_zero_equals_control_derivative():
    """The finite difference of CL at zero equals the adjoint and the stepped control derivatives."""
    h = math.radians(1e-3)
    c0 = _vlm_loads(0.0)[0]
    cp = _vlm_loads(h)[0]
    cm = _vlm_loads(-h)[0]
    forward = (cp - c0) / h
    backward = (c0 - cm) / h
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        adjoint = float(vt.gradients(_aircraft(0.0), _CONDITION, _settings(), outputs=("CL",),
                                     wrt=("controls",)).jacobian["CL"]["controls"]["flap"])
        stepped = vt.control_derivatives(_aircraft(0.0), _CONDITION, _settings(), step_deg=1e-3,
                                         hinge_alpha=False).derivatives["flap"].CL
    assert forward == pytest.approx(adjoint, rel=1e-5)
    assert backward == pytest.approx(adjoint, rel=1e-5)
    assert stepped == pytest.approx(adjoint, rel=1e-6)
