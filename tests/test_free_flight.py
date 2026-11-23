# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for the public free-body flight interface :func:`ventorum.free_flight`.

The public function must give the same histories as the private free-body
coupling on the same lattice and ground plane, to round-off, in free air and
in ground effect. The tests also check the state record that the callables
get, the JSON output, the storage options, the notes and the trust record,
the ground-contact stop and the input checks. The physics of the coupling is
verified in ``tests/test_unsteady_freeflight.py``.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

import ventorum as vt
from ventorum.aero.system import make_ground_plane
from ventorum.core.errors import InputError
from ventorum.geometry.lattice import build_lattice
from ventorum.unsteady.freeflight import _solve_free_flight
from ventorum.unsteady.gusts import one_minus_cosine


def _wing_tail(*, ref_x: float = -0.15) -> vt.Aircraft:
    """Build a small stable wing-tail aircraft with a tail elevator."""
    wing = vt.LiftingSurface(
        name="wing",
        semi_span=3.0,
        n_panels=4,
        sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=1.0)],
    )
    tail = vt.LiftingSurface(
        name="tail",
        semi_span=1.0,
        n_panels=4,
        position=np.array([1.8, 0.0, 0.0]),
        controls=[vt.ControlSurface(name="elevator", eta_start=0.0, eta_end=1.0, hinge_x_c=0.5, deflection=0.0)],
        sections=[vt.WingSection(y_frac=0.0, chord=0.5), vt.WingSection(y_frac=1.0, chord=0.5)],
    )
    ac = vt.Aircraft(surfaces=[wing, tail], ref_point=np.array([ref_x, 0.0, 0.0]))
    ac.compute_reference_values()
    return ac


_SETTINGS = vt.SolverSettings(solver_type="vlm", n_panels=4, n_chord=4, chord_spacing="uniform")
_MASS = 400.0
_INERTIA = np.diag([80.0, 120.0, 180.0])


def _private_run(ac, cond, n_steps, **kw):
    """Run the private free-body coupling on the lattice and ground plane of the public function."""
    lat = build_lattice(ac, _SETTINGS, collocation="vlm", n_chord=4, chord_spacing="uniform")
    ground = (
        None
        if cond.h is None
        else make_ground_plane(lat, cond.h, cond.alpha, cond.beta, ref_point=ac.ref_point, height_ref="ref")
    )
    return _solve_free_flight(
        lat,
        cond,
        S_ref=ac.S_ref,
        b_ref=ac.b_ref,
        c_ref=ac.c_ref,
        ref_point=ac.ref_point,
        n_steps=n_steps,
        ground=ground,
        **kw,
    )


@pytest.mark.parametrize("h", [None, 0.5])
def test_public_free_flight_equals_private_coupling(h):
    """free_flight equals the private coupling to round-off, with state feedback, thrust and a gust."""
    ac = _wing_tail()
    cond = vt.FlightCondition(V_inf=25.0, rho=1.225, alpha=math.radians(2.0), h=h)
    seen_states = []

    def elevator(t, state):
        seen_states.append(state)
        return -0.05 * state.pitch_dot

    def thrust(t, state):
        return np.array([-40.0, 0.0, 0.0]), np.array([0.0, 5.0 * state.pitch, 0.0])

    kw = {
        "mass": _MASS,
        "inertia": _INERTIA,
        "cg": np.array([-0.1, 0.0, 0.0]),
        "dofs": ("z", "pitch"),
        "initial_state": {"pitch_dot": 2.0e-3, "z_dot": -0.01},
        "motion": {"x": lambda t: 0.002 * math.sin(10.0 * t)},
        "controls": {"elevator": elevator},
        "thrust": thrust,
        "thrust_point": np.array([0.3, 0.0, 0.05]),
        "gust": one_minus_cosine(amplitude=0.5, length=4.0, direction="vertical", x_front=0.0),
    }
    res = vt.free_flight(ac, cond, _SETTINGS, n_steps=12, **kw)
    assert isinstance(res, vt.FreeFlightResult)
    assert seen_states and all(isinstance(s, vt.FlightState) for s in seen_states)

    ref = _private_run(ac, cond, 12, **kw)
    for name in (
        "time", "x", "z", "pitch", "z_dot", "pitch_dot", "CL", "CD", "Cm", "strip_cl",
        "thrust_force", "thrust_moment", "aero_force_cg", "aero_moment_cg", "energy", "work",
        "angular_momentum", "initial_force_imbalance", "initial_moment_imbalance",
        "subiterations", "residuals", "gust_velocity",
    ):
        np.testing.assert_allclose(getattr(res, name), getattr(ref, name), rtol=0.0, atol=1e-13, err_msg=name)
    np.testing.assert_allclose(res.controls["elevator"], ref.controls["elevator"], rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(res.s, 2.0 * cond.V_inf * res.time / ac.c_ref, rtol=1e-15, atol=0.0)
    assert res.dofs == ("z", "pitch")
    assert res.in_ground_effect is (h is not None)
    np.testing.assert_allclose(res.cg, [-0.1, 0.0, 0.0])
    assert res.mass == _MASS
    np.testing.assert_allclose(res.inertia, _INERTIA)


def test_to_dict_is_strict_json_and_store_options():
    """to_dict is strict JSON; 'strengths' adds the final ring strengths and 'loads' leaves them out."""
    ac = _wing_tail()
    cond = vt.FlightCondition(V_inf=25.0, rho=1.225, alpha=0.0)
    common = {"mass": _MASS, "inertia": _INERTIA, "dofs": ("z", "pitch"), "n_steps": 3}
    res_l = vt.free_flight(ac, cond, _SETTINGS, **common)
    d_l = json.loads(json.dumps(res_l.to_dict(), allow_nan=False))
    assert res_l.gamma_bound is None and "gamma_bound" not in d_l
    assert d_l["dofs"] == ["z", "pitch"]
    assert d_l["mass_kg"] == pytest.approx(_MASS)
    assert len(d_l["inertia_kg_m2"]) == 3
    assert len(d_l["energy"]["total_J"]) == 4
    assert set(d_l["initial_imbalance"]) == {"force_N", "moment_Nm", "force_body_N", "moment_body_Nm"}
    assert len(d_l["coupling"]["subiterations"]) == 4
    assert d_l["ground_contact"] is False

    res_s = vt.free_flight(ac, cond, _SETTINGS, store="strengths", **common)
    d_s = json.loads(json.dumps(res_s.to_dict(), allow_nan=False))
    assert res_s.gamma_bound is not None
    assert np.asarray(d_s["gamma_wake"]).shape == (3, res_s.n_strips)
    np.testing.assert_allclose(res_s.CL, res_l.CL, rtol=0.0, atol=0.0)


def test_t_end_notes_trust_and_trim_from_initial_imbalance():
    """t_end gives the step count, the notes and trust are set, and minus the imbalance trims the start."""
    ac = _wing_tail()
    cond = vt.FlightCondition(V_inf=25.0, rho=1.225, alpha=math.radians(1.0))
    probe = vt.free_flight(ac, cond, _SETTINGS, mass=_MASS, inertia=_INERTIA, dofs=(), n_steps=1)
    assert np.linalg.norm(probe.initial_force_imbalance) > 1.0  # weight is not balanced
    f_b = -probe.initial_force_imbalance_body
    m_b = -probe.initial_moment_imbalance_body

    t_end = 5.5 * probe.dt_default
    res = vt.free_flight(
        ac, cond, _SETTINGS, mass=_MASS, inertia=_INERTIA, dofs=("z", "pitch"), t_end=t_end,
        thrust=lambda t: (f_b, m_b),
    )
    assert res.n_steps == 6
    assert np.max(np.abs(res.initial_force_imbalance)) <= 1e-9
    assert np.max(np.abs(res.initial_moment_imbalance)) <= 1e-9
    # The march starts impulsively (no wake at t = 0), so the t = 0 trim does not trim the
    # steady state: the lift grows while the starting wake forms and the body moves. The
    # untrimmed body falls under its weight, much faster.
    untrimmed = vt.free_flight(
        ac, cond, _SETTINGS, mass=_MASS, inertia=_INERTIA, dofs=("z", "pitch"), t_end=t_end,
    )
    assert untrimmed.z[-1] < 0.0
    assert np.max(np.abs(res.z)) < 0.1 * abs(float(untrimmed.z[-1]))

    assert any("CPU" in n for n in res.notes)
    assert any("n_chord=4" in n for n in res.notes)
    assert any("inertial frame" in n for n in res.notes)
    assert res.trust is not None
    assert res.device == "cpu" and res.precision == "float64"
    assert res.base is not None


def test_ground_contact_stops_the_public_run():
    """A fast free descent at h/c = 0.5 stops with the ground-contact note and the valid steps."""
    wing = vt.LiftingSurface(
        name="wing",
        semi_span=3.0,
        n_panels=4,
        sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=1.0)],
    )
    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=0.0, h=0.5)
    res = vt.free_flight(
        wing, cond, _SETTINGS, mass=250.0, inertia=np.diag([50.0, 75.0, 100.0]), dofs=("z", "pitch"),
        initial_state={"z_dot": -10.0}, n_steps=15,
    )
    assert res.ground_contact is True
    assert res.ground_contact_step is not None and res.n_steps < 15
    assert res.time.shape == (res.n_steps + 1,)
    assert any("ground" in n.lower() and f"step {res.ground_contact_step}" in n.lower() for n in res.notes)
    json.dumps(res.to_dict(), allow_nan=False)


def test_public_input_checks():
    """Bad inputs raise InputError before or during the run."""
    ac = _wing_tail()
    cond = vt.FlightCondition(V_inf=25.0, rho=1.225, alpha=0.0)
    ok = {"mass": _MASS, "inertia": _INERTIA, "n_steps": 2}
    bad = [
        {"mass": _MASS, "inertia": _INERTIA},  # neither t_end nor n_steps
        {**ok, "t_end": 0.1},  # both
        {**ok, "store": "cp"},
        {**ok, "dt": -1.0},
        {**ok, "mass": 0.0},
        {**ok, "inertia": np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])},
        {**ok, "inertia": -np.eye(3)},
        {**ok, "dofs": ("heave",)},
        {**ok, "dofs": ("z",), "initial_state": {"pitch": 0.1}},
        {**ok, "dofs": ("z",), "motion": {"z": lambda t: 0.0}},
        {**ok, "thrust": lambda t: (np.array([np.nan, 0.0, 0.0]), np.zeros(3))},
        {**ok, "cg": [0.0, 0.0]},
        {**ok, "dofs": ("pitch",), "initial_state": {"pitch": math.radians(89.5)}},
    ]
    for kw in bad:
        with pytest.raises(InputError):
            vt.free_flight(ac, cond, _SETTINGS, **kw)
    with pytest.raises(InputError):
        vt.free_flight(ac, vt.FlightCondition(V_inf=120.0, rho=1.225, alpha=0.0), _SETTINGS, **ok)
    with pytest.raises(InputError):
        vt.free_flight(ac, cond, vt.SolverSettings(solver_type="linear", n_panels=4), **ok)


def test_public_names_are_exported():
    """The public names are in ventorum and ventorum.unsteady."""
    for name in ("free_flight", "FreeFlightResult", "FlightState"):
        assert hasattr(vt, name)
        assert name in vt.unsteady.__all__
