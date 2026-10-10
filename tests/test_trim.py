# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests of the aircraft trim solver.

Verifies:
1. Longitudinal trim of a wing-tail configuration to target CL with zero Cm.
2. Agreement between nonlinear trim and 2x2 linear prediction from derivatives.
3. Pitch stability elevator trend (higher CL requires more trailing-edge-up elevator).
4. Lateral trim with sideslip (aileron, elevator, and rudder for zero Cl, Cm, Cn).
5. Unreachable targets report limit status without exceptions.
6. Ground effect requires lower angle of attack than free air for the same CL.
7. Linear and nonlinear lifting line solvers trim successfully.
8. Input aircraft geometry and control deflections remain unchanged after trim.
9. Input validation rejects invalid configurations and parameters.
10. Execution occurs on CPU in float64 and restores prior device setting.
11. Serialization via to_dict produces valid JSON without internal objects.

The regression tests of the review check the failure paths: a failed solve
reports the last good point or NaN, a control with no effect gives a singular
Jacobian, only the documented solver failures become a status, an inner solve
that does not converge is not accepted, and the iteration count agrees with
the residual history.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import ventorum as vt
from ventorum import gpu


def _build_wing_tail_aircraft(hinge_x_c: float = 0.7, z_tail: float = 0.0) -> vt.Aircraft:
    """Build a standard two-surface wing-tail aircraft for trim tests."""
    wing = vt.LiftingSurface(
        name="wing",
        semi_span=5.0,
        sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=1.0)],
        n_panels=14,
    )
    tail = vt.LiftingSurface(
        name="tail",
        semi_span=1.5,
        position=np.array([4.0, 0.0, z_tail]),
        sections=[vt.WingSection(y_frac=0.0, chord=0.6), vt.WingSection(y_frac=1.0, chord=0.6)],
        controls=[
            vt.ControlSurface(
                name="elevator",
                eta_start=0.0,
                eta_end=1.0,
                hinge_x_c=hinge_x_c,
                deflection=0.0,
                symmetric=True,
            )
        ],
        n_panels=8,
    )
    return vt.Aircraft(
        name="WingTail",
        surfaces=[wing, tail],
        ref_point=np.array([0.25, 0.0, 0.0]),
    )


def test_longitudinal_trim_wing_tail():
    """Verify longitudinal trim reaches target CL and zero Cm within tolerance."""
    ac = _build_wing_tail_aircraft()
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(2.0))
    st = vt.SolverSettings(solver_type="vlm", n_panels=14, n_chord=4)

    target_cl = 0.5
    res = vt.trim(ac, cond, st, CL_target=target_cl, pitch_control="elevator")

    assert res.status == "trimmed"
    assert res.converged is True
    assert abs(res.CL - target_cl) <= 1.0e-6
    assert abs(res.Cm) <= 1.0e-7
    assert len(res.residual_history) == res.iterations + 1
    assert np.all(np.isfinite(res.jacobian))

    # Independent solve verification
    cond_check = vt.FlightCondition(V_inf=30.0, alpha=np.radians(res.alpha_deg))
    check_sol = vt.analyze(res.aircraft, cond_check, st)
    assert abs(check_sol.totals.CL - target_cl) <= 1.0e-6
    assert abs(check_sol.totals.Cm) <= 1.0e-7


def test_trim_matches_linear_prediction():
    """Verify trim result is within 0.05 deg of linear prediction from derivatives."""
    ac = _build_wing_tail_aircraft()
    cond_base = vt.FlightCondition(V_inf=30.0, alpha=0.0)
    st = vt.SolverSettings(solver_type="vlm", n_panels=14, n_chord=4)
    target_cl = 0.4

    # Central difference step = 0.5 deg
    d_step = np.radians(0.5)

    # Base solve
    sol_0 = vt.analyze(ac, cond_base, st)
    cl_0 = sol_0.totals.CL
    cm_0 = sol_0.totals.Cm

    # Alpha perturbation
    sol_ap = vt.analyze(ac, vt.FlightCondition(V_inf=30.0, alpha=d_step), st)
    sol_am = vt.analyze(ac, vt.FlightCondition(V_inf=30.0, alpha=-d_step), st)
    cl_alpha = (sol_ap.totals.CL - sol_am.totals.CL) / (2.0 * d_step)
    cm_alpha = (sol_ap.totals.Cm - sol_am.totals.Cm) / (2.0 * d_step)

    # Elevator perturbation
    ac_dp = ac.clone()
    ac_dp.set_deflection("elevator", d_step)
    sol_dp = vt.analyze(ac_dp, cond_base, st)

    ac_dm = ac.clone()
    ac_dm.set_deflection("elevator", -d_step)
    sol_dm = vt.analyze(ac_dm, cond_base, st)

    cl_delta = (sol_dp.totals.CL - sol_dm.totals.CL) / (2.0 * d_step)
    cm_delta = (sol_dp.totals.Cm - sol_dm.totals.Cm) / (2.0 * d_step)

    # 2x2 linear system: J * [delta_alpha, delta_elevator]^T = [target_cl - cl_0, -cm_0]^T
    j_mat = np.array([[cl_alpha, cl_delta], [cm_alpha, cm_delta]])
    rhs = np.array([target_cl - cl_0, -cm_0])
    d_lin = np.linalg.solve(j_mat, rhs)

    alpha_lin_deg = float(np.degrees(d_lin[0]))
    delta_lin_deg = float(np.degrees(d_lin[1]))

    # Run trim solver
    res = vt.trim(ac, cond_base, st, CL_target=target_cl, pitch_control="elevator")
    assert res.status == "trimmed"
    assert abs(res.alpha_deg - alpha_lin_deg) <= 0.05
    assert abs(res.deflections_deg["elevator"] - delta_lin_deg) <= 0.05


def test_elevator_trend_of_a_stable_aircraft():
    """Verify pitch-stable aircraft requires more trailing-edge-up elevator for higher CL."""
    ac = _build_wing_tail_aircraft()
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(2.0))
    st = vt.SolverSettings(solver_type="vlm", n_panels=14, n_chord=4)

    # Check that aircraft is pitch-stable: Cm_alpha < 0
    d_step = np.radians(0.5)
    sol_ap = vt.analyze(ac, vt.FlightCondition(V_inf=30.0, alpha=np.radians(2.0) + d_step), st)
    sol_am = vt.analyze(ac, vt.FlightCondition(V_inf=30.0, alpha=np.radians(2.0) - d_step), st)
    cm_alpha = (sol_ap.totals.Cm - sol_am.totals.Cm) / (2.0 * d_step)
    assert cm_alpha < 0.0, f"Configuration must be pitch-stable, got Cm_alpha={cm_alpha}"

    res_03 = vt.trim(ac, cond, st, CL_target=0.3, pitch_control="elevator")
    res_06 = vt.trim(ac, cond, st, CL_target=0.6, pitch_control="elevator")

    assert res_03.status == "trimmed"
    assert res_06.status == "trimmed"
    assert res_06.deflections_deg["elevator"] < res_03.deflections_deg["elevator"]


def test_lateral_trim_with_sideslip():
    """Verify 4-variable lateral trim at sideslip yields zero rolling, pitching, and yawing moments."""
    wing = vt.LiftingSurface(
        name="wing",
        semi_span=5.0,
        sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=1.0)],
        controls=[
            vt.ControlSurface(
                name="aileron",
                eta_start=0.6,
                eta_end=1.0,
                hinge_x_c=0.75,
                deflection=0.0,
                symmetric=False,
            )
        ],
        n_panels=14,
    )
    tail = vt.LiftingSurface(
        name="tail",
        semi_span=1.5,
        position=np.array([4.0, 0.0, 0.0]),
        sections=[vt.WingSection(y_frac=0.0, chord=0.6), vt.WingSection(y_frac=1.0, chord=0.6)],
        controls=[
            vt.ControlSurface(
                name="elevator",
                eta_start=0.0,
                eta_end=1.0,
                hinge_x_c=0.7,
                deflection=0.0,
                symmetric=True,
            )
        ],
        n_panels=8,
    )
    fin = vt.LiftingSurface(
        name="fin",
        semi_span=1.5,
        is_symmetric=False,
        dihedral=np.pi / 2.0,
        position=np.array([4.0, 0.0, 0.0]),
        sections=[vt.WingSection(y_frac=0.0, chord=0.8), vt.WingSection(y_frac=1.0, chord=0.4)],
        controls=[
            vt.ControlSurface(
                name="rudder",
                eta_start=0.0,
                eta_end=1.0,
                hinge_x_c=0.7,
                deflection=0.0,
                symmetric=True,
            )
        ],
        n_panels=8,
    )
    ac = vt.Aircraft(
        name="CompleteAircraft",
        surfaces=[wing, tail, fin],
        ref_point=np.array([0.25, 0.0, 0.0]),
    )

    beta_deg = 3.0
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(2.0), beta=np.radians(beta_deg))
    st = vt.SolverSettings(solver_type="vlm", n_panels=14, n_chord=4)

    target_cl = 0.5
    res = vt.trim(
        ac,
        cond,
        st,
        CL_target=target_cl,
        pitch_control="elevator",
        roll_control="aileron",
        yaw_control="rudder",
    )

    assert res.status == "trimmed"
    assert res.converged is True

    # Independent solve verification
    cond_check = vt.FlightCondition(
        V_inf=30.0,
        alpha=np.radians(res.alpha_deg),
        beta=np.radians(beta_deg),
    )
    check_sol = vt.analyze(res.aircraft, cond_check, st)
    assert abs(check_sol.totals.CL - target_cl) <= 1.0e-6
    assert abs(check_sol.totals.Cm) <= 1.0e-7
    assert abs(check_sol.totals.Cl) <= 1.0e-7
    assert abs(check_sol.totals.Cn) <= 1.0e-7


def test_unreachable_target_reports_limit():
    """Verify unreachable targets terminate with limit status and converged=False."""
    ac = _build_wing_tail_aircraft()
    cond = vt.FlightCondition(V_inf=30.0, alpha=0.0)
    st = vt.SolverSettings(solver_type="vlm", n_panels=12, n_chord=4)

    # 1. Unreachable CL with tight alpha bounds
    res_high_cl = vt.trim(
        ac,
        cond,
        st,
        CL_target=3.0,
        pitch_control="elevator",
        alpha_bounds_deg=(-10.0, 10.0),
    )
    assert res_high_cl.status in ("alpha_limit", "control_limit")
    assert res_high_cl.converged is False
    # Regression: a limit stop counted one iteration too many.
    assert len(res_high_cl.residual_history) == res_high_cl.iterations + 1

    # 2. Ineffective elevator (hinge 0.97) and far-aft reference point
    ac_ineffective = _build_wing_tail_aircraft(hinge_x_c=0.97)
    ac_ineffective.ref_point = np.array([3.0, 0.0, 0.0])

    res_weak = vt.trim(
        ac_ineffective,
        cond,
        st,
        CL_target=0.5,
        pitch_control="elevator",
    )
    assert res_weak.status in ("control_limit", "not_converged")
    assert res_weak.converged is False


def test_ground_effect_trim_needs_less_alpha():
    """Verify ground effect increases lift and reduces trimmed alpha for the same CL."""
    # Put tail high enough (z = 0.5) to avoid ground strike during alpha changes
    ac = _build_wing_tail_aircraft(z_tail=0.5)
    ac.compute_reference_values()
    c_ref = float(ac.c_ref)

    st = vt.SolverSettings(solver_type="vlm", n_panels=14, n_chord=4)
    target_cl = 0.5

    cond_fa = vt.FlightCondition(V_inf=30.0, alpha=np.radians(2.0), h=None)
    cond_ge = vt.FlightCondition(V_inf=30.0, alpha=np.radians(2.0), h=0.5 * c_ref)

    res_fa = vt.trim(ac, cond_fa, st, CL_target=target_cl, pitch_control="elevator")
    res_ge = vt.trim(ac, cond_ge, st, CL_target=target_cl, pitch_control="elevator")

    assert res_fa.status == "trimmed"
    assert res_ge.status == "trimmed"
    assert res_ge.alpha_deg < res_fa.alpha_deg


def test_lifting_line_trims():
    """Verify linear and nonlinear lifting line solvers converge to trimmed state."""
    ac = _build_wing_tail_aircraft()
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(2.0))
    target_cl = 0.4

    # Linear LLT
    st_lin = vt.SolverSettings(solver_type="linear", n_panels=12)
    res_lin = vt.trim(ac, cond, st_lin, CL_target=target_cl, pitch_control="elevator")
    assert res_lin.status == "trimmed"
    assert abs(res_lin.CL - target_cl) <= 1.0e-6
    assert abs(res_lin.Cm) <= 1.0e-7

    # Nonlinear LLT with tight tolerance
    st_nl = vt.SolverSettings(solver_type="nonlinear", n_panels=12, tolerance=1.0e-10)
    res_nl = vt.trim(ac, cond, st_nl, CL_target=target_cl, pitch_control="elevator")
    assert res_nl.status == "trimmed"
    assert abs(res_nl.CL - target_cl) <= 1.0e-6
    assert abs(res_nl.Cm) <= 1.0e-7


def test_input_aircraft_is_not_changed():
    """Verify that trim does not modify the user's input aircraft geometry or deflections."""
    ac = _build_wing_tail_aircraft()
    init_def = 0.05
    ac.set_deflection("elevator", init_def)

    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(2.0))
    st = vt.SolverSettings(solver_type="vlm", n_panels=12, n_chord=4)

    res = vt.trim(ac, cond, st, CL_target=0.5, pitch_control="elevator")
    assert res.status == "trimmed"

    # Input aircraft must remain unchanged
    elev_ctrl = ac.surfaces[1].controls[0]
    assert elev_ctrl.deflection == init_def
    assert res.aircraft is not ac


def test_invalid_input():
    """Verify input validation raises ValueError on illegal inputs."""
    ac = _build_wing_tail_aircraft()
    cond = vt.FlightCondition(V_inf=30.0, alpha=0.0)

    # 1. Unknown control name
    with pytest.raises(ValueError, match=r"Unknown pitch control 'nonexistent'.*\['elevator'\]"):
        vt.trim(ac, cond, CL_target=0.5, pitch_control="nonexistent")

    # 2. Roll control without yaw control
    with pytest.raises(ValueError, match="Both roll_control and yaw_control"):
        vt.trim(ac, cond, CL_target=0.5, pitch_control="elevator", roll_control="elevator")

    # 3. NaN CL_target
    with pytest.raises(ValueError, match="CL_target must be a finite number"):
        vt.trim(ac, cond, CL_target=float("nan"), pitch_control="elevator")

    # 4. Inverted alpha bounds
    with pytest.raises(ValueError, match="alpha_bounds_deg must be a tuple"):
        vt.trim(ac, cond, CL_target=0.5, pitch_control="elevator", alpha_bounds_deg=(15.0, 5.0))

    # 5. max_iterations < 1
    with pytest.raises(ValueError, match="max_iterations must be at least 1"):
        vt.trim(ac, cond, CL_target=0.5, pitch_control="elevator", max_iterations=0)


def test_trim_runs_on_cpu_float64(monkeypatch):
    """Verify trim runs on CPU float64 and restores prior device setting."""
    # Regression: the old test only read the keys that trim writes itself.
    # A spy on analyze now records the global device at each solve.
    real_analyze = vt.analyze
    devices: list[str] = []

    def spy(*args, **kwargs):
        devices.append(gpu.get_device())
        return real_analyze(*args, **kwargs)

    monkeypatch.setattr(vt, "analyze", spy)
    old = gpu.get_device()
    gpu.set_device("auto")
    try:
        ac = _build_wing_tail_aircraft()
        cond = vt.FlightCondition(V_inf=30.0, alpha=0.0)
        st = vt.SolverSettings(solver_type="vlm", n_panels=12, n_chord=4)

        res = vt.trim(ac, cond, st, CL_target=0.4, pitch_control="elevator")
        assert res.status == "trimmed"

        # Every solve of the trim ran with the global device set to "cpu"
        assert len(devices) > 0
        assert all(d == "cpu" for d in devices)

        # Device keys stored in result details
        assert res.result.details.get("device") == "cpu"
        assert res.result.details.get("precision") == "float64"

        # Global device restored
        assert gpu.get_device() == "auto"
    finally:
        gpu.set_device(old)


def test_device_is_restored_after_keyboard_interrupt(monkeypatch):
    """Verify a KeyboardInterrupt in a solve propagates and the device is restored."""

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(vt, "analyze", interrupt)
    old = gpu.get_device()
    gpu.set_device("auto")
    try:
        before = gpu.get_device()
        ac = _build_wing_tail_aircraft()
        cond = vt.FlightCondition(V_inf=30.0, alpha=0.0)
        with pytest.raises(KeyboardInterrupt):
            vt.trim(ac, cond, CL_target=0.4, pitch_control="elevator")
        assert gpu.get_device() == before
    finally:
        gpu.set_device(old)


def test_to_dict_is_json():
    """Verify to_dict returns a JSON-serializable dictionary without internal objects."""
    ac = _build_wing_tail_aircraft()
    cond = vt.FlightCondition(V_inf=30.0, alpha=0.0)
    st = vt.SolverSettings(solver_type="vlm", n_panels=12, n_chord=4)

    res = vt.trim(ac, cond, st, CL_target=0.4, pitch_control="elevator")
    data = res.to_dict()

    assert "result" not in data
    assert "aircraft" not in data
    assert data["status"] == "trimmed"
    assert data["converged"] is True
    assert isinstance(data["alpha_deg"], float)
    assert isinstance(data["deflections_deg"], dict)
    assert isinstance(data["jacobian"], list)
    assert isinstance(data["residual_history"], list)

    serialized = json.dumps(data)
    deserialized = json.loads(serialized)
    assert deserialized["status"] == "trimmed"


def _build_wing_tail_fin_aircraft() -> vt.Aircraft:
    """Build the wing-tail aircraft with a fin and a rudder."""
    ac = _build_wing_tail_aircraft()
    fin = vt.LiftingSurface(
        name="fin",
        semi_span=1.5,
        is_symmetric=False,
        dihedral=np.pi / 2.0,
        position=np.array([4.0, 0.0, 0.0]),
        sections=[vt.WingSection(y_frac=0.0, chord=0.8), vt.WingSection(y_frac=1.0, chord=0.4)],
        controls=[
            vt.ControlSurface(
                name="rudder",
                eta_start=0.0,
                eta_end=1.0,
                hinge_x_c=0.7,
                deflection=0.0,
                symmetric=True,
            )
        ],
        n_panels=8,
    )
    ac.surfaces.append(fin)
    return ac


def test_ground_strike_during_iteration_reports_last_good_point():
    """Verify a ground strike during the iteration reports the last good point, not zeros."""
    # Regression: the result gave CL = 0, Cm = 0 and a default SolverResult with
    # converged=True after a failed solve.
    ac = _build_wing_tail_aircraft(z_tail=0.0)
    ac.compute_reference_values()
    c_ref = float(ac.c_ref)
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(2.0), h=0.3 * c_ref)
    st = vt.SolverSettings(solver_type="vlm", n_panels=14, n_chord=4)

    res = vt.trim(ac, cond, st, CL_target=0.8, pitch_control="elevator")

    assert res.status == "not_converged"
    assert res.converged is False
    assert res.iterations >= 1
    assert len(res.residual_history) == res.iterations + 1
    assert np.isnan(res.residual_history[-1])
    assert np.isfinite(res.residual_history[-2])
    assert any("ground" in n for n in res.notes)
    assert any("last point with a good solve" in n for n in res.notes)

    # The reported values are those of the last good solve
    assert res.result is not None
    assert res.result.converged is True
    assert res.CL == res.result.totals.CL
    assert res.Cm == res.result.totals.Cm
    assert res.CL != 0.0
    assert np.all(np.isfinite(res.jacobian))

    # An independent solve at the reported point gives the reported values
    cond_check = vt.FlightCondition(V_inf=30.0, alpha=np.radians(res.alpha_deg), h=0.3 * c_ref)
    check = vt.analyze(res.aircraft, cond_check, st)
    assert check.totals.CL == pytest.approx(res.CL, abs=1.0e-9)
    assert check.totals.Cm == pytest.approx(res.Cm, abs=1.0e-9)


def test_ground_strike_at_start_point_reports_nan():
    """Verify a failed solve at the start point gives NaN coefficients and no result."""
    ac = _build_wing_tail_aircraft(z_tail=0.0)
    ac.compute_reference_values()
    c_ref = float(ac.c_ref)
    # At 15 deg the tail trailing edge is below the ground plane.
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(15.0), h=0.3 * c_ref)
    st = vt.SolverSettings(solver_type="vlm", n_panels=14, n_chord=4)

    res = vt.trim(ac, cond, st, CL_target=0.8, pitch_control="elevator")

    assert res.status == "not_converged"
    assert res.converged is False
    assert res.result is None
    assert res.drag_basis == "none"
    for value in (res.CL, res.CD, res.CY, res.Cl, res.Cm, res.Cn):
        assert np.isnan(value)
    assert res.iterations == 0
    assert len(res.residual_history) == 1
    assert np.isnan(res.residual_history[0])
    # Regression: a Jacobian that was not computed was reported as zeros.
    assert np.all(np.isnan(res.jacobian))
    assert res.alpha_deg == pytest.approx(15.0)
    assert any("No solve succeeded" in n for n in res.notes)

    # to_dict stays strict JSON: NaN becomes None
    data = res.to_dict()
    json.dumps(data, allow_nan=False)
    assert data["CL"] is None
    assert data["residual_history"] == [None]


def test_control_with_no_effect_gives_singular_jacobian():
    """Verify a pitch control with no effect on CL and Cm is detected as singular."""
    # Regression: the condition number (about 5e14) was below the old limit 1e15
    # and the iteration ran to max_iterations.
    ac = _build_wing_tail_fin_aircraft()
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(2.0))
    st = vt.SolverSettings(solver_type="vlm", n_panels=14, n_chord=4)

    res = vt.trim(ac, cond, st, CL_target=0.5, pitch_control="rudder")

    assert res.status == "not_converged"
    assert res.converged is False
    assert res.iterations <= 1
    assert any("Singular Jacobian" in n and "'rudder'" in n for n in res.notes)


def test_invalid_settings_and_condition_raise_before_iteration(monkeypatch):
    """Verify invalid settings or condition raise ValueError before any solve."""
    # Regression: every exception of a solve was caught and became "not_converged".
    calls: list[int] = []
    real_analyze = vt.analyze

    def spy(*args, **kwargs):
        calls.append(1)
        return real_analyze(*args, **kwargs)

    monkeypatch.setattr(vt, "analyze", spy)
    ac = _build_wing_tail_aircraft()
    cond = vt.FlightCondition(V_inf=30.0, alpha=0.0)

    with pytest.raises(ValueError, match="bogus"):
        vt.trim(ac, cond, vt.SolverSettings(solver_type="bogus"), CL_target=0.5, pitch_control="elevator")

    with pytest.raises(ValueError, match="V_inf"):
        vt.trim(ac, vt.FlightCondition(V_inf=-1.0, alpha=0.0), CL_target=0.5, pitch_control="elevator")

    assert calls == []


def test_unexpected_solver_exception_propagates(monkeypatch):
    """Verify an exception that is not a documented solver failure propagates."""

    def broken(*args, **kwargs):
        raise TypeError("injected error")

    monkeypatch.setattr(vt, "analyze", broken)
    old = gpu.get_device()
    ac = _build_wing_tail_aircraft()
    cond = vt.FlightCondition(V_inf=30.0, alpha=0.0)
    with pytest.raises(TypeError, match="injected error"):
        vt.trim(ac, cond, CL_target=0.5, pitch_control="elevator")
    assert gpu.get_device() == old


def test_inner_solve_not_converged_is_not_trimmed(monkeypatch):
    """Verify an inner solve with converged=False at the final point does not give "trimmed"."""
    # Regression: a solution with converged=False was accepted as a trim point.
    real_analyze = vt.analyze
    target_cl = 0.4

    def not_converged_at_trim_point(*args, **kwargs):
        sol = real_analyze(*args, **kwargs)
        # Only the final point meets the tolerances of the trim.
        if abs(sol.totals.CL - target_cl) <= 1.0e-6 and abs(sol.totals.Cm) <= 1.0e-7:
            sol.converged = False
        return sol

    monkeypatch.setattr(vt, "analyze", not_converged_at_trim_point)
    ac = _build_wing_tail_aircraft()
    cond = vt.FlightCondition(V_inf=30.0, alpha=0.0)
    st = vt.SolverSettings(solver_type="vlm", n_panels=12, n_chord=4)

    res = vt.trim(ac, cond, st, CL_target=target_cl, pitch_control="elevator")

    assert res.status == "not_converged"
    assert res.converged is False
    assert any("did not converge" in n for n in res.notes)
    assert len(res.residual_history) == res.iterations + 1
    assert np.isnan(res.residual_history[-1])
    # The reported point is the last point with a converged solve
    assert res.result is not None
    assert res.result.converged is True
