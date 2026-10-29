# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Body rates p, q, r in the lattice solvers, and the rate derivatives.

Checks:

* closed forms of the classical lifting line for the elliptic wing
  (stability axes): Cl_p, Cl_r, Cn_p, Cn_r, derived from Glauert's series;
* the three-quarter-chord rule: a pitch rate gives the lift of the equal
  incidence change (one chordwise panel), and the section moment of
  thin-airfoil theory (-pi/4 per unit q c / 2V);
* equivalent geometry at alpha 0 (zero base circulation): a roll rate gives
  the loads of the antisymmetric linear twist, in free air and in ground
  effect;
* symmetry, linearity in the step, batch and single solves, the axes
  transform, the CPU fallback of the GPU path and the refusals.
"""

from __future__ import annotations

import dataclasses
import types
import warnings

import numpy as np
import pytest

import ventorum as vt
from ventorum.core.axes import rotation_body_to_stability, rotation_body_to_wind
from ventorum.core.errors import ValidityError
from ventorum.reference.analytic import wing_sections_elliptic


@pytest.fixture(autouse=True)
def _cpu():
    """Select the CPU, and restore the previous device and precision after the test."""
    from ventorum import gpu

    old = gpu.get_device()
    old_prec = gpu._precision
    gpu.set_device("cpu")
    yield
    gpu.set_device(old)
    gpu.set_precision(old_prec)


def _elliptic(AR: float, a0: float = 2.0 * np.pi) -> vt.Aircraft:
    c0 = 4.0 / (np.pi * AR)
    secs = wing_sections_elliptic(c0, straight_line="quarter_chord")
    for s in secs:
        s.airfoil = vt.LinearAirfoil(a0=a0)
    return vt.Aircraft(surfaces=[vt.LiftingSurface(semi_span=0.5, sections=secs)])


def _rect(AR: float = 6.0, chord: float = 1.0, **kw) -> vt.Aircraft:
    """Rectangular wing with a straight quarter-chord line on x = 0."""
    secs = [vt.WingSection(0.0, chord, x_le=-0.25 * chord), vt.WingSection(1.0, chord, x_le=-0.25 * chord)]
    return vt.Aircraft(surfaces=[vt.LiftingSurface(semi_span=0.5 * AR * chord, sections=secs, **kw)])


def _settings(solver: str, n_panels: int = 40, n_chord: int | None = None) -> vt.SolverSettings:
    return vt.SolverSettings(solver_type=solver, n_panels=n_panels, n_chord=n_chord)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Closed forms of the lifting line, elliptic wing (stability axes)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("a0_factor", [1.0, 0.9])
def test_elliptic_wing_matches_lifting_line_closed_forms(a0_factor):
    """Linear lifting line, AR 8, stability axes, against Glauert-series closed forms.

    With mu = a0 / (pi A) (elliptic wing):
    Cl_p = -a0 pi A / (8 (pi A + 2 a0)), Cl_r = (CL / 8) (1 + (1 + mu) / (1 + 2 mu)),
    Cn_p = -(CL / 8) (1 - mu) / (1 + 2 mu), Cn_r = -3 CL^2 (1 + mu) / (8 pi A (1 + 2 mu)).
    """
    AR, a0 = 8.0, a0_factor * 2.0 * np.pi
    mu = a0 / (np.pi * AR)
    fc = vt.FlightCondition(V_inf=20.0, alpha=np.radians(4.0))
    res = vt.rate_derivatives(_elliptic(AR, a0), fc, _settings("linear"), axes="stability")
    d, CL = res.derivatives, res.base.totals.CL
    assert d["Cl_p"] == pytest.approx(-a0 * np.pi * AR / (8.0 * (np.pi * AR + 2.0 * a0)), rel=5e-3)
    assert d["Cl_r"] == pytest.approx(CL / 8.0 * (1.0 + (1.0 + mu) / (1.0 + 2.0 * mu)), rel=5e-3)
    assert d["Cn_p"] == pytest.approx(-CL / 8.0 * (1.0 - mu) / (1.0 + 2.0 * mu), rel=5e-3)
    assert d["Cn_r"] == pytest.approx(-3.0 * CL ** 2 * (1.0 + mu) / (8.0 * np.pi * AR * (1.0 + 2.0 * mu)), rel=5e-3)
    assert res.axes == "stability" and res.device == "cpu" and res.n_solves == 7


def test_nonlinear_lifting_line_with_linear_sections_agrees_with_linear():
    """The nonlinear lifting line (Newton, linear sections) gives the derivatives of the linear one."""
    fc = vt.FlightCondition(V_inf=20.0, alpha=np.radians(4.0))
    lin = vt.rate_derivatives(_elliptic(8.0), fc, _settings("linear"), axes="stability").matrix
    non = vt.rate_derivatives(_elliptic(8.0), fc, _settings("nonlinear"), axes="stability").matrix
    assert np.allclose(non, lin, rtol=1e-2, atol=1e-4)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Pitch rate: three-quarter-chord rule and thin-airfoil section moment
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("solver, h", [("vlm", None), ("vlm", 0.3), ("linear", None), ("linear", 2.5)])
def test_pitch_rate_lift_equals_incidence_change(solver, h):
    """One chordwise panel, rotation about the quarter-chord line, alpha 0: CL_q = dCL/d(incidence).

    The rotation velocity at the three-quarter chord is q c / 2 = V q_hat, so
    the boundary condition is that of an incidence change of q_hat. At alpha
    0 the base circulation is zero, so the small move of the geometry by the
    incidence (also relative to the ground) has no first-order effect.
    """
    ac = _rect(6.0)
    st = _settings(solver, 20, 1 if solver == "vlm" else None)
    fc = vt.FlightCondition(V_inf=20.0, alpha=0.0, h=h)
    rates = vt.rate_derivatives(ac, fc, st)
    eps = 1e-4
    up, dn = ac.clone(), ac.clone()
    up.surfaces[0].incidence = eps
    dn.surfaces[0].incidence = -eps
    cl_i = (vt.analyze(up, fc, st).totals.CL - vt.analyze(dn, fc, st).totals.CL) / (2.0 * eps)
    assert rates.derivatives["CL_q"] == pytest.approx(cl_i, rel=1e-6)
    # Thin-airfoil section moment of the pitch rate about the quarter chord.
    # The local speed enters to second order in the step (5e-5 at step 0.01).
    assert rates.derivatives["Cm_q"] == pytest.approx(-np.pi / 4.0, rel=1e-4)


def test_vlm_section_moment_converges_to_thin_airfoil_with_chordwise_panels():
    """Large aspect ratio, rotation about the quarter chord: the chordwise VLM approaches -pi/4."""
    ac = _rect(40.0)
    fc = vt.FlightCondition(V_inf=20.0, alpha=0.0)
    cm = [vt.rate_derivatives(ac, fc, _settings("vlm", 40, nc)).derivatives["Cm_q"] for nc in (4, 16)]
    err = [abs(c / (-np.pi / 4.0) - 1.0) for c in cm]
    assert err[1] < err[0] and err[1] < 0.04


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Roll rate: equivalent antisymmetric twist (alpha 0), free air and ground
# ═══════════════════════════════════════════════════════════════════════════════

def _half_wings(tip_twist: float, AR: float = 6.0) -> vt.Aircraft:
    """Right and left half wings with an antisymmetric linear twist (tip twist +/- *tip_twist* [rad])."""
    s = 0.5 * AR

    def half(twist: float, name: str) -> vt.LiftingSurface:
        secs = [vt.WingSection(0.0, 1.0, x_le=-0.25), vt.WingSection(1.0, 1.0, twist=twist, x_le=-0.25)]
        return vt.LiftingSurface(name=name, semi_span=s, sections=secs, is_symmetric=False)

    ac = vt.Aircraft(surfaces=[half(tip_twist, "right"), half(-tip_twist, "left").mirrored("left")])
    # With a twist the halves are not mirror copies, so the automatic
    # reference values would use one half: give the values of the whole wing.
    ac.S_ref, ac.b_ref, ac.c_ref = 2.0 * s, 2.0 * s, 1.0
    return ac


@pytest.mark.parametrize("h", [None, 0.5])
def test_roll_rate_equals_antisymmetric_twist(h):
    """A roll rate gives the normal velocity p y at a flat wing: the twist p y / V (tip twist = p_hat)."""
    st = _settings("vlm", 16, 4)
    fc = vt.FlightCondition(V_inf=20.0, alpha=0.0, h=h)
    rates = vt.rate_derivatives(_half_wings(0.0), fc, st)
    eps = 1e-4
    up = vt.analyze(_half_wings(eps), fc, st).totals
    dn = vt.analyze(_half_wings(-eps), fc, st).totals
    for name in ("Cl", "Cn", "CY"):
        twist = (getattr(up, name) - getattr(dn, name)) / (2.0 * eps)
        assert rates.derivatives[f"{name}_p"] == pytest.approx(twist, rel=2e-3, abs=1e-6)
    assert rates.derivatives["Cl_p"] < 0.0


def test_ground_effect_increases_roll_damping():
    """Near the ground the image vortices reduce the downwash: |Cl_p| grows as the height falls."""
    st = _settings("vlm", 16, 4)
    clp = [vt.rate_derivatives(_rect(6.0), vt.FlightCondition(V_inf=20.0, alpha=np.radians(2.0), h=h), st)
           .derivatives["Cl_p"] for h in (None, 1.0, 0.3)]
    assert clp[0] < 0.0 and abs(clp[1]) > abs(clp[0]) and abs(clp[2]) > abs(clp[1])


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Symmetry, linearity, batch, axes
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("solver", ["vlm", "linear"])
@pytest.mark.parametrize("h", [None, 2.5])
def test_symmetry_of_the_rate_derivatives(solver, h):
    """Symmetric wing and flow: q gives no lateral derivatives; p and r give no longitudinal ones."""
    fc = vt.FlightCondition(V_inf=20.0, alpha=np.radians(3.0), h=h)
    d = vt.rate_derivatives(_rect(6.0), fc, _settings(solver, 16)).derivatives
    for c in ("CY", "Cl", "Cn"):
        assert abs(d[f"{c}_q"]) < 1e-9
    for c in ("CL", "CD", "Cm"):
        for w in ("p", "r"):
            assert abs(d[f"{c}_{w}"]) < 1e-9


def test_vlm_derivatives_do_not_depend_on_the_step():
    """VLM: circulation linear and forces quadratic in the rates, so the central difference is exact."""
    ac = _rect(6.0, dihedral=np.radians(5.0))
    fc = vt.FlightCondition(V_inf=20.0, alpha=np.radians(3.0), beta=np.radians(2.0))
    a = vt.rate_derivatives(ac, fc, _settings("vlm", 12), step=0.01).matrix
    b = vt.rate_derivatives(ac, fc, _settings("vlm", 12), step=0.05).matrix
    assert np.allclose(a, b, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("solver", ["vlm", "linear", "nonlinear"])
def test_batch_and_single_solves_agree_with_rates(solver):
    """A case with rates in the batch of rate_derivatives equals a single analyze of that case."""
    ac = _rect(6.0, dihedral=np.radians(4.0))
    ac.ref_point = np.array([0.3, 0.0, -0.1])
    fc = vt.FlightCondition(V_inf=25.0, alpha=np.radians(3.0), p=0.4, q=-0.2, r=0.3)
    st = _settings(solver, 12)
    res = vt.rate_derivatives(ac, fc, st)
    one = vt.analyze(ac.clone(), fc, st).totals
    for name in ("CL", "CY", "Cl", "Cm", "Cn", "CDi"):
        assert getattr(res.base.totals, name) == pytest.approx(getattr(one, name), rel=1e-11, abs=1e-13)


def test_sweep_keeps_the_rates():
    """solve_sweep copies the condition with its rates for every angle."""
    ac = _rect(6.0)
    st = _settings("vlm", 12)
    fc = vt.FlightCondition(V_inf=20.0, alpha=0.0, p=0.5)
    sweep = vt.make_solver("vlm").solve_sweep(ac, fc, st, np.radians([2.0, 4.0])) \
        if hasattr(vt, "make_solver") else vt.VortexLatticeSolver().solve_sweep(ac, fc, st, np.radians([2.0, 4.0]))
    one = vt.analyze(ac.clone(), dataclasses.replace(fc, alpha=np.radians(4.0)), st).totals
    assert sweep[1].condition.p == 0.5
    assert sweep[1].totals.Cl == pytest.approx(one.Cl, rel=1e-10)


@pytest.mark.parametrize("axes", ["stability", "wind"])
def test_axes_transform_equals_direct_rates_in_those_axes(axes):
    """Derivatives in stability or wind axes equal differences with rates given in those axes.

    The rate vector in the target axes is turned to body rates, solved, and
    the moments are turned back with the reference lengths b and c.
    """
    ac = _rect(5.0, chord=0.8, dihedral=np.radians(6.0))
    alpha, beta = np.radians(6.0), np.radians(4.0)
    fc = vt.FlightCondition(V_inf=20.0, alpha=alpha, beta=beta)
    st = _settings("vlm", 12)
    res = vt.rate_derivatives(ac, fc, st, axes=axes)
    R = rotation_body_to_stability(alpha) if axes == "stability" else rotation_body_to_wind(alpha, beta)
    b, c, V = res.b_ref, res.c_ref, 20.0
    L = np.array([b, c, b])
    h = 0.01
    for j, w in enumerate("pqr"):
        cols = []
        for sgn in (1.0, -1.0):
            omega_axes = np.zeros(3)
            omega_axes[j] = sgn * h * 2.0 * V / L[j]
            pb, qb, rb = R.T @ omega_axes
            t = vt.analyze(ac.clone(), dataclasses.replace(fc, p=pb, q=qb, r=rb), st).totals
            m_axes = R @ (np.array([t.Cl, t.Cm, t.Cn]) * L) / L
            cols.append(np.array([t.CL, t.CDi, t.CY, *m_axes]))
        direct = (cols[0] - cols[1]) / (2.0 * h)
        assert np.allclose(res.matrix[:, j], direct, rtol=1e-7, atol=1e-9), w


def test_body_axes_derivatives_use_the_body_rates():
    """Body axes: the derivative of each coefficient per unit p b/2V equals the difference of two solves."""
    ac = _rect(6.0)
    fc = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    st = _settings("vlm", 12)
    res = vt.rate_derivatives(ac, fc, st, step=0.02)
    b = res.b_ref
    p = 0.02 * 2.0 * 30.0 / b
    up = vt.analyze(ac.clone(), dataclasses.replace(fc, p=p), st).totals
    dn = vt.analyze(ac.clone(), dataclasses.replace(fc, p=-p), st).totals
    assert res.derivatives["Cl_p"] == pytest.approx((up.Cl - dn.Cl) / 0.04, rel=1e-10)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Signs, hinge moments, notes, GPU fallback and refusals
# ═══════════════════════════════════════════════════════════════════════════════

def test_signs_of_the_damping_derivatives():
    """Wing with a tail aft of the reference point: Cl_p < 0, Cm_q < 0; a fin gives Cn_r < 0 and CY_r > 0."""
    wing = vt.LiftingSurface(name="wing", semi_span=3.0, sections=[vt.WingSection(0.0, 1.0, x_le=-0.25),
                                                                   vt.WingSection(1.0, 1.0, x_le=-0.25)])
    tail = vt.LiftingSurface(name="tail", semi_span=1.0, position=np.array([3.0, 0.0, 0.0]),
                             sections=[vt.WingSection(0.0, 0.6), vt.WingSection(1.0, 0.6)])
    fin = vt.LiftingSurface(name="fin", semi_span=0.8, is_symmetric=False, dihedral=np.pi / 2,
                            position=np.array([3.0, 0.0, 0.0]),
                            sections=[vt.WingSection(0.0, 0.6), vt.WingSection(1.0, 0.6)])
    ac = vt.Aircraft(surfaces=[wing, tail, fin])
    d = vt.rate_derivatives(ac, vt.FlightCondition(V_inf=30.0, alpha=np.radians(3.0)), _settings("vlm", 12)).derivatives
    assert d["Cl_p"] < 0.0
    assert d["Cm_q"] < 0.0 and d["CL_q"] > 0.0
    assert d["Cn_r"] < 0.0 and d["CY_r"] > 0.0


def test_chordwise_hinge_moment_includes_the_rotation():
    """Roll rate at alpha 0 on a wing with ailerons: the hinge moments are those of the equivalent twist."""
    def ac_with(tip_twist: float) -> vt.Aircraft:
        ac = _half_wings(tip_twist)
        for s in ac.surfaces:
            s.controls = [vt.ControlSurface(name="ail", eta_start=0.5, eta_end=0.9, hinge_x_c=0.75)]
        return ac

    st = _settings("vlm", 16, 8)
    V = 20.0
    fc = vt.FlightCondition(V_inf=V, alpha=0.0)
    s = 3.0
    eps = 1e-4
    with_rate = vt.hinge_moments(vt.analyze(ac_with(0.0), dataclasses.replace(fc, p=eps * V / s), st))
    with_twist = vt.hinge_moments(vt.analyze(ac_with(eps), fc, st))
    assert len(with_rate) == 2
    for a, b in zip(with_rate, with_twist):
        assert a.Ch == pytest.approx(b.Ch, rel=5e-3)
    assert with_rate[0].Ch == pytest.approx(-with_rate[1].Ch, rel=1e-9)


def test_large_rotation_gets_a_note():
    """A rotation speed above 0.1 V_inf on the geometry gives a trust note and a warning."""
    ac = _rect(6.0)
    fc = vt.FlightCondition(V_inf=10.0, alpha=np.radians(2.0), p=1.0)   # tip speed 0.3 V
    with pytest.warns(RuntimeWarning, match="rotation speed"):
        res = vt.analyze(ac, fc, _settings("vlm", 8))
    assert any("rotation speed" in n for n in res.details["notes"])


def test_no_note_for_small_rates():
    ac = _rect(6.0)
    fc = vt.FlightCondition(V_inf=30.0, alpha=np.radians(2.0), p=0.1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        vt.analyze(ac, fc, _settings("vlm", 8))


def test_gpu_path_returns_none_for_unsupported_solver(monkeypatch):
    """With the GPU selected, the pipeline returns None for a solver that it does not support (Fourier).

    The device check passes (the GPU is selected), so the solver check gives
    the None. A supported solver passes both checks and reaches the next step.
    """
    from ventorum import gpu
    from ventorum.gpu import pipeline

    class _Reached(Exception):
        pass

    def _use_gpu(*args, **kwargs):
        raise _Reached

    monkeypatch.setattr(gpu, "get_device", lambda: "gpu")
    monkeypatch.setattr(gpu, "use_gpu", _use_gpu)
    lattice = types.SimpleNamespace(n_panels=8)
    cond = vt.FlightCondition(V_inf=20.0, alpha=0.0, q=0.1)
    assert pipeline.solve_batch(vt.FourierSolver(), lattice, [cond], None, 1.0, 1.0, 1.0, None, 0, False,
                                0.0) is None
    with pytest.raises(_Reached):
        pipeline.solve_batch(vt.VortexLatticeSolver(), lattice, [cond], None, 1.0, 1.0, 1.0, None, 0, False, 0.0)


def test_fourier_solver_refuses_rates():
    with pytest.raises(ValidityError, match="body rates"):
        vt.analyze(_rect(6.0), vt.FlightCondition(V_inf=20.0, alpha=0.05, p=0.1), _settings("fourier", 20))
    with pytest.raises(ValueError, match="Fourier"):
        vt.rate_derivatives(_rect(6.0), vt.FlightCondition(V_inf=20.0), _settings("fourier", 20))


def test_panel3d_solver_is_refused_with_a_clear_message():
    """Regression: the 3D panel solver used to fail with an AttributeError (no lattice build)."""
    with pytest.raises(ValueError, match="3D panel method"):
        vt.rate_derivatives(_rect(6.0), vt.FlightCondition(V_inf=20.0), _settings("panel3d", 8, 8))


@pytest.mark.parametrize("kw, match", [({"step": 0.0}, "step"), ({"step": 0.2}, "step"),
                                       ({"axes": "earth"}, "axes")])
def test_invalid_inputs_are_refused(kw, match):
    with pytest.raises(ValueError, match=match):
        vt.rate_derivatives(_rect(6.0), vt.FlightCondition(V_inf=20.0), _settings("vlm", 8), **kw)


def test_non_finite_rate_is_refused():
    with pytest.raises(ValueError):
        vt.analyze(_rect(6.0), vt.FlightCondition(V_inf=20.0, r=float("nan")), _settings("vlm", 8))


def test_condition_clone_keeps_the_rates():
    fc = vt.FlightCondition(p=0.1, q=-0.2, r=0.3)
    c = fc.clone()
    assert (c.p, c.q, c.r) == (0.1, -0.2, 0.3)


def test_to_dict_is_json_ready():
    import json

    res = vt.rate_derivatives(_rect(6.0), vt.FlightCondition(V_inf=20.0), _settings("vlm", 8))
    out = json.loads(json.dumps(res.to_dict()))
    assert set(out["derivatives"]) == {f"{c}_{w}" for c in ("CL", "CD", "CY", "Cl", "Cm", "Cn") for w in "pqr"}
    assert out["device"] == "cpu" and out["axes"] == "body"


@pytest.mark.gpu
@pytest.mark.parametrize("solver", ["vlm", "linear"])
def test_gpu_device_runs_rate_cases_on_the_gpu(solver):
    """With the GPU selected, the cases with body rates run on the GPU and give the CPU derivatives."""
    from ventorum import gpu

    if not gpu.available():
        pytest.skip(f"GPU pipeline unavailable: {gpu.unavailable_reason()}")
    fc = vt.FlightCondition(V_inf=20.0, alpha=np.radians(3.0))
    st = _settings(solver, 12)
    cpu = vt.rate_derivatives(_rect(6.0), fc, st)
    gpu.set_device("gpu")
    gpu.set_precision("float64")
    on_gpu = vt.rate_derivatives(_rect(6.0), fc, st)
    assert on_gpu.device == "gpu"
    assert np.allclose(on_gpu.matrix, cpu.matrix, rtol=1e-10, atol=1e-10)
