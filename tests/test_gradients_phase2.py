# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests of the adjoint gradients with controls, body rates, the nonlinear lifting line and tabulated airfoils."""

from __future__ import annotations

import contextlib
import dataclasses
import warnings

import numpy as np
import pytest

import ventorum as vt
from ventorum import gpu


@contextlib.contextmanager
def _gpu_settings(device: str | None = None, precision: str | None = None):
    """Set the device and the precision; restore the raw stored settings after the block."""
    old_dev, old_prec = gpu.get_device(), gpu._precision
    try:
        if device is not None:
            gpu.set_device(device)
        if precision is not None:
            gpu.set_precision(precision)
        yield
    finally:
        gpu.set_device(old_dev)
        gpu.set_precision(old_prec)


@pytest.fixture
def float64_cpu():
    """Run the test on the CPU with float64 gradients; restore the settings after it."""
    with _gpu_settings(device="cpu", precision="float64"):
        yield


@pytest.fixture
def gpu_device():
    """Run the test with the GPU device selected; restore the settings after it."""
    if not gpu.available():
        pytest.skip(f"The GPU pipelines cannot run: {gpu.unavailable_reason()}")
    with _gpu_settings():
        yield


def _linear_section() -> vt.LinearAirfoil:
    return vt.LinearAirfoil(
        name="GradSec", a0=2.0 * np.pi, alpha_L0=np.radians(-2.0), Cd0=0.008, Cm0=-0.05)


def _tabulated_section(stall: bool = True) -> vt.TabulatedAirfoil:
    """A cambered polar with a smooth stall above 10 deg (made up, not measured data)."""
    a = np.radians(np.arange(-8.0, 22.0, 2.0))
    cl = 0.95 * 2.0 * np.pi * (a + np.radians(2.0))
    if stall:
        cl = cl - 8.0 * np.maximum(a - np.radians(10.0), 0.0) ** 2
    return vt.TabulatedAirfoil(name="GradTab", alpha=a, Cl_data=cl, Cd_data=0.008 + 0.01 * a ** 2,
                               Cm_data=-0.05 + 0.02 * a)


def _wing(airfoil=None, controls=None, sweep_deg: float = 0.0) -> vt.LiftingSurface:
    af = airfoil if airfoil is not None else _linear_section()
    w = vt.LiftingSurface(
        name="Wing", semi_span=5.0, sweep_le=np.radians(sweep_deg), dihedral=np.radians(3.0),
        sections=[vt.WingSection(y_frac=0.0, chord=2.0, airfoil=af),
                  vt.WingSection(y_frac=1.0, chord=1.0, airfoil=af)])
    if controls:
        w.controls = list(controls)
    return w


def _flap(deflection_deg: float = 5.0) -> vt.ControlSurface:
    return vt.ControlSurface(name="flap", eta_start=0.2, eta_end=0.55, hinge_x_c=0.75,
                             deflection=np.radians(deflection_deg), symmetric=True)


def _aileron(deflection_deg: float = 4.0) -> vt.ControlSurface:
    return vt.ControlSurface(name="aileron", eta_start=0.6, eta_end=0.95, hinge_x_c=0.7,
                             deflection=np.radians(deflection_deg), symmetric=False)


def _wing_tail_controls() -> vt.Aircraft:
    wing = _wing(controls=[_flap(), _aileron()], sweep_deg=10.0)
    tail = vt.LiftingSurface(
        name="Tail", semi_span=2.0, position=np.array([5.0, 0.0, 0.5]),
        sections=[vt.WingSection(y_frac=0.0, chord=0.8, airfoil=_linear_section()),
                  vt.WingSection(y_frac=1.0, chord=0.5, airfoil=_linear_section())])
    tail.controls = [vt.ControlSurface(name="elevator", eta_start=0.0, eta_end=1.0, hinge_x_c=0.7,
                                       deflection=np.radians(-2.0), symmetric=True)]
    return vt.Aircraft(name="WingTail", surfaces=[wing, tail])


_BETA = np.radians(2.0)
_RATES = {"p": 0.4, "q": 0.25, "r": -0.15}


def _settings(solver: str, n_chord: int | None = None) -> vt.SolverSettings:
    if solver == "vlm":
        return vt.SolverSettings(solver_type="vlm", n_panels=10, n_chord=n_chord)
    if solver == "nonlinear":
        return vt.SolverSettings(solver_type="nonlinear", n_panels=10, tolerance=1e-12)
    return vt.SolverSettings(solver_type="linear", n_panels=10)


def _cases() -> list[tuple]:
    """Configurations of the parity and Richardson tests: (id, aircraft, condition, settings, wrt).

    Every case has a sideslip (full system) and the inputs of its new
    feature. The node inputs are included for the vortex lattice only (the
    lifting-line node gradients carry the core caveat; see the notes).
    """
    cases = []
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=_BETA)
    cond_r = dataclasses.replace(cond, **_RATES)
    # Controls: chordwise mode (VLM, 4 panels), section mode (VLM 1 panel, linear LLT).
    ac = vt.Aircraft(surfaces=[_wing(controls=[_flap(), _aileron()], sweep_deg=10.0)])
    cases.append(("vlm4-controls", ac, cond, _settings("vlm", 4), ("alpha", "beta", "controls", "nodes")))
    cases.append(("vlm1-controls", ac, cond, _settings("vlm", 1), ("alpha", "beta", "controls", "nodes")))
    cases.append(("linear-controls", ac, cond, _settings("linear"), ("alpha", "beta", "controls", "sections")))
    # Controls at zero deflection (the derivative at zero is included).
    ac0 = vt.Aircraft(surfaces=[_wing(controls=[_flap(0.0), _aileron(0.0)])])
    cases.append(("vlm4-controls-zero", ac0, cond, _settings("vlm", 4), ("alpha", "controls")))
    # Wing and tail with an elevator.
    cases.append(("vlm2-wing-tail", _wing_tail_controls(), cond, _settings("vlm", 2),
                  ("alpha", "controls", "nodes", "reference")))
    # Body rates.
    ac = vt.Aircraft(surfaces=[_wing(sweep_deg=10.0)])
    cases.append(("vlm4-rates", ac, cond_r, _settings("vlm", 4), ("alpha", "beta", "p", "q", "r", "nodes",
                                                                   "reference")))
    cases.append(("vlm1-rates", ac, cond_r, _settings("vlm", 1), ("alpha", "p", "q", "r", "reference")))
    cases.append(("linear-rates", ac, cond_r, _settings("linear"), ("alpha", "V_inf", "p", "q", "r",
                                                                     "reference")))
    cases.append(("vlm4-rates-zero", ac, cond, _settings("vlm", 4), ("alpha", "p", "q", "r")))
    # Ground effect with rates (VLM).
    acg = vt.Aircraft(surfaces=[_wing()])
    acg.compute_reference_values()
    cond_g = dataclasses.replace(cond_r, h=0.5 * float(acg.c_ref))
    cases.append(("vlm4-ground-rates", acg, cond_g, _settings("vlm", 4), ("alpha", "h", "p", "q", "r")))
    # Tabulated airfoils in the linear solvers.
    act = vt.Aircraft(surfaces=[_wing(_tabulated_section(), controls=[_flap()])])
    cases.append(("vlm4-tabulated", act, cond, _settings("vlm", 4), ("alpha", "beta", "V_inf", "nodes")))
    cases.append(("linear-tabulated", act, cond, _settings("linear"), ("alpha", "beta", "controls")))
    # Nonlinear lifting line: linear and tabulated airfoils, controls and rates.
    acn = vt.Aircraft(surfaces=[_wing(controls=[_flap(), _aileron()])])
    cases.append(("nonlinear-linear", acn, cond, _settings("nonlinear"),
                  ("alpha", "beta", "V_inf", "controls", "sections")))
    acnt = vt.Aircraft(surfaces=[_wing(_tabulated_section(), controls=[_flap(), _aileron()])])
    cond8 = dataclasses.replace(cond, alpha=np.radians(8.0))
    cases.append(("nonlinear-tabulated", acnt, cond8, _settings("nonlinear"),
                  ("alpha", "beta", "V_inf", "rho", "controls")))
    cases.append(("nonlinear-tabulated-rates", acnt, dataclasses.replace(cond8, **_RATES),
                  _settings("nonlinear"), ("alpha", "p", "q", "r", "controls", "reference")))
    return cases


_IDS = [c[0] for c in _cases()]
_OUTPUTS = ("CL", "CY", "CDi", "CDp", "CD", "Cl", "Cm", "Cn")


def _failed(out: dict) -> dict:
    """Return the failed components and the reasons of a check_gradients result."""
    bad = {k: v for k, v in out["details"].items() if v["compared"] and not v["passed"]}
    return {"reasons": out["reasons"], "failed": bad}


@pytest.mark.parametrize("case_id,ac,cond,st,wrt", _cases(), ids=_IDS)
def test_forward_parity(case_id, ac, cond, st, wrt, float64_cpu) -> None:
    """The differentiated reference copy equals the production totals to 1e-12 relative (float64)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = vt.gradients(ac, cond, st, outputs=_OUTPUTS, wrt=wrt, axes="body")
    assert res.forward_mismatch <= 1e-12, (case_id, res.forward_mismatch)
    assert res.symmetric is False
    for w in wrt:
        assert w in res.jacobian["CL"], (case_id, w)


@pytest.mark.parametrize("case_id,ac,cond,st,wrt", _cases(), ids=_IDS)
def test_adjoint_vs_richardson(case_id, ac, cond, st, wrt, float64_cpu) -> None:
    """check_gradients passes (1e-6) for every requested input group, the new inputs included."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = vt.check_gradients(ac, cond, st, outputs=("CL", "CD", "Cl", "Cm", "Cn"), wrt=wrt,
                                 n_directions=2, rtol=1e-6, seed=0)
    assert set(out["groups"]) == set(wrt), (case_id, out["groups"])
    assert all(c["compared"] > 0 for c in out["groups"].values()), (case_id, out["groups"])
    assert out["pass"], (case_id, _failed(out))


@pytest.mark.parametrize("case_id,ac,cond,st,wrt", _cases(), ids=_IDS)
def test_kernels_equal_reference(case_id, ac, cond, st, wrt, float64_cpu) -> None:
    """The compiled VJP kernels and the PyTorch reference copy give the same gradients (1e-12)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = vt.gradients(ac, cond, st, outputs=_OUTPUTS, wrt=wrt, method="reference")
        ker = vt.gradients(ac, cond, st, outputs=_OUTPUTS, wrt=wrt, method="kernels")
    assert ref.partials == "reference:torch" and ker.partials.startswith("kernels:")
    for o in _OUTPUTS:
        a, b = _flat(ref.jacobian[o]), _flat(ker.jacobian[o])
        for key in a:
            scale = float(np.max(np.abs(a[key])))
            err = float(np.max(np.abs(a[key] - b[key])))
            assert err <= 1e-12 * scale or (scale == 0.0 and err == 0.0), (case_id, o, key, err, scale)


def _flat(jac: dict) -> dict[str, np.ndarray]:
    """Return the input groups of one output jacobian as flat arrays.

    The node arrays form one group per surface and tag, every other dict
    input (sections, reference, controls, planform) one group, and the
    scalar inputs (alpha, beta, V_inf, rho, h, phi, p, q, r) one group: a
    scalar derivative that is zero by symmetry has no relative scale of its
    own.
    """
    out: dict[str, np.ndarray] = {}
    scalars = []
    for key, val in jac.items():
        if key == "nodes":
            for s, arrs in val.items():
                for tag in ("le", "te"):
                    out[f"nodes.{s}.{tag}"] = np.asarray(arrs[tag], dtype=float).ravel()
        elif isinstance(val, dict):
            out[key] = np.concatenate([np.atleast_1d(np.asarray(v, dtype=float)).ravel()
                                       for v in val.values()])
        else:
            scalars.append(float(val))
    if scalars:
        out["scalars"] = np.asarray(scalars)
    return out


def _scalars(res) -> dict[str, float]:
    t = res.totals
    return {"CL": float(t.CL), "CY": float(t.CY), "Cl": float(t.Cl), "Cm": float(t.Cm),
            "Cn": float(t.Cn), "CD": float(t.CD_total if t.CD_total is not None else t.CDi)}


@pytest.mark.parametrize("solver,n_chord", [("vlm", 4), ("linear", None), ("nonlinear", None)])
def test_controls_match_control_derivatives(solver, n_chord, float64_cpu) -> None:
    """dC/d(deflection) equals control_derivatives within its difference error.

    control_derivatives uses central differences with a 1 deg step. Its
    difference error is estimated from the same differences with 1 deg and
    0.5 deg: (4/3) |FD(1 deg) - FD(0.5 deg)| (the leading h^2 term), with a
    margin of 25 % for the higher-order terms, plus the round-off.
    """
    ac = vt.Aircraft(surfaces=[_wing(controls=[_flap(), _aileron()])])
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    st = _settings(solver, n_chord)
    outs = ("CL", "CY", "Cl", "Cm", "Cn", "CD")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = vt.gradients(ac, cond, st, outputs=outs, wrt=("controls",), axes="body")
        cd = vt.control_derivatives(ac, cond, st, step_deg=1.0, axes="body", hinge_alpha=False)
    for name in ("flap", "aileron"):
        fds = []
        for step in (np.radians(1.0), np.radians(0.5)):
            vals = []
            for sgn in (1.0, -1.0):
                a = ac.clone()
                for s in a.surfaces:
                    for c in s.controls:
                        if c.name == name:
                            c.deflection = float(c.deflection) + sgn * step
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    vals.append(_scalars(vt.analyze(a, cond, st.clone())))
            fds.append({o: (vals[0][o] - vals[1][o]) / (2.0 * step) for o in outs})
        d = cd.derivatives[name]
        for o in outs:
            grad = float(res.jacobian[o]["controls"][name])
            diff_err = 1.25 * (4.0 / 3.0) * abs(fds[0][o] - fds[1][o]) + 1e-9 * max(1.0, abs(grad))
            assert abs(grad - float(getattr(d, o))) <= diff_err, (solver, name, o, grad, getattr(d, o), diff_err)


@pytest.mark.parametrize("solver,n_chord", [("vlm", 4), ("vlm", 1), ("linear", None), ("nonlinear", None)])
def test_rates_match_rate_derivatives(solver, n_chord, float64_cpu) -> None:
    """The nondimensional rate derivatives equal rate_derivatives within its difference error.

    rate_derivatives uses central differences with a step 0.01 of the
    nondimensional rate. Its difference error is estimated from the steps
    0.01 and 0.005 as in the control test.
    """
    ac = vt.Aircraft(surfaces=[_wing(sweep_deg=10.0)])
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=_BETA, p=0.1, q=0.05)
    st = _settings(solver, n_chord)
    outs = ("CL", "CY", "Cl", "Cm", "Cn", "CD")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = vt.gradients(ac, cond, st, outputs=outs, wrt=("p", "q", "r"), axes="body")
        rd = vt.rate_derivatives(ac, cond, st, step=0.01, axes="body")
        rd2 = vt.rate_derivatives(ac, cond, st, step=0.005, axes="body")
    ac.compute_reference_values()
    V = float(cond.V_inf)
    to_hat = {"p": 2.0 * V / float(ac.b_ref), "q": 2.0 * V / float(ac.c_ref), "r": 2.0 * V / float(ac.b_ref)}
    for o in outs:
        for w in ("p", "q", "r"):
            key = f"{o}_{w}"
            grad = float(res.jacobian[o][w]) * to_hat[w]
            diff_err = (1.25 * (4.0 / 3.0) * abs(rd.derivatives[key] - rd2.derivatives[key])
                        + 1e-9 * max(1.0, abs(grad)))
            assert abs(grad - rd.derivatives[key]) <= diff_err, (solver, key, grad, rd.derivatives[key])
    # The notes give the same nondimensional values.
    note = next(n for n in res.notes if n.startswith("Body rates"))
    assert f"Cl_p = {float(res.jacobian['Cl']['p']) * to_hat['p']:.6g}" in note, note


@pytest.mark.parametrize("case_id,ac,cond,st,wrt", _cases(), ids=_IDS)
def test_float32_gate(case_id, ac, cond, st, wrt) -> None:
    """Float32 gradients (CPU) agree with the float64 gradients within 1e-3 (per input group)."""
    out = {}
    for prec in ("float64", "float32"):
        with _gpu_settings(device="cpu", precision=prec), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out[prec] = vt.gradients(ac, cond, st, outputs=("CL", "CD", "Cl", "Cm", "Cn"), wrt=wrt)
    assert out["float32"].precision_gradient == "float32"
    for o in ("CL", "CD", "Cl", "Cm", "Cn"):
        a, b = _flat(out["float64"].jacobian[o]), _flat(out["float32"].jacobian[o])
        for key in a:
            scale = float(np.max(np.abs(a[key])))
            err = float(np.max(np.abs(a[key] - b[key])))
            assert err <= 1e-3 * scale or scale == 0.0, (case_id, o, key, err, scale)


def test_nonlinear_not_converged_is_refused(float64_cpu) -> None:
    """A nonlinear solve that did not converge raises GradientScopeError with the reason."""
    ac = vt.Aircraft(surfaces=[_wing(_tabulated_section())])
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(8.0))
    st = vt.SolverSettings(solver_type="nonlinear", n_panels=10, max_iterations=1, tolerance=1e-15)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(vt.GradientScopeError, match="did not converge"):
            vt.gradients(ac, cond, st, outputs=("CL",), wrt=("alpha",))
        with pytest.raises(vt.GradientScopeError, match="did not converge"):
            vt.gradient_vjp(ac, cond, st, seeds={"CL": 1.0}, wrt=("alpha",))


def _plateau_section() -> vt.TabulatedAirfoil:
    """A polar whose lift is constant above 12 deg (zero slope past the maximum lift)."""
    a = np.radians(np.arange(-8.0, 26.0, 2.0))
    cl = np.minimum(2.0 * np.pi * a, 2.0 * np.pi * np.radians(12.0))
    return vt.TabulatedAirfoil(name="Plateau", alpha=a, Cl_data=cl, Cd_data=0.01 + 0.0 * a)


def test_past_maximum_lift_warns(float64_cpu) -> None:
    """Strips with zero section lift slope give a warning and a note that the gradient is local."""
    ac = vt.Aircraft(surfaces=[_wing(_plateau_section())])
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(18.0))
    st = vt.SolverSettings(solver_type="nonlinear", n_panels=10, tolerance=1e-12)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        res = vt.gradients(ac, cond, st, outputs=("CL",), wrt=("alpha",))
    assert any("Past the maximum lift" in str(w.message) for w in rec)
    assert any(n.startswith("Past the maximum lift") and "may not be unique" in n for n in res.notes)
    assert res.forward_mismatch <= 1e-12
    # Below the maximum lift there is no such note.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        low = vt.gradients(ac, dataclasses.replace(cond, alpha=np.radians(3.0)), st,
                           outputs=("CL",), wrt=("alpha",))
    assert not any(n.startswith("Past the maximum lift") for n in low.notes)


def test_tabulated_outside_table_note(float64_cpu) -> None:
    """Section angles outside the table: a note names the strips; their polar derivative is zero."""
    a = np.radians(np.arange(-4.0, 8.0, 2.0))
    af = vt.TabulatedAirfoil(name="Short", alpha=a, Cl_data=2.0 * np.pi * a, Cd_data=0.01 + 0.02 * a ** 2)
    ac = vt.Aircraft(surfaces=[_wing(af)])
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(9.0))
    st = vt.SolverSettings(solver_type="nonlinear", n_panels=10, tolerance=1e-12)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = vt.gradients(ac, cond, st, outputs=("CL", "CD"), wrt=("alpha",))
    assert any("outside the table" in n for n in res.notes), res.notes
    assert res.forward_mismatch <= 1e-12


def test_piecewise_cubic_matches_scipy() -> None:
    """The torch evaluation of the production interpolation equals scipy (1e-14); slopes too.

    At a table point the derivative is the one of the interval to the right
    (one-sided); the shape-preserving cubic is continuous in its first
    derivative, so both sides agree to round-off.
    """
    import torch

    from ventorum.sensitivity.reference import pp_eval, pp_from_scipy, pp_slope

    af = _tabulated_section()
    f = af._pchip()
    pp = pp_from_scipy(f["Cl"], "cpu", torch.float64)
    ta = af.tables()[0]
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.uniform(ta[0], ta[-1], 50), ta, [ta[0] - 0.1, ta[-1] + 0.1]])
    xt = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    val = pp_eval(pp, xt)
    (g,) = torch.autograd.grad(val.sum(), xt)
    assert np.max(np.abs(val.detach().numpy() - af.Cl(x))) <= 1e-14
    slope = pp_slope(pp, xt.detach()).numpy()
    assert np.max(np.abs(slope - af.Cl_slope(x))) <= 1e-12
    inside = (x >= ta[0]) & (x <= ta[-1])
    assert np.max(np.abs(g.numpy()[inside] - slope[inside])) <= 1e-12
    assert np.all(g.numpy()[~inside] == 0.0)
    # One-sided derivatives at the interior table points agree with the left interval.
    left = f["Cl"].derivative()(ta[1:-1] - 1e-13)
    assert np.max(np.abs(pp_slope(pp, torch.tensor(ta[1:-1])).numpy() - left)) <= 1e-9


def test_newton_jacobian_matches_autograd(float64_cpu) -> None:
    """The Newton Jacobian of the nonlinear lifting line equals the autograd derivative of the residual."""
    import torch

    from ventorum.sensitivity import adjoint

    ac = vt.Aircraft(surfaces=[_wing(_tabulated_section(), controls=[_aileron()])])
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(8.0), beta=_BETA, p=0.2, q=0.1)
    st = _settings("nonlinear")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prep = adjoint._prepare_case(ac, cond, st, "body", ("alpha", "p"), ("CL",), "reference")
    model = prep["model"]
    t = adjoint.make_input_tensors(model, prep["base"], (), prep["device"], prep["precision"])
    g0 = torch.as_tensor(prep["g_unknown"], dtype=torch.float64)

    def residual(g: torch.Tensor) -> torch.Tensor:
        _, A, rhs = model.forward(t, g)
        return A @ g.detach() - rhs

    _, J, _ = model.forward(t, g0)
    J_ad = torch.autograd.functional.jacobian(residual, g0)
    scale = float(torch.max(torch.abs(J)))
    assert float(torch.max(torch.abs(J - J_ad))) <= 1e-12 * scale
    # The section Cl residual of the production state is at the solver tolerance (with round-off).
    assert float(torch.max(torch.abs(model.b["_nl_residual"]))) <= 1e-10


def test_symmetric_fold_rule_rates_and_controls(float64_cpu) -> None:
    """q and symmetric controls keep the fold; p, r and an antisymmetric control use the full system."""
    ac = vt.Aircraft(surfaces=[_wing(controls=[_flap()])])
    ac_a = vt.Aircraft(surfaces=[_wing(controls=[_flap(), _aileron(0.0)])])
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), q=0.1)
    st = _settings("vlm", 4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sym = vt.gradients(ac, cond, st, outputs=("CL", "Cm"), wrt=("q", "controls"))
        full = vt.gradients(ac, cond, st, outputs=("CL", "Cm"), wrt=("q", "controls", "p"))
        full_a = vt.gradients(ac_a, cond, st, outputs=("CL", "Cm"), wrt=("controls",))
    assert sym.symmetric is True
    assert full.symmetric is False and full_a.symmetric is False
    for o in ("CL", "Cm"):
        assert sym.jacobian[o]["q"] == pytest.approx(full.jacobian[o]["q"], rel=1e-12)
        assert sym.jacobian[o]["controls"]["flap"] == pytest.approx(
            full.jacobian[o]["controls"]["flap"], rel=1e-12)
        assert full.jacobian[o]["p"] == pytest.approx(0.0, abs=1e-10)
        # A symmetric output has zero derivative with the antisymmetric control at zero.
        assert abs(full_a.jacobian[o]["controls"]["aileron"]) <= 1e-10


def test_planform_with_deflected_section_controls(float64_cpu) -> None:
    """The planform chain keeps the base deflection of the section mode; Richardson check 1e-6."""
    ac = vt.Aircraft(surfaces=[_wing(controls=[_flap(), _aileron()])])
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=_BETA)
    st = _settings("linear")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = vt.gradients(ac, cond, st, outputs=("CL", "Cm"), wrt=("planform", "controls"))
        out = vt.check_gradients(ac, cond, st, outputs=("CL", "Cm"), wrt=("planform", "controls"),
                                 n_directions=0, rtol=1e-6)
    assert res.forward_mismatch <= 1e-12
    assert out["pass"], _failed(out)


def test_planform_coincident_break_nonlinear(float64_cpu) -> None:
    """The edge-mean branch of a coincident break runs Newton steps for the nonlinear lifting line."""
    from ventorum.sensitivity.planform import build_planform

    wing = vt.LiftingSurface(
        name="W", semi_span=4.0,
        sections=[vt.WingSection(y_frac=0.0, chord=2.0, twist=np.radians(1.0), x_le=0.0, z_le=0.0,
                                 airfoil=_linear_section()),
                  vt.WingSection(y_frac=0.5, chord=1.5, x_le=0.5, z_le=0.25, airfoil=_linear_section()),
                  vt.WingSection(y_frac=1.0, chord=1.0, twist=np.radians(-1.0), x_le=1.2, z_le=0.9,
                                 airfoil=_linear_section())])
    ac = vt.Aircraft(surfaces=[wing])
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=_BETA)
    st = vt.SolverSettings(solver_type="nonlinear", n_panels=10, spacing="uniform", tolerance=1e-12)
    geom = build_planform(ac, st, "llt")
    assert geom.coincident == [("W", 1)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = vt.gradients(ac, cond, st, outputs=("CL",), wrt=("planform",))
    g = float(res.jacobian["CL"]["planform"]["W"]["sections"][1]["y_frac"])

    def cl_at(v: float) -> float:
        a = ac.clone()
        a.surfaces[0].sections[1].y_frac = v
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(vt.analyze(a, cond, st.clone()).totals.CL)

    fd = {h: (cl_at(0.5 + h) - cl_at(0.5 - h)) / (2.0 * h) for h in (2e-4, 1e-4)}
    ref = (4.0 * fd[1e-4] - fd[2e-4]) / 3.0
    assert abs(g - ref) <= 1e-6 * abs(ref), (g, ref)


def test_scope_refusals_and_notes(float64_cpu) -> None:
    """Fourier, planform with tabulated airfoils and controls without controls are refused; LLT node note."""
    ac = vt.Aircraft(surfaces=[_wing()])
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(vt.GradientScopeError, match="Fourier"):
            vt.gradients(ac, cond, vt.SolverSettings(solver_type="fourier", n_panels=8),
                         outputs=("CL",), wrt=("alpha",))
        with pytest.raises(vt.GradientScopeError, match="planform input needs linear airfoils"):
            vt.gradients(vt.Aircraft(surfaces=[_wing(_tabulated_section())]), cond, _settings("linear"),
                         outputs=("CL",), wrt=("planform",))
        with pytest.raises(ValueError, match="control surfaces"):
            vt.gradients(ac, cond, _settings("vlm", 2), outputs=("CL",), wrt=("controls",))
        with pytest.raises(ValueError, match="finite"):
            vt.gradients(ac, dataclasses.replace(cond, p=float("nan")), _settings("vlm", 2),
                         outputs=("CL",), wrt=("p",))
        for solver in ("linear", "nonlinear"):
            res = vt.gradients(ac, cond, _settings(solver), outputs=("CL",), wrt=("nodes",))
            assert any(n.startswith("Lifting-line node gradients") for n in res.notes), (solver, res.notes)
        res = vt.gradients(ac, cond, _settings("vlm", 2), outputs=("CL",), wrt=("nodes",))
        assert not any(n.startswith("Lifting-line node gradients") for n in res.notes)


def test_vjp_with_controls_and_rates(float64_cpu) -> None:
    """gradient_vjp of a weighted sum equals the weighted sum of the gradients."""
    ac = vt.Aircraft(surfaces=[_wing(_tabulated_section(), controls=[_flap(), _aileron()])])
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(6.0), beta=_BETA, p=0.2)
    st = _settings("nonlinear")
    wrt = ("controls", "p", "q", "r", "alpha")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = vt.gradients(ac, cond, st, outputs=("CL", "Cl"), wrt=wrt)
        g = vt.gradient_vjp(ac, cond, st, seeds={"CL": 2.0, "Cl": -3.0}, wrt=wrt)
    for w in ("p", "q", "r", "alpha"):
        ref = 2.0 * res.jacobian["CL"][w] - 3.0 * res.jacobian["Cl"][w]
        assert g[w] == pytest.approx(ref, rel=1e-12, abs=1e-14)
    for name in ("flap", "aileron"):
        ref = 2.0 * res.jacobian["CL"]["controls"][name] - 3.0 * res.jacobian["Cl"]["controls"][name]
        assert g["controls"][name] == pytest.approx(ref, rel=1e-12, abs=1e-14)


def _gpu_cases() -> list[tuple]:
    by_id = {c[0]: c for c in _cases()}
    return [by_id[k] for k in ("vlm4-controls", "linear-rates", "nonlinear-tabulated-rates", "vlm4-tabulated")]


@pytest.mark.gpu
@pytest.mark.parametrize("case_id,ac,cond,st,wrt", _gpu_cases(), ids=[c[0] for c in _gpu_cases()])
def test_gpu_device(case_id, ac, cond, st, wrt, gpu_device) -> None:
    """GPU float64 equals CPU float64 (1e-10); GPU float32 is within 1e-3 of CPU float64."""
    outs = ("CL", "CD", "Cl", "Cm")
    res = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for dev, prec in (("gpu", "float32"), ("gpu", "float64"), ("cpu", "float64")):
            gpu.set_device(dev)
            gpu.set_precision(prec)
            res[(dev, prec)] = vt.gradients(ac, cond, st, outputs=outs, wrt=wrt)
    ref = res[("cpu", "float64")]
    assert res[("gpu", "float64")].device == "cuda"
    for (dev, prec), tol in ((("gpu", "float64"), 1e-10), (("gpu", "float32"), 1e-3)):
        r = res[(dev, prec)]
        for o in outs:
            a, b = _flat(ref.jacobian[o]), _flat(r.jacobian[o])
            for key in a:
                scale = float(np.max(np.abs(a[key])))
                err = float(np.max(np.abs(a[key] - b[key])))
                assert err <= tol * scale or scale == 0.0, (case_id, dev, prec, o, key, err, scale)


def test_user_guide_phase2_example_runs(float64_cpu) -> None:
    """Prove that the controls and rates example of the gradient how-to runs as written."""
    import re
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "docs" / "user" / "how_to_gradients.md").read_text(
        encoding="utf-8")
    blocks = [b for b in re.findall(r"```python\n(.*?)```", text, flags=re.S) if "controls" in b]
    assert blocks, "the guide must have an example with controls"
    namespace: dict = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec(compile(blocks[0], "docs/user/how_to_gradients.md", "exec"), namespace)  # noqa: S102
    res = namespace["cresult"]
    assert np.isfinite(res.jacobian["CL"]["controls"]["flap"])


def _flap_hinge(deflection_deg: float, hinge_x_c: float) -> vt.ControlSurface:
    return vt.ControlSurface(name="flap", eta_start=0.2, eta_end=0.6, hinge_x_c=hinge_x_c,
                             deflection=np.radians(deflection_deg), symmetric=True)


def _swept_wing(controls=None) -> vt.LiftingSurface:
    """A swept, tapered wing (sweep 20 deg, taper 0.35) for the fine chordwise meshes."""
    af = _linear_section()
    w = vt.LiftingSurface(
        name="Wing", semi_span=5.0, sweep_le=np.radians(20.0), dihedral=np.radians(5.0),
        sections=[vt.WingSection(y_frac=0.0, chord=2.0, airfoil=af),
                  vt.WingSection(y_frac=1.0, chord=0.7, airfoil=af)])
    if controls:
        w.controls = list(controls)
    return w


_FINE_CHORD = [(12, "uniform", 0.75), (16, "cosine", 0.7)]


@pytest.mark.parametrize("deflection_deg", [0.0, 5.0])
@pytest.mark.parametrize("n_chord,spacing,hinge", _FINE_CHORD)
def test_controls_fine_chordwise_mesh(n_chord, spacing, hinge, deflection_deg, float64_cpu) -> None:
    """Regression: the chordwise-mode control gradient of a fine chordwise mesh passes Richardson.

    The chordwise fractions of the reference copy came from a recursion that
    amplified the round-off at each panel. At 12 uniform panels the hinge
    node then was 1e-12 forward of the hinge, the hinge panel dropped out
    of the flap panels, and the gradient was about 3 percent low.
    """
    ac = vt.Aircraft(surfaces=[_swept_wing(controls=[_flap_hinge(deflection_deg, hinge)])])
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    st = vt.SolverSettings(solver_type="vlm", n_panels=8, n_chord=n_chord, chord_spacing=spacing)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = vt.check_gradients(ac, cond, st, outputs=("CL", "Cm"), wrt=("controls",),
                                 n_directions=0, rtol=1e-6)
    assert out["groups"]["controls"]["compared"] > 0
    assert out["pass"], _failed(out)


@pytest.mark.parametrize("n_chord,spacing,hinge", [
    (4, "uniform", 0.75), (8, "uniform", 0.7), (12, "uniform", 0.75), (16, "cosine", 0.7),
    (24, "cosine", 0.75), (32, "cosine", 0.7), (32, "uniform", 0.73)])
def test_flap_panels_equal_production_rotated_panels(n_chord, spacing, hinge, float64_cpu) -> None:
    """The flap panels of the reference copy are the panels that the production mesh turns."""
    from ventorum.sensitivity import adjoint

    ac = vt.Aircraft(surfaces=[_swept_wing(controls=[_flap_hinge(5.0, hinge), _aileron()])])
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=_BETA)
    st = vt.SolverSettings(solver_type="vlm", n_panels=8, n_chord=n_chord, chord_spacing=spacing)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prep = adjoint._prepare_case(ac, cond, st, "body", ("controls",), ("CL",), "reference")
    lat = prep["lattice"]
    turned = np.flatnonzero(
        np.linalg.norm(lat.normal_bc - lat.normal_bc_strip[lat.panel_strip], axis=1) > 1e-12)
    model = sorted(prep["model"].b["ctrl_chord"]["panels"].cpu().numpy().tolist())
    assert model == turned.tolist()
    # The recovered chordwise fractions are the production fractions.
    xi = np.asarray(prep["model"].b["xi"].cpu())
    d = np.diff(xi, axis=1).reshape(-1) * np.asarray(lat.chord)[lat.panel_strip]
    assert np.array_equal(d, np.asarray(lat.panel_length))


@pytest.mark.parametrize("n_chord", [24, 32])
def test_fine_cosine_chordwise_gradient(n_chord, float64_cpu) -> None:
    """Regression: a fine cosine chordwise mesh gives a gradient that passes Richardson.

    The old recursion of the chordwise fractions gave a forward mismatch of
    2.6e-10 at 24 and 4.3e-6 at 32 cosine panels, and the gradient raised
    RuntimeError also without controls.
    """
    ac = vt.Aircraft(surfaces=[_swept_wing()])
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=_BETA)
    st = vt.SolverSettings(solver_type="vlm", n_panels=8, n_chord=n_chord, chord_spacing="cosine")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = vt.gradients(ac, cond, st, outputs=("CL", "Cl"), wrt=("alpha", "beta"))
        out = vt.check_gradients(ac, cond, st, outputs=("CL", "Cl"), wrt=("alpha", "beta"),
                                 n_directions=0, rtol=1e-6)
    assert res.forward_mismatch <= 1e-12
    assert out["pass"], _failed(out)


def test_residual_check_detects_a_wrong_boundary_condition(float64_cpu) -> None:
    """The backward error of the production state is round-off; a wrong right-hand side is refused."""
    import torch

    from ventorum.sensitivity import adjoint

    ac = vt.Aircraft(surfaces=[_wing(controls=[_flap()])])
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=_BETA)
    for st in (_settings("vlm", 4), _settings("linear"), _settings("nonlinear")):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prep = adjoint._prepare_case(ac, cond, st, "body", ("alpha",), ("CL",), "reference")
            res = vt.gradients(ac, cond, st, outputs=("CL",), wrt=("alpha",))
        assert res.forward_mismatch <= 1e-12
        model = prep["model"]
        t = adjoint.make_input_tensors(model, prep["base"], ("alpha",), prep["device"],
                                       prep["precision"], prep)
        g = torch.as_tensor(prep["g_unknown"], dtype=model.b["dtype"], device=model.b["device"])
        with torch.no_grad():
            _, A, rhs = model.forward(t, g)
        omega = adjoint._residual_check(prep, A, rhs, g)
        assert omega <= 1e-10, (st.solver_type, omega)
        # One boundary condition with an angle error of 0.01 deg [rad].
        bad = rhs.clone()
        bad[3] += float(np.radians(0.01)) * float(torch.max(torch.abs(rhs)))
        with pytest.raises(RuntimeError, match="does not solve the system"):
            adjoint._residual_check(prep, A, bad, g)
        # The same error in the base input of the reference copy (alpha): the
        # output check sees it too, but the residual check is independent.
        base = dict(prep["base"])
        base["alpha"] = float(base["alpha"]) + float(np.radians(0.01))
        t2 = adjoint.make_input_tensors(model, base, ("alpha",), prep["device"], prep["precision"], prep)
        with torch.no_grad():
            _, A2, rhs2 = model.forward(t2, g)
        with pytest.raises(RuntimeError, match="does not solve the system"):
            adjoint._residual_check(prep, A2, rhs2, g)
