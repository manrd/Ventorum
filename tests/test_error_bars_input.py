# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests of the layer-2 input-uncertainty error bars (linear propagation)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

import ventorum as vt
from ventorum import gpu
from ventorum.core.error_bars import COEFFICIENT_NAMES


@pytest.fixture
def float64_precision():
    """Run the test with float64 gradient precision; restore it after it."""
    old = gpu.get_precision()
    gpu.set_precision("float64")
    yield
    gpu.set_precision(old)


@pytest.fixture
def cpu_device():
    """Run the test on the CPU; restore the device setting after it."""
    old = gpu.get_device()
    gpu.set_device("cpu")
    yield
    gpu.set_device(old)


def _plain_wing(name: str = "Wing") -> vt.LiftingSurface:
    return vt.LiftingSurface(
        name=name,
        semi_span=5.0,
        sections=[
            vt.WingSection(y_frac=0.0, chord=2.0),
            vt.WingSection(y_frac=1.0, chord=1.0),
        ],
    )


def _condition() -> vt.FlightCondition:
    return vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))


def _settings() -> vt.SolverSettings:
    return vt.SolverSettings(solver_type="vlm", n_panels=12, n_chord=2)


def test_single_input_equals_gradient_times_sigma(float64_precision):
    """Only alpha uncertain: half-width = k |dCL/dalpha| u_alpha (1e-10)."""
    wing = _plain_wing()
    cond, st = _condition(), _settings()
    sigma = np.radians(0.1)
    result = vt.input_error_bars(
        wing, condition=cond, settings=st,
        uncertainties={"alpha": sigma}, coverage_factor=2.0)
    grad = vt.gradients(wing, cond, st, outputs=("CL",), wrt=("alpha",))
    expected = 2.0 * abs(float(grad.jacobian["CL"]["alpha"])) * sigma
    layer = result.bars["CL"].layers["input"]
    assert layer.half_width == pytest.approx(expected, rel=1e-10)
    assert result.bars["CL"].status == "input_only"
    assert result.bars["CL"].coverage is None
    assert result.bars["CL"].input_shares == {"alpha": pytest.approx(1.0)}
    assert result.bars["CL"].gradient_methods == {"alpha": "adjoint"}


def test_independent_sum_and_shares(float64_precision):
    """Two independent inputs: u_c^2 is the sum of the squares; shares sum to 1."""
    wing = _plain_wing()
    cond, st = _condition(), _settings()
    unc = {"alpha": np.radians(0.1), "sections": {"Wing": {"a0": 0.05}}}
    result = vt.input_error_bars(
        wing, condition=cond, settings=st, uncertainties=unc, coverage_factor=2.0)
    grad = vt.gradients(
        wing, cond, st, outputs=("CL",), wrt=("alpha", "sections"), axes="body")
    n_strips = np.asarray(grad.jacobian["CL"]["sections"]["a0"]).size
    d_alpha = float(grad.jacobian["CL"]["alpha"])
    d_a0 = float(np.sum(np.asarray(grad.jacobian["CL"]["sections"]["a0"]).reshape(-1)))
    assert n_strips > 1
    expected_var = (d_alpha * unc["alpha"]) ** 2 + (d_a0 * 0.05) ** 2
    layer = result.bars["CL"].layers["input"]
    assert (layer.half_width / 2.0) ** 2 == pytest.approx(expected_var, rel=1e-10)
    shares = result.bars["CL"].input_shares
    assert sum(shares.values()) == pytest.approx(1.0)
    assert shares["alpha"] == pytest.approx((d_alpha * unc["alpha"]) ** 2 / expected_var)
    assert shares["sections.Wing.a0"] == pytest.approx((d_a0 * 0.05) ** 2 / expected_var)


def test_correlation(float64_precision):
    """Alpha and V_inf with rho = 0.5: u_c^2 equals J U J^T of the test."""
    wing = _plain_wing()
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=np.radians(3.0))
    st = _settings()
    sigmas = {"alpha": np.radians(0.1), "beta": np.radians(0.2)}
    rho = 0.5
    result = vt.input_error_bars(
        wing, condition=cond, settings=st,
        uncertainties=dict(sigmas),
        correlations={("alpha", "beta"): rho}, coverage_factor=2.0)
    grad = vt.gradients(
        wing, cond, st, outputs=tuple(COEFFICIENT_NAMES),
        wrt=("alpha", "beta"), axes="body")
    order = ["alpha", "beta"]
    sigma = np.array([sigmas[k] for k in order])
    cov = np.diag(sigma ** 2)
    cov[0, 1] += rho * sigma[0] * sigma[1]
    cov[1, 0] += rho * sigma[0] * sigma[1]
    for name in COEFFICIENT_NAMES:
        jac = np.array([float(grad.jacobian[name][k]) for k in order])
        expected = float(jac @ cov @ jac)
        layer = result.bars[name].layers["input"]
        assert (layer.half_width / 2.0) ** 2 == pytest.approx(expected, rel=1e-10, abs=1e-30)


def test_monte_carlo_agreement(float64_precision):
    """400 seeded samples: sample std of CL and Cm agrees with u_c within 15 %."""
    wing = _plain_wing()
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    st = vt.SolverSettings(solver_type="vlm", n_panels=10, n_chord=1)
    sigma_alpha = np.radians(0.05)
    sigma_v = 0.1
    sigma_a0 = 0.05
    unc = {"alpha": sigma_alpha, "V_inf": sigma_v,
           "sections": {"Wing": {"a0": sigma_a0}}}
    result = vt.input_error_bars(
        wing, condition=cond, settings=st, uncertainties=unc, coverage_factor=2.0)
    base_a0 = [float(sec.airfoil.a0) for sec in wing.sections]
    rng = np.random.default_rng(7)
    n_samples = 400
    cls, cms = np.empty(n_samples), np.empty(n_samples)
    for k in range(n_samples):
        surf = wing.clone()
        c = cond.clone()
        c.alpha = float(cond.alpha + rng.standard_normal() * sigma_alpha)
        c.V_inf = float(cond.V_inf + rng.standard_normal() * sigma_v)
        shift = float(rng.standard_normal() * sigma_a0)
        for sec, a0 in zip(surf.sections, base_a0):
            sec.airfoil.a0 = a0 + shift
        res = vt.analyze(surf, condition=c, settings=st.clone())
        cls[k] = float(res.totals.CL)
        cms[k] = float(res.moments("body")["Cm"])
    for name, samples in (("CL", cls), ("Cm", cms)):
        sample_std = float(np.std(samples, ddof=1))
        u_c = float(result.bars[name].layers["input"].half_width) / 2.0
        assert sample_std == pytest.approx(u_c, rel=0.15)


def test_section_correlated_along_span(float64_precision):
    """A section uncertainty uses the sum of the strip gradients."""
    wing = _plain_wing()
    cond, st = _condition(), _settings()
    sigma = 0.05
    result = vt.input_error_bars(
        wing, condition=cond, settings=st,
        uncertainties={"sections": {"Wing": {"a0": sigma}}}, coverage_factor=2.0)
    grad = vt.gradients(
        wing, cond, st, outputs=("CL",), wrt=("sections",), axes="body")
    arr = np.asarray(grad.jacobian["CL"]["sections"]["a0"], dtype=float).reshape(-1)
    expected = 2.0 * abs(float(np.sum(arr))) * sigma
    assert result.bars["CL"].layers["input"].half_width == pytest.approx(expected, rel=1e-10)


def _tabulated_wing() -> vt.LiftingSurface:
    deg = np.array([-5.0, 0.0, 5.0, 10.0])
    tabulated = vt.TabulatedAirfoil(
        name="tab",
        alpha=np.radians(deg),
        Cl_data=np.array([-0.5, 0.0, 0.5, 1.0]),
        Cd_data=np.array([0.010, 0.008, 0.010, 0.015]),
    )
    return vt.LiftingSurface(
        name="tab",
        semi_span=5.0,
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.25, airfoil=tabulated),
            vt.WingSection(y_frac=1.0, chord=1.25, airfoil=tabulated),
        ],
    )


def test_fallback_central_differences(float64_precision):
    """Nonlinear lifting line: flow inputs use central differences; sections refused."""
    wing = _tabulated_wing()
    cond = vt.FlightCondition(alpha=np.radians(5.0))
    st = vt.SolverSettings(solver_type="nonlinear", n_panels=16)
    sigma = np.radians(0.1)
    result = vt.input_error_bars(
        wing, condition=cond, settings=st,
        uncertainties={"alpha": sigma}, coverage_factor=2.0)
    assert result.bars["CL"].gradient_methods == {"alpha": "central differences"}
    assert "central differences" in result.bars["CL"].layers["input"].method
    step = np.radians(0.5)
    plus = vt.analyze(
        wing.clone(),
        condition=vt.FlightCondition(alpha=np.radians(5.0) + step),
        settings=st.clone())
    minus = vt.analyze(
        wing.clone(),
        condition=vt.FlightCondition(alpha=np.radians(5.0) - step),
        settings=st.clone())
    manual = (float(plus.totals.CL) - float(minus.totals.CL)) / (2.0 * step)
    expected = 2.0 * abs(manual) * sigma
    assert result.bars["CL"].layers["input"].half_width == pytest.approx(expected, rel=1e-8)
    with pytest.raises(ValueError, match="Section uncertainties need adjoint gradients"):
        vt.input_error_bars(
            wing, condition=cond, settings=st,
            uncertainties={"sections": {"tab": {"a0": 0.1}}})


def test_combine_rss(float64_precision):
    """Combined half-width = sqrt(h_num^2 + h_input^2); no numerical interval stays empty."""
    wing = _plain_wing()
    cond, st = _condition(), _settings()
    numerical = vt.numerical_error_bars(
        wing, condition=cond,
        settings=vt.SolverSettings(solver_type="vlm", n_panels=24, n_chord=2))
    inputs = vt.input_error_bars(
        wing, condition=cond, settings=st,
        uncertainties={"alpha": np.radians(0.1)}, coverage_factor=2.0)
    combined = vt.combine_error_bars(numerical, inputs)
    for name, bar in combined.bars.items():
        assert bar.status == "numerical_and_input"
        assert bar.coverage is None
        h_num = numerical.bars[name].layers["numerical"].half_width
        h_in = inputs.bars[name].layers["input"].half_width
        assert bar.value == pytest.approx(numerical.bars[name].value)
        assert bar.half_width == pytest.approx(math.sqrt(h_num ** 2 + h_in ** 2))
        assert bar.interval_low == pytest.approx(bar.value - bar.half_width)
        assert bar.interval_high == pytest.approx(bar.value + bar.half_width)
        assert set(bar.layers) == {"numerical", "input"}
    assert combined.axes == numerical.axes
    # A layer-1 bar with no interval gives no combined interval and a note.
    divergent = vt.numerical_error_bars(
        wing, condition=cond,
        settings=vt.SolverSettings(solver_type="vlm", n_panels=24, n_chord=2))
    names = [n for n, b in divergent.bars.items() if b.interval_low is None]
    if names:
        name = names[0]
    else:
        from ventorum.core.error_bars import LayerContribution

        name = "CL"
        bad = divergent.bars[name]
        bad.layers["numerical"] = LayerContribution(
            half_width=None, state="divergent",
            levels=list(bad.layers["numerical"].levels),
            values=list(bad.layers["numerical"].values))
        bad.interval_low = None
        bad.interval_high = None
    combo = vt.combine_error_bars(divergent, inputs)
    bar = combo.bars[name]
    assert bar.interval_low is None and bar.interval_high is None
    assert any("no combined interval" in n for n in bar.notes)


def test_refusals(float64_precision):
    """Negative, non-finite or unknown keys, h in free air, unknown surfaces: ValueError."""
    wing = _plain_wing()
    cond, st = _condition(), _settings()
    with pytest.raises(ValueError, match="never negative"):
        vt.input_error_bars(wing, condition=cond, settings=st, uncertainties={"alpha": -0.01})
    with pytest.raises(ValueError, match="finite"):
        vt.input_error_bars(wing, condition=cond, settings=st, uncertainties={"alpha": math.inf})
    with pytest.raises(ValueError, match="alpha"):
        vt.input_error_bars(wing, condition=cond, settings=st, uncertainties={"gamma": 0.01})
    with pytest.raises(ValueError, match="ground effect"):
        vt.input_error_bars(wing, condition=cond, settings=st, uncertainties={"h": 0.1})
    with pytest.raises(ValueError, match="Wing"):
        vt.input_error_bars(
            wing, condition=cond, settings=st,
            uncertainties={"sections": {"Tail": {"a0": 0.1}}})
    with pytest.raises(ValueError, match="a0"):
        vt.input_error_bars(
            wing, condition=cond, settings=st,
            uncertainties={"sections": {"Wing": {"stall": 0.1}}})
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        vt.input_error_bars(
            wing, condition=cond, settings=st,
            uncertainties={"alpha": 0.01, "beta": 0.01},
            correlations={("alpha", "beta"): 1.5})


def test_repeated_correlation_pair_refused(float64_precision):
    """The pairs (a, b) and (b, a) in one dict are refused; they must not add up."""
    wing = _plain_wing()
    cond, st = _condition(), _settings()
    with pytest.raises(ValueError, match="more than once"):
        vt.input_error_bars(
            wing, condition=cond, settings=st,
            uncertainties={"alpha": 0.01, "beta": 0.01},
            correlations={("alpha", "beta"): 0.4, ("beta", "alpha"): 0.4})


def test_covariance_not_positive_semidefinite_refused(float64_precision):
    """Correlations that give a covariance with a negative eigenvalue are refused."""
    wing = _plain_wing()
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=np.radians(3.0))
    st = _settings()
    corr = {("alpha", "beta"): 0.9, ("alpha", "phi"): 0.9, ("beta", "phi"): -0.9}
    with pytest.raises(ValueError, match="positive semi-definite") as info:
        vt.input_error_bars(
            wing, condition=cond, settings=st,
            uncertainties={"alpha": 0.01, "beta": 0.01, "phi": 0.01},
            correlations=corr)
    for first, second in corr:
        assert f"('{first}', '{second}')" in str(info.value)
    # The same correlations with one input without uncertainty have no effect.
    vt.input_error_bars(
        wing, condition=cond, settings=st,
        uncertainties={"alpha": 0.01, "beta": 0.01}, correlations=corr)


def test_to_dict_json(float64_precision):
    """to_dict of layer-2 and combined results is strict JSON."""
    wing = _plain_wing()
    cond, st = _condition(), _settings()
    layer2 = vt.input_error_bars(
        wing, condition=cond, settings=st,
        uncertainties={"alpha": np.radians(0.1)}, coverage_factor=2.0)
    text = json.dumps(layer2.to_dict(), allow_nan=False)
    assert json.loads(text)["bars"]["CL"]["status"] == "input_only"
    numerical = vt.numerical_error_bars(
        wing, condition=cond,
        settings=vt.SolverSettings(solver_type="vlm", n_panels=24, n_chord=2))
    combined = vt.combine_error_bars(numerical, vt.input_error_bars(
        wing, condition=cond, settings=st,
        uncertainties={"alpha": np.radians(0.1)}, coverage_factor=2.0))
    text = json.dumps(combined.to_dict(), allow_nan=False)
    assert json.loads(text)["bars"]["CL"]["status"] == "numerical_and_input"
    assert "| CL |" in combined.summary()


def test_float32_device(float64_precision, cpu_device):
    """CPU float32 half-widths agree with float64 within 1e-3 relative."""
    wing = _plain_wing()
    cond, st = _condition(), _settings()
    unc = {"alpha": np.radians(0.5), "sections": {"Wing": {"a0": 0.1}}}
    ref = vt.input_error_bars(wing, condition=cond, settings=st, uncertainties=unc)
    old = gpu.get_precision()
    gpu.set_precision("float32")
    try:
        got = vt.input_error_bars(wing, condition=cond, settings=st, uncertainties=unc)
    finally:
        gpu.set_precision(old)
    for name in ("CL", "CD", "Cm"):
        h64 = float(ref.bars[name].layers["input"].half_width)
        h32 = float(got.bars[name].layers["input"].half_width)
        assert h32 == pytest.approx(h64, rel=1e-3)
