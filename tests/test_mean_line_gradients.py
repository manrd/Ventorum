# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for adjoint gradients of vortex lattices with mean-line camber offsets."""

from __future__ import annotations

import numpy as np
import pytest

from ventorum.core.datatypes import (
    Aircraft,
    ControlSurface,
    FlightCondition,
    LiftingSurface,
    LinearAirfoil,
    SolverSettings,
    WingSection,
)
from ventorum.sensitivity import check_gradients, gradient_vjp, gradients
from ventorum.sensitivity.errors import GradientScopeError
import ventorum.gpu as gpu


@pytest.fixture(autouse=True)
def _float64_precision():
    old = gpu._precision
    gpu.set_precision("float64")
    yield
    gpu.set_precision(old)


def _cambered_wing(
    shape: str = "4412",
    *,
    semi_span: float = 4.0,
    taper: float = 0.6,
    dihedral_deg: float = 3.0,
    twist_tip_deg: float = -2.0,
    sweep_deg: float = 10.0,
    controls: list[ControlSurface] | None = None,
) -> Aircraft:
    """Build a cambered wing with taper, dihedral, twist and sweep."""
    wing = LiftingSurface(
        name="wing",
        semi_span=semi_span,
        sweep_le=np.radians(sweep_deg),
        dihedral=np.radians(dihedral_deg),
        sections=[
            WingSection(y_frac=0.0, chord=1.0, twist=0.0, shape=shape, airfoil=LinearAirfoil()),
            WingSection(
                y_frac=1.0,
                chord=taper,
                twist=np.radians(twist_tip_deg),
                shape=shape,
                airfoil=LinearAirfoil(),
            ),
        ],
    )
    if controls is not None:
        wing.controls = list(controls)
    return Aircraft(surfaces=[wing], ref_point=np.array([0.25, 0.0, 0.0]))


def _flap(deflection_deg: float = 0.0) -> ControlSurface:
    """Return a trailing-edge plain flap."""
    return ControlSurface(
        name="flap",
        eta_start=0.2,
        eta_end=0.6,
        hinge_x_c=0.75,
        deflection=np.radians(deflection_deg),
        symmetric=True,
    )


def _assert_finite(obj: object) -> None:
    """Assert that every leaf in a nested dict or sequence is finite."""
    if obj is None:
        return
    if isinstance(obj, dict):
        for v in obj.values():
            _assert_finite(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _assert_finite(v)
    else:
        assert np.all(np.isfinite(obj))


def test_no_old_refusal_for_cambered_mean_line():
    """Ensure the old blanket refusal for mean-line offset lattices is removed."""
    ac = _cambered_wing()
    cond = FlightCondition(V_inf=30.0, alpha=0.05)
    st = SolverSettings(n_panels=4, n_chord=4)
    res = gradients(ac, cond, st, wrt=("alpha",), outputs=("CL",))
    assert np.isfinite(res.jacobian["CL"]["alpha"])


@pytest.mark.parametrize("h", [None, 0.3])
def test_cambered_wing_alpha_and_nodes_check_gradients(h):
    """Check gradients of alpha and nodes in free air and in ground effect."""
    ac = _cambered_wing()
    cond = FlightCondition(V_inf=30.0, alpha=0.05, h=h)
    st = SolverSettings(n_panels=4, n_chord=4)
    report = check_gradients(
        ac, cond, st, wrt=("alpha", "nodes"), outputs=("CL", "CDi", "Cm"), rtol=1e-4
    )
    assert report["pass"], report.get("reasons")


@pytest.mark.parametrize("h", [None, 0.3])
def test_cambered_wing_planform_check_gradients(h):
    """Check gradients of planform parameters at zero control deflection."""
    ac = _cambered_wing(controls=[_flap(0.0)])
    cond = FlightCondition(V_inf=30.0, alpha=0.05, h=h)
    st = SolverSettings(n_panels=4, n_chord=4)
    report = check_gradients(
        ac, cond, st, wrt=("planform",), outputs=("CL", "CDi", "Cm"), rtol=1e-4
    )
    assert report["pass"], report.get("reasons")


@pytest.mark.parametrize("h", [None, 0.3])
def test_cambered_wing_deflected_flap_check_gradients(h):
    """Check gradients of alpha, nodes and controls with a deflected flap (10 deg)."""
    ac = _cambered_wing(controls=[_flap(10.0)])
    cond = FlightCondition(V_inf=30.0, alpha=0.05, h=h)
    st = SolverSettings(n_panels=4, n_chord=4)
    report = check_gradients(
        ac, cond, st, wrt=("alpha", "nodes", "controls"), outputs=("CL", "CDi", "Cm"), rtol=1e-4
    )
    assert report["pass"], report.get("reasons")


def test_span_varying_mean_lines_check_gradients():
    """Check gradients for a wing with different shapes at root and tip."""
    wing = LiftingSurface(
        name="wing",
        semi_span=4.0,
        sections=[
            WingSection(y_frac=0.0, chord=1.0, shape="4412", airfoil=LinearAirfoil()),
            WingSection(y_frac=1.0, chord=0.6, shape="2412", airfoil=LinearAirfoil()),
        ],
    )
    ac = Aircraft(surfaces=[wing], ref_point=np.array([0.25, 0.0, 0.0]))
    cond = FlightCondition(V_inf=30.0, alpha=0.05)
    st = SolverSettings(n_panels=4, n_chord=4)
    report = check_gradients(
        ac, cond, st, wrt=("alpha", "planform"), outputs=("CL", "CDi", "Cm"), rtol=1e-4
    )
    assert report["pass"], report.get("reasons")


def test_joined_surfaces_averaged_camber_check_gradients():
    """Check gradients for joined surfaces sharing an edge with averaged offsets."""
    inner = LiftingSurface(
        name="inner",
        semi_span=2.0,
        is_symmetric=False,
        sections=[
            WingSection(y_frac=0.0, chord=1.0, shape="4412", airfoil=LinearAirfoil()),
            WingSection(y_frac=1.0, chord=0.8, shape="4412", airfoil=LinearAirfoil()),
        ],
    )
    outer = LiftingSurface(
        name="outer",
        semi_span=2.0,
        is_symmetric=False,
        position=np.array([0.0, 2.0, 0.0]),
        sections=[
            WingSection(y_frac=0.0, chord=0.8, shape="2412", airfoil=LinearAirfoil()),
            WingSection(y_frac=1.0, chord=0.5, shape="0012", airfoil=LinearAirfoil()),
        ],
    )
    ac = Aircraft(surfaces=[inner, outer, inner.mirrored(), outer.mirrored()], ref_point=np.array([0.25, 0.0, 0.0]))
    cond = FlightCondition(V_inf=30.0, alpha=0.05)
    st = SolverSettings(n_panels=4, n_chord=4)
    report = check_gradients(
        ac, cond, st, wrt=("alpha", "nodes"), outputs=("CL", "CDi", "Cm"), rtol=1e-4
    )
    assert report["pass"], report.get("reasons")


def test_gradient_vjp_matches_gradients_to_roundoff():
    """Verify gradient_vjp against gradients Jacobian product down to round-off."""
    ac = _cambered_wing(controls=[_flap(5.0)])
    cond = FlightCondition(V_inf=30.0, alpha=0.05)
    st = SolverSettings(n_panels=4, n_chord=4)

    seeds = {"CL": 1.5, "Cm": -0.8}
    vjp_res = gradient_vjp(ac, cond, st, seeds=seeds, wrt=("alpha", "controls"))
    grad_res = gradients(ac, cond, st, outputs=("CL", "Cm"), wrt=("alpha", "controls"))

    # Alpha parity:
    expected_alpha = (
        seeds["CL"] * grad_res.jacobian["CL"]["alpha"]
        + seeds["Cm"] * grad_res.jacobian["Cm"]["alpha"]
    )
    np.testing.assert_allclose(vjp_res["alpha"], expected_alpha, rtol=1e-12, atol=1e-14)

    # Control parity:
    expected_ctrl = (
        seeds["CL"] * grad_res.jacobian["CL"]["controls"]["flap"]
        + seeds["Cm"] * grad_res.jacobian["Cm"]["controls"]["flap"]
    )
    np.testing.assert_allclose(
        vjp_res["controls"]["flap"], expected_ctrl, rtol=1e-12, atol=1e-14
    )


@pytest.mark.parametrize("backend", ["numpy", "numba", "cython", "torch"])
def test_vjp_backends_agree_with_torch_path(backend):
    """Every VJP backend gives the same gradients as the reference torch path."""
    from ventorum.sensitivity.kernels import available, set_vjp_backend

    if not available(backend, "cpu"):
        pytest.skip(f"Backend '{backend}' is not installed or available.")

    ac = _cambered_wing()
    cond = FlightCondition(V_inf=30.0, alpha=0.05)
    st = SolverSettings(n_panels=4, n_chord=4)

    g_torch = gradients(ac, cond, st, outputs=("CL", "Cm"), wrt=("alpha",), method="reference")
    try:
        set_vjp_backend(backend)
        g_backend = gradients(ac, cond, st, outputs=("CL", "Cm"), wrt=("alpha",), method="kernels" if backend != "torch" else "reference")
    finally:
        set_vjp_backend(None)

    for out in ("CL", "Cm"):
        np.testing.assert_allclose(
            g_backend.jacobian[out]["alpha"],
            g_torch.jacobian[out]["alpha"],
            rtol=1e-10,
            atol=1e-12,
        )


def test_refusal_sections_on_shaped_surface():
    """Input 'sections' on a shaped surface must still raise GradientScopeError."""
    ac = _cambered_wing()
    cond = FlightCondition(V_inf=30.0, alpha=0.05)
    st = SolverSettings(n_panels=4, n_chord=4)
    with pytest.raises(GradientScopeError, match="Input 'sections'"):
        gradients(ac, cond, st, wrt=("sections",), outputs=("CL",))


def test_mixed_surface_uses_old_chain_and_passes_check_gradients():
    """A mixed surface (one section shaped, one unshaped) falls back to turned normals."""
    wing = LiftingSurface(
        name="wing",
        semi_span=4.0,
        sections=[
            WingSection(y_frac=0.0, chord=1.0, shape="4412", airfoil=LinearAirfoil(alpha_L0=-0.03)),
            WingSection(y_frac=1.0, chord=0.6, shape=None, airfoil=LinearAirfoil(alpha_L0=-0.03)),
        ],
    )
    ac = Aircraft(surfaces=[wing], ref_point=np.array([0.25, 0.0, 0.0]))
    cond = FlightCondition(V_inf=30.0, alpha=0.05)
    st = SolverSettings(n_panels=4, n_chord=4)
    report = check_gradients(
        ac, cond, st, wrt=("alpha", "nodes"), outputs=("CL", "Cm"), rtol=1e-4
    )
    assert report["pass"], report.get("reasons")
