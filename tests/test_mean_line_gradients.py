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
    old_device, old = gpu.get_device(), gpu._precision
    gpu.set_device("cpu")
    gpu.set_precision("float64")
    yield
    gpu.set_precision(old)
    gpu.set_device(old_device)


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
        ac, cond, st, wrt=("alpha", "nodes"), outputs=("CL", "CDi", "Cm")
    )
    assert report["pass"], report.get("reasons")


@pytest.mark.parametrize("h", [None, 0.3])
def test_cambered_wing_planform_check_gradients(h):
    """Check gradients of planform parameters at zero control deflection."""
    ac = _cambered_wing(controls=[_flap(0.0)])
    cond = FlightCondition(V_inf=30.0, alpha=0.05, h=h)
    st = SolverSettings(n_panels=4, n_chord=4)
    report = check_gradients(
        ac, cond, st, wrt=("planform",), outputs=("CL", "CDi", "Cm")
    )
    assert report["pass"], report.get("reasons")


@pytest.mark.parametrize("h", [None, 0.3])
def test_cambered_wing_deflected_flap_check_gradients(h):
    """Check gradients of alpha, nodes and controls with a deflected flap (10 deg)."""
    ac = _cambered_wing(controls=[_flap(10.0)])
    cond = FlightCondition(V_inf=30.0, alpha=0.05, h=h)
    st = SolverSettings(n_panels=4, n_chord=4)
    report = check_gradients(
        ac, cond, st, wrt=("alpha", "nodes", "controls"), outputs=("CL", "CDi", "Cm")
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
        ac, cond, st, wrt=("alpha", "planform"), outputs=("CL", "CDi", "Cm")
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
        ac, cond, st, wrt=("alpha", "nodes"), outputs=("CL", "CDi", "Cm")
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


@pytest.mark.parametrize("backend", ["numpy", "numba", "cython", "torch",
                                     pytest.param("warp", marks=pytest.mark.gpu)])
def test_vjp_backends_agree_with_torch_path(backend):
    """Every VJP backend gives the same gradients as the reference torch path."""
    from ventorum.sensitivity.kernels import available, set_vjp_backend

    device = "cuda" if backend == "warp" else "cpu"
    if backend == "warp" and not gpu.available():
        pytest.skip(f"The GPU pipelines cannot run: {gpu.unavailable_reason()}")
    if not available(backend, device):
        pytest.skip(f"Backend '{backend}' is not installed or available.")
    gpu.set_device("gpu" if backend == "warp" else "cpu")

    ac = _cambered_wing(controls=[_flap(10.0)])
    cond = FlightCondition(V_inf=30.0, alpha=0.05)
    st = SolverSettings(n_panels=4, n_chord=4)

    wrt = ("alpha", "nodes", "controls")
    g_torch = gradients(ac, cond, st, outputs=("CL", "Cm"), wrt=wrt, method="reference")
    try:
        set_vjp_backend(backend)
        g_backend = gradients(ac, cond, st, outputs=("CL", "Cm"), wrt=wrt, method="kernels" if backend != "torch" else "reference")
    finally:
        set_vjp_backend(None)

    for out in ("CL", "Cm"):
        jac = g_torch.jacobian[out]
        # The absolute tolerance follows the size of the Jacobian (round-off
        # factor times its largest element), as in the other parity tests:
        # the backends sum the chain terms in a different order, and a fixed
        # absolute tolerance would be below the round-off of a small element
        # (1.6e-14 on an element of -0.0018 in the numpy backend).
        scale = max(float(np.max(np.abs(jac["alpha"]))), float(np.max(np.abs(jac["controls"]["flap"]))),
                    *(float(np.max(np.abs(jac["nodes"][s][e]))) for s in jac["nodes"] for e in ("le", "te")))
        atol = 1e-12 * scale
        np.testing.assert_allclose(
            g_backend.jacobian[out]["alpha"],
            jac["alpha"],
            rtol=1e-12,
            atol=atol,
        )
        for surface in jac["nodes"]:
            for edge in ("le", "te"):
                ref = jac["nodes"][surface][edge]
                new = g_backend.jacobian[out]["nodes"][surface][edge]
                np.testing.assert_allclose(new, ref, rtol=1e-12, atol=atol)
        np.testing.assert_allclose(
            g_backend.jacobian[out]["controls"]["flap"],
            jac["controls"]["flap"],
            rtol=1e-12, atol=atol,
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
        ac, cond, st, wrt=("alpha", "nodes"), outputs=("CL", "Cm")
    )
    assert report["pass"], report.get("reasons")


@pytest.mark.parametrize("h_over_c", [None, 0.3])
@pytest.mark.parametrize("wrt", [("alpha", "nodes"), ("planform",)])
def test_three_section_mean_line_span_fractions(h_over_c, wrt):
    """Carry section span fractions and moving stations through the mean line."""
    ac = _cambered_wing()
    ac.surfaces[0].sections.insert(
        1, WingSection(y_frac=0.5, chord=0.85, twist=np.radians(-0.7), shape="4412"),
    )
    ac.surfaces[0].sections[-1].shape = "2412"
    ac.compute_reference_values()
    h = None if h_over_c is None else h_over_c * ac.c_ref
    report = check_gradients(
        ac, FlightCondition(V_inf=30.0, alpha=0.05, h=h),
        SolverSettings(n_panels=4, n_chord=4), wrt=wrt, outputs=("CL", "CDi", "Cm"),
    )
    assert report["pass"], report
    if "planform" in wrt:
        for output in ("CL", "CDi", "Cm"):
            info = report["details"][f"{output}:planform.wing.sec[1].y_frac"]
            assert info["compared"] and info["passed"], info


def test_shaped_lifting_line_accepts_section_inputs():
    """The lifting line keeps alpha_L0 as an independent section input [rad]."""
    ac = _cambered_wing()
    for section in ac.surfaces[0].sections:
        section.airfoil = LinearAirfoil(alpha_L0=-0.03, Cm0=-0.04)
    st = SolverSettings(solver_type="linear", n_panels=4)
    cond = FlightCondition(V_inf=30.0, alpha=0.05)
    result = gradients(ac, cond, st, wrt=("sections",), outputs=("CL", "CDi", "Cm"))
    _assert_finite(result.jacobian)
    report = check_gradients(ac, cond, st, wrt=("sections",), outputs=("CL", "CDi", "Cm"))
    assert report["pass"], report


def test_joined_mean_lines_with_deflected_flap():
    """Turn a chordwise flap after the shared mean-line offsets are averaged."""
    inner = LiftingSurface(
        name="inner", semi_span=2.0, is_symmetric=True,
        sections=[WingSection(y_frac=0.0, chord=1.0, shape="4412"),
                  WingSection(y_frac=1.0, chord=0.8, shape="4412")],
        controls=[ControlSurface(name="flap", eta_start=0.2, eta_end=0.6,
                                 hinge_x_c=0.75, deflection=np.radians(10.0))],
    )
    outer = LiftingSurface(
        name="outer", semi_span=2.0, is_symmetric=False, position=np.array([0.0, 2.0, 0.0]),
        sections=[WingSection(y_frac=0.0, chord=0.8, shape="2412"),
                  WingSection(y_frac=1.0, chord=0.5, shape="0012")],
    )
    ac = Aircraft(surfaces=[inner, outer, outer.mirrored()],
                  ref_point=np.array([0.25, 0.0, 0.0]))
    report = check_gradients(
        ac, FlightCondition(V_inf=30.0, alpha=0.05), SolverSettings(n_panels=4, n_chord=4),
        wrt=("alpha", "nodes", "controls"), outputs=("CL", "CDi", "Cm"),
    )
    assert report["pass"], report


def _folded_wing(dihedral_deg: float = 30.0) -> Aircraft:
    """Return a short cambered wing whose root strip lies inside the fold blend of the mean-line offset."""
    wing = LiftingSurface(
        name="wing", semi_span=1.0, dihedral=np.radians(dihedral_deg), spacing="cosine",
        sections=[WingSection(y_frac=0.0, chord=0.5, shape="6409", airfoil=LinearAirfoil()),
                  WingSection(y_frac=1.0, chord=0.3, twist=np.radians(-2.0), shape="6409", airfoil=LinearAirfoil())],
    )
    return Aircraft(surfaces=[wing], ref_point=np.array([0.1, 0.0, 0.0]))


def _folded_junction() -> Aircraft:
    """Return two joined cambered surfaces at a dihedral break, with their mirror copies."""
    inner = LiftingSurface(
        name="inner", semi_span=0.6, is_symmetric=False, dihedral=np.radians(5.0), spacing="cosine",
        sections=[WingSection(y_frac=0.0, chord=0.5, shape="6409", airfoil=LinearAirfoil()),
                  WingSection(y_frac=1.0, chord=0.4, shape="6409", airfoil=LinearAirfoil())],
    )
    outer = LiftingSurface(
        name="outer", semi_span=0.5, is_symmetric=False, dihedral=np.radians(35.0), spacing="cosine",
        position=np.array([0.0, 0.6 * np.cos(np.radians(5.0)), 0.6 * np.sin(np.radians(5.0))]),
        sections=[WingSection(y_frac=0.0, chord=0.4, shape="4412", airfoil=LinearAirfoil()),
                  WingSection(y_frac=1.0, chord=0.25, shape="4412", airfoil=LinearAirfoil())],
    )
    return Aircraft(surfaces=[inner, outer, inner.mirrored(), outer.mirrored()], ref_point=np.array([0.1, 0.0, 0.0]))


@pytest.mark.parametrize("make", [_folded_wing, _folded_junction])
@pytest.mark.parametrize("h", [None, 0.4])
def test_camber_fold_blend_reference_matches_production(make, h):
    """The torch reference follows the fold blend and the last-panel leg node to round-off."""
    from ventorum.geometry.lattice import build_lattice

    ac = make()
    st = SolverSettings(n_panels=8, n_chord=4)
    lat = build_lattice(ac, st, n_chord=4)
    folds = [f for frame in lat.camber_frames for f in frame["folds"]]
    assert folds
    res = gradients(ac, FlightCondition(V_inf=30.0, alpha=0.05, beta=0.03, h=h), st,
                    wrt=("alpha", "nodes"), outputs=("CL", "CDi", "Cm", "CY", "Cl"))
    assert res.forward_mismatch < 1e-11


@pytest.mark.parametrize("make", [_folded_wing, _folded_junction])
def test_camber_fold_blend_check_gradients(make):
    """Check the node and planform gradients with strips inside the fold blend."""
    ac = make()
    cond = FlightCondition(V_inf=30.0, alpha=0.05, beta=0.03)
    st = SolverSettings(n_panels=8, n_chord=4)
    for wrt in (("alpha", "nodes"), ("planform",)):
        report = check_gradients(ac, cond, st, wrt=wrt, outputs=("CL", "CDi", "Cm", "CY", "Cl"))
        assert report["pass"], report.get("reasons")
        assert report["groups"][wrt[-1]]["compared"] > 0
