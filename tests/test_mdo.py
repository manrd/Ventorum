# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests of the OpenMDAO components of ``ventorum.mdo`` (optional extra ``mdo``).

The tests that build an OpenMDAO problem skip when OpenMDAO is not
installed. The import test hides OpenMDAO and runs everywhere.
"""

from __future__ import annotations

import contextlib
import importlib
import runpy
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import ventorum as vt
from ventorum import gpu

ROOT = Path(__file__).resolve().parents[1]


@contextlib.contextmanager
def _gpu_settings(device: str | None = None, precision: str | None = None):
    """Set the device and the precision; restore the stored settings after the block."""
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
def float64_precision():
    """Run the test with float64 precision; restore the setting after it."""
    with _gpu_settings(precision="float64"):
        yield


@pytest.fixture
def gpu_float32():
    """Run the test on the GPU in float32; skip when the GPU cannot run."""
    if not gpu.available():
        pytest.skip(f"The GPU pipelines cannot run: {gpu.unavailable_reason()}")
    with _gpu_settings(device="gpu", precision="float32"):
        yield


@pytest.fixture(autouse=True)
def _openmdao_workdir(tmp_path, monkeypatch):
    """Keep the output folders of OpenMDAO (reports) out of the repository."""
    monkeypatch.setenv("OPENMDAO_WORKDIR", str(tmp_path))
    monkeypatch.setenv("OPENMDAO_REPORTS", "0")


def _om():
    return pytest.importorskip("openmdao.api")


def _wing(name: str = "Wing", controls: bool = False) -> vt.LiftingSurface:
    foil = vt.LinearAirfoil(name="MdoSec", a0=2.0 * np.pi, alpha_L0=np.radians(-2.0), Cd0=0.008, Cm0=-0.05)
    wing = vt.LiftingSurface(
        name=name,
        semi_span=5.0,
        sections=[
            vt.WingSection(y_frac=0.0, chord=2.0, airfoil=foil),
            vt.WingSection(y_frac=1.0, chord=1.0, airfoil=foil),
        ],
    )
    if controls:
        wing.controls = [
            vt.ControlSurface(name="flap", eta_start=0.2, eta_end=0.55, hinge_x_c=0.75,
                              deflection=np.radians(3.0)),
            vt.ControlSurface(name="aileron", eta_start=0.6, eta_end=0.95, hinge_x_c=0.7,
                              deflection=0.0, symmetric=False),
        ]
    return wing


def _case(controls: bool = False, name: str = "Wing"):
    aircraft = vt.Aircraft(surfaces=[_wing(name, controls)])
    condition = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    settings = vt.SolverSettings(solver_type="vlm", n_panels=6, n_chord=2)
    return aircraft, condition, settings


def _problem(component):
    om = _om()
    prob = om.Problem()
    prob.model.add_subsystem("aero", component, promotes=["*"])
    prob.setup(force_alloc_complex=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prob.run_model()
    return prob


# Central-difference step of the free-stream speed [m/s]. The coefficients
# do not depend on V_inf in incompressible flow, so their exact partial is
# zero and the differences hold only round-off: about n ulp(C) / (2 h). With
# h = 1e-7 m/s one ulp of CL = 0.57 gives 5.6e-10, and a solve that differs
# by 3 ulp between V_inf + h and V_inf - h (measured on the Linux CI) gives
# 1.7e-9, above atol = 1e-9. With h = 1e-4 m/s (relative step 3.3e-6, near
# the one of alpha) the floor is 1000 times lower. The truncation error stays
# zero for the coefficients and for forces that are quadratic in V_inf.
V_INF_STEP = 1e-4


def _check(prob, step: float = 1e-7):
    comp = prob.model.aero
    names = {name.rsplit(".", 1)[-1] for name, _ in comp.list_inputs(out_stream=None)}
    if "V_inf" in names:
        comp.set_check_partial_options(wrt="V_inf", step=max(step, V_INF_STEP))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return prob.check_partials(out_stream=None, method="fd", form="central", step=step, compact_print=True)


def _assert_partials(data, rtol: float, atol: float, floor: float = 0.0) -> None:
    """Assert that every analytic partial agrees with the central differences.

    Each entry must satisfy ``|analytic - fd| <= atol + rtol |fd| + floor max|fd|``,
    with the maximum over the sub-Jacobian of the same output and input.
    *floor* is the round-off floor of the differences of a dimensional
    output (strip forces in N): an entry many orders smaller than the
    largest entry of its block cannot be resolved by differences to *rtol*.
    With ``floor == 0`` OpenMDAO's own ``assert_check_partials`` also runs.
    """
    assert data, "check_partials compared no partial"
    if floor == 0.0:
        from openmdao.utils.assert_utils import assert_check_partials

        assert_check_partials(data, atol=atol, rtol=rtol)
    n_compared = 0
    for comp in data.values():
        for key, entry in comp.items():
            fd = np.asarray(entry["J_fd"])
            for name in ("J_fwd", "J_rev"):
                if name not in entry:
                    continue
                an = np.asarray(entry[name])
                err = np.abs(an - fd)
                tol = atol + rtol * np.abs(fd) + floor * float(np.abs(fd).max())
                assert np.all(err <= tol), (key, name, float(err.max()))
                n_compared += 1
    assert n_compared > 0


# --- Import without the extra ------------------------------------------------------------------


def test_import_without_openmdao(monkeypatch) -> None:
    """With OpenMDAO hidden, ``import ventorum.mdo`` raises an ImportError that names the extra.

    The check of ``import ventorum`` without OpenMDAO runs in a new
    interpreter (next test): a second import of ventorum in this process
    would mix two copies of its modules.
    """
    for mod in [m for m in sys.modules if m == "openmdao" or m.startswith("openmdao.")]:
        monkeypatch.delitem(sys.modules, mod)
    monkeypatch.setitem(sys.modules, "openmdao", None)
    monkeypatch.setitem(sys.modules, "openmdao.api", None)
    monkeypatch.delitem(sys.modules, "ventorum.mdo", raising=False)
    with pytest.raises(ImportError, match=r"ventorum\[mdo\]"):
        importlib.import_module("ventorum.mdo")
    assert "ventorum.mdo" not in sys.modules


def test_import_ventorum_does_not_load_openmdao() -> None:
    """In a new interpreter with OpenMDAO hidden, ``import ventorum`` works and ``ventorum.mdo`` names the extra."""
    import subprocess

    code = (
        "import sys; sys.modules['openmdao'] = None; import ventorum\n"
        "print(ventorum.analyze is not None, any(m.startswith('openmdao.') for m in sys.modules))\n"
        "try:\n"
        "    import ventorum.mdo\n"
        "except ImportError as exc:\n"
        "    print('ventorum[mdo]' in str(exc))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, check=True)
    assert out.stdout.strip().splitlines()[-2:] == ["True False", "True"]


# --- Explicit component with analytic partials -------------------------------------------------


def test_component_values_equal_analyze_and_gradients(float64_precision) -> None:
    """The outputs equal ``vt.analyze``; the partials equal ``vt.gradients``."""
    _om()
    from ventorum.mdo import VentorumComponent

    aircraft, condition, settings = _case()
    comp = VentorumComponent(aircraft, settings, condition=condition,
                             outputs=("CL", "CD", "Cm"), inputs=("alpha", "V_inf", "nodes"))
    prob = _problem(comp)
    res = vt.analyze(aircraft, condition, settings)
    assert prob.get_val("CL")[0] == pytest.approx(res.totals.CL, rel=1e-12)
    assert prob.get_val("Cm")[0] == pytest.approx(res.moments("body")["Cm"], rel=1e-12)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        jac = prob.compute_totals(of=["CL", "CD"], wrt=["alpha", "nodes:Wing:le"])
        ref = vt.gradients(aircraft, condition, settings, outputs=("CL", "CD"), wrt=("alpha", "nodes"))
    assert jac["CL", "alpha"][0, 0] == pytest.approx(ref.jacobian["CL"]["alpha"], rel=1e-12)
    np.testing.assert_allclose(jac["CD", "nodes:Wing:le"].ravel(), ref.jacobian["CD"]["nodes"]["Wing"]["le"].ravel(),
                               rtol=1e-12, atol=1e-15)
    assert comp.gradient_result is not None
    assert comp.gradient_result.precision_gradient == "float64"


def test_component_check_partials_float64(float64_precision) -> None:
    """check_partials agrees with the analytic partials to 1e-6 relative in float64."""
    _om()
    from ventorum.mdo import VentorumComponent

    aircraft, condition, settings = _case()
    comp = VentorumComponent(aircraft, settings, condition=condition,
                             outputs=("CL", "CD", "Cm"), inputs=("alpha", "V_inf", "nodes"))
    prob = _problem(comp)
    _assert_partials(_check(prob), rtol=1e-6, atol=1e-9)


def test_component_controls_and_lateral(float64_precision) -> None:
    """Control deflections and beta as inputs, lateral outputs in stability axes."""
    _om()
    from ventorum.mdo import VentorumComponent

    aircraft, condition, settings = _case(controls=True)
    condition.beta = np.radians(2.0)
    comp = VentorumComponent(aircraft, settings, condition=condition, axes="stability",
                             outputs=("CL", "CY", "Cl", "Cn"), inputs=("beta", "controls"))
    prob = _problem(comp)
    assert prob.get_val("controls:flap", units="deg")[0] == pytest.approx(3.0)
    res = vt.analyze(aircraft, condition, settings)
    assert prob.get_val("Cl")[0] == pytest.approx(res.moments("stability")["Cl"], rel=1e-10, abs=1e-14)
    _assert_partials(_check(prob), rtol=1e-6, atol=1e-9)


def test_component_input_changes_the_analysis(float64_precision) -> None:
    """A new input value changes the case: alpha in deg and a control deflection."""
    _om()
    from ventorum.mdo import VentorumComponent

    aircraft, condition, settings = _case(controls=True)
    comp = VentorumComponent(aircraft, settings, condition=condition, inputs=("alpha", "controls"))
    prob = _problem(comp)
    prob.set_val("alpha", 6.0, units="deg")
    prob.set_val("controls:flap", 0.0)
    prob.run_model()
    ac2 = aircraft.clone()
    for c in ac2.surfaces[0].controls:
        if c.name == "flap":
            c.deflection = 0.0
    cond2 = vt.FlightCondition(V_inf=30.0, alpha=np.radians(6.0))
    res = vt.analyze(ac2, cond2, settings)
    assert prob.get_val("CL")[0] == pytest.approx(res.totals.CL, rel=1e-12)
    # The geometry of the component is not changed.
    assert aircraft.surfaces[0].controls[0].deflection == pytest.approx(np.radians(3.0))


def test_node_input_moves_the_lattice(float64_precision) -> None:
    """A node input equals the same NodeDisplacements on the surface."""
    _om()
    from ventorum.geometry import undeformed_nodes
    from ventorum.mdo import VentorumComponent

    aircraft, condition, settings = _case()
    comp = VentorumComponent(aircraft, settings, condition=condition, inputs=("nodes",))
    prob = _problem(comp)
    n_e = undeformed_nodes(aircraft, settings)["Wing"]["le"].shape[0]
    d_le = np.zeros((n_e, 3))
    d_le[:, 2] = np.linspace(0.0, 0.2, n_e)
    prob.set_val("nodes:Wing:le", d_le)
    prob.run_model()
    ac2 = aircraft.clone()
    ac2.surfaces[0].node_displacements = vt.NodeDisplacements(le=d_le, te=np.zeros((n_e, 3)))
    res = vt.analyze(ac2, condition, settings)
    assert prob.get_val("CD")[0] == pytest.approx(res.totals.CD_total, rel=1e-12)


def test_root_y_input_of_a_symmetric_surface_warns(float64_precision) -> None:
    """A nonzero root y-displacement input gives one warning and does not change CL."""
    om = _om()
    from ventorum.mdo import VentorumComponent

    aircraft, condition, settings = _case()
    comp = VentorumComponent(aircraft, settings, condition=condition, inputs=("nodes",), outputs=("CL",))
    prob = _problem(comp)
    cl0 = float(prob.get_val("CL")[0])
    le = prob.get_val("nodes:Wing:le").copy()
    le[0, 1] = 0.01
    prob.set_val("nodes:Wing:le", le)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        prob.run_model()
    root = [w for w in caught if issubclass(w.category, om.OpenMDAOWarning) and "root y-displacement" in str(w.message)]
    assert len(root) == 1
    assert float(prob.get_val("CL")[0]) == cl0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        jac = prob.compute_totals(of=["CL"], wrt=["nodes:Wing:le"])
    assert jac["CL", "nodes:Wing:le"].reshape(-1, 3)[0, 1] == 0.0


def test_root_y_displacement_of_the_geometry_is_refused() -> None:
    """A symmetric geometry with a nonzero root y-displacement raises ValueError at setup."""
    om = _om()
    from ventorum.geometry import undeformed_nodes
    from ventorum.mdo import VentorumComponent

    aircraft, condition, settings = _case()
    n_e = undeformed_nodes(aircraft, settings)["Wing"]["le"].shape[0]
    le = np.zeros((n_e, 3))
    le[0, 1] = 0.01
    aircraft.surfaces[0].node_displacements = vt.NodeDisplacements(le=le, te=np.zeros((n_e, 3)))
    prob = om.Problem(reports=False)
    prob.model.add_subsystem("aero", VentorumComponent(aircraft, settings, condition=condition, inputs=("nodes",)))
    with pytest.raises(ValueError, match="root"):
        prob.setup()


def test_surface_name_with_a_dot_is_changed(float64_precision) -> None:
    """OpenMDAO forbids a dot in a variable name: the dot becomes an underscore."""
    _om()
    from ventorum.mdo import VentorumComponent

    aircraft, condition, settings = _case(name="wing.main")
    comp = VentorumComponent(aircraft, settings, condition=condition, inputs=("nodes",), outputs=("CL",))
    prob = _problem(comp)
    assert prob.get_val("nodes:wing_main:te").shape[1] == 3
    assert comp.input_names["nodes:wing_main:te"] == ("nodes", "wing.main", "te")


def test_device_and_precision_are_not_changed(float64_precision) -> None:
    """The component uses the global device and precision settings and does not change them."""
    _om()
    from ventorum.mdo import VentorumComponent

    dev, prec = gpu.get_device(), gpu._precision
    aircraft, condition, settings = _case()
    prob = _problem(VentorumComponent(aircraft, settings, condition=condition, inputs=("alpha",), outputs=("CL",)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prob.compute_totals(of=["CL"], wrt=["alpha"])
    assert (gpu.get_device(), gpu._precision) == (dev, prec)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"inputs": ("planform",)}, "planform"),
        ({"inputs": ("sections",)}, "sections"),
        ({"inputs": ("reference",)}, "reference"),
        ({"inputs": ("h",)}, "condition.h"),
        ({"inputs": ("controls",)}, "control"),
        ({"inputs": ("speed",)}, "Unknown input"),
        ({"outputs": ("lift",)}, "Unknown output"),
        ({"outputs": ("CL", "section_cl")}, "VentorumMatrixFreeComponent"),
        ({"axes": "earth"}, "axes"),
    ],
)
def test_component_refusals(kwargs, match) -> None:
    """Refused names and configurations raise ValueError with the reason."""
    _om()
    from ventorum.mdo import VentorumComponent

    aircraft, condition, settings = _case()
    om = _om()
    prob = om.Problem()
    prob.model.add_subsystem("aero", VentorumComponent(aircraft, settings, condition=condition, **kwargs))
    with pytest.raises(ValueError, match=match):
        prob.setup()


def test_check_partials_ground_effect_rates_and_density(float64_precision) -> None:
    """check_partials with the height, the body rates, the bank angle and the density (VLM, ground effect)."""
    _om()
    from ventorum.mdo import VentorumComponent

    aircraft, _, settings = _case()
    condition = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), h=2.0, phi=np.radians(2.0),
                                   p=0.05, q=0.03, r=-0.02, rho=1.1)
    comp = VentorumComponent(aircraft, settings, condition=condition,
                             outputs=("CL", "CD", "Cl", "Cm", "Cn"), inputs=("h", "p", "q", "r", "phi", "rho"))
    prob = _problem(comp)
    res = vt.analyze(aircraft, condition, settings)
    assert prob.get_val("CL")[0] == pytest.approx(res.totals.CL, rel=1e-12)
    _assert_partials(_check(prob), rtol=1e-6, atol=1e-9)


def test_check_partials_linear_lifting_line(float64_precision) -> None:
    """check_partials with the linear lifting line (flow inputs and a control)."""
    _om()
    from ventorum.mdo import VentorumComponent

    aircraft, condition, _ = _case(controls=True)
    settings = vt.SolverSettings(solver_type="linear", n_panels=8)
    comp = VentorumComponent(aircraft, settings, condition=condition,
                             outputs=("CL", "CD", "Cm"), inputs=("alpha", "V_inf", "controls"))
    prob = _problem(comp)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # The sweep warning of the lifting line (tapered wing).
        res = vt.analyze(aircraft, condition, settings)
    assert prob.get_val("CL")[0] == pytest.approx(res.totals.CL, rel=1e-12)
    _assert_partials(_check(prob), rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("solver_type", ["linear", "vlm"])
def test_check_partials_v_inf_round_off_floor(float64_precision, solver_type: str) -> None:
    """The differences of the zero V_inf partials stay 10 times below atol = 1e-9.

    Regression test: with a step of 1e-7 m/s on V_inf the round-off of the
    central differences of CL was 1.7e-9 on the Linux CI, so a correct
    (zero) analytic partial failed check_partials.
    """
    _om()
    from ventorum.mdo import VentorumComponent

    aircraft, condition, _ = _case(controls=True)
    extra = {"n_panels": 8} if solver_type == "linear" else {"n_panels": 6, "n_chord": 2}
    settings = vt.SolverSettings(solver_type=solver_type, **extra)
    comp = VentorumComponent(aircraft, settings, condition=condition,
                             outputs=("CL", "CD", "Cm"), inputs=("alpha", "V_inf", "controls"))
    prob = _problem(comp)
    data = _check(prob)["aero"]
    for out in ("CL", "CD", "Cm"):
        entry = data[(out, "V_inf")]
        assert np.max(np.abs(entry["J_fwd"])) <= 1e-12, out
        assert np.max(np.abs(entry["J_fd"])) <= 1e-10, out


def test_control_name_on_two_surfaces(float64_precision) -> None:
    """One control input moves the controls with that name on two surfaces together."""
    _om()
    from ventorum.mdo import VentorumComponent

    wing = _wing()
    tail = vt.LiftingSurface(
        name="Tail", semi_span=2.0, position=np.array([5.0, 0.0, 0.3]),
        sections=[vt.WingSection(y_frac=0.0, chord=0.9), vt.WingSection(y_frac=1.0, chord=0.6)])
    for surf in (wing, tail):
        surf.controls = [vt.ControlSurface(name="flap", eta_start=0.2, eta_end=0.8, hinge_x_c=0.7,
                                           deflection=np.radians(2.0))]
    aircraft = vt.Aircraft(surfaces=[wing, tail])
    condition = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    settings = vt.SolverSettings(solver_type="vlm", n_panels=6, n_chord=2)
    comp = VentorumComponent(aircraft, settings, condition=condition, outputs=("CL", "Cm"), inputs=("controls",))
    prob = _problem(comp)
    assert list(comp.input_names) == ["controls:flap"]
    _assert_partials(_check(prob), rtol=1e-6, atol=1e-9)
    prob.set_val("controls:flap", 5.0, units="deg")
    prob.run_model()
    ac2 = aircraft.clone()
    for surf in ac2.surfaces:
        surf.controls[0].deflection = np.radians(5.0)
    res = vt.analyze(ac2, condition, settings)
    assert prob.get_val("Cm")[0] == pytest.approx(res.moments("body")["Cm"], rel=1e-12)


# --- Matrix-free component (vector outputs) ----------------------------------------------------


@pytest.mark.slow
def test_matrix_free_check_partials_float64(float64_precision) -> None:
    """The matrix-free component agrees with central differences in both modes."""
    _om()
    from ventorum.mdo import VentorumMatrixFreeComponent

    aircraft, condition, settings = _case()
    settings = vt.SolverSettings(solver_type="vlm", n_panels=4, n_chord=1)
    comp = VentorumMatrixFreeComponent(aircraft, settings, condition=condition,
                                       outputs=("CL", "section_cl", "strip_force"), inputs=("alpha", "nodes"))
    prob = _problem(comp)
    _assert_partials(_check(prob, step=1e-6), rtol=1e-6, atol=1e-9, floor=1e-8)


def test_matrix_free_reverse_product_equals_gradient_vjp(float64_precision) -> None:
    """The reverse product equals ``vt.gradient_vjp`` with the same seeds; strip_force is in N."""
    _om()
    from ventorum.mdo import VentorumMatrixFreeComponent

    aircraft, condition, settings = _case()
    comp = VentorumMatrixFreeComponent(aircraft, settings, condition=condition,
                                       outputs=("CL", "strip_force"), inputs=("alpha",))
    prob = _problem(comp)
    force = prob.get_val("strip_force")
    res = vt.analyze(aircraft, condition, settings)
    q_s = 0.5 * condition.rho * condition.V_inf**2 * aircraft.S_ref
    np.testing.assert_allclose(force, res.details["loads"].strip_force, rtol=1e-12)
    lift_dir = np.array([-np.sin(condition.alpha), 0.0, np.cos(condition.alpha)])
    assert float(np.sum(force @ lift_dir)) / q_s == pytest.approx(res.totals.CL, rel=1e-6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        jac = prob.compute_totals(of=["strip_force"], wrt=["alpha"])
        seed = np.zeros_like(force)
        seed[1, 2] = 1.0
        ref = vt.gradient_vjp(aircraft, condition, settings, seeds={"strip_force": seed}, wrt=("alpha",))
    assert jac["strip_force", "alpha"].reshape(force.shape)[1, 2] == pytest.approx(ref["alpha"], rel=1e-12)


def test_forward_cache_follows_the_precision() -> None:
    """After a change of the precision, the forward mode gives a fresh Jacobian (regression)."""
    _om()
    from ventorum.mdo import VentorumMatrixFreeComponent

    aircraft, condition, _ = _case()
    settings = vt.SolverSettings(solver_type="vlm", n_panels=4, n_chord=1)
    with _gpu_settings(precision="float32"):
        comp = VentorumMatrixFreeComponent(aircraft, settings, condition=condition,
                                           outputs=("CL", "strip_force"), inputs=("alpha",))
        om = _om()
        prob = om.Problem(reports=False)
        prob.model.add_subsystem("aero", comp, promotes=["*"])
        prob.setup(mode="fwd")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prob.run_model()
            prob.compute_totals(of=["CL"], wrt=["alpha"])
        assert comp.gradient_result.precision_gradient == "float32"
        gpu.set_precision("float64")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prob.run_model()
            jac = prob.compute_totals(of=["CL"], wrt=["alpha"])
            ref = vt.gradients(aircraft, condition, settings, outputs=("CL",), wrt=("alpha",))
        assert ref.precision_gradient == "float64"
        assert jac["CL", "alpha"][0, 0] == pytest.approx(ref.jacobian["CL"]["alpha"], rel=1e-12)
        # A change of the precision without a new compute also drops the kept Jacobian.
        gpu.set_precision("float32")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prob.compute_totals(of=["CL"], wrt=["alpha"])
        assert comp.gradient_result.precision_gradient == "float32"


def test_forward_cache_follows_the_inputs(float64_precision) -> None:
    """After a change of the inputs, the forward mode gives the Jacobian at the new inputs."""
    om = _om()
    from ventorum.mdo import VentorumMatrixFreeComponent

    aircraft, condition, _ = _case()
    settings = vt.SolverSettings(solver_type="vlm", n_panels=4, n_chord=1)
    prob = om.Problem(reports=False)
    prob.model.add_subsystem("aero", VentorumMatrixFreeComponent(
        aircraft, settings, condition=condition, outputs=("CD", "section_cl"), inputs=("alpha",)), promotes=["*"])
    prob.setup(mode="fwd")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prob.run_model()
        j1 = prob.compute_totals(of=["CD"], wrt=["alpha"])["CD", "alpha"][0, 0]
        prob.set_val("alpha", 8.0, units="deg")
        prob.run_model()
        j2 = prob.compute_totals(of=["CD"], wrt=["alpha"])["CD", "alpha"][0, 0]
        cond2 = vt.FlightCondition(V_inf=30.0, alpha=np.radians(8.0))
        ref = vt.gradients(aircraft, cond2, settings, outputs=("CD",), wrt=("alpha",)).jacobian["CD"]["alpha"]
    assert j2 == pytest.approx(ref, rel=1e-12)
    assert abs(j2 - j1) > 1e-3 * abs(ref)


# --- GPU in float32 ----------------------------------------------------------------------------


@pytest.mark.gpu
def test_component_gpu_float32(gpu_float32) -> None:
    """On the GPU in float32 the partials agree with differences to about 1e-3."""
    _om()
    from ventorum.mdo import VentorumComponent

    aircraft, condition, settings = _case()
    comp = VentorumComponent(aircraft, settings, condition=condition,
                             outputs=("CL", "CD"), inputs=("alpha",))
    prob = _problem(comp)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        jac = prob.compute_totals(of=["CL"], wrt=["alpha"])
    assert comp.gradient_result.precision_gradient == "float32"
    _assert_partials(_check(prob, step=1e-3), rtol=1e-3, atol=1e-6)
    assert np.isfinite(jac["CL", "alpha"]).all()


# --- Example of the user manual ----------------------------------------------------------------


@pytest.mark.slow
def test_twist_example_converges(float64_precision) -> None:
    """The twist optimisation example converges to the target CL with a lower CD."""
    _om()
    namespace = runpy.run_path(str(ROOT / "examples" / "12_openmdao_twist.py"), run_name="ventorum_example")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = namespace["main"]()
    assert out["success"]
    assert out["CL"] == pytest.approx(out["CL_target"], abs=1e-6)
    assert out["CD"] < out["CD_initial"]


def test_user_manual_first_problem_runs(float64_precision) -> None:
    """The first example of the OpenMDAO how-to runs as written; the included script exists."""
    import re

    _om()
    text = (ROOT / "docs" / "user" / "how_to_openmdao.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.S)
    assert blocks, "the guide must have a Python example"
    namespace: dict = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec(compile(blocks[0], "docs/user/how_to_openmdao.md", "exec"), namespace)  # noqa: S102
    prob = namespace["prob"]
    assert 0.0 < prob.get_val("CL")[0] < 1.0
    for path in re.findall(r"```\{literalinclude\} (\S+)", text):
        assert (ROOT / "docs" / "user" / path).resolve().is_file(), path
