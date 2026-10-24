"""Checks of the vortex lattice on the section mean surface."""

from __future__ import annotations

import copy
import warnings

import numpy as np
import pytest

from ventorum import gpu
from ventorum.aero import vortex
from ventorum.aero.system import GroundPlane, build_sources, make_unknown_map
from ventorum.airfoil import airfoil
from ventorum.core.datatypes import Aircraft, FlightCondition, LiftingSurface, LinearAirfoil, SolverSettings, WingSection
from ventorum.geometry.lattice import build_lattice
from ventorum.geometry.transform import rotation_matrix, transform_lattice
from ventorum.solvers.core import assemble_vlm
from ventorum.solvers.horseshoe import VortexLatticeSolver

_ALPHA = np.radians(5.0)

def _aircraft(shape="4412", *, incidence=0.0, polar=None, semi_span=4.0):
    af = polar if polar is not None else LinearAirfoil()
    return Aircraft(surfaces=[LiftingSurface(
        name="wing", semi_span=semi_span, incidence=incidence,
        sections=[WingSection(y_frac=e, chord=1.0, shape=shape, airfoil=af) for e in (0.0, 1.0)],
    )], ref_point=np.array([0.25, 0.0, 0.0]))


def _solve(ac, *, h=None, alpha=_ALPHA, n_chord=8, n_span=12):
    return VortexLatticeSolver().solve(
        ac, FlightCondition(V_inf=30.0, alpha=alpha, h=h),
        SolverSettings(n_panels=n_span, n_chord=n_chord),
    )


@pytest.fixture(autouse=True)
def _restore_backends():
    old_backend = vortex._kernel_backend
    old_device, old_precision = gpu.get_device(), gpu._precision
    gpu.set_device("cpu")
    yield
    vortex.set_kernel_backend(old_backend)
    gpu.set_device(old_device)
    gpu.set_precision(old_precision)


@pytest.mark.parametrize("h", [None, 0.2])
@pytest.mark.parametrize("slope_factor", [1.0, 1.8])
def test_uncambered_bit_identical(h, slope_factor):
    plain = _aircraft(None, polar=LinearAirfoil(a0=slope_factor * 2 * np.pi))
    shaped = _aircraft("0012", polar=LinearAirfoil(a0=slope_factor * 2 * np.pi))
    a, b = _solve(plain, h=h), _solve(shaped, h=h)
    for name in ("a", "b", "a_te", "b_te", "cp", "normal_bc", "normal_bc_strip", "dl", "qc_mid"):
        assert np.array_equal(getattr(a.details["lattice"], name), getattr(b.details["lattice"], name)), name
    assert np.array_equal(a.details["gamma"], b.details["gamma"])
    assert a.details["h_min"] == b.details["h_min"]
    for name in ("CL", "CDi", "Cm", "CY", "Cl", "Cn"):
        assert getattr(a.totals, name) == getattr(b.totals, name)


def test_points_on_mean_surface():
    lat = build_lattice(_aircraft(), SolverSettings(n_panels=4), n_chord=7)
    shape = airfoil("4412").normalized()
    for name in ("a", "b", "cp", "a_te", "b_te"):
        p = getattr(lat, name)
        np.testing.assert_allclose(p[:, 2], shape.camber(p[:, 0]), rtol=0, atol=2e-15)
    assert np.max(lat.a[:, 2]) > 0.03


def test_normals_of_mean_surface():
    lat = build_lattice(_aircraft(), SolverSettings(n_panels=4), n_chord=7)
    shape = airfoil("4412").normalized()
    x = lat.cp[:, 0]
    slope = (shape.camber(x + 1e-7) - shape.camber(x - 1e-7)) / 2e-7
    tangent = np.column_stack([np.ones_like(x), np.zeros_like(x), slope])
    np.testing.assert_allclose(np.sum(lat.normal_bc * tangent, axis=1), 0, atol=1e-9)
    np.testing.assert_allclose(lat.normal_bc[:, 1], 0, atol=1e-15)


@pytest.mark.parametrize("h", [None, 0.2])
def test_rigid_rotation_invariance(h):
    i, alpha = np.radians([3.0, 2.0])
    inclined = _solve(_aircraft(incidence=i), h=h, alpha=alpha)
    level = _solve(_aircraft(), h=h, alpha=alpha + i)
    np.testing.assert_allclose(inclined.details["gamma"], level.details["gamma"], rtol=1e-12, atol=1e-14)
    for name in ("CL", "CDi", "Cm"):
        np.testing.assert_allclose(getattr(inclined.totals, name), getattr(level.totals, name), rtol=1e-12, atol=1e-14)


def test_ground_mirror_explicit():
    lat = build_lattice(_aircraft(), SolverSettings(n_panels=4), n_chord=5)
    cond = FlightCondition(V_inf=30.0, alpha=np.radians(5.0))
    k = np.array([-np.sin(cond.alpha), 0.0, np.cos(cond.alpha)])
    ground = GroundPlane(k, float(np.array([0.25, 0.0, 0.0]) @ k - 0.2))
    real = build_sources(lat, np.array([np.cos(cond.alpha), 0.0, np.sin(cond.alpha)]), make_unknown_map(lat, False))
    image = copy.deepcopy(real)
    for name in ("a", "b", "a_te", "b_te"):
        setattr(image, name, ground.reflect_points(getattr(real, name)))
    image.wake_dir = ground.reflect_directions(real.wake_dir)
    v_real = vortex.horseshoe_velocity_tensor(lat.cp, real)
    v_image = vortex.horseshoe_velocity_tensor(lat.cp, image)
    sources = build_sources(lat, np.array([np.cos(cond.alpha), 0.0, np.sin(cond.alpha)]), make_unknown_map(lat, False), ground)
    combined = vortex.horseshoe_velocity_tensor(lat.cp, sources)
    np.testing.assert_allclose(combined[:, :lat.n_panels] + combined[:, lat.n_panels:], v_real - v_image, rtol=1e-12, atol=1e-14)


def test_reference_cambered_plate_2d():
    from ventorum.reference.cambered_plate import cambered_plate

    # A parabolic arc has alpha_L0 = -2 m and Cm_c/4 = -pi m in thin-airfoil theory.
    mean = lambda x: 0.04 * x * (1.0 - x)  # noqa: E731
    slope = lambda x: 0.04 * (1.0 - 2.0 * x)  # noqa: E731
    errors = []
    for n in (32, 64, 128):
        r = cambered_plate(mean, slope, alpha=-0.02, n_panels=n)
        errors.append(abs(r["Cm"] + np.pi * 0.01))
    assert errors[-1] < errors[0]
    assert abs(r["CL"]) < 0.002
    assert errors[-1] < 0.002
    a = cambered_plate(mean, slope, alpha=0.05, h=0.2, n_panels=64)
    b = cambered_plate(mean, slope, alpha=0.05, h=0.2, n_panels=64, explicit_mirror=True)
    np.testing.assert_allclose(a["gamma"], b["gamma"], rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("h", [None, 0.2])
def test_legs_effect(h):
    coarse = _solve(_aircraft(), h=h, n_chord=16, n_span=16).totals
    fine = _solve(_aircraft(), h=h, n_chord=48, n_span=16).totals
    print(f"legs h={h}: CL {coarse.CL:.16g} -> {fine.CL:.16g}; Cm {coarse.Cm:.16g} -> {fine.Cm:.16g}")
    assert abs(coarse.CL - fine.CL) <= 0.01 * abs(fine.CL)
    assert abs(coarse.Cm - fine.Cm) <= max(0.02 * abs(fine.Cm), 0.002)


@pytest.mark.parametrize("h", [None, 0.2])
def test_backend_parity(h):
    vortex.set_kernel_backend("numpy")
    a = _solve(_aircraft(polar=LinearAirfoil(Cm0=-0.1)), h=h, n_chord=5, n_span=5)
    vortex.set_kernel_backend("numba")
    b = _solve(_aircraft(polar=LinearAirfoil(Cm0=-0.1)), h=h, n_chord=5, n_span=5)
    np.testing.assert_allclose(a.details["gamma"], b.details["gamma"], rtol=1e-12, atol=1e-14)
    for name in ("CL", "CDi", "Cm", "CY", "Cl", "Cn"):
        np.testing.assert_allclose(getattr(a.totals, name), getattr(b.totals, name), rtol=1e-12, atol=1e-14)


@pytest.mark.gpu
@pytest.mark.parametrize("precision,tol", [("float64", 1e-12), ("float32", 1e-5)])
def test_backend_parity_gpu(precision, tol):
    if not gpu.available():
        pytest.skip(gpu.unavailable_reason())
    ac = _aircraft(polar=LinearAirfoil(Cm0=-0.1))
    cpu = _solve(ac, h=0.2, n_chord=5, n_span=5)
    gpu.set_device("gpu")
    gpu.set_precision(precision)
    result = _solve(ac, h=0.2, n_chord=5, n_span=5)
    assert result.details["device"] == "gpu"
    for name in ("CL", "CDi", "Cm", "CY", "Cl", "Cn"):
        a, b = getattr(cpu.totals, name), getattr(result.totals, name)
        assert abs(a - b) <= max(tol * abs(a), 1e-14)


@pytest.mark.slow
@pytest.mark.parametrize("h", [None, 0.2])
def test_thin_cambered_wing_against_panel3d(h):
    from ventorum.panel3d.mesh import mesh_lifting_surface
    from ventorum.panel3d.solver import PanelSolver

    ac = _aircraft("2402")
    mesh = mesh_lifting_surface(ac.surfaces[0], airfoil("2402").coords, n_chord=32, n_span=24, half=True)
    panel = PanelSolver(mesh, S_ref=8.0, b_ref=8.0, c_ref=1.0, ref_point=ac.ref_point,
                        backend="numba", far_ratio=np.inf)
    p = panel.solve(FlightCondition(V_inf=30.0, alpha=np.radians(5.0), h=h))
    v = _solve(ac, h=h, n_chord=32, n_span=24).totals
    print(f"panel h={h}: VLM CL={v.CL:.16g} Cm={v.Cm:.16g}; 3DP CL={p.CL:.16g} Cm={p.Cm:.16g}")
    assert abs(v.CL - p.CL) <= 0.03 * abs(p.CL)
    # The 3DP moment of this thin cambered wing converges slowly with the
    # chordwise panels, and in ground effect it differs from both 2D
    # references, which the VLM moment matches (see the test below). It is
    # reported here and checked against the 2D references instead.
    print(f"panel h={h}: 3DP moment reported only, VLM/3DP Cm ratio {v.Cm / p.Cm:.4f}")


@pytest.mark.parametrize("h", [None, 0.2])
def test_high_aspect_ratio_moment_against_2d(h):
    """At aspect ratio 40 the wing moment approaches the 2D section moment.

    In thin-airfoil theory the quarter-chord moment of a cambered section
    does not depend on the angle of attack to first order, so the induced
    angle changes it little; the tip effects are of order 1/AR (2.5 %).
    Limits: 3 % against the 2D lumped-vortex plate on the same mean line,
    5 % against Ventorum-2D on the real 2 % thick section (thickness alone
    changes the 2D moment by about 3 % in free air).
    """
    from ventorum.panel2d.solver import solve_inviscid
    from ventorum.reference.cambered_plate import cambered_plate

    alpha = np.radians(5.0)
    shape = airfoil("2402").normalized()
    slope = lambda x: (shape.camber(x + 1e-7) - shape.camber(x - 1e-7)) / 2e-7  # noqa: E731
    plate = cambered_plate(shape.camber, slope, alpha=alpha, h=h, n_panels=256)
    panel2d = solve_inviscid(airfoil("2402").coords, 5.0, height=h, backend="numpy")
    v = _solve(_aircraft("2402", semi_span=20.0), h=h, alpha=alpha, n_chord=16, n_span=60).totals
    print(f"AR 40 h={h}: VLM Cm={v.Cm:.6f}; 2D plate Cm={plate['Cm']:.6f}; Ventorum-2D Cm={float(panel2d.Cm):.6f}")
    assert abs(v.Cm - plate["Cm"]) <= 0.03 * abs(plate["Cm"])
    assert abs(v.Cm - float(panel2d.Cm)) <= 0.05 * abs(float(panel2d.Cm))


def test_symmetric_shape_keeps_gradients():
    """A symmetric shape with a zero-camber polar keeps the adjoint gradients."""
    from ventorum.sensitivity import check_gradients

    ac = _aircraft("0012", polar=LinearAirfoil())
    report = check_gradients(ac, FlightCondition(V_inf=30.0, alpha=0.05),
                             SolverSettings(n_panels=4, n_chord=4), wrt=("alpha", "nodes"), outputs=("CL",))
    assert report["pass"], report


def test_fallback_camber_trust_note():
    result = _solve(_aircraft(None, polar=LinearAirfoil(alpha_L0=-0.03)), h=0.2)
    text = "camber modelled by a turned normal (no mean line given); near the ground the chordwise load and the camber height are approximate"
    assert text in result.details["notes"]


@pytest.mark.parametrize("h", [None, 0.2])
def test_uncambered_sections_without_shape_have_no_camber_note(h):
    text = "camber modelled by a turned normal (no mean line given)"
    result = _solve(_aircraft(None, polar=LinearAirfoil()), h=h)
    assert not any(text in n for n in result.details["notes"])


def test_shape_polar_warning_once_per_section():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_lattice(_aircraft(), SolverSettings(n_panels=8), n_chord=4)
    mismatch = [w for w in caught if "alpha_L0" in str(w.message) and "mean line" in str(w.message)]
    assert len(mismatch) == 2


def test_cm0_not_counted_twice():
    a = _solve(_aircraft(polar=LinearAirfoil(Cm0=0.0)), alpha=-0.07)
    b = _solve(_aircraft(polar=LinearAirfoil(Cm0=-0.2)), alpha=-0.07)
    assert a.totals.Cm == b.totals.Cm
    from ventorum.reference.cambered_plate import cambered_plate

    shape = airfoil("4412").normalized()
    slope = lambda x: (shape.camber(x + 1e-7) - shape.camber(x - 1e-7)) / 2e-7  # noqa: E731
    ref = cambered_plate(shape.camber, slope, alpha=-0.07, n_panels=128)
    assert abs(a.totals.Cm - ref["Cm"]) < 0.01


def test_transform_carries_camber_offset():
    lat = build_lattice(_aircraft(), SolverSettings(n_panels=4), n_chord=5)
    R = rotation_matrix(roll=0.1, pitch=0.2, yaw=0.05)
    transformed = transform_lattice(lat, rotation=R, translation=np.array([1.0, 2.0, 3.0]))
    for name in ("a", "b", "cp", "a_te", "b_te"):
        np.testing.assert_allclose(getattr(transformed, name), getattr(lat, name) @ R.T + [1.0, 2.0, 3.0], atol=1e-14)
    np.testing.assert_allclose(transformed.normal_bc, lat.normal_bc @ R.T, atol=1e-14)


def test_alpha_l0_gradient_refused_for_geometric_camber():
    from ventorum.sensitivity import GradientScopeError, gradients

    with pytest.raises(GradientScopeError, match="alpha_L0.*mean line|mean line.*alpha_L0"):
        gradients(_aircraft(), FlightCondition(V_inf=30.0, alpha=0.05),
                  SolverSettings(n_panels=4, n_chord=4), outputs=("CL",), wrt=("sections",))


def test_llt_keeps_turned_normal():
    shaped = _aircraft(polar=LinearAirfoil(alpha_L0=-0.03))
    plain = copy.deepcopy(shaped)
    for sec in plain.surfaces[0].sections:
        sec.shape = None
    settings = SolverSettings(n_panels=4)
    a = build_lattice(shaped, settings, collocation="llt")
    b = build_lattice(plain, settings, collocation="llt")
    for name in ("a", "b", "cp", "normal_bc"):
        assert np.array_equal(getattr(a, name), getattr(b, name))


def test_camber_changes_influence_geometry():
    settings = SolverSettings(n_panels=4)
    shaped = build_lattice(_aircraft(), settings, n_chord=5)
    plain = build_lattice(_aircraft(None), settings, n_chord=5)
    condition = FlightCondition(V_inf=30.0, alpha=0.05)
    a, rhs_a, _ = assemble_vlm(shaped, condition)
    b, rhs_b, _ = assemble_vlm(plain, condition)
    assert not np.array_equal(a, b)
    assert not np.array_equal(rhs_a, rhs_b)


@pytest.mark.parametrize("h", [None, 0.2])
def test_mixed_sections_keep_surface_fallback(h):
    mixed = _aircraft(polar=LinearAirfoil(alpha_L0=-0.03, Cm0=-0.04))
    mixed.surfaces[0].sections[1].shape = None
    plain = copy.deepcopy(mixed)
    plain.surfaces[0].sections[0].shape = None
    a, b = _solve(mixed, h=h, n_span=4), _solve(plain, h=h, n_span=4)
    for name in ("a", "b", "a_te", "b_te", "cp", "normal_bc"):
        assert np.array_equal(getattr(a.details["lattice"], name), getattr(b.details["lattice"], name))
    assert np.array_equal(a.details["gamma"], b.details["gamma"])
    for name in ("CL", "CDi", "Cm"):
        assert getattr(a.totals, name) == getattr(b.totals, name)
    note = "camber of surface wing modelled by a turned normal: not every section gives a mean line"
    assert note in a.details["notes"]
    assert note in a.totals.trust.warnings


@pytest.mark.parametrize("h", [None, 0.2])
@pytest.mark.parametrize("with_control", [False, True])
def test_single_chord_panel_keeps_base_bits(h, with_control):
    from ventorum.core.datatypes import ControlSurface

    shaped = _aircraft(polar=LinearAirfoil(alpha_L0=-0.03, Cm0=-0.04))
    if with_control:
        shaped.surfaces[0].controls = [ControlSurface(
            name="flap", eta_start=0.0, eta_end=1.0, hinge_x_c=0.75, deflection=0.1,
        )]
    # The base builder does not read shape. Removing it gives the base inputs
    # for the unchanged one-panel and section-control paths.
    base = copy.deepcopy(shaped)
    for sec in base.surfaces[0].sections:
        sec.shape = None
    a, b = _solve(shaped, h=h, n_chord=1, n_span=4), _solve(base, h=h, n_chord=1, n_span=4)
    for name in ("a", "b", "a_te", "b_te", "cp", "normal_bc", "normal_bc_strip", "Cm0", "alpha_L0"):
        assert np.array_equal(getattr(a.details["lattice"], name), getattr(b.details["lattice"], name)), name
    assert np.array_equal(a.details["gamma"], b.details["gamma"])
    assert not np.any(a.details["lattice"].geometric_camber)
    for name in ("CL", "CDi", "Cm", "CY", "Cl", "Cn"):
        assert getattr(a.totals, name) == getattr(b.totals, name)


def test_span_varying_mean_lines():
    ac = _aircraft("2412")
    ac.surfaces[0].sections[1].shape = "4412"
    ac.surfaces[0].sections[1].chord = 0.7
    lat = build_lattice(ac, SolverSettings(n_panels=5), n_chord=5)
    lo, hi = airfoil("2412").normalized(), airfoil("4412").normalized()
    for name in ("a", "b", "cp", "a_te", "b_te"):
        p = getattr(lat, name)
        eta = np.abs(p[:, 1]) / ac.surfaces[0].semi_span
        chord = 1.0 - 0.3 * eta
        xi = p[:, 0] / chord
        expected = chord * ((1.0 - eta) * lo.camber(xi) + eta * hi.camber(xi))
        np.testing.assert_allclose(p[:, 2], expected, rtol=0, atol=2e-15)
    p = lat.cp
    eta = np.abs(p[:, 1]) / ac.surfaces[0].semi_span
    xi = p[:, 0] / (1.0 - 0.3 * eta)
    slope = ((1 - eta) * (lo.camber(xi + 1e-7) - lo.camber(xi - 1e-7))
             + eta * (hi.camber(xi + 1e-7) - hi.camber(xi - 1e-7))) / 2e-7
    tangent = np.column_stack([np.ones(len(p)), np.zeros(len(p)), slope])
    np.testing.assert_allclose(np.sum(lat.normal_bc * tangent, axis=1), 0.0, atol=1e-9)


def test_displaced_cambered_nodes_carry_offset():
    from ventorum.geometry.deformation import displacements_from_section_motion, undeformed_nodes

    ac = _aircraft()
    st = SolverSettings(n_panels=4, spacing="uniform")
    nodes = undeformed_nodes(ac, st)["wing"]
    angle, heave = 0.11, 0.06
    ac.surfaces[0].node_displacements = displacements_from_section_motion(
        nodes, heave=np.full(len(nodes["eta"]), heave), twist=np.full(len(nodes["eta"]), angle),
    )
    actual = build_lattice(ac, st, n_chord=5)
    base = build_lattice(_aircraft(), st, n_chord=5)
    expected = transform_lattice(base, rotation=rotation_matrix(pitch=angle), about=ac.ref_point,
                                 translation=np.array([0.0, 0.0, heave]))
    for name in ("a", "b", "cp", "a_te", "b_te", "normal_bc"):
        np.testing.assert_allclose(getattr(actual, name), getattr(expected, name), rtol=1e-12, atol=1e-14)


def test_displaced_chord_scales_camber_height():
    from ventorum.core.datatypes import NodeDisplacements
    from ventorum.geometry.deformation import undeformed_nodes

    ac = _aircraft()
    st = SolverSettings(n_panels=5)
    nodes = undeformed_nodes(ac, st)["wing"]
    heave = 0.08 * nodes["eta"] ** 2
    dle = np.column_stack([np.zeros_like(heave), np.zeros_like(heave), heave])
    dte = dle.copy()
    dte[:, 0] += 0.3 * nodes["eta"]
    ac.surfaces[0].node_displacements = NodeDisplacements(le=dle, te=dte, eta=nodes["eta"])
    base = copy.deepcopy(ac)
    for sec in base.surfaces[0].sections:
        sec.shape = None
    shaped = build_lattice(ac, st, n_chord=5)
    flat = build_lattice(base, st, n_chord=5)
    s = flat.panel_strip
    f = flat.cp_frac[s, None]
    chord_vec = (1 - f) * (flat.te_left[s] - flat.le_left[s]) + f * (flat.te_right[s] - flat.le_right[s])
    chord = np.linalg.norm(chord_vec, axis=1)
    xi = (np.tile(np.arange(5), flat.n_strips) + 0.75) / 5.0
    z = airfoil("4412").normalized().camber(xi)
    expected = chord[:, None] * z[:, None] * flat.normal[s]
    np.testing.assert_allclose(shaped.cp - flat.cp, expected, rtol=1e-12, atol=1e-14)


def test_mean_line_moment_at_exact_zero_lift():
    from scipy.optimize import brentq

    from ventorum.reference.cambered_plate import cambered_plate

    ac = _aircraft(polar=LinearAirfoil(Cm0=-0.2))
    alpha = brentq(lambda a: _solve(ac, alpha=a).totals.CL, -0.15, 0.0, xtol=1e-14)
    wing = _solve(ac, alpha=alpha).totals
    assert abs(wing.CL) < 1e-12
    shape = airfoil("4412").normalized()
    slope = lambda x: (shape.camber(x + 1e-7) - shape.camber(x - 1e-7)) / 2e-7  # noqa: E731
    alpha2d = brentq(lambda a: cambered_plate(shape.camber, slope, alpha=a, n_panels=128)["CL"], -0.15, 0.0)
    section = cambered_plate(shape.camber, slope, alpha=alpha2d, n_panels=128)
    assert abs(section["CL"]) < 1e-10
    assert abs(wing.Cm - section["Cm"]) < 0.01


def _joined_cambered_surfaces():
    surfaces = []
    for name, y, shape in (("inner", 0.0, "2412"), ("outer", 2.0, "4412")):
        surfaces.append(LiftingSurface(
            name=name, semi_span=2.0, is_symmetric=False, spacing="uniform", position=np.array([0.0, y, 0.0]),
            sections=[WingSection(y_frac=e, chord=1.0, shape=shape) for e in (0.0, 1.0)],
        ))
    return Aircraft(surfaces=surfaces, ref_point=np.array([0.25, 0.0, 0.0]))


def test_joined_edges_use_mean_camber_offset():
    ac = _joined_cambered_surfaces()
    lat = build_lattice(ac, SolverSettings(n_panels=4), n_chord=5)
    inner, outer = lat.surfaces
    ib = np.arange((inner.strips.stop - 1) * 5, inner.strips.stop * 5)
    oa = np.arange(outer.strips.start * 5, (outer.strips.start + 1) * 5)
    np.testing.assert_array_equal(lat.b[ib], lat.a[oa])
    x = lat.b[ib, 0]
    z = 0.5 * (airfoil("2412").normalized().camber(x) + airfoil("4412").normalized().camber(x))
    np.testing.assert_allclose(lat.b[ib, 2], z, rtol=0, atol=2e-15)
    assert any("camber offsets" in text and "inner" in text and "outer" in text for text in lat.join_warnings)
    transformed = transform_lattice(lat, rotation=rotation_matrix(roll=0.1, pitch=0.2))
    np.testing.assert_array_equal(transformed.b[ib], transformed.a[oa])


def test_equal_joined_camber_has_no_offset_warning():
    ac = _joined_cambered_surfaces()
    for sec in ac.surfaces[1].sections:
        sec.shape = "2412"
    lat = build_lattice(ac, SolverSettings(n_panels=4), n_chord=5)
    assert not any("camber offsets" in text for text in lat.join_warnings)


def test_cambered_chordwise_control_turns_geometric_normals():
    from ventorum.core.datatypes import ControlSurface
    from ventorum.geometry.lattice import _rotate

    ac = _aircraft()
    ac.surfaces[0].controls = [ControlSurface(
        name="flap", eta_start=0.0, eta_end=1.0, hinge_x_c=0.75, deflection=0.0,
    )]
    st = SolverSettings(n_panels=4)
    clean = build_lattice(ac, st, n_chord=8)
    ac.surfaces[0].controls[0].deflection = 0.12
    flap = build_lattice(ac, st, n_chord=8)
    for name in ("a", "b", "cp", "a_te", "b_te"):
        points = getattr(clean, name)
        expected = points.copy()
        for s in range(flap.n_strips):
            origin, end = clean.geom_cache["flap_hinges"][s]
            axis = (end - origin) / np.linalg.norm(end - origin)
            indices = np.arange(s * 8, (s + 1) * 8)
            indices = indices[points[indices, 0] > origin[0]]
            expected[indices] = origin + _rotate(points[indices] - origin, axis[None, :], 0.12)
        np.testing.assert_allclose(getattr(flap, name), expected, rtol=0, atol=2e-15)
    for s in range(flap.n_strips):
        for k in range(8):
            p = s * 8 + k
            expected = clean.normal_bc[p:p + 1]
            if k >= 6:
                expected = _rotate(expected, np.array([[0.0, 1.0, 0.0]]), 0.12)
            np.testing.assert_allclose(flap.normal_bc[p], expected[0], rtol=0, atol=2e-15)


@pytest.mark.parametrize("wrt", [("nodes",), ("planform",), ("alpha",), ("nodes", "alpha")])
@pytest.mark.parametrize("entry", ["gradients", "gradient_vjp", "check_gradients"])
def test_mean_line_gradient_chain_refusal_is_explicit(wrt, entry):
    import ventorum.sensitivity as sensitivity

    kwargs = {"wrt": wrt}
    if entry == "gradient_vjp":
        kwargs["seeds"] = {"CL": 1.0}
    else:
        kwargs["outputs"] = ("CL",)
    with pytest.raises(sensitivity.GradientScopeError, match="mean.line offset"):
        getattr(sensitivity, entry)(_aircraft(), FlightCondition(V_inf=30.0, alpha=0.05),
                                    SolverSettings(n_panels=4, n_chord=4), **kwargs)


@pytest.mark.parametrize("n_chord,mixed", [(1, False), (4, True)])
@pytest.mark.parametrize("wrt", [("nodes",), ("planform",)])
def test_fallback_gradient_chains_agree(n_chord, mixed, wrt):
    from ventorum.sensitivity import check_gradients

    gpu.set_precision("float64")
    ac = _aircraft(polar=LinearAirfoil(alpha_L0=-0.03, Cm0=-0.04))
    if mixed:
        ac.surfaces[0].sections[-1].shape = None
    result = check_gradients(ac, FlightCondition(V_inf=30.0, alpha=0.05),
                             SolverSettings(n_panels=4, n_chord=n_chord), outputs=("CL", "Cm"),
                             wrt=wrt, n_directions=0, rtol=1e-6, seed=0)
    assert result["pass"], result


@pytest.mark.gpu
def test_gpu_camber_moment_mask_keeps_shared_fallback_polar():
    from ventorum.core.datatypes import TabulatedAirfoil

    if not gpu.available():
        pytest.skip(gpu.unavailable_reason())
    angles = np.linspace(-0.2, 0.3, 12)
    polar = TabulatedAirfoil(alpha=angles, Cl_data=2 * np.pi * (angles + 0.03),
                             Cd_data=np.full_like(angles, 0.01), Cm_data=-0.08 + 0.02 * angles)
    ac = _aircraft(polar=polar)
    ac.surfaces.append(LiftingSurface(
        name="tail", semi_span=1.0, position=np.array([3.0, 0.0, 0.2]),
        sections=[WingSection(y_frac=e, chord=0.5, airfoil=polar) for e in (0.0, 1.0)],
    ))
    cpu = _solve(ac, n_chord=4, n_span=4)
    gpu.set_device("gpu")
    gpu.set_precision("float64")
    actual = _solve(ac, n_chord=4, n_span=4)
    assert actual.details["device"] == "gpu"
    cached = _solve(ac, n_chord=4, n_span=4)
    np.testing.assert_array_equal(cached.details["gamma"], actual.details["gamma"])
    for name in ("CL", "Cm", "CDi", "CD_total"):
        np.testing.assert_allclose(getattr(actual.totals, name), getattr(cpu.totals, name), rtol=1e-12, atol=1e-14)
    for a, b in zip(actual.spanwise, cpu.spanwise):
        np.testing.assert_allclose(a.Cm_section, b.Cm_section, rtol=1e-12, atol=1e-14)
    np.testing.assert_array_equal(polar.Cm_data, -0.08 + 0.02 * angles)


def test_viscous_increment_uses_geometric_normals():
    from ventorum.viscous.alpha_correction import _rhs_derivative

    lat = build_lattice(_aircraft(), SolverSettings(n_panels=4), n_chord=5)
    alpha, velocity, step = 0.05, 30.0, 1e-7
    cond = FlightCondition(V_inf=velocity, alpha=alpha)
    _, _, umap = assemble_vlm(lat, cond, use_symmetry=False)
    ua = np.array([-np.sin(alpha), 0.0, np.cos(alpha)])
    B = _rhs_derivative(lat, "vlm", umap, velocity, ua, np.arange(lat.n_strips))
    _, lower, _ = assemble_vlm(lat, FlightCondition(V_inf=velocity, alpha=alpha - step), use_symmetry=False)
    _, upper, _ = assemble_vlm(lat, FlightCondition(V_inf=velocity, alpha=alpha + step), use_symmetry=False)
    np.testing.assert_allclose(B.sum(axis=1), (lower - upper) / (2.0 * step), rtol=1e-9, atol=1e-8)
    assert np.ptp(B.sum(axis=1)[:lat.n_chord]) > 1e-3


def test_ground_camber_note_is_in_result_without_runtime_warning():
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="camber modelled by a turned normal")
        result = _solve(_aircraft(None, polar=LinearAirfoil(alpha_L0=-0.03)), h=0.2, n_span=4)
    note = "camber modelled by a turned normal (no mean line given); near the ground the chordwise load and the camber height are approximate"
    assert note in result.totals.trust.warnings


@pytest.mark.parametrize("h", [None, 0.2])
def test_mirrored_cambered_surfaces_equal_symmetric_surface(h):
    symmetric = _aircraft()
    right = copy.deepcopy(symmetric.surfaces[0])
    right.is_symmetric = False
    paired = Aircraft(surfaces=[right, right.mirrored()], ref_point=symmetric.ref_point.copy())
    a = _solve(symmetric, h=h, n_chord=5, n_span=4)
    b = _solve(paired, h=h, n_chord=5, n_span=4)
    perm = np.r_[20:40, 0:20]
    for name in ("a", "b", "cp", "a_te", "b_te", "normal_bc"):
        np.testing.assert_allclose(getattr(a.details["lattice"], name), getattr(b.details["lattice"], name)[perm],
                                   rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(a.details["gamma"], b.details["gamma"][perm], rtol=1e-12, atol=1e-14)
    for name in ("CL", "CDi", "Cm"):
        np.testing.assert_allclose(getattr(a.totals, name), getattr(b.totals, name), rtol=1e-12, atol=1e-14)


def test_joined_fallback_edge_contributes_zero_offset():
    ac = _joined_cambered_surfaces()
    for sec in ac.surfaces[0].sections:
        sec.shape = None
    lat = build_lattice(ac, SolverSettings(n_panels=4), n_chord=5)
    inner, outer = lat.surfaces
    ib = np.arange((inner.strips.stop - 1) * 5, inner.strips.stop * 5)
    oa = np.arange(outer.strips.start * 5, (outer.strips.start + 1) * 5)
    np.testing.assert_array_equal(lat.b[ib], lat.a[oa])
    np.testing.assert_allclose(lat.b[ib, 2], 0.5 * airfoil("4412").normalized().camber(lat.b[ib, 0]),
                               rtol=0, atol=2e-15)
    assert not np.any(lat.geometric_camber[inner.strips])


def test_three_joined_edges_each_contribute_once():
    right = _aircraft("2412").surfaces[0]
    right.is_symmetric = False
    left = right.mirrored()
    for sec in left.sections:
        sec.shape = "4412"
    fin = LiftingSurface(name="fin", semi_span=2.0, dihedral=np.pi / 2.0, is_symmetric=False,
                         sections=[WingSection(y_frac=e, chord=1.0, shape="3412") for e in (0.0, 1.0)])
    ac = Aircraft(surfaces=[right, left, fin])
    lat = build_lattice(ac, SolverSettings(n_panels=4), n_chord=5)
    root_right = lat.a[:5]
    root_left = lat.b[35:40]
    root_fin = lat.a[40:45]
    np.testing.assert_array_equal(root_right, root_left)
    np.testing.assert_array_equal(root_right, root_fin)
    x = root_right[:, 0]
    expected_z = (airfoil("2412").normalized().camber(x) + airfoil("4412").normalized().camber(x)) / 3.0
    expected_y = -airfoil("3412").normalized().camber(x) / 3.0
    np.testing.assert_allclose(root_right[:, 2], expected_z, rtol=0, atol=2e-15)
    np.testing.assert_allclose(root_right[:, 1], expected_y, rtol=0, atol=2e-15)


def test_claf_normals_follow_clamped_mean_line_outside_chord():
    ac = _aircraft(polar=LinearAirfoil(a0=4 * np.pi))
    lat = build_lattice(ac, SolverSettings(n_panels=4), n_chord=5)
    outside = lat.cp[:, 0] > 1.0
    assert np.any(outside)
    z = airfoil("4412").normalized().camber(lat.cp[outside, 0])
    np.testing.assert_allclose(lat.cp[outside, 2], z, rtol=0, atol=2e-15)
    np.testing.assert_allclose(lat.normal_bc[outside], lat.normal[lat.panel_strip[outside]], rtol=0, atol=2e-15)
