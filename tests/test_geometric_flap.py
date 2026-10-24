"""Checks of geometric flap deflection on the vortex lattice mean surface."""

from __future__ import annotations

import copy
import warnings

import numpy as np
import pytest

import ventorum as vt
from ventorum import gpu
from ventorum.aero import vortex
from ventorum.aero.hinge import _bound_forces, _hinge_line
from ventorum.aero.loads import LEG_SHARED, leg_force_rules
from ventorum.aero.system import build_sources, make_unknown_map
from ventorum.core.errors import GroundStrikeError
from ventorum.geometry import lattice as mesh_mod
from ventorum.geometry import lattice_cache


def _aircraft(delta=0.0, *, shape=None, symmetric=True, partial=False):
    return vt.Aircraft(surfaces=[vt.LiftingSurface(
        name="wing", semi_span=4.0,
        sections=[vt.WingSection(y_frac=e, chord=1.0, shape=shape) for e in (0.0, 1.0)],
        controls=[vt.ControlSurface(
            "flap", 0.25 if partial else 0.0, 0.75 if partial else 1.0,
            0.7, delta, symmetric=symmetric,
        )],
    )], ref_point=np.array([0.25, 0.0, 0.0]))


def _settings(n_chord=10, n_span=8):
    return vt.SolverSettings(n_chord=n_chord, n_panels=n_span, chord_spacing="uniform")


def _solve(ac, *, h=None, alpha=0.0, settings=None):
    return vt.VortexLatticeSolver().solve(
        ac, vt.FlightCondition(V_inf=30.0, alpha=alpha, h=h), settings or _settings(),
    )


def _old_solve(monkeypatch, ac, **kwargs):
    """Evaluate the former normal-only model with the same hinge snap."""
    with monkeypatch.context() as patch:
        patch.setattr(mesh_mod, "_rotate_flap_points", lambda points, *_: points.copy())
        lattice_cache.clear()
        result = _solve(ac, **kwargs)
    lattice_cache.clear()
    return result


def test_transformed_hinge_endpoints():
    from ventorum.geometry.transform import rotation_matrix, transform_lattice

    lat = mesh_mod.build_lattice(_aircraft(0.1, shape="2412"), _settings(), n_chord=10)
    rotation = rotation_matrix(0.2, -0.1, 0.3)
    about = np.array([0.2, 0.4, -0.1])
    shift = np.array([1.0, -2.0, 3.0])
    moved = transform_lattice(lat, rotation, about, shift)
    for strip, points in lat.geom_cache["flap_hinges"].items():
        expected = (np.asarray(points) - about) @ rotation.T + about + shift
        np.testing.assert_allclose(moved.geom_cache["flap_hinges"][strip], expected, rtol=0, atol=1e-15)
        mid, axis = _hinge_line(lat, strip, 0.7)
        new_mid, new_axis = _hinge_line(moved, strip, 0.7)
        np.testing.assert_allclose(new_mid, (mid - about) @ rotation.T + about + shift, atol=1e-15)
        np.testing.assert_allclose(new_axis, axis @ rotation.T, atol=1e-15)
    assert moved.geom_cache["flap_hinges"] is not lat.geom_cache["flap_hinges"]


@pytest.mark.parametrize("n_chord", [None, 10])
def test_ground_flap_refused_before_solve(monkeypatch, n_chord):
    from ventorum.ground_effect.solver import analyze_ground_effect

    def forbidden(*args, **kwargs):
        pytest.fail("A ground strike must be refused before the solver is called.")

    monkeypatch.setattr(vt.VortexLatticeSolver, "solve", forbidden)
    with pytest.raises(GroundStrikeError):
        analyze_ground_effect(_aircraft(0.3), 0.03, alpha_deg=0.0,
                              settings=_settings(n_chord), compute_strike_limit=False)


def test_ground_probe_deflected_root_and_tips():
    from ventorum.ground_effect.solver import clearance_info, find_bank_strike_limit, place_ground

    ac = _aircraft(0.3)
    settings = _settings()
    rp = ac.moment_reference()
    gp, probe = place_ground(ac, settings, 0.2, 0.0, 0.0, 0.0, rp, "te")
    assert probe.n_chord == settings.n_chord
    assert np.all(probe.te_left[:, 2] < 0.0)
    np.testing.assert_allclose(gp.height(probe.te_left), 0.2, rtol=0, atol=1e-15)
    clear = clearance_info(probe, gp, rp)
    np.testing.assert_allclose([clear["h_tip_left"], clear["h_tip_right"]], 0.2, rtol=0, atol=1e-15)
    assert find_bank_strike_limit(ac, settings, 0.03, 0.0, 0.0, rp, "ref") == 0.0


def test_ground_probe_automatic_final_count():
    from ventorum.ground_effect.solver import place_ground, prepare_ground_case

    ac = _aircraft(0.2)
    case = prepare_ground_case(ac, 0.2, 0.0, 0.0, 0.0, settings=_settings(None), height_ref="te")
    settings = case["settings"]
    assert settings.n_chord is not None
    gp, lat = place_ground(ac, settings, 0.2, 0.0, 0.0, 0.0, case["ref_point"], "te")
    assert lat.n_chord == settings.n_chord
    np.testing.assert_array_equal(gp.normal, case["ground"].normal)
    assert gp.offset == case["ground"].offset
    np.testing.assert_allclose(gp.height(lat.te_left), 0.2, rtol=0, atol=1e-15)


def test_ground_root_quarter_chord_aft_of_hinge():
    from ventorum.ground_effect.solver import place_ground

    ac = _aircraft(0.2)
    ac.surfaces[0].controls[0].hinge_x_c = 0.2
    gp, _ = place_ground(ac, _settings(), 0.2, 0.0, 0.0, 0.0, ac.moment_reference(), "qc")
    root = np.array([0.2 + 0.05 * np.cos(0.2), 0.0, -0.05 * np.sin(0.2)])
    np.testing.assert_allclose(gp.height(root[None, :]), 0.2, rtol=0, atol=1e-15)


@pytest.fixture(autouse=True)
def _restore_backends():
    old_backend = vortex._kernel_backend
    old_device, old_precision = gpu.get_device(), gpu._precision
    gpu.set_device("cpu")
    lattice_cache.clear()
    yield
    lattice_cache.clear()
    vortex.set_kernel_backend(old_backend)
    gpu.set_device(old_device)
    gpu.set_precision(old_precision)


@pytest.mark.parametrize("shape", [None, "0012", "2412"])
@pytest.mark.parametrize("h", [None, 0.2])
def test_undeflected_bit_identical(monkeypatch, shape, h):
    ac = _aircraft(shape=shape)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        old = _old_solve(monkeypatch, ac, h=h)
        new = _solve(ac, h=h)
    for name in ("a", "b", "cp", "a_te", "b_te", "normal_bc"):
        assert np.array_equal(getattr(old.details["lattice"], name), getattr(new.details["lattice"], name))
    assert np.array_equal(old.details["gamma"], new.details["gamma"])
    for name in ("CL", "CDi", "Cm", "CY", "Cl", "Cn"):
        assert getattr(old.totals, name) == getattr(new.totals, name)


@pytest.mark.parametrize("shape", [None, "2412"])
@pytest.mark.parametrize("symmetric", [False, True])
def test_flap_points_on_rotated_surface(shape, symmetric):
    delta = np.radians(13.0)
    ac = _aircraft(shape=shape, symmetric=symmetric)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        base = mesh_mod.build_lattice(ac, _settings(), n_chord=10)
        ac.set_deflection("flap", delta)
        lat = mesh_mod.build_lattice(ac, _settings(), n_chord=10)
    from ventorum.airfoil import airfoil

    zh = 0.0 if shape is None else airfoil(shape).normalized().camber(0.7)
    for s in range(lat.n_strips):
        angle = delta if (lat.strip_is_right[s] or symmetric) else -delta
        origin = np.array([0.7, 0.0, zh])
        c, sn = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, 0, sn], [0, 1, 0], [-sn, 0, c]])
        panels = slice(s * 10, (s + 1) * 10)
        for name in ("a", "b", "cp", "a_te", "b_te"):
            points = getattr(base, name)[panels]
            aft = points[:, 0] > 0.7
            expected = points.copy()
            expected[aft] = (points[aft] - origin) @ rotation.T + origin
            np.testing.assert_allclose(getattr(lat, name)[panels], expected, rtol=0, atol=3e-15)
        expected = base.normal_bc[panels].copy()
        expected[7:] = expected[7:] @ rotation.T
        np.testing.assert_allclose(lat.normal_bc[panels], expected, rtol=0, atol=3e-15)


def test_rigid_flap_rotation():
    """Check a full-chord rotation in the geometric operation, about x = 0 m.

    The public control validator and interior hinge snap exclude x/c = 0.
    This check does not change either rule to create a full-chord control.
    """
    delta = 0.13
    ac = _aircraft()
    ac.surfaces[0].controls = []
    flat = mesh_mod.build_lattice(ac, _settings(), n_chord=10)
    ac.surfaces[0].incidence = delta
    # Incidence pivots about c/4. Translate that pivot to the leading edge.
    ac.surfaces[0].position = np.array([0.25 * (np.cos(delta) - 1), 0, -0.25 * np.sin(delta)])
    inclined = mesh_mod.build_lattice(ac, _settings(), n_chord=10)
    for name in ("a", "b", "cp", "a_te", "b_te"):
        moved = mesh_mod._rotate_flap_points(getattr(flat, name), np.zeros(3), np.array([0, 1, 0]), delta)
        np.testing.assert_allclose(moved, getattr(inclined, name), rtol=0, atol=3e-15)


@pytest.mark.parametrize("shape", [None, "2412"])
def test_swept_twisted_hinge(shape):
    ac = _aircraft(shape=shape, symmetric=False)
    wing = ac.surfaces[0]
    wing.incidence, wing.dihedral, wing.sweep_le = 0.05, 0.12, 0.2
    wing.sections[1].chord, wing.sections[1].twist = 0.7, 0.08
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        base = mesh_mod.build_lattice(ac, _settings(), n_chord=10)
        ac.set_deflection("flap", 0.13)
        lat = mesh_mod.build_lattice(ac, _settings(), n_chord=10)
    for s in range(lat.n_strips):
        hinge, axis = _hinge_line(base, s, 0.7)
        da = 0.13 if lat.strip_is_right[s] else -0.13
        c, sn = np.cos(da), np.sin(da)
        for name in ("a", "b", "cp"):
            panels = slice(s * 10 + 7, (s + 1) * 10)
            v = getattr(base, name)[panels] - hinge
            expected = hinge + c * v + sn * np.cross(axis, v) + (1 - c) * (v @ axis)[:, None] * axis
            np.testing.assert_allclose(getattr(lat, name)[panels], expected, rtol=0, atol=3e-15)
        np.testing.assert_allclose(_hinge_line(lat, s, 0.7)[0], hinge, rtol=0, atol=3e-15)


def test_small_deflection_limit(monkeypatch):
    """At zero incidence the coefficient difference is cubic in delta [rad]."""
    errors = []
    # Stay below the vortex core scale to test the asymptotic limit.
    for delta in (1e-4, 5e-5, 2.5e-5):
        ac = _aircraft(delta)
        new = _solve(ac).totals
        old = _old_solve(monkeypatch, ac).totals
        errors.append(np.abs(np.array([new.CL - old.CL, new.Cm - old.Cm])) / delta)
    rates = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    print(f"small-deflection derivative convergence orders: {rates.tolist()}")
    assert np.all((rates > 1.8) & (rates < 2.2))


def test_ground_strike_of_flap():
    from ventorum.core.errors import GroundStrikeError

    _solve(_aircraft(), h=0.05)
    with pytest.raises(GroundStrikeError):
        _solve(_aircraft(np.radians(20)), h=0.05)


def test_upward_flap_clearance_uses_actual_points():
    """An upward flap clears the plane even when its reference TE is below it."""
    result = _solve(_aircraft(-0.4), h=0.11, alpha=0.2)
    lat, ground = result.details["lattice"], result.details["ground"]
    assert np.min(ground.height(lat.surfaces[0].edge_te)) < 0
    assert np.min(ground.height(lat.all_points())) > 0


def test_control_point_beyond_hinge_moves():
    """A high lift slope can put a forward panel control point aft of the hinge."""
    ac = _aircraft()
    for sec in ac.surfaces[0].sections:
        sec.airfoil.a0 = 4 * np.pi
    base = mesh_mod.build_lattice(ac, _settings(), n_chord=10)
    ac.set_deflection("flap", 0.2)
    lat = mesh_mod.build_lattice(ac, _settings(), n_chord=10)
    panels = 10 * np.arange(lat.n_strips) + 6
    assert np.all(base.cp[panels, 0] > 0.7)
    expected = np.column_stack([
        0.7 + (base.cp[panels, 0] - 0.7) * np.cos(0.2),
        base.cp[panels, 1], -(base.cp[panels, 0] - 0.7) * np.sin(0.2),
    ])
    np.testing.assert_allclose(lat.cp[panels], expected, rtol=0, atol=3e-15)
    np.testing.assert_array_equal(lat.a[panels], base.a[panels])
    gpu.set_precision("float64")
    checked = vt.check_gradients(
        ac, vt.FlightCondition(V_inf=30, alpha=0.04), _settings(10, 4),
        outputs=("CL", "Cm"), wrt=("controls",), rtol=1e-6,
    )
    assert checked["pass"], checked


@pytest.mark.parametrize("alignment", ["freestream", "body"])
def test_trailing_edge_moves(alignment):
    delta = 0.2
    st = _settings()
    st.wake_alignment = alignment
    result = _solve(_aircraft(delta), settings=st, alpha=0.05)
    lat = result.details["lattice"]
    expected = np.array([0.7 + 0.3 * np.cos(delta), -0.3 * np.sin(delta)])
    for name in ("a_te", "b_te"):
        np.testing.assert_allclose(getattr(lat, name)[:, [0, 2]], np.tile(expected, (lat.n_panels, 1)), atol=2e-15)
    src = build_sources(lat, result.details["wake_dir"], make_unknown_map(lat, False))
    np.testing.assert_array_equal(src.a_te, lat.a_te)
    np.testing.assert_array_equal(src.b_te, lat.b_te)
    direction = np.array([1, 0, 0]) if alignment == "body" else np.array([np.cos(0.05), 0, np.sin(0.05)])
    np.testing.assert_allclose(result.details["wake_dir"], direction, atol=2e-15)


def test_hinge_moment_axis():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = _solve(_aircraft(0.2, shape="2412"))
    from ventorum.airfoil import airfoil

    lat = result.details["lattice"]
    zh = airfoil("2412").normalized().camber(0.7)
    expected_moments = {}
    for side, right in (("right", True), ("left", False)):
        moment = 0.0
        for s in np.flatnonzero(lat.strip_is_right == right):
            mid, axis = _hinge_line(lat, s, 0.7)
            np.testing.assert_allclose(mid[[0, 2]], [0.7, zh], rtol=0, atol=2e-15)
            np.testing.assert_allclose(axis, [0, 1, 0], rtol=0, atol=2e-15)
            panels = s * 10 + np.arange(7, 10)
            points, forces = _bound_forces(result, panels)
            moment += np.sum(np.cross(points - mid, forces) @ axis)
        expected_moments[side] = moment
    for item in vt.hinge_moments(result):
        np.testing.assert_allclose(item.H, expected_moments[item.side], rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("h", [None, 0.2])
@pytest.mark.parametrize("delta", [0.0, 0.1])
def test_gradients_deflection(h, delta):
    gpu.set_precision("float64")
    ac = _aircraft(delta, symmetric=False, partial=True)
    checked = vt.check_gradients(
        ac, vt.FlightCondition(V_inf=30, alpha=0.04, h=h), _settings(4, 4),
        outputs=("CL", "Cm", "Cl", "CY"), wrt=("controls",), rtol=1e-6,
    )
    assert checked["pass"], checked


@pytest.mark.parametrize("h", [None, 0.2])
def test_backend_parity(h):
    ac = _aircraft(0.15, symmetric=False, partial=True)
    vortex.set_kernel_backend("numpy")
    ref = _solve(ac, h=h)
    vortex.set_kernel_backend("numba")
    result = _solve(ac, h=h)
    np.testing.assert_allclose(result.details["gamma"], ref.details["gamma"], rtol=1e-12, atol=1e-14)
    for name in ("CL", "CDi", "Cm", "CY", "Cl", "Cn"):
        np.testing.assert_allclose(getattr(result.totals, name), getattr(ref.totals, name), rtol=1e-12, atol=1e-14)


@pytest.mark.gpu
@pytest.mark.parametrize("h", [None, 0.2])
@pytest.mark.parametrize("precision,tol", [("float64", 1e-12), ("float32", 1e-5)])
def test_backend_parity_gpu(h, precision, tol):
    if not gpu.available():
        pytest.skip(gpu.unavailable_reason())
    ac = _aircraft(0.15, symmetric=False, partial=True)
    ref = _solve(ac, h=h, alpha=0.07)
    gpu.set_device("gpu")
    gpu.set_precision(precision)
    result = _solve(ac, h=h, alpha=0.07)
    assert result.details["device"] == "gpu"
    for name in ("CL", "CDi", "Cm", "CY", "Cl", "Cn"):
        a, b = getattr(ref.totals, name), getattr(result.totals, name)
        assert abs(a - b) <= max(tol * abs(a), 1e-14)


def test_aileron_side_force_reported():
    results = [_solve(_aircraft(d, symmetric=False, partial=True), alpha=0.07) for d in (0.1, -0.1)]
    cy = [r.totals.CY for r in results]
    print(f"aileron delta +/-0.1 rad at alpha=0.07 rad: CY={cy}")
    assert np.all(np.isfinite(cy))
    np.testing.assert_allclose(cy[0], -cy[1], rtol=1e-12, atol=1e-14)
    # Separated interior edges retain the existing tangential leg rule.
    lat = results[0].details["lattice"]
    _, _, kind = leg_force_rules(lat)
    for s in range(1, lat.n_strips - 1):
        panels = s * lat.n_chord + np.arange(lat.n_chord)
        assert np.all(kind[panels] == LEG_SHARED)
        assert np.all(kind[lat.n_panels + panels] == LEG_SHARED)


@pytest.mark.slow
@pytest.mark.parametrize("h", [None, 0.2])
def test_thin_flapped_wing_against_panel3d(monkeypatch, h):
    from ventorum.airfoil import airfoil
    from ventorum.panel3d.mesh import mesh_lifting_surface
    from ventorum.panel3d.solver import PanelSolver

    ac = _aircraft(np.radians(10), shape="0002")
    # Forty panels put the 0.7 hinge on an exact edge under the fixed snap
    # rule. The half panel mesh and the full lattice stay below 2000 panels.
    mesh = mesh_lifting_surface(ac.surfaces[0], airfoil("0002").coords, n_chord=40, n_span=23, half=True)
    assert len(mesh.panels) <= 2000
    panel = PanelSolver(mesh, S_ref=8, b_ref=8, c_ref=1, ref_point=ac.ref_point, backend="numba", far_ratio=np.inf)
    p = panel.solve(vt.FlightCondition(V_inf=30, alpha=np.radians(5), h=h))
    st = _settings(40, 23)
    result = _solve(ac, h=h, alpha=np.radians(5), settings=st)
    assert result.details["lattice"].n_panels <= 2000
    assert result.details["lattice"].control_info[0]["hinge_x_c_eff"] == 0.7
    print(f"flap mesh h={h}: 3DP panels={len(mesh.panels)}, "
          f"VLM panels={result.details['lattice'].n_panels}, hinge_x_c_eff=0.7")
    new = result.totals
    old = _old_solve(monkeypatch, copy.deepcopy(ac), h=h, alpha=np.radians(5), settings=st).totals
    print(f"flap h={h}: old CL={old.CL:.16g} Cm={old.Cm:.16g}; "
          f"new CL={new.CL:.16g} Cm={new.Cm:.16g}; 3DP CL={p.CL:.16g} Cm={p.Cm:.16g}")
    try:
        assert abs(new.CL - p.CL) <= 0.03 * abs(p.CL)
    finally:
        # Check the moment bound even when the lift bound fails.
        assert abs(new.Cm - p.Cm) <= max(0.10 * abs(p.Cm), 0.003)
