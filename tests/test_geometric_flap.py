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
from ventorum.geometry.controls import CONTROL_JOIN, ControlJoinField


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
    # An antisymmetric control through the root has a join there (from -delta
    # to +delta over 0.2 chords, see test_root_join_antisymmetric); the strips
    # next to the root are checked there.
    half = 0.5 * CONTROL_JOIN * 1.0
    for s in range(lat.n_strips):
        if not symmetric and min(abs(lat.le_left[s, 1]), abs(lat.le_right[s, 1])) < half:
            continue
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
    hinges = lat.geom_cache["flap_hinges"]
    angle = lat.geom_cache["flap_field"]["angle"]

    def turn(p, hinge, axis, da):
        c, sn = np.cos(da), np.sin(da)
        v = p - hinge
        return hinge + c * v + sn * np.cross(axis, v) + (1 - c) * (v @ axis)[:, None] * axis

    n = lat.n_strips
    for s in range(n):
        # The axis of an edge node is the central difference of the hinge
        # points of its neighbour nodes.
        h_l, h_r = hinges[s]
        ax_l = h_r - (hinges[s - 1][0] if s > 0 else h_l)
        ax_r = (hinges[s + 1][1] if s + 1 < n else h_r) - h_l
        ax_l, ax_r = ax_l / np.linalg.norm(ax_l), ax_r / np.linalg.norm(ax_r)
        panels = slice(s * 10 + 7, (s + 1) * 10)
        np.testing.assert_allclose(lat.a[panels], turn(base.a[panels], h_l, ax_l, angle[s, 0]), rtol=0, atol=3e-15)
        np.testing.assert_allclose(lat.b[panels], turn(base.b[panels], h_r, ax_r, angle[s, 1]), rtol=0, atol=3e-15)
        hinge, _ = _hinge_line(base, s, 0.7)
        np.testing.assert_allclose(_hinge_line(lat, s, 0.7)[0], hinge, rtol=0, atol=3e-15)
        if s + 1 < n:
            # The two strips at a node turn it about the same line: no gap.
            np.testing.assert_allclose(lat.te_right[s], lat.te_left[s + 1], rtol=0, atol=3e-15)


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


# --- hinge nodes of the chordwise legs ------------------------------------------

def _kinked_sources(h=None):
    """Return a lattice with a deflected part-span flap, its sources and the evaluation points."""
    ac = _aircraft(0.2, symmetric=False, partial=True)
    lat = mesh_mod.build_lattice(ac, _settings(6, 6), n_chord=6)
    ground = None
    if h is not None:
        from ventorum.aero.system import make_ground_plane
        ground = make_ground_plane(lat, h, 0.05, ref_point=ac.ref_point)
    wake = np.array([np.cos(0.05), 0.0, np.sin(0.05)])
    src = build_sources(lat, wake, make_unknown_map(lat, False), ground)
    points = np.vstack([lat.cp, 0.5 * (lat.a_te + lat.a_kink), 0.5 * (lat.a + lat.b)])
    return lat, src, points


def test_hinge_nodes_on_deflected_surface():
    """The forward legs of a deflected strip run bound point -> hinge point -> deflected TE."""
    lat, src, _ = _kinked_sources()
    assert src.has_kinks
    n_c = lat.n_chord
    hinges = lat.geom_cache["flap_hinges"]
    kinked = set(src.kinked().tolist())
    for s in range(lat.n_strips):
        for k in range(n_c):
            p = s * n_c + k
            forward = s in hinges and lat.a[p, 0] < 0.7
            if forward:
                np.testing.assert_array_equal(lat.a_kink[p], hinges[s][0])
                np.testing.assert_array_equal(lat.b_kink[p], hinges[s][1])
                assert p in kinked
            else:
                np.testing.assert_array_equal(lat.a_kink[p], lat.a_te[p])
                np.testing.assert_array_equal(lat.b_kink[p], lat.b_te[p])
                assert p not in kinked
    # An undeflected chordwise control keeps its nodes on the straight legs
    # (continuous loads in the deflection); a wing without controls has none.
    zero = mesh_mod.build_lattice(_aircraft(0.0), _settings(), n_chord=10)
    assert zero.a_kink is not None
    np.testing.assert_allclose(zero.a_kink[zero.a[:, 0] < 0.7, 0], 0.7, rtol=0, atol=1e-15)
    plain = _aircraft(0.0)
    plain.surfaces[0].controls = []
    assert mesh_mod.build_lattice(plain, _settings(), n_chord=10).a_kink is None


def _cpu_backends():
    out = ["numba"]
    if vortex._HAVE_CYTHON and vortex._cy_has_kinks():
        out.append("cython")
    if vortex._HAVE_TORCH:
        out.append("torch")
    return out


@pytest.mark.parametrize("h", [None, 0.2])
@pytest.mark.parametrize("backend", _cpu_backends())
def test_hinge_node_kernel_parity(h, backend):
    """Every compiled kernel with leg nodes agrees with the numpy reference to round-off."""
    lat, src, P = _kinked_sources(h)
    n = lat.n_panels
    normals = np.tile(lat.normal_bc, (3, 1))
    g = np.linspace(0.3, 1.7, n)
    tg = vortex.Targets(group=np.zeros(P.shape[0], dtype=np.int64), rc=np.full(P.shape[0], 0.01))

    def run():
        return (vortex.influence_matrix(P, normals, src, n),
                vortex.influence_matrix(P, normals, src, n, cache={}, cache_key="k"),
                vortex.induced_velocity(P, src, g),
                vortex.induced_velocity(P, src, g, cache={}, cache_key="k"),
                vortex.induced_velocity(P, src, g, tg),
                vortex.horseshoe_velocity_tensor(P, src),
                vortex.velocity_tensor_unknowns(P, src, n))

    vortex.set_kernel_backend("numpy")
    ref = run()
    vortex.set_kernel_backend(backend)
    for r, x in zip(ref, run()):
        scale = max(float(np.max(np.abs(r))), 1.0)
        np.testing.assert_allclose(x, r, rtol=0, atol=1e-12 * scale)


def test_hinge_node_matches_split_legs():
    """A leg with a node equals the straight horseshoe plus the two leg segments, minus the straight leg."""
    lat, src, P = _kinked_sources()
    idx = src.kinked()
    vortex.set_kernel_backend("numpy")
    v = vortex.horseshoe_velocity_tensor(P, src)[:, idx]
    straight = vortex.HorseshoeSet(a=src.a, b=src.b, a_te=src.a_te, b_te=src.b_te, wake_dir=src.wake_dir,
                                   rc=src.rc, sign=src.sign, column=src.column, group=src.group)
    v0 = vortex.horseshoe_velocity_tensor(P, straight)[:, idx]
    rc = src.rc[idx]
    seg = vortex.segment_velocity
    expected = (v0 - seg(P, src.a_te[idx], src.a[idx], rc) - seg(P, src.b[idx], src.b_te[idx], rc)
                + seg(P, src.a_te[idx], src.a_kink[idx], rc) + seg(P, src.a_kink[idx], src.a[idx], rc)
                + seg(P, src.b[idx], src.b_kink[idx], rc) + seg(P, src.b_kink[idx], src.b_te[idx], rc))
    np.testing.assert_allclose(v, expected, rtol=0, atol=1e-10 * np.max(np.abs(v)))


def _flap_slope(monkeypatch, old, alpha, n_span=20, n_chord=16):
    """Return dCL/ddelta [1/rad] at a small deflection by a central difference."""
    st = vt.SolverSettings(n_chord=n_chord, n_panels=n_span, chord_spacing="uniform")
    d = np.radians(0.1)
    cl = []
    for sign in (1.0, -1.0):
        ac = _aircraft(sign * d)
        ac.surfaces[0].controls[0].hinge_x_c = 0.75
        res = _old_solve(monkeypatch, ac, alpha=alpha, settings=st) if old else _solve(ac, alpha=alpha, settings=st)
        cl.append(res.totals.CL)
    lattice_cache.clear()
    return (cl[0] - cl[1]) / (2 * d)


def test_small_deflection_slope_at_incidence(monkeypatch):
    """At 3 deg incidence the flap slope dCL/ddelta matches the normal-turn model within 1 %.

    With straight forward legs the leg forces made the slope 24 % low. The
    test at zero incidence does not see that error.
    """
    new = _flap_slope(monkeypatch, False, np.radians(3.0))
    old = _flap_slope(monkeypatch, True, np.radians(3.0))
    print(f"dCL/ddelta at alpha 3 deg: geometric {new:.6f}, normal turn {old:.6f}")
    assert abs(new - old) <= 0.01 * abs(old)


@pytest.mark.slow
def test_full_span_flap_spanwise_refinement(monkeypatch):
    """A full-span flap converges with the spanwise mesh and stays near the normal-turn model."""
    ac = _aircraft(np.radians(10.0))
    out = {}
    for n_span in (24, 160):
        st = vt.SolverSettings(n_chord=20, n_panels=n_span)
        out[n_span] = (_solve(ac, settings=st).totals.CL, _old_solve(monkeypatch, ac, settings=st).totals.CL)
    print(f"full-span flap CL (geometric, normal turn): {out}")
    assert abs(out[160][0] - out[24][0]) <= 0.02 * abs(out[160][0])
    assert abs(out[160][0] - out[160][1]) <= 0.03 * abs(out[160][1])


# --- sealed side edges (joins) and leg-force rules -------------------------------

def test_join_field_geometry():
    """The join is linear over 0.2 local chords, centred on the control edge, on every mesh."""
    c = vt.ControlSurface("flap", 0.25, 0.75, 0.7, 0.2)
    field = ControlJoinField([c], "plain", lambda e: 2.0, lambda e: 4.0)
    half = 0.5 * CONTROL_JOIN * 2.0 / 4.0

    def delta(f, e):
        return sum(k * cc.deflection for cc, k in f.at(e)[0])

    for e0, sgn in ((0.25, 1.0), (0.75, -1.0)):
        assert delta(field, e0) == pytest.approx(0.1, abs=1e-15)
        assert delta(field, e0 - sgn * half) == 0.0
        assert delta(field, e0 + sgn * half) == pytest.approx(0.2, abs=1e-15)
        assert delta(field, e0 + sgn * 0.5 * half) == pytest.approx(0.15, abs=1e-15)
        assert field.at(e0 - sgn * 0.5 * half)[1] is c
    assert delta(field, 0.5) == 0.2 and delta(field, 0.0) == 0.0 and delta(field, 1.0) == 0.0
    # A symmetric control through the root has no join there; an
    # antisymmetric one goes from -delta to +delta over the root.
    sym = ControlJoinField([vt.ControlSurface("f", 0.0, 0.5, 0.7, 0.2)], "symmetric",
                           lambda e: 2.0, lambda e: 4.0)
    assert delta(sym, 0.0) == 0.2
    anti = ControlJoinField([vt.ControlSurface("f", 0.0, 0.5, 0.7, 0.2, symmetric=False)], "symmetric",
                            lambda e: 2.0, lambda e: 4.0)
    assert delta(anti, 0.0) == 0.0
    assert delta(anti, -0.5 * half) == pytest.approx(-0.1, abs=1e-15)


@pytest.mark.parametrize("n_span", [8, 13])
def test_join_sealed_side_edges(n_span):
    """A part-span flap has no gap at its side edges, and the join is the same on every mesh."""
    delta = np.radians(10.0)
    lat = mesh_mod.build_lattice(_aircraft(delta, partial=True), _settings(10, n_span), n_chord=10)
    np.testing.assert_allclose(lat.te_right[:-1], lat.te_left[1:], rtol=0, atol=3e-15)
    field = lat.geom_cache["flap_field"]
    half = 0.5 * CONTROL_JOIN / 4.0
    for s in range(lat.n_strips):
        y = abs(lat.le_left[s, 1]) / 4.0
        if np.isfinite(field["hinge"][s]):
            expected = delta * min(1.0, max(0.0, (y - (0.25 - half)) / (2 * half),)) * min(
                1.0, max(0.0, ((0.75 + half) - y) / (2 * half)))
            assert field["angle"][s, 0] == pytest.approx(expected, abs=1e-14)
        else:
            assert y <= 0.25 - half + 1e-12 or y >= 0.75 + half - 1e-12


def test_root_join_dihedral():
    """A symmetric flap through the root of a wing with dihedral keeps the root points together."""
    ac = _aircraft(np.radians(10.0))
    ac.surfaces[0].dihedral = np.radians(5.0)
    lat = mesh_mod.build_lattice(ac, _settings(), n_chord=10)
    root = int(np.argmin(np.abs(lat.le_right[:, 1])))
    np.testing.assert_allclose(lat.te_right[root], lat.te_left[root + 1], rtol=0, atol=3e-15)
    assert abs(lat.te_right[root, 1]) <= 1e-15


def test_root_join_antisymmetric():
    """An antisymmetric control through the root turns the root node by zero (the middle of its join)."""
    lat = mesh_mod.build_lattice(_aircraft(0.2, symmetric=False), _settings(), n_chord=10)
    root = int(np.argmin(np.abs(lat.le_right[:, 1])))
    field = lat.geom_cache["flap_field"]
    assert field["angle"][root, 1] == pytest.approx(0.0, abs=1e-15)
    assert field["angle"][root + 1, 0] == pytest.approx(0.0, abs=1e-15)
    np.testing.assert_allclose(lat.te_right[root], lat.te_left[root + 1], rtol=0, atol=3e-15)


def test_leg_rules_shared_aft_segments():
    """Regression: the copies of the shared aft leg segment of one strip are not a fold.

    All forward panels of a strip share the aft segment of their legs (hinge
    node to trailing edge). The fold test of the leg-force rule took two
    copies of one strip as a fold of pi, and the aft segments became kink
    legs with no force.
    """
    from ventorum.aero.loads import LEG_KINK, _leg_force_rules_geometric, leg_segments

    lat = mesh_mod.build_lattice(_aircraft(0.0), _settings(), n_chord=10)
    start, _ = leg_segments(lat)
    assert start.shape[0] == 4 * lat.n_panels
    # The matching on the four segment sets (the rules themselves use the
    # whole legs, see test_leg_rules_whole_legs).
    _, scale, kind = _leg_force_rules_geometric(lat)
    assert not np.any(kind == LEG_KINK)
    assert np.all(scale == 1.0)


def test_leg_rules_from_undeflected_lattice():
    """The leg-force rules of a deflected flap are those of the undeflected lattice, so CL is smooth."""
    ac = _aircraft(0.0)
    ac.surfaces[0].dihedral = np.radians(5.0)
    st = _settings(10, 8)
    rules0 = leg_force_rules(mesh_mod.build_lattice(ac, st, n_chord=10))
    for d in (1e-4, 0.2):
        ac.set_deflection("flap", d)
        rules = leg_force_rules(mesh_mod.build_lattice(ac, st, n_chord=10))
        np.testing.assert_array_equal(rules[2], rules0[2])
        np.testing.assert_array_equal(rules[1], rules0[1])
    cl = {}
    for d in (-1e-3, -1e-5, 1e-5, 1e-3):
        ac.set_deflection("flap", d)
        lattice_cache.clear()
        cl[d] = _solve(ac, alpha=0.07, settings=st).totals.CL
    slope_small = (cl[1e-5] - cl[-1e-5]) / 2e-5
    slope_large = (cl[1e-3] - cl[-1e-3]) / 2e-3
    assert abs(slope_large - slope_small) <= 1e-4 * abs(slope_small)


@pytest.mark.slow
def test_part_span_flap_spanwise_refinement(monkeypatch):
    """A part-span flap converges with the spanwise mesh and stays near the normal-turn model."""
    ac = _aircraft(np.radians(10.0))
    ac.surfaces[0].controls[0].eta_start = 0.2
    ac.surfaces[0].controls[0].eta_end = 0.6
    out = {}
    for n_span in (24, 160):
        st = vt.SolverSettings(n_chord=20, n_panels=n_span)
        out[n_span] = (_solve(ac, settings=st).totals.CL, _old_solve(monkeypatch, ac, settings=st).totals.CL)
    print(f"part-span flap CL (geometric, normal turn): {out}")
    assert abs(out[160][0] - out[24][0]) <= 0.03 * abs(out[160][0])
    assert abs(out[160][0] - out[160][1]) <= 0.03 * abs(out[160][1])


def test_leg_rules_whole_legs():
    """Both segments of a leg with a hinge node get the rule of the whole leg."""
    lat = mesh_mod.build_lattice(_aircraft(0.0, partial=True), _settings(), n_chord=10)
    plain = mesh_mod.build_lattice(_aircraft(0.0, partial=True), _settings(), n_chord=10)
    plain.a_kink = plain.b_kink = None
    plain.geom_cache.pop("leg_force_rules")
    _, scale0, kind0 = leg_force_rules(plain)
    _, scale, kind = leg_force_rules(lat)
    np.testing.assert_array_equal(kind, np.concatenate([kind0, kind0]))
    np.testing.assert_array_equal(scale, np.concatenate([scale0, scale0]))
