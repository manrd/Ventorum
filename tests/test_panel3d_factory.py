# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for the 3D panel method solver family integration.

Published sources
-----------------
* Morino, L. and Kuo, C. C., 1974, Subsonic potential aerodynamics for
  complex configurations: A general theory, AIAA Journal, Vol. 12, No. 2,
  pp. 191-197.
* Katz, J. and Plotkin, A., 2001, Low-Speed Aerodynamics, 2nd ed.,
  Cambridge University Press, section 9.4 and chapter 12.
"""

from __future__ import annotations

import json
import numpy as np
import pytest

from ventorum.airfoil import airfoil
from ventorum.core.datatypes import (
    Aircraft,
    FlightCondition,
    LiftingSurface,
    SolverResult,
    SolverSettings,
    WingSection,
    aircraft_from_json,
    aircraft_to_json,
)
from ventorum.solvers.factory import make_solver, resolve_solver_type
from ventorum.solvers.panel3d import PanelMethodSolver


def _naca0012_coords(n_points: int = 101) -> np.ndarray:
    """Return unit-chord Selig coordinates of a NACA 0012 airfoil."""
    return airfoil("0012", n_points=n_points).coords


def _simple_wing(
    shape: str | np.ndarray | None = "0012",
    semi_span: float = 4.0,
    chord: float = 1.0,
    sweep_le: float = 0.0,
) -> Aircraft:
    """Return a rectangular wing aircraft with the given section shape."""
    surf = LiftingSurface(
        name="Wing",
        semi_span=semi_span,
        sections=[
            WingSection(y_frac=0.0, chord=chord, shape=shape),
            WingSection(y_frac=1.0, chord=chord, shape=shape),
        ],
        sweep_le=sweep_le,
        is_symmetric=True,
    )
    ac = Aircraft(name="SimpleWing", surfaces=[surf])
    ac.compute_reference_values()
    return ac


def test_factory_resolves_and_creates_panel3d():
    """Factory resolves 'panel3d' and builds PanelMethodSolver."""
    canonical = resolve_solver_type("panel3d")
    assert canonical == "panel3d"
    solver = make_solver("panel3d")
    assert isinstance(solver, PanelMethodSolver)
    assert solver.name == "panel3d"


def test_wing_with_shapes_solve():
    """Factory builds and solves a wing with airfoil section shapes."""
    solver = make_solver("panel3d")
    ac = _simple_wing(shape="2412", semi_span=3.0, chord=1.0)
    cond = FlightCondition(V_inf=30.0, alpha=np.radians(3.0))
    st = SolverSettings(n_panels=8, n_chord=8)

    res = solver.solve(ac, cond, st)
    assert isinstance(res, SolverResult)
    assert res.converged
    assert res.solver_type == "panel3d"
    assert res.totals.CL > 0.1
    assert res.totals.CDi >= 0.0
    assert len(res.spanwise) == 1

    sw = res.spanwise[0]
    assert sw.surface_name == "Wing"
    assert len(sw.Cl) > 0
    assert len(sw.Cd_i) > 0
    assert len(sw.Cm_section) > 0

    assert "cp" in res.details
    assert isinstance(res.details["cp"], np.ndarray)
    assert len(res.details["cp"]) > 0


def test_default_shape_warning_when_none():
    """Section with shape=None defaults to NACA 0012 with a warning note."""
    solver = make_solver("panel3d")
    ac = _simple_wing(shape=None, semi_span=2.0, chord=1.0)
    cond = FlightCondition(V_inf=25.0, alpha=np.radians(2.0))
    st = SolverSettings(n_panels=6, n_chord=6)

    with pytest.warns(RuntimeWarning, match="default NACA 0012"):
        res = solver.solve(ac, cond, st)

    assert any("default NACA 0012" in n for n in res.details.get("notes", []))
    assert res.converged


def test_wing_plus_tail_solve():
    """Factory solves an aircraft configuration with wing and tail surfaces."""
    solver = make_solver("panel3d")
    wing = LiftingSurface(
        name="Wing",
        semi_span=3.0,
        sections=[
            WingSection(y_frac=0.0, chord=1.0, shape="0012"),
            WingSection(y_frac=1.0, chord=0.8, shape="0012"),
        ],
        is_symmetric=True,
    )
    tail = LiftingSurface(
        name="Tail",
        semi_span=1.2,
        sections=[
            WingSection(y_frac=0.0, chord=0.5, shape="0012"),
            WingSection(y_frac=1.0, chord=0.4, shape="0012"),
        ],
        position=np.array([2.5, 0.0, 0.2]),
        is_symmetric=True,
    )
    ac = Aircraft(name="WingTail", surfaces=[wing, tail])
    ac.compute_reference_values()

    cond = FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    st = SolverSettings(n_panels=6, n_chord=6)

    res = solver.solve(ac, cond, st)
    assert res.converged
    assert len(res.spanwise) == 2
    assert res.spanwise[0].surface_name == "Wing"
    assert res.spanwise[1].surface_name == "Tail"


def test_spanwise_strip_lift_sums_to_total_cl():
    """Spanwise strip lift sum matches total CL within 1e-3."""
    solver = make_solver("panel3d")
    ac = _simple_wing(shape="0012", semi_span=4.0, chord=1.0)
    cond = FlightCondition(V_inf=35.0, alpha=np.radians(4.0))
    st = SolverSettings(n_panels=8, n_chord=8)

    res = solver.solve(ac, cond, st)
    q_inf = 0.5 * cond.rho * cond.V_inf**2
    s_ref = ac.S_ref

    strip_lift_sum = 0.0
    for sw in res.spanwise:
        y_pts = sw.y
        n_s = len(y_pts)
        if n_s > 1:
            dy = np.diff(y_pts)
            widths = np.empty(n_s)
            widths[0] = dy[0]
            widths[-1] = dy[-1]
            widths[1:-1] = 0.5 * (dy[:-1] + dy[1:])
        else:
            widths = np.array([ac.b_ref])
        strip_lift_sum += float(np.sum(sw.local_lift * np.abs(widths)))

    cl_from_strips = strip_lift_sum / (q_inf * s_ref)
    assert abs(cl_from_strips - res.totals.CL) < 1.0e-3


def test_ground_effect_case():
    """Case in ground effect solves with image method and increases lift."""
    solver = make_solver("panel3d")
    ac = _simple_wing(shape="0012", semi_span=3.0, chord=1.0)
    cond_free = FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    cond_ge = FlightCondition(V_inf=30.0, alpha=np.radians(4.0), h=1.0)
    st = SolverSettings(n_panels=6, n_chord=6)

    res_free = solver.solve(ac, cond_free, st)
    res_ge = solver.solve(ac, cond_ge, st)

    assert res_ge.converged
    assert res_ge.totals.CL > res_free.totals.CL
    assert res_ge.details["h_min"] is not None


def test_five_angle_sweep_one_factorization(monkeypatch):
    """Five-angle sweep: the dispatch record reports the counted factorizations.

    The symmetric free-air sweep runs on a half mesh with only the
    symmetric system: one factorization. A sweep in ground effect needs one
    factorization per attitude. A sideslip sweep runs on the full mesh: one
    factorization. The counts of the dispatch record equal the counts of a
    wrapper around the LU routine.
    """
    import scipy.linalg

    calls = {"n": 0}
    original = scipy.linalg.lu_factor

    def counting(a, *args, **kwargs):
        calls["n"] += 1
        return original(a, *args, **kwargs)

    monkeypatch.setattr(scipy.linalg, "lu_factor", counting)
    solver = make_solver("panel3d")
    ac = _simple_wing(shape="0012", semi_span=3.0, chord=1.0)
    cond = FlightCondition(V_inf=25.0, alpha=0.0)
    st = SolverSettings(n_panels=6, n_chord=6, panel_ground_solver="direct")

    angles = np.radians([-2.0, 0.0, 2.0, 4.0, 6.0])
    results = solver.solve_sweep(ac, cond, st, angles)

    assert len(results) == 5
    cls = [r.totals.CL for r in results]
    # Lift increases monotonically with alpha
    for i in range(4):
        assert cls[i + 1] > cls[i]

    assert calls["n"] == 1
    for r in results:
        disp = r.details.get("dispatch", {})
        assert disp.get("factorization_count") == 1
        # "auto" runs the compiled Numba kernels on the CPU.
        assert disp.get("backend") == "numba"
        assert disp.get("device") == "cpu"
        assert disp.get("dtype") == "float64"
        assert r.symmetry_used

    calls["n"] = 0
    ground_conds = [FlightCondition(V_inf=25.0, alpha=float(a), h=0.8) for a in np.radians([1.0, 3.0, 5.0])]
    ground_results = solver.solve(ac, ground_conds, st)
    assert calls["n"] == 3
    assert all(r.details["dispatch"]["factorization_count"] == 3 for r in ground_results)

    calls["n"] = 0
    side_conds = [FlightCondition(V_inf=25.0, alpha=float(a), beta=np.radians(4.0)) for a in angles]
    side_results = solver.solve(ac, side_conds, st)
    assert calls["n"] == 1
    assert all(r.details["dispatch"]["factorization_count"] == 1 for r in side_results)
    assert not side_results[0].symmetry_used


def test_json_round_trip_with_and_without_shape():
    """JSON serialization round trip preserves shape and supports backward compatibility."""
    # 1. Round trip with shape string
    ac_with_shape = _simple_wing(shape="2412", semi_span=3.0, chord=1.0)
    json_str = aircraft_to_json(ac_with_shape)
    loaded = aircraft_from_json(json_str)
    assert loaded.surfaces[0].sections[0].shape == "2412"

    # 2. Round trip with shape coordinate array
    coords = _naca0012_coords(21)
    ac_with_coords = _simple_wing(shape=coords, semi_span=3.0, chord=1.0)
    json_coords = aircraft_to_json(ac_with_coords)
    loaded_coords = aircraft_from_json(json_coords)
    assert np.allclose(loaded_coords.surfaces[0].sections[0].shape, coords)

    # 3. Round trip without shape (shape=None)
    ac_no_shape = _simple_wing(shape=None, semi_span=3.0, chord=1.0)
    json_none = aircraft_to_json(ac_no_shape)
    loaded_none = aircraft_from_json(json_none)
    assert loaded_none.surfaces[0].sections[0].shape is None

    # 4. Backward compatibility: JSON string without 'shape' key
    raw_dict = json.loads(json_none)
    for s in raw_dict["surfaces"]:
        for sec in s["sections"]:
            sec.pop("shape", None)
    old_json = json.dumps(raw_dict)
    loaded_old = aircraft_from_json(old_json)
    assert loaded_old.surfaces[0].sections[0].shape is None


def test_trust_score_lower_for_poor_mesh():
    """Trust score is present and lower for a deliberately poor mesh of the same wing.

    The same wing is solved with a regular mesh and with a mesh of very
    elongated and skewed panels (four spanwise panels and many chordwise
    panels). The mesh adequacy penalty must grow and the score must fall.
    The regular mesh has 24 chordwise panels: the mesh builder now closes
    the open trailing edge of the library section, and with 8 chordwise
    panels the near-field drag rule rates the closed wing at the strong
    level (the mesh-error study puts n_chord 8 at that level). The former
    base panels of the open edge hid this with their base suction.
    """
    solver = make_solver("panel3d")
    cond = FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    ac = _simple_wing(shape="0012", semi_span=4.0, chord=1.0, sweep_le=0.0)

    res_clean = solver.solve(ac, cond, SolverSettings(n_panels=8, n_chord=24))
    res_poor = solver.solve(ac, cond, SolverSettings(n_panels=4, n_chord=40))

    assert res_clean.totals.trust is not None
    assert res_poor.totals.trust is not None
    stats_clean = res_clean.details["mesh_stats"]
    stats_poor = res_poor.details["mesh_stats"]
    assert stats_poor["max_aspect_ratio"] > stats_clean["max_aspect_ratio"]
    assert stats_poor["max_skew"] > stats_clean["max_skew"]
    # The elongated-panel rule is a spanwise resolution rule: the calibrated error model replaced its
    # penalty (it keeps its warning). The error model gives the two meshes about the same predicted
    # error: 8 x 24 panels E = exp(-0.46)/24 + exp(-2.7) 8^-0.66 = 4.33 %, 4 x 40 panels
    # E = exp(-0.46)/40 + exp(-2.7) 4^-0.66 = 4.27 % (k*E 6.5 % and 6.4 %, both LOW). The test
    # therefore checks the warning of the poor mesh and the error-model score of each mesh.
    from ventorum.core import trust as tr

    assert any("Elongated panels" in w for w in res_poor.totals.trust.warnings)
    assert not any("Elongated panels" in w for w in res_clean.totals.trust.warnings)
    for res, n_span, n_chord in ((res_clean, 8, 24), (res_poor, 4, 40)):
        e, k, _ = tr.predicted_error("panel3d", 8.0, n_chord=n_chord, n_span=n_span)
        assert res.totals.trust.factors["error_model"] == pytest.approx(1.0 - tr.error_score(k * e))
        assert res.totals.trust.rating == "LOW"


def test_panel_settings_reach_the_solver():
    """The panel fields of SolverSettings reach the solver through the factory and analyze.

    The fields are real dataclass fields: they can be set, clone() keeps
    them (also for a subclass), and unknown values are refused.
    """
    import dataclasses

    import ventorum as vt

    st = SolverSettings(
        n_panels=6,
        n_chord=6,
        solver_type="panel3d",
        panel_kutta="pressure",
        panel_backend="numba",
        panel_wake_length=12.5,
        panel_chord_spacing="uniform",
        panel_scale_root_thickness=False,
    )
    copy = st.clone()
    for name in ("panel_kutta", "panel_backend", "panel_wake_length", "panel_chord_spacing",
                 "panel_estimate_mesh_error", "panel_scale_root_thickness"):
        assert getattr(copy, name) == getattr(st, name)
    assert SolverSettings().panel_scale_root_thickness is True
    st.panel_estimate_mesh_error = False

    @dataclasses.dataclass
    class _MySettings(SolverSettings):
        extra: int = 3

    sub = _MySettings(n_panels=6, panel_kutta="pressure", extra=7).clone()
    assert isinstance(sub, _MySettings) and sub.extra == 7 and sub.panel_kutta == "pressure"

    ac = _simple_wing(shape="0012", semi_span=3.0, chord=1.0)
    cond = FlightCondition(V_inf=25.0, alpha=np.radians(3.0))
    for res in (make_solver("panel3d").solve(ac, cond, st), vt.analyze(ac, condition=cond, settings=st)):
        assert res.details["panel_result"].kutta == "pressure"
        assert res.iterations >= 1
        assert res.details["dispatch"]["backend"] == "numba"
        assert res.details["panel"]["wake_length"] == 12.5
        assert res.details["panel"]["chord_spacing"] == "uniform"
        assert res.details["panel"]["scale_root_thickness"] is False

    for name, value in (("panel_kutta", "linear"), ("panel_backend", "cuda"), ("panel_chord_spacing", "sine")):
        bad = SolverSettings(n_panels=6, n_chord=6)
        setattr(bad, name, value)
        with pytest.raises(ValueError, match=name):
            make_solver("panel3d").solve(ac, cond, bad)


def test_root_thickness_setting_reaches_the_mesh():
    """panel_scale_root_thickness (default True) scales the root section of a dihedral wing.

    The root section of a symmetric wing with 30 deg dihedral is thicker
    along z by 1 / cos(30 deg) with the default; False keeps the former
    mesh. Only the root nodes move.
    """
    ac = _simple_wing(shape="0012", semi_span=3.0, chord=1.0)
    ac.surfaces[0].dihedral = np.radians(30.0)
    cond = FlightCondition(V_inf=25.0, alpha=np.radians(3.0))
    solver = make_solver("panel3d")
    on = solver.solve(ac, cond, SolverSettings(n_panels=6, n_chord=8))
    off = solver.solve(ac, cond, SolverSettings(n_panels=6, n_chord=8, panel_scale_root_thickness=False))
    assert on.details["panel"]["scale_root_thickness"] is True
    assert off.details["panel"]["scale_root_thickness"] is False
    n_on = np.asarray(on.details["mesh"].nodes)
    n_off = np.asarray(off.details["mesh"].nodes)
    root = np.abs(n_off[:, 1]) < 1e-12
    assert np.array_equal(n_on[~root], n_off[~root])
    assert float(np.ptp(n_on[root, 2])) == pytest.approx(float(np.ptp(n_off[root, 2])) / np.cos(np.radians(30.0)),
                                                         rel=1e-12)


def test_default_chord_spacing_is_cosine():
    """The panel solver clusters the chordwise panels with cosine spacing by default.

    The vortex lattice keeps its own default (``chord_spacing``,
    uniform); the panel solver reads ``panel_chord_spacing``.
    """
    ac = _simple_wing(shape="0012", semi_span=3.0, chord=1.0)
    cond = FlightCondition(V_inf=25.0, alpha=np.radians(3.0))
    st = SolverSettings(n_panels=6, n_chord=8)
    assert st.chord_spacing == "uniform"
    res = make_solver("panel3d").solve(ac, cond, st)
    assert res.details["panel"]["chord_spacing"] == "cosine"
    nodes = np.asarray(res.details["mesh"].nodes)
    root = nodes[np.abs(nodes[:, 1]) < 1e-12]
    x_root = np.unique(np.round(root[:, 0], 12))
    x_cos = 0.5 * (1.0 - np.cos(np.pi * np.arange(0, 9) / 8.0))
    assert np.allclose(x_root, x_cos, atol=1e-12)


def test_ground_keyword_is_honoured():
    """An explicit ground plane gives the same result as condition.h, as for the VLM."""
    from ventorum.aero.system import GroundPlane, ground_normal

    solver = make_solver("panel3d")
    ac = _simple_wing(shape="0012", semi_span=3.0, chord=1.0)
    st = SolverSettings(n_panels=6, n_chord=6)
    alpha = np.radians(4.0)
    res_h = solver.solve(ac, FlightCondition(V_inf=30.0, alpha=alpha, h=0.6), st)
    k_g = ground_normal(alpha, 0.0, 0.0)
    plane = GroundPlane(normal=k_g, offset=float(np.dot(ac.moment_reference(), k_g) - 0.6))
    res_g = solver.solve(ac, FlightCondition(V_inf=30.0, alpha=alpha), st, ground=plane)
    res_free = solver.solve(ac, FlightCondition(V_inf=30.0, alpha=alpha), st)
    assert res_g.totals.CL == pytest.approx(res_h.totals.CL, rel=1e-12)
    assert res_g.totals.CDi == pytest.approx(res_h.totals.CDi, rel=1e-12)
    assert res_g.details["h_min"] == pytest.approx(res_h.details["h_min"], rel=1e-12)
    assert res_g.totals.CL > res_free.totals.CL


def test_gpu_auto_selection_follows_the_panel_rule(monkeypatch):
    """Backend "auto" follows the device and precision selection of ventorum.gpu."""
    from ventorum import gpu

    solver = make_solver("panel3d")
    st = SolverSettings()
    monkeypatch.setattr(gpu, "get_precision", lambda: "float64")

    monkeypatch.setattr(gpu, "get_device", lambda: "cpu")
    assert solver._resolve_backend(st, 5000, 1, False)[:3] == ("numba", "cpu", "float64")

    monkeypatch.setattr(gpu, "get_device", lambda: "auto")
    seen = {}

    def fake_use_gpu(work, family="vlm", n_cases=1, n_panels=None, return_reason=False):
        seen["work"] = work
        seen["family"] = family
        seen["panels"] = n_panels
        decision = work >= 1.0e6
        return (decision, "auto: test rule") if return_reason else decision

    monkeypatch.setattr(gpu, "use_gpu", fake_use_gpu)
    backend, device, precision, reason = solver._resolve_backend(st, 5000, 2, True)
    assert (backend, device, precision) == ("gpu", "gpu", "float64")
    assert seen["work"] == 2 * 5000 * 5000
    assert seen["family"] == "panel3d"
    assert seen["panels"] == 5000
    assert reason == "auto: test rule"
    assert solver._resolve_backend(st, 100, 1, False)[:3] == ("numba", "cpu", "float64")

    monkeypatch.setattr(gpu, "get_precision", lambda: "float32")
    explicit = SolverSettings(panel_backend="gpu")
    assert solver._resolve_backend(explicit, 10, 1, False)[:3] == ("gpu", "gpu", "float32")


def test_strip_map_covers_every_panel_of_a_control_mesh():
    """Every panel of a mesh with a deflected control maps to exactly one lattice strip.

    The control surface adds hinge nodes and side closures, so the panel
    order of a plain wing does not hold. The map assigns each panel by the
    position of its centroid between the station planes of the strips.
    """
    from ventorum.core.datatypes import ControlSurface
    from ventorum.geometry.lattice import _surface_eta, resolve_spacing
    from ventorum.panel3d.mesh import mesh_lifting_surface
    from ventorum.solvers.panel3d import _surface_strip_maps

    surf = LiftingSurface(
        name="Wing",
        semi_span=4.0,
        sections=[WingSection(y_frac=0.0, chord=1.0), WingSection(y_frac=1.0, chord=1.0)],
        is_symmetric=True,
        controls=[ControlSurface(name="flap", eta_start=0.3, eta_end=0.7, hinge_x_c=0.7,
                                 deflection=np.radians(5.0), symmetric=True)],
    )
    n_span = 6
    eta_e, _ = _surface_eta(surf, n_span, resolve_spacing(surf.spacing or "auto", surf, "vlm"))
    for half in (False, True):
        mesh = mesh_lifting_surface(surf, _naca0012_coords(81), n_chord=6, n_span=n_span, half=half)
        right, left = _surface_strip_maps(mesh, surf, n_span, np.asarray(eta_e), 0, half)
        mapped = [p for strip in right for p in strip]
        if left is not None:
            mapped += [p for strip in left for p in strip]
        assert sorted(mapped) == list(range(len(mesh.panels)))
        centroids = np.mean(np.asarray(mesh.nodes)[np.asarray(mesh.panels)], axis=1)
        y_e = np.asarray(eta_e) * surf.semi_span
        for j, strip in enumerate(right):
            y = centroids[strip, 1]
            assert np.all(y >= y_e[j] - 1e-9) and np.all(y <= y_e[j + 1] + 1e-9)
        if left is not None:
            for k, strip in enumerate(left):
                j = n_span - 1 - k
                y = -centroids[strip, 1]
                assert np.all(y >= y_e[j] - 1e-9) and np.all(y <= y_e[j + 1] + 1e-9)


def _closed_te_wing() -> Aircraft:
    """Rectangular wing (aspect ratio 8) with a NACA 0012 section with a closed trailing edge."""
    coords = airfoil("0012", closed_te=True).coords
    return _simple_wing(shape=coords, semi_span=4.0, chord=1.0)


@pytest.mark.slow
def test_well_resolved_wing_mesh_has_no_mesh_warning():
    """A mesh that the mesh-error study shows as adequate gets no mesh warning.

    With 16 spanwise and 40 chordwise panels the mesh error of this wing is
    below the warning level (2 % in CL, 4 % in CDi). The panel aspect ratio,
    skew and near/far-field drag rules must not warn, although the largest
    panel aspect ratio and skew of the mesh are large (leading-edge panels
    and tip caps).
    """
    from ventorum.core import trust as tr

    solver = make_solver("panel3d")
    cond = FlightCondition(V_inf=30.0, alpha=np.radians(5.0))
    res = solver.solve(_closed_te_wing(), cond, SolverSettings(n_panels=16, n_chord=40))
    trust = res.totals.trust
    stats = res.details["mesh_stats"]
    assert stats["max_aspect_ratio"] > tr.PANEL3D_MAX_ASPECT_RATIO
    assert stats["max_skew"] > tr.PANEL3D_MAX_SKEW
    assert trust.factors["mesh_adequacy"] == 0.0
    assert trust.factors["mesh_resolution"] == 0.0
    mesh_words = ("Elongated", "Skewed", "Warped", "Folded", "Near-field", "chordwise mesh")
    assert not any(word in w for w in trust.warnings for word in mesh_words), trust.warnings
    # The calibrated error model rates this mesh MODERATE: E = exp(-0.46)/40 + exp(-2.7) 16^-0.66
    # = 2.65 %, k*E = 4.0 % (was HIGH under the former mesh rules, which have no warning here).
    assert trust.rating == "MODERATE"


@pytest.mark.parametrize("n_chord, level", [(20, "warn"), (8, "strong")])
def test_coarse_chordwise_mesh_gets_the_mesh_warning(n_chord, level):
    """Meshes that the mesh-error study shows as inadequate get the warning of their level.

    With 16 spanwise panels, 20 chordwise panels (the former default count) give an
    error of this wing above the warning level and 8 chordwise panels above
    the strong level. The near-field/far-field drag rule gives the matching
    level.
    """
    from ventorum.core import trust as tr

    solver = make_solver("panel3d")
    cond = FlightCondition(V_inf=30.0, alpha=np.radians(5.0))
    res = solver.solve(_closed_te_wing(), cond, SolverSettings(n_panels=16, n_chord=n_chord))
    trust = res.totals.trust
    expected = tr.PANEL3D_PENALTY_RESOLUTION_WARN if level == "warn" else tr.PANEL3D_PENALTY_RESOLUTION_STRONG
    assert trust.factors["mesh_resolution"] == expected
    assert any("Near-field pressure drag" in w for w in trust.warnings)
    # The drag rule stays a cap (MODERATE at the warning level). The calibrated error model rates both
    # meshes LOW: n_chord 20 gives E = exp(-0.46)/20 + exp(-2.7) 16^-0.66 = 4.23 %, k*E = 6.3 %; n_chord 8
    # gives k*E = 12.4 % (the warning level was MODERATE under the former rules).
    assert trust.rating == "LOW"


@pytest.mark.slow
def test_chordwise_count_reaches_the_trust_score():
    """Regression: the panel solver did not give n_chord to the trust score.

    The chordwise rule of ``panel3d_mesh_checks`` acted only when the
    caller gave the chordwise count, and the panel solver did not give it,
    so the best indicator of the mesh error of the mesh-error study was
    off. Now a coarse chordwise mesh gets the chordwise warning with its
    count (also the default count when ``n_chord`` is None), and a fine
    chordwise mesh does not.
    """
    from ventorum.core import trust as tr

    solver = make_solver("panel3d")
    cond = FlightCondition(V_inf=30.0, alpha=np.radians(5.0))
    coarse_count = tr.PANEL3D_N_CHORD_WARN - 8
    fine_count = tr.PANEL3D_N_CHORD_WARN + 4
    coarse = solver.solve(_closed_te_wing(), cond, SolverSettings(n_panels=16, n_chord=coarse_count)).totals.trust
    fine = solver.solve(_closed_te_wing(), cond, SolverSettings(n_panels=16, n_chord=fine_count)).totals.trust
    # The calibrated error model replaced the chordwise rule: the chordwise count reaches the trust score
    # through the chordwise term exp(-0.46)/n_chord of the predicted error.
    for t, count in ((coarse, coarse_count), (fine, fine_count)):
        e, k, _ = tr.predicted_error("panel3d", 8.0, n_chord=count, n_span=16)
        assert t.factors["error_model"] == pytest.approx(1.0 - tr.error_score(k * e))
        assert not any("Coarse chordwise mesh" in w for w in t.warnings), t.warnings
    assert coarse.factors["error_model"] > fine.factors["error_model"]
    assert any("comes from the chordwise panel count" in w for w in coarse.warnings)
    # The default chordwise count (n_chord None) also reaches the rule.
    from ventorum.solvers.panel3d import _DEFAULT_N_CHORD

    default = solver.solve(_closed_te_wing(), cond, SolverSettings(n_panels=8)).totals.trust
    expected = _DEFAULT_N_CHORD < tr.PANEL3D_N_CHORD_WARN
    assert any(f"Coarse chordwise mesh: {_DEFAULT_N_CHORD} chordwise" in w for w in default.warnings) is expected


def test_ground_gap_rule_reaches_the_panel_solver():
    """In ground effect the rule "chordwise panels longer than the gap" acts for the panel method.

    The rule of the ground factor compares the chordwise panel length
    (largest strip chord over n_chord) with the smallest ground gap. With
    the trailing edge about 0.12 chord above the ground, 6 chordwise panels
    are longer than the gap and 12 are not.
    """
    solver = make_solver("panel3d")
    cond = FlightCondition(V_inf=30.0, alpha=np.radians(5.0), h=0.22)
    messages = {}
    for n_chord in (6, 12):
        res = solver.solve(_closed_te_wing(), cond, SolverSettings(n_panels=8, n_chord=n_chord))
        h_over_c = res.details["h_min"] / res.c_ref
        assert 1.0 / 12.0 < h_over_c < 1.0 / 6.0
        messages[n_chord] = [w for w in res.totals.trust.warnings if "longer than the ground gap" in w]
    assert len(messages[6]) == 1 and messages[6][0].startswith("Chordwise panels (6 on the largest chord")
    assert messages[12] == []


def test_auto_backend_uses_numba_on_cpu(monkeypatch):
    """With the device setting "cpu", "auto" runs the Numba kernels and says so."""
    from ventorum import gpu

    monkeypatch.setattr(gpu, "get_device", lambda: "cpu")
    backend, device, precision, reason = make_solver("panel3d")._resolve_backend(SolverSettings(), 500)
    assert (backend, device, precision) == ("numba", "cpu", "float64")
    assert reason.endswith("CPU kernels: numba")


def test_auto_backend_falls_back_to_numpy_without_numba(monkeypatch):
    """Without Numba, "auto" on the CPU uses the numpy reference kernels."""
    import builtins

    from ventorum import gpu

    real_import = builtins.__import__

    def no_numba(name, *args, **kwargs):
        if name == "numba" or name.startswith("numba."):
            raise ImportError("no numba in this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(gpu, "get_device", lambda: "cpu")
    monkeypatch.setattr(builtins, "__import__", no_numba)
    backend, _, _, reason = make_solver("panel3d")._resolve_backend(SolverSettings(), 500)
    assert backend == "numpy"
    assert "Numba is not available" in reason


def test_auto_backend_in_worker_thread_uses_numba(monkeypatch):
    """Inside a CPU worker thread "auto" also runs the Numba kernels."""
    from ventorum import gpu
    from ventorum.utils.parallel import _local

    monkeypatch.setattr(gpu, "get_device", lambda: "auto")
    monkeypatch.setattr(_local, "worker_threads", 2, raising=False)
    backend, device, _, reason = make_solver("panel3d")._resolve_backend(SolverSettings(), 500)
    assert (backend, device) == ("numba", "cpu")
    assert reason.startswith("called inside a CPU worker thread")


def test_auto_numba_matches_numpy():
    """The default CPU path (Numba) agrees with the numpy reference to round-off."""
    surf = LiftingSurface(
        name="W", semi_span=3.0,
        sections=[WingSection(y_frac=0.0, chord=1.0, shape="0012"), WingSection(y_frac=1.0, chord=1.0, shape="0012")],
    )
    cond = FlightCondition(alpha=float(np.radians(4.0)), V_inf=30.0)
    solver = make_solver("panel3d")
    res = {b: solver.solve(surf, cond, SolverSettings(n_panels=6, n_chord=8, panel_backend=b)) for b in ("numpy", "numba")}
    for name in ("CL", "CDi", "Cm"):
        a, b = getattr(res["numba"].totals, name), getattr(res["numpy"].totals, name)
        assert abs(a - b) <= 1e-12 * max(abs(b), 1e-3)
    cp_a, cp_b = res["numba"].details["cp"], res["numpy"].details["cp"]
    assert np.max(np.abs(cp_a - cp_b)) <= 1e-12 * max(float(np.max(np.abs(cp_b))), 1.0)


@pytest.mark.parametrize("in_ground", [False, True])
def test_auto_backend_dispatches_panel_family(monkeypatch, in_ground):
    """Each configuration uses one panel assembly and factorization estimate."""
    from ventorum import gpu

    seen = []

    def dispatch(work, family, n_cases, n_panels, return_reason=False):
        seen.append((work, family, n_cases, n_panels, return_reason))
        return False, "panel profile test"

    monkeypatch.setattr(gpu, "get_device", lambda: "auto")
    monkeypatch.setattr(gpu, "use_gpu", dispatch)
    backend, device, _, reason = make_solver("panel3d")._resolve_backend(
        SolverSettings(), 500, n_configs=3, in_ground=in_ground,
    )
    assert seen == [(750000.0, "panel3d", 3, 500, True)]
    assert (backend, device) == ("numba", "cpu")
    assert reason.startswith("panel profile test")


@pytest.mark.parametrize("gpu_seconds, expected", [(0.5, "gpu"), (2.0, "cpu")])
def test_auto_backend_follows_panel_profile(monkeypatch, gpu_seconds, expected):
    """The panel profile can select either device and appears in the reason."""
    from ventorum import gpu
    from ventorum.hardware import profile as prof

    model = {"n": [100, 500], "cpu": [1.0, 1.0], "gpu": [gpu_seconds, gpu_seconds],
             "winner": [expected, expected], "crossover_panels": 100 if expected == "gpu" else None}
    monkeypatch.setattr(prof, "_cached_loaded", True)
    monkeypatch.setattr(prof, "_cached", {"settings": {"gpu": {"panel3d": {"float32": model}}}})
    monkeypatch.setattr(gpu, "_device", "auto")
    monkeypatch.setattr(gpu, "_precision", "float32")
    monkeypatch.setattr(gpu, "available", lambda: True)
    backend, device, precision, reason = make_solver("panel3d")._resolve_backend(SolverSettings(), 500)
    assert device == expected
    assert backend == ("gpu" if expected == "gpu" else "numba")
    assert precision == ("float32" if expected == "gpu" else "float64")
    assert "profile gpu.panel3d.float32" in reason
