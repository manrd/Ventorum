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
    st = SolverSettings(n_panels=6, n_chord=6)

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
        assert disp.get("backend") == "numpy"
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
    """
    solver = make_solver("panel3d")
    cond = FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    ac = _simple_wing(shape="0012", semi_span=4.0, chord=1.0, sweep_le=0.0)

    res_clean = solver.solve(ac, cond, SolverSettings(n_panels=8, n_chord=8))
    res_poor = solver.solve(ac, cond, SolverSettings(n_panels=4, n_chord=40))

    assert res_clean.totals.trust is not None
    assert res_poor.totals.trust is not None
    stats_clean = res_clean.details["mesh_stats"]
    stats_poor = res_poor.details["mesh_stats"]
    assert stats_poor["max_aspect_ratio"] > stats_clean["max_aspect_ratio"]
    assert stats_poor["max_skew"] > stats_clean["max_skew"]
    pen_clean = res_clean.totals.trust.factors.get("mesh_adequacy", 0.0)
    pen_poor = res_poor.totals.trust.factors.get("mesh_adequacy", 0.0)
    assert pen_poor > pen_clean
    assert res_poor.totals.trust.score < res_clean.totals.trust.score


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
    )
    copy = st.clone()
    for name in ("panel_kutta", "panel_backend", "panel_wake_length", "panel_chord_spacing",
                 "panel_estimate_mesh_error"):
        assert getattr(copy, name) == getattr(st, name)
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

    for name, value in (("panel_kutta", "linear"), ("panel_backend", "cuda"), ("panel_chord_spacing", "sine")):
        bad = SolverSettings(n_panels=6, n_chord=6)
        setattr(bad, name, value)
        with pytest.raises(ValueError, match=name):
            make_solver("panel3d").solve(ac, cond, bad)


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


def test_gpu_auto_selection_follows_the_lattice_rule(monkeypatch):
    """Backend "auto" follows the device and precision selection of ventorum.gpu."""
    from ventorum import gpu

    solver = make_solver("panel3d")
    st = SolverSettings()
    monkeypatch.setattr(gpu, "get_precision", lambda: "float64")

    monkeypatch.setattr(gpu, "get_device", lambda: "cpu")
    assert solver._resolve_backend(st, 5000, 1, False)[:3] == ("numpy", "cpu", "float64")

    monkeypatch.setattr(gpu, "get_device", lambda: "auto")
    seen = {}

    def fake_use_gpu(work, family="vlm", n_cases=1, n_panels=None, return_reason=False):
        seen["work"] = work
        decision = work >= 1.0e6
        return (decision, "auto: test rule") if return_reason else decision

    monkeypatch.setattr(gpu, "use_gpu", fake_use_gpu)
    backend, device, precision, reason = solver._resolve_backend(st, 5000, 2, True)
    assert (backend, device, precision) == ("gpu", "gpu", "float64")
    assert seen["work"] == 2 * 5000 * 5000 * 2.0
    assert reason == "auto: test rule"
    assert solver._resolve_backend(st, 100, 1, False)[:3] == ("numpy", "cpu", "float64")

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
