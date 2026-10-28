"""Self-intersection check of the 3D panel mesh.

A symmetric wing with steep dihedral and ``scale_root_thickness=False``
can put the upper nodes of stations near the root past the plane of
symmetry, so that the two halves pass through each other. The fold test
compares only neighbour station rows and accepted such meshes
(``n_folded`` = 0); the lift then did not converge with the spanwise panel
count. ``check_mesh`` now counts the self-intersections, and the solver
refuses such meshes.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import ventorum as vt
from ventorum.core.trust import PANEL3D_PENALTY_NONCONVEX, panel3d_mesh_checks
from ventorum.panel3d.mesh import PanelMesh, _self_intersections, check_mesh, mesh_lifting_surface


def _naca(t=0.12, m=0.0, p=0.4, n=201):
    """NACA 4-digit section [-], upper trailing edge to lower trailing edge (cosine spacing)."""
    beta = np.linspace(0.0, np.pi, n)
    x = 0.5 * (1.0 - np.cos(beta))
    yt = 5.0 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)
    if m > 0.0:
        yc = np.where(x < p, m / p**2 * (2 * p * x - x**2), m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x**2))
    else:
        yc = np.zeros_like(x)
    upper = np.column_stack([x, yc + yt])[::-1]
    lower = np.column_stack([x, yc - yt])[1:]
    return np.vstack([upper, lower])


def _wing(dihedral_deg):
    """Rectangular wing: chord 1 m, semi-span 4 m, dihedral in deg."""
    return vt.LiftingSurface(
        name="W",
        semi_span=4.0,
        dihedral=np.radians(dihedral_deg),
        sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=1.0)],
    )


def _mesh(dihedral_deg, n_span, scale, half=False, foil=(0.12, 0.0)):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mesh_lifting_surface(_wing(dihedral_deg), _naca(foil[0], foil[1]), n_chord=16, n_span=n_span,
                                    half=half, scale_root_thickness=scale)


def _quads(dz):
    """Two quads: a unit square in z = 0 and a vertical quad through it, moved up by dz [m]."""
    nodes = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                      [0.5, 0.5, -0.5], [0.5, 0.5, 0.5], [0.5, 2, 0.5], [0.5, 2, -0.5]], dtype=float)
    nodes[4:, 2] += dz
    return nodes, np.array([[0, 1, 2, 3], [4, 5, 6, 7]])


def test_crossing_quads_count_and_separated_quads_do_not():
    """A vertical quad through a square counts; the same quad moved 2 m up does not."""
    comp = np.zeros(2, dtype=int)
    for prefilter in (True, False):
        nodes, panels = _quads(0.0)
        assert _self_intersections(nodes, panels, comp, False, 1e-9, prefilter=prefilter)[0] > 0
        nodes, panels = _quads(2.0)
        assert _self_intersections(nodes, panels, comp, False, 1e-9, prefilter=prefilter)[0] == 0


def test_check_mesh_counts_the_crossing_of_a_steep_unscaled_wing():
    """NACA 0012, 80 deg dihedral, no root scaling, 16 spanwise panels: the two halves cross.

    The fold test of neighbour rows does not see it (n_folded = 0); the
    self-intersection count does, in the full mesh and in the half mesh
    with its mirror image.
    """
    full = check_mesh(_mesh(80.0, 16, False))
    assert full["n_folded"] == 0
    assert full["n_self_intersections"] > 0
    assert full["self_intersecting_components"] == [0]
    assert not full["valid_panels"]
    half = check_mesh(_mesh(80.0, 16, False, half=True))
    assert half["n_self_intersections"] > 0
    assert not half["valid_panels"]


def test_prefilter_finds_the_same_pairs_as_all_pairs():
    """The bounding-sphere prefilter loses no crossing on a crossing wing mesh."""
    for half in (False, True):
        mesh = _mesh(80.0, 12, False, half=half)
        args = (np.asarray(mesh.nodes), np.asarray(mesh.panels), np.asarray(mesh.component), half, 1e-9)
        fast = _self_intersections(*args, prefilter=True)
        slow = _self_intersections(*args, prefilter=False)
        assert fast == slow
        assert fast[0] > 0


@pytest.mark.parametrize("dihedral", [0.0, 30.0, 60.0, 80.0, 85.0])
@pytest.mark.parametrize("half", [False, True])
def test_scaled_and_ordinary_wings_have_no_self_intersection(dihedral, half):
    """Root scaling at every dihedral, and an unscaled wing at 0 and 30 deg, give no crossing."""
    for n_span in (10, 16, 24):
        for foil in ((0.12, 0.0), (0.12, 0.02)):
            stats = check_mesh(_mesh(dihedral, n_span, True, half=half, foil=foil))
            assert stats["n_self_intersections"] == 0, (dihedral, n_span, foil)
            if dihedral <= 30.0:
                stats = check_mesh(_mesh(dihedral, n_span, False, half=half, foil=foil))
                assert stats["n_self_intersections"] == 0, (dihedral, n_span, foil)


def _settings(scale):
    return vt.SolverSettings(solver_type="panel3d", n_panels=16, n_chord=16, panel_cache=False,
                             panel_scale_root_thickness=scale)


def _aircraft():
    wing = vt.LiftingSurface(
        name="W",
        semi_span=4.0,
        dihedral=np.radians(80.0),
        sections=[vt.WingSection(y_frac=0.0, chord=1.0, shape=_naca()),
                  vt.WingSection(y_frac=1.0, chord=1.0, shape=_naca())],
    )
    return vt.Aircraft(name="Steep", surfaces=[wing])


def test_solver_refuses_a_self_intersecting_mesh():
    """Regression: the solver accepted the crossing mesh, and the lift jumped with the panel count."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError) as exc:
            vt.analyze(_aircraft(), alpha_deg=4.0, settings=_settings(False))
    msg = str(exc.value)
    assert "'W'" in msg
    assert "intersects itself or its mirror image" in msg
    assert "panel_scale_root_thickness=True" in msg
    assert "spanwise panel count" in msg and "dihedral" in msg


def test_solver_accepts_the_scaled_mesh():
    """The same wing with the root scaling (the default) solves to a finite lift."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = vt.analyze(_aircraft(), alpha_deg=4.0, settings=_settings(True))
    assert np.isfinite(res.totals.CL)
    assert res.totals.CL > 0.0


def test_trust_score_reports_self_intersections():
    """A count above zero gives a warning, a recommendation and the folded-panel penalty."""
    penalties, warns, recs = panel3d_mesh_checks({"n_self_intersections": 5})
    assert penalties["mesh_adequacy"] == pytest.approx(PANEL3D_PENALTY_NONCONVEX)
    assert any("Self-intersecting surface: 5 panel edges" in w for w in warns)
    assert any("panel_scale_root_thickness=True" in r for r in recs)
    penalties, warns, _ = panel3d_mesh_checks({"n_self_intersections": 0})
    assert penalties["mesh_adequacy"] == 0.0
    assert not any("Self-intersecting" in w for w in warns)


def test_two_components_are_not_tested_against_each_other():
    """The count is per closed surface: two crossing quads of different components do not count."""
    nodes, panels = _quads(0.0)
    mesh = PanelMesh(nodes=nodes, panels=panels, component=np.array([0, 1]), kind=np.array(["surface"] * 2),
                     wake_edges=np.zeros((0, 4), dtype=int))
    assert check_mesh(mesh)["n_self_intersections"] == 0


def _count_self_intersection_calls(monkeypatch):
    """Replace the counting function of the mesh module by a counter; return the call list."""
    from ventorum.panel3d import mesh as mesh_mod

    calls = []
    original = mesh_mod._self_intersections

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(mesh_mod, "_self_intersections", counted)
    return calls


def _shallow_aircraft():
    """Wing with 5 deg dihedral: no crossing with or without the root scaling."""
    wing = vt.LiftingSurface(
        name="W",
        semi_span=4.0,
        dihedral=np.radians(5.0),
        sections=[vt.WingSection(y_frac=0.0, chord=1.0, shape=_naca()),
                  vt.WingSection(y_frac=1.0, chord=1.0, shape=_naca())],
    )
    return vt.Aircraft(name="Shallow", surfaces=[wing])


def test_default_build_skips_the_self_intersection_count(monkeypatch):
    """With the root scaling (the default) the solver does not run the count: it costs time on every build."""
    calls = _count_self_intersection_calls(monkeypatch)
    res = vt.analyze(_shallow_aircraft(), alpha_deg=4.0, settings=_settings(True))
    assert calls == []
    assert "n_self_intersections" not in res.details["mesh_stats"]
    assert np.isfinite(res.totals.CL)


def test_unscaled_build_runs_the_self_intersection_count(monkeypatch):
    """Without the root scaling the solver runs the count on the mesh."""
    calls = _count_self_intersection_calls(monkeypatch)
    res = vt.analyze(_shallow_aircraft(), alpha_deg=4.0, settings=_settings(False))
    assert len(calls) >= 1
    assert res.details["mesh_stats"]["n_self_intersections"] == 0


def test_check_mesh_always_counts(monkeypatch):
    """A direct call of check_mesh is an explicit check: it counts also on a mesh with the root scaling."""
    calls = _count_self_intersection_calls(monkeypatch)
    stats = check_mesh(_mesh(5.0, 8, True))
    assert len(calls) == 1
    assert stats["n_self_intersections"] == 0
