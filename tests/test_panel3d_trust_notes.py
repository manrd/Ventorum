# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Tests for 3DP trust notes: far-field kernel bias, outer-strip Cl,
root thickness scaling above 75 deg dihedral, and a steep root dihedral
with and without the root thickness scaling.
"""

from __future__ import annotations

import math
import warnings

import numpy as np

import ventorum as vt
from ventorum.core.datatypes import FlightCondition, LiftingSurface, SolverSettings, WingSection
from ventorum.core.trust import (
    PANEL3D_FAR_FIELD_FREE_AIR_NOTE,
    PANEL3D_FAR_FIELD_GROUND_NOTE,
    PANEL3D_OUTER_STRIP_GROUND_NOTE,
    PANEL3D_ROOT_SCALE_WARNING,
    PANEL3D_STEEP_ROOT_CONVERGENCE_NOTE,
    PANEL3D_STEEP_ROOT_UNSCALED_WARNING,
    panel3d_mesh_checks,
)
from ventorum.panel3d.mesh import STEEP_ROOT_DIHEDRAL_DEG, mesh_lifting_surface
from ventorum.solvers.factory import make_solver


def _straight_wing(dihedral_deg: float = 0.0, semi_span: float = 4.0) -> LiftingSurface:
    """Return a symmetric wing with the given dihedral angle."""
    return LiftingSurface(
        name="Wing",
        semi_span=semi_span,
        dihedral=float(np.radians(dihedral_deg)),
        is_symmetric=True,
        sections=[
            WingSection(y_frac=0.0, chord=1.0, shape="0012"),
            WingSection(y_frac=1.0, chord=1.0, shape="0012"),
        ],
    )


def _exact_far_field(monkeypatch):
    """Make the solver use exact integration everywhere (far_ratio = inf)."""
    import ventorum.solvers.panel3d as p3

    class ExactPanelSolver(p3.PanelSolver):
        def __init__(self, *args, **kwargs):
            kwargs["far_ratio"] = np.inf
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(p3, "PanelSolver", ExactPanelSolver)


def test_far_field_note_ground_with_finite_far_ratio():
    """In ground effect with finite far_ratio, the far-field note appears without a RuntimeWarning."""
    wing = _straight_wing()
    cond = FlightCondition(V_inf=30.0, alpha=np.radians(3.0), h=0.20)
    settings = SolverSettings(solver_type="panel3d", n_panels=8, n_chord=6)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = vt.analyze(wing, condition=cond, settings=settings)

    # Trust note must not raise a RuntimeWarning.
    rt_warns = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert not any("far-field" in str(w.message).lower() for w in rt_warns)

    # Trust note appears in trust warnings and details notes.
    trust_warns = res.totals.trust.warnings
    assert PANEL3D_FAR_FIELD_GROUND_NOTE in trust_warns
    assert PANEL3D_FAR_FIELD_GROUND_NOTE in res.details["notes"]
    assert any("exact integration within about 0.01 %" in w and "ground" in w for w in trust_warns)
    assert any("far_ratio=inf" in w for w in trust_warns)


def test_far_field_note_ground_with_inf_far_ratio(monkeypatch):
    """In ground effect with infinite far_ratio, the far-field note does not appear."""
    wing = _straight_wing()
    cond = FlightCondition(V_inf=30.0, alpha=np.radians(3.0), h=0.20)
    settings = SolverSettings(solver_type="panel3d", n_panels=8, n_chord=6)
    _exact_far_field(monkeypatch)
    solver = make_solver("panel3d")

    res = solver.solve(wing, cond, settings)
    trust_warns = res.totals.trust.warnings
    assert PANEL3D_FAR_FIELD_GROUND_NOTE not in trust_warns
    assert not any("far-field kernel" in w.lower() for w in trust_warns)


def test_far_field_note_free_air_ns_above_80():
    """In free air with ns > 80 and finite far_ratio, the free-air far-field note appears."""
    wing = _straight_wing()
    cond = FlightCondition(V_inf=30.0, alpha=np.radians(3.0))
    settings = SolverSettings(solver_type="panel3d", n_panels=82, n_chord=4)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = vt.analyze(wing, condition=cond, settings=settings)

    rt_warns = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert not any("far-field" in str(w.message).lower() for w in rt_warns)

    trust_warns = res.totals.trust.warnings
    assert PANEL3D_FAR_FIELD_FREE_AIR_NOTE in trust_warns
    assert PANEL3D_FAR_FIELD_FREE_AIR_NOTE in res.details["notes"]
    assert any("free-air cases with more than 80 spanwise panels" in w for w in trust_warns)
    assert any("far_ratio=inf" in w for w in trust_warns)


def test_far_field_note_free_air_ns_at_or_below_80():
    """In free air with ns <= 80, the far-field note does not appear."""
    wing = _straight_wing()
    cond = FlightCondition(V_inf=30.0, alpha=np.radians(3.0))

    # ns = 80 (default)
    settings_80 = SolverSettings(solver_type="panel3d", n_panels=80, n_chord=4)
    res_80 = vt.analyze(wing, condition=cond, settings=settings_80)
    assert not any("far-field kernel" in w.lower() for w in res_80.totals.trust.warnings)

    # ns = 16
    settings_16 = SolverSettings(solver_type="panel3d", n_panels=16, n_chord=4)
    res_16 = vt.analyze(wing, condition=cond, settings=settings_16)
    assert not any("far-field kernel" in w.lower() for w in res_16.totals.trust.warnings)


def test_far_field_note_free_air_ns_above_80_inf_far_ratio(monkeypatch):
    """In free air with ns > 80 and infinite far_ratio, the far-field note does not appear."""
    wing = _straight_wing()
    cond = FlightCondition(V_inf=30.0, alpha=np.radians(3.0))
    settings = SolverSettings(solver_type="panel3d", n_panels=82, n_chord=4)
    _exact_far_field(monkeypatch)
    solver = make_solver("panel3d")

    res = solver.solve(wing, cond, settings)
    assert not any("far-field kernel" in w.lower() for w in res.totals.trust.warnings)


def test_outer_strip_note_in_ground_effect_only():
    """The outer-strip note appears in ground effect, and does not appear in free air."""
    wing = _straight_wing()
    settings = SolverSettings(solver_type="panel3d", n_panels=8, n_chord=6)

    # Ground effect: note appears without RuntimeWarning.
    cond_ground = FlightCondition(V_inf=30.0, alpha=np.radians(3.0), h=0.20)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res_ground = vt.analyze(wing, condition=cond_ground, settings=settings)

    rt_warns = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert not any("outermost strip" in str(w.message).lower() for w in rt_warns)

    trust_warns = res_ground.totals.trust.warnings
    assert PANEL3D_OUTER_STRIP_GROUND_NOTE in trust_warns
    assert PANEL3D_OUTER_STRIP_GROUND_NOTE in res_ground.details["notes"]
    assert any("outermost strip of a closed tip are not resolved" in w for w in trust_warns)

    # Free air: note does not appear.
    cond_free = FlightCondition(V_inf=30.0, alpha=np.radians(3.0))
    res_free = vt.analyze(wing, condition=cond_free, settings=settings)
    assert PANEL3D_OUTER_STRIP_GROUND_NOTE not in res_free.totals.trust.warnings
    assert not any("outermost strip" in w.lower() for w in res_free.totals.trust.warnings)


def test_dihedral_60_deg_no_warning_80_deg_warning():
    """At 60 deg dihedral no warning is issued; at 80 deg dihedral a RuntimeWarning and trust warning appear."""
    cond = FlightCondition(V_inf=30.0, alpha=np.radians(3.0))
    settings = SolverSettings(solver_type="panel3d", n_panels=8, n_chord=6)

    # 60 deg dihedral: no RuntimeWarning and no trust warning.
    wing_60 = _straight_wing(dihedral_deg=60.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res_60 = vt.analyze(wing_60, condition=cond, settings=settings)

    rt_60 = [w for w in caught if issubclass(w.category, RuntimeWarning) and "75 deg dihedral" in str(w.message)]
    assert len(rt_60) == 0
    assert PANEL3D_ROOT_SCALE_WARNING not in res_60.totals.trust.warnings

    # 80 deg dihedral: RuntimeWarning and trust warning.
    wing_80 = _straight_wing(dihedral_deg=80.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res_80 = vt.analyze(wing_80, condition=cond, settings=settings)

    rt_80 = [w for w in caught if issubclass(w.category, RuntimeWarning) and "75 deg dihedral" in str(w.message)]
    assert len(rt_80) >= 1
    assert PANEL3D_ROOT_SCALE_WARNING in res_80.totals.trust.warnings
    assert PANEL3D_ROOT_SCALE_WARNING in res_80.details["notes"]

    # 80 deg dihedral with scale_root_thickness=False: no warning.
    settings_no_scale = SolverSettings(solver_type="panel3d", n_panels=8, n_chord=6, panel_scale_root_thickness=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res_no_scale = vt.analyze(wing_80, condition=cond, settings=settings_no_scale)

    rt_no_scale = [w for w in caught if issubclass(w.category, RuntimeWarning) and "75 deg dihedral" in str(w.message)]
    assert len(rt_no_scale) == 0
    assert PANEL3D_ROOT_SCALE_WARNING not in res_no_scale.totals.trust.warnings


def test_fold_guard_tilt_above_75_deg_warning():
    """When fold guard turns a station by more than 75 deg, raise RuntimeWarning and add trust warning."""
    # Build a surface with steep dihedral break or test mesh_lifting_surface with tilt > 75 deg.
    from ventorum.panel3d import mesh as mesh_mod

    from ventorum.airfoil import airfoil
    coords = airfoil("0012").coords
    surf = _straight_wing(dihedral_deg=10.0)

    # Monkeypatch _guard_axes to simulate a fold guard tilt of 76 deg.
    orig_guard = mesh_mod._guard_axes

    def mocked_guard(axis, qc, offsets, margin=mesh_mod.FOLD_MARGIN):
        out_axis, tilt = orig_guard(axis, qc, offsets, margin)
        # Force station 1 tilt to 76 deg.
        tilt = tilt.copy()
        if tilt.size > 1:
            tilt[1] = math.radians(76.0)
        return out_axis, tilt

    from pytest import MonkeyPatch

    mp = MonkeyPatch()
    try:
        mp.setattr(mesh_mod, "_guard_axes", mocked_guard)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mesh = mesh_lifting_surface(surf, coords, n_chord=6, n_span=6, half=True, scale_root_thickness=True)

        rt = [w for w in caught if issubclass(w.category, RuntimeWarning) and "75 deg dihedral" in str(w.message)]
        assert len(rt) >= 1
        assert getattr(mesh, "root_scale_warning", False) is True

        # When solved, trust warning appears.
        cond = FlightCondition(V_inf=30.0, alpha=np.radians(2.0))
        settings = SolverSettings(solver_type="panel3d", n_panels=6, n_chord=6, panel_cache=False)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            res = vt.analyze(surf, condition=cond, settings=settings)
        assert PANEL3D_ROOT_SCALE_WARNING in res.totals.trust.warnings

    finally:
        mp.undo()


def test_mesh_checks_root_scale_warning_direct():
    """panel3d_mesh_checks adds the warning when mesh_stats has root_scale_warning=True."""
    # When root_scale_warning is True
    _, warns_true, _ = panel3d_mesh_checks({"root_scale_warning": True})
    assert PANEL3D_ROOT_SCALE_WARNING in warns_true

    # When root_scale_warning is False or absent
    _, warns_false, _ = panel3d_mesh_checks({"root_scale_warning": False})
    assert PANEL3D_ROOT_SCALE_WARNING not in warns_false

    _, warns_none, _ = panel3d_mesh_checks({})
    assert PANEL3D_ROOT_SCALE_WARNING not in warns_none


def test_trust_score_numbers_unchanged():
    """The added notes and warnings do not alter the trust score composite value or add penalties."""
    wing = _straight_wing()
    cond_ground = FlightCondition(V_inf=30.0, alpha=np.radians(3.0), h=0.20)
    settings = SolverSettings(solver_type="panel3d", n_panels=8, n_chord=6)
    res = vt.analyze(wing, condition=cond_ground, settings=settings)

    # Score factors must not have entries for far_field or outer_strip or root_scale.
    factors = res.totals.trust.factors
    assert "far_field" not in factors
    assert "outer_strip" not in factors
    assert "root_scale" not in factors


def test_parity_coefficients_bit_identical():
    """Aerodynamic totals CL, Cm, CDi are bit-identical whether notes are present or not."""
    wing = _straight_wing()
    cond_ground = FlightCondition(V_inf=30.0, alpha=np.radians(3.0), h=0.20)
    settings = SolverSettings(solver_type="panel3d", n_panels=8, n_chord=6, panel_cache=False)

    solver = make_solver("panel3d")
    res1 = solver.solve(wing, cond_ground, settings)
    res2 = solver.solve(wing, cond_ground, settings)

    assert res1.totals.CL == res2.totals.CL
    assert res1.totals.Cm == res2.totals.Cm
    assert res1.totals.CDi == res2.totals.CDi


def _singular_root_warnings(caught) -> list:
    """Return the RuntimeWarnings of a steep root without the root thickness scaling."""
    return [w for w in caught if issubclass(w.category, RuntimeWarning) and "singular" in str(w.message)]


def test_steep_root_constant_is_60_deg():
    """The steep-root limit is one code setting of 60 deg."""
    assert STEEP_ROOT_DIHEDRAL_DEG == 60.0


def test_steep_unscaled_root_mesh_warning():
    """Without the root scaling, a first-row dihedral of 60 deg or more gives a RuntimeWarning."""
    from ventorum.airfoil import airfoil

    coords = airfoil("0012").coords
    for dihedral, scale, expected in ((55.0, False, 0), (60.0, False, 1), (65.0, False, 1), (65.0, True, 0)):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mesh_lifting_surface(_straight_wing(dihedral), coords, n_chord=6, n_span=8,
                                 scale_root_thickness=scale)
        found = _singular_root_warnings(caught)
        assert len(found) == expected, (dihedral, scale)
        if expected:
            assert "'Wing'" in str(found[0].message)
            assert "scale_root_thickness=True" in str(found[0].message)


def test_steep_unscaled_root_trust_warning():
    """Without the root scaling, a steep root gives the trust warning and the Python warning in a solve."""
    cond = FlightCondition(V_inf=30.0, alpha=np.radians(3.0))
    settings = SolverSettings(solver_type="panel3d", n_panels=8, n_chord=6, panel_cache=False,
                              panel_scale_root_thickness=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = vt.analyze(_straight_wing(65.0), condition=cond, settings=settings)
    assert len(_singular_root_warnings(caught)) >= 1
    assert PANEL3D_STEEP_ROOT_UNSCALED_WARNING in res.totals.trust.warnings
    assert PANEL3D_STEEP_ROOT_CONVERGENCE_NOTE not in res.totals.trust.warnings
    assert res.details["mesh_stats"]["steep_root_dihedral"] is True

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = vt.analyze(_straight_wing(55.0), condition=cond, settings=settings)
    assert _singular_root_warnings(caught) == []
    assert PANEL3D_STEEP_ROOT_UNSCALED_WARNING not in res.totals.trust.warnings


def test_steep_scaled_root_convergence_note():
    """With the root scaling (the default), a steep root gives the trust note of a spanwise convergence study."""
    cond = FlightCondition(V_inf=30.0, alpha=np.radians(3.0))
    settings = SolverSettings(solver_type="panel3d", n_panels=8, n_chord=6, panel_cache=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = vt.analyze(_straight_wing(65.0), condition=cond, settings=settings)
    assert PANEL3D_STEEP_ROOT_CONVERGENCE_NOTE in res.totals.trust.warnings
    assert PANEL3D_STEEP_ROOT_UNSCALED_WARNING not in res.totals.trust.warnings
    assert _singular_root_warnings(caught) == []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = vt.analyze(_straight_wing(55.0), condition=cond, settings=settings)
    assert PANEL3D_STEEP_ROOT_CONVERGENCE_NOTE not in res.totals.trust.warnings


def test_steep_root_mesh_checks_direct():
    """panel3d_mesh_checks selects the note or the warning from the scaling flag; neither adds a penalty."""
    pen, warns, recs = panel3d_mesh_checks({"steep_root_dihedral": True, "scale_root_thickness": True})
    assert PANEL3D_STEEP_ROOT_CONVERGENCE_NOTE in warns
    assert PANEL3D_STEEP_ROOT_UNSCALED_WARNING not in warns
    assert any("spanwise panel count" in r for r in recs)
    assert pen["mesh_adequacy"] == 0.0

    pen, warns, recs = panel3d_mesh_checks({"steep_root_dihedral": True, "scale_root_thickness": False})
    assert PANEL3D_STEEP_ROOT_UNSCALED_WARNING in warns
    assert PANEL3D_STEEP_ROOT_CONVERGENCE_NOTE not in warns
    assert any("panel_scale_root_thickness=True" in r for r in recs)
    assert pen["mesh_adequacy"] == 0.0

    _, warns, _ = panel3d_mesh_checks({"steep_root_dihedral": False, "scale_root_thickness": False})
    assert PANEL3D_STEEP_ROOT_UNSCALED_WARNING not in warns
    assert PANEL3D_STEEP_ROOT_CONVERGENCE_NOTE not in warns
