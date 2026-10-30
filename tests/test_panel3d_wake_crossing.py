"""Wake sheets that pass through another closed surface in the 3D panel method.

The 3D panel method sheds a rigid flat wake of constant doublet strength
(along the body x axis in free air, along the free stream in ground
effect). Two equal wings in tandem at the same height, the rear wing 2 m
behind the front wing, put the front wake through the rear wing. The
solver returned CL of about -2.5 at alpha 4 deg (the VLM gives about 0.54)
with a passed contact check. The solver now refuses a wake sheet that
touches or crosses another closed surface. Configurations with a clear
vertical gap between a wake and the surfaces behind it do not change.
"""

import numpy as np
import pytest

import ventorum as vt
from ventorum.core.errors import ValidityError
from ventorum.panel3d.wake import PanelWake, check_wake_crossing

RTOL = 1e-10
ATOL = 1e-12


def _wing(name, semi_span, chord, position, incidence_deg=0.0):
    """Rectangular wing (default NACA 0012 sections) at a given position [m]."""
    return vt.LiftingSurface(
        name=name,
        semi_span=semi_span,
        incidence=np.radians(incidence_deg),
        position=np.array(position, dtype=float),
        sections=[vt.WingSection(y_frac=0.0, chord=chord), vt.WingSection(y_frac=1.0, chord=chord)],
    )


def _tandem(dz):
    """Two equal wings (semi-span 4 m, chord 1 m), the rear wing 3 m aft and dz [m] higher."""
    return vt.Aircraft(
        name="Tandem",
        surfaces=[_wing("Front", 4.0, 1.0, (0.0, 0.0, 0.0)), _wing("Rear", 4.0, 1.0, (3.0, 0.0, dz))],
        ref_point=np.array([1.5, 0.0, 0.0]),
    )


def _settings():
    return vt.SolverSettings(solver_type="panel3d", n_panels=8, n_chord=10, panel_cache=False)


def _totals(aircraft, alpha_deg=4.0, **kw):
    res = vt.analyze(aircraft, vt.FlightCondition(V_inf=30.0, alpha=np.radians(alpha_deg), **kw), _settings())
    t = res.totals
    return np.array([t.CL, t.CDi, t.Cm])


@pytest.mark.parametrize("dz", [0.0, 0.05, -0.05])
def test_tandem_with_wake_through_rear_wing_is_refused(dz):
    """The front wake passes through the rear wing (|dz| below the half thickness of 0.06 m).

    Regression: the solver returned CL of about -2.5 (dz = 0) and -4.8
    (dz = 0.05 m) at alpha 4 deg, rated UNRELIABLE.
    """
    with pytest.raises(ValidityError) as exc:
        vt.analyze(_tandem(dz), alpha_deg=4.0, settings=_settings())
    msg = str(exc.value)
    assert "wake of 3D panel component 0" in msg
    assert "component 1" in msg


def test_wake_through_rear_wing_in_ground_effect_is_refused():
    """In ground effect the wake follows the free stream; at alpha 0.5 deg it still crosses the rear wing."""
    with pytest.raises(ValidityError):
        _totals(_tandem(0.0), alpha_deg=0.5, h=1.0)


# Stored baselines: the values of the solver before the wake check, updated for
# the second-moment far-field expansion about the panel centroid (relative
# change 7e-5 to 4e-4). Updated again for the least-squares node values of
# the surface gradient: with the previous inverse-distance node values
# (_build_surface_gradient_operator_shepard) the four cases give the previous
# values to 5e-13 relative, so only the gradient moves them. CL changes by
# 1.2 to 1.3 %, Cm by 1 to 35 % (10 chordwise panels), and the Trefftz CDi
# does not change.
_BASELINES = {
    "tandem_offset": (0.554367196934527, 0.010964617383178119, 0.07505217808509079),
    "wing_tail": (0.3452886021683871, 0.004621505025221453, -0.02550505624312208),
    "biplane": (0.5066005708654079, 0.00805895008372906, -0.02923308977602326),
    "tandem_ground": (0.6666319507264357, 0.008827328079694476, -0.008389251198743803),
}


def _case(name):
    if name == "tandem_offset":
        return _tandem(0.3), {}
    if name == "wing_tail":
        surfaces = [_wing("Wing", 4.0, 1.0, (0.0, 0.0, 0.0)), _wing("Tail", 1.5, 0.6, (4.0, 0.0, 0.3), -2.0)]
        return vt.Aircraft(name="WingTail", surfaces=surfaces, ref_point=np.array([0.3, 0.0, 0.0])), {}
    if name == "biplane":
        surfaces = [_wing("Lower", 4.0, 1.0, (0.0, 0.0, 0.0)), _wing("Upper", 4.0, 1.0, (0.3, 0.0, 1.0))]
        return vt.Aircraft(name="Biplane", surfaces=surfaces, ref_point=np.array([0.3, 0.0, 0.5])), {}
    # Same height in ground effect: the free-stream wake at alpha 4 deg passes above the rear wing.
    return _tandem(0.0), {"h": 1.0}


@pytest.mark.parametrize("name", sorted(_BASELINES))
def test_configurations_with_clear_wake_do_not_change(name):
    """Offset tandem, wing and tail, biplane and a tandem in ground effect keep their results."""
    aircraft, kw = _case(name)
    np.testing.assert_allclose(_totals(aircraft, **kw), _BASELINES[name], rtol=RTOL, atol=ATOL)


def test_wake_just_below_rear_wing_solves():
    """A wake 5 mm below the rear wing (far less than one panel size) is not refused.

    The potential of the wake sheet is continuous on each side of the
    sheet, so a small gap gives the lift of a larger gap: the lift changes
    smoothly from a gap of 5 mm (dz = 0.065 m) to a gap of 40 mm (dz = 0.1 m).
    """
    near = _totals(_tandem(0.065))
    clear = _totals(_tandem(0.1))
    assert 0.4 < near[0] < 0.7
    assert abs(near[0] - clear[0]) < 0.01 * abs(clear[0])


def _cube(center, half):
    """Closed cube of 6 outward quad panels; returns nodes (8, 3) [m] and panels (6, 4)."""
    c = np.asarray(center, dtype=float)
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
    nodes = c + half * signs
    panels = np.array(
        [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]], dtype=int
    )
    return nodes, panels


def _sheet_case(z_center):
    """One flat wake panel in z = 0 (x 0 to 10 m, y -1 to 1 m) of component 0 and a cube of component 1."""
    cube_nodes, cube_panels = _cube((5.0, 0.0, z_center), 0.5)
    # Component 0: one panel (the trailing-edge panel of the wake strip).
    src_nodes = np.array([[-1.0, -1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 1.0, 0.0]])
    nodes = np.vstack([src_nodes, cube_nodes])
    panels = np.vstack([np.array([[0, 1, 2, 3]]), cube_panels + 4])
    component = np.array([0] + [1] * 6)
    sheet = np.array([[[0.0, -1.0, 0.0], [10.0, -1.0, 0.0], [10.0, 1.0, 0.0], [0.0, 1.0, 0.0]]])
    wake = PanelWake(
        panels=sheet,
        wake_up=np.array([0]),
        wake_lo=np.array([0]),
        strip_panel_indices=[[0]],
        p_left=sheet[:, 0],
        p_right=sheet[:, 3],
        wake_edges=np.array([[0, 0, 1, 2]]),
    )
    return nodes, panels, component, wake


@pytest.mark.parametrize("z_center, flagged", [(0.0, True), (0.3, True), (0.5, True), (-0.5, True), (0.6, False)])
def test_check_wake_crossing_flags_touch_and_cross_only(z_center, flagged):
    """A sheet through a cube or on a cube face is flagged; a sheet 0.1 m below the cube is not."""
    nodes, panels, component, wake = _sheet_case(z_center)
    out = check_wake_crossing(nodes, panels, component, wake)
    assert (out["n_crossing_edges"] > 0) is flagged
    if flagged:
        assert out["crossing_pairs"][0][:2] == (0, 1)
    else:
        assert out["crossing_pairs"] == []
        assert out["min_gap"] == pytest.approx(0.1, abs=1e-12)


def test_check_wake_crossing_ignores_the_own_component():
    """The wake of a component is not checked against the same component."""
    nodes, panels, _, wake = _sheet_case(0.0)
    out = check_wake_crossing(nodes, panels, np.zeros(panels.shape[0], dtype=int), wake)
    assert out["n_crossing_edges"] == 0
