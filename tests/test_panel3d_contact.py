"""Contact check of distinct closed surfaces in the 3D panel method.

Each surface of a 3D panel model is one closed body. Separate surfaces
that touch or pass through each other gave lift coefficients of the order
of -1e3 to -1e5 with a watertight mesh check, so the solver now refuses
them. Surfaces with a clear gap still solve.
"""

import numpy as np
import pytest

import ventorum as vt
from ventorum.panel3d.mesh import CONTACT_GAP_FRACTION, check_contact


def _wing_with_winglets(gap):
    """Wing (semi-span 4 m, chord 1 m) with 90 deg winglets at y = +/-(4 + gap) m."""
    wing = vt.LiftingSurface(
        name="Wing",
        semi_span=4.0,
        sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=1.0)],
    )
    winglet = vt.LiftingSurface(
        name="Winglet",
        semi_span=0.8,
        dihedral=np.radians(90.0),
        position=np.array([0.0, 4.0 + gap, 0.0]),
        is_symmetric=False,
        sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=0.5)],
    )
    return vt.Aircraft(name="Winglets", surfaces=[wing, winglet, winglet.mirrored()])


def _settings():
    return vt.SolverSettings(solver_type="panel3d", n_panels=8, n_chord=10, panel_cache=False)


@pytest.mark.parametrize("gap", [0.0, 0.02])
def test_touching_closed_surfaces_are_refused(gap):
    """Winglets that touch the wing tip (0 m) or pass into it (0.02 m < half thickness) are refused.

    Regression: the solver returned CL of about -1192 and -63380 for the
    touching case and about 0.61 for the 0.02 m case (the wing alone gives
    about 0.33).
    """
    with pytest.raises(ValueError) as exc:
        vt.analyze(_wing_with_winglets(gap), alpha_deg=4.0, settings=_settings())
    msg = str(exc.value)
    assert "Wing" in msg and "Winglet" in msg
    assert "joined closed surfaces" in msg
    assert "gap" in msg


def test_half_wing_and_mirror_with_closed_roots_are_refused():
    """A one-sided half wing from y = 0 plus its mirror copy has two root caps on each other."""
    right = vt.LiftingSurface(
        name="RightHalf",
        semi_span=4.0,
        is_symmetric=False,
        sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=1.0)],
    )
    aircraft = vt.Aircraft(name="Split", surfaces=[right, right.mirrored("LeftHalf")])
    with pytest.raises(ValueError, match="joined closed surfaces"):
        vt.analyze(aircraft, alpha_deg=4.0, settings=_settings())


def test_surfaces_with_a_clear_gap_still_solve():
    """Winglets 0.2 m from the wing tip are accepted; the lift is near the wing alone value."""
    res = vt.analyze(_wing_with_winglets(0.2), alpha_deg=4.0, settings=_settings())
    alone = vt.analyze(
        vt.Aircraft(name="Alone", surfaces=[_wing_with_winglets(0.2).surfaces[0]]),
        alpha_deg=4.0,
        settings=_settings(),
    )
    assert np.isfinite(res.totals.CL)
    assert res.totals.CL == pytest.approx(alone.totals.CL, rel=0.05)


def test_check_contact_reports_the_gap():
    """check_contact gives the smallest gap and flags no panel for separate surfaces."""
    from ventorum.solvers.panel3d import PanelMethodSolver

    res = PanelMethodSolver().solve(_wing_with_winglets(0.2), vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0)),
                                _settings())
    report = check_contact(res.details["mesh"])
    assert report["n_contact_panels"] == 0
    # The inner face of the winglet root section is about 0.2 - 0.06 m from the wing tip cap.
    assert 0.1 < report["min_gap"] < 0.2
    assert report["min_gap_ratio"] > CONTACT_GAP_FRACTION
