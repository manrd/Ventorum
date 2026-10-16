# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Refinement regressions of the 3D panel method: open trailing edges.

The mesh-error study of the 3D panel method found that the open (blunt)
trailing edge of the library NACA sections did not converge with
chordwise refinement.
"""

from __future__ import annotations

import numpy as np
import pytest

from ventorum.airfoil import airfoil
from ventorum.core.datatypes import Aircraft, FlightCondition, LiftingSurface, SolverSettings, WingSection
from ventorum.panel3d.convergence import order_and_extrapolation
from ventorum.panel3d.mesh import mesh_lifting_surface
from ventorum.panel3d.solver import PanelSolver
from ventorum.solvers.factory import make_solver

_COND = FlightCondition(V_inf=30.0, alpha=np.radians(5.0))


def _wing(shape=None) -> LiftingSurface:
    """Rectangular wing of aspect ratio 8 (chord 1 m, semi-span 4 m)."""
    return LiftingSurface(
        name="Wing",
        semi_span=4.0,
        sections=[WingSection(y_frac=0.0, chord=1.0, shape=shape), WingSection(y_frac=1.0, chord=1.0, shape=shape)],
        is_symmetric=True,
    )


def _half_cl(surf: LiftingSurface, coords: np.ndarray, n_chord: int, n_span: int) -> float:
    """CL of a half mesh at alpha 5 deg (reference area 8 m^2)."""
    mesh = mesh_lifting_surface(surf, coords, n_chord=n_chord, n_span=n_span, half=True)
    res = PanelSolver(mesh, S_ref=8.0, b_ref=8.0, c_ref=1.0, backend="numba").solve(_COND)
    return float(res.CL)


def test_open_trailing_edge_gives_the_closed_section_lift():
    """Regression: the open trailing edge of the library NACA 0012 lost lift on fine chordwise meshes.

    With base panels and the wake at the upper corner, CL of the open
    section fell below CL of the closed section as soon as the
    trailing-edge panels became shorter than the base height (by almost
    2 % at 32 chordwise panels). The default mesh now closes the open
    edge; CL agrees with the closed section to 0.5 %.
    """
    surf = _wing()
    cl_open = _half_cl(surf, airfoil("0012").coords, 32, 4)
    cl_closed = _half_cl(surf, airfoil("0012", closed_te=True).coords, 32, 4)
    print(f"n_chord 32: CL open {cl_open:.5f} closed {cl_closed:.5f}")
    assert abs(cl_open / cl_closed - 1.0) < 0.005


@pytest.mark.slow
@pytest.mark.parametrize("section", ["0012", "2412"])
def test_open_trailing_edge_converges_with_chordwise_refinement(section):
    """CL of the library sections converges with n_chord and agrees with the closed section.

    Aspect ratio 8, alpha 5 deg, 8 spanwise panels, n_chord 16, 24, 32 and
    48, solved through the factory with the library section (open
    trailing edge) and with the closed-trailing-edge section. The CL
    sequence of the open section must be monotonic, the Richardson value
    of its three finest levels must exist, and both the finest value and
    the Richardson value must be within 0.5 % of the closed section.

    Tolerance reasons: the closure of the mesh builder gives the closed
    NACA thickness law of Ladson et al. (1996) up to the interpolation of
    the section points, so for NACA 0012 the two wings are almost the same
    mesh. For NACA 2412 the published closed section moves the thickness
    normal to the camber line and the closure moves it normal to the
    chord, a small difference near the trailing edge. 0.5 % is the
    acceptance limit of this regression: well below the 2 % CL warning
    level of the trust score, and far below the difference of the former
    base-panel mesh, which grew with n_chord.
    """
    solver = make_solver("panel3d")
    levels = (16, 24, 32, 48)

    def series(shape):
        surf = _wing(shape=shape)
        ac = Aircraft(name="W", surfaces=[surf])
        ac.compute_reference_values()
        out = []
        for n_chord in levels:
            st = SolverSettings(n_panels=8, n_chord=n_chord, panel_backend="numba")
            out.append(float(solver.solve(ac, _COND, st).totals.CL))
        return out

    open_cl = series(section)
    closed_cl = series(airfoil(section, closed_te=True).coords)
    print(f"{section}: open {np.round(open_cl, 5)} closed {np.round(closed_cl, 5)}")
    steps = np.diff(open_cl)
    assert np.all(steps > 0.0) or np.all(steps < 0.0)
    conv = order_and_extrapolation(open_cl[1:], list(levels[1:]), quantity="CL")
    conv_closed = order_and_extrapolation(closed_cl[1:], list(levels[1:]), quantity="CL")
    print(f"{section}: open order {conv.observed_order} value {conv.f_extrapolated}; "
          f"closed order {conv_closed.observed_order} value {conv_closed.f_extrapolated}")
    assert conv.f_extrapolated is not None and conv_closed.f_extrapolated is not None
    assert abs(open_cl[-1] / closed_cl[-1] - 1.0) < 0.005
    assert abs(conv.f_extrapolated / conv_closed.f_extrapolated - 1.0) < 0.005
