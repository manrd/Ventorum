"""A ground sweep with a root flap uses the chordwise count of a single solve.

With ``height_ref='te'`` the height reference point is the root trailing
edge, rotated with the root flap. The sweep must place the ground with the
same rotated point when it selects its fixed chordwise count. Otherwise the
count, and thus CL, of a sweep case differ from a single solve of the same
case (before the fix: 16 against 32 panels and CL -21.9 % at h 0.1 m with
the trailing edge up).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import ventorum as vt
from ventorum import gpu
from ventorum.ground_effect import GroundEffectSweep, analyze_ground_effect

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


@pytest.fixture(autouse=True)
def _cpu():
    old_device, old_precision = gpu.get_device(), gpu._precision
    gpu.set_device("cpu")
    yield
    gpu.set_device(old_device)
    gpu.set_precision(old_precision)


def _flapped_wing(deflection_deg: float) -> vt.Aircraft:
    wing = vt.LiftingSurface(
        name="w", semi_span=3.0,
        sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=1.0)],
        controls=[vt.ControlSurface(name="flap", eta_start=0.0, eta_end=0.5, hinge_x_c=0.7,
                                    deflection=math.radians(deflection_deg), symmetric=True)],
        n_panels=10,
    )
    return vt.Aircraft(name="flapped", surfaces=[wing], ref_point=np.array([0.25, 0.0, 0.0]))


@pytest.mark.parametrize("deflection_deg", [-15.0, 15.0])
@pytest.mark.parametrize("h", [0.1, 0.2])
def test_sweep_case_equals_single_solve_with_root_flap(deflection_deg, h):
    ac = _flapped_wing(deflection_deg)
    settings = vt.SolverSettings(solver_type="vlm", n_panels=10)
    sweep = GroundEffectSweep(ac, settings=settings, backend="serial").run_sweep(
        [h], [3.0], [0.0], height_ref="te", compute_strike_limit=False)
    single = analyze_ground_effect(ac, h=h, alpha_deg=3.0, settings=settings, height_ref="te",
                                   compute_strike_limit=False)
    case = sweep.result_at(0, 0, 0)
    assert sweep.n_chord == single.n_chord
    assert case.n_chord == single.n_chord
    assert case.CL == pytest.approx(single.CL, rel=1e-10, abs=1e-12)
    assert case.Cm == pytest.approx(single.Cm, rel=1e-10, abs=1e-12)
    assert case.h_min == pytest.approx(single.h_min, rel=1e-12, abs=1e-14)
