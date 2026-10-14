# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Speed benchmark cases: shared geometries and angle sweeps.

The cases reuse the shapes of the AVL comparison harness:
a rectangular wing, a tapered wing, a swept tapered wing and a
wing with a horizontal tail. All runs are in free air. Ground
effect runs only for tools that support it (see the runner).
"""

from __future__ import annotations

import numpy as np

import ventorum as vt

#: Free-stream velocity [m/s] of every case.
V_INF = 50.0

#: Chordwise panels of every Ventorum run.
N_CHORD = 4

#: Lowest angle [deg] of the sweep.
ALPHA_MIN_DEG = -4.0

#: Highest angle [deg] of the sweep.
ALPHA_MAX_DEG = 10.0

#: Quick sweep angles [deg] (spans the range, includes a negative angle).
QUICK_ANGLES = [-4.0, 4.0, 10.0]


def build_rect():
    """Return a rectangular wing of aspect ratio 8 (chord 1 m)."""
    surf = vt.LiftingSurface(
        name="Wing",
        semi_span=4.0,
        sections=[vt.WingSection(0.0, 1.0), vt.WingSection(1.0, 1.0)],
    )
    return vt.Aircraft(name="rect_AR8", surfaces=[surf])


def build_tapered():
    """Return a straight tapered wing of aspect ratio 8 (taper 0.5)."""
    surf = vt.LiftingSurface(
        name="Wing",
        semi_span=4.0,
        sections=[vt.WingSection(0.0, 4.0 / 3.0), vt.WingSection(1.0, 2.0 / 3.0)],
    )
    return vt.Aircraft(name="tapered_AR8", surfaces=[surf])


def build_swept():
    """Return a swept (30 deg) tapered (0.5) wing of aspect ratio 6."""
    surf = vt.LiftingSurface(
        name="Wing",
        semi_span=3.0,
        sweep_le=np.radians(30.0),
        sections=[vt.WingSection(0.0, 4.0 / 3.0), vt.WingSection(1.0, 2.0 / 3.0)],
    )
    return vt.Aircraft(name="swept_tapered", surfaces=[surf])


def build_wing_tail():
    """Return a rectangular wing (AR 8) with a horizontal tail."""
    wing = vt.LiftingSurface(
        name="Wing",
        semi_span=4.0,
        sections=[vt.WingSection(0.0, 1.0), vt.WingSection(1.0, 1.0)],
    )
    tail = vt.LiftingSurface(
        name="HTail",
        semi_span=1.6,
        position=np.array([4.5, 0.0, 0.3]),
        sections=[vt.WingSection(0.0, 0.8), vt.WingSection(1.0, 0.6)],
    )
    return vt.Aircraft(name="wing_tail", surfaces=[wing, tail])


def case_list(quick: bool) -> list[dict]:
    """Return the case specs of this run (each has id, label, build)."""
    cases = [
        {"id": "rect", "label": "Rectangular wing AR 8", "build": build_rect},
        {"id": "tapered", "label": "Straight tapered wing AR 8", "build": build_tapered},
        {"id": "swept", "label": "Swept tapered wing AR 6", "build": build_swept},
        {"id": "wingtail", "label": "Wing and horizontal tail", "build": build_wing_tail},
    ]
    if quick:
        return [cases[0]]
    return cases


def sweep_angles(quick: bool) -> list[float]:
    """Return the sweep angles [deg] (33 points, or 3 when quick)."""
    if quick:
        return list(QUICK_ANGLES)
    return [float(v) for v in np.linspace(ALPHA_MIN_DEG, ALPHA_MAX_DEG, 33)]
