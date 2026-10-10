# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Tests of the geometry guard of the classical Fourier solver.

Glauert's Fourier solution models one straight, unswept, planar lifting
line. The solver refuses a wing outside that model with ``ValidityError``.
The agent tools map that error to ``invalid_method``. A wing inside the
model gives the same result as before the guard.
"""

from __future__ import annotations

import numpy as np
import pytest

import ventorum as vt
from ventorum.agent import wing_analysis
from ventorum.core.errors import ValidityError

FOURIER = vt.SolverSettings(solver_type="fourier")


def test_swept_by_x_le_is_refused():
    """Reproduction of the 4th review (item 5): sweep from x_le is refused.

    Before the guard the Fourier solver gave CL = 0.352 on this wing,
    22 % above the vortex lattice (CL = 0.288).
    """
    s = vt.LiftingSurface(
        name="S",
        semi_span=5.0,
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.0),
            vt.WingSection(y_frac=1.0, chord=1.0, x_le=3.5),
        ],
    )
    with pytest.raises(ValidityError) as exc:
        vt.analyze(s, alpha_deg=4.0, settings=FOURIER)
    msg = str(exc.value)
    assert "quarter-chord" in msg
    assert "vlm" in msg
    # The vortex lattice solves the same wing and gives the reviewed value.
    vlm = vt.analyze(s, alpha_deg=4.0, settings=vt.SolverSettings(solver_type="vlm"))
    assert vlm.totals.CL == pytest.approx(0.288, abs=5e-3)


def test_dihedral_by_z_le_is_refused():
    """A vertical offset of the sections (dihedral from z_le) is refused."""
    s = vt.LiftingSurface(
        name="D",
        semi_span=5.0,
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.0, z_le=0.0),
            vt.WingSection(y_frac=1.0, chord=1.0, z_le=1.0),
        ],
    )
    with pytest.raises(ValidityError) as exc:
        vt.analyze(s, alpha_deg=4.0, settings=FOURIER)
    msg = str(exc.value)
    assert "z_le" in msg
    assert "vlm" in msg


def test_dihedral_parameter_is_refused():
    """A dihedral angle on the surface is refused above 1e-9 times the semi-span."""
    s = vt.LiftingSurface(
        name="G",
        semi_span=5.0,
        dihedral=np.radians(3.0),
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.0),
            vt.WingSection(y_frac=1.0, chord=1.0),
        ],
    )
    with pytest.raises(ValidityError) as exc:
        vt.analyze(s, alpha_deg=4.0, settings=FOURIER)
    assert "vlm" in str(exc.value)


def test_small_sweep_below_limit_passes():
    """A quarter-chord sweep of 0.5 deg is below the 1 deg limit and solves."""
    b = 5.0  # semi-span [m]; the full span is 10 m
    tip_x = b * np.tan(np.radians(0.5))  # quarter-chord sweep = 0.5 deg
    s = vt.LiftingSurface(
        name="S05",
        semi_span=b,
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.0),
            vt.WingSection(y_frac=1.0, chord=1.0, x_le=tip_x),
        ],
    )
    res = vt.analyze(s, alpha_deg=4.0, settings=FOURIER)
    assert res.solver_type == "fourier"
    assert res.totals.CL > 0.0


def test_tapered_straight_quarter_chord_passes():
    """Taper with a straight, unswept quarter-chord line passes the guard.

    The section ``x_le`` values move the leading edge so that
    ``x_le + 0.25 * chord`` is constant over the span. The quarter-chord
    line is then the straight, unswept lifting line of the classical
    theory, and the leading edge of the tapered wing is not it.
    """
    root_chord, tip_chord, x_qc = 1.6, 0.8, 0.4
    s = vt.LiftingSurface(
        name="Taper",
        semi_span=5.0,
        sections=[
            vt.WingSection(y_frac=0.0, chord=root_chord, x_le=x_qc - 0.25 * root_chord),
            vt.WingSection(y_frac=1.0, chord=tip_chord, x_le=x_qc - 0.25 * tip_chord),
        ],
    )
    res = vt.analyze(s, alpha_deg=4.0, settings=FOURIER)
    assert res.solver_type == "fourier"
    assert res.totals.CL > 0.0
    assert np.isfinite(res.totals.CDi) and res.totals.CDi > 0.0


def test_agent_reports_invalid_method():
    """The wing analysis tool reports the refusal as error type invalid_method."""
    p = wing_analysis(
        {"span_m": 10.0, "chord_m": 1.0, "sweep_le_deg": 20.0},
        {"alpha_deg": 4.0},
        {"solver": "fourier"},
    )
    assert p["status"] == "error", p
    assert p["error"]["type"] == "invalid_method", p
    assert "vlm" in p["error"]["message"]
