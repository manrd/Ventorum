"""Shared-edge join of surfaces for the cross-surface vortex core (T-0012).

Surfaces that meet along an edge are one vortex sheet. They must be in one
core group also when their edges match only approximately. The reference of
each test is the same geometry with the join forced (one core group for all
surfaces), or the one-piece surface where one exists. Edges that are near
but not joined give a warning.
"""

import warnings

import numpy as np
import pytest

import ventorum as vt
from ventorum.geometry import lattice as lattice_module
from ventorum.geometry.lattice import build_lattice

COND = vt.FlightCondition(V_inf=20.0, alpha=np.radians(5.0))
N = 40


def _surface(name, semi_span, n, chord=1.0, tip_chord=None, **kw):
    tip_chord = chord if tip_chord is None else tip_chord
    return vt.LiftingSurface(
        name=name, semi_span=semi_span, n_panels=n, spacing="uniform",
        sections=[vt.WingSection(y_frac=0.0, chord=chord), vt.WingSection(y_frac=1.0, chord=tip_chord)], **kw,
    )


def _aircraft(surfaces, S_ref=8.0, b_ref=8.0):
    return vt.Aircraft(surfaces=surfaces, S_ref=S_ref, b_ref=b_ref, c_ref=1.0)


def _totals(ac):
    return vt.analyze(ac, condition=COND, settings=vt.SolverSettings(n_panels=N)).totals


def _groups(ac):
    lat = build_lattice(ac, vt.SolverSettings(n_panels=N), collocation="vlm")
    return {s.name: int(lat.strip_core_group[s.strips][0]) for s in lat.surfaces}, lat


def _force_join(monkeypatch):
    """Put all surfaces of the next analysis into one core group."""
    monkeypatch.setattr(lattice_module, "core_groups", lambda surfaces: np.zeros(len(surfaces), dtype=int))


def test_split_wing_with_small_gap():
    """A 0.05 mm gap on a 4 m wing must not change the lift (4th review: -6.6 %)."""
    one = _aircraft([_surface("one", 4.0, N)])
    inner = _surface("inner", 2.0, N // 2)
    outer = _surface("outer", 2.0, N // 2, is_symmetric=False, position=np.array([0.0, 2.0 + 5e-5, 0.0]))
    split = _aircraft([inner, outer, outer.mirrored()])
    groups, _ = _groups(split)
    assert len(set(groups.values())) == 1
    assert _totals(split).CL == pytest.approx(_totals(one).CL, rel=5e-3)


def test_dihedral_break_with_incidence(monkeypatch):
    """3 deg of incidence at a dihedral break (4th review: CL 4.5 % low).

    Reference: the same geometry with the join forced.
    """
    dih = np.radians(8.0)
    inner = _surface("inner", 2.0, N // 2)
    tip = np.array([0.0, 2.0, 0.0])
    outer = vt.LiftingSurface(
        name="outer", semi_span=2.0, n_panels=N // 2, spacing="uniform", dihedral=dih, is_symmetric=False,
        position=tip, incidence=np.radians(3.0),
        sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=1.0)],
    )
    ac = _aircraft([inner, outer, outer.mirrored()])
    groups, _ = _groups(ac)
    assert len(set(groups.values())) == 1
    got = _totals(ac).CL
    _force_join(monkeypatch)
    assert got == pytest.approx(_totals(ac).CL, rel=5e-3)


def test_winglet_with_smaller_chord(monkeypatch):
    """A winglet root with 95 % of the wing-tip chord (4th review: CDi 14 % off).

    Reference: the same geometry with the join forced.
    """
    wing = _surface("wing", 4.0, N)
    winglet = _surface("winglet", 0.6, 6, chord=0.95, tip_chord=0.6, is_symmetric=False,
                       dihedral=np.radians(80.0), position=np.array([0.0, 4.0, 0.0]))
    ac = _aircraft([wing, winglet, winglet.mirrored()])
    groups, _ = _groups(ac)
    assert len(set(groups.values())) == 1
    got = _totals(ac).CDi
    _force_join(monkeypatch)
    assert got == pytest.approx(_totals(ac).CDi, rel=1e-2)


@pytest.mark.parametrize("dihedral_deg", [30.0, 45.0])
def test_v_tail_from_two_halves(dihedral_deg):
    """A V-tail made of two mirrored halves equals the one mirrored V-tail."""
    dih = np.radians(dihedral_deg)
    wing = _surface("wing", 4.0, N)
    pos = np.array([5.0, 0.0, 0.4])
    one = _surface("vtail", 1.2, 12, dihedral=dih, position=pos)
    half = _surface("half", 1.2, 12, dihedral=dih, position=pos, is_symmetric=False)
    halves = [half, half.mirrored()]
    r1 = _totals(_aircraft([wing, one]))
    r2 = _totals(_aircraft([wing, *halves]))
    assert r2.CL == pytest.approx(r1.CL, rel=1e-9)
    assert r2.CDi == pytest.approx(r1.CDi, rel=1e-9)
    assert r2.Cm == pytest.approx(r1.Cm, rel=1e-9, abs=1e-12)


def test_separate_surfaces_are_not_joined():
    """A wing and a horizontal tail are two groups and give no join warning."""
    wing = _surface("wing", 4.0, N)
    tail = _surface("tail", 1.5, 12, position=np.array([5.0, 0.0, 0.3]))
    ac = _aircraft([wing, tail])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        groups, lat = _groups(ac)
    assert groups["wing"] != groups["tail"]
    assert not lat.join_warnings


def test_tail_close_behind_wing_gives_no_warning():
    """Edges on one line but apart along it (a tail 0.5 chord behind the wing tip) are no near miss."""
    wing = _surface("wing", 4.0, N)
    tail = _surface("tail", 4.0, N, position=np.array([1.5, 0.0, 0.0]))
    _, lat = _groups(_aircraft([wing, tail]))
    assert not lat.join_warnings


def test_gap_above_the_tolerance_is_not_joined():
    """A gap of 1.5 strip widths is not a join (the tolerance is one strip width)."""
    inner = _surface("inner", 2.0, N // 2)
    outer = _surface("outer", 2.0, N // 2, is_symmetric=False, position=np.array([0.0, 2.15, 0.0]))
    groups, _ = _groups(_aircraft([inner, outer, outer.mirrored()]))
    assert groups["inner"] != groups["outer"]


def test_small_overlap_is_not_joined():
    """Edges that overlap over less than half of the shorter one are not joined."""
    wing = _surface("wing", 4.0, N)
    # Root chord 0.3 at the tip of a wing with chord 1: 30 % overlap along the edge line.
    stub = _surface("stub", 0.5, 4, chord=0.3, is_symmetric=False, position=np.array([0.7, 4.0, 0.0]))
    # The shorter edge (0.3) lies fully on the wing tip edge, so it is joined.
    groups, _ = _groups(_aircraft([wing, stub, stub.mirrored()]))
    assert groups["wing"] == groups["stub"]
    # Shift the stub so that only 20 % of its edge lies on the wing tip edge.
    stub2 = _surface("stub2", 0.5, 4, chord=0.3, is_symmetric=False, position=np.array([0.94, 4.0, 0.0]))
    groups, _ = _groups(_aircraft([wing, stub2, stub2.mirrored()]))
    assert groups["wing"] != groups["stub2"]


def test_near_miss_warns():
    """A gap of 0.3 chord between two parallel edges warns, in the warning and in the result."""
    inner = _surface("inner", 2.0, N // 2)
    outer = _surface("outer", 2.0, N // 2, is_symmetric=False, position=np.array([0.0, 2.3, 0.0]))
    ac = _aircraft([inner, outer, outer.mirrored()])
    with pytest.warns(RuntimeWarning, match=r"inner.*outer|outer.*inner") as rec:
        res = vt.analyze(ac, condition=COND, settings=vt.SolverSettings(n_panels=N))
    text = " ".join(str(w.message) for w in rec)
    assert "300" in text and "mm" in text            # gap of 0.3 m = 300 mm
    assert any("outer" in n and "mm" in n for n in res.details["notes"])
