"""Shared-edge join of surfaces for the cross-surface vortex core.

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


def _relate_reference(si, sj):
    """Double-loop form of the edge relation, kept as the reference of the vectorised one."""
    wi, wj = lattice_module._edge_strip_widths(si), lattice_module._edge_strip_widths(sj)
    best = None
    for e in range(si.edge_le.shape[0]):
        a0, a1 = si.edge_le[e], si.edge_te[e]
        la = float(np.linalg.norm(a1 - a0))
        if la < 1e-12:
            continue
        ua = (a1 - a0) / la
        for f in range(sj.edge_le.shape[0]):
            b0, b1 = sj.edge_le[f], sj.edge_te[f]
            lb = float(np.linalg.norm(b1 - b0))
            if lb < 1e-12:
                continue
            ub = (b1 - b0) / lb
            gap = max(
                float(np.max(lattice_module._perpendicular_distance(np.vstack([b0, b1]), a0, ua))),
                float(np.max(lattice_module._perpendicular_distance(np.vstack([a0, a1]), b0, ub))),
            )
            t = np.sort([(b0 - a0) @ ua, (b1 - a0) @ ua])
            overlap = max(0.0, min(la, float(t[1])) - max(0.0, float(t[0]))) / min(la, lb)
            if overlap < lattice_module.JOIN_MIN_OVERLAP:
                continue
            tol = float(np.clip(
                lattice_module.JOIN_TOL_STRIP_FRACTION * min(wi[e], wj[f]),
                lattice_module.JOIN_TOL_CHORD_FRACTION * max(la, lb),
                lattice_module.JOIN_TOL_CHORD_MAX_FRACTION * min(la, lb),
            ))
            if gap <= tol:
                return True, None
            angle = np.degrees(np.arccos(min(1.0, abs(float(ua @ ub)))))
            chord = min(la, lb)
            if angle < lattice_module.NEAR_MISS_MAX_ANGLE_DEG and gap <= lattice_module.NEAR_MISS_MAX_GAP_CHORDS * chord:
                if best is None or gap < best[0]:
                    best = (gap, chord)
    return False, best


def _random_edges(rng, n, y0, dy, gap_z=0.0, tilt=0.0):
    from types import SimpleNamespace

    y = y0 + dy * np.sort(rng.random(n))
    le = np.column_stack([0.1 * rng.random(n), y, gap_z + tilt * y])
    te = le + np.column_stack([0.5 + rng.random(n), 0.01 * rng.standard_normal(n), 0.01 * rng.standard_normal(n)])
    return SimpleNamespace(edge_le=le, edge_te=te)


def test_vectorised_edge_relation_equals_the_double_loop():
    """The vectorised edge relation gives the same join and near miss as the double loop."""
    rng = np.random.default_rng(7)
    seen = {"joined": 0, "near": 0, "none": 0}
    for k in range(300):
        n_i, n_j = int(rng.integers(3, 12)), int(rng.integers(3, 12))
        si = _random_edges(rng, n_i, 0.0, 2.0)
        sj = _random_edges(rng, n_j, rng.uniform(-0.5, 2.5), rng.uniform(0.5, 2.0),
                           gap_z=rng.choice([0.0, 1e-4, 0.05, 0.3, 5.0]), tilt=rng.choice([0.0, 0.05, 0.5]))
        if k % 10 == 0:
            sj.edge_te[0] = sj.edge_le[0]  # a zero-length edge is skipped
        if k % 7 == 0:
            sj.edge_le[1:3] = si.edge_le[0:2]  # shared edges
            sj.edge_te[1:3] = si.edge_te[0:2]
        ref = _relate_reference(si, sj)
        new = lattice_module._relate_surfaces(si, sj)
        assert new[0] == ref[0]
        if ref[1] is None:
            assert new[1] is None
        else:
            assert new[1] is not None
            assert np.allclose(new[1], ref[1], rtol=1e-12, atol=0.0)
        seen["joined" if ref[0] else ("near" if ref[1] else "none")] += 1
    assert min(seen.values()) > 0, seen  # every outcome is tested
