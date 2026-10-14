# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for the wake-ray pair merge of the vortex kernels.

In the vortex-lattice lattice every chordwise panel of one strip ends at
the same two trailing-edge points, and every wake ray leaves in the same
wake direction. The panels of one strip therefore share their pair of
semi-infinite wake rays when they also share the core radius and the core
group (bit for bit). The helper ``ventorum.aero.vortex._wake_pair_groups``
finds these shared pairs, and :func:`ventorum.aero.vortex.induced_velocity`
evaluates each shared pair once with the summed circulation. The assembly
keeps the unmerged path. The results must equal the unmerged code to
round-off.

The gradient code (``ventorum.sensitivity``) has its own kernels and does
not use the merged source sets: one test proves this by blocking the merge
and running a gradient check.
"""

from __future__ import annotations

import numpy as np
import pytest

import ventorum as vt
from ventorum.aero import vortex as V
from ventorum.aero.system import UnknownMap, build_sources, make_ground_plane, panel_targets
from ventorum.geometry.lattice import build_lattice


def _backends():
    out = ["numpy", "numba"]
    if V._HAVE_CYTHON:
        out.append("cython")
    return out


@pytest.fixture(params=_backends())
def backend(request, monkeypatch):
    old = V._kernel_backend
    V.set_kernel_backend(request.param)
    monkeypatch.setattr(V, "_WAKE_MERGE_ENABLED", True)
    yield request.param
    V._kernel_backend = old


def _rect_wing():
    return vt.LiftingSurface(
        name="wing", semi_span=4.0,
        sections=[vt.WingSection(y_frac=0.0, chord=1.2),
                  vt.WingSection(y_frac=1.0, chord=0.6)])


def _wing_tail():
    wing = _rect_wing()
    tail = vt.LiftingSurface(
        name="tail", semi_span=1.2, position=np.array([3.5, 0.0, 0.2]),
        sections=[vt.WingSection(y_frac=0.0, chord=0.5),
                  vt.WingSection(y_frac=1.0, chord=0.4)])
    return vt.Aircraft(surfaces=[wing, tail])


def _flapped_wing():
    wing = vt.LiftingSurface(
        name="wing", semi_span=4.0,
        controls=[vt.ControlSurface(name="aileron", eta_start=0.6, eta_end=1.0,
                                    hinge_x_c=0.75, deflection=np.radians(5.0),
                                    symmetric=False)],
        sections=[vt.WingSection(y_frac=0.0, chord=1.2),
                  vt.WingSection(y_frac=1.0, chord=0.6)])
    return vt.Aircraft(surfaces=[wing])


def _lattice(ac, n_panels=16, n_chord=4, **kwargs):
    st = vt.SolverSettings(solver_type="vlm", n_panels=n_panels, n_chord=n_chord)
    return build_lattice(ac, st, collocation="vlm", n_chord=n_chord, **kwargs)


def _sources(lat):
    n = lat.n_panels
    umap = UnknownMap(np.arange(n), np.arange(n), False)
    return build_sources(lat, np.array([1.0, 0.0, 0.0]), umap, None)


def test_pair_counts_and_bit_equal_keys():
    """Measurement: rays, unique pairs, and bit equality of the keys."""
    for n_panels, n_chord in ((20, 4), (40, 4), (80, 8)):
        lat = _lattice(vt.Aircraft(surfaces=[_rect_wing()]), n_panels, n_chord)
        n = lat.n_panels
        assert n == lat.n_strips * n_chord
        hs = _sources(lat)
        rep, inverse = V._wake_pair_groups(hs)
        # Every strip merges its chordwise panels into one pair.
        assert rep.n == lat.n_strips
        assert np.bincount(inverse).tolist() == [n_chord] * lat.n_strips
        # The keys of one strip are equal bit for bit.
        for s in range(lat.n_strips):
            idx = np.arange(s * n_chord, (s + 1) * n_chord)
            assert np.array_equal(hs.a_te[idx], hs.a_te[idx[:1]].repeat(n_chord, axis=0))
            assert np.array_equal(hs.b_te[idx], hs.b_te[idx[:1]].repeat(n_chord, axis=0))
            assert np.array_equal(hs.rc[idx], hs.rc[idx[:1]].repeat(n_chord))
            assert np.array_equal(hs.group[idx], hs.group[idx[:1]].repeat(n_chord))
    lat = _lattice(_wing_tail(), 40, 4)
    rep, inverse = V._wake_pair_groups(_sources(lat))
    assert rep.n == lat.n_strips
    assert sorted(np.bincount(inverse).tolist()) == [4] * lat.n_strips


def test_no_merge_with_one_chordwise_panel():
    """A lifting-line lattice (one chordwise panel) has no shared pair."""
    ac = vt.Aircraft(surfaces=[_rect_wing()])
    st = vt.SolverSettings(solver_type="linear", n_panels=12)
    lat = build_lattice(ac, st, collocation="llt")
    n = lat.n_panels
    umap = UnknownMap(np.arange(n), np.arange(n), False)
    hs = build_sources(lat, np.array([1.0, 0.0, 0.0]), umap, None)
    assert V._wake_pair_groups(hs) is None
    lat1 = _lattice(ac, 8, 1)
    assert V._wake_pair_groups(_sources(lat1)) is None


def test_partial_merge_with_cosine_chord_spacing():
    """Cosine spacing changes the core radius per panel: only equal keys merge."""
    lat = _lattice(vt.Aircraft(surfaces=[_rect_wing()]), 8, 4, chord_spacing="cosine")
    hs = _sources(lat)
    rep, inverse = V._wake_pair_groups(hs)
    assert rep.n > lat.n_strips and rep.n < lat.n_panels
    # Every merged group has bit-equal keys (here the core radius).
    members = [np.flatnonzero(inverse == g) for g in range(rep.n)]
    assert any(len(j) > 1 for j in members)
    for j in members:
        if len(j) > 1:
            assert np.array_equal(hs.rc[j], hs.rc[j[:1]].repeat(len(j)))
    # The representative is the first source of its pair.
    seen = np.full(rep.n, -1)
    for j, g in enumerate(inverse):
        if seen[g] < 0:
            seen[g] = j
    assert np.array_equal(hs.a_te[seen], rep.a_te)
    assert np.array_equal(hs.b_te[seen], rep.b_te)


def test_images_merge_only_with_matching_keys():
    """Ground images form their own pairs; real and image rays never merge."""
    lat = _lattice(vt.Aircraft(surfaces=[_rect_wing()]), 8, 4)
    ground = make_ground_plane(lat, 0.8, np.radians(4.0), 0.0, height_ref="min")
    n = lat.n_panels
    umap = UnknownMap(np.arange(n), np.arange(n), False)
    hs = build_sources(lat, np.array([1.0, 0.0, 0.0]), umap, ground)
    rep, inverse = V._wake_pair_groups(hs)
    assert rep.n == 2 * lat.n_strips
    assert inverse[:n].max() < lat.n_strips
    assert inverse[n:].min() >= lat.n_strips


def _assert_same_totals(new, ref):
    for name in ("CL", "CDi", "CDi_nearfield", "Cm"):
        assert getattr(new.totals, name) == pytest.approx(getattr(ref.totals, name), rel=1e-12)
    for name in ("CY", "Cl", "Cn"):
        assert abs(getattr(new.totals, name) - getattr(ref.totals, name)) < 1e-14


def _assert_same_fields(new, ref):
    g_new = np.asarray(new.details["gamma"], dtype=float)
    g_ref = np.asarray(ref.details["gamma"], dtype=float)
    scale = max(float(np.max(np.abs(g_ref))), 1e-300)
    assert np.max(np.abs(g_new - g_ref)) <= 1e-12 * scale
    f_new = np.asarray(new.details["loads"].strip_force, dtype=float)
    f_ref = np.asarray(ref.details["loads"].strip_force, dtype=float)
    fscale = max(float(np.max(np.abs(f_ref))), 1e-300)
    assert np.max(np.abs(f_new - f_ref)) <= 1e-12 * fscale
    for s_new, s_ref in zip(new.spanwise, ref.spanwise):
        assert np.max(np.abs(s_new.gamma - s_ref.gamma)) <= 1e-12 * scale


def _parity_cases():
    beta = np.radians(3.0)
    out = []
    out.append(("sym", vt.Aircraft(surfaces=[_rect_wing()]),
                vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0)),
                vt.SolverSettings(solver_type="vlm", n_panels=12, n_chord=4)))
    out.append(("sideslip", vt.Aircraft(surfaces=[_rect_wing()]),
                vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=beta),
                vt.SolverSettings(solver_type="vlm", n_panels=12, n_chord=4)))
    acg = vt.Aircraft(surfaces=[_rect_wing()])
    acg.compute_reference_values()
    out.append(("ground", acg,
                vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=beta,
                                   h=0.5 * float(acg.c_ref)),
                vt.SolverSettings(solver_type="vlm", n_panels=12, n_chord=4)))
    out.append(("wingtail", _wing_tail(),
                vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=beta),
                vt.SolverSettings(solver_type="vlm", n_panels=10, n_chord=2)))
    out.append(("controls", _flapped_wing(),
                vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=beta),
                vt.SolverSettings(solver_type="vlm", n_panels=12, n_chord=4)))
    out.append(("rates", vt.Aircraft(surfaces=[_rect_wing()]),
                vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=beta,
                                   p=0.1, q=-0.2, r=0.3),
                vt.SolverSettings(solver_type="vlm", n_panels=12, n_chord=4)))
    out.append(("cosine", vt.Aircraft(surfaces=[_rect_wing()]),
                vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=beta),
                vt.SolverSettings(solver_type="vlm", n_panels=12, n_chord=4,
                                  chord_spacing="cosine")))
    out.append(("linear", vt.Aircraft(surfaces=[_rect_wing()]),
                vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=beta),
                vt.SolverSettings(solver_type="linear", n_panels=12)))
    return out


@pytest.mark.parametrize("case", [c[0] for c in _parity_cases()])
def test_merged_loads_equal_unmerged(backend, monkeypatch, case):
    """Merged wake pairs give the same result as the current code."""
    _, ac, cond, st = next(c for c in _parity_cases() if c[0] == case)
    monkeypatch.setattr(V, "_WAKE_MERGE_ENABLED", False)
    ref = vt.analyze(ac, condition=cond, settings=st)
    monkeypatch.setattr(V, "_WAKE_MERGE_ENABLED", True)
    new = vt.analyze(ac, condition=cond, settings=st)
    _assert_same_totals(new, ref)
    _assert_same_fields(new, ref)


def test_merged_sweep_equal_unmerged(backend, monkeypatch):
    """A sweep (kernel cache in the assembly and in the loads) is unchanged."""
    ac = vt.Aircraft(surfaces=[_rect_wing()])
    st = vt.SolverSettings(solver_type="vlm", n_panels=12, n_chord=4)
    alphas = np.radians(np.array([-4.0, 0.0, 4.0, 8.0]))
    monkeypatch.setattr(V, "_WAKE_MERGE_ENABLED", False)
    ref = vt.analyze_sweep(ac, alphas, settings=st)
    monkeypatch.setattr(V, "_WAKE_MERGE_ENABLED", True)
    new = vt.analyze_sweep(ac, alphas, settings=st)
    for a, b in zip(new, ref):
        _assert_same_totals(a, b)
        _assert_same_fields(a, b)


def test_nonlinear_lifting_line_unchanged(backend, monkeypatch):
    """The nonlinear lifting line keeps its result (the merge is exact there)."""
    ac = vt.Aircraft(surfaces=[_rect_wing()])
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(6.0), beta=np.radians(2.0))
    st = vt.SolverSettings(solver_type="nonlinear", n_panels=12)
    monkeypatch.setattr(V, "_WAKE_MERGE_ENABLED", False)
    ref = vt.analyze(ac, condition=cond, settings=st)
    monkeypatch.setattr(V, "_WAKE_MERGE_ENABLED", True)
    new = vt.analyze(ac, condition=cond, settings=st)
    _assert_same_totals(new, ref)
    _assert_same_fields(new, ref)


def test_matrix_parity_against_unmerged(backend, monkeypatch):
    """The influence matrix with merged pairs equals the current matrix."""
    lat = _lattice(vt.Aircraft(surfaces=[_rect_wing()]), 8, 4)
    n = lat.n_panels
    hs = _sources(lat)
    tg = panel_targets(lat, np.arange(n))
    monkeypatch.setattr(V, "_WAKE_MERGE_ENABLED", False)
    ref = V.influence_matrix(lat.cp, lat.normal_bc, hs, n, tg)
    monkeypatch.setattr(V, "_WAKE_MERGE_ENABLED", True)
    new = V.influence_matrix(lat.cp, lat.normal_bc, hs, n, tg)
    scale = max(float(np.max(np.abs(ref))), 1e-300)
    assert np.max(np.abs(new - ref)) <= 1e-12 * scale


def test_gradient_path_keeps_unmerged_sets(monkeypatch):
    """The sensitivity code does not use the merged source sets.

    The gradient path (``ventorum.sensitivity``) has its own kernels: no
    module there imports ``ventorum.aero.vortex``. The gradient check below
    still runs the production forward solves (which use the merge), so it
    proves that the parity of ``check_gradients`` is unchanged: both runs
    pass, with the same adjoint gradients.
    """
    import pathlib
    import re

    sens_dir = pathlib.Path(__file__).resolve().parent.parent / "ventorum" / "sensitivity"
    for path in sorted(sens_dir.glob("*.py")):
        for line in path.read_text(encoding="utf-8").splitlines():
            code = line.split("#")[0]
            if code.startswith(("import ", "from ")):
                assert re.search(r"aero\.vortex(?!_)", code) is None, (path.name, line)
                assert "_wake_pair_groups" not in code, (path.name, line)

    from ventorum.sensitivity.check import check_gradients

    ac = vt.Aircraft(surfaces=[_rect_wing()])
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    st = vt.SolverSettings(solver_type="vlm", n_panels=8, n_chord=2)
    grads = {}
    for enabled in (False, True):
        monkeypatch.setattr(V, "_WAKE_MERGE_ENABLED", enabled)
        out = check_gradients(ac, cond, st, outputs=("CL",), wrt=("alpha",),
                              n_directions=1, method="auto")
        assert out["pass"], out["reasons"]
        grads[enabled] = {k: v["grad"] for k, v in out["details"].items() if v.get("compared")}
    assert grads[True].keys() == grads[False].keys() and grads[True]
    for key in grads[True]:
        assert grads[True][key] == pytest.approx(grads[False][key], rel=1e-9)
