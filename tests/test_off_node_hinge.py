# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""A hinge that moves a chordwise node of a cambered vortex lattice.

On a cambered strip the legs of the horseshoes follow the mean surface
through the bound points of the strip (the leg chains of
``VortexLattice.leg_next``). A hinge that is not on a chordwise node moves
the nearest node onto the hinge on the control strips (and on the strips of
the joins at their side edges). The bound points of these strips are then at
other chord fractions than those of their neighbours, and at the edge
between them the two legs did not coincide. CL of a NACA 2412 wing with a
flap hinge at 0.7 chord and 4 chordwise panels went from 0.53 at 16
spanwise panels to 1.08 at 80 (0.35 to 0.60 at zero deflection). The legs
at such an edge now use one common node set: the union of the bound points
of the two strips on the mean surface.

A deflected edge between two different hinges, or a deflected control at a
surface joint, is an open slit; the flap increment does not converge with
the spanwise mesh there. Such an edge gets a trust note.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import ventorum as vt
from ventorum import gpu
from ventorum.aero import vortex
from ventorum.aero.system import build_sources, make_ground_plane, make_unknown_map
from ventorum.geometry import lattice as mesh_mod
from ventorum.geometry import lattice_cache

#: Chordwise panels [-]. With 4 uniform panels the nodes are at 0.25 chord
#: steps, so a hinge at 0.7 chord moves the node at 0.75 chord.
_N_CHORD = 4
#: Spanwise panel counts of the convergence checks [-].
_SPANS = (16, 24, 48, 80)
#: Largest CL spread over the spanwise panel counts, as a fraction of CL at
#: the finest mesh [-]. With the hinge on a node the spread is 0.85 % (flap
#: from 0 to 0.5 semi-span, 10 deg) and 0.06 % (zero deflection).
_SPREAD_TOL = 0.01

# CL of the code before the change at 16, 24, 48 and 80 spanwise panels:
# NACA 2412 wing, flap from 0 to 0.5 semi-span with the hinge on a node
# (0.75 chord), 10 deg, alpha 2 deg, CPU. rtol 1e-10 accepts other BLAS builds.
_ON_NODE_CL = (0.5812785870574789, 0.5860239408085428, 0.5862462400739522, 0.5860191892387581)
# CL, CDi, Cm, CY, Cl, Cn of the clean NACA 2412 wing (24 panels) before the change.
_CLEAN = (0.33640884030827095, 0.004677109685232716, -0.12815500732586582,
          4.028020043991498e-19, 9.486449403165572e-18, -1.6054059379433887e-20)


@pytest.fixture(autouse=True)
def _restore_backends():
    old_backend = vortex._kernel_backend
    old_device, old_precision = gpu.get_device(), gpu._precision
    gpu.set_device("cpu")
    lattice_cache.clear()
    yield
    lattice_cache.clear()
    vortex.set_kernel_backend(old_backend)
    gpu.set_device(old_device)
    gpu.set_precision(old_precision)


def _wing(controls=(), shape="2412") -> vt.LiftingSurface:
    return vt.LiftingSurface(
        name="w", semi_span=4.0,
        sections=[vt.WingSection(y_frac=e, chord=1.0, shape=shape) for e in (0.0, 1.0)],
        controls=list(controls),
    )


def _flap(hinge=0.7, deg=10.0, span=(0.0, 0.5)) -> vt.ControlSurface:
    return vt.ControlSurface("flap", span[0], span[1], hinge, np.radians(deg))


def _totals(wing, n_panels, n_chord=_N_CHORD, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return vt.VortexLatticeSolver().solve(
            vt.Aircraft(surfaces=[wing]), vt.FlightCondition(V_inf=30.0, alpha=np.radians(2.0)),
            vt.SolverSettings(n_panels=n_panels, n_chord=n_chord, **kw),
        ).totals


def _lattice(wing, n_panels=24, n_chord=_N_CHORD):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return mesh_mod.build_lattice(vt.Aircraft(surfaces=[wing]), vt.SolverSettings(n_panels=n_panels),
                                      n_chord=n_chord)


# --- convergence -----------------------------------------------------------------

@pytest.mark.parametrize("deg,span", [(10.0, (0.0, 0.5)), (0.0, (0.0, 0.5)), (0.0, (0.1, 0.4))])
def test_off_node_hinge_converges_spanwise(deg, span):
    """CL with a hinge off the chordwise nodes converges with the spanwise panels (formerly 0.63 to 0.96)."""
    cl = np.array([float(_totals(_wing([_flap(0.7, deg, span)]), ns).CL) for ns in _SPANS])
    spread = (cl.max() - cl.min()) / abs(cl[-1])
    assert spread < _SPREAD_TOL, cl
    # The hinge is aft of the hinge on a node, so the flap is larger: more lift.
    if deg != 0.0:
        on_node = np.array(_ON_NODE_CL)
        assert np.all(cl > on_node) and np.all(cl - on_node < 0.05), (cl, on_node)


def test_off_node_hinge_follows_on_node_hinge():
    """With a flap whose side edges are not strip edges, CL follows the CL of the hinge on a node."""
    span = (0.1, 0.4)
    off = np.array([float(_totals(_wing([_flap(0.7, 10.0, span)]), ns).CL) for ns in _SPANS])
    on = np.array([float(_totals(_wing([_flap(0.75, 10.0, span)]), ns).CL) for ns in _SPANS])
    # The spread of CL (2 %) comes from the flap side edges, which do not
    # fall on strip edges; it is the same for both hinges. Their difference
    # changes by less than 0.1 % of CL (formerly 0.53 to 1.08 against 0.48
    # to 0.49).
    diff = off - on
    assert np.ptp(diff) < 1e-3 * abs(off[-1]), (off, on)


def test_on_node_hinge_and_clean_wing_unchanged():
    """A hinge on a node and a clean wing give the results of the code before the change."""
    cl = [float(_totals(_wing([_flap(0.75)]), ns).CL) for ns in _SPANS]
    np.testing.assert_allclose(cl, _ON_NODE_CL, rtol=1e-10, atol=0.0)
    t = _totals(_wing(), 24)
    got = [float(getattr(t, k)) for k in ("CL", "CDi", "Cm", "CY", "Cl", "Cn")]
    np.testing.assert_allclose(got, _CLEAN, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("controls", [(), ("on_node",), ("on_node_0012",)])
def test_lattice_without_moved_node_is_bit_identical(controls, monkeypatch):
    """The common node set does not change a lattice without a moved hinge node, bit for bit."""
    shape = "0012" if "on_node_0012" in controls else "2412"
    wing = _wing([_flap(0.75)] if controls else [], shape)
    lat = _lattice(wing)
    assert "leg_mid_frac" not in lat.geom_cache
    monkeypatch.setattr(mesh_mod, "_common_leg_node_set", lambda *args, **kw: (None, None, None))
    lattice_cache.clear()
    ref = _lattice(wing)
    for name in ("a", "b", "a_te", "b_te", "cp", "normal_bc", "a_kink", "b_kink", "leg_next"):
        x, r = getattr(lat, name), getattr(ref, name)
        assert (x is None and r is None) or np.array_equal(x, r), name


# --- lattice -------------------------------------------------------------------

def _edge_segments(lat, s, side):
    """Return the leg segments [m] of strip *s* on its left (0) or right (1) edge, shape (k, 2, 3)."""
    n_c = lat.n_chord
    a_e, b_e = vortex.chain_ends(lat.a, lat.b, lat.a_te, lat.b_te, lat.leg_next)
    p = s * n_c + np.arange(n_c)
    if side == 0:
        pairs = list(zip(lat.a[p], lat.a_kink[p])) + list(zip(lat.a_kink[p], a_e[p]))
    else:
        pairs = list(zip(lat.b[p], lat.b_kink[p])) + list(zip(lat.b_kink[p], b_e[p]))
    # Order the two ends of each segment by x (aft end second).
    segs = [(u, v) if u[0] <= v[0] else (v, u) for u, v in pairs if np.linalg.norm(v - u) >= 1e-14]
    return np.array(segs)


def _same_segments(s1, s2, tol=1e-12):
    """Return True when the two segment sets are the same, each segment within *tol* [m]."""
    if s1.shape != s2.shape:
        return False
    used = np.zeros(len(s2), dtype=bool)
    for seg in s1:
        d = np.max(np.abs(s2 - seg[None]), axis=(1, 2))
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] > tol:
            return False
        used[j] = True
    return True


@pytest.mark.parametrize("deg", [0.0, 10.0])
def test_legs_coincide_at_moved_node_edges(deg):
    """At an edge between a strip with a moved node and one without, both legs run through the same nodes."""
    lat = _lattice(_wing([_flap(0.7, deg, (0.1, 0.4))]))
    n_c = lat.n_chord
    frac = lat.geom_cache["leg_mid_frac"]
    changed = np.flatnonzero(np.any(np.isfinite(frac.reshape(lat.n_strips, n_c, 2)), axis=(1, 2)))
    # Two side edges of the flap region on each wing half; each has a strip on both sides.
    assert changed.size == 8, changed
    n_checked = 0
    for s in range(lat.n_strips - 1):
        if lat.strip_surface[s] != lat.strip_surface[s + 1]:
            continue
        right, left = _edge_segments(lat, s, 1), _edge_segments(lat, s + 1, 0)
        # The leg of the strip that starts further aft lies on the leg of the other strip.
        x_r = float(lat.b[s * n_c, 0])
        x_l = float(lat.a[(s + 1) * n_c, 0])
        start = max(x_r, x_l) - 1e-9

        def aft(segs, x0=start):
            return segs[segs[:, 0, 0] >= x0]

        assert _same_segments(aft(right), aft(left)), f"edge {s}|{s + 1}"
        n_checked += 1
    assert n_checked == lat.n_strips - 1


def test_common_nodes_are_the_bound_points_of_both_strips():
    """The leg nodes at a moved-node edge are the bound points of both strips, on the mean surface."""
    lat = _lattice(_wing([_flap(0.7, 0.0, (0.1, 0.4))]))
    n_c = lat.n_chord
    off = lat.geom_cache["flap_field"]["hinge_off"]
    frac = lat.geom_cache["leg_mid_frac"]
    x0 = np.linspace(0.0, 1.0, n_c + 1)
    xb_clean = x0[:-1] + 0.25 * np.diff(x0)
    x1 = x0.copy()
    x1[3] = 0.7
    xb_moved = x1[:-1] + 0.25 * np.diff(x1)
    union = np.unique(np.concatenate([xb_clean, xb_moved]))
    for s in range(lat.n_strips):
        for side in (0, 1):
            f = frac[s * n_c:(s + 1) * n_c, side]
            if not np.any(np.isfinite(f)):
                continue
            assert off[s, side]
            # Each changed node is a bound point of the other strip, the
            # next bound point of its own strip (a straight leg) or the
            # trailing edge.
            for x in f[np.isfinite(f)]:
                assert np.min(np.abs(union - x)) < 1e-12 or x == 1.0, x


def test_moved_hinge_warning_text():
    """The warning about a moved hinge node says that the legs use a common node set."""
    lat = _lattice(_wing([_flap(0.7)]))
    msgs = [m for m in lat.mesh_warnings if "moves a chordwise node" in m]
    assert len(msgs) == 1
    assert "common node set" in msgs[0] and "converge" in msgs[0]
    # The example counts have a node at the hinge (0.7 chord) without a move.
    assert "n_chord = 10, 20, 30." in msgs[0], msgs[0]


# --- slit note -------------------------------------------------------------------

def _two_hinges(d1, d2):
    return [vt.ControlSurface("f1", 0.1, 0.4, 0.75, np.radians(d1)),
            vt.ControlSurface("f2", 0.4, 0.7, 0.625, np.radians(d2))]


def _slit_notes(notes):
    return [n for n in notes if "open slit" in n]


def test_slit_note_between_deflected_controls_with_different_hinges():
    """Two adjacent deflected controls with different hinges give a trust note that names their edge."""
    t = _totals(_wing(_two_hinges(10.0, 10.0), "0012"), 24, n_chord=8)
    notes = _slit_notes(t.trust.warnings)
    assert len(notes) == 1, t.trust.warnings
    note = notes[0]
    assert "'f1'" in note and "'f2'" in note and "does not converge" in note
    # The edge is at 0.4 semi-span on both wing halves.
    assert "y = -1.6, 1.6 m" in note, note


@pytest.mark.parametrize("controls", ["zero", "opposite", "single"])
def test_no_slit_note_on_sealed_edges(controls):
    """Undeflected edges and the sealed side edges of one control give no slit note."""
    ctrl = {"zero": _two_hinges(0.0, 0.0),
            # Opposite deflections give zero deflection at the common edge.
            "opposite": _two_hinges(10.0, -10.0),
            "single": [_flap(0.7, 10.0, (0.1, 0.4))]}[controls]
    # With 40 uniform spanwise panels the edge between the two controls is
    # at the middle of their join, where opposite deflections give zero.
    t = _totals(_wing(ctrl, "2412"), 40, n_chord=8, spacing="uniform")
    assert not _slit_notes(t.trust.warnings), t.trust.warnings


def test_slit_note_at_a_surface_joint():
    """A deflected flap at the end of a surface joined to a clean surface gives a slit note."""
    def surface(name, controls, symmetric):
        return vt.LiftingSurface(
            name=name, semi_span=2.0, is_symmetric=symmetric,
            sections=[vt.WingSection(y_frac=e, chord=1.0, shape="0012") for e in (0.0, 1.0)],
            controls=list(controls),
        )
    inner = surface("in", [vt.ControlSurface("flap", 0.5, 1.0, 0.75, np.radians(10.0))], True)
    outer = surface("out", [], False)
    outer.position = np.array([0.0, 2.0, 0.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        t = vt.VortexLatticeSolver().solve(
            vt.Aircraft(surfaces=[inner, outer, outer.mirrored()]),
            vt.FlightCondition(V_inf=30.0, alpha=np.radians(2.0)),
            vt.SolverSettings(n_panels=10, n_chord=8, spacing="uniform"),
        ).totals
    notes = _slit_notes(t.trust.warnings)
    assert notes and all("surface joint" in n for n in notes), t.trust.warnings
    assert any("'flap'" in n for n in notes)


# --- kernels and gradients ---------------------------------------------------------

def _off_node_sources(h=None, deg=10.0):
    """Return a lattice with a moved hinge node, its sources and evaluation points."""
    ac = vt.Aircraft(surfaces=[_wing([_flap(0.7, deg, (0.1, 0.4))])], ref_point=np.array([0.25, 0.0, 0.0]))
    lat = _lattice(ac.surfaces[0], n_panels=12)
    assert "leg_mid_frac" in lat.geom_cache
    ground = None if h is None else make_ground_plane(lat, h, 0.05, ref_point=ac.ref_point)
    wake = np.array([np.cos(0.05), 0.0, np.sin(0.05)])
    src = build_sources(lat, wake, make_unknown_map(lat, False), ground)
    a_e, _ = src.leg_ends()
    n = lat.n_panels
    points = np.vstack([lat.cp, 0.5 * (a_e[:n] + lat.a_kink), 0.5 * (lat.a + lat.b), lat.cp + [0.0, 0.0, 0.03]])
    return lat, src, points


def _cpu_backends():
    out = ["numba"]
    if vortex._HAVE_CYTHON and vortex._cy_has_chains():
        out.append("cython")
    if vortex._HAVE_TORCH:
        out.append("torch")
    return out


@pytest.mark.parametrize("h", [None, 0.2])
@pytest.mark.parametrize("backend", _cpu_backends())
def test_kernel_parity(h, backend):
    """Every compiled kernel agrees with the numpy reference on a lattice with a common node set."""
    lat, src, P = _off_node_sources(h)
    n = lat.n_panels
    normals = np.tile(lat.normal_bc, (4, 1))
    g = np.linspace(0.3, 1.7, n)

    def run():
        return (vortex.influence_matrix(P, normals, src, n),
                vortex.induced_velocity(P, src, g),
                vortex.horseshoe_velocity_tensor(P, src))

    vortex.set_kernel_backend("numpy")
    ref = run()
    vortex.set_kernel_backend(backend)
    for r, x in zip(ref, run()):
        scale = max(float(np.max(np.abs(r))), 1.0)
        np.testing.assert_allclose(x, r, rtol=0, atol=1e-12 * scale)


@pytest.mark.gpu
@pytest.mark.parametrize("h", [None, 0.2])
@pytest.mark.parametrize("precision,tol", [("float64", 1e-11), ("float32", 1e-5)])
def test_gpu_parity(h, precision, tol):
    """The GPU pipeline agrees with the CPU solve on a lattice with a common node set."""
    if not gpu.available():
        pytest.skip(gpu.unavailable_reason())
    wing = _wing([_flap(0.7, 10.0, (0.1, 0.4))])
    st = vt.SolverSettings(n_panels=12, n_chord=_N_CHORD)
    cond = vt.FlightCondition(V_inf=30.0, alpha=0.07, h=h)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ref = vt.VortexLatticeSolver().solve(wing, cond, st)
        gpu.set_device("gpu")
        gpu.set_precision(precision)
        res = vt.VortexLatticeSolver().solve(wing, cond, st)
    assert res.details["device"] == "gpu"
    for name in ("CL", "CDi", "Cm", "CY", "Cl", "Cn"):
        a, b = getattr(ref.totals, name), getattr(res.totals, name)
        assert abs(a - b) <= max(tol * abs(a), 1e-13), name


@pytest.mark.parametrize("wrt", [("alpha", "controls"), ("alpha", "nodes")])
def test_gradients(wrt):
    """The adjoint gradients of a lattice with a common node set pass check_gradients."""
    gpu.set_precision("float64")
    ac = vt.Aircraft(surfaces=[_wing([vt.ControlSurface("flap", 0.25, 0.75, 0.7, 0.1, symmetric=False)], "4412")],
                     ref_point=np.array([0.25, 0, 0]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lat = mesh_mod.build_lattice(ac, vt.SolverSettings(n_panels=8), n_chord=4)
        assert "leg_mid_frac" in lat.geom_cache
        checked = vt.check_gradients(ac, vt.FlightCondition(V_inf=30.0, alpha=0.05),
                                     vt.SolverSettings(n_panels=8, n_chord=4), outputs=("CL", "CDi", "Cm", "Cl"),
                                     wrt=wrt)
    assert checked["pass"], checked
