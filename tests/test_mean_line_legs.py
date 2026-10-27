# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Horseshoe legs that follow the mean surface of a cambered vortex lattice.

On a lattice on the section mean line, a straight leg from a bound point
on the curved mean line to the trailing edge leaves the mean surface. With
narrow strips it passes very close to control points of other strips, and
the solve is near-singular: the default settings gave CL 7.74 at alpha 0
for a NACA 6409 wing (about 0.50 on wider strips) and 0.078 for a NACA
4412 wing (about 0.33). The legs now run from the bound point through the
bound points of the panels aft of it on the same strip edge to the
trailing edge (the node chain of ``VortexLattice.leg_next``), with one
node on the mean surface between each two bound points.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import ventorum as vt
from ventorum import gpu
from ventorum.aero import vortex
from ventorum.aero.system import build_sources, make_ground_plane, make_unknown_map
from ventorum.core.trust import _MEAN_LINE_STRIP_JUMP, mean_line_strip_jump
from ventorum.geometry import lattice as mesh_mod
from ventorum.geometry import lattice_cache


def _wing(shape, *, tip_shape=None, controls=None, semi_span=4.0) -> vt.LiftingSurface:
    return vt.LiftingSurface(
        name="w", semi_span=semi_span,
        sections=[vt.WingSection(y_frac=0.0, chord=1.0, shape=shape),
                  vt.WingSection(y_frac=1.0, chord=1.0, shape=tip_shape or shape)],
        controls=controls or [],
    )


def _solve(wing, settings, *, alpha=0.0, h=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return vt.VortexLatticeSolver().solve(wing, vt.FlightCondition(V_inf=30.0, alpha=alpha, h=h), settings)


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


# --- lattice -------------------------------------------------------------------

def _mean_point(lat, s, side, x):
    """Return the point [m] at chord fraction *x* on the mean line of the left (0) or right (1) edge of strip *s*."""
    frames = lat.geom_cache["camber_frames"]
    fr = frames[0]
    e = s + side
    return fr["le"][e] + x * (fr["te"][e] - fr["le"][e]) + mesh_mod._joined_camber_offset(frames, 0, e, np.array(x))


def test_chain_nodes_on_mean_surface():
    """Each leg of a cambered strip passes through a mean-surface node and the bound points aft of it."""
    st = vt.SolverSettings(n_panels=8, n_chord=5, chord_spacing="uniform")
    lat = mesh_mod.build_lattice(vt.Aircraft(surfaces=[_wing("6409")]), st, n_chord=5)
    n_c = lat.n_chord
    assert lat.leg_next is not None
    a_e, b_e = vortex.chain_ends(lat.a, lat.b, lat.a_te, lat.b_te, lat.leg_next)
    for p in range(lat.n_panels):
        k = p % n_c
        last = k == n_c - 1
        assert bool(lat.leg_next[p]) is not last
        # The node at the middle chord fraction between the two bound points;
        # on the last panel half a panel length aft of its bound point (the
        # same fraction with uniform chordwise panels).
        xm = (k + 0.75) / n_c
        np.testing.assert_allclose(lat.a_kink[p], _mean_point(lat, p // n_c, 0, xm), rtol=0, atol=1e-14)
        np.testing.assert_allclose(lat.b_kink[p], _mean_point(lat, p // n_c, 1, xm), rtol=0, atol=1e-14)
        if last:
            np.testing.assert_array_equal(a_e[p], lat.a_te[p])
            np.testing.assert_array_equal(b_e[p], lat.b_te[p])
        else:
            np.testing.assert_array_equal(a_e[p], lat.a[p + 1])
            np.testing.assert_array_equal(b_e[p], lat.b[p + 1])


@pytest.mark.parametrize("shape", [None, "0012"])
def test_flat_strips_keep_straight_legs(shape):
    """A strip without a mean line, or with a zero mean line, keeps its straight legs (no chain, no node)."""
    lat = mesh_mod.build_lattice(vt.Aircraft(surfaces=[_wing(shape)]), vt.SolverSettings(n_panels=8), n_chord=4)
    assert lat.leg_next is None
    assert lat.a_kink is None


def test_flap_hinge_is_one_node_of_the_chain():
    """On a cambered flap strip the hinge point is the node between the two bound points around it."""
    flap = vt.ControlSurface("flap", 0.0, 1.0, 0.7, np.radians(10.0))
    st = vt.SolverSettings(n_panels=6, n_chord=6, chord_spacing="uniform")
    lat = mesh_mod.build_lattice(vt.Aircraft(surfaces=[_wing("2412", controls=[flap])]), st, n_chord=6)
    hinges = lat.geom_cache["flap_hinges"]
    n_c = lat.n_chord
    for s in range(lat.n_strips):
        # Chord fractions of the bound points (the mesh puts a panel edge at the hinge).
        d_xi = lat.panel_length[s * n_c:(s + 1) * n_c] / lat.chord[s]
        xb = np.concatenate(([0.0], np.cumsum(d_xi)[:-1])) + 0.25 * d_xi
        for k in range(n_c - 1):
            p = s * n_c + k
            assert lat.leg_next[p]
            if xb[k] <= 0.7 < xb[k + 1]:
                np.testing.assert_array_equal(lat.a_kink[p], hinges[s][0])
                np.testing.assert_array_equal(lat.b_kink[p], hinges[s][1])
                continue
            # Elsewhere the node is the mean-surface node; aft of the hinge
            # it turns with the flap (same distance to the hinge point).
            xm = 0.5 * (xb[k] + xb[k + 1])
            node = _mean_point(lat, s, 0, xm)
            if xm < 0.7:
                np.testing.assert_allclose(lat.a_kink[p], node, rtol=0, atol=1e-14)
            else:
                assert np.linalg.norm(lat.a_kink[p] - node) > 1e-3
                np.testing.assert_allclose(np.linalg.norm(lat.a_kink[p] - hinges[s][0]),
                                           np.linalg.norm(node - hinges[s][0]), rtol=1e-12)


# --- solutions -----------------------------------------------------------------

@pytest.mark.parametrize("shape", ["6409", "4412"])
def test_default_settings_strong_camber(shape):
    """The default settings agree with a converged mesh within 2 % (formerly CL 7.74 and 0.078)."""
    wing = _wing(shape)
    default = _solve(wing, vt.SolverSettings())
    fine = _solve(wing, vt.SolverSettings(n_panels=40, n_chord=16, spacing="cosine"))
    assert abs(default.totals.CL - fine.totals.CL) <= 0.02 * abs(fine.totals.CL)
    assert np.max(np.abs(default.details["gamma"])) < 10.0


def _refinement(sizes, settings_of):
    cl, jump = [], []
    for n in sizes:
        res = _solve(_wing("6409"), settings_of(n))
        cl.append(res.totals.CL)
        jump.append(mean_line_strip_jump(res.details["lattice"], res.spanwise))
    return np.array(cl), np.array(jump)


def _assert_converges(cl, jump):
    """Check smooth strip loads and a band of 2 % about the finest level.

    The steps of the converged values are 1e-5 to 3e-3 and not monotone
    (also with the former straight legs on wide strips), so a rule on the
    ratio of two steps tests round-off and small mesh effects, not the
    convergence. The band test checks that every level is within 2 % of the
    finest level.
    """
    assert np.all(jump < _MEAN_LINE_STRIP_JUMP), jump
    assert np.all(np.abs(cl - cl[-1]) <= 0.02 * abs(cl[-1])), cl
    assert np.all(np.abs(cl - 0.50) < 0.02), cl


def test_spanwise_refinement():
    """16, 40, 80, 160 strips (cosine) at 8 chordwise panels converge with smooth strip loads."""
    # Band test (all levels within 2 % of the finest), not a step-ratio rule: see _assert_converges.
    cl, jump = _refinement((16, 40, 80, 160), lambda n: vt.SolverSettings(n_panels=n, n_chord=8, spacing="cosine"))
    _assert_converges(cl, jump)


def test_chordwise_refinement():
    """4, 8, 16, 32 chordwise panels on 80 strips (cosine) converge with smooth strip loads."""
    # Band test (all levels within 2 % of the finest), not a step-ratio rule: see _assert_converges.
    cl, jump = _refinement((4, 8, 16, 32), lambda n: vt.SolverSettings(n_panels=80, n_chord=n, spacing="cosine"))
    _assert_converges(cl, jump)


@pytest.mark.slow
@pytest.mark.parametrize("n_chord", [4, 16, 32])
def test_spanwise_refinement_grid(n_chord):
    """The spanwise refinement converges at every chordwise count, also 160 strips (formerly singular)."""
    # Band test (all levels within 2 % of the finest), not a step-ratio rule: see _assert_converges.
    cl, jump = _refinement((16, 40, 80, 160),
                           lambda n: vt.SolverSettings(n_panels=n, n_chord=n_chord, spacing="cosine"))
    _assert_converges(cl, jump)


@pytest.mark.slow
@pytest.mark.parametrize("n_span", [16, 160])
def test_chordwise_refinement_grid(n_span):
    """The chordwise refinement converges on coarse and on narrow strips."""
    # Band test (all levels within 2 % of the finest), not a step-ratio rule: see _assert_converges.
    cl, jump = _refinement((4, 8, 16, 32),
                           lambda n: vt.SolverSettings(n_panels=n_span, n_chord=n, spacing="cosine"))
    _assert_converges(cl, jump)


# --- kernels -------------------------------------------------------------------

def _chained_sources(h=None, flap=False):
    """Return a cambered lattice (optionally with a deflected flap), its sources and evaluation points."""
    controls = [vt.ControlSurface("flap", 0.25, 0.75, 0.7, 0.2, symmetric=False)] if flap else None
    ac = vt.Aircraft(surfaces=[_wing("4412", tip_shape="6409", controls=controls)],
                     ref_point=np.array([0.25, 0.0, 0.0]))
    st = vt.SolverSettings(n_panels=6, n_chord=5, chord_spacing="uniform")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lat = mesh_mod.build_lattice(ac, st, n_chord=5)
    ground = None if h is None else make_ground_plane(lat, h, 0.05, ref_point=ac.ref_point)
    wake = np.array([np.cos(0.05), 0.0, np.sin(0.05)])
    src = build_sources(lat, wake, make_unknown_map(lat, False), ground)
    a_e, _ = src.leg_ends()
    n = lat.n_panels
    points = np.vstack([lat.cp, 0.5 * (a_e[:n] + lat.a_kink), 0.5 * (lat.a + lat.b), lat.cp + [0.0, 0.0, 0.03]])
    return lat, src, points


def test_chain_matches_explicit_segments():
    """A chained horseshoe equals the straight horseshoe with its legs replaced by the segments of its chain.

    Each leg of the chain runs bound point, mean-surface node, next bound point;
    the last leg runs bound point, mean-surface node, trailing edge.
    """
    lat, src, P = _chained_sources()
    vortex.set_kernel_backend("numpy")
    v = vortex.horseshoe_velocity_tensor(P, src)
    seg = vortex.segment_velocity
    n_c = lat.n_chord
    for j in (0, 2, n_c - 1, n_c + 1):
        s0 = j - j % n_c
        expected = (seg(P, src.a[j:j + 1], src.b[j:j + 1], src.rc[j:j + 1])
                    + vortex.ray_velocity(P, src.b_te[j:j + 1], src.wake_dir[j:j + 1], src.rc[j:j + 1])
                    - vortex.ray_velocity(P, src.a_te[j:j + 1], src.wake_dir[j:j + 1], src.rc[j:j + 1]))
        for q in range(j, s0 + n_c):
            end_a = src.a[q + 1:q + 2] if q < s0 + n_c - 1 else src.a_te[q:q + 1]
            end_b = src.b[q + 1:q + 2] if q < s0 + n_c - 1 else src.b_te[q:q + 1]
            rc = src.rc[q:q + 1]
            # The last panel has a node between its bound point and the trailing edge.
            ka, kb = lat.a_kink[q:q + 1], lat.b_kink[q:q + 1]
            expected = (expected + seg(P, ka, src.a[q:q + 1], rc) + seg(P, end_a, ka, rc)
                        + seg(P, src.b[q:q + 1], kb, rc) + seg(P, kb, end_b, rc))
        np.testing.assert_allclose(v[:, j:j + 1], expected, rtol=0, atol=1e-11 * np.max(np.abs(expected)))


def _cpu_backends():
    out = ["numba"]
    if vortex._HAVE_CYTHON and vortex._cy_has_chains():
        out.append("cython")
    if vortex._HAVE_TORCH:
        out.append("torch")
    return out


@pytest.mark.parametrize("flap", [False, True])
@pytest.mark.parametrize("h", [None, 0.2])
@pytest.mark.parametrize("backend", _cpu_backends())
def test_chain_kernel_parity(h, flap, backend):
    """Every compiled kernel with leg chains agrees with the numpy reference to round-off."""
    lat, src, P = _chained_sources(h, flap)
    assert src.has_chains
    n = lat.n_panels
    normals = np.tile(lat.normal_bc, (4, 1))
    g = np.linspace(0.3, 1.7, n)
    tg = vortex.Targets(group=np.zeros(P.shape[0], dtype=np.int64), rc=np.full(P.shape[0], 0.01))

    def run():
        return (vortex.influence_matrix(P, normals, src, n),
                vortex.influence_matrix(P, normals, src, n, cache={}, cache_key="k"),
                vortex.induced_velocity(P, src, g),
                vortex.induced_velocity(P, src, g, cache={}, cache_key="k"),
                vortex.induced_velocity(P, src, g, tg),
                vortex.horseshoe_velocity_tensor(P, src),
                vortex.velocity_tensor_unknowns(P, src, n))

    vortex.set_kernel_backend("numpy")
    ref = run()
    vortex.set_kernel_backend(backend)
    for r, x in zip(ref, run()):
        scale = max(float(np.max(np.abs(r))), 1.0)
        np.testing.assert_allclose(x, r, rtol=0, atol=1e-12 * scale)


@pytest.mark.parametrize("h", [None, 0.2])
def test_solver_backend_parity(h):
    """The numpy and Numba solves of a cambered wing with a deflected flap agree to round-off."""
    wing = _wing("4412", tip_shape="6409", controls=[vt.ControlSurface("flap", 0.25, 0.75, 0.7, 0.15,
                                                                       symmetric=False)])
    st = vt.SolverSettings(n_panels=12, n_chord=5)
    vortex.set_kernel_backend("numpy")
    ref = _solve(wing, st, alpha=0.05, h=h)
    vortex.set_kernel_backend("numba")
    res = _solve(wing, st, alpha=0.05, h=h)
    np.testing.assert_allclose(res.details["gamma"], ref.details["gamma"], rtol=1e-11, atol=1e-13)
    for name in ("CL", "CDi", "Cm", "CY", "Cl", "Cn"):
        np.testing.assert_allclose(getattr(res.totals, name), getattr(ref.totals, name), rtol=1e-11, atol=1e-13)


@pytest.mark.gpu
@pytest.mark.parametrize("flap", [False, True])
@pytest.mark.parametrize("h", [None, 0.2])
@pytest.mark.parametrize("precision,tol", [("float64", 1e-11), ("float32", 1e-5)])
def test_gpu_parity(h, flap, precision, tol):
    """The GPU pipeline with leg chains agrees with the CPU solve."""
    if not gpu.available():
        pytest.skip(gpu.unavailable_reason())
    controls = [vt.ControlSurface("flap", 0.25, 0.75, 0.7, 0.15, symmetric=False)] if flap else None
    wing = _wing("4412", tip_shape="6409", controls=controls)
    st = vt.SolverSettings(n_panels=12, n_chord=5)
    ref = _solve(wing, st, alpha=0.07, h=h)
    gpu.set_device("gpu")
    gpu.set_precision(precision)
    res = _solve(wing, st, alpha=0.07, h=h)
    assert res.details["device"] == "gpu"
    for name in ("CL", "CDi", "Cm", "CY", "Cl", "Cn"):
        a, b = getattr(ref.totals, name), getattr(res.totals, name)
        assert abs(a - b) <= max(tol * abs(a), 1e-13), name


# --- loads and gradients ------------------------------------------------------

def test_leg_strength_is_chain_sum():
    """The own leg segments of a panel carry the sum of the circulations before them on the chain."""
    from ventorum.aero.loads import leg_strength

    lat = mesh_mod.build_lattice(vt.Aircraft(surfaces=[_wing("6409")]), vt.SolverSettings(n_panels=4), n_chord=4)
    g = np.arange(1.0, lat.n_panels + 1.0)
    expected = np.cumsum(g.reshape(lat.n_strips, lat.n_chord), axis=1).ravel()
    np.testing.assert_array_equal(leg_strength(lat, g), expected)


@pytest.mark.parametrize("case", ["cambered", "span_varying", "flap"])
@pytest.mark.parametrize("h", [None, 0.3])
def test_gradients(case, h):
    """The adjoint gradients of a lattice with leg chains pass check_gradients at the default tolerance."""
    gpu.set_precision("float64")
    controls = None
    tip = None
    if case == "span_varying":
        tip = "6409"
    if case == "flap":
        controls = [vt.ControlSurface("flap", 0.25, 0.75, 0.7, 0.1, symmetric=False)]
    ac = vt.Aircraft(surfaces=[_wing("4412", tip_shape=tip, controls=controls)], ref_point=np.array([0.25, 0, 0]))
    wrt = ("alpha", "controls") if case == "flap" else ("alpha", "nodes")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        checked = vt.check_gradients(ac, vt.FlightCondition(V_inf=30.0, alpha=0.05, h=h),
                                     vt.SolverSettings(n_panels=4, n_chord=4), outputs=("CL", "CDi", "Cm", "Cl"),
                                     wrt=wrt)
    assert checked["pass"], checked
