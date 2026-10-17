# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests of the 2D inviscid panel solver with ground effect and several bodies.

1. Closed-form panel influences (stream function and velocity) against
   composite Gauss-Legendre quadrature.
2. Exact Karman-Trefftz and Joukowski conformal-mapping solutions, with the
   convergence order of the largest Cp error.
3. Free-air superposition and prescribed source strengths, against an
   independent dense assembly of the source influence by quadrature (free
   air and ground effect).
4. Ground effect by images against an explicit mirror body; several bodies.
5. numpy and torch backends (CPU and CUDA).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from ventorum.airfoil import airfoil
from ventorum.panel2d import solver as sv
from ventorum.panel2d.influence import (
    image_source_stream_influence,
    linear_vortex_stream_influence,
    linear_vortex_velocity_influence,
    source_panel_stream_influence,
    source_panel_velocity_influence,
    wake_source_stream_influence,
)
from ventorum.panel2d.solver import Panel2DResult, solve_inviscid
from ventorum.reference.karman_trefftz import karman_trefftz_airfoil, karman_trefftz_solution

_X64, _W64 = np.polynomial.legendre.leggauss(64)


def _quad(f, a: float, b: float, breaks=(), n_sub: int = 32) -> float:
    """Return the integral of f on [a, b] by composite 64-point Gauss quadrature.

    The interval is split at the given break points; each part is divided
    into ``n_sub`` subintervals graded (cubically) toward both of its ends.
    """
    pts = sorted({a, b} | {float(x) for x in breaks if a < x < b})
    total = 0.0
    m = n_sub // 2
    g = 0.5 * (np.arange(m + 1) / m) ** 3
    unit = np.unique(np.concatenate([g, 1.0 - g]))
    for lo, hi in zip(pts[:-1], pts[1:]):
        edges = lo + (hi - lo) * unit
        for s0, s1 in zip(edges[:-1], edges[1:]):
            t = 0.5 * (s1 - s0) * _X64 + 0.5 * (s0 + s1)
            total += 0.5 * (s1 - s0) * float(np.sum(_W64 * f(t)))
    return total


def _angle_body(eta, u):
    """Return the polar angle on the branch (-pi/2, 3pi/2] (cut along -eta)."""
    a = np.arctan2(eta + 0.0, u)
    return np.where(a <= -0.5 * np.pi, a + 2.0 * np.pi, a)


def _angle_wake(eta, u):
    """Return the polar angle on the branch [0, 2pi) (cut along +xi)."""
    a = np.arctan2(eta + 0.0, u)
    return np.where(a < 0.0, a + 2.0 * np.pi, a)


def _field_points(length):
    """Return local field points from 0.1 to 5 panel lengths from the panel middle."""
    pts = []
    for d in (0.1, 0.25, 0.5, 1.0, 2.0, 5.0):
        for ang in (0.1, 0.5, 1.2, 2.3, 3.8, 4.4, 5.1):
            pts.append((0.5 * length + d * length * math.cos(ang), d * length * math.sin(ang)))
    return pts


def _check_point(xi, eta, L):
    """Compare every closed-form influence at one field point with quadrature."""

    def r2(t):
        return (xi - t) ** 2 + eta**2

    def quad(f):
        return _quad(f, 0.0, L, [xi])

    p1, p2 = linear_vortex_stream_influence(xi, eta, L)
    q1 = quad(lambda t: (1.0 - t / L) * np.log(r2(t)) / (4 * np.pi))
    q2 = quad(lambda t: (t / L) * np.log(r2(t)) / (4 * np.pi))
    assert abs(p1 - q1) < 1e-10 * max(abs(q1), 1e-3)
    assert abs(p2 - q2) < 1e-10 * max(abs(q2), 1e-3)
    ps = source_panel_stream_influence(xi, eta, L)
    qs = quad(lambda t: _angle_body(eta, xi - t) / (2 * np.pi))
    assert abs(ps - qs) < 1e-10 * max(abs(qs), 1e-3), (xi, eta, ps, qs)
    pw = wake_source_stream_influence(xi, eta, L)
    qw = quad(lambda t: _angle_wake(eta, xi - t) / (2 * np.pi))
    assert abs(pw - qw) < 1e-10 * max(abs(qw), 1e-3), (xi, eta, pw, qw)
    u1, v1, u2, v2 = linear_vortex_velocity_influence(xi, eta, L)
    qu1 = quad(lambda t: (1 - t / L) * eta / r2(t) / (2 * np.pi))
    qu2 = quad(lambda t: (t / L) * eta / r2(t) / (2 * np.pi))
    qv1 = quad(lambda t: -(1 - t / L) * (xi - t) / r2(t) / (2 * np.pi))
    qv2 = quad(lambda t: -(t / L) * (xi - t) / r2(t) / (2 * np.pi))
    for a, b in ((u1, qu1), (u2, qu2), (v1, qv1), (v2, qv2)):
        assert abs(a - b) < 1e-10 * max(abs(b), 1e-3)
    us, vs = source_panel_velocity_influence(xi, eta, L)
    qus = quad(lambda t: (xi - t) / r2(t) / (2 * np.pi))
    qvs = quad(lambda t: eta / r2(t) / (2 * np.pi))
    assert abs(us - qus) < 1e-10 * max(abs(qus), 1e-3)
    assert abs(vs - qvs) < 1e-10 * max(abs(qvs), 1e-3)


def test_closed_form_influences_against_gauss_quadrature():
    """Closed-form panel influences agree with quadrature to 1e-10 relative."""
    L = 1.3
    for xi, eta in _field_points(L):
        _check_point(xi, eta, L)


def test_image_source_influence_against_quadrature():
    """Source panels below the ground: closed form against quadrature of the global angle."""
    ps = np.array([[0.2, -0.3], [0.9, -0.1], [1.4, -0.45]])
    pe = np.array([[0.7, -0.5], [0.4, -0.2], [1.2, -0.05]])
    field = np.array([[0.3, 0.05], [1.5, 0.4], [-0.2, 0.2], [0.8, 0.01]])
    got = image_source_stream_influence(field, ps, pe)
    for i, q in enumerate(field):
        for j in range(len(ps)):
            d = pe[j] - ps[j]
            length = float(np.hypot(*d))

            def f(t, j=j, q=q, d=d, length=length):
                px = ps[j, 0] + d[0] * t / length
                py = ps[j, 1] + d[1] * t / length
                return np.arctan2(q[1] - py, q[0] - px) / (2 * np.pi)

            ref = _quad(f, 0.0, length)
            assert abs(got[i, j] - ref) < 1e-10 * max(abs(ref), 1e-3)


def test_exact_karman_trefftz_checks():
    """Karman-Trefftz airfoil: Cl, pressure integration and Cp convergence.

    Cl within 0.5 % of exact at 200 panels (alpha 0, 4, 8 deg); Cl and
    Cl_pressure within 1 %; free-air Cd_pressure below 1e-3; the largest Cp
    error decreases from 100 to 200 to 400 panels with an observed order of
    at least 1. The Cp error is taken over all nodes except the trailing-edge
    node: at a sharp trailing edge of angle tau the exact speed is zero but it
    rises as r^(tau / (2 pi - tau)) (here r^0.029) away from the edge, so no
    node value converges to it.
    """
    center = (-0.07, 0.04)
    tau = 10.0
    coords = karman_trefftz_airfoil(center, tau, n_points=201)
    res = solve_inviscid(coords, [0.0, 4.0, 8.0])
    assert isinstance(res, Panel2DResult)
    for i, a in enumerate((0.0, 4.0, 8.0)):
        _, cl_ex = karman_trefftz_solution(center, tau, n_points=201, alpha_deg=a)
        assert abs(res.Cl[i] / cl_ex - 1.0) < 0.005
        assert abs(res.Cl_pressure[i] / res.Cl[i] - 1.0) < 0.01
        assert abs(res.Cd_pressure[i]) < 1e-3
    errs = []
    for n in (100, 200, 400):
        c = karman_trefftz_airfoil(center, tau, n_points=n + 1)
        cp_ex, _ = karman_trefftz_solution(center, tau, n_points=n + 1, alpha_deg=4.0)
        r = solve_inviscid(c, 4.0)
        errs.append(float(np.max(np.abs(r.Cp[1:-1] - cp_ex[1:-1]))))
    o1 = math.log(errs[0] / errs[1]) / math.log(2.0)
    o2 = math.log(errs[1] / errs[2]) / math.log(2.0)
    print(f"\n[panel2d] Cp error orders (all nodes but the trailing edge): {o1:.2f}, {o2:.2f}")
    assert errs[2] < errs[1] < errs[0]
    assert o1 >= 1.0 and o2 >= 1.0


def test_symmetric_joukowski_zero_lift():
    """Symmetric Joukowski section at alpha 0: |Cl| below 1e-10."""
    coords = karman_trefftz_airfoil((-0.1, 0.0), 0.0, n_points=101)
    assert abs(solve_inviscid(coords, 0.0).Cl) < 1e-10


def test_free_air_superposition():
    """Superposed result at alpha 5 deg equals a direct solve to 1e-12."""
    coords = karman_trefftz_airfoil((-0.07, 0.04), 10.0, n_points=101)
    multi = solve_inviscid(coords, [0.0, 5.0, 10.0])
    direct = solve_inviscid(coords, 5.0)
    assert abs(multi.Cl[1] - direct.Cl) < 1e-12
    assert abs(multi.Cm[1] - direct.Cm) < 1e-12
    assert np.allclose(multi.Cp[1], direct.Cp, atol=1e-12, rtol=0.0)


def _scratch_source_psi(nodes, layout, ground):
    """Return the source stream-function matrix assembled by quadrature in this test.

    Real panels: polar angle with the cut along the outer normal of the panel
    (branch (-pi/2, 3pi/2] in the panel frame); field points on a panel line
    are taken on the inner side. Mirror panels (ground effect): global polar
    angle in (0, pi).
    """
    nn = layout.n_nodes
    out = np.zeros((nn, layout.n_panels))
    for j in range(layout.n_panels):
        a = nodes[layout.idx1[j]]
        b = nodes[layout.idx2[j]]
        d = b - a
        length = float(np.hypot(*d))
        tx, ty = d / length
        for i in range(nn):
            r = nodes[i] - a
            xi = r[0] * tx + r[1] * ty
            eta = -r[0] * ty + r[1] * tx
            if abs(eta) < 1e-12 * length:
                eta = 0.0
            val = _quad(lambda t, xi=xi, eta=eta: _angle_body(eta, xi - t), 0.0, length, [xi], n_sub=8)
            if ground:
                q = nodes[i]

                def g(t, a=a, d=d, length=length, q=q):
                    px = a[0] + d[0] * t / length
                    py = -(a[1] + d[1] * t / length)
                    return np.arctan2(q[1] - py, q[0] - px)

                val += _quad(g, 0.0, length, n_sub=8)
            out[i, j] = val / (2.0 * np.pi)
    return out


@pytest.mark.parametrize("ground", [False, True])
def test_source_panels_against_scratch_assembly(ground):
    """Prescribed sources equal an independent quadrature assembly to 1e-10.

    Zero sources give the inviscid result bit for bit. The sharp-edged
    Karman-Trefftz section and the blunt NACA 0012 (with its trailing-edge
    panel) are both checked; ground effect at h/c 0.3, alpha 4 deg.
    """
    shapes = [karman_trefftz_airfoil((-0.07, 0.04), 10.0, n_points=61),
              airfoil("0012").repanel(n_panels=60).coords]
    for coords in shapes:
        layout = sv.panel_layout([coords])
        kw = {"height": 0.3} if ground else {}
        base = solve_inviscid(coords, 4.0, **kw)
        zero = solve_inviscid(coords, 4.0, sigma=np.zeros(layout.n_panels), **kw)
        assert np.array_equal(base.gamma, zero.gamma) and base.Cl == zero.Cl
        p1, p2 = sv.panel_ends(coords, layout)
        xm = 0.5 * (p1[:, 0] + p2[:, 0])
        ym = 0.5 * (p1[:, 1] + p2[:, 1])
        sigma = 0.02 * np.sin(np.pi * xm) + 0.005 + 0.2 * np.maximum(ym, 0.0)
        res = solve_inviscid(coords, 4.0, sigma=sigma, **kw)
        nodes = np.asarray(base.coordinates)
        psi = _scratch_source_psi(nodes, layout, ground)
        A = sv.system_matrix(nodes, layout, ground=ground)
        dg = np.linalg.solve(A, sv.full_rhs(-psi @ sigma, layout))[: layout.n_nodes]
        err = np.max(np.abs((res.gamma - base.gamma) - dg))
        assert err < 1e-10, err
        assert abs(res.Cl - base.Cl) > 1e-4


def test_uniform_source_on_circle():
    """A uniform source sheet on a circle does not change the surface speed.

    The flow of a uniform source sheet on a circle is radial outside the
    circle, so the tangential speed (gamma) stays that of the inviscid flow;
    the test checks this to the discretization level.
    """
    th = np.linspace(0.0, 2.0 * np.pi, 201)
    circle = np.column_stack([0.5 + 0.5 * np.cos(th), 0.5 * np.sin(th)])
    layout = sv.panel_layout([circle])
    base = solve_inviscid(circle, 0.0)
    res = solve_inviscid(circle, 0.0, sigma=np.full(layout.n_panels, 0.1))
    assert np.max(np.abs(res.gamma - base.gamma)) < 1e-3
    assert abs(res.Cl) < 1e-10


def test_ground_effect_properties():
    """Ground effect: explicit mirror body, large height, suction, under-ground error."""
    coords = karman_trefftz_airfoil((-0.1, 0.0), 0.0, n_points=61)
    res_ge = solve_inviscid(coords, 4.0, height=0.3)
    rot = np.asarray(res_ge.coordinates)
    mirror = rot.copy()
    mirror[:, 1] *= -1.0
    two = solve_inviscid([rot, mirror], 0.0)
    assert abs(res_ge.Cl - 2.0 * two.circulation[0]) < 1e-10
    fa = solve_inviscid(coords, 0.0)
    assert abs(solve_inviscid(coords, 0.0, height=1000.0).Cl - fa.Cl) < 1e-4
    assert solve_inviscid(coords, 0.0, height=0.1).Cl < 0.0
    with pytest.raises(ValueError, match="touching the ground"):
        solve_inviscid(coords, 10.0, height=0.01)


def test_multi_element_well_separated():
    """Two sections 20 chords apart give each its free-air Cl within 1e-3."""
    c0 = karman_trefftz_airfoil((-0.1, 0.0), 0.0, n_points=51)
    c1 = c0.copy()
    c1[:, 1] += 20.0
    single = solve_inviscid(c0, 4.0).Cl
    two = solve_inviscid([c0, c1], 4.0)
    assert abs(2.0 * two.circulation[0] - single) < 1e-3
    assert abs(2.0 * two.circulation[1] - single) < 1e-3


def test_multi_element_sources_in_outer_strip():
    """Sources with a body in the outer strip of another body's panel raise ValueError."""
    main = airfoil("0012").repanel(n_panels=40).coords
    flap = 0.3 * main.copy()
    flap[:, 0] += 0.5
    flap[:, 1] -= 0.08
    layout = sv.panel_layout([main, flap])
    with pytest.raises(ValueError, match="outer strip"):
        solve_inviscid([main, flap], 2.0, sigma=np.full(layout.n_panels, 0.01))


def test_backends_parity_and_accuracy():
    """torch float64 equals numpy to 1e-10; torch float32 within 1e-4 on Cl."""
    coords = karman_trefftz_airfoil((-0.07, 0.04), 10.0, n_points=81)
    alphas = [0.0, 4.0, 8.0]
    rn = solve_inviscid(coords, alphas)
    r64 = solve_inviscid(coords, alphas, backend="torch", device="cpu", dtype=torch.float64)
    assert np.allclose(rn.Cl, r64.Cl.numpy(), atol=1e-10, rtol=0.0)
    assert np.allclose(rn.Cm, r64.Cm.numpy(), atol=1e-10, rtol=0.0)
    assert np.allclose(rn.Cp, r64.Cp.numpy(), atol=1e-10, rtol=0.0)
    r32 = solve_inviscid(coords, alphas, backend="torch", device="cpu", dtype=torch.float32)
    assert np.max(np.abs(rn.Cl - r32.Cl.double().numpy())) < 1e-4
    ge_n = solve_inviscid(coords, alphas, height=0.4)
    ge_t = solve_inviscid(coords, alphas, height=0.4, backend="torch", dtype=torch.float64)
    assert np.allclose(ge_n.Cl, ge_t.Cl.numpy(), atol=1e-10, rtol=0.0)


@pytest.mark.gpu
def test_backends_gpu():
    """CUDA float64 to 1e-10 and float32 to 1e-4 on Cl, for a 33-angle ground-effect batch."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    coords = karman_trefftz_airfoil((-0.07, 0.04), 10.0, n_points=61)
    alphas = np.linspace(-2.0, 10.0, 33)
    ref = solve_inviscid(coords, alphas, height=0.4)
    c64 = solve_inviscid(coords, alphas, height=0.4, backend="torch", device="cuda", dtype=torch.float64)
    assert np.max(np.abs(c64.Cl.cpu().numpy() - ref.Cl)) < 1e-10
    c32 = solve_inviscid(coords, alphas, height=0.4, backend="torch", device="cuda", dtype=torch.float32)
    err32 = np.max(np.abs(c32.Cl.double().cpu().numpy() - ref.Cl))
    print(f"\n[panel2d] CUDA float32 largest Cl difference: {err32:.2e}")
    assert err32 < 1e-4
    f32 = solve_inviscid(coords, alphas, backend="torch", device="cuda", dtype=torch.float32)
    fref = solve_inviscid(coords, alphas)
    assert np.max(np.abs(f32.Cl.double().cpu().numpy() - fref.Cl)) < 1e-4


def _former_defaults(c):
    """Return the former default reference values: chord, moment point (x_min + c/4, 0) and pivot."""
    chord = sv.section_chord(c)
    point = (float(np.min(c[:, 0])) + 0.25 * chord, 0.0)
    return {"chord_ref": chord, "moment_ref": point, "pivot": point}


@pytest.mark.parametrize("code", ["0012", "2412"])
def test_default_reference_keeps_unit_sections_to_round_off(code):
    """Unflapped sections with the chord line on y = 0: the defaults give the former results to round-off.

    NACA 0012 and 2412 as generated have the leading edge (the point farthest from the trailing-edge
    midpoint) at (0, 0) and the trailing-edge midpoint at (1, 0). The default reference chord (their
    distance) is the former default 1, and the default moment point and pivot (the quarter-chord point on
    the chord line) are the former (x_min + c/4, 0) = (0.25, 0). Free air and ground effect; the array
    path and the shape path give the same bits. The derived chord can differ from 1 by one unit in the
    last place, depending on the math library of the platform, so the comparison with the former
    defaults is to round-off, not bit for bit.
    """
    shape = airfoil(code, n_points=161)
    c = shape.coords
    former = _former_defaults(c)
    assert abs(former["chord_ref"] - 1.0) < 1e-15
    assert abs(former["moment_ref"][0] - 0.25) < 1e-15 and former["moment_ref"][1] == 0.0
    point = sv.reference_point(c)
    assert point == sv.reference_point(shape)
    assert abs(point[0] - 0.25) < 1e-15 and abs(point[1]) < 1e-15
    for kw in ({}, {"height": 0.5}):
        new = solve_inviscid(c, [-2.0, 0.0, 4.0], **kw)
        via_shape = solve_inviscid(shape, [-2.0, 0.0, 4.0], **kw)
        old = solve_inviscid(c, [-2.0, 0.0, 4.0], **former, **kw)
        for name in ("Cl", "Cl_pressure", "Cd_pressure", "Cm"):
            assert np.allclose(getattr(new, name), getattr(old, name), rtol=0.0, atol=1e-12), (kw, name)
            assert np.array_equal(getattr(via_shape, name), getattr(new, name)), (kw, name)


def test_default_reference_of_a_repaneled_section_is_on_its_chord_line():
    """Repaneled NACA 0012: the chord line is y = 0 to round-off only, so the results agree to round-off.

    The spline repaneling puts the leading-edge node at y = 2e-16, so the quarter-chord point on the
    chord line has y of the same size (before: exactly 0). The coefficients agree with the former
    defaults to 1e-12 and are the same bits as with the quarter-chord point of the chord line given.
    """
    c = airfoil("0012").repanel(n_panels=160).coords
    ref = sv.section_reference(c)
    assert sv.section_chord(c) == ref.chord_ref
    assert abs(ref.chord_ref - 1.0) < 1e-15
    assert abs(float(np.max(c[:, 0]) - np.min(c[:, 0])) - 1.0) < 1e-15
    point = sv.reference_point(c)
    assert point == ref.quarter_chord
    assert abs(point[0] - 0.25) < 1e-15 and abs(point[1]) < 1e-15
    for kw in ({}, {"height": 0.5}):
        new = solve_inviscid(c, [0.0, 4.0], **kw)
        given = solve_inviscid(c, [0.0, 4.0], chord_ref=ref.chord_ref, moment_ref=point, pivot=point, **kw)
        old = solve_inviscid(c, [0.0, 4.0], **_former_defaults(c), **kw)
        for name in ("Cl", "Cl_pressure", "Cd_pressure", "Cm"):
            assert np.array_equal(getattr(new, name), getattr(given, name)), (kw, name)
            assert np.allclose(getattr(new, name), getattr(old, name), rtol=0.0, atol=1e-12), (kw, name)


def test_flapped_shape_gives_the_undeflected_reference_without_user_input():
    """A flapped AirfoilShape gives the chord and the quarter-chord point of the undeflected section.

    The solve of the flapped shape equals, bit for bit, the solve of its point array with the reference
    chord and the moment point of the clean section given; in ground effect the pivot is that point too.
    The point array alone uses the shorter, tilted chord line of the deflected outline.
    """
    clean = airfoil("2412", n_points=121)
    flapped = clean.with_flap(x_hinge=0.75, deflection=10.0)
    ref = clean.reference_geometry()
    chord, point, pivot = sv.resolve_reference(flapped)
    assert chord == ref.chord_ref and abs(chord - 1.0) < 1e-15
    assert point == pivot == ref.quarter_chord
    assert abs(point[0] - 0.25) < 1e-15 and abs(point[1]) < 1e-15
    for kw in ({}, {"height": 0.6}):
        via_shape = solve_inviscid(flapped, [0.0, 3.0], **kw)
        given = solve_inviscid(flapped.coords, [0.0, 3.0], chord_ref=ref.chord_ref, moment_ref=ref.quarter_chord,
                               pivot=ref.quarter_chord, **kw)
        for name in ("Cl", "Cl_pressure", "Cd_pressure", "Cm"):
            assert np.array_equal(getattr(via_shape, name), getattr(given, name)), (kw, name)
    own = sv.section_reference(flapped.coords)
    assert own.chord_ref < 0.996 and own.quarter_chord[1] < -0.005
    array_only = solve_inviscid(flapped.coords, 0.0)
    shape_res = solve_inviscid(flapped, 0.0)
    assert abs(array_only.Cl * own.chord_ref - shape_res.Cl * ref.chord_ref) < 1e-12


def test_explicit_reference_values_win_over_everything():
    """chord_ref and moment_ref win over the stored reference geometry; x_ref is an alias of moment_ref."""
    flapped = airfoil("0012", n_points=121).with_flap(x_hinge=0.7, deflection=8.0)
    chord, point = 0.8, (0.4, 0.05)
    via_shape = solve_inviscid(flapped, [0.0, 4.0], chord_ref=chord, moment_ref=point)
    via_array = solve_inviscid(flapped.coords, [0.0, 4.0], chord_ref=chord, moment_ref=point)
    alias = solve_inviscid(flapped, [0.0, 4.0], chord_ref=chord, x_ref=point)
    both = solve_inviscid(flapped, [0.0, 4.0], chord_ref=chord, x_ref=point, moment_ref=point)
    for name in ("Cl", "Cl_pressure", "Cd_pressure", "Cm"):
        assert np.array_equal(getattr(via_shape, name), getattr(via_array, name)), name
        assert np.array_equal(getattr(alias, name), getattr(via_shape, name)), name
        assert np.array_equal(getattr(both, name), getattr(via_shape, name)), name
    default = solve_inviscid(flapped, [0.0, 4.0])
    assert np.allclose(via_shape.Cl * chord, default.Cl * 1.0, rtol=0.0, atol=1e-12)
    assert sv.resolve_reference(flapped, chord, point) == (chord, point, (0.25, 0.0))
    with pytest.raises(ValueError, match="differ"):
        solve_inviscid(flapped, 0.0, x_ref=(0.25, 0.0), moment_ref=point)
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="chord_ref"):
            solve_inviscid(flapped, 0.0, chord_ref=bad)
    for bad in ((0.25,), (0.25, float("inf")), "quarter"):
        with pytest.raises(ValueError, match="moment_ref"):
            solve_inviscid(flapped, 0.0, moment_ref=bad)


@pytest.mark.parametrize("height", [None, 0.4])
def test_moment_transfer_between_two_points(height):
    """Cm about two points obeys the 2D moment transfer rule to 1e-12 (inviscid, free air and ground effect).

    With the body-frame pressure force coefficients Cn = Cl_p cos(alpha) + Cd_p sin(alpha) (along +y)
    and Ca = Cd_p cos(alpha) - Cl_p sin(alpha) (along +x), the nose-up moment about b is
    Cm_b = Cm_a + (x_b - x_a) / c Cn - (y_b - y_a) / c Ca. At alpha = 0 and y_b = y_a this is
    Cm_b = Cm_a + (x_b - x_a) / c Cl_pressure.
    """
    shape = airfoil("2412", n_points=121)
    alphas = np.array([-3.0, 0.0, 2.0, 6.0])
    kw = {} if height is None else {"height": height}
    pa, pb = (0.25, 0.0), (0.6, -0.15)
    ra = solve_inviscid(shape, alphas, moment_ref=pa, **kw)
    rb = solve_inviscid(shape, alphas, moment_ref=pb, **kw)
    for name in ("Cl", "Cl_pressure", "Cd_pressure"):
        assert np.array_equal(getattr(ra, name), getattr(rb, name)), name
    a = np.radians(alphas)
    cn = ra.Cl_pressure * np.cos(a) + ra.Cd_pressure * np.sin(a)
    ca = ra.Cd_pressure * np.cos(a) - ra.Cl_pressure * np.sin(a)
    expected = ra.Cm + (pb[0] - pa[0]) * cn - (pb[1] - pa[1]) * ca
    assert np.max(np.abs(rb.Cm - expected)) < 1e-12
    r0 = solve_inviscid(shape, 0.0, moment_ref=(0.7, 0.0), **kw)
    ra0 = solve_inviscid(shape, 0.0, **kw)
    assert abs(r0.Cm - (ra0.Cm + 0.45 * ra0.Cl_pressure)) < 1e-12


def test_section_shifted_in_y_gets_the_moment_point_on_its_chord_line():
    """A section moved by (0.3, 0.2) has its default moment point and pivot at (0.55, 0.2).

    Its coefficients equal those of the unmoved section to 1e-12, in free air and in ground effect (the
    pivot is on the chord line, so the same height places the same geometry). The former point
    (x_min + c/4, 0) lay 0.2 below the chord line.
    """
    shape = airfoil("2412", n_points=121)
    moved = shape.coords + np.array([0.3, 0.2])
    point = sv.reference_point(moved)
    assert point == pytest.approx((0.55, 0.2), abs=1e-15)
    assert sv.resolve_reference(moved)[2] == point
    for kw in ({}, {"height": 0.5}):
        base = solve_inviscid(shape.coords, [0.0, 5.0], **kw)
        res = solve_inviscid(moved, [0.0, 5.0], **kw)
        for name in ("Cl", "Cl_pressure", "Cd_pressure", "Cm"):
            assert np.allclose(getattr(res, name), getattr(base, name), rtol=0.0, atol=1e-12), (kw, name)
    # The former point is 0.2 below the chord line: about it, Cm differs by 0.2 Ca at 5 deg.
    former = solve_inviscid(moved, 5.0, moment_ref=(0.55, 0.0))
    assert abs(former.Cm - solve_inviscid(moved, 5.0).Cm) > 1e-3


def test_coefficients_do_not_depend_on_scale_and_position():
    """A section of chord 2 m that starts at x = 3 m has the coefficients of the unit section (1e-12).

    Before, the moment point was (0.25, 0) m for every section.
    """
    c = airfoil("2412").repanel(n_panels=120).coords
    big = 2.0 * c + np.array([3.0, 0.0])
    assert abs(sv.section_chord(big) - 2.0 * sv.section_chord(c)) < 1e-14
    unit = solve_inviscid(c, [0.0, 4.0])
    scaled = solve_inviscid(big, [0.0, 4.0])
    for name in ("Cl", "Cl_pressure", "Cd_pressure", "Cm"):
        assert np.allclose(getattr(scaled, name), getattr(unit, name), rtol=0.0, atol=1e-12), name


def test_flap_against_glauert_on_the_original_chord():
    """Plain flap: lift and moment per unit deflection against thin-airfoil theory on the undeflected chord.

    Thin-airfoil theory (Glauert, R and M 1095, 1927) for a flap hinged at x_h on the chord c, with
    x_h / c = (1 - cos theta_h) / 2: the camber slope is 0 ahead of the hinge and -delta on the flap, so
    A_n = 2 delta sin(n theta_h) / (n pi) and A_0 = alpha + delta (pi - theta_h) / pi. Then

        Cl = 2 pi alpha + 2 (pi - theta_h + sin theta_h) delta,
        Cm_c/4 = pi (A_2 - A_1) / 4 = -sin(theta_h) (1 - cos theta_h) delta / 2,

    with the coefficients on the chord c of the undeflected section. The panel result is divided by the
    thickness factor of its own lift slope (Cl_alpha / 2 pi of the clean section; 1.05 for 6 % thickness).
    NACA 0006, 160 panels, hinge at 0.75 c, delta = 2 deg: measured deviations 0.6 % (lift) and 0.1 %
    (moment); limit 1 %. The deflected section is shorter than the reference chord: with its own chord as
    the reference, Cl is larger by the chord ratio. The flapped shape holds the undeflected reference
    geometry, so it meets the same limits without the chord given.
    """
    x_h = 0.75
    delta = math.radians(2.0)
    theta_h = math.acos(1.0 - 2.0 * x_h)
    cl_glauert = 2.0 * (math.pi - theta_h + math.sin(theta_h)) * delta
    cm_glauert = -0.5 * math.sin(theta_h) * (1.0 - math.cos(theta_h)) * delta

    shape = airfoil("0006").repanel(n_panels=160)
    chord = shape.chord
    clean = solve_inviscid(shape.coords, [0.0, 1.0])
    thickness_factor = (clean.Cl[1] - clean.Cl[0]) / math.radians(1.0) / (2.0 * math.pi)
    flapped = shape.with_flap(x_hinge=x_h, deflection=math.degrees(delta)).coords
    res = solve_inviscid(flapped, 0.0, chord_ref=chord)
    assert abs(res.Cl / thickness_factor / cl_glauert - 1.0) < 0.01
    assert abs(res.Cm / thickness_factor / cm_glauert - 1.0) < 0.01
    own = solve_inviscid(flapped, 0.0)
    assert sv.section_chord(flapped) < chord
    assert abs(own.Cl * sv.section_chord(flapped) - res.Cl * chord) < 1e-12
    # The flapped shape gives the undeflected chord and its quarter-chord point without user input.
    via_shape = solve_inviscid(shape.with_flap(x_hinge=x_h, deflection=math.degrees(delta)), 0.0)
    assert via_shape.Cl == res.Cl
    assert abs(via_shape.Cl / thickness_factor / cl_glauert - 1.0) < 0.01
    assert abs(via_shape.Cm / thickness_factor / cm_glauert - 1.0) < 0.01
