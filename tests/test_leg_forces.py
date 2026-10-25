# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Kutta-Joukowski force on the on-wing legs of the vortex lattice.

The rule depends on the edge of the leg (``ventorum.aero.loads.leg_force_rules``).
On a shared edge the surface is impermeable, so the leg force uses only the
velocity tangent to the strip surface: the normal velocity at the leg points
is the leak of the discrete lattice between its control points. On a kink
edge the coincident legs carry no force. On a free edge (a tip) the full
velocity acts: its in-plane force is the side-edge force. The checks compare
the near field with far-field (Trefftz-plane) values and with measured data:

* the side force of an aileron on a flat wing: only the free tip edges give
  an in-plane force (the in-plane leak force of the shared edges gave
  CY_delta about +0.025/rad at alpha 3 deg),
* the yawing moment of an aileron against the moment of the Trefftz-plane
  induced drag, without the yawing moment of the side-edge force (the
  Trefftz plane has no chordwise position),
* the side force of a dihedral wing in sideslip against the far-field force,
* the rolling moment due to sideslip of flat unswept wings against NACA
  TN 1669 (A. Goodman and J. D. Brewer, 1948, figure 9).
"""

from __future__ import annotations

import numpy as np
import pytest

import ventorum as vt
from ventorum.aero import loads as L
from ventorum.aero.system import freestream_direction, lift_direction
from ventorum.core.axes import transform_moments


def _wing(semi_span=4.0, dihedral_deg=0.0, aileron_deg=None):
    controls = []
    if aileron_deg is not None:
        controls.append(vt.ControlSurface(name="aileron", eta_start=0.6, eta_end=1.0, hinge_x_c=0.75,
                                          deflection=np.radians(aileron_deg), symmetric=False))
    wing = vt.LiftingSurface(
        name="wing", semi_span=semi_span, dihedral=np.radians(dihedral_deg), controls=controls,
        sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=1.0)])
    ac = vt.Aircraft(name="a", surfaces=[wing], ref_point=np.array([0.25, 0.0, 0.0]))
    ac.compute_reference_values()
    return ac


def _solve(ac, alpha_deg, beta_deg=0.0, n_panels=16, n_chord=4):
    fc = vt.FlightCondition(V_inf=30.0, alpha=np.radians(alpha_deg), beta=np.radians(beta_deg))
    st = vt.SolverSettings(solver_type="vlm", n_panels=n_panels, n_chord=n_chord, spacing="cosine")
    return vt.analyze(ac, fc, st), fc


def _stability_cn(res, alpha_deg, beta_deg=0.0):
    t = res.totals
    return float(transform_moments(t.Cl, t.Cm, t.Cn, np.radians(alpha_deg), np.radians(beta_deg),
                                   "stability", b_ref=float(res.b_ref), c_ref=float(res.c_ref))["Cn"])


def _trefftz_yaw(res, ac, fc):
    """Yawing moment of the Trefftz-plane induced-drag distribution (drag on the right wing: nose right)."""
    lat = res.details["lattice"]
    g = res.details["gamma"]
    strip_gamma = g.reshape(lat.n_strips, lat.n_chord).sum(axis=1)
    wd = np.asarray(res.details["wake_dir"], dtype=float)[None, :]
    _, w_n, seg_len = L.trefftz_induced_drag_batch(lat, strip_gamma[None, :], wd, np.array([fc.rho]), [None])
    drag = -0.5 * fc.rho * strip_gamma * w_n[0] * seg_len
    q = 0.5 * fc.rho * fc.V_inf ** 2
    return float(np.sum(lat.qc_mid[:, 1] * drag)) / (q * ac.S_ref * ac.b_ref)


def _free_edge_loads(res, fc):
    """Force [N] and moment [N m] (geometry axes, about the reference point) of the legs on free edges."""
    lat = res.details["lattice"]
    g = np.asarray(res.details["gamma"], dtype=float)
    n_p = lat.n_panels
    _, _, kind = L.leg_force_rules(lat)
    # The leg segments of the loads: two sets, or four with the hinge nodes
    # of a chordwise control (see L.leg_segments).
    start, end = L.leg_segments(lat)
    n_sets = start.shape[0] // n_p
    mid = 0.5 * (start + end)
    pts = [lat.force_points] + list(mid.reshape(n_sets, n_p, 3))
    vel = [np.empty((n_p, 3)) for _ in pts]
    wd = np.asarray(res.details["wake_dir"], dtype=float)
    swap = [0, 2, 1] if n_sets == 2 else [0, 2, 1, 4, 3]
    L._near_field_velocity(lat, g, fc, res.details["ground"], wd, pts, swap, True, vel)
    v_leg = fc.V_inf * freestream_direction(fc.alpha, fc.beta) + np.vstack(vel[1:])
    seg = end - start
    free = kind == L.LEG_FREE
    f = fc.rho * np.concatenate([g] * n_sets)[free, None] * np.cross(v_leg[free], seg[free])
    rp = res.details["ref_point"]
    rp = np.zeros(3) if rp is None else rp
    return f.sum(axis=0), np.cross(mid[free] - rp, f).sum(axis=0)


def _shared_leg_and_bound_forces(res, fc):
    """Force [N] of the shared-edge legs (tangential rule) and of the bound vortices."""
    lat = res.details["lattice"]
    g = np.asarray(res.details["gamma"], dtype=float)
    n_p = lat.n_panels
    normal, scale, kind = L.leg_force_rules(lat)
    start, end = L.leg_segments(lat)
    n_sets = start.shape[0] // n_p
    pts = [lat.force_points] + list((0.5 * (start + end)).reshape(n_sets, n_p, 3))
    vel = [np.empty((n_p, 3)) for _ in pts]
    swap = [0, 2, 1] if n_sets == 2 else [0, 2, 1, 4, 3]
    L._near_field_velocity(lat, g, fc, res.details["ground"], np.asarray(res.details["wake_dir"], dtype=float),
                           pts, swap, True, vel)
    onset = fc.V_inf * freestream_direction(fc.alpha, fc.beta)
    v_leg = onset + np.vstack(vel[1:])
    v_leg = v_leg - np.sum(v_leg * normal, axis=1)[:, None] * normal
    shared = kind == L.LEG_SHARED
    f_leg = fc.rho * (np.concatenate([g] * n_sets) * scale)[shared, None] * np.cross(v_leg[shared],
                                                                                     (end - start)[shared])
    f_bound = fc.rho * g[:, None] * np.cross(onset + vel[0], lat.b - lat.a)
    return f_leg.sum(axis=0), f_bound.sum(axis=0)


def test_aileron_side_force_on_a_flat_wing_comes_from_the_free_edges():
    """Regression: the normal leak velocity on the legs gave an aileron side force proportional to alpha.

    The shared edges carry no in-plane force; the side force is the side-edge
    force of the free tip edges (full velocity), antisymmetric in the deflection.
    The geometric aileron also tilts the bound vortices in the join at its
    inboard end (a warped surface, as a dihedral): their side force is the
    rest of the near-field CY.
    """
    cy = {}
    for delta in (2.0, -2.0):
        ac = _wing(aileron_deg=delta)
        res, fc = _solve(ac, alpha_deg=3.0)
        f_free, _ = _free_edge_loads(res, fc)
        side = np.cross(lift_direction(fc.alpha), freestream_direction(fc.alpha, fc.beta))
        cy_free = float(f_free @ side) / (0.5 * fc.rho * fc.V_inf ** 2 * ac.S_ref)
        assert abs(cy_free) > 1e-6
        f_shared, f_bound = _shared_leg_and_bound_forces(res, fc)
        q = 0.5 * fc.rho * fc.V_inf ** 2 * ac.S_ref
        assert abs(float(f_shared @ side) / q) <= 1e-12 * abs(cy_free)
        assert res.totals.CY == pytest.approx(cy_free + float(f_bound @ side) / q, rel=1e-9, abs=1e-14)
        cy[delta] = res.totals.CY
    assert cy[2.0] == pytest.approx(-cy[-2.0], rel=1e-9)


def test_aileron_yaw_agrees_with_the_trefftz_plane():
    """Regression: the yawing moment due to aileron (stability axes) is within 5 % of the Trefftz value
    (it was 13 % low with the full velocity on all legs).

    The yawing moment of the side-edge force of the free tip edges is taken
    out of the near field: the Trefftz plane has no chordwise position, so
    it cannot contain the yaw of a side force ahead of or behind the
    reference point.
    """
    alpha = 3.0
    near, far = [], []
    for delta in (2.0, -2.0):
        ac = _wing(aileron_deg=delta)
        res, fc = _solve(ac, alpha, n_panels=24, n_chord=8)
        _, m_free = _free_edge_loads(res, fc)
        q_sb = 0.5 * fc.rho * fc.V_inf ** 2 * ac.S_ref * ac.b_ref
        cn_free = float(transform_moments(-m_free[0] / q_sb, 0.0, -m_free[2] / q_sb, np.radians(alpha), 0.0,
                                          "stability", b_ref=float(res.b_ref), c_ref=float(res.c_ref))["Cn"])
        # No independent reference of the yaw of the side-edge force exists
        # here: the Trefftz-plane trace of a flat wing with a deflected
        # aileron is flat (CY_farfield = 0), and no measured value of this
        # case is available. The check is only that this part is small
        # against the near-field yaw (below 10 %).
        assert abs(cn_free) < 0.1 * abs(_stability_cn(res, alpha))
        near.append(_stability_cn(res, alpha) - cn_free)
        far.append(_trefftz_yaw(res, ac, fc))
    cn_near = (near[0] - near[1]) / np.radians(4.0)
    cn_far = (far[0] - far[1]) / np.radians(4.0)
    assert cn_far > 0.0  # adverse yaw: the wing with the trailing edge down has more induced drag
    assert cn_near == pytest.approx(cn_far, rel=0.05)


def test_dihedral_wing_side_force_agrees_with_the_far_field():
    """Regression: the side force of a 10 deg dihedral wing in sideslip agrees with the far-field
    Kutta-Joukowski force of the wake trace.

    The comparison is normal to the free stream (wind axes): the far-field
    force of the trace has no drag component, so its body-axis y component
    misses the drag times sin(beta) of the near field. At alpha 4 deg the two
    agree within 5 %; at alpha 8 deg the near/far rule of the trust score
    (5 % or the CY floor) gives no warning.
    """
    from ventorum.core.trust import NEAR_FAR_FLOOR_CY

    ac = _wing(dihedral_deg=10.0)
    for alpha, rel in ((4.0, 0.05), (8.0, None)):
        res, fc = _solve(ac, alpha_deg=alpha, beta_deg=4.0, n_chord=8)
        t = res.totals
        D = freestream_direction(fc.alpha, fc.beta)
        side_dir = np.cross(lift_direction(fc.alpha), D)
        q_s = 0.5 * fc.rho * fc.V_inf ** 2 * ac.S_ref
        lat, g = res.details["lattice"], res.details["gamma"]
        wd = np.asarray(res.details["wake_dir"], dtype=float)
        seg = lat.b - lat.a
        seg_trace = seg - np.outer(seg @ wd, wd)
        f_far = fc.rho * fc.V_inf * np.sum(g[:, None] * np.cross(D, seg_trace), axis=0)
        cy_far = float(f_far @ side_dir) / q_s
        assert cy_far == pytest.approx(t.CY_farfield, rel=1e-9)
        if rel is not None:
            assert t.CY == pytest.approx(cy_far, rel=rel)
        diff = abs(t.CY - cy_far)
        assert not (diff > 0.05 * abs(cy_far) and diff > NEAR_FAR_FLOOR_CY)


@pytest.mark.parametrize("aspect_ratio, measured", [(1.34, -0.0080), (2.61, -0.0044), (5.16, -0.0011)])
def test_flat_wing_rolling_moment_due_to_sideslip_against_naca_tn1669(aspect_ratio, measured):
    """Cl_beta / CL [1/deg] of flat unswept untapered wings (stability axes) against NACA TN 1669, fig. 9
    (digitised, about +-0.0003). The legs give this effect; the tangential rule keeps it."""
    ac = _wing(semi_span=aspect_ratio / 2.0)
    cl = []
    for beta in (2.0, -2.0):
        res, _ = _solve(ac, alpha_deg=4.0, beta_deg=beta, n_panels=24, n_chord=8)
        t = res.totals
        cl.append(float(transform_moments(t.Cl, t.Cm, t.Cn, np.radians(4.0), np.radians(beta), "stability",
                                          b_ref=float(res.b_ref), c_ref=float(res.c_ref))["Cl"]))
    res0, _ = _solve(ac, alpha_deg=4.0, n_panels=24, n_chord=8)
    slope = (cl[0] - cl[1]) / 4.0 / res0.totals.CL
    assert slope < 0.0  # stable effective dihedral, as measured
    assert slope == pytest.approx(measured, abs=max(0.0008, 0.25 * abs(measured)))


@pytest.mark.gpu
def test_gpu_leg_forces_match_the_cpu():
    """The GPU loads use the same leg rules as the CPU (float64): tangential, kink and free edges."""
    from ventorum import gpu

    if not gpu.available():
        pytest.skip("no CUDA GPU")
    ac = _wing(aileron_deg=3.0, dihedral_deg=5.0)
    fc = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=np.radians(3.0))
    st = vt.SolverSettings(solver_type="vlm", n_panels=16, n_chord=4)
    old_dev, old_prec = gpu.get_device(), gpu.get_precision()
    try:
        gpu.set_device("cpu")
        cpu = vt.analyze(ac, fc, st)
        gpu.set_device("gpu")
        gpu.set_precision("float64")
        dev = vt.analyze(ac, fc, st)
    finally:
        gpu.set_device(old_dev)
        gpu.set_precision(old_prec)
    assert dev.details["device"] == "gpu"
    for name in ("CL", "CY", "Cl", "Cm", "Cn"):
        assert getattr(dev.totals, name) == pytest.approx(getattr(cpu.totals, name), rel=1e-9, abs=1e-12), name
