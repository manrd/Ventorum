# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Kutta-Joukowski force on the on-wing legs of the vortex lattice.

The surface is impermeable, so the leg force uses only the velocity tangent
to the strip surface. The normal velocity at the leg points is the leak of
the discrete lattice between its control points; its force is an in-plane
force that a thin surface cannot carry. The checks compare the near field
with far-field (Trefftz-plane) values and with measured data:

* the side force of an aileron on a flat wing (the in-plane leak force gave
  CY_delta about +0.025/rad at alpha 3 deg; a flat-trace far field gives 0),
* the yawing moment of an aileron against the moment of the Trefftz-plane
  induced drag,
* the body-axis side force of a dihedral wing in sideslip against the
  far-field force,
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
                                   "stability")["Cn"])


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


def test_aileron_gives_no_in_plane_side_force_on_a_flat_wing():
    """Regression: the normal leak velocity on the legs gave an aileron side force proportional to alpha."""
    for delta in (2.0, -2.0):
        res, _ = _solve(_wing(aileron_deg=delta), alpha_deg=3.0)
        assert abs(res.totals.CY) < 1e-12, res.totals.CY


def test_aileron_yaw_agrees_with_the_trefftz_plane():
    """Regression: the yawing moment due to aileron (stability axes) is within 5 % of the Trefftz value
    (it was 13 % low with the full velocity on the legs)."""
    alpha = 3.0
    near, far = [], []
    for delta in (2.0, -2.0):
        ac = _wing(aileron_deg=delta)
        res, fc = _solve(ac, alpha, n_panels=24, n_chord=8)
        near.append(_stability_cn(res, alpha))
        far.append(_trefftz_yaw(res, ac, fc))
    cn_near = (near[0] - near[1]) / np.radians(4.0)
    cn_far = (far[0] - far[1]) / np.radians(4.0)
    assert cn_far > 0.0  # adverse yaw: the wing with the trailing edge down has more induced drag
    assert cn_near == pytest.approx(cn_far, rel=0.05)


def test_dihedral_wing_body_side_force_agrees_with_the_far_field():
    """Regression: body-axis side force of a 10 deg dihedral wing in sideslip at alpha 8 deg is within 5 %
    of the far-field Kutta-Joukowski force of the wake trace (it was 14 % high)."""
    ac = _wing(dihedral_deg=10.0)
    res, fc = _solve(ac, alpha_deg=8.0, beta_deg=4.0, n_chord=8)
    t = res.totals
    D = freestream_direction(fc.alpha, fc.beta)
    lift_dir = lift_direction(fc.alpha)
    side_dir = np.cross(lift_dir, D)
    q_s = 0.5 * fc.rho * fc.V_inf ** 2 * ac.S_ref
    f_near = q_s * (t.CL * lift_dir + t.CY * side_dir + t.CDi_nearfield * D)
    lat, g = res.details["lattice"], res.details["gamma"]
    wd = np.asarray(res.details["wake_dir"], dtype=float)
    seg = lat.b - lat.a
    seg_trace = seg - np.outer(seg @ wd, wd)
    f_far = fc.rho * fc.V_inf * np.sum(g[:, None] * np.cross(D, seg_trace), axis=0)
    assert f_near[1] == pytest.approx(f_far[1], rel=0.05)


@pytest.mark.parametrize("aspect_ratio, measured", [(1.34, -0.0080), (2.61, -0.0044), (5.16, -0.0011)])
def test_flat_wing_rolling_moment_due_to_sideslip_against_naca_tn1669(aspect_ratio, measured):
    """Cl_beta / CL [1/deg] of flat unswept untapered wings (stability axes) against NACA TN 1669, fig. 9
    (digitised, about +-0.0003). The legs give this effect; the tangential rule keeps it."""
    ac = _wing(semi_span=aspect_ratio / 2.0)
    cl = []
    for beta in (2.0, -2.0):
        res, _ = _solve(ac, alpha_deg=4.0, beta_deg=beta, n_panels=24, n_chord=8)
        t = res.totals
        cl.append(float(transform_moments(t.Cl, t.Cm, t.Cn, np.radians(4.0), np.radians(beta), "stability")["Cl"]))
    res0, _ = _solve(ac, alpha_deg=4.0, n_panels=24, n_chord=8)
    slope = (cl[0] - cl[1]) / 4.0 / res0.totals.CL
    assert slope < 0.0  # stable effective dihedral, as measured
    assert slope == pytest.approx(measured, abs=max(0.0008, 0.25 * abs(measured)))


@pytest.mark.gpu
def test_gpu_leg_forces_match_the_cpu():
    """The GPU loads use the same tangential leg rule as the CPU (float64)."""
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
