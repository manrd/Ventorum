# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Leg forces at joined surface edges, and the warnings of the junction rules.

A fin on the tip of a tail and a winglet on the tip of a wing are joined
edges with a 90 deg fold. The two coincident legs of such an edge carry no
force (the velocity is projected on the edge line, see
``ventorum.aero.loads.leg_force_rules``). With the strip normal of each leg,
the velocity normal to one surface (tangential to the other) at the fin
root legs gave a side force that grew about as the square of the panel
count. The checks:

* CY_beta of the fin on the tail tip and of a 90 deg winglet converges with
  the panels and agrees with the CY_beta of the Trefftz plane,
* the coincident legs of a joined corner carry no net force,
* edges whose chordwise grids do not match are not joined, with a warning,
* an edge joined to an interior strip edge of another surface gives a warning,
* the stability tool reports the trust of all its cases and compares CY_beta
  with the CY_beta of the Trefftz plane.
"""

from __future__ import annotations

import math
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

import ventorum as vt
from ventorum.aero import loads as L
from ventorum.agent import tools as T
from ventorum.agent.schemas import build_aircraft_from_spec, parse_settings
from ventorum.geometry.lattice import build_lattice


def _fin_tail_spec(fin_dx=0.0):
    return {"name": "F", "ref_point_m": [0.3, 0.0, 0.0], "surfaces": [
        {"name": "wing", "span_m": 10.0, "chord_m": 1.0},
        {"name": "htail", "span_m": 3.0, "chord_m": 0.6, "position_m": [4.0, 0.0, 0.0]},
        {"name": "fin", "semi_span_m": 1.2, "chord_m": 0.6, "dihedral_deg": 90.0, "symmetric": False,
         "mirror": True, "position_m": [4.0 + fin_dx, 1.5, 0.0]}]}


def _winglet_spec():
    return {"name": "WL", "ref_point_m": [0.3, 0.0, 0.0], "surfaces": [
        {"name": "wing", "span_m": 10.0, "chord_m": 1.0},
        {"name": "winglet", "semi_span_m": 0.8, "chord_m": 1.0, "dihedral_deg": 90.0, "symmetric": False,
         "mirror": True, "position_m": [0.0, 5.0, 0.0]}]}


def _case(spec, n_panels):
    """Return the aircraft and the case settings as the stability tool makes them."""
    ac = build_aircraft_from_spec(aircraft=spec)
    base = ac.moment_reference()
    ac.ref_point = np.array([0.3, 0.0, float(base[2])])
    ac.compute_reference_values()
    sett = parse_settings({"n_panels": n_panels})
    strips = T._check_mesh(ac, sett)
    fc = vt.FlightCondition(V_inf=40.0, alpha=math.radians(3.0))
    case_sett, _ = T._common_settings(ac, sett, strips, [fc])
    return ac, case_sett


def _cy_beta(spec, n_panels):
    """Return CY_beta and the CY_beta of the Trefftz plane [1/rad] (central difference, beta +/- 1 deg)."""
    ac, st = _case(spec, n_panels)
    tot = {}
    for b in (1.0, -1.0):
        fc = vt.FlightCondition(V_inf=40.0, alpha=math.radians(3.0), beta=math.radians(b))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tot[b] = vt.analyze(ac.clone(), fc, st).totals
    d = math.radians(2.0)
    return (tot[1.0].CY - tot[-1.0].CY) / d, (tot[1.0].CY_farfield - tot[-1.0].CY_farfield) / d


@pytest.mark.slow
@pytest.mark.parametrize("spec", [_fin_tail_spec(), _winglet_spec()], ids=["fin_on_tail_tip", "winglet_90"])
def test_cy_beta_of_a_joined_90_deg_junction_converges(spec):
    """CY_beta changes less than 2 % from 40 to 80 panels and agrees with the Trefftz value within 2 %.

    Regression: with the strip normal of each leg, CY_beta of the fin on the
    tail tip grew by a factor of about 3 from 40 to 80 panels.
    """
    cy40, _ = _cy_beta(spec, 40)
    cy80, ff80 = _cy_beta(spec, 80)
    assert cy80 < 0.0 and ff80 < 0.0
    assert abs(cy80 - cy40) <= 0.02 * abs(cy80)
    assert abs(cy80 - ff80) <= 0.02 * abs(ff80)


def test_coincident_legs_of_a_joined_corner_carry_no_net_force():
    """The fin root legs and the tail tip legs coincide; they are kink legs and their net force is zero."""
    ac, st = _case(_fin_tail_spec(), 20)
    fc = vt.FlightCondition(V_inf=40.0, alpha=math.radians(3.0), beta=math.radians(2.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = vt.analyze(ac.clone(), fc, st)
    lat = res.details["lattice"]
    g = np.asarray(res.details["gamma"], dtype=float)
    n_p = lat.n_panels
    normal, scale, kind = L.leg_force_rules(lat)
    surf = lat.strip_surface[np.concatenate([lat.panel_strip, lat.panel_strip])]
    names = [s.name for s in lat.surfaces]
    start = np.vstack([lat.a_te, lat.b])
    end = np.vstack([lat.a, lat.b_te])
    mid = 0.5 * (start + end)
    # The legs at the right tail tip and the right fin root (y = 1.5 m, z = 0).
    corner = np.flatnonzero((np.abs(mid[:, 1] - 1.5) < 1e-9) & (np.abs(mid[:, 2]) < 1e-9))
    on = {names[k] for k in surf[corner]}
    assert on == {"htail", "fin"}
    assert np.all(kind[corner] == L.LEG_KINK) and np.all(scale[corner] == 0.0)
    pts = [lat.force_points, 0.5 * (lat.a_te + lat.a), 0.5 * (lat.b + lat.b_te)]
    vel = [np.empty((n_p, 3)) for _ in pts]
    L._near_field_velocity(lat, g, fc, None, np.asarray(res.details["wake_dir"], dtype=float), pts, [0, 2, 1],
                           True, vel)
    from ventorum.aero.system import freestream_direction

    v = fc.V_inf * freestream_direction(fc.alpha, fc.beta) + np.vstack([vel[1], vel[2]])
    g2 = np.concatenate([g, g])

    def net(nrm, scl):
        vt_ = v - np.sum(v * nrm, axis=1)[:, None] * nrm
        return (fc.rho * (g2 * scl)[:, None] * np.cross(vt_, end - start))[corner].sum(axis=0)

    assert np.max(np.abs(net(normal, scale))) == 0.0
    # With the strip normal of each leg (the old rule) the corner legs gave a net force.
    strip_normal = lat.normal[np.concatenate([lat.panel_strip, lat.panel_strip])]
    assert np.max(np.abs(net(strip_normal, np.ones_like(scale)))) > 1e-3


def test_free_tip_legs_use_the_full_velocity_and_interior_legs_the_strip_normal():
    """Leg kinds of a flat wing: free legs at the two tips, shared legs inside, no kink."""
    w = vt.LiftingSurface(name="w", semi_span=4.0, sections=[vt.WingSection(0.0, 1.0), vt.WingSection(1.0, 1.0)])
    lat = build_lattice(vt.Aircraft(surfaces=[w]), vt.SolverSettings(n_panels=8, n_chord=3), collocation="vlm")
    normal, scale, kind = L.leg_force_rules(lat)
    mid = 0.5 * (np.vstack([lat.a_te, lat.b]) + np.vstack([lat.a, lat.b_te]))
    tip = np.abs(np.abs(mid[:, 1]) - 4.0) < 1e-9
    assert np.all(kind[tip] == L.LEG_FREE) and np.all(normal[tip] == 0.0)
    assert np.all(kind[~tip] == L.LEG_SHARED) and np.all(scale == 1.0)
    assert np.allclose(normal[~tip], lat.normal[np.concatenate([lat.panel_strip, lat.panel_strip])][~tip])


def test_twisted_wing_has_no_kink():
    """Twist and taper never make a kink: the kinks come from the geometry definition."""
    w = vt.LiftingSurface(name="w", semi_span=4.0, sections=[
        vt.WingSection(0.0, 1.0, twist=math.radians(2.0)), vt.WingSection(1.0, 0.6, twist=math.radians(-4.0))])
    lat = build_lattice(vt.Aircraft(surfaces=[w]), vt.SolverSettings(n_panels=10, n_chord=2), collocation="vlm")
    _, scale, kind = L.leg_force_rules(lat)
    assert not np.any(kind == L.LEG_KINK) and np.all(scale == 1.0)


def _swept_twisted_wing():
    return vt.LiftingSurface(name="W", semi_span=4.0, sweep_le=math.radians(30.0), sections=[
        vt.WingSection(y_frac=0.0, chord=1.2), vt.WingSection(y_frac=1.0, chord=0.6, twist=math.radians(-3.0))])


def _tangential_rules(lattice):
    """Leg rules without kinks: the strip normal on every shared edge, the full velocity on the free edges."""
    _, _, kind = L.leg_force_rules(lattice)
    strip_normal = lattice.normal[np.concatenate([lattice.panel_strip, lattice.panel_strip])]
    free = kind == L.LEG_FREE
    return np.where(free[:, None], 0.0, strip_normal), np.ones(kind.size), np.where(free, L.LEG_FREE, L.LEG_SHARED)


def test_swept_twisted_wing_in_sideslip_has_no_kink(monkeypatch):
    """Regression: the piecewise-flat strips of a swept, tapered wing with washout fold about the edges.

    The fold scales as 1/n, so a fold threshold made the interior edges kinks
    as the mesh was refined, and Cl lost the leg forces. The kinks now come
    from the geometry definition: there are none here, Cl changes less than
    1 % from 20 to 80 panels, and it equals the value of the tangential rule.
    From 10 to 80 panels the change is about 1.2 %, as with the leg rule
    before the kink rules: that is the mesh convergence of Cl, not the rule
    (the limit of that step is 1.5 %).
    """
    cond = vt.FlightCondition(V_inf=30.0, alpha=math.radians(6.0), beta=math.radians(5.0))
    cl = {}
    for n in (10, 20, 80):
        st = vt.SolverSettings(solver_type="vlm", n_panels=n, n_chord=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = vt.analyze(vt.Aircraft(surfaces=[_swept_twisted_wing()]), cond, st)
        _, scale, kind = L.leg_force_rules(res.details["lattice"])
        assert not np.any(kind == L.LEG_KINK) and np.all(scale == 1.0)
        cl[n] = res.totals.Cl
    assert abs(cl[80] - cl[20]) <= 0.01 * abs(cl[80])
    assert abs(cl[80] - cl[10]) <= 0.015 * abs(cl[80])
    monkeypatch.setattr(L, "leg_force_rules", _tangential_rules)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = vt.analyze(vt.Aircraft(surfaces=[_swept_twisted_wing()]), cond,
                         vt.SolverSettings(solver_type="vlm", n_panels=80, n_chord=2)).totals.Cl
    assert cl[80] == pytest.approx(ref, rel=1e-12)


def test_loads_are_continuous_in_a_small_dihedral():
    """A flat wing at a dihedral of 0.0286 and 0.029 deg: no jump of CL, CY and Cl in sideslip.

    A fold below LEG_KINK_BLEND_RAD scales the leg force of the tangential
    rule, so the loads are continuous in the dihedral angle, also through 0.
    """
    cond = vt.FlightCondition(V_inf=20.0, alpha=math.radians(4.0), beta=math.radians(5.0))
    out = {}
    for d in (0.0, 0.0286, 0.029, 0.0294):
        w = vt.LiftingSurface(name="w", semi_span=4.0, dihedral=math.radians(d),
                              sections=[vt.WingSection(0.0, 1.0), vt.WingSection(1.0, 1.0)])
        st = vt.SolverSettings(solver_type="vlm", n_panels=24, spacing="cosine", n_chord=4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = vt.analyze(w, cond, st).totals
        out[d] = np.array([t.CL, t.CY, t.Cl])
    step_a = out[0.029] - out[0.0286]
    step_b = out[0.0294] - out[0.029]
    # Equal steps of the dihedral give nearly equal changes (no jump at either side).
    assert np.all(np.abs(step_a - step_b) <= 0.05 * np.abs(step_b) + 1e-9)
    # From 0 to 0.0286 deg the change is about 0.0286 / 0.0004 times the small step.
    assert np.all(np.abs(out[0.0286] - out[0.0]) <= 100.0 * np.abs(step_a) + 1e-8)


def test_offset_chordwise_grids_are_not_joined():
    """A fin root edge 0.2 m aft of the tail tip edge (chordwise grids do not match) is not joined; it warns."""
    ac = build_aircraft_from_spec(aircraft=_fin_tail_spec(fin_dx=0.2))
    lat = build_lattice(ac, vt.SolverSettings(n_panels=20), collocation="vlm")
    group = {s.name: int(lat.strip_core_group[s.strips][0]) for s in lat.surfaces}
    assert group["fin"] != group["htail"]
    assert any("chordwise grids do not match" in w and "htail" in w and "fin" in w for w in lat.join_warnings)
    # The matched junction is joined and gives no such warning.
    lat0 = build_lattice(build_aircraft_from_spec(aircraft=_fin_tail_spec()), vt.SolverSettings(n_panels=20),
                         collocation="vlm")
    group0 = {s.name: int(lat0.strip_core_group[s.strips][0]) for s in lat0.surfaces}
    assert group0["fin"] == group0["htail"]
    assert not lat0.join_warnings


def test_edge_on_an_interior_strip_edge_warns():
    """A tail tip edge on an interior strip edge of a fin that passes through the tail plane gives a warning."""
    tail = vt.LiftingSurface(name="tail", semi_span=1.5, n_panels=6, spacing="uniform",
                             sections=[vt.WingSection(0.0, 0.6), vt.WingSection(1.0, 0.6)])
    fin = vt.LiftingSurface(name="fin", semi_span=1.0, n_panels=4, spacing="uniform", is_symmetric=False,
                            dihedral=math.radians(90.0), position=np.array([0.0, 1.5, -0.5]),
                            sections=[vt.WingSection(0.0, 0.6), vt.WingSection(1.0, 0.6)])
    ac = vt.Aircraft(surfaces=[tail, fin, fin.mirrored()])
    lat = build_lattice(ac, vt.SolverSettings(n_panels=6), collocation="vlm")
    assert any("interior strip edge of surface 'fin" in w for w in lat.join_warnings)


def _trust(score, warns):
    return SimpleNamespace(score=score, rating="HIGH" if score > 0.8 else "MODERATE", warnings=list(warns),
                           recommendations=[], factors={}, uncertainty_CL=0.0, uncertainty_CDi=0.0,
                           uncertainty_LD=0.0)


def test_stability_trust_reports_all_cases():
    """The stability trust has the warnings of every case, the weakest score and the CY_beta comparison."""
    cases = {"base": (3.0, 0.0), "a+": (3.5, 0.0), "a-": (2.5, 0.0), "b+": (3.0, 1.0), "b-": (3.0, -1.0)}
    db = math.radians(2.0)

    def totals(score, warns, cy_ff=0.0):
        return SimpleNamespace(trust=_trust(score, warns), CY_farfield=cy_ff)

    t = {k: totals(0.9, ["common"]) for k in cases}
    t["b+"] = totals(0.7, ["common", "near/far CY"], cy_ff=-0.0075)
    t["b-"] = totals(0.9, ["common"], cy_ff=0.0075)
    # Near-field CY_beta -2.4/rad against the Trefftz value -0.43/rad: a warning.
    out = T._stability_trust(t, cases, -2.4, db)
    assert out["score"] == 0.7
    assert out["warnings"][0] == "common" and out["warnings"].count("common") == 1
    assert any(w.startswith("Case b+ (alpha=3 deg, beta=1 deg): near/far CY") for w in out["warnings"])
    assert any("CY_beta" in w and "Trefftz" in w and "junction" in w for w in out["warnings"])
    assert "case b+" in out["note"]
    # Near-field and Trefftz CY_beta agree: no CY_beta warning.
    out = T._stability_trust(t, cases, -0.432, db)
    assert not any("CY_beta" in w for w in out["warnings"])


def test_stability_tool_trust_on_a_fin_tail():
    """The stability tool runs on the fin on the tail tip; its trust has no CY_beta warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = T.stability_derivatives(aircraft=_fin_tail_spec(), flight_condition={"V_inf_m_s": 40.0, "alpha_deg": 3.0},
                                      settings={"n_panels": 20})
    assert out["status"] == "success"
    assert not any("CY_beta" in w for w in out["trust"]["warnings"])
