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


def test_twisted_wing_has_no_kink_inside_each_half():
    """Twist turns the strip normals about an axis across the edge line: the edges inside each half are not kinks.

    The root can be a kink: with twist and taper the quarter-chord line of
    the two halves meets at an angle.
    """
    w = vt.LiftingSurface(name="w", semi_span=4.0, sections=[
        vt.WingSection(0.0, 1.0, twist=math.radians(2.0)), vt.WingSection(1.0, 0.6, twist=math.radians(-4.0))])
    lat = build_lattice(vt.Aircraft(surfaces=[w]), vt.SolverSettings(n_panels=10, n_chord=2), collocation="vlm")
    _, _, kind = L.leg_force_rules(lat)
    mid = 0.5 * (np.vstack([lat.a_te, lat.b]) + np.vstack([lat.a, lat.b_te]))
    assert not np.any((kind == L.LEG_KINK) & (np.abs(mid[:, 1]) > 1e-9))


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
