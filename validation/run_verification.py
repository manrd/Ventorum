# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Verification report for Ventorum.

Verification checks that the code solves its equations correctly, against
closed forms and independent solutions of the same theories. It also measures
the known limits of each method. It is not validation against experiment:
see ``validation/experimental/README.md``.

Every number in the report is computed when this script runs. There are no
hard-coded results.

Usage:
    python validation/run_verification.py [output.md]
"""

from __future__ import annotations

import datetime
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np

import ventorum as vt
from ventorum.aero.system import build_sources, make_ground_plane, make_unknown_map
from ventorum.aero.vortex import induced_velocity
from ventorum.geometry.lattice import build_lattice
from ventorum.reference import (
    REFERENCES,
    circular_wing_cl_alpha,
    elliptic_ground_drag_ratio,
    elliptic_wing_cl_alpha,
    glauert_monoplane,
    helmbold_cl_alpha,
    wing_sections_elliptic,
)
from ventorum.solvers.lattice_base import assemble_system_matrix

warnings.simplefilter("ignore", RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
# Small angle: the references are linear theories. (The default wake follows
# the free stream, which adds a small nonlinearity; see case V3.)
ALPHA = np.radians(0.5)


# ── helpers ──────────────────────────────────────────────────────────────────

def pct(x: float) -> str:
    return f"{100.0 * x:+.3f} %"


def table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def elliptic(AR: float, straight: str = "quarter_chord", a0: float | None = None) -> vt.Aircraft:
    c0 = 4.0 / (np.pi * AR)
    secs = wing_sections_elliptic(c0, straight_line=straight)
    if a0 is not None:
        for s in secs:
            s.airfoil = vt.LinearAirfoil(a0=a0)
    ac = vt.Aircraft(surfaces=[vt.LiftingSurface(semi_span=0.5, sections=secs)])
    ac.compute_reference_values()
    return ac


def tapered(AR: float, taper: float, straight_qc: bool, sweep_deg: float = 0.0) -> vt.Aircraft:
    c_r = 2.0 / (AR * (1.0 + taper))
    c_t = taper * c_r
    if straight_qc:
        secs = [vt.WingSection(0.0, c_r, x_le=-0.25 * c_r), vt.WingSection(1.0, c_t, x_le=-0.25 * c_t)]
    else:
        secs = [vt.WingSection(0.0, c_r), vt.WingSection(1.0, c_t)]
    ac = vt.Aircraft(surfaces=[vt.LiftingSurface(semi_span=0.5, sections=secs, sweep_le=np.radians(sweep_deg))])
    ac.compute_reference_values()
    return ac


def ar_of(ac: vt.Aircraft) -> float:
    return ac.b_ref ** 2 / ac.S_ref


def run(ac, solver, n=40, **kw):
    sett = vt.SolverSettings(solver_type=solver, n_panels=n, **kw)
    return vt.analyze(ac, settings=sett, condition=vt.FlightCondition(alpha=ALPHA)).totals


def lift_slope(t, solver: str) -> float:
    # The lattice solvers are linear in sin(alpha); the Fourier solver in alpha.
    return t.CL / (ALPHA if solver == "fourier" else np.sin(ALPHA))


# ── cases ────────────────────────────────────────────────────────────────────

def case_elliptic() -> str:
    rows = []
    worst = 0.0
    for AR in (4.0, 8.0, 16.0):
        for a0f in (1.0, 0.9):
            ac = elliptic(AR, a0=a0f * 2 * np.pi)
            ref = elliptic_wing_cl_alpha(ar_of(ac), a0f * 2 * np.pi)
            for solver in ("fourier", "linear", "nonlinear"):
                t = run(ac, solver)
                err = lift_slope(t, solver) / ref - 1.0
                worst = max(worst, abs(err), abs(t.e - 1.0))
                rows.append([f"{AR:g}", f"{a0f:g}", solver, f"{ref:.4f}", f"{lift_slope(t, solver):.4f}",
                             pct(err), f"{t.e:.5f}"])
    s = ["## V1. Elliptic wing (Lanchester–Prandtl lifting-line theory)", "",
         f"Reference: {REFERENCES['prandtl_elliptic']}", "",
         "Elliptic planform with a straight quarter-chord line (301 sections, tip chord 1e-3 of the root), "
         "alpha = 0.5 deg, 40 panels per semi-span, default spacing. a0 is the section lift slope as a "
         "fraction of 2 pi.", "",
         table(["AR", "a0/2pi", "solver", "CL_alpha theory", "CL_alpha", "error", "e"], rows), "",
         f"Largest error in CL_alpha or e: {100 * worst:.3f} %.", ""]
    return "\n".join(s)


def case_glauert() -> str:
    rows = []
    worst = {"fourier": 0.0, "linear": 0.0}
    for AR in (4.0, 6.0, 8.0, 10.0):
        for taper in (1.0, 0.5, 0.25):
            cla_ref, e_ref = glauert_monoplane(AR, taper)
            ac = tapered(AR, taper, straight_qc=True)
            for solver, n in (("fourier", 60), ("linear", 40)):
                t = run(ac, solver, n=n)
                ecl = lift_slope(t, solver) / cla_ref - 1.0
                ee = t.e / e_ref - 1.0
                worst[solver] = max(worst[solver], abs(ecl), abs(ee))
                rows.append([f"{AR:g}", f"{taper:g}", solver, f"{cla_ref:.4f}", pct(ecl), f"{e_ref:.4f}", pct(ee)])
    s = ["## V2. Rectangular and tapered wings (Glauert's monoplane equation)", "",
         f"Reference: {REFERENCES['glauert_monoplane']} Solved independently in "
         "`ventorum.reference.glauert_monoplane` (odd terms, 60 terms, converged to 5 digits).", "",
         "Straight quarter-chord line, alpha = 0.5 deg, a0 = 2 pi.", "",
         table(["AR", "taper", "solver", "CL_alpha ref", "CL_alpha error", "e ref", "e error"], rows), "",
         f"Largest error: Fourier {100 * worst['fourier']:.3f} %, lifting line {100 * worst['linear']:.3f} %.", ""]
    return "\n".join(s)


def case_lifting_surface() -> str:
    b = 1.0
    secs = wing_sections_elliptic(b, straight_line="mid_chord", n_sections=401)
    ac = vt.Aircraft(surfaces=[vt.LiftingSurface(semi_span=b / 2.0, sections=secs)])
    rows_c = []
    for n, nc in ((20, 8), (30, 16), (40, 24)):
        t = run(ac, "vlm", n=n, n_chord=nc)
        cla = lift_slope(t, "vlm")
        rows_c.append([str(n), str(nc), f"{cla:.4f}", pct(cla / circular_wing_cl_alpha() - 1.0)])
    rows_w = []
    for a_deg in (0.5, 2.0, 4.0, 8.0):
        r = []
        for wake in ("body", "freestream"):
            t = vt.analyze(ac, settings=vt.SolverSettings(solver_type="vlm", n_panels=30, n_chord=16, wake_alignment=wake),
                            condition=vt.FlightCondition(alpha=np.radians(a_deg))).totals
            r.append(f"{t.CL / np.sin(np.radians(a_deg)):.4f}")
        rows_w.append([f"{a_deg:g}"] + r)
    rows_h = []
    for AR in (1.0, 2.0, 4.0, 6.0, 10.0, 20.0):
        ac = elliptic(AR)
        t = run(ac, "vlm", n=30, n_chord=8)
        cla = lift_slope(t, "vlm")
        ar = ar_of(ac)
        rows_h.append([f"{AR:g}", f"{cla:.4f}", f"{helmbold_cl_alpha(ar):.4f}", pct(cla / helmbold_cl_alpha(ar) - 1.0),
                       f"{elliptic_wing_cl_alpha(ar):.4f}", f"{t.e:.4f}"])
    s = ["## V3. Lifting-surface checks of the vortex-lattice method", "",
         f"Circular wing. Reference: {REFERENCES['kinner_circular']}", "",
         table(["panels/semi-span", "chordwise", "CL_alpha", "vs 1.790"], rows_c), "",
         "Effect of the wake model on the circular wing, CL / sin(alpha). With the wake along the body "
         "x-axis (as in AVL) the wake does not move with alpha; CL / sin(alpha) changes only through the "
         "local velocity in the force. With the wake along the free stream the trailing vortices leave "
         "the trailing edge at the angle of attack, which lowers their downwash on the wing and adds "
         "lift that grows with alpha:", "",
         table(["alpha [deg]", "body-axis wake", "free-stream wake (default)"], rows_w), "",
         f"Elliptic wings (straight quarter-chord line). Helmbold: {REFERENCES['helmbold']} "
         "Helmbold's formula is an approximation, so the deviation is information, not an error.", "",
         table(["AR", "VLM CL_alpha", "Helmbold", "VLM vs Helmbold", "lifting line", "VLM e"], rows_h), ""]
    return "\n".join(s)


def case_sweep() -> str:
    def hd(A, lam_half):
        return 2 * np.pi * A / (2 + np.sqrt(A * A * (1 + np.tan(lam_half) ** 2) + 4))
    rows = []
    base = {}
    for sweep in (0.0, 5.0, 15.0, 30.0, 45.0):
        w = vt.LiftingSurface(semi_span=2.5, sweep_le=np.radians(sweep),
                               sections=[vt.WingSection(0, 1.0), vt.WingSection(1, 1.0)])
        cl = {}
        for solver in ("vlm", "linear"):
            cl[solver] = [run(w, solver, n=n, n_chord=1 if solver == "vlm" else None).CL / np.sin(ALPHA) for n in (10, 40, 160)]
        if sweep == 0.0:
            base = {k: v[-1] for k, v in cl.items()}
        ratio = cl["vlm"][-1] / base["vlm"]
        ratio_ref = hd(5.0, np.radians(sweep)) / hd(5.0, 0.0)
        rows.append([f"{sweep:g}",
                     " / ".join(f"{v:.4f}" for v in cl["vlm"]),
                     f"{ratio:.4f} ({pct(ratio / ratio_ref - 1.0)} vs Helmbold-Diederich)",
                     " / ".join(f"{v:.4f}" for v in cl["linear"]),
                     pct(cl["linear"][-1] / cl["linear"][0] - 1.0)])
    s = ["## V4. Swept wings: grid convergence", "",
         "Rectangular wing of aspect ratio 5, sweep at the leading edge = sweep of the quarter-chord line. "
         "CL_alpha with 10 / 40 / 160 panels per semi-span (VLM with one chordwise panel).", "",
         table(["sweep [deg]", "VLM CL_alpha", "VLM swept/straight", "lifting line CL_alpha", "lifting line change 10 to 160"], rows), "",
         "The VLM converges. The lifting line does not converge with sweep (the lift falls as panels are "
         "added); this is the reason the VLM is the default solver.", ""]
    return "\n".join(s)


def case_tapered_straight_le() -> str:
    rows = []
    for AR in (4.0, 8.0):
        for taper in (0.5, 0.25):
            cla_ref, e_ref = glauert_monoplane(AR, taper)
            ac = tapered(AR, taper, straight_qc=False)
            dl_sweep = np.degrees(np.arctan(abs(0.25 * (taper - 1.0) * 2.0 / (AR * (1.0 + taper))) / 0.5))
            t = run(ac, "linear", n=40)
            v = run(ac, "vlm", n=40)
            rows.append([f"{AR:g}", f"{taper:g}", f"{dl_sweep:.1f}",
                         pct(lift_slope(t, 'linear') / cla_ref - 1.0), pct(t.e / e_ref - 1.0), t.trust.rating,
                         pct(lift_slope(v, 'vlm') / cla_ref - 1.0)])
    s = ["## V5. Tapered wings with a straight leading edge (lifting-line limit)", "",
         "A straight leading edge with taper gives a forward-swept quarter-chord line. Errors against "
         "the straight-line Glauert solution (40 panels, default spacing):", "",
         table(["AR", "taper", "quarter-chord sweep [deg]", "lifting line CL_alpha", "lifting line e",
                "lifting line trust", "VLM CL_alpha (lifting-surface effect)"], rows), ""]
    return "\n".join(s)


def case_convergence() -> str:
    w = vt.LiftingSurface(semi_span=4.0, sections=[vt.WingSection(0, 1.0), vt.WingSection(1, 1.0)])
    rows = []
    for solver in ("vlm", "linear"):
        for spacing in ("auto", "cosine", "uniform"):
            vals = []
            for n in (10, 20, 40, 80, 160):
                t = run(w, solver, n=n, spacing=spacing)
                vals.append((lift_slope(t, solver), t.e))
            A = assemble_system_matrix(w, vt.FlightCondition(alpha=ALPHA),
                                       vt.SolverSettings(solver_type=solver, n_panels=160, spacing=spacing))
            rows.append([solver, spacing, " / ".join(f"{c:.4f}" for c, _ in vals),
                         " / ".join(f"{e:.4f}" for _, e in vals), f"{np.linalg.cond(A):.1e}"])
    s = ["## V6. Mesh convergence and conditioning", "",
         "Rectangular wing, aspect ratio 8, alpha = 2 deg; 10 / 20 / 40 / 80 / 160 panels per semi-span. "
         "The condition number is for 160 panels. (The previous solver had condition numbers up to 1e16 "
         "here, from trailing legs that left the bound vortex along the free stream.)", "",
         table(["solver", "spacing", "CL_alpha", "e", "cond(A)"], rows), "",
         "Uniform spacing converges slowly (control points at the strip centres); cosine-type spacing "
         "with control points at the mid parameter converges within 10 to 20 panels.", ""]
    return "\n".join(s)


def case_ground() -> str:
    # 1. Image boundary condition on a tilted, banked ground.
    w = vt.LiftingSurface(semi_span=2.0, sections=[vt.WingSection(0, 1.0), vt.WingSection(1, 0.7)])
    ac = vt.Aircraft(surfaces=[w])
    lat = build_lattice(ac, vt.SolverSettings(n_panels=10), collocation="vlm", n_chord=3)
    alpha, phi = np.radians(5.0), np.radians(3.0)
    gp = make_ground_plane(lat, 0.4, alpha, 0.0, phi, height_ref="min")
    src = build_sources(lat, vt.freestream_direction(alpha, 0.0), make_unknown_map(lat, False), gp)
    rng = np.random.default_rng(0)
    pts = rng.uniform([-0.5, -2.5, -1.0], [2.0, 2.5, 1.0], size=(200, 3))
    pts = 0.5 * (pts + gp.reflect_points(pts))
    v = induced_velocity(pts, src, rng.normal(size=lat.n_panels))
    resid = float(np.max(np.abs(v @ gp.normal)) / np.max(np.abs(v)))

    # 2. Far from the ground the free-air result returns.
    rect = vt.LiftingSurface(semi_span=3.0, sections=[vt.WingSection(0, 1.0), vt.WingSection(1, 1.0)])
    sett = vt.SolverSettings(n_panels=16, n_chord=4)
    free = vt.analyze(rect, settings=sett, alpha_deg=4.0).totals
    far = vt.analyze_ground_effect(rect, h=1e4, alpha_deg=4.0, settings=sett, compute_strike_limit=False)

    # 3. Chordwise convergence of the VLM near the ground.
    rows_nc = []
    for hc in (1.0, 0.3, 0.1, 0.05):
        cls = []
        for nc in (4, 8, 16, 32):
            r = vt.analyze_ground_effect(rect, h=hc, alpha_deg=4.0, height_ref="min",
                                          settings=vt.SolverSettings(n_panels=16, n_chord=nc), compute_strike_limit=False)
            cls.append(r.CL)
        rows_nc.append([f"{hc:g}", " / ".join(f"{c:.4f}" for c in cls)])

    # 4. Lifting line against the VLM (lift increment over free air).
    rows_ll = []
    for AR in (4.0, 8.0):
        wg = vt.LiftingSurface(semi_span=AR / 2, sections=[vt.WingSection(0, 1.0), vt.WingSection(1, 1.0)])
        st = vt.SolverSettings(n_panels=24)
        oge = {s: vt.analyze(wg, settings=vt.SolverSettings(solver_type=s, n_panels=24), alpha_deg=4.0).totals.CL
               for s in ("vlm", "linear")}
        for hc in (2.0, 1.25, 0.5):
            out = {}
            for s in ("vlm", "linear"):
                lat_s = build_lattice(vt.Aircraft(surfaces=[wg]), st, collocation="vlm", n_chord=1)
                gp_s = make_ground_plane(lat_s, hc, np.radians(4.0), height_ref="min")
                sol = vt.VortexLatticeSolver() if s == "vlm" else vt.LinearLLTSolver()
                try:
                    r = sol.solve(wg, vt.FlightCondition(alpha=np.radians(4.0)),
                                  vt.SolverSettings(solver_type=s, n_panels=24), ground=gp_s)
                    out[s] = r.totals.CL / oge[s] - 1.0
                except ValueError:
                    out[s] = float("nan")
            ll = "refused" if np.isnan(out["linear"]) else pct(out["linear"])
            rows_ll.append([f"{AR:g}", f"{hc:g}", pct(out["vlm"]), ll])

    # 5. Induced drag against the image Trefftz-plane theory of an elliptic loading.
    rows_ge = []
    for AR in (10.0, 6.0):
        c0 = 4.0 / (np.pi * AR)
        secs = wing_sections_elliptic(c0)
        for sec in secs:
            sec.airfoil = vt.LinearAirfoil(alpha_L0=np.radians(-1.0))
        ell = vt.Aircraft(surfaces=[vt.LiftingSurface(semi_span=0.5, sections=secs)])
        ell.compute_reference_values()
        st = vt.SolverSettings(solver_type="vlm", n_panels=40, n_chord=6)
        fr = vt.analyze(ell, settings=st, alpha_deg=0.0).totals
        k_free = fr.CDi / fr.CL ** 2
        for hb in (0.5, 0.3, 0.2, 0.1, 0.05):
            g = vt.analyze_ground_effect(ell, h=hb, alpha_deg=0.0, height_ref="qc", settings=st,
                                         compute_strike_limit=False)
            ratio = g.CDi / g.CL ** 2 / k_free
            ref = elliptic_ground_drag_ratio(hb)
            rows_ge.append([f"{AR:g}", f"{hb:g}", f"{ratio:.5f}", f"{ref:.5f}", pct(ratio / ref - 1.0)])

    s = ["## V7. Ground effect (image method)", "",
         f"* Velocity normal to a ground plane tilted by alpha = 5 deg and banked by 3 deg, at 200 points on "
         f"the plane, random circulation: largest |v.n| / |v| = {resid:.1e}.",
         f"* h = 1e4 chords: CL {pct(far.CL / free.CL - 1.0)} and CDi {pct(far.CDi / free.CDi - 1.0)} against free air.",
         "",
         "VLM chordwise convergence (rectangular wing AR 6, alpha 4 deg, h = smallest clearance), CL with 4 / 8 / 16 / 32 chordwise panels:", "",
         table(["h_min/c", "CL"], rows_nc), "",
         "Lift increment over free air, lifting line against VLM (rectangular wings, alpha 4 deg). The "
         "lifting line is refused below h_min/c = 1:", "",
         table(["AR", "h_min/c", "VLM", "lifting line"], rows_ll), "",
         "Induced-drag ratio k_GE / k_free (k = CDi / CL^2) of planar elliptic wings parallel to the ground "
         "(alpha = 0, alpha_L0 = -1 deg, 40 x 6 panels), against the Trefftz-plane image theory of an "
         "elliptic loading (C. Wieselsberger, NACA TM 77, 1922; exact evaluation in "
         "`ventorum.reference.ground_effect`). The reference keeps the loading elliptic; the VLM loading "
         "departs from elliptic near the ground, so the difference grows below h/b = 0.2:", "",
         table(["AR", "h/b", "VLM", "image theory", "difference"], rows_ge), "",
         "Limits that verification cannot remove: the model is a thin, inviscid wing over a flat, "
         "rigid ground with a wake parallel to the ground. Thickness (suction under the wing), viscous "
         "effects and wake deformation are not modelled. Experimental validation is pending.", ""]
    return "\n".join(s)


def case_camber_moment() -> str:
    rows = []
    aL0, cm0 = np.radians(-4.0), -0.093
    af = vt.LinearAirfoil(a0=0.95 * 2 * np.pi, alpha_L0=aL0, Cm0=cm0)
    w = vt.LiftingSurface(semi_span=4.0, sections=[vt.WingSection(0, 1.0, airfoil=af), vt.WingSection(1, 1.0, airfoil=af)])
    for solver in ("vlm", "linear", "fourier"):
        a = aL0 if solver != "linear" else np.arcsin(aL0)
        sett = vt.SolverSettings(solver_type=solver, n_panels=20, wake_alignment="body")
        t = vt.analyze(w, condition=vt.FlightCondition(alpha=a), settings=sett).totals
        rows.append([solver, f"{t.CL:.1e}", f"{t.Cm:.6f}", f"{abs(t.Cm - cm0):.1e}"])
    s = ["## V8. Camber and section pitching moment", "",
         f"Rectangular wing, alpha_L0 = -4 deg, Cm0 = {cm0}, at zero lift (moments about the leading edge). "
         "The pitching moment must equal Cm0.", "",
         table(["solver", "CL", "Cm", "|Cm - Cm0|"], rows), ""]
    return "\n".join(s)


def case_signs() -> str:
    def clb(dih, wake, solver="vlm"):
        w = vt.LiftingSurface(semi_span=5.0, dihedral=np.radians(dih), sections=[vt.WingSection(0, 1.0), vt.WingSection(1, 1.0)])
        sett = vt.SolverSettings(solver_type=solver, n_panels=20, wake_alignment=wake)
        t = [vt.analyze(w, condition=vt.FlightCondition(alpha=np.radians(4), beta=np.radians(b)), settings=sett).totals for b in (-1, 1)]
        return (t[1].Cl - t[0].Cl) / np.radians(2)
    dihedrals = (-5.0, 0.0, 5.0)
    body = {d: clb(d, "body") for d in dihedrals}
    free = {d: clb(d, "freestream") for d in dihedrals}
    llt_body = {d: clb(d, "body", "linear") for d in dihedrals}
    rows = [[f"{d:+g}", f"{body[d]:+.5f}", f"{body[d] - body[0.0]:+.5f}", f"{free[d]:+.5f}", f"{llt_body[d]:+.5f}"]
            for d in dihedrals]

    def fin_beta_derivs():
        wing = vt.LiftingSurface(semi_span=5.0, sections=[vt.WingSection(0, 1.0), vt.WingSection(1, 1.0)])
        fin = vt.LiftingSurface(name="fin", semi_span=1.2, dihedral=np.radians(90.0), is_symmetric=False,
                                position=np.array([5.0, 0.0, 0.2]),
                                sections=[vt.WingSection(0, 1.0), vt.WingSection(1, 0.6)])
        ac = vt.Aircraft(surfaces=[wing, fin])
        sett = vt.SolverSettings(solver_type="vlm", n_panels=20, wake_alignment="body")
        t = [vt.analyze(ac, condition=vt.FlightCondition(alpha=np.radians(4), beta=np.radians(b)),
                         settings=sett).totals for b in (-1, 1)]
        db = np.radians(2)
        return (t[1].CY - t[0].CY) / db, (t[1].Cn - t[0].Cn) / db

    cyb_fin, cnb_fin = fin_beta_derivs()
    s = ["## V9. Sign conventions", "",
         "Rolling moment due to sideslip of a rectangular wing (AR 10, alpha 4 deg). Standard convention: "
         "positive dihedral gives Cl_beta < 0. The dihedral term (Cl_beta minus the value of the flat "
         "wing) changes sign with the dihedral. In the vortex-lattice method the flat wing has a Cl_beta "
         "of its own: the Kutta-Joukowski force on the chordwise vortex legs that lie on the surface is "
         "not zero in sideslip. The lifting line has no chordwise extent (forces on the bound vortex "
         "only), so with the body-axis wake its flat-wing value is zero. With the free-stream wake the "
         "wake skew adds a further term.", "",
         table(["dihedral [deg]", "VLM, body-axis wake [1/rad]", "VLM dihedral term [1/rad]",
                "VLM, free-stream wake [1/rad]", "lifting line, body-axis wake [1/rad]"], rows), "",
         "Fin case: the same rectangular wing with an aft fin (vertical surface of semi-span 1.2 m, "
         "root chord 1.0 m, tip chord 0.6 m, at x = 5.0 m, z = 0.2 m), alpha 4 deg, VLM with the "
         "body-axis wake, 20 panels. The fin gives a side force away from the wind (CY_beta < 0) and "
         "weathercock stability (Cn_beta > 0).", "",
         table(["CY_beta [1/rad]", "Cn_beta [1/rad]"], [[f"{cyb_fin:+.5f}", f"{cnb_fin:+.5f}"]]), ""]
    return "\n".join(s)


def case_split_surfaces() -> str:
    from ventorum.geometry import lattice as lattice_module

    n = 40
    sett = vt.SolverSettings(n_panels=n)
    cond = vt.FlightCondition(V_inf=20.0, alpha=np.radians(5.0))

    def surf(name, semi_span, n_panels, **kw):
        return vt.LiftingSurface(name=name, semi_span=semi_span, n_panels=n_panels, spacing="uniform",
                                  sections=[vt.WingSection(0.0, 1.0), vt.WingSection(1.0, 1.0)], **kw)

    def ac_of(surfaces):
        return vt.Aircraft(surfaces=surfaces, S_ref=8.0, b_ref=8.0, c_ref=1.0)

    def totals(ac, force=None):
        if force is None:
            return vt.analyze(ac, condition=cond, settings=sett).totals
        original = lattice_module.core_groups
        lattice_module.core_groups = lambda surfaces: np.full(len(surfaces), force, dtype=int) if force == 0             else np.arange(len(surfaces))
        try:
            return vt.analyze(ac, condition=cond, settings=sett).totals
        finally:
            lattice_module.core_groups = original

    one = totals(ac_of([surf("one", 4.0, n)]))
    rows = []
    for gap in (0.0, 5e-5):
        inner = surf("inner", 2.0, n // 2)
        outer = surf("outer", 2.0, n // 2, is_symmetric=False, position=np.array([0.0, 2.0 + gap, 0.0]))
        t = totals(ac_of([inner, outer, outer.mirrored()]))
        rows.append([f"split wing, gap {1e3 * gap:g} mm", pct(t.CL / one.CL - 1.0), pct(t.CDi / one.CDi - 1.0)])
    pos = np.array([5.0, 0.0, 0.4])
    wing = surf("wing", 4.0, n)
    vee = surf("vtail", 1.2, 12, dihedral=np.radians(35.0), position=pos)
    half = surf("half", 1.2, 12, dihedral=np.radians(35.0), position=pos, is_symmetric=False)
    t1, t2 = totals(ac_of([wing, vee])), totals(ac_of([wing, half, half.mirrored()]))
    rows.append(["V-tail from two halves", pct(t2.CL / t1.CL - 1.0), pct(t2.CDi / t1.CDi - 1.0)])

    rows2 = []
    for inc in (0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0):
        inner = surf("inner", 2.0, n // 2)
        outer = surf("outer", 2.0, n // 2, is_symmetric=False, dihedral=np.radians(8.0),
                     position=np.array([0.0, 2.0, 0.0]), incidence=np.radians(inc))
        ac = ac_of([inner, outer, outer.mirrored()])
        lat = build_lattice(ac, sett, collocation="vlm")
        group = lat.strip_core_group
        joined = group[lat.surfaces[0].strips][0] == group[lat.surfaces[1].strips][0]
        gap = float(np.linalg.norm(lat.surfaces[1].edge_te[0] - lat.surfaces[0].edge_te[-1]))
        width = 4.0 / n
        rows2.append([f"{inc:g}", f"{1e3 * gap:.1f}", f"{gap / width:.2f}", "yes" if joined else "no",
                      pct(totals(ac).CL / totals(ac, force=0).CL - 1.0),
                      pct(totals(ac, force=1).CL / totals(ac, force=0).CL - 1.0)])
    s = ["## V10. Split surfaces", "",
         "Surfaces that meet along an edge are one vortex sheet. The cross-surface vortex core must not act "
         "between them, so the surfaces must be in one core group even when their edges match only "
         "approximately. Wing of semi-span 4 m and chord 1 m, 40 strips, alpha = 5 deg. Differences are "
         "against the one-piece surface.", "",
         table(["configuration", "CL difference", "CDi difference"], rows), "",
         "Dihedral break of 8 deg at y = 2 m with an incidence step of the outer panel. The gap is the "
         "distance between the two trailing-edge nodes at the break, and the strip width is 0.1 m. The "
         "reference is the same geometry with the two surfaces forced into one core group. \"Separate\" "
         "is the same geometry with every surface in its own group: it shows what the cross-surface core "
         "costs at the join. The join tolerance is one strip width, with an upper limit of 0.1 chord.", "",
         table(["incidence step [deg]", "gap [mm]", "gap / strip width", "joined by the rule",
                "CL, rule vs forced join", "CL, separate vs forced join"], rows2), ""]
    return "\n".join(s)


def verdict(ok: bool) -> str:
    return "pass" if ok else "FAIL"


def _closed_panel_mesh(nodes: np.ndarray, panels: np.ndarray):
    from ventorum.panel3d.mesh import PanelMesh

    return PanelMesh(nodes=nodes, panels=panels, component=np.zeros(len(panels), dtype=int),
                     kind=np.full(len(panels), "surface"), wake_edges=np.empty((0, 4), dtype=int))


def _closed_body_solver(nodes: np.ndarray, panels: np.ndarray):
    # The CPU kernels that the solver family selects ("auto"): Numba when it
    # is available. They agree with the numpy reference kernels to round-off.
    from ventorum.panel3d.solver import PanelSolver
    from ventorum.solvers.panel3d import _auto_cpu_backend

    return PanelSolver(_closed_panel_mesh(nodes, panels), far_ratio=5.0, backend=_auto_cpu_backend()[0])


def case_panel3d_sphere() -> str:
    from ventorum.reference.analytic3d import sphere_cp, sphere_quad_mesh

    flows = (("across the pole axis", 0.0), ("along the pole axis", np.pi / 2.0))
    meshes = ((16, 32), (32, 64))
    max_err: dict[str, list[float]] = {name: [] for name, _ in flows}
    rows = []
    for n_lat, n_lon in meshes:
        nodes, panels = sphere_quad_mesh(1.0, n_lat, n_lon)
        solver = _closed_body_solver(nodes, panels)
        conds = [vt.FlightCondition(V_inf=1.0, alpha=a) for _, a in flows]
        for (name, _), cond, res in zip(flows, conds, solver.solve_conditions(conds)):
            err = res.cp - sphere_cp(solver.centroids, 1.0, vt.freestream_direction(float(cond.alpha), 0.0))
            e_max = float(np.max(np.abs(err)))
            e_rms = float(np.sqrt(np.sum(solver.areas * err ** 2) / np.sum(solver.areas)))
            q_inf = 0.5 * float(cond.rho) * cond.V_inf ** 2
            force = float(np.linalg.norm(np.sum(res.cp[:, None] * solver.areas[:, None] * solver.normals, axis=0)
                                         * q_inf))
            max_err[name].append(e_max)
            if n_lat == meshes[-1][0]:
                falls = e_max < max_err[name][0]
                cp_check = verdict(e_max < 0.03 and falls)
            else:
                cp_check = "-"
            rows.append([f"{n_lat} x {n_lon}", name, f"{e_max:.4f}", f"{e_rms:.5f}", cp_check,
                         "below 1e-12" if force < 1e-12 else f"{force:.1e}", verdict(force < 1e-3)])
    s = ["## V11. 3D panel method: sphere", "",
         "Reference: exact potential flow past a sphere, Cp = 1 - (9/4) sin^2(theta), with theta the angle "
         "between the surface point and the onset direction (H. Lamb, Hydrodynamics, 6th ed., 1932, "
         "article 92; Katz and Plotkin, Low-Speed Aerodynamics, 2nd ed., 2001, section 3.12). Evaluation in "
         "`ventorum.reference.analytic3d.sphere_cp`.", "",
         "Unit sphere, latitude-longitude meshes with the poles on the z axis (triangle fans at the poles), "
         "constant source and doublet panels (`PanelSolver`, far-field ratio 5, automatic CPU kernel selection), "
         "V = 1 m/s. With the flow "
         "across the pole axis the poles sit at the points of highest speed; with the flow along the pole "
         "axis they sit at the stagnation points. The Cp error is taken at the panel centroids; the rms "
         "error is area-weighted. The net force must be zero (d'Alembert's paradox).", "",
         "Limits: largest Cp error below 0.03 at 32 x 64 and smaller than at 16 x 32; net force below "
         "1e-3 N.", "",
         table(["mesh", "flow", "largest Cp error", "rms Cp error", "Cp check", "net force [N]",
                "force check"], rows), ""]
    return "\n".join(s)


# Semi-axes [m], onset [m/s] and tip rule of the ellipsoid case. A tip panel
# is a panel where the exact surface normal turns by more than 0.1 rad
# across the panel in both panel directions (the rule of the solver tests).
_ELL_AXES = (1.0, 0.5, 0.25)
_ELL_STREAM = (10.0, 3.0, -2.0)
_ELL_TIP_TURN = 0.1


def _ellipsoid_errors(n_lat: int, n_lon: int) -> tuple[np.ndarray, np.ndarray]:
    from ventorum.reference.analytic3d import ellipsoid_quad_mesh, ellipsoid_surface_velocity

    a, b, c = _ELL_AXES
    nodes, panels = ellipsoid_quad_mesh(a, b, c, n_lat, n_lon, spacing="cosine")
    solver = _closed_body_solver(nodes, panels)
    v_stream = np.asarray(_ELL_STREAM)
    v_mag = float(np.linalg.norm(v_stream))
    res = solver.solve(vt.FlightCondition(V_inf=v_mag, alpha=float(np.arctan2(v_stream[2], v_stream[0])),
                                          beta=float(-np.arcsin(v_stream[1] / v_mag))))
    speed = np.linalg.norm(res.velocity, axis=-1)
    exact = np.linalg.norm(ellipsoid_surface_velocity(solver.centroids, a, b, c, v_stream), axis=-1)
    rel_err = np.abs(speed - exact) / exact
    grad = nodes[panels] / np.array([a ** 2, b ** 2, c ** 2])
    normal = grad / np.linalg.norm(grad, axis=-1, keepdims=True)

    def turn(i: int, j: int) -> np.ndarray:
        return np.arccos(np.clip(np.sum(normal[:, i] * normal[:, j], axis=-1), -1.0, 1.0))

    tip = np.minimum(np.maximum(turn(0, 1), turn(3, 2)), np.maximum(turn(1, 2), turn(0, 3))) > _ELL_TIP_TURN
    return rel_err, tip


def case_panel3d_ellipsoid() -> str:
    levels = (16, 24, 32)
    rows = []
    tip_err = []
    for k, n_lat in enumerate(levels):
        rel_err, tip = _ellipsoid_errors(n_lat, 2 * n_lat)
        away = float(np.max(rel_err[~tip]))
        tip_err.append(float(np.max(rel_err[tip])))
        if k == 0:
            order, order_check = "-", "-"
        else:
            p = np.log(tip_err[k - 1] / tip_err[k]) / np.log(n_lat / levels[k - 1])
            order, order_check = f"{p:.2f}", verdict(p >= 1.0)
        away_check = verdict(away < 0.02) if n_lat == levels[-1] else "-"
        rows.append([f"{n_lat} x {2 * n_lat}", f"{int(np.sum(tip))} of {tip.size}", f"{100 * away:.3f} %",
                     away_check, f"{100 * tip_err[-1]:.3f} %", order, order_check])
    s = ["## V12. 3D panel method: ellipsoid at oblique onset", "",
         "Reference: exact potential flow past a tri-axial ellipsoid (H. Lamb, Hydrodynamics, 6th ed., 1932, "
         "article 114). For a uniform stream the surface velocity is the tangential part of a scaled stream "
         "vector; evaluation in `ventorum.reference.analytic3d.ellipsoid_surface_velocity`.", "",
         "Ellipsoid with semi-axes 1, 0.5 and 0.25 m, onset velocity (10, 3, -2) m/s, latitude-longitude "
         "meshes with cosine spacing in latitude and the poles on the z axis (`PanelSolver`, far-field ratio "
         "5, automatic CPU kernel selection). The error is the relative error of the surface speed at the "
         "panel centroids. A tip panel is a "
         "panel where the exact surface normal turns by more than 0.1 rad across the panel in both panel "
         "directions. At the tips (x = +-a) the surface is curved in both directions and the flat-panel "
         "centroid lies inside the surface by about h^2 / (8 R), with h the panel size and R the radius of "
         "curvature, so these panels are checked for convergence only. The order is the observed order of "
         "the tip error against the latitude count.", "",
         "Limits: largest error away from the tips below 2 % at 32 x 64; observed order of the tip error "
         "at least 1 on each refinement step.", "",
         table(["mesh", "tip panels", "largest error away from the tips", "check", "largest tip error",
                "tip error order", "order check"], rows), ""]
    return "\n".join(s)


def case_panel3d_thin_wing() -> str:
    from ventorum.solvers.factory import make_solver

    cond = vt.FlightCondition(alpha=np.radians(4.0))

    def wing(shape):
        secs = [vt.WingSection(0.0, 1.0, shape=shape), vt.WingSection(1.0, 1.0, shape=shape)]
        ac = vt.Aircraft(surfaces=[vt.LiftingSurface(name="wing", semi_span=4.0, sections=secs)])
        ac.compute_reference_values()
        return ac

    vlm = make_solver("vlm").solve(wing(None), cond,
                                   vt.SolverSettings(solver_type="vlm", n_panels=40, n_chord=16)).totals
    sett = vt.SolverSettings(solver_type="panel3d", n_chord=40, n_panels=40, panel_kutta="morino")
    res = {}
    rows = []
    for shape in ("0006", "0012", "0018", "0024"):
        t = make_solver("panel3d").solve(wing(shape), cond, sett).totals
        res[shape] = (t.CL, t.Cm, t.CDi)
        rows.append([f"NACA {shape}", f"{t.CL:.5f}", f"{t.Cm:.5f}", f"{t.CDi:.6f}"])
    zero = [2.0 * a - b for a, b in zip(res["0006"], res["0012"])]
    rows.append(["zero thickness (from 0006 and 0012)", f"{zero[0]:.5f}", f"{zero[1]:.5f}", f"{zero[2]:.6f}"])
    rows.append(["VLM, 40 x 16", f"{vlm.CL:.5f}", f"{vlm.Cm:.5f}", f"{vlm.CDi:.6f}"])
    rows_d = []
    for name, val, ref, lim in (("CL", zero[0], vlm.CL, 0.01), ("Cm", zero[1], vlm.Cm, 0.03),
                                ("CDi", zero[2], vlm.CDi, 0.03)):
        d = val / ref - 1.0
        rows_d.append([name, pct(d), f"{100 * lim:g} %", verdict(abs(d) < lim)])
    s = ["## V13. 3D panel method: thin-wing limit", "",
         "Reference: as the thickness goes to zero, the potential flow about a thick wing goes to the "
         "lifting-surface solution of the thin wing (Katz and Plotkin, Low-Speed Aerodynamics, 2nd ed., "
         "2001, chapters 9 and 12). The vortex-lattice method of V3 gives an independent solution of the "
         "thin-wing problem. Both are discretized, so the difference also holds the discretization error "
         "of both methods.", "",
         "Rectangular wing, span 8 m, chord 1 m (AR 8), symmetric NACA sections set by the section shape, "
         "alpha = 4 deg. The 3D panel method (`make_solver(\"panel3d\")`) has 40 spanwise panels per "
         "semi-span and 40 chordwise panels with the Morino Kutta condition; CDi is the Trefftz-plane "
         "induced drag. The zero-thickness values are a linear extrapolation in the thickness from NACA "
         "0006 and 0012. The VLM has 40 spanwise panels per semi-span and 16 chordwise panels. Moments "
         "are about the leading edge.", "",
         table(["section", "CL", "Cm", "CDi"], rows), "",
         "Zero-thickness extrapolation against the VLM:", "",
         table(["coefficient", "difference", "limit", "check"], rows_d), ""]
    return "\n".join(s)


def case_panel3d_ground_images() -> str:
    from ventorum.aero.system import ground_normal
    from ventorum.airfoil import naca
    from ventorum.panel3d.kernels import _precompute_geometry, doublet_potential, influence_matrices
    from ventorum.panel3d.mesh import mesh_lifting_surface
    from ventorum.panel3d.post import (
        build_surface_gradient_operator,
        compute_surface_velocity_and_cp,
        integrate_forces_and_moments,
        surface_gradient,
    )
    from ventorum.panel3d.solver import PanelSolver, _standardize_panels
    from ventorum.panel3d.wake import build_wake

    surf = vt.LiftingSurface(name="wing", semi_span=2.0,
                             sections=[vt.WingSection(0.0, 1.0), vt.WingSection(1.0, 1.0)])
    mesh = mesh_lifting_surface(surf, naca("0012", n_points=41, closed_te=True), n_chord=6, n_span=6,
                                half=False)
    alpha, h = np.radians(4.0), 0.5
    cond = vt.FlightCondition(V_inf=15.0, alpha=alpha, h=h)
    solver = PanelSolver(mesh, far_ratio=np.inf)
    res = solver.solve(cond)

    # Explicit mirror body in free air; its wake also follows the free stream.
    k_g = ground_normal(alpha, 0.0, 0.0)
    offset = float(np.dot(solver.ref_point, k_g) - h)
    nodes = mesh.nodes
    nodes_img = nodes - 2.0 * (np.dot(nodes, k_g) - offset)[:, None] * k_g[None, :]
    panels = _standardize_panels(mesh.panels, nodes)
    panels_img = _standardize_panels(panels[:, [0, 3, 2, 1]], nodes_img) + len(nodes)
    nodes_both = np.vstack([nodes, nodes_img])
    panels_both = np.vstack([panels, panels_img])
    n_real = len(panels)
    wake_edges = np.vstack([mesh.wake_edges,
                            mesh.wake_edges + np.array([n_real, n_real, len(nodes), len(nodes)])])
    geom = _precompute_geometry(nodes_both[panels_both])
    wake_dir = vt.freestream_direction(alpha, 0.0)
    wake = build_wake(nodes_both, panels_both, wake_edges, geom.normal, wake_length=solver.wake_length,
                      wake_dir=wake_dir)
    c_wake = doublet_potential(geom.centroid, wake.panels, far_ratio=np.inf)
    b_mat, a_mat = influence_matrices(geom.centroid, nodes_both[panels_both], far_ratio=np.inf)
    for s_i, strip in enumerate(wake.strip_panel_indices):
        cs = np.sum(c_wake[:, strip], axis=1)
        a_mat[:, wake.wake_up[s_i]] += cs
        a_mat[:, wake.wake_lo[s_i]] -= cs
    v_inf = float(cond.V_inf) * wake_dir
    mu = np.linalg.solve(a_mat, -b_mat @ (-np.sum(geom.normal * v_inf, axis=-1)))

    geom_r = _precompute_geometry(nodes[panels])
    grad_op = build_surface_gradient_operator(nodes, panels, geom_r.centroid, geom_r.normal, geom_r.area,
                                              mesh.wake_edges, kinds=mesh.kind)
    _, cp = compute_surface_velocity_and_cp(np.broadcast_to(v_inf, geom_r.centroid.shape), geom_r.normal,
                                            surface_gradient(grad_op, mu[:n_real]), float(cond.V_inf))
    f = integrate_forces_and_moments(cp=cp, areas=geom_r.area, normals=geom_r.normal, centroids=geom_r.centroid,
                                     condition=cond, S_ref=solver.S_ref, b_ref=solver.b_ref, c_ref=solver.c_ref,
                                     ref_point=solver.ref_point)
    lim = 1e-10
    rows = []
    for name, d in (("doublet strength mu", np.max(np.abs(res.mu - mu[:n_real]))),
                    ("Cp", np.max(np.abs(res.cp - cp))),
                    ("CL", abs(res.CL - f["CL"])),
                    ("Cm", abs(res.Cm - f["Cm"]))):
        rows.append([name, "below 1e-12" if d < 1e-12 else f"{d:.1e}", f"{lim:g}", verdict(d < lim)])
    s = ["## V14. 3D panel method: ground images", "",
         "Reference: the method of images. Over a flat ground the flow about the body is the free-air flow "
         "about the body and its mirror image (Katz and Plotkin, Low-Speed Aerodynamics, 2nd ed., 2001, "
         "section 9.4). The image solve of `PanelSolver` must give the same solution as a free-air solve "
         "of the body and an explicit mirror mesh, assembled in this script from the panel influences.", "",
         "Rectangular wing of semi-span 2 m and chord 1 m, NACA 0012, 6 spanwise panels per semi-span and "
         f"6 chordwise panels on each side, mesh of the full span ({n_real} panels), alpha = 4 deg, "
         "h/c = 0.5, exact influences (no far-field expansion). In both models the wake follows the free stream. "
         "The differences are the largest absolute differences on the real body; they must be at round-off level.", "",
         table(["quantity", "largest difference", "limit", "check"], rows), ""]
    return "\n".join(s)


def case_panel2d_karman_trefftz() -> str:
    from ventorum.panel2d.solver import solve_inviscid
    from ventorum.reference.karman_trefftz import karman_trefftz_airfoil, karman_trefftz_solution

    center, tau = (-0.07, 0.04), 10.0
    alphas = (0.0, 4.0, 8.0)
    res = solve_inviscid(karman_trefftz_airfoil(center, tau, n_points=201), list(alphas))
    rows = []
    for i, a in enumerate(alphas):
        _, cl_ex = karman_trefftz_solution(center, tau, n_points=201, alpha_deg=a)
        e_cl = res.Cl[i] / cl_ex - 1.0
        e_p = res.Cl_pressure[i] / res.Cl[i] - 1.0
        rows.append([f"{a:g}", f"{cl_ex:.5f}", f"{res.Cl[i]:.5f}", pct(e_cl), verdict(abs(e_cl) < 1e-3),
                     pct(e_p), verdict(abs(e_p) < 1e-2)])
    rows_cp = []
    errs = []
    levels = (100, 200, 400)
    for k, n in enumerate(levels):
        cp_ex, _ = karman_trefftz_solution(center, tau, n_points=n + 1, alpha_deg=4.0)
        r = solve_inviscid(karman_trefftz_airfoil(center, tau, n_points=n + 1), 4.0)
        errs.append(float(np.max(np.abs(r.Cp[1:-1] - cp_ex[1:-1]))))
        if k == 0:
            order, order_check = "-", "-"
        else:
            p = np.log(errs[k - 1] / errs[k]) / np.log(n / levels[k - 1])
            order, order_check = f"{p:.2f}", verdict(p >= 1.0)
        rows_cp.append([str(n), f"{errs[-1]:.4f}", verdict(errs[-1] < 0.03) if n == 200 else "-",
                        order, order_check])
    s = ["## V15. 2D panel method: Karman-Trefftz section", "",
         "Reference: exact potential flow about a Karman-Trefftz section by conformal mapping of the flow "
         "about a circle, with the Kutta condition at the trailing edge (L. M. Milne-Thomson, Theoretical "
         "Aerodynamics, 4th ed., 1958; Katz and Plotkin, Low-Speed Aerodynamics, 2nd ed., 2001, section "
         "6.6). Evaluation in `ventorum.reference.karman_trefftz`.", "",
         "Circle center (-0.07, 0.04), trailing-edge angle tau = 10 deg, inviscid 2D panel solver "
         "(`ventorum.panel2d.solver.solve_inviscid`) on the exact section nodes. Cl_pressure is the lift "
         "from the integrated surface pressure. Limits: Cl within 0.1 % of exact, Cl_pressure within 1 % "
         "of Cl.", "",
         table(["alpha [deg]", "Cl exact", "Cl (200 panels)", "error", "check", "Cl_pressure vs Cl",
                "check"], rows), "",
         "Largest Cp error at the nodes, alpha = 4 deg, all nodes except the trailing-edge node. At a sharp "
         "trailing edge of angle tau the exact speed is zero, but it rises as r^(tau / (2 pi - tau)) with "
         "the distance r from the edge, so no node value converges to it. The order is the observed order "
         "against the panel count. Limits: error below 0.03 at 200 panels; order at least 1 on each "
         "refinement step.", "",
         table(["panels", "largest Cp error", "check", "order", "order check"], rows_cp), ""]
    return "\n".join(s)


def main(out_path: str | None = None) -> Path:
    t0 = time.perf_counter()
    parts = [
        "# Ventorum verification report", "",
        f"Generated by `validation/run_verification.py` on {datetime.date.today().isoformat()} "
        f"(Python {platform.python_version()}, numpy {np.__version__}). All numbers are computed by "
        "the script; there are no hard-coded results.", "",
        "Verification checks that the solvers solve their equations correctly, against closed forms "
        "and independent solutions of the same theories, and measures the limits of each method. It "
        "is not validation against experiment. Experimental validation needs datasets with full "
        "provenance (see `validation/experimental/README.md`); none are included yet.", "",
    ]
    # The report verifies the reference (CPU) solvers, so its numbers do not
    # depend on the GPU of the machine. The GPU pipelines are checked against
    # them by tests/test_gpu_pipeline.py.
    from ventorum import gpu

    old_device = gpu.get_device()
    gpu.set_device("cpu")
    try:
        for fn in (case_elliptic, case_glauert, case_lifting_surface, case_sweep, case_tapered_straight_le,
                   case_convergence, case_ground, case_camber_moment, case_signs, case_split_surfaces,
                   case_panel3d_sphere, case_panel3d_ellipsoid, case_panel3d_thin_wing,
                   case_panel3d_ground_images, case_panel2d_karman_trefftz):
            parts.append(fn())
    finally:
        gpu.set_device(old_device)
    parts.append(f"Run time: {time.perf_counter() - t0:.1f} s.")
    out = Path(out_path) if out_path else ROOT / "docs" / "verification_report.md"
    out.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    print(main(sys.argv[1] if len(sys.argv) > 1 else None))
