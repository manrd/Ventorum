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
    s = ["## V1. Elliptic wing (Lanchester-Prandtl lifting-line theory)", "",
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
                sol = vt.HorseshoeSolver() if s == "vlm" else vt.LinearLLTSolver()
                try:
                    r = sol.solve(wg, vt.FlightCondition(alpha=np.radians(4.0)),
                                  vt.SolverSettings(solver_type=s, n_panels=24), ground=gp_s)
                    out[s] = r.totals.CL / oge[s] - 1.0
                except ValueError:
                    out[s] = float("nan")
            ll = "refused" if np.isnan(out["linear"]) else pct(out["linear"])
            rows_ll.append([f"{AR:g}", f"{hc:g}", pct(out["vlm"]), ll])

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
                   case_convergence, case_ground, case_camber_moment, case_signs, case_split_surfaces):
            parts.append(fn())
    finally:
        gpu.set_device(old_device)
    parts.append(f"Run time: {time.perf_counter() - t0:.1f} s.")
    out = Path(out_path) if out_path else ROOT / "docs" / "verification_report.md"
    out.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    print(main(sys.argv[1] if len(sys.argv) > 1 else None))
