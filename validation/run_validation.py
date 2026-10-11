# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Run the Markdown reference cases and write a validation report.

Each case file gives reference values with their source. For each case and
quantity the runner solves Ventorum at each angle of attack and writes a table:
alpha, reference value, its uncertainty, Ventorum value, difference, and
whether the difference is inside the uncertainty. A result with its own
``h_m`` list solves each point at its own height (the table then shows a
``h [m]`` column); else every point uses the case height ``[conditions]
h_m``.

A derivative quantity (for example ``Cl_beta``) is computed by central
differences (alpha 0.5 deg, beta 1 deg, one mesh for the cases of one
point) and given in the ``derivative_unit`` of the result; a moment
derivative uses the ``axes`` of the result. A result with the ``CL``
abscissa finds the angle of attack where Ventorum gives each lift
coefficient (alpha sweep from -10 to 20 deg and linear interpolation) and
evaluates the quantity there; the table then shows the CL and the alpha
found. A lift coefficient outside the CL range of the sweep is reported as
"not reached" and is not counted. A slope entry (``slope_of = "CL"``)
evaluates the quantity at 5 equally spaced lift coefficients of
``CL_range``, fits a straight line by least squares, and reports the
fitted slope and the largest residual.

Synthetic cases are marked and are never counted in the summary statistics.
Theory cases are shown in their own section "Theory cases (verification,
not counted)" and are never counted: they are verification, not validation.
Cases above Mach 0.3 are listed as "outside the envelope" and are not run.

Usage:
    python validation/run_validation.py --db FOLDER --out REPORT.md [--solver vlm]

The folder can also come from the environment variable VENTORUM_REFERENCE_DB.
The default output path is outside the repository (--out is required). The
runner prints a warning if the output path is inside the repository folder.
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
from pathlib import Path

import numpy as np

import ventorum as vt
from ventorum.aero.system import ground_normal
from ventorum.reference.database import MACH_LIMIT, ReferenceError, load_case
from ventorum.reference.derivatives import (
    CL_SWEEP_ALPHA_DEG,
    SLOPE_POINTS,
    alpha_for_cl,
    central_difference,
    coefficient,
    is_derivative,
    per_rad_to_unit,
    slope_fit,
    split_derivative,
)

ROOT = Path(__file__).resolve().parents[1]

SPEED_OF_SOUND_M_S = 340.3
RHO_KG_M3 = 1.225
N_PANELS = 40
SOLVERS = ("auto", "vlm", "linear", "nonlinear", "fourier")


def table(header, rows):
    """Format a Markdown table."""
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


#: Start of the error text of a lift coefficient that the alpha sweep does not reach.
NOT_REACHED = "not reached"


def ventorum_value(totals, quantity):
    """Read one coefficient from integrated totals (CD falls back to CDi)."""
    return coefficient(totals, quantity)


def point_height(case, result, h_point, alpha_deg):
    """Height [m] of the solve reference point for one result point.

    The listed height ``h_point`` [m] is the height of ``height_point`` (or
    of the moment reference point of the solve when the result has no
    ``height_point``). The returned height is the height of the moment
    reference point of the solve: ``h_ref = h_point + dot(ref - anchor, k)``
    with the ground normal ``k`` at the alpha and beta of the point.
    """
    if result.moment_point is not None:
        solve_ref = np.array(result.moment_point, dtype=float)
    else:
        solve_ref = case.aircraft.moment_reference()
    if result.height_point is not None:
        anchor = np.array(result.height_point, dtype=float)
    else:
        anchor = solve_ref
    k = ground_normal(float(np.radians(alpha_deg)), float(np.radians(case.beta_deg)))
    return float(h_point + float((solve_ref - anchor) @ k))


def solve_aircraft(case, result):
    """Return the aircraft of one result (a copy with the moment point when it has one)."""
    if result.moment_point is None:
        return case.aircraft
    aircraft = case.aircraft.clone()
    aircraft.ref_point = np.array(result.moment_point, dtype=float)
    return aircraft


def condition_factory(case, h_ref):
    """Return ``make(alpha_deg, beta_deg)``, a flight condition at the height ``h_ref`` [m] or in free air."""

    def make(alpha_deg, beta_deg):
        return vt.FlightCondition(
            V_inf=case.mach * SPEED_OF_SOUND_M_S,
            alpha=float(np.radians(alpha_deg)),
            beta=float(np.radians(beta_deg)),
            rho=RHO_KG_M3,
            h=h_ref,
        )

    return make


def evaluate(case, result, aircraft, alpha_deg, h_ref, settings):
    """Ventorum value of the quantity of *result* at one angle of attack [deg].

    A plain quantity comes from one solve. A derivative quantity comes from
    central differences (``ventorum.reference.derivatives.central_difference``)
    at the constant height ``h_ref`` [m] of the solve reference point, and
    is converted to ``result.derivative_unit``. Returns ``(value, note)``.
    """
    make = condition_factory(case, h_ref)
    if is_derivative(result.quantity):
        value = central_difference(
            aircraft, result.quantity, alpha_deg, case.beta_deg, make, settings, axes=result.axes
        )
        return per_rad_to_unit(value, result.derivative_unit), ""
    totals = vt.analyze(aircraft, condition=make(alpha_deg, case.beta_deg), settings=settings).totals
    return ventorum_value(totals, result.quantity)


class NotReached(Exception):
    """A lift coefficient outside the Ventorum CL range of the alpha sweep."""


def value_at_cl(case, result, aircraft, settings, cl_target, h_point):
    """Find the alpha where Ventorum gives *cl_target* and evaluate the quantity there.

    The alpha sweep and the search use one mesh. With a listed height
    ``h_point`` [m], each sweep angle uses the converted height of the
    solve reference point (``point_height``); sweep angles where that
    height is not > 0 are left out. A plain quantity is evaluated with the
    mesh of the sweep; a derivative uses one mesh for its own cases.
    Returns ``(alpha_deg, value, note)``. Raises ``NotReached`` when the
    target is outside the CL range of the sweep (no extrapolation).
    """
    if h_point is None:
        grid = list(CL_SWEEP_ALPHA_DEG)
        make_sweep = condition_factory(case, None)
    else:
        grid = [a for a in CL_SWEEP_ALPHA_DEG if point_height(case, result, h_point, a) > 0.0]

        def make_sweep(alpha_deg, beta_deg):
            h_ref = point_height(case, result, h_point, alpha_deg)
            return condition_factory(case, h_ref)(alpha_deg, beta_deg)

    alpha_deg, (cl_lo, cl_hi), sweep_settings = alpha_for_cl(
        aircraft, cl_target, case.beta_deg, make_sweep, settings, alpha_grid_deg=grid
    )
    if alpha_deg is None:
        raise NotReached(
            f"{NOT_REACHED}: CL {cl_target:g} is outside the Ventorum CL range of the alpha sweep "
            f"({cl_lo:.4f} to {cl_hi:.4f}); no extrapolation"
        )
    h_ref = None if h_point is None else point_height(case, result, h_point, alpha_deg)
    point_settings = settings if is_derivative(result.quantity) else sweep_settings
    value, note = evaluate(case, result, aircraft, alpha_deg, h_ref, point_settings)
    return alpha_deg, value, note


def run_cl_result(case, result, solver):
    """Rows of a result with the ``CL`` abscissa.

    Each row is ``(CL, reference, uncertainty, Ventorum, difference,
    inside, note, error, alpha_deg)``. A lift coefficient that the alpha
    sweep does not reach gives an error text that starts with
    ``not reached``.
    """
    heights = list(result.h_m) if result.h_m is not None else [case.h_m] * len(result.values)
    uncs = result.abs_uncertainty if result.abs_uncertainty is not None else [None] * len(result.values)
    aircraft = solve_aircraft(case, result)
    settings = vt.SolverSettings(solver_type=solver, n_panels=N_PANELS)
    rows = []
    for h_point, cl_target, ref, unc in zip(heights, result.CL, result.values, uncs):
        try:
            alpha_deg, got, note = value_at_cl(case, result, aircraft, settings, cl_target, h_point)
        except NotReached as exc:
            rows.append((cl_target, ref, unc, None, None, "n/a", "", str(exc), None))
            continue
        except Exception as exc:  # noqa: BLE001 - one bad point must not stop the report
            rows.append((cl_target, ref, unc, None, None, "n/a", "", f"{type(exc).__name__}: {exc}", None))
            continue
        diff = got - ref
        inside = "n/a" if unc is None else ("yes" if abs(diff) <= unc else "no")
        rows.append((cl_target, ref, unc, got, diff, inside, note, None, alpha_deg))
    return rows


def run_slope_result(case, result, solver):
    """Row of a slope entry (``slope_of = "CL"``).

    The quantity is evaluated at 5 equally spaced lift coefficients of
    ``CL_range`` and a straight line is fitted (least squares). The single
    row is ``(CL_range, reference, uncertainty, fitted slope, difference,
    inside, note, error, largest residual)``; the residual is in the unit
    of the quantity.
    """
    h_point = result.h_m[0] if result.h_m is not None else case.h_m
    unc = result.abs_uncertainty[0] if result.abs_uncertainty is not None else None
    ref = result.values[0]
    lo, hi = result.CL_range
    aircraft = solve_aircraft(case, result)
    settings = vt.SolverSettings(solver_type=solver, n_panels=N_PANELS)
    targets = [float(c) for c in np.linspace(lo, hi, SLOPE_POINTS)]
    values = []
    notes = set()
    try:
        for cl_target in targets:
            _, value, note = value_at_cl(case, result, aircraft, settings, cl_target, h_point)
            values.append(value)
            if note:
                notes.add(note)
    except NotReached as exc:
        return [((lo, hi), ref, unc, None, None, "n/a", "", str(exc), None)]
    except Exception as exc:  # noqa: BLE001 - one bad entry must not stop the report
        return [((lo, hi), ref, unc, None, None, "n/a", "", f"{type(exc).__name__}: {exc}", None)]
    slope, residual = slope_fit(targets, values)
    diff = slope - ref
    inside = "n/a" if unc is None else ("yes" if abs(diff) <= unc else "no")
    return [((lo, hi), ref, unc, slope, diff, inside, "; ".join(sorted(notes)), None, residual)]


def run_result(case, result, solver):
    """Solve Ventorum at each alpha of one result. Returns rows for the report.

    A result with the ``CL`` abscissa goes to ``run_cl_result`` and a slope
    entry to ``run_slope_result``. A derivative quantity is computed by
    central differences and given in its ``derivative_unit``.

    A result with its own ``h_m`` list solves each point at its own height;
    else every point uses ``case.h_m``. A listed height is the height of
    ``height_point`` (or of the moment reference point of the solve when the
    result has no ``height_point``); it is converted to the height of the
    solve reference point with the ground normal. A point whose converted
    height is not > 0 is reported as an error; the other points still run.
    """
    if result.slope_of is not None:
        return run_slope_result(case, result, solver)
    if result.CL is not None:
        return run_cl_result(case, result, solver)
    if result.h_m is not None:
        heights = list(result.h_m)
    else:
        heights = [case.h_m] * len(result.values)
    rows = []
    for h_point, alpha_deg, ref, unc in zip(
        heights,
        result.alpha_deg,
        result.values,
        result.abs_uncertainty if result.abs_uncertainty is not None else [None] * len(result.values),
    ):
        h_ref = None
        if h_point is not None:
            h_ref = point_height(case, result, h_point, alpha_deg)
            if h_ref <= 0.0:
                rows.append(
                    (
                        alpha_deg,
                        ref,
                        unc,
                        None,
                        None,
                        "n/a",
                        "",
                        f"ReferenceError: converted height of the solve reference point "
                        f"({h_ref:g} m) is not > 0 for listed height {h_point:g} m",
                    )
                )
                continue
        aircraft = solve_aircraft(case, result)
        settings = vt.SolverSettings(solver_type=solver, n_panels=N_PANELS)
        try:
            got, note = evaluate(case, result, aircraft, alpha_deg, h_ref, settings)
            diff = got - ref
            inside = "n/a" if unc is None else ("yes" if abs(diff) <= unc else "no")
            rows.append((alpha_deg, ref, unc, got, diff, inside, note, None))
        except Exception as exc:  # noqa: BLE001 - one bad point must not stop the report
            rows.append((alpha_deg, ref, unc, None, None, "n/a", "", f"{type(exc).__name__}: {exc}"))
    return rows


def case_section(case, solver, counted, heading="##"):
    """Report section of one case inside the envelope. Appends to counted.

    A theory case is shown with heading level 3 under the theory section
    and is never counted in the statistics.
    """
    grade_text = f" Grade: {case.grade}." if case.grade is not None else ""
    parts = [
        f"{heading} {case.id}",
        "",
        f"File: `{case.file}`. Title: {case.title}.",
        "",
        f"Source: {case.citation} ({case.location}). Type: {case.source_type}.{grade_text}",
        "",
        f"Conditions: Mach {case.mach:g}, Re {case.reynolds:g}, beta {case.beta_deg:g} deg"
        + (f", h {case.h_m:g} m" if case.h_m is not None else "")
        + f". Corrections: {case.corrections}.",
        "",
    ]
    if case.synthetic:
        parts += ["This is a synthetic case: it is shown but never counted in summary statistics.", ""]
    for result in case.results:
        moment = (
            f", moment point [{', '.join(f'{v:g}' for v in result.moment_point)}] m"
            if result.moment_point is not None
            else ""
        )
        anchor = (
            f", height point [{', '.join(f'{v:g}' for v in result.height_point)}] m"
            if result.height_point is not None
            else ""
        )
        parts += [f"### {result_heading(result)} ({result.extraction}{moment}{anchor}{derivative_text(result)})", ""]
        rows = run_result(case, result, solver)
        h_head = ["h [m]"] if result.h_m is not None else []
        fv = value_format(result)
        if result.slope_of is not None:
            header = ["CL range"] + h_head + [
                "reference", "uncertainty", "fitted slope", "difference", "inside", "largest residual"]
        elif result.CL is not None:
            header = ["CL", "alpha [deg]"] + h_head + [
                "reference", "uncertainty", "Ventorum", "difference", "inside"]
        else:
            header = ["alpha [deg]"] + h_head + ["reference", "uncertainty", "Ventorum", "difference", "inside"]
        table_rows = []
        for point_index, row in enumerate(rows):
            abscissa, ref, unc, got, diff, inside, note, error = row[:8]
            h_cell = [f"{result.h_m[point_index]:g}"] if result.h_m is not None else []
            if result.slope_of is not None:
                first = [f"{abscissa[0]:g} to {abscissa[1]:g}"]
            elif result.CL is not None:
                first = [f"{abscissa:g}", "-" if row[8] is None else f"{row[8]:.4f}"]
            else:
                first = [f"{abscissa:g}"]
            plain = result.derivative_unit is None and result.slope_of is None
            unc_cell = _fmt_unc(unc) if plain or unc is None else fv(unc)
            if error is not None:
                status = NOT_REACHED if error.startswith(NOT_REACHED) else "error"
                cells = first + h_cell + [fv(ref), unc_cell, status, "-", "n/a"]
                if result.slope_of is not None:
                    cells.append("-")
            else:
                mark = f"{fv(got)} ({note})" if note else fv(got)
                cells = first + h_cell + [fv(ref), unc_cell, mark, fv(diff, sign=True), inside]
                if result.slope_of is not None:
                    cells.append(f"{row[8]:.3e}")
            table_rows.append(cells)
            if error is None:
                if not case.synthetic and case.source_type != "theory":
                    counted["points"] += 1
                    if unc is not None:
                        counted["with_uncertainty"] += 1
                        if inside == "yes":
                            counted["inside"] += 1
                if case.synthetic:
                    counted["synthetic_points"] += 1
                if case.source_type == "theory":
                    counted["theory_points"] += 1
        parts += [
            table(header, table_rows),
            "",
        ]
        if result.slope_of is not None:
            parts += [
                f"The fitted slope is the least-squares line through {SLOPE_POINTS} equally spaced CL "
                "of the range; the largest residual is in the unit of the quantity.",
                "",
            ]
        errors = [(r[0], r[7]) for r in rows if r[7] is not None]
        for abscissa, message in errors:
            if message.startswith(NOT_REACHED):
                where = "slope" if result.slope_of is not None else f"point at CL {abscissa:g}"
                parts += [f"The {where} is {message}.", ""]
            elif result.slope_of is not None:
                parts += [f"Slope over CL {abscissa[0]:g} to {abscissa[1]:g} failed: {message}.", ""]
            elif result.CL is not None:
                parts += [f"Point at CL {abscissa:g} failed: {message}.", ""]
            else:
                parts += [f"Point at alpha {abscissa:g} deg failed: {message}.", ""]
    if case.notes:
        parts += [f"Quality notes: {case.notes}", ""]
    return "\n".join(parts)


def _fmt_unc(unc):
    return "n/a" if unc is None else f"{unc:.6f}"


def result_heading(result):
    """Heading text of one result: the quantity, or d(quantity)/dCL for a slope entry."""
    if result.slope_of is not None:
        return f"d({result.quantity})/d{result.slope_of}"
    return result.quantity


def derivative_text(result):
    """Heading text of the unit and the axes of a derivative quantity (empty for others)."""
    if result.derivative_unit is None:
        return ""
    parts = split_derivative(result.quantity)
    axes = f", {result.axes} axes" if parts is not None and parts[0] in ("Cl", "Cm", "Cn") else ""
    return f"{axes}, {result.quantity} in {result.derivative_unit}"


def value_format(result):
    """Return the number format of one result.

    Plain quantities keep 6 decimals. Derivative quantities and slopes use
    6 significant digits, because their values per degree are small.
    """
    small = result.derivative_unit is not None or result.slope_of is not None

    def fmt(value, sign=False):
        if small:
            return f"{value:+.6g}" if sign else f"{value:.6g}"
        return f"{value:+.6f}" if sign else f"{value:.6f}"

    return fmt


def build_report(db_folder, cases, outside, errors, solver):
    """Assemble the full report text. Returns (text, counted)."""
    counted = {
        "cases": 0,
        "points": 0,
        "with_uncertainty": 0,
        "inside": 0,
        "synthetic_points": 0,
        "theory_points": 0,
    }
    n_synthetic = 0
    n_theory = 0
    parts = [
        "# Ventorum validation report",
        "",
        f"Generated by `validation/run_validation.py` on {datetime.date.today().isoformat()}. "
        f"Database: `{db_folder}`. Solver: `{solver}` ({N_PANELS} panels per semi-span). "
        f"Free-stream speed follows the case Mach number with {SPEED_OF_SOUND_M_S:g} m/s "
        f"for the speed of sound (15 C, ISA sea level); coefficients do not depend on it. "
        "CD uses CD_total (with profile drag when the geometry gives cd0, else CDi).",
        "",
    ]
    theory_cases = [case for case in cases if case.source_type == "theory"]
    for case in cases:
        if case.source_type == "theory":
            continue  # theory cases are shown in their own section below
        if not case.synthetic:
            counted["cases"] += 1
        else:
            n_synthetic += 1
        parts.append(case_section(case, solver, counted))
    if theory_cases:
        parts += ["## Theory cases (verification, not counted)", ""]
        for case in theory_cases:
            if case.synthetic:
                n_synthetic += 1
            n_theory += 1
            parts.append(case_section(case, solver, counted, heading="###"))
    parts += [
        "## Summary",
        "",
        f"Reference points counted: {counted['points']} "
        f"({counted['inside']} of {counted['with_uncertainty']} with uncertainty inside it). "
        f"Cases counted: {counted['cases']}. Synthetic cases shown: {n_synthetic} "
        f"({counted['synthetic_points']} points, not counted). "
        f"Theory cases shown: {n_theory} ({counted['theory_points']} points, not counted).",
        "",
    ]
    if outside:
        parts += ["## Outside the envelope (not run)", ""]
        parts += [
            table(
                ["case", "mach", "reason"],
                [
                    [c.id, f"{c.mach:g}", f"Mach above {MACH_LIMIT:g} (outside the envelope), not run"]
                    for c in outside
                ],
            )
            + "\n"
        ]
        parts += [""]
    if errors:
        parts += ["## Files that could not be read", ""]
        parts += [table(["file", "error"], [[f, m] for f, m in errors]), ""]
    return "\n".join(parts) + "\n", counted


def main(argv=None):
    """Run the validation and write the report. Returns the report path."""
    parser = argparse.ArgumentParser(description="Run the Markdown reference cases and write a report.")
    parser.add_argument("--db", default=os.environ.get("VENTORUM_REFERENCE_DB"),
                        help="Folder with the <case-id>.md files (or VENTORUM_REFERENCE_DB).")
    parser.add_argument("--out", required=True, help="Output report path (outside the repository).")
    parser.add_argument("--solver", default="vlm", choices=list(SOLVERS), help="Solver to run.")
    args = parser.parse_args(argv)
    if not args.db:
        parser.error("no database folder: give --db or set VENTORUM_REFERENCE_DB.")
    db_folder = Path(args.db)
    if not db_folder.is_dir():
        parser.error(f"database folder {db_folder} does not exist.")
    out = Path(args.out)
    try:
        if out.resolve() == ROOT or ROOT in out.resolve().parents:
            print(f"Warning: output path {out} is inside the repository folder.", file=sys.stderr)
    except OSError:
        pass

    cases, outside, errors = [], [], []
    for path in sorted(db_folder.glob("*.md")):
        try:
            case = load_case(path)
        except ReferenceError as exc:
            errors.append((path.name, str(exc)))
            continue
        if case.in_envelope:
            cases.append(case)
        else:
            outside.append(case)

    text, _ = build_report(db_folder, cases, outside, errors, args.solver)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(str(out))
    return out


if __name__ == "__main__":
    main()
