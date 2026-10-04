# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Run the Markdown reference cases and write a validation report.

Each case file gives reference values with their source. For each case and
quantity the runner solves Ventorum at each angle of attack and writes a table:
alpha, reference value, its uncertainty, Ventorum value, difference, and
whether the difference is inside the uncertainty. A result with its own
``h_m`` list solves each point at its own height (the table then shows a
``h [m]`` column); else every point uses the case height ``[conditions]
h_m``.

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


def ventorum_value(totals, quantity):
    """Read one coefficient from integrated totals (CD falls back to CDi)."""
    if quantity == "CD":
        if totals.CD_total is not None:
            return float(totals.CD_total), ""
        return float(totals.CDi), "CDi (no profile drag)"
    return float(getattr(totals, quantity)), ""


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


def run_result(case, result, solver):
    """Solve Ventorum at each alpha of one result. Returns rows for the report.

    A result with its own ``h_m`` list solves each point at its own height;
    else every point uses ``case.h_m``. A listed height is the height of
    ``height_point`` (or of the moment reference point of the solve when the
    result has no ``height_point``); it is converted to the height of the
    solve reference point with the ground normal. A point whose converted
    height is not > 0 is reported as an error; the other points still run.
    """
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
        aircraft = case.aircraft
        if result.moment_point is not None:
            aircraft = case.aircraft.clone()
            aircraft.ref_point = np.array(result.moment_point, dtype=float)
        condition = vt.FlightCondition(
            V_inf=case.mach * SPEED_OF_SOUND_M_S,
            alpha=float(np.radians(alpha_deg)),
            beta=float(np.radians(case.beta_deg)),
            rho=RHO_KG_M3,
            h=h_ref,
        )
        settings = vt.SolverSettings(solver_type=solver, n_panels=N_PANELS)
        try:
            totals = vt.analyze(aircraft, condition=condition, settings=settings).totals
            got, note = ventorum_value(totals, result.quantity)
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
        parts += [f"### {result.quantity} ({result.extraction}{moment}{anchor})", ""]
        rows = run_result(case, result, solver)
        if result.h_m is not None:
            header = ["alpha [deg]", "h [m]", "reference", "uncertainty", "Ventorum", "difference", "inside"]
        else:
            header = ["alpha [deg]", "reference", "uncertainty", "Ventorum", "difference", "inside"]
        table_rows = []
        for point_index, (alpha_deg, ref, unc, got, diff, inside, note, error) in enumerate(rows):
            h_cell = [f"{result.h_m[point_index]:g}"] if result.h_m is not None else []
            if error is not None:
                table_rows.append([f"{alpha_deg:g}"] + h_cell + [f"{ref:.6f}", _fmt_unc(unc), "error", "-", "n/a"])
            else:
                mark = f"{got:.6f} ({note})" if note else f"{got:.6f}"
                table_rows.append(
                    [f"{alpha_deg:g}"] + h_cell + [f"{ref:.6f}", _fmt_unc(unc), mark, f"{diff:+.6f}", inside]
                )
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
        errors = [(r[0], r[7]) for r in rows if r[7] is not None]
        for alpha_deg, message in errors:
            parts += [f"Point at alpha {alpha_deg:g} deg failed: {message}.", ""]
    if case.notes:
        parts += [f"Quality notes: {case.notes}", ""]
    return "\n".join(parts)


def _fmt_unc(unc):
    return "n/a" if unc is None else f"{unc:.6f}"


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
