# Reference datasets

## New Markdown reference format (documented)

A reference case is one file `<case-id>.md`: free Markdown text for people,
and exactly one fenced `toml` code block that Ventorum reads. The template
with every field and short instructions is
`validation/reference_template.md`. The parser is
`ventorum.reference.database` (`load_case`, `load_database`).

Run the cases with:

```bash
python validation/run_validation.py --db FOLDER --out REPORT.md [--solver vlm]
```

The folder can also come from the environment variable
`VENTORUM_REFERENCE_DB`. `--out` is required; the report path must be outside
the repository (the runner warns when it is inside). The report traces
every value to its source. Synthetic cases are marked and never counted.
Cases above Mach 0.3 are listed as outside the envelope and not run.
A `theory` case has no `grade`: it is verification, so the runner shows it
in its own section and never counts it in the summary statistics.

Reference cases are private until their licence is clear: they never enter
this repository.

## Old JSON format (kept working)

The older JSON format below is kept working; the Markdown format above is
the documented one.

This folder holds the experimental data for validation. It is empty on
purpose.

## Why it is empty

The earlier validation suite contained "experimental" arrays that could not be
traced to a table or figure. Some are exact functions of the angle of attack,
which measured data are not: for example, the "NASA TP-3151" induced drag is
CDi = 0.0003 alpha^2 (alpha in degrees) at all seven points. They were removed
(see the git history of `validation/run_ntrs_validation_suite.py`). The archives (NTRS, the Cranfield NACA collection)
were not reachable from the environment where the suite was rebuilt, so no
values were transcribed. Values typed from memory are not accepted.

## How to add a dataset

1. Copy `TEMPLATE.json` to a new file, for example `fink_lastinger_tnd926_ar2.json`.
2. Fill every field. The loader (`ventorum.reference.experimental`) refuses a
   file with a missing field or a `TODO` value.
3. Take the values from a table when one exists. For a figure, record the
   digitizing tool and the estimated reading error in `transcription`.
4. Record the test conditions. Low-speed wing data depend on the Reynolds
   number (transition, maximum lift); ground-effect data depend on the ground
   simulation (fixed board or moving belt) and on the wing thickness.

## Candidate sources (to be transcribed from the documents)

* M. P. Fink and J. L. Lastinger, "Aerodynamic characteristics of
  low-aspect-ratio wings in close proximity to the ground", NASA TN D-926, 1961
  (thick, cambered rectangular wings, AR 1 to 4, ground effect).
* Low-speed tests of 45 deg swept wings, for example ARC R&M 2882
  (Weber and Brebner), and NACA RM A50K28 (Kolbe and Boltz) for a swept,
  tapered wing of aspect ratio 3.
* Rectangular-wing tests in NACA reports with tabulated force data.

Note: the thin-wing, inviscid models of this package do not represent
thickness or viscous effects. Near the ground, thick sections can lose lift
(the gap flow accelerates under the wing); a comparison with such data shows
the limit of the model, not a fit to be tuned.
