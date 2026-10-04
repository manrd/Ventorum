# Reference case template

Copy this file to `<case-id>.md` and fill every field. The file holds free
Markdown text for people (this part), and exactly one fenced code block
with the info string `toml` (below). Only that block is read, with
`tomllib` (Python standard library). No new dependency.

Rules:

- `identity.id` must equal the file name without `.md`.
- `entry_version` is an integer. Raise it when you correct the entry.
- `source.type` is `experiment`, `cfd` or `theory`. `source.grade` is `A`,
  `B` or `C`: A is an experiment with stated uncertainty and documented
  corrections; B is an experiment without stated uncertainty, or high-order
  CFD with a grid study; C is digitised plots or one unchecked source.
  A result of another low-order tool is never a reference.
- A `theory` case has no `grade`: the parser refuses a grade on it, and the
  runner shows it in its own section and never counts it in the statistics.
- `geometry` is the aircraft, in the structure that
  `build_aircraft_from_spec` reads: one surface object (as below), or an
  object with `surfaces`. Lengths are in metres, angles in degrees.
- `conditions`: `mach`, `reynolds`, `beta_deg` (degrees), `h_m` (optional
  height in metres), `corrections` (write `none stated` when the source
  states none).
- `[[results]]`: `quantity` is `CL`, `CD`, `CDi`, `Cm`, `CY`, `Cl` or `Cn`.
  `alpha_deg` and `values` are lists of the same length. `uncertainty` is a
  number or a list; `uncertainty_kind` is `absolute` or `relative` and is
  required with `uncertainty`. `moment_point` (3 values in metres) is
  required for moments (`Cm`, `Cl`, `Cn`). `h_m` is an optional list of
  heights in metres, one for each point (each value must be above 0); a
  result with this list solves each point at its own height, and the report
  shows a `h [m]` column. Without the list every point uses the case height
  `[conditions] h_m`. `height_point` (3 values in metres) names the point of
  the model whose height `h_m` gives; without it the height is the height of
  the moment reference point of the solve. A `height_point` without a `h_m`
  list is refused. `extraction` is `table` or `digitised`;
  `digitising_error` is required when digitised.
- The parser refuses a file with a missing required field, a value that
  starts with `TODO`, lists of different lengths, an unknown quantity, a
  `h_m` list with a length different from `alpha_deg` or a value that is
  not above 0, a `height_point` without 3 values, or a `height_point`
  without a `h_m` list.
- Take the values from a table when one exists. For a figure, record the
  digitizing tool and the estimated reading error.
- The example values below are illustrative only (`synthetic = true`).

```toml
[identity]
id = "template_example"
title = "Example case (illustrative values, not real data)"
entry_version = 1

[source]
citation = "A. Author and B. Author, Title of the report, Report Series 1234, 1961"
doi_or_url = "https://example.org/report-1234"
location = "Table II, page 12"
type = "experiment"
grade = "A"
synthetic = true

[geometry]
span_m = 6.0
root_chord_m = 1.2
tip_chord_m = 1.2

[conditions]
mach = 0.1
reynolds = 1000000.0
beta_deg = 0.0
corrections = "none stated"

[[results]]
quantity = "CL"
alpha_deg = [0.0, 2.0, 4.0]
values = [0.0, 0.25, 0.5]
uncertainty = 0.01
uncertainty_kind = "absolute"
extraction = "table"

[[results]]
quantity = "Cm"
alpha_deg = [0.0, 4.0]
values = [0.0, 0.02]
uncertainty = 0.05
uncertainty_kind = "relative"
moment_point = [0.3, 0.0, 0.0]
extraction = "digitised"
digitising_error = 0.005

[quality]
notes = "Example notes: tunnel, mounting, transition."
```
