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
- Derivative quantities: `quantity` can also be `<C>_<v>`, with `<C>` one
  of `CL`, `CD`, `CY`, `Cl`, `Cm`, `Cn` and `<v>` one of `alpha`, `beta`
  (for example `Cl_beta`). A derivative quantity needs `derivative_unit`:
  `1/deg` or `1/rad`, the unit of the stored values. A moment derivative
  (`Cl_*`, `Cm_*`, `Cn_*`) needs `moment_point`, and takes `axes`: `body`,
  `stability` or `wind` (default `stability`, as NACA stability-tunnel
  data). `derivative_unit` and `axes` are refused on other quantities.
  Rate and control derivatives are not in the format.
- Abscissa: a result has either `alpha_deg` (list of angles in degrees) or
  `CL` (list of lift coefficients), never both. With `CL`, the runner finds
  the angle of attack where Ventorum gives that lift coefficient (alpha
  sweep and linear interpolation, one mesh) and evaluates the quantity
  there. A lift coefficient beyond the Ventorum range of the sweep is
  reported as "not reached"; it is never extrapolated.
- Slope entries: `slope_of = "CL"` and `CL_range = [lo, hi]` (lo < hi) give
  exactly one value: the slope d(quantity)/dCL over that lift range. A slope
  entry has no `alpha_deg` or `CL` list. The runner evaluates the quantity
  at 5 equally spaced lift coefficients of the range, fits a straight line
  (least squares), and reports the fitted slope and the largest residual.
  For a derivative quantity, `derivative_unit` is the unit of the quantity:
  for example a slope in 1/deg means d(Cl_beta [1/deg])/dCL.
- Signs follow Ventorum: Cl > 0 right wing down, beta > 0 for the wind from
  the right. For a source in yaw angle psi = -beta, store the converted
  value (for example Cl_beta = -Cl_psi) and say so in the case text and in
  `[quality] notes`.
- The parser refuses a file with a missing required field, a value that
  starts with `TODO`, lists of different lengths, an unknown quantity, a
  `h_m` list with a length different from `alpha_deg` or a value that is
  not above 0, a `height_point` without 3 values, or a `height_point`
  without a `h_m` list. It also refuses a derivative quantity without
  `derivative_unit`, an unknown derivative variable, both `alpha_deg` and
  `CL` in one result, a slope entry with more than one value or without
  `CL_range`, and a `CL_range` with lo >= hi.
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

[[results]]
quantity = "Cl_beta"
CL = [0.2, 0.4]
values = [-0.0005, -0.001]
derivative_unit = "1/deg"
axes = "stability"
moment_point = [0.3, 0.0, 0.0]
extraction = "digitised"
digitising_error = 0.0002

[[results]]
quantity = "Cl_beta"
slope_of = "CL"
CL_range = [0.0, 0.4]
values = [-0.0025]
derivative_unit = "1/deg"
moment_point = [0.3, 0.0, 0.0]
extraction = "digitised"
digitising_error = 0.0003

[quality]
notes = "Example notes: tunnel, mounting, transition."
```
