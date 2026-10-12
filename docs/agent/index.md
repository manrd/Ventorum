# Agent guide

Ventorum has a tool interface for AI agents and automated design loops: Python calls, a command line and an MCP server. The interface is strict, so that an agent finds its mistakes at once.

## Start the server

- MCP (JSON-RPC 2.0 over stdio): `python -m ventorum.agent --mcp`
- Command line: `python -m ventorum.agent --list`, and `python -m ventorum.agent --tool NAME --args JSON`
- Python: `from ventorum.agent import call_tool`

## Rules of the interface

- **Keys carry their unit**, for example `span_m`, `root_chord_m`, `sweep_le_deg`, `V_inf_m_s`, `alpha_deg`, `rho_kg_m3`, `h_m`. Unknown or ambiguous keys (`alpha`, `aoa`, `span`, ...) are refused with a hint and the list of allowed keys.
- **Types are strict.** Numbers must be numbers and booleans must be booleans.
- **No explicit null in the input.** A key with the value `null` is refused with `invalid_input` that names the key, at the top level and in nested objects. Omit the key to get its default.
- **Mesh limit.** A mesh larger than 4000 panels is refused.
- **Every result has a status:** `"success"`, or `"error"` with `error.type` from the table below.
- **Moment axes.** Every tool that returns moments takes an optional `axes` input (`"body"`, `"stability"`, `"wind"` or `"all"`, default `"body"`); each success result carries the field `"axes"`. `body` is fixed to the aircraft, `stability` is turned by alpha about y, `wind` is turned by beta about z so that x lies along the free stream (see `docs/user/conventions.md`). `CL`, `CD` and `CY` are relative to the free stream in every set. With `"all"` the result keeps the body-axis values and adds the three sets under `"moments"` (per row for tables). Stability derivatives give the moment derivatives (Cm_alpha, Cl_beta, Cn_beta) in the selected axes: the axes of the reference condition, held fixed while alpha and beta change. CL_alpha and CY_beta are relative to the free stream in every set; static margin and neutral point use the body-axis Cm_alpha.
- **Strict JSON output.** A value that is not defined (for example L/D at zero drag) is `null`.
- **Audit log:** written only if the environment variable `VENTORUM_AGENT_AUDIT_LOG` gives a file path.

## Geometry arguments: wing and aircraft

Every tool that takes a geometry takes exactly one of `wing` or `aircraft`.
The schema has no `anyOf` for `wing` and `aircraft`, because some clients drop its keys: the tool checks that exactly one is given.

- `wing`: one lifting surface (for example a main wing).
- `aircraft`: several surfaces, with reference values and a moment reference point (`surfaces` is required).

A call must give exactly one of the two. A call with both or neither is refused with `invalid_input`.

In `ventorum_batch_evaluate`, each item of `candidates` is an object with exactly one of the keys `wing` or `aircraft`, for example `{"wing": {"name": "AR8", "span_m": 8.0, "chord_m": 1.0}}`.

Example with `wing`:

```json
{
  "wing": {
    "span_m": 12.0,
    "root_chord_m": 1.6,
    "tip_chord_m": 0.8
  }
}
```

Example with `aircraft`:

```json
{
  "aircraft": {
    "surfaces": [
      {"name": "wing", "span_m": 12.0, "root_chord_m": 1.6, "tip_chord_m": 0.8},
      {"name": "tail", "span_m": 4.0, "chord_m": 0.7, "position_m": [4.6, 0.0, 0.3]}
    ],
    "ref_point_m": [0.45, 0.0, 0.0]
  }
}
```

## Work budget

- **Work units.** One solve of a lattice with N panels costs N^2 units. The work of a call is the sum over all solves that the call will run (sweep angles, heights, mesh levels, batch candidates, derivative steps).
- **Budget.** A call whose estimated work is above 200000000 units is refused with `error.type` `invalid_input` before any solve. The message gives the estimate, the budget and how to reduce the work: fewer angles (smaller range or larger `alpha_step_deg`), fewer heights, fewer mesh levels or spacing schemes, fewer candidates, or fewer panels (smaller `n_panels` or `n_chord`).
- **Scope.** The budget covers `ventorum_wing_analysis`, `ventorum_polar_sweep`, `ventorum_ground_effect` (heights plus free air plus the Irodov height-pitch sweep), `ventorum_stability_derivatives` (5 solves), `ventorum_derivative_matrix` (the real solves times N^2: in free air 5 static, 1 + 2 n control (n controls) and 7 rate solves, so 13 + 2 n with all groups; in ground effect 9 static and height solves instead of 5, so 17 + 2 n), `ventorum_batch_evaluate`, `ventorum_mesh_convergence` (levels times the angles that are really solved), `ventorum_trim` (`(1 + max_iterations * (1 + 2 * n_unknowns)) * N^2`, with 2 unknowns in longitudinal trim and 4 in lateral trim) and `ventorum_error_bars` (`N1^2 + N2^2 + N3^2` of the three spanwise levels, plus `(1 + 2 max(1, k)) N1^2` with `input_uncertainties`, k = number of flow inputs). `ventorum_tune_machine` and `ventorum_undeformed_nodes` are outside the budget: they run no solve (their duration or mesh is stated in their description instead).

## Error types

| Exception | `error.type` | Meaning |
| --- | --- | --- |
| `ValueError` from the input checks (includes `InputError` and `VentorumError`, for example an absurd position such as 1e300 m) | `invalid_input` | The input does not agree with the schema or the validity limits. |
| `numpy.linalg.LinAlgError` | `invalid_input` | The linear system is singular. This usually happens with overlapping or degenerate geometry; check the mesh for duplicate or zero-area panels. |
| `ValidityError` | `invalid_method` | The selected method is not valid for this case (for example the Fourier solver in ground effect). |
| `GroundStrikeError` | `ground_strike` | The aircraft touches or crosses the ground plane; no result is computed. |
| Any other exception | `internal` | An unexpected failure. The message gives the exception type and text. |

## Machine capabilities and tuning

- `ventorum_machine_capabilities` (no input): public hardware data (never a machine identifier or a fingerprint), available kernel backends, Cython thread mode (`openmp`, `python` or null), torch device (or null), GPU data, GPU pipeline device and precision, and tuning profile status (`none`, `valid`, `other machine`, `old schema` or `disabled`), with one sentence of advice.
- `ventorum_tune_machine` (`quick` and `save`, both boolean, both default true): measures this machine and writes its tuning profile, the same result as `ventorum-tune --quick` (with `quick` true) or `ventorum-tune` (with `quick` false). It takes about half a minute with `quick` true and up to two minutes with `quick` false. Tuning changes the speed; the device choice can change results at the float32 round-off level (about 1e-7 to 1e-6 relative). This tool is outside the work budget: its duration is stated here instead. A second call while one runs is refused with `invalid_input` ("a tuning run is in progress").

## Control surfaces

- A surface object takes an optional `controls` key: a list of 1 to 10 objects with `name` (string, required, unique on the surface), `eta_start` and `eta_end` (span limits as fractions of the semi-span, no unit, defaults 0 and 1), `hinge_x_c` (hinge position as a fraction of the local chord, no unit, larger than 0 and smaller than 1, default 0.75), `deflection_deg` (degrees, positive = right-hand rotation about the hinge axis: trailing edge down on a horizontal surface, trailing edge to +y on a fin; range -30 to 30, default 0) and `symmetric` (boolean, default true; false = opposite deflection on the left copy, an aileron). All tools that take a geometry accept them with no other change. A surface with `mirror: true` gives the controls to both copies.

## Node displacements and undeformed nodes

- A surface object takes an optional `node_displacements` key with `le_m` and `te_m` (leading-edge and trailing-edge node displacements in metres, one `[dx, dy, dz]` triple per strip edge of the defining half, from root to tip) and the optional key `eta` (span stations as fractions of the semi-span, no unit). `le_m` and `te_m` must have the same length. The number of items must equal the number of strip edges of the defining half for the solver and settings of the call.
- `ventorum_undeformed_nodes` (`wing` or `aircraft`, `settings`): returns the `eta` stations (no unit) and the `le_m` and `te_m` node coordinates (metres) of each surface, with `n_edges`. Use it to build `node_displacements`. The Fourier solver takes no node displacements and is refused with `invalid_method`. No solve runs, so this tool is outside the work budget.

## Trim

- `ventorum_trim` (`wing` or `aircraft` with the named controls, `CL_target` (no unit), `pitch_control` (string), optional `roll_control` and `yaw_control` (strings, both or neither), `flight_condition` (`alpha_deg` is the start point of the iteration), `alpha_bounds_deg` (degrees), `max_iterations` (count), `settings`, `detail_level`, `axes`): angle of attack (degrees) and control deflections (degrees) for the target lift coefficient with zero moments, by Newton's method. The solves run on the selected device; the payload states the device and the precision (in float32 the tolerances are at least 1e-6). A target that is not reached is not an error: the payload has `"status": "success"`, `"trim_status"`, `"trimmed"` false and the notes. The work estimate is `(1 + max_iterations * (1 + 2 * n_unknowns)) * N^2` (2 unknowns in longitudinal trim, 4 in lateral trim); reduce the work with fewer panels (smaller `n_panels` or `n_chord`) or a smaller `max_iterations`.

## Numerical and input error bars

- `ventorum_error_bars` (`wing` or `aircraft`, `flight_condition`, `settings` (`n_panels` is the fine level N1, at least 16), `axes` (`body`, `stability` or `wind`; `all` is refused), `detail_level`, `input_uncertainties`): layer-1 numerical error bars of the six force and moment coefficients from three spanwise mesh levels. When `input_uncertainties` is omitted, only the numerical layer is included and `status_of_bars` is `numerical_only`.
- `input_uncertainties` gives the layer-2 standard uncertainties (one standard deviation, never negative), with the unit in each key name: `alpha_deg`, `beta_deg`, `phi_deg` [deg], `V_inf_m_s` [m/s], `rho_kg_m3` [kg/m^3], `h_m` [m] (ground effect only), and `sections`, a JSON object of surface name to `a0_per_rad` [1/rad], `alpha_L0_deg` [deg], `Cd0` [-], `Cm0` [-]. The tool converts the angles to rad. The input layer is combined with the numerical layer by root sum of squares and `status_of_bars` is `numerical_and_input`. An error message gives the value in the unit of the input.
- The work estimate is `N1^2 + N2^2 + N3^2`, plus `(1 + 2 max(1, k)) N1^2` for the input layer (one solve, then the adjoint or two solves per flow input; k = number of flow inputs). Reduce the work with fewer panels (smaller `n_panels`).

## Mesh convergence

- `ventorum_mesh_convergence` (`wing` or `aircraft`, `flight_condition`, `tolerance_pct` (percent), `target_metric`, `panel_counts`, `spacing_schemes`, `alpha_sweep_deg` (degrees), `ref_n_panels`, `solver`, `detail_level`): spanwise mesh convergence study against a fine reference mesh. It gives the smallest mesh within the tolerance (`minimal_mesh`) and the recommended mesh (`recommended_mesh`, `recommended_settings`). When no level meets the tolerance, `converged` is false and there is no recommended mesh.
- The selection is reproducible. It uses a deterministic cost (total panel count, then panels solved), not the measured solve time. When two levels have the same cost, the level evaluated first is kept (order of `spacing_schemes`, then of `panel_counts`). Two calls with the same arguments give the same recommended mesh. The `solve_time_ms` values are information only.

## Full aerodynamic derivative matrix

- `ventorum_derivative_matrix` (`wing` or `aircraft`, `flight_condition`, `settings`, `groups`, `controls`, `axes`, `x_cg_m`, `detail_level`): full aerodynamic derivative matrix of one flight state. Deflections in degrees in the agent input; derivative output per rad, with the unit in each key name (`..._per_rad`) for angles and deflections, and dimensionless for height and rates. The matrix is returned as a dictionary of rows (`CL, CD, CY, Cl, Cm, Cn`) keyed by column label. Output includes `units`, `methods`, `axes`, `diagnostics` and trust notes.
- `diagnostics` has the unit in each key name: `static_margin_fraction` and `neutral_point_x_m` (body-axis Cm_alpha), `irodov_margin` and `irodov` (`x_alpha_m`, `x_h_m`, `irodov_margin`, `verdict`) in ground effect, and `controls`, one entry per control name with `CY_farfield_delta_per_rad`, `Cn_trefftz_delta_per_rad` and the near/far `warnings`.
- All the cases of one call use one mesh. An automatic `n_chord` is resolved once over all the cases; when the mesh is above the panel limit, the count is reduced and `settings_used.mesh_note` says so.
- In the stability and wind axes, the static, height and control columns turn only their moment rows; the rate columns turn with the rates, so each column equals its source function in the same axes.

Example call:

```python
from ventorum.agent import call_tool

res = call_tool("ventorum_derivative_matrix", {
    "wing": {
        "span_m": 10.0,
        "chord_m": 1.0,
        "controls": [
            {"name": "elevator", "hinge_x_c": 0.75, "deflection_deg": 1.0},
        ],
    },
    "flight_condition": {"V_inf_m_s": 30.0, "alpha_deg": 3.0},
    "settings": {"n_panels": 12},
    "detail_level": "standard",
})
print(res["executive_summary"])
print("CL_alpha:", res["matrix"]["CL"]["alpha_per_rad"])
```

Example output (abridged: the rows CD, CY, Cl and Cn and the keys `rows`, `notes`, `trust`, `condition_used`, `settings_used`, `device` and `precision` are not shown; produced by the call above on the CPU):

```json
{
  "status": "success",
  "executive_summary": "[Ventorum RESULT] Derivative matrix of Wing in body axes: 6 columns (alpha, beta, delta:elevator, p_hat, q_hat, r_hat). Computed on cpu in float64.",
  "columns": ["alpha", "beta", "delta:elevator", "p_hat", "q_hat", "r_hat"],
  "units": {"alpha": "1/rad", "beta": "1/rad", "delta:elevator": "1/rad", "p_hat": "-", "q_hat": "-", "r_hat": "-"},
  "methods": {"alpha": "central, 0.5 deg", "beta": "central, 1 deg", "delta:elevator": "central, 1 deg",
              "p_hat": "central, 0.01", "q_hat": "central, 0.01", "r_hat": "central, 0.01"},
  "axes": "body",
  "matrix": {
    "CL": {"alpha_per_rad": 4.828981, "beta_per_rad": 0.0, "delta:elevator_per_rad": 2.697583,
           "p_hat": -0.0, "q_hat": 7.320158, "r_hat": -0.0},
    "Cm": {"alpha_per_rad": -1.175825, "beta_per_rad": -0.0, "delta:elevator_per_rad": -1.287283,
           "p_hat": -0.0, "q_hat": -2.502685, "r_hat": -0.0}
  },
  "diagnostics": {
    "static_margin_fraction": 0.24349,
    "neutral_point_x_m": 0.24349,
    "controls": {
      "elevator": {"CY_farfield_delta_per_rad": 0.0, "Cn_trefftz_delta_per_rad": 0.0, "warnings": []}
    }
  }
}
```

## Example

```python
from ventorum.agent import call_tool

res = call_tool("ventorum_wing_analysis", {
    "wing": {"span_m": 12.0, "root_chord_m": 1.6, "tip_chord_m": 0.8, "sweep_le_deg": 5.0},
    "flight_condition": {"V_inf_m_s": 45.0, "alpha_deg": 5.0},
    "detail_level": "summary",
})
print(res["executive_summary"])
```

`get_tool_schemas(format=...)` exports the same schemas for OpenAI, Anthropic, Gemini and MCP.

```{toctree}
:maxdepth: 1

tool_reference
```
