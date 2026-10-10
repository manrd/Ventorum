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

## Work budget

- **Work units.** One solve of a lattice with N panels costs N^2 units. The work of a call is the sum over all solves that the call will run (sweep angles, heights, mesh levels, batch candidates, derivative steps).
- **Budget.** A call whose estimated work is above 200000000 units is refused with `error.type` `invalid_input` before any solve. The message gives the estimate, the budget and how to reduce the work: fewer angles (smaller range or larger `alpha_step_deg`), fewer heights, fewer mesh levels or spacing schemes, fewer candidates, or fewer panels (smaller `n_panels` or `n_chord`).
- **Scope.** The budget covers `ventorum_wing_analysis`, `ventorum_polar_sweep`, `ventorum_ground_effect` (heights plus free air plus the Irodov height-pitch sweep), `ventorum_stability_derivatives` (5 solves), `ventorum_batch_evaluate`, `ventorum_mesh_convergence` (levels times the angles that are really solved), `ventorum_trim` (`(1 + max_iterations * (1 + 2 * n_unknowns)) * N^2`, with 2 unknowns in longitudinal trim and 4 in lateral trim) and `ventorum_error_bars` (`N1^2 + N2^2 + N3^2` of the three spanwise levels). `ventorum_tune_machine` and `ventorum_undeformed_nodes` are outside the budget: they run no solve (their duration or mesh is stated in their description instead).

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
- `ventorum_tune_machine` (`quick` and `save`, both boolean, both default true): measures this machine and writes its tuning profile, the same result as `ventorum-tune`. It takes about half a minute with `quick` true and up to two minutes with `quick` false. Tuning changes the speed; the device choice can change results at the float32 round-off level (about 1e-7 to 1e-6 relative). This tool is outside the work budget: its duration is stated here instead. A second call while one runs is refused with `invalid_input` ("a tuning run is in progress").

## Control surfaces

- A surface object takes an optional `controls` key: a list of 1 to 10 objects with `name` (string, required, unique on the surface), `eta_start` and `eta_end` (span limits as fractions of the semi-span, no unit, defaults 0 and 1), `hinge_x_c` (hinge position as a fraction of the local chord, no unit, default 0.75), `deflection_deg` (degrees, positive = right-hand rotation about the hinge axis: trailing edge down on a horizontal surface, trailing edge to +y on a fin; range -30 to 30, default 0) and `symmetric` (boolean, default true; false = opposite deflection on the left copy, an aileron). All tools that take a geometry accept them with no other change. A surface with `mirror: true` gives the controls to both copies.

## Node displacements and undeformed nodes

- A surface object takes an optional `node_displacements` key with `le_m` and `te_m` (leading-edge and trailing-edge node displacements in metres, one `[dx, dy, dz]` triple per strip edge of the defining half, from root to tip) and the optional key `eta` (span stations as fractions of the semi-span, no unit). `le_m` and `te_m` must have the same length. The number of items must equal the number of strip edges of the defining half for the solver and settings of the call.
- `ventorum_undeformed_nodes` (`wing`, `settings`): returns the `eta` stations (no unit) and the `le_m` and `te_m` node coordinates (metres) of each surface, with `n_edges`. Use it to build `node_displacements`. The Fourier solver takes no node displacements and is refused with `invalid_method`. No solve runs, so this tool is outside the work budget.

## Trim

- `ventorum_trim` (`wing` with the named controls, `CL_target` (no unit), `pitch_control` (string), optional `roll_control` and `yaw_control` (strings, both or neither), `flight_condition` (`alpha_deg` is the start point of the iteration), `alpha_bounds_deg` (degrees), `max_iterations` (count), `settings`, `detail_level`, `axes`): angle of attack (degrees) and control deflections (degrees) for the target lift coefficient with zero moments, by Newton's method. Every solve runs on the CPU in float64. A target that is not reached is not an error: the payload has `"status": "success"`, `"trim_status"`, `"trimmed"` false and the notes. The work estimate is `(1 + max_iterations * (1 + 2 * n_unknowns)) * N^2` (2 unknowns in longitudinal trim, 4 in lateral trim); reduce the work with fewer panels (smaller `n_panels`) or a smaller `max_iterations`.

## Numerical error bars

- `ventorum_error_bars` (`wing`, `flight_condition`, `settings` (`n_panels` is the fine level N1, at least 16), `axes` (`body`, `stability` or `wind`; `all` is refused), `detail_level`): layer-1 numerical error bars of the six force and moment coefficients from three spanwise mesh levels. Only the numerical (spanwise discretisation) layer is included; the other layers are not implemented, and `status_of_bars` is always `numerical_only`. The work estimate is `N1^2 + N2^2 + N3^2`; reduce the work with fewer panels (smaller `n_panels`).

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
