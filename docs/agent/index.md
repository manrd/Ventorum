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
- **Scope.** The budget covers `ventorum_wing_analysis`, `ventorum_polar_sweep`, `ventorum_ground_effect` (heights plus free air plus the Irodov height-pitch sweep), `ventorum_stability_derivatives` (5 solves), `ventorum_batch_evaluate` and `ventorum_mesh_convergence` (levels times the angles that are really solved). `ventorum_tune_machine` is outside the budget: its duration is stated in its description instead.

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
