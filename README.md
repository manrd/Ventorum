<p align="center">
  <img src="docs/_static/ventorum_logo.png" alt="Ventorum, Computational Aerodynamics Engine" width="446">
</p>

**Author:** Manuel Alejandro Rodriguez Diaz, PhD

> [!WARNING]
> **Work in progress.** Ventorum is under active development and is not yet ready for production use. The interfaces and the results can change between versions, and the validation against test data is not complete. Do not use the results for design decisions or certification without an independent check. A production-ready release is planned.

Ventorum is a Python package for the aerodynamic analysis of wings and aircraft in low-speed flow. It calculates spanwise loads, lift, induced drag and moments with a vortex-lattice method and with Lanchester–Prandtl lifting-line methods. It also calculates the effect of flat ground (wing-in-ground effect).

## Scope and limits

The methods are valid for:

- Incompressible flow: Mach number below 0.3 (sea-level speed of sound). There is no compressibility correction. Above Mach 0.3 a case is out of the envelope: `analyze` gives a warning and the trust rating is LOW or UNRELIABLE.
- Attached flow. The nonlinear lifting line uses section polars to show the start of stall. Results after stall are not reliable.
- Thin lifting surfaces. The surfaces have no thickness; section thickness enters only through the section polars.

The package does not model compressibility, leading-edge vortex lift (slender, highly swept wings), wake roll-up, unsteady flow or a fuselage.

## Solvers

| `solver_type` | Method | Use | Limits |
|---|---|---|---|
| `"auto"` (default) | Always `"vlm"`, so the method does not change when an airfoil changes from linear to tabulated | | |
| `"vlm"` (alias `"horseshoe"`) | Vortex lattice. Each panel has a horseshoe vortex: the bound segment is at the panel quarter chord, the trailing legs follow the surface to the trailing edge and then the wake direction. One or more chordwise panels. Induced drag in the Trefftz plane. | Any planform: sweep, dihedral, taper, twist, several surfaces, ground effect. | Linear section data. A tabulated polar is used through its linear part, with a warning. |
| `"linear"` | Numerical lifting line (W. F. Phillips and D. O. Snyder, Journal of Aircraft 37(4), 2000) | Unswept wings of moderate to high aspect ratio. | Not grid convergent with sweep or with a kinked quarter-chord line (a warning is given). Refused in ground effect below h_min/c = 1. |
| `"nonlinear"` | The same lifting line, solved with Newton iteration on the section polars (smooth PCHIP interpolation, line search, restarts; sweeps start from the previous angle) | Start of stall with tabulated polars on unswept wings. Select it explicitly. | As `"linear"`. Far past the maximum lift the lifting line can have more than one solution or none; a case that does not converge is reported (`converged=False`, low trust). |
| `"fourier"` | Classical Fourier series (H. Glauert, 1926) | One symmetric, unswept, planar wing. | No ground effect. |

The solvers use compiled (Numba) kernels on all CPU cores, and run independent cases (sweeps, batches) side by side. Run `ventorum-tune` once after installation: it measures the machine and stores the best thread settings. Without it, Ventorum uses defaults that work on any machine. See `docs/user/parallel.md`.

On a machine with an NVIDIA CUDA GPU, the vortex lattice and the linear and nonlinear lifting lines also run on the GPU (`pip install -e .[gpu]`): batches of flight conditions (sweeps, ground-effect grids) and large single solves. The device `"auto"` (default) selects the GPU only where it is faster. See `docs/user/how_to_gpu.md`.

## Conventions

- Geometry axes: x aft, y to the right, z up. The angles in `FlightCondition` are in radians.
- `LiftingSurface.semi_span` is the length along the dihedral line. If a section gives `z_le`, `semi_span` is the projected y extent.
- The reference values `S_ref`, `b_ref` and `c_ref` are projected on the x-y plane. `c_ref` is the mean aerodynamic chord.
- Moments (as in AVL): `Cl > 0` right wing down, `Cm > 0` nose up, `Cn > 0` nose right. Sideslip `beta > 0` is wind from the right. Positive dihedral gives `Cl_beta < 0`.
- Moments are about `Aircraft.ref_point` (for example the centre of gravity; the origin of the geometry axes if not set), in `analyze`, sweeps and ground effect.
- `S_ref`, `b_ref` and `c_ref` that you do not set come from the main surface and its mirror copy, and follow later changes of the geometry. The main surface is the surface with the largest planform area projected on the x-y plane (a half surface counts with its mirror copy). In a tie, the first surface in the list is the main surface. So the order of the surfaces does not make a tail or a fin the reference.
- Wake: `SolverSettings(wake_alignment="freestream")` (default) sends the wake along the free stream. `"body"` sends it along the x axis, as in AVL; then the wake and the system matrix do not change with the angle of attack, and the circulation is a linear function of the free-stream components.

## Installation

From the root directory:

```bash
pip install -e .
```

Dependencies: `numpy`, `scipy`, `matplotlib`, `numba` and `torch`. On a machine with no GPU, install the CPU build of PyTorch first (`pip install torch --index-url https://download.pytorch.org/whl/cpu`). The Cython kernels are compiled when a C compiler is present; without one, the installation continues with the Numba and numpy kernels. Optional: `pandas` (`pip install -e .[tables]`) for table export; NVIDIA Warp (`pip install -e .[gpu]`) for the GPU pipelines, with the CUDA build of PyTorch. Development: `pip install -e .[dev]` (pytest, pytest-cov, ruff). Documentation: `pip install -e .[docs]`.

## Quick Start

```python
import ventorum as vt

# 1. Define wing sections (root and tip)
sections = [
    vt.WingSection(y_frac=0.0, chord=2.0),
    vt.WingSection(y_frac=1.0, chord=1.0, twist=-0.05)  # wash-out [rad]
]

# 2. Build the lifting surface
wing = vt.LiftingSurface(
    name="Main Wing",
    semi_span=5.0,
    sections=sections
)

# 3. Run the analysis (solver "auto": vortex lattice for linear section data)
result = vt.analyze(wing, alpha_deg=5.0, V_inf=50.0)

# 4. Access total and spanwise results
print(f"CL = {result.totals.CL:.4f}")
print(f"CDi = {result.totals.CDi:.5f}")
print(f"Span Efficiency (e) = {result.totals.e:.3f}")

# 5. Plot the distributions
import matplotlib.pyplot as plt
fig = vt.plot_all_distributions(result)
plt.show()
```

## Ground Effect

```python
from ventorum.ground_effect import analyze_ground_effect

res = analyze_ground_effect(wing, h=0.6, alpha_deg=4.0, phi_deg=2.0, height_ref="te")
print(res.CL, res.CDi, res.Cl, res.h_min_over_c)
```

- The ground is flat, rigid and parallel to the free stream. The image method makes the velocity normal to the ground zero. The wake leaves the trailing edge parallel to the ground.
- `height_ref` sets the definition of h: `"ref"` (the moment reference point `Aircraft.ref_point`, the origin if not set), `"min"` (the lowest edge point), `"qc"` or `"te"` (the root quarter-chord or trailing-edge point).
- The vortex lattice selects the number of chordwise panels from the smallest gap (`n_chord=None`). `GroundEffectSweep` uses one value for all cases, the value at the lowest height, so the derivatives with height do not include a change of mesh. An angle-of-attack sweep in ground effect (`analyze_sweep`) and the stability-derivative tool do the same: all their cases use the count of the case with the smallest gap.
- Contact with the ground raises `GroundStrikeError`. The lifting-line solvers raise `ValidityError` below h_min/c = 1, because they under-predict the lift increase (see section V7 of the verification report). Both are in `ventorum.core.errors` and are subclasses of `ValueError`. In a sweep these cases are NaN, flagged (`is_strike`, `refused`) and explained in `errors`. The bank strike limit is searched up to 60 deg; `strike_limit_found` is False when there is no contact up to that limit.
- The plate is thin and the flow is inviscid. Thickness effects (for example suction under the wing at small gaps) and viscous effects are not modelled. The trust score gives a warning below h_min/c = 0.3.
- `GroundEffectSweep(...).run_sweep(...).compute_stability_derivatives()` gives the pitch and height aerodynamic centres and the Irodov margin `(x_alpha - x_h)/c` (stable if > 0). The bank derivative is restoring if `Cl_phi < 0`.
- `vt.analyze(..., condition=vt.FlightCondition(..., h=...))` uses the same convention: `h` is the height of the moment reference point.

## Verification

`python validation/run_verification.py` writes `docs/verification_report.md` (a few seconds). It compares the solvers with:

- The elliptic wing of the Lanchester–Prandtl lifting-line theory and Glauert's monoplane equation (solved independently in `ventorum.reference`).
- The flat circular wing (Kinner, lifting-surface theory) and Helmbold's formula.
- Grid convergence with sweep, mesh convergence and the condition number of the system.
- Image-method checks in ground effect, and the limits of the lifting line near the ground.
- Camber, section pitching moment and moment signs.

`pytest tests/test_analytical_verification.py` checks the same items with fixed tolerances.

This is verification (the equations are solved correctly). It is not validation against experiments. There are no experimental datasets yet. A dataset must have full provenance (source, table or figure, page, test conditions); `ventorum.reference.load_dataset` refuses a dataset without it. See `validation/experimental/README.md`.


## AI Agent Tool Interface

Ventorum has a tool interface for AI agents and automated design loops (Python calls, a command line, and an MCP server):

- **Strict inputs.** Keys carry their unit (`span_m`, `root_chord_m`, `sweep_le_deg`, `tip_twist_deg`, `V_inf_m_s`, `alpha_deg`, `rho_kg_m3` or `altitude_m`, `h_m`, ...). Unknown or ambiguous keys (`alpha`, `aoa`, `washout`, `aspect_ratio`, ...) are refused with a list of the allowed keys. Numbers must be numbers and booleans must be booleans. A mesh larger than 4000 panels is refused. Each call also estimates its total work (the sum of N^2 over all planned solves, with N the panel count of each lattice) and refuses a call above the budget of 200000000 units (about 12 solves of 4000 panels); use fewer angles, heights, levels, cases, or panels to stay within it. `get_tool_schemas(format=...)` exports the same schemas for OpenAI, Anthropic, Gemini and MCP.
- **Honest outputs.** Every result has `status` (`"success"` or `"error"` with `error.type`: `invalid_input`, `ground_strike`, `invalid_method` or `internal`). The output is strict JSON (no NaN). Ground strikes and refused heights are reported per row; failed batch candidates are listed, not dropped. The static margin uses the given CG position.
- **Tools:** `ventorum_wing_analysis`, `ventorum_polar_sweep`, `ventorum_ground_effect` (one chordwise mesh for all heights and free air; Irodov margins with 2 or more heights; a one-sided difference with 2), `ventorum_stability_derivatives`, `ventorum_batch_evaluate`, `ventorum_mesh_convergence`.
- **Trust score.** The uncertainty bands are heuristic and not calibrated against experiments; do not use them as error bars.
- **MCP:** `python -m ventorum.agent --mcp` (JSON-RPC 2.0 over stdio). **Command line:** `python -m ventorum.agent --list` and `--tool NAME --args JSON`.
- **Audit log:** written only if the environment variable `VENTORUM_AGENT_AUDIT_LOG` gives a file path.

```python
from ventorum.agent import call_tool

res = call_tool("ventorum_wing_analysis", {
    "wing": {"span_m": 12.0, "root_chord_m": 1.6, "tip_chord_m": 0.8, "sweep_le_deg": 5.0},
    "flight_condition": {"V_inf_m_s": 45.0, "alpha_deg": 5.0},
    "detail_level": "summary",
})
print(res["executive_summary"])
# [Ventorum RESULT] Wing: CL=0.4347, CDi=0.00607, CD=0.00607 (CDi only: the airfoils have no profile drag, cd0 = 0), L/D=71.65, e=0.991, Cm=-0.1865 about [0, 0, 0] m. Condition: alpha=5 deg, beta=0 deg, V=45 m/s, free air. Solver: vlm, converged=True. Trust 1.00 (HIGH), 0 warning(s).

bad = call_tool("ventorum_wing_analysis", {"wing": {"span_m": 12.0, "chord_m": 1.4}, "flight_condition": {"aoa": 4.5}})
print(bad["error"]["message"])
# flight_condition: unknown key 'aoa'. Hint: use alpha_deg (degrees). Allowed keys: V_inf_m_s, alpha_deg, altitude_m, beta_deg, h_m, rho_kg_m3.
```

### Trust Score Access (Non-AI Usage)

```python
import ventorum as vt

wing = vt.LiftingSurface(semi_span=6.0, sections=[vt.WingSection(chord=1.5), vt.WingSection(y_frac=1.0, chord=0.8)])
result = vt.analyze(wing, alpha_deg=4.0)

print(result.totals.trust.summary_str())
# Trust: 1.00 (HIGH) | heuristic band (not calibrated) CL +/- 0.015, CDi +/- 0.00040 | No warnings
```

## Examples

The `examples/` directory contains these scripts:

1. `01_elliptic_wing.py`: Elliptic wing; compares the solvers with the theoretical e = 1.
2. `02_tapered_wing.py`: Taper and twist.
3. `03_swept_wing.py`: Sweep and dihedral.
4. `04_nonlinear_stall.py`: Nonlinear section tables and the start of stall.
5. `05_wing_tail.py`: Wing and tail; downwash on the tail.
6. `06_alpha_sweep.py`: Angle-of-attack sweep and drag polar.
7. `07_xfoil_integration.py`: XFOIL polars in the nonlinear solver (skipped if XFOIL is not installed).
8. `08_multi_instance_parallel.py`: Several independent cases in parallel.
9. `09_aerosonde_uav.py`: Aerosonde UAV model with plots.
10. `10_ai_agent_tool_integration.py`: AI agent tool calls, MCP schemas, stability derivatives and trust score.

Examples 08 and 09 write their output to `examples/output/` (not kept in git). The other examples print their results.

## Geometry System

- `WingSection`: the cross-section at a spanwise fraction (`y_frac` from `0.0` to `1.0`). It has the chord, twist, optional leading-edge position and the airfoil data.
- `LiftingSurface`: a wing segment (or half-wing if symmetric) from root to tip, defined by a list of `WingSection` objects. A symmetric surface is mirrored about the plane y = 0, so its root must be on that plane. For a surface off the plane (for example twin fins), set `is_symmetric=False` and add `surf.mirrored()` as the left copy; camber and twist are mirrored too.
- `Aircraft`: a collection of `LiftingSurface` objects.

## JSON Configuration

```python
import ventorum as vt
from ventorum.core.config import save_aircraft_to_json, load_aircraft_from_json

wing = vt.LiftingSurface(name="Wing", semi_span=5.0,
                          sections=[vt.WingSection(y_frac=0.0, chord=1.2), vt.WingSection(y_frac=1.0, chord=0.6)])
aircraft = vt.Aircraft(name="Demo", surfaces=[wing])

# Save to disk
save_aircraft_to_json(aircraft, "my_plane.json")

# Load from disk
loaded_aircraft = load_aircraft_from_json("my_plane.json")
```

## Documentation

The full documentation (user manual, theory manual, agent guide, software description, verification and API reference) is in `docs/` and is built with Sphinx:

```bash
pip install -e .[docs]
sphinx-build -W --keep-going -b html docs docs/_build/html
```

Then open `docs/_build/html/index.html`. The documentation standards are in `docs/contributing/documentation.md`.

## Development

```bash
pip install -e .[dev]
ruff check .
python -m pytest -q
```

GitHub Actions runs the same checks, the documentation build, the verification script and all examples on every push (`.github/workflows/tests.yml`).

## License
MIT License.
