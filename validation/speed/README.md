# Cross-tool speed harness (black box)

This folder holds one harness that times each low-order tool and
Ventorum at equal accuracy on the same cases. Each tool runs as a
black box only: an external program or a separate Python
interpreter, never as a Ventorum dependency. No source code of any
tool is read, opened, searched or copied. The adapters use only the
user documentation of each tool for input and output formats.

## How to run

```bash
python validation/speed/runner.py --out DIR [--quick] [--tools avl,ventorum] [--timeout S] [--repeats N]
```

* `--out DIR` is required and must lie OUTSIDE the repository.
  Results stay private until the owner decides. The program writes
  `speed_comparison.md` (tables), `speed_comparison.json` (all
  numbers) and `runs/` (work folders of the convergence sweeps).
  Timing repeats run in the same folders; the tables state their
  samples.
* `--quick` runs 1 case, 3 angles and 1 timing repeat.
* The full run uses 4 cases (rectangular, straight tapered, swept
  tapered, wing and tail), 33 angles per sweep (from -4 to 10 deg)
  and medians of 5 repeats.
* `--tools` selects a subset of
  `avl, vspaero, flow5, aerosandbox, machupx, ventorum`.
  A missing tool is skipped with a note, never an error.
* The Python tools always run in a subprocess (one case script per
  case set); only the solve loop is timed. The import time is
  measured separately, kept in `result.json` beside the numbers, and
  excluded from the tables.

## Tool paths (environment variables, never machine paths in files)

| Tool | Option | Variable |
| --- | --- | --- |
| AVL | `--avl PATH` | `VENTORUM_AVL_EXE` |
| VSPAERO | `--vspaero PATH` | `VENTORUM_VSPAERO` |
| OpenVSP Python | (called as subprocess) | `VENTORUM_VSP_PYTHON` |
| Flow5 | `--flow5 PATH` | `VENTORUM_FLOW5` |
| AeroSandbox interpreter | `--aerosandbox-python PATH` | `VENTORUM_AEROSANDBOX_PYTHON` |
| MachUpX interpreter | `--machupx-python PATH` | `VENTORUM_MACHUPX_PYTHON` |

Python tools always run in a subprocess (one script per case set)
so their import time and threads do not mix with Ventorum
(see `base.run_json_script`). Only their documented public API is
used.

## Method

* Cases: rectangular wing AR 8, straight tapered wing AR 8, swept
  tapered wing AR 6 (30 deg leading edge sweep), wing and
  horizontal tail. All runs are in free air. Ground effect runs
  only for tools that support it (AVL and Ventorum declare
  support; the runner keeps the free-air set until the owner asks
  for the ground set).
* Equal accuracy: each tool runs at the mesh where its CL and CDi
  sweeps both change by less than 0.5 % on refinement. The change
  at each angle is divided by the peak absolute value of the fine
  sweep, and the largest value over the sweep must pass. This keeps
  a zero crossing (CL near zero lift) from failing the check. The
  report gives the mesh and the two changes. Without convergence
  the finest mesh is kept with a note.
* Compared values: CL [-] is the near-field lift of each tool (AVL
  `CLtot`); CDi [-] is the Trefftz-plane induced drag (AVL `CDff`).
* Timing: each value is the wall time of the whole sweep at the
  converged mesh. Ventorum runs one untimed warm-up sweep before
  its timed repeats. Timings need a quiet machine; treat small
  differences as noise.

## Documents used

* AVL user documentation (see `validation/avl/README.md`) for the
  AVL adapter (input format, output files, verdict of a run).
* OpenVSP published examples and tutorial: `Swept_Wing_API/README.md`
  (VLM sweep over an alpha range, reference values, X-Z symmetry
  option, spanwise/chordwise tessellation, panel density rule,
  `vsp -script` batch command), `TR1208.vspscript` (same inputs
  through the API), `TestAnalysisVSPAERO.vspscript` (thin geometry
  set for the vortex-lattice mode, `Tess_W`/`SectTess_U`
  tessellation, Mach 0.1 example, `CLtot` history results),
  `Wing.vspscript` (wing creation) and `DegenGeom.vspscript`
  (degenerate geometry), plus the API documentation. The installed
  programs were also run to check input defaults
  (`PrintAnalysisInputs`: Mach 0, `WakeNumIter` 5, `Symmetry` 0),
  output files and the settings echo.
* AeroSandbox package README (SI units, angle of attack in degrees,
  vortex-lattice tutorial) and API documentation
  (`VortexLatticeMethod` with spanwise/chordwise resolution and
  cosine spacing defaults, `Wing`/`WingXSec`/`Airplane` geometry,
  `OperatingPoint`, inviscid drag output).
* MachUpX documentation (`Input Files` page: scene and aircraft
  objects, solver types, SI units, grid `N` and `cosine_cluster`
  defaults; `User Interface` page: `Scene` class and `solve_forces`)
  and the published `examples/Traditional` example (solver dict,
  reference values, linear airfoil keys, grid, tail placement).
* The behavior found by running each tool (versions, output keys,
  axis directions, convergence of the wake) is recorded in the
  adapter docstrings and in Limits below.

## Tool settings (best documented settings of each tool)

* VSPAERO: vortex-lattice mode (thin geometry set only), free air,
  Mach 0 (the program reports Mach 0.001, still incompressible), one
  run over the alpha list of the case (single-point runs only when
  the angles are not uniform), symmetry off, `NCPU` on all cores,
  default wake iterations, reference values from the case. Mesh
  refinement by `SectTess_U` per wing section at equal spanwise
  density (chordwise tessellation at its default). Each adapter run
  records the tool version, the settings and the mesh: the version
  and the mesh in the `mesh_label` of the JSON output, the full
  settings in the run folder (`case.json`, `geom_info.json`, the
  analysis script, the `.vspaero` settings echo and the `.polar`
  file).
* AeroSandbox: `VortexLatticeMethod` with the mesh key as
  `spanwise_resolution`, chordwise resolution at its default (10),
  cosine spacing defaults, one operating point per angle at the case
  velocity, symmetric right-half wings, case reference values. The
  mesh key sets every surface. Records as above (`result.json` holds
  version, settings, solve time and import time).
* MachUpX: scene dict with SI units, linear solver (stated; the
  documentation recommends nonlinear for accuracy), default
  flat-plate airfoil (drag is induced drag only), `cosine_cluster`
  grid with count `N` set by the mesh key on every surface,
  secondary surfaces connected to the main wing root. Records as
  above (`result.json` plus the scene files).
* Each adapter checks the tool output before timing it: VSPAERO
  matches the `.vspaero` echo (Sref, angles) and the `.polar`
  columns by name; the Python tools validate the `result.json`
  shape. A mismatch fails the run, never silent zeros.

## Files

* `base.py`: shared types and helpers (results, medians,
  convergence check, tool search, subprocess JSON runner).
* `cases.py`: shared geometries and angle sweeps.
* `case_spec.py`: tool-independent case description (plain data for
  the helper scripts; no tool runs here).
* `adapter_avl.py`: real AVL sweeps through `validation.avl`.
* `adapter_ventorum.py`: in-process Ventorum reference sweeps.
* `adapter_vspaero.py`: real VSPAERO sweeps (geometry in a
  `py -3.13` subprocess, analysis with `vsp -script`).
* `vsp_make_geom.py`: OpenVSP geometry helper (runs under the
  OpenVSP Python interpreter).
* `adapter_aerosandbox.py` with `run_aerosandbox_case.py`, and
  `adapter_machupx.py` with `run_machupx_case.py`: real sweeps in a
  subprocess (one case script per case set).
* `adapter_flow5.py`: presence detection plus an honest skip (no
  batch mode documented; the window is never driven).
* `runner.py`: convergence loop, timing loop, Markdown and JSON
  reports. Output always goes outside the repository.

## Limits

* Profile drag is 0 on the Ventorum, AVL, AeroSandbox and MachUpX
  sides; VSPAERO profile drag (`CDo`) is not compared either. Only
  induced drag is compared.
* VSPAERO symmetry stays off although the tutorial documents an X-Z
  symmetry option: with symmetry on, the wake does not settle within
  the default 5 iterations (it settles after about 8), and the
  converged induced drag at low lift is unphysical (negative at
  CL 0.24 where the symmetric run gives the textbook value). The
  owner decides whether to revisit this.
* VSPAERO accepts Mach 0 but reports Mach 0.001; both are
  incompressible. Non-uniform angle lists (the quick list) run one
  single-point sweep per angle at the exact requested angles.
* MachUpX runs the linear solver (stated in every record); the
  documentation recommends the nonlinear solver for accuracy.
* The MachUpX vertical offset sign follows the standard body frame
  (z down); the measured effect of the sign is below 1 % on CL.
* Flow5 has no batch mode documented, so the harness skips it and
  never drives its window.
