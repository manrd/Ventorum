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
* The user documentation of VSPAERO, Flow5, AeroSandbox and MachUpX
  for their adapters. Until the owner confirms the documented best
  settings of a tool, its adapter detects the program but reports
  skipped with the reason (VSPAERO: input mapping pending;
  Flow5: no batch mode documented; Python tools: solve script
  pending a documented public API check). This keeps every timed
  number honest.

## Files

* `base.py`: shared types and helpers (results, medians,
  convergence check, tool search, subprocess JSON runner).
* `cases.py`: shared geometries and angle sweeps.
* `adapter_avl.py`: real AVL sweeps through `validation.avl`.
* `adapter_ventorum.py`: in-process Ventorum reference sweeps.
* `adapter_vspaero.py`, `adapter_flow5.py`,
  `adapter_aerosandbox.py`, `adapter_machupx.py`: presence
  detection plus honest skip reasons until the documented settings
  are confirmed.
* `runner.py`: convergence loop, timing loop, Markdown and JSON
  reports. Output always goes outside the repository.

## Limits

* Profile drag is 0 on the Ventorum and AVL sides; only induced
  drag is compared.
* The Python tool adapters time nothing until their documented API
  calls are confirmed.
* Flow5 has no batch mode documented, so the harness skips it and
  never drives its window.
