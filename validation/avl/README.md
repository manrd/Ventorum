# AVL comparison harness (black box)

This folder holds a harness that runs the same cases in Ventorum and
in AVL (the vortex-lattice program of M. Drela and H. Youngren) and
writes tables of the differences. AVL is used as a black box only:
the harness writes AVL input files, runs `avl.exe`, and reads its
output files. No AVL source code is read, opened, searched or copied.
The harness code is written from the AVL user documentation only.

## Documents used

* `avl_doc.txt`, sections "Geometry Input File", "Vortex Lattice
  Spacing Distributions", "OPER Routine", "Calculation Setup", "Flow
  Solution", "Output", "Stability derivatives" and "Program
  Execution" (MIT AVL 3.40 user primer, read on the web; the local
  AVL folder holds only the `avl.exe` program, no documents).
* The banner and the `FT`/`ST` output text of the installed
  `avl.exe` (version 3.52), observed by running the program.

## How to run

```bash
python validation/avl/compare.py --avl PATH --out DIR [--quick] [--workers N] [--timeout S]
```

* `--avl PATH` gives the AVL program, else the `VENTORUM_AVL_EXE`
  variable. Without AVL the program stops with a clear message.
* `--out DIR` is required and must lie OUTSIDE the repository. The
  program writes `avl_comparison.md` (tables), `avl_comparison.json`
  (all numbers) and `runs/` (the AVL input and output files of every
  tabulated run and of the extra runs for the derivatives). Timing
  repeats run in temporary folders and are discarded; the tables
  state their samples.
* `--quick` runs 2 cases, 3 angles and 1 timing repeat.

## Method

* One AVL SURFACE per Ventorum surface. A symmetric surface is
  written as its right half with `YDUPLICATE` about y = 0 (this needs
  `iYsym = 0`). A mirror copy (`mirror_y=True`) is written as its own
  surface with mirrored coordinates, ordered tip to root so sections
  still run left to right. AVL applies the spanwise spacing parameter
  from the first section to the last, so its sign changes for the
  mirror copy. The mirror copy gets the `COMPONENT` index of its
  partner surface; the two halves of a `YDUPLICATE` surface also share
  one index.
* One AVL SECTION per Ventorum section. Incidence [deg] = section
  twist + surface incidence - zero-lift angle of the section airfoil.
  The lift slope factor (`CLAF`) = `a0 / (2 pi)`.
* Spanwise count per surface (per half for a duplicated surface) =
  Ventorum `n_panels`; chordwise count = Ventorum `n_chord`.
* Free air: AVL alpha and beta equal Ventorum alpha and beta; Ventorum
  runs once with the body-axis wake (the same wake model as AVL) and
  once with the default free-stream wake.
* Ground effect (geometry-rotation method): the whole geometry turns
  nose up by alpha about the reference point, AVL runs at alpha = 0
  with the z-image plane as a solid wall at `Zsym = Zref - h`, and AVL
  body moments are compared with Ventorum stability moments. The
  body-wake column reads `n/a` there.
* Compared values: CL and CY are the near-field values of both
  programs (AVL `CLtot` and `CYtot`); CDi is the Trefftz-plane value
  (AVL `CDff`). Ventorum CY is the side force normal to lift and drag
  (wind axes), AVL `CYtot` is the body y force. The two are the same
  at beta = 0 (all sweep rows). For `CY_beta` they differ by a term of
  the order of the drag coefficient.
* Static derivatives in stability axes: Ventorum and AVL both by
  central differences (alpha +-0.5 deg and beta +-1 deg about
  alpha = 4 deg; `analyze` for Ventorum, extra AVL runs for AVL).
  In ground effect each AVL alpha run has its own rotated geometry, so
  the ground stays parallel to the free stream, as in Ventorum. The
  `ST` table of the AVL run at alpha = 4 deg is kept as a separate
  column (`AVL_ST`). In ground effect its alpha derivatives change
  alpha against a ground that stays fixed in the body axes, which is
  a different quantity. A failed extra run gives `n/a` with a note.
  AVL rate derivatives (from the `ST` table) are stored with
  "Ventorum: not available yet".
* Table cells: `n/a` marks a cell that is empty by plan (for example
  the body-wake column in ground effect); `failed` marks a failed
  Ventorum solve or AVL run. The notes of the AVL writer (for example
  an approximate spacing) are kept per row.
* Timing: AVL native (one process per angle, one after the other),
  AVL one session (one process runs the whole sweep; in free air on
  one loaded geometry, in ground effect it reads one rotated geometry
  per angle with `LOAD`), AVL enhanced (up to `--workers` processes at
  the same time), Ventorum CPU and Ventorum auto device. Ventorum runs
  one `analyze_sweep` call in free air and a loop of `analyze` calls
  (one per angle) in ground effect. Each Ventorum device runs one
  untimed warm-up sweep before its timed repeats.

## Rules found in the AVL documentation

* Section incidence (`Ainc`): "used only to modify the flow tangency
  boundary condition on the airfoil camber line, and does not rotate
  the geometry of the airfoil section itself". Ventorum instead turns
  each section about its quarter-chord point. In free air the
  compared cases use zero twist and zero incidence, so this
  difference does not apply there.
* Known difference in ground effect (chord tilt): the
  geometry-rotation method adds alpha to `Ainc`, and AVL keeps each
  section chord along the body x axis. The rotated AVL geometry moves
  its leading edges, but its panels stay parallel to the ground. The
  Ventorum chords tilt by alpha against the ground. This difference
  of the method is present in every ground-effect row.
* Solid wall: `iZsym = 1` makes the plane `Z = Zsym` a solid wall
  ("Ground effect is simulated with iZsym = 1, and Zsym = location of
  ground").
* Spacing: the AVL spacing parameter blends uniform (0.0), cosine
  (1.0, clustered at root and tip), sine (2.0, clustered at the first
  section) and mirrored sine (-2.0, clustered at the last section).
  Mapping used here: Ventorum `half-cosine` -> -2.0, `cosine` -> 1.0,
  `uniform` -> 0.0, `root-cosine` -> 2.0, `power` -> -2.0 (approximation,
  noted per case); chordwise `uniform` -> 0.0, `cosine` -> 1.0.
* Axes of the `FT` file: standard orientation (X forward, Z down);
  forces and moments are in stability axes tilted by alpha (primed
  roll and yaw moments `Cl'`, `Cn'` beside the body values).
  The `ST` table is in stability axes; `SB` would give body axes.
* Verdict of a run: the exit code means nothing; a run is good only
  when `ft.txt` and `st.txt` exist, parse completely, give the
  requested `Alpha` and `Beta` (absolute tolerance 1e-4 deg) and give
  the expected `# Surfaces`, `# Strips` and `# Vortices` counts in
  their header. The harness deletes old output files before each run:
  AVL asks "File exists. Append/Overwrite/Cancel" for an existing file,
  and that prompt takes the next line of the command stream. The
  command stream ends with a blank line (leaves OPER), 4 more blank
  lines (they get back to the top level if a prompt took a line that
  was not planned) and `QUIT`.

## Limits

* No control surfaces yet (AVL `CONTROL` blocks are never written).
* No bodies (`BODY` blocks are never written).
* Profile drag is 0 on both sides; only induced drag is compared.
* A nonzero section `Cm0` is refused unless `allow_cm0=True`.
* `TabulatedAirfoil` sections are refused.
* AVL has fixed array limits (compiled into the program) on the
  counts of strips, vortices and chordwise vortices. Above a limit AVL
  can reduce the mesh without an error, or stop. The harness compares
  the counts of the `FT` header with the expected counts and fails the
  run on a mismatch.
* Resolution: AVL prints most `FT` values with 5 decimals (`CDind`
  and `CDff` with 7), so differences below about 1e-5 are not
  resolved. The AVL derivatives come from central differences of
  these values, so their resolution is about 1e-5 divided by the
  angle step.
* Timings need a quiet machine; the worker count can include
  hyper-threads.
