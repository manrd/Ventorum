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
  tabulated run). Timing repeats run in temporary folders and are
  discarded; the tables state their samples.
* `--quick` runs 2 cases, 3 angles and 1 timing repeat.

## Method

* One AVL SURFACE per Ventorum surface. A symmetric surface is
  written as its right half with `YDUPLICATE` about y = 0 (this needs
  `iYsym = 0`). A mirror copy (`mirror_y=True`) is written as its own
  surface with mirrored coordinates, ordered tip to root so sections
  still run left to right.
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
* Static derivatives in stability axes: Ventorum by central
  differences (`analyze` at alpha +-0.5 deg and beta +-1 deg about
  alpha = 4 deg); AVL from the `ST` table of the alpha = 4 deg run.
  AVL rate derivatives are stored with "Ventorum: not available yet".

## Rules found in the AVL documentation

* Section incidence (`Ainc`): "used only to modify the flow tangency
  boundary condition on the airfoil camber line, and does not rotate
  the geometry of the airfoil section itself". Ventorum instead turns
  each section about its quarter-chord point. The compared cases use
  zero twist, so this known difference does not shift the numbers.
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
  when `ft.txt` and `st.txt` exist and parse completely. The command
  stream ends with a blank line (leaves OPER) and `QUIT`.

## Limits

* No control surfaces yet (AVL `CONTROL` blocks are never written).
* No bodies (`BODY` blocks are never written).
* Profile drag is 0 on both sides; only induced drag is compared.
* A nonzero section `Cm0` is refused unless `allow_cm0=True`.
* `TabulatedAirfoil` sections are refused.
* Timings need a quiet machine; the worker count can include
  hyper-threads.
