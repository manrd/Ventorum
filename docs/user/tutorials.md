# Tutorials

This page walks through the examples 01 to 06 in `examples/`. Each example is one script. Run it from the repository root with `MPLBACKEND=Agg` so plots do not open a window:

```bash
MPLBACKEND=Agg python examples/01_elliptic_wing.py
```

## 01: Elliptic wing (`01_elliptic_wing.py`)

Build an elliptic planform from 21 sections and run the Fourier and horseshoe solvers at 5 deg angle of attack. An elliptic wing has span efficiency `e = 1`. Compare the reported `e` with 1. Read `CL`, `CDi` and the Fourier coefficients from the output.

## 02: Tapered wing (`02_tapered_wing.py`)

Build a tapered wing (taper ratio 0.5) with washout and run it at 0, 3, 6 and 9 deg. Read how taper and twist change the lift distribution. Then run an alpha sweep with `vt.alpha_sweep` and plot the drag polar with `vt.plot_sweep_summary`.

## 03: Swept wing (`03_swept_wing.py`)

Build a wing with 30 deg leading-edge sweep and 5 deg dihedral. This geometry needs the horseshoe (vortex-lattice) solver. Plot the 3D geometry with `vt.plot_geometry`, read the spanwise distributions, and run an alpha sweep.

## 04: Nonlinear stall (`04_nonlinear_stall.py`)

Give the sections a tabulated polar (`TabulatedAirfoil`) and run the nonlinear solver. It iterates on the section polars with Newton iteration. Read where lift stops growing with angle of attack. Far past maximum lift a case can fail to converge; the solver reports it.

## 05: Wing and tail (`05_wing_tail.py`)

Build an `Aircraft` with a main wing and a horizontal tail behind it. Run one analysis and read the loads of each surface in `result.spanwise`. The tail sees the downwash of the wing. This example shows multi-surface interaction.

## 06: Alpha sweep (`06_alpha_sweep.py`)

Run `vt.analyze_sweep` over a range of angles of attack in degrees and plot the summary. Sweeps run their independent cases in parallel (see [Run a sweep in parallel](how_to_sweeps)).

## Next steps

- Tune the machine once: [Tune the machine](how_to_tune).
- Force one backend for a test: [Force a kernel backend](how_to_backends).
- Study ground effect: [Run a ground-effect study](how_to_ground_effect).
- See all scripts: [Examples](examples).
