# Solvers

| `solver_type` | Method | Use | Limits |
| --- | --- | --- | --- |
| `"auto"` (default) | Always `"vlm"` | | |
| `"vlm"` (alias `"horseshoe"`) | Vortex lattice with horseshoe vortices; one or more chordwise panels; induced drag in the Trefftz plane | Any planform: sweep, dihedral, taper, twist, several surfaces, ground effect | Linear section data. A tabulated polar is used through its linear part, with a warning. |
| `"linear"` | Numerical lifting line (Phillips and Snyder, 2000) | Unswept wings of moderate to high aspect ratio | Not grid convergent with sweep or a kinked quarter-chord line (a warning is given). Refused in ground effect below h_min/c = 1. |
| `"nonlinear"` | The same lifting line, solved with Newton iteration on the section polars | Start of stall with tabulated polars on unswept wings | As `"linear"`. Far past the maximum lift a case can fail to converge; this is reported. |
| `"fourier"` | Classical Fourier series (Glauert, 1926) | One symmetric, unswept, planar wing | No sideslip, no ground effect |

:::{note}
The kernel backends (numpy, Numba, Cython, PyTorch) are measured by the tuner per kernel and case size. See [Parallel execution and tuning](parallel.md) and [Force a kernel backend](how_to_backends).
:::

## Scope

- Incompressible flow, Mach number below 0.3. Above Mach 0.3 a case is outside the envelope: `analyze` gives a warning and the trust rating is LOW or UNRELIABLE.
- Attached flow. Results after stall are not reliable.
- Thin lifting surfaces. Thickness enters only through the section polars.
- Not modelled: compressibility, leading-edge and side-edge vortex lift, wake roll-up, unsteady flow, a fuselage.
