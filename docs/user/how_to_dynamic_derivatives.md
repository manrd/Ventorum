# Get unsteady dynamic derivatives

`ventorum.dynamic_derivatives` computes frequency-dependent longitudinal dynamic derivatives of a wing or aircraft in free air or in flat-ground effect. It solves small-amplitude harmonic pitch and heave oscillations about a steady mean state on a vortex-ring lattice.

## Basic usage

```python
import numpy as np
import ventorum as vt

wing = vt.LiftingSurface(
    name="wing",
    semi_span=3.0,
    sections=[
        vt.WingSection(y_frac=0.0, chord=1.0),
        vt.WingSection(y_frac=1.0, chord=1.0),
    ],
)
aircraft = vt.Aircraft(surfaces=[wing], ref_point=np.array([0.25, 0.0, 0.0]))
settings = vt.SolverSettings(solver_type="vlm", n_panels=12, n_chord=8)

# Free air at reduced frequencies k = 0, 0.05 and 0.2.
cond_free = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
res_free = vt.dynamic_derivatives(aircraft, cond_free, settings, k=(0.0, 0.05, 0.2))

print("k -> 0:", res_free.k0["CL_alpha"], res_free.k0["Cm_q"], res_free.k0["Cm_alpha_dot"])
print("Cm_q(k):", res_free.derivatives["Cm_q"])
print("Cm_alpha_dot(k):", res_free.derivatives["Cm_alpha_dot"])

# Ground effect at h / c_ref = 0.5.
cond_ge = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), h=0.5)
res_ge = vt.dynamic_derivatives(aircraft, cond_ge, settings, k=(0.0, 0.05, 0.2))

print("Ground effect k -> 0 CL_h:", res_ge.k0["CL_h"], "CL_h_dot_ground:", res_ge.k0["CL_h_dot_ground"])
```

- `k` is a sequence of non-negative reduced frequencies `k = omega c_ref / (2 V_inf)` [-]. The default is `(0.0,)` (the `k -> 0` limit only).
- `res.k0` always holds the `k -> 0` limits, even when `0.0` is not in `k`.
- `res.derivatives` maps each derivative name to a 1-D array aligned with `res.k`.
- `res.H_theta`, `res.H_z` and `res.H_pureq` hold the underlying complex frequency-domain responses per coefficient.
- `res.CL0`, `res.Cm0` and `res.mean_state` hold the steady mean-state coefficients about which the harmonic motion is linearised.
- `res.to_dict()` returns a strict JSON-compatible dictionary (complex responses are split into `"re"` and `"im"` lists).

## Derivative keys and units

For each coefficient `C` in `CL`, `CD`, `CY`, `Cl`, `Cm` and `Cn`, both `res.k0` and `res.derivatives` contain six keys:

| Key | Definition | Unit |
| --- | --- | --- |
| `<C>_alpha` | `Re H_theta(k)`, pitch incidence derivative | 1/rad |
| `<C>_q` | `Im H_pureq(k) / k`, pitch-rate derivative per unit `q c_ref / (2 V_inf)` | - |
| `<C>_alpha_dot` | `Im H_theta(k) / k - C_q(k)`, incidence-rate derivative per unit `(dalpha/dt) c_ref / (2 V_inf)` | - |
| `<C>_h` | `Re H_z(k)`, height derivative per unit `h / c_ref` (zero in free air) | - |
| `<C>_h_dot` | `Im H_z(k) / (2 k)`, heave-rate derivative per unit `(dz/dt) / V_inf` | - |
| `<C>_h_dot_ground` | `C_h_dot(k) + C_alpha(k)`, ground-effect part of the heave-rate derivative | - |

- `axes="body"` (default) returns moments in the body axes of the mean state. `axes="stability"` rotates the moments by `alpha` about the `y` axis (leaving `CL`, `CD`, `CY` and `Cm` unchanged).
- Pitch rotation is about `Aircraft.ref_point` (default the origin) at constant height. Set `ref_point` to the centre of gravity.
- For a symmetric configuration in symmetric flight, the lateral keys (`CY_*`, `Cl_*`, `Cn_*`) are zero.

## Mesh, wake and trust notes

- Use at least 8 chordwise panels (`n_chord >= 8`) for unsteady pitching-moment derivatives (`Cm_q`, `Cm_alpha_dot`). A chordwise count below 8 adds a warning to `res.trust.warnings` and `res.trust_notes`, while keeping the steady trust score and rating of the mean state (`res.base.totals.trust`).
- The default discrete wake length is `wake_chords=5.0` reference chords, closed to infinity by the exact 3D harmonic far-wake tail. Values below 5 chords add a note to `res.trust_notes`.
- All strips share one wake time step `dt = dx_w / V_inf`. When a strongly tapered wing gives a trailing-edge panel length ratio `dx_TE,s / dx_w` outside `[0.8, 1.25]`, `res.trust_notes` records the span of the ratio.
- Reduced frequencies above `k = 1.0` add a note that the frequency is outside the verified test range.

## Device and GPU execution

`dynamic_derivatives` follows the device and precision settings of `ventorum.gpu` (`gpu.set_device`, `gpu.set_precision`, `VENTORUM_DEVICE`, `VENTORUM_GPU_PRECISION`). When CUDA is selected, the solver assembles the bound, wake and tail influence matrices with NVIDIA Warp kernels and solves the reduced frequencies in batches with PyTorch on the GPU. `res.device` and `res.precision` (and `res.to_dict()`) record the device and precision used. Call `ventorum.gpu.release()` to free registered ring device arrays. When CUDA is unavailable or the GPU runs out of memory twice, the solver falls back to the CPU and records a fallback note in `res.notes` and `res.trust.warnings`.

## Limits

- Only the vortex-lattice solver (`solver_type="vlm"` or `"auto"`) is supported. Lifting-line and Fourier solvers are refused with `InputError`.
- Incompressible flow only: Mach numbers above 0.3 (either stated in `condition.mach` or from `V_inf / a`) are refused with `InputError`.
- Geometries that cannot map cleanly to four-corner vortex rings (surface junctions, curved mean-line leg chains with `n_chord > 1`, and deflected chordwise flaps with hinge kinks) are refused with `InputError`.
- Longitudinal motions (pitch and heave) only. Lateral dynamic rates (`p`, `r`, `beta_dot`) are not yet included.

See `examples/13_dynamic_derivatives.py` for a runnable script and the [theory chapter](../theory/unsteady_vortex_rings.md) for the equations, definitions and verification.
