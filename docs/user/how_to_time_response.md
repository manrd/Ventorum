# Compute time-domain unsteady response and gusts

`ventorum.time_response` marches a lifting surface or aircraft in the time domain with the prescribed-wake vortex-ring lattice solver, in free air or in flat-ground effect. It supports six-axis rigid-body motion, moving control surfaces and arbitrary convected gust or wind-shear fields. `ventorum.indicial_response` computes unit-step Wagner, Kussner and control-step responses as functions of distance travelled in semichords `s = 2 V_inf t / c_ref`.

## Basic usage: motion and moving controls

Pass an `Aircraft` or `LiftingSurface`, a mean `FlightCondition`, optional `SolverSettings`, and either `t_end` [s] or `n_steps` [-] (give exactly one).

```python
import math
import numpy as np
import ventorum as vt

flap = vt.ControlSurface(name="flap", eta_start=0.0, eta_end=1.0, hinge_x_c=0.75, deflection=0.0)
wing = vt.LiftingSurface(
    name="wing",
    semi_span=3.0,
    sections=[
        vt.WingSection(y_frac=0.0, chord=1.0),
        vt.WingSection(y_frac=1.0, chord=1.0),
    ],
    controls=[flap],
)
aircraft = vt.Aircraft(surfaces=[wing], ref_point=np.array([0.25, 0.0, 0.0]))
cond = vt.FlightCondition(V_inf=25.0, rho=1.225, alpha=math.radians(3.0))
settings = vt.SolverSettings(solver_type="vlm", n_panels=6, n_chord=8, chord_spacing="uniform")

omega = 2.0 * 0.25 * cond.V_inf / 1.0  # reduced frequency k = 0.25
res = vt.time_response(
    aircraft,
    cond,
    settings,
    n_steps=48,
    motion={"pitch": lambda t: math.radians(2.0) * math.sin(omega * t)},
    controls={"flap": lambda t: math.radians(5.0) * math.sin(omega * t + 0.25 * math.pi)},
)
print("CL range:", np.min(res.CL), np.max(res.CL))
print("Cm range:", np.min(res.Cm), np.max(res.Cm))
```

- `motion` maps any subset of `"x"`, `"y"`, `"z"` [m] (translations along the mean-flight basis) and `"roll"`, `"pitch"`, `"yaw"` [rad] (intrinsic 3-2-1 rotations about `ref_point`, AVL signs: `roll > 0` right wing down, `pitch > 0` nose up, `yaw > 0` nose right) to either a 1-D array of length `n_steps + 1` or a callable of time `t` [s].
- `controls` maps control-surface names (`ControlSurface.name`) to deflection histories [rad] (1-D array of length `n_steps + 1` or callable of `t` [s]). Moving controls require a chordwise lattice (`n_chord >= 2`).
- The rate of an array input comes from second-order differences on the time grid. The rate of a callable comes from a small central difference of the callable. A callable with a jump (a step input) has no finite rate: the solver finds the jump, uses the grid rate of the sampled values next to it (as for an array), and adds a note.
- `dt` [s] defaults to `dx_TE / V_inf`, where `dx_TE` is the strip-width-weighted mean trailing-edge panel length. A user `dt` that differs from the default by more than 25 % adds a note to `res.notes`.
- `res.CL`, `res.CD` and `res.CY` are in the mean wind axes. `res.Cl`, `res.Cm` and `res.Cn` are in the body axes of the mean state about the instantaneous reference point. All coefficients use the nominal `V_inf` and `rho` of `condition`.
- `store="loads"` (default) keeps the motion, control and gust histories, the six coefficients and `strip_cl`. Set `store="strengths"` to also keep the final `gamma_bound`, `gamma_wake` and `wake_corners`.
- `res.to_dict()` returns a strict JSON-compatible dictionary.

## Gusts and horizontal wind shear

Pass a gust callable `gust(points, t) -> velocities` [m/s] (geometry axes, shape `(n, 3)`). Ventorum provides constructors in `ventorum.unsteady.gusts` (re-exported on `ventorum`):

```python
# 1-cosine vertical gust (peak 2 m/s, total length 6 m, front starts at x = 0).
gust_cos = vt.one_minus_cosine(amplitude=2.0, length=6.0, direction="vertical", x_front=0.0)
res_gust = vt.time_response(aircraft, cond, settings, n_steps=64, gust=gust_cos)

# Sharp-edged headwind gust (adds 1.5 m/s airspeed behind the front).
gust_step = vt.sharp_edged(amplitude=1.5, direction="head", x_front=0.0)

# Logarithmic atmospheric surface-layer headwind shear near flat ground (h = 0.6 m).
cond_ge = vt.FlightCondition(V_inf=25.0, rho=1.225, alpha=math.radians(3.0), h=0.6)
profile = vt.logarithmic_shear(u_ref=8.0, z_ref=10.0, z0=0.05)
res_shear = vt.time_response(aircraft, cond_ge, settings, n_steps=48, gust=vt.wind_shear(profile, direction="head"))
```

- `direction` accepts `"vertical"` (`lift_direction(alpha)` in free air or `ground.normal` in ground effect; positive upward), `"head"` (along the mean free stream `d_0`; positive adds airspeed), `"side"` (towards the right wingtip), or a unit 3-vector in geometry axes.
- `wind_shear` models horizontal winds (`"head"`, `"side"`, or a unit 3-vector) whose speed depends on height above the ground (`ground.height`) or `z` in free air. Use `linear_shear(u0, du_dz, z_ref=0.0)`, `logarithmic_shear(u_ref, z_ref, z0)` (Stull 1988), or any user callable `profile(height) -> speed` [m/s] with an optional `time_factor(t)`.
- You can also pass any custom callable `gust(points, t) -> velocities` directly.

## Ground effect

Set `condition.h` [m] to run over a flat ground plane. The wing, deflected flaps and wake receive mirror images in the ground plane. If any wing or deflected flap node reaches or crosses the ground plane at any step, the call raises `InputError` naming the step and time.

When a gust is active in ground effect:

- Wake rows convect with only the gust component parallel to the ground so that wake nodes never cross the ground plane.
- The gust field itself is applied as given without mirror reflection. A gust with a component normal to the ground does not satisfy ground impermeability by itself.
- Both facts are recorded in `res.notes`.

## Indicial response helper

`ventorum.indicial_response` returns unit-step responses per radian (or per unit `w_g / V_inf` for `"gust"`) as functions of `s = 2 V_inf t / c_ref` [-]:

```python
wagner = vt.indicial_response(aircraft, cond, settings, kind="alpha", s_end=20.0)
kussner = vt.indicial_response(aircraft, cond, settings, kind="gust", s_end=20.0)
flap_step = vt.indicial_response(aircraft, cond, settings, kind="control", control="flap", s_end=20.0)
```

For `kind="alpha"` and `kind="control"`, `res.steady` gives the steady final derivatives from the steady VLM solver on the same lattice.

## Trust, notes and limits

- Use at least 8 chordwise panels (`n_chord >= 8`) for unsteady pitching-moment accuracy. Values below 8 add a warning to `res.trust.warnings` and `res.trust_notes` while preserving the steady trust score and rating of `res.base.totals.trust`.
- **Device and precision.** The time-domain solver runs on the CPU (float64) or on a CUDA GPU (`float64` or `float32`). Select the device with `ventorum.gpu.set_device("auto" | "cpu" | "gpu")` (or `VENTORUM_DEVICE`) and the GPU precision with `ventorum.gpu.set_precision("auto" | "float32" | "float64")` (or `VENTORUM_GPU_PRECISION`). With `"auto"`, small marches stay on the CPU and larger marches run on the GPU. `res.device` and `res.precision` record what ran, and `res.notes` states the device (or the fallback reason if CUDA is unavailable or runs out of memory). On the GPU, a Python `gust` callable runs on the host at every step and copies its velocities to the device, so it adds a host cost per step. Device memory is freed when the call ends.
- Wind shear is rotational flow. The potential-flow vortex-ring solver feels shear only as an onset velocity at wing and wake nodes; `res.notes` states this whenever `wind_shear` is used.
- Only the vortex-lattice solver (`solver_type="vlm"` or `"auto"`) is supported. Lifting-line and Fourier solvers raise `InputError`.
- Incompressible flow only: Mach numbers above 0.3 raise `InputError`.
- Prescribed wake only: wake rows convect with the local free-stream plus gust velocity without induced roll-up. No surface pressure coefficient (`Cp`) output is computed.

See `examples/14_time_response.py` for a runnable script (output on an AR 6 flapped wing with `n_panels=6, n_chord=8` at `V_inf = 25 m/s`, `alpha = 3 deg`: pitch plus flap gives `CL` in `[0.090, 0.517]` and `Cm` in `[-0.064, 0.054]`; the 2 m/s 1-cosine gust peaks at `CL = 0.482`; the logarithmic headwind shear at `h/c = 0.6` reaches `CL = 0.353`) and the [time-domain theory chapter](../theory/unsteady_time_domain.md) for the equations and verification.
