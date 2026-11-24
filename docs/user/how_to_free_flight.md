# Compute free-body flight with the time-domain solver

`ventorum.free_flight` couples the prescribed-wake vortex-ring lattice solver of `ventorum.time_response` to the rigid-body equations of motion. The aircraft moves freely in the degrees of freedom that you select, under the aerodynamic loads, gravity, a point thrust force and moment, and control deflections that can depend on time or on the state. The run is in free air or over flat ground (`condition.h` set), in incompressible flow (Mach 0.3 or less).

## Inputs

- `geometry`, `condition`, `settings`: as in `ventorum.time_response`. The pose, the rotations and the coefficients use `Aircraft.ref_point` [m].
- `mass` [kg] and `inertia` [kg m^2]: the inertia tensor about the centre of gravity in geometry axes, shape (3, 3), symmetric and positive definite.
- `cg` [m]: the centre of gravity in geometry axes. The default is `Aircraft.ref_point`.
- `dofs`: the free degrees of freedom, a subset of `("x", "y", "z", "roll", "pitch", "yaw")`. The default is all six. The other degrees of freedom follow `motion` (default zero).
- `t_end` [s] or `n_steps`: give exactly one. `dt` [s] defaults to `dx_TE / V_inf`.
- `initial_state`: initial displacements (`"z"`, `"pitch"`, ...) [m, rad] and rates (`"z_dot"`, `"pitch_dot"`, ...) [m/s, rad/s] of the free degrees of freedom.
- `controls`: control deflections [rad] keyed by control name, as arrays, callables `fn(t)` or callables `fn(t, state)`.
- `thrust`: a callable `thrust(t)` or `thrust(t, state)` that returns `(force, moment)` in body axes [N, N m]. `thrust_point` [m] is the point of the force (default the centre of gravity).
- `gust`: a gust field from `ventorum.unsteady.gusts`, as in `ventorum.time_response`.
- `gravity` [m/s^2]: default 9.80665.

`state` is a `ventorum.FlightState`: the time `t`, the six displacements `x, y, z, roll, pitch, yaw`, the six rates `x_dot, ..., yaw_dot` and the current control deflections `controls`. Inside the coupling sub-iterations of a step, the record holds the current iterate.

## Frames and signs

The inertial frame is the mean-flight frame. It translates at the constant mean flight velocity of `condition`. Its x axis is aft along the mean flight path, its z axis is up (along the lift direction in free air, along the ground normal in ground effect), and its y axis is to the right. Gravity acts along -z. The pose is the displacement of `ref_point` and the 3-2-1 Euler rotation about it, with AVL signs: roll > 0 right wing down, pitch > 0 nose up, yaw > 0 nose right. A positive control deflection moves the trailing edge down.

## Trim the mean state

The march starts impulsively: there is no wake at t = 0, and the starting wake forms over the first chord lengths of travel. The load at t = 0 is therefore not the steady load. On the wing-tail aircraft of `examples/15_free_flight.py` at alpha = 2 deg with 8 chordwise panels, CL is 0.046 at t = 0 and the steady CL is 0.160. Do not trim with `initial_force_imbalance_body`: it cancels the load at t = 0 only.

To trim the mean state, use the steady imbalance. `steady_force_imbalance_body` and `steady_moment_imbalance_body` are the steady vortex-lattice load of the mean state (with the induced drag `CDi`) plus the weight, about the centre of gravity, in body axes, without thrust. A constant thrust equal to minus these values trims the mean state:

```python
import numpy as np
import ventorum as vt

# aircraft, cond and settings as in examples/15_free_flight.py
probe = vt.free_flight(aircraft, cond, settings, mass=80.0, inertia=np.diag([16.0, 24.0, 36.0]), dofs=(), n_steps=1)
force_b = -probe.steady_force_imbalance_body
moment_b = -probe.steady_moment_imbalance_body

res = vt.free_flight(
    aircraft,
    cond,
    settings,
    mass=80.0,
    inertia=np.diag([16.0, 24.0, 36.0]),
    dofs=("z", "pitch"),
    n_steps=100,
    thrust=lambda t: (force_b, moment_b),
    controls={"elevator": lambda t, state: 0.10 * state.pitch_dot},  # pitch damper
)
```

The body still moves a little while the starting wake forms. Start a disturbance after this transient, for example a few chord lengths of travel after t = 0.

## Results

`res` is a `ventorum.FreeFlightResult`. It holds the histories of the pose and rates, the controls, the thrust, the coefficients `CL, CD, CY` (mean wind axes) and `Cl, Cm, Cn` (body axes of the mean state), the dimensional aerodynamic force and moment at the centre of gravity in the inertial frame, the kinetic and potential energy, the work of the aerodynamic loads and of the thrust, the angular momentum, the initial and steady imbalances, the coupling sub-iterations and residuals per step, the trust record and the notes. `res.to_dict()` gives a JSON-safe dictionary. `store="strengths"` also keeps the final ring strengths and wake corners.

## Ground effect

With `condition.h` set, the mean state is over flat ground. When a wing or flap node reaches the ground in free flight, the run stops at that step. The result holds the valid steps, `res.ground_contact` is True, `res.ground_contact_step` gives the step, and a note gives the time and the lowest node.

## Notes and limits

- The coupling is implicit, with sub-iterations and Aitken relaxation. `coupling_tol` [-] (default 1e-10) is the relative state tolerance and `max_subiterations` (default 20) the limit per step. A step that does not reach the tolerance gets a note, and the run continues. Light bodies need more sub-iterations, because the apparent-mass load is large compared with the inertia: increase `max_subiterations` for them.
- The rate of a time callable comes from a small central difference. A jump in a callable (a step input) has no finite rate. The solver finds a jump from the sampled values and rates, uses the grid rate of the sampled values next to it, as for an array input, and adds a note. The impulse of the jump is then spread over two time steps. For a defined input rate, give a smooth input (for example a sine doublet) or an array.
- **Device and precision.** The solver runs on the CPU (float64) or on a CUDA GPU (`float64` or `float32`), controlled by `ventorum.gpu.set_device` (`VENTORUM_DEVICE`) and `ventorum.gpu.set_precision` (`VENTORUM_GPU_PRECISION`). On the GPU, wake buffers stay on the device across coupling sub-iterations (only the step loads return to the host) and are freed when the call ends. `res.device` and `res.precision` record what ran, and `res.notes` states the device or any fallback reason. A Python `gust` callable is evaluated on the host at every step and copies its velocities to the device, which adds a host cost per step.
- Use at least 8 chordwise panels for the unsteady pitching moment; fewer panels give a trust warning.
- The wake is prescribed: it convects with the free stream plus the gust, with no roll-up. Only the vortex-lattice solver is supported.

See `examples/15_free_flight.py` for a runnable script. On the example wing-tail aircraft (80 kg, I_yy = 24 kg m^2, V_inf = 25 m/s, alpha = 4 deg, z and pitch free, trimmed with the steady imbalance), a 2 deg sine elevator doublet gives a pitch excursion of -0.237 deg; with the pitch damper `delta = doublet + 0.10 pitch_dot` the excursion is -0.189 deg. The [time-domain theory chapter](../theory/unsteady_time_domain.md) gives the equations and the verification.
