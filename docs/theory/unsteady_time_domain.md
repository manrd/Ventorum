# Time-domain unsteady vortex-ring lattice

## Model

The time-domain vortex-ring lattice solver (`ventorum.unsteady.timedomain._solve_time_rings`, exposed through `ventorum.time_response`, `ventorum.indicial_response` and, coupled to the rigid-body equations of motion, `ventorum.free_flight`) marches a lifting surface or aircraft along a prescribed six-axis rigid-body path with optional moving control surfaces and arbitrary convected gust or wind-shear velocity fields, in free air or over a flat ground plane. At each time step, the solver builds the instantaneous wing and flap geometry, enforces impermeability at every panel collocation point, sheds one row of quadrilateral wake rings behind the trailing edge with the trailing-edge circulation of the previous step, convects existing wake nodes with the local air velocity without induced deformation, and evaluates unsteady Kutta-Joukowski, apparent-mass and trailing-edge boundary loads.

## Equations

### Time march and wake shedding

A vortex-lattice mesh (`collocation="vlm"`) with $n_s$ spanwise strips and $n_c$ chordwise panels per strip maps to $N = n_s n_c$ quadrilateral vortex rings (`ventorum.unsteady.rings.build_ring_lattice`; Katz and Plotkin 2001, chapter 13). All strips share one default convective time step [s]:

```{math}
\Delta x_{\text{TE,mean}} = \frac{\sum_{s=1}^{n_s} w_s\,\Delta x_{\text{TE},s}}{\sum_{s=1}^{n_s} w_s}, \qquad
\Delta t_{\text{default}} = \frac{\Delta x_{\text{TE,mean}}}{V_\infty}, \qquad
\Delta x_w = V_\infty \Delta t.
```

Time instants are $t_n = n \Delta t$ [s] for $n = 0, 1, \dots, n_\text{steps}$, and nondimensional distance travelled in semichords is $s_n = 2 V_\infty t_n / c_\text{ref}$ [-]. The run starts impulsively with zero wake rows at $n = 0$ (Katz and Plotkin 2001, section 13.11). At every step $n \ge 1$, one new wake row leaves the instantaneous trailing-edge closure with circulation equal to the trailing-edge bound ring strength of step $n - 1$ (unsteady Kutta condition; Katz and Plotkin 2001, section 13.8):

```{math}
\Gamma_{w,s}^{(n,1)} = \Gamma_{\text{TE},s}^{(n-1)}.
```

### Six-axis rigid-body kinematics

Geometry axes are $x$ aft, $y$ right and $z$ up. The mean free stream is $\mathbf V_0 = V_\infty \mathbf d_0$ [m/s]. Translations $(x, y, z)$ [m] move the reference point $\mathbf x_{\text{ref},0}$ along the mean-flight basis $(\mathbf e_x, \mathbf e_y, \mathbf e_z)$, where $\mathbf e_x = \mathbf d_0$, $\mathbf e_z$ is `lift_direction(alpha)` in free air or `ground.normal` in ground effect, and $\mathbf e_y = \mathbf e_z \times \mathbf e_x$ (unit vector pointing right).

Rotations $(\phi, \theta, \psi)$ (`roll`, `pitch`, `yaw` [rad]) about the reference point follow the intrinsic 3-2-1 sequence of flight mechanics (yaw about $-z$, pitch about rotated $+y$, roll about rotated $-x$; Etkin and Reid 1996, chapter 4) with AVL signs (`roll > 0` right wing down, `pitch > 0` nose up, `yaw > 0` nose right). With body-to-reference rotation matrix $\mathbf R(t)$, each body point $\mathbf p_b$ moves to $\mathbf p(t) = \mathbf x_\text{ref}(t) + \mathbf R(t)(\mathbf p_b - \mathbf x_{\text{ref},0})$ with rigid-body velocity [m/s]:

```{math}
\mathbf v_\text{body}(\mathbf p, t) = \mathbf v_\text{trans}(t) + \boldsymbol\omega(t) \times \bigl(\mathbf p - \mathbf x_\text{ref}(t)\bigr),
```

where $\boldsymbol\omega(t)$ [rad/s] comes from the 3-2-1 Euler-angle rates (Etkin and Reid 1996, chapter 4).

#### Input rates

The rate of an array input comes from second-order finite differences on the time grid (one-sided at the two ends). The rate of a callable input comes from a central difference of the callable with the step $h = 10^{-7}\,\Delta t$ (one-sided at $t = 0$). A jump in a callable has no finite rate, so the solver tests the samples. With the sampled values $v_i$, the rates $\dot v_i$ and $\Delta v_i = |v_i - v_{i-1}|$:

1. A rate is too large at $t_i$ when $|\dot v_i|\,\Delta t > 100 \max(\Delta v_i, \Delta v_{i+1})$ (a jump at $t_i$; a smooth or kinked input gives a ratio near 1, a jump about $5 \times 10^6$). The samples $i-1$, $i$ and $i+1$ are marked.
2. A change is too large on $[t_{i-1}, t_i]$ when $\Delta v_i > 2\,\Delta t \max(|\dot v_{i-1}|, |\dot v_i|)$ (a jump between the two samples; a resolved smooth input gives about $1 + (\omega \Delta t)^2/8$, a kink at most 1). The samples $i-1$ and $i$ are marked.

Both tests have a round-off floor of $10^{-9} \max_i |v_i|$. A marked sample takes the grid rate of the array rule, so near a jump a callable gives the rates of its sampled array, the trapezoidal integral of the rate equals the jump, and a note gives the times. Resolved smooth inputs and continuous inputs with a kink pass both tests and keep their rates.

### Moving control surfaces

Time-varying control deflections $\delta(t)$ [rad] rotate the vortex-ring corners, collocation points, boundary normals, side polylines and physical trailing edges aft of the hinge line about each strip hinge axis by $\Delta\delta(t) = \delta(t) - \delta_0$, using Rodrigues' rotation formula and the sealed hinge geometry of `ventorum.geometry.lattice`. Points on the moving flap add the hinge rotation velocity $\mathbf v_\text{flap} = \boldsymbol\omega_\text{hinge} \times (\mathbf p - \mathbf H)$ [m/s] (rotated into mean-flight axes).

In free air with rigid-body motion only, the real bound influence matrix $\mathbf A_{bb}$ is invariant and LU-factorised once. When a control surface deflects, only the columns of rings whose corners move and the rows of moved flap collocation points are recomputed in $\mathbf A_{bb}$ before refactorisation. In ground effect, the mirror image of the moved wing, flaps and wake updates at every step where the pose or flap deflections change.

### Gust onset and prescribed wake convection

A gust field supplies an air velocity $\mathbf v_g(\mathbf p, t)$ [m/s] in geometry axes added to the mean free stream $\mathbf V_0$:

```{math}
\mathbf V_\text{onset}(\mathbf p, t) = \mathbf V_0 + \mathbf v_g(\mathbf p, t).
```

This local onset enters three places at every step:

1. **Impermeability boundary condition** at moved collocation points $\mathbf p_c$:

```{math}
\left(\mathbf A_{bb} - \mathbf A_{bb,\text{img}}\right)\boldsymbol\Gamma^{(n)}
= \left(\mathbf v_\text{surf} - \mathbf V_\text{onset} - \mathbf v_\text{wake}\right) \cdot \mathbf n_{bc}.
```

2. **Kutta-Joukowski loads** at bound spanwise segments and chordwise legs, using $\mathbf V_\text{tot} = \mathbf V_\text{onset} - \mathbf v_\text{surf} + \mathbf v_\text{ind}$.
3. **Trailing-edge boundary load correction** at the moved physical trailing edge for $n \ge 1$, with $\Delta\gamma_{\text{TE},s} = f_{\text{TE},s}(\Gamma_{w,s}^{(n,1)} - \Gamma_{\text{TE},s}^{(n)})$ and $f_{\text{TE},s} = \Delta x_{\text{TE},s} / (2(\Delta x_{\text{TE},s} + \Delta x_w))$.

Apparent-mass loads add $\rho\,(\mathrm d\Gamma_p / \mathrm dt)\,\mathbf A_p$ over each physical ring area vector $\mathbf A_p$ [m$^2$], using a second-order backward difference for $n \ge 2$ and a first-order backward difference at $n = 1$ (zero at $n = 0$; Katz and Plotkin 2001, chapter 13). Force coefficients $(C_L, C_D, C_Y)$ project onto mean wind axes and moment coefficients $(C_l, C_m, C_n)$ use body axes of the mean state about $\mathbf x_\text{ref}(t)$, all normalised by the nominal $V_\infty$ and $\rho$ of the mean condition.

Without a gust, wake nodes convect with the uniform base wake velocity $V_\infty \hat{\mathbf d}_s$ [m/s]. When a gust is given, the stationary wake-row cache is turned off and existing wake node lines advect from $t_{n-1}$ to $t_n$ by classical fourth-order Runge-Kutta (RK4) integration of:

```{math}
\frac{\mathrm d\mathbf p}{\mathrm dt} = V_\infty \hat{\mathbf d}_s + \mathbf v_{g,\text{conv}}(\mathbf p, t),
```

where $\mathbf v_{g,\text{conv}} = \mathbf v_g$ in free air and $\mathbf v_{g,\text{conv}} = \mathbf v_g - (\mathbf v_g \cdot \mathbf n_g)\mathbf n_g$ in ground effect so that wake nodes stay above and parallel to the ground plane.

The built-in gust fields (`ventorum.unsteady.gusts`) use streamwise distance behind the gust front $\xi(\mathbf p, t) = x_\text{front} + V_\infty t - \mathbf p \cdot \mathbf d_0$ [m]:

- **Sharp-edged gust** (`sharp_edged`): $\mathbf v_g = A\,\hat{\mathbf u}$ for $\xi \ge 0$ and $\mathbf 0$ for $\xi < 0$ (Bisplinghoff, Ashley and Halfman 1955, chapter 5).
- **1-cosine discrete gust** (`one_minus_cosine`): $\mathbf v_g = \tfrac{1}{2}A\bigl(1 - \cos(2\pi\xi / L)\bigr)\hat{\mathbf u}$ for $0 \le \xi \le L$ and $\mathbf 0$ outside (Hoblit 1988, chapter 2).
- **Horizontal wind shear** (`wind_shear`): $\mathbf v_g = f(t)\,U(h)\,\hat{\mathbf u}$ where $h$ is height above the ground (`ground.height`) or, in free air, the coordinate $h = \mathbf p \cdot \mathbf e_z$ along the mean-flight vertical $\mathbf e_z$ = `lift_direction(alpha)` from the geometry origin (so points on one line along the mean flight path see one wind speed; $h = z$ at $\alpha = 0$). Built-in profiles are `linear_shear` ($U(h) = u_0 + (\mathrm du/\mathrm dz)(h - z_\text{ref})$) and `logarithmic_shear` for the neutral atmospheric surface layer (Stull 1988, chapter 9; Garratt 1992, chapter 3):

```{math}
U(h) = U_\text{ref}\,\frac{\ln(h / z_0)}{\ln(z_\text{ref} / z_0)} \quad (h > z_0), \qquad
U(h) = 0 \quad (h \le z_0).
```

### Indicial response definitions

`ventorum.indicial_response` computes unit-step responses per unit step amplitude (subtracting the zero-perturbation baseline transient when the mean state carries lift) as functions of $s = 2 V_\infty t / c_\text{ref}$ [-]:

- `kind="alpha"` (**Wagner**, Wagner 1925): uniform vertical gust $w_g = V_\infty \Delta\alpha$ [m/s] over the whole wing at $t = 0$. Reference 2D approximation (Jones 1940, NACA Report 681):

```{math}
\Phi(s) \approx 1 - 0.165\,e^{-0.0455 s} - 0.335\,e^{-0.300 s}.
```

- `kind="gust"` (**Kussner**, Kussner 1936): sharp-edged vertical gust $w_g = V_\infty \Delta\alpha_g$ [m/s] with $x_\text{front}$ at the foremost leading-edge point along $\mathbf d_0$ at $t = 0$. Reference 2D two-exponential approximation (Sears and Sparks 1941; Bisplinghoff, Ashley and Halfman 1955, chapter 5):

```{math}
\psi(s) \approx 1 - 0.500\,e^{-0.130 s} - 0.500\,e^{-1.000 s}.
```

- `kind="control"`: step deflection $\Delta\delta$ [rad] of a named control surface at $t = 0$ with zero hinge rate for $t \ge 0^+$.

### Free-body flight coupling

`ventorum.free_flight` replaces the prescribed pose of the selected degrees of freedom with the rigid-body equations of motion (`ventorum.unsteady.freeflight`). The inertial frame is the mean-flight frame $(\mathbf e_x, \mathbf e_y, \mathbf e_z)$ above; it translates at $\mathbf V_0$ [m/s]. Gravity acts along $-\mathbf e_z$ with $g$ [m/s^2]. The loads on the body are the aerodynamic force and moment of the step (from the coefficients with the nominal $q_\infty S_\text{ref}$, moved from the reference point to the centre of gravity), the weight, and the thrust force $\mathbf F_T$ [N] and moment $\mathbf M_T$ [N m] in body axes at the point $\mathbf r_T$ [m].

With the centre of gravity at the body offset $\mathbf r$ [m] from the reference point, the generalised coordinates are the pose rates $\mathbf u = [\dot{\mathbf p}, \dot{\boldsymbol\eta}]$, with $\mathbf p = (x, y, z)$ and $\boldsymbol\eta = (\phi, \theta, \psi)$. The body angular velocity is $\boldsymbol\omega_b = \mathbf S(\phi, \theta)\,\dot{\boldsymbol\eta}$ and the centre-of-gravity velocity is $\mathbf v_{cg} = \dot{\mathbf p} + \mathbf J \dot{\boldsymbol\eta}$, with $\mathbf J = -\mathbf E^T \mathbf R [\mathbf r]_\times \mathbf S$. The Newton and Euler equations about the centre of gravity (Etkin and Reid 1996, chapter 4; Stevens and Lewis 2003, chapter 1), projected on the partial velocities (Kane and Levinson 1985), give

```{math}
\mathbf G^T
\begin{bmatrix} m\mathbf I & 0 \\ 0 & \mathbf I_b \end{bmatrix}
\mathbf G\,\dot{\mathbf u}
= \mathbf G^T
\begin{bmatrix}
\mathbf F_{\text{net},I} - m\,\mathbf a_\text{vel} \\
\mathbf M_{cg,b} - \boldsymbol\omega_b \times \mathbf I_b \boldsymbol\omega_b - \mathbf I_b \dot{\mathbf S}\dot{\boldsymbol\eta}
\end{bmatrix},
\qquad
\mathbf G = \begin{bmatrix} \mathbf I & \mathbf J \\ 0 & \mathbf S \end{bmatrix},
```

where $\mathbf a_\text{vel} = \mathbf E^T \mathbf R\bigl((\dot{\mathbf S}\dot{\boldsymbol\eta}) \times \mathbf r + \boldsymbol\omega_b \times (\boldsymbol\omega_b \times \mathbf r)\bigr)$ is the velocity-dependent part of the centre-of-gravity acceleration. Only the rows of the free degrees of freedom are solved; the prescribed degrees of freedom enter with their accelerations on the right-hand side. The 3-2-1 kinematics are singular at $|\theta| = 90$ deg, and pitch angles of 89 deg or more are refused.

Each step couples the loads and the motion implicitly. The free state at step $n + 1$ is predicted by second-order extrapolation, the loads of step $n + 1$ are computed from the restored solver state, and the equations of motion are integrated from $n$ to $n + 1$ with the trapezoidal rule. The sub-iterations repeat with Aitken dynamic relaxation (Irons and Tuck 1969; Kuttler and Wall 2008) until the relative change of the state is below `coupling_tol` (default $10^{-10}$), with at most `max_subiterations` (default 20) per step. The apparent-mass load makes the plain fixed-point iteration slow or unstable for light bodies; the relaxation removes most of this effect.

The run starts impulsively, so the load at $t = 0$ is not the steady load. The steady imbalance of the result is the steady vortex-lattice load of the mean state (with $C_{D_i}$) plus the weight, about the centre of gravity, without thrust. A constant thrust equal to minus this imbalance trims the mean state. In ground effect, the run stops when a wing or flap node reaches the ground, and the result holds the valid steps.

## Assumptions

- Incompressible, inviscid, attached potential flow (Mach $\le 0.3$).
- Thin lifting surfaces represented by quadrilateral vortex rings.
- Prescribed wake without induced wake roll-up or vortex-sheet stretching.
- In ground effect, the gust field is applied as given (not mirror-reflected); a gust with a ground-normal component does not satisfy ground impermeability by itself, and wake convection uses only the ground-parallel gust component.
- Wind shear is rotational flow. The potential-flow solver neglects background vorticity transport and feels shear only as a spatially varying onset velocity at the wing and wake nodes.

## Envelope

- **Solver scope.** Only `solver_type="vlm"` or `"auto"` is accepted; lifting-line and Fourier solvers raise `InputError`.
- **Compressibility.** Mach numbers above `MACH_LIMIT_INCOMPRESSIBLE` ($0.3$) raise `InputError`.
- **Ground clearance.** Any wing or deflected flap node reaching or crossing the ground plane at any step raises `InputError`.
- **Device and precision.** Time-domain marching runs on the CPU (float64) or on a CUDA GPU (`float64` or `float32`, via `ventorum.gpu.set_device` and `ventorum.gpu.set_precision`). With `"auto"`, the solver asks `ventorum.gpu.use_gpu` with march work $n_\text{steps} N N_w$ (doubled in ground effect, where $N$ is the bound ring count and $N_w = n_\text{steps} n_s$ is the final wake ring count). Requesting `"gpu"` without CUDA, or two GPU out-of-memory errors, falls back to the CPU in float64 with a note in `notes` and `trust.warnings`.
- **Chordwise resolution.** `n_chord < 8` adds a trust warning while keeping the steady trust score and rating of the mean state.

## Implementation

- `ventorum.unsteady.timedomain`: `_TimeDomainStepper`, `_solve_time_rings`, `_TimeDomainResponse`, `_select_time_device`, 3-2-1 kinematics, moving control rig, RK4 gust wake convection. On the GPU, `_TimeDomainStepper` uses `ring_velocity_gpu` and `ring_normal_influence_gpu` (`ventorum.gpu.ring_kernels`) in the selected precision (`float64` or `float32`), solves the bound system with PyTorch LU (`torch.linalg.lu_factor` and `torch.linalg.lu_solve`, factorised once when the matrix does not change), keeps the wake corner and circulation buffers on the device (so `save_state` and `restore_state` copy device buffers without host round trips), evaluates Python gust callables on the host per step and copies the velocities to the device, and releases its registered device buffers (`register_device_holder` and `release_gpu`) when the run ends.
- `ventorum.unsteady.gusts`: `sharp_edged`, `one_minus_cosine`, `wind_shear`, `linear_shear`, `logarithmic_shear`.
- `ventorum.unsteady.response`: `time_response`, `TimeResponseResult`, `indicial_response`, `IndicialResult`, `_device_execution_notes` (records `device`, `precision` and execution or fallback notes).
- `ventorum.unsteady.freeflight`: the implicit free-body coupling (`_solve_free_flight`) and `FlightState`.
- `ventorum.unsteady.flight`: `free_flight` and `FreeFlightResult`.
- `ventorum.agent.tools`, `ventorum.agent.schemas`, `ventorum.agent.dispatcher`: `ventorum_time_response`, `ventorum_indicial_response` and `ventorum_free_flight` tools.

## Verification

- `tests/test_unsteady_timedomain.py`: steady limit against `ventorum.analyze`; AR 40 Wagner build-up against Jones (1940); small-amplitude periodic limit against `_solve_harmonic_rings` in free air and ground effect; second-order time-step convergence; moving-flap and six-axis ground-image parity against explicit two-wing models; NumPy vs Numba backend parity.
- `tests/test_time_response.py`: round-off parity between `time_response` and `_solve_time_rings` with six-axis motion and moving flaps; zero-gust parity; uniform time-varying head-wind gust equivalence with `x_dot = -g` (free air and $h/c = 0.5$) and vertical gust equivalence with `z_dot = -w` (free air) to round-off; AR 40 Wagner (`kind="alpha"`) and Kussner (`kind="gust"`) verification against Jones (1940), Sears and Sparks (1941) / Bisplinghoff et al. (1955) and the exact 2D Kussner Fourier integral; 1-cosine gust agreement with Duhamel superposition of the solver's sharp-edged response; steady linear horizontal shear long-time limit against a direct steady ring solve; ground-effect horizontal gust mirror parity and vertical gust wake clearance; control-step limit against `ventorum.control_derivatives`; JSON serialisation, `store` options, trust and input checks.
- `tests/test_agent_time_response.py`: agent dispatcher and MCP server parity, waveform and gust parsers, work budget refusal and input checks.
- `tests/test_unsteady_freeflight.py`: all degrees of freedom frozen equals `_solve_time_rings` to round-off (free air and $h/c = 0.5$, with a moving flap and a gust); free fall, constant thrust and an offset centre of gravity without aerodynamics against the exact paths; torque-free rotation with energy and angular momentum conserved at second order in $\Delta t$; the large-mass limit; trimmed short-period frequency and damping against the linear model from `dynamic_derivatives`, in free air and at $h/c = 0.5$; second-order time-step convergence; a pitch damper against the linear model with the control derivatives; thrust callables; ground contact; input checks.
- `tests/test_free_flight.py` and `tests/test_agent_free_flight.py`: the public function and the agent tool equal the private coupling to round-off; steady-imbalance trim; the coupling options; JSON output, notes, ground contact and input checks.
- `tests/test_callable_jumps.py`: smooth, kinked and constant callables keep their rates bit for bit; a step at a sample time and between samples gives the rates of the sampled array and a note; `time_response` and `free_flight` with a step callable equal the run with its sampled array (the free-flight case diverged before); the agent step waveform on a sample time.
- `tests/test_gpu_time_march.py`: float64 (1e-11 relative) and float32 (float32 floor) GPU against CPU parity for `time_response`, `indicial_response` (`kind="alpha"`) and `free_flight` in free air and at $h/c = 0.5$ with six-axis motion, moving flaps, 1-cosine gusts, cambered rings and stationary ground caches; device-memory release; GPU out-of-memory and CUDA-hidden subprocess fallbacks; CPU device notes and automatic work selection.

## References

- J. Katz and A. Plotkin, *Low-Speed Aerodynamics*, 2nd ed., Cambridge University Press, 2001, chapter 13 (time-domain vortex-ring lattice, impulsive start, unsteady Kutta condition, unsteady Bernoulli loads and method of images).
- B. Etkin and L. D. Reid, *Dynamics of Flight: Stability and Control*, 3rd ed., Wiley, 1996, chapter 4 (3-2-1 Euler rotation sequence and angular-velocity kinematics, rigid-body equations of motion).
- B. L. Stevens and F. L. Lewis, *Aircraft Control and Simulation*, 2nd ed., Wiley, 2003, chapter 1 (rigid-body equations and Euler-angle kinematics).
- T. R. Kane and D. A. Levinson, *Dynamics: Theory and Applications*, McGraw-Hill, 1985 (partial velocities and the generalised equations of motion).
- B. M. Irons and R. C. Tuck, "A version of the Aitken accelerator for computer iteration", *International Journal for Numerical Methods in Engineering* 1, 1969, pp. 275-277.
- U. Kuttler and W. A. Wall, "Fixed-point fluid-structure interaction solvers with dynamic relaxation", *Computational Mechanics* 43, 2008, pp. 61-72.
- H. Wagner, "Uber die Entstehung des dynamischen Auftriebes von Tragflugeln", *ZAMM* 5(1), 1925, pp. 17-35 (indicial lift after an incidence step).
- H. G. Kussner, "Zusammenfassender Bericht uber den instationaren Auftrieb von Flugeln", *Luftfahrtforschung* 13(12), 1936, pp. 410-424 (sharp-edged gust penetration).
- R. T. Jones, "The Unsteady Lift of a Wing of Finite Aspect Ratio", NACA Report 681, 1940 (two-exponential Wagner approximation and finite-span tip relief).
- W. R. Sears and B. O. Sparks, "On the Reaction of an Elastic Wing to Vertical Gusts", *Journal of the Aeronautical Sciences* 9(2), 1941, pp. 64-67 (two-exponential Kussner approximation).
- R. L. Bisplinghoff, H. Ashley and R. L. Halfman, *Aeroelasticity*, Addison-Wesley, 1955, chapter 5 (Wagner, Kussner and discrete gust response).
- F. M. Hoblit, *Gust Loads on Aircraft: Concepts and Applications*, AIAA, 1988, chapter 2 (1-cosine discrete gust profile).
- R. B. Stull, *An Introduction to Boundary Layer Meteorology*, Kluwer Academic Publishers, 1988, chapter 9 (logarithmic wind profile of the neutral atmospheric surface layer).
- J. R. Garratt, *The Atmospheric Boundary Layer*, Cambridge University Press, 1992, chapter 3 (atmospheric surface-layer logarithmic wind profile).
