# Compressibility correction

## Model

At subsonic speeds, air is compressible. The vortex-lattice method models compressibility with the Prandtl-Glauert rule in the form given by B. H. Goethert (1946). The method transforms the subsonic compressible flow past a thin wing into an equivalent incompressible flow past an affine-transformed wing. The transformation stretches coordinates along the free stream by $1/\beta$, where $\beta = \sqrt{1 - M^2}$. The method solves the vortex lattice on the transformed wing and maps the velocities and forces back to the physical wing. The method goes back to L. Prandtl (1930), H. Glauert (1928) and B. H. Goethert (1946); see also Drela (2014) and Katz and Plotkin (2001).

## Equations

### Coordinate transformation

Let $M$ be the free-stream Mach number ($0 \le M < 0.7$, see [Envelope](#envelope)). The Prandtl-Glauert factor is

```{math}
\beta = \sqrt{1 - M^2} .
```

The transformation aligns with the free-stream direction $\hat{\mathbf u}_\infty$. Let $\mathbf R$ be the orthogonal rotation matrix from geometry axes to free-stream axes:

```{math}
\mathbf x_\text{fs} = \mathbf R\, \mathbf x, \qquad
\mathbf R = \begin{bmatrix} \hat{\mathbf u}_\infty^T \\ \hat{\mathbf u}_\text{side}^T \\ \hat{\mathbf u}_\text{lift}^T \end{bmatrix} .
```

The transformed coordinates $\mathbf x^*$ stretch along the free-stream axis:

```{math}
x_\text{fs}^* = \frac{x_\text{fs}}{\beta}, \qquad
y_\text{fs}^* = y_\text{fs}, \qquad
z_\text{fs}^* = z_\text{fs} .
```

In geometry axes, the transformed coordinates are $\mathbf x^* = \mathbf R^T \mathbf x_\text{fs}^*$.

### Boundary condition

The flow-tangency condition on the physical wing requires zero total normal velocity at each control point. On the transformed geometry, the effective normal vector in free-stream axes scales as

```{math}
\mathbf n_\text{fs}^* = \begin{bmatrix} n_{\text{fs}, x} / \beta \\ n_{\text{fs}, y} \\ n_{\text{fs}, z} \end{bmatrix} .
```

The effective normal vector in geometry axes is $\mathbf n^* = \mathbf R^T \mathbf n_\text{fs}^*$. The influence matrix $A^*$ is built from the transformed bound vortices and control points with the effective normal:

```{math}
A_{ij}^* = \mathbf v_j^*(\mathbf x_{cp, i}^*) \cdot \mathbf n_i^* .
```

The right-hand side uses the physical free-stream velocity:

```{math}
b_i = -\mathbf V_\infty \cdot \mathbf n_i .
```

The circulation vector $\mathbf \Gamma$ solves the linear system $A^* \mathbf \Gamma = \mathbf b$.

### Induced velocity and forces

The near-field perturbation velocity $\mathbf v^*$ is evaluated on the transformed geometry. In free-stream axes, the physical perturbation velocity $\mathbf v_\text{fs}$ scales as

```{math}
v_{\text{fs}, x} = \frac{v_{\text{fs}, x}^*}{\beta}, \qquad
v_{\text{fs}, y} = v_{\text{fs}, y}^*, \qquad
v_{\text{fs}, z} = v_{\text{fs}, z}^* .
```

The physical perturbation velocity in geometry axes is $\mathbf v = \mathbf R^T \mathbf v_\text{fs}$.

The near-field panel forces follow from the Kutta-Joukowski law with the physical total velocity $\mathbf V_\infty + \mathbf v$. The induced drag in the Trefftz plane uses the trailing vortex circulations and downstream wake geometry.

### Scaling of lift

In two-dimensional flow, the lift-curve slope scales with the classical Prandtl-Glauert rule:

```{math}
C_{L\alpha}(M) = \frac{C_{L\alpha}(0)}{\beta} .
```

On a finite wing of aspect ratio $\text{AR}$ and sweep angle $\Lambda$, the lift-curve slope follows the Helmbold-Diederich formula (Drela 2014):

```{math}
C_{L\alpha}(M) = \frac{2\pi\, \text{AR}}{2 + \sqrt{\text{AR}^2\beta^2 (1 + \tan^2\Lambda / \beta^2) + 4}} .
```

The vortex-lattice solver reproduces this scaling.

### Section quantities

The Prandtl-Glauert rule scales every section pressure coefficient by $1/\beta$ (Glauert 1928; Anderson, *Fundamentals of Aerodynamics*, chapter on subsonic compressible flow over airfoils). The solve on the stretched lattice gives this scaling to the lift and to the camber lift. Two section quantities are added after the solve, so they get the scaling there:

- The constant section moment $C_{m0}$ of a linear airfoil:

  ```{math}
  C_{m0}(M) = \frac{C_{m0}(0)}{\beta} .
  ```

  A strip on a camber-line lattice gets its camber moment from the loads of the mean surface, which the stretched lattice already scales. A tabulated airfoil keeps its table moment.

- The effective section angle, found from the strip lift with the compressible section slope $a_0/\beta$:

  ```{math}
  \alpha_\text{eff} = \alpha_{L0} + \frac{C_l\,\beta}{a_0} .
  ```

  The induced angle $\alpha_i = \alpha_\text{geom} - \alpha_\text{eff}$ and the angle of the lookup in a tabulated polar ($C_d$ and $C_m$) use this angle.

## Assumptions

- Subsonic, steady, inviscid flow ($M < 0.7$, see [Envelope](#envelope)).
- Small perturbations: thin airfoils, small angles of attack and sideslip.
- Linearized potential flow equation: $(1 - M^2) \phi_{xx} + \phi_{yy} + \phi_{zz} = 0$.
- Attached flow without boundary-layer separation or shock waves.
- Free air: compressibility correction is not supported in ground effect.

## Envelope

- Scope: incompressible flow, Mach number below 0.3. With `FlightCondition.mach` from 0.3 to below 0.7, the vortex lattice applies the Prandtl-Glauert correction in the form of Goethert; the result is an estimate with the trust rating UNRELIABLE. A Mach number of 0.7 or more is refused with `ValidityError` (`ventorum.core.constants.MACH_LIMIT_REFUSED`).
- Above $M = 0.3$ (`ventorum.core.constants.MACH_LIMIT_INCOMPRESSIBLE`) the trust score gives a warning and a penalty that grows with the Mach number. With $M \ge 0.7$ the local flow can become supersonic near the critical Mach number; shocks and wave drag can appear, and the linear rule has no meaning there.
- A speed above $M = 0.3$ with `mach` None or 0 gets the same warning and penalty, with $M = V_\infty / a$ (ISA speed of sound at the air density); no correction is applied.
- A stated Mach number that differs from $V_\infty / a$ by more than 0.05 (`ventorum.core.constants.MACH_SPEED_TOLERANCE`) gives a `RuntimeWarning` and a trust note. Here $a$ is the ISA sea-level speed of sound, 340.29 m/s, because the flight condition has no temperature. The correction uses the stated Mach number.
- The VLM solver (`solver_type="vlm"`) supports the compressibility correction on the CPU and GPU paths.
- The lifting-line solvers (linear, nonlinear, and Fourier) refuse `mach > 0` and raise `ValidityError`.
- Ground effect refuses `mach > 0` and raises `ValidityError`.
- The 3D panel method has no compressibility correction: it refuses `mach > 0` and raises `ValidityError`.
- Adjoint gradients support `mach > 0` for the vortex lattice in free air,
  with the new input `"mach"` (per unit Mach); ground effect and the lifting
  lines refuse it with `GradientScopeError`.

## Implementation

- `ventorum.core.datatypes.FlightCondition`: stores the optional `mach` field (the constructor accepts $0 \le M < 1$).
- `ventorum.core.trust.check_mach_scope`: refuses $M \ge 0.7$; `ventorum.utils.validation.validate_flight_condition` and the lattice solvers call it.
- `ventorum.core.trust.mach_speed_mismatch`: the comparison of the stated Mach number with the speed.
- `ventorum.aero.loads.compute_loads_batch` and the GPU loads kernels: the section slope $a_0/\beta$ and $C_{m0}/\beta$.
- `ventorum.core.trust`: applies Mach number penalties and envelope warnings.
- `ventorum.solvers.core.assemble_vlm`: applies the coordinate stretch and normal vector scaling in free-stream axes.
- `ventorum.aero.loads._near_field_velocity`: computes transformed induced velocities and scales the streamwise component.
- `ventorum.gpu.pipeline`: solves VLM batches with `mach > 0` on the GPU by transforming coordinates and normals per Mach group.
- `ventorum.sensitivity.reference`: maps the lattice points and normals to the
  stretched wing inside the differentiable reference and runs the kernel
  products there; the near-field velocity maps back with `v_x / beta`.
- `ventorum.sensitivity.adjoint` and `ventorum.sensitivity.check`: the `"mach"`
  input and the scope rule (vortex lattice in free air only).
- `ventorum.solvers.lattice_base`: validates solver compatibility and raises errors for unsupported configurations.

## Verification

- Bit-for-bit equality: solves with `mach=None` and `mach=0.0` give identical results.
- Discrete identity: solving at Mach $M$ matches the incompressible solve on an explicitly stretched geometry.
- Finite wing: aspect ratio 8 wing scales within 3% of the Helmbold-Diederich formula up to $M = 0.6$.
- Gradients: `check_gradients` at Mach 0.3 and 0.6 for every supported input
  and for `"mach"` (wing and tail with sideslip and control, float64, 1e-6).
- Section quantities: $C_{m0}$ of a linear airfoil at $M = 0.5$ equals the $M = 0$ value divided by $\beta$; the effective angle uses $a_0/\beta$ (`tests/test_compressibility_consistency.py`, with the GPU parity of both).
- Tests in `tests/test_compressibility.py` and `tests/test_compressibility_consistency.py` verify all rules and error conditions.

## References

- B. H. Goethert, "Plane and Three-Dimensional Flow at High Subsonic Speeds", NACA TM 1105, 1946.
- M. Drela, *Flight Vehicle Aerodynamics*, MIT Press, 2014.
- J. Katz and A. Plotkin, *Low-Speed Aerodynamics*, 2nd ed., Cambridge University Press, 2001.
- H. Glauert, "The Effect of Compressibility on the Lift of an Aerofoil", *Proceedings of the Royal Society of London*, Series A, Vol. 118, No. 779, 1928, pp. 113-119.
- L. Prandtl, "General Considerations on the Flow of Compressible Fluids", NACA TM 805, 1936 (translation of 1930 paper).
