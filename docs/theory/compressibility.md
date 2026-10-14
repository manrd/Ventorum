# Compressibility correction

## Model

At subsonic speeds, air is compressible. The vortex-lattice method models compressibility with the Prandtl-Glauert rule in the form given by B. H. Goethert (1946). The method transforms the subsonic compressible flow past a thin wing into an equivalent incompressible flow past an affine-transformed wing. The transformation stretches coordinates along the free stream by $1/\beta$, where $\beta = \sqrt{1 - M^2}$. The method solves the vortex lattice on the transformed wing and maps the velocities and forces back to the physical wing. The method goes back to L. Prandtl (1930), H. Glauert (1928) and B. H. Goethert (1946); see also Drela (2014) and Katz and Plotkin (2001).

## Equations

### Coordinate transformation

Let $M$ be the free-stream Mach number ($0 \le M < 1$). The Prandtl-Glauert factor is

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

## Assumptions

- Subsonic, steady, inviscid flow ($M < 1$).
- Small perturbations: thin airfoils, small angles of attack and sideslip.
- Linearized potential flow equation: $(1 - M^2) \phi_{xx} + \phi_{yy} + \phi_{zz} = 0$.
- Attached flow without boundary-layer separation or shock waves.
- Free air: compressibility correction is not supported in ground effect.

## Envelope

- Valid for Mach numbers $0 \le M < 1$.
- At $M > 0.7$, local flow can become supersonic. Shocks and wave drag can appear. The trust score gives a warning above $M = 0.7$ and reduces the confidence rating.
- The VLM solver (`solver_type="vlm"`) supports the compressibility correction on the CPU path.
- The GPU pipeline falls back to the CPU path when `mach` is set.
- The lifting-line solvers (linear, nonlinear, and Fourier) refuse `mach > 0` and raise `ValidityError`.
- Ground effect refuses `mach > 0` and raises `ValidityError`.
- Adjoint gradients and gradient checks refuse `mach > 0` and raise `GradientScopeError`.

## Implementation

- `ventorum.core.datatypes.FlightCondition`: stores the optional `mach` field ($0 \le M < 1$).
- `ventorum.core.trust`: applies Mach number penalties and envelope warnings.
- `ventorum.solvers.core.assemble_vlm`: applies the coordinate stretch and normal vector scaling in free-stream axes.
- `ventorum.aero.loads._near_field_velocity`: computes transformed induced velocities and scales the streamwise component.
- `ventorum.gpu.pipeline.can_run_on_gpu`: detects `mach > 0` and routes execution to the CPU path. This is a temporary limit of the first version: the GPU path and the gradients with `mach > 0` follow.
- `ventorum.solvers.lattice_base`: validates solver compatibility and raises errors for unsupported configurations.

## Verification

- Bit-for-bit equality: solves with `mach=None` and `mach=0.0` give identical results.
- Discrete identity: solving at Mach $M$ matches the incompressible solve on an explicitly stretched geometry.
- Finite wing: aspect ratio 8 wing scales within 3% of the Helmbold-Diederich formula up to $M = 0.6$.
- Tests in `tests/test_compressibility.py` verify all rules and error conditions.

## References

- B. H. Goethert, "Plane and Three-Dimensional Flow at High Subsonic Speeds", NACA TM 1105, 1946.
- M. Drela, *Flight Vehicle Aerodynamics*, MIT Press, 2014.
- J. Katz and A. Plotkin, *Low-Speed Aerodynamics*, 2nd ed., Cambridge University Press, 2001.
- H. Glauert, "The Effect of Compressibility on the Lift of an Aerofoil", *Proceedings of the Royal Society of London*, Series A, Vol. 118, No. 779, 1928, pp. 113-119.
- L. Prandtl, "General Considerations on the Flow of Compressible Fluids", NACA TM 805, 1936 (translation of 1930 paper).
