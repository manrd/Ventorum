# Classical lifting line

## Model

The classical solver (`solver_type="fourier"`) solves the Lanchester–Prandtl lifting-line equation for one straight, unswept, planar wing in symmetric flight. The wing is a single bound vortex on the quarter-chord line, with a sheet of trailing vortices that leaves it along the free stream. Each section acts as a two-dimensional airfoil at its effective angle of attack. H. Glauert (1926) wrote the circulation as a Fourier sine series in the angle $\theta$ and collocated the equation at a set of span stations. This gives a small, dense linear system for the coefficients and closed forms for the lift and the induced drag.

## Equations

The span is $b$ [m] and the span station is $y = \tfrac{b}{2}\cos\theta$, with $0 < \theta < \pi$ ($\theta = 0$ at the right tip). The circulation [m^2/s] is

```{math}
\Gamma(\theta) = 2 b V_\infty \sum_{n=1}^{N} A_n \sin(n\theta).
```

The trailing vortex sheet gives the induced angle of attack [rad] (Glauert 1926; Katz and Plotkin 2001, section 8.1)

```{math}
\alpha_i(\theta) = \sum_{n=1}^{N} n A_n \frac{\sin(n\theta)}{\sin\theta}.
```

The section has a linear lift curve, $C_l = a_0(\alpha_\text{eff} - \alpha_{L0})$, with $\alpha_\text{eff} = \alpha + \epsilon - \alpha_i$, where $\epsilon$ [rad] is the twist plus the surface incidence. The Kutta-Joukowski law gives $\Gamma = \tfrac12 V_\infty c\, C_l$. Put the series into these two relations and divide by $2 b V_\infty$:

```{math}
:label: eq-monoplane
\sum_{n=1}^{N} A_n \sin(n\theta) \left( 1 + \frac{n \mu}{\sin\theta} \right) = \mu \,(\alpha + \epsilon - \alpha_{L0}),
\qquad \mu = \frac{c\, a_0}{4 b}.
```

This is Glauert's form of the monoplane equation. Ventorum collocates it at the $N$ stations

```{math}
\theta_k = \frac{k\pi}{N+1}, \qquad k = 1, \dots, N,
```

which avoid the tips ($\theta = 0$ and $\pi$). The section data ($c$, $\epsilon$, $a_0$, $\alpha_{L0}$) at a station are the values at the span fraction $|\cos\theta_k|$, so the system is the same on both halves. For a symmetric wing the even coefficients are then zero to round-off.

The integrated results follow from the orthogonality of the sine functions (Glauert 1926):

```{math}
C_L = \pi\, AR\, A_1, \qquad
C_{Di} = \pi\, AR \sum_{n=1}^{N} n A_n^2, \qquad
e = \frac{C_L^2}{\pi\, AR\, C_{Di}},
```

with $AR = b^2/S_\text{ref}$. The section results at each station are $C_l = 2\Gamma/(V_\infty c)$, the induced drag coefficient $C_{d,i} = C_l\,\alpha_i$ and the local lift $\rho V_\infty \Gamma$ [N/m].

The profile drag uses the section value $C_{d0}$: $C_{Dp} = \frac{1}{S_\text{ref}} \int_{-b/2}^{b/2} c\, C_{d0}\, dy$.

### Pitching moment

The pitching moment about the reference point $(x_\text{ref}, z_\text{ref})$ has two parts.

1. The section moments: $M_0 = q_\infty \int c^2 C_{m0}\, dy$ [N m].
2. The moment of the section forces. The force per unit span, $\rho V_\infty \Gamma$ [N/m], is normal to the local flow, which the downwash turns by $\alpha_i$. It acts at the quarter-chord point $(x_{qc}, z_{qc})$ and has a lift part normal to the free stream, along $(-\sin\alpha, 0, \cos\alpha)$, and an induced-drag part $\alpha_i$ times as large along the free stream, $(\cos\alpha, 0, \sin\alpha)$. Thus

```{math}
dF_x = \rho V_\infty \Gamma\,(-\sin\alpha + \alpha_i \cos\alpha)\, dy, \qquad
dF_z = \rho V_\infty \Gamma\,(\cos\alpha + \alpha_i \sin\alpha)\, dy,
```

```{math}
M_y = \int \left[ (z_{qc} - z_\text{ref})\, dF_x - (x_{qc} - x_\text{ref})\, dF_z \right] + M_0,
\qquad C_m = \frac{M_y}{q_\infty S_\text{ref} c_\text{ref}}.
```

The force integral uses Gauss-Legendre quadrature in $\theta$ with $8N$ points ($dy = \tfrac{b}{2}\sin\theta\, d\theta$). The section data are linear between the given sections, so $c^2 C_{m0}$ is a cubic and $c\,C_{d0}$ a quadratic on each interval; Simpson's rule on each interval gives these integrals exactly.

## Assumptions

- One lifting surface, symmetric about the x-z plane, with a straight and unswept quarter-chord line in one plane.
- The trailing vortices are straight, in the plane of the wing, and parallel to the free stream; small angles ($\alpha_i \ll 1$).
- Linear section data. A tabulated polar enters through its lift slope and zero-lift angle from a linear fit (see [Numerical lifting line](numerical_lifting_line.md)).
- Inviscid, incompressible flow; high aspect ratio (each section acts as in two-dimensional flow).

## Envelope

The solver refuses a geometry or a condition outside the model, with the error `ValidityError`, and names the vortex-lattice solver in the message:

- quarter-chord sweep above `ventorum.solvers.fourier.SWEEP_LIMIT_DEG` on any interval between two sections. A tapered wing with a straight leading edge has a swept quarter-chord line, so it is refused; give it with a straight quarter-chord line;
- dihedral, or a vertical offset of a section, above a small fraction (`PLANAR_TOL_FRACTION`) of the semi-span;
- sideslip ($\beta \neq 0$);
- ground effect ($h$ given).

Inside the envelope, the model has the limits of lifting-line theory: low aspect ratio, stall and compressibility are outside it. The trust score lowers the rating for them (see [Trust score](trust_score.md)).

## Implementation

- `ventorum.solvers.fourier.FourierSolver`: `solve` (one angle) and `solve_sweep` (one factorisation, many right-hand sides).
- `ventorum.geometry.discretization.fourier_collocation_angles`: the stations $\theta_k$.
- `_fourier_system`: the matrix of equation {eq}`eq-monoplane`, the quadrature data and the exact span integrals; cached per geometry and $N$.
- `_check_geometry_in_model`: the geometry refusal.
- The number of terms $N$ is the panel count of the surface (`LiftingSurface.n_panels`), or `SolverSettings.n_panels` when the surface does not set it.
- An independent solution of the same equation, with odd terms only, is `ventorum.reference.glauert_monoplane`; the verification uses it.

## Verification

- [V1](../verification_report.md): elliptic wing, $C_{L\alpha} = a_0/(1 + a_0/(\pi AR))$ and $e = 1$.
- [V2](../verification_report.md): rectangular and tapered wings against the independent Fourier solution.
- [V8](../verification_report.md): camber and section pitching moment.

## References

- F. W. Lanchester, *Aerodynamics*, Constable, London, 1907.
- L. Prandtl, "Tragflügeltheorie. I. Mitteilung", *Nachrichten von der Gesellschaft der Wissenschaften zu Göttingen, Mathematisch-Physikalische Klasse*, 1918, pp. 451-477; "II. Mitteilung", 1919, pp. 107-137.
- H. Glauert, *The Elements of Aerofoil and Airscrew Theory*, Cambridge University Press, 1926, chapter XI.
- J. Katz and A. Plotkin, *Low-Speed Aerodynamics*, 2nd ed., Cambridge University Press, 2001, section 8.1.
