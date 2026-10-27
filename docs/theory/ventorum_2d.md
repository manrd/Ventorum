---
orphan: true
---

# Ventorum-2D: Airfoil geometry, panel method and viscous-inviscid interaction

## Model

Ventorum-2D provides two-dimensional airfoil analysis and section polar generation for the Ventorum aerodynamic solver suite. The system consists of six integrated parts:
1. Airfoil geometry tools that read standard coordinate formats, generate analytical NACA 4-digit and 5-digit profiles, resample nodes with curvature-adapted splines, and deflect plain trailing-edge flaps.
2. A potential-flow panel solver that uses a stream-function Dirichlet formulation with linear-strength vortex panels and constant-strength source panels. The solver models free-air flow by two-right-hand-side superposition and ground effect by the method of images.
3. An integral boundary layer with the published closures of Drela and Giles (1987): the momentum and kinetic-energy integral equations, the shear-stress lag equation and the envelope $e^N$ transition model.
4. A fully simultaneous viscous-inviscid solver: one global Newton iteration on the boundary-layer variables of both surfaces and of the wake, with the edge velocity eliminated through the source sensitivities of the panel method (wall and wake transpiration, Lighthill 1958).
5. A batched solver that runs the same equations for many cases with torch on the CPU or on a CUDA GPU.
6. A polar pipeline that encapsulates section solvers behind a standard protocol and builds multidimensional interpolation tables over Reynolds number, flap deflection, ground height, and angle of attack.

## Coordinate convention and geometry

### Selig order

An airfoil section is represented as a two-dimensional floating-point array of shape $(N, 2)$ in Selig order. The coordinates start at the trailing edge, progress along the upper surface forward to the leading edge, and return along the lower surface to the trailing edge. The node sequence is counterclockwise, so the signed shoelace area is positive:

```{math}
A_{\text{poly}} = \frac{1}{2} \sum_{i=1}^{N-1} (x_i y_{i+1} - x_{i+1} y_i) > 0.
```

A closed trailing edge repeats the same coordinate pair at the first and last indices ($x_1 = x_N, y_1 = y_N$). An open trailing edge contains distinct upper and lower trailing-edge points separated by a finite gap.

A normalized section has unit chord length ($c = 1.0$), with its leading edge located at the coordinate origin $(0, 0)$ and its trailing-edge midpoint located on the positive horizontal axis at $(1, 0)$. The `normalized` method transforms any input polygon to this reference frame by three steps:
1. Translate the leading edge to $(0, 0)$.
2. Rotate the trailing-edge midpoint onto the positive x axis.
3. Scale all coordinates by the inverse distance between the leading edge and the trailing-edge midpoint.

The leading edge is the point farthest from the trailing-edge midpoint. The same definition holds everywhere in the airfoil package: `ventorum.airfoil.geometry.leading_edge_index` gives the stored polygon point, and the spline of the repaneling uses the continuous form of the same definition (the spline point farthest from the trailing-edge midpoint). Surface separation splits the polygon into upper and lower surfaces, each directed from leading edge to trailing edge. Camber and thickness distributions follow from the surface ordinates:

```{math}
y_c(x) = \frac{1}{2}\left(y_{\text{upper}}(x) + y_{\text{lower}}(x)\right), \qquad
t(x) = y_{\text{upper}}(x) - y_{\text{lower}}(x).
```

### File formats

The geometry reader supports three plain-text formats:
- Selig format: an optional header line with the airfoil name, followed by coordinate pairs $x, y$ in Selig order.
- Lednicer format: an airfoil name line, a line specifying the point counts of the upper and lower surfaces, the upper surface points from leading edge to trailing edge, a blank line, and the lower surface points from leading edge to trailing edge. The reader joins the surfaces and merges the duplicated leading edge. A Lednicer file without a name line is detected when its first line holds two whole numbers of at least 2 whose sum equals the number of data lines.
- Plain format: coordinate pairs $x, y$ without header text.

Comment lines starting with `#` and blank lines are ignored. Files stored in clockwise order are detected by negative signed polygon area and reversed to Selig order.

### Geometry validation

The function `problems` checks candidate airfoil polygons for five geometric defects:
1. Point count fewer than 10 nodes.
2. Consecutive duplicated nodes with distance smaller than $10^{-12}$ chord units.
3. Clockwise vertex ordering (negative signed area).
4. Self-intersecting perimeter segments.
5. Non-monotonic chordwise distributions along individual surfaces.

The function `validate` raises an `AirfoilGeometryError` when any hard defect (1 to 4) is present. Non-monotonic chordwise points generate diagnostic notes because real leading edges can possess locally vertical tangents.

## NACA airfoils

### Designations

The NACA functions accept only these forms of a designation: "NACA 2412", "naca2412", "2412" and the integer 2412, and the same forms of a 5-digit designation. Hyphens, underscores and inner spaces raise an error. A 4-digit designation with zero camber and a nonzero position digit (for example 0412) gives a warning.

### Thickness distribution

The 4-digit and 5-digit NACA series share the analytical thickness envelope of Abbott and von Doenhoff (1959). For a maximum thickness ratio $t = t_{\text{max}} / c$ at chord station $x \in [0, 1]$, the half-thickness $y_t$ is:

```{math}
y_t = 5 t \left(0.2969 \sqrt{x} - 0.1260 x - 0.3516 x^2 + 0.2843 x^3 + a_4 x^4\right).
```

The coefficient $a_4$ selects the trailing-edge closure:
- Open trailing edge ($a_4 = -0.1015$): preserves the original NACA specification with a finite trailing-edge thickness $y_t(1) = 0.0105 t$.
- Closed trailing edge ($a_4 = -0.1036$): closes the section to zero thickness at $x = 1.0$ (Ladson, Brooks, Hill and Sproles, 1996).

### NACA 4-digit mean line

The 4-digit camber line uses maximum camber $m$ located at chordwise position $p$:

```{math}
y_c(x) = \begin{cases}
\frac{m}{p^2}\left(2 p x - x^2\right), & 0 \le x < p, \\
\frac{m}{(1-p)^2}\left((1 - 2p) + 2 p x - x^2\right), & p \le x \le 1.
\end{cases}
```

The local mean-line slope $dy_c/dx$ is:

```{math}
\frac{dy_c}{dx} = \begin{cases}
\frac{2m}{p^2}(p - x), & 0 \le x < p, \\
\frac{2m}{(1-p)^2}(p - x), & p \le x \le 1.
\end{cases}
```

### NACA 5-digit mean line

Standard non-reflex 5-digit mean lines follow Jacobs, Ward and Pinkerton (1933) and Jacobs and Pinkerton (1935). With position parameter $m$ and scale factor $k_1$:

```{math}
y_c(x) = \begin{cases}
\frac{k_1}{6}\left(x^3 - 3 m x^2 + m^2 (3 - m) x\right), & 0 \le x \le m, \\
\frac{k_1 m^3}{6}(1 - x), & m < x \le 1.
\end{cases}
```

The corresponding slope is:

```{math}
\frac{dy_c}{dx} = \begin{cases}
\frac{k_1}{6}\left(3 x^2 - 6 m x + m^2 (3 - m)\right), & 0 \le x \le m, \\
-\frac{k_1 m^3}{6}, & m < x \le 1.
\end{cases}
```

Parameters for design lift coefficient $C_{L,\text{des}} = 0.3$ (first digit 2) are (Abbott and von Doenhoff 1959, section 6.5):

| Mean-line designation | Position parameter $m$ | Scale factor $k_1$ |
| --- | --- | --- |
| 210 | 0.0580 | 361.40 |
| 220 | 0.1260 | 51.64 |
| 230 | 0.2025 | 15.957 |
| 240 | 0.2900 | 6.643 |
| 250 | 0.3910 | 3.230 |

For other design lift coefficients, $k_1$ scales linearly with the first digit: $k_1(d) = k_1(2) \cdot d / 2$.

### Surface coordinate construction

Surface points are displaced perpendicular to the mean camber line:

```{math}
\theta = \arctan\left(\frac{dy_c}{dx}\right), \qquad
x_u = x - y_t \sin\theta, \qquad y_u = y_c + y_t \cos\theta, \qquad
x_l = x + y_t \sin\theta, \qquad y_l = y_c - y_t \cos\theta.
```

Chordwise distribution uses cosine spacing $x = \frac{1}{2}(1 - \cos\beta)$ with uniform parameter $\beta \in [0, \pi]$.

## Panel node distributions

Panel methods require smooth node spacing with high resolution in regions of large surface curvature.

### Curvature-based repaneling

The function `repanel` redistributes nodes along the airfoil perimeter:
1. Arc-length spline fit: Parametric cubic splines with not-a-knot end conditions (de Boor, 1978) parameterize coordinates $x(s)$ and $y(s)$ as functions of cumulative polygon arc length $s$.
2. Curvature calculation: Local signed curvature $\kappa(s)$ follows from parametric derivatives:

```{math}
\kappa(s) = \frac{x'(s) y''(s) - y'(s) x''(s)}{\left((x'(s))^2 + (y'(s))^2\right)^{3/2}}.
```

3. Weight function: A node-density weight function $w(s)$ combines the curvature magnitude, normalized by its peak value, with Gaussian refinement terms centered at the leading edge ($s_{\text{le}}$) and at both trailing-edge ends ($s = 0$ and $s = s_{\text{total}}$ in Selig order):

```{math}
w(s) = 1 + c_\kappa \frac{|\kappa(s)|}{\max_s |\kappa|} + c_{\text{le}} \exp\left[-\left(\frac{s - s_{\text{le}}}{\sigma_{\text{le}}}\right)^2\right] + c_{\text{te}} \left\{\exp\left[-\left(\frac{s}{\sigma_{\text{te}}}\right)^2\right] + \exp\left[-\left(\frac{s_{\text{total}} - s}{\sigma_{\text{te}}}\right)^2\right]\right\}.
```

4. Equidistribution: New node stations are chosen such that the cumulative weight integral is partitioned into equal increments:

```{math}
W(s_k) = \int_0^{s_k} w(\tau) \, d\tau = \frac{k}{N-1} \int_0^{s_{\text{total}}} w(\tau) \, d\tau.
```

5. Ratio smoothing: Panel lengths are smoothed iteratively so that the ratio of adjacent panel lengths does not exceed 1.5. When the smoothing cannot meet this bound, `repanel` gives a warning. The leading edge point is preserved as an exact node.

The panel count must be at least 3.

### Cosine repaneling

The function `repanel_cosine` resamples upper and lower surfaces independently using Chebyshev cosine spacing in surface arc length. This clusters nodes near both the leading edge and the trailing edge.

## Plain-flap geometric deflection

The function `deflect_plain_flap` deflects the trailing-edge portion of an airfoil section about a specified hinge point. The deflection angle $\delta$ [deg] is positive when the trailing edge moves downward (clockwise rotation in the section plane).

The geometric construction follows four steps:
1. Hinge location: The hinge chord station $x_{\text{hinge}} \in (0.05, 0.98)$ and the hinge height $y_{\text{hinge}} \in [0, 1]$, a fraction of the local thickness from the lower surface (0) to the upper surface (1), define the hinge point:

```{math}
\mathbf{x}_h = \left(x_{\text{hinge}}, \, y_{\text{lower}}(x_{\text{hinge}}) + y_{\text{hinge}} \left[y_{\text{upper}}(x_{\text{hinge}}) - y_{\text{lower}}(x_{\text{hinge}})\right]\right).
```

2. Rigid rotation: All points with $x > x_{\text{hinge}}$ rotate rigidly about $\mathbf{x}_h$ by deflection angle $\delta$.
3. Compressed surface cut: The sign of the deflection selects the compressed side: the lower surface when the trailing edge moves down ($\delta > 0$), the upper surface when it moves up. On that side the fixed forward surface and the rotated aft surface overlap. Both polylines are cut at their crossing point; when they do not cross, they are cut at the hinge line.
4. Stretched surface arc: On the other side a gap opens. A circular arc centered at $\mathbf{x}_h$ connects the fixed forward curve to the rotated aft curve.

The output is checked: it must be a simple polygon (no self-intersection) with positive area. Otherwise the function raises an error.

The geometric deflection alters the enclosed polygon area by less than 2 % for deflections up to 10 deg.

## Inviscid panel method

The full description, with the closed-form influences, the branch cuts of the source stream function and the verification, is in [](panel2d.md). Summary:

- Stream-function Dirichlet formulation (Katz and Plotkin 2001, chapters 10 and 11; Drela 1989): linear-strength vortex panels with node values $\gamma_j$, constant-strength source panels with prescribed strengths $\sigma_k$, and $\psi = \psi_b$ at every node.
- Kutta condition $\gamma_{first} + \gamma_{last} = 0$. A sharp trailing edge replaces the two coincident node equations by linear extrapolations of $\gamma$; a blunt trailing edge has a gap panel with a constant vortex sheet and a constant source sheet set by the trailing-edge speeds (Drela 1989).
- Exact closed forms of all influences, with an equivalent far-field form (log1p and atan2) that keeps float32 accuracy.
- Free air: one LU factorization per geometry, two right-hand sides superposed for any angle. Sources add right-hand-side columns only.
- Ground effect: the method of images about the flat ground $y = 0$ (mirror vortices of opposite sign, mirror sources of the same sign); the geometry is rotated about the pivot, and `height_reference="pivot"` places the pivot (default the quarter chord) at the given height.
- $C_l$ by Kutta-Joukowski of the total circulation in free air, and by surface pressure integration in ground effect ($C_{l,circulation}$ keeps the circulation lift); pressure forces by exact integration of $C_p = 1 - \gamma^2$ over each wall panel.
- Reference values: the reference chord (`chord_ref`) and the moment reference point (`moment_ref`) are inputs of the solvers, the batch solver, the polar source and the agent tool. By default they come from the reference geometry of the section: the chord line from the leading edge to the trailing-edge midpoint of the section as given, its length, and the quarter-chord point on it. A flapped `AirfoilShape` holds the reference geometry of the section before the deflection, as in thin-airfoil theory (Glauert 1927); the polar source and the agent tool pass the flapped shape. The default pivot of the ground-effect placement is the same quarter-chord point.

## Integral boundary layer

The closures are in [](ibl2d_closure.md) and the discrete equations, the start conditions and the direct march in [](ibl2d_march.md). Summary:

- Momentum and kinetic-energy integral equations, the shear-stress lag equation in turbulent flow and in the wake (the lag-entrainment idea of Green, Weeks and Brooman 1977), and the envelope $e^N$ equation in laminar flow (Drela and Giles 1987; Drela 1986).
- Laminar closures of the Falkner-Skan family, turbulent closures of the Swafford profile family, wake closures with zero skin friction and doubled dissipation. Each closure returns analytic partial derivatives and runs with numpy or torch.
- One implementation of the discrete equations (`ventorum/ibl2d/equations.py`) for the direct march and for the coupled solver: logarithmic differences, trapezoidal source terms with the logarithmic mean of $1/u_e$, a stagnation-point similarity start, a transition interval split at the transition point, and an amplification increase that integrates the switch of the published growth rate exactly.

## Viscous-inviscid coupling

The full description is in [](panel2d_viscous.md). Summary:

- Wall and wake transpiration (Lighthill 1958): source panels of strength $\sigma = dm/d\xi$, $m = u_e\delta^*$, on the airfoil and on a wake line that follows the inviscid streamline from the trailing-edge bisector (one chord and 30 panels by default).
- The edge velocity is eliminated: $u_e = u_{e,inv} + D\,m$, with $D$ built from the source sensitivities of the factorized inviscid system (with images in ground effect).
- One global Newton iteration on $(\theta, m, c)$ at all stations of both sides and of the wake (fully simultaneous coupling; Drela and Giles 1987), with the exact Jacobian. The stagnation point and the transition intervals are updated between iterations.
- The first wake station takes $\theta_w = \theta_u + \theta_l$ and $\delta^*_w = \delta^*_u + \delta^*_l + h_{TE}$ (Drela 1989).
- Lift: `Cl` is the surface pressure lift (`Cl_pressure`) in ground effect, as in the inviscid solver (Coulliette and Plotkin 1996, eq. 30; see [](panel2d.md)). `Cl_circulation` keeps $2\Gamma/c$ in both solvers. In free air, the viscous `Cl` stays equal to `Cl_circulation`, with the same bits as before. The sequential and batched solvers use this rule.
- Drag: Squire-Young at the wake end (Squire and Young 1938). The skin-friction drag and the surface-pressure drag are given separately as a check.
- A case that does not converge returns `converged=False`, its residual, its work counters and a reason; its loads and distributions are NaN (the last iterate is not a solution). A singular linear system gives a failed angle with a reason, not an exception. There are no result overrides and no fallback to other models.
- One body only: more bodies raise `ValueError`. A plain flap is part of the single-element geometry.

Only the quasi-simultaneous first estimate of a cold start may retry. The cold start at the zero-lift angle of an angle continuation does not retry: in the test sample no retry there converged, and each retry used its full stage limit. A retry requires a residual-window stagnation stop. It retries once from that state, with only the residual-window rule off. The retry retains the coupling factor, transition history, damping, equations and tolerance. The attempt and retry together use the same stage iteration limit and the remaining work budget of the angle. Direct first estimates, warm starts and angle-continuation stages keep their original stop rules. A failed line search with a failed Levenberg-Marquardt fallback, divergence or another failure does not start a retry. The reason names the first stop and the retry outcome. The iteration and linear-solve counts include all retry work. A stage that converges without this stop follows the same path and keeps the same result bits.

## Batched execution and angle continuation

`ventorum.panel2d.viscous_batch.solve_viscous_batch` runs the same algorithm and the same equations with torch on the CPU or on CUDA. Cases with the same geometry and settings form angle sequences. Each pass solves one angle of each sequence, and the cases with equal node and wake counts share one batched Newton iteration (`torch.linalg.solve`). The first angle of a sequence and the failed first attempts use the fallback path of the sequential solver. In float64 the batch equals the sequential solver.

The sequential solver solves the angles $\ge 0$ upwards and the angles $< 0$ downwards, each from the converged state of the previous angle, with halved angle steps when needed. An angle without a converged neighbour starts cold at the inviscid zero-lift angle and continues to the angle in steps of at most 2 deg.

## Polar sources and tables

### Section polar protocol

The protocol `SectionPolarSource` defines the standard interface for 2D section polar generation:

```{math}
\text{polar}(\text{section}, Re, \boldsymbol{\alpha}, \dots) \to \text{TabulatedAirfoil}.
```

`Ventorum2DPolarSource` implements this protocol using the in-house viscous panel solver. It supports automated on-disk caching through `PolarCache`.

### Multidimensional polar tables

`PolarTable` provides multidimensional interpolation over Reynolds number, flap deflection, ground height, and angle of attack:
- Free-air tables store data in three dimensions: $(n_{Re}, n_{\text{flap}}, n_\alpha)$.
- Ground-effect tables store data in four dimensions: $(n_{Re}, n_{\text{flap}}, n_h, n_\alpha)$.
- Interpolation is linear in $\log_{10} Re$, linear in flap deflection $\delta_{\text{flap}}$, and linear in inverse height $1/h$ (reflecting image singularity decay). The result at one operating point is a tabulated polar: in angle of attack $\alpha$ it uses shape-preserving cubic (PCHIP) interpolation, and outside its angle range it holds the end values (no extrapolation in $\alpha$).
- Unconverged points are recorded as NaN. Queries outside the table bounds raise errors rather than extrapolating unverified data.

## Assumptions

- Steady, incompressible, irrotational outer potential flow.
- Boundary layer thickness small relative to chord and surface radius of curvature.
- Attached flow or mild trailing-edge separation; deep post-stall separation is outside the physical model envelope.
- One body (single element; a plain flap is part of the geometry).
- Flat, rigid ground; the image method gives the moving-ground (inviscid ground) condition. There is no boundary layer on the ground.

## Envelope

- Angles of attack within the attached and mildly separated pre-stall regime.
- Mach number below 0.3.
- Chord Reynolds numbers from $10^4$ to $10^8$.
- Any ground height where the geometry stays above the ground; there is no height limit. A wake line that reaches the ground stops the case with a reason.

## Implementation

- Airfoil geometry: `ventorum.airfoil.geometry.AirfoilShape`, `ventorum.airfoil.airfoil`, `ventorum.airfoil.io`.
- Analytical NACA sections: `ventorum.airfoil.naca`.
- Spline and repaneling: `ventorum.airfoil.paneling.AirfoilSpline`, `repanel`, `repanel_cosine`.
- Flap deflection: `ventorum.airfoil.flap.deflect_plain_flap`, `flap_hinge_point`.
- Inviscid panel solver: `ventorum.panel2d.solver.solve_inviscid`.
- Stream function and velocity kernels: `ventorum.panel2d.influence`.
- Boundary layer closures, equations and march: `ventorum.ibl2d.closure`, `ventorum.ibl2d.transition`, `ventorum.ibl2d.equations`, `ventorum.ibl2d.march`, `ventorum.ibl2d.dual`.
- Coupled viscous solver: `ventorum.panel2d.viscous.solve_viscous` (inviscid data in `ventorum.panel2d.coupling`, Newton engine in `ventorum.panel2d.vii`).
- Solver revision for result caches: `ventorum.panel2d.SOLVER_REVISION`.
- Batched solver: `ventorum.panel2d.viscous_batch.solve_viscous_batch`.
- Polar sources and tables: `ventorum.aero.polar_source.SectionPolarSource`, `ventorum.aero.polar_source_v2d.Ventorum2DPolarSource`, `ventorum.aero.polar_cache.PolarCache`, `ventorum.aero.polar_table.PolarTable`.

## Verification

The methods in this chapter are verified by tests in the repository test suite:
- Geometry, file formats and NACA profiles: `tests/test_airfoil_io.py`, `tests/test_naca.py`, `tests/test_airfoil_api.py`.
- Flap deflections: `tests/test_airfoil_flap.py`.
- Panel resampling: `tests/test_airfoil_paneling.py`.
- Inviscid panel solver: `tests/test_panel2d.py`.
- Boundary layer closures, equations and march: `tests/test_ibl_closure.py`, `tests/test_ibl_equations.py`, `tests/test_ibl_march.py`.
- Coupled viscous solver: `tests/test_panel2d_viscous.py`, `tests/test_panel2d_viscous_wake.py`.
- Batched viscous solver: `tests/test_panel2d_viscous_batch.py`.
- Polar sources and tables: `tests/test_polar_table.py`, `tests/test_polar_source_v2d.py`.

## References

- I. H. Abbott and A. E. von Doenhoff, *Theory of Wing Sections*, Dover Publications, New York, 1959.
- C. de Boor, *A Practical Guide to Splines*, Applied Mathematical Sciences, Vol. 27, Springer-Verlag, New York, 1978.
- M. Drela, "Two-Dimensional Transonic Aerodynamic Design and Analysis Using the Euler Equations", Ph.D. thesis, Department of Aeronautics and Astronautics, Massachusetts Institute of Technology, Cambridge, MA, 1986.
- M. Drela, "XFOIL: An Analysis and Design System for Low Reynolds Number Airfoils", *Low Reynolds Number Aerodynamics*, Lecture Notes in Engineering, Vol. 54, Springer-Verlag, Berlin, 1989, pp. 1-12.
- M. Drela and M. B. Giles, "Viscous-Inviscid Analysis of Transonic and Low Reynolds Number Airfoils", *AIAA Journal*, Vol. 25, No. 10, 1987, pp. 1347-1355.
- H. Glauert, *The Elements of Aerofoil and Airscrew Theory*, Cambridge University Press, Cambridge, 1926.
- H. Glauert, "Theoretical Relationships for an Aerofoil with Hinged Flap", Aeronautical Research Committee Reports and Memoranda, R&M 1095, 1927.
- P. D. Green, D. G. Weeks and J. W. F. Brooman, "Prediction of Turbulent Boundary Layers and Wakes in Compressible Flow by a Lag-Entrainment Method", Aeronautical Research Council Reports and Memoranda, R&M 3791, 1977.
- E. N. Jacobs and R. M. Pinkerton, "Tests in the Variable-Density Wind Tunnel of Related Airfoils Having the Maximum Camber Unusually Far Forward", NACA Report 537, 1935.
- E. N. Jacobs, K. E. Ward and R. M. Pinkerton, "The Characteristics of 78 Related Airfoil Sections from Tests in the Variable-Density Wind Tunnel", NACA Report 460, 1933.
- J. Katz and A. Plotkin, *Low-Speed Aerodynamics*, 2nd ed., Cambridge University Press, Cambridge, 2001.
- C. L. Ladson, C. W. Brooks, A. S. Hill and D. W. Sproles, "Computer Program to Obtain Ordinates for NACA Airfoils", NASA TM-4741, 1996.
- D. Lednicer, "A CFD Evaluation of Wind Tunnel Test Sections", 48th AIAA Aerospace Sciences Meeting, AIAA Paper 2010-0458, Orlando, FL, 2010.
- M. J. Lighthill, "On Displacement Thickness", *Journal of Fluid Mechanics*, Vol. 4, No. 4, 1958, pp. 383-392.
- M. S. Selig, J. J. Guglielmo, A. P. Broeren and P. Giguere, *Summary of Low-Speed Airfoil Data*, Vol. 1, SoarTech Publications, Virginia Beach, VA, 1995.
- H. B. Squire and A. D. Young, "The Calculation of the Profile Drag of Aerofoils", Aeronautical Research Committee Reports and Memoranda, R&M 1838, 1938.
- T. J. Swafford, "Analytical Approximation of Two-Dimensional Separated Turbulent Boundary-Layer Velocity Profiles", *AIAA Journal*, Vol. 21, No. 6, 1983, pp. 923-925.
- C. Wieselsberger, "Wing Resistance Near the Ground", NACA TM-77, 1922; translated from *Zeitschrift fuer Flugtechnik und Motorluftschiffahrt*, Vol. 12, No. 10, 1921.
