---
orphan: true
---

# Ventorum-2D: Airfoil geometry, panel method and viscous-inviscid interaction

## Model

Ventorum-2D provides two-dimensional airfoil analysis and section polar generation for the Ventorum aerodynamic solver suite. The system consists of six integrated parts:
1. Airfoil geometry tools that read standard coordinate formats, generate analytical NACA 4-digit and 5-digit profiles, resample nodes with curvature-adapted splines, and deflect plain trailing-edge flaps.
2. A potential-flow panel solver that uses a stream-function Dirichlet formulation with linear-strength vortex panels and constant-strength source panels. The solver models free-air flow by two-right-hand-side superposition and ground effect by the method of images.
3. A two-equation integral boundary layer solver that couples the momentum and kinetic-energy integral equations with shear-stress lag and envelope $e^N$ transition models.
4. A viscous-inviscid interaction algorithm that models boundary layer displacement through Lighthill wall transpiration sources and streamline wake tracing.
5. A batched multi-condition execution engine that vectorizes coupled solves across angles of attack, Reynolds numbers, and ground heights on CPU and CUDA GPU hardware.
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

The leading edge is identified as the stored polygon point farthest from the trailing-edge midpoint. Surface separation splits the polygon into upper and lower surfaces, each directed from leading edge to trailing edge. Camber and thickness distributions follow from the surface ordinates:

```{math}
y_c(x) = \frac{1}{2}\left(y_{\text{upper}}(x) + y_{\text{lower}}(x)\right), \qquad
t(x) = y_{\text{upper}}(x) - y_{\text{lower}}(x).
```

### File formats

The geometry reader supports three plain-text formats:
- Selig format: an optional header line with the airfoil name, followed by coordinate pairs $x, y$ in Selig order.
- Lednicer format: an airfoil name line, a line specifying the point counts of the upper and lower surfaces, the upper surface points from leading edge to trailing edge, a blank line, and the lower surface points from leading edge to trailing edge. The reader joins the surfaces and merges the duplicated leading edge.
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

Parameters for design lift coefficient $C_{L,\text{des}} = 0.3$ (first digit 2) are:

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

3. Weight function: A node-density weight function $w(s)$ combines local curvature magnitude with Gaussian refinement functions centered at the leading edge and at both trailing-edge corners:

```{math}
w(s) = 1.0 + c_\kappa |\kappa(s)| + c_{\text{le}} \exp\left(-\frac{s^2}{2 \sigma_{\text{le}}^2}\right) + c_{\text{te}} \exp\left(-\frac{(s - s_{\text{te}})^2}{2 \sigma_{\text{te}}^2}\right).
```

4. Equidistribution: New node stations are chosen such that the cumulative weight integral is partitioned into equal increments:

```{math}
W(s_k) = \int_0^{s_k} w(\tau) \, d\tau = \frac{k}{N-1} \int_0^{s_{\text{total}}} w(\tau) \, d\tau.
```

5. Ratio smoothing: Panel lengths are smoothed iteratively to ensure that the ratio of adjacent panel lengths does not exceed 1.5. The leading edge point is preserved as an exact node.

### Cosine repaneling

The function `repanel_cosine` resamples upper and lower surfaces independently using Chebyshev cosine spacing in surface arc length. This clusters nodes near both the leading edge and the trailing edge.

## Plain-flap geometric deflection

The function `deflect_plain_flap` deflects the trailing-edge portion of an airfoil section about a specified hinge point. The deflection angle $\delta$ [deg] is positive when the trailing edge moves downward (clockwise rotation in the section plane).

The geometric construction follows four steps:
1. Hinge location: The hinge chord station $x_{\text{hinge}} \in (0.05, 0.98)$ and thickness fraction $y_{\text{hinge}} \in [0, 1]$ define the hinge point:

```{math}
\mathbf{x}_h = \left(x_{\text{hinge}}, \, y_{\text{lower}}(x_{\text{hinge}}) + y_{\text{hinge}} \left[y_{\text{upper}}(x_{\text{hinge}}) - y_{\text{lower}}(x_{\text{hinge}})\right]\right).
```

2. Rigid rotation: All points with $x > x_{\text{hinge}}$ rotate rigidly about $\mathbf{x}_h$ by deflection angle $\delta$.
3. Compressed surface cut: On the side compressed by rotation (the lower surface for positive deflection), the fixed forward surface and the rotated aft surface overlap. Both curves are trimmed at their mutual intersection, removing the overlapping points.
4. Stretched surface arc: On the side stretched by rotation (the upper surface for positive deflection), a gap opens. A circular arc centered at $\mathbf{x}_h$ connects the fixed forward curve to the rotated aft curve.

The geometric deflection alters the enclosed polygon area by less than 2 % for deflections up to 10 deg.

## Inviscid panel method

### Stream function Dirichlet formulation

The two-dimensional inviscid panel solver represents incompressible, irrotational potential flow using stream functions (Katz and Plotkin, 2001; Drela, 1989). The velocity field $(u, v)$ relates to the stream function $\psi$ by:

```{math}
u = \frac{\partial \psi}{\partial y}, \qquad v = -\frac{\partial \psi}{\partial x}.
```

The total stream function is the superposition of free-stream, vortex-sheet, and source contributions:

```{math}
\psi(x, y) = \psi_\infty(x, y) + \psi_\gamma(x, y) + \psi_\sigma(x, y).
```

In free air at angle of attack $\alpha$, the free-stream stream function is:

```{math}
\psi_\infty(x, y) = V_\infty (y \cos\alpha - x \sin\alpha).
```

The surface of each closed lifting body forms a streamline:

```{math}
\psi(x_i, y_i) = \Psi_b,
```

where $(x_i, y_i)$ are collocation points at the surface nodes of body $b$, and $\Psi_b$ is an unknown constant stream function value for that body.

### Linear-strength vortex panels

The airfoil contour is discretized into $M$ flat panels connected by $N$ nodes. The vortex sheet strength varies linearly over panel $j$ from $\gamma_1$ at local coordinate $s = 0$ to $\gamma_2$ at $s = L$:

```{math}
\gamma(s) = \gamma_1 \left(1 - \frac{s}{L}\right) + \gamma_2 \left(\frac{s}{L}\right).
```

A point vortex of strength $d\Gamma = \gamma(s) ds$ induces a stream function at local coordinates $(\xi, \eta)$:

```{math}
d\psi = \frac{1}{4\pi} \ln\left((\xi - s)^2 + \eta^2\right) \gamma(s) \, ds.
```

With substitution $t = \xi - s$, the required indefinite integrals are:

```{math}
F_0(t, \eta) = \int \ln(t^2 + \eta^2) \, dt = t \ln(t^2 + \eta^2) - 2t + 2\eta \arctan\left(\frac{t}{\eta}\right),
```

```{math}
F_1(t, \eta) = \int t \ln(t^2 + \eta^2) \, dt = \frac{1}{2}(t^2 + \eta^2) \left(\ln(t^2 + \eta^2) - 1\right).
```

Integrating over panel length $L$ yields the unit nodal stream function influence coefficients:

```{math}
\psi = \psi_1 \gamma_1 + \psi_2 \gamma_2,
```

where:

```{math}
\psi_1 = \frac{1}{4\pi} \left[ \left(1 - \frac{\xi}{L}\right) I_0 + \frac{1}{L} \left(F_1(\xi, \eta) - F_1(\xi - L, \eta)\right) \right],
```

```{math}
\psi_2 = \frac{1}{4\pi} \left[ \frac{\xi}{L} I_0 - \frac{1}{L} \left(F_1(\xi, \eta) - F_1(\xi - L, \eta)\right) \right],
```

with $I_0 = F_0(\xi, \eta) - F_0(\xi - L, \eta)$.

### Constant-strength source panels

A panel carrying uniform source strength $\sigma$ induces the stream function:

```{math}
\psi_\sigma = \frac{\sigma}{2\pi} \left[ \xi \arctan\left(\frac{\eta}{\xi}\right) + \frac{1}{2}\eta \ln(\xi^2 + \eta^2) - (\xi - L) \arctan\left(\frac{\eta}{\xi - L}\right) - \frac{1}{2}\eta \ln\left((\xi - L)^2 + \eta^2\right) + \theta_{\text{panel}} L \right],
```

where $\theta_{\text{panel}}$ is the panel inclination angle in global coordinates.

### System assembly and Kutta condition

For $N_b$ nodes on body $b$, the stream function Dirichlet condition enforces:

```{math}
\sum_{j=1}^{M} \left(\psi_{1, ij} \gamma_{1, j} + \psi_{2, ij} \gamma_{2, j}\right) - \Psi_b = -\psi_\infty(x_i, y_i) - \sum_{j=1}^{M} \psi_{\sigma, ij} \sigma_j.
```

To close the system, each body requires one additional equation: the Kutta condition. The code enforces:

```{math}
\gamma_{\text{first}} + \gamma_{\text{last}} = 0.
```

For a sharp trailing edge (gap $\le 10^{-4} c$), this condition sets the sum of the upper and lower trailing-edge nodal vortex strengths to zero. For a blunt trailing edge (gap $> 10^{-4} c$), an additional flat base panel connects the lower trailing edge node ($N-1$) to the upper trailing edge node ($0$). The base panel carries linear vorticity from $\gamma_{N-1}$ to $\gamma_0$, closing the contour in the stream-function influence assembly, while the closure equation remains $\gamma_0 + \gamma_{N-1} = 0$.

### Two-right-hand-side superposition in free air

In free air, the geometric influence matrix $A$ does not depend on angle of attack $\alpha$. Matrix $A$ is factored once by LU decomposition. The solver evaluates two fundamental unit onset solutions:
1. Horizontal onset ($V_x = 1$): $\psi_{\infty, x} = y$.
2. Vertical onset ($V_y = 1$): $\psi_{\infty, y} = -x$.

For any angle of attack $\alpha$, the solution vector follows by linear superposition:

```{math}
\mathbf{x}(\alpha) = \cos\alpha \, \mathbf{x}_x + \sin\alpha \, \mathbf{x}_y + \mathbf{x}_\sigma,
```

where $\mathbf{x}_\sigma$ accounts for prescribed source distributions.

### Method of images for ground effect

Ground effect over a flat boundary at $y = 0$ is modeled by reflecting all panels across the wall:
- Vortex panel images: A real vortex panel at $(x, y)$ with vorticity $\gamma$ induces normal velocity on the ground. To enforce zero normal flow through $y = 0$, its image panel at $(x, -y)$ carries opposite vorticity $\gamma_{\text{image}} = -\gamma_{\text{real}}$. Its stream function influence subtracts:

```{math}
A_{\text{vortex}}^{\text{GE}} = A_{\text{vortex}}^{\text{real}} - A_{\text{vortex}}^{\text{image}}.
```

- Source panel images: A real source panel with strength $\sigma$ requires an image panel with equal strength $\sigma_{\text{image}} = +\sigma_{\text{real}}$. Its stream function influence adds:

```{math}
A_{\text{source}}^{\text{GE}} = A_{\text{source}}^{\text{real}} + A_{\text{source}}^{\text{image}}.
```

In ground effect, the body is rotated by $\alpha$ about the pivot point and translated to height $h$. Because the geometry changes with attitude and height, influence matrices are assembled and factored per height and attitude group.

### Forces and moments

1. Circulation lift: Integrating surface vorticity yields total circulation $\Gamma$:

```{math}
\Gamma = \sum_{j=1}^{M} \frac{1}{2}(\gamma_{1, j} + \gamma_{2, j}) L_j, \qquad C_l = \frac{2\Gamma}{V_\infty c_{\text{ref}}}.
```

2. Pressure distribution: Tangential surface velocity equals local vortex sheet strength ($u_e = \gamma$). Bernoulli's equation gives the pressure coefficient at each node:

```{math}
C_p = 1 - \left(\frac{u_e}{V_\infty}\right)^2 = 1 - \gamma^2.
```

3. Near-field force integration: Integrating $C_p$ along panel outward normal vectors gives normal force $C_n$, axial force $C_a$, and pitching moment $C_m$.

## Integral boundary layer

The viscous engine of Ventorum-2D is a two-equation integral boundary layer model based on Drela and Giles (1987).

### Governing equations

With arc length $s$, edge velocity ratio $u_e$, momentum thickness $\theta$, kinematic shape factor $H = \delta^* / \theta$, kinetic-energy shape factor $H^* = \theta^* / \theta$, skin-friction coefficient $C_f$, dissipation coefficient $C_D$, maximum shear-stress coefficient $C_\tau$, and amplification exponent $N$:

```{math}
\text{(M1)} \qquad \frac{d\theta}{ds} = \frac{C_f}{2} - (2 + H) \frac{\theta}{u_e} \frac{du_e}{ds},
```

```{math}
\text{(M2)} \qquad \frac{d\theta^*}{ds} = 2 C_D - 3 \frac{\theta^*}{u_e} \frac{du_e}{ds}, \qquad \theta^* = H^* \theta,
```

```{math}
\text{(M3)} \qquad \delta^* \frac{dC_\tau}{ds} = K (\sqrt{C_{\tau,\text{eq}}} - \sqrt{C_\tau}), \qquad K = K_c \sqrt{C_\tau},
```

```{math}
\text{(M4)} \qquad \frac{dN}{ds} = g(H) \frac{dRe_\theta}{ds} \quad (Re_\theta \ge Re_{\theta,0}), \qquad \frac{dN}{ds} = 0 \quad (Re_\theta < Re_{\theta,0}).
```

Equation (M1) is the von Karman momentum integral. Equation (M2) is the kinetic-energy integral. Equation (M3) is the Green, Weeks and Brooman (1977) lag-entrainment model. Equation (M4) is the envelope $e^N$ transition equation. The local momentum Reynolds number is $Re_\theta = u_e \theta Re_{\text{ref}}$.

### Laminar closures (Falkner-Skan family)

Laminar closures follow Drela (1986) fits to the Falkner-Skan boundary layer family:

```{math}
H^*(H) = \begin{cases}
a_1 + b_1 \frac{(4 - H)^2}{H}, & H \le 4, \\
a_1 - b_2 \frac{(H - 4)^2}{H}, & H > 4,
\end{cases}
```

```{math}
F(H) = Re_\theta \frac{C_f}{2} = c_0 + c_1 \frac{(7.4 - H)^2}{H - 1},
```

```{math}
G(H) = Re_\theta \frac{2 C_D}{H^*} = \begin{cases}
g_0 + g_1 (4 - H)^{5.5}, & H \le 4, \\
g_0 - g_2 \frac{(H - 4)^2}{1 + g_3 (H - 4)^2}, & H > 4.
\end{cases}
```

Formulas $H^*(H)$ and $G(H)$ maintain $C^1$ continuity at $H = 4.0$. $F(H)$ is smooth and passes through zero near $H = 4.0$ (laminar separation).

### Turbulent closures (Swafford profile family)

Turbulent closures are based on Swafford (1983) wall-wake velocity profiles:

```{math}
C_f(H, Re_\theta) = A_0 e^{-A_1 H} (\log_{10} Re_\theta)^{-(E_0 + E_1 H)} + S_0 \left(\tanh(4 - H) - \frac{1}{2}\right),
```

```{math}
H^*(H, Re_\theta) = s_0 + s_1 (H - 1) + s_2 (H - 1)^2 - s_3 \log_{10}(Re_\theta / 10^4),
```

```{math}
U_s(H, Re_\theta) = u_0 + u_1 (H - 1) + u_2 (H - 1)^2 - u_3 \log_{10}(Re_\theta / 10^4),
```

```{math}
C_{\tau,\text{eq}}(H, H^*, U_s) = k \left(\frac{H^* - 1}{H^* U_s}\right)^2,
```

```{math}
C_D(H, Re_\theta, U_s, C_\tau) = \frac{C_f}{2} U_s + k_d (C_\tau - C_{\tau,\text{eq}}).
```

### Transition model

Transition is predicted by the envelope $e^N$ method:

```{math}
Re_{\theta,0}(H) = r_0 + \exp(r_1 - H),
```

```{math}
g(H) = \frac{dN}{dRe_\theta} = n_0 + n_1 (H - 1).
```

Transition occurs when $N$ reaches $N_{\text{crit}}$ (default 9.0), or at a user-specified forced transition coordinate $x_{\text{tr}}$, or when laminar shape factor reaches $H \ge 3.4$. Upon transition, momentum thickness $\theta$ remains continuous and shape factor resets to an attached turbulent value ($H = 1.45$ in the coupled solver; $H = 1.34$ in the standalone march).

### Direct marching discretization

Stations are marched downstream using the implicit second-order trapezoidal rule. At each station, a local Newton-Raphson iteration solves for the state updates. At the stagnation point ($u_e = 0$), self-similar Hiemenz flow initializes the boundary layer:

```{math}
H_0 = 2.216, \qquad \theta_0 = \sqrt{\frac{F_0}{(2 + H_0) a_{\text{stag}} Re}},
```

where $a_{\text{stag}} = |du_e/ds|_{\text{stag}}$ and $F_0 = (Re_\theta C_f / 2)_{H_0}$. In standalone marches without a stagnation point, Blasius flat-plate conditions ($H_0 = 2.591$) initialize the march.

## Viscous-inviscid coupling

### Wall transpiration model

Real fluid boundary layers displace outer potential streamlines. Ventorum-2D models this displacement through wall transpiration (Lighthill, 1958). The mass flow defect within the boundary layer is:

```{math}
m(s) = u_e(s) \delta^*(s).
```

By mass conservation across the boundary layer, the equivalent normal blowing velocity at the original solid wall is:

```{math}
v_n(s) = \frac{d}{ds} \left[u_e(s) \delta^*(s)\right].
```

In the discrete panel solver, each panel carries a constant-strength source of strength:

```{math}
\sigma_j = \frac{m_{j+1} - m_j}{\Delta s_j}.
```

### Mass-defect influence matrix

The potential flow boundary condition with transpiration sources is:

```{math}
A \boldsymbol{\gamma} = \mathbf{b}_{\text{inv}} - \Psi_s \boldsymbol{\sigma},
```

where $A$ is the vortex influence matrix and $\Psi_s$ is the source stream function matrix. Decomposing by linear superposition gives:

```{math}
\boldsymbol{\gamma} = \boldsymbol{\gamma}_{\text{inv}} + D \boldsymbol{\sigma}, \qquad D = -A^{-1} \Psi_s.
```

Matrix $D$ is the mass-defect influence matrix. Because $A$ depends only on geometry, its LU factors are computed once. Boundary layer iterations update the surface edge velocity by back-substitution:

```{math}
u_e = u_{e,\text{inv}} + D \boldsymbol{\sigma}.
```

### Streamline wake tracing

The viscous wake trails from the trailing edge along the local potential streamline (Drela, 1989). Trajectory coordinates are integrated using a second-order Runge-Kutta (Heun predictor-corrector) scheme. Wake panels expand geometrically with ratio 1.10.

The wake carries source panels to model wake displacement:

```{math}
m_w(s) = |u_{e, w}(s)| \delta^*_w(s), \qquad \sigma_{w, j} = \frac{m_w(s_{j+1}) - m_w(s_j)}{\Delta s_{w, j}}.
```

The wake edge velocity relaxes toward the free stream:

```{math}
u_{e, w}(s) = 1 + (u_{e, \text{TE}} - 1) \exp(-3 s / c).
```

Wake source influences are evaluated without branch cuts across the airfoil surface.

### Blunt trailing-edge base treatment

For airfoils with a blunt trailing edge of thickness $h_{\text{te}}$, the base creates a separated dead-air region (Hoerner, 1965). The wake initial displacement thickness includes the base gap:

```{math}
\delta^*_{w, 0} = \delta^*_{\text{upper}} + \delta^*_{\text{lower}} + h_{\text{te}}.
```

The initial wake momentum thickness includes a base drag increment:

```{math}
\theta_{w, 0} = \theta_{\text{upper}} + \theta_{\text{lower}} + \frac{1}{2} C_{D, \text{base}} h_{\text{te}}, \qquad
C_{D, \text{base}} = \max(u_{e, \text{TE}}^2 - 1.0, \, 0.15).
```

### Profile drag calculation

Profile drag is evaluated by the Squire and Young (1938) formula at the trailing edge:

```{math}
C_d = 2 \theta_{\text{TE}} \left(\frac{u_{e, \text{TE}}}{V_\infty}\right)^{\frac{H_{\text{TE}} + 5}{2}},
```

where $\theta_{\text{TE}} = \theta_{\text{upper}} + \theta_{\text{lower}} + \theta_{\text{base}}$ and $H_{\text{TE}} = (\delta^*_{\text{upper}} + \delta^*_{\text{lower}} + h_{\text{te}}) / \theta_{\text{TE}}$.

A near-field momentum check integrates surface pressure and skin friction:

```{math}
C_{d, \text{near}} = \frac{1}{c} \oint C_p n_x \, ds + C_{D, \text{base}} \frac{h_{\text{te}}}{c} + \frac{1}{c} \oint C_f t_x \, ds.
```

### Flapped sections and deceleration limiter

Flap deflections generate strong adverse pressure gradients near the hinge line. To prevent numerical breakdown from the boundary-layer singularity near separation, the code applies a Veldman (2009) deceleration limiter:

```{math}
\Delta u_e \ge -0.3 \, \bar{u}_e \frac{\Delta s}{\theta}.
```

Separation is detected when laminar shape factor reaches $H \ge 3.8$, or turbulent shape factor reaches $H \ge 2.6$, or skin friction drops to $C_f \le 0$. When separation occurs, the solver flags the state and reports diagnostic strings without raising unhandled exceptions.

### Ground effect coupling

In ground effect, image sources reflect across $y = 0$:

```{math}
\delta^*_{\text{image}}(x, -y) = \delta^*_{\text{real}}(x, y).
```

The mass-defect matrix incorporates direct and image sources:

```{math}
D_{\text{GE}} = -A_{\text{GE}}^{-1} (\Psi_{s, \text{real}} + \Psi_{s, \text{image}}).
```

## Batched execution and angle continuation

The module `ventorum.panel2d.viscous_batch` executes sweeps over angles of attack, Reynolds numbers, and ground heights:
- Cases sharing identical panel counts are assembled into batched PyTorch tensors on CPU or GPU.
- Converged cases drop out of active iterations via boolean masking.
- Angle continuation organizes each sweep into positive passes (ascending from $\alpha = 0$) and negative passes (descending from $\alpha = 0$). Pass 0 starts from inviscid flow, and subsequent passes initialize from the previous converged solution.

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
- Interpolation is linear in $\log_{10} Re$, linear in flap deflection $\delta_{\text{flap}}$, linear in inverse height $1/h$ (reflecting image singularity decay), and linear in angle of attack $\alpha$.
- Unconverged points are recorded as NaN. Queries outside the table bounds raise errors rather than extrapolating unverified data.

## Assumptions

- Steady, incompressible, irrotational outer potential flow.
- Boundary layer thickness small relative to chord and surface radius of curvature.
- Attached flow or mild trailing-edge separation; deep post-stall separation is outside the physical model envelope.
- Rigid, non-deforming ground plane during ground-effect interactions.

## Envelope

- Angles of attack within the attached and mildly separated pre-stall regime.
- Mach number below 0.3.
- Chord Reynolds numbers from $10^4$ to $10^8$.
- Ground clearances $h/c > 0.05$ where the physical geometry does not penetrate the wall.

## Implementation

- Airfoil geometry: `ventorum.airfoil.geometry.AirfoilShape`, `ventorum.airfoil.airfoil`, `ventorum.airfoil.io`.
- Analytical NACA sections: `ventorum.airfoil.naca`.
- Spline and repaneling: `ventorum.airfoil.paneling.AirfoilSpline`, `repanel`, `repanel_cosine`.
- Flap deflection: `ventorum.airfoil.flap.deflect_plain_flap`, `flap_hinge_point`.
- Inviscid panel solver: `ventorum.panel2d.solver.solve_panel2d`, `compute_mass_defect_matrix`.
- Stream function kernels: `ventorum.panel2d.influence`.
- Boundary layer closures and march: `ventorum.ibl2d.closure`, `ventorum.ibl2d.transition`, `ventorum.ibl2d.march`.
- Coupled viscous solver: `ventorum.panel2d.viscous.solve_viscous`, `solve_polar`.
- Batched solver: `ventorum.panel2d.viscous_batch.solve_viscous_batch`.
- Polar sources and tables: `ventorum.aero.polar_source.SectionPolarSource`, `ventorum.aero.polar_source_v2d.Ventorum2DPolarSource`, `ventorum.aero.polar_cache.PolarCache`, `ventorum.aero.polar_table.PolarTable`.

## Verification

The methods in this chapter are verified by tests in the repository test suite:
- Geometry and NACA profiles: `tests/test_airfoil_geometry.py`, `tests/test_naca.py`, `tests/test_airfoil_io.py`.
- Flap deflections: `tests/test_airfoil_flap.py`.
- Panel resampling: `tests/test_airfoil_paneling.py`.
- Inviscid panel solver: `tests/test_panel2d.py`.
- Boundary layer closures and march: `tests/test_ibl_closure.py`, `tests/test_ibl_march.py`.
- Coupled viscous solver: `tests/test_panel2d_viscous.py`.
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
- S. F. Hoerner, *Fluid-Dynamic Drag: Practical Information on Aerodynamic Drag and Hydrodynamic Resistance*, Published by the Author, Midland Park, NJ, 1965.
- E. N. Jacobs and R. M. Pinkerton, "Tests in the Variable-Density Wind Tunnel of Related Airfoils Having the Maximum Camber Unusually Far Forward", NACA Report 537, 1935.
- E. N. Jacobs, K. E. Ward and R. M. Pinkerton, "The Characteristics of 78 Related Airfoil Sections from Tests in the Variable-Density Wind Tunnel", NACA Report 460, 1933.
- J. Katz and A. Plotkin, *Low-Speed Aerodynamics*, 2nd ed., Cambridge University Press, Cambridge, 2001.
- C. L. Ladson, C. W. Brooks, A. S. Hill and D. W. Sproles, "Computer Program to Obtain Ordinates for NACA Airfoils", NASA TM-4741, 1996.
- D. Lednicer, "A CFD Evaluation of Wind Tunnel Test Sections", 48th AIAA Aerospace Sciences Meeting, AIAA Paper 2010-0458, Orlando, FL, 2010.
- M. J. Lighthill, "On Displacement Thickness", *Journal of Fluid Mechanics*, Vol. 4, No. 4, 1958, pp. 383-392.
- M. S. Selig, J. J. Guglielmo, A. P. Broeren and P. Giguere, *Summary of Low-Speed Airfoil Data*, Vol. 1, SoarTech Publications, Virginia Beach, VA, 1995.
- H. B. Squire and A. D. Young, "The Calculation of the Profile Drag of Aerofoils", Aeronautical Research Committee Reports and Memoranda, R&M 1838, 1938.
- T. J. Swafford, "Analytical Approximation of Two-Dimensional Separated Turbulent Boundary-Layer Velocity Profiles", *AIAA Journal*, Vol. 21, No. 6, 1983, pp. 923-925.
- A. E. P. Veldman, "Quasi-Simultaneous Interaction Methods for Viscous-Inviscid Flow", *Computers & Fluids*, Vol. 38, No. 3, 2009, pp. 496-508.
- C. Wieselsberger, "Wing Resistance Near the Ground", NACA TM-77, 1922; translated from *Zeitschrift fuer Flugtechnik und Motorluftschiffahrt*, Vol. 12, No. 10, 1921.
