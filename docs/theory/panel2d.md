---
orphan: true
---

# 2D inviscid panel solver with ground effect (Ventorum-2D)

## Model

The solver (`ventorum/panel2d/solver.py`, influences in `ventorum/panel2d/influence.py`) computes incompressible, inviscid, irrotational flow about one or more bodies in Selig order (counterclockwise), in free air or above flat ground. Sources: Katz and Plotkin (2001), chapters 10 and 11 (panel singularities, stream-function formulation) and section 6.6 (images); Drela (1989), the paper only.

The stream function is the sum of the free stream, linear-strength vortex panels with node values $\gamma_j$ and constant-strength source panels with prescribed strengths $\sigma_k$:

```{math}
\psi = \psi_\infty + \sum_j \gamma_j\,\psi^{(\gamma)}_j + \sum_k \sigma_k\,\psi^{(\sigma)}_k,
\qquad \psi_\infty = y\cos\alpha - x\sin\alpha \;\;\text{(free air)}, \quad \psi_\infty = y \;\;\text{(ground effect)}.
```

Velocities follow from $u = \partial\psi/\partial y$, $v = -\partial\psi/\partial x$. All values are over the free-stream speed.

## Panel influences

Local panel frame: $\xi$ along the panel from its start point, $\eta$ along the left normal (for a body in Selig order the left normal points into the body), panel length $L$.

**Linear vortex panel.** With $\gamma(t) = \gamma_1(1 - t/L) + \gamma_2\,t/L$ and $\psi = \frac{1}{4\pi}\int_0^L \gamma(t)\ln\big((\xi - t)^2 + \eta^2\big)\,dt$ (positive $\gamma$ turns clockwise). With the antiderivatives $F_0(u) = u\ln(u^2 + \eta^2) - 2u + 2\eta\arctan(u/\eta)$ and $F_1(u) = \tfrac12(u^2 + \eta^2)\big(\ln(u^2 + \eta^2) - 1\big)$, $I_0 = F_0(\xi) - F_0(\xi - L)$ and $\Delta F_1 = F_1(\xi) - F_1(\xi - L)$:

```{math}
\psi_1 = \frac{1}{4\pi}\left[\left(1 - \frac{\xi}{L}\right) I_0 + \frac{\Delta F_1}{L}\right], \qquad
\psi_2 = \frac{1}{4\pi}\left[\frac{\xi}{L} I_0 - \frac{\Delta F_1}{L}\right].
```

For field points farther than two panel lengths from the panel middle the end terms cancel strongly (in float32 the influence matrix had errors of 1e-5 relative). An equivalent form is used there: with $x_c = \xi - L/2$, $r_a^2 = \xi^2 + \eta^2$, $r_b^2 = (\xi - L)^2 + \eta^2$, $\Lambda = \ln(r_a^2/r_b^2) = \mathrm{log1p}(2 x_c L/r_b^2)$ and $I = \mathrm{atan2}(\eta L, \eta^2 + \xi(\xi - L))$,

```{math}
P_0 = L\ln(r_a r_b) + x_c\Lambda - 2L + 2\eta I, \qquad
P_1 = \tfrac12\Lambda\,(x_c^2 - L^2/4 - \eta^2) - x_c L + 2 x_c \eta I,
```

```{math}
\psi_1 = \frac{1}{4\pi}\left(\frac{P_0}{2} - \frac{P_1}{L}\right), \qquad \psi_2 = \frac{1}{4\pi}\left(\frac{P_0}{2} + \frac{P_1}{L}\right),
```

where $P_0 = \int_0^L \ln r^2\,dt$ and $P_1 = \int_0^L (t - L/2)\ln r^2\,dt$. Both forms are exact.

**Constant source panel.** $\psi = \frac{1}{2\pi}\int_0^L \varphi(t)\,dt$ with $\varphi$ the polar angle of the field point seen from the source point. With the branch $(-\pi, \pi]$, $\int_0^L \mathrm{atan2}(\eta, \xi - t)\,dt = G(\xi) - G(\xi - L)$, $G(u) = u\,\mathrm{atan2}(\eta, u) + \tfrac12\eta\ln(u^2 + \eta^2)$ (far form: $\tfrac{L}{2}(\varphi_a + \varphi_b) - x_c I + \tfrac12\eta\Lambda$). The source stream function is multivalued; each use selects a branch cut:

- body panels: the cut runs along the local $-\eta$ direction (the outward normal), so $\varphi \in (-\pi/2, 3\pi/2]$. A field point on the panel line ($\eta$ = 0 or round-off) is taken on the inner side ($+\eta$). This removes the round-off sign of $\eta$ at the panel's own nodes. The function is single-valued over the body interior and over the plane except the strip $\{\eta < 0,\ 0 < \xi < L\}$ outside the panel; for several bodies the solver raises `ValueError` when a node of one body lies in such a strip of another body;
- wake panels: the cut runs downstream along $+\xi$, so $\varphi \in [0, 2\pi)$; the function is continuous everywhere upstream, also over the airfoil;
- mirror panels below the ground: the field points (real nodes) are above the ground, so the global polar angle lies in $(0, \pi)$; the branch is fixed from the global angle at the panel start.

**Velocities** (used for the wake): the source panel induces $u = \frac{1}{4\pi}\ln(r_1^2/r_2^2)$, $v = \frac{1}{2\pi}\big(\mathrm{atan2}(\eta, \xi - L) - \mathrm{atan2}(\eta, \xi)\big)$; the linear vortex panel induces $u = \frac{1}{2\pi}\int\gamma\,\eta/r^2\,dt$, $v = -\frac{1}{2\pi}\int\gamma(\xi - t)/r^2\,dt$, which follow in closed form from $I_0 = \int \eta/r^2$, $J_0 = \tfrac12\ln(r_1^2/r_2^2)$, $I_1 = \xi I_0 - \eta J_0$ and $J_1 = \xi J_0 - L + \eta I_0$.

## Boundary conditions

At each node $\psi = \psi_b$ (the constant of body $b$, an extra unknown). Kutta condition per body: $\gamma_{first} + \gamma_{last} = 0$.

**Sharp trailing edge** (gap below $10^{-4}$ chord): the first and last nodes coincide and their two stream-function equations would be the same. Both are replaced by linear extrapolation conditions, $\gamma_1 - 2\gamma_2 + \gamma_3 = 0$ and $\gamma_N - 2\gamma_{N-1} + \gamma_{N-2} = 0$: the trailing-edge values are the linear extrapolations of the next two nodes on each side, and the Kutta condition makes the extrapolated speeds equal. This keeps the symmetry of the problem and is well conditioned also for a cusped edge (a single replaced row is singular there: two coincident sheets of opposite strength at the cusp are invisible to the node conditions). Drela (1989) replaces the equation of the last node by an extrapolation of the mean vorticity; the form used here is the author's and is to be checked against that paper.

**Blunt trailing edge**: a gap panel from the last node to the first node carries a constant vortex sheet $\gamma_{TE} = -\tfrac12(\gamma_{first} - \gamma_{last})(\hat s\cdot\hat t)$ and a constant source sheet $\sigma_{TE} = \tfrac12(\gamma_{first} - \gamma_{last})(\hat t\cdot\hat n_{out})$, where $\hat s$ is the unit vector along the gap panel, $\hat t$ the trailing-edge bisector and $\hat n_{out}$ the outer normal of the gap panel (Drela 1989; the signs follow from the jump conditions of this convention: the flow leaves the gap with the trailing-edge speed along the bisector). The gap panel is not a wall: it is not in the pressure integral.

## Solution

Free air: the matrix does not depend on $\alpha$; it is factorized once (LU) and two right-hand sides (free stream along $x$ and $y$) are superposed. Source strengths add right-hand-side columns only; zero sources reproduce the inviscid result bit for bit.

Ground effect (flat ground $y = 0$): each body has a mirror image with opposite vortex strength and equal source strength. The angle of attack rotates the geometry clockwise about the pivot (default the quarter-chord point on the chord line of the reference geometry of body 0, see below); the free stream is along $+x$; `height_reference="pivot"` places the pivot at the height, `"trailing_edge"` the lowest trailing-edge point. Geometry under or touching the ground raises `ValueError`. Each angle has its own matrix; the torch backend builds and solves them as a batch.

## Forces and moments

In free air, $C_l$ is the circulation lift from Kutta-Joukowski ($2\Gamma/c$), including the gap-panel vortex. In ground effect, near a wall the Kutta-Joukowski relation does not hold; $C_l$ is the surface pressure lift ($C_{l,pressure}$, Coulliette and Plotkin 1996, eq. 30). The field $C_{l,circulation}$ keeps the circulation lift ($2\Gamma/c$) in both cases. Pressure forces: $C_p = 1 - \gamma^2$, integrated exactly over each wall panel for $\gamma$ linear on the panel (panel mean $1 - (\gamma_1^2 + \gamma_1\gamma_2 + \gamma_2^2)/3$). $C_m$ about the moment reference point in the body-0 frame before rotation, nose up positive.

Reference geometry: the chord line from the leading edge to the trailing-edge midpoint, and the reference chord $c$. Body 0 gives it (`section_reference`). An `AirfoilShape` can hold one; `AirfoilShape.with_flap` stores the one of the section before the deflection, the reference of thin-airfoil theory (Glauert 1927). Otherwise it is the one of body 0 as given: the leading edge is the point farthest from the trailing-edge midpoint (the definition of `ventorum.airfoil`), and $c$ is the distance between them (`section_chord`). A plain flap moves the trailing edge, so a deflected section given as an array is shorter than the section before the deflection and its chord line is tilted; give the flapped shape, or the explicit values.

- Reference chord $c$ (`chord_ref`): the explicit value, else the reference chord of the reference geometry.
- Moment reference point (`moment_ref`, a point $(x, y)$ in the body-0 frame before rotation; `x_ref` is its former name, kept as an alias): the explicit value, else the quarter-chord point on the chord line, $x_{LE} + (x_{TE} - x_{LE})/4$ (`reference_point`). For a section with the leading edge at the origin and the trailing-edge midpoint at $(1, 0)$ this is $(0.25, 0)$, the former default; these sections keep their results bit for bit. A section whose chord line is not on $y = 0$ (a cambered section whose farthest point is not at the origin, a section moved in $y$) has its moment point on its chord line; before, the point was $(x_{min} + c/4,\ 0)$.
- Pivot (`pivot`): the explicit value, else the same quarter-chord point of the reference geometry. An explicit `moment_ref` does not move the pivot.

Moment transfer: the moment is the pressure moment of the wall panels. For two points $a$ and $b$ in the body frame, $C_{m,b} = C_{m,a} + (x_b - x_a) C_n / c - (y_b - y_a) C_a / c$, with the body-frame pressure force coefficients $C_n = C_{l,pressure} \cos\alpha + C_{d,pressure} \sin\alpha$ (normal, along $+y$) and $C_a = C_{d,pressure} \cos\alpha - C_{l,pressure} \sin\alpha$ (axial, along $+x$). A test checks it to 1e-12 in free air and in ground effect. A test compares a 2 deg flap on NACA 0006 with the flap effectiveness and the moment of thin-airfoil theory on the undeflected chord (Glauert 1927).

## Assumptions and envelope

Potential flow; the Kutta condition fixes the circulation. Ground effect is the inviscid image (moving-ground equivalent). Sources on several bodies need the strip condition above.

## Verification

`tests/test_panel2d.py`, `tests/test_karman_trefftz.py`:

- closed-form stream-function and velocity influences against composite Gauss quadrature (break points at $t = \xi$), 0.1 to 5 panel lengths away, 1e-10 relative; image source influences likewise;
- Karman-Trefftz airfoil (10 deg trailing edge): $C_l$ within 0.5 % of exact at 200 panels, $C_l$ and $C_{l,pressure}$ within 1 %, $|C_{d,pressure}| < 10^{-3}$; the largest $C_p$ error over all nodes except the trailing-edge node decreases from 100 to 200 to 400 panels with an observed order of at least 1 (at a sharp edge of angle $\tau$ the exact speed is zero but rises as $r^{\tau/(2\pi - \tau)}$, here $r^{0.029}$, so no node value converges to it);
- symmetric Joukowski section at $\alpha = 0$: $|C_l| < 10^{-10}$; superposition equals a direct solve to 1e-12;
- prescribed sources against an independent dense assembly by quadrature (written in the test), in free air and in ground effect, sharp and blunt sections, 1e-10; a uniform source sheet on a circle leaves the surface speed unchanged;
- ground effect against an explicit mirror body (1e-10), $h/c$ = 1000 against free air (1e-4), suction near the ground, geometry under the ground;
- two sections 20 chords apart; sources with a body in the outer strip of another body raise `ValueError`;
- torch float64 equals numpy to 1e-10; float32 within 1e-4 on $C_l$ (CPU, and CUDA marked gpu).

## References

- J. Katz and A. Plotkin, Low-Speed Aerodynamics, 2nd ed., Cambridge University Press, 2001.
- M. Drela, "XFOIL: An analysis and design system for low Reynolds number airfoils", in Low Reynolds Number Aerodynamics, Lecture Notes in Engineering 54, Springer, 1989, pp. 1-12.
- H. Glauert, "Theoretical relationships for an aerofoil with hinged flap", ARC R and M 1095, 1927.
- L. M. Milne-Thomson, Theoretical Aerodynamics, 4th ed., Macmillan, 1958 (Karman-Trefftz reference).
- C. Coulliette and A. Plotkin, "Aerofoil ground effect revisited", The Aeronautical Journal 100, 1996.
