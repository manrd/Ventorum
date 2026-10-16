---
orphan: true
---

# Integral boundary-layer closures and transition (Ventorum-2D)

## Model

The integral boundary layer of Ventorum-2D solves the momentum equation, the kinetic-energy (shape-parameter) equation and, in turbulent flow, the shear-stress lag equation (see [](ibl2d_march.md)). These equations need closure relations: the energy shape factor $H^*$, the skin friction $C_f$, the dissipation coefficient $C_D$, the equilibrium shear-stress coefficient $C_{\tau,EQ}$ and the boundary-layer thickness $\delta$, as functions of the shape factor $H = \delta^*/\theta$ and the momentum-thickness Reynolds number $Re_\theta = u_e \theta / \nu$. Transition comes from the envelope $e^N$ method.

All relations are written from the published papers: Drela and Giles (1987) and Drela (1986). No program source code was read or used. The flow is incompressible (edge Mach number 0), so the kinematic shape factor $H_k$ of the papers equals $H$ and the density shape factor $H^{**}$ is zero.

The code is in `ventorum/ibl2d/closure.py` (labels C1 to C10) and `ventorum/ibl2d/transition.py` (labels E1 to E4). Each function returns the value and its analytic partial derivatives, and takes an `xp` argument (numpy or torch).

**To be checked against the paper.** The equation numbers of Drela and Giles (1987) are not cited here, because they could not be checked against the printed paper; each relation is named instead. The constants below are the best reading of the paper by the author of the code. They agree with independent checks where such checks exist (Blasius, exact Falkner-Skan profiles, Ludwieg-Tillmann; see Verification).

## Laminar closures (Falkner-Skan family)

(C1) Energy shape factor:

```{math}
H^* = 1.515 + 0.076\,\frac{(4 - H)^2}{H} \quad (H < 4), \qquad
H^* = 1.515 + 0.040\,\frac{(H - 4)^2}{H} \quad (H \ge 4).
```

(C2) Skin friction:

```{math}
Re_\theta \frac{C_f}{2} = -0.067 + 0.01977\,\frac{(7.4 - H)^2}{H - 1} \quad (H < 7.4), \qquad
Re_\theta \frac{C_f}{2} = -0.067 + 0.022\left(1 - \frac{1.4}{H - 6}\right)^2 \quad (H \ge 7.4).
```

(C3) Dissipation:

```{math}
Re_\theta \frac{2 C_D}{H^*} = 0.207 + 0.00205\,(4 - H)^{5.5} \quad (H < 4), \qquad
Re_\theta \frac{2 C_D}{H^*} = 0.207 - \frac{0.003\,(H - 4)^2}{1 + 0.02\,(H - 4)^2} \quad (H \ge 4).
```

All branch points are continuous in value and first derivative. $H^*$ has its minimum at $H = 4$, and $C_f = 0$ near $H = 4.03$ (Falkner-Skan separation).

## Turbulent closures

(C4) Energy shape factor, with $H_0 = 3 + 400/Re_\theta$ for $Re_\theta > 400$ and $H_0 = 4$ otherwise:

```{math}
H^* = 1.505 + \frac{4}{Re_\theta} + \left(0.165 - \frac{1.6}{\sqrt{Re_\theta}}\right)\frac{(H_0 - H)^{1.6}}{H} \quad (H < H_0),
```

```{math}
H^* = 1.505 + \frac{4}{Re_\theta} + (H - H_0)^2\left[\frac{0.04}{H} + \frac{0.007 \ln Re_\theta}{(H - H_0 + 4/\ln Re_\theta)^2}\right] \quad (H \ge H_0).
```

$H^*$ has its minimum at $H = H_0$ (the separation shape factor of the direct march).

(C5) Skin friction, the fit of Drela and Giles (1987) to the profile family of Swafford (1983):

```{math}
C_f = 0.3\,e^{-1.33 H}\,(\log_{10} Re_\theta)^{-1.74 - 0.31 H} + 0.00011\left[\tanh\left(4 - \frac{H}{0.875}\right) - 1\right].
```

(C6) Normalized slip velocity:

```{math}
U_s = \frac{H^*}{2}\left(1 - \frac{4}{3}\,\frac{H - 1}{H}\right).
```

(C7) Equilibrium shear-stress coefficient (from the equilibrium locus $G = A\sqrt{1 + B\beta}$):

```{math}
C_{\tau,EQ} = H^*\,\frac{0.015}{1 - U_s}\,\frac{(H - 1)^3}{H^3}.
```

(C8) Dissipation coefficient, with the shear-stress coefficient $C_\tau$ of the lag equation:

```{math}
C_D = \frac{C_f}{2}\,U_s + C_\tau\,(1 - U_s).
```

In the wake the skin friction is zero and the dissipation is doubled (two shear layers): $C_D = 2\,C_\tau (1 - U_s)$.

(C9) Shear-stress lag equation (the lag-entrainment idea of Green, Weeks and Brooman 1977, in the form of Drela and Giles 1987), with the constants $K_C = 5.6$ and $A = 6.7$:

```{math}
\frac{\delta}{C_\tau}\frac{dC_\tau}{d\xi} = 5.6\left(C_{\tau,EQ}^{1/2} - C_\tau^{1/2}\right)
+ 2\delta\left\{\frac{4}{3\delta^*}\left[\frac{C_f}{2} - \left(\frac{H - 1}{6.7\,H}\right)^2\right] - \frac{1}{u_e}\frac{du_e}{d\xi}\right\}.
```

(C10) Boundary-layer thickness in the lag equation:

```{math}
\delta = \theta\left(3.15 + \frac{1.72}{H - 1}\right) + \delta^*.
```

## Transition (envelope $e^N$ method)

(E1) Critical momentum-thickness Reynolds number:

```{math}
\log_{10} Re_{\theta 0} = \left(\frac{1.415}{H - 1} - 0.489\right)\tanh\left(\frac{20}{H - 1} - 12.9\right) + \frac{3.295}{H - 1} + 0.44.
```

(E2) Envelope amplification rate:

```{math}
\frac{dN}{dRe_\theta} = 0.01\left\{\left[2.4 H - 3.7 + 2.5\tanh(1.5 H - 4.65)\right]^2 + 0.25\right\}^{1/2}.
```

(E3) Conversion to the surface coordinate with the Falkner-Skan relations $\ell(H) = (6.54 H - 14.07)/H^2$ and $m(H) = \left[0.058 (H - 4)^2/(H - 1) - 0.068\right]/\ell(H)$:

```{math}
\theta\,\frac{dRe_\theta}{d\xi} = \frac{m + 1}{2}\,\ell = \frac{1}{2}\left[0.058\,\frac{(H - 4)^2}{H - 1} - 0.068 + \ell\right].
```

The second form has no singular point where $\ell = 0$.

(E4) Growth rate: $dN/d\xi = (dN/dRe_\theta)\,(m + 1)\ell/(2\theta)$ for $Re_\theta > Re_{\theta 0}$, and zero below. The switch is a jump of the published relation; the code keeps it. The discrete equations integrate it exactly inside an interval (see [](ibl2d_march.md)), so that the residuals stay continuous.

At the transition point the shear-stress coefficient starts at $C_\tau = 0.7\,C_{\tau,EQ}$, evaluated with the turbulent closures at the state of the transition point (constant `TRANSITION_CTAU_FACTOR`). **This factor is the best reading of Drela and Giles (1987) and is to be checked against the paper.**

## Assumptions

- Incompressible flow, two-dimensional, attached or mildly separated layers.
- The laminar relations are fits to similar (Falkner-Skan) profiles; non-similar flows use them as local closures.
- The turbulent relations are fits to the Swafford profile family and to equilibrium flows; the lag equation models non-equilibrium.
- The wake is a single layer with the sum of the two trailing-edge layers; its closures are those of a turbulent layer with zero skin friction and doubled dissipation.

## Envelope

The relations are valid for $H > 1$ (all relations divide by $H - 1$), $Re_\theta > 1$ in turbulent flow, and $U_s < 1$. The laminar relations cover $H$ from the stagnation value 2.24 to well beyond separation. The transition model is the envelope method for two-dimensional Tollmien-Schlichting waves in low free-stream turbulence; bypass transition is not modelled.

## Implementation

`ventorum/ibl2d/closure.py` and `ventorum/ibl2d/transition.py`. Every function is vectorized (no loop over stations) and runs with numpy or torch (CPU or CUDA). The partial derivatives are analytic; `ventorum/ibl2d/dual.py` combines them with the chain rule (forward-mode derivatives), so the Newton solvers get the exact Jacobian of the discrete equations.

## Verification

`tests/test_ibl_closure.py`:

- every partial derivative against central differences (step 1e-6 relative) on $H$ from 1.05 to 7 and $Re_\theta$ from 50 to 1e5, to 1e-6 relative;
- Blasius ($H = 2.591$): $H^*$ and $Re_\theta C_f/2$ within 1 % of 1.5725 and 0.2205;
- exact Falkner-Skan profiles, solved in the test for $\beta$ = 1, 0.3, 0, -0.1, -0.16: $H^*$ within 0.2 %, the dissipation factor within 1 %, the skin-friction factor within 0.01;
- Ludwieg-Tillmann at $H = 1.4$, $Re_\theta = 10^4$: within 15 %;
- $Re_{\theta 0}$ against the published closed form written in the test at $H$ = 2.216, 2.6, 3.0 (1e-12), and the other turbulent relations against their closed forms;
- $Re_{\theta 0}$ rises as $H$ falls from 3 to 2.2; the growth rate is zero below $Re_{\theta 0}$ and positive above;
- continuity at all branch points; numpy and torch agree to 1e-12; CUDA float32 values within 1e-5 relative (marked gpu).

## References

- M. Drela and M. B. Giles, "Viscous-inviscid analysis of transonic and low Reynolds number airfoils", AIAA Journal 25(10), 1987, pp. 1347-1355.
- M. Drela, "Two-dimensional transonic aerodynamic design and analysis using the Euler equations", doctoral thesis, MIT Gas Turbine Laboratory Report 187, 1986.
- T. J. Swafford, "Analytical approximation of two-dimensional separated turbulent boundary-layer velocity profiles", AIAA Journal 21(6), 1983, pp. 923-926.
- P. D. Green, D. G. Weeks and J. W. F. Brooman, "Prediction of turbulent boundary layers and wakes in compressible flow by a lag-entrainment method", ARC R and M 3791, 1977.
- V. M. Falkner and S. W. Skan, "Some approximate solutions of the boundary layer equations", ARC R and M 1314, 1930.
