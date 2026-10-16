---
orphan: true
---

# Integral boundary-layer equations and direct march (Ventorum-2D)

## Model

Ventorum-2D has one implementation of the discrete boundary-layer equations (`ventorum/ibl2d/equations.py`). The direct march (`ventorum/ibl2d/march.py`, edge velocity given) and the coupled viscous-inviscid Newton solver (`ventorum/panel2d/vii.py`) use the same functions, so both solve the same equations with the same closures ([](ibl2d_closure.md)).

Equations (incompressible; Drela and Giles 1987), with the surface coordinate $\xi$ from the stagnation point:

```{math}
\frac{d\theta}{d\xi} + (2 + H)\,\frac{\theta}{u_e}\frac{du_e}{d\xi} = \frac{C_f}{2},
```

```{math}
\theta\,\frac{dH^*}{d\xi} + H^*(1 - H)\,\frac{\theta}{u_e}\frac{du_e}{d\xi} = 2C_D - H^*\frac{C_f}{2},
```

the lag equation (C9) in turbulent flow and in the wake, and the envelope equation $dN/d\xi$ (E4) in laminar flow.

Variables per station: $\theta$, $\delta^*$ (the coupled solver uses the mass defect $m = u_e\delta^*$ instead, see [](panel2d_viscous.md)) and a third variable $c$: $N$ in laminar flow and $\sqrt{C_\tau}$ in turbulent flow and in the wake.

## Discretization

Between stations 1 and 2 ($\Delta\xi$) the derivatives are logarithmic differences and the source terms are trapezoidal averages:

```{math}
\ln\frac{\theta_2}{\theta_1} + (2 + \bar H)\ln\frac{u_{e2}}{u_{e1}} - \Delta\xi\,q\,\overline{\left(u_e \frac{C_f}{2\theta}\right)} = 0,
```

```{math}
\ln\frac{H^*_2}{H^*_1} + (1 - \bar H)\ln\frac{u_{e2}}{u_{e1}} - \Delta\xi\,q\,\overline{\left(u_e\,\frac{2C_D/H^* - C_f/2}{\theta}\right)} = 0,
```

```{math}
\ln\frac{c_2}{c_1} + \ln\frac{u_{e2}}{u_{e1}} - \Delta\xi\,\overline{L} = 0, \qquad
L = \frac{5.6\,(c_{EQ} - c)}{2\delta} + \frac{4}{3\delta^*}\left[\frac{C_f}{2} - \left(\frac{H - 1}{6.7 H}\right)^2\right],
```

where the bar is the mean of the two stations. The factor $1/u_e$ of the source terms is integrated with the logarithmic mean, $q = \ln(u_{e2}/u_{e1})/(u_{e2} - u_{e1})$; this is exact for the linear edge velocity near a stagnation point and second order in general (a series is used when $u_{e2} \approx u_{e1}$). For laminar flow, $u_e C_f/(2\theta) = f(H)/(Re\,\theta^2)$ stays finite at a stagnation point.

Amplification: $N_2 = N_1 + \Delta\xi\,\phi\,r_1$, with $r_1$ the growth rate (without the switch) of station 1 and $\phi$ the fraction of the interval where the excess $e = Re_\theta - Re_{\theta 0}$ (linear between the stations) is positive. This integrates the published switch exactly and keeps the equation continuous in the states. It uses the upstream rate (first order) so that the transition interval, whose downstream station is turbulent, uses the same increase; the transition point then moves continuously from one interval to the next.

## Start

- Stagnation point (Hiemenz flow, $u_e = a\xi$): $\theta$ and $H$ are constant; the two equations give $(2 + H)\,Re\,a\,\theta^2 = f(H)$ and $3 f(H) = (2 + H) g(H)$, with $f = Re_\theta C_f/2$ and $g = Re_\theta\,2C_D/H^*$ of the laminar closure. The closure gives $H = 2.240$ (exact Hiemenz value 2.216) and $\theta\sqrt{a\,Re} = 0.287$ (exact 0.2923).
- Flat plate ($u_e$ constant, Blasius): $\theta^2 = 2 f(H)\,\xi/(Re\,u_e)$ and $g(H) = f(H)$, which gives $H = 2.590$. A given $\theta_0$ uses this shape factor.

## Transition

The transition interval holds the point where $N$ reaches $n_{crit}$ (or the forced position). It is divided at the fraction $w = (n_{crit} - N_1)/(\Delta\xi\,r_1)$ (or the forced fraction; the smaller one applies). The state at the transition point is the linear interpolation of $\theta$, $\delta^*$ and $u_e$ ($H$ is continuous at transition). The laminar equations hold on the part before the point and the turbulent equations on the part after it; the momentum and kinetic-energy residuals of the two parts are added. The shear stress starts at $C_\tau = 0.7\,C_{\tau,EQ}$ ([](ibl2d_closure.md), to be checked against the paper).

The closures jump at the transition point and the turbulent layer then relaxes over a few thicknesses; a central average over this jump makes the stiff relaxation overshoot (in a test with steps of 100 momentum thicknesses the trapezoidal rule gave $H < 1$). The turbulent part of the transition interval therefore uses the implicit (backward Euler) rule. The turbulent interval that follows the transition interval uses the downstream weight $(1 + w)/2$: implicit when the transition point is at its upstream end ($w = 1$ in the transition interval), trapezoidal when it is at the upstream end of the transition interval ($w = 0$). The equations then change continuously when the transition point moves from one interval to the next. All other intervals use the trapezoidal rule.

## Separation in direct mode

With the edge velocity given, the kinetic-energy equation is singular where $dH^*/dH = 0$ (Goldstein singularity): $H = 4$ for the laminar closure and $H = H_0(Re_\theta)$ for the turbulent closure. When a station reaches this shape factor, or the station Newton iteration has no solution, the march stops, sets `separated` and returns the stations up to that point (the rest is NaN). The coupled solver has no such limit.

## Station solution and batch mode

Each station is solved by Newton iteration on $(\theta, \delta^*, c)$ with the exact Jacobian (forward-mode derivatives of the closures). The step is limited so that $\theta$, $\delta^*$ and $c$ change by at most 50 % and $H - 1$ falls by at most 50 %. The tolerance is 1e-10 in float64 and 1e-5 in float32.

`march_direct_batch` marches many surfaces at once: the station loop is sequential and the work at each station is vectorized over the batch, with numpy or torch (CPU or CUDA).

## Assumptions and envelope

Direct mode only (no separated flow). The surface coordinate must increase; the start is a stagnation point ($u_e = 0$ at the first station) or a flat-plate start.

## Verification

`tests/test_ibl_march.py` and `tests/test_ibl_equations.py`:

- laminar flat plate (Re 1e5 and 1e6): $\theta$ within 2 % of $0.664\,x/\sqrt{Re_x}$ and $H$ within 2 % of 2.59 after 5 % of the plate;
- Hiemenz flow: $H$ within 2 % of 2.216, $\theta$ constant, $\theta\sqrt{a Re}$ within 5 % of 0.2923;
- free transition on a flat plate at Re 1e7 with $n_{crit} = 9$ between $Re_x$ = 1e6 and 5e6; it moves downstream for $n_{crit}$ = 4, 9, 14;
- turbulent flat plate (transition forced at 1 % chord, Re 1e7): $C_f$ within 10 % of $0.0592\,Re_x^{-0.2}$ for $Re_x$ from 1e6 to 1e7;
- decelerating flow $u_e = 1 - 0.5 s$: the turbulent $H$ grows and separation is flagged; the laminar layer separates too;
- batch: numpy batch equals single runs to 1e-12, torch float64 equals numpy to 1e-10, CUDA float32 within 1e-4 on $\theta$ (marked gpu);
- the Jacobians of all interval kinds (laminar, turbulent, wake, transition, post-transition), of the stagnation start and of the wake start against central differences; the continuity of the post-transition interval.

## References

- M. Drela and M. B. Giles, "Viscous-inviscid analysis of transonic and low Reynolds number airfoils", AIAA Journal 25(10), 1987, pp. 1347-1355.
- M. Drela, doctoral thesis, MIT Gas Turbine Laboratory Report 187, 1986.
- K. Hiemenz, "Die Grenzschicht an einem in den gleichformigen Flussigkeitsstrom eingetauchten geraden Kreiszylinder", Dinglers Polytechnisches Journal 326, 1911.
- H. Schlichting and K. Gersten, Boundary-Layer Theory, 9th ed., Springer, 2017 (Blasius and Hiemenz solutions, turbulent flat-plate friction).
