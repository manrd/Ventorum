---
orphan: true
---

# Integral boundary-layer closures (Ventorum-2D)

## Model

The viscous engine of Ventorum-2D is a two-equation integral boundary layer
with a shear-stress lag equation and an envelope e^N transition model. The
flow is incompressible (edge Mach 0), so the density shape factor is not
needed and the kinematic shape factor equals `H`. This page states the
closure relations: the laminar fits to the Falkner-Skan profile family, the
turbulent fits to the Swafford profile family, the lag-entrainment rate
coefficient, and the transition correlations. The values of the fit
constants live in the code (`ventorum/ibl2d/closure.py`,
`ventorum/ibl2d/transition.py`); the equations below use their names.

## Equations

### Laminar closures (Falkner-Skan family)

With `H = delta_star / theta` (dimensionless), `Re_theta` the
momentum-thickness Reynolds number (dimensionless), `Hstar = theta_star /
theta` (dimensionless), `Cf` the skin-friction coefficient (dimensionless)
and `CD` the dissipation coefficient (dimensionless):

```{math}
\text{(L1)} \qquad Hstar(H) = a_1 + b_1 \frac{(4 - H)^2}{H} \quad (H \le 4),
\qquad Hstar(H) = a_1 - b_2 \frac{(H - 4)^2}{H} \quad (H > 4).
```

```{math}
\text{(L2)} \qquad F(H) = Re_\theta \frac{C_f}{2}
  = c_0 + c_1 \frac{(7.4 - H)^2}{H - 1}.
```

```{math}
\text{(L3)} \qquad G(H) = Re_\theta \frac{2 C_D}{Hstar}
  = g_0 + g_1 (4 - H)^{5.5} \quad (H \le 4),
```

```{math}
G(H) = g_0 - g_2 \frac{(H - 4)^2}{1 + g_3 (H - 4)^2} \quad (H > 4).
```

(L1) and (L3) keep the switch of the source at `H = 4`, with the same value
and the same slope on both sides. (L2) is one smooth formula; it is zero
near `H = 4` (laminar separation) and negative above it. Source: the
Falkner-Skan fits of M. Drela (1986), as used in Drela and Giles (1987).
Range of validity: `1 < H <= 7.5`, attached below `H = 4`.

### Turbulent closures (Swafford profile family)

```{math}
\text{(T1)} \qquad C_f(H, Re_\theta)
  = A_0 e^{-A_1 H} (\log_{10} Re_\theta)^{-(E_0 + E_1 H)}
  + S_0 \left( \tanh(4 - H) - \tfrac{1}{2} \right).
```

The first term is the attached skin friction from the wall-wake profile
family; the second term lowers `Cf` through separation and makes it
negative in reversed flow. Source: the profile fit of T. J. Swafford
(1983), in the form used by Drela and Giles (1987). Range of validity:
attached turbulent flow, `1 <= H <= 2.5`, `Re_theta >= 50` (the logarithm
needs `Re_theta > 1`).

```{math}
\text{(T2)} \qquad Hstar(H, Re_\theta)
  = s_0 + s_1 (H - 1) + s_2 (H - 1)^2 - s_3 \log_{10}(Re_\theta / 10^4).
```

```{math}
\text{(T3)} \qquad U_s(H, Re_\theta)
  = u_0 + u_1 (H - 1) + u_2 (H - 1)^2 - u_3 \log_{10}(Re_\theta / 10^4).
```

`Us = u_s / u_e` (dimensionless) is the slip velocity of the outer wake
profile. Source: the wall-wake profile family of Swafford (1983), as used
by Drela and Giles (1987). Range of validity: as (T1).

```{math}
\text{(T4)} \qquad C_{\tau,eq}(H, Hstar, U_s)
  = k \left( \frac{Hstar - 1}{Hstar \, U_s} \right)^2.
```

`Ctau_eq = tau_max / (rho_e u_e^2)` (dimensionless) is the equilibrium
shear-stress coefficient on the equilibrium locus. At a zero-pressure
gradient flat plate it equals half the local skin-friction coefficient
(the maximum shear is the wall shear there). `H` enters a coupled solve
through `Hstar`, so the direct partial with respect to `H` is zero. Source:
the equilibrium locus behind the lag-entrainment method of Green, Weeks
and Brooman (1977), as used by Drela and Giles (1987). Range of validity:
`Hstar > 1`, `Us > 0`.

```{math}
\text{(T5)} \qquad C_D(H, Re_\theta, U_s, C_\tau)
  = \frac{C_f}{2} U_s + k_d \, (C_\tau - C_{\tau,eq}).
```

At equilibrium the dissipation is the wall shear work `(Cf / 2) Us`; off
equilibrium it rises linearly with the shear excess. `Cf` is (T1) and
`Ctau_eq` is (T4) with `Hstar` from (T2). Source: Drela and Giles (1987).
Range of validity: as (T1), with `C_tau > 0`.

### Shear lag (Green, Weeks and Brooman idea)

```{math}
\text{(G1)} \qquad \delta_{star} \frac{dC_\tau}{ds}
  = K \, (\sqrt{C_{\tau,eq}} - \sqrt{C_\tau}), \qquad
  K(C_\tau) = K_c \sqrt{C_\tau}.
```

The rate scales with the turbulence level `sqrt(C_tau)`; `Kc` is the lag
constant. Source: the lag-entrainment idea of Green, Weeks and Brooman
(1977), in the form of Drela and Giles (1987). Range of validity:
`C_tau > 0`.

### Transition (envelope e^N)

```{math}
\text{(E1)} \qquad Re_{\theta,0}(H) = r_0 + e^{r_1 - H}.
```

The critical Reynolds number falls as `H` rises (an adverse gradient
destabilizes the layer) and approaches a floor in strongly adverse
gradients. Source: M. Drela (1986), as used in Drela and Giles (1987).
Range of validity: `1 <= H <= 7`.

```{math}
\text{(E2)} \qquad g(H) = \frac{dN}{dRe_\theta} = n_0 + n_1 (H - 1).
```

The envelope slope is positive: past the critical Reynolds number every
Reynolds-number increment adds to the exponent. Source: M. Drela (1986),
as used in Drela and Giles (1987). Range of validity: as (E1).

```{math}
\text{(E3)} \qquad \frac{dN}{ds} = g(H) \frac{dRe_\theta}{ds}
  \quad (Re_\theta \ge Re_{\theta,0}), \qquad \frac{dN}{ds} = 0
  \quad \text{else}.
```

The switch at the critical Reynolds number is the source relation itself.
Source: Drela and Giles (1987). Range of validity: as (E1).

## Assumptions

- The flow is incompressible (edge Mach 0); no density shape factor is
  used.
- The laminar closures describe the Falkner-Skan profile family
  (attached and separated); the turbulent closures describe the Swafford
  wall-wake profile family.
- The lag equation relaxes the maximum shear stress toward its
  equilibrium value over a few displacement thicknesses.
- The transition model tracks the envelope of the Tollmien-Schlichting
  amplification curves; it has no bypass or crossflow terms.

## Envelope

- Laminar relations: valid for `1 < H <= 7.5`; (L2) is singular at `H = 1`.
- Turbulent relations: valid for attached turbulent flow (`H <= 2.5`,
  `Re_theta >= 50`); values outside are smooth extrapolations.
- The lag coefficient needs `C_tau > 0`; the equilibrium locus needs
  `Hstar > 1` and `Us > 0`.
- The transition correlations cover `1 <= H <= 7`; the critical Reynolds
  number keeps a positive floor.

## Implementation

- `ventorum.ibl2d.closure`: `laminar_h_star`, `laminar_cf_factor`,
  `laminar_dissipation_factor`, `turbulent_cf`, `turbulent_h_star`,
  `turbulent_slip_velocity`, `equilibrium_shear_coefficient`,
  `turbulent_dissipation_coefficient`, `shear_lag_coefficient`.
- `ventorum.ibl2d.transition`: `critical_re_theta`, `amplification_rate`,
  `transition_growth_rate`.
- Each function returns the value and its analytic partial derivatives as
  a tuple, and takes an optional `xp` array module (numpy by default,
  torch for CPU or CUDA tensors).

## Verification

- `tests/test_ibl_closure.py` checks every partial derivative against a
  central finite difference (relative step 1e-6, float64) to 1e-6
  relative on `H` in [1.05, 7] and `Re_theta` in [50, 1e5].
- The Blasius flat-plate values (`H = 2.591`: `Hstar = 1.5725`,
  `Re_theta Cf / 2 = 0.2205`) agree within 2 %.
- The flat-plate skin friction (`H = 1.4`, `Re_theta = 1e4`) agrees with
  the Ludwieg-Tillmann correlation within 15 %.
- The critical Reynolds number rises as `H` falls from 3 to 2.2; growth
  is zero below it and positive above it.
- numpy and torch (float64, CPU) agree to 1e-12; a CUDA test (marked
  `gpu`) checks float32 to 1e-5.
- Continuity holds (1e-6 relative) at the `H = 4` switches; the switch of
  (E3) at the critical Reynolds number is recorded with its exact size.

## References

- M. Drela and M. B. Giles, "Viscous-inviscid analysis of transonic and
  low Reynolds number airfoils", AIAA Journal 25(10), 1987, pp. 1347-1355.
- M. Drela, doctoral thesis, MIT Gas Turbine Laboratory Report 187, 1986.
- T. J. Swafford, "Analytical approximation of two-dimensional separated
  turbulent boundary-layer velocity profiles", AIAA Journal 21(6), 1983.
- P. D. Green, D. G. Weeks and J. W. F. Brooman, "Prediction of turbulent
  boundary layers and wakes in compressible flow by a lag-entrainment
  method", ARC R and M 3791, 1977.
