---
orphan: true
---

# Direct integral boundary-layer march (Ventorum-2D)

## Model

The direct march integrates the two-equation integral boundary layer
along a surface with a prescribed edge velocity, in incompressible flow
only. It solves the integral momentum equation (M1) and the
kinetic-energy shape-factor equation (M2) of Drela and Giles (1987),
with the shear-stress lag equation (M3) in turbulent flow and the e^N
envelope equation (M4) in laminar flow. The closures live in
`ventorum.ibl2d.closure` and `ventorum.ibl2d.transition` and are stated
in `docs/theory/ibl2d_closure.md`.

With `s` the arc length [m or chord units], `ue` the edge velocity
ratio (dimensionless), `theta` the momentum thickness [m or chord
units], `H = delta_star / theta` the kinematic shape factor
(dimensionless), `Hstar` the energy shape factor (dimensionless), `Cf`
the skin-friction coefficient (dimensionless), `CD` the dissipation
coefficient (dimensionless), `Ctau` the maximum shear-stress coefficient
(dimensionless) and `N` the amplification exponent (dimensionless):

```{math}
\text{(M1)} \qquad \frac{d\theta}{ds}
  = \frac{C_f}{2} - (2 + H) \frac{\theta}{u_e} \frac{du_e}{ds}.
```

```{math}
\text{(M2)} \qquad \frac{d\theta^*}{ds}
  = 2 C_D - 3 \frac{\theta^*}{u_e} \frac{du_e}{ds},
  \qquad \theta^* = H^* \theta.
```

```{math}
\text{(M3)} \qquad \delta_{star} \frac{dC_\tau}{ds}
  = K (\sqrt{C_{\tau,eq}} - \sqrt{C_\tau}), \qquad
  K = K_c \sqrt{C_\tau}.
```

```{math}
\text{(M4)} \qquad \frac{dN}{ds}
  = g(H) \frac{dRe_\theta}{ds} \quad (Re_\theta \ge Re_{\theta,0}),
  \qquad \frac{dN}{ds} = 0 \quad \text{else}.
```

(M1) is the von Karman momentum integral. (M2) is the kinetic-energy
integral with the pressure-gradient factor 3, which reproduces the
Falkner-Skan similarity (Blasius and Hiemenz) with the laminar closures.
(M3) is the lag-entrainment rate of Green, Weeks and Brooman in the form
of Drela and Giles. (M4) integrates the envelope slope `g(H)` over the
momentum-thickness Reynolds number past the critical value. The
reference Reynolds number is `Re_ref = V_ref L / nu` with `L` equal to
one unit of `s`, so `Re_theta = ue theta Re_ref`.

## Discretization

Each step uses the implicit trapezoidal rule (second order). The solver
runs a Newton iteration per station with the analytic partial
derivatives of the closure module. Laminar steps solve for `theta` and
`H`; the exponent `N` follows explicitly because it does not feed back
into the closures. Turbulent steps solve for `theta`, `H` and `Ctau`
together. The edge gradient `due/ds` uses central differences inside
the grid and one-sided differences at the ends.

## Start

A stagnation start applies when `ue` is proportional to `s` near
`s = 0`. The shape factor solves `G / F = 3 / (2 + H)` from the laminar
closures (the similarity condition), and the thickness follows from the
momentum balance. Else a Blasius start applies with `H = 2.591`; the
thickness follows from the flat-plate similarity with the laminar skin
friction factor, unless the caller gives `theta0`.

## Transition and separation

Natural transition occurs where `N` reaches `ncrit`; forced transition
occurs at the given station. The solver interpolates the transition
point inside the step, keeps `theta` continuous there, initializes the
turbulent shape factor to the attached profile value, and restarts the
turbulent system with the equilibrium shear coefficient at the
transition state. Direct mode only: when `H` passes the laminar
separation limit `H = 4`, or when the turbulent skin friction reaches
zero, the march stops, marks the station as separated, and returns the
partial result with NaN after the last valid station.

## Output and batch mode

`BoundaryLayerResult` holds `theta`, `delta_star`, `H`, `Cf`, `Ctau`,
`N`, the transition and separation positions, and the transpiration
velocity `d(ue delta_star)/ds` for the later coupling.
`march_direct_batch` marches many independent surfaces at once: the
station loop stays sequential and the work per station is vectorized
over the batch through the closure array API (`xp=numpy` or
`xp=torch`, CPU or CUDA).

## Assumptions and envelope

- Incompressible flow, steady, no suction or blowing.
- Direct mode only; no inverse (prescribed displacement thickness) solve.
- Laminar closures cover `1 < H <= 7.5`; turbulent closures target
  attached flow and are smooth extrapolations outside.
- The e^N envelope tracks Tollmien-Schlichting growth only.

## Verification

- `tests/test_ibl_march.py` checks the laminar flat plate against the
  Blasius thickness, the stagnation shape factor against the Hiemenz
  value, free transition with monotonic `N`, the turbulent flat-plate
  skin friction against the one-fifth power law, separation in an
  adverse gradient, and numpy/torch/CUDA batch parity.
- `tests/test_ibl_closure.py` checks the closures used by the march.

## References

- M. Drela and M. B. Giles, "Viscous-inviscid analysis of transonic and
  low Reynolds number airfoils", AIAA Journal 25(10), 1987, pp.
  1347-1355.
- M. Drela, doctoral thesis, MIT Gas Turbine Laboratory Report 187,
  1986.
- E. Hiemenz, "Die Grenzschicht an einem in den gleichfoermigen
  Fluessigkeitsstrom eingetauchten geraden Kreiszylinder", Dinglers
  Polytechnisches Journal 326, 1911.
- H. Blasius, "Grenzschichten in Fluessigkeiten mit kleiner Reibung",
  Zeitschrift fuer Mathematik und Physik 56, 1908.
- P. D. Green, D. G. Weeks and J. W. F. Brooman, "Prediction of
  turbulent boundary layers and wakes in compressible flow by a
  lag-entrainment method", ARC R and M 3791, 1977.
- F. R. Houwink and A. E. P. Veldman, "Steady and unsteady separated
  flow computations for transonic airfoils", AIAA paper 83-1790, 1983.
- T. J. Swafford, "Analytical approximation of two-dimensional
  separated turbulent boundary-layer velocity profiles", AIAA Journal
  21(6), 1983.
