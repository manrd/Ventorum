---
orphan: true
---

# Parasite-drag build-up

## Model

The parasite (zero-lift) drag of parts that the lifting models do not cover (fuselages, hulls, nacelles, struts) comes from a component build-up in the form of D. P. Raymer, "Aircraft Design: A Conceptual Approach", AIAA, 6th ed. 2018, section 12.5. Each component contributes its skin friction times a form factor (which adds the pressure drag of its thickness) times an interference factor (which adds the drag of its junctions), scaled by its wetted area over the reference area. The module also cross-checks the polar-based profile drag of the lifting surfaces. It changes no solver result and it couples to no solver.

## Scope limit

The package covers incompressible flow (Mach below 0.3). The module applies no compressibility correction: the Mach factor $1.34\,M^{0.18}$ of the Raymer wing form factor is set to 1, and the skin-friction relations are the incompressible ones. This is a stated limit of the module, not a low-Mach evaluation of Raymer's factor.

## Equations

### Skin friction

Let $Re = \rho V L / \mu$ be the Reynolds number from the air density $\rho$ [kg/m^3] and the velocity $V$ [m/s] of the flight condition, the component reference length $L$ [m] and the dynamic viscosity $\mu$ [Pa s]. The viscosity is an input of `parasite_drag`; its default is the ISA sea-level value (ISO 2533:1975), the same constant as the strip Reynolds numbers of the viscous coupling.

Laminar flow uses the Blasius relation (H. Blasius, 1908):

```{math}
C_{f,\mathrm{lam}} = \frac{1.328}{\sqrt{Re}} .
```

Fully turbulent flow uses the Prandtl-Schlichting relation (H. Schlichting, "Boundary-Layer Theory", Springer, 9th ed. 2017):

```{math}
C_{f,\mathrm{turb}} = \frac{0.455}{(\log_{10} Re)^{2.58}} .
```

Mixed laminar-turbulent flow with transition at the fraction $x_{tr}$ [-] of the reference length uses the Schlichting composite rule with $Re_{tr} = x_{tr} Re$:

```{math}
C_f = C_{f,\mathrm{turb}}(Re) - x_{tr}\,\left[C_{f,\mathrm{turb}}(Re_{tr}) - C_{f,\mathrm{lam}}(Re_{tr})\right] .
```

Hence $x_{tr} = 0$ is fully turbulent and $x_{tr} = 1$ is fully laminar. At a transition Reynolds number of $5 \times 10^5$ the rule gives a correction term of $1612/Re$, against the value $1700/Re$ quoted in some textbooks.

The composite rule has a physical meaning only for a laminar run long enough that the laminar law gives less friction than the turbulent law. The two mean-friction laws give the same value at the Reynolds number $Re_{tr,\min}$ (`RE_TRANSITION_MIN`, about $1.2 \times 10^4$, computed in the module by bisection). A laminar run with $Re_{tr} < Re_{tr,\min}$ counts as turbulent. Without this guard, a run near $Re_{tr} = 1$ gives a negative $C_f$, and short runs give more than the fully turbulent value. With the guard and $Re \ge Re_{tr,\min}$, $C_f$ is continuous in $x_{tr}$ (the composite rule equals the fully turbulent value at the guard), it does not rise when the laminar run grows, and it stays between $C_{f,\mathrm{lam}}(Re)$ and $C_{f,\mathrm{turb}}(Re)$. Below $Re = Re_{tr,\min}$ the turbulent law is outside its range; the function then returns the turbulent value for $x_{tr} < 1$ and the laminar value for $x_{tr} = 1$.

Rough surfaces use the cutoff Reynolds number of Raymer (2018), section 12.5, with the equivalent sand roughness $k$ [m]:

```{math}
Re_{co} = 38.21\,\left(\frac{L}{k}\right)^{1.053} .
```

Raymer evaluates the turbulent skin friction at $\min(Re, Re_{co})$. The cutoff acts on the turbulent part only; the laminar part keeps its value. For mixed flow the module adds the roughness increment of the turbulent law, weighted by the turbulent fraction of the length (an approximation of this module):

```{math}
C_f = C_{f,\mathrm{smooth}}(Re, x_{tr}) + (1 - x_{tr})\,\left[C_{f,\mathrm{turb}}(\min(Re, Re_{co})) - C_{f,\mathrm{turb}}(Re)\right] .
```

The increment is zero below the cutoff, so $C_f$ is continuous across the cutoff. A fully turbulent rough plate gives $C_{f,\mathrm{turb}}(Re_{co})$, and a fully laminar plate keeps $C_{f,\mathrm{lam}}(Re)$.

### Form factors

Lifting sections use by default the wing form factor of Raymer (2018), section 12.5, in the incompressible scope above: the Mach factor is 1 and the sweep factor is kept. With the thickness ratio $t/c$ [-], the position of maximum thickness $x_t$ [-] and the sweep of the maximum-thickness line $\Lambda$ [rad]:

```{math}
FF = \left[1 + \frac{0.6}{x_t}\,\frac{t}{c} + 100\,\left(\frac{t}{c}\right)^4\right]\,(\cos\Lambda)^{0.28} .
```

The alternative is the wing form factor of E. Torenbeek, "Synthesis of Subsonic Airplane Design", Delft University Press, 1982, appendix F, on the streamwise thickness ratio:

```{math}
FF = 1 + 2.7\,\frac{t}{c} + 100\,\left(\frac{t}{c}\right)^4 .
```

This form has no sweep term. A variant on the thickness ratio normal to the maximum-thickness line is not used, because its source could not be cited precisely.

Streamlined bodies use by default the fuselage form factor of Raymer (2018), section 12.5, with the fineness $f = l/d$ [-]:

```{math}
FF = 1 + \frac{60}{f^3} + \frac{f}{400} .
```

The alternative is the streamlined-body form factor of S. F. Hoerner, "Fluid-Dynamic Drag", 1965, chapter 6, with $d/l = 1/f$:

```{math}
FF = 1 + 1.5\,(d/l)^{3/2} + 7\,(d/l)^3 .
```

Both body factors fall with rising fineness.

### Build-up

With the skin-friction coefficient $C_{f,i}$ [-], the form factor $FF_i$ [-], the interference factor $Q_i$ [-] (an input with documented typical values, never a hidden constant) and the wetted area $S_{wet,i}$ [m^2], each component gives, counted on the reference area $S_{ref}$ [m^2]:

```{math}
C_{D0,i} = C_{f,i}\,FF_i\,Q_i\,\frac{S_{wet,i}}{S_{ref}}, \qquad C_{D0} = \sum_i C_{D0,i} .
```

### Wetted area of a lifting surface

The planform area of a lifting surface comes from the section chords by the trapezium rule. The span distance between two sections is the true distance between their leading edges in the y-z plane: `semi_span` is measured along the dihedral line (or it is the y extent when the sections set `z_le`, and the z offsets add to the distance). The chords are streamwise, and two parallel streamwise chords at this distance bound a trapezium of area $(c_1 + c_2)/2$ times the distance, whatever the sweep. So sweep and dihedral need no cosine factor; section twist is neglected.

The exposed area leaves out the span length `covered_root_span` [m] next to the root section on each half (the part inside a fuselage); the default 0 gives the gross area. The wetted area is

```{math}
S_{wet} = 2\,S_{exposed}\,\left(1 + 0.2\,\frac{t}{c}\right) .
```

The factor 2 counts the upper and the lower surface. The term $0.2\,t/c$ is an approximation of the thickness factor between the planform and the wetted area of a section; Raymer (2018), chapter 7 (wetted area), gives a thickness-dependent factor of this kind.

## Trust

Each component carries a relative uncertainty (default 0.10, the typical scatter of form-factor methods). The absolute uncertainty of the total is the root sum square of the absolute component uncertainties, treated as independent. The estimate is a conceptual-design value: it checks the order of the profile drag, but it does not replace measured polars.
