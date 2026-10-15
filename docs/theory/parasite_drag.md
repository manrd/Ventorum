---
orphan: true
---

# Parasite-drag build-up

## Model

The parasite (zero-lift) drag of parts that the lifting models do not cover (fuselages, hulls, nacelles, struts) comes from a component build-up in the form of D. P. Raymer, "Aircraft Design: A Conceptual Approach", AIAA, 6th ed. 2018, section 12.5. Each component contributes its skin friction times a form factor (which adds the pressure drag of its thickness) times an interference factor (which adds the drag of its junctions), scaled by its wetted area over the reference area. The module also cross-checks the polar-based profile drag of the lifting surfaces. It changes no solver result and it couples to no solver.

## Equations

### Skin friction

Let $Re = \rho V L / \mu$ be the Reynolds number from the air density $\rho$ [kg/m^3] and the velocity $V$ [m/s] of the flight condition with the component reference length $L$ [m]; the viscosity $\mu$ [kg/(m s)] is the ISA sea-level value (ISO 2533:1975).

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

Rough surfaces use the cutoff Reynolds number of Raymer (2018), section 12.5, with the equivalent sand roughness $k$ [m]:

```{math}
Re_{co} = 38.21\,\left(\frac{L}{k}\right)^{1.053}, \qquad C_f = C_f(\min(Re, Re_{co})) .
```

When the flight Reynolds number exceeds the cutoff, the layer is roughness-limited and treated as fully turbulent at the cutoff Reynolds number.

### Form factors

Lifting sections use by default the wing form factor of Raymer (2018), section 12.5, adapted to incompressible flow: the Mach factor $1.34\,M^{0.18}$ is set to 1 (this package covers Mach below 0.3 and applies no compressibility correction here) and the sweep factor is kept. With the thickness ratio $t/c$ [-], the position of maximum thickness $x_t$ [-] and the sweep of the maximum-thickness line $\Lambda$ [rad]:

```{math}
FF = \left[1 + \frac{0.6}{x_t}\,\frac{t}{c} + 100\,\left(\frac{t}{c}\right)^4\right]\,(\cos\Lambda)^{0.28} .
```

The alternative is the wing form factor of E. Torenbeek, "Synthesis of Subsonic Airplane Design", Delft University Press, 1982, appendix F, evaluated on the thickness ratio normal to the maximum-thickness line $(t/c)_n = (t/c)/\cos\Lambda$:

```{math}
FF = 1 + 2.7\,(t/c)_n + 100\,(t/c)_n^4 .
```

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

The wetted area of a lifting surface is twice the planform area times $(1 + 0.2\,t/c)$, the approximation of Raymer (2018), section 7.9. For a wing-body combination the user subtracts the fuselage carry-through or passes the exposed area directly.

## Trust

Each component carries a relative uncertainty (default 0.10, the typical scatter of form-factor methods). The absolute uncertainty of the total is the root sum square of the absolute component uncertainties, treated as independent. The estimate is a conceptual-design value: it checks the order of the profile drag, but it does not replace measured polars.
