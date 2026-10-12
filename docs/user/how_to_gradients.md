# How to compute adjoint gradients

`ventorum.gradients` gives exact gradients of the solver outputs with respect to the
lattice nodes, the flow variables, the section properties, the reference values, the
control deflections and the body rates, for the vortex lattice and the linear and
nonlinear lifting lines, with linear or tabulated airfoils. One production solve and
one adjoint computation give all gradients, without finite differences.

```python
import numpy as np
import ventorum as vt

wing = vt.LiftingSurface(
    name="Wing",
    semi_span=5.0,
    sections=[
        vt.WingSection(y_frac=0.0, chord=2.0),
        vt.WingSection(y_frac=1.0, chord=1.0),
    ],
)
aircraft = vt.Aircraft(surfaces=[wing])
condition = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
settings = vt.SolverSettings(solver_type="vlm", n_panels=12, n_chord=2)

result = vt.gradients(
    aircraft, condition, settings,
    outputs=("CL", "CD", "Cm"),
    wrt=("alpha", "nodes"),
    axes="body",
)
print(f"CL = {result.values['CL']:.4f}, dCL/dalpha = {result.jacobian['CL']['alpha']:.4f} per rad")
```

The call solves the case once (`n_solves == 1`) and then one adjoint system per output
(`n_adjoint`). The mesh topology is fixed: spanwise stations, chordwise count and
fractions do not change. Angles use rad, lengths use m.

## Inputs and outputs

Inputs (`wrt`): `"nodes"` (per surface the leading- and trailing-edge nodes of the
defining half), `"alpha"`, `"beta"`, `"V_inf"`, `"rho"`, `"h"`, `"phi"`, `"sections"`
(per strip `a0`, `alpha_L0`, `Cd0`, `Cm0` of the lattice), `"reference"`
(`S_ref`, `b_ref`, `c_ref`, `ref_point`), `"planform"` (per surface and
per section, see below; it cannot be combined with `"nodes"`), `"controls"` (per
control name, per unit deflection [1/rad]) and `"p"`, `"q"`, `"r"` (body rates
[rad/s]; see below).

For a tabulated airfoil the strip values of `"sections"` are the linear fit of the
polar that the linear solvers use (`a0`, `alpha_L0`); its profile drag and section
moment come from the table at the section angle, so `Cd0` and `Cm0` of a tabulated
strip have zero gradients. The nonlinear lifting line takes the section lift from the
table, so `a0` and `alpha_L0` of a tabulated strip have zero gradients there.

Outputs: `"CL"`, `"CY"`, `"CDi"`, `"CDp"`, `"CD"`, `"Cl"`, `"Cm"`, `"Cn"` in the axes
`axes` (`"body"`, `"stability"`, `"wind"`). The vector outputs `"strip_force"` and
`"section_cl"` go through `gradient_vjp` with a seed per output:

Use the strips of the solved lattice for the seed shape:

```python
loads = vt.analyze(aircraft, condition, settings).details["loads"]
seed = np.ones_like(loads.strip_force)
direction = vt.gradient_vjp(
    aircraft, condition, settings,
    seeds={"strip_force": seed},
    wrt=("alpha",),
)
print(f"d(sum of strip forces)/dalpha = {direction['alpha']}")
```

## Planform parameters

`wrt="planform"` gives gradients with respect to the planform parameters of
each surface: semi-span [m], sweep of the leading edge [rad], dihedral [rad],
incidence [rad] and position [m], and per section chord [m], twist [rad],
leading-edge offsets `x_le` and `z_le` [m], span fraction `y_frac`
(dimensionless) and airfoil incidence `alpha_L0` [rad]. The jacobian maps each
surface name to these values, with the sections as a list in order. A section
that does not set `x_le` or `z_le` gives None there (it has no such
parameter). The root and tip span fractions are pinned (0 and 1) and give
0.0.

```python
import numpy as np
import ventorum as vt

pwing = vt.LiftingSurface(
    name="Wing",
    semi_span=5.0,
    sections=[
        vt.WingSection(y_frac=0.0, chord=2.0),
        vt.WingSection(y_frac=1.0, chord=1.0),
    ],
)
paircraft = vt.Aircraft(surfaces=[pwing])
pcondition = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
psettings = vt.SolverSettings(solver_type="vlm", n_panels=12, n_chord=2)

presult = vt.gradients(
    paircraft, pcondition, psettings,
    outputs=("CL", "CD", "Cm"),
    wrt=("planform",),
    axes="body",
)
print(f"dCL/dsemi_span = {presult.jacobian['CL']['planform']['Wing']['semi_span']:.4f} per m")
```

The chain rule runs through a differentiable port of the geometry builder,
not through finite differences. The mesh topology stays fixed: the panel
count, the spacing choice and the snap assignment of each spanwise station do
not change, but the stations move with the parameters (a station snapped to a
section tracks its span fraction). The automatic reference values follow the
planform. Planform inputs use the full system, also for symmetric cases. The
translation identity holds per surface: the position gradient equals the
summed node gradients of that surface.

Read the result notes. Two cases make the production output jump, and a note
tells you:

* A section break that lies exactly on a mesh edge. Any change of the span
  fraction of that section moves the edge, so the output jumps at the base
  point. The span-fraction gradient of that section is the derivative for any
  nonzero change (central differences agree with it).
* A mirror pair (a surface and its `mirrored()` copy) that gives the automatic
  reference values. The gradient is per copy; a change of both copies together
  is the sum of the two gradients. A change of one copy alone breaks the pair,
  so the automatic reference values change. `check_gradients` skips these
  components. Set `S_ref`, `b_ref` and `c_ref` explicitly to remove the jump.

## Control deflections and body rates

`wrt=("controls",)` gives, per control name, the derivative per unit change of the
`deflection` field [1/rad]. All controls with one name move together, and the left
half of an antisymmetric control (an aileron) moves the opposite way, as in
`vt.control_derivatives`. The derivative at zero deflection is included. The vortex
lattice with two or more chordwise panels turns the normals of the panels aft of the
hinge (chordwise mode); the lifting lines and the vortex lattice with one chordwise
panel use the thin-airfoil deflected section (section mode).

`wrt=("p", "q", "r")` gives the derivatives with respect to the body rates [rad/s]
(flight-dynamics body axes: `p > 0` right wing down, `q > 0` nose up, `r > 0` nose
right), also at zero rates. The rotation is quasi-steady about the moment reference
point, as in the production solve. A note gives the nondimensional derivatives per unit
`p b/(2V)`, `q c/(2V)` and `r b/(2V)`, the values of `vt.rate_derivatives` in body axes.

```python
import numpy as np
import ventorum as vt

cwing = vt.LiftingSurface(
    name="Wing",
    semi_span=5.0,
    sections=[
        vt.WingSection(y_frac=0.0, chord=2.0),
        vt.WingSection(y_frac=1.0, chord=1.0),
    ],
)
cwing.controls = [
    vt.ControlSurface(name="flap", eta_start=0.2, eta_end=0.55, hinge_x_c=0.75,
                      deflection=np.radians(5.0)),
    vt.ControlSurface(name="aileron", eta_start=0.6, eta_end=0.95, hinge_x_c=0.7,
                      deflection=0.0, symmetric=False),
]
caircraft = vt.Aircraft(surfaces=[cwing])
ccondition = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), q=0.1)
csettings = vt.SolverSettings(solver_type="vlm", n_panels=12, n_chord=4)

cresult = vt.gradients(
    caircraft, ccondition, csettings,
    outputs=("CL", "Cl", "Cm"),
    wrt=("controls", "p", "q", "r"),
)
print(f"dCL/dflap = {cresult.jacobian['CL']['controls']['flap']:.4f} per rad")
print(f"dCl/daileron = {cresult.jacobian['Cl']['controls']['aileron']:.4f} per rad")
print(f"dCm/dq = {cresult.jacobian['Cm']['q']:.4f} per rad/s")
```

The inputs p and r, and an antisymmetric control, break the symmetry about y = 0:
these gradients use the full system. The pitch rate q and symmetric controls keep the
fold when the rest of the case allows it.

## Nonlinear lifting line and tabulated airfoils

The nonlinear lifting line solves a residual that is nonlinear in the circulation. The
adjoint uses the Newton Jacobian of the residual at the converged state (Phillips and
Snyder 2000), so one adjoint solve per output gives the gradients, as for the linear
solvers. Use a tight `tolerance` (for example 1e-10) when you compare the gradients with
differences: the converged state has a residual of the order of the tolerance.

* A solve that did not converge raises `GradientScopeError`: the gradient of a state
  that does not solve the equations is not defined.
* Past the maximum lift (a section lift slope that is zero or negative on a strip) the
  call warns, and a note says that the gradient is local and that the solution may not be
  unique.
* The derivatives of a tabulated polar are the derivatives of the interpolation of the
  production code (a shape-preserving cubic with a continuous first derivative). At a
  table point the derivative of the interval to the right is used, and a note names the
  strips. Outside the table the polar keeps its end value, so its derivative is zero; a
  note names those strips too.
* The planform input needs linear airfoils (`GradientScopeError` otherwise).

Lifting-line node gradients: the vortex core on the collinear bound segments of the
lifting line makes the node sensitivity of the lifting line depend on a length scale of
about 1e-5 m (the core size), so these node gradients are not physical at that scale. An
investigation of this effect is open. A result note repeats this caveat when the
lifting line returns node or planform gradients. Use the vortex lattice for node
gradients, or check them with differences at the mesh scale.

## Trust

The reference copy also gives the output values. `forward_mismatch` reports the largest
relative difference to the production result. Above 1e-10 (when the gradients and the
production solve are both float64; else above 1e-4) the call raises `RuntimeError`: the gradient would not be the gradient of the reported
result. `check_gradients` checks the gradients against Richardson-extrapolated central
differences of the production solve. The default tolerance is `rtol=1e-6`. Use
`rtol=1e-3` for float32 gradients (float32 is the default precision) and `rtol=1e-6` for
float64 gradients:

```python
check = vt.check_gradients(
    aircraft, condition, settings,
    outputs=("CL", "CD", "Cm"),
    wrt=("alpha", "nodes"),
    n_directions=3,
    rtol=1e-3 if result.precision_gradient == "float32" else 1e-6,
)
print(f"gradients pass: {check['pass']} ({check['compared']} compared, {check['skipped']} skipped)")
print(check["reasons"])
```

The check fails when a component is outside the tolerance, and also when an input group
has no compared component; `reasons` says why.

Gradients run on the device of the production solve (the device setting of
`ventorum.gpu`). They run in float32 by default (the GPU precision setting, or the precision of the
machine profile, selects it)
and in float64 when that setting selects float64. `precision_primal` gives the precision
of the production solve. The symmetric system is folded only when the case is
symmetric and every requested input and output is symmetric (node inputs count as
symmetric when every surface is symmetric); else the full system is used.

## Speed: the path of the partial derivatives

The keyword `method` selects how the partial derivatives of the O(n^2) parts (system
rows, load points, Trefftz points) are made:

* `"auto"` (the default): the backend that `set_vjp_backend` forces, else the choice of
  the machine profile (`ventorum-tune` measures it per case size and device), else the
  default backend: Numba (then Cython, then numpy) on the CPU, NVIDIA Warp on the GPU,
  and the reference copy on a GPU without Warp.
* `"kernels"`: the compiled vector-Jacobian-product kernels of the device (a
  `ValueError` when the device has none).
* `"reference"`: the PyTorch reference copy (slower; the correctness reference).

All paths give the same gradients to round-off. `result.partials` names the path and
the backend, for example `"kernels:numba"`:

```python
fast = vt.gradients(aircraft, condition, settings, outputs=("CL",), wrt=("alpha",), method="kernels")
print(fast.partials)
```

## Limits

The Fourier solver is refused with `GradientScopeError` (a `ValueError`): it is a
closed form, and central differences of the production solve are the method there.
Ground effect is supported: `h` is the height of the moment reference point, and the
ground plane and its images follow the inputs. With body rates the ground images stay
mirror images of the vortices, as in the production solve.
