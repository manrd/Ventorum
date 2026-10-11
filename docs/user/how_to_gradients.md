# How to compute adjoint gradients

`ventorum.gradients` gives exact gradients of the solver outputs with respect to the
lattice nodes, the flow variables, the linear section properties and the reference
values, for the vortex lattice and the linear lifting line. One production solve and
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
(per strip `a0`, `alpha_L0`, `Cd0`, `Cm0` of linear airfoils) and `"reference"`
(`S_ref`, `b_ref`, `c_ref`, `ref_point`).

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

## Trust

The reference copy also gives the output values. `forward_mismatch` reports the largest
relative difference to the production result. Above 1e-10 in float64 (1e-4 in float32)
the call raises `RuntimeError`: the gradient would not be the gradient of the reported
result. `check_gradients` checks the gradients against Richardson-extrapolated central
differences of the production solve. Use `rtol=1e-3` for float32 gradients (the
default) and `rtol=1e-6` for float64 gradients:

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
`ventorum.gpu`). They run in float32 by default (the GPU precision setting selects it)
and in float64 when that setting selects float64. `precision_primal` gives the precision
of the production solve. The symmetric system is folded only when the case is
symmetric and every requested input and output is symmetric (node inputs count as
symmetric when every surface is symmetric); else the full system is used.

## Speed: the path of the partial derivatives

The keyword `method` selects how the partial derivatives of the O(n^2) parts (system
rows, load points, Trefftz points) are made:

* `"auto"` (the default): the choice of the machine profile (`ventorum-tune` measures
  it per case size and device); without a profile the compiled kernels, Numba on the
  CPU and NVIDIA Warp on the GPU.
* `"kernels"`: the compiled vector-Jacobian-product kernels of the device.
* `"reference"`: the PyTorch reference copy (slower; the correctness reference).

All paths give the same gradients to round-off. `result.partials` names the path and
the backend, for example `"kernels:numba"`:

```python
fast = vt.gradients(aircraft, condition, settings, outputs=("CL",), wrt=("alpha",), method="kernels")
print(fast.partials)
```

## Limits

Phase 1 covers the vortex lattice and the linear lifting line with linear airfoils.
The nonlinear lifting line, tabulated airfoils, nonzero control deflections, nonzero
body rates and the Fourier solver are refused with `GradientScopeError` (a `ValueError`); a later version adds them. Ground
effect is supported: `h` is the height of the moment reference point, and the ground
plane and its images follow the inputs.
