# Run the 3D panel method

Ventorum-3DP solves steady, incompressible, inviscid potential flow around three-dimensional lifting configurations and non-lifting bodies. It uses a low-order boundary element formulation with flat quadrilateral and triangular panels (Morino and Kuo, 1974; Katz and Plotkin, 2001). The formulation enforces an internal Dirichlet boundary condition of zero perturbation potential inside the closed body.

The linear aerodynamic influence matrix is factorized once per configuration by LU decomposition. It solves for six fundamental unit onset flow states (three translation velocities and three angular rotation rates). Arbitrary angles of attack, sideslip, and rotation rates in free air follow by linear superposition.

## When to use the vortex lattice instead

The vortex-lattice method (VLM) models thin lifting surfaces with horseshoe vortices along the mean camber surface. It solves fast with few panels and low memory.

* **Use the vortex lattice** for preliminary design, rapid sweeps, dynamic simulations, and optimization loops where thin-wing assumptions hold.
* **Use the 3D panel method** when you need:
  * Detailed surface pressure distributions ($C_p$) across upper and lower surfaces;
  * Accurate capture of section thickness and displacement effects;
  * Suction peaks and cavitation margins;
  * Physical ground clearance defined by the lower surface skin;
  * Near-field surface pressure integration alongside far-field Trefftz-plane drag.

## Build a wing with section shapes

A panel mesh requires cross-section coordinates. Define section shapes on `WingSection` objects using standard airfoil names (for example `"2412"`, `"0012"`) or coordinate arrays. If `shape` is omitted, the solver uses NACA 0012 and records a note in the trust score.

```python
import numpy as np
import ventorum as vt
from ventorum.core.datatypes import Aircraft, FlightCondition, LiftingSurface, SolverSettings, WingSection

wing = LiftingSurface(
    name="MainWing",
    semi_span=4.0,
    sections=[
        WingSection(y_frac=0.0, chord=1.0, shape="2412"),
        WingSection(y_frac=1.0, chord=0.7, shape="2412"),
    ],
    is_symmetric=True,
)
aircraft = Aircraft(name="Monoplane", surfaces=[wing])
```

By default, `panel_scale_root_thickness=True`. On a symmetric wing with dihedral, the root section thickness is multiplied by $1 / \cos(\Gamma)$, where $\Gamma$ is the root dihedral angle [rad]. This scaling keeps the thickness normal to the wing surface equal to the section thickness.

## Run in free air and in ground effect

Run the analysis with `solver="panel3d"`. In free air, the solver factorizes the system matrix once for the geometry. It solves unit onset flow states and superposes them for arbitrary flight conditions.

In ground effect, set `h` on the flight condition. The method of images enforces zero normal velocity on the ground plane (Wieselsberger, 1921). The wake leaves the trailing edge parallel to the ground plane. Contact with the ground raises `GroundStrikeError`.

```python
# Free-air condition
cond_free = FlightCondition(V_inf=30.0, alpha=np.radians(4.0))

# Settings with small mesh counts for quick execution
settings = SolverSettings(
    solver_type="panel3d",
    n_panels=6,
    n_chord=6,
)

# Solve the free-air condition
res_free = vt.analyze(aircraft, condition=cond_free, settings=settings)

# Ground-effect condition at clearance height h = 0.8 m
cond_ground = FlightCondition(V_inf=30.0, alpha=np.radians(4.0), h=0.8)
res_ground = vt.analyze(aircraft, condition=cond_ground, settings=settings)
```

## Read Cp, forces, trust, and dispatch records

A `SolverResult` contains integrated aerodynamic totals, surface pressure coefficients, spanwise distributions, trust ratings, and execution dispatch metadata:

```python
# Integrated aerodynamic coefficients
totals = res_free.totals
print(f"CL: {totals.CL:.4f}")
print(f"CDi (Trefftz far field): {totals.CDi:.6f}")
print(f"CDi (near-field pressure): {totals.CDi_nearfield:.6f}")
print(f"Cm: {totals.Cm:.4f}")

# Surface pressure coefficients (Cp) on panel centroids
cp = res_free.details["cp"]
print(f"Panel count: {len(cp)}")
print(f"Minimum Cp: {np.min(cp):.4f}")

# Spanwise strip loads
strip = res_free.spanwise[0]
print(f"Spanwise strips: {len(strip.Cl)}")

# Trust score and warnings
trust = totals.trust
print(f"Trust rating: {trust.rating}")
for warning in trust.warnings:
    print(f"Trust warning: {warning}")

# Execution dispatch metadata
dispatch = res_free.details["dispatch"]
print(f"Backend: {dispatch['backend']}")
print(f"Device: {dispatch['device']}")
print(f"Factorizations in this call: {dispatch['factorization_count']}")
```

## Choose the backend and device

Ventorum-3DP supports multiple execution backends via `panel_backend`:

* `"auto"` (default): uses the GPU when the workload exceeds tuned or default thresholds; otherwise uses the CPU. On the CPU, it chooses compiled Numba kernels if installed, or pure NumPy reference kernels.
* `"numba"`: compiled CPU kernels with multi-threading.
* `"numpy"`: pure NumPy reference CPU kernels.
* `"gpu"`: GPU execution with NVIDIA Warp influence kernels and PyTorch linear algebra.

```python
from ventorum import gpu

# Select the CPU backend explicitly
settings_cpu = SolverSettings(
    solver_type="panel3d",
    n_panels=6,
    n_chord=6,
    panel_backend="numpy",
)
res_cpu = vt.analyze(aircraft, condition=cond_free, settings=settings_cpu)

# GPU acceleration (if supported hardware and packages are available)
if gpu.available():
    settings_gpu = SolverSettings(
        solver_type="panel3d",
        n_panels=6,
        n_chord=6,
        panel_backend="gpu",
    )
    res_gpu = vt.analyze(aircraft, condition=cond_free, settings=settings_gpu)
    print(f"GPU device: {res_gpu.details['dispatch']['device']}")
```

On the GPU, influence kernels assemble in `float64` to prevent on-plane singular tolerance issues near long wake panels. The assembled matrix is then cast to the solve precision (`float32` by default; see [Run solves on the GPU](how_to_gpu)).

## Solver settings reference

The panel solver accepts the following knobs in `SolverSettings`:

* `n_chord`: chordwise panel count per surface (default: 40 when None).
* `n_panels`: spanwise panel count per semi-span (default: 80).
* `panel_chord_spacing`: `"cosine"` (default, clustered at leading and trailing edges) or `"uniform"`.
* `panel_kutta`: `"morino"` (default, linear potential jump condition) or `"pressure"` (iterative non-linear zero pressure jump condition).
* `panel_backend`: `"auto"` (default), `"numba"`, `"numpy"`, or `"gpu"`.
* `panel_ground_solver`: `"auto"` (default), `"direct"`, or `"krylov"`. Reuses an anchor LU factorization with GMRES for CPU ground sweeps with Morino Kutta. Free air, GPU backends, and pressure Kutta use direct LU solves.
* `panel_cache`: `True` (default) or `False`. Caches mesh, mesh check, and factorized solvers in memory. Free the cache with `ventorum.panel3d.clear_cache()`.
* `panel_wake_length`: wake length [m] (default: 50 times the largest streamwise chord).
* `panel_estimate_mesh_error`: `False` (default) or `True`. Runs a directional mesh convergence study (three levels each in chordwise and spanwise directions) and attaches Richardson error estimates to the trust score.

```python
# Custom settings with pressure Kutta and direct ground solver
custom_settings = SolverSettings(
    solver_type="panel3d",
    n_panels=6,
    n_chord=6,
    panel_kutta="pressure",
    panel_ground_solver="direct",
    panel_cache=True,
    panel_estimate_mesh_error=False,
)
res_custom = vt.analyze(aircraft, condition=cond_free, settings=custom_settings)
print(f"Kutta condition: {res_custom.details['panel']['kutta']}")
```

## Memory limits and configuration budget

Dense influence matrices require storage proportional to $N^2$ and factorization effort proportional to $N^3$, where $N$ is the total panel count.

To manage memory during large sweeps and repeated runs:
* **Host factor budget:** 25 % of physical system memory.
* **Device factor budget:** 50 % of free GPU memory plus device factors currently held.
* **Configuration limit:** the solver retains at most 256 configurations (free air and ground orientations). The limit applies at the start of each solve call, evicting the least recently used configurations.
* **Factor release:** Morino LU factors are dropped once unit onset solutions are computed, unless retained as a CPU Krylov anchor.

The dispatch dictionary records dropped factors in `dispatch["factors_dropped"]` and evicted configurations in `dispatch["configurations_evicted"]`.

## Control surfaces and edge ramp

Define control surfaces on a `LiftingSurface` using `ControlSurface`. Plain flaps deflect the camber and skin aft of the hinge line.

Deflected controls use `"transition"` side edges by default. The deflection ramps linearly over a spanwise transition width of `CONTROL_TRANSITION = 0.2` local chords centred on the control boundary.

A sharp open gap creates a fluid window between the upper and lower surfaces with a singular potential difference and strong side-edge vortex shedding. This causes the strip circulation next to the gap to drop as the mesh is refined. The sealed transition ramp eliminates this singularity and converges smoothly with spanwise refinement.

```python
from ventorum.core.datatypes import ControlSurface

# Define a wing with a 10 deg trailing-edge plain flap
flap = ControlSurface(
    name="flap",
    eta_start=0.2,
    eta_end=0.7,
    hinge_x_c=0.75,
    deflection=np.radians(10.0),
    symmetric=True,
)

wing_flapped = LiftingSurface(
    name="FlappedWing",
    semi_span=4.0,
    sections=[
        WingSection(y_frac=0.0, chord=1.0, shape="0012"),
        WingSection(y_frac=1.0, chord=1.0, shape="0012"),
    ],
    controls=[flap],
    is_symmetric=True,
)
aircraft_flapped = Aircraft(name="FlappedAircraft", surfaces=[wing_flapped])

res_flap = vt.analyze(aircraft_flapped, condition=cond_free, settings=settings)
print(f"Flapped wing CL: {res_flap.totals.CL:.4f}")
```

## References

* Morino, L. and Kuo, C. C., 1974, Subsonic potential aerodynamics for complex configurations: A general theory, AIAA Journal, Vol. 12, No. 2, pp. 191-197.
* Hess, J. L. and Smith, A. M. O., 1967, Calculation of potential flow about arbitrary bodies, Progress in Aeronautical Sciences, Vol. 8, pp. 1-138.
* Katz, J. and Plotkin, A., 2001, Low-Speed Aerodynamics, 2nd ed., Cambridge University Press.
* Wieselsberger, C., 1921, Ueber den Fluegelwiderstand in der Naehe des Bodens, Zeitschrift fuer Flugtechnik und Motorluftschiffahrt, Vol. 12, No. 10.
