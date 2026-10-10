# Trim an aircraft

The trim solver finds the angle of attack and control deflections that give a target lift coefficient with zero aerodynamic moments.

## Longitudinal trim

Longitudinal trim solves for angle of attack and pitch control deflection:

$$C_L = C_{L,\text{target}}, \quad C_m = 0$$

Moments are evaluated in body axes about the moment reference point of the aircraft.

```python
import numpy as np
import ventorum as vt

wing = vt.LiftingSurface(
    name="wing",
    semi_span=6.0,
    sections=[
        vt.WingSection(y_frac=0.0, chord=1.8),
        vt.WingSection(y_frac=1.0, chord=1.0),
    ],
    n_panels=20,
)

h_tail = vt.LiftingSurface(
    name="h_tail",
    semi_span=2.0,
    position=np.array([4.6, 0.0, 0.3]),
    sections=[
        vt.WingSection(y_frac=0.0, chord=0.9),
        vt.WingSection(y_frac=1.0, chord=0.5),
    ],
    controls=[
        vt.ControlSurface(
            name="elevator",
            eta_start=0.0,
            eta_end=1.0,
            hinge_x_c=0.70,
            deflection=0.0,
            symmetric=True,
        )
    ],
    n_panels=10,
)

aircraft = vt.Aircraft(
    name="Aircraft",
    surfaces=[wing, h_tail],
    ref_point=np.array([0.45, 0.0, 0.0]),
)

condition = vt.FlightCondition(V_inf=35.0, alpha=np.radians(2.0))
settings = vt.SolverSettings(solver_type="vlm", n_panels=20, n_chord=4)

res = vt.trim(
    aircraft,
    condition,
    settings,
    CL_target=0.50,
    pitch_control="elevator",
)

print(f"Trim status: {res.status}")
print(f"Trim alpha:  {res.alpha_deg:.2f} deg")
print(f"Elevator:    {res.deflections_deg['elevator']:.2f} deg")
```

The user aircraft is never modified in place. The trimmed aircraft clone is available in `res.aircraft`.

## Lateral trim with sideslip

When sideslip is present, lateral trim solves for four unknowns simultaneously:
- Angle of attack $\alpha$
- Pitch control deflection $\delta_{\text{pitch}}$
- Roll control deflection $\delta_{\text{roll}}$
- Yaw control deflection $\delta_{\text{yaw}}$

These variables satisfy four equations:

$$C_L = C_{L,\text{target}}, \quad C_m = 0, \quad C_l = 0, \quad C_n = 0$$

Add an aileron to the wing and a vertical fin with a rudder:

```python
# Wing with aileron
wing.controls = [
    vt.ControlSurface(
        name="aileron",
        eta_start=0.6,
        eta_end=1.0,
        hinge_x_c=0.75,
        deflection=0.0,
        symmetric=False,
    )
]

# Vertical fin with rudder
v_fin = vt.LiftingSurface(
    name="v_fin",
    semi_span=1.8,
    is_symmetric=False,
    dihedral=np.pi / 2.0,
    position=np.array([4.6, 0.0, 0.0]),
    sections=[
        vt.WingSection(y_frac=0.0, chord=0.9),
        vt.WingSection(y_frac=1.0, chord=0.5),
    ],
    controls=[
        vt.ControlSurface(
            name="rudder",
            eta_start=0.0,
            eta_end=1.0,
            hinge_x_c=0.70,
            deflection=0.0,
            symmetric=True,
        )
    ],
    n_panels=8,
)

aircraft = vt.Aircraft(
    name="CompleteAircraft",
    surfaces=[wing, h_tail, v_fin],
    ref_point=np.array([0.45, 0.0, 0.0]),
)

# Flight condition with 3 deg sideslip
condition_lateral = vt.FlightCondition(
    V_inf=35.0,
    alpha=np.radians(2.0),
    beta=np.radians(3.0),
)

res_lat = vt.trim(
    aircraft,
    condition_lateral,
    settings,
    CL_target=0.50,
    pitch_control="elevator",
    roll_control="aileron",
    yaw_control="rudder",
)

print(f"Status:   {res_lat.status}")
print(f"Alpha:    {res_lat.alpha_deg:.2f} deg")
print(f"Elevator: {res_lat.deflections_deg['elevator']:.2f} deg")
print(f"Aileron:  {res_lat.deflections_deg['aileron']:.2f} deg")
print(f"Rudder:   {res_lat.deflections_deg['rudder']:.2f} deg")
```

Both `roll_control` and `yaw_control` must be supplied together for lateral trim.

## Ground effect

When `condition.h` is set, the trim solver models flight in ground effect. The height of `Aircraft.ref_point` remains constant while the angle of attack changes.

Because ground effect increases wing lift at a given angle of attack, the trimmed angle of attack in ground effect is usually lower than in free air.

## Execution device and precision

Every solve within `trim` runs on the CPU in 64-bit precision (`float64`). Finite differences in 32-bit precision suffer from numerical noise and can fail to converge. The global device setting is restored when `trim` finishes. Do not run `trim` concurrently with other functions that modify the device setting.

## Background

For the mathematical formulation and convergence rules, see [Trim solver theory](../theory/trim.md). For a complete runnable example, see `examples/11_trim.py`.
