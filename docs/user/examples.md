# Examples

The `examples/` directory has these scripts:

| Script | Content |
| --- | --- |
| `01_elliptic_wing.py` | Elliptic wing; the solvers against the theoretical e = 1 |
| `02_tapered_wing.py` | Taper and twist |
| `03_swept_wing.py` | Sweep and dihedral |
| `04_nonlinear_stall.py` | Nonlinear section tables and the start of stall |
| `05_wing_tail.py` | Wing and tail; downwash on the tail |
| `06_alpha_sweep.py` | Angle-of-attack sweep and drag polar |
| `07_xfoil_integration.py` | XFOIL polars in the nonlinear solver (skipped if XFOIL is not installed) |
| `08_multi_instance_parallel.py` | Several independent cases in parallel |
| `09_aerosonde_uav.py` | Aerosonde UAV model with plots |
| `10_ai_agent_tool_integration.py` | AI agent tool calls, MCP schemas, stability derivatives and trust score |
| `11_trim.py` | Trim of a wing, tail and fin with aileron, elevator and rudder: longitudinal trim to a target CL with Cm = 0, and lateral trim with sideslip with zero roll, pitch and yaw moments |
| `12_openmdao_twist.py` | Twist optimisation in OpenMDAO with adjoint gradients: minimum CD at a fixed CL (needs the extra `mdo`; without OpenMDAO it prints a message and stops) |
| `airfoil_ground_effect.py` | 2D inviscid NACA 4412 in ground effect: Cl against h/c and Cp pictures, saved as PNG files |
| `airfoil_naca_flap.py` | 2D inviscid NACA 2412 with a plain flap: geometry and Cp pictures against the clean section, saved as PNG files |
