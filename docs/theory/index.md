# Theory manual

Ventorum uses the name Lanchester–Prandtl lifting-line theory, to credit both F. W. Lanchester and L. Prandtl.

The theory manual gives, for each model in the code, the equations, the assumptions, the envelope of validity, the implementation and the references, with credit to the authors of the methods. Each chapter has the same parts:

1. The model in one paragraph.
2. The equations, from a cited source or from a derivation in the chapter.
3. The assumptions.
4. The envelope: what is outside the model, and what the code does there.
5. The implementation: the modules and functions that compute the model.
6. The verification cases that check it, in the [verification report](../verification_report.md).
7. The references.

The values of the model parameters come from the code (the API reference shows them). The verification results come from `validation/run_verification.py` and are in the verification report, not in the chapters. Where a chapter gives a result value to explain a method or a limit, the value comes from a test that the chapter names or from a study of the method; such a value is rounded and is not a verification result.

| Chapter | Model | Main references |
| --- | --- | --- |
| [Conventions](conventions.md) | Axes, angles, reference values, notation | B. L. Stevens, F. L. Lewis and E. N. Johnson (2016) |
| [Vortex filaments](vortex_kernels.md) | Biot-Savart law of the bent horseshoe vortex, the core model and the cross-surface core | J. Katz and A. Plotkin (2001); L. Rosenhead (1930) |
| [Classical lifting line](classical_lifting_line.md) | Glauert's Fourier-series solution of the Lanchester–Prandtl lifting-line equation | F. W. Lanchester (1907); L. Prandtl (1918, 1919); H. Glauert (1926) |
| [Numerical lifting line](numerical_lifting_line.md) | Linear and nonlinear lifting line with section polars | W. F. Phillips and D. O. Snyder (2000); W. F. Phillips (2004) |
| [Vortex-lattice method](vortex_lattice.md) | Bent horseshoe vortices on chordwise panels and deformed geometry | V. M. Falkner (1943); Katz and Plotkin (2001); M. Drela (2014) |
| [Compressibility](compressibility.md) | Prandtl-Glauert rule and Goethert transformation for subsonic flow | B. H. Goethert (1946); M. Drela (2014); J. Katz and A. Plotkin (2001) |
| [Forces and moments](forces_and_moments.md) | Kutta-Joukowski near-field forces, Trefftz-plane induced drag, section data, static derivatives | Katz and Plotkin (2001); Drela (2014); E. Trefftz (1921) |
| [Ground effect](ground_effect.md) | Method of images over flat ground, height and bank derivatives (one state or a sweep), Irodov criterion | C. Wieselsberger (1921); K. V. Rozhdestvensky (2000, 2006); R. D. Irodov (1970) |
| [Adjoint gradients](gradients.md) | Discrete adjoint gradients of the vortex lattice and the linear and nonlinear lifting lines | J. R. R. A. Martins and J. T. Hwang (2013); J. R. R. A. Martins and A. Ning (2021) |
| [Control surfaces](control_surfaces.md) | Thin-airfoil flap theory, panel deflection and section modification, hinge moments, control derivatives | H. Glauert (1926); Katz and Plotkin (2001); Etkin and Reid (1996) |
| [Body rates](rotation_rates.md) | Quasi-steady rotation in the boundary condition and the loads, rate derivatives p, q, r | Katz and Plotkin (2001); Etkin and Reid (1996); Phillips and Snyder (2000) |
| [Trim solver](trim.md) | Longitudinal and lateral trim for target lift and zero moments | B. Etkin and L. D. Reid (1996); J. E. Dennis and R. B. Schnabel (1996) |
| [Symmetry](symmetry.md) | Use of the mirror symmetry of the flow | Katz and Plotkin (2001) |
| [Trust score and error bars](trust_score.md) | Envelope checks (experimental) and the planned error-bar layers | W. L. Oberkampf and C. J. Roy (2010); ASME V&V 20 (2009); P. J. Roache (1998) |
| [3D panel method (Ventorum-3DP)](ventorum_3dp.md) | Source and doublet panels on closed surfaces, Dirichlet condition, Kutta conditions, symmetry and ground images, Trefftz drag | L. Morino and C. C. Kuo (1974); J. L. Hess and A. M. O. Smith (1967); Katz and Plotkin (2001) |
| [3D panel solver](panel3d.md) | Solver details of Ventorum-3DP: ground images, condition grouping, root section with dihedral, fold guard, solver interface | Morino and Kuo (1974); B. Maskew (1987); Katz and Plotkin (2001) |
| [3D panel influence kernels](panel3d_kernels.md) | Exact and far-field influences of the constant source and doublet panels | Hess and Smith (1964, 1967); J. N. Newman (1986); Katz and Plotkin (2001) |
| [3D panel control surfaces](panel3d_controls.md) | Plain control surfaces in 3D panel meshes | H. Glauert (1926, 1927); Katz and Plotkin (2001); B. Maskew (1987) |
| [2D airfoil model (Ventorum-2D)](ventorum_2d.md) | Airfoil geometry, 2D panel method with ground effect, integral boundary layer and viscous-inviscid interaction | I. H. Abbott and A. E. von Doenhoff (1959); M. Drela and M. B. Giles (1987); M. Drela (1989) |
| [Airfoils](airfoils.md) | Airfoil geometry, NACA sections, paneling and flaps | Abbott and von Doenhoff (1959); M. S. Selig et al. (1995); E. N. Jacobs et al. (1933) |
| [Airfoil geometry](airfoil_geometry.md) | Coordinate formats, splines and repaneling | Selig et al. (1995); Abbott and von Doenhoff (1959); Katz and Plotkin (2001) |
| [NACA airfoils](naca_airfoils.md) | NACA 4-digit and 5-digit generators | Abbott and von Doenhoff (1959); C. L. Ladson et al. (1996); Jacobs et al. (1933) |
| [Airfoil flaps](airfoil_flaps.md) | Plain-flap deflection of airfoil coordinates | H. Glauert (1926, 1927) |
| [2D panel solver](panel2d.md) | Inviscid 2D panels with ground effect and several elements | Katz and Plotkin (2001); M. Drela (1989); L. M. Milne-Thomson (1958) |
| [2D viscous solver](panel2d_viscous.md) | Viscous-inviscid interaction with a global Newton coupling, profile drag | M. J. Lighthill (1958); Drela and Giles (1987); H. B. Squire and A. D. Young (1938) |
| [Boundary-layer closures](ibl2d_closure.md) | Integral boundary-layer closure relations and e^N transition | Drela and Giles (1987); M. Drela (1986); T. J. Swafford (1983) |
| [Boundary-layer march](ibl2d_march.md) | Integral boundary-layer equations and the direct march | Drela and Giles (1987); Drela (1986); H. Schlichting and K. Gersten (2017) |
| [Viscous coupling](viscous_coupling.md) | Angle-of-attack correction with section polars, and parasite drag | S. Gallay and E. Laurendeau (2015); Glauert (1927); Katz and Plotkin (2001) |
| [Viscous alpha-correction coupling](viscous_alpha_correction.md) | Alpha-correction coupling of the lattice, lifting-line and panel solvers with 2D polars | J. B. Tseng and C. E. Lan (1988) |
| [Parasite drag](parasite_drag.md) | Component build-up of the parasite drag | D. P. Raymer (2018); E. Torenbeek (1982); S. F. Hoerner (1965) |

```{toctree}
:maxdepth: 1
:hidden:

conventions
vortex_kernels
classical_lifting_line
numerical_lifting_line
vortex_lattice
compressibility
forces_and_moments
ground_effect
gradients
control_surfaces
rotation_rates
trim
symmetry
trust_score
ventorum_3dp
panel3d
panel3d_kernels
panel3d_controls
ventorum_2d
airfoils
airfoil_geometry
naca_airfoils
airfoil_flaps
panel2d
panel2d_viscous
ibl2d_closure
ibl2d_march
viscous_coupling
viscous_alpha_correction
parasite_drag
```
