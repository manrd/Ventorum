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

The chapters contain no result values. The values of the model parameters come from the code (the API reference shows them); the results come from `validation/run_verification.py`.

| Chapter | Model | Main references |
| --- | --- | --- |
| [Conventions](conventions.md) | Axes, angles, reference values, notation | B. L. Stevens, F. L. Lewis and E. N. Johnson (2016) |
| [Vortex filaments](vortex_kernels.md) | Biot-Savart law of the bent horseshoe vortex, the core model and the cross-surface core | J. Katz and A. Plotkin (2001); L. Rosenhead (1930) |
| [Classical lifting line](classical_lifting_line.md) | Glauert's Fourier-series solution of the Lanchester–Prandtl lifting-line equation | F. W. Lanchester (1907); L. Prandtl (1918, 1919); H. Glauert (1926) |
| [Numerical lifting line](numerical_lifting_line.md) | Linear and nonlinear lifting line with section polars | W. F. Phillips and D. O. Snyder (2000); W. F. Phillips (2004) |
| [Vortex-lattice method](vortex_lattice.md) | Bent horseshoe vortices on chordwise panels and deformed geometry | V. M. Falkner (1943); Katz and Plotkin (2001); M. Drela (2014) |
| [Forces and moments](forces_and_moments.md) | Kutta-Joukowski near-field forces, Trefftz-plane induced drag, section data, static derivatives | Katz and Plotkin (2001); Drela (2014); E. Trefftz (1921) |
| [Ground effect](ground_effect.md) | Method of images over flat ground, height and bank derivatives, Irodov criterion | C. Wieselsberger (1921); K. V. Rozhdestvensky (2000, 2006); R. D. Irodov (1970) |
| [Control surfaces](control_surfaces.md) | Thin-airfoil flap theory, panel deflection and section modification | H. Glauert (1926); Katz and Plotkin (2001) |
| [Trim solver](trim.md) | Longitudinal and lateral trim for target lift and zero moments | B. Etkin and L. D. Reid (1996); J. E. Dennis and R. B. Schnabel (1996) |
| [Symmetry](symmetry.md) | Use of the mirror symmetry of the flow | Katz and Plotkin (2001) |
| [Trust score and error bars](trust_score.md) | Envelope checks (experimental) and the planned error-bar layers | W. L. Oberkampf and C. J. Roy (2010); ASME V&V 20 (2009); P. J. Roache (1998) |

```{toctree}
:maxdepth: 1
:hidden:

conventions
vortex_kernels
classical_lifting_line
numerical_lifting_line
vortex_lattice
forces_and_moments
ground_effect
control_surfaces
trim
symmetry
trust_score
```

The 3D panel method is planned for a later phase. It has no chapter yet.
