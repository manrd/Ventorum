# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Unsteady vortex-ring aerodynamics package (in development).

This package holds the shared unsteady vortex-ring core:

* :mod:`ventorum.unsteady.rings`: quadrilateral vortex-ring lattice geometry,
  ground mirror images and steady parity helpers.
* :mod:`ventorum.unsteady.tail`: closed-form 3D semi-infinite harmonic
  far-wake tail.
* :mod:`ventorum.unsteady.harmonic`: frequency-domain harmonic vortex-ring
  lattice solver for longitudinal motions (heave and pitch) in free air and in
  flat-ground effect.

The package is in development and has no public interface yet.

References
----------
* J. Katz and A. Plotkin, "Low-Speed Aerodynamics", 2nd ed., Cambridge
  University Press, 2001, chapters 12 and 13 (vortex-ring lattice and
  unsteady vortex-lattice method).
* K. C. Hall, "Eigenanalysis of unsteady flows about airfoils, cascades, and
  wings", AIAA Journal 32(12), 1994, pp. 2426-2432 (harmonic wake relation;
  cited through Parenteau 2021).
* M. Parenteau, "A Non-Linear Frequency Domain Potential Flow Model for
  Compressible, Transonic and Viscous Aeroelastic Analyses", PhD thesis,
  Polytechnique Montreal, 2021, chapter 4 (frequency-domain vortex lattice;
  cited through study material).
"""

from __future__ import annotations

__all__: list[str] = []
