# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Unsteady vortex-ring aerodynamics package.

This package holds the unsteady vortex-ring core and public dynamic-derivative
interface:

* :mod:`ventorum.unsteady.rings`: quadrilateral vortex-ring lattice geometry,
  ground mirror images and steady parity helpers.
* :mod:`ventorum.unsteady.tail`: closed-form 3D semi-infinite harmonic
  far-wake tail.
* :mod:`ventorum.unsteady.harmonic`: frequency-domain harmonic vortex-ring
  lattice solver for longitudinal motions (heave and pitch) in free air and in
  flat-ground effect.
* :mod:`ventorum.unsteady.derivatives`: public :func:`dynamic_derivatives`
  function and :class:`DynamicDerivativesResult`.
* :mod:`ventorum.unsteady.gusts`: convected gust and wind-shear fields
  (:func:`sharp_edged`, :func:`one_minus_cosine`, :func:`wind_shear`,
  :func:`linear_shear`, :func:`logarithmic_shear`).
* :mod:`ventorum.unsteady.response`: public :func:`time_response` and
  :func:`indicial_response` functions and their result records.

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
* W. P. Rodden and J. P. Giesing, "Application of oscillatory aerodynamic
  theory to estimation of dynamic stability derivatives", Journal of Aircraft
  7(3), 1970, pp. 272-275 (equivalent derivatives at reduced frequency k).
"""

from __future__ import annotations

from ventorum.unsteady.derivatives import DynamicDerivativesResult, dynamic_derivatives
from ventorum.unsteady.gusts import (
    linear_shear,
    logarithmic_shear,
    one_minus_cosine,
    sharp_edged,
    wind_shear,
)
from ventorum.unsteady.response import (
    IndicialResult,
    TimeResponseResult,
    indicial_response,
    time_response,
)

__all__ = [
    "DynamicDerivativesResult",
    "IndicialResult",
    "TimeResponseResult",
    "dynamic_derivatives",
    "indicial_response",
    "linear_shear",
    "logarithmic_shear",
    "one_minus_cosine",
    "sharp_edged",
    "time_response",
    "wind_shear",
]
