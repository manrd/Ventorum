# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Reference solutions for verification of the solvers.

Each function gives a closed form, or an independent numerical solution of a
classical equation, with its source. These are theory references (the same
model, solved another way), not experimental data.

Provenance status: the sources below could not be opened in the session that
wrote this module (no network access to the archives). The formulas are
standard textbook results. The circular-wing value is the widely quoted
result of Kinner's exact solution; check it against the paper before quoting
it in a report.
"""

from __future__ import annotations

import numpy as np

REFERENCES = {
    "prandtl_elliptic": (
        "F. W. Lanchester, 'Aerodynamics', Constable, London, 1907; "
        "L. Prandtl, 'Tragflugeltheorie I/II', Nachrichten der Gesellschaft der Wissenschaften "
        "zu Goettingen, 1918-1919. Elliptic loading: CL_alpha = a0 / (1 + a0 / (pi AR)), e = 1."
    ),
    "glauert_monoplane": (
        "H. Glauert, 'The Elements of Aerofoil and Airscrew Theory', Cambridge University Press, "
        "1926, chapter XI (Fourier solution of the monoplane equation)."
    ),
    "helmbold": (
        "H. B. Helmbold, 'Der unverwundene Ellipsenfluegel als tragende Flaeche', Jahrbuch 1942 der "
        "Deutschen Luftfahrtforschung, pp. I 111-113. CL_alpha = a0 AR / (a0/pi + sqrt((a0/pi)^2 + AR^2)) "
        "(approximate lifting-surface result for elliptic wings)."
    ),
    "kinner_circular": (
        "W. Kinner, 'Die kreisfoermige Tragflaeche auf potentialtheoretischer Grundlage', "
        "Ingenieur-Archiv 8, 1937, pp. 47-80; see also P. F. Jordan, 'Exact solutions for lifting "
        "surfaces', AIAA Journal 11(8), 1973. Flat circular wing: CL_alpha = 1.790 per rad "
        "(value not re-read from the source in this session)."
    ),
}


def elliptic_wing_cl_alpha(AR: float, a0: float = 2.0 * np.pi) -> float:
    """Lifting-line lift slope of an elliptic wing [1/rad] (Lanchester-Prandtl theory)."""
    return a0 / (1.0 + a0 / (np.pi * AR))


def helmbold_cl_alpha(AR: float, a0: float = 2.0 * np.pi) -> float:
    """Helmbold's approximate lifting-surface lift slope of an elliptic wing [1/rad]."""
    k = a0 / np.pi
    return a0 * AR / (k + np.sqrt(k * k + AR * AR))


def circular_wing_cl_alpha() -> float:
    """Flat circular wing (AR = 4/pi), exact lifting-surface theory (Kinner) [1/rad]."""
    return 1.790


def glauert_monoplane(
    AR: float,
    taper: float = 1.0,
    a0: float = 2.0 * np.pi,
    n_terms: int = 60,
) -> tuple[float, float]:
    """Independent solution of the monoplane equation for a straight tapered wing.

    Symmetric loading (odd Fourier terms), collocation at
    ``theta_k = k pi / (2 N)``, ``k = 1..N``. The chord is linear from root to
    tip: ``c = c_r (1 - (1 - taper) |2y/b|)``.

    Returns
    -------
    (CL_alpha [1/rad], e)
    """
    b = 1.0
    c_r = 2.0 * b / (AR * (1.0 + taper))
    k = np.arange(1, n_terms + 1)
    theta = k * np.pi / (2 * n_terms)
    n = 2 * np.arange(n_terms) + 1
    c = c_r * (1.0 - (1.0 - taper) * np.cos(theta))
    mu = c * a0 / (4.0 * b)
    M = np.sin(np.outer(theta, n)) * (1.0 + np.outer(mu / np.sin(theta), n))
    A = np.linalg.solve(M, mu)  # alpha = 1 rad
    cl_alpha = np.pi * AR * A[0]
    delta = float(np.sum(n[1:] * A[1:] ** 2) / A[0] ** 2)
    return float(cl_alpha), 1.0 / (1.0 + delta)


def wing_sections_elliptic(
    root_chord: float,
    n_sections: int = 301,
    straight_line: str = "quarter_chord",
    tip_chord_fraction: float = 1.0e-3,
):
    """Sections of an elliptic planform.

    Parameters
    ----------
    root_chord : float
        Root chord [m]. The span comes from the surface that uses the sections.
    straight_line : {"quarter_chord", "leading_edge", "mid_chord"}
        Which chordwise line is straight. Classical lifting-line theory
        assumes a straight quarter-chord line; a straight leading edge gives
        a curved (locally swept) quarter-chord line.
    tip_chord_fraction : float
        The tip chord is this fraction of the root chord (it must be > 0).
    """
    from ventorum.core.datatypes import WingSection

    frac = {"quarter_chord": 0.25, "leading_edge": 0.0, "mid_chord": 0.5}[straight_line]
    secs = []
    for eta in np.linspace(0.0, 1.0, n_sections):
        c = float(max(root_chord * np.sqrt(max(1.0 - eta * eta, 0.0)), tip_chord_fraction * root_chord))
        secs.append(WingSection(y_frac=float(eta), chord=c, x_le=-frac * c))
    return secs
