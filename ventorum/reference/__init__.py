# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Reference solutions (closed forms and independent implementations) for verification."""

from ventorum.reference.analytic import (  # noqa: F401
    REFERENCES,
    circular_wing_cl_alpha,
    elliptic_wing_cl_alpha,
    glauert_monoplane,
    helmbold_cl_alpha,
    wing_sections_elliptic,
)
from ventorum.reference.database import (  # noqa: F401
    GRADES,
    MACH_LIMIT,
    MOMENT_QUANTITIES,
    QUANTITIES,
    SOURCE_TYPES,
    ReferenceCase,
    ReferenceError,
    ResultEntry,
    load_case,
    load_database,
)
