# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""OpenMDAO components of Ventorum (optional extra ``mdo``).

:class:`VentorumComponent` wraps one Ventorum analysis in an OpenMDAO
``ExplicitComponent`` with analytic partials from
:func:`ventorum.gradients`. :class:`VentorumMatrixFreeComponent` also gives
the vector outputs ``strip_force`` and ``section_cl``, with matrix-free
products from :func:`ventorum.gradient_vjp`.

The components let Ventorum run inside OpenMDAO optimisation and coupled
models. OpenMDAO (Gray et al. 2019) assembles the total derivatives of a
model from the partial derivatives of its components; a component that
gives its partials from an adjoint solve keeps the cost of the coupled
derivatives independent of the number of inputs (Martins and Hwang 2013).

This module imports OpenMDAO. ``import ventorum`` does not import this
module, so Ventorum works without OpenMDAO. Install the extra with
``pip install "ventorum[mdo]"``.

References
----------
J. S. Gray, J. T. Hwang, J. R. R. A. Martins, K. T. Moore and B. A. Naylor,
"OpenMDAO: an open-source framework for multidisciplinary design, analysis,
and optimization", Structural and Multidisciplinary Optimization 59,
1075-1104, 2019.

J. R. R. A. Martins and J. T. Hwang, "Review and unification of methods for
computing derivatives of multidisciplinary computational models", AIAA
Journal 51(11), 2582-2599, 2013.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any

import numpy as np

try:
    import openmdao.api as om
except ImportError as exc:  # pragma: no cover - the test hides OpenMDAO to reach this branch
    raise ImportError(
        'ventorum.mdo needs OpenMDAO, which is not installed. Install the optional extra: '
        'pip install "ventorum[mdo]" (openmdao>=3.30,<4).'
    ) from exc

from ventorum.core.datatypes import (
    Aircraft,
    FlightCondition,
    LiftingSurface,
    NodeDisplacements,
    SolverSettings,
)

__all__ = ["VentorumComponent", "VentorumMatrixFreeComponent"]

#: Scalar outputs (dimensionless coefficients).
SCALAR_OUTPUTS = ("CL", "CY", "CDi", "CDp", "CD", "Cl", "Cm", "Cn")
#: Vector outputs of the matrix-free component, with their OpenMDAO units.
VECTOR_OUTPUTS = {"strip_force": "N", "section_cl": None}
#: Scalar inputs and their OpenMDAO units.
SCALAR_INPUT_UNITS = {
    "alpha": "rad",
    "beta": "rad",
    "V_inf": "m/s",
    "rho": "kg/m**3",
    "h": "m",
    "phi": "rad",
    "p": "rad/s",
    "q": "rad/s",
    "r": "rad/s",
}
#: Input groups that the components accept.
INPUT_GROUPS = tuple(SCALAR_INPUT_UNITS) + ("nodes", "controls")
#: Input groups of ``vt.gradients`` that the components refuse, with the reason.
_REFUSED_INPUTS = {
    "sections": "the section properties are per strip of the lattice, and the geometry has no "
                "per-strip field to set them",
    "reference": "the components keep the reference values of the geometry",
    "planform": "the planform inputs are not part of the components yet",
}
_NAME_RE = re.compile(r"[^A-Za-z0-9_]")


def _om_name(name: str) -> str:
    """Return *name* with each character other than a letter, a digit or ``_`` changed to ``_``."""
    return _NAME_RE.sub("_", str(name))


class _VentorumBase(om.ExplicitComponent):
    """Common setup and analysis of the Ventorum components."""

    _vt_vector_outputs_allowed = False

    def __init__(self, geometry: Any = None, settings: Any = None, **kwargs: Any) -> None:
        if geometry is not None:
            kwargs["geometry"] = geometry
        if settings is not None:
            kwargs["settings"] = settings
        super().__init__(**kwargs)
        self.gradient_result: Any = None
        self.last_result: Any = None
        self.input_names: dict[str, tuple] = {}

    def initialize(self) -> None:
        """Declare the options of the component."""
        self.options.declare("geometry", types=(Aircraft, LiftingSurface), recordable=False,
                             desc="Aircraft or LiftingSurface. The component does not change it.")
        self.options.declare("settings", default=None, types=(SolverSettings,), allow_none=True,
                             recordable=False, desc="Solver settings (vlm, linear or nonlinear).")
        self.options.declare("condition", default=None, types=(FlightCondition,), allow_none=True,
                             recordable=False, desc="Base flight condition. Inputs replace its values.")
        self.options.declare("outputs", default=("CL", "CD", "Cm"), types=(tuple, list),
                             desc="Output names.")
        self.options.declare("inputs", default=("alpha", "nodes"), types=(tuple, list),
                             desc="Input groups, named as in ventorum.gradients.")
        self.options.declare("axes", default="body", types=str, desc="Axes of the moments.")
        self.options.declare("method", default="auto", types=str,
                             desc="Path of the partial derivatives (see ventorum.gradients).")

    # --- Setup ----------------------------------------------------------------------------------

    def _vt_check_options(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        outs = tuple(self.options["outputs"])
        groups = tuple(self.options["inputs"])
        if not outs:
            raise ValueError("No output requested.")
        if not groups:
            raise ValueError("No input requested.")
        allowed = SCALAR_OUTPUTS + (tuple(VECTOR_OUTPUTS) if self._vt_vector_outputs_allowed else ())
        for o in outs:
            if o in VECTOR_OUTPUTS and not self._vt_vector_outputs_allowed:
                raise ValueError(
                    f"Output {o!r} is a vector output; use VentorumMatrixFreeComponent, which gives the "
                    "products of ventorum.gradient_vjp.")
            if o not in allowed:
                raise ValueError(f"Unknown output {o!r}; use one of {list(allowed)}.")
        for g in groups:
            if g in _REFUSED_INPUTS:
                raise ValueError(f"Input {g!r} is not available in ventorum.mdo: {_REFUSED_INPUTS[g]}.")
            if g not in INPUT_GROUPS:
                raise ValueError(f"Unknown input {g!r}; use one of {list(INPUT_GROUPS)}.")
        if len(set(outs)) != len(outs) or len(set(groups)) != len(groups):
            raise ValueError("Output and input names must be unique.")
        if self.options["axes"] not in ("body", "stability", "wind"):
            raise ValueError(f"'axes' must be one of ['body', 'stability', 'wind'], got {self.options['axes']!r}.")
        return outs, groups

    def setup(self) -> None:
        """Add the inputs and the outputs from the options."""
        from ventorum.geometry import undeformed_nodes

        outs, groups = self._vt_check_options()
        geometry = self.options["geometry"]
        ac = Aircraft(name="SingleWing", surfaces=[geometry]) if isinstance(geometry, LiftingSurface) else geometry
        self._vt_aircraft = ac.clone()
        settings = self.options["settings"]
        self._vt_settings = settings.clone() if settings is not None else SolverSettings()
        condition = self.options["condition"]
        self._vt_condition = dataclasses.replace(condition) if condition is not None else FlightCondition()
        self._vt_outputs = outs
        self._vt_groups = groups
        self.input_names = {}
        self._vt_symmetric: set[str] = set()

        for g in groups:
            if g in SCALAR_INPUT_UNITS:
                value = getattr(self._vt_condition, g)
                if g == "h" and value is None:
                    raise ValueError(
                        'Input "h" needs a height: set condition.h [m] (ground effect) in the base condition.')
                self._vt_add(g, ("scalar", g), float(value), SCALAR_INPUT_UNITS[g])
            elif g == "nodes":
                nodes = undeformed_nodes(self._vt_aircraft, self._vt_settings, solver=self._vt_settings.solver_type)
                for surf in self._vt_aircraft.surfaces:
                    n_e = nodes[surf.name]["le"].shape[0]
                    d = getattr(surf, "node_displacements", None)
                    if d is not None and surf.is_symmetric and (d.le[0, 1] != 0.0 or d.te[0, 1] != 0.0):
                        raise ValueError(
                            f"[{surf.name}] is symmetric, but its node_displacements move the root to "
                            f"y = {float(d.le[0, 1]):.4g} m (le) and {float(d.te[0, 1]):.4g} m (te). The root of a "
                            "symmetric surface stays on y = 0; set row 0, column 1 of le and te to 0.")
                    for part in ("le", "te"):
                        val = np.zeros((n_e, 3)) if d is None else np.asarray(getattr(d, part), dtype=float).copy()
                        self._vt_add(f"nodes:{_om_name(surf.name)}:{part}", ("nodes", surf.name, part), val, "m")
                self._vt_eta = {s.name: nodes[s.name]["eta"] for s in self._vt_aircraft.surfaces}
                self._vt_symmetric = {s.name for s in self._vt_aircraft.surfaces if s.is_symmetric}
            elif g == "controls":
                names = sorted({c.name for s in self._vt_aircraft.surfaces for c in getattr(s, "controls", [])})
                if not names:
                    raise ValueError('Input "controls" needs an aircraft with one or more control surfaces.')
                for name in names:
                    defl = {float(c.deflection) for s in self._vt_aircraft.surfaces
                            for c in getattr(s, "controls", []) if c.name == name}
                    if len(defl) != 1:
                        raise ValueError(
                            f"The controls named {name!r} have different deflections {sorted(defl)} rad. The input "
                            "moves all controls with one name together; give them one deflection.")
                    self._vt_add(f"controls:{_om_name(name)}", ("controls", name), defl.pop(), "rad")

        vector_shapes: dict[str, tuple[int, ...]] = {}
        if any(o in VECTOR_OUTPUTS for o in outs):
            vec = self._vt_vectors(self._vt_analyze(self._vt_aircraft, self._vt_condition))
            vector_shapes = {o: vec[o].shape for o in outs if o in VECTOR_OUTPUTS}
        self._vt_vector_shapes = vector_shapes
        for o in outs:
            if o in VECTOR_OUTPUTS:
                self.add_output(o, val=np.zeros(vector_shapes[o]), units=VECTOR_OUTPUTS[o])
            else:
                self.add_output(o, val=0.0)

    def _vt_add(self, om_name: str, key: tuple, val: Any, units: str) -> None:
        if om_name in self.input_names:
            raise ValueError(
                f"Two inputs have the OpenMDAO name {om_name!r} after the change of the characters that "
                "OpenMDAO does not accept. Give the surfaces and the controls unique names of letters, "
                "digits and '_'.")
        self.input_names[om_name] = key
        self.add_input(om_name, val=val, units=units)

    # --- Analysis -------------------------------------------------------------------------------

    def _vt_case(self, inputs: Any, warn: bool = False) -> tuple[Aircraft, FlightCondition]:
        """Return the aircraft and the condition with the input values.

        With *warn*, issue an OpenMDAO warning when an input moves the root
        of a symmetric surface off y = 0 (the component uses 0 there).
        """
        ac = self._vt_aircraft.clone()
        changes: dict[str, float] = {}
        node_parts: dict[str, dict[str, np.ndarray]] = {}
        control_values: dict[str, float] = {}
        for om_name, key in self.input_names.items():
            value = np.asarray(inputs[om_name], dtype=float)
            if key[0] == "scalar":
                changes[key[1]] = float(value.reshape(-1)[0])
            elif key[0] == "nodes":
                node_parts.setdefault(key[1], {})[key[2]] = value.reshape(-1, 3).copy()
            else:
                control_values[key[1]] = float(value.reshape(-1)[0])
        for surf in ac.surfaces:
            if surf.name in node_parts:
                parts = node_parts[surf.name]
                if surf.is_symmetric:
                    # The root of a symmetric surface stays on the plane y = 0: the
                    # lattice refuses other roots. The component holds this value at
                    # 0, so the output does not depend on the input there and the
                    # partial is 0 (see _vt_flat). vt.gradients gives a small nonzero
                    # derivative for this entry; the component does not use it.
                    root_y = (float(parts["le"][0, 1]), float(parts["te"][0, 1]))
                    if warn and any(v != 0.0 for v in root_y):
                        om.issue_warning(
                            f"The root y-displacement of the symmetric surface {surf.name!r} is "
                            f"{root_y} m (le, te). The root stays on y = 0: the component uses 0 "
                            "there, and the partials of that entry are 0.", prefix=self.pathname)
                    parts["le"][0, 1] = 0.0
                    parts["te"][0, 1] = 0.0
                surf.node_displacements = NodeDisplacements(le=parts["le"], te=parts["te"],
                                                            eta=self._vt_eta[surf.name].copy())
            for c in getattr(surf, "controls", []):
                if c.name in control_values:
                    c.deflection = control_values[c.name]
        condition = dataclasses.replace(self._vt_condition, **changes)
        return ac, condition

    def _vt_analyze(self, ac: Aircraft, condition: FlightCondition) -> Any:
        import ventorum as vt

        return vt.analyze(ac, condition, self._vt_settings)

    def _vt_scalars(self, result: Any) -> dict[str, float]:
        """Return the scalar outputs of *result* in the axes of the component."""
        t = result.totals
        m = result.moments(self.options["axes"])
        return {
            "CL": float(t.CL),
            "CY": float(t.CY),
            "CDi": float(t.CDi),
            "CDp": float(t.CDp if t.CDp is not None else 0.0),
            "CD": float(t.CD_total if t.CD_total is not None else t.CDi),
            "Cl": float(m["Cl"]),
            "Cm": float(m["Cm"]),
            "Cn": float(m["Cn"]),
        }

    @staticmethod
    def _vt_vectors(result: Any) -> dict[str, np.ndarray]:
        """Return the vector outputs of *result*: strip forces [N] and section lift coefficients."""
        loads = result.details["loads"]
        return {
            "strip_force": np.asarray(loads.strip_force, dtype=float),
            "section_cl": np.concatenate([np.asarray(sp.Cl, dtype=float) for sp in loads.spanwise]),
        }

    def compute(self, inputs: Any, outputs: Any) -> None:
        """Run one Ventorum analysis with the input values.

        Parameters
        ----------
        inputs : Vector
            Input values: angles [rad], speed [m/s], density [kg/m^3],
            height [m], rates [rad/s], node displacements [m] and control
            deflections [rad].
        outputs : Vector
            Output values: coefficients (dimensionless), strip forces [N].
        """
        ac, condition = self._vt_case(inputs, warn=True)
        result = self._vt_analyze(ac, condition)
        self.last_result = result
        scalars = self._vt_scalars(result)
        vectors = self._vt_vectors(result) if any(o in VECTOR_OUTPUTS for o in self._vt_outputs) else {}
        for o in self._vt_outputs:
            outputs[o] = vectors[o] if o in VECTOR_OUTPUTS else scalars[o]

    def _vt_flat(self, gradient: dict, om_name: str) -> np.ndarray:
        """Return the gradient of one OpenMDAO input as a flat array."""
        key = self.input_names[om_name]
        if key[0] == "scalar":
            return np.array([float(gradient[key[1]])])
        if key[0] == "nodes":
            g = np.array(gradient["nodes"][key[1]][key[2]], dtype=float)
            if key[1] in self._vt_symmetric:
                # The component holds the root y-displacement of a symmetric surface at
                # 0 (see _vt_case), so the output does not depend on this input entry.
                g[0, 1] = 0.0
            return g.reshape(-1)
        return np.array([float(gradient["controls"][key[1]])])


class VentorumComponent(_VentorumBase):
    """OpenMDAO explicit component of one Ventorum analysis, with analytic partials.

    ``compute`` runs :func:`ventorum.analyze` once. ``compute_partials``
    runs :func:`ventorum.gradients` once (one production solve and one
    adjoint solve per output) and gives the partials of every output with
    respect to every input. The device and the precision follow the global
    settings of :mod:`ventorum.gpu`; the component does not change them.

    Parameters
    ----------
    geometry : Aircraft or LiftingSurface
        Geometry. The component keeps a copy and does not change it. The
        mesh topology is fixed.
    settings : SolverSettings or None
        Solver settings: ``"vlm"``, ``"linear"`` or ``"nonlinear"``.
    condition : FlightCondition or None
        Base flight condition (keyword). The inputs replace its values.
        None gives ``FlightCondition()``.
    outputs : tuple of str
        Scalar outputs from ``"CL", "CY", "CDi", "CDp", "CD", "Cl", "Cm",
        "Cn"`` (dimensionless). Moments in the axes *axes*.
    inputs : tuple of str
        Input groups from ``"alpha", "beta"`` [rad], ``"V_inf"`` [m/s],
        ``"rho"`` [kg/m^3], ``"h"`` [m] (needs ``condition.h``), ``"phi"``
        [rad], ``"p", "q", "r"`` [rad/s], ``"nodes"`` and ``"controls"``.
        ``"nodes"`` adds per surface the inputs ``nodes:<surface>:le`` and
        ``nodes:<surface>:te``: the node displacements [m] of the
        defining half, shape (n_edges, 3), as
        :class:`ventorum.NodeDisplacements`. On a symmetric surface the
        root stays on the plane y = 0: the component uses 0 for the
        y-displacement of the root nodes (row 0, column 1), so the outputs
        do not depend on that entry and its partials are 0. A nonzero input
        value there gives an OpenMDAO warning; a nonzero value in the
        geometry raises ``ValueError`` at setup. ``"controls"`` adds per
        control name the input ``controls:<name>``: the deflection [rad]
        of all controls with that name. A character other than a letter, a
        digit or ``_`` in a surface or control name becomes ``_``.
    axes : str
        Axes of the moments: ``"body"``, ``"stability"`` or ``"wind"``.
    method : str
        Path of the partial derivatives, as in :func:`ventorum.gradients`.

    Attributes
    ----------
    input_names : dict
        OpenMDAO input name to its Ventorum key.
    last_result : SolverResult or None
        Result of the last ``compute``.
    gradient_result : GradientResult or None
        Result of the last ``compute_partials`` (notes, device, precision).

    Notes
    -----
    The partials are exact derivatives of the discrete solver by the
    discrete adjoint (Martins and Hwang 2013). OpenMDAO (Gray et al. 2019)
    combines them into total derivatives. In float64 they agree with
    central differences to about 1e-6 relative or better; in float32 (the
    default precision on the GPU) to about 1e-3.
    """

    def setup(self) -> None:
        """Add the inputs and the outputs, and declare the dense partials."""
        super().setup()
        for o in self._vt_outputs:
            for name in self.input_names:
                self.declare_partials(o, name)

    def compute_partials(self, inputs: Any, partials: Any) -> None:
        """Give the partials of the outputs from one adjoint computation.

        Parameters
        ----------
        inputs : Vector
            Input values (units as in :meth:`compute`).
        partials : Jacobian
            Partial derivatives of each output with respect to each input,
            per input unit (1/rad, s/m, m^3/kg, 1/m).
        """
        import ventorum as vt

        ac, condition = self._vt_case(inputs)
        res = vt.gradients(ac, condition, self._vt_settings, outputs=self._vt_outputs, wrt=self._vt_groups,
                           axes=self.options["axes"], method=self.options["method"])
        self.gradient_result = res
        for o in self._vt_outputs:
            for name in self.input_names:
                partials[o, name] = self._vt_flat(res.jacobian[o], name).reshape(1, -1)


class VentorumMatrixFreeComponent(_VentorumBase):
    """OpenMDAO explicit component of one Ventorum analysis, with matrix-free products.

    This component also gives the vector outputs ``strip_force`` (strip
    forces [N], shape (n_strips, 3), geometry axes) and ``section_cl``
    (section lift coefficients, shape (n_strips,)). The reverse product
    runs :func:`ventorum.gradient_vjp` once with the output seeds (one
    production solve and one adjoint solve). Use it with the reverse mode
    of OpenMDAO (``prob.setup(mode="rev")``).

    The forward product needs the full Jacobian: one adjoint solve per
    scalar output and one per component of each vector output. The
    component keeps that Jacobian for the last input values. The forward
    mode is correct but slow; use the reverse mode.

    Parameters
    ----------
    geometry : Aircraft or LiftingSurface
        Geometry, as in :class:`VentorumComponent`.
    settings : SolverSettings or None
        Solver settings.
    condition : FlightCondition or None
        Base flight condition (keyword).
    outputs : tuple of str
        Scalar outputs and ``"strip_force"``, ``"section_cl"``.
    inputs : tuple of str
        Input groups, as in :class:`VentorumComponent`.
    axes : str
        Axes of the moments.
    method : str
        Path of the partial derivatives, as in :func:`ventorum.gradients`.
    """

    _vt_vector_outputs_allowed = True

    def setup(self) -> None:
        """Add the inputs and the outputs."""
        super().setup()
        self._vt_dense_key: bytes | None = None
        self._vt_dense: dict[str, dict[str, np.ndarray]] = {}

    def _vt_key(self, inputs: Any) -> bytes:
        return b"".join(np.asarray(inputs[n], dtype=float).tobytes() for n in self.input_names)

    def _vt_vjp(self, ac: Aircraft, condition: FlightCondition, seeds: dict) -> dict:
        import ventorum as vt

        return vt.gradient_vjp(ac, condition, self._vt_settings, seeds=seeds, wrt=self._vt_groups,
                               axes=self.options["axes"], method=self.options["method"])

    def _vt_dense_jacobian(self, inputs: Any) -> dict[str, dict[str, np.ndarray]]:
        """Return the full Jacobian {output: {input: (n_out, n_in)}} at the input values."""
        key = self._vt_key(inputs)
        if key == self._vt_dense_key:
            return self._vt_dense
        import ventorum as vt

        ac, condition = self._vt_case(inputs)
        dense: dict[str, dict[str, np.ndarray]] = {}
        scalars = tuple(o for o in self._vt_outputs if o not in VECTOR_OUTPUTS)
        if scalars:
            res = vt.gradients(ac, condition, self._vt_settings, outputs=scalars, wrt=self._vt_groups,
                               axes=self.options["axes"], method=self.options["method"])
            self.gradient_result = res
            for o in scalars:
                dense[o] = {n: self._vt_flat(res.jacobian[o], n).reshape(1, -1) for n in self.input_names}
        for o in self._vt_outputs:
            if o not in VECTOR_OUTPUTS:
                continue
            shape = self._vt_vector_shapes[o]
            size = int(np.prod(shape))
            rows: dict[str, list[np.ndarray]] = {n: [] for n in self.input_names}
            for k in range(size):
                seed = np.zeros(size)
                seed[k] = 1.0
                g = self._vt_vjp(ac, condition, {o: seed.reshape(shape)})
                for n in self.input_names:
                    rows[n].append(self._vt_flat(g, n))
            dense[o] = {n: np.vstack(rows[n]) for n in self.input_names}
        self._vt_dense_key, self._vt_dense = key, dense
        return dense

    def compute_jacvec_product(self, inputs: Any, d_inputs: Any, d_outputs: Any, mode: str) -> None:
        """Give the product of the Jacobian with a vector.

        Parameters
        ----------
        inputs : Vector
            Input values.
        d_inputs : Vector
            Input direction (``"fwd"``) or result of the transposed product (``"rev"``).
        d_outputs : Vector
            Result of the product (``"fwd"``) or output seed (``"rev"``).
        mode : str
            ``"fwd"`` or ``"rev"``.
        """
        names = [n for n in self.input_names if n in d_inputs]
        if not names:
            return
        if mode == "rev":
            if self._vt_key(inputs) == self._vt_dense_key:
                for o in self._vt_outputs:
                    if o not in d_outputs:
                        continue
                    w = np.asarray(d_outputs[o], dtype=float).reshape(-1)
                    for n in names:
                        d_inputs[n] += (w @ self._vt_dense[o][n]).reshape(d_inputs[n].shape)
                return
            seeds = {}
            for o in self._vt_outputs:
                if o not in d_outputs:
                    continue
                w = np.asarray(d_outputs[o], dtype=float)
                if np.any(w != 0.0):
                    seeds[o] = float(w.reshape(-1)[0]) if o not in VECTOR_OUTPUTS else w.copy()
            if not seeds:
                return
            ac, condition = self._vt_case(inputs)
            g = self._vt_vjp(ac, condition, seeds)
            for n in names:
                d_inputs[n] += self._vt_flat(g, n).reshape(d_inputs[n].shape)
            return
        dense = self._vt_dense_jacobian(inputs)
        for o in self._vt_outputs:
            if o not in d_outputs:
                continue
            total = np.zeros(int(np.prod(d_outputs[o].shape)) if np.ndim(d_outputs[o]) else 1)
            for n in names:
                total = total + dense[o][n] @ np.asarray(d_inputs[n], dtype=float).reshape(-1)
            d_outputs[o] += total.reshape(np.shape(d_outputs[o]))
