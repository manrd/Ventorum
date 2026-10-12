# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Example 12: Twist optimisation in OpenMDAO with adjoint gradients.

Minimises the drag coefficient CD at a fixed lift coefficient CL. The
design variables are the twist angles at four spanwise control stations.
A small component interpolates the twist to the node stations and turns it
into node displacements; the Ventorum component gives CL and CD with
analytic partials from ``ventorum.gradients``.

This example needs the optional extra ``mdo`` (``pip install "ventorum[mdo]"``).
Without OpenMDAO it prints a message and stops.
"""

from __future__ import annotations

import numpy as np

import ventorum as vt


def main() -> dict:
    """Run the optimisation and return the results.

    Returns
    -------
    dict
        ``success``, ``iterations``, ``CL``, ``CL_target``, ``CD``,
        ``CD_initial`` and ``twist_deg`` (twist at the control stations [deg]).
    """
    try:
        import openmdao.api as om

        from ventorum.mdo import VentorumComponent
    except ImportError as exc:
        print(f"Example skipped: {exc}")
        return {}

    foil = vt.LinearAirfoil(name="Thin", a0=2.0 * np.pi, alpha_L0=0.0, Cd0=0.008)
    wing = vt.LiftingSurface(
        name="Wing",
        semi_span=5.0,
        sections=[
            vt.WingSection(y_frac=0.0, chord=2.0, airfoil=foil),
            vt.WingSection(y_frac=1.0, chord=0.6, airfoil=foil),
        ],
    )
    aircraft = vt.Aircraft(surfaces=[wing])
    condition = vt.FlightCondition(V_inf=30.0, alpha=np.radians(5.0))
    settings = vt.SolverSettings(solver_type="vlm", n_panels=12, n_chord=2)
    cl_target = 0.4

    # Node stations of the mesh: twist about the quarter chord moves the
    # leading edge up by 0.25 c theta and the trailing edge down by
    # 0.75 c theta (small angles, c the local chord [m]).
    nodes = vt.geometry.undeformed_nodes(aircraft, settings)["Wing"]
    chord = nodes["te"][:, 0] - nodes["le"][:, 0]
    n_e = chord.size
    # Linear interpolation from the control stations to the node stations.
    # Few control stations keep the twist smooth: a free twist per node
    # station lets the optimiser find a saw-tooth that only the mesh sees.
    eta_cp = np.linspace(0.0, 1.0, 4)
    interp = np.column_stack([np.interp(nodes["eta"], eta_cp, np.eye(eta_cp.size)[k])
                              for k in range(eta_cp.size)])

    class TwistToNodes(om.ExplicitComponent):
        """Node displacements [m] of the twist [rad] at the control stations, about the quarter chord."""

        def setup(self):
            self.add_input("twist", val=np.zeros(eta_cp.size), units="rad")
            self.add_output("nodes:Wing:le", val=np.zeros((n_e, 3)), units="m")
            self.add_output("nodes:Wing:te", val=np.zeros((n_e, 3)), units="m")
            d_le = np.zeros((n_e, 3, eta_cp.size))
            d_le[:, 2, :] = 0.25 * chord[:, None] * interp
            self.declare_partials("nodes:Wing:le", "twist", val=d_le.reshape(3 * n_e, -1))
            self.declare_partials("nodes:Wing:te", "twist", val=-3.0 * d_le.reshape(3 * n_e, -1))

        def compute(self, inputs, outputs):
            theta = interp @ inputs["twist"]
            le = np.zeros((n_e, 3))
            te = np.zeros((n_e, 3))
            le[:, 2] = 0.25 * chord * theta
            te[:, 2] = -0.75 * chord * theta
            outputs["nodes:Wing:le"] = le
            outputs["nodes:Wing:te"] = te

    prob = om.Problem(reports=False)
    prob.model.add_subsystem("twist", TwistToNodes(), promotes=["*"])
    prob.model.add_subsystem(
        "aero",
        VentorumComponent(aircraft, settings, condition=condition, outputs=("CL", "CD"), inputs=("nodes",)),
        promotes=["*"],
    )
    prob.model.add_design_var("twist", lower=-0.2, upper=0.2)
    prob.model.add_objective("CD", scaler=100.0)
    prob.model.add_constraint("CL", equals=cl_target)
    prob.driver = om.ScipyOptimizeDriver(optimizer="SLSQP", tol=1e-10, maxiter=100, disp=False)
    prob.setup(mode="rev")

    prob.run_model()
    cd_initial = float(prob.get_val("CD")[0])
    cl_initial = float(prob.get_val("CL")[0])
    result = prob.run_driver()

    out = {
        "success": bool(result.success),
        "iterations": int(result.iter_count),
        "CL": float(prob.get_val("CL")[0]),
        "CL_target": cl_target,
        "CD": float(prob.get_val("CD")[0]),
        "CD_initial": cd_initial,
        "twist_deg": np.degrees(prob.get_val("twist")),
    }
    print(f"Initial: CL = {cl_initial:.5f}, CD = {cd_initial:.6f}")
    print(f"Optimum: CL = {out['CL']:.5f}, CD = {out['CD']:.6f} "
          f"(success {out['success']}, {out['iterations']} iterations)")
    print("Twist at the control stations [deg], root to tip:", np.array2string(out["twist_deg"], precision=2))
    return out


if __name__ == "__main__":
    main()
