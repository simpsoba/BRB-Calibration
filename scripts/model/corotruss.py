"""Corotational truss (BRB) model using OpenSees SteelMPF.

Uses E_hat = Q*E with Q from BRB geometry.
"""

from __future__ import annotations

import numpy as np

import openseespy.opensees as ops

from .brace_geometry import compute_Q
from .material import ops_BRB_material


def run_simulation(
    displacement: np.ndarray,
    *,
    L_T: float,
    L_y: float,
    A_sc: float,
    A_t: float,
    fyp: float,
    fyn: float,
    E: float,
    b_p: float,
    b_n: float,
    R0: float = 20.0,
    cR1: float = 0.925,
    cR2: float = 0.15,
    a1: float = 0.0,
    a2: float = 1.0,
    a3: float = 0.0,
    a4: float = 1.0,
    steel_model: str = "steelmpf",  # accepted for CSV compatibility; ignored
) -> np.ndarray:
    """
    Run the BRB corotruss simulation and return force history for the given displacement history.

    Young's modulus is adjusted with E_hat = Q*E, where
    Q = L_T / (L_y + (L_T - L_y) * A_sc / A_t).
    """
    del steel_model  # SteelMPF only
    displacement = np.asarray(displacement, dtype=float)
    n = len(displacement)

    Q = compute_Q(L_T, L_y, A_sc, A_t)
    E_hat = Q * E

    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 2)

    ops.node(1, 0.0, 0.0)
    ops.fix(1, 1, 1)
    ops.node(2, L_T, 0.0)
    ops.fix(2, 0, 1)

    brb_mat_tag = ops_BRB_material(
        fyp=fyp,
        fyn=fyn,
        E=E_hat,
        b_p=b_p,
        b_n=b_n,
        R0=R0,
        cR1=cR1,
        cR2=cR2,
        a1=a1,
        a2=a2,
        a3=a3,
        a4=a4,
    )

    ops.element("corotTruss", 1, 1, 2, A_sc, brb_mat_tag)

    # Path series: uniform dt=1 => times 0..n; length n+1 with zero initial disp.
    dt = 1.0
    path_values = np.empty(n + 1, dtype=np.float64)
    path_values[0] = 0.0
    path_values[1:] = displacement
    ops.timeSeries("Path", 1, "-dt", dt, "-values", *path_values, "-useLast")
    ops.pattern("Plain", 1, 1)
    ops.sp(2, 1, 1.0)

    ops.integrator("LoadControl", dt)
    ops.constraints("Transformation")
    ops.numberer("Plain")
    ops.system("UmfPack")
    ops.analysis("Static", "-noWarnings")

    force = np.zeros(n)
    for i in range(n):
        ok = ops.analyze(1)
        if ok != 0:
            raise RuntimeError(
                f"OpenSees analyze failed at step {i + 1}/{n}. "
                "Try smaller displacement increments or check material/geometry."
            )
        axial_force = ops.eleResponse(1, "axialForce")
        force[i] = axial_force[0] if isinstance(axial_force, (list, tuple)) else axial_force

    return force
