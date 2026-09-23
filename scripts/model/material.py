"""OpenSees BRB uniaxial SteelMPF material for the corotational truss."""

from __future__ import annotations

import openseespy.opensees as ops

BRB_MAT_TAG = 1000


def ops_BRB_material(
    *,
    fyp: float = 40.0,
    fyn: float = 40.0,
    E: float = 29000.0,
    b_p: float = 0.02,
    b_n: float = 0.02,
    R0: float = 20.0,
    cR1: float = 0.925,
    cR2: float = 0.15,
    a1: float = 0.0,
    a2: float = 1.0,
    a3: float = 0.0,
    a4: float = 1.0,
) -> int:
    """Define uniaxial SteelMPF for the BRB truss; returns the material tag."""
    ops.uniaxialMaterial(
        "SteelMPF",
        BRB_MAT_TAG,
        fyp,
        fyn,
        E,
        b_p,
        b_n,
        R0,
        cR1,
        cR2,
        a1,
        a2,
        a3,
        a4,
    )
    return BRB_MAT_TAG


STEEL_MPF_TAG = BRB_MAT_TAG


if __name__ == "__main__":
    ops_BRB_material()
