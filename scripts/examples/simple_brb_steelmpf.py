# Simple BRB model with displacement-controlled analysis (SteelMPF).
#
# Edit the blocks marked EDIT below, then run from the repository root:
#   python scripts/examples/simple_brb_steelmpf.py
#
# Geometry / material defaults match specimen STF01. Replace the calibrated
# SteelMPF numbers with values from a calibration run
# (e.g. results/calibration/single_specimen/STF01/parameters.csv).
#
# Naming matches the rest of this repo: L_T, L_y, A_sc, A_t, Q, E_hat = Q*E0.

### import needed libraries ###
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import openseespy.opensees as ops

### define units ###
kips = 1.0
inch = 1.0

ksi = kips / inch**2

### paths ###
REPO_ROOT = Path(__file__).resolve().parents[2]
# EDIT: experimental F–u CSV (columns Deformation[in], Force[kip])
TEST_CSV = REPO_ROOT / "data" / "resampled" / "STF01" / "force_deformation.csv"

### define fixed / known properties ###
# geometric properties (EDIT for another specimen)
A_sc = 2.00 * inch**2  # steel core area
A_t = 4.00 * inch**2  # transition area
L_T = 116.44 * inch  # total brace length (work-point span used in the model)
L_y = 76.44 * inch  # yielding-core length

# stiffness factor Q: E_hat = Q * E0  (same as scripts/model/brace_geometry.compute_Q)
Q = 1.0 / ((L_y / L_T) + ((L_T - L_y) / L_T) * (A_sc / A_t))

# material properties (fixed)
E0 = 29000.0 * ksi
fyp = 41.90 * ksi
fyn = 41.90 * ksi
a2 = 1.0
a4 = 1.0

# EDIT: SteelMPF parameters from calibration (example: STF01 individual fit)
R0 = 19.0170206
cR1 = 0.8936523307
cR2 = 0.1474889923
a1 = 0.02485363708
a3 = 0.02526905506
b_p = 0.0105133977
b_n = 0.02408616102

### load experiment ###
test_data = pd.read_csv(TEST_CSV)
target_displacement = test_data["Deformation[in]"].to_numpy(dtype=float)
target_force = test_data["Force[kip]"].to_numpy(dtype=float)

### initialize model space ###
ops.wipe()
ops.model("basic", "-ndm", 2, "-ndf", 2)

### define nodes and constraints ###
ops.node(1, 0.0, 0.0)
ops.fix(1, 1, 1)
ops.node(2, L_T, 0.0)
ops.fix(2, 0, 1)  # fix y only; x prescribed by pattern

### define materials ###
# uniaxialMaterial('SteelMPF', matTag, fyp, fyn, E0, b_p, b_n, R0, cR1, cR2, a1, a2, a3, a4)
mat_tag = 1
ops.uniaxialMaterial(
    "SteelMPF",
    mat_tag,
    fyp,
    fyn,
    Q * E0,
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

### define elements ###
ele_tag = 1
ops.element("corotTruss", ele_tag, 1, 2, A_sc, mat_tag)

### displacement-driven analysis ###
n = len(target_displacement)
dt = 1.0
# Explicit leading zero (same as the pipeline). Prefer this over Path -prependZero.
path_values = np.empty(n + 1, dtype=float)
path_values[0] = 0.0
path_values[1:] = target_displacement

ops.timeSeries("Path", 1, "-dt", dt, "-values", *path_values.tolist(), "-useLast")
ops.pattern("Plain", 1, 1)
ops.sp(2, 1, 1.0)

ops.integrator("LoadControl", dt)
ops.constraints("Transformation")
ops.numberer("Plain")
ops.system("UmfPack")
ops.analysis("Static", "-noWarnings")

disp = []
force = []
ctrl_node = 2
for i in range(n):
    ok = ops.analyze(1)
    if ok != 0:
        raise RuntimeError(f"OpenSees analyze failed at step {i + 1}/{n}.")
    disp.append(ops.nodeDisp(ctrl_node, 1))
    axial = ops.eleResponse(ele_tag, "axialForce")
    force.append(axial[0] if isinstance(axial, (list, tuple)) else float(axial))

### plots ###
plt.figure()
plt.plot(target_displacement, target_force, color="black", label="Experiment")
plt.plot(disp, force, color="C0", label="SteelMPF")
plt.xlabel("Displacement [in]")
plt.ylabel("Force [kip]")
plt.title("BRB force–deformation")
plt.legend()
plt.tight_layout()
plt.show()
