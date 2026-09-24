"""Generate notebooks/*.ipynb (one-shot authoring helper)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT / "scripts" / "examples"))
from notebook_support import LOSS_WEIGHTS_HELP, NOTEBOOK_SETTINGS_HELP  # noqa: E402

ROOT_BOOTSTRAP = '''
ROOT = Path.cwd().resolve()
if not (ROOT / "scripts" / "examples" / "notebook_support.py").is_file():
    for cand in (ROOT, *ROOT.parents):
        if (cand / "scripts" / "examples" / "notebook_support.py").is_file():
            ROOT = cand
            break
    else:
        raise SystemExit("Could not find BRB-Calibration repo root (need scripts/examples/notebook_support.py).")
import os
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "scripts" / "examples"))
'''.strip()

CATALOG_MD = """## Browse the specimen catalog

Before you pick a specimen name, look at what is in the repo:

- a **table** from `config/calibration/BRB-Specimens.csv` (geometry + layout flags)
- a **grid of raw hysteresis** from `data/raw/{Name}/` (unfiltered)

Path-ordered tests draw as loops; digitized tests draw as points.
Then edit **Parameters** below and choose a `Name` you like."""

CATALOG_CODE = f"""from pathlib import Path
import sys

{ROOT_BOOTSTRAP}
import importlib
import notebook_support
importlib.reload(notebook_support)
from notebook_support import show_specimen_catalog  # noqa: E402

catalog = show_specimen_catalog()
print("catalog rows:", len(catalog))"""


def cell_md(src: str) -> dict:
    lines = src.strip("\n").split("\n")
    src_lines = [l + "\n" for l in lines[:-1]] + ([lines[-1] + "\n"] if lines else [])
    return {"cell_type": "markdown", "metadata": {}, "source": src_lines}


def cell_code(src: str) -> dict:
    lines = src.strip("\n").split("\n")
    src_lines = [l + "\n" for l in lines[:-1]] + ([lines[-1] + "\n"] if lines else [])
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src_lines,
    }


def write_nb(name: str, cells: list[dict]) -> None:
    import uuid

    for c in cells:
        c.setdefault("id", uuid.uuid4().hex[:12])
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "BRB py312",
                "language": "python",
                "name": "brb-py312",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }
    path = NB_DIR / name
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote", path)


def main() -> None:
    write_nb(
        "00_steelmpf_parameters.ipynb",
        [
            cell_md(
                """# Understanding SteelMPF parameters

Before calibrating, it helps to see **what each SteelMPF parameter does** on a fixed
brace geometry and displacement history.

In this notebook you will:
- pick a base SteelMPF vector (sensible defaults — not all `a1`–`a4` equal to 1)
- choose a displacement history: synthetic growing cycles, or a real specimen record
- vary **one parameter at a time** and overlay the resulting hysteresis

No optimizer runs here. Simulation uses the same corotational BRB + SteelMPF path
as calibration (`simulate_force_disp` in `scripts/examples/notebook_support.py`).

Edit **Parameters**, then run the cells below in order."""
            ),
            cell_md(CATALOG_MD),
            cell_code(CATALOG_CODE),
            cell_md(
                """## Parameter roles (SteelMPF)

OpenSees syntax (see also `scripts/examples/simple_brb_steelmpf.py`):

`uniaxialMaterial SteelMPF $mattag $fyp $fyn $E0 $bp $bn $R0 $cR1 $cR2 <$a1 $a2 $a3 $a4>`

Descriptions below follow this repo’s write-up (`docs/BRB Calibration using SteelMPF.pdf`
and `paper/main.tex`). The OpenSees wiki text for `a1`–`a4` / `cR*` is easy to misread:
it sometimes swaps tension vs compression wording and reuses the labels `a1`/`a2` for the
recommended `cR1`/`cR2` values.

| Parameter | Role |
|-----------|------|
| `fyp`, `fyn` | Yield strength in tension (+) / compression (−) [ksi] |
| `E` | Initial modulus [ksi] (brace model uses `E_hat = Q E`) |
| `b_p`, `b_n` | Strain-hardening ratio in tension / compression |
| `R0` | Initial curvature of the elastic ↔ post-yield transition (Bauschinger rounding; ≈20 typical) |
| `cR1`, `cR2` | How that transition curvature `R` degrades with plastic excursion (≈0.925 and ≈0.15 typical) |
| `a1` | **Compression** isotropic shift scale: shifts the compression yield envelope by a proportion of `fyn` |
| `a2` | Compression isotropic threshold: shift activates after a max plastic **tensile** strain of about `a2 · fyp / E_hat` |
| `a3` | **Tension** isotropic shift scale: shifts the tension yield envelope by a proportion of `fyp` |
| `a4` | Tension isotropic threshold: shift activates after a max plastic **compressive** strain of about `a4 · fyn / E_hat` |

In equations (paper): tension isotropic uses `(a3, a4)`; compression isotropic uses `(a1, a2)`.
Keep `a2` and `a4` near 1 and put the interesting values in `a1` / `a3`. Setting all four to 1
(or all to 0) makes the isotropic terms idle or degenerate, so one-at-a-time plots look flat."""
            ),
            cell_md(
                """## Parameters

`BASE` is geometry + SteelMPF. `SWEEPS` lists alternate values for each free
parameter (base value is always plotted too). `DISPLACEMENT_HISTORY` is
`"synthetic"` or a specimen `Name` with resampled data."""
            ),
            cell_code(
                f"""from pathlib import Path
import sys

# --- edit these ---
SPECIMEN_GEOMETRY = "PC160"  # catalog geometry / fy
DISPLACEMENT_HISTORY = "synthetic"  # "synthetic" or a Name with data/resampled/{{Name}}/...

# Growing-cycle synthetic protocol [in] (ignored when DISPLACEMENT_HISTORY is a specimen Name)
SYNTHETIC_AMPS = [0.4, 0.8, 1.6, 2.4, 3.2]
SYNTHETIC_N_QUARTER = 35

BASE_STEEL = {{
    "b_p": 0.01,
    "b_n": 0.025,
    "R0": 20.0,
    "cR1": 0.925,
    "cR2": 0.15,
    # Isotropic: a1/a2 = compression, a3/a4 = tension (a2=a4=1 keeps a1/a3 active).
    "a1": 0.04,
    "a2": 1.0,
    "a3": 0.04,
    "a4": 1.0,
}}

# One-at-a-time values (base is overlaid automatically)
SWEEPS = {{
    "b_p": [0.002, 0.01, 0.04],
    "b_n": [0.005, 0.025, 0.06],
    "R0": [10.0, 20.0, 40.0],
    "cR1": [0.7, 0.925, 0.98],
    "cR2": [0.05, 0.15, 0.3],
    "a1": [0.0, 0.04, 0.12],
    "a3": [0.0, 0.04, 0.12],
    "a2": [0.5, 1.0, 2.0],
    "a4": [0.5, 1.0, 2.0],
}}

NORMALIZED = True
# ------------------

{ROOT_BOOTSTRAP}
import importlib
import notebook_support
importlib.reload(notebook_support)
from notebook_support import (  # noqa: E402
    geometry_from_catalog,
    load_experiment_force_disp,
    make_cyclic_drive,
    plot_steelmpf_param_sweep,
    simulate_force_disp,
)
print("repo:", ROOT)
print("geometry:", SPECIMEN_GEOMETRY)
print("displacement history:", DISPLACEMENT_HISTORY)"""
            ),
            cell_md(
                """## Build displacement history + base response

Uses catalog geometry for `SPECIMEN_GEOMETRY`. A synthetic history is fast and easy to
read; a real Name uses that specimen's resampled deformation history."""
            ),
            cell_code(
                """import matplotlib.pyplot as plt

geom = geometry_from_catalog(SPECIMEN_GEOMETRY, E=29000.0)
BASE = {**geom, **BASE_STEEL}

if str(DISPLACEMENT_HISTORY).strip().lower() == "synthetic":
    disp = make_cyclic_drive(SYNTHETIC_AMPS, n_quarter=SYNTHETIC_N_QUARTER)
    history_label = "synthetic"
else:
    disp, _exp_force = load_experiment_force_disp(str(DISPLACEMENT_HISTORY))
    history_label = str(DISPLACEMENT_HISTORY)

print(f"displacement history={history_label!r}  n={len(disp)}")
print("BASE:", {k: BASE[k] for k in ("b_p", "b_n", "R0", "cR1", "cR2", "a1", "a2", "a3", "a4")})

# Reference hysteresis at BASE
_, F_base = simulate_force_disp(disp, **BASE)
fyA = BASE["fyp"] * BASE["A_sc"]
plt.figure(figsize=(6.5, 4.8))
if NORMALIZED:
    plt.plot(100.0 * disp / BASE["L_y"], F_base / fyA, color="0.2")
    plt.xlabel(r"Axial strain, $\\delta / L_y$ (%)")
    plt.ylabel(r"Axial force, $P / (f_y A_{sc})$")
else:
    plt.plot(disp, F_base, color="0.2")
    plt.xlabel("Deformation [in]")
    plt.ylabel("Force [kip]")
plt.title(f"Base SteelMPF  ({history_label}, geometry {SPECIMEN_GEOMETRY})")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()"""
            ),
            cell_md(
                """## One-at-a-time sweeps

Each figure keeps every other parameter at `BASE` and overlays the listed values
of one parameter. Look for slope changes (`b_*`), changes in the transition from the
elastic to the post-yield asymptotes / Bauschinger rounding (`R0`/`cR*`), and loop
“inflation” / ratcheting from isotropic terms (`a*`).

Plot helper: `plot_steelmpf_param_sweep` in `scripts/examples/notebook_support.py`."""
            ),
            cell_code(
                """# Hardening ratios
for name in ("b_p", "b_n"):
    plot_steelmpf_param_sweep(
        disp, BASE, name, SWEEPS[name], normalized=NORMALIZED, title=f"Sweep {name}"
    )

# Transition elastic <-> post-yield asymptotes (Bauschinger rounding)
for name in ("R0", "cR1", "cR2"):
    plot_steelmpf_param_sweep(
        disp, BASE, name, SWEEPS[name], normalized=NORMALIZED, title=f"Sweep {name}"
    )

# Isotropic hardening (compression a1/a2, tension a3/a4)
for name in ("a1", "a2", "a3", "a4"):
    plot_steelmpf_param_sweep(
        disp, BASE, name, SWEEPS[name], normalized=NORMALIZED, title=f"Sweep {name}"
    )"""
            ),
        ],
    )

    write_nb(
        "01_single_specimen.ipynb",
        [
            cell_md(
                """# Single-specimen SteelMPF calibration

In this notebook, you will:
- prepare (or inspect) filtered / resampled test data for one specimen
- choose SteelMPF seeds and which parameters to free
- set how the objective is weighted
- run L-BFGS-B and compare experimental vs simulated hysteresis

Edit the **Parameters** cell, then run the remaining cells in order.

Requires a working OpenSeesPy install (set `$env:PYTHON` / `export PYTHON=` if needed).
Results are written under `results/notebooks/single/{SPECIMEN}/`."""
            ),
            cell_md(CATALOG_MD),
            cell_code(CATALOG_CODE),
            cell_md(LOSS_WEIGHTS_HELP),
            cell_md(NOTEBOOK_SETTINGS_HELP),
            cell_md(
                """## Parameters

Edit the block below, then run all cells under it.

- **`SPECIMEN`** — catalog `Name` (see the browse cell above).
- **`OPTIMIZE_PARAMS`** — which SteelMPF parameters L-BFGS-B may change.
- **`STEEL`** — seeds (starting values; fixed params stay here).
- **`LOSS`** — objective weights (usually L2 or L1 feature match).
- **`BOUNDS`** — optional box bounds for free params (`None` = notebook defaults)."""
            ),
            cell_code(
                f"""from pathlib import Path
import sys

# --- edit these ---
SPECIMEN = "PC160"  # catalog Name (browse cell above)
PREPARE_DATA = True  # rebuild filtered/resampled from data/raw if needed

# Names L-BFGS-B is allowed to change (everything else stays at STEEL seeds).
OPTIMIZE_PARAMS = ["b_p", "b_n", "R0", "cR1", "cR2", "a1", "a3"]
# OPTIMIZE_PARAMS = ["cR1", "cR2", "a1", "a3"]  # fewer free params (pipeline individual style)

# Seeds: starting SteelMPF values. For b_p/b_n: a numeric guess, or a keyword
# that extracts from experimental cycles (median, mean, weighted_mean, ...).
STEEL = {{
    "E": 29000,
    "b_p": "median",
    "b_n": "median",
    "R0": 20,
    "cR1": 0.8875,
    "cR2": 0.15,
    "a1": 0.04,
    "a2": 1.0,   # usually leave near 1
    "a3": 0.04,
    "a4": 1.0,   # usually leave near 1
}}

# Objective: turn on L2 *or* L1 feature matching (see settings help above).
LOSS = {{
    "w_feat_l2": 1,   # set to 0 and w_feat_l1=1 for an L1 run
    "w_feat_l1": 0,
    "w_energy_l2": 0,
    "w_energy_l1": 0,
    "w_unordered_binenv_l2": 0,
    "w_unordered_binenv_l1": 0,
    "use_amplitude_weights": True,
    "amplitude_weight_power": 2,
    "amplitude_weight_eps": 0.05,
}}

# Optional: override default box bounds for free params (None -> notebook defaults)
# BOUNDS = {{"b_p": (0.0, 0.05), "b_n": (0.0, 0.05), "R0": (1.0, 50.0)}}
BOUNDS = None
# ------------------

{ROOT_BOOTSTRAP}
import importlib
import notebook_support
importlib.reload(notebook_support)
from notebook_support import (  # noqa: E402
    apparent_b_summary,
    load_cycle_partition_meta,
    load_experiment_force_disp,
    params_dict_from_row,
    plot_cycle_energy_bars,
    plot_cycles_multi_axes,
    plot_prep_histories,
    plot_prep_stages,
    prepare_specimens,
    run_single_specimen,
    simulate_force_disp,
    yield_disp_from_params,
)
print("repo:", ROOT)
print("specimen:", SPECIMEN)
print("LOSS:", {{k: LOSS[k] for k in LOSS if str(k).startswith("w_")}})"""
            ),
            cell_md(
                """## Prepare data

Before we fit, the lab record is cleaned into the displacement history the model will step through.

Think of three versions of the same test:

1. **Raw** — the measured force–deformation trace (invalid ends trimmed).
2. **Filtered** — same number of points, but quieter (less noise / glitches).
3. **Resampled** — fewer points along the **same** path. This is what OpenSees
   and the optimizer actually use (faster, and still faithful to the protocol).

Set `PREPARE_DATA = True` above if you want to rebuild filtered/resampled from raw
(safe to re-run). Leave it `False` if those files are already on disk.

The next cell plots two views (same idea as the report time-history figures):

- **Hysteresis** — force vs deformation (the familiar loops).
- **Histories vs point index** — raw vs filtered only (same sampling length).
- **Histories vs cumulative |Δδ|** — raw, filtered, and resampled on path length
  (`Σ|Δδ|`), so different sample counts still line up on the same protocol.
  Filtered uses a dashed line; resampled uses dash-dot."""
            ),
            cell_code(
                """if PREPARE_DATA:
    prepare_specimens([SPECIMEN], e_ksi=float(STEEL["E"]))
else:
    print(f"Using existing filtered/resampled CSVs for {SPECIMEN!r} (PREPARE_DATA=False)")

plot_prep_stages(SPECIMEN)
plot_prep_histories(SPECIMEN)"""
            ),
            cell_md(
                """## Run calibration

This starts the optimizer (L-BFGS-B), then writes fitted parameters and overlay
PNGs under `results/notebooks/single/`.

Tip: if you skipped **Prepare data**, keep `PREPARE_DATA = True` so missing
filtered/resampled files are built here automatically."""
            ),
            cell_code(
                """paths = run_single_specimen(
    SPECIMEN,
    optimize_params=OPTIMIZE_PARAMS,
    steel=STEEL,
    loss=LOSS,
    prepare_data=PREPARE_DATA,
    bounds=BOUNDS,
)
print("parameters:", paths["params"])
print("plots dir: ", paths["plots_dir"])"""
            ),
            cell_md(
                """## Results

Here we load the fit, re-run the model on the experimental displacement history, and overlay
experiment vs simulation.

Want to poke at a parameter? Change a value in `p` below and re-run **just this
cell** — no need to re-optimize."""
            ),
            cell_code(
                """import matplotlib.pyplot as plt
import pandas as pd

params = pd.read_csv(paths["params"])
display(params.T)

# Geometry + SteelMPF from the fitted row - edit any value to explore manually.
p = params_dict_from_row(params.iloc[0])
# p["b_p"] = 0.02
# p["R0"] = 18.0

exp_disp, exp_force = load_experiment_force_disp(SPECIMEN)
num_disp, num_force = simulate_force_disp(exp_disp, **p)

plt.figure()
plt.plot(exp_disp, exp_force, color="0.45", label="Experimental")
plt.plot(num_disp, num_force, linestyle="--", label="Numerical")
plt.xlabel("Deformation [in]")
plt.ylabel("Force [kip]")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Normalized axes (same quantities used in the report overlays)
fyA = p["fyp"] * p["A_sc"]
plt.figure()
plt.plot(100.0 * exp_disp / p["L_y"], exp_force / fyA, color="0.45", label="Experimental")
plt.plot(100.0 * num_disp / p["L_y"], num_force / fyA, linestyle="--", label="Numerical")
plt.xlabel(r"Axial strain, $\\delta / L_y$ (%)")
plt.ylabel(r"Axial force, $P / (f_y A_{sc})$")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()"""
            ),
            cell_md(
                """## Diagnostics

A closer look at what the objective is “looking at”:

- **Characteristic points** — one small plot per cycle, with peaks / unload /
  re-yield points numbered 1–14. **S** and **E** are cycle start and end.
  Titles show amplitude `A` and how much that cycle is weighted (`w_c`).
- **Energy** — per-cycle `|∫ F du|` for experiment vs the model.
- **Apparent b** — simple hardening slopes from the data (useful seeds / sanity check).

Changed `p` in **Results**? Re-run this cell to refresh the numerical overlays.

### Characteristic points 1–14 (plain language)

These are the same landmarks the optimizer uses. Yield force is
`F_thr = f_y A_sc`.

| # | What it marks |
|---|----------------|
| 1 | Cycle start (same as **S**) |
| 2 | First near-zero deformation after the first peak |
| 3 | Peak tension |
| 4 | Peak compression |
| 5 | Tension-side “yield” corner |
| 6 | Compression-side “yield” corner |
| 7 | Zero-force crossing at the **largest** deformation |
| 8 | Zero-force crossing at the **smallest** deformation |
| 9–12 | Half-peak / half-yield path samples (where the loop is reloading) |
| 13–14 | Partial unload after each peak (about 90% of yield force) |

Why markers often **pile up near the unload → re-yield knee**: several of those
rules intentionally sit in that band, so on a BRB loop they land close together
even though each rule is different. Small cycles may leave some slots empty.

<details>
<summary>Source files (if you want to change definitions)</summary>

- Cycle weights `w_c`: `scripts/calibrate/amplitude_mse_partition.py`
- Landmarks 1–14: `scripts/calibrate/cycle_feature_loss.py` (`extract_cycle_landmarks`)
- These plots: `scripts/examples/notebook_support.py`
  (`plot_cycles_multi_axes`, `plot_cycle_energy_bars`)

</details>"""
            ),
            cell_code(
                """# Same partition + landmarks the optimizer uses (see paths in the markdown above).
_, _, cycle_meta = load_cycle_partition_meta(
    SPECIMEN,
    use_amplitude_weights=bool(LOSS["use_amplitude_weights"]),
    amplitude_weight_power=float(LOSS["amplitude_weight_power"]),
    amplitude_weight_eps=float(LOSS["amplitude_weight_eps"]),
)
print(f"{SPECIMEN}: {len(cycle_meta)} weight cycles")

dy = yield_disp_from_params(p)  # scripts/examples/notebook_support.py -> yield_disp_from_params
# -> scripts/calibrate/cycle_feature_loss.py -> yield_displacement_dy_in
plot_cycles_multi_axes(  # scripts/examples/notebook_support.py
    exp_disp,
    exp_force,
    cycle_meta,
    fy_ksi=p["fyp"],
    A_sc=p["A_sc"],
    dy_in=dy,
    force_num=num_force,
    ncols=6,
    title=f"{SPECIMEN}: characteristic points by cycle",
)

plot_cycle_energy_bars(  # scripts/examples/notebook_support.py
    exp_disp, exp_force, cycle_meta, force_num=num_force
)

bsum = apparent_b_summary(SPECIMEN, p)  # -> scripts/calibrate/extract_bn_bp.py
print("apparent b / Q:")
for k, v in bsum.items():
    print(f"  {k}: {v:.6g}")
print(f"fitted b_p={p['b_p']:.6g}, b_n={p['b_n']:.6g}")"""
            ),
        ],
    )

    write_nb(
        "02_individual.ipynb",
        [
            cell_md(
                """# Individual calibration (several specimens)

In this notebook, you will:
- pick a short list of specimens (keep it to 2–3 for a quick run)
- set one shared SteelMPF / loss configuration
- optimize **each** specimen separately and compare overlays

This matches the individual path in `optimize_brb_mse.py` (one fit per specimen).

Edit the **Parameters** cell, then run the remaining cells in order.

Outputs: `results/notebooks/individual/` and
`results/plots/calibration/individual_optimize/notebook_individual/`."""
            ),
            cell_md(CATALOG_MD),
            cell_code(CATALOG_CODE),
            cell_md(LOSS_WEIGHTS_HELP),
            cell_md(NOTEBOOK_SETTINGS_HELP),
            cell_md(
                """## Parameters

- **`SPECIMENS`** — list to fit (keep 2–3 for a quick demo; grow the list to mirror
  the pipeline’s multi-specimen individual run).
- **`SET_ID_ROW`** — one shared seed / loss / free-parameter recipe (same idea as
  `config/calibration/set_id_settings.csv`). See the settings help above for
  `set_id`, `inherit_from_set`, seeds, and `optimize_params`.

To mirror the automated L2 then L1 workflow: run once with the defaults below
(`set_id=1`, L2), then switch to the commented L1 block (`set_id=3`) and run again."""
            ),
            cell_code(
                f"""from pathlib import Path
import sys

# --- edit these ---
SPECIMENS = ["PC3SB", "PC160", "PC250"]  # keep to 2–3 for a quick demo
PREPARE_DATA = True

# One shared recipe for every specimen in SPECIMENS.
# set_id         -> label on CSVs/plots (1 = L2, 3 = L1 in the pipeline)
# inherit_from_set -> -999 = start from seeds; else start from that set_id's optimized params
# optimize_params  -> which SteelMPF names L-BFGS-B may change
# E, b_p, ...      -> seeds (fixed params stay here; free params start here)
SET_ID_ROW = {{
    "set_id": 1,
    "inherit_from_set": -999,
    "E": 29000,
    "b_p": "weighted_mean",  # numeric guess, or extract from cycles: median, mean, weighted_mean, ...
    "b_n": "weighted_mean",
    "R0": 20,
    "cR1": 0.8875,
    "cR2": 0.15,
    "a1": 0.04,
    "a2": 1.0,
    "a3": 0.04,
    "a4": 1.0,
    "optimize_params": ["cR1", "cR2", "a1", "a3"],
    # "optimize_params": ["b_p", "b_n", "R0", "cR1", "cR2", "a1", "a3"],  # freer
    "w_feat_l2": 1,
    "w_feat_l1": 0,
    "w_energy_l2": 0,
    "w_energy_l1": 0,
    "w_unordered_binenv_l2": 0,
    "w_unordered_binenv_l1": 0,
    "use_amplitude_weights": True,
    "amplitude_weight_power": 2,
    "amplitude_weight_eps": 0.05,
}}

# --- optional: L1 pass (pipeline set_id=3) — uncomment to replace the block above ---
# SET_ID_ROW = {{
#     "set_id": 3,
#     "inherit_from_set": -999,  # or 1 to start from set_id=1, then change weights / free params
#     "E": 29000,
#     "b_p": "weighted_mean",
#     "b_n": "weighted_mean",
#     "R0": 20,
#     "cR1": 0.8875,
#     "cR2": 0.15,
#     "a1": 0.04,
#     "a2": 1.0,
#     "a3": 0.04,
#     "a4": 1.0,
#     "optimize_params": ["cR1", "cR2", "a1", "a3"],
#     "w_feat_l2": 0,
#     "w_feat_l1": 1,
#     "w_energy_l2": 0,
#     "w_energy_l1": 0,
#     "w_unordered_binenv_l2": 0,
#     "w_unordered_binenv_l1": 0,
#     "use_amplitude_weights": True,
#     "amplitude_weight_power": 2,
#     "amplitude_weight_eps": 0.05,
# }}
# ------------------

{ROOT_BOOTSTRAP}
from notebook_support import (  # noqa: E402
    load_experiment_force_disp,
    params_dict_from_row,
    run_individual_specimens,
    simulate_force_disp,
)
print("specimens:", SPECIMENS)
print("set_id:", SET_ID_ROW["set_id"], "inherit_from_set:", SET_ID_ROW["inherit_from_set"])
print("optimize_params:", SET_ID_ROW["optimize_params"])
print("feat weights: L2=", SET_ID_ROW["w_feat_l2"], "L1=", SET_ID_ROW["w_feat_l1"])"""
            ),
            cell_md(
                """## Run individual optimize

Fits each specimen on its own (same seeds / loss / free params for all), then
writes overlays so you can compare them side by side."""
            ),
            cell_code(
                """paths = run_individual_specimens(
    SPECIMENS,
    set_id_row=SET_ID_ROW,
    prepare_data=PREPARE_DATA,
)
print("params:  ", paths["params"])
print("overlays:", paths["overlays_dir"])"""
            ),
            cell_md(
                """## Results

For each specimen: load the experiment, rebuild the model force from that
specimen's fitted parameters, and plot. Edit `p` inside the loop if you want to
explore without re-fitting."""
            ),
            cell_code(
                """import matplotlib.pyplot as plt
import pandas as pd

params = pd.read_csv(paths["params"])
display(params)

sid = int(SET_ID_ROW["set_id"])
for name in SPECIMENS:
    row = params[(params["Name"].astype(str) == name) & (params["set_id"] == sid)].iloc[0]
    p = params_dict_from_row(row)
    exp_disp, exp_force = load_experiment_force_disp(name)
    num_disp, num_force = simulate_force_disp(exp_disp, **p)

    fyA = p["fyp"] * p["A_sc"]
    plt.figure()
    plt.plot(100.0 * exp_disp / p["L_y"], exp_force / fyA, color="0.45", label="Experimental")
    plt.plot(100.0 * num_disp / p["L_y"], num_force / fyA, linestyle="--", label="Numerical")
    plt.xlabel(r"Axial strain, $\\delta / L_y$ (%)")
    plt.ylabel(r"Axial force, $P / (f_y A_{sc})$")
    plt.title(name)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()"""
            ),
        ],
    )

    write_nb(
        "03_generalized.ipynb",
        [
            cell_md(
                """# Generalized calibration (shared parameters)

In this notebook, you will:
- choose which specimens enter a **joint** fit (and their relative weights)
- set one shared SteelMPF seed / free-parameter / loss configuration
- optimize one parameter vector for the whole group
- check **validation** specimens the fit never saw: one path-ordered BRB plus
  digitized specimens (including a multi-amplitude displacement history)

**Requirement:** each train Name needs path-ordered resampled data
(`data/resampled/{Name}/force_deformation.csv`). Digitized unordered specimens
cannot enter the joint fit, but they can be validation checks (via `J_binenv`).
Catalog `generalized_weight` is ignored here — use `TRAIN_SPECIMENS` weights
instead (default 1).

Edit the **Parameters** cell, then run the remaining cells in order.

Outputs: `results/notebooks/generalized/`."""
            ),
            cell_md(CATALOG_MD),
            cell_code(CATALOG_CODE),
            cell_md(LOSS_WEIGHTS_HELP),
            cell_md(NOTEBOOK_SETTINGS_HELP),
            cell_md(
                """## What `J_binenv` is

Some validation specimens are **digitized clouds**: many `(deformation, force)`
points with no reliable cycle-by-cycle time order. For those, the usual
characteristic-point score is not a fair comparison.

`J_binenv` (binned envelope) is the alternative:

1. Slice the deformation axis into bins.
2. In each bin, read the **highest** and **lowest** force on the experiment and
   on the model.
3. Score how well those upper/lower envelopes match.

So: **for digitized validation, look at `J_binenv` first.** Train specimens also
report it, which makes train vs validation easier to compare on one scale.
(The fit usually leaves the binenv **weight** at 0; the column is still filled
for reporting.)"""
            ),
            cell_md(
                """## Parameters

- **`TRAIN_SPECIMENS`** — who enters the joint fit, and with what relative weight.
- **`VALIDATION_SPECIMENS`** — not in the joint fit: metrics and overlays only (weight 0).
- **`SET_ID_ROW`** — shared SteelMPF seeds, free parameters, and loss weights
  (see settings help above for `set_id`, `inherit_from_set`, seeds, `optimize_params`).

You can run this notebook more than once with different recipes. Defaults below
are L2 (`set_id=1`). To run L1 as well, switch to the commented block
(`set_id=3`, `w_feat_l1=1`) and run again so the two result sets stay separate."""
            ),
            cell_code(
                f"""from pathlib import Path
import sys

# --- edit these ---
# Name -> specimen weight in the joint objective (default 1).
TRAIN_SPECIMENS = {{
    "PC250": 1.0,
    "PC3SB": 1.0,
    "PC160": 1.0,
}}
# Validation (not in the joint fit): path-ordered + digitized unordered.
# B6 / A5 = digitized; CB225 = growing multi-amplitude protocol.
VALIDATION_SPECIMENS = ["PC350", "B6", "CB225", "A5"]
PREPARE_DATA = True

# Shared recipe for the joint fit (and for scoring validation).
# set_id / inherit_from_set / optimize_params / seeds — see settings help above.
SET_ID_ROW = {{
    "set_id": 1,
    "inherit_from_set": -999,
    "E": 29000,
    "b_p": 0.005,   # numeric guess, or extract from cycles: "median", "mean", "weighted_mean", ...
    "b_n": 0.025,
    "R0": 20,
    "cR1": 0.8875,
    "cR2": 0.15,
    "a1": 0.04,
    "a2": 1.0,
    "a3": 0.04,
    "a4": 1.0,
    "optimize_params": ["b_p", "b_n", "cR1", "cR2", "a1", "a3"],
    # "optimize_params": ["cR1", "cR2", "a1", "a3"],  # fewer free params
    "w_feat_l2": 1,
    "w_feat_l1": 0,
    "w_energy_l2": 0,
    "w_energy_l1": 0,
    "w_unordered_binenv_l2": 0,
    "w_unordered_binenv_l1": 0,
    "use_amplitude_weights": True,
    "amplitude_weight_power": 2,
    "amplitude_weight_eps": 0.05,
}}

# --- optional: L1 pass (pipeline set_id=3) — uncomment to replace the block above ---
# SET_ID_ROW = {{
#     "set_id": 3,
#     "inherit_from_set": -999,
#     "E": 29000,
#     "b_p": 0.005,
#     "b_n": 0.025,
#     "R0": 20,
#     "cR1": 0.8875,
#     "cR2": 0.15,
#     "a1": 0.04,
#     "a2": 1.0,
#     "a3": 0.04,
#     "a4": 1.0,
#     "optimize_params": ["b_p", "b_n", "cR1", "cR2", "a1", "a3"],
#     "w_feat_l2": 0,
#     "w_feat_l1": 1,
#     "w_energy_l2": 0,
#     "w_energy_l1": 0,
#     "w_unordered_binenv_l2": 0,
#     "w_unordered_binenv_l1": 0,
#     "use_amplitude_weights": True,
#     "amplitude_weight_power": 2,
#     "amplitude_weight_eps": 0.05,
# }}
# ------------------

{ROOT_BOOTSTRAP}
from notebook_support import (  # noqa: E402
    digitized_unordered_sim_arrays,
    ensure_sys_path,
    load_experiment_force_disp,
    params_dict_from_row,
    run_generalized_demo,
    simulate_force_disp,
    summarize_generalized_train_validation_metrics,
)
ensure_sys_path()
print("train weights:", TRAIN_SPECIMENS)
print("validation:", VALIDATION_SPECIMENS)
print("set_id:", SET_ID_ROW["set_id"], "inherit_from_set:", SET_ID_ROW["inherit_from_set"])
print("optimize_params:", SET_ID_ROW["optimize_params"])
print("feat weights: L2=", SET_ID_ROW["w_feat_l2"], "L1=", SET_ID_ROW["w_feat_l1"])"""
            ),
            cell_md(
                """## Run generalized optimize

Learns **one** shared SteelMPF vector from the weighted train list, then scores
train and validation specimens. Validation names never enter the fit — they are
a hold-out check."""
            ),
            cell_code(
                """paths = run_generalized_demo(
    TRAIN_SPECIMENS,
    set_id_row=SET_ID_ROW,
    eval_specimens=VALIDATION_SPECIMENS,
    prepare_data=PREPARE_DATA,
)
print("params: ", paths["params"])
print("metrics:", paths["metrics"])
print("plots:  ", paths["plots_dir"])
print("weights:", paths["train_weights"])
print("validation:", paths.get("validation_specimens", paths["eval_specimens"]))"""
            ),
            cell_md(
                """## Results — train vs validation

Two scores show up a lot:

- **`J_feat`** — characteristic-point match (path-ordered cycles; what the fit
  usually minimizes).
- **`J_binenv`** — envelope match (especially useful for digitized clouds; also
  printed for path-ordered tests so everything shares one scale).

Rule of thumb: train specimens should tend to look **better** (lower scores)
than validation specimens the model has not seen."""
            ),
            cell_code(
                """import matplotlib.pyplot as plt
import pandas as pd

params = pd.read_csv(paths["params"])
metrics = pd.read_csv(paths["metrics"])
display(params[["Name", "set_id", "b_p", "b_n", "R0", "cR1", "cR2", "a1", "a3"]].head(12))

cmp = summarize_generalized_train_validation_metrics(
    metrics,
    train_names=list(TRAIN_SPECIMENS),
    eval_names=VALIDATION_SPECIMENS,
)
display(cmp)

train_bin = cmp.loc[cmp["role"] == "train", "J_binenv"]
val_bin = cmp.loc[cmp["role"] == "validation", "J_binenv"]
if len(train_bin) and len(val_bin):
    print(
        f"mean J_binenv  train={train_bin.mean():.6g}  "
        f"validation={val_bin.mean():.6g}"
    )

sid = int(SET_ID_ROW["set_id"])

# Train path-ordered overlays
for name in TRAIN_SPECIMENS:
    row = params[(params["Name"].astype(str) == name) & (params["set_id"] == sid)].iloc[0]
    p = params_dict_from_row(row)
    exp_disp, exp_force = load_experiment_force_disp(name)
    num_disp, num_force = simulate_force_disp(exp_disp, **p)
    fyA = p["fyp"] * p["A_sc"]
    jrow = cmp.loc[cmp["Name"] == name].iloc[0]
    plt.figure()
    plt.plot(100.0 * exp_disp / p["L_y"], exp_force / fyA, color="0.45", label="Experimental")
    plt.plot(100.0 * num_disp / p["L_y"], num_force / fyA, linestyle="--", label="Numerical")
    plt.xlabel(r"Axial strain, $\\delta / L_y$ (%)")
    plt.ylabel(r"Axial force, $P / (f_y A_{sc})$")
    plt.title(
        f"{name} [train]  J_feat={jrow['J_feat']:.4g}  J_binenv={jrow['J_binenv']:.4g}"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Validation: path-ordered hysteresis or digitized cloud
from specimen_catalog import get_specimen_record, read_catalog, uses_unordered_inputs

_cat = read_catalog()
for name in VALIDATION_SPECIMENS:
    rows = params[(params["Name"].astype(str) == name) & (params["set_id"] == sid)]
    if rows.empty:
        print(f"(skip {name}: no params row)")
        continue
    p = params_dict_from_row(rows.iloc[0])
    jrow = cmp.loc[cmp["Name"] == name]
    j_feat = float(jrow["J_feat"].iloc[0]) if len(jrow) else float("nan")
    j_bin = float(jrow["J_binenv"].iloc[0]) if len(jrow) else float("nan")
    if uses_unordered_inputs(get_specimen_record(name, _cat)):
        dig = digitized_unordered_sim_arrays(name, p)
        fyA = p["fyp"] * p["A_sc"]
        plt.figure()
        plt.scatter(
            100.0 * dig["u_cloud"] / p["L_y"],
            dig["F_cloud"] / fyA,
            s=6,
            c="0.55",
            alpha=0.5,
            label="Experimental cloud",
        )
        plt.plot(
            100.0 * dig["D_drive"] / p["L_y"],
            dig["F_sim"] / fyA,
            linestyle="--",
            label="Numerical",
        )
        plt.xlabel(r"Axial strain, $\\delta / L_y$ (%)")
        plt.ylabel(r"Axial force, $P / (f_y A_{sc})$")
        plt.title(f"{name} [validation, digitized]  J_binenv={j_bin:.4g}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        continue

    exp_disp, exp_force = load_experiment_force_disp(name)
    num_disp, num_force = simulate_force_disp(exp_disp, **p)
    fyA = p["fyp"] * p["A_sc"]
    plt.figure()
    plt.plot(100.0 * exp_disp / p["L_y"], exp_force / fyA, color="0.45", label="Experimental")
    plt.plot(100.0 * num_disp / p["L_y"], num_force / fyA, linestyle="--", label="Numerical")
    plt.xlabel(r"Axial strain, $\\delta / L_y$ (%)")
    plt.ylabel(r"Axial force, $P / (f_y A_{sc})$")
    plt.title(f"{name} [validation]  J_feat={j_feat:.4g}  J_binenv={j_bin:.4g}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()"""
            ),
        ],
    )

    write_nb(
        "04_generalized_l2_l1.ipynb",
        [
            cell_md(
                """# Generalized calibration with L2 and L1

In this notebook, you will:
- choose which specimens enter a **joint** fit (and their relative weights)
- choose **validation** specimens the fit never sees (metrics and overlays only)
- set shared SteelMPF seeds / free parameters
- optimize **two** recipes in one run: L2 (`set_id=1`) and L1 (`set_id=3`)
- compare train vs validation scores, then plot overlays (optional focus Name first)

**Requirement:** each train Name needs path-ordered resampled data
(`data/resampled/{Name}/force_deformation.csv`). Digitized unordered specimens
cannot enter the joint fit, but they can be validation checks (via `J_binenv`).
Catalog `generalized_weight` is ignored here — use `TRAIN_SPECIMENS` weights
instead (default 1).

Defaults in the Parameters cell are one useful split; edit Names and weights
freely.

Edit the **Parameters** cell, then run the remaining cells in order.

Outputs: `results/notebooks/generalized_l2_l1/`.

### Optional: faster batch run

If this notebook is too slow, use the automated pipeline from the repo root
(`./run.sh` or `.\\run.ps1`). Who trains vs who is validation-only is the
`generalized_weight` column in `config/calibration/BRB-Specimens.csv`
(**1** = train, **0** = validation scored after the fit). L2 / L1 recipes live
in `config/calibration/set_id_settings_generalized.csv`."""
            ),
            cell_md(CATALOG_MD),
            cell_code(CATALOG_CODE),
            cell_md(LOSS_WEIGHTS_HELP),
            cell_md(NOTEBOOK_SETTINGS_HELP),
            cell_md(
                """## What `J_binenv` is

Some validation specimens are **digitized clouds**: many `(deformation, force)`
points with no reliable cycle-by-cycle time order. For those, the usual
characteristic-point score is not a fair comparison.

`J_binenv` (binned envelope) is the alternative:

1. Slice the deformation axis into bins.
2. In each bin, read the **highest** and **lowest** force on the experiment and
   on the model.
3. Score how well those upper/lower envelopes match.

So: **for digitized validation, look at `J_binenv` first.** Path-ordered
specimens also report it, which makes train vs validation easier to compare on
one scale. (The fit usually leaves the binenv **weight** at 0; the column is
still filled for reporting.)"""
            ),
            cell_md(
                """## Parameters

- **`TRAIN_SPECIMENS`** — who enters the joint fit, and with what relative weight.
- **`VALIDATION_SPECIMENS`** — built from `FOCUS_VALIDATION` plus
  `EXTRA_VALIDATION`: metrics and overlays only (weight 0).
- **`FOCUS_VALIDATION`** — plotted first in the results section (any validation
  Name; keep it out of `TRAIN_SPECIMENS` if you want a true check).
- **`SET_ID_ROWS`** — shared seeds / free params with two loss recipes (L2 then
  L1) in one optimizer call.

Change Names, weights, or recipes below, then run all cells under this one."""
            ),
            cell_code(
                f"""from pathlib import Path
import sys

# --- edit these ---
FOCUS_VALIDATION = "PC3SB"
# Extra validation (not in the joint fit). B6 / A5 = digitized clouds;
# CB225 = path-ordered digitized with a growing multi-amplitude protocol.
EXTRA_VALIDATION = ["CB225", "B6", "A5"]
PREPARE_DATA = True

# Name -> specimen weight in the joint objective (default 1).
TRAIN_SPECIMENS = {{
    "PC250": 1.0,
    "PC350": 1.0,
    "PC500": 1.0,
    "PC160": 1.0,
    "PC750A": 1.0,
    "PC750B": 1.0,
    "PC1200A": 1.0,
    "PC1200B": 1.0,
}}
# Too slow? Prefer the automated pipeline (./run.sh or .\\run.ps1) and set
# generalized_weight in config/calibration/BRB-Specimens.csv
# (1 = train, 0 = validation only).

# Shared SteelMPF seeds / free params; only loss weights and set_id differ below.
_STEEL = {{
    "E": 29000,
    "b_p": 0.005,   # numeric guess, or extract from cycles: "median", "mean", "weighted_mean", ...
    "b_n": 0.025,
    "R0": 20,
    "cR1": 0.8875,
    "cR2": 0.15,
    "a1": 0.04,
    "a2": 1.0,
    "a3": 0.04,
    "a4": 1.0,
    "optimize_params": ["b_p", "b_n", "cR1", "cR2", "a1", "a3"],
    "w_energy_l2": 0,
    "w_energy_l1": 0,
    "w_unordered_binenv_l2": 0,
    "w_unordered_binenv_l1": 0,
    "use_amplitude_weights": True,
    "amplitude_weight_power": 2,
    "amplitude_weight_eps": 0.05,
}}

SET_ID_ROWS = [
    {{**_STEEL, "set_id": 1, "inherit_from_set": -999, "w_feat_l2": 1, "w_feat_l1": 0}},
    {{**_STEEL, "set_id": 3, "inherit_from_set": -999, "w_feat_l2": 0, "w_feat_l1": 1}},
]
# ------------------

{ROOT_BOOTSTRAP}
from notebook_support import (  # noqa: E402
    NOTEBOOK_RESULTS,
    digitized_unordered_sim_arrays,
    ensure_sys_path,
    load_experiment_force_disp,
    params_dict_from_row,
    run_generalized_demo,
    simulate_force_disp,
    summarize_generalized_train_validation_metrics,
)
ensure_sys_path()

VALIDATION_SPECIMENS = [FOCUS_VALIDATION] + [n for n in EXTRA_VALIDATION if n != FOCUS_VALIDATION]
assert FOCUS_VALIDATION not in TRAIN_SPECIMENS, FOCUS_VALIDATION
print("train:", TRAIN_SPECIMENS)
print("validation (focus first):", VALIDATION_SPECIMENS)
print("set_ids:", [r["set_id"] for r in SET_ID_ROWS])"""
            ),
            cell_md(
                """## Run the joint fits

Learns **one** shared SteelMPF vector from the weighted train list for each
`set_id` row (here: L2, then L1), then scores train and validation specimens.
Validation Names never enter the fit — they are a check afterward."""
            ),
            cell_code(
                """paths = run_generalized_demo(
    TRAIN_SPECIMENS,
    set_id_row=SET_ID_ROWS,
    eval_specimens=VALIDATION_SPECIMENS,
    prepare_data=PREPARE_DATA,
    work_dir=NOTEBOOK_RESULTS / "generalized_l2_l1",
)
print("params: ", paths["params"])
print("metrics:", paths["metrics"])
print("plots:  ", paths["plots_dir"])
print("set_ids:", paths.get("set_ids"))
print("train:  ", paths["train_weights"])
print("validation:", paths.get("validation_specimens", paths["eval_specimens"]))"""
            ),
            cell_md(
                """## Results — train vs validation

The table tags each Name as **train** or **validation**, for each `set_id` that
was run.

- **`J_feat`** — characteristic-point match (what the fit usually minimizes).
- **`J_binenv`** — envelope match (especially useful for digitized clouds).

Rule of thumb: train specimens should tend to look **better** (lower scores)
than validation specimens the model has not seen.

Overlays below plot experiment vs model for validation Names (`FOCUS_VALIDATION`
first), then the other validation Names, then train Names."""
            ),
            cell_code(
                """import matplotlib.pyplot as plt
import pandas as pd
from specimen_catalog import get_specimen_record, read_catalog, uses_unordered_inputs

params = pd.read_csv(paths["params"])
metrics = pd.read_csv(paths["metrics"])

cmp = summarize_generalized_train_validation_metrics(
    metrics,
    train_names=list(TRAIN_SPECIMENS),
    eval_names=VALIDATION_SPECIMENS,
)
display(cmp)

focus = cmp[cmp["Name"].astype(str) == FOCUS_VALIDATION]
print(f"Focus validation: {FOCUS_VALIDATION}")
display(focus)

set_ids = [int(r["set_id"]) for r in SET_ID_ROWS]
_cat = read_catalog()


def _overlay(name: str, sid: int, role: str) -> None:
    rows = params[(params["Name"].astype(str) == name) & (params["set_id"] == sid)]
    if rows.empty:
        print(f"(skip {name} set_id={sid}: no params row)")
        return
    p = params_dict_from_row(rows.iloc[0])
    jrow = cmp[(cmp["Name"].astype(str) == name) & (cmp["set_id"] == sid)]
    j_feat = float(jrow["J_feat"].iloc[0]) if len(jrow) else float("nan")
    j_bin = float(jrow["J_binenv"].iloc[0]) if len(jrow) else float("nan")
    loss = "L2" if sid == 1 else ("L1" if sid == 3 else f"set{sid}")
    fyA = p["fyp"] * p["A_sc"]
    plt.figure()
    if uses_unordered_inputs(get_specimen_record(name, _cat)):
        dig = digitized_unordered_sim_arrays(name, p)
        plt.scatter(
            100.0 * dig["u_cloud"] / p["L_y"],
            dig["F_cloud"] / fyA,
            s=6,
            c="0.55",
            alpha=0.5,
            label="Experimental cloud",
        )
        plt.plot(
            100.0 * dig["D_drive"] / p["L_y"],
            dig["F_sim"] / fyA,
            linestyle="--",
            label="Numerical",
        )
    else:
        exp_disp, exp_force = load_experiment_force_disp(name)
        num_disp, num_force = simulate_force_disp(exp_disp, **p)
        plt.plot(100.0 * exp_disp / p["L_y"], exp_force / fyA, color="0.45", label="Experimental")
        plt.plot(100.0 * num_disp / p["L_y"], num_force / fyA, linestyle="--", label="Numerical")
    plt.xlabel(r"Axial strain, $\\delta / L_y$ (%)")
    plt.ylabel(r"Axial force, $P / (f_y A_{sc})$")
    plt.title(f"{name} [{role}, {loss}]  J_feat={j_feat:.4g}  J_binenv={j_bin:.4g}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


print("=== Validation overlays (focus first) ===")
for sid in set_ids:
    _overlay(FOCUS_VALIDATION, sid, "validation")

print("=== Other validation overlays ===")
for name in VALIDATION_SPECIMENS:
    if name == FOCUS_VALIDATION:
        continue
    for sid in set_ids:
        _overlay(name, sid, "validation")

print("=== Train overlays ===")
for name in TRAIN_SPECIMENS:
    for sid in set_ids:
        _overlay(name, sid, "train")
"""
            ),
        ],
    )

    (NB_DIR / "README.md").write_text(
        """# Notebooks

Interactive demos for SteelMPF calibration. The rest of the repo (`run.sh` /
`run.ps1`) runs the same ideas automatically over the full catalog; if you prefer
Jupyter, **these notebooks are enough** — you do not need the automated pipeline.

Run from the **repo root** so `scripts/` and `config/` resolve. Use a Python kernel
with working OpenSeesPy (`$env:PYTHON` / `export PYTHON=` if needed).

Each notebook starts by **browsing the specimen catalog** (table + raw hysteresis
grid) so you can pick a `Name` without digging through `data/raw/` CSVs.

| Notebook | What you do |
|----------|-------------|
| [`00_steelmpf_parameters.ipynb`](00_steelmpf_parameters.ipynb) | See what each SteelMPF parameter does (one-at-a-time sweeps) |
| [`01_single_specimen.ipynb`](01_single_specimen.ipynb) | Fit one specimen; edit seeds, free params, and loss weights |
| [`02_individual.ipynb`](02_individual.ipynb) | Fit 2–3 specimens separately with one shared configuration |
| [`03_generalized.ipynb`](03_generalized.ipynb) | Joint fit + validation (path-ordered and digitized); compare `J_binenv` |
| [`04_generalized_l2_l1.ipynb`](04_generalized_l2_l1.ipynb) | Joint fit + validation; same idea as `03`, with defaults that run L2 and L1 together |

Helpers live in [`scripts/examples/notebook_support.py`](../scripts/examples/notebook_support.py).
Generated files go to `results/notebooks/` (gitignored).

Suggested order: start with `00` for intuition, then `01` → `02` → `03`.
Notebook `04` is another joint-fit + validation notebook (defaults differ —
including two loss recipes in one run); edit the Parameters cell like any other.
In each calibration notebook (`01`–`04`): browse the catalog, read the objective
and **settings** sections (`set_id`, `inherit_from_set`, seeds, `optimize_params`),
edit **Parameters**, run the cells in order. Commented L1 (`set_id=3`) blocks in
`02` / `03` show how to run L2 then L1 by hand; `04` runs both in one pass.
Results cells load experimental and numerical force–displacement arrays so you
can plot or tweak parameters yourself. Notebook `01` also has a **Diagnostics**
section (per-cycle characteristic points, energy, apparent b). Notebooks `03`
and `04` report train vs validation `J_feat` / `J_binenv`.
""",
        encoding="utf-8",
    )
    print("wrote", NB_DIR / "README.md")


if __name__ == "__main__":
    main()
