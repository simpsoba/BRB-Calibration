"""Generate notebooks/*.ipynb (one-shot authoring helper)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(exist_ok=True)


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
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
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
        "01_single_specimen.ipynb",
        [
            cell_md(
                """# 01 — Single-specimen SteelMPF calibration

Fit **one** BRB specimen. Edit the **Parameters** cell, then run all cells.

Uses the same optimizer as `scripts/calibrate_single/calibrate_one_specimen.py`.
Outputs go under `results/notebooks/single/{SPECIMEN}/`.

Requires a working OpenSeesPy install (set `$env:PYTHON` / `export PYTHON=` if needed)."""
            ),
            cell_md("## Parameters (edit me)"),
            cell_code(
                """from pathlib import Path
import sys

# --- edit these ---
SPECIMEN = "STF01"
METRIC = "l2"  # "l1" or "l2" landmark feature loss
PREPARE_DATA = True  # build filtered/resampled from data/raw if needed

OPTIMIZE_PARAMS = ["b_p", "b_n", "R0", "cR1", "cR2", "a1", "a3"]

STEEL = {
    "E": 29000,
    "b_p": "median",   # number or apparent-b keyword: median, weighted_mean, ...
    "b_n": "median",
    "R0": 20,
    "cR1": 0.8875,
    "cR2": 0.15,
    "a1": 0.04,
    "a2": 1.0,
    "a3": 0.04,
    "a4": 1.0,
}

# Optional: override default box bounds for free params (None -> notebook defaults)
BOUNDS = None
# ------------------

ROOT = Path.cwd()
if not (ROOT / "scripts" / "examples" / "notebook_support.py").is_file():
    raise SystemExit("Run this notebook from the BRB-Calibration repo root.")
sys.path.insert(0, str(ROOT / "scripts" / "examples"))
from notebook_support import display_png, run_single_specimen  # noqa: E402
print("repo:", ROOT)
print("specimen:", SPECIMEN, "metric:", METRIC)"""
            ),
            cell_md("## Run calibration"),
            cell_code(
                """paths = run_single_specimen(
    SPECIMEN,
    optimize_params=OPTIMIZE_PARAMS,
    steel=STEEL,
    metric=METRIC,
    prepare_data=PREPARE_DATA,
    bounds=BOUNDS,
)
print("parameters:", paths["params"])
print("overlay:   ", paths["overlay"])"""
            ),
            cell_md("## Results"),
            cell_code(
                """import pandas as pd

params = pd.read_csv(paths["params"])
display(params.T)
display_png(paths["overlay"], width=640)"""
            ),
        ],
    )

    write_nb(
        "02_individual.ipynb",
        [
            cell_md(
                """# 02 — Individual calibration (2–3 specimens)

Optimize each listed specimen **separately** with one shared feature-parameter
configuration (same idea as `set_id_settings.csv` + `optimize_brb_mse.py`).

Default demo: `PC3SB`, `STF01`, `PC250`. Edit the **Parameters** cell, then run all.

Outputs: `results/notebooks/individual/` and overlays under
`results/plots/calibration/individual_optimize/notebook_individual/`."""
            ),
            cell_md("## Parameters (edit me)"),
            cell_code(
                """from pathlib import Path
import sys

# --- edit these ---
SPECIMENS = ["PC3SB", "STF01", "PC250"]  # keep to 2–3 for a quick demo
PREPARE_DATA = True

# One set_id row (mirrors config/calibration/set_id_settings.csv)
SET_ID_ROW = {
    "set_id": 1,
    "inherit_from_set": -999,
    "E": 29000,
    "b_p": "weighted_mean",
    "b_n": "weighted_mean",
    "R0": 20,
    "cR1": 0.8875,
    "cR2": 0.15,
    "a1": 0.04,
    "a2": 1.0,
    "a3": 0.04,
    "a4": 1.0,
    # free parameters in L-BFGS-B:
    "optimize_params": ["cR1", "cR2", "a1", "a3"],
    # feature loss: use L2 (set w_feat_l1=1, w_feat_l2=0 for L1)
    "w_feat_l2": 1,
    "w_feat_l1": 0,
    "w_energy_l2": 0,
    "w_energy_l1": 0,
    "w_unordered_binenv_l2": 0,
    "w_unordered_binenv_l1": 0,
    "use_amplitude_weights": True,
    "amplitude_weight_power": 2,
    "amplitude_weight_eps": 0.05,
}
# ------------------

ROOT = Path.cwd()
if not (ROOT / "scripts" / "examples" / "notebook_support.py").is_file():
    raise SystemExit("Run this notebook from the BRB-Calibration repo root.")
sys.path.insert(0, str(ROOT / "scripts" / "examples"))
from notebook_support import display_png, run_individual_specimens  # noqa: E402
print("specimens:", SPECIMENS)"""
            ),
            cell_md("## Run individual optimize + overlays"),
            cell_code(
                """paths = run_individual_specimens(
    SPECIMENS,
    set_id_row=SET_ID_ROW,
    prepare_data=PREPARE_DATA,
)
print("params:  ", paths["params"])
print("overlays:", paths["overlays_dir"])"""
            ),
            cell_md("## Results"),
            cell_code(
                """import pandas as pd

params = pd.read_csv(paths["params"])
display(params)

sid = int(SET_ID_ROW["set_id"])
for name in SPECIMENS:
    png = paths["overlays_dir"] / f"{name}_set{sid}_force_def_norm.png"
    print(name, "->", png)
    display_png(png, width=520)"""
            ),
        ],
    )

    write_nb(
        "03_generalized.ipynb",
        [
            cell_md(
                """# 03 — Generalized (shared-parameter) calibration

One shared SteelMPF vector fit jointly over a small train list
(same idea as `optimize_generalized_brb_mse.py`).

Train Names should have `generalized_weight > 0` in
`config/calibration/BRB-Specimens.csv`. Default demo: `PC250`, `PC350`, `PC3SB`.

Edit the **Parameters** cell, then run all. Outputs: `results/notebooks/generalized/`."""
            ),
            cell_md("## Parameters (edit me)"),
            cell_code(
                """from pathlib import Path
import sys

# --- edit these ---
TRAIN_SPECIMENS = ["PC250", "PC350", "PC3SB"]  # keep small for a notebook demo
PREPARE_DATA = True

SET_ID_ROW = {
    "set_id": 1,
    "inherit_from_set": -999,
    "E": 29000,
    "b_p": 0.005,
    "b_n": 0.025,
    "R0": 20,
    "cR1": 0.8875,
    "cR2": 0.15,
    "a1": 0.04,
    "a2": 1.0,
    "a3": 0.04,
    "a4": 1.0,
    "optimize_params": ["b_p", "b_n", "cR1", "cR2", "a1", "a3"],
    "w_feat_l2": 1,
    "w_feat_l1": 0,
    "w_energy_l2": 0,
    "w_energy_l1": 0,
    "w_unordered_binenv_l2": 0,
    "w_unordered_binenv_l1": 0,
    "use_amplitude_weights": True,
    "amplitude_weight_power": 2,
    "amplitude_weight_eps": 0.05,
}
# ------------------

ROOT = Path.cwd()
if not (ROOT / "scripts" / "examples" / "notebook_support.py").is_file():
    raise SystemExit("Run this notebook from the BRB-Calibration repo root.")
sys.path.insert(0, str(ROOT / "scripts" / "examples"))
from notebook_support import display_png, run_generalized_demo  # noqa: E402
print("train:", TRAIN_SPECIMENS)"""
            ),
            cell_md("## Run generalized optimize + overlays"),
            cell_code(
                """paths = run_generalized_demo(
    TRAIN_SPECIMENS,
    set_id_row=SET_ID_ROW,
    prepare_data=PREPARE_DATA,
)
print("params: ", paths["params"])
print("metrics:", paths["metrics"])
print("plots:  ", paths["plots_dir"])"""
            ),
            cell_md("## Results"),
            cell_code(
                """import pandas as pd

params = pd.read_csv(paths["params"])
metrics = pd.read_csv(paths["metrics"])
display(params.head(12))
cols = ["Name", "set_id", "final_J_total", "final_J_feat_raw"]
display(metrics[cols].head(12) if set(cols).issubset(metrics.columns) else metrics.head(12))

sid = int(SET_ID_ROW["set_id"])
combined = paths["plots_dir"] / f"config_set_{sid}" / f"set{sid}_combined_force_def_norm.png"
if combined.is_file():
    display_png(combined, width=900)
else:
    for name in TRAIN_SPECIMENS:
        cands = list(paths["plots_dir"].rglob(f"*{name}*force_def_norm.png"))
        if cands:
            print(name, "->", cands[0])
            display_png(cands[0], width=520)
        else:
            print(name, ": no overlay found under", paths["plots_dir"])"""
            ),
        ],
    )

    (NB_DIR / "README.md").write_text(
        """# Notebooks

Walkthrough notebooks for the three calibration modes. **Run from the repo root**
(so `scripts/` and `config/` resolve). Use a kernel with working OpenSeesPy
(`$env:PYTHON` / `export PYTHON=` if needed).

| Notebook | What it does |
|----------|----------------|
| [`01_single_specimen.ipynb`](01_single_specimen.ipynb) | One specimen; edit free params / seeds / L1 vs L2 |
| [`02_individual.ipynb`](02_individual.ipynb) | 2–3 specimens, each fit separately |
| [`03_generalized.ipynb`](03_generalized.ipynb) | Shared SteelMPF vector on a small train list |

Helpers live in [`scripts/examples/notebook_support.py`](../scripts/examples/notebook_support.py).
Notebook outputs go to `results/notebooks/` (gitignored).

These call the same calibrate scripts as the CLI / `run.ps1` — edit the
**Parameters** cell, not optimizer internals.
""",
        encoding="utf-8",
    )
    print("wrote", NB_DIR / "README.md")


if __name__ == "__main__":
    main()
