# Notebooks

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
