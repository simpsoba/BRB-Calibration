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

Helpers live in [`scripts/examples/notebook_support.py`](../scripts/examples/notebook_support.py).
Generated files go to `results/notebooks/` (gitignored).

Suggested order: start with `00` for intuition, then `01` → `02` → `03`.
In each calibration notebook: browse the catalog, read the objective section, edit
**Parameters**, run the cells in order. Results cells load experimental and
numerical force–displacement arrays so you can plot or tweak parameters yourself.
Notebook `01` also has a **Diagnostics** section (per-cycle characteristic
points, energy, apparent b). Notebook `03` reports train vs validation `J_feat` /
`J_binenv`.
