# Notebooks

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
