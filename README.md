# BRB-Calibration

Calibrate OpenSees **SteelMPF** parameters for buckling-restrained braces (BRBs)
from experimental hysteresis. Each brace is a corotational truss with a prescribed
displacement history; L-BFGS-B fits selected steel parameters so simulated force
matches the test:

```text
uniaxialMaterial SteelMPF $mattag $fyp $fyn $E0 $bp $bn $R0 $cR1 $cR2 <$a1 $a2 $a3 $a4>
```

**Stay in the clone** — scripts and notebooks expect the repo layout (`config/`,
`data/`, `scripts/`). Do not copy folders out.

## Install

```bash
git clone https://github.com/simpsoba/BRB-Calibration.git
cd BRB-Calibration
pip install -r requirements.txt
```

Requires Python 3 with OpenSeesPy, NumPy, Pandas, SciPy, Matplotlib.

```bash
python -c "import openseespy.opensees"
```

If that fails, use an environment where OpenSeesPy works and point tools at it
(`export PYTHON=…` / `$env:PYTHON = "…"`).

## Start with the notebooks

The intended way to explore and run calibrations is the Jupyter notebooks under
[`notebooks/`](notebooks/README.md). Open them from the **repo root** with a
kernel that has OpenSeesPy.

Each notebook begins by **browsing the specimen catalog** (table + raw hysteresis
grid), so you can pick a `Name` without digging through CSVs. Then edit the
**Parameters** cell and run the rest in order.

| Notebook | What you do |
|----------|-------------|
| [`00_steelmpf_parameters.ipynb`](notebooks/00_steelmpf_parameters.ipynb) | See what each SteelMPF parameter does (one-at-a-time sweeps) |
| [`01_single_specimen.ipynb`](notebooks/01_single_specimen.ipynb) | Prepare data, fit one specimen, inspect results / diagnostics |
| [`02_individual.ipynb`](notebooks/02_individual.ipynb) | Fit 2–3 specimens separately with one shared configuration |
| [`03_generalized.ipynb`](notebooks/03_generalized.ipynb) | Joint fit + validation (path-ordered and digitized); compare `J_binenv` |
| [`04_generalized_l2_l1.ipynb`](notebooks/04_generalized_l2_l1.ipynb) | Joint fit + validation; same idea as `03`, defaults run L2 and L1 together |

Suggested order: `00` → `01` → `02` → `03`. Notebook `04` is optional (edit Parameters like the others).

Notebook outputs land under `results/notebooks/` (gitignored). Helpers live in
[`scripts/examples/notebook_support.py`](scripts/examples/notebook_support.py).

For a plain script view of the numerical model (geometry and SteelMPF numbers at
the top of one file):

```bash
python scripts/examples/simple_brb_steelmpf.py
```

### What the notebooks let you change

Most settings you will change are in the **Parameters** cell of each notebook
(specimen names, seeds, free parameters, loss weights). The underlying catalog
and defaults still live in config CSVs if you need them:

| Change | File |
|--------|------|
| Specimens, geometry, layout flags | [`config/calibration/BRB-Specimens.csv`](config/calibration/BRB-Specimens.csv) |
| Parameter bounds | [`config/calibration/params_limits.csv`](config/calibration/params_limits.csv) |

Raw experiments: `data/raw/{Name}/`.

### Naming

Symbols used in simulation, the specimen catalog, and parameter CSVs (US customary).
SteelMPF roles match the notebook `00` / paper write-up. Units live here, not in
column-name suffixes.

| Symbol | Meaning | Unit |
|--------|---------|------|
| `L_T` | Total brace length | in |
| `L_y` | Yielding (core) length | in |
| `A_sc` | Steel core area | in² |
| `A_t` | Transition-segment area | in² |
| `fyp`, `fyn` | Yield strength in tension (+) / compression (−) | ksi |
| `E` | Initial modulus (brace model uses `E_hat = Q·E`) | ksi |
| `Q` | Brace stiffness modification factor | — |
| `b_p`, `b_n` | Strain-hardening ratio in tension / compression | — |
| `R0` | Initial curvature of the elastic ↔ post-yield transition (Bauschinger rounding; ≈20 typical) | — |
| `cR1`, `cR2` | How that transition curvature `R` degrades with plastic excursion (≈0.925 and ≈0.15 typical) | — |
| `a1` | **Compression** isotropic shift scale: shifts the compression yield envelope by a proportion of `fyn` | — |
| `a2` | Compression isotropic threshold: shift activates after a max plastic **tensile** strain of about `a2 · fyp / E_hat` | — |
| `a3` | **Tension** isotropic shift scale: shifts the tension yield envelope by a proportion of `fyp` | — |
| `a4` | Tension isotropic threshold: shift activates after a max plastic **compressive** strain of about `a4 · fyn / E_hat` | — |

The catalog stores a single yield `fyp` (used for both `fyp` and `fyn` unless a row
also has `fyn`). `Q` is computed from `L_T`, `L_y`, `A_sc`, `A_t`.

Displacement histories and forces in the CSVs are inches and kips.

## Automated workflow

The same calibration ideas can be run in batch over the catalog without Jupyter.
Use this when you want the full pipeline: postprocess → individual fits →
generalized fit → summary reports.

Notebooks and the automated path share the same helpers. Notebook demos write
under `results/notebooks/`; the automated path writes under
`results/calibration/` and `results/plots/`.

### How to run it

```bash
# macOS / Linux
chmod +x run.sh clean_outputs.sh   # once, if needed
export PYTHON=/path/to/python      # if the default python lacks OpenSeesPy
./run.sh

# Windows PowerShell
$env:PYTHON = "C:\path\to\python.exe"   # optional
.\run.ps1
```

Optional log mirror: `$env:PIPELINE_LOG = "pipeline_log.txt"` then `.\run.ps1`.

Config files the pipeline reads:

| Change | File |
|--------|------|
| Specimens / who trains | [`config/calibration/BRB-Specimens.csv`](config/calibration/BRB-Specimens.csv) |
| Individual seeds, loss, free params | [`config/calibration/set_id_settings.csv`](config/calibration/set_id_settings.csv) |
| Generalized seeds / loss | [`config/calibration/set_id_settings_generalized.csv`](config/calibration/set_id_settings_generalized.csv) |
| Bounds | [`config/calibration/params_limits.csv`](config/calibration/params_limits.csv) |
| Single-specimen CLI defaults | [`scripts/calibrate_single/input.csv`](scripts/calibrate_single/input.csv) |

You can also run pieces alone (after postprocess / initial params exist), for
example:

```bash
# One specimen via the single-specimen entry point
python scripts/calibrate_single/calibrate_one_specimen.py STF01 --prepare-data --metric l2

# One specimen from the individual optimizer
python scripts/calibrate/optimize_brb_mse.py --specimen STF01
```

### What it produces

| Stage | Main outputs |
|-------|----------------|
| Postprocess | `data/filtered/`, `data/resampled/`, plots under `results/plots/postprocess/` |
| Individual | `results/calibration/individual_optimize/optimized_brb_parameters.csv` (+ metrics); overlays under `results/plots/calibration/individual_optimize/` |
| Generalized | `results/calibration/generalized_optimize/generalized_brb_parameters.csv` (+ metrics / overlays) |
| Reports | `summary_statistics/`, `results/calibration/calibration_individual_generalized_report.md` |

Single-specimen CLI results (if you use that entry point) go under
`results/calibration/single_specimen/{Name}/`.

### How to interpret results

The objective is a weighted sum of cycle **feature** mismatch (`J_feat`, L2
and/or L1 on characteristic points) plus optional energy and digitized-envelope
(`J_binenv`) terms. Weights come from the settings CSVs (or `--metric` for the
single-specimen CLI).

- **`final_J_total`** — weighted objective after optimization (primary score;
  lower is better). Compare `set_id` rows with this.
- **`final_J_feat_raw` / `final_J_feat_l1_raw`** — landmark feature errors.
- **`J_binenv`** — binned force-envelope match (especially useful for digitized
  specimens; also reported for path-ordered tests).
- **Overlay PNGs** — experiment vs simulation on the same axes.

### Clean regenerable outputs

```bash
./clean_outputs.sh    # macOS / Linux
# .\clean_outputs.ps1 # Windows
```

Preserves `data/raw/` and `config/calibration/`.

## Layout

```text
config/calibration/   # Specimens, set_id settings, bounds
config/figure.mplstyle
data/raw/             # Experimental CSVs
notebooks/            # Primary walkthroughs (00–04)
scripts/model/        # SteelMPF + corotational truss
scripts/postprocess/  # Filter / resample
scripts/calibrate/    # Individual + generalized optimization
scripts/calibrate_single/  # One-specimen CLI entry point
scripts/examples/     # Notebook helpers + simple BRB script
results/              # Generated (gitignored)
```
