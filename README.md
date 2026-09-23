# BRB-Calibration

Calibrate OpenSees **SteelMPF** parameters for buckling-restrained braces (BRBs) from experimental hysteresis. Each brace is a corotational truss driven by a fixed displacement history; L-BFGS-B fits selected steel parameters so simulated force matches the test.

This repo supports component-level calibration used in modeling mass-timber rocking/pivoting walls with BRB boundary elements. The material is stock OpenSees:

```text
uniaxialMaterial SteelMPF $mattag $fyp $fyn $E0 $bp $bn $R0 $cR1 $cR2 <$a1 $a2 $a3 $a4>
```

## Install

```bash
git clone https://github.com/gaaraujo/BRB-Calibration.git
cd BRB-Calibration
pip install -r requirements.txt
```

Requires Python 3 with OpenSeesPy, NumPy, Pandas, SciPy, Matplotlib.

Check OpenSeesPy once:

```bash
python -c "import openseespy.opensees"
```

If that fails, use an environment with a working OpenSeesPy build, then point the runners at it:

```bash
# macOS / Linux
export PYTHON=/path/to/python
./run.sh

# Windows PowerShell
$env:PYTHON = "C:\path\to\python.exe"
.\run.ps1
```

## Where to start

**Do not copy folders out of the repo.** Scripts expect the repo layout (`config/`, `data/`, `scripts/`). Stay in the clone and use the commands below.

If you only want parameters for **one** test specimen, start with §1 (single specimen). That is the smallest end-to-end path: edit one CSV, run one command, read one params file and one overlay plot.

To see the numerical model in plain, editable form (geometry and SteelMPF numbers at the top of one file):

```bash
python scripts/examples/simple_brb_steelmpf.py
```

## What to edit

| Change | File |
|--------|------|
| Specimens, geometry, train flags | [`config/calibration/BRB-Specimens.csv`](config/calibration/BRB-Specimens.csv) |
| Cohort seeds, loss weights, `optimize_params` | [`config/calibration/set_id_settings.csv`](config/calibration/set_id_settings.csv) |
| Generalized (group) seeds / loss | [`config/calibration/set_id_settings_generalized.csv`](config/calibration/set_id_settings_generalized.csv) |
| Parameter bounds | [`config/calibration/params_limits.csv`](config/calibration/params_limits.csv) |
| Single-specimen seeds / loss / bounds | [`scripts/calibrate_single/input.csv`](scripts/calibrate_single/input.csv) |

Raw experiments live under `data/raw/{Name}/`. Do not edit optimizer `.py` files for routine knobs.

**Naming (use these everywhere):** simulation / CSV steel columns are `L_T`, `L_y`, `A_sc`, `A_t`, `fyp`, `fyn`, `E`, `b_p`, `b_n`, `R0`, `cR1`, `cR2`, `a1`–`a4`, and stiffness factor `Q` (`E_hat = Q*E`). The specimen catalog uses `L_T_in`, `L_y_in`, `A_c_in2` (→ `A_sc`), `A_t_in2`, `f_yc_ksi` (→ `fyp`/`fyn`).

## 1. Single specimen

Fit one brace; print best-fit SteelMPF parameters; write an experiment-vs-simulation plot.

```bash
python scripts/calibrate_single/calibrate_one_specimen.py STF01 --prepare-data --metric l2
```

- `--metric l1` or `l2` sets the feature-loss norm (overrides feature weights in `input.csv`).
- `--prepare-data` builds filtered/resampled files from `data/raw/{Name}/` when needed.
- `--debug-plots` adds optional diagnostic figures.

**Results**

| Output | Path |
|--------|------|
| Parameters | `results/calibration/single_specimen/{Name}/parameters.csv` |
| Metrics | `.../parameters_metrics.csv` (`final_J_total`, `final_J_feat_raw`, …) |
| Overlay (normalized) | `results/plots/calibration/single_specimen/{Name}/{Name}_set1_force_def_norm.png` |

## 2. Individual calibration (many specimens)

Optimize each training specimen separately (`individual_optimize` in the catalog).

```bash
# Full pipeline (postprocess → individual → generalized → reports)
chmod +x run.sh clean_outputs.sh   # once, on macOS/Linux if needed
./run.sh          # macOS / Linux
# .\run.ps1       # Windows PowerShell

# Or one / several specimens after postprocess + initial parameters exist:
python scripts/calibrate/optimize_brb_mse.py --specimen STF01
python scripts/calibrate/plot_params_vs_filtered.py --specimen STF01 \
  --params results/calibration/individual_optimize/optimized_brb_parameters.csv
```

**Results**

| Output | Path |
|--------|------|
| Parameters | `results/calibration/individual_optimize/optimized_brb_parameters.csv` |
| Metrics | `.../optimized_brb_parameters_metrics.csv` |
| Overlays | `results/plots/calibration/individual_optimize/overlays/` |
| Best L2 vs L1 montage | `results/plots/calibration/individual_optimize/overlays_best_l1_l2/steelmpf/all_bestL2_bestL1_force_def_norm.png` |

Compare `set_id` rows using `final_J_total` (lower is better). Seeds and which parameters are free are controlled in `set_id_settings.csv`.

## 3. Generalized (group) calibration

One shared SteelMPF vector over specimens with positive `generalized_weight` in the catalog.

```bash
python scripts/calibrate/optimize_generalized_brb_mse.py \
  --output-params results/calibration/generalized_optimize/generalized_brb_parameters.csv \
  --output-metrics results/calibration/generalized_optimize/generalized_params_eval_metrics.csv \
  --output-plots-dir results/plots/calibration/generalized_optimize/overlays
```

(Also run automatically at the end of `run.sh` / `run.ps1`.)

**Results:** same table pattern under `results/calibration/generalized_optimize/` and overlays under `results/plots/calibration/generalized_optimize/overlays/`. A short comparison of individual vs generalized is written to `results/calibration/calibration_individual_generalized_report.md`.

## How to read metrics

The objective combines cycle **feature** mismatch (`J_feat`, L2 and/or L1 on characteristic points) with optional energy and digitized-envelope terms. Weights come from the settings CSV (or `--metric` for single-specimen feature L1/L2).

- `final_J_total` — weighted objective after optimization (primary score).
- `final_J_feat_raw` / `final_J_feat_l1_raw` — landmark feature errors.
- Overlay PNGs — experiment vs simulation on the same axes.

## Optional: Bayesian / quoFEM (`bayesian/`)

Standalone helper for uncertainty quantification on specimen **CB225** with quoFEM/TMCMC. See [`bayesian/README.md`](bayesian/README.md). Not required for deterministic L-BFGS calibration above.

## Layout

```text
config/calibration/   # Specimens, set_id settings, bounds  ← edit these
config/figure.mplstyle
data/raw/             # Experimental CSVs
scripts/model/        # SteelMPF + corotational truss
scripts/postprocess/  # Filter / resample
scripts/calibrate/    # Individual + generalized optimization
scripts/calibrate_single/  # One-specimen entry point
bayesian/             # Optional UQ
results/              # Generated (gitignored)
```

## Clean regenerable outputs

```bash
./clean_outputs.sh    # macOS / Linux
# .\clean_outputs.ps1 # Windows
```

Preserves `data/raw/` and `config/calibration/`.

## Related work

Component calibration here supports numerical modeling of mass plywood panel rocking/pivoting walls with BRB boundary elements (manuscript shared separately via Overleaf; local `paper/` is gitignored).
