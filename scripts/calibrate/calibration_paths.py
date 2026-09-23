"""
Default locations under ``results/calibration`` and ``results/plots/calibration``,
plus calibration **input** CSVs under ``config/calibration/`` (catalog, seeds, limits, per-set settings)
and **summary** artifacts under ``summary_statistics/`` (parameter tables, per-``set_id`` eval rollups).

Import these instead of duplicating path strings across scripts.
"""
from __future__ import annotations

from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent

# Catalog, seeds, limits, and per-set settings (not under ``data/raw``).
CALIBRATION_CONFIG_DIR = _PROJECT_ROOT / "config" / "calibration"
BRB_SPECIMENS_CSV = CALIBRATION_CONFIG_DIR / "BRB-Specimens.csv"
PARAM_LIMITS_CSV = CALIBRATION_CONFIG_DIR / "params_limits.csv"
SET_ID_SETTINGS_CSV = CALIBRATION_CONFIG_DIR / "set_id_settings.csv"
# Per-``set_id`` optimize_params, loss weights, and **numeric** steel seeds (incl. ``b_p``/``b_n``) for
# ``optimize_generalized_brb_mse`` only (no apparent-``b`` statistic keywords in ``b_p``/``b_n`` cells).
SET_ID_SETTINGS_GENERALIZED_CSV = CALIBRATION_CONFIG_DIR / "set_id_settings_generalized.csv"

# Human-readable / tabular summaries (not raw inputs or per-run metrics blobs).
SUMMARY_STATISTICS_DIR = _PROJECT_ROOT / "summary_statistics"
# Default Markdown/CSV stems for SteelMPF calibration summaries.
CALIBRATION_PARAMETER_SUMMARY_MD = SUMMARY_STATISTICS_DIR / "calibration_parameter_summary.md"
CALIBRATION_PARAMETER_SUMMARY_GENERALIZED_CSV = (
    SUMMARY_STATISTICS_DIR / "calibration_parameter_summary_generalized.csv"
)
CALIBRATION_PARAMETER_SUMMARY_INDIVIDUAL_CSV = (
    SUMMARY_STATISTICS_DIR / "calibration_parameter_summary_individual.csv"
)
GENERALIZED_SET_ID_EVAL_SUMMARY_CSV = SUMMARY_STATISTICS_DIR / "generalized_set_id_eval_summary.csv"
GENERALIZED_UNORDERED_J_SUMMARY_TRAIN_CSV = (
    SUMMARY_STATISTICS_DIR / "generalized_unordered_J_binenv_summary_train.csv"
)
GENERALIZED_UNORDERED_J_SUMMARY_VALIDATION_CSV = (
    SUMMARY_STATISTICS_DIR / "generalized_unordered_J_binenv_summary_validation.csv"
)

RESULTS_CALIBRATION = _PROJECT_ROOT / "results" / "calibration"
INDIVIDUAL_OPTIMIZE_DIR = RESULTS_CALIBRATION / "individual_optimize"
GENERALIZED_OPTIMIZE_DIR = RESULTS_CALIBRATION / "generalized_optimize"

INITIAL_BRB_PARAMETERS_PATH = INDIVIDUAL_OPTIMIZE_DIR / "initial_brb_parameters.csv"
OPTIMIZED_BRB_PARAMETERS_PATH = INDIVIDUAL_OPTIMIZE_DIR / "optimized_brb_parameters.csv"
OPTIMIZED_BRB_PARAMETERS_METRICS_PATH = INDIVIDUAL_OPTIMIZE_DIR / "optimized_brb_parameters_metrics.csv"
BEST_L2_L1_OVERLAY_METRICS_TABLE_PATH = (
    INDIVIDUAL_OPTIMIZE_DIR / "bestL2_bestL1_metrics_table.csv"
)

SPECIMEN_APPARENT_BN_BP_PATH = RESULTS_CALIBRATION / "specimen_apparent_bn_bp.csv"

GENERALIZED_BRB_PARAMETERS_PATH = GENERALIZED_OPTIMIZE_DIR / "generalized_brb_parameters.csv"
GENERALIZED_PARAMS_EVAL_METRICS_PATH = GENERALIZED_OPTIMIZE_DIR / "generalized_params_eval_metrics.csv"

# Per-specimen hysteresis artifacts: ``{Name}_set{k}_simulated.csv`` (``Deformation[in],Force[kip],Force_sim[kip]``).
# Stem matches the metrics/parameters CSV stem used when the eval/optimize run wrote histories.
INDIVIDUAL_SIMULATED_FORCE_DIR = INDIVIDUAL_OPTIMIZE_DIR / "optimized_brb_parameters_simulated_force"
# Initial-BRB preset overlays (``plot_preset_overlays.py`` reads ``initial_brb_parameters.csv``): sim CSVs.
INITIAL_PARAMS_SIMULATED_FORCE_DIR = INDIVIDUAL_OPTIMIZE_DIR / "initial_params_simulated_force"
INITIAL_PARAMS_SIMULATED_FORCE_ALL_SPECIMENS_DIR = (
    INDIVIDUAL_OPTIMIZE_DIR / "initial_params_simulated_force_all_specimens"
)
GENERALIZED_SIMULATED_FORCE_DIR = GENERALIZED_OPTIMIZE_DIR / "generalized_params_eval_metrics_simulated_force"
# ``plot_generalized_train_mean_bn_bp_overlays.py``: same generalized steel per set_id, b_p/b_n = train-only weighted mean.
GENERALIZED_TRAIN_MEAN_BN_BP_SIMULATED_FORCE_DIR = (
    GENERALIZED_OPTIMIZE_DIR / "generalized_train_mean_bn_bp_simulated_force"
)

PLOTS_CALIBRATION = _PROJECT_ROOT / "results" / "plots" / "calibration"
PLOTS_INDIVIDUAL_OPTIMIZE = PLOTS_CALIBRATION / "individual_optimize"
PLOTS_INITIAL_PARAMS_OVERLAYS = PLOTS_INDIVIDUAL_OPTIMIZE / "overlays_initial_params"
PLOTS_INITIAL_PARAMS_ALL_SPECIMENS_OVERLAYS = (
    PLOTS_INDIVIDUAL_OPTIMIZE / "overlays_initial_params_all_specimens"
)
PLOTS_GENERALIZED_OPTIMIZE = PLOTS_CALIBRATION / "generalized_optimize"
PLOTS_GENERALIZED_TRAIN_MEAN_BN_BP_OVERLAYS = PLOTS_GENERALIZED_OPTIMIZE / "overlays_train_mean_bn_bp"
