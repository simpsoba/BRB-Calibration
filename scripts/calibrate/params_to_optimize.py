"""
SteelMPF BRB parameters updated by L-BFGS-B in optimize_brb_mse / optimize_generalized_brb_mse.

Import this module from lightweight scripts (e.g. report_calibration_param_tables) to avoid
pulling OpenSees via optimize_brb_mse.
"""
from __future__ import annotations

from calibrate.steel_model import STEELMPF_SIM_KEYS

# Default subset optimized by L-BFGS-B. Override per set_id via optimize_params in
# config/calibration/set_id_settings.csv.
PARAMS_TO_OPTIMIZE = ["R0", "cR1", "cR2", "a1", "a3"]

SIM_PARAMS_FROM_ROW: tuple[str, ...] = STEELMPF_SIM_KEYS

# When a parameters CSV omits optional columns, fill before simulation / optimization.
SIM_PARAM_FILL_DEFAULTS: dict[str, float] = {
    "E": 29000.0,
    "R0": 20.0,
    "cR1": 0.925,
    "cR2": 0.15,
    "a1": 0.04,
    "a2": 1.0,
    "a3": 0.04,
    "a4": 1.0,
}

PARAMS_IN_SUMMARY_TABLES: tuple[str, ...] = (
    "b_p",
    "b_n",
    *PARAMS_TO_OPTIMIZE,
    "a2",
    "a4",
)


def params_in_summary_tables_for_steel_model(_steel_model: object = None) -> list[str]:
    """Parameter columns for Markdown/CSV summaries (SteelMPF only)."""
    return list(PARAMS_IN_SUMMARY_TABLES)
