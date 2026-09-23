"""
Combined normalized **initial** BRB overlays (``set{k}_combined_force_def_norm.png``) before
individual L-BFGS.

Reads ``results/calibration/individual_optimize/initial_brb_parameters.csv`` (default; override with
``--params``). Produces **two** separate montage sets when ``--scope both`` (default):

1. **Training cohort** — ``individual_optimize=true`` and path-ordered resampled data only.
   Simulated CSVs: ``initial_params_simulated_force/``. PNGs: ``overlays_initial_params/``.

2. **All specimens** — every path-ordered row in the parameters table (any ``individual_optimize``)
   plus digitized-unordered specimens in the parameters table.
   Simulated CSVs: ``initial_params_simulated_force_all_specimens/``.
   PNGs: ``overlays_initial_params_all_specimens/``.

Typical::

    python scripts/calibrate/plot_preset_overlays.py

Requires: ``build_initial_brb_parameters.py``, resampled path-ordered data where applicable, and
digitized inputs for unordered specimens.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
_SCRIPTS = _PROJECT_ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from calibrate.calibration_paths import (  # noqa: E402
    BRB_SPECIMENS_CSV,
    INITIAL_BRB_PARAMETERS_PATH,
    INITIAL_PARAMS_SIMULATED_FORCE_ALL_SPECIMENS_DIR,
    INITIAL_PARAMS_SIMULATED_FORCE_DIR,
    PLOTS_INDIVIDUAL_OPTIMIZE,
    PLOTS_INITIAL_PARAMS_ALL_SPECIMENS_OVERLAYS,
    PLOTS_INITIAL_PARAMS_OVERLAYS,
)
from calibrate.plot_compare_calibration_overlays import (  # noqa: E402
    _discover_simulated_index,
    _order_specimens,
    plot_combined_for_set,
)
from calibrate.plot_params_vs_filtered import run_multi_specimen_simulated_csvs  # noqa: E402
from specimen_catalog import read_catalog  # noqa: E402


def _run_one_preset_batch(
    *,
    banner: str,
    params_df: pd.DataFrame,
    params_path: Path,
    sim_dir: Path,
    plots_dir: Path,
    catalog: pd.DataFrame,
    catalog_names: list[str],
    specimen: str | None,
    require_individual_optimize: bool,
    include_digitized_unordered: bool,
) -> bool:
    print(banner)
    if sim_dir.is_dir():
        for f in sim_dir.glob("*_set*_simulated.csv"):
            try:
                f.unlink()
            except OSError:
                pass

    run_multi_specimen_simulated_csvs(
        params_df,
        sim_dir,
        params_path_label=params_path,
        specimen=specimen,
        override_b_p=None,
        override_b_n=None,
        require_individual_optimize=require_individual_optimize,
        include_digitized_unordered=include_digitized_unordered,
    )

    idx = _discover_simulated_index(sim_dir)
    if not idx:
        print(f"No *_simulated.csv under {sim_dir}; combined figures skipped for this batch.")
        return False

    any_written = False
    for set_id in sorted(idx.keys()):
        specimen_names = _order_specimens(idx[set_id], catalog_names)
        if plot_combined_for_set(
            set_id,
            specimen_names,
            plots_dir,
            params_path,
            sim_dir,
            catalog,
            grid_cols=3,
            stage_weight_fn=None,
        ):
            any_written = True
            print(
                f"Wrote {plots_dir / f'set{set_id}_combined_force_def_norm.png'} "
                f"({len(specimen_names)} specimens, 3-column grid)"
            )

    if any_written:
        print(f"Done this batch. PNGs under {plots_dir}")
    else:
        print("No combined figures written for this batch.")
    return any_written


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Initial-BRB combined overlays from initial_brb_parameters.csv. "
            "Default: training cohort and all-specimens batches (separate folders)."
        ),
    )
    p.add_argument(
        "--params",
        type=Path,
        default=INITIAL_BRB_PARAMETERS_PATH,
        help=f"Parameters CSV (default: {INITIAL_BRB_PARAMETERS_PATH}).",
    )
    p.add_argument(
        "--scope",
        choices=("both", "train", "all"),
        default="both",
        help=(
            "train: individual_optimize path-ordered only; "
            "all: every path-ordered + digitized unordered; "
            "both: write both separate overlay sets."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=PLOTS_INITIAL_PARAMS_OVERLAYS.name,
        help=(
            f"Subfolder for the **train** batch under {PLOTS_INDIVIDUAL_OPTIMIZE.relative_to(_PROJECT_ROOT)}/ "
            f"(default: {PLOTS_INITIAL_PARAMS_OVERLAYS.name!r}). "
            "The **all-specimens** batch always uses overlays_initial_params_all_specimens/."
        ),
    )
    p.add_argument(
        "--specimen",
        type=str,
        default=None,
        help="If set, only this specimen Name (path-ordered and/or digitized-unordered per batch).",
    )
    args = p.parse_args()

    params_path = Path(args.params).expanduser().resolve()
    if not params_path.is_file():
        raise SystemExit(
            f"Parameters CSV not found: {params_path}\n"
            "Run: python scripts/calibrate/build_initial_brb_parameters.py"
        )
    params_df = pd.read_csv(params_path)
    print(f"Parameters from {params_path} (per-row b_p, b_n and steel from CSV)")

    catalog = read_catalog(BRB_SPECIMENS_CSV)
    catalog_names = catalog["Name"].astype(str).tolist()

    train_plots_dir = PLOTS_INDIVIDUAL_OPTIMIZE / args.output_dir
    all_plots_dir = PLOTS_INITIAL_PARAMS_ALL_SPECIMENS_OVERLAYS

    if args.scope in ("both", "train"):
        _run_one_preset_batch(
            banner="--- Batch: individual_optimize training cohort (path-ordered) ---",
            params_df=params_df,
            params_path=params_path,
            sim_dir=INITIAL_PARAMS_SIMULATED_FORCE_DIR,
            plots_dir=train_plots_dir,
            catalog=catalog,
            catalog_names=catalog_names,
            specimen=args.specimen,
            require_individual_optimize=True,
            include_digitized_unordered=False,
        )

    if args.scope in ("both", "all"):
        _run_one_preset_batch(
            banner="--- Batch: all specimens (path-ordered + digitized unordered) ---",
            params_df=params_df,
            params_path=params_path,
            sim_dir=INITIAL_PARAMS_SIMULATED_FORCE_ALL_SPECIMENS_DIR,
            plots_dir=all_plots_dir,
            catalog=catalog,
            catalog_names=catalog_names,
            specimen=args.specimen,
            require_individual_optimize=False,
            include_digitized_unordered=True,
        )


if __name__ == "__main__":
    main()
