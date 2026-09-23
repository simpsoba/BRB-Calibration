"""
Calibrate one specimen with SteelMPF (seeds/bounds/loss in ``input.csv``).

Geometry and fy from ``config/calibration/BRB-Specimens.csv``. Default ``input.csv`` has one
``set_id``. From the repo root::

    python scripts/calibrate_single/calibrate_one_specimen.py STF01 --prepare-data --metric l2

Outputs: ``results/calibration/single_specimen/{Name}/parameters.csv`` and overlay PNGs under
``results/plots/calibration/single_specimen/{Name}/``. Use ``--debug-plots`` for apparent-b and
cycle diagnostics. ``--replot`` regenerates overlays from a saved ``parameters.csv``.
"""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
_SCRIPTS = _PROJECT_ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS / "postprocess"))
sys.path.insert(0, str(_SCRIPT_DIR))

from calibrate.amplitude_mse_partition import (  # noqa: E402
    build_amplitude_weights,
    energy_scale_s_e,
)
from calibrate.calibration_io import metrics_dataframe  # noqa: E402
from calibrate.calibration_loss_settings import CalibrationLossSettings  # noqa: E402
from calibrate.cycle_feature_loss import (  # noqa: E402
    deformation_scale_s_d,
    load_p_y_kip_catalog,
)
from calibrate.build_initial_brb_parameters import resolve_b_arm_from_stats  # noqa: E402
from calibrate.extract_bn_bp import extract_bn_bp_one_specimen, get_b_lists_one_specimen  # noqa: E402
from calibrate.optimize_brb_mse import (  # noqa: E402
    DEBUG_PARTITION,
    FAILURE_PENALTY,
    _loss_weight_snapshot,
    _metrics_dict_for_breakdown,
    force_scale_s_f,
    optimize_one_specimen,
    plot_cycle_weight_hysteresis,
    save_simulated_force_history,
    save_simulated_force_history_csv,
)
from calibrate.plot_params_vs_filtered import run_one_specimen  # noqa: E402
from calibrate.specimen_weights import catalog_metrics_fields  # noqa: E402
from cycle_points import find_cycle_points, load_cycle_points_resampled, run_specimen  # noqa: E402
from filter_force import (  # noqa: E402
    _process_digitized_unordered,
    _process_path_ordered,
)
from load_input import (  # noqa: E402
    FALLBACK_B_N,
    FALLBACK_B_P,
    STEEL_MODEL,
    SingleCalibrateInput,
    load_single_calibrate_inputs,
)
from plot_all_sets_overlays import plot_all_sets_force_def_grid  # noqa: E402
from resample_filtered import (  # noqa: E402
    process_specimen,
    process_specimen_digitized_unordered,
)
from specimen_catalog import (  # noqa: E402
    get_specimen_record,
    list_names_digitized_unordered,
    read_catalog,
    resampled_force_deformation_csv,
    resolve_resampled_force_deformation_csv,
    uses_unordered_inputs,
)

DEFAULT_INPUT = _SCRIPT_DIR / "input.csv"
RESULTS_SINGLE = _PROJECT_ROOT / "results" / "calibration" / "single_specimen"
PLOTS_SINGLE = _PROJECT_ROOT / "results" / "plots" / "calibration" / "single_specimen"


@dataclass(frozen=True)
class CalibrateRunContext:
    specimen_out: Path
    csv_path: Path
    cat_row: pd.Series
    overlay_dir: Path


def _format_b_spec(spec: float | str) -> str:
    if isinstance(spec, float):
        return f"{spec:g}"
    return str(spec)


def _has_apparent_b_stats(stats: dict, arm: str) -> bool:
    for suffix in (
        "median",
        "mean",
        "weighted_mean",
        "q1",
        "q3",
        "min",
        "max",
        "max_amplitude",
    ):
        v = stats.get(f"b_{arm}_{suffix}")
        if v is None:
            continue
        try:
            if np.isfinite(float(v)):
                return True
        except (TypeError, ValueError):
            continue
    return False


def _resolve_b_seeds(
    specimen: str,
    stats: dict,
    cfg: SingleCalibrateInput,
) -> tuple[float, float]:
    if isinstance(cfg.b_p_spec, float):
        b_p = float(cfg.b_p_spec)
    elif not _has_apparent_b_stats(stats, "p"):
        print(
            f"  Warning: no apparent b_p stats for {specimen!r}; "
            f"using default {FALLBACK_B_P:g}"
        )
        b_p = FALLBACK_B_P
    else:
        b_p = resolve_b_arm_from_stats(
            stats, arm="p", spec=cfg.b_p_spec, fallback=FALLBACK_B_P
        )

    if isinstance(cfg.b_n_spec, float):
        b_n = float(cfg.b_n_spec)
    elif not _has_apparent_b_stats(stats, "n"):
        print(
            f"  Warning: no apparent b_n stats for {specimen!r}; "
            f"using default {FALLBACK_B_N:g}"
        )
        b_n = FALLBACK_B_N
    else:
        b_n = resolve_b_arm_from_stats(
            stats, arm="n", spec=cfg.b_n_spec, fallback=FALLBACK_B_N
        )
    return b_p, b_n


def _plot_apparent_b(specimen: str, cat_row: pd.Series, *, plots_base: Path) -> None:
    """Write per-specimen apparent-b slope overlay and segment histogram under ``plots_base/apparent_b/``."""
    from calibrate.plot_b_histograms_and_scatter import plot_histogram_one_specimen  # noqa: E402
    from calibrate.plot_b_slopes import (  # noqa: E402
        plot_one_digitized_unordered,
        plot_one_specimen,
    )

    apparent_root = plots_base / "apparent_b"
    slopes_dir = apparent_root / "b_slopes"
    hist_dir = apparent_root / "b_histograms"

    catalog = read_catalog()
    rec = get_specimen_record(specimen, catalog)
    is_unordered = uses_unordered_inputs(rec) or specimen in list_names_digitized_unordered(catalog)

    if not is_unordered:
        plot_one_specimen(specimen, cat_row, slopes_dir)
        slopes_png = slopes_dir / f"{specimen}.png"
        if slopes_png.is_file():
            print(f"  Wrote apparent-b slopes: {slopes_png}")
        else:
            print(f"  Skipped apparent-b slopes (no resampled F-u for {specimen!r})")
    elif plot_one_digitized_unordered(specimen, cat_row, slopes_dir):
        print(f"  Wrote apparent-b slopes: {slopes_dir / f'{specimen}.png'}")
    else:
        print(f"  Skipped apparent-b slopes (no digitized envelope for {specimen!r})")

    b_n_list, b_p_list = get_b_lists_one_specimen(specimen)
    if b_n_list or b_p_list:
        plot_histogram_one_specimen(specimen, b_n_list, b_p_list, hist_dir)
        print(f"  Wrote apparent-b histogram: {hist_dir / f'{specimen}.png'}")
    else:
        print(f"  Skipped apparent-b histogram (no segment b values for {specimen!r})")


def _plot_cycle_debug(
    specimen: str,
    set_id: int,
    D_exp: np.ndarray,
    F_exp: np.ndarray,
    amp_meta: list[dict],
    pointwise_weights: np.ndarray,
    cat_row: pd.Series,
    *,
    plots_base: Path,
) -> None:
    """Cycle-weight hysteresis map (path-ordered specimens only)."""
    cycles_dir = plots_base / "cycles"
    cycles_dir.mkdir(parents=True, exist_ok=True)
    f_yc = float(cat_row["f_yc_ksi"])
    A_c = float(cat_row["A_c_in2"])
    L_y = float(cat_row["L_y_in"])

    plot_cycle_weight_hysteresis(
        specimen,
        set_id,
        D_exp,
        F_exp,
        pointwise_weights,
        amp_meta,
        cycles_dir,
        f_yc=f_yc,
        A_c=A_c,
        L_y=L_y,
    )
    print(f"  Wrote cycle weights: {cycles_dir / f'{specimen}_set{set_id}_cycle_weights.png'}")


def _parameter_row(
    specimen: str,
    cat_row: pd.Series,
    cfg: SingleCalibrateInput,
    *,
    b_p: float,
    b_n: float,
) -> pd.Series:
    fy = float(cat_row["f_yc_ksi"])
    row = {
        "ID": int(cat_row["ID"]),
        "Name": specimen,
        "set_id": cfg.set_id,
        "steel_model": STEEL_MODEL,
        "L_T": float(cat_row["L_T_in"]),
        "L_y": float(cat_row["L_y_in"]),
        "A_sc": float(cat_row["A_c_in2"]),
        "A_t": float(cat_row["A_t_in2"]),
        "fyp": fy,
        "fyn": fy,
        "b_p": b_p,
        "b_n": b_n,
        **cfg.steel_seeds,
    }
    return pd.Series(row)


def _bounds_for_active(
    active: list[str], cfg: SingleCalibrateInput
) -> dict[str, tuple[float, float]]:
    missing = [p for p in active if p not in cfg.param_bounds]
    if missing:
        raise ValueError(
            f"No bounds in input.csv for optimized parameters: {missing}"
        )
    return {p: cfg.param_bounds[p] for p in active}


def _validate_specimen(specimen: str) -> tuple[pd.DataFrame, pd.Series]:
    catalog = read_catalog()
    by_name = catalog.set_index("Name")
    if specimen not in by_name.index:
        raise SystemExit(
            f"Unknown specimen {specimen!r}. Add it to config/calibration/BRB-Specimens.csv."
        )
    return catalog, by_name.loc[specimen]


def _resolve_force_deformation_path(path: Path) -> Path:
    p = path.expanduser()
    if not p.is_absolute():
        p = (_PROJECT_ROOT / p).resolve()
    else:
        p = p.resolve()
    if not p.is_file():
        raise SystemExit(f"force_deformation CSV not found: {p}")
    return p


def _default_force_deformation_path(specimen: str, *, prepare_data: bool) -> Path:
    """Default F-u CSV: resampled layout (same as optimize_brb_mse / plot_params_vs_filtered)."""
    found = resolve_resampled_force_deformation_csv(specimen, _PROJECT_ROOT)
    if found is not None:
        return found.resolve()
    canonical = resampled_force_deformation_csv(specimen, _PROJECT_ROOT).resolve()
    if prepare_data:
        return canonical
    raise SystemExit(
        f"No resampled force_deformation.csv for {specimen!r} "
        f"(expected {canonical}). "
        "Run with --prepare-data or pass --force-deformation."
    )


def _load_force_deformation_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        raise SystemExit(f"Force-deformation CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "Force[kip]" not in df.columns or "Deformation[in]" not in df.columns:
        raise SystemExit(f"{csv_path}: missing Force[kip] or Deformation[in] columns")
    return df


def _prepare_specimen_data(specimen: str, *, e_ksi: float) -> None:
    """Run cycle_points -> filter_force -> resample_filtered for one catalog specimen."""
    catalog = read_catalog()
    rec = get_specimen_record(specimen, catalog)
    catalog_by_name = catalog.set_index("Name")

    print(f"  Preparing data for {specimen!r}...")
    cp = run_specimen(specimen, save=True, overwrite=True)
    if cp is None:
        raise SystemExit(
            f"cycle_points failed for {specimen!r}: no valid raw F-u under data/raw/{specimen}/"
        )
    points, segments, wrote = cp
    print(
        f"    cycle_points: {len(points)} points, {len(segments)} segments"
        + (" (wrote JSON)" if wrote else " (JSON unchanged)")
    )

    if uses_unordered_inputs(rec):
        _process_digitized_unordered(specimen, catalog)
    else:
        _process_path_ordered(specimen, catalog)
    print("    filter_force: done")

    e_by_name = {specimen: e_ksi}
    if uses_unordered_inputs(rec):
        process_specimen_digitized_unordered(specimen, catalog_by_name, e_by_name, catalog)
    else:
        process_specimen(specimen, catalog_by_name, e_by_name, catalog)
    print("    resample_filtered: done")


def _cycle_points_for_csv(specimen: str, df: pd.DataFrame) -> list[dict]:
    loaded = load_cycle_points_resampled(specimen)
    if loaded is not None:
        points, _segments = loaded
        if len(points) > 0:
            return points
    return find_cycle_points(df)[0]


def _apply_metric_override(loss: CalibrationLossSettings, metric: str | None) -> CalibrationLossSettings:
    """Map ``--metric l1|l2`` onto feature loss weights (energy/binenv unchanged)."""
    if metric is None:
        return loss
    m = metric.strip().lower()
    if m == "l2":
        return CalibrationLossSettings(
            w_feat_l2=1.0,
            w_feat_l1=0.0,
            w_energy_l2=loss.w_energy_l2,
            w_energy_l1=loss.w_energy_l1,
            w_unordered_binenv_l2=loss.w_unordered_binenv_l2,
            w_unordered_binenv_l1=loss.w_unordered_binenv_l1,
            use_amplitude_weights=loss.use_amplitude_weights,
            amplitude_weight_power=loss.amplitude_weight_power,
            amplitude_weight_eps=loss.amplitude_weight_eps,
        )
    if m == "l1":
        return CalibrationLossSettings(
            w_feat_l2=0.0,
            w_feat_l1=1.0,
            w_energy_l2=loss.w_energy_l2,
            w_energy_l1=loss.w_energy_l1,
            w_unordered_binenv_l2=loss.w_unordered_binenv_l2,
            w_unordered_binenv_l1=loss.w_unordered_binenv_l1,
            use_amplitude_weights=loss.use_amplitude_weights,
            amplitude_weight_power=loss.amplitude_weight_power,
            amplitude_weight_eps=loss.amplitude_weight_eps,
        )
    raise ValueError(f"metric must be 'l1' or 'l2'; got {metric!r}")


def _print_optimized_params(out_row: pd.Series, active: list[str], jtot: float) -> None:
    keys = ["fyp", "fyn", "E", "b_p", "b_n", "R0", "cR1", "cR2", "a1", "a2", "a3", "a4"]
    parts = []
    for k in keys:
        if k not in out_row.index:
            continue
        try:
            v = float(out_row[k])
        except (TypeError, ValueError):
            continue
        mark = "*" if k in active else ""
        parts.append(f"{k}{mark}={v:.6g}")
    print(f"  Best-fit SteelMPF (*=optimized):  {', '.join(parts)}")
    print(f"  final_J_total = {jtot:.6g}")


def calibrate_and_plot(
    specimen: str,
    force_deformation_csv: Path,
    cfg: SingleCalibrateInput,
    *,
    prepare_data: bool = False,
    plot_apparent_b: bool = False,
    debug_plots: bool = False,
    metric: str | None = None,
    out_dir: Path | None = None,
    plots_dir: Path | None = None,
    use_amplitude_weights: bool | None = None,
    param_rows_out: list[pd.Series] | None = None,
    metrics_rows_out: list[dict] | None = None,
) -> CalibrateRunContext:
    catalog, cat_row = _validate_specimen(specimen)

    # Prepare first so --prepare-data works from a clean data/resampled tree.
    if prepare_data:
        _prepare_specimen_data(specimen, e_ksi=float(cfg.steel_seeds["E"]))
    csv_path = _resolve_force_deformation_path(force_deformation_csv)

    df = _load_force_deformation_csv(csv_path)
    print(f"  Using force-deformation: {csv_path}")
    print(f"  Input settings: set_id={cfg.set_id}, steel_model={STEEL_MODEL}")

    D_exp = df["Deformation[in]"].to_numpy(dtype=float)
    F_exp = df["Force[kip]"].to_numpy(dtype=float)
    points = _cycle_points_for_csv(specimen, df)

    b_stats = extract_bn_bp_one_specimen(
        specimen,
        df,
        points,
        float(cat_row["L_T_in"]),
        float(cat_row["L_y_in"]),
        float(cat_row["A_c_in2"]),
        float(cat_row["A_t_in2"]),
        float(cat_row["f_yc_ksi"]),
    )
    b_p, b_n = _resolve_b_seeds(specimen, b_stats, cfg)
    print(
        f"  Apparent b seeds: b_p={b_p:.6g} ({_format_b_spec(cfg.b_p_spec)}), "
        f"b_n={b_n:.6g} ({_format_b_spec(cfg.b_n_spec)})"
    )

    overlay_dir = plots_dir or (PLOTS_SINGLE / specimen)
    if plot_apparent_b and debug_plots:
        _plot_apparent_b(specimen, cat_row, plots_base=overlay_dir)

    prow = _parameter_row(specimen, cat_row, cfg, b_p=b_p, b_n=b_n)
    loss = _apply_metric_override(cfg.loss, metric)
    if use_amplitude_weights is not None:
        loss = CalibrationLossSettings(
            w_feat_l2=loss.w_feat_l2,
            w_feat_l1=loss.w_feat_l1,
            w_energy_l2=loss.w_energy_l2,
            w_energy_l1=loss.w_energy_l1,
            w_unordered_binenv_l2=loss.w_unordered_binenv_l2,
            w_unordered_binenv_l1=loss.w_unordered_binenv_l1,
            use_amplitude_weights=use_amplitude_weights,
            amplitude_weight_power=loss.amplitude_weight_power,
            amplitude_weight_eps=loss.amplitude_weight_eps,
        )

    active = list(cfg.optimize_params)
    bounds = _bounds_for_active(active, cfg)

    use_amp_w = loss.use_amplitude_weights
    mse_weights, amp_meta = build_amplitude_weights(
        D_exp,
        points,
        p=loss.amplitude_weight_power,
        eps=loss.amplitude_weight_eps,
        debug_partition=DEBUG_PARTITION,
        use_amplitude_weights=use_amp_w,
    )

    s_f_ref = force_scale_s_f(F_exp)
    s_d_ref = deformation_scale_s_d(D_exp)
    s_e_ref = energy_scale_s_e(D_exp, F_exp)
    p_y_ref = load_p_y_kip_catalog(
        _PROJECT_ROOT,
        specimen,
        float(prow["fyp"]),
        float(prow["A_sc"]),
    )

    print(f"  Optimizing: {', '.join(active)}")
    out_row, bd_initial, bd_final, _F0, F_sim_final = optimize_one_specimen(
        specimen,
        prow,
        D_exp,
        F_exp,
        amp_meta,
        active,
        bounds,
        p_y_ref=p_y_ref,
        s_d=s_d_ref,
        loss=loss,
    )

    if bd_final is None:
        raise SystemExit(f"Optimization failed for {specimen!r} (simulation or loss breakdown).")

    specimen_out = out_dir or (RESULTS_SINGLE / specimen)
    specimen_out.mkdir(parents=True, exist_ok=True)
    params_path = specimen_out / "parameters.csv"

    sim_dir = specimen_out / "parameters_simulated_force"
    save_simulated_force_history(
        sim_dir, specimen, cfg.set_id, D_exp, F_exp, F_sim_final
    )
    save_simulated_force_history_csv(
        sim_dir, specimen, cfg.set_id, D_exp, F_exp, F_sim_final
    )

    catalog_by_name = catalog.set_index("Name")
    mi = _metrics_dict_for_breakdown(bd_initial, loss, "initial") if bd_initial else {}
    mf = _metrics_dict_for_breakdown(bd_final, loss, "final")
    metrics_record = {
        "Name": specimen,
        "set_id": cfg.set_id,
        "specimen_weight": 1.0,
        "contributes_to_aggregate": True,
        **catalog_metrics_fields(specimen, catalog_by_name),
        "weight_config": "single_specimen",
        "calibration_stage": "optimize",
        "aggregate_by_set_id": False,
        **mi,
        **mf,
        **_loss_weight_snapshot(loss),
        "S_F": s_f_ref,
        "S_D": s_d_ref,
        "S_E": s_e_ref,
        "P_y_ref": p_y_ref,
        "n_cycles": len(amp_meta),
        "success": mf["final_J_total"] < FAILURE_PENALTY * 0.5,
    }
    if param_rows_out is not None:
        param_rows_out.append(out_row)
    else:
        pd.DataFrame([out_row]).to_csv(params_path, index=False)

    if metrics_rows_out is not None:
        metrics_rows_out.append(metrics_record)
    else:
        metrics_dataframe([metrics_record]).to_csv(
            specimen_out / "parameters_metrics.csv", index=False
        )

    jtot = mf["final_J_total"]
    _print_optimized_params(out_row, active, jtot)

    overlay_dir.mkdir(parents=True, exist_ok=True)

    if debug_plots:
        catalog = read_catalog()
        if not uses_unordered_inputs(get_specimen_record(specimen, catalog)):
            _plot_cycle_debug(
                specimen,
                int(cfg.set_id),
                D_exp,
                F_exp,
                amp_meta,
                mse_weights,
                cat_row,
                plots_base=overlay_dir,
            )
        else:
            print(f"  Skipped cycle debug (digitized unordered specimen {specimen!r})")

    return CalibrateRunContext(
        specimen_out=specimen_out,
        csv_path=csv_path,
        cat_row=cat_row,
        overlay_dir=overlay_dir,
    )


def replot_from_saved(
    specimen: str,
    force_deformation_csv: Path,
    *,
    out_dir: Path | None = None,
    plots_dir: Path | None = None,
) -> CalibrateRunContext:
    """Regenerate overlays from ``parameters.csv`` (skip optimization)."""
    _catalog, cat_row = _validate_specimen(specimen)
    specimen_out = out_dir or (RESULTS_SINGLE / specimen)
    params_path = specimen_out / "parameters.csv"
    if not params_path.is_file():
        raise SystemExit(
            f"No saved parameters at {params_path}. "
            "Run calibration first or pass --out-dir to the results folder."
        )

    params_df = pd.read_csv(params_path)
    if params_df.empty:
        raise SystemExit(f"Empty parameters file: {params_path}")
    if "set_id" not in params_df.columns:
        raise SystemExit(f"{params_path}: missing set_id column")

    csv_path = _resolve_force_deformation_path(force_deformation_csv)
    _load_force_deformation_csv(csv_path)
    overlay_dir = plots_dir or (PLOTS_SINGLE / specimen)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    set_ids = params_df["set_id"].tolist()
    print(f"  Replot from {params_path} ({len(set_ids)} set_id(s): {set_ids})")
    print(f"  Using force-deformation: {csv_path}")

    run_one_specimen(
        specimen,
        params_df,
        cat_row,
        overlay_dir,
        norm_xy_half=None,
        override_b_p=None,
        override_b_n=None,
        force_deformation_csv=csv_path,
    )
    print(f"  Wrote overlays under {overlay_dir}")

    exp_df = pd.read_csv(csv_path)
    grid_paths = plot_all_sets_force_def_grid(
        specimen,
        params_df,
        cat_row,
        exp_df["Deformation[in]"].to_numpy(dtype=float),
        exp_df["Force[kip]"].to_numpy(dtype=float),
        overlay_dir,
    )
    for grid_path in grid_paths:
        print(f"  Wrote all-set overlay grid: {grid_path}")

    return CalibrateRunContext(
        specimen_out=specimen_out,
        csv_path=csv_path,
        cat_row=cat_row,
        overlay_dir=overlay_dir,
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Calibrate one specimen (SteelMPF) and write best-fit params + overlay plots. "
            "Edit scripts/calibrate_single/input.csv for seeds, bounds, and loss weights."
        ),
    )
    p.add_argument(
        "specimen",
        nargs="?",
        default=None,
        help="Specimen Name (e.g. STF01). Same as --specimen.",
    )
    p.add_argument(
        "--specimen",
        dest="specimen_flag",
        type=str,
        default=None,
        help="Specimen Name (alternative to positional argument).",
    )
    p.add_argument(
        "--force-deformation",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Path to force_deformation.csv (relative to repo root or absolute). "
            "Default: data/resampled/{Name}/force_deformation.csv."
        ),
    )
    p.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Calibration input CSV (default: {DEFAULT_INPUT})",
    )
    p.add_argument(
        "--metric",
        choices=("l1", "l2"),
        default=None,
        help="Feature loss norm: l2 (w_feat_l2=1) or l1 (w_feat_l1=1). Overrides input.csv feature weights.",
    )
    p.add_argument(
        "--debug-plots",
        action="store_true",
        help="Write apparent-b and cycle/landmark debug figures under the plots folder.",
    )
    p.add_argument(
        "--replot",
        action="store_true",
        help="Skip optimization; reload parameters.csv and regenerate overlays.",
    )
    p.add_argument(
        "--prepare-data",
        action="store_true",
        help="Run postprocess from data/raw/{Name}/ before calibrating.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=f"Output directory (default: {RESULTS_SINGLE}/<Name>/)",
    )
    p.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help=f"Overlay PNG directory (default: {PLOTS_SINGLE}/<Name>/)",
    )
    p.add_argument(
        "--amplitude-weights",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override input.csv use_amplitude_weights for J_feat cycle weights.",
    )
    args = p.parse_args()

    specimen = args.specimen_flag or args.specimen
    if not specimen:
        p.error("specimen Name is required (positional or --specimen).")

    specimen = str(specimen).strip()
    input_path = Path(args.input).expanduser().resolve()

    prepare_data = bool(args.prepare_data) and not args.replot
    if args.replot and args.prepare_data:
        print("  Note: --prepare-data ignored with --replot")
    if args.force_deformation is None:
        force_csv = _default_force_deformation_path(specimen, prepare_data=prepare_data)
    else:
        force_csv = args.force_deformation

    if args.replot:
        run_ctx = replot_from_saved(
            specimen,
            force_csv,
            out_dir=args.out_dir,
            plots_dir=args.plots_dir,
        )
    else:
        try:
            cfgs = load_single_calibrate_inputs(input_path)
        except (FileNotFoundError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc

        set_ids = [cfg.set_id for cfg in cfgs]
        print(f"  Loaded input: {input_path} ({len(cfgs)} set_id(s): {set_ids})")

        param_rows: list[pd.Series] = []
        metrics_rows: list[dict] = []
        run_ctx: CalibrateRunContext | None = None
        for i, cfg in enumerate(cfgs):
            if len(cfgs) > 1:
                print(f"\n=== set_id={cfg.set_id} ===")
            run_ctx = calibrate_and_plot(
                specimen,
                force_csv,
                cfg,
                prepare_data=prepare_data and i == 0,
                plot_apparent_b=(i == 0),
                debug_plots=bool(args.debug_plots),
                metric=args.metric,
                out_dir=args.out_dir,
                plots_dir=args.plots_dir,
                use_amplitude_weights=args.amplitude_weights,
                param_rows_out=param_rows,
                metrics_rows_out=metrics_rows,
            )

        if run_ctx is None:
            raise SystemExit("No calibration set_id configurations loaded.")

        params_path = run_ctx.specimen_out / "parameters.csv"
        pd.DataFrame(param_rows).to_csv(params_path, index=False)
        metrics_dataframe(metrics_rows).to_csv(
            run_ctx.specimen_out / "parameters_metrics.csv", index=False
        )
        run_one_specimen(
            specimen,
            pd.DataFrame(param_rows),
            run_ctx.cat_row,
            run_ctx.overlay_dir,
            norm_xy_half=None,
            override_b_p=None,
            override_b_n=None,
            force_deformation_csv=run_ctx.csv_path,
        )
        print(f"  Wrote overlays under {run_ctx.overlay_dir}")

        exp_df = pd.read_csv(run_ctx.csv_path)
        grid_paths = plot_all_sets_force_def_grid(
            specimen,
            pd.DataFrame(param_rows),
            run_ctx.cat_row,
            exp_df["Deformation[in]"].to_numpy(dtype=float),
            exp_df["Force[kip]"].to_numpy(dtype=float),
            run_ctx.overlay_dir,
        )
        for grid_path in grid_paths:
            print(f"  Wrote all-set overlay grid: {grid_path}")

    out_dir = run_ctx.specimen_out
    plots_dir = args.plots_dir or (PLOTS_SINGLE / specimen)
    params_path = out_dir / "parameters.csv"
    print(
        f"\nDone: {specimen}\n"
        f"  Parameters: {params_path}\n"
        f"  Metrics:    {out_dir / 'parameters_metrics.csv'}\n"
        f"  Overlay:    {plots_dir / f'{specimen}_set1_force_def_norm.png'} "
        f"(and other set_id overlays if multiple)\n"
        f"  Plots dir:  {plots_dir}"
    )


if __name__ == "__main__":
    main()
