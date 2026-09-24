"""
Generalized optimization of shared steel parameters across specimens (per CSV configuration or global).

Default optimized columns come from ``params_to_optimize.PARAMS_TO_OPTIMIZE``. Rows in
``config/calibration/set_id_settings_generalized.csv`` (``set_id`` column) each define **one** generalized
configuration: seeds, ``optimize_params``, and loss weights. **Steel seeds and ``optimize_params`` are read
only from that file** (never from ``set_id_settings.csv``).

**Per-configuration mode (default):** for **each** ``set_id`` row in the generalized CSV, the script runs one
L-BFGS-B problem over **all** catalog specimens that have path-ordered resampled data, **one loss term per
specimen** with ``generalized_weight`` from ``BRB-Specimens.csv``. Geometry and yield/areas come from the
catalog; ``steel_model`` and baseline steel parameters come from that generalized CSV row (not from
``--params``). The number of runs equals the number of data rows in the generalized CSV.

**Pooled mode** (``--no-by-set-id``): one L-BFGS-B over the same full specimen pool; **L-BFGS starting values**
for jointly optimized parameters are the **mean** of the per-row numeric seeds from **every**
``set_id_settings_generalized.csv`` row (not from ``--params`` or ``set_id_settings.csv``); loss / steel_model
must be consistent across those rows when a mapping CSV is present.

``b_p`` / ``b_n`` in the generalized CSV must be **numeric** (statistic keywords apply only in
``set_id_settings.csv`` for individual calibration).

After optimization, evaluates specimens and writes metrics, plots, NPZ/CSV histories. The output
parameters CSV is built from **BRB-Specimens.csv** plus the merged generalized steel vector (no individual
optimization CSV is read). In per-configuration mode it contains **one row per (specimen, generalized
``set_id``)** — one block per successful generalized run; in pooled mode it contains a single block tagged
with the smallest input ``set_id``.

Digitized unordered specimens use the pipeline resampled deformation drive when available.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
_SCRIPTS = _PROJECT_ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS / "postprocess"))

from calibrate.amplitude_mse_partition import (  # noqa: E402
    build_amplitude_weights,
    energy_scale_s_e,
)
from calibrate.calibration_io import metrics_dataframe  # noqa: E402
from calibrate.calibration_loss_settings import (  # noqa: E402
    CalibrationLossSettings,
    DEFAULT_CALIBRATION_LOSS_SETTINGS,
)
from calibrate.cycle_feature_loss import (  # noqa: E402
    deformation_scale_s_d,
    load_p_y_kip_catalog,
)
from calibrate.lbfgsb_reparam import (  # noqa: E402
    prepare_lbfgsb_parameterization,
    variable_params_from_optimizer_x,
)
from calibrate.optimize_brb_mse import (  # noqa: E402
    AMPLITUDE_WEIGHTS_ARG_HELP,
    DEBUG_PARTITION,
    FAILURE_PENALTY,
    OUTPUT_CSV_FLOAT_FMT,
    PARAMS_TO_OPTIMIZE,
    _dataframe_for_param_csv,
    _loss_weight_snapshot,
    _metrics_dict_for_breakdown,
    _metrics_dict_nan_prefix,
    force_scale_s_f,
    run_simulation_kwargs_from_prow,
    save_simulated_force_history,
    save_simulated_force_history_csv,
    simulated_force_history_dir,
    simulate_and_loss_breakdown,
)
from calibrate.param_limits import bounds_dict_for  # noqa: E402
from calibrate.digitized_unordered_eval_lib import (  # noqa: E402
    compute_unordered_cloud_metrics,
    eval_row_with_envelope_bn_from_unordered,
    load_digitized_unordered_series,
)
from calibrate.plot_params_vs_filtered import (  # noqa: E402
    plot_force_def_digitized_unordered_overlays,
    plot_force_def_overlays,
    plot_unordered_binned_cloud_envelopes,
)
from calibrate.averaged_params_lib import (  # noqa: E402
    BP_BN_SPECIMEN_LOCAL,
    generalized_joint_optimize_param_names,
    merge_averaged_into_row,
    restore_individual_bp_bn,
)
from calibrate.build_initial_brb_parameters import (  # noqa: E402
    OUT_COLS,
    _full_numeric_seed,
    _is_missing_sentinel,
    generalized_init_param_series_for_set_id,
)
from calibrate.params_to_optimize import SIM_PARAMS_FROM_ROW  # noqa: E402
from calibrate.calibration_paths import (  # noqa: E402
    GENERALIZED_BRB_PARAMETERS_PATH,
    GENERALIZED_PARAMS_EVAL_METRICS_PATH,
    GENERALIZED_SET_ID_EVAL_SUMMARY_CSV,
    PARAM_LIMITS_CSV,
    PLOTS_GENERALIZED_OPTIMIZE,
    SET_ID_SETTINGS_GENERALIZED_CSV,
)
from calibrate.pipeline_log import kv, line, run_banner, saved_artifacts, section  # noqa: E402
from calibrate.report_generalized_set_id_eval_summary import (  # noqa: E402
    write_generalized_set_id_eval_summary,
    write_generalized_unordered_j_split_summaries,
)
from calibrate.set_id_optimize_params import (  # noqa: E402
    assert_global_loss_settings_consistent,
    assert_global_optimize_params_consistent,
    assert_global_steel_model_consistent,
    resolve_loss_settings_for_set_id,
    resolve_optimize_params_for_set_id,
)
from calibrate.set_id_settings import (  # noqa: E402
    apply_param_value_ties,
    load_inherit_from_set_by_set_id,
    load_param_alias_ties_by_set_id,
    load_set_id_optimize_and_loss,
    load_set_id_settings,
    load_steel_model_by_set_id,
    sync_tied_columns_in_output_row,
)
from calibrate.steel_model import (  # noqa: E402
    SHARED_STEEL_KEYS,
    STEELMPF_ISO_KEYS,
    normalize_steel_model,
    sim_param_keys_for_model,
)
from calibrate.specimen_weights import (  # noqa: E402
    catalog_metrics_fields,
    make_generalized_weight_fn,
    weight_config_tag,
)
from model.corotruss import run_simulation  # noqa: E402
from postprocess.cycle_points import find_cycle_points, load_cycle_points_resampled  # noqa: E402
from postprocess.plot_dimensions import (  # noqa: E402
    COLOR_NUMERICAL_COHORT,
    COLOR_NUMERICAL_COHORT_AUX,
)
from postprocess.plot_specimens import compute_raw_filtered_global_norm_limits  # noqa: E402
from specimen_catalog import (  # noqa: E402
    GENERALIZED_CONFIG_SET_ID_COL,
    list_names_digitized_unordered,
    path_ordered_resampled_force_csv_stems,
    read_catalog,
    resolve_resampled_force_deformation_csv,
)
DEFAULT_METRICS_OUT = GENERALIZED_PARAMS_EVAL_METRICS_PATH
DEFAULT_PARAMS_OUT = GENERALIZED_BRB_PARAMETERS_PATH
DEFAULT_PLOTS_DIR = PLOTS_GENERALIZED_OPTIMIZE / "overlays"
DEFAULT_CLOUD_PLOTS_DIR = DEFAULT_PLOTS_DIR


@dataclass
class GeneralizedInstance:
    name: str
    set_id: Any
    prow: pd.Series
    D_exp: np.ndarray
    F_exp: np.ndarray
    amp_meta: list[dict]
    loss: CalibrationLossSettings
    p_y_ref: float
    s_d: float
    weight: float


def _generalized_settings_row(settings_df: pd.DataFrame, set_id: int) -> pd.Series:
    m = settings_df["set_id"] == int(set_id)
    if not m.any():
        raise KeyError(f"generalized settings: no row for set_id={set_id}")
    return settings_df.loc[m].iloc[0]


def _prow_from_catalog_and_generalized_settings(
    cat_row: pd.Series | pd.DataFrame,
    settings_row: pd.Series,
    *,
    specimen_name: str,
    generalized_config_set_id: int,
) -> pd.Series:
    """
    One simulation row: geometry and nominal yield from ``BRB-Specimens.csv``; ``steel_model`` and all
    numeric steel seeds from the matching row in ``set_id_settings_generalized.csv`` (via ``_full_numeric_seed``).

    ``cat_row`` is typically ``catalog.set_index('Name').loc[name]``, which has no ``Name`` column; pass
    ``specimen_name`` explicitly (the catalog lookup key).
    """
    name = str(specimen_name).strip()
    if isinstance(cat_row, pd.DataFrame):
        if len(cat_row) != 1:
            raise SystemExit(
                f"{name}: expected one BRB-Specimens.csv row for this Name, got {len(cat_row)}"
            )
        cat_row = cat_row.iloc[0]
    sm = normalize_steel_model(settings_row.get("steel_model"))
    steel = _full_numeric_seed(sm, settings_row)
    fy = float(cat_row["f_yc_ksi"])
    L_T = float(cat_row["L_T_in"])
    L_y = float(cat_row["L_y_in"])
    A_sc = float(cat_row["A_c_in2"])
    A_t = float(cat_row["A_t_in2"])
    out: dict[str, object] = {}
    if "ID" in cat_row.index and pd.notna(cat_row.get("ID")):
        out["ID"] = int(cat_row["ID"])
    out.update(
        {
        "Name": name,
        "set_id": int(generalized_config_set_id),
        "steel_model": sm,
        "L_T": L_T,
        "L_y": L_y,
        "A_sc": A_sc,
        "A_t": A_t,
        "fyp": fy,
        "fyn": fy,
        }
    )
    for k in sim_param_keys_for_model(sm):
        if k in out:
            continue
        if k not in steel:
            raise SystemExit(
                f"{name}: generalized set_id={generalized_config_set_id} / steel_model={sm!r}: "
                f"could not resolve simulation field {k!r} from {settings_row.get('set_id', '?')!r} row "
                "(check numeric seeds in set_id_settings_generalized.csv)."
            )
        out[k] = float(steel[k])
    return pd.Series(out)


def _collect_instances(
    catalog_by_name: pd.DataFrame,
    available: list[str],
    weight_fn: Any,
    *,
    generalized_config_set_id: int,
    settings_row: pd.Series,
    loss_by_set_id: dict[int, CalibrationLossSettings],
    amplitude_weights_override: bool | None,
) -> list[GeneralizedInstance]:
    """Build one ``GeneralizedInstance`` per resampled path-ordered name (catalog + generalized settings row)."""
    out: list[GeneralizedInstance] = []
    for sid in available:
        if sid not in catalog_by_name.index:
            continue
        cat_row = catalog_by_name.loc[sid]
        csv_path = resolve_resampled_force_deformation_csv(sid, _PROJECT_ROOT)
        if csv_path is None or not csv_path.is_file():
            continue
        df = pd.read_csv(csv_path)
        if "Force[kip]" not in df.columns or "Deformation[in]" not in df.columns:
            continue
        D_exp = df["Deformation[in]"].to_numpy(dtype=float)
        F_exp = df["Force[kip]"].to_numpy(dtype=float)
        loaded = load_cycle_points_resampled(sid)
        points, _ = loaded if loaded is not None else find_cycle_points(df)
        s_d_ref = deformation_scale_s_d(D_exp)
        prow = _prow_from_catalog_and_generalized_settings(
            cat_row,
            settings_row,
            specimen_name=sid,
            generalized_config_set_id=generalized_config_set_id,
        )
        p_y_catalog = load_p_y_kip_catalog(
            _PROJECT_ROOT,
            sid,
            float(prow["fyp"]),
            float(prow["A_sc"]),
        )
        w = float(weight_fn(sid))
        loss = resolve_loss_settings_for_set_id(loss_by_set_id, generalized_config_set_id)
        use_amp_w = (
            bool(amplitude_weights_override)
            if amplitude_weights_override is not None
            else loss.use_amplitude_weights
        )
        _mse_w, amp_meta = build_amplitude_weights(
            D_exp,
            points,
            p=loss.amplitude_weight_power,
            eps=loss.amplitude_weight_eps,
            debug_partition=DEBUG_PARTITION,
            use_amplitude_weights=use_amp_w,
        )
        out.append(
            GeneralizedInstance(
                name=sid,
                set_id=prow.get("set_id", "?"),
                prow=prow,
                D_exp=D_exp,
                F_exp=F_exp,
                amp_meta=amp_meta,
                loss=loss,
                p_y_ref=p_y_catalog,
                s_d=s_d_ref,
                weight=w,
            )
        )
    return out


def _generalized_objective_value(
    x: np.ndarray,
    train: list[GeneralizedInstance],
    params_to_optimize: list[str],
    use_norm: list[bool],
    Ls: list[float],
    Us: list[float],
    *,
    exp_landmark_cache: dict | None = None,
    param_ties_by_set_id: dict[int, dict[str, str]] | None = None,
    param_tie_key: int,
) -> float:
    """Weighted mean J over training instances at optimizer x."""
    ptab = param_ties_by_set_id or {}
    opt_set = frozenset(params_to_optimize)
    variable = variable_params_from_optimizer_x(x, params_to_optimize, use_norm, Ls, Us)
    num = 0.0
    den = 0.0
    ties = ptab.get(int(param_tie_key), {})
    for inst in train:
        if inst.weight <= 0.0:
            continue
        sm, fixed = run_simulation_kwargs_from_prow(inst.prow)
        params = {**fixed, **variable}
        restore_individual_bp_bn(
            params,
            inst.prow,
            skip=frozenset(params_to_optimize) & frozenset(BP_BN_SPECIMEN_LOCAL),
        )
        apply_param_value_ties(params, ties, opt_set)
        try:
            F_sim = np.asarray(run_simulation(inst.D_exp, steel_model=sm, **params), dtype=float)
        except Exception:
            return float(FAILURE_PENALTY)
        prow = inst.prow
        bd = simulate_and_loss_breakdown(
            inst.D_exp,
            inst.F_exp,
            F_sim,
            inst.amp_meta,
            s_d=inst.s_d,
            loss=inst.loss,
            fy_ksi=float(prow["fyp"]),
            a_sc=float(prow["A_sc"]),
            L_T=float(prow["L_T"]),
            L_y=float(prow["L_y"]),
            A_t=float(prow["A_t"]),
            E_ksi=float(prow["E"]),
            exp_landmark_cache=exp_landmark_cache,
            full_metrics=False,
        )
        if bd is None:
            return float(FAILURE_PENALTY)
        num += inst.weight * bd.j_total
        den += inst.weight
    if den <= 0.0:
        return float(FAILURE_PENALTY)
    return num / den


def _optimize_one_generalized_group(
    train: list[GeneralizedInstance],
    init_averaged: pd.Series,
    params_to_optimize: list[str],
    bounds_dict: dict[str, tuple[float, float]],
    *,
    param_ties_by_set_id: dict[int, dict[str, str]] | None = None,
    param_tie_key: int,
) -> tuple[pd.Series, Any]:
    """
    Return optimized generalized Series and scipy OptimizeResult.

    **Starting vector** for jointly optimized parameters comes **only** from ``init_averaged``
    (built with ``generalized_init_param_series_for_set_id`` from ``set_id_settings_generalized.csv``).
    ``train[0].prow`` is only a merge carrier for geometry and **non-optimized** steel fields (from catalog +
    generalized settings row); it must not define initial values for ``params_to_optimize``.
    """
    rep = train[0].prow.copy()
    skip_bn = frozenset(params_to_optimize) & frozenset(BP_BN_SPECIMEN_LOCAL)
    merged = merge_averaged_into_row(
        rep, init_averaged, params_to_optimize, skip_bp_bn_restore=skip_bn
    )
    for name in params_to_optimize:
        if name not in init_averaged.index:
            raise SystemExit(
                f"internal: generalized init Series missing {name!r} "
                "(expected numeric seeds from set_id_settings_generalized.csv for this set_id)."
            )
        merged[name] = float(init_averaged[name])
    use_norm, Ls, Us, x0, scipy_bounds = prepare_lbfgsb_parameterization(
        params_to_optimize,
        bounds_dict,
        merged,
        specimen_hint="generalized",
    )
    exp_landmark_cache: dict = {}

    def fun(x: np.ndarray) -> float:
        """Scalar objective for L-BFGS-B."""
        return _generalized_objective_value(
            x,
            train,
            params_to_optimize,
            use_norm,
            Ls,
            Us,
            exp_landmark_cache=exp_landmark_cache,
            param_ties_by_set_id=param_ties_by_set_id,
            param_tie_key=param_tie_key,
        )

    res = minimize(
        fun,
        x0,
        method="L-BFGS-B",
        bounds=scipy_bounds,
        options={"ftol": 1e-8, "gtol": 1e-6},
    )
    physical = variable_params_from_optimizer_x(
        res.x, params_to_optimize, use_norm, Ls, Us
    )
    out = pd.Series({k: float(physical[k]) for k in params_to_optimize})
    return out, res


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(
        description=(
            "Generalized L-BFGS-B on shared steel parameters (default PARAMS_TO_OPTIMIZE; "
            "set_id_settings_generalized.csv). One optimization per ``set_id`` row in that CSV over all "
            "path-ordered specimens with positive generalized_weight; then evaluation. "
            "Joint-parameter **starting values** always come from that generalized CSV (see "
            "generalized_init_param_series_for_set_id), not from individual calibration seeds. "
            "No individual optimization CSV is read: geometry from BRB-Specimens.csv, steel from the "
            "generalized settings CSV."
        ),
    )
    p.add_argument(
        "--output-params",
        type=str,
        default=str(DEFAULT_PARAMS_OUT),
        help="Output parameters CSV (merged generalized steel params per row).",
    )
    p.add_argument(
        "--no-by-set-id",
        action="store_true",
        help="One shared steel vector for all generalized CSV configurations (pool and optimize jointly).",
    )
    p.add_argument(
        "--output-metrics",
        type=str,
        default=str(DEFAULT_METRICS_OUT),
        help="Output metrics CSV path.",
    )
    p.add_argument(
        "--output-plots-dir",
        type=str,
        default=str(DEFAULT_PLOTS_DIR),
        help="Directory for resampled generalized-eval hysteresis overlays.",
    )
    p.add_argument(
        "--output-cloud-plots-dir",
        type=str,
        default=str(DEFAULT_CLOUD_PLOTS_DIR),
        help="Digitized unordered overlays (default: same as --output-plots-dir).",
    )
    p.add_argument(
        "--specimen",
        type=str,
        default=None,
        help="If set, only this Name is optimized and evaluated.",
    )
    p.add_argument(
        "--train-specimens",
        type=str,
        default=None,
        help=(
            "Comma-separated Names to restrict train + eval (e.g. PC250,PC350,PC3SB). "
            "Useful for notebook demos; ignored names with generalized_weight=0 still "
            "do not enter the joint fit."
        ),
    )
    _pl_rel = PARAM_LIMITS_CSV
    try:
        _pl_rel = PARAM_LIMITS_CSV.relative_to(_PROJECT_ROOT)
    except ValueError:
        pass
    p.add_argument(
        "--param-limits",
        type=Path,
        default=None,
        help=(
            "Parameter box limits CSV (parameter, lower, upper). "
            f"Default: {_pl_rel}."
        ),
    )
    p.add_argument(
        "--amplitude-weights",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=AMPLITUDE_WEIGHTS_ARG_HELP,
    )
    _so_rel = SET_ID_SETTINGS_GENERALIZED_CSV
    try:
        _so_rel = SET_ID_SETTINGS_GENERALIZED_CSV.relative_to(_PROJECT_ROOT)
    except ValueError:
        pass
    p.add_argument(
        "--set-id-settings",
        type=Path,
        default=None,
        help=(
            "Generalized-stage CSV: one L-BFGS problem per ``set_id`` row (seeds + optimize_params + loss weights). "
            f"Default: {_so_rel}."
        ),
    )
    args = p.parse_args()

    run_banner("optimize_generalized_brb_mse.py")

    by_set_id = not args.no_by_set_id
    out_metrics = Path(args.output_metrics).expanduser().resolve()
    out_params = Path(args.output_params).expanduser().resolve()
    plots_dir_ordered = Path(args.output_plots_dir).expanduser().resolve()
    plots_dir_unordered = Path(args.output_cloud_plots_dir).expanduser().resolve()
    plots_dir_ordered.mkdir(parents=True, exist_ok=True)
    plots_dir_unordered.mkdir(parents=True, exist_ok=True)
    sim_hist_dir = simulated_force_history_dir(out_metrics)

    orig_param_columns = list(OUT_COLS)

    catalog = read_catalog()
    catalog_by_name = catalog.set_index("Name")
    norm_xy_half = compute_raw_filtered_global_norm_limits(catalog, project_root=_PROJECT_ROOT)
    generalized_w_fn = make_generalized_weight_fn(catalog)
    weight_tag = weight_config_tag(catalog)

    resampled_stems = path_ordered_resampled_force_csv_stems(catalog, project_root=_PROJECT_ROOT)
    catalog_names = set(catalog["Name"].astype(str).str.strip())
    available_resampled = sorted(catalog_names & resampled_stems)
    unordered_eligible = set(list_names_digitized_unordered(catalog))
    available_unordered = sorted((catalog_names & unordered_eligible) - set(available_resampled))

    if not available_resampled:
        raise SystemExit(
            "Generalized optimization needs at least one path-ordered resampled specimen "
            "(data/resampled/{Name}/force_deformation.csv)."
        )

    if args.specimen:
        if args.specimen not in available_resampled and args.specimen not in available_unordered:
            raise SystemExit(
                f"Specimen {args.specimen!r} not in resampled or digitized unordered eval set."
            )
        available_resampled = [args.specimen] if args.specimen in available_resampled else []
        available_unordered = [args.specimen] if args.specimen in available_unordered else []
        if not available_resampled:
            raise SystemExit("Generalized training requires resampled data for --specimen.")

    if args.train_specimens:
        if args.specimen:
            raise SystemExit("Use only one of --specimen or --train-specimens.")
        wanted = [s.strip() for s in str(args.train_specimens).split(",") if s.strip()]
        if not wanted:
            raise SystemExit("--train-specimens is empty.")
        missing = [
            n
            for n in wanted
            if n not in available_resampled and n not in available_unordered
        ]
        if missing:
            raise SystemExit(
                f"--train-specimens not found in resampled/unordered data: {missing}"
            )
        wanted_set = set(wanted)
        available_resampled = [n for n in available_resampled if n in wanted_set]
        available_unordered = [n for n in available_unordered if n in wanted_set]
        if not available_resampled:
            raise SystemExit(
                "--train-specimens: need at least one path-ordered resampled Name for training."
            )
        line(f"train/eval specimen filter: {available_resampled}")

    default_list = list(PARAMS_TO_OPTIMIZE)
    opt_csv = (
        Path(args.set_id_settings).expanduser().resolve()
        if args.set_id_settings
        else SET_ID_SETTINGS_GENERALIZED_CSV
    )
    opt_map, loss_map = load_set_id_optimize_and_loss(opt_csv)
    alias_by_set_id = load_param_alias_ties_by_set_id(opt_csv)
    inherit_from_set = load_inherit_from_set_by_set_id(opt_csv)
    if opt_map or loss_map:
        line(f"set_id settings CSV: {opt_csv}")
    else:
        line(
            "set_id settings: (none) - using defaults: PARAMS_TO_OPTIMIZE and "
            "DEFAULT_CALIBRATION_LOSS_SETTINGS"
        )
    if inherit_from_set:
        chains = ", ".join(f"{c}<-{p}" for c, p in sorted(inherit_from_set.items()))
        line(
            f"inherit_from_set chains (child<-parent, applied at run time per generalized set_id): {chains}"
        )
        line(
            "  explicit (non--999) steel cells in a child's generalized settings row are preserved; "
            "parent's optimum only fills missing/-999 cells."
        )

    settings_st = load_set_id_settings(opt_csv)
    run_ids = sorted({int(x) for x in pd.to_numeric(settings_st["set_id"], errors="coerce").dropna()})
    if not run_ids:
        raise SystemExit(f"{opt_csv.name}: no numeric set_id rows in generalized settings CSV")

    bounds_cache: dict[tuple[str, ...], dict[str, tuple[float, float]]] = {}

    def _bounds_for_active(active: list[str]) -> dict[str, tuple[float, float]]:
        key = tuple(active)
        if key not in bounds_cache:
            bounds_cache[key] = bounds_dict_for(list(active), limits_path=args.param_limits)
        return bounds_cache[key]

    def _seed_keys_for_sm(sm: object) -> frozenset[str]:
        normalize_steel_model(sm)
        bn = ("b_p", "b_n")
        return frozenset((*SHARED_STEEL_KEYS, *bn, *STEELMPF_ISO_KEYS))

    def _overlay_parent_on_settings_row(
        settings_row: pd.Series,
        parent_opt_vec: pd.Series,
        target_steel_model: object,
    ) -> tuple[pd.Series, list[str]]:
        """
        Overlay parent's just-optimized values onto settings_row for missing/-999 steel cells.

        Explicit numeric cells in the child's settings_row win over inheritance (so the child can
        still pin individual columns). Returns (overlaid_row, names_overlaid).
        """
        out = settings_row.copy()
        allow = _seed_keys_for_sm(target_steel_model)
        overlaid: list[str] = []
        for k in allow:
            if k not in out.index:
                continue
            if not _is_missing_sentinel(out[k]):
                continue
            if k not in parent_opt_vec.index:
                continue
            v = parent_opt_vec.get(k)
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(fv):
                continue
            out[k] = fv
            overlaid.append(k)
        return out, sorted(overlaid)

    def _overlay_parent_on_init_vec(
        init_vec: pd.Series,
        parent_opt_vec: pd.Series,
        active_params: list[str],
        child_settings_row: pd.Series,
    ) -> tuple[pd.Series, list[str], list[str]]:
        """
        Warm-start: copy parent's optimized values onto init_vec for active params they share.

        Active params with an explicit (non--999) cell in the child's generalized settings row are
        preserved as-is, matching the non-active overlay semantics.
        Returns (overlaid_init_vec, names_overlaid, names_kept_explicit).
        """
        out = init_vec.copy()
        overlaid: list[str] = []
        kept_explicit: list[str] = []
        for p in active_params:
            if p not in parent_opt_vec.index or p not in out.index:
                continue
            if (
                p in child_settings_row.index
                and not _is_missing_sentinel(child_settings_row[p])
            ):
                kept_explicit.append(p)
                continue
            v = parent_opt_vec.get(p)
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(fv):
                continue
            out[p] = fv
            overlaid.append(p)
        return out, sorted(overlaid), sorted(kept_explicit)

    lim_path = Path(args.param_limits).expanduser().resolve() if args.param_limits else PARAM_LIMITS_CSV
    line(f"parameter limits: {lim_path}")

    per_sid_init: dict[int, pd.Series] = {}
    for rid in run_ids:
        active_here = resolve_optimize_params_for_set_id(opt_map, rid, default_list)
        per_sid_init[rid] = generalized_init_param_series_for_set_id(rid, opt_csv, active_here)
    kv(
        "L-BFGS init (joint params)",
        f"numeric seeds from {opt_csv.name} per generalized set_id only (not from individual set_id_settings or --params steel columns for those params)",
    )

    generalized_pool: dict[Any, pd.Series] = {}
    global_init_vec: pd.Series | None = None
    global_merge_active: list[str] = list(default_list)

    section("Joint optimization (L-BFGS-B)")
    if by_set_id:
        line(
            f"generalized set_id row(s) from {opt_csv.name}: {run_ids} -> {len(run_ids)} L-BFGS problem(s) "
            f"(each specimen: catalog geometry + generalized CSV steel; positive generalized_weight)."
        )
        for run_id in run_ids:
            g_row = _generalized_settings_row(settings_st, int(run_id))
            parent_rid = inherit_from_set.get(int(run_id))
            inherit_label = ""
            if parent_rid is not None:
                parent_opt = generalized_pool.get(parent_rid)
                if parent_opt is None:
                    line(
                        f"set_id={run_id}: inherit_from_set={parent_rid} requested but parent has no "
                        "optimized result yet (check ordering / parent skip); using settings seeds only."
                    )
                else:
                    g_row, overlaid_cols = _overlay_parent_on_settings_row(
                        g_row, parent_opt, g_row.get("steel_model")
                    )
                    if overlaid_cols:
                        inherit_label = (
                            f"  [inherited from set {parent_rid}: "
                            f"{len(overlaid_cols)} steel col(s)]"
                        )
            instances = _collect_instances(
                catalog_by_name,
                available_resampled,
                generalized_w_fn,
                generalized_config_set_id=int(run_id),
                settings_row=g_row,
                loss_by_set_id=loss_map,
                amplitude_weights_override=args.amplitude_weights,
            )
            train_run = [i for i in instances if i.weight > 0.0]
            if not train_run:
                line(
                    f"skip set_id={run_id}: no training instances "
                    "(positive generalized_weight + resampled path-ordered data in catalog)"
                )
                continue
            active = resolve_optimize_params_for_set_id(opt_map, run_id, default_list)
            active_gen = generalized_joint_optimize_param_names(active)
            if not active_gen:
                raise SystemExit(f"set_id={run_id}: empty optimize_params (was {active!r}).")
            init_vec = per_sid_init[run_id]
            if parent_rid is not None and generalized_pool.get(parent_rid) is not None:
                child_row_for_warmstart = _generalized_settings_row(settings_st, int(run_id))
                init_vec, warm_active, kept_active = _overlay_parent_on_init_vec(
                    init_vec,
                    generalized_pool[parent_rid],
                    active_gen,
                    child_row_for_warmstart,
                )
                if warm_active or kept_active:
                    msg = f"set_id={run_id}: L-BFGS init"
                    if warm_active:
                        msg += f"  warm-started for {warm_active} from set {parent_rid}"
                    if kept_active:
                        msg += f"  kept explicit child cells for {kept_active}"
                    line(msg + ".")
                per_sid_init[run_id] = init_vec
            n_train_names = len({i.name for i in train_run})
            line(
                f"set_id={run_id}  ({len(train_run)} training instances from {n_train_names} specimens, "
                f"init from {opt_csv.name}){inherit_label}  optimize (joint): {active_gen}..."
            )
            opt_vec, res = _optimize_one_generalized_group(
                train_run,
                init_vec,
                active_gen,
                _bounds_for_active(active_gen),
                param_ties_by_set_id=alias_by_set_id,
                param_tie_key=run_id,
            )
            generalized_pool[run_id] = opt_vec
            line(
                f"L-BFGS-B  success={res.success}  fun={res.fun:.6g}  message={res.message!r}"
            )
    else:
        global_loss = (
            assert_global_loss_settings_consistent(
                loss_map, run_ids, DEFAULT_CALIBRATION_LOSS_SETTINGS
            )
            if loss_map
            else DEFAULT_CALIBRATION_LOSS_SETTINGS
        )
        global_active = (
            assert_global_optimize_params_consistent(opt_map, run_ids, default_list)
            if opt_map
            else default_list
        )
        global_merge_active = list(global_active)
        if opt_map:
            assert_global_steel_model_consistent(load_steel_model_by_set_id(opt_csv), run_ids)

        acc = pd.Series(dtype=float)
        tw = 0.0
        for rid in run_ids:
            acc = acc.add(per_sid_init[rid], fill_value=0.0)
            tw += 1.0
        if tw <= 0.0:
            raise SystemExit("global mode: zero generalized CSV rows for init merge")
        init_vec = acc / tw

        probe_rid = int(run_ids[0])
        g_row0 = _generalized_settings_row(settings_st, probe_rid)
        instances0 = _collect_instances(
            catalog_by_name,
            available_resampled,
            generalized_w_fn,
            generalized_config_set_id=probe_rid,
            settings_row=g_row0,
            loss_by_set_id=loss_map,
            amplitude_weights_override=args.amplitude_weights,
        )
        train_all = [i for i in instances0 if i.weight > 0.0]
        if not train_all:
            raise SystemExit("No training instances with positive specimen weight (global mode).")

        global_gen = generalized_joint_optimize_param_names(global_active)
        if not global_gen:
            raise SystemExit(f"Global optimize list is empty (was {global_active!r}).")
        n_train_names_g = len({i.name for i in train_all})
        line(
            f"global pool  ({len(train_all)} training instances from {n_train_names_g} specimens, "
            f"init averaged over {len(run_ids)} CSV row(s))  optimize (joint): {global_gen}..."
        )
        tie_key = int(min(run_ids))
        opt_vec, res = _optimize_one_generalized_group(
            train_all,
            init_vec,
            global_gen,
            _bounds_for_active(global_gen),
            param_ties_by_set_id=alias_by_set_id,
            param_tie_key=tie_key,
        )
        generalized_pool["_global"] = opt_vec
        global_init_vec = init_vec
        line(
            f"L-BFGS-B  success={res.success}  fun={res.fun:.6g}  message={res.message!r}"
        )

    if by_set_id:
        param_merge_rids: list[int] = [int(r) for r in run_ids if int(r) in generalized_pool]
        if not param_merge_rids:
            raise SystemExit(
                "No successful generalized set_id runs in pool; cannot merge output parameters."
            )
    else:
        param_merge_rids = [int(run_ids[0])]
        shared_for_param_merge = generalized_pool["_global"]
        active_for_param_merge = global_merge_active
        alias_for_param_merge = alias_by_set_id.get(int(min(run_ids)), {})

    kv("output parameters", str(out_params))
    kv("output metrics", str(out_metrics))
    kv("plots (path-ordered)", str(plots_dir_ordered))
    if plots_dir_unordered.resolve() != plots_dir_ordered.resolve():
        kv("plots (digitized unordered)", str(plots_dir_unordered))
    kv("weights", repr(weight_tag))
    kv("loss settings", f"from {opt_csv.name} (one configuration per set_id row)")
    if args.amplitude_weights is None:
        kv("J_feat cycle weights", "per-set (see CSV)")
    else:
        kv(
            "J_feat cycle weights",
            "amplitude (CLI override)" if bool(args.amplitude_weights) else "uniform (CLI override)",
        )

    rows_out: list[dict[str, Any]] = []

    path_eval_runs: list[tuple[int, list[GeneralizedInstance]]] = []
    if by_set_id:
        for rid in run_ids:
            if rid not in generalized_pool:
                continue
            gr = _generalized_settings_row(settings_st, int(rid))
            parent_rid_eval = inherit_from_set.get(int(rid))
            if parent_rid_eval is not None and parent_rid_eval in generalized_pool:
                gr, _ = _overlay_parent_on_settings_row(
                    gr, generalized_pool[parent_rid_eval], gr.get("steel_model")
                )
            path_eval_runs.append(
                (
                    int(rid),
                    _collect_instances(
                        catalog_by_name,
                        available_resampled,
                        generalized_w_fn,
                        generalized_config_set_id=int(rid),
                        settings_row=gr,
                        loss_by_set_id=loss_map,
                        amplitude_weights_override=args.amplitude_weights,
                    ),
                )
            )
    else:
        rid0 = int(run_ids[0])
        gr0 = _generalized_settings_row(settings_st, rid0)
        path_eval_runs.append(
            (
                rid0,
                _collect_instances(
                    catalog_by_name,
                    available_resampled,
                    generalized_w_fn,
                    generalized_config_set_id=rid0,
                    settings_row=gr0,
                    loss_by_set_id=loss_map,
                    amplitude_weights_override=args.amplitude_weights,
                ),
            )
        )

    plot_multi_cfg = by_set_id and len(path_eval_runs) > 1

    section("Generalized evaluation -- path-ordered (resampled)")
    for eval_rid, instances_ev in path_eval_runs:
        plot_here = (
            (plots_dir_ordered / f"config_set_{eval_rid}") if plot_multi_cfg else plots_dir_ordered
        )
        plot_here.mkdir(parents=True, exist_ok=True)
        hist_here = (
            (sim_hist_dir / f"config_set_{eval_rid}") if plot_multi_cfg else sim_hist_dir
        )
        hist_here.mkdir(parents=True, exist_ok=True)
        for inst in instances_ev:
            sid = inst.name
            set_id = inst.set_id
            D_exp = inst.D_exp
            F_exp = inst.F_exp
            amp_meta = inst.amp_meta
            prow = inst.prow
            specimen_w = inst.weight
            contributes = specimen_w > 0.0
            cm = catalog_metrics_fields(sid, catalog_by_name)

            init_shared = (
                per_sid_init[eval_rid] if by_set_id else (global_init_vec if global_init_vec is not None else pd.Series(dtype=float))
            )
            if not by_set_id and global_init_vec is None:
                raise SystemExit("internal: global_init_vec missing for evaluation")

            try:
                shared = (
                    generalized_pool[eval_rid]
                    if by_set_id
                    else generalized_pool["_global"]
                )
            except KeyError as e:
                line(f"skip {sid} generalized set_id={eval_rid}: {e}")
                continue

            active = resolve_optimize_params_for_set_id(opt_map, eval_rid, default_list)
            joint_skip = frozenset(generalized_joint_optimize_param_names(active)) & frozenset(
                BP_BN_SPECIMEN_LOCAL
            )
            eval_row_init = merge_averaged_into_row(
                prow, init_shared, active, skip_bp_bn_restore=joint_skip
            )
            eval_row = merge_averaged_into_row(prow, shared, active, skip_bp_bn_restore=joint_skip)
            ties_here = alias_by_set_id.get(eval_rid, {})
            opt_act = frozenset(active)
            s_f_ref = force_scale_s_f(F_exp)
            s_e_ref = energy_scale_s_e(D_exp, F_exp)
            n_cycles = len(amp_meta)

            exp_landmark_cache_eval: dict = {}
            try:
                sm0, kw0 = run_simulation_kwargs_from_prow(eval_row_init)
                apply_param_value_ties(kw0, ties_here, opt_act)
                F_sim_init = np.asarray(
                    run_simulation(D_exp, steel_model=sm0, **kw0),
                    dtype=float,
                )
            except Exception as exc:
                line(
                    f"{sid} generalized set_id={eval_rid}: initial simulation failed: {exc}"
                )
                continue

            if F_sim_init.shape != F_exp.shape:
                line(
                    f"{sid} generalized set_id={eval_rid}: length mismatch initial sim vs exp"
                )
                continue

            loss_here = inst.loss

            bd_init = simulate_and_loss_breakdown(
                D_exp,
                F_exp,
                F_sim_init,
                amp_meta,
                s_d=inst.s_d,
                loss=loss_here,
                fy_ksi=float(eval_row_init["fyp"]),
                a_sc=float(eval_row_init["A_sc"]),
                L_T=float(eval_row_init["L_T"]),
                L_y=float(eval_row_init["L_y"]),
                A_t=float(eval_row_init["A_t"]),
                E_ksi=float(eval_row_init["E"]),
                exp_landmark_cache=exp_landmark_cache_eval,
            )
            if bd_init is None:
                line(
                    f"{sid} generalized set_id={eval_rid}: initial loss breakdown failed"
                )
                continue

            try:
                sm1, kw1 = run_simulation_kwargs_from_prow(eval_row)
                apply_param_value_ties(kw1, ties_here, opt_act)
                F_sim = np.asarray(
                    run_simulation(D_exp, steel_model=sm1, **kw1),
                    dtype=float,
                )
            except Exception as exc:
                line(f"{sid} generalized set_id={eval_rid}: simulation failed: {exc}")
                continue

            if F_sim.shape != F_exp.shape:
                line(
                    f"{sid} generalized set_id={eval_rid}: length mismatch sim vs exp"
                )
                continue

            bd = simulate_and_loss_breakdown(
                D_exp,
                F_exp,
                F_sim,
                amp_meta,
                s_d=inst.s_d,
                loss=loss_here,
                fy_ksi=float(eval_row["fyp"]),
                a_sc=float(eval_row["A_sc"]),
                L_T=float(eval_row["L_T"]),
                L_y=float(eval_row["L_y"]),
                A_t=float(eval_row["A_t"]),
                E_ksi=float(eval_row["E"]),
                exp_landmark_cache=exp_landmark_cache_eval,
            )
            if bd is None:
                line(f"{sid} generalized set_id={eval_rid}: loss breakdown failed")
                continue

            jtot0 = bd_init.j_total
            jtot = bd.j_total
            cloud_init = compute_unordered_cloud_metrics(D_exp, F_exp, D_exp, F_sim_init)
            cloud_final = compute_unordered_cloud_metrics(D_exp, F_exp, D_exp, F_sim)
            mi = _metrics_dict_for_breakdown(bd_init, loss_here, "initial")
            mf = _metrics_dict_for_breakdown(bd, loss_here, "final")

            try:
                plot_force_def_overlays(
                    sid,
                    D_exp,
                    F_exp,
                    F_sim,
                    fy_ksi=float(eval_row["fyp"]),
                    A_c_in2=float(eval_row["A_sc"]),
                    L_y_in=float(eval_row["L_y"]),
                    set_id=set_id,
                    out_dir=plot_here,
                    norm_xy_half=norm_xy_half,
                    numerical_color=(
                        COLOR_NUMERICAL_COHORT if contributes else COLOR_NUMERICAL_COHORT_AUX
                    ),
                    show_numerical_curve=True,
                )
            except Exception as exc:
                line(f"{sid} generalized set_id={eval_rid}: plotting failed: {exc}")

            try:
                fyA = float(eval_row["fyp"]) * float(eval_row["A_sc"])
                ly = float(eval_row["L_y"])
                exp_plot_xy = np.column_stack(
                    [cloud_final.exp_points_raw[:, 0] / ly, cloud_final.exp_points_raw[:, 1] / fyA]
                )
                num_plot_xy = np.column_stack(
                    [cloud_final.num_points_raw[:, 0] / ly, cloud_final.num_points_raw[:, 1] / fyA]
                )
            except Exception as exc:
                line(
                    f"{sid} generalized set_id={eval_rid}: cloud plot arrays failed: {exc}"
                )
            else:
                try:
                    plot_unordered_binned_cloud_envelopes(
                        sid,
                        set_id,
                        exp_plot_xy,
                        num_plot_xy,
                        out_dir=plot_here,
                    )
                except Exception as exc:
                    line(
                        f"{sid} generalized set_id={eval_rid}: binned cloud envelope plotting failed: {exc}"
                    )

            saved_npz = save_simulated_force_history(
                hist_here, sid, set_id, D_exp, F_exp, F_sim
            )
            saved_csv = save_simulated_force_history_csv(
                hist_here, sid, set_id, D_exp, F_exp, F_sim
            )
            saved_artifacts(
                saved_npz.name if saved_npz is not None else None,
                saved_csv.name if saved_csv is not None else None,
            )

            rows_out.append(
                {
                    "Name": sid,
                    "set_id": set_id,
                    GENERALIZED_CONFIG_SET_ID_COL: eval_rid,
                    "specimen_weight": specimen_w,
                    "contributes_to_aggregate": contributes,
                    **cm,
                    "weight_config": weight_tag,
                    "calibration_stage": "generalized_opt",
                    "aggregate_by_set_id": by_set_id,
                    **mi,
                    **mf,
                    **_loss_weight_snapshot(loss_here),
                    "S_F": s_f_ref,
                    "S_D": inst.s_d,
                    "S_E": s_e_ref,
                    "P_y_ref": inst.p_y_ref,
                    "n_cycles": n_cycles,
                    "success": jtot < FAILURE_PENALTY * 0.5,
                }
            )
            line(
                f"{sid} generalized set_id={eval_rid}: J={jtot:.6g}  "
                f"J_binenv={cloud_final.J_binenv:.6g}"
            )

    section("Generalized evaluation -- digitized unordered")
    unordered_eval_rids: list[int] = []
    if by_set_id:
        for rid in run_ids:
            if int(rid) in generalized_pool:
                unordered_eval_rids.append(int(rid))
    else:
        unordered_eval_rids.append(int(run_ids[0]))

    for eval_rid in unordered_eval_rids:
        plot_u_here = (
            (plots_dir_unordered / f"config_set_{eval_rid}") if plot_multi_cfg else plots_dir_unordered
        )
        plot_u_here.mkdir(parents=True, exist_ok=True)
        hist_u_here = (
            (sim_hist_dir / f"config_set_{eval_rid}") if plot_multi_cfg else sim_hist_dir
        )
        hist_u_here.mkdir(parents=True, exist_ok=True)
        shared_u = generalized_pool[eval_rid] if by_set_id else generalized_pool["_global"]
        init_shared_u = per_sid_init[eval_rid] if by_set_id else global_init_vec
        if not by_set_id and global_init_vec is None:
            raise SystemExit("internal: global_init_vec missing for digitized unordered evaluation")

        du_settings_row = _generalized_settings_row(settings_st, int(eval_rid))
        if by_set_id:
            parent_rid_du = inherit_from_set.get(int(eval_rid))
            if parent_rid_du is not None and parent_rid_du in generalized_pool:
                du_settings_row, _ = _overlay_parent_on_settings_row(
                    du_settings_row,
                    generalized_pool[parent_rid_du],
                    du_settings_row.get("steel_model"),
                )
        for sid in available_unordered:
            if sid not in catalog_by_name.index:
                line(f"skip {sid}: not in BRB-Specimens.csv")
                continue
            cat_row = catalog_by_name.loc[sid]
            base_prow = _prow_from_catalog_and_generalized_settings(
                cat_row,
                du_settings_row,
                specimen_name=sid,
                generalized_config_set_id=int(eval_rid),
            )
            series = load_digitized_unordered_series(
                sid,
                _PROJECT_ROOT,
                steel_row=base_prow,
                catalog_row=cat_row,
            )
            if series is None:
                line(f"skip {sid}: digitized unordered CSVs missing or invalid")
                continue
            D_drive, u_c, F_c = series
            p_y_catalog = load_p_y_kip_catalog(
                _PROJECT_ROOT,
                sid,
                float(base_prow["fyp"]),
                float(base_prow["A_sc"]),
            )
            specimen_w = generalized_w_fn(sid)
            cm = catalog_metrics_fields(sid, catalog_by_name)
            s_d_ref = deformation_scale_s_d(D_drive)
            s_f_cloud = force_scale_s_f(F_c)
            s_e_cloud = energy_scale_s_e(u_c, F_c)

            set_id_out = int(eval_rid)
            contributes = specimen_w > 0.0
            active = resolve_optimize_params_for_set_id(opt_map, eval_rid, default_list)
            joint_skip = frozenset(generalized_joint_optimize_param_names(active)) & frozenset(
                BP_BN_SPECIMEN_LOCAL
            )
            loss_here = resolve_loss_settings_for_set_id(loss_map, eval_rid)
            eval_row = merge_averaged_into_row(
                base_prow, shared_u, active, skip_bp_bn_restore=joint_skip
            )
            sim_row = eval_row_with_envelope_bn_from_unordered(eval_row, cat_row, u_c, F_c)
            eval_row_init = merge_averaged_into_row(
                base_prow, init_shared_u, active, skip_bp_bn_restore=joint_skip
            )
            sim_row_init = eval_row_with_envelope_bn_from_unordered(
                eval_row_init, cat_row, u_c, F_c
            )

            try:
                sm0, kw0 = run_simulation_kwargs_from_prow(sim_row_init)
                apply_param_value_ties(
                    kw0,
                    alias_by_set_id.get(eval_rid, {}),
                    frozenset(active),
                )
                F_sim_init = np.asarray(
                    run_simulation(D_drive, steel_model=sm0, **kw0),
                    dtype=float,
                )
            except Exception as exc:
                line(
                    f"{sid} generalized set_id={eval_rid}: "
                    f"initial simulation failed: {exc}"
                )
                continue

            try:
                sm1, kw1 = run_simulation_kwargs_from_prow(sim_row)
                apply_param_value_ties(
                    kw1,
                    alias_by_set_id.get(eval_rid, {}),
                    frozenset(active),
                )
                F_sim = np.asarray(
                    run_simulation(D_drive, steel_model=sm1, **kw1),
                    dtype=float,
                )
            except Exception as exc:
                line(
                    f"{sid} generalized set_id={eval_rid}: "
                    f"simulation failed: {exc}"
                )
                continue

            if F_sim.shape != D_drive.shape:
                line(
                    f"{sid} generalized set_id={eval_rid}: "
                    "length mismatch sim vs deformation_history"
                )
                continue

            if F_sim_init.shape != D_drive.shape:
                line(
                    f"{sid} generalized set_id={eval_rid}: "
                    "length mismatch initial sim vs deformation_history"
                )
                continue

            cloud_init = compute_unordered_cloud_metrics(u_c, F_c, D_drive, F_sim_init)
            cloud_final = compute_unordered_cloud_metrics(u_c, F_c, D_drive, F_sim)
            F_exp_na = np.full_like(D_drive, np.nan, dtype=float)

            try:
                plot_force_def_digitized_unordered_overlays(
                    sid,
                    D_drive,
                    F_sim,
                    u_c,
                    F_c,
                    fy_ksi=float(sim_row["fyp"]),
                    A_c_in2=float(sim_row["A_sc"]),
                    L_y_in=float(sim_row["L_y"]),
                    set_id=set_id_out,
                    out_dir=plot_u_here,
                    norm_xy_half=norm_xy_half,
                    numerical_color=COLOR_NUMERICAL_COHORT_AUX,
                )
            except Exception as exc:
                line(
                    f"{sid} generalized set_id={eval_rid}: plotting failed: {exc}"
                )

            try:
                fyA = float(sim_row["fyp"]) * float(sim_row["A_sc"])
                ly = float(sim_row["L_y"])
                exp_plot_xy = np.column_stack(
                    [cloud_final.exp_points_raw[:, 0] / ly, cloud_final.exp_points_raw[:, 1] / fyA]
                )
                num_plot_xy = np.column_stack(
                    [cloud_final.num_points_raw[:, 0] / ly, cloud_final.num_points_raw[:, 1] / fyA]
                )
            except Exception as exc:
                line(
                    f"{sid} generalized set_id={eval_rid}: "
                    f"cloud plot arrays failed: {exc}"
                )
            else:
                try:
                    plot_unordered_binned_cloud_envelopes(
                        sid,
                        set_id_out,
                        exp_plot_xy,
                        num_plot_xy,
                        out_dir=plot_u_here,
                    )
                except Exception as exc:
                    line(
                        f"{sid} generalized set_id={eval_rid}: "
                        f"binned cloud envelope plotting failed: {exc}"
                    )

            saved_npz = save_simulated_force_history(
                hist_u_here, sid, set_id_out, D_drive, F_exp_na, F_sim
            )
            saved_csv = save_simulated_force_history_csv(
                hist_u_here, sid, set_id_out, D_drive, F_exp_na, F_sim
            )
            saved_artifacts(
                saved_npz.name if saved_npz is not None else None,
                saved_csv.name if saved_csv is not None else None,
            )

            u_row: dict[str, object] = {
                "Name": sid,
                "set_id": set_id_out,
                GENERALIZED_CONFIG_SET_ID_COL: eval_rid,
                "specimen_weight": specimen_w,
                "contributes_to_aggregate": contributes,
                **cm,
                "weight_config": weight_tag,
                "calibration_stage": "generalized_opt",
                "aggregate_by_set_id": by_set_id,
                **_metrics_dict_nan_prefix("initial"),
                **_metrics_dict_nan_prefix("final"),
                "initial_unordered_J_binenv": cloud_init.J_binenv,
                "final_unordered_J_binenv": cloud_final.J_binenv,
                "initial_unordered_J_binenv_l1": cloud_init.J_binenv_l1,
                "final_unordered_J_binenv_l1": cloud_final.J_binenv_l1,
                **_loss_weight_snapshot(loss_here),
                "S_F": s_f_cloud,
                "S_D": s_d_ref,
                "S_E": s_e_cloud,
                "P_y_ref": p_y_catalog,
                "n_cycles": 0,
                "success": bool(np.isfinite(cloud_final.J_binenv)),
            }
            rows_out.append(u_row)
            line(
                f"{sid} generalized set_id={eval_rid}: "
                "digitized unordered overlay + sim + "
                f"J_binenv(init={cloud_init.J_binenv:.6g}, final={cloud_final.J_binenv:.6g})"
            )

    specimen_set_param_rows: list[dict[str, Any]] = []
    specimen_names_in_order = sorted(set(available_resampled) | set(available_unordered))
    for merge_rid in param_merge_rids:
        if by_set_id:
            shared_for_param_merge = generalized_pool[merge_rid]
            active_for_param_merge = resolve_optimize_params_for_set_id(
                opt_map, merge_rid, default_list
            )
            alias_for_param_merge = alias_by_set_id.get(merge_rid, {})
        merge_g_row = _generalized_settings_row(settings_st, int(merge_rid))
        if by_set_id:
            parent_rid_merge = inherit_from_set.get(int(merge_rid))
            if parent_rid_merge is not None and parent_rid_merge in generalized_pool:
                merge_g_row, _ = _overlay_parent_on_settings_row(
                    merge_g_row,
                    generalized_pool[parent_rid_merge],
                    merge_g_row.get("steel_model"),
                )
        joint_skip = frozenset(
            generalized_joint_optimize_param_names(active_for_param_merge)
        ) & frozenset(BP_BN_SPECIMEN_LOCAL)
        for name in specimen_names_in_order:
            if name not in catalog_by_name.index:
                continue
            cat_row = catalog_by_name.loc[name]
            base = _prow_from_catalog_and_generalized_settings(
                cat_row,
                merge_g_row,
                specimen_name=name,
                generalized_config_set_id=int(merge_rid),
            )
            merged_out = merge_averaged_into_row(
                base.copy(), shared_for_param_merge, active_for_param_merge, skip_bp_bn_restore=joint_skip
            )
            merged_out[GENERALIZED_CONFIG_SET_ID_COL] = merge_rid
            sync_tied_columns_in_output_row(
                merged_out,
                alias_for_param_merge,
                frozenset(active_for_param_merge),
            )
            restore_individual_bp_bn(merged_out, base, skip=joint_skip)
            specimen_set_param_rows.append(merged_out.to_dict())
    p_df = pd.DataFrame(specimen_set_param_rows)
    out_param_cols = list(orig_param_columns)
    if GENERALIZED_CONFIG_SET_ID_COL not in out_param_cols:
        out_param_cols.append(GENERALIZED_CONFIG_SET_ID_COL)
    p_df = p_df[[c for c in out_param_cols if c in p_df.columns]]
    p_df = _dataframe_for_param_csv(p_df)
    out_params.parent.mkdir(parents=True, exist_ok=True)
    p_df.to_csv(out_params, index=False, float_format=OUTPUT_CSV_FLOAT_FMT)
    section("Outputs")
    n_blocks = len(param_merge_rids)
    kv(
        "wrote parameters",
        f"{out_params}  ({len(p_df)} rows, specimen set x {n_blocks} generalized "
        f"set_id block{'s' if n_blocks != 1 else ''})",
    )

    if not rows_out:
        line("no metrics rows (no successful evaluations).")
        return

    out_df = metrics_dataframe(rows_out)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_metrics, index=False)
    kv("wrote metrics", f"{out_metrics}  ({len(out_df)} rows)")
    try:
        summary_path = write_generalized_set_id_eval_summary(
            out_df, GENERALIZED_SET_ID_EVAL_SUMMARY_CSV
        )
        kv("wrote set_id eval summary", str(summary_path))
        j_train, j_val = write_generalized_unordered_j_split_summaries(out_df)
        kv("wrote unordered J summary (train)", str(j_train))
        kv("wrote unordered J summary (validation)", str(j_val))
    except Exception as exc:
        line(f"set_id / unordered-J summaries skipped: {exc}")


if __name__ == "__main__":
    main()
