"""
Build **combined** normalized overlay figures (one per ``set_id``) as real matplotlib subplots,
not stitched images.

Reads ``{Name}_set{k}_simulated.csv`` from each calibration method's numerical-model history
directory (``Deformation[in], Force[kip], Force_sim[kip]`` — same rows as the resampled drive for
path-ordered specimens). Normalization and styling match ``plot_params_vs_filtered`` /
``plot_specimens``. **Digitized unordered** rows use samples from the unordered force–deformation set plus the
numerical-model response along the drive (experimental ``Force[kip]`` may be all-NaN in those CSVs).

Writes ``set{k}_combined_force_def_norm.png`` into each method's ``overlays/`` folder (or
``individual_optimize/overlays_initial_params/``, with params inferred as ``initial_brb_parameters.csv``.
For generalized **multi-config** runs the optimizer writes its per-config eval outputs into
``overlays/config_set_{k}/`` and ``..._simulated_force/config_set_{k}/`` subfolders; the script
discovers those subdirectories and writes the combined PNG into the matching
``overlays/config_set_{k}/set{k}_combined_force_def_norm.png``. Each subplot uses
the specimen name as its title. One figure-level legend for the grid.
Generalized figures use ``#001F3F`` solid ``Numerical (train)`` vs ``#FF0E00`` ``Numerical (validation)``
for specimens outside the training set (catalog ``generalized_weight`` zero, or digitized unordered panels).
Individual combined figures use the train color only.
Default targets: individual (3 columns) and generalized (4 columns).

Use ``--ignore NAME [NAME ...]`` to omit specimens from the combined grid (simulated CSVs are
unchanged; only the montage panels are dropped).
"""
from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
_SCRIPTS = _PROJECT_ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS / "postprocess"))

from calibrate.calibration_paths import (  # noqa: E402
    BRB_SPECIMENS_CSV,
    GENERALIZED_BRB_PARAMETERS_PATH,
    GENERALIZED_SIMULATED_FORCE_DIR,
    INITIAL_BRB_PARAMETERS_PATH,
    INITIAL_PARAMS_SIMULATED_FORCE_ALL_SPECIMENS_DIR,
    INDIVIDUAL_SIMULATED_FORCE_DIR,
    OPTIMIZED_BRB_PARAMETERS_PATH,
    PLOTS_GENERALIZED_OPTIMIZE,
    PLOTS_INDIVIDUAL_OPTIMIZE,
)
from calibrate.digitized_unordered_eval_lib import load_digitized_unordered_series  # noqa: E402
from calibrate.overlay_axis_specs import (  # noqa: E402
    COMBINED_NORM_OVERLAY_X,
    COMBINED_NORM_OVERLAY_Y,
    apply_symmetric_axis_specs,
)
from calibrate.plot_params_vs_filtered import (  # noqa: E402
    DIGITIZED_UNORDERED_LEGEND_MARKERSIZE_PT,
    DIGITIZED_UNORDERED_OVERLAY_SCATTER_S,
    LINEWIDTH_EXPERIMENTAL,
    LINEWIDTH_SIMULATED,
)
from calibrate.specimen_weights import make_generalized_weight_fn  # noqa: E402
from postprocess.plot_dimensions import (  # noqa: E402
    AXES_SPINE_LINEWIDTH,
    COLOR_EXPERIMENTAL,
    COLOR_NUMERICAL_COHORT,
    COLOR_NUMERICAL_COHORT_AUX,
    PLOT_FONT_SIZE_GRID_MONTAGE_PT,
    SAVE_DPI,
    configure_matplotlib_style,
    figsize_for_grid,
    overlay_grid_montage_rcparams,
    style_axes_spines_and_ticks,
)
from postprocess.plot_specimens import (  # noqa: E402
    NORM_FORCE_LABEL,
    NORM_STRAIN_LABEL,
)
from specimen_catalog import get_specimen_record, read_catalog, uses_unordered_inputs  # noqa: E402

plt.rcParams["figure.facecolor"] = "white"
configure_matplotlib_style()

CATALOG_PATH = BRB_SPECIMENS_CSV
SIM_PATTERN = re.compile(r"^(.+)_set(\d+)_simulated\.csv$", re.IGNORECASE)
GRID_COLS_INDIVIDUAL: int = 3
GRID_COLS_GENERALIZED_AVERAGED: int = 4

DEFAULT_TARGETS: tuple[tuple[Path, Path, Path, int], ...] = (
    (
        PLOTS_INDIVIDUAL_OPTIMIZE / "overlays",
        OPTIMIZED_BRB_PARAMETERS_PATH,
        INDIVIDUAL_SIMULATED_FORCE_DIR,
        GRID_COLS_INDIVIDUAL,
    ),
    (
        PLOTS_GENERALIZED_OPTIMIZE / "overlays",
        GENERALIZED_BRB_PARAMETERS_PATH,
        GENERALIZED_SIMULATED_FORCE_DIR,
        GRID_COLS_GENERALIZED_AVERAGED,
    ),
)


def _infer_from_overlay_dir(overlay_dir: Path) -> tuple[Path, Path] | None:
    """Map ``.../<method>_optimize/overlays`` to default params CSV and numerical-model CSV directory."""
    p = overlay_dir.resolve()
    if p.name == "overlays_initial_params":
        parent = p.parent
        if parent.name == "individual_optimize":
            return (
                INITIAL_BRB_PARAMETERS_PATH,
                parent / "initial_params_simulated_force",
            )
        return None
    if p.name == "overlays_initial_params_all_specimens":
        parent = p.parent
        if parent.name == "individual_optimize":
            return (
                INITIAL_BRB_PARAMETERS_PATH,
                INITIAL_PARAMS_SIMULATED_FORCE_ALL_SPECIMENS_DIR,
            )
        return None
    if p.name != "overlays":
        return None
    parent = p.parent
    if parent.name == "individual_optimize":
        return (
            parent / "optimized_brb_parameters.csv",
            parent / "optimized_brb_parameters_simulated_force",
        )
    if parent.name == "generalized_optimize":
        return (
            parent / "generalized_brb_parameters.csv",
            parent / "generalized_params_eval_metrics_simulated_force",
        )
    return None


def _grid_cols_for_overlay_dir(overlay_dir: Path) -> int:
    """Three columns for ``individual_optimize/overlays``; four for generalized (or unknown)."""
    p = overlay_dir.resolve()
    if p.parent.name == "individual_optimize" and p.name in (
        "overlays",
        "overlays_initial_params",
        "overlays_initial_params_all_specimens",
    ):
        return GRID_COLS_INDIVIDUAL
    return GRID_COLS_GENERALIZED_AVERAGED


def _discover_simulated_index(sim_dir: Path) -> dict[int, set[str]]:
    """set_id -> specimen names that have a ``*_set*_simulated.csv`` numerical history in ``sim_dir``."""
    out: dict[int, set[str]] = {}
    if not sim_dir.is_dir():
        return out
    for path in sim_dir.glob("*_set*_simulated.csv"):
        m = SIM_PATTERN.match(path.name)
        if not m:
            continue
        name, sid_s = m.group(1), m.group(2)
        out.setdefault(int(sid_s), set()).add(name)
    return out


_CONFIG_SET_DIR_PATTERN = re.compile(r"^config_set_(\d+)$")


def _config_set_subdirs(parent: Path) -> list[tuple[int, Path]]:
    """``[(K, parent/config_set_K), ...]`` sorted by K. Empty if ``parent`` lacks such subdirs."""
    if not parent.is_dir():
        return []
    found: list[tuple[int, Path]] = []
    for child in parent.iterdir():
        if not child.is_dir():
            continue
        m = _CONFIG_SET_DIR_PATTERN.match(child.name)
        if not m:
            continue
        found.append((int(m.group(1)), child))
    found.sort(key=lambda kv: kv[0])
    return found


def _expand_config_set_targets(
    targets: tuple[tuple[Path, Path, Path, int], ...],
) -> list[tuple[Path, Path, Path, int]]:
    """
    Expand any (out_dir, params_csv, sim_dir, grid_cols) where ``sim_dir`` contains
    ``config_set_K/`` subdirectories into one target per subdirectory pointing at
    (out_dir/config_set_K, params_csv, sim_dir/config_set_K, grid_cols).

    The generalized optimizer writes its per-config eval outputs into ``config_set_K/``
    subfolders (when ``plot_multi_cfg=True``), so combined montages must follow the same
    layout. Targets without such subdirs (e.g., the individual side, or single-config runs)
    pass through unchanged.
    """
    expanded: list[tuple[Path, Path, Path, int]] = []
    for out_dir, params_csv, sim_dir, grid_cols in targets:
        subdirs = _config_set_subdirs(sim_dir)
        if not subdirs:
            expanded.append((out_dir, params_csv, sim_dir, grid_cols))
            continue
        for _set_k, sim_sub in subdirs:
            out_sub = out_dir / sim_sub.name
            expanded.append((out_sub, params_csv, sim_sub, grid_cols))
    return expanded


def _order_specimens(names: set[str], catalog_names: list[str]) -> list[str]:
    """Catalog order first, then remaining names sorted."""
    cat_set = set(catalog_names)
    ordered = [n for n in catalog_names if n in names]
    tail = sorted(names - cat_set)
    return ordered + tail


def _normalize_specimen_name(name: str) -> str:
    return str(name).strip()


def _filter_ignored_specimens(
    specimen_names: list[str],
    ignore: list[str],
) -> list[str]:
    """Drop ``ignore`` names from the combined-grid specimen list (order preserved)."""
    if not ignore:
        return specimen_names
    ignore_set = {_normalize_specimen_name(n) for n in ignore if _normalize_specimen_name(n)}
    if not ignore_set:
        return specimen_names
    return [n for n in specimen_names if _normalize_specimen_name(n) not in ignore_set]


def _specimen_is_unordered_digitized(name: str, catalog: pd.DataFrame) -> bool:
    """True if catalog uses digitized unordered F–u."""
    try:
        return uses_unordered_inputs(get_specimen_record(name, catalog))
    except KeyError:
        return False


def _numerical_color_for_combined_cell(
    specimen_id: str,
    catalog: pd.DataFrame,
    stage_weight_fn: Callable[[str], float] | None,
) -> str:
    """
    Individual method: all ``Numerical`` traces use the train color (per-specimen optimization).
    Generalized: train color if the specimen has positive stage weight on the path-ordered objective;
    digitized-unordered panels and zero-weight path specimens use the validation color.
    """
    if stage_weight_fn is None:
        return COLOR_NUMERICAL_COHORT
    if _specimen_is_unordered_digitized(specimen_id, catalog):
        return COLOR_NUMERICAL_COHORT_AUX
    return COLOR_NUMERICAL_COHORT if stage_weight_fn(specimen_id) > 0.0 else COLOR_NUMERICAL_COHORT_AUX


def _stage_weight_fn_for_overlay_out_dir(
    out_dir: Path,
    catalog: pd.DataFrame,
) -> Callable[[str], float] | None:
    """
    Weight function for generalized combined grids; ``None`` for individual overlays.

    Matches both the legacy single-config layout (``generalized_optimize/overlays``) and the
    multi-config layout (``generalized_optimize/overlays/config_set_K``) so the train/validation
    color split is preserved for both.
    """
    p = out_dir.resolve()
    if p.parent.name == "generalized_optimize" and p.name in ("overlays", "overlays_train_mean_bn_bp"):
        return make_generalized_weight_fn(catalog)
    # Multi-config: out_dir is ``generalized_optimize/overlays/config_set_K``.
    if (
        _CONFIG_SET_DIR_PATTERN.match(p.name)
        and p.parent.name in ("overlays", "overlays_train_mean_bn_bp")
        and p.parent.parent.name == "generalized_optimize"
    ):
        return make_generalized_weight_fn(catalog)
    return None


def _params_row_for(params_df: pd.DataFrame, name: str, set_id: int) -> pd.Series | None:
    """First params row for (Name, set_id) or None."""
    name_s = str(name).strip()
    sid = int(set_id)
    sel = params_df[params_df["Name"].astype(str).str.strip() == name_s]
    if sel.empty:
        return None
    set_num = pd.to_numeric(sel["set_id"], errors="coerce")
    m = set_num == sid
    if not m.any():
        return None
    return sel.loc[m].iloc[0]


def _numerical_legend_line2ds_for_grid(
    specimen_names: list[str],
    catalog: pd.DataFrame,
    stage_weight_fn: Callable[[str], float] | None,
) -> tuple[list[Line2D], list[str]]:
    """Dashed numerical legend entries: train color only for individual; train + validation for generalized."""
    present = {_numerical_color_for_combined_cell(n, catalog, stage_weight_fn) for n in specimen_names}
    out_h: list[Line2D] = []
    out_labels: list[str] = []
    if stage_weight_fn is None:
        out_h.append(
            Line2D(
                [0],
                [0],
                color=COLOR_NUMERICAL_COHORT,
                linestyle="--",
                linewidth=LINEWIDTH_SIMULATED,
                label="Numerical",
            )
        )
        out_labels.append("Numerical")
        return out_h, out_labels
    if COLOR_NUMERICAL_COHORT in present:
        out_h.append(
            Line2D(
                [0],
                [0],
                color=COLOR_NUMERICAL_COHORT,
                linestyle="-",
                linewidth=LINEWIDTH_SIMULATED,
                label="Numerical (train)",
            )
        )
        out_labels.append("Numerical (train)")
    if COLOR_NUMERICAL_COHORT_AUX in present:
        out_h.append(
            Line2D(
                [0],
                [0],
                color=COLOR_NUMERICAL_COHORT_AUX,
                linestyle="--",
                linewidth=LINEWIDTH_SIMULATED,
                label="Numerical (validation)",
            )
        )
        out_labels.append("Numerical (validation)")
    return out_h, out_labels


def _combined_overlay_legend_handles_labels_and_handler_map(
    specimen_names: list[str],
    catalog: pd.DataFrame,
    *,
    stage_weight_fn: Callable[[str], float] | None,
) -> tuple[list, list[str], dict]:
    """
    Path-ordered and digitized-unordered panels are both experimental test data; use one legend entry
    with line and/or marker when both modalities appear in the same figure.
    """
    unordered_any = any(_specimen_is_unordered_digitized(n, catalog) for n in specimen_names)
    path_any = any(not _specimen_is_unordered_digitized(n, catalog) for n in specimen_names)
    num_h, num_lab = _numerical_legend_line2ds_for_grid(specimen_names, catalog, stage_weight_fn)

    if path_any and unordered_any:
        h_line = Line2D(
            [0],
            [0],
            color=COLOR_EXPERIMENTAL,
            linestyle="-",
            linewidth=LINEWIDTH_EXPERIMENTAL,
        )
        h_mark = Line2D(
            [0],
            [0],
            linestyle="None",
            marker="o",
            color=COLOR_EXPERIMENTAL,
            markersize=DIGITIZED_UNORDERED_LEGEND_MARKERSIZE_PT,
            alpha=0.55,
        )
        return (
            [(h_line, h_mark), *num_h],
            ["Experimental", *num_lab],
            {tuple: HandlerTuple(ndivide=None)},
        )

    handles: list = []
    labels: list[str] = []
    if path_any:
        handles.append(
            Line2D(
                [0],
                [0],
                color=COLOR_EXPERIMENTAL,
                linestyle="-",
                linewidth=LINEWIDTH_EXPERIMENTAL,
                label="Experimental",
            )
        )
        labels.append("Experimental")
    elif unordered_any:
        handles.append(
            Line2D(
                [0],
                [0],
                linestyle="None",
                marker="o",
                color=COLOR_EXPERIMENTAL,
                markersize=DIGITIZED_UNORDERED_LEGEND_MARKERSIZE_PT,
                alpha=0.55,
                label="Experimental",
            )
        )
        labels.append("Experimental")

    handles.extend(num_h)
    labels.extend(num_lab)
    return handles, labels, {}


def _ly_from_params(prow: pd.Series, catalog_row: pd.Series) -> float:
    """Effective brace length L_y from params row or catalog."""
    if "L_y" in prow.index and pd.notna(prow.get("L_y")):
        return float(prow["L_y"])
    return float(catalog_row["L_y"])


def _plot_one_subplot_norm(
    ax: plt.Axes,
    specimen_id: str,
    set_id: int,
    csv_path: Path,
    catalog: pd.DataFrame,
    params_df: pd.DataFrame,
    *,
    norm_x_half: float,
    norm_y_half: float,
    numerical_color: str,
) -> None:
    """One normalized overlay cell for combined montage."""
    cat_by = catalog.set_index("Name")
    if specimen_id not in cat_by.index:
        ax.text(0.5, 0.5, "not in catalog", ha="center", va="center", transform=ax.transAxes)
        return
    catalog_row = cat_by.loc[specimen_id]
    prow = _params_row_for(params_df, specimen_id, set_id)
    if prow is None:
        ax.text(0.5, 0.5, "no params row", ha="center", va="center", transform=ax.transAxes)
        return

    fy = float(prow["fyp"])
    A_c = float(prow["A_sc"])
    L_y = _ly_from_params(prow, catalog_row)
    fyA = fy * A_c
    if fyA <= 0 or L_y <= 0 or not np.isfinite(fyA) or not np.isfinite(L_y):
        ax.text(0.5, 0.5, "bad geometry", ha="center", va="center", transform=ax.transAxes)
        return

    df = pd.read_csv(csv_path)
    if "Deformation[in]" not in df.columns or "Force_sim[kip]" not in df.columns:
        ax.text(0.5, 0.5, "bad CSV cols", ha="center", va="center", transform=ax.transAxes)
        return
    D = df["Deformation[in]"].to_numpy(dtype=float)
    F_sim = df["Force_sim[kip]"].to_numpy(dtype=float)
    F_exp = df["Force[kip]"].to_numpy(dtype=float) if "Force[kip]" in df.columns else np.full_like(D, np.nan)

    is_unordered = _specimen_is_unordered_digitized(specimen_id, catalog)
    hx, hy = norm_x_half, norm_y_half

    if is_unordered:
        series = load_digitized_unordered_series(
            specimen_id,
            _PROJECT_ROOT,
            steel_row=prow,
            catalog_row=catalog_row,
        )
        if series is None:
            ax.text(0.5, 0.5, "unordered load fail", ha="center", va="center", transform=ax.transAxes)
            return
        _D_drive, u_c, F_c = series
        d_n = D / L_y
        F_sim_n = F_sim / fyA
        u_n = np.asarray(u_c, dtype=float) / L_y
        F_c_n = np.asarray(F_c, dtype=float) / fyA
        ax.scatter(
            u_n,
            F_c_n,
            s=DIGITIZED_UNORDERED_OVERLAY_SCATTER_S,
            alpha=0.35,
            c=COLOR_EXPERIMENTAL,
            edgecolors="none",
            rasterized=True,
        )
        ax.plot(
            d_n,
            F_sim_n,
            color=numerical_color,
            alpha=0.95,
            linewidth=LINEWIDTH_SIMULATED,
            linestyle="--",
        )
    else:
        if not np.any(np.isfinite(F_exp)):
            ax.text(0.5, 0.5, "no F_exp", ha="center", va="center", transform=ax.transAxes)
            return
        d_n = D / L_y
        F_exp_n = F_exp / fyA
        F_sim_n = F_sim / fyA
        ax.plot(
            d_n,
            F_exp_n,
            color=COLOR_EXPERIMENTAL,
            alpha=0.95,
            linewidth=LINEWIDTH_EXPERIMENTAL,
            linestyle="-",
        )
        ax.plot(
            d_n,
            F_sim_n,
            color=numerical_color,
            alpha=0.95,
            linewidth=LINEWIDTH_SIMULATED,
            linestyle="--",
        )

    ax.set_xlim(-hx, hx)
    ax.set_ylim(-hy, hy)
    ax.set_title(specimen_id)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="k", linewidth=AXES_SPINE_LINEWIDTH)
    ax.axvline(0, color="k", linewidth=AXES_SPINE_LINEWIDTH)
    style_axes_spines_and_ticks(ax)


def _label_outer_visible_panels(axs: np.ndarray, n_panels: int, ncol: int) -> None:
    """Perimeter tick labels on each visible panel (works with global sharex/sharey)."""
    for j in range(n_panels):
        row, col = j // ncol, j % ncol
        axs[row, col].label_outer()


def plot_combined_for_set(
    set_id: int,
    specimen_names: list[str],
    out_dir: Path,
    params_csv: Path,
    sim_dir: Path,
    catalog: pd.DataFrame,
    *,
    grid_cols: int,
    stage_weight_fn: Callable[[str], float] | None = None,
) -> bool:
    """Write set{k}_combined_force_def_norm.png for one set_id."""
    if not params_csv.is_file():
        print(f"  (skip set {set_id}) Missing parameters CSV: {params_csv}")
        return False
    params_df = pd.read_csv(params_csv)
    if "Name" not in params_df.columns or "set_id" not in params_df.columns:
        print(f"  (skip set {set_id}) parameters CSV needs Name and set_id: {params_csv}")
        return False

    n = len(specimen_names)
    if n == 0:
        return False
    ncol = max(1, int(grid_cols))
    nrow = int(np.ceil(n / ncol))
    x_spec = COMBINED_NORM_OVERLAY_X
    y_spec = COMBINED_NORM_OVERLAY_Y

    # Global sharex/sharey: uneven column heights (empty bottom cells) break sharex='col'
    # tick-label placement; label_outer() restores bottom-row x labels on every column.
    with plt.rc_context(overlay_grid_montage_rcparams()):
        fig, axs = plt.subplots(
            nrow,
            ncol,
            figsize=figsize_for_grid(nrow, ncol),
            layout="constrained",
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        axs = np.asarray(axs).reshape(nrow, ncol)

        fig.supxlabel(NORM_STRAIN_LABEL, fontsize=PLOT_FONT_SIZE_GRID_MONTAGE_PT)
        fig.supylabel(NORM_FORCE_LABEL, fontsize=PLOT_FONT_SIZE_GRID_MONTAGE_PT)

        for i, sid_name in enumerate(specimen_names):
            r, c = i // ncol, i % ncol
            ax = axs[r, c]
            csv_p = sim_dir / f"{sid_name}_set{set_id}_simulated.csv"
            if csv_p.is_file():
                _plot_one_subplot_norm(
                    ax,
                    sid_name,
                    set_id,
                    csv_p,
                    catalog,
                    params_df,
                    norm_x_half=x_spec.half,
                    norm_y_half=y_spec.half,
                    numerical_color=_numerical_color_for_combined_cell(sid_name, catalog, stage_weight_fn),
                )
            else:
                ax.text(0.5, 0.5, "—", ha="center", va="center", transform=ax.transAxes, color="0.5")
                ax.set_title(sid_name)

        n_cells = nrow * ncol
        for j in range(n, n_cells):
            r, c = j // ncol, j % ncol
            axs[r, c].set_visible(False)

        apply_symmetric_axis_specs(
            axs[0, 0],
            x=x_spec,
            y=y_spec,
            normalized_strain_x=True,
            pct_decimals=0,
            norm_force_decimals=0,
        )
        _label_outer_visible_panels(axs, n, ncol)

        leg_h, leg_lab, leg_handler_map = _combined_overlay_legend_handles_labels_and_handler_map(
            specimen_names,
            catalog,
            stage_weight_fn=stage_weight_fn,
        )
        if leg_h:
            fig.legend(
                handles=leg_h,
                labels=leg_lab,
                handler_map=leg_handler_map,
                loc="outside upper center",
                ncol=len(leg_h),
                handlelength=3.0,
                handleheight=1.15,
                frameon=False,
            )

        out_path = out_dir / f"set{set_id}_combined_force_def_norm.png"
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=SAVE_DPI, facecolor="white")
        plt.close(fig)
    return True


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(
        description=(
            "Write set{k}_combined_force_def_norm.png from numerical-model *_simulated.csv histories "
            "and the calibration parameters CSV (real subplots, not image stitching)."
        ),
    )
    p.add_argument(
        "--overlays-dir",
        type=str,
        default=None,
        help=(
            "Output directory for combined PNGs (default: each method's overlays/). "
            "If passed alone, params CSV and numerical-model CSV dir are inferred when this is "
            ".../<individual|generalized>_optimize/overlays."
        ),
    )
    p.add_argument(
        "--params",
        type=str,
        default=None,
        help="Parameters CSV (Name, set_id, fyp, A_sc, L_y, …). Default: inferred from overlays layout.",
    )
    p.add_argument(
        "--simulated-force-dir",
        type=str,
        default=None,
        help=(
            "Directory of numerical-model histories ({Name}_set{k}_simulated.csv). "
            "Default: inferred from calibration layout."
        ),
    )
    p.add_argument(
        "--ignore",
        nargs="*",
        default=[],
        metavar="NAME",
        help=(
            "Specimen Name(s) to omit from combined grid figures "
            "(e.g. --ignore STF01 or --ignore STF01 PC250)."
        ),
    )
    args = p.parse_args()

    catalog = read_catalog(CATALOG_PATH)
    catalog_names = catalog["Name"].astype(str).tolist()

    if args.overlays_dir or args.params or args.simulated_force_dir:
        if args.overlays_dir is None:
            print(
                "With --params or --simulated-force-dir (numerical-model CSV folder), "
                "also pass --overlays-dir (output folder)."
            )
            return
        od = Path(args.overlays_dir).expanduser().resolve()
        params_path = Path(args.params).expanduser().resolve() if args.params else None
        sim_d = Path(args.simulated_force_dir).expanduser().resolve() if args.simulated_force_dir else None
        inferred = _infer_from_overlay_dir(od) if (params_path is None or sim_d is None) else None
        if params_path is None:
            if inferred is None:
                print("Could not infer --params; pass it explicitly.")
                return
            params_path = inferred[0]
        if sim_d is None:
            if inferred is None:
                print("Could not infer numerical-model CSV directory (--simulated-force-dir); pass it explicitly.")
                return
            sim_d = inferred[1]
        gc = _grid_cols_for_overlay_dir(od)
        targets: tuple[tuple[Path, Path, Path, int], ...] = ((od, params_path, sim_d, gc),)
    else:
        targets = DEFAULT_TARGETS

    # Expand any target whose sim_dir contains ``config_set_K/`` subdirectories into one target
    # per subdir. Required for generalized multi-config runs, where the optimizer writes per-config
    # eval outputs (and per-specimen overlays) into ``overlays/config_set_K`` and
    # ``..._simulated_force/config_set_K``.
    targets = tuple(_expand_config_set_targets(targets))

    ignore_names = [_normalize_specimen_name(n) for n in args.ignore]
    ignore_names = [n for n in ignore_names if n]
    if ignore_names:
        print(f"Ignoring specimens in combined grids: {ignore_names}")

    any_written = False
    for out_dir, params_csv, sim_dir, grid_cols in targets:
        w_fn = _stage_weight_fn_for_overlay_out_dir(out_dir, catalog)
        idx = _discover_simulated_index(sim_dir)
        if not idx:
            print(f"  (skip) No numerical-model *_simulated.csv under {sim_dir}")
            continue
        for set_id in sorted(idx.keys()):
            specimen_names = _order_specimens(idx[set_id], catalog_names)
            specimen_names = _filter_ignored_specimens(specimen_names, ignore_names)
            if not specimen_names:
                print(f"  (skip set {set_id}) All specimens ignored under {sim_dir}")
                continue
            if plot_combined_for_set(
                set_id,
                specimen_names,
                out_dir,
                params_csv,
                sim_dir,
                catalog,
                grid_cols=grid_cols,
                stage_weight_fn=w_fn,
            ):
                any_written = True
                print(
                    f"Wrote {out_dir / f'set{set_id}_combined_force_def_norm.png'} "
                    f"({len(specimen_names)} specimens, {grid_cols}-column grid)"
                )

    if not any_written:
        print("No combined figures written (no input numerical-model *_simulated.csv found).")


if __name__ == "__main__":
    main()