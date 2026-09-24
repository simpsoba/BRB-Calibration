"""
Plot normalized force-deformation for each specimen (trimmed raw + filtered only).

Characteristic markers on path-ordered plots use ``data/cycle_points_original`` (``n`` must match
``data/filtered/{Name}/force_deformation.csv`` / trim-valid length).

``resampled_and_filtered`` overlays (with resampled markers) are produced by ``resample_filtered.py``.
Normalization: axial force / (f_y * A_sc), deformation / L_y (strain axis shown as percent).
Notation: P = axial force; f, f_y = stress; not F for force on figure axes.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import pandas as pd
import seaborn as sns

from cycle_points import load_cycle_points_for_trimmed_length
from plot_dimensions import (
    AXES_SPINE_LINEWIDTH,
    AXES_SPINE_LINEWIDTH_SINGLE_AX,
    COLOR_EXPERIMENTAL,
    HYSTERESIS_LINEWIDTH_SCALE,
    LEGEND_FONT_SIZE_SINGLE_AX_PT,
    LEGEND_FONT_SIZE_SMALL_PT,
    SAVE_DPI,
    PLOT_FONT_SIZE_PT,
    PLOT_FONT_SIZE_GRID_MONTAGE_PT,
    SINGLE_FIGSIZE_IN,
    configure_matplotlib_style,
    figsize_for_grid,
    figsize_two_stacked_axes,
    grid_montage_rcparams,
    single_axis_style_context,
    style_axes_spines_and_ticks,
    style_single_axis_spines,
)
from load_raw import load_raw_full, load_raw_valid
from specimen_catalog import (
    deformation_history_csv_path,
    filtered_deformation_history_csv,
    force_deformation_unordered_csv_path,
    get_specimen_record,
    list_names_for_filter_outputs,
    read_catalog,
    resampled_deformation_history_csv,
    resolve_filtered_force_deformation_csv,
    resolve_resampled_force_deformation_csv,
    uses_unordered_inputs,
)

# Project root: parent of scripts/postprocess -> scripts -> root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CATALOG_PATH = PROJECT_ROOT / "config" / "calibration" / "BRB-Specimens.csv"
# Reorganized plot output: force-deformation (F vs delta) and time histories (F, delta vs index)
PLOTS_BASE = PROJECT_ROOT / "results" / "plots" / "postprocess"
PLOTS_FORCE_DEF = PLOTS_BASE / "force_deformation"
PLOTS_FORCE_DEF_RAW_FILTERED = PLOTS_FORCE_DEF / "raw_and_filtered"
PLOTS_FORCE_DEF_TRIM_COMPARISON = PLOTS_FORCE_DEF / "trim_comparison"
PLOTS_TIME_HISTORIES = PLOTS_BASE / "time_histories"

# Markers and labels for characteristic points (cycle_points)
POINT_STYLE = {
    "zero_def": ("o", "green", r"$\delta=0$"),
    "zero_force": ("s", "gray", r"$P=0$"),
    "max_def": ("^", "darkred", r"$\delta_{\max}$"),
    "min_def": ("v", "darkblue", r"$\delta_{\min}$"),
    "zero_def+zero_force": ("o", "green", r"$\delta=0$"),
    "min_def+zero_def": ("v", "darkblue", r"$\delta_{\min}$"),
    "max_def+zero_force": ("^", "darkred", r"$\delta_{\max}$"),
}

sns.set_theme()
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
configure_matplotlib_style()


def _set_axes_frame(ax, *, linewidth: float | None = None) -> None:
    """Black frame on the data plane; ticks on all four sides (see ``style_axes_spines_and_ticks``)."""
    style_axes_spines_and_ticks(ax, linewidth=linewidth)


# Normalized axial P–δ: strain as percent; core yield use f_y, A_sc (matches catalog f_yc, A_c).
NORM_STRAIN_LABEL = r"Axial strain, $\delta/L_y$ (%)"
NORM_FORCE_LABEL = r"Axial force, $P/\left(f_y A_{sc}\right)$"
PHYS_FORCE_KIP_LABEL = r"Axial force $P$ [kip]"
ALL_SPECIMENS_GRID_COLS = 4


def _fraction_to_percent_tick_formatter(*, pct_decimals: int = 2) -> FuncFormatter:
    """Data are strain as fraction of unity; ticks show percent numbers (label carries ``(%)``)."""

    def _fmt_pct_val(x: float, _pos: int) -> str:
        """Format axis tick as percent string."""
        if not np.isfinite(x):
            return ""
        p = 100.0 * x
        if pct_decimals <= 0:
            return f"{p:.0f}"
        return f"{p:.{pct_decimals}f}"

    return FuncFormatter(_fmt_pct_val)


def apply_normalized_fu_axes(ax: plt.Axes, *, pct_decimals: int = 2) -> None:
    """Tick δ/L_y on **x** as percent numbers only; label already says (%)."""
    ax.xaxis.set_major_formatter(_fraction_to_percent_tick_formatter(pct_decimals=pct_decimals))


def apply_strain_fraction_ticks(ax: plt.Axes, *, axis: str = "y", pct_decimals: int = 2) -> None:
    """Tick δ/L_y on **x** or **y** as percent numbers (axis label should use ``NORM_STRAIN_LABEL``)."""
    fmt = _fraction_to_percent_tick_formatter(pct_decimals=pct_decimals)
    if axis == "y":
        ax.yaxis.set_major_formatter(fmt)
    else:
        ax.xaxis.set_major_formatter(fmt)


def get_specimens_with_raw() -> list[str]:
    """Specimens to plot: path-ordered primary F-u and/or digitized unordered inputs on disk."""
    return list_names_for_filter_outputs(read_catalog(), project_root=PROJECT_ROOT)


def load_specimen_data(
    specimen_id: str, catalog: pd.DataFrame | None = None
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    """Load raw F-u (unordered samples or path series), optional filtered, optional resampled ``force_deformation``."""
    cat = catalog if catalog is not None else read_catalog()
    rec = get_specimen_record(specimen_id, cat)
    if uses_unordered_inputs(rec):
        fd = force_deformation_unordered_csv_path(specimen_id, PROJECT_ROOT)
        raw_df = pd.read_csv(fd) if fd.is_file() else None
        fp = resolve_filtered_force_deformation_csv(specimen_id, PROJECT_ROOT)
        filtered_df = pd.read_csv(fp) if fp is not None and fp.is_file() else None
        rp = resolve_resampled_force_deformation_csv(specimen_id, PROJECT_ROOT)
        resampled_df = pd.read_csv(rp) if rp is not None and rp.is_file() else None
        return raw_df, filtered_df, resampled_df
    raw_df = load_raw_valid(specimen_id)
    fp = resolve_filtered_force_deformation_csv(specimen_id, PROJECT_ROOT)
    filtered_df = pd.read_csv(fp) if fp is not None and fp.is_file() else None
    rp = resolve_resampled_force_deformation_csv(specimen_id, PROJECT_ROOT)
    resampled_df = pd.read_csv(rp) if rp is not None and rp.is_file() else None
    return raw_df, filtered_df, resampled_df


def normalize(df: pd.DataFrame, f_yc: float, A_c: float, L_y: float) -> pd.DataFrame:
    r"""Normalize axial force (kip) and deformation to $P/(f_y A_{sc})$ and $\delta/L_y$ (fraction of unity)."""
    out = df.copy()
    out["Force_norm"] = df["Force[kip]"] / (f_yc * A_c)
    out["Deformation_norm"] = df["Deformation[in]"] / L_y
    return out


def _max_abs_finite(arr: np.ndarray) -> float:
    """Largest |x| over finite values; 0 if empty or all non-finite (avoids NaN axis limits)."""
    if arr is None or len(arr) == 0:
        return 0.0
    a = np.asarray(arr, dtype=float).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a)))


def compute_raw_filtered_global_norm_limits(
    catalog: pd.DataFrame,
    *,
    project_root: Path | None = None,
    margin: float = 1.05,
    specimens: list[str] | None = None,
) -> tuple[float, float]:
    """
    Positive half-ranges for symmetric normalized F–u axes: ``xlim=±x`` and ``ylim=±y``.

    Matches ``plot_all_specimens`` / ``all_specimens_raw_filtered.png``: over trimmed raw plus
    filtered ``Force[kip]`` / ``Deformation[in]``, normalized per specimen, take the largest
    ``|δ/L_y|`` and ``|F/(f_yc A_c)|``, then apply ``margin``.
    """
    root = project_root or PROJECT_ROOT
    names = specimens if specimens is not None else list_names_for_filter_outputs(catalog, project_root=root)
    catalog_by_name = catalog.set_index("Name")
    global_x_max = 0.0
    global_y_max = 0.0
    for specimen_id in names:
        catalog_row = catalog_by_name.loc[specimen_id]
        raw_df, filtered_df, _resampled_df = load_specimen_data(specimen_id, catalog)
        if raw_df is None:
            continue
        f_yc = float(catalog_row["fyp"])
        A_c = float(catalog_row["A_sc"])
        L_y = float(catalog_row["L_y"])
        raw_n = normalize(raw_df, f_yc, A_c, L_y)
        x_vals = raw_n["Deformation_norm"].values
        y_vals = raw_n["Force_norm"].values
        if filtered_df is not None:
            filtered_n = normalize(filtered_df, f_yc, A_c, L_y)
            x_vals = np.concatenate([x_vals, filtered_n["Deformation_norm"].values])
            y_vals = np.concatenate([y_vals, filtered_n["Force_norm"].values])
        global_x_max = max(global_x_max, _max_abs_finite(x_vals))
        global_y_max = max(global_y_max, _max_abs_finite(y_vals))
    if global_x_max == 0 or not np.isfinite(global_x_max):
        global_x_max = 1.0
    if global_y_max == 0 or not np.isfinite(global_y_max):
        global_y_max = 1.0
    global_x_max *= margin
    global_y_max *= margin
    return (global_x_max, global_y_max)


def set_symmetric_axes(ax, x_data: np.ndarray, y_data: np.ndarray, margin: float = 1.05) -> None:
    """Set x and y limits symmetric about zero (no equal aspect)."""
    x_max = _max_abs_finite(np.asarray(x_data)) * margin
    y_max = _max_abs_finite(np.asarray(y_data)) * margin
    if x_max == 0 or not np.isfinite(x_max):
        x_max = 1.0
    if y_max == 0 or not np.isfinite(y_max):
        y_max = 1.0
    ax.set_xlim(-x_max, x_max)
    ax.set_ylim(-y_max, y_max)


def plot_characteristic_points(
    ax, curve_n: pd.DataFrame, points: list[dict], add_legend: bool = True
) -> None:
    """Scatter characteristic points on axes. ``curve_n`` must have Deformation_norm, Force_norm; indices match ``points[].idx``."""
    if not points:
        return
    n = len(curve_n)
    used_labels = set()
    for pt in points:
        idx = pt.get("idx", -1)
        if idx < 0 or idx >= n:
            continue
        t = pt.get("type", "")
        style = POINT_STYLE.get(t) or POINT_STYLE.get(t.split("+")[0], ("o", "black", t))
        marker, color, label = style
        x = float(curve_n["Deformation_norm"].iloc[idx])
        y = float(curve_n["Force_norm"].iloc[idx])
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        lbl = label if add_legend and label not in used_labels else None
        if lbl:
            used_labels.add(label)
        ax.scatter(x, y, marker=marker, c=color, s=25, zorder=5, edgecolors="k", linewidths=0.5, label=lbl)


def scatter_characteristic_points_physical(
    ax: plt.Axes,
    df: pd.DataFrame,
    points: list[dict],
    *,
    def_col: str = "Deformation[in]",
    force_col: str = "Force[kip]",
    add_legend: bool = False,
    scatter_size: float = 25.0,
) -> None:
    """Scatter cycle points using physical F-u columns (for resampled overlay plots)."""
    if not points or def_col not in df.columns or force_col not in df.columns:
        return
    n = len(df)
    used_labels = set()
    for pt in points:
        idx = pt.get("idx", -1)
        if idx < 0 or idx >= n:
            continue
        t = pt.get("type", "")
        style = POINT_STYLE.get(t) or POINT_STYLE.get(t.split("+")[0], ("o", "black", t))
        marker, color, label = style
        x = float(df[def_col].iloc[idx])
        y = float(df[force_col].iloc[idx])
        lbl = label if add_legend and label not in used_labels else None
        if lbl:
            used_labels.add(label)
        ax.scatter(
            x,
            y,
            marker=marker,
            c=color,
            s=scatter_size,
            zorder=6,
            edgecolors="k",
            linewidths=0.5,
            label=lbl,
        )


def plot_one_specimen(
    specimen_id: str,
    catalog_row: pd.Series,
    raw_df: pd.DataFrame,
    filtered_df: pd.DataFrame | None,
    out_dir: Path,
    *,
    unordered_digitized: bool = False,
) -> None:
    """One figure per specimen: trimmed raw + filtered (normalized); markers in filtered index space."""
    f_yc = float(catalog_row["fyp"])
    A_c = float(catalog_row["A_sc"])
    L_y = float(catalog_row["L_y"])
    raw_n = normalize(raw_df, f_yc, A_c, L_y)
    filtered_n = normalize(filtered_df, f_yc, A_c, L_y) if filtered_df is not None else None

    with single_axis_style_context():
        fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE_IN, layout="constrained")
        if unordered_digitized:
            ax.scatter(
                raw_n["Deformation_norm"],
                raw_n["Force_norm"],
                label="Raw (unordered)",
                color=COLOR_EXPERIMENTAL,
                alpha=0.35,
                s=6,
                linewidths=0,
            )
            if filtered_n is not None:
                ax.scatter(
                    filtered_n["Deformation_norm"],
                    filtered_n["Force_norm"],
                    label="Filtered (unordered)",
                    color="tab:orange",
                    alpha=0.45,
                    s=6,
                    linewidths=0,
                )
        else:
            ax.plot(
                raw_n["Deformation_norm"],
                raw_n["Force_norm"],
                label="Trimmed",
                color=COLOR_EXPERIMENTAL,
                alpha=0.7,
                linewidth=0.8 * HYSTERESIS_LINEWIDTH_SCALE,
            )
            if filtered_n is not None:
                ax.plot(
                    filtered_n["Deformation_norm"],
                    filtered_n["Force_norm"],
                    label="Filtered",
                    color="tab:orange",
                    linewidth=1.0 * HYSTERESIS_LINEWIDTH_SCALE,
                )
        parts_x = [raw_n["Deformation_norm"].values]
        parts_y = [raw_n["Force_norm"].values]
        if filtered_n is not None:
            parts_x.append(filtered_n["Deformation_norm"].values)
            parts_y.append(filtered_n["Force_norm"].values)
        x_all = np.concatenate(parts_x)
        y_all = np.concatenate(parts_y)
        set_symmetric_axes(ax, x_all, y_all)
        if filtered_df is not None and filtered_n is not None:
            pts = load_cycle_points_for_trimmed_length(specimen_id, len(filtered_df))
            if pts:
                plot_characteristic_points(ax, filtered_n, pts, add_legend=True)
        ax.set_xlabel(NORM_STRAIN_LABEL)
        ax.set_ylabel(NORM_FORCE_LABEL)
        apply_normalized_fu_axes(ax)
        h, lab = ax.get_legend_handles_labels()
        fig.legend(
            h,
            lab,
            loc="outside upper center",
            ncol=3,
            fontsize=LEGEND_FONT_SIZE_SINGLE_AX_PT,
            frameon=False,
        )
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="k", linewidth=AXES_SPINE_LINEWIDTH_SINGLE_AX)
        ax.axvline(0, color="k", linewidth=AXES_SPINE_LINEWIDTH_SINGLE_AX)
        style_single_axis_spines(ax)
        fig.savefig(out_dir / f"{specimen_id}.png", dpi=SAVE_DPI)
        plt.close(fig)


def plot_time_histories_scatter_one(
    specimen_id: str, catalog_row: pd.Series, out_dir: Path
) -> None:
    """
    Digitized: top = raw + filtered deformation drive; bottom = resampled drive only.

    Ordinate is **axial strain** ``δ/L_y`` (same convention as normalized F–δ plots). Deformation-history
    CSVs do not carry axial force; there is no force panel here.

    The top panel uses a **normalized abscissa** in [0, 1] for each series (start → end of that CSV).
    Plotting raw index 0..n-1 and filtered ``Step`` 0..m-1 on one axis hides the shorter series when
    ``n != m`` (e.g. stale filtered file still on disk after the raw drive was replaced).
    """
    L_y = float(catalog_row["L_y"])
    if L_y <= 0 or not np.isfinite(L_y):
        L_y = 1.0

    dh_raw_path = deformation_history_csv_path(specimen_id, PROJECT_ROOT)
    dh_f_path = filtered_deformation_history_csv(specimen_id, PROJECT_ROOT)
    dh_r_path = resampled_deformation_history_csv(specimen_id, PROJECT_ROOT)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=figsize_two_stacked_axes(), layout="constrained", sharex=False
    )

    n0 = n1 = None
    if dh_raw_path.is_file():
        dh0 = pd.read_csv(dh_raw_path)
        if "Deformation[in]" in dh0.columns and len(dh0) > 0:
            n0 = len(dh0)
            x0 = np.linspace(0.0, 1.0, n0) if n0 > 1 else np.zeros(1)
            ax_top.plot(
                x0,
                dh0["Deformation[in]"].to_numpy(dtype=float) / L_y,
                color=COLOR_EXPERIMENTAL,
                alpha=0.55,
                linewidth=0.7,
                label="Raw drive",
            )
    if dh_f_path.is_file():
        dh1 = pd.read_csv(dh_f_path)
        if "Deformation[in]" in dh1.columns and len(dh1) > 0:
            n1 = len(dh1)
            x1 = np.linspace(0.0, 1.0, n1) if n1 > 1 else np.zeros(1)
            ax_top.plot(
                x1,
                dh1["Deformation[in]"].to_numpy(dtype=float) / L_y,
                color="tab:orange",
                linewidth=0.9,
                label="Filtered drive",
            )
    ax_top.set_ylabel(NORM_STRAIN_LABEL)
    ax_top.set_xlabel(r"Normalized position along drive (0 = first row, 1 = last)")
    apply_strain_fraction_ticks(ax_top, axis="y")
    ax_top.grid(True, alpha=0.3)
    ax_top.axhline(0, color="k", linewidth=AXES_SPINE_LINEWIDTH)
    h_top, lab_top = ax_top.get_legend_handles_labels()
    if h_top:
        ax_top.legend(
            loc="upper right",
            fontsize=LEGEND_FONT_SIZE_SMALL_PT,
            frameon=False,
        )
    _set_axes_frame(ax_top)

    if dh_r_path.is_file():
        dh2 = pd.read_csv(dh_r_path)
        if "Deformation[in]" in dh2.columns:
            s2 = dh2["Step"] if "Step" in dh2.columns else np.arange(len(dh2))
            ax_bot.plot(
                s2,
                dh2["Deformation[in]"].to_numpy(dtype=float) / L_y,
                color="tab:green",
                linewidth=0.9,
                label="Resampled drive",
            )
    ax_bot.set_ylabel(NORM_STRAIN_LABEL)
    ax_bot.set_xlabel("Step")
    apply_strain_fraction_ticks(ax_bot, axis="y")
    ax_bot.grid(True, alpha=0.3)
    ax_bot.axhline(0, color="k", linewidth=AXES_SPINE_LINEWIDTH)
    h_bot, lab_bot = ax_bot.get_legend_handles_labels()
    if h_bot:
        ax_bot.legend(
            loc="upper right",
            fontsize=LEGEND_FONT_SIZE_SMALL_PT,
            frameon=False,
        )
    _set_axes_frame(ax_bot)

    fig.savefig(out_dir / f"{specimen_id}.png", dpi=SAVE_DPI)
    plt.close(fig)


def plot_all_specimens(
    specimens: list[str],
    catalog: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Combined figure: one subplot per specimen, shared axes, combined legend.
    Axis limits use the largest |force| and |deformation| across all specimens so every one fits.
    """
    n = len(specimens)
    ncol = ALL_SPECIMENS_GRID_COLS
    nrow = (n + ncol - 1) // ncol
    catalog_by_name = catalog.set_index("Name")

    global_x_max, global_y_max = compute_raw_filtered_global_norm_limits(
        catalog, project_root=PROJECT_ROOT, specimens=specimens
    )

    with plt.rc_context(grid_montage_rcparams()):
        fig, axs = plt.subplots(
            nrow,
            ncol,
            figsize=figsize_for_grid(nrow, ncol),
            layout="constrained",
            sharex="col",
            sharey="row",
        )
        if n == 1:
            axs = np.array([[axs]])
        else:
            axs = np.asarray(axs).reshape(nrow, ncol)
        for idx, specimen_id in enumerate(specimens):
            row, col = idx // ncol, idx % ncol
            ax = axs[row, col]
            catalog_row = catalog_by_name.loc[specimen_id]
            raw_df, filtered_df, _resampled_df = load_specimen_data(specimen_id, catalog)
            if raw_df is None:
                continue
            unordered_digitized = uses_unordered_inputs(get_specimen_record(specimen_id, catalog))
            f_yc = float(catalog_row["fyp"])
            A_c = float(catalog_row["A_sc"])
            L_y = float(catalog_row["L_y"])
            raw_n = normalize(raw_df, f_yc, A_c, L_y)
            use_label = idx == 0
            filtered_n = None
            if unordered_digitized:
                ax.scatter(
                    raw_n["Deformation_norm"],
                    raw_n["Force_norm"],
                    color=COLOR_EXPERIMENTAL,
                    alpha=0.35,
                    s=4,
                    linewidths=0,
                    label="Raw (unordered)" if use_label else None,
                )
                if filtered_df is not None:
                    filtered_n = normalize(filtered_df, f_yc, A_c, L_y)
                    ax.scatter(
                        filtered_n["Deformation_norm"],
                        filtered_n["Force_norm"],
                        color="tab:orange",
                        alpha=0.4,
                        s=4,
                        linewidths=0,
                        label="Filtered (unordered)" if use_label else None,
                    )
            else:
                ax.plot(
                    raw_n["Deformation_norm"],
                    raw_n["Force_norm"],
                    color=COLOR_EXPERIMENTAL,
                    alpha=0.7,
                    linewidth=0.6 * HYSTERESIS_LINEWIDTH_SCALE,
                    label="Trimmed" if use_label else None,
                )
                if filtered_df is not None:
                    filtered_n = normalize(filtered_df, f_yc, A_c, L_y)
                    ax.plot(
                        filtered_n["Deformation_norm"],
                        filtered_n["Force_norm"],
                        color="tab:orange",
                        linewidth=0.8 * HYSTERESIS_LINEWIDTH_SCALE,
                        label="Filtered" if use_label else None,
                    )
            ax.set_xlim(-global_x_max, global_x_max)
            ax.set_ylim(-global_y_max, global_y_max)
            if not unordered_digitized and filtered_df is not None and filtered_n is not None:
                pts = load_cycle_points_for_trimmed_length(specimen_id, len(filtered_df))
                if pts:
                    plot_characteristic_points(ax, filtered_n, pts, add_legend=False)
            ax.set_title(specimen_id)
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color="k", linewidth=AXES_SPINE_LINEWIDTH)
            ax.axvline(0, color="k", linewidth=AXES_SPINE_LINEWIDTH)
            _set_axes_frame(ax)
            apply_normalized_fu_axes(ax)
        for j in range(n, nrow * ncol):
            row, col = j // ncol, j % ncol
            axs[row, col].set_visible(False)
        fig.supxlabel(NORM_STRAIN_LABEL, fontsize=PLOT_FONT_SIZE_GRID_MONTAGE_PT)
        fig.supylabel(NORM_FORCE_LABEL, fontsize=PLOT_FONT_SIZE_GRID_MONTAGE_PT)
        fig.legend(loc="outside upper right", ncol=3)
        fig.savefig(out_dir / "all_specimens_raw_filtered.png", dpi=SAVE_DPI)
        plt.close(fig)


def plot_raw_before_after_trim_one(
    specimen_id: str,
    catalog_row: pd.Series,
    raw_before: pd.DataFrame,
    raw_after: pd.DataFrame,
    out_dir: Path,
) -> None:
    """One figure per specimen: raw before trim vs raw after trim (normalized)."""
    f_yc = float(catalog_row["fyp"])
    A_c = float(catalog_row["A_sc"])
    L_y = float(catalog_row["L_y"])
    before_n = normalize(raw_before, f_yc, A_c, L_y)
    after_n = normalize(raw_after, f_yc, A_c, L_y)
    with single_axis_style_context():
        fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE_IN, layout="constrained")
        ax.plot(
            before_n["Deformation_norm"],
            before_n["Force_norm"],
            label="Before trim",
            color=COLOR_EXPERIMENTAL,
            alpha=0.6,
            linewidth=0.6 * HYSTERESIS_LINEWIDTH_SCALE,
        )
        ax.plot(
            after_n["Deformation_norm"],
            after_n["Force_norm"],
            label="After trim",
            color="tab:orange",
            linewidth=0.8 * HYSTERESIS_LINEWIDTH_SCALE,
        )
        x_all = np.concatenate([before_n["Deformation_norm"].values, after_n["Deformation_norm"].values])
        y_all = np.concatenate([before_n["Force_norm"].values, after_n["Force_norm"].values])
        set_symmetric_axes(ax, x_all, y_all)
        ax.set_xlabel(NORM_STRAIN_LABEL)
        ax.set_ylabel(NORM_FORCE_LABEL)
        apply_normalized_fu_axes(ax)
        h, lab = ax.get_legend_handles_labels()
        fig.legend(
            h,
            lab,
            loc="outside upper center",
            ncol=2,
            fontsize=LEGEND_FONT_SIZE_SINGLE_AX_PT,
            frameon=False,
        )
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="k", linewidth=AXES_SPINE_LINEWIDTH_SINGLE_AX)
        ax.axvline(0, color="k", linewidth=AXES_SPINE_LINEWIDTH_SINGLE_AX)
        style_single_axis_spines(ax)
        fig.savefig(out_dir / f"{specimen_id}.png", dpi=SAVE_DPI)
        plt.close(fig)


def plot_time_histories_one(
    specimen_id: str,
    catalog_row: pd.Series,
    raw_full: pd.DataFrame,
    raw_trimmed: pd.DataFrame,
    filtered_df: pd.DataFrame | None,
    points: list[dict],
    out_dir: Path,
) -> None:
    """One figure per specimen: normalized axial force and strain vs index; raw, trimmed, filtered; cycle points."""
    f_yc = float(catalog_row["fyp"])
    A_c = float(catalog_row["A_sc"])
    L_y = float(catalog_row["L_y"])
    fyA = f_yc * A_c
    if fyA <= 0 or not np.isfinite(fyA):
        fyA = 1.0
    if L_y <= 0 or not np.isfinite(L_y):
        L_y = 1.0

    fig, (ax_force, ax_def) = plt.subplots(
        2, 1, figsize=figsize_two_stacked_axes(), layout="constrained", sharex=True
    )
    n_raw = len(raw_full)
    n_trim = len(raw_trimmed)
    x_raw = np.arange(n_raw)
    x_trim = np.arange(n_trim)
    ax_force.plot(
        x_raw, raw_full["Force[kip]"].to_numpy(dtype=float) / fyA, color=COLOR_EXPERIMENTAL, alpha=0.5, linewidth=0.5
    )
    ax_force.plot(
        x_trim,
        raw_trimmed["Force[kip]"].to_numpy(dtype=float) / fyA,
        color="tab:orange",
        linewidth=0.6,
    )
    if filtered_df is not None and len(filtered_df) == n_trim:
        ax_force.plot(
            x_trim,
            filtered_df["Force[kip]"].to_numpy(dtype=float) / fyA,
            color="tab:green",
            linewidth=0.7,
        )
    ax_force.set_ylabel(NORM_FORCE_LABEL)
    ax_force.grid(True, alpha=0.3)
    ax_force.axhline(0, color="k", linewidth=AXES_SPINE_LINEWIDTH)

    ax_def.plot(
        x_raw,
        raw_full["Deformation[in]"].to_numpy(dtype=float) / L_y,
        color=COLOR_EXPERIMENTAL,
        alpha=0.5,
        linewidth=0.5,
    )
    ax_def.plot(
        x_trim,
        raw_trimmed["Deformation[in]"].to_numpy(dtype=float) / L_y,
        color="tab:orange",
        linewidth=0.6,
    )
    if filtered_df is not None and len(filtered_df) == n_trim:
        ax_def.plot(
            x_trim,
            filtered_df["Deformation[in]"].to_numpy(dtype=float) / L_y,
            color="tab:green",
            linewidth=0.7,
        )
    ax_def.set_xlabel("Point index")
    ax_def.set_ylabel(NORM_STRAIN_LABEL)
    apply_strain_fraction_ticks(ax_def, axis="y")
    ax_def.grid(True, alpha=0.3)
    ax_def.axhline(0, color="k", linewidth=AXES_SPINE_LINEWIDTH)

    # Mark characteristic points (indices in trimmed space); include zero_force (squares) on both subplots
    used_force = set()
    used_def = set()
    for pt in points:
        idx = pt.get("idx", -1)
        if idx < 0 or idx >= n_trim:
            continue
        t = pt.get("type", "")
        style = POINT_STYLE.get(t) or POINT_STYLE.get(t.split("+")[0], ("o", "black", t))
        marker, color, label = style
        # Force subplot: delta=0 (green circles), P=0, delta_max, delta_min
        if "zero_def" in t or "zero_force" in t or "max_def" in t or "min_def" in t:
            f_val = float(raw_trimmed["Force[kip]"].iloc[idx]) / fyA
            lbl = label if label not in used_force else None
            if lbl:
                used_force.add(label)
            ax_force.scatter(idx, f_val, marker=marker, c=color, s=30, zorder=5, edgecolors="k", linewidths=0.5, label=lbl)
        # Deformation subplot: delta=0, P=0 (squares), delta_max, delta_min
        if "zero_def" in t or "zero_force" in t or "max_def" in t or "min_def" in t:
            d_val = float(raw_trimmed["Deformation[in]"].iloc[idx]) / L_y
            lbl = label if label not in used_def else None
            if lbl:
                used_def.add(label)
            ax_def.scatter(idx, d_val, marker=marker, c=color, s=30, zorder=5, edgecolors="k", linewidths=0.5, label=lbl)
    _set_axes_frame(ax_force)
    _set_axes_frame(ax_def)

    # Single figure legend: Raw, Trimmed, Filtered, then characteristic points (no duplicates)
    legend_handles = [
        Line2D([0], [0], color=COLOR_EXPERIMENTAL, alpha=0.5, linewidth=1.5, label="Raw"),
        Line2D([0], [0], color="tab:orange", linewidth=1.5, label="Trimmed"),
    ]
    if filtered_df is not None and len(filtered_df) == n_trim:
        legend_handles.append(Line2D([0], [0], color="tab:green", linewidth=1.5, label="Filtered"))
    for key in ("zero_def", "zero_force", "max_def", "min_def"):
        marker, color, label = POINT_STYLE[key]
        legend_handles.append(
            Line2D(
                [0], [0], marker=marker, color="w", markerfacecolor=color, markeredgecolor="k",
                markeredgewidth=0.5, markersize=6, linestyle="", label=label,
            )
        )
    fig.legend(
        handles=legend_handles,
        loc="outside upper center",
        ncol=7,
        fontsize=LEGEND_FONT_SIZE_SMALL_PT,
        frameon=False,
    )
    fig.savefig(out_dir / f"{specimen_id}.png", dpi=SAVE_DPI)
    plt.close(fig)


def plot_raw_before_after_trim_all(
    specimens: list[str],
    catalog: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Combined figure: one subplot per specimen, before vs after trim, shared axes.
    Uses the same x/y limits as the raw_and_filtered all-specimens plot (from trimmed + filtered data).
    """
    n = len(specimens)
    ncol = ALL_SPECIMENS_GRID_COLS
    nrow = (n + ncol - 1) // ncol
    catalog_by_name = catalog.set_index("Name")
    # Same limits as plot_all_specimens: from trimmed raw + filtered
    global_x_max = 0.0
    global_y_max = 0.0
    for specimen_id in specimens:
        if uses_unordered_inputs(get_specimen_record(specimen_id, catalog)):
            continue
        raw_df, filtered_df, _resampled_df = load_specimen_data(specimen_id, catalog)
        if raw_df is None:
            continue
        catalog_row = catalog_by_name.loc[specimen_id]
        f_yc = float(catalog_row["fyp"])
        A_c = float(catalog_row["A_sc"])
        L_y = float(catalog_row["L_y"])
        raw_n = normalize(raw_df, f_yc, A_c, L_y)
        x_vals = raw_n["Deformation_norm"].values
        y_vals = raw_n["Force_norm"].values
        if filtered_df is not None:
            filtered_n = normalize(filtered_df, f_yc, A_c, L_y)
            x_vals = np.concatenate([x_vals, filtered_n["Deformation_norm"].values])
            y_vals = np.concatenate([y_vals, filtered_n["Force_norm"].values])
        global_x_max = max(global_x_max, _max_abs_finite(x_vals))
        global_y_max = max(global_y_max, _max_abs_finite(y_vals))
    margin = 1.05
    if global_x_max == 0 or not np.isfinite(global_x_max):
        global_x_max = 1.0
    if global_y_max == 0 or not np.isfinite(global_y_max):
        global_y_max = 1.0
    global_x_max *= margin
    global_y_max *= margin
    with plt.rc_context(grid_montage_rcparams()):
        fig, axs = plt.subplots(
            nrow,
            ncol,
            figsize=figsize_for_grid(nrow, ncol),
            layout="constrained",
            sharex="col",
            sharey="row",
        )
        if n == 1:
            axs = np.array([[axs]])
        else:
            axs = np.asarray(axs).reshape(nrow, ncol)
        trim_legend_done = False
        for idx, specimen_id in enumerate(specimens):
            row, col = idx // ncol, idx % ncol
            ax = axs[row, col]
            if uses_unordered_inputs(get_specimen_record(specimen_id, catalog)):
                ax.set_visible(False)
                continue
            raw_before = load_raw_full(specimen_id)
            raw_after = load_raw_valid(specimen_id)
            if raw_before is None or raw_after is None:
                ax.set_title(specimen_id)
                ax.set_xlim(-global_x_max, global_x_max)
                ax.set_ylim(-global_y_max, global_y_max)
                apply_normalized_fu_axes(ax)
                _set_axes_frame(ax)
                continue
            catalog_row = catalog_by_name.loc[specimen_id]
            f_yc = float(catalog_row["fyp"])
            A_c = float(catalog_row["A_sc"])
            L_y = float(catalog_row["L_y"])
            before_n = normalize(raw_before, f_yc, A_c, L_y)
            after_n = normalize(raw_after, f_yc, A_c, L_y)
            use_label = not trim_legend_done
            ax.plot(
                before_n["Deformation_norm"],
                before_n["Force_norm"],
                color=COLOR_EXPERIMENTAL,
                alpha=0.6,
                linewidth=0.5 * HYSTERESIS_LINEWIDTH_SCALE,
                label="Before trim" if use_label else None,
            )
            ax.plot(
                after_n["Deformation_norm"],
                after_n["Force_norm"],
                color="tab:orange",
                linewidth=0.6 * HYSTERESIS_LINEWIDTH_SCALE,
                label="After trim" if use_label else None,
            )
            if use_label:
                trim_legend_done = True
            ax.set_xlim(-global_x_max, global_x_max)
            ax.set_ylim(-global_y_max, global_y_max)
            ax.set_title(specimen_id)
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color="k", linewidth=AXES_SPINE_LINEWIDTH)
            ax.axvline(0, color="k", linewidth=AXES_SPINE_LINEWIDTH)
            _set_axes_frame(ax)
            apply_normalized_fu_axes(ax)
        for j in range(n, nrow * ncol):
            r, c = j // ncol, j % ncol
            axs[r, c].set_visible(False)
        fig.supxlabel(NORM_STRAIN_LABEL, fontsize=PLOT_FONT_SIZE_GRID_MONTAGE_PT)
        fig.supylabel(NORM_FORCE_LABEL, fontsize=PLOT_FONT_SIZE_GRID_MONTAGE_PT)
        if trim_legend_done:
            fig.legend(
                handles=[
                    Line2D(
                        [0],
                        [0],
                        color=COLOR_EXPERIMENTAL,
                        alpha=0.6,
                        linewidth=1.5 * HYSTERESIS_LINEWIDTH_SCALE,
                        label="Before trim",
                    ),
                    Line2D(
                        [0],
                        [0],
                        color="tab:orange",
                        linewidth=1.5 * HYSTERESIS_LINEWIDTH_SCALE,
                        label="After trim",
                    ),
                ],
                loc="outside upper right",
                ncol=2,
                frameon=False,
            )
        fig.savefig(out_dir / "all_specimens_raw_trim_comparison.png", dpi=SAVE_DPI)
        plt.close(fig)


def main() -> None:
    """CLI entry point."""
    PLOTS_FORCE_DEF_RAW_FILTERED.mkdir(parents=True, exist_ok=True)
    PLOTS_FORCE_DEF_TRIM_COMPARISON.mkdir(parents=True, exist_ok=True)
    PLOTS_TIME_HISTORIES.mkdir(parents=True, exist_ok=True)
    catalog = read_catalog(CATALOG_PATH)
    specimens = get_specimens_with_raw()
    if not specimens:
        print("No specimens with raw data found.")
        return
    catalog_by_name = catalog.set_index("Name")

    # Force-deformation: trimmed raw + filtered (markers from filtered-length cycle JSON)
    for specimen_id in specimens:
        raw_df, filtered_df, _resampled_df = load_specimen_data(specimen_id, catalog)
        if raw_df is None:
            continue
        catalog_row = catalog_by_name.loc[specimen_id]
        unordered_digitized = uses_unordered_inputs(get_specimen_record(specimen_id, catalog))
        plot_one_specimen(
            specimen_id,
            catalog_row,
            raw_df,
            filtered_df,
            PLOTS_FORCE_DEF_RAW_FILTERED,
            unordered_digitized=unordered_digitized,
        )
        rel = PLOTS_FORCE_DEF_RAW_FILTERED.relative_to(PROJECT_ROOT)
        print(f"Saved {rel / (specimen_id + '.png')}")
    plot_all_specimens(specimens, catalog, PLOTS_FORCE_DEF_RAW_FILTERED)
    rel_all = PLOTS_FORCE_DEF_RAW_FILTERED / "all_specimens_raw_filtered.png"
    print(f"Saved {rel_all.relative_to(PROJECT_ROOT)}")

    # Force-deformation: before vs after trim
    for specimen_id in specimens:
        if uses_unordered_inputs(get_specimen_record(specimen_id, catalog)):
            continue
        raw_before = load_raw_full(specimen_id)
        raw_after = load_raw_valid(specimen_id)
        if raw_before is None or raw_after is None:
            continue
        catalog_row = catalog_by_name.loc[specimen_id]
        plot_raw_before_after_trim_one(specimen_id, catalog_row, raw_before, raw_after, PLOTS_FORCE_DEF_TRIM_COMPARISON)
        print(f"Saved results/plots/force_deformation/trim_comparison/{specimen_id}.png")
    plot_raw_before_after_trim_all(specimens, catalog, PLOTS_FORCE_DEF_TRIM_COMPARISON)
    print(f"Saved results/plots/force_deformation/trim_comparison/all_specimens_raw_trim_comparison.png")

    # Time histories: path-ordered = F,u vs index + cycle points; digitized = drive + unordered panels
    for specimen_id in specimens:
        if uses_unordered_inputs(get_specimen_record(specimen_id, catalog)):
            plot_time_histories_scatter_one(
                specimen_id, catalog_by_name.loc[specimen_id], PLOTS_TIME_HISTORIES
            )
            print(f"Saved {PLOTS_TIME_HISTORIES.relative_to(PROJECT_ROOT) / (specimen_id + '.png')}")
            continue
        raw_full = load_raw_full(specimen_id)
        raw_trimmed = load_raw_valid(specimen_id)
        fp = resolve_filtered_force_deformation_csv(specimen_id, PROJECT_ROOT)
        filtered_df = pd.read_csv(fp) if fp is not None and fp.is_file() else None
        if raw_full is None or raw_trimmed is None:
            continue
        n_trim = len(raw_trimmed)
        points_th = load_cycle_points_for_trimmed_length(specimen_id, n_trim)
        plot_time_histories_one(
            specimen_id,
            catalog_by_name.loc[specimen_id],
            raw_full,
            raw_trimmed,
            filtered_df,
            points_th,
            PLOTS_TIME_HISTORIES,
        )
        print(f"Saved {PLOTS_TIME_HISTORIES.relative_to(PROJECT_ROOT) / (specimen_id + '.png')}")


if __name__ == "__main__":
    main()
