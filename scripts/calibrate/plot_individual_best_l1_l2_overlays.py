"""
Plot per-specimen overlays for the *best* individual-optimization L2 and L1 fits.

- Experimental hysteresis is shown in grey.
- Numerical curves for the best L2 and best L1 (per specimen) are plotted on top.

Best set selection:
- Read ``results/calibration/individual_optimize/optimized_brb_parameters_metrics.csv``.
- Map each ``set_id`` to ``steel_model`` using ``optimized_brb_parameters.csv``.
- For each ``steel_model`` and each specimen with ``individual_optimize=true`` and ``success=true`` (excluding CB225),
  pick **within that model**:
  - best L2: minimum ``final_J_total`` among rows with any L2 weight > 0
  - best L1: minimum ``final_J_total`` among rows with any L1 weight > 0

Numerical histories are read from:
``results/calibration/individual_optimize/optimized_brb_parameters_simulated_force/{Name}_set{k}_simulated.csv``

Outputs:
- Under ``.../overlays_best_l1_l2/steelmpf/``: one PNG per specimen
  ``{Name}_bestL2_bestL1_force_def_norm.png`` and combined ``all_bestL2_bestL1_force_def_norm.png``.
- Best L1/L2 ``set_id`` are chosen **within** each ``steel_model`` (from ``optimized_brb_parameters.csv`` via ``set_id``).
- Combined metrics CSV at ``overlays_best_l1_l2/bestL2_bestL1_metrics_table.csv`` includes a ``steel_model`` column.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
_SCRIPTS = _PROJECT_ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS / "postprocess"))

from calibrate.calibration_paths import (  # noqa: E402
    BRB_SPECIMENS_CSV,
    INDIVIDUAL_SIMULATED_FORCE_DIR,
    OPTIMIZED_BRB_PARAMETERS_PATH,
    PLOTS_INDIVIDUAL_OPTIMIZE,
    SET_ID_SETTINGS_CSV,
)
from calibrate.set_id_settings import read_set_id_settings_table  # noqa: E402
from calibrate.steel_model import (  # noqa: E402
    STEEL_MODEL_STEELMPF,
    normalize_steel_model,
)
from postprocess.plot_dimensions import (  # noqa: E402
    COLOR_NUMERICAL_COHORT,
    COLOR_NUMERICAL_COHORT_AUX,
    HYSTERESIS_LINEWIDTH_SCALE_OVERLAYS,
    AXES_SPINE_LINEWIDTH,
    LEGEND_FONT_SIZE_SMALL_PT,
    OVERLAY_GRID_TICK_LENGTH_SCALE,
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
    apply_normalized_fu_axes,
)
from specimen_catalog import get_specimen_record, read_catalog  # noqa: E402

plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
configure_matplotlib_style()

EXPORT_DPI = 300


METRICS_CSV = (
    _PROJECT_ROOT
    / "results"
    / "calibration"
    / "individual_optimize"
    / "optimized_brb_parameters_metrics.csv"
)

EXPERIMENTAL_GREY = "0.65"
COLOR_L2 = COLOR_NUMERICAL_COHORT
COLOR_L1 = COLOR_NUMERICAL_COHORT_AUX
LW_EXP = 0.9 * 1.5
LW_NUM = 0.9
LW_FU_EXP = LW_EXP * HYSTERESIS_LINEWIDTH_SCALE_OVERLAYS
LW_FU_NUM = LW_NUM * HYSTERESIS_LINEWIDTH_SCALE_OVERLAYS


def _best_l1_l2_overlay_legend_handles() -> tuple[list[plt.Line2D], list[str]]:
    """Experimental + best L1/L2 line handles (colors match ``_plot_one_specimen_norm`` / force / stiffness panels)."""
    return (
        [
            plt.Line2D([0], [0], color=EXPERIMENTAL_GREY, linestyle="-", linewidth=LW_FU_EXP),
            plt.Line2D([0], [0], color=COLOR_L1, linestyle="--", linewidth=LW_FU_NUM),
            plt.Line2D([0], [0], color=COLOR_L2, linestyle="--", linewidth=LW_FU_NUM),
        ],
        ["Experimental", r"Best $L_1$", r"Best $L_2$"],
    )

# Multipanel force-history plot (selected specimens).
FORCE_HISTORY_SPECIMENS: tuple[str, ...] = ("PC160", "PC750B", "PC250", "PC3SB")
# Total figure height for selected force / tangent multipanel PNGs (was 16 in; 75% row stack).
SELECTED_HISTORIES_FIG_HEIGHT_IN = 16.0 * 0.75
STIFFNESS_HISTORY_YLIM: tuple[float, float] = (-0.05, 0.1)
STIFFNESS_HISTORY_Y_TICKS: tuple[float, float, float] = (-0.05, 0.0, 0.1)
STIFFNESS_HISTORY_XLIM: tuple[float, float] = (0.0, 400.0)
STIFFNESS_HISTORY_XLIM_MIDWINDOW: tuple[float, float] = (400.0, 800.0)
STIFFNESS_HISTORY_YLIM_MIDWINDOW: tuple[float, float] = (-0.05, 0.15)
STIFFNESS_HISTORY_Y_TICKS_MIDWINDOW: tuple[float, ...] = (-0.05, 0.0, 0.05, 0.1, 0.15)
TANGENT_STIFFNESS_MIDWINDOW_SPECIMENS: tuple[str, ...] = ("PC160", "PC750B")

PRIMARY_X_LABEL = NORM_STRAIN_LABEL
PRIMARY_Y_LABEL = NORM_FORCE_LABEL


def _fraction_to_percent_tick_formatter(*, pct_decimals: int = 1) -> FuncFormatter:
    """Data are strain as fraction of unity; ticks show percent numbers (label carries (%))."""

    def _fmt_pct_val(x: float, _pos: int) -> str:
        if not np.isfinite(x):
            return ""
        p = 100.0 * x
        if pct_decimals <= 0:
            return f"{p:.0f}"
        return f"{p:.{pct_decimals}f}"

    return FuncFormatter(_fmt_pct_val)


def _ceil_to_step(x: float, step: float) -> float:
    """Ceil x to the next multiple of step (for symmetric axis half-ranges)."""
    if not np.isfinite(x):
        return float("nan")
    if step <= 0:
        return float(x)
    return float(np.ceil(float(x) / float(step)) * float(step))


def _cluster_common_halfranges(
    values_by_name: dict[str, float],
    *,
    rel_tol: float,
    step: float,
) -> dict[str, float]:
    """
    Cluster positive values by <= rel_tol relative difference to the current cluster reference,
    assign each member the cluster's max, then ceil to the nearest step.
    """
    ordered = sorted(values_by_name.items(), key=lambda kv: kv[1])
    groups: list[dict] = []
    for name, v in ordered:
        if not np.isfinite(v) or v <= 0:
            continue
        if not groups:
            groups.append({"ref": float(v), "members": [name], "max": float(v)})
            continue
        g = groups[-1]
        ref = float(g["ref"])
        if ref > 0 and abs(float(v) - ref) / ref <= float(rel_tol):
            g["members"].append(name)
            g["max"] = max(float(g["max"]), float(v))
        else:
            groups.append({"ref": float(v), "members": [name], "max": float(v)})
    out: dict[str, float] = {}
    for g in groups:
        common = _ceil_to_step(float(g["max"]), step)
        for nm in g["members"]:
            out[str(nm)] = common
    return out


def _as_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return False
    s = str(x).strip().lower()
    return s in ("1", "true", "t", "yes", "y")


def _has_any_weight(row: pd.Series, cols: list[str]) -> bool:
    for c in cols:
        if c in row.index:
            v = row.get(c)
            try:
                if float(v) > 0.0:
                    return True
            except Exception:
                continue
    return False


def _set_id_to_steel_model(params_df: pd.DataFrame) -> dict[int, str]:
    """Map ``set_id`` -> normalized ``steel_model`` from optimized parameters CSV."""
    if "set_id" not in params_df.columns:
        return {}
    df = params_df.copy()
    df["_sid"] = pd.to_numeric(df["set_id"], errors="coerce")
    df = df[np.isfinite(df["_sid"])]
    out: dict[int, str] = {}
    for sid, g in df.groupby("_sid"):
        sm = (
            normalize_steel_model(g.iloc[0].get("steel_model"))
            if "steel_model" in g.columns
            else normalize_steel_model(None)
        )
        out[int(sid)] = sm
    return out


def _models_in_order(models: set[str]) -> list[str]:
    normalize_steel_model(None)
    out = sorted(models) if models else [STEEL_MODEL_STEELMPF]
    return [m for m in out if normalize_steel_model(m) == STEEL_MODEL_STEELMPF] or [STEEL_MODEL_STEELMPF]


def _pick_best_set_ids(metrics_df: pd.DataFrame, *, steel_model: str | None = None) -> dict[str, dict[str, int]]:
    """
    Returns: {Name: {"L2": set_id, "L1": set_id}}.

    If ``steel_model`` is set, restrict to rows whose ``steel_model`` column matches (after normalization).
    """
    required = {"Name", "set_id", "success", "individual_optimize", "final_J_total"}
    missing = sorted(required - set(metrics_df.columns))
    if missing:
        raise RuntimeError(f"metrics CSV missing columns: {missing}")

    l2_cols = ["W_FEAT_L2", "W_ENERGY_L2", "W_UNORDERED_BINENV_L2"]
    l1_cols = ["W_FEAT_L1", "W_ENERGY_L1", "W_UNORDERED_BINENV_L1"]

    out: dict[str, dict[str, int]] = {}
    df = metrics_df.copy()
    df["Name"] = df["Name"].astype(str).str.strip()
    df = df[df["Name"].str.upper() != "CB225"]
    df = df[df["individual_optimize"].map(_as_bool)]
    df = df[df["success"].map(_as_bool)]
    df["set_id_num"] = pd.to_numeric(df["set_id"], errors="coerce")
    df["J"] = pd.to_numeric(df["final_J_total"], errors="coerce")
    df = df[np.isfinite(df["set_id_num"]) & np.isfinite(df["J"])]
    if steel_model is not None:
        sm = normalize_steel_model(steel_model)
        if "steel_model" not in df.columns:
            raise RuntimeError(
                "plot_individual_best_l1_l2_overlays: expected merged ``steel_model`` on metrics "
                "(from parameters CSV by set_id)."
            )
        df = df[df["steel_model"].astype(str).str.strip().str.lower() == sm]

    for name, g in df.groupby("Name"):
        best: dict[str, int] = {}
        g2 = g[g.apply(lambda r: _has_any_weight(r, l2_cols), axis=1)]
        g1 = g[g.apply(lambda r: _has_any_weight(r, l1_cols), axis=1)]
        if not g2.empty:
            r2 = g2.sort_values("J", ascending=True).iloc[0]
            best["L2"] = int(r2["set_id_num"])
        if not g1.empty:
            r1 = g1.sort_values("J", ascending=True).iloc[0]
            best["L1"] = int(r1["set_id_num"])
        if "L2" in best and "L1" in best:
            out[str(name).strip()] = best
    return out


def _metrics_row_for(metrics_df: pd.DataFrame, name: str, set_id: int, objective: str) -> pd.Series | None:
    """
    Pick the metrics row for (Name, set_id) matching objective family ("L2" or "L1") via weights.
    """
    name_s = str(name).strip()
    sid = int(set_id)
    df = metrics_df.copy()
    df["Name"] = df["Name"].astype(str).str.strip()
    df["set_id_num"] = pd.to_numeric(df["set_id"], errors="coerce")
    sel = df[(df["Name"] == name_s) & (df["set_id_num"] == sid)]
    if sel.empty:
        return None

    l2_cols = ["W_FEAT_L2", "W_ENERGY_L2", "W_UNORDERED_BINENV_L2"]
    l1_cols = ["W_FEAT_L1", "W_ENERGY_L1", "W_UNORDERED_BINENV_L1"]
    want = str(objective).strip().upper()
    if want == "L2":
        m = sel.apply(lambda r: _has_any_weight(r, l2_cols), axis=1)
        sel = sel[m]
    elif want == "L1":
        m = sel.apply(lambda r: _has_any_weight(r, l1_cols), axis=1)
        sel = sel[m]
    else:
        return None
    if sel.empty:
        return None
    # If duplicates exist, prefer the minimum final_J_total.
    sel = sel.copy()
    sel["J"] = pd.to_numeric(sel.get("final_J_total"), errors="coerce")
    sel = sel.sort_values("J", ascending=True, na_position="last")
    return sel.iloc[0]


def _params_row_for(params_df: pd.DataFrame, name: str, set_id: int) -> pd.Series | None:
    sel = params_df[params_df["Name"].astype(str).str.strip() == str(name).strip()]
    if sel.empty:
        return None
    set_num = pd.to_numeric(sel["set_id"], errors="coerce")
    m = set_num == int(set_id)
    if not m.any():
        return None
    return sel.loc[m].iloc[0]


def _ly_from_params_or_catalog(prow: pd.Series, catalog_row: pd.Series) -> float:
    if "L_y" in prow.index and pd.notna(prow.get("L_y")):
        return float(prow["L_y"])
    return float(catalog_row["L_y"])


def _read_simulated_csv(sim_dir: Path, name: str, set_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    p = sim_dir / f"{name}_set{set_id}_simulated.csv"
    if not p.is_file():
        return None
    df = pd.read_csv(p)
    if "Deformation[in]" not in df.columns or "Force_sim[kip]" not in df.columns:
        return None
    D = df["Deformation[in]"].to_numpy(dtype=float)
    F_sim = df["Force_sim[kip]"].to_numpy(dtype=float)
    if "Force[kip]" in df.columns:
        F_exp = df["Force[kip]"].to_numpy(dtype=float)
    else:
        F_exp = np.full_like(D, np.nan)
    return D, F_exp, F_sim


def _delta_y_hat_from_params_row(prow: pd.Series) -> float | None:
    """Yield deformation scale (in): $\\hat{\\delta}_y = (f_{yp}/\\hat{E})L_T$ (same as tangent cumulative-x normalization)."""
    try:
        Fy = float(prow["fyp"])
        Ehat = float(prow["E"])
        L_T = float(prow["L_T"])
    except Exception:
        return None
    Dy = (Fy / Ehat) * L_T if (np.isfinite(Fy) and np.isfinite(Ehat) and np.isfinite(L_T) and Ehat != 0) else np.nan
    if not np.isfinite(Dy) or Dy == 0:
        return None
    return float(Dy)


def _plot_force_history_panel(
    ax: plt.Axes,
    *,
    specimen: str,
    sim_dir: Path,
    params_df: pd.DataFrame,
    set_l1: int,
    set_l2: int,
    y_half: float | None = None,
    x_mode: str = "index",
) -> bool:
    """
    Plot normalized force history: Experimental + simulated (best L1 and best L2).

    ``x_mode`` ``"index"``: x is step index (labels hidden). ``"cum_abs_deformation"``: x is
    $\\sum|\\Delta\\delta|/\\hat{\\delta}_y$ per trace (L1 uses its drive and parameters).
    """
    prow2 = _params_row_for(params_df, specimen, set_l2)
    prow1 = _params_row_for(params_df, specimen, set_l1)
    if prow2 is None or prow1 is None:
        ax.text(0.5, 0.5, "no params", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(specimen, pad=10.0)
        return False
    fy = float(prow2["fyp"])
    A_c = float(prow2["A_sc"])
    fyA = fy * A_c
    if not np.isfinite(fyA) or fyA <= 0:
        ax.text(0.5, 0.5, "bad fyA", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(specimen, pad=10.0)
        return False

    d2 = _read_simulated_csv(sim_dir, specimen, set_l2)
    d1 = _read_simulated_csv(sim_dir, specimen, set_l1)
    if d2 is None or d1 is None:
        ax.text(0.5, 0.5, "missing CSV", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(specimen, pad=10.0)
        return False
    D2, F_exp, F_l2 = d2
    D1, _F_exp2, F_l1 = d1
    n = int(
        min(
            F_exp.shape[0],
            F_l1.shape[0],
            F_l2.shape[0],
            np.asarray(D2).shape[0],
            np.asarray(D1).shape[0],
        )
    )
    if n <= 1:
        ax.text(0.5, 0.5, "empty", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(specimen, pad=10.0)
        return False

    D2 = np.asarray(D2[:n], dtype=float)
    D1 = np.asarray(D1[:n], dtype=float)
    F_exp = np.asarray(F_exp[:n], dtype=float)
    F_l1 = np.asarray(F_l1[:n], dtype=float)
    F_l2 = np.asarray(F_l2[:n], dtype=float)

    if x_mode == "cum_abs_deformation":
        Dy2 = _delta_y_hat_from_params_row(prow2)
        Dy1 = _delta_y_hat_from_params_row(prow1)
        if Dy1 is None or Dy2 is None:
            ax.text(0.5, 0.5, "bad Dy", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(specimen, pad=10.0)
            return False
        x_exp = _cum_abs_deformation_over_Dy(D2, Dy=Dy2)
        x_l1 = _cum_abs_deformation_over_Dy(D1, Dy=Dy1)
        x_l2 = _cum_abs_deformation_over_Dy(D2, Dy=Dy2)
    else:
        x_exp = np.arange(n, dtype=float)
        x_l1 = x_exp
        x_l2 = x_exp

    ax.plot(
        x_exp,
        F_exp / fyA,
        color=EXPERIMENTAL_GREY,
        linewidth=LW_EXP,
        alpha=0.95,
        zorder=1,
    )
    ax.plot(
        x_l1,
        F_l1 / fyA,
        color=COLOR_L1,
        linestyle="--",
        linewidth=LW_NUM,
        alpha=0.95,
        zorder=3,
    )
    ax.plot(
        x_l2,
        F_l2 / fyA,
        color=COLOR_L2,
        linestyle="--",
        linewidth=LW_NUM,
        alpha=0.95,
        zorder=4,
    )
    ax.set_title(specimen, pad=10.0)
    if x_mode == "index":
        ax.tick_params(axis="x", which="both", labelbottom=False)
    if y_half is not None and np.isfinite(y_half) and float(y_half) > 0:
        yh = float(y_half)
        ax.set_ylim(-yh, yh)
        ax.set_yticks([-yh, 0.0, yh])
    ax.grid(True, alpha=0.25)
    ax.axhline(0, color="k", linewidth=AXES_SPINE_LINEWIDTH)
    style_axes_spines_and_ticks(ax)
    return True


def _max_cum_abs_deformation_ratio_across_sel(
    sel: list[str],
    *,
    best: dict,
    params_df: pd.DataFrame,
    sim_dir: Path,
) -> float:
    """Maximum cumulative absolute deformation ratio among experimental/L1/L2 traces for listed specimens."""
    x_max_all = 0.0
    for s in sel:
        prow2 = _params_row_for(params_df, s, int(best[s]["L2"]))
        prow1 = _params_row_for(params_df, s, int(best[s]["L1"]))
        if prow2 is None or prow1 is None:
            continue
        Dy2 = _delta_y_hat_from_params_row(prow2)
        Dy1 = _delta_y_hat_from_params_row(prow1)
        if Dy1 is None or Dy2 is None:
            continue
        d2 = _read_simulated_csv(sim_dir, s, int(best[s]["L2"]))
        d1 = _read_simulated_csv(sim_dir, s, int(best[s]["L1"]))
        if d2 is None or d1 is None:
            continue
        D2, F_exp, F_l2 = d2
        D1, _F_e2, F_l1 = d1
        n = int(
            min(
                F_exp.shape[0],
                F_l1.shape[0],
                F_l2.shape[0],
                np.asarray(D2).shape[0],
                np.asarray(D1).shape[0],
            )
        )
        if n <= 1:
            continue
        D2a = np.asarray(D2[:n], dtype=float)
        D1a = np.asarray(D1[:n], dtype=float)
        for x_arr in (
            _cum_abs_deformation_over_Dy(D2a, Dy=Dy2),
            _cum_abs_deformation_over_Dy(D1a, Dy=Dy1),
        ):
            xm = float(np.nanmax(x_arr))
            if np.isfinite(xm):
                x_max_all = max(x_max_all, xm)
    return float(x_max_all)


def _tangent_stiffness_history(
    u: np.ndarray,
    f: np.ndarray,
) -> np.ndarray:
    """
    Tangent stiffness history df/du using:
    - forward diff at i=0
    - backward diff at i=n-1
    - central diff otherwise
    """
    u = np.asarray(u, dtype=float)
    f = np.asarray(f, dtype=float)
    n = int(min(u.shape[0], f.shape[0]))
    if n <= 1:
        return np.full(n, np.nan, dtype=float)
    u = u[:n]
    f = f[:n]
    k = np.full(n, np.nan, dtype=float)

    du0 = u[1] - u[0]
    if np.isfinite(du0) and du0 != 0:
        k[0] = (f[1] - f[0]) / du0

    for i in range(1, n - 1):
        du = u[i + 1] - u[i - 1]
        if not np.isfinite(du) or du == 0:
            continue
        k[i] = (f[i + 1] - f[i - 1]) / du

    duN = u[n - 1] - u[n - 2]
    if np.isfinite(duN) and duN != 0:
        k[n - 1] = (f[n - 1] - f[n - 2]) / duN

    return k


def _cum_abs_deformation_over_Dy(u: np.ndarray, *, Dy: float) -> np.ndarray:
    """Cumulative absolute deformation, Σ|Δu|, normalized by Dy."""
    u = np.asarray(u, dtype=float)
    n = int(u.shape[0])
    if n <= 0:
        return np.zeros(0, dtype=float)
    if not np.isfinite(Dy) or Dy == 0:
        return np.full(n, np.nan, dtype=float)
    du = np.diff(u)
    cum = np.concatenate([[0.0], np.cumsum(np.abs(du))])
    return cum / float(Dy)


def _pointwise_inelastic_deformation(delta: np.ndarray, *, delta_y: float) -> np.ndarray:
    """δ_inel = sign(δ) max(|δ| − δ_y, 0) (vectorized)."""
    d = np.asarray(delta, dtype=float)
    mag = np.maximum(np.abs(d) - float(delta_y), 0.0)
    return np.sign(d) * mag


def _cum_inelastic_deformation_over_deltay(delta: np.ndarray, *, delta_y: float) -> np.ndarray:
    """Cumulative Σ|Δδ_inel| along the path, with δ_inel from ``_pointwise_inelastic_deformation``, normalized by δ_y."""
    d = np.asarray(delta, dtype=float)
    n = int(d.shape[0])
    if n <= 0:
        return np.zeros(0, dtype=float)
    if not np.isfinite(delta_y) or delta_y == 0:
        return np.full(n, np.nan, dtype=float)
    delta_inel = _pointwise_inelastic_deformation(d, delta_y=delta_y)
    d_delta_inel = np.diff(delta_inel)
    cum = np.concatenate([[0.0], np.cumsum(np.abs(d_delta_inel))])
    return cum / float(delta_y)


def _plot_tangent_stiffness_panel(
    ax: plt.Axes,
    *,
    specimen: str,
    sim_dir: Path,
    params_df: pd.DataFrame,
    set_l1: int,
    set_l2: int,
    cumulative_x: str = "abs",
    y_lim: tuple[float, float] | None = None,
    y_ticks: tuple[float, ...] | None = None,
) -> bool:
    """
    Plot tangent stiffness history normalized by initial stiffness K0 = Ehat*A_sc/L_T.

    Uses experimental force/deformation (from the L2 simulated history CSV) and simulated forces for L1/L2.

    ``cumulative_x``: ``"abs"`` uses Σ|Δδ|/δ̂_y; ``"inelastic"`` uses Σ|Δδ_inel|/δ̂_y with
    δ_inel = sign(δ) max(|δ| − δ̂_y, 0).

    Optional ``y_lim`` / ``y_ticks`` override the default stiffness-ratio axis window.
    """
    p2 = _params_row_for(params_df, specimen, set_l2)
    p1 = _params_row_for(params_df, specimen, set_l1)
    if p2 is None or p1 is None:
        ax.text(0.5, 0.5, "no params", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(specimen, pad=10.0)
        return False

    d2 = _read_simulated_csv(sim_dir, specimen, set_l2)
    d1 = _read_simulated_csv(sim_dir, specimen, set_l1)
    if d2 is None or d1 is None:
        ax.text(0.5, 0.5, "missing CSV", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(specimen, pad=10.0)
        return False

    D2, F_exp, F_l2 = d2
    D1, _F_exp2, F_l1 = d1
    n = int(min(D2.shape[0], F_exp.shape[0], F_l2.shape[0], D1.shape[0], F_l1.shape[0]))
    if n <= 2:
        ax.text(0.5, 0.5, "too short", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(specimen, pad=10.0)
        return False

    D2 = np.asarray(D2[:n], dtype=float)
    D1 = np.asarray(D1[:n], dtype=float)
    F_exp = np.asarray(F_exp[:n], dtype=float)
    F_l1 = np.asarray(F_l1[:n], dtype=float)
    F_l2 = np.asarray(F_l2[:n], dtype=float)

    # Initial stiffness K0 per curve (use the parameters row for that curve).
    def _k0_from(prow: pd.Series) -> float | None:
        try:
            Ehat = float(prow["E"])
            A_sc = float(prow["A_sc"])
            L_T = float(prow["L_T"])
        except Exception:
            return None
        k0 = (Ehat * A_sc / L_T) if (np.isfinite(Ehat) and np.isfinite(A_sc) and np.isfinite(L_T) and L_T != 0) else np.nan
        if not np.isfinite(k0) or k0 == 0:
            return None
        return float(k0)

    k0_exp = _k0_from(p2)  # experimental uses L2 stiffness definition
    k0_l1 = _k0_from(p1)
    k0_l2 = _k0_from(p2)
    Dy_exp = _delta_y_hat_from_params_row(p2)
    Dy_l1 = _delta_y_hat_from_params_row(p1)
    Dy_l2 = _delta_y_hat_from_params_row(p2)
    if k0_exp is None or k0_l1 is None or k0_l2 is None or Dy_exp is None or Dy_l1 is None or Dy_l2 is None:
        ax.text(0.5, 0.5, "bad K0", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(specimen, pad=10.0)
        return False

    k_exp = _tangent_stiffness_history(D2, F_exp) / k0_exp
    k_l1 = _tangent_stiffness_history(D1, F_l1) / k0_l1
    k_l2 = _tangent_stiffness_history(D2, F_l2) / k0_l2

    # X axis: cumulative measure normalized by δ̂_y (Dy from parameters).
    if cumulative_x == "inelastic":
        x_exp = _cum_inelastic_deformation_over_deltay(D2, delta_y=Dy_exp)
        x_l1 = _cum_inelastic_deformation_over_deltay(D1, delta_y=Dy_l1)
        x_l2 = _cum_inelastic_deformation_over_deltay(D2, delta_y=Dy_l2)
    else:
        x_exp = _cum_abs_deformation_over_Dy(D2, Dy=Dy_exp)
        x_l1 = _cum_abs_deformation_over_Dy(D1, Dy=Dy_l1)
        x_l2 = _cum_abs_deformation_over_Dy(D2, Dy=Dy_l2)

    ax.plot(x_exp, k_exp, color=EXPERIMENTAL_GREY, linewidth=LW_EXP, alpha=0.95, zorder=1)
    ax.plot(x_l1, k_l1, color=COLOR_L1, linestyle="--", linewidth=LW_NUM, alpha=0.95, zorder=3)
    ax.plot(x_l2, k_l2, color=COLOR_L2, linestyle="--", linewidth=LW_NUM, alpha=0.95, zorder=4)
    ax.set_title(specimen, pad=10.0)
    ylo, yhi = STIFFNESS_HISTORY_YLIM if y_lim is None else y_lim
    ax.set_ylim(ylo, yhi)
    ax.set_yticks(list(STIFFNESS_HISTORY_Y_TICKS) if y_ticks is None else list(y_ticks))
    ax.grid(True, alpha=0.25)
    ax.axhline(0, color="k", linewidth=AXES_SPINE_LINEWIDTH)
    style_axes_spines_and_ticks(ax)
    return True


def _plot_one_specimen_norm(
    ax: plt.Axes,
    name: str,
    set_l2: int,
    set_l1: int,
    *,
    sim_dir: Path,
    params_df: pd.DataFrame,
    catalog: pd.DataFrame,
    norm_xy_half: tuple[float, float] | None,
    title: bool = True,
    show_primary_axis_labels: bool = True,
) -> bool:
    cat_by = catalog.set_index("Name")
    if name not in cat_by.index:
        ax.text(0.5, 0.5, "not in catalog", ha="center", va="center", transform=ax.transAxes)
        return False
    catalog_row = cat_by.loc[name]
    if not get_specimen_record(name, catalog).individual_optimize:
        ax.text(0.5, 0.5, "not path-ordered", ha="center", va="center", transform=ax.transAxes)
        return False

    prow2 = _params_row_for(params_df, name, set_l2)
    prow1 = _params_row_for(params_df, name, set_l1)
    if prow2 is None or prow1 is None:
        ax.text(0.5, 0.5, "no params", ha="center", va="center", transform=ax.transAxes)
        return False

    D2 = _read_simulated_csv(sim_dir, name, set_l2)
    D1 = _read_simulated_csv(sim_dir, name, set_l1)
    if D2 is None or D1 is None:
        ax.text(0.5, 0.5, "missing sim CSV", ha="center", va="center", transform=ax.transAxes)
        return False

    D_l2, F_exp, F_l2 = D2
    D_l1, _F_exp2, F_l1 = D1
    if D_l1.shape != D_l2.shape:
        ax.text(0.5, 0.5, "drive mismatch", ha="center", va="center", transform=ax.transAxes)
        return False

    fy = float(prow2["fyp"])
    A_c = float(prow2["A_sc"])
    L_y = _ly_from_params_or_catalog(prow2, catalog_row)
    fyA = fy * A_c
    if fyA <= 0 or L_y <= 0 or not np.isfinite(fyA) or not np.isfinite(L_y):
        ax.text(0.5, 0.5, "bad geometry", ha="center", va="center", transform=ax.transAxes)
        return False

    # Single (primary) axes: δ/Ly and P/(fy A_sc)
    d_n = D_l2 / L_y
    F_exp_n = F_exp / fyA
    F_l2_n = F_l2 / fyA
    F_l1_n = F_l1 / fyA

    if not np.any(np.isfinite(F_exp_n)):
        ax.text(0.5, 0.5, "no F_exp", ha="center", va="center", transform=ax.transAxes)
        return False

    ax.plot(
        d_n,
        F_exp_n,
        color=EXPERIMENTAL_GREY,
        alpha=0.95,
        linewidth=LW_FU_EXP,
        linestyle="-",
        zorder=1,
    )
    ax.plot(
        d_n,
        F_l1_n,
        color=COLOR_L1,
        alpha=0.95,
        linewidth=LW_FU_NUM,
        linestyle="--",
        zorder=3,
    )
    ax.plot(
        d_n,
        F_l2_n,
        color=COLOR_L2,
        alpha=0.95,
        linewidth=LW_FU_NUM,
        linestyle="--",
        zorder=4,
    )

    if norm_xy_half is not None:
        hx, hy = norm_xy_half
        ax.set_xlim(-hx, hx)
        ax.set_ylim(-hy, hy)
    if title:
        ax.set_title(name, pad=10.0)
    if show_primary_axis_labels:
        ax.set_xlabel(PRIMARY_X_LABEL)
        ax.set_ylabel(PRIMARY_Y_LABEL)
    apply_normalized_fu_axes(ax, pct_decimals=1)
    if norm_xy_half is not None:
        hx, hy = norm_xy_half
        ax.set_xticks([-hx, 0.0, hx])
        ax.set_yticks([-hy, 0.0, hy])
    # Add breathing room for text vs data.
    ax.tick_params(axis="both", which="both", pad=6.0)
    ax.xaxis.labelpad = 10.0
    ax.yaxis.labelpad = 12.0
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="k", linewidth=AXES_SPINE_LINEWIDTH)
    ax.axvline(0, color="k", linewidth=AXES_SPINE_LINEWIDTH)
    style_axes_spines_and_ticks(ax)
    return True


def _render_best_l1_l2_overlays_for_model(
    *,
    steel_model: str,
    best: dict[str, dict[str, int]],
    out_dir: Path,
    catalog: pd.DataFrame,
    params_df: pd.DataFrame,
    metrics_w: pd.DataFrame,
    sim_dir: Path,
    set_id_settings_by_id: pd.DataFrame,
    rows_out_all: list[dict],
) -> None:
    specimen_names = [n for n in catalog["Name"].astype(str).tolist() if n in best]
    tail = sorted(set(best.keys()) - set(specimen_names))
    specimen_names = specimen_names + tail

    # Per-specimen limits, then commonize within 10% and round to requested increments:
    # - x (strain fraction) in multiples of 0.20% -> 0.0020
    # - y (P/(fyAsc)) in multiples of 0.2
    rel_tol = 0.10
    x_step = 0.0020
    y_step = 0.2

    def _max_abs_strain_for(name: str) -> float | None:
        sid_l2 = int(best[name]["L2"])
        prow = _params_row_for(params_df, name, sid_l2)
        if prow is None:
            return None
        cat_by = catalog.set_index("Name")
        if name not in cat_by.index:
            return None
        L_y = _ly_from_params_or_catalog(prow, cat_by.loc[name])
        if not np.isfinite(L_y) or L_y <= 0:
            return None
        D2 = _read_simulated_csv(sim_dir, name, sid_l2)
        if D2 is None:
            return None
        D, _F_exp, _F_sim = D2
        s = np.asarray(D, dtype=float) / float(L_y)
        mx = float(np.nanmax(np.abs(s)))
        if not np.isfinite(mx) or mx <= 0:
            return None
        return mx

    def _max_abs_norm_force_for(name: str) -> float | None:
        sid_l2 = int(best[name]["L2"])
        prow = _params_row_for(params_df, name, sid_l2)
        if prow is None:
            return None
        fy = float(prow["fyp"])
        A_c = float(prow["A_sc"])
        fyA = fy * A_c
        if not np.isfinite(fyA) or fyA <= 0:
            return None
        D2 = _read_simulated_csv(sim_dir, name, sid_l2)
        if D2 is None:
            return None
        _D, F_exp, _F_sim = D2
        f = np.asarray(F_exp, dtype=float) / float(fyA)
        mx = float(np.nanmax(np.abs(f)))
        if not np.isfinite(mx) or mx <= 0:
            return None
        return mx

    strain_max: dict[str, float] = {}
    force_max: dict[str, float] = {}
    for n in specimen_names:
        mx = _max_abs_strain_for(n)
        if mx is not None:
            strain_max[n] = mx
        my = _max_abs_norm_force_for(n)
        if my is not None:
            force_max[n] = my

    x_half_by_name = _cluster_common_halfranges(strain_max, rel_tol=rel_tol, step=x_step)
    y_half_by_name = _cluster_common_halfranges(force_max, rel_tol=rel_tol, step=y_step)

    # Metrics table (one row per objective per specimen)
    metric_cols = [
        "final_J_feat_raw",
        "final_J_feat_l1_raw",
        "final_J_E_raw",
        "final_J_E_l1_raw",
        "final_unordered_J_binenv",
        "final_unordered_J_binenv_l1",
    ]
    normalize_steel_model(steel_model)
    param_cols = ["b_p", "b_n", "R0", "cR1", "cR2", "a1", "a2", "a3", "a4"]
    rows_out: list[dict] = []

    def _bp_bn_source_token_for_set(set_id: int) -> str:
        """
        Return provenance token(s) for b_p / b_n seeds from set_id_settings.csv.

        Typical values are 'median', 'q1', 'weighted_mean', or a numeric literal.
        If b_p and b_n tokens differ, return 'b_p=<tok>;b_n=<tok>'.
        """
        sid = int(set_id)
        if sid not in set_id_settings_by_id.index:
            return "unknown"
        row = set_id_settings_by_id.loc[sid]

        def _tok(v) -> str:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "median"
            s = str(v).strip()
            if s == "" or s.lower() == "nan":
                return "median"
            return s

        bp_tok = _tok(row["b_p"]) if "b_p" in row.index else "median"
        bn_tok = _tok(row["b_n"]) if "b_n" in row.index else "median"
        if str(bp_tok).strip().lower() == str(bn_tok).strip().lower():
            return str(bp_tok).strip()
        return f"b_p={bp_tok};b_n={bn_tok}"

    for name in specimen_names:
        for obj in ("L2", "L1"):
            sid = int(best[name][obj])
            mrow = _metrics_row_for(metrics_w, name, sid, obj)
            prow = _params_row_for(params_df, name, sid)
            out_row = {
                "steel_model": steel_model,
                "Name": name,
                "set_id": sid,
                "objective": obj,
                "bp_bn": _bp_bn_source_token_for_set(sid),
            }
            for c in param_cols:
                out_row[c] = (
                    float(prow[c]) if (prow is not None and c in prow.index and pd.notna(prow[c])) else np.nan
                )
            for c in metric_cols:
                out_row[c] = float(mrow[c]) if (mrow is not None and c in mrow.index and pd.notna(mrow[c])) else np.nan
            rows_out.append(out_row)

    rows_out_all.extend(rows_out)

    # Per specimen PNGs
    s_len = OVERLAY_GRID_TICK_LENGTH_SCALE
    per_specimen_rc = {
        "font.size": 18.0,
        "axes.titlesize": 18.0,
        "axes.labelsize": 18.0,
        "xtick.labelsize": 16.0,
        "ytick.labelsize": 16.0,
        "xtick.major.size": 3.5 * s_len,
        "ytick.major.size": 3.5 * s_len,
        "xtick.minor.size": 2.0 * s_len,
        "ytick.minor.size": 2.0 * s_len,
        "legend.fontsize": max(14.0, float(LEGEND_FONT_SIZE_SMALL_PT) * 2.0 - 2.0),
    }
    for name in specimen_names:
        with plt.rc_context(per_specimen_rc):
            fig, ax = plt.subplots(figsize=(6.8, 5.4), layout="constrained")
            fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.05, wspace=0.02, hspace=0.02)
        ok = _plot_one_specimen_norm(
            ax,
            name,
            best[name]["L2"],
            best[name]["L1"],
            sim_dir=sim_dir,
            params_df=params_df,
            catalog=catalog,
            norm_xy_half=(
                (x_half_by_name.get(name), y_half_by_name.get(name))
                if (name in x_half_by_name and name in y_half_by_name)
                else None
            ),
        )
        if ok:
            handles, labels = _best_l1_l2_overlay_legend_handles()
            fig.legend(
                handles,
                labels,
                loc="outside upper center",
                ncol=3,
                fontsize=LEGEND_FONT_SIZE_SMALL_PT,
                frameon=False,
            )
            fig.savefig(
                out_dir / f"{name}_bestL2_bestL1_force_def_norm.png",
                dpi=EXPORT_DPI,
                facecolor="white",
            )
        plt.close(fig)

    # Combined grid
    n = len(specimen_names)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    grid_rc = overlay_grid_montage_rcparams()
    grid_rc.update(
        {
            "font.size": 17.0,
            "axes.titlesize": 17.0,
            "axes.labelsize": 17.0,
            "xtick.labelsize": 15.0,
            "ytick.labelsize": 15.0,
        }
    )
    with plt.rc_context(grid_rc):
        fig, axs = plt.subplots(
            nrow,
            ncol,
            figsize=(
                figsize_for_grid(nrow, ncol)[0] * 1.25,
                figsize_for_grid(nrow, ncol)[1] * 1.25,
            ),
            layout="constrained",
            sharex=False,
            sharey=False,
            squeeze=False,
        )
        fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.05, wspace=0.03, hspace=0.03)
        axs = np.asarray(axs).reshape(nrow, ncol)

        fig.supxlabel(PRIMARY_X_LABEL, fontsize=PLOT_FONT_SIZE_GRID_MONTAGE_PT)
        fig.supylabel(PRIMARY_Y_LABEL, fontsize=PLOT_FONT_SIZE_GRID_MONTAGE_PT)

        for i, name in enumerate(specimen_names):
            r, c = i // ncol, i % ncol
            if r >= nrow:
                break
            ax = axs[r, c]
            xh = x_half_by_name.get(name)
            yh = y_half_by_name.get(name)
            norm_xy = (float(xh), float(yh)) if (xh is not None and yh is not None) else None
            _plot_one_specimen_norm(
                ax,
                name,
                best[name]["L2"],
                best[name]["L1"],
                sim_dir=sim_dir,
                params_df=params_df,
                catalog=catalog,
                norm_xy_half=norm_xy,
                show_primary_axis_labels=False,
            )

        for j in range(n, nrow * ncol):
            r, c = j // ncol, j % ncol
            axs[r, c].set_visible(False)

        handles, labels = _best_l1_l2_overlay_legend_handles()
        fig.legend(
            handles=handles,
            labels=labels,
            loc="outside upper center",
            ncol=3,
            handlelength=3.0,
            frameon=False,
        )
        fig.savefig(
            out_dir / "all_bestL2_bestL1_force_def_norm.png",
            dpi=EXPORT_DPI,
            facecolor="white",
        )
        plt.close(fig)

    # Selected force histories (multipanel)
    sel = [s for s in FORCE_HISTORY_SPECIMENS if s in best]
    if sel:
        # Use a single column (4 rows for the requested four).
        ncol_h = 1
        nrow_h = int(len(sel))

        # Common y-limits across panels (based on all selected traces, normalized by fyA).
        y_max_all = 0.0
        for s in sel:
            prow = _params_row_for(params_df, s, int(best[s]["L2"]))
            if prow is None:
                continue
            fyA = float(prow["fyp"]) * float(prow["A_sc"])
            if not np.isfinite(fyA) or fyA <= 0:
                continue
            d2 = _read_simulated_csv(sim_dir, s, int(best[s]["L2"]))
            d1 = _read_simulated_csv(sim_dir, s, int(best[s]["L1"]))
            if d2 is None or d1 is None:
                continue
            _D2, F_exp, F_l2 = d2
            _D1, _F_exp2, F_l1 = d1
            y_max_all = max(
                float(y_max_all),
                float(np.nanmax(np.abs(np.asarray(F_exp, dtype=float) / fyA))),
                float(np.nanmax(np.abs(np.asarray(F_l1, dtype=float) / fyA))),
                float(np.nanmax(np.abs(np.asarray(F_l2, dtype=float) / fyA))),
            )
        y_half_common = _ceil_to_step(1.05 * float(y_max_all), 0.2) if y_max_all > 0 else None

        _force_hist_variants: tuple[tuple[str, str, str | None], ...] = (
            ("selected_force_histories.png", "index", None),
            (
                "selected_force_histories_cum_deformation_ratio.png",
                "cum_abs_deformation",
                r"Cumulative deformation ratio, $\sum|\Delta\delta|/\hat{\delta}_y$",
            ),
        )
        for png_name_f, x_mode_f, sup_x_f in _force_hist_variants:
            with plt.rc_context(grid_rc):
                fig, axs = plt.subplots(
                    nrow_h,
                    ncol_h,
                    figsize=(12.0, SELECTED_HISTORIES_FIG_HEIGHT_IN),
                    layout="constrained",
                    squeeze=False,
                    sharex=(x_mode_f == "cum_abs_deformation"),
                )
                fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.05, wspace=0.05, hspace=0.05)
                axs = np.asarray(axs).reshape(nrow_h, ncol_h)

                for i, s in enumerate(sel):
                    r, c = i // ncol_h, i % ncol_h
                    ax = axs[r, c]
                    _plot_force_history_panel(
                        ax,
                        specimen=s,
                        sim_dir=sim_dir,
                        params_df=params_df,
                        set_l1=int(best[s]["L1"]),
                        set_l2=int(best[s]["L2"]),
                        y_half=y_half_common,
                        x_mode=x_mode_f,
                    )

                if x_mode_f == "cum_abs_deformation" and len(sel) > 0:
                    xm = _max_cum_abs_deformation_ratio_across_sel(
                        sel, best=best, params_df=params_df, sim_dir=sim_dir
                    )
                    if xm > 0:
                        axs[0, 0].set_xlim(0.0, xm * 1.02)

                for j in range(len(sel), nrow_h * ncol_h):
                    r, c = j // ncol_h, j % ncol_h
                    axs[r, c].set_visible(False)

                fig.supylabel(NORM_FORCE_LABEL, fontsize=PLOT_FONT_SIZE_GRID_MONTAGE_PT)
                if sup_x_f is not None:
                    fig.supxlabel(sup_x_f, fontsize=PLOT_FONT_SIZE_GRID_MONTAGE_PT)
                handles, labels = _best_l1_l2_overlay_legend_handles()
                fig.legend(
                    handles=handles,
                    labels=labels,
                    loc="outside upper center",
                    ncol=3,
                    handlelength=3.0,
                    frameon=False,
                )
                fig.savefig(out_dir / png_name_f, dpi=EXPORT_DPI, facecolor="white")
                plt.close(fig)

        # Tangent stiffness histories (multipanel): total vs inelastic cumulative x.
        _tangent_x_variants: tuple[tuple[str, str, str], ...] = (
            (
                "selected_tangent_stiffness_histories.png",
                "abs",
                r"Cumulative deformation ratio, $\sum|\Delta\delta|/\hat{\delta}_y$",
            ),
            (
                "selected_tangent_stiffness_histories_cum_inelastic.png",
                "inelastic",
                r"Cumulative inelastic deformation ratio, $\sum|\Delta\delta_{\mathrm{inel}}|/\hat{\delta}_y$",
            ),
        )
        for png_name, cum_x, sup_x in _tangent_x_variants:
            with plt.rc_context(grid_rc):
                fig, axs = plt.subplots(
                    nrow_h,
                    ncol_h,
                    figsize=(12.0, SELECTED_HISTORIES_FIG_HEIGHT_IN),
                    layout="constrained",
                    squeeze=False,
                    sharex=True,
                )
                fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.05, wspace=0.05, hspace=0.05)
                axs = np.asarray(axs).reshape(nrow_h, ncol_h)

                for i, s in enumerate(sel):
                    r, c = i // ncol_h, i % ncol_h
                    ax = axs[r, c]
                    _plot_tangent_stiffness_panel(
                        ax,
                        specimen=s,
                        sim_dir=sim_dir,
                        params_df=params_df,
                        set_l1=int(best[s]["L1"]),
                        set_l2=int(best[s]["L2"]),
                        cumulative_x=cum_x,
                    )

                if len(sel) > 0:
                    axs[0, 0].set_xlim(STIFFNESS_HISTORY_XLIM)

                for j in range(len(sel), nrow_h * ncol_h):
                    r, c = j // ncol_h, j % ncol_h
                    axs[r, c].set_visible(False)

                fig.supylabel(
                    r"Stiffness ratio, $K_{\mathrm{tan}}/\hat{K}_0$",
                    fontsize=PLOT_FONT_SIZE_GRID_MONTAGE_PT,
                )
                fig.supxlabel(sup_x, fontsize=PLOT_FONT_SIZE_GRID_MONTAGE_PT)
                handles, labels = _best_l1_l2_overlay_legend_handles()
                fig.legend(
                    handles=handles,
                    labels=labels,
                    loc="outside upper center",
                    ncol=3,
                    handlelength=3.0,
                    frameon=False,
                )
                fig.savefig(out_dir / png_name, dpi=EXPORT_DPI, facecolor="white")
                plt.close(fig)

    sel_tangent_mid = [s for s in TANGENT_STIFFNESS_MIDWINDOW_SPECIMENS if s in best]
    if sel_tangent_mid:
        nrow_mid = len(sel_tangent_mid)
        ncol_mid = 1
        fig_h_mid = SELECTED_HISTORIES_FIG_HEIGHT_IN * float(nrow_mid) / float(len(FORCE_HISTORY_SPECIMENS))
        _mid_tangent_x_variants: tuple[tuple[str, str, str], ...] = (
            (
                "selected_tangent_stiffness_histories_x400_800_PC160_PC750B.png",
                "abs",
                r"Cumulative deformation ratio, $\sum|\Delta\delta|/\hat{\delta}_y$",
            ),
            (
                "selected_tangent_stiffness_histories_cum_inelastic_x400_800_PC160_PC750B.png",
                "inelastic",
                r"Cumulative inelastic deformation ratio, $\sum|\Delta\delta_{\mathrm{inel}}|/\hat{\delta}_y$",
            ),
        )
        for png_mid, cum_x_mid, sup_x_mid in _mid_tangent_x_variants:
            with plt.rc_context(grid_rc):
                fig, axs = plt.subplots(
                    nrow_mid,
                    ncol_mid,
                    figsize=(12.0, fig_h_mid),
                    layout="constrained",
                    squeeze=False,
                    sharex=True,
                )
                fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.05, wspace=0.05, hspace=0.05)
                axs = np.asarray(axs).reshape(nrow_mid, ncol_mid)

                for i, s in enumerate(sel_tangent_mid):
                    r, c = i // ncol_mid, i % ncol_mid
                    ax = axs[r, c]
                    _plot_tangent_stiffness_panel(
                        ax,
                        specimen=s,
                        sim_dir=sim_dir,
                        params_df=params_df,
                        set_l1=int(best[s]["L1"]),
                        set_l2=int(best[s]["L2"]),
                        cumulative_x=cum_x_mid,
                        y_lim=STIFFNESS_HISTORY_YLIM_MIDWINDOW,
                        y_ticks=STIFFNESS_HISTORY_Y_TICKS_MIDWINDOW,
                    )

                axs[0, 0].set_xlim(STIFFNESS_HISTORY_XLIM_MIDWINDOW)

                for j in range(len(sel_tangent_mid), nrow_mid * ncol_mid):
                    r, c = j // ncol_mid, j % ncol_mid
                    axs[r, c].set_visible(False)

                fig.supylabel(
                    r"Stiffness ratio, $K_{\mathrm{tan}}/\hat{K}_0$",
                    fontsize=PLOT_FONT_SIZE_GRID_MONTAGE_PT,
                )
                fig.supxlabel(sup_x_mid, fontsize=PLOT_FONT_SIZE_GRID_MONTAGE_PT)
                handles, labels = _best_l1_l2_overlay_legend_handles()
                fig.legend(
                    handles=handles,
                    labels=labels,
                    loc="outside upper center",
                    ncol=3,
                    handlelength=3.0,
                    frameon=False,
                )
                fig.savefig(out_dir / png_mid, dpi=EXPORT_DPI, facecolor="white")
                plt.close(fig)



def main() -> None:
    p = argparse.ArgumentParser(
        description="Plot experimental (grey) + best L2 + best L1 overlays for individual-optimize training specimens (excluding CB225).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(PLOTS_INDIVIDUAL_OPTIMIZE / "overlays_best_l1_l2"),
        help=(
            "Output root directory (default: results/plots/calibration/individual_optimize/overlays_best_l1_l2). "
            "PNGs are written under <output-dir>/<steel_model>/."
        ),
    )
    p.add_argument(
        "--metrics-csv",
        type=str,
        default=str(METRICS_CSV),
        help="Path to optimized_brb_parameters_metrics.csv (default: repo results path).",
    )
    p.add_argument(
        "--params-csv",
        type=str,
        default=str(OPTIMIZED_BRB_PARAMETERS_PATH),
        help="Path to optimized_brb_parameters.csv (default: repo results path).",
    )
    p.add_argument(
        "--simulated-force-dir",
        type=str,
        default=str(INDIVIDUAL_SIMULATED_FORCE_DIR),
        help="Directory containing {Name}_set{k}_simulated.csv (default: repo results path).",
    )
    p.add_argument(
        "--specimen",
        type=str,
        default=None,
        help="If set, only plot this specimen name (still requires it to be in metrics and not CB225).",
    )
    args = p.parse_args()
    root_out = Path(args.output_dir).expanduser().resolve()
    root_out.mkdir(parents=True, exist_ok=True)

    metrics_csv = Path(args.metrics_csv).expanduser().resolve()
    params_csv = Path(args.params_csv).expanduser().resolve()
    sim_dir = Path(args.simulated_force_dir).expanduser().resolve()

    catalog = read_catalog(BRB_SPECIMENS_CSV)
    params_df = pd.read_csv(params_csv)
    metrics_df = pd.read_csv(metrics_csv)
    set_id_settings_df = read_set_id_settings_table()
    if "set_id" not in set_id_settings_df.columns:
        raise RuntimeError(f"set_id_settings CSV missing 'set_id' column: {SET_ID_SETTINGS_CSV}")
    set_id_settings_df["set_id"] = pd.to_numeric(
        set_id_settings_df["set_id"].astype(str).str.strip(),
        errors="coerce",
    )
    set_id_settings_df = set_id_settings_df[np.isfinite(set_id_settings_df["set_id"])].copy()
    set_id_settings_df["set_id"] = set_id_settings_df["set_id"].astype(int)
    set_id_settings_by_id = set_id_settings_df.set_index("set_id", drop=True)

    sid_to_sm = _set_id_to_steel_model(params_df)
    metrics_w = metrics_df.copy()
    metrics_w["set_id_num"] = pd.to_numeric(metrics_w["set_id"], errors="coerce")
    metrics_w["steel_model"] = metrics_w["set_id_num"].apply(
        lambda x: sid_to_sm.get(int(x), normalize_steel_model(None)) if pd.notna(x) else normalize_steel_model(None)
    )

    models_here = _models_in_order(set(sid_to_sm.values()) if sid_to_sm else {"steelmpf"})
    rows_out_all: list[dict] = []

    for steel_model in models_here:
        best = _pick_best_set_ids(metrics_w, steel_model=steel_model)
        if args.specimen is not None:
            s = str(args.specimen).strip()
            best = {k: v for k, v in best.items() if k == s}

        if not best:
            print(
                f"No specimens with both best L2 and best L1 for steel_model={steel_model!r}; skipping."
            )
            continue

        out_dir = root_out / steel_model
        out_dir.mkdir(parents=True, exist_ok=True)

        _render_best_l1_l2_overlays_for_model(
            steel_model=steel_model,
            best=best,
            out_dir=out_dir,
            catalog=catalog,
            params_df=params_df,
            metrics_w=metrics_w,
            sim_dir=sim_dir,
            set_id_settings_by_id=set_id_settings_by_id,
            rows_out_all=rows_out_all,
        )

    if not rows_out_all:
        print("No overlays written (no steel_model had both best L1 and best L2 for any specimen).")
        return
    metrics_table_path = root_out / "bestL2_bestL1_metrics_table.csv"
    pd.DataFrame(rows_out_all).to_csv(metrics_table_path, index=False)
    print(f"Wrote overlays into {root_out} (per-model subfolders + combined metrics table)")



if __name__ == "__main__":
    main()

