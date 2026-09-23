"""Combined force–deformation overlays for all set_ids (single-specimen calibration)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
_SCRIPTS = _PROJECT_ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS / "postprocess"))

from calibrate.overlay_axis_specs import (  # noqa: E402
    ForceDefOverlayAxisSpecs,
    apply_symmetric_axis_specs,
    compute_force_def_overlay_axis_specs,
)
from calibrate.plot_params_vs_filtered import (  # noqa: E402
    LINEWIDTH_EXPERIMENTAL,
    LINEWIDTH_SIMULATED,
    _path_ordered_sim_kwargs,
)
from model.corotruss import run_simulation  # noqa: E402
from postprocess.plot_dimensions import (  # noqa: E402
    AXES_SPINE_LINEWIDTH,
    COLOR_EXPERIMENTAL,
    COLOR_NUMERICAL_COHORT,
    GRID_AX_H_IN,
    GRID_AX_W_IN,
    LEGEND_FONT_SIZE_GRID_MONTAGE_PT,
    PLOT_FONT_SIZE_GRID_MONTAGE_PT,
    SAVE_DPI,
    overlay_grid_montage_rcparams,
    style_axes_spines_and_ticks,
)
from postprocess.plot_specimens import (  # noqa: E402
    NORM_FORCE_LABEL,
    NORM_STRAIN_LABEL,
    PHYS_FORCE_KIP_LABEL,
)

_SET_PANEL_SCALE = 1.35
_COLS_PER_ROW_GROUP = 2


def _all_sets_montage_shape(n_sets: int) -> tuple[int, int]:
    """Return ``(n_rows, n_cols)`` with cols:rows ≈ 2:1."""
    if n_sets < 1:
        return 0, 0
    n_row_groups = max(1, int(np.ceil(np.sqrt(n_sets / _COLS_PER_ROW_GROUP))))
    n_cols = _COLS_PER_ROW_GROUP * n_row_groups
    n_rows = int(np.ceil(n_sets / n_cols))
    return n_rows, n_cols


def _figsize_all_sets_grid(n_rows: int, n_cols: int) -> tuple[float, float]:
    w = GRID_AX_W_IN * _SET_PANEL_SCALE * max(1, n_cols)
    h = GRID_AX_H_IN * _SET_PANEL_SCALE * max(1, n_rows)
    return (w, h)


def _overlay_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=COLOR_EXPERIMENTAL,
            linewidth=LINEWIDTH_EXPERIMENTAL,
            label="Experimental",
        ),
        Line2D(
            [0],
            [0],
            color=COLOR_NUMERICAL_COHORT,
            linewidth=LINEWIDTH_SIMULATED,
            linestyle="--",
            label="Numerical",
        ),
    ]


def _plot_physical_panel(
    ax: plt.Axes,
    *,
    set_id: int | str,
    D: np.ndarray,
    F_exp: np.ndarray,
    F_sim: np.ndarray,
) -> None:
    ax.plot(
        D,
        F_exp,
        color=COLOR_EXPERIMENTAL,
        linewidth=LINEWIDTH_EXPERIMENTAL,
        linestyle="-",
    )
    ax.plot(
        D,
        F_sim,
        color=COLOR_NUMERICAL_COHORT,
        linewidth=LINEWIDTH_SIMULATED,
        linestyle="--",
    )
    ax.set_title(f"set {set_id}")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="k", linewidth=AXES_SPINE_LINEWIDTH)
    ax.axvline(0, color="k", linewidth=AXES_SPINE_LINEWIDTH)
    style_axes_spines_and_ticks(ax)


def _plot_normalized_panel(
    ax: plt.Axes,
    *,
    set_id: int | str,
    D: np.ndarray,
    F_exp: np.ndarray,
    F_sim: np.ndarray,
    fy: float,
    A_sc: float,
    L_y: float,
) -> None:
    d_norm = D / L_y if L_y > 0 else D
    fyA = fy * A_sc
    if fyA <= 0:
        ax.set_visible(False)
        return

    ax.plot(
        d_norm,
        F_exp / fyA,
        color=COLOR_EXPERIMENTAL,
        linewidth=LINEWIDTH_EXPERIMENTAL,
        linestyle="-",
    )
    ax.plot(
        d_norm,
        F_sim / fyA,
        color=COLOR_NUMERICAL_COHORT,
        linewidth=LINEWIDTH_SIMULATED,
        linestyle="--",
    )
    ax.set_title(f"set {set_id}")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="k", linewidth=AXES_SPINE_LINEWIDTH)
    ax.axvline(0, color="k", linewidth=AXES_SPINE_LINEWIDTH)
    style_axes_spines_and_ticks(ax)


def _apply_montage_tick_labels(axs: np.ndarray, n_panels: int, n_cols: int) -> None:
    """Keep numeric tick labels only on the montage perimeter (works with shared axes)."""
    for j in range(n_panels):
        row, col = j // n_cols, j % n_cols
        axs[row, col].label_outer()


def _save_all_sets_montage(
    *,
    panels: list[tuple[int | str, np.ndarray, float, float, float]],
    D: np.ndarray,
    F_exp: np.ndarray,
    out_path: Path,
    mode: str,
    overlay_axes: ForceDefOverlayAxisSpecs,
) -> None:
    n = len(panels)
    n_rows, n_cols = _all_sets_montage_shape(n)
    with plt.rc_context(overlay_grid_montage_rcparams()):
        fig, axs = plt.subplots(
            n_rows,
            n_cols,
            figsize=_figsize_all_sets_grid(n_rows, n_cols),
            layout="constrained",
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        axs = np.asarray(axs)

        if mode == "physical":
            fig.supxlabel("Deformation [in]", fontsize=PLOT_FONT_SIZE_GRID_MONTAGE_PT)
            fig.supylabel(PHYS_FORCE_KIP_LABEL, fontsize=PLOT_FONT_SIZE_GRID_MONTAGE_PT)
        else:
            fig.supxlabel(NORM_STRAIN_LABEL, fontsize=PLOT_FONT_SIZE_GRID_MONTAGE_PT)
            fig.supylabel(NORM_FORCE_LABEL, fontsize=PLOT_FONT_SIZE_GRID_MONTAGE_PT)

        for j, (set_id, F_sim, fy, A_sc, L_y) in enumerate(panels):
            row, col = j // n_cols, j % n_cols
            ax = axs[row, col]
            if mode == "physical":
                _plot_physical_panel(
                    ax,
                    set_id=set_id,
                    D=D,
                    F_exp=F_exp,
                    F_sim=F_sim,
                )
            else:
                _plot_normalized_panel(
                    ax,
                    set_id=set_id,
                    D=D,
                    F_exp=F_exp,
                    F_sim=F_sim,
                    fy=fy,
                    A_sc=A_sc,
                    L_y=L_y,
                )

        for j in range(n, n_rows * n_cols):
            row, col = j // n_cols, j % n_cols
            axs[row, col].set_visible(False)

        if mode == "physical":
            apply_symmetric_axis_specs(
                axs[0, 0],
                x=overlay_axes.physical_x,
                y=overlay_axes.physical_y,
            )
        else:
            apply_symmetric_axis_specs(
                axs[0, 0],
                x=overlay_axes.normalized_x,
                y=overlay_axes.normalized_y,
                normalized_strain_x=True,
            )

        _apply_montage_tick_labels(axs, n, n_cols)

        fig.legend(
            handles=_overlay_legend_handles(),
            loc="outside upper center",
            ncol=2,
            fontsize=LEGEND_FONT_SIZE_GRID_MONTAGE_PT,
            frameon=False,
        )
        fig.savefig(out_path, dpi=SAVE_DPI, facecolor="white")
        plt.close(fig)


def plot_all_sets_force_def_grid(
    specimen_id: str,
    params_df: pd.DataFrame,
    catalog_row: pd.Series,
    displacement: np.ndarray,
    F_exp: np.ndarray,
    out_dir: Path,
) -> list[Path]:
    """
    Two montage figures (physical and normalized), one panel per ``set_id``.
    Shared axis titles use ``supxlabel`` / ``supylabel`` on each figure.
    """
    if params_df.empty:
        return []

    D = np.asarray(displacement, dtype=float)
    F_exp = np.asarray(F_exp, dtype=float)
    panels: list[tuple[int | str, np.ndarray, float, float, float]] = []

    for _, prow in params_df.iterrows():
        set_id = prow.get("set_id", 1)
        sk = _path_ordered_sim_kwargs(
            prow,
            catalog_row,
            specimen_id=specimen_id,
            set_id=set_id,
            override_b_p=None,
            override_b_n=None,
        )
        if sk is None:
            continue
        sm, sim_kw = sk
        try:
            F_sim = np.asarray(run_simulation(D, steel_model=sm, **sim_kw), dtype=float)
        except Exception:
            continue
        if F_sim.shape != F_exp.shape:
            continue
        panels.append(
            (
                set_id,
                F_sim,
                float(sim_kw["fyp"]),
                float(sim_kw["A_sc"]),
                float(sim_kw["L_y"]),
            )
        )

    if not panels:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    path_phys = out_dir / f"{specimen_id}_all_sets_force_def.png"
    path_norm = out_dir / f"{specimen_id}_all_sets_force_def_norm.png"

    overlay_axes = compute_force_def_overlay_axis_specs(
        D,
        F_exp,
        [(F_sim, fy, A_sc, L_y) for _, F_sim, fy, A_sc, L_y in panels],
    )

    _save_all_sets_montage(
        panels=panels,
        D=D,
        F_exp=F_exp,
        out_path=path_phys,
        mode="physical",
        overlay_axes=overlay_axes,
    )
    _save_all_sets_montage(
        panels=panels,
        D=D,
        F_exp=F_exp,
        out_path=path_norm,
        mode="normalized",
        overlay_axes=overlay_axes,
    )
    return [path_phys, path_norm]
