"""Shared figure sizes (inches), DPI, and typography for BRB-Calibration plots."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import matplotlib.pyplot as plt

# Point size: labels, ticks, titles, legend (single consistent base).
PLOT_FONT_SIZE_PT: float = 9.0

# Larger type for multi-panel montages (specimen grids, combined overlays).
PLOT_FONT_SIZE_GRID_MONTAGE_PT: float = 18.0

# ``results/plots/apparent_b/b_vs_geometry/*``: same grid style, smaller type for denser panels.
B_VS_GEOMETRY_FONT_SCALE: float = 0.7
PLOT_FONT_SIZE_B_VS_GEOMETRY_PT: float = PLOT_FONT_SIZE_GRID_MONTAGE_PT * B_VS_GEOMETRY_FONT_SCALE

# Compact legends for small single-axis figures (e.g. 2.5×2.25 in); keeps in-axes area clear.
LEGEND_FONT_SIZE_SMALL_PT: float = 6.5

# Legend text on grid montages (matches ``legend.fontsize`` pattern vs base plot size).
LEGEND_FONT_SIZE_GRID_MONTAGE_PT: float = PLOT_FONT_SIZE_GRID_MONTAGE_PT - 0.75

# Legend spacing (matplotlib defaults: labelspacing 0.5, columnspacing 2.0, handletextpad 0.8, borderpad 0.5).
# Rows / handle / border: tight vs defaults; column gap is 2× the prior compact value (still below default 2.0).
LEGEND_LABELSPACING: float = 0.125
LEGEND_COLUMNSPACING: float = 1.0
LEGEND_HANDLETEXTPAD: float = 0.4
LEGEND_BORDERPAD: float = 0.25

# Single axes (one specimen loop / one diagnostic panel).
SINGLE_FIGSIZE_IN: tuple[float, float] = (2.5, 2.25)

# Each grid cell in montages / multi-panel figures.
GRID_AX_W_IN: float = 2.5
GRID_AX_H_IN: float = 2.25

# Two rows × one column (time histories, digitized raw/filtered vs resampled drive).
TWO_STACKED_FIG_W_IN: float = 6.0

SAVE_DPI: int = 300

# Experimental / test hysteresis and clouds (experimental overlays solid ``'-'``; numerical ``'--'``).
COLOR_EXPERIMENTAL: str = "#B0B0B0"

# Stroke scale for F–δ (and similar) hysteresis traces vs prior nominal linewidths (specimen loops, b-slopes, etc.).
HYSTERESIS_LINEWIDTH_SCALE: float = 0.6
LINEWIDTH_HYSTERESIS_EXPERIMENTAL: float = 0.9 * HYSTERESIS_LINEWIDTH_SCALE
LINEWIDTH_HYSTERESIS_SIMULATED: float = LINEWIDTH_HYSTERESIS_EXPERIMENTAL

# Combined exp/sim overlays (``plot_params_vs_filtered``, ``plot_compare_calibration_overlays``, bayesian): 80 % of nominal 0.9.
HYSTERESIS_LINEWIDTH_SCALE_OVERLAYS: float = 0.8
LINEWIDTH_HYSTERESIS_OVERLAY_EXPERIMENTAL: float = 0.9 * HYSTERESIS_LINEWIDTH_SCALE_OVERLAYS
LINEWIDTH_HYSTERESIS_OVERLAY_SIMULATED: float = LINEWIDTH_HYSTERESIS_OVERLAY_EXPERIMENTAL

# Numerical / simulated in calibration overlays (pair with ``COLOR_EXPERIMENTAL``):
# ``COLOR_NUMERICAL_COHORT`` — train; ``COLOR_NUMERICAL_COHORT_AUX`` — validation
# (zero-weight path rows, scatter-cloud cohort panels).
COLOR_NUMERICAL_COHORT: str = "#001F3F"
COLOR_NUMERICAL_COHORT_AUX: str = "#FF0E00"

# Full data-area rectangle (four spines); separate from ``Axes.patch`` fill outline.
AXES_SPINE_COLOR: str = "black"
AXES_SPINE_LINEWIDTH: float = 0.4

# One-axis figures at ``SINGLE_FIGSIZE_IN``: thinner frame/ticks and smaller type than montages.
SINGLE_AX_PLOT_FRAME_TICK_FONT_SCALE: float = 0.6
AXES_SPINE_LINEWIDTH_SINGLE_AX: float = AXES_SPINE_LINEWIDTH * SINGLE_AX_PLOT_FRAME_TICK_FONT_SCALE
PLOT_FONT_SIZE_SINGLE_AX_PT: float = PLOT_FONT_SIZE_PT * SINGLE_AX_PLOT_FRAME_TICK_FONT_SCALE
LEGEND_FONT_SIZE_SINGLE_AX_PT: float = LEGEND_FONT_SIZE_SMALL_PT * SINGLE_AX_PLOT_FRAME_TICK_FONT_SCALE


def style_major_tick_lines(
    ax: plt.Axes,
    *,
    linewidth: float | None = None,
    color: str | None = None,
) -> None:
    """Set major tick **line** color and width to match spines (tick labels unchanged)."""
    lw = AXES_SPINE_LINEWIDTH if linewidth is None else linewidth
    col = AXES_SPINE_COLOR if color is None else color
    for axis in (ax.xaxis, ax.yaxis):
        for line in axis.get_ticklines(minor=False):
            line.set_color(col)
            line.set_linewidth(lw)


def single_axis_plot_rcparams() -> dict[str, float]:
    """rcParams for one small panel (``SINGLE_FIGSIZE_IN``): scaled font and major-tick stroke vs ``configure_matplotlib_style``."""
    fs = PLOT_FONT_SIZE_SINGLE_AX_PT
    tw = AXES_SPINE_LINEWIDTH_SINGLE_AX
    leg = max(fs - 0.75, 3.0)
    return {
        "font.size": fs,
        "axes.titlesize": fs,
        "axes.labelsize": fs,
        "xtick.labelsize": fs,
        "ytick.labelsize": fs,
        "legend.fontsize": leg,
        "figure.titlesize": fs,
        "xtick.major.width": tw,
        "ytick.major.width": tw,
    }


def style_single_axis_spines(ax: plt.Axes, *, color: str | None = None) -> None:
    """Spine + major tick lines for ``SINGLE_FIGSIZE_IN`` figures (matches ``single_axis_plot_rcparams`` tick width)."""
    style_axes_spines_and_ticks(ax, linewidth=AXES_SPINE_LINEWIDTH_SINGLE_AX, color=color)


@contextmanager
def single_axis_style_context() -> Iterator[None]:
    """Temporarily apply ``single_axis_plot_rcparams`` (font + major tick width) around one small-panel figure."""
    rc = single_axis_plot_rcparams()
    prev = {k: plt.rcParams[k] for k in rc}
    plt.rcParams.update(rc)
    try:
        yield
    finally:
        plt.rcParams.update(prev)


def style_axes_spines_and_ticks(
    ax: plt.Axes,
    *,
    linewidth: float | None = None,
    color: str | None = None,
) -> None:
    """Spine and tick lines on the data plane: hide axes-patch border, four black spines, inward ticks (labels bottom+left)."""
    lw = AXES_SPINE_LINEWIDTH if linewidth is None else linewidth
    col = AXES_SPINE_COLOR if color is None else color
    ax.patch.set_linewidth(0.0)
    ax.patch.set_edgecolor("none")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(col)
        spine.set_linewidth(lw)
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        bottom=True,
        left=True,
        right=True,
        labeltop=False,
        labelright=False,
    )
    style_major_tick_lines(ax, linewidth=lw, color=col)


def configure_matplotlib_style() -> None:
    """Load ``config/figure.mplstyle`` then apply repo font/legend defaults."""
    style_path = Path(__file__).resolve().parents[2] / "config" / "figure.mplstyle"
    if style_path.is_file():
        plt.style.use(str(style_path))
    plt.rcParams.update(
        {
            "font.size": PLOT_FONT_SIZE_PT,
            "axes.titlesize": PLOT_FONT_SIZE_PT,
            "axes.labelsize": PLOT_FONT_SIZE_PT,
            "xtick.labelsize": PLOT_FONT_SIZE_PT,
            "ytick.labelsize": PLOT_FONT_SIZE_PT,
            "legend.fontsize": PLOT_FONT_SIZE_PT - 0.75,
            "figure.titlesize": PLOT_FONT_SIZE_PT,
            "legend.labelspacing": LEGEND_LABELSPACING,
            "legend.columnspacing": LEGEND_COLUMNSPACING,
            "legend.handletextpad": LEGEND_HANDLETEXTPAD,
            "legend.borderpad": LEGEND_BORDERPAD,
        }
    )


def grid_montage_rcparams() -> dict[str, float]:
    """rcParams for multi-panel figures: ~2× base font so subplots stay readable."""
    fs = PLOT_FONT_SIZE_GRID_MONTAGE_PT
    return {
        "font.size": fs,
        "axes.titlesize": fs,
        "axes.labelsize": fs,
        "xtick.labelsize": fs,
        "ytick.labelsize": fs,
        "legend.fontsize": LEGEND_FONT_SIZE_GRID_MONTAGE_PT,
        "figure.titlesize": fs,
    }


# Major/minor tick **length** only (rc ``*.major.size`` / ``*.minor.size``) on combined exp/sim F–δ overlay grids.
OVERLAY_GRID_TICK_LENGTH_SCALE: float = SINGLE_AX_PLOT_FRAME_TICK_FONT_SCALE


def overlay_grid_montage_rcparams() -> dict[str, float]:
    """Like ``grid_montage_rcparams`` with shorter major/minor tick marks (length only; linewidth matches spines)."""
    d = dict(grid_montage_rcparams())
    s = OVERLAY_GRID_TICK_LENGTH_SCALE
    # Matplotlib defaults (pt): major ~3.5, minor ~2.0.
    major_len = 3.5 * s
    minor_len = 2.0 * s
    d.update(
        {
            "xtick.major.size": major_len,
            "ytick.major.size": major_len,
            "xtick.minor.size": minor_len,
            "ytick.minor.size": minor_len,
        }
    )
    return d


def b_vs_geometry_rcparams() -> dict[str, float]:
    """rcParams for apparent-b vs geometry montages (``B_VS_GEOMETRY_FONT_SCALE`` × grid montage type)."""
    s = B_VS_GEOMETRY_FONT_SCALE
    fs = PLOT_FONT_SIZE_GRID_MONTAGE_PT * s
    return {
        "font.size": fs,
        "axes.titlesize": fs,
        "axes.labelsize": fs,
        "xtick.labelsize": fs,
        "ytick.labelsize": fs,
        "legend.fontsize": LEGEND_FONT_SIZE_GRID_MONTAGE_PT * s,
        "figure.titlesize": fs,
    }


def figsize_for_grid(nrow: int, ncol: int) -> tuple[float, float]:
    """Total figure width/height for an ``nrow`` × ``ncol`` axes grid (use with ``layout='constrained'``)."""
    nr = max(1, int(nrow))
    nc = max(1, int(ncol))
    return (GRID_AX_W_IN * nc, GRID_AX_H_IN * nr)


def figsize_two_stacked_axes() -> tuple[float, float]:
    """Two subplot rows, one column (e.g. force + deformation vs index)."""
    return (TWO_STACKED_FIG_W_IN, GRID_AX_H_IN * 2)
