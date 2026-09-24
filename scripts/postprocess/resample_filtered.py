"""
Resample filtered ``force_deformation.csv`` (path-ordered) or filtered deformation drive (digitized)
into ``data/resampled/{Name}/`` using |Deltau| spacing from Dy/4.

Reads landmarks from ``data/cycle_points_original/{Name}.json`` (filtered grid). Writes
``data/cycle_points_resampled/{Name}.json`` with indices for the resampled length.

**Digitized scatter-cloud** rows: same **segment-based |Du| resampling** as path-ordered (cycle
points on the filtered drive from ``filter_force.py``; dummy ``Force[kip]=0`` for
``resample_by_segments``). Filtered F-u **cloud** is copied unchanged to ``resampled/``.
``data/cycle_points_resampled/{Name}.json`` is written for the resampled drive grid.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
_SCRIPTS = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(SCRIPT_DIR))

from calibrate.calibration_paths import INITIAL_BRB_PARAMETERS_PATH  # noqa: E402
from calibrate.deformation_history_drive import PREPEND_LEAD_FRAC  # noqa: E402
from calibrate.resample_experiment import (  # noqa: E402
    DEF_COL,
    FORCE_COL,
    d_sampling_from_brace_params,
    remap_cycle_points,
    resample_by_segments,
)
from cycle_points import (  # noqa: E402
    find_cycle_points,
    load_cycle_points_original,
    stored_cycle_points_grid_n,
)
from specimen_catalog import (  # noqa: E402
    DEFORMATION_HISTORY_CSV,
    FORCE_DEFORMATION_CSV,
    get_specimen_record,
    read_catalog,
    resolve_filtered_force_deformation_csv,
    uses_unordered_inputs,
    write_deformation_history_step_csv,
)

import matplotlib.pyplot as plt  # noqa: E402
from plot_dimensions import (  # noqa: E402
    AXES_SPINE_LINEWIDTH_SINGLE_AX,
    COLOR_EXPERIMENTAL,
    HYSTERESIS_LINEWIDTH_SCALE,
    LEGEND_FONT_SIZE_SINGLE_AX_PT,
    SAVE_DPI,
    SINGLE_FIGSIZE_IN,
    configure_matplotlib_style,
    single_axis_style_context,
    style_single_axis_spines,
)
from plot_specimens import (  # noqa: E402
    NORM_FORCE_LABEL,
    NORM_STRAIN_LABEL,
    apply_normalized_fu_axes,
    scatter_characteristic_points_physical,
    set_symmetric_axes,
)

FILTERED_DIR = PROJECT_ROOT / "data" / "filtered"
RESAMPLED_DIR = PROJECT_ROOT / "data" / "resampled"
CYCLE_POINTS_RESAMPLED_DIR = PROJECT_ROOT / "data" / "cycle_points_resampled"
CATALOG_PATH = PROJECT_ROOT / "config" / "calibration" / "BRB-Specimens.csv"
PLOTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "plots"
    / "postprocess"
    / "force_deformation"
    / "resampled_and_filtered"
)
DEFAULT_E_KSI = 29000.0


def _lead_replace_first_zero_deformation(u: np.ndarray, d_samp: float) -> np.ndarray:
    """If the first sample is ~0 in, set it to ``PREPEND_LEAD_FRAC * d_samp`` (SteelMPF workaround)."""
    u = np.asarray(u, dtype=float).copy()
    if u.size == 0:
        return u
    if np.isclose(u[0], 0.0, rtol=0.0, atol=1e-12):
        u[0] = float(PREPEND_LEAD_FRAC) * float(d_samp)
    return u


plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
configure_matplotlib_style()


def _e_ksi_by_name(initial_df: pd.DataFrame | None) -> dict[str, float]:
    """Map specimen Name to E [ksi] from catalog for resampling."""
    if initial_df is None or initial_df.empty or "Name" not in initial_df.columns:
        return {}
    if "E" not in initial_df.columns:
        return {}
    out: dict[str, float] = {}
    for name, grp in initial_df.groupby(initial_df["Name"].astype(str)):
        row = grp.iloc[0]
        out[str(name)] = float(row["E"])
    return out


def _plot_overlay(
    specimen_id: str,
    df_f: pd.DataFrame,
    df_r: pd.DataFrame,
    resampled_points: list[dict],
    out_path: Path,
    *,
    f_yc: float,
    A_c: float,
    L_y: float,
) -> None:
    """Overlay raw vs filtered or similar for resample QA."""
    fyA = float(f_yc) * float(A_c)
    L_y = float(L_y)
    if fyA <= 0 or L_y <= 0 or not np.isfinite(fyA) or not np.isfinite(L_y):
        fyA, L_y = 1.0, 1.0
    u_f = df_f[DEF_COL].to_numpy(dtype=float) / L_y
    F_f = df_f[FORCE_COL].to_numpy(dtype=float) / fyA
    u_r = df_r[DEF_COL].to_numpy(dtype=float) / L_y
    F_r = df_r[FORCE_COL].to_numpy(dtype=float) / fyA
    with single_axis_style_context():
        fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE_IN, layout="constrained")
        ax.plot(
            u_f,
            F_f,
            color="tab:orange",
            linewidth=0.9 * HYSTERESIS_LINEWIDTH_SCALE,
            label="Filtered",
            alpha=0.85,
        )
        ax.plot(
            u_r,
            F_r,
            color=COLOR_EXPERIMENTAL,
            linewidth=0.7 * HYSTERESIS_LINEWIDTH_SCALE,
            label="Resampled",
            alpha=0.9,
        )
        df_r_n = df_r.assign(
            _u_norm=df_r[DEF_COL].to_numpy(dtype=float) / L_y,
            _F_norm=df_r[FORCE_COL].to_numpy(dtype=float) / fyA,
        )
        scatter_characteristic_points_physical(
            ax,
            df_r_n,
            resampled_points,
            def_col="_u_norm",
            force_col="_F_norm",
        )
        x_all = np.concatenate([u_f, u_r])
        y_all = np.concatenate([F_f, F_r])
        set_symmetric_axes(ax, x_all, y_all)
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
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=SAVE_DPI)
        plt.close(fig)


def _filtered_specimen_names(catalog_by_name: pd.DataFrame) -> list[str]:
    """Names that have filtered force_deformation outputs on disk."""
    names: set[str] = set()
    if FILTERED_DIR.is_dir():
        for p in FILTERED_DIR.iterdir():
            if (
                p.is_dir()
                and (p / FORCE_DEFORMATION_CSV).is_file()
                and p.name in catalog_by_name.index
            ):
                names.add(p.name)
        for p in FILTERED_DIR.glob("*.csv"):
            if p.stem in catalog_by_name.index:
                names.add(p.stem)
    return sorted(names)


def process_specimen(
    name: str,
    catalog_by_name: pd.DataFrame,
    e_by_name: dict[str, float],
    catalog_df: pd.DataFrame,
) -> None:
    """Resample one path-ordered specimen; write CSVs and optional overlay."""
    rec = get_specimen_record(name, catalog_df)
    if not rec.path_ordered:
        print(f"  Skip {name}: path_ordered=false (use digitized cloud resample branch)")
        return
    fpath = resolve_filtered_force_deformation_csv(name, PROJECT_ROOT)
    if fpath is None or not fpath.is_file():
        return
    df = pd.read_csv(fpath)
    if FORCE_COL not in df.columns or DEF_COL not in df.columns:
        print(f"Skip {name}: missing columns")
        return

    n = len(df)
    json_n = stored_cycle_points_grid_n(name)
    loaded = load_cycle_points_original(name)
    json_points_ok = (
        loaded is not None
        and json_n == n
        and all(0 <= int(p["idx"]) < n for p in loaded[0])
    )
    if json_points_ok:
        points, _seg = loaded
        boundary_indices = sorted({int(p["idx"]) for p in points} | {0, n})
    else:
        if loaded is not None and json_n is not None and json_n != n:
            print(
                f"  {name}: cycle_points JSON n={json_n} != filtered n={n}; "
                "re-detecting characteristic points on filtered CSV"
            )
        elif loaded is not None:
            print(
                f"  {name}: cycle_points indices out of range for filtered n={n}; "
                "re-detecting characteristic points on filtered CSV"
            )
        points, _ = find_cycle_points(df)
        boundary_indices = sorted({int(p["idx"]) for p in points} | {0, n})

    row = catalog_by_name.loc[name]
    L_T = float(row["L_T"])
    L_y = float(row["L_y"])
    A_sc = float(row["A_sc"])
    A_t = float(row["A_t"])
    fyp = float(row["fyp"])
    E_ksi = float(e_by_name.get(name, DEFAULT_E_KSI))

    u_col = df[DEF_COL].to_numpy(dtype=float)
    d_samp = d_sampling_from_brace_params(
        fyp_ksi=fyp,
        L_T_in=L_T,
        L_y_in=L_y,
        A_sc_in2=A_sc,
        A_t_in2=A_t,
        E_ksi=E_ksi,
        u_fallback=u_col,
    )
    # Guard against zero/invalid spacing (would divide by zero in resample_segment_along_u_path).
    d_samp = float(max(d_samp, 1e-15))

    out_df, remap = resample_by_segments(df, boundary_indices, d_samp)
    u_lead = _lead_replace_first_zero_deformation(out_df[DEF_COL].to_numpy(), d_samp)
    out_df = out_df.copy()
    out_df[DEF_COL] = u_lead
    new_points = remap_cycle_points(points, remap, out_df)

    rdir = RESAMPLED_DIR / name
    rdir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(rdir / FORCE_DEFORMATION_CSV, index=False)
    write_deformation_history_step_csv(rdir / DEFORMATION_HISTORY_CSV, out_df[DEF_COL].values)

    CYCLE_POINTS_RESAMPLED_DIR.mkdir(parents=True, exist_ok=True)
    with open(CYCLE_POINTS_RESAMPLED_DIR / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump({"points": new_points, "n": len(out_df)}, f, indent=2)

    try:
        _plot_overlay(
            name,
            df,
            out_df,
            new_points,
            PLOTS_DIR / f"{name}.png",
            f_yc=fyp,
            A_c=A_sc,
            L_y=L_y,
        )
    except Exception as exc:
        print(
            f"  {name}: overlay plot skipped ({type(exc).__name__}: {exc}); "
            "resampled CSV and cycle_points_resampled JSON were still written."
        )
    print(
        f"  {name}: resampled n={len(out_df)} (filtered n={n}), D_sampling={d_samp:.6g} in"
    )


def process_specimen_digitized_unordered(
    name: str,
    catalog_by_name: pd.DataFrame,
    e_by_name: dict[str, float],
    catalog_df: pd.DataFrame,
) -> None:
    """Resample filtered deformation drive with same |Du| segments as path-ordered; cloud copy-only."""
    fdir = FILTERED_DIR / name
    dh_f = fdir / DEFORMATION_HISTORY_CSV
    fd_f = fdir / FORCE_DEFORMATION_CSV
    if not dh_f.is_file() or not fd_f.is_file():
        print(f"  Skip {name}: missing filtered/{FORCE_DEFORMATION_CSV} or {DEFORMATION_HISTORY_CSV}")
        return
    df_dh = pd.read_csv(dh_f)
    if DEF_COL not in df_dh.columns:
        print(f"  Skip {name}: filtered deformation_history missing {DEF_COL}")
        return
    df = pd.DataFrame(
        {
            DEF_COL: df_dh[DEF_COL].to_numpy(dtype=float),
            FORCE_COL: np.zeros(len(df_dh), dtype=float),
        }
    )
    n = len(df)
    json_n = stored_cycle_points_grid_n(name)
    loaded = load_cycle_points_original(name)
    json_points_ok = (
        loaded is not None
        and json_n == n
        and all(0 <= int(p["idx"]) < n for p in loaded[0])
    )
    if json_points_ok:
        points, _seg = loaded
        boundary_indices = sorted({int(p["idx"]) for p in points} | {0, n})
    else:
        if loaded is not None and json_n is not None and json_n != n:
            print(
                f"  {name} (digitized): cycle_points JSON n={json_n} != filtered drive n={n}; "
                "re-detecting on filtered drive"
            )
        elif loaded is not None:
            print(
                f"  {name} (digitized): cycle_points indices invalid for n={n}; re-detecting"
            )
        points, _ = find_cycle_points(df)
        boundary_indices = sorted({int(p["idx"]) for p in points} | {0, n})

    row = catalog_by_name.loc[name]
    L_T = float(row["L_T"])
    L_y = float(row["L_y"])
    A_sc = float(row["A_sc"])
    A_t = float(row["A_t"])
    fyp = float(row["fyp"])
    E_ksi = float(e_by_name.get(name, DEFAULT_E_KSI))
    u_col = df[DEF_COL].to_numpy(dtype=float)
    d_samp = d_sampling_from_brace_params(
        fyp_ksi=fyp,
        L_T_in=L_T,
        L_y_in=L_y,
        A_sc_in2=A_sc,
        A_t_in2=A_t,
        E_ksi=E_ksi,
        u_fallback=u_col,
    )
    d_samp = float(max(d_samp, 1e-15))

    out_df, remap = resample_by_segments(df, boundary_indices, d_samp)
    u_lead = _lead_replace_first_zero_deformation(out_df[DEF_COL].to_numpy(), d_samp)
    out_df = out_df.copy()
    out_df[DEF_COL] = u_lead
    new_points = remap_cycle_points(points, remap, out_df)

    rdir = RESAMPLED_DIR / name
    rdir.mkdir(parents=True, exist_ok=True)
    write_deformation_history_step_csv(rdir / DEFORMATION_HISTORY_CSV, out_df[DEF_COL].values)
    df_fd = pd.read_csv(fd_f)
    df_fd.to_csv(rdir / FORCE_DEFORMATION_CSV, index=False)

    CYCLE_POINTS_RESAMPLED_DIR.mkdir(parents=True, exist_ok=True)
    with open(CYCLE_POINTS_RESAMPLED_DIR / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump({"points": new_points, "n": len(out_df)}, f, indent=2)

    print(
        f"  {name} (digitized): drive n={n} -> {len(out_df)}, D_sampling={d_samp:.6g} in; "
        f"cloud rows={len(df_fd)} copied"
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Resample filtered outputs into data/resampled/{Name}/; write cycle_points_resampled JSON (path-ordered F-u and digitized drive).",
    )
    parser.add_argument(
        "--specimen",
        type=str,
        default=None,
        help="Single specimen Name; default: all with filtered data and catalog row",
    )
    args = parser.parse_args()

    catalog = read_catalog(CATALOG_PATH)
    catalog_by_name = catalog.set_index("Name")

    initial_df = None
    if INITIAL_BRB_PARAMETERS_PATH.exists():
        initial_df = pd.read_csv(INITIAL_BRB_PARAMETERS_PATH)
    e_by_name = _e_ksi_by_name(initial_df)

    names = _filtered_specimen_names(catalog_by_name)
    if args.specimen:
        if args.specimen not in names:
            print(f"Specimen {args.specimen} not found or not in catalog.")
            return
        names = [args.specimen]

    if not names:
        print("No specimens to resample.")
        return

    print(
        "Resampling filtered -> resampled (per-specimen folders; cycle_points_resampled for path + digitized drive)...",
        flush=True,
    )
    for name in names:
        try:
            rec = get_specimen_record(name, catalog)
            if uses_unordered_inputs(rec):
                process_specimen_digitized_unordered(str(name), catalog_by_name, e_by_name, catalog)
            else:
                process_specimen(str(name), catalog_by_name, e_by_name, catalog)
        except Exception as exc:
            print(
                f"ERROR resample_filtered specimen {name!r}: {type(exc).__name__}: {exc}",
                flush=True,
            )
            raise
    print(f"Wrote CSVs under {RESAMPLED_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Wrote cycle JSON under {CYCLE_POINTS_RESAMPLED_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Wrote plots under {PLOTS_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
