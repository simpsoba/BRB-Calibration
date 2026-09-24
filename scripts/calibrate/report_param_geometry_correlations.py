"""
Compute correlations between optimal calibration parameters and geometry features.

Uses the same optimum rule as ``plot_individual_optimal_params_vs_geometry.py``: minimum
``final_J_feat_raw`` over successful metrics rows in the requested ``set_id`` range, joined to
``optimized_brb_parameters.csv``. Geometry features match the montage plots.

Outputs: tidy CSV of pairwise correlations (Pearson and Spearman) and heatmap PNGs under
``param_geometry_correlations/`` (default paths in ``--out-csv`` / ``--out-dir``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_SCRIPTS = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SCRIPTS))

from postprocess.specimen_catalog import read_catalog  # noqa: E402



def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_set_range(spec: str) -> list[int]:
    spec = spec.strip()
    if "-" in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def _resolve_set_ids(spec: str, metrics: pd.DataFrame) -> list[int]:
    """``all`` = every ``set_id`` with at least one successful metrics row; else parse range/list."""
    spec = spec.strip()
    if spec.lower() == "all":
        mask = metrics["success"].astype(bool) & np.isfinite(
            pd.to_numeric(metrics["final_J_feat_raw"], errors="coerce")
        )
        s = metrics.loc[mask, "set_id"]
        out = sorted(pd.to_numeric(s, errors="coerce").dropna().astype(int).unique().tolist())
        if not out:
            raise ValueError(
                "No successful metrics rows; cannot use --sets all. "
                "Pass an explicit range (e.g. --sets 1-12)."
            )
        return out
    return _parse_set_range(spec)


_METRIC_PARAM_CHECK: tuple[str, ...] = (
    "R0",
    "cR1",
    "cR2",
    "a1",
    "a3",
    "b_p",
    "b_n",
    "E",
)


def _read_csv_skip_hash(path: Path) -> pd.DataFrame:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    data_lines = [ln for ln in lines if not ln.strip().startswith("#")]
    from io import StringIO

    df = pd.read_csv(StringIO("\n".join(data_lines)), skipinitialspace=True)
    df.columns = df.columns.astype(str).str.strip()
    return df


def _pick_optimal_rows(
    metrics: pd.DataFrame, optimized: pd.DataFrame, set_ids: list[int]
) -> pd.DataFrame:
    """Best row per specimen (min ``final_J_feat_raw``) over successful metrics in ``set_ids``."""
    m = metrics[
        metrics["set_id"].isin(set_ids)
        & metrics["success"].astype(bool)
        & np.isfinite(metrics["final_J_feat_raw"])
    ].copy()
    if m.empty:
        raise ValueError("No successful metrics rows in the given set range.")

    need_cols = list(dict.fromkeys(["Name", "set_id", "steel_model", *_METRIC_PARAM_CHECK]))
    missing = [c for c in need_cols if c not in optimized.columns]
    if missing:
        raise KeyError(f"optimized_brb_parameters missing columns: {missing}")
    opt = optimized[need_cols].copy()
    merged = m.merge(opt, on=["Name", "set_id"], how="inner")
    sub = merged.copy()
    for c in _METRIC_PARAM_CHECK:
        if c not in sub.columns:
            continue
        sub = sub[np.isfinite(pd.to_numeric(sub[c], errors="coerce"))]
    if sub.empty:
        return pd.DataFrame(columns=list(merged.columns))
    sub = sub.sort_values(["Name", "final_J_feat_raw"])
    return sub.groupby("Name", as_index=False).first()


def _resolve_Q(catalog_row: pd.Series) -> float:
    from model.brace_geometry import compute_Q

    L_T = float(catalog_row["L_T"])
    L_y = float(catalog_row["L_y"])
    A_sc = float(catalog_row["A_sc"])
    A_t = float(catalog_row["A_t"])
    return compute_Q(L_T, L_y, A_sc, A_t)


def _geometry_features(catalog_row: pd.Series, E_kpsi: float, Q: float) -> dict[str, float]:
    L_y = float(catalog_row["L_y"])
    L_T = float(catalog_row["L_T"])
    A_sc = float(catalog_row["A_sc"])
    fy = float(catalog_row["fyp"])
    if A_sc <= 0:
        raise ValueError(f"Non-positive A_sc for {catalog_row.get('Name')!r}")
    Ly2_A = L_y**2 / A_sc
    LT2_A = L_T**2 / A_sc
    E_over_fy = E_kpsi / fy
    QE_over_fy = Q * E_over_fy
    Ly2 = L_y * L_y
    LT2 = L_T * L_T
    E_Asc_over_fy_Ly2 = (E_kpsi * A_sc) / (fy * Ly2) if Ly2 else np.nan
    E_Asc_over_fy_LT2 = (E_kpsi * A_sc) / (fy * LT2) if LT2 else np.nan
    QE_Asc_over_fy_Ly2 = Q * E_Asc_over_fy_Ly2 if np.isfinite(E_Asc_over_fy_Ly2) else np.nan
    QE_Asc_over_fy_LT2 = Q * E_Asc_over_fy_LT2 if np.isfinite(E_Asc_over_fy_LT2) else np.nan
    return {
        "L_y": L_y,
        "L_T": L_T,
        "A_sc": A_sc,
        "Ly2_over_A_sc": Ly2_A,
        "LT2_over_A_sc": LT2_A,
        "E_div_fy": E_over_fy,
        "Q": Q,
        "QE_div_fy": QE_over_fy,
        "E_Asc_over_fy_Ly2": E_Asc_over_fy_Ly2,
        "E_Asc_over_fy_LT2": E_Asc_over_fy_LT2,
        "QE_Asc_over_fy_Ly2": QE_Asc_over_fy_Ly2,
        "QE_Asc_over_fy_LT2": QE_Asc_over_fy_LT2,
    }


GEOMETRY_COLS: list[str] = [
    "L_y",
    "L_T",
    "A_sc",
    "Ly2_over_A_sc",
    "LT2_over_A_sc",
    "E_div_fy",
    "Q",
    "QE_div_fy",
    "E_Asc_over_fy_Ly2",
    "E_Asc_over_fy_LT2",
    "QE_Asc_over_fy_Ly2",
    "QE_Asc_over_fy_LT2",
]

# Parameters to correlate. Exclude R0 and E because they are kept constant in the runs.
PARAM_COLS: list[str] = ["cR1", "cR2", "a1", "a3", "b_p", "b_n"]

GEOMETRY_LATEX: dict[str, str] = {
    "L_y": r"$L_y$",
    "L_T": r"$L_T$",
    "A_sc": r"$A_{sc}$",
    "Ly2_over_A_sc": r"$L_y^2/A_{sc}$",
    "LT2_over_A_sc": r"$L_T^2/A_{sc}$",
    "E_div_fy": r"$E/f_y$",
    "Q": r"$Q$",
    "QE_div_fy": r"$QE/f_y$",
    "E_Asc_over_fy_Ly2": r"$\frac{E A_{sc}}{f_y L_y^2}$",
    "E_Asc_over_fy_LT2": r"$\frac{E A_{sc}}{f_y L_T^2}$",
    "QE_Asc_over_fy_Ly2": r"$\frac{Q E A_{sc}}{f_y L_y^2}$",
    "QE_Asc_over_fy_LT2": r"$\frac{Q E A_{sc}}{f_y L_T^2}$",
}

PARAM_LATEX: dict[str, str] = {
    "cR1": r"$c_{R1}$",
    "cR2": r"$c_{R2}$",
    "a1": r"$a_1$",
    "a3": r"$a_3$",
    "b_p": r"$b_p$",
    "b_n": r"$b_n$",
}

def _build_train_frame(
    catalog: pd.DataFrame, best: pd.DataFrame, param_cols: list[str] | None = None
) -> pd.DataFrame:
    cols = list(param_cols) if param_cols is not None else list(PARAM_COLS)
    cat = catalog.set_index("Name")
    b = best.set_index("Name")
    rows: list[dict[str, object]] = []
    for name, crow in cat.iterrows():
        if name not in b.index:
            continue
        # "Train cohort" is the same selection used in the geometry plots.
        gw = pd.to_numeric(crow.get("generalized_weight"), errors="coerce")
        if not (np.isfinite(gw) and float(gw) > 0.0):
            continue
        opt = b.loc[name]
        E = float(opt["E"])
        Q = _resolve_Q(crow)
        g = _geometry_features(crow, E, Q)
        rec: dict[str, object] = {"Name": str(name)}
        rec.update({k: float(v) for k, v in g.items()})
        for p in cols:
            if p not in opt.index:
                continue
            rec[p] = float(opt[p])
        rows.append(rec)
    return pd.DataFrame(rows)


def _build_extended_b_frame(
    catalog: pd.DataFrame, best: pd.DataFrame, apparent: pd.DataFrame
) -> pd.DataFrame:
    """
    Extended dataset "like the plots":
    - geometry is included for all specimens present in the catalog
    - cR1/cR2/a1/a3 are only populated for the train cohort (generalized_weight > 0), from optimal rows
    - b_p/b_n are populated for train from optimal rows, and for non-train from apparent means
    """
    cat = catalog.set_index("Name")
    b = best.set_index("Name")
    app = apparent.set_index("Name")
    rows: list[dict[str, object]] = []
    for name, crow in cat.iterrows():
        arow = app.loc[name] if name in app.index else None

        # E used for geometry normalization. Use optimal E when available; otherwise fall back to 29000.
        E_kpsi = 29000.0
        if name in b.index:
            try:
                E_kpsi = float(b.loc[name].get("E"))
            except Exception:
                E_kpsi = 29000.0
        Q = _resolve_Q(crow)
        g = _geometry_features(crow, float(E_kpsi), float(Q))
        rec: dict[str, object] = {"Name": str(name)}
        rec.update({k: float(v) for k, v in g.items()})

        gw = pd.to_numeric(crow.get("generalized_weight"), errors="coerce")
        is_train = bool(np.isfinite(gw) and float(gw) > 0.0)

        # Steel params: train only (from optimal rows).
        if is_train and name in b.index:
            opt = b.loc[name]
            for p in ("cR1", "cR2", "a1", "a3"):
                rec[p] = float(opt[p])
        else:
            for p in ("cR1", "cR2", "a1", "a3"):
                rec[p] = np.nan

        # b_p / b_n: train from optimal; non-train from apparent means.
        if is_train and name in b.index:
            opt = b.loc[name]
            rec["b_p"] = float(opt["b_p"])
            rec["b_n"] = float(opt["b_n"])
        else:
            bp = arow.get("b_p_mean") if arow is not None else np.nan
            bn = arow.get("b_n_mean") if arow is not None else np.nan
            rec["b_p"] = float(bp) if pd.notna(bp) else np.nan
            rec["b_n"] = float(bn) if pd.notna(bn) else np.nan

        rows.append(rec)
    return pd.DataFrame(rows)


def _pairwise_n(x: pd.Series, y: pd.Series) -> int:
    a = pd.to_numeric(x, errors="coerce")
    b = pd.to_numeric(y, errors="coerce")
    m = np.isfinite(a.to_numpy(dtype=float)) & np.isfinite(b.to_numpy(dtype=float))
    return int(np.sum(m))


def _tidy_correlations(df: pd.DataFrame, *, geometry_cols: list[str], param_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for p in param_cols:
        for g in geometry_cols:
            n = _pairwise_n(df[p], df[g])
            pear = df[[p, g]].corr(method="pearson", min_periods=2).iloc[0, 1]
            spear = df[[p, g]].corr(method="spearman", min_periods=2).iloc[0, 1]
            rows.append(
                {
                    "param": p,
                    "geometry": g,
                    "n": n,
                    "pearson_r": float(pear) if pd.notna(pear) else np.nan,
                    "spearman_r": float(spear) if pd.notna(spear) else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    out = out.sort_values(["param", "geometry"]).reset_index(drop=True)
    return out


def _corr_matrix(
    df: pd.DataFrame, *, geometry_cols: list[str], param_cols: list[str], method: str
) -> pd.DataFrame:
    mat = pd.DataFrame(index=param_cols, columns=geometry_cols, dtype=float)
    for p in param_cols:
        for g in geometry_cols:
            v = df[[p, g]].corr(method=str(method), min_periods=2).iloc[0, 1]
            mat.loc[p, g] = float(v) if pd.notna(v) else np.nan
    return mat


def _pairwise_n_matrix(
    df: pd.DataFrame, *, row_cols: list[str], col_cols: list[str]
) -> pd.DataFrame:
    nmat = pd.DataFrame(index=row_cols, columns=col_cols, dtype=float)
    for r in row_cols:
        for c in col_cols:
            nmat.loc[r, c] = _pairwise_n(df[r], df[c])
    return nmat


def _heatmap_corr(
    mat: pd.DataFrame,
    *,
    out_path: Path,
    title: str,
    cbar_label: str,
    n_mat: pd.DataFrame | None = None,
    x_label_map: dict[str, str] | None = None,
    y_label_map: dict[str, str] | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    z = mat.to_numpy(dtype=float)
    fig_w = max(10.0, 0.6 * mat.shape[1])
    fig_h = max(5.0, 0.45 * mat.shape[0])
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), layout="constrained")
    im = ax.imshow(z, vmin=-1.0, vmax=1.0, cmap="coolwarm", aspect="auto")
    ax.set_title(title)
    x_label_map = x_label_map or {}
    y_label_map = y_label_map or {}
    ax.set_yticks(
        np.arange(mat.shape[0]),
        labels=[y_label_map.get(str(x), str(x)) for x in mat.index],
    )
    ax.set_xticks(
        np.arange(mat.shape[1]),
        labels=[x_label_map.get(str(x), str(x)) for x in mat.columns],
        rotation=45,
        ha="right",
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label(cbar_label)
    # Annotate with rho only, or rho + n (extended).
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = z[i, j]
            if not np.isfinite(v):
                txt = "—"
            else:
                if n_mat is not None:
                    n = n_mat.iloc[i, j]
                    nn = int(n) if pd.notna(n) else 0
                    txt = f"{v:+.2f}\n(n={nn})"
                else:
                    txt = f"{v:+.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _combined_train_heatmap(
    mat_pg: pd.DataFrame,
    mat_pp: pd.DataFrame,
    *,
    out_path: Path,
    title: str,
    cbar_label: str,
    n_pg: pd.DataFrame | None = None,
    n_pp: pd.DataFrame | None = None,
    param_latex_map: dict[str, str] | None = None,
) -> None:
    """
    One heatmap with columns [geometry..., params...] and rows [params...].
    Left block: params vs geometry. Right block: params vs params.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if list(mat_pg.index) != list(mat_pp.index):
        raise ValueError("Expected params-vs-geometry and params-vs-params to share the same row index.")

    cols = list(mat_pg.columns) + list(mat_pp.columns)
    combined = pd.concat([mat_pg, mat_pp], axis=1)
    combined = combined.loc[mat_pg.index, cols]

    combined_n: pd.DataFrame | None = None
    if n_pg is not None and n_pp is not None:
        combined_n = pd.concat([n_pg, n_pp], axis=1).loc[mat_pg.index, cols]

    z = combined.to_numpy(dtype=float)
    fig_w = max(14.0, 0.55 * combined.shape[1] + 4.0)
    fig_h = max(6.0, 0.5 * combined.shape[0] + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), layout="constrained")
    im = ax.imshow(z, vmin=-1.0, vmax=1.0, cmap="coolwarm", aspect="auto")
    if str(title).strip():
        ax.set_title(title)

    pl = param_latex_map or PARAM_LATEX
    x_map: dict[str, str] = {}
    x_map.update(GEOMETRY_LATEX)
    x_map.update(pl)
    y_map = pl

    ax.set_yticks(
        np.arange(combined.shape[0]),
        labels=[y_map.get(str(x), str(x)) for x in combined.index],
    )
    ax.set_xticks(
        np.arange(combined.shape[1]),
        labels=[x_map.get(str(x), str(x)) for x in combined.columns],
        rotation=45,
        ha="right",
    )

    # Separator between geometry columns and parameter columns.
    sep_x = len(mat_pg.columns) - 0.5
    ax.axvline(sep_x, color="k", linewidth=1.0, alpha=0.6)

    for i in range(combined.shape[0]):
        for j in range(combined.shape[1]):
            v = z[i, j]
            if not np.isfinite(v):
                txt = "—"
            else:
                if combined_n is not None:
                    n = combined_n.iloc[i, j]
                    nn = int(n) if pd.notna(n) else 0
                    txt = f"{v:+.2f}\n(n={nn})"
                else:
                    txt = f"{v:+.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label(cbar_label)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    root = _repo_root()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=root)
    p.add_argument(
        "--catalog",
        type=Path,
        default=root / "config" / "calibration" / "BRB-Specimens.csv",
    )
    p.add_argument(
        "--metrics",
        type=Path,
        default=root
        / "results"
        / "calibration"
        / "individual_optimize"
        / "optimized_brb_parameters_metrics.csv",
    )
    p.add_argument(
        "--optimized-params",
        type=Path,
        default=root
        / "results"
        / "calibration"
        / "individual_optimize"
        / "optimized_brb_parameters.csv",
    )
    p.add_argument(
        "--sets",
        type=str,
        default="all",
        help=(
            "Comma list or a-b inclusive set_id range for picking per-specimen optima. "
            "Default 'all' uses every set_id with at least one successful metrics row."
        ),
    )
    p.add_argument(
        "--apparent-bn-bp",
        type=Path,
        default=root / "results" / "calibration" / "specimen_apparent_bn_bp.csv",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=root / "summary_statistics" / "param_geometry_correlations_train.csv",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=root
        / "results"
        / "plots"
        / "calibration"
        / "individual_optimize"
        / "param_geometry_correlations",
    )
    args = p.parse_args()

    catalog = read_catalog(Path(args.catalog))
    metrics = pd.read_csv(Path(args.metrics))
    optimized = pd.read_csv(Path(args.optimized_params))
    apparent = pd.read_csv(Path(args.apparent_bn_bp))
    set_ids = _resolve_set_ids(args.sets, metrics)

    # Render mathtext nicely (no external LaTeX dependency).
    plt.rcParams.update(
        {
            "text.usetex": False,
            "mathtext.fontset": "stix",
            "font.family": "STIXGeneral",
        }
    )

    best = _pick_optimal_rows(metrics, optimized, set_ids)
    if best.empty:
        raise SystemExit(
            "No optimum rows in the set range: cannot build correlation tables "
            "(same basis as geometry plots under /steelmpf)."
        )
    df_train = _build_train_frame(catalog, best)
    if df_train.empty:
        raise SystemExit(
            "No train-cohort rows available for correlation (check generalized_weight and inputs)."
        )
    df_ext = _build_extended_b_frame(catalog, best, apparent)

    tidy = _tidy_correlations(df_train, geometry_cols=GEOMETRY_COLS, param_cols=PARAM_COLS)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    tidy.to_csv(args.out_csv, index=False)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pairwise n from the extended dataset (availability differs by parameter).
    n_pg = _pairwise_n_matrix(df_ext, row_cols=PARAM_COLS, col_cols=GEOMETRY_COLS)
    n_pp = _pairwise_n_matrix(df_ext, row_cols=PARAM_COLS, col_cols=PARAM_COLS)

    # --- Spearman (train vs extended) ---
    s_pg_train = _corr_matrix(
        df_train, geometry_cols=GEOMETRY_COLS, param_cols=PARAM_COLS, method="spearman"
    )
    s_pp_train = _corr_matrix(
        df_train, geometry_cols=PARAM_COLS, param_cols=PARAM_COLS, method="spearman"
    )
    s_pg_train.to_csv(out_dir / "spearman_matrix_train.csv")
    s_pp_train.to_csv(out_dir / "spearman_params_matrix_train.csv")
    _combined_train_heatmap(
        s_pg_train,
        s_pp_train,
        out_path=out_dir / "spearman_heatmaps_train.png",
        title="",
        cbar_label="Spearman ρ",
    )

    s_pg_ext = _corr_matrix(
        df_ext, geometry_cols=GEOMETRY_COLS, param_cols=PARAM_COLS, method="spearman"
    )
    s_pp_ext = _corr_matrix(
        df_ext, geometry_cols=PARAM_COLS, param_cols=PARAM_COLS, method="spearman"
    )
    s_pg_ext.to_csv(out_dir / "spearman_matrix_train_extended.csv")
    s_pp_ext.to_csv(out_dir / "spearman_params_matrix_train_extended.csv")
    _combined_train_heatmap(
        s_pg_ext,
        s_pp_ext,
        out_path=out_dir / "spearman_heatmaps_train_extended.png",
        title="",
        cbar_label="Spearman ρ",
        n_pg=n_pg,
        n_pp=n_pp,
    )

    # --- Pearson (train vs extended) ---
    p_pg_train = _corr_matrix(
        df_train, geometry_cols=GEOMETRY_COLS, param_cols=PARAM_COLS, method="pearson"
    )
    p_pp_train = _corr_matrix(
        df_train, geometry_cols=PARAM_COLS, param_cols=PARAM_COLS, method="pearson"
    )
    p_pg_train.to_csv(out_dir / "pearson_matrix_train.csv")
    p_pp_train.to_csv(out_dir / "pearson_params_matrix_train.csv")
    _combined_train_heatmap(
        p_pg_train,
        p_pp_train,
        out_path=out_dir / "pearson_heatmaps_train.png",
        title="",
        cbar_label="Pearson r",
    )

    p_pg_ext = _corr_matrix(
        df_ext, geometry_cols=GEOMETRY_COLS, param_cols=PARAM_COLS, method="pearson"
    )
    p_pp_ext = _corr_matrix(
        df_ext, geometry_cols=PARAM_COLS, param_cols=PARAM_COLS, method="pearson"
    )
    p_pg_ext.to_csv(out_dir / "pearson_matrix_train_extended.csv")
    p_pp_ext.to_csv(out_dir / "pearson_params_matrix_train_extended.csv")
    _combined_train_heatmap(
        p_pg_ext,
        p_pp_ext,
        out_path=out_dir / "pearson_heatmaps_train_extended.png",
        title="",
        cbar_label="Pearson r",
        n_pg=n_pg,
        n_pp=n_pp,
    )

if __name__ == "__main__":
    main()

