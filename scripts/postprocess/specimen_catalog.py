"""
BRB-Specimens.csv: specimen metadata, **where input CSVs live**, and who is in which pipeline.

**Start here if you** add a specimen, change file layout, or wonder why a name is skipped.

**On-disk inputs (under ``<project_root>/data/raw/{Name}/`` only; not the catalog field ``experimental_layout``).**
  Filtered and resampled products use the **same subfolder + filenames**: ``data/filtered/{Name}/`` and
  ``data/resampled/{Name}/`` each contain ``force_deformation.csv`` and ``deformation_history.csv``
  (``Step``, ``Deformation[in]``; no lab ``Time`` column).

  Layout is **not** split into ordered vs unordered folders--``BRB-Specimens.csv`` (``path_ordered``,
  ``experimental_layout``) tells how to interpret the files.

  - ``force_deformation.csv`` -- always the F-u CSV for that specimen: a **single path-ordered** series
    (``Deformation[in]``, ``Force[kip]``) when the row is in the filter/resample stack, or **unordered**
    ``(F,u)`` samples when ``digitized`` + ``path_ordered=false``.
  - ``deformation_history.csv`` -- prescribed ``Deformation[in]`` (and optional ``Time[s]``) for
    digitized unordered rows; omit for path-only specimens.

  Optional QA figures (not read by the calibration code): ``deformation_history.png``,
  ``force_deformation.png`` in the same folder.

**Catalog columns (see root README table)**
  Geometry / yield use the same names as simulation CSVs: ``L_T``, ``L_y``, ``A_sc``, ``A_t``, ``fyp``.
  ``read_catalog`` still accepts the older ``L_T_in`` / ``A_c_in2`` / ``f_yc_ksi`` headers and renames them.
  ``experimental_layout`` -- ``raw`` (lab specimen set) or ``digitized``.
  ``individual_optimize`` -- if true, the specimen may run ``optimize_brb_mse`` when resampled data exist.
  ``generalized_weight`` -- non-negative; for **path-ordered** rows it is the train weight in the
    automated generalized fit (``run.sh`` / ``run.ps1``). **1** = train; **0** = validation only
    (still scored/plotted, not in the joint objective). **Unordered** digitized rows never enter
    that aggregate (effective weight 0 in ``specimen_weights.py``).
    Which generalized **configuration** (``set_id`` row in ``set_id_settings_generalized.csv``) applies is defined
    only in that CSV when running ``optimize_generalized_brb_mse`` (not in this catalog).
  ``path_ordered`` -- if true, ``force_deformation.csv`` is treated as a path series and the specimen
    can enter filter/resample when that file exists. If false with ``digitized``, use
    ``deformation_history.csv`` + ``force_deformation.csv`` as the unordered F-u samples + drive pair.
  ``skip_filter_resample`` -- if true, path-ordered **filtered** = trim-valid raw (no Savitzky-Golay);
    digitized **drive** is not Savitzky-smoothed (unordered F-u file is always an unsmoothed copy of raw). Resampling
    still runs for both.

  Legacy CSV values for ``experimental_layout`` (``raw_single``, ``digitized_ordered``,
  ``digitized_cloud``, etc.) are normalized to ``raw`` / ``digitized``. If ``path_ordered`` is
  omitted, it defaults to true except for legacy ``digitized_cloud`` / ``digitized_split`` rows,
  which default to false.

**Main APIs**
  ``read_catalog`` / ``get_specimen_record`` -- parse and validate rows.
  ``uses_unordered_inputs`` -- true when ``digitized`` and not ``path_ordered``.
  ``specimen_raw_dir`` -- ``data/raw/{Name}/``.
  ``primary_f_u_csv_path`` -- path-ordered F-u for filter/resample (None for unordered digitized).
  ``deformation_history_csv_path`` / ``force_deformation_unordered_csv_path`` -- same folder; unordered F-u CSV paths
    (check ``.is_file()`` before use).
  ``deformation_history_png_path`` / ``force_deformation_png_path`` -- optional image paths (convention only).
  ``list_names_for_standard_pipeline`` / ``list_names_for_cycle_points`` / ``list_names_digitized_unordered``
    -- names that qualify for each step (see docstrings).
  ``max_abs_strain_delta_over_Ly`` -- peak test strain ``max |δ| / L_y`` from resolved F--u (see ``resolve_force_deformation_csv_for_max_strain``).

**Imports:** postprocess scripts add ``scripts/postprocess`` to ``sys.path`` and ``import specimen_catalog``.
Calibration scripts usually insert ``postprocess`` the same way. Pass ``project_root`` when resolving
paths against a clone or temp directory.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

# Catalog columns for optimize + generalized (see README).
INDIVIDUAL_OPTIMIZE_COL = "individual_optimize"
GENERALIZED_WEIGHT_COL = "generalized_weight"
# Metrics column: which ``set_id`` row from ``set_id_settings_generalized.csv`` produced this eval row.
GENERALIZED_CONFIG_SET_ID_COL = "generalized_config_set_id"

# Geometry / yield: same names as simulation and parameter CSVs (units in the README).
CATALOG_GEOMETRY_COLS = ("L_T", "L_y", "A_sc", "A_t", "fyp")
# Older catalog headers accepted on load, then renamed to the canonical names above.
_CATALOG_COLUMN_ALIASES = {
    "L_T_in": "L_T",
    "L_y_in": "L_y",
    "A_c_in2": "A_sc",
    "A_t_in2": "A_t",
    "f_yc_ksi": "fyp",
}

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CATALOG_PATH = PROJECT_ROOT / "config" / "calibration" / "BRB-Specimens.csv"
DATA_DIR = PROJECT_ROOT / "data"

DEFORMATION_HISTORY_CSV = "deformation_history.csv"
FORCE_DEFORMATION_CSV = "force_deformation.csv"
DEFORMATION_HISTORY_PNG = "deformation_history.png"
FORCE_DEFORMATION_PNG = "force_deformation.png"
# Filtered/resampled ``deformation_history.csv`` use step index instead of lab time.
STEP_COL = "Step"
TIME_COL = "Time[s]"
DEFORMATION_IN_COL = "Deformation[in]"
FORCE_KIP_COL = "Force[kip]"


def experimental_inputs_root(project_root: Path | None = None) -> Path:
    """``<project_root>/data/raw`` -- one subfolder per specimen ``Name``."""
    return (project_root or PROJECT_ROOT) / "data" / "raw"


def specimen_raw_dir(name: str, project_root: Path | None = None) -> Path:
    """``data/raw/{Name}/``."""
    return experimental_inputs_root(project_root) / str(name)


def filtered_specimen_dir(name: str, project_root: Path | None = None) -> Path:
    """``data/filtered/{Name}/`` -- ``force_deformation.csv`` + ``deformation_history.csv`` (Step, Deformation)."""
    return (project_root or PROJECT_ROOT) / "data" / "filtered" / str(name)


def resampled_specimen_dir(name: str, project_root: Path | None = None) -> Path:
    """``data/resampled/{Name}/`` -- same filenames as under ``filtered``/raw."""
    return (project_root or PROJECT_ROOT) / "data" / "resampled" / str(name)


def filtered_force_deformation_csv(name: str, project_root: Path | None = None) -> Path:
    """Path to filtered force_deformation.csv."""
    return filtered_specimen_dir(name, project_root) / FORCE_DEFORMATION_CSV


def filtered_deformation_history_csv(name: str, project_root: Path | None = None) -> Path:
    """Path to filtered deformation_history.csv."""
    return filtered_specimen_dir(name, project_root) / DEFORMATION_HISTORY_CSV


def resampled_force_deformation_csv(name: str, project_root: Path | None = None) -> Path:
    """Path to resampled force_deformation.csv."""
    return resampled_specimen_dir(name, project_root) / FORCE_DEFORMATION_CSV


def resampled_deformation_history_csv(name: str, project_root: Path | None = None) -> Path:
    """Path to resampled deformation_history.csv."""
    return resampled_specimen_dir(name, project_root) / DEFORMATION_HISTORY_CSV


def resolve_filtered_force_deformation_csv(name: str, project_root: Path | None = None) -> Path | None:
    """New layout ``data/filtered/{Name}/force_deformation.csv`` or legacy flat ``data/filtered/{Name}.csv``."""
    root = project_root or PROJECT_ROOT
    p = filtered_force_deformation_csv(name, root)
    if p.is_file():
        return p
    leg = root / "data" / "filtered" / f"{name}.csv"
    return leg if leg.is_file() else None


def resolve_resampled_force_deformation_csv(name: str, project_root: Path | None = None) -> Path | None:
    """New layout ``data/resampled/{Name}/force_deformation.csv`` or legacy ``data/resampled/{Name}.csv``."""
    root = project_root or PROJECT_ROOT
    p = resampled_force_deformation_csv(name, root)
    if p.is_file():
        return p
    leg = root / "data" / "resampled" / f"{name}.csv"
    return leg if leg.is_file() else None


def resolve_force_deformation_csv_for_max_strain(
    name: str,
    catalog: pd.DataFrame | None = None,
    *,
    project_root: Path | None = None,
) -> Path | None:
    """
    F--u CSV for peak ``|δ|``: resampled (path-ordered), else filtered/raw, matching ``extract_bn_bp`` precedence.
    """
    root = project_root or PROJECT_ROOT
    cat = catalog if catalog is not None else read_catalog()
    rec = get_specimen_record(str(name), cat)
    if uses_unordered_inputs(rec):
        p = resolve_filtered_force_deformation_csv(str(name), root)
        if p is not None and p.is_file():
            return p
        fu = force_deformation_unordered_csv_path(str(name), root)
        return fu if fu.is_file() else None
    pr = resolve_resampled_force_deformation_csv(str(name), root)
    if pr is not None and pr.is_file():
        return pr
    pf = resolve_filtered_force_deformation_csv(str(name), root)
    if pf is not None and pf.is_file():
        return pf
    prim = primary_f_u_csv_path(str(name), cat, project_root=root)
    if prim is not None and prim.is_file():
        return prim
    return None


def max_abs_strain_delta_over_Ly(
    name: str,
    catalog: pd.DataFrame | None = None,
    *,
    project_root: Path | None = None,
    ly_in: float | None = None,
) -> float | None:
    """``max |Deformation[in]| / L_y`` (dimensionless) from the resolved experimental CSV."""
    root = project_root or PROJECT_ROOT
    cat = catalog if catalog is not None else read_catalog()
    path = resolve_force_deformation_csv_for_max_strain(str(name), cat, project_root=root)
    if path is None:
        return None
    row = cat[cat["Name"].astype(str) == str(name)]
    if row.empty:
        return None
    if ly_in is None:
        ly_in = float(catalog_geometry(row.iloc[0])["L_y"])
    if ly_in <= 0 or not np.isfinite(ly_in):
        return None
    try:
        df = pd.read_csv(path, usecols=[DEFORMATION_IN_COL])
    except ValueError:
        df = pd.read_csv(path)
        if DEFORMATION_IN_COL not in df.columns:
            return None
    u = pd.to_numeric(df[DEFORMATION_IN_COL], errors="coerce").to_numpy(dtype=float)
    u = u[np.isfinite(u)]
    if u.size == 0:
        return None
    return float(np.max(np.abs(u)) / ly_in)


def write_deformation_history_step_csv(path: Path, deformation_in, *, def_col_name: str | None = None) -> None:
    """Write ``Step`` + ``Deformation[in]`` (no time column)."""
    col = def_col_name or DEFORMATION_IN_COL
    s = pd.Series(deformation_in, dtype=float)
    u = s.to_numpy()
    df = pd.DataFrame({STEP_COL: np.arange(len(u), dtype=int), col: u})
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


Layout = Literal["raw", "digitized"]

LAYOUT_VALUES = frozenset({"raw", "digitized"})

# Legacy CSV spellings -> canonical ``raw`` / ``digitized``.
_LAYOUT_ALIASES = {
    "raw_single": "raw",
    "digitized_ordered": "digitized",
    "digitized_cloud": "digitized",
    "digitized_single": "digitized",
    "digitized_split": "digitized",
}


def _canonical_layout_value(x) -> str:
    """Normalize experimental_layout cell to canonical string."""
    if pd.isna(x):
        return "raw"
    s = str(x).strip().lower()
    if s in LAYOUT_VALUES:
        return s
    return _LAYOUT_ALIASES.get(s, s)


@dataclass(frozen=True)
class SpecimenRecord:
    name: str
    experimental_layout: Layout
    path_ordered: bool
    #: If True, exclude from filter/resample/cycle_points standard pipeline (pre-cleaned data, etc.).
    skip_filter_resample: bool = False
    #: If True, include in ``optimize_brb_mse`` when resampled data exist.
    individual_optimize: bool = True


def uses_unordered_inputs(rec: SpecimenRecord) -> bool:
    """True when ``deformation_history.csv`` + unordered ``force_deformation.csv`` (not path-only primary)."""
    return rec.experimental_layout == "digitized" and not rec.path_ordered


def _parse_bool_cell(x) -> bool:
    """Parse spreadsheet bool (true/false/empty)."""
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in ("true", "1", "yes", "y"):
        return True
    if s in ("false", "0", "no", "n", ""):
        return False
    raise ValueError(f"Expected boolean string, got {x!r}")


def _canonicalize_catalog_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map legacy unit-suffixed headers onto ``L_T``, ``L_y``, ``A_sc``, ``A_t``, ``fyp``."""
    rename: dict[str, str] = {}
    for old, new in _CATALOG_COLUMN_ALIASES.items():
        if old in df.columns and new not in df.columns:
            rename[old] = new
    if rename:
        df = df.rename(columns=rename)
    drop = [
        c
        for c, new in _CATALOG_COLUMN_ALIASES.items()
        if c in df.columns and new in df.columns
    ]
    if drop:
        df = df.drop(columns=drop)
    return df


def catalog_geometry(row: pd.Series) -> dict[str, float]:
    """
    Brace geometry + yield from one catalog row (canonical names, or legacy aliases).

    Catalog ``fyp`` is used for both tension and compression unless ``fyn`` is present.
    """
    src = row.copy()
    for old, new in _CATALOG_COLUMN_ALIASES.items():
        if new not in src.index and old in src.index:
            src[new] = src[old]
    missing = [c for c in CATALOG_GEOMETRY_COLS if c not in src.index or pd.isna(src.get(c))]
    if missing:
        raise KeyError(f"catalog row missing {missing} for {src.get('Name')!r}")
    fy = float(src["fyp"])
    fyn = float(src["fyn"]) if "fyn" in src.index and pd.notna(src.get("fyn")) else fy
    return {
        "L_T": float(src["L_T"]),
        "L_y": float(src["L_y"]),
        "A_sc": float(src["A_sc"]),
        "A_t": float(src["A_t"]),
        "fyp": fy,
        "fyn": fyn,
    }


def read_catalog(catalog_path: Path | None = None) -> pd.DataFrame:
    """Load ``BRB-Specimens.csv``, default columns, validate ``experimental_layout``."""
    path = catalog_path or CATALOG_PATH
    # skipinitialspace + stripped headers: spreadsheets often pad headers (e.g. ``Name   ``).
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = df.columns.astype(str).str.strip()
    df = _canonicalize_catalog_columns(df)
    missing_geom = [c for c in CATALOG_GEOMETRY_COLS if c not in df.columns]
    if missing_geom:
        raise ValueError(
            f"BRB-Specimens.csv missing geometry columns {missing_geom}; "
            f"expected {list(CATALOG_GEOMETRY_COLS)} "
            f"(legacy aliases {list(_CATALOG_COLUMN_ALIASES)} are also accepted)."
        )
    if "Name" not in df.columns:
        raise ValueError("BRB-Specimens.csv must have Name column")
    df = df.copy()
    # Padded ``Name`` cells (e.g. ``PC250  ``) must match folder stems / stripped lookups.
    df["Name"] = df["Name"].apply(lambda x: str(x).strip() if pd.notna(x) else x)
    for req in (INDIVIDUAL_OPTIMIZE_COL, GENERALIZED_WEIGHT_COL):
        if req not in df.columns:
            raise ValueError(f"BRB-Specimens.csv must include column {req!r}")
    if "pool_joint_cohort" in df.columns:
        raise ValueError(
            "BRB-Specimens.csv: remove column 'pool_joint_cohort'; "
            "use 'individual_optimize' and 'generalized_weight'."
        )
    if "data_mode" in df.columns:
        df = df.drop(columns=["data_mode"])
    if "experimental_layout" not in df.columns:
        df["experimental_layout"] = "raw"
    lay_before = df["experimental_layout"].map(
        lambda x: str(x).strip().lower() if pd.notna(x) else ""
    )
    if "path_ordered" not in df.columns:
        df["path_ordered"] = True
        df.loc[lay_before.isin({"digitized_cloud", "digitized_split"}), "path_ordered"] = False
    if "skip_filter_resample" not in df.columns:
        df["skip_filter_resample"] = False
    df["experimental_layout"] = df["experimental_layout"].map(_canonical_layout_value)

    for _, row in df.iterrows():
        _parse_bool_cell(row[INDIVIDUAL_OPTIMIZE_COL])
        lay = str(row["experimental_layout"]).strip()
        if lay not in LAYOUT_VALUES:
            raise ValueError(f"Invalid experimental_layout {lay!r} for {row.get('Name')}")
        for wcol in (GENERALIZED_WEIGHT_COL,):
            v = row.get(wcol)
            if pd.notna(v):
                w = float(v)
                if not (w == w and w >= 0.0):
                    raise ValueError(f"Invalid {wcol} for {row.get('Name')}: {v!r}")
    return df


def get_specimen_record(name: str, catalog: pd.DataFrame | None = None) -> SpecimenRecord:
    """Typed view of one catalog row by Name."""
    cat = catalog if catalog is not None else read_catalog()
    key = str(name).strip()
    cat = cat[cat["Name"].astype(str).str.strip() == key]
    if cat.empty:
        raise KeyError(f"Name {key!r} not in catalog")
    row = cat.iloc[0]
    lay = _canonical_layout_value(row["experimental_layout"])
    if lay not in LAYOUT_VALUES:
        raise ValueError(f"Invalid experimental_layout {lay!r} for {key}")
    skip_fr = False
    if "skip_filter_resample" in row.index and pd.notna(row.get("skip_filter_resample")):
        try:
            skip_fr = _parse_bool_cell(row["skip_filter_resample"])
        except ValueError:
            skip_fr = False
    io = _parse_bool_cell(row[INDIVIDUAL_OPTIMIZE_COL])
    return SpecimenRecord(
        name=key,
        experimental_layout=lay,  # type: ignore[arg-type]
        path_ordered=_parse_bool_cell(row["path_ordered"]),
        skip_filter_resample=skip_fr,
        individual_optimize=io,
    )


def primary_f_u_csv_path(
    name: str, catalog: pd.DataFrame | None = None, *, project_root: Path | None = None
) -> Path | None:
    """
    Path-ordered ``force_deformation.csv`` for filter/resample if it exists.
    Digitized unordered (``digitized`` + ``path_ordered`` false) -> None.
    """
    root = project_root or PROJECT_ROOT
    rec = get_specimen_record(name, catalog)
    if uses_unordered_inputs(rec):
        return None
    p = specimen_raw_dir(name, root) / FORCE_DEFORMATION_CSV
    return p if p.is_file() else None


def deformation_history_csv_path(name: str, project_root: Path | None = None) -> Path:
    """``data/raw/{Name}/deformation_history.csv`` (may not exist for path-only specimens)."""
    return specimen_raw_dir(name, project_root) / DEFORMATION_HISTORY_CSV


def force_deformation_unordered_csv_path(name: str, project_root: Path | None = None) -> Path:
    """``data/raw/{Name}/force_deformation.csv`` (scatter cloud when catalog says scatter-cloud)."""
    return specimen_raw_dir(name, project_root) / FORCE_DEFORMATION_CSV


def deformation_history_png_path(name: str, project_root: Path | None = None) -> Path:
    """Optional QA image; same folder as CSVs."""
    return specimen_raw_dir(name, project_root) / DEFORMATION_HISTORY_PNG


def force_deformation_png_path(name: str, project_root: Path | None = None) -> Path:
    """Optional QA image; same folder as CSVs."""
    return specimen_raw_dir(name, project_root) / FORCE_DEFORMATION_PNG


def list_names_for_standard_pipeline(
    catalog: pd.DataFrame | None = None, *, project_root: Path | None = None
) -> list[str]:
    """Names with path_ordered True and an existing primary F-u CSV (includes ``skip_filter_resample``)."""
    cat = catalog if catalog is not None else read_catalog()
    root = project_root or PROJECT_ROOT
    out: list[str] = []
    for name in cat["Name"].astype(str).unique():
        rec = get_specimen_record(name, cat)
        if not rec.path_ordered:
            continue
        p = primary_f_u_csv_path(name, cat, project_root=root)
        if p is not None and p.is_file():
            out.append(name)
    return sorted(out)


def list_names_for_cycle_points(
    catalog: pd.DataFrame | None = None, *, project_root: Path | None = None
) -> list[str]:
    """path_ordered + primary F-u CSV exists (same name set as ``list_names_for_standard_pipeline``).

    Used by ``cycle_points.py`` so ``data/cycle_points_original`` indices match the trim-valid grid used
    for ``data/filtered`` (including ``skip_filter_resample`` rows, where filtered equals that raw).
    """
    cat = catalog if catalog is not None else read_catalog()
    root = project_root or PROJECT_ROOT
    out: list[str] = []
    for name in cat["Name"].astype(str).unique():
        rec = get_specimen_record(name, cat)
        if not rec.path_ordered:
            continue
        p = primary_f_u_csv_path(name, cat, project_root=root)
        if p is not None and p.is_file():
            out.append(name)
    return sorted(out)


def list_names_digitized_unordered(
    catalog: pd.DataFrame | None = None, *, project_root: Path | None = None
) -> list[str]:
    """Names with ``digitized`` + ``path_ordered`` false and both unordered CSVs present on disk."""
    cat = catalog if catalog is not None else read_catalog()
    root = project_root or PROJECT_ROOT
    out: list[str] = []
    for name in cat["Name"].astype(str).unique():
        rec = get_specimen_record(name, cat)
        if not uses_unordered_inputs(rec):
            continue
        dh = deformation_history_csv_path(name, root)
        fd = force_deformation_unordered_csv_path(name, root)
        if dh.is_file() and fd.is_file():
            out.append(name)
    return sorted(out)


def list_names_for_filter_outputs(
    catalog: pd.DataFrame | None = None, *, project_root: Path | None = None
) -> list[str]:
    """Union of path-ordered pipeline names and digitized unordered names (inputs on disk)."""
    a = set(list_names_for_standard_pipeline(catalog, project_root=project_root))
    b = set(list_names_digitized_unordered(catalog, project_root=project_root))
    return sorted(a | b)


def path_ordered_resampled_force_csv_stems(
    catalog: pd.DataFrame | None = None, *, project_root: Path | None = None
) -> set[str]:
    """Names with a resampled path-ordered ``force_deformation.csv`` (excludes unordered rows)."""
    cat = catalog if catalog is not None else read_catalog()
    root = project_root or PROJECT_ROOT
    out: set[str] = set()
    for name in cat["Name"].astype(str).unique():
        n = str(name).strip()
        rec = get_specimen_record(n, cat)
        if uses_unordered_inputs(rec):
            continue
        p = resolve_resampled_force_deformation_csv(n, root)
        if p is not None and p.is_file():
            out.add(n)
    return out
