"""
Unified per-``set_id`` configuration.

Source of truth: ``config/calibration/set_id_settings.csv`` (one row per set_id).

This file merges steel seeds / b_p,b_n sourcing and per-set optimize/loss settings into one CSV.

**Parameter aliases (SteelMPF seed columns):** a cell may contain another column's
canonical name (e.g. ``b_lc`` = ``b_ic``) instead of a number. That means the slave parameter
tracks the master while the slave is **not** listed in ``optimize_params``. During optimization,
if the slave **is** optimized, the alias is ignored (the optimizer owns the slave). Initial seeds
for ``build_initial_brb_parameters`` resolve aliases to numeric values after the master is known.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Set
from pathlib import Path

import numpy as np
import pandas as pd

from calibrate.calibration_loss_settings import (
    CalibrationLossSettings,
    DEFAULT_CALIBRATION_LOSS_SETTINGS,
    calibration_loss_settings_from_partial_dict,
)
from calibrate.calibration_paths import SET_ID_SETTINGS_CSV
from calibrate.set_id_optimize_params import (
    _CANONICAL_BY_NORMALIZED_KEY,
    _normalize_optimize_token_key,
    _parse_optimize_params_cell,
)
from calibrate.steel_model import (
    SHARED_STEEL_KEYS,
    STEELMPF_ISO_KEYS,
    normalize_steel_model,
)


def _seed_columns_for_steel_model(_steel_model: object = None) -> frozenset[str]:
    """Numeric seed columns allowed in ``set_id_settings.csv`` (SteelMPF)."""
    normalize_steel_model(_steel_model)
    return frozenset((*SHARED_STEEL_KEYS, *STEELMPF_ISO_KEYS))


def parse_param_alias_bindings_from_row(row: pd.Series, steel_model: object) -> dict[str, str]:
    """
    Return slave_param -> master_param for cells whose value is another seed column name.

    Numeric cells and unrecognized strings are skipped (strings that are not valid parameter
    names are ignored so ``b_p`` / ``b_n`` statistic keywords keep working elsewhere).
    """
    allow = _seed_columns_for_steel_model(steel_model)
    ties: dict[str, str] = {}
    for k in allow:
        if k not in row.index:
            continue
        v = row[k]
        if pd.isna(v):
            continue
        if isinstance(v, (bool, np.bool_)):
            continue
        if isinstance(v, (int, float, np.integer, np.floating)):
            fv = float(v)
            if np.isfinite(fv):
                continue
        s = str(v).strip()
        if not s or s.lower() == "nan":
            continue
        try:
            float(s)
            continue
        except ValueError:
            pass
        key = _normalize_optimize_token_key(s)
        master = _CANONICAL_BY_NORMALIZED_KEY.get(key)
        if master is None:
            continue
        if master not in allow:
            raise ValueError(
                f"set_id={row.get('set_id')}: column {k!r} references {master!r}, "
                f"which is not a seed column for this steel_model"
            )
        if master == k:
            raise ValueError(f"set_id={row.get('set_id')}: column {k!r} cannot alias itself")
        ties[str(k)] = master
    return ties


def apply_param_value_ties(
    kw: MutableMapping[str, float],
    ties: Mapping[str, str],
    optimized_param_names: Set[str],
) -> None:
    """
    For each slave -> master alias, set ``kw[slave] = kw[master]`` if slave is **not** optimized.

    Call after assembling simulation kwargs. Masters must
    already be present in ``kw``.
    """
    for slave, master in ties.items():
        if slave in optimized_param_names:
            continue
        if master not in kw:
            raise KeyError(
                f"param alias {slave!r} -> {master!r}: master missing from simulation kwargs"
            )
        kw[slave] = float(kw[master])


def sync_tied_columns_in_output_row(
    row: pd.Series,
    ties: Mapping[str, str],
    optimized_param_names: Set[str],
) -> None:
    """Copy master values onto tied slave columns for rows written after optimization."""
    for slave, master in ties.items():
        if slave in optimized_param_names:
            continue
        if master in row.index and slave in row.index:
            row[slave] = row[master]


def read_set_id_settings_table(path: Path | None = None) -> pd.DataFrame:
    """
    Read ``set_id_settings.csv`` without validating ``set_id`` rows.

    Uses ``skipinitialspace=True`` and strips header names so aligned / padded spreadsheets
    (spaces after commas before quoted cells) parse as correct columns.
    """
    csv_path = Path(path).expanduser().resolve() if path is not None else SET_ID_SETTINGS_CSV
    df = pd.read_csv(csv_path, comment="#", skipinitialspace=True)
    df.columns = df.columns.astype(str).str.strip()
    return df


def load_set_id_settings(path: Path | None = None) -> pd.DataFrame:
    """
    Read the unified per-set_id settings CSV.

    Validates presence/uniqueness of `set_id` and normalizes it to int.

    ``skipinitialspace=True`` is required when cells use spaces for visual alignment after
    commas **before** a quoted field (e.g. `, "cR1,cR2,a1,a3"`). Without it, Python's CSV
    parser treats the comma inside quotes as delimiters (broken rows / wrong columns).
    Header names are stripped so spaces after commas in the header row still match ``set_id``,
    ``steel_model``, etc.
    """
    csv_path = Path(path).expanduser().resolve() if path is not None else SET_ID_SETTINGS_CSV
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing set_id settings CSV: {csv_path}")
    df = read_set_id_settings_table(csv_path)
    if df.empty:
        raise ValueError(f"{csv_path}: no data rows")
    if "set_id" not in df.columns:
        raise ValueError(f"{csv_path}: expected column 'set_id'; got {list(df.columns)}")

    sid = pd.to_numeric(df["set_id"].astype(str).str.strip(), errors="coerce")
    if sid.isna().any():
        bad = df.loc[sid.isna(), "set_id"].tolist()
        raise ValueError(f"{csv_path}: invalid set_id value(s): {bad}")
    sid_int = sid.astype(int)
    if (sid_int < 1).any():
        bad = df.loc[sid_int < 1, "set_id"].tolist()
        raise ValueError(f"{csv_path}: set_id must be >= 1; got {bad}")
    if sid_int.duplicated().any():
        dups = sorted(set(sid_int[sid_int.duplicated()].tolist()))
        raise ValueError(f"{csv_path}: duplicate set_id(s): {dups}")

    out = df.copy()
    out["set_id"] = sid_int
    out = out.sort_values("set_id")
    return out


def load_param_alias_ties_by_set_id(path: Path | None = None) -> dict[int, dict[str, str]]:
    """``set_id`` -> ``{slave_param: master_param}`` from string alias cells in the settings CSV."""
    df = load_set_id_settings(path)
    out: dict[int, dict[str, str]] = {}
    for _, row in df.iterrows():
        sm = normalize_steel_model(row.get("steel_model"))
        sid = int(row["set_id"])
        ties = parse_param_alias_bindings_from_row(row, sm)
        if ties:
            out[sid] = ties
    return out


LOSS_SETTINGS_KEYS: tuple[str, ...] = (
    "w_feat_l2",
    "w_feat_l1",
    "w_energy_l2",
    "w_energy_l1",
    "w_unordered_binenv_l2",
    "w_unordered_binenv_l1",
    "use_amplitude_weights",
    "amplitude_weight_power",
    "amplitude_weight_eps",
)


def load_set_id_optimize_and_loss(
    path: Path | None = None,
) -> tuple[dict[int, list[str]], dict[int, CalibrationLossSettings]]:
    """
    Return:
    - `set_id -> optimize_params` list (may be empty if column missing/blank for all rows)
    - `set_id -> CalibrationLossSettings` (partial per-row overrides; missing cells use defaults)
    """
    df = load_set_id_settings(path)
    csv_path = Path(path).expanduser().resolve() if path is not None else SET_ID_SETTINGS_CSV

    opt: dict[int, list[str]] = {}
    loss: dict[int, CalibrationLossSettings] = {}
    has_opt_col = "optimize_params" in df.columns
    for idx, row in df.iterrows():
        sid = int(row["set_id"])
        sm = normalize_steel_model(row.get("steel_model"))
        if has_opt_col and pd.notna(row.get("optimize_params")):
            raw = row.get("optimize_params")
            if str(raw).strip() and str(raw).strip().lower() != "nan":
                opt[sid] = _parse_optimize_params_cell(
                    raw, path=csv_path, set_id=sid, steel_model=sm
                )

        kv: dict[str, object] = {}
        for k in LOSS_SETTINGS_KEYS:
            if k in df.columns:
                kv[k] = row.get(k)
        loss[sid] = calibration_loss_settings_from_partial_dict(
            {str(k).lower(): v for k, v in kv.items()},
            default=DEFAULT_CALIBRATION_LOSS_SETTINGS,
        )
    return opt, loss


def resolve_set_id_settings_csv_arg(path: Path | None) -> Path:
    """CLI helper: if arg is None, return default; always returns resolved Path."""
    return (Path(path).expanduser().resolve() if path is not None else SET_ID_SETTINGS_CSV).resolve()


def load_steel_model_by_set_id(path: Path | None = None) -> dict[int, str]:
    """``set_id`` -> normalized ``steel_model`` from the settings CSV."""
    df = load_set_id_settings(path)
    return {int(row["set_id"]): normalize_steel_model(row.get("steel_model")) for _, row in df.iterrows()}


_INHERIT_FROM_SET_LEGACY_COL = "init_from_set_id"
_INHERIT_FROM_SET_PRIMARY_COL = "inherit_from_set"


def _is_missing_set_id_inherit_cell(v: object) -> bool:
    """Treat blank / NaN / -999 sentinel as 'no inheritance'."""
    if v is None:
        return True
    if isinstance(v, float) and pd.isna(v):
        return True
    s = str(v).strip()
    if s == "" or s.lower() in ("nan", "none", "-999"):
        return True
    try:
        return int(float(s)) == -999
    except (TypeError, ValueError):
        return False


def load_inherit_from_set_by_set_id(path: Path | None = None) -> dict[int, int]:
    """
    ``set_id`` -> parent ``set_id`` from ``inherit_from_set`` (or legacy ``init_from_set_id``).

    Rows with blank / ``-999`` / missing column are omitted. Validates self-reference and >= 1.
    Used by the optimizer to overlay a parent's just-optimized parameter values onto a child's
    seed row before L-BFGS, so non-optimized params for the child track the parent's solution
    within a single pipeline run.
    """
    df = load_set_id_settings(path)
    csv_path = Path(path).expanduser().resolve() if path is not None else SET_ID_SETTINGS_CSV
    out: dict[int, int] = {}
    for _, row in df.iterrows():
        sid = int(row["set_id"])
        primary = (
            row.get(_INHERIT_FROM_SET_PRIMARY_COL)
            if _INHERIT_FROM_SET_PRIMARY_COL in df.columns
            else None
        )
        legacy = (
            row.get(_INHERIT_FROM_SET_LEGACY_COL)
            if _INHERIT_FROM_SET_LEGACY_COL in df.columns
            else None
        )
        chosen = primary if not _is_missing_set_id_inherit_cell(primary) else legacy
        if _is_missing_set_id_inherit_cell(chosen):
            continue
        try:
            parent = int(pd.to_numeric(chosen, errors="raise"))
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"{csv_path}: set_id={sid}: inherit_from_set must be an integer or -999/blank"
            ) from e
        if parent < 1:
            raise ValueError(
                f"{csv_path}: set_id={sid}: inherit_from_set must be >= 1 (got {parent})"
            )
        if parent == sid:
            raise ValueError(
                f"{csv_path}: set_id={sid}: inherit_from_set cannot equal set_id"
            )
        out[sid] = parent
    return out

