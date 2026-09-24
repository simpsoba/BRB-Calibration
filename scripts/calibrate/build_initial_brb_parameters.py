"""
Build ``results/calibration/individual_optimize/initial_brb_parameters.csv`` from the specimen catalog and
``extract_bn_bp.py`` output.

**SteelMPF** seeds and apparent-**b** sourcing for each calibration ``set_id`` are read from
``config/calibration/set_id_settings.csv`` (unified per-``set_id`` config), unless
``--set-id-settings`` points elsewhere.

The CSV is **wide**: kinematic columns ``E``, ``b_p``, ``b_n``, ``R0``, ``cR1``, ``cR2``, and ``a1``–``a4``.
Blank cells fall back to defaults.

Geometry and ``fyp`` / ``fyn`` always come from the specimen catalog. Columns ``b_p`` and ``b_n`` are each
either a **numeric** literal or a **stat name** (case-insensitive) referencing ``specimen_apparent_bn_bp.csv``:
``median``, ``mean``, ``weighted_mean``, ``q1``, ``q3``, ``min``, ``max``, ``max_amplitude``
(``b`` from the half-cycle with largest **segment** excursion ``max(|u|)`` from ``zero_def`` to peak in that direction;
not ``max(b)`` across cycles). Blank ``b_p`` / ``b_n`` default to ``median``.

Run after ``resample_filtered.py`` and ``extract_bn_bp.py``.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
_SCRIPTS = _PROJECT_ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS / "postprocess"))
sys.path.insert(0, str(_SCRIPTS))

from calibrate.calibration_paths import (  # noqa: E402
    BRB_SPECIMENS_CSV,
    INITIAL_BRB_PARAMETERS_PATH,
    GENERALIZED_BRB_PARAMETERS_PATH,
    OPTIMIZED_BRB_PARAMETERS_PATH,
    SPECIMEN_APPARENT_BN_BP_PATH,
    SET_ID_SETTINGS_CSV,
)
from calibrate.set_id_settings import (  # noqa: E402
    load_set_id_settings,
    parse_param_alias_bindings_from_row,
)
from specimen_catalog import read_catalog  # noqa: E402
from calibrate.steel_model import (  # noqa: E402
    SHARED_STEEL_KEYS,
    STEELMPF_ISO_KEYS,
    STEEL_MODEL_STEELMPF,
    normalize_steel_model,
)

CATALOG_PATH = BRB_SPECIMENS_CSV
DEFAULT_BN_BP_PATH = SPECIMEN_APPARENT_BN_BP_PATH
DEFAULT_OUTPUT_PATH = INITIAL_BRB_PARAMETERS_PATH

NUMERIC_SEED_KEYS: tuple[str, ...] = (
    *SHARED_STEEL_KEYS,
    "b_p",
    "b_n",
    *STEELMPF_ISO_KEYS,
)

STEEL_DEFAULT: dict[str, float] = {
    "E": 29000.0,
    "R0": 20.0,
    "cR1": 0.925,
    "cR2": 0.15,
    "a1": 0.04,
    "a2": 1.0,
    "a3": 0.04,
    "a4": 1.0,
    "b_p": 0.01,
    "b_n": 0.025,
}

B_STAT_NAMES: frozenset[str] = frozenset(
    {"median", "mean", "weighted_mean", "q1", "q3", "min", "max", "max_amplitude"}
)

_BN_BP_STAT_COLUMNS: tuple[str, ...] = (
    "b_p_mean",
    "b_p_median",
    "b_p_weighted_mean",
    "b_p_max_amplitude",
    "b_p_q1",
    "b_p_q3",
    "b_p_min",
    "b_p_max",
    "b_n_mean",
    "b_n_median",
    "b_n_weighted_mean",
    "b_n_max_amplitude",
    "b_n_q1",
    "b_n_q3",
    "b_n_min",
    "b_n_max",
)


@dataclass(frozen=True)
class InitialBrbSeedRow:
    set_id: int
    steel_model: str
    steel_seed: dict[str, float]
    steel_overrides: dict[str, float]
    steel_ties: dict[str, str]
    b_p_spec: float | str
    b_n_spec: float | str
    init_from_set_id: int | None = None
    init_from_csv: Path | None = None


def _override_keys_for_steel_model(steel_model: str) -> frozenset[str]:
    normalize_steel_model(steel_model)
    bn = ("b_p", "b_n")
    return frozenset((*SHARED_STEEL_KEYS, *bn, *STEELMPF_ISO_KEYS))


def _validate_steel_default(d: dict[str, float]) -> None:
    keys = set(d)
    need = set(NUMERIC_SEED_KEYS)
    if keys != need:
        miss = sorted(need - keys)
        extra = sorted(keys - need)
        parts = []
        if miss:
            parts.append(f"missing {miss}")
        if extra:
            parts.append(f"extra {extra}")
        raise ValueError(f"STEEL_DEFAULT: {', '.join(parts)}")


def _row_set_id_label(row: pd.Series) -> str:
    return f"set_id={row.get('set_id', '?')!r}"


def _is_missing_sentinel(x: object) -> bool:
    """Treat -999 (and string '-999') as a blank cell for settings CSVs."""
    if x is None or pd.isna(x):
        return True
    if isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool):
        try:
            return float(x) == -999.0
        except Exception:
            return False
    s = str(x).strip()
    return s == "-999"


def _resolve_numeric_seed_overrides(row: pd.Series, steel_model: str) -> dict[str, float]:
    """
    Numeric steel seed overrides from the settings row, including cells that alias another column
    (see ``parse_param_alias_bindings_from_row``).
    """
    sm = normalize_steel_model(steel_model)
    allow = _override_keys_for_steel_model(sm)
    aliases = parse_param_alias_bindings_from_row(row, sm)
    resolved: dict[str, float] = {k: float(STEEL_DEFAULT[k]) for k in NUMERIC_SEED_KEYS}

    for k in allow:
        if k not in row.index:
            continue
        if _is_missing_sentinel(row[k]):
            continue
        if k in aliases:
            continue
        if k in ("b_p", "b_n"):
            # Statistic keywords (median, q1, …) are resolved per specimen in ``build_initial_rows``;
            # only numeric literals update the shared seed dict (e.g. generalized settings CSV).
            spec = _parse_b_spec(row[k], label=f"{_row_set_id_label(row)} {k}")
            if not isinstance(spec, float):
                continue
            resolved[k] = float(spec)
            continue
        resolved[k] = float(row[k])

    pending = dict(aliases)
    for _ in range(len(pending) + len(allow) + 4):
        if not pending:
            break
        progressed = False
        for slave, master in list(pending.items()):
            if master in resolved:
                resolved[slave] = float(resolved[master])
                del pending[slave]
                progressed = True
        if not progressed:
            raise ValueError(
                f"{_row_set_id_label(row)}: unresolved seed parameter alias(es) {pending}; "
                f"ensure each master column is numeric or resolvable."
            )

    return {k: float(resolved[k]) for k in NUMERIC_SEED_KEYS}


def _explicit_numeric_overrides_only(row: pd.Series, steel_model: str) -> dict[str, float]:
    """
    Return only explicitly-provided numeric seed cells from a settings row.

    Used when a set_id inherits a full parameter vector (e.g. from an optimized CSV) and wants
    to override only a few columns via ``set_id_settings.csv``.
    """
    sm = normalize_steel_model(steel_model)
    allow = _override_keys_for_steel_model(sm)
    aliases = parse_param_alias_bindings_from_row(row, sm)
    out: dict[str, float] = {}
    for k in allow:
        if k not in row.index:
            continue
        v = row[k]
        if _is_missing_sentinel(v):
            continue
        if k in aliases:
            # Alias bindings are applied after a base dict exists.
            continue
        if k in ("b_p", "b_n"):
            spec = _parse_b_spec(v, label=f"{_row_set_id_label(row)} {k}")
            if isinstance(spec, float) and np.isfinite(float(spec)):
                out[k] = float(spec)
            continue
        if isinstance(v, (bool, np.bool_)):
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(float(fv)):
            out[k] = float(fv)
    return out


def _full_numeric_seed(steel_model: str, row: pd.Series) -> dict[str, float]:
    return _resolve_numeric_seed_overrides(row, steel_model)


def parse_b_p_n_spec(raw: object, *, label: str) -> float | str:
    """Parse ``b_p`` / ``b_n`` cell: numeric literal or apparent-``b`` statistic keyword."""
    if _is_missing_sentinel(raw):
        return "median"
    if raw is None:
        return "median"
    if isinstance(raw, (float, np.floating)) and np.isnan(float(raw)):
        return "median"
    if isinstance(raw, (int, float, np.integer, np.floating)) and not isinstance(raw, bool):
        v = float(raw)
        if np.isfinite(v):
            return v
    if pd.isna(raw):
        return "median"
    s = str(raw).strip()
    if not s or s.lower() in ("nan", "none", ""):
        return "median"
    try:
        v = float(s)
        if np.isfinite(v):
            return v
    except ValueError:
        pass
    key = s.strip().lower().replace(" ", "_")
    aliases = {
        "1st_quartile": "q1",
        "first_quartile": "q1",
        "3rd_quartile": "q3",
        "third_quartile": "q3",
    }
    key = aliases.get(key, key)
    if key in B_STAT_NAMES:
        return key
    raise ValueError(f"{label}: expected a number or stat in {sorted(B_STAT_NAMES)}, got {raw!r}")


def _parse_b_spec(raw: object, *, label: str) -> float | str:
    return parse_b_p_n_spec(raw, label=label)


def _finite_scalar(x: float) -> bool:
    return isinstance(x, (int, float)) and bool(np.isfinite(float(x)))


def _get_bn_col(row: pd.Series, col: str) -> float:
    if col not in row.index:
        return float("nan")
    v = row[col]
    if pd.isna(v):
        return float("nan")
    return float(v)


def resolve_b_arm_from_stats(
    stats: dict[str, float] | pd.Series,
    *,
    arm: str,
    spec: float | str,
    fallback: float | None = None,
) -> float:
    """
    Resolve one ``b_p`` / ``b_n`` seed from a numeric literal or statistic keyword.

    ``stats`` must expose ``b_{p|n}_{median,mean,...}`` keys (as in ``extract_bn_bp_one_specimen`` or
    ``specimen_apparent_bn_bp.csv``). Non-numeric ``spec`` uses the same fallback chain as
    ``set_id_settings.csv``; ``fallback`` overrides ``STEEL_DEFAULT`` when every stat is missing.
    """
    if arm not in ("p", "n"):
        raise ValueError("arm must be 'p' or 'n'")
    if isinstance(spec, float):
        return float(spec)

    row = stats if isinstance(stats, pd.Series) else pd.Series(stats)
    med = _get_bn_col(row, f"b_{arm}_median")
    mean = _get_bn_col(row, f"b_{arm}_mean")
    wmean = _get_bn_col(row, f"b_{arm}_weighted_mean")
    b_max_amp = _get_bn_col(row, f"b_{arm}_max_amplitude")
    q1 = _get_bn_col(row, f"b_{arm}_q1")
    q3 = _get_bn_col(row, f"b_{arm}_q3")
    vmin = _get_bn_col(row, f"b_{arm}_min")
    vmax = _get_bn_col(row, f"b_{arm}_max")
    dflt = float(
        fallback
        if fallback is not None
        else (STEEL_DEFAULT["b_p"] if arm == "p" else STEEL_DEFAULT["b_n"])
    )

    stat = spec
    if stat == "median":
        if _finite_scalar(med):
            return float(med)
        if _finite_scalar(mean):
            return float(mean)
        return dflt
    if stat == "mean":
        if _finite_scalar(mean):
            return float(mean)
        return dflt
    if stat == "weighted_mean":
        if _finite_scalar(wmean):
            return float(wmean)
        if _finite_scalar(mean):
            return float(mean)
        if _finite_scalar(med):
            return float(med)
        return dflt
    if stat == "max_amplitude":
        if _finite_scalar(b_max_amp):
            return float(b_max_amp)
        if _finite_scalar(wmean):
            return float(wmean)
        if _finite_scalar(mean):
            return float(mean)
        if _finite_scalar(med):
            return float(med)
        return dflt
    if stat == "q1":
        if _finite_scalar(q1):
            return float(q1)
        if _finite_scalar(med):
            return float(med)
        if _finite_scalar(mean):
            return float(mean)
        return dflt
    if stat == "q3":
        if _finite_scalar(q3):
            return float(q3)
        if _finite_scalar(med):
            return float(med)
        if _finite_scalar(mean):
            return float(mean)
        return dflt
    if stat == "min":
        if _finite_scalar(vmin):
            return float(vmin)
        if _finite_scalar(med):
            return float(med)
        if _finite_scalar(mean):
            return float(mean)
        return dflt
    if stat == "max":
        if _finite_scalar(vmax):
            return float(vmax)
        if _finite_scalar(med):
            return float(med)
        if _finite_scalar(mean):
            return float(mean)
        return dflt
    raise ValueError(f"unknown b stat {stat!r}")


def _resolve_b_arm(row: pd.Series, *, arm: str, spec: float | str) -> float:
    return resolve_b_arm_from_stats(row, arm=arm, spec=spec)


def _ensure_bn_bp_stat_columns(bn_bp: pd.DataFrame) -> pd.DataFrame:
    b = bn_bp.copy()
    for c in _BN_BP_STAT_COLUMNS:
        if c not in b.columns:
            b[c] = np.nan
    return b


def _load_seeds(df: pd.DataFrame, path: Path) -> list[InitialBrbSeedRow]:
    df = df.copy()
    if "set_id" not in df.columns:
        raise ValueError(f"{path}: seed CSV must have column 'set_id' (one row per calibration set)")
    if "steel_model" not in df.columns:
        df["steel_model"] = np.nan
    for col in NUMERIC_SEED_KEYS:
        if col not in df.columns:
            df[col] = np.nan
    for col in ("b_p", "b_n"):
        if col not in df.columns:
            df[col] = np.nan
    # Back-compat: older name was init_from_set_id; preferred is inherit_from_set.
    for col in ("inherit_from_set", "init_from_set_id", "init_from_csv"):
        if col not in df.columns:
            df[col] = np.nan
    seen: set[int] = set()
    rows: list[InitialBrbSeedRow] = []
    for i, row in df.iterrows():
        sid = row["set_id"]
        if pd.isna(sid):
            raise ValueError(f"{path}: row {i + 2}: set_id is required")
        try:
            set_id = int(pd.to_numeric(sid, errors="raise"))
        except (ValueError, TypeError) as e:
            raise ValueError(f"{path}: row {i + 2}: set_id must be an integer") from e
        if set_id < 1:
            raise ValueError(f"{path}: row {i + 2}: set_id must be >= 1")
        if set_id in seen:
            raise ValueError(f"{path}: duplicate set_id {set_id}")
        seen.add(set_id)
        sm = normalize_steel_model(row.get("steel_model"))
        steel_seed = _full_numeric_seed(sm, row)
        steel_overrides = _explicit_numeric_overrides_only(row, sm)
        steel_ties = parse_param_alias_bindings_from_row(row, sm)
        bp = _parse_b_spec(row.get("b_p"), label=f"{path} row {i + 2} b_p")
        bn = _parse_b_spec(row.get("b_n"), label=f"{path} row {i + 2} b_n")
        init_sid: int | None = None
        inherit_cell = row.get("inherit_from_set")
        legacy_cell = row.get("init_from_set_id")
        chosen = inherit_cell if not _is_missing_sentinel(inherit_cell) else legacy_cell
        if not _is_missing_sentinel(chosen):
            try:
                init_sid = int(pd.to_numeric(chosen, errors="raise"))
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"{path}: row {i + 2}: inherit_from_set must be an integer or -999/blank"
                ) from e
            if init_sid < 1:
                raise ValueError(f"{path}: row {i + 2}: inherit_from_set must be >= 1")
            if init_sid == set_id:
                raise ValueError(
                    f"{path}: row {i + 2}: inherit_from_set cannot equal set_id ({set_id})"
                )

        init_csv: Path | None = None
        if not _is_missing_sentinel(row.get("init_from_csv")):
            s = str(row.get("init_from_csv")).strip()
            if s and s.lower() not in ("nan", "none"):
                init_csv = Path(s).expanduser().resolve()
        if init_sid is not None and init_csv is None:
            init_csv = OPTIMIZED_BRB_PARAMETERS_PATH

        rows.append(
            InitialBrbSeedRow(
                set_id=set_id,
                steel_model=sm,
                steel_seed=steel_seed,
                steel_overrides=steel_overrides,
                steel_ties=steel_ties,
                b_p_spec=bp,
                b_n_spec=bn,
                init_from_set_id=init_sid,
                init_from_csv=init_csv,
            )
        )
    rows.sort(key=lambda r: r.set_id)
    return rows


def load_initial_brb_seeds(path: Path) -> list[InitialBrbSeedRow]:
    """Parse ``set_id_settings.csv`` into one ``InitialBrbSeedRow`` per calibration ``set_id``."""
    df = load_set_id_settings(path)
    return _load_seeds(df, path)


def _numeric_bn_seed_for_generalized(
    raw: object,
    *,
    col: str,
    settings_path: Path,
    set_id: int,
) -> float:
    """
    ``b_p`` / ``b_n`` cells in ``set_id_settings_generalized.csv`` must be finite numbers.

    Statistic keywords (``median``, ``q1``, …) apply per specimen in ``set_id_settings.csv`` for
    individual calibration only; generalized seeds are one shared backbone per ``set_id``.
    """
    label = f"{settings_path} set_id={set_id} {col}"
    spec = _parse_b_spec(raw, label=label)
    if not isinstance(spec, float):
        raise ValueError(
            f"{label}: generalized settings require a numeric {col} seed (got {spec!r}). "
            "Statistic keywords belong in set_id_settings.csv for individual runs."
        )
    if not np.isfinite(float(spec)):
        raise ValueError(f"{label}: expected a finite float; got {raw!r}")
    return float(spec)


def generalized_init_param_series_for_set_id(
    set_id: int,
    settings_path: Path,
    active_params: list[str],
) -> pd.Series:
    """
    Initial generalized iterate (subset ``active_params``) from **only** the generalized settings CSV
    (``set_id_settings_generalized.csv`` by default): numeric steel seeds via ``_full_numeric_seed``, and
    numeric ``b_p`` / ``b_n`` when those columns are needed for initialization. Does not read
    ``set_id_settings.csv``. The ``set_id`` argument is the row key in that generalized CSV (not individual
    calibration ``set_id`` on ``--params``).
    """
    df = load_set_id_settings(settings_path)
    m = df["set_id"] == int(set_id)
    if not m.any():
        raise KeyError(f"{settings_path}: no row for set_id={set_id}")
    row = df.loc[m].iloc[0]
    sm = normalize_steel_model(row.get("steel_model"))

    # Optional generalized-level inheritance: initialize from a previously optimized generalized set_id.
    inherit_cell = row.get("inherit_from_set") if "inherit_from_set" in row.index else None
    legacy_cell = row.get("init_from_set_id") if "init_from_set_id" in row.index else None
    chosen = inherit_cell if not _is_missing_sentinel(inherit_cell) else legacy_cell
    inherit_sid: int | None = None
    if not _is_missing_sentinel(chosen):
        try:
            inherit_sid = int(pd.to_numeric(chosen, errors="raise"))
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"{settings_path}: set_id={set_id}: inherit_from_set must be an integer or -999/blank"
            ) from e
        if inherit_sid < 1:
            raise ValueError(f"{settings_path}: set_id={set_id}: inherit_from_set must be >= 1")
        if inherit_sid == int(set_id):
            raise ValueError(f"{settings_path}: set_id={set_id}: inherit_from_set cannot equal set_id")

    inherited: dict[str, float] = {}
    if inherit_sid is not None:
        gpath = GENERALIZED_BRB_PARAMETERS_PATH
        if gpath.is_file():
            # Optional pre-seed from a previously written generalized CSV (warm-starts the L-BFGS init
            # vector reported in the optimizer banner). The actual parent->child inheritance for this
            # run is applied intra-run by ``optimize_generalized_brb_mse.py`` after each parent set_id
            # finishes, so a missing CSV here is silently fine on a fresh pipeline run.
            gdf = pd.read_csv(gpath, comment="#", skipinitialspace=True)
            gdf.columns = gdf.columns.astype(str).str.strip()
            if {"set_id", "steel_model"}.issubset(set(gdf.columns)):
                m_base = pd.to_numeric(gdf["set_id"], errors="coerce").astype(int) == int(inherit_sid)
                if bool(m_base.any()):
                    m_exact = m_base & (gdf["steel_model"].astype(str) == str(sm))
                    grow = gdf.loc[m_exact].iloc[0] if bool(m_exact.any()) else gdf.loc[m_base].iloc[0]
                    allow = _override_keys_for_steel_model(sm)
                    for k in allow:
                        if k in grow.index and pd.notna(grow.get(k)):
                            try:
                                v = float(grow.get(k))
                            except (TypeError, ValueError):
                                continue
                            if np.isfinite(v):
                                inherited[k] = float(v)
            if not inherited:
                # CSV exists but lacks a usable row for the requested inherit target -> real misconfig.
                print(
                    f"Warning: generalized set_id={set_id}: inherit_from_set={inherit_sid} requested, "
                    f"but no usable row found in {gpath.name}; will rely on intra-run inheritance "
                    "from the parent's just-optimized values (and on settings seeds where missing)."
                )
        # Missing-file case is intentionally silent: intra-run inheritance in
        # optimize_generalized_brb_mse.py will overlay the parent's optimum onto this child's init
        # vector and settings row before the L-BFGS call, so the pre-seed is not required.

    # Base numeric seeds from the generalized settings row itself (current behavior).
    steel = _full_numeric_seed(sm, row)
    explicit = _explicit_numeric_overrides_only(row, sm)

    vals: dict[str, float] = {}
    for p in active_params:
        # Explicit numeric cells win first.
        if p in explicit:
            vals[p] = float(explicit[p])
            continue
        # Then inherited generalized params (if any).
        if p in inherited:
            vals[p] = float(inherited[p])
            continue
        # Then the row's seed dict.
        if p in steel:
            vals[p] = float(steel[p])
            continue
        if p == "b_p":
            if "b_p" in inherited:
                vals["b_p"] = float(inherited["b_p"])
            else:
                vals["b_p"] = _numeric_bn_seed_for_generalized(
                    row.get("b_p"), col="b_p", settings_path=settings_path, set_id=set_id
                )
            continue
        if p == "b_n":
            if "b_n" in inherited:
                vals["b_n"] = float(inherited["b_n"])
            else:
                vals["b_n"] = _numeric_bn_seed_for_generalized(
                    row.get("b_n"), col="b_n", settings_path=settings_path, set_id=set_id
                )
            continue
        if p in row.index and not _is_missing_sentinel(row.get(p)):
            try:
                vals[p] = float(row[p])
            except (TypeError, ValueError):
                pass
    missing = [p for p in active_params if p not in vals]
    if missing:
        raise ValueError(
            f"{settings_path} set_id={set_id}: could not resolve generalized init values for {missing!r}. "
            "Supply numeric seeds in the generalized settings row."
        )
    return pd.Series(vals)


OUT_COLS = [
    "ID",
    "Name",
    "set_id",
    "steel_model",
    "L_T",
    "L_y",
    "A_sc",
    "A_t",
    "fyp",
    "fyn",
    "E",
    "b_p",
    "b_n",
    "R0",
    "cR1",
    "cR2",
    "a1",
    "a2",
    "a3",
    "a4",
]


def build_initial_rows(
    catalog: pd.DataFrame,
    bn_bp: pd.DataFrame,
    *,
    seeds: list[InitialBrbSeedRow],
) -> list[dict]:
    """Assemble initial_brb_parameters rows from catalog and seeds."""
    _validate_steel_default(STEEL_DEFAULT)
    if not seeds:
        raise ValueError("initial BRB seeds list is empty; add rows to set_id_settings.csv")

    # Lazy cache: optimized CSV path -> DataFrame (or None if file is missing).
    # The pre-seed via this CSV is only a warm start for the rows written into
    # ``initial_brb_parameters.csv``; the real per-set inheritance is applied intra-run by
    # ``optimize_brb_mse.py`` (parent's just-optimized values overlaid onto each child set_id),
    # so a missing source CSV is silently treated as "no warm start available".
    opt_cache: dict[Path, pd.DataFrame | None] = {}

    def _load_opt(path: Path) -> pd.DataFrame | None:
        p = Path(path).expanduser().resolve()
        if p not in opt_cache:
            if not p.is_file():
                opt_cache[p] = None
            else:
                df = pd.read_csv(p, comment="#", skipinitialspace=True)
                df.columns = df.columns.astype(str).str.strip()
                need = ("Name", "set_id", "steel_model")
                missing = [c for c in need if c not in df.columns]
                if missing:
                    raise ValueError(f"{p}: missing required columns {missing}")
                opt_cache[p] = df
        return opt_cache[p]

    def _inherit_steel_from_optimized(
        *,
        name: str,
        steel_model: str,
        init_from_set_id: int,
        init_from_csv: Path,
    ) -> dict[str, float] | None:
        df = _load_opt(init_from_csv)
        if df is None:
            return None
        # Prefer exact steel_model match, but allow cross-model inheritance by taking only overlapping keys.
        m_base = (df["Name"].astype(str) == str(name)) & (
            pd.to_numeric(df["set_id"], errors="coerce").astype(int) == int(init_from_set_id)
        )
        if not bool(m_base.any()):
            return None
        sm = str(steel_model)
        m_exact = m_base & (df["steel_model"].astype(str) == sm)
        if bool(m_exact.any()):
            row = df.loc[m_exact].iloc[0]
        else:
            # Fallback: pick the first available material row for that (Name, set_id).
            row = df.loc[m_base].iloc[0]

        allow = _override_keys_for_steel_model(sm)
        out: dict[str, float] = {}
        # Pull only keys valid for the *target* model (intersection with optimized CSV columns).
        for k in allow:
            if k in row.index and pd.notna(row.get(k)):
                try:
                    v = float(row.get(k))
                except (TypeError, ValueError):
                    continue
                if np.isfinite(v):
                    out[k] = float(v)
        return out
    cat = catalog.copy()
    if "Name" not in cat.columns:
        raise ValueError("Catalog must have a Name column")
    cat["Name"] = cat["Name"].astype(str)

    bfull = _ensure_bn_bp_stat_columns(bn_bp)
    need_bn = ["Name", *_BN_BP_STAT_COLUMNS]
    missing = [c for c in need_bn if c not in bfull.columns]
    if missing:
        raise ValueError(f"bn_bp CSV missing columns: {missing}")
    bn = bfull[need_bn].copy()
    bn["Name"] = bn["Name"].astype(str)

    merged = cat.merge(bn, on="Name", how="left")
    if merged.empty:
        return []

    merged = merged.sort_values("ID" if "ID" in merged.columns else "Name")

    rows_out: list[dict] = []
    for _, r in merged.iterrows():
        name = str(r["Name"])
        cid = int(r["ID"])
        L_T = float(r["L_T"])
        L_y = float(r["L_y"])
        A_sc = float(r["A_sc"])
        A_t = float(r["A_t"])
        fy = float(r["fyp"])

        for seed in seeds:
            inherited: dict[str, float] | None = None
            if seed.init_from_set_id is not None:
                inherit_csv = (
                    Path(seed.init_from_csv)
                    if seed.init_from_csv is not None
                    else OPTIMIZED_BRB_PARAMETERS_PATH
                )
                inherit_csv_resolved = inherit_csv.expanduser().resolve()
                inherited = _inherit_steel_from_optimized(
                    name=name,
                    steel_model=seed.steel_model,
                    init_from_set_id=int(seed.init_from_set_id),
                    init_from_csv=inherit_csv,
                )
                # Per-specimen warning only when the CSV exists but lacks a matching row (real
                # misconfig). Missing-file case is intentionally silent: the parent->child overlay
                # is applied intra-run by ``optimize_brb_mse.py`` after each parent set finishes.
                if inherited is None and inherit_csv_resolved.is_file():
                    print(
                        f"Warning: {name!r} set_id={seed.set_id}: no row found in "
                        f"{inherit_csv.name} for "
                        f"init_from_set_id={seed.init_from_set_id}, steel_model={seed.steel_model}; "
                        "will rely on intra-run inheritance from the parent's just-optimized values "
                        "(and on settings seeds where missing)."
                    )

            # Always start from seed.steel_seed so all NUMERIC_SEED_KEYS are present. Then layer:
            # inherited (subset for target model) -> explicit overrides -> alias ties.
            steel = dict(seed.steel_seed)
            if inherited is not None:
                steel.update(inherited)
            steel.update(seed.steel_overrides)
            for slave, master in seed.steel_ties.items():
                if master in steel:
                    steel[slave] = float(steel[master])

            if inherited is not None and "b_p" in inherited and not isinstance(seed.b_p_spec, float):
                b_p = float(inherited["b_p"])
            else:
                b_p = _resolve_b_arm(r, arm="p", spec=seed.b_p_spec)
            if inherited is not None and "b_n" in inherited and not isinstance(seed.b_n_spec, float):
                b_n = float(inherited["b_n"])
            else:
                b_n = _resolve_b_arm(r, arm="n", spec=seed.b_n_spec)
            row_d: dict[str, object] = {
                "ID": cid,
                "Name": name,
                "set_id": seed.set_id,
                "steel_model": seed.steel_model,
                "L_T": L_T,
                "L_y": L_y,
                "A_sc": A_sc,
                "A_t": A_t,
                "fyp": fy,
                "fyn": fy,
                "E": steel["E"],
                "b_p": b_p,
                "b_n": b_n,
                "R0": steel["R0"],
                "cR1": steel["cR1"],
                "cR2": steel["cR2"],
                "a1": steel["a1"],
                "a2": steel["a2"],
                "a3": steel["a3"],
                "a4": steel["a4"],
            }
            rows_out.append(row_d)
    return rows_out


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description="Build initial_brb_parameters.csv from catalog + extract_bn_bp output.")
    p.add_argument("--catalog", type=Path, default=CATALOG_PATH, help="BRB-Specimens.csv")
    p.add_argument(
        "--bn-bp",
        type=Path,
        default=DEFAULT_BN_BP_PATH,
        help="Output CSV from extract_bn_bp.py",
    )
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="initial_brb_parameters.csv path")
    p.add_argument(
        "--set-id-settings",
        type=Path,
        default=SET_ID_SETTINGS_CSV,
        help=(
            "set_id_settings.csv: one row per set_id (steel_model, E, b_p/b_n, kinematic + model-specific seeds). "
            "Default: config/calibration/set_id_settings.csv."
        ),
    )
    args = p.parse_args()

    catalog = read_catalog(Path(args.catalog).expanduser().resolve())
    if not args.bn_bp.is_file():
        print(f"Missing {args.bn_bp}; run extract_bn_bp.py after resample_filtered.py.")
        sys.exit(1)
    bn_bp = pd.read_csv(args.bn_bp)

    seed_path = Path(args.set_id_settings).expanduser().resolve()
    if not seed_path.is_file():
        print(f"Steel seed CSV not found: {seed_path}", file=sys.stderr)
        sys.exit(1)
    try:
        seeds = load_initial_brb_seeds(seed_path)
    except ValueError as e:
        print(e)
        sys.exit(1)

    try:
        rows = build_initial_rows(catalog, bn_bp, seeds=seeds)
    except ValueError as e:
        print(e)
        sys.exit(1)
    if not rows:
        print("No rows after catalog merge; check BRB-Specimens.csv.")
        sys.exit(1)

    out = pd.DataFrame(rows)
    out = out[OUT_COLS]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    n_sp = out["Name"].nunique()
    n_sets = len(seeds)
    print(
        f"Wrote {args.output} ({len(out)} rows, {n_sp} specimens × {n_sets} set_ids). "
        f"Steel/b seeds: {seed_path}."
    )


if __name__ == "__main__":
    main()
