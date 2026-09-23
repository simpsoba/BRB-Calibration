"""
Per-``set_id`` lists of parameters optimized by L-BFGS-B (individual / generalized stages).

``optimize_params`` cells split on **commas** first; each segment is normalized (lowercase; underscores
and spaces removed) as a **whole**, so ``c r 1`` and ``c_r_1`` map to ``cR1``. If the whole segment
does not match, the segment is split on whitespace and each piece is normalized (so ``R0 cR1`` in one
segment still works). Separate parameters should use commas when names contain spaces, e.g.
``R0, c r 1, a1``. Resolved lists use canonical names matching ``params_to_optimize.SIM_PARAMS_FROM_ROW``.
Allowed tokens are SteelMPF simulation parameters (see ``OPTIMIZABLE_SIM_PARAM_NAMES``).
"""
from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from calibrate.calibration_loss_settings import (
    CalibrationLossSettings,
    DEFAULT_CALIBRATION_LOSS_SETTINGS,
    calibration_loss_settings_from_partial_dict,
)
from calibrate.steel_model import (
    STEEL_MODEL_STEELMPF,
    normalize_steel_model,
)

# Keys accepted in optimize_params cells (SteelMPF).
OPTIMIZABLE_SIM_PARAM_NAMES: frozenset[str] = frozenset(
    (
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
    )
)


def optimizable_names_for_steel_model(_steel_model: object = None) -> frozenset[str]:
    """Parameters allowed in ``optimize_params`` (SteelMPF only)."""
    normalize_steel_model(_steel_model)
    return OPTIMIZABLE_SIM_PARAM_NAMES


def _normalize_optimize_token_key(token: str) -> str:
    """Lowercase; remove underscores and whitespace for flexible CSV spelling."""
    return re.sub(r"[\s_]+", "", str(token).lower())


_CANONICAL_BY_NORMALIZED_KEY: dict[str, str] = {
    _normalize_optimize_token_key(name): name for name in OPTIMIZABLE_SIM_PARAM_NAMES
}


def _canonical_list_from_segment(segment: str) -> tuple[list[str], list[str]]:
    """
    Map one comma-separated segment to canonical names.

    First tries the whole segment with spaces/underscores stripped (so ``c r 1`` -> ``cR1``).
    If that fails, splits on whitespace and maps each piece (so ``R0 cR1`` still works).
    """
    seg = segment.strip()
    if not seg:
        return [], []
    key_whole = _normalize_optimize_token_key(seg)
    if key_whole:
        c = _CANONICAL_BY_NORMALIZED_KEY.get(key_whole)
        if c is not None:
            return [c], []
    out: list[str] = []
    bad: list[str] = []
    for p in seg.split():
        key = _normalize_optimize_token_key(p)
        if not key:
            bad.append(p)
            continue
        c = _CANONICAL_BY_NORMALIZED_KEY.get(key)
        if c is None:
            bad.append(p)
        else:
            out.append(c)
    return out, bad


def _parse_optimize_params_cell(
    raw: object, *, path: Path, set_id: int, steel_model: str = STEEL_MODEL_STEELMPF
) -> list[str]:
    """Split on commas; per segment, normalize whole string or whitespace-separated pieces."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        raise ValueError(f"{path}: set_id={set_id}: empty optimize_params")
    s = str(raw).strip()
    if not s or s.lower() == "nan":
        raise ValueError(f"{path}: set_id={set_id}: empty optimize_params")
    segments = [x.strip() for x in s.split(",") if x.strip()]
    if not segments:
        raise ValueError(f"{path}: set_id={set_id}: no parameter tokens in optimize_params")
    canonical: list[str] = []
    bad: list[str] = []
    for seg in segments:
        got, seg_bad = _canonical_list_from_segment(seg)
        if seg_bad:
            bad.extend(seg_bad)
        else:
            canonical.extend(got)
    if bad:
        raise ValueError(
            f"{path}: set_id={set_id}: unknown parameter name(s) {bad}; "
            f"allowed (any spelling that normalizes to these) {sorted(OPTIMIZABLE_SIM_PARAM_NAMES)}"
        )
    # Preserve order but drop duplicates (by canonical name)
    seen: set[str] = set()
    out: list[str] = []
    for c in canonical:
        if c not in seen:
            seen.add(c)
            out.append(c)
    sm = normalize_steel_model(steel_model)
    allowed = optimizable_names_for_steel_model(sm)
    disallowed = [c for c in out if c not in allowed]
    if disallowed:
        raise ValueError(
            f"{path}: set_id={set_id} steel_model={sm!r}: parameter(s) {disallowed} not allowed "
            f"for this material; allowed: {sorted(allowed)}"
        )
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


def resolve_optimize_params_for_set_id(
    mapping: dict[int, list[str]],
    set_id: object,
    default: list[str],
) -> list[str]:
    """Return ``mapping[int(set_id)]`` if present, else ``default``. Empty ``mapping`` always uses default."""
    if not mapping:
        return list(default)
    try:
        sid = int(pd.to_numeric(set_id, errors="raise"))
    except (ValueError, TypeError):
        return list(default)
    return list(mapping.get(sid, default))


def resolve_loss_settings_for_set_id(
    mapping: dict[int, CalibrationLossSettings],
    set_id: object,
    default: CalibrationLossSettings = DEFAULT_CALIBRATION_LOSS_SETTINGS,
) -> CalibrationLossSettings:
    """Return `mapping[int(set_id)]` if present, else `default`. Empty `mapping` always uses default."""
    if not mapping:
        return default
    try:
        sid = int(pd.to_numeric(set_id, errors="raise"))
    except (ValueError, TypeError):
        return default
    return mapping.get(sid, default)


def build_param_cols_by_set_id_from_mapping(
    mapping: dict[int, list[str]],
    default: list[str],
    *,
    set_ids: list[object],
) -> dict[int, list[str]]:
    """``set_id -> param list`` for every ``set_id`` in ``set_ids`` (int keys)."""
    out: dict[int, list[str]] = {}
    for sid_raw in set_ids:
        try:
            sid = int(pd.to_numeric(sid_raw, errors="raise"))
        except (ValueError, TypeError):
            continue
        out[sid] = resolve_optimize_params_for_set_id(mapping, sid, default)
    return out


def unique_weighted_train_set_ids(
    params_df: pd.DataFrame,
    weight_fn: Callable[[str], float],
    *,
    set_id_column: str = "set_id",
) -> list[int]:
    """Distinct numeric ids in ``set_id_column`` among rows with positive ``weight_fn(Name)``."""
    if set_id_column not in params_df.columns:
        return []
    w_series = params_df["Name"].astype(str).map(weight_fn)
    train = params_df.loc[w_series > 0.0]
    out: set[int] = set()
    for s in train[set_id_column].unique():
        if pd.isna(s):
            continue
        try:
            out.add(int(pd.to_numeric(s, errors="raise")))
        except (ValueError, TypeError):
            continue
    return sorted(out)


def union_param_cols(default: list[str], param_cols_by_set_id: dict[int, list[str]] | None) -> list[str]:
    """Ordered union of ``default`` and all lists in ``param_cols_by_set_id`` (for CSV column checks)."""
    seen: set[str] = set()
    out: list[str] = []
    for c in default:
        if c not in seen:
            seen.add(c)
            out.append(c)
    if param_cols_by_set_id:
        for lst in param_cols_by_set_id.values():
            for c in lst:
                if c not in seen:
                    seen.add(c)
                    out.append(c)
    return out


def assert_global_optimize_params_consistent(
    mapping: dict[int, list[str]],
    train_set_ids: list[object],
    default: list[str],
) -> list[str]:
    """
    For pooled / global mode: every training ``set_id`` must resolve to the same optimize list.

    Raises ``SystemExit`` if the mapping yields more than one distinct list. Returns that list
    (or ``default`` when ``mapping`` is empty or ``train_set_ids`` is empty).
    """
    if not mapping or not train_set_ids:
        return list(default)
    resolved: dict[int, tuple[str, ...]] = {}
    for sid_raw in set(train_set_ids):
        try:
            sid = int(pd.to_numeric(sid_raw, errors="raise"))
        except (ValueError, TypeError):
            continue
        resolved[sid] = tuple(resolve_optimize_params_for_set_id(mapping, sid, default))
    uniq = set(resolved.values())
    if len(uniq) > 1:
        pretty = {str(k): list(v) for k, v in sorted(resolved.items())}
        raise SystemExit(
            "Global (pooled) mode requires identical optimize_params for every training cohort id "
            f"when a generalized settings CSV is present; resolved lists differ: {pretty}"
        )
    return list(next(iter(uniq))) if uniq else list(default)


def assert_global_loss_settings_consistent(
    mapping: dict[int, CalibrationLossSettings],
    train_set_ids: list[object],
    default: CalibrationLossSettings = DEFAULT_CALIBRATION_LOSS_SETTINGS,
) -> CalibrationLossSettings:
    """
    For pooled / global mode: every training `set_id` must resolve to identical loss settings.

    Raises `SystemExit` if the mapping yields more than one distinct setting object.
    """
    if not mapping or not train_set_ids:
        return default
    resolved: dict[int, CalibrationLossSettings] = {}
    for sid_raw in set(train_set_ids):
        try:
            sid = int(pd.to_numeric(sid_raw, errors="raise"))
        except (ValueError, TypeError):
            continue
        resolved[sid] = resolve_loss_settings_for_set_id(mapping, sid, default)
    uniq = set(resolved.values())
    if len(uniq) > 1:
        pretty = {str(k): v for k, v in sorted(resolved.items())}
        raise SystemExit(
            "Global (pooled) mode requires identical loss settings for every training cohort id "
            f"when a generalized settings CSV is present; resolved settings differ: {pretty}"
        )
    return next(iter(uniq)) if uniq else default


def assert_global_steel_model_consistent(
    steel_model_by_set_id: dict[int, str],
    train_set_ids: list[object],
) -> str:
    """
    For pooled / global mode: every training ``set_id`` must use the same ``steel_model``.

    Returns the common model (or ``steelmpf`` when mapping or train list is empty).
    """
    if not steel_model_by_set_id or not train_set_ids:
        return STEEL_MODEL_STEELMPF
    resolved: dict[int, str] = {}
    for sid_raw in set(train_set_ids):
        try:
            sid = int(pd.to_numeric(sid_raw, errors="raise"))
        except (ValueError, TypeError):
            continue
        if sid not in steel_model_by_set_id:
            raise SystemExit(
                f"Global (pooled) mode: cohort id={sid} missing from generalized settings steel_model map "
                f"(have cohort ids {sorted(steel_model_by_set_id.keys())})"
            )
        resolved[sid] = steel_model_by_set_id[sid]
    uniq = set(resolved.values())
    if len(uniq) > 1:
        pretty = {str(k): v for k, v in sorted(resolved.items())}
        raise SystemExit(
            "Global (pooled) mode requires the same steel_model for every training cohort id when "
            f"a generalized settings CSV is present; got: {pretty}"
        )
    return next(iter(uniq)) if uniq else STEEL_MODEL_STEELMPF
