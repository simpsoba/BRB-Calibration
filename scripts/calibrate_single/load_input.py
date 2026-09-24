"""Load ``input.csv`` for ``calibrate_one_specimen.py`` (SteelMPF only).

Rows are ``section,key,<values>`` where ``<values>`` is one token or a comma-separated list
(CSV-aware: quoted fields may contain commas) with **one entry per ``meta.set_id``**. A single
value is broadcast to every set_id. Blank entries inherit the previous set_id's value on the same row.

``steel.b_p`` / ``steel.b_n`` accept a numeric literal or an apparent-``b`` statistic keyword
(``median``, ``q1``, … — same as ``set_id_settings.csv``). If a keyword is used but no apparent
stats exist for the specimen, ``calibrate_one_specimen.py`` warns and uses ``FALLBACK_B_P`` /
``FALLBACK_B_N`` (0.005 / 0.025).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

from calibrate.build_initial_brb_parameters import parse_b_p_n_spec  # noqa: E402
from calibrate.calibration_loss_settings import (  # noqa: E402
    CalibrationLossSettings,
    parse_bool_cell,
)
from calibrate.steel_model import STEEL_MODEL_STEELMPF  # noqa: E402

STEEL_MODEL = STEEL_MODEL_STEELMPF

# SteelMPF numeric seed columns in input.csv (model through a4; no ultimate-strength tail).
STEELMPF_NUMERIC_SEED_KEYS = (
    "E",
    "R0",
    "cR1",
    "cR2",
    "a1",
    "a2",
    "a3",
    "a4",
)
STEELMPF_B_SPEC_KEYS = ("b_p", "b_n")

# When ``steel.b_p`` / ``b_n`` is a stat keyword and no apparent-``b`` stats exist for the specimen.
FALLBACK_B_P = 0.005
FALLBACK_B_N = 0.025


@dataclass(frozen=True)
class SingleCalibrateInput:
    set_id: int
    optimize_params: list[str]
    b_p_spec: float | str
    b_n_spec: float | str
    steel_seeds: dict[str, float]
    loss: CalibrationLossSettings
    param_bounds: dict[str, tuple[float, float]]


def _parse_optimize_params(raw: object) -> list[str]:
    s = str(raw).strip().strip('"').strip("'")
    if not s:
        raise ValueError("meta.optimize_params is empty")
    return [p.strip() for p in s.split(",") if p.strip()]


def _parse_bound(raw: object, *, param: str) -> tuple[float, float]:
    parts = [x.strip() for x in str(raw).split(",")]
    if len(parts) != 2:
        raise ValueError(
            f"bound.{param}: expected 'lower,upper', got {raw!r}"
        )
    lo, hi = float(parts[0]), float(parts[1])
    if not (hi > lo):
        raise ValueError(f"bound.{param}: upper must exceed lower (got {lo}, {hi})")
    return lo, hi


def _split_csv_value_field(value_blob: str) -> list[str]:
    """Split the third CSV column into per-set_id tokens (respects quoted commas)."""
    blob = value_blob.strip()
    if not blob:
        return [""]
    return next(csv.reader(StringIO(blob), skipinitialspace=True))


def _read_input_table(path: Path) -> list[tuple[str, str, list[str]]]:
    """Return ``(section, key, value_parts)`` for each non-comment row."""
    rows: list[tuple[str, str, list[str]]] = []
    with open(path, encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("section,"):
                continue
            first = line.find(",")
            second = line.find(",", first + 1) if first >= 0 else -1
            if first < 0 or second < 0:
                raise ValueError(
                    f"{path}:{line_no}: expected section,key,<values> (got {raw_line.rstrip()!r})"
                )
            section = line[:first].strip().lower()
            key = line[first + 1 : second].strip()
            value_blob = line[second + 1 :]
            parts = _split_csv_value_field(value_blob)
            rows.append((section, key, parts))
    return rows


def _fill_forward_per_set(parts: list[str]) -> list[str]:
    """Blank tokens inherit the prior set_id value (first token must be non-blank)."""
    if not parts:
        return parts
    out = list(parts)
    for i in range(1, len(out)):
        if not str(out[i]).strip():
            out[i] = out[i - 1]
    return out


def _value_at_index(
    parts: list[str],
    n_sets: int,
    index: int,
    *,
    label: str,
) -> str:
    if len(parts) == 1:
        return parts[0]
    if len(parts) != n_sets:
        raise ValueError(
            f"{label}: expected 1 or {n_sets} comma-separated value(s) "
            f"(one per meta.set_id), got {len(parts)}: {parts!r}"
        )
    filled = _fill_forward_per_set(parts)
    if not str(filled[0]).strip():
        raise ValueError(f"{label}: first set_id value cannot be empty")
    return filled[index]


def load_single_calibrate_inputs(path: Path) -> list[SingleCalibrateInput]:
    """Read SteelMPF ``input.csv``; return one ``SingleCalibrateInput`` per ``meta.set_id``."""
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Missing calibration input: {p}")

    table = _read_input_table(p)
    meta: dict[str, list[str]] = {}
    steel_raw: dict[str, list[str]] = {}
    loss_kv: dict[str, list[str]] = {}
    bounds_raw: dict[str, list[str]] = {}

    for section, key, parts in table:
        if section == "meta":
            meta[key] = parts
        elif section == "steel":
            steel_raw[key] = parts
        elif section == "loss":
            loss_kv[key.lower()] = parts
        elif section == "bound":
            bounds_raw[key] = parts
        else:
            raise ValueError(f"{p}: unknown section {section!r} (use meta|steel|loss|bound)")

    if "set_id" not in meta:
        raise ValueError(f"{p}: missing meta.set_id")
    if "optimize_params" not in meta:
        raise ValueError(f"{p}: missing meta.optimize_params")

    set_ids = [int(x.strip()) for x in meta["set_id"]]
    if not set_ids:
        raise ValueError(f"{p}: meta.set_id is empty")
    n_sets = len(set_ids)

    missing_steel = [
        k
        for k in (*STEELMPF_NUMERIC_SEED_KEYS, *STEELMPF_B_SPEC_KEYS)
        if k not in steel_raw
    ]
    if missing_steel:
        raise ValueError(f"{p}: missing steel rows: {missing_steel}")

    configs: list[SingleCalibrateInput] = []
    for i, set_id in enumerate(set_ids):
        label_set = f"set_id={set_id}"
        bounds: dict[str, tuple[float, float]] = {}
        for param, parts in bounds_raw.items():
            raw = _value_at_index(
                parts,
                n_sets,
                i,
                label=f"bound.{param} ({label_set})",
            )
            bounds[param] = _parse_bound(raw, param=param)

        optimize_params = _parse_optimize_params(
            _value_at_index(
                meta["optimize_params"],
                n_sets,
                i,
                label=f"meta.optimize_params ({label_set})",
            )
        )
        missing_bounds = [pname for pname in optimize_params if pname not in bounds]
        if missing_bounds:
            raise ValueError(
                f"{p}: optimize_params {missing_bounds} for {label_set} "
                f"have no bound.* rows in input.csv"
            )

        steel_seeds = {
            k: float(
                _value_at_index(
                    steel_raw[k],
                    n_sets,
                    i,
                    label=f"steel.{k} ({label_set})",
                )
            )
            for k in STEELMPF_NUMERIC_SEED_KEYS
        }
        b_p_spec = parse_b_p_n_spec(
            _value_at_index(
                steel_raw["b_p"],
                n_sets,
                i,
                label=f"steel.b_p ({label_set})",
            ),
            label=f"steel.b_p ({label_set})",
        )
        b_n_spec = parse_b_p_n_spec(
            _value_at_index(
                steel_raw["b_n"],
                n_sets,
                i,
                label=f"steel.b_n ({label_set})",
            ),
            label=f"steel.b_n ({label_set})",
        )

        loss = CalibrationLossSettings(
            w_feat_l2=float(
                _value_at_index(
                    loss_kv.get("w_feat_l2", ["1.0"]),
                    n_sets,
                    i,
                    label=f"loss.w_feat_l2 ({label_set})",
                )
            ),
            w_feat_l1=float(
                _value_at_index(
                    loss_kv.get("w_feat_l1", ["0.0"]),
                    n_sets,
                    i,
                    label=f"loss.w_feat_l1 ({label_set})",
                )
            ),
            w_energy_l2=float(
                _value_at_index(
                    loss_kv.get("w_energy_l2", ["0.0"]),
                    n_sets,
                    i,
                    label=f"loss.w_energy_l2 ({label_set})",
                )
            ),
            w_energy_l1=float(
                _value_at_index(
                    loss_kv.get("w_energy_l1", ["0.0"]),
                    n_sets,
                    i,
                    label=f"loss.w_energy_l1 ({label_set})",
                )
            ),
            w_unordered_binenv_l2=float(
                _value_at_index(
                    loss_kv.get("w_unordered_binenv_l2", ["0.0"]),
                    n_sets,
                    i,
                    label=f"loss.w_unordered_binenv_l2 ({label_set})",
                )
            ),
            w_unordered_binenv_l1=float(
                _value_at_index(
                    loss_kv.get("w_unordered_binenv_l1", ["0.0"]),
                    n_sets,
                    i,
                    label=f"loss.w_unordered_binenv_l1 ({label_set})",
                )
            ),
            use_amplitude_weights=parse_bool_cell(
                _value_at_index(
                    loss_kv.get("use_amplitude_weights", ["true"]),
                    n_sets,
                    i,
                    label=f"loss.use_amplitude_weights ({label_set})",
                )
            ),
            amplitude_weight_power=float(
                _value_at_index(
                    loss_kv.get("amplitude_weight_power", ["2.0"]),
                    n_sets,
                    i,
                    label=f"loss.amplitude_weight_power ({label_set})",
                )
            ),
            amplitude_weight_eps=float(
                _value_at_index(
                    loss_kv.get("amplitude_weight_eps", ["0.05"]),
                    n_sets,
                    i,
                    label=f"loss.amplitude_weight_eps ({label_set})",
                )
            ),
        )

        configs.append(
            SingleCalibrateInput(
                set_id=set_id,
                optimize_params=optimize_params,
                b_p_spec=b_p_spec,
                b_n_spec=b_n_spec,
                steel_seeds=steel_seeds,
                loss=loss,
                param_bounds=bounds,
            )
        )
    return configs


def load_single_calibrate_input(path: Path) -> SingleCalibrateInput:
    """Load ``input.csv``; error if more than one ``meta.set_id`` is listed."""
    configs = load_single_calibrate_inputs(path)
    if len(configs) != 1:
        raise ValueError(
            f"{path}: meta.set_id lists {len(configs)} set_ids; "
            "use load_single_calibrate_inputs() or list one set_id."
        )
    return configs[0]
