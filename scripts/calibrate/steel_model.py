"""SteelMPF material constants and simulation parameter keys."""

from __future__ import annotations

STEEL_MODEL_STEELMPF = "steelmpf"

# Isotropic hardening parameters (stock OpenSeesPy SteelMPF through a4).
STEELMPF_ISO_KEYS: tuple[str, ...] = ("a1", "a2", "a3", "a4")

# Shared kinematic / modulus seeds in CSV.
SHARED_STEEL_KEYS: tuple[str, ...] = ("E", "R0", "cR1", "cR2")

BRACE_GEOMETRY_SIM_KEYS: tuple[str, ...] = ("L_T", "L_y", "A_sc", "A_t", "fyp", "fyn")
SHARED_SIM_KEYS: tuple[str, ...] = (
    *BRACE_GEOMETRY_SIM_KEYS,
    "E",
    "b_p",
    "b_n",
    "R0",
    "cR1",
    "cR2",
)
STEELMPF_SIM_KEYS: tuple[str, ...] = (*SHARED_SIM_KEYS, *STEELMPF_ISO_KEYS)


def sim_param_keys_for_model(_steel_model: object = None) -> tuple[str, ...]:
    """OpenSees ``run_simulation`` kwargs (SteelMPF only)."""
    return STEELMPF_SIM_KEYS


def normalize_steel_model(raw: object = None) -> str:
    """Return ``steelmpf``. Blank / missing values default to SteelMPF; other names raise."""
    if raw is None:
        return STEEL_MODEL_STEELMPF
    try:
        import pandas as pd

        if isinstance(raw, (float, int)) and not isinstance(raw, bool):
            if isinstance(raw, float) and pd.isna(raw):
                return STEEL_MODEL_STEELMPF
        if pd.isna(raw):
            return STEEL_MODEL_STEELMPF
    except Exception:
        pass
    s = str(raw).strip().lower()
    if not s or s == "nan":
        return STEEL_MODEL_STEELMPF
    if s in (STEEL_MODEL_STEELMPF, "steel_mpf", "mpf"):
        return STEEL_MODEL_STEELMPF
    raise ValueError(
        f"steel_model must be 'steelmpf' (SteelMPF only); got {raw!r}"
    )
