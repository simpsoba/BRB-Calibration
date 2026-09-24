"""
Averaged / generalized evaluation helpers for **digitized unordered** specimens (catalog: ``path_ordered=false``):
load the deformation drive and unordered F–u samples, merge envelope ``b_p``/``b_n`` into a parameter row.

The drive for simulation prefers ``data/resampled/{Name}/deformation_history.csv`` from the postprocess
pipeline (same |Du| segmentation as path-ordered). If that file is missing, falls back to raw
``deformation_history.csv`` plus ``prepare_deformation_drive`` (legacy).

``digitized`` + ``path_ordered=true`` specimens are **not** handled here--they follow the standard
resampled ``force_deformation`` path like other ordered tests.
"""
from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from calibrate.deformation_history_drive import (
    DEFAULT_RDP_EPSILON_IN,
    prepare_deformation_drive,
    prepend_lead_deformation,
)
from calibrate.resample_experiment import d_sampling_from_brace_params
from calibrate.digitized_unordered_bn import compute_envelope_bn_unordered

_POST = Path(__file__).resolve().parent.parent / "postprocess"
if str(_POST) not in sys.path:
    sys.path.insert(0, str(_POST))
from specimen_catalog import (  # noqa: E402
    deformation_history_csv_path,
    force_deformation_unordered_csv_path,
    resampled_deformation_history_csv,
)

DEF_COL = "Deformation[in]"
FORCE_COL = "Force[kip]"

DEFAULT_BINENV_N_BINS = 32


@dataclass(frozen=True)
class UnorderedCloudMetricsResult:
    """One-sided NN (numerical→experimental) + binned max/min force bands in (D,F) space."""

    J_nearest: float
    J_binenv: float
    J_nearest_l1: float
    J_binenv_l1: float
    num_to_exp_nearest_idx: np.ndarray
    S_D_exp: float
    S_F_exp: float
    exp_points_raw: np.ndarray
    num_points_raw: np.ndarray
    exp_points_norm: np.ndarray
    num_points_norm: np.ndarray


def _nearest_indices_and_sqdist_mean(
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    *,
    chunk_size: int = 2048,
) -> tuple[np.ndarray, float]:
    """
    Return nearest target index for each source point and mean squared nearest distance.

    Chunking keeps memory bounded for large clouds.
    """
    n_src = int(source_xy.shape[0])
    nearest = np.empty(n_src, dtype=int)
    min_sq = np.empty(n_src, dtype=float)

    for i0 in range(0, n_src, chunk_size):
        i1 = min(i0 + chunk_size, n_src)
        src = source_xy[i0:i1]
        # squared L2 distance in normalized space: (x1-x2)^2 + (y1-y2)^2
        d2 = np.sum((src[:, None, :] - target_xy[None, :, :]) ** 2, axis=2)
        arg = np.argmin(d2, axis=1)
        nearest[i0:i1] = arg
        min_sq[i0:i1] = d2[np.arange(i1 - i0), arg]

    return nearest, float(np.mean(min_sq))


def num_to_exp_nearest_indices(
    num_xy: np.ndarray,
    exp_xy: np.ndarray,
    *,
    chunk_size: int = 2048,
) -> np.ndarray:
    """For each row of ``num_xy``, index of nearest row in ``exp_xy`` (same normalization for both)."""
    idx, _ = _nearest_indices_and_sqdist_mean(
        np.asarray(num_xy, dtype=float),
        np.asarray(exp_xy, dtype=float),
        chunk_size=chunk_size,
    )
    return idx


def compute_unordered_binenv_metrics(
    u_exp: np.ndarray,
    F_exp: np.ndarray,
    D_num: np.ndarray,
    F_num: np.ndarray,
    *,
    scale_eps: float = 1e-12,
    n_binenv_bins: int = DEFAULT_BINENV_N_BINS,
) -> tuple[float, float]:
    """
    Binned max/min force mismatch per deformation bin (no nearest-neighbor / O(N²) cost).

    Returns ``(J_binenv_L2, J_binenv_L1)`` with the same bins as the cloud diagnostic.
    """
    exp_xy_raw = np.column_stack([np.asarray(u_exp, dtype=float), np.asarray(F_exp, dtype=float)])
    num_xy_raw = np.column_stack([np.asarray(D_num, dtype=float), np.asarray(F_num, dtype=float)])

    exp_mask = np.isfinite(exp_xy_raw).all(axis=1)
    num_mask = np.isfinite(num_xy_raw).all(axis=1)
    exp_xy_raw = exp_xy_raw[exp_mask]
    num_xy_raw = num_xy_raw[num_mask]

    if exp_xy_raw.shape[0] == 0 or num_xy_raw.shape[0] == 0:
        return float("nan"), float("nan")

    s_d = float(np.max(exp_xy_raw[:, 0]) - np.min(exp_xy_raw[:, 0]))
    s_f = float(np.max(exp_xy_raw[:, 1]) - np.min(exp_xy_raw[:, 1]))
    if not np.isfinite(s_d) or not np.isfinite(s_f) or abs(s_d) <= scale_eps or abs(s_f) <= scale_eps:
        return float("nan"), float("nan")

    j_binenv = _compute_binned_binenv_l2(
        exp_xy_raw, num_xy_raw, s_f, n_bins=n_binenv_bins, scale_eps=scale_eps
    )
    j_binenv_l1 = _compute_binned_binenv_l1(
        exp_xy_raw, num_xy_raw, s_f, n_bins=n_binenv_bins, scale_eps=scale_eps
    )
    return float(j_binenv), float(j_binenv_l1)


def _compute_binned_binenv_l2(
    exp_xy_raw: np.ndarray,
    num_xy_raw: np.ndarray,
    s_f: float,
    *,
    n_bins: int = DEFAULT_BINENV_N_BINS,
    scale_eps: float = 1e-12,
) -> float:
    """
    Mean over deformation bins of e_b, with

        e_b = (1/2) * [ ((F_exp^up - F_num^up) / S_F)^2 + ((F_exp^lo - F_num^lo) / S_F)^2 ]

    where per bin F^up = max F and F^lo = min F within that cloud in the bin (same bin edges for
    both clouds). Bins are equal-width on [min(u_exp), max(u_exp)]. A bin contributes only if both
    clouds have at least one finite point in that bin.
    """
    if exp_xy_raw.shape[0] == 0 or num_xy_raw.shape[0] == 0:
        return float("nan")
    if not np.isfinite(s_f) or abs(s_f) <= scale_eps:
        return float("nan")

    u_e = exp_xy_raw[:, 0]
    f_e = exp_xy_raw[:, 1]
    u_n = num_xy_raw[:, 0]
    f_n = num_xy_raw[:, 1]

    u_min = float(np.min(u_e))
    u_max = float(np.max(u_e))
    if not np.isfinite(u_min) or not np.isfinite(u_max) or u_max <= u_min:
        return float("nan")

    edges = np.linspace(u_min, u_max, int(n_bins) + 1)
    errs: list[float] = []

    for b in range(int(n_bins)):
        lo, hi = float(edges[b]), float(edges[b + 1])
        if b == int(n_bins) - 1:
            m_e = (u_e >= lo) & (u_e <= hi)
            m_n = (u_n >= lo) & (u_n <= hi)
        else:
            m_e = (u_e >= lo) & (u_e < hi)
            m_n = (u_n >= lo) & (u_n < hi)

        if int(np.count_nonzero(m_e)) < 1 or int(np.count_nonzero(m_n)) < 1:
            continue

        f_up_e = float(np.max(f_e[m_e]))
        f_lo_e = float(np.min(f_e[m_e]))
        f_up_n = float(np.max(f_n[m_n]))
        f_lo_n = float(np.min(f_n[m_n]))
        du = (f_up_e - f_up_n) / s_f
        dl = (f_lo_e - f_lo_n) / s_f
        errs.append(0.5 * (du * du + dl * dl))

    if not errs:
        return float("nan")
    return float(np.mean(errs))


def _compute_binned_binenv_l1(
    exp_xy_raw: np.ndarray,
    num_xy_raw: np.ndarray,
    s_f: float,
    *,
    n_bins: int = DEFAULT_BINENV_N_BINS,
    scale_eps: float = 1e-12,
) -> float:
    """L1 analogue of ``_compute_binned_binenv_l2``: mean of ``(1/2)(|ΔF_up|+|ΔF_lo|)/S_F`` per bin."""
    if exp_xy_raw.shape[0] == 0 or num_xy_raw.shape[0] == 0:
        return float("nan")
    if not np.isfinite(s_f) or abs(s_f) <= scale_eps:
        return float("nan")

    u_e = exp_xy_raw[:, 0]
    f_e = exp_xy_raw[:, 1]
    u_n = num_xy_raw[:, 0]
    f_n = num_xy_raw[:, 1]

    u_min = float(np.min(u_e))
    u_max = float(np.max(u_e))
    if not np.isfinite(u_min) or not np.isfinite(u_max) or u_max <= u_min:
        return float("nan")

    edges = np.linspace(u_min, u_max, int(n_bins) + 1)
    errs: list[float] = []

    for b in range(int(n_bins)):
        lo, hi = float(edges[b]), float(edges[b + 1])
        if b == int(n_bins) - 1:
            m_e = (u_e >= lo) & (u_e <= hi)
            m_n = (u_n >= lo) & (u_n <= hi)
        else:
            m_e = (u_e >= lo) & (u_e < hi)
            m_n = (u_n >= lo) & (u_n < hi)

        if int(np.count_nonzero(m_e)) < 1 or int(np.count_nonzero(m_n)) < 1:
            continue

        f_up_e = float(np.max(f_e[m_e]))
        f_lo_e = float(np.min(f_e[m_e]))
        f_up_n = float(np.max(f_n[m_n]))
        f_lo_n = float(np.min(f_n[m_n]))
        du = abs(f_up_e - f_up_n) / s_f
        dl = abs(f_lo_e - f_lo_n) / s_f
        errs.append(0.5 * (du + dl))

    if not errs:
        return float("nan")
    return float(np.mean(errs))


def compute_unordered_cloud_metrics(
    u_exp: np.ndarray,
    F_exp: np.ndarray,
    D_num: np.ndarray,
    F_num: np.ndarray,
    *,
    scale_eps: float = 1e-12,
    n_binenv_bins: int = DEFAULT_BINENV_N_BINS,
) -> UnorderedCloudMetricsResult:
    """
    Cloud diagnostics for experimental vs numerical hysteresis (plots / reporting).

    **J_nearest** and **J_nearest_l1** are no longer computed (always NaN); use
    ``num_to_exp_nearest_indices`` in display space if you need proximity maps.
    **J_binenv** / **J_binenv_l1**: binned max/min force mismatch (L2 / L1 of upper/lower force gaps per bin).

    Normalization uses experimental ranges only:
    S_D = max(u_exp) - min(u_exp), S_F = max(F_exp) - min(F_exp).
    """
    exp_xy_raw = np.column_stack([np.asarray(u_exp, dtype=float), np.asarray(F_exp, dtype=float)])
    num_xy_raw = np.column_stack([np.asarray(D_num, dtype=float), np.asarray(F_num, dtype=float)])

    exp_mask = np.isfinite(exp_xy_raw).all(axis=1)
    num_mask = np.isfinite(num_xy_raw).all(axis=1)
    exp_xy_raw = exp_xy_raw[exp_mask]
    num_xy_raw = num_xy_raw[num_mask]

    if exp_xy_raw.shape[0] == 0 or num_xy_raw.shape[0] == 0:
        return UnorderedCloudMetricsResult(
            J_nearest=float("nan"),
            J_binenv=float("nan"),
            J_nearest_l1=float("nan"),
            J_binenv_l1=float("nan"),
            num_to_exp_nearest_idx=np.full(num_xy_raw.shape[0], -1, dtype=int),
            S_D_exp=float("nan"),
            S_F_exp=float("nan"),
            exp_points_raw=np.empty((0, 2), dtype=float),
            num_points_raw=np.empty((0, 2), dtype=float),
            exp_points_norm=np.empty((0, 2), dtype=float),
            num_points_norm=np.empty((0, 2), dtype=float),
        )

    s_d = float(np.max(exp_xy_raw[:, 0]) - np.min(exp_xy_raw[:, 0]))
    s_f = float(np.max(exp_xy_raw[:, 1]) - np.min(exp_xy_raw[:, 1]))
    if not np.isfinite(s_d) or not np.isfinite(s_f) or abs(s_d) <= scale_eps or abs(s_f) <= scale_eps:
        warnings.warn(
            "compute_unordered_cloud_metrics: invalid experimental scale (S_D or S_F near zero); returning NaN.",
            UserWarning,
            stacklevel=2,
        )
        return UnorderedCloudMetricsResult(
            J_nearest=float("nan"),
            J_binenv=float("nan"),
            J_nearest_l1=float("nan"),
            J_binenv_l1=float("nan"),
            num_to_exp_nearest_idx=np.full(num_xy_raw.shape[0], -1, dtype=int),
            S_D_exp=s_d,
            S_F_exp=s_f,
            exp_points_raw=np.empty((0, 2), dtype=float),
            num_points_raw=np.empty((0, 2), dtype=float),
            exp_points_norm=np.empty((0, 2), dtype=float),
            num_points_norm=np.empty((0, 2), dtype=float),
        )

    exp_xy = np.column_stack([exp_xy_raw[:, 0] / s_d, exp_xy_raw[:, 1] / s_f])
    num_xy = np.column_stack([num_xy_raw[:, 0] / s_d, num_xy_raw[:, 1] / s_f])

    j_binenv = _compute_binned_binenv_l2(
        exp_xy_raw, num_xy_raw, s_f, n_bins=n_binenv_bins, scale_eps=scale_eps
    )
    j_binenv_l1 = _compute_binned_binenv_l1(
        exp_xy_raw, num_xy_raw, s_f, n_bins=n_binenv_bins, scale_eps=scale_eps
    )
    num_to_exp_idx = np.full(num_xy_raw.shape[0], -1, dtype=int)

    return UnorderedCloudMetricsResult(
        J_nearest=float("nan"),
        J_binenv=j_binenv,
        J_nearest_l1=float("nan"),
        J_binenv_l1=j_binenv_l1,
        num_to_exp_nearest_idx=num_to_exp_idx,
        S_D_exp=s_d,
        S_F_exp=s_f,
        exp_points_raw=exp_xy_raw,
        num_points_raw=num_xy_raw,
        exp_points_norm=exp_xy,
        num_points_norm=num_xy,
    )




def load_digitized_unordered_series(
    specimen_id: str,
    project_root: Path,
    *,
    prepare_drive: bool = True,
    use_pipeline_resampled_drive: bool = True,
    steel_row: pd.Series | None = None,
    catalog_row: pd.Series | None = None,
    drive_rdp_epsilon_in: float = DEFAULT_RDP_EPSILON_IN,
    drive_d_sampling_in: float | None = None,
    drive_median_kernel: int = 0,
    drive_resample: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Return ``(D_drive, u_unordered, F_unordered)`` for digitized unordered inputs, or None if missing/invalid.

    If ``use_pipeline_resampled_drive`` (default True) and
    ``data/resampled/{Name}/deformation_history.csv`` exists with ``Deformation[in]``, ``D_drive`` is
    that column (postprocess segment resample; no RDP/prepend here).

    Otherwise loads raw ``deformation_history.csv``. When ``prepare_drive`` is True, applies
    ``prepare_deformation_drive`` (median/RDP/uniform |Du|, lead sample prepended). When False, prepends
    ``0.1 * D_\mathrm{sample}`` to raw samples only.

    ``u_unordered`` / ``F_unordered`` always come from raw ``force_deformation.csv`` (unordered F–u samples).
    """
    fd = force_deformation_unordered_csv_path(specimen_id, project_root)
    if not fd.is_file():
        return None
    df_fd = pd.read_csv(fd)
    if DEF_COL not in df_fd.columns or FORCE_COL not in df_fd.columns:
        return None
    u_c = df_fd[DEF_COL].to_numpy(dtype=float)
    F_c = df_fd[FORCE_COL].to_numpy(dtype=float)

    dh_res = resampled_deformation_history_csv(specimen_id, project_root)
    if use_pipeline_resampled_drive and dh_res.is_file():
        df_res = pd.read_csv(dh_res)
        if DEF_COL in df_res.columns and len(df_res) > 0:
            D_drive = df_res[DEF_COL].to_numpy(dtype=float)
            return D_drive, u_c, F_c

    dh = deformation_history_csv_path(specimen_id, project_root)
    if not dh.is_file():
        return None
    df_dh = pd.read_csv(dh)
    if DEF_COL not in df_dh.columns:
        return None
    D_raw = df_dh[DEF_COL].to_numpy(dtype=float)

    if use_pipeline_resampled_drive:
        warnings.warn(
            f"{specimen_id}: no pipeline resampled deformation_history at {dh_res}; "
            "falling back to raw drive + prepare_deformation_drive / lead prepend.",
            UserWarning,
            stacklevel=2,
        )

    brace = None
    if steel_row is not None and catalog_row is not None:
        brace = {
            "fyp_ksi": float(steel_row["fyp"]),
            "L_T_in": float(catalog_row["L_T"]),
            "L_y_in": float(catalog_row["L_y"]),
            "A_sc_in2": float(catalog_row["A_sc"]),
            "A_t_in2": float(catalog_row["A_t"]),
            "E_ksi": float(steel_row["E"]),
        }

    if prepare_drive:
        D_drive = prepare_deformation_drive(
            D_raw,
            rdp_epsilon_in=drive_rdp_epsilon_in,
            median_kernel=drive_median_kernel,
            d_sampling_in=drive_d_sampling_in,
            resample=drive_resample,
            brace=brace,
            u_fallback=D_raw,
        )
    else:
        if drive_d_sampling_in is not None and float(drive_d_sampling_in) > 0.0:
            d_sp = float(drive_d_sampling_in)
        elif brace is not None:
            d_sp = d_sampling_from_brace_params(
                fyp_ksi=float(brace["fyp_ksi"]),
                L_T_in=float(brace["L_T_in"]),
                L_y_in=float(brace["L_y_in"]),
                A_sc_in2=float(brace["A_sc_in2"]),
                A_t_in2=float(brace["A_t_in2"]),
                E_ksi=float(brace["E_ksi"]),
                u_fallback=D_raw,
            )
        else:
            umax = float(np.nanmax(np.abs(D_raw)))
            d_sp = umax / 100.0 if umax > 0 else 1e-6
        D_drive = prepend_lead_deformation(D_raw, d_sp)

    return D_drive, u_c, F_c


def eval_row_with_envelope_bn_from_unordered(
    eval_row: pd.Series,
    catalog_row: pd.Series,
    u_unordered: np.ndarray,
    F_unordered: np.ndarray,
) -> pd.Series:
    """Copy of ``eval_row`` with ``b_p`` / ``b_n`` replaced by envelope estimates from unordered F–u."""
    out = eval_row.copy()
    L_T = float(catalog_row["L_T"])
    L_y = float(catalog_row["L_y"])
    A_sc = float(catalog_row["A_sc"])
    A_t = float(catalog_row["A_t"])
    f_yc = float(catalog_row["fyp"])
    E_ksi = float(eval_row["E"])
    diag = compute_envelope_bn_unordered(
        u_unordered,
        F_unordered,
        L_T=L_T,
        L_y=L_y,
        A_sc=A_sc,
        A_t=A_t,
        f_yc=f_yc,
        E_ksi_val=E_ksi,
    )
    if np.isfinite(diag.b_p):
        out["b_p"] = diag.b_p
    if np.isfinite(diag.b_n):
        out["b_n"] = diag.b_n
    return out
