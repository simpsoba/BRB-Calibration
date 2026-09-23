"""
Resample filtered F-u data along characteristic segments with fixed step in |Deltau|.

Used by postprocess/resample_filtered.py. No OpenSees import (tests friendly).
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd


FORCE_COL = "Force[kip]"
DEF_COL = "Deformation[in]"


def compute_Q_brace(L_T: float, L_y: float, A_sc: float, A_t: float) -> float:
    """Same as ``model.brace_geometry.compute_Q`` (import deferred-safe for postprocess)."""
    from model.brace_geometry import compute_Q

    return compute_Q(L_T, L_y, A_sc, A_t)


def yield_deformation_inches(
    fyp_ksi: float,
    L_T_in: float,
    E_hat_ksi: float,
) -> float:
    """Dy = fyp * L_T / E_hat (inches)."""
    if not np.isfinite(E_hat_ksi) or E_hat_ksi <= 0.0:
        return float("nan")
    return float(fyp_ksi * L_T_in / E_hat_ksi)


def d_sampling_from_brace_params(
    *,
    fyp_ksi: float,
    L_T_in: float,
    L_y_in: float,
    A_sc_in2: float,
    A_t_in2: float,
    E_ksi: float,
    u_fallback: np.ndarray,
) -> float:
    """
    D_sampling = Dy / 4 with Dy = fyp * L_T / E_hat, E_hat = Q * E.
    If invalid, fall back to max(|u|)/100 and warn.
    """
    Q = compute_Q_brace(L_T_in, L_y_in, A_sc_in2, A_t_in2)
    E_hat = Q * float(E_ksi)
    Dy = yield_deformation_inches(fyp_ksi, L_T_in, E_hat)
    if np.isfinite(Dy) and Dy > 0.0:
        return float(Dy / 4.0)
    umax = float(np.nanmax(np.abs(np.asarray(u_fallback, dtype=float))))
    if not np.isfinite(umax) or umax <= 0.0:
        umax = 1.0
    fb = umax / 100.0
    warnings.warn(
        f"Invalid Dy; using D_sampling = max(|u|)/100 ~ {fb:.6g} in.",
        UserWarning,
        stacklevel=2,
    )
    return fb


def _normalized_boundaries(boundary_indices: list[int], n: int) -> list[int]:
    """Normalize segment boundaries for resampling helper."""
    b = sorted(set(int(x) for x in boundary_indices) | {0, n})
    return b


def _segment_slices(b: list[int], n: int) -> list[tuple[int, int]]:
    """Return iloc (start, stop) slices; inclusive path through shared boundary rows."""
    out: list[tuple[int, int]] = []
    for j in range(len(b) - 1):
        a, c = b[j], b[j + 1]
        if c >= n:
            out.append((a, n))
        else:
            out.append((a, c + 1))
    return out


def resample_segment_along_u_path(
    u: np.ndarray,
    f: np.ndarray,
    d_sampling: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Uniform spacing in cumulative |Deltau| along index order; linear interp u and F.
    Endpoints match first/last input samples.
    """
    u = np.asarray(u, dtype=float)
    f = np.asarray(f, dtype=float)
    m = len(u)
    if m == 0:
        return np.array([]), np.array([])
    if m == 1:
        return u.copy(), f.copy()
    du = np.abs(np.diff(u))
    L_u = float(np.sum(du))
    if L_u <= 1e-15:
        return np.array([float(u[0]), float(u[-1])]), np.array([float(f[0]), float(f[-1])])
    n_pts = max(2, int(np.ceil(L_u / d_sampling)) + 1)
    s = np.concatenate([[0.0], np.cumsum(du)])
    idx_fp = np.arange(m, dtype=float)
    tq = np.linspace(0.0, L_u, n_pts)
    idx_q = np.interp(tq, s, idx_fp)
    u_out = np.interp(idx_q, idx_fp, u)
    f_out = np.interp(idx_q, idx_fp, f)
    return u_out, f_out


def resample_by_segments(
    df: pd.DataFrame,
    boundary_indices: list[int],
    d_sampling: float,
) -> tuple[pd.DataFrame, dict[int, int]]:
    """
    Resample each segment between sorted boundaries; concatenate with deduped joints.

    Returns
    -------
    out_df
        Columns FORCE_COL, DEF_COL.
    filtered_index_to_resampled
        Map each boundary index (0..n-1 and n for end) appearing in boundaries to resampled row index.
    """
    n = len(df)
    if FORCE_COL not in df.columns or DEF_COL not in df.columns:
        raise ValueError(f"Expected columns {FORCE_COL}, {DEF_COL}")
    u_all = df[DEF_COL].to_numpy(dtype=float)
    f_all = df[FORCE_COL].to_numpy(dtype=float)

    b = _normalized_boundaries(boundary_indices, n)
    slices = _segment_slices(b, n)
    segs_res: list[tuple[np.ndarray, np.ndarray]] = []
    for a, stop in slices:
        u_seg = u_all[a:stop]
        f_seg = f_all[a:stop]
        if len(u_seg) == 0:
            continue
        ur, fr = resample_segment_along_u_path(u_seg, f_seg, d_sampling)
        segs_res.append((ur, fr))

    if not segs_res:
        return pd.DataFrame({DEF_COL: [], FORCE_COL: []}), {}

    u_cat = segs_res[0][0]
    f_cat = segs_res[0][1]
    for k in range(1, len(segs_res)):
        ur, fr = segs_res[k]
        if len(ur) <= 1:
            continue
        u_cat = np.concatenate([u_cat, ur[1:]])
        f_cat = np.concatenate([f_cat, fr[1:]])

    out_df = pd.DataFrame({DEF_COL: u_cat, FORCE_COL: f_cat})

    # Remap boundary b[j] -> resampled index
    remap: dict[int, int] = {}
    cur = 0
    for j, (ur, _) in enumerate(segs_res):
        L = len(ur)
        Bj = b[j]
        Bj1 = b[j + 1]
        if j == 0:
            remap[Bj] = 0
        if j == 0:
            remap[Bj1] = cur + L - 1
            cur += L
        else:
            remap[Bj1] = cur + L - 2
            cur += L - 1

    return out_df, remap


def remap_cycle_points(
    points: list[dict[str, Any]],
    remap_idx: dict[int, int],
    out_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """New points list with idx on resampled grid and updated force/deformation."""
    n_out = len(out_df)
    new_points: list[dict[str, Any]] = []
    skipped_no_remap: list[int] = []
    skipped_range: list[int] = []
    for p in points:
        old = int(p["idx"])
        ni = remap_idx.get(old)
        if ni is None:
            skipped_no_remap.append(old)
            continue
        if not (0 <= ni < n_out):
            skipped_range.append(int(ni))
            continue
        q = dict(p)
        q["idx"] = ni
        q["deformation_in"] = float(out_df[DEF_COL].iloc[ni])
        q["force_kip"] = float(out_df[FORCE_COL].iloc[ni])
        new_points.append(q)
    if skipped_no_remap:
        warnings.warn(
            f"No remap for {len(skipped_no_remap)} cycle point index(es) "
            f"(e.g. {skipped_no_remap[:5]}{'...' if len(skipped_no_remap) > 5 else ''}); skipping.",
            UserWarning,
            stacklevel=2,
        )
    if skipped_range:
        warnings.warn(
            f"{len(skipped_range)} remapped idx out of range (e.g. {skipped_range[:3]}); skipping.",
            UserWarning,
            stacklevel=2,
        )
    return new_points
