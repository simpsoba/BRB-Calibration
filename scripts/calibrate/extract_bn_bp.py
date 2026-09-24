"""Extract apparent b_n and b_p from resampled force–deformation (path-ordered and digitized).

Writes ``results/calibration/specimen_apparent_bn_bp.csv`` for seeding ``set_id_settings.csv``.
Plastic-region fits use yield→peak deformation fractions (see constants near the top of this file).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
_SCRIPTS = _PROJECT_ROOT / "scripts"
sys.path.insert(0, str(_SCRIPT_DIR))  # calibration_paths, digitized_unordered_bn
sys.path.insert(0, str(_SCRIPTS / "postprocess"))  # so cycle_points can "from load_raw import ..."
sys.path.insert(0, str(_SCRIPTS))
from calibration_paths import BRB_SPECIMENS_CSV, SPECIMEN_APPARENT_BN_BP_PATH  # noqa: E402
from cycle_feature_loss import PLASTIC_STRESS_RATIO_FY  # noqa: E402
from digitized_unordered_bn import compute_envelope_bn_unordered  # noqa: E402
from model.brace_geometry import compute_Q
from postprocess.cycle_points import find_cycle_points, load_cycle_points_resampled
from specimen_catalog import (  # noqa: E402
    force_deformation_unordered_csv_path,
    list_names_digitized_unordered,
    path_ordered_resampled_force_csv_stems,
    read_catalog,
    resolve_filtered_force_deformation_csv,
    resolve_force_deformation_csv_for_max_strain,
    resolve_resampled_force_deformation_csv,
)

E_ksi = 29000.0
CATALOG_PATH = BRB_SPECIMENS_CSV
DEF_COL = "Deformation[in]"
FORCE_COL = "Force[kip]"
# Minimum deformation range (first to last point of fit) to use a segment for b; fraction of L_y
MIN_DEF_RANGE_FRAC = 0.005
MIN_DEF_RANGE_ABS_IN = 0.001  # fallback when L_y is 0
# Near-parallel plastic vs elastic asymptotes in $\\sigma_0$ intersection
_SIG0_K_PARALLEL_TOL_FRAC = 1e-6

# Apparent-b amplitude weighting (for weighted_mean outputs).
# Uses the same functional form as cycle amplitude weights:
#   w = (A/Amax)^p + eps
APPARENT_B_WEIGHT_POWER = 2.0
APPARENT_B_WEIGHT_EPS = 0.05

# Drop segment ``b`` values outside max(0, Q1 - k*IQR) .. Q3 + k*IQR (separate k for b_p and b_n lists).
APPARENT_B_IQR_MULTIPLIER = 3.0
APPARENT_B_MIN = 0.0

# Secant slope for ``b_plastic_onset="half_elastic_secant"``: ``F = F_r + SECANT_K_FRAC * k_init * (u-u_r)``.
B_PLASTIC_ONSET_SECANT_K_FRAC = 1.0 / 3.0
# b-fit start: fraction of yield→peak-``F`` **deformation span still remaining** past ``u_{\\mathrm{level}}``
# (see ``_plastic_onset_index_yield_deform_frac_to_peak_force``). Default 0.75 ⇒ start at 25% of span from yield.
B_PLASTIC_ONSET_DEFORM_FRAC_REMAINING_TO_PEAK_F = 0.75
# Plastic line fit end along yield→peak-``F`` ``du`` (same ``du`` as onset): last index with ``u`` no past
# ``u_yield + B_PLASTIC_FIT_DEFORM_FRAC_YIELD_TO_PEAK_F * du`` (parallels ``(1-r)`` at onset).
B_PLASTIC_FIT_DEFORM_FRAC_YIELD_TO_PEAK_F = 0.99


def _apparent_b_iqr_inlier_mask(
    values: list[float],
    multiplier: float = APPARENT_B_IQR_MULTIPLIER,
    *,
    b_min: float = APPARENT_B_MIN,
) -> list[bool]:
    """True for finite entries within ``[max(b_min, Q1 - k IQR), Q3 + k IQR]``; non-finite → False."""
    n = len(values)
    if n == 0:
        return []
    b = np.asarray(values, dtype=float)
    out = [False] * n
    finite_idx = [i for i in range(n) if np.isfinite(b[i])]
    if not finite_idx:
        return out
    bf = b[finite_idx]
    if bf.size < 2:
        for i in finite_idx:
            out[i] = float(b[i]) >= float(b_min)
        return out
    q1, q3 = np.quantile(bf, [0.25, 0.75])
    iqr = float(q3 - q1)
    if not np.isfinite(iqr) or iqr <= 0.0:
        lo, hi = float(np.min(bf)), float(np.max(bf))
    else:
        lo = float(q1) - float(multiplier) * iqr
        hi = float(q3) + float(multiplier) * iqr
    lo = max(float(b_min), lo)
    for i in finite_idx:
        v = float(b[i])
        out[i] = lo <= v <= hi
    return out


def _mask_select_lists(mask: list[bool], *lists: list) -> tuple[list, ...]:
    return tuple([lst[i] for i, keep in enumerate(mask) if keep and i < len(lst)] for lst in lists)


def _filter_by_directional_b_iqr(
    items: list,
    *,
    b_of,
    is_tension_of,
    multiplier: float = APPARENT_B_IQR_MULTIPLIER,
) -> list:
    """Keep items whose ``b`` lies within k*IQR of same-direction peers; preserve input order."""
    if not items:
        return []
    tens = [x for x in items if is_tension_of(x)]
    comp = [x for x in items if not is_tension_of(x)]
    p_mask = _apparent_b_iqr_inlier_mask([float(b_of(x)) for x in tens], multiplier)
    n_mask = _apparent_b_iqr_inlier_mask([float(b_of(x)) for x in comp], multiplier)
    out: list = []
    ti = ci = 0
    for x in items:
        if is_tension_of(x):
            if p_mask[ti]:
                out.append(x)
            ti += 1
        else:
            if n_mask[ci]:
                out.append(x)
            ci += 1
    return out


def _delta_y_hat_inches(fy_ksi: float, E_hat_ksi: float, L_T_in: float) -> float | None:
    """Yield deformation $\\hat{\\delta}_y = (f_y/\\hat{E}) L_T$ [in]; None if invalid."""
    if L_T_in <= 0 or not np.isfinite(L_T_in):
        return None
    if not np.isfinite(fy_ksi) or not np.isfinite(E_hat_ksi) or E_hat_ksi == 0:
        return None
    return float((fy_ksi / E_hat_ksi) * L_T_in)


def _x_plastic_over_deltay_ratio(peak_mag_in: float, Dy: float | None) -> float:
    """$(\\delta-\\hat{\\delta}_y)/\\hat{\\delta}_y$ for peak magnitude $\\delta$ [in]."""
    if Dy is None or not np.isfinite(Dy) or Dy <= 0:
        return float("nan")
    if not np.isfinite(peak_mag_in):
        return float("nan")
    return (float(peak_mag_in) - float(Dy)) / float(Dy)


def _x_amp_over_deltay_ratio(peak_mag_in: float, Dy: float | None) -> float:
    """$\\delta/\\hat{\\delta}_y$ for peak magnitude $\\delta$ [in] (plastic-fit amplitude)."""
    if Dy is None or not np.isfinite(Dy) or Dy <= 0:
        return float("nan")
    if not np.isfinite(peak_mag_in):
        return float("nan")
    return float(peak_mag_in) / float(Dy)


def _last_opposite_peak_index_before(
    points: list[dict],
    bound_idx: int,
    *,
    segment_is_tension: bool,
) -> int | None:
    """
    Latest cycle landmark on the side **opposite** to the current half-cycle, with index
    strictly less than ``bound_idx`` (chronological order along the resampled path).

    Tension ($b_p$): last ``min_def`` before ``bound_idx`` (latest compression peak).
    Compression ($b_n$): last ``max_def`` before ``bound_idx`` (latest tension peak).
    Same ``min_def`` / ``max_def`` convention as ``find_cycle_points`` / zero-to-peak segments.
    """
    if bound_idx <= 0 or not points:
        return None
    sorted_pts = sorted(points, key=lambda p: int(p["idx"]))
    needle = "min_def" if segment_is_tension else "max_def"
    last_j: int | None = None
    for pt in sorted_pts:
        j = int(pt["idx"])
        if j >= bound_idx:
            break
        t = str(pt.get("type", ""))
        if needle in t:
            last_j = j
    return last_j


def _prior_opposite_peak_magnitude(
    u: np.ndarray,
    start_idx: int,
    *,
    segment_is_tension: bool,
    points: list[dict] | None = None,
) -> float:
    """
    Largest opposite-direction deformation magnitude recorded **before** ``start_idx``.

    Prefix ``u[:start_idx]`` is the full resampled history prior to the current half-cycle's
    ``zero_def`` (same ``start_idx`` as segment lists from ``_segments_opposite_peak_to_peak`` or
    ``_segments_zero_to_peak``).

    - **Tension** half-cycle ($b_p$, ``segment_is_tension=True``): opposite direction is compression →
      $\\max(0,\\,-\\min u)$ on the prefix (deepest prior compression).
    - **Compression** half-cycle ($b_n$, ``segment_is_tension=False``): opposite is tension →
      $\\max(0,\\,\\max u)$ on the prefix (largest prior tensile deformation).

    ``points`` is unused (kept for stable call sites). For the **latest** opposite landmark
    before plastic onset (elastic-ray anchor), see ``_last_opposite_peak_index_before`` in
    ``_sigma0_norm_from_plastic_fit`` — that is a different convention.
    """
    _ = points
    if start_idx <= 0:
        return float("nan")
    pref = np.asarray(u[:start_idx], dtype=float)
    pref = pref[np.isfinite(pref)]
    if pref.size == 0:
        return float("nan")
    if segment_is_tension:
        m = float(np.min(pref))
        return float(max(0.0, -m))
    m = float(np.max(pref))
    return float(max(0.0, m))


def _cum_abs_deformation_over_Dy(u: np.ndarray, *, Dy: float) -> np.ndarray:
    """Cumulative $\\sum|\\Delta u|$, normalized by $\\hat{\\delta}_y$ (same as overlay script)."""
    u = np.asarray(u, dtype=float)
    n = int(u.shape[0])
    if n <= 0:
        return np.zeros(0, dtype=float)
    if not np.isfinite(Dy) or Dy == 0:
        return np.full(n, np.nan, dtype=float)
    du = np.diff(u)
    cum = np.concatenate([[0.0], np.cumsum(np.abs(du))])
    return cum / float(Dy)


def _pointwise_inelastic_deformation(delta: np.ndarray, *, delta_y: float) -> np.ndarray:
    """$\\delta_{\\mathrm{inel}} = \\mathrm{sign}(\\delta)\\max(|\\delta|-\\delta_y,0)$."""
    d = np.asarray(delta, dtype=float)
    mag = np.maximum(np.abs(d) - float(delta_y), 0.0)
    return np.sign(d) * mag


def _cum_inelastic_deformation_over_deltay(delta: np.ndarray, *, delta_y: float) -> np.ndarray:
    """$\\sum|\\Delta\\delta_{\\mathrm{inel}}|/\\hat{\\delta}_y$ along the path."""
    d = np.asarray(delta, dtype=float)
    n = int(d.shape[0])
    if n <= 0:
        return np.zeros(0, dtype=float)
    if not np.isfinite(delta_y) or delta_y == 0:
        return np.full(n, np.nan, dtype=float)
    delta_inel = _pointwise_inelastic_deformation(d, delta_y=delta_y)
    d_delta_inel = np.diff(delta_inel)
    cum = np.concatenate([[0.0], np.cumsum(np.abs(d_delta_inel))])
    return cum / float(delta_y)


def _half_cycle_peak_abs_u(u: np.ndarray, start_idx: int, end_idx: int) -> float:
    """Largest ``|u|`` along the resampled path from ``zero_def`` to peak (inclusive indices)."""
    s, e = int(start_idx), int(end_idx)
    n = int(len(u))
    if e < s or s < 0 or e >= n:
        return float("nan")
    seg = np.asarray(u[s : e + 1], dtype=float)
    if seg.size == 0:
        return float("nan")
    if not np.all(np.isfinite(seg)):
        return float("nan")
    return float(np.max(np.abs(seg)))


def _segments_zero_to_peak(points: list[dict]) -> list[tuple[int, int, str]]:
    """
    From points (with idx, type), build segments from zero_def to max_def or min_def.
    Returns list of (start_idx, end_idx, end_type) with end_type in ('max_def', 'min_def').
    Only includes segments with end_idx > start_idx.
    """
    sorted_pts = sorted(points, key=lambda p: p["idx"])
    out: list[tuple[int, int, str]] = []
    last_zero_def: int | None = None
    for pt in sorted_pts:
        idx, t = pt["idx"], pt.get("type", "")
        if "zero_def" in t:
            last_zero_def = idx
        if "max_def" in t and last_zero_def is not None and idx > last_zero_def:
            out.append((last_zero_def, idx, "max_def"))
            last_zero_def = None
        if "min_def" in t and last_zero_def is not None and idx > last_zero_def:
            out.append((last_zero_def, idx, "min_def"))
            last_zero_def = None
    return out


def _segments_opposite_peak_to_peak(points: list[dict]) -> list[tuple[int, int, str]]:
    """
    Branch segments between opposite-direction **amplitude peaks** (no ``zero_def`` cut).

    - Tension branch: ``last min_def`` → ``max_def`` (or index ``0`` → ``max_def`` if no prior
      ``min_def`` yet).
    - Compression branch: ``last max_def`` → ``min_def`` (or ``0`` → ``min_def`` if no prior
      ``max_def``).

    Returns ``(start_idx, end_idx, end_type)`` with ``end_type`` in ``('max_def', 'min_def')`` and
    ``end_idx > start_idx``. Used for apparent ``b`` / secant plastic onset so the fit window is
    not tied to a zero-crossing.
    """
    sorted_pts = sorted(points, key=lambda p: int(p["idx"]))
    out: list[tuple[int, int, str]] = []
    last_min_def: int | None = None
    last_max_def: int | None = None
    for pt in sorted_pts:
        idx = int(pt["idx"])
        t = str(pt.get("type", ""))
        if "max_def" in t:
            a = last_min_def if last_min_def is not None else 0
            if idx > a:
                out.append((a, idx, "max_def"))
            last_max_def = idx
        elif "min_def" in t:
            a = last_max_def if last_max_def is not None else 0
            if idx > a:
                out.append((a, idx, "min_def"))
            last_min_def = idx
    return out


def _elastic_ray_anchor_for_b_plastic_onset(
    u: np.ndarray,
    F: np.ndarray,
    start_idx: int,
    points: list[dict],
    *,
    is_first_path_half_cycle: bool,
    segment_is_tension: bool,
) -> tuple[float, float] | None:
    """
    Unload anchor ``(u_r, F_r)`` at ``start_idx``: first sample of the branch (opposite peak or
    index ``0`` when using ``_segments_opposite_peak_to_peak``; ``zero_def`` when using
    ``_segments_zero_to_peak``).

    ``points``, ``is_first_path_half_cycle``, and ``segment_is_tension`` are unused but kept for
    a stable call signature.

    Elastic: ``F = F_r + k_init (u-u_r)``; secant uses ``B_PLASTIC_ONSET_SECANT_K_FRAC * k_init``.
    """
    _ = points, is_first_path_half_cycle, segment_is_tension
    j = int(start_idx)
    if not (0 <= j < len(u)):
        return None
    ur, fr = float(u[j]), float(F[j])
    if np.isfinite(ur) and np.isfinite(fr):
        return (ur, fr)
    return None


def _plastic_onset_index_yield_deform_frac_to_peak_force(
    u: np.ndarray,
    F: np.ndarray,
    start: int,
    end: int,
    fy_A: float,
    *,
    segment_is_tension: bool,
) -> tuple[int, int, int] | None:
    """
    b-fit geometry on ``[start, end]`` (one opposite-peak-to-peak branch). Returns ``(j, i_y, i_{pk})``:

    1. First index where ``F > f_y A_{sc}`` (C→T) or ``F < -f_y A_{sc}`` (T→C).

    2. On ``[i_{\\mathrm{yield}},\\,\\mathrm{end}]``, first index where ``F`` attains the branch extremum
       after yield (maximum for C→T, minimum for T→C), giving ``u_{\\mathrm{peak\\,F}}``.

    3. Let ``du = u_{\\mathrm{peak\\,F}}-u_{\\mathrm{yield}}`` and ``r =`` ``B_PLASTIC_ONSET_DEFORM_FRAC_REMAINING_TO_PEAK_F``.
       ``u_{\\mathrm{level}} = u_{\\mathrm{yield}} + (1-r)\\,du`` (so a fraction ``r`` of ``|du|`` remains to
       peak-``F`` in deformation). From ``i_{\\mathrm{yield}}`` forward,
       first index whose ``u`` has reached ``u_{\\mathrm{level}}`` (``u \\ge u_{\\mathrm{level}}`` if
       ``du>0``, else ``u \\le u_{\\mathrm{level}}``).

    Returns ``(j, i_{\\mathrm{yield}}, i_{\\mathrm{peak\\,F}})`` or ``None`` if yield is missing,
    ``u_{\\mathrm{peak\\,F}}=u_{\\mathrm{yield}}``, the onset level is never reached, or ``j`` is not strictly
    before ``end``.
    """
    if int(end) < int(start) or not np.isfinite(float(fy_A)) or fy_A <= 0.0:
        return None
    s, e = int(start), int(end)
    fy = float(fy_A)
    rem = float(B_PLASTIC_ONSET_DEFORM_FRAC_REMAINING_TO_PEAK_F)
    if not (0.0 < rem < 1.0) or not np.isfinite(rem):
        return None
    i_yield: int | None = None
    for i in range(s, e + 1):
        fi = float(F[i])
        if not np.isfinite(fi):
            continue
        if segment_is_tension:
            if fi > fy:
                i_yield = i
                break
        else:
            if fi < -fy:
                i_yield = i
                break
    if i_yield is None:
        return None
    u_y = float(u[i_yield])
    if not np.isfinite(u_y):
        return None
    i_peak_f: int | None = None
    f_best: float | None = None
    for i in range(i_yield, e + 1):
        fi = float(F[i])
        if not np.isfinite(fi):
            continue
        if segment_is_tension:
            if f_best is None or fi > f_best:
                f_best = fi
                i_peak_f = i
        else:
            if f_best is None or fi < f_best:
                f_best = fi
                i_peak_f = i
    if i_peak_f is None:
        return None
    u_pf = float(u[i_peak_f])
    if not np.isfinite(u_pf):
        return None
    du = u_pf - u_y
    scale = max(abs(u_y), abs(u_pf), 1.0)
    if abs(du) <= 1e-12 * scale:
        return None
    u_level = u_y + (1.0 - rem) * du
    j: int | None = None
    if du > 0.0:
        for i in range(i_yield, e + 1):
            ui = float(u[i])
            if np.isfinite(ui) and ui >= u_level:
                j = i
                break
    elif du < 0.0:
        for i in range(i_yield, e + 1):
            ui = float(u[i])
            if np.isfinite(ui) and ui <= u_level:
                j = i
                break
    else:
        return None
    if j is None or not (s <= j < e):
        return None
    return (int(j), int(i_yield), int(i_peak_f))


def _b_from_segment(
    u: np.ndarray,
    F: np.ndarray,
    start: int,
    end: int,
    E_hat: float,
    A_sc: float,
    L_T: float,
    f_yc: float,
    L_y: float,
) -> float | None:
    """
    For segment [start, end] (inclusive), local tangent stiffness on **experimental** ``F``;
    first index where ``|k| <= LOW_STIFFNESS_K_FRAC * k_init`` with ``k_init = E_hat*A_sc/L_T``;
    fit F vs u from that index to end. Return b = ksh / kinit, or None if not enough points.
    """
    if end <= start + 1:
        return None
    fy_A = f_yc * A_sc
    if fy_A <= 0:
        return None
    kinit = E_hat * A_sc / L_T if L_T != 0 else 0.0
    if kinit <= 0:
        return None
    u_seg = u[start : end + 1]
    F_seg = F[start : end + 1]
    n_seg = len(u_seg)
    e_half = end + 1
    sl = local_tangent_stiffness_exp(
        u, F, start, e_half, cd_step=LANDMARK_SLOPE_CD_STEP
    )
    mask_seg = low_stiffness_mask_slopes(sl, kinit, frac=LOW_STIFFNESS_K_FRAC)
    i_plastic = None
    for i in range(len(mask_seg)):
        if mask_seg[i]:
            i_plastic = i
            break
    if i_plastic is None or i_plastic >= n_seg - 1:
        return None
    # Fit F = ksh * u + c from i_plastic to end
    u_fit = np.asarray(u_seg[i_plastic:], dtype=float)
    F_fit = np.asarray(F_seg[i_plastic:], dtype=float)
    # Extra guard: ensure |u| exceeds the elastic deformation estimate
    # D_e = F * L_T / (E_hat * A_sc) = F / k_init.
    if kinit > 0:
        de = np.abs(F_fit) / float(kinit)
        m_de = np.isfinite(de) & np.isfinite(u_fit) & (np.abs(u_fit) > de)
        u_fit = u_fit[m_de]
        F_fit = F_fit[m_de]
    if len(u_fit) < 2:
        return None
    min_def_range = MIN_DEF_RANGE_FRAC * L_y if L_y > 0 else MIN_DEF_RANGE_ABS_IN
    if np.abs(u_fit[-1] - u_fit[0]) < min_def_range:
        return None
    # Linear regression: slope = cov(u,F) / var(u)
    cov = np.cov(u_fit, F_fit)[0, 1]
    var_u = np.var(u_fit)
    if var_u <= 0:
        return None
    ksh = cov / var_u
    b = ksh / kinit
    # Clamp to reasonable range (strain hardening ratio typically in [0, 0.05] or so)
    b = float(np.clip(b, 0.0, 0.2))
    return b


def _fit_end_index_yield_peak_u_frac(
    u: np.ndarray,
    *,
    j_onset: int,
    i_yield: int,
    i_peak_f: int,
    seg_end: int,
    frac: float,
) -> int:
    """
    Last index ``i`` in ``[j_{\\mathrm{onset}},\\,\\mathrm{seg\\_end}]`` with ``u[i]`` on the start side of
    ``u_{\\mathrm{yield}} + f\\,(u_{\\mathrm{peak\\,F}}-u_{\\mathrm{yield}})`` (``f = frac``), scanning forward.
    If ``du`` vanishes or ``u`` endpoints are non-finite, returns ``seg_end``. If ``f \\ge 1``, caps at
    ``\\min(i_{\\mathrm{peak\\,F}},\\,\\mathrm{seg\\_end})``.
    """
    j = int(j_onset)
    e = int(seg_end)
    if j > e:
        return e
    uy = float(u[int(i_yield)])
    upf = float(u[int(i_peak_f)])
    if not (np.isfinite(uy) and np.isfinite(upf)):
        return e
    du = upf - uy
    scale = max(abs(uy), abs(upf), 1.0)
    if abs(du) <= 1e-12 * scale:
        return e
    f = float(frac)
    if not np.isfinite(f) or f <= 0.0:
        return j
    if f >= 1.0:
        return int(min(int(i_peak_f), e))
    u_lim = uy + f * du
    last_i = j
    if du > 0.0:
        for i in range(j, e + 1):
            ui = float(u[i])
            if np.isfinite(ui) and ui <= u_lim:
                last_i = i
    else:
        for i in range(j, e + 1):
            ui = float(u[i])
            if np.isfinite(ui) and ui >= u_lim:
                last_i = i
    return int(last_i)


def _plastic_region_fit(
    u: np.ndarray,
    F: np.ndarray,
    start: int,
    end: int,
    E_hat: float,
    A_sc: float,
    L_T: float,
    f_yc: float,
    L_y: float,
    *,
    b_plastic_onset: str = "force_threshold",
    points: list[dict] | None = None,
    is_first_path_half_cycle: bool = False,
    segment_is_tension: bool = False,
) -> dict[str, object] | None:
    """
    Apparent $b$ and overlay line: plastic **onset** from
    ``_plastic_onset_index_yield_deform_frac_to_peak_force`` when ``b_plastic_onset="half_elastic_secant"``,
    else legacy force threshold. Over post-onset samples
    (with the usual ``|u|>|F|/k_{\\mathrm{init}}`` trim for legacy onset only), **linear least squares**
    in $(u,F)$ with the constraint that the line passes through the onset sample
    $(u_0,F_0)=(u[i_{\\mathrm{onset}}],F[i_{\\mathrm{onset}}])$: minimize
    $\\sum_i (F_i - F_0 - k_{\\mathrm{sh}}(u_i-u_0))^2$, so
    $k_{\\mathrm{sh}}=\\sum (u_i-u_0)(F_i-F_0)/\\sum(u_i-u_0)^2$, $b=k_{\\mathrm{sh}}/k_{\\mathrm{init}}$,
    $c=F_0-k_{\\mathrm{sh}}u_0$. For ``half_elastic_secant``, the fit range ends at
    ``_fit_end_index_yield_peak_u_frac`` with ``B_PLASTIC_FIT_DEFORM_FRAC_YIELD_TO_PEAK_F`` along the same
    yield→peak-``F`` ``du`` as onset. Legacy fits to ``end``. Fits are **dropped** if $k_{\\mathrm{sh}}\\le 0$,
    $b\\notin(0,0.2]$, or force signs in the window disagree with ``segment_is_tension``.
    """
    if end <= start + 1:
        return None
    fy_A = f_yc * A_sc
    if fy_A <= 0:
        return None
    kinit = E_hat * A_sc / L_T if L_T != 0 else 0.0
    if kinit <= 0:
        return None
    u_seg = u[start : end + 1]
    F_seg = F[start : end + 1]
    n_seg = len(u_seg)
    thr = PLASTIC_STRESS_RATIO_FY * fy_A
    i_plastic: int | None = None
    onset_skip_m_de = False
    j_on_global: int | None = None
    i_yield_g: int | None = None
    i_peak_f_g: int | None = None
    if b_plastic_onset == "half_elastic_secant":
        tup = _plastic_onset_index_yield_deform_frac_to_peak_force(
            u,
            F,
            int(start),
            int(end),
            fy_A,
            segment_is_tension=bool(segment_is_tension),
        )
        if tup is not None:
            j_on, iy, ipk = tup
            ip = int(j_on - int(start))
            if 0 <= ip < n_seg - 1:
                i_plastic = ip
                onset_skip_m_de = True
                j_on_global = int(j_on)
                i_yield_g = int(iy)
                i_peak_f_g = int(ipk)
    if i_plastic is None:
        for i in range(n_seg):
            fv = float(F_seg[i])
            if np.isfinite(fv) and abs(fv) >= thr:
                i_plastic = i
                break
    # Legacy onset only: advance until |F| > f_y A_sc (yield-based onset already uses yield in rule 1).
    if i_plastic is not None and not onset_skip_m_de:
        j = int(i_plastic)
        while j < n_seg:
            fv = float(F_seg[j])
            if np.isfinite(fv) and abs(fv) > float(fy_A):
                i_plastic = j
                break
            j += 1
        else:
            i_plastic = None
    if i_plastic is None or i_plastic >= n_seg - 1:
        return None
    if (
        onset_skip_m_de
        and j_on_global is not None
        and i_yield_g is not None
        and i_peak_f_g is not None
    ):
        end_fit = int(
            _fit_end_index_yield_peak_u_frac(
                u,
                j_onset=int(j_on_global),
                i_yield=int(i_yield_g),
                i_peak_f=int(i_peak_f_g),
                seg_end=int(end),
                frac=float(B_PLASTIC_FIT_DEFORM_FRAC_YIELD_TO_PEAK_F),
            )
        )
    else:
        end_fit = int(end)
    end_fit = min(int(end), max(int(start), end_fit))
    if end_fit < int(start) + int(i_plastic) + 1:
        return None
    u_seg = u[int(start) : end_fit + 1]
    F_seg = F[int(start) : end_fit + 1]
    n_seg = len(u_seg)
    if i_plastic >= n_seg - 1:
        return None
    u_cand = np.asarray(u_seg[i_plastic:], dtype=float)
    F_cand = np.asarray(F_seg[i_plastic:], dtype=float)
    # Legacy path trims ``|u| <= |F|/k_init`` before fitting. Three-criterion onset uses all post-onset samples.
    if kinit > 0 and not onset_skip_m_de:
        de = np.abs(F_cand) / float(kinit)
        m_de = np.isfinite(de) & np.isfinite(u_cand) & (np.abs(u_cand) > de)
    else:
        m_de = np.ones(len(u_cand), dtype=bool)
    if int(np.count_nonzero(m_de)) < 2:
        return None
    g_seg = np.arange(int(start), end_fit + 1, dtype=int)
    local = np.arange(i_plastic, n_seg, dtype=int)
    idx_fit_globals = g_seg[local][m_de]
    idx_fit_start_global = int(idx_fit_globals[0])
    u_fit_exp = u_cand[m_de]
    F_fit_exp = F_cand[m_de]
    if len(u_fit_exp) < 2:
        return None
    u0 = float(u[idx_fit_start_global])
    F0 = float(F[idx_fit_start_global])
    if not (np.isfinite(u0) and np.isfinite(F0)):
        return None
    min_def_range = MIN_DEF_RANGE_FRAC * L_y if L_y > 0 else MIN_DEF_RANGE_ABS_IN
    if np.abs(float(u_fit_exp[-1]) - float(u_fit_exp[0])) < min_def_range:
        return None
    du = np.asarray(u_fit_exp, dtype=float) - u0
    dF = np.asarray(F_fit_exp, dtype=float) - F0
    m = np.isfinite(du) & np.isfinite(dF)
    du = du[m]
    dF = dF[m]
    denom = float(np.dot(du, du))
    if denom <= 0.0 or not np.isfinite(denom):
        return None
    ksh = float(np.dot(du, dF) / denom)
    if not np.isfinite(ksh) or ksh <= 0.0:
        return None
    b = float(ksh / float(kinit))
    if not np.isfinite(b) or b <= 0.0 or b > 0.2:
        return None
    ksh_plot = float(ksh)
    c = float(F0 - ksh_plot * u0)
    u_fit = np.asarray(u_fit_exp, dtype=float)
    F_fit = np.asarray(F_fit_exp, dtype=float)
    F_fitted = ksh_plot * u_fit + c
    amp = float(np.max(np.abs(u_fit)))
    mF = float(np.mean(F_fit))
    f0 = float(F_fit[0])
    if segment_is_tension:
        if mF <= 0.0 or f0 <= 0.0:
            return None
    else:
        if mF >= 0.0 or f0 >= 0.0:
            return None
    is_tension = bool(np.mean(F_fit) >= 0.0)
    if is_tension != segment_is_tension:
        return None
    return {
        "b": b,
        "u_fit": u_fit,
        "F_fit": F_fit,
        "F_fitted": F_fitted,
        "is_tension": is_tension,
        "amp": amp,
        "ksh_plot": float(ksh_plot),
        "c": float(c),
        "kinit": float(kinit),
        "idx_fit_start_global": idx_fit_start_global,
        "fy_A": float(fy_A),
        "f_yc": float(f_yc),
        "A_sc": float(A_sc),
    }


def _segment_line_data(
    u: np.ndarray,
    F: np.ndarray,
    start: int,
    end: int,
    E_hat: float,
    A_sc: float,
    L_T: float,
    f_yc: float,
    L_y: float,
    *,
    b_plastic_onset: str = "force_threshold",
    points: list[dict] | None = None,
    is_first_path_half_cycle: bool = False,
    segment_is_tension: bool = False,
) -> tuple[float, np.ndarray, np.ndarray, bool, float, int] | None:
    """
    Same as _b_from_segment but also return ``(b, u_fit, F_fitted, ...)`` for the onset-anchored
    linear fit on the post-onset window (see ``_plastic_region_fit``).
    ``idx_fit_start_global`` is the plastic-onset index on the experimental path.
    """
    r = _plastic_region_fit(
        u,
        F,
        start,
        end,
        E_hat,
        A_sc,
        L_T,
        f_yc,
        L_y,
        b_plastic_onset=b_plastic_onset,
        points=points,
        is_first_path_half_cycle=is_first_path_half_cycle,
        segment_is_tension=segment_is_tension,
    )
    if r is None:
        return None
    return (
        float(r["b"]),
        np.asarray(r["u_fit"], dtype=float),
        np.asarray(r["F_fitted"], dtype=float),
        bool(r["is_tension"]),
        float(r["amp"]),
        int(r["idx_fit_start_global"]),
    )


def _sigma0_norm_from_plastic_fit(
    u: np.ndarray,
    F: np.ndarray,
    fit: dict[str, object],
    points: list[dict],
) -> tuple[float, dict[str, float]]:
    """
    Intersection of this segment's plastic line $F=k_{\\mathrm{sh}}u+c$ (extended as needed)
    with the elastic asymptote $F=F_r+k_{\\mathrm{init}}(u-u_r)$ anchored at the **latest**
    opposite-side **cycle peak** before plastic onset: last ``min_def`` landmark before
    $i_{\\mathrm{plastic\\,start}}$ for tension fits, last ``max_def`` for compression
    (same peaks as zero-to-peak / landmark logic). If no such landmark exists, falls back to
    the global extremum of $u$ on $[0,\\,i_{\\mathrm{plastic\\,start}})$.

    Returns ``(F^*/(f_y A_{sc}), debug scalars)``; norm is NaN if undefined.
    """
    idx0 = int(fit["idx_fit_start_global"])
    if idx0 <= 0:
        return (float("nan"), {})
    is_tension = bool(fit["is_tension"])
    u_pre = np.asarray(u[:idx0], dtype=float)
    if u_pre.size == 0:
        return (float("nan"), {})
    j_lm = _last_opposite_peak_index_before(points, idx0, segment_is_tension=is_tension)
    if j_lm is not None and 0 <= j_lm < len(u):
        j = int(j_lm)
    else:
        j = int(np.argmin(u_pre)) if is_tension else int(np.argmax(u_pre))
    u_r = float(u[j])
    F_r = float(F[j])
    if not (np.isfinite(u_r) and np.isfinite(F_r)):
        return (float("nan"), {})
    ksh = float(fit["ksh_plot"])
    c = float(fit["c"])
    kinit = float(fit["kinit"])
    fy_a = float(fit["fy_A"])
    denom = ksh - kinit
    tol = _SIG0_K_PARALLEL_TOL_FRAC * max(abs(kinit), 1.0)
    if abs(denom) < tol or not np.isfinite(denom):
        return (float("nan"), {})
    u_star = (F_r - kinit * u_r - c) / denom
    F_star = ksh * u_star + c
    if not (np.isfinite(u_star) and np.isfinite(F_star)) or fy_a <= 0:
        return (float("nan"), {})
    sig0 = float(F_star / fy_a)
    dbg = {
        "u_r": u_r,
        "F_r": F_r,
        "u_star": float(u_star),
        "F_star": float(F_star),
        "rev_idx": int(j),
    }
    return (sig0, dbg)


def _global_u_extrema_indices(u: np.ndarray) -> tuple[int | None, int | None]:
    """``(j_min, j_max)`` for finite samples: max compressive deformation, max tensile deformation."""
    u_arr = np.asarray(u, dtype=float)
    if u_arr.size == 0:
        return (None, None)
    mask = np.isfinite(u_arr)
    if not np.any(mask):
        return (None, None)
    u_masked = np.where(mask, u_arr, np.nan)
    j_min = int(np.nanargmin(u_masked))
    j_max = int(np.nanargmax(u_masked))
    return (j_min, j_max)


def _equiv_sigma0_intersection_dbg(
    fit: dict[str, object],
    u_r: float,
    F_r: float,
    rev_idx: int,
) -> tuple[float, dict[str, float]]:
    """
    $F^*/(f_y A_{sc})$ and intersection geometry for the plastic line vs elastic asymptote from a
    **fixed** anchor (global max compressive or max tensile point). Empty ``dbg`` if undefined.
    """
    if not (np.isfinite(u_r) and np.isfinite(F_r)):
        return (float("nan"), {})
    ksh = float(fit["ksh_plot"])
    c = float(fit["c"])
    kinit = float(fit["kinit"])
    fy_a = float(fit["fy_A"])
    denom = ksh - kinit
    tol = _SIG0_K_PARALLEL_TOL_FRAC * max(abs(kinit), 1.0)
    if abs(denom) < tol or not np.isfinite(denom):
        return (float("nan"), {})
    u_star = (F_r - kinit * u_r - c) / denom
    F_star = ksh * u_star + c
    if not (np.isfinite(u_star) and np.isfinite(F_star)) or fy_a <= 0:
        return (float("nan"), {})
    sig0 = float(F_star / fy_a)
    dbg = {
        "u_r": u_r,
        "F_r": F_r,
        "u_star": float(u_star),
        "F_star": float(F_star),
        "rev_idx": int(rev_idx),
    }
    return (sig0, dbg)


def _equiv_sigma0_norm_from_plastic_fit(
    fit: dict[str, object],
    u_r: float,
    F_r: float,
) -> float:
    """
    $F^*/(f_y A_{sc})$ only; same geometry as ``_equiv_sigma0_intersection_dbg``.
    """
    s, _ = _equiv_sigma0_intersection_dbg(fit, u_r, F_r, -1)
    return s


def _origin_elastic_plastic_intersection_dbg(
    fit: dict[str, object],
) -> tuple[float, dict[str, float]]:
    """
    Intersection of the apparent hardening line with the **initial stiffness** line through the origin.

    - Elastic: $F = k_{\\mathrm{init}} u$ with $k_{\\mathrm{init}} = \\hat{E} A_{sc}/L_T$ (stored as ``kinit`` by
      ``_plastic_region_fit``, same as in $b=k_{\\mathrm{sh}}/k_{\\mathrm{init}}$).
    - Plastic (apparent $b$ line): $F = (b\\,k_{\\mathrm{init}})u + c$ using ``ksh_plot`` and ``c`` from that fit
      (``ksh_plot`` is $b$ times $k_{\\mathrm{init}}$ after the usual $b$ clip).

    Returns $(F^*/(f_y A_{sc}),\\,\\texttt{dbg})$ with ``u_star``, ``F_star``; empty ``dbg`` if parallel or invalid.
    """
    # kinit = E_hat * A_sc / L_T; ksh_plot = b * kinit (apparent post-yield slope in force–deformation space)
    ksh = float(fit["ksh_plot"])
    c = float(fit["c"])
    kinit = float(fit["kinit"])
    fy_a = float(fit["fy_A"])
    denom = kinit - ksh
    tol = _SIG0_K_PARALLEL_TOL_FRAC * max(abs(kinit), 1.0)
    if abs(denom) < tol or not np.isfinite(denom):
        return (float("nan"), {})
    u_star = c / denom
    F_star = kinit * u_star
    if not (np.isfinite(u_star) and np.isfinite(F_star)) or fy_a <= 0:
        return (float("nan"), {})
    sig = float(F_star / fy_a)
    dbg = {"u_star": float(u_star), "F_star": float(F_star)}
    return (sig, dbg)


def _get_b_lists(
    u: np.ndarray,
    F: np.ndarray,
    n: int,
    points: list[dict],
    E_hat: float,
    A_sc: float,
    L_T: float,
    L_y: float,
    fy: float,
) -> tuple[
    list[float],
    list[float],
    list[float],
    list[float],
    list[int],
    list[int],
    list[int],
    list[int],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
]:
    """
    Return (b_p_list, b_n_list, amp_p_list, amp_n_list, end_idx_p, end_idx_n, start_idx_p, start_idx_n,
    stress_norm_p_list, stress_norm_n_list, sigma0_norm_p_list, sigma0_norm_n_list,
    equiv_sigma0_norm_p_list, equiv_sigma0_norm_n_list,
    amp_peak_p_list, amp_peak_n_list).

    ``amp_peak_*`` are ``max(|u|)`` over the full opposite-peak-to-peak branch (same branches as the parallel
    ``b_*`` / plastic-window ``amp_*`` entries). Used to pick ``b`` at the **largest excursion** branch;
    ``amp_*`` remain plastic-fit-window amplitudes for weighted means and plots.

    Skips segments only when the plastic-line fit is unavailable (e.g. too small def range in the fit window).
    Every valid half-cycle is fitted; then ``b_p`` / ``b_n`` entries outside
    ``[max(0, Q1 - k IQR), Q3 + k IQR]`` (``k =`` ``APPARENT_B_IQR_MULTIPLIER``) are dropped separately by direction.

    Amplitudes are max(|u|) over the fitted hardening region of each segment.
    ``end_idx_*`` / ``start_idx_*`` are indices into ``u`` for each kept segment (opposite peak → peak).
    ``stress_norm_*`` lists hold $P/(f_y A_{sc})$ at the intersection of each apparent $b$-line
    $F=(b\\,k_{\\mathrm{init}})u+c$ with the origin elastic line $F=k_{\\mathrm{init}}u$,
    $k_{\\mathrm{init}}=\\hat{E}A_{sc}/L_T$ (see ``_origin_elastic_plastic_intersection_dbg``).
    ``sigma0_norm_*`` lists hold $\\sigma_0/f_y = F^*/(f_y A_{sc})$ at the plastic / elastic asymptote
    intersection for that segment's $b$-fit (see ``_sigma0_norm_from_plastic_fit``): one value per kept fit,
    or NaN when the intersection is undefined.
    ``equiv_sigma0_norm_*`` lists hold the same normalization using the elastic asymptote from **global**
    max compressive deformation (for tension $b$-fits) or global max tensile deformation (for compression
    $b$-fits), intersected with each plastic line (see ``_equiv_sigma0_norm_from_plastic_fit``).
    """
    segments = _segments_opposite_peak_to_peak(points)
    j_min_g, j_max_g = _global_u_extrema_indices(u)
    u_at_min = float(u[j_min_g]) if j_min_g is not None else float("nan")
    F_at_min = float(F[j_min_g]) if j_min_g is not None else float("nan")
    u_at_max = float(u[j_max_g]) if j_max_g is not None else float("nan")
    F_at_max = float(F[j_max_g]) if j_max_g is not None else float("nan")
    b_p_list: list[float] = []
    b_n_list: list[float] = []
    amp_p_list: list[float] = []
    amp_n_list: list[float] = []
    end_p_list: list[int] = []
    end_n_list: list[int] = []
    start_p_list: list[int] = []
    start_n_list: list[int] = []
    stress_p_list: list[float] = []
    stress_n_list: list[float] = []
    sigma0_p_list: list[float] = []
    sigma0_n_list: list[float] = []
    equiv_sigma0_p_list: list[float] = []
    equiv_sigma0_n_list: list[float] = []
    amp_peak_p_list: list[float] = []
    amp_peak_n_list: list[float] = []
    for seg_i, (start_idx, end_idx, end_type) in enumerate(segments):
        if end_idx >= n or start_idx < 0:
            continue
        is_tension_seg = "max_def" in str(end_type)
        fit = _plastic_region_fit(
            u,
            F,
            start_idx,
            end_idx,
            E_hat,
            A_sc,
            L_T,
            fy,
            L_y,
            b_plastic_onset="half_elastic_secant",
            points=points,
            is_first_path_half_cycle=(seg_i == 0),
            segment_is_tension=is_tension_seg,
        )
        if fit is None:
            continue
        b = float(fit["b"])
        u_fit = np.asarray(fit["u_fit"], dtype=float)
        is_tension = bool(fit["is_tension"])
        amp = float(fit["amp"])
        amp_peak = _half_cycle_peak_abs_u(u, start_idx, end_idx)
        sig0_norm, _ = _sigma0_norm_from_plastic_fit(u, F, fit, points)
        if is_tension:
            eq_sig = _equiv_sigma0_norm_from_plastic_fit(fit, u_at_min, F_at_min)
        else:
            eq_sig = _equiv_sigma0_norm_from_plastic_fit(fit, u_at_max, F_at_max)
        p_norm, _ = _origin_elastic_plastic_intersection_dbg(fit)
        if is_tension:
            b_p_list.append(b)
            amp_p_list.append(amp)
            amp_peak_p_list.append(amp_peak)
            end_p_list.append(int(end_idx))
            start_p_list.append(int(start_idx))
            stress_p_list.append(p_norm)
            sigma0_p_list.append(sig0_norm)
            equiv_sigma0_p_list.append(eq_sig)
        else:
            b_n_list.append(b)
            amp_n_list.append(amp)
            amp_peak_n_list.append(amp_peak)
            end_n_list.append(int(end_idx))
            start_n_list.append(int(start_idx))
            stress_n_list.append(p_norm)
            sigma0_n_list.append(sig0_norm)
            equiv_sigma0_n_list.append(eq_sig)
    p_mask = _apparent_b_iqr_inlier_mask(b_p_list)
    n_mask = _apparent_b_iqr_inlier_mask(b_n_list)
    (
        b_p_list,
        amp_p_list,
        end_p_list,
        start_p_list,
        stress_p_list,
        sigma0_p_list,
        equiv_sigma0_p_list,
        amp_peak_p_list,
    ) = _mask_select_lists(
        p_mask,
        b_p_list,
        amp_p_list,
        end_p_list,
        start_p_list,
        stress_p_list,
        sigma0_p_list,
        equiv_sigma0_p_list,
        amp_peak_p_list,
    )
    (
        b_n_list,
        amp_n_list,
        end_n_list,
        start_n_list,
        stress_n_list,
        sigma0_n_list,
        equiv_sigma0_n_list,
        amp_peak_n_list,
    ) = _mask_select_lists(
        n_mask,
        b_n_list,
        amp_n_list,
        end_n_list,
        start_n_list,
        stress_n_list,
        sigma0_n_list,
        equiv_sigma0_n_list,
        amp_peak_n_list,
    )
    return (
        b_p_list,
        b_n_list,
        amp_p_list,
        amp_n_list,
        end_p_list,
        end_n_list,
        start_p_list,
        start_n_list,
        stress_p_list,
        stress_n_list,
        sigma0_p_list,
        sigma0_n_list,
        equiv_sigma0_p_list,
        equiv_sigma0_n_list,
        amp_peak_p_list,
        amp_peak_n_list,
    )


def iter_sig0_overlay_segments(
    u: np.ndarray,
    F: np.ndarray,
    n: int,
    points: list[dict],
    E_hat: float,
    A_sc: float,
    L_T: float,
    L_y: float,
    fy: float,
) -> list[dict[str, object]]:
    """
    One dict per kept half-cycle for diagnostic overlays: plastic fit segment, backward-extended plastic line,
    elastic line from prior opposite peak, $\\sigma_0$ / $\\sigma_0^{\\mathrm{eq}}$, and plastic $\\cap$
    origin elastic ($F=k_{\\mathrm{init}}u$) for ``stress_norm_origin`` / ``dbg_origin``. Physical units: u [in], F [kip].
    """
    out: list[dict[str, object]] = []
    segments = _segments_opposite_peak_to_peak(points)
    j_min_g, j_max_g = _global_u_extrema_indices(u)
    for seg_i, (start_idx, end_idx, end_type) in enumerate(segments):
        if end_idx >= n or start_idx < 0:
            continue
        is_tension_seg = "max_def" in str(end_type)
        fit = _plastic_region_fit(
            u,
            F,
            start_idx,
            end_idx,
            E_hat,
            A_sc,
            L_T,
            fy,
            L_y,
            b_plastic_onset="half_elastic_secant",
            points=points,
            is_first_path_half_cycle=(seg_i == 0),
            segment_is_tension=is_tension_seg,
        )
        if fit is None:
            continue
        u_fit = np.asarray(fit["u_fit"], dtype=float)
        is_tension = bool(fit["is_tension"])
        sig0_norm, dbg = _sigma0_norm_from_plastic_fit(u, F, fit, points)
        j_eq = j_min_g if is_tension else j_max_g
        dbg_equiv: dict[str, float] = {}
        sig0_equiv_norm = float("nan")
        if j_eq is not None and 0 <= j_eq < n:
            sig0_equiv_norm, dbg_equiv = _equiv_sigma0_intersection_dbg(
                fit, float(u[j_eq]), float(F[j_eq]), int(j_eq)
            )
        stress_origin_norm, dbg_origin = _origin_elastic_plastic_intersection_dbg(fit)
        ksh = float(fit["ksh_plot"])
        c = float(fit["c"])
        kinit = float(fit["kinit"])
        F_fit_line = ksh * u_fit + c
        pts_u = [float(np.min(u_fit)), float(np.max(u_fit))]
        if dbg:
            pts_u.append(float(dbg["u_r"]))
            pts_u.append(float(dbg["u_star"]))
        if dbg_equiv:
            pts_u.append(float(dbg_equiv["u_star"]))
        if dbg_origin:
            pts_u.append(float(dbg_origin["u_star"]))
        if not dbg and not dbg_equiv and not dbg_origin:
            pts_u.append(float(u[int(start_idx)]))
            pts_u.append(float(u[int(end_idx)]))
        span = max(pts_u) - min(pts_u)
        pad = 0.06 * (span + 1e-6)
        u_lo = min(pts_u) - pad
        u_hi = max(pts_u) + pad
        u_long = np.linspace(u_lo, u_hi, 120)
        F_pl_long = ksh * u_long + c
        if dbg:
            u_r = float(dbg["u_r"])
            F_r = float(dbg["F_r"])
            F_el_long = F_r + kinit * (u_long - u_r)
        else:
            F_el_long = np.full_like(u_long, np.nan, dtype=float)
        out.append(
            {
                "is_tension": is_tension,
                "sigma0_norm": float(sig0_norm),
                "sigma0_equiv_norm": float(sig0_equiv_norm),
                "stress_norm_origin": float(stress_origin_norm),
                "u_fit": u_fit,
                "F_fit_line": F_fit_line,
                "u_plastic_long": u_long,
                "F_plastic_long": F_pl_long,
                "u_elastic_long": u_long,
                "F_elastic_long": F_el_long,
                "dbg": dbg,
                "dbg_equiv": dbg_equiv,
                "dbg_origin": dbg_origin,
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
                "b": float(fit["b"]),
            }
        )
    return _filter_by_directional_b_iqr(
        out,
        b_of=lambda seg: float(seg["b"]),
        is_tension_of=lambda seg: bool(seg["is_tension"]),
    )


def get_sig0_overlay_segments_one_specimen(specimen_id: str) -> list[dict[str, object]] | None:
    """Resampled path-ordered segments with overlay geometry for ``plot_sig0_slopes``; ``None`` if no data."""
    resampled_path = resolve_resampled_force_deformation_csv(specimen_id, _PROJECT_ROOT)
    if resampled_path is None or not resampled_path.is_file():
        return None
    catalog = read_catalog(CATALOG_PATH)
    row = catalog[catalog["Name"].astype(str) == specimen_id]
    if row.empty:
        return None
    row = row.iloc[0]
    L_T = float(row["L_T"])
    L_y = float(row["L_y"])
    A_sc = float(row["A_sc"])
    A_t = float(row["A_t"])
    fy = float(row["fyp"])
    df = pd.read_csv(resampled_path)
    if "Force[kip]" not in df.columns or "Deformation[in]" not in df.columns:
        return None
    u = df["Deformation[in]"].values.astype(float, copy=False)
    F = df["Force[kip]"].values.astype(float, copy=False)
    n = len(u)
    loaded = load_cycle_points_resampled(specimen_id)
    if loaded is None:
        points, _ = find_cycle_points(df)
    else:
        points, _ = loaded
    Q = compute_Q(L_T, L_y, A_sc, A_t)
    E_hat = Q * E_ksi
    return iter_sig0_overlay_segments(u, F, n, points, E_hat, A_sc, L_T, L_y, fy)


def get_b_and_amplitude_lists_one_specimen(
    specimen_id: str,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """
    Load resampled F-u and cycle points; return (b_n_list, b_p_list, amp_n_list, amp_p_list).

    Amplitudes match ``_segment_line_data``: max ``|u|`` over the plastic hardening fit window [in].
    Lists are parallel within tension vs compression; order follows opposite-peak-to-peak branches along the path.
    """
    resampled_path = resolve_resampled_force_deformation_csv(specimen_id, _PROJECT_ROOT)
    if resampled_path is None or not resampled_path.is_file():
        return ([], [], [], [])
    catalog = read_catalog(CATALOG_PATH)
    row = catalog[catalog["Name"].astype(str) == specimen_id]
    if row.empty:
        return ([], [], [], [])
    row = row.iloc[0]
    L_T = float(row["L_T"])
    L_y = float(row["L_y"])
    A_sc = float(row["A_sc"])
    A_t = float(row["A_t"])
    fy = float(row["fyp"])
    df = pd.read_csv(resampled_path)
    if "Force[kip]" not in df.columns or "Deformation[in]" not in df.columns:
        return ([], [], [], [])
    u = df["Deformation[in]"].values
    F = df["Force[kip]"].values
    n = len(u)
    loaded = load_cycle_points_resampled(specimen_id)
    if loaded is None:
        points, _ = find_cycle_points(df)
    else:
        points, _ = loaded
    Q = compute_Q(L_T, L_y, A_sc, A_t)
    E_hat = Q * E_ksi
    t = _get_b_lists(u, F, n, points, E_hat, A_sc, L_T, L_y, fy)
    return (t[1], t[0], t[3], t[2])  # b_n, b_p, amp_n, amp_p


def get_b_segment_scatter_metrics_one_specimen(specimen_id: str) -> dict[str, object] | None:
    """
    Segment-level $b_n$ / $b_p$ with parallel abscissas for scatter plots.

    Returns ``None`` if no resampled CSV. Otherwise a dict with list keys
    ``b_n``, ``b_p``, ``x_amp_fit_ratio_n``, ``x_amp_fit_ratio_p`` (plastic-window
    amplitude $\\delta_c^{\\max}/\\hat{\\delta}_y$, matching ``bn_bp_vs_cycle_amplitude_norm_hat_delta_y``),
    ``x_plastic_ratio_n`` / ``_p`` ($(\\delta_c^{\\max}-\\hat{\\delta}_y)/\\hat{\\delta}_y$
    at segment peak, $\\hat{\\delta}_y=(f_y/\\hat{E})L_T$),
    ``x_plastic_opp_ratio_n`` / ``_p`` (same normalization using the largest prior opposite-direction
    excursion before that half-cycle),
    ``x_cum_def_ratio_n`` / ``_p`` ($\\sum|\\Delta\\delta|/\\hat{\\delta}_y$ at peak),
    ``x_cum_inel_ratio_n`` / ``_p`` ($\\sum|\\Delta\\delta_{\\mathrm{inel}}|/\\hat{\\delta}_y$),
    ``stress_norm_origin_n`` / ``stress_norm_origin_p`` ($P/(f_y A_{sc})$ at plastic line $\\cap$ $F=k_{\\mathrm{init}}u$ through the origin; same $k_{\\mathrm{init}}$ as $\\sigma_0$ / $\\sigma_0^{\\mathrm{eq}}$ but anchored at $(u,F)=(0,0)$), and
    ``sigma0_norm_n`` / ``sigma0_norm_p`` ($\\sigma_0/f_y$ from plastic / elastic asymptote intersection), and
    ``equiv_sigma0_norm_n`` / ``equiv_sigma0_norm_p`` (same plastic-line intersection with the elastic
    asymptote from **global** max compressive / max tensile deformation).
    """
    resampled_path = resolve_resampled_force_deformation_csv(specimen_id, _PROJECT_ROOT)
    if resampled_path is None or not resampled_path.is_file():
        return None
    catalog = read_catalog(CATALOG_PATH)
    row = catalog[catalog["Name"].astype(str) == specimen_id]
    if row.empty:
        return None
    row = row.iloc[0]
    L_T = float(row["L_T"])
    L_y = float(row["L_y"])
    A_sc = float(row["A_sc"])
    A_t = float(row["A_t"])
    fy = float(row["fyp"])
    df = pd.read_csv(resampled_path)
    if "Force[kip]" not in df.columns or "Deformation[in]" not in df.columns:
        return None
    u = df["Deformation[in]"].values.astype(float, copy=False)
    F = df["Force[kip]"].values.astype(float, copy=False)
    n = len(u)
    loaded = load_cycle_points_resampled(specimen_id)
    if loaded is None:
        points, _ = find_cycle_points(df)
    else:
        points, _ = loaded
    Q = compute_Q(L_T, L_y, A_sc, A_t)
    E_hat = Q * E_ksi
    Dy = _delta_y_hat_inches(fy, E_hat, L_T)
    cum_abs = _cum_abs_deformation_over_Dy(u, Dy=Dy if Dy is not None else float("nan"))
    cum_inel = _cum_inelastic_deformation_over_deltay(
        u, delta_y=Dy if Dy is not None else float("nan")
    )

    (
        b_p_list,
        b_n_list,
        amp_p_list,
        amp_n_list,
        end_p_list,
        end_n_list,
        start_p_list,
        start_n_list,
        stress_norm_p_list,
        stress_norm_n_list,
        sigma0_norm_p_list,
        sigma0_norm_n_list,
        equiv_sigma0_norm_p_list,
        equiv_sigma0_norm_n_list,
        _amp_peak_p_unused,
        _amp_peak_n_unused,
    ) = _get_b_lists(u, F, n, points, E_hat, A_sc, L_T, L_y, fy)

    def xs_at_ends(ends: list[int]) -> tuple[list[float], list[float], list[float]]:
        x_pl: list[float] = []
        x_cd: list[float] = []
        x_ci: list[float] = []
        for j in ends:
            if j < 0 or j >= n:
                x_pl.append(float("nan"))
                x_cd.append(float("nan"))
                x_ci.append(float("nan"))
                continue
            um = abs(float(u[j]))
            x_pl.append(_x_plastic_over_deltay_ratio(um, Dy))
            x_cd.append(float(cum_abs[j]) if j < len(cum_abs) else float("nan"))
            x_ci.append(float(cum_inel[j]) if j < len(cum_inel) else float("nan"))
        return (x_pl, x_cd, x_ci)

    def xs_opposite_prior(starts: list[int], *, segment_is_tension: bool) -> list[float]:
        out: list[float] = []
        for st in starts:
            dmag = _prior_opposite_peak_magnitude(
                u, int(st), segment_is_tension=segment_is_tension, points=points
            )
            out.append(_x_plastic_over_deltay_ratio(dmag, Dy))
        return out

    x_pl_n, x_cd_n, x_ci_n = xs_at_ends(end_n_list)
    x_pl_p, x_cd_p, x_ci_p = xs_at_ends(end_p_list)
    x_pl_opp_n = xs_opposite_prior(start_n_list, segment_is_tension=False)
    x_pl_opp_p = xs_opposite_prior(start_p_list, segment_is_tension=True)
    x_amp_fit_n = [_x_amp_over_deltay_ratio(float(a), Dy) for a in amp_n_list]
    x_amp_fit_p = [_x_amp_over_deltay_ratio(float(a), Dy) for a in amp_p_list]

    return {
        "b_n": b_n_list,
        "b_p": b_p_list,
        "stress_norm_origin_n": stress_norm_n_list,
        "stress_norm_origin_p": stress_norm_p_list,
        "sigma0_norm_n": sigma0_norm_n_list,
        "sigma0_norm_p": sigma0_norm_p_list,
        "equiv_sigma0_norm_n": equiv_sigma0_norm_n_list,
        "equiv_sigma0_norm_p": equiv_sigma0_norm_p_list,
        "x_amp_fit_ratio_n": x_amp_fit_n,
        "x_amp_fit_ratio_p": x_amp_fit_p,
        "x_plastic_ratio_n": x_pl_n,
        "x_plastic_ratio_p": x_pl_p,
        "x_plastic_opp_ratio_n": x_pl_opp_n,
        "x_plastic_opp_ratio_p": x_pl_opp_p,
        "x_cum_def_ratio_n": x_cd_n,
        "x_cum_def_ratio_p": x_cd_p,
        "x_cum_inel_ratio_n": x_ci_n,
        "x_cum_inel_ratio_p": x_ci_p,
    }


def get_unordered_envelope_xmetrics_one_specimen(
    specimen_id: str,
    *,
    project_root: Path,
    catalog: pd.DataFrame,
) -> dict[str, float] | None:
    """
    Single-point abscissas for extended B-vs-$x$ plots (unordered cloud): values at
    $\\arg\\max|\\delta|$ along the resolved experimental record. Prior-opposite-peak plastic uses the
    same prefix rule as path-ordered segments: ``_prior_opposite_peak_magnitude(u,\\,i_{\\max|u|},\\,\\cdot)``
    for compression vs tension, then ``_x_plastic_over_deltay_ratio`` (so one $x$ per panel matches the
    envelope $b_n$ / $b_p$ at the outer-loop strain point).
    """
    row = catalog[catalog["Name"].astype(str) == str(specimen_id)]
    if row.empty:
        return None
    row = row.iloc[0]
    L_T = float(row["L_T"])
    L_y = float(row["L_y"])
    A_sc = float(row["A_sc"])
    A_t = float(row["A_t"])
    fy = float(row["fyp"])
    path = resolve_force_deformation_csv_for_max_strain(str(specimen_id), catalog, project_root=project_root)
    if path is None or not path.is_file():
        return None
    try:
        df = pd.read_csv(path, usecols=[DEF_COL])
    except ValueError:
        df = pd.read_csv(path)
        if DEF_COL not in df.columns:
            return None
    u = pd.to_numeric(df[DEF_COL], errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(u)
    u = u[m]
    if u.size == 0:
        return None
    if L_y <= 0 or not np.isfinite(L_y):
        return None
    Q = compute_Q(L_T, L_y, A_sc, A_t)
    E_hat = Q * E_ksi
    Dy = _delta_y_hat_inches(fy, E_hat, L_T)
    idx = int(np.argmax(np.abs(u)))
    um = float(np.abs(u[idx]))
    cum_abs = _cum_abs_deformation_over_Dy(u, Dy=Dy if Dy is not None else float("nan"))
    cum_inel = _cum_inelastic_deformation_over_deltay(
        u, delta_y=Dy if Dy is not None else float("nan")
    )
    x_amp = _x_amp_over_deltay_ratio(um, Dy)
    if Dy is not None and np.isfinite(Dy):
        x_plastic = _x_plastic_over_deltay_ratio(um, Dy)
        x_cd = float(cum_abs[idx]) if idx < len(cum_abs) else float("nan")
        x_ci = float(cum_inel[idx]) if idx < len(cum_inel) else float("nan")
        dmag_n = _prior_opposite_peak_magnitude(
            u, idx, segment_is_tension=False, points=None
        )
        dmag_p = _prior_opposite_peak_magnitude(
            u, idx, segment_is_tension=True, points=None
        )
        x_opp_n = _x_plastic_over_deltay_ratio(dmag_n, Dy)
        x_opp_p = _x_plastic_over_deltay_ratio(dmag_p, Dy)
    else:
        x_plastic = float("nan")
        x_cd = float("nan")
        x_ci = float("nan")
        x_opp_n = float("nan")
        x_opp_p = float("nan")
    return {
        "x_amp_fit_ratio": x_amp,
        "x_plastic_ratio": x_plastic,
        "x_plastic_opp_ratio_n": float(x_opp_n),
        "x_plastic_opp_ratio_p": float(x_opp_p),
        "x_cum_def_ratio": x_cd,
        "x_cum_inel_ratio": x_ci,
    }


def get_b_lists_one_specimen(specimen_id: str) -> tuple[list[float], list[float]]:
    """Load resampled F-u and cycle points; return (b_n_list, b_p_list). Empty if no resampled CSV."""
    bn, bp, _, _ = get_b_and_amplitude_lists_one_specimen(specimen_id)
    return (bn, bp)


def extract_bn_bp_one_specimen(
    specimen_id: str,
    df: pd.DataFrame,
    points: list[dict],
    L_T: float,
    L_y: float,
    A_sc: float,
    A_t: float,
    fy: float,
) -> dict:
    """
    Compute Q, E_hat, and b_n / b_p (mean and median) for one specimen.
    Returns dict with catalog fields plus Q, E_hat, L_y_sqrt_A_sc, fy_over_E,
    fy_over_E_hat, and segment-wise ``b_n`` / ``b_p`` stats (mean, median, weighted_mean,
    q1, q3, min, max, max_amplitude).
    """
    Q = compute_Q(L_T, L_y, A_sc, A_t)
    E_hat = Q * E_ksi
    kinit = E_hat * A_sc / L_T if L_T > 0 else 0.0

    u = df["Deformation[in]"].values
    F = df["Force[kip]"].values
    n = len(u)
    if n == 0:
        return {}

    (
        b_p_list,
        b_n_list,
        amp_p_list,
        amp_n_list,
        *_unused_lists,
        amp_peak_p_list,
        amp_peak_n_list,
    ) = _get_b_lists(u, F, n, points, Q * E_ksi, A_sc, L_T, L_y, fy)

    def mean_or_nan(x: list[float]) -> float:
        """Mean of finite values or NaN."""
        return float(np.mean(x)) if x else np.nan

    def median_or_nan(x: list[float]) -> float:
        """Median of finite values or NaN."""
        return float(np.median(x)) if x else np.nan

    def q1_or_nan(x: list[float]) -> float:
        """First quartile or NaN."""
        return float(np.quantile(np.asarray(x, dtype=float), 0.25)) if x else np.nan

    def q3_or_nan(x: list[float]) -> float:
        """Third quartile or NaN."""
        return float(np.quantile(np.asarray(x, dtype=float), 0.75)) if x else np.nan

    def min_or_nan(x: list[float]) -> float:
        """Min of finite values or NaN."""
        return float(np.min(x)) if x else np.nan

    def max_or_nan(x: list[float]) -> float:
        """Max of finite values or NaN."""
        return float(np.max(x)) if x else np.nan

    def weighted_mean_or_nan(x: list[float], amps: list[float]) -> float:
        """Amplitude-weighted mean (normalized by max amp) or NaN."""
        if not x or not amps or len(x) != len(amps):
            return float("nan")
        b = np.asarray(x, dtype=float)
        a = np.asarray(amps, dtype=float)
        m = np.isfinite(b) & np.isfinite(a) & (a >= 0.0)
        if not np.any(m):
            return float("nan")
        b = b[m]
        a = a[m]
        amax = float(np.max(a)) if a.size else float("nan")
        if not np.isfinite(amax) or amax <= 0.0:
            return float(np.mean(b)) if b.size else float("nan")
        w = (a / amax) ** float(APPARENT_B_WEIGHT_POWER) + float(APPARENT_B_WEIGHT_EPS)
        sw = float(np.sum(w))
        if not np.isfinite(sw) or sw <= 0.0:
            return float("nan")
        return float(np.sum(w * b) / sw)

    def b_at_max_amplitude_or_nan(x: list[float], rank_amps: list[float]) -> float:
        """
        ``b`` from the half-cycle with largest ranking amplitude among paired finite entries.

        ``rank_amps`` should be parallel to ``x`` (same half-cycles). Prefer the **last** half-cycle
        when two share the same ranking amplitude (latest largest outer loop).
        """
        if not x or not rank_amps or len(x) != len(rank_amps):
            return float("nan")
        best_b: float | None = None
        best_a = float("-inf")
        for bi, ai in zip(x, rank_amps):
            if not (np.isfinite(bi) and np.isfinite(ai)):
                continue
            a = float(ai)
            if a >= best_a:
                best_a = a
                best_b = float(bi)
        return float(best_b) if best_b is not None else float("nan")

    L_y_sqrt_A_sc = L_y / (A_sc ** 0.5) if A_sc > 0 else np.nan
    fy_over_E = fy / E_ksi if E_ksi != 0 else np.nan
    fy_over_E_hat = fy / E_hat if E_hat != 0 else np.nan

    return {
        "Q": Q,
        "E_hat": E_hat,
        "L_y_sqrt_A_sc": L_y_sqrt_A_sc,
        "fy_over_E": fy_over_E,
        "fy_over_E_hat": fy_over_E_hat,
        "b_n_mean": mean_or_nan(b_n_list),
        "b_n_median": median_or_nan(b_n_list),
        "b_n_q1": q1_or_nan(b_n_list),
        "b_n_q3": q3_or_nan(b_n_list),
        "b_n_min": min_or_nan(b_n_list),
        "b_n_max": max_or_nan(b_n_list),
        "b_n_weighted_mean": weighted_mean_or_nan(b_n_list, amp_n_list),
        "b_n_max_amplitude": b_at_max_amplitude_or_nan(b_n_list, amp_peak_n_list),
        "b_p_mean": mean_or_nan(b_p_list),
        "b_p_median": median_or_nan(b_p_list),
        "b_p_q1": q1_or_nan(b_p_list),
        "b_p_q3": q3_or_nan(b_p_list),
        "b_p_min": min_or_nan(b_p_list),
        "b_p_max": max_or_nan(b_p_list),
        "b_p_weighted_mean": weighted_mean_or_nan(b_p_list, amp_p_list),
        "b_p_max_amplitude": b_at_max_amplitude_or_nan(b_p_list, amp_peak_p_list),
    }


def extract_bn_bp_unordered_row(sid: str, row: pd.Series) -> dict[str, float]:
    """Envelope ``b_p`` / ``b_n`` from F-u cloud; single pair -> means only, quartiles NaN."""
    fd_path = resolve_filtered_force_deformation_csv(sid, _PROJECT_ROOT)
    if fd_path is None or not fd_path.is_file():
        fd_path = force_deformation_unordered_csv_path(sid, _PROJECT_ROOT)
    if not fd_path.is_file():
        raise FileNotFoundError(str(fd_path))
    df_fd = pd.read_csv(fd_path)
    if DEF_COL not in df_fd.columns or FORCE_COL not in df_fd.columns:
        raise ValueError(f"{sid}: {fd_path} missing {DEF_COL} / {FORCE_COL}")
    u = df_fd[DEF_COL].to_numpy(dtype=float)
    F = df_fd[FORCE_COL].to_numpy(dtype=float)
    m = np.isfinite(u) & np.isfinite(F)
    u, F = u[m], F[m]
    if len(u) == 0:
        raise ValueError(f"{sid}: no finite F-u points")
    L_T = float(row["L_T"])
    L_y = float(row["L_y"])
    A_sc = float(row["A_sc"])
    A_t = float(row["A_t"])
    fy = float(row["fyp"])
    Q = compute_Q(L_T, L_y, A_sc, A_t)
    E_hat = Q * E_ksi
    diag = compute_envelope_bn_unordered(
        u, F, L_T=L_T, L_y=L_y, A_sc=A_sc, A_t=A_t, f_yc=fy, E_ksi_val=E_ksi
    )
    L_y_sqrt_A_sc = L_y / (A_sc**0.5) if A_sc > 0 else np.nan
    nan = float("nan")
    return {
        "Q": Q,
        "E_hat": E_hat,
        "L_y_sqrt_A_sc": L_y_sqrt_A_sc,
        "fy_over_E": fy / E_ksi if E_ksi != 0 else np.nan,
        "fy_over_E_hat": fy / E_hat if E_hat != 0 else np.nan,
        "b_n_mean": float(diag.b_n) if np.isfinite(diag.b_n) else nan,
        "b_n_median": nan,
        "b_n_q1": nan,
        "b_n_q3": nan,
        "b_n_min": nan,
        "b_n_max": nan,
        "b_n_weighted_mean": float(diag.b_n) if np.isfinite(diag.b_n) else nan,
        "b_n_max_amplitude": nan,
        "b_p_mean": float(diag.b_p) if np.isfinite(diag.b_p) else nan,
        "b_p_median": nan,
        "b_p_q1": nan,
        "b_p_q3": nan,
        "b_p_min": nan,
        "b_p_max": nan,
        "b_p_weighted_mean": float(diag.b_p) if np.isfinite(diag.b_p) else nan,
        "b_p_max_amplitude": nan,
    }


def get_specimens_with_resampled() -> list[str]:
    """Path-ordered specimen IDs that have resampled ``force_deformation.csv`` (excludes scatter)."""
    catalog = read_catalog()
    return sorted(path_ordered_resampled_force_csv_stems(catalog, project_root=_PROJECT_ROOT))


def main() -> None:
    """CLI entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Extract b_n, b_p and geometry params per specimen; save extended catalog CSV.")
    parser.add_argument("--specimen", type=str, default=None, help="Single specimen ID; default: all with resampled data")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output CSV path; default: results/calibration/specimen_apparent_bn_bp.csv",
    )
    args = parser.parse_args()

    catalog = read_catalog(CATALOG_PATH)
    catalog_by_name = catalog.set_index("Name")

    with_resampled = set(get_specimens_with_resampled())
    cloud_only = set(list_names_digitized_unordered(read_catalog()))
    # Catalog order; path-ordered (resampled) and/or scatter-cloud rows
    names_all: list[str] = []
    for name in catalog["Name"].astype(str):
        if name in with_resampled or name in cloud_only:
            names_all.append(name)

    if not names_all:
        print("No specimens with resampled data and/or digitized cloud inputs.")
        return

    if args.specimen:
        if args.specimen not in names_all:
            print(f"Specimen {args.specimen} not in catalog or missing resampled/cloud inputs.")
            return
        names_all = [args.specimen]

    out_path = Path(args.output) if args.output else SPECIMEN_APPARENT_BN_BP_PATH
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for sid in names_all:
        row = catalog_by_name.loc[sid].to_dict()
        out_row = {k: v for k, v in row.items()}
        if "Name" not in out_row:
            out_row["Name"] = sid

        if sid in with_resampled:
            rp = resolve_resampled_force_deformation_csv(sid, _PROJECT_ROOT)
            if rp is None or not rp.is_file():
                print(f"Skip {sid}: missing resampled force_deformation.csv")
                continue
            df = pd.read_csv(rp)
            if "Force[kip]" not in df.columns or "Deformation[in]" not in df.columns:
                print(f"Skip {sid}: missing Force or Deformation columns")
                continue
            loaded = load_cycle_points_resampled(sid)
            if loaded is None:
                points, _ = find_cycle_points(df)
            else:
                points, _ = loaded
            L_T = float(row["L_T"])
            L_y = float(row["L_y"])
            A_sc = float(row["A_sc"])
            A_t = float(row["A_t"])
            fy = float(row["fyp"])
            extra = extract_bn_bp_one_specimen(sid, df, points, L_T, L_y, A_sc, A_t, fy)
            out_row.update(extra)
            print(
                f"  {sid} (resampled): b_n_median={extra.get('b_n_median', np.nan):.6f}, "
                f"b_n_mean={extra.get('b_n_mean', np.nan):.6f}, b_p_median={extra.get('b_p_median', np.nan):.6f}"
            )
        else:
            try:
                extra = extract_bn_bp_unordered_row(sid, catalog_by_name.loc[sid])
            except (OSError, ValueError) as e:
                print(f"Skip {sid} (scatter): {e}")
                continue
            out_row.update(extra)
            print(
                f"  {sid} (cloud): b_n_mean={extra.get('b_n_mean', np.nan):.6f}, "
                f"b_p_mean={extra.get('b_p_mean', np.nan):.6f} (median/q1/q3 NaN)"
            )

        rows.append(out_row)

    out_df = pd.DataFrame(rows)
    # Order columns: catalog first, then added (Q, E_hat, ...)
    catalog_cols = [c for c in catalog.columns if c in out_df.columns]
    rest = [c for c in out_df.columns if c not in catalog_cols]
    out_df = out_df[catalog_cols + rest]
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
