"""
Helpers for the calibration notebooks under ``notebooks/``.

Keeps calibration logic in the existing scripts; notebooks only edit
parameters and call into these wrappers.
"""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve()
PROJECT_ROOT = _THIS.parents[2]
SCRIPTS = PROJECT_ROOT / "scripts"
NOTEBOOK_RESULTS = PROJECT_ROOT / "results" / "notebooks"


def ensure_sys_path() -> Path:
    """Insert ``scripts/`` (and postprocess) on ``sys.path``; return project root."""
    for p in (SCRIPTS, SCRIPTS / "postprocess", SCRIPTS / "calibrate_single"):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    return PROJECT_ROOT


def python_exe() -> str:
    """Prefer ``$PYTHON`` when set (working OpenSeesPy env)."""
    import os

    return os.environ.get("PYTHON") or sys.executable


def run_py(script_rel: str, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a repo script with the same interpreter as the notebook."""
    cmd = [python_exe(), str(PROJECT_ROOT / script_rel), *args]
    print("+", " ".join(cmd))
    return subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        check=check,
        text=True,
    )


def display_png(path: Path | str, *, width: int | None = 720) -> None:
    """Show a PNG inline in Jupyter (no-op-ish outside IPython)."""
    p = Path(path)
    if not p.is_file():
        print(f"(missing plot) {p}")
        return
    try:
        from IPython.display import Image, display

        display(Image(filename=str(p), width=width))
    except Exception:
        print(f"Plot saved: {p}")


def write_single_input_csv(
    path: Path,
    *,
    set_id: int,
    optimize_params: Sequence[str],
    steel: Mapping[str, object],
    loss: Mapping[str, object],
    bounds: Mapping[str, tuple[float, float]],
) -> Path:
    """Write a ``calibrate_one_specimen``-style ``input.csv`` from notebook dicts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "section,key,value",
        f"meta,set_id,{int(set_id)}",
        f'meta,optimize_params,"{",".join(optimize_params)}"',
    ]
    for k, v in steel.items():
        lines.append(f"steel,{k},{v}")
    for k, v in loss.items():
        val = str(v).lower() if isinstance(v, bool) else v
        lines.append(f"loss,{k},{val}")
    for name, (lo, hi) in bounds.items():
        lines.append(f'bound,{name},"{lo},{hi}"')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path.resolve()


_SET_ID_COLUMNS = (
    "set_id",
    "inherit_from_set",
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
    "optimize_params",
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


def write_set_id_settings_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> Path:
    """Write individual or generalized ``set_id_settings*.csv`` from notebook row dicts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_SET_ID_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            out = {c: row.get(c, "") for c in _SET_ID_COLUMNS}
            if "inherit_from_set" not in row or row.get("inherit_from_set") in (None, ""):
                out["inherit_from_set"] = -999
            opt = out.get("optimize_params", "")
            if isinstance(opt, (list, tuple)):
                out["optimize_params"] = ",".join(opt)
            # DictWriter QUOTE_MINIMAL will quote fields that contain commas.
            out["optimize_params"] = str(out["optimize_params"]).strip().strip('"')
            for key in ("use_amplitude_weights",):
                if isinstance(out.get(key), bool):
                    out[key] = "TRUE" if out[key] else "FALSE"
            w.writerow(out)
    return path.resolve()


def default_loss_weights(*, metric: str = "l2") -> dict[str, object]:
    """Characteristic-point (feature) loss only: L2 or L1."""
    m = str(metric).strip().lower()
    if m not in ("l1", "l2"):
        raise ValueError("metric must be 'l1' or 'l2'")
    return {
        "w_feat_l2": 1 if m == "l2" else 0,
        "w_feat_l1": 1 if m == "l1" else 0,
        "w_energy_l2": 0,
        "w_energy_l1": 0,
        "w_unordered_binenv_l2": 0,
        "w_unordered_binenv_l1": 0,
        "use_amplitude_weights": True,
        "amplitude_weight_power": 2,
        "amplitude_weight_eps": 0.05,
    }


def default_steel_seeds() -> dict[str, object]:
    return {
        "E": 29000,
        "b_p": "median",
        "b_n": "median",
        "R0": 20,
        "cR1": 0.8875,
        "cR2": 0.15,
        "a1": 0.04,
        "a2": 1.0,
        "a3": 0.04,
        "a4": 1.0,
    }


def default_bounds(optimize_params: Sequence[str]) -> dict[str, tuple[float, float]]:
    """Subset of ``params_limits.csv`` defaults for common free params."""
    catalog = {
        "b_p": (0.0, 0.05),
        "b_n": (0.0, 0.05),
        "R0": (1.0, 50.0),
        "cR1": (0.8, 0.99),
        "cR2": (0.001, 3.0),
        "a1": (0.0, 0.2),
        "a3": (0.0, 0.2),
    }
    missing = [p for p in optimize_params if p not in catalog]
    if missing:
        raise KeyError(
            f"No notebook default bounds for {missing}; add them or pass bounds= explicitly."
        )
    return {p: catalog[p] for p in optimize_params}


def run_single_specimen(
    specimen: str,
    *,
    optimize_params: Sequence[str],
    steel: Mapping[str, object] | None = None,
    loss: Mapping[str, object] | None = None,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    metric: str | None = None,
    prepare_data: bool = True,
    set_id: int = 1,
    out_dir: Path | None = None,
    plots_dir: Path | None = None,
    work_dir: Path | None = None,
) -> dict[str, Path]:
    """Calibrate one specimen; return paths to params CSV and normalized overlay."""
    ensure_sys_path()
    from calibrate.plot_params_vs_filtered import run_one_specimen  # noqa: WPS433
    from calibrate_single.calibrate_one_specimen import (  # noqa: WPS433
        _default_force_deformation_path,
        calibrate_and_plot,
    )
    from calibrate_single.load_input import load_single_calibrate_inputs  # noqa: WPS433
    from calibrate_single.plot_all_sets_overlays import plot_all_sets_force_def_grid  # noqa: WPS433

    work = Path(work_dir or (NOTEBOOK_RESULTS / "single" / specimen))
    work.mkdir(parents=True, exist_ok=True)
    steel = dict(steel or default_steel_seeds())
    loss = dict(loss or default_loss_weights(metric=(metric or "l2")))
    bounds = dict(bounds or default_bounds(optimize_params))
    input_csv = write_single_input_csv(
        work / "input.csv",
        set_id=set_id,
        optimize_params=optimize_params,
        steel=steel,
        loss=loss,
        bounds=bounds,
    )
    cfg = load_single_calibrate_inputs(input_csv)[0]
    force_csv = _default_force_deformation_path(specimen, prepare_data=prepare_data)
    out = Path(out_dir or (work / "calibration"))
    plots = Path(plots_dir or (work / "plots"))
    ctx = calibrate_and_plot(
        specimen,
        force_csv,
        cfg,
        prepare_data=prepare_data,
        metric=metric,
        out_dir=out,
        plots_dir=plots,
    )
    params = out / "parameters.csv"
    param_df = pd.read_csv(params)
    run_one_specimen(
        specimen,
        param_df,
        ctx.cat_row,
        plots,
        force_deformation_csv=ctx.csv_path,
    )
    plot_all_sets_force_def_grid(
        specimen,
        param_df,
        ctx.cat_row,
        pd.read_csv(ctx.csv_path)["Deformation[in]"].to_numpy(dtype=float),
        pd.read_csv(ctx.csv_path)["Force[kip]"].to_numpy(dtype=float),
        plots,
    )
    overlay = plots / f"{specimen}_set{set_id}_force_def_norm.png"
    return {"params": params, "overlay": overlay, "plots_dir": plots, "work_dir": work}


def prepare_specimens(names: Sequence[str], *, e_ksi: float = 29000.0) -> None:
    """Build filtered/resampled CSVs for each Name (idempotent overwrite)."""
    ensure_sys_path()
    from calibrate_single.calibrate_one_specimen import _prepare_specimen_data  # noqa: WPS433

    for name in names:
        print(f"Preparing data for {name!r}...")
        _prepare_specimen_data(str(name), e_ksi=float(e_ksi))


def run_individual_specimens(
    specimens: Sequence[str],
    *,
    set_id_row: Mapping[str, object],
    prepare_data: bool = True,
    work_dir: Path | None = None,
    overlay_subdir: str = "notebook_individual",
) -> dict[str, object]:
    """
    Optimize 2–3 specimens with one shared notebook ``set_id`` configuration.

    Uses the same optimizer as the CLI individual path, with notebook-local
    settings / outputs under ``results/notebooks/individual/``.
    Overlays land in
    ``results/plots/calibration/individual_optimize/{overlay_subdir}/``.
    """
    ensure_sys_path()
    work = Path(work_dir or (NOTEBOOK_RESULTS / "individual"))
    work.mkdir(parents=True, exist_ok=True)
    names = [str(s) for s in specimens]

    if prepare_data:
        prepare_specimens(names, e_ksi=float(set_id_row.get("E", 29000)))

    run_py("scripts/calibrate/extract_bn_bp.py")
    settings_path = write_set_id_settings_csv(work / "set_id_settings.csv", [set_id_row])
    initial_path = work / "initial_brb_parameters.csv"
    run_py(
        "scripts/calibrate/build_initial_brb_parameters.py",
        "--set-id-settings",
        str(settings_path),
        "--output",
        str(initial_path),
    )

    init_df = pd.read_csv(initial_path)
    init_df = init_df[init_df["Name"].astype(str).isin(names)].copy()
    if init_df.empty:
        raise RuntimeError(
            f"No initial rows for {names}; check individual_optimize flags / resampled data."
        )
    init_df.to_csv(initial_path, index=False)

    opt_path = work / "optimized_brb_parameters.csv"
    run_py(
        "scripts/calibrate/optimize_brb_mse.py",
        "--initial-params",
        str(initial_path),
        "--set-id-settings",
        str(settings_path),
        "--output",
        str(opt_path),
    )

    for name in names:
        run_py(
            "scripts/calibrate/plot_params_vs_filtered.py",
            "--specimen",
            name,
            "--params",
            str(opt_path),
            "--output-dir",
            overlay_subdir,
        )

    overlays_dir = (
        PROJECT_ROOT
        / "results"
        / "plots"
        / "calibration"
        / "individual_optimize"
        / overlay_subdir
    )
    return {
        "params": opt_path,
        "settings": settings_path,
        "work_dir": work,
        "overlays_dir": overlays_dir,
    }


def run_generalized_demo(
    train_specimens: Sequence[str] | Mapping[str, float],
    *,
    set_id_row: Mapping[str, object],
    train_weights: Mapping[str, float] | None = None,
    eval_specimens: Sequence[str] | None = None,
    prepare_data: bool = True,
    work_dir: Path | None = None,
) -> dict[str, object]:
    """
    Run generalized (shared-parameter) calibration on an explicit train list.

    Specimens need path-ordered resampled ``force_deformation`` data. Catalog
    ``generalized_weight`` is ignored.

    ``train_specimens`` may be a list of Names (weight 1 each) or a ``{Name: weight}``
    mapping. Optional ``train_weights`` overlays/extends weights when Names are a list.

    ``eval_specimens`` are validation Names (path-ordered and/or digitized unordered):
    metrics and overlays after the fit, weight 0 so they do not enter the joint objective.
    """
    ensure_sys_path()
    work = Path(work_dir or (NOTEBOOK_RESULTS / "generalized"))
    work.mkdir(parents=True, exist_ok=True)

    if isinstance(train_specimens, Mapping):
        weight_by_name = {str(k): float(v) for k, v in train_specimens.items()}
    else:
        weight_by_name = {str(s): 1.0 for s in train_specimens}
        if train_weights:
            for k, v in train_weights.items():
                weight_by_name[str(k)] = float(v)
    names = list(weight_by_name.keys())
    if not names:
        raise ValueError("train_specimens is empty")

    validation = [str(s).strip() for s in (eval_specimens or []) if str(s).strip()]
    prepare_names = list(dict.fromkeys([*names, *validation]))

    if prepare_data:
        # Digitized-unordered validation specimens skip the path-ordered prepare pipeline.
        ensure_sys_path()
        from specimen_catalog import (  # noqa: WPS433
            get_specimen_record,
            read_catalog,
            uses_unordered_inputs,
        )

        cat = read_catalog()
        path_ordered_prep = [
            n
            for n in prepare_names
            if not uses_unordered_inputs(get_specimen_record(n, cat))
        ]
        if path_ordered_prep:
            prepare_specimens(path_ordered_prep, e_ksi=float(set_id_row.get("E", 29000)))

    settings_path = write_set_id_settings_csv(
        work / "set_id_settings_generalized.csv", [set_id_row]
    )
    params_out = work / "generalized_brb_parameters.csv"
    metrics_out = work / "generalized_params_eval_metrics.csv"
    plots_dir = work / "overlays"

    train_arg = ",".join(f"{n}:{weight_by_name[n]:g}" for n in names)
    cmd_args = [
        "--set-id-settings",
        str(settings_path),
        "--train-specimens",
        train_arg,
        "--output-params",
        str(params_out),
        "--output-metrics",
        str(metrics_out),
        "--output-plots-dir",
        str(plots_dir),
        "--output-cloud-plots-dir",
        str(plots_dir),
    ]
    if validation:
        cmd_args.extend(["--eval-specimens", ",".join(validation)])
    run_py("scripts/calibrate/optimize_generalized_brb_mse.py", *cmd_args)
    return {
        "params": params_out,
        "metrics": metrics_out,
        "settings": settings_path,
        "plots_dir": plots_dir,
        "work_dir": work,
        "train_weights": weight_by_name,
        "eval_specimens": validation,
        "validation_specimens": validation,
    }


def load_experiment_force_disp(specimen: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load resampled experimental hysteresis for ``specimen``.

    Returns ``(exp_disp, exp_force)`` in inches and kips.
    """
    ensure_sys_path()
    from specimen_catalog import resolve_resampled_force_deformation_csv  # noqa: WPS433

    csv_path = resolve_resampled_force_deformation_csv(str(specimen), PROJECT_ROOT)
    if csv_path is None or not csv_path.is_file():
        raise FileNotFoundError(
            f"No resampled force_deformation.csv for {specimen!r}. "
            "Run with PREPARE_DATA=True or prepare data first."
        )
    df = pd.read_csv(csv_path)
    if "Deformation[in]" not in df.columns or "Force[kip]" not in df.columns:
        raise ValueError(f"{csv_path}: need Deformation[in] and Force[kip] columns")
    exp_disp = df["Deformation[in]"].to_numpy(dtype=float)
    exp_force = df["Force[kip]"].to_numpy(dtype=float)
    return exp_disp, exp_force


_SIM_PARAM_KEYS = (
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


def params_dict_from_row(row: pd.Series | Mapping[str, object]) -> dict[str, float]:
    """
    Pull geometry + SteelMPF floats needed by ``simulate_force_disp`` from a params row.

    Units (repo convention, US customary as used in the catalog / OpenSees inputs):

    | key | meaning | unit |
    |-----|---------|------|
    | `L_T`, `L_y` | total / yielding length | in |
    | `A_sc`, `A_t` | core / transition area | in² |
    | `fyp`, `fyn` | yield stress (+/−) | ksi |
    | `E` | elastic modulus (before brace `Q`) | ksi |
    | `b_p`, `b_n` | kinematic hardening ratio | — |
    | `R0`, `cR1`, `cR2` | MP transition curvature | — |
    | `a1`, `a2` | compression isotropic scale / threshold | — |
    | `a3`, `a4` | tension isotropic scale / threshold | — |

    Displacement drive and simulated force are inches and kips.
    """
    if isinstance(row, pd.Series):
        get = row.__getitem__
        has = row.index.__contains__
    else:
        get = row.__getitem__
        has = row.__contains__
    out: dict[str, float] = {}
    missing: list[str] = []
    for k in _SIM_PARAM_KEYS:
        if not has(k) or pd.isna(get(k)):
            missing.append(k)
            continue
        out[k] = float(get(k))
    if missing:
        raise KeyError(f"params row missing required keys: {missing}")
    return out


def simulate_force_disp(
    exp_disp: np.ndarray,
    *,
    L_T: float,
    L_y: float,
    A_sc: float,
    A_t: float,
    fyp: float,
    fyn: float,
    E: float,
    b_p: float,
    b_n: float,
    R0: float = 20.0,
    cR1: float = 0.925,
    cR2: float = 0.15,
    a1: float = 0.0,
    a2: float = 1.0,
    a3: float = 0.0,
    a4: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the corotational BRB + SteelMPF model for a prescribed displacement history.

    Returns ``(num_disp, num_force)``. ``num_disp`` is the commanded history (same
    abscissa as ``exp_disp``); ``num_force`` is the simulated axial force [kip].
    """
    ensure_sys_path()
    from model.corotruss import run_simulation  # noqa: WPS433

    D = np.asarray(exp_disp, dtype=float)
    F = np.asarray(
        run_simulation(
            D,
            L_T=float(L_T),
            L_y=float(L_y),
            A_sc=float(A_sc),
            A_t=float(A_t),
            fyp=float(fyp),
            fyn=float(fyn),
            E=float(E),
            b_p=float(b_p),
            b_n=float(b_n),
            R0=float(R0),
            cR1=float(cR1),
            cR2=float(cR2),
            a1=float(a1),
            a2=float(a2),
            a3=float(a3),
            a4=float(a4),
        ),
        dtype=float,
    )
    return D, F


def geometry_from_catalog(specimen: str, *, E: float = 29000.0) -> dict[str, float]:
    """Catalog geometry + fy for ``simulate_force_disp`` (SteelMPF keys filled by caller)."""
    ensure_sys_path()
    from specimen_catalog import read_catalog  # noqa: WPS433

    row = read_catalog().loc[lambda df: df["Name"].astype(str) == str(specimen)].iloc[0]
    fy = float(row["f_yc_ksi"])
    return {
        "L_T": float(row["L_T_in"]),
        "L_y": float(row["L_y_in"]),
        "A_sc": float(row["A_c_in2"]),
        "A_t": float(row["A_t_in2"]),
        "fyp": fy,
        "fyn": fy,
        "E": float(E),
    }


def make_cyclic_drive(
    amplitudes_in: Sequence[float],
    *,
    n_quarter: int = 40,
) -> np.ndarray:
    """
    Synthetic BRB-like drive: for each amplitude A, path 0→+A→0→−A→0.

    ``n_quarter`` samples per quarter-cycle leg (excluding the shared endpoint).
    """
    amps = [float(a) for a in amplitudes_in if float(a) > 0.0]
    if not amps:
        raise ValueError("amplitudes_in must contain at least one positive amplitude")
    nq = max(2, int(n_quarter))
    parts: list[np.ndarray] = [np.array([0.0], dtype=float)]
    for A in amps:
        legs = (
            np.linspace(0.0, A, nq + 1)[1:],
            np.linspace(A, 0.0, nq + 1)[1:],
            np.linspace(0.0, -A, nq + 1)[1:],
            np.linspace(-A, 0.0, nq + 1)[1:],
        )
        parts.extend(legs)
    return np.concatenate(parts)


def plot_steelmpf_param_sweep(
    disp: np.ndarray,
    base: Mapping[str, float],
    param: str,
    values: Sequence[float],
    *,
    normalized: bool = True,
    title: str | None = None,
):
    """
    Overlay hysteresis for ``base`` and copies with ``param`` set to each value in ``values``.

    ``base`` must include geometry + SteelMPF keys accepted by ``simulate_force_disp``.
    Implementation: ``scripts/examples/notebook_support.py`` → ``plot_steelmpf_param_sweep``.
    """
    ensure_sys_path()
    import matplotlib.pyplot as plt

    if param not in _SIM_PARAM_KEYS:
        raise KeyError(f"unknown SteelMPF/geometry key {param!r}")
    D = np.asarray(disp, dtype=float)
    base_p = {k: float(base[k]) for k in _SIM_PARAM_KEYS}
    fyA = base_p["fyp"] * base_p["A_sc"]
    Ly = base_p["L_y"]
    if fyA <= 0.0 or Ly <= 0.0:
        raise ValueError("need positive fyp*A_sc and L_y for plotting")

    def _xy(force: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if normalized:
            return 100.0 * D / Ly, force / fyA
        return D, force

    _, F0 = simulate_force_disp(D, **base_p)
    x0, y0 = _xy(F0)
    v0 = float(base_p[param])
    # Drop sweep entries that match base (otherwise they redraw the same curve on top).
    variants = [float(v) for v in values if not np.isclose(float(v), v0, rtol=0.0, atol=1e-12)]
    # Colorblind-friendlier cycle + distinct dash patterns (not color-only).
    colors = (
        "#0077BB",
        "#EE7733",
        "#009988",
        "#CC3311",
        "#33BBEE",
        "#EE3377",
        "#BBBBBB",
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    # Solid base underneath so dashed variants stay visible on overlapping branches.
    (h_base,) = ax.plot(
        x0,
        y0,
        color="0.15",
        lw=2.4,
        ls="-",
        solid_capstyle="round",
        label=f"base ({param}={v0:g})",
        zorder=1,
    )
    # Named styles first (reliable in Jupyter); then custom dash patterns.
    linestyles: list[str | tuple] = [
        "--",
        ":",
        "-.",
        (0, (6, 2)),
        (0, (1, 1)),
        (0, (4, 1, 1, 1)),
        (0, (5, 1, 1, 1, 1, 1)),
    ]
    variant_handles = []
    for i, v in enumerate(variants):
        p = dict(base_p)
        p[param] = float(v)
        _, F = simulate_force_disp(D, **p)
        x, y = _xy(F)
        (h,) = ax.plot(
            x,
            y,
            lw=2.0,
            color=colors[i % len(colors)],
            ls=linestyles[i % len(linestyles)],
            dash_capstyle="round",
            label=f"{param}={float(v):g}",
            zorder=2 + i,
        )
        variant_handles.append(h)
    if normalized:
        ax.set_xlabel(r"Axial strain, $\delta / L_y$ (%)")
        ax.set_ylabel(r"Axial force, $P / (f_y A_{sc})$")
    else:
        ax.set_xlabel("Deformation [in]")
        ax.set_ylabel("Force [kip]")
    ax.set_title(title or f"Varying {param}")
    # Legend order: variants first, base last (bottom).
    handles = variant_handles + [h_base]
    ax.legend(handles=handles, fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


def load_cycle_partition_meta(
    specimen: str,
    *,
    use_amplitude_weights: bool = True,
    amplitude_weight_power: float = 2.0,
    amplitude_weight_eps: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    Load resampled hysteresis and weight-cycle partition for ``specimen``.

    Returns ``(disp, force, meta)`` where ``meta`` rows have ``start``, ``end``,
    ``amp``, ``w_c``, ``kind``, ``incomplete`` (same as the calibration partition).
    """
    ensure_sys_path()
    from calibrate.amplitude_mse_partition import build_amplitude_weights  # noqa: WPS433
    from postprocess.cycle_points import find_cycle_points, load_cycle_points_resampled  # noqa: WPS433

    disp, force = load_experiment_force_disp(specimen)
    loaded = load_cycle_points_resampled(str(specimen))
    if loaded is None:
        from specimen_catalog import resolve_resampled_force_deformation_csv  # noqa: WPS433

        csv_path = resolve_resampled_force_deformation_csv(str(specimen), PROJECT_ROOT)
        df = pd.read_csv(csv_path)
        points, _ = find_cycle_points(df)
    else:
        points, _ = loaded
    _w, meta = build_amplitude_weights(
        disp,
        points,
        p=float(amplitude_weight_power),
        eps=float(amplitude_weight_eps),
        use_amplitude_weights=bool(use_amplitude_weights),
    )
    return disp, force, meta


def yield_disp_from_params(p: Mapping[str, float]) -> float:
    """Dy [in] from geometry + fy / E as used by characteristic-point gating."""
    ensure_sys_path()
    from calibrate.cycle_feature_loss import yield_displacement_dy_in  # noqa: WPS433

    return float(
        yield_displacement_dy_in(
            fy_ksi=float(p["fyp"]),
            E_ksi=float(p["E"]),
            L_T_in=float(p["L_T"]),
            L_y_in=float(p["L_y"]),
            A_sc_in2=float(p["A_sc"]),
            A_t_in2=float(p["A_t"]),
        )
    )


def plot_cycles_multi_axes(
    disp: np.ndarray,
    force_exp: np.ndarray,
    meta: list[dict],
    *,
    fy_ksi: float,
    A_sc: float,
    dy_in: float,
    force_num: np.ndarray | None = None,
    ncols: int = 6,
    max_cycles: int | None = None,
    title: str | None = None,
):
    """
    One axes per weight cycle: experimental hysteresis, optional numerical overlay,
    and characteristic points (numbered 1–14) on the experimental curve.

    Cycle interval is half-open ``[start, end)`` (same as ``build_amplitude_weights`` /
    ``extract_cycle_landmarks``). Start/end markers are vertices at ``start`` and
    ``end - 1``.

    Implementation: ``scripts/examples/notebook_support.py`` → ``plot_cycles_multi_axes``;
    landmarks from ``scripts/calibrate/cycle_feature_loss.py`` → ``extract_cycle_landmarks``.
    """
    ensure_sys_path()
    import math

    import matplotlib.pyplot as plt
    from calibrate.cycle_feature_loss import extract_cycle_landmarks  # noqa: WPS433

    D = np.asarray(disp, dtype=float)
    Fe = np.asarray(force_exp, dtype=float)
    Fn = None if force_num is None else np.asarray(force_num, dtype=float)
    rows_meta = list(meta)
    if max_cycles is not None:
        rows_meta = rows_meta[: int(max_cycles)]
    n = len(rows_meta)
    if n == 0:
        print("(no cycles to plot)")
        return None

    ncols = max(1, int(ncols))
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.4 * ncols, 2.2 * nrows),
        squeeze=False,
        sharex=False,
        sharey=False,
        layout="constrained",
    )
    for i, m in enumerate(rows_meta):
        ax = axes[i // ncols][i % ncols]
        s, e = int(m["start"]), int(m["end"])
        if e <= s:
            ax.set_title(f"c{i} (empty)", fontsize=8)
            ax.axis("off")
            continue
        ax.plot(D[s:e], Fe[s:e], color="0.45", lw=1.0, label="exp")
        if Fn is not None and len(Fn) == len(D):
            ax.plot(D[s:e], Fn[s:e], ls="--", lw=1.0, label="num")
        # Cycle endpoints (half-open [s, e) → last sample is e-1).
        ax.scatter(
            [D[s]],
            [Fe[s]],
            s=36,
            marker="s",
            facecolors="none",
            edgecolors="#0077BB",
            linewidths=1.2,
            zorder=6,
            label="start",
        )
        ax.scatter(
            [D[e - 1]],
            [Fe[e - 1]],
            s=36,
            marker="s",
            facecolors="none",
            edgecolors="#CC3311",
            linewidths=1.2,
            zorder=6,
            label="end",
        )
        ax.annotate("S", (D[s], Fe[s]), textcoords="offset points", xytext=(-8, -8), fontsize=6, color="#0077BB")
        ax.annotate("E", (D[e - 1], Fe[e - 1]), textcoords="offset points", xytext=(2, -8), fontsize=6, color="#CC3311")
        lm = extract_cycle_landmarks(
            D, Fe, s, e, fy_ksi=float(fy_ksi), a_sc=float(A_sc), dy_in=float(dy_in)
        )
        for k, pt in enumerate(lm):
            if pt is None:
                continue
            ax.scatter([pt[0]], [pt[1]], s=18, zorder=5)
            ax.annotate(
                str(k + 1),
                (pt[0], pt[1]),
                textcoords="offset points",
                xytext=(2, 2),
                fontsize=6,
            )
        amp = float(m.get("amp", float("nan")))
        wc = float(m.get("w_c", float("nan")))
        ax.set_title(f"c{i}  A={amp:.2g}  w={wc:.2g}", fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=7)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    # Shared axis labels (constrained layout reserves space; tight_layout often clips these).
    fig.supxlabel("Deformation [in]", fontsize=11)
    fig.supylabel("Force [kip]", fontsize=11)
    if title:
        fig.suptitle(title, fontsize=11)
    plt.show()
    return fig


def plot_cycle_energy_bars(
    disp: np.ndarray,
    force_exp: np.ndarray,
    meta: list[dict],
    force_num: np.ndarray | None = None,
    *,
    title: str | None = "Per-cycle energy |∫ F du|",
):
    """
    Bar chart of experimental (and optional numerical) cycle energies.

    Implementation: ``scripts/examples/notebook_support.py`` → ``plot_cycle_energy_bars``;
    per-cycle work from ``scripts/calibrate/amplitude_mse_partition.py`` →
    ``cycle_abs_trapz_work`` (same as energy term in calibration).
    """
    ensure_sys_path()
    import matplotlib.pyplot as plt
    from calibrate.amplitude_mse_partition import cycle_abs_trapz_work  # noqa: WPS433

    D = np.asarray(disp, dtype=float)
    Fe = np.asarray(force_exp, dtype=float)
    Fn = None if force_num is None else np.asarray(force_num, dtype=float)
    e_exp: list[float] = []
    e_num: list[float] = []
    for m in meta:
        s, e = int(m["start"]), int(m["end"])
        e_exp.append(cycle_abs_trapz_work(D, Fe, s, e))
        if Fn is not None and len(Fn) == len(D):
            e_num.append(cycle_abs_trapz_work(D, Fn, s, e))
    x = np.arange(len(e_exp))
    plt.figure(figsize=(max(6.0, 0.28 * len(e_exp)), 3.2))
    plt.bar(x - (0.2 if e_num else 0.0), e_exp, width=0.4 if e_num else 0.7, label="exp", color="0.55")
    if e_num:
        plt.bar(x + 0.2, e_num, width=0.4, label="num")
        plt.legend()
    plt.xlabel("cycle index")
    plt.ylabel("energy [kip·in]")
    if title:
        plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def apparent_b_summary(specimen: str, p: Mapping[str, float]) -> dict[str, float]:
    """
    Apparent hardening summary (median / mean / weighted_mean of per-branch b fits).

    Uses the same extractor as the apparent-b seed path.
    """
    ensure_sys_path()
    from calibrate.extract_bn_bp import extract_bn_bp_one_specimen  # noqa: WPS433
    from postprocess.cycle_points import find_cycle_points, load_cycle_points_resampled  # noqa: WPS433
    from specimen_catalog import resolve_resampled_force_deformation_csv  # noqa: WPS433

    csv_path = resolve_resampled_force_deformation_csv(str(specimen), PROJECT_ROOT)
    if csv_path is None or not csv_path.is_file():
        raise FileNotFoundError(f"No resampled CSV for {specimen!r}")
    df = pd.read_csv(csv_path)
    loaded = load_cycle_points_resampled(str(specimen))
    if loaded is None:
        points, _ = find_cycle_points(df)
    else:
        points, _ = loaded
    out = extract_bn_bp_one_specimen(
        str(specimen),
        df,
        points,
        float(p["L_T"]),
        float(p["L_y"]),
        float(p["A_sc"]),
        float(p["A_t"]),
        float(p["fyp"]),
    )
    keys = (
        "b_p_median",
        "b_n_median",
        "b_p_mean",
        "b_n_mean",
        "b_p_weighted_mean",
        "b_n_weighted_mean",
        "Q",
        "E_hat",
    )
    return {k: float(out[k]) for k in keys if k in out and pd.notna(out[k])}


def _prep_stage_force_paths(specimen: str) -> list[tuple[str, Path]]:
    """Raw / filtered / resampled ``force_deformation.csv`` paths for ``specimen``."""
    return [
        ("raw", PROJECT_ROOT / "data" / "raw" / str(specimen) / "force_deformation.csv"),
        ("filtered", PROJECT_ROOT / "data" / "filtered" / str(specimen) / "force_deformation.csv"),
        ("resampled", PROJECT_ROOT / "data" / "resampled" / str(specimen) / "force_deformation.csv"),
    ]


def _prep_stage_style(label: str) -> dict[str, object]:
    """Line style by prep stage (raw solid; filtered / resampled easier to spot)."""
    key = str(label).strip().lower()
    if key.startswith("raw"):
        return {"color": "0.55", "ls": "-", "lw": 0.9, "alpha": 0.9}
    if key.startswith("filtered"):
        return {"color": "#0077BB", "ls": "--", "lw": 1.2, "alpha": 0.95}
    if key.startswith("resampled"):
        return {"color": "#EE7733", "ls": "-.", "lw": 1.4, "alpha": 0.95}
    return {"color": "0.2", "ls": "-", "lw": 1.0, "alpha": 0.9}


def plot_prep_stages(specimen: str, *, max_points: int = 8000) -> None:
    """
    Overlay raw, filtered, and resampled hysteresis when those files exist.

    Dense series are downsampled to ``max_points`` for responsive plotting.
    """
    ensure_sys_path()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5.0), layout="constrained")
    plotted = 0
    for label, path in _prep_stage_force_paths(specimen):
        if not path.is_file():
            print(f"(skip {label}: missing {path})")
            continue
        df = pd.read_csv(path)
        if "Deformation[in]" not in df.columns or "Force[kip]" not in df.columns:
            print(f"(skip {label}: missing columns in {path})")
            continue
        d = df["Deformation[in]"].to_numpy(dtype=float)
        f = df["Force[kip]"].to_numpy(dtype=float)
        n = len(d)
        if n == 0:
            print(f"(skip {label}: empty {path})")
            continue
        if n > int(max_points):
            idx = np.linspace(0, n - 1, int(max_points)).astype(int)
            d, f = d[idx], f[idx]
        style = _prep_stage_style(label)
        ax.plot(d, f, label=f"{label.capitalize()} (n={n})", **style)
        plotted += 1
    if plotted == 0:
        print("(no prep-stage CSVs found)")
        plt.close(fig)
        return
    ax.set_xlabel("Deformation [in]")
    ax.set_ylabel("Force [kip]")
    ax.set_title(f"{specimen}: prep stages (hysteresis)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.show()


def _cum_abs_deformation(disp: np.ndarray) -> np.ndarray:
    """Cumulative path length along the deformation drive: Σ|Δδ| (inches)."""
    d = np.asarray(disp, dtype=float)
    if d.size == 0:
        return d
    if d.size == 1:
        return np.array([0.0], dtype=float)
    return np.concatenate(([0.0], np.cumsum(np.abs(np.diff(d)))))


def _downsample_series(
    *arrays: np.ndarray, max_points: int
) -> tuple[np.ndarray, ...]:
    n = len(arrays[0])
    if n <= int(max_points):
        return arrays
    idx = np.linspace(0, n - 1, int(max_points)).astype(int)
    return tuple(a[idx] for a in arrays)


def plot_prep_histories(specimen: str, *, max_points: int = 8000) -> None:
    """
    Overlay prep-stage force and strain histories.

    Two figures (same y-axes as report time histories):

    1. vs **point index** — raw and filtered only (same sampling length).
    2. vs **cumulative |Δδ|** — raw, filtered, and resampled on shared path length.
    """
    ensure_sys_path()
    import matplotlib.pyplot as plt

    geom = geometry_from_catalog(str(specimen))
    L_y = float(geom["L_y"])
    fyA = float(geom["fyp"]) * float(geom["A_sc"])
    if not (np.isfinite(L_y) and L_y > 0 and np.isfinite(fyA) and fyA > 0):
        raise ValueError(f"Bad geometry for {specimen!r}: L_y={L_y}, fy*A_sc={fyA}")

    series: list[tuple[str, int, np.ndarray, np.ndarray, np.ndarray]] = []
    for label, path in _prep_stage_force_paths(specimen):
        if not path.is_file():
            print(f"(skip {label}: missing {path})")
            continue
        df = pd.read_csv(path)
        if "Deformation[in]" not in df.columns or "Force[kip]" not in df.columns:
            print(f"(skip {label}: missing columns in {path})")
            continue
        d = df["Deformation[in]"].to_numpy(dtype=float)
        f = df["Force[kip]"].to_numpy(dtype=float)
        if d.size == 0:
            print(f"(skip {label}: empty {path})")
            continue
        cum = _cum_abs_deformation(d)
        series.append((label.capitalize(), len(d), d, f, cum))

    if not series:
        print("(no prep-stage CSVs found)")
        return

    def _draw(x_kind: str, stages: list[tuple[str, int, np.ndarray, np.ndarray, np.ndarray]]) -> None:
        if not stages:
            return
        fig, (ax_f, ax_d) = plt.subplots(
            2, 1, figsize=(8.0, 5.2), sharex=True, layout="constrained"
        )
        for short, n, d, f, cum in stages:
            if x_kind == "index":
                x = np.arange(n, dtype=float)
                x, d, f = _downsample_series(x, d, f, max_points=max_points)
            else:
                x, d, f = _downsample_series(cum, d, f, max_points=max_points)
            strain_pct = 100.0 * d / L_y
            force_n = f / fyA
            style = _prep_stage_style(short)
            ax_f.plot(x, force_n, label=f"{short} (n={n})", **style)
            ax_d.plot(x, strain_pct, label=f"{short} (n={n})", **style)
        ax_f.set_ylabel(r"Axial force, $P/(f_y A_{sc})$")
        ax_f.legend(fontsize=8, loc="best")
        ax_f.grid(True, alpha=0.3)
        ax_f.axhline(0.0, color="0.5", lw=0.6)
        ax_d.set_ylabel(r"Axial strain, $\delta/L_y$ (%)")
        ax_d.legend(fontsize=8, loc="best")
        ax_d.grid(True, alpha=0.3)
        ax_d.axhline(0.0, color="0.5", lw=0.6)
        if x_kind == "index":
            ax_f.set_title(f"{specimen}: raw vs filtered (point index)")
            ax_d.set_xlabel("Point index")
        else:
            ax_f.set_title(f"{specimen}: prep stages vs cumulative |Δδ|")
            ax_d.set_xlabel(r"Cumulative $|\Delta\delta|$ [in]")
        plt.show()

    # Point index: same sampling length only (exclude resampled).
    _draw("index", [s for s in series if s[0].lower() != "resampled"])
    _draw("cum", series)


def digitized_unordered_sim_arrays(
    specimen: str,
    p: Mapping[str, float],
) -> dict[str, object]:
    """
    Drive + cloud for a digitized unordered specimen with shared SteelMPF ``p``.

    Returns dict with ``u_cloud``, ``F_cloud``, ``D_drive``, ``F_sim``, and
    ``J_binenv`` / ``J_binenv_l1`` from ``compute_unordered_cloud_metrics``.
    Envelope ``b_p``/``b_n`` from the cloud replace those in ``p`` for the sim
    (same rule as generalized eval).
    """
    ensure_sys_path()
    from calibrate.digitized_unordered_eval_lib import (  # noqa: WPS433
        compute_unordered_cloud_metrics,
        eval_row_with_envelope_bn_from_unordered,
        load_digitized_unordered_series,
    )
    from model.corotruss import run_simulation  # noqa: WPS433
    from specimen_catalog import read_catalog  # noqa: WPS433

    catalog = read_catalog()
    cat_row = catalog.set_index("Name").loc[str(specimen)]
    steel_row = pd.Series({k: float(p[k]) for k in _SIM_PARAM_KEYS})
    series = load_digitized_unordered_series(
        str(specimen),
        PROJECT_ROOT,
        steel_row=steel_row,
        catalog_row=cat_row,
    )
    if series is None:
        raise FileNotFoundError(
            f"Digitized unordered series missing for {specimen!r} "
            "(need deformation_history + unordered force_deformation)."
        )
    D_drive, u_c, F_c = series
    sim_row = eval_row_with_envelope_bn_from_unordered(steel_row, cat_row, u_c, F_c)
    F_sim = np.asarray(
        run_simulation(
            D_drive,
            L_T=float(sim_row["L_T"]),
            L_y=float(sim_row["L_y"]),
            A_sc=float(sim_row["A_sc"]),
            A_t=float(sim_row["A_t"]),
            fyp=float(sim_row["fyp"]),
            fyn=float(sim_row["fyn"]),
            E=float(sim_row["E"]),
            b_p=float(sim_row["b_p"]),
            b_n=float(sim_row["b_n"]),
            R0=float(sim_row["R0"]),
            cR1=float(sim_row["cR1"]),
            cR2=float(sim_row["cR2"]),
            a1=float(sim_row["a1"]),
            a2=float(sim_row["a2"]),
            a3=float(sim_row["a3"]),
            a4=float(sim_row["a4"]),
        ),
        dtype=float,
    )
    cloud = compute_unordered_cloud_metrics(u_c, F_c, D_drive, F_sim)
    return {
        "u_cloud": np.asarray(u_c, dtype=float),
        "F_cloud": np.asarray(F_c, dtype=float),
        "D_drive": np.asarray(D_drive, dtype=float),
        "F_sim": F_sim,
        "J_binenv": float(cloud.J_binenv),
        "J_binenv_l1": float(cloud.J_binenv_l1),
        "sim_row": sim_row,
    }


def summarize_generalized_train_validation_metrics(
    metrics: pd.DataFrame,
    *,
    train_names: Sequence[str],
    eval_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Compact train vs validation table with ``J_feat`` and ``J_binenv`` (L2).

    ``role`` is ``train`` when Name is in ``train_names`` and weight > 0, else
    ``validation``.
    """
    df = metrics.copy()
    train_set = {str(n) for n in train_names}
    eval_set = {str(n) for n in (eval_names or [])}
    roles: list[str] = []
    for _, row in df.iterrows():
        name = str(row["Name"])
        w = float(row["specimen_weight"]) if "specimen_weight" in df.columns else 0.0
        if name in train_set and w > 0.0:
            roles.append("train")
        elif name in eval_set or w <= 0.0:
            roles.append("validation")
        else:
            roles.append("other")
    out = pd.DataFrame(
        {
            "Name": df["Name"].astype(str),
            "role": roles,
            "J_feat": df["final_J_feat_raw"] if "final_J_feat_raw" in df.columns else np.nan,
            "J_binenv": (
                df["final_unordered_J_binenv"]
                if "final_unordered_J_binenv" in df.columns
                else np.nan
            ),
            "J_total": df["final_J_total"] if "final_J_total" in df.columns else np.nan,
            "specimen_weight": df["specimen_weight"] if "specimen_weight" in df.columns else np.nan,
        }
    )
    return out.sort_values(["role", "Name"]).reset_index(drop=True)


# Backward-compatible alias
summarize_generalized_train_holdout_metrics = summarize_generalized_train_validation_metrics


def show_specimen_catalog(
    *,
    max_points: int = 2500,
    ncols: int = 4,
    only_with_raw: bool = True,
) -> pd.DataFrame:
    """
    Show the BRB specimen catalog and a grid of **raw** experimental hysteresis.

    Use this before picking ``SPECIMEN`` / ``SPECIMENS`` in a notebook so you do
    not have to hunt through ``data/raw/`` CSVs by hand.

    Path-ordered tests are drawn as lines; digitized tests as scatter points.
    Returns the catalog DataFrame (also displayed).
    """
    ensure_sys_path()
    import math

    import matplotlib.pyplot as plt
    from specimen_catalog import (  # noqa: WPS433
        get_specimen_record,
        read_catalog,
        uses_unordered_inputs,
    )

    try:
        from IPython.display import display  # type: ignore
    except ImportError:
        display = print  # type: ignore[assignment]

    cat = read_catalog()
    show_cols = [
        c
        for c in (
            "Name",
            "experimental_layout",
            "path_ordered",
            "f_yc_ksi",
            "A_c_in2",
            "A_t_in2",
            "L_T_in",
            "L_y_in",
            "individual_optimize",
            "generalized_weight",
        )
        if c in cat.columns
    ]
    display(cat[show_cols].reset_index(drop=True))

    rows: list[tuple[str, np.ndarray, np.ndarray, bool]] = []
    for name in cat["Name"].astype(str).tolist():
        raw_csv = PROJECT_ROOT / "data" / "raw" / name / "force_deformation.csv"
        if not raw_csv.is_file():
            if only_with_raw:
                print(f"(skip {name}: no data/raw/{name}/force_deformation.csv)")
            continue
        df = pd.read_csv(raw_csv)
        # Accept common deformation column aliases.
        if "Deformation[in]" not in df.columns and "Displacement[in]" in df.columns:
            df = df.rename(columns={"Displacement[in]": "Deformation[in]"})
        if "Deformation[in]" not in df.columns or "Force[kip]" not in df.columns:
            print(f"(skip {name}: missing Force/Deformation columns)")
            continue
        d = df["Deformation[in]"].to_numpy(dtype=float)
        f = df["Force[kip]"].to_numpy(dtype=float)
        n = len(d)
        if n == 0:
            continue
        if n > int(max_points):
            idx = np.linspace(0, n - 1, int(max_points)).astype(int)
            d, f = d[idx], f[idx]
        rec = get_specimen_record(name, cat)
        rows.append((name, d, f, bool(uses_unordered_inputs(rec))))

    if not rows:
        print("(no raw hysteresis files found under data/raw/)")
        return cat

    n = len(rows)
    nc = max(1, int(ncols))
    nr = int(math.ceil(n / nc))
    fig, axes = plt.subplots(
        nr,
        nc,
        figsize=(2.6 * nc, 2.2 * nr),
        squeeze=False,
        layout="constrained",
    )
    for i, (name, d, f, unordered) in enumerate(rows):
        ax = axes[i // nc][i % nc]
        if unordered:
            # ~50% larger than a typical small overview marker (s=6 → s=9).
            ax.scatter(d, f, s=9, c="0.35", alpha=0.7, linewidths=0)
            ax.set_title(f"{name} (digitized)", fontsize=8)
        else:
            ax.plot(d, f, color="0.25", lw=0.7)
            ax.set_title(name, fontsize=8)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.25)
    for j in range(n, nr * nc):
        axes[j // nc][j % nc].axis("off")
    fig.supxlabel("Deformation [in]", fontsize=10)
    fig.supylabel("Force [kip]", fontsize=10)
    fig.suptitle("Raw experimental hysteresis (unfiltered)", fontsize=11)
    plt.show()
    return cat


LOSS_WEIGHTS_HELP = """
## What the objective measures

We ask the optimizer to make the model hysteresis look like the experiment.
It does that by minimizing a **weighted sum** of a few error terms:

`J_total = Σ w_k · metric_k`

| Weight | Plain-language meaning |
|--------|------------------------|
| `w_feat_l2` / `w_feat_l1` | How well we hit the **important points** on each cycle (peaks, unload, re-yield). Demos usually turn **exactly one** of these on (set to 1). |
| `w_energy_l2` / `w_energy_l1` | How well per-cycle **energy** (`|∫ F du|`) matches. Usually left at 0. |
| `w_unordered_binenv_l2` / `w_unordered_binenv_l1` | How well the **outer force envelope** matches when you only have a cloud of `(D, F)` points (digitized tests). Usually weight 0 in the fit; still reported as a diagnostic. |

Even when a weight is 0, the metrics tables still **print** that column so you can look at it.

## Amplitude weighting (inside the point-match term)

These settings do **not** add a new term. They only change how cycles are mixed
**inside** the characteristic-point error `J_feat`:

`J_feat = Σ_c w_c · j_c / Σ_c w_c`

| Setting | Effect |
|---------|--------|
| `use_amplitude_weights=False` | Every cycle counts equally (`w_c = 1`). |
| `use_amplitude_weights=True` | Larger cycles count more: `w_c = (A_c / A_max)^p + ε`. |
| `amplitude_weight_power` (`p`) | How strongly amplitude boosts the weight. |
| `amplitude_weight_eps` (`ε`) | Small floor so tiny cycles are not ignored completely. |
""".strip()

