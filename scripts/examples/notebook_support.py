"""
Helpers for the handoff notebooks under ``notebooks/``.

Keeps calibration logic in the existing scripts; notebooks only edit feature params
and call into these wrappers.
"""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

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
    """Feature-only loss: L2 or L1 on landmarks."""
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
    metric: str | None = "l2",
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
    train_specimens: Sequence[str],
    *,
    set_id_row: Mapping[str, object],
    prepare_data: bool = True,
    work_dir: Path | None = None,
) -> dict[str, object]:
    """
    Run generalized (shared-parameter) calibration on a small train list.

    Specimens should have ``generalized_weight > 0`` in ``BRB-Specimens.csv``.
    """
    ensure_sys_path()
    work = Path(work_dir or (NOTEBOOK_RESULTS / "generalized"))
    work.mkdir(parents=True, exist_ok=True)
    names = [str(s) for s in train_specimens]

    if prepare_data:
        prepare_specimens(names, e_ksi=float(set_id_row.get("E", 29000)))

    settings_path = write_set_id_settings_csv(
        work / "set_id_settings_generalized.csv", [set_id_row]
    )
    params_out = work / "generalized_brb_parameters.csv"
    metrics_out = work / "generalized_params_eval_metrics.csv"
    plots_dir = work / "overlays"

    run_py(
        "scripts/calibrate/optimize_generalized_brb_mse.py",
        "--set-id-settings",
        str(settings_path),
        "--train-specimens",
        ",".join(names),
        "--output-params",
        str(params_out),
        "--output-metrics",
        str(metrics_out),
        "--output-plots-dir",
        str(plots_dir),
        "--output-cloud-plots-dir",
        str(plots_dir),
    )
    return {
        "params": params_out,
        "metrics": metrics_out,
        "settings": settings_path,
        "plots_dir": plots_dir,
        "work_dir": work,
    }
