#!/usr/bin/env bash
set -euo pipefail

# Run from anywhere: cd to this script's directory (repo root).
cd "$(dirname "$0")"

# Full pipeline. Live terminal output by default.
# Mirror to a file: PIPELINE_LOG=pipeline_log.txt ./run.sh
#
# Python: uses $PYTHON if set, else `python3`, else `python`.
# Example: PYTHON=/path/to/venv/bin/python ./run.sh

# J_feat cycle weights w_c (--amplitude-weights on calibration steps).
USE_AMPLITUDE_WEIGHTS=false
AMP_W_ARGS=()
if [[ "${USE_AMPLITUDE_WEIGHTS}" == "true" || "${USE_AMPLITUDE_WEIGHTS}" == "1" ]]; then
  AMP_W_ARGS=(--amplitude-weights)
fi

if [[ -n "${PYTHON:-}" ]]; then
  PY="$PYTHON"
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "No Python found. Install Python 3 or set PYTHON=/path/to/python" >&2
  exit 1
fi

run_py() {
  "$PY" "$@"
}

_pipeline_print_footer() {
  local _start_s=$1 _pipe_end_s _elapsed_s _h _m _s
  _pipe_end_s=$(date +%s)
  _elapsed_s=$((_pipe_end_s - _start_s))
  _h=$((_elapsed_s / 3600))
  _m=$(((_elapsed_s % 3600) / 60))
  _s=$((_elapsed_s % 60))
  echo ""
  echo "========================================================================"
  echo "  BRB-Calibration pipeline finished"
  echo "  $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
  if ((_h >= 1)); then
    echo "  Elapsed:  ${_h}h ${_m}m ${_s}s"
  elif ((_m >= 1)); then
    echo "  Elapsed:  ${_m}m ${_s}s"
  else
    echo "  Elapsed:  ${_s}s"
  fi
  echo "========================================================================"
  echo ""
}

run_pipeline_steps() {
  echo "Using Python: $PY"
  if ! run_py -c "import openseespy.opensees" 2>/dev/null; then
    echo "OpenSeesPy failed to import with: $PY" >&2
    echo "Install deps (pip install -r requirements.txt) or set PYTHON to an env where" >&2
    echo "  import openseespy.opensees" >&2
    echo "succeeds (macOS/Linux often need the official OpenSeesPy wheel for your Python version)." >&2
    exit 1
  fi

  run_py scripts/calibrate/print_calibration_config_heads.py

  # Optional full reset: ./clean_outputs.sh

  run_py scripts/postprocess/cycle_points.py --overwrite
  run_py scripts/postprocess/filter_force.py
  run_py scripts/postprocess/resample_filtered.py
  run_py scripts/postprocess/plot_specimens.py

  run_py scripts/calibrate/extract_bn_bp.py
  run_py scripts/calibrate/build_initial_brb_parameters.py
  run_py scripts/calibrate/plot_b_slopes.py
  run_py scripts/calibrate/plot_b_histograms_and_scatter.py

  run_py scripts/calibrate/plot_preset_overlays.py

  run_py scripts/calibrate/optimize_brb_mse.py "${AMP_W_ARGS[@]}" \
    --initial-params results/calibration/individual_optimize/initial_brb_parameters.csv \
    --output results/calibration/individual_optimize/optimized_brb_parameters.csv
  run_py scripts/calibrate/plot_params_vs_filtered.py \
    --params results/calibration/individual_optimize/optimized_brb_parameters.csv \
    --output-dir overlays
  # Best L2 vs best L1 per specimen (normalized montage)
  run_py scripts/calibrate/plot_individual_best_l1_l2_overlays.py

  run_py scripts/calibrate/optimize_generalized_brb_mse.py "${AMP_W_ARGS[@]}" \
    --output-params results/calibration/generalized_optimize/generalized_brb_parameters.csv \
    --output-metrics results/calibration/generalized_optimize/generalized_params_eval_metrics.csv \
    --output-plots-dir results/plots/calibration/generalized_optimize/overlays

  run_py scripts/calibrate/plot_compare_calibration_overlays.py

  run_py scripts/calibrate/report_calibration_param_tables.py --write summary_statistics/calibration_parameter_summary.md

  run_py scripts/calibrate/report_individual_vs_generalized_metrics.py
}

run_pipeline() {
  local _pipe_start_s _rc=0
  _pipe_start_s=$(date +%s)
  echo ""
  echo "========================================================================"
  echo "  BRB-Calibration pipeline"
  echo "  $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
  echo "========================================================================"
  echo ""

  run_pipeline_steps || _rc=$?
  _pipeline_print_footer "$_pipe_start_s"
  return "$_rc"
}

if [[ -n "${PIPELINE_LOG:-}" ]]; then
  run_pipeline 2>&1 | tee "$PIPELINE_LOG"
else
  run_pipeline
fi
