#!/usr/bin/env bash
# Wipe regenerated pipeline outputs. Never removes data/raw/ or config/calibration/*.csv
# (catalog, seeds, limits, loss settings); other top-level data/* folders (if any) are left untouched.
#
# Removes contents of:
#   data/filtered, data/resampled, data/cycle_points_original, data/cycle_points_resampled
#   results/plots/  (postprocess QA, apparent_b/, calibration/individual_optimize overlays,
#                    cycle_weights, debug_landmarks, debug_energy, calibration/single_specimen/, …)
#   results/calibration/  (individual_optimize, generalized_optimize, single_specimen,
#                          specimen_apparent_bn_bp.csv, calibration reports, …)
#   summary_statistics/
#
# Does not remove docs/readme snapshots or run_snapshots/.
# Run from repo root: ./clean_outputs.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

for d in data/filtered data/resampled data/cycle_points_original data/cycle_points_resampled; do
  if [[ -d "$ROOT/$d" ]]; then
    find "$ROOT/$d" -mindepth 1 -maxdepth 1 -exec rm -rf {} + 2>/dev/null || true
  fi
done

rm -rf "$ROOT/results/plots"

if [[ -d "$ROOT/results/calibration" ]]; then
  rm -rf "$ROOT/results/calibration"/*
fi

SUM_DIR="$ROOT/summary_statistics"
if [[ -d "$SUM_DIR" ]]; then
  find "$SUM_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf {} + 2>/dev/null || true
fi

echo "Clean complete (kept data/raw and config/calibration inputs; wiped postprocess data/*, all of results/plots and results/calibration, and summary_statistics/)."
