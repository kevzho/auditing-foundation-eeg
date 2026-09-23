#!/usr/bin/env bash
set -euo pipefail

export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.mplconfig}"

python src/experiments/run_calibration_workflow.py \
  --merge-summaries \
  --results-dir results

python src/scripts/make_calibration_paper_stats.py \
  --bci4-metrics results/calibration_workflow/validation_only_selection_subject_metrics.csv \
  --external-comparison results/validation_only_selection_dataset_comparison.csv \
  --out-dir results/paper_calibration_stats \
  --bootstrap 10000 \
  --seed 20260630
