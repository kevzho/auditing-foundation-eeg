#!/usr/bin/env bash
set -euo pipefail

export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.mplconfig}"

python src/scripts/make_reliability_figures.py \
  --probability-dir results/calibration_workflow/probabilities \
  --probability-dir results/calibration_workflow_bci_iiia/probabilities \
  --probability-dir results/calibration_workflow_bnci2014_001/probabilities \
  --probability-dir results/calibration_workflow_bnci2014_004/probabilities \
  --out-dir results/paper_calibration_stats/reliability \
  --bins 10
