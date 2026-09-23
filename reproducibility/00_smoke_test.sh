#!/usr/bin/env bash
set -euo pipefail

export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.mplconfig}"

python src/experiments/run_calibration_workflow.py \
  --dataset bci4_2a \
  --subjects 1 \
  --experiments baseline \
  --dry-run \
  --results-dir results/calibration_workflow_smoke
