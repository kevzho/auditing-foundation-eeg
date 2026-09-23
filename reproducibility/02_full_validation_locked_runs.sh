#!/usr/bin/env bash
set -euo pipefail

export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.mplconfig}"

# Primary BCI IV-2a run. Requires data/BCICIV_2a_gdf/*.gdf and official labels.
python src/experiments/run_calibration_workflow.py \
  --dataset bci4_2a \
  --experiments all \
  --save-probabilities \
  --log-epochs 5 \
  --results-dir results/calibration_workflow

# BCI Competition III Dataset IIIa external check. Requires local GDF files.
python src/experiments/run_calibration_workflow.py \
  --dataset bci_iiia \
  --external-confirmatory \
  --allow-stratified-external-split \
  --save-probabilities \
  --log-epochs 5 \
  --results-dir results/calibration_workflow_bci_iiia

# MOABB external checks. These may download/cache data through MOABB.
python src/experiments/run_calibration_workflow.py \
  --dataset BNCI2014_001 \
  --external-confirmatory \
  --save-probabilities \
  --log-epochs 5 \
  --results-dir results/calibration_workflow_bnci2014_001

python src/experiments/run_calibration_workflow.py \
  --dataset BNCI2014_004 \
  --external-confirmatory \
  --save-probabilities \
  --log-epochs 5 \
  --results-dir results/calibration_workflow_bnci2014_004

bash reproducibility/01_readout_from_saved_results.sh
bash reproducibility/03_reliability_figures_from_probabilities.sh
