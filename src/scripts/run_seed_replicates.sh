#!/usr/bin/env bash
# Replicate seeds for the configurations the manuscript's headline claims rest on.
#
# Seed 42 is already reported. The seed drives BOTH weight initialisation and
# the fit/selection split (see _split_train_val_indices), so a replicate seed
# varies the two things a single run cannot separate. make_fm_audit_report.py
# collapses seeds within (dataset, experiment_key, subject) before any paired
# test, and reports the per-subject spread separately.
#
# Ordered by how much each result depends on it, so an interruption costs the
# least important work:
#   1. the tuned broadband convnet -- anchors the band/model decomposition, and
#      is where subject 2 selected epoch 2 of 200 on seed 42
#   2. CBraMod fine-tuned + its random-init control on BCI IV-2a -- the +0.1466
#      headline
#   3. the same pair on BNCI2014_004 -- the replication
#
# Roughly 9 hours total on CPU. Both resume levels apply, so re-running after an
# interruption picks up where it stopped.
set -euo pipefail

OUT=results/fm_probe_cpu
mkdir -p "${OUT}"
SEEDS="43 44"

step () {
  local csv="$1" desc="$2"; shift 2
  if [[ -f "${OUT}/${csv}" ]]; then
    echo "=== SKIP (complete): ${desc} ==="
    return 0
  fi
  echo "=== ${desc} ==="
  "$@"
}

for S in ${SEEDS}; do
  step "fm_probe_supervised_broadband_tuned_seed${S}_subject_metrics.csv" \
    "[1] tuned broadband convnet, seed ${S}" \
    python src/experiments/run_supervised_on_fm_data.py \
    --dataset bci4_2a --tune --epochs 200 --device cpu --seed "${S}" --out-dir "${OUT}"
done

for S in ${SEEDS}; do
  step "fm_probe_cbramod_finetune_sel_seed${S}_subject_metrics.csv" \
    "[2] cbramod finetune bci4_2a, seed ${S}" \
    python src/experiments/run_bd_fm_probe.py --model cbramod --mode finetune \
    --dataset bci4_2a --tag sel --lr 5e-4 --freeze-blocks 0 \
    --device cpu --patches 4 --epochs 60 --resume --seed "${S}" --out-dir "${OUT}"

  step "fm_probe_cbramod_finetune_randinit_sel_seed${S}_subject_metrics.csv" \
    "[3] cbramod finetune random-init bci4_2a, seed ${S}" \
    python src/experiments/run_bd_fm_probe.py --model cbramod --mode finetune \
    --dataset bci4_2a --tag sel --lr 5e-4 --freeze-blocks 0 --random-init \
    --device cpu --patches 4 --epochs 60 --resume --seed "${S}" --out-dir "${OUT}"
done

for S in ${SEEDS}; do
  step "fm_probe_cbramod_finetune_bnci2014_004_sel_seed${S}_subject_metrics.csv" \
    "[4] cbramod finetune bnci2014_004, seed ${S}" \
    python src/experiments/run_bd_fm_probe.py --model cbramod --mode finetune \
    --dataset bnci2014_004 --tag sel --lr 5e-4 --freeze-blocks 0 \
    --device cpu --patches 4 --epochs 60 --resume --seed "${S}" --out-dir "${OUT}"

  step "fm_probe_cbramod_finetune_bnci2014_004_randinit_sel_seed${S}_subject_metrics.csv" \
    "[5] cbramod finetune random-init bnci2014_004, seed ${S}" \
    python src/experiments/run_bd_fm_probe.py --model cbramod --mode finetune \
    --dataset bnci2014_004 --tag sel --lr 5e-4 --freeze-blocks 0 --random-init \
    --device cpu --patches 4 --epochs 60 --resume --seed "${S}" --out-dir "${OUT}"
done

echo
echo "Seed replicates complete. Regenerate the report with:"
echo "  python src/scripts/make_fm_audit_report.py \\"
echo "    --fm-dir results/fm_probe --fm-dir ${OUT} --require-deterministic"
