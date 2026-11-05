#!/usr/bin/env bash
# Repeat MPS runs on a deterministic CPU backend in a separate output directory.
# The untagged LaBraM run is omitted because its JSON lacks recipe parameters.
set -euo pipefail

OUT=results/fm_probe_cpu
mkdir -p "${OUT}"
COMMON="--device cpu --patches 4 --epochs 60 --resume --out-dir ${OUT}"

# Finished CSVs skip a config; --resume handles completed subjects within one.
step () {
  local n="$1" csv="$2" desc="$3"; shift 3
  if [[ -f "${OUT}/${csv}" ]]; then
    echo "=== [${n}/7] SKIP (already complete): ${desc} ==="
    return 0
  fi
  echo "=== [${n}/7] ${desc} ==="
  "$@"
}

step 1 fm_probe_cbramod_finetune_sel_subject_metrics.csv \
  "cbramod finetune, bci4_2a (headline: 0.4796 on MPS)" \
  python src/experiments/run_bd_fm_probe.py --model cbramod --mode finetune \
  --dataset bci4_2a --tag sel --lr 5e-4 --freeze-blocks 0 ${COMMON}

step 2 fm_probe_cbramod_finetune_randinit_sel_subject_metrics.csv \
  "cbramod finetune random-init control, bci4_2a" \
  python src/experiments/run_bd_fm_probe.py --model cbramod --mode finetune \
  --dataset bci4_2a --tag sel --lr 5e-4 --freeze-blocks 0 --random-init ${COMMON}

step 3 fm_probe_cbramod_finetune_bnci2014_004_sel_subject_metrics.csv \
  "cbramod finetune, bnci2014_004 (headline: 0.8085 on MPS)" \
  python src/experiments/run_bd_fm_probe.py --model cbramod --mode finetune \
  --dataset bnci2014_004 --tag sel --lr 5e-4 --freeze-blocks 0 ${COMMON}

step 4 fm_probe_cbramod_finetune_bnci2014_004_randinit_sel_subject_metrics.csv \
  "cbramod finetune random-init control, bnci2014_004" \
  python src/experiments/run_bd_fm_probe.py --model cbramod --mode finetune \
  --dataset bnci2014_004 --tag sel --lr 5e-4 --freeze-blocks 0 --random-init ${COMMON}

step 5 fm_probe_finetune_sel_subject_metrics.csv \
  "labram finetune (frz10_lr1e4, validation-selected recipe)" \
  python src/experiments/run_fm_probe.py --mode finetune \
  --dataset bci4_2a --tag sel --lr 1e-4 --freeze-blocks 10 ${COMMON}

step 6 fm_probe_finetune_randinit_sel_subject_metrics.csv \
  "labram finetune random-init control" \
  python src/experiments/run_fm_probe.py --mode finetune \
  --dataset bci4_2a --tag sel --lr 1e-4 --freeze-blocks 10 --random-init ${COMMON}

step 7 fm_probe_supervised_broadband_tuned_subject_metrics.csv \
  "supervised convnet on broadband input, architecture tuned (headline: 0.6084)" \
  python src/experiments/run_supervised_on_fm_data.py \
  --dataset bci4_2a --tune --epochs 200 --device cpu --out-dir "${OUT}"

echo
echo "All deterministic reruns complete. Compare against the MPS artifacts with:"
echo "  python src/scripts/compare_platform_runs.py \\"
echo "    --reference results/fm_probe/<config>_subject_metrics.csv \\"
echo "    --candidate ${OUT}/<config>_subject_metrics.csv \\"
echo "    --reference-label mps-nondeterministic --candidate-label cpu-deterministic \\"
echo "    --out-dir results/paper_calibration_stats/platform_check"
