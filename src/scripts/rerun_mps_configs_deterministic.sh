#!/usr/bin/env bash
# Re-run on a deterministic backend every configuration whose retained result
# was produced on MPS.
#
# MPS has no deterministic index_put kernel, so those runs carry
# deterministic=False in their platform fingerprint. Section 8b of
# docs/fm_uncertainty_audit_scope.md requires final reported numbers to come
# from a deterministic backend, and the two headline results -- CBraMod
# fine-tuned and the tuned broadband convnet -- are both in that set.
#
# Output goes to results/fm_probe_cpu/ rather than overwriting results/fm_probe/,
# so the MPS artifacts survive for the platform-sensitivity comparison that
# src/scripts/compare_platform_runs.py produces.
#
# Parameters below are transcribed from each run's *_run.json. One config is
# deliberately absent: fm_probe_finetune_run.json (LaBraM p2/p4, un-tagged)
# predates the current run-json format and records no lr, epoch count, or
# freeze depth, so it cannot be reproduced faithfully without guessing. Decide
# whether to reconstruct it or to retire those two rows in favour of
# labram_finetune_sel before reporting.
#
# Wall clock: CBraMod fine-tune measured at ~6.6 min/subject on CPU, so roughly
# an hour per 9-subject config and ~5-6 hours for the whole script.
set -euo pipefail

OUT=results/fm_probe_cpu
mkdir -p "${OUT}"
COMMON="--device cpu --patches 4 --epochs 60 --resume --out-dir ${OUT}"

# Two levels of resume. Config level: a finished config writes its CSV only
# after the last subject, so that file's presence proves completion and the
# step is skipped. Subject level: --resume on the runners skips subjects whose
# rows are already in the .partial file, provided the recorded arguments match.
# Re-running this script after any interruption therefore picks up where it
# stopped. Delete a CSV to force a config to run again.
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
