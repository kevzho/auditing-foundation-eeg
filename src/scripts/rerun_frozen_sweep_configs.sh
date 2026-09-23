#!/usr/bin/env bash
# Re-run the sweep configurations that use --freeze-blocks.
#
# Those were first produced before frozen blocks were held in eval mode during
# training. Dropout was therefore still firing inside blocks that could not
# learn from it. The fix changes their numbers, so the sweep table has to be
# regenerated from the same code as the production runs -- otherwise the recipe
# is selected under one implementation and reported under another.
#
# Configurations without freezing (lr1e4, lr1e5, lpft_lr1e4) are unaffected:
# with nothing frozen, the eval-mode pass is a no-op.
set -euo pipefail

OUT=results/fm_probe_sweep
COMMON="--mode finetune --patches 4 --subjects 1 4 9 --epochs 60 --device mps --no-deterministic --out-dir ${OUT}"

run () {
  local tag="$1"; shift
  echo "########## sweep ${tag}: $* ##########"
  python src/experiments/run_fm_probe.py ${COMMON} --tag "${tag}" "$@"
}

run frz10_lr1e4 --lr 1e-4 --freeze-blocks 10
run frz10_lr1e5 --lr 1e-5 --freeze-blocks 10
run headonly    --lr 1e-3 --freeze-blocks 12

echo "sweep reruns complete"
