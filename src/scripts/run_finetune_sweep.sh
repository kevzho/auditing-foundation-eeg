#!/usr/bin/env bash
# Fine-tuning recipe sweep for the LaBraM probe.
#
# Motivation: at the reference recipe (lr 5e-4, all parameters trainable) the
# best selection-Brier epoch was 0-4 out of 60 on all 18 runs -- earlier than
# the 6-epoch warmup finishes. Lowering the learning rate to 1e-4 does not fix
# it; selection Brier instead climbs from 0.83 to 1.11, i.e. past chance and
# confidently wrong. A 5.8M-parameter transformer has ~232 fit trials per
# subject, so the sweep has to move capacity as well as step size.
#
# Three subjects span the supervised range (s1 mid, s4 weakest, s9 strongest);
# the winner is then run on all nine.
set -euo pipefail

OUT=results/fm_probe_sweep
SUBJECTS="1 4 9"
COMMON="--mode finetune --patches 4 --subjects ${SUBJECTS} --epochs 60 --device mps --no-deterministic --out-dir ${OUT}"

mkdir -p "${OUT}"

run () {
  local tag="$1"; shift
  echo "=== ${tag}: $* ==="
  python src/experiments/run_fm_probe.py ${COMMON} --tag "${tag}" "$@" 2>&1 | grep -v FutureWarning | grep -v "warnings.warn"
}

# Axis 1: step size, everything trainable.
run lr1e4       --lr 1e-4
run lr1e5       --lr 1e-5

# Axis 2: capacity. Freeze the embeddings and the lowest N blocks.
run frz10_lr1e4 --lr 1e-4 --freeze-blocks 10
run frz10_lr1e5 --lr 1e-5 --freeze-blocks 10
run headonly    --lr 1e-3 --freeze-blocks 12

# Axis 3: LP-FT -- let the random head settle before the encoder moves at all.
run lpft_lr1e4  --lr 1e-4 --head-warmup-epochs 15

echo "sweep complete"
