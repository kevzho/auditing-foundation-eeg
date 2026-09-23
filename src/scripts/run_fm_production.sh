#!/usr/bin/env bash
# Full n=9 foundation-model runs for the audit.
#
# Ordered so the highest-value results land first: if this is interrupted, what
# has already been written is still a complete story.
#
# Each control matches its counterpart's platform exactly. The frozen results
# already on disk were produced on CPU with deterministic algorithms enabled and
# patches {2,4}; the fine-tuning results on MPS with determinism off (MPS has no
# deterministic index_put kernel). Mixing those would confound the pretrained vs
# random-init contrast with a platform difference, which is the very thing
# compare_platform_runs.py exists to detect.
set -euo pipefail

OUT=results/fm_probe
FROZEN="--mode frozen --subjects 1 2 3 4 5 6 7 8 9 --device cpu --out-dir ${OUT}"
FT="--mode finetune --subjects 1 2 3 4 5 6 7 8 9 --device mps --no-deterministic --epochs 60 --out-dir ${OUT}"

# Fine-tuning recipe chosen on VALIDATION Brier in the sweep (see
# run_finetune_sweep.sh): freezing the embeddings and lowest 10 blocks at
# lr 1e-4 wins on validation, and independently on held-out too. Rank agreement
# across all six configurations is only moderate (Spearman 0.60, n=6), so the
# claim is that this recipe wins under both criteria -- not that validation
# Brier reliably orders recipes in general.
RECIPE="--freeze-blocks 10 --lr 1e-4"

step () { echo; echo "########## $* ##########"; }

# --- 1. CBraMod, the second foundation model: frozen + its control ----------
step "cbramod frozen (pretrained)"
python src/experiments/run_bd_fm_probe.py --model cbramod ${FROZEN} --patches 2 4

step "cbramod frozen (random init control)"
python src/experiments/run_bd_fm_probe.py --model cbramod ${FROZEN} --patches 2 4 --random-init

# --- 2. LaBraM random-init control for the frozen result already on disk ----
step "labram frozen (random init control)"
python src/experiments/run_fm_probe.py ${FROZEN} --patches 2 4 --random-init

# --- 3. LaBraM fine-tuning with the validation-selected recipe --------------
step "labram finetune (validation-selected recipe)"
python src/experiments/run_fm_probe.py ${FT} --patches 4 ${RECIPE} --tag sel

step "labram finetune (random init control)"
python src/experiments/run_fm_probe.py ${FT} --patches 4 ${RECIPE} --tag sel --random-init

# --- 4. CBraMod fine-tuning: small recipe sweep, then n=9 -------------------
# The recipe is selected per model on validation, not carried over from LaBraM.
step "cbramod finetune recipe sweep (subjects 1 4 9)"
for cfg in "default::" "frz10_lr1e4:--freeze-blocks 10 --lr 1e-4:" "lr1e5:--lr 1e-5:"; do
  tag="${cfg%%:*}"; rest="${cfg#*:}"; flags="${rest%%:*}"
  echo "--- sweep ${tag} ${flags} ---"
  python src/experiments/run_bd_fm_probe.py --model cbramod --mode finetune \
    --subjects 1 4 9 --device mps --no-deterministic --epochs 60 \
    --out-dir results/fm_probe_sweep --tag "${tag}" ${flags}
done

echo
echo "production runs complete; pick the CBraMod recipe on validation Brier, then run n=9"
