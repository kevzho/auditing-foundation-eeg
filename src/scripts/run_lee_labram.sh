#!/usr/bin/env bash
# LaBraM on Lee2019_MI at n=54 (docs section 10.2, v3).
#
# Why this is not optional. The v3 paper argues that nine-subject frozen-probe
# nulls are underpowered rather than negative -- demonstrated by CBraMod, whose
# n=9 null (+0.050, p 0.098) became p 3.9e-04 at n=54 with the effect size
# barely moving. The paper's most striking secondary finding is that LaBraM
# behaves completely differently under the same control, separating from its
# random-init baseline nowhere at all.
#
# That secondary finding now rests on exactly the evidence standard the paper's
# own thesis rejects. Asserting "LaBraM's pretraining does not transfer" from
# n=9, in a paper whose argument is that n=9 cannot support such a claim, is
# the error the paper is named after. Either this runs, or the LaBraM
# comparison is reported as undetermined.
#
# The outcome is genuinely open, and both directions are worth the compute:
#
#   - LaBraM stays null at n=54  -> the two models really do differ, the
#     finding is real, and it is now supported at a sample size the paper's
#     own argument accepts. This is the strongest available outcome.
#   - LaBraM separates at n=54   -> the model difference was itself a power
#     artifact, the paper loses a secondary claim, and its central thesis gets
#     a second independent demonstration. Also publishable, and arguably a
#     cleaner story: one mechanism explaining everything.
#
# LaBraM runs through run_fm_probe.py, not the braindecode runner -- CBraMod
# arrived via braindecode and LaBraM did not, and the two write different
# stems for the same experiment. Flags are transcribed from the n=9 `sel`
# configuration (fm_probe_finetune_sel_run.json): lr 1e-4, 10 frozen blocks,
# 60 epochs, batch 16. That recipe was selected on BCI IV-2a validation.
#
# NOTE ON THE `sel` TAG: unlike run_lee2019.sh, this script does not re-earn
# the label with a per-dataset sweep, so the tag is inherited rather than
# measured here. That is a known, deliberate shortcut -- the comparison being
# made is pretrained vs random init under an *identical* recipe, which is
# valid whether or not the recipe is optimal for Lee2019. Do not quote these
# arms as "LaBraM's best on Lee2019"; quote them only as the paired control.
set -uo pipefail

ROOT=/Volumes/kz_extended/CalibMI
cd "${ROOT}" || { echo "FATAL: ${ROOT} unreachable" >&2; exit 1; }

SUBJ=""
for s in $(seq 1 54); do
  [ -f "${ROOT}/data/fm/lee2019_mi_subject${s}_train.npz" ] \
  && [ -f "${ROOT}/data/fm/lee2019_mi_subject${s}_eval.npz" ] \
  && SUBJ="${SUBJ}${SUBJ:+ }${s}"
done
N=$(echo ${SUBJ} | wc -w | tr -d ' ')
echo "[lee-labram] ${N} subjects available"
[ "${N}" -ge 40 ] || { echo "FATAL: only ${N} subjects, expected ~54" >&2; exit 1; }

step () { echo; echo "########## $(date '+%Y-%m-%d %H:%M:%S')  $* ##########"; }

FT="--mode finetune --dataset lee2019_mi --subjects ${SUBJ} --patches 4 \
    --device mps --no-deterministic --epochs 60 --lr 1e-4 --freeze-blocks 10 \
    --finetune-batch-size 16 --out-dir results/fm_probe --tag sel --resume"

# Frozen first: it is the cheap arm and it is the one the paper's thesis turns
# on. If anything interrupts this, the frozen pair is what we want on disk.
step "lee labram frozen (pretrained)"
python -u src/experiments/run_fm_probe.py --mode frozen --dataset lee2019_mi \
  --subjects ${SUBJ} --patches 4 --device cpu --out-dir results/fm_probe

step "lee labram frozen (random-init control)"
python -u src/experiments/run_fm_probe.py --mode frozen --dataset lee2019_mi \
  --subjects ${SUBJ} --patches 4 --device cpu --out-dir results/fm_probe --random-init

step "lee labram finetune (pretrained)"
python -u src/experiments/run_fm_probe.py ${FT}

step "lee labram finetune (random-init control)"
python -u src/experiments/run_fm_probe.py ${FT} --random-init

echo LEELABRAMDONE
