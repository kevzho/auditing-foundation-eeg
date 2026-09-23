#!/usr/bin/env bash
# Lee2019_MI pooled leave-one-subject-out (docs section 8m, second reason).
#
# This is the arm `run_lee2019.sh` omitted. Lee2019 was queued for two reasons
# and that script only served the first: it broke the n=9 ceiling, but it never
# ran the pooled regime, so the reason that actually needed 62 channels went
# unaddressed while the dataset sat preprocessed on disk.
#
# What is being tested. On BNCI2014_004 the pooled regime *reverses*: CBraMod
# fine-tuned beats its random-init control per subject, but pooled across
# subjects the control wins (-0.052). On BCI IV-2a pooling helps (+0.111). The
# standing explanation is that BNCI2014_004 has three electrodes, too sparse
# for a pooled model to learn a subject-invariant spatial filter. Section 8j
# flags that as post-hoc and unfalsified, and it has stayed unfalsified because
# no dense-montage dataset had been run.
#
# Lee2019 is that dataset: 62 channels, two classes, 54 subjects. It shares the
# class count with BNCI2014_004, so class count is held fixed and montage
# density is the thing that varies.
#
#   - Pooling helps at 62 channels  -> the electrode explanation survives a
#     test that could have killed it, and it can be stated as a finding rather
#     than a guess.
#   - Pooling reverses at 62 channels -> the explanation is wrong. Report that.
#     A post-hoc story that fails its first real test is worth more on the page
#     than one that was never tested.
#
# Either outcome is publishable. Only leaving it untested is not.
#
# 54 folds, each pooling 53 subjects' training sessions against the 54th's
# held-out session. That is ~6x the pooled training data of the n=9 runs, so
# expect the per-fold probe fit to dominate rather than feature extraction.
#
# --resume on both arms. The first attempt at the control was killed at fold
# 25 of 54 and lost all 24 completed folds, because rows were held in memory
# and written only after the final fold. run_fm_loso.py now flushes each fold
# to a .partial file and skips folds already on disk, so a repeat kill costs
# one fold. Resumption is refused if the recorded arguments differ.
#
# Frozen linear probe on cpu with deterministic algorithms, matching every
# other LOSO arm in results/fm_probe -- these numbers must sit in the same
# table as the n=9 ones, and a backend change would put them in the report's
# "Reproducibility caveats" section for no reason.
set -uo pipefail

ROOT=/Volumes/kz_extended/CalibMI
cd "${ROOT}" || { echo "FATAL: ${ROOT} unreachable" >&2; exit 1; }

# Same on-disk subject discovery as run_lee2019.sh. A hardcoded `seq 1 54`
# would abort the whole job on the first missing file, and the paired test
# downstream depends on both arms seeing an identical subject set.
SUBJ=""
for s in $(seq 1 54); do
  [ -f "${ROOT}/data/fm/lee2019_mi_subject${s}_train.npz" ] \
  && [ -f "${ROOT}/data/fm/lee2019_mi_subject${s}_eval.npz" ] \
  && SUBJ="${SUBJ}${SUBJ:+ }${s}"
done
N=$(echo ${SUBJ} | wc -w | tr -d ' ')
echo "[lee-loso] ${N} subjects available"
[ "${N}" -ge 40 ] || { echo "FATAL: only ${N} subjects, expected ~54" >&2; exit 1; }

step () { echo; echo "########## $(date '+%Y-%m-%d %H:%M:%S')  $* ##########"; }

# Pretrained first: an interruption then leaves the arm the comparison needs,
# not the control on its own.
step "lee loso cbramod (pretrained)"
python -u src/experiments/run_fm_loso.py --model cbramod --dataset lee2019_mi \
  --subjects ${SUBJ} --patches 4 --device cpu --out-dir results/fm_probe --resume

step "lee loso cbramod (random-init control)"
python -u src/experiments/run_fm_loso.py --model cbramod --dataset lee2019_mi \
  --subjects ${SUBJ} --patches 4 --device cpu --out-dir results/fm_probe --random-init --resume

echo LEELOSODONE
