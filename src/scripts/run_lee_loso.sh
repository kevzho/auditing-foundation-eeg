#!/usr/bin/env bash
# Lee2019_MI pooled leave-one-subject-out probe on 54 subjects and 62 channels.
# Both arms use deterministic CPU runs and resume completed folds.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}" || { echo "FATAL: ${ROOT} unreachable" >&2; exit 1; }

# Use the same available subjects for both arms.
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

step "lee loso cbramod (pretrained)"
python -u src/experiments/run_fm_loso.py --model cbramod --dataset lee2019_mi \
  --subjects ${SUBJ} --patches 4 --device cpu --out-dir results/fm_probe --resume

step "lee loso cbramod (random-init control)"
python -u src/experiments/run_fm_loso.py --model cbramod --dataset lee2019_mi \
  --subjects ${SUBJ} --patches 4 --device cpu --out-dir results/fm_probe --random-init --resume

echo LEELOSODONE
