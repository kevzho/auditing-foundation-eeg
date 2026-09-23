#!/usr/bin/env bash
# Recipe-matched supervised arms for the decomposition (docs section 8h).
#
# Two gaps this closes, both found by auditing rather than planned:
#
#  1. ShallowConvNet's narrowband number came from `run_calibration_workflow.py`
#     (`baseline_ce_cropped_shallow_convnet`: multi-scale crops with
#     aggregation), while its broadband number came from this runner
#     (single windows). Pairing them put the training recipe inside the term
#     labelled "preprocessing". This runs the narrowband arm through the *same*
#     loop so the pair differs only in input.
#  2. ShallowConvNet has both bands on BCI IV-2a only, so the under-tuned /
#     tuned inversion -- the section 8h lesson worth more than the number -- is
#     still a single-dataset result. The BNCI2014_004 broadband arms fix that.
#
# Hardened like the SOTA sweep: absolute root, resume from disk, inputs checked
# first, outputs checked after. The first attempt at arm 1 was killed three
# subjects in when the volume remounted.
set -uo pipefail

ROOT=/Volumes/kz_extended/CalibMI
OUT="${ROOT}/results/supervised_sota"
SUBJ="1 2 3 4 5 6 7 8 9"

cd "${ROOT}" || { echo "FATAL: ${ROOT} unreachable" >&2; exit 1; }
mkdir -p "${OUT}"

run_cell () {  # arch band dir dataset tune_flag stem
  local arch="$1" band="$2" dir="$3" ds="$4" tune="$5" stem="$6"
  local csv="${OUT}/${stem}_subject_metrics.csv"

  if [ -f "${csv}" ]; then
    echo "########## ${stem} -- already on disk, skipping ##########"; return 0
  fi
  cd "${ROOT}" || { echo "FATAL: ${ROOT} went away" >&2; exit 1; }
  for s in ${SUBJ}; do
    [ -f "${dir}/${ds}_subject${s}_train.npz" ] || { echo "SKIPPED ${stem}: missing ${ds}_subject${s}_train.npz" >&2; return 0; }
  done

  echo "########## ${stem} ##########"
  python -u "${ROOT}/src/experiments/run_supervised_on_fm_data.py" \
    --arch "${arch}" --band "${band}" --fm-dir "${dir}" --dataset "${ds}" \
    --subjects ${SUBJ} --device mps --no-deterministic \
    --epochs 200 ${tune} --out-dir "${OUT}" 2>&1 \
    | grep --line-buffered -viE "warning|Malloc"

  if [ ! -f "${csv}" ]; then
    echo "FAILED ${stem}: no output written" >&2
  else
    n=$(( $(wc -l < "${csv}") - 1 ))
    [ "${n}" -eq 9 ] || echo "FAILED ${stem}: ${n} of 9 subjects" >&2
  fi
}

# 1. the recipe-matched narrowband arm -- blocks the whole decomposition.
run_cell shallow_convnet narrowband "${ROOT}/data" bci4_2a --tune \
         supervised_shallow_convnet_narrowband_tuned

# 2. dataset 2, both tuned and untuned, so the inversion lesson replicates.
run_cell shallow_convnet broadband "${ROOT}/data/fm" bnci2014_004 --tune \
         supervised_shallow_convnet_broadband_tuned_bnci2014_004
run_cell shallow_convnet broadband "${ROOT}/data/fm" bnci2014_004 "" \
         supervised_shallow_convnet_broadband_bnci2014_004
run_cell shallow_convnet narrowband "${ROOT}/data/narrowband" bnci2014_004 --tune \
         supervised_shallow_convnet_narrowband_tuned_bnci2014_004

cd "${ROOT}" && python src/scripts/make_preprocessing_decomposition.py
echo CONTROLSDONE
