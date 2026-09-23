#!/usr/bin/env bash
# Split the preprocessing term into ICA and filter parts (docs section 8h,
# second correction).
#
# The term section 8h called "the filter band" is really the whole classical
# pipeline: 8-30 Hz IIR *plus EOG-guided ICA* at 250 Hz, against 0.1-75 Hz FIR
# with no ICA at 200 Hz. ICA is artifact removal, not a filter setting, and
# attributing its effect to the band would be wrong.
#
# `preprocess_fm.py` already takes --ica, so one broadband arm with ICA applied
# and everything else held fixed separates the two:
#
#   ICA term          broadband+ICA  vs  broadband        -> artifact removal
#   band/rate/design  narrowband     vs  broadband+ICA    -> the filter itself
#   model term        broadband      vs  foundation model -> unchanged
#
# No downloads: this re-reads the BCI IV-2a GDFs already on disk. Run after the
# SOTA sweep; it shares the GPU.
#
# BLOCKED until run_supervised_on_fm_data.py grows --tag. Without it the ICA arm
# writes experiment_key `shallow_convnet_broadband_tuned` -- byte-identical to
# the no-ICA arm it is supposed to be contrasted against. Two arms differing
# only in ICA, sharing one key and one filename, is the silent confound this
# whole control exists to remove. The guard below refuses to run rather than
# produce that. The flag is not added yet because the SOTA sweep has cells
# pending in separate processes that re-read the runner.
set -uo pipefail

ROOT=/Volumes/kz_extended/CalibMI
cd "${ROOT}" || { echo "FATAL: ${ROOT} unreachable" >&2; exit 1; }

if ! python src/experiments/run_supervised_on_fm_data.py --help 2>&1 | grep -q -- "--tag"; then
  echo "refusing to run: run_supervised_on_fm_data.py has no --tag, so the ICA" >&2
  echo "arm would overwrite / be indistinguishable from the no-ICA arm." >&2
  exit 2
fi

ICA_DIR=data/fm_ica
OUT=results/supervised_sota

if [ "$(ls "${ROOT}/${ICA_DIR}"/bci4_2a_subject*.npz 2>/dev/null | wc -l)" -ge 18 ]; then
  echo "########## ica arrays already on disk, skipping preprocessing ##########"
else
  echo "########## preprocess bci4_2a broadband WITH ica ##########"
  python -u "${ROOT}/src/preprocess_fm.py" --npz-dir "${ROOT}/${ICA_DIR}" --ica 2>&1 \
    | grep --line-buffered -viE "warning|Malloc"
fi

# Both architectures, because they disagree about the sign of the pipeline term
# (docs section 8h): ShallowConvNet says the FM-required input costs +0.037,
# ATCNet says it helps -0.077. If ICA is what separates them, that shows up as
# the two architectures disagreeing here too -- and if it is not, the
# disagreement is about the band and the rate, which is a different sentence in
# the paper. One architecture could not tell those apart.
#
# Same grid, loop and device as the no-ICA broadband arms, so the only thing
# differing between an arm and its counterpart is ICA. Verified on disk: the
# provenance blobs differ in `ica_applied` and nothing else.
for arch in shallow_convnet atcnet; do
  stem="supervised_${arch}_broadband_ica_tuned"
  csv="${OUT}/${stem}_subject_metrics.csv"
  if [ -f "${csv}" ]; then
    echo "########## ${stem} -- already on disk, skipping ##########"; continue
  fi
  cd "${ROOT}" || { echo "FATAL: ${ROOT} went away" >&2; exit 1; }
  echo "########## ${stem} ##########"
  python -u "${ROOT}/src/experiments/run_supervised_on_fm_data.py" \
    --arch "${arch}" --band broadband --fm-dir "${ROOT}/${ICA_DIR}" --dataset bci4_2a \
    --subjects 1 2 3 4 5 6 7 8 9 --device mps --no-deterministic \
    --epochs 200 --tune --tag ica --out-dir "${OUT}" 2>&1 \
    | grep --line-buffered -viE "warning|Malloc"
  if [ ! -f "${csv}" ]; then
    echo "FAILED ${stem}: no output written" >&2
  else
    n=$(( $(wc -l < "${csv}") - 1 ))
    [ "${n}" -eq 9 ] || echo "FAILED ${stem}: ${n} of 9 subjects" >&2
  fi
done

cd "${ROOT}" && python src/scripts/make_preprocessing_decomposition.py
echo ICADONE
