#!/usr/bin/env bash
# State-of-the-art supervised comparators (docs section 10.3, priority 1).
#
# The paper's own finding is that baseline quality decides a benchmark's
# verdict. Comparing foundation models only against ShallowConvNet -- a 2017
# reference architecture inherited from the previous project and never chosen
# for this purpose -- would repeat the mistake the paper documents.
#
# ATCNet and EEGConformer are the strongest architectures cited in the
# manuscript's literature paragraph, taken from braindecode's reference
# implementations. Each gets a per-subject learning-rate grid selected on the
# validation subset, so they are not handicapped by an unsearched optimiser.
#
# Both bands, both datasets: the narrowband arm gives the headline
# FM-vs-supervised gap, the broadband arm gives the matched-input comparison
# and the preprocessing decomposition.
#
# Hardened after a run lost six cells in nine seconds: the external volume
# remounted mid-sweep (disk8s1 -> disk5s1), the working directory stopped
# resolving, and every remaining `python src/...` died with "can't open file"
# while the loop raced to the end. Nothing crashed loudly; the log simply
# stopped containing results. Hence: absolute root, a liveness check before each
# cell, and resume so a repeat costs only the cell in flight.
set -uo pipefail

ROOT=/Volumes/kz_extended/CalibMI
OUT="${ROOT}/results/supervised_sota"

cd "${ROOT}" || { echo "FATAL: ${ROOT} unreachable" >&2; exit 1; }
mkdir -p "${OUT}"

# A missing input must stop the cell, not be discovered 40 minutes in on
# subject 5. The broadband ATCNet cell died exactly that way.
check_inputs () {  # dir dataset subjects...
  local dir="$1" ds="$2"; shift 2
  for s in "$@"; do
    [ -f "${dir}/${ds}_subject${s}_train.npz" ] || { echo "MISSING ${dir}/${ds}_subject${s}_train.npz" >&2; return 1; }
    [ -f "${dir}/${ds}_subject${s}_eval.npz" ]  || { echo "MISSING ${dir}/${ds}_subject${s}_eval.npz" >&2; return 1; }
  done
  return 0
}

SUBJ="1 2 3 4 5 6 7 8 9"

for arch in atcnet eegconformer; do
  for ds in bci4_2a bnci2014_004; do
    for band in narrowband broadband; do
      if [ "${band}" = "narrowband" ]; then
        [ "${ds}" = "bci4_2a" ] && dir="${ROOT}/data" || dir="${ROOT}/data/narrowband"
      else
        dir="${ROOT}/data/fm"
      fi

      stem="supervised_${arch}_${band}_tuned"
      [ "${ds}" != "bci4_2a" ] && stem="${stem}_${ds}"
      csv="${OUT}/${stem}_subject_metrics.csv"

      if [ -f "${csv}" ]; then
        echo "########## ${arch} ${ds} ${band} -- already on disk, skipping ##########"
        continue
      fi

      echo "########## ${arch} ${ds} ${band} ##########"
      cd "${ROOT}" || { echo "FATAL: ${ROOT} went away" >&2; exit 1; }
      if ! check_inputs "${dir}" "${ds}" ${SUBJ}; then
        echo "SKIPPED ${arch} ${ds} ${band}: inputs incomplete" >&2
        continue
      fi

      python -u "${ROOT}/src/experiments/run_supervised_on_fm_data.py" \
        --arch "${arch}" --band "${band}" --fm-dir "${dir}" --dataset "${ds}" \
        --subjects ${SUBJ} --device mps --no-deterministic \
        --epochs 200 --tune --out-dir "${OUT}" 2>&1 \
        | grep --line-buffered -viE "warning|Malloc"

      # A cell that wrote no CSV, or a short one, is a failure that would
      # otherwise be invisible until analysis.
      if [ ! -f "${csv}" ]; then
        echo "FAILED ${arch} ${ds} ${band}: no output written" >&2
      else
        n=$(( $(wc -l < "${csv}") - 1 ))
        [ "${n}" -eq 9 ] || echo "FAILED ${arch} ${ds} ${band}: ${n} of 9 subjects" >&2
      fi
    done
  done
done
echo SOTADONE
