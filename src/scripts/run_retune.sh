#!/usr/bin/env bash
# Re-run a braindecode comparator with a learning-rate grid that is actually
# searched rather than truncated.  Usage: run_retune.sh <arch>
#
# The as-specified sweep selected a grid *endpoint* on most subjects for both
# architectures, in opposite directions:
#
#   ATCNet        (113,732 params)  top edge 1e-2  on 4/9 narrowband, 3/9 broadband
#   EEGConformer  (789,572 params)  bottom edge 3e-4 on 6/9 narrowband
#
# A grid whose endpoint keeps winning has not been searched. Each architecture
# therefore gets the grid extended in the direction it was pushing, which is why
# this takes the arch as an argument instead of applying one grid to both --
# a shared grid would leave one of them still pinned at an edge.
#
# The paper's own claim is that an under-tuned classical baseline produces a
# qualitatively wrong verdict. Shipping either comparator pinned to an edge
# would be that mistake committed knowingly, in the direction that flatters the
# foundation models by shrinking the gap they must clear.
#
# Both bands get the identical grid and budget: the decomposition needs arms
# differing only in input, so an asymmetric retune would be worse than none.
# Results are tagged `retune`, leaving the as-specified arms on disk so the pair
# stays auditable -- the same convention as section 8h's tuned/untuned pair.
#
# 400 epochs for both. ATCNet was hitting the 200 ceiling on 3/9; EEGConformer
# was not, but it is being pushed to lower learning rates and lower rates
# converge more slowly, so keeping the budget binding there would reintroduce
# the same problem from the other side.
set -uo pipefail

ARCH="${1:?usage: run_retune.sh <atcnet|eegconformer>}"
ROOT=/Volumes/kz_extended/CalibMI
OUT="${ROOT}/results/supervised_sota"
SUBJ="1 2 3 4 5 6 7 8 9"
EPOCHS=400

case "${ARCH}" in
  atcnet)       GRID="3e-2 1e-2 3e-3 1e-3 3e-4" ;;
  eegconformer) GRID="1e-3 3e-4 1e-4 3e-5 1e-5" ;;
  *) echo "unknown arch ${ARCH}" >&2; exit 2 ;;
esac

cd "${ROOT}" || { echo "FATAL: ${ROOT} unreachable" >&2; exit 1; }

for ds in bci4_2a bnci2014_004; do
  for band in narrowband broadband; do
    if [ "${band}" = "narrowband" ]; then
      [ "${ds}" = "bci4_2a" ] && dir="${ROOT}/data" || dir="${ROOT}/data/narrowband"
    else
      dir="${ROOT}/data/fm"
    fi

    stem="supervised_${ARCH}_${band}_retune_tuned"
    [ "${ds}" != "bci4_2a" ] && stem="${stem}_${ds}"
    csv="${OUT}/${stem}_subject_metrics.csv"

    if [ -f "${csv}" ]; then
      echo "########## ${stem} -- already on disk, skipping ##########"; continue
    fi
    cd "${ROOT}" || { echo "FATAL: ${ROOT} went away" >&2; exit 1; }
    skip=0
    for s in ${SUBJ}; do
      [ -f "${dir}/${ds}_subject${s}_train.npz" ] || skip=1
    done
    [ "${skip}" -eq 0 ] || { echo "SKIPPED ${stem}: inputs incomplete" >&2; continue; }

    echo "########## ${stem} ##########"
    python -u "${ROOT}/src/experiments/run_supervised_on_fm_data.py" \
      --arch "${ARCH}" --band "${band}" --fm-dir "${dir}" --dataset "${ds}" \
      --subjects ${SUBJ} --device mps --no-deterministic \
      --epochs "${EPOCHS}" --tune --lr-grid ${GRID} --tag retune \
      --out-dir "${OUT}" 2>&1 | grep --line-buffered -viE "warning|Malloc"

    if [ ! -f "${csv}" ]; then
      echo "FAILED ${stem}: no output written" >&2
    else
      n=$(( $(wc -l < "${csv}") - 1 ))
      [ "${n}" -eq 9 ] || echo "FAILED ${stem}: ${n} of 9 subjects" >&2
    fi
  done
done

# The diagnostic that motivated this. A widened grid still selecting an endpoint
# has not settled the question either, and saying so is the point.
cd "${ROOT}" && ARCH="${ARCH}" GRID="${GRID}" python - <<'PY'
import glob, json, os, pandas as pd
arch = os.environ["ARCH"]
grid = [float(v) for v in os.environ["GRID"].split()]
lo, hi = min(grid), max(grid)
for f in sorted(glob.glob(f"results/supervised_sota/supervised_{arch}_*_retune_tuned*_subject_metrics.csv")):
    d = pd.read_csv(f)
    a = [json.loads(r) for r in d["audit_json"]]
    lrs = [x["selected_config"]["lr"] for x in a]
    eps = [x["selected_epoch"] for x in a]
    budget = max(x["epochs"] for x in a)
    print(
        f"{f.split('/')[-1]:66s} acc {d.heldout_accuracy.mean():.4f} "
        f"top {sum(l == hi for l in lrs)}/9 bottom {sum(l == lo for l in lrs)}/9 "
        f"ceiling {sum(e >= 0.9 * budget for e in eps)}/9"
    )
PY
echo "RETUNEDONE ${ARCH}"
