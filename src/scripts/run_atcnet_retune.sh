#!/usr/bin/env bash
# ATCNet with the grid actually searched (docs section 10.3, first SOTA cell).
#
# The as-specified sweep selected the *top* of the learning-rate grid on 4 of 9
# subjects narrowband and 3 of 9 broadband, and an epoch at or near the 200
# ceiling on 3 of 9 and 2 of 9. A grid whose endpoint keeps winning has been
# truncated, not searched, and a budget that keeps binding is a budget, not a
# converged result.
#
# The paper's own claim is that an under-tuned classical baseline produces a
# qualitatively wrong verdict. Shipping this ATCNet without widening the grid
# would be that mistake committed knowingly, in the direction that flatters the
# foundation models by shrinking the gap they must clear.
#
# Both bands get the identical widened grid and budget: the decomposition needs
# arms that differ only in input, so an asymmetric retune would be worse than no
# retune. Results are tagged `retune`, so the as-specified arms stay on disk and
# the pair remains auditable -- the same convention as the tuned/untuned
# ShallowConvNet pair in section 8h.
#
# ATCNet ends up with a larger budget than ShallowConvNet's 200 epochs. That is
# deliberate and conservative: we want the strongest supervised comparator we
# can build, and every extra epoch spent on it makes the foundation-model
# deficit harder to claim, not easier.
set -uo pipefail

ROOT=/Volumes/kz_extended/CalibMI
OUT="${ROOT}/results/supervised_sota"
SUBJ="1 2 3 4 5 6 7 8 9"
GRID="3e-2 1e-2 3e-3 1e-3 3e-4"
EPOCHS=400

cd "${ROOT}" || { echo "FATAL: ${ROOT} unreachable" >&2; exit 1; }

for ds in bci4_2a bnci2014_004; do
  for band in narrowband broadband; do
    if [ "${band}" = "narrowband" ]; then
      [ "${ds}" = "bci4_2a" ] && dir="${ROOT}/data" || dir="${ROOT}/data/narrowband"
    else
      dir="${ROOT}/data/fm"
    fi

    stem="supervised_atcnet_${band}_retune_tuned"
    [ "${ds}" != "bci4_2a" ] && stem="${stem}_${ds}"
    csv="${OUT}/${stem}_subject_metrics.csv"

    if [ -f "${csv}" ]; then
      echo "########## ${stem} -- already on disk, skipping ##########"; continue
    fi
    cd "${ROOT}" || { echo "FATAL: ${ROOT} went away" >&2; exit 1; }
    for s in ${SUBJ}; do
      [ -f "${dir}/${ds}_subject${s}_train.npz" ] || { echo "SKIPPED ${stem}: missing inputs" >&2; continue 2; }
    done

    echo "########## ${stem} ##########"
    python -u "${ROOT}/src/experiments/run_supervised_on_fm_data.py" \
      --arch atcnet --band "${band}" --fm-dir "${dir}" --dataset "${ds}" \
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

# Re-check the diagnostic that motivated this: if the widened grid still selects
# an endpoint, the retune has not settled the question either.
cd "${ROOT}" && python - <<'PY'
import glob, json, pandas as pd
lo, hi = 3e-4, 3e-2
for f in sorted(glob.glob("results/supervised_sota/supervised_atcnet_*_retune_tuned*_subject_metrics.csv")):
    d = pd.read_csv(f)
    a = [json.loads(r) for r in d["audit_json"]]
    lrs = [x["selected_config"]["lr"] for x in a]
    eps = [x["selected_epoch"] for x in a]
    budget = max(x["epochs"] for x in a)
    print(
        f"{f.split('/')[-1]:60s} acc {d.heldout_accuracy.mean():.4f} "
        f"top-edge {sum(l == hi for l in lrs)}/9 bottom-edge {sum(l == lo for l in lrs)}/9 "
        f"epoch-ceiling {sum(e >= 0.9 * budget for e in eps)}/9"
    )
PY
echo RETUNEDONE
