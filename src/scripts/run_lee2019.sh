#!/usr/bin/env bash
# Lee2019_MI experiments (docs section 10.3, priority 2).
#
# Why this dataset earns its cost:
#
#   1. It breaks the n=9 ceiling. With nine subjects the smallest attainable
#      two-sided Wilcoxon p is 2^(1-9) = 0.0039 no matter how large the effect,
#      and power against modest effects is poor. At n=54 that ceiling is gone.
#   2. It is the dense-montage dataset. BNCI2014_004's failed pooling result
#      (docs section 8j) is currently explained post-hoc by its three
#      electrodes; 62 channels tests that explanation instead of asserting it.
#
# Protocol matches the rest of the project: session 1 calibrates, session 2 is
# held out. That is a genuine cross-day split, not the within-session run split
# that MOABB's run labels would otherwise suggest.
#
# Order matters. The pretrained-vs-random-init contrast is the paper's headline
# and runs first, so an interruption still leaves the primary result complete.
set -uo pipefail

ROOT=/Volumes/kz_extended/CalibMI
cd "${ROOT}" || { echo "FATAL: ${ROOT} unreachable" >&2; exit 1; }

# Subject list comes from what is on disk, not from `seq 1 54`. Two subjects
# were lost to Wasabi read timeouts during preprocessing, and a hardcoded range
# would abort the whole 52-subject job on the first missing file. Requiring
# *both* bands keeps every arm on an identical subject set, which the paired
# tests depend on -- a supervised arm silently running on more subjects than the
# foundation-model arm would break the pairing rather than error.
SUBJ=""
for s in $(seq 1 54); do
  have=1
  for d in fm narrowband; do
    for k in train eval; do
      [ -f "${ROOT}/data/${d}/lee2019_mi_subject${s}_${k}.npz" ] || have=0
    done
  done
  [ "${have}" -eq 1 ] && SUBJ="${SUBJ}${SUBJ:+ }${s}"
done
N=$(echo ${SUBJ} | wc -w | tr -d ' ')
echo "[lee2019] ${N} subjects have both bands; running on those"
[ "${N}" -ge 40 ] || { echo "FATAL: only ${N} subjects available, expected ~54" >&2; exit 1; }
OUT=results/fm_probe
FT="--mode finetune --dataset lee2019_mi --subjects ${SUBJ} --device mps --no-deterministic --epochs 60 --out-dir ${OUT}"

# Timestamped so the three-subject sweep doubles as the timing probe for the
# 54-subject arms. Nothing in the repo records wall-clock, so the cost of this
# run has never been measured -- extrapolate from the sweep before assuming the
# production arms fit the calendar.
step () { echo; echo "########## $(date '+%Y-%m-%d %H:%M:%S')  $* ##########"; }

# --- 0. earn the "sel" label on this dataset -------------------------------
# The recipe tagged `sel` elsewhere was selected on BCI IV-2a validation, where
# the sweep winner happened to coincide with the runner defaults (lr 5e-4,
# no freezing, no head warmup, 60 epochs). Carrying that label to a dataset with
# 62 channels, two classes and six times the subjects would be assuming what it
# claims to have measured. Same three-config sweep, same protocol, scored on
# validation Brier only -- on three subjects, as on BCI IV-2a, so it costs
# little against a 54-subject run.
#
# If the winner is not "default", the production arms below must be updated
# before they run, or the `sel` tag is a lie. That check is enforced after the
# sweep rather than left to whoever reads the log: a comment saying STOP does
# not stop anything, and this script is meant to be launched and left alone.
# Three spread-out subjects drawn from the available set rather than named
# literally: subject 48 and 49 were lost to download timeouts, and a hardcoded
# id would either crash the sweep or silently pick a different subject than
# the comment claims.
SWEEP_SUBJ=$(echo ${SUBJ} | awk '{print $1, $(int(NF/2)), $NF}')
step "lee cbramod finetune recipe sweep (subjects ${SWEEP_SUBJ})"
for cfg in "default::" "frz10_lr1e4:--freeze-blocks 10 --lr 1e-4:" "lr1e5:--lr 1e-5:"; do
  tag="${cfg%%:*}"; rest="${cfg#*:}"; flags="${rest%%:*}"
  echo "--- sweep ${tag} ${flags} ---"
  python -u src/experiments/run_bd_fm_probe.py --model cbramod --mode finetune \
    --dataset lee2019_mi --subjects ${SWEEP_SUBJ} --device mps --no-deterministic \
    --epochs 60 --out-dir results/fm_probe_sweep --tag "lee_${tag}" ${flags}
done

# --- 0b. enforce the sweep verdict -----------------------------------------
# Rank on *validation* Brier, never held-out -- ranking on held-out is the exact
# selection violation the whole protocol exists to prevent, and doing it here
# would contaminate the 54-subject run that follows.
#
# The production arms below hardcode the runner defaults, which is what the
# `sel` tag claims was selected. If a different config wins on this dataset,
# that claim is false and the honest move is to stop and update the flags, not
# to ship a tag that names a selection that never happened.
step "sweep verdict"
WINNER=$(python - <<'PY'
import glob, sys
import pandas as pd
rows = []
for f in glob.glob("results/fm_probe_sweep/*lee_*_subject_metrics.csv"):
    d = pd.read_csv(f)
    tag = f.split("lee_")[1].split("_subject_metrics")[0]
    rows.append((tag, d["val_Brier"].mean(), len(d)))
if not rows:
    sys.exit("NO_SWEEP_RESULTS")
rows.sort(key=lambda r: r[1])
for tag, brier, n in rows:
    print(f"  {tag:<20} val_Brier {brier:.4f}  (n={n})", file=sys.stderr)
print(rows[0][0])
PY
)
echo "[lee] sweep winner on validation Brier: ${WINNER}"
if [ "${WINNER}" != "default" ]; then
  echo "FATAL: sweep winner is '${WINNER}', not 'default'." >&2
  echo "The production arms below pass no recipe flags, so tagging them 'sel'" >&2
  echo "would name a selection that did not happen. Update FT with the winning" >&2
  echo "flags and re-launch. Sweep artifacts are on disk; nothing is lost." >&2
  exit 1
fi

# --- 1. headline: does pretraining beat random init at n=54? ---------------
step "lee cbramod finetune (pretrained)"
python -u src/experiments/run_bd_fm_probe.py --model cbramod ${FT} --tag sel

step "lee cbramod finetune (random-init control)"
python -u src/experiments/run_bd_fm_probe.py --model cbramod ${FT} --tag sel --random-init

# --- 2. frozen probes: the regime where pretraining looks worthless --------
step "lee cbramod frozen (pretrained)"
python -u src/experiments/run_bd_fm_probe.py --model cbramod --mode frozen \
  --dataset lee2019_mi --subjects ${SUBJ} --device cpu --patches 4 --out-dir ${OUT}

step "lee cbramod frozen (random-init control)"
python -u src/experiments/run_bd_fm_probe.py --model cbramod --mode frozen \
  --dataset lee2019_mi --subjects ${SUBJ} --device cpu --patches 4 --out-dir ${OUT} --random-init

# --- 3. supervised comparators, both bands ---------------------------------
for arch in shallow_convnet atcnet; do
  for band in narrowband broadband; do
    dir=data/fm; [ "${band}" = "narrowband" ] && dir=data/narrowband
    step "lee ${arch} ${band}"
    python -u src/experiments/run_supervised_on_fm_data.py \
      --arch "${arch}" --band "${band}" --fm-dir "${dir}" --dataset lee2019_mi \
      --subjects ${SUBJ} --device mps --no-deterministic --epochs 200 --tune \
      --out-dir results/supervised_sota
  done
done

echo LEEDONE
