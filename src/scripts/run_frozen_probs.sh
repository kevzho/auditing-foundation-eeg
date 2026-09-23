#!/usr/bin/env bash
# Re-run the frozen probes to capture per-trial probabilities (docs 10.3).
#
# Why this exists: `write_probability_npz` was added to both runners after the
# frozen arms were run, so `results/fm_probe` has subject metrics for all eight
# frozen configurations and not one probability file. Accuracy and Brier survive
# the aggregation; AUROC and anything else trial-level does not. No code change
# is needed -- run_bd_fm_probe.py:470 and run_fm_probe.py:646 already write the
# npz for every subject in both modes. These arms just predate it.
#
# What it is for: the manuscript title claims pretraining is "not a
# representation", resting on frozen-probe accuracy sitting near chance. But
# near-chance accuracy and above-chance discrimination can coexist -- the
# fine-tuned random-init control already shows exactly that on BCI IV-2a
# (accuracy 0.335 against chance 0.25, AUROC 0.599 against chance 0.5). If the
# frozen probes rank trials above chance, the claim needs narrowing to "not
# linearly decodable at the decision threshold". That is a title-level question
# and it cannot be answered from the CSVs on disk.
#
# Frozen mode is a feature extraction followed by an sklearn logistic probe over
# C_GRID, not gradient training, so all eight arms are cheap on CPU. Every flag
# below is transcribed from the original runs' JSON so the only difference
# between these artifacts and the reported ones is the added npz -- device cpu,
# deterministic, patches {2,4}, seed 42.
#
# Output goes to a SEPARATE directory on purpose. `resolve_sources` in
# make_fm_audit_report.py resolves later directories over earlier ones, so the
# report can be regenerated with
#
#     --fm-dir results/fm_probe results/fm_probe_frozen_probs
#
# to prefer these. Writing in place would put the headline CSVs at the mercy of
# a partial rerun, and there is no reason to take that risk for a rerun whose
# entire purpose is to add a file alongside them.
#
# This doubles as a determinism check that has never actually been run. These
# arms recorded deterministic=true on cpu; re-running them must reproduce
# REPORT.md's accuracies exactly. If it does not, that is a finding about the
# reproducibility claim and it is better to learn it here than from a judge.
# `compare_frozen_reruns.py` does that comparison once this finishes.
set -uo pipefail

ROOT=/Volumes/kz_extended/CalibMI
cd "${ROOT}" || { echo "FATAL: ${ROOT} unreachable" >&2; exit 1; }

OUT=results/fm_probe_frozen_probs
mkdir -p "${OUT}"

# CBraMod goes through the braindecode runner, LaBraM through the original one.
# The two write different stems for the same experiment -- `fm_probe_cbramod_frozen_*`
# versus `fm_probe_frozen_*` -- which is why the LaBraM arms are not simply
# another value of a --model flag here.
step () { echo; echo "########## $* ##########"; }

for dataset in bci4_2a bnci2014_004; do
  for init in "" "--random-init"; do
    label="pretrained"; [ -n "${init}" ] && label="randinit"

    step "cbramod frozen ${dataset} ${label}"
    python -u src/experiments/run_bd_fm_probe.py \
      --model cbramod --mode frozen --dataset "${dataset}" \
      --patches 2 4 --device cpu --seed 42 --resume \
      --out-dir "${OUT}" ${init}

    step "labram frozen ${dataset} ${label}"
    python -u src/experiments/run_fm_probe.py \
      --mode frozen --dataset "${dataset}" \
      --patches 2 4 --device cpu --seed 42 --resume \
      --out-dir "${OUT}" ${init}
  done
done

echo FROZENPROBSDONE
