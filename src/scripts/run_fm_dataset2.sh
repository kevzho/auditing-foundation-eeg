#!/usr/bin/env bash
# Second dataset: BNCI2014_004 (2-class, 9 subjects, 5 sessions, 3 electrodes).
#
# Why this one. BNCI2014_001 is MOABB's build of BCI IV-2a, so it would not be
# an independent replication. BNCI2014_004 is: different subjects, two-class
# rather than four, and a genuine multi-session split (3 train / 2 test) that
# exercises session shift rather than a single train/test pair.
#
# The honest caveat, stated here so it reaches the paper: it records only C3,
# Cz and C4. Foundation models pretrained on dense clinical montages are being
# asked to work from three electrodes, which is a harder setting -- not a
# neutral replication. It answers "is the deficit specific to 4-class?", not
# "does the deficit hold at equal channel count".
#
# Frozen only. Fine-tuning adds ~25 min per configuration and the frozen probe
# is what isolates the pretrained representation, which is the question here.
set -euo pipefail

OUT=results/fm_probe
COMMON="--dataset bnci2014_004 --mode frozen --subjects 1 2 3 4 5 6 7 8 9 --device cpu --patches 2 4 --out-dir ${OUT}"

step () { echo; echo "########## $* ##########"; }

step "d2 cbramod frozen (pretrained)"
python src/experiments/run_bd_fm_probe.py --model cbramod ${COMMON}

step "d2 cbramod frozen (random init control)"
python src/experiments/run_bd_fm_probe.py --model cbramod ${COMMON} --random-init

step "d2 labram frozen (pretrained)"
python src/experiments/run_fm_probe.py ${COMMON}

step "d2 labram frozen (random init control)"
python src/experiments/run_fm_probe.py ${COMMON} --random-init

echo
echo "dataset 2 complete"
