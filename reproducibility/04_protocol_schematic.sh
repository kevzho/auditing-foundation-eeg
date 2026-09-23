#!/usr/bin/env bash
set -euo pipefail

export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.mplconfig}"

python src/scripts/make_protocol_schematic.py \
  --out-dir figures
