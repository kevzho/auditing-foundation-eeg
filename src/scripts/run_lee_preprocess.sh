#!/usr/bin/env bash
# Lee2019_MI dual-band preprocessing, one subject per process (docs 10.3 item 2).
#
# Written after the first attempt died at subject 7 of 54. The external volume
# remounted (disk8s1 -> disk5s1), every relative path stopped resolving, and a
# single process holding all 54 subjects took the whole run down with it -- the
# traceback was `FileNotFoundError: 'data/narrowband'`, seven subjects in.
#
# Three properties follow from that:
#   1. One subject per process, so a fault costs one subject, not the run.
#   2. Resume by inspecting what is on disk, not by remembering where we were.
#   3. Re-resolve the root before every subject and abort loudly if it is gone,
#      rather than emitting 47 identical tracebacks into a log nobody reads.
#
# --purge-raw stays: the raw set is ~59 GB against ~27 GB free, so sources are
# deleted per subject as soon as both bands are written. A subject is therefore
# only "done" when *both* npz pairs exist -- subject 7 had its broadband pair
# and no narrowband, and re-running it must redo both.
set -uo pipefail

ROOT=/Volumes/kz_extended/CalibMI
RETRIES=2

cd "${ROOT}" || { echo "FATAL: ${ROOT} unreachable" >&2; exit 1; }

done_subject () {  # subject -> 0 if both bands present
  local s="$1"
  [ -f "${ROOT}/data/fm/lee2019_mi_subject${s}_train.npz" ] \
  && [ -f "${ROOT}/data/fm/lee2019_mi_subject${s}_eval.npz" ] \
  && [ -f "${ROOT}/data/narrowband/lee2019_mi_subject${s}_train.npz" ] \
  && [ -f "${ROOT}/data/narrowband/lee2019_mi_subject${s}_eval.npz" ]
}

for s in $(seq 1 54); do
  cd "${ROOT}" || { echo "FATAL: ${ROOT} went away before subject ${s}" >&2; exit 1; }

  if done_subject "${s}"; then
    echo "[lee] subject ${s} already complete, skipping"
    continue
  fi

  ok=1
  for attempt in $(seq 1 $((RETRIES + 1))); do
    echo "########## lee subject ${s} (attempt ${attempt}) ##########"
    python -u "${ROOT}/src/preprocess_fm_moabb.py" \
      --dataset Lee2019_MI --subjects "${s}" \
      --bands broadband narrowband --purge-raw 2>&1 \
      | grep --line-buffered -viE "warning|^ *[0-9]+%\||hash of downloaded|known_hash"
    if done_subject "${s}"; then ok=0; break; fi
    echo "[lee] subject ${s} incomplete after attempt ${attempt}" >&2
    cd "${ROOT}" || { echo "FATAL: ${ROOT} went away during subject ${s}" >&2; exit 1; }
  done
  [ "${ok}" -eq 0 ] || echo "FAILED lee subject ${s} after $((RETRIES + 1)) attempts" >&2

  avail=$(df -g "${ROOT}" | tail -1 | awk '{print $4}')
  echo "[lee] subject ${s} done; ${avail} GiB free"
  # Purging happens per subject, so free space should stay roughly flat. If it
  # does not, stop before filling the volume and corrupting a write.
  if [ "${avail}" -lt 5 ]; then
    echo "FATAL: only ${avail} GiB free, stopping before a truncated write" >&2
    exit 1
  fi
done
echo LEEPREPDONE
