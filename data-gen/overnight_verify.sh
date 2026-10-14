#!/usr/bin/env bash
# After generation (GEN_PID) and the fit driver are done, gate both datasets clip by clip.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/Users/joakimeriksson/work/kokoro-voice-server/.venv/bin/python
log() { echo "$(date +%H:%M:%S) [verify] $*"; }
while [ -n "${GEN_PID:-}" ] && kill -0 "$GEN_PID" 2>/dev/null; do sleep 60; done
log "generation finished: $(grep -E '^\[' data-gen/voxgen.log | tail -1)"
until grep -q DONE data-gen/overnight.log; do sleep 60; done
for v in female male; do
  log "verifying $v"
  $PY scripts/verify_vox_dataset.py data-gen/dataset_vox_$v --ref data-gen/refs/vox_${v}_ref.wav 2>&1 \
    | grep -vE "Warning|warn\(|_nested|super\(\)"
done
log "DONE"
