#!/usr/bin/env bash
# Overnight driver (15-16 Oct 2026): finish the VoxCPM2 voice fits unattended.
# 1. wait for the female fit   2. wait for >=300 male clips   3. fit the male
# 4. collect best packs under provisional names in data-gen/fitted_voices/extra/
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/Users/joakimeriksson/work/kokoro-voice-server/.venv/bin/python
log() { echo "$(date +%H:%M:%S) [overnight] $*"; }

log "waiting for the female fit (pid ${FEMALE_PID:-none})"
while [ -n "${FEMALE_PID:-}" ] && kill -0 "$FEMALE_PID" 2>/dev/null; do sleep 30; done
log "female fit finished: $(grep -E 'BEST|no fit' data-gen/fit_female.log | tail -1)"

log "waiting for >=300 male clips"
until [ "$(ls data-gen/dataset_vox_male/wavs 2>/dev/null | wc -l)" -ge 300 ]; do sleep 60; done
log "male clips available: $(ls data-gen/dataset_vox_male/wavs | wc -l)"

$PY scripts/fit_voicepack.py --target data-gen/dataset_vox_male/wavs \
    --out data-gen/fitted_voices/vox_male --steps 300 > data-gen/fit_male.log 2>&1
log "male fit finished: $(grep -E 'BEST|no fit' data-gen/fit_male.log | tail -1)"

mkdir -p data-gen/fitted_voices/extra
[ -f data-gen/fitted_voices/vox_female/best.pt ] && cp data-gen/fitted_voices/vox_female/best.pt data-gen/fitted_voices/extra/Vera.pt
[ -f data-gen/fitted_voices/vox_male/best.pt ]   && cp data-gen/fitted_voices/vox_male/best.pt   data-gen/fitted_voices/extra/Viktor.pt
log "packs: $(ls data-gen/fitted_voices/extra/)"
log "DONE"
