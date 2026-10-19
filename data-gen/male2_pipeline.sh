#!/usr/bin/env bash
# New male voice (Oct 19: take 8, refs/vox_male2_ref.wav, ~128 Hz): generate the
# corpus with VoxCPM2 -> verify clip by clip -> hand over to at_home_viktor.sh,
# which trains on the Spark once it is reachable.
#   nohup caffeinate -i data-gen/male2_pipeline.sh > data-gen/male2_pipeline.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
log() { echo "$(date +%H:%M:%S) [male2] $*"; }
log "generating"
(cd data-gen && ~/.venvs/voxcpm/bin/python batch_voxcpm.py --ref refs/vox_male2_ref.wav \
    --speaker sv_vox_male2 --corpus corpus_vox_sv.txt --out dataset_vox_male2 --device mps) 2>&1 \
    | grep --line-buffered -E '^\[|DONE|skip|Error|Traceback' || { log "generation failed"; exit 1; }
log "verifying"
/Users/joakimeriksson/work/kokoro-voice-server/.venv/bin/python scripts/verify_vox_dataset.py \
    data-gen/dataset_vox_male2 --ref data-gen/refs/vox_male2_ref.wav 2>&1 \
    | grep -vE "Warning|warn\(|_nested|super\(\)"
[ -f data-gen/dataset_vox_male2/verified.csv ] || { log "no verified.csv"; exit 1; }
log "handing over to at_home_viktor.sh"
DATASET=dataset_vox_male2 exec data-gen/at_home_viktor.sh
