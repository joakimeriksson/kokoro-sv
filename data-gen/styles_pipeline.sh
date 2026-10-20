#!/usr/bin/env bash
# Style sets -> gate -> Spark style fine-tunes, for both voices (Oct 2026).
# Klara's sets are generated on the Mac, Viktor's on the Spark (styles_gen.sh);
# verification runs here (Whisper small + ECAPA live in the voice-server venv).
# Each voice continues from its approved checkpoint; gpu_run.sh on the Spark
# serializes the jobs.
#   nohup caffeinate -i data-gen/styles_pipeline.sh > data-gen/styles_pipeline.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
SPARK="${SPARK:-joakim@192.168.101.114}"
R=work/swedish-kokoro
PY=/Users/joakimeriksson/work/kokoro-voice-server/.venv/bin/python
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10)
RSYNC=(rsync -a -e "ssh -o BatchMode=yes")
log() { echo "$(date +%H:%M:%S) [styles] $*"; }

verify() {  # <styles dir> <ref>
    for d in "$1"/*/; do
        d="${d%/}"
        # a whisper is far from the normal-voice reference for ECAPA; gate it looser
        case "$(basename "$d")" in whisper) sim=0.55 ;; *) sim=0.65 ;; esac
        log "verify $d (ref-sim >= $sim)"
        $PY scripts/verify_vox_dataset.py "$d" --ref "$2" --min-ref-sim $sim --max-cer 0.30 2>&1 \
            | grep -E "kept|median|rejected"
    done
}

launch() {  # <name> <plain dataset> <styles dir> <base ckpt>
    # -f: a detached remote job has kept ssh sessions open before; don't wait on it
    "${SSH[@]}" -f -n "$SPARK" "cd $R && setsid nohup training/scripts/voice_job.sh $1 $2 3 $3 $4 \
        > finetune_$1.log 2>&1 < /dev/null &"
    log "launched $1 on the Spark"
}

"${RSYNC[@]}" training/scripts/finetune_voice.py training/scripts/eval_voice.py training/scripts/voice_job.sh \
    "$SPARK:$R/training/scripts/" || { log "cannot reach the Spark"; exit 1; }

# --- Klara: generated here ----------------------------------------------------
until grep -q STYLES_DONE data-gen/styles_klara.log 2>/dev/null; do sleep 60; done
verify data-gen/dataset_vox_klara_styles data-gen/refs/vox_female_ref.wav
"${RSYNC[@]}" data-gen/dataset_vox_klara_styles/ "$SPARK:$R/data-gen/dataset_vox_klara_styles/"
launch klara_styles data-gen/dataset_vox_female data-gen/dataset_vox_klara_styles \
    training/outputs/checkpoints/klara/epoch_2nd_00004.pth

# --- Viktor: generated on the Spark ------------------------------------------
until "${SSH[@]}" "$SPARK" "grep -q STYLES_DONE $R/styles_viktor.log" < /dev/null; do sleep 120; done
"${RSYNC[@]}" "$SPARK:$R/data-gen/dataset_vox_viktor_styles/" data-gen/dataset_vox_viktor_styles/
verify data-gen/dataset_vox_viktor_styles data-gen/refs/vox_male2_ref.wav
"${RSYNC[@]}" --include '*/' --include 'verified.csv' --exclude '*' \
    data-gen/dataset_vox_viktor_styles/ "$SPARK:$R/data-gen/dataset_vox_viktor_styles/"
launch viktor_styles data-gen/dataset_vox_male2 data-gen/dataset_vox_viktor_styles \
    training/outputs/checkpoints/viktor/epoch_2nd_00001.pth
log "DONE (training continues on the Spark: finetune_{klara,viktor}_styles.log)"
