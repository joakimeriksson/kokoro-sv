#!/usr/bin/env bash
# Laptop side: wait until the Spark answers on the home LAN, then ship Viktor's
# verified VoxCPM2 set + the job script and start the fine-tune there. Also pulls
# Klara's eval renders, which never made it to the laptop.
#
#   nohup data-gen/at_home_viktor.sh > data-gen/at_home_viktor.log 2>&1 &
#   (stop: pkill -f at_home_viktor.sh)
set -uo pipefail
SPARK="${SPARK:-joakim@192.168.101.114}"
REMOTE=work/swedish-kokoro
cd "$(dirname "$0")/.."
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=8)

deadline=$(( $(date +%s) + 4 * 24 * 3600 ))
until "${SSH[@]}" "$SPARK" true 2>/dev/null; do
    [ "$(date +%s)" -lt "$deadline" ] || { echo "gave up waiting for the Spark: $(date)"; exit 1; }
    sleep 120
done
echo "Spark reachable: $(date)"

if "${SSH[@]}" "$SPARK" "pgrep -f 'voice_job.sh viktor' >/dev/null"; then
    echo "a viktor job is already running on the Spark; not starting another"; exit 0
fi

rsync -a -e "ssh -o BatchMode=yes" data-gen/dataset_vox_male/ "$SPARK:$REMOTE/data-gen/dataset_vox_male/" || exit 1
rsync -a -e "ssh -o BatchMode=yes" training/scripts/voice_job.sh "$SPARK:$REMOTE/training/scripts/" || exit 1
"${SSH[@]}" "$SPARK" "cd $REMOTE && nohup training/scripts/voice_job.sh viktor data-gen/dataset_vox_male 5 \
    > finetune_viktor.log 2>&1 < /dev/null & echo launched pid \$!" || exit 1

mkdir -p data-gen/fitted_voices/klara_eval
rsync -a -e "ssh -o BatchMode=yes" --exclude '_refs' "$SPARK:$REMOTE/training/outputs/eval/klara/" data-gen/fitted_voices/klara_eval/
echo "klara eval renders -> data-gen/fitted_voices/klara_eval  ($(date))"
