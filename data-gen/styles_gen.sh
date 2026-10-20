#!/usr/bin/env bash
# Style sets for one voice: 5 styles x N clips via VoxCPM2 controllable cloning,
# each on its own slice of the corpus. Output: dataset_vox_<voice>_styles/<style>/
#   styles_gen.sh <voice: klara|viktor> <ref wav> <python> [N=400] [device]
#   ONLY="sad angry" styles_gen.sh ...   # just those styles (same corpus slices)
set -uo pipefail
cd "$(dirname "$0")"
VOICE="$1"; REF="$2"; PY="$3"; N="${4:-400}"; DEV="${5:-auto}"
STYLES=(
  "whisper|whispering softly, very quiet"
  "excited|very excited and happy, energetic"
  "laugh|laughing while speaking, amused"
  "sad|sad and disappointed, slow"
  "angry|angry and stern, raised voice"
)
k=0
for s in "${STYLES[@]}"; do
  name="${s%%|*}"; instr="${s#*|}"
  if [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $name "* ]]; then k=$((k + 1)); continue; fi
  echo "$(date +%H:%M:%S) [$VOICE] $name"
  "$PY" batch_voxcpm.py --ref "$REF" --speaker "sv_vox_${VOICE}_${name}" --corpus corpus_vox_sv.txt \
      --out "dataset_vox_${VOICE}_styles/$name" --start $((k * N)) --limit "$N" \
      --instruction "$instr" --device "$DEV" 2>&1 | grep --line-buffered -E '^\[|DONE|skip|Error|Traceback'
  k=$((k + 1))
done
echo "$(date +%H:%M:%S) [$VOICE] STYLES_DONE"
