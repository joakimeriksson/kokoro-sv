#!/usr/bin/env bash
# The Spark generates Viktor's style sets at ~11 clips/min, the Mac at ~30. After
# Klara's sets, the Mac renders Viktor's last two styles and ships them; the
# Spark's (resumable) generator then skips every clip that already exists.
set -uo pipefail
cd "$(dirname "$0")"
SPARK="${SPARK:-joakim@192.168.101.114}"
until grep -q STYLES_DONE styles_klara.log 2>/dev/null; do sleep 60; done
ONLY="sad angry" ./styles_gen.sh viktor refs/vox_male2_ref.wav /Users/joakimeriksson/.venvs/voxcpm/bin/python 400 mps
for s in sad angry; do
  rsync -a --ignore-existing -e "ssh -o BatchMode=yes" "dataset_vox_viktor_styles/$s/wavs/" \
      "$SPARK:work/swedish-kokoro/data-gen/dataset_vox_viktor_styles/$s/wavs/"
done
echo "$(date +%H:%M:%S) shipped Viktor sad+angry to the Spark"
