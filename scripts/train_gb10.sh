#!/usr/bin/env bash
# Fine-tune Kokoro-82M on Swedish — GB10 (DGX Spark, aarch64) port of train_3090.sh.
# Differences vs the 3090 script:
#   * uses recipe/.venv binaries DIRECTLY (uv sync is broken on aarch64: misaki->spacy
#     has no ARM wheels; the venv was built manually — see RUN2 notes)
#   * per-voice: pass a voice name (female|male) -> config_sv_<voice>.yml, data_<voice>/
#   * option-B configs: joint_epoch 999 so the adversarial/SLM stage (RUN1 comb
#     artifact) never engages; Stage 2 still trains prosody/duration jointly.
#
#   bash train_gb10.sh female --smoke   # 1-epoch 50-clip gate, watch losses
#   bash train_gb10.sh female           # full Stage 1 -> Stage 2 -> voicepack
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VOICE="${1:?usage: train_gb10.sh <female|male> [--smoke]}"
CONFIG="$ROOT/configs/config_sv_$VOICE.yml"
DATA="$ROOT/data_$VOICE"
ST="$ROOT/recipe/StyleTTS2"
PY="$ROOT/recipe/.venv/bin/python"
ACC="$ROOT/recipe/.venv/bin/accelerate"

SMOKE=0
[ "${2:-}" = "--smoke" ] && SMOKE=1

for f in "$ST/train_first.py" "$ST/train_second.py" "$ROOT/models/kokoro_base.pth" \
         "$DATA/train_list.txt" "$DATA/val_list.txt" "$DATA/OOD_texts.txt" "$CONFIG"; do
  [ -f "$f" ] || { echo "ERROR: required file missing: $f" >&2; exit 1; }
done
if grep -q '|' "$DATA/OOD_texts.txt"; then
  echo "ERROR: $DATA/OOD_texts.txt contains filelist separators (must be phoneme strings only)." >&2
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ST"

if [ "$SMOKE" = "1" ]; then
  echo "==> smoke gate ($VOICE): 1 epoch on 50 clips — losses must be FINITE and in range"
  head -n 50 "$DATA/train_list.txt" > "$DATA/_smoke_train.txt"
  head -n 20 "$DATA/val_list.txt"   > "$DATA/_smoke_val.txt"
  SMOKE_CFG="$ROOT/configs/configs/config_sv_$VOICE.smoke.yml"
  sed -e 's#train_list.txt#_smoke_train.txt#' \
      -e 's#val_list.txt#_smoke_val.txt#' \
      -e 's#^epochs_1st:.*#epochs_1st: 1#' \
      -e 's#^epochs:.*#epochs: 1#' \
      -e 's#^batch_size:.*#batch_size: 2#' "$CONFIG" > "$SMOKE_CFG"
  echo "    PASS: Mel 0.8-1.5 | Gen 3-6 | Disc 4-6 | Mono 0.01-0.1 | S2S 1-6 (no NaN)"
  "$ACC" launch train_first.py --config_path "$SMOKE_CFG"
  echo "==> smoke finished ($VOICE)."
  exit 0
fi

echo "==> Stage 1 ($VOICE): acoustic + duration/prosody, warm-started"
"$ACC" launch train_first.py  --config_path "$CONFIG"

echo "==> Stage 2 ($VOICE): joint training (adversarial/SLM DISABLED via joint_epoch)"
echo "    CHECK: Stage-2 mel loss must START ~0.43, NOT ~7.5."
"$ACC" launch train_second.py --config_path "$CONFIG"

echo "==> extract voicepack ($VOICE)"
OUT="$ROOT/output/sv_kokoro_$VOICE"
CKPT="$OUT/epoch_2nd_best.pth"
[ -f "$CKPT" ] || CKPT="$(find "$OUT" -maxdepth 1 -type f \( -name '*2nd*.pth' -o -name '*best*.pth' \) | sort | tail -n 1)"
[ -n "$CKPT" ] && [ -f "$CKPT" ] || { echo "ERROR: no Stage-2 checkpoint under $OUT" >&2; exit 1; }
"$PY" "$ROOT/recipe/scripts/extract_voicepack.py" \
  --model "$CKPT" \
  --style-encoder-model "$OUT/first_stage.pth" \
  --audio-dir "$DATA/wavs_24k" \
  --output "$ROOT/output/sv_${VOICE}.voicepack.pt" --num-samples 200

echo "✅ $VOICE done: checkpoint $CKPT, voicepack output/sv_${VOICE}.voicepack.pt"
