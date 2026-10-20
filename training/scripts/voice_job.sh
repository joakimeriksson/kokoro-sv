#!/usr/bin/env bash
# One-voice fine-tune job, the recipe that produced Klara (Oct 15-16): smoke run
# (40 clips, 1 epoch) -> full stage-2 fine-tune -> eval_voice.py on every epoch.
# Runs ON the Spark from the repo root; each GPU step takes the gpu_run.sh lock.
#
#   nohup training/scripts/voice_job.sh viktor data-gen/dataset_vox_male 5 > finetune_viktor.log 2>&1 &
#   # style run: + style sets dir and the approved checkpoint to continue from
#   nohup training/scripts/voice_job.sh klara_styles data-gen/dataset_vox_female 3 \
#       data-gen/dataset_vox_klara_styles training/outputs/checkpoints/klara/epoch_2nd_00004.pth > ... &
#
# Log markers: <NAME>_SMOKE_EXIT, <NAME>_JOB_EXIT, eval in eval_<name>.log (EVAL_VOICE_DONE).
set -uo pipefail
NAME="$1"; DSET="$2"; EPOCHS="${3:-5}"; STYLES="${4:-}"; BASE="${5:-}"
EXTRA=(); EVAL_EXTRA=()
[ -n "$STYLES" ] && { EXTRA+=(--styles "$STYLES"); EVAL_EXTRA+=(--styles "$STYLES"); }
[ -n "$BASE" ] && { EXTRA+=(--base "$BASE"); EVAL_EXTRA+=(--base "$BASE"); }
TAG=$(echo "$NAME" | tr '[:lower:]' '[:upper:]')
cd "$(dirname "$0")/../.."

# finetune_voice.py needs g2p_sv (neural G2P); take the first venv that has it
PY=""
for cand in .venv/bin/python recipe/.venv/bin/python; do
    if [ -x "$cand" ] && SV_NEURAL_G2P=nst_g2p "$cand" -c "import g2p_sv" 2>/dev/null; then
        PY="$cand"; break
    fi
done
[ -n "$PY" ] || { echo "${TAG}_JOB_EXIT no-python-with-g2p_sv"; exit 1; }
echo "python: $PY  dataset: $DSET  epochs: $EPOCHS  start: $(date)"

./gpu_run.sh "$PY" training/scripts/finetune_voice.py --dataset "$DSET" --name "$NAME" ${EXTRA[@]+"${EXTRA[@]}"} --smoke --launch
rc=$?; echo "${TAG}_SMOKE_EXIT $rc  $(date)"
[ $rc = 0 ] || exit $rc

./gpu_run.sh "$PY" training/scripts/finetune_voice.py --dataset "$DSET" --name "$NAME" \
    ${EXTRA[@]+"${EXTRA[@]}"} --epochs-2nd "$EPOCHS" --launch
rc=$?; echo "${TAG}_JOB_EXIT $rc  $(date)"
[ $rc = 0 ] || exit $rc

./gpu_run.sh recipe/.venv/bin/python training/scripts/eval_voice.py --dataset "$DSET" --name "$NAME" ${EVAL_EXTRA[@]+"${EVAL_EXTRA[@]}"} \
    > "eval_${NAME}.log" 2>&1
echo "${TAG}_EVAL_EXIT $?  $(date)"
