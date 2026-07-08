#!/usr/bin/env bash
# ALL compute (GPU AND heavy CPU) goes through this wrapper — unified memory: flock guarantees a single
# GPU consumer — two concurrent CUDA processes have hard-frozen the GB10 twice
# (NVRM NV_ERR_NO_MEMORY -> driver wedge -> power cycle). Usage:
#   gpu_run.sh <command...>          # waits (up to 12h) for the lock, then runs
LOCK=/home/joakim/work/ai-smarthome/.gpu-lock/gpu.lock
exec flock -w 43200 "$LOCK" "$@"
