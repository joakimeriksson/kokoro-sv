"""Evaluate single-voice fine-tune checkpoints (finetune_voice.py) against the base.

For the base checkpoint and every epoch_2nd_*.pth in the run: extract the voice's
pack from its own clips, render configs/testset_sv.txt, then score with
eval_renders.py (ASR-CER -> DNSMOS -> comb) and F0 variability (semitone std per
utterance: the "is it still flat" number). The target's own F0 variability is
measured on its training clips for comparison.

    ../gpu_run.sh recipe/.venv/bin/python training/scripts/eval_voice.py \\
        --dataset data-gen/dataset_vox_female --name klara

Run under gpu_run.sh (caller wraps once; subprocesses inherit the lock).
"""
from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SK = ROOT.parent
PY = SK / "recipe" / ".venv" / "bin" / "python"
EXTRACT = SK / "recipe" / "scripts" / "extract_voicepack.py"
BASE = ROOT / "outputs" / "checkpoints" / "kokoro_se_base" / "epoch_2nd_00005.pth"
TESTSET = ROOT / "configs" / "testset_sv.txt"
ENV = {"SV_NEURAL_G2P": "nst_g2p", "PATH": "/usr/bin:/bin"}


def f0_std(wavs) -> float:
    stds = []
    for p in wavs:
        x, sr = sf.read(str(p), dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        f0 = librosa.yin(x, fmin=60, fmax=450, sr=sr)
        v = f0[(f0 > 60) & (f0 < 440)]
        if len(v) > 20:
            stds.append(float(np.std(12 * np.log2(v / np.median(v)))))
    return float(np.mean(stds)) if stds else 0.0


def f0_median(wavs) -> float:
    meds = []
    for p in wavs:
        x, sr = sf.read(str(p), dtype="float32")
        f0 = librosa.yin(x if x.ndim == 1 else x.mean(axis=1), fmin=60, fmax=450, sr=sr)
        v = f0[(f0 > 60) & (f0 < 440)]
        if len(v) > 20:
            meds.append(float(np.median(v)))
    return float(np.median(meds)) if meds else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--refs", type=int, default=30)
    args = ap.parse_args()

    dset = Path(args.dataset).resolve()
    run = ROOT / "outputs" / "checkpoints" / args.name
    out_root = ROOT / "outputs" / "eval" / args.name
    refs = out_root / "_refs"
    refs.mkdir(parents=True, exist_ok=True)
    for old in refs.glob("*.wav"):
        old.unlink()
    rows = [r for r in csv.DictReader(open(dset / "verified.csv", encoding="utf-8"), delimiter="|")
            if r["keep"] == "1"]
    # skip the 200 CandyTron lines at the top: refs should be ordinary sentences
    for i, r in enumerate(rows[200:200 + args.refs]):
        (refs / f"{i:03d}.wav").symlink_to(dset / r["audio"])

    target = [dset / r["audio"] for r in rows[200:400]]
    print(f"target clips: F0-std {f0_std(target):.2f} st, median F0 {f0_median(target):.0f} Hz",
          flush=True)

    ckpts = [("base", BASE)] + [(p.stem.replace("epoch_2nd_", "e"), p)
                               for p in sorted(run.glob("epoch_2nd_*.pth"))]
    for tag, ckpt in ckpts:
        vp = out_root / f"vp_{tag}.pt"
        out = out_root / tag
        # style half from the same checkpoint: its decoder was trained on it
        subprocess.run([str(PY), str(EXTRACT), "--model", str(ckpt), "--style-encoder-model",
                        str(ckpt), "--audio-dir", str(refs), "--output", str(vp),
                        "--num-samples", str(args.refs)], check=True, capture_output=True)
        subprocess.run([str(PY), str(SK / "synth_real.py"), "--checkpoint", str(ckpt),
                        "--voicepack", str(vp), "--texts", str(TESTSET), "--out-dir", str(out)],
                       check=True, cwd=SK, env=ENV, capture_output=True)
        r = subprocess.run([str(PY), str(SK / "eval_renders.py"), "--texts", str(TESTSET), str(out)],
                           cwd=SK, capture_output=True, text=True)
        line = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "FAILED " + r.stderr[-300:]
        wavs = sorted(out.glob("test_*.wav"))
        print(f"{tag:8s} F0-std {f0_std(wavs):.2f} st  F0 {f0_median(wavs):.0f} Hz | {line}",
              flush=True)
    print("EVAL_VOICE_DONE", flush=True)


if __name__ == "__main__":
    main()
