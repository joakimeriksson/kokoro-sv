"""One locked job: battery-sweep base checkpoints + prosody-responsiveness test.

Run under gpu_run.sh (caller wraps once; subprocesses inherit the lock).
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

ROOT = Path(__file__).resolve().parents[1]
SK = ROOT.parent
PY = SK / "recipe" / ".venv" / "bin" / "python"
EXTRACT = SK / "recipe" / "scripts" / "extract_voicepack.py"
CKPT_DIR = ROOT / "outputs" / "checkpoints" / "kokoro_se_base"
FIRST_STAGE = CKPT_DIR / "first_stage.pth"
MANIFEST = ROOT / "data" / "manifests" / "train_mix.jsonl"
TESTSET = ROOT / "configs" / "testset_sv.txt"
RESP = ROOT / "outputs" / "response"
ENV = {"SV_NEURAL_G2P": "nst_g2p", "PATH": "/usr/bin:/bin"}


def spk_wavs(speaker: str, n: int = 30):
    out = []
    for line in MANIFEST.open(encoding="utf-8"):
        e = json.loads(line)
        if e["speaker_id"] == speaker:
            out.append(e["audio_path"])
        if len(out) >= n:
            break
    return out


def refs_dir(speaker: str, tag: str) -> Path:
    d = RESP / f"refs_{tag}"
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*.wav"):
        old.unlink()
    for i, w in enumerate(spk_wavs(speaker)):
        (d / f"{i:03d}.wav").symlink_to(w)
    return d


def extract(ckpt: Path, refs: Path, out_vp: Path):
    subprocess.run([str(PY), str(EXTRACT), "--model", str(ckpt),
                    "--style-encoder-model", str(FIRST_STAGE),
                    "--audio-dir", str(refs), "--output", str(out_vp),
                    "--num-samples", "30"], check=True, capture_output=True)


def render(ckpt: Path, vp: Path, out: Path):
    subprocess.run([str(PY), str(SK / "synth_real.py"), "--checkpoint", str(ckpt),
                    "--voicepack", str(vp), "--texts", str(TESTSET), "--out-dir", str(out)],
                   check=True, cwd=SK, env=ENV, capture_output=True)


def battery(out: Path) -> str:
    r = subprocess.run([str(PY), str(SK / "eval_renders.py"), "--texts", str(TESTSET), str(out)],
                       cwd=SK, capture_output=True, text=True)
    return r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "FAILED"


def f0_var(out: Path) -> float:
    stds = []
    for p in sorted(out.glob("test_*.wav")):
        x, sr = sf.read(str(p), dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        f0 = librosa.yin(x, fmin=60, fmax=450, sr=sr)
        v = f0[(f0 > 60) & (f0 < 440)]
        if len(v) > 20:
            stds.append(float(np.std(12 * np.log2(v / np.median(v)))))
    return float(np.mean(stds)) if stds else 0.0


LOW_SPK = sys.argv[1] if len(sys.argv) > 1 else "libri_Unknown - Ja och Nej"
HIGH_SPK = sys.argv[2] if len(sys.argv) > 2 else "nst_718"

print("=== BATTERY SWEEP (voicepack from high-variance ref for consistency) ===", flush=True)
hi_refs = refs_dir(HIGH_SPK, "sweep")
best, best_ckpt = None, None
for ep in ("00003", "00005", "00006", "00007"):
    ckpt = CKPT_DIR / f"epoch_2nd_{ep}.pth"
    if not ckpt.exists():
        continue
    vp = RESP / f"sweep_{ep}.pt"
    extract(ckpt, hi_refs, vp)
    out = RESP / f"sweep_{ep}"
    render(ckpt, vp, out)
    line = battery(out)
    print(f"epoch {ep}: {line}", flush=True)
    # parse CER (col 2) to pick best intelligible
    try:
        cer = float(line.split()[1])
        if best is None or cer < best:
            best, best_ckpt = cer, ckpt
    except (ValueError, IndexError):
        pass

best_ckpt = best_ckpt or CKPT_DIR / "epoch_2nd_00007.pth"
print(f"\n=== PROSODY RESPONSIVENESS on {best_ckpt.name} ===", flush=True)
results = {}
for tag, spk in (("low", LOW_SPK), ("high", HIGH_SPK)):
    refs = refs_dir(spk, tag)
    vp = RESP / f"vp_{tag}.pt"
    extract(best_ckpt, refs, vp)
    out = RESP / tag
    render(best_ckpt, vp, out)
    results[tag] = f0_var(out)
    print(f"{tag}-variance speaker '{spk}' -> output F0-std {results[tag]:.2f} st", flush=True)

delta = results["high"] - results["low"]
verdict = "RESPONSIVE — style steers prosody (dynamic voices possible)" if delta > 0.8 \
    else "FLAT — predictor near mean (needs prosody fine-tune / more data)"
print(f"\nSPREAD: {delta:+.2f} st  =>  {verdict}", flush=True)
print("EVAL_BASE_DONE", flush=True)
