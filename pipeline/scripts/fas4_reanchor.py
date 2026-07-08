"""Fas 4: re-anchor CandyTron 018/98 to REAL NST audio on the responsive base.

Pulls each speaker's real NST clips (streaming KTH/nst), extracts a voicepack
through the base model (no Chatterbox in the chain), renders the testset,
batteries it, and measures F0 dynamics vs the old Chatterbox-distilled voices.

Run under gpu_run.sh (single locked job).
"""
from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import fsspec
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SK = ROOT.parent
PY = SK / "recipe" / ".venv" / "bin" / "python"
EXTRACT = SK / "recipe" / "scripts" / "extract_voicepack.py"
CKPT = ROOT / "outputs" / "checkpoints" / "kokoro_se_base" / "epoch_2nd_00005.pth"
FIRST_STAGE = ROOT / "outputs" / "checkpoints" / "kokoro_se_base" / "first_stage.pth"
TESTSET = ROOT / "configs" / "testset_sv.txt"
OUT = ROOT / "outputs" / "fas4"
ENV = {"SV_NEURAL_G2P": "nst_g2p", "PATH": "/usr/bin:/bin"}

SPEAKERS = {"018": "female", "98": "male"}  # NST speaker_id -> label


def pull_real_nst(speaker_id: str, dst: Path, max_clips: int = 60):
    dst.mkdir(parents=True, exist_ok=True)
    for old in dst.glob("*.wav"):
        old.unlink()
    fs = fsspec.filesystem("http", block_size=1 << 22)
    n = 0
    for i in range(10):
        if n >= max_clips:
            break
        url = f"https://huggingface.co/api/datasets/KTH/nst/parquet/default/train/{i}.parquet"
        with fs.open(url) as f:
            pf = pq.ParquetFile(f)
            for rg in range(pf.num_row_groups):
                if n >= max_clips:
                    break
                spk_col = pf.read_row_group(rg, columns=["speaker_id"]).column(0).to_pylist()
                if speaker_id not in spk_col:
                    continue
                t = pf.read_row_group(rg, columns=["speaker_id", "audio"])
                for spk, audio in zip(t.column(0).to_pylist(), t.column(1).to_pylist()):
                    if spk != speaker_id or n >= max_clips:
                        continue
                    x, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
                    if x.ndim > 1:
                        x = x[:, 0]
                    if len(x) / sr < 2.0:
                        continue
                    sf.write(str(dst / f"{n:03d}.wav"), x, sr, subtype="PCM_16")
                    n += 1
    return n


def f0_var(d: Path) -> float:
    import librosa
    stds = []
    for p in sorted(d.glob("test_*.wav")):
        x, sr = sf.read(str(p), dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        f0 = librosa.yin(x, fmin=60, fmax=450, sr=sr)
        v = f0[(f0 > 60) & (f0 < 440)]
        if len(v) > 20:
            stds.append(float(np.std(12 * np.log2(v / np.median(v)))))
    return float(np.mean(stds)) if stds else 0.0


for spk, label in SPEAKERS.items():
    print(f"\n=== {label} (real NST {spk}) on multi-speaker base ===", flush=True)
    refs = OUT / f"refs_{label}"
    got = pull_real_nst(spk, refs)
    print(f"pulled {got} real NST clips", flush=True)
    vp = OUT / f"vp_{label}_real.pt"
    subprocess.run([str(PY), str(EXTRACT), "--model", str(CKPT),
                    "--style-encoder-model", str(FIRST_STAGE),
                    "--audio-dir", str(refs), "--output", str(vp),
                    "--num-samples", str(min(got, 60))], check=True, capture_output=True)
    rend = OUT / label
    subprocess.run([str(PY), str(SK / "synth_real.py"), "--checkpoint", str(CKPT),
                    "--voicepack", str(vp), "--texts", str(TESTSET), "--out-dir", str(rend)],
                   check=True, cwd=SK, env=ENV, capture_output=True)
    bat = subprocess.run([str(PY), str(SK / "eval_renders.py"), "--texts", str(TESTSET), str(rend)],
                         cwd=SK, capture_output=True, text=True)
    line = bat.stdout.strip().splitlines()[-1] if bat.stdout.strip() else "FAILED"
    print(f"battery: {line}", flush=True)
    print(f"F0-std: {f0_var(rend):.2f} st  (old Chatterbox {label}: "
          f"{'5.9' if label == 'female' else '3.3'} st)", flush=True)

print("\nFAS4_DONE", flush=True)
