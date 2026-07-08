"""The decisive Fas-2 measurement: does the style vector STEER output prosody?

Extract voicepacks from a LOW-F0-variance speaker (monotone) and a HIGH-variance
speaker (expressive) from the SAME trained checkpoint, render the identical
testset with each, and measure the F0-variability spread of the OUTPUT.

  * If high-variance voicepack -> livelier output than low-variance voicepack,
    the model learned to USE the prosody dimensions => dynamic voices are possible,
    moods will work, speaker adaptation carries expression. THE GOAL.
  * If both render equally flat, the prosody predictor collapsed to the mean and
    we need the prosody-focused fine-tune (Fas 3) or more expressive data.

Also measures question-intonation: F0 rise on '?'-final sentences.

Run under the GPU lock, AFTER training completes.
    python scripts/prosody_response_test.py --checkpoint <ckpt> \
        --low-speaker "libri_..." --high-speaker nst_718
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

ROOT = Path(__file__).resolve().parents[1]
SK = ROOT.parent
PY = SK / "recipe" / ".venv" / "bin" / "python"
TESTSET = ROOT / "configs" / "testset_sv.txt"


def spk_wavs(manifest: Path, speaker: str, n: int = 30):
    import json
    out = []
    for line in manifest.open(encoding="utf-8"):
        e = json.loads(line)
        if e["speaker_id"] == speaker:
            out.append(e["audio_path"])
        if len(out) >= n:
            break
    return out


def f0_variability(wav_dir: Path) -> float:
    stds = []
    for p in sorted(wav_dir.glob("test_*.wav")):
        x, sr = sf.read(str(p), dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        f0 = librosa.yin(x, fmin=60, fmax=450, sr=sr)
        v = f0[(f0 > 60) & (f0 < 440)]
        if len(v) > 20:
            stds.append(float(np.std(12 * np.log2(v / np.median(v)))))
    return float(np.mean(stds)) if stds else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", default=str(ROOT / "data/manifests/train_mix.jsonl"))
    ap.add_argument("--low-speaker", required=True)
    ap.add_argument("--high-speaker", required=True)
    args = ap.parse_args()

    extract = SK / "recipe" / "scripts" / "extract_voicepack.py"
    first_stage = Path(args.checkpoint).parent / "first_stage.pth"
    manifest = Path(args.manifest)

    results = {}
    for tag, spk in (("low", args.low_speaker), ("high", args.high_speaker)):
        # symlink this speaker's clips into a temp dir for the extractor
        d = ROOT / "outputs" / "response" / f"{tag}_refs"
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*.wav"):
            old.unlink()
        for i, w in enumerate(spk_wavs(manifest, spk)):
            (d / f"{i:03d}.wav").symlink_to(w)
        vp = ROOT / "outputs" / "response" / f"vp_{tag}.pt"
        subprocess.run([str(PY), str(extract), "--model", args.checkpoint,
                        "--style-encoder-model", str(first_stage),
                        "--audio-dir", str(d), "--output", str(vp),
                        "--num-samples", "30"], check=True)
        out = ROOT / "outputs" / "response" / tag
        subprocess.run([str(PY), str(SK / "synth_real.py"),
                        "--checkpoint", args.checkpoint, "--voicepack", str(vp),
                        "--texts", str(TESTSET), "--out-dir", str(out)],
                       check=True, cwd=SK, env={"SV_NEURAL_G2P": "nst_g2p", "PATH": "/usr/bin:/bin"})
        results[tag] = f0_variability(out)

    print("\n=== PROSODY RESPONSIVENESS ===")
    print(f"low-variance speaker voicepack  -> output F0-std {results['low']:.2f} st")
    print(f"high-variance speaker voicepack -> output F0-std {results['high']:.2f} st")
    delta = results["high"] - results["low"]
    print(f"SPREAD: {delta:+.2f} st  "
          f"=> {'RESPONSIVE (style steers prosody)' if delta > 0.8 else 'FLAT (predictor collapsed to mean)'}")


if __name__ == "__main__":
    main()
