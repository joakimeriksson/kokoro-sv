"""Generate candidate reference takes with VoxCPM2 voice design, for picking a new
voice by ear. Design mode invents a new speaker per call, so each take IS a voice;
the one picked becomes refs/vox_*_ref.wav for batch_voxcpm.py.

Every take says batch_voxcpm.PROMPT_TEXT (the prompt_text used for cloning) and is
printed with its median F0 so takes outside --f0-min/--f0-max can be skipped.

    ~/.venvs/voxcpm/bin/python design_takes.py --out refs/male_candidates --n 3 \\
        --f0-min 125 --f0-max 170 \\
        --desc "A young Swedish man with a bright, medium-high voice, friendly and lively"
"""
from __future__ import annotations

import argparse
import os

import librosa
import numpy as np
import soundfile as sf

from batch_voxcpm import OUT_SR, PROMPT_TEXT, resample


def median_f0(x: np.ndarray, sr: int) -> float:
    f0 = librosa.yin(x, fmin=60, fmax=450, sr=sr)
    v = f0[(f0 > 60) & (f0 < 440)]
    return float(np.median(v)) if len(v) else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--desc", action="append", required=True, help="voice description (repeatable)")
    ap.add_argument("--n", type=int, default=3, help="takes per description")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--f0-min", type=float, default=0)
    ap.add_argument("--f0-max", type=float, default=1000)
    args = ap.parse_args()

    from voxcpm import VoxCPM
    model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False, optimize=False)
    sr = model.tts_model.sample_rate
    os.makedirs(args.out, exist_ok=True)

    for d, desc in enumerate(args.desc):
        for k in range(args.n):
            wav = model.generate(text=f"({desc}){PROMPT_TEXT}", cfg_value=args.cfg,
                                 inference_timesteps=args.steps)
            wav = resample(np.asarray(wav, dtype=np.float32).reshape(-1), sr, OUT_SR)
            f0 = median_f0(wav, OUT_SR)
            keep = args.f0_min <= f0 <= args.f0_max
            name = f"d{d}_t{k}_{f0:.0f}hz.wav"
            if keep:
                sf.write(os.path.join(args.out, name), wav, OUT_SR, subtype="PCM_16")
            print(f"{'KEEP' if keep else 'skip'} {name}  [{desc}]", flush=True)


if __name__ == "__main__":
    main()
