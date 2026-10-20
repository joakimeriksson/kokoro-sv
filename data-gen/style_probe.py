"""Probe whether VoxCPM2 can render Klara/Viktor in styles Kokoro has never seen
(whisper, laughing, excited, sad, angry), as a possible source of style-tagged
training data. Two modes per style:
  ctrl -- reference_wav_path only: timbre from the take, style from "(instruction)"
  ult  -- ultimate cloning (+ prompt wav/text): continues the take's prosody too

    ~/.venvs/voxcpm/bin/python style_probe.py --out refs/style_probe
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import soundfile as sf

from batch_voxcpm import OUT_SR, PROMPT_TEXT, resample

VOICES = {"Klara": "refs/vox_female_ref.wav", "Viktor": "refs/vox_male2_ref.wav"}
STYLES = [
    ("neutral", "", "Vi har tretton sorters godis idag. Vilken sort vill du ha?"),
    ("whisper", "whispering softly, very quiet",
     "Psst, kom hit. Jag har gömt lite extra godis bakom disken."),
    ("excited", "very excited and happy, energetic",
     "Wow! Vi har fått in nya godissorter idag, det är helt fantastiskt!"),
    ("laugh", "laughing while speaking, amused",
     "Haha, det där var ju det roligaste jag har hört på länge!"),
    ("sad", "sad and disappointed, slow",
     "Åh nej, nu tog godiset slut. Det är verkligen synd."),
    ("angry", "angry and stern, raised voice",
     "Nej! Du får inte ta mer godis nu, det räcker!"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=10)
    args = ap.parse_args()

    from voxcpm import VoxCPM
    model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False, optimize=False)
    sr = model.tts_model.sample_rate
    os.makedirs(args.out, exist_ok=True)
    for name, ref in VOICES.items():
        for style, instr, text in STYLES:
            prompt = f"({instr}){text}" if instr else text
            for mode in ("ctrl", "ult"):
                kw = dict(reference_wav_path=ref)
                if mode == "ult":
                    kw.update(prompt_wav_path=ref, prompt_text=PROMPT_TEXT)
                wav = model.generate(text=prompt, cfg_value=2.0, inference_timesteps=args.steps, **kw)
                wav = resample(np.asarray(wav, dtype=np.float32).reshape(-1), sr, OUT_SR)
                path = os.path.join(args.out, f"{name}_{style}_{mode}.wav")
                sf.write(path, wav, OUT_SR, subtype="PCM_16")
                print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
