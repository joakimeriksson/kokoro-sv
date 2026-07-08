"""Audition previews: clone candidate refs through Chatterbox on CandyTron lines.

    uv run python preview_synth.py --voice sv_female_018=refs/candidates/018_ref.wav \
                                   --voice sv_male_98=refs/candidates/98_ref.wav \
                                   --voice baseline=refs/female_ref.wav

Writes output/preview/<voice>__<idx>.wav for each line in LINES.
"""
from __future__ import annotations

import argparse
import os
import time

import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

LINES = [
    "Hej, jag är CandyTron, din godisrobot!",
    "Vill du ha en klubba eller en bit choklad?",
    "Tryck på knappen så får du en överraskningsgodis.",
    "Här kommer en sur klubba, akta tänderna!",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", action="append", required=True,
                    metavar="NAME=REF_WAV", help="repeatable: voice name + ref clip")
    ap.add_argument("--out-dir", default="output/preview")
    args = ap.parse_args()
    voices = dict(v.split("=", 1) for v in args.voice)

    os.makedirs(args.out_dir, exist_ok=True)
    print("loading Chatterbox Multilingual on cuda ...", flush=True)
    model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")

    for name, ref in voices.items():
        for i, text in enumerate(LINES):
            t0 = time.time()
            wav = model.generate(text, language_id="sv", audio_prompt_path=ref)
            path = os.path.join(args.out_dir, f"{name}__{i}.wav")
            ta.save(path, wav, model.sr)
            print(f"{path}  {wav.shape[-1]/model.sr:.1f}s  ({time.time()-t0:.0f}s gen)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
