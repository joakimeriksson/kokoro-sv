#!/usr/bin/env python
"""Just for fun: Swedish words in a French voice. 🇫🇷🇸🇪

A voicepack is only a *style vector* — timbre + prosody — so you can borrow a
French speaker from base Kokoro-82M and have our Swedish model speak Swedish in
that voice. Same trick works for any Kokoro voice (Italian, Japanese, …).

    python examples/swedish_with_french_voice.py
    python examples/swedish_with_french_voice.py --voice if_sara --text "Bonjour, jag heter CandyTron!"

--voice is any voice from hexgrad/Kokoro-82M (ff_siwis = French female,
if_sara = Italian, jf_alpha = Japanese, …). The Swedish G2P still drives the words.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import soundfile as sf
from scipy.signal import iirnotch, filtfilt
from huggingface_hub import hf_hub_download

VOICES_REPO = os.environ.get("KOKORO_SV_VOICES", "Joakim/kokoro-sv-voices")
KOKORO_REPO = "hexgrad/Kokoro-82M"          # base model's multilingual voice bank
os.environ.setdefault("SV_NEURAL_G2P", "nst_g2p")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def notch(a, sr=24000):
    for f0 in (2400, 4800, 7200, 9600):
        b, c = iirnotch(f0, Q=35, fs=sr); a = filtfilt(b, c, a)
    return a.astype("float32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="ff_siwis", help="a voice from hexgrad/Kokoro-82M (French: ff_siwis)")
    ap.add_argument("--text", default="Hej, jag är CandyTron, men idag pratar jag med fransk röst!")
    ap.add_argument("--out", default="swedish_french.wav")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # our Swedish model...
    from kokoro import KModel
    from g2p_sv import SwedishG2P
    model = KModel(repo_id=KOKORO_REPO,
                   config=hf_hub_download(VOICES_REPO, "config.json"),
                   model=hf_hub_download(VOICES_REPO, "kokoro_sv.pth")).to(device).eval()
    # ...driven by a foreign voicepack from the base model's bank
    foreign = torch.load(hf_hub_download(KOKORO_REPO, f"voices/{args.voice}.pt"),
                         map_location=device, weights_only=True)
    g2p = SwedishG2P(backend="neural")

    ipa = g2p(args.text).replace("ʏ", "y")
    ids = [i for i in (model.vocab.get(p) for p in ipa) if i is not None]
    with torch.no_grad():
        audio = model.forward_with_tokens(torch.LongTensor([[0] + ids + [0]]).to(device),
                                          foreign[len(ids) - 1].to(device), speed=1.0)[0].squeeze().cpu().numpy()
    sf.write(args.out, notch(audio), 24000)
    print(f"Swedish text in '{args.voice}' voice -> {args.out}")


if __name__ == "__main__":
    main()
