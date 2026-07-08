#!/usr/bin/env python
"""Download the trained Swedish Kokoro voices from HuggingFace and speak.

    python examples/speak.py                                  # default voice + line
    python examples/speak.py --voice Stina --text "Hej där!"
    python examples/speak.py --voice Björn --out bjorn.wav

Everything (model weights, the chosen voicepack, the neural Swedish G2P, its
lexicon) downloads from HuggingFace on first run and is cached. No local data
needed — just `pip install kokoro huggingface_hub torch scipy soundfile`.
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

# --- where the trained artifacts live on HuggingFace --------------------------
VOICES_REPO = os.environ.get("KOKORO_SV_VOICES", "Joakim/kokoro-sv-voices")  # the voice pack
os.environ.setdefault("SV_NEURAL_G2P", "nst_g2p")                            # neural G2P module
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))                                          # import g2p_sv from the repo

NOTCH_HZ = (2400, 4800, 7200, 9600)   # remove fine-tune upsampler tones (lossless)


def notch(audio, sr=24000):
    for f0 in NOTCH_HZ:
        b, a = iirnotch(f0, Q=35, fs=sr)
        audio = filtfilt(b, a, audio)
    return audio.astype("float32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="Stina", help="voice name (see list_voices.py)")
    ap.add_argument("--text", default="Hej, jag är CandyTron, din godisrobot!")
    ap.add_argument("--out", default="out.wav")
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args()

    # 1. download model weights + config + the chosen voicepack from HF
    model_path = hf_hub_download(VOICES_REPO, "kokoro_sv.pth")
    config_path = hf_hub_download(VOICES_REPO, "config.json")
    voice_path = hf_hub_download(VOICES_REPO, f"voices/{args.voice}.pt")

    # 2. build the model + neural Swedish G2P (G2P model auto-downloads from HF)
    from kokoro import KModel
    from g2p_sv import SwedishG2P
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = KModel(repo_id="hexgrad/Kokoro-82M", config=config_path, model=model_path).to(device).eval()
    voice = torch.load(voice_path, map_location=device, weights_only=True)
    g2p = SwedishG2P(backend="neural")

    # 3. phonemize -> tokens -> synthesize (voicepack indexed by token count)
    ipa = g2p(args.text).replace("ʏ", "y")
    ids = [i for i in (model.vocab.get(p) for p in ipa) if i is not None]
    with torch.no_grad():
        audio = model.forward_with_tokens(
            torch.LongTensor([[0] + ids + [0]]).to(device),
            voice[len(ids) - 1].to(device), speed=args.speed)[0].squeeze().cpu().numpy()

    sf.write(args.out, notch(audio), 24000)
    print(f"[{args.voice}] {args.text!r}\n-> {args.out}  ({len(audio)/24000:.1f}s)")


if __name__ == "__main__":
    main()
