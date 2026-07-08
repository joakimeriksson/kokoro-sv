#!/usr/bin/env python
"""Blend two voicepacks into an in-between voice — voicepacks interpolate freely.

    python examples/blend_voices.py --a Stina --b Signe --mix 0.5
    python examples/blend_voices.py --a Björn --b Nils --mix 0.7   # 70% Björn

`mix` is the weight of voice A. This is pure tensor math (no GPU needed for the
blend itself) — a cheap way to get new voices and intensity between two speakers.
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
os.environ.setdefault("SV_NEURAL_G2P", "nst_g2p")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def notch(a, sr=24000):
    for f0 in (2400, 4800, 7200, 9600):
        b, c = iirnotch(f0, Q=35, fs=sr); a = filtfilt(b, c, a)
    return a.astype("float32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="Stina"); ap.add_argument("--b", default="Signe")
    ap.add_argument("--mix", type=float, default=0.5, help="weight of voice A (0..1)")
    ap.add_argument("--text", default="Så här låter en blandning av två röster.")
    ap.add_argument("--out", default="blend.wav")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    va = torch.load(hf_hub_download(VOICES_REPO, f"voices/{args.a}.pt"), map_location=device, weights_only=True)
    vb = torch.load(hf_hub_download(VOICES_REPO, f"voices/{args.b}.pt"), map_location=device, weights_only=True)
    blend = args.mix * va + (1 - args.mix) * vb          # the interpolation

    from kokoro import KModel
    from g2p_sv import SwedishG2P
    model = KModel(repo_id="hexgrad/Kokoro-82M",
                   config=hf_hub_download(VOICES_REPO, "config.json"),
                   model=hf_hub_download(VOICES_REPO, "kokoro_sv.pth")).to(device).eval()
    g2p = SwedishG2P(backend="neural")
    ipa = g2p(args.text).replace("ʏ", "y")
    ids = [i for i in (model.vocab.get(p) for p in ipa) if i is not None]
    with torch.no_grad():
        a = model.forward_with_tokens(torch.LongTensor([[0] + ids + [0]]).to(device),
                                      blend[len(ids) - 1].to(device), speed=1.0)[0].squeeze().cpu().numpy()
    sf.write(args.out, notch(a), 24000)
    print(f"{args.mix:.0%} {args.a} + {1-args.mix:.0%} {args.b} -> {args.out}")


if __name__ == "__main__":
    main()
