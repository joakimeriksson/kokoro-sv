#!/usr/bin/env python
"""Render the same Swedish line with every voice in the pack, side by side.

    python examples/list_voices.py                 # all voices -> out_voices/<Name>.wav
    python examples/list_voices.py --text "God morgon!"

Downloads the pack from HuggingFace and writes one wav per voice so you can
audition them. Good first thing to run after `speak.py`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import soundfile as sf
from scipy.signal import iirnotch, filtfilt
from huggingface_hub import hf_hub_download, list_repo_files

VOICES_REPO = os.environ.get("KOKORO_SV_VOICES", "Joakim/kokoro-sv-voices")
os.environ.setdefault("SV_NEURAL_G2P", "nst_g2p")
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def notch(a, sr=24000):
    for f0 in (2400, 4800, 7200, 9600):
        b, c = iirnotch(f0, Q=35, fs=sr); a = filtfilt(b, c, a)
    return a.astype("float32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="Hej, vad heter du och var kommer du ifrån?")
    ap.add_argument("--out-dir", default="out_voices")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(exist_ok=True)

    voices = sorted(f.split("/")[1][:-3] for f in list_repo_files(VOICES_REPO)
                    if f.startswith("voices/") and f.endswith(".pt"))
    print("voices in pack:", ", ".join(voices))

    from kokoro import KModel
    from g2p_sv import SwedishG2P
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = KModel(repo_id="hexgrad/Kokoro-82M",
                   config=hf_hub_download(VOICES_REPO, "config.json"),
                   model=hf_hub_download(VOICES_REPO, "kokoro_sv.pth")).to(device).eval()
    g2p = SwedishG2P(backend="neural")
    ipa = g2p(args.text).replace("ʏ", "y")
    ids = [i for i in (model.vocab.get(p) for p in ipa) if i is not None]
    toks = torch.LongTensor([[0] + ids + [0]]).to(device)

    for name in voices:
        vp = torch.load(hf_hub_download(VOICES_REPO, f"voices/{name}.pt"),
                        map_location=device, weights_only=True)
        with torch.no_grad():
            a = model.forward_with_tokens(toks, vp[len(ids) - 1].to(device), speed=1.0)[0].squeeze().cpu().numpy()
        sf.write(out / f"{name}.wav", notch(a), 24000)
        print(f"  {name} -> {out/name}.wav")


if __name__ == "__main__":
    main()
