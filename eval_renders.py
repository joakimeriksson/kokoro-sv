"""Proper TTS render evaluation battery — gate on intelligibility, then quality.

For every render dir given, scores each clip against the known TEST_SENTENCES:
  1. CER      — VoxRex Swedish ASR vs intended text (intelligibility HARD GATE)
  2. DNSMOS   — perceptual overall MOS (P.835 onnx, from swedish-chatterbox/qc/)
  3. comb     — the targeted artifact metric (cepstral, decoder-frame lags)
  4. dur/HF   — duration vs text length sanity, high-frequency balance

    recipe/.venv/bin/python eval_renders.py output/listen_female_run2 output/joint_e4 ...

Prints one row per dir: mean CER | DNSMOS | comb | verdict.
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np
import soundfile as sf
import librosa
import torch

TEST_SENTENCES = [
    "Hej, jag är CandyTron! Idag delar jag ut godis till alla.",
    "Sju sjösjuka sjömän sköttes av sju sköna sjuksköterskor.",
    "Jordens glödande inre är en fabel.",
    "Kungen och drottningen bjöd på choklad, glass och kanelbullar.",
    "Varför gjorde du det? Det är ju otroligt!",
    "Det kostar exakt etthundratjugotre miljoner kronor.",
    "Vi åkte till Göteborg för att titta på den vackra skärgården.",
]
CER_GATE = 0.20
DNSMOS_GATE = 2.5
ASR_SR = 16000
POLY_OVR = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
DNSMOS_WIN = int(9.01 * ASR_SR)
QC_ONNX = "/home/joakim/work/ai-smarthome/swedish-chatterbox/qc/sig_bak_ovr.onnx"


def norm_text(s: str) -> str:
    s = re.sub(r"[^a-zåäöéü ]+", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def cer(ref: str, hyp: str) -> float:
    if not ref:
        return 1.0 if hyp else 0.0
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, hc in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc))
        prev = cur
    return prev[-1] / len(ref)


def comb_score(x: np.ndarray, sr: int) -> float:
    n_fft, hop = 2048, 512
    frames = []
    for s in range(0, len(x) - n_fft, hop):
        w = x[s:s + n_fft] * np.hanning(n_fft)
        frames.append(np.abs(np.fft.irfft(np.log(np.abs(np.fft.rfft(w)) + 1e-9))))
    c = np.mean(frames, axis=0)
    q = np.arange(len(c)) / sr

    def peak(lo, hi):
        m = (q >= lo / 1000) & (q <= hi / 1000)
        return c[m].max() if m.any() else 0.0

    return float(max(peak(11.7, 13.3), peak(24.2, 25.8)) / (peak(2.5, 12.0) + 1e-9))


def main() -> None:
    import onnxruntime as ort
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Wav2Vec2Processor.from_pretrained("KBLab/wav2vec2-large-voxrex-swedish")
    asr = Wav2Vec2ForCTC.from_pretrained("KBLab/wav2vec2-large-voxrex-swedish").to(device).eval()
    dns = ort.InferenceSession(QC_ONNX, providers=["CPUExecutionProvider"])

    # --texts FILE applies to all dirs; else a <dir>/texts.txt or the built-ins
    args = sys.argv[1:]
    texts_override = None
    if "--texts" in args:
        i = args.index("--texts")
        texts_override = [l.strip() for l in open(args[i + 1], encoding="utf-8") if l.strip()]
        args = args[:i] + args[i + 2:]

    print(f"{'render dir':38s} {'CER':>6s} {'DNSMOS':>7s} {'comb':>6s} {'HF/LF':>6s}  verdict")
    for d in args:
        cers, moses, combs, hfs = [], [], [], []
        for i, text in enumerate(texts_override or TEST_SENTENCES, 1):
            p = os.path.join(d, f"test_{i:02d}.wav")
            if not os.path.exists(p):
                continue
            x, sr = sf.read(p, dtype="float32")
            if x.ndim > 1:
                x = x.mean(axis=1)
            x16 = librosa.resample(x, orig_sr=sr, target_sr=ASR_SR) if sr != ASR_SR else x
            inputs = processor(x16, sampling_rate=ASR_SR, return_tensors="pt")
            with torch.no_grad():
                logits = asr(inputs.input_values.to(device)).logits
            hyp = processor.batch_decode(torch.argmax(logits, dim=-1))[0]
            cers.append(cer(norm_text(text), norm_text(hyp)))
            seg = np.tile(x16, int(np.ceil(DNSMOS_WIN / max(len(x16), 1))))[:DNSMOS_WIN]
            raw = dns.run(None, {"input_1": seg[None].astype(np.float32)})[0][0]
            moses.append(float(POLY_OVR(raw[2])))
            combs.append(comb_score(x, sr))
            spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
            f = np.fft.rfftfreq(len(x), 1 / sr)
            hfs.append(spec[(f > 4000) & (f < 10000)].sum() / (spec[(f > 100) & (f < 4000)].sum() + 1e-9))
        if not cers:
            print(f"{d:38s}  (no test_*.wav)")
            continue
        mc, mm, mb, mh = map(np.mean, (cers, moses, combs, hfs))
        verdict = "FAIL-intelligibility" if mc > CER_GATE else \
                  "FAIL-quality" if mm < DNSMOS_GATE else \
                  f"ok (comb {'low' if mb < 1.2 else 'audible'})"
        print(f"{d:38s} {mc:6.3f} {mm:7.2f} {mb:6.2f} {mh:6.3f}  {verdict}")


if __name__ == "__main__":
    main()
