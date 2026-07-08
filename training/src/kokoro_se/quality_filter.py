"""Quality gates for training clips — the RUN2 battery, packaged.

Gate order matters (learned the hard way — see swedish-kokoro RUN2):
  1. duration / clipping / speech-ratio  (cheap, no models)
  2. ASR character-error-rate vs transcript (KBLab VoxRex) — intelligibility
  3. DNSMOS P.835 overall — perceptual quality

Never rank or filter on a single narrow signal metric.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

MIN_SEC, MAX_SEC = 1.0, 15.0
CER_MAX = 0.15
DNSMOS_MIN = 2.6            # synthetic/consumer audio centers ~3.0
DNSMOS_ONNX = Path(__file__).resolve().parents[2] / "data" / "sig_bak_ovr.onnx"
_POLY_OVR = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
_ASR_SR = 16000


def cheap_gates(x: np.ndarray, sr: int) -> str | None:
    """Return a rejection reason or None."""
    dur = len(x) / sr
    if dur < MIN_SEC:
        return "too_short"
    if dur > MAX_SEC:
        return "too_long"
    peak = np.abs(x).max()
    if peak > 0.999:
        return "clipping"
    if peak < 0.02:
        return "silent"
    frame = int(0.03 * sr)
    n = len(x) // frame
    fe = np.sqrt((x[: n * frame].reshape(n, frame) ** 2).mean(axis=1))
    if (fe > fe.max() * 0.1).mean() < 0.3:
        return "low_speech_ratio"
    return None


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


class QualityScorer:
    """Lazy-loads VoxRex ASR + DNSMOS. One instance per process; GPU-lock the
    calling process on shared-memory boxes (GB10: ONE compute job at a time)."""

    def __init__(self, device: str = "cuda"):
        import torch
        import onnxruntime as ort
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        self.torch = torch
        self.device = device if torch.cuda.is_available() else "cpu"
        self.processor = Wav2Vec2Processor.from_pretrained("KBLab/wav2vec2-large-voxrex-swedish")
        self.asr = Wav2Vec2ForCTC.from_pretrained("KBLab/wav2vec2-large-voxrex-swedish").to(self.device).eval()
        self.dns = ort.InferenceSession(str(DNSMOS_ONNX), providers=["CPUExecutionProvider"])

    def score(self, x: np.ndarray, sr: int, text: str) -> dict:
        import librosa
        x16 = librosa.resample(x, orig_sr=sr, target_sr=_ASR_SR) if sr != _ASR_SR else x
        inputs = self.processor(x16, sampling_rate=_ASR_SR, return_tensors="pt")
        with self.torch.no_grad():
            logits = self.asr(inputs.input_values.to(self.device)).logits
        hyp = self.processor.batch_decode(self.torch.argmax(logits, dim=-1))[0]
        c = cer(norm_text(text), norm_text(hyp))

        win = int(9.01 * _ASR_SR)
        seg = np.tile(x16, int(np.ceil(win / max(len(x16), 1))))[:win]
        raw = self.dns.run(None, {"input_1": seg[None].astype(np.float32)})[0][0]
        mos = float(_POLY_OVR(raw[2]))

        passed = c <= CER_MAX and mos >= DNSMOS_MIN
        return {"cer": round(c, 3), "dnsmos": round(mos, 2),
                "quality_score": round(min(mos / 5.0, 1.0) * (1 - min(c, 1.0)), 3),
                "pass": passed, "asr_text": hyp}
