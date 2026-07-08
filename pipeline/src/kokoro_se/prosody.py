"""Prosody feature extraction + coarse prosody classing.

Used for analysis and data SELECTION (e.g. the prosody-focused fine-tune mix),
not fed into Kokoro directly. Pitch stats are in semitones around the clip
median so male/female clips are comparable (RUN2 lesson: raw-Hz and fixed
quefrency bands are pitch-biased).
"""
from __future__ import annotations

import numpy as np
import librosa


def extract_prosody(x: np.ndarray, sr: int) -> dict:
    f0 = librosa.yin(x, fmin=60, fmax=450, sr=sr)
    voiced = f0[(f0 > 60) & (f0 < 440)]
    if len(voiced) < 10:
        return {}
    st = 12 * np.log2(voiced / np.median(voiced))

    frame = int(0.03 * sr)
    n = len(x) // frame
    fe = np.sqrt((x[: n * frame].reshape(n, frame) ** 2).mean(axis=1))
    speech_frames = fe > (fe.max() * 0.1)
    pause_ratio = 1.0 - speech_frames.mean()

    # rough syllable-rate proxy: energy-envelope peaks per voiced second
    env = fe / (fe.max() + 1e-9)
    peaks = ((env[1:-1] > env[:-2]) & (env[1:-1] > env[2:]) & (env[1:-1] > 0.25)).sum()
    dur = len(x) / sr
    speech_sec = max(dur * (1 - pause_ratio), 0.1)

    return {
        "f0_mean": round(float(np.median(voiced)), 1),
        "f0_std": round(float(np.std(st)), 2),          # semitones
        "f0_range": round(float(np.percentile(st, 95) - np.percentile(st, 5)), 1),
        "energy_mean": round(float(fe.mean()), 4),
        "energy_std": round(float(fe.std()), 4),
        "speech_rate": round(float(peaks / speech_sec), 2),
        "pause_ratio": round(float(pause_ratio), 3),
        "duration": round(dur, 2),
    }


def prosody_class(p: dict, text: str = "") -> str:
    """Coarse label: neutral | frågande | energisk | långsam | snabb | expressiv."""
    if not p:
        return "neutral"
    if text.rstrip().endswith("?"):
        return "frågande"
    # thresholds ≈ p85/p15 of neutral NST read speech (measured 2026-07-06):
    # f0_std p50=7.6 p85=8.1; speech_rate p50=7.1 p85=7.9; pause p85=0.61
    if p["f0_std"] > 8.6 and p["energy_std"] > p["energy_mean"] * 0.8:
        return "expressiv"
    if p["f0_std"] > 8.1:
        return "energisk"
    if p["speech_rate"] > 8.2:
        return "snabb"
    if p["speech_rate"] < 5.5 or p["pause_ratio"] > 0.65:
        return "långsam"
    return "neutral"
