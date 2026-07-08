"""Audio preparation: convert to mono 24 kHz WAV, loudness-normalize, segment."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

TARGET_SR = 24000
TARGET_RMS_DB = -23.0  # rough loudness target (RMS dBFS)


def load_mono(path: str | Path, sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    x, in_sr = sf.read(str(path), dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if in_sr != sr:
        x = librosa.resample(x, orig_sr=in_sr, target_sr=sr)
    return x, sr


def rms_normalize(x: np.ndarray, target_db: float = TARGET_RMS_DB) -> np.ndarray:
    rms = np.sqrt(np.mean(x ** 2)) + 1e-9
    gain = 10 ** ((target_db - 20 * np.log10(rms)) / 20)
    y = x * gain
    peak = np.abs(y).max()
    if peak > 0.99:  # never clip
        y = y * (0.99 / peak)
    return y


def convert_to_wav(src: str | Path, dst: str | Path, sr: int = TARGET_SR,
                   normalize: bool = True, pad_tail_ms: int = 200) -> float:
    """Convert any soundfile-readable audio to mono/24k/16-bit WAV.

    pad_tail_ms: trailing silence (RUN2 lesson — StyleTTS2 issue #81 stop-token
    padding massively reduces end-of-utterance artifacts). Returns duration (s).
    """
    x, _ = load_mono(src, sr)
    if normalize:
        x = rms_normalize(x)
    if pad_tail_ms:
        x = np.concatenate([x, np.zeros(int(sr * pad_tail_ms / 1000), dtype="float32")])
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), x, sr, subtype="PCM_16")
    return len(x) / sr


def segment_on_silence(x: np.ndarray, sr: int, min_sec: float = 2.0,
                       max_sec: float = 12.0, top_db: float = 35.0):
    """Split long audio into ~2-12 s chunks at silence boundaries.

    Yields (start_sample, end_sample). Greedy merge of librosa's non-silent
    intervals up to max_sec; drops chunks under min_sec.
    """
    intervals = librosa.effects.split(x, top_db=top_db)
    cur_s, cur_e = None, None
    for s, e in intervals:
        if cur_s is None:
            cur_s, cur_e = s, e
            continue
        if (e - cur_s) / sr <= max_sec:
            cur_e = e
        else:
            if (cur_e - cur_s) / sr >= min_sec:
                yield cur_s, cur_e
            cur_s, cur_e = s, e
    if cur_s is not None and (cur_e - cur_s) / sr >= min_sec:
        yield cur_s, cur_e
