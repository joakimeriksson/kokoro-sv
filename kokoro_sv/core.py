"""SwedishKokoro — one-line Swedish TTS on top of Kokoro-82M.

    from kokoro_sv import SwedishKokoro
    tts = SwedishKokoro()
    tts.speak("Hej, jag är CandyTron!", voice="Stina", out="hej.wav")
    print(tts.voices)                       # ['Alice', ..., 'Björn', ...]
    blend = tts.blend("Björn", "Nils", 0.7)
    tts.speak("...", voice=blend, out="mix.wav")

Downloads the model, voices, and neural G2P from HuggingFace on first use (cached).
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import numpy as np

VOICES_REPO = os.environ.get("KOKORO_SV_VOICES", "Joakim/kokoro-sv-voices")
KOKORO_BASE = "hexgrad/Kokoro-82M"
_NOTCH_HZ = (2400, 4800, 7200, 9600)      # fine-tune upsampler tones (removed at inference)


def _stub_misaki():
    """Kokoro's pipeline imports misaki (EN/DE G2P) at import time; on some platforms
    (e.g. aarch64 Linux) it has no wheels. We drive the model directly and never use
    it, so stub it if absent. No-op where misaki is installed (Mac, x86)."""
    try:
        import misaki  # noqa: F401
        return
    except Exception:
        pass

    class _Stub(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            return type(name, (), {})

    for m in ("misaki", "misaki.en", "misaki.espeak"):
        sys.modules.setdefault(m, _Stub(m))
    sys.modules["misaki"].en = sys.modules["misaki.en"]
    sys.modules["misaki"].espeak = sys.modules["misaki.espeak"]


def _notch(audio, sr=24000):
    from scipy.signal import iirnotch, filtfilt
    for f0 in _NOTCH_HZ:
        b, a = iirnotch(f0, Q=35, fs=sr)
        audio = filtfilt(b, a, audio)
    return audio.astype("float32")


class SwedishKokoro:
    def __init__(self, voices_repo: str = VOICES_REPO, device: str | None = None):
        import torch
        from huggingface_hub import hf_hub_download
        self._torch = torch
        self._hf = hf_hub_download
        self.repo = voices_repo
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        _stub_misaki()
        from kokoro import KModel
        model_path = hf_hub_download(voices_repo, "kokoro_sv.pth")
        config_path = hf_hub_download(voices_repo, "config.json")
        self.model = KModel(repo_id=KOKORO_BASE, config=config_path, model=model_path).to(self.device).eval()

        # neural Swedish G2P (vendored chain: sys.path + env, model downloads from HF)
        vend = str(Path(__file__).resolve().parent / "_vendor")
        if vend not in sys.path:
            sys.path.insert(0, vend)
        os.environ.setdefault("SV_NEURAL_G2P", "nst_g2p")
        os.environ.setdefault("SV_G2P_DIR", str(Path(vend) / "g2p"))
        from g2p_sv import SwedishG2P
        self.g2p = SwedishG2P(backend="neural")
        self._voice_cache: dict[str, object] = {}

    # -- voices --------------------------------------------------------------
    @property
    def voices(self) -> list[str]:
        from huggingface_hub import list_repo_files
        return sorted(f.split("/")[1][:-3] for f in list_repo_files(self.repo)
                      if f.startswith("voices/") and f.endswith(".pt"))

    def _load_voice(self, name: str):
        if name not in self._voice_cache:
            p = self._hf(self.repo, f"voices/{name}.pt")
            self._voice_cache[name] = self._torch.load(p, map_location=self.device, weights_only=True)
        return self._voice_cache[name]

    def blend(self, a: str, b: str, mix: float = 0.5):
        """Interpolate two voices into a new voicepack tensor. mix = weight of `a`."""
        va, vb = self._load_voice(a), self._load_voice(b)
        return mix * va + (1 - mix) * vb

    def kokoro_voice(self, name: str, repo: str = KOKORO_BASE):
        """Borrow any voice from base Kokoro-82M — Swedish words, foreign voice, for fun.

            tts.speak("Hej!", voice=tts.kokoro_voice("ff_siwis"))   # French 🇫🇷
            # if_sara = Italian, jf_alpha = Japanese, ...
        """
        p = self._hf(repo, f"voices/{name}.pt")
        return self._torch.load(p, map_location=self.device, weights_only=True)

    # -- synthesis -----------------------------------------------------------
    def synthesize(self, text: str, voice="Stina", speed: float = 1.0) -> np.ndarray:
        """Return 24 kHz float32 audio. `voice` is a name or a voicepack tensor."""
        torch = self._torch
        vp = self._load_voice(voice) if isinstance(voice, str) else voice
        ipa = self.g2p(text).replace("ʏ", "y")
        ids = [i for i in (self.model.vocab.get(p) for p in ipa) if i is not None]
        with torch.no_grad():
            audio = self.model.forward_with_tokens(
                torch.LongTensor([[0] + ids + [0]]).to(self.device),
                vp[len(ids) - 1].to(self.device), speed=speed)[0].squeeze().cpu().numpy()
        return _notch(audio)

    def speak(self, text: str, voice="Stina", out: str = "out.wav", speed: float = 1.0) -> str:
        import soundfile as sf
        sf.write(out, self.synthesize(text, voice, speed), 24000)
        return out
