# Mac quickstart — Swedish Kokoro voices

Run the published voices on a Mac (Apple Silicon or Intel). Everything downloads
from HuggingFace on first run — no training, no local data.

```bash
git clone https://github.com/joakimeriksson/kokoro-sv.git
cd kokoro-sv
python3 -m venv .venv && source .venv/bin/activate
pip install kokoro huggingface_hub torch scipy soundfile
export SV_NEURAL_G2P=nst_g2p

python examples/speak.py --voice Stina --text "Hej från min Mac!"
afplay out.wav
```

Then try:

```bash
python examples/list_voices.py                       # render all 10 voices
python examples/swedish_with_french_voice.py         # Swedish words, French voice 🇫🇷
python examples/blend_voices.py --a Björn --b Nils --mix 0.7
```

## Where things come from

| what | source |
|---|---|
| Swedish model + 10 voices + config | HF `Joakim/kokoro-sv-voices` |
| neural Swedish G2P (model + lexicon) | HF `Joakim/swedish-kokoro`, auto-downloaded |
| model architecture | `hexgrad/Kokoro-82M` (class only; our weights override) |

## Notes

- **No CUDA needed** — falls back to CPU; Kokoro-82M is tiny, so it's fast.
- **`misaki` installs fine on Mac** (unlike aarch64 Linux) — plain `pip install kokoro` works.
- If the neural G2P ever hiccups, `brew install espeak-ng` gives it a fallback.
- Voices: **Alice, Elsa, Ebba, Stina, Greta** (female) · **Björn, Lars, Nils, Anton, Oskar** (male).
  Males are a touch soft on this v1 base (training data was female-heavy); a
  gender-balanced v2 is the planned improvement.

Coming soon: `pip install kokoro-sv` for a one-line API (`SwedishKokoro().speak(...)`)
instead of the example scripts.
