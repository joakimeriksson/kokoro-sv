# kokoro-sv — Swedish voices for Kokoro-82M

Ten named Swedish voices with dynamic prosody — one line to speak. The model and
the neural Swedish G2P download from HuggingFace on first use (cached). No training,
no local data, no CUDA required.

## Install

```bash
pip install kokoro-sv
```

## Use

```python
from kokoro_sv import SwedishKokoro

tts = SwedishKokoro()                       # downloads model + G2P from HF (first run)
tts.speak("Hej, jag är CandyTron!", voice="Stina", out="hej.wav")
print(tts.voices)                           # the 10 voice names

# blend two voices into an in-between one
tts.speak("God morgon!", voice=tts.blend("Björn", "Nils", 0.7), out="mix.wav")

# borrow any Kokoro voice for fun — Swedish words in a French accent 🇫🇷
tts.speak("Bonjour!", voice=tts.kokoro_voice("ff_siwis"), out="fr.wav")

# raw audio (24 kHz float32 numpy) instead of a file
audio = tts.synthesize("Ett, två, tre.", voice="Ebba")
```

## CLI

```bash
kokoro-sv voices
kokoro-sv speak "Hej, hur mår du?" --voice Stina --out hej.wav
kokoro-sv blend "God morgon"       --a Björn --b Nils --mix 0.7 --out mix.wav
```

## Voices

**Female:** Alice · Elsa · Ebba · Stina · Greta  
**Male:** Björn · Lars · Nils · Anton · Oskar

Voices interpolate (`blend`) for intermediate identities and intensity. The male
voices are a touch soft on this v1 base (a gender-balanced v2 is planned).

## Notes

- **No GPU needed** — falls back to CPU; Kokoro-82M is tiny, so it's fast.
- First run downloads ~330 MB (model + voices), cached afterward.
- **Python 3.10+.**
- On macOS/x86 `pip install kokoro-sv` pulls `kokoro`/`misaki` cleanly. (On aarch64
  Linux, `misaki`/`spacy` have no wheels — install with `--no-deps` and add deps manually.)

## More

Training, the reproducible multi-speaker pipeline, evaluation, and the full story:
**https://github.com/joakimeriksson/kokoro-sv**

Voices trained from CC0 data (NST + Swedish LibriVox). Built on
[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (Apache-2.0), StyleTTS2, and
[kikiri-tts](https://github.com/semidark/kikiri-tts). Apache-2.0.
