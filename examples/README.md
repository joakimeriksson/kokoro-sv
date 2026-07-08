# examples — use the trained Swedish Kokoro voices

These scripts **download the trained model and voices from HuggingFace** and
synthesize Swedish. No local data or training needed.

```bash
pip install kokoro huggingface_hub torch scipy soundfile
export SV_NEURAL_G2P=nst_g2p          # neural Swedish G2P (auto-downloads its model)

python examples/speak.py --voice Stina --text "Hej, jag är CandyTron!"
python examples/list_voices.py        # render every voice in the pack
python examples/blend_voices.py --a Björn --b Nils --mix 0.7
```

| script | what |
|---|---|
| `speak.py` | one voice, one sentence → wav (the quickstart) |
| `list_voices.py` | render the same line with every voice in the pack |
| `blend_voices.py` | interpolate two voicepacks into an in-between voice |

The voice pack repo is set by `$KOKORO_SV_VOICES` (default `Joakim/kokoro-sv-voices`).
Each script downloads `kokoro_sv.pth` + `config.json` + the chosen `voices/<Name>.pt`,
phonemizes with the neural Swedish G2P, and applies the four upsampler-tone notch
filters at inference (built in). First run downloads ~330 MB (cached after).
