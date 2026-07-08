"""Generate mood-definition clips with Chatterbox for both CandyTron voices.

For each voice x mood: synthesize the mood's definition lines (mood_lines.py)
with the mood's exaggeration setting, cloning the voice's NST ref clip.
Output: output/moods/<voice>_<mood>/wavs/NNN.wav
"""
import os
import time

import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from mood_lines import MOODS

VOICES = {
    "female018": "refs/candidates/018_ref.wav",
    "male98": "refs/candidates/98_ref.wav",
}

model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
t0 = time.time()
n = 0
for vname, ref in VOICES.items():
    for mood, spec in MOODS.items():
        outdir = f"output/moods/{vname}_{mood}/wavs"
        os.makedirs(outdir, exist_ok=True)
        for i, text in enumerate(spec["lines"]):
            path = f"{outdir}/{i:03d}.wav"
            if os.path.exists(path):
                continue
            wav = model.generate(text, language_id="sv", audio_prompt_path=ref,
                                 exaggeration=spec["exaggeration"])
            ta.save(path, wav, model.sr)
            n += 1
        print(f"{vname}/{mood}: done ({time.time()-t0:.0f}s elapsed)", flush=True)
print(f"MOOD_GEN_DONE: {n} new clips in {time.time()-t0:.0f}s")
