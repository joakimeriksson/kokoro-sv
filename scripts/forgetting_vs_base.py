"""English/French/Spanish/Italian: original Kokoro-82M vs the Swedish fine-tune, same voices."""
import re, warnings, logging, numpy as np, torch
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
from huggingface_hub import hf_hub_download
from kokoro import KModel, KPipeline
from faster_whisper import WhisperModel

T = {
 "a": ("en", "af_heart", ["Hello! Would you like a piece of candy today?",
                          "The robot moves the chocolate from one square to another.",
                          "I did not quite hear you, could you say that again?"]),
 "f": ("fr", "ff_siwis", ["Bonjour ! Voulez-vous un bonbon aujourd'hui ?",
                          "Le robot déplace le chocolat d'une case à l'autre.",
                          "Je ne vous ai pas bien entendu, pouvez-vous répéter ?"]),
 "e": ("es", "ef_dora",  ["¡Hola! ¿Quieres un caramelo hoy?",
                          "El robot mueve el chocolate de una casilla a otra.",
                          "No te he oído bien, ¿puedes repetirlo?"]),
 "i": ("it", "if_sara",  ["Ciao! Vuoi una caramella oggi?",
                          "Il robot sposta il cioccolato da una casella all'altra.",
                          "Non ti ho sentito bene, puoi ripetere?"]),
}
base = KModel(repo_id="hexgrad/Kokoro-82M").eval()
sv = KModel(repo_id="hexgrad/Kokoro-82M", config=hf_hub_download("Joakim/kokoro-sv-voices", "config.json"),
            model=hf_hub_download("Joakim/kokoro-sv-voices", "kokoro_sv.pth")).eval()
asr = WhisperModel("small", device="cpu", compute_type="int8")
def norm(s): return re.sub(r"[^\w ]", "", s.lower()).split()
def cer(ref, hyp):
    r, h = " ".join(norm(ref)), " ".join(norm(hyp))
    d = list(range(len(h) + 1))
    for i, a in enumerate(r, 1):
        p, d[0] = d[0], i
        for j, b in enumerate(h, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j-1] + 1, p + (a != b))
    return d[len(h)] / max(1, len(r))
import soundfile as sf
OUT = "output/forgetting"
import os; os.makedirs(OUT, exist_ok=True)
for code, (lang, voice, sents) in T.items():
    row = {}
    for name, m in (("base", base), ("swedish", sv)):
        pipe = KPipeline(lang_code=code, repo_id="hexgrad/Kokoro-82M", model=m)
        cs, sw = [], []
        for k, s in enumerate(sents):
            audio = np.concatenate([a.numpy() if hasattr(a, "numpy") else np.asarray(a) for _, _, a in pipe(s, voice=voice)])
            p = f"{OUT}/{lang}_{name}_{k}.wav"; sf.write(p, audio, 24000)
            segs, info = asr.transcribe(p, language=lang, beam_size=1)
            cs.append(cer(s, " ".join(x.text for x in segs)))
            segs, info = asr.transcribe(p, beam_size=1)      # language ID: does it still sound like <lang>?
            list(segs); sw.append(info.language_probability if info.language == lang else 0.0)
        row[name] = (np.mean(cs), np.mean(sw))
    b, s = row["base"], row["swedish"]
    print(f"  {lang}: CER base {b[0]:.3f} -> swedish model {s[0]:.3f}   |  heard as {lang}: base {b[1]:.2f} -> swedish model {s[1]:.2f}", flush=True)
