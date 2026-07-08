"""Build the voice gallery page from outputs/gallery/ renders."""
import base64
import json
import html as H
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import lameenc

ROOT = Path(__file__).resolve().parents[1]
GAL = ROOT / "outputs" / "gallery"
SENTS = (GAL / "_sentences.txt").read_text(encoding="utf-8").strip().splitlines()
manifest = json.loads((GAL / "_manifest.json").read_text())

# per-speaker F0-std + friendly label from the training manifest
f0 = defaultdict(list)
for line in (ROOT / "data" / "manifests" / "train_mix.jsonl").open(encoding="utf-8"):
    e = json.loads(line)
    if e.get("prosody"):
        f0[e["speaker_id"]].append(e["prosody"]["f0_std"])

NAMES = json.loads((ROOT / "configs" / "voice_names.json").read_text(encoding="utf-8"))


def source_of(spk):
    if spk.startswith("nst_"):
        return f"NST {spk[4:]}"
    if spk.startswith("libri_Unknown - "):
        return "LibriVox: " + spk.replace("libri_Unknown - ", "").replace(" - ", " — ")
    return spk


def friendly(spk):
    return NAMES.get(spk, {}).get("name", source_of(spk))


def role_of(spk):
    return NAMES.get(spk, {}).get("role", "")


items = []
for safe, meta in manifest.items():
    spk = meta["speaker"]
    d = GAL / safe
    wavs = sorted(d.glob("test_*.wav"))
    if not wavs:
        continue
    mf = statistics.mean(f0[spk]) if f0.get(spk) else 0.0
    items.append((mf, spk, safe, wavs))
items.sort()  # monotone -> expressive

b64 = {}
for _, spk, safe, wavs in items:
    for i, w in enumerate(wavs, 1):
        x, sr = sf.read(str(w), dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        x = x / (np.abs(x).max() + 1e-9) * 0.9
        enc = lameenc.Encoder()
        enc.set_bit_rate(80); enc.set_in_sample_rate(sr); enc.set_channels(1); enc.set_quality(2)
        b64[f"{safe}_{i}"] = base64.b64encode(bytes(enc.encode((x * 32767).astype("<i2").tobytes())) + bytes(enc.flush())).decode()


def btn(k, label):
    return (f'<button class="play" data-key="{k}" aria-label="{H.escape(label)}">'
            '<svg class="ic-play" viewBox="0 0 24 24" width="17" height="17"><path d="M8 5v14l11-7z" fill="currentColor"/></svg>'
            '<svg class="ic-pause" viewBox="0 0 24 24" width="17" height="17"><path d="M6 5h4v14H6zM14 5h4v14h-4z" fill="currentColor"/></svg></button>')


cards = ""
for mf, spk, safe, wavs in items:
    g = NAMES.get(spk, {}).get("gender", "")
    role = role_of(spk)
    tag_html = f'<span class="tag">{H.escape(role)}</span>' if role else ""
    plays = "".join(f'<div class="p" title="{H.escape(SENTS[i-1])}">{btn(f"{safe}_{i}", SENTS[i-1])}<span>{i}</span></div>'
                    for i in range(1, len(wavs) + 1))
    dot = "F" if g == "F" else ("M" if g == "M" else "")
    cards += (f'<div class="card" data-g="{dot}"><div class="meta">'
              f'<strong>{H.escape(friendly(spk))}</strong>{tag_html}'
              f'<span class="dyn">{H.escape(source_of(spk))} · dynamik {mf:.1f} st</span>'
              f'</div><div class="plays">{plays}</div></div>')

page = f'''<title>Röstgalleri — alla röster i flertalar-basen</title>
<style>
:root {{ --bg:#F6F7FB; --card:#FFF; --ink:#1B2130; --muted:#6B7480; --line:#E4E8F0; --accent:#6D28D9; --accent-soft:#EDE6FB; }}
body {{ background:var(--bg); color:var(--ink); margin:0; padding:2.5rem 1.25rem 4rem; font:16px/1.5 system-ui,sans-serif; }}
main {{ max-width:52rem; margin:0 auto; }}
h1 {{ font-size:1.6rem; margin:0 0 .4rem; }} h1 em {{ color:var(--accent); font-style:normal; }}
header p {{ color:var(--muted); margin:.25rem 0 0; max-width:40rem; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(15rem,1fr)); gap:.85rem; margin-top:1.75rem; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:.85rem 1rem; border-left:3px solid var(--line); }}
.card[data-g="F"] {{ border-left-color:#C2185B; }}
.card[data-g="M"] {{ border-left-color:#2563EB; }}
.meta {{ display:flex; flex-direction:column; gap:.15rem; margin-bottom:.6rem; }}
.meta strong {{ font-size:.98rem; }}
.tag {{ color:var(--accent); font-size:.72rem; font-weight:600; }}
.dyn {{ color:var(--muted); font-size:.74rem; font-variant-numeric:tabular-nums; }}
.plays {{ display:flex; gap:.5rem; }}
.p {{ display:flex; align-items:center; gap:.25rem; }} .p span {{ color:var(--muted); font-size:.72rem; }}
.play {{ width:2.2rem; height:2.2rem; border-radius:50%; border:1px solid var(--line); background:var(--accent-soft); color:var(--accent); display:grid; place-items:center; cursor:pointer; }}
.play:hover, .play.playing {{ background:var(--accent); color:#fff; }}
.ic-pause {{ display:none; }} .playing .ic-pause {{ display:block; }} .playing .ic-play {{ display:none; }}
.note {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:.9rem 1.1rem; color:var(--muted); font-size:.88rem; margin-top:1.5rem; }}
</style>
<main>
<header><h1>Röstgalleri — <em>{len(items)} svenska röster</em> ur samma modell</h1>
<p>Alla dessa är samma flertalar-Kokoro med olika voicepacks (512 KB var). Sorterade från
mest monotona (vänster/topp) till mest uttrycksfulla. Knapp 1 = signaturmeningen, knapp 2 = en fråga.
Fler röster kan skapas fritt ur vilket ljud som helst — plus oändliga blandningar.</p></header>
<div class="grid">{cards}</div>
<p class="note">Detta är bara talarna som fanns i träningsdatan. Vilken ren ~30 s-inspelning som helst
(t.ex. din egen röst) kan bli en ny voicepack, och två voicepacks kan blandas för mellanlägen och intensitet.</p>
</main>
<script>
const B64 = {json.dumps(b64)};
const audio = new Audio(); let cur = null;
document.querySelectorAll(".play").forEach(b => b.addEventListener("click", () => {{
  if (cur === b && !audio.paused) {{ audio.pause(); return; }}
  if (cur && cur !== b) cur.classList.remove("playing");
  if (cur !== b) {{ audio.src = "data:audio/mpeg;base64," + B64[b.dataset.key]; cur = b; }}
  audio.play();
}}));
audio.addEventListener("play", () => cur && cur.classList.add("playing"));
audio.addEventListener("pause", () => cur && cur.classList.remove("playing"));
audio.addEventListener("ended", () => cur && cur.classList.remove("playing"));
</script>'''
(ROOT / "outputs" / "voice_gallery.html").write_text(page, encoding="utf-8")
print(f"wrote voice_gallery.html — {len(items)} voices ({len(page)/1e6:.2f} MB)")
