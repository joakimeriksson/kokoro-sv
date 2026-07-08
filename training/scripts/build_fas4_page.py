"""Fas 4 before/after: old flat Chatterbox-distilled voices vs new real-NST-on-base."""
import base64
import json
import html as H
from pathlib import Path

import numpy as np
import soundfile as sf
import lameenc

ROOT = Path(__file__).resolve().parents[1]
SENTS = [l.strip() for l in (ROOT / "configs" / "testset_sv.txt").open(encoding="utf-8") if l.strip()]
F4 = ROOT / "outputs" / "fas4"
SECTIONS = [
    ("Kvinnlig röst (018)", [
        ("f_old", F4 / "female_old", "Före — Chatterbox-destillat", "F0-std 5.9 st · komb 0.4"),
        ("f_new", F4 / "female", "Efter — riktig NST på flertalar-bas", "F0-std 9.15 st · komb 2.0"),
    ]),
    ("Manlig röst (98)", [
        ("m_old", F4 / "male_old", "Före — Chatterbox-destillat", "F0-std 3.3 st · komb 0.4"),
        ("m_new", F4 / "male", "Efter — riktig NST på flertalar-bas", "F0-std 6.90 st · komb 1.7"),
    ]),
]

b64 = {}
for _, cols in SECTIONS:
    for key, d, _, _ in cols:
        for i in range(1, len(SENTS) + 1):
            p = d / f"test_{i:02d}.wav"
            x, sr = sf.read(str(p), dtype="float32")
            if x.ndim > 1:
                x = x.mean(axis=1)
            x = x / (np.abs(x).max() + 1e-9) * 0.9
            enc = lameenc.Encoder()
            enc.set_bit_rate(80); enc.set_in_sample_rate(sr); enc.set_channels(1); enc.set_quality(2)
            b64[f"{key}_{i}"] = base64.b64encode(bytes(enc.encode((x * 32767).astype("<i2").tobytes())) + bytes(enc.flush())).decode()


def btn(k, label):
    return (f'<button class="play" data-key="{k}" aria-label="{H.escape(label)}">'
            '<svg class="ic-play" viewBox="0 0 24 24" width="18" height="18"><path d="M8 5v14l11-7z" fill="currentColor"/></svg>'
            '<svg class="ic-pause" viewBox="0 0 24 24" width="18" height="18"><path d="M6 5h4v14H6zM14 5h4v14h-4z" fill="currentColor"/></svg></button>')


sections = ""
for title, cols in SECTIONS:
    heads = "".join(f'<div><strong>{H.escape(n)}</strong><span>{H.escape(m)}</span></div>' for _, _, n, m in cols)
    rows = ""
    for i, s in enumerate(SENTS, 1):
        cells = "".join(f'<div class="cell">{btn(f"{k}_{i}", n)}</div>' for k, _, n, _ in cols)
        rows += f'<div class="row"><p class="line">{i}. {H.escape(s)}</p><div class="cells">{cells}</div></div>'
    sections += f'<section><h2>{H.escape(title)}</h2><div class="board"><div class="vheads">{heads}</div>{rows}</div></section>'

page = f'''<title>Fas 4 — före/efter: dynamiska CandyTron-röster</title>
<style>
:root {{ --bg:#F7F9FB; --card:#FFF; --ink:#1A2230; --muted:#68727F; --line:#E2E8F0; --accent:#0D9488; --accent-soft:#D7F4F0; }}
body {{ background:var(--bg); color:var(--ink); margin:0; padding:2.5rem 1.25rem 4rem; font:16px/1.55 system-ui,sans-serif; }}
main {{ max-width:46rem; margin:0 auto; }}
h1 {{ font-size:1.6rem; margin:0 0 .5rem; text-wrap:balance; }} h1 em {{ color:var(--accent); font-style:normal; }}
header p {{ color:var(--muted); margin:.25rem 0 0; }}
section {{ margin-top:2.25rem; }}
h2 {{ font-size:.8rem; letter-spacing:.14em; text-transform:uppercase; color:var(--muted); border-bottom:1px solid var(--line); padding-bottom:.5rem; margin:0 0 1rem; }}
.board {{ background:var(--card); border:1px solid var(--line); border-radius:12px; }}
.vheads {{ display:grid; grid-template-columns:repeat(2,1fr); gap:.5rem; padding:1rem 1.25rem .75rem; border-bottom:1px solid var(--line); font-size:.82rem; }}
.vheads div {{ display:flex; flex-direction:column; }} .vheads span {{ color:var(--muted); font-size:.74rem; }}
.row {{ padding:.8rem 1.25rem; border-bottom:1px solid var(--line); }} .row:last-child {{ border-bottom:none; }}
.line {{ margin:0 0 .5rem; font-size:.92rem; }}
.cells {{ display:grid; grid-template-columns:repeat(2,1fr); gap:.5rem; }}
.play {{ width:2.4rem; height:2.4rem; border-radius:50%; border:1px solid var(--line); background:var(--accent-soft); color:var(--accent); display:grid; place-items:center; cursor:pointer; }}
.play:hover, .play.playing {{ background:var(--accent); color:#fff; }}
.ic-pause {{ display:none; }} .playing .ic-pause {{ display:block; }} .playing .ic-play {{ display:none; }}
.note {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:.9rem 1.1rem; color:var(--muted); font-size:.9rem; margin-top:1.5rem; }}
</style>
<main>
<header><h1>CandyTron-rösterna, <em>nu dynamiska</em></h1>
<p>Vänster: gamla rösten (destillerad genom Chatterbox, prosodiskt platt). Höger: samma talares
RIKTIGA NST-inspelning som voicepack på den nya flertalar-basen — Chatterbox är helt borta ur kedjan.
Tonhöjdsdynamiken nästan fördubblas, och talet blir tydligare (CER lägre).</p></header>
{sections}
<p class="note">Öppet kvar: den nya basen har mer komb (~1,7–2,0 mot 0,4 för enskild talare) trots notchfiltret
— nästa steg är att jaga den (prosodi-finetuning / notch-tuning / decoder-lr). Rösterna är dynamiska och
begripliga; det som återstår är den sista signalrenheten.</p>
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
(ROOT / "outputs" / "fas4_page.html").write_text(page, encoding="utf-8")
print(f"wrote fas4_page.html ({len(page)/1e6:.2f} MB)")
