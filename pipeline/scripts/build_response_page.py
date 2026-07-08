"""Listening page: the +4.66 st prosody-responsiveness result — same base model,
same sentences, two speaker voicepacks (monotone narrator vs expressive nst_718)."""
import base64
import json
import html as H
from pathlib import Path

import numpy as np
import soundfile as sf
import lameenc

ROOT = Path(__file__).resolve().parents[1]
SENTS = [l.strip() for l in (ROOT / "configs" / "testset_sv.txt").open(encoding="utf-8") if l.strip()]
COLS = [("low", "Monoton talare", "F0-std 4.85 st"),
        ("high", "Expressiv talare", "F0-std 9.52 st")]

b64 = {}
for tag, _, _ in COLS:
    for i in range(1, len(SENTS) + 1):
        p = ROOT / "outputs" / "response" / tag / f"test_{i:02d}.wav"
        x, sr = sf.read(str(p), dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        x = x / (np.abs(x).max() + 1e-9) * 0.9
        enc = lameenc.Encoder()
        enc.set_bit_rate(80); enc.set_in_sample_rate(sr); enc.set_channels(1); enc.set_quality(2)
        b64[f"{tag}_{i}"] = base64.b64encode(bytes(enc.encode((x * 32767).astype("<i2").tobytes())) + bytes(enc.flush())).decode()


def btn(k, label):
    return (f'<button class="play" data-key="{k}" aria-label="{H.escape(label)}">'
            '<svg class="ic-play" viewBox="0 0 24 24" width="18" height="18"><path d="M8 5v14l11-7z" fill="currentColor"/></svg>'
            '<svg class="ic-pause" viewBox="0 0 24 24" width="18" height="18"><path d="M6 5h4v14H6zM14 5h4v14h-4z" fill="currentColor"/></svg></button>')


rows = ""
for i, s in enumerate(SENTS, 1):
    cells = "".join(f'<div class="cell">{btn(f"{t}_{i}", n)}</div>' for t, n, _ in COLS)
    rows += f'<div class="row"><p class="line">{i}. {H.escape(s)}</p><div class="cells">{cells}</div></div>'
heads = "".join(f'<div><strong>{H.escape(n)}</strong><span>{H.escape(d)}</span></div>' for _, n, d in COLS)

page = f'''<title>Prosodirespons — samma modell, olika talare</title>
<style>
:root {{ --bg:#F7F9FB; --card:#FFF; --ink:#1A2230; --muted:#68727F; --line:#E2E8F0; --accent:#2563EB; --accent-soft:#E7EEFE; }}
body {{ background:var(--bg); color:var(--ink); margin:0; padding:2.5rem 1.25rem 4rem; font:16px/1.55 system-ui,sans-serif; }}
main {{ max-width:46rem; margin:0 auto; }}
h1 {{ font-size:1.6rem; margin:0 0 .5rem; text-wrap:balance; }} h1 em {{ color:var(--accent); font-style:normal; }}
header p {{ color:var(--muted); margin:.25rem 0 0; }}
.board {{ background:var(--card); border:1px solid var(--line); border-radius:12px; margin-top:1.75rem; }}
.vheads {{ display:grid; grid-template-columns:repeat(2,1fr); gap:.5rem; padding:1rem 1.25rem .75rem; border-bottom:1px solid var(--line); font-size:.82rem; }}
.vheads div {{ display:flex; flex-direction:column; }} .vheads span {{ color:var(--muted); font-size:.74rem; }}
.row {{ padding:.85rem 1.25rem; border-bottom:1px solid var(--line); }} .row:last-child {{ border-bottom:none; }}
.line {{ margin:0 0 .5rem; font-size:.94rem; }}
.cells {{ display:grid; grid-template-columns:repeat(2,1fr); gap:.5rem; }}
.play {{ width:2.4rem; height:2.4rem; border-radius:50%; border:1px solid var(--line); background:var(--accent-soft); color:var(--accent); display:grid; place-items:center; cursor:pointer; }}
.play:hover, .play.playing {{ background:var(--accent); color:#fff; }}
.ic-pause {{ display:none; }} .playing .ic-pause {{ display:block; }} .playing .ic-play {{ display:none; }}
.note {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:.9rem 1.1rem; color:var(--muted); font-size:.9rem; margin-top:1.5rem; }}
</style>
<main>
<header><h1>Samma modell, <em>olika prosodi</em></h1>
<p>Kokoro-se-base (23 talare) renderar identiska meningar med två olika voicepacks.
Enda skillnaden är stilvektorn — modellen följer talarens tonhöjdsdynamik: 4,85 vs 9,52 semitoner.
Det här är beviset att rösten kan bli dynamisk (gamla Chatterbox-rösterna satt fast på 3–6 oavsett voicepack).</p></header>
<div class="board"><div class="vheads">{heads}</div>{rows}</div>
<p class="note">SPREAD +4,66 st = RESPONSIV. Nästa steg: förankra CandyTron-rösterna (riktiga NST 018/98)
ovanpå den här basen, och bygg om känslolägena — de bör nu ligga inom modellens manifold.</p>
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
(ROOT / "outputs" / "response_page.html").write_text(page, encoding="utf-8")
print(f"wrote response_page.html ({len(page)/1e6:.2f} MB)")
