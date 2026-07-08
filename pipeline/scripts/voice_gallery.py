"""Render every distinct base-model speaker so you can hear all the voices.

For each speaker in the training mix (>90s audio): extract a voicepack from the
multi-speaker base and render two sentences. Run under gpu_run.sh (one job).
"""
from __future__ import annotations

import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SK = ROOT.parent
PY = SK / "recipe" / ".venv" / "bin" / "python"
EXTRACT = SK / "recipe" / "scripts" / "extract_voicepack.py"
CKPT = ROOT / "outputs" / "checkpoints" / "kokoro_se_base" / "epoch_2nd_00005.pth"
FIRST = ROOT / "outputs" / "checkpoints" / "kokoro_se_base" / "first_stage.pth"
MANIFEST = ROOT / "data" / "manifests" / "train_mix.jsonl"
GAL = ROOT / "outputs" / "gallery"
ENV = {"SV_NEURAL_G2P": "nst_g2p", "PATH": "/usr/bin:/bin"}

SENTENCES = ["Hej, jag är CandyTron, din godisrobot!",
             "Vilken fantastisk dag det blev, tycker du inte det?"]

# group speakers
dur = defaultdict(float)
clips = defaultdict(list)
for line in MANIFEST.open(encoding="utf-8"):
    e = json.loads(line)
    dur[e["speaker_id"]] += e["duration_sec"]
    clips[e["speaker_id"]].append(e["audio_path"])
speakers = sorted([s for s, d in dur.items() if d >= 90])

GAL.mkdir(parents=True, exist_ok=True)
(GAL / "_sentences.txt").write_text("\n".join(SENTENCES) + "\n", encoding="utf-8")
manifest_out = {}

for spk in speakers:
    safe = re.sub(r"[^A-Za-z0-9]+", "_", spk).strip("_")[:40]
    refs = GAL / f"_refs_{safe}"
    refs.mkdir(exist_ok=True)
    for old in refs.glob("*.wav"):
        old.unlink()
    for i, w in enumerate(clips[spk][:30]):
        if Path(w).exists():
            (refs / f"{i:03d}.wav").symlink_to(w)
    vp = GAL / f"vp_{safe}.pt"
    try:
        subprocess.run([str(PY), str(EXTRACT), "--model", str(CKPT),
                        "--style-encoder-model", str(FIRST), "--audio-dir", str(refs),
                        "--output", str(vp), "--num-samples", "30"],
                       check=True, capture_output=True)
        out = GAL / safe
        subprocess.run([str(PY), str(SK / "synth_real.py"), "--checkpoint", str(CKPT),
                        "--voicepack", str(vp), "--texts", str(GAL / "_sentences.txt"),
                        "--out-dir", str(out)], check=True, cwd=SK, env=ENV, capture_output=True)
        manifest_out[safe] = {"speaker": spk, "dir": safe}
        print(f"OK {spk}", flush=True)
    except subprocess.CalledProcessError as ex:
        print(f"FAIL {spk}: {ex.stderr.decode()[-200:] if ex.stderr else ex}", flush=True)

(GAL / "_manifest.json").write_text(json.dumps(manifest_out, ensure_ascii=False, indent=1))
print(f"GALLERY_DONE {len(manifest_out)}/{len(speakers)}", flush=True)
