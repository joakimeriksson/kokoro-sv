"""Extract + render speakers from a manifest through the CURRENT base, merging
them into outputs/gallery/ (no retrain). Usage: add_speakers_to_gallery.py <manifest> [max_speakers]
"""
from __future__ import annotations

import json
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SK = ROOT.parent
PY = SK / "recipe" / ".venv" / "bin" / "python"
EXTRACT = SK / "recipe" / "scripts" / "extract_voicepack.py"
CKPT = ROOT / "outputs" / "checkpoints" / "kokoro_se_base" / "epoch_2nd_00005.pth"
FIRST = ROOT / "outputs" / "checkpoints" / "kokoro_se_base" / "first_stage.pth"
GAL = ROOT / "outputs" / "gallery"
ENV = {"SV_NEURAL_G2P": "nst_g2p", "PATH": "/usr/bin:/bin"}

manifest_path = Path(sys.argv[1])
max_spk = int(sys.argv[2]) if len(sys.argv) > 2 else 12

dur, clips, f0m = defaultdict(float), defaultdict(list), defaultdict(list)
for line in manifest_path.open(encoding="utf-8"):
    e = json.loads(line)
    dur[e["speaker_id"]] += e["duration_sec"]
    clips[e["speaker_id"]].append(e["audio_path"])
    if e.get("prosody"):
        f0m[e["speaker_id"]].append(e["prosody"]["f0_mean"])
# most audio first, deepest-pitched preferred for male-ness
speakers = sorted([s for s, d in dur.items() if d >= 90],
                  key=lambda s: (-dur[s]))[:max_spk]

mf = json.loads((GAL / "_manifest.json").read_text()) if (GAL / "_manifest.json").exists() else {}
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
        subprocess.run([str(PY), str(EXTRACT), "--model", str(CKPT), "--style-encoder-model",
                        str(FIRST), "--audio-dir", str(refs), "--output", str(vp),
                        "--num-samples", "30"], check=True, capture_output=True)
        out = GAL / safe
        subprocess.run([str(PY), str(SK / "synth_real.py"), "--checkpoint", str(CKPT),
                        "--voicepack", str(vp), "--texts", str(GAL / "_sentences.txt"),
                        "--out-dir", str(out)], check=True, cwd=SK, env=ENV, capture_output=True)
        mp = statistics.median(f0m[spk]) if f0m.get(spk) else 0
        mf[safe] = {"speaker": spk, "dir": safe}
        print(f"OK {spk}  pitch={mp:.0f}Hz", flush=True)
    except subprocess.CalledProcessError as ex:
        print(f"FAIL {spk}: {ex.stderr.decode()[-160:] if ex.stderr else ex}", flush=True)

(GAL / "_manifest.json").write_text(json.dumps(mf, ensure_ascii=False, indent=1))
print(f"ADD_DONE total_in_gallery={len(mf)}", flush=True)
