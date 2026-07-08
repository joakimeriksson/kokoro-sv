"""Assemble the HuggingFace pack: multi-speaker base + chosen named voicepacks.

    build_hf_pack.py "Signe,Astrid,Ebba,Vera,Stina,Björn,Sven,Hjalmar,Rune,Olof"

Produces hf_pack/ with kokoro_sv.pth (KModel), voices/<Name>.pt, samples/<Name>.wav,
config.json, README.md. Run under gpu_run.sh (checkpoint conversion uses the model).
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SK = ROOT.parent
sys.path.insert(0, str(SK))
GAL = ROOT / "outputs" / "gallery"
CKPT = ROOT / "outputs" / "checkpoints" / "kokoro_se_base" / "epoch_2nd_00005.pth"
PACK = ROOT / "hf_pack"
NAMES = json.loads((ROOT / "configs" / "voice_names.json").read_text(encoding="utf-8"))
NAME2SPK = {v["name"]: k for k, v in NAMES.items()}

chosen = [n.strip() for n in sys.argv[1].split(",")] if len(sys.argv) > 1 else []
PACK.mkdir(exist_ok=True)
(PACK / "voices").mkdir(exist_ok=True)
(PACK / "samples").mkdir(exist_ok=True)

# 1. convert base checkpoint -> KModel format
import os
os.environ.setdefault("SV_NEURAL_G2P", "nst_g2p")
from synth_real import convert_checkpoint
convert_checkpoint(str(CKPT), str(PACK / "kokoro_sv.pth"))
# CRITICAL: convert to STOCK format so `pip install kokoro` loads the decoder
# (recipe format uses new weight-norm API + module. prefix -> stock drops it -> noise)
import subprocess as _sp
_sp.run([sys.executable, str(SK / "convert_to_stock.py"), "--in", str(PACK / "kokoro_sv.pth"),
         "--out", str(PACK / "kokoro_sv.pth")], check=True)
shutil.copy(SK / "recipe" / "training" / "config.json", PACK / "config.json")

# 2. gather voicepacks + samples for the chosen names
roster = []
for name in chosen:
    spk = NAME2SPK.get(name)
    if not spk:
        print(f"!! unknown name {name}"); continue
    safe = re.sub(r"[^A-Za-z0-9]+", "_", spk).strip("_")[:40]
    vp = GAL / f"vp_{safe}.pt"
    smp = GAL / safe / "test_01.wav"
    if not vp.exists():
        print(f"!! no voicepack for {name} ({spk})"); continue
    shutil.copy(vp, PACK / "voices" / f"{name}.pt")
    if smp.exists():
        shutil.copy(smp, PACK / "samples" / f"{name}.wav")
    g = NAMES[spk].get("gender", "?")
    roster.append((name, g, NAMES[spk].get("role", "")))
    print(f"added {name} ({g})")

(PACK / "voices_manifest.json").write_text(
    json.dumps({n: {"gender": g, "role": r} for n, g, r in roster}, ensure_ascii=False, indent=1))
print(f"PACK_DONE {len(roster)} voices -> {PACK}")
