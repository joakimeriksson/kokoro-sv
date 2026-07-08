"""Add prosody features + coarse class to every entry of a manifest (in place).

    python scripts/extract_prosody.py --manifest data/manifests/nst.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqdm import tqdm

from kokoro_se.dataset_manifest import read_manifest, write_manifest
from kokoro_se.audio_utils import load_mono
from kokoro_se.prosody import extract_prosody, prosody_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()

    entries = list(read_manifest(args.manifest))
    for e in tqdm(entries, desc="prosody"):
        if e.get("prosody"):
            continue
        try:
            x, sr = load_mono(e["audio_path"])
        except Exception:
            continue
        p = extract_prosody(x, sr)
        e["prosody"] = p
        e["prosody_class"] = prosody_class(p, e["text"])
    write_manifest(entries, args.manifest)
    from collections import Counter
    print(Counter(e.get("prosody_class", "?") for e in entries))


if __name__ == "__main__":
    main()
