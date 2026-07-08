"""Build a training manifest from configs/datasets.yaml mix weights.

    python scripts/build_mix.py --config configs/datasets.yaml --out data/manifests/train_mix.jsonl
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import yaml

from kokoro_se.dataset_manifest import read_manifest, write_manifest, manifest_stats
from kokoro_se.prosody import prosody_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/datasets.yaml")
    ap.add_argument("--out", default="data/manifests/train_mix.jsonl")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / args.config).read_text())
    sel = cfg.get("selection", {})
    rng = random.Random(args.seed)

    picked = []
    for ds, weight in cfg["mix"].items():
        if weight <= 0:
            continue
        mpath = root / "data" / "manifests" / f"{ds}.jsonl"
        if not mpath.exists():
            print(f"  !! no manifest for '{ds}' ({mpath}) — skipping its {weight:.0%}")
            continue
        target_sec = cfg["target_hours"] * 3600 * weight
        pool = [e for e in read_manifest(mpath)
                if sel.get("min_clip_sec", 0) <= e["duration_sec"] <= sel.get("max_clip_sec", 1e9)
                and (e.get("quality_score") is None
                     or e["quality_score"] >= sel.get("min_quality_score", 0))]
        # oversample preferred prosody classes by duplicating their tickets
        prefer = set(sel.get("prefer_prosody_classes", []))
        tickets = []
        for e in pool:
            k = prosody_class(e.get("prosody") or {}, e["text"])
            tickets.append((e, 2 if k in prefer else 1))
        weighted = [e for e, w in tickets for _ in range(w)]
        rng.shuffle(weighted)
        got, seen = 0.0, set()
        for e in weighted:
            if got >= target_sec:
                break
            if id(e) in seen:
                continue
            seen.add(id(e))
            picked.append(e)
            got += e["duration_sec"]
        print(f"  {ds}: {got/3600:.2f} h of {target_sec/3600:.2f} h target ({len(seen)} clips)")

    rng.shuffle(picked)
    out = root / args.out
    write_manifest(picked, out)
    print(f"mix -> {out}: {manifest_stats(out)}")


if __name__ == "__main__":
    main()
