"""Bridge: kokoro-se train manifest -> StyleTTS2/kikiri training run.

Converts data/manifests/train_mix.jsonl into StyleTTS2 filelists (phonemized
with the proven Swedish neural G2P), generates a multi-speaker config from the
verified v3 template, and (optionally) launches training via the sibling
swedish-kokoro environment. Speaker IDs become numeric indices (meldataset
does int(speaker_id) — RUN2 gotcha).

    # prep filelists + config only
    python scripts/train_kokoro.py --manifest data/manifests/train_mix.jsonl --name kokoro_se_base
    # smoke: tiny subset, 1 epoch — ALWAYS run before a full run (RUN2 discipline)
    python scripts/train_kokoro.py --manifest ... --name kokoro_se_base --smoke --launch
    # full run
    python scripts/train_kokoro.py --manifest ... --name kokoro_se_base --launch
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SK = ROOT.parent
GPU_LOCK = ROOT.parent / "gpu_run.sh"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SK))

from kokoro_se.dataset_manifest import read_manifest  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifests/train_mix.jsonl")
    ap.add_argument("--name", default="kokoro_se_base")
    ap.add_argument("--val", type=int, default=150)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--epochs-2nd", type=int, default=8)
    args = ap.parse_args()

    import os
    os.environ.setdefault("SV_NEURAL_G2P", "nst_g2p")
    from g2p_sv import SwedishG2P, validate_against_kokoro, kokoro_vocab  # noqa: E402

    g2p = SwedishG2P(backend="neural")
    vocab = kokoro_vocab()

    entries = list(read_manifest(ROOT / args.manifest))
    speakers = sorted({e["speaker_id"] for e in entries})
    spk_idx = {s: i for i, s in enumerate(speakers)}
    print(f"{len(entries)} clips, {len(speakers)} speakers (numeric ids 0..{len(speakers)-1})")

    lines, all_ph, oov = [], [], defaultdict(int)
    for e in entries:
        ph = g2p(e["text"])
        if not ph or len(ph) > 400:
            continue
        bad = validate_against_kokoro(ph, vocab)
        if bad:
            for s in bad:
                oov[s] += 1
            continue
        rel = str(Path(e["audio_path"]).resolve().relative_to(ROOT / "data"))
        lines.append((rel, ph, spk_idx[e["speaker_id"]], e["speaker_id"]))
        all_ph.append(ph)
    if oov:
        print(f"!! dropped clips with OOV symbols: {dict(oov)}")
    print(f"{len(lines)} clips after G2P/vocab gates")

    # val split must contain multiple speakers with >=2 clips (multispeaker refs)
    rng = random.Random(1337)
    rng.shuffle(lines)
    val, train = lines[: args.val], lines[args.val:]

    ddir = ROOT / "data" / "training" / args.name
    ddir.mkdir(parents=True, exist_ok=True)
    for fname, subset in (("train_list.txt", train), ("val_list.txt", val)):
        with open(ddir / fname, "w", encoding="utf-8") as f:
            f.writelines(f"{rel}|{ph}|{idx}\n" for rel, ph, idx, _ in subset)
    with open(ddir / "OOD_texts.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(all_ph) + "\n")
    with open(ddir / "speaker_map.json", "w", encoding="utf-8") as f:
        json.dump(spk_idx, f, ensure_ascii=False, indent=1)
    print(f"filelists -> {ddir} ({len(train)} train / {len(val)} val)")

    # config from the verified v3 template, multispeaker enabled, absolute paths
    tmpl = (SK / "configs" / "config_sv_female_v3.yml").read_text()
    cfg = tmpl.replace('log_dir: "../../output/sv_kokoro_female_v3"',
                       f'log_dir: "{ROOT}/outputs/checkpoints/{args.name}"')
    cfg = cfg.replace('train_data: "../../data_female_v3/train_list.txt"', f'train_data: "{ddir}/train_list.txt"')
    cfg = cfg.replace('val_data: "../../data_female_v3/val_list.txt"', f'val_data: "{ddir}/val_list.txt"')
    cfg = cfg.replace('root_path: "../../data_female_v3"', f'root_path: "{ROOT}/data"')
    cfg = cfg.replace('OOD_data: "../../data_female_v3/OOD_texts.txt"', f'OOD_data: "{ddir}/OOD_texts.txt"')
    cfg = cfg.replace("multispeaker: false", "multispeaker: true")
    cfg = cfg.replace("epochs_2nd: 8", f"epochs_2nd: {args.epochs_2nd}")
    if args.smoke:
        cfg = cfg.replace("epochs_1st: 6", "epochs_1st: 1").replace("epochs: 6", "epochs: 1")
        cfg = cfg.replace("batch_size: 4", "batch_size: 2")
        # tiny subset spanning many speakers
        by_spk = defaultdict(list)
        for l in train:
            by_spk[l[2]].append(l)
        sub = [l for spk in list(by_spk)[:10] for l in by_spk[spk][:5]]
        with open(ddir / "_smoke_train.txt", "w", encoding="utf-8") as f:
            f.writelines(f"{rel}|{ph}|{idx}\n" for rel, ph, idx, _ in sub)
        with open(ddir / "_smoke_val.txt", "w", encoding="utf-8") as f:
            f.writelines(f"{rel}|{ph}|{idx}\n" for rel, ph, idx, _ in sub[:20])
        cfg = cfg.replace(f'{ddir}/train_list.txt', f'{ddir}/_smoke_train.txt')
        cfg = cfg.replace(f'{ddir}/val_list.txt', f'{ddir}/_smoke_val.txt')
        print(f"smoke subset: {len(sub)} clips across {min(10, len(by_spk))} speakers")

    cfg_path = ROOT / "configs" / f"train_{args.name}{'_smoke' if args.smoke else ''}.yml"
    cfg_path.write_text(cfg)
    print(f"config -> {cfg_path}")

    if args.launch:
        # NOTE: no GPU_LOCK here — the CALLER wraps this whole script with gpu_run.sh
        # (flock is not reentrant; double-wrapping self-deadlocks. Learned 2026-07-07.)
        acc = SK / "recipe" / ".venv" / "bin" / "accelerate"
        cmd = [str(acc), "launch", "train_first.py", "--config_path", str(cfg_path)]
        print("launching stage 1:", " ".join(cmd[1:]))
        env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        subprocess.run(cmd, cwd=SK / "recipe" / "StyleTTS2", check=True, env=env)
        if not args.smoke:
            subprocess.run([str(acc), "launch", "train_second.py",
                            "--config_path", str(cfg_path)],
                           cwd=SK / "recipe" / "StyleTTS2", check=True, env=env)


if __name__ == "__main__":
    main()
