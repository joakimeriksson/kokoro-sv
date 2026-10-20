"""Fine-tune the Swedish multi-speaker base on ONE voice -> a separate model.

A fitted style pack (scripts/fit_voicepack.py) only moves timbre: the prosody
(pitch movement, timing) comes from the predictor, which all packs share, so a
fitted voice sounds as flat as the base. This continues stage 2 from the base
checkpoint (every module, including the trained predictor_encoder and the
discriminators) on a single synthesized voice, e.g. Klara's VoxCPM2 set, so the
predictor learns that voice's prosody. Checkpoints go to
outputs/checkpoints/<name>/; the published kokoro_sv.pth and its ten voices are
untouched (the result is served as its own model).

    # filelists + config only
    python training/scripts/finetune_voice.py --dataset data-gen/dataset_vox_female --name klara
    # smoke: 40 clips, 1 epoch -- ALWAYS first (RUN2 discipline)
    ../gpu_run.sh python training/scripts/finetune_voice.py --dataset ... --name klara --smoke --launch
    # full run
    ../gpu_run.sh python training/scripts/finetune_voice.py --dataset ... --name klara --launch

The dataset is the batch_voxcpm.py contract (wavs/ + speakers.csv) gated by
scripts/verify_vox_dataset.py; only verified.csv rows with keep=1 are used.

--styles DIR adds style sets (DIR/<style>/ in the same contract, e.g. from
data-gen/styles_gen.sh), each as its own speaker id (the plain set is id 0).
train_second.py draws a clip's style reference from the same id, so the style
encoder learns to carry the style; a pack extracted from one style's clips then
speaks in that style. Continue from the voice's approved checkpoint with --base.
Needs recipe/StyleTTS2/train_second.py to honour `keep_predictor_encoder`
(otherwise loading a stage-2 checkpoint resets predictor_encoder to a copy of
style_encoder, and the copy is never registered with the optimizer).
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SK = ROOT.parent
sys.path.insert(0, str(SK))
BASE = ROOT / "outputs" / "checkpoints" / "kokoro_se_base" / "epoch_2nd_00005.pth"


def unprefixed(src: Path, dst: Path) -> Path:
    """Stage-2 checkpoints are saved from DataParallel-wrapped modules ("module."
    keys) but train_second loads BEFORE wrapping with strict=False, so a raw
    stage-2 checkpoint silently loads nothing for bert/predictor/decoder/...
    Write a copy with the prefix stripped."""
    if dst.exists():
        return dst
    import torch
    state = torch.load(src, map_location="cpu", weights_only=False)
    net = {k: {(p[7:] if p.startswith("module.") else p): v for p, v in sd.items()}
           for k, sd in state["net"].items()}
    assert not any(p.startswith("module.") for sd in net.values() for p in sd)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"net": net, "epoch": 0, "iters": 0}, dst)
    print(f"base without DataParallel prefix -> {dst}")
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="dir with wavs/ and verified.csv")
    ap.add_argument("--name", required=True)
    ap.add_argument("--base", default=str(BASE))
    ap.add_argument("--val", type=int, default=50)
    ap.add_argument("--epochs-2nd", type=int, default=4)
    ap.add_argument("--styles", help="dir of style sets <style>/{wavs,verified.csv}, one speaker id each")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--launch", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("SV_NEURAL_G2P", "nst_g2p")
    from g2p_sv import SwedishG2P, validate_against_kokoro, kokoro_vocab  # noqa: E402

    g2p = SwedishG2P(backend="neural")
    vocab = kokoro_vocab()

    dset = Path(args.dataset).resolve()
    sets = [(dset, 0)]
    if args.styles:
        sdirs = sorted(d for d in Path(args.styles).resolve().iterdir() if (d / "verified.csv").exists())
        sets += [(d, i) for i, d in enumerate(sdirs, 1)]
        print("speaker ids: 0=plain, " + ", ".join(f"{i}={d.name}" for d, i in sets[1:]))

    lines, oov = [], defaultdict(int)
    for sdir, spk in sets:
        rows = [r for r in csv.DictReader(open(sdir / "verified.csv", encoding="utf-8"), delimiter="|")
                if r["keep"] == "1"]
        print(f"{len(rows)} kept clips in {sdir}")
        for r in rows:
            ph = g2p(r["text"])
            if not ph or len(ph) > 400:
                continue
            bad = validate_against_kokoro(ph, vocab)
            if bad:
                for s in bad:
                    oov[s] += 1
                continue
            # paths are relative to root_path (the plain set); style sets live beside it
            lines.append((os.path.relpath(sdir / r["audio"], dset), ph, spk))
    if oov:
        print(f"!! dropped clips with OOV symbols: {dict(oov)}")
    print(f"{len(lines)} clips after G2P/vocab gates")

    rng = random.Random(1337)
    rng.shuffle(lines)
    val, train = lines[: args.val], lines[args.val:]
    if args.smoke:
        train, val = train[:40], val[:10]

    ddir = ROOT / "data" / "training" / f"{args.name}{'_smoke' if args.smoke else ''}"
    ddir.mkdir(parents=True, exist_ok=True)
    for fname, subset in (("train_list.txt", train), ("val_list.txt", val)):
        with open(ddir / fname, "w", encoding="utf-8") as f:
            f.writelines(f"{rel}|{ph}|{spk}\n" for rel, ph, spk in subset)
    with open(ddir / "OOD_texts.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(ph for _, ph, _ in lines) + "\n")
    print(f"filelists -> {ddir} ({len(train)} train / {len(val)} val)")

    # config from the verified v3 template: multispeaker (style refs drive the
    # predictor), stage 2 warm-started from the full base checkpoint
    log_dir = ROOT / "outputs" / "checkpoints" / f"{args.name}{'_smoke' if args.smoke else ''}"
    base = unprefixed(Path(args.base).resolve(),
                      Path(args.base).resolve().with_name(Path(args.base).stem + "_unprefixed.pth"))
    tmpl = (SK / "configs" / "config_sv_female_v3.yml").read_text()
    subs = {
        'log_dir: "../../output/sv_kokoro_female_v3"': f'log_dir: "{log_dir}"',
        'train_data: "../../data_female_v3/train_list.txt"': f'train_data: "{ddir}/train_list.txt"',
        'val_data: "../../data_female_v3/val_list.txt"': f'val_data: "{ddir}/val_list.txt"',
        'root_path: "../../data_female_v3"': f'root_path: "{dset}"',
        'OOD_data: "../../data_female_v3/OOD_texts.txt"': f'OOD_data: "{ddir}/OOD_texts.txt"',
        'pretrained_model: "../../models/kokoro_base.pth"': f'pretrained_model: "{base}"',
        "second_stage_load_pretrained: false": "second_stage_load_pretrained: true\n"
                                               "keep_predictor_encoder: true",
        "multispeaker: false": "multispeaker: true",
        "epochs_2nd: 8": f"epochs_2nd: {1 if args.smoke else args.epochs_2nd}",
    }
    cfg = tmpl
    for old, new in subs.items():
        if old not in cfg:
            sys.exit(f"template changed, cannot find: {old}")
        cfg = cfg.replace(old, new)
    if args.smoke:
        cfg = cfg.replace("batch_size: 4", "batch_size: 2")

    cfg_path = ROOT / "configs" / f"finetune_{args.name}{'_smoke' if args.smoke else ''}.yml"
    cfg_path.write_text(cfg)
    print(f"config -> {cfg_path}")

    if args.launch:
        # the CALLER wraps this script with gpu_run.sh (flock is not reentrant)
        # via -m: the venv was moved, so bin/accelerate's shebang points nowhere
        py = SK / "recipe" / ".venv" / "bin" / "python"
        cmd = [str(py), "-m", "accelerate.commands.launch", "train_second.py",
               "--config_path", str(cfg_path)]
        print("launching stage 2:", " ".join(cmd[3:]), flush=True)
        env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        subprocess.run(cmd, cwd=SK / "recipe" / "StyleTTS2", check=True, env=env)


if __name__ == "__main__":
    main()
