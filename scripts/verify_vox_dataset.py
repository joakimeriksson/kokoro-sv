"""Gate a synthesized single-speaker dataset (wavs/ + speakers.csv) clip by clip.

Per clip:
  * ECAPA cosine to the reference take, and to the dataset's own centroid
    (VoxCPM2 occasionally drifts to another speaker mid-dataset),
  * Whisper CER against the text it was asked to say (dropped words, babble),
  * duration and peak level.

Writes <dataset>/verified.csv (all columns + keep flag) and prints a summary.
Thresholds follow docs/VOX_VOICES_PLAN.md step 2.

  python scripts/verify_vox_dataset.py data-gen/dataset_vox_female \\
      --ref data-gen/refs/vox_female_ref.wav
Run in the kokoro-voice-server venv (speechbrain, faster-whisper).
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import warnings

import numpy as np
import soundfile as sf
import torch
import torchaudio

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)


def norm_text(s: str) -> str:
    return " ".join(re.sub(r"[^\w ]", "", s.lower()).split())


def cer(r: str, h: str) -> float:
    r, h = norm_text(r), norm_text(h)
    d = list(range(len(h) + 1))
    for i, a in enumerate(r, 1):
        p, d[0] = d[0], i
        for j, b in enumerate(h, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (a != b))
    return d[len(h)] / max(1, len(r))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--min-ref-sim", type=float, default=0.70)
    ap.add_argument("--min-centroid-sim", type=float, default=0.75)
    ap.add_argument("--max-cer", type=float, default=0.20)
    ap.add_argument("--whisper", default="small")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    from speechbrain.inference.speaker import EncoderClassifier
    from faster_whisper import WhisperModel

    enc = EncoderClassifier.from_hparams(
        "speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.expanduser("~/.cache/speechbrain/spkrec-ecapa-voxceleb"),
        run_opts={"device": "cpu"})
    asr = WhisperModel(args.whisper, device="cpu", compute_type="int8", cpu_threads=args.threads)

    def load16(path: str) -> torch.Tensor:
        x, sr = sf.read(path, dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        return torchaudio.functional.resample(torch.from_numpy(x)[None], sr, 16000)

    @torch.no_grad()
    def embed(a16: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(enc.encode_batch(a16).reshape(-1), dim=0)

    ref = embed(load16(args.ref))
    rows = list(csv.DictReader(open(os.path.join(args.dataset, "speakers.csv"), encoding="utf-8"),
                               delimiter="|"))
    embs, out = [], []
    for k, row in enumerate(rows):
        a16 = load16(os.path.join(args.dataset, row["audio"]))
        e = embed(a16)
        embs.append(e)
        segs, _ = asr.transcribe(a16[0].numpy(), language="sv", beam_size=1)
        heard = " ".join(s.text for s in segs).strip()
        out.append({**row, "ref_sim": round(float(e @ ref), 3), "heard": heard,
                    "cer": round(cer(row["text"], heard), 3),
                    "seconds": round(a16.shape[1] / 16000, 2),
                    "peak": round(float(a16.abs().max()), 3)})
        if (k + 1) % 200 == 0:
            print(f"  {k + 1}/{len(rows)}", flush=True)

    centroid = torch.nn.functional.normalize(torch.stack(embs).mean(0), dim=0)
    for o, e in zip(out, embs):
        o["centroid_sim"] = round(float(e @ centroid), 3)
        reasons = []
        if o["ref_sim"] < args.min_ref_sim:
            reasons.append("ref_sim")
        if o["centroid_sim"] < args.min_centroid_sim:
            reasons.append("centroid_sim")
        if o["cer"] > args.max_cer:
            reasons.append("cer")
        if o["peak"] >= 0.999:
            reasons.append("clipped")
        o["keep"] = int(not reasons)
        o["reject"] = ",".join(reasons)

    path = os.path.join(args.dataset, "verified.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()), delimiter="|")
        w.writeheader()
        w.writerows(out)

    kept = [o for o in out if o["keep"]]
    hours = sum(o["seconds"] for o in kept) / 3600
    print(f"{args.dataset}: kept {len(kept)}/{len(out)} clips, {hours:.2f} h")
    print(f"  ref_sim median {np.median([o['ref_sim'] for o in out]):.3f}, "
          f"centroid_sim median {np.median([o['centroid_sim'] for o in out]):.3f}, "
          f"CER median {np.median([o['cer'] for o in out]):.3f}")
    for reason in ("ref_sim", "centroid_sim", "cer", "clipped"):
        n = sum(reason in o["reject"].split(",") for o in out)
        if n:
            print(f"  rejected for {reason}: {n}")
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
