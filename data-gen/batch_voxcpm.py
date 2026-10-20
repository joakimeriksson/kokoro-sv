"""Mass-synthesize a single-speaker Swedish dataset with VoxCPM2 (distillation
source for a new Kokoro voice), cloning ONE fixed reference take.

Same output contract as batch_synth.py (Chatterbox), so prepare_data.py --raw
consumes it unchanged:
    <out>/wavs/<idx>.wav      24 kHz mono
    <out>/speakers.csv        audio|text|speaker_id|duration|dnsmos

Uses VoxCPM2 "ultimate cloning": reference_wav_path + prompt_wav_path +
prompt_text, which continues the reference speaker's timbre AND prosody. Measured
on the two chosen takes: every new sentence stays on the voice (ECAPA similarity
0.79-0.90 to the reference; the other voice ~0.37). A style instruction in
parentheses made the voices livelier but drifted identity (0.69-0.81), so none
is used. Give numbers as digits ("Candytron 4000"): VoxCPM2 reads digits
naturally but stumbles on long spelled-out compounds ("fyratusen").

With --instruction it uses controllable cloning instead: reference_wav_path only
(timbre) plus "(instruction)" before the text (style), for style sets such as
whispering. Ultimate cloning reads an instruction aloud, so it cannot take one;
controllable cloning drifts identity more (style_probe.py), so gate the set with
scripts/verify_vox_dataset.py.

Resumable (skips existing wavs); rewrites the manifest every 10 clips.

    python batch_voxcpm.py --ref refs/vox_female_ref.wav --speaker sv_vox_female \\
        --corpus corpus_vox_sv.txt --out dataset_vox_female [--limit 20]
"""
from __future__ import annotations

import argparse
import csv
import inspect
import os
import time

import numpy as np
import soundfile as sf

PROMPT_TEXT = ("Det finns tretton sorters godis på bordet, till exempel Geisha, "
               "Marianne och Plopp. Säg vilken du vill ha, så plockar jag den åt dig.")
OUT_SR = 24000


def write_manifest(path: str, rows: list[tuple]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="|")
        w.writerow(["audio", "text", "speaker_id", "duration", "dnsmos"])
        w.writerows(rows)


def resample(x: np.ndarray, sr: int, to: int) -> np.ndarray:
    if sr == to:
        return x
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(sr, to)
    return resample_poly(x, to // g, sr // g).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus_vox_sv.txt")
    ap.add_argument("--ref", required=True, help="the reference take (refs/vox_*_ref.wav)")
    ap.add_argument("--prompt-text", default=PROMPT_TEXT, help="exact words of the reference take")
    ap.add_argument("--out", required=True)
    ap.add_argument("--speaker", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = whole corpus")
    ap.add_argument("--start", type=int, default=0, help="first corpus line to use")
    ap.add_argument("--instruction", default="", help="style, e.g. 'whispering softly'")
    ap.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--offset", type=int, default=0)
    args = ap.parse_args()

    from voxcpm import VoxCPM

    lines = [l.strip() for l in open(args.corpus, encoding="utf-8") if l.strip()]
    lines = lines[args.start:]
    if args.limit:
        lines = lines[: args.limit]
    wavs_dir = os.path.join(args.out, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)
    manifest = os.path.join(args.out, "speakers.csv" if args.stride == 1
                            else f"speakers_w{args.offset}.csv")

    kw = {} if args.device == "auto" else {"device": args.device}
    if "optimize" in inspect.signature(VoxCPM.from_pretrained).parameters and args.device == "mps":
        kw["optimize"] = False
    print(f"loading VoxCPM2 ({args.device}) ...", flush=True)
    model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False, **kw)
    sr = model.tts_model.sample_rate
    print(f"ref={args.ref}  speaker={args.speaker}  target={len(lines)} clips", flush=True)

    rows: list[tuple] = []
    t0 = time.time()
    new, audio_sec = 0, 0.0
    for i, text in enumerate(lines):
        if i % args.stride != args.offset:
            continue
        wid = f"{i:06d}"
        path = os.path.join(wavs_dir, f"{wid}.wav")
        if os.path.exists(path):
            dur = sf.info(path).duration
        else:
            try:
                if args.instruction:
                    wav = model.generate(text=f"({args.instruction}){text}", reference_wav_path=args.ref,
                                         cfg_value=args.cfg, inference_timesteps=args.steps)
                else:
                    wav = model.generate(text=text, reference_wav_path=args.ref,
                                         prompt_wav_path=args.ref, prompt_text=args.prompt_text,
                                         cfg_value=args.cfg, inference_timesteps=args.steps)
            except Exception as e:  # noqa: BLE001
                print(f"skip {wid}: {e}", flush=True)
                continue
            wav = resample(np.asarray(wav, dtype=np.float32).reshape(-1), sr, OUT_SR)
            sf.write(path, wav, OUT_SR, subtype="PCM_16")
            dur = len(wav) / OUT_SR
            new += 1
        audio_sec += dur
        rows.append((f"wavs/{wid}.wav", text, args.speaker, f"{dur:.3f}", "4.0"))
        if new and new % 10 == 0:
            el = time.time() - t0
            rate = new / el * 60 if el else 0
            eta = (len(lines) - i - 1) / rate if rate else 0
            print(f"[{i+1}/{len(lines)}] +{new} new | {rate:.1f} clips/min | "
                  f"{audio_sec/3600:.2f}h audio | ETA {eta:.0f} min", flush=True)
            write_manifest(manifest, rows)

    write_manifest(manifest, rows)
    print(f"DONE: {len(rows)} clips, {audio_sec/3600:.2f}h -> {manifest}", flush=True)


if __name__ == "__main__":
    main()
