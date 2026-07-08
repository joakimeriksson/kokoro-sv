"""Convert raw datasets to the common manifest + 24 kHz mono WAVs.

Supported:
  nst     — streams HuggingFace KTH/nst parquet (no 44 GB download; column-
            projected reads, same approach as the swedish-kokoro audition)
  css10   — local CSS10 Swedish directory (download the sv zip from Kaggle:
            https://www.kaggle.com/datasets/bryanpark/swedish-single-speaker-speech-dataset
            and unzip into data/raw/css10_sv/)

    python scripts/prepare_dataset.py nst --max-hours 8 --out data/processed/nst
    python scripts/prepare_dataset.py css10 --raw data/raw/css10_sv --out data/processed/css10
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import soundfile as sf
from tqdm import tqdm

from kokoro_se.dataset_manifest import ManifestEntry, write_manifest
from kokoro_se.text_normalization import normalize_text_sv
from kokoro_se.audio_utils import rms_normalize, TARGET_SR
from kokoro_se.quality_filter import cheap_gates
import librosa


def _write_clip(x, sr, dst: Path) -> float:
    if sr != TARGET_SR:
        x = librosa.resample(x, orig_sr=sr, target_sr=TARGET_SR)
    x = rms_normalize(x)
    x = np.concatenate([x, np.zeros(int(TARGET_SR * 0.2), dtype="float32")])  # tail pad
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), x, TARGET_SR, subtype="PCM_16")
    return len(x) / TARGET_SR


def prepare_nst(out: Path, max_hours: float, gender: str | None = None,
                exclude_speakers: set[str] | None = None) -> list[ManifestEntry]:
    """gender: 'Male' / 'Female' / None (all). exclude_speakers: ids already used."""
    import fsspec
    import pyarrow.parquet as pq

    exclude_speakers = exclude_speakers or set()
    # KTH/nst gender is a ClassLabel: Male=0, Female=1, Unspecified=2
    gender_idx = {"Male": 0, "Female": 1}.get(gender) if gender else None
    # Prefer locally-cached shards (robust vs flaky HTTP range-reads); fall back to streaming.
    cache = Path(__file__).resolve().parents[1] / "data" / "raw" / "nst_parquet"
    local = sorted(cache.glob("*.parquet")) if cache.exists() else []
    sources = [str(p) for p in local] or \
        [f"https://huggingface.co/api/datasets/KTH/nst/parquet/default/train/{i}.parquet" for i in range(10)]
    fs = fsspec.filesystem("http", block_size=1 << 22)
    entries, hours = [], 0.0
    for src in sources:
        if hours >= max_hours:
            break
        f = open(src, "rb") if src.endswith(".parquet") and Path(src).exists() else fs.open(src)
        with f:
            pf = pq.ParquetFile(f)
            cols = ["speaker_id", "text", "audio"] + (["gender"] if gender else [])
            for rg in range(pf.num_row_groups):
                if hours >= max_hours:
                    break
                t = pf.read_row_group(rg, columns=cols)
                data = {c: t.column(c).to_pylist() for c in cols}
                for i in range(len(data["text"])):
                    if hours >= max_hours:
                        break
                    spk, text, audio = data["speaker_id"][i], data["text"][i], data["audio"][i]
                    if spk in exclude_speakers:
                        continue
                    if gender_idx is not None and data["gender"][i] != gender_idx:
                        continue
                    x, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
                    if x.ndim > 1:
                        x = x[:, 0]  # NST close mic
                    if cheap_gates(x, sr):
                        continue
                    norm = normalize_text_sv(text)
                    if len(norm) < 8:
                        continue
                    wav = out / "wavs" / f"nst_{spk}_{len(entries):06d}.wav"
                    dur = _write_clip(x, sr, wav)
                    entries.append(ManifestEntry(str(wav), norm, f"nst_{spk}", "nst",
                                                 round(dur, 2), TARGET_SR))
                    hours += dur / 3600
        print(f"  nst: {hours:.2f} h collected", flush=True)
    return entries


def prepare_tts_swedish(out: Path, max_hours: float) -> list[ManifestEntry]:
    """datadriven-company/TTS-Swedish: 40 h LibriVox, 9 speakers, CC0, with DNSMOS."""
    import fsspec
    import pyarrow.parquet as pq

    urls = [f"https://huggingface.co/datasets/datadriven-company/TTS-Swedish/resolve/main/data/train-{i:05d}-of-00015.parquet"
            for i in range(15)]
    fs = fsspec.filesystem("http", block_size=1 << 22)
    entries, hours = [], 0.0
    per_speaker: dict[str, float] = {}
    cap_per_speaker = max_hours * 3600 / 6  # balance: no speaker over ~1/6 of total
    for url in urls:
        if hours >= max_hours:
            break
        with fs.open(url) as f:
            pf = pq.ParquetFile(f)
            cols = [c for c in ("mp3", "text", "speaker_id", "dnsmos")
                    if c in pf.schema_arrow.names]
            for rg in range(pf.num_row_groups):
                if hours >= max_hours:
                    break
                t = pf.read_row_group(rg, columns=cols)
                rows = {c: t.column(c).to_pylist() for c in cols}
                for i in range(len(rows["text"])):
                    if hours >= max_hours:
                        break
                    spk = str(rows.get("speaker_id", ["libri"])[i])
                    if per_speaker.get(spk, 0.0) > cap_per_speaker:
                        continue
                    dns = rows.get("dnsmos", [None])[i]
                    if dns is not None and float(dns) < 3.6:
                        continue
                    audio = rows["mp3"][i]
                    x, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
                    if x.ndim > 1:
                        x = x.mean(axis=1)
                    if cheap_gates(x, sr):
                        continue
                    norm = normalize_text_sv(rows["text"][i])
                    if len(norm) < 8:
                        continue
                    wav = out / "wavs" / f"libri_{spk}_{len(entries):06d}.wav"
                    dur = _write_clip(x, sr, wav)
                    entries.append(ManifestEntry(str(wav), norm, f"libri_{spk}",
                                                 "tts_swedish", round(dur, 2), TARGET_SR,
                                                 quality_score=round(float(dns) / 5, 3) if dns else None))
                    hours += dur / 3600
                    per_speaker[spk] = per_speaker.get(spk, 0.0) + dur
        print(f"  tts_swedish: {hours:.2f} h, {len(per_speaker)} speakers", flush=True)
    return entries


def prepare_css10(raw: Path, out: Path, max_hours: float) -> list[ManifestEntry]:
    tx = raw / "transcript.txt"
    if not tx.exists():
        raise SystemExit(f"{tx} not found — download CSS10 sv from Kaggle and unzip into {raw}")
    entries, hours = [], 0.0
    lines = tx.read_text(encoding="utf-8").strip().splitlines()
    for line in tqdm(lines, desc="css10"):
        if hours >= max_hours:
            break
        parts = line.split("|")
        if len(parts) < 3:
            continue
        rel, text = parts[0], parts[2] if parts[2].strip() else parts[1]
        src = raw / rel
        if not src.exists():
            continue
        x, sr = sf.read(str(src), dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        if cheap_gates(x, sr):
            continue
        norm = normalize_text_sv(text)
        if len(norm) < 8:
            continue
        wav = out / "wavs" / f"css10_{len(entries):06d}.wav"
        dur = _write_clip(x, sr, wav)
        entries.append(ManifestEntry(str(wav), norm, "css10_sv_speaker", "css10_sv",
                                     round(dur, 2), TARGET_SR))
        hours += dur / 3600
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["nst", "css10", "tts_swedish"])
    ap.add_argument("--raw", default=None, help="raw dir (css10)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-hours", type=float, default=8.0)
    ap.add_argument("--gender", choices=["Male", "Female"], default=None,
                    help="nst only: keep only this gender")
    ap.add_argument("--exclude-manifest", default=None,
                    help="nst only: skip speaker_ids already present in this manifest")
    ap.add_argument("--name", default=None, help="manifest basename override")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    name = args.name or args.dataset
    out = Path(args.out) if args.out else root / "data" / "processed" / name
    if args.dataset == "nst":
        excl = set()
        if args.exclude_manifest:
            import json as _j
            for line in Path(args.exclude_manifest).open(encoding="utf-8"):
                excl.add(_j.loads(line)["speaker_id"].replace("nst_", ""))
        entries = prepare_nst(out, args.max_hours, gender=args.gender, exclude_speakers=excl)
    elif args.dataset == "tts_swedish":
        entries = prepare_tts_swedish(out, args.max_hours)
    else:
        entries = prepare_css10(Path(args.raw or root / "data/raw/css10_sv"), out, args.max_hours)

    mpath = root / "data" / "manifests" / f"{name}.jsonl"
    n = write_manifest(entries, mpath)
    print(f"wrote {n} entries -> {mpath}")


if __name__ == "__main__":
    main()
