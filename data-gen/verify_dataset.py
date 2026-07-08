"""QC pass over a synthesized dataset: ASR/CER + DNSMOS + rate/tail outliers.

For every clip in <dataset>/speakers.csv:
  1. ASR with KBLab/wav2vec2-large-voxrex-swedish -> char error rate vs intended text
     (catches Chatterbox hallucinations, truncations, repeated words)
  2. DNSMOS P.835 overall MOS (real score to replace the placeholder 4.0)
  3. speech-rate (chars/sec) and trailing-silence outliers

Incremental: results append to <dataset>/qc_report.csv; already-scored clips are
skipped, so it can run alongside/after generation and be re-run cheaply.

    .venv/bin/python verify_dataset.py --dataset dataset_female_018
    .venv/bin/python verify_dataset.py --dataset dataset_female_018 --write-clean

--write-clean writes <dataset>/speakers_qc.csv: passing rows only, with the real
DNSMOS in the dnsmos column (prepare_data.py consumes it directly). Failing wavs
are listed in <dataset>/qc_failed.txt — delete them and re-run batch_synth.py to
regenerate (it skips existing files).
"""
from __future__ import annotations

import argparse
import csv
import os
import re

import numpy as np
import soundfile as sf
import torch

CER_FAIL = 0.15          # char error rate above this -> bad clip
DNSMOS_FAIL = 2.6        # overall MOS below this -> bad clip (Chatterbox output centers ~3.0)
RATE_RANGE = (6.0, 30.0) # chars/sec sanity window
TAIL_FAIL = 2.0          # >2 s trailing near-silence -> bad clip
ASR_SR = 16000

# raw->MOS polynomial mapping from microsoft/DNS-Challenge dnsmos_local.py
POLY_OVR = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
DNSMOS_WIN = int(9.01 * ASR_SR)


def norm_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-zåäöéü ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def cer(ref: str, hyp: str) -> float:
    if not ref:
        return 1.0 if hyp else 0.0
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, hc in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc))
        prev = cur
    return prev[-1] / len(ref)


def tail_silence_sec(x: np.ndarray, sr: int, thresh_db: float = -40.0) -> float:
    if not len(x):
        return 0.0
    env = np.abs(x)
    peak = env.max() + 1e-9
    above = np.nonzero(env > peak * (10 ** (thresh_db / 20)))[0]
    return (len(x) - 1 - above[-1]) / sr if len(above) else len(x) / sr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0, help="max NEW clips to score (0 = all)")
    ap.add_argument("--write-clean", action="store_true",
                    help="write speakers_qc.csv with passing rows + real DNSMOS")
    args = ap.parse_args()

    ds = args.dataset
    report_path = os.path.join(ds, "qc_report.csv")
    done: dict[str, dict] = {}
    if os.path.exists(report_path):
        with open(report_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="|"):
                done[row["audio"]] = row

    with open(os.path.join(ds, "speakers.csv"), newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="|"))
    todo = [r for r in rows if r["audio"] not in done
            and os.path.exists(os.path.join(ds, r["audio"]))]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{ds}: {len(rows)} rows, {len(done)} scored, {len(todo)} to score", flush=True)

    if todo:
        import librosa
        import onnxruntime as ort
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        print("loading VoxRex ASR + DNSMOS ...", flush=True)
        processor = Wav2Vec2Processor.from_pretrained("KBLab/wav2vec2-large-voxrex-swedish")
        asr = Wav2Vec2ForCTC.from_pretrained("KBLab/wav2vec2-large-voxrex-swedish").to(args.device).eval()
        dns = ort.InferenceSession(os.path.join(os.path.dirname(__file__), "qc", "sig_bak_ovr.onnx"),
                                   providers=["CPUExecutionProvider"])

        new_file = not os.path.exists(report_path)
        with open(report_path, "a", newline="", encoding="utf-8") as rep:
            w = csv.writer(rep, delimiter="|")
            if new_file:
                w.writerow(["audio", "cer", "dnsmos", "rate", "tail", "flags", "asr_text"])
            for k, r in enumerate(todo):
                x, sr = sf.read(os.path.join(ds, r["audio"]), dtype="float32")
                if x.ndim > 1:
                    x = x.mean(axis=1)
                x16 = librosa.resample(x, orig_sr=sr, target_sr=ASR_SR) if sr != ASR_SR else x

                inputs = processor(x16, sampling_rate=ASR_SR, return_tensors="pt")
                with torch.no_grad():
                    logits = asr(inputs.input_values.to(args.device)).logits
                hyp = processor.batch_decode(torch.argmax(logits, dim=-1))[0]
                c = cer(norm_text(r["text"]), norm_text(hyp))

                # DNSMOS over 9 s windows (repeat-pad short clips), mean OVRL
                seg = np.tile(x16, int(np.ceil(DNSMOS_WIN / max(len(x16), 1))))[:DNSMOS_WIN] \
                    if len(x16) < DNSMOS_WIN else None
                scores = []
                for start in range(0, max(len(x16) - DNSMOS_WIN, 0) + 1, DNSMOS_WIN // 2) if seg is None else [0]:
                    win = x16[start:start + DNSMOS_WIN] if seg is None else seg
                    raw = dns.run(None, {"input_1": win[None].astype(np.float32)})[0][0]
                    scores.append(POLY_OVR(raw[2]))
                mos = float(np.mean(scores))

                dur = len(x) / sr
                rate = len(norm_text(r["text"])) / dur if dur else 0
                tail = tail_silence_sec(x, sr)
                flags = []
                if c > CER_FAIL: flags.append("cer")
                if mos < DNSMOS_FAIL: flags.append("dnsmos")
                if not RATE_RANGE[0] <= rate <= RATE_RANGE[1]: flags.append("rate")
                if tail > TAIL_FAIL: flags.append("tail")
                w.writerow([r["audio"], f"{c:.3f}", f"{mos:.2f}", f"{rate:.1f}",
                            f"{tail:.2f}", "+".join(flags), hyp])
                done[r["audio"]] = {"audio": r["audio"], "cer": f"{c:.3f}", "dnsmos": f"{mos:.2f}",
                                    "rate": f"{rate:.1f}", "tail": f"{tail:.2f}",
                                    "flags": "+".join(flags), "asr_text": hyp}
                if (k + 1) % 25 == 0:
                    rep.flush()
                    print(f"[{k+1}/{len(todo)}] scored", flush=True)

    # recompute flags from stored metrics so threshold changes apply without re-scoring
    def reflag(d):
        flags = []
        if float(d["cer"]) > CER_FAIL: flags.append("cer")
        if float(d["dnsmos"]) < DNSMOS_FAIL: flags.append("dnsmos")
        if not RATE_RANGE[0] <= float(d["rate"]) <= RATE_RANGE[1]: flags.append("rate")
        if float(d["tail"]) > TAIL_FAIL: flags.append("tail")
        d["flags"] = "+".join(flags)
        return d

    scored = [reflag(done[r["audio"]]) for r in rows if r["audio"] in done]
    bad = [d for d in scored if d["flags"]]
    print(f"\nscored {len(scored)}/{len(rows)} | failing {len(bad)} "
          f"({100*len(bad)/max(len(scored),1):.1f}%)", flush=True)
    for reason in ("cer", "dnsmos", "rate", "tail"):
        n = sum(1 for d in bad if reason in d["flags"].split("+"))
        print(f"  {reason:7s} {n}")

    with open(os.path.join(ds, "qc_failed.txt"), "w", encoding="utf-8") as f:
        for d in bad:
            f.write(f"{d['audio']}  [{d['flags']}] cer={d['cer']} mos={d['dnsmos']}\n")

    if args.write_clean:
        out = os.path.join(ds, "speakers_qc.csv")
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter="|")
            w.writerow(["audio", "text", "speaker_id", "duration", "dnsmos"])
            kept = 0
            for r in rows:
                d = done.get(r["audio"])
                if d is None or d["flags"]:
                    continue
                w.writerow([r["audio"], r["text"], r["speaker_id"], r["duration"], d["dnsmos"]])
                kept += 1
        print(f"wrote {out}: {kept} passing rows")


if __name__ == "__main__":
    main()
