"""Fold a phase-2 dataset (striped workers) into the main dataset.

Hardlinks <p2>/wavs/NNNNNN.wav into <main>/wavs/9NNNNN.wav (offset 900000, no
collision with corpus indices < 13489) and appends all speakers_w*.csv rows to
<main>/speakers.csv. Idempotent: already-merged files are skipped.

    .venv/bin/python merge_dataset.py dataset_female_018_p2 dataset_female_018
"""
import csv
import glob
import os
import sys

p2, main = sys.argv[1], sys.argv[2]
main_csv = os.path.join(main, "speakers.csv")

with open(main_csv, newline="", encoding="utf-8") as f:
    existing_rows = list(csv.reader(f, delimiter="|"))
have = {r[0] for r in existing_rows[1:]}

added = 0
for wcsv in sorted(glob.glob(os.path.join(p2, "speakers_w*.csv"))):
    with open(wcsv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="|"):
            idx = int(r["audio"].split("/")[-1].split(".")[0])
            new_rel = f"wavs/{900000 + idx:06d}.wav"
            if new_rel in have:
                continue
            src = os.path.join(p2, r["audio"])
            dst = os.path.join(main, new_rel)
            if not os.path.exists(src):
                continue
            if not os.path.exists(dst):
                os.link(src, dst)
            existing_rows.append([new_rel, r["text"], r["speaker_id"], r["duration"], r["dnsmos"]])
            have.add(new_rel)
            added += 1

with open(main_csv, "w", newline="", encoding="utf-8") as f:
    csv.writer(f, delimiter="|").writerows(existing_rows)
print(f"merged {added} phase-2 clips into {main} ({len(existing_rows)-1} rows total)")
