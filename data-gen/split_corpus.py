"""Build phase-2 corpora: sentence-split the not-yet-usable part of corpus_sv_v2.

For each voice: lines beyond its generation progress, PLUS lines already generated
whose clip exceeded 12 s (those get dropped by prepare_data's duration filter, so
their text is re-added here, split into usable pieces).

Splitting: on sentence boundaries (.!?…), merging fragments <25 chars with their
neighbor; keep pieces 10..150 chars (~<=12 s of speech).

    .venv/bin/python split_corpus.py dataset_female_018 corpus_v3_female.txt
"""
import csv
import re
import sys


def split_line(text: str, max_len: int = 150, min_len: int = 25) -> list[str]:
    if len(text) <= max_len:
        return [text]
    parts = re.split(r"(?<=[.!?…])\s+", text)
    # merge short fragments forward
    merged: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if merged and (len(merged[-1]) < min_len or len(p) < min_len) \
                and len(merged[-1]) + 1 + len(p) <= max_len:
            merged[-1] += " " + p
        else:
            merged.append(p)
    out = []
    for m in merged:
        if len(m) <= max_len:
            if len(m) >= 10:
                out.append(m)
            continue
        # single overlong sentence: try comma/semicolon split once
        subs = re.split(r"(?<=[,;:])\s+", m)
        buf = ""
        for s in subs:
            if len(buf) + len(s) + 1 <= max_len:
                buf = (buf + " " + s).strip()
            else:
                if len(buf) >= 10:
                    out.append(buf)
                buf = s
        if 10 <= len(buf) <= max_len:
            out.append(buf)
        # still-overlong remainders are dropped
    return out


def main() -> None:
    dataset, out_path = sys.argv[1], sys.argv[2]
    corpus = [l.strip() for l in open("corpus_sv_v2.txt", encoding="utf-8") if l.strip()]

    done_short: set[str] = set()   # texts already generated at usable length
    max_idx = -1
    with open(f"{dataset}/speakers.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="|"):
            idx = int(r["audio"].split("/")[-1].split(".")[0])
            max_idx = max(max_idx, idx)
            if float(r["duration"]) <= 12.0:
                done_short.add(r["text"])

    todo: list[str] = []
    for i, line in enumerate(corpus):
        if i <= max_idx and line in done_short:
            continue  # already have a usable clip of exactly this text
        todo.extend(split_line(line))

    seen: set[str] = set(done_short)
    uniq = []
    for t in todo:
        if t not in seen:
            seen.add(t)
            uniq.append(t)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(uniq) + "\n")
    lens = [len(t) for t in uniq]
    print(f"{dataset}: progress idx {max_idx}, {len(done_short)} usable clips kept; "
          f"{out_path}: {len(uniq)} lines, avg {sum(lens)/len(lens):.0f} chars, max {max(lens)}")


if __name__ == "__main__":
    main()
