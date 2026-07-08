"""Common manifest format — one JSON object per line (JSONL).

Every dataset is converted to this shape before anything else touches it:

    {
      "audio_path": "path/to/audio.wav",
      "text": "Normaliserad svensk text.",
      "speaker_id": "speaker_001",
      "dataset": "nst",
      "duration_sec": 4.2,
      "sample_rate": 24000,
      "quality_score": 0.91,          # filled by quality_filter (0..1, DNSMOS-scaled)
      "prosody": { ... }              # filled by prosody.extract (optional)
    }
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator


@dataclass
class ManifestEntry:
    audio_path: str
    text: str
    speaker_id: str
    dataset: str
    duration_sec: float
    sample_rate: int
    quality_score: float | None = None
    prosody: dict | None = None
    extra: dict = field(default_factory=dict)

    def to_json(self) -> str:
        d = asdict(self)
        if not d["extra"]:
            d.pop("extra")
        return json.dumps(d, ensure_ascii=False)


def write_manifest(entries, path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write((e.to_json() if isinstance(e, ManifestEntry) else json.dumps(e, ensure_ascii=False)) + "\n")
            n += 1
    return n


def read_manifest(path: str | Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def manifest_stats(path: str | Path) -> dict:
    n, hours, speakers, datasets = 0, 0.0, set(), set()
    for e in read_manifest(path):
        n += 1
        hours += e["duration_sec"] / 3600
        speakers.add(e["speaker_id"])
        datasets.add(e["dataset"])
    return {"entries": n, "hours": round(hours, 2),
            "speakers": len(speakers), "datasets": sorted(datasets)}
