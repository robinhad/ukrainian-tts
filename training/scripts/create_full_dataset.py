#!/usr/bin/env python3
"""Materialize and group-split the complete pinned Lada corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from training.frontend.sanitize import sanitize_text

DATASET = "speech-uk/opentts-lada"
REVISION = "729289b58251da4a21ce85f8808dd908f28b0d7f"
PARQUET_FILE = "data/train-00000-of-00001.parquet"
SEED = "777"


def source_group(source_path: str) -> str:
    """Group adjacent source segments when the dataset has no document column."""
    stem = Path(source_path).stem
    if stem.isdigit():
        return f"sequential_block_{int(stem) // 50:06d}"
    prefix = re.split(r"[_-]", stem, maxsplit=1)[0]
    return f"prefix_{prefix}"


def assign_groups(groups: dict[str, list[dict]]) -> dict[str, str]:
    total = sum(map(len, groups.values()))
    target = round(total * 0.02)
    ordered = sorted(
        groups,
        key=lambda group: hashlib.sha256(f"{SEED}\0{group}".encode()).hexdigest(),
    )
    assignment: dict[str, str] = {}
    cursor = 0
    for split in ("eval", "dev"):
        count = 0
        while cursor < len(ordered) and count < target:
            group = ordered[cursor]
            assignment[group] = split
            count += len(groups[group])
            cursor += 1
    for group in ordered[cursor:]:
        assignment[group] = "train"
    return assignment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--min-duration", type=float, default=2.0)
    parser.add_argument("--max-duration", type=float, default=12.0)
    args = parser.parse_args()

    parquet_path = hf_hub_download(
        repo_id=DATASET,
        repo_type="dataset",
        revision=REVISION,
        filename=PARQUET_FILE,
    )
    parquet = pq.ParquetFile(parquet_path)
    rows = []
    seen_audio: set[str] = set()
    seen_text: set[str] = set()
    excluded = defaultdict(int)
    for batch in parquet.iter_batches(
        batch_size=128, columns=["audio", "duration", "transcription"]
    ):
        for source in batch.to_pylist():
            text = str(source.get("transcription") or "").strip()
            duration = float(source.get("duration") or 0.0)
            audio = source.get("audio") or {}
            content = audio.get("bytes")
            source_path = str(audio.get("path") or "audio.ogg")
            if not text or content is None:
                excluded["missing_pair"] += 1
                continue
            if not args.min_duration <= duration <= args.max_duration:
                excluded["duration"] += 1
                continue
            audio_hash = hashlib.sha256(content).hexdigest()
            sanitized = sanitize_text(text)
            text_hash = hashlib.sha256(sanitized.encode("utf-8")).hexdigest()
            if audio_hash in seen_audio:
                excluded["duplicate_audio"] += 1
                continue
            if text_hash in seen_text:
                excluded["duplicate_text"] += 1
                continue
            seen_audio.add(audio_hash)
            seen_text.add(text_hash)
            identifier = f"lada_{Path(source_path).stem}_{audio_hash[:10]}"
            extension = Path(source_path).suffix or ".ogg"
            rows.append(
                {
                    "utterance_id": identifier,
                    "speaker_id": "lada",
                    "audio_path": "",
                    "text_raw": text,
                    "duration_source": duration,
                    "source": f"hf://datasets/{DATASET}@{REVISION}",
                    "source_license": "Apache-2.0",
                    "source_audio_path": source_path,
                    "source_group": source_group(source_path),
                    "audio_sha256_source": audio_hash,
                    "text_sha256_source": text_hash,
                    "_content": content,
                    "_extension": extension,
                }
            )

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["source_group"]].append(row)
    assignment = assign_groups(groups)
    raw_dir = args.output_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        path = raw_dir / f"{row['utterance_id']}{row.pop('_extension')}"
        path.write_bytes(row.pop("_content"))
        row["audio_path"] = str(path.resolve())
        row["split"] = assignment[row["source_group"]]

    rows.sort(key=lambda row: row["utterance_id"])
    output = args.output_root / "source_records.jsonl"
    with output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    counts = {split: sum(row["split"] == split for row in rows) for split in ("train", "dev", "eval")}
    print(
        json.dumps(
            {
                "artifact": str(output),
                "source_rows": parquet.metadata.num_rows,
                "selected": len(rows),
                "groups": len(groups),
                "counts": counts,
                "excluded": dict(sorted(excluded.items())),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
