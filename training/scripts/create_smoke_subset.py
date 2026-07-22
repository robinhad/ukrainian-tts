#!/usr/bin/env python3
"""Materialize a deterministic, real Lada smoke subset from Hugging Face."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DATASET = "speech-uk/opentts-lada"
REVISION = "729289b58251da4a21ce85f8808dd908f28b0d7f"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train", type=int, default=256)
    parser.add_argument("--dev", type=int, default=32)
    parser.add_argument("--eval", type=int, default=32)
    parser.add_argument("--candidate-limit", type=int, default=1000)
    args = parser.parse_args()

    from datasets import Audio, load_dataset

    raw_dir = args.output_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(DATASET, split="train", revision=REVISION, streaming=True)
    dataset = dataset.cast_column("audio", Audio(decode=False))
    candidates = []
    for row in dataset:
        text = str(row.get("transcription") or "").strip()
        duration = float(row.get("duration") or 0.0)
        audio = row.get("audio") or {}
        content = audio.get("bytes")
        source_path = str(audio.get("path") or "audio.ogg")
        if not text or content is None or not 2.0 <= duration <= 12.0:
            continue
        audio_hash = hashlib.sha256(content).hexdigest()
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        identifier = f"lada_{Path(source_path).stem}_{audio_hash[:10]}"
        extension = Path(source_path).suffix or ".ogg"
        output_path = raw_dir / f"{identifier}{extension}"
        candidates.append(
            {
                "utterance_id": identifier,
                "speaker_id": "lada",
                "audio_path": str(output_path.resolve()),
                "text_raw": text,
                "duration_source": duration,
                "source": f"hf://datasets/{DATASET}@{REVISION}",
                "source_license": "Apache-2.0",
                "source_audio_path": source_path,
                "audio_sha256_source": audio_hash,
                "text_sha256_source": text_hash,
                "_content": content,
                "_sort": hashlib.sha256(f"777\0{audio_hash}\0{text_hash}".encode()).hexdigest(),
            }
        )
        if len(candidates) >= args.candidate_limit:
            break

    required = args.train + args.dev + args.eval
    if len(candidates) < required:
        raise RuntimeError(f"only {len(candidates)} eligible records found; need {required}")
    candidates.sort(key=lambda row: row["_sort"])
    selected = candidates[:required]
    split_limits = (("smoke_eval", args.eval), ("smoke_dev", args.dev), ("smoke_train", args.train))
    cursor = 0
    records = []
    for split, count in split_limits:
        for row in selected[cursor : cursor + count]:
            Path(row["audio_path"]).write_bytes(row.pop("_content"))
            row.pop("_sort")
            row["split"] = split
            records.append(row)
        cursor += count
    records.sort(key=lambda row: row["utterance_id"])
    output = args.output_root / "source_records.jsonl"
    with output.open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"artifact": str(output), "counts": {k: v for k, v in split_limits}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
