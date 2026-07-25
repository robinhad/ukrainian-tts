#!/usr/bin/env python3
"""Validate a curated CSV for sources that have per-file licenses."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from training.scripts.source_policy import load_registry, validate_record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    registry = load_registry(args.registry)
    source = registry["sources"].get(args.source_id)
    if not source or source.get("decision") != "allow_per_file":
        parser.error("The source must use per-file license review.")
    records = []
    with args.input.open(encoding="utf-8", newline="") as stream:
        for index, row in enumerate(csv.DictReader(stream), 2):
            audio = Path(row["audio_path"]).resolve()
            if not audio.is_file():
                raise SystemExit(f"Line {index}: audio does not exist: {audio}")
            text = str(row.get("text_raw") or "").strip()
            record = {
                "audio_path": str(audio),
                "source_id": args.source_id,
                "source": row["source_url"],
                "source_license": row["source_license"],
                "license_evidence": row["license_evidence"],
                "source_audio_path": row["source_url"],
                "source_group": row.get("source_group") or row["utterance_id"],
                "audio_sha256_source": hashlib.sha256(audio.read_bytes()).hexdigest(),
                "source_speaker_hint": row.get("speaker_id") or None,
            }
            errors = validate_record(record, registry)
            if errors:
                raise SystemExit(f"Line {index}: {'; '.join(errors)}")
            if not text and source.get("kind") == "labeled":
                raise SystemExit(f"Line {index}: labeled audio has no text.")
            if source.get("kind") == "labeled":
                record.update(
                    {
                        "utterance_id": row["utterance_id"],
                        "speaker_id": row.get("speaker_id") or row["utterance_id"],
                        "speaker_stratum_id": (
                            row.get("speaker_id") or row["utterance_id"]
                        ),
                        "canonical_raw_audio_path": str(audio),
                        "text_raw": text,
                        "text_sha256_source": hashlib.sha256(text.encode()).hexdigest(),
                        "speaker_embedding_mode": "utterance",
                        "speaker_embedding_model": (
                            "speechbrain/spkrec-ecapa-voxceleb"
                            "@0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"
                        ),
                        "speaker_identity_status": "curated_per_file",
                        "label_kind": "human",
                    }
                )
            records.append(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "records": len(records), "artifact": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
