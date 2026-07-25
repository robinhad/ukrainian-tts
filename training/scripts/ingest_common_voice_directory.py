#!/usr/bin/env python3
"""Ingest a direct Mozilla Common Voice directory after the user downloads it."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import soundfile as sf

from training.frontend.sanitize import sanitize_text
from training.scripts.ingest_hf_parquet import canonicalize_audio
from training.scripts.source_policy import check_disk, load_registry, validate_record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    registry = load_registry(args.registry)
    source_id = "common_voice_26_uk"
    source = registry["sources"][source_id]
    metadata = args.source_root / "validated.tsv"
    clips = args.source_root / "clips"
    if not metadata.is_file() or not clips.is_dir():
        raise SystemExit("The Common Voice directory needs validated.tsv and clips/.")
    if check_disk(args.output_root, registry)["status"] == "STOP":
        raise SystemExit("Free disk space is below the 60 GiB stop threshold.")

    target_root = args.output_root / source_id
    raw_dir = target_root / "canonical_raw_16k"
    records = []
    rejected: dict[str, int] = {}
    with metadata.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if args.limit is not None and len(records) >= args.limit:
                break
            text = str(row.get("sentence") or "").strip()
            relative = str(row.get("path") or "").strip()
            audio = clips / relative
            sanitized = sanitize_text(text)
            if not sanitized or not audio.is_file():
                rejected["missing_pair"] = rejected.get("missing_pair", 0) + 1
                continue
            content = audio.read_bytes()
            source_hash = hashlib.sha256(content).hexdigest()
            identifier = f"common_voice_26_uk_{Path(relative).stem}_{source_hash[:12]}"
            target = raw_dir / f"{identifier}.flac"
            if not target.is_file():
                canonicalize_audio(content, relative, target)
            info = sf.info(target)
            if not 2.0 <= info.duration <= 12.0:
                target.unlink(missing_ok=True)
                rejected["duration"] = rejected.get("duration", 0) + 1
                continue
            client = str(row.get("client_id") or identifier)
            text_hash = hashlib.sha256(sanitized.encode()).hexdigest()
            record = {
                "utterance_id": identifier,
                "speaker_id": identifier,
                "speaker_stratum_id": f"{source_id}:{client}",
                "audio_path": str(target.resolve()),
                "canonical_raw_audio_path": str(target.resolve()),
                "text_raw": text,
                "duration_source": float(info.duration),
                "duration": float(info.duration),
                "sample_rate": int(info.samplerate),
                "source_id": source_id,
                "source": source["uri"],
                "source_license": source["license"],
                "license_evidence": source["uri"],
                "source_audio_path": relative,
                "source_group": f"{source_id}:{client}",
                "audio_sha256_source": source_hash,
                "text_sha256_source": text_hash,
                "speaker_embedding_mode": "utterance",
                "speaker_embedding_model": (
                    "speechbrain/spkrec-ecapa-voxceleb"
                    "@0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"
                ),
                "speaker_identity_status": "common_voice_client_id",
                "label_kind": "human",
            }
            errors = validate_record(record, registry)
            if errors:
                raise SystemExit("; ".join(errors))
            records.append(record)
    output = target_root / "records.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in records
        ),
        encoding="utf-8",
    )
    report = {
        "status": "PASS",
        "records": len(records),
        "rejected": dict(sorted(rejected.items())),
        "artifact": str(output),
    }
    (target_root / "ingest_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
