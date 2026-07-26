#!/usr/bin/env python3
"""Export the available Common Voice and Lada source records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-records", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    outputs = {
        "common_voice_available_uk": [],
        "opentts_lada": [],
    }
    for line in args.source_records.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        identifier = str(record["utterance_id"])
        if identifier.startswith("cv22_"):
            source_id = "common_voice_available_uk"
            record["speaker_stratum_id"] = identifier
            record["license_evidence"] = (
                "User selected the existing local Common Voice copy on 2026-07-26."
            )
        elif str(record.get("speaker_id")) == "lada":
            source_id = "opentts_lada"
            record["speaker_stratum_id"] = "lada"
            record["license_evidence"] = (
                "hf://datasets/speech-uk/opentts-lada@"
                "729289b58251da4a21ce85f8808dd908f28b0d7f/README.md"
            )
        else:
            continue
        record["source_id"] = source_id
        record["canonical_raw_audio_path"] = str(
            Path(record["audio_path"]).resolve()
        )
        record["duration"] = float(record["duration_source"])
        record["sample_rate"] = int(record.get("sample_rate") or 0)
        record["source_original_split"] = str(record.get("split") or "unknown")
        record["label_kind"] = "human"
        outputs[source_id].append(record)

    report = {}
    for source_id, rows in outputs.items():
        target = args.output_root / source_id / "records.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in sorted(rows, key=lambda item: item["utterance_id"])
            ),
            encoding="utf-8",
        )
        report[source_id] = {"records": len(rows), "artifact": str(target)}
    print(json.dumps({"status": "PASS", "sources": report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
