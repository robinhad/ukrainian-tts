#!/usr/bin/env python3
"""Export the pinned Lada source as expanded-v3 raw records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    frame = pd.read_parquet(args.manifest).sort_values("utterance_id")
    if args.limit is not None:
        frame = frame.head(args.limit)
    rows = []
    for source in frame.to_dict("records"):
        path = Path(source["audio_path"]).resolve()
        rows.append(
            {
                "utterance_id": str(source["utterance_id"]),
                "speaker_id": "lada",
                "speaker_stratum_id": "lada",
                "audio_path": str(path),
                "canonical_raw_audio_path": str(path),
                "text_raw": str(source["text_raw"]),
                "duration_source": float(source["duration"]),
                "duration": float(source["duration"]),
                "sample_rate": int(source["sample_rate"]),
                "source_id": "opentts_lada",
                "source": str(source["source"]),
                "source_license": "Apache-2.0",
                "source_audio_path": str(
                    source.get("source_audio_path") or source["audio_path"]
                ),
                "source_group": f"lada:{source['source_group']}",
                "source_original_split": str(source["split"]),
                "audio_sha256_source": str(source["audio_sha256"]),
                "text_sha256_source": str(source["text_sha256"]),
                "speaker_embedding_mode": "utterance",
                "speaker_embedding_model": (
                    "speechbrain/spkrec-ecapa-voxceleb"
                    "@0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"
                ),
                "speaker_identity_status": "explicit_lada",
                "label_kind": "human",
                "license_evidence": (
                    "hf://datasets/speech-uk/opentts-lada@"
                    "729289b58251da4a21ce85f8808dd908f28b0d7f/README.md"
                ),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(
        json.dumps(
            {
                "status": "PASS",
                "records": len(rows),
                "artifact": str(args.output),
                "sha256": digest,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
