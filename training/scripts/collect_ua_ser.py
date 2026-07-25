#!/usr/bin/env python3
"""Collect the human transcripts and speaker IDs from UA-SER."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from training.frontend.sanitize import sanitize_text
from training.scripts.source_policy import load_registry, read_hf_token


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    registry = load_registry(args.registry)
    source = registry["sources"]["ua_ser"]
    token = read_hf_token()
    metadata = Path(
        hf_hub_download(
            source["repo_id"],
            "dataset.csv",
            repo_type="dataset",
            revision=source["revision"],
            token=token,
        )
    )
    rows = []
    with metadata.open(encoding="utf-8", newline="") as stream:
        for index, item in enumerate(csv.DictReader(stream)):
            if args.limit is not None and index >= args.limit:
                break
            filename = str(item["filename"])
            path = Path(
                hf_hub_download(
                    source["repo_id"],
                    f"clips/{filename}",
                    repo_type="dataset",
                    revision=source["revision"],
                    token=token,
                )
            )
            content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            text = str(item.get("text") or "").strip()
            sanitized = sanitize_text(text)
            if not sanitized:
                continue
            speaker = str(item.get("speaker_id") or Path(filename).stem)
            identifier = f"ua_ser_{Path(filename).stem}_{content_hash[:12]}"
            rows.append(
                {
                    "utterance_id": identifier,
                    "speaker_id": identifier,
                    "speaker_stratum_id": f"ua_ser:{speaker}",
                    "source_id": "ua_ser",
                    "source": (
                        f"hf://datasets/{source['repo_id']}@{source['revision']}"
                    ),
                    "source_license": source["license"],
                    "license_evidence": (
                        f"hf://datasets/{source['repo_id']}@"
                        f"{source['revision']}/README.md"
                    ),
                    "audio_path": str(path.resolve()),
                    "canonical_raw_audio_path": str(path.resolve()),
                    "source_audio_path": f"clips/{filename}",
                    "audio_sha256_source": content_hash,
                    "text_raw": text,
                    "text_sha256_source": hashlib.sha256(
                        sanitized.encode("utf-8")
                    ).hexdigest(),
                    "source_group": f"ua_ser:{speaker}",
                    "source_original_split": str(item.get("split") or "unknown"),
                    "speaker_embedding_mode": "utterance",
                    "speaker_embedding_model": (
                        "speechbrain/spkrec-ecapa-voxceleb"
                        "@0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"
                    ),
                    "speaker_identity_status": "ua_ser_speaker_id",
                    "label_kind": "human",
                }
            )
    output = args.output_root / "ua_ser" / "records.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "records": len(rows), "artifact": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
