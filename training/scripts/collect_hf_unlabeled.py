#!/usr/bin/env python3
"""Collect audio from one permitted unlabeled Hugging Face source."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import soundfile as sf
from huggingface_hub import hf_hub_download

from training.scripts.ingest_hf_parquet import (
    canonicalize_audio,
    iter_rows,
    parquet_files,
)
from training.scripts.source_policy import (
    check_disk,
    load_registry,
    read_hf_token,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--maximum-shards", type=int)
    parser.add_argument("--shard-start", type=int, default=0)
    parser.add_argument("--shard-count", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--records-name", default="unlabeled_records.jsonl")
    parser.add_argument("--report-name", default="collect_report.json")
    args = parser.parse_args()

    registry = load_registry(args.registry)
    source = registry["sources"].get(args.source_id)
    if not source or source.get("connector") != "huggingface":
        parser.error("The source must be a Hugging Face source.")
    if not source.get("enabled") or source.get("decision") != "allow":
        parser.error("The source policy does not allow this source.")
    if source.get("kind") != "unlabeled":
        parser.error("The source must have the unlabeled kind.")
    if check_disk(args.output_root, registry)["status"] == "STOP":
        raise SystemExit("Free disk space is below the 60 GiB stop threshold.")

    token = read_hf_token()
    files = parquet_files(
        source["repo_id"],
        source["revision"],
        token,
        args.maximum_shards,
        source.get("subset"),
    )
    if args.shard_start < 0:
        parser.error("--shard-start must not be negative.")
    if args.shard_count is not None and args.shard_count < 1:
        parser.error("--shard-count must be positive.")
    stop = (
        args.shard_start + args.shard_count
        if args.shard_count is not None
        else None
    )
    files = files[args.shard_start:stop]
    if not files:
        raise SystemExit("The source has no accessible Parquet shards.")
    source_root = args.output_root / args.source_id
    batch_name = Path(args.records_name).stem
    raw_dir = source_root / "unlabeled_raw_16k" / batch_name
    records = []
    cache_artifacts = []
    excluded: dict[str, int] = {}
    seen_hashes: set[str] = set()
    for filename in files:
        if check_disk(args.output_root, registry)["status"] == "STOP":
            raise SystemExit("Free disk space is below the 60 GiB stop threshold.")
        shard: Path | str
        if filename.startswith(("https://", "http://")):
            shard = filename
        else:
            shard = Path(
                hf_hub_download(
                    repo_id=source["repo_id"],
                    repo_type="dataset",
                    revision=source["revision"],
                    filename=filename,
                    token=token,
                )
            )
            cache_artifacts.append(
                {
                    "snapshot_path": str(shard),
                    "blob_path": str(shard.resolve()),
                }
            )
        for row_index, row in enumerate(iter_rows(shard, token)):
            audio = row.get("audio") or {}
            content = audio.get("bytes") if isinstance(audio, dict) else None
            source_path = str(audio.get("path") or f"{row_index}.audio")
            if not content:
                excluded["missing_audio"] = excluded.get("missing_audio", 0) + 1
                continue
            source_hash = hashlib.sha256(content).hexdigest()
            if source_hash in seen_hashes:
                excluded["duplicate_audio"] = (
                    excluded.get("duplicate_audio", 0) + 1
                )
                continue
            seen_hashes.add(source_hash)
            identifier = f"{args.source_id}_source_{source_hash[:20]}"
            target = raw_dir / f"{identifier}.flac"
            if not target.is_file():
                canonicalize_audio(content, source_path, target)
            info = sf.info(target)
            if info.duration < 2.0:
                target.unlink(missing_ok=True)
                excluded["too_short"] = excluded.get("too_short", 0) + 1
                continue
            records.append(
                {
                    "source_id": args.source_id,
                    "source": (
                        f"hf://datasets/{source['repo_id']}@{source['revision']}"
                    ),
                    "source_license": source["license"],
                    "license_evidence": source.get("evidence"),
                    "audio_path": str(target.resolve()),
                    "source_audio_path": source_path,
                    "audio_sha256_source": source_hash,
                    "source_group": f"{args.source_id}:{Path(filename).stem}",
                    "source_speaker_hint": None,
                }
            )
            if args.limit is not None and len(records) >= args.limit:
                break
        if args.limit is not None and len(records) >= args.limit:
            break
    output = source_root / args.records_name
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    report = {
        "status": "PASS",
        "source_id": args.source_id,
        "records": len(records),
        "excluded": dict(sorted(excluded.items())),
        "artifact": str(output),
        "shard_start": args.shard_start,
        "shard_count": len(files),
        "cache_artifacts": cache_artifacts,
    }
    (source_root / args.report_name).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
