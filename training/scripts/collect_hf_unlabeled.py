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
from training.scripts.cleanup_unlabeled_cache import sweep_deferred_batches
from training.scripts.source_policy import (
    check_disk,
    load_registry,
    read_hf_token,
)


def ensure_disk(
    output_root: Path,
    source_id: str,
    registry: dict,
) -> None:
    disk = check_disk(output_root, registry)
    if disk["status"] != "STOP":
        return
    trigger_gib = float(registry["policy"]["free_disk_stop_gib"])
    sweep_deferred_batches(
        output_root / source_id / "markers",
        output_root,
        trigger_gib,
    )
    if check_disk(output_root, registry)["status"] == "STOP":
        raise SystemExit(
            f"Free disk space is below {trigger_gib:g} GiB. "
            "No processed cache can be removed."
        )


def existing_collection(
    records_path: Path,
    report_path: Path,
    source_id: str,
    shard_start: int,
    shard_count: int | None,
) -> dict | None:
    """Return a complete existing collection that is safe to resume."""
    if shard_count is None or not records_path.is_file() or not report_path.is_file():
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        records = [
            json.loads(line)
            for line in records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError):
        return None
    if (
        report.get("status") != "PASS"
        or report.get("source_id") != source_id
        or int(report.get("shard_start", -1)) != shard_start
        or int(report.get("shard_count", -1)) != shard_count
        or int(report.get("records", -1)) != len(records)
        or not records
    ):
        return None
    if any(not Path(record.get("audio_path", "")).is_file() for record in records):
        return None
    return report


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
    if args.shard_start < 0:
        parser.error("--shard-start must not be negative.")
    if args.shard_count is not None and args.shard_count < 1:
        parser.error("--shard-count must be positive.")
    ensure_disk(args.output_root, args.source_id, registry)

    source_root = args.output_root / args.source_id
    output = source_root / args.records_name
    report_path = source_root / args.report_name
    if args.limit is None:
        resumed = existing_collection(
            output,
            report_path,
            args.source_id,
            args.shard_start,
            args.shard_count,
        )
        if resumed is not None:
            print(
                json.dumps(
                    {**resumed, "resumed_existing_collection": True},
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

    token = read_hf_token()
    files = parquet_files(
        source["repo_id"],
        source["revision"],
        token,
        args.maximum_shards,
        source.get("subset"),
    )
    stop = (
        args.shard_start + args.shard_count
        if args.shard_count is not None
        else None
    )
    files = files[args.shard_start:stop]
    if not files:
        raise SystemExit("The source has no accessible Parquet shards.")
    batch_name = Path(args.records_name).stem
    raw_dir = source_root / "unlabeled_raw_16k" / batch_name
    records = []
    cache_artifacts = []
    excluded: dict[str, int] = {}
    seen_hashes: set[str] = set()
    for filename in files:
        ensure_disk(args.output_root, args.source_id, registry)
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
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
