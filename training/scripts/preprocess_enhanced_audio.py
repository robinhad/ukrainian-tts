#!/usr/bin/env python3
"""Create DeepFilterNet3, mastered, two-pass EBU R128 model copies."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from training.audio_enhancement.pipeline import (
    EnhancementConfig,
    EnhancedAudioProcessor,
)
from training.audio_enhancement.postprocess import load_config


def json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-records", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--postprocess-config", type=Path)
    parser.add_argument("--ffmpeg", default="auto")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--max-new-records",
        type=int,
        help="Exit with status 75 after this many new records so native state can be recycled.",
    )
    parser.add_argument("--resume", action="store_true")
    retry_group = parser.add_mutually_exclusive_group()
    retry_group.add_argument(
        "--retry-failures",
        dest="retry_failures",
        action="store_true",
        help="Retry failed records until they reach the attempt limit.",
    )
    retry_group.add_argument(
        "--no-retry-failures",
        dest="retry_failures",
        action="store_false",
        help="Keep failed records without another attempt.",
    )
    parser.set_defaults(retry_failures=True)
    parser.add_argument(
        "--maximum-attempts",
        type=int,
        default=2,
        help="Maximum enhancement attempts for one record.",
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Record rejected files and return success after the shard completes.",
    )
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("the shard index must be inside the shard count")
    if args.max_new_records is not None and args.max_new_records < 1:
        parser.error("--max-new-records must be positive")
    if args.maximum_attempts < 1:
        parser.error("--maximum-attempts must be positive")

    frame = pd.read_parquet(args.manifest).sort_values("utterance_id")
    rows = frame.to_dict(orient="records")[args.shard_index :: args.num_shards]
    if args.limit is not None:
        rows = rows[: args.limit]
    config = replace(
        EnhancementConfig(),
        final_postprocess=load_config(args.postprocess_config),
    )
    processor = EnhancedAudioProcessor(
        config,
        args.model_cache,
        ffmpeg_binary=args.ffmpeg,
    )
    args.output_records.parent.mkdir(parents=True, exist_ok=True)

    completed = 0
    failed = 0
    started = time.monotonic()
    mode = "w"
    existing = set()
    prior_terminal: list[dict] = []
    prior_attempts: dict[str, int] = {}
    if args.resume and args.output_records.exists():
        for line in args.output_records.read_text(encoding="utf-8").splitlines():
            prior = json.loads(line)
            identifier = str(prior["utterance_id"])
            attempts = int(
                prior.get(
                    "enhancement_attempts",
                    1 if prior.get("enhancement_status") == "failed" else 0,
                )
            )
            prior_attempts[identifier] = attempts
            retryable = (
                args.retry_failures
                and prior.get("enhancement_status") == "failed"
                and attempts < args.maximum_attempts
            )
            if not retryable:
                existing.add(prior["utterance_id"])
                prior_terminal.append(prior)
    pending = [row for row in rows if str(row["utterance_id"]) not in existing]
    has_more = args.max_new_records is not None and len(pending) > args.max_new_records
    if args.max_new_records is not None:
        pending = pending[: args.max_new_records]
    with args.output_records.open(mode, encoding="utf-8") as stream:
        for prior in prior_terminal:
            stream.write(json.dumps(prior, ensure_ascii=False, sort_keys=True) + "\n")
        for row in pending:
            utterance_id = str(row["utterance_id"])
            if utterance_id in existing:
                continue
            target = args.output_root / f"{utterance_id}.wav"
            record = dict(row)
            record["enhancement_attempts"] = prior_attempts.get(utterance_id, 0) + 1
            record["source_qc_flags"] = record.get("qc_flags")
            record["canonical_raw_audio_path"] = str(Path(row["audio_path"]).resolve())
            try:
                if target.is_file():
                    try:
                        metrics = processor.inspect_existing_output(target)
                        record["enhancement_recovered_from_existing_output"] = True
                    except Exception:
                        metrics = processor.process(
                            Path(record["canonical_raw_audio_path"]), target
                        )
                else:
                    metrics = processor.process(
                        Path(record["canonical_raw_audio_path"]), target
                    )
                record.update(metrics)
                record["audio_path"] = str(target.resolve())
                record["qc_flags"] = []
                record["enhancement_status"] = "ok"
                completed += 1
            except Exception as error:
                record["enhancement_status"] = "failed"
                record["enhancement_error"] = f"{type(error).__name__}: {error}"
                record["qc_flags"] = [f"enhancement_failed:{type(error).__name__}"]
                failed += 1
            stream.write(
                json.dumps(
                    record,
                    default=json_default,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            stream.flush()
            total = completed + failed
            elapsed = max(time.monotonic() - started, 0.001)
            print(
                json.dumps(
                    {
                        "completed": completed,
                        "failed": failed,
                        "rate_files_per_second": total / elapsed,
                        "shard": args.shard_index,
                        "total_in_shard": len(rows),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if failed and not args.allow_failures:
        return 1
    return 75 if has_more else 0


if __name__ == "__main__":
    raise SystemExit(main())
