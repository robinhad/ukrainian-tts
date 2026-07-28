#!/usr/bin/env python3
"""Split deferred long speech and create filtered pseudo-label records."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from training.frontend.sanitize import sanitize_text
from training.scripts.process_unlabeled import (
    append_jsonl,
    hypothesis_confidence,
    hypothesis_language,
    load_asr_model,
    ukrainian_letter_ratio,
)
from training.scripts.source_policy import load_registry


def strict_low_energy_split(
    audio: np.ndarray,
    sample_rate: int,
    *,
    minimum_seconds: float,
    maximum_seconds: float,
) -> list[tuple[float, float]]:
    """Split audio at low energy and keep every part within the limit."""
    if sample_rate <= 0:
        raise ValueError("The sample rate must be positive.")
    if not 0 < minimum_seconds <= maximum_seconds:
        raise ValueError("The duration limits are invalid.")
    end = len(audio) / sample_rate
    if end <= maximum_seconds:
        return [(0.0, end)]
    parts: list[tuple[float, float]] = []
    cursor = 0.0
    while end - cursor > maximum_seconds:
        latest = min(cursor + maximum_seconds, end - minimum_seconds)
        earliest = max(cursor + minimum_seconds, latest - 1.0)
        if latest <= earliest:
            cut = latest
        else:
            frame = max(1, round(sample_rate * 0.04))
            first = max(0, round(earliest * sample_rate))
            last = min(
                len(audio) - frame,
                max(first, round(latest * sample_rate) - frame),
            )
            starts = np.arange(first, last + 1, frame, dtype=np.int64)
            energy = np.asarray(
                [
                    float(np.mean(np.square(audio[index : index + frame])))
                    for index in starts
                ]
            )
            cut = float(starts[int(np.argmin(energy))]) / sample_rate
        parts.append((cursor, cut))
        cursor = cut
    parts.append((cursor, end))
    return parts


def progress_key(row: dict[str, Any]) -> str:
    return str(row["utterance_id"])


def load_progress(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            result[str(item["source_utterance_id"])] = item
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-records", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--minimum-seconds", type=float, default=2.0)
    parser.add_argument("--maximum-seconds", type=float, default=20.0)
    args = parser.parse_args()
    if args.num_shards < 1:
        parser.error("--num-shards must be positive.")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index is outside the shard range.")
    if not 0 < args.minimum_seconds <= args.maximum_seconds:
        parser.error("The duration limits are invalid.")

    registry = load_registry(args.registry)
    asr_cfg = registry["models"]["asr"]
    rows = sorted(
        (
            json.loads(line)
            for line in args.records.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ),
        key=progress_key,
    )
    rows = [
        row
        for index, row in enumerate(rows)
        if index % args.num_shards == args.shard_index
    ]
    progress = load_progress(args.progress)
    accepted_count = sum(
        int(item.get("accepted_parts", 0)) for item in progress.values()
    )
    rejected: Counter[str] = Counter()
    for item in progress.values():
        rejected.update(item.get("rejected", {}))

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.output_records.parent.mkdir(parents=True, exist_ok=True)
    args.output_records.touch(exist_ok=True)
    asr_model = load_asr_model(registry, args.model_cache)

    for row in rows:
        key = progress_key(row)
        if key in progress:
            continue
        audio, sample_rate = sf.read(
            row["audio_path"],
            dtype="float32",
            always_2d=True,
        )
        mono = np.mean(audio, axis=1)
        accepted_rows: list[dict[str, Any]] = []
        source_rejected: Counter[str] = Counter()
        parts = strict_low_energy_split(
            mono,
            sample_rate,
            minimum_seconds=args.minimum_seconds,
            maximum_seconds=args.maximum_seconds,
        )
        for part_index, (start, end) in enumerate(parts):
            duration = end - start
            if duration < args.minimum_seconds:
                source_rejected["duration_too_short"] += 1
                continue
            if duration > args.maximum_seconds + 1e-6:
                raise RuntimeError(
                    f"The strict splitter produced {duration:.6f} seconds."
                )
            samples = mono[
                round(start * sample_rate) : round(end * sample_rate)
            ]
            material = f"{key}:{part_index}:{start:.6f}:{end:.6f}"
            identifier = (
                f"{row['source_id']}_long_"
                f"{hashlib.sha256(material.encode()).hexdigest()[:20]}"
            )
            target = args.output_root / f"{identifier}.flac"
            sf.write(target, samples, sample_rate, subtype="PCM_16")
            hypothesis = asr_model.transcribe(
                [str(target)],
                timestamps=True,
                return_hypotheses=True,
            )[0]
            text = sanitize_text(str(hypothesis.text))
            language = hypothesis_language(hypothesis)
            confidence = hypothesis_confidence(hypothesis)
            script_ratio = ukrainian_letter_ratio(text)
            if (
                language is not None
                and language != asr_cfg["accepted_language"]
            ):
                source_rejected["language"] += 1
                target.unlink(missing_ok=True)
                continue
            if confidence is None or confidence < float(
                asr_cfg["minimum_mean_confidence"]
            ):
                source_rejected["confidence"] += 1
                target.unlink(missing_ok=True)
                continue
            if script_ratio < float(asr_cfg["minimum_ukrainian_letter_ratio"]):
                source_rejected["script_ratio"] += 1
                target.unlink(missing_ok=True)
                continue
            audio_hash = hashlib.sha256(target.read_bytes()).hexdigest()
            accepted_rows.append(
                {
                    **row,
                    "utterance_id": identifier,
                    "audio_path": str(target.resolve()),
                    "canonical_raw_audio_path": str(target.resolve()),
                    "text_raw": text,
                    "duration_source": duration,
                    "duration": duration,
                    "sample_rate": sample_rate,
                    "source_group": (
                        f"{row['source_group']}:long-part-{part_index:03d}"
                    ),
                    "parent_audio_sha256_source": (
                        row["audio_sha256_source"]
                    ),
                    "audio_sha256_source": audio_hash,
                    "text_sha256_source": hashlib.sha256(
                        text.encode("utf-8")
                    ).hexdigest(),
                    "label_kind": "pseudo",
                    "processing_status": "recovered_long",
                    "deferred_reason": None,
                    "asr_model": (
                        f"{asr_cfg['repo_id']}@{asr_cfg['revision']}"
                    ),
                    "asr_language": language or "uk_script_inferred",
                    "asr_mean_confidence": confidence,
                    "long_parent_utterance_id": key,
                    "long_part_index": part_index,
                    "long_part_start_seconds": start,
                    "long_part_end_seconds": end,
                }
            )
        append_jsonl(args.output_records, accepted_rows)
        item = {
            "source_utterance_id": key,
            "parts": len(parts),
            "accepted_parts": len(accepted_rows),
            "rejected": dict(sorted(source_rejected.items())),
        }
        append_jsonl(args.progress, [item])
        progress[key] = item
        accepted_count += len(accepted_rows)
        rejected.update(source_rejected)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report = {
        "status": "PASS",
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "source_records": len(rows),
        "processed_source_records": len(progress),
        "accepted_parts": accepted_count,
        "rejected": dict(sorted(rejected.items())),
        "output_records": str(args.output_records),
        "progress": str(args.progress),
        "minimum_seconds": args.minimum_seconds,
        "maximum_seconds": args.maximum_seconds,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    torch.set_grad_enabled(False)
    raise SystemExit(main())
