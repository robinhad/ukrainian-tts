#!/usr/bin/env python3
"""Create resumable Sidon and de-essing training WAV files."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from training.audio_enhancement.review_pipeline import (
    SidonDeessOnlyConfig,
    apply_deessing_only,
    inspect_wav,
    sha256,
    write_float_wav,
    write_json,
)
from training.audio_enhancement.postprocess import load_config
from training.scripts.run_enhancement_review_backend import SidonDeessOnlyBackend


def json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def load_prior(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[str(row["utterance_id"])] = row
    return rows


def output_is_valid(
    target: Path,
    source: Path,
    config: SidonDeessOnlyConfig,
) -> bool:
    metadata_path = target.with_suffix(".wav.json")
    if not target.is_file() or target.is_symlink() or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return (
            metadata.get("status") == "PASS"
            and metadata.get("processing_config_hash") == config.digest
            and metadata.get("source") == str(source.resolve())
            and metadata.get("source_sha256") == sha256(source)
            and metadata.get("audio_sha256") == sha256(target)
            and not inspect_wav(target)["errors"]
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


def result_row(
    *,
    utterance_id: str,
    source: Path,
    target: Path,
    attempts: int,
    status: str,
    config: SidonDeessOnlyConfig,
    check: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "utterance_id": utterance_id,
        "source_audio_path": str(source.resolve()),
        "audio_path": str(target.resolve()),
        "processing_config_hash": config.digest,
        "processing_status": status,
        "processing_attempts": attempts,
    }
    if check is not None:
        row.update(
            {
                "audio_sha256": sha256(target),
                "duration": check["duration"],
                "sample_rate": check["sample_rate"],
                "channels": check["channels"],
                "format": check["format"],
            }
        )
    if error is not None:
        row["processing_error"] = error
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-audio-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-results", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--postprocess-config", type=Path)
    parser.add_argument("--ffmpeg", default="auto")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--maximum-attempts", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-failures", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("the shard index must be inside the shard count")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.maximum_attempts < 1:
        parser.error("--maximum-attempts must be positive")

    manifest = pd.read_parquet(args.manifest, columns=["utterance_id"])
    inputs = pd.read_parquet(
        args.input_audio_manifest,
        columns=["utterance_id", "audio_path"],
    )
    if manifest["utterance_id"].duplicated().any():
        raise SystemExit("The training manifest has duplicate utterance IDs.")
    if inputs["utterance_id"].duplicated().any():
        raise SystemExit("The input-audio manifest has duplicate utterance IDs.")
    frame = manifest.merge(
        inputs,
        on="utterance_id",
        how="left",
        validate="one_to_one",
    ).sort_values("utterance_id")
    if frame["audio_path"].isna().any():
        raise SystemExit("The input-audio manifest does not cover the corpus.")
    rows = frame.iloc[args.shard_index :: args.num_shards]
    if args.limit is not None:
        rows = rows.head(args.limit)

    config = replace(
        SidonDeessOnlyConfig(),
        final_postprocess=load_config(args.postprocess_config),
    )
    prior = load_prior(args.output_results) if args.resume else {}
    terminal: dict[str, dict[str, Any]] = {}
    pending: list[tuple[str, Path, int]] = []
    for row in rows.itertuples(index=False):
        identifier = str(row.utterance_id)
        source = Path(str(row.audio_path))
        target = args.output_root / f"{identifier}.wav"
        old = prior.get(identifier)
        attempts = int(old.get("processing_attempts", 0)) if old else 0
        if (
            old
            and old.get("processing_status") == "ok"
            and output_is_valid(target, source, config)
        ):
            terminal[identifier] = old
        elif (
            old
            and old.get("processing_status") == "failed"
            and attempts >= args.maximum_attempts
        ):
            terminal[identifier] = old
        else:
            pending.append((identifier, source, attempts + 1))

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.output_results.parent.mkdir(parents=True, exist_ok=True)
    args.model_cache.mkdir(parents=True, exist_ok=True)
    backend = SidonDeessOnlyBackend(args.device, args.model_cache)
    completed = 0
    failures = 0
    started_all = time.monotonic()
    with args.output_results.open("w", encoding="utf-8") as stream:
        for identifier in sorted(terminal):
            stream.write(json.dumps(terminal[identifier], sort_keys=True) + "\n")
        for identifier, source, attempts in pending:
            target = args.output_root / f"{identifier}.wav"
            try:
                audio, sample_rate = sf.read(
                    source,
                    always_2d=True,
                    dtype="float32",
                )
                if (
                    sample_rate != 24_000
                    or audio.shape[1] != 1
                    or not len(audio)
                    or not np.isfinite(audio).all()
                ):
                    raise RuntimeError("The canonical input WAV is invalid.")
                started = time.monotonic()
                output = backend.process(audio[:, 0], int(sample_rate))[
                    "sidon_deess_only"
                ]
                backend_seconds = time.monotonic() - started
                with tempfile.TemporaryDirectory(prefix="uktts-sidon-v6-") as temporary:
                    intermediate = Path(temporary) / "sidon.wav"
                    write_float_wav(intermediate, output[0], output[1])
                    postprocessing = apply_deessing_only(
                        intermediate,
                        target,
                        config,
                        ffmpeg_binary=args.ffmpeg,
                    )
                check = inspect_wav(target)
                if check["errors"]:
                    raise RuntimeError(f"Output validation failed: {check['errors']}")
                metadata = {
                    "status": "PASS",
                    "utterance_id": identifier,
                    "profile": "sidon_deess_only",
                    "source": str(source.resolve()),
                    "source_sha256": sha256(source),
                    "audio_sha256": sha256(target),
                    "processing_config": asdict(config),
                    "processing_config_hash": config.digest,
                    "backend": backend.identity,
                    "backend_seconds": backend_seconds,
                    "real_time_factor": backend_seconds / max(check["duration"], 1e-9),
                    "postprocessing": postprocessing,
                    "waveform": check,
                }
                write_json(target.with_suffix(".wav.json"), metadata)
                result = result_row(
                    utterance_id=identifier,
                    source=source,
                    target=target,
                    attempts=attempts,
                    status="ok",
                    config=config,
                    check=check,
                )
                completed += 1
            except Exception as error:
                result = result_row(
                    utterance_id=identifier,
                    source=source,
                    target=target,
                    attempts=attempts,
                    status="failed",
                    config=config,
                    error=f"{type(error).__name__}: {error}",
                )
                failures += 1
            stream.write(
                json.dumps(result, default=json_default, sort_keys=True) + "\n"
            )
            stream.flush()
            elapsed = max(time.monotonic() - started_all, 1e-6)
            print(
                json.dumps(
                    {
                        "completed": completed,
                        "failed": failures,
                        "processed": completed + failures,
                        "pending_at_start": len(pending),
                        "rate_files_per_second": (completed + failures) / elapsed,
                        "shard_index": args.shard_index,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return 0 if failures == 0 or args.allow_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
