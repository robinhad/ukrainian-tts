#!/usr/bin/env python3
"""Export deterministic training-audio samples for each source dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from training.scripts.generate_per_dataset_listening_eval import (
    select_rows,
    validate_audio,
)


def sha256(path: Path) -> str:
    """Calculate the SHA-256 value for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_training_rows(
    frame: pd.DataFrame,
    count_per_dataset: int,
    split_suffix: str,
) -> pd.DataFrame:
    """Select balanced samples only from the requested training split."""
    if "split" not in frame.columns:
        raise ValueError("manifest column is missing: split")
    training = frame.loc[
        frame["split"].astype(str).str.endswith(split_suffix, na=False)
    ].copy()
    if training.empty:
        raise ValueError(f"no rows use a split that ends with {split_suffix!r}")
    return select_rows(training, count_per_dataset)


def atomic_copy(source: Path, destination: Path) -> None:
    """Copy one file through a temporary path and do not make a symlink."""
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    if destination.is_symlink() or not destination.is_file():
        raise RuntimeError(f"audio copy failed: {destination}")


def atomic_text(path: Path, content: str) -> None:
    """Write one text file through a temporary path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_tsv(path: Path, rows: list[dict], fields: list[str]) -> None:
    """Write a UTF-8 TSV file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--count-per-dataset", type=int, default=10)
    parser.add_argument("--split-suffix", default="_train")
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"output directory is not empty: {args.output}")

    frame = pd.read_parquet(args.manifest)
    selected = select_training_rows(
        frame,
        args.count_per_dataset,
        args.split_suffix,
    )
    args.output.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    errors: list[dict] = []
    for row in selected.itertuples(index=False):
        dataset = str(row.source_id)
        index = int(row.listening_index)
        utterance_id = str(row.utterance_id)
        source = Path(row.audio_path)
        destination = args.output / dataset / f"{index:02d}_{utterance_id}.wav"
        atomic_copy(source, destination)
        check = validate_audio(destination)
        copied_sha256 = sha256(destination)
        manifest_sha256 = str(getattr(row, "audio_sha256", ""))
        item_errors = list(check["errors"])
        if manifest_sha256 and copied_sha256 != manifest_sha256:
            item_errors.append("audio SHA-256 does not match the manifest")
        if item_errors:
            errors.append({"utterance_id": utterance_id, "errors": item_errors})
        records.append(
            {
                "dataset": dataset,
                "item": f"{index:02d}",
                "utterance_id": utterance_id,
                "speaker_id": str(row.speaker_id),
                "embedding_audio_variant": str(row.embedding_audio_variant),
                "duration_seconds": f"{check['duration']:.3f}",
                "text": str(row.text_sanitized),
                "processing_profile": str(row.preprocessing_profile),
                "wav": str(destination.relative_to(args.output)),
                "audio_sha256": copied_sha256,
            }
        )

    fields = [
        "dataset",
        "item",
        "utterance_id",
        "speaker_id",
        "embedding_audio_variant",
        "duration_seconds",
        "text",
        "processing_profile",
        "wav",
        "audio_sha256",
    ]
    write_tsv(args.output / "manifest.tsv", records, fields)

    datasets = {
        str(dataset): {
            "items": int(len(group)),
            "duration_seconds": round(
                sum(float(item["duration_seconds"]) for item in records if item["dataset"] == dataset),
                3,
            ),
            "raw_embedding_items": int((group["embedding_audio_variant"] == "raw").sum()),
            "clean_embedding_items": int((group["embedding_audio_variant"] == "clean").sum()),
        }
        for dataset, group in selected.groupby("source_id", sort=True)
    }
    report = {
        "status": "PASS" if not errors else "FAIL",
        "mode": "training_audio_samples_per_dataset",
        "contains_synthesis": False,
        "storage": "independent_wav_copies",
        "uses_symlinks": False,
        "manifest": str(args.manifest.resolve()),
        "split_suffix": args.split_suffix,
        "output": str(args.output.resolve()),
        "dataset_count": len(datasets),
        "items_per_dataset": args.count_per_dataset,
        "item_count": len(records),
        "selection": "duration_spaced_with_equal_raw_clean_embedding_assignment",
        "datasets": datasets,
        "errors": errors,
        "records": records,
    }
    atomic_text(
        args.report,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    readme = f"""# Training-Audio Samples by Dataset

This document uses ASD-STE100 Simplified Technical English style. An approved
STE checker did not certify this document.

This set contains audio that the latest v7 model used for training. It does not
contain synthesized audio. The set has {len(datasets)} datasets and
{args.count_per_dataset} WAV files for each dataset. Each WAV file is an
independent copy. The set does not use symlinks.

The selection covers the duration range of each dataset. It also has an equal
number of `raw` and `clean` speaker-embedding assignments. All copied waveforms
use the clean v7 processing profile. Use `manifest.tsv` to see the text,
duration, processing profile, and SHA-256 value.

The automatic validation status is `{report['status']}`. Automatic validation
checks the file structure, sample rate, channel count, waveform values, and
file hash. It does not measure speech quality.
"""
    atomic_text(args.output / "README.md", readme)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "records"},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
