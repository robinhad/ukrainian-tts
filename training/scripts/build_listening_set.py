#!/usr/bin/env python3
"""Make a deterministic listening set without copying large audio files."""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

import pandas as pd


def evenly_spaced_rows(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    """Select rows across the duration range."""
    if count < 1 or count > len(frame):
        raise ValueError(f"count must be in the range 1..{len(frame)}")
    ordered = frame.sort_values(["duration", "utterance_id"], kind="stable")
    if count == 1:
        return ordered.iloc[[len(ordered) // 2]]
    indices = [round(i * (len(ordered) - 1) / (count - 1)) for i in range(count)]
    return ordered.iloc[indices]


def make_link(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    destination.symlink_to(os.path.relpath(source.resolve(), destination.parent.resolve()))


def parse_candidate(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("candidate must use LABEL=WAV_DIR")
    label, path = value.split("=", 1)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", label):
        raise argparse.ArgumentTypeError(f"invalid candidate label: {label}")
    return label, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--candidate", action="append", type=parse_candidate, required=True)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_parquet(args.manifest)
    selected = evenly_spaced_rows(frame, args.count)
    rows = []
    for item in selected.itertuples(index=False):
        utterance_id = str(item.utterance_id)
        reference = args.raw_dir / f"{utterance_id}.ogg"
        reference_link = args.output / "reference_raw" / reference.name
        make_link(reference, reference_link)
        row = {
            "utterance_id": utterance_id,
            "duration": f"{float(item.duration):.3f}",
            "text": str(item.text_sanitized),
            "reference_raw": str(reference_link.relative_to(args.output)),
        }
        for label, wav_dir in args.candidate:
            candidate = wav_dir / f"{utterance_id}.wav"
            candidate_link = args.output / label / candidate.name
            make_link(candidate, candidate_link)
            row[label] = str(candidate_link.relative_to(args.output))
        rows.append(row)

    args.output.mkdir(parents=True, exist_ok=True)
    fields = ["utterance_id", "duration", "text", "reference_raw"] + [
        label for label, _ in args.candidate
    ]
    with (args.output / "manifest.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Made {len(rows)} listening items in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
