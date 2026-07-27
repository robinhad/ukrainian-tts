#!/usr/bin/env python3
"""Select raw or clean audio for an exact 50/50 embedding extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


def score(value: str) -> str:
    return hashlib.sha256(f"expanded-v3-embedding-777\0{value}".encode()).hexdigest()


def assign_variants(frame: pd.DataFrame) -> dict[str, str]:
    """Assign half of all rows to raw audio and stratify by speaker."""
    groups: dict[str, list[str]] = defaultdict(list)
    stratum_column = (
        "speaker_stratum_id"
        if "speaker_stratum_id" in frame.columns
        else "speaker_id"
    )
    for row in frame.itertuples():
        groups[str(getattr(row, stratum_column))].append(str(row.utterance_id))
    base_raw = {speaker: len(items) // 2 for speaker, items in groups.items()}
    target_raw = len(frame) // 2
    extras = target_raw - sum(base_raw.values())
    odd_groups = sorted(
        (speaker for speaker, items in groups.items() if len(items) % 2),
        key=score,
    )
    for speaker in odd_groups[:extras]:
        base_raw[speaker] += 1
    assignment: dict[str, str] = {}
    for speaker, items in groups.items():
        ordered = sorted(items, key=score)
        raw_count = base_raw[speaker]
        assignment.update(
            {
                utterance_id: ("raw" if index < raw_count else "clean")
                for index, utterance_id in enumerate(ordered)
            }
        )
    return assignment


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def kaldi_speaker_id(utterance_id: str, speaker_id: str) -> str:
    """Keep Kaldi speaker sorting valid without changing utterance IDs."""
    return f"{utterance_id}--{speaker_id}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--kaldi-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_parquet(args.manifest).sort_values("utterance_id")
    required = {
        "utterance_id",
        "speaker_id",
        "split",
        "audio_path",
        "canonical_raw_audio_path",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"The manifest has no required columns: {missing}")
    assignment = assign_variants(frame)
    frame["embedding_audio_variant"] = [
        assignment[str(identifier)] for identifier in frame["utterance_id"]
    ]
    frame["embedding_audio_path"] = [
        str(raw) if variant == "raw" else str(clean)
        for variant, raw, clean in zip(
            frame["embedding_audio_variant"],
            frame["canonical_raw_audio_path"],
            frame["audio_path"],
        )
    ]
    missing_audio = [
        path for path in frame["embedding_audio_path"] if not Path(path).is_file()
    ]
    if missing_audio:
        raise SystemExit(f"Embedding audio does not exist: {missing_audio[:10]}")
    for split, split_frame in frame.groupby("split", sort=True):
        directory = args.kaldi_root / str(split)
        wav_lines = [
            f"{row.utterance_id} {row.embedding_audio_path}"
            for row in split_frame.itertuples()
        ]
        speakers: dict[str, list[str]] = defaultdict(list)
        for row in split_frame.itertuples():
            utterance_id = str(row.utterance_id)
            speaker = kaldi_speaker_id(utterance_id, str(row.speaker_id))
            speakers[speaker].append(utterance_id)
        write_lines(directory / "wav.scp", sorted(wav_lines))
        write_lines(
            directory / "spk2utt",
            [
                f"{speaker} {' '.join(sorted(utterances))}"
                for speaker, utterances in sorted(speakers.items())
            ],
        )
        write_lines(
            directory / "utt2spk",
            sorted(
                f"{row.utterance_id} "
                f"{kaldi_speaker_id(str(row.utterance_id), str(row.speaker_id))}"
                for row in split_frame.itertuples()
            ),
        )
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output_manifest, index=False)
    counts = frame["embedding_audio_variant"].value_counts().to_dict()
    per_speaker_delta = {}
    stratum_column = (
        "speaker_stratum_id"
        if "speaker_stratum_id" in frame.columns
        else "speaker_id"
    )
    for speaker, speaker_frame in frame.groupby(stratum_column):
        values = speaker_frame["embedding_audio_variant"].value_counts()
        per_speaker_delta[str(speaker)] = abs(
            int(values.get("raw", 0)) - int(values.get("clean", 0))
        )
    report = {
        "status": "PASS",
        "records": len(frame),
        "counts": {key: int(value) for key, value in sorted(counts.items())},
        "total_delta": abs(int(counts.get("raw", 0)) - int(counts.get("clean", 0))),
        "maximum_speaker_delta": max(per_speaker_delta.values(), default=0),
        "manifest": str(args.output_manifest),
        "kaldi_root": str(args.kaldi_root),
    }
    if report["total_delta"] > 1 or report["maximum_speaker_delta"] > 1:
        report["status"] = "FAIL"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
