#!/usr/bin/env python3
"""Convert Parquet smoke manifests into deterministic ESPnet data directories."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd


def write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def kaldi_speaker_id(utterance_id: str, speaker_id: str) -> str:
    """Keep Kaldi speaker sorting valid without changing utterance IDs."""
    return f"{utterance_id}--{speaker_id}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifests = [
        path for path in sorted(args.manifest_dir.glob("*.parquet"))
        if path.stem not in {"all", "all_with_embeddings"}
    ]
    for manifest in manifests:
        frame = pd.read_parquet(manifest).sort_values("utterance_id")
        target = args.output_root / manifest.stem
        target.mkdir(parents=True, exist_ok=True)
        wav, text, utt2spk = [], [], []
        speakers = defaultdict(list)
        for row in frame.itertuples():
            utt = str(row.utterance_id)
            speaker = kaldi_speaker_id(utt, str(row.speaker_id))
            wav.append(f"{utt} {row.audio_path}")
            text.append(f"{utt} {row.text_sanitized}")
            utt2spk.append(f"{utt} {speaker}")
            speakers[speaker].append(utt)
        write_lines(target / "wav.scp", wav)
        write_lines(target / "text", text)
        write_lines(target / "utt2spk", utt2spk)
        write_lines(target / "spk2utt", [f"{spk} {' '.join(sorted(utts))}" for spk, utts in sorted(speakers.items())])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
