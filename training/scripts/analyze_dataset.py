#!/usr/bin/env python3
"""Create a detailed machine-readable full-corpus coverage report."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def percentile(values, points=(0, 1, 5, 25, 50, 75, 95, 99, 100)) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {str(point): float(np.percentile(array, point)) for point in points}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_parquet(args.manifest)
    phonemes = Counter(token for sequence in frame.espeak_phonemes for token in sequence)
    characters = Counter(character for text in frame.text_sanitized for character in text)
    qc_flags = Counter(flag for flags in frame.qc_flags for flag in flags)
    text_lengths = frame.text_sanitized.str.len().to_numpy()
    phoneme_lengths = frame.espeak_phonemes.map(len).to_numpy()
    durations = frame.duration.to_numpy()
    chars_per_second = text_lengths / durations
    comparison_text = frame.text_sanitized.map(
        lambda text: re.sub(r"[^\w]+", "", str(text).casefold(), flags=re.UNICODE)
    )
    group_leakage = (
        frame.groupby("source_group")["split"].nunique().gt(1).sum()
        if "source_group" in frame.columns else None
    )
    ratio_low, ratio_high = np.percentile(chars_per_second, [1, 99])
    report = {
        "utterances": len(frame),
        "duration_hours": float(durations.sum() / 3600),
        "duration_seconds_percentiles": percentile(durations),
        "splits": frame.split.value_counts().sort_index().to_dict(),
        "speakers": frame.speaker_id.value_counts().to_dict(),
        "sample_rates": frame.sample_rate.value_counts().sort_index().to_dict(),
        "text_length_percentiles": percentile(text_lengths),
        "phoneme_length_percentiles": percentile(phoneme_lengths),
        "max_phoneme_sequences": [
            {"utterance_id": str(frame.iloc[index].utterance_id), "tokens": int(phoneme_lengths[index])}
            for index in np.argsort(phoneme_lengths)[-20:][::-1]
        ],
        "qc_flags": dict(sorted(qc_flags.items())),
        "duplicate_audio_hashes": int(frame.audio_sha256.duplicated().sum()),
        "duplicate_text_hashes": int(frame.text_sha256.duplicated().sum()),
        "near_duplicate_comparison_texts": int(comparison_text.duplicated().sum()),
        "source_group_split_leakage": int(group_leakage) if group_leakage is not None else None,
        "character_coverage": dict(sorted(characters.items())),
        "phoneme_coverage": dict(sorted(phonemes.items())),
        "rare_phonemes_le_10": dict(sorted((token, count) for token, count in phonemes.items() if count <= 10)),
        "chars_per_second_percentiles": percentile(chars_per_second),
        "unusual_text_audio_ratio_count": int(
            ((chars_per_second < ratio_low) | (chars_per_second > ratio_high)).sum()
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "artifact": str(args.output),
                "utterances": report["utterances"],
                "duration_hours": report["duration_hours"],
                "phoneme_tokens": len(phonemes),
                "rare_phonemes": len(report["rare_phonemes_le_10"]),
                "source_group_split_leakage": report["source_group_split_leakage"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
