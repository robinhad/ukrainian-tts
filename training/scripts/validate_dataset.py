#!/usr/bin/env python3
"""Validate manifests, audio invariants and split leakage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


def text_sequence_errors(
    utterance_id: str,
    text: str,
    phonemes: object,
    *,
    max_text_characters: int,
    max_phoneme_tokens: int,
) -> list[str]:
    """Return hard errors for empty or implausibly long text input."""
    errors = []
    if not text.strip() or len(phonemes) == 0:
        errors.append(f"{utterance_id}: empty text or phonemes")
    if len(text) > max_text_characters:
        errors.append(
            f"{utterance_id}: text has {len(text)} characters; "
            f"maximum={max_text_characters}"
        )
    if len(phonemes) > max_phoneme_tokens:
        errors.append(
            f"{utterance_id}: phonemes have {len(phonemes)} tokens; "
            f"maximum={max_phoneme_tokens}"
        )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-text-characters", type=int, default=500)
    parser.add_argument("--max-phoneme-tokens", type=int, default=500)
    args = parser.parse_args()
    if args.max_text_characters < 1 or args.max_phoneme_tokens < 1:
        parser.error("text and phoneme limits must be positive")
    frame = pd.read_parquet(args.manifest)
    errors = []
    required = {
        "utterance_id", "speaker_id", "audio_path", "text_raw", "text_sanitized",
        "espeak_phonemes", "duration", "sample_rate", "split", "qc_flags",
        "audio_sha256", "text_sha256",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        errors.append(f"missing columns: {missing}")
    if frame["utterance_id"].duplicated().any():
        errors.append("duplicate utterance IDs")
    for key in ("audio_sha256", "text_sha256"):
        leakage = frame.groupby(key)["split"].nunique()
        if (leakage > 1).any():
            errors.append(f"split leakage by {key}: {(leakage > 1).sum()} hashes")
    durations = []
    clipping = 0
    for row in frame.itertuples():
        path = Path(row.audio_path)
        if not path.is_file():
            errors.append(f"missing audio: {path}")
            continue
        audio, sample_rate = sf.read(path, always_2d=True, dtype="float32")
        if sample_rate != 24000:
            errors.append(f"{row.utterance_id}: sample_rate={sample_rate}")
        if audio.shape[1] != 1 or len(audio) == 0:
            errors.append(f"{row.utterance_id}: invalid channels/duration")
        if not np.isfinite(audio).all():
            errors.append(f"{row.utterance_id}: non-finite audio")
        if audio.size and np.max(np.abs(audio)) >= 0.999:
            clipping += 1
        durations.append(len(audio) / sample_rate)
        errors.extend(text_sequence_errors(
            str(row.utterance_id),
            str(row.text_sanitized),
            row.espeak_phonemes,
            max_text_characters=args.max_text_characters,
            max_phoneme_tokens=args.max_phoneme_tokens,
        ))
    report = {
        "status": "PASS" if not errors else "FAIL", "utterances": len(frame),
        "splits": frame["split"].value_counts().sort_index().to_dict(),
        "duration_hours": sum(durations) / 3600,
        "duration_min": min(durations) if durations else None,
        "duration_median": float(np.median(durations)) if durations else None,
        "duration_max": max(durations) if durations else None,
        "clipping_files": clipping, "errors": errors,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
