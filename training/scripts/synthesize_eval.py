#!/usr/bin/env python3
"""Validate ESPnet JETS evaluation WAVs and persist reproducibility metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inference_timings(path: Path | None) -> dict[str, tuple[float, float]]:
    """Return utterance -> (seconds, RTF) from adjacent ESPnet log messages."""
    if path is None or not path.is_file():
        return {}
    speed = None
    timings: dict[str, tuple[float, float]] = {}
    speed_pattern = re.compile(r"inference speed = ([0-9.]+) points / sec")
    item_pattern = re.compile(r"INFO: (\S+) \(size:\d+->(\d+)\)")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = speed_pattern.search(line)
        if match:
            speed = float(match.group(1))
            continue
        match = item_pattern.search(line)
        if match and speed:
            samples = int(match.group(2))
            seconds = samples / speed
            timings[match.group(1)] = (seconds, 24000.0 / speed)
            speed = None
    return timings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--inference-log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_parquet(args.manifest).set_index("utterance_id")
    timings = inference_timings(args.inference_log)
    errors: list[str] = []
    records = []
    expected = set(map(str, frame.index))
    wavs = {path.stem: path for path in sorted(args.wav_dir.glob("*.wav"))}
    if set(wavs) != expected:
        missing = sorted(expected - set(wavs))
        extra = sorted(set(wavs) - expected)
        errors.append(f"wav/id mismatch: missing={missing}, extra={extra}")

    checkpoint_hash = sha256(args.checkpoint)
    config_hash = sha256(args.config)
    for utterance_id in sorted(expected & set(wavs)):
        path = wavs[utterance_id]
        info = sf.info(path)
        waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        flags = []
        if sample_rate != 24000:
            errors.append(f"{utterance_id}: sample_rate={sample_rate}")
        if info.channels != 1 or waveform.shape[1] != 1:
            errors.append(f"{utterance_id}: channels={info.channels}")
        if waveform.size == 0 or info.frames == 0:
            errors.append(f"{utterance_id}: empty waveform")
        if not np.isfinite(waveform).all():
            errors.append(f"{utterance_id}: non-finite samples")
        peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
        if peak == 0.0:
            errors.append(f"{utterance_id}: all-zero waveform")
        if peak >= 0.999:
            flags.append("possible_clipping")
        duration = info.frames / sample_rate if sample_rate else 0.0
        if not 0.1 <= duration <= 30.0:
            errors.append(f"{utterance_id}: anomalous duration={duration:.3f}")
        generation_seconds, rtf = timings.get(utterance_id, (None, None))
        row = frame.loc[utterance_id]
        phonemes = row["espeak_phonemes"]
        if hasattr(phonemes, "tolist"):
            phonemes = phonemes.tolist()
        records.append(
            {
                "utterance_id": utterance_id,
                "text_raw": row["text_raw"],
                "text_sanitized": row["text_sanitized"],
                "espeak_phonemes": phonemes,
                "frontend_config_hash": row["frontend_config_hash"],
                "espeak_version": row["espeak_version"],
                "checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_sha256": checkpoint_hash,
                "config": str(args.config.resolve()),
                "config_sha256": config_hash,
                "wav": str(path.resolve()),
                "wav_sha256": sha256(path),
                "sample_rate": sample_rate,
                "channels": info.channels,
                "duration": duration,
                "peak_absolute": peak,
                "generation_seconds": generation_seconds,
                "real_time_factor": rtf,
                "warnings": flags,
            }
        )

    rtfs = [x["real_time_factor"] for x in records if x["real_time_factor"] is not None]
    report = {
        "status": "PASS" if not errors else "FAIL",
        "wav_count": len(records),
        "errors": errors,
        "clipping_warnings": sum("possible_clipping" in x["warnings"] for x in records),
        "duration_min": min((x["duration"] for x in records), default=0.0),
        "duration_max": max((x["duration"] for x in records), default=0.0),
        "rtf_median": float(np.median(rtfs)) if rtfs else None,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
