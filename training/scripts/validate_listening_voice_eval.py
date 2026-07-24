#!/usr/bin/env python3
"""Validate a small speaker-conditioned listening evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--wav-dir", type=Path, required=True)
    parser.add_argument("--text", required=True)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    errors: list[str] = []
    generated = []
    for voice_record in report["voices"]:
        voice = voice_record["voice"]
        wav_path = args.wav_dir / f"{voice}.wav"
        metadata_path = wav_path.with_suffix(".wav.json")
        if not wav_path.is_file() or not metadata_path.is_file():
            errors.append(f"{voice}: output or metadata is missing")
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        waveform, sample_rate = sf.read(
            wav_path, dtype="float32", always_2d=True
        )
        info = sf.info(wav_path)
        if sample_rate != 24000:
            errors.append(f"{voice}: sample rate is {sample_rate}")
        if info.channels != 1 or waveform.shape[1] != 1:
            errors.append(f"{voice}: output is not mono")
        if waveform.size == 0 or not np.isfinite(waveform).all():
            errors.append(f"{voice}: output is empty or non-finite")
        peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
        if peak == 0.0:
            errors.append(f"{voice}: output is all zero")
        if peak >= 0.999:
            errors.append(f"{voice}: output has possible clipping")
        if metadata.get("text_raw") != args.text:
            errors.append(f"{voice}: metadata text does not match")
        generated.append(
            {
                "voice": voice,
                "wav": str(wav_path.resolve()),
                "metadata": str(metadata_path.resolve()),
                "duration": float(info.duration),
                "peak_absolute": peak,
                "real_time_factor": metadata.get("real_time_factor"),
                "sample_rate": sample_rate,
                "channels": info.channels,
            }
        )

    report["listening_sentence"] = args.text
    report["perceptual_issue"] = {
        "status": "FAIL",
        "issue": "metallic_timbre",
        "source": "user_listening_report",
        "next_action": "Use this five-voice set to separate speaker-conditioning effects from the shared model artifact.",
    }
    report["generated"] = generated
    report["errors"] = errors
    report["status"] = "PASS" if not errors and len(generated) == 5 else "FAIL"
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "wav_count": len(generated),
                "errors": errors,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
