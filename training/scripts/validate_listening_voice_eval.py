#!/usr/bin/env python3
"""Validate a small speaker-conditioned listening evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--wav-dir", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--human-listening-status",
        choices=("NOT_RUN", "PASS", "FAIL"),
        default="NOT_RUN",
    )
    parser.add_argument("--perceptual-issue", action="append", default=[])
    parser.add_argument(
        "--human-listening-source", default="user_listening_report"
    )
    parser.add_argument("--human-listening-note", action="append", default=[])
    args = parser.parse_args()

    expected_checkpoint = (
        str(args.checkpoint.resolve()) if args.checkpoint is not None else None
    )
    expected_checkpoint_sha256 = (
        file_hash(args.checkpoint) if args.checkpoint is not None else None
    )
    expected_config = str(args.config.resolve()) if args.config is not None else None
    expected_config_sha256 = (
        file_hash(args.config) if args.config is not None else None
    )

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
        if expected_checkpoint is not None and metadata.get("checkpoint") != expected_checkpoint:
            errors.append(f"{voice}: metadata checkpoint does not match")
        if (
            expected_checkpoint_sha256 is not None
            and metadata.get("checkpoint_sha256") != expected_checkpoint_sha256
        ):
            errors.append(f"{voice}: metadata checkpoint hash does not match")
        if expected_config is not None and metadata.get("config") != expected_config:
            errors.append(f"{voice}: metadata config does not match")
        if (
            expected_config_sha256 is not None
            and metadata.get("config_sha256") != expected_config_sha256
        ):
            errors.append(f"{voice}: metadata config hash does not match")
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
                "checkpoint_sha256": metadata.get("checkpoint_sha256"),
                "config_sha256": metadata.get("config_sha256"),
                "frontend_config_hash": metadata.get("frontend_config_hash"),
                "espeak_version": metadata.get("espeak_version"),
            }
        )

    automatic_status = "PASS" if not errors and len(generated) == 5 else "FAIL"
    if automatic_status == "FAIL" or args.human_listening_status == "FAIL":
        release_status = "FAIL"
    elif args.human_listening_status == "PASS":
        release_status = "PASS"
    else:
        release_status = "NOT_READY"

    report["listening_sentence"] = args.text
    report["model"] = {
        "checkpoint": expected_checkpoint,
        "checkpoint_sha256": expected_checkpoint_sha256,
        "config": expected_config,
        "config_sha256": expected_config_sha256,
    }
    report.pop("perceptual_issue", None)
    report["automatic_validation"] = {
        "status": automatic_status,
        "errors": errors,
    }
    report["human_listening"] = {
        "status": args.human_listening_status,
        "issues": args.perceptual_issue,
        "notes": args.human_listening_note,
        "source": (
            args.human_listening_source
            if args.human_listening_status != "NOT_RUN"
            else None
        ),
    }
    report["release_status"] = release_status
    report["generated"] = generated
    report["errors"] = errors
    # Keep this field as the automatic status for compatibility with the
    # finalizer and with earlier machine-readable reports.
    report["status"] = automatic_status
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
    return 0 if automatic_status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
