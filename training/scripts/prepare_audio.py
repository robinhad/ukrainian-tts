#!/usr/bin/env python3
"""Create trimmed 24 kHz mono PCM WAV model copies without mastering."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 24_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def convert(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source), "-map_metadata", "-1", "-ac", "1", "-ar", str(SAMPLE_RATE),
            "-c:a", "pcm_s16le", str(target),
        ],
        check=True,
    )


def trim_config_hash(*, top_db: float, padding_ms: float, frame_length: int, hop_length: int) -> str:
    payload = {
        "algorithm": "relative_frame_rms_v1",
        "frame_length": frame_length,
        "hop_length": hop_length,
        "padding_ms": padding_ms,
        "top_db": top_db,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def trim_silence(
    audio: np.ndarray,
    sample_rate: int,
    *,
    top_db: float = 40.0,
    padding_ms: float = 100.0,
    frame_length: int = 1024,
    hop_length: int = 256,
) -> tuple[np.ndarray, dict[str, float | int | bool | str]]:
    """Remove only leading and trailing low-RMS frames and keep safety padding."""
    if audio.ndim != 2 or audio.shape[1] != 1:
        raise ValueError("trim_silence requires mono audio with shape [samples, 1]")
    if top_db <= 0 or padding_ms < 0 or frame_length < 1 or hop_length < 1:
        raise ValueError("invalid silence-trim configuration")

    sample_count = len(audio)
    if sample_count == 0:
        raise ValueError("cannot trim empty audio")
    signal = audio[:, 0].astype(np.float64, copy=False)
    effective_frame = min(frame_length, sample_count)
    final_start = sample_count - effective_frame
    starts = np.arange(0, final_start + 1, hop_length, dtype=np.int64)
    if starts.size == 0 or starts[-1] != final_start:
        starts = np.append(starts, final_start)
    cumulative_energy = np.concatenate(([0.0], np.cumsum(np.square(signal))))
    energies = (
        cumulative_energy[starts + effective_frame] - cumulative_energy[starts]
    ) / effective_frame
    rms = np.sqrt(np.maximum(energies, 0.0))
    peak_rms = float(np.max(rms))
    threshold_rms = peak_rms * (10.0 ** (-top_db / 20.0))
    active = np.flatnonzero(rms > threshold_rms) if peak_rms > 0.0 else np.array([], dtype=int)

    if active.size:
        active_start = int(starts[int(active[0])])
        active_end = min(
            sample_count,
            int(starts[int(active[-1])]) + effective_frame,
        )
        padding = round(sample_rate * padding_ms / 1000.0)
        trim_start = max(0, active_start - padding)
        trim_end = min(sample_count, active_end + padding)
    else:
        trim_start = 0
        trim_end = sample_count

    trimmed = audio[trim_start:trim_end]
    removed_end = sample_count - trim_end
    metadata: dict[str, float | int | bool | str] = {
        "trim_applied": bool(trim_start or removed_end),
        "duration_before_trim": sample_count / sample_rate,
        "trim_start_seconds": trim_start / sample_rate,
        "trim_end_seconds": removed_end / sample_rate,
        "trim_removed_seconds": (trim_start + removed_end) / sample_rate,
        "trim_threshold_rms": threshold_rms,
        "trim_config_hash": trim_config_hash(
            top_db=top_db,
            padding_ms=padding_ms,
            frame_length=frame_length,
            hop_length=hop_length,
        ),
    }
    return trimmed, metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-records", type=Path, required=True)
    trim_group = parser.add_mutually_exclusive_group()
    trim_group.add_argument("--trim-silence", dest="trim_silence", action="store_true")
    trim_group.add_argument("--no-trim-silence", dest="trim_silence", action="store_false")
    parser.set_defaults(trim_silence=True)
    parser.add_argument("--trim-top-db", type=float, default=40.0)
    parser.add_argument("--trim-padding-ms", type=float, default=100.0)
    parser.add_argument("--trim-frame-length", type=int, default=1024)
    parser.add_argument("--trim-hop-length", type=int, default=256)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.records.read_text(encoding="utf-8").splitlines()]
    prepared = []
    for row in records:
        flags = []
        source = Path(row["audio_path"])
        target = args.output_root / f"{row['utterance_id']}.wav"
        try:
            convert(source, target)
            audio, sample_rate = sf.read(target, always_2d=True, dtype="float32")
            if args.trim_silence:
                audio, trim_metadata = trim_silence(
                    audio,
                    sample_rate,
                    top_db=args.trim_top_db,
                    padding_ms=args.trim_padding_ms,
                    frame_length=args.trim_frame_length,
                    hop_length=args.trim_hop_length,
                )
                sf.write(target, audio, sample_rate, subtype="PCM_16")
                audio, sample_rate = sf.read(target, always_2d=True, dtype="float32")
            else:
                trim_metadata = {
                    "trim_applied": False,
                    "duration_before_trim": len(audio) / sample_rate,
                    "trim_start_seconds": 0.0,
                    "trim_end_seconds": 0.0,
                    "trim_removed_seconds": 0.0,
                    "trim_threshold_rms": 0.0,
                    "trim_config_hash": "disabled",
                }
            duration = len(audio) / sample_rate
            if sample_rate != SAMPLE_RATE:
                flags.append("wrong_sample_rate")
            if audio.shape[1] != 1:
                flags.append("not_mono")
            if duration <= 0:
                flags.append("zero_duration")
            if duration < 2:
                flags.append("too_short")
            if duration > 12:
                flags.append("too_long")
            if not np.isfinite(audio).all():
                flags.append("non_finite")
            if audio.size and float(np.max(np.abs(audio))) >= 0.999:
                flags.append("clipping")
            row.update(
                audio_path=str(target.resolve()), duration=duration, sample_rate=sample_rate,
                channels=audio.shape[1], format="WAV/PCM_16", qc_flags=flags,
                audio_sha256=sha256(target), **trim_metadata,
            )
        except Exception as error:
            row.update(qc_flags=[f"corrupt:{type(error).__name__}"], duration=0.0, sample_rate=0)
        prepared.append(row)
    args.output_records.parent.mkdir(parents=True, exist_ok=True)
    with args.output_records.open("w", encoding="utf-8") as stream:
        for row in prepared:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"artifact": str(args.output_records), "records": len(prepared)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
