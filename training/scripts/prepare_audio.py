#!/usr/bin/env python3
"""Create untouched-from-mastering 24 kHz mono PCM WAV model copies."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf


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
            "-i", str(source), "-map_metadata", "-1", "-ac", "1", "-ar", "24000",
            "-c:a", "pcm_s16le", str(target),
        ],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-records", type=Path, required=True)
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
            duration = len(audio) / sample_rate
            if sample_rate != 24000:
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
                audio_sha256=sha256(target),
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
