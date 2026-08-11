#!/usr/bin/env python3
"""Prepare ESPnet raw data with direct PCM24 paths and no audio rewrite."""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import soundfile as sf


def write_atomic(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write("\n".join(lines) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def ids(path: Path) -> set[str]:
    return {
        line.split(maxsplit=1)[0]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--kaldi-data-root", type=Path, required=True)
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--train-set", required=True)
    parser.add_argument("--valid-set", required=True)
    parser.add_argument("--test-set", required=True)
    args = parser.parse_args()

    for data_set in (args.train_set, args.valid_set, args.test_set):
        manifest_path = args.manifest_dir / f"{data_set}.parquet"
        source_dir = args.kaldi_data_root / data_set
        destination = args.dump_dir / "raw"
        if data_set in {args.train_set, args.valid_set}:
            destination = destination / "org"
        destination = destination / data_set
        frame = pd.read_parquet(
            manifest_path,
            columns=["utterance_id", "audio_path", "sample_rate", "channels", "format"],
        ).sort_values("utterance_id")
        if frame["utterance_id"].duplicated().any():
            raise SystemExit(f"The manifest has duplicate IDs: {manifest_path}")
        expected = set(frame["utterance_id"].astype(str))
        for name in ("wav.scp", "text", "utt2spk"):
            path = source_dir / name
            if not path.is_file() or ids(path) != expected:
                raise SystemExit(f"The Kaldi file has incorrect IDs: {path}")
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("text", "utt2spk", "spk2utt"):
            shutil.copy2(source_dir / name, destination / name)

        wav_lines = []
        sample_lines = []
        for row in frame.itertuples(index=False):
            path = Path(str(row.audio_path)).resolve()
            info = sf.info(path)
            if (
                path.is_symlink()
                or int(row.sample_rate) != 24_000
                or int(row.channels) != 1
                or str(row.format) != "WAV/PCM_24"
                or info.samplerate != 24_000
                or info.channels != 1
                or info.format not in {"WAV", "WAVEX"}
                or info.subtype != "PCM_24"
                or info.frames <= 0
            ):
                raise SystemExit(f"The direct PCM24 input is invalid: {path}")
            identifier = str(row.utterance_id)
            wav_lines.append(f"{identifier} {path}")
            sample_lines.append(f"{identifier} {info.frames}")
        write_atomic(destination / "wav.scp", wav_lines)
        write_atomic(destination / "utt2num_samples", sample_lines)
        write_atomic(destination / "feats_type", ["raw"])
        if ids(destination / "wav.scp") != expected:
            raise SystemExit(
                f"The direct PCM24 output has incorrect IDs: {destination}"
            )
        print(f"{data_set}: {len(frame)} direct PCM24 paths")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
