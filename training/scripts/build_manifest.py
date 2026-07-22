#!/usr/bin/env python3
"""Build machine-readable manifests with the exact frontend output."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from training.frontend.phonemize import UkrainianPhonemizer


FIELDS = [
    "utterance_id", "speaker_id", "audio_path", "text_raw", "text_sanitized",
    "espeak_phonemes", "duration", "sample_rate", "source", "source_license",
    "split", "qc_flags", "audio_sha256", "text_sha256", "espeak_version",
    "frontend_config_hash",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    frontend = UkrainianPhonemizer(cache_path=args.cache)
    rows = []
    for line in args.records.read_text(encoding="utf-8").splitlines():
        source = json.loads(line)
        sanitized, tokens = frontend.phonemize(source["text_raw"])
        source.update(
            text_sanitized=sanitized,
            espeak_phonemes=tokens,
            text_sha256=hashlib.sha256(sanitized.encode("utf-8")).hexdigest(),
            espeak_version=frontend.config.espeak_version,
            frontend_config_hash=frontend.config.digest,
        )
        rows.append({field: source.get(field) for field in FIELDS})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).sort_values("utterance_id")
    frame.to_parquet(args.output_dir / "all.parquet", index=False)
    for split, split_frame in frame.groupby("split"):
        split_frame.to_parquet(args.output_dir / f"{split}.parquet", index=False)
    metadata = {"frontend_config": asdict(frontend.config), "frontend_config_hash": frontend.config.digest}
    (args.output_dir / "frontend.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"artifact": str(args.output_dir / 'all.parquet'), "records": len(frame)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
