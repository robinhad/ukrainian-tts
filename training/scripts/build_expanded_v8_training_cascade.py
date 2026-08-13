#!/usr/bin/env python3
"""Build the v8 training manifests from completed cascade outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from training.scripts.preprocess_training_cascade_audio import (
    PROFILE,
    PROFILE_HASH,
    PROFILE_NAME,
)

EXPECTED_RECORDS = 74_156
V7_NAME = "expanded_v7_sidon_deess_declick_limit_novoa"
V8_NAME = PROFILE_NAME
SPLIT_MAP = {
    f"{V7_NAME}_train": f"{V8_NAME}_train",
    f"{V7_NAME}_dev": f"{V8_NAME}_dev",
    f"{V7_NAME}_eval": f"{V8_NAME}_eval",
}


def load_results(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--processing-results", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-raw-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    base = pd.read_parquet(args.base_manifest)
    raw = pd.read_parquet(args.raw_manifest)
    results = load_results(args.processing_results)
    for name, frame in (("base", base), ("raw", raw), ("results", results)):
        if "utterance_id" not in frame or frame["utterance_id"].duplicated().any():
            raise SystemExit(f"The {name} data has invalid utterance IDs.")
    if len(base) != EXPECTED_RECORDS or len(raw) != EXPECTED_RECORDS:
        raise SystemExit("The v8 input record count is incorrect.")
    expected = set(base["utterance_id"].astype(str))
    if set(raw["utterance_id"].astype(str)) != expected:
        raise SystemExit("The raw embedding input does not cover the v8 corpus.")
    if set(results["utterance_id"].astype(str)) != expected:
        raise SystemExit("The processing results do not cover the v8 corpus.")
    if set(results["processing_status"].astype(str)) != {"ok"}:
        raise SystemExit("The processing results contain a failed record.")
    if results["processing_config_hash"].astype(str).ne(PROFILE_HASH).any():
        raise SystemExit("The processing results have a profile-hash mismatch.")

    result_columns = [
        "utterance_id",
        "audio_path",
        "audio_sha256",
        "duration",
        "sample_rate",
        "channels",
        "format",
        "processing_attempts",
        "degenerate_output_fallback",
    ]
    if "degenerate_output_fallback" not in results:
        results["degenerate_output_fallback"] = False
    frame = base.drop(
        columns=[name for name in result_columns[1:] if name in base],
        errors="ignore",
    ).merge(results[result_columns], on="utterance_id", validate="one_to_one")
    frame["degenerate_output_fallback"] = frame[
        "degenerate_output_fallback"
    ].fillna(False).astype(bool)
    unknown = sorted(set(frame["split"]) - set(SPLIT_MAP))
    if unknown:
        raise SystemExit(f"The base manifest has unexpected splits: {unknown}")
    frame["split"] = frame["split"].map(SPLIT_MAP)
    frame["qc_flags"] = [[] for _ in range(len(frame))]
    frame["preprocessing_profile"] = PROFILE_NAME
    frame["preprocessing_config_hash"] = PROFILE_HASH
    frame["clearervoice_applied"] = True
    frame["sidon_applied"] = True
    frame["deepfilternet_applied"] = True
    frame["rnnoise_applied"] = True
    frame["highpass_applied"] = False
    frame["post_highpass_applied"] = False
    frame["deessing_applied"] = True
    frame["declicking_applied"] = True
    frame["compression_applied"] = False
    frame["loudness_normalization_applied"] = True
    frame["loudness_normalization_type"] = "linear_source_match"
    frame["limiting_applied"] = True
    frame["embedding_audio_variant"] = None
    frame["embedding_audio_path"] = None
    if frame["source_id"].astype(str).str.contains("voa", case=False).any():
        raise SystemExit("The v8 corpus contains VOA records.")
    if not frame["sample_rate"].astype(int).eq(24_000).all():
        raise SystemExit("The v8 corpus has a non-24-kHz record.")
    if not frame["channels"].astype(int).eq(1).all():
        raise SystemExit("The v8 corpus has a non-mono record.")
    if not frame["format"].astype(str).eq("WAV/PCM_24").all():
        raise SystemExit("The v8 corpus has a non-PCM24 record.")
    if any(Path(str(path)).is_symlink() or not Path(str(path)).is_file() for path in frame["audio_path"]):
        raise SystemExit("The v8 corpus has a missing file or a symbolic link.")

    frame = frame.sort_values("utterance_id").reset_index(drop=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output_dir / "all.parquet", index=False)
    for split, part in frame.groupby("split", sort=True):
        part.to_parquet(args.output_dir / f"{split}.parquet", index=False)
    raw = raw.sort_values("utterance_id").reset_index(drop=True)
    args.output_raw_manifest.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(args.output_raw_manifest, index=False)

    report = {
        "status": "PASS",
        "records": int(len(frame)),
        "hours": round(float(frame["duration"].sum() / 3600), 7),
        "profile": PROFILE,
        "profile_hash": PROFILE_HASH,
        "voa_records": 0,
        "degenerate_output_fallback_records": int(
            frame["degenerate_output_fallback"].sum()
        ),
        "splits": {
            str(split): {
                "records": int(len(part)),
                "hours": round(float(part["duration"].sum() / 3600), 7),
            }
            for split, part in frame.groupby("split", sort=True)
        },
        "sources": {
            str(source): {
                "records": int(len(part)),
                "hours": round(float(part["duration"].sum() / 3600), 7),
            }
            for source, part in frame.groupby("source_id", sort=True)
        },
        "manifest": str((args.output_dir / "all.parquet").resolve()),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
