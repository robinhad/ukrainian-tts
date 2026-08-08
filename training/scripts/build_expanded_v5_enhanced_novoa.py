#!/usr/bin/env python3
"""Build the enhanced v5 corpus and exclude all VOA records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


PROFILE = {
    "name": "expanded_v3_enhanced_v1",
    "boundary_trim": True,
    "deepfilternet": True,
    "highpass": True,
    "deessing": True,
    "compression": True,
    "two_pass_ebu_r128": True,
    "sample_rate": 24000,
    "channels": 1,
    "format": "WAV/PCM_16",
}
PROFILE_HASH = hashlib.sha256(
    json.dumps(PROFILE, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
SPLIT_MAP = {
    "expanded_v3_train": "expanded_v5_enhanced_novoa_train",
    "expanded_v3_dev": "expanded_v5_enhanced_novoa_dev",
    "expanded_v3_eval": "expanded_v5_enhanced_novoa_eval",
}


def is_voa(value: object) -> bool:
    """Return true for every known VOA source identifier."""
    normalized = str(value).casefold().replace("-", "_").replace(" ", "_")
    return "voa" in normalized or "voice_of_america" in normalized


def selection_score(identifier: str) -> str:
    return hashlib.sha256(
        f"expanded-v5-enhanced-novoa-777\0{identifier}".encode()
    ).hexdigest()


def smoke_subset(frame: pd.DataFrame) -> pd.DataFrame:
    limits = {"train": 256, "dev": 32, "eval": 32}
    parts = []
    for suffix, limit in limits.items():
        split = f"expanded_v5_enhanced_novoa_{suffix}"
        part = frame.loc[frame["split"] == split].copy()
        part["_selection_score"] = part["utterance_id"].astype(str).map(
            selection_score
        )
        parts.append(
            part.sort_values("_selection_score")
            .head(limit)
            .drop(columns="_selection_score")
        )
    return pd.concat(parts, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enhanced-manifest", type=Path, required=True)
    parser.add_argument("--pre-enhancement-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    args = parser.parse_args()

    enhanced = pd.read_parquet(args.enhanced_manifest)
    raw = pd.read_parquet(args.pre_enhancement_manifest)
    required = {
        "utterance_id",
        "source_id",
        "split",
        "speaker_id",
        "audio_path",
        "duration",
        "sample_rate",
    }
    missing = sorted(required - set(enhanced.columns))
    if missing:
        raise SystemExit(f"The enhanced manifest has no columns: {missing}")
    if enhanced["utterance_id"].duplicated().any():
        raise SystemExit("The enhanced manifest has duplicate utterance IDs.")
    if raw["utterance_id"].duplicated().any():
        raise SystemExit("The pre-enhancement manifest has duplicate utterance IDs.")

    voa_mask = enhanced["source_id"].map(is_voa)
    frame = enhanced.loc[~voa_mask].copy()
    unknown_splits = sorted(set(frame["split"]) - set(SPLIT_MAP))
    if unknown_splits:
        raise SystemExit(f"The manifest has unexpected splits: {unknown_splits}")
    frame["split"] = frame["split"].map(SPLIT_MAP)
    frame["preprocessing_profile"] = PROFILE["name"]
    frame["preprocessing_config_hash"] = PROFILE_HASH
    frame["deepfilternet_applied"] = True
    frame["highpass_applied"] = True
    frame["deessing_applied"] = True
    frame["compression_applied"] = True
    frame["loudness_normalization_applied"] = True
    frame["embedding_audio_variant"] = None
    frame["embedding_audio_path"] = None

    raw_index = raw.set_index(raw["utterance_id"].astype(str), drop=False)
    missing_raw = sorted(set(frame["utterance_id"].astype(str)) - set(raw_index.index))
    if missing_raw:
        raise SystemExit(f"Pre-enhancement audio is missing for IDs: {missing_raw[:10]}")
    raw_frame = raw_index.loc[frame["utterance_id"].astype(str)].reset_index(drop=True)
    raw_frame = raw_frame[["utterance_id", "audio_path"]].copy()

    missing_clean_paths = [
        str(path) for path in frame["audio_path"] if not Path(str(path)).is_file()
    ]
    missing_raw_paths = [
        str(path) for path in raw_frame["audio_path"] if not Path(str(path)).is_file()
    ]
    if missing_clean_paths or missing_raw_paths:
        raise SystemExit(
            "Audio files are missing: "
            f"enhanced={missing_clean_paths[:5]}, pre-enhancement={missing_raw_paths[:5]}"
        )
    if frame["sample_rate"].astype(int).ne(24000).any():
        raise SystemExit("The enhanced manifest has audio that is not 24 kHz.")

    if args.mode == "smoke":
        frame = smoke_subset(frame)
        raw_frame = raw_frame.set_index("utterance_id").loc[
            frame["utterance_id"]
        ].reset_index()

    frame = frame.sort_values("utterance_id").reset_index(drop=True)
    raw_frame = raw_frame.sort_values("utterance_id").reset_index(drop=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output_dir / "all.parquet", index=False)
    for split, split_frame in frame.groupby("split", sort=True):
        split_frame.to_parquet(args.output_dir / f"{split}.parquet", index=False)
    args.raw_manifest.parent.mkdir(parents=True, exist_ok=True)
    raw_frame.to_parquet(args.raw_manifest, index=False)

    split_report = {}
    for split, split_frame in frame.groupby("split", sort=True):
        split_report[str(split)] = {
            "records": int(len(split_frame)),
            "hours": round(float(split_frame["duration"].sum() / 3600), 6),
        }
    source_report = {}
    for source, source_frame in frame.groupby("source_id", sort=True):
        source_report[str(source)] = {
            "records": int(len(source_frame)),
            "hours": round(float(source_frame["duration"].sum() / 3600), 6),
        }
    report = {
        "status": "PASS",
        "mode": args.mode,
        "profile": PROFILE,
        "profile_hash": PROFILE_HASH,
        "input_records": int(len(enhanced)),
        "excluded_voa_records": int(voa_mask.sum()),
        "records": int(len(frame)),
        "hours": round(float(frame["duration"].sum() / 3600), 6),
        "splits": split_report,
        "sources": source_report,
        "voa_records_after_filter": int(frame["source_id"].map(is_voa).sum()),
        "enhanced_paths_missing": 0,
        "pre_enhancement_paths_missing": 0,
        "manifest": str((args.output_dir / "all.parquet").resolve()),
        "raw_manifest": str(args.raw_manifest.resolve()),
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
