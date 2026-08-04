#!/usr/bin/env python3
"""Build the versioned trim-only corpus from the accepted expanded-v3 IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROFILE = {
    "name": "expanded_v4_trim_only_v1",
    "boundary_trim": True,
    "deepfilternet": False,
    "highpass": False,
    "deessing": False,
    "compression": False,
    "loudness_normalization": False,
    "sample_rate": 24000,
    "channels": 1,
    "format": "WAV/PCM_16",
}
PROFILE_HASH = hashlib.sha256(
    json.dumps(PROFILE, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


def flags(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value} if value else set()
    try:
        return {str(item) for item in value}
    except TypeError:
        return set()


def selection_score(identifier: str) -> str:
    return hashlib.sha256(f"expanded-v4-trim-only-777\0{identifier}".encode()).hexdigest()


def smoke_subset(frame: pd.DataFrame) -> pd.DataFrame:
    limits = {"train": 256, "dev": 32, "eval": 32}
    parts = []
    for suffix, limit in limits.items():
        split = f"expanded_v4_trim_only_{suffix}"
        part = frame.loc[frame["split"] == split].copy()
        part["_selection_score"] = [selection_score(str(x)) for x in part["utterance_id"]]
        parts.append(part.sort_values("_selection_score").head(limit).drop(columns="_selection_score"))
    return pd.concat(parts, ignore_index=True).sort_values("utterance_id")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-manifest", type=Path, required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--minimum-duration", type=float, default=2.0)
    parser.add_argument("--maximum-duration", type=float, default=20.0)
    args = parser.parse_args()

    active = pd.read_parquet(args.active_manifest).sort_values("utterance_id")
    canonical = pd.read_parquet(args.canonical_manifest).sort_values("utterance_id")
    if active["utterance_id"].duplicated().any() or canonical["utterance_id"].duplicated().any():
        raise SystemExit("An input manifest has duplicate utterance IDs.")
    if "canonical_raw_audio_path" not in active:
        raise SystemExit("The active manifest has no original pre-trim path.")

    canonical = canonical.set_index("utterance_id", drop=False)
    identifiers = active["utterance_id"].astype(str)
    missing = sorted(set(identifiers) - set(canonical.index.astype(str)))
    if missing:
        raise SystemExit(f"Canonical audio is missing for IDs: {missing[:10]}")
    selected = canonical.loc[identifiers].reset_index(drop=True)

    frame = active.reset_index(drop=True).copy()
    audio_fields = (
        "audio_path",
        "duration",
        "sample_rate",
        "audio_sha256",
        "qc_flags",
        "trim_applied",
        "duration_before_trim",
        "trim_start_seconds",
        "trim_end_seconds",
        "trim_removed_seconds",
        "trim_threshold_rms",
        "trim_config_hash",
    )
    for field in audio_fields:
        if field in selected:
            frame[field] = selected[field].to_numpy()

    parsed_flags = frame["qc_flags"].map(flags)
    clipped = parsed_flags.map(lambda item: "clipping" in item)
    too_short = parsed_flags.map(lambda item: "too_short" in item) | (
        frame["duration"].astype(float) < args.minimum_duration
    )
    too_long = frame["duration"].astype(float) > args.maximum_duration
    rejected = clipped | too_short | too_long
    rejected_frame = frame.loc[rejected].copy()
    frame = frame.loc[~rejected].copy()
    frame["qc_flags"] = [[] for _ in range(len(frame))]

    split_map = {
        "expanded_v3_train": "expanded_v4_trim_only_train",
        "expanded_v3_dev": "expanded_v4_trim_only_dev",
        "expanded_v3_eval": "expanded_v4_trim_only_eval",
    }
    unknown_splits = sorted(set(frame["split"]) - set(split_map))
    if unknown_splits:
        raise SystemExit(f"Unexpected input splits: {unknown_splits}")
    frame["split"] = frame["split"].map(split_map)
    frame["preprocessing_profile"] = PROFILE["name"]
    frame["preprocessing_config_hash"] = PROFILE_HASH
    for key in (
        "deepfilternet",
        "highpass",
        "deessing",
        "compression",
        "loudness_normalization",
    ):
        frame[f"{key}_applied"] = False
    frame["embedding_audio_variant"] = None
    frame["embedding_audio_path"] = None

    missing_clean = [path for path in frame["audio_path"] if not Path(str(path)).is_file()]
    missing_raw = [
        path for path in frame["canonical_raw_audio_path"] if not Path(str(path)).is_file()
    ]
    if missing_clean or missing_raw:
        raise SystemExit(
            f"Audio paths are missing: clean={missing_clean[:5]}, raw={missing_raw[:5]}"
        )

    if args.mode == "smoke":
        frame = smoke_subset(frame)

    frame = frame.sort_values("utterance_id").reset_index(drop=True)
    raw = frame[["utterance_id", "canonical_raw_audio_path"]].rename(
        columns={"canonical_raw_audio_path": "audio_path"}
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output_dir / "all.parquet", index=False)
    for split, split_frame in frame.groupby("split", sort=True):
        split_frame.to_parquet(args.output_dir / f"{split}.parquet", index=False)
    args.raw_manifest.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(args.raw_manifest, index=False)

    report = {
        "status": "PASS",
        "mode": args.mode,
        "profile": PROFILE,
        "profile_hash": PROFILE_HASH,
        "input_records": int(len(active)),
        "records": int(len(frame)),
        "duration_hours": float(frame["duration"].sum() / 3600),
        "splits": {str(k): int(v) for k, v in frame["split"].value_counts().sort_index().items()},
        "rejected": {
            "clipping": int(clipped.sum()),
            "too_short": int(too_short.sum()),
            "too_long": int(too_long.sum()),
            "unique_records": int(len(rejected_frame)),
        },
        "clean_paths_missing": 0,
        "raw_paths_missing": 0,
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
