#!/usr/bin/env python3
"""Build the non-VOA Sidon and de-essing training manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from training.audio_enhancement.review_pipeline import SidonDeessOnlyConfig
from training.scripts.run_enhancement_review_backend import (
    SIDON_CODE_COMMIT,
    SIDON_MODEL_REVISION,
    W2V_BERT_MODEL_REVISION,
)

EXPECTED_RECORDS = 74_156
PROFILE = {
    "name": "expanded_v6_sidon_deess_only",
    "boundary_trim": True,
    "sidon": True,
    "sidon_code_commit": SIDON_CODE_COMMIT,
    "sidon_model_revision": SIDON_MODEL_REVISION,
    "feature_preprocessor_revision": W2V_BERT_MODEL_REVISION,
    "sidon_internal_input_highpass_hz": 50,
    "deessing": True,
    "declicking": True,
    "post_highpass": False,
    "loudness_normalization": False,
    "compression": False,
    "limiting": True,
    "sample_rate": 24_000,
    "channels": 1,
    "format": "WAV/PCM_24",
    "processing_config": asdict(SidonDeessOnlyConfig()),
}
PROFILE_HASH = hashlib.sha256(
    json.dumps(PROFILE, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
SPLIT_MAP = {
    "expanded_v5_enhanced_novoa_train": "expanded_v6_sidon_deess_novoa_train",
    "expanded_v5_enhanced_novoa_dev": "expanded_v6_sidon_deess_novoa_dev",
    "expanded_v5_enhanced_novoa_eval": "expanded_v6_sidon_deess_novoa_eval",
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
    parser.add_argument("--input-audio-manifest", type=Path, required=True)
    parser.add_argument("--processing-results", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    base = pd.read_parquet(args.base_manifest)
    raw = pd.read_parquet(args.input_audio_manifest)
    results = load_results(args.processing_results)
    for name, frame in (("base", base), ("raw", raw), ("results", results)):
        if "utterance_id" not in frame or frame["utterance_id"].duplicated().any():
            raise SystemExit(f"The {name} data has invalid utterance IDs.")
    if len(base) != EXPECTED_RECORDS:
        raise SystemExit(f"The base manifest has {len(base)} records.")
    expected = set(base["utterance_id"].astype(str))
    if set(results["utterance_id"].astype(str)) != expected:
        raise SystemExit("The processing results do not exactly cover the corpus.")
    if set(raw["utterance_id"].astype(str)) != expected:
        raise SystemExit("The input-audio manifest does not exactly cover the corpus.")
    if set(results["processing_status"].astype(str)) != {"ok"}:
        failed = (
            results.loc[
                results["processing_status"].astype(str) != "ok", "utterance_id"
            ]
            .astype(str)
            .head(10)
            .tolist()
        )
        raise SystemExit(f"Sidon processing has failed records: {failed}")
    if (
        results["processing_config_hash"]
        .astype(str)
        .ne(SidonDeessOnlyConfig().digest)
        .any()
    ):
        raise SystemExit("The processing results have a config-hash mismatch.")

    result_columns = [
        "utterance_id",
        "audio_path",
        "audio_sha256",
        "duration",
        "sample_rate",
        "channels",
        "format",
        "processing_attempts",
    ]
    frame = base.drop(
        columns=[name for name in result_columns[1:] if name in base],
        errors="ignore",
    ).merge(results[result_columns], on="utterance_id", validate="one_to_one")
    unknown_splits = sorted(set(frame["split"]) - set(SPLIT_MAP))
    if unknown_splits:
        raise SystemExit(f"The base manifest has unexpected splits: {unknown_splits}")
    frame["split"] = frame["split"].map(SPLIT_MAP)
    frame["qc_flags"] = [[] for _ in range(len(frame))]
    frame["preprocessing_profile"] = PROFILE["name"]
    frame["preprocessing_config_hash"] = PROFILE_HASH
    frame["deepfilternet_applied"] = False
    frame["sidon_applied"] = True
    frame["highpass_applied"] = False
    frame["post_highpass_applied"] = False
    frame["deessing_applied"] = True
    frame["declicking_applied"] = True
    frame["compression_applied"] = False
    frame["loudness_normalization_applied"] = False
    frame["limiting_applied"] = True
    frame["embedding_audio_variant"] = None
    frame["embedding_audio_path"] = None

    missing = [path for path in frame["audio_path"] if not Path(str(path)).is_file()]
    if missing:
        raise SystemExit(f"Processed WAV files are missing: {missing[:10]}")
    if frame["sample_rate"].astype(int).ne(24_000).any():
        raise SystemExit("The processed corpus has a non-24-kHz record.")
    if frame["channels"].astype(int).ne(1).any():
        raise SystemExit("The processed corpus has a non-mono record.")
    if frame["source_id"].astype(str).str.contains("voa", case=False).any():
        raise SystemExit("The processed corpus contains VOA records.")

    frame = frame.sort_values("utterance_id").reset_index(drop=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output_dir / "all.parquet", index=False)
    for split, split_frame in frame.groupby("split", sort=True):
        split_frame.to_parquet(args.output_dir / f"{split}.parquet", index=False)
    raw = raw.sort_values("utterance_id").reset_index(drop=True)
    args.raw_manifest.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(args.raw_manifest, index=False)

    split_report = {
        str(split): {
            "records": int(len(part)),
            "hours": round(float(part["duration"].sum() / 3600), 6),
        }
        for split, part in frame.groupby("split", sort=True)
    }
    source_report = {
        str(source): {
            "records": int(len(part)),
            "hours": round(float(part["duration"].sum() / 3600), 6),
        }
        for source, part in frame.groupby("source_id", sort=True)
    }
    report = {
        "status": "PASS",
        "records": int(len(frame)),
        "hours": round(float(frame["duration"].sum() / 3600), 6),
        "profile": PROFILE,
        "profile_hash": PROFILE_HASH,
        "splits": split_report,
        "sources": source_report,
        "voa_records": 0,
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
