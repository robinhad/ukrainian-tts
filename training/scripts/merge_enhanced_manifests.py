#!/usr/bin/env python3
"""Merge enhanced shards and create deterministic split manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--records-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    source = pd.read_parquet(args.source_manifest)
    rows = []
    files = sorted(args.records_dir.glob("records-*.jsonl"))
    if not files:
        raise SystemExit("No enhanced record shards exist.")
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    frame = pd.DataFrame(rows)
    duplicate_ids = int(frame["utterance_id"].duplicated().sum())
    if duplicate_ids:
        raise SystemExit(f"The enhanced records contain {duplicate_ids} duplicate IDs.")
    expected = set(source["utterance_id"])
    actual = set(frame["utterance_id"])
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    failures = frame[frame["enhancement_status"] != "ok"]
    if missing or extra or len(failures):
        report = {
            "expected_records": len(source),
            "actual_records": len(frame),
            "extra_ids": extra[:100],
            "failed_records": len(failures),
            "failure_examples": failures[
                ["utterance_id", "enhancement_error"]
            ].head(100).to_dict(orient="records"),
            "missing_ids": missing[:100],
            "status": "FAIL",
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        raise SystemExit("The enhanced record set is incomplete.")

    frame = frame.sort_values("utterance_id").reset_index(drop=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output_dir / "all.parquet", index=False)
    for split, split_frame in frame.groupby("split", sort=True):
        split_frame.to_parquet(args.output_dir / f"{split}.parquet", index=False)

    output_i = np.array(
        [item["second_pass"]["output_i"] for item in frame["loudness"]],
        dtype=np.float64,
    )
    output_tp = np.array(
        [item["second_pass"]["output_tp"] for item in frame["loudness"]],
        dtype=np.float64,
    )
    report = {
        "actual_records": len(frame),
        "enhancement_config_hashes": sorted(frame["enhancement_config_hash"].unique()),
        "expected_records": len(source),
        "loudness_output_i": {
            "maximum": float(np.max(output_i)),
            "median": float(np.median(output_i)),
            "minimum": float(np.min(output_i)),
            "outside_one_lu": int(np.sum(np.abs(output_i + 23.0) > 1.0)),
            "target_lufs": -23.0,
        },
        "maximum_true_peak_db": float(np.max(output_tp)),
        "splits": {
            str(split): int(count)
            for split, count in frame["split"].value_counts().sort_index().items()
        },
        "status": "PASS",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

