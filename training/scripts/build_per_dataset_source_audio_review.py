#!/usr/bin/env python3
"""Make a source-audio A/B review for two preprocessing revisions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from training.scripts.generate_per_dataset_listening_eval import (
    make_link,
    select_rows,
    validate_audio,
)


def pair_revisions(
    current: pd.DataFrame,
    previous: pd.DataFrame,
    count_per_dataset: int,
) -> pd.DataFrame:
    """Select current rows and add the matching previous audio path."""
    required = {"utterance_id", "audio_path"}
    missing = sorted(required - set(previous.columns))
    if missing:
        raise ValueError(f"previous manifest columns are missing: {missing}")
    if previous["utterance_id"].duplicated().any():
        raise ValueError("previous manifest has duplicate utterance IDs")
    selected = select_rows(current, count_per_dataset)
    previous_paths = previous[["utterance_id", "audio_path"]].rename(
        columns={"audio_path": "previous_audio_path"}
    )
    paired = selected.merge(
        previous_paths,
        on="utterance_id",
        how="left",
        validate="one_to_one",
    )
    missing_paths = paired.loc[
        paired["previous_audio_path"].isna(), "utterance_id"
    ].tolist()
    if missing_paths:
        raise ValueError(f"previous revision is missing IDs: {missing_paths}")
    return paired


def write_tsv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-manifest", type=Path, required=True)
    parser.add_argument("--current-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--count-per-dataset", type=int, default=10)
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"output directory is not empty: {args.output}")
    previous = pd.read_parquet(args.previous_manifest)
    current = pd.read_parquet(args.current_manifest)
    selected = pair_revisions(current, previous, args.count_per_dataset)
    args.output.mkdir(parents=True, exist_ok=True)

    errors = []
    records = []
    for row in selected.itertuples(index=False):
        dataset = str(row.source_id)
        index = int(row.listening_index)
        utterance_id = str(row.utterance_id)
        item_name = f"{index:02d}_{utterance_id}.wav"
        dataset_root = args.output / dataset
        previous_link = dataset_root / "previous_enhanced_v3" / item_name
        current_link = dataset_root / "current_trim_only_v4" / item_name
        make_link(Path(row.previous_audio_path), previous_link)
        make_link(Path(row.audio_path), current_link)
        previous_check = validate_audio(previous_link)
        current_check = validate_audio(current_link)
        item_errors = [
            *(f"previous: {value}" for value in previous_check["errors"]),
            *(f"current: {value}" for value in current_check["errors"]),
        ]
        if item_errors:
            errors.append({"utterance_id": utterance_id, "errors": item_errors})
        records.append(
            {
                "dataset": dataset,
                "item": f"{index:02d}",
                "utterance_id": utterance_id,
                "embedding_variant": str(row.embedding_audio_variant),
                "text": str(row.text_sanitized),
                "previous_enhanced_v3": str(previous_link),
                "current_trim_only_v4": str(current_link),
                "previous_check": previous_check,
                "current_check": current_check,
            }
        )

    manifest_fields = [
        "dataset",
        "item",
        "utterance_id",
        "embedding_variant",
        "text",
        "previous_enhanced_v3",
        "current_trim_only_v4",
    ]
    write_tsv(args.output / "manifest.tsv", records, manifest_fields)
    feedback_fields = [
        *manifest_fields,
        "previous_quality_1_to_5",
        "current_quality_1_to_5",
        "preferred_previous_or_current",
        "previous_metallic_yes_no",
        "current_metallic_yes_no",
        "previous_rasp_yes_no",
        "current_rasp_yes_no",
        "previous_noise_issue",
        "current_noise_issue",
        "comments",
    ]
    write_tsv(args.output / "feedback_template.tsv", records, feedback_fields)

    datasets = {}
    for dataset, group in selected.groupby("source_id", sort=True):
        dataset_records = [item for item in records if item["dataset"] == dataset]
        datasets[str(dataset)] = {
            "items": int(len(group)),
            "previous_clipping_warnings": sum(
                item["previous_check"]["possible_clipping"]
                for item in dataset_records
            ),
            "current_clipping_warnings": sum(
                item["current_check"]["possible_clipping"]
                for item in dataset_records
            ),
        }
    report = {
        "status": "PASS" if not errors else "FAIL",
        "mode": "source_audio_only",
        "contains_synthesis": False,
        "previous_profile": {
            "name": "expanded_v3_enhanced",
            "deepfilternet": True,
            "highpass": True,
            "deessing": True,
            "compression": True,
            "two_pass_ebu_r128": True,
        },
        "current_profile": {
            "name": "expanded_v4_trim_only",
            "boundary_trim": True,
            "deepfilternet": False,
            "highpass": False,
            "deessing": False,
            "compression": False,
            "two_pass_ebu_r128": False,
        },
        "previous_manifest": str(args.previous_manifest.resolve()),
        "current_manifest": str(args.current_manifest.resolve()),
        "output": str(args.output.resolve()),
        "dataset_count": len(datasets),
        "items_per_dataset": args.count_per_dataset,
        "pair_count": len(records),
        "datasets": datasets,
        "errors": errors,
        "records": records,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    readme = f"""# Source-Audio Processing Review

This document uses ASD-STE100 Simplified Technical English style. An approved
STE checker did not certify this document.

This set contains source audio only. It does not contain synthesized audio.
The set has {len(datasets)} datasets and {args.count_per_dataset} matched pairs
for each dataset.

Listen to `previous_enhanced_v3` first. This revision uses DeepFilterNet,
high-pass filtering, de-essing, compression, and two-pass EBU R128
normalization. Then listen to `current_trim_only_v4`. This revision uses only
boundary silence trim.

Use `manifest.tsv` to see the text and paths. Enter the listening results in
`feedback_template.tsv`. Use a value from 1 to 5 for quality. Record the
preferred revision and all audible problems.

The automatic validation status is `{report['status']}`. Automatic validation
does not measure naturalness or noise quality.
"""
    (args.output / "README.md").write_text(readme, encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "records"},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
