#!/usr/bin/env python3
"""Prepare and finalize the seven-way source-audio enhancement review."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from training.audio_enhancement.review_pipeline import (
    ReviewProcessingConfig,
    inspect_wav,
    sha256,
    write_json,
)


EXPECTED_DATASETS = {
    "common_voice_available_uk",
    "fleurs_uk",
    "opentts_lada",
    "opentts_mykyta",
    "opentts_tetiana",
    "tg_voices_uk",
    "ua_ser",
    "ukr_dialects",
    "voa_ukr_user_grant",
}
PROFILES = [
    "trim_only",
    "historical_dfn3_compressed",
    "dfn3_no_compression",
    "sidon_no_compression",
    "resemble_denoise_no_compression",
    "resemble_full_no_compression",
    "mossformer2_no_compression",
]


def select_rows(v3: pd.DataFrame, v4: pd.DataFrame, count: int) -> pd.DataFrame:
    if count < 2:
        raise ValueError("count must be at least 2")
    train = v4[v4["split"].astype(str).str.endswith("_train")].copy()
    previous = v3[["utterance_id", "audio_path"]].rename(
        columns={"audio_path": "historical_dfn_path"}
    )
    eligible = train.merge(previous, on="utterance_id", how="inner", validate="one_to_one")
    found = set(map(str, eligible["source_id"].unique()))
    if found != EXPECTED_DATASETS:
        raise ValueError(f"dataset mismatch: found={sorted(found)}")
    selected = []
    for dataset, frame in eligible.groupby("source_id", sort=True):
        existing = frame[
            frame["canonical_raw_audio_path"].map(lambda value: Path(value).is_file())
            & frame["audio_path"].map(lambda value: Path(value).is_file())
            & frame["historical_dfn_path"].map(lambda value: Path(value).is_file())
        ].sort_values(["duration", "utterance_id"], kind="stable")
        if len(existing) < count:
            raise ValueError(f"{dataset} has only {len(existing)} eligible records")
        indices = [round(i * (len(existing) - 1) / (count - 1)) for i in range(count)]
        subset = existing.iloc[indices].copy()
        subset["listening_index"] = range(1, count + 1)
        selected.append(subset)
    result = pd.concat(selected, ignore_index=True)
    if result["utterance_id"].duplicated().any():
        raise ValueError("the selected utterance IDs are not unique")
    return result


def copy_real_file(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    shutil.copy2(source, target)
    if target.is_symlink() or not target.is_file():
        raise RuntimeError(f"failed to make an independent copy: {target}")


def prepare(args: argparse.Namespace) -> int:
    selected = select_rows(
        pd.read_parquet(args.v3_manifest),
        pd.read_parquet(args.v4_manifest),
        args.count_per_dataset,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for row in selected.itertuples(index=False):
        dataset = str(row.source_id)
        index = int(row.listening_index)
        uid = str(row.utterance_id)
        name = f"{index:02d}_{uid}.wav"
        trim_path = args.output / dataset / "trim_only" / name
        historical_path = args.output / dataset / "historical_dfn3_compressed" / name
        copy_real_file(Path(row.audio_path), trim_path)
        copy_real_file(Path(row.historical_dfn_path), historical_path)
        record = {
            "dataset": dataset,
            "listening_index": index,
            "utterance_id": uid,
            "text": str(row.text_sanitized),
            "duration": float(row.duration),
            "raw_audio_path": str(Path(row.canonical_raw_audio_path).resolve()),
            "trim_only": str(trim_path.resolve()),
            "historical_dfn3_compressed": str(historical_path.resolve()),
        }
        for profile in ("trim_only", "historical_dfn3_compressed"):
            path = Path(record[profile])
            check = inspect_wav(path)
            write_json(
                path.with_suffix(".wav.json"),
                {
                    "status": "PASS" if not check["errors"] else "FAIL",
                    "profile": profile,
                    "compression_applied": profile == "historical_dfn3_compressed",
                    "source": str(path),
                    "audio_sha256": sha256(path),
                    "waveform": check,
                },
            )
        records.append(record)
    selection = args.output / "selection.jsonl"
    with selection.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "records": len(records), "selection": str(selection)}))
    return 0


def load_selection(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_tsv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def finalize(args: argparse.Namespace) -> int:
    config = ReviewProcessingConfig()
    records = load_selection(args.output / "selection.jsonl")
    rows = []
    errors = []
    profiles = {profile: {"files": 0, "errors": 0} for profile in PROFILES}
    for record in records:
        item = {
            "dataset": record["dataset"],
            "item": f"{record['listening_index']:02d}",
            "utterance_id": record["utterance_id"],
            "duration_seconds": f"{record['duration']:.3f}",
            "text": record["text"],
        }
        name = f"{record['listening_index']:02d}_{record['utterance_id']}.wav"
        for profile in PROFILES:
            path = args.output / record["dataset"] / profile / name
            item[profile] = str(path.relative_to(args.output))
            if not path.is_file() or path.is_symlink():
                errors.append({"profile": profile, "path": str(path), "error": "missing or symlink"})
                profiles[profile]["errors"] += 1
                continue
            check = inspect_wav(path)
            profiles[profile]["files"] += 1
            profiles[profile]["errors"] += len(check["errors"])
            for error in check["errors"]:
                errors.append({"profile": profile, "path": str(path), "error": error})
            metadata = path.with_suffix(".wav.json")
            if not metadata.is_file():
                errors.append({"profile": profile, "path": str(metadata), "error": "missing metadata"})
                profiles[profile]["errors"] += 1
                continue
            try:
                details = json.loads(metadata.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                errors.append(
                    {"profile": profile, "path": str(metadata), "error": f"invalid metadata: {error}"}
                )
                profiles[profile]["errors"] += 1
                continue
            expected_compression = profile == "historical_dfn3_compressed"
            if details.get("status") != "PASS":
                errors.append({"profile": profile, "path": str(metadata), "error": "metadata status is not PASS"})
                profiles[profile]["errors"] += 1
            if details.get("compression_applied") is not expected_compression:
                errors.append({"profile": profile, "path": str(metadata), "error": "compression policy mismatch"})
                profiles[profile]["errors"] += 1
            if profile not in {"trim_only", "historical_dfn3_compressed"}:
                if details.get("processing_config_hash") != config.digest:
                    errors.append({"profile": profile, "path": str(metadata), "error": "processing config mismatch"})
                    profiles[profile]["errors"] += 1
                second_pass = details.get("loudness", {}).get("second_pass", {})
                try:
                    output_i = float(second_pass["output_i"])
                    output_tp = float(second_pass["output_tp"])
                except (KeyError, TypeError, ValueError):
                    errors.append({"profile": profile, "path": str(metadata), "error": "missing loudness result"})
                    profiles[profile]["errors"] += 1
                else:
                    if abs(output_i - config.target_lufs) > 1.0:
                        errors.append({"profile": profile, "path": str(metadata), "error": f"loudness={output_i}"})
                        profiles[profile]["errors"] += 1
                    if output_tp > config.target_true_peak_db + 0.05:
                        errors.append({"profile": profile, "path": str(metadata), "error": f"true_peak={output_tp}"})
                        profiles[profile]["errors"] += 1
        rows.append(item)

    manifest_fields = [
        "dataset", "item", "utterance_id", "duration_seconds", "text", *PROFILES
    ]
    write_tsv(args.output / "manifest.tsv", rows, manifest_fields)
    feedback_fields = list(manifest_fields)
    for profile in PROFILES:
        feedback_fields.extend(
            [
                f"{profile}_quality_1_to_5",
                f"{profile}_metallic_yes_no",
                f"{profile}_rasp_yes_no",
                f"{profile}_robotic_yes_no",
                f"{profile}_speech_loss_yes_no",
                f"{profile}_noise_note",
                f"{profile}_pronunciation_change",
            ]
        )
    feedback_fields.extend(["preferred_profile", "comments"])
    write_tsv(args.output / "feedback_template.tsv", rows, feedback_fields)
    report = {
        "status": "PASS" if not errors else "FAIL",
        "mode": "matched_training_source_audio",
        "contains_synthesis": False,
        "uses_symlinks": False,
        "dataset_count": len(EXPECTED_DATASETS),
        "items_per_dataset": len(records) // len(EXPECTED_DATASETS),
        "utterance_count": len(records),
        "profile_count": len(PROFILES),
        "wav_count": sum(value["files"] for value in profiles.values()),
        "profiles": profiles,
        "compression_policy": {
            "historical_dfn3_compressed": True,
            "all_other_profiles": False,
        },
        "output": str(args.output.resolve()),
        "errors": errors,
    }
    write_json(args.report, report)
    readme = f"""# Audio Enhancement Listening Review

This document uses ASD-STE100 Simplified Technical English style. An approved
STE checker did not certify this document.

This set contains source audio only. It does not contain synthesized audio.
It has {len(EXPECTED_DATASETS)} datasets, {len(records)} matched utterances,
and {len(PROFILES)} processing profiles. Each WAV file is an independent copy.
The set does not use symbolic links.

Listen to all profiles for one item before you continue to the next item. The
historical DeepFilterNet3 profile contains compression. No other profile
contains compression. Enter the results in `feedback_template.tsv`.

Use a value from 1 to 5 for quality. Record metallic sound, rasp, robotic
sound, speech loss, remaining noise, and pronunciation changes. Select one
preferred profile for each item.

The automatic validation status is `{report['status']}`. Automatic validation
does not measure naturalness.
"""
    (args.output / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "errors"}, indent=2))
    return 0 if not errors else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "finalize"])
    parser.add_argument(
        "--v3-manifest",
        type=Path,
        default=REPOSITORY_ROOT / "training/data/expanded_v3/manifests/all.parquet",
    )
    parser.add_argument(
        "--v4-manifest",
        type=Path,
        default=REPOSITORY_ROOT / "training/data/expanded_v4_trim_only/manifests/all.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "training/eval/generated/per_dataset_train_audio_enhancement_review_v2",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPOSITORY_ROOT / "training/reports/per_dataset_train_audio_enhancement_review_v2.json",
    )
    parser.add_argument("--count-per-dataset", type=int, default=10)
    args = parser.parse_args()
    return prepare(args) if args.stage == "prepare" else finalize(args)


if __name__ == "__main__":
    raise SystemExit(main())
