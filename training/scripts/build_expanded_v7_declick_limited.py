#!/usr/bin/env python3
"""Build v7 manifests from the v6 audio and the final-stage batch report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

EXPECTED_RECORDS = 74_156
V6_NAME = "expanded_v6_sidon_deess_novoa"
V7_NAME = "expanded_v7_sidon_deess_declick_limit_novoa"
SPLIT_MAP = {
    f"{V6_NAME}_train": f"{V7_NAME}_train",
    f"{V6_NAME}_dev": f"{V7_NAME}_dev",
    f"{V6_NAME}_eval": f"{V7_NAME}_eval",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def profile(report: dict[str, Any], source_hash: str) -> dict[str, Any]:
    results = report.get("results", [])
    if not results:
        raise SystemExit("The final-stage report has no successful result.")
    first = results[0]
    return {
        "name": V7_NAME,
        "source_dataset": V6_NAME,
        "source_manifest_sha256": source_hash,
        "boundary_trim": True,
        "sidon": True,
        "deessing": True,
        "declicking": True,
        "post_highpass": False,
        "loudness_normalization": False,
        "compression": False,
        "limiting": True,
        "sample_rate": 24_000,
        "channels": 1,
        "format": "WAV/PCM_24",
        "final_filter_chain": report["filter_chain"],
        "final_processing_config_hash": report["processing_config_hash"],
        "final_processing_config": first["processing_config"],
        "ffmpeg_version": first["ffmpeg_version"],
        "source_preserved": all(
            bool(result.get("source_preserved")) for result in results
        ),
        "atomic_write": all(bool(result.get("atomic_write")) for result in results),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--batch-report", type=Path, required=True)
    parser.add_argument("--source-audio-root", type=Path, required=True)
    parser.add_argument("--output-audio-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--source-raw-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    base = pd.read_parquet(args.base_manifest)
    if len(base) != EXPECTED_RECORDS or base["utterance_id"].duplicated().any():
        raise SystemExit("The v6 manifest does not have 74,156 unique records.")
    batch = json.loads(args.batch_report.read_text(encoding="utf-8"))
    if (
        batch.get("status") != "PASS"
        or batch.get("completed_files") != EXPECTED_RECORDS
        or batch.get("failed_files") != 0
    ):
        raise SystemExit("The final-stage batch report is not complete and PASS.")
    results = batch.get("results", [])
    if len(results) != EXPECTED_RECORDS:
        raise SystemExit("The final-stage batch result count is incorrect.")
    by_source = {str(Path(item["source"]).resolve()): item for item in results}
    if len(by_source) != EXPECTED_RECORDS:
        raise SystemExit("The final-stage batch report has duplicate sources.")

    source_root = args.source_audio_root.resolve()
    output_root = args.output_audio_root.resolve()
    expected_sources = {str(Path(path).resolve()) for path in base["audio_path"]}
    if set(by_source) != expected_sources:
        raise SystemExit("The final-stage sources do not match the v6 manifest.")

    output_paths: list[str] = []
    output_hashes: list[str] = []
    durations: list[float] = []
    for row in base.itertuples(index=False):
        source = Path(str(row.audio_path)).resolve()
        try:
            relative = source.relative_to(source_root)
        except ValueError as error:
            raise SystemExit(
                f"A v6 file is outside its audio root: {source}"
            ) from error
        expected_output = (output_root / relative).resolve()
        result = by_source[str(source)]
        output = Path(result["output"]).resolve()
        output_audio = result["output_audio"]
        if output != expected_output or not output.is_file() or output.is_symlink():
            raise SystemExit(f"The v7 output is missing or invalid: {output}")
        if (
            int(output_audio["sample_rate"]) != 24_000
            or int(output_audio["channels"]) != 1
            or output_audio["subtype"] != "PCM_24"
        ):
            raise SystemExit(f"The v7 stream properties are invalid: {output}")
        if result.get("source_preserved") is not True:
            raise SystemExit(f"The source-preservation check failed: {source}")
        output_paths.append(str(output))
        output_hashes.append(str(result["output_sha256"]))
        durations.append(float(output_audio["duration"]))

    source_manifest_hash = sha256(args.base_manifest)
    processing_profile = profile(batch, source_manifest_hash)
    profile_hash = hashlib.sha256(
        json.dumps(processing_profile, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    frame = base.copy()
    unknown_splits = sorted(set(frame["split"]) - set(SPLIT_MAP))
    if unknown_splits:
        raise SystemExit(f"The v6 manifest has unknown splits: {unknown_splits}")
    frame["split"] = frame["split"].map(SPLIT_MAP)
    frame["audio_path"] = output_paths
    frame["audio_sha256"] = output_hashes
    frame["duration"] = durations
    frame["sample_rate"] = 24_000
    frame["channels"] = 1
    frame["format"] = "WAV/PCM_24"
    frame["processing_attempts"] = 1
    frame["preprocessing_profile"] = processing_profile["name"]
    frame["preprocessing_config_hash"] = profile_hash
    frame["declicking_applied"] = True
    frame["limiting_applied"] = True
    frame["embedding_audio_variant"] = None
    frame["embedding_audio_path"] = None
    if frame["source_id"].astype(str).str.contains("voa", case=False).any():
        raise SystemExit("The v7 manifest contains VOA records.")

    frame = frame.sort_values("utterance_id").reset_index(drop=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output_dir / "all.parquet", index=False)
    for split, part in frame.groupby("split", sort=True):
        part.to_parquet(args.output_dir / f"{split}.parquet", index=False)
    raw = pd.read_parquet(args.source_raw_manifest).sort_values("utterance_id")
    if len(raw) != EXPECTED_RECORDS or raw["utterance_id"].duplicated().any():
        raise SystemExit("The raw embedding manifest is invalid.")
    args.raw_manifest.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(args.raw_manifest, index=False)

    splits = {
        str(split): {
            "records": int(len(part)),
            "hours": round(float(part["duration"].sum() / 3600), 6),
        }
        for split, part in frame.groupby("split", sort=True)
    }
    sources = {
        str(source): {
            "records": int(len(part)),
            "hours": round(float(part["duration"].sum() / 3600), 6),
        }
        for source, part in frame.groupby("source_id", sort=True)
    }
    result = {
        "status": "PASS",
        "records": int(len(frame)),
        "hours": round(float(frame["duration"].sum() / 3600), 6),
        "profile": processing_profile,
        "profile_hash": profile_hash,
        "splits": splits,
        "sources": sources,
        "voa_records": 0,
        "manifest": str((args.output_dir / "all.parquet").resolve()),
        "raw_manifest": str(args.raw_manifest.resolve()),
        "batch_report": str(args.batch_report.resolve()),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
