#!/usr/bin/env python3
"""Audit the VOA-inclusive v9 corpus before scratch training."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

NAME = "expanded_v9_cascade_with_voa"
EXPECTED_RECORDS = 208_867
EXPECTED_VOA_RECORDS = 134_711
EXPECTED_SPLITS = {
    f"{NAME}_train": 206_045,
    f"{NAME}_dev": 1_404,
    f"{NAME}_eval": 1_418,
}
EXPECTED_EMBEDDINGS = {"raw": 104_433, "clean": 104_434}


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def gate(name: str, passed: bool, evidence: str, artifact: Path) -> dict:
    return {
        "gate": name,
        "status": "PASS" if passed else "FAIL",
        "evidence": evidence,
        "artifact": str(artifact.resolve()) if artifact.exists() else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--exp-dir", type=Path, required=True)
    parser.add_argument("--reports-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = args.data_root / "manifests/all.parquet"
    hybrid_path = args.data_root / "hybrid_manifest/all.parquet"
    build_path = args.reports_root / f"{NAME}_full_dataset.json"
    validation_path = args.reports_root / f"{NAME}_full_validation.json"
    embedding_path = args.reports_root / f"{NAME}_full_hybrid_embeddings.json"
    token_path = args.dump_dir / "token_list/phn_espeak_ng_ukrainian/tokens.txt"
    stats_path = args.exp_dir / "tts_stats_raw_phn_espeak_ng_ukrainian"
    frame = pd.read_parquet(manifest) if manifest.is_file() else pd.DataFrame()
    hybrid = pd.read_parquet(hybrid_path) if hybrid_path.is_file() else pd.DataFrame()
    build = load(build_path)
    validation = load(validation_path)
    embedding = load(embedding_path)
    counts = frame["split"].value_counts().to_dict() if len(frame) else {}
    hours = float(frame["duration"].sum() / 3600) if len(frame) else 0
    free = shutil.disk_usage(args.workspace).free / 1024**3
    required_true = [
        "clearervoice_applied",
        "sidon_applied",
        "deessing_applied",
        "declicking_applied",
        "deepfilternet_applied",
        "rnnoise_applied",
        "loudness_normalization_applied",
        "limiting_applied",
    ]
    profile_ok = bool(len(frame)) and all(
        frame[column].fillna(False).astype(bool).all() for column in required_true
    ) and all(
        [
            not frame["post_highpass_applied"].fillna(False).astype(bool).any(),
            not frame["compression_applied"].fillna(False).astype(bool).any(),
            frame["loudness_normalization_type"]
            .astype(str)
            .eq("linear_source_match")
            .all(),
            frame["sample_rate"].astype(int).eq(24_000).all(),
            frame["channels"].astype(int).eq(1).all(),
            frame["format"].astype(str).eq("WAV/PCM_24").all(),
        ]
    )
    emb_counts = embedding.get("counts", {}) if embedding else {}
    archive_records = sum(
        int(item.get("records", 0))
        for item in (embedding or {}).get("archives", {}).values()
    )
    token_hash = (
        hashlib.sha256(token_path.read_bytes()).hexdigest()
        if token_path.is_file()
        else None
    )
    token_count = (
        len([line for line in token_path.read_text().splitlines() if line.strip()])
        if token_path.is_file()
        else 0
    )
    stats_files = list(stats_path.rglob("*.npz")) if stats_path.is_dir() else []
    voa = (
        int(frame["source_id"].astype(str).str.contains("voa", case=False).sum())
        if len(frame)
        else -1
    )
    fallback_count = int(
        frame.get("degenerate_output_fallback", pd.Series(dtype=bool))
        .fillna(False)
        .astype(bool)
        .sum()
    )
    gates = [
        gate(
            "Disk reserve",
            free > 30,
            f"{free:.2f} GiB is free. The threshold is 30 GiB.",
            args.workspace,
        ),
        gate(
            "Dataset build",
            bool(build and build.get("status") == "PASS"),
            "The build report must have PASS status.",
            build_path,
        ),
        gate(
            "Expected corpus",
            len(frame) == EXPECTED_RECORDS
            and counts == EXPECTED_SPLITS
            and 468 < hours < 472,
            f"The corpus has {len(frame)} records and {hours:.6f} hours.",
            manifest,
        ),
        gate(
            "VOA inclusion",
            voa == EXPECTED_VOA_RECORDS,
            f"The corpus has {voa} VOA records.",
            manifest,
        ),
        gate(
            "Processing profile and PCM24",
            profile_ok,
            "The requested cascade and final PCM24 format must be present.",
            manifest,
        ),
        gate(
            "Degenerate-output fallback limit",
            0 <= fallback_count <= 500,
            f"The corpus has {fallback_count} recorded fallback files.",
            manifest,
        ),
        gate(
            "Dataset validation",
            bool(validation and validation.get("status") == "PASS"),
            "The validator must have PASS status.",
            validation_path,
        ),
        gate(
            "Exact 50/50 embeddings",
            bool(
                embedding
                and embedding.get("status") == "PASS"
                and emb_counts == EXPECTED_EMBEDDINGS
                and archive_records == EXPECTED_RECORDS
                and len(hybrid) == EXPECTED_RECORDS
            ),
            f"Embedding counts are {emb_counts}. Archive records: {archive_records}.",
            embedding_path,
        ),
        gate(
            "Fresh token list",
            token_count >= 80,
            f"The token list has {token_count} entries and SHA-256 {token_hash}.",
            token_path,
        ),
        gate(
            "Fresh speech, pitch, and energy statistics",
            len(stats_files) >= 6,
            f"The statistics directory has {len(stats_files)} NPZ files.",
            stats_path,
        ),
    ]
    status = "PASS" if all(item["status"] == "PASS" for item in gates) else "FAIL"
    report = {
        "status": status,
        "manifest_records": len(frame),
        "duration_hours": round(hours, 7),
        "voa_records": voa,
        "embedding_counts": emb_counts,
        "degenerate_output_fallback_records": fallback_count,
        "free_disk_gib": round(free, 2),
        "token_sha256": token_hash,
        "gates": gates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
