#!/usr/bin/env python3
"""Audit the enhanced non-VOA corpus before v5 training."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

from training.scripts.build_expanded_v5_enhanced_novoa import is_voa


EXPECTED_RECORDS = 74_156
EXPECTED_HOURS = 92.2657368
EXPECTED_SPLITS = {
    "expanded_v5_enhanced_novoa_train": 71_334,
    "expanded_v5_enhanced_novoa_dev": 1_404,
    "expanded_v5_enhanced_novoa_eval": 1_418,
}
EXPECTED_TOKEN_SHA256 = (
    "dc4ea2634513b87e2e70b27d9545f5dd14cea91dd99391f615a24ded5ee964f2"
)


def load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def gate(name: str, passed: bool, evidence: str, artifact: Path | None) -> dict:
    return {
        "gate": name,
        "status": "PASS" if passed else "FAIL",
        "evidence": evidence,
        "artifact": str(artifact.resolve()) if artifact and artifact.exists() else None,
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

    manifest_path = args.data_root / "manifests" / "all.parquet"
    hybrid_path = args.data_root / "hybrid_manifest" / "all.parquet"
    build_path = args.reports_root / "expanded_v5_enhanced_novoa_full_dataset.json"
    validation_path = (
        args.reports_root / "expanded_v5_enhanced_novoa_full_validation.json"
    )
    embedding_path = (
        args.reports_root
        / "expanded_v5_enhanced_novoa_full_hybrid_embeddings.json"
    )
    token_path = (
        args.dump_dir
        / "token_list"
        / "phn_espeak_ng_ukrainian"
        / "tokens.txt"
    )
    stats_path = args.exp_dir / "tts_stats_raw_phn_espeak_ng_ukrainian"
    build = load(build_path)
    validation = load(validation_path)
    embedding = load(embedding_path)
    frame = pd.read_parquet(manifest_path) if manifest_path.is_file() else pd.DataFrame()
    hybrid = pd.read_parquet(hybrid_path) if hybrid_path.is_file() else pd.DataFrame()
    free_gib = shutil.disk_usage(args.workspace).free / 1024**3

    counts = frame["split"].value_counts().to_dict() if len(frame) else {}
    hours = float(frame["duration"].sum() / 3600) if len(frame) else 0.0
    profile_columns = (
        "deepfilternet_applied",
        "highpass_applied",
        "deessing_applied",
        "compression_applied",
        "loudness_normalization_applied",
    )
    profile_ok = bool(len(frame)) and all(
        name in frame and frame[name].fillna(False).astype(bool).all()
        for name in profile_columns
    )
    voa_records = (
        int(frame["source_id"].map(is_voa).sum()) if "source_id" in frame else -1
    )
    embedding_counts = embedding.get("counts", {}) if embedding else {}
    archive_records = sum(
        int(row.get("records", 0))
        for row in (embedding.get("archives", {}) if embedding else {}).values()
    )
    token_hash = (
        hashlib.sha256(token_path.read_bytes()).hexdigest()
        if token_path.is_file()
        else None
    )
    stats_files = list(stats_path.rglob("*.npz")) if stats_path.is_dir() else []

    gates = [
        gate(
            "Disk reserve",
            free_gib > 30,
            f"{free_gib:.2f} GiB is free. The stop threshold is 30 GiB.",
            args.workspace,
        ),
        gate(
            "Dataset build",
            bool(build and build.get("status") == "PASS"),
            f"The build report has {build.get('records') if build else 0} records.",
            build_path,
        ),
        gate(
            "Expected corpus",
            len(frame) == EXPECTED_RECORDS
            and abs(hours - EXPECTED_HOURS) < 1e-4
            and counts == EXPECTED_SPLITS,
            f"The corpus has {len(frame)} records and {hours:.6f} hours.",
            manifest_path,
        ),
        gate(
            "VOA exclusion",
            voa_records == 0,
            f"The corpus has {voa_records} VOA records.",
            manifest_path,
        ),
        gate(
            "Enhanced v3 profile",
            profile_ok,
            "All retained rows use the enhanced v3 processing profile.",
            manifest_path,
        ),
        gate(
            "Dataset validation",
            bool(validation and validation.get("status") == "PASS"),
            "The dataset validator report must have PASS status.",
            validation_path,
        ),
        gate(
            "Exact 50/50 embeddings",
            bool(
                embedding
                and embedding.get("status") == "PASS"
                and embedding_counts == {"clean": 37_078, "raw": 37_078}
                and archive_records == EXPECTED_RECORDS
                and len(hybrid) == EXPECTED_RECORDS
            ),
            f"Embedding counts are {embedding_counts}. Archive records: {archive_records}.",
            embedding_path,
        ),
        gate(
            "Token list",
            token_hash == EXPECTED_TOKEN_SHA256,
            f"The token-list SHA-256 is {token_hash}.",
            token_path,
        ),
        gate(
            "Speech, pitch, and energy statistics",
            len(stats_files) >= 6,
            f"The statistics directory has {len(stats_files)} NPZ files.",
            stats_path,
        ),
    ]
    status = "PASS" if all(item["status"] == "PASS" for item in gates) else "FAIL"
    report = {
        "status": status,
        "manifest_records": int(len(frame)),
        "duration_hours": round(hours, 7),
        "voa_records": voa_records,
        "free_disk_gib": round(free_gib, 2),
        "gates": gates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
