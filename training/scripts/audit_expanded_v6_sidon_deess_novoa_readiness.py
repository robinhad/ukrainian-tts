#!/usr/bin/env python3
"""Audit the Sidon and de-essing corpus before v6 training."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

EXPECTED_RECORDS = 74_156
EXPECTED_SPLITS = {
    "expanded_v6_sidon_deess_novoa_train": 71_334,
    "expanded_v6_sidon_deess_novoa_dev": 1_404,
    "expanded_v6_sidon_deess_novoa_eval": 1_418,
}
EXPECTED_TOKEN_SHA256 = "dc4ea2634513b87e2e70b27d9545f5dd14cea91dd99391f615a24ded5ee964f2"


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def gate(name: str, passed: bool, evidence: str, artifact: Path) -> dict:
    return {"gate": name, "status": "PASS" if passed else "FAIL", "evidence": evidence,
            "artifact": str(artifact.resolve()) if artifact.exists() else None}


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
    build_path = args.reports_root / "expanded_v6_sidon_deess_novoa_full_dataset.json"
    validation_path = args.reports_root / "expanded_v6_sidon_deess_novoa_full_validation.json"
    embedding_path = args.reports_root / "expanded_v6_sidon_deess_novoa_full_hybrid_embeddings.json"
    token_path = args.dump_dir / "token_list/phn_espeak_ng_ukrainian/tokens.txt"
    stats_path = args.exp_dir / "tts_stats_raw_phn_espeak_ng_ukrainian"
    frame = pd.read_parquet(manifest) if manifest.is_file() else pd.DataFrame()
    hybrid = pd.read_parquet(hybrid_path) if hybrid_path.is_file() else pd.DataFrame()
    build, validation, embedding = load(build_path), load(validation_path), load(embedding_path)
    counts = frame["split"].value_counts().to_dict() if len(frame) else {}
    hours = float(frame["duration"].sum() / 3600) if len(frame) else 0
    free = shutil.disk_usage(args.workspace).free / 1024**3
    profile_ok = bool(len(frame)) and all([
        frame["sidon_applied"].fillna(False).astype(bool).all(),
        frame["deessing_applied"].fillna(False).astype(bool).all(),
        not frame["deepfilternet_applied"].fillna(False).astype(bool).any(),
        not frame["post_highpass_applied"].fillna(False).astype(bool).any(),
        not frame["compression_applied"].fillna(False).astype(bool).any(),
        not frame["loudness_normalization_applied"].fillna(False).astype(bool).any(),
        not frame["limiting_applied"].fillna(False).astype(bool).any(),
    ])
    emb_counts = embedding.get("counts", {}) if embedding else {}
    archive_records = sum(int(x.get("records", 0)) for x in (embedding or {}).get("archives", {}).values())
    token_hash = hashlib.sha256(token_path.read_bytes()).hexdigest() if token_path.is_file() else None
    stats_files = list(stats_path.rglob("*.npz")) if stats_path.is_dir() else []
    voa = int(frame["source_id"].astype(str).str.contains("voa", case=False).sum()) if len(frame) else -1
    gates = [
        gate("Disk reserve", free > 30, f"{free:.2f} GiB is free. The threshold is 30 GiB.", args.workspace),
        gate("Dataset build", bool(build and build.get("status") == "PASS"), "The build report must have PASS status.", build_path),
        gate("Expected corpus", len(frame) == EXPECTED_RECORDS and counts == EXPECTED_SPLITS and 90 < hours < 94, f"The corpus has {len(frame)} records and {hours:.6f} hours.", manifest),
        gate("VOA exclusion", voa == 0, f"The corpus has {voa} VOA records.", manifest),
        gate("Sidon and de-essing profile", profile_ok, "Sidon and light de-essing are on. Other enhancement steps are off.", manifest),
        gate("Dataset validation", bool(validation and validation.get("status") == "PASS"), "The validator must have PASS status.", validation_path),
        gate("Exact 50/50 embeddings", bool(embedding and embedding.get("status") == "PASS" and emb_counts == {"clean": 37_078, "raw": 37_078} and embedding.get("reused_raw_vectors") == 37_078 and embedding.get("recomputed_clean_vectors") == 37_078 and archive_records == EXPECTED_RECORDS and len(hybrid) == EXPECTED_RECORDS), f"Embedding counts are {emb_counts}. Archive records: {archive_records}.", embedding_path),
        gate("Token list", token_hash == EXPECTED_TOKEN_SHA256, f"The token-list SHA-256 is {token_hash}.", token_path),
        gate("Speech, pitch, and energy statistics", len(stats_files) >= 6, f"The statistics directory has {len(stats_files)} NPZ files.", stats_path),
    ]
    status = "PASS" if all(x["status"] == "PASS" for x in gates) else "FAIL"
    report = {"status": status, "manifest_records": len(frame), "duration_hours": round(hours, 7), "voa_records": voa, "free_disk_gib": round(free, 2), "gates": gates}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
