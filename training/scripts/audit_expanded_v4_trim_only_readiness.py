#!/usr/bin/env python3
"""Create a strict readiness report for the trim-only corpus."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


def load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def gate(name: str, passed: bool, evidence: str, artifact: Path | None) -> dict[str, Any]:
    return {
        "gate": name,
        "status": "PASS" if passed else "FAIL",
        "evidence": evidence,
        "artifact": str(artifact) if artifact else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--exp-dir", type=Path, required=True)
    parser.add_argument("--reports-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--reference-token-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prefix = f"expanded_v4_trim_only_{args.mode}"
    build_path = args.reports_root / f"{prefix}_dataset.json"
    validation_path = args.reports_root / f"{prefix}_validation.json"
    hybrid_path = args.reports_root / f"{prefix}_hybrid_embeddings.json"
    manifest_path = args.data_root / "manifests" / "all.parquet"
    token_path = args.dump_dir / "token_list" / "phn_espeak_ng_ukrainian" / "tokens.txt"
    stats_path = args.exp_dir / "tts_stats_raw_phn_espeak_ng_ukrainian"
    build = load(build_path)
    validation = load(validation_path)
    hybrid = load(hybrid_path)
    frame = pd.read_parquet(manifest_path) if manifest_path.is_file() else pd.DataFrame()
    expected = 320 if args.mode == "smoke" else 207_505
    free_gib = shutil.disk_usage(args.workspace).free / 1024**3

    gates = [
        gate("Disk reserve", free_gib > 30, f"{free_gib:.2f} GiB is free. The limit is 30 GiB.", args.workspace),
        gate("Dataset build", bool(build and build.get("status") == "PASS"), f"The builder report has {build.get('records') if build else 0} records.", build_path if build else None),
        gate("Expected record count", len(frame) == expected, f"The manifest has {len(frame)} of {expected} expected records.", manifest_path if manifest_path.is_file() else None),
        gate("Dataset validation", bool(validation and validation.get("status") == "PASS"), f"The validator found {validation.get('clipping_files') if validation else 'unknown'} clipping files.", validation_path if validation else None),
    ]
    disabled = (
        "deepfilternet_applied",
        "highpass_applied",
        "deessing_applied",
        "compression_applied",
        "loudness_normalization_applied",
    )
    profile_ok = bool(len(frame)) and all(
        name in frame and not frame[name].fillna(True).astype(bool).any() for name in disabled
    )
    gates.append(gate("Trim-only profile", profile_ok, "DeepFilterNet, high-pass, de-essing, compression, and loudness normalization are disabled.", manifest_path if len(frame) else None))
    counts = hybrid.get("counts", {}) if hybrid else {}
    embedding_ok = bool(
        hybrid
        and hybrid.get("status") == "PASS"
        and hybrid.get("records") == len(frame)
        and counts.get("raw", 0) + counts.get("clean", 0) == len(frame)
        and abs(counts.get("raw", 0) - counts.get("clean", 0)) <= 1
        and hybrid.get("maximum_speaker_delta", 2) <= 1
    )
    gates.append(gate("True 50/50 embeddings", embedding_ok, f"The report has {counts.get('raw')} pre-trim and {counts.get('clean')} post-trim vectors.", hybrid_path if hybrid else None))
    token_count = (
        sum(1 for _ in token_path.open(encoding="utf-8"))
        if token_path.is_file()
        else 0
    )
    token_ok = (
        token_count == 138
        and args.reference_token_list.is_file()
        and token_path.read_bytes() == args.reference_token_list.read_bytes()
    )
    gates.append(
        gate(
            "Token list",
            token_ok,
            f"The token list has {token_count} entries and must equal the epoch-367 list.",
            token_path if token_path.is_file() else None,
        )
    )
    stats_ok = stats_path.is_dir() and len(list(stats_path.rglob("*.npz"))) >= 6
    gates.append(gate("Speech, pitch, and energy statistics", stats_ok, "The required NPZ statistics exist." if stats_ok else "The required statistics do not exist.", stats_path if stats_ok else None))
    status = "PASS" if all(item["status"] == "PASS" for item in gates) else "FAIL"
    report = {
        "status": status,
        "mode": args.mode,
        "authorized_voa_minimum_exception": True,
        "manifest_records": len(frame),
        "free_disk_gib": round(free_gib, 2),
        "gates": gates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
