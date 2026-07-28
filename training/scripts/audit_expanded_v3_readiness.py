#!/usr/bin/env python3
"""Create a strict readiness report for the expanded-v3 training run."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from training.scripts.source_policy import check_disk, load_registry


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def gate(
    name: str,
    status: str,
    evidence: str,
    artifact: Path | None,
    next_action: str,
) -> dict[str, str | None]:
    return {
        "gate": name,
        "status": status,
        "evidence": evidence,
        "artifact": str(artifact) if artifact else None,
        "next_action": next_action,
    }


def minimum_source_duration_gates(
    frame: pd.DataFrame,
    registry: dict[str, Any],
) -> list[dict[str, str | None]]:
    """Return gates for sources that define a minimum retained duration."""
    results = []
    for source_id, source in sorted(registry["sources"].items()):
        minimum = source.get("minimum_retained_hours")
        if minimum is None:
            continue
        retained = float(
            frame.loc[frame["source_id"].astype(str) == source_id, "duration"].sum()
            / 3600
        )
        passed = retained >= float(minimum)
        results.append(
            gate(
                f"{source_id} retained duration",
                "PASS" if passed else "FAIL",
                (
                    f"The manifest has {retained:.3f} hours. "
                    f"The minimum is {float(minimum):.3f} hours."
                ),
                None,
                (
                    "Preserve the retained source duration."
                    if passed
                    else "Rebuild the source with the configured pseudo-label settings."
                ),
            )
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--reports-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    args = parser.parse_args()

    registry = load_registry(args.registry)
    disk = check_disk(args.workspace, registry)
    prefix = f"expanded_v3_{args.mode}"
    manifest_path = args.data_root / "manifests" / "all.parquet"
    validation_path = args.reports_root / f"{prefix}_validation.json"
    hybrid_path = args.reports_root / f"{prefix}_hybrid_embeddings.json"
    inference_path = args.reports_root / f"{prefix}_inference.json"
    source_report_path = args.reports_root / f"{prefix}_sources.json"
    validation = load_json(validation_path)
    hybrid = load_json(hybrid_path)
    inference = load_json(inference_path)
    source_report = load_json(source_report_path)

    gates: list[dict[str, str | None]] = []
    disk_ok = disk["status"] != "STOP"
    gates.append(
        gate(
            "Disk reserve",
            "PASS" if disk_ok else "FAIL",
            (
                f"{disk['free_gib']} GiB is free. "
                f"The stop threshold is {disk['stop_gib']} GiB."
            ),
            args.workspace,
            "Free disk space before the next stage." if not disk_ok else "Recheck during processing.",
        )
    )

    source_counts: Counter[str] = Counter()
    manifest_records = 0
    frame = pd.DataFrame(columns=["source_id", "duration"])
    if manifest_path.is_file():
        frame = pd.read_parquet(manifest_path, columns=["source_id", "duration"])
        manifest_records = len(frame)
        source_counts.update(str(value) for value in frame["source_id"])
    manifest_ok = manifest_records > 0
    gates.append(
        gate(
            "Clean training manifest",
            "PASS" if manifest_ok else "NOT RUN",
            f"The manifest has {manifest_records} records.",
            manifest_path if manifest_path.is_file() else None,
            "Prepare and validate the clean audio manifest." if not manifest_ok else "Preserve its hash.",
        )
    )
    if args.mode == "full" and manifest_ok:
        gates.extend(minimum_source_duration_gates(frame, registry))

    enabled = {
        source_id
        for source_id, source in registry["sources"].items()
        if source.get("enabled") and source.get("decision") in {"allow", "allow_per_file"}
    }
    present = set(source_counts)
    missing = sorted(enabled - present)
    if args.mode == "smoke":
        source_ok = bool(present)
        source_status = "PASS" if source_ok else "NOT RUN"
        source_evidence = (
            f"The smoke manifest has {len(present)} sources: {', '.join(sorted(present))}."
        )
    else:
        source_ok = not missing and bool(present)
        source_status = "PASS" if source_ok else "FAIL"
        source_evidence = (
            f"Present sources: {', '.join(sorted(present)) or 'none'}. "
            f"Missing enabled sources: {', '.join(missing) or 'none'}."
        )
    gates.append(
        gate(
            "Enabled source coverage",
            source_status,
            source_evidence,
            source_report_path if source_report else manifest_path if manifest_ok else None,
            "Ingest each missing source with valid license evidence." if not source_ok else "Keep the source revisions fixed.",
        )
    )

    validation_ok = bool(validation and validation.get("status") == "PASS")
    gates.append(
        gate(
            "Dataset validation",
            "PASS" if validation_ok else "NOT RUN",
            "The dataset validator returned PASS." if validation_ok else "No PASS validation report exists.",
            validation_path if validation else None,
            "Run the dataset validator." if not validation_ok else "Repeat after any manifest change.",
        )
    )

    hybrid_counts = hybrid.get("counts", {}) if hybrid else {}
    hybrid_ok = bool(
        hybrid
        and hybrid.get("status") == "PASS"
        and hybrid.get("records") == manifest_records
        and hybrid_counts.get("raw") == hybrid_counts.get("clean")
    )
    gates.append(
        gate(
            "Hybrid speaker embeddings",
            "PASS" if hybrid_ok else "NOT RUN",
            (
                f"The report has {hybrid_counts.get('raw')} raw and "
                f"{hybrid_counts.get('clean')} clean embeddings."
                if hybrid
                else "No hybrid embedding report exists."
            ),
            hybrid_path if hybrid else None,
            "Create the exact 50/50 ECAPA embedding set." if not hybrid_ok else "Preserve the ECAPA revision.",
        )
    )

    dump_name = "dump_expanded_v3_smoke" if args.mode == "smoke" else "dump_expanded_v3"
    exp_name = "exp_expanded_v3_smoke" if args.mode == "smoke" else "exp_expanded_v3"
    token_path = args.workspace / dump_name / "token_list" / "phn_espeak_ng_ukrainian" / "tokens.txt"
    stats_root = args.workspace / exp_name / "tts_stats_raw_phn_espeak_ng_ukrainian"
    token_ok = token_path.is_file() and token_path.stat().st_size > 0
    stats_ok = stats_root.is_dir() and any(stats_root.rglob("*.npz"))
    gates.append(
        gate(
            "Token list",
            "PASS" if token_ok else "NOT RUN",
            f"The token list has {sum(1 for _ in token_path.open()) if token_ok else 0} lines.",
            token_path if token_ok else None,
            "Run ESPnet token collection." if not token_ok else "Preserve the token list.",
        )
    )
    gates.append(
        gate(
            "Pitch and energy statistics",
            "PASS" if stats_ok else "NOT RUN",
            "ESPnet statistics exist." if stats_ok else "ESPnet statistics do not exist.",
            stats_root if stats_ok else None,
            "Run ESPnet statistics collection." if not stats_ok else "Preserve the statistics.",
        )
    )

    if args.mode == "smoke":
        checkpoint = (
            args.workspace
            / exp_name
            / "tts_jets_uk_24k_expanded_v3_smoke"
            / "1epoch.pth"
        )
        checkpoint_ok = checkpoint.is_file() and checkpoint.stat().st_size > 0
        inference_ok = bool(inference and inference.get("status") == "PASS")
        gates.append(
            gate(
                "Smoke checkpoint",
                "PASS" if checkpoint_ok else "NOT RUN",
                "A real 100-iteration checkpoint exists." if checkpoint_ok else "No smoke checkpoint exists.",
                checkpoint if checkpoint_ok else None,
                "Run the dual-GPU smoke training." if not checkpoint_ok else "Keep the checkpoint off Git.",
            )
        )
        gates.append(
            gate(
                "Smoke inference",
                "PASS" if inference_ok else "NOT RUN",
                (
                    f"Inference validated {len(inference.get('records', []))} WAV files."
                    if inference_ok
                    else "No PASS inference report exists."
                ),
                inference_path if inference else None,
                "Run inference and validate each WAV." if not inference_ok else "Use the full-data gate next.",
            )
        )

    critical_fail = any(item["status"] == "FAIL" for item in gates)
    critical_pending = any(item["status"] == "NOT RUN" for item in gates)
    status = "PASS" if not critical_fail and not critical_pending else "FAIL"
    report = {
        "status": status,
        "mode": args.mode,
        "manifest_records": manifest_records,
        "source_counts": dict(sorted(source_counts.items())),
        "missing_enabled_sources": missing if args.mode == "full" else [],
        "disk": disk,
        "gates": gates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
