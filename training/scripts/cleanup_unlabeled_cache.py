#!/usr/bin/env python3
"""Record a completed batch and reclaim cache only at the disk trigger."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


GIB = 1024**3
RETENTION_SUFFIX = ".retain-until-60"


def remove_file(path: Path) -> int:
    if not path.is_file() and not path.is_symlink():
        return 0
    size = path.stat().st_size
    path.unlink()
    return size


def free_disk_gib(path: Path) -> float:
    probe = path.resolve()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free / GIB


def remove_batch(records: Path, collect_report: Path) -> int:
    removed_bytes = 0
    if records.is_file():
        for line in records.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                audio_path = Path(record["audio_path"])
                removed_bytes += remove_file(audio_path)
                removed_bytes += remove_file(
                    audio_path.with_name(audio_path.name + RETENTION_SUFFIX)
                )
    report = json.loads(collect_report.read_text(encoding="utf-8"))
    for item in report.get("cache_artifacts", []):
        snapshot = Path(item["snapshot_path"])
        blob = Path(item["blob_path"])
        removed_bytes += remove_file(snapshot)
        if blob != snapshot:
            removed_bytes += remove_file(blob)
    removed_bytes += remove_file(records)
    return removed_bytes


def restore_retained_audio(records: Path) -> int:
    """Restore paths that an already-running legacy worker removed."""
    restored = 0
    if not records.is_file():
        return restored
    for line in records.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        audio_path = Path(json.loads(line)["audio_path"])
        retained = audio_path.with_name(audio_path.name + RETENTION_SUFFIX)
        if not retained.is_file():
            continue
        if not audio_path.exists():
            os.link(retained, audio_path)
            restored += 1
        retained.unlink()
    return restored


def write_marker(
    marker: Path,
    records: Path,
    collect_report: Path,
    trigger_gib: float,
    cleanup_status: str,
    removed_bytes: int,
    free_before_gib: float,
    free_after_gib: float,
) -> None:
    report = json.loads(collect_report.read_text(encoding="utf-8"))
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {
                "status": "PASS",
                "shard_start": report["shard_start"],
                "shard_count": report["shard_count"],
                "cleanup_status": cleanup_status,
                "cleanup_trigger_gib": trigger_gib,
                "records_path": str(records.resolve()),
                "collect_report_path": str(collect_report.resolve()),
                "removed_bytes": removed_bytes,
                "free_gib_before": round(free_before_gib, 2),
                "free_gib_after": round(free_after_gib, 2),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def sweep_deferred_batches(
    markers_root: Path,
    workspace: Path,
    trigger_gib: float,
) -> list[Path]:
    """Remove oldest deferred batches only when free space reaches the trigger."""
    cleaned: list[Path] = []
    if free_disk_gib(workspace) > trigger_gib:
        return cleaned
    # A pipeline version can put markers in a child directory. Search all
    # version directories so the collector can reclaim a completed batch.
    for marker in sorted(markers_root.rglob("*.json")):
        try:
            data = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("cleanup_status") != "DEFERRED":
            continue
        records_value = data.get("records_path")
        report_value = data.get("collect_report_path")
        if not records_value or not report_value:
            continue
        before = free_disk_gib(workspace)
        removed = remove_batch(Path(records_value), Path(report_value))
        after = free_disk_gib(workspace)
        data.update(
            {
                "cleanup_status": "CLEANED",
                "removed_bytes": int(data.get("removed_bytes", 0)) + removed,
                "free_gib_before": round(before, 2),
                "free_gib_after": round(after, 2),
            }
        )
        marker.write_text(
            json.dumps(data, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        cleaned.append(marker)
        if after > trigger_gib:
            break
    return cleaned


def finalize_batch(
    records: Path,
    collect_report: Path,
    marker: Path,
    trigger_gib: float,
) -> dict[str, object]:
    workspace = marker.parent
    before = free_disk_gib(workspace)
    sweep_deferred_batches(marker.parent, workspace, trigger_gib)
    removed_bytes = 0
    if free_disk_gib(workspace) <= trigger_gib:
        removed_bytes = remove_batch(records, collect_report)
        cleanup_status = "CLEANED"
    else:
        restore_retained_audio(records)
        cleanup_status = "DEFERRED"
    after = free_disk_gib(workspace)
    write_marker(
        marker,
        records,
        collect_report,
        trigger_gib,
        cleanup_status,
        removed_bytes,
        before,
        after,
    )
    return json.loads(marker.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--collect-report", type=Path, required=True)
    parser.add_argument("--marker", type=Path, required=True)
    parser.add_argument("--free-disk-trigger-gib", type=float, default=60.0)
    args = parser.parse_args()

    if args.free_disk_trigger_gib < 0:
        parser.error("--free-disk-trigger-gib must not be negative.")
    result = finalize_batch(
        args.records,
        args.collect_report,
        args.marker,
        args.free_disk_trigger_gib,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
