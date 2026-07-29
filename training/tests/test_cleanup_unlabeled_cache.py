import json
import os
from pathlib import Path

from training.scripts.cleanup_unlabeled_cache import (
    RETENTION_SUFFIX,
    finalize_batch,
    sweep_deferred_batches,
)


def make_batch(root: Path, tag: str) -> tuple[Path, Path, Path, Path]:
    audio = root / f"{tag}.flac"
    audio.write_bytes(b"audio")
    records = root / f"records-{tag}.jsonl"
    records.write_text(
        json.dumps({"audio_path": str(audio)}) + "\n",
        encoding="utf-8",
    )
    report = root / f"collect-{tag}.json"
    report.write_text(
        json.dumps(
            {
                "shard_start": 1,
                "shard_count": 1,
                "cache_artifacts": [],
            }
        ),
        encoding="utf-8",
    )
    marker = root / "markers" / f"{tag}.json"
    return audio, records, report, marker


def test_cleanup_is_deferred_above_trigger(tmp_path: Path):
    audio, records, report, marker = make_batch(tmp_path, "001-001")
    result = finalize_batch(records, report, marker, trigger_gib=0)
    assert result["status"] == "PASS"
    assert result["cleanup_status"] == "DEFERRED"
    assert audio.is_file()
    assert records.is_file()


def test_cleanup_runs_at_trigger(tmp_path: Path):
    audio, records, report, marker = make_batch(tmp_path, "001-001")
    result = finalize_batch(records, report, marker, trigger_gib=10**9)
    assert result["status"] == "PASS"
    assert result["cleanup_status"] == "CLEANED"
    assert not audio.exists()
    assert not records.exists()


def test_deferred_cleanup_restores_a_legacy_worker_path(tmp_path: Path):
    audio, records, report, marker = make_batch(tmp_path, "001-001")
    retained = audio.with_name(audio.name + RETENTION_SUFFIX)
    os.link(audio, retained)
    audio.unlink()
    result = finalize_batch(records, report, marker, trigger_gib=0)
    assert result["cleanup_status"] == "DEFERRED"
    assert audio.read_bytes() == b"audio"
    assert not retained.exists()


def test_sweep_finds_markers_in_pipeline_version_directory(tmp_path: Path):
    audio, records, report, marker = make_batch(tmp_path, "001-001")
    versioned_marker = marker.parent / "v4" / marker.name
    result = finalize_batch(records, report, versioned_marker, trigger_gib=0)
    assert result["cleanup_status"] == "DEFERRED"

    cleaned = sweep_deferred_batches(
        tmp_path / "markers",
        tmp_path,
        trigger_gib=10**9,
    )

    assert cleaned == [versioned_marker]
    assert not audio.exists()
    assert not records.exists()
    assert json.loads(versioned_marker.read_text())["cleanup_status"] == "CLEANED"
