import json
from pathlib import Path

from training.scripts.collect_hf_unlabeled import existing_collection


def make_collection(root: Path) -> tuple[Path, Path, Path]:
    audio = root / "source.flac"
    audio.write_bytes(b"audio")
    records = root / "records.jsonl"
    records.write_text(
        json.dumps({"audio_path": str(audio)}) + "\n",
        encoding="utf-8",
    )
    report = root / "report.json"
    report.write_text(
        json.dumps(
            {
                "status": "PASS",
                "source_id": "source",
                "shard_start": 31,
                "shard_count": 10,
                "records": 1,
            }
        ),
        encoding="utf-8",
    )
    return audio, records, report


def test_complete_existing_collection_can_resume(tmp_path: Path):
    _, records, report = make_collection(tmp_path)
    result = existing_collection(records, report, "source", 31, 10)
    assert result is not None
    assert result["status"] == "PASS"


def test_existing_collection_requires_all_audio(tmp_path: Path):
    audio, records, report = make_collection(tmp_path)
    audio.unlink()
    assert existing_collection(records, report, "source", 31, 10) is None


def test_existing_collection_rejects_a_different_shard_range(tmp_path: Path):
    _, records, report = make_collection(tmp_path)
    assert existing_collection(records, report, "source", 41, 10) is None
