import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_seed_preserves_existing_records_and_reshards(tmp_path: Path) -> None:
    source_audio = tmp_path / "source.wav"
    source_audio.write_bytes(b"source")
    clean_audio = tmp_path / "clean.wav"
    clean_audio.write_bytes(b"clean")
    existing_audio = tmp_path / "existing.wav"
    existing_audio.write_bytes(b"existing")
    source_manifest = tmp_path / "source.parquet"
    clean_manifest = tmp_path / "clean.parquet"
    records_dir = tmp_path / "records"
    records_dir.mkdir()
    pd.DataFrame(
        [
            {
                "utterance_id": "a",
                "audio_path": str(source_audio),
                "canonical_raw_audio_path": str(source_audio),
            },
            {
                "utterance_id": "b",
                "audio_path": str(source_audio),
                "canonical_raw_audio_path": str(source_audio),
            },
        ]
    ).to_parquet(source_manifest, index=False)
    pd.DataFrame(
        [
            {
                "utterance_id": "b",
                "audio_path": str(clean_audio),
                "enhancement_status": "ok",
            }
        ]
    ).to_parquet(clean_manifest, index=False)
    (records_dir / "records-9.jsonl").write_text(
        json.dumps(
            {
                "utterance_id": "a",
                "audio_path": str(existing_audio),
                "enhancement_status": "ok",
                "enhancement_attempts": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "seed_reused_enhanced_records.py"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--source-manifest",
            str(source_manifest),
            "--clean-manifest",
            str(clean_manifest),
            "--output-records-dir",
            str(records_dir),
            "--num-shards",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(result.stdout)
    first = json.loads((records_dir / "records-0.jsonl").read_text())
    second = json.loads((records_dir / "records-1.jsonl").read_text())

    assert report["preserved_existing"] == 1
    assert report["reused_clean"] == 1
    assert first["utterance_id"] == "a"
    assert first["audio_path"] == str(existing_audio)
    assert second["utterance_id"] == "b"
    assert second["audio_path"] == str(clean_audio.resolve())
