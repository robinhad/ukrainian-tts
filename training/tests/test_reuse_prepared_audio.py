from pathlib import Path

import pytest

from training.scripts.reuse_prepared_audio import (
    PREPARATION_FIELDS,
    reconcile,
)


def prepared_row(tmp_path: Path) -> dict:
    audio = tmp_path / "utt.wav"
    audio.write_bytes(b"wav")
    row = {
        "utterance_id": "utt",
        "text_raw": "Тест.",
        "split": "train",
        "audio_sha256_source": "source-hash",
    }
    row.update({field: None for field in PREPARATION_FIELDS})
    row.update({
        "audio_path": str(audio),
        "duration": 1.0,
        "sample_rate": 24000,
    })
    return row


def test_reconcile_reuses_only_selected_prepared_rows(tmp_path: Path):
    prepared = prepared_row(tmp_path)
    source = {
        "utterance_id": "utt",
        "text_raw": "Тест.",
        "split": "train",
        "audio_sha256_source": "source-hash",
        "speaker_id": "speaker",
        "audio_path": "/raw/utt.opus",
    }

    result = reconcile([source], [prepared])

    assert len(result) == 1
    assert result[0]["audio_path"] == prepared["audio_path"]
    assert result[0]["speaker_id"] == "speaker"


def test_reconcile_rejects_changed_text(tmp_path: Path):
    prepared = prepared_row(tmp_path)
    source = prepared.copy()
    source["text_raw"] = "Інший текст."

    with pytest.raises(RuntimeError, match="identity changed in text_raw"):
        reconcile([source], [prepared])
