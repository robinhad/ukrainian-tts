import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from kaldiio import WriteHelper

from training.inference.synthesize import load_speaker_embedding
from training.scripts.create_multispeaker_dataset import (
    EMBEDDING_MODEL,
    load_lada,
    remove_cross_source_text_overlap,
    split_for_text,
    split_name,
    stable_score,
    transcription_issue,
)
from training.frontend.sanitize import sanitize_text


def test_common_voice_text_split_is_deterministic():
    text_hash = "a" * 64
    assert split_for_text(text_hash) == split_for_text(text_hash)
    assert stable_score("sample") == stable_score("sample")
    assert split_name("smoke", "train") == "multispeaker_smoke_train"
    assert split_name("full", "eval") == "multispeaker_eval"
    assert not any(character.isalpha() for character in sanitize_text("-"))


def test_common_voice_rejects_embedded_metadata_and_long_text():
    assert transcription_issue("Добрий текст.", 500) is None
    assert (
        transcription_issue("Добрий текст.\tmetadata", 500)
        == "embedded_metadata_separator"
    )
    assert transcription_issue("а" * 501, 500) == "text_too_long"


def test_lada_manifest_becomes_embedding_conditioned(tmp_path: Path):
    manifest = tmp_path / "lada.parquet"
    pd.DataFrame(
        [
            {
                "utterance_id": "lada_1",
                "speaker_id": "lada",
                "audio_path": "/tmp/lada.wav",
                "text_raw": "Тест.",
                "duration": 3.0,
                "source": "test",
                "source_license": "Apache-2.0",
                "source_group": "group",
                "split": "train",
                "audio_sha256": "a" * 64,
                "text_sha256": "b" * 64,
            }
        ]
    ).to_parquet(manifest, index=False)
    rows = load_lada(manifest)
    assert len(rows) == 1
    assert rows[0]["speaker_id"] == "lada"
    assert rows[0]["speaker_embedding_mode"] == "utterance"
    assert rows[0]["speaker_embedding_model"] == EMBEDDING_MODEL
    assert rows[0]["speaker_identity_status"] == "explicit_lada"


def test_common_voice_duplicate_of_explicit_speaker_is_removed():
    text = "Однаковий текст."
    text_hash = hashlib.sha256(sanitize_text(text).encode("utf-8")).hexdigest()
    cv_rows = [{"text_raw": text, "text_sha256_source": text_hash}]
    explicit_rows = [{"text_raw": text}]
    filtered, excluded = remove_cross_source_text_overlap(cv_rows, explicit_rows)
    assert filtered == []
    assert excluded == 1


def test_load_speaker_embedding_from_ark(tmp_path: Path):
    ark = tmp_path / "speakers.ark"
    scp = tmp_path / "speakers.scp"
    vector = np.arange(192, dtype=np.float32)
    with WriteHelper(f"ark,scp:{ark},{scp}") as writer:
        writer["dmytro"] = vector
    loaded = load_speaker_embedding(ark, "dmytro")
    np.testing.assert_array_equal(loaded, vector)
