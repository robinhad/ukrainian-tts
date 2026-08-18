from __future__ import annotations

from pathlib import Path

import pandas as pd

from training.scripts.build_expanded_v9_with_voa import (
    EXPECTED_RECORDS,
    EXPECTED_VOA_RECORDS,
    NAME,
    PROFILE,
    PROFILE_HASH,
    SPLIT_MAP,
)
from training.scripts.prepare_hybrid_embeddings import assign_variants
from training.scripts.preprocess_training_cascade_audio import (
    PROFILE_HASH as V8_PROFILE_HASH,
)


ROOT = Path(__file__).resolve().parents[1]


def test_v9_profile_keeps_the_v8_audio_chain() -> None:
    assert NAME == "expanded_v9_cascade_with_voa"
    assert PROFILE["name"] == NAME
    assert PROFILE["order"] == [
        "existing boundary trim",
        "MossFormer2_SE_48K",
        "Sidon",
        "light de-essing",
        "FFmpeg adeclick",
        "FFmpeg alimiter",
        "DeepFilterNet3",
        "Xiph RNNoise85",
        "PCM 24-bit encoding",
    ]
    assert PROFILE_HASH != V8_PROFILE_HASH
    assert V8_PROFILE_HASH == (
        "09471a87613a680615753d75d6fe36479903dffd052ba7fcd834f13cdeb1ddd3"
    )
    assert PROFILE_HASH == (
        "a4b5a34940a0428f4ffa98ae80432784c9cfc932316eb0a9481636374628fffd"
    )
    assert EXPECTED_RECORDS == 208_867
    assert EXPECTED_VOA_RECORDS == 134_711
    assert set(SPLIT_MAP.values()) == {
        f"{NAME}_train",
        f"{NAME}_dev",
        f"{NAME}_eval",
    }


def test_v9_assignment_is_exact_and_deterministic_for_odd_total() -> None:
    frame = pd.DataFrame(
        {
            "utterance_id": [f"u{index:02d}" for index in range(11)],
            "speaker_stratum_id": ["a"] * 5 + ["b"] * 6,
            "speaker_id": ["speaker"] * 11,
        }
    )
    first = assign_variants(frame)
    second = assign_variants(frame.sample(frac=1, random_state=9))
    assert first == second
    assert list(first.values()).count("raw") == 5
    assert list(first.values()).count("clean") == 6


def test_v9_training_is_scratch_and_targets_500k() -> None:
    training = (ROOT / "scripts/run_expanded_v9_with_voa_training.sh").read_text()
    pipeline = (ROOT / "scripts/run_expanded_v9_with_voa_pipeline.sh").read_text()
    assert "--init_param" not in training
    assert "scratch_random_initialization" in training
    assert "TARGET_ITERATIONS=${1:-500000}" in training
    assert "launch_expanded_v9_with_voa_training.sh\" 500000" in pipeline


def test_v9_keeps_the_exact_embedding_split() -> None:
    audit = (ROOT / "scripts/audit_expanded_v9_with_voa_readiness.py").read_text()
    assert 'EXPECTED_EMBEDDINGS = {"raw": 104_433, "clean": 104_434}' in audit
