import pandas as pd

from training.scripts.build_expanded_v5_enhanced_novoa import (
    PROFILE,
    SPLIT_MAP,
    is_voa,
    smoke_subset,
)


def test_voa_filter_detects_known_identifiers() -> None:
    assert is_voa("voa_ukr_user_grant")
    assert is_voa("VOICE_OF_AMERICA")
    assert not is_voa("common_voice_available_uk")


def test_v5_profile_and_splits_are_fixed() -> None:
    assert PROFILE["deepfilternet"]
    assert PROFILE["highpass"]
    assert PROFILE["deessing"]
    assert PROFILE["compression"]
    assert PROFILE["two_pass_ebu_r128"]
    assert set(SPLIT_MAP.values()) == {
        "expanded_v5_enhanced_novoa_train",
        "expanded_v5_enhanced_novoa_dev",
        "expanded_v5_enhanced_novoa_eval",
    }


def test_smoke_selection_is_deterministic() -> None:
    rows = []
    for suffix, count in (("train", 300), ("dev", 40), ("eval", 40)):
        rows.extend(
            {
                "utterance_id": f"{suffix}-{index}",
                "split": f"expanded_v5_enhanced_novoa_{suffix}",
            }
            for index in range(count)
        )
    frame = pd.DataFrame(rows)
    first = smoke_subset(frame).sort_values("utterance_id").reset_index(drop=True)
    second = smoke_subset(frame).sort_values("utterance_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(first, second)
    assert first["split"].value_counts().to_dict() == {
        "expanded_v5_enhanced_novoa_train": 256,
        "expanded_v5_enhanced_novoa_dev": 32,
        "expanded_v5_enhanced_novoa_eval": 32,
    }
