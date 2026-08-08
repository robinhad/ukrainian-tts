from __future__ import annotations

import pandas as pd
import pytest

from training.scripts.build_per_dataset_source_audio_review import (
    copy_audio,
    pair_revisions,
)


def make_current() -> pd.DataFrame:
    rows = []
    for dataset in ("dataset_a", "dataset_b"):
        for variant in ("raw", "clean"):
            for index in range(6):
                utterance_id = f"{dataset}_{variant}_{index}"
                rows.append(
                    {
                        "utterance_id": utterance_id,
                        "source_id": dataset,
                        "duration": float(index + 1),
                        "embedding_audio_variant": variant,
                        "audio_path": f"current/{utterance_id}.wav",
                    }
                )
    return pd.DataFrame(rows)


def test_pair_revisions_selects_matched_rows() -> None:
    current = make_current()
    previous = pd.DataFrame(
        {
            "utterance_id": current["utterance_id"],
            "audio_path": [f"previous/{value}.wav" for value in current["utterance_id"]],
        }
    )

    paired = pair_revisions(current, previous, 10)

    assert len(paired) == 20
    assert paired["previous_audio_path"].notna().all()
    assert set(paired.groupby("source_id").size()) == {10}


def test_pair_revisions_rejects_missing_previous_item() -> None:
    current = make_current()
    previous = pd.DataFrame(
        {
            "utterance_id": current["utterance_id"].iloc[1:],
            "audio_path": [
                f"previous/{value}.wav"
                for value in current["utterance_id"].iloc[1:]
            ],
        }
    )

    with pytest.raises(ValueError, match="missing IDs"):
        pair_revisions(current, previous, 10)


def test_copy_audio_makes_an_independent_file(tmp_path) -> None:
    source = tmp_path / "source.wav"
    destination = tmp_path / "review" / "copy.wav"
    source.write_bytes(b"audio-data")

    copy_audio(source, destination)

    assert destination.read_bytes() == b"audio-data"
    assert destination.is_file()
    assert not destination.is_symlink()
