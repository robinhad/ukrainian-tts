from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from training.scripts.export_training_samples_per_dataset import (
    atomic_copy,
    select_training_rows,
)


def make_frame() -> pd.DataFrame:
    rows = []
    for dataset in ("dataset_a", "dataset_b"):
        for variant in ("raw", "clean"):
            for index in range(6):
                rows.append(
                    {
                        "utterance_id": f"{dataset}_{variant}_{index}",
                        "source_id": dataset,
                        "duration": float(index + 1),
                        "embedding_audio_variant": variant,
                        "split": "revision_train" if index else "revision_eval",
                    }
                )
    return pd.DataFrame(rows)


def test_select_training_rows_excludes_other_splits() -> None:
    selected = select_training_rows(make_frame(), 10, "_train")

    assert len(selected) == 20
    assert not selected["utterance_id"].str.endswith("_0").any()
    counts = selected.groupby(["source_id", "embedding_audio_variant"]).size()
    assert set(counts) == {5}


def test_select_training_rows_rejects_unknown_split() -> None:
    with pytest.raises(ValueError, match="no rows"):
        select_training_rows(make_frame(), 10, "_missing")


def test_atomic_copy_makes_a_physical_copy(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    destination = tmp_path / "output" / "copy.wav"
    source.write_bytes(b"audio-data")

    atomic_copy(source, destination)

    assert destination.read_bytes() == b"audio-data"
    assert destination.is_file()
    assert not destination.is_symlink()
