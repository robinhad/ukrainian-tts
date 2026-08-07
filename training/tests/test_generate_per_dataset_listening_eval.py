from __future__ import annotations

import pandas as pd
import pytest

from training.scripts.generate_per_dataset_listening_eval import select_rows


def make_frame() -> pd.DataFrame:
    rows = []
    for dataset in ("dataset_a", "dataset_b"):
        for variant in ("raw", "clean"):
            for index in range(8):
                rows.append(
                    {
                        "utterance_id": f"{dataset}_{variant}_{index}",
                        "source_id": dataset,
                        "duration": float(index + 1),
                        "embedding_audio_variant": variant,
                    }
                )
    return pd.DataFrame(rows)


def test_select_rows_balances_each_dataset_and_embedding_variant() -> None:
    selected = select_rows(make_frame(), 10)

    assert len(selected) == 20
    counts = selected.groupby(
        ["source_id", "embedding_audio_variant"]
    ).size()
    assert set(counts) == {5}
    assert set(selected.groupby("source_id")["listening_index"].max()) == {10}
    assert selected["utterance_id"].is_unique


def test_select_rows_is_deterministic() -> None:
    frame = make_frame()
    first = select_rows(frame.sample(frac=1.0, random_state=1), 10)
    second = select_rows(frame.sample(frac=1.0, random_state=2), 10)

    assert first["utterance_id"].tolist() == second["utterance_id"].tolist()


def test_select_rows_rejects_odd_count() -> None:
    with pytest.raises(ValueError, match="even number"):
        select_rows(make_frame(), 9)
