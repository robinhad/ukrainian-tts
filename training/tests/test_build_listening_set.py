from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from training.scripts.build_listening_set import balanced_rows, reference_path


def test_reference_path_uses_manifest_audio_path():
    item = SimpleNamespace(
        utterance_id="utt-1",
        audio_path="/data/processed/utt-1.wav",
    )

    assert reference_path(item, Path("/legacy")) == Path(
        "/data/processed/utt-1.wav"
    )


def test_reference_path_uses_legacy_raw_directory():
    item = SimpleNamespace(utterance_id="utt-1")

    assert reference_path(item, Path("/legacy")) == Path("/legacy/utt-1.ogg")


def test_reference_path_requires_one_source():
    item = SimpleNamespace(utterance_id="utt-1")

    with pytest.raises(ValueError, match="provide --raw-dir"):
        reference_path(item, None)


def test_balanced_rows_selects_each_source():
    frame = pd.DataFrame(
        {
            "utterance_id": [f"a-{index}" for index in range(4)]
            + [f"b-{index}" for index in range(4)],
            "source": ["a"] * 4 + ["b"] * 4,
            "duration": [1.0, 2.0, 3.0, 4.0] * 2,
        }
    )

    selected = balanced_rows(frame, count=4, column="source")

    assert selected["source"].value_counts().to_dict() == {"a": 2, "b": 2}
    assert selected["utterance_id"].tolist() == ["a-0", "a-3", "b-0", "b-3"]
