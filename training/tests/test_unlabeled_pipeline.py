from types import SimpleNamespace

import numpy as np

from training.scripts.process_unlabeled import (
    diarization_chunks,
    diarize_in_chunks,
    hypothesis_confidence,
    low_energy_split,
    non_overlapping_segments,
    parse_segment,
    source_progress_key,
    ukrainian_letter_ratio,
)


def test_diarization_keeps_only_single_speaker_intervals():
    segments = [(0.0, 4.0, 0), (3.0, 5.0, 1), (5.0, 7.0, 1)]
    assert non_overlapping_segments(segments) == [
        (0.0, 3.0, 0),
        (4.0, 7.0, 1),
    ]


def test_diarization_segment_parser_accepts_model_formats():
    assert parse_segment("0.25, 2.50, speaker_1") == (0.25, 2.5, 1)
    assert parse_segment([0, 3, 2]) == (0.0, 3.0, 2)


def test_ukrainian_script_filter_rejects_latin_text():
    assert ukrainian_letter_ratio("Українське речення.") == 1.0
    assert ukrainian_letter_ratio("only Latin text") == 0.0


def test_long_segment_splits_near_low_energy():
    sample_rate = 100
    audio = np.ones(3000, dtype=np.float32)
    audio[1150:1250] = 0
    parts = low_energy_split(audio, sample_rate, 0.0, 20.0, 12.0)
    assert len(parts) == 2
    assert 11.0 <= parts[0][1] <= 12.5
    assert parts[-1][1] == 20.0


def test_confidence_uses_normalized_token_scores():
    hypothesis = SimpleNamespace(
        confidence=None,
        mean_confidence=None,
        word_confidence=None,
        token_confidence=[0.7, 0.9],
        score=-100.0,
        y_sequence=[1, 2],
    )
    assert hypothesis_confidence(hypothesis) == 0.8


def test_diarization_chunks_bound_feature_extraction():
    audio = np.arange(1300, dtype=np.float32)
    chunks = diarization_chunks(audio, sample_rate=10, maximum_seconds=60)
    assert [len(chunk) for _, _, chunk in chunks] == [600, 600, 100]
    assert [offset for _, offset, _ in chunks] == [0.0, 60.0, 120.0]


def test_chunked_diarization_preserves_source_time_and_speaker_scope():
    class FakeDiarizer:
        def __init__(self):
            self.calls = 0

        def diarize(self, audio, batch_size, sample_rate):
            self.calls += 1
            assert batch_size == 1
            assert sample_rate == 10
            return [["0.0 2.0 speaker_1"]]

    model = FakeDiarizer()
    segments = diarize_in_chunks(
        model,
        np.zeros(1300, dtype=np.float32),
        sample_rate=10,
        maximum_seconds=60,
        maximum_speakers=4,
    )
    assert model.calls == 3
    assert segments == [
        (0.0, 2.0, 1),
        (60.0, 62.0, 5),
        (120.0, 122.0, 9),
    ]


def test_progress_key_is_stable_and_source_specific():
    first = {
        "audio_sha256_source": "a" * 64,
        "source_group": "group-a",
    }
    second = {**first, "source_group": "group-b"}
    assert source_progress_key(first) == source_progress_key(first)
    assert source_progress_key(first) != source_progress_key(second)
