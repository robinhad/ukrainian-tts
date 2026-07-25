import numpy as np

from training.scripts.process_unlabeled import (
    low_energy_split,
    non_overlapping_segments,
    parse_segment,
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
