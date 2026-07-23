import numpy as np

from training.scripts.prepare_audio import trim_silence


def test_trim_silence_keeps_padding():
    sample_rate = 24_000
    leading = np.zeros(sample_rate, dtype=np.float32)
    tone = np.full(sample_rate, 0.5, dtype=np.float32)
    trailing = np.zeros(sample_rate, dtype=np.float32)
    audio = np.concatenate((leading, tone, trailing))[:, None]

    trimmed, metadata = trim_silence(audio, sample_rate)

    assert metadata["trim_applied"] is True
    assert 0.80 < metadata["trim_start_seconds"] < 1.0
    assert 0.80 < metadata["trim_end_seconds"] < 1.0
    assert 1.1 < len(trimmed) / sample_rate < 1.4
    assert np.max(trimmed) == 0.5


def test_trim_silence_keeps_non_silent_audio():
    audio = np.full((24_000, 1), 0.25, dtype=np.float32)

    trimmed, metadata = trim_silence(audio, 24_000)

    assert len(trimmed) == len(audio)
    assert metadata["trim_applied"] is False
    assert metadata["trim_removed_seconds"] == 0.0


def test_trim_silence_does_not_delete_all_zero_audio():
    audio = np.zeros((24_000, 1), dtype=np.float32)

    trimmed, metadata = trim_silence(audio, 24_000)

    assert len(trimmed) == len(audio)
    assert metadata["trim_applied"] is False
