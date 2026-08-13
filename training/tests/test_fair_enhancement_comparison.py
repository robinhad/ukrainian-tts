from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from training.audio_enhancement.fair_comparison import (
    atomic_write_pcm24,
    fit_sample_count,
    hnr_db,
    noise_floor_dbfs,
    waveform_metrics,
)
from training.scripts.run_fair_enhancement_comparison import SUFFIXES


def test_comparison_has_both_resemble_modes() -> None:
    assert SUFFIXES["resemble_denoise"] == "_resemble_denoise.wav"
    assert SUFFIXES["resemble_enhance"] == "_resemble_enhance.wav"


def test_comparison_has_clearervoice_resemble_cascade() -> None:
    assert SUFFIXES["clearervoice_resemble_enhance"] == (
        "_clearervoice_resemble_enhance.wav"
    )


def test_fit_sample_count_crops_and_pads() -> None:
    source = np.arange(10, dtype=np.float32)[:, None]
    assert fit_sample_count(source, 6).shape == (6, 1)
    padded = fit_sample_count(source, 14)
    assert padded.shape == (14, 1)
    assert np.all(padded[10:] == 0)


def test_fit_sample_count_supports_a_short_backend_result() -> None:
    dry = np.arange(480, dtype=np.float32)
    wet = fit_sample_count(dry[:-32, None], len(dry))[:, 0]

    assert wet.shape == dry.shape
    assert np.all(wet[-32:] == 0)


def test_metrics_keep_sample_and_channel_counts() -> None:
    sample_rate = 24_000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    waveform = (0.1 * np.sin(2 * np.pi * 200 * time))[:, None]
    metrics = waveform_metrics(waveform, sample_rate)

    assert metrics["sample_count"] == sample_rate
    assert metrics["channel_count"] == 1
    assert metrics["duration_seconds"] == 1.0
    assert metrics["finite"] is True
    assert hnr_db(waveform, sample_rate) is not None
    assert np.isfinite(noise_floor_dbfs(waveform, sample_rate))


def test_atomic_pcm24_writer_makes_physical_file(tmp_path: Path) -> None:
    target = tmp_path / "output.wav"
    atomic_write_pcm24(target, np.zeros((100, 2), dtype=np.float32), 24_000)

    info = sf.info(target)
    assert target.is_file()
    assert not target.is_symlink()
    assert info.subtype == "PCM_24"
    assert info.frames == 100
    assert info.channels == 2
