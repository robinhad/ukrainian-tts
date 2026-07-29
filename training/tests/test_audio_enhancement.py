from pathlib import Path

import numpy as np
import soundfile as sf

from training.audio_enhancement.pipeline import (
    EnhancedAudioProcessor,
    EnhancementConfig,
    parse_loudnorm,
    sha256,
)


def test_enhancement_config_is_conservative_and_pinned() -> None:
    config = EnhancementConfig()
    assert config.deepfilter_model == "DeepFilterNet3"
    assert config.attenuation_limit_db == 18.0
    assert config.highpass_hz == 70.0
    assert config.compressor_ratio == 1.5
    assert config.target_lufs == -23.0
    assert config.output_sample_rate == 24_000
    assert config.trim_top_db == 40.0
    assert config.trim_padding_ms == 100.0
    assert len(config.digest) == 64


def test_parse_loudnorm_json() -> None:
    stderr = """
    [Parsed_loudnorm_0] {
      "input_i" : "-20.10",
      "input_tp" : "-2.00",
      "input_lra" : "1.20",
      "input_thresh" : "-30.30",
      "output_i" : "-23.00",
      "output_tp" : "-4.90",
      "output_lra" : "1.10",
      "output_thresh" : "-33.20",
      "normalization_type" : "dynamic",
      "target_offset" : "0.00"
    }
    """
    result = parse_loudnorm(stderr)
    assert result["input_i"] == -20.1
    assert result["target_offset"] == 0.0


def test_inspect_existing_output_measures_without_changing_audio(tmp_path: Path) -> None:
    sample_rate = 24_000
    seconds = 3
    time = np.arange(sample_rate * seconds) / sample_rate
    audio = 0.05 * np.sin(2 * np.pi * 220 * time)
    target = tmp_path / "enhanced.wav"
    sf.write(target, audio, sample_rate, subtype="PCM_16")
    before = sha256(target)

    processor = object.__new__(EnhancedAudioProcessor)
    processor.config = EnhancementConfig()
    metrics = processor.inspect_existing_output(target)

    assert sha256(target) == before
    assert metrics["audio_sha256"] == before
    assert metrics["channels"] == 1
    assert metrics["duration"] == seconds
    assert metrics["sample_rate"] == sample_rate
    assert np.isfinite(metrics["loudness"]["second_pass"]["output_i"])
    assert (
        metrics["loudness"]["second_pass"]["normalization_type"]
        == "measured_existing_output"
    )
