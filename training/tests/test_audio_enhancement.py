from pathlib import Path

import numpy as np
import soundfile as sf

from training.audio_enhancement.pipeline import EnhancementConfig, parse_loudnorm


def test_enhancement_config_is_conservative_and_pinned() -> None:
    config = EnhancementConfig()
    assert config.deepfilter_model == "DeepFilterNet3"
    assert config.attenuation_limit_db == 18.0
    assert config.highpass_hz == 70.0
    assert config.compressor_ratio == 1.5
    assert config.target_lufs == -23.0
    assert config.output_sample_rate == 24_000
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

