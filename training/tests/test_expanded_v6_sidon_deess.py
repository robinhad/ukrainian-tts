import json
from pathlib import Path

import numpy as np
import soundfile as sf

from training.audio_enhancement.review_pipeline import SidonDeessOnlyConfig, sha256
from training.scripts.build_expanded_v6_sidon_deess_novoa import PROFILE, PROFILE_HASH, SPLIT_MAP
from training.scripts.preprocess_sidon_deess_audio import output_is_valid


def test_profile_has_only_requested_processing() -> None:
    assert PROFILE["sidon"] is True
    assert PROFILE["deessing"] is True
    assert PROFILE["post_highpass"] is False
    assert PROFILE["compression"] is False
    assert PROFILE["loudness_normalization"] is False
    assert PROFILE["limiting"] is False
    assert len(PROFILE_HASH) == 64
    assert set(SPLIT_MAP.values()) == {
        "expanded_v6_sidon_deess_novoa_train",
        "expanded_v6_sidon_deess_novoa_dev",
        "expanded_v6_sidon_deess_novoa_eval",
    }


def test_resume_requires_physical_valid_wav_and_matching_source(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    target = tmp_path / "target.wav"
    audio = np.zeros(2400, dtype=np.float32)
    audio[100:300] = 0.1
    sf.write(source, audio, 24_000, subtype="PCM_16")
    sf.write(target, audio, 24_000, subtype="PCM_16")
    config = SidonDeessOnlyConfig()
    metadata = {
        "status": "PASS",
        "source": str(source.resolve()),
        "source_sha256": sha256(source),
        "audio_sha256": sha256(target),
        "processing_config_hash": config.digest,
    }
    target.with_suffix(".wav.json").write_text(json.dumps(metadata))
    assert output_is_valid(target, source, config)
    sf.write(source, audio * 2, 24_000, subtype="PCM_16")
    assert not output_is_valid(target, source, config)
