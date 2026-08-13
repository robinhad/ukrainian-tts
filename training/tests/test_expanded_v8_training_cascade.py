from __future__ import annotations

import json
from pathlib import Path

from training.scripts.preprocess_training_cascade_audio import (
    Cascade,
    PROFILE,
    PROFILE_HASH,
    PROFILE_NAME,
    release_host_memory,
    trusted_restart_output_is_valid,
)


def test_v8_profile_has_the_requested_order() -> None:
    assert PROFILE_NAME == (
        "expanded_v8_clearervoice_sidon_deess_declick_limit_"
        "deepfilternet3_rnnoise85_novoa"
    )
    assert PROFILE["order"] == [
        "existing boundary trim",
        "MossFormer2_SE_48K",
        "Sidon",
        "light de-essing",
        "FFmpeg adeclick",
        "FFmpeg alimiter",
        "DeepFilterNet3",
        "Xiph RNNoise85",
        "PCM 24-bit encoding",
    ]
    assert PROFILE["rnnoise_wet_mix"] == 0.85
    assert PROFILE["compression"] is False
    assert PROFILE["dynamic_loudness_normalization"] is False
    assert len(PROFILE_HASH) == 64


def test_v8_profile_uses_float_filters_and_pcm24_output() -> None:
    assert PROFILE["declick_limiter_filter"].startswith(
        "aformat=sample_fmts=fltp,adeclick="
    )
    assert "alimiter=limit=0.891251" in PROFILE["declick_limiter_filter"]
    assert PROFILE["output_format"] == "WAV/PCM_24"
    assert PROFILE["output_sample_rate"] == 24_000
    assert PROFILE["output_channels"] == 1


def test_v8_worker_can_release_unused_host_memory() -> None:
    assert isinstance(release_host_memory(), bool)


def test_v8_trusted_restart_matches_recorded_output(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    target = tmp_path / "output.wav"
    target.write_bytes(b"x" * 45)
    target.with_suffix(".wav.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "profile_hash": PROFILE_HASH,
                "source": str(source.resolve()),
                "output_sha256": "recorded-hash",
            }
        ),
        encoding="utf-8",
    )
    assert trusted_restart_output_is_valid(
        target, source, {"audio_sha256": "recorded-hash"}
    )
    assert not trusted_restart_output_is_valid(
        target, source, {"audio_sha256": "different-hash"}
    )


def test_v8_cascade_has_a_recorded_degenerate_output_fallback() -> None:
    assert callable(Cascade.process_without_clearervoice)
    assert callable(Cascade.process_with_pre_deepfilter_gain)
