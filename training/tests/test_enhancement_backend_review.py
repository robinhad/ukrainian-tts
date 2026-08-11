from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from training.audio_enhancement.review_pipeline import (
    ReviewProcessingConfig,
    SidonDeessOnlyConfig,
)
from training.scripts.build_enhancement_backend_review import (
    EXPECTED_DATASETS,
    copy_real_file,
    select_rows,
)


def test_review_config_has_no_compression() -> None:
    config = ReviewProcessingConfig()

    assert config.compression_applied is False
    assert "compress" not in config.mastering_filter.lower()
    assert config.highpass_hz == 70.0
    assert config.target_lufs == -23.0
    assert config.output_sample_rate == 24_000


def test_sidon_deess_profile_has_final_declick_and_limit() -> None:
    config = SidonDeessOnlyConfig()

    assert config.boundary_trim_applied is True
    assert config.sidon_applied is True
    assert config.deessing_applied is True
    assert config.post_highpass_applied is False
    assert config.loudness_normalization_applied is False
    assert config.compression_applied is False
    assert config.declicking_applied is True
    assert config.limiting_applied is True
    assert "deesser=" in config.deesser_filter
    assert "highpass" not in config.deesser_filter
    assert "loudnorm" not in config.deesser_filter
    assert "compress" not in config.deesser_filter
    assert config.final_postprocess.filter_chain == (
        "aformat=sample_fmts=fltp,"
        "adeclick=w=55:o=75:a=2:t=4:b=2,"
        "alimiter=limit=0.891251:attack=5:release=80:"
        "level=false:latency=true"
    )


def test_selection_is_balanced_and_includes_voa(tmp_path: Path) -> None:
    v3_rows = []
    v4_rows = []
    for dataset in sorted(EXPECTED_DATASETS):
        for index, duration in enumerate((2.0, 5.0, 10.0, 20.0)):
            uid = f"{dataset}-{index}"
            raw = tmp_path / f"{uid}-raw.wav"
            trim = tmp_path / f"{uid}-trim.wav"
            historical = tmp_path / f"{uid}-historical.wav"
            for path in (raw, trim, historical):
                path.touch()
            v3_rows.append({"utterance_id": uid, "audio_path": str(historical)})
            v4_rows.append(
                {
                    "utterance_id": uid,
                    "audio_path": str(trim),
                    "canonical_raw_audio_path": str(raw),
                    "source_id": dataset,
                    "split": f"{dataset}_train",
                    "duration": duration,
                    "text_sanitized": uid,
                }
            )

    selected = select_rows(pd.DataFrame(v3_rows), pd.DataFrame(v4_rows), count=2)

    assert len(selected) == 2 * len(EXPECTED_DATASETS)
    assert set(selected["source_id"]) == EXPECTED_DATASETS
    assert len(selected[selected["source_id"] == "voa_ukr_user_grant"]) == 2
    assert selected["utterance_id"].is_unique


def test_copy_real_file_does_not_make_a_symbolic_link(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    target = tmp_path / "nested" / "target.wav"
    sf.write(source, np.ones(2400, dtype=np.float32) * 0.01, 24_000)

    copy_real_file(source, target)

    assert target.is_file()
    assert not target.is_symlink()
    assert source.read_bytes() == target.read_bytes()
