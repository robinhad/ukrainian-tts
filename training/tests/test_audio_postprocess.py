import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from training.audio_enhancement import postprocess
from training.audio_enhancement.postprocess import (
    AudioPostprocessError,
    DeClickLimiterConfig,
    apply_overrides,
    load_config,
    process_audio_file,
    sha256,
)


def make_stereo(path: Path, sample_rate: int = 44_100) -> None:
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    left = 0.25 * np.sin(2 * np.pi * 220 * time)
    right = 0.20 * np.sin(2 * np.pi * 330 * time)
    left[10_000] = 1.0
    right[20_000] = -1.0
    sf.write(path, np.column_stack([left, right]), sample_rate, subtype="FLOAT")


def test_default_filter_chain_is_exact() -> None:
    config = DeClickLimiterConfig()

    assert config.filter_chain == (
        "aformat=sample_fmts=fltp,"
        "adeclick=w=55:o=75:a=2:t=4:b=2,"
        "alimiter=limit=0.891251:attack=5:release=80:"
        "level=false:latency=true"
    )
    assert len(config.digest) == 64


def test_config_file_and_cli_overrides(tmp_path: Path) -> None:
    path = tmp_path / "postprocess.yaml"
    path.write_text(
        "declick_peak_limit:\n" "  declick_window: 60\n" "  limiter_release_ms: 100\n",
        encoding="utf-8",
    )

    config = apply_overrides(load_config(path), {"limiter_attack_ms": 7})

    assert config.declick_window == 60
    assert config.limiter_release_ms == 100
    assert config.limiter_attack_ms == 7


def test_invalid_config_value_is_rejected() -> None:
    with pytest.raises(ValueError, match="declick_overlap"):
        DeClickLimiterConfig(declick_overlap=49)


def test_single_file_is_atomic_pcm24_and_preserves_stream_shape(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    target = tmp_path / "output.wav"
    make_stereo(source)
    source_hash = sha256(source)

    result = process_audio_file(source, target)

    info = sf.info(target)
    audio, sample_rate = sf.read(target, always_2d=True, dtype="float32")
    assert result["status"] == "PASS"
    assert result["source_preserved"] is True
    assert sha256(source) == source_hash
    assert target.is_file() and not target.is_symlink()
    assert info.format in {"WAV", "WAVEX"}
    assert info.subtype == "PCM_24"
    assert sample_rate == 44_100
    assert info.channels == audio.shape[1] == 2
    assert float(np.max(np.abs(audio))) <= 0.891251 + 2e-6
    assert not list(tmp_path.glob(".output.*.wav"))


def test_failed_ffmpeg_keeps_existing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.wav"
    target = tmp_path / "output.wav"
    make_stereo(source)
    sf.write(target, np.ones(1000) * 0.1, 16_000, subtype="PCM_24")
    target_hash = sha256(target)
    monkeypatch.setattr(postprocess, "validate_ffmpeg", lambda _: "test ffmpeg")
    monkeypatch.setattr(
        postprocess.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 9, "", "failure"),
    )

    with pytest.raises(AudioPostprocessError, match="exit 9"):
        process_audio_file(source, target, overwrite=True, ffmpeg_binary="fake")

    assert sha256(target) == target_hash
    assert not list(tmp_path.glob(".output.*.wav"))


def test_source_cannot_be_the_output(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    make_stereo(source)

    with pytest.raises(AudioPostprocessError, match="must differ"):
        process_audio_file(source, source, overwrite=True)


def test_batch_cli_processes_nested_files(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    report = tmp_path / "batch.json"
    input_root.mkdir()
    make_stereo(input_root / "one.wav", 32_000)
    (input_root / "nested").mkdir()
    make_stereo(input_root / "nested" / "two.wav", 48_000)

    run = subprocess.run(
        [
            "training/.venv/bin/python",
            "training/scripts/declick_and_limit_audio.py",
            "batch",
            "--input-dir",
            str(input_root),
            "--output-dir",
            str(output_root),
            "--recursive",
            "--report",
            str(report),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert run.returncode == 0, run.stderr
    summary = json.loads(report.read_text(encoding="utf-8"))
    assert summary["status"] == "PASS"
    assert summary["completed_files"] == 2
    assert sf.info(output_root / "one.wav").subtype == "PCM_24"
    assert sf.info(output_root / "nested" / "two.wav").subtype == "PCM_24"
    assert sf.info(output_root / "one.wav").samplerate == 32_000
    assert sf.info(output_root / "nested" / "two.wav").samplerate == 48_000
