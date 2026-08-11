import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from training.scripts.build_expanded_v7_declick_limited import (
    SPLIT_MAP,
    V7_NAME,
    profile,
)

ROOT = Path(__file__).resolve().parents[1]
FILTER_CHAIN = (
    "aformat=sample_fmts=fltp,"
    "adeclick=w=55:o=75:a=2:t=4:b=2,"
    "alimiter=limit=0.891251:attack=5:release=80:"
    "level=false:latency=true"
)
BEST_SHA256 = "3a8f3f40d00abfcaadc3f457a7485ddcc816cbb8a26d0cec193eeae39c7d7d48"


def test_v7_profile_keeps_the_final_stage_and_pcm24() -> None:
    report = {
        "filter_chain": FILTER_CHAIN,
        "processing_config_hash": "config-hash",
        "results": [
            {
                "processing_config": {"limiter_limit": 0.891251},
                "ffmpeg_version": "ffmpeg version 7.0.2-static",
                "source_preserved": True,
                "atomic_write": True,
            }
        ],
    }

    value = profile(report, "manifest-hash")

    assert value["name"] == V7_NAME
    assert value["final_filter_chain"] == FILTER_CHAIN
    assert value["format"] == "WAV/PCM_24"
    assert value["source_preserved"] is True
    assert value["atomic_write"] is True
    assert set(SPLIT_MAP.values()) == {
        f"{V7_NAME}_train",
        f"{V7_NAME}_dev",
        f"{V7_NAME}_eval",
    }


def test_v7_training_uses_both_gpus_and_v6_epoch94() -> None:
    script = (ROOT / "scripts/run_expanded_v7_declick_limited_training.sh").read_text(
        encoding="utf-8"
    )

    assert "GPU_COUNT=2" in script
    assert "best_checkpoints/94epoch.pth" in script
    assert BEST_SHA256 in script
    assert "--use_amp false" in script
    assert "--batch_bins ${BATCH_BINS}" in script


def test_direct_raw_data_keeps_pcm24_paths(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "manifests"
    data_root = tmp_path / "data"
    dump_dir = tmp_path / "dump"
    manifest_dir.mkdir()
    sets = {"sample_train": "train", "sample_dev": "dev", "sample_eval": "eval"}
    expected_paths = []
    for index, data_set in enumerate(sets):
        identifier = f"utt-{index}"
        audio = tmp_path / f"{identifier}.wav"
        sf.write(audio, np.ones(2400, dtype=np.float32) * 0.1, 24_000, subtype="PCM_24")
        expected_paths.append(str(audio.resolve()))
        pd.DataFrame(
            [
                {
                    "utterance_id": identifier,
                    "audio_path": str(audio.resolve()),
                    "sample_rate": 24_000,
                    "channels": 1,
                    "format": "WAV/PCM_24",
                }
            ]
        ).to_parquet(manifest_dir / f"{data_set}.parquet", index=False)
        directory = data_root / data_set
        directory.mkdir(parents=True)
        (directory / "wav.scp").write_text(
            f"{identifier} {audio.resolve()}\n", encoding="utf-8"
        )
        (directory / "text").write_text(f"{identifier} Тест.\n", encoding="utf-8")
        (directory / "utt2spk").write_text(f"{identifier} speaker\n", encoding="utf-8")
        (directory / "spk2utt").write_text(f"speaker {identifier}\n", encoding="utf-8")

    run = subprocess.run(
        [
            "training/.venv/bin/python",
            "training/scripts/prepare_direct_pcm24_raw_data.py",
            "--manifest-dir",
            str(manifest_dir),
            "--kaldi-data-root",
            str(data_root),
            "--dump-dir",
            str(dump_dir),
            "--train-set",
            "sample_train",
            "--valid-set",
            "sample_dev",
            "--test-set",
            "sample_eval",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert run.returncode == 0, run.stderr
    outputs = [
        dump_dir / "raw/org/sample_train/wav.scp",
        dump_dir / "raw/org/sample_dev/wav.scp",
        dump_dir / "raw/sample_eval/wav.scp",
    ]
    for output, expected in zip(outputs, expected_paths):
        assert (
            output.read_text(encoding="utf-8").strip().split(maxsplit=1)[1] == expected
        )
    assert not list(dump_dir.rglob("*.wav"))
