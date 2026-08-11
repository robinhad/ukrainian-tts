from pathlib import Path

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
