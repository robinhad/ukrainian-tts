import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "validate_listening_voice_eval.py"
)


def make_inputs(tmp_path: Path) -> tuple[Path, Path, str]:
    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    text = "Український тест."
    voices = []
    waveform = 0.1 * np.sin(
        2 * np.pi * 220 * np.arange(2400, dtype=np.float32) / 24000
    )
    for index in range(1, 6):
        voice = f"voice_{index:02d}"
        wav = wav_dir / f"{voice}.wav"
        sf.write(wav, waveform, 24000, subtype="PCM_16")
        wav.with_suffix(".wav.json").write_text(
            json.dumps({"text_raw": text, "real_time_factor": 0.1}),
            encoding="utf-8",
        )
        voices.append({"voice": voice})
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"voices": voices}), encoding="utf-8")
    return report, wav_dir, text


def run_validator(
    report: Path, wav_dir: Path, text: str, *extra: str
) -> dict:
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--report",
            str(report),
            "--wav-dir",
            str(wav_dir),
            "--text",
            text,
            *extra,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(report.read_text(encoding="utf-8"))


def test_new_output_needs_human_listening(tmp_path: Path) -> None:
    report, wav_dir, text = make_inputs(tmp_path)
    result = run_validator(report, wav_dir, text)

    assert result["status"] == "PASS"
    assert result["automatic_validation"]["status"] == "PASS"
    assert result["human_listening"] == {
        "issues": [],
        "notes": [],
        "source": None,
        "status": "NOT_RUN",
    }
    assert result["release_status"] == "NOT_READY"
    assert "perceptual_issue" not in result


def test_failed_human_listening_blocks_release(tmp_path: Path) -> None:
    report, wav_dir, text = make_inputs(tmp_path)
    result = run_validator(
        report,
        wav_dir,
        text,
        "--human-listening-status",
        "FAIL",
        "--perceptual-issue",
        "raspy_audio",
        "--perceptual-issue",
        "intermittent_robotic_quality",
        "--human-listening-note",
        "The user hears artifacts.",
    )

    assert result["status"] == "PASS"
    assert result["human_listening"]["status"] == "FAIL"
    assert result["human_listening"]["issues"] == [
        "raspy_audio",
        "intermittent_robotic_quality",
    ]
    assert result["release_status"] == "FAIL"
