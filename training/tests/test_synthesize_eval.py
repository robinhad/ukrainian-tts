import json
import sys

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from training.scripts import synthesize_eval


@pytest.mark.parametrize("frames,expected", [(1200, 0), (0, 1)])
def test_quality_warnings_reach_evaluator_but_empty_audio_does_not(
    tmp_path, monkeypatch, frames, expected
):
    from training.quality import backends, evaluate

    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    sf.write(wav_dir / "example.wav", np.zeros(frames), 24000)
    manifest = tmp_path / "manifest.parquet"
    pd.DataFrame([{
        "utterance_id": "example", "text_raw": "тест", "text_sanitized": "тест",
        "espeak_phonemes": ["t"], "frontend_config_hash": "test",
        "espeak_version": "test",
    }]).to_parquet(manifest)
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"test model identity")
    config = tmp_path / "config.yaml"
    config.write_text("models: {}\n")
    panel = tmp_path / "panel.jsonl"
    panel.write_text("{}\n")
    output = tmp_path / "report.json"
    evaluated = []
    monkeypatch.setattr(backends, "Models", lambda config: object())
    monkeypatch.setattr(evaluate, "evaluate", lambda *a, **kw: evaluated.append(a))
    for name in ("UKTTS_QUALITY_ORIGINAL", "UKTTS_QUALITY_PREVIOUS", "UKTTS_QUALITY_BEST"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(sys, "argv", [
        "synthesize_eval", "--wav-dir", str(wav_dir), "--manifest", str(manifest),
        "--checkpoint", str(checkpoint), "--config", str(config),
        "--output", str(output), "--quality-panel", str(panel),
        "--quality-config", str(config),
    ])
    assert synthesize_eval.main() == expected
    report = json.loads(output.read_text())
    if frames:
        assert report["status"] == "PASS"
        assert report["duration_warnings"] == report["silence_warnings"] == 1
        assert report["errors"] == []
        assert len(evaluated) == 1
    else:
        assert any("empty waveform" in error for error in report["errors"])
        assert not evaluated
