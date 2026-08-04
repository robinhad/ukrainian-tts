import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_expanded_v4_trim_only.py"


def test_trim_only_builder_filters_and_uses_correct_audio(tmp_path: Path) -> None:
    raw_files = []
    canonical_files = []
    for index in range(8):
        raw = tmp_path / f"raw-{index}.wav"
        clean = tmp_path / f"canonical-{index}.wav"
        raw.write_bytes(b"raw")
        clean.write_bytes(b"clean")
        raw_files.append(raw)
        canonical_files.append(clean)
    splits = [
        "expanded_v3_train",
        "expanded_v3_train",
        "expanded_v3_train",
        "expanded_v3_train",
        "expanded_v3_dev",
        "expanded_v3_dev",
        "expanded_v3_eval",
        "expanded_v3_eval",
    ]
    active = pd.DataFrame(
        {
            "utterance_id": [f"u{i}" for i in range(8)],
            "speaker_id": ["s"] * 8,
            "split": splits,
            "audio_path": [str(tmp_path / "old.wav")] * 8,
            "canonical_raw_audio_path": [str(path) for path in raw_files],
            "duration": [3.0] * 8,
            "sample_rate": [24000] * 8,
            "qc_flags": [[] for _ in range(8)],
        }
    )
    canonical = active.copy()
    canonical["audio_path"] = [str(path) for path in canonical_files]
    canonical["audio_sha256"] = [f"hash-{i}" for i in range(8)]
    canonical["duration"] = [3.0, 3.0, 3.0, 3.0, 3.0, 1.9, 3.0, 3.0]
    canonical["qc_flags"] = [[], ["clipping"], [], [], [], ["too_short"], [], []]
    active_path = tmp_path / "active.parquet"
    canonical_path = tmp_path / "canonical.parquet"
    active.to_parquet(active_path, index=False)
    canonical.to_parquet(canonical_path, index=False)
    output = tmp_path / "manifests"
    raw_manifest = tmp_path / "raw.parquet"
    report = tmp_path / "report.json"

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--active-manifest",
            str(active_path),
            "--canonical-manifest",
            str(canonical_path),
            "--output-dir",
            str(output),
            "--raw-manifest",
            str(raw_manifest),
            "--report",
            str(report),
        ],
        check=True,
    )
    result = pd.read_parquet(output / "all.parquet")
    raw_result = pd.read_parquet(raw_manifest)
    summary = json.loads(report.read_text())

    assert set(result.utterance_id) == {"u0", "u2", "u3", "u4", "u6", "u7"}
    assert all("canonical-" in path for path in result.audio_path)
    assert all("raw-" in path for path in raw_result.audio_path)
    assert not result.deepfilternet_applied.any()
    assert not result.highpass_applied.any()
    assert not result.deessing_applied.any()
    assert not result.compression_applied.any()
    assert not result.loudness_normalization_applied.any()
    assert summary["rejected"] == {
        "clipping": 1,
        "too_long": 0,
        "too_short": 1,
        "unique_records": 2,
    }
    assert set(result.split) == {
        "expanded_v4_trim_only_train",
        "expanded_v4_trim_only_dev",
        "expanded_v4_trim_only_eval",
    }
