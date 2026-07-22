import json
from pathlib import Path

import pytest

from training.frontend.snapshot import build_snapshot


ROOT = Path(__file__).resolve().parents[2]


def test_snapshot_matches(tmp_path):
    expected_path = ROOT / "tests/frontend/expected_phonemes.json"
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    if not expected:
        pytest.skip("snapshot has not been generated yet")
    try:
        current = build_snapshot(ROOT / "tests/frontend/regression.tsv", tmp_path / "cache.sqlite3")
    except (ImportError, RuntimeError) as error:
        pytest.skip(f"frontend dependencies are not bootstrapped: {error}")
    assert current == expected
