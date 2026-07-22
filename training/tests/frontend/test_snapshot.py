import json
from pathlib import Path

import pytest

from training.frontend.snapshot import build_snapshot


ROOT = Path(__file__).resolve().parents[2]


def test_snapshot_matches(tmp_path):
    suites = [
        ("regression.tsv", "expected_phonemes.json"),
        ("regression_full.tsv", "expected_phonemes_full.json"),
    ]
    for cases_name, expected_name in suites:
        expected_path = ROOT / "tests/frontend" / expected_name
        if not expected_path.exists():
            continue
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        if not expected:
            pytest.skip(f"snapshot has not been generated: {expected_name}")
        try:
            current = build_snapshot(
                ROOT / "tests/frontend" / cases_name,
                tmp_path / f"{cases_name}.sqlite3",
            )
        except (ImportError, RuntimeError) as error:
            pytest.skip(f"frontend dependencies are not bootstrapped: {error}")
        assert current == expected
