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


def test_full_snapshot_has_compact_punctuation_tokens():
    snapshot = json.loads(
        (ROOT / "tests/frontend/expected_phonemes_full.json").read_text(
            encoding="utf-8"
        )
    )
    tokens = {
        token
        for case in snapshot["cases"].values()
        for token in case["tokens"]
    }
    punctuation = set(",.;:!?—–-…«»“”\"()[]{}")
    mixed = {
        token
        for token in tokens
        if token not in {"(uk)", "(en)"}
        and any(character in punctuation for character in token)
        and any(character not in punctuation for character in token)
    }
    assert not mixed
    assert len(tokens) < 150
