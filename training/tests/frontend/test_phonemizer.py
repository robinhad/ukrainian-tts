from pathlib import Path

import pytest

from training.frontend.phonemize import (
    FrontendConfig,
    UkrainianPhonemizer,
    validate_pinned_config,
)


@pytest.fixture(scope="module")
def frontend(tmp_path_factory):
    try:
        return UkrainianPhonemizer(cache_path=tmp_path_factory.mktemp("cache") / "phones.sqlite3")
    except (ImportError, RuntimeError) as error:
        pytest.skip(f"frontend dependencies are not bootstrapped: {error}")


@pytest.mark.parametrize(
    "text",
    [
        "П'ять об'єктів.",
        "У 2026 році.",
        "Київ—Львів",
        "Python працює в Україні.",
        "Справді?! Так…",
    ],
)
def test_non_empty_and_deterministic(frontend, text):
    first = frontend.phonemize(text)
    second = frontend.phonemize(text)
    assert first == second
    assert first[1]


def test_stress_marker_is_preserved(frontend):
    _, tokens = frontend.phonemize("Український синтез мовлення.")
    assert any("ˈ" in token or "ˌ" in token for token in tokens)


def test_version_mismatch_is_rejected():
    with pytest.raises(RuntimeError, match="version"):
        validate_pinned_config(
            FrontendConfig(espeak_version="0.0.0", espeak_data_hash="invalid")
        )
