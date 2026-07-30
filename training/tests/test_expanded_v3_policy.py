import json
from pathlib import Path

from training.scripts.source_policy import (
    check_disk,
    load_registry,
    validate_record,
    validate_registry,
)


REGISTRY = Path(__file__).parents[1] / "conf" / "expanded_v3_sources.yaml"


def test_expanded_registry_is_default_deny_and_uses_30_gib_stop():
    registry = load_registry(REGISTRY)
    assert validate_registry(registry) == []
    assert registry["policy"]["free_disk_stop_gib"] == 30
    assert registry["sources"]["speech_uk_cv22_opus"]["decision"] == "deny"
    assert "YODAS" in registry["policy"]["deny_families"]
    assert "MLCommons" in registry["policy"]["deny_families"]


def test_registry_rejects_common_voice_derivative_record():
    registry = load_registry(REGISTRY)
    record = {
        "source_id": "speech_uk_cv22_opus",
        "source_license": "CC0-1.0",
        "audio_sha256_source": "a" * 64,
    }
    errors = validate_record(record, registry)
    assert any("denied" in error for error in errors)


def test_per_file_source_is_disabled_without_complete_license_evidence():
    registry = load_registry(REGISTRY)
    record = {
        "source_id": "tatoeba_uk",
        "source_license": "CC-BY-4.0",
        "audio_sha256_source": "a" * 64,
    }
    errors = validate_record(record, registry)
    assert any("denied" in error for error in errors)
    assert not registry["sources"]["tatoeba_uk"]["enabled"]


def test_policy_file_has_no_token_value():
    content = REGISTRY.read_text(encoding="utf-8")
    token_path = Path("/home/ballvan/Projects/hf_token.txt")
    if token_path.is_file():
        token = token_path.read_text(encoding="utf-8").strip()
        assert token
        assert token not in content
