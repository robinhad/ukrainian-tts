"""Pinned ESPnet/eSpeak-ng Ukrainian phonemization with an SQLite cache."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from .sanitize import SANITIZER_VERSION, sanitize_text


TRAINING_ROOT = Path(__file__).resolve().parents[1]


def configure_pinned_espeak() -> Path | None:
    """Select the repository-local eSpeak runtime when no override is supplied."""

    install = TRAINING_ROOT / "vendor" / "espeak-ng-install"
    data = install / "share" / "espeak-ng-data"
    libraries = sorted((install / "lib").glob("libespeak-ng.so*"))
    if data.is_dir() and libraries:
        os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", str(libraries[0]))
        os.environ.setdefault("ESPEAK_DATA_PATH", str(data))
    value = os.environ.get("ESPEAK_DATA_PATH")
    return Path(value) if value else None


@dataclass(frozen=True)
class FrontendConfig:
    g2p: str = "espeak_ng_ukrainian"
    language: str = "uk"
    with_stress: bool = True
    preserve_punctuation: bool = True
    frontend_version: str = "uk_espeak_v2"
    sanitizer_version: str = SANITIZER_VERSION
    espeak_version: str = "unknown"
    espeak_data_hash: str = "unknown"

    @property
    def digest(self) -> str:
        payload = json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def hash_tree(root: Path) -> str:
    """Hash file names and contents in stable order."""

    digest = hashlib.sha256()
    if not root.is_dir():
        raise FileNotFoundError(f"eSpeak data directory does not exist: {root}")
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def detect_espeak_version() -> str:
    from phonemizer.backend import EspeakBackend

    return ".".join(str(part) for part in EspeakBackend.version())


def default_config() -> FrontendConfig:
    data_root = configure_pinned_espeak()
    config = FrontendConfig(
        espeak_version=detect_espeak_version(),
        espeak_data_hash=hash_tree(data_root) if data_root else "unknown",
    )
    validate_pinned_config(config)
    return config


def validate_pinned_config(config: FrontendConfig) -> None:
    """Fail closed when inference would use a different frontend snapshot."""

    version_file = TRAINING_ROOT / "vendor" / "ESPEAK_NG_VERSION"
    hash_file = TRAINING_ROOT / "vendor" / "ESPEAK_NG_DATA_HASH"
    if not version_file.exists() or not hash_file.exists():
        return
    expected_version = version_file.read_text(encoding="utf-8").strip()
    expected_hash = hash_file.read_text(encoding="utf-8").strip()
    errors = []
    if config.espeak_version != expected_version:
        errors.append(f"eSpeak version {config.espeak_version} != pinned {expected_version}")
    if config.espeak_data_hash != expected_hash:
        errors.append("eSpeak language data hash differs from the pinned snapshot")
    if errors:
        raise RuntimeError("; ".join(errors))


class UkrainianPhonemizer:
    def __init__(self, config: FrontendConfig | None = None, cache_path: Path | None = None) -> None:
        self.config = config or default_config()
        self.cache_path = cache_path
        self._tokenizer = None
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(cache_path) as database:
                database.execute(
                    "CREATE TABLE IF NOT EXISTS phonemes "
                    "(cache_key TEXT PRIMARY KEY, tokens_json TEXT NOT NULL)"
                )

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from espnet2.text.phoneme_tokenizer import PhonemeTokenizer

            self._tokenizer = PhonemeTokenizer(g2p_type=self.config.g2p)
        return self._tokenizer

    def phonemize(self, text: str) -> tuple[str, list[str]]:
        sanitized = sanitize_text(text)
        if not sanitized:
            raise ValueError("text is empty after sanitation")
        key = hashlib.sha256(f"{self.config.digest}\0{sanitized}".encode()).hexdigest()
        cached = self._read_cache(key)
        if cached is not None:
            return sanitized, cached
        tokens = list(self.tokenizer.text2tokens(sanitized))
        if not tokens or not any(token.strip() for token in tokens):
            raise ValueError(f"empty phoneme sequence for: {sanitized!r}")
        self._write_cache(key, tokens)
        return sanitized, tokens

    def _read_cache(self, key: str) -> list[str] | None:
        if self.cache_path is None:
            return None
        with sqlite3.connect(self.cache_path) as database:
            row = database.execute(
                "SELECT tokens_json FROM phonemes WHERE cache_key = ?", (key,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def _write_cache(self, key: str, tokens: Sequence[str]) -> None:
        if self.cache_path is None:
            return
        with sqlite3.connect(self.cache_path) as database:
            database.execute(
                "INSERT OR REPLACE INTO phonemes(cache_key, tokens_json) VALUES (?, ?)",
                (key, json.dumps(list(tokens), ensure_ascii=False)),
            )
