"""Minimal technical sanitation; intentionally not a verbalizer."""

from __future__ import annotations

import re
import unicodedata

SANITIZER_VERSION = "1"

_APOSTROPHES = str.maketrans(
    {"`": "'", "’": "'", "‘": "'", "ʼ": "'", "՚": "'", "＇": "'"}
)
_WHITESPACE_RE = re.compile(r"\s+")


def sanitize_text(text: str) -> str:
    """Apply only Unicode, apostrophe, control and whitespace cleanup."""

    if not isinstance(text, str):
        raise TypeError("text must be str")
    text = unicodedata.normalize("NFC", text).translate(_APOSTROPHES)
    cleaned: list[str] = []
    for char in text:
        category = unicodedata.category(char)
        cleaned.append(" " if category in {"Cc", "Cf"} or category.startswith("Z") else char)
    return _WHITESPACE_RE.sub(" ", "".join(cleaned)).strip()
