import unicodedata

import pytest

from training.frontend.sanitize import sanitize_text


def test_unicode_nfc():
    text = "Украі\u0308на"
    result = sanitize_text(text)
    assert unicodedata.is_normalized("NFC", result)
    assert result == "Україна"


@pytest.mark.parametrize("apostrophe", ["`", "’", "‘", "ʼ", "՚", "＇"])
def test_apostrophes(apostrophe):
    assert sanitize_text(f"п{apostrophe}ять") == "п'ять"


def test_controls_and_spaces():
    assert sanitize_text("  раз\tдва\nтри\u00a0чотири\u200bп'ять  ") == "раз два три чотири п'ять"


def test_no_verbalization_or_lowercasing():
    assert sanitize_text("OpenAI: 2026 рік, 10 USD") == "OpenAI: 2026 рік, 10 USD"


def test_empty_and_type_errors():
    assert sanitize_text("\x00 \t") == ""
    with pytest.raises(TypeError):
        sanitize_text(None)
