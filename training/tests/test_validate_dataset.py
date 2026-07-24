from training.scripts.validate_dataset import text_sequence_errors


def test_text_sequence_errors_rejects_extreme_phoneme_sequence():
    errors = text_sequence_errors(
        "utt-1",
        "Короткий текст.",
        ["a"] * 501,
        max_text_characters=500,
        max_phoneme_tokens=500,
    )

    assert errors == ["utt-1: phonemes have 501 tokens; maximum=500"]


def test_text_sequence_errors_accepts_normal_sequence():
    assert text_sequence_errors(
        "utt-1",
        "Короткий текст.",
        ["k", "o"],
        max_text_characters=500,
        max_phoneme_tokens=500,
    ) == []
