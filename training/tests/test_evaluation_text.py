import pytest

from training.transfer.evaluation_text import plus_to_acute, evaluation_root, StressPunctuationAdapter


def test_repository_stress_notation_preserves_text_and_word_boundaries():
    marked = "прив+іт, +як теб+е зв+ати? п'ятн+иця"
    converted = plus_to_acute(marked)
    assert converted == "приві́т, я́к тебе́ зва́ти? п'ятни́ця"
    assert converted.replace('\u0301', '') == marked.replace('+', '')
    assert plus_to_acute('ба́бин') == 'ба́бин'


def test_uninterpreted_plus_fails_instead_of_silent_token_change():
    with pytest.raises(ValueError):
        plus_to_acute('два + два')


def test_frontend_evaluations_have_separate_history(tmp_path):
    assert evaluation_root(tmp_path, True) == tmp_path/'repository_stress'
    assert evaluation_root(tmp_path, False) == tmp_path


def test_empty_punctuation_is_allowed_but_missing_word_phones_fail():
    class Empty:
        def phonemize(self, text):
            raise ValueError(f'empty phoneme sequence for: {text!r}')
    adapter = StressPunctuationAdapter(Empty())
    assert adapter.phonemize('-') == ('-', [])
    with pytest.raises(ValueError):
        adapter.phonemize('ранок')
