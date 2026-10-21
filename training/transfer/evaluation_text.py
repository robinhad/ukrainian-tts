"""Repository stress marking before the checkpoint's frozen IPA frontend."""
import re

from .frontend import TransferFrontend


class StressPunctuationAdapter:
    """Allow punctuation-only gaps between stressed words to emit no phones."""
    def __init__(self, phonemizer):
        self.phonemizer = phonemizer

    def phonemize(self, text):
        try:
            return self.phonemizer.phonemize(text)
        except ValueError as error:
            if not any(c.isalnum() for c in text) and str(error).startswith('empty phoneme sequence for:'):
                return text, []
            raise


def plus_to_acute(text):
    """Repository +vowel notation -> combining acute after the same vowel."""
    result = re.sub(r'\+([аеєиіїоуюяАЕЄИІЇОУЮЯ])', lambda m: m[1] + '\u0301', text)
    if '+' in result:
        raise ValueError('Unconverted plus in repository stress output')
    return result


def prepare_text(text, frontend):
    from ukrainian_tts.formatter import preprocess_text
    from ukrainian_tts.stress import sentence_to_stress
    normalized = preprocess_text(text)
    marked = sentence_to_stress(normalized)
    stressed = plus_to_acute(marked)
    phones, ids = frontend.encode(stressed)
    return dict(original_text=text, normalized_text=normalized,
                repository_stress_text=marked, text=stressed,
                expected_phones=phones, token_ids=ids)


def prepare_panel(panel, tokens, bundle):
    frontend = TransferFrontend(tokens)
    frontend.verify(bundle/'frontend.json')
    frontend.phonemizer = StressPunctuationAdapter(frontend.phonemizer)
    return [dict(row, **prepare_text(row['text'], frontend)) for row in panel]


def evaluation_root(root, repository_stress):
    return root/'repository_stress' if repository_stress else root
