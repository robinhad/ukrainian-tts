"""One fail-closed frontend for data preparation and inference."""
import hashlib
import inspect
import json
import re
from dataclasses import asdict
from pathlib import Path

from training.frontend.phonemize import UkrainianPhonemizer
from training.frontend.sanitize import sanitize_text

VOWELS = set('аеєиіїоуюяАЕЄИІЇОУЮЯ')
IPA_VOWELS = set('aeiouyɑɐɒæəɛɜɞɪɔœøɵɶʉʊʌɨɯɤ')
ACUTE_WORD = re.compile(r"[А-Яа-яІіЇїЄєҐґ'\u0301]+")


class TransferFrontend:
    def __init__(self, tokens):
        self.phonemizer = UkrainianPhonemizer()
        self.tokens = list(tokens)
        self.ids = {token: i for i, token in enumerate(tokens)}
        if len(self.ids) != len(tokens):
            raise ValueError('Duplicate token inventory')

    def snapshot(self):
        import espnet2.text.phoneme_tokenizer as implementation
        # Includes implementation hashes, not local filesystem paths.
        return dict(config=asdict(self.phonemizer.config), tokens=self.tokens,
                    code_hashes={name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                 for name, path in [('transfer', __file__),
                                     ('phonemizer', inspect.getfile(UkrainianPhonemizer)),
                                     ('sanitizer', inspect.getfile(sanitize_text)),
                                     ('espnet_tokenizer', implementation.__file__)]})

    def verify(self, path):
        if json.loads(Path(path).read_text()) != self.snapshot():
            raise ValueError('Frontend or inventory differs from converted checkpoint; regenerate the bundle/data')

    def phonemes(self, text):
        text = sanitize_text(text)
        if '\u0301' not in text:
            return self.phonemizer.phonemize(text)[1]
        # eSpeak ignores Ukrainian combining acute accents. Apply explicit lexical
        # stress to its vowel PHONE, never split a phone into Unicode characters.
        output, cursor = [], 0
        for match in ACUTE_WORD.finditer(text):
            word = match[0]
            if '\u0301' not in word:
                continue
            prefix = text[cursor:match.start()].strip()
            if prefix:
                output.extend(self.phonemizer.phonemize(prefix)[1])
            if word.count('\u0301') != 1:
                raise ValueError('Expected one lexical stress accent per marked word')
            accent = word.index('\u0301')
            if accent == 0 or word[accent - 1] not in VOWELS:
                raise ValueError('Stress accent must follow a Ukrainian vowel')
            ordinal = sum(c in VOWELS for c in word[:accent]) - 1
            phones = self.phonemizer.phonemize(word.replace('\u0301', ''))[1]
            phones = [p.lstrip('ˈˌ') for p in phones]
            vowels = [i for i, phone in enumerate(phones) if phone[0] in IPA_VOWELS]
            if len(vowels) != sum(c in VOWELS for c in word):
                raise ValueError('Cannot reliably align explicit lexical stress to eSpeak vowels')
            phones[vowels[ordinal]] = 'ˈ' + phones[vowels[ordinal]]
            output.extend(phones)
            cursor = match.end()
        if cursor == 0:
            raise ValueError('Unsupported stress annotation')
        if text[cursor:].strip():
            output.extend(self.phonemizer.phonemize(text[cursor:])[1])
        return output

    def encode(self, text):
        phones = self.phonemes(text)
        unknown = sorted(set(phones) - self.ids.keys())
        if unknown:
            raise ValueError(f'Phones missing from target inventory: {unknown}')
        if any(p in {'<blank>', '<unk>', '<sos/eos>'} for p in phones):
            raise ValueError('Special IDs must not appear in input text')
        return phones, [self.ids[p] for p in phones]
