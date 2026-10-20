"""Auditable ARPAbet → IPA → PHOIBLE row initialization."""
import csv
import re
import unicodedata
from pathlib import Path

# CMU/ARPAbet broad American English transcription. Length is not phonemic
# in ARPAbet: do not invent IPA length marks to force exact Ukrainian matches.
ARPABET = dict(zip(
    'AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG OW OY P R S SH T TH UH UW V W Y Z ZH'.split(),
    'ɑ æ ʌ ɔ aʊ aɪ b tʃ d ð ɛ ɝ eɪ f ɡ h ɪ i dʒ k l m n ŋ oʊ ɔɪ p ɹ s ʃ t θ ʊ u v w j z ʒ'.split(),
))
STRESS = {'': '', '0': '', '1': 'ˈ', '2': 'ˌ'}
PUNCT_ALIASES = {'…': '..', '“': "'", '”': "'", '«': "'", '»': "'", '"': "'"}
SPECIAL = {'<blank>', '<unk>', '<sos/eos>'}


def canonical(phone):
    return unicodedata.normalize('NFC', phone).replace('͡', '').replace('͜', '').replace('g', 'ɡ')


def split_stress(phone):
    return (phone[0], canonical(phone[1:])) if phone.startswith(('ˈ', 'ˌ')) else ('', canonical(phone))


def english_ipa(token):
    match = re.fullmatch(r'([A-Z]+)([012]?)', token)
    if not match or match[1] not in ARPABET:
        return None
    base = {'AH0': 'ə', 'ER0': 'ɚ'}.get(token, ARPABET[match[1]])
    return STRESS[match[2]] + base


class Features:
    def __init__(self, path):
        with Path(path).open(encoding='utf-8-sig') as stream:
            reader = csv.DictReader(stream, delimiter='\t')
            self.names = [k for k in reader.fieldnames if k not in {'segment', 'tone', 'stress'}]
            self.rows = {canonical(row['segment']): row for row in reader}

    def distance(self, a, b):
        a, b = self.rows.get(a), self.rows.get(b)
        if not a or not b or a['syllabic'] != b['syllabic']:
            return None
        known = [k for k in self.names if a[k] in {'+', '-', '0'} and b[k] in {'+', '-', '0'}]
        coverage = len(known) / len(self.names)
        if coverage < .9:
            return None
        return sum(a[k] != b[k] for k in known) / len(known), coverage


def map_tokens(source, target, features, max_distance=.15):
    if not 0 <= max_distance <= 1:
        raise ValueError('max_distance must be in [0, 1]')
    if len(set(target)) != len(target):
        raise ValueError('Duplicate target tokens would share semantic IDs')
    source_ipa = [english_ipa(token) for token in source]
    report = []
    for target_id, token in enumerate(target):
        row = dict(target_id=target_id, target_token=token, target_ipa=None,
                   source_id=None, source_token=None, source_ipa=None,
                   method='random', distance=None, coverage=None, stress_match=None,
                   reason='No reliable English equivalent or PHOIBLE neighbour')
        equivalent = token if token in source else PUNCT_ALIASES.get(token)
        punctuation = all(unicodedata.category(c).startswith('P') for c in token)
        if (token in SPECIAL or punctuation) and equivalent in source:
            row.update(source_id=source.index(equivalent), method='semantic', reason='Special token or punctuation by meaning')
        elif token not in SPECIAL and not punctuation and not token.startswith('('):
            stress, base = split_stress(token)
            row['target_ipa'] = token
            row['reason'] = ('Target IPA absent from pinned PHOIBLE and no equivalent English sound'
                             if base not in features.rows else
                             'No English candidate meeting syllabicity, coverage and distance thresholds')
            candidates = []
            for index, ipa in enumerate(source_ipa):
                if ipa is None:
                    continue
                en_stress, en_base = split_stress(ipa)
                metric = (0., 1.) if base == en_base else features.distance(base, en_base)
                if metric is not None and metric[0] <= max_distance:
                    candidates.append((base != en_base, metric[0], stress != en_stress, index, metric[1]))
            if candidates:
                approximate, distance, stress_diff, index, coverage = min(candidates)
                row.update(source_id=index, method='phoible' if approximate else 'equivalent',
                           distance=distance, coverage=coverage, stress_match=not stress_diff,
                           reason='Nearest articulatory features' if approximate else 'Same broad IPA sound')
                if stress_diff:
                    row['reason'] += '; English stress differs (independent target row)'
        elif punctuation or token in SPECIAL:
            row['reason'] = 'No semantically equivalent source punctuation or special token'
        else:
            row['reason'] = 'Language marker or tokenization artifact; no reliable sound mapping'
        if row['source_id'] is not None:
            row['source_token'] = source[row['source_id']]
            row['source_ipa'] = source_ipa[row['source_id']]
        report.append(row)
    return report
