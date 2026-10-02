import pytest

from training.quality.mfa_trim import boundaries


def alignment(entries):
    return {'tiers': {'phones': {'entries': entries}}}


def test_boundary_padding_preserves_internal_silence():
    grid = alignment([[0, .4, 'sil'], [.4, .9, 'a_B'], [.9, 1.4, 'sil'], [1.4, 2, 's_E'], [2, 2.5, 'sil']])
    start, end, flags = boundaries(grid, 2.5)
    assert start == pytest.approx(.3) and end == pytest.approx(2.1)
    assert not flags


@pytest.mark.parametrize('entries,flag', [
    ([], 'no_aligned_speech'),
    ([[0, 1, 'sil']], 'no_aligned_speech'),
    ([[0, .5, 'spn'], [.5, 1, 'a']], 'unknown_phone'),
    ([[.5, .4, 'a']], 'invalid_phone_intervals'),
    ([[float('nan'), 1, 'a']], 'invalid_phone_intervals'),
    ([[0, .8, 'a'], [.5, 1, 'b']], 'invalid_phone_intervals'),
    ([[.5, .6, 'a']], 'short_speech_span'),
    ([[1.5, 2.1, 'a']], 'large_boundary_removal'),
])
def test_questionable_alignment_is_flagged(entries, flag):
    assert flag in boundaries(alignment(entries), 4)[2]


def test_missing_alignment_and_edges():
    assert boundaries({}, 2) == (0, 2, ['missing_or_multiple_phone_tiers'])
    assert boundaries(alignment([[0, 2, 'a']]), 2) == (0, 2, [])
