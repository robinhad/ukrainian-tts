from training.transfer.rhotic_diagnostic import align, normalize


def test_phone_alignment_preserves_r_l_substitution_and_deletion():
    reference = ['r', 'a', 'l', 'i', 'r']
    hypothesis = ['l', 'a', 'l', 'i']
    pairs = align(reference, hypothesis)
    assert pairs == [(0, 0), (1, 1), (2, 2), (3, 3), (4, None)]
    assert align(['a', 'r'], ['a', 't', 'r']) == [(0, 0), (None, 1), (1, 2)]
    assert align([], ['r']) == [(None, 0)]
    assert align(['r'], []) == [(0, None)]


def test_normalization_keeps_rhotic_distinctions_and_phone_boundaries():
    assert normalize(['(uk)', 'r', 'ˈa', ',', 'rʲ', 'ɹ', 'ʁ', 'l', 't͡ʃ', '.']) == [
        'r', 'a', 'rʲ', 'ɹ', 'ʁ', 'l', 'tʃ']


def test_summary_keeps_expected_r_and_l_separate():
    from training.transfer.rhotic_report import summary
    result = summary([{'condition': 'test', 'rhotics': [
        {'expected': 'r', 'recognized': 'l'},
        {'expected': 'r', 'recognized': '<deleted>'},
        {'expected': 'l', 'recognized': 'l'}]}])
    assert result['test']['r_counts'] == {'l': 1, '<deleted>': 1}
    assert result['test']['l_counts'] == {'l': 1}
