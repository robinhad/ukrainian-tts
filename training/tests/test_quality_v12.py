import pytest

from training.scripts.process_quality_v12 import retention, transcript


def test_threshold_is_inclusive_and_uses_processed_score_only():
    assert retention(3.5, 'quality_v10_train') == (True, True)
    assert retention(3.499999, 'quality_v10_train') == (False, False)
    assert retention(3.499999, 'quality_v10_dev') == (False, False)


def test_holdout_population_is_not_quality_filtered():
    assert retention(1.5, 'quality_v10_eval') == (False, True)
    assert retention(3.5, 'quality_v10_eval') == (True, True)


@pytest.mark.parametrize('score', [float('nan'), float('inf'), 0., 6.])
def test_invalid_scores_cannot_pass_filter(score):
    with pytest.raises(ValueError):
        retention(score, 'quality_v10_train')


def test_mfa_transcript_preserves_unsupported_tokens_for_review():
    text, unsupported = transcript('Приві\u0301т, світе! 123 ABC')
    assert text == 'привіт світе 123 abc'
    assert unsupported == ['123', 'abc']


def test_alignment_batches_keep_split_context_on_resume(tmp_path, monkeypatch):
    import hashlib
    import json
    from types import SimpleNamespace
    from training.scripts import materialize_non_voa
    from training.scripts.process_quality_v12 import source_batches

    contents = [str(i).encode() for i in range(7)]
    splits = ['train', 'eval', 'train', 'train', 'eval', 'train', 'train']
    rows = [{'utterance_id': str(i), 'source_id': 'example',
             'audio_sha256_source': hashlib.sha256(content).hexdigest(),
             'split': 'quality_v10_' + split}
            for i, (content, split) in enumerate(zip(contents, splits))]
    downloads = tmp_path / 'downloads.json'
    downloads.write_text(json.dumps({'example': {}}))
    monkeypatch.setattr(materialize_non_voa, 'input_rows',
                        lambda _: iter({'audio': {'bytes': c}} for c in contents))
    args = SimpleNamespace(downloads=downloads, alignment_chunk_size=2)
    def ids(completed):
        batches = list(source_batches(args, rows, completed))
        assert all(len({r['split'] for r, _ in batch}) == 1 for batch in batches)
        return [[r['utterance_id'] for r, _ in batch] for batch in batches]
    assert ids(set()) == [['0', '2'], ['1', '4'], ['3', '5'], ['6']]
    assert ids({'0'}) == ids(set())
    assert ids({'0', '2', '3'}) == [['1', '4'], ['3', '5'], ['6']]
    assert ids({str(i) for i in range(7)}) == []
