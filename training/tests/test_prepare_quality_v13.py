import json

import pandas as pd
import pytest

from training.scripts.prepare_quality_v13 import filter_keyed, select_training


def fixture_data(tmp_path, scores):
    records = []
    for i, score in enumerate(scores):
        identifier = f'item{i}'
        row = {'utterance_id': identifier, 'source_id': 'source', 'duration': 2.0,
               'audio_sha256': f'hash{i}'}
        records.append(row)
        (tmp_path / f'{identifier}.json').write_text(json.dumps({
            'utterance_id': identifier, 'output_sha256': row['audio_sha256'],
            'heldout': False, 'processed_scores': {'MOS_OVRL': score}}))
    return pd.DataFrame(records)


def test_inclusive_processed_threshold(tmp_path):
    frame = fixture_data(tmp_path, [3.9999, 4.0, 4.2])
    assert [r['utterance_id'] for r in select_training(frame, tmp_path)] == ['item1', 'item2']


@pytest.mark.parametrize('change', [{'heldout': True}, {'output_sha256': 'changed'}])
def test_rejects_heldout_or_unbound_scores(tmp_path, change):
    frame = fixture_data(tmp_path, [4.2])
    path = tmp_path / 'item0.json'
    path.write_text(json.dumps({**json.loads(path.read_text()), **change}))
    with pytest.raises(ValueError):
        select_training(frame, tmp_path)


def test_filtered_indices_preserve_values_and_reject_missing_keys(tmp_path):
    source, output = tmp_path / 'source', tmp_path / 'output'
    source.write_text('keep feature.ark:42\nremove feature.ark:99\n')
    filter_keyed(source, output, {'keep'})
    assert output.read_text() == 'keep feature.ark:42\n'
    with pytest.raises(ValueError, match='Missing'):
        filter_keyed(source, output, {'absent'})
