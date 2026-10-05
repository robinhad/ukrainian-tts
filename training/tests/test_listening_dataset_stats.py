import pandas as pd
import pytest

from training.scripts.listening_dataset_stats import summarize


def frame():
    return pd.DataFrame([
        {'utterance_id': 'a', 'speaker_id': 'same', 'source_id': 'one', 'duration': 3600, 'split': 'test_train'},
        {'utterance_id': 'b', 'speaker_id': 'same', 'source_id': 'two', 'duration': 1800, 'split': 'test_train'},
        {'utterance_id': 'c', 'speaker_id': 'c', 'source_id': 'two', 'duration': 900, 'split': 'test_train'},
    ])


def test_totals_use_hours_and_global_speaker_ids_with_placeholder_caveat():
    report = summarize(frame())
    assert report['total_hours'] == 1.75
    assert report['total_speaker_ids'] == 2  # Same ID across sources must not be double-counted.
    assert report['recording_level_speaker_ids'] == 1
    assert report['verified_unique_people'] is None
    assert sum(row['hours'] for row in report['datasets']) == 1.75
    assert sum(row['hours'] for row in report['speakers']) == 1.75
    assert report['speakers'][0]['hours'] == 1.5


@pytest.mark.parametrize('field,value', [('split', 'test_eval'), ('duration', -1), ('speaker_id', None)])
def test_invalid_or_nontraining_rows_are_rejected(field, value):
    data = frame()
    data.loc[0, field] = value
    with pytest.raises(ValueError):
        summarize(data)
