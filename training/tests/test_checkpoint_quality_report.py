import pytest

from training.scripts.report_checkpoint_quality import FIELDS, portable_row


def test_removes_paths_and_transcripts_but_keeps_worst_segments():
    row = {'utterance_id': 'item_1', 'audio_path': '/private/audio.wav',
           'text': 'private transcript', 'whisper_text': 'private recognition',
           'sigmos_overall': 3.2, 'worst_segments': {'sigmos_overall': [
               {'start_seconds': 1.5, 'end_seconds': 4.5, 'score': 2.9}]}}
    result = portable_row(row, FIELDS['per_file'], '100k')
    assert set(result) == {'checkpoint', 'utterance_id', 'sigmos_overall', 'worst_segments'}
    assert result['worst_segments'] == row['worst_segments']


def test_rejects_paths_in_allowed_fields():
    with pytest.raises(ValueError, match='path-like'):
        portable_row({'utterance_id': '/private/item'}, FIELDS['per_file'], '100k')
