import json

import pandas as pd
import pytest

from training.scripts.generate_checkpoint_listening import page
from training.transfer.listening import select_heldout


def test_transfer_panel_is_fixed_single_voice_and_disjoint():
    records = [dict(utterance_id=str(i), audio_sha256=f'hash-{i}', speaker_id='tetiana',
                    source_id='opentts_tetiana', split='quality_eval') for i in range(12)]
    records.append(dict(records[0], utterance_id='other', speaker_id='other'))
    evaluation = pd.DataFrame(records)
    used = pd.DataFrame([dict(utterance_id='train', audio_sha256='hash-train')])
    first = select_heldout(evaluation, used, 'tetiana', 'opentts_tetiana', 10)
    second = select_heldout(evaluation.sample(frac=1), used, 'tetiana', 'opentts_tetiana', 10)
    assert first == second and len(first) == 10
    assert all(row['speaker_id'] == 'tetiana' for row in first)
    with pytest.raises(ValueError, match='overlaps'):
        select_heldout(evaluation, pd.DataFrame([records[0]]), 'tetiana', 'opentts_tetiana', 10)
    used.loc[0, 'audio_sha256'] = 'hash-0'
    with pytest.raises(ValueError, match='overlaps'):
        select_heldout(evaluation, used, 'tetiana', 'opentts_tetiana', 10)


def test_transfer_page_does_not_claim_reference_speaker_conditioning(tmp_path):
    (tmp_path / 'report.json').write_text(json.dumps(dict(
        model_kind='phonetic_transfer', speaker_id='tetiana', sample_rate=22050,
        train_items=553, train_hours=.6578)))
    row = dict(source_id='opentts_tetiana', text='Тест', duration_seconds=2,
               synthesis='generated.wav', reference='original.wav', processed='processed.wav')
    result = page([row], 5000, output=tmp_path)
    assert 'voice: Tetiana' in result and '22,050 Hz' in result
    assert 'reference audio is for comparison only' in result
    assert 'Speaker conditioning uses the processed reference' not in result
    assert '553 recordings, 39.5 minutes' in result
    assert result.count('<audio controls') == 3
