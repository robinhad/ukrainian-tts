import json

import numpy as np
import pytest
import soundfile as sf

from training.quality.common import file_hash, write_json, write_tables
from training.scripts.select_normalized_best import choose, select


def test_selection_tie_cutoff_and_invalid_scores():
    assert choose(3.5, 3.5) == ('normalized_original', 3.5, True)
    assert choose(3.49, 3.4) == ('normalized_original', 3.49, False)
    assert choose(3.1, 3.8) == ('processed', 3.8, True)
    with pytest.raises(ValueError, match='Nonfinite'):
        choose(float('nan'), 4)


def test_selector_copies_exact_scored_audio_and_checks_provenance(tmp_path):
    records = []
    for name, score in [('normalized', 3.7), ('processed', 3.6)]:
        audio = tmp_path / f'{name}.wav'
        sf.write(audio, np.linspace(-.1, .1, 2400), 24000, subtype='PCM_24')
        row = {'source_id': 'source', 'utterance_id': 'item', 'audio_path': str(audio),
               'audio_sha256': file_hash(audio), 'reference_sha256': 'reference',
               'processing_input_sha256': 'canonical', 'sigmos_overall': score}
        write_json(audio.with_suffix('.json'), {'input_sha256': 'canonical', 'output_sha256': file_hash(audio)})
        write_tables(tmp_path, name, [row])
        records.append(row)
    decisions = select(tmp_path / 'normalized.jsonl', tmp_path / 'processed.jsonl', tmp_path / 'selected', copy_audio=True)
    assert decisions[0]['selected_variant'] == 'normalized_original'
    selected = json.loads((tmp_path / 'selected/retained.jsonl').read_text())
    assert file_hash(selected['audio_path']) == records[0]['audio_sha256']
    write_json(tmp_path / 'processed.json', {'input_sha256': 'different', 'output_sha256': records[1]['audio_sha256']})
    with pytest.raises(ValueError, match='Canonical processing input mismatch'):
        select(tmp_path / 'normalized.jsonl', tmp_path / 'processed.jsonl', tmp_path / 'bad')
