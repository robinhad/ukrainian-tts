import base64
import json
import re
import zipfile

import numpy as np
import pytest
import soundfile as sf

from training.quality.common import file_hash
from training.scripts.score_checkpoint_listening import MAPPING, ROLES, score_listening


class FakeModels:
    identity = {'test_model': True}

    def score(self, audio, rate):
        # Distinct signals ensure a role cannot silently inherit another role's score.
        value = float(audio.mean())
        return {**{f'sigmos_{name}': value for name in MAPPING.values()}, 'audiobox_pq': value}


def make_listening(root):
    row = {'utterance_id': 'heldout', 'source_id': 'source', 'text': 'Тест',
           'duration_seconds': 1, 'audio_sha256': {}}
    for i, role in enumerate(ROLES, 1):
        row[role] = f'{role}.wav'
        sf.write(root / row[role], np.full(24000, i / 10), 24000, subtype='PCM_24')
        row['audio_sha256'][role] = file_hash(root / row[role])
    (root / 'selection.jsonl').write_text(json.dumps(row) + '\n')
    (root / 'report.json').write_text(json.dumps({'sample_count': 1, 'checkpoint_step': 127000}))
    return row


def test_quality_is_bound_to_audio_roles_and_portable(tmp_path):
    row = make_listening(tmp_path)
    results = score_listening(tmp_path, None, models=FakeModels())
    assert [r['sigmos_overall'] for r in results] == pytest.approx([.1, .2, .3], abs=1e-6)
    assert [r['audio_sha256'] for r in results] == [row['audio_sha256'][role] for role in ROLES]
    html = (tmp_path / 'preview.html').read_text()
    assert html.count('SigMOS overall') == 5
    for score in ['0.100', '0.200', '0.300']:
        assert f'<td>{score}</td>' in html
    for name in ('quality.csv', 'quality.jsonl'):
        match = re.search(f'download="{name}" href="data:[^,]+,([^"]+)"', html)
        assert base64.b64decode(match[1]) == (tmp_path / name).read_bytes()
    with zipfile.ZipFile(tmp_path / 'listening_set.zip') as archive:
        assert archive.testzip() is None
        assert archive.read('quality.jsonl') == (tmp_path / 'quality.jsonl').read_bytes()


def test_modified_audio_rejected_before_report_changes(tmp_path):
    row = make_listening(tmp_path)
    before = (tmp_path / 'selection.jsonl').read_bytes()
    (tmp_path / row['processed']).write_bytes(b'changed waveform')
    with pytest.raises(ValueError, match='hash mismatch'):
        score_listening(tmp_path, None, models=FakeModels())
    assert (tmp_path / 'selection.jsonl').read_bytes() == before
    assert not (tmp_path / 'quality.jsonl').exists()
