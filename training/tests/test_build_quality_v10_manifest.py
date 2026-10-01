import json

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from training.quality.common import file_hash
from training.scripts.build_quality_v10_manifest import build_manifest


def inputs(tmp_path, **extra):
    raw = tmp_path / 'raw.wav'
    processed = tmp_path / 'processed'
    processed.mkdir()
    audio = np.sin(np.arange(2400) * 0.1).astype(np.float32) * 0.1
    sf.write(raw, audio, 24000, subtype='PCM_24')
    output = processed / 'sample.wav'
    sf.write(output, audio * 0.9, 24000, subtype='PCM_24')
    profile = tmp_path / 'profile.yaml'
    profile.write_text('backend: identity\n')
    output.with_suffix('.json').write_text(json.dumps({
        'input_sha256': file_hash(raw), 'output_sha256': file_hash(output),
        'profile': {'backend': 'identity'},
    }))
    manifest = tmp_path / 'raw.parquet'
    pd.DataFrame([{
        'utterance_id': 'sample', 'source_id': 'example_uk',
        'audio_path': str(raw), 'split': 'quality_v10_train',
        'text_raw': 'Привіт', 'espeak_phonemes': ['p', 'r', 'ɪ'],
        'qc_flags': [], **extra,
    }]).to_parquet(manifest, index=False)
    return manifest, processed, profile, tmp_path / 'records.jsonl'


def test_real_parquet_array_columns_survive_export(tmp_path):
    manifest, processed, profile, destination = inputs(tmp_path)
    assert isinstance(pd.read_parquet(manifest).to_dict('records')[0]['espeak_phonemes'],
                      np.ndarray)
    build_manifest(manifest, processed, profile, destination)
    row = json.loads(destination.read_text())
    assert row['espeak_phonemes'] == ['p', 'r', 'ɪ']
    assert row['qc_flags'] == []
    assert row['text_raw'] == 'Привіт'
    assert row['split'] == 'quality_v10_train'
    assert row['audio_sha256'] == file_hash(processed / 'sample.wav')
    assert row['format'] == 'WAV/PCM_24'
    assert row['duration'] == 0.1
    assert json.loads(destination.with_suffix('.summary.json').read_text())['records'] == 1


def test_stale_audio_cannot_replace_existing_manifest(tmp_path):
    manifest, processed, profile, destination = inputs(tmp_path)
    destination.write_text('previous manifest\n')
    sf.write(processed / 'sample.wav', np.zeros(2400), 24000, subtype='PCM_24')
    with pytest.raises(ValueError, match='Stale processing output'):
        build_manifest(manifest, processed, profile, destination)
    assert destination.read_text() == 'previous manifest\n'


def test_serialization_failure_does_not_publish_partial_manifest(tmp_path):
    manifest, processed, profile, destination = inputs(tmp_path, invalid=float('inf'))
    destination.write_text('previous manifest\n')
    with pytest.raises(ValueError, match='Out of range float'):
        build_manifest(manifest, processed, profile, destination)
    assert destination.read_text() == 'previous manifest\n'
    assert not list(tmp_path.glob('.records.jsonl.*.tmp'))
