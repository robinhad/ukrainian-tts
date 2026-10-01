import numpy as np
import pytest
import soundfile as sf

from training.quality.common import file_hash, write_tables
from training.quality.processing import process


def test_pipelined_processing_uses_canonical_input_and_preserves_audio(tmp_path):
    original = tmp_path / 'native.wav'
    canonical = tmp_path / 'canonical.wav'
    sf.write(original, np.zeros(1600), 16000)
    sf.write(canonical, np.linspace(-.3, .3, 2400), 24000, subtype='PCM_24')
    rows = [{'utterance_id': f'item{i}', 'source_id': 'a', 'audio_path': str(original),
             'reference_sha256': file_hash(original), 'processing_audio_path': str(canonical),
             'processing_input_sha256': file_hash(canonical)} for i in range(12)]
    write_tables(tmp_path, 'panel', rows)
    profile = tmp_path / 'profile.yaml'
    profile.write_text('backend: identity\nwet: 1.0\n')
    one = process(tmp_path / 'panel.jsonl', profile, tmp_path / 'one', device='cpu', cpu_workers=1)
    four = process(tmp_path / 'panel.jsonl', profile, tmp_path / 'four', device='cpu', cpu_workers=4)
    assert [r['output_sha256'] for r in one] == [r['output_sha256'] for r in four]
    expected, _ = sf.read(canonical)
    for result in four:
        actual, rate = sf.read(result['output_path'])
        assert rate == 24000 and np.array_equal(actual, expected)
    resumed = process(tmp_path / 'panel.jsonl', profile, tmp_path / 'four', device='cpu', resume=True)
    assert [r['output_sha256'] for r in resumed] == [r['output_sha256'] for r in four]
    sf.write(canonical, expected * .5, 24000, subtype='PCM_24')
    with pytest.raises(ValueError, match='Frozen input changed'):
        process(tmp_path / 'panel.jsonl', profile, tmp_path / 'four', device='cpu', resume=True)


def test_postprocessing_worker_errors_propagate(tmp_path):
    audio = tmp_path / 'silent.wav'
    sf.write(audio, np.zeros(2400), 24000)
    write_tables(tmp_path, 'panel', [
        {'utterance_id': 'silent', 'source_id': 'a', 'audio_path': str(audio)}])
    profile = tmp_path / 'profile.yaml'
    profile.write_text('backend: identity\n')
    with pytest.raises(RuntimeError, match='Invalid processor output: silent'):
        process(tmp_path / 'panel.jsonl', profile, tmp_path / 'out', device='cpu')
    assert not (tmp_path / 'out/processing.jsonl').exists()
