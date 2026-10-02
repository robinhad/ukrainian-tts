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


def test_postprocessing_worker_errors_propagate(tmp_path, monkeypatch):
    audio = tmp_path / 'silent.wav'
    sf.write(audio, np.ones(2400) * .1, 24000)
    write_tables(tmp_path, 'panel', [
        {'utterance_id': 'silent', 'source_id': 'a', 'audio_path': str(audio)}])
    profile = tmp_path / 'profile.yaml'
    profile.write_text('backend: identity\n')
    def fail(*args):
        raise OSError('simulated write failure')
    monkeypatch.setattr('training.audio_enhancement.fair_comparison.atomic_write_pcm24', fail)
    with pytest.raises(OSError, match='simulated write failure'):
        process(tmp_path / 'panel.jsonl', profile, tmp_path / 'out', device='cpu')
    assert not (tmp_path / 'out/processing.jsonl').exists()


def test_suppressed_output_is_preserved_for_quality_reporting(monkeypatch):
    from training.quality.processing import match_loudness_for_reporting
    def below_gate(*args):
        raise RuntimeError('invalid integrated loudness: -inf')
    monkeypatch.setattr('training.audio_enhancement.fair_comparison.measure_lufs', below_gate)
    audio = np.ones((2400, 1), dtype=np.float32) * 1e-8
    result, report = match_loudness_for_reporting(audio, audio * 1e6, 24000)
    assert np.array_equal(result, audio)
    assert report['loudness_match_status'] == 'below_measurement_gate'
    assert report['report_only'] is True
    def broken(*args):
        raise RuntimeError('FFmpeg did not return a loudness measurement')
    monkeypatch.setattr('training.audio_enhancement.fair_comparison.measure_lufs', broken)
    with pytest.raises(RuntimeError, match='FFmpeg did not return'):
        match_loudness_for_reporting(audio, audio, 24000)


def test_normalized_model_output_returns_to_original_scale_before_blending(tmp_path, monkeypatch):
    from training.quality import processing
    import torch
    source = tmp_path / 'quiet.wav'
    sf.write(source, np.linspace(-.001, .001, 2400), 24000, subtype='PCM_24')
    original, _ = sf.read(source)
    write_tables(tmp_path, 'panel', [{'utterance_id': 'quiet', 'source_id': 'a', 'audio_path': str(source)}])
    profile = tmp_path / 'profile.yaml'
    profile.write_text('backend: deepfilternet3\nwet: 0.5\ninput_peak_dbfs: -3\nmodel_cache: unused\n')
    seen = []
    def enhance(name, backend, audio, rate):
        seen.append(float(np.max(np.abs(audio))))
        return audio * .5
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'set_device', lambda device: None)
    monkeypatch.setattr('training.quality.devices.require_gpu_models', lambda backend: {'test': ['cuda:0']})
    monkeypatch.setattr('training.scripts.run_fair_enhancement_comparison.make_backend', lambda args: (object(), {}))
    monkeypatch.setattr('training.scripts.run_fair_enhancement_comparison.backend_channel', enhance)
    monkeypatch.setattr(processing, 'match_loudness_for_reporting', lambda audio, reference, rate, target: (audio, {}))
    rows = process(tmp_path / 'panel.jsonl', profile, tmp_path / 'out')
    actual, _ = sf.read(rows[0]['output_path'])
    assert seen == pytest.approx([10 ** (-3 / 20)])
    # The model halves the wet signal; mixing half wet and half dry gives 0.75.
    assert np.max(np.abs(actual - original * .75)) <= 1.3e-7
    assert rows[0]['model_input_gain_db'] > 50


def test_optional_output_loudness_with_real_meter_preserves_duration_and_peak(tmp_path):
    from training.audio_enhancement.fair_comparison import measure_lufs
    rate = 24000
    audio = .002 * np.sin(2 * np.pi * 440 * np.arange(rate * 2) / rate)
    source = tmp_path / 'quiet.wav'
    sf.write(source, audio, rate, subtype='PCM_24')
    write_tables(tmp_path, 'panel', [{'utterance_id': 'quiet', 'source_id': 'a', 'audio_path': str(source)}])
    profile = tmp_path / 'profile.yaml'
    profile.write_text('backend: identity\noutput_lufs: -23\n')
    rows = process(tmp_path / 'panel.jsonl', profile, tmp_path / 'out', device='cpu')
    actual, sr = sf.read(rows[0]['output_path'])
    assert sr == rate and len(actual) == len(audio)
    assert abs(measure_lufs(actual, sr) + 23) < .1
    assert np.max(np.abs(actual)) < 10 ** (-.1 / 20)


@pytest.mark.parametrize('amplitude', [.2, .999])
def test_normalized_original_matches_input_loudness_and_caps_peak(tmp_path, amplitude):
    rate = 24000
    source = tmp_path / 'input.wav'
    sf.write(source, amplitude * np.sin(2 * np.pi * 440 * np.arange(rate * 2) / rate), rate, subtype='PCM_24')
    original, _ = sf.read(source, dtype='float32')
    write_tables(tmp_path, 'panel', [{'utterance_id': 'item', 'source_id': 'a', 'audio_path': str(source)}])
    profile = tmp_path / 'profile.yaml'
    profile.write_text('backend: identity\nmatch_input_loudness: true\n')
    rows = process(tmp_path / 'panel.jsonl', profile, tmp_path / 'out', device='cpu')
    actual, sr = sf.read(rows[0]['output_path'])
    expected_gain = min(1, 10 ** (-.1 / 20) / np.max(np.abs(original)))
    assert rows[0]['loudness_match'] is not None
    assert sr == rate and len(actual) == len(original)
    assert np.max(np.abs(actual - original * expected_gain)) < 2e-7


def test_per_recording_seed_is_independent_of_model_call_order(tmp_path, monkeypatch):
    import torch
    from training.quality import processing
    source = tmp_path / 'input.wav'
    sf.write(source, np.linspace(-.1, .1, 2400), 24000, subtype='PCM_24')
    rows = [{'utterance_id': name, 'source_id': 'a', 'audio_path': str(source)}
            for name in ['one', 'two', 'three']]
    write_tables(tmp_path, 'forward', rows)
    write_tables(tmp_path, 'reverse', rows[::-1])
    profile = tmp_path / 'profile.yaml'
    profile.write_text('backend: deepfilternet3\nmodel_cache: unused\nrandom_seed: 777\n')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'set_device', lambda device: None)
    monkeypatch.setattr('training.quality.devices.require_gpu_models', lambda backend: {})
    monkeypatch.setattr('training.scripts.run_fair_enhancement_comparison.make_backend', lambda args: (object(), {}))
    monkeypatch.setattr('training.scripts.run_fair_enhancement_comparison.backend_channel',
                        lambda name, backend, audio, rate: audio + torch.rand(len(audio)).numpy() * .01)
    monkeypatch.setattr(processing, 'match_loudness_for_reporting', lambda audio, *args: (audio, {}))
    forward = process(tmp_path / 'forward.jsonl', profile, tmp_path / 'one', cpu_workers=1)
    reverse = process(tmp_path / 'reverse.jsonl', profile, tmp_path / 'two', cpu_workers=2)
    assert {r['utterance_id']: r['output_sha256'] for r in forward} == {
        r['utterance_id']: r['output_sha256'] for r in reverse}
    assert len({r['output_sha256'] for r in forward}) == 3


def test_cpu_rnnoise_is_allowed_and_device_is_reported(tmp_path, monkeypatch):
    from training.quality import processing
    source = tmp_path / 'input.wav'
    sf.write(source, np.linspace(-.1, .1, 2400), 24000, subtype='PCM_24')
    write_tables(tmp_path, 'panel', [{'utterance_id': 'one', 'source_id': 'a', 'audio_path': str(source)}])
    profile = tmp_path / 'profile.yaml'
    profile.write_text('backend: rnnoise85\nmodel_cache: unused\n')
    monkeypatch.setattr('training.scripts.run_fair_enhancement_comparison.make_backend',
                        lambda args: (tmp_path / 'rnnoise', {'name': 'RNNoise85'}))
    monkeypatch.setattr('training.scripts.run_fair_enhancement_comparison.backend_channel',
                        lambda name, backend, audio, rate: audio)
    monkeypatch.setattr(processing, 'match_loudness_for_reporting',
                        lambda audio, reference, rate, target: (audio, {}))
    rows = process(tmp_path / 'panel.jsonl', profile, tmp_path / 'out', device='cpu')
    assert rows[0]['backend_identity']['model_devices'] == {'rnnoise': ['cpu']}
    assert rows[0]['input_frames'] == rows[0]['output_frames'] == 2400


def test_independent_model_workers_merge_and_propagate_failures(tmp_path):
    from training.quality.parallel_processing import process_parallel
    source = tmp_path / 'input.wav'
    sf.write(source, np.linspace(-.1, .1, 2400), 24000, subtype='PCM_24')
    rows = [{'utterance_id': f'item{i}', 'source_id': 'a', 'audio_path': str(source)} for i in range(4)]
    write_tables(tmp_path, 'panel', rows)
    profile = tmp_path / 'profile.yaml'
    profile.write_text('backend: identity\n')
    serial = process(tmp_path / 'panel.jsonl', profile, tmp_path / 'serial', device='cpu')
    parallel = process_parallel(tmp_path / 'panel.jsonl', profile, tmp_path / 'parallel', device='cpu', chunk_size=1)
    assert [r['output_sha256'] for r in parallel] == [r['output_sha256'] for r in serial]
    resumed = process_parallel(tmp_path / 'panel.jsonl', profile, tmp_path / 'parallel', device='cpu', resume=True, chunk_size=1)
    assert [r['output_sha256'] for r in resumed] == [r['output_sha256'] for r in serial]
    with pytest.raises(RuntimeError, match='memory below'):
        process_parallel(tmp_path / 'panel.jsonl', profile, tmp_path / 'low_memory',
                         device='cpu', minimum_available_gib=1e9)
    assert not (tmp_path / 'low_memory/processing.jsonl').exists()
    zero = tmp_path / 'zero.wav'
    sf.write(zero, np.zeros(2400), 24000)
    rows[0]['audio_path'] = str(zero)
    write_tables(tmp_path, 'bad_panel', rows)
    with pytest.raises(RuntimeError, match='Processing worker failed'):
        process_parallel(tmp_path / 'bad_panel.jsonl', profile, tmp_path / 'failed', device='cpu', chunk_size=1)
    assert not (tmp_path / 'failed/processing.jsonl').exists()
