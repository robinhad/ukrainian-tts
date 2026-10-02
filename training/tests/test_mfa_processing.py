import json

import numpy as np
import pytest
import soundfile as sf

from training.quality.common import file_hash, read_rows, write_tables
from training.scripts.prepare_mfa_processing import prepare


def fixture_panel(tmp_path, flagged=False):
    root = tmp_path / 'mfa'
    root.mkdir()
    source = root / 'source.wav'
    sf.write(source, np.arange(24000, dtype=np.float32) / 48000, 24000, subtype='PCM_24')
    grid = root / 'aligned/speaker/sample.json'
    grid.parent.mkdir(parents=True)
    grid.write_text('{}')
    row = {'utterance_id': 'test', 'source_id': 'a', 'sample_id': 'sample',
           'audio_path': str(source), 'reference_sha256': file_hash(source),
           'alignment_audio_path': str(source), 'alignment_audio_sha256': file_hash(source),
           'processing_audio_path': 'obsolete-energy-input.wav'}
    cut = {'utterance_id': 'test', 'source_id': 'a', 'input_sha256': file_hash(source),
           'reference_sha256': file_hash(source), 'alignment_sha256': file_hash(grid),
           'start_frame': 0 if flagged else 2400, 'end_frame': 24000 if flagged else 20000,
           'input_frames': 24000, 'output_frames': 24000 if flagged else 17600,
           'status': 'review_preserved_untrimmed' if flagged else 'mfa_aligned',
           'flags': ['unknown_phone'] if flagged else []}
    write_tables(root, 'panel', [row])
    write_tables(root, 'processing', [cut])
    return root, source, cut


@pytest.mark.parametrize('flagged', [False, True])
def test_mfa_inputs_exact_crop_without_gain_or_energy_fallback(tmp_path, flagged):
    root, source, cut = fixture_panel(tmp_path, flagged)
    rows = prepare(root, tmp_path / 'output')
    original, _ = sf.read(source)
    actual, rate = sf.read(rows[0]['processing_audio_path'])
    assert rate == 24000
    assert np.array_equal(actual, original[cut['start_frame']:cut['end_frame']])
    assert rows[0]['boundary_method'] == 'mfa'
    assert read_rows(tmp_path / 'output/panel.jsonl') == rows


def test_review_alignment_cannot_silently_trim(tmp_path):
    root, _, cut = fixture_panel(tmp_path, True)
    cut.update(start_frame=1, output_frames=23999)
    write_tables(root, 'processing', [cut])
    with pytest.raises(ValueError, match='preserve the complete input'):
        prepare(root, tmp_path / 'output')


def test_changed_alignment_is_rejected(tmp_path):
    root, _, _ = fixture_panel(tmp_path)
    (root / 'aligned/speaker/sample.json').write_text('{"changed": true}')
    with pytest.raises(ValueError, match='alignment changed'):
        prepare(root, tmp_path / 'output')


def test_mfa_profile_rejects_historical_energy_inputs(tmp_path):
    from training.quality.processing import process
    panel = tmp_path / 'panel.jsonl'
    panel.write_text(json.dumps({'utterance_id': 'a', 'source_id': 'b'}) + '\n')
    profile = tmp_path / 'profile.yaml'
    profile.write_text('backend: identity\nboundary_method: mfa\n')
    with pytest.raises(ValueError, match='requires audited MFA'):
        process(panel, profile, tmp_path / 'output', device='cpu')


def test_population_sd_and_selectors_do_not_use_native_reference():
    from training.scripts.report_mfa_processing import selections, summarize
    summary = summarize([1., 2., 3., 4.])
    assert summary['median'] == 2.5
    assert summary['stddev_population'] == pytest.approx(np.sqrt(1.25))
    common = {'sample_id': 'a', 'source_id': 's', 'duration_seconds': 2., 'audio_sha256': 'hash'}
    profiles = {name: {'a': {**common, 'sigmos_overall': score}} for name, score in
                [('original', 5.), ('normalized_original', 3.5), ('enhancement', 3.5), ('other', 3.6)]}
    selected = selections(profiles, ['normalized_original', 'enhancement', 'other'], 'enhancement', 3.5)
    assert selected[0]['selected_profile'] == 'normalized_original'
    assert selected[0]['retained'] is True  # Inclusive threshold and normalized-first exact tie.
    assert selected[1]['selected_profile'] == 'other'
    assert selected[1]['sigmos_overall'] == 3.6
