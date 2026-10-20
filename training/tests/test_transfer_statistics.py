import copy
import json

import numpy as np
import pytest
import torch
import yaml

from training.transfer.convert import sha256
from training.transfer.statistics import (
    FEATURES, accumulate, buffers, enable_phase_history, migrate, replace_buffers, validate_buffers,
)


def test_ranking_ignores_archived_phase_and_preserves_step():
    from espnet2.train.reporter import Reporter
    enable_phase_history()
    reporter = Reporter()
    reporter.load_state_dict({'epoch': 6, 'stats': {
        6: {'train': {'total_count': 6000}, 'valid_previous_normalization': {'loss': 1.}},
    }})
    with reporter.observe('train', 7) as sub:
        assert sub.total_count == 6000
    reporter.stats[7]['valid'] = {'loss': 2.}
    assert reporter.sort_epochs('valid', 'loss', 'min') == [7]
    assert reporter.sort_epochs('valid_previous_normalization', 'loss', 'min') == [6]


def test_accumulation_excludes_padding_and_weights_frames():
    features = {}
    for key in FEATURES:
        features[key] = torch.tensor([[[1.], [3.], [999.]], [[5.], [999.], [999.]]])
        features[key + '_lengths'] = torch.tensor([2, 1])
    totals = {}
    accumulate(totals, features)
    for stats in totals.values():
        assert stats['count'] == 3
        np.testing.assert_array_equal(stats['sum'], [9.])
        np.testing.assert_array_equal(stats['sum_square'], [35.])


def make_config(directory, offset):
    directory.mkdir()
    (directory / 'stats').mkdir()
    config = dict(token_list=['a'], feats_extract_conf={'fs': 22050},
                  pitch_extract_conf={}, energy_extract_conf={})
    for key, name in FEATURES.items():
        path = directory / 'stats' / f'{key}_stats.npz'
        values = np.array([[1. + offset], [3. + offset]])
        np.savez(path, count=2, sum=values.sum(0), sum_square=(values**2).sum(0))
        config[name + '_conf'] = {'stats_file': str(path)}
    return config


def test_buffer_validation_and_parameter_preservation(tmp_path):
    config = make_config(tmp_path / 'bundle', 0)
    state = {**buffers(config), 'weight': torch.randn(2, 3)}
    original = copy.deepcopy(state)
    state['normalize.mean'] += .01
    with pytest.raises(ValueError, match='normalization differs'):
        validate_buffers(state, config)
    assert len(replace_buffers(state, config)) == 6
    for key in state:
        assert torch.equal(state[key], original[key])


def test_migration_preserves_learned_state_and_excludes_old_ranking(tmp_path):
    source, bundle, output = (tmp_path / name for name in ('source', 'bundle', 'output'))
    old = make_config(source, 0)
    new = make_config(bundle, 2)
    text, wav = source / 'text', source / 'wav.scp'
    text.write_text('u a\n')
    wav.write_text('u sample.wav\n')
    old.update(train_data_path_and_name_and_type=[
        [str(text), 'text', 'text'], [str(wav), 'speech', 'sound']],
        num_iters_per_epoch=1000, init_param=[], output_dir=str(source))
    for name in ('finetune.yaml', 'config.yaml'):
        (source / name).write_text(yaml.safe_dump(old))
    (bundle / 'config.yaml').write_text(yaml.safe_dump(new))
    (bundle / 'normalization.json').write_text(json.dumps(dict(
        text_sha256=sha256(text), wav_scp_sha256=sha256(wav),
        features={key: {'sha256': sha256(bundle / 'stats' / f'{key}_stats.npz')}
                  for key in FEATURES})))
    state = dict(model={**buffers(old), 'learned.weight': torch.randn(2, 3)},
                 optimizers=[{'state': {0: {'exp_avg': torch.randn(2, 3), 'step': 6000}}}],
                 schedulers=[{'last_epoch': 6}], scaler=None,
                 reporter={'epoch': 6, 'stats': {6: {
                     'train': {'total_count': 6000}, 'valid': {'loss': 2.}}}})
    torch.save(state, source / 'checkpoint.pth')
    migrate(source, bundle, output)
    migrated = torch.load(output / 'checkpoint.pth', weights_only=False)
    validate_buffers(migrated['model'], new)
    assert torch.equal(state['model']['learned.weight'], migrated['model']['learned.weight'])
    assert torch.equal(state['optimizers'][0]['state'][0]['exp_avg'],
                       migrated['optimizers'][0]['state'][0]['exp_avg'])
    assert migrated['optimizers'][0]['state'][0]['step'] == 6000
    assert migrated['schedulers'] == state['schedulers']
    assert migrated['reporter']['epoch'] == 6
    assert migrated['reporter']['stats'][6] == {
        'train': {'total_count': 6000}, 'valid_previous_normalization': {'loss': 2.}}
    assert (output / 'latest.pth').resolve() == output / '6epoch.pth'
    unchanged = torch.load(source / 'checkpoint.pth', weights_only=False)
    validate_buffers(unchanged['model'], old)
