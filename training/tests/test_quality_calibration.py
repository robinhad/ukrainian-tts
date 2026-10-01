from datetime import timedelta

import torch

from training.scripts.calibrate_quality_v10 import checkpoint_measurements


def test_calibration_rejects_skipped_updates_nonfinite_weights_and_missing_optimizer(tmp_path):
    checkpoint = tmp_path / 'checkpoint.pth'
    payload = {'reporter': {'epoch': 1, 'stats': {1: {'train': {
        'time': timedelta(seconds=2), 'total_count': 100}}}},
        'model': {'weight': torch.ones(1)},
        'optimizers': [{'state': {0: {'step': torch.tensor(100)}}},
                       {'state': {0: {'step': torch.tensor(100)}}}]}
    torch.save(payload, checkpoint)
    result = checkpoint_measurements(checkpoint, 100)
    assert result['checkpoint_valid'] and result['training_seconds'] == 2
    payload['optimizers'][1]['state'][0]['step'] = torch.tensor(99)
    torch.save(payload, checkpoint)
    result = checkpoint_measurements(checkpoint, 100)
    assert not result['checkpoint_valid'] and result['optimizer_steps'] == [99, 100]
    payload['optimizers'][1]['state'][0]['step'] = torch.tensor(100)
    payload['model']['weight'][0] = float('nan')
    torch.save(payload, checkpoint)
    result = checkpoint_measurements(checkpoint, 100)
    assert not result['checkpoint_valid'] and result['nonfinite_model_tensors'] == ['weight']
    payload['model']['weight'][0] = 1
    payload['optimizers'][1]['state'].clear()
    torch.save(payload, checkpoint)
    assert not checkpoint_measurements(checkpoint, 100)['checkpoint_valid']
    payload['optimizers'].pop()
    torch.save(payload, checkpoint)
    assert not checkpoint_measurements(checkpoint, 100)['checkpoint_valid']
