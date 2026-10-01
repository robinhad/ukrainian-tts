import os

import pytest
import torch

from training.quality.optimizer_state import restore_adam_step_devices


def test_cpu_adam_state_and_update_are_preserved():
    parameter = torch.nn.Parameter(torch.ones(3))
    optimizer = torch.optim.AdamW([parameter])
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    state = optimizer.state[parameter]
    original = dict(state)
    assert restore_adam_step_devices([optimizer]) == 0
    assert all(state[key] is value for key, value in original.items())
    optimizer.step()
    assert int(state['step']) == 2


@pytest.mark.parametrize('option', ['capturable', 'fused'])
def test_device_required_counters_are_not_copied(option):
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], **{option: True})
    # Copying a meta tensor to CPU would fail; these modes must bypass that path.
    counter = torch.empty((), device='meta')
    optimizer.state[parameter]['step'] = counter
    assert restore_adam_step_devices([optimizer]) == 0
    assert optimizer.state[parameter]['step'] is counter


def test_other_optimizer_types_are_not_modified():
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.SGD([parameter])
    counter = torch.empty((), device='meta')
    optimizer.state[parameter]['step'] = counter
    assert restore_adam_step_devices([optimizer]) == 0
    assert optimizer.state[parameter]['step'] is counter


@pytest.mark.skipif(
    os.getenv('UKTTS_TEST_GPU') != '1' or not os.getenv('SLURM_JOB_ID'),
    reason='Explicit GPU integration test inside a SLURM allocation',
)
def test_gpu_resume_preserves_updates_and_required_counter_devices(tmp_path):
    from espnet2.train.reporter import Reporter
    from espnet2.train.trainer import Trainer

    def make_model():
        model = torch.nn.ParameterList([
            torch.nn.Parameter(torch.ones(64, device='cuda')) for _ in range(541)
        ])
        optimizers = [torch.optim.AdamW(list(model)[:427], lr=0.0002),
                      torch.optim.AdamW(list(model)[427:], lr=0.0002)]
        return model, optimizers

    def update(model, optimizers):
        for parameter in model:
            parameter.grad = torch.full_like(parameter, 0.01)
        for optimizer in optimizers:
            optimizer.step()

    original, original_optimizers = make_model()
    update(original, original_optimizers)
    checkpoint = tmp_path / 'checkpoint.pth'
    torch.save({'model': original.state_dict(), 'reporter': Reporter().state_dict(),
                'optimizers': [o.state_dict() for o in original_optimizers],
                'schedulers': [None, None], 'scaler': None}, checkpoint)
    resumed, resumed_optimizers = make_model()
    Trainer.resume(checkpoint, resumed, Reporter(), resumed_optimizers,
                   [None, None], None, ngpu=1)
    for optimizer in resumed_optimizers:
        for state in optimizer.state.values():
            assert state['step'].device.type == 'cpu'
            assert state['exp_avg'].device.type == 'cuda'
            assert state['exp_avg_sq'].device.type == 'cuda'
    for _ in range(5):
        update(original, original_optimizers)
        update(resumed, resumed_optimizers)
    assert all(torch.equal(a, b) for a, b in zip(original, resumed))
    for a, b in zip(original_optimizers, resumed_optimizers):
        for sa, sb in zip(a.state.values(), b.state.values()):
            assert all(torch.equal(sa[key], sb[key]) for key in sa)

    for option in ('capturable', 'fused'):
        parameter = torch.nn.Parameter(torch.ones(64, device='cuda'))
        optimizer = torch.optim.AdamW([parameter], **{option: True})
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        assert optimizer.state[parameter]['step'].device.type == 'cuda'
        assert restore_adam_step_devices([optimizer]) == 0
        assert optimizer.state[parameter]['step'].device.type == 'cuda'
