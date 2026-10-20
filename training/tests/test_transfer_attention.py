"""Run CUDA tests explicitly inside an allocation: UKTTS_TEST_FLASH=1 pytest ..."""
import copy
import os

import pytest
import torch
import yaml

from training.transfer.train import validate_resume


def test_resume_requires_complete_state_and_matching_data(tmp_path):
    config = dict(train_shape_file=['voice/text'], max_epoch=20, resume=False, init_param=['donor'])
    (tmp_path / 'finetune.yaml').write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match='complete checkpoint'):
        validate_resume(tmp_path, config.copy())
    torch.save(dict(model={}, reporter={'epoch': 1}, optimizers=[{}], schedulers=[{}]),
               tmp_path / 'checkpoint.pth')
    resumed = dict(config, transfer_attention_backend='flash2')
    assert validate_resume(tmp_path, resumed) == 1
    assert resumed['resume'] is True and resumed['init_param'] == []
    assert resumed['max_epoch'] == 20
    with pytest.raises(ValueError, match='training/data'):
        validate_resume(tmp_path, dict(config, train_shape_file=['mixed/text']))
    with pytest.raises(ValueError, match='target'):
        validate_resume(tmp_path, dict(config, max_epoch=1))


def test_eta_ignores_restart_gap_and_old_backend(tmp_path):
    from training.transfer.monitor import snapshot
    log = tmp_path / 'train.log'
    log.write_text('2026-10-20 10:00:00,000 INFO: 1epoch:train:991-1000batch\n'
                   '2026-10-20 11:00:00,000 INFO: The training was resumed using checkpoint.pth\n'
                   '2026-10-20 11:00:10,000 INFO: 2epoch:train:1-10batch\n'
                   '2026-10-20 11:00:20,000 INFO: 2epoch:train:11-20batch\n')
    result = snapshot(log, 20000)
    assert result['completed_steps'] == 1020
    assert result['seconds_per_step'] == 1


cuda_flash = pytest.mark.skipif(os.environ.get('UKTTS_TEST_FLASH') != '1',
                                reason='requires explicit CUDA FlashAttention allocation')


@cuda_flash
@pytest.mark.parametrize('padded', [False, True])
def test_flash_matches_outputs_and_gradients_including_padded_queries(padded):
    from espnet2.legacy.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
    from training.transfer.attention import enable_flash_attention
    torch.manual_seed(777)
    baseline = MultiHeadedAttention(2, 256, .2).cuda().eval()
    flash = copy.deepcopy(baseline)
    original_parameters = {name: id(param) for name, param in flash.named_parameters()}
    enable_flash_attention(flash)
    assert original_parameters == {name: id(param) for name, param in flash.named_parameters()}
    assert set(baseline.state_dict()) == set(flash.state_dict())
    assert all(torch.equal(value, flash.state_dict()[key])
               for key, value in baseline.state_dict().items())
    inputs = torch.randn(2, 37, 256, device='cuda', requires_grad=True)
    other = inputs.detach().clone().requires_grad_()
    mask = None
    if padded:
        mask = torch.arange(37, device='cuda')[None, None, :] < torch.tensor(
            [37, 11], device='cuda')[:, None, None]
    expected = baseline(inputs, inputs, inputs, mask)
    actual = flash(other, other, other, mask)
    # Includes the second item's 26 padded query positions, not only valid tokens.
    torch.testing.assert_close(actual, expected, atol=.004, rtol=.03)
    weight = torch.randn_like(expected)
    (expected * weight).sum().backward()
    (actual * weight).sum().backward()
    torch.testing.assert_close(other.grad, inputs.grad, atol=.008, rtol=.05)
    for (name, param), (_, ref) in zip(flash.named_parameters(), baseline.named_parameters()):
        torch.testing.assert_close(param.grad, ref.grad, atol=.03, rtol=.08, msg=name)
    # Train-time attention dropout remains active; both kernels must run.
    flash.train()
    flash.zero_grad(set_to_none=True)
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    # Prime CUPTI before measuring; its first context can contain no CUDA
    # events while the tracing infrastructure initializes.
    with torch.profiler.profile(activities=activities):
        torch.ones(1, device='cuda').square()
        torch.cuda.synchronize()
    with torch.profiler.profile(activities=activities) as prof:
        result = flash(other, other, other, mask)
        result.square().mean().backward()
        torch.cuda.synchronize()
    names = [event.name for event in prof.events() if event.device_type == torch.autograd.DeviceType.CUDA]
    assert any('flash_fwd' in name for name in names), names
    assert any('flash_bwd' in name for name in names), names
    assert not torch.equal(result, actual)
    assert torch.isfinite(result).all()
    assert all(torch.isfinite(p.grad).all() for p in flash.parameters())
    assert flash.flash_calls == 2
