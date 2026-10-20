import copy

import torch

from training.transfer.convert import EMBEDDING
from training.transfer.rhotic_ab import initialize, regroup_optimizer
from training.transfer.monitor import snapshot


def test_two_arms_differ_only_in_r_initialization():
    state = {EMBEDDING: torch.randn(4, 8), 'backbone': torch.randn(8, 8)}
    donor = {EMBEDDING: torch.randn(4, 8)}
    tokens = ['r', 'l', 'ɹ', 'a']
    original = copy.deepcopy(state)
    a = initialize(state, donor, tokens, 'kept_r')
    b = initialize(state, donor, tokens, 'english_r')
    assert all(torch.equal(a[k], original[k]) for k in state)
    assert all(torch.equal(state[k], original[k]) for k in state)
    assert torch.equal(b[EMBEDDING][0], donor[EMBEDDING][2])
    assert torch.equal(b[EMBEDDING][1:], original[EMBEDDING][1:])
    assert torch.equal(b['backbone'], original['backbone'])


def test_tenfold_embedding_update_and_resume_groups():
    generator = torch.nn.Sequential(torch.nn.Embedding(4, 3), torch.nn.Linear(3, 2))
    original = torch.optim.AdamW(generator.parameters(), lr=2e-5, weight_decay=0)
    optimizer = regroup_optimizer(original, generator, generator[0].weight, 10)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.99)
    before = [p.detach().clone() for p in generator.parameters()]
    for parameter in generator.parameters(): parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    changes = [(old-new).abs().mean().item() for old, new in zip(before, generator.parameters())]
    assert abs(changes[0] / changes[1] - 10) < .03
    scheduler.step()
    state = optimizer.state_dict()
    restored = regroup_optimizer(torch.optim.AdamW(generator.parameters(), lr=2e-5, weight_decay=0),
                                 generator, generator[0].weight, 10)
    restored.load_state_dict(state)
    assert [g['lr'] for g in restored.param_groups] == [g['lr'] for g in optimizer.param_groups]
    assert restored.param_groups[1]['lr'] / restored.param_groups[0]['lr'] == 10
    covered = [id(p) for group in restored.param_groups for p in group['params']]
    assert len(covered) == len(set(covered)) == len(list(generator.parameters()))


def test_monitor_500_step_epochs(tmp_path):
    log = tmp_path / 'train.log'
    log.write_text('2026-10-21 00:00:00,000 INFO: 3epoch:train:1-10batch: train_time=0.9\n')
    assert snapshot(log, 2000, 500)['completed_steps'] == 1010
