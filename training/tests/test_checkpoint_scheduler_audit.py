import copy

import torch

from training.scripts.audit_finetune_checkpoint import audit_schedulers


def saved_state():
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=0.0002)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999875)
    for _ in range(50):
        optimizer.step()
        scheduler.step()
    return {"optimizers": [optimizer.state_dict()],
            "schedulers": [scheduler.state_dict()]}


def test_accepts_continued_schedule():
    state = saved_state()
    assert audit_schedulers(state, 50, 0.999875)[0]
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=0.0002)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999875)
    optimizer.load_state_dict(state["optimizers"][0])
    scheduler.load_state_dict(state["schedulers"][0])
    optimizer.step()
    scheduler.step()
    assert audit_schedulers({"optimizers": [optimizer.state_dict()],
                             "schedulers": [scheduler.state_dict()]}, 51, 0.999875)[0]


def test_rejects_reset_or_missing_schedule():
    original = saved_state()
    for key, value in [("last_epoch", 0), ("gamma", 0.9), ("_last_lr", [0.0002])]:
        state = copy.deepcopy(original)
        state["schedulers"][0][key] = value
        assert not audit_schedulers(state, 50, 0.999875)[0]
    original["schedulers"] = []
    assert not audit_schedulers(original, 50, 0.999875)[0]
