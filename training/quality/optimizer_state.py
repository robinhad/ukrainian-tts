"""Restore PyTorch's Adam step-counter placement after checkpoint loading."""
from __future__ import annotations


def restore_adam_step_devices(optimizers):
    import torch

    moved = 0
    for optimizer in optimizers:
        if not isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
            continue
        for group in optimizer.param_groups:
            # Capturable and fused Adam require device-resident step counters.
            if group.get('capturable') or group.get('fused'):
                continue
            for parameter in group['params']:
                state = optimizer.state.get(parameter, {})
                step = state.get('step')
                if isinstance(step, torch.Tensor) and step.device.type != 'cpu':
                    state['step'] = step.cpu()
                    moved += 1
    return moved
