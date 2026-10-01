"""Verify the actual tensor devices of every enhancement model."""
from __future__ import annotations


def validate_sigmos_gpu_profile(events):
    """Permit CPU integer shape bookkeeping, never CPU neural computation."""
    from collections import Counter

    counts = Counter()
    shape_ops = {'Shape', 'Gather', 'Concat', 'Slice', 'Unsqueeze', 'Equal', 'Cast', 'Reshape', 'Squeeze'}
    for event in events:
        args = event.get('args', {})
        provider, operation = args.get('provider'), args.get('op_name')
        if event.get('cat') != 'Node' or not provider:
            continue
        counts[f'{provider}:{operation}'] += 1
        if provider == 'CPUExecutionProvider':
            types = {kind for field in ('input_type_shape', 'output_type_shape')
                     for tensor in args.get(field, []) for kind in tensor}
            if operation not in shape_ops or not types or types - {'int64', 'int32', 'bool'}:
                raise RuntimeError(f'SigMOS placed numerical model computation on CPU: {operation}, {types}')
    if not any(counts[f'CUDAExecutionProvider:{name}'] for name in ('Conv', 'FusedConv', 'Gemm', 'MatMul', 'GRU')):
        raise RuntimeError('SigMOS profile contains no GPU neural computation')
    return dict(sorted(counts.items()))


def require_gpu_models(backend):
    import torch

    found = {}
    visited = set()

    def inspect(value, name):
        if id(value) in visited:
            return
        visited.add(id(value))
        if isinstance(value, torch.nn.Module):
            tensors = list(value.parameters()) + list(value.buffers())
            if isinstance(value, torch.jit.ScriptModule):
                # Frozen TorchScript embeds weights as graph constants, removing
                # them from parameters()/buffers(). Integer shape constants are
                # control metadata; floating constants include the model weights.
                def constants(block):
                    for node in block.nodes():
                        if node.kind() == 'prim::Constant':
                            for result in node.outputs():
                                tensor = result.toIValue()
                                if isinstance(tensor, torch.Tensor) and tensor.is_floating_point():
                                    tensors.append(tensor)
                        for child in node.blocks():
                            constants(child)
                constants(value.inlined_graph)
            devices = {str(t.device) for t in tensors}
            if devices:
                if any(not device.startswith('cuda:') for device in devices):
                    raise RuntimeError(f'Enhancement model {name} is not entirely on GPU: {sorted(devices)}')
                found[name] = sorted(devices)
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                inspect(item, f'{name}.{index}')
        else:
            for attribute in ('model', 'models', 'feature', 'decoder'):
                child = getattr(value, attribute, None)
                if child is not None:
                    inspect(child, f'{name}.{attribute}')

    inspect(backend, 'backend')
    if not found:
        raise RuntimeError('No GPU model tensors found in the requested enhancement backend')
    return found
