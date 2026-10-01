from types import SimpleNamespace

import pytest
import torch

from training.quality.devices import require_gpu_models, validate_sigmos_gpu_profile


def test_cpu_model_fallback_is_rejected_inside_a_cascade():
    wrapper = SimpleNamespace(models=[SimpleNamespace(model=torch.nn.Linear(2, 2))])
    with pytest.raises(RuntimeError, match='not entirely on GPU'):
        require_gpu_models((SimpleNamespace(), wrapper))


def test_backend_with_no_model_is_not_claimed_to_use_gpu():
    with pytest.raises(RuntimeError, match='No GPU model tensors'):
        require_gpu_models(SimpleNamespace())


def test_frozen_script_cpu_weights_are_detected():
    model = torch.jit.freeze(torch.jit.trace(torch.nn.Linear(2, 2).eval(), torch.zeros(1, 2)))
    assert not list(model.parameters())
    with pytest.raises(RuntimeError, match='not entirely on GPU'):
        require_gpu_models(SimpleNamespace(feature=model))


def test_sigmos_allows_cpu_shapes_but_rejects_cpu_model_operators():
    gpu = {'cat': 'Node', 'args': {'provider': 'CUDAExecutionProvider', 'op_name': 'Conv'}}
    shape = {'cat': 'Node', 'args': {'provider': 'CPUExecutionProvider', 'op_name': 'Gather',
             'input_type_shape': [{'int64': [4]}], 'output_type_shape': [{'int64': [1]}]}}
    assert validate_sigmos_gpu_profile([gpu, shape])['CUDAExecutionProvider:Conv'] == 1
    with pytest.raises(RuntimeError, match='no GPU neural'):
        validate_sigmos_gpu_profile([shape])
    bad = {'cat': 'Node', 'args': {**shape['args'], 'input_type_shape': [{'float': [4]}]}}
    with pytest.raises(RuntimeError, match='numerical model computation on CPU'):
        validate_sigmos_gpu_profile([gpu, bad])
