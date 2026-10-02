"""Official SigMOS with verified CUDA placement, without loading ASR models."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

from .common import file_hash


class SigMOSOnly:
    def __init__(self, root, device='cuda:0'):
        import torch
        import onnxruntime as ort
        from .devices import validate_sigmos_gpu_profile
        if not device.startswith('cuda') or not torch.cuda.is_available():
            raise RuntimeError('SigMOS scoring requires a SLURM GPU allocation')
        root = Path(root)
        spec = importlib.util.spec_from_file_location('uktts_selection_sigmos', root / 'sigmos.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.model = module.SigMOS(str(root))
        if 'CUDAExecutionProvider' not in ort.get_available_providers():
            raise RuntimeError('Missing ONNX CUDA provider')
        ort.preload_dlls()
        if not any(d.ep_name == 'CUDAExecutionProvider' for d in ort.get_ep_devices()):
            name = 'onnxruntime_providers_cuda.dll' if sys.platform == 'win32' else 'libonnxruntime_providers_cuda.so'
            ort.register_execution_provider_library('CUDAExecutionProvider', str(Path(ort.__file__).parent / 'capi' / name))
        model_path = root / 'model-sigmos_1697718653_41d092e8-epo-200.onnx'
        options = ort.SessionOptions()
        options.inter_op_num_threads = options.intra_op_num_threads = 1
        with tempfile.TemporaryDirectory(prefix='uktts-selection-placement-') as temporary:
            options.enable_profiling = True
            options.profile_file_prefix = str(Path(temporary) / 'profile')
            self.model.session = ort.InferenceSession(str(model_path), options, providers=[
                ('CUDAExecutionProvider', {'device_id': torch.device(device).index or 0, 'use_tf32': '0'})])
            self.model.session.disable_fallback()
            if self.model.session.get_providers()[0] != 'CUDAExecutionProvider':
                raise RuntimeError('SigMOS CUDA fallback')
            self.model.run(np.zeros(48000 * 3, dtype=np.float32), sr=48000)
            placement = validate_sigmos_gpu_profile(json.loads(Path(self.model.session.end_profiling()).read_text()))
        self.identity = {'code_sha256': file_hash(root / 'sigmos.py'),
                         'model_sha256': file_hash(model_path), 'placement': placement}

    def score(self, audio, rate):
        result = self.model.run(np.asarray(audio, dtype=np.float32), sr=rate)
        if not all(np.isfinite(v) for v in result.values()):
            raise ValueError('Nonfinite SigMOS result')
        return result
