"""Official model adapters. Imports and model loading happen only for real scoring."""
from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly
from math import gcd

from .common import file_hash


def resample(audio, rate, target):
    if rate == target:
        return np.asarray(audio, dtype=np.float32)
    factor = gcd(rate, target)
    return resample_poly(audio, target // factor, rate // factor).astype(np.float32)


class Models:
    def __init__(self, config):
        import torch
        import whisper
        import onnxruntime as ort
        from audiobox_aesthetics.infer import initialize_predictor
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import FetchConfig

        config = dict(config)
        repository = Path(__file__).resolve().parents[2]
        for key in ('sigmos_dir', 'audiobox_checkpoint', 'whisper_cache', 'ecapa_cache'):
            if not Path(config[key]).is_absolute():
                config[key] = str(repository / config[key])
        self.torch = torch
        self.whisper_module = whisper
        self.device = config['device']
        if self.device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; use a SLURM GPU allocation')
        root = Path(config['sigmos_dir'])
        spec = importlib.util.spec_from_file_location('uktts_sigmos', root / 'sigmos.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.sigmos = module.SigMOS(str(root))
        sigmos_placement = None
        if self.device.startswith('cuda'):
            if 'CUDAExecutionProvider' not in ort.get_available_providers():
                raise RuntimeError('SigMOS requires the ONNX Runtime CUDA execution provider')
            ort.preload_dlls()
            # Some CUDA wheels omit the build-info marker used for auto-registration.
            if not any(d.ep_name == 'CUDAExecutionProvider' for d in ort.get_ep_devices()):
                name = 'onnxruntime_providers_cuda.dll' if sys.platform == 'win32' else 'libonnxruntime_providers_cuda.so'
                ort.register_execution_provider_library(
                    'CUDAExecutionProvider', str(Path(ort.__file__).parent / 'capi' / name))
            options = ort.SessionOptions()
            options.inter_op_num_threads = options.intra_op_num_threads = 1
            with tempfile.TemporaryDirectory(prefix='uktts-sigmos-placement-') as temporary:
                options.enable_profiling = True
                options.profile_file_prefix = str(Path(temporary) / 'profile')
                self.sigmos.session = ort.InferenceSession(
                    str(root / 'model-sigmos_1697718653_41d092e8-epo-200.onnx'), options,
                    providers=[('CUDAExecutionProvider', {
                        'device_id': torch.device(self.device).index or 0, 'use_tf32': '0'})])
                self.sigmos.session.disable_fallback()
                if self.sigmos.session.get_providers()[0] != 'CUDAExecutionProvider':
                    raise RuntimeError('SigMOS silently fell back from CUDA')
                self.sigmos.run(np.zeros(48000 * 3, dtype=np.float32), sr=48000)
                from .devices import validate_sigmos_gpu_profile
                sigmos_placement = validate_sigmos_gpu_profile(
                    json.loads(Path(self.sigmos.session.end_profiling()).read_text()))
        checkpoint = Path(config['audiobox_checkpoint']).resolve(strict=True)
        self.audiobox = initialize_predictor(ckpt=str(checkpoint))
        self.audiobox.device = torch.device(self.device)
        self.audiobox.model.to(self.device)
        self.whisper = whisper.load_model(config['whisper'], device=self.device,
                                          download_root=config['whisper_cache'])
        self.ecapa = EncoderClassifier.from_hparams(
            source=config['ecapa'], fetch_config=FetchConfig(revision=config['ecapa_revision']),
            savedir=config['ecapa_cache'], run_opts={'device': self.device})
        model_devices = None
        if self.device.startswith('cuda'):
            from .devices import require_gpu_models
            model_devices = require_gpu_models((self.whisper, self.audiobox.model, self.ecapa))
        self.identity = {
            'model_devices': model_devices,
            'sigmos_code_sha256': file_hash(root / 'sigmos.py'),
            'sigmos_model_sha256': file_hash(root / 'model-sigmos_1697718653_41d092e8-epo-200.onnx'),
            'sigmos_providers': self.sigmos.session.get_providers(),
            'sigmos_gpu_placement_probe': sigmos_placement,
            'audiobox_sha256': file_hash(checkpoint),
            'whisper_model': config['whisper'],
            'whisper_sha256': file_hash(Path(config['whisper_cache']) / f"{config['whisper']}.pt"),
            'whisper_language': 'uk', 'whisper_temperature': 0,
            'whisper_decoder': 'greedy_no_timestamps_batched_under_30s_v1',
            'ecapa_source': config['ecapa'], 'ecapa_revision': config['ecapa_revision'],
            'ecapa_files': {p.name: file_hash(p) for p in sorted(Path(config['ecapa_cache']).glob('*'))
                            if p.is_file()},
            'packages': {name: importlib.metadata.version(name) for name in
                         ('torch', 'torchaudio', 'onnxruntime-gpu', 'audiobox-aesthetics',
                          'openai-whisper', 'speechbrain')},
        }
        if config.get('parakeet'):
            from .parakeet import Parakeet
            self.parakeet = Parakeet(config['parakeet'], self.device)
            self.identity['parakeet'] = self.parakeet.identity

    def quality(self, audio, rate):
        return self.quality_many([audio], rate)[0]

    def quality_many(self, audios, rate):
        # Official SigMOS frontend uses FFT resampling. Do not replace it with a proxy.
        scores = [self.sigmos.run(np.asarray(audio, dtype=np.float32), sr=rate) for audio in audios]
        mapping = {'MOS_OVRL': 'overall', 'MOS_SIG': 'speech', 'MOS_NOISE': 'noise',
                   'MOS_COL': 'coloration', 'MOS_DISC': 'discontinuity',
                   'MOS_LOUD': 'loudness', 'MOS_REVERB': 'reverb'}
        with self.torch.inference_mode():
            pq = self.audiobox.forward([{'path': self.torch.from_numpy(audio.copy()).unsqueeze(0),
                                        'sample_rate': rate} for audio in audios])
        return [{**{f'sigmos_{name}': float(mos[key]) for key, name in mapping.items()},
                 'audiobox_pq': float(aes['PQ'])} for mos, aes in zip(scores, pq)]

    def transcribe(self, audio, rate):
        if len(audio) / rate <= 30:
            return self.transcribe_batch([(audio, rate)])[0]
        result = self.whisper.transcribe(resample(audio, rate, 16000), language='uk',
                                         task='transcribe', temperature=0,
                                         condition_on_previous_text=False,
                                         fp16=self.device.startswith('cuda'), verbose=None)
        return result['text']

    def transcribe_batch(self, clips):
        if any(len(audio) / rate > 30 for audio, rate in clips):
            return [self.transcribe(audio, rate) for audio, rate in clips]
        whisper = self.whisper_module
        mel = self.torch.stack([whisper.log_mel_spectrogram(
            whisper.pad_or_trim(resample(audio, rate, 16000)), n_mels=self.whisper.dims.n_mels)
            for audio, rate in clips]).to(self.device)
        options = whisper.DecodingOptions(language='uk', task='transcribe', temperature=0,
                                           without_timestamps=True,
                                           fp16=self.device.startswith('cuda'))
        with self.torch.inference_mode():
            results = whisper.decode(self.whisper, mel, options)
        return [r.text for r in results]

    def embedding(self, audio, rate):
        tensor = self.torch.from_numpy(resample(audio, rate, 16000)).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            return self.ecapa.encode_batch(tensor).squeeze().cpu().numpy()
