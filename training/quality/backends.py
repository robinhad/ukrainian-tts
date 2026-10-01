"""Official model adapters. Imports and model loading happen only for real scoring."""
from __future__ import annotations

import importlib.metadata
import importlib.util
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
        checkpoint = Path(config['audiobox_checkpoint']).resolve(strict=True)
        self.audiobox = initialize_predictor(ckpt=str(checkpoint))
        self.audiobox.device = torch.device(self.device)
        self.audiobox.model.to(self.device)
        self.whisper = whisper.load_model(config['whisper'], device=self.device,
                                          download_root=config['whisper_cache'])
        self.ecapa = EncoderClassifier.from_hparams(
            source=config['ecapa'], fetch_config=FetchConfig(revision=config['ecapa_revision']),
            savedir=config['ecapa_cache'], run_opts={'device': self.device})
        self.identity = {
            'sigmos_code_sha256': file_hash(root / 'sigmos.py'),
            'sigmos_model_sha256': file_hash(root / 'model-sigmos_1697718653_41d092e8-epo-200.onnx'),
            'audiobox_sha256': file_hash(checkpoint),
            'whisper_model': config['whisper'],
            'whisper_sha256': file_hash(Path(config['whisper_cache']) / f"{config['whisper']}.pt"),
            'whisper_language': 'uk', 'whisper_temperature': 0,
            'whisper_decoder': 'greedy_no_timestamps_batched_under_30s_v1',
            'ecapa_source': config['ecapa'], 'ecapa_revision': config['ecapa_revision'],
            'ecapa_files': {p.name: file_hash(p) for p in sorted(Path(config['ecapa_cache']).glob('*'))
                            if p.is_file()},
            'packages': {name: importlib.metadata.version(name) for name in
                         ('torch', 'torchaudio', 'onnxruntime', 'audiobox-aesthetics',
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
