"""Pinned Parakeet ASR in the isolated NeMo environment, with resumable file caches."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
from pathlib import Path

from .common import digest, file_hash, write_json


class Parakeet:
    def __init__(self, config, device):
        from huggingface_hub import snapshot_download

        root = Path(__file__).resolve().parents[2]
        self.python = root / config['python']
        if not self.python.is_file():
            raise FileNotFoundError('Parakeet environment missing; run bootstrap_nemo_env.sh')
        self.device = device
        self.batch_size = config.get('batch_size', 8)
        if self.batch_size < 1:
            raise ValueError('Parakeet batch_size must be positive')
        folder = snapshot_download(config['repo'], revision=config['revision'],
                                   allow_patterns=['*.nemo'], cache_dir=str(root / config['cache']))
        checkpoints = list(Path(folder).glob('*.nemo'))
        if len(checkpoints) != 1:
            raise ValueError('Expected exactly one pinned Parakeet checkpoint')
        self.checkpoint = checkpoints[0]
        packages = json.loads(subprocess.check_output([
            str(self.python), '-m', 'training.quality.parakeet', '--identity'], cwd=root, text=True))
        self.identity = {'repo': config['repo'], 'revision': config['revision'],
                         'checkpoint_sha256': file_hash(self.checkpoint), 'packages': packages,
                         'worker_sha256': file_hash(Path(__file__)), 'sample_rate': 16000,
                         'device': device,
                         'language': 'automatic', 'decoding': 'checkpoint_default',
                         'batch_size': self.batch_size}

    def transcribe_files(self, records, wav_dir, output):
        output = Path(output) / 'parakeet'
        output.mkdir(parents=True, exist_ok=True)
        jobs, hypotheses = [], {}
        for record in records:
            identifier = record['utterance_id']
            path = Path(wav_dir) / f'{identifier}.wav' if wav_dir else Path(record['audio_path'])
            cache = output / 'cache' / f'{digest(identifier)}.json'
            try:
                key = digest([self.identity, file_hash(path)])
                if cache.exists():
                    previous = json.loads(cache.read_text())
                    if previous.get('key') == key and 'text' in previous:
                        hypotheses[identifier] = previous['text']
                        continue
                jobs.append({'utterance_id': identifier, 'path': str(path.resolve()),
                             'cache': str(cache.resolve()), 'key': key})
            except Exception as error:
                hypotheses[identifier] = error
        if jobs:
            request = output / 'request.json'
            write_json(request, {'jobs': jobs, 'checkpoint': str(self.checkpoint),
                                 'device': self.device, 'batch_size': self.batch_size})
            # One model load for the entire panel. The worker writes each batch atomically.
            subprocess.run([str(self.python), '-m', 'training.quality.parakeet',
                            '--request', str(request.resolve())], check=True,
                           cwd=Path(__file__).resolve().parents[2])
            for job in jobs:
                result = json.loads(Path(job['cache']).read_text())
                if result.get('key') != job['key']:
                    raise ValueError('Parakeet returned stale cache provenance')
                hypotheses[job['utterance_id']] = (RuntimeError(result['error']) if 'error' in result
                                                  else result['text'])
        return hypotheses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--identity', action='store_true')
    parser.add_argument('--request', type=Path)
    args = parser.parse_args()
    if args.identity:
        packages = {name: importlib.metadata.version(name) for name in
                    ('nemo_toolkit', 'torch', 'torchaudio', 'numpy', 'soundfile', 'scipy')}
        direct = importlib.metadata.distribution('nemo_toolkit').read_text('direct_url.json')
        packages['nemo_source'] = json.loads(direct)['vcs_info'] if direct else None
        print(json.dumps(packages))
        return
    if args.request is None:
        parser.error('--request is required for transcription')
    import numpy as np
    import soundfile as sf
    import torch
    from nemo.collections.asr.models import ASRModel
    from .backends import resample

    request = json.loads(args.request.read_text())
    model = ASRModel.restore_from(request['checkpoint'], map_location=request['device'], strict=True)
    model.to(request['device'])
    model.eval()
    if request['device'].startswith('cuda'):
        from .devices import require_gpu_models
        print(f'Parakeet GPU model verified: {require_gpu_models(model)}', flush=True)
    batch_size = request['batch_size']
    for start in range(0, len(request['jobs']), batch_size):
        jobs, audio = [], []
        for job in request['jobs'][start:start + batch_size]:
            try:
                samples, rate = sf.read(job['path'], dtype='float32', always_2d=True)
                if not samples.size or not np.isfinite(samples).all():
                    raise ValueError('Invalid Parakeet audio')
                audio.append(resample(samples.mean(axis=1), rate, 16000))
                jobs.append(job)
            except Exception as error:
                write_json(Path(job['cache']), {'key': job['key'], 'error': str(error)})
        if jobs:
            with torch.inference_mode():
                results = model.transcribe(audio=audio, batch_size=batch_size,
                                           return_hypotheses=True, num_workers=0, verbose=False)
            if len(results) != len(jobs):
                raise ValueError('Parakeet returned incorrect file coverage')
            for job, result in zip(jobs, results):
                if not isinstance(result.text, str):
                    raise TypeError('Parakeet returned a non-text hypothesis')
                write_json(Path(job['cache']), {'key': job['key'], 'text': result.text})
        print(f'Parakeet: {min(start + batch_size, len(request["jobs"]))}/{len(request["jobs"])}', flush=True)


if __name__ == '__main__':
    main()
