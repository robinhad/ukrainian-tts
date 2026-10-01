"""Apply an explicit processing combination with the existing model adapters."""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from .common import digest, file_hash, read_rows, write_json, write_tables


def process(panel, profile_path, output, device='cuda', resume=False):
    from training.audio_enhancement.fair_comparison import (
        atomic_write_pcm24, fit_sample_count, match_loudness_and_prevent_clipping, measure_lufs)
    from training.scripts.run_fair_enhancement_comparison import make_backend, backend_channel
    from training.scripts.run_enhancement_review_backend import SidonBackend

    profile = yaml.safe_load(Path(profile_path).read_text())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = read_rows(panel)
    if not rows or any('voa' in str(r['source_id']).lower() for r in rows):
        raise ValueError('Processing requires a non-empty non-VOA panel')
    wet = float(profile.get('wet', 1.0))
    if not 0 <= wet <= 1:
        raise ValueError('wet must be in [0, 1]')
    backend_name = profile['backend']
    identity, backend = {'name': backend_name}, None
    if backend_name == 'sidon':
        backend = SidonBackend(device, Path(profile['model_cache']) / 'sidon')
        identity = backend.identity
    elif backend_name != 'identity':
        backend, identity = make_backend(argparse.Namespace(
            backend=backend_name, device=device, model_cache=Path(profile['model_cache']),
            rnnoise_binary=Path(profile.get('rnnoise_binary', 'training/vendor/rnnoise/examples/rnnoise_demo'))))
    results = []
    for row in rows:
        if shutil.disk_usage(output).free < 30 * 1024**3:
            raise RuntimeError('Disk reserve below 30 GiB')
        source = Path(row['audio_path'])
        input_hash = file_hash(source)
        if row.get('reference_sha256', input_hash) != input_hash:
            raise ValueError('Frozen input changed')
        target = output / f"{row['utterance_id']}.wav"
        metadata = output / f"{row['utterance_id']}.json"
        key = digest([input_hash, profile, identity])
        if resume and target.is_file() and metadata.is_file():
            old = json.loads(metadata.read_text())
            if old['key'] == key and old['output_sha256'] == file_hash(target):
                results.append(old)
                continue
        started = time.monotonic()
        audio, rate = sf.read(source, dtype='float32', always_2d=True)
        if rate != 24000 or audio.shape[1] != 1:
            raise ValueError('Processing inputs must be canonical mono 24 kHz audio')
        if backend_name == 'identity':
            enhanced = audio.copy()
        elif backend_name == 'sidon':
            result, result_rate = backend.process(audio[:, 0], rate)['sidon_no_compression']
            from .backends import resample
            enhanced = resample(np.asarray(result).reshape(-1), result_rate, rate)[:, None]
        else:
            enhanced = backend_channel(backend_name, backend, audio[:, 0], rate)[:, None]
        raw_output_frames = len(enhanced)
        enhanced = fit_sample_count(enhanced, len(audio))
        if not np.isfinite(enhanced).all() or not np.any(enhanced):
            raise RuntimeError(f"Invalid processor output: {row['utterance_id']}")
        mixed = wet * enhanced + (1 - wet) * audio
        # Preserve the no-processing control exactly, apart from PCM24 encoding.
        loudness = None
        if backend_name != 'identity':
            mixed, loudness = match_loudness_and_prevent_clipping(mixed, rate, measure_lufs(audio, rate))
        atomic_write_pcm24(target, mixed, rate)
        item = {'utterance_id': row['utterance_id'], 'source_id': row['source_id'],
                'key': key, 'profile': profile, 'backend_identity': identity,
                'input_sha256': input_hash, 'output_sha256': file_hash(target),
                'output_path': str(target.resolve()), 'input_frames': len(audio),
                'processor_output_frames': raw_output_frames, 'output_frames': len(mixed),
                'processor_duration_changed': raw_output_frames != len(audio),
                'loudness_match': loudness, 'elapsed_seconds': time.monotonic() - started}
        write_json(metadata, item)
        results.append(item)
        write_json(output / 'progress.json', {'completed': len(results), 'expected': len(rows)})
    write_tables(output, 'processing', results)
    return results
